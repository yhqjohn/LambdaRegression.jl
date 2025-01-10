"""
    Regression

Frontier regression runner interfaces.

This module defines the **top-level API** for running population-based regression
over typed lambda-calculus programs. The implementation is intentionally staged:
we stabilize the interfaces (components/config/context plumbing) before the
end-to-end evolutionary loop.

Design goals
- High composability: configuration is a `NamedTuple` and components are injected
  as a `NamedTuple`.
- Concurrency readiness: SR/regrssion paths must not depend on global factories
  (e.g. shared `newtyvar` counters or `gensym` naming).
- Const injection: inputs and intermediate values flow through a constant pool
  whose types are visible to individuals.

Status
- `fit!` implements a minimal typed population loop with injected `evaluate` and `loss`.
- For observation hooks, `fit!` supports `callback(state)` and an optional `components.init_population(...)` hook.
- Thread-safe primitives for SR paths live in `ASTs.Contexts.EvalContext` and
  `TypeSystem.make_tyvar_factory`.
"""
module Regression

using Random
using TestItems

import Base.Threads

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, make_tyvar_factory, get_type_ctx
using ..ASTs.Contexts: EvalContext, ConstSymbolFactory, new_constant!
using ..ASTs.NodeASTs: AstTree, AstNode, AppNode, AbsNode, ConstNode, IndexNode, SymbolGenerator
using ..Reduces: beta_reduce!
using ..GenerateTree
using ..Mutations
using ..Mutations: Populations
using ..Mutations.Populations: Individual

export Regressor, SupervisedDataset, FitResult, fit!
export default_config, default_components, default_regressor
export batch_generate_population

struct NotImplementedErr end
struct MissingComponentErr
    key::Symbol
end
struct MaxReduceStepsErr
    steps::Int
end
struct NonConstOutputErr end
struct MissingConstValueErr
    name::Symbol
end
struct NonClosedLambdaErr
    f::Symbol
end

"""
    SupervisedDataset(X, Y)

Simple supervised dataset container.

`X` and `Y` are vectors of samples; lengths must match.
"""
struct SupervisedDataset{TX,TY}
    X::Vector{TX}
    Y::Vector{TY}
    function SupervisedDataset(X::Vector{TX}, Y::Vector{TY}) where {TX,TY}
        length(X) == length(Y) || throw(ArgumentError("X and Y must have the same length"))
        return new{TX,TY}(X, Y)
    end
end

"""
    Regressor(target_ty; components=..., config=..., rng=...)

Top-level regression runner object.

- `target_ty` is the program type we are trying to fit (e.g. `A → B`).
- `components` is a composable `NamedTuple` of injected strategies, factories
  and contexts.
- `config` is a composable `NamedTuple` of hyperparameters.

This type is intentionally lightweight and mostly acts as a stable interface.
"""
struct Regressor
    target_ty::Ty
    components::NamedTuple
    config::NamedTuple
    rng::AbstractRNG
end

"""
    FitResult

Result container for `fit!`.
"""
struct FitResult
    best
    status::Symbol
    message::String
    history::Vector{Any}
end

########################################################
# Loss helpers
########################################################

function _sum_sq_err(yhat::Number, y::Number)
    d = yhat - y
    return float(d * d), 1
end

function _sum_sq_err(yhat::AbstractArray, y::AbstractArray)
    size(yhat) == size(y) || throw(DimensionMismatch("mse: size mismatch $(size(yhat)) vs $(size(y))"))
    s = 0.0
    n = 0
    for (a, b) in zip(yhat, y)
        da = float(a - b)
        s += da * da
        n += 1
    end
    return s, n
end

function _sum_sq_err(yhat::Vector{<:AbstractArray}, y::Vector{<:AbstractArray})
    length(yhat) == length(y) || throw(DimensionMismatch("mse: outer length mismatch $(length(yhat)) vs $(length(y))"))
    s = 0.0
    n = 0
    for (a, b) in zip(yhat, y)
        (si, ni) = _sum_sq_err(a, b)
        s += si
        n += ni
    end
    return s, n
end

"""
    mse(yhat, y) -> Float64

Basic mean-squared error for:
- scalars,
- arrays/tensors (`AbstractArray`),
- nested array-of-arrays (e.g. `Vector{Vector{Float64}}` matrices).
"""
function mse(yhat, y)::Float64
    (s, n) = _sum_sq_err(yhat, y)
    n > 0 || throw(ArgumentError("mse: empty inputs"))
    return s / n
end

########################################################
# Evaluation helpers
########################################################

function _node_count(tree::AstTree)::Int
    c = 0
    for _ in tree
        c += 1
    end
    return c
end

function _apply_inputs(program::AstTree, input_slots::Vector{Symbol})::AstTree
    node = deepcopy(program.root)
    for slot in input_slots
        node = AppNode(node, ConstNode(slot))
    end
    return AstTree(node)
end

function _is_closed(node::AstNode, depth::Int)::Bool
    if node isa IndexNode
        return (node::IndexNode).index < depth
    elseif node isa AbsNode
        return _is_closed((node::AbsNode).body, depth + 1)
    elseif node isa AppNode
        n = node::AppNode
        return _is_closed(n.func, depth) && _is_closed(n.arg, depth)
    else
        # ConstNode is always closed; VarNode indicates a free variable in our discipline.
        return node isa ConstNode
    end
end

_is_closed_abs(abs::AbsNode)::Bool = _is_closed(abs.body, 1)

function _capture_env(ctx::EvalContext)::Dict{Symbol,Any}
    env = Dict{Symbol,Any}()
    merge!(env, ctx.globals)
    merge!(env, ctx.locals)
    merge!(env, ctx.overlay)
    return env
end

struct LiftedClosure
    ast::AstTree
    captured_env::Dict{Symbol,Any}
    make_eval_context
    const_symfactory_ctor
    max_steps::Int
    closure_lift_max_per_pass::Int
end

function _normalize_with_lift!(tree::AstTree, ctx::EvalContext, components::NamedTuple, config::NamedTuple, steps::Int)::Bool
    for _ in 1:steps
        normalized = beta_reduce!(tree, ctx)
        if normalized
            progressed = closure_lift!(tree, ctx, components, config)
            progressed || return true
        end
    end
    return false
end

function (cl::LiftedClosure)(x)
    call_ctx = cl.make_eval_context(;
        overlay=Dict{Symbol,Any}(),
        locals=Dict{Symbol,Any}(),
        globals=cl.captured_env,
        symfactory=cl.const_symfactory_ctor(),
    )
    arg = new_constant!(call_ctx, x)
    call_tree = AstTree(AppNode(deepcopy(cl.ast.root), arg))
    ok = _normalize_with_lift!(
        call_tree,
        call_ctx,
        (; make_eval_context=cl.make_eval_context, const_symfactory_ctor=cl.const_symfactory_ctor),
        (;
            closure_lift_max_per_pass=cl.closure_lift_max_per_pass,
            closure_max_reduce_steps=cl.max_steps,
            max_reduce_steps=cl.max_steps,
        ),
        cl.max_steps,
    )
    ok || throw(MaxReduceStepsErr(cl.max_steps))
    if call_tree.root isa ConstNode
        name = (call_tree.root::ConstNode).name
        haskey(call_ctx, name) || throw(MissingConstValueErr(name))
        return call_ctx[name]
    elseif call_tree.root isa AbsNode
        # Higher-order case (partial application): return a callable closure constant.
        #
        # This is critical for curried lambdas, e.g. op(acc) returning (v -> ...),
        # so that host primitives like AFOLDL can call op(acc)(v).
        return LiftedClosure(
            AstTree(deepcopy(call_tree.root)),
            _capture_env(call_ctx),
            cl.make_eval_context,
            cl.const_symfactory_ctor,
            cl.max_steps,
            cl.closure_lift_max_per_pass,
        )
    else
        throw(NonConstOutputErr())
    end
end

function closure_lift!(tree::AstTree, ctx::EvalContext, components::NamedTuple, config::NamedTuple)::Bool
    max_per_pass = get(config, :closure_lift_max_per_pass, 64)
    lifted = Ref(0)

    function visit(node::AstNode, under_abs::Bool)
        under_abs && return
        node isa AbsNode && return
        if node isa AppNode
            app = node::AppNode
            if app.func isa ConstNode && app.arg isa AbsNode && lifted[] < max_per_pass
                f = (app.func::ConstNode).name
                abs = app.arg::AbsNode
                _is_closed_abs(abs) || throw(NonClosedLambdaErr(f))
                cl = LiftedClosure(
                    AstTree(deepcopy(abs)),
                    _capture_env(ctx),
                    components.make_eval_context,
                    components.const_symfactory_ctor,
                    get(config, :closure_max_reduce_steps, get(config, :max_reduce_steps, 64)),
                    max_per_pass,
                )
                app.arg = new_constant!(ctx, cl)
                lifted[] += 1
            end
            visit(app.func, under_abs)
            visit(app.arg, under_abs)
        end
        return
    end

    visit(tree.root, false)
    return lifted[] > 0
end

"""
    evaluate_default(ind::Individual, x, components, config) -> Union{Any,Err}

Default evaluation strategy:
- inject input values into an `EvalContext` overlay under `components.input_slots`,
- evaluate `(program x1 x2 ...)` by beta-reducing for `config.max_reduce_steps` steps,
- require a `ConstNode` result whose value is in the evaluation context.
"""
function evaluate_default(ind::Individual, x, components::NamedTuple, config::NamedTuple)
    input_slots = components.input_slots::Vector{Symbol}
    length(input_slots) == 0 && return Err(ArgumentError("input_slots must be non-empty for evaluate_default"))

    # Map sample -> slot values.
    overlay = Dict{Symbol,Any}()
    if length(input_slots) == 1
        overlay[input_slots[1]] = x
    else
        x isa Tuple || return Err(ArgumentError("multi-input expects x::Tuple"))
        length(x) == length(input_slots) || return Err(ArgumentError("input tuple length mismatch"))
        for (slot, v) in zip(input_slots, x)
            overlay[slot] = v
        end
    end

    ctx = components.make_eval_context(;
        overlay=overlay,
        locals=ind.local_consts,
        globals=components.global_consts,
        symfactory=components.const_symfactory_ctor(),
    )

    tree = _apply_inputs(ind.tree, input_slots)
    steps = get(config, :max_reduce_steps, 64)
    normalized = _normalize_with_lift!(tree, ctx, components, config, steps)
    normalized || return Err(MaxReduceStepsErr(steps))

    tree.root isa ConstNode || return Err(NonConstOutputErr())
    name = (tree.root::ConstNode).name
    haskey(ctx, name) || return Err(MissingConstValueErr(name))
    return ctx[name]
end

########################################################
# Selection helpers
########################################################

function _tournament_index(rng::AbstractRNG, fitness::Vector{Float64}, tournament_size::Int)::Int
    n = length(fitness)
    best = rand(rng, 1:n)
    best_f = fitness[best]
    for _ in 2:tournament_size
        j = rand(rng, 1:n)
        fj = fitness[j]
        if fj < best_f
            best = j
            best_f = fj
        end
    end
    return best
end

########################################################
# Deterministic RNG derivation (schedule-independent)
########################################################

function _mix64(x::UInt64)::UInt64
    x ⊻= x >> 30
    x *= 0xbf58476d1ce4e5b9
    x ⊻= x >> 27
    x *= 0x94d049bb133111eb
    x ⊻= x >> 31
    return x
end

function _role_code(role::Symbol)::UInt64
    role === :init_tree && return 0x01
    role === :init_mutate && return 0x02
    role === :mutate && return 0x03
    role === :crossover && return 0x04
    role === :eval && return 0x05
    return 0xFF
end

function derive_rng(seed::UInt64, generation::Int, index::Int, role::Symbol)::MersenneTwister
    x = seed ⊻ (UInt64(generation) << 32) ⊻ UInt64(index) ⊻ (_role_code(role) << 56)
    s = _mix64(x)
    # `reinterpret(UInt32, s::UInt64)` errors because scalar `reinterpret` is a bitcast.
    # Seed via a small `Vector{UInt32}` instead (deterministic and portable).
    return MersenneTwister(UInt32[UInt32(s & 0xffffffff), UInt32(s >> 32)])
end

function _sorted_indices_by_fitness(fitness::Vector{Float64})::Vector{Int}
    return sortperm(fitness)
end

"""
    default_config(; kwargs...) -> NamedTuple

Load a default configuration. This function makes strong assumptions and is
intended only as a convenience.
"""
function default_config(;
    population_size::Int=200,
    generations::Int=50,
    error_threshold::Float64=1e-6,
    init_mutation_rounds::Int=1,
    max_reduce_steps::Int=64,
    closure_max_reduce_steps::Int=max_reduce_steps,
    closure_lift_max_per_pass::Int=64,
    mutation_rate::Float64=0.7,
    crossover_rate::Float64=0.2,
    elitism::Int=1,
    tournament_size::Int=7,
    error_penalty::Float64=1e12,
    repro=(;
        require_change=true,
        max_resample=10,
        on_fail=:immigrant,              # :immigrant or :clone
        immigrant_mutation_rounds=init_mutation_rounds,
        immigrant_max_try=3,
        immigrant_gen_max_depth=nothing,
        immigrant_gen_max_try=nothing,
        immigrants_per_gen=0,
    ),
    stats=(;
        enabled=false,
        mutation=false,
        reproduction=false,
        print=false,
        every=1,
        topk_types=8,
    ),
    log=(; enabled=false, every=1, topk=5, io=stdout),
    parallel=(; enabled=true, mode=:individual, schedule=:dynamic),
    kwargs...
)
    base = (;
        population_size,
        generations,
        error_threshold,
        init_mutation_rounds,
        max_reduce_steps,
        closure_max_reduce_steps,
        closure_lift_max_per_pass,
        mutation_rate,
        crossover_rate,
        elitism,
        tournament_size,
        error_penalty,
        repro,
        stats,
        log,
        parallel,
    )
    return merge(base, (;kwargs...))
end

"""
    default_components(; kwargs...) -> NamedTuple

Load default component wiring for regression.

This is a **loader only**: it does not run any business logic. The returned
components are designed to be easy to override via `merge`.
"""
function default_components(;
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    input_slots::Vector{Symbol}=Symbol[],
    input_types::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    # --- Tier1 defaults (can be overridden via merge/kwargs) ---
    gen_ctors::Vector{<:GenerateTree.AbstractCtor}=[
        GenerateTree.AbsCtor(),
        GenerateTree.IndexCtor(),
        GenerateTree.GlobalConstCtor(),
        GenerateTree.ApplySpineCtor(max_spine_args=3),
    ],
    gen_chooser::GenerateTree.AbstractChooser=GenerateTree.FocusedChooser(max_spine_args=3),
    gen_max_depth::Int=9,
    gen_max_try::Int=80,
    mutations::Vector{<:Mutations.AbstractMutation}=[
        Mutations.ScalarConstMutation(),
        Mutations.BoolConstMutation(),
        Mutations.ArrayConstMutation(),
        Mutations.SubtreeReplaceMutation(),
        Mutations.PointMutation(),
        Mutations.CSEMutation(),
        Mutations.BetaReduceMutation(),
        Mutations.EtaReduceMutation(),
    ],
    mutation_chooser::Mutations.AbstractPairChooser=Mutations.WeightedPairChooser(),
    mutation_provider::Mutations.AbstractSiteProvider=Mutations.IndexedProvider(),
    mutation_max_try::Int=25,
    mutation_prune_consts::Bool=true,
    crossover_op=Populations.TypedSubtreeSwapCrossover(),
    kwargs...
)
    global_consts_ty === nothing && (global_consts_ty = get_type_ctx(global_consts))

    # Thread-safe evaluation context constructor:
    # writes go to `overlay` only; `globals` is assumed read-only.
    make_eval_context = (;
        overlay::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        locals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        globals::Dict{Symbol,Any}=global_consts,
        symfactory::ConstSymbolFactory=const_symfactory_ctor(),
    ) -> EvalContext(overlay=overlay, locals=locals, globals=globals, symfactory=symfactory)

    base = (;
        global_consts,
        global_consts_ty,
        input_slots,
        input_types,
        tyfactory_ctor,
        const_symfactory_ctor,
        make_eval_context,
        gen_ctors,
        gen_chooser,
        gen_max_depth,
        gen_max_try,
        mutations,
        mutation_chooser,
        mutation_provider,
        mutation_max_try,
        mutation_prune_consts,
        crossover_op,
    )
    return merge(base, (;kwargs...))
end

"""
    default_regressor(target_ty::Ty; kwargs...) -> Regressor

Convenience constructor: loads `default_config` and `default_components`.
"""
function default_regressor(target_ty::Ty;
    rng::AbstractRNG=MersenneTwister(2000),
    config::NamedTuple=default_config(),
    components::NamedTuple=default_components(),
    kwargs...
)
    reg = Regressor(target_ty, components, config, rng)
    return reg
end

"""
    batch_generate_population(reg, data; N, seed, global_type_ctx, mutate_rounds, purpose=:init, recorder=nothing) -> Union{Vector{Individual},Err}

Generate a population of `N` individuals by:
1) generating trees via injected `components.generate_tree` (default: `GenerateTree.generate_tree`),
2) applying `mutate_rounds` mutation rounds in-place (default: `Mutations.mutate_tree`).

This is a utility intended for initialization and immigrant injection.
"""
function batch_generate_population(reg::Regressor, data::SupervisedDataset;
    N::Int,
    seed::UInt64,
    global_type_ctx::Dict{Symbol,TyLike},
    mutate_rounds::Int,
    purpose::Symbol=:init,
    recorder=nothing,
    gen_max_depth_override::Union{Nothing,Int}=nothing,
    gen_max_try_override::Union{Nothing,Int}=nothing,
)
    components = reg.components
    gen_fn = haskey(components, :generate_tree) ? components.generate_tree : GenerateTree.generate_tree
    mut_fn = haskey(components, :mutate_tree) ? components.mutate_tree : Mutations.mutate_tree

    pop = Vector{Individual}(undef, N)
    for i in 1:N
        rng_i = derive_rng(seed, 0, i, Symbol(purpose, :_tree))
        tyfactory = components.tyfactory_ctor()
        symfactory = SymbolGenerator()
        const_symfactory = components.const_symfactory_ctor()
        local_consts = Dict{Symbol,Any}()
        local_consts_ty = Dict{Symbol,TyLike}()

        tree = gen_fn(
            reg.target_ty;
            ctors=components.gen_ctors,
            chooser=components.gen_chooser,
            max_depth=(gen_max_depth_override === nothing ? components.gen_max_depth : gen_max_depth_override),
            max_try=(gen_max_try_override === nothing ? components.gen_max_try : gen_max_try_override),
            tyfactory=tyfactory,
            symfactory=symfactory,
            rng=rng_i,
            global_consts_ty=global_type_ctx,
            local_consts=local_consts,
            local_consts_ty=local_consts_ty,
        )
        tree isa Err && return tree

        cur_tree = tree
        for r in 1:mutate_rounds
            rng_m = derive_rng(seed, 0, (i << 8) + r, Symbol(purpose, :_mutate))
            mutated = mut_fn(
                cur_tree,
                reg.target_ty;
                rng=rng_m,
                mutations=components.mutations,
                chooser=components.mutation_chooser,
                provider=components.mutation_provider,
                max_try=components.mutation_max_try,
                prune_consts=components.mutation_prune_consts,
                copy_tree=true,
                copy_ctx=true,
                global_consts_ty=global_type_ctx,
                local_consts=local_consts,
                local_consts_ty=local_consts_ty,
                gen_ctors=components.gen_ctors,
                gen_chooser=components.gen_chooser,
                gen_max_depth=components.gen_max_depth,
                gen_max_try=components.gen_max_try,
                tyfactory=tyfactory,
                symfactory=symfactory,
                const_symfactory=const_symfactory,
                recorder=recorder,
            )
            mutated isa Err && break
            cur_tree = mutated
        end

        pop[i] = Individual(cur_tree, reg.target_ty; local_consts=local_consts, local_consts_ty=local_consts_ty, fitness=nothing, meta=(;))
    end
    return pop
end

"""
    fit!(reg::Regressor, data::SupervisedDataset; callback=nothing) -> Union{FitResult,Err}

Run a minimal population-based regression loop.

This function is intentionally conservative: it aims to "run end-to-end" with
strong DI boundaries, not to be optimal.
"""
function fit!(reg::Regressor, data::SupervisedDataset; callback=nothing)
    length(data.X) == length(data.Y) || return Err(ArgumentError("X and Y must have the same length"))
    components = reg.components
    config = reg.config

    # Required component keys.
    required = Symbol[
        :gen_ctors, :gen_chooser, :gen_max_depth, :gen_max_try,
        :mutations, :mutation_chooser, :mutation_provider, :mutation_max_try, :mutation_prune_consts,
        :crossover_op,
        :global_consts, :global_consts_ty,
        :input_slots, :input_types,
        :tyfactory_ctor, :const_symfactory_ctor, :make_eval_context,
    ]
    for k in required
        haskey(components, k) || return Err(MissingComponentErr(k))
    end

    loss_fn = haskey(components, :loss) ? components.loss : mse
    eval_fn = haskey(components, :evaluate) ? components.evaluate : evaluate_default
    complexity_fn = haskey(components, :complexity) ? components.complexity : _node_count

    N = config.population_size
    gens = config.generations
    threshold = config.error_threshold
    error_penalty = config.error_penalty

    seed = get(config, :seed, rand(reg.rng, UInt64))

    # Merge typing contexts used by crossover (and optionally by evaluation logging).
    global_type_ctx = merge(components.global_consts_ty, components.input_types)

    # --- Init population ---
    pop = if haskey(components, :init_population)
        res = components.init_population(reg, data; N=N, seed=seed, global_type_ctx=global_type_ctx)
        res isa Err && return res
        length(res) == N || return Err(ArgumentError("init_population must return exactly N individuals"))
        res
    else
        init_recorder = nothing
        cfg_stats = get(config, :stats, (; enabled=false))
        if get(cfg_stats, :enabled, false) && get(cfg_stats, :mutation, false)
            init_recorder = Mutations.MutationAttemptRecorder()
        end
        res = batch_generate_population(
            reg,
            data;
            N=N,
            seed=seed,
            global_type_ctx=global_type_ctx,
            mutate_rounds=config.init_mutation_rounds,
            purpose=:init,
            recorder=init_recorder,
        )
        res isa Err && return res
        res
    end

    history = Any[]
    best = nothing
    best_loss = error_penalty

    for g in 1:gens
        losses = fill(error_penalty, N)
        fitness = fill(error_penalty, N)
        complexities = fill(0, N)
        eval_err_counts = zeros(Int, N)
        eval_exception_counts = zeros(Int, N)
        max_reduce_steps_counts = zeros(Int, N)
        non_const_output_counts = zeros(Int, N)
        missing_const_value_counts = zeros(Int, N)
        non_closed_lambda_counts = zeros(Int, N)

        parallel = get(config, :parallel, (; enabled=false))
        do_parallel = get(parallel, :enabled, false) && Threads.nthreads() > 1

        function _classify_eval_issue!(i::Int, issue)
            issue isa MaxReduceStepsErr && (max_reduce_steps_counts[i] += 1; return nothing)
            issue isa NonConstOutputErr && (non_const_output_counts[i] += 1; return nothing)
            issue isa MissingConstValueErr && (missing_const_value_counts[i] += 1; return nothing)
            issue isa NonClosedLambdaErr && (non_closed_lambda_counts[i] += 1; return nothing)
            return nothing
        end

        function _eval_ind!(i::Int)
            ind = pop[i]
            total = 0.0
            count = 0
            for (xj, yj) in zip(data.X, data.Y)
                yhat = try
                    eval_fn(ind, xj, components, config)
                catch e
                    eval_exception_counts[i] += 1
                    _classify_eval_issue!(i, e)
                    total += error_penalty
                    count += 1
                    continue
                end
                if yhat isa Err
                    eval_err_counts[i] += 1
                    _classify_eval_issue!(i, yhat.value)
                    total += error_penalty
                else
                    try
                        total += loss_fn(yhat, yj)
                    catch
                        total += error_penalty
                    end
                end
                count += 1
            end
            losses[i] = total / max(count, 1)
            complexities[i] = complexity_fn(ind.tree)
            fitness[i] = losses[i] # parsimony can be injected later
            return nothing
        end

        if do_parallel
            Threads.@threads :dynamic for i in 1:N
                _eval_ind!(i)
            end
        else
            for i in 1:N
                _eval_ind!(i)
            end
        end

        # Track best.
        perm = _sorted_indices_by_fitness(fitness)
        best_idx = perm[1]
        gen_best_loss = losses[best_idx]
        if gen_best_loss < best_loss
            best_loss = gen_best_loss
            best = pop[best_idx]
        end
        eval_stats = (;
            n_eval_err=sum(eval_err_counts),
            n_eval_exception=sum(eval_exception_counts),
            n_max_reduce_steps=sum(max_reduce_steps_counts),
            n_non_const_output=sum(non_const_output_counts),
            n_missing_const_value=sum(missing_const_value_counts),
            n_non_closed_lambda=sum(non_closed_lambda_counts),
            per_individual=(;
                eval_err=copy(eval_err_counts),
                eval_exception=copy(eval_exception_counts),
                max_reduce_steps=copy(max_reduce_steps_counts),
                non_const_output=copy(non_const_output_counts),
                missing_const_value=copy(missing_const_value_counts),
                non_closed_lambda=copy(non_closed_lambda_counts),
            ),
        )

        # Logging: main thread only.
        logcfg = get(config, :log, (; enabled=false))
        if get(logcfg, :enabled, false) && (g % get(logcfg, :every, 1) == 0)
            io = get(logcfg, :io, stdout)
            topk = min(get(logcfg, :topk, 5), N)
            println(io, "gen=$g best_loss=$(gen_best_loss)")
            for rank in 1:topk
                i = perm[rank]
                tree_str = sprint(show, MIME"text/plain"(), pop[i].tree)
                println(io, "  [$rank] loss=$(losses[i]) nodes=$(complexities[i])  $tree_str")
            end
            try
                flush(io)
            catch
            end
        end

        if callback !== nothing
            state = (;
                gen=g,
                pop=pop,
                losses=losses,
                fitness=fitness,
                complexities=complexities,
                perm=perm,
                gen_best_loss=gen_best_loss,
                best=best,
                best_loss=best_loss,
                eval_stats=eval_stats,
                seed=seed,
                components=components,
                config=config,
            )
            cb_res = callback(state)
            cb_res isa Err && return cb_res
            if cb_res === :stop
                push!(history, (; gen=g, best_loss=gen_best_loss, eval_stats, repro_stats=nothing, mutation_stats=nothing))
                return FitResult(best, :stopped, "Stopped by callback", history)
            end
        end

        if gen_best_loss <= threshold
            push!(history, (; gen=g, best_loss=gen_best_loss, eval_stats, repro_stats=nothing, mutation_stats=nothing))
            return FitResult(best, :threshold, "Reached error threshold", history)
        end

        # --- Reproduce ---
        reprocfg = get(config, :repro, (; require_change=false, max_resample=0, on_fail=:clone, immigrant_mutation_rounds=0, immigrant_max_try=1, immigrants_per_gen=0))
        statscfg = get(config, :stats, (; enabled=false))
        do_stats = get(statscfg, :enabled, false)
        rec_mut = (do_stats && get(statscfg, :mutation, false)) ? Mutations.MutationAttemptRecorder() : nothing
        track_repro = do_stats && get(statscfg, :reproduction, false)
        repro_stats = track_repro ? Dict{Symbol,Int}(
            :n_resample => 0,
            :n_immigrant => 0,
            :n_clone_fallback => 0,
            :n_child_total => 0,
            :n_child_clone => 0,
            :n_child_changed => 0,
            :crossover_attempt => 0,
            :crossover_ok => 0,
            :crossover_err => 0,
            :mutation_attempt => 0,
            :mutation_ok => 0,
            :mutation_err => 0,
        ) : nothing

        @inline function _inc!(st, key::Symbol, delta::Int=1)
            st === nothing && return nothing
            st[key] = get(st, key, 0) + delta
            return nothing
        end

        next_pop = Vector{Individual}()
        elitism = min(get(config, :elitism, 1), N)
        for j in 1:elitism
            push!(next_pop, pop[perm[j]])
        end

        tournament_size = get(config, :tournament_size, 7)
        mutation_rate = get(config, :mutation_rate, 0.7)
        crossover_rate = get(config, :crossover_rate, 0.2)
        crossover_op = components.crossover_op

        pending_children = Individual[]
        require_change = get(reprocfg, :require_change, false)
        max_resample = max(0, get(reprocfg, :max_resample, 0))
        on_fail = get(reprocfg, :on_fail, :clone)
        immigrant_rounds = max(0, get(reprocfg, :immigrant_mutation_rounds, 0))
        immigrant_max_try = max(1, get(reprocfg, :immigrant_max_try, 1))

        function _make_immigrant(slot_idx::Int)
            for k in 1:immigrant_max_try
                seed_i = seed ⊻ UInt64(g) ⊻ (UInt64(slot_idx) << 32) ⊻ UInt64(k)
                ig_depth = get(reprocfg, :immigrant_gen_max_depth, nothing)
                ig_try = get(reprocfg, :immigrant_gen_max_try, nothing)
                res = batch_generate_population(
                    reg,
                    data;
                    N=1,
                    seed=seed_i,
                    global_type_ctx=global_type_ctx,
                    mutate_rounds=immigrant_rounds,
                    purpose=:immigrant,
                    recorder=rec_mut,
                    gen_max_depth_override=ig_depth,
                    gen_max_try_override=ig_try,
                )
                res isa Err && continue
                return res[1]
            end
            return Err(Mutations.MaxTryReachedErr())
        end

        immigrants_fixed = max(0, get(reprocfg, :immigrants_per_gen, 0))
        if immigrants_fixed > 0
            k = min(immigrants_fixed, N - length(next_pop))
            if k > 0
                ig_depth = get(reprocfg, :immigrant_gen_max_depth, nothing)
                ig_try = get(reprocfg, :immigrant_gen_max_try, nothing)
                res = batch_generate_population(
                    reg,
                    data;
                    N=k,
                    seed=seed ⊻ UInt64(g) ⊻ 0x9e3779b97f4a7c15,
                    global_type_ctx=global_type_ctx,
                    mutate_rounds=immigrant_rounds,
                    purpose=:immigrant,
                    recorder=rec_mut,
                    gen_max_depth_override=ig_depth,
                    gen_max_try_override=ig_try,
                )
                if !(res isa Err)
                    append!(next_pop, res)
                    _inc!(repro_stats, :n_immigrant, k)
                end
            end
        end

        while length(next_pop) < N
            if !isempty(pending_children)
                push!(next_pop, popfirst!(pending_children))
                continue
            end

            produced = false
            slot_idx = length(next_pop) + 1
            last_parent = pop[perm[1]]

            for attempt in 1:max(1, max_resample)
                attempt > 1 && _inc!(repro_stats, :n_resample, 1)

                rng_sel = derive_rng(seed, g, (slot_idx << 8) + attempt, :eval)
                p1 = pop[_tournament_index(rng_sel, fitness, tournament_size)]
                p2 = pop[_tournament_index(rng_sel, fitness, tournament_size)]
                last_parent = p1

                base_children = (p1, p2)
                crossed = false
                rng_cross = derive_rng(seed, g, (slot_idx << 8) + attempt, :crossover)
                if crossover_op !== nothing && rand(rng_cross) < crossover_rate
                    _inc!(repro_stats, :crossover_attempt, 1)
                    res = Populations.crossover(
                        crossover_op,
                        p1,
                        p2;
                        rng=rng_cross,
                        global_consts_ty=global_type_ctx,
                        tyfactory_ctor=components.tyfactory_ctor,
                        const_symfactory_ctor=components.const_symfactory_ctor,
                        prune=true,
                    )
                    if res isa Err
                        _inc!(repro_stats, :crossover_err, 1)
                    else
                        _inc!(repro_stats, :crossover_ok, 1)
                        base_children = res
                        crossed = true
                    end
                end

                new_children = Individual[]
                ci = 0
                for child0 in base_children
                    ci += 1
                    child = child0
                    changed = crossed
                    rng_mut = derive_rng(seed, g, (slot_idx << 10) + (attempt << 2) + ci, :mutate)
                    if rand(rng_mut) < mutation_rate
                        _inc!(repro_stats, :mutation_attempt, 1)
                        tyfactory = components.tyfactory_ctor()
                        symfactory = SymbolGenerator()
                        const_symfactory = components.const_symfactory_ctor()
                        local_consts = copy(child.local_consts)
                        local_consts_ty = copy(child.local_consts_ty)
                        mutated = Mutations.mutate_tree(
                            child.tree,
                            reg.target_ty;
                            rng=rng_mut,
                            mutations=components.mutations,
                            chooser=components.mutation_chooser,
                            provider=components.mutation_provider,
                            max_try=components.mutation_max_try,
                            prune_consts=components.mutation_prune_consts,
                            copy_tree=true,
                            copy_ctx=true,
                            global_consts_ty=global_type_ctx,
                            local_consts=local_consts,
                            local_consts_ty=local_consts_ty,
                            gen_ctors=components.gen_ctors,
                            gen_chooser=components.gen_chooser,
                            gen_max_depth=components.gen_max_depth,
                            gen_max_try=components.gen_max_try,
                            tyfactory=tyfactory,
                            symfactory=symfactory,
                            const_symfactory=const_symfactory,
                            recorder=rec_mut,
                        )
                        if mutated isa Err
                            _inc!(repro_stats, :mutation_err, 1)
                        else
                            _inc!(repro_stats, :mutation_ok, 1)
                            changed = true
                            child = Individual(mutated, reg.target_ty; local_consts=local_consts, local_consts_ty=local_consts_ty, fitness=nothing, meta=(;))
                        end
                    end

                    if require_change && !changed
                        continue
                    end
                    _inc!(repro_stats, :n_child_total, 1)
                    if changed
                        _inc!(repro_stats, :n_child_changed, 1)
                    else
                        _inc!(repro_stats, :n_child_clone, 1)
                    end
                    push!(new_children, child)
                    length(next_pop) + length(new_children) >= N && break
                end

                if !isempty(new_children)
                    append!(pending_children, new_children)
                    produced = true
                    break
                end
            end

            if produced
                continue
            end

            if on_fail == :immigrant
                imm = _make_immigrant(slot_idx)
                if imm isa Err
                    _inc!(repro_stats, :n_clone_fallback, 1)
                    push!(next_pop, last_parent)
                else
                    _inc!(repro_stats, :n_immigrant, 1)
                    push!(next_pop, imm)
                end
            else
                _inc!(repro_stats, :n_clone_fallback, 1)
                push!(next_pop, last_parent)
            end
        end

        pop = next_pop

        mutation_stats = rec_mut === nothing ? nothing : Mutations.mutation_recorder_summary(rec_mut; topk=get(statscfg, :topk_types, 8))
        if do_stats && get(statscfg, :print, false) && (g % get(statscfg, :every, 1) == 0)
            io = get(logcfg, :io, stdout)
            if repro_stats !== nothing
                println(io, "  repro: children=$(repro_stats[:n_child_total]) clone=$(repro_stats[:n_child_clone]) changed=$(repro_stats[:n_child_changed]) resample=$(repro_stats[:n_resample]) immigrant=$(repro_stats[:n_immigrant]) clone_fallback=$(repro_stats[:n_clone_fallback]) cross(ok/err)=$(repro_stats[:crossover_ok])/$(repro_stats[:crossover_err]) mut(ok/err)=$(repro_stats[:mutation_ok])/$(repro_stats[:mutation_err])")
            end
            if mutation_stats !== nothing
                tot = mutation_stats.totals
                println(io, "  mut: success=$(tot.success) apply_err=$(tot.apply_err) infer_err=$(tot.infer_err) incompatible=$(tot.incompatible) no_site=$(tot.no_eligible_site) bloat=$(tot.bloat_exceeded) max_try=$(tot.max_try_exhausted)")
            end
            try
                flush(io)
            catch
            end
        end

        push!(history, (; gen=g, best_loss=gen_best_loss, eval_stats, repro_stats=repro_stats === nothing ? nothing : copy(repro_stats), mutation_stats))
    end

    return FitResult(best, :max_generations, "Reached max generations", history)
end

@testsnippet RegBaseImports begin
    using LambdaRegression.Regression: default_components
    using LambdaRegression.TypeSystem: make_tyvar_factory
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory
    using LambdaRegression.Regression: mse
    using Random
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike
    using LambdaRegression.Regression: evaluate_default, default_components, default_config
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.TypeSystem
    using LambdaRegression.Utils: Err
    using LambdaRegression.GenerateTree: AbsCtor, IndexCtor, GlobalConstCtor, ApplySpineCtor, FocusedChooser
    using LambdaRegression.Mutations
    using LambdaRegression.Regression: Regressor, SupervisedDataset, fit!, evaluate_default, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.Regression: Regressor, SupervisedDataset, fit!, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike
    using LambdaRegression.GenerateTree: AbsCtor, IndexCtor, FocusedChooser
    using LambdaRegression.Mutations: WeightedPairChooser
    using LambdaRegression.Regression:
        Regressor,
        SupervisedDataset,
        fit!,
        default_components,
        default_config,
        MaxReduceStepsErr,
        NonConstOutputErr,
        NonClosedLambdaErr,
        MissingConstValueErr
end

@testitem "Regression default_components provides non-global factories" setup=[RegBaseImports] begin
    c = default_components()
    @test c.tyfactory_ctor === make_tyvar_factory
    f = c.tyfactory_ctor()
    @test f(0).id == 1
    sf = c.const_symfactory_ctor()
    @test sf isa ConstSymbolFactory
end

@testitem "Regression mse basic cases" setup=[RegBaseImports] begin
    @test mse(2.0, 1.0) == 1.0
    @test mse([1.0, 2.0], [1.0, 4.0]) == 2.0
    @test mse([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 6.0]]) == 1.0
end

@testitem "Regression evaluate_default identity works for AoA matrix" setup=[RegBaseImports] begin
    x = [[1.0, 2.0], [3.0, 4.0]]
    A = TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    target_ty = TyFun(A, A)
    tree = AstTree(AbsNode(:x, IndexNode(0)))
    ind = Individual(tree, target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())

    comps = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=20)
    y = evaluate_default(ind, x, comps, cfg)
    @test y == x
end

@testitem "Regression closure_lift unlocks AMAP with AST lambdas" setup=[RegBaseImports] begin
    x = [1.0, -2.0, 3.0]
    A = TyArray(TyNat(length(x)), FloatTy)
    target_ty = TyFun(A, A)

    inner = AbsNode(:y, AppNode(ConstNode(:SQR), IndexNode(0)))
    body = AppNode(AppNode(ConstNode(:AMAP), inner), IndexNode(0))
    tree = AstTree(AbsNode(:x, body))
    ind = Individual(tree, target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())

    global_consts = Dict{Symbol,Any}(
        :AMAP => TypeSystem.AMAP,
        :SQR => TypeSystem.SQR,
    )
    comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=200, closure_max_reduce_steps=200)
    y = evaluate_default(ind, x, comps, cfg)
    @test y == [v * v for v in x]
end

@testitem "Regression closure_lift supports curried lambdas for AFOLDL" setup=[RegBaseImports] begin
    x = [1.0, -2.0, 3.0]
    A = TyArray(TyNat(length(x)), FloatTy)
    target_ty = TyFun(A, FloatTy)

    # λx. AFOLDL (λacc.λv. FADD acc (SQR v)) ZERO x
    fold_fun = AbsNode(
        :acc,
        AbsNode(
            :v,
            AppNode(
                AppNode(ConstNode(:FADD), IndexNode(1)),
                AppNode(ConstNode(:SQR), IndexNode(0)),
            ),
        ),
    )
    body = AppNode(
        AppNode(AppNode(ConstNode(:AFOLDL), fold_fun), ConstNode(:ZERO)),
        IndexNode(0),
    )
    tree = AstTree(AbsNode(:x, body))
    ind = Individual(tree, target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())

    global_consts = Dict{Symbol,Any}(
        :AFOLDL => TypeSystem.AFOLDL,
        :FADD => TypeSystem.FADD,
        :SQR => TypeSystem.SQR,
        :ZERO => 0.0,
    )
    comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=600, closure_max_reduce_steps=300)
    y = evaluate_default(ind, x, comps, cfg)
    y isa Err && error("evaluate_default returned Err: $(y.value)")
    @test y ≈ sum(v * v for v in x)
end

@testitem "Regression non-closed lambda lift throws; fit! penalizes (non-fatal)" setup=[RegBaseImports] begin
    x = [1.0, 2.0]
    A = TyArray(TyNat(length(x)), FloatTy)
    target_ty = TyFun(A, A)

    # λx. AMAP (λy. #2) x   (inner lambda refers to an out-of-scope binder)
    bad_inner = AbsNode(:y, IndexNode(2))
    bad_body = AppNode(AppNode(ConstNode(:AMAP), bad_inner), IndexNode(0))
    bad_tree = AstTree(AbsNode(:x, bad_body))

    good_tree = AstTree(AbsNode(:x, IndexNode(0)))

    global_consts = Dict{Symbol,Any}(:AMAP => TypeSystem.AMAP)
    global_consts_ty = get_type_ctx(global_consts)

    base = default_components(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    comps = merge(
        base,
        (;
            gen_ctors=[AbsCtor(), IndexCtor(), GlobalConstCtor(), ApplySpineCtor(max_spine_args=3)],
            gen_chooser=FocusedChooser(max_spine_args=3),
            gen_max_depth=5,
            gen_max_try=10,
            mutations=[Mutations.SubtreeReplaceMutation()],
            mutation_chooser=Mutations.WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=5,
            mutation_prune_consts=true,
            crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) -> begin
                pop = Vector{Individual}(undef, N)
                pop[1] = Individual(bad_tree, reg.target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())
                pop[2] = Individual(good_tree, reg.target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())
                return pop
            end,
        ),
    )

    cfg = default_config(
        population_size=2,
        generations=1,
        error_threshold=0.0,
        max_reduce_steps=80,
        closure_max_reduce_steps=80,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=2,
    )
    data = SupervisedDataset([x], [[1.0, 2.0]])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))

    @test_throws LambdaRegression.Regression.NonClosedLambdaErr evaluate_default(
        Individual(bad_tree, target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}()),
        x,
        base,
        cfg,
    )

    res = fit!(reg, data)
    @test !(res isa Err)
end

@testitem "Regression fit! penalizes host primitive errors (non-fatal)" setup=[RegBaseImports] begin
    BOMB(x) = error("boom")

    target_ty = TyFun(FloatTy, FloatTy)
    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:BOMB), IndexNode(0))))

    global_consts = Dict{Symbol,Any}(:BOMB => BOMB)
    base = default_components(
        global_consts=global_consts,
        global_consts_ty=Dict{Symbol,TyLike}(:BOMB => TyFun(FloatTy, FloatTy)),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    comps = merge(
        base,
        (;
            gen_ctors=[AbsCtor(), IndexCtor(), GlobalConstCtor(), ApplySpineCtor(max_spine_args=3)],
            gen_chooser=FocusedChooser(max_spine_args=3),
            gen_max_depth=5,
            gen_max_try=10,
            mutations=[Mutations.SubtreeReplaceMutation()],
            mutation_chooser=Mutations.WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=5,
            mutation_prune_consts=true,
            crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) -> [Individual(tree, reg.target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}()) for _ in 1:N],
        ),
    )
    cfg = default_config(
        population_size=2,
        generations=1,
        error_threshold=0.0,
        max_reduce_steps=40,
        closure_max_reduce_steps=40,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=2,
    )
    data = SupervisedDataset([1.0], [1.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))
    res = fit!(reg, data)
    @test !(res isa Err)
end

@testitem "Regression fit! avoids implicit cloning via immigrant fallback when require_change=true" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    id_tree = AstTree(AbsNode(:x, IndexNode(0)))

    gen_stub = (target_ty; kwargs...) -> deepcopy(id_tree)

    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    comps = merge(
        base,
        (;
            generate_tree=gen_stub,
            gen_ctors=[AbsCtor(), IndexCtor()],
            gen_chooser=FocusedChooser(),
            gen_max_depth=3,
            gen_max_try=3,
            mutations=[Mutations.SubtreeReplaceMutation()],
            mutation_chooser=WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=1,
            mutation_prune_consts=true,
            crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) -> [Individual(id_tree, reg.target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}()) for _ in 1:N],
        ),
    )

    cfg = default_config(
        population_size=6,
        generations=1,
        error_threshold=-1.0,
        mutation_rate=0.0,   # forces require_change path to resample then immigrant
        crossover_rate=0.0,
        elitism=1,
        tournament_size=2,
        repro=(; require_change=true, max_resample=2, on_fail=:immigrant, immigrant_mutation_rounds=0, immigrant_max_try=2, immigrants_per_gen=0),
        stats=(; enabled=true, mutation=true, reproduction=true, print=false, every=1, topk_types=4),
    )
    data = SupervisedDataset([1.0], [1.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))

    res = fit!(reg, data)
    @test !(res isa Err)
    @test length(res.history) == 1
    rs = res.history[1].repro_stats
    @test rs[:n_immigrant] == 5
    @test rs[:n_clone_fallback] == 0
end

@testitem "Regression fit! callback exposes eval_stats classifications" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    ind_tree = AstTree(AbsNode(:x, IndexNode(0)))
    ind = Individual(ind_tree, target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())

    eval_fn = (ind, x, components, config) -> begin
        if x == 1.0
            return Err(MaxReduceStepsErr(7))
        elseif x == 2.0
            return Err(NonConstOutputErr())
        elseif x == 3.0
            throw(NonClosedLambdaErr(:f))
        elseif x == 4.0
            return Err(MissingConstValueErr(:z))
        else
            return x
        end
    end

    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    comps = merge(
        base,
        (;
            evaluate=eval_fn,
            gen_ctors=[AbsCtor(), IndexCtor()],
            gen_chooser=FocusedChooser(),
            gen_max_depth=3,
            gen_max_try=3,
            mutations=[Mutations.SubtreeReplaceMutation()],
            mutation_chooser=Mutations.WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=1,
            mutation_prune_consts=true,
            crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) -> [ind for _ in 1:N],
        ),
    )

    cfg = default_config(
        population_size=1,
        generations=1,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
    )
    data = SupervisedDataset([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))

    state_ref = Ref{Any}(nothing)
    res = fit!(reg, data; callback=(state -> (state_ref[] = state; nothing)))
    @test !(res isa Err)
    @test state_ref[] !== nothing
    @test length(res.history) == 1

    stats = state_ref[].eval_stats
    @test stats.n_eval_err == 3
    @test stats.n_eval_exception == 1
    @test stats.n_max_reduce_steps == 1
    @test stats.n_non_const_output == 1
    @test stats.n_missing_const_value == 1
    @test stats.n_non_closed_lambda == 1

    @test sum(stats.per_individual.eval_err) == stats.n_eval_err
    @test sum(stats.per_individual.eval_exception) == stats.n_eval_exception
    @test sum(stats.per_individual.max_reduce_steps) == stats.n_max_reduce_steps
    @test sum(stats.per_individual.non_const_output) == stats.n_non_const_output
    @test sum(stats.per_individual.missing_const_value) == stats.n_missing_const_value
    @test sum(stats.per_individual.non_closed_lambda) == stats.n_non_closed_lambda
    @test res.history[1].eval_stats.n_eval_err == stats.n_eval_err
end

end # module Regression
