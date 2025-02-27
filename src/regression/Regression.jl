"""
    Regression

Frontier regression runner interfaces.

This module defines the **top-level API** for running population-based
regression over typed lambda-calculus programs. The implementation is
intentionally composable: the startup components construct a typed population,
while callers may replace initialization or any evaluation, selection,
mutation, and crossover component through the same NamedTuple boundary.

Design goals
- High composability: configuration is a `NamedTuple` and components are injected
  as a `NamedTuple`.
- Concurrency readiness: SR/regrssion paths must not depend on global factories
  (e.g. shared `newtyvar` counters or `gensym` naming).
- Const injection: inputs and intermediate values flow through a constant pool
  whose types are visible to individuals.

Status
- `fit!` implements a typed population loop with injected `evaluate` and `loss`.
- `default_components` supplies bounded procedural initialization, native
  FunctionDef evaluation, procedural mutations, and automatic core/procedural
  crossover dispatch. Its regression generation world includes guarded safe
  list elimination for both initialization and generator-backed variation.
- `prepare_default_evaluator` reuses procedural preparation across a batch, and
  `fit!` maps non-finite outputs or losses to the configured error penalty.
- For diagnostics, `fit!` supports `callback(state)`, optional per-generation
  timing through `config.timing.enabled`, and an optional
  `components.init_population(...)` hook.
- Thread-safe primitives for SR paths live in `ASTs.Contexts.EvalContext` and
  `TypeSystem.make_tyvar_factory`.
"""
module Regression

using Random
using TestItems

import Base.Threads

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, TyConst, TyFun, TyList, TyArray, TyNat, FloatTy
using ..TypeSystem: make_tyvar_factory, get_type_ctx
using ..ASTs.Contexts: EvalContext, ConstSymbolFactory, new_constant!
using ..ASTs.NodeASTs: AstTree, AstNode, AppNode, AbsNode, ConstNode, IndexNode, SymbolGenerator
using ..Reduces: beta_reduce!
using ..Procedural
using ..ProceduralMutations
using ..CanonicalGeneration
using ..Mutations
using ..Mutations: Populations
using ..Mutations.Populations: Individual

export Regressor, SupervisedDataset, FitResult, fit!
export default_config, default_components, default_regressor
export PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
export TensorSchema, TensorSeries, SeriesTask, series_task
export series_mse, series_components, series_regressor
export AbstractConstantSearch, ConstantSearchBudget, ConstantSearchStats
export ConstantSearchResult, GaussianConstantSearch, search_constants

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
struct NonFiniteOutputErr end

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

`best` is the best valid individual observed and is `nothing` when
`status == :no_valid_candidate`. `seed` is the actual root seed used by the
run. The compatibility four-argument constructor uses zero when no run seed is
available.
"""
struct FitResult
    best
    status::Symbol
    message::String
    history::Vector{Any}
    seed::UInt64
end

FitResult(best, status::Symbol, message::String, history::Vector{Any}) =
    FitResult(best, status, message, history, UInt64(0))

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

_contains_procedural(tree::AstTree) = any(
    node -> node isa Procedural.ScopeNode || node isa Procedural.FunctionDefNode,
    tree,
)

function _finite_prediction(value)::Bool
    if value isa Real
        return isfinite(value)
    elseif value isa AbstractArray || value isa Tuple
        return all(_finite_prediction, value)
    end
    return false
end

function _finite_real_loss(value)
    value isa Real && isfinite(value) || return nothing
    converted = try
        Float64(value)
    catch
        return nothing
    end
    return isfinite(converted) ? converted : nothing
end

function _known_candidate_exception(issue)
    issue isa MaxReduceStepsErr && return true
    issue isa NonConstOutputErr && return true
    issue isa MissingConstValueErr && return true
    issue isa NonClosedLambdaErr && return true
    issue isa NonFiniteOutputErr && return true
    return issue isa DomainError || issue isa DivideError ||
           issue isa OverflowError || issue isa InexactError
end

_no_candidate_exception(::Any) = false

function _within_bloat_limit(
    tree::AstTree,
    policy::Mutations.BloatPolicy;
    parent_nodes::Union{Nothing,Int}=nothing,
)::Bool
    policy.enabled || return true
    nodes = length(tree)
    nodes <= policy.max_nodes || return false
    parent_nodes === nothing && return true
    return nodes - parent_nodes <= policy.max_nodes_delta
end

function _remove_redundant_global_locals!(local_consts, local_consts_ty, global_consts)
    for name in intersect(Set(keys(local_consts)), Set(keys(global_consts)))
        local_value = local_consts[name]
        global_value = global_consts[name]
        (local_value === global_value || isequal(local_value, global_value)) || continue
        delete!(local_consts, name)
        delete!(local_consts_ty, name)
    end
    return nothing
end

struct PreparedDefaultEvaluator{C<:NamedTuple,K<:NamedTuple}
    individual::Individual
    components::C
    config::K
    procedural::Union{Nothing,Procedural.PreparedProcedural}
end

"""
    prepare_default_evaluator(individual, components, config; copy_tree=true)

Prepare the default evaluator once for repeated samples. The callable result owns
a procedural tree copy by default; `fit!` uses `copy_tree=false` while its current
population is immutable during evaluation.
"""
function prepare_default_evaluator(
    ind::Individual,
    components::NamedTuple,
    config::NamedTuple;
    copy_tree::Bool=true,
)
    prepared = _contains_procedural(ind.tree) ?
        Procedural.prepare_procedural(ind.tree; copy_tree) : nothing
    return PreparedDefaultEvaluator(ind, components, config, prepared)
end

function _default_mutation_driver(tree::AstTree, target_ty::Ty; kwargs...)
    return ProceduralMutations.mutate_procedural(tree, target_ty; kwargs...)
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
- run trees containing `ScopeNode` or `FunctionDefNode` through the native prepared
  procedural evaluator with one shared budget,
- evaluate `(program x1 x2 ...)` by beta-reducing for `config.max_reduce_steps` steps,
- require a `ConstNode` result whose value is in the evaluation context.
"""
function _evaluate_default_prepared(
    prepared::PreparedDefaultEvaluator,
    x,
)
    ind = prepared.individual
    components = prepared.components
    config = prepared.config
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

    if prepared.procedural !== nothing
        arguments = length(input_slots) == 1 ? (x,) : x
        steps = get(
            config,
            :procedural_max_steps,
            max(256, 4 * get(config, :max_reduce_steps, 64)),
        )
        result = Procedural.run_procedural(
            prepared.procedural,
            ctx;
            args=arguments,
            max_steps=steps,
        )
        if result isa Err
            issue = result.value
            issue isa Procedural.ProceduralStepLimitError &&
                return Err(MaxReduceStepsErr(steps))
            issue isa Procedural.ProceduralEvaluationCycleError &&
                return Err(MaxReduceStepsErr(steps))
            issue isa Procedural.ProceduralMissingConstantError &&
                return Err(MissingConstValueErr(issue.name))
            issue isa Procedural.ProceduralEvaluationError &&
                return Err(NonConstOutputErr())
        end
        _finite_prediction(result) || return Err(NonFiniteOutputErr())
        return result
    end

    tree = _apply_inputs(ind.tree, input_slots)
    steps = get(config, :max_reduce_steps, 64)
    normalized = _normalize_with_lift!(tree, ctx, components, config, steps)
    normalized || return Err(MaxReduceStepsErr(steps))

    tree.root isa ConstNode || return Err(NonConstOutputErr())
    name = (tree.root::ConstNode).name
    haskey(ctx, name) || return Err(MissingConstValueErr(name))
    result = ctx[name]
    _finite_prediction(result) || return Err(NonFiniteOutputErr())
    return result
end

(prepared::PreparedDefaultEvaluator)(x) = _evaluate_default_prepared(prepared, x)

function evaluate_default(ind::Individual, x, components::NamedTuple, config::NamedTuple)
    return prepare_default_evaluator(ind, components, config; copy_tree=false)(x)
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

function _tournament_index(
    rng::AbstractRNG,
    fitness::Vector{Float64},
    valid::AbstractVector{Bool},
    tournament_size::Int,
)::Int
    length(fitness) == length(valid) ||
        throw(DimensionMismatch("fitness/valid length mismatch"))
    n = length(fitness)
    best = rand(rng, 1:n)
    best_key = (!valid[best], fitness[best])
    for _ in 2:tournament_size
        j = rand(rng, 1:n)
        key = (!valid[j], fitness[j])
        if key < best_key
            best = j
            best_key = key
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
    role === :select && return 0x06
    role === :immigrant && return 0x07
    return 0xFF
end

function _derive_seed(seed::UInt64, role::Symbol, coordinates::Integer...)::UInt64
    state = _mix64(seed ⊻ _mix64(_role_code(role)))
    for coordinate in coordinates
        state = _mix64(state ⊻ _mix64(UInt64(coordinate)))
    end
    return state
end

function derive_rng(seed::UInt64, role::Symbol, coordinates::Integer...)::MersenneTwister
    s = _derive_seed(seed, role, coordinates...)
    return MersenneTwister(UInt32[UInt32(s & 0xffffffff), UInt32(s >> 32)])
end

function derive_rng(seed::UInt64, generation::Int, index::Int, role::Symbol)::MersenneTwister
    return derive_rng(seed, role, generation, index)
end

function _sorted_indices_by_fitness(fitness::Vector{Float64})::Vector{Int}
    return sortperm(fitness)
end

function _sorted_indices_by_fitness(
    fitness::Vector{Float64},
    valid::AbstractVector{Bool},
)::Vector{Int}
    length(fitness) == length(valid) ||
        throw(DimensionMismatch("fitness/valid length mismatch"))
    return sortperm(eachindex(fitness); by=i -> (!valid[i], fitness[i], i))
end

function _preflight_input_signature(target_ty::Ty, slots, input_types)
    current = target_ty
    for slot in slots
        current = TypeSystem.prune(current)
        current isa TypeSystem.TyFun ||
            return ArgumentError("target_ty does not accept all configured input slots")
        TypeSystem.iscompatible(current.arg, input_types[slot]) ||
            return ArgumentError("target_ty argument for input slot :$slot is incompatible with input_types")
        current = current.res
    end
    return nothing
end

function _preflight_regression_config(
    reg::Regressor,
    data::SupervisedDataset,
    components::NamedTuple,
    config::NamedTuple,
    N;
    require_input::Bool,
)
    isempty(data.X) && return ArgumentError("dataset must not be empty")
    slots = components.input_slots
    slots isa AbstractVector{Symbol} ||
        return ArgumentError("input_slots must be a vector of symbols")
    components.input_types isa AbstractDict{Symbol,<:TyLike} ||
        return ArgumentError("input_types must map symbols to types")
    require_input && isempty(slots) &&
        return ArgumentError("input_slots must be non-empty for evaluate_default")
    length(unique(slots)) == length(slots) ||
        return ArgumentError("input_slots must not contain duplicates")
    Set(keys(components.input_types)) == Set(slots) ||
        return ArgumentError("input_types keys must exactly match input_slots")
    signature_error = _preflight_input_signature(reg.target_ty, slots, components.input_types)
    signature_error === nothing || return signature_error
    if length(slots) > 1
        for (j, x) in enumerate(data.X)
            x isa Tuple || return ArgumentError("multi-input sample $j must be a Tuple")
            length(x) == length(slots) ||
                return ArgumentError("multi-input sample $j has wrong arity")
        end
    end

    N isa Int && N > 0 || return ArgumentError("population_size must be a positive integer")
    generations = get(config, :generations, 0)
    tournament_size = get(config, :tournament_size, 0)
    elitism = get(config, :elitism, -1)
    generations isa Int && generations > 0 ||
        return ArgumentError("generations must be a positive integer")
    tournament_size isa Int && tournament_size > 0 ||
        return ArgumentError("tournament_size must be a positive integer")
    elitism isa Int && 0 <= elitism <= N ||
        return ArgumentError("elitism must be an integer in 0:population_size")
    for (name, rate) in (
        (:mutation_rate, get(config, :mutation_rate, NaN)),
        (:crossover_rate, get(config, :crossover_rate, NaN)),
    )
        rate isa Real && isfinite(rate) && 0 <= rate <= 1 ||
            return ArgumentError("$name must be finite and in [0, 1]")
    end
    penalty = get(config, :error_penalty, NaN)
    penalty isa Real && isfinite(penalty) && penalty > 0 ||
        return ArgumentError("error_penalty must be finite and positive")
    threshold = get(config, :error_threshold, NaN)
    threshold isa Real && isfinite(threshold) ||
        return ArgumentError("error_threshold must be finite")
    for name in (:max_reduce_steps, :procedural_max_steps, :closure_max_reduce_steps)
        value = get(config, name, 0)
        value isa Int && value > 0 ||
            return ArgumentError("$name must be a positive integer")
    end
    lift_limit = get(config, :closure_lift_max_per_pass, -1)
    lift_limit isa Int && lift_limit >= 0 ||
        return ArgumentError("closure_lift_max_per_pass must be a non-negative integer")
    get(config, :strict_errors, false) isa Bool ||
        return ArgumentError("strict_errors must be Bool")
    if haskey(config, :seed)
        raw_seed = config.seed
        raw_seed isa Integer && 0 <= raw_seed <= typemax(UInt64) ||
            return ArgumentError("seed must be a non-negative integer representable as UInt64")
    end
    repro = get(config, :repro, (;))
    max_resample = get(repro, :max_resample, 0)
    immigrant_max_try = get(repro, :immigrant_max_try, 0)
    immigrants_per_gen = get(repro, :immigrants_per_gen, 0)
    capability_mutation_max_try = get(repro, :capability_mutation_max_try, 1)
    capability_max_resample = get(repro, :capability_max_resample, 0)
    capability_elites = get(repro, :capability_elites, (;))
    capability_offspring = get(repro, :capability_offspring, (;))
    max_resample isa Int && max_resample >= 0 ||
        return ArgumentError("repro.max_resample must be a non-negative integer")
    immigrant_max_try isa Int && immigrant_max_try > 0 ||
        return ArgumentError("repro.immigrant_max_try must be a positive integer")
    immigrants_per_gen isa Int && immigrants_per_gen >= 0 ||
        return ArgumentError("repro.immigrants_per_gen must be a non-negative integer")
    capability_mutation_max_try isa Int && capability_mutation_max_try > 0 ||
        return ArgumentError(
            "repro.capability_mutation_max_try must be a positive integer",
        )
    capability_max_resample isa Int && capability_max_resample >= 0 ||
        return ArgumentError(
            "repro.capability_max_resample must be a non-negative integer",
        )
    capability_total = 0
    for (name, requests) in (
        (:capability_elites, capability_elites),
        (:capability_offspring, capability_offspring),
    )
        requests isa Union{NamedTuple,AbstractDict} || return ArgumentError(
            "repro.$name must be a NamedTuple or dictionary",
        )
        for (tag, count_) in pairs(requests)
            tag isa Symbol || return ArgumentError("repro.$name tags must be symbols")
            count_ isa Int && count_ >= 0 || return ArgumentError(
                "repro.$name counts must be non-negative integers",
            )
            count_ <= N || return ArgumentError(
                "a repro.$name count must not exceed population_size",
            )
            capability_total += count_
        end
    end
    get(repro, :require_change, false) isa Bool ||
        return ArgumentError("repro.require_change must be Bool")
    get(repro, :on_fail, :clone) in (:clone, :immigrant) ||
        return ArgumentError("repro.on_fail must be :clone or :immigrant")
    for section_name in (:stats, :log)
        section = get(config, section_name, (;))
        get(section, :enabled, false) isa Bool ||
            return ArgumentError("$section_name.enabled must be Bool")
        every = get(section, :every, 1)
        every isa Int && every > 0 ||
            return ArgumentError("$section_name.every must be a positive integer")
    end
    timing = get(config, :timing, (; enabled=false))
    get(timing, :enabled, false) isa Bool ||
        return ArgumentError("timing.enabled must be Bool")
    constant_search = get(config, :constant_search, (; enabled=false))
    get(constant_search, :enabled, false) isa Bool ||
        return ArgumentError("constant_search.enabled must be Bool")
    conditional_share = get(constant_search, :conditional_share, 0.4)
    conditional_share isa Real && isfinite(conditional_share) &&
        0 <= conditional_share <= 1 || return ArgumentError(
            "constant_search.conditional_share must be finite and in [0, 1]",
        )
    max_proposals = get(constant_search, :max_proposals, 16)
    max_objective_evals = get(constant_search, :max_objective_evals, 17)
    max_proposals isa Int && max_proposals >= 0 || return ArgumentError(
        "constant_search.max_proposals must be a non-negative integer",
    )
    max_objective_evals isa Int && max_objective_evals >= 1 || return ArgumentError(
        "constant_search.max_objective_evals must be a positive integer",
    )
    if get(constant_search, :enabled, false)
        haskey(components, :constant_search) || return ArgumentError(
            "constant_search component is required when constant search is enabled",
        )
        components.constant_search isa AbstractConstantSearch || return ArgumentError(
            "constant_search component must be an AbstractConstantSearch",
        )
    end
    parallel = get(config, :parallel, (; enabled=false))
    get(parallel, :enabled, false) isa Bool ||
        return ArgumentError("parallel.enabled must be Bool")

    components.mutation_max_try isa Int && components.mutation_max_try > 0 ||
        return ArgumentError("mutation_max_try must be a positive integer")
    components.mutation_prune_consts isa Bool ||
        return ArgumentError("mutation_prune_consts must be Bool")
    components.mutations isa AbstractVector ||
        return ArgumentError("mutations must be a vector")
    candidate_failure = get(components, :candidate_failure, _known_candidate_exception)
    applicable(candidate_failure, MaxReduceStepsErr(1)) ||
        return ArgumentError("candidate_failure must accept one failure value")
    candidate_failure(MaxReduceStepsErr(1)) isa Bool ||
        return ArgumentError("candidate_failure must return Bool")
    loss_failure = get(components, :loss_failure, _no_candidate_exception)
    applicable(loss_failure, DomainError(0)) ||
        return ArgumentError("loss_failure must accept one failure value")
    loss_failure(DomainError(0)) isa Bool ||
        return ArgumentError("loss_failure must return Bool")
    get(components, :evaluation_threadsafe, false) isa Bool ||
        return ArgumentError("evaluation_threadsafe must be Bool")
    bloat = get(components, :bloat_policy, Mutations.BloatPolicy())
    bloat isa Mutations.BloatPolicy ||
        return ArgumentError("bloat_policy must be a BloatPolicy")
    if capability_total > 0
        haskey(components, :capability_tags) ||
            return ArgumentError(
                "capability_tags component is required by capability reproduction",
            )
    end
    Set(keys(components.global_consts)) == Set(keys(components.global_consts_ty)) ||
        return ArgumentError("global_consts and global_consts_ty keys must match")
    isempty(intersect(Set(keys(components.global_consts)), Set(slots))) ||
        return ArgumentError("global constants and input slots must use disjoint names")
    return nothing
end

function _validate_individual_contract(
    ind,
    reg::Regressor,
    components::NamedTuple,
    label::AbstractString,
)
    ind isa Individual || return ArgumentError("$label is not an Individual")
    TypeSystem.iscompatible(ind.target_ty, reg.target_ty) ||
        return ArgumentError("$label has incompatible target type")
    Set(keys(ind.local_consts)) == Set(keys(ind.local_consts_ty)) ||
        return ArgumentError("$label local value/type keys differ")
    reserved = union(Set(keys(components.global_consts)), Set(components.input_slots))
    isempty(intersect(Set(keys(ind.local_consts)), reserved)) ||
        return ArgumentError("$label local constants collide with global or input names")
    for node in ind.tree
        node isa ConstNode || continue
        name = node.name
        haskey(components.global_consts, name) ||
            haskey(ind.local_consts, name) ||
            name in components.input_slots ||
            return ArgumentError("$label references unresolved constant :$name")
    end
    ctx = merge(components.global_consts_ty, components.input_types, ind.local_consts_ty)
    factory = components.tyfactory_ctor()
    meta = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    checker = get(components, :type_inference, nothing)
    inferred = checker === nothing ?
        TypeSystem.infer(ind.tree, ctx, factory, meta) :
        checker(ind.tree, ctx, factory, meta)
    inferred isa Err &&
        return ArgumentError("$label type inference returned Err($(repr(inferred.value)))")
    inferred isa TyLike ||
        return ArgumentError("$label type inference returned a non-type")
    TypeSystem.iscompatible(inferred, reg.target_ty) ||
        return ArgumentError("$label inferred type is incompatible with target")
    return nothing
end

function _preflight_initial_population(pop, reg::Regressor, components::NamedTuple, N::Int)
    pop isa AbstractVector || return ArgumentError("init_population must return a vector")
    length(pop) == N ||
        return ArgumentError("init_population must return exactly N individuals")
    policy = get(components, :bloat_policy, Mutations.BloatPolicy())
    for (i, ind) in enumerate(pop)
        issue = _validate_individual_contract(ind, reg, components, "initial individual $i")
        issue === nothing || return issue
        _within_bloat_limit(ind.tree, policy) ||
            return ArgumentError("initial individual $i exceeds the configured structural limit")
    end
    return nothing
end

"""
    default_config(; kwargs...) -> NamedTuple

Load a bounded active-development startup configuration. This function makes
strong assumptions and is intended for quick integrated runs, not as a stable
stable general-purpose configuration.

Set `timing=(enabled=true,)` to add `timing_stats` to callback state and each
history entry. The record separates initialization, population evaluation, and
reproduction, reports crossover and mutation time inside reproduction plus the
unclassified remainder, and records the effective evaluation thread count.

`repro.capability_elites` reserves bounded best-valid survivor quotas by
observable tag. The startup default keeps one `:function_def` and one
`:self_call` representative when available; quotas never manufacture a missing
capability and are clipped by the population capacity. Override
`components.capability_tags` to define another task-independent tagger, or set
`capability_elites=(;)` for ordinary fitness-only elitism.

`repro.capability_offspring` reserves mutation-only offspring slots from the
best valid tagged parent. The startup default gives `:self_call` one slot when
that niche exists; sequence profiles may additionally
request a `:function_def` slot to bootstrap recursion from a finite definition.
Set it to `(;)` for tournament-only parent selection.
`repro.capability_mutation_max_try` bounds the
inner typed-mutation attempts for each such slot independently of ordinary
reproduction; the startup default is one. `capability_max_resample` separately
bounds outer retries and defaults to zero, after which the valid tagged parent
is retained.
"""
function default_config(;
    population_size::Int=64,
    generations::Int=20,
    error_threshold::Float64=1e-6,
    max_reduce_steps::Int=64,
    procedural_max_steps::Int=max(256, 4 * max_reduce_steps),
    closure_max_reduce_steps::Int=max_reduce_steps,
    closure_lift_max_per_pass::Int=64,
    mutation_rate::Float64=0.65,
    crossover_rate::Float64=0.5,
    elitism::Int=2,
    tournament_size::Int=3,
    error_penalty::Float64=1e12,
    strict_errors::Bool=false,
    repro=(;
        require_change=true,
        max_resample=10,
        on_fail=:immigrant,              # :immigrant or :clone
        immigrant_max_try=3,
        immigrants_per_gen=0,
        capability_mutation_max_try=1,
        capability_max_resample=0,
        capability_elites=(; function_def=1, self_call=1),
        capability_offspring=(; self_call=1),
    ),
    stats=(;
        enabled=false,
        mutation=false,
        reproduction=false,
        print=false,
        every=1,
        topk_types=8,
    ),
    timing=(; enabled=false),
    constant_search=(;
        enabled=false,
        conditional_share=0.4,
        max_proposals=16,
        max_objective_evals=17,
    ),
    log=(; enabled=false, every=1, topk=5, io=stdout),
    parallel=(; enabled=false, mode=:individual, schedule=:dynamic),
    kwargs...
)
    repro = merge((;
        require_change=true,
        max_resample=10,
        on_fail=:immigrant,
        immigrant_max_try=3,
        immigrants_per_gen=0,
        capability_mutation_max_try=1,
        capability_max_resample=0,
        capability_elites=(; function_def=1, self_call=1),
        capability_offspring=(; self_call=1),
    ), repro)
    stats = merge((;
        enabled=false,
        mutation=false,
        reproduction=false,
        print=false,
        every=1,
        topk_types=8,
    ), stats)
    timing = merge((; enabled=false), timing)
    constant_search = merge((;
        enabled=false,
        conditional_share=0.4,
        max_proposals=16,
        max_objective_evals=17,
    ), constant_search)
    log = merge((; enabled=false, every=1, topk=5, io=stdout), log)
    parallel = merge((; enabled=false, mode=:individual, schedule=:dynamic), parallel)
    base = (;
        population_size,
        generations,
        error_threshold,
        max_reduce_steps,
        procedural_max_steps,
        closure_max_reduce_steps,
        closure_lift_max_per_pass,
        mutation_rate,
        crossover_rate,
        elitism,
        tournament_size,
        error_penalty,
        strict_errors,
        repro,
        stats,
        timing,
        constant_search,
        log,
        parallel,
    )
    return merge(base, (;kwargs...))
end

function _merge_config_overrides(config::NamedTuple, overrides::NamedTuple)
    result = merge(config, overrides)
    for key in (:repro, :stats, :timing, :constant_search, :log, :parallel)
        haskey(overrides, key) || continue
        section = merge(get(config, key, (;)), get(overrides, key, (;)))
        result = merge(result, NamedTuple{(key,)}((section,)))
    end
    return result
end

function _has_binary_global_application(ind::Individual)
    return any(ind.tree) do node
        node isa AppNode || return false
        node.func isa AppNode || return false
        return node.func.func isa ConstNode
    end
end

function _procedural_capability_tags(ind::Individual)
    tags = Symbol[]
    observations = Procedural.functiondef_recursion_observations(ind.tree)
    if !isempty(observations)
        push!(tags, :function_def)
        any(observation -> observation.self_calls > 0, observations) &&
            push!(tags, :self_call)
    end
    _has_binary_global_application(ind) &&
        push!(tags, :binary_global_application)
    return Tuple(tags)
end

_procedural_candidate_admissibility(ind::Individual) =
    Procedural.obvious_functiondef_loop_reason(ind.tree)

function _individual_candidate_rejection(ind::Individual, components::NamedTuple)
    filter = get(components, :candidate_admissibility, _ -> nothing)
    applicable(filter, ind) || throw(ArgumentError(
        "candidate_admissibility must accept one Individual",
    ))
    reason = filter(ind)
    reason === nothing && return nothing
    reason isa Symbol || throw(ArgumentError(
        "candidate_admissibility must return nothing or a Symbol reason",
    ))
    return reason
end

function _individual_capability_tags(ind::Individual, components::NamedTuple)
    tagger = get(components, :capability_tags, _ -> ())
    applicable(tagger, ind) || throw(ArgumentError(
        "capability_tags must accept one Individual",
    ))
    raw = tagger(ind)
    raw === nothing && return ()
    raw isa Symbol && return (raw,)
    tags = Symbol[]
    try
        for tag in raw
            tag isa Symbol || throw(ArgumentError(
                "capability_tags must return symbols",
            ))
            tag in tags || push!(tags, tag)
        end
    catch issue
        issue isa ArgumentError && rethrow()
        throw(ArgumentError("capability_tags must return an iterable of symbols"))
    end
    return Tuple(tags)
end

function _procedural_capability_mutation_chooser(
    tag::Symbol,
    mutations::AbstractVector{<:Mutations.AbstractMutation},
    ordinary::Mutations.AbstractPairChooser,
)
    ordinary isa Mutations.WeightedPairChooser || return ordinary
    tag in (:function_def, :self_call, :binary_global_application) || return ordinary
    weights = copy(ordinary.weights)
    if tag in (:self_call, :binary_global_application)
        weights[Mutations.UnaryCompositionMutation] = 100.0
        weights[Mutations.BinaryCompositionMutation] = 4.0
        weights[Mutations.ContextualExpansionMutation] = 2.0
    end
    for mutation in mutations
        mutation isa ProceduralMutations.FunctionDefMutation || continue
        mode = ProceduralMutations._mode(mutation)
        if tag === :function_def && mode === :recursive_body
            weights[typeof(mutation)] = 8.0
        elseif tag in (:function_def, :self_call) && mode === :remove
            weights[typeof(mutation)] = 0.0
        end
    end
    return Mutations.WeightedPairChooser(weights, ordinary.cache_eligibility)
end

struct _StartupFunctionDefMutationGenerator end

function _guarded_list_map_scaffold(target_ty::Ty, ctx::NamedTuple)
    target = TypeSystem.prune(target_ty)
    target isa TyFun || return nothing
    input_ty = TypeSystem.prune(target.arg)
    output_ty = TypeSystem.prune(target.res)
    input_ty isa TyList && output_ty isa TyList || return nothing
    TypeSystem.iscompatible(input_ty.elem, output_ty.elem) &&
        TypeSystem.iscompatible(output_ty.elem, input_ty.elem) || return nothing
    required = (:UNCONS_CASE, :CONS, :NIL)
    all(name -> haskey(ctx.global_consts, name), required) || return nothing
    all(name -> haskey(ctx.global_consts_ty, name), required) || return nothing

    on_item = AbsNode(
        :head,
        AbsNode(
            :tail,
            AppNode(
                AppNode(ConstNode(:CONS), IndexNode(1)),
                AppNode(IndexNode(3), IndexNode(0)),
            ),
        ),
    )
    body = AbsNode(
        :series,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                on_item,
            ),
            ConstNode(:NIL),
        ),
    )
    return CanonicalGeneration.GenerationResult(
        AstTree(Procedural.FunctionDefNode(body)),
        target_ty,
        Dict{Symbol,Any}(),
        Dict{Symbol,TyLike}(),
        Dict{Symbol,Any}(:startup_scaffold => :guarded_list_map),
    )
end

function _guarded_list_real_scaffold(target_ty::Ty, ctx::NamedTuple)
    target = TypeSystem.prune(target_ty)
    target isa TyFun || return nothing
    TypeSystem.prune(target.arg) isa TyList || return nothing
    TypeSystem.prune(target.res) == FloatTy || return nothing
    required = (:UNCONS_CASE, :ZERO)
    all(name -> haskey(ctx.global_consts, name), required) || return nothing
    all(name -> haskey(ctx.global_consts_ty, name), required) || return nothing

    on_item = AbsNode(
        :head,
        AbsNode(
            :tail,
            AppNode(IndexNode(3), IndexNode(0)),
        ),
    )
    body = AbsNode(
        :series,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                on_item,
            ),
            ConstNode(:ZERO),
        ),
    )
    return CanonicalGeneration.GenerationResult(
        AstTree(Procedural.FunctionDefNode(body)),
        target_ty,
        Dict{Symbol,Any}(),
        Dict{Symbol,TyLike}(),
        Dict{Symbol,Any}(:startup_scaffold => :guarded_list_real),
    )
end

function (::_StartupFunctionDefMutationGenerator)(
    mode::Symbol,
    target_ty::Ty,
    scope,
    env,
    ctx,
    functiondef_provider,
    generation_kwargs,
)
    if mode === :recursive_body
        scaffold = _guarded_list_map_scaffold(target_ty, ctx)
        scaffold === nothing &&
            (scaffold = _guarded_list_real_scaffold(target_ty, ctx))
        scaffold === nothing || return scaffold
    end
    return ProceduralMutations.generate_functiondef_replacement(
        ProceduralMutations.DefaultFunctionDefMutationGenerator(),
        mode,
        target_ty,
        scope,
        env,
        ctx,
        functiondef_provider,
        generation_kwargs,
    )
end

"""
    default_components(; kwargs...) -> NamedTuple

Load default component wiring for regression, including self-contained typed
population initialization and procedural first-class dispatch.

`generation_providers` is the common provider world used by initialization,
ordinary generator-backed replacement/expansion, and FunctionDef body
generation. Its default extends the finite canonical core with guarded safe
list elimination; replacing it changes all of those routes together.
`contextual_expansion_site_selection` and `binary_composition_site_selection`
independently control where the two composition families preserve an existing
subexpression. `capability_tags` classifies individuals for optional bounded
capability elites; the default observes FunctionDef presence and genuine self
calls, plus the generic presence of a two-argument global application, without
recognizing a particular target or primitive name. `functiondef_provider` is shared
by initialization and FunctionDef body mutations; the startup provider applies
the cheap `DirectKnotFilter` while leaving general termination to evaluator
budgets. `candidate_admissibility` applies the matching cheap post-variation
AST-level subset and returns an inspectable Symbol reason or `nothing`; it is
not part of typing and does not claim general termination.
`functiondef_recursive_body_site_weight` changes only the relative mass of the
recursive-body mode after a FunctionDef site already exists.
`capability_mutation_chooser` receives `(tag, mutations, ordinary_chooser)` for
the bounded capability-offspring slots. The startup policy favors recursive-
body construction for a finite FunctionDef niche and direct unary/binary growth
for an existing self-call niche; ordinary reproduction keeps the caller's
chooser unchanged.
For `List T → List T`, the startup recursive-body generator first builds a
guarded identity-map scaffold when `UNCONS_CASE`, `CONS`, and `NIL` are present.
For `List T → Float`, it can build the smaller guarded tail-recursion scaffold
when `UNCONS_CASE` and `ZERO` are present. These supply finite productive
recursion shapes without choosing the head transformation; later typed
mutations still own that search. Other targets use the ordinary injected
FunctionDef generator unchanged.
Initialization growth routes the already configured unary composition, binary
composition, and contextual expansion operators. Each accepted edit must grow
the term and may not move farther from its slot's target node count; equal-
distance growth is allowed because odd-sized typed compositions can straddle an
even target size.

This is a **loader only**: it does not run any business logic. The returned
components are designed to be easy to override via `merge`.
`constant_search` defaults to the structured Gaussian adapter, but `fit!` uses
it only when `config.constant_search.enabled=true`.
"""
function default_components(;
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    input_slots::Vector{Symbol}=Symbol[],
    input_types::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    functiondef_generation_weight::Float64=1.0,
    generation_providers=(
        CanonicalGeneration.default_head_providers()...,
        CanonicalGeneration.SafeListAccessProvider(),
    ),
    contextual_expansion_site_selection::Mutations.AbstractSiteSelectionPolicy=
        Mutations.ExponentialSiteSelectionPolicy(),
    binary_composition_site_selection::Mutations.AbstractSiteSelectionPolicy=
        Mutations.LocalCallSiteSelectionPolicy(
            Mutations.ExponentialSiteSelectionPolicy(scope_bias=0.0);
            multiplier=32.0,
        ),
    capability_tags=_procedural_capability_tags,
    capability_mutation_chooser=_procedural_capability_mutation_chooser,
    candidate_admissibility=_procedural_candidate_admissibility,
    functiondef_provider=Procedural.FunctionDefProvider(
        admissibility=Procedural.DirectKnotFilter(),
    ),
    functiondef_recursive_body_site_weight::Float64=4.0,
    mutations::Vector{<:Mutations.AbstractMutation}=[
        Mutations.ParameterIntroductionMutation(),
        Mutations.ScalarConstMutation(),
        Mutations.BoolConstMutation(),
        Mutations.ArrayConstMutation(),
        Mutations.PointMutation(),
        Mutations.CSEMutation(),
        Mutations.BetaReduceMutation(),
        Mutations.EtaReduceMutation(),
        Mutations.LetHoistMutation(),
        Mutations.LetSinkMutation(),
        Mutations.LetPruneMutation(),
        Mutations.SourceBindingInsertionMutation(),
        Mutations.SubtreeReplacementMutation(;
            replacement_generator=Mutations.ScopedReplacementGenerator(
                Procedural.generate_procedural_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(
                    default=1,
                    function_def=functiondef_generation_weight,
                ),
                sampling_depth_limit=3,
                sampling_node_budget=48,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.UnaryCompositionMutation(),
        Mutations.ContextualExpansionMutation(;
            site_selection=contextual_expansion_site_selection,
            replacement_generator=Mutations.ScopedReplacementGenerator(
                Procedural.generate_procedural_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(
                    default=1,
                    function_def=functiondef_generation_weight,
                ),
                sampling_depth_limit=3,
                sampling_node_budget=48,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.BinaryCompositionMutation(;
            site_selection=binary_composition_site_selection,
            replacement_generator=Mutations.ScopedReplacementGenerator(
                Procedural.generate_procedural_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(
                    default=1,
                    function_def=functiondef_generation_weight,
                ),
                sampling_depth_limit=3,
                sampling_node_budget=48,
                guarded_expansion_depth=1,
            ),
        ),
        ProceduralMutations.FunctionDefMutation(
            :finite_body;
            functiondef_provider,
            generation_kwargs=(;
                providers=generation_providers,
                sampling_depth_limit=4,
                sampling_node_budget=64,
                guarded_expansion_depth=1,
            ),
        ),
        ProceduralMutations.FunctionDefMutation(
            :recursive_body;
            functiondef_provider,
            replacement_generator=_StartupFunctionDefMutationGenerator(),
            site_weight=(mode, site, env, ctx) ->
                functiondef_recursive_body_site_weight,
            generation_kwargs=(;
                providers=generation_providers,
                sampling_depth_limit=8,
                sampling_node_budget=96,
                guarded_expansion_depth=1,
            ),
        ),
        ProceduralMutations.FunctionDefMutation(
            :remove;
            functiondef_provider,
            generation_kwargs=(;
                providers=generation_providers,
                sampling_depth_limit=4,
                sampling_node_budget=64,
                guarded_expansion_depth=1,
            ),
        ),
    ],
    mutation_chooser::Mutations.AbstractPairChooser=Mutations.WeightedPairChooser(),
    mutation_provider::Mutations.AbstractSiteProvider=Mutations.IndexedProvider(),
    mutation_max_try::Int=25,
    mutation_prune_consts::Bool=true,
    mutation_bloat_policy::Mutations.BloatPolicy=Mutations.BloatPolicy(),
    mutation_adaptive_policy::Mutations.AdaptivePolicy=Mutations.AdaptivePolicy(),
    bloat_policy::Mutations.BloatPolicy=mutation_bloat_policy,
    type_inference=(tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta),
    candidate_failure=_known_candidate_exception,
    loss_failure=_no_candidate_exception,
    evaluation_threadsafe::Bool=true,
    crossover_op=ProceduralMutations.AutoTypedCrossover(),
    constant_search=nothing,
    mutation_driver=_default_mutation_driver,
    generation_config_factory=rng -> CanonicalGeneration.GenerationConfig(;
        rng,
        providers=generation_providers,
        weights=CanonicalGeneration.GenerationWeights(
            default=1,
            function_def=functiondef_generation_weight,
        ),
        sampling_depth_limit=8,
        sampling_node_budget=96,
        guarded_expansion_depth=1,
    ),
    force_lambda_root_probability::Float64=0.5,
    init_candidates_per_individual::Int=2,
    init_target_nodes::Tuple{Vararg{Int}}=(3, 5, 8, 12),
    init_max_try::Int=8,
    init_growth_attempts::Int=2,
    init_growth_max_try::Int=8,
    init_growth_mutation=nothing,
    init_population=nothing,
    make_immigrant=nothing,
    kwargs...
)
    global_consts_ty === nothing && (global_consts_ty = get_type_ctx(global_consts))
    isfinite(functiondef_generation_weight) && functiondef_generation_weight >= 0 ||
        throw(ArgumentError(
            "functiondef_generation_weight must be finite and non-negative",
        ))
    isfinite(functiondef_recursive_body_site_weight) &&
        functiondef_recursive_body_site_weight >= 0 || throw(ArgumentError(
            "functiondef_recursive_body_site_weight must be finite and non-negative",
        ))
    init_max_try >= 1 || throw(ArgumentError("init_max_try must be positive"))
    init_candidates_per_individual >= 1 || throw(ArgumentError(
        "init_candidates_per_individual must be positive",
    ))
    !isempty(init_target_nodes) && all(>(0), init_target_nodes) || throw(ArgumentError(
        "init_target_nodes must contain positive sizes",
    ))
    init_growth_attempts >= 0 || throw(ArgumentError(
        "init_growth_attempts must be non-negative",
    ))
    init_growth_max_try >= 1 || throw(ArgumentError(
        "init_growth_max_try must be positive",
    ))
    0 <= force_lambda_root_probability <= 1 || throw(ArgumentError(
        "force_lambda_root_probability must be in [0, 1]",
    ))

    init_growth_mutations = if init_growth_mutation === nothing
        Mutations.AbstractMutation[
            mutation for mutation in mutations
            if mutation isa Union{
                Mutations.UnaryCompositionMutation,
                Mutations.ContextualExpansionMutation,
                Mutations.BinaryCompositionMutation,
            }
        ]
    elseif init_growth_mutation isa Mutations.AbstractMutation
        Mutations.AbstractMutation[init_growth_mutation]
    elseif init_growth_mutation isa AbstractVector &&
            all(mutation -> mutation isa Mutations.AbstractMutation, init_growth_mutation)
        Mutations.AbstractMutation[init_growth_mutation...]
    else
        throw(ArgumentError(
            "init_growth_mutation must be an AbstractMutation, a vector of " *
            "AbstractMutation, or nothing",
        ))
    end
    init_growth_mutation = init_growth_mutations
    init_growth_chooser = Mutations.WeightedPairChooser()

    function grow_initial_candidate(
        result::CanonicalGeneration.GenerationResult,
        target_ty::Ty,
        seed::UInt64,
        slot::Int,
        attempt_offset::Int,
        target_nodes::Int,
    )
        tree = result.tree
        local_consts = copy(result.local_consts)
        local_consts_ty = copy(result.local_const_types)
        _remove_redundant_global_locals!(local_consts, local_consts_ty, global_consts)
        initial_nodes = _node_count(tree)
        accepted = 0

        if !isempty(init_growth_mutations)
            for attempt in 1:init_growth_attempts
                _node_count(tree) == target_nodes && break
                old_nodes = _node_count(tree)
                old_distance = abs(old_nodes - target_nodes)
                best = nothing
                best_distance = typemax(Int)
                best_nodes = typemax(Int)
                for (growth_index, growth_mutation) in pairs(init_growth_mutations)
                    attempt_consts = copy(local_consts)
                    attempt_const_types = copy(local_consts_ty)
                    rng = derive_rng(
                        seed,
                        :init_mutate,
                        slot,
                        attempt_offset,
                        attempt,
                        growth_index,
                    )
                    mutated = mutation_driver(
                        tree,
                        target_ty;
                        rng,
                        mutations=Mutations.AbstractMutation[growth_mutation],
                        chooser=init_growth_chooser,
                        provider=Mutations.IndexedProvider(),
                        max_try=init_growth_max_try,
                        prune_consts=mutation_prune_consts,
                        copy_tree=true,
                        copy_ctx=true,
                        global_consts,
                        global_consts_ty=merge(global_consts_ty, input_types),
                        local_consts=attempt_consts,
                        local_consts_ty=attempt_const_types,
                        tyfactory=tyfactory_ctor(),
                        symfactory=SymbolGenerator(),
                        const_symfactory=const_symfactory_ctor(),
                        bloat_policy,
                        adaptive_policy=mutation_adaptive_policy,
                        type_inference,
                    )
                    mutated isa Err && continue
                    new_nodes = _node_count(mutated)
                    new_nodes > old_nodes || continue
                    new_distance = abs(new_nodes - target_nodes)
                    new_distance <= old_distance || continue
                    _within_bloat_limit(
                        mutated,
                        bloat_policy;
                        parent_nodes=old_nodes,
                    ) || continue
                    if new_distance < best_distance ||
                       (new_distance == best_distance && new_nodes < best_nodes)
                        best = (mutated, attempt_consts, attempt_const_types)
                        best_distance = new_distance
                        best_nodes = new_nodes
                    end
                end
                best === nothing && continue
                tree, local_consts, local_consts_ty = best
                _remove_redundant_global_locals!(
                    local_consts,
                    local_consts_ty,
                    global_consts,
                )
                accepted += 1
            end
        end

        result.metadata[:init_growth_attempts] = init_growth_attempts
        result.metadata[:init_growth_accepted] = accepted
        result.metadata[:init_initial_nodes] = initial_nodes
        result.metadata[:init_final_nodes] = _node_count(tree)
        return tree, local_consts, local_consts_ty
    end

    # Thread-safe evaluation context constructor:
    # writes go to `overlay` only; `globals` is assumed read-only.
    make_eval_context = (;
        overlay::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        locals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        globals::Dict{Symbol,Any}=global_consts,
        symfactory::ConstSymbolFactory=const_symfactory_ctor(),
    ) -> EvalContext(overlay=overlay, locals=locals, globals=globals, symfactory=symfactory)

    function generate_lambda_root(target_ty::Ty, rng, config)
        argument_types = Ty[]
        result_type = TypeSystem.prune(target_ty)
        while result_type isa TyFun
            push!(argument_types, result_type.arg)
            result_type = TypeSystem.prune(result_type.res)
        end
        isempty(argument_types) && return nothing
        result = Procedural.generate_procedural_scoped(
            result_type;
            scope_tys=reverse(argument_types),
            global_consts,
            global_consts_ty,
            functiondef_provider,
            config,
        )
        result isa Err && return result
        root = result.tree.root
        for index in Iterators.reverse(eachindex(argument_types))
            root = AbsNode(Symbol("x", index), root)
        end
        result.metadata[:forced_lambda_root] = true
        return CanonicalGeneration.GenerationResult(
            AstTree(root),
            target_ty,
            result.local_consts,
            result.local_const_types,
            result.metadata,
        )
    end

    function generate_individual(target_ty, seed::UInt64, slot::Int, attempt_offset::Int=0)
        last_error = nothing
        candidates = CanonicalGeneration.GenerationResult[]
        for attempt in 1:init_max_try
            rng = derive_rng(
                seed,
                0,
                (slot << 8) + attempt_offset + attempt,
                :init_tree,
            )
            config = generation_config_factory(rng)
            config isa CanonicalGeneration.GenerationConfig || throw(ArgumentError(
                "generation_config_factory must return GenerationConfig",
            ))
            result = if TypeSystem.prune(target_ty) isa TypeSystem.TyFun &&
                        rand(rng) < force_lambda_root_probability
                generate_lambda_root(target_ty, rng, config)
            else
                Procedural.generate_procedural(
                    target_ty;
                    global_consts,
                    global_consts_ty,
                    functiondef_provider,
                    config,
                )
            end
            if result isa Err
                last_error = result
                continue
            end
            result isa CanonicalGeneration.GenerationResult || throw(ArgumentError(
                "procedural generator must return GenerationResult or Err",
            ))
            push!(candidates, result)
            length(candidates) >= init_candidates_per_individual && break
        end
        isempty(candidates) && return last_error === nothing ?
            Err(Mutations.MaxTryReachedErr()) : last_error
        target_nodes = init_target_nodes[mod1(slot, length(init_target_nodes))]
        candidate_sizes = _node_count.(getfield.(candidates, :tree))
        selected_index = argmin([
            (abs(size - target_nodes), -size)
            for size in candidate_sizes
        ])
        result = candidates[selected_index]
        result.metadata[:init_target_nodes] = target_nodes
        result.metadata[:init_candidate_count] = length(candidates)
        tree, local_consts, local_consts_ty = grow_initial_candidate(
            result,
            target_ty,
            seed,
            slot,
            attempt_offset,
            target_nodes,
        )
        return Individual(
            tree,
            target_ty;
            local_consts,
            local_consts_ty,
            meta=(generation=result.metadata,),
        )
    end

    default_init_population = function (reg, data; N, seed, global_type_ctx)
        population = Vector{Individual}(undef, N)
        generated_slots = Vector{Any}(undef, N)
        parallel_init = Threads.nthreads() > 1 && get(
            get(reg.config, :parallel, (; enabled=false)),
            :enabled,
            false,
        )
        if parallel_init
            Threads.@threads :dynamic for slot in 1:N
                generated_slots[slot] = generate_individual(
                    reg.target_ty,
                    UInt64(seed),
                    slot,
                )
            end
        else
            for slot in 1:N
                generated_slots[slot] = generate_individual(
                    reg.target_ty,
                    UInt64(seed),
                    slot,
                )
            end
        end
        for slot in 1:N
            generated = generated_slots[slot]
            generated isa Err && return generated
            population[slot] = generated
        end
        return population
    end

    default_make_immigrant = function (
        reg,
        data;
        seed,
        global_type_ctx,
        slot,
        attempt,
        recorder=nothing,
    )
        return generate_individual(
            reg.target_ty,
            UInt64(seed),
            slot,
            attempt << 16,
        )
    end

    init_population === nothing && (init_population = default_init_population)
    make_immigrant === nothing && (make_immigrant = default_make_immigrant)
    constant_search === nothing && (constant_search = GaussianConstantSearch())

    base = (;
        global_consts,
        global_consts_ty,
        input_slots,
        input_types,
        tyfactory_ctor,
        const_symfactory_ctor,
        functiondef_generation_weight,
        functiondef_recursive_body_site_weight,
        generation_providers,
        contextual_expansion_site_selection,
        binary_composition_site_selection,
        capability_tags,
        capability_mutation_chooser,
        candidate_admissibility,
        make_eval_context,
        mutations,
        mutation_chooser,
        mutation_provider,
        mutation_max_try,
        mutation_prune_consts,
        mutation_bloat_policy,
        mutation_adaptive_policy,
        bloat_policy,
        type_inference,
        candidate_failure,
        loss_failure,
        evaluation_threadsafe,
        crossover_op,
        constant_search,
        mutation_driver,
        functiondef_provider,
        generation_config_factory,
        force_lambda_root_probability,
        init_candidates_per_individual,
        init_target_nodes,
        init_max_try,
        init_growth_attempts,
        init_growth_max_try,
        init_growth_mutation,
        init_population,
        make_immigrant,
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
    resolved_config = _merge_config_overrides(config, (;kwargs...))
    return Regressor(target_ty, components, resolved_config, rng)
end

"""
    fit!(reg::Regressor, data::SupervisedDataset; callback=nothing) -> Union{FitResult,Err}

Run a minimal population-based regression loop.

This function is intentionally conservative: it aims to "run end-to-end" with
strong DI boundaries, not to be optimal. When `config.timing.enabled` is true,
the callback and returned history expose per-generation evaluation,
reproduction, and total generation seconds, together with the run's initial
population construction time. Constant search is an optional subchannel of an
ordinary mutation event: `conditional_share` controls routing after the
mutation-rate gate, and its finite proposal/evaluation budget and timing are
reported separately. Capability-reserved mutation slots remain structural.
"""
function fit!(reg::Regressor, data::SupervisedDataset; callback=nothing)
    length(data.X) == length(data.Y) || return Err(ArgumentError("X and Y must have the same length"))
    components = reg.components
    config = reg.config

    # Required component keys.
    required = Symbol[
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

    preflight_error = _preflight_regression_config(
        reg,
        data,
        components,
        config,
        N;
        require_input=eval_fn === evaluate_default,
    )
    preflight_error === nothing || return Err(preflight_error)
    parallel = get(config, :parallel, (; enabled=false))
    if get(parallel, :enabled, false) &&
       eval_fn !== evaluate_default &&
       !get(components, :evaluation_threadsafe, false)
        return Err(ArgumentError(
            "parallel custom evaluation requires evaluation_threadsafe=true",
        ))
    end

    seed = haskey(config, :seed) ? UInt64(config.seed) : rand(reg.rng, UInt64)

    # Merge typing contexts used by crossover (and optionally by evaluation logging).
    global_type_ctx = merge(components.global_consts_ty, components.input_types)

    # --- Init population ---
    timing_enabled = get(get(config, :timing, (; enabled=false)), :enabled, false)
    initialization_started_ns = time_ns()
    pop = if haskey(components, :init_population)
        res = components.init_population(reg, data; N=N, seed=seed, global_type_ctx=global_type_ctx)
        res isa Err && return res
        postflight_error = _preflight_initial_population(res, reg, components, N)
        postflight_error === nothing || return Err(postflight_error)
        res
    else
        return Err(MissingComponentErr(:init_population))
    end
    initialization_seconds = (time_ns() - initialization_started_ns) / 1.0e9

    history = Any[]
    best = nothing
    best_loss = error_penalty

    for g in 1:gens
        generation_started_ns = time_ns()
        losses = fill(error_penalty, N)
        fitness = fill(error_penalty, N)
        complexities = fill(0, N)
        eval_err_counts = zeros(Int, N)
        eval_exception_counts = zeros(Int, N)
        max_reduce_steps_counts = zeros(Int, N)
        non_const_output_counts = zeros(Int, N)
        missing_const_value_counts = zeros(Int, N)
        non_closed_lambda_counts = zeros(Int, N)
        non_finite_output_counts = zeros(Int, N)
        non_finite_loss_counts = zeros(Int, N)
        loss_exception_counts = zeros(Int, N)
        candidate_failure_counts = zeros(Int, N)
        candidate_rejected_counts = zeros(Int, N)
        invalid_sample_counts = zeros(Int, N)
        candidate_rejection_reasons = [
            _individual_candidate_rejection(pop[i], components)
            for i in 1:N
        ]

        parallel = get(config, :parallel, (; enabled=false))
        do_parallel = get(parallel, :enabled, false) && Threads.nthreads() > 1
        strict_errors = get(config, :strict_errors, false)
        candidate_failure = get(components, :candidate_failure, _known_candidate_exception)
        loss_failure = get(components, :loss_failure, _no_candidate_exception)

        function _classify_eval_issue!(i::Int, issue)
            issue isa MaxReduceStepsErr && (max_reduce_steps_counts[i] += 1; return nothing)
            issue isa NonConstOutputErr && (non_const_output_counts[i] += 1; return nothing)
            issue isa MissingConstValueErr && (missing_const_value_counts[i] += 1; return nothing)
            issue isa NonClosedLambdaErr && (non_closed_lambda_counts[i] += 1; return nothing)
            issue isa NonFiniteOutputErr && (non_finite_output_counts[i] += 1; return nothing)
            return nothing
        end

        function _constant_training_objective(ind::Individual)
            rejection = _individual_candidate_rejection(ind, components)
            rejection === nothing || return Err(ArgumentError(
                "constant-search candidate rejected: $rejection",
            ))
            prepared_default = eval_fn === evaluate_default ?
                prepare_default_evaluator(
                    ind,
                    components,
                    config;
                    copy_tree=false,
                ) : nothing
            total = 0.0
            count = 0
            for (xj, yj) in zip(data.X, data.Y)
                yhat = try
                    prepared_default === nothing ?
                        eval_fn(ind, xj, components, config) :
                        prepared_default(xj)
                catch issue
                    classified = candidate_failure(issue)
                    strict_errors && !classified && rethrow()
                    return Err(issue)
                end
                if yhat isa Err
                    issue = yhat.value
                    classified = candidate_failure(issue)
                    strict_errors && !classified && throw(issue)
                    return yhat
                end
                _finite_prediction(yhat) || return Err(NonFiniteOutputErr())
                sample_loss = try
                    loss_fn(yhat, yj)
                catch issue
                    classified = loss_failure(issue)
                    strict_errors && !classified && rethrow()
                    return Err(issue)
                end
                finite_loss = _finite_real_loss(sample_loss)
                finite_loss === nothing && return Err(NonFiniteOutputErr())
                total += finite_loss
                count += 1
            end
            return total / max(count, 1)
        end

        function _eval_ind!(i::Int)
            ind = pop[i]
            if candidate_rejection_reasons[i] !== nothing
                candidate_rejected_counts[i] = 1
                invalid_sample_counts[i] = length(data.X)
                losses[i] = error_penalty
                complexities[i] = complexity_fn(ind.tree)
                fitness[i] = error_penalty
                return nothing
            end
            prepared_default = eval_fn === evaluate_default ?
                prepare_default_evaluator(
                    ind,
                    components,
                    config;
                    copy_tree=false,
                ) : nothing
            total = 0.0
            count = 0
            for (xj, yj) in zip(data.X, data.Y)
                yhat = try
                    prepared_default === nothing ?
                        eval_fn(ind, xj, components, config) :
                        prepared_default(xj)
                catch e
                    classified = candidate_failure(e)
                    strict_errors && !classified && rethrow()
                    eval_exception_counts[i] += 1
                    classified && (candidate_failure_counts[i] += 1)
                    _classify_eval_issue!(i, e)
                    invalid_sample_counts[i] += 1
                    total += error_penalty
                    count += 1
                    continue
                end
                if yhat isa Err
                    issue = yhat.value
                    classified = candidate_failure(issue)
                    strict_errors && !classified && throw(issue)
                    eval_err_counts[i] += 1
                    classified && (candidate_failure_counts[i] += 1)
                    _classify_eval_issue!(i, issue)
                    invalid_sample_counts[i] += 1
                    total += error_penalty
                elseif !_finite_prediction(yhat)
                    non_finite_output_counts[i] += 1
                    invalid_sample_counts[i] += 1
                    total += error_penalty
                else
                    try
                        sample_loss = loss_fn(yhat, yj)
                        finite_loss = _finite_real_loss(sample_loss)
                        if finite_loss === nothing
                            non_finite_loss_counts[i] += 1
                            invalid_sample_counts[i] += 1
                            total += error_penalty
                        else
                            total += finite_loss
                        end
                    catch e
                        strict_errors && !loss_failure(e) && rethrow()
                        loss_exception_counts[i] += 1
                        loss_failure(e) && (candidate_failure_counts[i] += 1)
                        invalid_sample_counts[i] += 1
                        total += error_penalty
                    end
                end
                count += 1
            end
            average_loss = total / max(count, 1)
            losses[i] = isfinite(average_loss) ? average_loss : error_penalty
            complexities[i] = complexity_fn(ind.tree)
            fitness[i] = losses[i] # parsimony can be injected later
            return nothing
        end

        evaluation_started_ns = time_ns()
        if do_parallel
            Threads.@threads :dynamic for i in 1:N
                _eval_ind!(i)
            end
        else
            for i in 1:N
                _eval_ind!(i)
            end
        end
        evaluation_seconds = (time_ns() - evaluation_started_ns) / 1.0e9
        crossover_seconds = 0.0
        mutation_seconds = 0.0
        constant_search_seconds = 0.0
        timing_snapshot(reproduction_seconds::Float64=0.0) = timing_enabled ? (;
            initialization_seconds,
            evaluation_seconds,
            reproduction_seconds,
            crossover_seconds,
            mutation_seconds,
            constant_search_seconds,
            reproduction_other_seconds=max(
                0.0,
                reproduction_seconds - crossover_seconds - mutation_seconds -
                    constant_search_seconds,
            ),
            generation_seconds=(time_ns() - generation_started_ns) / 1.0e9,
            threads=do_parallel ? Threads.nthreads() : 1,
        ) : nothing

        # Track best.
        valid_candidates = invalid_sample_counts .== 0
        perm = _sorted_indices_by_fitness(fitness, valid_candidates)
        best_idx = perm[1]
        gen_best_loss = losses[best_idx]
        gen_has_valid_candidate = valid_candidates[best_idx]
        if gen_has_valid_candidate && (best === nothing || gen_best_loss < best_loss)
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
            n_non_finite_output=sum(non_finite_output_counts),
            n_non_finite_loss=sum(non_finite_loss_counts),
            n_loss_exception=sum(loss_exception_counts),
            n_candidate_failure=sum(candidate_failure_counts),
            n_candidate_rejected=sum(candidate_rejected_counts),
            candidate_rejection_reasons=let counts = Dict{Symbol,Int}()
                for reason in candidate_rejection_reasons
                    reason === nothing && continue
                    counts[reason] = get(counts, reason, 0) + 1
                end
                counts
            end,
            n_invalid_samples=sum(invalid_sample_counts),
            n_valid_candidates=count(identity, valid_candidates),
            per_individual=(;
                eval_err=copy(eval_err_counts),
                eval_exception=copy(eval_exception_counts),
                max_reduce_steps=copy(max_reduce_steps_counts),
                non_const_output=copy(non_const_output_counts),
                missing_const_value=copy(missing_const_value_counts),
                non_closed_lambda=copy(non_closed_lambda_counts),
                non_finite_output=copy(non_finite_output_counts),
                non_finite_loss=copy(non_finite_loss_counts),
                loss_exception=copy(loss_exception_counts),
                candidate_failure=copy(candidate_failure_counts),
                candidate_rejected=copy(candidate_rejected_counts),
                candidate_rejection_reason=copy(candidate_rejection_reasons),
                invalid_sample=copy(invalid_sample_counts),
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
                gen_has_valid_candidate=gen_has_valid_candidate,
                valid_candidates=valid_candidates,
                best=best,
                best_loss=best_loss,
                eval_stats=eval_stats,
                timing_stats=timing_snapshot(),
                seed=seed,
                components=components,
                config=config,
            )
            cb_res = callback(state)
            cb_res isa Err && return cb_res
            if cb_res === :stop
                push!(history, (; gen=g, best_loss=gen_best_loss, gen_has_valid_candidate, eval_stats, repro_stats=nothing, mutation_stats=nothing, timing_stats=timing_snapshot()))
                return FitResult(best, :stopped, "Stopped by callback", history, seed)
            end
        end

        if gen_has_valid_candidate && gen_best_loss <= threshold
            push!(history, (; gen=g, best_loss=gen_best_loss, gen_has_valid_candidate, eval_stats, repro_stats=nothing, mutation_stats=nothing, timing_stats=timing_snapshot()))
            return FitResult(best, :threshold, "Reached error threshold", history, seed)
        end

        if g == gens
            push!(history, (; gen=g, best_loss=gen_best_loss, gen_has_valid_candidate, eval_stats, repro_stats=nothing, mutation_stats=nothing, timing_stats=timing_snapshot()))
            break
        end

        # --- Reproduce ---
        reproduction_started_ns = time_ns()
        reprocfg = get(config, :repro, (; require_change=false, max_resample=0, on_fail=:clone, immigrant_max_try=1, immigrants_per_gen=0))
        statscfg = get(config, :stats, (; enabled=false))
        do_stats = get(statscfg, :enabled, false)
        rec_mut = (do_stats && get(statscfg, :mutation, false)) ?
            Mutations.MutationAttemptRecorder(;
                record_error_types=get(statscfg, :mutation_error_types, false),
            ) : nothing
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
            :constant_search_attempt => 0,
            :constant_search_applicable => 0,
            :constant_search_improved => 0,
            :constant_search_no_improvement => 0,
            :constant_search_ineligible => 0,
            :constant_search_invalid_source => 0,
            :constant_search_proposals => 0,
            :constant_search_objective_evaluations => 0,
            :constant_search_nonfinite_outcomes => 0,
            :constant_search_invalid_outcomes => 0,
            :constant_search_scalar_proposals => 0,
            :constant_search_tensor_proposals => 0,
            :bloat_rejected => 0,
            :candidate_rejected => 0,
            :n_capability_elite => 0,
            :n_capability_parent_attempt => 0,
            :n_capability_offspring => 0,
        ) : nothing

        @inline function _inc!(st, key::Symbol, delta::Int=1)
            st === nothing && return nothing
            st[key] = get(st, key, 0) + delta
            return nothing
        end

        next_pop = Vector{Individual}()
        selected_indices = Set{Int}()
        tag_cache = Dict{Int,Tuple}()
        tags_for(index::Int) = get!(tag_cache, index) do
            _individual_capability_tags(pop[index], components)
        end
        elitism = min(get(config, :elitism, 1), N)
        for j in 1:elitism
            index = perm[j]
            push!(next_pop, pop[index])
            push!(selected_indices, index)
        end

        capability_requests = get(reprocfg, :capability_elites, (;))
        if !isempty(capability_requests) && length(next_pop) < N
            for (tag, requested) in pairs(capability_requests)
                requested > 0 || continue
                represented = count(
                    index -> valid_candidates[index] && tag in tags_for(index),
                    selected_indices,
                )
                needed = requested - represented
                needed > 0 || continue
                for index in perm
                    length(next_pop) < N || break
                    index in selected_indices && continue
                    valid_candidates[index] || continue
                    tag in tags_for(index) || continue
                    push!(next_pop, pop[index])
                    push!(selected_indices, index)
                    _inc!(repro_stats, :n_capability_elite, 1)
                    needed -= 1
                    needed == 0 && break
                end
            end
        end

        tournament_size = get(config, :tournament_size, 7)
        mutation_rate = get(config, :mutation_rate, 0.7)
        crossover_rate = get(config, :crossover_rate, 0.2)
        constantcfg = get(config, :constant_search, (; enabled=false))
        constant_search_enabled = get(constantcfg, :enabled, false)
        constant_search_share = Float64(get(constantcfg, :conditional_share, 0.4))
        constant_search_budget = ConstantSearchBudget(
            max_proposals=get(constantcfg, :max_proposals, 16),
            max_objective_evals=get(constantcfg, :max_objective_evals, 17),
        )
        crossover_op = components.crossover_op

        pending_children = Individual[]
        require_change = get(reprocfg, :require_change, false)
        max_resample = max(0, get(reprocfg, :max_resample, 0))
        on_fail = get(reprocfg, :on_fail, :clone)
        immigrant_max_try = max(1, get(reprocfg, :immigrant_max_try, 1))
        make_immigrant = haskey(components, :make_immigrant) ? components.make_immigrant : nothing

        function _make_immigrant(slot_idx::Int)
            make_immigrant === nothing && return Err(MissingComponentErr(:make_immigrant))
            for k in 1:immigrant_max_try
                seed_i = _derive_seed(seed, :immigrant, g, slot_idx, k, 0)
                res = make_immigrant(reg, data; seed=seed_i, global_type_ctx=global_type_ctx, slot=slot_idx, attempt=k, recorder=rec_mut)
                res isa Err && continue
                contract_error = _validate_individual_contract(
                    res,
                    reg,
                    components,
                    "immigrant for slot $slot_idx",
                )
                contract_error === nothing || throw(contract_error)
                if !_within_bloat_limit(
                    res.tree,
                    get(components, :bloat_policy, Mutations.BloatPolicy()),
                )
                    _inc!(repro_stats, :bloat_rejected, 1)
                    continue
                end
                if _individual_candidate_rejection(res, components) !== nothing
                    _inc!(repro_stats, :candidate_rejected, 1)
                    continue
                end
                return res
            end
            return Err(Mutations.MaxTryReachedErr())
        end

        immigrants_fixed = max(0, get(reprocfg, :immigrants_per_gen, 0))
        if immigrants_fixed > 0 && make_immigrant !== nothing
            k = min(immigrants_fixed, N - length(next_pop))
            if k > 0
                for fixed_idx in 1:k
                    seed_i = _derive_seed(seed, :immigrant, g, fixed_idx, 1, 1)
                    res = make_immigrant(reg, data; seed=seed_i, global_type_ctx=global_type_ctx, slot=fixed_idx, attempt=1, recorder=rec_mut)
                    if !(res isa Err)
                        contract_error = _validate_individual_contract(
                            res,
                            reg,
                            components,
                            "fixed immigrant $fixed_idx",
                        )
                        contract_error === nothing || throw(contract_error)
                        within_limit = _within_bloat_limit(
                            res.tree,
                            get(components, :bloat_policy, Mutations.BloatPolicy()),
                        )
                        if !within_limit
                            _inc!(repro_stats, :bloat_rejected, 1)
                        elseif _individual_candidate_rejection(res, components) !== nothing
                            _inc!(repro_stats, :candidate_rejected, 1)
                        else
                            push!(next_pop, res)
                            _inc!(repro_stats, :n_immigrant, 1)
                        end
                    end
                end
            end
        end

        capability_parent_indices = Dict{Symbol,Int}()
        capability_parent_tags = Symbol[]
        capability_offspring = get(reprocfg, :capability_offspring, (;))
        remaining_slots = N - length(next_pop)
        for (tag, requested) in pairs(capability_offspring)
            requested > 0 && remaining_slots > 0 || continue
            parent_index = findfirst(perm) do index
                valid_candidates[index] && tag in tags_for(index)
            end
            parent_index === nothing && continue
            capability_parent_indices[tag] = perm[parent_index]
            count_ = min(requested, remaining_slots)
            append!(capability_parent_tags, fill(tag, count_))
            remaining_slots -= count_
        end

        while length(next_pop) < N
            if !isempty(pending_children)
                push!(next_pop, popfirst!(pending_children))
                continue
            end

            produced = false
            slot_idx = length(next_pop) + 1
            last_parent = pop[perm[1]]
            forced_tag = isempty(capability_parent_tags) ? nothing :
                first(capability_parent_tags)
            forced_parent = forced_tag === nothing ? nothing :
                pop[capability_parent_indices[forced_tag]]

            reproduction_attempts = forced_parent === nothing ?
                max_resample + 1 : get(reprocfg, :capability_max_resample, 0) + 1
            for attempt in 1:reproduction_attempts
                attempt > 1 && _inc!(repro_stats, :n_resample, 1)

                rng_sel = derive_rng(seed, :select, g, slot_idx, attempt)
                p1 = if forced_parent === nothing
                    pop[_tournament_index(
                        rng_sel,
                        fitness,
                        valid_candidates,
                        tournament_size,
                    )]
                else
                    _inc!(repro_stats, :n_capability_parent_attempt, 1)
                    forced_parent
                end
                p2 = pop[_tournament_index(rng_sel, fitness, valid_candidates, tournament_size)]
                last_parent = p1

                base_children = forced_parent === nothing ? (p1, p2) : (p1,)
                crossed = false
                rng_cross = derive_rng(seed, :crossover, g, slot_idx, attempt)
                if forced_parent === nothing && crossover_op !== nothing &&
                   rand(rng_cross) < crossover_rate
                    _inc!(repro_stats, :crossover_attempt, 1)
                    crossover_started_ns = timing_enabled ? time_ns() : UInt64(0)
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
                    timing_enabled && (crossover_seconds +=
                        (time_ns() - crossover_started_ns) / 1.0e9)
                    if res isa Err
                        _inc!(repro_stats, :crossover_err, 1)
                    else
                        res isa Tuple && length(res) == 2 ||
                            throw(ArgumentError("crossover must return two children"))
                        policy = get(components, :bloat_policy, Mutations.BloatPolicy())
                        crossover_valid = true
                        for (ci, candidate) in enumerate(res)
                            contract_error = _validate_individual_contract(
                                candidate,
                                reg,
                                components,
                                "crossover child $ci",
                            )
                            contract_error === nothing || throw(contract_error)
                            parent_nodes = ci == 1 ? length(p1.tree) : length(p2.tree)
                            if !_within_bloat_limit(
                                candidate.tree,
                                policy;
                                parent_nodes,
                            )
                                crossover_valid = false
                                break
                            end
                        end
                        if crossover_valid
                            _inc!(repro_stats, :crossover_ok, 1)
                            base_children = res
                            crossed = true
                        else
                            _inc!(repro_stats, :crossover_err, 1)
                            _inc!(repro_stats, :bloat_rejected, 1)
                        end
                    end
                end

                new_children = Individual[]
                ci = 0
                for child0 in base_children
                    ci += 1
                    child = child0
                    changed = crossed
                    rng_mut = derive_rng(seed, :mutate, g, slot_idx, attempt, ci)
                    if forced_parent !== nothing || rand(rng_mut) < mutation_rate
                        constant_route = false
                        if forced_parent === nothing && constant_search_enabled
                            rng_route = derive_rng(
                                seed,
                                :constant_search_route,
                                g,
                                slot_idx,
                                attempt,
                                ci,
                            )
                            constant_route = rand(rng_route) < constant_search_share
                        end
                        if constant_route
                            _inc!(repro_stats, :constant_search_attempt, 1)
                            rng_constant = derive_rng(
                                seed,
                                :constant_search,
                                g,
                                slot_idx,
                                attempt,
                                ci,
                            )
                            search_started_ns = timing_enabled ? time_ns() : UInt64(0)
                            searched = search_constants(
                                components.constant_search,
                                child,
                                _constant_training_objective;
                                rng=rng_constant,
                                budget=constant_search_budget,
                            )
                            timing_enabled && (constant_search_seconds +=
                                (time_ns() - search_started_ns) / 1.0e9)
                            search_stats = searched.stats
                            _inc!(
                                repro_stats,
                                :constant_search_proposals,
                                search_stats.proposals,
                            )
                            _inc!(
                                repro_stats,
                                :constant_search_objective_evaluations,
                                search_stats.objective_evaluations,
                            )
                            _inc!(
                                repro_stats,
                                :constant_search_nonfinite_outcomes,
                                search_stats.nonfinite_outcomes,
                            )
                            _inc!(
                                repro_stats,
                                :constant_search_invalid_outcomes,
                                search_stats.invalid_outcomes,
                            )
                            _inc!(
                                repro_stats,
                                :constant_search_scalar_proposals,
                                search_stats.scalar_proposals,
                            )
                            _inc!(
                                repro_stats,
                                :constant_search_tensor_proposals,
                                search_stats.tensor_proposals,
                            )
                            if searched.status === :ineligible
                                _inc!(repro_stats, :constant_search_ineligible, 1)
                                constant_route = false
                            elseif searched.status === :improved
                                _inc!(repro_stats, :constant_search_applicable, 1)
                                _inc!(repro_stats, :constant_search_improved, 1)
                                contract_error = _validate_individual_contract(
                                    searched.individual,
                                    reg,
                                    components,
                                    "constant-search child $ci",
                                )
                                contract_error === nothing || throw(contract_error)
                                child = searched.individual
                                changed = true
                            elseif searched.status === :no_improvement
                                _inc!(repro_stats, :constant_search_applicable, 1)
                                _inc!(repro_stats, :constant_search_no_improvement, 1)
                            elseif searched.status === :invalid_source
                                _inc!(repro_stats, :constant_search_applicable, 1)
                                _inc!(repro_stats, :constant_search_invalid_source, 1)
                            else
                                throw(ArgumentError(
                                    "unknown constant-search status: $(searched.status)",
                                ))
                            end
                        end
                        if !constant_route
                            _inc!(repro_stats, :mutation_attempt, 1)
                            tyfactory = components.tyfactory_ctor()
                            symfactory = SymbolGenerator()
                            const_symfactory = components.const_symfactory_ctor()
                            local_consts = copy(child.local_consts)
                            local_consts_ty = copy(child.local_consts_ty)
                            mutation_driver = get(
                                components,
                                :mutation_driver,
                                Mutations.mutate_tree,
                            )
                            chooser = components.mutation_chooser
                            if forced_tag !== nothing
                                chooser_router = get(
                                    components,
                                    :capability_mutation_chooser,
                                    (tag, mutations, ordinary) -> ordinary,
                                )
                                applicable(
                                    chooser_router,
                                    forced_tag,
                                    components.mutations,
                                    chooser,
                                ) || throw(ArgumentError(
                                    "capability_mutation_chooser must accept " *
                                    "(tag, mutations, ordinary_chooser)",
                                ))
                                chooser = chooser_router(
                                    forced_tag,
                                    components.mutations,
                                    chooser,
                                )
                                chooser isa Mutations.AbstractPairChooser || throw(
                                    ArgumentError(
                                        "capability_mutation_chooser must return an " *
                                        "AbstractPairChooser",
                                    ),
                                )
                            end
                            mutation_started_ns = timing_enabled ? time_ns() : UInt64(0)
                            mutated = mutation_driver(
                                child.tree,
                                reg.target_ty;
                                rng=rng_mut,
                                mutations=components.mutations,
                                chooser,
                                provider=components.mutation_provider,
                                max_try=forced_parent === nothing ?
                                    components.mutation_max_try : get(
                                        reprocfg,
                                        :capability_mutation_max_try,
                                        1,
                                    ),
                                prune_consts=components.mutation_prune_consts,
                                copy_tree=true,
                                copy_ctx=true,
                                global_consts=components.global_consts,
                                global_consts_ty=global_type_ctx,
                                local_consts=local_consts,
                                local_consts_ty=local_consts_ty,
                                tyfactory=tyfactory,
                                symfactory=symfactory,
                                const_symfactory=const_symfactory,
                                bloat_policy=get(
                                    components,
                                    :bloat_policy,
                                    Mutations.BloatPolicy(),
                                ),
                                adaptive_policy=get(
                                    components,
                                    :mutation_adaptive_policy,
                                    Mutations.AdaptivePolicy(),
                                ),
                                type_inference=get(components, :type_inference, nothing),
                                recorder=rec_mut,
                            )
                            timing_enabled && (mutation_seconds +=
                                (time_ns() - mutation_started_ns) / 1.0e9)
                            if mutated isa Err
                                _inc!(repro_stats, :mutation_err, 1)
                            else
                                _remove_redundant_global_locals!(
                                    local_consts,
                                    local_consts_ty,
                                    components.global_consts,
                                )
                                candidate = Individual(
                                    mutated,
                                    reg.target_ty;
                                    local_consts,
                                    local_consts_ty,
                                    fitness=nothing,
                                    meta=(;),
                                )
                                contract_error = _validate_individual_contract(
                                    candidate,
                                    reg,
                                    components,
                                    "mutation child $ci",
                                )
                                contract_error === nothing || throw(contract_error)
                                if _within_bloat_limit(
                                    candidate.tree,
                                    get(components, :bloat_policy, Mutations.BloatPolicy());
                                    parent_nodes=length(child.tree),
                                )
                                    _inc!(repro_stats, :mutation_ok, 1)
                                    changed = true
                                    child = candidate
                                else
                                    _inc!(repro_stats, :mutation_err, 1)
                                    _inc!(repro_stats, :bloat_rejected, 1)
                                end
                            end
                        end
                    end

                    if require_change && !changed
                        continue
                    end
                    if _individual_candidate_rejection(child, components) !== nothing
                        _inc!(repro_stats, :candidate_rejected, 1)
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
                    if forced_parent !== nothing
                        popfirst!(capability_parent_tags)
                        _inc!(repro_stats, :n_capability_offspring, 1)
                    end
                    produced = true
                    break
                end
            end

            if produced
                continue
            end

            if forced_parent !== nothing
                popfirst!(capability_parent_tags)
                _inc!(repro_stats, :n_clone_fallback, 1)
                push!(next_pop, forced_parent)
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
                println(io, "  repro: children=$(repro_stats[:n_child_total]) clone=$(repro_stats[:n_child_clone]) changed=$(repro_stats[:n_child_changed]) capability_elite=$(repro_stats[:n_capability_elite]) capability_offspring=$(repro_stats[:n_capability_offspring]) candidate_rejected=$(repro_stats[:candidate_rejected]) resample=$(repro_stats[:n_resample]) immigrant=$(repro_stats[:n_immigrant]) clone_fallback=$(repro_stats[:n_clone_fallback]) cross(ok/err)=$(repro_stats[:crossover_ok])/$(repro_stats[:crossover_err]) mut(ok/err)=$(repro_stats[:mutation_ok])/$(repro_stats[:mutation_err]) const(attempt/improved/ineligible)=$(repro_stats[:constant_search_attempt])/$(repro_stats[:constant_search_improved])/$(repro_stats[:constant_search_ineligible])")
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

        reproduction_seconds = (time_ns() - reproduction_started_ns) / 1.0e9
        push!(history, (; gen=g, best_loss=gen_best_loss, gen_has_valid_candidate, eval_stats, repro_stats=repro_stats === nothing ? nothing : copy(repro_stats), mutation_stats, timing_stats=timing_snapshot(reproduction_seconds)))
    end

    best === nothing && return FitResult(
        nothing,
        :no_valid_candidate,
        "No valid candidate was evaluated",
        history,
        seed,
    )
    return FitResult(best, :max_generations, "Reached max generations", history, seed)
end

include("ConstantSearch.jl")
include("TaskSchemas.jl")

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
    using LambdaRegression.Mutations
    using LambdaRegression.Regression: Regressor, SupervisedDataset, fit!, evaluate_default, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.Regression: Regressor, SupervisedDataset, fit!, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike
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
    using Random

    c = default_components()
    @test c.tyfactory_ctor === make_tyvar_factory
    f = c.tyfactory_ctor()
    @test f(0).id == 1
    sf = c.const_symfactory_ctor()
    @test sf isa ConstSymbolFactory
    @test any(m -> m isa Mutations.LetHoistMutation, c.mutations)
    @test any(m -> m isa Mutations.LetSinkMutation, c.mutations)
    @test any(m -> m isa Mutations.LetPruneMutation, c.mutations)
    @test !any(
        m -> m isa Mutations.LetHoistMutation && m.max_bindings > 1,
        c.mutations,
    )
    @test count(
        m -> m isa LambdaRegression.ProceduralMutations.FunctionDefMutation,
        c.mutations,
    ) == 3
    @test c.functiondef_provider.admissibility isa
        LambdaRegression.Procedural.DirectKnotFilter
    functiondef_mutations = filter(
        m -> m isa LambdaRegression.ProceduralMutations.FunctionDefMutation,
        c.mutations,
    )
    @test all(
        mutation -> mutation.functiondef_provider === c.functiondef_provider,
        functiondef_mutations,
    )
    recursive_body = only(filter(functiondef_mutations) do mutation
        LambdaRegression.ProceduralMutations._mode(mutation) === :recursive_body
    end)
    @test recursive_body.site_weight_fn(:recursive_body, nothing, nothing, nothing) == 4.0
    @test recursive_body.generation_kwargs.sampling_depth_limit == 8
    @test recursive_body.generation_kwargs.sampling_node_budget == 96
    direct_knot = Individual(
        AstTree(LambdaRegression.Procedural.FunctionDefNode(AbsNode(
            :x,
            AppNode(IndexNode(1), IndexNode(0)),
        ))),
        TyFun(FloatTy, FloatTy),
    )
    @test c.candidate_admissibility(direct_knot) === :eta_self_forwarding
    binary_global = Individual(
        AstTree(AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:FADD), IndexNode(0)),
                IndexNode(0),
            ),
        )),
        TyFun(FloatTy, FloatTy),
    )
    @test :binary_global_application in c.capability_tags(binary_global)
    unary_global = Individual(
        AstTree(AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0)))),
        TyFun(FloatTy, FloatTy),
    )
    @test :binary_global_application ∉ c.capability_tags(unary_global)
    @test count(
        m -> m isa LambdaRegression.Mutations.ContextualExpansionMutation,
        c.mutations,
    ) == 1
    @test count(
        m -> m isa LambdaRegression.Mutations.UnaryCompositionMutation,
        c.mutations,
    ) == 1
    @test c.init_growth_attempts == 2
    @test length(c.init_growth_mutation) == 3
    @test any(m -> m isa Mutations.UnaryCompositionMutation, c.init_growth_mutation)
    @test any(m -> m isa Mutations.BinaryCompositionMutation, c.init_growth_mutation)
    @test any(m -> m isa Mutations.ContextualExpansionMutation, c.init_growth_mutation)
    @test c.force_lambda_root_probability == 0.5
    @test c.functiondef_generation_weight == 1.0
    @test c.functiondef_recursive_body_site_weight == 4.0
    @test c.crossover_op isa LambdaRegression.ProceduralMutations.AutoTypedCrossover
    @test c.binary_composition_site_selection isa
        LambdaRegression.Mutations.LocalCallSiteSelectionPolicy
    @test c.binary_composition_site_selection.multiplier == 32.0
    self_call_chooser = c.capability_mutation_chooser(
        :self_call,
        c.mutations,
        c.mutation_chooser,
    )
    @test self_call_chooser isa LambdaRegression.Mutations.WeightedPairChooser
    @test self_call_chooser.weights[
        LambdaRegression.Mutations.UnaryCompositionMutation
    ] == 100.0
    recursive_chooser = c.capability_mutation_chooser(
        :function_def,
        c.mutations,
        c.mutation_chooser,
    )
    @test any(functiondef_mutations) do mutation
        LambdaRegression.ProceduralMutations._mode(mutation) === :recursive_body &&
            recursive_chooser.weights[typeof(mutation)] == 8.0
    end
    binary_chooser = c.capability_mutation_chooser(
        :binary_global_application,
        c.mutations,
        c.mutation_chooser,
    )
    @test binary_chooser.weights[
        LambdaRegression.Mutations.UnaryCompositionMutation
    ] == 100.0
    safe_provider = LambdaRegression.CanonicalGeneration.SafeListAccessProvider
    @test any(provider -> provider isa safe_provider, c.generation_providers)
    generation_config = c.generation_config_factory(MersenneTwister(1))
    @test any(provider -> provider isa safe_provider, generation_config.providers)
    generator_backed = filter(c.mutations) do mutation
        mutation isa Union{
            LambdaRegression.Mutations.SubtreeReplacementMutation,
            LambdaRegression.Mutations.ContextualExpansionMutation,
            LambdaRegression.Mutations.BinaryCompositionMutation,
            LambdaRegression.ProceduralMutations.FunctionDefMutation,
        }
    end
    @test length(generator_backed) == 6
    @test all(generator_backed) do mutation
        any(provider -> provider isa safe_provider, mutation.generation_kwargs.providers)
    end
    cfg = default_config()
    @test cfg.population_size == 64
    @test cfg.generations == 20
    @test cfg.tournament_size == 3
    @test cfg.crossover_rate == 0.5
    @test cfg.repro.capability_mutation_max_try == 1
    @test cfg.repro.capability_max_resample == 0
    @test cfg.repro.capability_offspring == (; self_call=1)
end

@testitem "Regression mse basic cases" setup=[RegBaseImports] begin
    @test mse(2.0, 1.0) == 1.0
    @test mse([1.0, 2.0], [1.0, 4.0]) == 2.0
    @test mse([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 6.0]]) == 1.0
end

@testitem "Regression startup constructs a guarded list-map scaffold" setup=[RegBaseImports] begin
    using Random
    using LambdaRegression: Procedural, ProceduralMutations
    using LambdaRegression.TypeSystem:
        CONS, UNCONS_CASE, TyList, get_type_ctx, iscompatible

    list_float = TyList(FloatTy)
    target_ty = TyFun(list_float, list_float)
    globals = Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :CONS => CONS,
        :NIL => Float64[],
    )
    global_types = get_type_ctx(Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :CONS => CONS,
    ))
    global_types[:NIL] = list_float
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
    )
    recursive_mutation = only(filter(components.mutations) do mutation
        mutation isa ProceduralMutations.FunctionDefMutation &&
            ProceduralMutations._mode(mutation) === :recursive_body
    end)
    source = AstTree(Procedural.FunctionDefNode(
        AbsNode(:series, ConstNode(:NIL)),
    ))
    result = ProceduralMutations.mutate_procedural(
        source,
        target_ty;
        rng=MersenneTwister(1),
        mutations=[recursive_mutation],
        chooser=Mutations.WeightedPairChooser(
            weights=Dict{Type,Float64}(typeof(recursive_mutation) => 1.0),
        ),
        provider=Mutations.IndexedProvider(),
        max_try=1,
        global_consts=globals,
        global_consts_ty=global_types,
    )

    @test result isa AstTree
    @test Procedural.functiondef_recursion_summary(result).self_calls == 1
    inferred = Procedural.infer_procedural(result, global_types)
    @test !(inferred isa Err)
    @test iscompatible(inferred, target_ty)
    @test iscompatible(target_ty, inferred)
end

@testitem "Regression startup constructs a guarded list-to-real scaffold" setup=[RegBaseImports] begin
    using Random
    using LambdaRegression: Procedural, ProceduralMutations
    using LambdaRegression.TypeSystem:
        UNCONS_CASE, TyList, get_type_ctx, iscompatible

    list_float = TyList(FloatTy)
    target_ty = TyFun(list_float, FloatTy)
    globals = Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :ZERO => 0.0,
    )
    global_types = get_type_ctx(globals)
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
    )
    recursive_mutation = only(filter(components.mutations) do mutation
        mutation isa ProceduralMutations.FunctionDefMutation &&
            ProceduralMutations._mode(mutation) === :recursive_body
    end)
    source = AstTree(Procedural.FunctionDefNode(
        AbsNode(:series, ConstNode(:ZERO)),
    ))
    result = ProceduralMutations.mutate_procedural(
        source,
        target_ty;
        rng=MersenneTwister(1111),
        mutations=[recursive_mutation],
        chooser=Mutations.WeightedPairChooser(
            weights=Dict{Type,Float64}(typeof(recursive_mutation) => 1.0),
        ),
        provider=Mutations.IndexedProvider(),
        max_try=1,
        global_consts=globals,
        global_consts_ty=global_types,
    )

    @test result isa AstTree
    @test Procedural.functiondef_recursion_summary(result).self_calls == 1
    inferred = Procedural.infer_procedural(result, global_types)
    @test !(inferred isa Err)
    @test iscompatible(inferred, target_ty)
    @test iscompatible(target_ty, inferred)
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

@testitem "Regression evaluate_default runs FunctionDef natively" setup=[RegBaseImports] begin
    using LambdaRegression: Procedural

    target_ty = TyFun(FloatTy, FloatTy)
    tree = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    ind = Individual(
        tree,
        target_ty;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    comps = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )

    @test evaluate_default(ind, 3.5, comps, default_config()) == 3.5
    prepared = LambdaRegression.Regression.prepare_default_evaluator(
        ind,
        comps,
        default_config(),
    )
    @test prepared(4.5) == 4.5
    @test prepared(5.5) == 5.5
end

@testitem "Regression startup defaults initialize and fit without injected hooks" setup=[RegBaseImports] begin
    using Random
    using LambdaRegression.TypeSystem: IntTy

    target_ty = TyFun(FloatTy, FloatTy)
    comps = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_candidates_per_individual=1,
    )
    cfg = default_config(
        population_size=4,
        generations=1,
        mutation_rate=0.0,
        crossover_rate=0.0,
        parallel=(enabled=false,),
        seed=UInt64(9),
    )
    data = SupervisedDataset([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(1111))

    result = fit!(reg, data)
    @test !(result isa Err)
    @test result.best !== nothing
    @test result.history[1].best_loss == 0.0

    scalar_components = default_components(
        global_consts=Dict{Symbol,Any}(:one => 1),
        global_consts_ty=Dict{Symbol,TyLike}(:one => IntTy),
        input_slots=Symbol[],
        input_types=Dict{Symbol,TyLike}(),
        force_lambda_root_probability=1.0,
        init_candidates_per_individual=1,
    )
    scalar_reg = Regressor(IntTy, scalar_components, cfg, MersenneTwister(2000))
    scalar_population = scalar_components.init_population(
        scalar_reg,
        data;
        N=1,
        seed=UInt64(10),
        global_type_ctx=scalar_components.global_consts_ty,
    )
    @test !(scalar_population isa Err)
    @test scalar_population[1].target_ty === IntTy
end

@testitem "Regression default initialization is deterministic across thread modes" setup=[RegBaseImports] begin
    using LambdaRegression.TypeSystem: FADD

    target_ty = TyFun(FloatTy, FloatTy)
    globals = Dict{Symbol,Any}(:FADD => FADD)
    global_types = get_type_ctx(globals)
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_candidates_per_individual=1,
        init_growth_attempts=1,
    )
    serial_config = default_config(
        population_size=6,
        generations=1,
        parallel=(enabled=false,),
        seed=UInt64(19),
    )
    parallel_config = merge(serial_config, (; parallel=(enabled=true,)))
    global_type_ctx = merge(global_types, components.input_types)
    serial = components.init_population(
        Regressor(target_ty, components, serial_config, MersenneTwister(1111)),
        nothing;
        N=6,
        seed=UInt64(19),
        global_type_ctx,
    )
    parallel = components.init_population(
        Regressor(target_ty, components, parallel_config, MersenneTwister(1111)),
        nothing;
        N=6,
        seed=UInt64(19),
        global_type_ctx,
    )

    @test !(serial isa Err)
    @test !(parallel isa Err)
    @test getfield.(serial, :tree) == getfield.(parallel, :tree)
    @test getfield.(serial, :local_consts) == getfield.(parallel, :local_consts)
    @test getfield.(serial, :local_consts_ty) == getfield.(parallel, :local_consts_ty)
    @test [ind.meta.generation for ind in serial] ==
          [ind.meta.generation for ind in parallel]
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

    matrix = [1.0 -2.0 3.0; 4.0 5.0 -6.0]
    MatrixTy = TyArray(
        TyNat(size(matrix, 1)),
        TyArray(TyNat(size(matrix, 2)), FloatTy),
    )
    row_map = AbsNode(
        :row,
        AppNode(AppNode(ConstNode(:AMAP), inner), IndexNode(0)),
    )
    matrix_body = AppNode(AppNode(ConstNode(:AMAP), row_map), IndexNode(0))
    matrix_tree = AstTree(AbsNode(:matrix, matrix_body))
    matrix_ind = Individual(
        matrix_tree,
        TyFun(MatrixTy, MatrixTy);
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    matrix_comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:matrix],
        input_types=Dict{Symbol,TyLike}(:matrix => MatrixTy),
    )
    matrix_y = evaluate_default(matrix_ind, matrix, matrix_comps, cfg)
    matrix_y isa Err && error("nested AMAP returned Err: $(matrix_y.value)")
    @test matrix_y == matrix .^ 2
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

@testitem "Regression non-closed lambda lift throws; fit! rejects invalid initialization" setup=[RegBaseImports] begin
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
            mutations=Mutations.AbstractMutation[],
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
    @test res isa Err
    @test res.value isa ArgumentError
    @test occursin("type inference returned Err", res.value.msg)
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
            mutations=Mutations.AbstractMutation[],
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

@testitem "Regression fit! falls back to clone without immigrant hook" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    id_tree = AstTree(AbsNode(:x, IndexNode(0)))

    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    comps = merge(
        base,
        (;
            mutations=Mutations.AbstractMutation[],
            mutation_chooser=WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=1,
            mutation_prune_consts=true,
            crossover_op=nothing,
            make_immigrant=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) -> [Individual(id_tree, reg.target_ty; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}()) for _ in 1:N],
        ),
    )

    cfg = default_config(
        population_size=6,
        generations=2,
        error_threshold=-1.0,
        mutation_rate=0.0,   # forces require_change path to the missing immigrant hook, then clone fallback
        crossover_rate=0.0,
        elitism=1,
        tournament_size=2,
        repro=(; require_change=true, max_resample=2, on_fail=:immigrant, immigrant_max_try=2, immigrants_per_gen=0),
        stats=(; enabled=true, mutation=true, reproduction=true, print=false, every=1, topk_types=4),
    )
    data = SupervisedDataset([1.0], [1.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))

    res = fit!(reg, data)
    @test !(res isa Err)
    @test length(res.history) == 2
    rs = res.history[1].repro_stats
    @test rs[:n_immigrant] == 0
    @test rs[:n_clone_fallback] == 5
    @test res.history[2].repro_stats === nothing
end

@testitem "Regression fit! uses default crossover before finite clone fallback" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    id_tree = AstTree(AbsNode(:x, IndexNode(0)))

    # Keep the default TypedSubtreeSwapCrossover. Identical typed parents make
    # the finite no-op/error path deterministic without replacing the operator.
    comps = merge(
        default_components(
            global_consts=Dict{Symbol,Any}(),
            global_consts_ty=Dict{Symbol,TyLike}(),
            input_slots=Symbol[:x],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        ),
        (;
            mutations=Mutations.AbstractMutation[],
            mutation_chooser=WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=1,
            mutation_prune_consts=true,
            init_population=(reg, data; N, seed, global_type_ctx) -> [
                Individual(id_tree, reg.target_ty;
                    local_consts=Dict{Symbol,Any}(),
                    local_consts_ty=Dict{Symbol,TyLike}(),
                ) for _ in 1:N
            ],
        ),
    )

    cfg = default_config(
        population_size=4,
        generations=2,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=1.0,
        elitism=1,
        tournament_size=2,
        repro=(; require_change=true, max_resample=1, on_fail=:clone, immigrant_max_try=1, immigrants_per_gen=0),
        stats=(; enabled=true, mutation=false, reproduction=true, print=false, every=1, topk_types=4),
    )
    data = SupervisedDataset([1.0], [1.0])
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(8000))

    res = fit!(reg, data)
    @test !(res isa Err)
    @test length(res.history) == 2
    rs = res.history[1].repro_stats
    @test rs[:crossover_attempt] > 0
    @test rs[:crossover_ok] + rs[:crossover_err] == rs[:crossover_attempt]
    @test rs[:n_clone_fallback] > 0
    @test rs[:n_child_total] + rs[:n_clone_fallback] == 3
    @test res.history[2].repro_stats === nothing
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
            mutations=Mutations.AbstractMutation[],
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

@testitem "Regression fit! records opt-in generation timing" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    ind = Individual(
        AstTree(AbsNode(:x, IndexNode(0))),
        target_ty;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    components = merge(
        base,
        (;
            init_population=(reg, data; N, kwargs...) -> [deepcopy(ind) for _ in 1:N],
        ),
    )
    config = default_config(
        population_size=2,
        generations=2,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        parallel=(enabled=false,),
        timing=(enabled=true,),
    )
    data = SupervisedDataset([-1.0, 1.0], [-1.0, 1.0])
    callback_timing = Any[]
    result = fit!(
        Regressor(target_ty, components, config, MersenneTwister(1234)),
        data;
        callback=state -> (push!(callback_timing, state.timing_stats); nothing),
    )

    @test !(result isa Err)
    @test length(result.history) == 2
    @test length(callback_timing) == 2
    @test all(t -> t !== nothing, callback_timing)
    @test all(t -> t.initialization_seconds >= 0.0, callback_timing)
    @test all(t -> t.evaluation_seconds >= 0.0, callback_timing)
    @test all(t -> t.generation_seconds >= t.evaluation_seconds, callback_timing)
    @test all(t -> t.threads == 1, callback_timing)
    @test result.history[1].timing_stats.reproduction_seconds >= 0.0
    @test result.history[1].timing_stats.crossover_seconds >= 0.0
    @test result.history[1].timing_stats.mutation_seconds >= 0.0
    @test result.history[1].timing_stats.reproduction_other_seconds >= 0.0
    @test result.history[2].timing_stats.reproduction_seconds == 0.0

    disabled = merge(config, (; generations=1, timing=(enabled=false,)))
    disabled_result = fit!(
        Regressor(target_ty, components, disabled, MersenneTwister(8000)),
        data,
    )
    @test disabled_result.history[1].timing_stats === nothing
end

@testitem "Regression fit! penalizes nonfinite values and validates input signatures" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    bad_tree = AstTree(AbsNode(:x, ConstNode(:BAD)))
    bad_ind = Individual(
        bad_tree,
        target_ty;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    base = default_components(
        global_consts=Dict{Symbol,Any}(:BAD => Inf),
        global_consts_ty=Dict{Symbol,TyLike}(:BAD => FloatTy),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    fixed_population = (reg, data; N, seed, global_type_ctx) -> [bad_ind for _ in 1:N]
    components = merge(base, (; init_population=fixed_population))
    config = default_config(
        population_size=1,
        generations=1,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        seed=UInt64(123),
        parallel=(enabled=false,),
    )
    data = SupervisedDataset([1.0], [1.0])

    result = fit!(Regressor(target_ty, components, config, MersenneTwister(1)), data)
    @test !(result isa Err)
    @test result.status == :no_valid_candidate
    @test result.best === nothing
    @test result.seed == UInt64(123)
    @test result.history[1].best_loss == config.error_penalty
    @test result.history[1].eval_stats.n_non_finite_output == 1

    nonfinite_loss_components = merge(
        components,
        (;
            evaluate=(ind, x, components, config) -> x,
            loss=(yhat, y) -> NaN,
        ),
    )
    loss_result = fit!(
        Regressor(target_ty, nonfinite_loss_components, config, MersenneTwister(1111)),
        data,
    )
    @test !(loss_result isa Err)
    @test loss_result.history[1].eval_stats.n_non_finite_loss == 1

    duplicate_slots = merge(
        components,
        (;
            input_slots=Symbol[:x, :x],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        ),
    )
    @test fit!(Regressor(target_ty, duplicate_slots, config, MersenneTwister(1234)), data) isa Err

    too_many_slots = merge(
        components,
        (;
            input_slots=Symbol[:x, :y],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy, :y => FloatTy),
        ),
    )
    @test fit!(Regressor(target_ty, too_many_slots, config, MersenneTwister(8000)), data) isa Err

    scalar_ind = Individual(
        AstTree(ConstNode(:BAD)),
        FloatTy;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    scalar_population = (reg, data; N, seed, global_type_ctx) -> [scalar_ind for _ in 1:N]
    scalar_custom = merge(
        components,
        (;
            input_slots=Symbol[],
            input_types=Dict{Symbol,TyLike}(),
            evaluate=(ind, x, components, config) -> 1.0,
            init_population=scalar_population,
        ),
    )
    scalar_result = fit!(
        Regressor(FloatTy, scalar_custom, config, MersenneTwister(42)),
        data,
    )
    @test !(scalar_result isa Err)

    captured_mutation_kwargs = Ref{Any}(nothing)
    capture_driver = function (tree, target_ty; kwargs...)
        captured_mutation_kwargs[] = kwargs
        return deepcopy(tree)
    end
    capture_components = merge(
        components,
        (;
            mutation_driver=capture_driver,
            crossover_op=nothing,
        ),
    )
    capture_config = merge(
        config,
        (;
            population_size=2,
            generations=2,
            mutation_rate=1.0,
            crossover_rate=0.0,
            elitism=1,
        ),
    )
    capture_result = fit!(
        Regressor(target_ty, capture_components, capture_config, MersenneTwister(555)),
        data,
    )
    @test !(capture_result isa Err)
    @test captured_mutation_kwargs[] !== nothing
    @test captured_mutation_kwargs[][:global_consts] === capture_components.global_consts
    @test captured_mutation_kwargs[][:bloat_policy] === capture_components.mutation_bloat_policy
end

@testitem "Regression validity outranks finite penalty and gates threshold" setup=[RegBaseImports] begin
    @test LambdaRegression.Regression._tournament_index(
        MersenneTwister(2000),
        [1.0, 1.0e8],
        [false, true],
        100,
    ) == 2

    target_ty = TyFun(FloatTy, FloatTy)
    tree = AstTree(AbsNode(:x, IndexNode(0)))
    invalid = Individual(tree, target_ty; meta=(; kind=:invalid))
    valid = Individual(tree, target_ty; meta=(; kind=:valid))
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    components = merge(base, (;
        evaluate=(ind, x, components, config) ->
            ind.meta.kind === :invalid ? Err(DomainError(x)) : 1.0e8,
        loss=(yhat, y) -> abs(yhat - y),
        mutations=Mutations.AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, kwargs...) -> [invalid, valid],
    ))
    config = default_config(
        population_size=2,
        generations=1,
        error_penalty=1.0,
        error_threshold=2.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        parallel=(enabled=false,),
    )
    data = SupervisedDataset([0.0], [0.0])

    result = fit!(Regressor(target_ty, components, config, MersenneTwister(2000)), data)
    @test !(result isa Err)
    @test result.status == :max_generations
    @test result.best.meta.kind === :valid

    all_invalid = merge(components, (;
        init_population=(reg, data; N, kwargs...) -> [invalid, invalid],
    ))
    invalid_result = fit!(
        Regressor(target_ty, all_invalid, config, MersenneTwister(2000)),
        data,
    )
    @test !(invalid_result isa Err)
    @test invalid_result.status == :no_valid_candidate
    @test invalid_result.best === nothing
end

@testitem "Regression reproduction enforces bloat and deterministic substreams" setup=[RegBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    parent_tree = AstTree(AbsNode(:x, IndexNode(0)))
    grown_tree = AstTree(AbsNode(
        :x,
        AppNode(AbsNode(:y, IndexNode(1)), ConstNode(:ONE)),
    ))
    parent = Individual(parent_tree, target_ty)
    policy = Mutations.BloatPolicy(enabled=true, max_nodes=2, max_nodes_delta=0)
    grow_driver = (tree, target_ty; kwargs...) -> deepcopy(grown_tree)
    components = default_components(
        global_consts=Dict{Symbol,Any}(:ONE => 1.0),
        global_consts_ty=Dict{Symbol,TyLike}(:ONE => FloatTy),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        bloat_policy=policy,
        mutation_bloat_policy=policy,
        mutation_driver=grow_driver,
        crossover_op=nothing,
        init_population=(reg, data; N, kwargs...) -> [parent for _ in 1:N],
    )
    config = default_config(
        population_size=2,
        generations=2,
        seed=UInt64(55),
        error_threshold=-1.0,
        mutation_rate=1.0,
        crossover_rate=0.0,
        elitism=1,
        tournament_size=2,
        repro=(; require_change=true, max_resample=0, on_fail=:clone),
        stats=(; enabled=true, mutation=false, reproduction=true),
        parallel=(enabled=false,),
    )
    data = SupervisedDataset([1.0], [1.0])

    first = fit!(Regressor(target_ty, components, config, MersenneTwister(1)), data)
    second = fit!(Regressor(target_ty, components, config, MersenneTwister(2000)), data)
    @test !(first isa Err)
    @test !(second isa Err)
    @test first.seed == second.seed == UInt64(55)
    @test first.history[1].repro_stats == second.history[1].repro_stats
    @test first.history[1].repro_stats[:bloat_rejected] == 2
    @test first.history[1].repro_stats[:mutation_err] == 2
    @test first.history[1].repro_stats[:n_clone_fallback] == 1
    @test first.history[2].repro_stats === nothing
end

@testitem "Regression capability elites preserve a valid self-call niche" setup=[RegBaseImports] begin
    using LambdaRegression: Procedural
    using LambdaRegression.TypeSystem: FADD, UNCONS_CASE, TyList

    list_float = TyList(FloatTy)
    target_ty = TyFun(list_float, FloatTy)
    globals = Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :FADD => FADD,
        :ONE => 1.0,
        :ZERO => 0.0,
    )
    global_types = get_type_ctx(globals)
    best = Individual(AstTree(AbsNode(:xs, ConstNode(:ONE))), target_ty)
    ordinary = Individual(AstTree(AbsNode(:xs, ConstNode(:ZERO))), target_ty)
    on_item = AbsNode(
        :head,
        AbsNode(:tail, AppNode(IndexNode(3), IndexNode(0))),
    )
    recursive = Individual(AstTree(Procedural.FunctionDefNode(AbsNode(
        :xs,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                on_item,
            ),
            ConstNode(:ZERO),
        ),
    ))), target_ty)
    base = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=Symbol[:xs],
        input_types=Dict{Symbol,TyLike}(:xs => list_float),
    )
    components = merge(base, (;
        mutations=Mutations.AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, kwargs...) -> [best, ordinary, recursive],
    ))
    config = default_config(
        population_size=3,
        generations=2,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        tournament_size=1,
        repro=(;
            require_change=false,
            max_resample=0,
            on_fail=:clone,
            immigrant_max_try=1,
            immigrants_per_gen=0,
            capability_elites=(; self_call=1),
        ),
        stats=(; enabled=true, mutation=false, reproduction=true),
        parallel=(; enabled=false),
        seed=UInt64(77),
    )
    saw_self_call_in_second = Ref(false)
    result = fit!(
        Regressor(target_ty, components, config, MersenneTwister(8000)),
        SupervisedDataset([Float64[]], [1.0]);
        callback=state -> begin
            state.gen == 2 || return nothing
            saw_self_call_in_second[] = any(state.pop) do individual
                :self_call in components.capability_tags(individual)
            end
            return nothing
        end,
    )

    @test !(result isa Err)
    @test saw_self_call_in_second[]
    @test result.history[1].repro_stats[:n_capability_elite] == 1

    offspring_config = default_config(
        population_size=3,
        generations=2,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=2,
        tournament_size=1,
        repro=(;
            require_change=false,
            max_resample=0,
            on_fail=:clone,
            immigrant_max_try=1,
            immigrants_per_gen=0,
            capability_elites=(;),
            capability_offspring=(; self_call=1),
        ),
        stats=(; enabled=true, mutation=false, reproduction=true),
        parallel=(; enabled=false),
        seed=UInt64(78),
    )
    offspring_saw_self_call = Ref(false)
    offspring_result = fit!(
        Regressor(target_ty, components, offspring_config, MersenneTwister(42)),
        SupervisedDataset([Float64[]], [1.0]);
        callback=state -> begin
            state.gen == 2 || return nothing
            offspring_saw_self_call[] = any(state.pop) do individual
                :self_call in components.capability_tags(individual)
            end
            return nothing
        end,
    )
    @test !(offspring_result isa Err)
    @test offspring_saw_self_call[]
    @test offspring_result.history[1].repro_stats[:n_capability_elite] == 0
    @test offspring_result.history[1].repro_stats[:n_capability_offspring] == 1
end

@testitem "Regression rejects obvious FunctionDef loops before evaluation" setup=[RegBaseImports] begin
    using LambdaRegression: Procedural

    target_ty = TyFun(FloatTy, FloatTy)
    knot = Individual(
        AstTree(Procedural.FunctionDefNode(AbsNode(
            :x,
            AppNode(IndexNode(1), IndexNode(0)),
        ))),
        target_ty,
    )
    eval_calls = Ref(0)
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    components = merge(base, (;
        evaluate=(ind, x, components, config) -> begin
            eval_calls[] += 1
            return x
        end,
        mutations=Mutations.AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, kwargs...) -> [knot],
    ))
    config = default_config(
        population_size=1,
        generations=1,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        parallel=(; enabled=false),
    )
    result = fit!(
        Regressor(target_ty, components, config, MersenneTwister(1)),
        SupervisedDataset([1.0, 2.0], [1.0, 2.0]),
    )

    @test !(result isa Err)
    @test result.status == :no_valid_candidate
    @test eval_calls[] == 0
    stats = result.history[1].eval_stats
    @test stats.n_candidate_rejected == 1
    @test stats.n_max_reduce_steps == 0
    @test stats.n_invalid_samples == 2
    @test stats.candidate_rejection_reasons == Dict(:eta_self_forwarding => 1)
end

end # module Regression
