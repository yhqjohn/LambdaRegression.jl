"""
    GenerateTree

Clean-room typed AST generation.

This module intentionally exposes a single entry point: [`generate_tree`](@ref).
The goal in this branch is to keep tree generation **small and reliable** as a
foundation for later work; advanced controls (ctors/choosers/keywords) and the
population-based stack are intentionally removed/suspended.

`generate_tree(target_ty)` returns a `NamedTuple`:
- `tree::AstTree`: a generated lambda-calculus AST (locally nameless De Bruijn)
- `consts::Dict{Symbol,Any}`: values for every `ConstNode` symbol used
- `consts_ty::Dict{Symbol,TyLike}`: types for `consts` (for [`TypeSystem.infer`](@ref))
"""
module GenerateTree

using Random
using StatsBase
using TestItems

using ..Utils: Err, @check
using ..ASTs.NodeASTs: AstTree, AstNode, AbsNode, AppNode, ConstNode, IndexNode, SymbolGenerator
using ..ASTs.Contexts: ConstSymbolFactory
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, TyScheme, TyConst, TyVar, TyFun, TyMu, TyProd, TySum
using ..TypeSystem: prune, instantiate, make_tyvar_factory, get_type_ctx, infer, iscompatible, unify!, typectx_fingerprint

export generate_tree
export AbstractCtor, AbsCtor, IndexCtor, GlobalConstCtor, ApplySpineCtor
export AbstractChooser, FocusedChooser

abstract type AbstractCtor end

"""Compatibility marker for allowing lambda introduction in legacy `generate_tree` calls."""
struct AbsCtor <: AbstractCtor end

"""Compatibility marker for allowing de Bruijn variable heads in legacy `generate_tree` calls."""
struct IndexCtor <: AbstractCtor end

"""Compatibility marker for allowing global/local constant heads in legacy `generate_tree` calls."""
struct GlobalConstCtor <: AbstractCtor end

"""Compatibility marker for allowing application spines up to `max_spine_args` in legacy calls."""
struct ApplySpineCtor <: AbstractCtor
    max_spine_args::Int
end
ApplySpineCtor(; max_spine_args::Int=3) = ApplySpineCtor(max_spine_args)

abstract type AbstractChooser end

"""Compatibility chooser shim for legacy mutation/regression APIs."""
struct FocusedChooser <: AbstractChooser
    max_spine_args::Int
end
FocusedChooser(; max_spine_args::Int=3) = FocusedChooser(max_spine_args)

struct UnsupportedTargetTypeErr
    ty::Ty
end

struct GenerationFailedErr
    ty::Ty
end

struct InvalidOptionsErr
    msg::String
end

struct UninhabitedErr
    ty::Ty
end

struct GenBudget
    nodes_left::Int
    height_left::Int
end

mutable struct GenStats
    unify_calls::Int
    unify_failures::Int
    rollback_calls::Int
    intro_attempts::Int
    elim_attempts::Int
    head_trials::Int
    k_trials::Int
    k_hist::Dict{Int,Int}
    failure_hist::Dict{Symbol,Int}
end

GenStats() = GenStats(0, 0, 0, 0, 0, 0, 0, Dict{Int,Int}(), Dict{Symbol,Int}())

mutable struct GenTrace
    run_id::String
    seed::Int
    root_goal_key::String
    event_limit::Int
    dropped_events::Int
    events::Vector{NamedTuple}
end

GenTrace(run_id::String, seed::Int, root_goal_key::String, event_limit::Int) =
    GenTrace(run_id, seed, root_goal_key, event_limit, 0, NamedTuple[])

function _inc!(d::Dict{K,Int}, k::K) where {K}
    d[k] = get(d, k, 0) + 1
    return nothing
end

struct GenOptions
    max_nodes::Int
    max_height::Int
    allow_intro::Bool
    allow_elim::Bool
    intro_bias::Float64
    k_policy::Symbol
    k_temperature::Float64
    use_feasibility_pruning::Bool
    feasibility_cache_mode::Symbol
    use_count_sampling::Bool
    include_index_heads::Bool
    include_const_heads::Bool
end

mutable struct GenCtx
    consts::Dict{Symbol,Any}
    consts_ty::Dict{Symbol,TyLike}
    head_providers::Vector{Function}
    leaf_factories::Vector{Function}
    symfactory::ConstSymbolFactory
    tyctx_fingerprint::UInt
    feasibility_cache::Dict{Any,Bool}
    count_cache::Dict{Any,UInt64}
    trace::Union{Nothing,GenTrace}
end

_goal_key(goal::Ty) = sprint(show, prune(goal))

function _trace_event!(
    trace::Union{Nothing,GenTrace},
    event_kind::Symbol;
    depth::Int=0,
    goal::Union{Nothing,Ty}=nothing,
    env_size::Union{Nothing,Int}=nothing,
    budget::Union{Nothing,GenBudget}=nothing,
    head_name::Union{Nothing,String}=nothing,
    head_kind::Union{Nothing,Symbol}=nothing,
    k::Union{Nothing,Int}=nothing,
    failure_reason::Union{Nothing,Symbol}=nothing,
    elapsed_ns::Integer=0,
)
    trace === nothing && return nothing
    if length(trace.events) >= trace.event_limit
        trace.dropped_events += 1
        return nothing
    end
    push!(trace.events, (
        run_id=trace.run_id,
        seed=trace.seed,
        depth=depth,
        goal_key=goal === nothing ? trace.root_goal_key : _goal_key(goal),
        env_size=isnothing(env_size) ? -1 : env_size,
        budget_nodes=budget === nothing ? -1 : budget.nodes_left,
        budget_height=budget === nothing ? -1 : budget.height_left,
        event_kind=event_kind,
        head_name=isnothing(head_name) ? "" : head_name,
        head_kind=isnothing(head_kind) ? Symbol("") : head_kind,
        k=isnothing(k) ? -1 : k,
        failure_reason=isnothing(failure_reason) ? Symbol("") : failure_reason,
        elapsed_ns=Int(elapsed_ns),
    ))
    return nothing
end

function _trace_state_key(ev)
    return (ev.goal_key, ev.env_size, ev.budget_nodes, ev.budget_height, ev.depth)
end

function trace_run_summary(
    trace::GenTrace,
    result::Union{AstTree,Err},
    stats::Union{Nothing,GenStats};
    canonical_key::AbstractString="",
    elapsed_ns::Integer=0,
)
    status = result isa Err ? "fail" : "ok"
    tree_str = result isa Err ? string(typeof(result.value)) : sprint(show, result.root)
    unify_calls = stats === nothing ? 0 : stats.unify_calls
    unify_failures = stats === nothing ? 0 : stats.unify_failures
    rollback_calls = stats === nothing ? 0 : stats.rollback_calls
    head_trials = stats === nothing ? 0 : stats.head_trials
    k_trials = stats === nothing ? 0 : stats.k_trials
    return "run=$(trace.run_id) seed=$(trace.seed) status=$(status) goal=$(trace.root_goal_key) tree=$(repr(tree_str)) canonical=$(repr(canonical_key)) elapsed_ms=$(round(elapsed_ns / 1_000_000; digits=3)) unify=$(unify_calls)/$(unify_failures) rollback=$(rollback_calls) head_trials=$(head_trials) k_trials=$(k_trials) dropped_events=$(trace.dropped_events)"
end

"""
    trace_hot_summary(trace::GenTrace; topk::Int=5) -> String

Summarise the hottest traced states and the most common head / failure events.

Anchoring strategy: aggregates only recorded trace events; it never influences generation.
Required `ctx` fields: `ctx.trace` must be a `GenTrace`.
Failure modes: returns a short placeholder string when tracing was enabled but no events were recorded.
"""
function trace_hot_summary(trace::GenTrace; topk::Int=5)
    isempty(trace.events) && return "no trace events recorded"
    state_elapsed = Dict{Any,Int}()
    state_fails = Dict{Any,Int}()
    head_counts = Dict{Tuple{String,Int},Int}()
    fail_counts = Dict{Symbol,Int}()
    for ev in trace.events
        if ev.event_kind === :state_success || ev.event_kind === :state_fail
            key = _trace_state_key(ev)
            state_elapsed[key] = get(state_elapsed, key, 0) + ev.elapsed_ns
            ev.event_kind === :state_fail && (state_fails[key] = get(state_fails, key, 0) + 1)
        end
        ev.event_kind === :head_trial && (head_counts[(ev.head_name, ev.depth)] = get(head_counts, (ev.head_name, ev.depth), 0) + 1)
        ev.failure_reason !== Symbol("") && (fail_counts[ev.failure_reason] = get(fail_counts, ev.failure_reason, 0) + 1)
    end
    hottest = join(first(["$(k[1])@d$(k[5]) env=$(k[2]) budget=$(k[3])/$(k[4]) elapsed_ms=$(round(v / 1_000_000; digits=3))" for (k, v) in sort!(collect(state_elapsed); by=x -> -x[2])], min(topk, length(state_elapsed))), " | ")
    failing = join(first(["$(k[1])@d$(k[5]) fails=$(v)" for (k, v) in sort!(collect(state_fails); by=x -> -x[2])], min(topk, length(state_fails))), " | ")
    head_lines = String[]
    for ((name, depth), count) in first(sort!(collect(head_counts); by=x -> -x[2]), min(topk, length(head_counts)))
        push!(head_lines, string(name == "" ? "<anon>" : name, "@d", depth, ":", count))
    end
    heads = join(head_lines, " | ")
    fails = join(first(["$(reason):$(count)" for (reason, count) in sort!(collect(fail_counts); by=x -> -x[2])], min(topk, length(fail_counts))), " | ")
    return "hottest_states=$(repr(hottest))\nfrequent_fail_states=$(repr(failing))\nhead_trials=$(repr(heads))\nfailure_reasons=$(repr(fails))"
end

"""
    generate_tree(target_ty::TyLike, seed::Integer=0) -> Union{NamedTuple,Err}

Generate a small AST that **infers** to `target_ty` under the returned `consts_ty`.

Only a minimal set of type constructors is supported in this finite generator:
- `TyConst` (`:Int`, `:Float`, `:Bool`)
- `TyFun`
- `TyProd` (via `PAIR`)
- `TySum` (via `INL`/`INR`)

If generation is not supported, returns `Err(UnsupportedTargetTypeErr(...))`.
"""
function generate_tree(target_ty::TyLike, seed::Integer=0; rng::Union{Nothing,AbstractRNG}=nothing, kwargs...)
    rng === nothing && (rng = MersenneTwister(seed))
    return generate_tree(rng, target_ty; seed_hint=Int(seed), kwargs...)
end

"""
    generate_tree(rng::AbstractRNG, target_ty::TyLike; kwargs...) -> Union{NamedTuple,Err}

Typed, β-normal focusing generator.

Keyword arguments (stable defaults):
- `consts::Union{Nothing,Dict{Symbol,Any}}=nothing`: values for `ConstNode`s (used by evaluation).
- `consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing`: explicit typing context override/extension.
- `consts_ty_extra::AbstractDict{Symbol,TyLike}=Dict()`: extra type-only entries (e.g. typed combinators without values).
- `max_height::Int=12`: AST height budget (the "depth D" control).
- `max_nodes::Union{Nothing,Int}=nothing`: node budget; defaults to `4*max_height + 1`.
- `allow_intro::Bool=true`: allow λ-introduction branch.
- `allow_elim::Bool=true`: allow elimination (head+spine) branch.
- `intro_bias::Float64=0.55`: when both branches are possible, intro is tried first with this probability.
- `k_policy::Symbol=:boltzmann` or `:uniform`: sampling order for spine length `k`.
- `k_temperature::Float64=2.0`: temperature for Boltzmann (`τ>0`).
- `head_providers::Vector{Function}=Function[]`: optional injected head candidates.
- `leaf_factories::Vector{Function}=Function[]`: optional injected leaf/constant factories.
- `return_stats::Bool=false`: return `stats` in the output tuple.
- `trace::Bool=false`: record observation-only trace events for debugging and inspection.
- `trace_event_limit::Int=10_000`: maximum number of trace events kept in memory.
- `trace_run_id::Union{Nothing,AbstractString}=nothing`: optional stable run identifier.
- `use_feasibility_pruning::Bool=true`: enable feasibility pruning (bundle-key memoised).
- `use_sat_pruning::Union{Nothing,Bool}=nothing`: deprecated alias for `use_feasibility_pruning`.
- `feasibility_cache_mode::Symbol=:bundle`: cache mode for feasibility checks.
- `sat_cache_mode::Union{Nothing,Symbol}=nothing`: deprecated alias for `feasibility_cache_mode`.
"""
function generate_tree(
    rng::AbstractRNG,
    target_ty::TyLike;
    consts::Union{Nothing,Dict{Symbol,Any}}=nothing,
    consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    consts_ty_extra::AbstractDict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    max_height::Int=12,
    max_nodes::Union{Nothing,Int}=nothing,
    allow_intro::Bool=true,
    allow_elim::Bool=true,
    intro_bias::Float64=0.55,
    k_policy::Symbol=:boltzmann,
    k_temperature::Float64=2.0,
    head_providers::Vector{Function}=Function[],
    leaf_factories::Vector{Function}=Function[],
    return_stats::Bool=false,
    trace::Bool=false,
    trace_event_limit::Int=10_000,
    trace_run_id::Union{Nothing,AbstractString}=nothing,
    seed_hint::Int=-1,
    use_feasibility_pruning::Bool=true,
    use_sat_pruning::Union{Nothing,Bool}=nothing,
    feasibility_cache_mode::Symbol=:bundle,
    sat_cache_mode::Union{Nothing,Symbol}=nothing,
    verify_infer::Bool=true,
    use_count_sampling::Bool=false,
    ctors::Union{Nothing,AbstractVector{<:AbstractCtor}}=nothing,
    chooser::Union{Nothing,AbstractChooser}=nothing,
    max_depth::Union{Nothing,Int}=nothing,
    max_try::Union{Nothing,Int}=nothing,
    tyfactory=nothing,
    symfactory=nothing,
    global_consts_ty::Union{Nothing,AbstractDict{Symbol,<:TyLike}}=nothing,
    local_consts_ty::Union{Nothing,AbstractDict{Symbol,<:TyLike}}=nothing,
    local_consts::Union{Nothing,Dict{Symbol,Any}}=nothing,
    var_ty::Union{Nothing,AbstractDict{Symbol,<:TyLike}}=nothing,
    scope_stack::Union{Nothing,AbstractVector{Symbol}}=nothing,
)
    legacy_mode = !isnothing(ctors) || !isnothing(chooser) || !isnothing(max_depth) || !isnothing(max_try) ||
                  !isnothing(global_consts_ty) || !isnothing(local_consts_ty) || !isnothing(local_consts) ||
                  !isnothing(var_ty) || !isnothing(scope_stack) || !isnothing(tyfactory) || !isnothing(symfactory)
    if legacy_mode
        return _generate_tree_legacy(
            rng,
            target_ty;
            ctors,
            chooser,
            max_depth,
            max_try,
            global_consts_ty,
            local_consts_ty,
            local_consts,
            var_ty,
            scope_stack,
            use_feasibility_pruning,
            use_sat_pruning,
            feasibility_cache_mode,
            sat_cache_mode,
            verify_infer,
            use_count_sampling,
        )
    end
    use_sat_pruning === nothing || (use_feasibility_pruning = use_sat_pruning)
    sat_cache_mode === nothing || (feasibility_cache_mode = sat_cache_mode)
    return _generate_tree_core(
        rng,
        target_ty;
        consts,
        consts_ty,
        consts_ty_extra,
        max_height,
        max_nodes,
        allow_intro,
        allow_elim,
        intro_bias,
        k_policy,
        k_temperature,
        head_providers,
        leaf_factories,
        return_stats,
        trace,
        trace_event_limit,
        trace_run_id,
        seed_hint,
        use_feasibility_pruning,
        feasibility_cache_mode,
        verify_infer,
        use_count_sampling,
    )
end

function _legacy_ctor_flags(ctors::Union{Nothing,AbstractVector{<:AbstractCtor}})
    ctors === nothing && return (allow_intro=true, allow_elim=true, max_spine_args=3, include_index_heads=true, include_const_heads=true)
    allow_intro = isempty(ctors) ? true : any(c -> c isa AbsCtor, ctors)
    allow_elim = isempty(ctors) ? true : any(c -> !(c isa AbsCtor), ctors)
    include_index_heads = isempty(ctors) ? true : any(c -> c isa IndexCtor, ctors)
    include_const_heads = isempty(ctors) ? true : any(c -> c isa GlobalConstCtor, ctors)
    max_spine_args = 0
    for ctor in ctors
        if ctor isa ApplySpineCtor
            max_spine_args = max(max_spine_args, ctor.max_spine_args)
        elseif ctor isa IndexCtor || ctor isa GlobalConstCtor
            max_spine_args = max(max_spine_args, 0)
        end
    end
    max_spine_args = max(max_spine_args, 0)
    return (; allow_intro, allow_elim, max_spine_args, include_index_heads, include_const_heads)
end

function _legacy_initial_env(var_ty::Union{Nothing,AbstractDict{Symbol,<:TyLike}}, scope_stack::Union{Nothing,AbstractVector{Symbol}}, factory)
    (var_ty === nothing || scope_stack === nothing) && return Ty[]
    env = Ty[]
    for name in scope_stack
        haskey(var_ty, name) || continue
        ty_like = var_ty[name]
        ty = ty_like isa TyScheme ? instantiate(ty_like, 0, factory) : deepcopy(ty_like)
        push!(env, prune(ty))
    end
    return env
end

function _generate_tree_legacy(
    rng::AbstractRNG,
    target_ty::TyLike;
    ctors,
    chooser,
    max_depth,
    max_try,
    global_consts_ty,
    local_consts_ty,
    local_consts,
    var_ty,
    scope_stack,
    use_feasibility_pruning,
    use_sat_pruning,
    feasibility_cache_mode,
    sat_cache_mode,
    verify_infer,
    use_count_sampling,
)
    flags = _legacy_ctor_flags(ctors)
    chooser_max = chooser isa FocusedChooser ? chooser.max_spine_args : flags.max_spine_args
    depth = something(max_depth, 12)
    nodes = 4 * depth + 1
    consts_seed = merge(_default_consts(), local_consts === nothing ? Dict{Symbol,Any}() : copy(local_consts))
    tyctx = copy(global_consts_ty === nothing ? Dict{Symbol,TyLike}() : global_consts_ty)
    local_tyctx = local_consts_ty === nothing ? Dict{Symbol,TyLike}() : local_consts_ty
    factory = make_tyvar_factory()
    initial_env = _legacy_initial_env(var_ty, scope_stack, factory)
    result = _generate_tree_core(
        rng,
        target_ty;
        consts=consts_seed,
        consts_ty=tyctx,
        consts_ty_extra=local_tyctx,
        max_height=depth,
        max_nodes=nodes,
        allow_intro=flags.allow_intro,
        allow_elim=flags.allow_elim,
        k_policy=:boltzmann,
        k_temperature=2.0,
        return_stats=false,
        trace=false,
        seed_hint=-1,
        use_feasibility_pruning=use_sat_pruning === nothing ? use_feasibility_pruning : use_sat_pruning,
        feasibility_cache_mode=sat_cache_mode === nothing ? feasibility_cache_mode : sat_cache_mode,
        verify_infer=verify_infer && isempty(initial_env),
        use_count_sampling=use_count_sampling,
        initial_env=initial_env,
        compat_max_spine_args=max(chooser_max, flags.max_spine_args),
        include_index_heads=flags.include_index_heads,
        include_const_heads=flags.include_const_heads,
    )
    result isa Err && return result
    if local_consts !== nothing
        empty!(local_consts)
        merge!(local_consts, filter(((k, _),) -> !haskey(_default_consts(), k), pairs(result.consts)))
    end
    if local_consts_ty !== nothing
        empty!(local_consts_ty)
        for (k, v) in pairs(result.consts_ty)
            if !haskey(tyctx, k)
                local_consts_ty[k] = v
            end
        end
    end
    return result.tree
end

function _generate_tree_core(
    rng::AbstractRNG,
    target_ty::TyLike;
    consts::Union{Nothing,Dict{Symbol,Any}}=nothing,
    consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    consts_ty_extra::AbstractDict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    max_height::Int=12,
    max_nodes::Union{Nothing,Int}=nothing,
    allow_intro::Bool=true,
    allow_elim::Bool=true,
    intro_bias::Float64=0.55,
    k_policy::Symbol=:boltzmann,
    k_temperature::Float64=2.0,
    head_providers::Vector{Function}=Function[],
    leaf_factories::Vector{Function}=Function[],
    return_stats::Bool=false,
    trace::Bool=false,
    trace_event_limit::Int=10_000,
    trace_run_id::Union{Nothing,AbstractString}=nothing,
    seed_hint::Int=-1,
    use_feasibility_pruning::Bool=true,
    feasibility_cache_mode::Symbol=:bundle,
    verify_infer::Bool=true,
    use_count_sampling::Bool=false,
    initial_env::Vector{Ty}=Ty[],
    compat_max_spine_args::Int=typemax(Int),
    include_index_heads::Bool=true,
    include_const_heads::Bool=true,
)
    max_height >= 1 || return Err(InvalidOptionsErr("max_height must be >= 1"))
    if max_nodes === nothing
        max_nodes = 4 * max_height + 1
    end
    max_nodes >= 1 || return Err(InvalidOptionsErr("max_nodes must be >= 1"))
    allow_intro || allow_elim || return Err(InvalidOptionsErr("allow_intro and allow_elim cannot both be false"))
    0.0 <= intro_bias <= 1.0 || return Err(InvalidOptionsErr("intro_bias must be in [0,1]"))
    (k_policy === :boltzmann || k_policy === :uniform) || return Err(InvalidOptionsErr("k_policy must be :boltzmann or :uniform"))
    k_policy === :uniform || (k_temperature > 0.0) || return Err(InvalidOptionsErr("k_temperature must be > 0"))
    (feasibility_cache_mode === :monomorphic || feasibility_cache_mode === :bundle) || return Err(InvalidOptionsErr("feasibility_cache_mode must be :monomorphic or :bundle"))

    consts_local = consts === nothing ? _default_consts() : copy(consts)
    tyctx = get_type_ctx(consts_local)
    consts_ty !== nothing && merge!(tyctx, consts_ty)
    for (k, v) in pairs(consts_ty_extra)
        tyctx[k] = v
    end

    factory = make_tyvar_factory()
    goal = target_ty isa TyScheme ? instantiate(target_ty, 0, factory) : deepcopy(target_ty)
    goal = prune(goal)

    options = GenOptions(
        max_nodes,
        max_height,
        allow_intro,
        allow_elim,
        intro_bias,
        k_policy,
        k_temperature,
        use_feasibility_pruning,
        feasibility_cache_mode,
        use_count_sampling,
        include_index_heads,
        include_const_heads,
    )

    stats = return_stats ? GenStats() : nothing
    trace_state = trace ? GenTrace(
        trace_run_id === nothing ? "seed-$(rand(rng, UInt))" : String(trace_run_id),
        seed_hint,
        _goal_key(goal),
        trace_event_limit,
    ) : nothing
    ctx = GenCtx(
        consts_local,
        tyctx,
        head_providers,
        leaf_factories,
        ConstSymbolFactory(prefix=:c),
        typectx_fingerprint(tyctx),
        Dict{Any,Bool}(),
        Dict{Any,UInt64}(),
        trace_state,
    )

    trail = TypeSystem.TyTrail()
    namegen = SymbolGenerator()
    budget = GenBudget(options.max_nodes, options.max_height)
    env = copy(initial_env)
    t0 = time_ns()
    _trace_event!(ctx.trace, :run_start; depth=0, goal=goal, env_size=length(env), budget=budget)

    tree = _gen_normal(rng, goal, env, budget, ctx, options, trail, factory, namegen, stats, 0; compat_max_spine_args=compat_max_spine_args)
    if tree isa Err
        _trace_event!(ctx.trace, :run_end; depth=0, goal=goal, env_size=length(env), budget=budget, failure_reason=:generation_failed, elapsed_ns=time_ns() - t0)
        return trace ? (; error=tree, consts=ctx.consts, consts_ty=ctx.consts_ty, stats, trace=ctx.trace) : tree
    end

    if verify_infer
        inferred = infer(tree, ctx.consts_ty)
        if inferred isa Err
            _trace_event!(ctx.trace, :run_end; depth=0, goal=goal, env_size=length(env), budget=budget, failure_reason=:infer_failed, elapsed_ns=time_ns() - t0)
            return trace ? (; error=inferred, consts=ctx.consts, consts_ty=ctx.consts_ty, stats, trace=ctx.trace) : inferred
        end
        if !(iscompatible(inferred, goal) || iscompatible(goal, inferred))
            err = Err(GenerationFailedErr(goal))
            _trace_event!(ctx.trace, :run_end; depth=0, goal=goal, env_size=length(env), budget=budget, failure_reason=:infer_incompatible, elapsed_ns=time_ns() - t0)
            return trace ? (; error=err, consts=ctx.consts, consts_ty=ctx.consts_ty, stats, trace=ctx.trace) : err
        end
    end
    _trace_event!(ctx.trace, :run_end; depth=0, goal=goal, env_size=length(env), budget=budget, elapsed_ns=time_ns() - t0)

    if return_stats
        return trace ? (; tree, consts=ctx.consts, consts_ty=ctx.consts_ty, stats, trace=ctx.trace) : (; tree, consts=ctx.consts, consts_ty=ctx.consts_ty, stats)
    end
    return trace ? (; tree, consts=ctx.consts, consts_ty=ctx.consts_ty, trace=ctx.trace) : (; tree, consts=ctx.consts, consts_ty=ctx.consts_ty)
end

function _default_consts()::Dict{Symbol,Any}
    return Dict{Symbol,Any}(
        :zero => 0,
        :one => 1,
        :fzero => 0.0,
        :fone => 1.0,
        :btrue => true,
        :bfalse => false,
        :COND => TypeSystem.COND,
        :INL => TypeSystem.INL,
        :INR => TypeSystem.INR,
        :PAIR => TypeSystem.PAIR,
        :FST => TypeSystem.FST,
        :SND => TypeSystem.SND,
    )
end

struct HeadCandidate
    node::AstNode
    ty_like::TyLike
    name::Union{Nothing,Symbol} # for stats/debug
end

function _is_monomorphic(t::Ty)::Bool
    t = prune(t)
    if t isa TyVar
        v = t::TyVar
        return v.instance !== nothing && _is_monomorphic(v.instance)
    elseif t isa TyConst || t isa TypeSystem.TyNat
        return true
    elseif t isa TyFun
        f = t::TyFun
        return _is_monomorphic(f.arg) && _is_monomorphic(f.res)
    elseif t isa TyProd
        p = t::TyProd
        return _is_monomorphic(p.left) && _is_monomorphic(p.right)
    elseif t isa TySum
        s = t::TySum
        return _is_monomorphic(s.left) && _is_monomorphic(s.right)
    elseif t isa TypeSystem.TyList
        return _is_monomorphic((t::TypeSystem.TyList).elem)
    elseif t isa TypeSystem.TyArray
        a = t::TypeSystem.TyArray
        return _is_monomorphic(a.len) && _is_monomorphic(a.elem)
    elseif t isa TyMu
        m = t::TyMu
        return _is_monomorphic(m.body)
    end
    return true
end

function _typekey_bundle(types::Vector{Ty})
    free_ids = IdDict{TyVar,Int}()
    keys = Any[]
    for t in types
        mu_stack = TyVar[]
        seen = IdDict{Ty,Nothing}()
        push!(keys, _typekey_walk_bundle(t, free_ids, mu_stack, seen))
    end
    return Tuple(keys)
end

function _typekey_walk_bundle(t::Ty, free_ids::IdDict{TyVar,Int}, mu_stack::Vector{TyVar}, seen::IdDict{Ty,Nothing})
    if t isa TyVar
        v = t::TyVar
        if !isempty(mu_stack)
            j = findlast(x -> x === v, mu_stack)
            if j !== nothing
                depth = length(mu_stack) - j
                return (:mu_ref, depth)
            end
        end
        if v.instance !== nothing
            return _typekey_walk_bundle(v.instance, free_ids, mu_stack, seen)
        end
        idx = get!(free_ids, v) do
            length(free_ids)
        end
        return (:v, idx)
    end

    if haskey(seen, t)
        return (:cycle,)
    end
    seen[t] = nothing

    if t isa TyConst
        return (:const, (t::TyConst).name)
    elseif t isa TypeSystem.TyNat
        return (:nat, (t::TypeSystem.TyNat).value)
    elseif t isa TyFun
        f = t::TyFun
        return (:fun, _typekey_walk_bundle(f.arg, free_ids, mu_stack, seen), _typekey_walk_bundle(f.res, free_ids, mu_stack, seen))
    elseif t isa TyProd
        p = t::TyProd
        return (:prod, _typekey_walk_bundle(p.left, free_ids, mu_stack, seen), _typekey_walk_bundle(p.right, free_ids, mu_stack, seen))
    elseif t isa TySum
        s = t::TySum
        return (:sum, _typekey_walk_bundle(s.left, free_ids, mu_stack, seen), _typekey_walk_bundle(s.right, free_ids, mu_stack, seen))
    elseif t isa TypeSystem.TyList
        l = t::TypeSystem.TyList
        return (:list, _typekey_walk_bundle(l.elem, free_ids, mu_stack, seen))
    elseif t isa TypeSystem.TyArray
        a = t::TypeSystem.TyArray
        return (:array, _typekey_walk_bundle(a.len, free_ids, mu_stack, seen), _typekey_walk_bundle(a.elem, free_ids, mu_stack, seen))
    elseif t isa TyMu
        m = t::TyMu
        push!(mu_stack, m.var)
        body_k = _typekey_walk_bundle(m.body, free_ids, mu_stack, seen)
        pop!(mu_stack)
        return (:mu, body_k)
    else
        return (:unknown, Symbol(string(typeof(t))))
    end
end

function _feasibility_key(goal::Ty, env::Vector{Ty}, budget::GenBudget, ctx::GenCtx, options::GenOptions)
    if options.feasibility_cache_mode === :monomorphic
        # Only cache fully monomorphic states to avoid shared-free-var correlation bugs.
        all_mono = _is_monomorphic(goal) && all(_is_monomorphic, env)
        all_mono || return nothing
        return (:feasibility_mono, ctx.tyctx_fingerprint, budget.nodes_left, budget.height_left, TypeSystem.typekey(goal), Tuple(TypeSystem.typekey.(env)))
    end
    # Bundle-key mode: preserve shared TyVar identity across (goal, env...).
    return (:feasibility_bundle, ctx.tyctx_fingerprint, budget.nodes_left, budget.height_left, _typekey_bundle(Ty[goal; env...]))
end

function _is_feasible(
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    factory,
)::Bool
    budget.nodes_left >= 1 || return false
    budget.height_left >= 1 || return false

    key = _feasibility_key(goal, env, budget, ctx, options)
    if key !== nothing
        cached = get(ctx.feasibility_cache, key, nothing)
        cached === nothing || return cached
    end

    # Over-approx: feasibility pruning is only used for pruning. Avoid false negatives by being permissive on node splitting.
    trail = TypeSystem.TyTrail()
    ok = _is_feasible_uncached(goal, env, budget, ctx, options, trail, factory)

    if key !== nothing
        ctx.feasibility_cache[key] = ok
    end
    return ok
end

function _is_feasible_uncached(
    goal0::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
)::Bool
    goal = prune(goal0)

    can_intro = options.allow_intro && budget.nodes_left >= 2 && budget.height_left >= 2
    can_elim = options.allow_elim

    if can_intro
        m = TypeSystem.mark(trail)
        a = factory(0)
        b = factory(0)
        if !(_try_unify!(goal, TyFun(a, b), trail, nothing) isa Err)
            push!(env, a)
            ok = _is_feasible(b, env, GenBudget(budget.nodes_left - 1, budget.height_left - 1), ctx, options, factory)
            pop!(env)
            TypeSystem.rollback!(trail, m)
            ok && return true
        else
            TypeSystem.rollback!(trail, m)
        end
    end

    if can_elim
        heads = _collect_heads(env, ctx, options)
        isempty(heads) && return false

        kmax_nodes = (budget.nodes_left - 1) ÷ 2
        kmax_height = budget.height_left - 1
        kmax = min(kmax_nodes, kmax_height)
        kmax >= 0 || return false

        for head in heads
            head_ty = head.ty_like isa TyScheme ? instantiate(head.ty_like, 0, factory) : (head.ty_like::Ty)
            for k in 0:kmax
                m = TypeSystem.mark(trail)
                arg_tys = Ty[factory(0) for _ in 1:k]
                chain = goal
                for i in k:-1:1
                    chain = TyFun(arg_tys[i], chain)
                end
                if _try_unify!(head_ty, chain, trail, nothing) isa Err
                    TypeSystem.rollback!(trail, m)
                    continue
                end

                nodes_for_args = budget.nodes_left - (1 + k)
                nodes_for_args >= 0 || begin
                    TypeSystem.rollback!(trail, m)
                    continue
                end
                ok = true
                for i in 1:k
                    h_i = _height_upper_for_arg(budget.height_left, k, i)
                    h_i >= 1 || (ok = false; break)
                    # Over-approx nodes: give each arg the full remaining nodes budget.
                    ok = _is_feasible(arg_tys[i], env, GenBudget(nodes_for_args, h_i), ctx, options, factory)
                    ok || break
                end
                TypeSystem.rollback!(trail, m)
                ok && return true
            end
        end
    end

    return false
end

function _count_key(tag::Symbol, goal::Ty, env::Vector{Ty}, budget::GenBudget, ctx::GenCtx, options::GenOptions)
    if options.feasibility_cache_mode === :monomorphic
        all_mono = _is_monomorphic(goal) && all(_is_monomorphic, env)
        all_mono || return nothing
        return (tag, :mono, ctx.tyctx_fingerprint, budget.nodes_left, budget.height_left, TypeSystem.typekey(goal), Tuple(TypeSystem.typekey.(env)))
    end
    return (tag, :bundle, ctx.tyctx_fingerprint, budget.nodes_left, budget.height_left, _typekey_bundle(Ty[goal; env...]))
end

const _COUNT_SATURATE = typemax(UInt64)

@inline function _sat_add(a::UInt64, b::UInt64)::UInt64
    s = a + b
    s < a ? _COUNT_SATURATE : s
end

@inline function _sat_mul(a::UInt64, b::UInt64)::UInt64
    (a == 0 || b == 0) && return 0
    a > _COUNT_SATURATE ÷ b ? _COUNT_SATURATE : (a * b)
end

function _count(
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    factory,
)::UInt64
    budget.nodes_left >= 1 || return 0
    budget.height_left >= 1 || return 0

    key = _count_key(:count, goal, env, budget, ctx, options)
    if key !== nothing
        cached = get(ctx.count_cache, key, nothing)
        cached === nothing || return cached
    end

    trail = TypeSystem.TyTrail()
    c = _count_uncached(goal, env, budget, ctx, options, trail, factory)
    if key !== nothing
        ctx.count_cache[key] = c
    end
    return c
end

function _count_intro_root(goal0::Ty, env::Vector{Ty}, budget::GenBudget, ctx::GenCtx, options::GenOptions, factory)::UInt64
    budget.nodes_left >= 2 || return 0
    budget.height_left >= 2 || return 0
    key = _count_key(:count_intro_root, goal0, env, budget, ctx, options)
    if key !== nothing
        cached = get(ctx.count_cache, key, nothing)
        cached === nothing || return cached
    end
    trail = TypeSystem.TyTrail()
    goal = prune(goal0)
    m = TypeSystem.mark(trail)
    a = factory(0)
    b = factory(0)
    c::UInt64 = 0
    if !(_try_unify!(goal, TyFun(a, b), trail, nothing) isa Err)
        push!(env, a)
        c = _count(b, env, GenBudget(budget.nodes_left - 1, budget.height_left - 1), ctx, options, factory)
        pop!(env)
    end
    TypeSystem.rollback!(trail, m)
    if key !== nothing
        ctx.count_cache[key] = c
    end
    return c
end

function _count_elim_root(goal0::Ty, env::Vector{Ty}, budget::GenBudget, ctx::GenCtx, options::GenOptions, factory)::UInt64
    budget.nodes_left >= 1 || return 0
    budget.height_left >= 1 || return 0
    key = _count_key(:count_elim_root, goal0, env, budget, ctx, options)
    if key !== nothing
        cached = get(ctx.count_cache, key, nothing)
        cached === nothing || return cached
    end
    trail = TypeSystem.TyTrail()
    goal = prune(goal0)
    total::UInt64 = 0

    heads = _collect_heads(env, ctx, options)
    if !isempty(heads)
        kmax_nodes = (budget.nodes_left - 1) ÷ 2
        kmax_height = budget.height_left - 1
        kmax = min(kmax_nodes, kmax_height)
        if kmax >= 0
            for head in heads
                head_ty = head.ty_like isa TyScheme ? instantiate(head.ty_like, 0, factory) : (head.ty_like::Ty)
                for k in 0:kmax
                    m = TypeSystem.mark(trail)
                    arg_tys = Ty[factory(0) for _ in 1:k]
                    chain = goal
                    for i in k:-1:1
                        chain = TyFun(arg_tys[i], chain)
                    end
                    if _try_unify!(head_ty, chain, trail, nothing) isa Err
                        TypeSystem.rollback!(trail, m)
                        continue
                    end

                    nodes_for_args = budget.nodes_left - (1 + k)
                    nodes_for_args >= 0 || begin
                        TypeSystem.rollback!(trail, m)
                        continue
                    end

                    c_k::UInt64 = 1
                    for i in 1:k
                        h_i = _height_upper_for_arg(budget.height_left, k, i)
                        h_i >= 1 || (c_k = 0; break)
                        c_i = _count(arg_tys[i], env, GenBudget(nodes_for_args, h_i), ctx, options, factory)
                        c_k = _sat_mul(c_k, c_i)
                        c_k == 0 && break
                    end

                    total = _sat_add(total, c_k)
                    TypeSystem.rollback!(trail, m)
                    total == _COUNT_SATURATE && break
                end
                total == _COUNT_SATURATE && break
            end
        end
    end

    if key !== nothing
        ctx.count_cache[key] = total
    end
    return total
end

function _count_uncached(
    goal0::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
)::UInt64
    goal = prune(goal0)
    total::UInt64 = 0

    if options.allow_intro && budget.nodes_left >= 2 && budget.height_left >= 2
        m = TypeSystem.mark(trail)
        a = factory(0)
        b = factory(0)
        if !(_try_unify!(goal, TyFun(a, b), trail, nothing) isa Err)
            push!(env, a)
            total = _sat_add(total, _count(b, env, GenBudget(budget.nodes_left - 1, budget.height_left - 1), ctx, options, factory))
            pop!(env)
        end
        TypeSystem.rollback!(trail, m)
        total == _COUNT_SATURATE && return total
    end

    options.allow_elim || return total

    heads = _collect_heads(env, ctx, options)
    isempty(heads) && return total

    kmax_nodes = (budget.nodes_left - 1) ÷ 2
    kmax_height = budget.height_left - 1
    kmax = min(kmax_nodes, kmax_height)
    kmax >= 0 || return total

    for head in heads
        head_ty = head.ty_like isa TyScheme ? instantiate(head.ty_like, 0, factory) : (head.ty_like::Ty)
        for k in 0:kmax
            m = TypeSystem.mark(trail)
            arg_tys = Ty[factory(0) for _ in 1:k]
            chain = goal
            for i in k:-1:1
                chain = TyFun(arg_tys[i], chain)
            end
            if _try_unify!(head_ty, chain, trail, nothing) isa Err
                TypeSystem.rollback!(trail, m)
                continue
            end

            nodes_for_args = budget.nodes_left - (1 + k)
            nodes_for_args >= 0 || begin
                TypeSystem.rollback!(trail, m)
                continue
            end

            # Over-approx nodes splitting: allow each arg up to nodes_for_args.
            c_k::UInt64 = 1
            for i in 1:k
                h_i = _height_upper_for_arg(budget.height_left, k, i)
                h_i >= 1 || (c_k = 0; break)
                c_i = _count(arg_tys[i], env, GenBudget(nodes_for_args, h_i), ctx, options, factory)
                c_k = _sat_mul(c_k, c_i)
                c_k == 0 && break
            end

            total = _sat_add(total, c_k)
            TypeSystem.rollback!(trail, m)
            total == _COUNT_SATURATE && return total
        end
    end

    return total
end

function _record_failure!(stats::Union{Nothing,GenStats}, reason::Symbol)
    stats === nothing && return nothing
    _inc!(stats.failure_hist, reason)
    return nothing
end

function _mark!(trail)::Int
    return TypeSystem.mark(trail)
end

function _rollback!(
    trail,
    m::Int,
    stats::Union{Nothing,GenStats},
    trace::Union{Nothing,GenTrace}=nothing;
    depth::Int=0,
    goal::Union{Nothing,Ty}=nothing,
    env_size::Union{Nothing,Int}=nothing,
    budget::Union{Nothing,GenBudget}=nothing,
    failure_reason::Union{Nothing,Symbol}=nothing,
)
    TypeSystem.rollback!(trail, m)
    stats === nothing || (stats.rollback_calls += 1)
    _trace_event!(trace, :rollback; depth=depth, goal=goal, env_size=env_size, budget=budget, failure_reason=failure_reason)
    return nothing
end

function _try_unify!(
    t1::Ty,
    t2::Ty,
    trail,
    stats::Union{Nothing,GenStats},
    trace::Union{Nothing,GenTrace}=nothing;
    depth::Int=0,
    goal::Union{Nothing,Ty}=nothing,
    env_size::Union{Nothing,Int}=nothing,
    budget::Union{Nothing,GenBudget}=nothing,
    head_name::Union{Nothing,String}=nothing,
    head_kind::Union{Nothing,Symbol}=nothing,
    k::Union{Nothing,Int}=nothing,
)
    stats === nothing || (stats.unify_calls += 1)
    r = unify!(t1, t2; trail=trail)
    if r isa Err
        stats === nothing || (stats.unify_failures += 1)
        _trace_event!(trace, :unify_err; depth=depth, goal=goal, env_size=env_size, budget=budget, head_name=head_name, head_kind=head_kind, k=k, failure_reason=:unify_err)
        return r
    end
    _trace_event!(trace, :unify_ok; depth=depth, goal=goal, env_size=env_size, budget=budget, head_name=head_name, head_kind=head_kind, k=k)
    return nothing
end

function _k_order(rng::AbstractRNG, kmax::Int, options::GenOptions)
    ks = collect(0:kmax)
    if isempty(ks)
        return ks
    end
    if options.k_policy === :uniform
        shuffle!(rng, ks)
        return ks
    end
    τ = options.k_temperature
    ws = Float64[exp(-(2.0 * k) / τ) for k in ks]
    return sample(rng, ks, Weights(ws), length(ks); replace=false)
end

function _split_nodes_upper(rng::AbstractRNG, nodes_for_args::Int, k::Int)
    k == 0 && return Int[]
    nodes_for_args >= k || return Int[] # caller should prevent this
    remaining = nodes_for_args
    split = Vector{Int}(undef, k)
    for i in 1:(k - 1)
        # Leave at least 1 for each remaining arg.
        max_i = remaining - (k - i)
        n_i = rand(rng, 1:max_i)
        split[i] = n_i
        remaining -= n_i
    end
    split[k] = remaining
    return split
end

function _height_upper_for_arg(height_left::Int, k::Int, i::Int)
    # argument ai (1-indexed) ends up under (k - i + 1) App nodes.
    return height_left - (k - i + 1)
end

function _collect_heads(env::Vector{Ty}, ctx::GenCtx, options::GenOptions)
    heads = HeadCandidate[]
    # Bound variables (De Bruijn): IndexNode(0) refers to env[end]
    if options.include_index_heads
        for i in 0:(length(env) - 1)
            push!(heads, HeadCandidate(IndexNode(i), env[end - i], nothing))
        end
    end
    if options.include_const_heads
        for (sym, ty_like) in pairs(ctx.consts_ty)
            push!(heads, HeadCandidate(ConstNode(sym), ty_like, sym))
        end
    end
    for p in ctx.head_providers
        provided = p()
        provided isa Err && continue
        for item in provided
            node, ty_like, name = item
            push!(heads, HeadCandidate(node, ty_like, name))
        end
    end
    return heads
end

function _maybe_run_leaf_factories!(rng::AbstractRNG, goal::Ty, ctx::GenCtx, factory, stats::Union{Nothing,GenStats})
    isempty(ctx.leaf_factories) && return nothing
    for f in ctx.leaf_factories
        r = f(rng, goal, ctx, factory)
        r isa Err && _record_failure!(stats, :leaf_factory_err)
    end
    # Update fingerprint if types changed (cheap enough here).
    ctx.tyctx_fingerprint = typectx_fingerprint(ctx.consts_ty)
    return nothing
end

function _gen_normal(
    rng::AbstractRNG,
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
    namegen::SymbolGenerator,
    stats::Union{Nothing,GenStats},
    depth::Int,
    ; compat_max_spine_args::Int=typemax(Int),
)::Union{AstTree,Err}
    t0 = time_ns()
    budget.nodes_left >= 1 || return Err(UninhabitedErr(goal))
    budget.height_left >= 1 || return Err(UninhabitedErr(goal))
    goal = prune(goal)
    _trace_event!(ctx.trace, :state_enter; depth=depth, goal=goal, env_size=length(env), budget=budget)

    _maybe_run_leaf_factories!(rng, goal, ctx, factory, stats)

    can_intro = options.allow_intro && budget.nodes_left >= 2 && budget.height_left >= 2
    can_elim = options.allow_elim

    if !can_intro && !can_elim
        _record_failure!(stats, :no_branches)
        return Err(UninhabitedErr(goal))
    end

    # Decide try-order; always try both if the first fails.
    try_intro_first = can_intro && (!can_elim || rand(rng) < options.intro_bias)
    if options.use_count_sampling && can_intro && can_elim
        c_intro = _count_intro_root(goal, env, budget, ctx, options, factory)
        c_elim = _count_elim_root(goal, env, budget, ctx, options, factory)
        if c_intro == 0 && c_elim == 0
            try_intro_first = rand(rng) < options.intro_bias
        elseif c_intro == 0
            try_intro_first = false
        elseif c_elim == 0
            try_intro_first = true
        else
            try_intro_first = rand(rng) < (float(c_intro) / float(c_intro + c_elim))
        end
    end
    for act in (try_intro_first ? (:intro, :elim) : (:elim, :intro))
        if act === :intro
            stats === nothing || (stats.intro_attempts += 1)
            _trace_event!(ctx.trace, :branch_choice; depth=depth, goal=goal, env_size=length(env), budget=budget, head_kind=:intro)
            m = _mark!(trail)
            a = factory(0)
            b = factory(0)
            r = _try_unify!(goal, TyFun(a, b), trail, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, head_kind=:intro)
            if r isa Err
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:intro_unify_fail)
                _record_failure!(stats, :intro_unify_fail)
                continue
            end
            push!(env, a)
            body = _gen_normal(
                rng,
                b,
                env,
                GenBudget(budget.nodes_left - 1, budget.height_left - 1),
                ctx,
                options,
                trail,
                factory,
                namegen,
                stats,
                depth + 1,
                compat_max_spine_args=compat_max_spine_args,
            )
            pop!(env)
            if body isa Err
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:intro_body_fail)
                _record_failure!(stats, :intro_body_fail)
                continue
            end
            _trace_event!(ctx.trace, :state_success; depth=depth, goal=goal, env_size=length(env), budget=budget, head_kind=:intro, elapsed_ns=time_ns() - t0)
            return AstTree(AbsNode(namegen(), body.root))
        else
            can_elim || continue
            stats === nothing || (stats.elim_attempts += 1)
            _trace_event!(ctx.trace, :branch_choice; depth=depth, goal=goal, env_size=length(env), budget=budget, head_kind=:elim)
            elim = _gen_elim(
                rng,
                goal,
                env,
                budget,
                ctx,
                options,
                trail,
                factory,
                namegen,
                stats,
                depth,
                compat_max_spine_args=compat_max_spine_args,
            )
            if !(elim isa Err)
                _trace_event!(ctx.trace, :state_success; depth=depth, goal=goal, env_size=length(env), budget=budget, head_kind=:elim, elapsed_ns=time_ns() - t0)
                return elim
            end
        end
    end
    _trace_event!(ctx.trace, :state_fail; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:state_uninhabited, elapsed_ns=time_ns() - t0)
    return Err(UninhabitedErr(goal))
end

function _gen_elim(
    rng::AbstractRNG,
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
    namegen::SymbolGenerator,
    stats::Union{Nothing,GenStats},
    depth::Int,
    ; compat_max_spine_args::Int=typemax(Int),
)::Union{AstTree,Err}
    heads = _collect_heads(env, ctx, options)
    isempty(heads) && return Err(UninhabitedErr(goal))
    shuffle!(rng, heads)

    kmax_nodes = (budget.nodes_left - 1) ÷ 2
    kmax_height = budget.height_left - 1
    kmax = min(kmax_nodes, kmax_height, compat_max_spine_args)
    kmax >= 0 || return Err(UninhabitedErr(goal))

    for head in heads
        stats === nothing || (stats.head_trials += 1)
        head_kind = head.node isa IndexNode ? :index : :const
        head_name = head.node isa IndexNode ? "idx$((head.node::IndexNode).index)" : string((head.node::ConstNode).name)
        _trace_event!(ctx.trace, :head_trial; depth=depth, goal=goal, env_size=length(env), budget=budget, head_name=head_name, head_kind=head_kind)
        head_ty = head.ty_like isa TyScheme ? instantiate(head.ty_like, 0, factory) : (head.ty_like::Ty)
        for k in _k_order(rng, kmax, options)
            stats === nothing || (stats.k_trials += 1)
            stats === nothing || _inc!(stats.k_hist, k)
            _trace_event!(ctx.trace, :k_trial; depth=depth, goal=goal, env_size=length(env), budget=budget, head_name=head_name, head_kind=head_kind, k=k)

            m = _mark!(trail)
            arg_tys = Ty[factory(0) for _ in 1:k]
            chain = goal
            for i in k:-1:1
                chain = TyFun(arg_tys[i], chain)
            end
            r = _try_unify!(head_ty, chain, trail, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, head_name=head_name, head_kind=head_kind, k=k)
            if r isa Err
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:elim_unify_fail)
                _record_failure!(stats, :elim_unify_fail)
                continue
            end

            nodes_for_args = budget.nodes_left - (1 + k)
            nodes_for_args >= k || begin
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:elim_nodes_too_small)
                _record_failure!(stats, :elim_nodes_too_small)
                continue
            end

            if options.use_feasibility_pruning && k > 0
                ok_feasible = true
                for i in 1:k
                    h_i = _height_upper_for_arg(budget.height_left, k, i)
                    h_i >= 1 || (ok_feasible = false; break)
                    ok_feasible = _is_feasible(arg_tys[i], env, GenBudget(nodes_for_args, h_i), ctx, options, factory)
                    ok_feasible || break
                end
                ok_feasible || begin
                    _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:elim_feasibility_prune)
                    _record_failure!(stats, :elim_feasibility_prune)
                    continue
                end
            end

            node_budgets = _split_nodes_upper(rng, nodes_for_args, k)
            length(node_budgets) == k || begin
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:elim_split_fail)
                _record_failure!(stats, :elim_split_fail)
                continue
            end

            args = AstTree[]
            ok = true
            for i in 1:k
                h_i = _height_upper_for_arg(budget.height_left, k, i)
                if h_i < 1
                    ok = false
                    _record_failure!(stats, :elim_height_too_small)
                    break
                end
                arg_budget = GenBudget(node_budgets[i], h_i)
                arg_tree = _gen_normal(rng, arg_tys[i], env, arg_budget, ctx, options, trail, factory, namegen, stats, depth + 1)
                if arg_tree isa Err
                    ok = false
                    _trace_event!(ctx.trace, :arg_fail; depth=depth, goal=goal, env_size=length(env), budget=budget, head_name=head_name, head_kind=head_kind, k=k, failure_reason=:elim_arg_fail)
                    _record_failure!(stats, :elim_arg_fail)
                    break
                end
                push!(args, arg_tree)
            end

            if !ok
                _rollback!(trail, m, stats, ctx.trace; depth=depth, goal=goal, env_size=length(env), budget=budget, failure_reason=:elim_arg_fail)
                continue
            end

            node = head.node
            for a in args
                node = AppNode(node, a.root)
            end
            return AstTree(node)
        end
    end
    return Err(UninhabitedErr(goal))
end

function _gen_elim_forced(
    rng::AbstractRNG,
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
    namegen::SymbolGenerator,
    stats::Union{Nothing,GenStats},
    head_sym::Symbol,
    k::Int;
    arg_retries::Int=40,
)::Union{AstTree,Err}
    haskey(ctx.consts_ty, head_sym) || return Err(UninhabitedErr(goal))
    k >= 0 || return Err(UninhabitedErr(goal))

    head = HeadCandidate(ConstNode(head_sym), ctx.consts_ty[head_sym], head_sym)
    head_ty = head.ty_like isa TyScheme ? instantiate(head.ty_like, 0, factory) : (head.ty_like::Ty)

    kmax_nodes = (budget.nodes_left - 1) ÷ 2
    kmax_height = budget.height_left - 1
    k <= min(kmax_nodes, kmax_height) || return Err(UninhabitedErr(goal))

    m = _mark!(trail)
    arg_tys = Ty[factory(0) for _ in 1:k]
    chain = goal
    for i in k:-1:1
        chain = TyFun(arg_tys[i], chain)
    end
    r = _try_unify!(head_ty, chain, trail, stats)
    if r isa Err
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end

    nodes_for_args = budget.nodes_left - (1 + k)
    nodes_for_args >= k || begin
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end
    node_budgets = _split_nodes_upper(rng, nodes_for_args, k)
    length(node_budgets) == k || begin
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end

    args = AstTree[]
    for i in 1:k
        h_i = _height_upper_for_arg(budget.height_left, k, i)
        h_i >= 1 || begin
            _rollback!(trail, m, stats)
            return Err(UninhabitedErr(goal))
        end
        arg_budget = GenBudget(node_budgets[i], h_i)
        arg_tree = nothing
        for _ in 1:arg_retries
            t = _gen_normal(rng, arg_tys[i], env, arg_budget, ctx, options, trail, factory, namegen, stats, 1)
            if !(t isa Err)
                arg_tree = t
                break
            end
        end
        arg_tree === nothing && begin
            _rollback!(trail, m, stats)
            return Err(UninhabitedErr(goal))
        end
        push!(args, arg_tree)
    end

    node = head.node
    for a in args
        node = AppNode(node, a.root)
    end
    return AstTree(node)
end

function _gen_elim_forced_index(
    rng::AbstractRNG,
    goal::Ty,
    env::Vector{Ty},
    budget::GenBudget,
    ctx::GenCtx,
    options::GenOptions,
    trail,
    factory,
    namegen::SymbolGenerator,
    stats::Union{Nothing,GenStats},
    index::Int,
    k::Int;
    node_budgets_override::Union{Nothing,Vector{Int}}=nothing,
    arg_retries::Int=20,
)::Union{AstTree,Err}
    0 <= index < length(env) || return Err(UninhabitedErr(goal))
    k >= 0 || return Err(UninhabitedErr(goal))

    head_node = IndexNode(index)
    head_ty = env[end - index]

    kmax_nodes = (budget.nodes_left - 1) ÷ 2
    kmax_height = budget.height_left - 1
    k <= min(kmax_nodes, kmax_height) || return Err(UninhabitedErr(goal))

    m = _mark!(trail)
    arg_tys = Ty[factory(0) for _ in 1:k]
    chain = goal
    for i in k:-1:1
        chain = TyFun(arg_tys[i], chain)
    end
    r = _try_unify!(head_ty, chain, trail, stats)
    if r isa Err
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end

    nodes_for_args = budget.nodes_left - (1 + k)
    nodes_for_args >= k || begin
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end
    node_budgets = node_budgets_override === nothing ? _split_nodes_upper(rng, nodes_for_args, k) : node_budgets_override
    length(node_budgets) == k || begin
        _rollback!(trail, m, stats)
        return Err(UninhabitedErr(goal))
    end

    args = AstTree[]
    for i in 1:k
        h_i = _height_upper_for_arg(budget.height_left, k, i)
        h_i >= 1 || begin
            _rollback!(trail, m, stats)
            return Err(UninhabitedErr(goal))
        end
        arg_budget = GenBudget(node_budgets[i], h_i)
        arg_tree = nothing
        for _ in 1:arg_retries
            t = _gen_normal(rng, arg_tys[i], env, arg_budget, ctx, options, trail, factory, namegen, stats, 1)
            if !(t isa Err)
                arg_tree = t
                break
            end
        end
        arg_tree === nothing && begin
            _rollback!(trail, m, stats)
            return Err(UninhabitedErr(goal))
        end
        push!(args, arg_tree)
    end

    node = head_node
    for a in args
        node = AppNode(node, a.root)
    end
    return AstTree(node)
end

"""
    _inject_random_float_constant!(rng, goal, ctx::GenCtx, factory) -> Union{Nothing,Err}

Example leaf factory (not enabled by default): injects a fresh float constant when the goal is `Float`.
"""
function _inject_random_float_constant!(rng::AbstractRNG, goal::Ty, ctx::GenCtx, factory)
    goal = prune(goal)
    goal isa TyConst || return nothing
    (goal::TyConst).name === :Float || return nothing
    name = _next_const_symbol!(ctx.symfactory)
    # Ensure no collision.
    haskey(ctx.consts, name) && return nothing
    v = rand(rng)
    ctx.consts[name] = v
    ctx.consts_ty[name] = TypeSystem.FloatTy
    return nothing
end

# ConstSymbolFactory helper (local; mirrors ASTs.Contexts but avoids importing its private).
function _next_const_symbol!(f::ConstSymbolFactory)::Symbol
    f.counter += 1
    return Symbol(String(f.prefix), f.counter)
end

@testsnippet GTBaseImports begin
    using Test
    using Random
    import LambdaRegression
    using LambdaRegression.Utils: Err
    using LambdaRegression.GenerateTree: generate_tree
    using LambdaRegression.ASTs.NodeASTs: AstTree, AstNode, AbsNode, AppNode, ConstNode, IndexNode, SymbolGenerator
    using LambdaRegression.TypeSystem: TyLike, TyConst, TyFun, TyMu, TyProd, TySum, Ty, IntTy, FloatTy, BoolTy
    using LambdaRegression.TypeSystem: infer, iscompatible, make_tyvar_factory, get_type_ctx, ∀, →
end

@testitem "GenerateTree cleanroom: base + higher-order types" setup=[GTBaseImports] begin
    # Base type: uses ConstNode + returned consts_ty.
    r1 = generate_tree(IntTy, 7; max_height=5, max_nodes=21)
    @test !(r1 isa Err)
    @test r1.tree isa AstTree
    @test iscompatible(infer(r1.tree, r1.consts_ty), IntTy)

    # Product and sum.
    prod_ty = TyProd(IntTy, BoolTy)
    r2 = generate_tree(prod_ty, 3; max_height=5, max_nodes=21)
    @test !(r2 isa Err)
    @test iscompatible(infer(r2.tree, r2.consts_ty), prod_ty)

    sum_ty = TySum(FloatTy, IntTy)
    r3 = generate_tree(sum_ty, 4; max_height=5, max_nodes=21)
    @test !(r3 isa Err)
    @test iscompatible(infer(r3.tree, r3.consts_ty), sum_ty)

    # Higher-order function: (Int -> Int) -> Int -> Int
    ho_ty = TyFun(TyFun(IntTy, IntTy), TyFun(IntTy, IntTy))
    r4 = generate_tree(ho_ty, 4; max_height=5, max_nodes=21)
    @test !(r4 isa Err)
    @test iscompatible(infer(r4.tree, r4.consts_ty), ho_ty)
end

function _contains_beta_redex(node::AstNode)::Bool
    if node isa AppNode
        n = node::AppNode
        (n.func isa AbsNode) && return true
        return _contains_beta_redex(n.func) || _contains_beta_redex(n.arg)
    elseif node isa AbsNode
        return _contains_beta_redex((node::AbsNode).body)
    else
        return false
    end
end

function _contains_head_spine(node::AstNode, head_sym::Symbol; min_k::Int=0)::Bool
    if node isa AppNode
        # peel left-associated spine
        k = 0
        cur = node
        while cur isa AppNode
            k += 1
            cur = (cur::AppNode).func
        end
        if cur isa ConstNode && (cur::ConstNode).name === head_sym && k >= min_k
            return true
        end
    end
    if node isa AbsNode
        return _contains_head_spine((node::AbsNode).body, head_sym; min_k=min_k)
    elseif node isa AppNode
        n = node::AppNode
        return _contains_head_spine(n.func, head_sym; min_k=min_k) || _contains_head_spine(n.arg, head_sym; min_k=min_k)
    end
    return false
end

function _root_head_kind(node::AstNode)::Symbol
    cur = node
    while cur isa AppNode
        cur = (cur::AppNode).func
    end
    if cur isa ConstNode
        return :const
    elseif cur isa IndexNode
        return :index
    elseif cur isa AbsNode
        return :abs
    end
    return :other
end

function _root_spine_len(node::AstNode)::Int
    k = 0
    cur = node
    while cur isa AppNode
        k += 1
        cur = (cur::AppNode).func
    end
    return k
end

function _spine_len_bucket(k::Int)::String
    k == 0 && return "0"
    k == 1 && return "1"
    k == 2 && return "2"
    k <= 4 && return "3-4"
    return "5+"
end

"""
    pattern_features(tree::AstTree, consts_ty::AbstractDict=Dict()) -> NamedTuple

Extract observation-only structural pattern flags from a generated tree.

Anchoring strategy: analyses the generated tree after the fact; it never feeds back into generation.
Required `ctx` fields: none; `consts_ty` is optional and only used to detect higher-order result types.
Failure modes: if post-hoc `infer` fails, `is_higher_order` is reported as `false`.
"""
function pattern_features(tree::AstTree, consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}())
    inferred = isempty(consts_ty) ? Err(nothing) : infer(tree, consts_ty)
    root_head_kind = _root_head_kind(tree.root)
    root_spine_len = _root_spine_len(tree.root)
    has_mu = !(inferred isa Err) && _type_contains_mu(inferred)
    is_higher_order = !(inferred isa Err) && prune(inferred) isa TyFun
    return (
        has_cond_overapply=_contains_head_spine(tree.root, :COND; min_k=4),
        has_fst_overapply=_contains_head_spine(tree.root, :FST; min_k=2),
        has_mu=has_mu,
        is_higher_order=is_higher_order,
        head_kind=root_head_kind,
        spine_len_bucket=_spine_len_bucket(root_spine_len),
    )
end

@testitem "GenerateTree: β-normal invariant (sampled)" setup=[GTBaseImports] begin
    goal = TyFun(IntTy, IntTy)
    for seed in (4, 5, 6)
        r = generate_tree(goal, seed; max_height=7, max_nodes=29)
        @test !(r isa Err)
        @test !LambdaRegression.GenerateTree._contains_beta_redex(r.tree.root)
    end
end

@testitem "GenerateTree: tracing off/on stays self-consistent" setup=[GTBaseImports] begin
    goal = TyFun(IntTy, IntTy)
    plain = generate_tree(goal, 4; max_height=7, max_nodes=29, return_stats=true)
    traced = generate_tree(goal, 4; max_height=7, max_nodes=29, return_stats=true, trace=true, trace_run_id="trace-consistency")

    @test !(plain isa Err)
    @test !haskey(traced, :error)
    @test sprint(show, plain.tree.root) == sprint(show, traced.tree.root)
    @test traced.trace.run_id == "trace-consistency"

    unify_ok = count(ev -> ev.event_kind === :unify_ok, traced.trace.events)
    unify_err = count(ev -> ev.event_kind === :unify_err, traced.trace.events)
    rollbacks = count(ev -> ev.event_kind === :rollback, traced.trace.events)
    @test unify_ok + unify_err == traced.stats.unify_calls
    @test unify_err == traced.stats.unify_failures
    @test rollbacks == traced.stats.rollback_calls
end

@testitem "GenerateTree: pretty summaries are stable" setup=[GTBaseImports] begin
    goal = IntTy
    r = generate_tree(goal, 8; max_height=7, max_nodes=29, return_stats=true, trace=true, trace_run_id="pretty-fixed")
    @test !haskey(r, :error)
    summary = LambdaRegression.GenerateTree.trace_run_summary(r.trace, r.tree, r.stats; canonical_key="canon-demo", elapsed_ns=123456)
    hot = LambdaRegression.GenerateTree.trace_hot_summary(r.trace; topk=3)
    @test occursin("run=pretty-fixed", summary)
    @test occursin("canonical=", summary)
    @test occursin("hottest_states=", hot)
    @test occursin("failure_reasons=", hot)
end

@testitem "GenerateTree: COND over-apply (k>=4) reachable" setup=[GTBaseImports] begin
    consts = Dict{Symbol,Any}(
        :COND => LambdaRegression.TypeSystem.COND,
        :btrue => true,
        :bfalse => false,
        :zero => 0,
        :one => 1,
    )
    tyctx = get_type_ctx(consts)
    rng = Random.MersenneTwister(2000)
    factory = make_tyvar_factory()
    ctx = LambdaRegression.GenerateTree.GenCtx(consts, tyctx, Function[], Function[], LambdaRegression.ASTs.Contexts.ConstSymbolFactory(prefix=:c), LambdaRegression.TypeSystem.typectx_fingerprint(tyctx), Dict{Any,Bool}(), Dict{Any,UInt64}(), nothing)
    options = LambdaRegression.GenerateTree.GenOptions(80, 16, true, true, 1.0, :uniform, 2.0, false, :bundle, false, true, true)
    trail = LambdaRegression.TypeSystem.TyTrail()
    namegen = SymbolGenerator()
    budget = LambdaRegression.GenerateTree.GenBudget(45, 12)

    t = LambdaRegression.GenerateTree._gen_elim_forced(rng, IntTy, Ty[], budget, ctx, options, trail, factory, namegen, nothing, :COND, 4; arg_retries=25)
    @test !(t isa Err)
    @test LambdaRegression.GenerateTree._contains_head_spine(t.root, :COND; min_k=4)
    @test iscompatible(infer(t, ctx.consts_ty), IntTy)
end

@testitem "GenerateTree: FST over-apply (k>1) reachable" setup=[GTBaseImports] begin
    consts = Dict{Symbol,Any}(
        :FST => LambdaRegression.TypeSystem.FST,
        :PAIR => LambdaRegression.TypeSystem.PAIR,
        :zero => 0,
        :one => 1,
        :bfalse => false,
    )
    tyctx = get_type_ctx(consts)
    rng = Random.MersenneTwister(2000)
    factory = make_tyvar_factory()
    ctx = LambdaRegression.GenerateTree.GenCtx(consts, tyctx, Function[], Function[], LambdaRegression.ASTs.Contexts.ConstSymbolFactory(prefix=:c), LambdaRegression.TypeSystem.typectx_fingerprint(tyctx), Dict{Any,Bool}(), Dict{Any,UInt64}(), nothing)
    options = LambdaRegression.GenerateTree.GenOptions(80, 18, true, true, 1.0, :uniform, 2.0, false, :bundle, false, true, true)
    trail = LambdaRegression.TypeSystem.TyTrail()
    namegen = SymbolGenerator()
    budget = LambdaRegression.GenerateTree.GenBudget(45, 12)

    t = LambdaRegression.GenerateTree._gen_elim_forced(rng, IntTy, Ty[], budget, ctx, options, trail, factory, namegen, nothing, :FST, 2; arg_retries=25)
    @test !(t isa Err)
    @test LambdaRegression.GenerateTree._contains_head_spine(t.root, :FST; min_k=2)
    @test iscompatible(infer(t, ctx.consts_ty), IntTy)
end

function _type_contains_mu(t::Ty)::Bool
    t = prune(t)
    if t isa TyMu
        return true
    elseif t isa TyFun
        return _type_contains_mu((t::TyFun).arg) || _type_contains_mu((t::TyFun).res)
    elseif t isa TyProd
        return _type_contains_mu((t::TyProd).left) || _type_contains_mu((t::TyProd).right)
    elseif t isa TySum
        return _type_contains_mu((t::TySum).left) || _type_contains_mu((t::TySum).right)
    else
        return false
    end
end

@testitem "GenerateTree: μ-type via forced self-application core path" setup=[GTBaseImports] begin
    # Deterministic coverage for μ-types: build λx. x x using generator core (Index head + k=1).
    rng = Random.MersenneTwister(2000)
    factory = make_tyvar_factory()

    ctx = LambdaRegression.GenerateTree.GenCtx(
        Dict{Symbol,Any}(),
        Dict{Symbol,TyLike}(),
        Function[],
        Function[],
        LambdaRegression.ASTs.Contexts.ConstSymbolFactory(prefix=:c),
        UInt(0),
        Dict{Any,Bool}(),
        Dict{Any,UInt64}(),
        nothing,
    )
    options = LambdaRegression.GenerateTree.GenOptions(7, 5, true, true, 1.0, :boltzmann, 2.0, false, :bundle, false, true, true)
    trail = LambdaRegression.TypeSystem.TyTrail()
    namegen = SymbolGenerator()

    # Create the λ binder type via shape unification.
    α = factory(0)
    β = factory(0)
    goal = TyFun(α, β)
    env = Ty[]
    push!(env, α)

    # Force body = Index0 Index0 by forcing Index head, k=1, and forcing the single arg to be a leaf.
    body = LambdaRegression.GenerateTree._gen_elim_forced_index(
        rng,
        β,
        env,
        LambdaRegression.GenerateTree.GenBudget(3, 3),
        ctx,
        options,
        trail,
        factory,
        namegen,
        nothing,
        0,
        1;
        node_budgets_override=[1],
        arg_retries=1,
    )
    @test !(body isa Err)

    tree = AstTree(AbsNode(:x, body.root))
    ty = infer(tree, ctx.consts_ty)
    @test !(ty isa Err)
    @test LambdaRegression.GenerateTree._type_contains_mu(ty)
end

@testitem "GenerateTree: uninhabited target fails fast (no hang)" setup=[GTBaseImports] begin
    # With no constants and no intro, base types are uninhabited.
    r = generate_tree(IntTy, 0; consts=Dict{Symbol,Any}(), consts_ty=Dict{Symbol,TyLike}(), allow_intro=false, max_height=4)
    @test r isa Err
end

@testitem "GenerateTree: Feasibility/Count consistency (small, over-approx)" setup=[GTBaseImports] begin
    ctx = LambdaRegression.GenerateTree.GenCtx(
        Dict{Symbol,Any}(),
        Dict{Symbol,TyLike}(),
        Function[],
        Function[],
        LambdaRegression.ASTs.Contexts.ConstSymbolFactory(prefix=:c),
        UInt(0),
        Dict{Any,Bool}(),
        Dict{Any,UInt64}(),
        nothing,
    )
    factory = make_tyvar_factory()
    budget = LambdaRegression.GenerateTree.GenBudget(3, 3)

    opts_no_intro = LambdaRegression.GenerateTree.GenOptions(7, 5, false, true, 0.5, :boltzmann, 2.0, false, :bundle, false, true, true)
    feasible1 = LambdaRegression.GenerateTree._is_feasible(IntTy, Ty[], budget, ctx, opts_no_intro, factory)
    cnt1 = LambdaRegression.GenerateTree._count_elim_root(IntTy, Ty[], budget, ctx, opts_no_intro, factory)
    @test feasible1 == false
    @test cnt1 == 0

    α = factory(0)
    β = factory(0)
    goal2 = TyFun(α, β)
    opts_intro = LambdaRegression.GenerateTree.GenOptions(7, 5, true, true, 0.5, :boltzmann, 2.0, false, :bundle, false, true, true)
    feasible2 = LambdaRegression.GenerateTree._is_feasible(goal2, Ty[], budget, ctx, opts_intro, factory)
    cnt2 = LambdaRegression.GenerateTree._count_intro_root(goal2, Ty[], budget, ctx, opts_intro, factory)
    @test feasible2 == true
    @test cnt2 > 0
end

@testitem "GenerateTree: feasibility pruning aliases stay compatible" setup=[GTBaseImports] begin
    r_new = generate_tree(IntTy, 8; max_height=6, max_nodes=25, return_stats=true, trace=false, use_feasibility_pruning=true)
    r_old = generate_tree(IntTy, 8; max_height=6, max_nodes=25, return_stats=true, trace=false, use_sat_pruning=true)
    @test !(r_new isa Err)
    @test !(r_old isa Err)
    @test sprint(show, r_new.tree.root) == sprint(show, r_old.tree.root)
    @test r_new.stats.unify_calls == r_old.stats.unify_calls

    r_alias_off = generate_tree(IntTy, 8; max_height=6, max_nodes=25, return_stats=true, trace=false, use_feasibility_pruning=true, use_sat_pruning=false)
    r_explicit_off = generate_tree(IntTy, 8; max_height=6, max_nodes=25, return_stats=true, trace=false, use_feasibility_pruning=false)
    @test !(r_alias_off isa Err)
    @test !(r_explicit_off isa Err)
    @test sprint(show, r_alias_off.tree.root) == sprint(show, r_explicit_off.tree.root)
end

@testitem "GenerateTree: feasibility pruning defaults on" setup=[GTBaseImports] begin
    r_default = generate_tree(IntTy, 8; max_height=7, max_nodes=29, return_stats=true, trace=false)
    r_explicit = generate_tree(IntTy, 8; max_height=7, max_nodes=29, return_stats=true, trace=false, use_feasibility_pruning=true)
    r_bundle = generate_tree(IntTy, 8; max_height=7, max_nodes=29, return_stats=true, trace=false, use_feasibility_pruning=true, feasibility_cache_mode=:bundle)
    @test !(r_default isa Err)
    @test !(r_explicit isa Err)
    @test !(r_bundle isa Err)
    @test sprint(show, r_default.tree.root) == sprint(show, r_explicit.tree.root)
    @test sprint(show, r_default.tree.root) == sprint(show, r_bundle.tree.root)
    @test r_default.stats.unify_calls == r_explicit.stats.unify_calls
    @test r_default.stats.unify_calls == r_bundle.stats.unify_calls
end

end # module GenerateTree
