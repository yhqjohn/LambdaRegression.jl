"""
    Regression

Task, evaluator, and resumable population-search interfaces.

This module defines task/evaluation APIs and composable components for age-based
search over typed lambda-calculus programs. It does not own a monolithic search
loop; callers use the resumable age-search operations directly or compose them
through `Workflows`.

Design goals
- High composability: configuration is a `NamedTuple` and components are injected
  as a `NamedTuple`.
- Concurrency readiness: SR/regrssion paths must not depend on global factories
  (e.g. shared `newtyvar` counters or `gensym` naming).
- Const injection: inputs and intermediate values flow through a constant pool
  whose types are visible to individuals.

Capabilities
- `default_components` supplies bounded FunctionDef-aware initialization, native
  FunctionDef evaluation, common-analysis-backed typed mutations (including
  FunctionDef operators), and automatic core/FunctionDef crossover dispatch.
  Initialization and generator-backed variation share the configured canonical
  generation providers and the general FunctionDef mutation generator.
  Initialization can softly mix its existing nearest-target candidate choice
  with uniform choice among successful candidates.
- `full_mutation_provider` optionally replaces complete-program mutation through
  a checked provider/mixture/codec interface; local scalar edits and finite
  generation do not invoke it. `schedule_budget`/`schedule_offset` define its
  cumulative proposal horizon, also used by scheduled mutation-family weights.
- `prepare_regression_evaluator` extracts only the task's execution inputs and
  explicit limits, then delegates preparation to `Execution`.
- `prepare_task_evaluator` resolves an explicit preparer, an injected sample
  evaluator, or the default Execution path once per candidate. `score_samples`
  owns the retained age-search per-sample loss loop while making
  their continue-with-penalty and stop-with-error policies explicit.
- `initialize_age_search`, `advance_age_search!`, and `inject_age_migrants!`
  provide a bounded resumable age-based complete-AST population slice; they
  preserve their own state and account for initialization, evolution, and
  migration objective work separately.
- `search_typed_roots` and `search_typed_hole` provide a bounded typed-island
  bridge that exports and rechecks complete ASTs for every task score.
- `initialize_age_search`, `advance_age_search!`, and
  `inject_age_migrants!` own the retained resumable search lifecycle; the
  `Workflows` module composes these operations for mixed phases.
- `prepare_scalar_region` and `search_scalar_region` expose the automatic
  scalar-region entry while keeping complete ASTs authoritative.
- `initialize_joint_search`, `advance_joint_search!`, `inject_joint_programs!`
  and `joint_search_result` evolve complete multi-region configurations using
  the regularized maintenance engine, atomic region exchange and complete-task
  scalar-constant adaptation. Capacity is per task, not per hole or template.
- Thread-safe primitives for SR paths live in `Execution.EvalContext` and
  `TypeSystem.make_tyvar_factory`.
"""
module Regression

using Random
import Serialization
using TestItems

import Base.Threads

using ..Utils: Err
using ..Schedules: ProposalProgress, schedule_value
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, TyFun, TyArray, TyNat, FloatTy
using ..TypeSystem: make_tyvar_factory, get_type_ctx
using ..Programs: ConstSymbolFactory
using ..Execution:
    EvalContext, MaxReduceStepsErr, ClosureDepthLimitErr,
    NonConstOutputErr, MissingConstValueErr,
    NonFiniteOutputErr, _finite_prediction
import ..Execution
using ..Primitives
using ..ASTs.NodeASTs:
    AstTree, AstNode, AppNode, AbsNode, ConstNode, FunctionDefNode, IndexNode, SymbolGenerator
using ..CanonicalGeneration
using ..ProgramAnalysis
using ..Mutations
using ..Mutations: Populations
using ..Programs: Program, _publish_program
using ..Programs: with_constants

export Regressor, SupervisedDataset
export default_config, default_components, default_regressor
export prepare_regression_evaluator, prepare_task_evaluator, score_samples
export TensorSchema, TensorSeries, SeriesTask, series_task
export series_mse, series_components, series_regressor
export AbstractConstantSearch, ConstantSearchBudget, ConstantSearchStats
export ConstantSearchResult, GaussianConstantSearch, search_constants
export AgeSearchState, AgeSearchIndividual, AgeSearchMigration
export AgeSearchCounters, AgeSearchReport, AgeSearchNoValidErr
export initialize_age_search, advance_age_search!, inject_age_migrants!
export TypedRefinementError, TypedRefinementResult
export search_typed_roots, search_typed_hole

struct NotImplementedErr end
struct MissingComponentErr
    key::Symbol
end

include("mutation_providers.jl")

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
    issue isa ClosureDepthLimitErr && return true
    issue isa NonConstOutputErr && return true
    issue isa MissingConstValueErr && return true
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

"""Assemble default Execution values from Regression components.

This is the default branch used by `prepare_task_evaluator`. The resulting
Execution object stores only the program and values needed to execute it.
"""
function prepare_regression_evaluator(
    program::Program,
    components::NamedTuple,
    config::NamedTuple;
    copy_tree::Bool=true,
)
    max_steps = get(config, :max_reduce_steps, 64)
    return Execution.prepare_default_evaluator(
        program;
        input_slots=components.input_slots,
        global_consts=components.global_consts,
        make_eval_context=components.make_eval_context,
        const_symfactory_ctor=components.const_symfactory_ctor,
        max_steps=get(config, :procedural_max_steps, max(256, 4 * max_steps)),
        max_call_depth=get(config, :closure_max_depth, 512),
        copy_tree,
    )
end

"""Prepare one candidate evaluator from Regression task components.

An explicit `components.prepare_evaluate` takes precedence and receives
`(program, target_ty, components, config; copy_tree)`. Otherwise an explicit
`components.evaluate` is wrapped as a per-sample callable; only when neither is
present is the default Execution evaluator prepared. This presence-based
dispatch ensures a custom evaluator is never hidden by the default path.
"""
function prepare_task_evaluator(
    program::Program,
    target_ty::Ty,
    components::NamedTuple,
    config::NamedTuple;
    copy_tree::Bool=true,
)
    if haskey(components, :prepare_evaluate)
        return components.prepare_evaluate(
            program, target_ty, components, config; copy_tree,
        )
    elseif haskey(components, :evaluate)
        evaluate = components.evaluate
        return x -> evaluate(program, x, target_ty, components, config)
    end
    return prepare_regression_evaluator(program, components, config; copy_tree)
end

"""Score the samples of one supervised task with an already prepared evaluator.

`failure_mode=:continue_penalty` adds the finite non-negative penalty for each
expected failed sample and continues; `:return_error` stops at the first
expected failure and returns it in `score`. Unexpected thrown evaluator/loss
errors propagate. A returned `Err` is penalized in continue mode unless strict
returned-error handling rejects an unclassified failure. `on_samples`, when
provided, is called once with the exact number of attempted samples, including
when an unexpected exception propagates. `strict_loss_conversion=true` keeps
the direct `Float64` conversion behavior of the retained scorer;
age search instead treats conversion failure as a penalized bad loss.
"""
function score_samples(
    data::SupervisedDataset,
    evaluator,
    loss_fn;
    failure_mode::Symbol,
    failure_penalty::Real=0.0,
    candidate_failure=_known_candidate_exception,
    loss_failure=_no_candidate_exception,
    strict_returned_errors::Bool=false,
    strict_loss_conversion::Bool=false,
    on_samples=nothing,
)
    failure_mode in (:continue_penalty, :return_error) || throw(ArgumentError(
        "failure_mode must be :continue_penalty or :return_error",
    ))
    penalty = Float64(failure_penalty)
    isfinite(penalty) && penalty >= 0 || throw(ArgumentError(
        "failure_penalty must be finite and non-negative",
    ))

    total = 0.0
    valid = true
    sample_count = 0
    try
        for (x, y) in zip(data.X, data.Y)
            sample_count += 1
            yhat = try
                evaluator(x)
            catch issue
                candidate_failure(issue) || rethrow()
                if failure_mode === :return_error
                    return (; score=Err(issue), valid=false,
                        samples=sample_count)
                end
                valid = false
                total += penalty
                continue
            end

            if yhat isa Err
                issue = yhat.value
                strict_returned_errors && !candidate_failure(issue) && throw(issue)
                if failure_mode === :return_error
                    return (; score=yhat, valid=false,
                        samples=sample_count)
                end
                valid = false
                total += penalty
                continue
            elseif !_finite_prediction(yhat)
                issue = NonFiniteOutputErr()
                if failure_mode === :return_error
                    return (; score=Err(issue), valid=false,
                        samples=sample_count)
                end
                valid = false
                total += penalty
                continue
            end

            sample_loss = try
                loss_fn(yhat, y)
            catch issue
                loss_failure(issue) || rethrow()
                if failure_mode === :return_error
                    return (; score=Err(issue), valid=false,
                        samples=sample_count)
                end
                valid = false
                total += penalty
                continue
            end
            finite_loss = if strict_loss_conversion
                sample_loss isa Real && isfinite(sample_loss) ?
                    Float64(sample_loss) : nothing
            else
                _finite_real_loss(sample_loss)
            end
            if finite_loss === nothing
                issue = NonFiniteOutputErr()
                if failure_mode === :return_error
                    return (; score=Err(issue), valid=false,
                        samples=sample_count)
                end
                valid = false
                total += penalty
            else
                total += finite_loss
            end
        end
    finally
        on_samples === nothing || on_samples(sample_count)
    end

    score = total / max(length(data.X), 1)
    if !isfinite(score)
        issue = NonFiniteOutputErr()
        if failure_mode === :return_error
            return (; score=Err(issue), valid=false,
                samples=sample_count)
        end
        return (; score=penalty, valid=false,
            samples=sample_count)
    end
    return (; score=Float64(score), valid, samples=sample_count)
end

"""Prepare the default evaluator through Regression's task-facing boundary.

Current task callers should use `prepare_task_evaluator` or the shared
`score_samples` path; direct Execution callers should use
`Execution.evaluate_default` with explicit execution keywords.
The wrapper preserves Regression's existing non-finite-output `Err` result.
"""
function evaluate_default(
    program::Program,
    x,
    _target_ty::Ty,
    components::NamedTuple,
    config::NamedTuple,
)
    return prepare_regression_evaluator(
        program, components, config; copy_tree=false,
    )(x)
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
    role === :init_functiondef_root && return 0x08
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

function _select_initial_candidate_index(
    candidate_sizes::AbstractVector{<:Integer},
    target_nodes::Int,
    preference::Float64,
    rng::Union{Nothing,AbstractRNG},
)
    nearest_target_index() = argmin([
        (abs(size - target_nodes), -size)
        for size in candidate_sizes
    ])
    preference == 1 && return nearest_target_index()
    rng === nothing && throw(ArgumentError(
        "non-default initialization size preference requires a selection RNG",
    ))
    preference == 0 && return rand(rng, eachindex(candidate_sizes))
    return rand(rng) < preference ?
        nearest_target_index() :
        rand(rng, eachindex(candidate_sizes))
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


function _validate_program_contract(
    program,
    target_ty::Ty,
    components::NamedTuple,
    label::AbstractString;
    check_type::Bool=true,
)
    program isa Program || return ArgumentError("$label is not a Program")
    Set(keys(program.local_consts)) == Set(keys(program.local_consts_ty)) ||
        return ArgumentError("$label local value/type keys differ")
    reserved = union(Set(keys(components.global_consts)), Set(components.input_slots))
    isempty(intersect(Set(keys(program.local_consts)), reserved)) ||
        return ArgumentError("$label local constants collide with global or input names")
    for node in program.tree
        node isa ConstNode || continue
        name = node.name
        haskey(components.global_consts, name) ||
            haskey(program.local_consts, name) ||
            name in components.input_slots ||
            return ArgumentError("$label references unresolved constant :$name")
    end
    if check_type
        ctx = merge(components.global_consts_ty, components.input_types, program.local_consts_ty)
        query_ctx, detached_targets = TypeSystem._detached_type_inputs(ctx, target_ty)
        query_target = only(detached_targets)
        factory = components.tyfactory_ctor()
        inferred = TypeSystem.infer_term(program.tree, query_ctx; factory)
        inferred isa Err &&
            return ArgumentError("$label type inference returned Err($(repr(inferred.value)))")
        inferred isa TyLike ||
            return ArgumentError("$label type inference returned a non-type")
        TypeSystem.iscompatible(inferred, query_target) ||
            return ArgumentError("$label inferred type is incompatible with target")
    end
    return nothing
end

"""Whether a mutation driver verifies every successful final tree against its
target and detached constant-type context before returning it. Unknown drivers
are unchecked; a wrapper may forward this contract only when it preserves the
same final tree and declarations without editing them afterward.
"""
_mutation_result_typechecked(::Any) = false
_mutation_result_typechecked(::typeof(Mutations.mutate_tree)) = true

function _preflight_initial_population(pop, reg::Regressor, components::NamedTuple, N::Int)
    pop isa AbstractVector || return ArgumentError("init_population must return a vector")
    length(pop) == N ||
        return ArgumentError("init_population must return exactly N individuals")
    policy = get(components, :bloat_policy, Mutations.BloatPolicy())
    for (i, ind) in enumerate(pop)
        issue = _validate_program_contract(ind, reg.target_ty, components, "initial program $i")
        issue === nothing || return issue
        _within_bloat_limit(ind.tree, policy) ||
            return ArgumentError("initial individual $i exceeds the configured structural limit")
    end
    return nothing
end

"""
    default_config(; ...) -> NamedTuple

Return defaults for bounded age search, initialization, and evaluation.
"""
function default_config(;
    population_size::Int=64,
    max_reduce_steps::Int=64,
    procedural_max_steps::Int=max(256, 4 * max_reduce_steps),
    closure_max_depth::Int=512,
    tournament_size::Int=3,
    error_penalty::Float64=1e12,
    strict_errors::Bool=false,
    parallel=(; enabled=false),
)
    parallel = _normalize_parallel_config(parallel)
    base = (;
        population_size,
        max_reduce_steps,
        procedural_max_steps,
        closure_max_depth,
        tournament_size,
        error_penalty,
        strict_errors,
        parallel,
    )
    return base
end

function _normalize_parallel_config(parallel::NamedTuple)
    unknown = filter(key -> key !== :enabled, keys(parallel))
    isempty(unknown) || throw(ArgumentError(
        "unknown parallel config option(s): $(join(string.(unknown), ", "))",
    ))
    return merge((; enabled=false), parallel)
end

function _merge_config_overrides(config::NamedTuple, overrides::NamedTuple)
    unknown = filter(key -> !hasproperty(config, key), keys(overrides))
    isempty(unknown) || throw(ArgumentError(
        "unknown regressor config override(s): $(join(string.(unknown), ", "))",
    ))
    result = merge(config, overrides)
    section = _normalize_parallel_config(get(config, :parallel, (;)))
    if haskey(overrides, :parallel)
        section = merge(section, _normalize_parallel_config(overrides.parallel))
    end
    return merge(result, (; parallel=section))
end


_procedural_candidate_admissibility(ind::Program) =
    obvious_functiondef_loop_reason(ind)

function _generation_providers_with_functiondefs(providers)
    config = CanonicalGeneration.GenerationConfig(providers=Tuple(providers))
    return CanonicalGeneration.with_functiondef_provider(config).providers
end

function _program_candidate_rejection(ind::Program, components::NamedTuple)
    filter = get(components, :candidate_admissibility, _ -> nothing)
    applicable(filter, ind) || throw(ArgumentError(
        "candidate_admissibility must accept one Program",
    ))
    reason = filter(ind)
    reason === nothing && return nothing
    reason isa Symbol || throw(ArgumentError(
        "candidate_admissibility must return nothing or a Symbol reason",
    ))
    return reason
end

_fold_step_joint_weight_kwargs(probability::Real) = probability == 0 ?
    NamedTuple() : (; fold_step_joint_use_probability=probability)

_list_state_helper_weight_kwargs(weight::Real) = weight == 0 ?
    NamedTuple() : (; list_state_helper=weight)

function _checked_nonnegative_generation_weight(name::Symbol, value::Real)
    value isa Bool && throw(ArgumentError("$name must be finite and non-negative"))
    weight = Float64(value)
    isfinite(weight) && weight >= 0 || throw(ArgumentError(
        "$name must be finite and non-negative"))
    return weight
end

"""
    default_components(; kwargs...) -> NamedTuple

Build a task's typed primitive context and composable components for bounded
initialization, variation, candidate admission, and evaluation.

The shared `generation_providers` world configures initialization and
generator-backed replacements. Defaults include the safe-list generation
provider and the general FunctionDef mutation generator. The same configured
generation world is used across target types; it does not install task-specific
recursive templates. `candidate_admissibility` applies a cheap AST-level
rejection filter; it is not a typing check or a termination proof.

Initialization collects the existing bounded number of successful candidates.
`init_size_preference` is the probability of selecting the candidate nearest
the slot's target node count (ties still prefer the larger candidate); on the
remaining probability it chooses uniformly from those successful candidates.
The default `1` preserves the previous choice without consuming selection RNG.
This only softens the choice within the successful candidates from the current
initialization call; it does not retain the original candidate pool afterward.

`init_functiondef_root_probability` optionally mixes a genuinely self-calling
FunctionDef root into function-typed slots. One independent per-slot draw
selects this route; zero consumes no draw and preserves ordinary initialization.
The selected route spends the existing first initialization attempt, forwarding
the configured provider and generation config. A successful root is selected
directly, without competing against smaller ordinary candidates. On failure,
remaining attempts collect ordinary candidates under the existing size policy;
the failed attempt is not refunded. Values must be finite in `[0,1)` and a
nonzero value requires the default `init_population`. It is a proposal prior,
not a promise of successful recursion, termination, or a final AST size bound.

Initialization growth uses the configured unary composition, binary
composition, and contextual expansion operators. The returned NamedTuple is
a loader result: it runs no business logic and can be customized with
`merge`. AgeSearch consumes `crossover_op` through one directed
`Populations.recombine` call per proposal, so the configured operator must
support that seam. This is local to AgeSearch; it does not make `recombine` a
requirement of every `Populations.AbstractCrossover`, which also exposes the
independent two-child `crossover` interface.

An optional `initialization_candidate_snapshot` callback can capture successful
pre-growth candidates when `initialization_observer` is also supplied. It runs
synchronously with a read-only borrowed `(tree, local_consts, local_consts_ty)`
view, may run concurrently by slot during parallel initialization, must not use
search RNGs or mutate the borrowed materials, must be thread-safe, and must
return a detached snapshot. The existing observer still receives the completed
per-slot diagnostics serially after initialization joins.

`fold_step_joint_use_probability` is forwarded to the default generation
configuration and generator-backed mutation weights. It defaults to zero and
must be finite in `[0,1)`. It biases bounded AFOLDL callback proposals only;
it does not alter fold selection weights. A supplied custom
`generation_config_factory` or mutation list remains caller-owned.

`list_state_helper` is forwarded as a non-negative finite
`GenerationWeights` weight to the default generation configuration and
generator-backed mutation weights. It defaults to zero and is a weight, not a
probability. It biases bounded list-state helper derivations. A supplied custom
`generation_config_factory` or mutation list remains caller-owned.
"""
function default_components(;
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    input_slots::Vector{Symbol}=Symbol[],
    input_types::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    functiondef_generation_weight::Float64=1.0,
    fold_step_joint_use_probability::Real=0.0,
    list_state_helper::Real=0.0,
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
    candidate_admissibility=_procedural_candidate_admissibility,
    functiondef_provider=CanonicalGeneration.FunctionDefProvider(
        admissibility=CanonicalGeneration.DirectKnotFilter(),
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
                CanonicalGeneration.generate_canonical_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=_generation_providers_with_functiondefs(generation_providers),
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    function_def=functiondef_generation_weight,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
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
                CanonicalGeneration.generate_canonical_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=_generation_providers_with_functiondefs(generation_providers),
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    function_def=functiondef_generation_weight,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
                ),
                sampling_depth_limit=3,
                sampling_node_budget=48,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.BinaryCompositionMutation(;
            site_selection=binary_composition_site_selection,
            replacement_generator=Mutations.ScopedReplacementGenerator(
                CanonicalGeneration.generate_canonical_scoped;
                include_local_consts=true,
            ),
            generation_kwargs=(;
                providers=_generation_providers_with_functiondefs(generation_providers),
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    function_def=functiondef_generation_weight,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
                ),
                sampling_depth_limit=3,
                sampling_node_budget=48,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.FunctionDefMutation(
            :finite_body;
            functiondef_provider,
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
                ),
                sampling_depth_limit=4,
                sampling_node_budget=64,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.FunctionDefMutation(
            :recursive_body;
            functiondef_provider,
            site_weight=(mode, site, env, ctx) ->
                functiondef_recursive_body_site_weight,
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
                ),
                sampling_depth_limit=8,
                sampling_node_budget=96,
                guarded_expansion_depth=1,
            ),
        ),
        Mutations.FunctionDefMutation(
            :remove;
            functiondef_provider,
            generation_kwargs=(;
                providers=generation_providers,
                weights=CanonicalGeneration.GenerationWeights(;
                    default=1,
                    _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
                    _list_state_helper_weight_kwargs(list_state_helper)...,
                ),
                sampling_depth_limit=4,
                sampling_node_budget=64,
                guarded_expansion_depth=1,
            ),
        ),
    ],
    mutation_chooser::Mutations.AbstractPairChooser=Mutations.WeightedPairChooser(),
    mutation_provider::Mutations.AbstractSiteProvider=Mutations.IndexedProvider(),
    full_mutation_provider::AbstractMutationProvider=OrdinaryMutationProvider(),
    schedule_budget::Int=1,
    schedule_offset::Int=0,
    mutation_max_try::Int=25,
    mutation_prune_consts::Bool=true,
    mutation_adaptive_policy::Mutations.AdaptivePolicy=Mutations.AdaptivePolicy(),
    bloat_policy::Mutations.BloatPolicy=Mutations.BloatPolicy(),
    candidate_failure=_known_candidate_exception,
    loss_failure=_no_candidate_exception,
    crossover_op=Populations.TypedSubtreeSwapCrossover(require_same_depth=false),
    mutation_driver=Mutations.mutate_tree,
    generation_config_factory=rng -> CanonicalGeneration.GenerationConfig(;
        rng,
        providers=generation_providers,
        weights=CanonicalGeneration.GenerationWeights(;
            default=1,
            function_def=functiondef_generation_weight,
            _fold_step_joint_weight_kwargs(fold_step_joint_use_probability)...,
            _list_state_helper_weight_kwargs(list_state_helper)...,
        ),
        sampling_depth_limit=8,
        sampling_node_budget=96,
        guarded_expansion_depth=1,
    ),
    force_lambda_root_probability::Float64=0.5,
    init_functiondef_root_probability::Real=0,
    init_candidates_per_individual::Int=2,
    init_target_nodes::Tuple{Vararg{Int}}=(3, 5, 8, 12),
    init_size_preference::Real=1,
    init_max_try::Int=8,
    init_growth_attempts::Int=2,
    init_growth_max_try::Int=8,
    init_growth_mutation=nothing,
    init_population=nothing,
    initialization_observer=nothing,
    initialization_candidate_snapshot=nothing,
    kwargs...
)
    ProposalProgress(schedule_offset, schedule_budget)
    haskey(kwargs, :type_inference) && throw(ArgumentError(
        "type_inference was removed; ProgramAnalysis owns mutation typing and sites",
    ))
    global_consts_ty === nothing && (global_consts_ty = get_type_ctx(global_consts))
    isfinite(functiondef_generation_weight) && functiondef_generation_weight >= 0 ||
        throw(ArgumentError(
            "functiondef_generation_weight must be finite and non-negative",
        ))
    fold_step_joint_use_probability isa Bool && throw(ArgumentError(
        "fold_step_joint_use_probability must be finite and in [0, 1)",
    ))
    isfinite(fold_step_joint_use_probability) &&
        0 <= fold_step_joint_use_probability < 1 || throw(ArgumentError(
            "fold_step_joint_use_probability must be finite and in [0, 1)",
        ))
    fold_step_joint_use_probability = Float64(fold_step_joint_use_probability)
    list_state_helper = _checked_nonnegative_generation_weight(
        :list_state_helper, list_state_helper)
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
    init_size_preference isa Bool && throw(ArgumentError(
        "init_size_preference must be a finite real number in [0, 1]",
    ))
    isfinite(init_size_preference) && 0 <= init_size_preference <= 1 ||
        throw(ArgumentError(
            "init_size_preference must be a finite real number in [0, 1]",
        ))
    init_size_preference = Float64(init_size_preference)
    init_functiondef_root_probability isa Bool && throw(ArgumentError(
        "init_functiondef_root_probability must be a finite real number in [0, 1)",
    ))
    isfinite(init_functiondef_root_probability) &&
        0 <= init_functiondef_root_probability < 1 || throw(ArgumentError(
            "init_functiondef_root_probability must be a finite real number in [0, 1)",
        ))
    init_functiondef_root_probability = Float64(init_functiondef_root_probability)
    init_functiondef_root_probability < 1 || throw(ArgumentError(
        "init_functiondef_root_probability must remain below 1 after Float64 conversion",
    ))
    init_functiondef_root_probability == 0 || init_population === nothing ||
        throw(ArgumentError(
            "init_functiondef_root_probability requires the default init_population; " *
            "custom initialization must implement its own route mixture",
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
    initialization_observer === nothing || applicable(initialization_observer, Any[]) ||
        throw(ArgumentError(
            "initialization_observer must accept the ordered diagnostics vector",
        ))
    initialization_candidate_snapshot === nothing || initialization_observer !== nothing ||
        throw(ArgumentError(
            "initialization_candidate_snapshot requires initialization_observer",
        ))

    # The observer is invoked only after serial/parallel initialization has
    # joined by slot. Generated metadata never enters a published Program
    # payload or a long-lived component cache.

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
                        global_consts,
                        global_consts_ty=merge(global_consts_ty, input_types),
                        local_consts=attempt_consts,
                        local_consts_ty=attempt_const_types,
                        tyfactory=tyfactory_ctor(),
                        symfactory=SymbolGenerator(),
                        const_symfactory=const_symfactory_ctor(),
                        bloat_policy,
                        adaptive_policy=mutation_adaptive_policy,
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
        effective_config = CanonicalGeneration.with_functiondef_provider(
            config,
            functiondef_provider,
        )
        result = CanonicalGeneration.generate_canonical_scoped(
            result_type;
            scope_tys=reverse(argument_types),
            global_consts,
            global_consts_ty,
            config=effective_config,
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

    function generate_individual(
        target_ty,
        seed::UInt64,
        slot::Int,
        attempt_offset::Int=0;
        return_diagnostics::Bool=false,
    )
        last_error = nothing
        candidates = CanonicalGeneration.GenerationResult[]
        candidate_attempts = initialization_candidate_snapshot === nothing ? nothing : Int[]
        prefer_functiondef_root = init_functiondef_root_probability > 0 &&
            TypeSystem.prune(target_ty) isa TypeSystem.TyFun &&
            rand(derive_rng(seed, :init_functiondef_root, slot, attempt_offset)) <
                init_functiondef_root_probability
        functiondef_root_applied = false
        generation_attempts = 0
        for attempt in 1:init_max_try
            generation_attempts = attempt
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
            preferred_attempt = prefer_functiondef_root && attempt == 1
            result = if preferred_attempt
                CanonicalGeneration.generate_functiondef_root_scoped(
                    target_ty;
                    global_consts,
                    global_consts_ty,
                    functiondef_provider,
                    config,
                )
            elseif TypeSystem.prune(target_ty) isa TypeSystem.TyFun &&
                        rand(rng) < force_lambda_root_probability
                generate_lambda_root(target_ty, rng, config)
            else
                CanonicalGeneration.generate_canonical(
                    target_ty;
                    global_consts,
                    global_consts_ty,
                    config=CanonicalGeneration.with_functiondef_provider(
                        config,
                        functiondef_provider,
                    ),
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
            candidate_attempts === nothing || push!(candidate_attempts, attempt)
            if preferred_attempt
                functiondef_root_applied = true
                break
            end
            length(candidates) >= init_candidates_per_individual && break
        end
        isempty(candidates) && return last_error === nothing ?
            Err(Mutations.MaxTryReachedErr()) : last_error
        target_nodes = init_target_nodes[mod1(slot, length(init_target_nodes))]
        candidate_sizes = _node_count.(getfield.(candidates, :tree))
        selection_rng = functiondef_root_applied || init_size_preference == 1 ? nothing : derive_rng(
            seed, :init_size_selection, slot, attempt_offset)
        selected_index = functiondef_root_applied ? 1 : _select_initial_candidate_index(
            candidate_sizes, target_nodes, init_size_preference, selection_rng)
        result = candidates[selected_index]
        result.metadata[:init_target_nodes] = target_nodes
        result.metadata[:init_candidate_count] = length(candidates)
        result.metadata[:init_size_preference] = init_size_preference
        result.metadata[:init_size_selection_strategy] =
            functiondef_root_applied ? :functiondef_root :
            init_size_preference == 1 ? :nearest_target :
            init_size_preference == 0 ? :uniform_successful_candidates :
            :nearest_target_uniform_successful_mixture
        result.metadata[:init_selected_index] = selected_index
        result.metadata[:init_selected_candidate_nodes] = candidate_sizes[selected_index]
        result.metadata[:init_selected_growth_before_nodes] = candidate_sizes[selected_index]
        if init_functiondef_root_probability > 0
            result.metadata[:init_functiondef_root_probability] = init_functiondef_root_probability
            result.metadata[:init_functiondef_root_requested] = prefer_functiondef_root
            result.metadata[:init_functiondef_root_attempted] = prefer_functiondef_root
            result.metadata[:init_functiondef_root_succeeded] = functiondef_root_applied
            result.metadata[:init_functiondef_root_fallback] = prefer_functiondef_root && !functiondef_root_applied
            result.metadata[:init_generation_attempts] = generation_attempts
        end
        if initialization_candidate_snapshot !== nothing
            snapshots = map(eachindex(candidates)) do index
                candidate = candidates[index]
                snapshot = initialization_candidate_snapshot((;
                    tree=candidate.tree,
                    local_consts=candidate.local_consts,
                    local_consts_ty=candidate.local_const_types,
                ))
                (; candidate_index=index,
                    successful_attempt=candidate_attempts[index],
                    node_count=candidate_sizes[index], snapshot)
            end
            result.metadata[:init_successful_attempts] = Tuple(candidate_attempts)
            result.metadata[:init_candidate_sizes] = Tuple(candidate_sizes)
            result.metadata[:init_candidate_snapshots] = Tuple(snapshots)
        end
        tree, local_consts, local_consts_ty = grow_initial_candidate(
            result,
            target_ty,
            seed,
            slot,
            attempt_offset,
            target_nodes,
        )
        if initialization_candidate_snapshot !== nothing
            result.metadata[:init_selected_growth_after_nodes] = result.metadata[:init_final_nodes]
        end
        program = _publish_program(tree, local_consts, local_consts_ty)
        return return_diagnostics ?
            (; program, diagnostics=Dict{Symbol,Any}(result.metadata)) : program
    end

    default_init_population = function (reg, data; N, seed, global_type_ctx)
        population = Vector{Program}(undef, N)
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
                    slot;
                    return_diagnostics=true,
                )
            end
        else
            for slot in 1:N
                generated_slots[slot] = generate_individual(
                    reg.target_ty,
                    UInt64(seed),
                    slot;
                    return_diagnostics=true,
                )
            end
        end
        diagnostics = Vector{Any}(undef, N)
        for slot in 1:N
            generated = generated_slots[slot]
            generated isa Err && return generated
            generated isa NamedTuple && hasproperty(generated, :program) ||
                throw(ArgumentError("default initialization diagnostics are malformed"))
            population[slot] = generated.program
            diagnostics[slot] = generated.diagnostics
        end
        initialization_observer === nothing || initialization_observer(copy(diagnostics))
        return population
    end

    init_population === nothing && (init_population = default_init_population)
    base = (;
        global_consts,
        global_consts_ty,
        input_slots,
        input_types,
        tyfactory_ctor,
        const_symfactory_ctor,
        functiondef_generation_weight,
        fold_step_joint_use_probability,
        list_state_helper,
        functiondef_recursive_body_site_weight,
        generation_providers,
        contextual_expansion_site_selection,
        binary_composition_site_selection,
        candidate_admissibility,
        make_eval_context,
        mutations,
        mutation_chooser,
        mutation_provider,
        full_mutation_provider,
        schedule_budget,
        schedule_offset,
        mutation_max_try,
        mutation_prune_consts,
        mutation_adaptive_policy,
        bloat_policy,
        candidate_failure,
        loss_failure,
        crossover_op,
        mutation_driver,
        functiondef_provider,
        generation_config_factory,
        force_lambda_root_probability,
        init_functiondef_root_probability,
        init_candidates_per_individual,
        init_target_nodes,
        init_size_preference,
        init_max_try,
        init_growth_attempts,
        init_growth_max_try,
        init_growth_mutation,
        init_population,
        initialization_observer,
        initialization_candidate_snapshot,
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

include("ConstantSearch.jl")
include("TaskSchemas.jl")
include("age_search.jl")
include("search_replay.jl")
include("scalar_refinement.jl")
include("joint_search.jl")
include("typed_refinement.jl")

@testsnippet RegBaseImports begin
    using LambdaRegression.Regression: default_components
    using LambdaRegression.Regression: prepare_regression_evaluator
    using LambdaRegression.TypeSystem: make_tyvar_factory
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.Regression: mse
    using Random
    using LambdaRegression.Mutations.Populations: Program
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike
    using LambdaRegression.Regression: evaluate_default, default_components, default_config
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.TypeSystem
    using LambdaRegression.Utils: Err
    using LambdaRegression.Mutations
    using LambdaRegression.Regression: Regressor, SupervisedDataset, evaluate_default, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.Regression: Regressor, SupervisedDataset, default_components, default_config
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike
    using LambdaRegression.Mutations: WeightedPairChooser
    using LambdaRegression.Execution:
        MaxReduceStepsErr,
        ClosureDepthLimitErr,
        NonConstOutputErr,
        MissingConstValueErr,
        NonFiniteOutputErr
end

@testitem "Regression task evaluator dispatch and shared sample scoring" setup=[RegBaseImports] begin
    using LambdaRegression.Regression:
        prepare_task_evaluator, score_samples, mse, default_config
    using LambdaRegression.Utils: Err

    data = SupervisedDataset([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    program = Program(AstTree(AbsNode(:x, IndexNode(0))))
    components = default_components(
        input_slots=[:x], input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        global_consts=Dict{Symbol,Any}(), global_consts_ty=Dict{Symbol,TyLike}(),
    )
    config = default_config()

    default_eval = prepare_task_evaluator(program, TyFun(FloatTy, FloatTy), components, config)
    @test default_eval(2.0) == 2.0

    custom_calls = Ref(0)
    custom = merge(components, (; evaluate=(candidate, x, ty, c, cfg) -> begin
        custom_calls[] += 1
        x + 4
    end))
    custom_eval = prepare_task_evaluator(program, TyFun(FloatTy, FloatTy), custom, config)
    @test custom_eval(2.0) == 6.0
    @test custom_calls[] == 1

    prepared_calls = Ref(0)
    explicitly_prepared = merge(custom, (; prepare_evaluate=(candidate, ty, c, cfg; copy_tree=true) -> begin
        prepared_calls[] += 1
        x -> x + 8
    end))
    prepared_eval = prepare_task_evaluator(program, TyFun(FloatTy, FloatTy), explicitly_prepared, config)
    @test prepared_eval(2.0) == 10.0
    @test prepared_calls[] == 1
    @test custom_calls[] == 1

    sample_updates = Ref(0)
    continuing = score_samples(data, x -> x == 2.0 ? Err(:invalid) : x, mse;
        failure_mode=:continue_penalty, failure_penalty=99.0,
        on_samples=n -> (sample_updates[] += n))
    @test continuing.valid == false
    @test continuing.samples == 3
    @test continuing.score == 33.0
    @test sample_updates[] == 3

    stopped_updates = Ref(0)
    stopping = score_samples(data, x -> x == 2.0 ? Err(ArgumentError("bad sample")) : x, mse;
        failure_mode=:return_error, candidate_failure=issue -> issue isa ArgumentError,
        strict_returned_errors=true,
        on_samples=n -> (stopped_updates[] += n))
    @test stopping.score isa Err
    @test stopping.samples == 2
    @test stopping.valid == false
    @test stopped_updates[] == 2

    unexpected_updates = Ref(0)
    @test_throws ErrorException score_samples(data, _ -> error("unexpected"), mse;
        failure_mode=:return_error,
        on_samples=n -> (unexpected_updates[] += n))
    @test unexpected_updates[] == 1

    loss_updates = Ref(0)
    loss_failure = score_samples(data, identity,
        (_, _) -> throw(DomainError(1.0, "loss domain"));
        failure_mode=:continue_penalty, failure_penalty=99.0,
        loss_failure=issue -> issue isa DomainError,
        on_samples=n -> (loss_updates[] += n))
    @test loss_failure.valid == false
    @test loss_failure.samples == 3
    @test loss_failure.score == 99.0
    @test loss_updates[] == 3

    nonfinite = score_samples(data, x -> x == 2.0 ? Inf : x, mse;
        failure_mode=:continue_penalty, failure_penalty=99.0)
    @test nonfinite.valid == false
    @test nonfinite.samples == 3
    @test nonfinite.score == 33.0

    @test_throws ArgumentError score_samples(data, identity, mse;
        failure_mode=:other)
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
        m -> m isa LambdaRegression.Mutations.FunctionDefMutation,
        c.mutations,
    ) == 3
    @test c.functiondef_provider.admissibility isa
        LambdaRegression.CanonicalGeneration.DirectKnotFilter
    functiondef_mutations = filter(
        m -> m isa LambdaRegression.Mutations.FunctionDefMutation,
        c.mutations,
    )
    @test all(
        mutation -> mutation.functiondef_provider === c.functiondef_provider,
        functiondef_mutations,
    )
    recursive_body = only(filter(functiondef_mutations) do mutation
        LambdaRegression.Mutations._mode(mutation) === :recursive_body
    end)
    @test recursive_body.site_weight_fn(:recursive_body, nothing, nothing, nothing) == 4.0
    @test recursive_body.replacement_generator isa
        LambdaRegression.Mutations.DefaultFunctionDefMutationGenerator
    @test recursive_body.generation_kwargs.sampling_depth_limit == 8
    @test recursive_body.generation_kwargs.sampling_node_budget == 96
    direct_knot = Program(AstTree(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
            :x,
            AppNode(IndexNode(1), IndexNode(0)),
        )),
    ))
    @test c.candidate_admissibility(direct_knot) === :eta_self_forwarding
    @test count(
        m -> m isa LambdaRegression.Mutations.ContextualExpansionMutation,
        c.mutations,
    ) == 1
    @test_throws ArgumentError default_components(type_inference=(args...) -> true)
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
    @test c.crossover_op isa LambdaRegression.Mutations.Populations.TypedSubtreeSwapCrossover
    @test !c.crossover_op.require_same_depth
    @test c.binary_composition_site_selection isa
        LambdaRegression.Mutations.LocalCallSiteSelectionPolicy
    @test c.binary_composition_site_selection.multiplier == 32.0
    safe_provider = LambdaRegression.CanonicalGeneration.SafeListAccessProvider
    @test any(provider -> provider isa safe_provider, c.generation_providers)
    generation_config = c.generation_config_factory(MersenneTwister(1))
    @test any(provider -> provider isa safe_provider, generation_config.providers)
    generator_backed = filter(c.mutations) do mutation
        mutation isa Union{
            LambdaRegression.Mutations.SubtreeReplacementMutation,
            LambdaRegression.Mutations.ContextualExpansionMutation,
            LambdaRegression.Mutations.BinaryCompositionMutation,
            LambdaRegression.Mutations.FunctionDefMutation,
        }
    end
    @test length(generator_backed) == 6
    @test all(generator_backed) do mutation
        any(provider -> provider isa safe_provider, mutation.generation_kwargs.providers)
    end
    cfg = default_config()
    @test cfg.population_size == 64
    @test cfg.tournament_size == 3
    @test cfg.error_penalty == 1e12
    @test cfg.parallel.enabled == false
    @test cfg.parallel == (enabled=false,)
    @test_throws ArgumentError default_config(
        parallel=(enabled=false, mode=:individual),
    )
    @test_throws ArgumentError LambdaRegression.Regression.default_regressor(
        FloatTy;
        config=merge(cfg, (; parallel=(enabled=false, schedule=:dynamic))),
    )
    @test !hasproperty(cfg, :generations)
    @test !hasproperty(cfg, :mutation_rate)
    @test !hasproperty(cfg, :repro)
    @test_throws MethodError default_config(generations=1)
end

@testitem "Regression default_components forwards fold-step generation bias" setup=[RegBaseImports] begin
    using Random
    CG = LambdaRegression.CanonicalGeneration
    generator_types = Union{
        LambdaRegression.Mutations.SubtreeReplacementMutation,
        LambdaRegression.Mutations.ContextualExpansionMutation,
        LambdaRegression.Mutations.BinaryCompositionMutation,
        LambdaRegression.Mutations.FunctionDefMutation,
    }

    neutral = default_components()
    neutral_weights = neutral.generation_config_factory(MersenneTwister(1)).weights
    @test get(neutral_weights.overrides, :fold_step_joint_use_probability, 0.0) == 0.0

    probability = 0.65
    biased = default_components(fold_step_joint_use_probability=probability)
    initial_weights = biased.generation_config_factory(MersenneTwister(1)).weights
    @test initial_weights.overrides[:fold_step_joint_use_probability] == probability
    generator_backed = filter(mutation -> mutation isa generator_types, biased.mutations)
    @test length(generator_backed) == 6
    @test all(generator_backed) do mutation
        mutation.generation_kwargs.weights.overrides[:fold_step_joint_use_probability] == probability
    end

    custom_factory = rng -> CG.GenerationConfig(; rng,
        weights=CG.GenerationWeights(fold_step_joint_use_probability=0.25))
    custom = default_components(fold_step_joint_use_probability=probability,
        generation_config_factory=custom_factory)
    @test custom.generation_config_factory === custom_factory
    custom_weights = custom.generation_config_factory(MersenneTwister(1)).weights
    @test custom_weights.overrides[:fold_step_joint_use_probability] == 0.25

    for invalid in (false, true, -0.1, 1.0, Inf, NaN)
        @test_throws ArgumentError default_components(
            fold_step_joint_use_probability=invalid)
    end
end

@testitem "Regression default_components forwards list-state helper weight" setup=[RegBaseImports] begin
    using Random
    CG = LambdaRegression.CanonicalGeneration
    Mutations = LambdaRegression.Mutations
    generator_types = Union{
        Mutations.SubtreeReplacementMutation,
        Mutations.ContextualExpansionMutation,
        Mutations.BinaryCompositionMutation,
        Mutations.FunctionDefMutation,
    }

    neutral = default_components()
    neutral_weights = neutral.generation_config_factory(MersenneTwister(1)).weights
    @test get(neutral_weights.overrides, :list_state_helper, 0.0) == 0.0
    neutral_generators = filter(mutation -> mutation isa generator_types, neutral.mutations)
    @test length(neutral_generators) == 6
    @test all(mutation ->
        get(mutation.generation_kwargs.weights.overrides, :list_state_helper, 0.0) == 0.0,
        neutral_generators)

    weight = 2.5 # A generation weight, not a probability.
    biased = default_components(list_state_helper=weight)
    @test biased.list_state_helper == weight
    initial_weights = biased.generation_config_factory(MersenneTwister(1)).weights
    @test initial_weights.overrides[:list_state_helper] == weight
    generator_backed = filter(mutation -> mutation isa generator_types, biased.mutations)
    @test length(generator_backed) == 6
    @test all(generator_backed) do mutation
        mutation.generation_kwargs.weights.overrides[:list_state_helper] == weight
    end

    custom_factory = rng -> CG.GenerationConfig(; rng,
        weights=CG.GenerationWeights(list_state_helper=0.75))
    custom_factory_components = default_components(list_state_helper=weight,
        generation_config_factory=custom_factory)
    @test custom_factory_components.generation_config_factory === custom_factory
    @test custom_factory_components.generation_config_factory(MersenneTwister(1)).
        weights.overrides[:list_state_helper] == 0.75
    @test all(mutation ->
        mutation.generation_kwargs.weights.overrides[:list_state_helper] == weight,
        filter(mutation -> mutation isa generator_types, custom_factory_components.mutations))

    user_mutations = [Mutations.ParameterIntroductionMutation()]
    custom_mutations_components = default_components(list_state_helper=weight,
        mutations=user_mutations)
    @test custom_mutations_components.mutations === user_mutations
    @test custom_mutations_components.generation_config_factory(MersenneTwister(1)).
        weights.overrides[:list_state_helper] == weight

    for invalid in (false, true, -0.1, Inf, -Inf, NaN, big(10)^1000)
        @test_throws ArgumentError default_components(list_state_helper=invalid)
    end
end

@testitem "Regression mse basic cases" setup=[RegBaseImports] begin
    @test mse(2.0, 1.0) == 1.0
    @test mse([1.0, 2.0], [1.0, 4.0]) == 2.0
    @test mse([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 6.0]]) == 1.0
end

@testitem "Regression evaluate_default identity works for AoA matrix" setup=[RegBaseImports] begin
    using LambdaRegression.Programs: Program
    x = [[1.0, 2.0], [3.0, 4.0]]
    A = TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    target_ty = TyFun(A, A)
    tree = AstTree(AbsNode(:x, IndexNode(0)))
    program = Program(tree; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())

    comps = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=20)
    y = prepare_regression_evaluator(program, comps, cfg)(x)
    @test y == x
end

@testitem "Regression evaluate_default runs FunctionDef natively" setup=[RegBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.Programs: Program

    tree = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0))))
    program = Program(
        tree;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    comps = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )

    default_execution = (; input_slots=comps.input_slots,
        global_consts=comps.global_consts,
        make_eval_context=comps.make_eval_context,
        const_symfactory_ctor=comps.const_symfactory_ctor,
        max_steps=default_config().procedural_max_steps)
    @test LambdaRegression.Execution.evaluate_default(
        program, 3.5; default_execution...,
    ) == 3.5
    prepared = prepare_regression_evaluator(program, comps, default_config())
    @test prepared(4.5) == 4.5
    @test prepared(5.5) == 5.5

    missing_program = Program(
        AstTree(FunctionDefNode(ConstNode(:missing)));
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    missing = LambdaRegression.Execution.evaluate_default(
        missing_program, 3.5; default_execution...,
    )
    @test missing isa Err
    @test missing.value isa MissingConstValueErr
    @test missing.value.name == :missing

    looping_program = Program(
        AstTree(FunctionDefNode(IndexNode(0)));
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    limited_execution = merge(default_execution, (; max_steps=8))
    limited = LambdaRegression.Execution.evaluate_default(
        looping_program, 3.5; limited_execution...,
    )
    @test limited isa Err
    @test limited.value isa MaxReduceStepsErr
    @test limited.value.steps == 8
end

@testitem "Regression startup initializer builds typed programs without a search loop" setup=[RegBaseImports] begin
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: IntTy

    diagnostics = Any[]
    components = default_components(
        global_consts=Dict{Symbol,Any}(:one => 1),
        global_consts_ty=Dict{Symbol,TyLike}(:one => IntTy),
        input_slots=Symbol[],
        input_types=Dict{Symbol,TyLike}(),
        force_lambda_root_probability=1.0,
        init_candidates_per_individual=1,
        initialization_observer=items -> push!(diagnostics, items),
    )
    reg = Regressor(IntTy, components, default_config(population_size=1), MersenneTwister(2000))
    population = components.init_population(
        reg, nothing; N=1, seed=UInt64(10),
        global_type_ctx=components.global_consts_ty,
    )

    @test !(population isa Err)
    @test only(population) isa Program
    @test length(diagnostics) == 1
    @test length(only(diagnostics)) == 1
    @test only(only(diagnostics)) isa AbstractDict
end

@testitem "Regression initialization candidate snapshots are detached and do not alter generation" setup=[RegBaseImports] begin
    using LambdaRegression.Primitives: FADD

    target_ty = TyFun(FloatTy, FloatTy)
    globals = Dict{Symbol,Any}(:FADD => FADD)
    global_types = get_type_ctx(globals)

    capture = candidate -> begin
        names = sort!(collect(keys(candidate.local_consts)); by=String)
        symbols = Tuple((;
            name=String(name),
            value=sprint(show, candidate.local_consts[name]),
            declared_type=sprint(show, candidate.local_consts_ty[name]),
        ) for name in names)
        (; expression=sprint(show, MIME"text/plain"(), candidate.tree), symbols)
    end

    function initialize(; snapshots::Bool, size_preference=nothing)
        diagnostics = Any[]
        rng = MersenneTwister(42)
        preference_options = size_preference === nothing ? NamedTuple() :
            (; init_size_preference=size_preference)
        components = default_components(;
            global_consts=globals,
            global_consts_ty=global_types,
            input_slots=Symbol[:x],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy),
            init_candidates_per_individual=2,
            init_max_try=4,
            init_growth_attempts=1,
            initialization_observer=items -> push!(diagnostics, items),
            initialization_candidate_snapshot=snapshots ? capture : nothing,
            preference_options...,
        )
        config = default_config(population_size=4, parallel=(enabled=false,))
        reg = Regressor(target_ty, components, config, rng)
        population = components.init_population(
            reg, nothing; N=4, seed=UInt64(71),
            global_type_ctx=merge(global_types, components.input_types),
        )
        return (; population, diagnostics, next_rng=rand(rng, UInt64))
    end

    off = initialize(snapshots=false)
    explicit_default = initialize(snapshots=false, size_preference=1.0)
    on = initialize(snapshots=true)
    repeated = initialize(snapshots=true)

    @test !(off.population isa Err)
    @test !(on.population isa Err)
    @test getfield.(off.population, :tree) == getfield.(on.population, :tree)
    @test getfield.(off.population, :local_consts) == getfield.(on.population, :local_consts)
    @test getfield.(off.population, :local_consts_ty) == getfield.(on.population, :local_consts_ty)
    @test getfield.(off.population, :tree) == getfield.(explicit_default.population, :tree)
    @test getfield.(off.population, :local_consts) == getfield.(explicit_default.population, :local_consts)
    @test getfield.(off.population, :local_consts_ty) == getfield.(explicit_default.population, :local_consts_ty)
    @test off.next_rng == on.next_rng
    @test off.next_rng == explicit_default.next_rng
    @test only(on.diagnostics) == only(repeated.diagnostics)
    @test getfield.(on.population, :tree) == getfield.(repeated.population, :tree)
    @test length(only(on.diagnostics)) == length(on.population)
    @test all(!haskey(row, :init_candidate_snapshots) for row in only(off.diagnostics))

    for (slot, row) in pairs(only(on.diagnostics))
        snapshots = row[:init_candidate_snapshots]
        @test length(snapshots) == row[:init_candidate_count]
        @test Tuple(snapshot.node_count for snapshot in snapshots) == row[:init_candidate_sizes]
        @test Tuple(snapshot.successful_attempt for snapshot in snapshots) == row[:init_successful_attempts]
        @test issorted(collect(row[:init_successful_attempts]))
        @test row[:init_candidate_sizes][row[:init_selected_index]] == row[:init_selected_growth_before_nodes]
        @test row[:init_size_preference] == 1.0
        @test row[:init_size_selection_strategy] == :nearest_target
        @test row[:init_selected_candidate_nodes] ==
            row[:init_candidate_sizes][row[:init_selected_index]]
        @test row[:init_selected_growth_after_nodes] == row[:init_final_nodes]
        @test all(snapshot.snapshot.expression isa String for snapshot in snapshots)
        @test all(snapshot.snapshot.symbols isa Tuple for snapshot in snapshots)
        @test all(all(symbol.name isa String && symbol.value isa String &&
            symbol.declared_type isa String for symbol in snapshot.snapshot.symbols)
            for snapshot in snapshots)
        @test !(snapshots[1].snapshot isa AbstractDict)
    end
    @test_throws ArgumentError default_components(
        initialization_candidate_snapshot=capture,
    )
end

@testitem "Regression initialization size preference uses target and uniform policies" setup=[RegBaseImports] begin
    using LambdaRegression.Regression: _select_initial_candidate_index

    sizes = [5, 9, 11, 20]
    target_nodes = 10
    legacy_index = argmin([(abs(size - target_nodes), -size) for size in sizes])
    rng_default = MersenneTwister(1234)
    rng_unchanged = MersenneTwister(1234)
    @test _select_initial_candidate_index(sizes, target_nodes, 1.0, rng_default) ==
        legacy_index
    @test rand(rng_default, UInt64) == rand(rng_unchanged, UInt64)

    expected_uniform = rand(MersenneTwister(2000), eachindex(sizes))
    @test _select_initial_candidate_index(
        sizes, target_nodes, 0.0, MersenneTwister(2000)) == expected_uniform

    @test default_components().init_size_preference == 1.0
    @test default_components(init_size_preference=1).init_size_preference == 1.0
    @test default_components(init_size_preference=0).init_size_preference == 0.0
    for invalid in (-0.1, 1.1, Inf, -Inf, NaN, false, true)
        @test_throws ArgumentError default_components(init_size_preference=invalid)
    end
end

@testitem "Regression initialization FunctionDef root mixture preserves routes and attempt budgets" setup=[RegBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.ProgramAnalysis: functiondef_recursion_summary
    using LambdaRegression.Regression: derive_rng, _role_code
    using LambdaRegression.TypeSystem: BoolTy
    CG = LambdaRegression.CanonicalGeneration
    seed = UInt64(43)
    gate = rand(derive_rng(seed, :init_functiondef_root, 1, 0))
    selected_probability = (gate + 1) / 2
    @test 0 < selected_probability < 1
    @test _role_code(:init_functiondef_root) != _role_code(:init_size_selection)

    function initialize(; probability=nothing, body_attempts=1, max_try=4,
                        candidates=2, size_preference=1, fixed_rng=true)
        config_calls = Ref(0)
        filter_calls = Ref(0)
        factory_rngs = Any[]
        diagnostics = Any[]
        provider = CG.FunctionDefProvider(
            max_nesting=1, body_attempts=body_attempts,
            admissibility=(candidate, relation, query) -> begin
                filter_calls[] += 1
                true
            end,
        )
        weights = CG.GenerationWeights(
            list_recursive_body_probability=0.8,
            list_recursive_tail_probability=0.8,
            functiondef_self_call_probability=0.8,
            safe_list_short_fallback_probability=0.8,
        )
        options = probability === nothing ? NamedTuple() :
            (; init_functiondef_root_probability=probability)
        components = default_components(;
            global_consts=Dict{Symbol,Any}(:root_route_probe => false),
            global_consts_ty=Dict{Symbol,TyLike}(:root_route_probe => BoolTy),
            functiondef_provider=provider,
            generation_config_factory=rng -> begin
                config_calls[] += 1
                push!(factory_rngs, rng)
                CG.GenerationConfig(;
                    rng=fixed_rng ? Random.Xoshiro(2000) : rng,
                    providers=(CG.ContextHeadProvider(),), weights,
                    sampling_depth_limit=3, sampling_node_budget=12,
                )
            end,
            force_lambda_root_probability=1.0,
            init_candidates_per_individual=candidates,
            init_max_try=max_try, init_growth_attempts=0,
            init_size_preference=size_preference,
            initialization_observer=items -> append!(diagnostics, items),
            initialization_candidate_snapshot=candidate -> (;),
            options...,
        )
        reg = Regressor(TyFun(FloatTy, FloatTy), components,
            default_config(population_size=1), MersenneTwister(42))
        population = components.init_population(reg, nothing;
            N=1, seed, global_type_ctx=components.global_consts_ty)
        return (; population, diagnostics, config_calls=config_calls[],
            filter_calls=filter_calls[], components, provider,
            next_rngs=[rand(rng, UInt64) for rng in factory_rngs])
    end

    preferred = initialize(probability=selected_probability)
    @test !(preferred.population isa Err)
    @test preferred.config_calls == 1
    @test preferred.filter_calls == 1
    @test preferred.components.functiondef_provider === preferred.provider
    @test only(preferred.population).tree.root isa FunctionDefNode
    @test functiondef_recursion_summary(only(preferred.population)).self_calls > 0
    row = only(preferred.diagnostics)
    @test row[:init_functiondef_root_requested]
    @test row[:init_functiondef_root_attempted]
    @test row[:init_functiondef_root_succeeded]
    @test !row[:init_functiondef_root_fallback]
    @test row[:init_generation_attempts] == 1
    @test row[:init_candidate_count] == 1
    @test row[:init_successful_attempts] == (1,)
    @test row[:init_size_selection_strategy] === :functiondef_root
    @test row[:init_selected_growth_before_nodes] == row[:init_initial_nodes]
    @test row[:init_initial_nodes] == row[:init_final_nodes]
    @test row[:sampling_nodes_used] <= 12
    uniform_preferred = initialize(probability=selected_probability, size_preference=0)
    @test only(uniform_preferred.diagnostics)[:init_size_selection_strategy] === :functiondef_root
    @test getfield.(preferred.population, :tree) == getfield.(uniform_preferred.population, :tree)

    fallback = initialize(probability=selected_probability, body_attempts=0)
    @test !(fallback.population isa Err)
    @test fallback.config_calls == 3 # One failed preferred + two ordinary successes.
    @test fallback.filter_calls == 0
    fallback_row = only(fallback.diagnostics)
    @test fallback_row[:init_functiondef_root_requested]
    @test fallback_row[:init_functiondef_root_attempted]
    @test !fallback_row[:init_functiondef_root_succeeded]
    @test fallback_row[:init_functiondef_root_fallback]
    @test fallback_row[:init_generation_attempts] == 3
    @test fallback_row[:init_successful_attempts] == (2, 3)
    @test fallback_row[:init_candidate_count] == 2
    @test fallback_row[:init_size_selection_strategy] === :nearest_target
    bounded = initialize(probability=selected_probability, body_attempts=0, max_try=2)
    @test !(bounded.population isa Err)
    @test bounded.config_calls == 2
    @test only(bounded.diagnostics)[:init_candidate_count] == 1
    @test only(bounded.diagnostics)[:init_successful_attempts] == (2,)
    exhausted = initialize(probability=selected_probability, body_attempts=0, max_try=1)
    @test exhausted.population isa Err
    @test exhausted.config_calls == 1
    @test isempty(exhausted.diagnostics)

    ordinary = initialize(probability=gate / 2)
    @test !(ordinary.population isa Err)
    @test ordinary.config_calls == 2
    ordinary_row = only(ordinary.diagnostics)
    @test !ordinary_row[:init_functiondef_root_requested]
    @test !ordinary_row[:init_functiondef_root_attempted]
    @test !ordinary_row[:init_functiondef_root_succeeded]
    @test !ordinary_row[:init_functiondef_root_fallback]
    @test ordinary_row[:init_generation_attempts] == 2

    omitted = initialize(fixed_rng=false)
    zero = initialize(probability=0, fixed_rng=false)
    @test !(zero.population isa Err)
    @test getfield.(omitted.population, :tree) == getfield.(zero.population, :tree)
    @test getfield.(omitted.population, :local_consts) == getfield.(zero.population, :local_consts)
    @test getfield.(omitted.population, :local_consts_ty) == getfield.(zero.population, :local_consts_ty)
    @test omitted.diagnostics == zero.diagnostics
    @test omitted.next_rngs == zero.next_rngs
    @test !haskey(only(zero.diagnostics), :init_functiondef_root_probability)
    @test default_components().init_functiondef_root_probability == 0
    custom_init = (args...; kwargs...) -> nothing
    @test default_components(init_population=custom_init,
        init_functiondef_root_probability=0).init_population === custom_init
    @test_throws ArgumentError default_components(init_population=custom_init,
        init_functiondef_root_probability=0.5)
    for invalid in (-0.1, 1, 1.1, Inf, -Inf, NaN, false, true)
        @test_throws ArgumentError default_components(init_functiondef_root_probability=invalid)
    end
end

@testitem "Regression default initialization is deterministic across thread modes" setup=[RegBaseImports] begin
    using LambdaRegression.Primitives: FADD

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
        parallel=(enabled=false,),
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
end

@testitem "Regression closure_lift unlocks AMAP with AST lambdas" setup=[RegBaseImports] begin
    x = [1.0, -2.0, 3.0]
    A = TyArray(TyNat(length(x)), FloatTy)
    target_ty = TyFun(A, A)

    inner = AbsNode(:y, AppNode(ConstNode(:SQR), IndexNode(0)))
    body = AppNode(AppNode(ConstNode(:AMAP), inner), IndexNode(0))
    tree = AstTree(AbsNode(:x, body))

    global_consts = Dict{Symbol,Any}(
        :AMAP => LambdaRegression.Primitives.AMAP,
        :SQR => LambdaRegression.Primitives.SQR,
    )
    comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=200)
    program = Program(
        tree;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    y = prepare_regression_evaluator(program, comps, cfg)(x)
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
    matrix_comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:matrix],
        input_types=Dict{Symbol,TyLike}(:matrix => MatrixTy),
    )
    matrix_program = Program(
        matrix_tree;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    matrix_y = prepare_regression_evaluator(matrix_program, matrix_comps, cfg)(matrix)
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

    global_consts = Dict{Symbol,Any}(
        :AFOLDL => LambdaRegression.Primitives.AFOLDL,
        :FADD => LambdaRegression.Primitives.FADD,
        :SQR => LambdaRegression.Primitives.SQR,
        :ZERO => 0.0,
    )
    comps = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    cfg = default_config(max_reduce_steps=600)
    program = Program(
        tree;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    y = prepare_regression_evaluator(program, comps, cfg)(x)
    y isa Err && error("evaluate_default returned Err: $(y.value)")
    @test y ≈ sum(v * v for v in x)
end

@testitem "Regression native callback depth and shared budget are bounded" setup=[RegBaseImports] begin
    chain = f -> f(0.0)
    chain_type = TyFun(TyFun(FloatTy, FloatTy), FloatTy)
    globals = Dict{Symbol,Any}(:CHAIN => chain, :ZERO => 0.0)
    global_types = Dict{Symbol,TyLike}(:CHAIN => chain_type, :ZERO => FloatTy)
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )

    nested = let nested = ConstNode(:ZERO)
        for i in 1:12
            nested = AppNode(ConstNode(:CHAIN), AbsNode(Symbol("d", i), nested))
        end
        nested
    end
    tree = AstTree(AbsNode(:x, nested))
    target_ty = TyFun(FloatTy, FloatTy)
    program = Program(
        tree;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    cfg = default_config(
        max_reduce_steps=200,
        closure_max_depth=4,
    )

    issue = prepare_regression_evaluator(program, components, cfg)(0.0)
    @test issue isa Err
    @test issue.value isa ClosureDepthLimitErr
    @test issue.value.limit == 4

    generous = default_config(
        max_reduce_steps=400,
        closure_max_depth=64,
    )
    generous_evaluator = prepare_regression_evaluator(program, components, generous)
    @test generous_evaluator(0.0) == 0.0
    @test generous_evaluator(0.0) == 0.0

    identity_program = Program(
        AstTree(AbsNode(:x, IndexNode(0)));
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    @test prepare_regression_evaluator(identity_program, components, cfg)(3.0) == 3.0

    low_budget = default_config(
        max_reduce_steps=8,
        procedural_max_steps=8,
        closure_max_depth=64,
    )
    budget_issue = prepare_regression_evaluator(program, components, low_budget)(0.0)
    @test budget_issue isa Err
    @test budget_issue.value isa MaxReduceStepsErr
    @test budget_issue.value.steps == 8
end

@testitem "Regression unbound callback reference is rejected by task evaluation" setup=[RegBaseImports] begin
    x = [1.0, 2.0]
    A = TyArray(TyNat(length(x)), FloatTy)
    bad_inner = AbsNode(:y, IndexNode(2))
    bad_body = AppNode(AppNode(ConstNode(:AMAP), bad_inner), IndexNode(0))
    bad_program = Program(AstTree(AbsNode(:x, bad_body)))
    global_consts = Dict{Symbol,Any}(:AMAP => LambdaRegression.Primitives.AMAP)
    components = default_components(
        global_consts=global_consts,
        global_consts_ty=get_type_ctx(global_consts),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
    )
    config = default_config(max_reduce_steps=80)

    result = prepare_regression_evaluator(bad_program, components, config)(x)
    @test result isa Err
    @test result.value isa NonConstOutputErr
end


end # module Regression
