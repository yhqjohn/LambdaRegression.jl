"""
    AgeSearch

Small resumable outer search over complete lambda/FunctionDef AST programs.

The state borrows the caller's regressor, component world, and dataset; the
caller must keep the task-defining parts semantically stable for the state's
lifetime. Caller-owned mutable resources keep their identity, so intentional
recorder/provider mutations remain observable. One state may share such a
resource across its sequential islands; independent states need separate
mutable resources unless sharing is deliberate. Default component closures
must continue to match the world held by their component fields.

The state owns one RNG stream per island, scored populations, counters, a
bounded recent-offspring buffer, and a loss-by-written-complexity archive.
`advance_age_search!` performs a bounded number of *outer* proposal attempts;
the existing mutation and crossover implementations retain ownership of their
inner retry limits. Caller workflows retain outer scheduling and teacher
ownership; this layer exposes narrow hooks for parent selection, replacement,
and valid-candidate observation. AgeSearch's `crossover_op` must
support the directed `Populations.recombine` seam; each crossover proposal
chooses one recipient/donor direction and produces at most one child. This is
an AgeSearch component requirement, not a requirement imposed on every
`AbstractCrossover`, whose separate two-child `crossover` seam remains
available. A changed task or component world requires a new state. Candidate
scoring uses Regression's shared per-sample loop while this state retains its
own invalid-candidate accounting and penalty policy.
"""

export AgeSearchState, AgeSearchIndividual, AgeSearchMigration
export AgeSearchCounters, AgeSearchReport
export AgeSearchNoValidErr
export initialize_age_search, advance_age_search!, inject_age_migrants!
export AbstractAgeSearchPolicy
export age_select_parent, age_replacement_index, age_observe_candidate!
export age_parent_candidates

"""
    AbstractAgeSearchPolicy

Optional caller-owned hooks for parent selection, replacement, and observation
of valid scored candidates. Extend `age_select_parent(policy, state, island,
candidates, rng, tournament_size, role)` to return one of `candidates`;
`role` is `:parent` or `:donor`. Extend `age_replacement_index(policy, state,
record, island)` to return a population position or `nothing` to reject admission.
`age_observe_candidate!(policy, state, record, island, replaced, admitted)` is
called after every valid candidate is scored and its admission decision is
applied. It receives the actual scored record and the complete replaced record,
or `nothing` when admission is rejected. A policy does not supply candidate
scores.

Each hook has an inherited default: tournament selection, oldest-member
replacement, and no-op observation. A subtype can override only the hooks it
needs. Passing `policy=nothing` uses the same defaults.
"""
abstract type AbstractAgeSearchPolicy end

"""Finite failure returned when every island currently lacks a valid candidate."""
struct AgeSearchNoValidErr end

"""
    AgeSearchIndividual

Owned, scored record used by an age-search population. `parent_ids` is an
explicit ancestry tuple; an empty tuple means that the caller did not provide
provenance (for example, a plain migrant `Program`).
"""
struct AgeSearchIndividual
    program::Program
    id::UInt64
    parent_ids::Tuple{Vararg{UInt64}}
    origin::Symbol
    loss::Float64
    complexity::Int
    valid::Bool
    birth::UInt64
end

const _AgeRecord = AgeSearchIndividual

"""Cumulative accounting for initialization, evolution, and migration.

`objective_calls`, `sample_executions`, and `invalids` are totals across all
sources. The `init_*`, `advance_*`, and `migration_*` fields retain the same
dimensions by source.
"""
mutable struct AgeSearchCounters
    init_objective_calls::Int
    init_sample_executions::Int
    init_invalids::Int
    objective_calls::Int
    sample_executions::Int
    invalids::Int
    raw_attempts::Int
    mutation_attempts::Int
    crossover_attempts::Int
    expected_failures::Int
    unchanged::Int
    accepted::Int
    migration_attempts::Int
    migration_objective_calls::Int
    migration_sample_executions::Int
    migration_invalids::Int
    migration_accepted::Int
    migration_duplicates::Int
    advance_objective_calls::Int
    advance_sample_executions::Int
    advance_invalids::Int
    operator_labels::Dict{Symbol,Int}
end

AgeSearchCounters() = AgeSearchCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Dict{Symbol,Int}(),
)

"""Per-call accounting returned by `advance_age_search!`."""
struct AgeSearchReport
    attempts::Int
    accepted::Int
    expected_failures::Int
    unchanged::Int
    invalids::Int
    mutation_attempts::Int
    crossover_attempts::Int
    objective_calls::Int
    sample_executions::Int
    cursor::Int
end

"""Mutable state for a persistent complete-AST age search.

`reg` and `data` are borrowed from the caller and must remain stable while the
state is used. Populations, score records, RNG streams, archive, and counters
are owned by this state.
"""
mutable struct AgeSearchState
    reg::Regressor
    data::SupervisedDataset
    seed::UInt64
    populations::Vector{Vector{AgeSearchIndividual}}
    rngs::Vector{AbstractRNG}
    cursor::Int
    birth_clock::UInt64
    next_id::UInt64
    recent_valid_offspring::Vector{AgeSearchIndividual}
    recent_capacity::Int
    archive::Dict{Int,AgeSearchIndividual}
    best::Union{Nothing,AgeSearchIndividual}
    counters::AgeSearchCounters
end

"""Migration input retaining explicit provenance when it is available."""
struct AgeSearchMigration
    program::Program
    parent_ids::Tuple{Vararg{UInt64}}
end

# Optional caller-owned timing callbacks. Clock tokens do not wrap the operator
# call or change its return/exception dispatch; spans here never nest. Only
# advance proposals activate diagnostics, so initialization/migration are not
# silently included. Callback work belongs to the caller's measured overhead.
_age_span_begin(::Nothing, category::Symbol) = nothing
_age_span_begin(diagnostics, category::Symbol) = diagnostics.begin_span(category)
_age_span_end(::Nothing, category::Symbol, token) = nothing
_age_span_end(diagnostics, category::Symbol, token) = diagnostics.end_span(category, token)
_age_proposal_begin(::Nothing, state) = nothing
_age_proposal_begin(diagnostics, state) = diagnostics.begin_proposal(state)
_age_proposal_end(::Nothing, state, label, left, right, child) = nothing
_age_proposal_end(diagnostics, state, label, left, right, child) =
    diagnostics.end_proposal(state, label, left, right, child)

function _age_checked_weight(value, label::Symbol)
    value isa Real || throw(ArgumentError("$label must be real"))
    out = Float64(value)
    isfinite(out) && out >= 0 || throw(ArgumentError(
        "$label must be finite and non-negative",
    ))
    return out
end

function _age_required_components(reg::Regressor)
    c = reg.components
    required = (
        :global_consts, :global_consts_ty, :input_slots, :input_types,
        :tyfactory_ctor, :const_symfactory_ctor, :make_eval_context,
    )
    for key in required
        haskey(c, key) || return MissingComponentErr(key)
    end
    haskey(c, :init_population) || return MissingComponentErr(:init_population)
    return nothing
end

function _age_preflight_task(reg::Regressor, data::SupervisedDataset)
    c = reg.components
    slots = c.input_slots
    slots isa AbstractVector{Symbol} || return ArgumentError(
        "input_slots must be a vector of symbols",
    )
    input_types = c.input_types
    input_types isa AbstractDict{Symbol,<:TyLike} || return ArgumentError(
        "input_types must map symbols to types",
    )
    uses_default_evaluator = !haskey(c, :prepare_evaluate) && !haskey(c, :evaluate)
    uses_default_evaluator && isempty(slots) && return ArgumentError(
        "input_slots must be non-empty for evaluate_default",
    )
    length(unique(slots)) == length(slots) || return ArgumentError(
        "input_slots must not contain duplicates",
    )
    Set(keys(input_types)) == Set(slots) || return ArgumentError(
        "input_types keys must exactly match input_slots",
    )
    signature_error = _preflight_input_signature(reg.target_ty, slots, input_types)
    signature_error === nothing || return signature_error
    if length(slots) > 1
        for (j, x) in enumerate(data.X)
            x isa Tuple || return ArgumentError("multi-input sample $j must be a Tuple")
            length(x) == length(slots) || return ArgumentError(
                "multi-input sample $j has wrong arity",
            )
        end
    end
    Set(keys(c.global_consts)) == Set(keys(c.global_consts_ty)) || return ArgumentError(
        "global_consts and global_consts_ty keys must match",
    )
    isempty(intersect(Set(keys(c.global_consts)), Set(slots))) || return ArgumentError(
        "global constants and input slots must use disjoint names",
    )
    return nothing
end

function _age_error_penalty(reg::Regressor)
    value = get(reg.config, :error_penalty, 1.0e6)
    value isa Real || throw(ArgumentError("error_penalty must be real"))
    out = Float64(value)
    isfinite(out) && out >= 0 || throw(ArgumentError(
        "error_penalty must be finite and non-negative",
    ))
    return out
end

function _age_expected_exception(issue, components::NamedTuple; loss::Bool=false)
    predicate = get(components, loss ? :loss_failure : :candidate_failure,
        loss ? _no_candidate_exception : _known_candidate_exception)
    predicate(issue) && return true
    loss && return false
    return issue isa DomainError || issue isa DivideError ||
           issue isa OverflowError || issue isa InexactError ||
           issue isa Mutations.MaxTryReachedErr ||
           issue isa Mutations.NoEligibleSiteErr ||
           issue isa Mutations.NoEligibleMutationErr ||
           issue isa Populations.NoEligibleCrossoverPairErr ||
           issue isa Populations.CrossoverPairPoolExhaustedErr ||
           issue isa Populations.CrossoverMaxTryReachedErr
end

function _age_same_program(a::Program, b::Program)
    return a.tree == b.tree &&
        isequal(a.local_consts, b.local_consts) &&
        isequal(a.local_consts_ty, b.local_consts_ty)
end

function _age_next_id!(state::AgeSearchState)
    id = state.next_id
    state.next_id += UInt64(1)
    return id
end

function _age_birth!(state::AgeSearchState)
    state.birth_clock += UInt64(1)
    return state.birth_clock
end

function _age_make_record(
    state::AgeSearchState,
    program::Program,
    parent_ids::Tuple{Vararg{UInt64}},
    origin::Symbol,
    loss::Float64,
    complexity::Int,
    valid::Bool;
    birth::UInt64=UInt64(0),
)
    return AgeSearchIndividual(
        program, _age_next_id!(state), parent_ids, origin, loss, complexity,
        valid, birth,
    )
end

function _age_archive!(state::AgeSearchState, record::AgeSearchIndividual)
    record.valid || return nothing
    old = get(state.archive, record.complexity, nothing)
    if old === nothing || record.loss < old.loss
        state.archive[record.complexity] = record
    end
    if state.best === nothing || record.loss < state.best.loss
        state.best = record
    end
    return nothing
end

function _age_push_recent!(state::AgeSearchState, record::AgeSearchIndividual)
    state.recent_capacity > 0 || return nothing
    push!(state.recent_valid_offspring, record)
    while length(state.recent_valid_offspring) > state.recent_capacity
        popfirst!(state.recent_valid_offspring)
    end
    return nothing
end

function _age_oldest_index(population::Vector{AgeSearchIndividual})
    isempty(population) && throw(ArgumentError("age-search population cannot be empty"))
    oldest = 1
    for i in 2:length(population)
        population[i].birth < population[oldest].birth && (oldest = i)
    end
    return oldest
end

function _age_insert!(
    state::AgeSearchState,
    record::AgeSearchIndividual,
    population::Int;
    policy::Union{Nothing,AbstractAgeSearchPolicy}=nothing,
)
    _age_archive!(state, record)
    pop = state.populations[population]
    index = age_replacement_index(policy, state, record, population)
    if index === nothing
        age_observe_candidate!(policy, state, record, population, nothing, false)
        return false
    end
    index isa Integer && !(index isa Bool) || throw(ArgumentError(
        "age_replacement_index must return an integer population position or nothing",
    ))
    1 <= index <= length(pop) || throw(BoundsError(pop, index))
    slot = Int(index)
    replaced = pop[slot]
    pop[slot] = record
    age_observe_candidate!(policy, state, record, population, replaced, true)
    return true
end

function _age_valid_records(state::AgeSearchState, population::Int)
    return [record for record in state.populations[population] if record.valid && isfinite(record.loss)]
end

function _age_tournament(rng::AbstractRNG, candidates::Vector{AgeSearchIndividual}, size::Int)
    isempty(candidates) && throw(ArgumentError("tournament requires a valid candidate"))
    size >= 1 || throw(ArgumentError("tournament size must be positive"))
    picked = AgeSearchIndividual[]
    for _ in 1:size
        push!(picked, candidates[rand(rng, 1:length(candidates))])
    end
    best_loss = minimum(record.loss for record in picked)
    ties = filter(record -> record.loss == best_loss, picked)
    return ties[rand(rng, 1:length(ties))]
end

function age_select_parent(
    ::Union{Nothing,AbstractAgeSearchPolicy},
    state,
    island,
    candidates,
    rng,
    tournament_size,
    role::Symbol,
)
    tournament_size >= 1 || throw(ArgumentError("tournament size must be positive"))
    return _age_tournament(rng, candidates, tournament_size)
end

function age_replacement_index(
    ::Union{Nothing,AbstractAgeSearchPolicy},
    state,
    record,
    island,
)
    return _age_oldest_index(state.populations[island])
end

function age_observe_candidate!(
    ::Union{Nothing,AbstractAgeSearchPolicy},
    state,
    record,
    island,
    replaced,
    admitted,
)
    return nothing
end

function _age_selected_parent(
    policy::Union{Nothing,AbstractAgeSearchPolicy},
    state::AgeSearchState,
    island::Int,
    candidates::Vector{AgeSearchIndividual},
    rng::AbstractRNG,
    tournament_size::Int,
    role::Symbol,
)
    tournament_size >= 1 || throw(ArgumentError("tournament size must be positive"))
    pool = age_parent_candidates(policy, state, island, candidates, role)
    pool isa Vector{AgeSearchIndividual} && !isempty(pool) || throw(ArgumentError(
        "age_parent_candidates must return a nonempty scored-record vector"))
    all(record -> record.valid && isfinite(record.loss), pool) || throw(ArgumentError(
        "age_parent_candidates must contain valid scored records from the stable task"))
    selected = age_select_parent(
        policy, state, island, pool, rng, tournament_size, role,
    )
    selected isa AgeSearchIndividual || throw(ArgumentError(
        "age_select_parent must return an AgeSearchIndividual from candidates",
    ))
    any(candidate -> candidate === selected, pool) || throw(ArgumentError(
        "age_select_parent must return an AgeSearchIndividual from candidates",
    ))
    return selected
end

"""Optional owned, same-task scored archive material for mating; default uses live candidates.

An extension may add actual scored witnesses, never synthetic scores or records
from another task. The returned vector is borrowed for synchronous selection.
"""
age_parent_candidates(::Union{Nothing,AbstractAgeSearchPolicy}, state,
    island, candidates, role::Symbol) = candidates

function _age_validate_candidate(
    state::AgeSearchState,
    program::Program,
    label::AbstractString;
    type_checked::Bool=false,
)
    issue = _validate_program_contract(program, state.reg.target_ty,
        state.reg.components, label; check_type=!type_checked)
    issue === nothing || throw(issue)
    policy = get(state.reg.components, :bloat_policy, Mutations.BloatPolicy())
    _within_bloat_limit(program.tree, policy) || return :bloat
    rejection = _program_candidate_rejection(program, state.reg.components)
    rejection === nothing || return rejection
    return nothing
end

"""Score one candidate and account for its original-task objective work."""
function _age_count_samples!(state::AgeSearchState, source::Symbol, count::Int)
    state.counters.sample_executions += count
    if source === :init
        state.counters.init_sample_executions += count
    elseif source === :migration
        state.counters.migration_sample_executions += count
    else
        state.counters.advance_sample_executions += count
    end
    return nothing
end

function _age_score!(state::AgeSearchState, program::Program, source::Symbol)
    components = state.reg.components
    config = state.reg.config
    loss_fn = get(components, :loss, mse)
    penalty = _age_error_penalty(state.reg)
    complexity_fn = get(components, :complexity, _node_count)
    complexity = complexity_fn(program.tree)
    complexity isa Integer || throw(ArgumentError("complexity must return an integer"))
    complexity = Int(complexity)

    state.counters.objective_calls += 1
    if source === :init
        state.counters.init_objective_calls += 1
    elseif source === :migration
        state.counters.migration_objective_calls += 1
    else
        state.counters.advance_objective_calls += 1
    end

    diagnostics = get(components, :age_diagnostics, nothing)
    token = _age_span_begin(diagnostics, :preparation)
    prepared = try
        prepare_task_evaluator(
        program, state.reg.target_ty, components, config; copy_tree=false,
        )
    finally
        _age_span_end(diagnostics, :preparation, token)
    end
    token = _age_span_begin(diagnostics, :scoring)
    result = try
        score_samples(state.data, prepared, loss_fn;
        failure_mode=:continue_penalty,
        failure_penalty=penalty,
        candidate_failure=issue -> _age_expected_exception(issue, components),
        loss_failure=issue -> _age_expected_exception(issue, components; loss=true),
        strict_returned_errors=get(config, :strict_errors, false),
        on_samples=count -> _age_count_samples!(state, source, count))
    finally
        _age_span_end(diagnostics, :scoring, token)
    end
    if !result.valid
        state.counters.invalids += 1
        if source === :init
            state.counters.init_invalids += 1
        elseif source === :migration
            state.counters.migration_invalids += 1
        else
            state.counters.advance_invalids += 1
        end
    end
    return result.valid, Float64(result.score), complexity
end

function _age_score_record!(
    state::AgeSearchState,
    program::Program,
    parent_ids::Tuple{Vararg{UInt64}},
    origin::Symbol,
    source::Symbol;
    type_checked::Bool=false,
)
    diagnostics = get(state.reg.components, :age_diagnostics, nothing)
    token = _age_span_begin(diagnostics, :validation)
    admissibility = try
        _age_validate_candidate(
        state, program, "$(origin) candidate"; type_checked,
        )
    finally
        _age_span_end(diagnostics, :validation, token)
    end
    if admissibility !== nothing
        complexity_fn = get(state.reg.components, :complexity, _node_count)
        complexity = Int(complexity_fn(program.tree))
        state.counters.invalids += 1
        if source === :init
            state.counters.init_invalids += 1
        elseif source === :migration
            state.counters.migration_invalids += 1
        else
            state.counters.advance_invalids += 1
        end
        return _age_make_record(
            state,
            program,
            parent_ids,
            origin,
            _age_error_penalty(state.reg),
            complexity,
            false,
        )
    end
    valid, loss, complexity = _age_score!(state, program, source)
    return _age_make_record(state, program, parent_ids, origin, loss, complexity, valid)
end

function _age_mutation(state::AgeSearchState, parent::AgeSearchIndividual, rng::AbstractRNG;
        progress::ProposalProgress=_proposal_progress(state.reg.components,
            max(0, state.counters.raw_attempts - 1)), check_admission::Bool=true)
    c = state.reg.components
    provider = get(c, :full_mutation_provider, OrdinaryMutationProvider())
    return _checked_propose_mutation(provider, parent.program, state.reg.target_ty;
        rng, progress, components=c, check_admission)
end

function _age_crossover(
    state::AgeSearchState,
    left::AgeSearchIndividual,
    right::AgeSearchIndividual,
    rng::AbstractRNG,
)
    c = state.reg.components
    op = get(c, :crossover_op, nothing)
    op === nothing && return Err(Populations.NoEligibleCrossoverPairErr())
    direction = rand(rng, Bool) ? :left : :right
    recipient, donor, selection_roles = direction === :left ?
        (left, right, (:left, :right)) :
        (right, left, (:right, :left))
    child = Populations.recombine(
        op,
        recipient.program,
        donor.program;
        recipient_target=state.reg.target_ty,
        donor_target=state.reg.target_ty,
        rng,
        global_consts_ty=merge(c.global_consts_ty, c.input_types),
        tyfactory_ctor=c.tyfactory_ctor,
        const_symfactory_ctor=c.const_symfactory_ctor,
        selection_roles,
        prune=true,
    )
    child isa Err && return child
    child isa Program || throw(ArgumentError(
        "crossover_op recombine must return a Program or Err",
    ))
    return child
end

function _age_call_proposal(
    proposal,
    state::AgeSearchState,
    population::Int,
    left::AgeSearchIndividual,
    right::AgeSearchIndividual,
    rng::AbstractRNG,
)
    result = proposal(state, population, left, right; rng)
    result isa Err && return result
    if result isa Program
        return (:proposal, result)
    elseif result isa Tuple && length(result) == 2 && result[1] isa Symbol
        result[2] isa Program || throw(ArgumentError("proposal child must be a Program"))
        return (result[1], result[2])
    end
    throw(ArgumentError("proposal must return Program, (label, Program), or Err"))
end

function _age_add_operator!(counters::AgeSearchCounters, label::Symbol)
    counters.operator_labels[label] = get(counters.operator_labels, label, 0) + 1
    return nothing
end

function _age_snapshot(c::AgeSearchCounters)
    return (
        raw_attempts=c.raw_attempts,
        mutation_attempts=c.mutation_attempts,
        crossover_attempts=c.crossover_attempts,
        expected_failures=c.expected_failures,
        unchanged=c.unchanged,
        accepted=c.accepted,
        objective_calls=c.objective_calls,
        sample_executions=c.sample_executions,
        invalids=c.invalids,
        cursor=0,
    )
end

"""
    initialize_age_search(reg, data; population_count=2,
                          population_size=reg.config.population_size,
                          seed=nothing, recent_capacity=32, initial_transform=nothing)

Construct and score each island exactly once through `init_population`, with a
distinct derived seed per island. Initialization has no hidden population fill;
invalid records remain visible but cannot breed. The state borrows `reg` and
`data` without copying them. Keep the task, component world, and dataset stable
for the state's lifetime; injected mutable resources retain their identity.
Construct default components against the final world and do not replace
world-defining component fields afterward, because default callables may close
over that world.

Optional `initial_transform(state, program; rng)` runs on each ordinary initial
program before its first score, archive insertion or population comparison. It
must return an owned `Program` or `Err`; the latter stops initialization. The
callback owns and accounts for its additional work. It may read the partial
state's counters, but must not mutate the state or the supplied program.
"""
function initialize_age_search(
    reg::Regressor,
    data::SupervisedDataset;
    population_count::Int=2,
    population_size::Int=get(reg.config, :population_size, 0),
    seed=nothing,
    recent_capacity::Int=32,
    initial_transform=nothing,
)
    length(data.X) == length(data.Y) || return Err(ArgumentError("X and Y must have the same length"))
    isempty(data.X) && return Err(ArgumentError("age-search data must contain at least one sample"))
    population_count > 0 || return Err(ArgumentError("population_count must be positive"))
    population_size > 0 || return Err(ArgumentError("population_size must be positive"))
    recent_capacity >= 0 || return Err(ArgumentError("recent_capacity must be non-negative"))
    required = _age_required_components(reg)
    required === nothing || return Err(required)
    task_error = _age_preflight_task(reg, data)
    task_error === nothing || return Err(task_error)
    root_seed = seed === nothing ? rand(reg.rng, UInt64) : UInt64(seed)
    state = AgeSearchState(
        reg,
        data,
        root_seed,
        Vector{Vector{AgeSearchIndividual}}(),
        AbstractRNG[],
        1,
        UInt64(0),
        UInt64(1),
        AgeSearchIndividual[],
        recent_capacity,
        Dict{Int,AgeSearchIndividual}(),
        nothing,
        AgeSearchCounters(),
    )
    global_type_ctx = merge(reg.components.global_consts_ty, reg.components.input_types)
    for island in 1:population_count
        island_seed = _derive_seed(root_seed, :age_init, island)
        push!(state.rngs, MersenneTwister(UInt32[UInt32(island_seed & 0xffffffff), UInt32(island_seed >> 32)]))
        result = reg.components.init_population(
            reg,
            data;
            N=population_size,
            seed=island_seed,
            global_type_ctx=global_type_ctx,
        )
        result isa Err && return result
        issue = _preflight_initial_population(result, reg, reg.components, population_size)
        issue === nothing || return Err(issue)
        pop = AgeSearchIndividual[]
        for (slot, raw) in enumerate(result)
            raw isa Program || throw(ArgumentError("init_population must return Programs"))
            candidate = raw
            if initial_transform !== nothing
                candidate = initial_transform(state, raw; rng=state.rngs[island])
                candidate isa Err && return candidate
                candidate isa Program || throw(ArgumentError("initial_transform must return Program or Err"))
            end
            record = _age_score_record!(state, candidate, (), :initial, :init)
            if record.birth == 0
                record = AgeSearchIndividual(
                    record.program, record.id, record.parent_ids, record.origin,
                    record.loss, record.complexity, record.valid, _age_birth!(state),
                )
            end
            push!(pop, record)
            _age_archive!(state, record)
        end
        push!(state.populations, pop)
    end
    return state
end

"""
    advance_age_search!(state; max_attempts, max_evaluations,
                        mutation_weight=1.0, crossover_weight=1.0,
                        proposal=nothing)

Run at most `max_attempts` outer proposals and at most `max_evaluations`
candidate objective calls. Each changed candidate is scored against the
borrowed task (which the caller must keep stable). Valid candidates are admitted
or rejected through `policy`; omitting it retains oldest-member replacement.
Rejected valid candidates remain in the score archive and are reported to the
policy observer, but do not count as accepted offspring.
"""
function advance_age_search!(
    state::AgeSearchState;
    max_attempts::Int,
    max_evaluations::Int,
    mutation_weight=1.0,
    crossover_weight=1.0,
    proposal=nothing,
    policy::Union{Nothing,AbstractAgeSearchPolicy}=nothing,
    replay=nothing,
)
    max_attempts >= 0 || throw(ArgumentError("max_attempts must be non-negative"))
    max_evaluations >= 0 || throw(ArgumentError("max_evaluations must be non-negative"))
    for endpoint in (0.0, 1.0)
        _age_checked_weight(schedule_value(mutation_weight, endpoint), :mutation_weight)
        _age_checked_weight(schedule_value(crossover_weight, endpoint), :crossover_weight)
    end
    _replay_start!(replay, state)
    if max_attempts == 0 || max_evaluations == 0
        _replay_finish!(replay, state)
        return AgeSearchReport(0, 0, 0, 0, 0, 0, 0, 0, 0, state.cursor)
    end
    if !any(!isempty(_age_valid_records(state, island)) for island in eachindex(state.populations))
        _replay_finish!(replay, state)
        return Err(AgeSearchNoValidErr())
    end
    before = _age_snapshot(state.counters)
    objective_start = state.counters.objective_calls
    samples_start = state.counters.sample_executions
    accepted_start = state.counters.accepted
    expected_start = state.counters.expected_failures
    unchanged_start = state.counters.unchanged
    invalid_start = state.counters.invalids
    mutation_start = state.counters.mutation_attempts
    crossover_start = state.counters.crossover_attempts
    diagnostics = get(state.reg.components, :age_diagnostics, nothing)

    for _ in 1:max_attempts
        state.counters.objective_calls - objective_start < max_evaluations || break
        left = right = child_result = nothing
        label = :no_valid_parent
        _age_proposal_begin(diagnostics, state)
        try
        island = state.cursor
        state.cursor = mod1(state.cursor + 1, length(state.populations))
        state.counters.raw_attempts += 1
        progress = _proposal_progress(state.reg.components, state.counters.raw_attempts - 1)
        mw = _age_checked_weight(schedule_value(mutation_weight, progress), :mutation_weight)
        cw = _age_checked_weight(schedule_value(crossover_weight, progress), :crossover_weight)
        valid = _age_valid_records(state, island)
        if isempty(valid)
            state.counters.expected_failures += 1
            _age_add_operator!(state.counters, :no_valid_parent)
            continue
        end
        rng = state.rngs[island]
        tournament_size = Int(get(state.reg.config, :tournament_size, 2))
        left = _age_selected_parent(
            policy, state, island, valid, rng, tournament_size, :parent,
        )
        right = _age_selected_parent(
            policy, state, island, valid, rng, tournament_size, :donor,
        )
        label = :mutation
        child_result = nothing
        type_checked = false
        if proposal !== nothing
            proposal_result = _age_call_proposal(proposal, state, island, left, right, rng)
            proposal_result isa Err || ((label, child_result) = proposal_result)
            proposal_result isa Err && (state.counters.expected_failures += 1; _age_add_operator!(state.counters, :proposal_failure); continue)
        else
            total_weight = mw + (isempty(valid) || get(state.reg.components, :crossover_op, nothing) === nothing ? 0.0 : cw)
            total_weight > 0 || throw(ArgumentError("at least one proposal weight must be positive"))
            choose_crossover = cw > 0 && get(state.reg.components, :crossover_op, nothing) !== nothing &&
                rand(rng) * total_weight >= mw
            if choose_crossover
                label = :crossover
                state.counters.crossover_attempts += 1
                token = _age_span_begin(diagnostics, :crossover)
                child_result = try
                    _age_crossover(state, left, right, rng)
                finally
                    _age_span_end(diagnostics, :crossover, token)
                end
            else
                state.counters.mutation_attempts += 1
                token = _age_span_begin(diagnostics, :mutation)
                child_result = try
                    _age_mutation(state, left, rng; progress, check_admission=false)
                finally
                    _age_span_end(diagnostics, :mutation, token)
                end
                # The checked provider boundary inferred this final tree, or
                # reused the exact ordinary driver's final-tree validation.
                # No provider-controlled trait may bypass that boundary.
                type_checked = true
            end
        end
        _age_add_operator!(state.counters, label)
        if child_result isa Err
            state.counters.expected_failures += 1
            continue
        end
        child_result isa Program || throw(ArgumentError("proposal operator returned an invalid child"))
        if _age_same_program(child_result, left.program) ||
           (label !== :mutation && _age_same_program(child_result, right.program))
            state.counters.unchanged += 1
            continue
        end
        # The guard at the top of the loop reserves one objective call for this
        # candidate; scoring itself may execute every sample exactly once.
        record = _age_score_record!(
            state,
            child_result,
            label === :mutation ? (left.id,) : (left.id, right.id),
            label,
            :advance;
            type_checked,
        )
        if !record.valid
            continue
        end
        record = AgeSearchIndividual(
            record.program, record.id, record.parent_ids, record.origin,
            record.loss, record.complexity, record.valid, _age_birth!(state),
        )
        _age_insert!(state, record, island; policy) || continue
        _age_push_recent!(state, record)
        state.counters.accepted += 1
        finally
            _age_proposal_end(diagnostics, state, label, left, right, child_result)
            _replay_periodic!(replay, state)
        end
    end
    _replay_finish!(replay, state)
    return AgeSearchReport(
        state.counters.raw_attempts - before.raw_attempts,
        state.counters.accepted - accepted_start,
        state.counters.expected_failures - expected_start,
        state.counters.unchanged - unchanged_start,
        state.counters.invalids - invalid_start,
        state.counters.mutation_attempts - mutation_start,
        state.counters.crossover_attempts - crossover_start,
        state.counters.objective_calls - objective_start,
        state.counters.sample_executions - samples_start,
        state.cursor,
    )
end

function _age_migration_parts(value)
    if value isa AgeSearchIndividual
        return value.program, value.parent_ids
    elseif value isa AgeSearchMigration
        return value.program, value.parent_ids
    elseif value isa Program
        return value, Tuple{Vararg{UInt64}}()
    elseif value isa NamedTuple && haskey(value, :program)
        parents = haskey(value, :parent_ids) ?
            Tuple(UInt64(parent) for parent in value.parent_ids) : Tuple{Vararg{UInt64}}()
        value.program isa Program || throw(ArgumentError("migration program must be Program"))
        return value.program, parents
    end
    throw(ArgumentError("migrant must be Program, AgeSearchIndividual, AgeSearchMigration, or a named record"))
end

function _age_live_duplicate(state::AgeSearchState, candidate::Program, population::Int)
    return any(record -> _age_same_program(record.program, candidate), state.populations[population])
end

"""
    inject_age_migrants!(state, individuals; population=1,
                         max_evaluations=length(individuals), max_migrants=...)

Freshly validate and score caller-provided programs in the borrowed task,
then offer valid non-duplicates to `policy` for admission. Omitting it uses
ordinary oldest replacement. Any loss stored on an incoming record is
deliberately ignored; rejected valid candidates are archived and observed but
do not count as accepted migrants. Plain `Program`s have empty ancestry; use
`AgeSearchMigration` or a named record with `parent_ids` to provide provenance.
"""
function inject_age_migrants!(
    state::AgeSearchState,
    individuals;
    population::Int=1,
    max_evaluations::Int=typemax(Int),
    max_migrants::Int=typemax(Int),
    policy::Union{Nothing,AbstractAgeSearchPolicy}=nothing,
)
    1 <= population <= length(state.populations) || throw(BoundsError(state.populations, population))
    max_evaluations >= 0 || throw(ArgumentError("max_evaluations must be non-negative"))
    max_migrants >= 0 || throw(ArgumentError("max_migrants must be non-negative"))
    report = (attempts=0, accepted=0, duplicates=0, invalids=0, objective_calls=0, sample_executions=0)
    (max_evaluations == 0 || max_migrants == 0) && return report
    for value in individuals
        report.attempts < max_migrants || break
        report.objective_calls < max_evaluations || break
        raw, parents = _age_migration_parts(value)
        report = merge(report, (attempts=report.attempts + 1,))
        state.counters.migration_attempts += 1
        _age_validate_candidate(state, raw, "migrant $(report.attempts)")
        if _age_live_duplicate(state, raw, population)
            report = merge(report, (duplicates=report.duplicates + 1,))
            state.counters.migration_duplicates += 1
            continue
        end
        candidate = Program(
            raw.tree;
            local_consts=raw.local_consts,
            local_consts_ty=raw.local_consts_ty,
        )
        before_objective = state.counters.migration_objective_calls
        before_samples = state.counters.migration_sample_executions
        record = _age_score_record!(state, candidate, parents, :migration, :migration)
        report = merge(report, (
            objective_calls=report.objective_calls + state.counters.migration_objective_calls - before_objective,
            sample_executions=report.sample_executions + state.counters.migration_sample_executions - before_samples,
        ))
        if !record.valid
            report = merge(report, (invalids=report.invalids + 1,))
            continue
        end
        record = AgeSearchIndividual(
            record.program, record.id, record.parent_ids, record.origin,
            record.loss, record.complexity, record.valid, _age_birth!(state),
        )
        if _age_insert!(state, record, population; policy)
            report = merge(report, (accepted=report.accepted + 1,))
            state.counters.migration_accepted += 1
        end
    end
    return report
end

########################################################
# Focused state-machine tests
########################################################

@testsnippet AgeSearchBaseImports begin
    using Test
    using Random
    using LambdaRegression.Utils: Err
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode, ConstNode, FunctionDefNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: AbstractMutation, ScalarConstMutation
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        Regressor, SupervisedDataset, default_components, default_config,
        initialize_age_search, advance_age_search!, inject_age_migrants!,
        AgeSearchMigration, AgeSearchNoValidErr

    function age_test_fixture(; target_ty=TyFun(FloatTy, FloatTy), evaluate=nothing, population_size=2)
        base = default_components(
            global_consts=Dict{Symbol,Any}(),
            global_consts_ty=Dict{Symbol,TyLike}(),
            input_slots=Symbol[:x],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy),
            mutations=AbstractMutation[],
            crossover_op=nothing,
        )
        identity = Program(AstTree(AbsNode(:z, IndexNode(0))))
        evaluate === nothing && (evaluate = (ind, x, target_ty, components, config) -> x)
        components = merge(base, (;
            init_population=(reg, data; N, seed, global_type_ctx) ->
                [identity for _ in 1:N],
            evaluate,
        ))
        config = default_config(
            population_size=population_size,
            tournament_size=1,
            error_penalty=99.0,
        )
        reg = Regressor(target_ty, components, config, MersenneTwister(2000))
        data = SupervisedDataset([1.0, 2.0], [1.0, 2.0])
        return reg, data
    end
end

@testitem "Age search initial transform runs before scoring and can stop partial initialization" setup=[AgeSearchBaseImports] begin
    reg, data = age_test_fixture()
    seen = Int[]
    partial = Ref{Any}(nothing)
    transform = (state, program; rng) -> begin
        partial[] = state
        push!(seen, state.counters.objective_calls)
        length(seen) == 2 && return Err(:bounded_stop)
        return Program(program.tree; local_consts=program.local_consts,
            local_consts_ty=program.local_consts_ty)
    end
    result = initialize_age_search(reg, data; population_count=1,
        initial_transform=transform, seed=3)
    @test result isa Err
    @test result.value === :bounded_stop
    @test seen == [0, 1]
    @test partial[].counters.objective_calls == 1
    @test length(partial[].archive) == 1
    @test partial[].best !== nothing
end

@testitem "Age search crossover calls one selected recombine direction" setup=[AgeSearchBaseImports] begin
    Populations = LambdaRegression.Mutations.Populations
    Regression = LambdaRegression.Regression

    mutable struct RecombineSpy <: Populations.AbstractCrossover
        calls::Vector{NamedTuple}
        fail::Bool
        binary_calls::Int
    end
    function Populations.recombine(
        spy::RecombineSpy,
        recipient::Program,
        donor::Program;
        recipient_target,
        donor_target,
        selection_roles,
        kwargs...,
    )
        push!(spy.calls, (; recipient, donor, recipient_target, donor_target, selection_roles))
        return spy.fail ? Err(Populations.NoEligibleCrossoverPairErr()) : recipient
    end
    function Populations.crossover(
        spy::RecombineSpy,
        left::Program,
        right::Program;
        kwargs...,
    )
        spy.binary_calls += 1
        return (left, right)
    end

    struct BinaryOnlySpy <: Populations.AbstractCrossover
        binary_calls::Base.RefValue{Int}
    end
    function Populations.crossover(
        spy::BinaryOnlySpy,
        left::Program,
        right::Program;
        kwargs...,
    )
        spy.binary_calls[] += 1
        return (left, right)
    end

    function age_state(crossover_op)
        reg, data = age_test_fixture(population_size=1)
        components = merge(reg.components, (; crossover_op))
        checked = Regressor(reg.target_ty, components, reg.config, MersenneTwister(42))
        return initialize_age_search(
            checked,
            data;
            population_count=1,
            population_size=1,
            seed=72,
        )
    end
    individual(program, id) = Regression.AgeSearchIndividual(
        program, UInt64(id), (), :test, 0.0, 1, true, UInt64(id),
    )

    left_program = Program(AstTree(AbsNode(:left, IndexNode(0))))
    right_program = Program(AstTree(AbsNode(:right, IndexNode(0))))
    spy = RecombineSpy(NamedTuple[], false, 0)
    state = age_state(spy)
    left, right = individual(left_program, 1), individual(right_program, 2)

    seed = 991
    selected_left = rand(MersenneTwister(seed), Bool)
    child = LambdaRegression.Regression._age_crossover(
        state, left, right, MersenneTwister(seed),
    )
    @test child isa Program
    @test length(spy.calls) == 1
    call = only(spy.calls)
    @test call.recipient === (selected_left ? left.program : right.program)
    @test call.donor === (selected_left ? right.program : left.program)
    @test call.recipient_target == state.reg.target_ty
    @test call.donor_target == state.reg.target_ty
    @test call.selection_roles == (selected_left ? (:left, :right) : (:right, :left))
    @test spy.binary_calls == 0

    empty!(spy.calls)
    spy.fail = true
    failed = LambdaRegression.Regression._age_crossover(
        state, left, right, MersenneTwister(seed),
    )
    @test failed isa Err
    @test length(spy.calls) == 1
    @test spy.binary_calls == 0

    binary_only = BinaryOnlySpy(Ref(0))
    binary_state = age_state(binary_only)
    interface_error = try
        LambdaRegression.Regression._age_crossover(
            binary_state,
            left,
            right,
            MersenneTwister(seed),
        )
        nothing
    catch error
        error
    end
    @test interface_error isa ErrorException
    @test occursin("BinaryOnlySpy", sprint(showerror, interface_error))
    @test occursin("directional recombine required", sprint(showerror, interface_error))
    @test binary_only.binary_calls[] == 0
end

@testitem "Age search chunked continuation preserves cursor and accounting" setup=[AgeSearchBaseImports] begin
    reg, data = age_test_fixture(population_size=2)
    candidate = Program(
        AstTree(AbsNode(:z, ConstNode(:k)));
        local_consts=Dict{Symbol,Any}(:k => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
    )
    proposal = (state, island, left, right; rng) -> candidate
    whole = initialize_age_search(reg, data; population_count=2, population_size=2, seed=11)
    chunked = initialize_age_search(reg, data; population_count=2, population_size=2, seed=11)
    advance_age_search!(whole; max_attempts=4, max_evaluations=4, proposal)
    advance_age_search!(chunked; max_attempts=2, max_evaluations=2, proposal)
    advance_age_search!(chunked; max_attempts=2, max_evaluations=2, proposal)
    @test whole.cursor == chunked.cursor
    @test whole.birth_clock == chunked.birth_clock
    @test whole.counters.raw_attempts == chunked.counters.raw_attempts == 4
    @test whole.counters.advance_objective_calls == chunked.counters.advance_objective_calls
    @test whole.counters.advance_objective_calls <= 4
    @test [[r.birth for r in p] for p in whole.populations] ==
        [[r.birth for r in p] for p in chunked.populations]
end

@testitem "Age search policy hooks select parents, admit records, observe rejection, and rescore migrants" setup=[AgeSearchBaseImports] begin
    Regression = LambdaRegression.Regression

    mutable struct PolicySpy <: Regression.AbstractAgeSearchPolicy
        replacement::Any
        selector::Symbol
        roles::Vector{Symbol}
        observations::Vector{Any}
    end
    PolicySpy(replacement; selector=:valid) =
        PolicySpy(replacement, selector, Symbol[], Any[])

    function Regression.age_select_parent(
        policy::PolicySpy, state, island, candidates, rng, tournament_size,
        role::Symbol,
    )
        push!(policy.roles, role)
        if policy.selector === :forged
            chosen = first(candidates)
            return Regression.AgeSearchIndividual(
                chosen.program, chosen.id, chosen.parent_ids, chosen.origin,
                -1.0e9, chosen.complexity, chosen.valid, chosen.birth,
            )
        end
        return role === :parent ? last(candidates) : first(candidates)
    end

    Regression.age_replacement_index(policy::PolicySpy, state, record, island) =
        policy.replacement
    Regression.age_observe_candidate!(policy::PolicySpy, state, record, island, replaced, admitted) =
        push!(policy.observations, (record, island, replaced, admitted))

    function policy_state(; evaluate=nothing)
        reg, data = age_test_fixture(evaluate=evaluate, population_size=2)
        NodeASTs = LambdaRegression.ASTs.NodeASTs
        components = merge(reg.components, (; complexity=tree ->
            only(NodeASTs.children(tree.root)) isa ConstNode ? 10 : 1))
        checked = Regression.Regressor(reg.target_ty, components, reg.config, MersenneTwister(555))
        state = initialize_age_search(checked, data;
            population_count=1, population_size=2, seed=502)
        return state
    end

    candidate = Program(
        AstTree(AbsNode(:z, ConstNode(:candidate)));
        local_consts=Dict{Symbol,Any}(:candidate => 3.0),
        local_consts_ty=Dict{Symbol,TyLike}(:candidate => FloatTy),
    )

    accepted_state = policy_state()
    old_replacement = accepted_state.populations[1][2]
    accepting = PolicySpy(2)
    proposal = (state, island, left, right; rng) -> candidate
    accepted = advance_age_search!(accepted_state;
        max_attempts=1, max_evaluations=1, proposal, policy=accepting)
    observed = only(accepting.observations)
    @test accepting.roles == [:parent, :donor]
    @test accepted.accepted == 1
    @test observed[1].parent_ids == (UInt64(2), UInt64(1))
    @test observed[1].loss == 0.0
    @test observed[2] == 1
    @test observed[3] === old_replacement
    @test observed[4] === true
    @test accepted_state.populations[1][2] === observed[1]
    @test only(accepted_state.recent_valid_offspring) === observed[1]

    rejected_state = policy_state()
    rejecting = PolicySpy(nothing)
    rejected = advance_age_search!(rejected_state;
        max_attempts=1, max_evaluations=1, proposal, policy=rejecting)
    rejected_record = only(rejecting.observations)[1]
    @test rejected.accepted == 0
    @test rejected.objective_calls == 1
    @test rejected_state.counters.accepted == 0
    @test isempty(rejected_state.recent_valid_offspring)
    @test rejected_state.archive[10] === rejected_record
    @test only(rejecting.observations)[3] === nothing
    @test only(rejecting.observations)[4] === false
    @test all(record -> record.id != rejected_record.id, rejected_state.populations[1])

    evaluate_migrant = (ind, x, target_ty, components, config) ->
        haskey(ind.local_consts, :bad) ? x + 2 : x
    migrant_state = policy_state(evaluate=evaluate_migrant)
    migrant = Program(
        AstTree(AbsNode(:z, ConstNode(:bad)));
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
    )
    stale = Regression.AgeSearchIndividual(
        migrant, UInt64(999), (UInt64(77),), :external,
        -1.0e9, 999, true, UInt64(1),
    )
    migrant_policy = PolicySpy(nothing)
    migration = inject_age_migrants!(migrant_state, [stale];
        max_evaluations=1, max_migrants=1, policy=migrant_policy)
    migrant_observation = only(migrant_policy.observations)
    @test migration.attempts == 1
    @test migration.accepted == 0
    @test migration.objective_calls == 1
    @test migration.sample_executions == 2
    @test migrant_observation[1].id != stale.id
    @test migrant_observation[1].origin === :migration
    @test migrant_observation[1].loss == 4.0
    @test migrant_observation[1].complexity == 10
    @test migrant_observation[1].parent_ids == (UInt64(77),)
    @test migrant_observation[3] === nothing
    @test migrant_observation[4] === false
    @test migrant_state.archive[10] === migrant_observation[1]
    @test migrant_state.counters.migration_accepted == 0

    forged_state = policy_state()
    forged_selection = PolicySpy(1; selector=:forged)
    @test_throws ArgumentError advance_age_search!(forged_state;
        max_attempts=1, max_evaluations=1, proposal, policy=forged_selection)
    @test forged_state.counters.advance_objective_calls == 0

    forged_replacement_state = policy_state()
    forged_replacement = PolicySpy(first(forged_replacement_state.populations[1]))
    @test_throws ArgumentError advance_age_search!(forged_replacement_state;
        max_attempts=1, max_evaluations=1, proposal, policy=forged_replacement)
    @test all(record -> record.id != UInt64(3), forged_replacement_state.populations[1])
end

@testitem "Age search applies candidate admission once before scoring" setup=[AgeSearchBaseImports] begin
    reg, data = age_test_fixture(population_size=1)
    calls = Ref(0)
    components = merge(reg.components, (;
        candidate_admissibility=program -> begin
            calls[] += 1
            return nothing
        end,
    ))
    checked = Regressor(reg.target_ty, components, reg.config, MersenneTwister(2000))
    state = initialize_age_search(checked, data;
        population_count=1, population_size=1, seed=18)
    @test calls[] == 1

    candidate = Program(
        AstTree(AbsNode(:z, ConstNode(:k)));
        local_consts=Dict{Symbol,Any}(:k => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
    )
    proposal = (state, island, left, right; rng) -> candidate
    report = advance_age_search!(state;
        max_attempts=1, max_evaluations=1, proposal)
    @test report.objective_calls == 1
    @test calls[] == 2
end

@testitem "Age search certified mutation skips only the repeated type check" setup=[AgeSearchBaseImports] begin
    @test LambdaRegression.Regression._mutation_result_typechecked(
        LambdaRegression.Mutations.mutate_tree,
    )
    @test !LambdaRegression.Regression._mutation_result_typechecked((args...; kwargs...) -> nothing)
    reg, data = age_test_fixture(population_size=1)
    factory_calls = Ref(0)
    original_factory = reg.components.tyfactory_ctor
    components = merge(reg.components, (;
        tyfactory_ctor=() -> begin
            factory_calls[] += 1
            return original_factory()
        end,
    ))
    checked = Regressor(reg.target_ty, components, reg.config, MersenneTwister(1111))
    state = initialize_age_search(checked, data;
        population_count=1, population_size=1, seed=20)
    factory_calls[] = 0
    candidate = only(state.populations[1]).program
    @test LambdaRegression.Regression._age_validate_candidate(
        state, candidate, "certified"; type_checked=true,
    ) === nothing
    @test factory_calls[] == 0
    @test LambdaRegression.Regression._age_validate_candidate(
        state, candidate, "ordinary",
    ) === nothing
    @test factory_calls[] == 1

    unresolved = Program(AstTree(AbsNode(:z, ConstNode(:missing))))
    @test_throws ArgumentError LambdaRegression.Regression._age_validate_candidate(
        state, unresolved, "bad"; type_checked=true,
    )
end

@testitem "Age search borrows task and data and preserves caller resources and init seeds" setup=[AgeSearchBaseImports] begin
    globals = Dict{Symbol,Any}(:offset => 1.0)
    global_types = Dict{Symbol,TyLike}(:offset => FloatTy)
    inputs = Symbol[:x]
    input_types = Dict{Symbol,TyLike}(:x => FloatTy)
    observed_diagnostics = Any[]
    observer = diagnostics -> push!(observed_diagnostics, diagnostics)
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=inputs,
        input_types=input_types,
        mutations=AbstractMutation[],
        crossover_op=nothing,
        initialization_observer=observer,
        force_lambda_root_probability=1.0,
        init_candidates_per_individual=1,
        init_target_nodes=(2,),
        init_max_try=2,
    )

    default_init = components.init_population
    init_seeds = UInt64[]
    borrowed_data = SupervisedDataset([1.0, 2.0], [1.0, 2.0])
    original_X = copy(borrowed_data.X)
    original_Y = copy(borrowed_data.Y)
    init_population = function (init_reg, init_data; N, seed, global_type_ctx)
        # The default initializer closes over these exact world dictionaries.
        @test init_reg.components.global_consts === globals
        @test init_reg.components.global_consts_ty === global_types
        @test init_data === borrowed_data
        push!(init_seeds, seed)
        default_init(init_reg, init_data; N, seed, global_type_ctx)
    end
    components = merge(components, (; init_population, init_seeds))
    config = default_config(
        population_size=1,
        tournament_size=1,
        error_penalty=99.0,
    )
    reg = Regressor(TyFun(FloatTy, FloatTy), components, config, MersenneTwister(1234))
    root_seed = UInt64(41)

    state = initialize_age_search(
        reg, borrowed_data; population_count=2, population_size=1, seed=root_seed,
    )

    expected_seeds = [
        LambdaRegression.Regression._derive_seed(root_seed, :age_init, island)
        for island in 1:2
    ]
    @test state.reg === reg
    @test state.data === borrowed_data
    @test state.reg.components.init_population === init_population
    @test state.reg.components.init_seeds === init_seeds
    @test init_seeds == expected_seeds
    @test [rand(rng, UInt64) for rng in state.rngs] == [
        rand(Random.MersenneTwister(UInt32[
            UInt32(seed & 0xffffffff), UInt32(seed >> 32),
        ]), UInt64) for seed in expected_seeds
    ]
    @test state.counters.init_objective_calls == 2
    @test state.counters.init_sample_executions == 4
    @test state.birth_clock == 2
    @test length(observed_diagnostics) == 2
    @test state.reg.components.initialization_observer === observer
    @test borrowed_data.X == original_X
    @test borrowed_data.Y == original_Y
end

@testitem "Age search accepts worse and neutral valid births" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> haskey(ind.local_consts, :bad) ? x + 2 : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=2)
    state = initialize_age_search(reg, data; population_count=1, population_size=2, seed=12)
    bad = Program(
        AstTree(AbsNode(:z, ConstNode(:bad)));
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
    )
    old_best = state.best
    proposal = (state, island, left, right; rng) -> bad
    report = advance_age_search!(state; max_attempts=1, max_evaluations=1, proposal)
    @test report.accepted == 1
    @test state.best.id == old_best.id
    @test any(record -> record.origin == :proposal && record.loss > old_best.loss, state.populations[1])
    @test state.birth_clock == 3
end

@testitem "Age search bounds invalid and unchanged proposals and zero budget" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> haskey(ind.local_consts, :bad) ? Err(:invalid) : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    state = initialize_age_search(reg, data; population_count=1, population_size=1, seed=13)
    invalid = Program(
        AstTree(AbsNode(:z, ConstNode(:bad)));
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
    )
    n = Ref(0)
    proposal = (state, island, left, right; rng) -> begin
        n[] += 1
        n[] == 1 ? (:invalid, invalid) : (:clone, left.program)
    end
    report = advance_age_search!(state; max_attempts=2, max_evaluations=1, proposal)
    @test report.attempts == 1
    @test report.invalids == 1
    @test state.counters.advance_objective_calls == 1
    before = (state.cursor, state.birth_clock, state.counters.raw_attempts)
    zero = advance_age_search!(state; max_attempts=4, max_evaluations=0, proposal)
    @test zero.attempts == 0
    @test (state.cursor, state.birth_clock, state.counters.raw_attempts) == before
end

@testitem "Age search migration owns candidates, rescoring stale loss, and archives old best" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> haskey(ind.local_consts, :bad) ? x + 2 : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    state = initialize_age_search(reg, data; population_count=1, population_size=1, seed=14)
    old_best = state.best
    migration_before = (
        attempts=state.counters.migration_attempts,
        objective_calls=state.counters.migration_objective_calls,
        sample_executions=state.counters.migration_sample_executions,
        accepted=state.counters.migration_accepted,
    )
    migrant = Program(
        AstTree(AbsNode(:z, ConstNode(:bad)));
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
    )
    migration = inject_age_migrants!(
        state,
        [AgeSearchMigration(migrant, (UInt64(77),))];
        population=1,
        max_evaluations=1,
        max_migrants=1,
    )
    migrant.local_consts[:bad] = 123.0
    @test migration.accepted == 1
    @test migration.objective_calls == 1
    @test migration.sample_executions == 2
    @test state.counters.migration_attempts == migration_before.attempts + 1
    @test state.counters.migration_objective_calls == migration_before.objective_calls + 1
    @test state.counters.migration_sample_executions == migration_before.sample_executions + 2
    @test state.counters.migration_accepted == migration_before.accepted + 1
    @test state.populations[1][1].loss == 4.0
    @test state.populations[1][1].parent_ids == (UInt64(77),)
    @test state.archive[old_best.complexity].id == old_best.id
    @test state.populations[1][1].program.local_consts[:bad] == 1.0
end

@testitem "Age search propagates unexpected evaluator exceptions" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> error("unexpected evaluator failure")
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    @test_throws ErrorException initialize_age_search(reg, data; population_count=1, population_size=1, seed=15)
end

@testitem "Age search honors strict errors for unknown returned Err" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> Err(ErrorException("unknown candidate failure"))
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    strict_config = merge(reg.config, (; strict_errors=true))
    strict_reg = Regressor(reg.target_ty, reg.components, strict_config, MersenneTwister(555))
    @test_throws ErrorException initialize_age_search(strict_reg, data; population_count=1, population_size=1, seed=152)
end

@testitem "Age search shared task scoring penalizes each bad sample and continues" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, target_ty, components, config) -> x == 2.0 ? Err(:invalid) : x
    reg, _ = age_test_fixture(evaluate=evaluate, population_size=1)
    data = SupervisedDataset([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    state = initialize_age_search(reg, data; population_count=1, population_size=1, seed=153)
    record = only(state.populations[1])
    @test record.valid == false
    @test record.loss == 33.0
    @test state.counters.init_sample_executions == 3
    @test state.counters.sample_executions == 3
    @test state.counters.init_invalids == 1
    @test state.counters.invalids == 1

    should_throw = Ref(false)
    evaluator = (ind, x, target_ty, components, config) ->
        should_throw[] ? error("unexpected after initialization") : x
    throwing_reg, throwing_data = age_test_fixture(
        evaluate=evaluator, population_size=1,
    )
    throwing_state = initialize_age_search(
        throwing_reg, throwing_data; population_count=1, population_size=1, seed=154,
    )
    should_throw[] = true
    @test_throws ErrorException LambdaRegression.Regression._age_score!(
        throwing_state, only(throwing_state.populations[1]).program, :advance,
    )
    @test throwing_state.counters.advance_sample_executions == 1
end

@testitem "Age search rejects empty objective data" setup=[AgeSearchBaseImports] begin
    reg, _ = age_test_fixture(population_size=1)
    empty_data = SupervisedDataset(Float64[], Float64[])
    result = initialize_age_search(reg, empty_data; population_count=1, population_size=1, seed=151)
    @test result isa Err
    @test result.value isa ArgumentError
end

@testitem "Age search validates a real complete FunctionDef binding AST" setup=[AgeSearchBaseImports] begin
    target_ty = TyFun(FloatTy, FloatTy)
    tree = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0))))
    individual = Program(tree)
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        mutations=AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, seed, global_type_ctx) -> [individual for _ in 1:N],
    )
    reg = Regressor(target_ty, base, default_config(population_size=1), MersenneTwister(555))
    state = initialize_age_search(reg, SupervisedDataset([3.0], [3.0]); population_count=1, population_size=1, seed=16)
    @test state.populations[1][1].valid
    @test state.populations[1][1].loss == 0.0
end

@testitem "Age search invokes the real mutation toolbox once" setup=[AgeSearchBaseImports] begin
    candidate = Program(
        AstTree(AbsNode(:z, ConstNode(:k)));
        local_consts=Dict{Symbol,Any}(:k => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
    )
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        mutations=AbstractMutation[ScalarConstMutation(σ=0.2)],
        crossover_op=nothing,
        mutation_max_try=1,
        init_population=(reg, data; N, seed, global_type_ctx) -> [candidate for _ in 1:N],
    )
    components = merge(base, (;
        evaluate=(ind, x, target_ty, c, cfg) -> first(values(ind.local_consts)),
    ))
    reg = Regressor(TyFun(FloatTy, FloatTy), components, default_config(population_size=1), MersenneTwister(2000))
    state = initialize_age_search(reg, SupervisedDataset([0.0], [1.0]); population_count=1, population_size=1, seed=17)
    report = advance_age_search!(state; max_attempts=1, max_evaluations=1, mutation_weight=1.0, crossover_weight=0.0)
    @test report.mutation_attempts == 1
    @test report.accepted == 1
    @test state.populations[1][1].program.local_consts != candidate.local_consts
end
