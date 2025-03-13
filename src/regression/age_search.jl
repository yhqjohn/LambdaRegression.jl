"""
    AgeSearch

Small resumable outer search over complete lambda/FunctionDef AST individuals.

The state owns its frozen task, one RNG stream per island, populations, a
bounded recent-offspring buffer, and a loss-by-written-complexity archive.
`advance_age_search!` performs a bounded number of *outer* proposal attempts;
the existing mutation and crossover implementations retain ownership of their
inner retry limits. This layer intentionally does not add scheduling policy,
teacher structures, or a second evaluator. The state copies the dataset,
regressor container, and candidate material; injected component callables and
external resources remain caller-owned and must stay unchanged for the state
lifetime. A changed task or component world requires a new state.
"""

export AgeSearchState, AgeSearchIndividual, AgeSearchMigration
export AgeSearchCounters, AgeSearchReport
export AgeSearchNoValidErr
export initialize_age_search, advance_age_search!, inject_age_migrants!

"""Finite failure returned when every island currently lacks a valid candidate."""
struct AgeSearchNoValidErr end

"""
    AgeSearchIndividual

Owned, scored record used by an age-search population. `parent_ids` is an
explicit ancestry tuple; an empty tuple means that the caller did not provide
provenance (for example, a plain migrant `Individual`).
"""
struct AgeSearchIndividual
    individual::Individual
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

"""Mutable state for a persistent complete-AST age search."""
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
    individual::Individual
    parent_ids::Tuple{Vararg{UInt64}}
end

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
    eval_fn = get(c, :evaluate, evaluate_default)
    eval_fn === evaluate_default && isempty(slots) && return ArgumentError(
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

function _age_copy_individual(ind::Individual)
    return Individual(
        deepcopy(ind.tree), ind.target_ty;
        local_consts=deepcopy(ind.local_consts),
        local_consts_ty=deepcopy(ind.local_consts_ty),
        fitness=ind.fitness,
        meta=deepcopy(ind.meta),
    )
end

function _age_same_individual(a::Individual, b::Individual)
    return a.target_ty == b.target_ty && a.tree == b.tree &&
        isequal(a.local_consts, b.local_consts) &&
        isequal(a.local_consts_ty, b.local_consts_ty)
end

function _age_same_target(a::Individual, b::Individual)
    return TypeSystem.iscompatible(a.target_ty, b.target_ty) &&
        TypeSystem.iscompatible(b.target_ty, a.target_ty)
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

function _age_record(
    state::AgeSearchState,
    ind::Individual,
    parent_ids::Tuple{Vararg{UInt64}},
    origin::Symbol,
    loss::Float64,
    complexity::Int,
    valid::Bool;
    birth::UInt64=UInt64(0),
)
    owned = _age_copy_individual(ind)
    scored = Individual(
        owned.tree, owned.target_ty;
        local_consts=owned.local_consts,
        local_consts_ty=owned.local_consts_ty,
        fitness=valid ? loss : nothing,
        meta=owned.meta,
    )
    return AgeSearchIndividual(
        scored, _age_next_id!(state), parent_ids, origin, loss, complexity,
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

function _age_insert!(state::AgeSearchState, record::AgeSearchIndividual, population::Int)
    pop = state.populations[population]
    index = _age_oldest_index(pop)
    pop[index] = record
    _age_archive!(state, record)
    return nothing
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

function _age_validate_candidate(
    state::AgeSearchState,
    ind::Individual,
    label::AbstractString,
)
    issue = _validate_individual_contract(ind, state.reg, state.reg.components, label)
    issue === nothing || throw(issue)
    policy = get(state.reg.components, :bloat_policy, Mutations.BloatPolicy())
    _within_bloat_limit(ind.tree, policy) || return :bloat
    rejection = _individual_candidate_rejection(ind, state.reg.components)
    rejection === nothing || return rejection
    return nothing
end

"""Score one candidate and account for its original-task objective work."""
function _age_score!(state::AgeSearchState, ind::Individual, source::Symbol)
    components = state.reg.components
    config = state.reg.config
    eval_fn = get(components, :evaluate, evaluate_default)
    loss_fn = get(components, :loss, mse)
    penalty = _age_error_penalty(state.reg)
    complexity_fn = get(components, :complexity, _node_count)
    complexity = complexity_fn(ind.tree)
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

    rejection = _individual_candidate_rejection(ind, components)
    if rejection !== nothing
        state.counters.invalids += 1
        if source === :init
            state.counters.init_invalids += 1
        elseif source === :migration
            state.counters.migration_invalids += 1
        else
            state.counters.advance_invalids += 1
        end
        return false, penalty, complexity
    end

    prepared = eval_fn === evaluate_default ?
        prepare_default_evaluator(ind, components, config; copy_tree=true) : nothing
    total = 0.0
    valid = true
    for (x, y) in zip(state.data.X, state.data.Y)
        state.counters.sample_executions += 1
        if source === :init
            state.counters.init_sample_executions += 1
        elseif source === :migration
            state.counters.migration_sample_executions += 1
        else
            state.counters.advance_sample_executions += 1
        end
        yhat = try
            prepared === nothing ? eval_fn(ind, x, components, config) : prepared(x)
        catch issue
            _age_expected_exception(issue, components) || rethrow()
            valid = false
            total += penalty
            continue
        end
        if yhat isa Err
            strict_errors = get(config, :strict_errors, false)
            strict_errors && !_age_expected_exception(yhat.value, components) && throw(yhat.value)
            valid = false
            total += penalty
            continue
        elseif !_finite_prediction(yhat)
            valid = false
            total += penalty
            continue
        end
        sample_loss = try
            loss_fn(yhat, y)
        catch issue
            _age_expected_exception(issue, components; loss=true) || rethrow()
            valid = false
            total += penalty
            continue
        end
        finite_loss = _finite_real_loss(sample_loss)
        if finite_loss === nothing
            valid = false
            total += penalty
        else
            total += finite_loss
        end
    end
    loss = total / max(length(state.data.X), 1)
    isfinite(loss) || (valid = false; loss = penalty)
    if !valid
        state.counters.invalids += 1
        if source === :init
            state.counters.init_invalids += 1
        elseif source === :migration
            state.counters.migration_invalids += 1
        else
            state.counters.advance_invalids += 1
        end
    end
    return valid, Float64(loss), complexity
end

function _age_score_record!(
    state::AgeSearchState,
    ind::Individual,
    parent_ids::Tuple{Vararg{UInt64}},
    origin::Symbol,
    source::Symbol,
)
    admissibility = _age_validate_candidate(state, ind, "$(origin) candidate")
    if admissibility !== nothing
        complexity_fn = get(state.reg.components, :complexity, _node_count)
        complexity = Int(complexity_fn(ind.tree))
        state.counters.invalids += 1
        if source === :init
            state.counters.init_invalids += 1
        elseif source === :migration
            state.counters.migration_invalids += 1
        else
            state.counters.advance_invalids += 1
        end
        return _age_record(
            state,
            ind,
            parent_ids,
            origin,
            _age_error_penalty(state.reg),
            complexity,
            false,
        )
    end
    valid, loss, complexity = _age_score!(state, ind, source)
    return _age_record(state, ind, parent_ids, origin, loss, complexity, valid)
end

function _age_mutation(state::AgeSearchState, parent::AgeSearchIndividual, rng::AbstractRNG)
    c = state.reg.components
    mutation_driver = get(c, :mutation_driver, Mutations.mutate_tree)
    local_consts = copy(parent.individual.local_consts)
    local_consts_ty = copy(parent.individual.local_consts_ty)
    mutated = mutation_driver(
        parent.individual.tree,
        state.reg.target_ty;
        rng,
        mutations=c.mutations,
        chooser=c.mutation_chooser,
        provider=c.mutation_provider,
        max_try=c.mutation_max_try,
        prune_consts=c.mutation_prune_consts,
        copy_tree=true,
        copy_ctx=true,
        global_consts=c.global_consts,
        global_consts_ty=merge(c.global_consts_ty, c.input_types),
        local_consts,
        local_consts_ty,
        tyfactory=c.tyfactory_ctor(),
        symfactory=SymbolGenerator(),
        const_symfactory=c.const_symfactory_ctor(),
        bloat_policy=get(c, :bloat_policy, Mutations.BloatPolicy()),
        adaptive_policy=get(c, :mutation_adaptive_policy, Mutations.AdaptivePolicy()),
        type_inference=get(c, :type_inference, nothing),
    )
    mutated isa Err && return mutated
    mutated isa AstTree || throw(ArgumentError("mutation_driver must return AstTree or Err"))
    return Individual(
        mutated, state.reg.target_ty;
        local_consts, local_consts_ty,
        fitness=nothing,
        meta=(;),
    )
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
    children = Populations.crossover(
        op,
        left.individual,
        right.individual;
        rng,
        global_consts_ty=merge(c.global_consts_ty, c.input_types),
        tyfactory_ctor=c.tyfactory_ctor,
        const_symfactory_ctor=c.const_symfactory_ctor,
        prune=true,
    )
    children isa Err && return children
    children isa Tuple && length(children) == 2 || throw(ArgumentError(
        "crossover_op must return a two-child tuple or Err",
    ))
    chosen = rand(rng, 1:2)
    child = children[chosen]
    child isa Individual || throw(ArgumentError("crossover children must be Individuals"))
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
    if result isa Individual
        return (:proposal, result)
    elseif result isa Tuple && length(result) == 2 && result[1] isa Symbol
        result[2] isa Individual || throw(ArgumentError("proposal child must be an Individual"))
        return (result[1], result[2])
    end
    throw(ArgumentError("proposal must return Individual, (label, Individual), or Err"))
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
                          seed=nothing, recent_capacity=32)

Construct and score each island exactly once through `init_population`, with a
distinct derived seed per island. Initialization has no hidden population fill;
invalid records remain visible but cannot breed.
"""
function initialize_age_search(
    reg::Regressor,
    data::SupervisedDataset;
    population_count::Int=2,
    population_size::Int=get(reg.config, :population_size, 0),
    seed=nothing,
    recent_capacity::Int=32,
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
    frozen_reg = deepcopy(reg)
    frozen_data = SupervisedDataset(deepcopy(data.X), deepcopy(data.Y))
    state = AgeSearchState(
        frozen_reg,
        frozen_data,
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
    global_type_ctx = merge(frozen_reg.components.global_consts_ty, frozen_reg.components.input_types)
    for island in 1:population_count
        island_seed = _derive_seed(root_seed, :age_init, island)
        push!(state.rngs, MersenneTwister(UInt32[UInt32(island_seed & 0xffffffff), UInt32(island_seed >> 32)]))
        result = frozen_reg.components.init_population(
            frozen_reg,
            frozen_data;
            N=population_size,
            seed=island_seed,
            global_type_ctx=global_type_ctx,
        )
        result isa Err && return result
        issue = _preflight_initial_population(result, frozen_reg, frozen_reg.components, population_size)
        issue === nothing || return Err(issue)
        pop = AgeSearchIndividual[]
        for (slot, raw) in enumerate(result)
            candidate = _age_copy_individual(raw)
            record = _age_score_record!(state, candidate, (), :initial, :init)
            if record.birth == 0
                record = AgeSearchIndividual(
                    record.individual, record.id, record.parent_ids, record.origin,
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
candidate objective calls. Each changed candidate is scored in the frozen
original task; a valid changed candidate replaces the oldest member of its
round-robin island, regardless of whether its loss improves.
"""
function advance_age_search!(
    state::AgeSearchState;
    max_attempts::Int,
    max_evaluations::Int,
    mutation_weight::Real=1.0,
    crossover_weight::Real=1.0,
    proposal=nothing,
)
    max_attempts >= 0 || throw(ArgumentError("max_attempts must be non-negative"))
    max_evaluations >= 0 || throw(ArgumentError("max_evaluations must be non-negative"))
    mw = _age_checked_weight(mutation_weight, :mutation_weight)
    cw = _age_checked_weight(crossover_weight, :crossover_weight)
    if max_attempts == 0 || max_evaluations == 0
        return AgeSearchReport(0, 0, 0, 0, 0, 0, 0, 0, 0, state.cursor)
    end
    any(!isempty(_age_valid_records(state, island)) for island in eachindex(state.populations)) ||
        return Err(AgeSearchNoValidErr())
    before = _age_snapshot(state.counters)
    objective_start = state.counters.objective_calls
    samples_start = state.counters.sample_executions
    accepted_start = state.counters.accepted
    expected_start = state.counters.expected_failures
    unchanged_start = state.counters.unchanged
    invalid_start = state.counters.invalids
    mutation_start = state.counters.mutation_attempts
    crossover_start = state.counters.crossover_attempts

    for _ in 1:max_attempts
        state.counters.objective_calls - objective_start < max_evaluations || break
        island = state.cursor
        state.cursor = mod1(state.cursor + 1, length(state.populations))
        state.counters.raw_attempts += 1
        valid = _age_valid_records(state, island)
        if isempty(valid)
            state.counters.expected_failures += 1
            _age_add_operator!(state.counters, :no_valid_parent)
            continue
        end
        rng = state.rngs[island]
        tournament_size = Int(get(state.reg.config, :tournament_size, 2))
        left = _age_tournament(rng, valid, tournament_size)
        right = _age_tournament(rng, valid, tournament_size)
        label = :mutation
        child_result = nothing
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
                child_result = _age_crossover(state, left, right, rng)
            else
                state.counters.mutation_attempts += 1
                child_result = _age_mutation(state, left, rng)
            end
        end
        _age_add_operator!(state.counters, label)
        if child_result isa Err
            state.counters.expected_failures += 1
            continue
        end
        child_result isa Individual || throw(ArgumentError("proposal operator returned an invalid child"))
        if _age_same_individual(child_result, left.individual) ||
           (label !== :mutation && _age_same_individual(child_result, right.individual))
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
            :advance,
        )
        if !record.valid
            continue
        end
        record = AgeSearchIndividual(
            record.individual, record.id, record.parent_ids, record.origin,
            record.loss, record.complexity, record.valid, _age_birth!(state),
        )
        _age_insert!(state, record, island)
        _age_push_recent!(state, record)
        state.counters.accepted += 1
    end
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
        return value.individual, value.parent_ids
    elseif value isa AgeSearchMigration
        return value.individual, value.parent_ids
    elseif value isa Individual
        return value, Tuple{Vararg{UInt64}}()
    elseif value isa NamedTuple && haskey(value, :individual)
        parents = haskey(value, :parent_ids) ?
            Tuple(UInt64(parent) for parent in value.parent_ids) : Tuple{Vararg{UInt64}}()
        value.individual isa Individual || throw(ArgumentError("migration individual must be Individual"))
        return value.individual, parents
    end
    throw(ArgumentError("migrant must be Individual, AgeSearchIndividual, AgeSearchMigration, or a named record"))
end

function _age_live_duplicate(state::AgeSearchState, candidate::Individual, population::Int)
    return any(record -> _age_same_individual(record.individual, candidate), state.populations[population])
end

"""
    inject_age_migrants!(state, individuals; population=1,
                         max_evaluations=length(individuals), max_migrants=...)

Freshly validate and score caller-provided individuals in the frozen task,
then insert valid non-duplicates using ordinary oldest replacement. Any loss
stored on an incoming record is deliberately ignored. Plain `Individual`s have
empty ancestry; use `AgeSearchMigration` or a named record with `parent_ids` to
provide provenance explicitly.
"""
function inject_age_migrants!(
    state::AgeSearchState,
    individuals;
    population::Int=1,
    max_evaluations::Int=typemax(Int),
    max_migrants::Int=typemax(Int),
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
        candidate = _age_copy_individual(raw)
        _age_validate_candidate(state, candidate, "migrant $(report.attempts)")
        if _age_live_duplicate(state, candidate, population)
            report = merge(report, (duplicates=report.duplicates + 1,))
            state.counters.migration_duplicates += 1
            continue
        end
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
            record.individual, record.id, record.parent_ids, record.origin,
            record.loss, record.complexity, record.valid, _age_birth!(state),
        )
        _age_insert!(state, record, population)
        report = merge(report, (accepted=report.accepted + 1,))
        state.counters.migration_accepted += 1
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
    using LambdaRegression: Procedural
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: AbstractMutation, ScalarConstMutation
    using LambdaRegression.Mutations.Populations: Individual
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
        identity = Individual(AstTree(AbsNode(:z, IndexNode(0))), target_ty)
        evaluate === nothing && (evaluate = (ind, x, components, config) -> x)
        components = merge(base, (;
            init_population=(reg, data; N, seed, global_type_ctx) ->
                [identity for _ in 1:N],
            evaluate,
        ))
        config = default_config(
            population_size=population_size,
            generations=1,
            tournament_size=1,
            error_penalty=99.0,
        )
        reg = Regressor(target_ty, components, config, MersenneTwister(2000))
        data = SupervisedDataset([1.0, 2.0], [1.0, 2.0])
        return reg, data
    end
end

@testitem "Age search chunked continuation preserves cursor and accounting" setup=[AgeSearchBaseImports] begin
    reg, data = age_test_fixture(population_size=2)
    candidate = Individual(
        AstTree(AbsNode(:z, ConstNode(:k))),
        TyFun(FloatTy, FloatTy);
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

@testitem "Age search accepts worse and neutral valid births" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, components, config) -> haskey(ind.local_consts, :bad) ? x + 2 : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=2)
    state = initialize_age_search(reg, data; population_count=1, population_size=2, seed=12)
    bad = Individual(
        AstTree(AbsNode(:z, ConstNode(:bad))), TyFun(FloatTy, FloatTy);
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
    evaluate = (ind, x, components, config) -> haskey(ind.local_consts, :bad) ? Err(:invalid) : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    state = initialize_age_search(reg, data; population_count=1, population_size=1, seed=13)
    invalid = Individual(
        AstTree(AbsNode(:z, ConstNode(:bad))), TyFun(FloatTy, FloatTy);
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
    )
    n = Ref(0)
    proposal = (state, island, left, right; rng) -> begin
        n[] += 1
        n[] == 1 ? (:invalid, invalid) : (:clone, left.individual)
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
    evaluate = (ind, x, components, config) -> haskey(ind.local_consts, :bad) ? x + 2 : x
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    state = initialize_age_search(reg, data; population_count=1, population_size=1, seed=14)
    old_best = state.best
    migrant = Individual(
        AstTree(AbsNode(:z, ConstNode(:bad))), TyFun(FloatTy, FloatTy);
        local_consts=Dict{Symbol,Any}(:bad => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy),
        fitness=0.0,
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
    @test state.populations[1][1].loss == 4.0
    @test state.populations[1][1].parent_ids == (UInt64(77),)
    @test state.archive[old_best.complexity].id == old_best.id
    @test state.populations[1][1].individual.local_consts[:bad] == 1.0
end

@testitem "Age search propagates unexpected evaluator exceptions" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, components, config) -> error("unexpected evaluator failure")
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    @test_throws ErrorException initialize_age_search(reg, data; population_count=1, population_size=1, seed=15)
end

@testitem "Age search honors strict errors for unknown returned Err" setup=[AgeSearchBaseImports] begin
    evaluate = (ind, x, components, config) -> Err(ErrorException("unknown candidate failure"))
    reg, data = age_test_fixture(evaluate=evaluate, population_size=1)
    strict_config = merge(reg.config, (; strict_errors=true))
    strict_reg = Regressor(reg.target_ty, reg.components, strict_config, MersenneTwister(555))
    @test_throws ErrorException initialize_age_search(strict_reg, data; population_count=1, population_size=1, seed=152)
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
    tree = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    individual = Individual(tree, target_ty)
    base = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        mutations=AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, seed, global_type_ctx) -> [individual for _ in 1:N],
    )
    reg = Regressor(target_ty, base, default_config(population_size=1, generations=1), MersenneTwister(555))
    state = initialize_age_search(reg, SupervisedDataset([3.0], [3.0]); population_count=1, population_size=1, seed=16)
    @test state.populations[1][1].valid
    @test state.populations[1][1].loss == 0.0
end

@testitem "Age search invokes the real mutation toolbox once" setup=[AgeSearchBaseImports] begin
    candidate = Individual(
        AstTree(AbsNode(:z, ConstNode(:k))), TyFun(FloatTy, FloatTy);
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
        evaluate=(ind, x, c, cfg) -> first(values(ind.local_consts)),
    ))
    reg = Regressor(TyFun(FloatTy, FloatTy), components, default_config(population_size=1, generations=1), MersenneTwister(2000))
    state = initialize_age_search(reg, SupervisedDataset([0.0], [1.0]); population_count=1, population_size=1, seed=17)
    report = advance_age_search!(state; max_attempts=1, max_evaluations=1, mutation_weight=1.0, crossover_weight=0.0)
    @test report.mutation_attempts == 1
    @test report.accepted == 1
    @test state.populations[1][1].individual.local_consts != candidate.local_consts
end
