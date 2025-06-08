"""
    age_parallel_status(state; policy=nothing, proposal=nothing, replay=nothing)

Report the effective outer-proposal parallelism and any serial fallback reason.
Default-policy proposals are batched. When workers outnumber islands, multiple
proposals use each island's batch-start population; this changes the trajectory.
Stateful selection hooks,
replay, diagnostics, external mutation providers and custom scoring stay serial.
Enabling parallel execution requires borrowed data, host primitives, type/constant
factories and custom generation/mutation extensions to support concurrent read-only use. The built-in
mutable site index and generation relation caches are isolated automatically.
"""
function age_parallel_status(state::AgeSearchState;
        policy=nothing, proposal=nothing, replay=nothing)
    parallel = _normalize_parallel_config(get(state.reg.config, :parallel, (;)))
    workers = min(Int(get(parallel, :workers, Threads.nthreads(:default))),
        Threads.nthreads(:default))
    c = state.reg.components
    reason = if !parallel.enabled
        :disabled
    elseif workers <= 1
        :single_worker
    elseif policy !== nothing || proposal !== nothing || replay !== nothing ||
            get(c, :age_diagnostics, nothing) !== nothing
        :stateful_hooks
    elseif !(get(c, :full_mutation_provider, OrdinaryMutationProvider()) isa OrdinaryMutationProvider) ||
            get(c, :mutation_driver, Mutations.mutate_tree) !== Mutations.mutate_tree ||
            !(get(c, :mutation_provider, nothing) isa Mutations.IndexedProvider)
        :custom_mutation
    elseif get(c, :crossover_op, nothing) !== nothing &&
            !(c.crossover_op isa Populations.TypedSubtreeSwapCrossover &&
              c.crossover_op.selection isa Populations.UniformCrossoverSelectionPolicy)
        :custom_crossover
    elseif haskey(c, :evaluate) || haskey(c, :prepare_evaluate) ||
            get(c, :loss, mse) !== mse || get(c, :complexity, _node_count) !== _node_count ||
            get(c, :make_eval_context, nothing) !== get(c, :_parallel_default_eval_context, missing) ||
            get(c, :candidate_admissibility, nothing) !== _procedural_candidate_admissibility ||
            get(c, :candidate_failure, _known_candidate_exception) !== _known_candidate_exception ||
            get(c, :loss_failure, _no_candidate_exception) !== _no_candidate_exception
        :custom_scoring
    else
        workers > length(state.populations) ? :batched_islands : :independent_islands
    end
    enabled = reason in (:independent_islands, :batched_islands)
    return (; enabled, workers=enabled ? workers : 1, reason,
        trajectory_changes=enabled && workers > length(state.populations))
end

# One existing serial attempt owns this small scratch state. Programs are
# borrowed read-only; population slots, RNG and all bookkeeping are private.
# In particular, workers never advance the caller's RNG before ordered commit.
function _age_attempt_state(state, island, reg, ordinal)
    counters = AgeSearchCounters()
    counters.raw_attempts = ordinal
    return AgeSearchState(reg, state.data, state.seed,
        [copy(state.populations[island])], AbstractRNG[copy(state.rngs[island])],
        1, state.birth_clock, UInt64(1), AgeSearchIndividual[], 1,
        Dict{Int,AgeSearchIndividual}(), nothing, counters)
end

function _age_commit_attempt!(state, island, scratch, ordinal; commit_rng=true)
    # A rejected/invalid scored record still consumes an ID in the serial loop.
    next_id = state.next_id
    state.next_id += scratch.next_id - UInt64(1)
    commit_rng && (state.rngs[island] = scratch.rngs[1])
    state.cursor = mod1(island + 1, length(state.populations))
    for name in fieldnames(AgeSearchCounters)
        name === :operator_labels && continue
        value = getfield(scratch.counters, name)
        name === :raw_attempts && (value -= ordinal)
        setfield!(state.counters, name, getfield(state.counters, name) + value)
    end
    for (label, count) in scratch.counters.operator_labels
        state.counters.operator_labels[label] =
            get(state.counters.operator_labels, label, 0) + count
    end
    if !isempty(scratch.recent_valid_offspring)
        local_record = only(scratch.recent_valid_offspring)
        record = AgeSearchIndividual(local_record.program, next_id,
            local_record.parent_ids, local_record.origin, local_record.loss,
            local_record.complexity, true, _age_birth!(state))
        _age_insert!(state, record, island)
        _age_push_recent!(state, record)
    end
    return nothing
end

"""
    advance_age_search!(state; max_attempts, max_evaluations, kwargs...)

Advance bounded complete-program proposals and return an `AgeSearchReport`.
The serial engine retains policy, proposal, replay and diagnostic semantics.
With `config.parallel=(enabled=true, workers=8)`, default-policy proposals run
in parallel, including construction, validation and full-task scoring.
When workers outnumber islands, a batch may propose multiple children from the
same island snapshot. A once-per-logger warning reports the changed trajectory.
`age_parallel_status` explains fallback and potential trajectory changes.

Batches reserve at most one objective call per proposal and commit in original
island order; unused reservations return immediately. With fewer islands than
workers, worker count and call budgets affect batch boundaries and random streams.
All workers join before an error escapes; a failed
batch publishes nothing (earlier batches remain committed). User side effects
are not rolled back. This is not a transactional retry/checkpoint guarantee.
"""
function advance_age_search!(state::AgeSearchState;
        max_attempts::Int, max_evaluations::Int,
        mutation_weight=1.0, crossover_weight=1.0,
        proposal=nothing, policy::Union{Nothing,AbstractAgeSearchPolicy}=nothing,
        replay=nothing)
    status = age_parallel_status(state; policy, proposal, replay)
    if !status.enabled || max_attempts <= 0 || max_evaluations <= 0
        return _advance_age_search_serial!(state; max_attempts, max_evaluations,
            mutation_weight, crossover_weight, proposal, policy, replay)
    end
    for endpoint in (0.0, 1.0)
        _age_checked_weight(schedule_value(mutation_weight, endpoint), :mutation_weight)
        _age_checked_weight(schedule_value(crossover_weight, endpoint), :crossover_weight)
    end
    any(!isempty(_age_valid_records(state, i)) for i in eachindex(state.populations)) ||
        return Err(AgeSearchNoValidErr())
    before = _age_snapshot(state.counters)
    island_count = length(state.populations)
    batch_capacity = max(island_count, status.workers)
    # Per-slot workspaces survive all batches in this call. They contain no
    # live search state and are released when advance returns or throws.
    regs = [Regressor(state.reg.target_ty,
        merge(state.reg.components, (; mutation_provider=Mutations.IndexedProvider())),
        state.reg.config, state.reg.rng) for _ in 1:batch_capacity]
    caches = [CanonicalGeneration._GenerationRelationCache() for _ in 1:batch_capacity]
    while state.counters.raw_attempts - before.raw_attempts < max_attempts &&
            state.counters.objective_calls - before.objective_calls < max_evaluations
        width = min(batch_capacity,
            max_attempts - (state.counters.raw_attempts - before.raw_attempts),
            max_evaluations - (state.counters.objective_calls - before.objective_calls))
        islands = [mod1(state.cursor + i - 1, island_count) for i in 1:width]
        ordinal = state.counters.raw_attempts
        repeated_islands = width > island_count
        # Split from copied mother streams before dispatch. Publishing these
        # streams only after successful join prevents both duplicate jobs and
        # partial RNG advancement on failure. One-attempt-per-island batches
        # keep the existing exact serial random stream.
        mother_rngs = repeated_islands ? copy.(state.rngs) : nothing
        job_rngs = if repeated_islands
            @warn "Parallel proposals exceed the island count: children use batch-start populations and independent random substreams. This changes the serial search trajectory; worker count and call budgets may also change it." maxlog=1
            [begin
                seed = rand(mother_rngs[island], UInt64)
                MersenneTwister(UInt32[UInt32(seed & 0xffffffff), UInt32(seed >> 32)])
            end for island in islands]
        else
            nothing
        end
        results = _parallel_map(collect(1:width), (; enabled=true, workers=status.workers)) do index
            island = islands[index]
            scratch = _age_attempt_state(state, island, regs[index], ordinal + index - 1)
            repeated_islands && (scratch.rngs[1] = job_rngs[index])
            # An empty island must still consume its ordinary scheduled attempt.
            if isempty(_age_valid_records(scratch, 1))
                scratch.counters.raw_attempts += 1
                scratch.counters.expected_failures += 1
                _age_add_operator!(scratch.counters, :no_valid_parent)
            else
                CanonicalGeneration._with_generation_cache(caches[index]) do
                    _advance_age_search_serial!(scratch; max_attempts=1,
                        max_evaluations=1, mutation_weight, crossover_weight)
                end
            end
            scratch
        end
        for index in eachindex(islands)
            _age_commit_attempt!(state, islands[index], results[index], ordinal + index - 1;
                commit_rng=!repeated_islands)
        end
        repeated_islands && (state.rngs .= mother_rngs)
    end
    c = state.counters
    return AgeSearchReport(c.raw_attempts - before.raw_attempts,
        c.accepted - before.accepted, c.expected_failures - before.expected_failures,
        c.unchanged - before.unchanged, c.invalids - before.invalids,
        c.mutation_attempts - before.mutation_attempts,
        c.crossover_attempts - before.crossover_attempts,
        c.objective_calls - before.objective_calls,
        c.sample_executions - before.sample_executions, state.cursor)
end
