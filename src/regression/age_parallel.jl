# Only policies whose selection may precede a batch of publications opt in.
# All policy callbacks stay on the coordinator, never on a worker.
_age_parallel_policy(::Nothing) = true
_age_parallel_policy(::AbstractAgeSearchPolicy) = false

"""
    age_parallel_status(state; policy=nothing, proposal=nothing, replay=nothing)

Report effective proposal parallelism, fallback reason and potential trajectory
changes. Default selection and explicitly supported built-in policies can batch
proposals. Policy hooks run synchronously on the coordinator; workers receive
selected parents and own their index, generation cache and random stream.
An optional `parallel.batch_size` limits each proposal wave; without it a wave
covers at least every island or available worker.

Unknown policies, custom scoring/factories, external proposal callbacks, replay
and diagnostics retain serial semantics. Shared task data and opaque user
primitives/extensions must still support concurrent read-only access.
"""
function age_parallel_status(state::AgeSearchState;
        policy=nothing, proposal=nothing, replay=nothing)
    parallel = _normalize_parallel_config(get(state.reg.config, :parallel, (;)))
    island_count = length(state.populations)
    requested_workers = min(Int(get(parallel, :workers, Threads.nthreads(:default))),
        Threads.nthreads(:default))
    batch_capacity = Int(get(parallel, :batch_size,
        max(island_count, requested_workers)))
    workers = min(requested_workers, batch_capacity)
    c = state.reg.components
    reason = if !parallel.enabled
        :disabled
    elseif workers <= 1
        :single_worker
    elseif !_age_parallel_policy(policy) || proposal !== nothing || replay !== nothing ||
            get(c, :age_diagnostics, nothing) !== nothing
        :stateful_hooks
    elseif !_parallel_factories_safe(c)
        :custom_factories
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
        policy !== nothing ? :coordinator_policy :
            batch_capacity > island_count ? :batched_islands : :independent_islands
    end
    enabled = reason in (:independent_islands, :batched_islands, :coordinator_policy)
    return (; enabled, workers=enabled ? workers : 1, reason,
        trajectory_changes=enabled && (policy !== nothing || batch_capacity > island_count))
end

# A policy may prepare a read-only selection view for one coordinator-owned batch.
# Commit and admission continue to use the original policy object.
_age_batch_selection(policy, state, valid_by_island) = policy

# A worker has no population, archive, birth clock, policy or live-state pointer.
# Its mutable provider/cache are reused by only one worker at a time.
struct _AgeWorkerSettings
    target_ty::Ty
    components::NamedTuple
    config::NamedTuple
end

mutable struct _AgeWorkerContext <: _AbstractAgeContext
    reg::_AgeWorkerSettings
    data::SupervisedDataset
    counters::AgeSearchCounters
    next_id::UInt64
    cache::CanonicalGeneration._GenerationRelationCache
end

function _AgeWorkerContext(state::AgeSearchState)
    # All randomness comes from the job. Copying the unused regressor RNG here
    # would allocate a large MersenneTwister buffer for every short advance call.
    reg = _AgeWorkerSettings(state.reg.target_ty,
        _age_private_components(state.reg.components), state.reg.config)
    return _AgeWorkerContext(reg, state.data, AgeSearchCounters(), UInt64(1),
        CanonicalGeneration._GenerationRelationCache())
end

struct _AgeAttemptJob
    island::Int
    ordinal::Int
    rng::AbstractRNG
    left::Union{Nothing,AgeSearchIndividual}
    right::Union{Nothing,AgeSearchIndividual}
    mutation_weight::Float64
    crossover_weight::Float64
    progress::ProposalProgress
end

struct _AgeAttemptResult
    record::Union{Nothing,AgeSearchIndividual}
    counters::AgeSearchCounters
    rng::AbstractRNG
end

function _age_execute_attempt!(context::_AgeWorkerContext, job::_AgeAttemptJob)
    context.counters = AgeSearchCounters()
    context.counters.raw_attempts = 1
    context.next_id = UInt64(1)
    if job.left === nothing
        context.counters.expected_failures = 1
        _age_add_operator!(context.counters, :no_valid_parent)
        return _AgeAttemptResult(nothing, context.counters, job.rng)
    end
    _, _, record = CanonicalGeneration._with_generation_cache(context.cache) do
        _age_compute_attempt!(context, job.left, job.right, job.rng,
            job.mutation_weight, job.crossover_weight, job.progress)
    end
    return _AgeAttemptResult(record, context.counters, job.rng)
end

function _age_commit_attempt!(state, job::_AgeAttemptJob, result::_AgeAttemptResult;
        commit_rng=true, policy=nothing)
    commit_rng && (state.rngs[job.island] = result.rng)
    state.cursor = mod1(job.island + 1, length(state.populations))
    for name in fieldnames(AgeSearchCounters)
        name === :operator_labels && continue
        setfield!(state.counters, name,
            getfield(state.counters, name) + getfield(result.counters, name))
    end
    for (label, count) in result.counters.operator_labels
        state.counters.operator_labels[label] =
            get(state.counters.operator_labels, label, 0) + count
    end
    local_record = result.record
    local_record === nothing && return nothing
    id = _age_next_id!(state) # Invalid scored records consume an ID, but no birth.
    local_record.valid || return nothing
    record = AgeSearchIndividual(local_record.program, id,
        local_record.parent_ids, local_record.origin, local_record.loss,
        local_record.complexity, true, _age_birth!(state))
    if _age_insert!(state, record, job.island; policy)
        _age_push_recent!(state, record)
        state.counters.accepted += 1
    end
    return nothing
end

"""
    advance_age_search!(state; max_attempts, max_evaluations, kwargs...)

Advance bounded complete-program proposals, using private workers when eligible.
Selection and policy feedback are coordinator-owned. Parallel batches select
from batch-start populations and publish in logical order. Shared policies or
repeated island visits therefore change serial trajectories (warned once).

Workers join before an error escapes; a failed computation batch publishes no
records or random streams. Selection callbacks and earlier batches are not
rolled back. Concurrent/reentrant mutation of the same state is rejected.
"""
function advance_age_search!(state::AgeSearchState; kwargs...)
    return _with_age_operation(state) do
        _advance_age_search!(state; kwargs...)
    end
end

function _advance_age_search!(state::AgeSearchState;
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
    parallel = get(state.reg.config, :parallel, (;))
    batch_capacity = Int(get(parallel, :batch_size, max(island_count, status.workers)))
    contexts = [_AgeWorkerContext(state) for _ in 1:status.workers]
    tournament_size = Int(get(state.reg.config, :tournament_size, 2))
    while state.counters.raw_attempts - before.raw_attempts < max_attempts &&
            state.counters.objective_calls - before.objective_calls < max_evaluations
        width = min(batch_capacity,
            max_attempts - (state.counters.raw_attempts - before.raw_attempts),
            max_evaluations - (state.counters.objective_calls - before.objective_calls))
        repeated_islands = width > island_count
        mother_rngs = repeated_islands ? copy.(state.rngs) : nothing
        if repeated_islands || (policy !== nothing && width > 1)
            @warn "Parallel proposals use batch-start parents with delayed policy feedback. This changes the serial search trajectory; worker count and call budgets may also change it." maxlog=1
        end
        jobs = _AgeAttemptJob[]
        valid_by_island = Dict{Int,Vector{AgeSearchIndividual}}()
        for index in 1:width
            island = mod1(state.cursor + index - 1, island_count)
            get!(valid_by_island, island) do
                _age_valid_records(state, island)
            end
        end
        selection_policy = _age_batch_selection(policy, state, valid_by_island)
        for index in 1:width
            island = mod1(state.cursor + index - 1, island_count)
            ordinal = state.counters.raw_attempts + index - 1
            rng = if repeated_islands
                seed = rand(mother_rngs[island], UInt64)
                MersenneTwister(UInt32[UInt32(seed & 0xffffffff), UInt32(seed >> 32)])
            else
                copy(state.rngs[island])
            end
            progress = _proposal_progress(state.reg.components, ordinal)
            mw = _age_checked_weight(schedule_value(mutation_weight, progress), :mutation_weight)
            cw = _age_checked_weight(schedule_value(crossover_weight, progress), :crossover_weight)
            valid = valid_by_island[island]
            left = isempty(valid) ? nothing : _age_selected_parent(
                selection_policy, state, island, valid, rng, tournament_size, :parent)
            right = isempty(valid) ? nothing : _age_selected_parent(
                selection_policy, state, island, valid, rng, tournament_size, :donor)
            push!(jobs, _AgeAttemptJob(island, ordinal, rng, left, right, mw, cw, progress))
        end
        results = _parallel_map(_age_execute_attempt!, jobs,
            (; enabled=true, workers=status.workers); workspaces=contexts)
        for index in eachindex(jobs)
            _age_commit_attempt!(state, jobs[index], results[index];
                commit_rng=!repeated_islands, policy)
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
