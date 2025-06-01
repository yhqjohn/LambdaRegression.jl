"""Return bounded unique programs to seed the persistent joint session."""
function _joint_initial_programs(state::AgeSearchState, rng::AbstractRNG, limit::Int)
    records = _continuous_material_pool(state, nothing)
    shuffle!(rng, records)
    programs = Program[]
    for record in records
        any(program -> _same_program(program, record.program), programs) && continue
        push!(programs, record.program)
        length(programs) >= limit && break
    end
    return programs
end

function _joint_age_materials(state::AgeSearchState, policy)
    records = AgeSearchIndividual[]
    seen = Set{UInt64}()
    for record in Iterators.flatten((values(state.archive),
            Iterators.flatten(state.populations)))
        record.valid && record.id ∉ seen || continue
        push!(records, record)
        push!(seen, record.id)
    end
    return policy === nothing ? records : _structure_candidates(policy, records)
end

"""Select bounded, randomly sampled non-champions when any are available."""
function _joint_nonchampions(records, champion, rng::AbstractRNG, count::Int)
    count == 0 && return eltype(records)[]
    alternatives = [record for record in records if champion === nothing || record.id != champion.id]
    shuffle!(rng, alternatives)
    selected = alternatives[1:min(count, length(alternatives))]
    if isempty(selected) && champion !== nothing
        push!(selected, champion)
    end
    return selected
end

function _joint_record(program::Program)
    return AgeSearchIndividual(program, UInt64(0), (), :joint_global,
        Inf, 0, true, UInt64(0))
end

"""Use the age-search mutation and typed crossover operators with finite retries."""
function _joint_global_proposal(state::AgeSearchState; retry_limit::Int=4)
    retry_limit >= 0 || throw(ArgumentError("global_proposal_attempts must be nonnegative"))
    components = state.reg.components
    has_crossover = get(components, :crossover_op, nothing) !== nothing
    # This closure owns a separate cumulative global-proposal clock. Local
    # scalar attempts never advance it; retry attempts share one progress value.
    global_attempts = Ref(0)
    return function (parent::Program, donor::Program, rng::AbstractRNG)
        progress = Regression._proposal_progress(components, global_attempts[])
        global_attempts[] += 1
        left, right = _joint_record(parent), _joint_record(donor)
        for _ in 1:retry_limit
            result = if has_crossover && rand(rng, Bool)
                Regression._age_crossover(state, left, right, rng)
            else
                Regression._age_mutation(state, left, rng; progress)
            end
            result isa Err && continue
            result isa Program || throw(ArgumentError(
                "age-search mutation and crossover must return Program or Err"))
            (_same_program(result, parent) || _same_program(result, donor)) && continue
            return result
        end
        return nothing
    end
end

function _joint_result_counts(result)
    return (; evaluations=result.evaluations,
        constant_refinement_evaluations=result.constant_refinement_evaluations,
        invalid_evaluations=result.invalid_evaluations,
        constant_refinement_invalid_evaluations=result.constant_refinement_invalid_evaluations,
        attempts=result.attempts, migrations=result.migrations, completed=result.completed,
        counters=result.counters)
end

function _joint_result_delta(before, after)
    return (; evaluations=after.evaluations - before.evaluations,
        constant_refinement_evaluations=after.constant_refinement_evaluations -
            before.constant_refinement_evaluations,
        invalid_evaluations=after.invalid_evaluations - before.invalid_evaluations,
        constant_refinement_invalid_evaluations=
            after.constant_refinement_invalid_evaluations -
                before.constant_refinement_invalid_evaluations,
        attempts=after.attempts - before.attempts,
        migrations=after.migrations - before.migrations)
end

"""
    run_joint_search!(age_state; rng, config=ScalarSearchConfig(...),
        island_count=2, windows=4, attempts_per_window=2048,
        evaluations_per_window=128, outer=(attempts=128, evaluations=32),
        exchange_count=2, policy=nothing, named_input_policy=:declared,
        max_nodes=2048, max_depth=256, group_exchange_probability=0.25,
        replacement=:age, rtr_window=8, global_probability=0.02,
        on_observe=nothing, kwargs...)

Coordinate one persistent complete-program joint session with bounded outer
age-search work. Initial and exchanged programs always enter the receiving
engine through its scorer; no score or complexity value crosses the bridge.
Each finite window advances the joint population, offers randomly sampled
joint archive/live material to the outer age search, advances outer search with
its existing policy, then offers bounded age archive/live material back to the
same joint session. Non-champions are sampled whenever available. The joint
population size is `island_count * config.population_size`, independent of the
number of seed programs or discovered scalar regions.

The run keeps layer-specific score, sample, invalid, attempt, population, and
elapsed-time accounting. `on_observe(entry, age_state, session, selected)` is
synchronous and receives borrowed state and selected records; do not mutate or
retain them. All windows and bridge counts are finite. `kwargs` are forwarded
to `initialize_joint_search` after the workflow supplies the task, scorer, and
existing age-search mutation/crossover proposal.
`migration_interval=4` injects small same-task archive/live samples directly
between joint islands after each fourth joint action window; zero disables it.
Migration uses no new objective calls and skips exhausted destinations.
"""
function run_joint_search!(state::AgeSearchState; rng::AbstractRNG,
        config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
        island_count::Int=2, windows::Int=4,
        attempts_per_window::Int=2048, evaluations_per_window::Int=128,
        outer=(attempts=128, evaluations=32), exchange_count::Int=2,
        migration_interval::Int=4,
        policy=nothing, global_proposal_attempts::Int=4,
        named_input_policy::Symbol=:declared, max_nodes::Int=2048,
        max_depth::Int=256,
        group_exchange_probability::Real=0.25, replacement::Symbol=:age,
        rtr_window::Int=8, global_probability::Real=0.02,
        ncycles_per_iteration::Int=64, frequency_window::Int=100000,
        loss_normalization::Real=1.0, event_capacity::Int=256,
        on_observe=nothing, kwargs...)
    windows >= 0 || throw(ArgumentError("windows must be nonnegative"))
    island_count > 0 || throw(ArgumentError("island_count must be positive"))
    attempts_per_window >= 0 && evaluations_per_window >= 0 ||
        throw(ArgumentError("joint window limits must be nonnegative"))
    exchange_count >= 0 || throw(ArgumentError("exchange_count must be nonnegative"))
    migration_interval >= 0 || throw(ArgumentError("migration_interval must be nonnegative"))
    global_proposal_attempts >= 0 ||
        throw(ArgumentError("global_proposal_attempts must be nonnegative"))
    outer isa NamedTuple && hasproperty(outer, :attempts) &&
        hasproperty(outer, :evaluations) ||
        throw(ArgumentError("outer must contain attempts and evaluations"))
    outer.attempts isa Integer && outer.evaluations isa Integer &&
        outer.attempts >= 0 && outer.evaluations >= 0 ||
        throw(ArgumentError("outer limits must be nonnegative integers"))
    config.evolution_strategy === :regularized || throw(ArgumentError(
        "joint search requires ScalarSearchConfig(evolution_strategy=:regularized)"))
    if policy isa StructurePolicy && policy.describe isa _StructureClassifier
        policy.describe.boundary_policy.named_input_policy === named_input_policy ||
            throw(ArgumentError("descriptor and joint search named-input policies must match"))
    end
    policy isa StructurePolicy && _seed_structure_policy!(policy, state)

    started = time_ns()
    population_size = island_count * config.population_size
    initial_programs = _joint_initial_programs(state, rng, population_size)
    isempty(initial_programs) && return Err(ArgumentError(
        "joint search requires at least one valid archive/live/recent program"))

    joint_counts = ObjectiveCounts()
    age_objective_start = state.counters.objective_calls
    age_sample_start = state.counters.sample_executions
    age_invalid_start = state.counters.invalids
    age_attempt_start = state.counters.raw_attempts
    age_init_objective_start = state.counters.init_objective_calls
    age_init_sample_start = state.counters.init_sample_executions
    age_init_invalid_start = state.counters.init_invalids
    age_migration_objective_start = state.counters.migration_objective_calls
    age_migration_sample_start = state.counters.migration_sample_executions
    function joint_score(program)
        invalid_before = joint_counts.invalid
        loss = score_candidate!(joint_counts, program, state.reg, state.data;
            copy_tree=false)
        joint_counts.invalid == invalid_before || return Err(:invalid_joint_candidate)
        return loss
    end
    type_context = Dict{Symbol,TyLike}(state.reg.components.global_consts_ty)
    input_types = Dict{Symbol,TyLike}(state.reg.components.input_types)
    initialization_started = time_ns()
    session = Regression.initialize_joint_search(
        initial_programs, state.reg.target_ty, type_context, joint_score;
        config, rng, island_count, input_types,
        global_proposal=_joint_global_proposal(state;
            retry_limit=global_proposal_attempts),
        named_input_policy, max_nodes, max_depth, group_exchange_probability,
        replacement, rtr_window, global_probability, ncycles_per_iteration,
        frequency_window, loss_normalization, event_capacity, kwargs...)
    initialization_seconds = (time_ns() - initialization_started) / 1e9
    initialization_calls = joint_counts.calls
    initialization_samples = joint_counts.samples
    initialization_invalid = joint_counts.invalid
    ledger = NamedTuple[]
    termination = :windows

    for window in 1:windows
        window_started = time_ns()
        joint_before = _joint_result_counts(Regression.joint_search_result(session))
        joint_calls_before, joint_samples_before, joint_invalid_before =
            joint_counts.calls, joint_counts.samples, joint_counts.invalid
        joint_started = time_ns()
        Regression.advance_joint_search!(session;
            max_evaluations=evaluations_per_window,
            max_attempts=attempts_per_window,
            migrate=migration_interval > 0 && window % migration_interval == 0)
        joint_seconds = (time_ns() - joint_started) / 1e9
        joint_advance_calls = joint_counts.calls - joint_calls_before
        joint_after = Regression.joint_search_result(session)
        joint_progress = _joint_result_delta(joint_before,
            _joint_result_counts(joint_after))

        joint_sources = _joint_nonchampions(joint_after.candidates,
            joint_after.best, rng, exchange_count)
        joint_to_age_started = time_ns()
        joint_to_age = inject_age_migrants!(state,
            [AgeSearchMigration(candidate.program, ()) for candidate in joint_sources];
            population=mod1(window, length(state.populations)),
            max_evaluations=exchange_count, max_migrants=exchange_count, policy)
        joint_to_age_seconds = (time_ns() - joint_to_age_started) / 1e9

        age_before = state.counters.objective_calls
        age_started = time_ns()
        age_report = advance_age_search!(state;
            max_attempts=Int(outer.attempts),
            max_evaluations=Int(outer.evaluations), policy)
        age_advance_calls = state.counters.objective_calls - age_before
        age_seconds = (time_ns() - age_started) / 1e9

        outer_sources = _joint_nonchampions(_joint_age_materials(state, policy),
            state.best, rng, exchange_count)
        joint_import_started = time_ns()
        joint_import_calls_before = joint_counts.calls
        joint_to_joint = Regression.inject_joint_programs!(session,
            [record.program for record in outer_sources];
            island=mod1(window, island_count), max_count=exchange_count)
        age_to_joint_seconds = (time_ns() - joint_import_started) / 1e9
        age_to_joint_score_calls = joint_counts.calls - joint_import_calls_before
        joint_window_end = Regression.joint_search_result(session)

        entry = (; window,
            joint_progress,
            joint_objective_calls=joint_counts.calls - joint_calls_before,
            joint_advance_score_calls=joint_advance_calls,
            joint_sample_executions=joint_counts.samples - joint_samples_before,
            joint_invalid_scores=joint_counts.invalid - joint_invalid_before,
            age_to_joint_score_calls,
            joint_to_age=(; joint_to_age...,
                joint_export_ids=[candidate.id for candidate in joint_sources]),
            outer=age_report,
            outer_objective_calls=age_advance_calls,
            age_to_joint=(; joint_to_joint...,
                age_import_source_ids=[record.id for record in outer_sources]),
            population=(; joint=length.(joint_window_end.population),
                age=length.(state.populations)),
            best=(; joint=joint_window_end.best,
                age=state.best),
            elapsed=(; joint=joint_seconds, joint_to_age=joint_to_age_seconds,
                age=age_seconds, age_to_joint=age_to_joint_seconds),
            elapsed_seconds=(time_ns() - window_started) / 1e9)
        push!(ledger, entry)
        on_observe === nothing || on_observe(entry, state, session,
            (; joint=joint_sources, age=outer_sources))
        if joint_window_end.completed
            termination = :joint_complete
            break
        end
    end

    joint_final = Regression.joint_search_result(session)
    joint_final.completed && (termination = :joint_complete)
    age_objective_calls = state.counters.objective_calls - age_objective_start
    age_sample_executions = state.counters.sample_executions - age_sample_start
    age_invalids = state.counters.invalids - age_invalid_start
    age_attempts = state.counters.raw_attempts - age_attempt_start
    age_migration_objective_calls =
        state.counters.migration_objective_calls - age_migration_objective_start
    age_migration_sample_executions =
        state.counters.migration_sample_executions - age_migration_sample_start
    joint_total_calls = joint_counts.calls
    total_objective_calls = joint_total_calls + age_objective_calls
    total_sample_executions = joint_counts.samples + age_sample_executions
    total_invalids = joint_counts.invalid + age_invalids
    return (; state, session, best=(; joint=joint_final.best, age=state.best),
        joint=joint_final, ledger, termination,
        preexisting_age=(;
            objective_calls=age_objective_start,
            sample_executions=age_sample_start,
            invalid=age_invalid_start,
            raw_attempts=age_attempt_start,
            init_objective_calls=age_init_objective_start,
            init_sample_executions=age_init_sample_start,
            init_invalid=age_init_invalid_start,
            migration_objective_calls=age_migration_objective_start,
            migration_sample_executions=age_migration_sample_start),
        initialization=(; objective_calls=initialization_calls,
            sample_executions=initialization_samples, invalid=initialization_invalid,
            seed_programs=length(initial_programs),
            population_size=island_count * config.population_size,
            elapsed_seconds=initialization_seconds),
        counts=(;
            joint=(; objective_calls=joint_total_calls,
                sample_executions=joint_counts.samples, invalid=joint_counts.invalid,
                evaluations=joint_final.evaluations,
                constant_refinement_evaluations=joint_final.constant_refinement_evaluations,
                attempts=joint_final.attempts, migrations=joint_final.migrations),
            age=(; objective_calls=age_objective_calls,
                sample_executions=age_sample_executions, invalid=age_invalids,
                advance_attempts=age_attempts,
                migration_objective_calls=age_migration_objective_calls,
                migration_sample_executions=age_migration_sample_executions),
            total=(; objective_calls=total_objective_calls,
                sample_executions=total_sample_executions, invalid=total_invalids),
            caps=(; joint_initialization_evaluations=population_size,
                joint_attempts=windows * attempts_per_window,
                joint_advance_evaluations=windows * evaluations_per_window,
                joint_to_age_imports=windows * exchange_count,
                age_attempts=windows * Int(outer.attempts),
                age_advance_evaluations=windows * Int(outer.evaluations),
                age_to_joint_imports=windows * exchange_count)),
        population=(; joint=length.(joint_final.population),
            age=length.(state.populations)),
        structure=structure_snapshot(policy),
        elapsed=(; initialization=initialization_seconds,
            joint=sum((entry.elapsed.joint for entry in ledger); init=0.0),
            joint_to_age=sum((entry.elapsed.joint_to_age for entry in ledger); init=0.0),
            age=sum((entry.elapsed.age for entry in ledger); init=0.0),
            age_to_joint=sum((entry.elapsed.age_to_joint for entry in ledger); init=0.0)),
        elapsed_seconds=(time_ns() - started) / 1e9)
end

@testitem "Joint workflow migrates at bounded explicit windows without rescoring" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.Workflows: run_joint_search!
    config = ScalarSearchConfig(population_size=2, max_evaluations=16,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(),
        constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    for interval in (0, 4)
        state = continuous_fixture(size=2)
        result = run_joint_search!(state; rng=MersenneTwister(8000), config,
            island_count=2, windows=4, attempts_per_window=0,
            evaluations_per_window=0, outer=(attempts=0, evaluations=0),
            exchange_count=0, migration_interval=interval)
        @test result.counts.joint.objective_calls == 4
        @test result.counts.total.objective_calls == 4
        @test sum(result.population.joint) == 4
        @test all(entry.joint_progress.migrations == 0 for entry in result.ledger[1:3])
        @test result.joint.migrations == (interval == 0 ? 0 : 4)
        @test result.ledger[4].joint_progress.migrations == result.joint.migrations
    end
end

@testitem "Joint workflow keeps a fixed population and accounts rescoring" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.Workflows: run_joint_search!

    state = continuous_fixture(size=2)
    observed = NamedTuple[]
    config = ScalarSearchConfig(population_size=2, max_evaluations=8,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(),
        constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    result = run_joint_search!(state; rng=MersenneTwister(8000), config,
        island_count=2, windows=1, attempts_per_window=2,
        evaluations_per_window=0, outer=(attempts=1, evaluations=1),
        exchange_count=1, global_probability=0.0,
        on_observe=(entry, observed_state, session, selected) -> begin
            @test observed_state === state
            @test length(selected.joint) <= 1
            push!(observed, entry)
        end)

    @test result.initialization.seed_programs == 2
    @test result.initialization.population_size == 4
    @test sum(result.population.joint) == 4
    @test result.initialization.objective_calls == 4
    @test result.counts.joint.objective_calls >=
        result.counts.joint.sample_executions ÷ length(state.data.X)
    @test result.counts.total.objective_calls ==
        result.counts.joint.objective_calls + result.counts.age.objective_calls
    @test result.ledger[1].outer.attempts == 1
    @test all(id -> id isa Int, result.ledger[1].joint_to_age.joint_export_ids)
    @test all(id -> id isa UInt64, result.ledger[1].age_to_joint.age_import_source_ids)
    @test length(result.ledger) == length(observed) == 1
end
