@testsnippet ContinuousSessionTestFixture begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ScalarSearch: ScalarSearchConfig, search_multi_scalar

    continuous_session_config() = ScalarSearchConfig(
        population_size=2,
        max_evaluations=24,
        max_complexity=6,
        initial_depth=0,
        tournament_size=1,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(),
        constant_refinement_iterations=1,
        max_constant_refinement_evaluations=8,
        evolution_strategy=:regularized,
    )

    continuous_session_backend_options(; round_observer=nothing,
            hof_donor_policy=:local_best) = (
        island_count=2,
        rounds=3,
        ncycles_per_iteration=1,
        hof_donor_policy,
        round_observer,
    )
end

@testitem "Continuous workflow retains one fixed scalar session across outer and typed actions" setup=[ContinuousWorkflowFixture, ContinuousSessionTestFixture] begin
    state = continuous_fixture()
    scalar_config = continuous_session_config()
    session_options = (
        rounds_per_action=1,
        backend_options=continuous_session_backend_options(),
    )
    scalar = (enabled=true, config=scalar_config, max_return=1,
        material_return=0, allow_zero_input=true, session=session_options)
    typed = (enabled=true, attempts=0, evaluations=1, population_size=1,
        max_return=1)
    observed = NamedTuple[]
    result = run_continuous_search!(state; rng=MersenneTwister(1),
        config=ContinuousSearchConfig(max_actions=7, outer_weight=2.0,
            typed_weight=1.0, scalar_weight=3.0, constant_weight=0.0),
        outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
        typed, scalar, on_action=entry -> push!(observed, entry))

    @test [entry.kind for entry in result.ledger] ==
        [:scalar, :outer, :typed, :scalar, :outer, :scalar, :scalar]
    scalar_entries = filter(entry -> entry.kind === :scalar, result.ledger)
    @test length(scalar_entries) == 4
    session_rows = [entry.report.session for entry in scalar_entries]
    @test all(row -> row !== nothing, session_rows)
    first_session = session_rows[1:3]
    @test [row.event for row in session_rows] ==
        [:started, :resumed, :resumed, :started]
    @test [row.rounds_completed for row in session_rows] == [1, 2, 3, 1]
    @test [row.completed for row in session_rows] == [false, false, true, false]
    @test first_session[1].id == first_session[2].id == first_session[3].id
    @test session_rows[4].id != first_session[1].id
    @test all(row -> row.initialization_count == 2, session_rows)
    @test [entry.source_id for entry in scalar_entries[1:3]] ==
        fill(scalar_entries[1].source_id, 3)
    @test [entry.path for entry in scalar_entries[1:3]] ==
        fill(scalar_entries[1].path, 3)

    counter_fields = (:objective_calls, :structural_evaluations,
	    :constant_refinement_evaluations, :total_evaluations,
	    :baseline_objective_calls)
    for field in counter_fields
        @test sum(getproperty(row.delta, field) for row in first_session) ==
            getproperty(first_session[end].cumulative, field)
    end
    @test first_session[1].initialization_count ==
        first_session[2].initialization_count == first_session[3].initialization_count

    @test result.scalar_sessions.started == 2
    @test result.scalar_sessions.completed == 1
    @test result.scalar_sessions.unfinished.status === :discarded_at_run_end
    @test result.scalar_sessions.unfinished.id == session_rows[4].id
    @test result.termination === :max_actions
    @test length(observed) == length(result.ledger)
end

@testitem "Continuous workflow scalar session rounds match one uninterrupted multi-island advance" setup=[ContinuousWorkflowFixture, ContinuousSessionTestFixture] begin
    scalar_config = continuous_session_config()
    uninterrupted_trace = NamedTuple[]
    session_trace = NamedTuple[]
    uninterrupted_options = continuous_session_backend_options(
        hof_donor_policy=:global_pareto,
        round_observer=(round, snapshot) ->
            push!(uninterrupted_trace, (; round, state=snapshot())))
    session_options = continuous_session_backend_options(
        hof_donor_policy=:global_pareto,
        round_observer=(round, snapshot) ->
            push!(session_trace, (; round, state=snapshot())))

    uninterrupted_backend = function (X, y; kwargs...)
        result = search_multi_scalar(X, y; kwargs..., uninterrupted_options...)
        @test result.diagnostics.hof_donor_policy === :global_pareto
        return result
    end
    baseline_config = ContinuousSearchConfig(max_actions=1, outer_weight=1.0,
        scalar_weight=2.0, constant_weight=0.0)
    retained_config = ContinuousSearchConfig(max_actions=4, outer_weight=1.0,
        scalar_weight=2.0, constant_weight=0.0)
    uninterrupted = run_continuous_search!(continuous_fixture();
        rng=MersenneTwister(42), config=baseline_config,
        outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
        typed=(enabled=false,),
        scalar=(enabled=true, config=scalar_config, max_return=1,
            material_return=0, allow_zero_input=true,
            search_backend=uninterrupted_backend))
    retained = run_continuous_search!(continuous_fixture();
        rng=MersenneTwister(42), config=retained_config,
        outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
        typed=(enabled=false,),
        scalar=(enabled=true, config=scalar_config, max_return=1,
            material_return=0, allow_zero_input=true,
            session=(rounds_per_action=1, backend_options=session_options)))

    uninterrupted_entry = only(uninterrupted.ledger)
    retained_scalars = filter(entry -> entry.kind === :scalar, retained.ledger)
    @test [entry.kind for entry in retained.ledger] ==
        [:scalar, :outer, :scalar, :scalar]
    @test length(retained_scalars) == 3
    retained_entry = last(retained_scalars)
    @test uninterrupted_entry.kind === retained_entry.kind === :scalar
    @test all(entry -> entry.source_id == uninterrupted_entry.source_id,
        retained_scalars)
    @test all(entry -> entry.path == uninterrupted_entry.path, retained_scalars)
    @test length(uninterrupted_trace) == length(session_trace) == 3
    @test repr(uninterrupted_trace) == repr(session_trace)
    @test [entry.report.session.event for entry in retained_scalars] ==
        [:started, :resumed, :resumed]
    @test length(unique(entry.report.session.id for entry in retained_scalars)) == 1
    @test [entry.report.session.rounds_completed for entry in retained_scalars] ==
        [1, 2, 3]
    @test retained_entry.report.session.completed
    @test retained.scalar_sessions.completed == 1
    @test retained.scalar_sessions.unfinished === nothing
end

@testitem "Continuous workflow restores retained source family credit before snapshot" setup=[ContinuousWorkflowFixture, ContinuousSessionTestFixture] begin
    state = continuous_fixture()
    policy = StructurePolicy(continuous_key; capacity=2, representatives=4,
        development_evaluations=100)
    CW._seed_structure_policy!(policy, state)
    source = state.populations[1][1]
    source_key = CW._structure_key(policy, source)
    @test !isempty(policy.families[source_key].representatives)

    delete!(policy.families, source_key)
    CW._credit_development!(policy, state, source, 3, 0.1, false)

    family = policy.families[source_key]
    @test family.development_actions == 1
    @test family.development_evaluations == 3
    @test length(family.representatives) == 1
    @test only(family.representatives).id == source.id
    snapshot = structure_snapshot(policy)
    snapshot_family = only(filter(item -> isequal(item.key, source_key),
        snapshot.families))
    @test snapshot_family.representative_ids == [source.id]
    @test snapshot_family.best_loss == source.loss
end

@testitem "Continuous workflow rejects invalid scalar session options" setup=[ContinuousWorkflowFixture, ContinuousSessionTestFixture] begin
    scalar_config = continuous_session_config()
    backend_options = continuous_session_backend_options()
    common = (; rng=MersenneTwister(42),
        config=ContinuousSearchConfig(max_actions=1, scalar_weight=100.0,
            constant_weight=0.0),
        outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
        typed=(enabled=false,))
    function run_with_session(session; search_backend=nothing)
        scalar_options = (; enabled=true, config=scalar_config, max_return=1,
            material_return=0, allow_zero_input=true, session)
        scalar = search_backend === nothing ? scalar_options :
            merge(scalar_options, (; search_backend))
        run_continuous_search!(continuous_fixture(); common..., scalar)
    end

    @test_throws ArgumentError run_with_session((rounds_per_action=1,
        backend_options, unsupported_option=true))
    @test_throws ArgumentError run_with_session((rounds_per_action=0,
        backend_options))
    forbidden_backend = (X, y; kwargs...) -> error(
        "search_backend must not run with a retained session")
    @test_throws ArgumentError run_with_session((rounds_per_action=1,
        backend_options); search_backend=forbidden_backend)
end
