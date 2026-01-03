@testitem "Age parallel proposals preserve full serial state and continuation" begin
    using Test, Random, Logging
    using LambdaRegression
    using LambdaRegression.Regression
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    R = LambdaRegression.Regression
    globals = Dict{Symbol,Any}(:FADD => FADD)
    tree = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:k))))
    c = default_components(; global_consts=globals, global_consts_ty=get_type_ctx(globals),
        input_slots=[:x], input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_population=(reg, data; N, seed, global_type_ctx) ->
            [Program(tree; local_consts=Dict{Symbol,Any}(:k => Float64(i)),
                local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy)) for i in 1:N])
    data = SupervisedDataset([0.25, 1.5, 2.0], [0.0, 1.0, 4.0])
    make_state(enabled) = initialize_age_search(Regressor(TyFun(FloatTy, FloatTy), c,
        default_config(population_size=4, parallel=(; enabled, workers=8)), MersenneTwister(2000)),
        data; population_count=8, seed=31)
    serial, parallel = make_state(false), make_state(true)
    recordkey(r) = (r.program.tree, r.program.local_consts, r.program.local_consts_ty,
        r.id, r.parent_ids, r.origin, r.loss, r.complexity, r.valid, r.birth)
    function check_states(a, b)
        @test recordkey.(a.populations[1]) == recordkey.(b.populations[1])
        @test [recordkey.(p) for p in a.populations] == [recordkey.(p) for p in b.populations]
        @test (a.cursor, a.birth_clock, a.next_id) == (b.cursor, b.birth_clock, b.next_id)
        @test all(getfield(a.counters, f) == getfield(b.counters, f) for f in fieldnames(AgeSearchCounters))
        @test Dict(k => recordkey(v) for (k,v) in a.archive) == Dict(k => recordkey(v) for (k,v) in b.archive)
        @test recordkey(a.best) == recordkey(b.best)
        @test recordkey.(a.recent_valid_offspring) == recordkey.(b.recent_valid_offspring)
        @test [rand(copy(r), UInt64, 8) for r in a.rngs] == [rand(copy(r), UInt64, 8) for r in b.rngs]
    end
    check_states(serial, parallel)
    @test age_parallel_status(parallel).enabled == (Threads.nthreads(:default) > 1)
    # Incomplete island sweeps, a tight evaluation cap, then continuation.
    for (attempts, evaluations) in ((19, 3), (24, 24), (7, 2))
        a = advance_age_search!(serial; max_attempts=attempts, max_evaluations=evaluations)
        b = advance_age_search!(parallel; max_attempts=attempts, max_evaluations=evaluations)
        @test all(getfield(a, f) == getfield(b, f) for f in fieldnames(AgeSearchReport))
        @test b.objective_calls <= evaluations
        check_states(serial, parallel)
    end
    @test age_parallel_status(parallel; proposal=identity).reason in (:stateful_hooks, :single_worker)
    @test age_parallel_status(parallel; replay=identity).reason in (:stateful_hooks, :single_worker)
    original_reg = parallel.reg
    for components in (
            merge(c, (; make_eval_context=(; kwargs...) -> error("must stay serial"))),
            Base.structdiff(c, (; mutation_provider=c.mutation_provider)))
        parallel.reg = Regressor(original_reg.target_ty, components,
            original_reg.config, original_reg.rng)
        @test !age_parallel_status(parallel).enabled
    end
    parallel.reg = original_reg
    @test_throws ArgumentError default_config(parallel=(enabled=true, workers=0))
    @test_throws ArgumentError default_config(parallel=(enabled=1,))
    @test_throws ArgumentError default_config(parallel=(enabled=true, batch_size=0))
    @test_throws ArgumentError default_config(parallel=(enabled=true, batch_size=true))
    @test default_config(parallel=(enabled=true, batch_size=3)).parallel.batch_size == 3

    if Threads.nthreads(:default) >= 8
        # A single island can occupy all workers. Its children must use the
        # batch-start parents, not other still-unpublished children.
        make_small() = initialize_age_search(Regressor(TyFun(FloatTy, FloatTy), c,
            default_config(population_size=4, parallel=(enabled=true, workers=8)),
            MersenneTwister(2000)), data; population_count=1, seed=31)
        a, b = make_small(), make_small()
        @test age_parallel_status(a).workers == 8
        @test age_parallel_status(a).trajectory_changes
        parent_ids = Set(r.id for r in a.populations[1])
        expected_rng = copy(a.rngs[1])
        for _ in 1:8
            rand(expected_rng, UInt64)
        end
        logger = Test.TestLogger()
        with_logger(logger) do
            for state in (a, b)
                report = advance_age_search!(state; max_attempts=8, max_evaluations=8)
                @test report.attempts == 8
                @test report.objective_calls <= 8
                @test all(all(id in parent_ids for id in r.parent_ids)
                    for r in state.recent_valid_offspring)
            end
            check_states(a, b)
            @test rand(copy(a.rngs[1]), UInt64, 8) == rand(expected_rng, UInt64, 8)
            for state in (a, b)
                report = advance_age_search!(state; max_attempts=11, max_evaluations=3)
                @test report.attempts <= 11
                @test report.objective_calls <= 3
                @test length(unique(r.id for r in state.populations[1])) == 4
            end
            check_states(a, b)
        end
        @test count(log -> log.level == Logging.Warn &&
            occursin("changes the serial search trajectory", string(log.message)), logger.logs) == 1
    end
end

@testitem "Age parallel batch size bounds waves and reports trajectory changes" begin
    using Test, Random, Logging
    using LambdaRegression
    using LambdaRegression.Regression
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: TyFun, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    import Base.Threads
    R = LambdaRegression.Regression
    struct FailingBatchSelectionPolicy <: R.AbstractAgeSearchPolicy end
    R._age_parallel_policy(::FailingBatchSelectionPolicy) = true
    R._age_batch_selection(::FailingBatchSelectionPolicy, state, valid_by_island) =
        error("expected batch selection failure")
    globals = Dict{Symbol,Any}(:FADD => FADD)
    tree = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:k))))
    components = default_components(; global_consts=globals,
        global_consts_ty=get_type_ctx(globals), input_slots=[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_population=(reg, data; N, seed, global_type_ctx) ->
            [Program(tree; local_consts=Dict{Symbol,Any}(:k => Float64(i)),
                local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy)) for i in 1:N])
    data = SupervisedDataset([0.25, 1.5, 2.0], [0.0, 1.0, 4.0])
    function make_state(populations, workers, batch_size; enabled=true)
        reg = Regressor(TyFun(FloatTy, FloatTy), components,
            default_config(population_size=4,
                parallel=(; enabled, workers, batch_size)), MersenneTwister(2000))
        return initialize_age_search(reg, data; population_count=populations, seed=31)
    end
    recordkey(r) = (r.program.tree, r.program.local_consts, r.program.local_consts_ty,
        r.id, r.parent_ids, r.origin, r.loss, r.complexity, r.valid, r.birth)
    function check_states(a, b)
        @test [recordkey.(p) for p in a.populations] == [recordkey.(p) for p in b.populations]
        @test (a.cursor, a.birth_clock, a.next_id) == (b.cursor, b.birth_clock, b.next_id)
        @test all(getfield(a.counters, f) == getfield(b.counters, f)
            for f in fieldnames(AgeSearchCounters))
        @test Dict(k => recordkey(v) for (k, v) in a.archive) ==
            Dict(k => recordkey(v) for (k, v) in b.archive)
        @test recordkey.(a.recent_valid_offspring) == recordkey.(b.recent_valid_offspring)
        @test [rand(copy(r), UInt64, 8) for r in a.rngs] ==
            [rand(copy(r), UInt64, 8) for r in b.rngs]
    end

    # Four-slot waves do not revisit any of eight islands and retain the serial
    # state even when the configured worker count is larger than the batch.
    serial, bounded = make_state(8, 8, 4; enabled=false), make_state(8, 8, 4)
    status = age_parallel_status(bounded)
    @test status.workers == min(4, Threads.nthreads(:default))
    @test !status.trajectory_changes
    @test R._age_initial_parallel_status(bounded, 32).workers ==
        min(8, Threads.nthreads(:default))
    serial_report = advance_age_search!(serial; max_attempts=5, max_evaluations=5)
    bounded_report = advance_age_search!(bounded; max_attempts=5, max_evaluations=5)
    @test serial_report == bounded_report
    check_states(serial, bounded)

    # A six-proposal batch over two islands is intentionally wider than the
    # available two-worker team; repeated-island selection uses batch-start data.
    if Threads.nthreads(:default) >= 2
        a, b = make_state(2, 2, 6), make_state(2, 2, 6)
        @test age_parallel_status(a).workers == 2
        @test age_parallel_status(a).trajectory_changes
        logger = Test.TestLogger()
        reports = with_logger(logger) do
            first = advance_age_search!(a; max_attempts=6, max_evaluations=6)
            second = advance_age_search!(b; max_attempts=6, max_evaluations=6)
            (first, second)
        end
        @test reports[1] == reports[2]
        @test reports[1].attempts == 6
        check_states(a, b)

        failed, fresh = make_state(8, 8, 4), make_state(8, 8, 4)
        @test_throws ErrorException advance_age_search!(failed;
            max_attempts=4, max_evaluations=4, policy=FailingBatchSelectionPolicy())
        check_states(failed, fresh)
        recovered = advance_age_search!(failed; max_attempts=1, max_evaluations=1)
        @test recovered.attempts == 1
    end
end

@testitem "Age parallel commit accounts invalid IDs and unproductive random streams" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Schedules: ProposalProgress
    R = LambdaRegression.Regression
    reg, data = age_test_fixture()
    state = initialize_age_search(reg, data; population_count=2, seed=5)
    ordinal = 1
    job_rng = MersenneTwister(42)
    rand(job_rng, UInt64, 3)
    expected_rng = copy(job_rng)
    job = R._AgeAttemptJob(1, ordinal, job_rng, nothing, nothing,
        1.0, 1.0, ProposalProgress(0, 1))
    counters = R.AgeSearchCounters()
    counters.raw_attempts = 1
    counters.invalids = 1
    counters.advance_invalids = 1
    previous = first(state.populations[1])
    invalid_record = R.AgeSearchIndividual(previous.program, UInt64(0), (),
        :mutation, Inf, previous.complexity, false, UInt64(0))
    result = R._AgeAttemptResult(invalid_record, counters, job_rng)
    old_id, old_birth = state.next_id, state.birth_clock
    R._age_commit_attempt!(state, job, result)
    @test state.next_id == old_id + 1
    @test state.birth_clock == old_birth
    @test state.counters.raw_attempts == 1
    @test state.counters.invalids == 1
    @test rand(copy(state.rngs[1]), UInt64) == rand(expected_rng, UInt64)
    @test state.cursor == 2
end

@testitem "Age search rejects concurrent and reentrant state mutation and releases guard" setup=[AgeSearchBaseImports] begin
    import Base.Threads
    R = LambdaRegression.Regression
    reg, data = age_test_fixture()
    state = initialize_age_search(reg, data; population_count=1, seed=71)

    R._with_age_operation(state) do
        @test_throws ArgumentError advance_age_search!(state;
            max_attempts=0, max_evaluations=0)
        @test_throws ArgumentError inject_age_migrants!(state, Program[])
        other = Threads.@spawn try
            advance_age_search!(state; max_attempts=0, max_evaluations=0)
        catch issue
            issue
        end
        @test fetch(other) isa ArgumentError
    end

    @test_throws ErrorException R._with_age_operation(state) do
        error("expected operation failure")
    end
    report = advance_age_search!(state; max_attempts=0, max_evaluations=0)
    @test report isa R.AgeSearchReport

    @test_throws ErrorException advance_age_search!(state;
        max_attempts=1, max_evaluations=1,
        proposal=(state, island, left, right; rng) -> error("expected proposal failure"),
    )
    report = advance_age_search!(state; max_attempts=0, max_evaluations=0)
    @test report isa R.AgeSearchReport
end

@testitem "Age search falls back for custom type and constant factories" setup=[AgeSearchBaseImports] begin
    import Base.Threads
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.TypeSystem: make_tyvar_factory
    R = LambdaRegression.Regression
    reg, data = age_test_fixture()
    factory_variants = (
        merge(reg.components, (; tyfactory_ctor=() -> make_tyvar_factory())),
        merge(reg.components, (; const_symfactory_ctor=() -> ConstSymbolFactory())),
    )

    if Threads.nthreads(:default) > 1
        for components in factory_variants
            parallel_reg = Regressor(reg.target_ty, components,
                default_config(parallel=(enabled=true, workers=2)), reg.rng)
            state = initialize_age_search(parallel_reg, data;
                population_count=1, population_size=2, seed=73)
            @test R.age_parallel_status(state).reason == :custom_factories
            @test R._age_initial_parallel_status(state, 2).reason == :custom_factories
        end
    else
        @test true
    end
end
