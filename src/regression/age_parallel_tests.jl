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

@testitem "Age parallel commit accounts invalid IDs and unproductive random streams" setup=[AgeSearchBaseImports] begin
    R = LambdaRegression.Regression
    reg, data = age_test_fixture()
    state = initialize_age_search(reg, data; population_count=2, seed=5)
    ordinal = 17
    scratch = R._age_attempt_state(state, 1, reg, ordinal)
    rand(scratch.rngs[1], UInt64, 3)
    expected_rng = copy(scratch.rngs[1])
    scratch.counters.raw_attempts += 1
    scratch.counters.invalids = 1
    scratch.counters.advance_invalids = 1
    scratch.next_id += UInt64(1)
    old_id, old_birth = state.next_id, state.birth_clock
    R._age_commit_attempt!(state, 1, scratch, ordinal)
    @test state.next_id == old_id + 1
    @test state.birth_clock == old_birth
    @test state.counters.raw_attempts == 1
    @test state.counters.invalids == 1
    @test rand(copy(state.rngs[1]), UInt64) == rand(expected_rng, UInt64)
    @test state.cursor == 2
end
