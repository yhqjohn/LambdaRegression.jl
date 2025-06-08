function _age_initial_parallel_status(state::AgeSearchState, item_count::Int;
        initial_transform=nothing)
    parallel = _normalize_parallel_config(get(state.reg.config, :parallel, (;)))
    workers = min(Int(get(parallel, :workers, Threads.nthreads(:default))),
        Threads.nthreads(:default), item_count)
    c = state.reg.components
    reason = if !parallel.enabled
        :disabled
    elseif workers <= 1
        :single_worker_or_individual
    elseif initial_transform !== nothing
        :stateful_initial_transform
    elseif get(c, :age_diagnostics, nothing) !== nothing
        :diagnostics
    elseif haskey(c, :evaluate) || haskey(c, :prepare_evaluate) ||
            get(c, :loss, mse) !== mse || get(c, :complexity, _node_count) !== _node_count ||
            get(c, :make_eval_context, nothing) !== get(c, :_parallel_default_eval_context, missing) ||
            get(c, :candidate_admissibility, nothing) !== _procedural_candidate_admissibility ||
            get(c, :candidate_failure, _known_candidate_exception) !== _known_candidate_exception ||
            get(c, :loss_failure, _no_candidate_exception) !== _no_candidate_exception
        :custom_scoring
    else
        :independent_initial_individuals
    end
    return (; enabled=reason === :independent_initial_individuals,
        workers=reason === :independent_initial_individuals ? workers : 1, reason)
end

function _age_preflight_initial_population_parallel(
    population,
    reg::Regressor,
    components::NamedTuple,
    count::Int,
    workers::Int,
)
    population isa AbstractVector || return ArgumentError("init_population must return a vector")
    length(population) == count || return ArgumentError(
        "init_population must return exactly N individuals",
    )
    policy = get(components, :bloat_policy, Mutations.BloatPolicy())
    issues = _parallel_map(collect(eachindex(population)),
            (; enabled=true, workers)) do index
        issue = _validate_program_contract(
            population[index], reg.target_ty, components, "initial program $index",
        )
        issue === nothing || return issue
        _within_bloat_limit(population[index].tree, policy) || return ArgumentError(
            "initial individual $index exceeds the configured structural limit",
        )
        return nothing
    end
    for issue in issues
        issue === nothing || return issue
    end
    return nothing
end

function _age_initial_scratch_state(state::AgeSearchState)
    return AgeSearchState(
        state.reg,
        state.data,
        state.seed,
        Vector{Vector{AgeSearchIndividual}}(),
        AbstractRNG[],
        1,
        state.birth_clock,
        UInt64(1),
        AgeSearchIndividual[],
        0,
        Dict{Int,AgeSearchIndividual}(),
        nothing,
        AgeSearchCounters(),
    )
end

function _age_score_initial_population_parallel(
    state::AgeSearchState,
    population::AbstractVector,
    workers::Int,
)
    return _parallel_map(collect(eachindex(population)), (; enabled=true, workers)) do index
        scratch = _age_initial_scratch_state(state)
        record = _age_score_record!(
            scratch, population[index], (), :initial, :init; type_checked=true,
        )
        (; record, counters=scratch.counters)
    end
end

function _age_publish_initial_scores!(
    state::AgeSearchState,
    population::Vector{AgeSearchIndividual},
    scores,
)
    for score in scores
        counters = score.counters
        state.counters.objective_calls += counters.objective_calls
        state.counters.init_objective_calls += counters.init_objective_calls
        state.counters.sample_executions += counters.sample_executions
        state.counters.init_sample_executions += counters.init_sample_executions
        state.counters.invalids += counters.invalids
        state.counters.init_invalids += counters.init_invalids

        worker_record = score.record
        record = _age_make_record(
            state, worker_record.program, worker_record.parent_ids,
            worker_record.origin, worker_record.loss, worker_record.complexity,
            worker_record.valid,
        )
        record = AgeSearchIndividual(
            record.program, record.id, record.parent_ids, record.origin,
            record.loss, record.complexity, record.valid, _age_birth!(state),
        )
        push!(population, record)
        _age_archive!(state, record)
    end
    return nothing
end

@testitem "Age initialization parallel scoring publishes in slot order" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: get_type_ctx
    using LambdaRegression.Regression: AgeSearchState, _age_initial_parallel_status
    import Base.Threads

    target_ty = TyFun(FloatTy, FloatTy)
    globals = Dict{Symbol,Any}(:FADD => FADD)
    global_types = get_type_ctx(globals)
    components = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_candidates_per_individual=1,
        init_growth_attempts=0,
    )
    serial_reg = Regressor(target_ty, components, default_config(
        population_size=6, parallel=(enabled=false,),
    ), MersenneTwister(1111))
    parallel_reg = Regressor(target_ty, components, default_config(
        population_size=6, parallel=(enabled=true, workers=8),
    ), MersenneTwister(1111))
    data = SupervisedDataset([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    serial = initialize_age_search(
        serial_reg, data; population_count=1, population_size=6, seed=62,
    )
    parallel = initialize_age_search(
        parallel_reg, data; population_count=1, population_size=6, seed=62,
    )

    @test serial isa AgeSearchState
    @test parallel isa AgeSearchState
    if Threads.nthreads(:default) >= 2
        status = _age_initial_parallel_status(parallel, 6)
        @test status.enabled
        @test status.workers >= 2
    end
    serial_records = only(serial.populations)
    parallel_records = only(parallel.populations)
    @test length(serial_records) == length(parallel_records) == 6
    @test [record.id for record in parallel_records] == UInt64.(1:6)
    @test [record.birth for record in parallel_records] == UInt64.(1:6)
    @test [record.loss for record in parallel_records] == [record.loss for record in serial_records]
    @test parallel.birth_clock == serial.birth_clock == 6
    @test parallel.next_id == serial.next_id == 7
    @test parallel.counters.init_objective_calls == 6
    @test parallel.counters.init_sample_executions == 18
    @test length(parallel.archive) <= 6
    @test sort!(collect(keys(parallel.archive))) == sort!(collect(keys(serial.archive)))

    invalid_globals = Dict{Symbol,Any}(:NAN => NaN)
    invalid_components = default_components(
        global_consts=invalid_globals,
        global_consts_ty=get_type_ctx(invalid_globals),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        init_population=(reg, data; N, seed, global_type_ctx) -> [
            Program(AstTree(AbsNode(:x, ConstNode(:NAN)))) for _ in 1:N
        ],
    )
    serial_invalid_reg = Regressor(target_ty, invalid_components, default_config(
        population_size=3, parallel=(enabled=false,),
    ), MersenneTwister(8000))
    parallel_invalid_reg = Regressor(target_ty, invalid_components, default_config(
        population_size=3, parallel=(enabled=true, workers=8),
    ), MersenneTwister(8000))
    serial_invalid = initialize_age_search(
        serial_invalid_reg, data; population_count=1, population_size=3, seed=64,
    )
    parallel_invalid = initialize_age_search(
        parallel_invalid_reg, data; population_count=1, population_size=3, seed=64,
    )
    serial_invalid_records = only(serial_invalid.populations)
    parallel_invalid_records = only(parallel_invalid.populations)
    if Threads.nthreads(:default) >= 2
        @test _age_initial_parallel_status(parallel_invalid, 3).enabled
    end
    @test all(!record.valid for record in parallel_invalid_records)
    @test [record.id for record in parallel_invalid_records] == UInt64.(1:3)
    @test [record.birth for record in parallel_invalid_records] == UInt64.(1:3)
    @test [record.loss for record in parallel_invalid_records] ==
        [record.loss for record in serial_invalid_records]
    @test parallel_invalid.birth_clock == serial_invalid.birth_clock == 3
    @test parallel_invalid.next_id == serial_invalid.next_id == 4
    @test parallel_invalid.counters.init_invalids == serial_invalid.counters.init_invalids == 3
    @test parallel_invalid.counters.init_objective_calls ==
        serial_invalid.counters.init_objective_calls == 3
    @test parallel_invalid.counters.init_sample_executions ==
        serial_invalid.counters.init_sample_executions == 9
    @test isempty(parallel_invalid.archive)
    @test isempty(serial_invalid.archive)
end
