"""One managed family's real witnesses and accounting history; never a shared score."""
mutable struct StructureFamily
    representatives::Vector{AgeSearchIndividual}
    admissions::Int
    parent_selections::Int
    ledger_parent_selections::Int
    development_actions::Int
    development_evaluations::Int
    refused_actions::Int
    development_seconds::Float64
    touched::Int
end

@testsnippet ContinuousWorkflowFixture begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode, ConstNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: AbstractMutation
    using LambdaRegression.Regression: default_components, default_config,
        default_regressor, SupervisedDataset, initialize_age_search
    using LambdaRegression.Workflows: StructurePolicy, ContinuousSearchConfig,
        run_continuous_search!, structure_snapshot
    const CW = LambdaRegression.Workflows
    const CR = LambdaRegression.Regression
    continuous_identity() = Program(AstTree(AbsNode(:x, IndexNode(0))))
    continuous_constant(value) = Program(AstTree(AbsNode(:x, ConstNode(:c)));
        local_consts=Dict{Symbol,Any}(:c => value),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    function continuous_fixture(; size=2)
        seeds = [continuous_identity(), continuous_constant(0.0)]
        components = default_components(global_consts=Dict{Symbol,Any}(
                :SQR => LambdaRegression.Primitives.SQR),
            global_consts_ty=Dict{Symbol,TyLike}(:SQR => TyFun(FloatTy, FloatTy)), input_slots=[:x],
            input_types=Dict{Symbol,TyLike}(:x => FloatTy),
            mutations=AbstractMutation[], crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) ->
                [seeds[mod1(i, length(seeds))] for i in 1:N])
        reg = default_regressor(TyFun(FloatTy, FloatTy);
            rng=MersenneTwister(1234), components,
            config=default_config(population_size=size, tournament_size=1))
        return initialize_age_search(reg, SupervisedDataset([1.0, 2.0], [1.0, 2.0]);
            population_count=1, population_size=size, seed=3)
    end
    continuous_key(program) = program.tree.root.body isa IndexNode ? :identity : :constant
    struct ContinuousHashedFamilyKey
        value::Symbol
        salt::UInt
    end
    Base.isequal(a::ContinuousHashedFamilyKey, b::ContinuousHashedFamilyKey) =
        a.value === b.value && a.salt == b.salt
    Base.hash(key::ContinuousHashedFamilyKey, seed::UInt) =
        hash(key.value, seed) ⊻ key.salt
    continuous_hashed_key(salt) = program ->
        ContinuousHashedFamilyKey(continuous_key(program), UInt(salt))
end

@testitem "Continuous workflow family survival and real ledger mating" setup=[ContinuousWorkflowFixture] begin
    state = continuous_fixture()
    policy = StructurePolicy(continuous_key; capacity=2, representatives=2,
        family_probability=1.0, development_evaluations=4, refusal_limit=2)
    CW._seed_structure_policy!(policy, state)
    @test length(policy.families) == 2
    source = state.populations[1][1]
    other = state.populations[1][2]
    # Both last representatives have not yet received local development.
    new = CR._age_score_record!(state, continuous_constant(1.5), (), :test, :advance)
    # A separate descriptor labels a new concrete constant as a new family.
    policy2 = StructurePolicy(p -> p.tree.root.body isa IndexNode ? :identity :
        p.local_consts[:c]; capacity=3, representatives=2,
        family_probability=1.0, development_evaluations=4, refusal_limit=2)
    CW._seed_structure_policy!(policy2, state)
    @test CR.age_replacement_index(policy2, state, new, 1) === nothing
    @test !CR._age_insert!(state, new, 1; policy=policy2)
    @test any(r -> r.id == new.id, CW._structure_candidates(policy2, [source]))
    @test policy2.families[1.5].admissions == 0
    @test CR._age_selected_parent(policy2, state, 1, [source], MersenneTwister(2000), 1,
        :parent) isa CR.AgeSearchIndividual
    @test sum(f.parent_selections for f in values(policy2.families)) == 1
    # Verify the last interface too: a real worse-than-best witness that was
    # refused live admission reaches the next complete-program proposal.
    seed = findfirst(1:32) do trial
        CR._age_selected_parent(policy2, state, 1, [source], MersenneTwister(trial),
            1, :parent) === new
    end
    @test seed !== nothing
    state.rngs[1] = MersenneTwister(seed)
    inherited = Ref(UInt64(0))
    proposal = (state, island, left, right; rng) -> begin
        inherited[] = left.id
        continuous_constant(1.25)
    end
    CR.advance_age_search!(state; max_attempts=1, max_evaluations=1, policy=policy2, proposal)
    @test inherited[] == new.id
    @test new.loss > source.loss
    CW._credit_development!(policy2, state, other, 0, 0.1, true)
    @test CR.age_replacement_index(policy2, state, new, 1) === nothing
    CW._credit_development!(policy2, state, other, 0, 0.1, true)
    @test CR.age_replacement_index(policy2, state, new, 1) == 2
    @test policy2.families[0.0].development_evaluations == 0
    @test policy2.families[0.0].refused_actions == 2
    @test CR._age_insert!(state, new, 1; policy=policy2)
    @test policy2.families[1.5].development_actions == 0
    @test policy2.families[1.5].admissions == 1
    @test length(policy2.families) <= policy2.capacity
    @test_throws ArgumentError CW._seed_structure_policy!(policy2, continuous_fixture())
end

@testitem "Continuous workflow sampling is stable across descriptor key hashes" setup=[ContinuousWorkflowFixture] begin
    state = continuous_fixture(size=8)
    records = state.populations[1]
    candidates = reverse(copy(records))
    push!(candidates, first(records))
    salt_a, salt_b = UInt(0), UInt(1)
    @test hash(ContinuousHashedFamilyKey(:identity, salt_a), UInt(0)) !=
        hash(ContinuousHashedFamilyKey(:identity, salt_b), UInt(0))

    policy_a = StructurePolicy(continuous_hashed_key(salt_a);
        capacity=8, representatives=8, family_probability=1.0)
    policy_b = StructurePolicy(continuous_hashed_key(salt_b);
        capacity=8, representatives=8, family_probability=1.0)
    pool_a = CW._structure_candidates(policy_a, candidates)
    pool_b = CW._structure_candidates(policy_b, candidates)
    expected_ids = sort!(unique([record.id for record in records]))
    @test [record.id for record in pool_a] == expected_ids
    @test [record.id for record in pool_b] == expected_ids

    selected_a = [CR.age_select_parent(policy_a, state, 1, pool_a,
        MersenneTwister(seed), 1, :parent).id for seed in 1:64]
    selected_b = [CR.age_select_parent(policy_b, state, 1, pool_b,
        MersenneTwister(seed), 1, :parent).id for seed in 1:64]
    @test selected_a == selected_b
end

@testitem "Continuous workflow bounded history and unknown descriptions" setup=[ContinuousWorkflowFixture] begin
    state = continuous_fixture(size=1)
    policy = StructurePolicy(p -> p.tree.root.body isa IndexNode ? :identity :
        p.local_consts[:c]; capacity=2, representatives=1, history_capacity=1,
        development_evaluations=0)
    CW._seed_structure_policy!(policy, state)
    for value in (1.0, 2.0, 3.0)
        record = CR._age_score_record!(state, continuous_constant(value), (), :test, :advance)
        CR._age_insert!(state, record, 1; policy)
        CW._prune_structure_keys!(policy, state)
    end
    @test length(policy.families) <= 2
    @test length(policy.history) <= 1
    @test length(policy.keys) <= 3
    unknown = StructurePolicy(_ -> nothing)
    CW._seed_structure_policy!(unknown, state)
    @test isempty(unknown.families)
    @test CR.age_replacement_index(unknown, state, state.populations[1][1], 1) == 1
end

@testitem "Continuous workflow real descriptions preserve independent scores" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ASTs.NodeASTs: AppNode
    using LambdaRegression.StructureDiversity: StructureWorld, StructureBoundaryPolicy
    state = continuous_fixture()
    world = StructureWorld(state.reg.components.global_consts_ty;
        values=state.reg.components.global_consts,
        input_types=state.reg.components.input_types, target_ty=state.reg.target_ty)
    policy = StructurePolicy(world; capacity=4, representatives=3)
    CW._seed_structure_policy!(policy, state)
    original = state.populations[1][1]
    square = Program(AstTree(AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0)))))
    edited = CR._age_score_record!(state, square, (original.id,), :test, :advance)
    @test isequal(CW._structure_key(policy, original), CW._structure_key(policy, edited))
    @test edited.loss > original.loss
    @test CR._age_insert!(state, edited, 1; policy)
    family = policy.families[CW._structure_key(policy, edited)]
    losses = Dict(r.id => r.loss for r in family.representatives)
    @test losses[original.id] == original.loss
    @test losses[edited.id] == edited.loss
    @test policy.describe(Program(AstTree(ConstNode(:missing)))) === nothing
    @test_throws ArgumentError run_continuous_search!(state; rng=MersenneTwister(1),
        policy, config=ContinuousSearchConfig(max_actions=0), outer=(;),
        typed=(enabled=false,), scalar=(enabled=true, named_input_policy=:declared))
    declared = StructurePolicy(world;
        boundary_policy=StructureBoundaryPolicy(named_input_policy=:declared))
    @test declared.describe.boundary_policy.named_input_policy === :declared
end

@testitem "StructurePolicy batch selection freezes candidates and preserves draws" setup=[ContinuousWorkflowFixture] begin
    state, reference_state = continuous_fixture(), continuous_fixture()
    policy = StructurePolicy(continuous_key; capacity=4, representatives=3,
        family_probability=1.0)
    reference_policy = StructurePolicy(continuous_key; capacity=4, representatives=3,
        family_probability=1.0)
    valid = CR._age_valid_records(state, 1)
    reference_valid = CR._age_valid_records(reference_state, 1)
    selection = CR._age_batch_selection(policy, state, Dict(1 => valid))

    family_counts(candidate) = sort!([(repr(key), family.parent_selections,
        family.ledger_parent_selections) for (key, family) in candidate.families]; by=first)
    for seed in 1:32
        selected = CR._age_selected_parent(selection, state, 1, valid,
            MersenneTwister(seed), 2, :parent)
        reference = CR._age_selected_parent(reference_policy, reference_state, 1,
            reference_valid, MersenneTwister(seed), 2, :parent)
        @test selected.id == reference.id
    end
    @test family_counts(policy) == family_counts(reference_policy)

    old_ids = Set(record.id for record in
        CR.age_parent_candidates(selection, state, 1, valid, :parent))
    raw = CR._age_score_record!(state, continuous_constant(3.0), (), :test, :advance)
    published = CR.AgeSearchIndividual(raw.program, raw.id, raw.parent_ids,
        raw.origin, raw.loss, raw.complexity, raw.valid, CR._age_birth!(state))
    @test CR._age_insert!(state, published, 1; policy)
    @test Set(record.id for record in
        CR.age_parent_candidates(selection, state, 1, valid, :parent)) == old_ids

    next_valid = CR._age_valid_records(state, 1)
    next_selection = CR._age_batch_selection(policy, state, Dict(1 => next_valid))
    next_ids = Set(record.id for record in
        CR.age_parent_candidates(next_selection, state, 1, next_valid, :parent))
    @test published.id ∈ next_ids
    @test published.id ∉ old_ids
end

@testitem "Parallel StructurePolicy hooks stay coordinator-owned and repeat deterministically" setup=[ContinuousWorkflowFixture] begin
    import Base.Threads
    using LambdaRegression.StructureDiversity: StructureWorld
    using LambdaRegression.Mutations: ScalarConstMutation

    struct ContinuousPolicyProbe{P} <: CR.AbstractAgeSearchPolicy
        inner::P
        events::Channel{Tuple{Symbol,Task}}
    end
    probe_event!(probe, name) = put!(probe.events, (name, current_task()))
    CR._age_parallel_policy(::ContinuousPolicyProbe) = true
    function CR._age_batch_selection(probe::ContinuousPolicyProbe, state, valid_by_island)
        probe_event!(probe, :batch_selection)
        return ContinuousPolicyProbe(
            CR._age_batch_selection(probe.inner, state, valid_by_island), probe.events)
    end
    function CR.age_parent_candidates(probe::ContinuousPolicyProbe, state, island,
            candidates, role::Symbol)
        probe_event!(probe, :parent_candidates)
        return CR.age_parent_candidates(probe.inner, state, island, candidates, role)
    end
    function CR.age_select_parent(probe::ContinuousPolicyProbe, state, island,
            candidates, rng, tournament_size, role::Symbol; eligible=nothing)
        probe_event!(probe, :select_parent)
        return CR.age_select_parent(probe.inner, state, island, candidates, rng,
            tournament_size, role; eligible)
    end
    function CR.age_replacement_index(probe::ContinuousPolicyProbe, state, record, island)
        probe_event!(probe, :replacement_index)
        return CR.age_replacement_index(probe.inner, state, record, island)
    end
    function CR.age_observe_candidate!(probe::ContinuousPolicyProbe, state, record,
            island, replaced, admitted)
        probe_event!(probe, :observe_candidate)
        return CR.age_observe_candidate!(probe.inner, state, record, island,
            replaced, admitted)
    end

    population_size, workers = 6, 8
    data = SupervisedDataset([-2.0, -1.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, 0.0, 1.0])
    input_types = Dict{Symbol,TyLike}(:x => FloatTy)
    world = StructureWorld(Dict{Symbol,TyLike}(); input_types=input_types,
        target_ty=TyFun(FloatTy, FloatTy))
    canonical_policy = StructurePolicy(world; capacity=12, representatives=3,
        family_probability=0.75, development_evaluations=0)

    function make_state()
        components = default_components(input_slots=[:x], input_types=input_types,
            mutations=AbstractMutation[ScalarConstMutation(σ=0.5)],
            mutation_max_try=1, crossover_op=nothing,
            init_population=(reg, data; N, seed, global_type_ctx) ->
                [continuous_constant(Float64(i)) for i in 1:N])
        reg = default_regressor(TyFun(FloatTy, FloatTy); rng=MersenneTwister(1234),
            components, config=default_config(population_size=population_size,
                tournament_size=2, parallel=(enabled=true, workers=workers)))
        return initialize_age_search(reg, data; population_count=2,
            population_size=population_size, seed=73)
    end

    function make_probe(events)
        inner = StructurePolicy(program -> begin
            put!(events, (:describe, current_task()))
            canonical_policy.describe(program)
        end; capacity=12, representatives=3, family_probability=0.75,
            development_evaluations=0)
        return ContinuousPolicyProbe(inner, events)
    end

    threads = Threads.nthreads(:default)
    state_for_status = make_state()
    real_status = CR.age_parallel_status(state_for_status; policy=StructurePolicy(world))
    @test real_status.enabled == (threads > 1)
    @test real_status.reason == (threads > 1 ? :coordinator_policy : :single_worker)
    @test real_status.workers == (threads > 1 ? min(workers, threads) : 1)
    @test real_status.trajectory_changes == real_status.enabled
    if threads >= workers
        @test real_status.workers == workers
    end
    @test :policy ∉ fieldnames(CR._AgeWorkerContext)
    @test :policy ∉ fieldnames(CR._AgeAttemptJob)
    @test :rng ∉ fieldnames(CR._AgeWorkerSettings)

    record_signature(record) = (record.program.tree, record.program.local_consts,
        record.program.local_consts_ty, record.id, record.parent_ids, record.origin,
        record.loss, record.complexity, record.valid, record.birth)
    function state_signature(state)
        counters = Tuple(getfield(state.counters, field)
            for field in fieldnames(typeof(state.counters)))
        return (populations=[record_signature.(population) for population in state.populations],
            archive=sort!([(id, record_signature(record)) for (id, record) in state.archive];
                by=first),
            recent=record_signature.(state.recent_valid_offspring),
            best=record_signature(state.best), cursor=state.cursor,
            birth_clock=state.birth_clock, next_id=state.next_id, counters,
            rngs=[rand(copy(rng), UInt64, 8) for rng in state.rngs])
    end
    function policy_signature(policy)
        family_state = [(key, record_signature.(family.representatives),
            family.admissions, family.parent_selections, family.ledger_parent_selections,
            family.development_actions, family.development_evaluations,
            family.refused_actions, family.touched)
            for (key, family) in policy.inner.families]
        sort!(family_state; by=entry -> repr(first(entry)))
        history_state = [(key, value.admissions, value.actions, value.evaluations,
            value.refused, value.touched) for (key, value) in policy.inner.history]
        sort!(history_state; by=entry -> repr(first(entry)))
        key_state = sort!(collect(policy.inner.keys); by=first)
        return (family_state, history_state, key_state, policy.inner.clock,
            policy.inner.overflow)
    end
    function run_once()
        state = make_state()
        events = Channel{Tuple{Symbol,Task}}(8192)
        policy = make_probe(events)
        coordinator = current_task()
        status = CR.age_parallel_status(state; policy)
        report = CR.advance_age_search!(state; max_attempts=32,
            max_evaluations=32, policy)
        observed = Tuple{Symbol,Task}[]
        while isready(events)
            push!(observed, take!(events))
        end
        return (; state, policy, status, report, observed, coordinator)
    end

    first_run, second_run = run_once(), run_once()
    @test first_run.status.enabled == (threads > 1)
    @test first_run.status.workers == (threads > 1 ? min(workers, threads) : 1)
    @test first_run.status.trajectory_changes == first_run.status.enabled
    @test first_run.report == second_run.report
    @test first_run.report.objective_calls > 0
    @test state_signature(first_run.state) == state_signature(second_run.state)
    @test policy_signature(first_run.policy) == policy_signature(second_run.policy)
    @test !isempty(first_run.observed)
    @test all(task === first_run.coordinator for (_, task) in first_run.observed)
    @test all(task === second_run.coordinator for (_, task) in second_run.observed)
    kinds = (:batch_selection, :describe, :parent_candidates, :select_parent,
        :replacement_index, :observe_candidate)
    @test all(kind in first.(first_run.observed) for kind in kinds)
    @test [(kind, count(event -> first(event) === kind, first_run.observed))
        for kind in kinds] == [(kind, count(event -> first(event) === kind,
            second_run.observed)) for kind in kinds]
end

@testitem "Continuous workflow mixed material sources retain exploration" setup=[ContinuousWorkflowFixture] begin
    state = continuous_fixture()
    append!(state.recent_valid_offspring, state.populations[1])
    pool = CW._continuous_material_pool(state, nothing)
    @test length(pool) == length(unique(r.id for r in pool)) == 2
    chosen = Set(CW._continuous_source(state, nothing, MersenneTwister(seed), 1.0).id
        for seed in 1:32)
    @test chosen == Set(r.id for r in pool)
    @test maximum(r.loss for r in pool if r.id in chosen) > minimum(r.loss for r in pool)
end

@testitem "Continuous constant source filtering survives family selection" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.Programs: numeric_parameter_blocks
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch

    state = continuous_fixture(size=2)
    policy = StructurePolicy(continuous_key; capacity=2, representatives=2,
        family_probability=1.0)
    CW._seed_structure_policy!(policy, state)
    eligible_ids = [record.id for record in state.populations[1]
        if !isempty(numeric_parameter_blocks(record.program))]
    @test length(eligible_ids) == 1
    eligible_id = only(eligible_ids)

    # A scored, same-family representative is fed back by family selection.
    # The source filter must still exclude it after that feedback expansion.
    rejected = CR._age_score_record!(state, continuous_constant(4.0), (), :test, :advance)
    @test !isempty(numeric_parameter_blocks(rejected.program))
    push!(policy.families[:constant].representatives, rejected)
    rejected_filter_calls = Ref(0)
    source_filter = function (record)
        if record.id == rejected.id
            rejected_filter_calls[] += 1
        end
        return record.id == eligible_id
    end
    result = run_continuous_search!(state; rng=MersenneTwister(2000), policy,
        config=ContinuousSearchConfig(max_actions=1, constant_weight=100.0,
            source_exploration_probability=0.0),
        outer=(attempts=0, evaluations=0, max_nodes=16, max_depth=4),
        typed=(enabled=false,), scalar=(enabled=false,),
        constant=(enabled=true, strategy=GaussianConstantSearch(),
            budget=ConstantSearchBudget(max_proposals=1, max_objective_evals=1),
            source_filter))

    entry = only(result.ledger)
    @test entry.kind === :constant
    @test entry.source_id == eligible_id
    @test entry.source_id != rejected.id
    @test rejected_filter_calls[] >= 2
end

@testitem "Continuous constant source filter can refuse every source" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch

    state = continuous_fixture(size=2)
    calls_before = state.counters.objective_calls
    filter_calls = Ref(0)
    source_filter = record -> (filter_calls[] += 1; false)
    result = run_continuous_search!(state; rng=MersenneTwister(1),
        config=ContinuousSearchConfig(max_actions=1, constant_weight=100.0),
        outer=(attempts=0, evaluations=0, max_nodes=16, max_depth=4),
        typed=(enabled=false,), scalar=(enabled=false,),
        constant=(enabled=true, strategy=GaussianConstantSearch(),
            budget=ConstantSearchBudget(max_proposals=1, max_objective_evals=1),
            source_filter))

    entry = only(result.ledger)
    @test entry.kind === :constant
    @test entry.refused
    @test entry.report.reason === :no_eligible_constant_source
    @test filter_calls[] > 0
    @test entry.report.constant_calls == 0
    @test result.counts.constant.calls == 0
    @test state.counters.objective_calls == calls_before
end

@testitem "Continuous workflow local publication and accounting" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    state = continuous_fixture()
    policy = StructurePolicy(continuous_key; development_evaluations=4)
    cfg = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(), constant_refinement_iterations=0,
        max_constant_refinement_evaluations=0, evolution_strategy=:regularized)
    observed = NamedTuple[]
    result = run_continuous_search!(state; rng=MersenneTwister(1), policy,
        config=ContinuousSearchConfig(max_actions=4, scalar_weight=3.0),
        outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
        typed=(enabled=false,), scalar=(enabled=true, config=cfg, max_return=1,
            material_return=1), on_action=e -> push!(observed, e))
    @test result.state === state
    @test length(observed) == length(result.ledger) == 4
    @test count(e -> e.kind === :outer, observed) == 1
    @test count(e -> e.kind === :scalar, observed) == 3
    @test [e.kind for e in observed] == [:scalar, :outer, :scalar, :scalar]
    @test result.counts.constant.calls == 0
    @test result.counts.scalar.calls > 0
    @test sum(e.development_objective_calls for e in observed) == result.counts.scalar.calls
    @test sum(f.development_evaluations for f in values(policy.families)) == result.counts.scalar.calls
    @test any(e -> e.kind === :scalar && !e.refused && e.returned > 0, observed)
    @test all(e -> e.elapsed_seconds >= 0, observed)
    @test length(policy.keys) <= length(state.populations[1]) +
        policy.capacity * policy.representative_capacity
end

@testitem "Continuous workflow forwards contextual scalar backend without changing defaults" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ScalarSearch: ScalarSearchConfig, search_scalar
    cfg = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(),
        constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    backend_calls = Ref(0)
    backend = function (X, y; kwargs...)
        backend_calls[] += 1
        @test kwargs[:config] === cfg
        @test kwargs[:stop_loss] === nothing
        @test length(kwargs[:initial_population]) == 1
        @test applicable(kwargs[:objective], first(kwargs[:initial_population]))
        return search_scalar(X, y; kwargs...)
    end
    function run_with(options)
        run_continuous_search!(continuous_fixture(); rng=MersenneTwister(1),
            config=ContinuousSearchConfig(max_actions=4, scalar_weight=3.0),
            outer=(attempts=0, evaluations=0, max_nodes=32, max_depth=8),
            typed=(enabled=false,),
            scalar=merge((enabled=true, config=cfg, max_return=1,
                material_return=1), options))
    end
    ordinary = run_with((;))
    supplied = run_with((; search_backend=backend))
    @test backend_calls[] > 0
    @test backend_calls[] == count(e -> e.kind === :scalar && !e.refused, supplied.ledger)
    @test supplied.counts.scalar.calls == ordinary.counts.scalar.calls
    @test supplied.best.loss == ordinary.best.loss
    @test CW._same_program(supplied.best.program, ordinary.best.program)
    @test [(e.kind, e.path, e.refused, e.returned) for e in supplied.ledger] ==
        [(e.kind, e.path, e.refused, e.returned) for e in ordinary.ledger]
end

@testitem "Continuous workflow keeps zero-input scalar search opt-in" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.ScalarSearch: ScalarSearchConfig

    source_program = Program(AstTree(AbsNode(:x, ConstNode(:c)));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    components = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=[:x], input_types=Dict{Symbol,TyLike}(:x => BoolTy),
        mutations=AbstractMutation[], crossover_op=nothing,
        init_population=(reg, data; N, seed, global_type_ctx) ->
            [source_program for _ in 1:N])
    reg = default_regressor(TyFun(BoolTy, FloatTy);
        rng=MersenneTwister(2000), components,
        config=default_config(population_size=1, tournament_size=1))
    state = initialize_age_search(reg, SupervisedDataset([true], [1.0]);
        population_count=1, population_size=1, seed=17)
    scalar_config = ScalarSearchConfig(population_size=2, max_evaluations=2,
        max_complexity=4, initial_depth=0, tournament_size=1, elite_count=1,
        constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    calls_before = state.counters.objective_calls
    result = run_continuous_search!(state; rng=MersenneTwister(1),
        config=ContinuousSearchConfig(max_actions=1, scalar_weight=100.0),
        outer=(attempts=0, evaluations=0, max_nodes=16, max_depth=4),
        typed=(enabled=false,),
        scalar=(enabled=true, config=scalar_config, max_return=1,
            material_return=0))

    @test result.ledger[1].kind === :scalar
    @test result.ledger[1].refused
    @test any(r -> r.reason === :disabled_zero_input, result.ledger[1].refusals)
    @test result.counts.scalar.calls == 0
    @test state.counters.objective_calls == calls_before
end

@testitem "Continuous workflow constant action scores and reclassifies matrix material" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.TypeSystem: TyArray, TyNat
    using LambdaRegression.Programs: numeric_parameter_blocks

    matrix_ty = TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    source_values = zeros(2, 2)
    source_program = Program(AstTree(AbsNode(:x, ConstNode(:A)));
        local_consts=Dict{Symbol,Any}(:A => source_values),
        local_consts_ty=Dict{Symbol,TyLike}(:A => matrix_ty))
    components = merge(default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=[:x], input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        mutations=AbstractMutation[], crossover_op=nothing,
        init_population=(reg, data; N, seed, global_type_ctx) ->
            [source_program for _ in 1:N]),
        (; loss=(prediction, target) -> -sum(abs2, prediction)))
    reg = default_regressor(TyFun(FloatTy, matrix_ty);
        rng=MersenneTwister(8000), components,
        config=default_config(population_size=2, tournament_size=1))
    state = initialize_age_search(reg,
        SupervisedDataset([1.0], [zeros(2, 2)]);
        population_count=1, population_size=2, seed=21)
    source = state.populations[1][1]
    original_value = source.program.local_consts[:A]
    policy = StructurePolicy(p -> all(iszero, p.local_consts[:A]) ? :zero : :nonzero;
        capacity=3, representatives=2, development_evaluations=0)
    strategy = CR.GaussianConstantSearch(σ=0.25, tensor_modes=(:global_bias,))
    budget = CR.ConstantSearchBudget(max_proposals=1, max_objective_evals=2)
    result = run_continuous_search!(state; rng=MersenneTwister(42), policy,
        config=ContinuousSearchConfig(max_actions=1, constant_weight=100.0),
        outer=(attempts=0, evaluations=0, max_nodes=16, max_depth=4),
        typed=(enabled=false,), scalar=(enabled=false,),
        constant=(enabled=true, strategy, budget))

    entry = only(result.ledger)
    @test entry.kind === :constant
    @test !entry.refused
    @test entry.returned == entry.incoming == 1
    @test entry.development_objective_calls == 2
    @test entry.objective_calls == 3 # two search scores and one migrant rescore
    @test entry.report.status === :improved
    @test entry.report.constant_calls == 2
    @test entry.report.constant_stats.tensor_proposals == 1
    @test entry.report.constant_stats.objective_evaluations == result.counts.constant.calls == 2
    @test result.counts.constant.samples == 2
    @test numeric_parameter_blocks(source.program) == [:A => :tensor]
    @test source.program.local_consts[:A] === original_value
    @test source.program.local_consts[:A] == zeros(2, 2)
    @test source.loss == 0.0
    candidate_index = findfirst(record -> record.id != source.id &&
        !CW._same_program(record.program, source.program), state.populations[1])
    @test candidate_index !== nothing
    candidate = state.populations[1][candidate_index]
    @test candidate.loss < source.loss
    @test size(candidate.program.local_consts[:A]) == (2, 2)
    @test candidate.program.local_consts_ty[:A] == matrix_ty
    @test candidate.program.local_consts[:A] !== original_value
    @test CW._structure_key(policy, source) === :zero
    @test CW._structure_key(policy, candidate) === :nonzero
    @test any(record -> CW._same_program(record.program, source.program) &&
        record.loss == source.loss, policy.families[:zero].representatives)
    @test any(record -> record.id == candidate.id && record.loss == candidate.loss,
        policy.families[:nonzero].representatives)
end

@testitem "Continuous workflow constant action refuses sources without numeric blocks" setup=[ContinuousWorkflowFixture] begin
    using LambdaRegression.Programs: numeric_parameter_blocks
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch
    state = continuous_fixture(size=1)
    @test isempty(numeric_parameter_blocks(state.populations[1][1].program))
    calls_before = state.counters.objective_calls
    result = run_continuous_search!(state; rng=MersenneTwister(1),
        config=ContinuousSearchConfig(max_actions=1, constant_weight=100.0),
        outer=(attempts=0, evaluations=0, max_nodes=16, max_depth=4),
        typed=(enabled=false,), scalar=(enabled=false,),
        constant=(enabled=true, strategy=GaussianConstantSearch(),
            budget=ConstantSearchBudget(max_proposals=1, max_objective_evals=2)))

    entry = only(result.ledger)
    @test entry.kind === :constant
    @test entry.refused
    @test entry.report.reason === :no_numeric_blocks
    @test entry.report.constant_calls == 0
    @test entry.report.constant_stats === nothing
    @test result.counts.constant.calls == 0
    @test state.counters.objective_calls == calls_before
end

"""
    StructurePolicy(describe; capacity=64, representatives=3, history_capacity=256,
        family_probability=0.5, development_evaluations=64, refusal_limit=4)

Bounded family-aware parent selection and replacement over real scored programs.
`describe(program)` returns a canonical key, a `(canonical_key, view, version)`
record, or `nothing` for unknown/unmanaged programs. Keys must remain read-only.
The callback closes over one stable task/profile world; family equality never
licenses score, type-evidence or local-session reuse. A state/policy pair must
stay in that same world. Scored representatives may breed before live admission.

Same-family children replace that family's oldest local member. Cross-family
children may replace redundant members or unprotected last representatives;
otherwise admission is refused. Protection requires previous live admission
and expires after actual local objective calls or bounded refusal actions.
The history window remembers released non-live families only; outside that
finite window rediscovery can receive fresh opportunity. Mutable policy/state
ownership is exclusive, and all callbacks are synchronous.
With parallel age search, hooks run on the coordinator task and each batch
selects parents before publishing results, delaying cross-island policy feedback
and changing the trajectory; unknown policies retain serial execution.
"""
mutable struct StructurePolicy{F} <: Regression.AbstractAgeSearchPolicy
    describe::F
    capacity::Int
    representative_capacity::Int
    history_capacity::Int
    family_probability::Float64
    development_limit::Int
    refusal_limit::Int
    families::Dict{Any,StructureFamily}
    history::Dict{Any,NamedTuple}
    keys::Dict{UInt64,Any}
    clock::Int
    overflow::Int
    owner::Union{Nothing,AgeSearchState}
end

# Hooks stay on the coordinator. Selection may use a batch-start view; family
# feedback is applied in ordered publication, without sharing mutable policy
# state with workers.
Regression._age_parallel_policy(::StructurePolicy) = true

function StructurePolicy(describe; capacity::Int=64, representatives::Int=3,
        history_capacity::Int=256, family_probability::Real=0.5,
        development_evaluations::Int=64, refusal_limit::Int=4)
    capacity > 0 && representatives > 0 && history_capacity >= 0 ||
        throw(ArgumentError("family/representative capacities must be positive; history nonnegative"))
    0 <= family_probability <= 1 || throw(ArgumentError("family_probability must lie in [0,1]"))
    development_evaluations >= 0 && refusal_limit > 0 ||
        throw(ArgumentError("development limit must be nonnegative and refusal limit positive"))
    return StructurePolicy(describe, capacity, representatives, history_capacity,
        Float64(family_probability), development_evaluations, refusal_limit,
        Dict{Any,StructureFamily}(), Dict{Any,NamedTuple}(), Dict{UInt64,Any}(), 0, 0, nothing)
end

struct _StructureClassifier
    world::StructureWorld
    boundary_policy::StructureBoundaryPolicy
end
function (classifier::_StructureClassifier)(program::Program)
    description = describe_structure(program, classifier.world;
        boundary_policy=classifier.boundary_policy)
    return description isa Err ? nothing : description.canonical_key
end

"""
    StructurePolicy(world::StructureWorld;
        boundary_policy=StructureBoundaryPolicy(), kwargs...)

Connect the canonical descriptor to bounded family opportunities. The world
must describe this search task and remain read-only for the state's lifetime.
Expected analysis refusals are unmanaged (`nothing`), never one shared family.
This adapter retains canonical keys only, not per-candidate typed evidence.
With scalar actions enabled, `scalar.named_input_policy` (default `:used`)
must equal the boundary policy's value; a mismatch throws before search.
"""
function StructurePolicy(world::StructureWorld;
        boundary_policy::StructureBoundaryPolicy=StructureBoundaryPolicy(), kwargs...)
    return StructurePolicy(_StructureClassifier(world, boundary_policy); kwargs...)
end

function _structure_key(policy::StructurePolicy, record::AgeSearchIndividual)
    return get!(policy.keys, record.id) do
        description = policy.describe(record.program)
        description === nothing && return nothing
        if description isa NamedTuple && hasproperty(description, :canonical_key)
            description.canonical_key === nothing && return nothing
            return (get(description, :view, :default), get(description, :version, 1),
                description.canonical_key)
        end
        return description
    end
end

function _live_family_counts(policy::StructurePolicy, state)
    counts = Dict{Any,Int}()
    for record in Iterators.flatten(state.populations)
        record.valid || continue
        key = _structure_key(policy, record)
        key === nothing || (counts[key] = get(counts, key, 0) + 1)
    end
    return counts
end

function _remember_family!(policy::StructurePolicy, key, family)
    policy.history_capacity == 0 && return nothing
    policy.history[key] = (; admissions=family.admissions,
        actions=family.development_actions, evaluations=family.development_evaluations,
        refused=family.refused_actions, seconds=family.development_seconds,
        touched=policy.clock)
    while length(policy.history) > policy.history_capacity
        oldest = first(keys(policy.history))
        for candidate in keys(policy.history)
            policy.history[candidate].touched < policy.history[oldest].touched &&
                (oldest = candidate)
        end
        delete!(policy.history, oldest)
    end
end

function _manage_family!(policy::StructurePolicy, state, key)
    key === nothing && return nothing
    haskey(policy.families, key) && return policy.families[key]
    if length(policy.families) >= policy.capacity
        live = _live_family_counts(policy, state)
        releasable = [k for k in keys(policy.families) if get(live, k, 0) == 0]
        if isempty(releasable)
            policy.overflow += 1
            return nothing
        end
        oldest = first(releasable)
        for candidate in releasable
            policy.families[candidate].touched < policy.families[oldest].touched &&
                (oldest = candidate)
        end
        _remember_family!(policy, oldest, policy.families[oldest])
        delete!(policy.families, oldest)
    end
    past = pop!(policy.history, key, nothing)
    family = StructureFamily(AgeSearchIndividual[],
        past === nothing ? 0 : past.admissions, 0, 0,
        past === nothing ? 0 : past.actions,
        past === nothing ? 0 : past.evaluations,
        past === nothing ? 0 : past.refused,
        past === nothing ? 0.0 : past.seconds, policy.clock)
    policy.families[key] = family
    return family
end

function _retain_representative!(policy::StructurePolicy, family, record)
    old = findfirst(r -> _same_program(r.program, record.program), family.representatives)
    old === nothing || deleteat!(family.representatives, old)
    push!(family.representatives, record)
    sort!(family.representatives; by=r -> (r.loss, r.complexity, r.id))
    if length(family.representatives) > policy.representative_capacity
        # Keep a real quality witness and recent alternative material. This
        # bounded heuristic makes no claim of optimal material diversity.
        best = first(family.representatives)
        rest = family.representatives[2:end]
        sort!(rest; by=r -> r.id, rev=true)
        family.representatives = [best; rest[1:(policy.representative_capacity - 1)]]
    end
end

function _observe_structure!(policy::StructurePolicy, state, record; admitted::Bool)
    policy.clock += 1
    key = _structure_key(policy, record)
    family = _manage_family!(policy, state, key)
    family === nothing && return nothing
    family.touched = policy.clock
    admitted && (family.admissions += 1)
    _retain_representative!(policy, family, record)
end

function _seed_structure_policy!(policy::StructurePolicy, state)
    policy.owner === state && return nothing
    policy.owner === nothing || throw(ArgumentError("a structure policy belongs to one age-search state"))
    policy.owner = state
    for record in Iterators.flatten(state.populations)
        record.valid && _observe_structure!(policy, state, record; admitted=true)
    end
end
_seed_structure_policy!(::Nothing, state) = nothing

function _prune_structure_keys!(policy::StructurePolicy, state)
    keep = Set(record.id for record in Iterators.flatten(state.populations))
    for family in values(policy.families), record in family.representatives
        push!(keep, record.id)
    end
    for id in collect(keys(policy.keys))
        id in keep || delete!(policy.keys, id)
    end
end
_prune_structure_keys!(::Nothing, state) = nothing

function _protected_family(policy::StructurePolicy, key)
    family = get(policy.families, key, nothing)
    return family !== nothing && family.admissions > 0 &&
        family.development_evaluations < policy.development_limit &&
        family.refused_actions < policy.refusal_limit
end

function Regression.age_replacement_index(policy::StructurePolicy, state::AgeSearchState,
        record::AgeSearchIndividual, island::Int)
    population = state.populations[island]
    child_key = _structure_key(policy, record)
    same = child_key === nothing ? Int[] :
        findall(r -> isequal(_structure_key(policy, r), child_key), population)
    if !isempty(same)
        return same[argmin([population[i].birth for i in same])]
    end
    counts = _live_family_counts(policy, state)
    eligible = findall(population) do old
        key = _structure_key(policy, old)
        key === nothing || get(counts, key, 0) > 1 || !_protected_family(policy, key)
    end
    isempty(eligible) && return nothing
    return eligible[argmin([population[i].birth for i in eligible])]
end

function Regression.age_observe_candidate!(policy::StructurePolicy, state::AgeSearchState,
        record::AgeSearchIndividual, island::Int, replaced, admitted::Bool)
    _observe_structure!(policy, state, record; admitted)
end

function _structure_candidates(policy::StructurePolicy, candidates)
    pool = AgeSearchIndividual[]
    ids = Set{UInt64}()
    for record in candidates
        record.id in ids && continue
        push!(pool, record)
        push!(ids, record.id)
    end
    for family in values(policy.families), record in family.representatives
        record.id in ids && continue
        push!(pool, record)
        push!(ids, record.id)
    end
    sort!(pool; by=record -> record.id)
    return pool
end

struct _StructureIslandSelection
    candidates::Vector{AgeSearchIndividual}
    family_groups::Vector{Vector{AgeSearchIndividual}}
    live_ids::Set{UInt64}
end

struct _StructureBatchSelection{P<:StructurePolicy} <: Regression.AbstractAgeSearchPolicy
    policy::P
    islands::Dict{Int,_StructureIslandSelection}
end

function _structure_family_groups(policy::StructurePolicy,
        candidates::Vector{AgeSearchIndividual})
    groups = Dict{Any,Vector{AgeSearchIndividual}}()
    for record in candidates
        key = _structure_key(policy, record)
        group = key === nothing ? (:unknown_record, record.id) : (:known, key)
        push!(get!(groups, group, AgeSearchIndividual[]), record)
    end
    group_keys = collect(keys(groups))
    sort!(group_keys; by=key -> minimum(record.id for record in groups[key]))
    return [groups[key] for key in group_keys]
end

function Regression._age_batch_selection(policy::StructurePolicy,
        state::AgeSearchState, valid_by_island)
    _seed_structure_policy!(policy, state)
    islands = Dict{Int,_StructureIslandSelection}()
    for (island, valid) in valid_by_island
        candidates = isempty(valid) ? valid : _structure_candidates(policy, valid)
        family_groups = !isempty(candidates) && policy.family_probability > 0 ?
            _structure_family_groups(policy, candidates) : Vector{AgeSearchIndividual}[]
        live_ids = Set(record.id for record in state.populations[island])
        islands[island] = _StructureIslandSelection(candidates, family_groups, live_ids)
    end
    return _StructureBatchSelection(policy, islands)
end

function Regression.age_parent_candidates(policy::StructurePolicy, state::AgeSearchState,
        island::Int, candidates::Vector{AgeSearchIndividual}, role::Symbol)
    _seed_structure_policy!(policy, state)
    return _structure_candidates(policy, candidates)
end

function Regression.age_parent_candidates(selection::_StructureBatchSelection,
        state::AgeSearchState, island::Int, candidates::Vector{AgeSearchIndividual},
        role::Symbol)
    return selection.islands[island].candidates
end

function Regression.age_select_parent(policy::StructurePolicy, state::AgeSearchState,
        island::Int, candidates::Vector{AgeSearchIndividual}, rng::AbstractRNG,
        tournament_size::Int, role::Symbol; eligible=nothing)
    _seed_structure_policy!(policy, state)
    live_ids = Set(record.id for record in state.populations[island])
    selected = if rand(rng) < policy.family_probability
        expanded = _structure_candidates(policy, candidates)
        pool = eligible === nothing ? expanded : filter(eligible, expanded)
        isempty(pool) && return nothing
        groups = Dict{Any,Vector{AgeSearchIndividual}}()
        for record in pool
            key = _structure_key(policy, record)
            # Unknown programs remain eligible as individual opportunities.
            group = key === nothing ? (:unknown_record, record.id) : (:known, key)
            push!(get!(groups, group, AgeSearchIndividual[]), record)
        end
        group_keys = collect(keys(groups))
        sort!(group_keys; by=key -> minimum(record.id for record in groups[key]))
        Regression._age_tournament(rng, groups[group_keys[rand(rng, eachindex(group_keys))]],
            tournament_size)
    else
        pool = eligible === nothing ? candidates : filter(eligible, candidates)
        isempty(pool) && return nothing
        Regression._age_tournament(rng, pool, tournament_size)
    end
    family = get(policy.families, _structure_key(policy, selected), nothing)
    if family !== nothing && role !== :development
        family.parent_selections += 1
        selected.id in live_ids || (family.ledger_parent_selections += 1)
    end
    return selected
end

function Regression.age_select_parent(selection::_StructureBatchSelection,
        state::AgeSearchState, island::Int, candidates::Vector{AgeSearchIndividual},
        rng::AbstractRNG, tournament_size::Int, role::Symbol; eligible=nothing)
    policy = selection.policy
    island_view = selection.islands[island]
    selected = if rand(rng) < policy.family_probability
        pool = eligible === nothing ? candidates : filter(eligible, candidates)
        isempty(pool) && return nothing
        groups = eligible === nothing ? island_view.family_groups :
            _structure_family_groups(policy, pool)
        Regression._age_tournament(rng,
            groups[rand(rng, eachindex(groups))], tournament_size)
    else
        pool = eligible === nothing ? candidates : filter(eligible, candidates)
        isempty(pool) && return nothing
        Regression._age_tournament(rng, pool, tournament_size)
    end
    family = get(policy.families, _structure_key(policy, selected), nothing)
    if family !== nothing && role !== :development
        family.parent_selections += 1
        selected.id in island_view.live_ids || (family.ledger_parent_selections += 1)
    end
    return selected
end

function _credit_development!(policy::StructurePolicy, state, source, evaluations,
        seconds, refused)
    key = _structure_key(policy, source)
    family = _manage_family!(policy, state, key)
    family === nothing && return nothing
    family.development_actions += 1
    # A retained session can outlive its source family's managed residence.
    # Reintroducing its accounting also needs a real witness for snapshots.
    isempty(family.representatives) && _retain_representative!(policy, family, source)
    family.development_evaluations += evaluations
    family.development_seconds += seconds
    refused && (family.refused_actions += 1)
end
_credit_development!(::Nothing, state, source, evaluations, seconds, refused) = nothing

"""Detached, compact family accounting; IDs are local to this descriptor policy."""
function structure_snapshot(policy::StructurePolicy)
    families = [(; key, representative_ids=[r.id for r in f.representatives],
        best_loss=minimum(r.loss for r in f.representatives), admissions=f.admissions,
        parent_selections=f.parent_selections, ledger_parent_selections=f.ledger_parent_selections,
        development_actions=f.development_actions,
        development_evaluations=f.development_evaluations, refused_actions=f.refused_actions,
        development_seconds=f.development_seconds)
        for (key, f) in policy.families if !isempty(f.representatives)]
    return (; families, retained_history=length(policy.history), overflow=policy.overflow)
end
structure_snapshot(::Nothing) = nothing

"""
    ContinuousSearchConfig(; max_actions=100, outer_weight=1, typed_weight=1,
        scalar_weight=2, constant_weight=1, max_seconds=Inf, max_migrants=3,
        path_attempts=8)

Finite serial action schedule, with complete search plus independently bounded
typed, scalar and numerical-constant actions. The constant action searches
program-local numeric payloads without changing syntax; it is separate from
scalar structural search and remains disabled unless enabled in its runner
options. Weights govern action count rather than CPU share; the ledger measures
actual end-to-end costs. Positive outer weight is required.
`max_seconds` is checked between actions and cannot interrupt a primitive or
bridge. Per-engine attempt/evaluation limits are supplied separately. No global
barrier requires every enabled layer to execute. Optional scalar sessions retain
local state between actions within one call. Multicore and cross-call resume
remain active, revisable extensions.
`source_exploration_probability` reserves local-source draws for uniformly
sampled, ID-deduplicated material; remaining draws use quality/family selection.
"""
Base.@kwdef struct ContinuousSearchConfig
    max_actions::Int = 100
    outer_weight::Float64 = 1.0
    typed_weight::Float64 = 1.0
    scalar_weight::Float64 = 2.0
    constant_weight::Float64 = 1.0
    max_seconds::Float64 = Inf
    max_migrants::Int = 3
    path_attempts::Int = 8
    source_exploration_probability::Float64 = 0.5
end

function _continuous_material_pool(state, policy)
    candidates = AgeSearchIndividual[]
    seen = Set{UInt64}()
    for record in Iterators.flatten((collect(values(state.archive)),
            collect(Iterators.flatten(state.populations)), state.recent_valid_offspring))
        record.valid && record.id ∉ seen || continue
        push!(candidates, record)
        push!(seen, record.id)
    end
    return policy === nothing ? candidates : _structure_candidates(policy, candidates)
end

function _continuous_source(state, policy, rng, exploration_probability; eligible=nothing)
    candidates = _continuous_material_pool(state, policy)
    eligible === nothing || filter!(eligible, candidates)
    isempty(candidates) && return nothing
    rand(rng) < exploration_probability && return candidates[rand(rng, eachindex(candidates))]
    if policy === nothing
        return Regression._age_tournament(rng, candidates,
            Int(get(state.reg.config, :tournament_size, 2)))
    end
    tournament_size = Int(get(state.reg.config, :tournament_size, 2))
    return Regression.age_select_parent(policy, state, 1, candidates, rng,
        tournament_size, :development; eligible)
end

function _constant_source_eligible(record::AgeSearchIndividual, source_filter)
    isempty(numeric_parameter_blocks(record.program)) && return false
    source_filter === nothing && return true
    applicable(source_filter, record) || throw(ArgumentError(
        "constant.source_filter must accept an AgeSearchIndividual",
    ))
    accepted = source_filter(record)
    accepted isa Bool || throw(ArgumentError(
        "constant.source_filter must return Bool",
    ))
    return accepted
end

function _continuous_local(state, source, kind, options, outer, counts, rng,
        path_attempts, scalar_config; session_options=nothing)
    reg = state.reg
    global_types = Dict{Symbol,TyLike}(reg.components.global_consts_ty)
    input_types = Dict{Symbol,TyLike}(reg.components.input_types)
    scorer = program -> score_candidate!(counts, program, reg, state.data;
        copy_tree=(kind !== :scalar))
    paths = _paths(source.program.tree.root)
    shuffle!(rng, paths)
    refusals = NamedTuple[]
    for path in Iterators.take(paths, path_attempts)
        result = if kind === :typed
            search_typed_hole(source.program, path, global_types, scorer;
                target_ty=reg.target_ty, input_types, rng,
                max_return=Int(options.max_return),
                max_nodes=Int(outer.max_nodes), max_depth=Int(outer.max_depth),
                attempts=Int(options.attempts), evaluations=Int(options.evaluations),
                population_size=Int(options.population_size))
        else
            region = prepare_scalar_region(source.program, reg.target_ty, global_types, path;
                input_types, named_input_policy=get(options, :named_input_policy, :used),
                max_nodes=Int(outer.max_nodes), max_depth=Int(outer.max_depth))
            if region isa Err
                region
            elseif scalar_complexity(region.expr) > scalar_config.max_complexity
                Err(Regression.ScalarRegionError(:scalar_size))
            elseif session_options !== nothing
                Regression.initialize_scalar_region_session(region, scorer;
                    config=scalar_config, rng,
                    max_return=Int(options.max_return),
                    material_return=Int(get(options, :material_return, 1)),
                    allow_zero_input=get(options, :allow_zero_input, false),
                    backend_options=session_options.backend_options)
            else
                search_scalar_region(region, scorer; config=scalar_config, rng,
                    max_return=Int(options.max_return),
                    material_return=Int(get(options, :material_return, 1)), stop_loss=nothing,
                    allow_zero_input=get(options, :allow_zero_input, false),
                    search_backend=get(options, :search_backend, search_scalar))
            end
        end
        if result isa Err
            push!(refusals, (; path=copy(path), reason=_refinement_summary(result).reason))
            continue
        end
        return (; result, path=copy(path), refusals)
    end
    return (; result=nothing, path=Int[], refusals)
end

function _continuous_session_options(scalar, scalar_config)
    options = get(scalar, :session, nothing)
    options === nothing && return nothing
    scalar.enabled || throw(ArgumentError("scalar.session requires enabled scalar search"))
    options isa NamedTuple || throw(ArgumentError("scalar.session must be a NamedTuple or nothing"))
    all(key -> key in (:rounds_per_action, :backend_options), keys(options)) ||
        throw(ArgumentError("unknown scalar.session option"))
    hasproperty(scalar, :search_backend) && throw(ArgumentError(
        "scalar.session uses the retained multi-island backend; omit search_backend"))
    rounds_per_action = get(options, :rounds_per_action, 1)
    rounds_per_action isa Integer && rounds_per_action > 0 ||
        throw(ArgumentError("scalar.session.rounds_per_action must be a positive integer"))
    backend_options = get(options, :backend_options, (;))
    backend_options isa NamedTuple || throw(ArgumentError("backend_options must be a NamedTuple"))
    all(key -> key in (:island_count, :rounds, :migrant_count, :top_migrant_count,
        :ncycles_per_iteration, :frequency_window, :top_migration_rate, :donor_limit,
        :hof_donor_policy, :round_observer), keys(backend_options)) ||
        throw(ArgumentError("unknown or reserved scalar session backend option"))
    scalar_config.evolution_strategy === :regularized ||
        throw(ArgumentError("scalar.session requires evolution_strategy=:regularized"))
    for key in (:island_count, :rounds, :ncycles_per_iteration, :frequency_window)
        value = get(backend_options, key, 1)
        value isa Integer && value > 0 || throw(ArgumentError("$key must be a positive integer"))
    end
    islands = get(backend_options, :island_count, 2)
    scalar_config.max_evaluations ÷ islands >= scalar_config.population_size ||
        throw(ArgumentError("scalar session budget must initialize every island"))
    return (; rounds_per_action=Int(rounds_per_action), backend_options)
end

const _EMPTY_SESSION_COUNTS = (objective_calls=0, structural_evaluations=0,
	constant_refinement_evaluations=0, total_evaluations=0,
	baseline_objective_calls=0)

function _continuous_session_summary(slot, previous; event)
    progress = Regression.scalar_region_progress(slot.session)
    delta = map(-, progress.cumulative, previous)
    return (; id=slot.id, source_id=slot.source.id, path=copy(slot.path), event,
        progress..., delta)
end

"""
    run_continuous_search!(state; rng, config=ContinuousSearchConfig(),
        outer, typed, scalar, constant=(enabled=false,), policy=nothing,
        on_action=nothing)

Compose complete-program evolution and immediately published bounded local
refinement actions over one stable task. A `StructurePolicy` optionally changes
opportunity and survival. All migrants are validated and rescored in this task;
family scores are never substituted for individual scores. Local development
cost belongs to the selected source family even when the result changes family.
Callbacks observe a compact action record synchronously; mutable state/policy
must remain exclusively owned. By default local calls restart their bounded backend state.
Optional `scalar.search_backend` follows `search_scalar_region`'s contextual
backend contract, including the supplied objective, RNG, initial population and
budgets. It defaults to `search_scalar`; family membership never permits sharing
local backend state or scores between different contexts.

Opt into one retained scalar session with `scalar.session=(rounds_per_action=1,
backend_options=(island_count=2, rounds=4))`. `scalar.config` budgets cover the
whole session, not each action. The first action selects an ordinary source and
eligible hole; later scalar actions advance that same owned context. Outer,
typed and constant actions still publish independently. They never retarget or
reset the scalar session. Each publication is an owned full program and passes
ordinary migrant rescoring. All task data, primitive values and scoring settings
must remain stable. A completed session releases the slot for a new selection.
`search_backend` cannot be combined with this option. Each ledger's
`report.session.delta` counts only that action; `cumulative` and ordinary bridge
diagnostics cover the session so far. Objective counters include initialization
baselines and hot materialization refusals; development calls count actual whole
program scoring, and migrant rescoring is charged separately to the outer engine.
At the run's action/time limit an unfinished session is discarded and explicitly
reported in `scalar_sessions.unfinished`; a subsequent call starts afresh.
This is bounded in-memory scheduling, not cross-call or disk resume.

`scalar.allow_zero_input` defaults to `false`; set it to `true` to run a scalar
session over a region with no dependencies. A constant action is separate:
`constant=(enabled=true, strategy, budget)` requires an
`AbstractConstantSearch` strategy and `ConstantSearchBudget`. It first filters
the source pool to programs with `numeric_parameter_blocks`, then applies the
usual exploration/family selection. Optional `constant.source_filter` defaults
to `nothing`; when supplied, it receives each numerically eligible
`AgeSearchIndividual` and must return `Bool`. Family representatives added
during selection pass through the same
filter. When no source remains, the refusal reason is `:no_numeric_blocks`
without a filter and `:no_eligible_constant_source` when one is supplied. It
returns at most the search's best improved program through ordinary migrant
validation, rescoring, and structure classification. Enablement flags do not
alter `numeric_parameter_blocks` or a structural description; leaving a
payload out of one optimizer is never a new structure on its own.
"""
function run_continuous_search!(state::AgeSearchState; rng::AbstractRNG,
        config::ContinuousSearchConfig=ContinuousSearchConfig(), outer, typed,
        scalar, constant=(enabled=false,), policy=nothing, on_action=nothing)
    constant isa NamedTuple && hasproperty(constant, :enabled) ||
        throw(ArgumentError("constant options must include enabled"))
    constant.enabled isa Bool ||
        throw(ArgumentError("constant.enabled must be Bool"))
    if constant.enabled
        hasproperty(constant, :strategy) && hasproperty(constant, :budget) ||
            throw(ArgumentError("enabled constant search requires strategy and budget"))
        constant.strategy isa Regression.AbstractConstantSearch ||
            throw(ArgumentError("constant.strategy must implement AbstractConstantSearch"))
        constant.budget isa Regression.ConstantSearchBudget ||
            throw(ArgumentError("constant.budget must be ConstantSearchBudget"))
    end
    if policy isa StructurePolicy && policy.describe isa _StructureClassifier && scalar.enabled
        get(scalar, :named_input_policy, :used) === policy.describe.boundary_policy.named_input_policy ||
            throw(ArgumentError("descriptor and scalar action named-input policies must match"))
    end
    config.max_actions >= 0 && config.max_migrants >= 0 && config.path_attempts > 0 ||
        throw(ArgumentError("action/migrant limits must be nonnegative; path attempts positive"))
    config.max_seconds > 0 || throw(ArgumentError("max_seconds must be positive"))
    0 <= config.source_exploration_probability <= 1 ||
        throw(ArgumentError("source_exploration_probability must lie in [0,1]"))
    config.outer_weight > 0 || throw(ArgumentError("outer exploration weight must be positive"))
    all(w -> isfinite(w) && w >= 0,
        (config.outer_weight, config.typed_weight, config.scalar_weight,
            config.constant_weight)) ||
        throw(ArgumentError("action weights must be finite and nonnegative"))
    kinds = Symbol[:outer]
    weights = Float64[config.outer_weight]
    typed.enabled && config.typed_weight > 0 &&
        (push!(kinds, :typed); push!(weights, config.typed_weight))
    scalar.enabled && config.scalar_weight > 0 &&
        (push!(kinds, :scalar); push!(weights, config.scalar_weight))
    constant.enabled && config.constant_weight > 0 &&
        (push!(kinds, :constant); push!(weights, config.constant_weight))
    # Deterministic weighted deficit scheduling gives positive-weight outer
    # exploration opportunities without a stochastic starvation interval.
    scalar_config = scalar.enabled ? _scalar_config(scalar) : nothing
    session_options = _continuous_session_options(scalar, scalar_config)
    deficits = zeros(length(kinds))
    _seed_structure_policy!(policy, state)
    active_session = nothing
    sessions_started = 0
    sessions_completed = 0
    termination = :max_actions
    typed_counts, scalar_counts, constant_counts =
        ObjectiveCounts(), ObjectiveCounts(), ObjectiveCounts()
    ledger = NamedTuple[]
    started = time_ns()
    for action in 1:config.max_actions
        if (time_ns() - started) / 1e9 >= config.max_seconds
            termination = :max_seconds
            break
        end
        deficits .+= weights
        selected = argmax(deficits)
        deficits[selected] -= sum(weights)
        kind = kinds[selected]
        action_started = time_ns()
        source_id = nothing
        path = Int[]
        report = nothing
        incoming = 0
        returned = 0
        refused = false
        refusals = NamedTuple[]
        local_calls = 0
        before_objectives = state.counters.objective_calls + typed_counts.calls +
            scalar_counts.calls + constant_counts.calls
        if kind === :outer
            report = advance_age_search!(state;
                max_attempts=Int(outer.attempts), max_evaluations=Int(outer.evaluations), policy)
            refused = report isa Err
        elseif kind === :constant
            source_filter = get(constant, :source_filter, nothing)
            source = _continuous_source(state, policy, rng,
                config.source_exploration_probability;
                eligible=record -> _constant_source_eligible(record, source_filter))
            if source === nothing
                refused = true
                reason = if source_filter === nothing
                    :no_numeric_blocks
                else
                    :no_eligible_constant_source
                end
                refusals = [(; reason)]
                report = (; status=:refused, reason,
                    constant_calls=0, constant_stats=nothing)
            else
                source_id = source.id
                before_calls = constant_counts.calls
                result = Regression.search_constants(constant.strategy,
                    source.program,
                    program -> score_candidate!(constant_counts, program,
                        state.reg, state.data);
                    rng,
                    budget=constant.budget)
                local_calls = constant_counts.calls - before_calls
                refused = result.status in (:ineligible, :invalid_source)
                refused && push!(refusals, (; reason=result.status))
                _credit_development!(policy, state, source, local_calls,
                    (time_ns() - action_started) / 1e9, refused)
                report = (; status=result.status, loss=result.loss,
                    constant_calls=local_calls, constant_stats=result.stats)
                if result.status === :improved
                    returned = 1
                    migration = inject_age_migrants!(state,
                        [AgeSearchMigration(result.program, (source.id,))];
                        population=mod1(action, length(state.populations)),
                        max_migrants=config.max_migrants, policy)
                    incoming = migration.accepted
                    report = merge(report, (; migration))
                end
            end
        else
            retained = kind === :scalar && session_options !== nothing
            continuing = retained && active_session !== nothing
            source = continuing ? active_session.source :
                _continuous_source(state, policy, rng, config.source_exploration_probability)
            if source === nothing
                refused = true
            else
                source_id = source.id
                counts = kind === :typed ? typed_counts : scalar_counts
                before_calls = counts.calls
                options = kind === :typed ? typed : scalar
                local_result = if continuing
                    (; result=active_session.session, path=copy(active_session.path),
                        refusals=NamedTuple[])
                else
                    _continuous_local(state, source, kind, options,
                        outer, counts, rng, config.path_attempts, scalar_config;
                        session_options=retained ? session_options : nothing)
                end
                path, refusals = local_result.path, local_result.refusals
                result = local_result.result
                session_report = nothing
                if retained && result !== nothing
                    previous = continuing ?
                        Regression.scalar_region_progress(result).cumulative : _EMPTY_SESSION_COUNTS
                    if !continuing
                        sessions_started += 1
                        active_session = (; session=result, source, path=copy(path),
                            id=sessions_started)
                    end
                    Regression.advance_scalar_region!(result;
                        max_rounds=session_options.rounds_per_action)
                    session_report = _continuous_session_summary(active_session, previous;
                        event=continuing ? :resumed : :started)
                    result = Regression.scalar_region_result(result)
                    if session_report.completed
                        sessions_completed += 1
                        active_session = nothing
                    end
                end
                local_calls = counts.calls - before_calls
                refused = result === nothing || result isa Err
                _credit_development!(policy, state, source, local_calls,
                    (time_ns() - action_started) / 1e9, refused)
                if result !== nothing
                    report = _refinement_summary(result)
                    session_report === nothing || (report = merge(report, (; session=session_report)))
                    result isa Err && push!(refusals, (; path=copy(path), reason=report.reason))
                end
                if !refused
                    candidates = result.candidates
                    returned = length(candidates)
                    migration = inject_age_migrants!(state,
                        [AgeSearchMigration(c.program, (source.id,)) for c in candidates];
                        population=mod1(action, length(state.populations)),
                        max_migrants=config.max_migrants, policy)
                    incoming = migration.accepted
                    report = merge(report, (; migration))
                end
            end
        end
        _prune_structure_keys!(policy, state)
        entry = (; action, kind, source_id, path, returned, incoming, refused,
            refusals, development_objective_calls=local_calls,
            objective_calls=state.counters.objective_calls + typed_counts.calls +
                scalar_counts.calls + constant_counts.calls - before_objectives, report,
            elapsed_seconds=(time_ns() - action_started) / 1e9,
            best_loss=state.best === nothing ? Inf : state.best.loss)
        push!(ledger, entry)
        on_action === nothing || on_action(entry)
    end
    unfinished = active_session === nothing ? nothing : merge(
        _continuous_session_summary(active_session,
            Regression.scalar_region_progress(active_session.session).cumulative;
            event=:run_end), (; status=:discarded_at_run_end))
    return (; state, best=state.best, ledger, structure=structure_snapshot(policy),
        termination,
        scalar_sessions=(; started=sessions_started, completed=sessions_completed, unfinished),
        counts=(; outer=state.counters, typed=typed_counts, scalar=scalar_counts,
            constant=constant_counts),
        elapsed_seconds=(time_ns() - started) / 1e9)
end
