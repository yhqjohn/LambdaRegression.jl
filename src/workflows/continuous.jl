"""One managed family's real witnesses and development history; never a shared score."""
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
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
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
    @test result.counts.scalar.calls > 0
    @test sum(e.development_objective_calls for e in observed) == result.counts.scalar.calls
    @test sum(f.development_evaluations for f in values(policy.families)) == result.counts.scalar.calls
    @test any(e -> e.kind === :scalar && !e.refused && e.returned > 0, observed)
    @test all(e -> e.elapsed_seconds >= 0, observed)
    @test length(policy.keys) <= length(state.populations[1]) +
        policy.capacity * policy.representative_capacity
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

function Regression.age_parent_candidates(policy::StructurePolicy, state::AgeSearchState,
        island::Int, candidates::Vector{AgeSearchIndividual}, role::Symbol)
    _seed_structure_policy!(policy, state)
    return _structure_candidates(policy, candidates)
end

function Regression.age_select_parent(policy::StructurePolicy, state::AgeSearchState,
        island::Int, candidates::Vector{AgeSearchIndividual}, rng::AbstractRNG,
        tournament_size::Int, role::Symbol)
    _seed_structure_policy!(policy, state)
    live_ids = Set(record.id for record in state.populations[island])
    selected = if rand(rng) < policy.family_probability
        pool = _structure_candidates(policy, candidates)
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
        Regression._age_tournament(rng, candidates, tournament_size)
    end
    family = get(policy.families, _structure_key(policy, selected), nothing)
    if family !== nothing && role !== :development
        family.parent_selections += 1
        selected.id in live_ids || (family.ledger_parent_selections += 1)
    end
    return selected
end

function _credit_development!(policy::StructurePolicy, state, source, evaluations,
        seconds, refused)
    key = _structure_key(policy, source)
    family = _manage_family!(policy, state, key)
    family === nothing && return nothing
    family.development_actions += 1
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
        scalar_weight=2, max_seconds=Inf, max_migrants=3, path_attempts=8)

Finite serial action schedule, with complete search plus independently bounded
typed/scalar actions. Weights govern action count rather than CPU share; the
ledger measures actual end-to-end costs. Positive outer weight is required.
`max_seconds` is checked between actions and cannot interrupt a primitive or
bridge. Per-engine attempt/evaluation limits are supplied separately. No global
barrier requires every enabled layer to execute. Multicore and exact local-state
resume are active, revisable extensions, not supported by this first entry.
`source_exploration_probability` reserves local-source draws for uniformly
sampled, ID-deduplicated material; remaining draws use quality/family selection.
"""
Base.@kwdef struct ContinuousSearchConfig
    max_actions::Int = 100
    outer_weight::Float64 = 1.0
    typed_weight::Float64 = 1.0
    scalar_weight::Float64 = 2.0
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

function _continuous_source(state, policy, rng, exploration_probability)
    candidates = _continuous_material_pool(state, policy)
    isempty(candidates) && return nothing
    rand(rng) < exploration_probability && return candidates[rand(rng, eachindex(candidates))]
    if policy === nothing
        return Regression._age_tournament(rng, candidates,
            Int(get(state.reg.config, :tournament_size, 2)))
    end
    return Regression.age_select_parent(policy, state, 1, candidates, rng,
        Int(get(state.reg.config, :tournament_size, 2)), :development)
end

function _continuous_local(state, source, kind, options, outer, counts, rng,
        path_attempts, scalar_config)
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
            else
                search_scalar_region(region, scorer; config=scalar_config, rng,
                    max_return=Int(options.max_return),
                    material_return=Int(get(options, :material_return, 1)), stop_loss=nothing)
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

"""
    run_continuous_search!(state; rng, config=ContinuousSearchConfig(),
        outer, typed, scalar, policy=nothing, on_action=nothing)

Compose complete-program evolution and immediately published bounded local
refinement actions over one stable task. A `StructurePolicy` optionally changes
opportunity and survival. All migrants are validated and rescored in this task;
family scores are never substituted for individual scores. Local development
cost belongs to the selected source family even when the result changes family.
Callbacks observe a compact action record synchronously; mutable state/policy
must remain exclusively owned. Local calls restart their bounded backend state.
"""
function run_continuous_search!(state::AgeSearchState; rng::AbstractRNG,
        config::ContinuousSearchConfig=ContinuousSearchConfig(), outer, typed,
        scalar, policy=nothing, on_action=nothing)
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
        (config.outer_weight, config.typed_weight, config.scalar_weight)) ||
        throw(ArgumentError("action weights must be finite and nonnegative"))
    kinds = Symbol[:outer]
    weights = Float64[config.outer_weight]
    typed.enabled && config.typed_weight > 0 &&
        (push!(kinds, :typed); push!(weights, config.typed_weight))
    scalar.enabled && config.scalar_weight > 0 &&
        (push!(kinds, :scalar); push!(weights, config.scalar_weight))
    # Deterministic weighted deficit scheduling gives positive-weight outer
    # exploration opportunities without a stochastic starvation interval.
    deficits = zeros(length(kinds))
    _seed_structure_policy!(policy, state)
    scalar_config = scalar.enabled ? _scalar_config(scalar) : nothing
    typed_counts, scalar_counts = ObjectiveCounts(), ObjectiveCounts()
    ledger = NamedTuple[]
    started = time_ns()
    for action in 1:config.max_actions
        (time_ns() - started) / 1e9 >= config.max_seconds && break
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
        before_objectives = state.counters.objective_calls + typed_counts.calls + scalar_counts.calls
        if kind === :outer
            report = advance_age_search!(state;
                max_attempts=Int(outer.attempts), max_evaluations=Int(outer.evaluations), policy)
            refused = report isa Err
        else
            source = _continuous_source(state, policy, rng, config.source_exploration_probability)
            if source === nothing
                refused = true
            else
                source_id = source.id
                counts = kind === :typed ? typed_counts : scalar_counts
                before_calls = counts.calls
                options = kind === :typed ? typed : scalar
                local_result = _continuous_local(state, source, kind, options,
                    outer, counts, rng, config.path_attempts, scalar_config)
                path, refusals = local_result.path, local_result.refusals
                local_calls = counts.calls - before_calls
                refused = local_result.result === nothing
                _credit_development!(policy, state, source, local_calls,
                    (time_ns() - action_started) / 1e9, refused)
                if !refused
                    report = _refinement_summary(local_result.result)
                    candidates = local_result.result.candidates
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
                scalar_counts.calls - before_objectives, report,
            elapsed_seconds=(time_ns() - action_started) / 1e9,
            best_loss=state.best === nothing ? Inf : state.best.loss)
        push!(ledger, entry)
        on_action === nothing || on_action(entry)
    end
    return (; state, best=state.best, ledger, structure=structure_snapshot(policy),
        counts=(; outer=state.counters, typed=typed_counts, scalar=scalar_counts),
        elapsed_seconds=(time_ns() - started) / 1e9)
end
