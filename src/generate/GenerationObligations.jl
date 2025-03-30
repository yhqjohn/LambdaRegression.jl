# Finite syntax evidence crossed with positive occurrence obligations. A mask
# states what is guaranteed, never what is absent. Forbidden bindings remain
# an independent lexical constraint. This layer has no extension/filter proof.
const _ObligationMask = Union{UInt,BigInt}
const _ObligationFactKey = Tuple{Int,_ObligationMask}

# Opt-in, task-local diagnostics for associating sampled derivations with the
# joint certificate edges that produced them. The normal path does no tracing.
const _OBLIGATION_EDGE_DIAGNOSTIC_KEY = :_lambda_regression_obligation_edge_diagnostic

function _notify_obligation_edge_diagnostic(observation)
    observer = get(task_local_storage(), _OBLIGATION_EDGE_DIAGNOSTIC_KEY, nothing)
    observer === nothing || observer(observation)
    return nothing
end

struct _ObligationEdge
    parent::Int
    plan::_EvidencePlan
    root_binder::Int
    children::Vector{Int}
end

mutable struct _GenerationObligations
    world::_GenerationEvidence
    binders::Vector{Int}
    binder_types::Vector{Int}
    calls::BitVector
    full_mask::_ObligationMask
    fact_ids::Dict{_ObligationFactKey,Int}
    facts::Vector{_ObligationFactKey}
    costs::Vector{Int}
    certificates::Vector{Int}
    edges::Vector{_ObligationEdge}
    options::Vector{Vector{Int}}
    dependents::Vector{Vector{Int}}
    expanded::BitVector
    pending::Vector{Int}
    pending_cursor::Int
    updates::Vector{Int}
    complete::Bool
end

_obligation_zero(mask::_ObligationMask) = mask isa UInt ? UInt(0) : BigInt(0)
_obligation_bit(mask::_ObligationMask, slot::Int) = one(mask) << (slot - 1)
_obligation_full_mask(count::Int) = count < 8 * sizeof(UInt) ?
    (UInt(1) << count) - UInt(1) : (BigInt(1) << count) - 1

function _new_generation_obligations(world::_GenerationEvidence, query::GenerationQuery)
    # Calls imply use, but passing a function as a value does not imply calls.
    indices = sort!(unique!(Int[r.index for r in query.requirements
        if r isa Union{UsesScopeBinder,CallsScopeBinder}]))
    calls = BitVector(any(r -> r isa CallsScopeBinder && r.index == index,
        query.requirements) for index in indices)
    types = Int[_evidence_type_id(world, query.scope_tys[index + 1]) for index in indices]
    return _GenerationObligations(world, indices, types, calls,
        _obligation_full_mask(length(indices)), Dict{_ObligationFactKey,Int}(),
        _ObligationFactKey[], Int[], Int[], _ObligationEdge[], Vector{Int}[],
        Vector{Int}[], BitVector(), Int[], 1, Int[], false)
end

function _obligation_fact!(graph::_GenerationObligations, ordinary::Int, mask::_ObligationMask)
    return get!(graph.fact_ids, (ordinary, mask)) do
        push!(graph.facts, (ordinary, mask))
        push!(graph.costs, typemax(Int))
        push!(graph.certificates, 0)
        push!(graph.options, Int[])
        push!(graph.dependents, Int[])
        push!(graph.expanded, false)
        graph.complete = false
        id = length(graph.facts)
        push!(graph.pending, id)
        id
    end
end

"""Return a replay-admissible edge cost, or `nothing` for an unproved edge.

`options` contains candidate allocations, not a certified menu. Nonzero masks
need a joint certificate; zero masks deliberately borrow ordinary evidence.
Sentinel costs and addition overflow cannot become budget-admissible values.
This shared check is used by saturation and both sampling consumers.
"""
function _obligation_edge_cost(graph::_GenerationObligations, edge::_ObligationEdge)
    cost = edge.plan.charge
    for child in edge.children
        child_cost = graph.costs[child]
        child_cost == typemax(Int) && return nothing
        ordinary, mask = graph.facts[child]
        if mask == _obligation_zero(mask)
            graph.world.certificates[ordinary] != 0 || return nothing
            graph.world.costs[ordinary] != typemax(Int) || return nothing
        else
            graph.certificates[child] != 0 || return nothing
        end
        child_cost > typemax(Int) - cost && return nothing
        cost += child_cost
    end
    # Raw descriptors retained by partial preparation must obey the same fact
    # ceiling as a complete ordinary menu, even in a larger caller balance.
    budget = graph.world.facts[graph.facts[edge.parent][1]][5]
    return cost == typemax(Int) || (budget >= 0 && cost > budget) ? nothing : cost
end

function _obligation_relax!(graph::_GenerationObligations, edge_id::Int)
    edge = graph.edges[edge_id]
    cost = _obligation_edge_cost(graph, edge)
    cost === nothing && return
    cost < graph.costs[edge.parent] || return
    graph.costs[edge.parent] = cost
    graph.certificates[edge.parent] = edge_id
    push!(graph.updates, edge.parent)
end

function _obligation_edge!(graph::_GenerationObligations, parent::Int,
    plan::_EvidencePlan, root_binder::Int, children::Vector{Int})
    push!(graph.edges, _ObligationEdge(parent, plan, root_binder, children))
    edge_id = length(graph.edges)
    push!(graph.options[parent], edge_id)
    for child in children
        push!(graph.dependents[child], edge_id)
    end
    _obligation_relax!(graph, edge_id)
end

function _obligation_allocations!(emit, mask::_ObligationMask, count::Int)
    zero = _obligation_zero(mask)
    if count == 0
        mask == zero && emit(_ObligationMask[])
        return
    end
    bits = _ObligationMask[]
    remaining = mask
    while remaining != zero
        bit = remaining & -remaining
        push!(bits, bit)
        remaining &= ~bit
    end
    assignment = _ObligationMask[zero for _ in 1:count]
    # This enumerates finite obligation placements, not combinations of terms.
    # k obligations may require count^k placements: no hidden cap discards them.
    function visit(index::Int)
        if index > length(bits)
            emit(assignment)
            return
        end
        bit = bits[index]
        for child in eachindex(assignment)
            assignment[child] |= bit
            visit(index + 1)
            assignment[child] &= ~bit
        end
    end
    visit(1)
end

function _obligation_sync_ordinary!(graph::_GenerationObligations, ordinary::Int)
    fact = get(graph.fact_ids, (ordinary, _obligation_zero(graph.full_mask)), 0)
    fact == 0 && return
    cost = graph.world.costs[ordinary]
    graph.world.certificates[ordinary] == 0 && return
    cost < graph.costs[fact] || return
    graph.costs[fact] = cost
    push!(graph.updates, fact)
end

function _obligation_refresh_zero!(graph::_GenerationObligations)
    for (ordinary, mask) in graph.facts
        mask == _obligation_zero(mask) && _obligation_sync_ordinary!(graph, ordinary)
    end
end

function _obligation_drain_updates!(graph::_GenerationObligations)
    index = 1
    while index <= length(graph.updates)
        for edge in graph.dependents[graph.updates[index]]
            _obligation_relax!(graph, edge)
        end
        index += 1
    end
    empty!(graph.updates)
end

function _obligation_expand!(graph::_GenerationObligations, relation::CoreRelation,
    fact::Int; certificate_only::Bool=false)
    graph.expanded[fact] && return
    graph.expanded[fact] = true
    ordinary, mask = graph.facts[fact]
    if mask == _obligation_zero(mask)
        # A zero guarantee may incidentally use/call a designated binder. This
        # is safe only because forbidden occurrences are separate constraints.
        _obligation_sync_ordinary!(graph, ordinary)
        return
    end
    _, type_id, _, _, _ = graph.world.facts[ordinary]
    plans = if certificate_only
        # Register every syntactic descriptor, not just the chosen ordinary
        # witness. Proof work remains in the resumable agendas below.
        (_EvidencePlan(step.kind, step.value,
            Int[_evidence_fact!(graph.world, child...) for child in step.children],
            step.charge, step.weight)
            for step in _evidence_steps!(graph.world, relation, ordinary))
    else
        _evidence_menu!(graph.world, relation, ordinary)
    end
    for plan in plans
        if plan.kind == :variable
            for slot in eachindex(graph.binders)
                graph.calls[slot] && continue
                graph.binder_types[slot] == type_id || continue
                bit = _obligation_bit(mask, slot)
                (mask & ~bit) == 0 || continue
                _obligation_edge!(graph, fact, plan, graph.binders[slot], Int[])
            end
            continue
        end
        selectors = Tuple{Int,_ObligationMask}[(-1, _obligation_zero(mask))]
        if plan.kind == :scope_application
            head_type = _evidence_type_id(graph.world, plan.value)
            for slot in eachindex(graph.binders)
                graph.binder_types[slot] == head_type || continue
                # Nonempty application of this exact identity satisfies both
                # its occurrence and call requirements, normalised to one bit.
                push!(selectors, (graph.binders[slot], _obligation_bit(mask, slot)))
            end
        end
        for (binder, guarantee) in selectors
            remaining = mask & ~guarantee
            _obligation_allocations!(remaining, length(plan.children)) do assignments
                children = Int[_obligation_fact!(graph, child, assignments[index])
                    for (index, child) in enumerate(plan.children)]
                _obligation_edge!(graph, fact, plan, binder, children)
            end
        end
    end
end

function _obligation_saturate!(graph::_GenerationObligations, relation::CoreRelation)
    _evidence_saturate!(graph.world, relation)
    _obligation_refresh_zero!(graph)
    while graph.pending_cursor <= length(graph.pending)
        fact = graph.pending[graph.pending_cursor]
        graph.pending_cursor += 1
        _obligation_expand!(graph, relation, fact)
    end
    empty!(graph.pending)
    graph.pending_cursor = 1
    _obligation_drain_updates!(graph)
    graph.complete = true
    return graph
end

function _obligation_prepare_certificate!(graph::_GenerationObligations,
    relation::CoreRelation, root::Int, budget::Int)
    world = graph.world
    _obligation_refresh_zero!(graph)
    while true
        # Propagate all known improvements before charging a certificate.
        # An unresolved cycle is pending, never a negative visiting result.
        _evidence_drain_updates!(world, graph)
        _obligation_drain_updates!(graph)
        graph.certificates[root] != 0 && graph.costs[root] != typemax(Int) &&
            graph.costs[root] <= budget && return graph
        joint_pending = graph.pending_cursor <= length(graph.pending)
        ordinary_pending = world.pending_cursor <= length(world.pending)
        if !joint_pending && !ordinary_pending
            empty!(graph.pending)
            graph.pending_cursor = 1
            empty!(world.pending)
            world.pending_cursor = 1
            graph.complete = true
            return graph
        end
        if joint_pending
            fact = graph.pending[graph.pending_cursor]
            graph.pending_cursor += 1
            _obligation_expand!(graph, relation, fact; certificate_only=true)
        end
        if ordinary_pending
            fact = world.pending[world.pending_cursor]
            world.pending_cursor += 1
            _evidence_expand!(world, relation, fact)
        end
    end
end

"""Prepare joint core occurrence evidence, retaining exact lexical identities.

The request supplies closed target/scope types, capability masks, depth/node
bounds and positive `UsesScopeBinder`/`CallsScopeBinder` requirements. Forbidden
bindings remain independent. A finite ordinary grammar is crossed with masks
of guaranteed occurrences; costs certify the actual child allocation, rather
than separately checking whether each occurrence is possible.

With `certificate_only=true`, reviewed built-in providers may stop at the first
budget-feasible certificate. Its cost is an upper bound, NOT a minimum. Pending
work stays resumable; a later full consumer still obtains the complete menu.
Unknown providers retain full preparation and its global policy preflight.

`source_depth >= 0` requests the existing anchored-source grammar, crossed
with the same exact occurrences. Its preparation depth is independent of the
parent's ordinary depth: unbounded finite facts for monomorphic contexts, or
the inherited node ceiling for scheme-layered contexts. The source allowance
remains in the ordinary fact identity and is never renewed at child crossings.

Returns prepared core evidence, `nothing` when that core has no admissible
certificate, or `_UnsupportedEvidenceSampling` for an unmigrated catalogue,
planner or core-affecting requirement. A core-negative result is NOT a negative
result for safe-list, guarded or provider extension construction. Other finite
post-filters are still the caller's responsibility.
"""
function _prepare_generation_obligations(relation::CoreRelation, query::GenerationQuery;
    certificate_only::Bool=false, source_depth::Int=-1)
    relation, query = _relation_for_query(relation, query)
    _finite_evidence_request_supported(relation, query) || return _UnsupportedEvidenceSampling()
    relation.argument_planner isa DefaultArgumentPlanner || return _UnsupportedEvidenceSampling()
    any(query.requirements) do requirement
        requirement isa Union{UsesScopeBinder,CallsScopeBinder,ForbidsScopeBinder} && return false
        generation_requirement_affects_core_search(requirement)
    end && return _UnsupportedEvidenceSampling()
    scope = _scope_type_vector(query.scope_tys)
    forbidden = _forbidden_scope_binders(query)
    indices = _required_scope_binder_indices(query, length(scope))
    isempty(indices) && return _UnsupportedEvidenceSampling()
    any(index -> index in forbidden, indices) && return nothing
    world = _finite_evidence_world(relation, query.target_ty, scope; sampling=true)
    _evidence_has_local_argument_policy(world, relation) && return _UnsupportedEvidenceSampling()
    certificate_only &= all(_evidence_early_certificate_supported, relation.providers)
    available = Int[_evidence_type_id(world, ty) for (slot, ty) in enumerate(scope)
        if slot - 1 ∉ forbidden]
    context = _evidence_context!(world, available)
    preparation_depth = source_depth < 0 ? query.depth_left :
        (_evidence_has_schemes(relation) ? query.node_budget : -1)
    ordinary = _evidence_fact!(world, context,
        _evidence_type_id(world, query.target_ty), preparation_depth, source_depth,
        world.layered ? query.node_budget : -1)
    if !certificate_only
        _evidence_saturate!(world, relation)
        world.unsupported_sampling && return _UnsupportedEvidenceSampling()
        world.certificates[ordinary] == 0 && return nothing
    end
    # This world already owns the checked context/policy version. Reuse only
    # the same concrete obligation request, not just its type-availability set:
    # same-type binders, call versus use, and forbidden slots stay distinct.
    # Store deterministic graph preparation, never a sampled term or root mode.
    calls = Tuple(_required_scope_call_indices(query, length(scope)))
    # The ordinary fact already carries a scheme-layered node ceiling. In the
    # monomorphic graph costs do not depend on the caller's available balance;
    # apply that balance below, not as a second identity for the same proof.
    key = _StructuralKey((ordinary, Tuple(_evidence_type_id(world, ty) for ty in scope),
        forbidden, Tuple(indices), calls))
    graph, root = get!(world.obligations, key) do
        graph = _new_generation_obligations(world, query)
        root = _obligation_fact!(graph, ordinary, graph.full_mask)
        (graph, root)
    end
    if certificate_only
        _obligation_prepare_certificate!(graph, relation, root, query.node_budget)
    else
        _obligation_saturate!(graph, relation)
    end
    world.unsupported_sampling && return _UnsupportedEvidenceSampling()
    graph.costs[root] == typemax(Int) && return nothing
    graph.costs[root] > query.node_budget && return nothing
    return (; graph, root, scope, forbidden, relation)
end

@testitem "CanonicalGeneration joint sources preserve mapper occurrences and independent depth" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, TyScheme,
        FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    input = TyArray(TyNat(4), FloatTy)
    output = TyArray(TyNat(4), BoolTy)
    predicate = TyFun(FloatTy, BoolTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
        global_consts=Dict{Symbol,Any}(:predicate => iszero),
        global_consts_ty=Dict{Symbol,Any}(:predicate => predicate))
    scope = Ty[FloatTy, FloatTy, input]
    query = CG.GenerationQuery(output; scope_tys=scope, depth_left=0,
        node_budget=5, requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    # The array has no direct proxy and only the mapper can use the required
    # Float. Ordinary depth zero must not exclude the admitted source map.
    @test CG._prepare_generation_obligations(relation, query) === nothing
    @test CG._prepare_generation_obligations(relation, query; source_depth=0) === nothing
    prepared = CG._prepare_generation_obligations(relation, query; source_depth=1)
    @test prepared !== nothing
    if prepared !== nothing
        @test prepared.graph.costs[prepared.root] == 5
        @test CG._prepare_generation_obligations(relation, query;
            source_depth=1).graph === prepared.graph
    end
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    source = CG._sample_finite_required_derivation(state, relation, query; source_depth=1)
    @test source isa CG.HeadApplicationDerivation
    if source isa CG.HeadApplicationDerivation
        @test source.head.name == :AMAP
        @test CG.derivation_result_type(source) == output
        @test CG.derivation_node_count(source) == 5
        @test CG._array_source_depth(source) == 1
        @test CG.derivation_uses_scope_binder(source, 1)
        @test !CG.derivation_uses_scope_binder(source, 0)
        @test CG._generation_requirements_satisfied(source, query)
    end
    @test state.nodes_left == 4
    too_small = CG._derive_generation_query(query; node_budget=4)
    failed = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    @test CG._sample_finite_required_derivation(failed, relation, too_small;
        source_depth=1) === nothing
    @test failed.nodes_left == 9
    @test isempty(relation.derivations)
    @test isempty(relation.source_derivations)

    # External schemes still use the original layered node ceiling at every
    # source/ordinary crossing, rather than introducing unbounded scheme DFS.
    len, elem = newtyvar(0), newtyvar(0)
    layered = CG.CoreRelation(providers=relation.providers,
        global_consts=Dict{Symbol,Any}(:predicate => iszero,
            :LENF => LambdaRegression.Primitives.LENF),
        global_consts_ty=Dict{Symbol,Any}(:predicate => predicate,
            :LENF => TyScheme([len, elem], TyFun(TyArray(len, elem), FloatTy))))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 5)
    result = CG._sample_finite_required_derivation(state, layered, query; source_depth=1)
    @test result !== nothing
    @test result !== nothing && CG._generation_requirements_satisfied(result, query)
    @test state.nodes_left == 0
    for world in values(layered.evidence_worlds)
        @test all(fact -> fact[3] >= 0 && fact[5] >= 0, world.facts)
        @test all(edge -> all(child -> world.facts[child][5] < world.facts[edge.parent][5],
            edge.children), world.edges)
    end
end

@testitem "CanonicalGeneration joint first certificate resumes a complete exact menu" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    make_relation() = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:id => identity),
        global_consts_ty=Dict{Symbol,Any}(:id => TyFun(FloatTy, FloatTy)))
    relation = make_relation()
    query = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy],
        depth_left=6, node_budget=16,
        requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    first = CG._prepare_generation_obligations(relation, query; certificate_only=true)
    @test first !== nothing
    @test first.graph.costs[first.root] == 1
    @test !first.graph.complete
    @test first.graph.world.pending_cursor <= length(first.graph.world.pending)
    @test CG._obligation_replay(first.graph, first.root, first.scope, first.forbidden).index == 1
    sizes = (length(first.graph.facts), length(first.graph.edges))
    @test CG._prepare_generation_obligations(relation, query; certificate_only=true).graph === first.graph
    @test (length(first.graph.facts), length(first.graph.edges)) == sizes
    full = CG._prepare_generation_obligations(relation, query)
    @test full.graph === first.graph
    @test full.graph.complete
    @test all(full.graph.expanded)
    @test isempty(full.graph.world.pending)
    @test isempty(full.graph.pending)
    fresh = CG._prepare_generation_obligations(make_relation(), query)
    @test full.graph.costs[full.root] == fresh.graph.costs[fresh.root]
    @test length(full.graph.options[full.root]) == length(fresh.graph.options[fresh.root])
    for seed in 1:8
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 16)
        term = CG._sample_finite_required_derivation(state, relation, query)
        @test CG.derivation_uses_scope_binder(term, 1)
        @test !CG.derivation_uses_scope_binder(term, 0)
        @test CG.derivation_node_count(term) == 16 - state.nodes_left
    end

    # The ordinary shortest proof is a constant. The required proof must see
    # an alternative head, not restrict itself to that ordinary certificate.
    predicate = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:no => false, :pred => iszero),
        global_consts_ty=Dict{Symbol,Any}(:no => BoolTy, :pred => TyFun(FloatTy, BoolTy)))
    required = CG.GenerationQuery(BoolTy; scope_tys=Ty[FloatTy], depth_left=1,
        node_budget=2, requirements=(CG.UsesScopeBinder(0),))
    proof = CG._prepare_generation_obligations(predicate, required; certificate_only=true)
    @test proof !== nothing
    @test proof.graph.costs[proof.root] == 2
    @test CG.derivation_uses_scope_binder(
        CG._obligation_replay(proof.graph, proof.root, proof.scope, proof.forbidden), 0)
    @test CG._prepare_generation_obligations(predicate,
        CG._derive_generation_query(required; node_budget=1); certificate_only=true) === nothing
    @test proof.graph.complete

    unbounded = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=0,
        requirements=(CG.UsesScopeBinder(0),))
    proof = CG._prepare_generation_obligations(make_relation(), unbounded; certificate_only=true)
    @test proof !== nothing
    @test proof.graph.costs[proof.root] == 1

    struct PolicyFreeNonEarlyJointFixture <: CG.AbstractHeadProvider end
    CG.generation_catalog_scope_invariant(::PolicyFreeNonEarlyJointFixture) = true
    CG._evidence_heads_policy_free(::PolicyFreeNonEarlyJointFixture) = true
    CG.add_heads!(::PolicyFreeNonEarlyJointFixture, heads, universe; kwargs...) = heads
    provider = PolicyFreeNonEarlyJointFixture()
    @test CG._evidence_heads_policy_free(provider)
    @test !CG._evidence_early_certificate_supported(provider)
    conservative = CG.CoreRelation(providers=(CG.ContextHeadProvider(), provider),
        global_consts=Dict{Symbol,Any}(:id => identity),
        global_consts_ty=Dict{Symbol,Any}(:id => TyFun(FloatTy, FloatTy)))
    proof = CG._prepare_generation_obligations(conservative, query; certificate_only=true)
    @test proof !== nothing
    @test proof.graph.complete
end

@testitem "CanonicalGeneration joint partial zero facts follow ordinary improvements" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:no => false, :id => identity,
            :combine => ((x,b) -> b)),
        global_consts_ty=Dict{Symbol,Any}(:no => BoolTy,
            :id => TyFun(BoolTy, BoolTy),
            :combine => TyFun(FloatTy, TyFun(BoolTy, BoolTy))))
    query = CG.GenerationQuery(BoolTy; scope_tys=Ty[FloatTy], depth_left=4,
        node_budget=9, requirements=(CG.UsesScopeBinder(0),))
    world = CG._finite_evidence_world(relation, BoolTy, Ty[FloatTy]; sampling=true)
    context = CG._evidence_context!(world, [CG._evidence_type_id(world, FloatTy)])
    b = CG._evidence_type_id(world, BoolTy)
    x = CG._evidence_fact!(world, context, CG._evidence_type_id(world, FloatTy), 0)
    CG._evidence_expand!(world, relation, x)
    heads = world.catalog.heads
    zero = only(filter(h -> h.name == :no, heads))
    id = only(filter(h -> h.name == :id, heads))
    combine = only(filter(h -> h.name == :combine, heads))
    # A valid five-node certificate exists before the shorter branch expands.
    chain = [CG._evidence_fact!(world, context, b, depth) for depth in 0:4]
    CG._evidence_add_edge!(world, chain[1], :constant, zero, Int[], 1)
    for i in 2:5
        CG._evidence_add_edge!(world, chain[i], :head, id, [chain[i-1]], 1)
    end
    graph = CG._new_generation_obligations(world, query)
    root = CG._obligation_fact!(graph, chain[end], graph.full_mask)
    used = CG._obligation_fact!(graph, x, graph.full_mask)
    empty = CG._obligation_fact!(graph, chain[end], UInt(0))
    CG._obligation_expand!(graph, relation, used; certificate_only=true)
    CG._obligation_sync_ordinary!(graph, chain[end])
    CG._obligation_edge!(graph, root,
        CG._EvidencePlan(:head, combine, [x, chain[end]], 1, 1.0), -1, [used, empty])
    world.obligations[CG._StructuralKey(:partial_test)] = (graph, root)
    CG._obligation_prepare_certificate!(graph, relation, root, 9)
    @test graph.costs[root] == 7
    @test CG.derivation_node_count(CG._obligation_replay(graph, root, Ty[FloatTy], ())) == 7
    # A different, full ordinary consumer must notify the suspended joint graph.
    CG._evidence_saturate!(world, relation)
    @test world.costs[chain[end]] == 1
    @test graph.costs[empty] == 1
    @test graph.costs[root] == 3
    @test CG.derivation_node_count(CG._obligation_replay(graph, root, Ty[FloatTy], ())) == 3
    CG._obligation_prepare_certificate!(graph, relation, root, 3)
    @test graph.costs[root] == 3

    # A raw candidate may not exceed its own scheme-layered fact ceiling,
    # even when a larger caller has enough spare nodes.
    bounded = CG._evidence_fact!(world, context, b, 4, -1, 2)
    parent = CG._obligation_fact!(graph, bounded, graph.full_mask)
    CG._obligation_edge!(graph, parent,
        CG._EvidencePlan(:head, combine, [x, chain[end]], 1, 1.0), -1, [used, empty])
    @test CG._obligation_edge_cost(graph, last(graph.edges)) === nothing
    @test graph.costs[parent] == typemax(Int)
end

@testitem "CanonicalGeneration joint certificate drains ungrounded cycles without negative visiting" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyConst, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    a, b = TyConst(:A), TyConst(:B)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:ab => identity, :ba => identity),
        global_consts_ty=Dict{Symbol,Any}(:ab => TyFun(a,b), :ba => TyFun(b,a)))
    query = CG.GenerationQuery(a; scope_tys=Ty[FloatTy], depth_left=4,
        node_budget=8, requirements=(CG.UsesScopeBinder(0),))
    @test CG._prepare_generation_obligations(relation, query; certificate_only=true) === nothing
    world = only(values(relation.evidence_worlds))
    graph, root = only(values(world.obligations))
    @test graph.complete
    @test graph.costs[root] == typemax(Int)
    @test isempty(world.pending)
    @test isempty(graph.pending)
    @test CG._prepare_generation_obligations(relation, query) === nothing
    # The private monomorphic existence world also supports genuinely cyclic
    # facts (no depth coordinate descent). Do not bypass the public query's
    # nonnegative-depth contract to exercise that internal case.
    context = CG._evidence_context!(world, [CG._evidence_type_id(world, FloatTy)])
    ordinary = CG._evidence_fact!(world, context, CG._evidence_type_id(world, a), -1)
    cycle = CG._new_generation_obligations(world, query)
    root = CG._obligation_fact!(cycle, ordinary, cycle.full_mask)
    CG._obligation_prepare_certificate!(cycle, relation, root, 8)
    @test cycle.complete
    @test cycle.costs[root] == typemax(Int)
    @test length(cycle.facts) == 2
    @test isempty(world.pending)
    @test isempty(cycle.pending)
    CG._obligation_saturate!(cycle, relation)
    @test cycle.costs[root] == typemax(Int)
end

@testitem "CanonicalGeneration joint preparation reuse separates exact requests and root state" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    query = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy],
        depth_left=0, node_budget=1, requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    first = CG._prepare_generation_obligations(relation, query)
    counts = (length(first.graph.facts), length(first.graph.edges))
    repeated = CG._prepare_generation_obligations(relation, query)
    @test repeated.graph === first.graph
    @test repeated.root == first.root
    @test (length(repeated.graph.facts), length(repeated.graph.edges)) == counts
    changed_binder = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy],
        depth_left=0, node_budget=1, requirements=(CG.UsesScopeBinder(0),))
    other = CG._prepare_generation_obligations(relation, changed_binder)
    @test other.graph !== first.graph
    @test CG._obligation_replay(other.graph, other.root, other.scope, other.forbidden).index == 0
    @test CG._obligation_replay(first.graph, first.root, first.scope, first.forbidden).index == 1
    larger = CG.GenerationQuery(FloatTy; scope_tys=query.scope_tys,
        depth_left=0, node_budget=2, requirements=query.requirements)
    @test CG._prepare_generation_obligations(relation, larger).graph === first.graph
    smaller = CG._derive_generation_query(query; node_budget=0)
    @test CG._prepare_generation_obligations(relation, smaller) === nothing
    @test CG._prepare_generation_obligations(relation, larger).graph === first.graph
    reordered = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, BoolTy],
        depth_left=0, node_budget=1, requirements=changed_binder.requirements)
    @test CG._prepare_generation_obligations(relation, reordered).graph !== other.graph
    states = [CG.DerivationSamplingState(Random.Xoshiro(2000), 1,
        CG.GenerationCompletionState(probability=p)) for p in (1.0, 0.0)]
    for state in states
        result = CG._sample_finite_required_derivation(state, relation, query)
        @test result.index == 1
        @test state.nodes_left == 0
    end
    @test states[1].completion.completing
    @test !states[2].completion.completing
    @test CG._prepare_generation_obligations(relation, query).graph === first.graph

    fun = TyFun(FloatTy, FloatTy)
    use = CG.GenerationQuery(FloatTy; scope_tys=Ty[fun, FloatTy],
        depth_left=1, node_budget=3, requirements=(CG.UsesScopeBinder(0),))
    call = CG._derive_generation_query(use; requirements=(CG.CallsScopeBinder(0),))
    used = CG._prepare_generation_obligations(relation, use)
    called = CG._prepare_generation_obligations(relation, call)
    @test used.graph !== called.graph
    @test CG._prepare_generation_obligations(relation, call).graph === called.graph
    # Mutating an external context/policy produces a different owned world.
    relation.local_consts[:zero] = 0.0
    relation.local_consts_ty[:zero] = FloatTy
    after = CG._prepare_generation_obligations(relation, query)
    @test after.graph !== first.graph
    @test after.graph.world !== first.graph.world
end

"""Check that a constrained caller has no possible mixed subconstruction.

This conservative cut uses the effective query's inherited budgets and provider
exhaustion protocol, not just the names of its root options. Unknown providers
are not assumed exhausted. Only this cut permits a core-negative answer to
replace the old constrained caller; mixed queries retain their original routes.
"""
function _pure_evidence_requirement_query(relation::CoreRelation, query::GenerationQuery,
    allow_safe_list_access::Bool)
    allow_safe_list_access && _safe_list_access_enabled(relation) && return false
    query.guarded_left > 0 &&
        _bucket_enabled(relation.weights, _GUARDED_EXPANSION_WEIGHT) && return false
    providers = _admitted_providers(relation.generation_providers, query.capability_mask)
    return all(provider -> generation_provider_exhausted(provider, relation, query), providers)
end

function _obligation_compose(plan::_EvidencePlan, binder::Int, args::Vector{CoreDerivation}, scope::Vector{Ty})
    plan.kind == :variable && return VariableDerivation(binder, scope[binder + 1])
    plan.kind == :constant && return ConstantDerivation(plan.value)
    plan.kind == :lambda && return LambdaDerivation(plan.value, only(args))
    plan.kind == :scope_application && return ScopeApplicationDerivation(binder, scope[binder + 1], args)
    return HeadApplicationDerivation(plan.value, args)
end

function _obligation_replay(graph::_GenerationObligations, fact::Int,
    scope::Vector{Ty}, forbidden::Tuple, offset::Int=0)
    ordinary, mask = graph.facts[fact]
    mask == _obligation_zero(mask) && return _evidence_replay(graph.world, ordinary, scope, forbidden)
    edge = graph.edges[graph.certificates[fact]]
    plan = edge.plan
    binder = if edge.root_binder >= 0
        edge.root_binder + offset
    elseif plan.kind == :scope_application
        slot = findfirst(eachindex(scope)) do slot
            slot - 1 ∉ forbidden && _evidence_type_id(graph.world, scope[slot]) ==
                _evidence_type_id(graph.world, plan.value)
        end
        slot === nothing && error("obligation replay lost a scoped head")
        slot - 1
    else
        -1
    end
    lambda = plan.kind == :lambda
    child_scope = lambda ? Ty[plan.value; scope] : scope
    child_forbidden = lambda ? _shift_scope_binders(forbidden, 1) : forbidden
    args = CoreDerivation[_obligation_replay(graph, child, child_scope,
        child_forbidden, offset + lambda) for child in edge.children]
    derivation = _obligation_compose(plan, binder, args, scope)
    _notify_obligation_edge_diagnostic((graph=graph, fact=fact,
        edge_id=graph.certificates[fact], binder=binder, derivation, args,
        scope, forbidden, offset))
    return derivation
end

function _sample_obligation_fact!(state::DerivationSamplingState, graph::_GenerationObligations,
    relation::CoreRelation, fact::Int, scope::Vector{Ty}, forbidden::Tuple, offset::Int=0)
    cost = graph.costs[fact]
    cost <= state.nodes_left || return nothing
    ordinary, mask = graph.facts[fact]
    mask == _obligation_zero(mask) && return _sample_evidence_fact!(state, graph.world,
        relation, ordinary, scope, forbidden; certificate_ready=true)
    if state.completion.completing || _completion_boundary!(state)
        _consume_sampling_nodes!(state, cost) || return nothing
        return _obligation_replay(graph, fact, scope, forbidden, offset)
    end
    options = NamedTuple[]
    for edge_id in graph.options[fact]
        edge = graph.edges[edge_id]
        edge_cost = _obligation_edge_cost(graph, edge)
        edge_cost === nothing && continue
        edge_cost <= state.nodes_left || continue
        if edge.root_binder < 0 && edge.plan.kind == :scope_application
            head_type = _evidence_type_id(graph.world, edge.plan.value)
            for (slot, ty) in enumerate(scope)
                slot - 1 ∈ forbidden && continue
                _evidence_type_id(graph.world, ty) == head_type || continue
                push!(options, (edge=edge_id, binder=slot - 1, weight=edge.plan.weight))
            end
        else
            push!(options, (edge=edge_id, binder=edge.root_binder + offset, weight=edge.plan.weight))
        end
    end
    pool, total = _positive_option_pool(options)
    isempty(pool) && error("a joint certificate has no budget-admissible sampling plan")
    selected = pool[_weighted_option_index(state.rng, pool, total)]
    edge = graph.edges[selected.edge]
    plan = edge.plan
    _consume_sampling_nodes!(state, plan.charge) || return nothing
    lambda = plan.kind == :lambda
    child_scope = lambda ? Ty[plan.value; scope] : scope
    child_forbidden = lambda ? _shift_scope_binders(forbidden, 1) : forbidden
    reserved = sum(child -> graph.costs[child], edge.children; init=0)
    args = CoreDerivation[]
    for child in edge.children
        reserved -= graph.costs[child]
        state.nodes_left -= reserved
        arg = _sample_obligation_fact!(state, graph, relation, child, child_scope,
            child_forbidden, offset + lambda)
        state.nodes_left += reserved
        arg isa _UnsupportedEvidenceSampling && return arg
        arg === nothing && error("joint evidence failed a budget-reserved child")
        push!(args, arg)
    end
    derivation = _obligation_compose(plan, selected.binder, args, scope)
    _notify_obligation_edge_diagnostic((graph=graph, fact=fact,
        edge_id=selected.edge, binder=selected.binder, derivation, args,
        scope, forbidden, offset))
    return derivation
end

function _sample_finite_required_derivation(state::DerivationSamplingState,
    relation::CoreRelation, query::GenerationQuery; source_depth::Int=-1)
    available = min(state.nodes_left, query.node_budget)
    query = _derive_generation_query(query; node_budget=available)
    prepared = _prepare_generation_obligations(relation, query; source_depth)
    prepared isa _UnsupportedEvidenceSampling && return prepared
    prepared === nothing && return nothing
    (; graph, root, scope, forbidden, relation) = prepared
    saved_nodes = state.nodes_left
    outside_reserve = saved_nodes - available
    state.nodes_left = available
    try
        sampled = _sample_obligation_fact!(state, graph, relation, root, scope, forbidden)
        sampled isa _UnsupportedEvidenceSampling && (state.nodes_left = available)
        return sampled
    finally
        state.nodes_left += outside_reserve
    end
end

@testitem "CanonicalGeneration joint evidence preserves exact use call and lexical identity" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    exact = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy],
        depth_left=0, node_budget=1, requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 8,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._sample_finite_required_derivation(state, relation, exact)
    @test result isa CG.VariableDerivation
    @test result.index == 1
    @test state.nodes_left == 7
    @test state.completion.completing
    @test state.completion.boundaries == 1
    contradictory = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy],
        depth_left=0, node_budget=1, requirements=(CG.UsesScopeBinder(0), CG.ForbidsScopeBinder(0)))
    @test CG._prepare_generation_obligations(relation, contradictory) === nothing

    function_ty = TyFun(FloatTy, FloatTy)
    lexical = CG.GenerationQuery(function_ty; scope_tys=Ty[FloatTy],
        depth_left=1, node_budget=2, requirements=(CG.UsesScopeBinder(0),))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 2)
    result = CG._sample_finite_required_derivation(state, relation, lexical)
    @test result isa CG.LambdaDerivation
    @test result.body isa CG.VariableDerivation
    @test result.body.index == 1 # root binder, not the new lambda's same-type binder
    @test CG.derivation_uses_scope_binder(result, 0)
    @test state.nodes_left == 0

    mere_use = CG.GenerationQuery(function_ty; scope_tys=Ty[function_ty],
        depth_left=0, node_budget=3, requirements=(CG.UsesScopeBinder(0),))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 3)
    result = CG._sample_finite_required_derivation(state, relation, mere_use)
    @test result isa CG.VariableDerivation
    @test !CG.derivation_calls_scope_binder(result, 0)
    bare_call = CG.GenerationQuery(function_ty; scope_tys=Ty[function_ty],
        depth_left=0, node_budget=3, requirements=(CG.CallsScopeBinder(0),))
    @test CG._prepare_generation_obligations(relation, bare_call) === nothing
    call = CG.GenerationQuery(function_ty; scope_tys=Ty[function_ty],
        depth_left=2, node_budget=3, requirements=(CG.CallsScopeBinder(0), CG.UsesScopeBinder(0)))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 3,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._sample_finite_required_derivation(state, relation, call)
    @test result isa CG.LambdaDerivation
    @test result.body isa CG.ScopeApplicationDerivation
    @test result.body.index == 1
    @test CG.derivation_calls_scope_binder(result, 0)
    @test CG.derivation_node_count(result) == 3
    @test state.completion.boundaries == 1

    # Delegation of an empty guarantee does not check the root switch twice.
    prepared = CG._prepare_generation_obligations(relation, exact)
    empty_fact = CG._obligation_fact!(prepared.graph, prepared.graph.facts[prepared.root][1], UInt(0))
    CG._obligation_saturate!(prepared.graph, prepared.relation)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 1,
        CG.GenerationCompletionState(probability=1.0))
    CG._sample_obligation_fact!(state, prepared.graph, prepared.relation,
        empty_fact, prepared.scope, prepared.forbidden)
    @test state.completion.boundaries == 1
end

@testitem "CanonicalGeneration joint evidence certifies allocations instead of independent reachability" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    scope = Ty[FloatTy, FloatTy]
    for index in (0, 1)
        query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=0,
            node_budget=1, requirements=(CG.UsesScopeBinder(index),))
        @test CG._prepare_generation_obligations(relation, query) !== nothing
    end
    joint = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=0,
        node_budget=2, requirements=(CG.UsesScopeBinder(0), CG.UsesScopeBinder(1)))
    @test CG._prepare_generation_obligations(relation, joint) === nothing

    combined = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:combine => ((x,y) -> x < y)),
        global_consts_ty=Dict{Symbol,Any}(:combine => TyFun(FloatTy, TyFun(FloatTy, BoolTy))))
    small = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=1,
        node_budget=2, requirements=joint.requirements)
    large = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=1,
        node_budget=3, requirements=joint.requirements)
    @test CG._prepare_generation_obligations(combined, small) === nothing
    @test CG._prepare_generation_obligations(combined, large) !== nothing
    assignments = Set{Tuple{Int,Int}}()
    for seed in 1:20
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 3)
        result = CG._sample_finite_required_derivation(state, combined, large)
        @test result isa CG.HeadApplicationDerivation
        @test CG.derivation_uses_scope_binder(result, 0)
        @test CG.derivation_uses_scope_binder(result, 1)
        @test CG.derivation_node_count(result) == 3
        @test state.nodes_left == 0
        push!(assignments, (first(result.args).index, last(result.args).index))
    end
    @test assignments == Set([(0, 1), (1, 0)])
    @test isempty(combined.derivations)

    # A wide mask must not impose an undocumented obligation-count limit.
    full = CG._obligation_full_mask(70)
    @test full isa BigInt
    @test count_ones(full) == 70
    placements = CG._ObligationMask[]
    CG._obligation_allocations!(full, 1) do allocation
        append!(placements, allocation)
    end
    @test placements == [full]
end

@testitem "CanonicalGeneration joint evidence preserves source identity and pure caller boundary" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyArray, TyNat, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.AMapProvider(), CG.ToListProvider()),
        global_consts=Dict{Symbol,Any}(:predicate => iszero),
        global_consts_ty=Dict{Symbol,Any}(:predicate => TyFun(FloatTy, BoolTy)))
    query = CG.GenerationQuery(TyList(BoolTy); scope_tys=Ty[array, array], depth_left=1,
        node_budget=4, guarded_left=0,
        requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    @test CG._pure_evidence_requirement_query(relation, query, false)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 4,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._sample_required_scope_certificate_derivation(state, relation,
        query.target_ty, Ty[array, array], 1, 0, false, query)
    @test result isa CG.HeadApplicationDerivation
    @test result.head.name == :TOLIST
    @test only(result.args).head.name == :AMAP
    @test CG.derivation_uses_scope_binder(result, 1)
    @test !CG.derivation_uses_scope_binder(result, 0)
    @test CG.derivation_node_count(result) == 4
    @test state.nodes_left == 0
    @test isempty(relation.derivations)
    @test isempty(relation.source_derivations)

    guarded = CG.GenerationQuery(query.target_ty; scope_tys=query.scope_tys,
        depth_left=1, node_budget=4, guarded_left=1, requirements=query.requirements)
    @test !CG._pure_evidence_requirement_query(relation, guarded, false)
    safe = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    @test !CG._pure_evidence_requirement_query(safe, query, true)
    @test CG._pure_evidence_requirement_query(safe, query, false)
    functiondefs = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.FunctionDefProvider()))
    available = CG.GenerationQuery(TyFun(FloatTy, FloatTy); depth_left=2, node_budget=8,
        guarded_left=0, provider_fuel=1)
    @test !CG._pure_evidence_requirement_query(functiondefs, available, false)
    exhausted = CG.GenerationQuery(available.target_ty; depth_left=2, node_budget=8,
        guarded_left=0, provider_fuel=0)
    @test CG._pure_evidence_requirement_query(functiondefs, exhausted, false)
end
