"""Private, root-owned certified construction frontier.

Each slot owns an exact request and a complete certificate supplied by an
admitted builder. Exploration can replace a slot by one certified step;
completion only replays the retained certificates. This is an incremental
protocol, not a decision procedure or a claim of full mixed-language support.
Guarded constructs, sources, custom policies and unknown extensions remain
complete atoms.
"""
struct _GenerationFrontierItem
    query::GenerationQuery
    certificate::CoreDerivation
    cost::Int
    explore::Bool
    functiondef_root::Bool
end

struct _GenerationFrontierStep
    seed::CoreDerivation
    children::Vector{_GenerationFrontierItem}
    charge::Int
end

mutable struct _GenerationFrontierFrame
    step::_GenerationFrontierStep
    args::Vector{CoreDerivation}
    remaining::Int
    parent::Int
    position::Int
end

# Certificates are read-only within this session. Memoise output charges, not
# queries or negative decisions. Decomposing a retained certificate must not
# repeatedly count the same complete subtree.
function _frontier_cost!(costs::IdDict{CoreDerivation,Int}, d::CoreDerivation)
    return get!(costs, d) do
        d isa LambdaDerivation && return 1 + _frontier_cost!(costs, d.body)
        if d isa Union{HeadApplicationDerivation,ScopeApplicationDerivation}
            charge = d isa HeadApplicationDerivation ? 1 : length(d.args)
            return charge + sum(arg -> _frontier_cost!(costs, arg), d.args; init=0)
        elseif d isa SafeListAccessDerivation
            return 3 + _frontier_cost!(costs, d.source) +
                _frontier_cost!(costs, d.body) + _frontier_cost!(costs, d.fallback)
        elseif d isa FunctionDefDerivation
            return 1 + _frontier_cost!(costs, d.body)
        end
        return derivation_node_count(d)
    end
end

function _frontier_certificate_matches(d::CoreDerivation, q::GenerationQuery)
    _generation_requirements_satisfied(d, q) || return false
    any(i -> derivation_uses_scope_binder(d, i), _forbidden_scope_binders(q)) && return false
    ty = derivation_result_type(d)
    return ty === q.target_ty || TypeSystem.iscompatible(ty, q.target_ty)
end

function _frontier_item(costs, q, d; explore=true, functiondef_root=false)
    return _GenerationFrontierItem(q, d, _frontier_cost!(costs, d), explore, functiondef_root)
end

function _frontier_functiondef_body_query(q, d::FunctionDefDerivation, nodes::Int)
    return _derive_generation_query(_close_functiondef_prefix(q); target_ty=d.target_ty,
        added_scope_tys=(d.target_ty,), node_budget=nodes,
        requirements=(FunctionDefFuel(d.remaining), FunctionDefSelfBinder(0), CallsScopeBinder(0)))
end

# Fixed heads discharge only their own binding. Every other positive identity
# is assigned to an actual contributing child, which must continue to fulfil
# it even when its term is replaced later.
function _frontier_argument_queries(q, d, argument_tys::Vector{Ty}, nodes::Int, rng)
    assigned = [Int[] for _ in argument_tys]
    for index in _required_scope_binder_indices(q, length(q.scope_tys))
        d isa ScopeApplicationDerivation && index == d.index && !isempty(d.args) && continue
        needs_call = any(r -> r isa CallsScopeBinder && r.index == index, q.requirements)
        # Zero-mask witnesses may incidentally contribute the same identity.
        # Keep every actual allocation reachable, not just its first occurrence.
        child, contributors = 0, 0
        for (i, arg) in pairs(d.args)
            contributes = needs_call ? derivation_calls_scope_binder(arg, index) :
                derivation_uses_scope_binder(arg, index)
            contributes || continue
            contributors += 1
            (contributors == 1 || rand(rng, 1:contributors) == 1) && (child = i)
        end
        child == 0 && error("frontier certificate lost an allocated binder obligation")
        push!(assigned[child], index)
    end
    return GenerationQuery[_derive_generation_query(
        _scope_binder_requirement_subset(q, assigned[i]); target_ty=argument_tys[i],
        depth_left=q.depth_left - 1, node_budget=nodes) for i in eachindex(argument_tys)]
end

function _frontier_argument_types(ty::TyLike, count::Int)
    ty isa Ty || return nothing
    args = Ty[]
    for _ in 1:count
        view = generation_function_view(ty)
        view isa TyFun || return nothing
        push!(args, view.arg)
        ty = view.res
    end
    return args
end

_frontier_unknown_core_requirements(q::GenerationQuery) = any(q.requirements) do requirement
    requirement isa Union{UsesScopeBinder,CallsScopeBinder,ForbidsScopeBinder} && return false
    generation_requirement_affects_core_search(requirement)
end

_frontier_prepare_supported(relation, q) =
    relation.argument_planner isa DefaultArgumentPlanner &&
    _finite_evidence_request_supported(relation, q) &&
    !_frontier_unknown_core_requirements(q)

@testitem "CanonicalGeneration frontier retains original FunctionDef parameter identity" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    fun, list = TyFun(FloatTy, FloatTy), TyList(FloatTy)
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:LIST => [1.0]),
        global_consts_ty=Dict{Symbol,Any}(:LIST => list),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    q = CG.GenerationQuery(fun; scope_tys=Ty[fun], depth_left=5,
        node_budget=24, requirements=(CG.FunctionDefSelfBinder(0),
            CG.CallsScopeBinder(0)))
    call = CG.ScopeApplicationDerivation(1, fun,
        CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)])
    lambda = CG.LambdaDerivation(FloatTy, call)
    costs = IdDict{CG.CoreDerivation,Int}()
    lambda_step = CG._frontier_step(relation, CG._frontier_item(costs, q, lambda),
        costs, Random.Xoshiro(1111))
    body = only(lambda_step.children)
    marker = only(r for r in body.query.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (1, 1, true)
    call_step = CG._frontier_step(relation, body, costs, Random.Xoshiro(555))
    argument = only(call_step.children)
    marker = only(r for r in argument.query.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (1, 1, false)
    @test argument.query.target_ty == FloatTy

    world = CG._finite_evidence_world(relation, list, Ty[FloatTy, fun]; sampling=true)
    sourcehead = only(filter(head -> head.name == :LIST, world.catalog.heads))
    source = CG.ConstantDerivation(sourcehead)
    safe = CG.SafeListAccessDerivation(source, list,
        CG.VariableDerivation(1, FloatTy), call, FloatTy)
    safe_step = CG._frontier_step(relation,
        CG._frontier_item(costs, body.query, safe), costs, Random.Xoshiro(42))
    safe_body = only(safe_step.children)
    marker = only(r for r in safe_body.query.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (3, 1, false)
    @test safe_body.query.scope_tys[4] == fun
end

"""Reveal one admitted step of a complete certificate, without proof search.

`relation` supplies the default argument policy; `item` supplies exact scope,
obligations, capabilities and bounds. Safe source/fallback contributions are
locked, proxy use is preserved as one OR branch, and guarded constructs and
custom/anchored primitive source-array arguments stay atomic. Unsupported decomposition returns an atomic step,
never a negative answer to the original generation request.
"""
function _frontier_step(relation::CoreRelation, item::_GenerationFrontierItem, costs, rng)
    q, d = item.query, item.certificate
    children = _GenerationFrontierItem[]
    _frontier_unknown_core_requirements(q) && return _GenerationFrontierStep(d, children, item.cost)
    if d isa FunctionDefDerivation && item.functiondef_root
        childq = _frontier_functiondef_body_query(q, d, q.node_budget - 1)
        push!(children, _frontier_item(costs, childq, d.body))
        return _GenerationFrontierStep(d, children, 1)
    elseif d isa LambdaDerivation && q.depth_left > 0
        target = generation_function_view(q.target_ty)
        target isa TyFun || return _GenerationFrontierStep(d, children, item.cost)
        childq = _derive_generation_query(q; target_ty=target.res,
            added_scope_tys=(d.arg_ty,), depth_left=q.depth_left - 1,
            node_budget=q.node_budget - 1)
        childq = _advance_functiondef_prefix(childq)
        push!(children, _frontier_item(costs, childq, d.body))
        return _GenerationFrontierStep(d, children, 1)
    elseif d isa SafeListAccessDerivation && q.depth_left > 1
        # Replacing a source independently would invalidate proxy types and
        # incidental discharged requirements. Both supporting terms are atoms.
        charge = 3 + _frontier_cost!(costs, d.source) + _frontier_cost!(costs, d.fallback)
        parent = _without_satisfied_scope_binder_requirements(q, d.source)
        parent = _without_satisfied_scope_binder_requirements(parent, d.fallback)
        proxies = Int[i for i in (0, 1) if derivation_uses_scope_binder(d.body, i)]
        isempty(proxies) && error("safe-list certificate has no proxy contribution")
        proxy = length(proxies) == 1 ? only(proxies) : rand(rng, proxies)
        childq = _derive_generation_query(parent; target_ty=d.result_ty,
            added_scope_tys=(d.source_ty, d.source_ty.elem), depth_left=q.depth_left - 2,
            node_budget=q.node_budget - charge, capability_mask=(:safe_list_access,),
            requirements=(UsesScopeBinder(proxy),))
        childq = _close_functiondef_prefix(childq)
        push!(children, _frontier_item(costs, childq, d.body))
        return _GenerationFrontierStep(d, children, charge)
    elseif d isa Union{HeadApplicationDerivation,ScopeApplicationDerivation} &&
            !isempty(d.args) && q.depth_left > 0 &&
            relation.argument_planner isa DefaultArgumentPlanner
        if d isa HeadApplicationDerivation &&
                (_is_guarded_head(d.head) ||
                 _metadata_get(d.head.metadata, :argument_planner, nothing) !== nothing)
            # Generic child queries do not preserve guarded depth, COND's
            # condition grammar or AFOLDL's grounded/synthesized-step policy.
            # This applies equally to retained and newly prepared certificates.
            return _GenerationFrontierStep(d, children, item.cost)
        end
        ty = d isa HeadApplicationDerivation ? d.head.type : d.head_ty
        argtys = _frontier_argument_types(ty, length(d.args))
        argtys === nothing && return _GenerationFrontierStep(d, children, item.cost)
        charge = d isa HeadApplicationDerivation ? 1 : length(d.args)
        queries = _frontier_argument_queries(_close_functiondef_prefix(q), d,
            argtys, q.node_budget - charge, rng)
        for i in eachindex(d.args)
            # Only primitive array-source operands use the anchored grammar.
            # A scoped function's array operand is an ordinary Known child.
            push!(children, _frontier_item(costs, queries[i], d.args[i];
                explore=d isa ScopeApplicationDerivation ||
                    _argument_class(argtys[i]) != :source_array))
        end
        return _GenerationFrontierStep(d, children, charge)
    end
    return _GenerationFrontierStep(d, children, item.cost)
end

_frontier_compose(d::LambdaDerivation, args) = LambdaDerivation(d.arg_ty, only(args))
_frontier_compose(d::HeadApplicationDerivation, args) = HeadApplicationDerivation(d.head, args)
_frontier_compose(d::ScopeApplicationDerivation, args) = ScopeApplicationDerivation(d.index, d.head_ty, args)
_frontier_compose(d::SafeListAccessDerivation, args) =
    SafeListAccessDerivation(d.source, d.source_ty, only(args), d.fallback, d.result_ty)
_frontier_compose(d::FunctionDefDerivation, args) = FunctionDefDerivation(
    d.target_ty, only(args), d.base_witness, d.self_index, d.capabilities, d.remaining)

"""Run one complete, certified root frontier using a shared completion state.

`prepare(relation, query, rng)` may supply ONE complete admitted alternative or
`nothing`; it has a finite preparation contract and may not own/reset the root
mode or shared extension-attempt ledger. Preparation and step admission happen
before any prefix/node commitment. Failure retains the old certificate and
reveals it once, not an uncharged retry. After switching no callback is invoked.
The caller validates initial policy admission and any final post-filter.
"""
function _sample_certified_frontier!(state::DerivationSamplingState, relation::CoreRelation,
    query::GenerationQuery, certificate::CoreDerivation;
    prepare=_prepare_frontier_candidate, step=_frontier_step, functiondef_root::Bool=false)
    available = min(state.nodes_left, query.node_budget)
    outside = state.nodes_left - available
    state.nodes_left = available
    try
        return _run_certified_frontier!(state, relation, query, certificate,
            prepare, step, functiondef_root)
    finally
        state.nodes_left += outside
    end
end

function _run_certified_frontier!(state::DerivationSamplingState, relation::CoreRelation,
    query::GenerationQuery, certificate::CoreDerivation, prepare, prepare_step,
    functiondef_root::Bool)
    costs = IdDict{CoreDerivation,Int}()
    initial = _frontier_item(costs, query, certificate; functiondef_root)
    initial.cost <= min(state.nodes_left, query.node_budget) || return nothing
    _frontier_certificate_matches(certificate, query) ||
        throw(ArgumentError("initial frontier certificate does not satisfy its request"))
    frames = _GenerationFrontierFrame[]
    pending = Tuple{_GenerationFrontierItem,Int,Int}[(initial, 0, 0)]
    reserved = initial.cost
    result = nothing
    while !isempty(pending)
        # Any exploration order must remain possible: a cheaper sibling can
        # release its reservation before a different sibling grows. Swap-pop
        # preserves each item's parent/position without shifting the frontier.
        slot = state.completion.completing || length(pending) == 1 ?
            length(pending) : rand(state.rng, eachindex(pending))
        item, parent, position = pending[slot]
        pending[slot] = last(pending)
        pop!(pending)
        reserved -= item.cost
        available = min(item.query.node_budget, state.nodes_left - reserved)
        item.cost <= available || error("frontier lost a pending sibling reservation")
        q = _derive_generation_query(item.query; node_budget=available)
        if item.explore && !state.completion.completing
            _completion_boundary!(state)
        end
        if !item.explore || state.completion.completing
            _consume_sampling_nodes!(state, item.cost) || error("certified replay lost node fuel")
            result = item.certificate
        else
            # The fixed FunctionDef wrapper is not a second candidate route.
            alternative = item.functiondef_root ? nothing : prepare(relation, q, state.rng)
            # q is only the current sibling-reserved lease. Child structural
            # ceilings inherit the original request, not this transient lease.
            selected = item
            if alternative !== nothing
                alternative isa CoreDerivation || throw(ArgumentError("frontier preparation must return a derivation or nothing"))
                cost = _frontier_cost!(costs, alternative)
                if cost <= available && _frontier_certificate_matches(alternative, q)
                    selected = _GenerationFrontierItem(item.query, alternative, cost, true, false)
                end
            end
            step = prepare_step(relation, selected, costs, state.rng)
            total = step.charge + sum(child -> child.cost, step.children; init=0)
            total <= available || error("prepared frontier step exceeds its certificate lease")
            _consume_sampling_nodes!(state, step.charge) || error("certified step lost node fuel")
            if !isempty(step.children)
                push!(frames, _GenerationFrontierFrame(step,
                    Vector{CoreDerivation}(undef, length(step.children)), length(step.children), parent, position))
                frameid = length(frames)
                for i in reverse(eachindex(step.children))
                    child = step.children[i]
                    push!(pending, (child, frameid, i))
                    reserved += child.cost
                end
                continue
            end
            result = step.seed
        end
        # Finish ancestors iteratively. No inference, catalogue or chooser is
        # called by composition; all pending certificates were committed first.
        while parent != 0
            frame = frames[parent]
            frame.args[position] = result
            frame.remaining -= 1
            frame.remaining != 0 && break
            result = _frontier_compose(frame.step.seed, frame.args)
            parent, position = frame.parent, frame.position
        end
    end
    reserved == 0 || error("frontier leaked a pending reservation")
    return result
end

_bounded_packet_provider_supported(::AbstractHeadProvider) = false
_bounded_packet_provider_supported(::Union{ContextHeadProvider,NilFactoryProvider,
    ToListProvider,AMapProvider,GuardedExpansionProvider,WeightTagProvider}) = true
_bounded_packet_provider_supported(provider::SafeListAccessProvider) =
    isempty(provider.source_collectors)
_bounded_packet_provider_supported(provider::TypedValueFactoryProvider) =
    all(factory -> factory isa Union{ValueFactories.NilFactory,
        ValueFactories.FloatScalarFactory,ValueFactories.IntScalarFactory,
        ValueFactories.FloatArrayFactory,ValueFactories.IntArrayFactory,
        ValueFactories.BoolArrayFactory}, provider.factories)
_bounded_packet_provider_supported(provider::FunctionDefProvider) =
    provider.base_witness isa FiniteBaseWitness &&
    _bounded_packet_filter_supported(provider.admissibility)
_bounded_packet_filter_supported(::Any) = false
_bounded_packet_filter_supported(::Union{AcceptFunctionDef,DirectKnotFilter}) = true
_bounded_packet_filter_supported(filter::GenuineSelfCallFilter) =
    _bounded_packet_filter_supported(filter.inner)

function _bounded_packet_request_supported(relation::CoreRelation, query::GenerationQuery)
    relation, query = _relation_for_query(relation, query)
    relation.argument_planner isa DefaultArgumentPlanner || return false
    all(_bounded_packet_provider_supported, relation.providers) || return false
    _finite_evidence_request_supported(relation, query) || return false
    return !_frontier_unknown_core_requirements(query)
end

"""Prepare one complete packet through the reviewed bounded mixed sampler.

The query owns its already-created root requirements and attempt ledger.
Preparation leases nodes but does not create another root mode or ledger, and
must finish before the root's next boundary. A failed or incompatible packet
does not prove mixed non-inhabitation. Completion never calls this function.
Only its known finite built-in configuration is admitted by the public root;
custom callbacks and recursive targets retain the explicitly bounded entry.
"""
function _prepare_bounded_mixed_packet(relation::CoreRelation, query::GenerationQuery,
    rng, allow_safe_list_access::Bool)
    temporary = DerivationSamplingState(rng, query.node_budget)
    certificate = _sample_generation_derivation(temporary, relation, query,
        _scope_type_vector(query.scope_tys), allow_safe_list_access)
    certificate === nothing && return nothing
    _frontier_certificate_matches(certificate, query) || return nothing
    cost = derivation_node_count(certificate)
    cost <= query.node_budget - temporary.nodes_left ||
        error("bounded packet preparation returned an uncharged certificate")
    cost <= query.node_budget || return nothing
    rigid = _contains_rigid_dummy(query.target_ty) ||
        _scope_any(_contains_rigid_dummy, query.scope_tys)
    rigid && !rigid_safe_derivation(certificate) && return nothing
    return certificate
end

"""Expose only descendants whose mixed-request policy is known at this entry.

All exposed slots originate in the reviewed mixed sampler. Lambda bodies and
non-array scoped-call arguments inherit that same request policy. Ordinary
head parameters, proxy/Safe bodies and filtered definitions have different
internal policies that cannot be reconstructed from the term; their complete
packets stay atomic, for BOTH initial and alternative certificates. This is
conservative decomposition, not a restriction of complete-packet support.
"""
function _mixed_packet_frontier_step(relation::CoreRelation,
    item::_GenerationFrontierItem, costs, rng)
    if item.certificate isa Union{LambdaDerivation,ScopeApplicationDerivation}
        return _frontier_step(relation, item, costs, rng)
    end
    return _GenerationFrontierStep(item.certificate, _GenerationFrontierItem[], item.cost)
end

@testitem "CanonicalGeneration public root completion preserves bounded zero mode and rigid targets" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyScheme, FloatTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    config(seed, sampler=CG.WeightedDerivationSampler()) = CG.GenerationConfig(;
        rng=Random.Xoshiro(seed), providers=(CG.ContextHeadProvider(),), sampler,
        sampling_depth_limit=3, sampling_node_budget=2, guarded_expansion_depth=0)
    original = config(31)
    zero = config(31, CG.WeightedDerivationSampler(completion_probability=0,
        completion_grace_steps=9))
    old_result = CG.generate_canonical(target; config=original)
    zero_result = CG.generate_canonical(target; config=zero)
    @test old_result isa CG.GenerationResult
    @test zero_result isa CG.GenerationResult
    @test old_result.tree.root == zero_result.tree.root
    @test old_result.metadata[:sampling_nodes_used] == zero_result.metadata[:sampling_nodes_used] == 2
    @test !haskey(zero_result.metadata, :certified_completion)
    @test rand(original.rng) == rand(zero.rng)
    sampler = CG.WeightedDerivationSampler(completion_probability=1)
    result = CG.generate_canonical(target; config=config(31, sampler))
    @test result isa CG.GenerationResult
    @test result.metadata[:sampling_nodes_used] == 2
    @test result.metadata[:certified_completion].protocol === :shared_root_bounded_packets
    @test result.metadata[:certified_completion].initial_preparation === :bounded_candidate
    @test result.metadata[:certified_completion].switched
    @test result.metadata[:certified_completion].boundaries == 1
    forced_config() = CG.GenerationConfig(; rng=Random.Xoshiro(2000), sampler,
        providers=(CG.ContextHeadProvider(),), sampling_depth_limit=3,
        sampling_node_budget=12, guarded_expansion_depth=0)
    forced = CG.generate_functiondef_root_scoped(target; config=forced_config(),
        functiondef_provider=CG.FunctionDefProvider(admissibility=CG.AcceptFunctionDef()))
    @test forced isa CG.GenerationResult
    @test forced.metadata[:certified_completion].switched
    overridden = CG.generate_functiondef_root_scoped(target; config=forced_config(),
        functiondef_provider=CG.FunctionDefProvider(admissibility=CG.AcceptFunctionDef()),
        completion_probability=0)
    reference = CG.generate_functiondef_root_scoped(target; config=forced_config(),
        functiondef_provider=CG.FunctionDefProvider(admissibility=CG.AcceptFunctionDef()),
        completion_probability=0, completion_grace_steps=0)
    @test typeof(overridden) === typeof(reference)
    if overridden isa CG.GenerationResult
        @test !haskey(overridden.metadata, :certified_completion)
    end
    scoped = CG.generate_canonical_scoped(FloatTy; scope_tys=Ty[FloatTy], config=config(47, sampler))
    @test scoped isa CG.GenerationResult
    @test scoped.metadata[:sampling_nodes_used] == 1
    @test scoped.metadata[:certified_completion].boundaries == 1
    alpha = newtyvar(0)
    rigid = CG.generate_canonical(TyScheme([alpha], TyFun(alpha, alpha)); config=config(31, sampler))
    @test rigid isa CG.GenerationResult
    @test rigid.metadata[:sampling_nodes_used] == 2
    @test rigid.metadata[:certified_completion].switched
    @test_throws ArgumentError CG.WeightedDerivationSampler(completion_probability=NaN)
    @test_throws ArgumentError CG.WeightedDerivationSampler(completion_probability=-0.1)
    @test_throws ArgumentError CG.WeightedDerivationSampler(completion_grace_steps=-1)
    custom_planner = CG.GenerationConfig(; sampler, argument_planner=identity)
    @test_throws ArgumentError CG.generate_canonical(target; config=custom_planner)
    @test_throws ArgumentError CG.generate_canonical_scoped(FloatTy;
        scope_tys=Ty[FloatTy], config=custom_planner)
end

@testitem "CanonicalGeneration mixed packet steps retain head and proxy policies atomically" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(:FADD => +, :zero => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
            :zero => FloatTy))
    head_scope = Ty[FloatTy]
    head_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:FADD => +),
        global_consts_ty=Dict{Symbol,Any}(:FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))))
    head_query = CG.GenerationQuery(FloatTy; scope_tys=head_scope, depth_left=3,
        guarded_left=0, node_budget=3, requirements=(CG.UsesScopeBinder(0),))
    head_option = only(filter(option -> option.kind == :productive_head &&
        option.value.head.name == :FADD,
        CG._sampling_options(head_relation, FloatTy, head_scope, 3;
            allow_head_application=true, lambda_kind=:lambda_generation)))
    head = CG._sample_option(CG.DerivationSamplingState(Random.Xoshiro(2000), 3),
        head_relation, head_option, head_scope, 3, 0, false, head_query)
    @test head isa CG.HeadApplicationDerivation
    scope = Ty[list]
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
        guarded_left=0, node_budget=8)
    safe_option = only(CG._safe_list_access_generation_options(relation, FloatTy, scope))
    safe = CG._sample_option(CG.DerivationSamplingState(Random.Xoshiro(2000), 8),
        relation, safe_option, scope, 3, 0, true, query)
    @test safe isa CG.SafeListAccessDerivation
    for (q, certificate) in ((head_query, head), (query, safe))
        @test certificate !== nothing && CG._frontier_certificate_matches(certificate, q)
        calls = Ref(0)
        prepare = (relation, query, rng) -> begin
            calls[] += 1
            calls[] == 1 || error("an internal parameter/proxy policy was reopened")
            certificate
        end
        state = CG.DerivationSamplingState(Random.Xoshiro(1111), q.node_budget + 4,
            CG.GenerationCompletionState(probability=1, grace_steps=3))
        result = CG._sample_certified_frontier!(state, relation, q, certificate;
            prepare, step=CG._mixed_packet_frontier_step)
        @test result === certificate
        @test calls[] == 1
        @test state.nodes_left == q.node_budget + 4 - CG.derivation_node_count(certificate)
        @test state.completion.boundaries == 1
    end
end

@testitem "CanonicalGeneration frontier can explore scoped array arguments" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyArray, TyNat, TyFun, FloatTy

    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    function_ty = TyFun(array, FloatTy)
    scope = Ty[function_ty, array]
    argument = CG.VariableDerivation(1, array)
    certificate = CG.ScopeApplicationDerivation(0, function_ty, CG.CoreDerivation[argument])
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
        node_budget=6)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    costs = IdDict{CG.CoreDerivation,Int}()
    item = CG._frontier_item(costs, query, certificate)
    step = CG._frontier_step(relation, item, costs, Random.Xoshiro(2000))
    @test length(step.children) == 1
    @test only(step.children).explore
    @test only(step.children).certificate === argument
end

@testitem "CanonicalGeneration root packet qualification checks actual callbacks before preparing" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyVar, TyMu, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    VF = LambdaRegression.ValueFactories
    query = CG.GenerationQuery(TyFun(FloatTy, FloatTy); depth_left=3, node_budget=12)
    builtin = CG.CoreRelation(providers=(CG.default_head_providers()...,
        CG.TypedValueFactoryProvider(VF.FloatScalarFactory()), CG.SafeListAccessProvider(),
        CG.FunctionDefProvider(admissibility=CG.GenuineSelfCallFilter())))
    @test CG._bounded_packet_request_supported(builtin, query)
    calls = Ref(0)
    base = (args...) -> (calls[] += 1; nothing)
    custom = CG.FunctionDefProvider(base_witness=base)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), custom))
    @test !CG._bounded_packet_request_supported(relation, query)
    @test calls[] == 0
    configuration = CG.GenerationConfig(providers=relation.providers,
        sampler=CG.WeightedDerivationSampler(completion_probability=1))
    @test_throws ArgumentError CG.generate_canonical(query.target_ty; config=configuration)
    @test calls[] == 0
    @test !CG._bounded_packet_provider_supported(CG.FunctionDefProvider(
        admissibility=CG.GenuineSelfCallFilter(candidate -> true)))
    collector = (args...) -> (calls[] += 1; CG.ListAccessSource[])
    @test !CG._bounded_packet_request_supported(CG.CoreRelation(providers=(
        CG.ContextHeadProvider(), CG.SafeListAccessProvider(source_collectors=(collector,)))), query)
    @test calls[] == 0
    struct CustomPacketFactory <: VF.AbstractValueFactory end
    @test !CG._bounded_packet_provider_supported(CG.TypedValueFactoryProvider(CustomPacketFactory()))
    binder = TyVar(99704, 0, nothing)
    recursive = TyMu(binder, TyFun(FloatTy, binder))
    @test !CG._bounded_packet_request_supported(builtin,
        CG.GenerationQuery(recursive; scope_tys=Ty[], depth_left=3, node_budget=12))
    # Qualification is not root construction: no provider budget is created.
    @test isempty(query.requirements)
    @test isempty(builtin.evidence_worlds)
end

@testitem "CanonicalGeneration frontier keeps initial and replacement guarded certificates atomic" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.GuardedExpansionProvider()))
    scope = Ty[FloatTy, BoolTy]
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
        guarded_left=1, node_budget=4, requirements=(CG.UsesScopeBinder(0),))
    options = CG._guarded_sampling_options(relation, FloatTy, scope)
    option = only(filter(option -> option.value.head.name == :COND, options))
    # Obtain both certificates through the real selected guarded sampler,
    # rather than asserting policy preservation for a hand-written term.
    function sample_certificate(seed)
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 4)
        result = CG._sample_option(state, relation, option, scope, 3, 1, false, query)
        @test result isa CG.HeadApplicationDerivation
        @test CG.derivation_node_count(result) == 4
        @test state.nodes_left == 0
        @test CG._frontier_certificate_matches(result, query)
        return result
    end
    initial, replacement = sample_certificate(31), sample_certificate(47)
    for candidate in (nothing, replacement)
        calls = Ref(0)
        prepare = (relation, query, rng) -> begin
            calls[] += 1
            calls[] == 1 || error("guarded interior was opened without its parameter policy")
            candidate
        end
        state = CG.DerivationSamplingState(Random.Xoshiro(2000), 8,
            CG.GenerationCompletionState(probability=1, grace_steps=3))
        result = CG._sample_certified_frontier!(state, relation, query, initial; prepare)
        @test result === (candidate === nothing ? initial : replacement)
        @test CG._guarded_depth(result) == 1
        @test CG.derivation_uses_scope_binder(result, 0)
        @test calls[] == 1
        @test state.completion.boundaries == 1
        @test state.nodes_left == 4
    end
end

"""Prepare one selected ordinary/joint core step with complete child replay.

The query anchors closed types and exact concrete binder obligations. This
builder never recursively samples its child evidence: only the selected edge
is replayed. Mixed-positive preparation is a bounded optional alternative;
missing core evidence is not a mixed-negative decision.
"""
function _prepare_frontier_core_candidate(relation::CoreRelation, q::GenerationQuery, rng)
    relation, q = _relation_for_query(relation, q)
    _frontier_prepare_supported(relation, q) || return nothing
    scope, forbidden = _scope_type_vector(q.scope_tys), _forbidden_scope_binders(q)
    if !isempty(_required_scope_binder_indices(q, length(scope)))
        prepared = _prepare_generation_obligations(relation, q)
        prepared isa _UnsupportedEvidenceSampling && return nothing
        prepared === nothing && return nothing
        (; graph, root) = prepared
        ordinary, mask = graph.facts[root]
        if mask != _obligation_zero(mask)
            options = NamedTuple[]
            for edgeid in graph.options[root]
                edge = graph.edges[edgeid]
                edgecost = _obligation_edge_cost(graph, edge)
                edgecost === nothing && continue
                edgecost <= q.node_budget || continue
                if edge.root_binder < 0 && edge.plan.kind == :scope_application
                    for (slot, ty) in enumerate(scope)
                        slot - 1 in forbidden && continue
                        _evidence_type_id(graph.world, ty) == _evidence_type_id(graph.world, edge.plan.value) || continue
                        push!(options, (edge=edge, binder=slot - 1, weight=edge.plan.weight))
                    end
                else
                    push!(options, (edge=edge, binder=edge.root_binder, weight=edge.plan.weight))
                end
            end
            pool, total = _positive_option_pool(options)
            isempty(pool) && return nothing
            chosen = pool[_weighted_option_index(rng, pool, total)]
            edge, plan = chosen.edge, chosen.edge.plan
            lambda = plan.kind == :lambda
            childscope = lambda ? Ty[plan.value; scope] : scope
            childforbid = lambda ? _shift_scope_binders(forbidden, 1) : forbidden
            args = CoreDerivation[_obligation_replay(graph, child, childscope, childforbid, Int(lambda)) for child in edge.children]
            return _obligation_compose(plan, chosen.binder, args, scope)
        end
        world, fact = graph.world, ordinary
    else
        prepared = _prepare_finite_core_certificate(relation, q.target_ty, scope,
            q.depth_left, q.node_budget; forbidden)
        prepared isa _UnsupportedEvidenceSampling && return nothing
        prepared === nothing && return nothing
        (; world, fact) = prepared
    end
    steps = _evidence_steps!(world, relation, fact)
    world.unsupported_sampling && return nothing
    options = _evidence_concrete_options(world, steps, scope, forbidden)
    filter!(option -> option.plan.charge + length(option.plan.children) <= q.node_budget, options)
    pool, total = _positive_option_pool(options)
    isempty(pool) && return nothing
    chosen = pool[_weighted_option_index(rng, pool, total)]
    plan = _evidence_prepare_step!(world, relation, chosen.plan)
    world.unsupported_sampling && return nothing
    all(child -> child != 0 && world.certificates[child] != 0, plan.children) || return nothing
    _evidence_plan_cost(world, plan) <= q.node_budget || return nothing
    lambda = plan.kind == :lambda
    childscope = lambda ? Ty[plan.value; scope] : scope
    childforbid = lambda ? _shift_scope_binders(forbidden, 1) : forbidden
    args = CoreDerivation[_evidence_replay(world, child, childscope, childforbid) for child in plan.children]
    return _obligation_compose(plan, chosen.binder, args, scope)
end

function _prepare_frontier_candidate(relation::CoreRelation, q::GenerationQuery, rng)
    relation, q = _relation_for_query(relation, q)
    _frontier_prepare_supported(relation, q) || return nothing
    # The root controller owns all switches. A temporary, finite node lease is
    # preparation only; it shares RNG and the query's attempt-ledger identity.
    mixed_first = _safe_list_access_enabled(relation) && rand(rng, Bool)
    if mixed_first
        temporary = DerivationSamplingState(rng, q.node_budget)
        mixed = _sample_certified_required_candidate(temporary, relation, q, true)
        mixed !== nothing && return mixed
    end
    core = _prepare_frontier_core_candidate(relation, q, rng)
    core !== nothing && return core
    # Core absence is not mixed absence. Try the other finite preparer at most
    # once, never reopen a provider filter or the root completion state.
    if !mixed_first && _safe_list_access_enabled(relation)
        temporary = DerivationSamplingState(rng, q.node_budget)
        return _sample_certified_required_candidate(temporary, relation, q, true)
    end
    return nothing
end

@testitem "CanonicalGeneration certified frontier preserves prefix and pending siblings on switch" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:ADD => +),
        global_consts_ty=Dict{Symbol,Any}(:ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy], depth_left=2,
        node_budget=3, requirements=(CG.UsesScopeBinder(1),))
    head = only(filter(r -> r.head.name == :ADD, CG._certified_mixed_head_rules(relation, q))).head
    certificate = CG.HeadApplicationDerivation(head,
        CG.CoreDerivation[CG.VariableDerivation(0, FloatTy), CG.VariableDerivation(1, FloatTy)])
    calls = Ref(0)
    prepare = (relation, query, rng) -> begin
        calls[] += 1
        calls[] == 1 || error("preparation ran after completion")
        nothing
    end
    mode = CG.GenerationCompletionState(probability=1, grace_steps=1)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 8, mode)
    result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
    @test result isa CG.HeadApplicationDerivation
    @test result.head === head
    @test result.args[1] === certificate.args[1]
    @test result.args[2] === certificate.args[2]
    @test CG.derivation_uses_scope_binder(result, 1)
    @test CG.derivation_node_count(result) == 3
    @test state.nodes_left == 5 # outside lease retained; base charged once
    @test calls[] == 1
    @test mode.completing
    @test mode.boundaries == 2 # the sibling shares, not resets, the switch
end

@testitem "CanonicalGeneration frontier retains both actual use allocations" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:ADD => +, :ZERO => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
            :ZERO => FloatTy), providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=2,
        node_budget=3, requirements=(CG.UsesScopeBinder(0),))
    heads = CG._finite_evidence_world(relation, FloatTy, Ty[FloatTy]; sampling=true).catalog.heads
    add = only(filter(head -> head.name == :ADD, heads))
    zero = CG.ConstantDerivation(only(filter(head -> head.name == :ZERO, heads)))
    x = CG.VariableDerivation(0, FloatTy)
    certificate = CG.HeadApplicationDerivation(add, CG.CoreDerivation[x, x])
    prepare = (relation, query, rng) -> query.depth_left == 2 ? nothing :
        (CG.UsesScopeBinder(0) in query.requirements ? x : zero)
    left_zero, right_zero = Ref(false), Ref(false)
    for seed in 1:16
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 3)
        result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
        @test result isa CG.HeadApplicationDerivation
        @test CG.derivation_uses_scope_binder(result, 0)
        @test CG.derivation_node_count(result) == 3
        @test state.nodes_left == 0
        left_zero[] |= result.args[1] === zero && result.args[2] === x
        right_zero[] |= result.args[1] === x && result.args[2] === zero
    end
    @test left_zero[] # ADD(ZERO,x) must not be lost to incidental left use
    @test right_zero[]
end

@testitem "CanonicalGeneration frontier assigns call only to actual called identities" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    fn = TyFun(FloatTy, FloatTy)
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:APPLY => ((f, x) -> f(x))),
        global_consts_ty=Dict{Symbol,Any}(:APPLY => TyFun(fn, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[fn, FloatTy, fn], depth_left=3,
        node_budget=4, requirements=(CG.CallsScopeBinder(0), CG.ForbidsScopeBinder(2)))
    head = only(filter(r -> r.head.name == :APPLY, CG._certified_mixed_head_rules(relation, q))).head
    mention = CG.VariableDerivation(0, fn)
    call = CG.ScopeApplicationDerivation(0, fn, CG.CoreDerivation[CG.VariableDerivation(1, FloatTy)])
    d = CG.HeadApplicationDerivation(head, CG.CoreDerivation[mention, call])
    for seed in 1:4
        queries = CG._frontier_argument_queries(q, d, Ty[fn, FloatTy], 3, Random.Xoshiro(seed))
        @test !(CG.CallsScopeBinder(0) in queries[1].requirements)
        @test CG.CallsScopeBinder(0) in queries[2].requirements
        @test all(child -> CG.ForbidsScopeBinder(2) in child.requirements, queries)
        @test CG._frontier_certificate_matches(mention, queries[1])
        @test CG._frontier_certificate_matches(call, queries[2])
    end
end

@testitem "CanonicalGeneration frontier retains both actual proxy guarantees" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:LIST => [1.0], :P => ((xs, x) -> x)),
        global_consts_ty=Dict{Symbol,Any}(:LIST => list, :P => TyFun(list, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=4, node_budget=8)
    heads = CG._finite_evidence_world(relation, FloatTy, Ty[FloatTy]; sampling=true).catalog.heads
    source = CG.ConstantDerivation(only(filter(head -> head.name == :LIST, heads)))
    p = only(filter(head -> head.name == :P, heads))
    body = CG.HeadApplicationDerivation(p,
        CG.CoreDerivation[CG.VariableDerivation(0, list), CG.VariableDerivation(1, FloatTy)])
    certificate = CG.SafeListAccessDerivation(source, list, body,
        CG.VariableDerivation(0, FloatTy), FloatTy)
    costs = IdDict{CG.CoreDerivation,Int}()
    item = CG._frontier_item(costs, q, certificate)
    selected = Set{Int}()
    for seed in 1:16
        step = CG._frontier_step(relation, item, costs, Random.Xoshiro(seed))
        child = only(step.children)
        guarantee = only(filter(r -> r isa CG.UsesScopeBinder, child.query.requirements))
        push!(selected, guarantee.index)
        @test CG._frontier_certificate_matches(body, child.query)
        @test step.seed.source === source
        @test step.seed.fallback === certificate.fallback
        @test step.charge + child.cost == 8
    end
    @test selected == Set((0, 1))
end

@testitem "CanonicalGeneration frontier releases siblings without retaining temporary ceilings" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:F => identity, :G => identity, :SELECT => ((x, b) -> x)),
        global_consts_ty=Dict{Symbol,Any}(:F => TyFun(FloatTy, FloatTy),
            :G => TyFun(BoolTy, BoolTy), :SELECT => TyFun(FloatTy, TyFun(BoolTy, FloatTy))),
        providers=(CG.ContextHeadProvider(),))
    scope = Ty[FloatTy, BoolTy]
    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=9, node_budget=11)
    heads = CG._finite_evidence_world(relation, FloatTy, scope; sampling=true).catalog.heads
    f = only(filter(head -> head.name == :F, heads))
    g = only(filter(head -> head.name == :G, heads))
    select = only(filter(head -> head.name == :SELECT, heads))
    function wrapped(head, leaf, count)
        d = leaf
        for _ in 1:count
            d = CG.HeadApplicationDerivation(head, CG.CoreDerivation[d])
        end
        return d
    end
    x, b = CG.VariableDerivation(0, FloatTy), CG.VariableDerivation(1, BoolTy)
    certificate = CG.HeadApplicationDerivation(select,
        CG.CoreDerivation[wrapped(f, x, 4), wrapped(g, b, 4)])
    wanted_left, cheaper_right = wrapped(f, x, 5), wrapped(g, b, 2)
    wanted = CG.HeadApplicationDerivation(select,
        CG.CoreDerivation[wrapped(f, x, 6), cheaper_right])
    leases = Int[]
    prepare = (relation, query, rng) -> begin
        if query.target_ty === BoolTy && query.depth_left == 8
            return cheaper_right
        elseif query.target_ty === FloatTy && query.depth_left == 7
            push!(leases, query.node_budget)
            return query.node_budget >= 6 ? wanted_left : nothing
        end
        return nothing
    end
    found = Ref(false)
    for seed in 1:16
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 11)
        result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
        @test result isa CG.HeadApplicationDerivation
        @test result.head === select
        @test result.args[1].head === f # the already committed prefix remains
        @test CG.derivation_signature(result.args[2]) == CG.derivation_signature(cheaper_right)
        @test CG.derivation_node_count(result) <= 11
        @test state.nodes_left == 11 - CG.derivation_node_count(result)
        found[] |= CG.derivation_signature(result) == CG.derivation_signature(wanted)
    end
    @test any(lease -> lease >= 6, leases)
    @test found[] # costs (5,5) -> (7,3), still within the same root budget
end

@testitem "CanonicalGeneration certified frontier rejects over-budget preparation without retry" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:ADD => +),
        global_consts_ty=Dict{Symbol,Any}(:ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=2, node_budget=3)
    head = only(filter(r -> r.head.name == :ADD, CG._certified_mixed_head_rules(relation, q))).head
    variable = CG.VariableDerivation(0, FloatTy)
    certificate = CG.HeadApplicationDerivation(head, CG.CoreDerivation[variable, variable])
    calls, leases = Ref(0), Int[]
    prepare = (relation, query, rng) -> begin
        calls[] += 1
        push!(leases, query.node_budget)
        calls[] == 1 ? nothing : certificate
    end
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 8)
    result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
    @test CG.derivation_signature(result) == CG.derivation_signature(certificate)
    @test leases == [3, 1, 1]
    @test calls[] == 3
    @test state.completion.boundaries == 3
    @test !state.completion.completing
    @test state.nodes_left == 5 # a proposed child cannot borrow this outside surplus
    poor = CG.DerivationSamplingState(Random.Xoshiro(2000), 2)
    @test CG._sample_certified_frontier!(poor, relation, q, certificate; prepare) === nothing
    @test poor.nodes_left == 2
    @test calls[] == 3
end

@testitem "CanonicalGeneration certified frontier rebases call and forbidden identities" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    fn = TyFun(FloatTy, FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(fn; scope_tys=Ty[FloatTy, fn], depth_left=2,
        node_budget=6, requirements=(CG.CallsScopeBinder(1), CG.ForbidsScopeBinder(0)))
    body = CG.ScopeApplicationDerivation(2, fn, CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)])
    certificate = CG.LambdaDerivation(FloatTy, body)
    queries = CG.GenerationQuery[]
    prepare = (relation, query, rng) -> begin
        push!(queries, query)
        nothing
    end
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 6)
    result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
    @test CG.derivation_calls_scope_binder(result, 1)
    @test !CG.derivation_uses_scope_binder(result, 0)
    @test queries[2].scope_tys == (FloatTy, FloatTy, fn)
    @test CG.CallsScopeBinder(2) in queries[2].requirements
    @test CG.ForbidsScopeBinder(1) in queries[2].requirements
    @test !(CG.CallsScopeBinder(2) in queries[3].requirements) # head itself guarantees call
    @test CG.ForbidsScopeBinder(1) in queries[3].requirements
    @test state.nodes_left == 3
    invalid = CG.LambdaDerivation(FloatTy, CG.VariableDerivation(1, FloatTy))
    @test !CG._frontier_certificate_matches(invalid, q)
end

@testitem "CanonicalGeneration certified frontier locks safe supports and retains proxy OR" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    fn, list = TyFun(FloatTy, FloatTy), TyList(FloatTy)
    relation = CG.CoreRelation(global_consts=Dict{Symbol,Any}(:LIST => [1.0]),
        global_consts_ty=Dict{Symbol,Any}(:LIST => list),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[fn, FloatTy, FloatTy], depth_left=3,
        node_budget=8, requirements=(CG.CallsScopeBinder(0), CG.ForbidsScopeBinder(2)))
    world = CG._finite_evidence_world(relation, list, CG._scope_type_vector(q.scope_tys); sampling=true)
    sourcehead = only(filter(head -> head.name == :LIST, world.catalog.heads))
    source = CG.ConstantDerivation(sourcehead)
    fallback = CG.ScopeApplicationDerivation(0, fn, CG.CoreDerivation[CG.VariableDerivation(1, FloatTy)])
    body = CG.VariableDerivation(1, FloatTy)
    certificate = CG.SafeListAccessDerivation(source, list, body, fallback, FloatTy)
    queries = CG.GenerationQuery[]
    prepare = (relation, query, rng) -> begin
        push!(queries, query)
        nothing
    end
    state = CG.DerivationSamplingState(Random.Xoshiro(1), 8)
    result = CG._sample_certified_frontier!(state, relation, q, certificate; prepare)
    @test result.source === source
    @test result.fallback === fallback
    @test CG.derivation_calls_scope_binder(result, 0)
    @test !CG.derivation_uses_scope_binder(result, 2)
    @test length(queries) == 2
    child = queries[2]
    @test CG.UsesScopeBinder(1) in child.requirements
    @test !(CG.UsesScopeBinder(0) in child.requirements)
    @test !(CG.CallsScopeBinder(2) in child.requirements) # guaranteed by fixed fallback
    @test CG.ForbidsScopeBinder(4) in child.requirements
    @test :safe_list_access in child.capability_mask
    @test child.depth_left == 1
    @test state.nodes_left == 1
end

@testitem "FunctionDef forced public root exposes optional certified completion without filter restart" begin
    using Test, Random
    using LambdaRegression.TypeSystem: TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    calls = Ref(0)
    accept = candidate -> (calls[] += 1; true)
    provider = CG.FunctionDefProvider(admissibility=accept, body_attempts=4)
    config(seed) = CG.GenerationConfig(rng=Random.Xoshiro(seed),
        providers=(CG.ContextHeadProvider(),), sampling_depth_limit=3, sampling_node_budget=12)
    result = CG.generate_functiondef_root_scoped(target; config=config(31),
        functiondef_provider=provider, completion_probability=1)
    @test result isa CG.GenerationResult
    @test result.metadata[:sampling_nodes_used] == 4
    @test result.metadata[:certified_completion].switched
    @test result.metadata[:certified_completion].boundaries == 1
    @test result.metadata[:functiondef_verification] === :verified
    @test calls[] == 1
    # DirectKnotFilter rejects eta-forwarding. Missing acceptance evidence is
    # NOT repaired by another body attempt or resetting the completion mode.
    rejected = Ref(0)
    reject = candidate -> (rejected[] += 1; false)
    @test CG.generate_functiondef_root_scoped(target; config=config(31),
        functiondef_provider=CG.FunctionDefProvider(admissibility=reject, body_attempts=4),
        completion_probability=1) isa LambdaRegression.Utils.Err
    @test rejected[] == 1
    for _ in 1:2 # cacheable relation path, including a subsequent hit
        @test CG.generate_functiondef_root_scoped(target; config=config(31),
            functiondef_provider=CG.FunctionDefProvider(admissibility=CG.DirectKnotFilter()),
            completion_probability=1) isa LambdaRegression.Utils.Err
    end
    @test_throws ArgumentError CG.generate_functiondef_root_scoped(target; completion_probability=NaN)
    @test_throws ArgumentError CG.generate_functiondef_root_scoped(target; completion_probability=-0.1)
    @test_throws ArgumentError CG.generate_functiondef_root_scoped(target; completion_grace_steps=-1)
    legacy_config, zero_config = config(47), config(47)
    legacy = CG.generate_functiondef_root_scoped(target; config=legacy_config, functiondef_provider=provider)
    explicit_zero = CG.generate_functiondef_root_scoped(target; config=zero_config,
        functiondef_provider=provider, completion_probability=0, completion_grace_steps=10)
    @test typeof(legacy) === typeof(explicit_zero)
    @test rand(legacy_config.rng) == rand(zero_config.rng)
    if legacy isa CG.GenerationResult
        @test legacy.tree.root == explicit_zero.tree.root
        @test legacy.metadata[:sampling_nodes_used] == explicit_zero.metadata[:sampling_nodes_used]
        @test !haskey(explicit_zero.metadata, :certified_completion)
    end
end

@testitem "CanonicalGeneration frontier preparation preserves unsupported constraints and shared attempts" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    struct UnknownConstraint <: CG.AbstractGenerationRequirement end
    unknown = UnknownConstraint()
    q = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=1,
        node_budget=4, requirements=(unknown,))
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    @test CG._prepare_frontier_core_candidate(relation, q, Random.Xoshiro(2000)) === nothing
    @test CG._prepare_frontier_candidate(relation, q, Random.Xoshiro(2000)) === nothing
    certificate = CG.VariableDerivation(0, FloatTy)
    # The caller has certified an unknown constraint; this driver may retain
    # the complete atom, but must not invent an independent child contract.
    item = CG._frontier_item(IdDict{CG.CoreDerivation,Int}(), q, certificate)
    step = CG._frontier_step(relation, item, IdDict{CG.CoreDerivation,Int}(), Random.Xoshiro(2000))
    @test isempty(step.children)
    @test step.seed === certificate
    custom = CG.CoreRelation(providers=(CG.ContextHeadProvider(),), argument_planner=identity)
    plain = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy], depth_left=1, node_budget=4)
    @test CG._prepare_frontier_core_candidate(custom, plain, Random.Xoshiro(2000)) === nothing

    provider = CG._require_genuine_self_call(CG.FunctionDefProvider(admissibility=candidate -> false,
        body_attempts=4))
    target = TyFun(FloatTy, FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), provider))
    budget = CG.FunctionDefAttemptBudget(provider.route_token, fill(4, provider.max_nesting))
    rootq = CG.GenerationQuery(target; depth_left=3, provider_fuel=3,
        node_budget=12, requirements=(budget,))
    options = NamedTuple[]
    CG.add_generation_options!(provider, options, relation, rootq)
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 12,
        CG.GenerationCompletionState(probability=1))
    @test CG._sample_certified_functiondef_root(state, relation, provider, options, rootq) === nothing
    @test budget.remaining_by_level == [4, 3]
    @test state.completion.completing
    @test state.nodes_left == 8 # filter rejection does not re-prove/replenish the request
    @test CG._sample_certified_functiondef_root(state, relation, provider, options, rootq) === nothing
    @test budget.remaining_by_level == [4, 3] # no search after inherited switch
end

@testitem "CanonicalGeneration frontier joint menu rejects unproved edges and retains zero masks" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:TRUE => true, :G => (x -> x > 0),
            :H => identity, :G2 => ((x, b) -> x > 0 && b)),
        global_consts_ty=Dict{Symbol,Any}(:TRUE => BoolTy, :G => TyFun(FloatTy, BoolTy),
            :H => TyFun(BoolTy, BoolTy), :G2 => TyFun(FloatTy, TyFun(BoolTy, BoolTy))),
        providers=(CG.ContextHeadProvider(),), weights=CG.GenerationWeights(H=1e6))
    q = CG.GenerationQuery(BoolTy; scope_tys=Ty[FloatTy], depth_left=1,
        node_budget=12, requirements=(CG.UsesScopeBinder(0),))
    prepared = CG._prepare_generation_obligations(relation, q)
    @test prepared !== nothing
    graph = prepared.graph
    bad = [graph.edges[id] for id in graph.options[prepared.root]
        if graph.edges[id].plan.kind == :head && graph.edges[id].plan.value.name == :H]
    @test !isempty(bad)
    @test all(edge -> CG._obligation_edge_cost(graph, edge) === nothing, bad)
    @test any(edge -> any(child -> graph.costs[child] == typemax(Int), edge.children), bad)
    good = [graph.edges[id] for id in graph.options[prepared.root]
        if graph.edges[id].plan.kind == :head && graph.edges[id].plan.value.name == :G2 &&
            CG._obligation_edge_cost(graph, graph.edges[id]) !== nothing]
    @test !isempty(good)
    @test any(good) do edge
        any(edge.children) do child
            ordinary, mask = graph.facts[child]
            mask == CG._obligation_zero(mask) && graph.certificates[child] == 0 &&
                graph.world.certificates[ordinary] != 0
        end
    end
    edge = first(good)
    oldcost = graph.costs[first(edge.children)]
    try
        graph.costs[first(edge.children)] = typemax(Int) - 1
        @test CG._obligation_edge_cost(graph, edge) === nothing # finite-cost addition overflows
        graph.costs[first(edge.children)] = typemax(Int) - 2
        @test CG._obligation_edge_cost(graph, edge) === nothing # exact sentinel total is not evidence
    finally
        graph.costs[first(edge.children)] = oldcost
    end
    for seed in (31, 47, 53, 101)
        result = CG._prepare_frontier_core_candidate(relation, q, Random.Xoshiro(seed))
        @test result isa CG.HeadApplicationDerivation
        @test result.head.name != :H
        @test CG.derivation_uses_scope_binder(result, 0)
        @test CG.derivation_result_type(result) === BoolTy
        @test CG.derivation_node_count(result) <= 12
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 12)
        ordinary_consumer = CG._sample_finite_required_derivation(state, relation, q)
        @test ordinary_consumer isa CG.CoreDerivation
        @test CG.derivation_uses_scope_binder(ordinary_consumer, 0)
    end
end

@testitem "FunctionDef cached singleton policy keeps owned route and fresh root budgets" begin
    using Test, Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun
    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    observed_route_ids = UInt64[]
    struct SingletonRecordingAccept end
    function (::SingletonRecordingAccept)(candidate, relation, query)
        budget = only(filter(requirement -> requirement isa CG.FunctionDefAttemptBudget,
            query.requirements))
        push!(observed_route_ids, budget.route_token.id)
        return true
    end
    global_consts = Dict{Symbol,Any}(:frontier_owned_route_probe => false)
    global_consts_ty = Dict{Symbol,Any}(:frontier_owned_route_probe => BoolTy)

    make_provider() = CG.FunctionDefProvider(
        admissibility=CG.GenuineSelfCallFilter(SingletonRecordingAccept()),
        max_nesting=7,
        body_attempts=1,
    )
    make_config() = CG.GenerationConfig(
        rng=Random.Xoshiro(2000),
        providers=(CG.ContextHeadProvider(),),
        sampling_depth_limit=3,
        sampling_node_budget=12,
    )

    function cached_entry(provider, config)
        effective = CG.with_functiondef_provider(config, provider)
        empty_values = Dict{Symbol,Any}()
        empty_types = Dict{Symbol,Any}()
        key = CG._scoped_lazy_relation_cache_key(
            effective.weights,
            global_consts,
            global_consts_ty,
            empty_values,
            empty_types,
            empty_values,
            empty_types,
            effective.providers,
            effective.argument_planner,
            effective.guarded_expansion_depth,
        )
        @test key !== nothing
        return CG._get_scoped_lazy_relation_cache_entry(
            key,
            effective.weights,
            effective.providers,
            effective.argument_planner,
            effective.guarded_expansion_depth,
            global_consts,
            global_consts_ty,
            empty_values,
            empty_types,
        )
    end

    first_provider, second_provider = make_provider(), make_provider()
    @test CG.generation_provider_cache_key(first_provider) !== nothing
    @test CG.generation_provider_cache_key(first_provider) ==
        CG.generation_provider_cache_key(second_provider)
    @test first_provider.route_token !== second_provider.route_token

    first = CG.generate_functiondef_root_scoped(target;
        config=make_config(), functiondef_provider=first_provider,
        global_consts, global_consts_ty, completion_probability=0)
    @test first isa CG.GenerationResult
    @test first.metadata[:forced_functiondef_root]
    @test first.metadata[:sampling_nodes_used] == 4
    @test first.metadata[:functiondef_verification] === :verified
    @test observed_route_ids == [first_provider.route_token.id]

    entry = cached_entry(first_provider, make_config())
    owned = only(filter(provider -> provider isa CG.FunctionDefProvider, entry.relation.providers))
    @test owned === first_provider
    @test owned.route_token === first_provider.route_token

    second = CG.generate_functiondef_root_scoped(target;
        config=make_config(), functiondef_provider=second_provider,
        global_consts, global_consts_ty, completion_probability=0)
    @test second isa CG.GenerationResult
    @test second.metadata[:forced_functiondef_root]
    @test second.metadata[:sampling_nodes_used] == 4
    @test second.metadata[:functiondef_verification] === :verified
    @test observed_route_ids == [first_provider.route_token.id, first_provider.route_token.id]
    @test cached_entry(second_provider, make_config()) === entry
    @test only(filter(provider -> provider isa CG.FunctionDefProvider, entry.relation.providers)) === owned
    @test owned.route_token !== second_provider.route_token

    certified = CG.generate_functiondef_root_scoped(target;
        config=make_config(), functiondef_provider=make_provider(),
        global_consts, global_consts_ty, completion_probability=1)
    @test certified isa CG.GenerationResult
    @test certified.metadata[:sampling_nodes_used] == 4
    @test certified.metadata[:functiondef_verification] === :verified
    @test observed_route_ids == fill(first_provider.route_token.id, 3)

    query = CG.GenerationQuery(target; depth_left=3, provider_fuel=3, node_budget=12)
    first_budget = only(CG.generation_root_requirements(owned, entry.relation, query))
    next_budget = only(CG.generation_root_requirements(owned, entry.relation, query))
    @test first_budget.route_token === owned.route_token
    @test next_budget.route_token === owned.route_token
    @test first_budget !== next_budget
    @test first_budget.remaining_by_level == fill(1, owned.max_nesting)
    @test next_budget.remaining_by_level == fill(1, owned.max_nesting)
end

@testitem "CanonicalGeneration extension option cache snapshots shared attempt budgets" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyFun
    CG = LambdaRegression.CanonicalGeneration

    struct FrontierBudgetMenuAnchor <: CG.AbstractHeadProvider end
    CG.add_heads!(::FrontierBudgetMenuAnchor, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::FrontierBudgetMenuAnchor) = true
    CG.generation_provider_cache_key(::FrontierBudgetMenuAnchor) = (:frontier_budget_menu_anchor,)
    function CG.add_generation_options!(::FrontierBudgetMenuAnchor, options, relation, query)
        push!(options, (weight=1.0, token=:anchor))
        return options
    end

    target = TyFun(FloatTy, FloatTy)
    provider = CG.FunctionDefProvider(
        admissibility=CG.AcceptFunctionDef(),
        max_nesting=2,
        body_attempts=1,
    )
    anchor = FrontierBudgetMenuAnchor()
    relation = CG.CoreRelation(providers=(anchor, provider))
    exhausted_budget = CG.FunctionDefAttemptBudget(provider.route_token, [1, 0])
    fresh_budget = CG.FunctionDefAttemptBudget(provider.route_token, [1, 1])
    make_query(budget) = CG.GenerationQuery(target; scope_tys=Ty[target],
        depth_left=2, provider_fuel=2, node_budget=8,
        requirements=(CG.CallsScopeBinder(0), budget))
    exhausted_query = make_query(exhausted_budget)
    fresh_query = make_query(fresh_budget)

    @test exhausted_query.target_ty === fresh_query.target_ty
    @test exhausted_query.scope_tys == fresh_query.scope_tys
    @test exhausted_query.depth_left == fresh_query.depth_left
    @test exhausted_query.provider_fuel == fresh_query.provider_fuel
    @test exhausted_budget.route_token === fresh_budget.route_token
    @test fresh_budget.route_token === provider.route_token

    exhausted_options = CG._generation_extension_options(relation, exhausted_query)
    @test length(exhausted_options) == 1
    @test only(exhausted_options).value.provider === anchor
    @test !any(option -> option.value.provider === provider, exhausted_options)
    exhausted_key = CG._generation_query_cache_key(
        exhausted_query, relation.generation_providers)
    @test exhausted_key !== nothing

    fresh_key = CG._generation_query_cache_key(fresh_query, relation.generation_providers)
    @test fresh_key !== nothing
    @test fresh_key != exhausted_key
    fresh_options = CG._generation_extension_options(relation, fresh_query)
    @test fresh_options !== exhausted_options
    @test any(option -> option.value.provider === anchor, fresh_options)
    fresh_extension = only(filter(option -> option.value.provider === provider, fresh_options))
    @test fresh_extension.value.option.remaining == 2
    @test length(relation.sampling_options) == 2

    # The required-scope entry point must observe the same budget snapshots as
    # its underlying lazy menu builder, including when a spent budget returns
    # to the already-cached exhausted state.
    required = (0,)
    scope = Ty[target]
    exhausted_scope_options = CG._required_scope_certificate_options(
        relation, target, scope, 2, required, exhausted_query)
    @test !any(option -> option.kind == :generation_extension &&
        option.value.provider === provider, exhausted_scope_options)
    @test CG._required_scope_lazy_certificate_options(
        relation, target, scope, 2, required, (), exhausted_query, true) ===
        exhausted_scope_options

    fresh_scope_options = CG._required_scope_certificate_options(
        relation, target, scope, 2, required, fresh_query)
    @test any(option -> option.kind == :generation_extension &&
        option.value.provider === provider, fresh_scope_options)
    @test CG._required_scope_lazy_certificate_options(
        relation, target, scope, 2, required, (), fresh_query, true) ===
        fresh_scope_options

    @test CG._consume_functiondef_attempt!(fresh_query, provider)
    @test fresh_budget.remaining_by_level == [1, 0]
    spent_key = CG._generation_query_cache_key(fresh_query, relation.generation_providers)
    @test spent_key == exhausted_key
    spent_options = CG._generation_extension_options(relation, fresh_query)
    @test spent_options === exhausted_options
    @test !any(option -> option.value.provider === provider, spent_options)

    spent_scope_options = CG._required_scope_certificate_options(
        relation, target, scope, 2, required, fresh_query)
    @test spent_scope_options === exhausted_scope_options
    @test !any(option -> option.kind == :generation_extension &&
        option.value.provider === provider, spent_scope_options)
end

@testitem "CanonicalGeneration cache type keys refine bounded closed trees and retain fallback identity" begin
    using Test
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyArray, TyConst
    using LambdaRegression.TypeSystem: TyFun, TyLike, TyList, TyMu, TyNat, TyScheme, TyVar
    CG = LambdaRegression.CanonicalGeneration
    TS = LambdaRegression.TypeSystem

    closed_pairs = (
        (TyFun(FloatTy, BoolTy), TyFun(TyConst(:Float), TyConst(:Bool)), true),
        (TyFun(FloatTy, BoolTy), TyFun(FloatTy, FloatTy), false),
        (TyList(FloatTy), TyList(TyConst(:Float)), true),
        (TyList(FloatTy), TyList(BoolTy), false),
        (TyArray(TyNat(4), FloatTy), TyArray(TyNat(4), TyConst(:Float)), true),
        (TyArray(TyNat(4), FloatTy), TyArray(TyNat(5), FloatTy), false),
    )
    for (left, right, expected_equal) in closed_pairs
        left_key, right_key = CG._cache_tylike_key(left), CG._cache_tylike_key(right)
        @test left_key[1] === :exact_closed_tree
        @test right_key[1] === :exact_closed_tree
        @test (left_key == right_key) == expected_equal
    end
    dummy_rigid = TyConst(:CanonicalGenerationRigidDummy9300)
    @test CG._cache_tylike_key(dummy_rigid) == (:exact_closed_tree, CG._cache_type_key(dummy_rigid))
    @test CG._cache_tylike_key(dummy_rigid) != CG._cache_tylike_key(TyConst(:CanonicalGenerationRigidDummy9301))

    changed_fun = TyFun(FloatTy, FloatTy)
    before_change = CG._cache_tylike_key(changed_fun)
    setfield!(changed_fun, :res, BoolTy)
    after_change = CG._cache_tylike_key(changed_fun)
    @test before_change != after_change
    @test after_change == CG._cache_tylike_key(TyFun(FloatTy, BoolTy))

    left_var, right_var = TyVar(9310, 0, nothing), TyVar(9311, 0, nothing)
    left_scheme = TyScheme([left_var], TyFun(left_var, TyList(left_var)))
    right_scheme = TyScheme([right_var], TyFun(right_var, TyList(right_var)))
    left_scheme_key, right_scheme_key = CG._cache_tylike_key(left_scheme), CG._cache_tylike_key(right_scheme)
    @test left_scheme_key[1] === :semantic_type
    @test left_scheme_key == right_scheme_key

    function make_period_type(labels, uid)
        binder = TyVar(uid, 0, nothing)
        body = binder
        for label in Iterators.reverse(labels)
            body = TyFun(label, body)
        end
        return TyMu(binder, body)
    end
    period_one = make_period_type([FloatTy], 9320)
    period_four = make_period_type(fill(FloatTy, 4), 9321)
    @test TS.isequal_mu(period_one, period_four)
    period_key = CG._cache_tylike_key(period_one)
    @test period_key[1] === :semantic_type
    @test period_key == CG._cache_tylike_key(period_four)

    long_list = let ty = FloatTy
        for _ in 1:64
            ty = TyList(ty)
        end
        ty
    end
    long_key = CG._cache_tylike_key(long_list)
    @test CG._grounded_closed_fragment_fuel(long_list, 64) == -1
    @test long_key[1] === :semantic_type
    @test long_key == (:semantic_type, TS.semantic_typekey(long_list))

    cyclic_list = TyList(FloatTy)
    cyclic_fun = TyFun(FloatTy, cyclic_list)
    setfield!(cyclic_list, :elem, cyclic_fun)
    cycle_key = CG._cache_tylike_key(cyclic_fun)
    @test CG._grounded_closed_fragment_fuel(cyclic_fun, 64) == -1
    @test cycle_key[1] === :semantic_type
    @test cycle_key == (:semantic_type, TS.semantic_typekey(cyclic_fun))

    open_var = TyVar(9330, 0, nothing)
    @test CG._grounded_closed_fragment_fuel(open_var, 64) == -1
    @test_throws ArgumentError CG._cache_tylike_key(open_var)

    local_values = Dict{Symbol,Any}(:probe => 1.0)
    mono_context = Dict{Symbol,TyLike}(:probe => FloatTy)
    changed_context = Dict{Symbol,TyLike}(:probe => BoolTy)
    scheme_context = Dict{Symbol,TyLike}(:probe => left_scheme)
    mono_key = CG._cacheable_local_context_key(local_values, mono_context)
    @test mono_key != CG._cacheable_local_context_key(local_values, changed_context)
    @test mono_key != CG._cacheable_local_context_key(local_values, scheme_context)
end

@testitem "FunctionDef certified root pays original extension depth and fuel" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    observed = Any[]
    accept = (candidate, relation, query) -> begin
        budget = only(filter(requirement -> requirement isa CG.FunctionDefAttemptBudget,
            query.requirements))
        push!(observed, (query=query, balances=copy(budget.remaining_by_level),
            route_token=budget.route_token))
        true
    end
    provider = CG.FunctionDefProvider(admissibility=accept, max_nesting=3, body_attempts=1)
    config(depth) = CG.GenerationConfig(rng=Random.Xoshiro(2000),
        providers=(CG.ContextHeadProvider(),), sampling_depth_limit=depth,
        sampling_node_budget=12)
    outer_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), provider))
    outer_query = CG.GenerationQuery(target; depth_left=3, provider_fuel=3, node_budget=12)
    outer_options = NamedTuple[]
    CG.add_generation_options!(provider, outer_options, outer_relation, outer_query)
    @test only(outer_options).remaining == 3 # the public option retains outer remaining
    # A syntactic self call under the function's lambda needs provider-body
    # depth >= 2. Dispatch consumes one depth before that body is constructed.
    for probability in (0.0, 1.0)
        @test CG.generate_functiondef_root_scoped(target; config=config(2),
            functiondef_provider=provider, completion_probability=probability) isa LambdaRegression.Utils.Err
    end
    @test isempty(observed)
    for probability in (0.0, 1.0)
        result = CG.generate_functiondef_root_scoped(target; config=config(3),
            functiondef_provider=provider, completion_probability=probability)
        @test result isa CG.GenerationResult
        @test result.metadata[:sampling_nodes_used] == 4
    end
    @test length(observed) == 2
    @test all(event -> event.query.depth_left == 2 && event.query.provider_fuel == 2, observed)
    @test all(event -> event.query.provider_fuel < provider.max_nesting, observed)
    @test all(event -> event.balances == [1, 0, 1], observed) # charge the effective level 2
    @test all(event -> event.route_token === provider.route_token, observed)
    # Insufficient provider fuel rejects before base/certificate preparation.
    forced = CG._require_genuine_self_call(provider)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), forced))
    q = CG.GenerationQuery(target; depth_left=3, provider_fuel=0, node_budget=12)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 12,
        CG.GenerationCompletionState(probability=1))
    @test CG._sample_certified_functiondef_root(state, relation, forced, NamedTuple[], q) === nothing
    @test state.nodes_left == 12
    @test !state.completion.completing
end
