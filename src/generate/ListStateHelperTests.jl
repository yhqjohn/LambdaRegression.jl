@testitem "CanonicalGeneration computed application preserves types identities and filters" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, FloatTy, BoolTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.Execution: evaluate_native
    CG = LambdaRegression.CanonicalGeneration
    identity = CG.LambdaDerivation(FloatTy, CG.VariableDerivation(0, FloatTy))
    one = CG.ConstantDerivation(CG.HeadEntry(:ONE, :constant, FloatTy,
        CG.CoreProductive(), CG.WeightBucket(:constant); metadata=(value=1.0,)))
    d = CG.ComputedApplicationDerivation(identity, CG.CoreDerivation[one])
    @test CG.derivation_node_count(d) == 4
    @test CG.derivation_result_type(d) == FloatTy
    @test CG.rigid_safe_derivation(d)
    @test CG.derivation_signature(CG._copy_generation_witness(d)) == CG.derivation_signature(d)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    values, types = Dict{Symbol,Any}(), Dict{Symbol,TyLike}()
    tree = AstTree(CG.materialize_generation_derivation(CG.DefaultAstMaterializer(), d,
        values, types, relation, Random.Xoshiro(555)))
    @test iscompatible(infer(tree, merge(relation.global_consts_ty, types)), FloatTy)
    @test evaluate_native(tree, merge(relation.global_consts, values)) == 1.0
    @test CG._array_source_depth(d) == typemax(Int)
    @test CG._guarded_depth(d) == 0
    @test_throws ArgumentError CG.ComputedApplicationDerivation(one, CG.CoreDerivation[one])
    @test_throws ArgumentError CG.ComputedApplicationDerivation(identity, CG.CoreDerivation[])
    bool = CG.ConstantDerivation(CG.HeadEntry(:TRUE, :constant, BoolTy,
        CG.CoreProductive(), CG.WeightBucket(:constant); metadata=(value=true,)))
    @test_throws ArgumentError CG.ComputedApplicationDerivation(identity, CG.CoreDerivation[bool])

    fun = TyFun(FloatTy, FloatTy)
    call = CG.ComputedApplicationDerivation(CG.VariableDerivation(1, fun),
        CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)])
    @test CG.derivation_uses_scope_binder(call, 1)
    @test CG.derivation_calls_scope_binder(call, 1)
    @test CG._signature_calls_scope_binder(CG.derivation_signature(call), 1)
    @test !CG.derivation_calls_scope_binder(call, 0)
    body = CG.LambdaDerivation(FloatTy, call)
    candidate = CG.FunctionDefCandidate(fun, body, identity)
    @test !CG.DirectKnotFilter()(candidate)
    @test CG._has_strict_saturated_self_call(candidate)
    @test CG._has_strict_saturated_self_call_signature(CG.derivation_signature(body), fun)
    @test CG._has_unchanged_self_call_derivation(candidate.body, 0)
    @test CG._has_unchanged_self_call_signature(CG.derivation_signature(call), 1, [0])
    list = LambdaRegression.TypeSystem.TyList(FloatTy)
    mutable_value = [1.0]
    mutable struct ListFactory
        value::Vector{Float64}
        calls::Int
    end
    (f::ListFactory)(head, relation, rng) = (f.calls += 1; f.value)
    mutable_factory = ListFactory(mutable_value, 0)
    h = CG.HeadEntry(:MUTABLE, :constant, list, CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=mutable_value, factory=mutable_factory))
    constant = CG.ConstantDerivation(h)
    listfun = TyFun(list, list)
    fnbody = CG.HeadApplicationDerivation(CG.HeadEntry(:FN, :constant, listfun,
        CG.CoreProductive(), CG.WeightBucket(:constant); metadata=(value=x->x,)),
        CG.CoreDerivation[constant])
    listidentity = CG.LambdaDerivation(list, CG.VariableDerivation(0, list))
    definition = CG.FunctionDefDerivation(listfun, CG.LambdaDerivation(list, fnbody),
        listidentity, 0, (:function_def,), 0)
    original = CG.ComputedApplicationDerivation(definition, CG.CoreDerivation[constant])
    copied = CG._copy_generation_witness(original)
    @test copied.args !== original.args
    @test copied.args[1] === constant
    @test copied.head.body.body.args !== fnbody.args
    @test copied.head.body.body.args[1].head === h
    @test copied.head.body.body.args[1].head.metadata.value === mutable_value
    @test copied.head.body.body.args[1].head.metadata.factory === mutable_factory
    @test mutable_factory.calls == 0
    factory_values, factory_types = Dict{Symbol,Any}(), Dict{Symbol,TyLike}()
    factory_tree = AstTree(CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
        copied, factory_values, factory_types, relation, Random.Xoshiro(2000)))
    @test mutable_factory.calls == 2 # once in the function body and once in its actual
    @test iscompatible(infer(factory_tree, factory_types), list)
end

@testitem "CanonicalGeneration List helper first body family mix is once only" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    helper = TyFun(FloatTy,TyFun(list,FloatTy))
    relation = CG.CoreRelation(weights=CG.GenerationWeights(list_recursive_body_probability=.95))
    scope = Ty[list,FloatTy,helper,list]
    requirements = (CG.FunctionDefSelfBinder(2,2,true),CG._ListStateHelperSelf(2,helper))
    query = CG.GenerationQuery(FloatTy;scope_tys=scope,depth_left=6,node_budget=32,requirements)
    safe(i) = (kind=:safe_list_access,
        value=(source=CG.ListAccessSource(list,CG.VariableDerivation(i,list)),target_ty=FloatTy),weight=1.0)
    ordinary = (kind=:variable,value=CG.VariableDerivation(1,FloatTy),weight=8.0)
    options = NamedTuple[safe(0),safe(3),ordinary]
    original = copy(options)
    weighted = CG._prefer_state_helper_body_options(options,relation,query)
    safemass(items) = sum(o.weight for o in items if o.kind===:safe_list_access;init=0.0)
    @test CG._at_state_helper_body_entry(query)
    @test sum(o.weight for o in weighted) ≈ 10
    @test safemass(weighted) ≈ 9.6
    @test weighted[1].weight ≈ 9.36 # family mix, then exact own-source mix
    @test weighted[2].weight ≈ .24
    @test weighted[3].weight ≈ .4
    @test all(o -> o.weight>0,weighted)
    @test options==original

    # Selection closes the existing prefix. Later scalar children still get
    # the independent source preference, but never the body-family boost.
    closed = CG._close_functiondef_prefix(query)
    later = CG._prefer_state_helper_body_options(options,relation,closed)
    @test !CG._at_state_helper_body_entry(closed)
    @test safemass(later) ≈ 2
    @test later[3] === ordinary
    nested = CG._advance_functiondef_prefix(CG._derive_generation_query(closed;added_scope_tys=(FloatTy,)))
    @test !CG._at_state_helper_body_entry(nested)
    unrelated = CG.GenerationQuery(FloatTy;scope_tys=scope,depth_left=6,node_budget=32,
        requirements=(CG.FunctionDefSelfBinder(2,2,true),))
    @test CG._prefer_state_helper_body_options(options,relation,unrelated) === options
    before_second = CG.GenerationQuery(TyFun(list,FloatTy);scope_tys=Ty[FloatTy,helper,list],
        depth_left=7,node_budget=32,
        requirements=(CG.FunctionDefSelfBinder(1,1,true),CG._ListStateHelperSelf(1,helper)))
    @test !CG._at_state_helper_body_entry(before_second)
    @test CG._prefer_state_helper_body_options(options,relation,nothing) === options
    zero = CG.CoreRelation(weights=CG.GenerationWeights(list_recursive_body_probability=0))
    @test CG._prefer_state_helper_body_options(options,zero,query) === options
    @test CG._prefer_state_helper_body_options(NamedTuple[ordinary],relation,query) == [ordinary]
    allsafe = NamedTuple[safe(0),safe(3)]
    @test CG._prefer_state_helper_body_options(allsafe,relation,query) ==
        CG._prefer_state_helper_list_sources(allsafe,relation,query)
    # A prospective certified menu is mixed while its choice query is open;
    # construction receives closed, not a rewritten/open descendant query.
    @test only(r for r in closed.requirements if r isa CG.FunctionDefSelfBinder).prefix_open == false
    @test only(r for r in nested.requirements if r isa CG.FunctionDefSelfBinder).leading_parameters == 2
end

@testitem "CanonicalGeneration List helper source policy and zero mode stay bounded" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    using LambdaRegression.Utils: Err
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    # The state and traversed-list formals deliberately have identical types.
    helper = TyFun(list, TyFun(list, list))
    weights = CG.GenerationWeights(list_state_helper=8, list_recursive_body_probability=.95)
    relation = CG.CoreRelation(; weights,
        providers=(CG.ContextHeadProvider(),CG.SafeListAccessProvider()))
    scope = Ty[list,list,helper,list]
    query = CG.GenerationQuery(list; scope_tys=scope, depth_left=5, node_budget=16,
        requirements=(CG.FunctionDefSelfBinder(2,2,false),CG._ListStateHelperSelf(2,helper)))
    CG._safe_list_access_generation_options(relation,list,scope) # populate concretely typed menu
    options = CG._safe_list_access_generation_options(relation,list,scope) # borrow stored Vector{NamedTuple}
    oldweights = getproperty.(options,:weight)
    @test length(options) == 3
    preferred = CG._prefer_state_helper_list_sources(options,relation,query)
    @test options !== preferred
    @test sum(o.weight for o in preferred) ≈ sum(oldweights)
    @test all(o -> o.weight > 0, preferred)
    @test getproperty.(options,:weight) == oldweights
    sourceindex(o) = o.value.source.derivation.index
    @test only(o.weight for o in preferred if sourceindex(o)==0) ≈ 2.9
    @test all(o -> o.weight ≈ .05, filter(o -> sourceindex(o)!=0,preferred))
    @test CG._safe_list_access_generation_options(relation,list,scope) === options
    blocked = CG._derive_generation_query(query; requirements=(CG.ForbidsScopeBinder(0),))
    @test CG._prefer_state_helper_list_sources(options,relation,blocked) === options
    unrelated = CG.GenerationQuery(list; scope_tys=scope, depth_left=5, node_budget=16)
    @test CG._prefer_state_helper_list_sources(options,relation,unrelated) === options
    shifted = CG.rebase_generation_requirement(CG._ListStateHelperSelf(2,helper),2)
    @test shifted.self_index == 4 && shifted.helper_ty === helper
    @test CG._has_list_state_helper_context(query)
    @test !CG._has_list_state_helper_context(unrelated)

    provider = CG.FunctionDefProvider(max_nesting=1,admissibility=CG.DirectKnotFilter())
    providers = (CG.ContextHeadProvider(),CG.SafeListAccessProvider(),provider)
    a,b = Random.Xoshiro(42),Random.Xoshiro(42)
    target = TyFun(list,FloatTy)
    function sample(rng,weights)
        CG.generate_canonical(target;global_consts=Dict{Symbol,Any}(:ZERO=>0.0),
            config=CG.GenerationConfig(;rng,weights,providers,
                sampling_depth_limit=3,sampling_node_budget=24))
    end
    plain,zero = sample(a,CG.GenerationWeights()),sample(b,CG.GenerationWeights(list_state_helper=0))
    @test typeof(plain)==typeof(zero)
    if plain isa Err
        @test typeof(plain.value)==typeof(zero.value)
    else
        @test plain.tree==zero.tree
        @test plain.local_consts==zero.local_consts
    end
    @test rand(a)==rand(b)
    for bad in (-1.0,Inf,NaN)
        @test_throws ArgumentError CG.GenerationWeights(list_state_helper=bad)
    end
    for kwargs in ((;sampler=CG.WeightedDerivationSampler(completion_probability=.1)),
                   (;sampler=(rng,set)->nothing), (;argument_planner=(args...)->nothing))
        @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(;weights,kwargs...))
    end

    # Initial evidence respects a forbidden only Float source and spends no FD
    # attempt. This also exercises helper-only seeding without an ordinary FD bucket.
    onlyhelper = CG.GenerationWeights(default=0,variable=1,lambda=1,
        function_def=0,list_state_helper=8)
    r = CG.CoreRelation(;weights=onlyhelper,providers)
    root = CG.GenerationQuery(FloatTy;scope_tys=Ty[list,FloatTy],depth_left=10,node_budget=48,
        requirements=(CG.ForbidsScopeBinder(1),))
    q = CG._derive_generation_query(root;requirements=CG.generation_root_requirements(provider,r,root))
    ledger = only(filter(x -> x isa CG.FunctionDefAttemptBudget,q.requirements))
    option = only(CG._generation_extension_options(r,q))
    state = CG.DerivationSamplingState(Random.Xoshiro(555),48)
    @test CG._sample_option(state,r,option,Ty[list,FloatTy],10,0,true,q) === nothing
    @test state.nodes_left==48
    @test only(ledger.remaining_by_level)==4
    @test !CG.generation_provider_applicable(provider,r,root) # missing real shared ledger
end

@testitem "CanonicalGeneration List helper real generation materializes and executes" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode
    using LambdaRegression.Execution: evaluate_native
    using LambdaRegression.Primitives: FADD
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    p = 1-eps(Float64)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, lambda=100,
        scope_application=1, function_def=0, list_state_helper=5, safe_list_access=100,
        proxy_constructive=3, proxy_general=1, constraint_evidence=0,
        functiondef_self_call_probability=p, list_recursive_body_probability=p,
        list_recursive_tail_probability=p, safe_list_short_fallback_probability=p)
    mutable struct HelperAudit
        bodies::Vector{CG.CoreDerivation}
    end
    function (a::HelperAudit)(candidate, args...)
        push!(a.bodies, candidate.body)
        return CG.GenuineSelfCallFilter(CG.DirectKnotFilter())(candidate, args...)
    end
    function CG._functiondef_sampling_admissible(a::HelperAudit, candidate::CG.FunctionDefCandidate)
        push!(a.bodies, candidate.body)
        return CG._functiondef_sampling_admissible(CG.GenuineSelfCallFilter(CG.DirectKnotFilter()), candidate)
    end
    audit = HelperAudit(CG.CoreDerivation[])
    provider = CG.FunctionDefProvider(max_nesting=1, body_attempts=4, admissibility=audit)
    relation = CG.CoreRelation(; weights,
        global_consts=Dict{Symbol,Any}(:ADD=>FADD),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider(), provider))
    scope = Ty[list, FloatTy]
    root = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=10, node_budget=96,
        requirements=(CG.UsesScopeBinder(0),))
    function construct_fixture()
        # Bounded construction may fail normally. This predeclared small set
        # verifies one real success, not a natural supply or discovery rate.
        for seed in (513, 514, 515, 516)
            query = CG._derive_generation_query(root;
                requirements=CG.generation_root_requirements(provider, relation, root))
            option = only(CG._generation_extension_options(relation, query))
            state = CG.DerivationSamplingState(Random.Xoshiro(seed), 96)
            result = CG._sample_option(state, relation, option, scope, 10, 0, true, query)
            result === nothing || return (; result, state, query, seed)
        end
        return nothing
    end
    fixture = construct_fixture()
    @test fixture !== nothing
    if fixture !== nothing
        result, state, query = fixture.result, fixture.state, fixture.query
        @test result isa CG.ComputedApplicationDerivation
        @test result.head isa CG.FunctionDefDerivation
        definition = result.head
        @test CG._has_state_helper_update(CG.FunctionDefCandidate(definition.target_ty,
            definition.body, definition.base_witness))
        @test result.args[2] == CG.VariableDerivation(0, list)
        @test CG.derivation_node_count(result) == 96 - state.nodes_left
        @test CG._derivation_weight(result, weights) > 0
        ledger = only(filter(r -> r isa CG.FunctionDefAttemptBudget, query.requirements))
        @test 2 <= only(ledger.remaining_by_level) <= 3
        values, types = Dict{Symbol,Any}(), Dict{Symbol,TyLike}()
        node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(), result,
            values, types, relation, Random.Xoshiro(42))
        tree = AstTree(AbsNode(:initial, AbsNode(:items, node)))
        target = TyFun(FloatTy, TyFun(list, FloatTy))
        @test iscompatible(infer(tree, merge(relation.global_consts_ty, types)), target)
        for xs in (Float64[], [2.0], [2.0,3.0])
            value = evaluate_native(tree, merge(relation.global_consts, values);
                args=(1.0,xs), max_steps=4096, max_call_depth=32)
            @test value isa Float64 && isfinite(value)
        end
    end
end

@testitem "CanonicalGeneration List state operand joint proposal preserves exact roles" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy, typekey
    using LambdaRegression.Primitives: FADD
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    helper = TyFun(FloatTy, TyFun(list, FloatTy))
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, scope_application=1,
        proxy_constructive=1, proxy_general=0, proxy_head=0, proxy_scope_application=0,
        proxy_split=1, constraint_evidence=0, list_recursive_body_probability=1-eps(),
        list_recursive_tail_probability=1-eps())
    relation = CG.CoreRelation(; weights, global_consts=Dict{Symbol,Any}(:ADD=>FADD),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    scope = Ty[list, FloatTy, list, FloatTy, helper, FloatTy]
    requirements = (CG.FunctionDefSelfBinder(4, 2, false), CG.CallsScopeBinder(4),
        CG._ListStateHelperSelf(4, helper), CG._ListRecursionHint(4, 2, 0, 1),
        CG._ListStateUpdateRequest(4, 3, 1), CG.ForbidsScopeBinder(2))
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        node_budget=32, requirements)
    rule = only(filter(r -> r.index == 4, CG._scope_application_rules(FloatTy, scope, 3)))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 32)
    call = CG._sample_preferred_list_recursive_call!(state, relation, rule, scope, 4, 0, true, query)
    @test call isa CG.ScopeApplicationDerivation
    if call isa CG.ScopeApplicationDerivation
        update, tail = call.args
        @test CG.derivation_uses_scope_binder(update, 3)
        @test CG.derivation_uses_scope_binder(update, 1)
        @test !CG.derivation_uses_scope_binder(update, 4)
        @test !CG.derivation_uses_scope_binder(update, 2)
        @test tail == CG.VariableDerivation(0, list)
        @test CG.derivation_node_count(call) == 32 - state.nodes_left
        @test CG._find_state_update_call(call, 4, 3, 1, 0, helper)
    end
    blocked = CG._derive_generation_query(query; requirements=(CG.ForbidsScopeBinder(3),))
    short = CG.DerivationSamplingState(Random.Xoshiro(1111), 32)
    @test CG._sample_preferred_list_recursive_call!(short, relation, rule, scope, 4, 0, true, blocked) === nothing
    @test short.nodes_left == 32
    moved = CG.rebase_generation_requirement(CG._ListStateUpdateRequest(4, 3, 1), 2)
    @test (moved.self_index, moved.state_index, moved.item_index) == (6, 5, 3)
    @test CG._state_helper_hint_admitted(query, CG._ListRecursionHint(4, 2, 0, 1))
    @test !CG._state_helper_hint_admitted(query, CG._ListRecursionHint(4, 1, 0, 1))
end

@testitem "CanonicalGeneration List helper menu shares helper only ledger and masks" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    provider = CG.FunctionDefProvider(max_nesting=1, body_attempts=4,
        admissibility=CG.DirectKnotFilter())
    weights = CG.GenerationWeights(default=0, variable=1, lambda=1, scope_application=1,
        function_def=0, list_state_helper=5)
    relation = CG.CoreRelation(; providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider(), provider), weights)
    root = CG.GenerationQuery(FloatTy; scope_tys=Ty[list, list, FloatTy], depth_left=9, node_budget=48)
    supplied = CG.generation_root_requirements(provider, relation, root)
    @test length(supplied) == 1
    query = CG._derive_generation_query(root; requirements=supplied)
    ledger = only(supplied)
    @test ledger.route_token === provider.route_token
    @test CG.generation_provider_applicable(provider, relation, query)
    options = CG._generation_extension_options(relation, query)
    @test length(options) == 2
    @test sum(o.weight for o in options) == 5
    @test !CG.generation_provider_applicable(provider, relation,
        CG._derive_generation_query(query; target_ty=TyFun(list, FloatTy)))
    masked = CG._derive_generation_query(query; requirements=(CG._ListStateHelperMask(),))
    @test !CG.generation_provider_applicable(provider, relation, masked)
    @test isempty(CG._generation_extension_options(relation, masked))
    @test CG._generation_query_cache_key(query, (provider,)) !=
        CG._generation_query_cache_key(masked, (provider,))
    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 48)
    # No combining head: the genuine state-update proposal must fail, spend no
    # more than the existing two attempts, and never replenish the shared ledger.
    result = CG._sample_option(state, relation, first(options), collect(Ty, query.scope_tys),
        query.depth_left, query.guarded_left, true, query)
    @test result === nothing
    @test 2 <= only(ledger.remaining_by_level) < 4
    @test state.nodes_left == 48
    before = only(ledger.remaining_by_level)
    CG._consume_functiondef_attempt!(query, provider) # model one ordinary attempt
    @test only(ledger.remaining_by_level) == before - 1
    @test isempty(CG._generation_extension_options(relation, query))
    @test !CG.generation_provider_applicable(provider, relation,
        CG._derive_generation_query(query; depth_left=1, node_budget=1))
    for kwargs in ((; sampler=CG.WeightedDerivationSampler(completion_probability=.1)),
                   (; argument_planner=(args...)->nothing))
        @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(; weights, kwargs...))
    end
end
