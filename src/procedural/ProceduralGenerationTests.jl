@testitem "Procedural FunctionDef generation requires a finite base witness" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural, TypeSystem
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, Ty, TyFun, TyMu, TyScheme, make_tyvar_factory

    CG = CanonicalGeneration
    mutable struct CountingBaseWitness
        calls::Base.RefValue{Int}
    end
    function (strategy::CountingBaseWitness)(relation, query, capability_mask)
        strategy.calls[] += 1
        return CG.known_generation_witness(relation, query; capability_mask)
    end

    counter = Ref(0)
    provider = Procedural.FunctionDefProvider(
        base_witness=CountingBaseWitness(counter),
        max_nesting=2,
        body_attempts=1,
    )
    target = TyFun(FloatTy, FloatTy)
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), provider),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    query = CG.GenerationQuery(
        target;
        depth_left=8,
        guarded_left=2,
        provider_fuel=4,
        node_budget=64,
    )

    options = CG._generation_extension_options(relation, query)
    functiondef_options = filter(option -> option.value.provider === provider, options)
    @test length(functiondef_options) == 1
    @test CG.generation_provider_applicable(provider, relation, query)
    @test !CG.generation_provider_applicable(
        provider,
        relation,
        CG.GenerationQuery(
            FloatTy;
            depth_left=8,
            provider_fuel=4,
            node_budget=64,
        ),
    )
    @test counter[] == 0

    state = CG.DerivationSamplingState(Random.Xoshiro(8000), 64)
    CG.sample_generation_extension(
        provider,
        state,
        relation,
        only(functiondef_options).value.option,
        query,
    )
    @test 1 <= counter[] <= provider.max_nesting

    cacheable = Procedural.FunctionDefProvider(max_nesting=2, body_attempts=4)
    @test CG.generation_provider_cache_key(provider) === nothing
    @test CG.generation_provider_cache_key(cacheable) !== nothing
    @test CG.generation_provider_cache_key(
        Procedural.FunctionDefProvider(max_nesting=2, body_attempts=5),
    ) != CG.generation_provider_cache_key(cacheable)

    changing_target = TyFun(FloatTy, BoolTy)
    changing_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:truth => true),
        global_consts_ty=Dict{Symbol,Any}(:truth => BoolTy),
        providers=(CG.ContextHeadProvider(), cacheable),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    changing_query = CG.GenerationQuery(
        changing_target;
        depth_left=6,
        guarded_left=1,
        provider_fuel=3,
        node_budget=32,
    )
    changing_option = only(filter(
        option -> option.value.provider === cacheable,
        CG._generation_extension_options(changing_relation, changing_query),
    )).value.option
    @test Procedural._functiondef_base_witness(
        cacheable,
        changing_relation,
        changing_option,
        changing_query,
    ) !== nothing
    delete!(changing_relation.global_consts, :truth)
    delete!(changing_relation.global_consts_ty, :truth)
    @test Procedural._functiondef_base_witness(
        cacheable,
        changing_relation,
        changing_option,
        changing_query,
    ) === nothing

    alpha = make_tyvar_factory()(0)
    impossible = TyScheme([alpha], TyFun(TyFun(alpha, alpha), alpha))
    @test Procedural.generate_procedural(
        impossible;
        functiondef_provider=cacheable,
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(42),
            providers=(CG.ContextHeadProvider(),),
            weights=CG.GenerationWeights(default=1, function_def=1),
            sampling_depth_limit=8,
            sampling_node_budget=64,
        ),
    ) isa LambdaRegression.Utils.Err

    binder = make_tyvar_factory()(0)
    recursive_target = TyMu(binder, TyFun(IntTy, binder))
    recursive_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => 1),
        global_consts_ty=Dict{Symbol,Any}(
            :mk => TyFun(IntTy, recursive_target),
            :seed => IntTy,
        ),
        providers=(CG.ContextHeadProvider(), cacheable),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    recursive_query = CG.GenerationQuery(
        recursive_target;
        depth_left=3,
        guarded_left=0,
        provider_fuel=2,
        node_budget=8,
    )
    @test length(CG._generation_extension_options(recursive_relation, recursive_query)) == 1

    exhausted = Procedural.FunctionDefProvider(max_nesting=0)
    exhausted_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), exhausted),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    @test isempty(CG._generation_extension_options(exhausted_relation, query))
end

@testitem "Procedural FunctionDef default body sampling does not require self" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = Procedural.FunctionDefProvider(
        max_nesting=1,
        body_attempts=1,
    )
    weights = CG.GenerationWeights(
        default=0,
        constant=1,
        function_def=1,
    )
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:ID => identity),
        global_consts_ty=Dict{Symbol,Any}(:ID => target),
        providers=(CG.ContextHeadProvider(), provider),
        weights=weights,
    )
    bare_query = CG.GenerationQuery(
        target;
        depth_left=4,
        guarded_left=0,
        provider_fuel=4,
        node_budget=16,
        capability_mask=relation.capability_mask,
    )
    query = CG.GenerationQuery(
        target;
        depth_left=bare_query.depth_left,
        guarded_left=bare_query.guarded_left,
        provider_fuel=bare_query.provider_fuel,
        node_budget=bare_query.node_budget,
        capability_mask=bare_query.capability_mask,
        requirements=CG.generation_root_requirements(provider, relation, bare_query),
    )
    extension = only(filter(
        option -> option.value.provider === provider,
        CG._generation_extension_options(relation, query),
    ))
    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 16)
    derivation = CG.sample_generation_extension(
        provider,
        state,
        relation,
        extension.value.option,
        query,
    )

    @test derivation isa Procedural.FunctionDefDerivation
    derivation isa Procedural.FunctionDefDerivation || return
    @test !CG.derivation_uses_scope_binder(derivation.body, 0)
    @test !CG.derivation_calls_scope_binder(derivation.body, 0)
end

@testitem "Procedural subtree replacement can introduce and revisit FunctionDef" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Mutations, Procedural
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations: DefaultSiteLockPolicy, IndexedProvider
    using LambdaRegression.Mutations: PathSiteSelectionPolicy, ScopedReplacementGenerator
    using LambdaRegression.Mutations: SubtreeReplacementMutation, WeightedPairChooser
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err
    using LambdaRegression.Trees: children

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    original = AstTree(AbsNode(:x, IndexNode(0)))
    function contains_functiondef(node::AstNode)
        node isa Procedural.FunctionDefNode && return true
        return any(contains_functiondef, children(node))
    end
    function procedural_inference(tree, type_ctx, factory, meta)
        return Procedural.infer_procedural(tree, type_ctx; factory, meta)
    end

    functiondef = Procedural.FunctionDefProvider(max_nesting=1, body_attempts=8)
    replacement = SubtreeReplacementMutation(
        site_lock=DefaultSiteLockPolicy(spine_only=false),
        site_selection=PathSiteSelectionPolicy(Int[]),
        replacement_generator=ScopedReplacementGenerator(
            Procedural.generate_procedural_scoped,
        ),
        generation_kwargs=(;
            functiondef_provider=functiondef,
            providers=CG.test_catalog_providers(),
            weights=CG.GenerationWeights(default=1, function_def=100),
            sampling_depth_limit=10,
            sampling_node_budget=96,
            guarded_expansion_depth=2,
        ),
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(SubtreeReplacementMutation => 1.0),
    )

    function conflicting_scoped(target_ty; kwargs...)
        tree = AstTree(Procedural.FunctionDefNode(AbsNode(:x, ConstNode(:k))))
        return GenerationResult(
            tree,
            target_ty,
            Dict{Symbol,Any}(:k => 2.0),
            Dict{Symbol,TyLike}(:k => FloatTy),
            Dict{Symbol,Any}(:source => :procedural_conflict_fixture),
        )
    end
    conflict_consts = Dict{Symbol,Any}(:k => 1.0)
    conflict_types = Dict{Symbol,TyLike}(:k => FloatTy)
    conflict_mutation = SubtreeReplacementMutation(
        site_lock=DefaultSiteLockPolicy(spine_only=false),
        site_selection=PathSiteSelectionPolicy(Int[]),
        replacement_generator=ScopedReplacementGenerator(conflicting_scoped),
    )
    conflicted = Mutations.mutate_tree(
        AstTree(AbsNode(:x, ConstNode(:k))),
        target;
        rng=Random.Xoshiro(2000),
        mutations=[conflict_mutation],
        chooser,
        provider=IndexedProvider(),
        type_inference=procedural_inference,
        local_consts=conflict_consts,
        local_consts_ty=conflict_types,
        max_try=1,
    )
    @test conflicted isa AstTree
    @test conflicted.root isa Procedural.FunctionDefNode
    renamed_const = conflicted.root.body.body
    @test renamed_const isa ConstNode
    @test renamed_const.name != :k
    @test conflict_consts[renamed_const.name] == 2.0

    local_consts = Dict{Symbol,Any}()
    local_const_types = Dict{Symbol,TyLike}()

    found = Ref{Union{Nothing,AstTree}}(nothing)
    for seed in 1:32
        candidate = Mutations.mutate_tree(
            original,
            target;
            rng=Random.Xoshiro(seed),
            mutations=[replacement],
            chooser,
            provider=IndexedProvider(),
            type_inference=procedural_inference,
            local_consts,
            local_consts_ty=local_const_types,
            max_try=1,
        )
        candidate isa Err && continue
        contains_functiondef(candidate.root) || continue
        found[] = candidate
        break
    end

    first = found[]
    @test first isa AstTree
    @test contains_functiondef(first.root)
    @test !(Procedural.infer_procedural(first, local_const_types) isa Err)

    second = Mutations.mutate_tree(
        first,
        target;
        rng=Random.Xoshiro(8000),
        mutations=[replacement],
        chooser,
        provider=IndexedProvider(),
        type_inference=procedural_inference,
        local_consts,
        local_consts_ty=local_const_types,
        max_try=8,
    )
    @test second isa AstTree
    @test !(Procedural.infer_procedural(second, local_const_types) isa Err)
end

@testitem "Procedural DirectKnotFilter tracks the exact recursive binder" begin
    using Test
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyFun, TyList

    CG = CanonicalGeneration
    F = TyFun(FloatTy, FloatTy)
    base = CG.LambdaDerivation(FloatTy, CG.VariableDerivation(0, FloatTy))
    direct = CG.VariableDerivation(0, F)
    eta = CG.LambdaDerivation(
        FloatTy,
        CG.ScopeApplicationDerivation(
            1,
            F,
            CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)],
        ),
    )
    one = CG.ConstantDerivation(CG.HeadEntry(
        :ONE,
        :constant,
        FloatTy,
        CG.CoreProductive(),
        CG.WeightBucket(:constant);
        metadata=(value=1.0,),
    ))
    changed = CG.LambdaDerivation(
        FloatTy,
        CG.ScopeApplicationDerivation(1, F, CG.CoreDerivation[one]),
    )

    filter = Procedural.DirectKnotFilter()
    @test !filter(Procedural.FunctionDefCandidate(F, direct, base, 0))
    @test !filter(Procedural.FunctionDefCandidate(F, eta, base, 0))
    @test !filter(Procedural.FunctionDefCandidate(F, changed, base, 0))
    @test filter(Procedural.FunctionDefCandidate(F, CG.VariableDerivation(1, F), base, 0))
    fadd_head = CG.HeadEntry(
        :FADD,
        :constant,
        TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        CG.CoreProductive(),
        CG.WeightBucket(:constant),
    )
    wrapped_unchanged_derivation = CG.LambdaDerivation(
        FloatTy,
        CG.HeadApplicationDerivation(
            fadd_head,
            CG.CoreDerivation[
                CG.ScopeApplicationDerivation(
                    1,
                    F,
                    CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)],
                ),
                one,
            ],
        ),
    )
    @test !filter(Procedural.FunctionDefCandidate(
        F,
        wrapped_unchanged_derivation,
        base,
        0,
    ))

    ListFloat = TyList(FloatTy)
    ListF = TyFun(ListFloat, FloatTy)
    list_base = CG.LambdaDerivation(ListFloat, one)
    safe_body = CG.ScopeApplicationDerivation(
        3,
        ListF,
        CG.CoreDerivation[CG.VariableDerivation(0, ListFloat)],
    )
    safe = CG.LambdaDerivation(
        ListFloat,
        CG.SafeListAccessDerivation(
            CG.VariableDerivation(0, ListFloat),
            ListFloat,
            safe_body,
            one,
            FloatTy,
        ),
    )
    candidate = Procedural.FunctionDefCandidate(ListF, safe, list_base, 0)
    @test CG.derivation_uses_scope_binder(safe, 0)
    @test filter(candidate)

    fallback_self = CG.ScopeApplicationDerivation(
        1,
        ListF,
        CG.CoreDerivation[CG.VariableDerivation(0, ListFloat)],
    )
    unsafe_fallback = CG.LambdaDerivation(
        ListFloat,
        CG.SafeListAccessDerivation(
            CG.VariableDerivation(0, ListFloat),
            ListFloat,
            safe_body,
            fallback_self,
            FloatTy,
        ),
    )
    @test !filter(Procedural.FunctionDefCandidate(
        ListF,
        unsafe_fallback,
        list_base,
        0,
    ))

    direct_node = Procedural.FunctionDefNode(IndexNode(0))
    eta_node = Procedural.FunctionDefNode(AbsNode(
        :x,
        AppNode(IndexNode(1), IndexNode(0)),
    ))
    changed_node = Procedural.FunctionDefNode(AbsNode(
        :x,
        ConstNode(:ONE),
    ))
    redirected_node = Procedural.FunctionDefNode(AbsNode(
        :x,
        AppNode(IndexNode(1), ConstNode(:ONE)),
    ))
    wrapped_unchanged = Procedural.FunctionDefNode(AbsNode(
        :x,
        AppNode(
            AppNode(ConstNode(:FADD), AppNode(IndexNode(1), IndexNode(0))),
            ConstNode(:ZERO),
        ),
    ))
    deferred_unchanged = Procedural.FunctionDefNode(AbsNode(
        :x,
        AppNode(
            ConstNode(:K),
            AbsNode(:y, AppNode(IndexNode(2), IndexNode(1))),
        ),
    ))
    @test Procedural.obvious_functiondef_loop_reason(direct_node) === :direct_self
    @test Procedural.obvious_functiondef_loop_reason(eta_node) ===
        :eta_self_forwarding
    @test Procedural.obvious_functiondef_loop_reason(changed_node) === nothing
    @test Procedural.obvious_functiondef_loop_reason(redirected_node) ===
        :unconditional_self_call
    @test Procedural.obvious_functiondef_loop_reason(wrapped_unchanged) ===
        :unchanged_self_call
    @test Procedural.obvious_functiondef_loop_reason(deferred_unchanged) === nothing
end

@testitem "Procedural DirectKnotFilter honors custom derivation signatures" begin
    using Test
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun

    CG = CanonicalGeneration
    struct SignatureDerivation <: CG.CoreDerivation
        ty
        signature
    end
    CG.derivation_node_count(::SignatureDerivation) = 1
    CG.derivation_result_type(derivation::SignatureDerivation) = derivation.ty
    CG.derivation_signature(derivation::SignatureDerivation) = derivation.signature

    F = TyFun(FloatTy, FloatTy)
    base = CG.LambdaDerivation(FloatTy, CG.VariableDerivation(0, FloatTy))
    eta = SignatureDerivation(
        F,
        (
            :lambda,
            CG.typekey(FloatTy),
            (
                :scope_application,
                1,
                CG.typekey(F),
                ((:var, 0, CG.typekey(FloatTy)),),
            ),
        ),
    )
    safe_argument = SignatureDerivation(
        FloatTy,
        (
            :safe_list_access,
            :UNCONS_CASE,
            :source_type,
            CG.typekey(FloatTy),
            (:var, 0, :source_type),
            (:var, 0, CG.typekey(FloatTy)),
            (:var, 0, CG.typekey(FloatTy)),
        ),
    )
    unsafe_body = CG.LambdaDerivation(
        FloatTy,
        CG.ScopeApplicationDerivation(1, F, CG.CoreDerivation[safe_argument]),
    )

    filter = Procedural.DirectKnotFilter()
    @test !filter(Procedural.FunctionDefCandidate(F, eta, base, 0))
    @test isempty(CG.derivation_capabilities(safe_argument))
    @test !filter(Procedural.FunctionDefCandidate(F, unsafe_body, base, 0))
end

@testitem "Procedural FunctionDef generation is opt in and post verified" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural, TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, TyFun, iscompatible

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = Procedural.FunctionDefProvider(max_nesting=2, body_attempts=8)
    global_consts = Dict{Symbol,Any}(:FADD => TypeSystem.FADD)
    weights = CG.GenerationWeights(
        default=0,
        lambda=1,
        variable=1,
        constant=1,
        scope_application=2,
        function_def=1.0e6,
    )
    base_config = CG.GenerationConfig(;
        rng=Random.Xoshiro(2000),
        providers=(CG.ContextHeadProvider(),),
        weights,
        sampling_depth_limit=10,
        sampling_node_budget=96,
    )

    canonical = Procedural.generate_procedural(
        target;
        functiondef_provider=nothing,
        global_consts,
        config=base_config,
    )
    @test canonical isa CG.GenerationResult
    @test !(canonical.tree.root isa Procedural.FunctionDefNode)
    @test canonical.metadata[:procedural_verification] == :not_needed

    observed = Ref{Any}(nothing)
    for seed in 902:933
        result = Procedural.generate_procedural(
            target;
            functiondef_provider=provider,
            global_consts,
            config=CG.GenerationConfig(;
                rng=Random.Xoshiro(seed),
                providers=(CG.ContextHeadProvider(),),
                weights,
                sampling_depth_limit=10,
                sampling_node_budget=96,
            ),
        )
        result isa CG.GenerationResult || continue
        result.tree.root isa Procedural.FunctionDefNode || continue
        observed[] = result
        break
    end

    @test observed[] isa CG.GenerationResult
    result = observed[]
    inferred = Procedural.infer_procedural(result.tree, result.local_const_types)
    @test iscompatible(inferred, target)
    metadata = result.metadata[:derivation_metadata][:function_defs]
    @test length(metadata) >= 1
    @test metadata[1][:self_index] == 0
    @test metadata[1][:base_signature] !== nothing
    @test :function_def in metadata[1][:used_capabilities]
    @test :recursive_type_query in metadata[1][:provider_capabilities]
    @test CG._signature_contains(result.metadata[:derivation], :function_def)
    @test result.metadata[:procedural_verification] == :verified
end

@testitem "Procedural generation verifies global and scoped contexts" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.ASTs.NodeASTs: ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, Ty, iscompatible

    CG = CanonicalGeneration
    global_result = Procedural.generate_procedural(
        FloatTy;
        functiondef_provider=nothing,
        global_consts=Dict{Symbol,Any}(:ONE => 1.0),
        global_consts_ty=Dict{Symbol,Any}(:ONE => FloatTy),
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(8000),
            providers=(CG.ContextHeadProvider(),),
            weights=CG.GenerationWeights(default=0, constant=1),
            sampling_depth_limit=2,
            sampling_node_budget=4,
        ),
    )
    @test global_result isa CG.GenerationResult
    @test global_result.tree.root == ConstNode(:ONE)
    @test iscompatible(
        Procedural.infer_procedural(global_result.tree, global_result.local_const_types),
        FloatTy,
    )

    scoped_result = Procedural.generate_procedural_scoped(
        FloatTy;
        scope_tys=Ty[FloatTy],
        functiondef_provider=nothing,
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(42),
            providers=(CG.ContextHeadProvider(),),
            weights=CG.GenerationWeights(default=0, variable=1),
            sampling_depth_limit=2,
            sampling_node_budget=4,
        ),
    )
    @test scoped_result isa CG.GenerationResult
    @test scoped_result.tree.root == IndexNode(0)
end

@testitem "Canonical proxy sampling conjoins inherited binder requirements" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyFun, TyList

    CG = CanonicalGeneration
    ListFloat = TyList(FloatTy)
    ListF = TyFun(ListFloat, FloatTy)
    scope = Ty[ListFloat, FloatTy, ListFloat, ListF]
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=1),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=4,
        provider_fuel=3,
        node_budget=16,
        requirements=(CG.UsesScopeBinder(3),),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(42), 16)
    body = CG._sample_proxy_using_derivation(
        state,
        relation,
        FloatTy,
        scope,
        4,
        0,
        2,
        0;
        generation_query=query,
    )

    @test body isa CG.CoreDerivation
    @test CG._uses_scope_prefix(body, 2)
    @test CG.derivation_uses_scope_binder(body, 3)
end

@testitem "Canonical proxy fast path splits proxy and self requirements" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyFun, TyList, TyNat

    CG = CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    F = TyFun(Matrix32, FloatTy)
    scope = Ty[TyList(Vec2), Vec2, Matrix32, F]
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(
            :FADD => TypeSystem.FADD,
            :LENF => TypeSystem.LENF,
        ),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            scope_application=1,
            array_source=1,
        ),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=8,
        provider_fuel=8,
        node_budget=64,
        requirements=(CG.UsesScopeBinder(3),),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 64)

    derivation = CG._sample_fast_split_requirement_head_derivation(
        state,
        relation,
        FloatTy,
        scope,
        8,
        0,
        2,
        0,
        false,
        query,
    )

    @test derivation isa CG.CoreDerivation
    @test CG._uses_scope_prefix(derivation, 2, 0)
    @test CG.derivation_uses_scope_binder(derivation, 3)
    @test CG._signature_contains(CG.derivation_signature(derivation), :FADD)
    @test CG._signature_contains(CG.derivation_signature(derivation), :LENF)
end

@testitem "Canonical required function binders use proxy arguments directly" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyFun, TyList

    CG = CanonicalGeneration
    ListFloat = TyList(FloatTy)
    ListF = TyFun(ListFloat, FloatTy)
    scope = Ty[ListFloat, FloatTy, ListFloat, ListF]
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=1),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=1,
        provider_fuel=1,
        node_budget=2,
        requirements=(CG.UsesScopeBinder(3),),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(555), 2)
    body = CG._sample_fast_required_scope_application_derivation(
        state,
        relation,
        FloatTy,
        scope,
        1,
        0,
        2,
        0,
        true,
        query,
    )

    @test body isa CG.ScopeApplicationDerivation
    @test body.index == 3
    @test length(body.args) == 1
    @test body.args[1] isa CG.VariableDerivation
    @test body.args[1].index == 0
    @test state.nodes_left == 0
end

@testitem "Canonical proxy fast proposals preserve generic sampling and planner boundaries" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, TypeSystem
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyArray, TyFun, TyList, TyNat

    CG = CanonicalGeneration
    scope = Ty[FloatTy, TyFun(FloatTy, FloatTy), TyFun(FloatTy, FloatTy)]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(1),),
    )
    weights = CG.GenerationWeights(
        default=0,
        variable=1,
        scope_application=1,
        proxy_constructive=0,
        proxy_general=1,
    )
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=weights,
    )
    observed_nested = Ref(false)
    for seed in 1:128
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 32)
        derivation = CG._sample_proxy_using_derivation(
            state,
            relation,
            FloatTy,
            scope,
            5,
            0,
            1,
            0;
            allow_direct_result=false,
            generation_query=query,
        )
        derivation === nothing && continue
        if CG.derivation_uses_scope_binder(derivation, 1) &&
           CG.derivation_uses_scope_binder(derivation, 2)
            observed_nested[] = true
            break
        end
    end
    @test observed_nested[]

    rejecting_planner = (derivations, args...) -> empty(derivations)
    rejected_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:FADD => TypeSystem.FADD),
        providers=(CG.ContextHeadProvider(),),
        argument_planner=rejecting_planner,
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            scope_application=1,
        ),
    )
    rejected_state = CG.DerivationSamplingState(Random.Xoshiro(42), 32)
    @test CG._sample_fast_split_requirement_head_derivation(
        rejected_state,
        rejected_relation,
        FloatTy,
        scope,
        5,
        0,
        1,
        0,
        false,
        query,
    ) === nothing

    no_application_state = CG.DerivationSamplingState(Random.Xoshiro(555), 32)
    @test CG._sample_proxy_using_derivation(
        no_application_state,
        rejected_relation,
        FloatTy,
        scope,
        5,
        0,
        1,
        0;
        allow_head_application=false,
        allow_direct_result=false,
        generation_query=query,
    ) === nothing

    tri = x -> y -> z -> x + y + z
    weighted_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:FADD => TypeSystem.FADD, :TRI => tri),
        global_consts_ty=Dict{Symbol,Any}(
            :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
            :TRI => TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        ),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            lambda=1,
            constant=3,
            scope_application=5,
        ),
    )
    split_plans = CG._fast_split_requirement_head_plans(
        weighted_relation,
        FloatTy,
        scope,
        5,
        1,
        0,
        query,
    )
    fadd_plans = [plan for plan in split_plans if plan.rule.head.name == :FADD]
    tri_plans = [plan for plan in split_plans if plan.rule.head.name == :TRI]
    @test length(fadd_plans) == 2
    @test length(tri_plans) == 6
    @test sum(plan.weight for plan in fadd_plans) ≈ 3
    @test sum(plan.weight for plan in tri_plans) ≈ 3

    scope_plans = CG._fast_required_scope_application_plans(
        weighted_relation,
        FloatTy,
        scope,
        5,
        1,
        0,
        query,
    )
    @test length(scope_plans) == 1
    @test only(scope_plans).weight ≈ 5

    nested_scope = Ty[
        FloatTy,
        TyList(FloatTy),
        FloatTy,
        TyFun(FloatTy, FloatTy),
    ]
    nested_query = CG.GenerationQuery(
        TyFun(BoolTy, FloatTy);
        scope_tys=nested_scope,
        depth_left=6,
        provider_fuel=6,
        node_budget=48,
        requirements=(CG.UsesScopeBinder(2), CG.UsesScopeBinder(3)),
    )
    nested_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 48)
    nested = CG._sample_fast_proxy_insertion_derivation(
        nested_state,
        weighted_relation,
        TyFun(BoolTy, FloatTy),
        nested_scope,
        6,
        0,
        2,
        0,
        false,
        nested_query;
        allow_direct_result=false,
    )
    @test nested isa CG.LambdaDerivation
    @test CG._uses_scope_prefix(nested, 2, 0)
    @test CG.derivation_uses_scope_binder(nested, 2)
    @test CG.derivation_uses_scope_binder(nested, 3)

    constructive_lambda_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            lambda=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    constructive_lambda_state = CG.DerivationSamplingState(Random.Xoshiro(1), 8)
    constructive_lambda = CG._sample_proxy_using_derivation(
        constructive_lambda_state,
        constructive_lambda_relation,
        TyFun(BoolTy, FloatTy),
        Ty[FloatTy],
        2,
        0,
        1,
        0;
        allow_direct_result=false,
    )
    @test constructive_lambda isa CG.LambdaDerivation
    @test CG._uses_scope_prefix(constructive_lambda, 1, 0)

    lambda_disabled_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            lambda=0,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    lambda_disabled_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 8)
    @test CG._sample_proxy_using_derivation(
        lambda_disabled_state,
        lambda_disabled_relation,
        TyFun(BoolTy, FloatTy),
        Ty[FloatTy],
        2,
        0,
        1,
        0;
        allow_direct_result=false,
    ) === nothing

    safe_scope = Ty[FloatTy, TyList(FloatTy)]
    safe_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=safe_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=24,
        requirements=(CG.UsesScopeBinder(1),),
    )
    safe_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            safe_list_access=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    safe_state = CG.DerivationSamplingState(Random.Xoshiro(1111), 24)
    safe = CG._sample_proxy_using_derivation(
        safe_state,
        safe_relation,
        FloatTy,
        safe_scope,
        4,
        0,
        1,
        0;
        allow_head_application=false,
        allow_safe_list_access=true,
        allow_direct_result=false,
        generation_query=safe_query,
    )
    @test safe isa CG.SafeListAccessDerivation
    @test !CG._signature_contains(CG.derivation_signature(safe), :app)

    fallback_scope = Ty[TyList(FloatTy), FloatTy]
    fallback_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=fallback_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=6,
        requirements=(CG.UsesScopeBinder(1),),
    )
    fallback_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            safe_list_access=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    fallback_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 6)
    fallback_safe = CG._sample_safe_list_access(
        fallback_state,
        fallback_relation,
        CG.ListAccessSource(
            TyList(FloatTy),
            CG.VariableDerivation(0, TyList(FloatTy)),
        ),
        FloatTy,
        fallback_scope,
        4,
        0,
        fallback_query,
    )
    @test fallback_safe isa CG.SafeListAccessDerivation
    @test fallback_safe.fallback isa CG.VariableDerivation
    @test fallback_safe.fallback.index == 1
    @test fallback_safe.body isa CG.VariableDerivation
    @test fallback_safe.body.index == 1

    identity_local = (derivations, args...) -> derivations
    local_head = CG.HeadEntry(
        :LOCAL,
        :global,
        TyFun(FloatTy, FloatTy),
        CG.CoreProductive(),
        CG.ConstantWeight();
        metadata=(argument_planner=identity_local,),
    )
    local_rule = CG.CoreRule(local_head, FloatTy, Ty[FloatTy], Symbol[:ordinary])
    forced = CG.VariableDerivation(1, FloatTy)
    rejecting_global = (derivations, args...) -> empty(derivations)
    global_reject_relation = CG.CoreRelation(
        argument_planner=rejecting_global,
        weights=CG.GenerationWeights(default=1),
    )
    @test CG._fast_forced_argument_admitted(
        global_reject_relation,
        local_rule,
        1,
        forced,
        Ty[FloatTy, FloatTy],
    )

    first_only = (derivations, args...) -> isempty(derivations) ? derivations : derivations[1:1]
    first_head = CG.HeadEntry(
        :FIRST,
        :global,
        TyFun(FloatTy, FloatTy),
        CG.CoreProductive(),
        CG.ConstantWeight();
        metadata=(argument_planner=first_only,),
    )
    first_rule = CG.CoreRule(first_head, FloatTy, Ty[FloatTy], Symbol[:ordinary])
    default_relation = CG.CoreRelation(weights=CG.GenerationWeights(default=1))
    @test !CG._fast_forced_argument_admitted(
        default_relation,
        first_rule,
        1,
        forced,
        Ty[FloatTy, FloatTy],
    )

    Vec2 = TyArray(TyNat(2), FloatTy)
    source_head = CG.HeadEntry(
        :SOURCE,
        :global,
        TyFun(Vec2, FloatTy),
        CG.CoreProductive(),
        CG.ConstantWeight();
        metadata=(source_array_depth_limit=0,),
    )
    source_rule = CG.CoreRule(source_head, FloatTy, Ty[Vec2], Symbol[:source_array])
    recursive_source = CG.ScopeApplicationDerivation(
        0,
        TyFun(FloatTy, Vec2),
        CG.CoreDerivation[CG.VariableDerivation(1, FloatTy)],
    )
    @test !CG._fast_forced_argument_admitted(
        default_relation,
        source_rule,
        1,
        recursive_source,
        Ty[TyFun(FloatTy, Vec2), FloatTy],
    )
end

@testitem "Procedural Known0 rejects opaque AST presets" begin
    using Test
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.ASTs.NodeASTs: IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = Procedural.FunctionDefProvider()
    relation = CG.CoreRelation(
        presets=Dict{Symbol,Any}(:recursive_preset => Procedural.FunctionDefNode(IndexNode(0))),
        preset_types=Dict{Symbol,Any}(:recursive_preset => target),
        providers=(CG.ContextHeadProvider(), provider),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    query = CG.GenerationQuery(
        target;
        depth_left=5,
        provider_fuel=3,
        node_budget=24,
    )

    witness = CG.known_generation_witness(
        relation,
        query;
        capability_mask=provider.base_capability_mask,
    )
    @test witness isa CG.CoreDerivation
    @test !CG._signature_contains(CG.derivation_signature(witness), :recursive_preset)
    @test length(CG._generation_extension_options(relation, query)) == 1
end

@testitem "Procedural provider retries and configuration are finite and explicit" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    mutable struct RejectAll
        calls::Base.RefValue{Int}
    end
    function (policy::RejectAll)(candidate)
        policy.calls[] += 1
        return false
    end

    target = TyFun(FloatTy, FloatTy)
    rejection = RejectAll(Ref(0))
    provider = Procedural.FunctionDefProvider(
        admissibility=rejection,
        body_attempts=3,
    )
    weights = CG.GenerationWeights(
        default=0,
        lambda=1,
        variable=1,
        scope_application=2,
        function_def=1.0e6,
    )
    result = Procedural.generate_procedural(
        target;
        functiondef_provider=provider,
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(2000),
            providers=(CG.ContextHeadProvider(),),
            weights,
            sampling_depth_limit=8,
            sampling_node_budget=64,
        ),
    )
    @test result isa CG.GenerationResult
    @test 0 < rejection.calls[] <= provider.body_attempts * provider.max_nesting
    @test !(result.tree.root isa Procedural.FunctionDefNode)

    embedded = CG.GenerationConfig(;
        rng=Random.Xoshiro(1),
        providers=(CG.ContextHeadProvider(), Procedural.FunctionDefProvider()),
        weights,
    )
    disabled = Procedural._with_functiondef_provider(embedded, nothing)
    @test !any(provider -> provider isa Procedural.FunctionDefProvider, disabled.providers)

    custom_sampler = (rng, sample_set) -> first(sample_set.derivations)
    unsupported = CG.GenerationConfig(;
        rng=Random.Xoshiro(1111),
        providers=(CG.ContextHeadProvider(),),
        weights,
        sampler=custom_sampler,
    )
    @test_throws ArgumentError Procedural.generate_procedural(
        target;
        functiondef_provider=Procedural.FunctionDefProvider(),
        config=unsupported,
    )
end

@testitem "Procedural FunctionDef composes with safe-list tail recursion" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyList, iscompatible

    CG = CanonicalGeneration
    ListFloat = TyList(FloatTy)
    target = TyFun(ListFloat, FloatTy)
    provider = Procedural.FunctionDefProvider(
        max_nesting=1,
        body_attempts=8,
        admissibility=Procedural.GenuineSelfCallFilter(),
    )
    weights = CG.GenerationWeights(
        default=0,
        lambda=1,
        variable=1,
        constant=1,
        scope_application=2,
        safe_list_access=1.0e8,
        function_def=1.0e9,
    )

    function find_signature(signature, tag)
        items = if signature isa Tuple || signature isa AbstractVector
            signature
        else
            return nothing
        end
        signature isa Tuple && !isempty(signature) && signature[1] === tag &&
            return signature
        for item in items
            found = find_signature(item, tag)
            found === nothing || return found
        end
        return nothing
    end

    observed = Ref{Any}(nothing)
    unguarded_recursive_argument = Ref(false)
    for seed in 1:32
        result = Procedural.generate_procedural(
            target;
            functiondef_provider=provider,
            global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
            global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy),
            config=CG.GenerationConfig(;
                rng=Random.Xoshiro(seed),
                providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
                weights,
                sampling_depth_limit=10,
                sampling_node_budget=96,
            ),
        )
        result isa CG.GenerationResult || continue
        functiondef = find_signature(result.metadata[:derivation], :function_def)
        recursive_body = functiondef === nothing ? nothing :
            find_signature(functiondef[4], :scope_application)
        if recursive_body !== nothing && any(
            argument -> find_signature(argument, :safe_list_access) !== nothing,
            recursive_body[4],
        )
            unguarded_recursive_argument[] = true
        end
        safe = find_signature(result.metadata[:derivation], :safe_list_access)
        safe === nothing && continue
        success = safe[6]
        recursive_call = find_signature(success, :scope_application)
        recursive_call === nothing && continue
        recursive_call[2] == 3 || continue
        any(
            argument -> argument isa Tuple && argument[1] === :var && argument[2] == 0,
            recursive_call[4],
        ) || continue
        observed[] = result
        break
    end

    @test !unguarded_recursive_argument[]
    @test observed[] isa CG.GenerationResult
    observed[] isa CG.GenerationResult || return
    result = observed[]
    @test iscompatible(
        Procedural.infer_procedural(result.tree, result.local_const_types),
        target,
    )
    entry = result.metadata[:derivation_metadata][:function_defs][1]
    @test :safe_list_access in entry[:used_capabilities]
    @test entry[:nesting_remaining] == 0
    @test any(
        item -> item.self_calls > 0,
        result.metadata[:functiondef_observations],
    )
end

@testitem "FunctionDef observations distinguish self use from recursive calls" begin
    using Test
    using LambdaRegression: Procedural
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode, LetNode

    nonrecursive = Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    returned_self = Procedural.FunctionDefNode(AbsNode(:x, IndexNode(1)))
    recursive = Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    passed_self = Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(0), IndexNode(1))),
    )
    curried_recursive = Procedural.FunctionDefNode(
        AbsNode(:x, AbsNode(
            :y,
            AppNode(AppNode(IndexNode(2), IndexNode(1)), IndexNode(0)),
        )),
    )
    nested_outer_call = Procedural.FunctionDefNode(
        AbsNode(:x, Procedural.FunctionDefNode(
            AbsNode(:y, AppNode(IndexNode(3), IndexNode(0))),
        )),
    )
    scoped_call = Procedural.FunctionDefNode(
        AbsNode(:x, Procedural.ScopeNode(
            AstNode[ConstNode(:base), IndexNode(0)],
            AppNode(IndexNode(3), IndexNode(0)),
        )),
    )
    let_call = Procedural.FunctionDefNode(
        AbsNode(:x, LetNode(
            :value,
            ConstNode(:base),
            AppNode(IndexNode(2), IndexNode(0)),
        )),
    )

    @test Procedural.functiondef_recursion_summary(nonrecursive) == (
        definitions=1,
        recursive_definitions=0,
        self_references=0,
        self_calls=0,
    )
    @test Procedural.functiondef_recursion_summary(returned_self) == (
        definitions=1,
        recursive_definitions=0,
        self_references=1,
        self_calls=0,
    )
    @test Procedural.functiondef_recursion_summary(recursive) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test Procedural.functiondef_recursion_summary(passed_self) == (
        definitions=1,
        recursive_definitions=0,
        self_references=1,
        self_calls=0,
    )
    @test Procedural.functiondef_recursion_summary(curried_recursive) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test Procedural.functiondef_recursion_summary(nested_outer_call) == (
        definitions=2,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test Procedural.functiondef_recursion_summary(scoped_call) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test Procedural.functiondef_recursion_summary(let_call) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
end

@testitem "Procedural Known0 rejects mu-aliased fixed-point scope heads when selected" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun, TyMu, make_tyvar_factory

    CG = CanonicalGeneration
    F = TyFun(FloatTy, BoolTy)
    binder = make_tyvar_factory()(0)
    fixed_point_alias = TyMu(binder, TyFun(TyFun(F, F), F))
    provider = Procedural.FunctionDefProvider(max_nesting=1)
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), provider),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    query = CG.GenerationQuery(
        F;
        scope_tys=Ty[fixed_point_alias],
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
    )
    options = NamedTuple[]

    CG.add_generation_options!(provider, options, relation, query)

    @test length(options) == 1
    state = CG.DerivationSamplingState(Random.Xoshiro(42), 32)
    @test CG.sample_generation_extension(
        provider,
        state,
        relation,
        only(options),
        query,
    ) === nothing
end

@testitem "Safe-list success masks access from nested generation extensions" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, IntTy, Ty, TyList

    CG = CanonicalGeneration

    mutable struct MaskObserverProvider <: CG.AbstractHeadProvider
        masks::Vector{Tuple{Vararg{Symbol}}}
    end

    CG.add_heads!(::MaskObserverProvider, heads, universe; kwargs...) = heads
    CG.provider_capabilities(::MaskObserverProvider) = ()
    CG.provides_generation_options(::MaskObserverProvider) = true
    CG.generation_provider_cache_key(::MaskObserverProvider) = nothing
    function CG.add_generation_options!(
        provider::MaskObserverProvider,
        options,
        relation,
        query::CG.GenerationQuery,
    )
        push!(provider.masks, query.capability_mask)
        return options
    end

    observer = MaskObserverProvider(Tuple{Vararg{Symbol}}[])
    safe_list = CG.SafeListAccessProvider()
    list_int = TyList(IntTy)
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:zero => FloatTy),
        providers=(CG.ContextHeadProvider(), safe_list, observer),
        weights=CG.GenerationWeights(default=1, safe_list_access=1),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[list_int],
        depth_left=4,
        guarded_left=0,
        provider_fuel=4,
        node_budget=32,
    )
    source = CG.ListAccessSource(list_int, CG.VariableDerivation(0, list_int))
    state = CG.DerivationSamplingState(Random.Xoshiro(555), 32)

    CG._sample_safe_list_access(
        state,
        relation,
        source,
        FloatTy,
        Ty[list_int],
        4,
        0,
        query,
    )

    @test !isempty(observer.masks)
    @test all(mask -> :safe_list_access in mask, observer.masks)
    masked_relation, _ = CG._relation_for_query(
        relation,
        CG.GenerationQuery(
            FloatTy;
            scope_tys=Ty[list_int],
            depth_left=4,
            provider_fuel=4,
            node_budget=32,
            capability_mask=(:safe_list_access,),
        ),
    )
    @test !any(provider -> provider isa CG.SafeListAccessProvider, masked_relation.providers)
end

@testitem "FunctionDef metadata aggregates derivation capabilities" begin
    using Test
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    F = TyFun(FloatTy, FloatTy)
    random_head = CG.HeadEntry(
        :RANDOM_F,
        :test,
        F,
        CG.CoreProductive(),
        CG.WeightBucket(:constant);
        metadata=(value=identity, capabilities=(:randomness,)),
    )
    body = CG.ConstantDerivation(random_head)
    derivation = Procedural.FunctionDefDerivation(
        F,
        body,
        body,
        0,
        (:function_def, :fixed_point, :recursive_type_query),
        0,
    )

    entry = only(CG.derivation_metadata(derivation)[:function_defs])
    @test :randomness in entry[:used_capabilities]
end

@testitem "Procedural verification inspects actual FunctionDef nodes" begin
    using Test
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using LambdaRegression.Utils: Err

    result = CanonicalGeneration.GenerationResult(
        AstTree(Procedural.FunctionDefNode(ConstNode(:ZERO))),
        FloatTy,
        Dict{Symbol,Any}(:ZERO => 0.0),
        Dict{Symbol,TyLike}(:ZERO => FloatTy),
        Dict{Symbol,Any}(),
    )

    verified = Procedural._verify_procedural_generation(
        result,
        FloatTy;
        fixed_point=Procedural.fixed_point_witness,
        kwargs=(;),
    )

    @test verified isa Err
end

@testitem "Procedural FunctionDef samples recursive mu heads under budget" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyMu, iscompatible, make_tyvar_factory

    CG = CanonicalGeneration
    binder = make_tyvar_factory()(0)
    target = TyMu(binder, TyFun(IntTy, binder))
    result = Procedural.generate_procedural(
        target;
        functiondef_provider=Procedural.FunctionDefProvider(
            max_nesting=1,
            body_attempts=4,
            admissibility=Procedural.GenuineSelfCallFilter(Procedural.AcceptFunctionDef()),
        ),
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => 1),
        global_consts_ty=Dict{Symbol,Any}(:mk => TyFun(IntTy, target), :seed => IntTy),
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(2000),
            providers=(CG.ContextHeadProvider(),),
            weights=CG.GenerationWeights(
                default=0,
                variable=1,
                constant=1,
                scope_application=2,
                function_def=1.0e6,
            ),
            sampling_depth_limit=4,
            sampling_node_budget=12,
        ),
    )

    @test result isa CG.GenerationResult
    @test result.tree.root isa Procedural.FunctionDefNode
    @test result.metadata[:sampling_nodes_used] <= 12
    @test iscompatible(
        Procedural.infer_procedural(result.tree, result.local_const_types),
        target,
    )
    entry = result.metadata[:derivation_metadata][:function_defs][1]
    @test :recursive_type_query in entry[:used_capabilities]
end

@testitem "Procedural FunctionDef attempts share one root budget" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun

    CG = CanonicalGeneration
    mutable struct RejectEveryCandidate
        calls::Int
    end
    function (policy::RejectEveryCandidate)(candidate)
        policy.calls += 1
        return false
    end

    policy = RejectEveryCandidate(0)
    provider = Procedural.FunctionDefProvider(
        admissibility=policy,
        max_nesting=1,
        body_attempts=2,
    )
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), provider),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    target = TyFun(FloatTy, FloatTy)
    bare_query = CG.GenerationQuery(
        target;
        depth_left=6,
        guarded_left=0,
        provider_fuel=6,
        node_budget=48,
    )
    root_requirements = CG.generation_root_requirements(provider, relation, bare_query)
    @test length(root_requirements) == 1
    budget = only(root_requirements)
    @test CG.generation_requirement_cache_key(budget) !== nothing
    other_provider = Procedural.FunctionDefProvider(max_nesting=1, body_attempts=3)
    other_budget = only(CG.generation_root_requirements(
        other_provider,
        relation,
        bare_query,
    ))
    combined_query = CG._derive_generation_query(
        bare_query;
        requirements=(budget, other_budget),
    )
    @test Procedural._functiondef_attempt_budget(combined_query, provider) === budget
    @test Procedural._functiondef_attempt_budget(combined_query, other_provider) === other_budget

    twin_provider = Procedural.FunctionDefProvider(max_nesting=1, body_attempts=3)
    other_twin_provider = Procedural.FunctionDefProvider(max_nesting=1, body_attempts=3)
    twin_budget = only(CG.generation_root_requirements(
        twin_provider,
        relation,
        bare_query,
    ))
    other_twin_budget = only(CG.generation_root_requirements(
        other_twin_provider,
        relation,
        bare_query,
    ))
    twin_query = CG._derive_generation_query(
        bare_query;
        requirements=(twin_budget, other_twin_budget),
    )
    @test CG.generation_requirement_cache_key(twin_budget) !=
        CG.generation_requirement_cache_key(other_twin_budget)
    @test Procedural._functiondef_attempt_budget(twin_query, twin_provider) === twin_budget
    @test Procedural._functiondef_attempt_budget(twin_query, other_twin_provider) === other_twin_budget

    layered_provider = Procedural.FunctionDefProvider(max_nesting=2, body_attempts=2)
    layered_budget = only(CG.generation_root_requirements(
        layered_provider,
        relation,
        bare_query,
    ))
    layered_root = CG._derive_generation_query(
        bare_query;
        requirements=(layered_budget,),
    )
    layered_nested = CG._derive_generation_query(
        layered_root;
        requirements=(Procedural.FunctionDefFuel(1),),
    )
    @test Procedural._functiondef_attempts_remaining(layered_root, layered_provider) == 2
    @test Procedural._consume_functiondef_attempt!(layered_nested, layered_provider)
    @test Procedural._functiondef_attempts_remaining(layered_nested, layered_provider) == 1
    @test Procedural._functiondef_attempts_remaining(layered_root, layered_provider) == 2

    query = CG._derive_generation_query(bare_query; requirements=root_requirements)
    options = NamedTuple[]
    CG.add_generation_options!(provider, options, relation, query)
    option = only(options)
    @test !CG.generation_provider_exhausted(provider, relation, query)
    @test !CG._generation_extensions_exhausted(relation, query)

    first_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 48)
    @test CG.sample_generation_extension(provider, first_state, relation, option, query) === nothing
    @test only(budget.remaining_by_level) == 1
    @test policy.calls == 1
    @test !CG.generation_provider_exhausted(provider, relation, query)
    second_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 48)
    @test CG.sample_generation_extension(provider, second_state, relation, option, query) === nothing
    @test only(budget.remaining_by_level) == 0
    @test policy.calls == 2
    @test CG.generation_provider_exhausted(provider, relation, query)
    @test CG._generation_extensions_exhausted(relation, query)
    constrained = CG._derive_generation_query(
        query;
        requirements=(CG.UsesScopeBinder(0),),
    )
    @test !CG._generation_extensions_exhausted(relation, constrained)

    nested_base_query = CG.GenerationQuery(
        target;
        scope_tys=(target, target),
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
        requirements=(Procedural.FunctionDefSelfBinder(0),),
    )
    nested_base = Procedural._functiondef_base_witness(
        provider,
        relation,
        (
            target_ty=target,
            base_depth_left=4,
            base_node_budget=16,
        ),
        nested_base_query,
    )
    @test nested_base isa CG.VariableDerivation
    @test nested_base.index == 1

    fully_forbidden_query = CG.GenerationQuery(
        target;
        scope_tys=(target, target),
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
        requirements=(
            Procedural.FunctionDefSelfBinder(0),
            CG.ForbidsScopeBinder(1),
        ),
    )
    fully_forbidden_base = Procedural._functiondef_base_witness(
        provider,
        relation,
        (
            target_ty=target,
            base_depth_left=4,
            base_node_budget=16,
        ),
        fully_forbidden_query,
    )
    @test fully_forbidden_base isa CG.LambdaDerivation
    @test fully_forbidden_base.body isa CG.VariableDerivation
    @test fully_forbidden_base.body.index == 0

    owned_base_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => true),
        global_consts_ty=Dict{Symbol,Any}(
            :mk => TyFun(BoolTy, FloatTy),
            :seed => BoolTy,
        ),
        weights=CG.GenerationWeights(default=1),
    )
    owned_base_query = CG.GenerationQuery(
        FloatTy;
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
    )
    owned_base = Procedural._functiondef_base_witness(
        provider,
        owned_base_relation,
        (
            target_ty=FloatTy,
            base_depth_left=4,
            base_node_budget=16,
        ),
        owned_base_query,
    )
    cached_base = only(values(owned_base_relation.known_witnesses)).witness
    @test owned_base isa CG.HeadApplicationDerivation
    @test cached_base isa CG.HeadApplicationDerivation
    @test owned_base !== cached_base
    @test owned_base.args !== cached_base.args
    calls_after_first = policy.calls

    second_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 48)
    @test CG.sample_generation_extension(provider, second_state, relation, option, query) === nothing
    @test policy.calls == calls_after_first

    nonfunction_root = CG.GenerationQuery(
        FloatTy;
        depth_left=6,
        guarded_left=0,
        provider_fuel=6,
        node_budget=48,
    )
    inherited = CG.generation_root_requirements(provider, relation, nonfunction_root)
    @test length(inherited) == 1
    inherited_budget = only(inherited)
    nested_query = CG._derive_generation_query(
        nonfunction_root;
        target_ty=target,
        depth_left=5,
        provider_fuel=5,
        node_budget=40,
        requirements=inherited,
    )
    @test any(requirement -> requirement === inherited_budget, nested_query.requirements)
end
