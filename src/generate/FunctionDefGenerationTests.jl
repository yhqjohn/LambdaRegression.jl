@testitem "FunctionDef FunctionDef generation requires a finite base witness" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, TypeSystem
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
    provider = CanonicalGeneration.FunctionDefProvider(
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

    cacheable = CanonicalGeneration.FunctionDefProvider(max_nesting=2, body_attempts=4)
    @test CG.generation_provider_cache_key(provider) === nothing
    @test CG.generation_provider_cache_key(cacheable) !== nothing
    @test CG.generation_provider_cache_key(
        CanonicalGeneration.FunctionDefProvider(max_nesting=2, body_attempts=5),
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
    @test CanonicalGeneration._functiondef_base_witness(
        cacheable,
        changing_relation,
        changing_option,
        changing_query,
    ) !== nothing
    delete!(changing_relation.global_consts, :truth)
    delete!(changing_relation.global_consts_ty, :truth)
    @test CanonicalGeneration._functiondef_base_witness(
        cacheable,
        changing_relation,
        changing_option,
        changing_query,
    ) === nothing

    alpha = make_tyvar_factory()(0)
    impossible = TyScheme([alpha], TyFun(TyFun(alpha, alpha), alpha))
    @test CanonicalGeneration.generate_canonical(
        impossible;
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(42),
            providers=(CG.ContextHeadProvider(), cacheable),
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

    exhausted = CanonicalGeneration.FunctionDefProvider(max_nesting=0)
    exhausted_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), exhausted),
        weights=CG.GenerationWeights(default=1, function_def=1),
    )
    @test isempty(CG._generation_extension_options(exhausted_relation, query))
end

@testitem "FunctionDef FunctionDef default body sampling does not require self" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = CanonicalGeneration.FunctionDefProvider(
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

    @test derivation isa CanonicalGeneration.FunctionDefDerivation
    derivation isa CanonicalGeneration.FunctionDefDerivation || return
    @test !CG.derivation_uses_scope_binder(derivation.body, 0)
    @test !CG.derivation_calls_scope_binder(derivation.body, 0)
end

@testitem "FunctionDef subtree replacement can introduce and revisit FunctionDef" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, Mutations
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
        node isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode && return true
        return any(contains_functiondef, children(node))
    end
    functiondef = CanonicalGeneration.FunctionDefProvider(max_nesting=1, body_attempts=8)
    replacement = SubtreeReplacementMutation(
        site_lock=DefaultSiteLockPolicy(spine_only=false),
        site_selection=PathSiteSelectionPolicy(Int[]),
        replacement_generator=ScopedReplacementGenerator(
            CanonicalGeneration.generate_canonical_scoped,
        ),
        generation_kwargs=(;
            providers=(CG.test_catalog_providers()..., functiondef),
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
        tree = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, ConstNode(:k))))
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
        local_consts=conflict_consts,
        local_consts_ty=conflict_types,
        max_try=1,
    )
    @test conflicted isa AstTree
    @test conflicted.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode
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
    @test !(LambdaRegression.TypeSystem.infer_term(first, local_const_types) isa Err)

    second = Mutations.mutate_tree(
        first,
        target;
        rng=Random.Xoshiro(8000),
        mutations=[replacement],
        chooser,
        provider=IndexedProvider(),
        local_consts,
        local_consts_ty=local_const_types,
        max_try=8,
    )
    @test second isa AstTree
    @test !(LambdaRegression.TypeSystem.infer_term(second, local_const_types) isa Err)
end

@testitem "FunctionDef DirectKnotFilter tracks the exact recursive binder" begin
    using Test
    using LambdaRegression: CanonicalGeneration
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

    filter = CanonicalGeneration.DirectKnotFilter()
    @test !filter(CanonicalGeneration.FunctionDefCandidate(F, direct, base, 0))
    @test !filter(CanonicalGeneration.FunctionDefCandidate(F, eta, base, 0))
    @test !filter(CanonicalGeneration.FunctionDefCandidate(F, changed, base, 0))
    @test filter(CanonicalGeneration.FunctionDefCandidate(F, CG.VariableDerivation(1, F), base, 0))
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
    @test !filter(CanonicalGeneration.FunctionDefCandidate(
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
    candidate = CanonicalGeneration.FunctionDefCandidate(ListF, safe, list_base, 0)
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
    @test !filter(CanonicalGeneration.FunctionDefCandidate(
        ListF,
        unsafe_fallback,
        list_base,
        0,
    ))

    direct_node = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0))
    eta_node = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :x,
        AppNode(IndexNode(1), IndexNode(0)),
    ))
    changed_node = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :x,
        ConstNode(:ONE),
    ))
    redirected_node = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :x,
        AppNode(IndexNode(1), ConstNode(:ONE)),
    ))
    wrapped_unchanged = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :x,
        AppNode(
            AppNode(ConstNode(:FADD), AppNode(IndexNode(1), IndexNode(0))),
            ConstNode(:ZERO),
        ),
    ))
    deferred_unchanged = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :x,
        AppNode(
            ConstNode(:K),
            AbsNode(:y, AppNode(IndexNode(2), IndexNode(1))),
        ),
    ))
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(direct_node) === :direct_self
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(eta_node) ===
        :eta_self_forwarding
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(changed_node) === nothing
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(redirected_node) ===
        :unconditional_self_call
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(wrapped_unchanged) ===
        :unchanged_self_call
    @test LambdaRegression.ProgramAnalysis.obvious_functiondef_loop_reason(deferred_unchanged) === nothing
end

@testitem "FunctionDef DirectKnotFilter honors custom derivation signatures" begin
    using Test
    using LambdaRegression: CanonicalGeneration
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

    filter = CanonicalGeneration.DirectKnotFilter()
    @test !filter(CanonicalGeneration.FunctionDefCandidate(F, eta, base, 0))
    @test isempty(CG.derivation_capabilities(safe_argument))
    @test !filter(CanonicalGeneration.FunctionDefCandidate(F, unsafe_body, base, 0))
end

@testitem "FunctionDef FunctionDef generation is opt in and post verified" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration, TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, TyFun, iscompatible

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = CanonicalGeneration.FunctionDefProvider(max_nesting=2, body_attempts=8)
    global_consts = Dict{Symbol,Any}(:FADD => LambdaRegression.Primitives.FADD)
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

    canonical = CanonicalGeneration.generate_canonical(
        target;
        global_consts,
        config=base_config,
    )
    @test canonical isa CG.GenerationResult
    @test !(canonical.tree.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode)
    @test canonical.metadata[:functiondef_verification] == :not_needed

    observed = Ref{Any}(nothing)
    for seed in 902:933
        result = CanonicalGeneration.generate_canonical(
            target;
            global_consts,
            config=CG.GenerationConfig(;
                rng=Random.Xoshiro(seed),
                providers=(CG.ContextHeadProvider(), provider),
                weights,
                sampling_depth_limit=10,
                sampling_node_budget=96,
            ),
        )
        result isa CG.GenerationResult || continue
        result.tree.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode || continue
        observed[] = result
        break
    end

    @test observed[] isa CG.GenerationResult
    result = observed[]
    inferred = LambdaRegression.TypeSystem.infer_term(result.tree, result.local_const_types)
    @test iscompatible(inferred, target)
    metadata = result.metadata[:derivation_metadata][:function_defs]
    @test length(metadata) >= 1
    @test metadata[1][:self_index] == 0
    @test metadata[1][:base_signature] !== nothing
    @test :function_def in metadata[1][:used_capabilities]
    @test :recursive_type_query in metadata[1][:provider_capabilities]
    @test CG._signature_contains(result.metadata[:derivation], :function_def)
    @test result.metadata[:functiondef_verification] == :verified
end

@testitem "FunctionDef generation verifies global and scoped contexts" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.ASTs.NodeASTs: ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, Ty, iscompatible

    CG = CanonicalGeneration
    global_result = CanonicalGeneration.generate_canonical(
        FloatTy;
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
        LambdaRegression.TypeSystem.infer_term(global_result.tree, global_result.local_const_types),
        FloatTy,
    )

    scoped_result = CanonicalGeneration.generate_canonical_scoped(
        FloatTy;
        scope_tys=Ty[FloatTy],
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
            :FADD => LambdaRegression.Primitives.FADD,
            :LENF => LambdaRegression.Primitives.LENF,
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
        global_consts=Dict{Symbol,Any}(:FADD => LambdaRegression.Primitives.FADD),
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
        global_consts=Dict{Symbol,Any}(:FADD => LambdaRegression.Primitives.FADD, :TRI => tri),
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
    # Descriptors are not a feasibility-filtered menu. The original f1 x
    # forced route and a generic placement keep f1's total head weight; f2
    # can also contain both obligations in its one child, as f2 (f1 x).
    first_head_plans = [plan for plan in scope_plans if plan.rule.index == 1]
    second_head_plans = [plan for plan in scope_plans if plan.rule.index == 2]
    @test sum(plan.weight for plan in first_head_plans) ≈ 5
    @test sum(plan.weight for plan in second_head_plans) ≈ 5
    @test any(plan -> plan.forced == [0], first_head_plans)
    @test any(plan -> plan.forced == [-1], first_head_plans)
    @test !isempty(second_head_plans)
    observed_joint_child = Ref(false)
    for plan in second_head_plans, seed in 1:16
        derivation = CG._materialize_fast_required_scope_application(
            CG.DerivationSamplingState(Random.Xoshiro(seed), 32), weighted_relation,
            plan, scope, 4, 0, 1, 0, false, query)
        derivation === nothing && continue
        @test CG.derivation_uses_scope_binder(derivation, 1)
        @test CG._uses_scope_prefix(derivation, 1, 0)
        @test CG._generation_requirements_satisfied(derivation, query)
        observed_joint_child[] = true
    end
    @test observed_joint_child[]
    @test CG._fast_required_scope_application_plans(
        weighted_relation,
        FloatTy,
        scope,
        5,
        1,
        0,
        query,
    ) === scope_plans

    changed_depth_plans = CG._fast_required_scope_application_plans(
        weighted_relation,
        FloatTy,
        scope,
        4,
        1,
        0,
        query,
    )
    @test changed_depth_plans !== scope_plans
    changed_requirement_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(2),),
    )
    changed_requirement_plans = CG._fast_required_scope_application_plans(
        weighted_relation,
        FloatTy,
        scope,
        5,
        1,
        0,
        changed_requirement_query,
    )
    @test changed_requirement_plans !== scope_plans
    changed_budget_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=31,
        requirements=(CG.UsesScopeBinder(1),),
    )
    changed_budget_plans = CG._fast_required_scope_application_plans(
        weighted_relation,
        FloatTy,
        scope,
        5,
        1,
        0,
        changed_budget_query,
    )
    @test changed_budget_plans !== scope_plans

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

@testitem "FunctionDef Known0 rejects opaque AST presets" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.ASTs.NodeASTs: IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    provider = CanonicalGeneration.FunctionDefProvider()
    relation = CG.CoreRelation(
        presets=Dict{Symbol,Any}(:recursive_preset => LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0))),
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

@testitem "FunctionDef provider retries and configuration are finite and explicit" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
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
    provider = CanonicalGeneration.FunctionDefProvider(
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
    result = CanonicalGeneration.generate_canonical(
        target;
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(2000),
            providers=(CG.ContextHeadProvider(), provider),
            weights,
            sampling_depth_limit=8,
            sampling_node_budget=64,
        ),
    )
    @test result isa CG.GenerationResult
    @test 0 < rejection.calls[] <= provider.body_attempts * provider.max_nesting
    @test !(result.tree.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode)

    embedded = CG.GenerationConfig(;
        rng=Random.Xoshiro(1),
        providers=(CG.ContextHeadProvider(), CanonicalGeneration.FunctionDefProvider()),
        weights,
    )
    disabled = CanonicalGeneration.with_functiondef_provider(embedded, nothing)
    @test !any(provider -> provider isa CanonicalGeneration.FunctionDefProvider, disabled.providers)

    custom_sampler = (rng, sample_set) -> first(sample_set.derivations)
    unsupported = CG.GenerationConfig(;
        rng=Random.Xoshiro(1111),
        providers=(CG.ContextHeadProvider(), CanonicalGeneration.FunctionDefProvider()),
        weights,
        sampler=custom_sampler,
    )
    @test_throws ArgumentError CanonicalGeneration.generate_canonical(
        target;
        config=unsupported,
    )

    unsupported_planner = CG.GenerationConfig(;
        rng=Random.Xoshiro(1234),
        providers=(CG.ContextHeadProvider(), CanonicalGeneration.FunctionDefProvider()),
        weights,
        argument_planner=(args...) -> args,
    )
    @test_throws ArgumentError CanonicalGeneration.generate_canonical(
        target;
        config=unsupported_planner,
    )
    @test_throws ArgumentError CanonicalGeneration.generate_canonical_scoped(
        target;
        config=unsupported_planner,
    )
end

@testitem "Canonical FunctionDef verification skips only proven core configurations" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.ASTs.NodeASTs: ConstNode, FunctionDefNode

    CG = CanonicalGeneration
    core = CG.GenerationConfig(providers=(CG.ContextHeadProvider(),))
    @test !CG._functiondef_verification_required(core, Dict{Symbol,Any}())

    custom_sampler = CG.GenerationConfig(
        providers=(CG.ContextHeadProvider(),),
        sampler=(rng, candidates) -> first(candidates.derivations),
    )
    @test CG._functiondef_verification_required(custom_sampler, Dict{Symbol,Any}())

    custom_planner = CG.GenerationConfig(
        providers=(CG.ContextHeadProvider(),),
        argument_planner=(args...) -> args,
    )
    @test CG._functiondef_verification_required(custom_planner, Dict{Symbol,Any}())

    custom_materializer = CG.GenerationConfig(
        providers=(CG.ContextHeadProvider(),),
        materializer=(args...) -> nothing,
    )
    @test CG._functiondef_verification_required(custom_materializer, Dict{Symbol,Any}())

    custom_collector = CG.GenerationConfig(
        providers=(
            CG.ContextHeadProvider(),
            CG.SafeListAccessProvider(source_collectors=((relation, scope) -> (),)),
        ),
    )
    @test CG._functiondef_verification_required(custom_collector, Dict{Symbol,Any}())
    @test CG._functiondef_verification_required(
        core,
        Dict{Symbol,Any}(:recursive => FunctionDefNode(ConstNode(:value))),
    )
end

@testitem "FunctionDef FunctionDef composes with safe-list tail recursion" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyList, iscompatible

    CG = CanonicalGeneration
    ListFloat = TyList(FloatTy)
    target = TyFun(ListFloat, FloatTy)
    provider = CanonicalGeneration.FunctionDefProvider(
        max_nesting=1,
        body_attempts=8,
        admissibility=CanonicalGeneration.GenuineSelfCallFilter(),
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
        result = CanonicalGeneration.generate_canonical(
            target;
            global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
            global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy),
            config=CG.GenerationConfig(;
                rng=Random.Xoshiro(seed),
                providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider(), provider),
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
        LambdaRegression.TypeSystem.infer_term(result.tree, result.local_const_types),
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
    using LambdaRegression: ProgramAnalysis
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, ConstNode, IndexNode

    nonrecursive = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    returned_self = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(1)))
    recursive = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    passed_self = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(0), IndexNode(1))),
    )
    curried_recursive = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AbsNode(
            :y,
            AppNode(AppNode(IndexNode(2), IndexNode(1)), IndexNode(0)),
        )),
    )
    nested_outer_call = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            AbsNode(:y, AppNode(IndexNode(3), IndexNode(0))),
        )),
    )
    let_like_call = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(
            AbsNode(:value, AppNode(IndexNode(2), IndexNode(0))),
            ConstNode(:base),
        )),
    )

    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(nonrecursive) == (
        definitions=1,
        recursive_definitions=0,
        self_references=0,
        self_calls=0,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(returned_self) == (
        definitions=1,
        recursive_definitions=0,
        self_references=1,
        self_calls=0,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(recursive) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(passed_self) == (
        definitions=1,
        recursive_definitions=0,
        self_references=1,
        self_calls=0,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(curried_recursive) == (
        definitions=1,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(nested_outer_call) == (
        definitions=2,
        recursive_definitions=1,
        self_references=1,
        self_calls=1,
    )
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(let_like_call).self_calls == 1
end

@testitem "FunctionDef Known0 rejects mu-aliased fixed-point scope heads when selected" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun, TyMu, make_tyvar_factory

    CG = CanonicalGeneration
    F = TyFun(FloatTy, BoolTy)
    binder = make_tyvar_factory()(0)
    fixed_point_alias = TyMu(binder, TyFun(TyFun(F, F), F))
    provider = CanonicalGeneration.FunctionDefProvider(max_nesting=1)
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
    using LambdaRegression: CanonicalGeneration
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
    derivation = CanonicalGeneration.FunctionDefDerivation(
        F,
        body,
        body,
        0,
        (:function_def, :fixed_point, :recursive_type_query),
        0,
    )

    metadata = CG.derivation_metadata(derivation)
    @test CG._derivation_metadata_tree_eligible(metadata)
    copied = CG._copy_derivation_metadata(metadata)
    copied_again = CG._copy_derivation_metadata(metadata)
    @test isequal(metadata, copied)
    @test isequal(metadata, copied_again)
    @test copied !== metadata
    @test copied[:function_defs] !== metadata[:function_defs]
    @test copied[:function_defs] !== copied_again[:function_defs]

    entry = only(metadata[:function_defs])
    @test isequal(entry, Dict{Symbol,Any}(
        :target => LambdaRegression.TypeSystem.typekey(F),
        :self_index => 0,
        :self_binder => :debruijn,
        :base_signature => CG.derivation_signature(body),
        :body_signature => CG.derivation_signature(body),
        :provider_capabilities => derivation.capabilities,
        :used_capabilities => CG._used_functiondef_capabilities(derivation),
        :nesting_remaining => 0,
    ))
    attached = Dict{Symbol,Any}()
    CG._attach_derivation_metadata!(attached, derivation)
    @test isequal(attached[:derivation_metadata], metadata)

    copied_entry = only(copied[:function_defs])
    copied_entry[:self_index] = 7
    @test entry[:self_index] == 0
    @test only(copied_again[:function_defs])[:self_index] == 0
    @test :randomness in entry[:used_capabilities]
end

@testitem "FunctionDef metadata copy falls back for opaque and aliased values" begin
    using Test
    using LambdaRegression: CanonicalGeneration

    CG = CanonicalGeneration

    text = string("provider-metadata-", repeat("x", 128), "-string")
    @test !CG._derivation_metadata_tree_eligible(text)
    expected_text = deepcopy(text)
    copied_text = CG._copy_derivation_metadata(text)
    @test isequal(copied_text, expected_text)
    @test pointer(copied_text) != pointer(text)

    struct CustomIsbitsMetadataLeaf
        value::Int
    end
    custom_isbits = CustomIsbitsMetadataLeaf(7)
    @test isbitstype(typeof(custom_isbits))
    @test !CG._derivation_metadata_tree_eligible(custom_isbits)
    @test CG._copy_derivation_metadata(custom_isbits) === custom_isbits

    shared = Any[:shared]
    aliased = Dict{Symbol,Any}(:left => shared, :right => shared)
    @test !CG._derivation_metadata_tree_eligible(aliased)
    copied_aliases = CG._copy_derivation_metadata(aliased)
    @test isequal(copied_aliases, aliased)
    @test copied_aliases[:left] === copied_aliases[:right]
    @test copied_aliases[:left] !== shared

    cyclic = Dict{Symbol,Any}()
    cyclic[:self] = cyclic
    @test !CG._derivation_metadata_tree_eligible(cyclic)
    copied_cycle = CG._copy_derivation_metadata(cyclic)
    @test copied_cycle[:self] === copied_cycle

    owner = Vector{Any}(undef, 4)
    owner .= Any[:one, :two, :three, :four]
    shared_view = view(owner, 2:3)
    view_tree = Dict{Symbol,Any}(:owner => owner, :view => shared_view)
    @test !CG._derivation_metadata_tree_eligible(view_tree)
    copied_view_tree = CG._copy_derivation_metadata(view_tree)
    @test parent(copied_view_tree[:view]) === copied_view_tree[:owner]

    spare_capacity = Vector{Any}(undef, 8)
    resize!(spare_capacity, 2)
    spare_capacity .= Any[:one, :two]
    @test !CG._derivation_metadata_tree_eligible(Dict{Symbol,Any}(:spare => spare_capacity))

    partially_assigned = Vector{Any}(undef, 2)
    partially_assigned[1] = :present
    @test !CG._derivation_metadata_tree_eligible(Dict{Symbol,Any}(:partial => partially_assigned))
    copied_partial = CG._copy_derivation_metadata(Dict{Symbol,Any}(:partial => partially_assigned))
    @test copied_partial[:partial][1] === :present
    @test !isassigned(copied_partial[:partial], 2)

    mutable struct OpaqueMetadataLeaf
        payload::Any
    end
    const OPAQUE_DEEPCOPY_CALLS = Ref(0)
    function Base.deepcopy_internal(value::OpaqueMetadataLeaf, memo::IdDict)
        OPAQUE_DEEPCOPY_CALLS[] += 1
        haskey(memo, value) && return memo[value]
        copied = OpaqueMetadataLeaf(nothing)
        memo[value] = copied
        copied.payload = Base.deepcopy_internal(value.payload, memo)
        return copied
    end

    opaque_payload = Any[:opaque]
    opaque = OpaqueMetadataLeaf(opaque_payload)
    opaque_tree = Dict{Symbol,Any}(
        :shared => opaque_payload,
        :opaque => opaque,
        :opaque_alias => opaque,
    )
    @test !CG._derivation_metadata_tree_eligible(opaque_tree)
    OPAQUE_DEEPCOPY_CALLS[] = 0
    copied_opaque = CG._copy_derivation_metadata(opaque_tree)
    @test OPAQUE_DEEPCOPY_CALLS[] > 0
    @test isequal(copied_opaque[:shared], opaque_payload)
    @test copied_opaque[:opaque] === copied_opaque[:opaque_alias]
    @test copied_opaque[:opaque].payload === copied_opaque[:shared]
    @test copied_opaque[:shared] !== opaque_payload
end

@testitem "CanonicalGeneration metadata values copy independently per merge key" begin
    using Test
    using LambdaRegression: CanonicalGeneration

    CG = CanonicalGeneration
    struct SharedMetadataValueDerivation <: CG.CoreDerivation
        key::Symbol
        value::Vector{Any}
    end
    CG.derivation_metadata(derivation::SharedMetadataValueDerivation) =
        Dict{Symbol,Any}(derivation.key => derivation.value)

    shared = Any[:shared]
    merged = CG._merge_derivation_metadata((
        SharedMetadataValueDerivation(:left, shared),
        SharedMetadataValueDerivation(:right, shared),
    ))
    @test isequal(merged[:left], shared)
    @test isequal(merged[:right], shared)
    @test merged[:left] !== shared
    @test merged[:right] !== shared
    @test merged[:left] !== merged[:right]
end

@testitem "Scoped generation material ownership and relation borrowing" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyLike

    CG = CanonicalGeneration
    config = CG.GenerationConfig(providers=(CG.ContextHeadProvider(),))
    values = Dict{Symbol,Any}(:lifecycle_local => 2.0)
    types = Dict{Symbol,TyLike}(:lifecycle_local => FloatTy)
    inputs = (;
        global_consts=Dict{Symbol,Any}(:lifecycle_global => true),
        global_consts_ty=Dict{Symbol,TyLike}(:lifecycle_global => BoolTy),
        presets=Dict{Symbol,Any}(), preset_types=Dict{Symbol,TyLike}(),
    )
    inspect = (relation, output_values, output_types) ->
        (; relation, output_values, output_types)
    @test CG._uses_lazy_sampling(config)
    @test CG._scoped_lazy_relation_cache_key(
        config.weights, inputs.global_consts, inputs.global_consts_ty,
        values, types, inputs.presets, inputs.preset_types,
        config.providers, config.argument_planner, config.guarded_expansion_depth,
    ) !== nothing
    first = CG._with_scoped_generation_materials(inspect, config;
        inputs..., local_consts=values, local_consts_ty=types)
    second = CG._with_scoped_generation_materials(inspect, config;
        inputs..., local_consts=copy(values), local_consts_ty=copy(types))
    @test first.relation === second.relation
    @test first.output_values !== values
    @test first.output_types !== types
    @test first.output_values !== second.output_values
    @test first.output_types !== second.output_types
    @test second.relation.local_consts === second.output_values
    @test second.relation.local_consts_ty === second.output_types
    @test first.output_values == second.output_values == values

    masked = CG._with_scoped_generation_materials(inspect, config;
        inputs..., local_consts=values, local_consts_ty=types,
        source_local_consts=false, capability_mask=(:function_def,))
    @test isempty(masked.relation.local_consts)
    @test isempty(masked.relation.local_consts_ty)
    @test masked.output_values == values
    @test masked.output_types == types
    @test masked.relation.capability_mask == (:function_def,)

    private = CG._with_scoped_generation_materials(inspect, config;
        inputs..., local_consts=values, local_consts_ty=types, cache_relation=false)
    @test private.relation !== first.relation
    @test private.relation.local_consts !== private.output_values
    @test private.relation.local_consts == private.output_values
    private_masked = CG._with_scoped_generation_materials(inspect, config;
        inputs..., local_consts=values, local_consts_ty=types,
        source_local_consts=false, cache_relation=false)
    @test isempty(private_masked.relation.local_consts)
    @test isempty(private_masked.relation.local_consts_ty)
    @test private_masked.output_values == values
    @test private_masked.output_types == types

    float_globals = merge(inputs, (;
        global_consts=Dict{Symbol,Any}(:lifecycle_global => 1.0),
        global_consts_ty=Dict{Symbol,TyLike}(:lifecycle_global => FloatTy),
    ))
    @test CG._scoped_lazy_relation_cache_key(
        config.weights, float_globals.global_consts, float_globals.global_consts_ty,
        values, types, inputs.presets, inputs.preset_types,
        config.providers, config.argument_planner, config.guarded_expansion_depth,
    ) === nothing
    noncacheable = CG._with_scoped_generation_materials(inspect, config;
        float_globals..., local_consts=values, local_consts_ty=types)
    @test noncacheable.relation.local_consts !== noncacheable.output_values
    @test noncacheable.output_values == values
end

@testitem "Scoped generation borrowing releases locks on constructor failure" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, TyLike

    CG = CanonicalGeneration
    config = CG.GenerationConfig(providers=(CG.ContextHeadProvider(),))
    entry = Ref{Any}()
    @test_throws ErrorException CG._with_scoped_generation_materials(config;
        global_consts=Dict{Symbol,Any}(:lifecycle_failure => true),
        global_consts_ty=Dict{Symbol,TyLike}(:lifecycle_failure => BoolTy),
        local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}(),
        presets=Dict{Symbol,Any}(), preset_types=Dict{Symbol,TyLike}(),
    ) do relation, output_values, output_types
        entry[] = only(filter(item -> item.relation === relation,
            collect(values(CG._SCOPED_LAZY_RELATION_CACHE))))
        @test islocked(entry[].lock)
        @test relation.local_consts === output_values
        @test relation.local_consts_ty === output_types
        error("bounded constructor failure")
    end
    @test !islocked(entry[].lock)
end

@testitem "FunctionDef verification inspects actual FunctionDef nodes" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, ConstNode
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    result = CanonicalGeneration.GenerationResult(
        AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(ConstNode(:ZERO))),
        FloatTy,
        Dict{Symbol,Any}(:ZERO => 0.0),
        Dict{Symbol,TyLike}(:ZERO => FloatTy),
        Dict{Symbol,Any}(),
    )

    verified = CanonicalGeneration._verify_functiondef_result(
        result,
        FloatTy;
        kwargs=(;),
    )

    @test verified isa Err

    global_values = Dict{Symbol,Any}(:v => 1.0)
    valid = CanonicalGeneration.GenerationResult(
        AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            AbsNode(:x, ConstNode(:v)),
        )),
        TyFun(FloatTy, FloatTy),
        Dict{Symbol,Any}(),
        Dict{Symbol,TyLike}(),
        Dict{Symbol,Any}(),
    )
    context = CanonicalGeneration._verification_context(
        valid,
        (; global_consts=global_values),
    )
    @test context[:v] == FloatTy
    verified_global = CanonicalGeneration._verify_functiondef_result(
        valid,
        TyFun(FloatTy, FloatTy);
        kwargs=(; global_consts=global_values),
    )
    @test verified_global isa CanonicalGeneration.GenerationResult
    @test verified_global.metadata[:functiondef_verification] == :verified

    shadowed = CanonicalGeneration.GenerationResult(
        valid.tree,
        valid.target_ty,
        Dict{Symbol,Any}(:v => 1.0),
        Dict{Symbol,TyLike}(:v => FloatTy),
        Dict{Symbol,Any}(),
    )
    shadow_context = CanonicalGeneration._verification_context(
        shadowed,
        (; global_consts_ty=Dict{Symbol,Any}(:v => BoolTy), local_consts=Dict{Symbol,Any}(:v => 1.0)),
    )
    @test shadow_context[:v] == FloatTy
    verified_shadow = CanonicalGeneration._verify_functiondef_result(
        shadowed,
        TyFun(FloatTy, FloatTy);
        kwargs=(; global_consts_ty=Dict{Symbol,Any}(:v => BoolTy), local_consts=Dict{Symbol,Any}(:v => 1.0)),
    )
    @test verified_shadow isa CanonicalGeneration.GenerationResult
end

@testitem "FunctionDef FunctionDef samples recursive mu heads under budget" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyMu, iscompatible, make_tyvar_factory

    CG = CanonicalGeneration
    binder = make_tyvar_factory()(0)
    target = TyMu(binder, TyFun(IntTy, binder))
    result = CanonicalGeneration.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => 1),
        global_consts_ty=Dict{Symbol,Any}(:mk => TyFun(IntTy, target), :seed => IntTy),
        config=CG.GenerationConfig(;
            rng=Random.Xoshiro(2000),
            providers=(CG.ContextHeadProvider(), CanonicalGeneration.FunctionDefProvider(
                max_nesting=1,
                body_attempts=4,
                admissibility=CanonicalGeneration.GenuineSelfCallFilter(CanonicalGeneration.AcceptFunctionDef()),
            )),
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
    @test result.tree.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode
    @test result.metadata[:sampling_nodes_used] <= 12
    @test iscompatible(
        LambdaRegression.TypeSystem.infer_term(result.tree, result.local_const_types),
        target,
    )
    entry = result.metadata[:derivation_metadata][:function_defs][1]
    @test :recursive_type_query in entry[:used_capabilities]
end

@testitem "FunctionDef FunctionDef attempts share one root budget" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
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
    provider = CanonicalGeneration.FunctionDefProvider(
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
    other_provider = CanonicalGeneration.FunctionDefProvider(max_nesting=1, body_attempts=3)
    other_budget = only(CG.generation_root_requirements(
        other_provider,
        relation,
        bare_query,
    ))
    combined_query = CG._derive_generation_query(
        bare_query;
        requirements=(budget, other_budget),
    )
    @test CanonicalGeneration._functiondef_attempt_budget(combined_query, provider) === budget
    @test CanonicalGeneration._functiondef_attempt_budget(combined_query, other_provider) === other_budget

    twin_provider = CanonicalGeneration.FunctionDefProvider(max_nesting=1, body_attempts=3)
    other_twin_provider = CanonicalGeneration.FunctionDefProvider(max_nesting=1, body_attempts=3)
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
    @test CanonicalGeneration._functiondef_attempt_budget(twin_query, twin_provider) === twin_budget
    @test CanonicalGeneration._functiondef_attempt_budget(twin_query, other_twin_provider) === other_twin_budget

    layered_provider = CanonicalGeneration.FunctionDefProvider(max_nesting=2, body_attempts=2)
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
        requirements=(CanonicalGeneration.FunctionDefFuel(1),),
    )
    @test CanonicalGeneration._functiondef_attempts_remaining(layered_root, layered_provider) == 2
    @test CanonicalGeneration._consume_functiondef_attempt!(layered_nested, layered_provider)
    @test CanonicalGeneration._functiondef_attempts_remaining(layered_nested, layered_provider) == 1
    @test CanonicalGeneration._functiondef_attempts_remaining(layered_root, layered_provider) == 2

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
        requirements=(CanonicalGeneration.FunctionDefSelfBinder(0),),
    )
    nested_base = CanonicalGeneration._functiondef_base_witness(
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
            CanonicalGeneration.FunctionDefSelfBinder(0),
            CG.ForbidsScopeBinder(1),
        ),
    )
    fully_forbidden_base = CanonicalGeneration._functiondef_base_witness(
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
    owned_base = CanonicalGeneration._functiondef_base_witness(
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

@testitem "FunctionDef signature predicates preserve ordered binder semantics" begin
    using Test
    using LambdaRegression: CanonicalGeneration

    CG = CanonicalGeneration

    struct OffsetSignatureArgs <: AbstractVector{Any}
        values::Vector{Any}
        visited::Vector{Int}
    end
    Base.size(values::OffsetSignatureArgs) = (length(values.values),)
    Base.axes(values::OffsetSignatureArgs) = (0:(length(values.values) - 1),)
    function Base.getindex(values::OffsetSignatureArgs, index::Int)
        index in axes(values, 1) || throw(BoundsError(values, index))
        push!(values.visited, index)
        return values.values[index + 1]
    end
    function Base.iterate(values::OffsetSignatureArgs, state=first(axes(values, 1)))
        state > last(axes(values, 1)) && return nothing
        return (values[state], state + 1)
    end

    var(index) = (:var, index, :ty)
    eta_one = (:lambda, :arg, (:scope_application, 1, :fun, (var(0),)))
    eta_two = (
        :lambda,
        :first_arg,
        (:lambda, :second_arg, (:scope_application, 2, :fun, (var(1), var(0)))),
    )
    @test CG._is_eta_forwarding_signature(eta_one, 0)
    @test CG._is_eta_forwarding_signature(eta_two, 0)
    @test !CG._is_eta_forwarding_signature((:function_def, :name, :type, eta_one), 0)
    @test !CG._is_eta_forwarding_signature(
        (:lambda, :arg, (:scope_application, 1, :fun, (var(1),))),
        0,
    )

    unchanged = (:scope_application, 1, :fun, (var(0),))
    changed = (:scope_application, 1, :fun, (var(1),))
    @test CG._has_unchanged_self_call_signature(unchanged, 1, [0])
    @test CG._has_unchanged_self_call_signature((:head, :global, :f, :type, (unchanged,)), 1, [0])
    @test !CG._has_unchanged_self_call_signature(changed, 1, [0])
    @test !CG._has_unchanged_self_call_signature((:function_def, :name, :type, unchanged), 1, [0])
    @test !CG._has_unchanged_self_call_signature(
        (:lambda, :arg, (:scope_application, 1, :fun, ())),
        1,
        [0],
    )

    offset_args = OffsetSignatureArgs(Any[var(0)], Int[])
    offset_unchanged = (:scope_application, 1, :fun, offset_args)
    @test CG._has_unchanged_self_call_signature(offset_unchanged, 1, [0])
    @test offset_args.visited == [0]

    safe_marker = (:safe_list_access, :UNCONS_CASE, :list, :result, :source, :body, :fallback)
    early_arguments = OffsetSignatureArgs(Any[var(0), safe_marker, var(2)], Int[])
    early_call = (:scope_application, 0, :fun, early_arguments)
    @test CG._has_unguarded_safe_list_recursive_argument(early_call, 0)
    @test early_arguments.visited == [0, 1]

    safe_from_source = (
        :safe_list_access, :UNCONS_CASE, :list, :result,
        (:scope_application, 0, :fun, (safe_marker,)),
        :body_not_visited,
        :fallback_not_visited,
    )
    @test CG._has_unguarded_safe_list_recursive_argument(safe_from_source, 0)
    guarded_body_call = (:scope_application, 2, :fun, (safe_marker,))
    safe_in_guarded_body = (
        :safe_list_access, :UNCONS_CASE, :list, :result,
        :source,
        guarded_body_call,
        :fallback,
    )
    @test !CG._has_unguarded_safe_list_recursive_argument(safe_in_guarded_body, 0)
    recursive_fallback = (:scope_application, 0, :fun, (var(0),))
    safe_in_fallback = (
        :safe_list_access, :UNCONS_CASE, :list, :result,
        :source,
        :body,
        recursive_fallback,
    )
    @test CG._has_unguarded_safe_list_recursive_argument(safe_in_fallback, 0)

    for malformed in (
        (:lambda, :arg),
        (:scope_application, 0, :fun),
        (:unknown, eta_one),
    )
        @test !CG._is_eta_forwarding_signature(malformed, 0)
        @test !CG._has_unchanged_self_call_signature(malformed, 0, [0])
        @test !CG._has_unguarded_safe_list_recursive_argument(malformed, 0)
    end
end

@testitem "Required-scope plan vectors preserve shape ownership and RNG" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun

    CG = CanonicalGeneration

    indices = Int[1, 2]
    original_indices = copy(indices)
    observed_placements = Set{Tuple{Int,Int}}()
    for seed in 1:64
        assigned = CG._distribute_scope_requirements(indices, 2, Random.Xoshiro(seed))
        @test assigned isa Vector{Vector{Int}}
        @test sort!(vcat(assigned...)) == original_indices
        @test indices == original_indices

        replayed = CG._distribute_scope_requirements(indices, 2, Random.Xoshiro(seed))
        @test replayed == assigned
        @test indices == original_indices

        first_locations = findall(bucket -> 1 in bucket, assigned)
        second_locations = findall(bucket -> 2 in bucket, assigned)
        @test length(first_locations) == 1
        @test length(second_locations) == 1
        if length(first_locations) == 1 && length(second_locations) == 1
            push!(observed_placements, (only(first_locations), only(second_locations)))
        end
    end
    @test observed_placements == Set([(1, 1), (1, 2), (2, 1), (2, 2)])

    empty_assignment = CG._distribute_scope_requirements(
        Int[], 0, Random.Xoshiro(1))
    @test empty_assignment isa Vector{Vector{Int}}
    @test isempty(empty_assignment)
    @test_throws ArgumentError CG._distribute_scope_requirements(
        Int[1], 0, Random.Xoshiro(1))

    app_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=1),
    )
    prefix_scope = Ty[FloatTy, TyFun(FloatTy, FloatTy), FloatTy]
    prefix_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=prefix_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=64,
        requirements=(CG.UsesScopeBinder(1), CG.UsesScopeBinder(2)),
    )
    prefix_plans = CG._fast_required_scope_application_plans(
        app_relation,
        FloatTy,
        prefix_scope,
        5,
        1,
        0,
        prefix_query,
    )
    prefix_plan_index = findfirst(
        plan -> !isempty(plan.exact_choices) && !isempty(plan.prefix_choices),
        prefix_plans,
    )
    @test prefix_plan_index !== nothing
    if prefix_plan_index !== nothing
        prefix_plan = prefix_plans[prefix_plan_index]
        @test prefix_plan.forced isa Vector{Int}
        @test prefix_plan.exact_choices isa Vector{CG._RequiredScopeExactChoice}
        @test prefix_plan.prefix_choices isa Vector{Int}
        @test CG._fast_required_scope_application_plans(
            app_relation, FloatTy, prefix_scope, 5, 1, 0, prefix_query,
        ) === prefix_plans

        saved_forced = copy(prefix_plan.forced)
        saved_forced_vector = prefix_plan.forced
        saved_exact_choices = prefix_plan.exact_choices
        saved_indices = [choice.index for choice in prefix_plan.exact_choices]
        saved_arguments = [copy(choice.arguments) for choice in prefix_plan.exact_choices]
        saved_argument_vectors = [choice.arguments for choice in prefix_plan.exact_choices]
        saved_prefix_choices = copy(prefix_plan.prefix_choices)
        saved_prefix_vector = prefix_plan.prefix_choices
        @test all(choice.arguments isa Vector{Int} for choice in prefix_plan.exact_choices)

        CG._materialize_fast_required_scope_application(
            CG.DerivationSamplingState(Random.Xoshiro(42), 64),
            app_relation,
            prefix_plan,
            prefix_scope,
            4,
            0,
            1,
            0,
            false,
            prefix_query,
        )

        @test prefix_plan.forced === saved_forced_vector
        @test prefix_plan.forced == saved_forced
        @test prefix_plan.exact_choices === saved_exact_choices
        @test [choice.index for choice in prefix_plan.exact_choices] == saved_indices
        @test [choice.arguments for choice in prefix_plan.exact_choices] == saved_arguments
        @test all(
            prefix_plan.exact_choices[index].arguments === saved_argument_vectors[index]
            for index in eachindex(saved_argument_vectors)
        )
        @test prefix_plan.prefix_choices === saved_prefix_vector
        @test prefix_plan.prefix_choices == saved_prefix_choices
    end
end
@testitem "SafeList plan caches use full structural keys" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyFun, typekey

    CG = CanonicalGeneration
    target = FloatTy
    scope = Ty[FloatTy, FloatTy, FloatTy]
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=1, variable=1),
    )
    query(index, depth) = CG.GenerationQuery(
        target;
        scope_tys=scope,
        depth_left=depth,
        provider_fuel=depth,
        node_budget=24,
        requirements=(CG.UsesScopeBinder(index),),
    )
    known_key = CG._known_key(scope, target)

    first_query = query(1, 5)
    first_plans = CG._fast_required_productive_head_plans(
        relation, target, scope, 5, first_query,
    )
    first_key = CG._StructuralKey((:fast_required_productive_head_plans, known_key, 5, (1,)))
    @test get(relation.sampling_options, first_key, nothing) === first_plans
    @test !haskey(relation.sampling_options, first_key.value)
    @test CG._fast_required_productive_head_plans(relation, target, scope, 5, first_query) === first_plans

    second_query = query(2, 5)
    second_plans = CG._fast_required_productive_head_plans(
        relation, target, scope, 5, second_query,
    )
    second_key = CG._StructuralKey((:fast_required_productive_head_plans, known_key, 5, (2,)))
    @test second_plans !== first_plans
    @test get(relation.sampling_options, second_key, nothing) === second_plans
    deeper_plans = CG._fast_required_productive_head_plans(relation, target, scope, 6, query(1, 6))
    deeper_key = CG._StructuralKey((:fast_required_productive_head_plans, known_key, 6, (1,)))
    @test deeper_plans !== first_plans
    @test get(relation.sampling_options, deeper_key, nothing) === deeper_plans

    catalog = CG._relation_catalog(relation, target, scope)
    rules_key = CG._StructuralKey((:productive_rules_for_target, known_key, typekey(target)))
    rules = CG._productive_rules_for_target(relation, catalog, known_key, target)
    @test get(relation.sampling_options, rules_key, nothing) === rules
    @test CG._productive_rules_for_target(relation, catalog, known_key, target) === rules
    higher_target = TyFun(FloatTy, FloatTy)
    higher_catalog_key = CG._known_key(scope, higher_target)
    higher_catalog = CG._relation_catalog(relation, higher_target, scope)
    higher_rules_key = CG._StructuralKey((
        :productive_rules_for_target,
        higher_catalog_key,
        typekey(higher_target),
    ))
    higher_rules = CG._productive_rules_for_target(
        relation,
        higher_catalog,
        higher_catalog_key,
        higher_target,
    )
    @test higher_rules !== rules
    @test get(relation.sampling_options, higher_rules_key, nothing) === higher_rules

    first_split = CG._fast_split_requirement_head_plans(
        relation, target, scope, 5, 1, 0, first_query,
    )
    split_key = CG._StructuralKey((:fast_split_requirement_head_plans, known_key, 5, (0,), (1,)))
    @test get(relation.sampling_options, split_key, nothing) === first_split
    @test CG._fast_split_requirement_head_plans(
        relation, target, scope, 5, 1, 0, first_query,
    ) === first_split
    second_split = CG._fast_split_requirement_head_plans(
        relation, target, scope, 5, 1, 0, second_query,
    )
    second_split_key = CG._StructuralKey((:fast_split_requirement_head_plans, known_key, 5, (0,), (2,)))
    @test second_split !== first_split
    @test get(relation.sampling_options, second_split_key, nothing) === second_split
end

@testitem "CanonicalGeneration cache keys preserve ordered scope tuples" begin
    using Test
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyArray, TyFun, TyList, TyMu, TyNat,
        make_tyvar_factory, typekey

    CG = CanonicalGeneration
    recursive_var = make_tyvar_factory()(0)
    recursive = TyMu(
        recursive_var,
        TyFun(TyFun(FloatTy, BoolTy), TyFun(recursive_var, TyList(FloatTy))),
    )
    nested = TyFun(TyList(BoolTy), TyArray(TyNat(3), TyFun(FloatTy, recursive)))
    vector_scope = Ty[FloatTy, BoolTy, nested, recursive]
    long_vector_scope = Ty[vector_scope[mod1(index, length(vector_scope))] for index in 1:40]
    scopes = (
        Ty[],
        (),
        (FloatTy,),
        vector_scope,
        (FloatTy, nested, recursive),
        long_vector_scope,
        Tuple(long_vector_scope),
    )

    providers = (CG.ContextHeadProvider(),)
    relation = CG.CoreRelation(
        providers=providers,
        weights=CG.GenerationWeights(default=1),
    )
    target = TyFun(FloatTy, recursive)

    for scope in scopes
        original_entries = collect(scope)
        old_typekeys = Tuple(typekey.(scope))
        old_cache_keys = Tuple(CG._cache_type_key.(scope))

        typekeys = CG._scope_typekeys(scope)
        cache_keys = CG._scope_typekeys(scope, CG._cache_type_key)

        @test typekeys == old_typekeys
        @test isequal(typekeys, old_typekeys)
        @test hash(typekeys) == hash(old_typekeys)
        @test cache_keys == old_cache_keys
        @test isequal(cache_keys, old_cache_keys)
        @test hash(cache_keys) == hash(old_cache_keys)
        @test length(typekeys) == length(scope)
        @test length(cache_keys) == length(scope)
        @test all(scope[index] === original_entries[index] for index in eachindex(original_entries))

        query = CG.GenerationQuery(
            target;
            scope_tys=scope,
            depth_left=3,
            guarded_left=1,
            provider_fuel=2,
            node_budget=16,
        )
        old_query_scope_key = Tuple(CG._cache_type_key.(query.scope_tys))
        expected_query_key = (
            :generation_query,
            query.capability_mask,
            CG._cacheable_providers_key(providers),
            CG._cache_type_key(query.target_ty),
            old_query_scope_key,
            query.depth_left,
            query.guarded_left,
            query.provider_fuel,
            query.node_budget,
            CG._rigid_demand_key(query),
            CG._cacheable_requirements_key(query.requirements),
        )
        @test CG._generation_query_cache_key(query, providers) == expected_query_key

        effective = CG._effective_generation_query(relation, query)
        expected_witness_key = (
            :known_witness,
            CG._cache_type_key(effective.target_ty),
            Tuple(CG._cache_type_key.(effective.scope_tys)),
            effective.depth_left,
            effective.capability_mask,
            CG._weights_key(relation.weights),
            relation.guarded_expansion_depth,
            CG._cacheable_global_consts_key(relation.global_consts, relation.global_consts_ty),
            CG._cacheable_local_context_key(relation.local_consts, relation.local_consts_ty),
            CG._cacheable_providers_key(CG._admitted_providers(relation.providers, effective.capability_mask)),
            CG._forbidden_scope_binders(effective),
        )
        @test CG._known_witness_cache_key(relation, query) == expected_witness_key
    end
end
@testitem "FunctionDef leading parameter identity survives later local binders" begin
    using Test
    using LambdaRegression.TypeSystem: TyFun, FloatTy, Ty
    CG = LambdaRegression.CanonicalGeneration

    function_ty = TyFun(FloatTy, FloatTy)
    root = CG.GenerationQuery(function_ty; scope_tys=(function_ty,), depth_left=4,
        node_budget=12, requirements=(CG.FunctionDefSelfBinder(0), CG.CallsScopeBinder(0)))
    first = CG._derive_generation_query(root; target_ty=FloatTy,
        added_scope_tys=(FloatTy,), depth_left=3)
    leading = CG._advance_functiondef_prefix(first)
    marker = only(r for r in leading.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (1, 1, true)

    closed = CG._close_functiondef_prefix(leading)
    local_scope = CG._derive_generation_query(closed; added_scope_tys=(FloatTy, FloatTy),
        depth_left=2)
    marker = only(r for r in local_scope.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (3, 1, false)
    @test marker.index - 1 == 2 # original parameter, not either new local binder

    later_lambda = CG._derive_generation_query(local_scope; added_scope_tys=(FloatTy,),
        depth_left=1)
    later_lambda = CG._advance_functiondef_prefix(later_lambda)
    marker = only(r for r in later_lambda.requirements if r isa CG.FunctionDefSelfBinder)
    @test (marker.index, marker.leading_parameters, marker.prefix_open) == (4, 1, false)
    @test CG.generation_requirement_cache_key(marker) !=
        CG.generation_requirement_cache_key(CG.FunctionDefSelfBinder(4))
end

@testitem "FunctionDef changed self argument uses a finite nonbare array root" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyArray, TyFun, TyList, TyNat, FloatTy
    CG = LambdaRegression.CanonicalGeneration

    array_ty = TyArray(TyNat(4), FloatTy)
    self_ty = TyFun(array_ty, FloatTy)
    scope = Ty[TyList(FloatTy), FloatTy, array_ty, self_ty] # tail, head, xs, self
    marker = CG.FunctionDefSelfBinder(3, 1, false)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        array_source=1, amap=1, scope_application=1, constraint_evidence=1e12)
    function relation()
        CG.CoreRelation(
            global_consts=Dict{Symbol,Any}(:SQR => LambdaRegression.Primitives.SQR),
            global_consts_ty=Dict{Symbol,Any}(:SQR => TyFun(FloatTy, FloatTy)),
            providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
            weights=weights,
        )
    end

    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        provider_fuel=5, node_budget=16,
        requirements=(CG.CallsScopeBinder(3), marker))
    rule_relation = relation()
    plans = CG._fast_required_scope_application_plans(
        rule_relation, FloatTy, scope, 5, 0, 0, query)
    self_plans = filter(plan -> plan.rule.index == 3, plans)
    @test length(self_plans) == 1

    if length(self_plans) == 1
        state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
        @test state.completion.probability == 0.0
        result = CG._materialize_fast_required_scope_application(
            state, rule_relation, only(self_plans), scope, 4, 0, 0, 0, false, query)

        @test result isa CG.ScopeApplicationDerivation
        if result isa CG.ScopeApplicationDerivation
            @test result.index == 3
            @test CG.derivation_calls_scope_binder(result, 3)
            argument = only(result.args)
            @test !(argument isa CG.VariableDerivation && argument.index == 2)
            @test argument isa CG.HeadApplicationDerivation
            if argument isa CG.HeadApplicationDerivation
                @test argument.head.name == :AMAP
                @test argument.args[1] isa CG.ConstantDerivation
                if argument.args[1] isa CG.ConstantDerivation
                    @test argument.args[1].head.name == :SQR
                end
                @test argument.args[2] isa CG.VariableDerivation
                if argument.args[2] isa CG.VariableDerivation
                    @test argument.args[2].index == 2
                end
            end
            @test CG.derivation_node_count(result) == 4
            @test 16 - state.nodes_left == CG.derivation_node_count(result)
        end
        @test !state.completion.completing
    end

    # The nonbare AMAP certificate costs three nodes; with only one child
    # node available, preparation reports :budget and the materializer keeps
    # the ordinary self(xs) route instead of claiming a false certificate.
    limited_relation = relation()
    too_small = CG._prepare_finite_core_nonbare_certificate(
        limited_relation, array_ty, scope, 4, 1, 2)
    @test too_small.status == :budget
    @test too_small.proven_cost == 3

    limited_query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        provider_fuel=5, node_budget=2,
        requirements=(CG.CallsScopeBinder(3), marker))
    limited_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 2)
    @test limited_state.completion.probability == 0.0
    fallback = CG._sample_fast_required_scope_application_derivation(
        limited_state, limited_relation, FloatTy, scope, 5, 0, 0, 0, false,
        limited_query)
    @test fallback isa CG.ScopeApplicationDerivation
    if fallback isa CG.ScopeApplicationDerivation
        @test fallback.index == 3
        @test only(fallback.args) isa CG.VariableDerivation
        if only(fallback.args) isa CG.VariableDerivation
            @test only(fallback.args).index == 2
        end
        @test CG.derivation_node_count(fallback) == 2
        @test 2 - limited_state.nodes_left == CG.derivation_node_count(fallback)
    end
    @test !limited_state.completion.completing

    # The complete non-bare plan keeps both same-typed mapper constants legal;
    # ordinary replay can explore either while preserving charges and duties.
    diverse_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(
            :SQR => LambdaRegression.Primitives.SQR,
            :CUBE => (x -> x * x * x),
        ),
        global_consts_ty=Dict{Symbol,Any}(
            :SQR => TyFun(FloatTy, FloatTy),
            :CUBE => TyFun(FloatTy, FloatTy),
        ),
        providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
        weights=weights,
    )
    prepared = CG._prepare_finite_core_nonbare_certificate(
        diverse_relation, array_ty, scope, 4, 14, 2)
    @test prepared isa NamedTuple
    if prepared isa NamedTuple
        @test prepared.status == :ready
        if prepared.status == :ready
            amap_roots = filter(option -> option.plan.kind == :head &&
                option.plan.value.name == :AMAP, prepared.options)
            @test !isempty(amap_roots)
            if !isempty(amap_roots)
                root = first(amap_roots)
                @test root.weight > 0
                @test root.cost == CG._evidence_plan_cost(prepared.world, root.plan)
                @test length(root.plan.children) == 2
                if length(root.plan.children) == 2
                    mapper_fact = root.plan.children[1]
                    mapper_menu = CG._evidence_menu!(prepared.world,
                        diverse_relation, mapper_fact)
                    mapper_choices = filter(plan -> plan.kind == :constant &&
                        plan.value.name in (:SQR, :CUBE), mapper_menu)
                    @test Set(plan.value.name for plan in mapper_choices) ==
                        Set((:SQR, :CUBE))
                    @test all(plan -> plan.weight > 0 &&
                        CG._evidence_plan_cost(prepared.world, plan) == 1,
                        mapper_choices)
                end
            end
        end
    end

    sampled_mappers = Set{Symbol}()
    for seed in 1:12
        sample_state = CG.DerivationSamplingState(Random.Xoshiro(seed), 16)
        sampled = CG._sample_preplanned_functiondef_call!(
            sample_state, diverse_relation, query)
        @test sampled isa CG.ScopeApplicationDerivation
        if sampled isa CG.ScopeApplicationDerivation
            @test sampled.index == 3
            @test CG.derivation_result_type(sampled) == FloatTy
            @test CG._generation_requirements_satisfied(sampled, query)
            @test CG.derivation_calls_scope_binder(sampled, 3)
            sampled_argument = only(sampled.args)
            @test !(sampled_argument isa CG.VariableDerivation && sampled_argument.index == 2)
            @test sampled_argument isa CG.HeadApplicationDerivation
            if sampled_argument isa CG.HeadApplicationDerivation
                @test sampled_argument.head.name == :AMAP
                mapper = sampled_argument.args[1]
                @test mapper isa CG.ConstantDerivation
                if mapper isa CG.ConstantDerivation
                    push!(sampled_mappers, mapper.head.name)
                    @test mapper.head.name in (:SQR, :CUBE)
                end
            end
            @test CG.derivation_node_count(sampled) == 16 - sample_state.nodes_left
            @test sample_state.nodes_left >= 0
            @test !sample_state.completion.completing
        end
    end
    @test sampled_mappers == Set((:SQR, :CUBE))
end

@testitem "FunctionDef preplanned self call is consumed by a mixed head packet" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyArray, TyFun, TyNat, FloatTy
    CG = LambdaRegression.CanonicalGeneration

    array_ty = TyArray(TyNat(4), FloatTy)
    self_ty = TyFun(array_ty, FloatTy)
    scope = Ty[FloatTy, array_ty, self_ty] # head, xs, self
    marker = CG.FunctionDefSelfBinder(2, 1, false)
    evidence_weight = 1e12
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        array_source=1, amap=1, scope_application=1, safe_list_access=1,
        constraint_evidence=evidence_weight)
    function make_relation()
        CG.CoreRelation(
            global_consts=Dict{Symbol,Any}(
                :ADD => (x -> y -> x + y),
                :SQR => LambdaRegression.Primitives.SQR,
            ),
            global_consts_ty=Dict{Symbol,Any}(
                :ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
                :SQR => TyFun(FloatTy, FloatTy),
            ),
            providers=(CG.ContextHeadProvider(), CG.AMapProvider(),
                CG.SafeListAccessProvider()),
            weights=weights,
        )
    end
    function packet_query(node_budget)
        CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
            guarded_left=0, provider_fuel=5, node_budget,
            requirements=(CG.UsesScopeBinder(0), CG.CallsScopeBinder(2), marker))
    end
    function add_rule(relation, query)
        only(filter(rule -> rule.head.name == :ADD,
            CG._certified_mixed_head_rules(relation, query)))
    end
    gate_passes(seed, count) = begin
        rng = Random.Xoshiro(seed)
        all(rand(rng) * (1.0 + evidence_weight) < evidence_weight for _ in 1:count)
    end

    relation = make_relation()
    query = packet_query(16)
    rule = add_rule(relation, query)
    # The fixed allocations put head's Use in ADD's first child and self's
    # Call in its second; safe_child=0 imposes no unrelated safe-list demand.
    assigned = [Int[0], Int[2]]
    @test gate_passes(733, 2)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
    result = CG._materialize_selected_mixed_head_packet(
        state, relation, rule, query, 4, true, assigned, 0)
    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :ADD
        @test CG.derivation_uses_scope_binder(result.args[1], 0)
        @test CG.derivation_calls_scope_binder(result.args[2], 2)
        @test CG._generation_requirements_satisfied(result, query)
        self_call = result.args[2]
        @test self_call isa CG.ScopeApplicationDerivation
        if self_call isa CG.ScopeApplicationDerivation
            @test self_call.index == 2
            argument = only(self_call.args)
            @test !(argument isa CG.VariableDerivation && argument.index == 1)
            @test argument isa CG.HeadApplicationDerivation
            if argument isa CG.HeadApplicationDerivation
                @test argument.head.name == :AMAP
                @test argument.args[1] isa CG.ConstantDerivation
                if argument.args[1] isa CG.ConstantDerivation
                    @test argument.args[1].head.name == :SQR
                end
                @test argument.args[2] isa CG.VariableDerivation
                if argument.args[2] isa CG.VariableDerivation
                    @test argument.args[2].index == 1
                end
            end
            @test CG.derivation_node_count(self_call) == 4
        end
        @test 16 - state.nodes_left == CG.derivation_node_count(result)
    end

    # This packet needs at least the ADD node, head variable, complete self
    # call and a three-node nonbare array argument. Failure refunds the caller.
    limited_relation = make_relation()
    limited_query = packet_query(3)
    limited_rule = add_rule(limited_relation, limited_query)
    limited_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 12)
    @test CG._materialize_selected_mixed_head_packet(
        limited_state, limited_relation, limited_rule, limited_query, 4, true,
        assigned, 0) === nothing
    @test limited_state.nodes_left == 12
    @test !limited_state.completion.completing

    # The isolated pure-Calls consumer succeeds with the same finite evidence
    # and charges precisely the replayed full self call.
    call_relation = make_relation()
    call_query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        provider_fuel=5, node_budget=12,
        requirements=(CG.CallsScopeBinder(2), marker))
    @test gate_passes(947, 1)
    call_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 12)
    call = CG._sample_preplanned_functiondef_call!(call_state, call_relation, call_query)
    @test call isa CG.ScopeApplicationDerivation
    if call isa CG.ScopeApplicationDerivation
        @test call.index == 2
        argument = only(call.args)
        @test !(argument isa CG.VariableDerivation && argument.index == 1)
        @test argument isa CG.HeadApplicationDerivation
        if argument isa CG.HeadApplicationDerivation
            @test argument.head.name == :AMAP
            @test argument.args[1] isa CG.ConstantDerivation
            if argument.args[1] isa CG.ConstantDerivation
                @test argument.args[1].head.name == :SQR
            end
            @test argument.args[2] isa CG.VariableDerivation
            if argument.args[2] isa CG.VariableDerivation
                @test argument.args[2].index == 1
            end
        end
        @test CG.derivation_node_count(call) == 4
        @test 12 - call_state.nodes_left == CG.derivation_node_count(call)
    end

    # xs is the only finite source array. Forbidding it makes the nonbare
    # argument certificate unavailable, and the pure consumer fails finitely.
    negative_relation = make_relation()
    forbidden = (1,)
    unavailable = CG._prepare_finite_core_nonbare_certificate(
        negative_relation, array_ty, scope, 4, 10, 1; forbidden)
    @test unavailable.status == :unavailable
    negative_query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        provider_fuel=5, node_budget=12,
        requirements=(CG.CallsScopeBinder(2), CG.ForbidsScopeBinder(1), marker))
    @test gate_passes(1181, 1)
    negative_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 12)
    @test CG._sample_preplanned_functiondef_call!(
        negative_state, negative_relation, negative_query) === nothing
    @test negative_state.nodes_left == 12
    @test !negative_state.completion.completing
end
