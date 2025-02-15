@testitem "Procedural crossover is binder-aware and preserves core rejection" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Mutations.Populations:
        Individual, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    target = TyFun(IntTy, IntTy)
    type_context = Dict{Symbol,TyLike}(:one => IntTy)
    left = Individual(
        AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        target,
    )
    right = Individual(
        AstTree(Procedural.FunctionDefNode(AbsNode(:x, ConstNode(:one)))),
        target,
    )
    left_snapshot, right_snapshot = deepcopy(left.tree), deepcopy(right.tree)

    core_result = crossover(
        TypedSubtreeSwapCrossover(min_sites=1),
        left,
        right;
        global_consts_ty=type_context,
    )
    @test core_result isa Err
    @test core_result.value isa LambdaRegression.Mutations.Populations.UnsupportedCrossoverNodeErr

    op = ProceduralMutations.ProceduralTypedSubtreeSwapCrossover(
        max_try=100,
        min_sites=1,
    )
    result = crossover(
        ProceduralMutations.AutoTypedCrossover(procedural=op),
        left,
        right;
        rng=MersenneTwister(2000),
        global_consts_ty=type_context,
    )
    @test !(result isa Err)
    @test left.tree == left_snapshot
    @test right.tree == right_snapshot
    for child in result
        inferred = Procedural.infer_procedural(child.tree, type_context)
        @test !(inferred isa Err)
        @test LambdaRegression.TypeSystem.iscompatible(inferred, target)
    end

    # Algorithm W treats `(λx. body) arg` as let-like and therefore does not
    # attach metadata to the intermediate `AbsNode`; its body must still remain
    # reachable as a crossover site.
    letlike = AstTree(Procedural.FunctionDefNode(AbsNode(
        :z,
        AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
    )))
    sites = ProceduralMutations._collect_procedural_sites(
        letlike,
        type_context,
        LambdaRegression.TypeSystem.make_tyvar_factory(),
    )
    @test !(sites isa Err)
    @test any(site -> site.path == [1, 1, 1, 1], sites)
end

@testitem "FunctionDef recursive-body mutation preserves a genuine self call" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    target = TyFun(FloatTy, FloatTy)
    type_context = Dict{Symbol,TyLike}(:SQR => TyFun(FloatTy, FloatTy))
    source = AstTree(Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
    ))
    checker = (tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta)
    mutation = ProceduralMutations.FunctionDefMutation(
        :recursive_body;
        functiondef_provider=Procedural.FunctionDefProvider(
            max_nesting=1,
            body_attempts=24,
        ),
        generation_kwargs=(;
            sampling_depth_limit=8,
            sampling_node_budget=96,
            guarded_expansion_depth=0,
        ),
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(2000),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=8,
        global_consts_ty=type_context,
        type_inference=checker,
    )

    @test result isa AstTree
    @test result.root isa Procedural.FunctionDefNode
    @test Procedural.functiondef_recursion_summary(result).self_calls > 0
    @test iscompatible(Procedural.infer_procedural(result, type_context), target)
end

@testitem "FunctionDef recursive admissibility distinguishes calls from passed self values" begin
    using Test
    using LambdaRegression
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = LambdaRegression.CanonicalGeneration
    Procedural = LambdaRegression.Procedural
    target = TyFun(FloatTy, FloatTy)
    base = CG.VariableDerivation(0, target)
    passed_self = CG.LambdaDerivation(
        FloatTy,
        CG.ScopeApplicationDerivation(
            2,
            TyFun(target, FloatTy),
            CG.CoreDerivation[CG.VariableDerivation(1, target)],
        ),
    )
    called_self = CG.LambdaDerivation(
        FloatTy,
        CG.ScopeApplicationDerivation(
            1,
            target,
            CG.CoreDerivation[CG.VariableDerivation(0, FloatTy)],
        ),
    )
    filter = Procedural.GenuineSelfCallFilter(candidate -> true)

    @test !filter(Procedural.FunctionDefCandidate(target, passed_self, base, 0))
    @test filter(Procedural.FunctionDefCandidate(target, called_self, base, 0))
end

@testitem "FunctionDef finite-body mutation preserves polymorphic Scope binders" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, TyScheme, get_type_ctx, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    CG = LambdaRegression.CanonicalGeneration
    TS = LambdaRegression.TypeSystem

    mutable struct SchemeUsingGenerator
        saw_scheme::Bool
        result::Any
        placeholder::Union{Nothing,Symbol}
    end
    function (generator::SchemeUsingGenerator)(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        @test scope isa ProceduralMutations.FunctionDefScope
        slot = findfirst(ty -> ty isa TyScheme && !isempty(ty.vars), scope.binder_types)
        @test slot !== nothing
        generator.saw_scheme = true
        identity_name = scope.placeholder_by_slot[slot]
        generator.placeholder = identity_name
        identity = () -> ConstNode(identity_name)
        identity_alias_name = Symbol(identity_name, 1)
        identity_alias = () -> ConstNode(identity_alias_name)
        condition = AppNode(identity(), ConstNode(:TRUE))
        then_branch = AppNode(identity_alias(), IndexNode(0))
        else_branch = AppNode(identity_alias(), IndexNode(0))
        body = AbsNode(
            :x,
            AppNode(
                AppNode(
                    AppNode(ConstNode(:COND), condition),
                    then_branch,
                ),
                else_branch,
            ),
        )
        result = CG.GenerationResult(
            AstTree(body),
            target_ty,
            Dict{Symbol,Any}(
                identity_name => ctx.global_consts[identity_name],
                identity_alias_name => ctx.global_consts[identity_name],
            ),
            Dict{Symbol,TyLike}(
                identity_name => scope.binder_types[slot],
                identity_alias_name => TyFun(FloatTy, FloatTy),
            ),
            Dict{Symbol,Any}(),
        )
        generator.result = result
        return result
    end

    values = Dict{Symbol,Any}(
        :SQR => TS.SQR,
        :TRUE => TS.TRUE,
        :COND => TS.COND,
    )
    types = get_type_ctx(values)
    type_only_name = Symbol("##functiondef_scope_1_1")
    types[type_only_name] = TyFun(FloatTy, FloatTy)
    target = TyFun(FloatTy, FloatTy)
    source = AstTree(Procedural.ScopeNode(
        AstNode[AbsNode(:id_arg, IndexNode(0))],
        Procedural.FunctionDefNode(
            AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
        ),
    ))
    checker = (tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta)
    preset_name = Symbol("##functiondef_scope_1")
    generator = SchemeUsingGenerator(false, nothing, nothing)
    mutation = ProceduralMutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=generator,
        generation_kwargs=(;
            presets=Dict{Symbol,Any}(preset_name => TS.SQR),
            preset_types=Dict{Symbol,TyLike}(preset_name => TyFun(FloatTy, FloatTy)),
        ),
    )
    local_values = Dict{Symbol,Any}()
    local_types = Dict{Symbol,TyLike}()

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(555),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
        local_consts=local_values,
        local_consts_ty=local_types,
        type_inference=checker,
    )

    @test result isa AstTree
    @test generator.saw_scheme
    @test generator.placeholder != preset_name
    @test generator.placeholder != type_only_name
    @test haskey(generator.result.local_consts, generator.placeholder)
    @test haskey(generator.result.local_const_types, generator.placeholder)
    @test length(generator.result.local_consts) == 2
    @test isempty(local_values)
    @test isempty(local_types)
    @test iscompatible(Procedural.infer_procedural(result, types), target)
    function_def = (result.root::Procedural.ScopeNode).body::Procedural.FunctionDefNode
    identity_refs = count(
        node -> node isa IndexNode && node.index == 2,
        AstTree(function_def.body),
    )
    @test identity_refs == 3
end

@testitem "FunctionDef finite modes reject FunctionDef replacements" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    CG = LambdaRegression.CanonicalGeneration

    struct NestedFunctionDefGenerator end
    function (::NestedFunctionDefGenerator)(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        return CG.GenerationResult(
            AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0)))),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    target = TyFun(FloatTy, FloatTy)
    source = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
        cache_eligibility=true,
    )
    for mode in (:finite_body, :remove)
        mutation = ProceduralMutations.FunctionDefMutation(
            mode;
            replacement_generator=NestedFunctionDefGenerator(),
        )
        result = ProceduralMutations.mutate_procedural(
            source,
            target;
            rng=MersenneTwister(8000),
            mutations=[mutation],
            chooser,
            provider=IndexedProvider(),
            max_try=1,
        )
        @test result isa Err
    end
end

@testitem "FunctionDef custom generator receives the mutation RNG" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    CG = LambdaRegression.CanonicalGeneration

    mutable struct ConfigInspectingGenerator
        saw_runtime_rng::Bool
    end
    function (generator::ConfigInspectingGenerator)(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        generator.saw_runtime_rng = generation_kwargs.config.rng === env.rng
        return Err(LambdaRegression.Mutations.NoEligibleSiteErr())
    end

    target = TyFun(FloatTy, FloatTy)
    source = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    generator = ConfigInspectingGenerator(false)
    mutation = ProceduralMutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=generator,
        generation_kwargs=(;
            config=CG.GenerationConfig(rng=MersenneTwister(1111)),
        ),
    )
    caller_rng = MersenneTwister(1234)
    result = ProceduralMutations.mutate_procedural(
        source,
        target;
        rng=caller_rng,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
    @test generator.saw_runtime_rng
end

@testitem "FunctionDef removal retains finite witnesses for recursive targets" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, iscompatible, make_tyvar_factory
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    binder = make_tyvar_factory()(0)
    target = TyMu(binder, TyFun(IntTy, binder))
    values = Dict{Symbol,Any}(:step => identity)
    type_context = Dict{Symbol,TyLike}(:step => TyFun(IntTy, target))
    source = AstTree(Procedural.FunctionDefNode(ConstNode(:step)))
    checker = (tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta)
    mutation = ProceduralMutations.FunctionDefMutation(
        :remove;
        functiondef_provider=Procedural.FunctionDefProvider(max_nesting=1),
        include_local_consts=true,
        generation_kwargs=(;
            sampling_depth_limit=5,
            sampling_node_budget=32,
            guarded_expansion_depth=0,
        ),
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(42),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        local_consts=values,
        local_consts_ty=type_context,
        type_inference=checker,
    )

    @test result isa AstTree
    @test !(result.root isa Procedural.FunctionDefNode)
    @test iscompatible(Procedural.infer_procedural(result, type_context), target)
end

@testitem "FunctionDef mutation keeps nonredundant local candidates" begin
    using Test
    using LambdaRegression
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyLike, TyNat

    ProceduralMutations = LambdaRegression.ProceduralMutations
    unary = TyFun(FloatTy, FloatTy)
    global_values = Dict{Symbol,Any}(:GLOBAL_SQR => abs2)
    global_types = Dict{Symbol,TyLike}(:GLOBAL_SQR => unary)
    local_values = Dict{Symbol,Any}(
        :materialized_sqr => abs2,
        :float_factory_value => 1.25,
    )
    local_types = Dict{Symbol,TyLike}(
        :materialized_sqr => unary,
        :float_factory_value => FloatTy,
    )
    ctx = (;
        global_consts=global_values,
        global_consts_ty=global_types,
        local_consts=local_values,
        local_consts_ty=local_types,
    )

    candidates, candidate_types = ProceduralMutations._candidate_local_context(ctx)

    @test candidates == Dict{Symbol,Any}(:float_factory_value => 1.25)
    @test candidate_types == Dict{Symbol,TyLike}(:float_factory_value => FloatTy)
    @test local_values == Dict{Symbol,Any}(
        :materialized_sqr => abs2,
        :float_factory_value => 1.25,
    )
    @test local_types == Dict{Symbol,TyLike}(
        :materialized_sqr => unary,
        :float_factory_value => FloatTy,
    )

    vector_ty = TyArray(TyNat(2), FloatTy)
    generated_ctx = (;
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(
            :materialized_amap => LambdaRegression.TypeSystem.AMAP,
            :factory_array => [0.25, -0.5],
        ),
        local_consts_ty=Dict{Symbol,TyLike}(
            :materialized_amap => TyFun(TyFun(FloatTy, FloatTy), TyFun(vector_ty, vector_ty)),
            :factory_array => vector_ty,
        ),
    )
    generated_candidates, generated_types = ProceduralMutations._candidate_local_context(
        generated_ctx,
        LambdaRegression.CanonicalGeneration.test_catalog_providers(),
    )
    @test Set(keys(generated_candidates)) == Set((:materialized_amap, :factory_array))
    @test Set(keys(generated_types)) == Set((:materialized_amap, :factory_array))

    deduplicated_candidates, deduplicated_types =
        ProceduralMutations._candidate_local_context(
            generated_ctx,
            LambdaRegression.CanonicalGeneration.test_catalog_providers();
            deduplicate_provider_values=true,
        )
    @test collect(keys(deduplicated_candidates)) == [:factory_array]
    @test collect(keys(deduplicated_types)) == [:factory_array]

    default_mutation = ProceduralMutations.FunctionDefMutation(:finite_body)
    opted_in_mutation = ProceduralMutations.FunctionDefMutation(
        :finite_body;
        deduplicate_provider_values=true,
    )
    @test !default_mutation.deduplicate_provider_values
    @test opted_in_mutation.deduplicate_provider_values
end

@testitem "FunctionDef scope recovery reuses prepared type metadata" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    Mutations = LambdaRegression.Mutations
    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    target = TyFun(FloatTy, FloatTy)
    tree = AstTree(Procedural.FunctionDefNode(AbsNode(:inner, IndexNode(0))))
    meta = Dict{LambdaRegression.ASTs.NodeASTs.AstNode,Tuple{TyLike,TyLike}}()
    inferred = Procedural.infer_procedural(tree, Dict{Symbol,TyLike}(); meta)
    @test inferred isa TyFun
    env = Mutations.MutationEnv(
        Random.Xoshiro(1111),
        target,
        Dict{Symbol,TyLike}(),
        tree,
        meta,
        false,
        IdDict{Any,Any}(),
        Mutations.AdaptivePolicy(),
        IdDict{DataType,Mutations.MutationStat}(),
        Mutations.BloatPolicy(),
    )
    ctx = (;
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        tyfactory=(args...) -> error("scope recovery unexpectedly repeated inference"),
    )

    recovered = ProceduralMutations._visible_functiondef_context(tree, Int[], env, ctx)

    @test recovered.target_ty == target
    @test isempty(recovered.scope.scope_tys)
end

@testitem "FunctionDef mutation spends the enclosing nesting budget" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    target = TyFun(FloatTy, FloatTy)
    type_context = Dict{Symbol,TyLike}(:SQR => target)
    source = AstTree(Procedural.FunctionDefNode(
        Procedural.FunctionDefNode(
            AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
        ),
    ))
    checker = (tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta)
    inner_only = (mode, site, env, ctx) -> length(site.path) == 1 ? 1.0 : 0.0
    mutation = ProceduralMutations.FunctionDefMutation(
        :recursive_body;
        functiondef_provider=Procedural.FunctionDefProvider(
            max_nesting=1,
            body_attempts=24,
        ),
        generation_kwargs=(;
            sampling_depth_limit=8,
            sampling_node_budget=96,
            guarded_expansion_depth=0,
        ),
        site_weight=inner_only,
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(2000),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts_ty=type_context,
        type_inference=checker,
    )

    @test result isa Err
end

@testitem "FunctionDef recursive-body mutation rejects a passed self value" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    CG = LambdaRegression.CanonicalGeneration

    struct PassedSelfGenerator end
    function (::PassedSelfGenerator)(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        body = AbsNode(:x, AppNode(ConstNode(:consume_function), IndexNode(1)))
        return CG.GenerationResult(
            AstTree(Procedural.FunctionDefNode(body)),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    target = TyFun(FloatTy, FloatTy)
    values = Dict{Symbol,Any}(:consume_function => _ -> 0.0)
    type_context = Dict{Symbol,TyLike}(:consume_function => TyFun(target, FloatTy))
    source = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    checker = (tree, ctx, factory, meta) ->
        Procedural.infer_procedural(tree, ctx; factory, meta)
    mutation = ProceduralMutations.FunctionDefMutation(
        :recursive_body;
        replacement_generator=PassedSelfGenerator(),
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(555),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=type_context,
        type_inference=checker,
    )

    @test result isa Err
end

@testitem "procedural mutation wrapper supplies lowering-aware inference" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    target = TyFun(FloatTy, FloatTy)
    values = Dict{Symbol,Any}(:SQR => abs2)
    type_context = Dict{Symbol,TyLike}(:SQR => target)
    source = AstTree(Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
    ))
    mutation = ProceduralMutations.FunctionDefMutation(:remove)

    result = ProceduralMutations.mutate_procedural(
        source,
        target;
        rng=MersenneTwister(1234),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=type_context,
    )

    @test result isa AstTree
    @test !(result.root isa Procedural.FunctionDefNode)
    @test iscompatible(Procedural.infer_procedural(result, type_context), target)
end

@testitem "FunctionDef body mutation rejects a structural no-op" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    Procedural = LambdaRegression.Procedural
    ProceduralMutations = LambdaRegression.ProceduralMutations
    CG = LambdaRegression.CanonicalGeneration

    struct SameBodyGenerator end
    function (::SameBodyGenerator)(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        return CG.GenerationResult(
            AstTree(AbsNode(:x, IndexNode(0))),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    target = TyFun(FloatTy, FloatTy)
    source = AstTree(Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    mutation = ProceduralMutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=SameBodyGenerator(),
    )
    result = ProceduralMutations.mutate_procedural(
        source,
        target;
        rng=MersenneTwister(42),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ProceduralMutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
end
