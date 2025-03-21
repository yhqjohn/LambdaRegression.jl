@testitem "FunctionDef crossover reads the common analysis contract" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Mutations.Populations:
        Program, TypedSubtreeSwapCrossover, crossover,
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_path
    using LambdaRegression.Utils: Err

    target = TyFun(IntTy, IntTy)
    type_context = Dict{Symbol,TyLike}(:one => IntTy)
    left = Program(
        AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0)))),
    )
    right = Program(
        AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, ConstNode(:one)))),
    )
    left_snapshot, right_snapshot = deepcopy(left.tree), deepcopy(right.tree)

    bounded_core = TypedSubtreeSwapCrossover(
        max_try=100,
        max_open_answers_per_pair=2,
        require_same_depth=false,
        min_sites=1,
    )

    result = crossover(
        bounded_core,
        left,
        right;
        rng=MersenneTwister(2000),
        global_consts_ty=type_context,
        target_a=target,
        target_b=target,
    )
    @test !(result isa Err)
    @test left.tree == left_snapshot
    @test right.tree == right_snapshot
    for child in result
        inferred = LambdaRegression.TypeSystem.infer_term(child.tree, type_context)
        @test !(inferred isa Err)
        @test LambdaRegression.TypeSystem.iscompatible(inferred, target)
    end

    # Algorithm W treats `(λx. body) arg` as let-like and therefore does not
    # attach metadata to the intermediate `AbsNode`; its body must still remain
    # reachable as a crossover site.
    letlike = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(
        :z,
        AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
    )))
    analysis = prepare_typed_analysis(letlike, type_context; target_ty=target)
    sites = analysis isa Err ? analysis : [
        typed_analysis_path(analysis, id) for id in typed_analysis_site_ids(analysis)
    ]
    @test !(sites isa Err)
    @test any(path -> path == (1, 1, 1, 1), sites)
end

@testitem "Typed crossover uses contextual directional transport" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover,
        crossover
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    int_function = TyFun(IntTy, IntTy)
    type_context = Dict{Symbol,TyLike}(
        :ADD => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :ZERO => IntTy,
        :USE_FUN => TyFun(int_function, IntTy),
    )
    left = Program(
        AstTree(AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:ADD), IndexNode(0)),
                ConstNode(:ZERO),
            ),
        )),
    )
    right = Program(
        AstTree(AbsNode(
            :x,
            AppNode(
                ConstNode(:USE_FUN),
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(ConstNode(:ADD), IndexNode(1)),
                        ConstNode(:ZERO),
                    ),
                ),
            ),
        )),
    )
    left_before, right_before = deepcopy(left), deepcopy(right)
    core = TypedSubtreeSwapCrossover(
        max_try=1,
        require_same_depth=false,
        min_sites=1,
        selection=PathCrossoverSelectionPolicy(
            [1, 1, 2],
            [1, 2, 1, 2],
        ),
    )

    result = crossover(
        core,
        left,
        right;
        rng=MersenneTwister(8000),
        global_consts_ty=type_context,
        target_a=int_function,
        target_b=int_function,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_left, child_right = result
        @test child_left.tree != left.tree
        @test child_right.tree != right.tree
        @test iscompatible(infer(child_left.tree, type_context), int_function)
        @test iscompatible(infer(child_right.tree, type_context), int_function)
    end
    @test left.tree == left_before.tree
    @test left.local_consts == left_before.local_consts
    @test left.local_consts_ty == left_before.local_consts_ty
    @test right.tree == right_before.tree
    @test right.local_consts == right_before.local_consts
    @test right.local_consts_ty == right_before.local_consts_ty
end

@testitem "FunctionDef recursive-body mutation preserves a genuine self call" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree

    Mutations = LambdaRegression.Mutations
    target = TyFun(FloatTy, FloatTy)
    type_context = Dict{Symbol,TyLike}(:SQR => TyFun(FloatTy, FloatTy))
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
    ))
    mutation = Mutations.FunctionDefMutation(
        :recursive_body;
        functiondef_provider=LambdaRegression.CanonicalGeneration.FunctionDefProvider(
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
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=8,
        global_consts_ty=type_context,
    )

    @test result isa AstTree
    @test result.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode
    @test LambdaRegression.ProgramAnalysis.functiondef_recursion_summary(result).self_calls > 0
    @test iscompatible(LambdaRegression.TypeSystem.infer_term(result, type_context), target)
end

@testitem "FunctionDef recursive admissibility distinguishes calls from passed self values" begin
    using Test
    using LambdaRegression
    using LambdaRegression.TypeSystem: FloatTy, TyFun

    CG = LambdaRegression.CanonicalGeneration
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
    filter = LambdaRegression.CanonicalGeneration.GenuineSelfCallFilter(candidate -> true)

    @test !filter(LambdaRegression.CanonicalGeneration.FunctionDefCandidate(target, passed_self, base, 0))
    @test filter(LambdaRegression.CanonicalGeneration.FunctionDefCandidate(target, called_self, base, 0))
end

@testitem "FunctionDef finite-body mutation preserves polymorphic let-like binders" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, TyScheme, get_type_ctx, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree

    Mutations = LambdaRegression.Mutations
    CG = LambdaRegression.CanonicalGeneration

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
        @test scope isa Mutations.FunctionDefScope
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
        :SQR => LambdaRegression.Primitives.SQR,
        :TRUE => LambdaRegression.Primitives.TRUE,
        :COND => LambdaRegression.Primitives.COND,
    )
    types = get_type_ctx(values)
    type_only_name = Symbol("##functiondef_scope_1_1")
    types[type_only_name] = TyFun(FloatTy, FloatTy)
    target = TyFun(FloatTy, FloatTy)
    source = AstTree(AppNode(
        AbsNode(:identity_binding,
            LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
                AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
            ),
        ),
        AbsNode(:id_arg, IndexNode(0)),
    ))
    preset_name = Symbol("##functiondef_scope_1")
    generator = SchemeUsingGenerator(false, nothing, nothing)
    mutation = Mutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=generator,
        generation_kwargs=(;
            presets=Dict{Symbol,Any}(preset_name => LambdaRegression.Primitives.SQR),
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
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
        local_consts=local_values,
        local_consts_ty=local_types,
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
    @test iscompatible(LambdaRegression.TypeSystem.infer_term(result, types), target)
    function_def = result.root.func.body::LambdaRegression.ASTs.NodeASTs.FunctionDefNode
    identity_refs = count(
        node -> node isa IndexNode && node.index == 2,
        AstTree(function_def.body),
    )
    @test identity_refs == 3
end

@testitem "FunctionDef mutation uses root-constrained binder schemes" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, FunctionDefNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyScheme
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.Utils: Err

    TypeSystem = LambdaRegression.TypeSystem
    Mutations = LambdaRegression.Mutations

    target = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    source = AstTree(AppNode(
        AbsNode(:id, FunctionDefNode(AbsNode(
            :x,
            FunctionDefNode(AbsNode(:y, IndexNode(0))),
        ))),
        AbsNode(:z, IndexNode(0)),
    ))
    captured = Ref{Any}(nothing)
    generator = function (
        mode,
        target_ty,
        scope,
        env,
        ctx,
        provider,
        generation_kwargs,
    )
        captured[] = (scope=scope, env=env, ctx=ctx)
        return Err(Mutations.NoEligibleSiteErr())
    end
    mutation = Mutations.FunctionDefMutation(
        :finite_body;
        site_weight=(mode, site, env, ctx) -> site.path == [1, 1, 1, 1] ? 1.0 : 0.0,
        replacement_generator=generator,
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(1),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(typeof(mutation) => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
    @test captured[] !== nothing
    scope = captured[].scope
    @test scope.functiondef_depth == 1
    @test length(scope.binder_types) == 3

    lambda_scheme, self_scheme, let_scheme = scope.binder_types
    @test lambda_scheme isa TyScheme && isempty(lambda_scheme.vars)
    @test TypeSystem.iscompatible(lambda_scheme.ty, FloatTy)
    @test self_scheme isa TyScheme && isempty(self_scheme.vars)
    @test TypeSystem.iscompatible(self_scheme.ty, target)
    @test let_scheme isa TyScheme && length(let_scheme.vars) == 1
    @test let_scheme.ty isa TyFun
    @test let_scheme.ty.arg === let_scheme.ty.res

    @test scope.full_slot_by_scope_index == [1, 2]
    @test haskey(scope.placeholder_by_slot, 3)
    @test captured[].ctx.typed_analysis isa
        LambdaRegression.ProgramAnalysis.TypedAnalysis
    @test captured[].env.tree.root !== captured[].ctx.typed_analysis.index.tree.root
end

@testitem "FunctionDef mutation carries analysis through a provider wrapper" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.Utils: Err

    Mutations = LambdaRegression.Mutations

    mutable struct DecoratedIndexedProvider <: Mutations.AbstractSiteProvider
        inner::IndexedProvider
    end
    Mutations.reset!(provider::DecoratedIndexedProvider) = Mutations.reset!(provider.inner)
    Mutations.prepare!(
        provider::DecoratedIndexedProvider,
        tree::AstTree,
        type_ctx::Dict{Symbol,<:TyLike},
        env::Any,
        ctx::NamedTuple,
    ) =
        Mutations.prepare!(provider.inner, tree, type_ctx, env, ctx)
    Mutations.sites(provider::DecoratedIndexedProvider) = Mutations.sites(provider.inner)
    Mutations.sites_by_kind(provider::DecoratedIndexedProvider, kind::Symbol) =
        Mutations.sites_by_kind(provider.inner, kind)

    captured = Ref{Any}(nothing)
    generator = function (mode, target_ty, scope, env, ctx, provider, generation_kwargs)
        captured[] = (scope=scope, env=env, ctx=ctx)
        return Err(Mutations.NoEligibleSiteErr())
    end
    mutation = Mutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=generator,
    )
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, IndexNode(0)),
    ))
    target = TyFun(FloatTy, FloatTy)
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
        cache_eligibility=true,
    )

    function probe()
        captured[] = nothing
        provider = DecoratedIndexedProvider(IndexedProvider())
        result = mutate_tree(
            source,
            target;
            rng=MersenneTwister(2000),
            mutations=[mutation],
            chooser,
            provider,
            max_try=1,
        )
        @test result isa Err
        @test result.value isa Mutations.MaxTryReachedErr
        @test captured[] !== nothing
        @test captured[].scope.functiondef_depth == 0
        return captured[].ctx.typed_analysis
    end

    analysis = probe()
    @test analysis isa LambdaRegression.ProgramAnalysis.TypedAnalysis
end

@testitem "FunctionDef finite modes reject FunctionDef replacements" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    Mutations = LambdaRegression.Mutations
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
            AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0)))),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    target = TyFun(FloatTy, FloatTy)
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
        cache_eligibility=true,
    )
    for mode in (:finite_body, :remove)
        mutation = Mutations.FunctionDefMutation(
            mode;
            replacement_generator=NestedFunctionDefGenerator(),
        )
        result = Mutations.mutate_tree(
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

    Mutations = LambdaRegression.Mutations
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
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    generator = ConfigInspectingGenerator(false)
    mutation = Mutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=generator,
        generation_kwargs=(;
            config=CG.GenerationConfig(rng=MersenneTwister(1111)),
        ),
    )
    caller_rng = MersenneTwister(1234)
    result = Mutations.mutate_tree(
        source,
        target;
        rng=caller_rng,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
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

    Mutations = LambdaRegression.Mutations
    binder = make_tyvar_factory()(0)
    target = TyMu(binder, TyFun(IntTy, binder))
    values = Dict{Symbol,Any}(:step => identity)
    type_context = Dict{Symbol,TyLike}(:step => TyFun(IntTy, target))
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(ConstNode(:step)))
    mutation = Mutations.FunctionDefMutation(
        :remove;
        functiondef_provider=LambdaRegression.CanonicalGeneration.FunctionDefProvider(max_nesting=1),
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
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        local_consts=values,
        local_consts_ty=type_context,
    )

    @test result isa AstTree
    @test !(result.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode)
    @test iscompatible(LambdaRegression.TypeSystem.infer_term(result, type_context), target)
end

@testitem "FunctionDef mutation keeps nonredundant local candidates" begin
    using Test
    using LambdaRegression
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyLike, TyNat

    Mutations = LambdaRegression.Mutations
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

    candidates, candidate_types = Mutations._candidate_local_context(ctx)

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
            :materialized_amap => LambdaRegression.LambdaRegression.Primitives.AMAP,
            :factory_array => [0.25, -0.5],
        ),
        local_consts_ty=Dict{Symbol,TyLike}(
            :materialized_amap => TyFun(TyFun(FloatTy, FloatTy), TyFun(vector_ty, vector_ty)),
            :factory_array => vector_ty,
        ),
    )
    generated_candidates, generated_types = Mutations._candidate_local_context(
        generated_ctx,
        LambdaRegression.CanonicalGeneration.test_catalog_providers(),
    )
    @test Set(keys(generated_candidates)) == Set((:materialized_amap, :factory_array))
    @test Set(keys(generated_types)) == Set((:materialized_amap, :factory_array))

    deduplicated_candidates, deduplicated_types =
        Mutations._candidate_local_context(
            generated_ctx,
            LambdaRegression.CanonicalGeneration.test_catalog_providers();
            deduplicate_provider_values=true,
        )
    @test collect(keys(deduplicated_candidates)) == [:factory_array]
    @test collect(keys(deduplicated_types)) == [:factory_array]

    default_mutation = Mutations.FunctionDefMutation(:finite_body)
    opted_in_mutation = Mutations.FunctionDefMutation(
        :finite_body;
        deduplicate_provider_values=true,
    )
    @test !default_mutation.deduplicate_provider_values
    @test opted_in_mutation.deduplicate_provider_values
end

@testitem "FunctionDef scope recovery reuses shared typed analysis" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    Mutations = LambdaRegression.Mutations
    target = TyFun(FloatTy, FloatTy)
    tree = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:inner, IndexNode(0))))
    type_context = Dict{Symbol,TyLike}()
    analysis = LambdaRegression.ProgramAnalysis.analyze_program(
        tree,
        type_context;
        target_ty=target,
    )
    @test analysis isa LambdaRegression.ProgramAnalysis.TypedAnalysis
    env = Mutations.MutationEnv(
        Random.Xoshiro(1111),
        target,
        type_context,
        tree,
        analysis,
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

    recovered = Mutations._visible_functiondef_context(tree, Int[], env, ctx)

    @test recovered.target_ty == target
    @test isempty(recovered.scope.scope_tys)
end

@testitem "FunctionDef scope recovery preserves the caller type variables" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type_ctx,
        make_tyvar_factory
    using LambdaRegression.Primitives: FADD

    Mutations = LambdaRegression.Mutations
    factory = make_tyvar_factory()
    α = factory(0)
    target = TyFun(α, α)
    tree = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:inner, AppNode(AppNode(ConstNode(:FADD), IndexNode(0)),
            ConstNode(:ONE)))))
    type_context = get_type_ctx(Dict{Symbol,Any}(:FADD => FADD, :ONE => 1.0))
    analysis = LambdaRegression.ProgramAnalysis.analyze_program(
        tree,
        type_context;
        target_ty=target,
    )
    @test analysis isa LambdaRegression.ProgramAnalysis.TypedAnalysis
    env = Mutations.MutationEnv(
        Random.Xoshiro(1234),
        target,
        type_context,
        tree,
        analysis,
        false,
        IdDict{Any,Any}(),
        Mutations.AdaptivePolicy(),
        IdDict{DataType,Mutations.MutationStat}(),
        Mutations.BloatPolicy(),
    )
    ctx = (;
        global_consts=Dict{Symbol,Any}(:FADD => FADD, :ONE => 1.0),
        global_consts_ty=type_context,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        tyfactory=make_tyvar_factory(),
    )
    recovered = Mutations._visible_functiondef_context(tree, Int[], env, ctx)
    @test recovered.target_ty == TyFun(FloatTy, FloatTy)
    @test α.instance === nothing
end

@testitem "FunctionDef mutation spends the enclosing nesting budget" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.Utils: Err

    Mutations = LambdaRegression.Mutations
    target = TyFun(FloatTy, FloatTy)
    type_context = Dict{Symbol,TyLike}(:SQR => target)
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
        ),
    ))
    inner_only = (mode, site, env, ctx) -> length(site.path) == 1 ? 1.0 : 0.0
    mutation = Mutations.FunctionDefMutation(
        :recursive_body;
        functiondef_provider=LambdaRegression.CanonicalGeneration.FunctionDefProvider(
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
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts_ty=type_context,
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

    Mutations = LambdaRegression.Mutations
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
            AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(body)),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    target = TyFun(FloatTy, FloatTy)
    values = Dict{Symbol,Any}(:consume_function => _ -> 0.0)
    type_context = Dict{Symbol,TyLike}(:consume_function => TyFun(target, FloatTy))
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    mutation = Mutations.FunctionDefMutation(
        :recursive_body;
        replacement_generator=PassedSelfGenerator(),
    )

    result = mutate_tree(
        source,
        target;
        rng=MersenneTwister(555),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=type_context,
    )

    @test result isa Err
end

@testitem "FunctionDef removal uses shared mutation validation" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, iscompatible
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser

    Mutations = LambdaRegression.Mutations
    target = TyFun(FloatTy, FloatTy)
    values = Dict{Symbol,Any}(:SQR => abs2)
    type_context = Dict{Symbol,TyLike}(:SQR => target)
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))),
    ))
    mutation = Mutations.FunctionDefMutation(:remove)

    result = Mutations.mutate_tree(
        source,
        target;
        rng=MersenneTwister(1234),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
        global_consts=values,
        global_consts_ty=type_context,
    )

    @test result isa AstTree
    @test !(result.root isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode)
    @test iscompatible(LambdaRegression.TypeSystem.infer_term(result, type_context), target)
end

@testitem "FunctionDef body mutation rejects a structural no-op" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    Mutations = LambdaRegression.Mutations
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
    source = AstTree(LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0))))
    mutation = Mutations.FunctionDefMutation(
        :finite_body;
        replacement_generator=SameBodyGenerator(),
    )
    result = Mutations.mutate_tree(
        source,
        target;
        rng=MersenneTwister(42),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
end

@testitem "FunctionDef failed final typing does not commit tree or constants" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: IndexedProvider, WeightedPairChooser
    using LambdaRegression.Utils: Err

    struct WrongRootReplacement end
    function (::WrongRootReplacement)(mode, target_ty, scope, env, ctx, provider, kwargs)
        return LambdaRegression.CanonicalGeneration.GenerationResult(
            AstTree(ConstNode(:attempt_value)),
            target_ty,
            Dict{Symbol,Any}(:attempt_value => 2.5),
            Dict{Symbol,TyLike}(:attempt_value => FloatTy),
            Dict{Symbol,Any}(),
        )
    end

    Mutations = LambdaRegression.Mutations
    target = TyFun(FloatTy, FloatTy)
    source = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0))))
    source_before = deepcopy(source)
    local_consts = Dict{Symbol,Any}(:stable => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:stable => FloatTy)
    consts_before = copy(local_consts)
    types_before = copy(local_consts_ty)
    mutation = Mutations.FunctionDefMutation(
        :remove;
        replacement_generator=WrongRootReplacement(),
    )

    result = Mutations.mutate_tree(
        source,
        target;
        rng=MersenneTwister(1),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(Mutations.FunctionDefMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
        local_consts,
        local_consts_ty,
    )

    @test result isa Err
    @test source == source_before
    @test local_consts == consts_before
    @test local_consts_ty == types_before
end
