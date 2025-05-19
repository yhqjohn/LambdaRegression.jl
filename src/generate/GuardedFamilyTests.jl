@testitem "CanonicalGeneration guarded family formula identity and zeros" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyFun
    CG=LambdaRegression.CanonicalGeneration
    P=LambdaRegression.Primitives
    head(name,value;metadata=(;))=CG.HeadEntry(name,:custom,TyFun(FloatTy,FloatTy),
        CG.Decoration(),CG.WeightBucket(:guarded_expansion);
        metadata=merge((;value,guarded=true),metadata))
    option(h,w)=(kind=:guarded_head,value=CG.CoreRule(h,FloatTy,[FloatTy],[:value]),weight=w)
    items=[option(head(:COND_A,P.COND),1.0),option(head(:COND_ALIAS,P.COND),5.0),
        option(head(:FOLD,P.AFOLDL),2.0)]
    relation(p;default=1)=CG.CoreRelation(weights=CG.GenerationWeights(;default,guarded_family_balance=p))
    @test CG._balance_guarded_families(items,CG.CoreRelation()) === items
    @test CG._balance_guarded_families(items,CG.CoreRelation(weights=CG.GenerationWeights(default=2))) === items
    @test CG._balance_guarded_families(items,relation(0)) === items
    @test [x.weight for x in CG._balance_guarded_families(items,relation(1))] ≈ [.8,4,3.2]
    @test [x.weight for x in CG._balance_guarded_families(items,relation(.5))] ≈ [.9,4.5,2.6]
    for p in (0,.5,1)
        balanced=CG._balance_guarded_families(items,relation(p))
        @test sum(x.weight for x in balanced)≈8
        @test balanced[2].weight/balanced[1].weight≈5
        @test all(x.weight>0 for x in balanced)
        @test getproperty.(balanced,:value)==getproperty.(items,:value)
    end
    zeros=[merge(x,(;weight=0.0)) for x in items]
    @test CG._balance_guarded_families(zeros,relation(1)) === zeros
    partly=copy(items);partly[1]=merge(partly[1],(;weight=0.0))
    @test first(CG._balance_guarded_families(partly,relation(1))).weight==0
    custom(name,source,metadata)=CG.HeadEntry(name,source,TyFun(FloatTy,FloatTy),
        CG.Decoration(),CG.WeightBucket(:guarded_expansion);metadata)
    a=custom(:a,:factory,(;factory=()->nothing,guarded_family=:same))
    b=custom(:alias,:another,(;factory=()->nothing,guarded_family=:same))
    @test CG._guarded_family_identity(a,relation(1))===CG._guarded_family_identity(b,relation(1))
    c=custom(:a,:factory,(;))
    @test CG._guarded_family_identity(c,relation(1))===(:head,:factory,:a)
    bad=custom(:bad,:factory,(;guarded_family=1))
    @test_throws ArgumentError CG._balance_guarded_families([option(bad,1.0)],relation(1))
    @test CG._balance_guarded_families([option(bad,1.0)],relation(0))[1].weight==1
    context=CG.CoreRelation(global_consts=Dict(:alias=>P.COND))
    alias=custom(:alias,:constant,(;))
    @test CG._guarded_family_identity(alias,context)===CG._guarded_family_identity(first(items).value.head,context)
    for p in (-.1,1.1,NaN,Inf)
        @test_throws ArgumentError CG.GenerationWeights(guarded_family_balance=p)
    end
    @test_throws ArgumentError CG.GenerationWeights(1.0,Dict(:guarded_family_balance=>2.0))
    @test CG._weights_key(CG.GenerationWeights()) != CG._weights_key(CG.GenerationWeights(guarded_family_balance=.5))
    @test CG.GenerationWeights(overrides=copy(relation(.5).weights.overrides)).overrides[:guarded_family_balance]==.5
end

@testitem "CanonicalGeneration guarded family production menu preserves language and higher order" begin
    using Test,Random
    using LambdaRegression.TypeSystem: Ty,FloatTy,TyFun,TyArray,TyNat,TyMu,newtyvar,TyLike,infer,iscompatible,typekey
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.Utils: Err
    CG=LambdaRegression.CanonicalGeneration
    array=TyArray(TyNat(4),FloatTy)
    globals=CG.test_catalog_consts()
    providers=(CG.default_head_providers()...,CG.FunctionDefProvider())
    rel(p)=CG.CoreRelation(;global_consts=globals,providers,
        weights=CG.GenerationWeights(guarded_expansion=2,guarded_family_balance=p))
    menus=[CG._guarded_sampling_options(rel(p),FloatTy,Ty[array]) for p in (0,.5,1)]
    @test length.(menus)==[16,16,16]
    @test all(x->sum(o.weight for o in x)≈2,menus)
    @test all(typekey(o.value.head.type)==typekey(menus[1][i].value.head.type)
        for menu in menus for (i,o) in enumerate(menu))
    mass(menu,name)=sum(o.weight for o in menu if o.value.head.name===name)
    @test mass(menus[1],:AFOLDL)≈.375
    @test mass(menus[2],:AFOLDL)≈.6875
    @test mass(menus[3],:AFOLDL)≈1
    @test mass(menus[3],:COND)≈1
    rng1,rng2=MersenneTwister(1),MersenneTwister(1)
    result_default=CG.generate_canonical(TyFun(FloatTy,FloatTy);global_consts=globals,
        config=CG.GenerationConfig(;rng=rng1,providers,sampling_depth_limit=3,sampling_node_budget=24))
    result_zero=CG.generate_canonical(TyFun(FloatTy,FloatTy);global_consts=globals,
        config=CG.GenerationConfig(;rng=rng2,providers,sampling_depth_limit=3,sampling_node_budget=24,
            weights=CG.GenerationWeights(guarded_family_balance=0)))
    @test typeof(result_default)==typeof(result_zero)
    if result_default isa Err
        @test typeof(result_default.value)==typeof(result_zero.value)
    else
        @test result_default.tree==result_zero.tree
        @test result_default.local_consts==result_zero.local_consts
    end
    @test rand(rng1)==rand(rng2)
    relation=rel(.5)
    scope=Ty[array,FloatTy]
    option=only(filter(o->o.value.head.name===:AFOLDL &&
        CG._afoldl_step_parts(o.value).acc_ty==FloatTy,
        CG._guarded_sampling_options(relation,FloatTy,scope)))
    state=CG.DerivationSamplingState(MersenneTwister(2000),48)
    query=CG.GenerationQuery(FloatTy;scope_tys=scope,depth_left=3,guarded_left=1,node_budget=48)
    derivation=CG._sample_option(state,relation,option,scope,3,1,false,query)
    @test derivation isa CG.CoreDerivation
    values,types=Dict{Symbol,Any}(),Dict{Symbol,TyLike}()
    root=CG.LambdaDerivation(FloatTy,CG.LambdaDerivation(array,derivation))
    tree=AstTree(CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),root,
        values,types,relation,MersenneTwister(2000)))
    @test iscompatible(infer(tree,merge(relation.global_consts_ty,types)),TyFun(FloatTy,TyFun(array,FloatTy)))
    # Recursive types retain the original explicit boundary without an extension.
    variable=newtyvar(0); mu=TyMu(variable,TyFun(FloatTy,variable))
    for p in (0,1)
        result=CG.generate_canonical(mu;weights=CG.GenerationWeights(guarded_family_balance=p))
        @test result isa Err
        @test result.value isa CG.UnsupportedCanonicalGenerationTarget
    end
end

@testitem "CanonicalGeneration guarded family rejects unsupported sampler combinations" begin
    using Test
    CG=LambdaRegression.CanonicalGeneration
    weight=CG.GenerationWeights(guarded_family_balance=.5)
    @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(weights=weight,
        sampler=CG.WeightedDerivationSampler(completion_probability=.1)))
    @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(weights=weight,
        sampler=(rng,set)->nothing))
    @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(weights=weight,
        argument_planner=(args...)->nothing))
    @test CG._uses_lazy_sampling(CG.GenerationConfig(weights=weight))
end
