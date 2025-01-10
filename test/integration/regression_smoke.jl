using Test
using Random

using LambdaRegression
using LambdaRegression.TypeSystem
using LambdaRegression.GenerateTree: AbsCtor, IndexCtor, GlobalConstCtor, ApplySpineCtor, FloatCtor, FocusedChooser
using LambdaRegression.Mutations
using LambdaRegression.Mutations.Populations: TypedSubtreeSwapCrossover
using LambdaRegression.Regression

function _common_components(; global_consts=Dict{Symbol,Any}(), global_consts_ty=Dict{Symbol,TyLike}(), input_slots, input_types, gen_ctors, gen_chooser, gen_max_depth, gen_max_try, mutations)
    return merge(
        default_components(
            global_consts=global_consts,
            global_consts_ty=global_consts_ty,
            input_slots=input_slots,
            input_types=input_types,
        ),
        (;
            gen_ctors=gen_ctors,
            gen_chooser=gen_chooser,
            gen_max_depth=gen_max_depth,
            gen_max_try=gen_max_try,
            mutations=mutations,
            mutation_chooser=Mutations.WeightedPairChooser(),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=20,
            mutation_prune_consts=true,
            crossover_op=nothing,
        )
    )
end

@testset "Regression integration A: matrix->matrix identity (single candidate)" begin
    n, m = 2, 3
    A = TyArray(TyNat(n), TyArray(TyNat(m), TypeSystem.FloatTy))
    target_ty = TyFun(A, A)

    rng_data = MersenneTwister(2000)
    X = [
        [randn(rng_data, m) for _ in 1:n],
        [randn(rng_data, m) for _ in 1:n],
        [randn(rng_data, m) for _ in 1:n],
    ]
    Y = deepcopy(X)
    data = SupervisedDataset(X, Y)

    comps = _common_components(
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
        gen_ctors=[AbsCtor(), IndexCtor()],
        gen_chooser=FocusedChooser(),
        gen_max_depth=6,
        gen_max_try=30,
        mutations=Mutations.AbstractMutation[],
    )
    cfg = default_config(
        population_size=20,
        generations=5,
        error_threshold=1e-12,
        init_mutation_rounds=0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        tournament_size=3,
        log=(; enabled=true, every=1, topk=3, io=stdout),
        parallel=(; enabled=false),
        max_reduce_steps=40,
    )
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(2000))
    res = fit!(reg, data)
    @test res.status == :threshold
    @test res.best !== nothing
end

@testset "Regression integration B: matrix->scalar constant (single candidate)" begin
    n, m = 2, 2
    A = TyArray(TyNat(n), TyArray(TyNat(m), TypeSystem.FloatTy))
    target_ty = TyFun(A, TypeSystem.FloatTy)

    global_consts = Dict{Symbol,Any}(:K => 0.5)
    global_consts_ty = get_type_ctx(global_consts)

    rng_data = MersenneTwister(1)
    X = [
        [randn(rng_data, m) for _ in 1:n],
        [randn(rng_data, m) for _ in 1:n],
    ]
    Y = fill(0.5, length(X))
    data = SupervisedDataset(X, Y)

    comps = _common_components(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => A),
        gen_ctors=[AbsCtor(), GlobalConstCtor()],
        gen_chooser=FocusedChooser(),
        gen_max_depth=4,
        gen_max_try=20,
        mutations=Mutations.AbstractMutation[],
    )
    cfg = default_config(
        population_size=10,
        generations=4,
        error_threshold=1e-12,
        init_mutation_rounds=0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
        tournament_size=3,
        log=(; enabled=true, every=1, topk=3, io=stdout),
        parallel=(; enabled=false),
        max_reduce_steps=40,
    )
    reg = Regressor(target_ty, comps, cfg, MersenneTwister(1))
    res = fit!(reg, data)
    @test res.status == :threshold
    @test res.best !== nothing
end

@testset "Regression integration C: 1D float array->float (traditional SR smoke)" begin
    k = 2
    Xty = TyArray(TyNat(k), TypeSystem.FloatTy)
    target_ty = TyFun(Xty, TypeSystem.FloatTy)

    GET1(xs) = xs[1]
    GET2(xs) = xs[2]

    global_consts = Dict{Symbol,Any}(
        :GET1 => GET1,
        :GET2 => GET2,
        :FADD => TypeSystem.FADD,
        :FMUL => TypeSystem.FMUL,
    )
    global_consts_ty = merge(
        get_type_ctx(Dict{Symbol,Any}(:FADD => TypeSystem.FADD, :FMUL => TypeSystem.FMUL)),
        Dict{Symbol,TyLike}(
            :GET1 => TyFun(Xty, TypeSystem.FloatTy),
            :GET2 => TyFun(Xty, TypeSystem.FloatTy),
        ),
    )

    rng_data = MersenneTwister(1111)
    X = [randn(rng_data, k) for _ in 1:4]
    Y = [GET1(x) * GET1(x) + GET2(x) for x in X]
    data = SupervisedDataset(X, Y)

    comps = merge(
        default_components(
            global_consts=global_consts,
            global_consts_ty=global_consts_ty,
            input_slots=Symbol[:x],
            input_types=Dict{Symbol,TyLike}(:x => Xty),
        ),
        (;
            gen_ctors=[AbsCtor(), IndexCtor(), GlobalConstCtor(), ApplySpineCtor(max_spine_args=3), FloatCtor(0.0, 1.0)],
            gen_chooser=FocusedChooser(max_spine_args=3),
            gen_max_depth=5,
            gen_max_try=15,
            mutations=Mutations.AbstractMutation[
                Mutations.PointMutation(),
                Mutations.ScalarConstMutation(σ=0.2),
            ],
            mutation_chooser=Mutations.WeightedPairChooser(cache_eligibility=true),
            mutation_provider=Mutations.IndexedProvider(),
            mutation_max_try=10,
            mutation_prune_consts=true,
            crossover_op=TypedSubtreeSwapCrossover(max_try=20, min_sites=1, allow_root=true),
        )
    )

    cfg = default_config(
        population_size=15,
        generations=4,
        error_threshold=1e-6,
        init_mutation_rounds=0,
        mutation_rate=0.7,
        crossover_rate=0.1,
        elitism=1,
        tournament_size=3,
        log=(; enabled=true, every=1, topk=3, io=stdout),
        parallel=(; enabled=false),
        max_reduce_steps=30,
    )

    reg = Regressor(target_ty, comps, cfg, MersenneTwister(1111))
    res = fit!(reg, data)
    @test res.best !== nothing
    @test length(res.history) > 0
end
