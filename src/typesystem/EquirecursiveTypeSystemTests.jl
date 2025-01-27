using TestItems

@testitem "TypeSystem closed μ equality correctness fixtures" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyArray, TyConst, TyFun, TyList, TyMu, TyNat

    fresh = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)

    function μtype(body)
        binder = fresh(0)
        return TyMu(binder, body(binder))
    end

    @test isdefined(TypeSystem, :isequal_mu)

    if isdefined(TypeSystem, :isequal_mu)
        isequal_mu = getfield(TypeSystem, :isequal_mu)

        int_cycle = μtype(x -> TyFun(int_ty, x))
        repeated_int_cycle = μtype(x -> TyFun(int_ty, TyFun(int_ty, x)))
        @test isequal_mu(int_cycle, repeated_int_cycle)

        self_application_cycle = μtype(f -> TyFun(TyFun(f, f), f))
        reassociated_self_application_cycle = μtype(f -> TyFun(f, TyFun(f, f)))
        @test isequal_mu(self_application_cycle, reassociated_self_application_cycle)

        nested_left = μtype(x -> TyFun(
            μtype(f -> TyFun(f, TyFun(f, f))),
            x,
        ))
        nested_right = μtype(x -> TyFun(
            μtype(f -> TyFun(f, f)),
            TyFun(μtype(f -> TyFun(f, f)), x),
        ))
        @test isequal_mu(nested_left, nested_right)

        alpha_left = μtype(x -> TyFun(int_ty, x))
        alpha_right = μtype(y -> TyFun(int_ty, y))
        @test isequal_mu(alpha_left, alpha_right)

        unfolded_cycle = μtype(x -> TyFun(int_ty, x))
        @test isequal_mu(unfolded_cycle, TyFun(int_ty, unfolded_cycle))

        list_cycle = μtype(x -> TyList(x))
        @test isequal_mu(list_cycle, TyList(list_cycle))

        period_two = μtype(x -> TyFun(int_ty, TyFun(bool_ty, x)))
        period_four = μtype(x -> TyFun(
            int_ty,
            TyFun(bool_ty, TyFun(int_ty, TyFun(bool_ty, x))),
        ))
        @test isequal_mu(period_two, period_four)

        captured_outer_left = μtype(x -> TyList(μtype(y -> TyFun(y, x))))
        captured_outer_right = μtype(a -> TyList(μtype(b -> TyFun(b, a))))
        @test isequal_mu(captured_outer_left, captured_outer_right)

        immediate_int = μtype(x -> TyFun(int_ty, x))
        immediate_bool = μtype(x -> TyFun(bool_ty, x))
        @test !isequal_mu(immediate_int, immediate_bool)

        shallow_cycle = μtype(x -> TyFun(int_ty, x))
        deeper_mismatch = μtype(x -> TyFun(int_ty, TyFun(bool_ty, x)))
        @test !isequal_mu(shallow_cycle, deeper_mismatch)

        int_bool_cycle = μtype(x -> TyFun(int_ty, TyFun(bool_ty, x)))
        bool_int_cycle = μtype(x -> TyFun(bool_ty, TyFun(int_ty, x)))
        @test !isequal_mu(int_bool_cycle, bool_int_cycle)

        rigid_binder = μtype(x -> TyFun(x, int_ty))
        @test !isequal_mu(rigid_binder, TyFun(bool_ty, int_ty))

        captured_outer = μtype(x -> TyList(μtype(y -> TyFun(y, x))))
        captured_inner = μtype(a -> TyList(μtype(b -> TyFun(b, b))))
        @test !isequal_mu(captured_outer, captured_inner)

        recursive_list = μtype(x -> TyList(x))
        recursive_function = μtype(x -> TyFun(int_ty, x))
        @test !isequal_mu(recursive_list, recursive_function)

        array_three = μtype(x -> TyArray(TyNat(3), x))
        array_four = μtype(x -> TyArray(TyNat(4), x))
        @test !isequal_mu(array_three, array_four)

        naked = μtype(x -> x)
        nested_naked = μtype(x -> μtype(_ -> x))
        @test_throws ArgumentError isequal_mu(naked, naked)
        @test_throws ArgumentError isequal_mu(nested_naked, nested_naked)
    end

    α = fresh(0)
    @test TypeSystem.unify!(α, α) === nothing
    @test α.instance === nothing
end

@testitem "TypeSystem transactional guarded unification" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyMu, TyProd

    fresh = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)

    recursive = fresh(0)
    @test TypeSystem.unify!(recursive, TyFun(int_ty, recursive)) === nothing
    @test recursive.instance isa TyMu
    explicit_binder = fresh(0)
    explicit = TyMu(explicit_binder, TyFun(int_ty, explicit_binder))
    @test TypeSystem.isequal_mu(recursive.instance, explicit)

    older = fresh(0)
    deeper = fresh(3)
    @test TypeSystem.unify!(older, TyFun(deeper, int_ty)) === nothing
    @test deeper.level == 0

    linked = fresh(0)
    lowered = fresh(4)
    trail = TypeSystem.TyTrail()
    failed = TypeSystem.unify!(
        TyProd(linked, int_ty),
        TyProd(TyFun(lowered, int_ty), bool_ty);
        trail,
    )
    @test failed !== nothing
    @test linked.instance === nothing
    @test lowered.level == 4
    @test isempty(trail.stack)

    # A failed predicate must not poison the next independent equation.
    @test TypeSystem.unify!(linked, int_ty) === nothing
    @test linked.instance === int_ty
end

@testitem "TypeSystem instance queries are transactional" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyScheme

    fresh = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)

    free = fresh(0)
    @test !TypeSystem.islesseq(free, int_ty)
    @test free.instance === nothing
    @test TypeSystem.islesseq(int_ty, free)
    @test free.instance === nothing

    left_var = fresh(0)
    right_a = fresh(0)
    right_b = fresh(0)
    narrower = TyScheme([left_var], TyFun(left_var, left_var))
    wider = TyScheme([right_a, right_b], TyFun(right_a, right_b))
    @test TypeSystem.islesseq(narrower, wider)
    @test !TypeSystem.islesseq(wider, narrower)
    @test all(v -> v.instance === nothing, (left_var, right_a, right_b))

    incompatible_left = TyFun(int_ty, int_ty)
    incompatible_right = TyFun(bool_ty, bool_ty)
    @test !TypeSystem.iscompatible(incompatible_left, incompatible_right)
    @test !TypeSystem.iscompatible(incompatible_right, incompatible_left)
end

@testitem "TypeSystem closed μ unification agrees with equality" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyMu

    fresh = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)

    left_binder = fresh(0)
    right_binder = fresh(0)
    left = TyMu(left_binder, TyFun(int_ty, left_binder))
    right = TyMu(right_binder, TyFun(int_ty, TyFun(int_ty, right_binder)))
    @test TypeSystem.isequal_mu(left, right)
    @test TypeSystem.unify!(left, right) === nothing
    @test left_binder.instance === nothing
    @test right_binder.instance === nothing

    mismatch_binder = fresh(0)
    mismatch = TyMu(mismatch_binder, TyFun(bool_ty, mismatch_binder))
    @test !TypeSystem.isequal_mu(left, mismatch)
    @test TypeSystem.unify!(left, mismatch) !== nothing
    @test left_binder.instance === nothing
    @test mismatch_binder.instance === nothing

    inferred_meta = fresh(0)
    @test TypeSystem.unify!(inferred_meta, TyFun(int_ty, inferred_meta)) === nothing
    inferred = TypeSystem.prune(inferred_meta)
    explicit_binder = fresh(0)
    explicit = TyMu(explicit_binder, TyFun(int_ty, explicit_binder))
    @test TypeSystem.isequal_mu(inferred, explicit)
    @test TypeSystem.unify!(inferred, explicit) === nothing
    @test explicit_binder.instance === nothing

    # Reusing an already folded recursive solution must follow stable graph
    # identities rather than repeatedly rebuilding and unfolding syntax.
    alias_meta = fresh(0)
    @test TypeSystem.unify!(alias_meta, inferred) === nothing
    @test TypeSystem.isequal_mu(TypeSystem.prune(alias_meta), explicit)

    malformed_binder = fresh(0)
    malformed = TyMu(malformed_binder, malformed_binder)
    @test TypeSystem.unify!(malformed, int_ty) !== nothing
    @test malformed_binder.instance === nothing
end

@testitem "TypeSystem rigid query atoms are not metas" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyConst

    @test isdefined(TypeSystem, :TyRigid)
    if isdefined(TypeSystem, :TyRigid)
        TyRigid = getfield(TypeSystem, :TyRigid)
        left = TyRigid(1)
        same = TyRigid(1)
        other = TyRigid(2)
        @test TypeSystem.unify!(left, same) === nothing
        @test TypeSystem.unify!(left, other) !== nothing
        @test TypeSystem.unify!(left, TyConst(:Int)) !== nothing
    end
end

@testitem "TypeSystem variable and scheme identities stay stable" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyProd, TyScheme

    first_factory = TypeSystem.make_tyvar_factory()
    second_factory = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)

    first = first_factory(0)
    second = second_factory(0)
    @test first !== second
    @test first != second
    @test TypeSystem.unify!(TyProd(first, second), TyProd(int_ty, int_ty)) === nothing
    @test TypeSystem.prune(first) == int_ty
    @test TypeSystem.prune(second) == int_ty

    deep = first_factory(4)
    remembered = Set([deep])
    old_hash = hash(deep)
    owner = first_factory(0)
    @test TypeSystem.unify!(owner, TyFun(deep, int_ty)) === nothing
    @test deep.level == 0
    @test hash(deep) == old_hash
    @test deep in remembered

    quantified = first_factory(0)
    scheme = TyScheme([quantified], TyFun(quantified, quantified))
    scheme_hash = hash(scheme)
    scheme_set = Set([scheme])
    for _ in 1:8
        TypeSystem.newtyvar(0)
    end
    @test hash(scheme) == scheme_hash
    @test scheme in scheme_set
end
