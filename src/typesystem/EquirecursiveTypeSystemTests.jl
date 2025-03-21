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

        @test isdefined(TypeSystem, :semantic_typekey)
        if isdefined(TypeSystem, :semantic_typekey)
            semantic_typekey = getfield(TypeSystem, :semantic_typekey)
            @test semantic_typekey(int_cycle) == semantic_typekey(repeated_int_cycle)
            @test semantic_typekey(period_two) == semantic_typekey(period_four)
            @test semantic_typekey(int_cycle) != semantic_typekey(immediate_bool)
            @test semantic_typekey(array_three) != semantic_typekey(array_four)
            scheme_left_var = fresh(0)
            scheme_right_var = fresh(0)
            rigid_collision = TypeSystem.TyScheme(
                [scheme_left_var],
                TyFun(scheme_left_var, TypeSystem.TyRigid(1)),
            )
            bound_reuse = TypeSystem.TyScheme(
                [scheme_right_var],
                TyFun(scheme_right_var, scheme_right_var),
            )
            @test semantic_typekey(rigid_collision) != semantic_typekey(bound_reuse)
            @test TypeSystem.typekey(int_cycle) != TypeSystem.typekey(repeated_int_cycle)
            @test TypeSystem.typectx_fingerprint(Dict(:x => int_cycle)) ==
                  TypeSystem.typectx_fingerprint(Dict(:x => repeated_int_cycle))
        end

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
    @test recursive.instance.var !== recursive
    @test recursive.instance.var.instance === nothing
    @test isempty(TypeSystem.collect_free_vars(recursive.instance))
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

    outer = fresh(0)
    inner = fresh(0)
    @test TypeSystem.unify!(outer, TyFun(int_ty, inner)) === nothing
    @test TypeSystem.unify!(inner, TyFun(bool_ty, outer)) === nothing
    multi_entry = fresh(0)
    @test TypeSystem.unify!(multi_entry, TyProd(outer, inner)) === nothing
    @test isempty(TypeSystem.collect_free_vars(TypeSystem.prune(multi_entry)))
end

@testitem "TypeSystem facade commit rolls back extension failures" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: Ty, TyVar

    struct ThrowingType <: Ty
        child::Ty
    end
    TypeSystem.type_constructor_tag(::ThrowingType) = :ThrowingType
    TypeSystem.type_children(t::ThrowingType) = (t.child,)
    TypeSystem.rebuild_type(::ThrowingType, ::Tuple{Ty}) = error("intentional rebuild failure")

    fresh = TypeSystem.make_tyvar_factory()
    owner = fresh(0)
    nested = fresh(4)
    @test_throws ErrorException TypeSystem.unify!(owner, ThrowingType(nested))
    @test owner.instance === nothing
    @test nested.instance === nothing
    @test nested.level == 4
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

    copied_quantified = deepcopy(quantified)
    @test copied_quantified === quantified
    copied_scheme = TyScheme([quantified], TyFun(copied_quantified, quantified))
    copied_instance = TypeSystem.instantiate(copied_scheme, 0, first_factory)
    @test copied_instance.arg === copied_instance.res

    bound_alias = first_factory(0)
    stale_alias = deepcopy(bound_alias)
    @test TypeSystem.unify!(bound_alias, int_ty) === nothing
    bool_ty = TyConst(:Bool)
    @test TypeSystem.unify!(
        TyProd(stale_alias, bound_alias),
        TyProd(bool_ty, bool_ty),
    ) !== nothing
    @test TypeSystem.prune(bound_alias) == int_ty
    @test TypeSystem.prune(stale_alias) == int_ty

    low_alias = first_factory(0)
    high_alias = deepcopy(low_alias)
    @test high_alias === low_alias
    high_alias.level = 4
    @test TypeSystem.unify!(
        TyProd(high_alias, low_alias),
        TyProd(int_ty, int_ty),
    ) === nothing
    @test low_alias.level == 4
    @test high_alias.level == 4

    merged_alias = first_factory(0)
    merged_copy = deepcopy(merged_alias)
    representative = first_factory(0)
    @test TypeSystem.unify!(
        TyProd(representative, representative),
        TyProd(merged_copy, merged_alias),
    ) === nothing
    @test TypeSystem.unify!(representative, int_ty) === nothing
    @test TypeSystem.prune(merged_alias) == int_ty
    @test TypeSystem.prune(merged_copy) == int_ty

    reverse_alias = first_factory(1)
    reverse_copy = deepcopy(reverse_alias)
    reverse_representative = first_factory(1)
    @test TypeSystem.unify!(
        TyProd(reverse_copy, reverse_alias),
        TyProd(reverse_representative, reverse_representative),
    ) === nothing
    @test TypeSystem.unify!(reverse_representative, int_ty) === nothing
    @test TypeSystem.prune(reverse_alias) == int_ty
    @test TypeSystem.prune(reverse_copy) == int_ty

    unmerged_alias = first_factory(1)
    unmerged_copy = deepcopy(unmerged_alias)
    @test unmerged_copy === unmerged_alias
    unmerged_pair = TyProd(unmerged_copy, unmerged_alias)
    @test TypeSystem.unify!(unmerged_pair, unmerged_pair) === nothing
    @test unmerged_alias.instance === nothing
    @test unmerged_copy.instance === nothing
    @test TypeSystem.collect_free_vars(unmerged_pair) == Set([unmerged_alias])

    copied_binder = first_factory(0)
    copied_occurrence = deepcopy(copied_binder)
    copied_mu = TypeSystem.TyMu(
        copied_binder,
        TyFun(int_ty, copied_occurrence),
    )
    alpha_binder = first_factory(0)
    alpha_mu = TypeSystem.TyMu(alpha_binder, TyFun(int_ty, alpha_binder))
    @test TypeSystem.isequal_mu(copied_mu, alpha_mu)
end

@testitem "Algorithm W uses let regions rather than lambda depth" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.Utils: Err
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyProd, TyScheme, TyLike

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    ctx = merge(
        TypeSystem.get_type_ctx(Dict{Symbol,Any}(
            :PAIR => LambdaRegression.Primitives.PAIR,
            :one => 1,
            :truth => true,
        )),
        Dict{Symbol,TyLike}(
            :ID => TyScheme([quantified], TyFun(quantified, quantified)),
        ),
    )

    # let y = ID (λz.z) in (y one, y truth)
    rhs = AppNode(ConstNode(:ID), AbsNode(:z, IndexNode(0)))
    left_use = AppNode(IndexNode(0), ConstNode(:one))
    right_use = AppNode(IndexNode(0), ConstNode(:truth))
    pair_body = AppNode(AppNode(ConstNode(:PAIR), left_use), right_use)
    let_term = AstTree(AppNode(AbsNode(:y, pair_body), rhs))
    inferred = TypeSystem.infer(let_term, ctx, fresh)
    @test inferred == TyProd(TypeSystem.IntTy, TypeSystem.BoolTy)

    # λx. let y = x in (y one, y truth) must not generalise the captured x.
    escaped_rhs = IndexNode(0)
    escaped_left = AppNode(IndexNode(0), ConstNode(:one))
    escaped_right = AppNode(IndexNode(0), ConstNode(:truth))
    escaped_pair = AppNode(AppNode(ConstNode(:PAIR), escaped_left), escaped_right)
    escaped_let = AppNode(AbsNode(:y, escaped_pair), escaped_rhs)
    escaped = TypeSystem.infer(AstTree(AbsNode(:x, escaped_let)), ctx, fresh)
    @test escaped isa Err

    self_application = AstTree(AbsNode(:x, AppNode(IndexNode(0), IndexNode(0))))
    recursive_scheme = TypeSystem.infer_scheme(self_application, ctx; factory=fresh)
    @test recursive_scheme isa TyScheme
    @test length(recursive_scheme.vars) == 1
    @test Set(recursive_scheme.vars) == TypeSystem.collect_free_vars(recursive_scheme.ty)
end

@testitem "TypeSystem constructor protocol composes with recursive kernels" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: Ty, TyFun, TyMu, TyScheme, IntTy, BoolTy

    struct BoxType <: Ty
        child::Ty
    end
    TypeSystem.type_constructor_tag(::BoxType) = :TestBox
    TypeSystem.type_children(box::BoxType) = (box.child,)
    TypeSystem.rebuild_type(::BoxType, children::Tuple{Ty}) = BoxType(children[1])

    @test TypeSystem.typekey(BoxType(IntTy)) != TypeSystem.typekey(BoxType(BoolTy))

    fresh = TypeSystem.make_tyvar_factory()
    left_binder = fresh(0)
    right_binder = fresh(0)
    left = TyMu(left_binder, BoxType(left_binder))
    right = BoxType(TyMu(right_binder, BoxType(right_binder)))
    @test TypeSystem.isequal_mu(left, right)

    recursive = fresh(0)
    @test TypeSystem.unify!(recursive, BoxType(recursive)) === nothing
    @test TypeSystem.isequal_mu(TypeSystem.prune(recursive), left)

    quantified = fresh(0)
    scheme = TyScheme([quantified], TyFun(quantified, BoxType(quantified)))
    instance = TypeSystem.instantiate(scheme, 0, fresh)
    @test instance.arg === instance.res.child
end

@testitem "Algorithm W accepts a minimal alternate term adapter" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: TyFun, TyLike

    abstract type MiniTerm end
    struct MiniConst <: MiniTerm
        name::Symbol
    end
    struct MiniVar <: MiniTerm
        index::Int
    end
    struct MiniAbs <: MiniTerm
        name::Symbol
        body::MiniTerm
    end
    struct MiniApp <: MiniTerm
        func::MiniTerm
        arg::MiniTerm
    end
    struct MiniAdapter end

    @test isdefined(TypeSystem, :term_view)
    @test isdefined(TypeSystem, :infer_term)
    if isdefined(TypeSystem, :term_view) && isdefined(TypeSystem, :infer_term)
        TypeSystem.term_view(::MiniAdapter, term::MiniConst) = (:const, term.name)
        TypeSystem.term_view(::MiniAdapter, term::MiniVar) = (:var, term.index)
        TypeSystem.term_view(::MiniAdapter, term::MiniAbs) = (:abs, term.name, term.body)
        TypeSystem.term_view(::MiniAdapter, term::MiniApp) = (:app, term.func, term.arg)

        ctx = TypeSystem.get_type_ctx(Dict{Symbol,Any}(:one => 1))
        identity = MiniAbs(:x, MiniVar(0))
        identity_ty = TypeSystem.infer_term(identity, Dict{Symbol,TyLike}(), MiniAdapter())
        @test identity_ty isa TyFun && identity_ty.arg === identity_ty.res
        identity_scheme = TypeSystem.infer_scheme_term(identity, Dict{Symbol,TyLike}(), MiniAdapter())
        @test length(identity_scheme.vars) == 1

        let_apply = MiniApp(
            MiniAbs(:id, MiniApp(MiniVar(0), MiniConst(:one))),
            identity,
        )
        @test TypeSystem.infer_term(let_apply, ctx, MiniAdapter()) == TypeSystem.IntTy
    end
end

@testitem "TypeSystem finite μ graph laws and stress" begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.TypeSystem: Ty, TyConst, TyFun, TyMu, TyProd, TyScheme

    struct CountedPair <: Ty
        left::Ty
        right::Ty
    end
    struct CountedUnary <: Ty
        child::Ty
    end
    const counted_rebuilds = Ref(0)
    const counted_visits = Ref(0)
    TypeSystem.type_constructor_tag(::CountedPair) = :CountedPair
    function TypeSystem.type_children(t::CountedPair)
        counted_visits[] += 1
        return (t.left, t.right)
    end
    TypeSystem.type_constructor_tag(::CountedUnary) = :CountedUnary
    TypeSystem.type_children(t::CountedUnary) = (t.child,)
    function TypeSystem.rebuild_type(::CountedPair, children::Tuple{Ty,Ty})
        counted_rebuilds[] += 1
        return CountedPair(children...)
    end
    TypeSystem.rebuild_type(::CountedUnary, children::Tuple{Ty}) = CountedUnary(children[1])

    fresh = TypeSystem.make_tyvar_factory()
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)

    function cycle(labels)
        binder = fresh(0)
        body = binder
        for label in Iterators.reverse(labels)
            body = TyFun(label, body)
        end
        return TyMu(binder, body)
    end

    period_one = cycle([int_ty])
    period_two = cycle([int_ty, int_ty])
    period_three = cycle([int_ty, int_ty, int_ty])
    @test TypeSystem.isequal_mu(period_one, period_one)
    @test TypeSystem.isequal_mu(period_one, period_two)
    @test TypeSystem.isequal_mu(period_two, period_one)
    @test TypeSystem.isequal_mu(period_two, period_three)
    @test TypeSystem.isequal_mu(period_one, period_three)
    @test TypeSystem.semantic_typekey(period_one) == TypeSystem.semantic_typekey(period_three)

    changed = cycle([int_ty, bool_ty])
    @test !TypeSystem.isequal_mu(period_one, changed)
    @test TypeSystem.semantic_typekey(period_one) != TypeSystem.semantic_typekey(changed)

    shared_child = TyFun(int_ty, bool_ty)
    shared = TyProd(shared_child, shared_child)
    duplicated = TyProd(TyFun(int_ty, bool_ty), TyFun(int_ty, bool_ty))
    @test TypeSystem.isequal_mu(shared, duplicated)
    @test TypeSystem.semantic_typekey(shared) == TypeSystem.semantic_typekey(duplicated)

    shared_dag = let node = int_ty
        for _ in 1:14
            node = TyProd(node, node)
        end
        node
    end
    equality_graph = TypeSystem._mu_eq_graph(shared_dag)
    @test length(equality_graph.nodes) <= 15
    unification_graph = TypeSystem._TypeGraph()
    TypeSystem._elaborate_graph!(unification_graph, shared_dag)
    @test length(unification_graph.nodes) <= 15

    non_sibling_dag = let node = int_ty, marker = TyFun(bool_ty, int_ty)
        for _ in 1:20
            node = TyProd(TyProd(node, marker), TyProd(marker, node))
        end
        node
    end
    @test TypeSystem.isequal_mu(non_sibling_dag, non_sibling_dag)
    equality_graph = TypeSystem._mu_eq_graph(non_sibling_dag)
    @test length(equality_graph.nodes) <= 160
    unification_graph = TypeSystem._TypeGraph()
    TypeSystem._elaborate_graph!(unification_graph, non_sibling_dag)
    @test length(unification_graph.nodes) <= 160

    quantified = fresh(0)
    counted_dag = let node = quantified
        for _ in 1:14
            node = CountedPair(node, node)
        end
        node
    end
    counted_rebuilds[] = 0
    TypeSystem.prune(counted_dag)
    @test counted_rebuilds[] <= 14
    counted_rebuilds[] = 0
    TypeSystem.instantiate(TyScheme([quantified], counted_dag), 0, fresh)
    @test counted_rebuilds[] <= 14
    counted_visits[] = 0
    TypeSystem.collect_free_vars(counted_dag)
    @test counted_visits[] <= 14
    counted_visits[] = 0
    counted_key = TypeSystem.typekey(counted_dag)
    @test counted_visits[] <= 14
    # The key stores a flat DAG table; hash work is bounded by these records,
    # not by recursively expanding the shared input tree.
    key_records = counted_key[4]
    @test length(key_records) <= 15
    @test all(
        record -> all(child -> 1 <= child <= length(key_records), record[3]),
        key_records,
    )
    counted_hash = hash(counted_key)
    counted_visits[] = 0
    repeated_key = TypeSystem.typekey(counted_dag)
    @test counted_visits[] <= 14
    @test repeated_key == counted_key
    @test hash(repeated_key) == counted_hash

    recursive_quantified = fresh(0)
    recursive_binder = fresh(0)
    recursive_scheme = TyScheme(
        [recursive_quantified],
        TyMu(recursive_binder, TyFun(recursive_quantified, recursive_binder)),
    )
    first_instance = TypeSystem.instantiate(recursive_scheme, 0, fresh)
    second_instance = TypeSystem.instantiate(recursive_scheme, 0, fresh)
    first_free = TypeSystem.collect_free_vars(first_instance)
    second_free = TypeSystem.collect_free_vars(second_instance)
    @test length(first_free) == 1
    @test length(second_free) == 1
    @test isempty(intersect(first_free, second_free))

    rigid_free = fresh(0)
    rigid_binder = fresh(0)
    rigid_recursive = TyMu(rigid_binder, TyFun(rigid_free, rigid_binder))
    concrete_binder = fresh(0)
    concrete_recursive = TyMu(concrete_binder, TyFun(int_ty, concrete_binder))
    @test !TypeSystem.islesseq(rigid_recursive, concrete_recursive)
    @test rigid_free.instance === nothing

    long_chain = let node = int_ty
        for _ in 1:1024
            node = TyFun(int_ty, node)
        end
        node
    end
    # A 1024-deep chain may add a repeated leaf record at each level, but must
    # still elaborate to a linear number of nodes.
    long_chain_graph = TypeSystem._mu_eq_graph(long_chain)
    @test length(long_chain_graph.nodes) <= 2 * 1024 + 1

    immutable_chain = let node = int_ty
        for _ in 1:1024
            node = CountedUnary(node)
        end
        node
    end
    immutable_chain_graph = TypeSystem._mu_eq_graph(immutable_chain)
    @test length(immutable_chain_graph.nodes) <= 2 * 1024 + 1

    long_cycle = cycle(fill(int_ty, 257))
    doubled_cycle = cycle(fill(int_ty, 514))
    @test TypeSystem.isequal_mu(long_cycle, doubled_cycle)
    @test TypeSystem.semantic_typekey(long_cycle) == TypeSystem.semantic_typekey(doubled_cycle)
    @test TypeSystem.unify!(long_cycle, doubled_cycle) === nothing
end
