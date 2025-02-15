using TestItems

@testitem "Procedural nodes expose ordered lexical child contexts" begin
    using Test
    using LambdaRegression: ASTs, Procedural, Trees
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode

    scope = Procedural.ScopeNode(
        AstNode[ConstNode(:a), IndexNode(0)],
        IndexNode(1),
    )
    empty_scope = Procedural.ScopeNode(AstNode[], ConstNode(:body))
    function_def = Procedural.FunctionDefNode(IndexNode(0))

    @test Trees.children(scope) == AstNode[ConstNode(:a), IndexNode(0), IndexNode(1)]
    @test Trees.arity(scope) == 3
    @test Trees.children(empty_scope) == AstNode[ConstNode(:body)]
    @test Procedural.child_role(scope, 1) == (kind=:binding, ordinal=1)
    @test Procedural.child_role(scope, 3) == (kind=:body, ordinal=nothing)
    @test Procedural.child_role(function_def, 1) == (kind=:function_body, ordinal=nothing)
    @test_throws BoundsError Procedural.child_role(scope, 4)

    outer = (Procedural.BinderId(:function_def, (9,), 1),)
    @test Procedural.visible_binders(scope, 1; path=(2,), outer) == outer
    @test Procedural.visible_binders(scope, 2; path=(2,), outer) == (
        Procedural.BinderId(:scope, (2,), 1),
        outer...,
    )
    @test Procedural.visible_binders(scope, 3; path=(2,), outer) == (
        Procedural.BinderId(:scope, (2,), 2),
        Procedural.BinderId(:scope, (2,), 1),
        outer...,
    )
    @test Procedural.visible_binders(function_def, 1; path=(2, 3), outer) == (
        Procedural.BinderId(:function_def, (2, 3), 1),
        outer...,
    )

    # Public procedural names alias the neutral protocol used by core lambdas.
    @test Procedural.BinderId === ASTs.LexicalContexts.BinderId
    abs_outer = ASTs.LexicalContexts.visible_binders(
        AbsNode(:x, function_def),
        1;
        path=(4,),
        outer,
    )
    self_context = Procedural.visible_binders(
        function_def,
        1;
        path=(4, 1),
        outer=abs_outer,
    )
    mixed_scope = Procedural.ScopeNode(
        AstNode[ConstNode(:a), IndexNode(0)],
        IndexNode(1),
    )
    @test Procedural.visible_binders(
        mixed_scope,
        3;
        path=(4, 1, 1),
        outer=self_context,
    ) == (
        Procedural.BinderId(:scope, (4, 1, 1), 2),
        Procedural.BinderId(:scope, (4, 1, 1), 1),
        Procedural.BinderId(:function_def, (4, 1), 1),
        Procedural.BinderId(:abs, (4,), 1),
        outer...,
    )

    Trees.setchild!(scope, 2, ConstNode(:replacement))
    @test Trees.children(scope)[2] == ConstNode(:replacement)
    @test_throws BoundsError Trees.setchild!(scope, 4, ConstNode(:bad))

    same_scope = Procedural.ScopeNode(
        AstNode[ConstNode(:a), ConstNode(:replacement)],
        IndexNode(1),
    )
    @test scope == same_scope
    @test hash(scope) == hash(same_scope)
    @test sprint(show, MIME("text/plain"), scope) ==
        "{ x = a; y = replacement; #1 }"
    @test sprint(show, MIME("text/plain"), function_def) == "fn f. #0"

    recursive = Procedural.FunctionDefNode(
        AbsNode(:ignored, AppNode(IndexNode(1), IndexNode(0))),
    )
    nested = Procedural.FunctionDefNode(
        AbsNode(:ignored, Procedural.FunctionDefNode(
            AbsNode(:ignored, AppNode(IndexNode(1), IndexNode(0))),
        )),
    )
    dependent_scope = Procedural.ScopeNode(
        AstNode[ConstNode(:a), IndexNode(0)],
        AppNode(IndexNode(0), IndexNode(1)),
    )

    @test sprint(show, MIME("text/plain"), recursive) == "fn f(x). #1 #0"
    @test sprint(show, MIME("text/plain"), nested) ==
        "fn f(x). fn g(y). #1 #0"
    @test sprint(show, MIME("text/plain"), dependent_scope) ==
        "{ x = a; y = #0; #0 #1 }"
    @test sprint(
        show,
        MIME("text/plain"),
        AppNode(recursive, dependent_scope),
    ) == "(fn f(x). #1 #0) ({ x = a; y = #0; #0 #1 })"
end

@testitem "Procedural lowering preserves exact De Bruijn contexts" begin
    using Test
    using LambdaRegression: ASTs, Procedural
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode

    scope = Procedural.ScopeNode(
        AstNode[ConstNode(:first), IndexNode(0)],
        IndexNode(1),
    )
    expected = AppNode(
        AbsNode(
            Symbol("0"),
            AppNode(AbsNode(Symbol("1"), IndexNode(1)), IndexNode(0)),
        ),
        ConstNode(:first),
    )

    @test Procedural.lower_procedural(scope) == expected
    @test Procedural.lower_procedural(
        Procedural.ScopeNode(AstNode[], IndexNode(3)),
    ) == IndexNode(3)

    recursive_identity = Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    lowered = Procedural.lower_procedural(recursive_identity)
    @test lowered isa AppNode
    @test lowered.func == Procedural.fixed_point_witness()
    @test lowered.arg == AbsNode(
        :self,
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )

    nested = Procedural.FunctionDefNode(
        Procedural.FunctionDefNode(IndexNode(1)),
    )
    nested_lowered = Procedural.lower_procedural(nested)
    @test nested_lowered isa AppNode
    @test nested_lowered.func == Procedural.fixed_point_witness()
    @test nested_lowered.arg.body == IndexNode(0)

    core = AstTree(AbsNode(:x, IndexNode(0)))
    @test Procedural.lower_procedural(core) == core
    @test Procedural.lower_procedural(core) !== core
end

@testitem "Procedural lowering omits Y only for an unused self binder" begin
    using Test
    using LambdaRegression: Procedural
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode

    self_free = Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    lowered_self_free = Procedural.lower_procedural(
        self_free;
        fixed_point=() -> error("unused self must not request a fixed-point witness"),
    )
    @test lowered_self_free == AbsNode(:x, IndexNode(0))
    @test self_free isa Procedural.FunctionDefNode
    @test self_free.body == AbsNode(:x, IndexNode(0))

    outer_capture = AbsNode(
        :outer,
        Procedural.FunctionDefNode(AbsNode(:x, IndexNode(2))),
    )
    @test Procedural.lower_procedural(
        outer_capture;
        fixed_point=() -> error("unused self must not request a fixed-point witness"),
    ) == AbsNode(:outer, AbsNode(:x, IndexNode(1)))

    returned_self = Procedural.FunctionDefNode(AbsNode(:x, IndexNode(1)))
    lowered_returned_self = Procedural.lower_procedural(
        returned_self;
        fixed_point=ConstNode(:Y),
    )
    @test lowered_returned_self == AppNode(
        ConstNode(:Y),
        AbsNode(:self, AbsNode(:x, IndexNode(1))),
    )

    direct_knot_ty = Procedural.infer_procedural(
        Procedural.FunctionDefNode(IndexNode(0)),
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}(),
    )
    @test direct_knot_ty isa LambdaRegression.TypeSystem.TyFun
    direct_knot_ty = LambdaRegression.TypeSystem.prune(direct_knot_ty)
    @test direct_knot_ty.arg isa LambdaRegression.TypeSystem.TyVar
    @test direct_knot_ty.res isa LambdaRegression.TypeSystem.TyVar
    @test direct_knot_ty.arg !== direct_knot_ty.res
    @test LambdaRegression.TypeSystem.typekey(direct_knot_ty) !=
        LambdaRegression.TypeSystem.typekey(
            LambdaRegression.TypeSystem.TyFun(direct_knot_ty.arg, direct_knot_ty.arg),
        )

    scoped_knot = Procedural.ScopeNode(
        AstNode[Procedural.FunctionDefNode(IndexNode(0))],
        IndexNode(0),
    )
    scoped_knot_ty = Procedural.infer_procedural(
        scoped_knot,
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}(),
    )
    @test scoped_knot_ty isa LambdaRegression.TypeSystem.TyFun
    @test !LambdaRegression.TypeSystem.iscompatible(
        scoped_knot_ty,
        LambdaRegression.TypeSystem.IntTy,
    )
    custom_meta = IdDict{AstNode,Tuple{
        LambdaRegression.TypeSystem.TyLike,
        LambdaRegression.TypeSystem.TyLike,
    }}()
    custom_scoped_knot_ty = Procedural.infer_procedural(
        scoped_knot,
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}();
        fixed_point=() -> Procedural.fixed_point_witness(),
        meta=custom_meta,
    )
    @test custom_scoped_knot_ty isa LambdaRegression.TypeSystem.TyFun
    @test !LambdaRegression.TypeSystem.iscompatible(
        custom_scoped_knot_ty,
        LambdaRegression.TypeSystem.IntTy,
    )
    custom_upper = custom_meta[scoped_knot][2]
    custom_first = LambdaRegression.TypeSystem.instantiate(
        custom_upper,
        0,
        LambdaRegression.TypeSystem.make_tyvar_factory(),
    )
    custom_second = LambdaRegression.TypeSystem.instantiate(
        custom_upper,
        0,
        LambdaRegression.TypeSystem.make_tyvar_factory(),
    )
    @test custom_first.arg !== custom_second.arg
    @test custom_first.res !== custom_second.res

    meta = IdDict{AstNode,Tuple{
        LambdaRegression.TypeSystem.TyLike,
        LambdaRegression.TypeSystem.TyLike,
    }}()
    knot = Procedural.FunctionDefNode(IndexNode(0))
    @test Procedural.infer_procedural(
        knot,
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}();
        meta,
    ) isa LambdaRegression.TypeSystem.TyFun
    upper = meta[knot][2]
    first = LambdaRegression.TypeSystem.instantiate(
        upper,
        0,
        LambdaRegression.TypeSystem.make_tyvar_factory(),
    )
    second = LambdaRegression.TypeSystem.instantiate(
        upper,
        0,
        LambdaRegression.TypeSystem.make_tyvar_factory(),
    )
    @test first isa LambdaRegression.TypeSystem.TyFun
    @test second isa LambdaRegression.TypeSystem.TyFun
    @test first.arg !== second.arg
    @test first.res !== second.res
end

@testitem "Procedural lowering metadata preserves equal source identities" begin
    using Test
    using LambdaRegression: Procedural
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstNode, IndexNode

    context = Procedural._LoweringContext(
        nothing,
        Set{Symbol}(),
        0,
        AstNode[],
        IdDict{AstNode,Vector{AstNode}}(),
    )
    lowered = AppNode(IndexNode(0), IndexNode(1))
    first_source = AppNode(IndexNode(0), IndexNode(1))
    second_source = AppNode(IndexNode(0), IndexNode(1))
    Procedural._record_source!(context, lowered, first_source)
    Procedural._record_source!(context, lowered, second_source)

    sources = context.sources_by_lowered[lowered]
    @test first_source == second_source
    @test first_source !== second_source
    @test length(sources) == 2
    @test sources[1] === first_source
    @test sources[2] === second_source
end

@testitem "Procedural adapters preserve HM generalisation and finite reduction" begin
    using Test
    using LambdaRegression: Procedural, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike, TyProd

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    ctx = merge(
        TypeSystem.get_type_ctx(Dict{Symbol,Any}(
            :PAIR => TypeSystem.PAIR,
            :one => 1,
            :truth => true,
        )),
        Dict{Symbol,TyLike}(
            :ID => TypeSystem.TyScheme(
                [quantified],
                TyFun(quantified, quantified),
            ),
        ),
    )

    rhs = AppNode(ConstNode(:ID), AbsNode(:z, IndexNode(0)))
    left_use = AppNode(IndexNode(0), ConstNode(:one))
    right_use = AppNode(IndexNode(0), ConstNode(:truth))
    pair_body = AppNode(AppNode(ConstNode(:PAIR), left_use), right_use)
    polymorphic_scope = Procedural.ScopeNode(AstNode[rhs], pair_body)

    @test Procedural.infer_procedural(
        AstTree(polymorphic_scope),
        ctx;
        factory=fresh,
    ) == TyProd(IntTy, BoolTy)

    recursive_function = Procedural.FunctionDefNode(
        AbsNode(:x, IndexNode(0)),
    )
    recursive_ty = Procedural.infer_procedural(
        AstTree(recursive_function),
        Dict{Symbol,TyLike}(),
    )
    @test TypeSystem.iscompatible(recursive_ty, TyFun(IntTy, IntTy))

    evaluation_types = Dict{Symbol,TyLike}(:answer => IntTy)
    application = AstTree(AppNode(recursive_function, ConstNode(:answer)))
    @test Procedural.infer_procedural(application, evaluation_types) === IntTy
    @test Procedural.reduce_procedural!(application, 128)
    @test application.root == ConstNode(:answer)
    @test TypeSystem.infer(application, evaluation_types) === IntTy
end

@testitem "Native procedural inference matches lowering across mixed binders" begin
    using Test
    using LambdaRegression: Procedural, TypeSystem, Utils
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyLike

    ctx = Dict{Symbol,TyLike}(:one => IntTy)
    # Inside the lambda: #0 is x, #1 is self, and #2 is the outer Scope identity.
    recursive = Procedural.FunctionDefNode(
        AbsNode(
            :x,
            AppNode(
                IndexNode(2),
                AppNode(IndexNode(1), IndexNode(0)),
            ),
        ),
    )
    scope = Procedural.ScopeNode(
        AstNode[AbsNode(:z, IndexNode(0))],
        recursive,
    )
    term = AppNode(scope, ConstNode(:one))
    source_snapshot = deepcopy(term)
    native_meta = IdDict{AstNode,Tuple{TyLike,TyLike}}()

    native = Procedural.infer_procedural(term, ctx; meta=native_meta)
    lowered = Procedural._infer_procedural_lowered(
        term,
        ctx,
        TypeSystem.make_tyvar_factory(),
        nothing,
        Procedural.fixed_point_witness,
    )

    @test !(native isa Utils.Err)
    @test !(lowered isa Utils.Err)
    @test TypeSystem.iscompatible(native, lowered)
    @test TypeSystem.iscompatible(lowered, native)
    @test TypeSystem.iscompatible(native, IntTy)
    @test term == source_snapshot
    @test term !== source_snapshot
    @test haskey(native_meta, scope)
    @test haskey(native_meta, recursive)
    @test haskey(native_meta, recursive.body)
end

@testitem "Procedural lowering keeps anonymous binders capture free" begin
    using Test
    using LambdaRegression: Procedural, TypeSystem, Utils
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode, LetNode, VarNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    ctx = TypeSystem.get_type_ctx(Dict{Symbol,Any}(:one => 1))

    scope_collision = AbsNode(
        Symbol("0"),
        Procedural.ScopeNode(
            AstNode[ConstNode(:one)],
            VarNode(Symbol("0")),
        ),
    )
    scope_ty = Procedural.infer_procedural(scope_collision, ctx)
    @test scope_ty isa TyFun
    @test scope_ty.arg == scope_ty.res
    lowered_scope = Procedural.lower_procedural(scope_collision)
    @test lowered_scope.body.func.name != Symbol("0")

    function_collision = AbsNode(
        :self,
        Procedural.FunctionDefNode(
            AbsNode(:x, AppNode(IndexNode(1), VarNode(:self))),
        ),
    )
    function_ty = Procedural.infer_procedural(function_collision, ctx)
    @test function_ty isa TyFun
    @test function_ty.res isa TyFun
    lowered_function = Procedural.lower_procedural(function_collision)
    @test lowered_function.body.arg.name != :self
    @test lowered_function.body.arg.body.body.arg == VarNode(:self)

    legacy_let = LetNode(:x, ConstNode(:one), IndexNode(0))
    @test Procedural.lower_procedural(legacy_let) isa LambdaRegression.ASTs.NodeASTs.AppNode
    @test Procedural.infer_procedural(legacy_let, ctx) === IntTy

    non_function = Procedural.infer_procedural(
        Procedural.FunctionDefNode(ConstNode(:one)),
        ctx,
    )
    @test non_function isa Utils.Err
    @test non_function.value isa Procedural.NonFunctionDefinitionError

    nested_non_function = Procedural.infer_procedural(
        Procedural.FunctionDefNode(
            AbsNode(
                :x,
                Procedural.ScopeNode(
                    AstNode[Procedural.FunctionDefNode(ConstNode(:one))],
                    IndexNode(1),
                ),
            ),
        ),
        ctx,
    )
    @test nested_non_function isa Utils.Err
    @test nested_non_function.value isa Procedural.NonFunctionDefinitionError

    recursive_head = Procedural.infer_procedural(
        Procedural.FunctionDefNode(
            LambdaRegression.ASTs.NodeASTs.AppNode(IndexNode(0), IndexNode(0)),
        ),
        Dict{Symbol,TyLike}(),
    )
    @test !(recursive_head isa Utils.Err)
end

@testitem "Procedural fixed-point witness has the designated closed scheme" begin
    using Test
    using LambdaRegression: Procedural, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.TypeSystem: TyFun, TyLike, TyScheme

    scheme = TypeSystem.infer_scheme(
        AstTree(Procedural.fixed_point_witness()),
        Dict{Symbol,TyLike}(),
    )
    variable = TypeSystem.make_tyvar_factory()(0)
    expected = TyScheme(
        [variable],
        TyFun(TyFun(variable, variable), variable),
    )

    @test scheme isa TyScheme
    @test TypeSystem.islesseq(scheme, expected)
    @test TypeSystem.islesseq(expected, scheme)
end

@testitem "Procedural adapters expose source metadata and callable witnesses" begin
    using Test
    using LambdaRegression: Procedural
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, IndexNode
    using LambdaRegression.TypeSystem: TyLike

    struct CallableFixedPoint end
    (::CallableFixedPoint)() = Procedural.fixed_point_witness()

    source = Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    metadata = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    inferred = Procedural.infer_procedural(
        source,
        Dict{Symbol,TyLike}();
        meta=metadata,
        fixed_point=CallableFixedPoint(),
    )

    @test inferred isa LambdaRegression.TypeSystem.TyFun
    @test haskey(metadata, source)
    @test haskey(metadata, source.body)
    @test all(node === source || node === source.body || node === source.body.body for node in keys(metadata))

    captured_index = IndexNode(2)
    captured = AbsNode(
        :outer,
        Procedural.FunctionDefNode(AbsNode(:x, captured_index)),
    )
    captured_metadata = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    @test Procedural.infer_procedural(
        captured,
        Dict{Symbol,TyLike}();
        meta=captured_metadata,
    ) isa LambdaRegression.TypeSystem.TyFun
    @test haskey(captured_metadata, captured_index)

    lowered = Procedural.lower_procedural(source; fixed_point=CallableFixedPoint())
    @test lowered == AbsNode(:x, IndexNode(0))

    recursive_source = Procedural.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    witness = Procedural.fixed_point_witness()
    from_node = Procedural.lower_procedural(recursive_source; fixed_point=witness)
    from_tree = Procedural.lower_procedural(recursive_source; fixed_point=AstTree(witness))
    @test from_node.func == witness
    @test from_tree.func == witness
    @test from_node.func !== witness
    @test from_tree.func !== witness
    @test from_node.func !== from_tree.func
end

@testitem "Procedural FunctionDef runs finite recursive fixtures" begin
    using Test
    using LambdaRegression: ASTs, Procedural, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyFun, TyLike, TyList

    app(function_node, arguments...) = foldl(AppNode, arguments; init=function_node)

    integer_values = Dict{Symbol,Any}(
        :COND => TypeSystem.COND,
        :IEQ => TypeSystem.IEQ,
        :IADD => TypeSystem.IADD,
        :ISUB => TypeSystem.ISUB,
        :IMUL => TypeSystem.IMUL,
        :int0 => 0,
        :int1 => 1,
        :int2 => 2,
        :int4 => 4,
        :int6 => 6,
    )
    integer_types = TypeSystem.get_type_ctx(integer_values)
    # The reducer evaluates host calls only after their arguments are constants.
    # Use the existing Church selector at runtime so an unselected recursive
    # branch is not evaluated; its type is still the ordinary polymorphic COND.
    integer_values[:COND] = ASTs.Contexts.COND

    n = IndexNode(0)
    self = IndexNode(1)
    n_minus_1 = app(ConstNode(:ISUB), n, ConstNode(:int1))
    factorial_body = AbsNode(
        :n,
        app(
            ConstNode(:COND),
            app(ConstNode(:IEQ), n, ConstNode(:int0)),
            ConstNode(:int1),
            app(ConstNode(:IMUL), n, AppNode(self, n_minus_1)),
        ),
    )
    factorial = Procedural.FunctionDefNode(factorial_body)
    @test TypeSystem.iscompatible(
        Procedural.infer_procedural(AstTree(factorial), integer_types),
        TyFun(IntTy, IntTy),
    )
    factorial_4 = AstTree(AppNode(factorial, ConstNode(:int4)))
    @test Procedural.reduce_procedural!(factorial_4, 10_000, integer_values)
    @test integer_values[factorial_4.root.name] == 24
    prepared_factorial = Procedural.prepare_procedural(factorial)
    @test Procedural.run_procedural(
        prepared_factorial,
        integer_values;
        args=(4,),
        max_steps=10_000,
    ) == 24

    n_minus_2 = app(ConstNode(:ISUB), n, ConstNode(:int2))
    fibonacci_body = AbsNode(
        :n,
        app(
            ConstNode(:COND),
            app(ConstNode(:IEQ), n, ConstNode(:int0)),
            ConstNode(:int0),
            app(
                ConstNode(:COND),
                app(ConstNode(:IEQ), n, ConstNode(:int1)),
                ConstNode(:int1),
                app(
                    ConstNode(:IADD),
                    AppNode(self, n_minus_1),
                    AppNode(self, n_minus_2),
                ),
            ),
        ),
    )
    fibonacci = Procedural.FunctionDefNode(fibonacci_body)
    @test TypeSystem.iscompatible(
        Procedural.infer_procedural(AstTree(fibonacci), integer_types),
        TyFun(IntTy, IntTy),
    )
    fibonacci_6 = AstTree(AppNode(fibonacci, ConstNode(:int6)))
    @test Procedural.reduce_procedural!(fibonacci_6, 20_000, integer_values)
    @test integer_values[fibonacci_6.root.name] == 8

    list_values = Dict{Symbol,Any}(
        :UNCONS_CASE => TypeSystem.UNCONS_CASE,
        :FADD => TypeSystem.FADD,
        :float0 => 0.0,
        :values => [1.0, 2.0, 3.0],
        :tail2 => [2.0, 3.0],
        :tail1 => [3.0],
        :tail0 => Float64[],
        :head3 => 1.0,
        :head2 => 2.0,
        :head1 => 3.0,
    )
    list_types = TypeSystem.get_type_ctx(
        list_values,
        Dict{Symbol,TyLike}(
            :values => TyList(FloatTy),
            :tail2 => TyList(FloatTy),
            :tail1 => TyList(FloatTy),
            :tail0 => TyList(FloatTy),
        ),
    )
    function finite_uncons(xs)
        isempty(xs) && return AbsNode(:ok, AbsNode(:fallback, IndexNode(0)))
        length(xs) <= 3 || throw(ArgumentError("fixture list is outside its finite table"))
        success = app(
            IndexNode(1),
            ConstNode(Symbol(:head, length(xs))),
            ConstNode(Symbol(:tail, length(xs) - 1)),
        )
        return AbsNode(:ok, AbsNode(:fallback, success))
    end
    list_values[:UNCONS_CASE] = finite_uncons

    list = IndexNode(0)
    # In the success body: tail=#0, head=#1, list=#2, recursive self=#3.
    success = AbsNode(
        :head,
        AbsNode(
            :tail,
            app(
                ConstNode(:FADD),
                IndexNode(1),
                AppNode(IndexNode(3), IndexNode(0)),
            ),
        ),
    )
    list_sum_body = AbsNode(
        :list,
        app(
            ConstNode(:UNCONS_CASE),
            list,
            success,
            ConstNode(:float0),
        ),
    )
    list_sum = Procedural.FunctionDefNode(list_sum_body)
    @test TypeSystem.iscompatible(
        Procedural.infer_procedural(AstTree(list_sum), list_types),
        TyFun(TyList(FloatTy), FloatTy),
    )
    sum_values = AstTree(AppNode(list_sum, ConstNode(:values)))
    @test Procedural.reduce_procedural!(sum_values, 10_000, list_values)
    @test list_values[sum_values.root.name] == 6.0
end

@testitem "Native procedural evaluation shares a finite recursion budget" begin
    using Test
    using LambdaRegression: Procedural, Utils
    using LambdaRegression.ASTs.NodeASTs: AbsNode, IndexNode, VarNode

    identity = Procedural.prepare_procedural(
        Procedural.FunctionDefNode(AbsNode(:x, IndexNode(0))),
    )
    @test Procedural.run_procedural(identity, Dict{Symbol,Any}(); args=(42,)) == 42

    named_identity = Procedural.prepare_procedural(
        Procedural.FunctionDefNode(AbsNode(:x, VarNode(:x))),
    )
    @test Procedural.run_procedural(
        named_identity,
        Dict{Symbol,Any}();
        args=(7,),
    ) == 7

    looping = Procedural.prepare_procedural(
        Procedural.FunctionDefNode(IndexNode(0)),
    )
    stopped = Procedural.run_procedural(
        looping,
        Dict{Symbol,Any}();
        args=(42,),
        max_steps=32,
    )
    @test stopped isa Utils.Err
    @test stopped.value isa Procedural.ProceduralStepLimitError
end
