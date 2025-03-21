using TestItems

@testitem "FunctionDef lowering preserves exact De Bruijn contexts" begin
    using Test
    using LambdaRegression: Transforms
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, IndexNode

    recursive_identity = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    lowered = Transforms.lower_functiondefs(recursive_identity)
    @test lowered isa AppNode
    @test lowered.func == Transforms.fixed_point_witness()
    @test lowered.arg == AbsNode(
        :self,
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )

    nested = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(1)),
    )
    nested_lowered = Transforms.lower_functiondefs(nested)
    @test nested_lowered isa AppNode
    @test nested_lowered.func == Transforms.fixed_point_witness()
    @test nested_lowered.arg.body == IndexNode(0)

    core = AstTree(AbsNode(:x, IndexNode(0)))
    @test Transforms.lower_functiondefs(core) == core
    @test Transforms.lower_functiondefs(core) !== core
end

@testitem "FunctionDef lowering omits Y only for an unused self binder" begin
    using Test
    using LambdaRegression: Transforms
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode

    self_free = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    lowered_self_free = Transforms.lower_functiondefs(
        self_free;
        fixed_point=() -> error("unused self must not request a fixed-point witness"),
    )
    @test lowered_self_free == AbsNode(:x, IndexNode(0))
    @test self_free isa LambdaRegression.ASTs.NodeASTs.FunctionDefNode
    @test self_free.body == AbsNode(:x, IndexNode(0))

    outer_capture = AbsNode(
        :outer,
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(2))),
    )
    @test Transforms.lower_functiondefs(
        outer_capture;
        fixed_point=() -> error("unused self must not request a fixed-point witness"),
    ) == AbsNode(:outer, AbsNode(:x, IndexNode(1)))

    returned_self = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(1)))
    lowered_returned_self = Transforms.lower_functiondefs(
        returned_self;
        fixed_point=ConstNode(:Y),
    )
    @test lowered_returned_self == AppNode(
        ConstNode(:Y),
        AbsNode(:self, AbsNode(:x, IndexNode(1))),
    )

    direct_knot_ty = LambdaRegression.TypeSystem.infer_term(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0)),
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

    let_knot = AppNode(
        AbsNode(:definition, IndexNode(0)),
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0)),
    )
    let_knot_ty = LambdaRegression.TypeSystem.infer_term(
        let_knot,
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}(),
    )
    @test let_knot_ty isa LambdaRegression.TypeSystem.TyFun
    @test !LambdaRegression.TypeSystem.iscompatible(
        let_knot_ty,
        LambdaRegression.TypeSystem.IntTy,
    )
    custom_meta = IdDict{AstNode,Tuple{
        LambdaRegression.TypeSystem.TyLike,
        LambdaRegression.TypeSystem.TyLike,
    }}()
    custom_let_knot_ty = LambdaRegression.TypeSystem.infer_term(
        let_knot,
        Dict{Symbol,LambdaRegression.TypeSystem.TyLike}();
        meta=custom_meta,
    )
    @test custom_let_knot_ty isa LambdaRegression.TypeSystem.TyFun
    @test !LambdaRegression.TypeSystem.iscompatible(
        custom_let_knot_ty,
        LambdaRegression.TypeSystem.IntTy,
    )
    custom_upper = custom_meta[let_knot][2]
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
    knot = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0))
    @test LambdaRegression.TypeSystem.infer_term(
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

@testitem "TypeSystem inference preserves HM generalisation and finite reduction" begin
    using Test
    using LambdaRegression: Execution, Transforms, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike, TyProd

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    ctx = merge(
        TypeSystem.get_type_ctx(Dict{Symbol,Any}(
            :PAIR => LambdaRegression.Primitives.PAIR,
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
    polymorphic_let = AppNode(AbsNode(:poly_id, pair_body), rhs)

    @test LambdaRegression.TypeSystem.infer_term(
        AstTree(polymorphic_let),
        ctx;
        factory=fresh,
    ) == TyProd(IntTy, BoolTy)

    second_rhs = AppNode(IndexNode(0), ConstNode(:one))
    nested_body = AppNode(
        AppNode(ConstNode(:PAIR), IndexNode(0)),
        AppNode(IndexNode(1), ConstNode(:truth)),
    )
    nested_let = AppNode(
        AbsNode(:first, AppNode(AbsNode(:second, nested_body), second_rhs)),
        rhs,
    )
    @test LambdaRegression.TypeSystem.infer_term(
        AstTree(nested_let),
        ctx;
        factory=TypeSystem.make_tyvar_factory(),
    ) == TyProd(IntTy, BoolTy)

    recursive_function = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, IndexNode(0)),
    )
    recursive_ty = LambdaRegression.TypeSystem.infer_term(
        AstTree(recursive_function),
        Dict{Symbol,TyLike}(),
    )
    @test TypeSystem.iscompatible(recursive_ty, TyFun(IntTy, IntTy))

    evaluation_types = Dict{Symbol,TyLike}(:answer => IntTy)
    application = AstTree(AppNode(recursive_function, ConstNode(:answer)))
    @test LambdaRegression.TypeSystem.infer_term(application, evaluation_types) === IntTy
    application.root = Transforms.lower_functiondefs(application.root)
    @test Execution.beta_reduce!(application, 128, evaluation_types)
    @test application.root == ConstNode(:answer)
    @test TypeSystem.infer(application, evaluation_types) === IntTy
end

@testitem "Native TypeSystem inference matches lowering across mixed binders" begin
    using Test
    using LambdaRegression: Transforms, TypeSystem, Utils
    using LambdaRegression.ASTs.NodeASTs: AstNode, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyLike

    ctx = Dict{Symbol,TyLike}(:one => IntTy)
    # Inside the lambda: #0 is x, #1 is self, and #2 is the outer let-like identity.
    recursive = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(
            :x,
            AppNode(
                IndexNode(2),
                AppNode(IndexNode(1), IndexNode(0)),
            ),
        ),
    )
    let_application = AppNode(
        AbsNode(:identity, recursive),
        AbsNode(:z, IndexNode(0)),
    )
    term = AppNode(let_application, ConstNode(:one))
    source_snapshot = deepcopy(term)
    native_meta = IdDict{AstNode,Tuple{TyLike,TyLike}}()

    native = TypeSystem.infer_term(term, ctx; meta=native_meta)
    lowered = TypeSystem.infer_term(
        Transforms.lower_functiondefs(term),
        ctx;
        factory=TypeSystem.make_tyvar_factory(),
    )

    @test !(native isa Utils.Err)
    @test !(lowered isa Utils.Err)
    @test TypeSystem.iscompatible(native, lowered)
    @test TypeSystem.iscompatible(lowered, native)
    @test TypeSystem.iscompatible(native, IntTy)
    @test term == source_snapshot
    @test term !== source_snapshot
    @test haskey(native_meta, let_application)
    @test haskey(native_meta, recursive)
    @test haskey(native_meta, recursive.body)
end

@testitem "FunctionDef lowering keeps generated self binders capture free" begin
    using Test
    using LambdaRegression: Transforms, TypeSystem, Utils
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, ConstNode, IndexNode, VarNode
    using LambdaRegression.TypeSystem: TyFun, TyLike

    ctx = TypeSystem.get_type_ctx(Dict{Symbol,Any}(:one => 1))

    function_collision = AbsNode(
        :self,
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            AbsNode(:x, AppNode(IndexNode(1), VarNode(:self))),
        ),
    )
    function_ty = TypeSystem.infer_term(function_collision, ctx)
    @test function_ty isa TyFun
    @test function_ty.res isa TyFun
    lowered_function = Transforms.lower_functiondefs(function_collision)
    @test lowered_function.body.arg.name != :self
    @test lowered_function.body.arg.body.body.arg == VarNode(:self)

    non_function = TypeSystem.infer_term(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(ConstNode(:one)),
        ctx,
    )
    @test non_function isa Utils.Err
    @test non_function.value isa TypeSystem.NonFunctionDefinitionError

    nested_non_function = TypeSystem.infer_term(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            AbsNode(
                :x,
                AppNode(
                    AbsNode(:discarded, IndexNode(1)),
                    LambdaRegression.ASTs.NodeASTs.FunctionDefNode(ConstNode(:one)),
                ),
            ),
        ),
        ctx,
    )
    @test nested_non_function isa Utils.Err
    @test nested_non_function.value isa TypeSystem.NonFunctionDefinitionError

    recursive_head = TypeSystem.infer_term(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
            LambdaRegression.ASTs.NodeASTs.AppNode(IndexNode(0), IndexNode(0)),
        ),
        Dict{Symbol,TyLike}(),
    )
    @test !(recursive_head isa Utils.Err)
end

@testitem "FunctionDef fixed-point witness has the designated closed scheme" begin
    using Test
    using LambdaRegression: Transforms, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.TypeSystem: TyFun, TyLike, TyScheme

    scheme = TypeSystem.infer_scheme(
        AstTree(Transforms.fixed_point_witness()),
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

@testitem "TypeSystem inference exposes source metadata alongside lowering witnesses" begin
    using Test
    using LambdaRegression: Transforms
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, IndexNode
    using LambdaRegression.TypeSystem: TyLike

    struct CallableFixedPoint end
    (::CallableFixedPoint)() = Transforms.fixed_point_witness()

    source = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0)))
    metadata = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    inferred = LambdaRegression.TypeSystem.infer_term(
        source,
        Dict{Symbol,TyLike}();
        meta=metadata,
    )

    @test inferred isa LambdaRegression.TypeSystem.TyFun
    @test haskey(metadata, source)
    @test haskey(metadata, source.body)
    @test all(node === source || node === source.body || node === source.body.body for node in keys(metadata))

    captured_index = IndexNode(2)
    captured = AbsNode(
        :outer,
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, captured_index)),
    )
    captured_metadata = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    @test LambdaRegression.TypeSystem.infer_term(
        captured,
        Dict{Symbol,TyLike}();
        meta=captured_metadata,
    ) isa LambdaRegression.TypeSystem.TyFun
    @test haskey(captured_metadata, captured_index)

    lowered = Transforms.lower_functiondefs(source; fixed_point=CallableFixedPoint())
    @test lowered == AbsNode(:x, IndexNode(0))

    recursive_source = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    witness = Transforms.fixed_point_witness()
    from_node = Transforms.lower_functiondefs(recursive_source; fixed_point=witness)
    from_tree = Transforms.lower_functiondefs(recursive_source; fixed_point=AstTree(witness))
    @test from_node.func == witness
    @test from_tree.func == witness
    @test from_node.func !== witness
    @test from_tree.func !== witness
    @test from_node.func !== from_tree.func
end

@testitem "Execution evaluates finite recursive FunctionDef fixtures" begin
    using Test
    using LambdaRegression: ASTs, Execution, Transforms, TypeSystem
    using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyFun, TyLike, TyList

    app(function_node, arguments...) = foldl(AppNode, arguments; init=function_node)

    integer_values = Dict{Symbol,Any}(
        :COND => LambdaRegression.Primitives.COND,
        :IEQ => LambdaRegression.Primitives.IEQ,
        :IADD => LambdaRegression.Primitives.IADD,
        :ISUB => LambdaRegression.Primitives.ISUB,
        :IMUL => LambdaRegression.Primitives.IMUL,
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
    integer_values[:COND] = LambdaRegression.Primitives.COND

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
    factorial = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(factorial_body)
    @test TypeSystem.iscompatible(
        TypeSystem.infer_term(AstTree(factorial), integer_types),
        TyFun(IntTy, IntTy),
    )
    factorial_4 = AstTree(AppNode(factorial, ConstNode(:int4)))
    factorial_4.root = Transforms.lower_functiondefs(factorial_4.root)
    @test Execution.beta_reduce!(factorial_4, 10_000, integer_values)
    @test integer_values[factorial_4.root.name] == 24
    prepared_factorial = Execution.prepare_native_evaluator(factorial)
    @test Execution.run_native_evaluator(
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
    fibonacci = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(fibonacci_body)
    @test TypeSystem.iscompatible(
        TypeSystem.infer_term(AstTree(fibonacci), integer_types),
        TyFun(IntTy, IntTy),
    )
    fibonacci_6 = AstTree(AppNode(fibonacci, ConstNode(:int6)))
    fibonacci_6.root = Transforms.lower_functiondefs(fibonacci_6.root)
    @test Execution.beta_reduce!(fibonacci_6, 20_000, integer_values)
    @test integer_values[fibonacci_6.root.name] == 8

    list_values = Dict{Symbol,Any}(
        :UNCONS_CASE => LambdaRegression.Primitives.UNCONS_CASE,
        :FADD => LambdaRegression.Primitives.FADD,
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
    list_sum = LambdaRegression.ASTs.NodeASTs.FunctionDefNode(list_sum_body)
    @test TypeSystem.iscompatible(
        TypeSystem.infer_term(AstTree(list_sum), list_types),
        TyFun(TyList(FloatTy), FloatTy),
    )
    sum_values = AstTree(AppNode(list_sum, ConstNode(:values)))
    sum_values.root = Transforms.lower_functiondefs(sum_values.root)
    @test Execution.beta_reduce!(sum_values, 10_000, list_values)
    @test list_values[sum_values.root.name] == 6.0
end

@testitem "Execution native evaluator shares a finite recursion budget" begin
    using Test
    using LambdaRegression: Execution, Utils
    using LambdaRegression.ASTs.NodeASTs: AbsNode, IndexNode, VarNode

    identity = Execution.prepare_native_evaluator(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, IndexNode(0))),
    )
    @test Execution.run_native_evaluator(identity, Dict{Symbol,Any}(); args=(42,)) == 42

    named_identity = Execution.prepare_native_evaluator(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(AbsNode(:x, VarNode(:x))),
    )
    @test Execution.run_native_evaluator(
        named_identity,
        Dict{Symbol,Any}();
        args=(7,),
    ) == 7

    looping = Execution.prepare_native_evaluator(
        LambdaRegression.ASTs.NodeASTs.FunctionDefNode(IndexNode(0)),
    )
    stopped = Execution.run_native_evaluator(
        looping,
        Dict{Symbol,Any}();
        args=(42,),
        max_steps=32,
    )
    @test stopped isa Utils.Err
    @test stopped.value isa Execution.NativeStepLimitError
end
