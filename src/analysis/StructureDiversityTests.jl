# Counterexample-led contract checks for StructureDiversity. The parent module
# includes this file after defining and exporting the descriptor API.

@testitem "Structure diversity alias spines, grammar changes and opaque mu boundaries" begin
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, IntTy
    using LambdaRegression.Primitives: FSIN, FADD
    using LambdaRegression.ProgramAnalysis: analyze_program

    types = Dict{Symbol,TyLike}(:FSIN => TyFun(FloatTy, FloatTy),
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    world = StructureWorld(types; input_types=Dict(:x => FloatTy),
        values=Dict(:FSIN => FSIN, :FADD => FADD), target_ty=FloatTy)
    raw = StructureBoundaryPolicy(view=:bindings)
    direct = Program(AstTree(AppNode(ConstNode(:FSIN), ConstNode(:x))))
    aliased = Program(AstTree(AppNode(AppNode(AbsNode(:f, IndexNode(0)),
        ConstNode(:FSIN)), ConstNode(:x))))
    a = describe_structure(direct, world; boundary_policy=raw)
    b = describe_structure(aliased, world; boundary_policy=raw)
    @test a.canonical_key == b.canonical_key
    @test length(b.calls) == 1
    @test only(b.calls).summary.given_arity == 1
    @test !only(b.calls).summary.unknown
    @test all(e -> 1 <= e.consumer <= length(b.nodes) &&
        1 <= e.source <= length(b.nodes), b.edges)

    ordinary = Program(AstTree(ConstNode(:x)))
    changed_grammar = Program(AstTree(ConstNode(:x));
        local_consts=Dict(:FADD => identity),
        local_consts_ty=Dict(:FADD => TyFun(IntTy, IntTy)))
    c = describe_structure(ordinary, world)
    d = describe_structure(changed_grammar, world)
    @test length(c.holes) == length(d.holes) == 1
    @test only(c.holes).available_ports == only(d.holes).available_ports
    @test c.canonical_key != d.canonical_key

    recursive_var = newtyvar(0)
    recursive_type = TyMu(recursive_var, TySum(FloatTy, recursive_var))
    mu_program = Program(AstTree(ConstNode(:data)))
    mu_world = StructureWorld(Dict(:data => recursive_type); target_ty=recursive_type)
    mu_description = describe_structure(mu_program, mu_world)
    @test isempty(mu_description.holes)
    @test length(mu_description.nodes) == 1
    @test ncodeunits(mu_description.canonical_key.encoding) < 5000

    identity_program = Program(AstTree(AbsNode(:x, IndexNode(0))))
    evidence = analyze_program(identity_program, Dict{Symbol,TyLike}();
        target_ty=TyFun(FloatTy, FloatTy))
    wrong_world = StructureWorld(Dict{Symbol,TyLike}();
        target_ty=TyFun(IntTy, IntTy))
    @test_throws ArgumentError describe_structure(identity_program, wrong_world; analysis=evidence)
end

@testitem "Structure diversity normalizes binders and keeps scalar material separate" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity: StructureWorld,
        StructureBoundaryPolicy, describe_structure
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    world = StructureWorld(
        Dict{Symbol,TyLike}(:FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)));
        values=Dict{Symbol,Any}(:FADD => FADD),
        target_ty=TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
    )
    bindings = StructureBoundaryPolicy(view=:bindings)

    named = Program(AstTree(AbsNode(:x, AbsNode(:y, VarNode(:x)))))
    indexed = Program(AstTree(AbsNode(:renamed, AbsNode(:other, IndexNode(1)))))
    @test describe_structure(named, world; boundary_policy=bindings).canonical_key ==
        describe_structure(indexed, world; boundary_policy=bindings).canonical_key

    both = Program(AstTree(AbsNode(:x, AbsNode(:y,
        add(VarNode(:x), VarNode(:y))))))
    left_only = Program(AstTree(AbsNode(:renamed_x, AbsNode(:renamed_y,
        VarNode(:renamed_x)))))
    both_description = describe_structure(both, world)
    left_description = describe_structure(left_only, world)
    @test both_description.canonical_key == left_description.canonical_key
    @test length(both_description.holes) == 1
    @test length(left_description.holes) == 1
    @test only(both_description.holes).available_ports ==
        only(left_description.holes).available_ports
    @test only(both_description.holes).used_slots == [1, 2]
    @test only(left_description.holes).used_slots == [2]
end

@testitem "Structure diversity named input policy changes hole ports" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode, VarNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity: StructureWorld,
        StructureBoundaryPolicy, describe_structure
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    tree = AstTree(AppNode(AppNode(ConstNode(:FADD), ConstNode(:x)), ConstNode(:x)))
    program = Program(tree)
    world = StructureWorld(
        Dict{Symbol,TyLike}(:FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)));
        input_types=Dict{Symbol,TyLike}(:x => FloatTy, :unused => FloatTy),
        values=Dict{Symbol,Any}(:FADD => FADD),
    )
    used = describe_structure(program, world;
        boundary_policy=StructureBoundaryPolicy(named_input_policy=:used))
    declared = describe_structure(program, world;
        boundary_policy=StructureBoundaryPolicy(named_input_policy=:declared))

    @test length(used.holes) == 1
    @test length(declared.holes) == 1
    used_ports = only(used.holes).available_ports
    declared_ports = only(declared.holes).available_ports
    @test "input:x" in used_ports
    @test !("input:unused" in used_ports)
    @test "input:x" in declared_ports
    @test "input:unused" in declared_ports
    @test used.canonical_key != declared.canonical_key
end

@testitem "Structure diversity preserves alias safety, shared bodies, and captures" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity:
        StructureWorld, StructureBoundaryPolicy, describe_structure
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    types = Dict{Symbol,TyLike}(:FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    world = StructureWorld(types; values=Dict{Symbol,Any}(:FADD => FADD))
    bindings = StructureBoundaryPolicy(view=:bindings)

    direct = Program(AstTree(ConstNode(:FADD)))
    function_alias = Program(AstTree(AppNode(
        AbsNode(:alias, VarNode(:alias)), ConstNode(:FADD),
    )))
    @test describe_structure(direct, world; boundary_policy=bindings).canonical_key ==
        describe_structure(function_alias, world; boundary_policy=bindings).canonical_key

    float_alias = Program(AstTree(AbsNode(:x, AppNode(
        AbsNode(:y, add(VarNode(:y), VarNode(:y))), IndexNode(0),
    ))))
    direct_float = Program(AstTree(AbsNode(:x,
        add(VarNode(:x), VarNode(:x)))))
    @test describe_structure(float_alias, world; boundary_policy=bindings).canonical_key !=
        describe_structure(direct_float, world; boundary_policy=bindings).canonical_key
    float_alias_description = describe_structure(float_alias, world)
    direct_float_description = describe_structure(direct_float, world)
    @test !isempty(float_alias_description.holes)
    @test !isempty(direct_float_description.holes)
    @test length(float_alias_description.holes) == 2
    @test any(hole -> length(hole.available_ports) == 2, float_alias_description.holes)
    @test length(only(direct_float_description.holes).available_ports) == 1

    captured = Program(AstTree(AbsNode(:outer,
        AbsNode(:inner, VarNode(:outer)))))
    captured_description = describe_structure(captured, world; boundary_policy=bindings)
    @test any(edge -> edge.role === :capture, captured_description.edges)

    shared = Program(AstTree(AbsNode(:x, AbsNode(:y, AppNode(
        AbsNode(:f, add(
            AppNode(VarNode(:f), VarNode(:y)),
            AppNode(VarNode(:f), VarNode(:y)),
        )),
        AbsNode(:u, add(VarNode(:u), VarNode(:x))),
    )))))
    independent = Program(AstTree(AbsNode(:x, AbsNode(:y, add(
        AppNode(AbsNode(:u, add(VarNode(:u), VarNode(:x))), VarNode(:y)),
        AppNode(AbsNode(:v, add(VarNode(:v), VarNode(:x))), VarNode(:y)),
    )))))
    shared_description = describe_structure(shared, world; boundary_policy=bindings)
    independent_description = describe_structure(independent, world; boundary_policy=bindings)
    @test shared_description.canonical_key != independent_description.canonical_key
    shared_values = filter(edge -> edge.role === :bound_value, shared_description.edges)
    independent_values = filter(edge -> edge.role === :bound_value, independent_description.edges)
    @test length(shared_values) == 1
    @test length(independent_values) == 2
    @test length(unique(edge.source for edge in shared_values)) == 1
    @test length(unique(edge.source for edge in independent_values)) == 2

    shared_analysis = analyze_program(shared, merge(copy(types), shared.local_consts_ty))
    f_occurrence = findfirst(node -> node isa AbsNode && node.name === :f,
        shared_analysis.index.nodes)
    f_binder = shared_analysis.index.owner_binder[f_occurrence]
    f_binding_node = only(node.id for node in shared_description.nodes
        if node.occurrence == f_occurrence)
    @test count(edge -> edge.role === :binding && edge.source == f_binding_node,
        shared_description.edges) == count(==(f_binder),
            shared_analysis.index.reference_binder) == 2
end

@testitem "Structure diversity keeps unknown calls and lexical shadowing conservative" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, VarNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity:
        StructureWorld, StructureBoundaryPolicy, describe_structure
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    calls = Ref(0)
    opaque = x -> (calls[] += 1; x)
    function_ty = TyFun(FloatTy, FloatTy)
    types = Dict{Symbol,TyLike}(
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :opaque => function_ty,
        :missing => function_ty,
    )
    world = StructureWorld(types;
        input_types=Dict{Symbol,TyLike}(:x => FloatTy, :y => FloatTy),
        values=Dict{Symbol,Any}(:FADD => FADD, :opaque => opaque),
    )
    bindings = StructureBoundaryPolicy(view=:bindings)

    opaque_program = Program(AstTree(AppNode(ConstNode(:opaque), ConstNode(:x))))
    opaque_description = describe_structure(opaque_program, world;
        boundary_policy=bindings)
    @test only(opaque_description.calls).summary.unknown
    @test only(opaque_description.calls).summary.callee_dependency
    @test calls[] == 0

    missing_program = Program(AstTree(AppNode(ConstNode(:missing), ConstNode(:x))))
    missing_description = describe_structure(missing_program, world;
        boundary_policy=bindings)
    @test :opaque_global in missing_description.uncertainty
    @test only(missing_description.calls).summary.unknown

    shadowed = Program(AstTree(AbsNode(:FADD,
        AppNode(AppNode(VarNode(:FADD), ConstNode(:x)), ConstNode(:y)))))
    shadowed_description = describe_structure(shadowed, world;
        boundary_policy=bindings)
    @test only(shadowed_description.calls).summary.unknown
    @test any(node -> node.kind === :reference, shadowed_description.nodes)
    @test !any(node -> node.kind === :resource, shadowed_description.nodes)
end

@testitem "Structure diversity preserves fixed recursion and rejects foreign evidence" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity:
        StructureWorld, StructureBoundaryPolicy, describe_structure
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    target = TyFun(FloatTy, FloatTy)
    world = StructureWorld(Dict{Symbol,TyLike}(); target_ty=target)
    bindings = StructureBoundaryPolicy(view=:bindings)
    recursive = Program(AstTree(FunctionDefNode(AbsNode(:x,
        AppNode(IndexNode(1), IndexNode(0)),
    ))))
    recursive_description = describe_structure(recursive, world;
        boundary_policy=bindings)
    @test recursive_description.source_node_count == length(recursive_description.nodes)
    @test any(node -> node.kind === :fixpoint, recursive_description.nodes)
    @test :recursive_flow in recursive_description.uncertainty

    context = Dict{Symbol,TyLike}(:one => FloatTy)
    first = Program(AstTree(ConstNode(:one)))
    second = Program(AstTree(ConstNode(:one)))
    evidence = analyze_program(first, context)
    @test evidence isa TypedAnalysis
    @test_throws ArgumentError describe_structure(second,
        StructureWorld(context); analysis=evidence, boundary_policy=bindings)
end

@testitem "Structure diversity keys obey hash and dictionary equality boundaries" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.StructureDiversity:
        StructureWorld, StructureBoundaryPolicy, describe_structure, bucket_counts
    using LambdaRegression.TypeSystem: TyLike

    world = StructureWorld(Dict{Symbol,TyLike}())
    policy = StructureBoundaryPolicy(view=:bindings)
    left = Program(AstTree(AbsNode(:x, IndexNode(0))))
    alpha = Program(AstTree(AbsNode(:renamed, IndexNode(0))))
    changed = Program(AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1)))))
    left_description = describe_structure(left, world; boundary_policy=policy)
    alpha_description = describe_structure(alpha, world; boundary_policy=policy)
    changed_description = describe_structure(changed, world; boundary_policy=policy)

    left_key = left_description.canonical_key
    alpha_key = alpha_description.canonical_key
    changed_key = changed_description.canonical_key
    @test isequal(left_key, alpha_key)
    @test hash(left_key) == hash(alpha_key)
    @test !isequal(left_key, changed_key)

    counts = bucket_counts((left_description, alpha_description, changed_description))
    @test length(counts) == 2
    @test sort!(collect(values(counts))) == [1, 2]
    separate_world = StructureWorld(Dict{Symbol,TyLike}())
    other_world_description = describe_structure(left, separate_world;
        boundary_policy=policy)
    @test !isequal(left_key, other_world_description.canonical_key)
    @test length(Dict(left_key => :same, alpha_key => :alpha,
        other_world_description.canonical_key => :other_world)) == 2
end
