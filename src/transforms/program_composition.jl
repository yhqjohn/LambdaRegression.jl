using ..Programs: Program
import ..Programs
import ..TypeSystem

export chain_programs, ProgramChainError

"""Expected chaining refusal; `stage` is the receiving stage for a mismatch."""
struct ProgramChainError{D}
    stage::Int
    reason::Symbol
    detail::D
end

function Base.showerror(io::IO, error::ProgramChainError)
    print(io, "program chain stage ", error.stage, ": ", error.reason)
    error.detail === nothing || print(io, " (", error.detail, ")")
end

function _chain_fresh_name!(used::Set{Symbol}, base::Symbol)
    name = base
    suffix = 0
    while name in used
        suffix += 1
        name = Symbol(base, "_", suffix)
    end
    push!(used, name)
    return name
end

function _chain_stage_node(index::SyntaxIndex, id::Int, renames::Dict{Symbol,Symbol})
    node = index.nodes[id]
    child_ids = index.children[id]
    if node isa ConstNode
        return ConstNode(get(renames, node.name, node.name))
    elseif node isa VarNode
        return VarNode(node.name)
    elseif node isa IndexNode
        return IndexNode(node.index)
    elseif node isa AbsNode
        return AbsNode(node.name, _chain_stage_node(index, only(child_ids), renames))
    elseif node isa FunctionDefNode
        return FunctionDefNode(_chain_stage_node(index, only(child_ids), renames))
    elseif node isa AppNode
        return AppNode(
            _chain_stage_node(index, child_ids[1], renames),
            _chain_stage_node(index, child_ids[2], renames),
        )
    end
    throw(ArgumentError("unsupported program-chain node $(typeof(node))"))
end

function _chain_shared_functions(programs, indices, context)
    preserved = Set{Symbol}()
    for program in programs, (name, value) in program.local_consts
        value isa Function || continue
        haskey(context, name) && continue
        name in preserved && continue
        declaration = program.local_consts_ty[name]
        safe = all(eachindex(programs)) do stage
            other = programs[stage]
            if haskey(other.local_consts, name)
                other.local_consts[name] === value || return false
                other_type = other.local_consts_ty[name]
                return TypeSystem.islesseq(declaration, other_type) &&
                       TypeSystem.islesseq(other_type, declaration)
            end
            # Never turn an unresolved global into another stage's local.
            return !any(node -> node isa ConstNode && node.name === name, indices[stage].nodes)
        end
        safe && push!(preserved, name)
    end
    return preserved
end

"""
    chain_programs(programs::Program...; context) -> Union{Program,Err}

Compose one or more closed unary stages in left-to-right order: stages `p, q`
produce the editable source `λx. q(p(x))`. `context` supplies the shared global
type world (empty by default); a `Program` carries no world. Local declarations shadow global
names in their original stage and are deterministically renamed on import.
An unambiguous local host-function name is preserved (and shared across stages
only for identical functions with equivalent types), keeping primitive names
visible to downstream editing. Supplied global names are always protected.
Every stage occurrence receives an independent copy of ordinary local values,
preserving aliases inside that occurrence. Host functions retain `Program`'s
borrowed-resource contract. Caller trees, values, and type graphs are unchanged.

The algorithm resolves scopes with `SyntaxIndex`, infers each renamed stage in
one detached type context, constrains callable heads, and unifies consecutive
ports. `Err(ProgramChainError(stage, reason, detail))` reports empty input (stage
0), open terms, inference/callability failures, or an incompatible input at the
receiving stage. Higher-order and recursive types use the ordinary type system.
Unsupported production syntax follows the common analysis exception boundary.

No beta reduction is performed. Downstream search preparation decides which
editable regions to expose; chaining supplies neither a workflow nor a solver.
"""
function chain_programs(
    programs::Program...;
    context::AbstractDict{Symbol,<:TypeSystem.TyLike}=Dict{Symbol,TypeSystem.TyLike}(),
)
    isempty(programs) && return Err(ProgramChainError(0, :empty_chain, nothing))
    indices = map(analyze_syntax, programs)
    used = Set{Symbol}(keys(context))
    for (stage, index) in enumerate(indices)
        union!(used, keys(programs[stage].local_consts))
        for (id, node) in enumerate(index.nodes)
            if node isa Union{VarNode,IndexNode} && index.reference_binder[id] == 0
                return Err(ProgramChainError(stage, :open_stage, (occurrence=id, node=node)))
            end
            node isa Union{ConstNode,VarNode,AbsNode} && push!(used, node.name)
        end
    end

    values = Dict{Symbol,Any}()
    types = Dict{Symbol,TypeSystem.TyLike}()
    stages = AstNode[]
    shared_functions = _chain_shared_functions(programs, indices, context)
    for (stage, program) in enumerate(programs)
        renames = Dict{Symbol,Symbol}()
        # Copy each occurrence separately, even when the same Program is reused.
        stage_values = Programs._copy_local_values(program.local_consts)
        for (ordinal, name) in enumerate(sort!(collect(keys(stage_values)); by=String))
            renamed = name in shared_functions ? name :
                _chain_fresh_name!(used, Symbol("chain", stage, "_c", ordinal))
            renames[name] = renamed
            values[renamed] = stage_values[name]
            types[renamed] = program.local_consts_ty[name]
        end
        push!(stages, _chain_stage_node(indices[stage], 1, renames))
    end

    query_context = TypeSystem._detached_type_ctx(merge(
        Dict{Symbol,TypeSystem.TyLike}(context), types,
    ))
    factory = TypeSystem.make_tyvar_factory()
    previous_output = nothing
    for (stage, node) in enumerate(stages)
        inferred = TypeSystem.infer_term(node, query_context; factory)
        inferred isa Err && return Err(ProgramChainError(stage, :stage_type, inferred.value))
        ports = TypeSystem.TyFun(factory(0), factory(0))
        callable = TypeSystem.unify!(inferred, ports)
        callable isa Err && return Err(ProgramChainError(stage, :not_callable, callable.value))
        if previous_output !== nothing
            matched = TypeSystem.unify!(previous_output, ports.arg)
            matched isa Err && return Err(ProgramChainError(stage, :stage_mismatch, matched.value))
        end
        previous_output = ports.res
    end

    body::AstNode = IndexNode(0)
    for node in stages
        body = AppNode(node, body)
    end
    input_name = _chain_fresh_name!(used, :chain_input)
    tree = AstTree(AbsNode(input_name, body))
    # Values are already owned; Program still takes its ordinary syntax copy.
    return Program(tree; local_consts=values, local_consts_ty=types, copy_values=identity)
end

@testitem "Transforms chains editable programs in stage order" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike, TyFun, FloatTy, iscompatible
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.Transforms: chain_programs
    using LambdaRegression.Execution: evaluate_default

    context = Dict{Symbol,TyLike}(
        :add => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :mul => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :chain1_c1 => FloatTy,
    )
    stage(op, c) = Program(
        AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(op), IndexNode(0)), ConstNode(:c))));
        local_consts=Dict{Symbol,Any}(:c => c),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy),
    )
    first_stage, second_stage = stage(:add, 2.0), stage(:mul, 3.0)
    combined = chain_programs(first_stage, second_stage; context)
    @test combined isa Program
    combined isa Program || return
    @test combined.tree.root isa AbsNode
    @test combined.tree.root.body.func isa AbsNode
    @test combined.tree.root.body.arg.func isa AbsNode
    @test Set(keys(combined.local_consts)) == Set([:chain1_c1_1, :chain2_c1])
    globals = Dict{Symbol,Any}(:add => (x -> y -> x + y), :mul => (x -> y -> x * y))
    @test evaluate_default(combined, 4.0; input_slots=[:x], global_consts=globals) == 18.0
    @test evaluate_default(chain_programs(second_stage, first_stage; context), 4.0;
        input_slots=[:x], global_consts=globals) == 14.0
    typed = analyze_program(combined, merge(context, combined.local_consts_ty))
    @test iscompatible(typed.root_type, TyFun(FloatTy, FloatTy))
    @test first_stage.local_consts == Dict(:c => 2.0)
    @test second_stage.local_consts == Dict(:c => 3.0)
end

@testitem "Transforms chain isolates local groups and respects local shadowing" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike, TyFun, TyArray, TyNat, FloatTy
    using LambdaRegression.Transforms: chain_programs

    vector_ty = TyArray(TyNat(1), FloatTy)
    material = [2.0]
    stage = Program(AstTree(AbsNode(:x, ConstNode(:a)));
        local_consts=Dict{Symbol,Any}(:a => material, :b => material),
        local_consts_ty=Dict{Symbol,TyLike}(:a => vector_ty, :b => vector_ty))
    context = Dict{Symbol,TyLike}(:a => FloatTy)
    combined = chain_programs(stage, stage; context)
    @test combined isa Program
    combined isa Program || return
    @test combined.local_consts[:chain1_c1] === combined.local_consts[:chain1_c2]
    @test combined.local_consts[:chain2_c1] === combined.local_consts[:chain2_c2]
    @test combined.local_consts[:chain1_c1] !== combined.local_consts[:chain2_c1]
    @test combined.local_consts[:chain1_c1] !== stage.local_consts[:a]
    combined.local_consts[:chain1_c1][1] = 9.0
    @test combined.local_consts[:chain2_c1] == [2.0]
    @test stage.local_consts[:a] == [2.0]
    @test material == [2.0]
    @test combined.tree.root.body.func.body == ConstNode(:chain2_c1)
end

@testitem "Transforms chain rejects invalid stages without mutating caller types" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike, TyFun, FloatTy, BoolTy, make_tyvar_factory
    using LambdaRegression.Transforms: chain_programs, ProgramChainError
    using LambdaRegression.Utils: Err

    factory = make_tyvar_factory()
    shared = factory(0)
    context = Dict{Symbol,TyLike}(
        :f => TyFun(shared, shared),
        :g => TyFun(FloatTy, BoolTy),
        :h => TyFun(FloatTy, FloatTy),
        :value => FloatTy,
    )
    named(name) = Program(AstTree(ConstNode(name)))
    result = chain_programs(named(:f), named(:h); context)
    @test result isa Program
    @test shared.instance === nothing
    @test context[:f].arg === shared
    @test context[:f].res === shared
    rejected = chain_programs(named(:f), named(:g), named(:h); context)
    @test rejected isa Err{<:ProgramChainError}
    @test rejected.value.stage == 3
    @test rejected.value.reason == :stage_mismatch
    @test shared.instance === nothing
    @test chain_programs(; context).value.reason == :empty_chain
    @test chain_programs(named(:value); context).value.reason == :not_callable
    @test chain_programs(named(:missing); context).value.reason == :stage_type
    @test chain_programs(Program(AstTree(IndexNode(0))); context).value.reason == :open_stage
    @test chain_programs(Program(AstTree(VarNode(:f))); context).value.reason == :open_stage
end

@testitem "Transforms chain preserves higher-order and recursive binder scopes" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike, TyFun, IntTy, iscompatible
    using LambdaRegression.Transforms: chain_programs
    using LambdaRegression.ProgramAnalysis: analyze_program, analyze_syntax
    using LambdaRegression.Execution: evaluate_default

    context = Dict{Symbol,TyLike}()
    identity_stage = Program(AstTree(AbsNode(:named, VarNode(:named))))
    # The unused let argument contains a self call; call-by-need leaves it cold.
    recursive = Program(AstTree(FunctionDefNode(AbsNode(:x,
        AppNode(AbsNode(:unused, IndexNode(1)), AppNode(IndexNode(1), IndexNode(0))),
    ))))
    combined = chain_programs(recursive, identity_stage; context)
    @test combined isa Program
    combined isa Program || return
    @test evaluate_default(combined, 7; input_slots=[:x]) == 7
    index = analyze_syntax(combined)
    @test all(index.reference_binder[id] != 0 for (id, node) in enumerate(index.nodes)
        if node isa Union{IndexNode,VarNode})
    @test count(node -> node isa FunctionDefNode, index.nodes) == 1

    function_ty = TyFun(IntTy, IntTy)
    higher_context = Dict{Symbol,TyLike}(
        :producer => TyFun(IntTy, function_ty),
        :consumer => TyFun(function_ty, IntTy),
    )
    higher = chain_programs(Program(AstTree(ConstNode(:producer))),
        Program(AstTree(ConstNode(:consumer))); context=higher_context)
    @test higher isa Program
    typed = analyze_program(higher, higher_context)
    @test iscompatible(typed.root_type, function_ty)
end

@testitem "Transforms chain preserves editable primitive names without capturing globals" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.TypeSystem: TyLike, FloatTy, TyFun, get_type
    using LambdaRegression.Transforms: chain_programs
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.ScalarRegions: scalar_region_index, scalar_region_facts
    using LambdaRegression.Execution: evaluate_default
    using LambdaRegression.Utils: Err

    add_stage = Program(AstTree(AbsNode(:x,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:c))));
        local_consts=Dict{Symbol,Any}(:FADD => FADD, :c => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:FADD => get_type(FADD), :c => FloatTy))
    combined = chain_programs(add_stage, add_stage)
    @test combined isa Program
    combined isa Program || return
    @test combined.local_consts[:FADD] === FADD
    @test count(==(:FADD), collect(keys(combined.local_consts))) == 1
    @test evaluate_default(combined, 3.0; input_slots=[:x]) == 7.0
    analysis = analyze_program(combined, combined.local_consts_ty;
        target_ty=TyFun(FloatTy, FloatTy))
    regions = scalar_region_index(combined, analysis; type_context=Dict{Symbol,TyLike}())
    facts = [scalar_region_facts(regions, id) for id in eachindex(analysis.index.nodes)]
    add_occurrences = findall(analysis.index.nodes) do node
        node isa AppNode && node.func isa AppNode && node.func.func == ConstNode(:FADD)
    end
    @test length(add_occurrences) == 2
    @test all(id -> facts[id] !== nothing && :+ in facts[id].binary_operators, add_occurrences)

    # A bare missing reference must not acquire another stage's captured value.
    missing = Program(AstTree(ConstNode(:FADD)))
    rejected = chain_programs(add_stage, missing)
    @test rejected isa Err
    @test rejected.value.stage == 2
    @test rejected.value.reason == :stage_type

    # Different host functions under one original name remain separate.
    f, g = (x -> x + 1.0), (x -> x * 2.0)
    local_stage(value) = Program(AstTree(ConstNode(:f));
        local_consts=Dict{Symbol,Any}(:f => value),
        local_consts_ty=Dict{Symbol,TyLike}(:f => TyFun(FloatTy, FloatTy)))
    different = chain_programs(local_stage(f), local_stage(g))
    @test !haskey(different.local_consts, :f)
    @test evaluate_default(different, 3.0; input_slots=[:x]) == 8.0
end

@testitem "Transforms chain unifies equirecursive ports" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem:
        TyLike, TyFun, TyMu, TyProd, IntTy, make_tyvar_factory, iscompatible
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.Transforms: chain_programs

    fresh = make_tyvar_factory()
    a, b = fresh(0), fresh(0)
    one_cycle = TyMu(a, TyProd(IntTy, a))
    two_cycle = TyMu(b, TyProd(IntTy, TyProd(IntTy, b)))
    context = Dict{Symbol,TyLike}(
        :make => TyFun(IntTy, one_cycle),
        :take => TyFun(two_cycle, IntTy),
    )
    combined = chain_programs(Program(AstTree(ConstNode(:make))),
        Program(AstTree(ConstNode(:take))); context)
    @test combined isa Program
    combined isa Program || return
    typed = analyze_program(combined, context)
    @test iscompatible(typed.root_type, TyFun(IntTy, IntTy))
    @test a.instance === nothing
    @test b.instance === nothing
end
