"""
    ProgramAnalysis

Compact common syntax and typing facts for one program.  `SyntaxIndex` is the
structural source of truth: each syntactic occurrence receives one integer ID,
with parent edges, children, structural depth, subtree interval, a linked
lexical environment head, and resolved binder references.  `TypedAnalysis`
adds one W metadata table and the exact schemes recorded at the binders.

The index borrows a published `Program` tree.  An external `AstTree` is copied
once at this boundary.  Paths, visible binders, and captures are derived only
when requested; no per-node path, copied environment, or eager site/capture
projection is stored.  `ctx` is the complete query context supplied by the
caller; local declaration types must be merged into it before analysis.
"""
module ProgramAnalysis

using TestItems

using ..ASTs.NodeASTs:
    AstNode, AstTree, AppNode, AbsNode, ConstNode, FunctionDefNode,
    IndexNode, VarNode, children, copy_occurrences
using ..Programs: Program
using ..TypeSystem
using ..Utils: Err

export BindingInfo, SyntaxIndex, TypedAnalysis
export analyze_syntax, analyze_program, path, visible_binder_ids, captured_binder_ids
export UnsupportedSyntaxError
export functiondef_recursion_observations, functiondef_recursion_summary
export obvious_functiondef_loop_reason

"""One lexical binder in the linked environment graph."""
struct BindingInfo
    owner::Int
    kind::Symbol
    name::Union{Nothing,Symbol}
    outer::Int
    depth::Int
end

"""Expected refusal for a retired or unknown production node."""
struct UnsupportedSyntaxError <: Exception
    node_type::DataType
    occurrence::Int
end
Base.showerror(io::IO, err::UnsupportedSyntaxError) = print(
    io,
    "unsupported common-language node ", err.node_type,
    " at occurrence ", err.occurrence,
)

"""Compact occurrence index; all vectors use one-based occurrence IDs."""
struct SyntaxIndex
    tree::AstTree
    nodes::Vector{AstNode}
    parent::Vector{Int}
    parent_edge::Vector{Int}
    children::Vector{Vector{Int}}
    depth::Vector{Int}
    subtree_end::Vector{Int}
    env_head::Vector{Int}
    reference_binder::Vector{Int}
    owner_binder::Vector{Int}
    bindings::Vector{BindingInfo}
end

"""One W analysis attached to a compact `SyntaxIndex`."""
struct TypedAnalysis
    index::SyntaxIndex
    metadata::Dict{Int,Tuple{TypeSystem.TyLike,TypeSystem.TyLike}}
    binder_schemes::Dict{Int,TypeSystem.TyLike}
    root_type::TypeSystem.TyLike
end

Base.length(index::SyntaxIndex) = length(index.nodes)
Base.length(analysis::TypedAnalysis) = length(analysis.index)

function _reject_common_node(node::AstNode, id::Int)
    node isa Union{AppNode,AbsNode,ConstNode,FunctionDefNode,IndexNode,VarNode} ||
        throw(UnsupportedSyntaxError(typeof(node), id))
    return nothing
end

"""Build a compact syntax index from an owned or borrowed AST tree."""
function _build_index(tree::AstTree)
    nodes = AstNode[]
    parent = Int[]
    parent_edge = Int[]
    children_index = Vector{Vector{Int}}()
    depth = Int[]
    subtree_end = Int[]
    env_head = Int[]
    reference_binder = Int[]
    owner_binder = Int[]
    bindings = BindingInfo[]
    active = IdDict{AstNode,Nothing}()

    function visit(
        node::AstNode,
        parent_id::Int,
        edge::Int,
        structural_depth::Int,
        current_env::Int,
        binder_stack::Vector{Int},
    )
        id = length(nodes) + 1
        _reject_common_node(node, id)
        haskey(active, node) && throw(ArgumentError("cyclic host AST occurrence"))
        active[node] = nothing

        push!(nodes, node)
        push!(parent, parent_id)
        push!(parent_edge, edge)
        push!(children_index, Int[])
        push!(depth, structural_depth)
        push!(subtree_end, 0)
        push!(env_head, current_env)
        push!(reference_binder, 0)
        push!(owner_binder, 0)

        if node isa IndexNode
            reference_binder[id] = 0 <= node.index < length(binder_stack) ?
                binder_stack[end - node.index] : 0
        elseif node isa VarNode
            named = findlast(
                binder_id -> bindings[binder_id].name === node.name,
                binder_stack,
            )
            reference_binder[id] = named === nothing ? 0 : binder_stack[named]
        end

        # The stack is temporary traversal state; the persistent index stores
        # only its head and one immutable binding record per lexical binder.
        child_env = current_env
        pushed_binder = false
        if node isa AbsNode || node isa FunctionDefNode
            kind = node isa AbsNode ? :abs : :function_def
            name = node isa AbsNode ? node.name : nothing
            outer_depth = current_env == 0 ? 0 : bindings[current_env].depth
            push!(bindings, BindingInfo(id, kind, name, current_env, outer_depth + 1))
            binder_id = length(bindings)
            owner_binder[id] = binder_id
            child_env = binder_id
            push!(binder_stack, binder_id)
            pushed_binder = true
        end

        for (child_edge, child) in enumerate(children(node))
            push!(children_index[id], visit(
                child, id, child_edge, structural_depth + 1, child_env,
                binder_stack,
            ))
        end
        pushed_binder && pop!(binder_stack)
        subtree_end[id] = length(nodes) + 1
        delete!(active, node)
        return id
    end

    visit(tree.root, 0, 0, 0, 0, Int[])
    return SyntaxIndex(
        tree, nodes, parent, parent_edge, children_index, depth, subtree_end,
        env_head, reference_binder, owner_binder, bindings,
    )
end

"""Build an owned structural index from external syntax."""
analyze_syntax(tree::AstTree) = _build_index(copy_occurrences(tree))

"""Build a borrowed structural index from a published program."""
analyze_syntax(program::Program) = _build_index(program.tree)

function _path(index::SyntaxIndex, id::Int)
    1 <= id <= length(index) || throw(BoundsError(index.nodes, id))
    result = Int[]
    current = id
    while index.parent[current] != 0
        push!(result, index.parent_edge[current])
        current = index.parent[current]
    end
    reverse!(result)
    return result
end

"""Recover one occurrence path along the compact parent chain."""
path(index::SyntaxIndex, id::Int) = _path(index, id)
path(analysis::TypedAnalysis, id::Int) = _path(analysis.index, id)

"""Recover visible snapshot binder IDs, innermost first, on demand."""
function visible_binder_ids(index::SyntaxIndex, id::Int)
    1 <= id <= length(index) || throw(BoundsError(index.nodes, id))
    result = Int[]
    current = index.env_head[id]
    while current != 0
        push!(result, current)
        current = index.bindings[current].outer
    end
    return result
end
visible_binder_ids(analysis::TypedAnalysis, id::Int) = visible_binder_ids(analysis.index, id)

"""Return captured snapshot binder IDs for one explicit subtree query."""
function captured_binder_ids(index::SyntaxIndex, id::Int)
    1 <= id <= length(index) || throw(BoundsError(index.nodes, id))
    stop = index.subtree_end[id]
    result = Int[]
    seen = Set{Int}()
    for occurrence in id:(stop - 1)
        binder = index.reference_binder[occurrence]
        binder == 0 && continue
        owner = index.bindings[binder].owner
        id <= owner < stop && continue
        binder in seen && continue
        push!(seen, binder)
        push!(result, binder)
    end
    return result
end
captured_binder_ids(analysis::TypedAnalysis, id::Int) = captured_binder_ids(analysis.index, id)

"""Temporary adapter that lets Algorithm W consume integer occurrence IDs."""
struct _IndexTermAdapter
    index::SyntaxIndex
    binder_schemes::Dict{Int,TypeSystem.TyLike}
end

import ..TypeSystem: term_view, constrain_fixpoint, record_binding

function term_view(adapter::_IndexTermAdapter, id::Int)
    node = adapter.index.nodes[id]
    children_ids = adapter.index.children[id]
    node isa ConstNode && return (:const, node.name)
    node isa VarNode && return (:var, node.name)
    node isa IndexNode && return (:var, node.index)
    node isa AbsNode && return (:abs, node.name, children_ids[1])
    node isa AppNode && return (:app, children_ids[1], children_ids[2])
    node isa FunctionDefNode && return (:fix, nothing, children_ids[1])
    throw(UnsupportedSyntaxError(typeof(node), id))
end

function constrain_fixpoint(
    adapter::_IndexTermAdapter,
    id::Int,
    ty::TypeSystem.Ty,
    factory,
)
    return TypeSystem.constrain_fixpoint(
        TypeSystem.NodeTermAdapter(), adapter.index.nodes[id], ty, factory,
    )
end

function record_binding(
    adapter::_IndexTermAdapter,
    owner::Int,
    scheme::TypeSystem.TyLike,
)
    binder = adapter.index.owner_binder[owner]
    binder == 0 || (adapter.binder_schemes[binder] = scheme)
    return nothing
end

function _normalize_bound(bound::TypeSystem.TyLike)
    bound isa TypeSystem.TyScheme || return TypeSystem.prune(bound)
    body = TypeSystem.prune(bound.ty)
    free = TypeSystem.collect_free_vars(body)
    vars = TypeSystem.TyVar[
        variable for variable in bound.vars
        if variable.instance === nothing && variable in free
    ]
    return TypeSystem.TyScheme(vars, body)
end

"""Infer one indexed program while preserving the exact recorded binder schemes."""
function _typed_analysis(
    index::SyntaxIndex,
    ctx::AbstractDict{Symbol,<:TypeSystem.TyLike};
    factory,
    target_ty,
)
    metadata = Dict{Int,Tuple{TypeSystem.TyLike,TypeSystem.TyLike}}()
    binder_schemes = Dict{Int,TypeSystem.TyLike}()
    adapter = _IndexTermAdapter(index, binder_schemes)
    inference_ctx, detached_targets = target_ty === nothing ?
        (TypeSystem._detached_type_ctx(ctx), ()) :
        TypeSystem._detached_type_inputs(ctx, target_ty)
    root_type = TypeSystem.infer_term(
        1, inference_ctx, adapter; factory, meta=metadata,
    )
    root_type isa Err && return root_type
    if target_ty !== nothing
        constrained = TypeSystem.unify!(root_type, only(detached_targets))
        constrained isa Err && return constrained
        root_type = TypeSystem.prune(root_type)
    end
    for (id, bounds) in metadata
        metadata[id] = (_normalize_bound(bounds[1]), _normalize_bound(bounds[2]))
    end
    for (binder, scheme) in binder_schemes
        binder_schemes[binder] = _normalize_bound(scheme)
    end
    return TypedAnalysis(index, metadata, binder_schemes, TypeSystem.prune(root_type))
end

"""Analyze external syntax, owning its tree copy."""
function analyze_program(
    tree::AstTree,
    ctx::AbstractDict{Symbol,<:TypeSystem.TyLike};
    factory=TypeSystem.make_tyvar_factory(),
    target_ty::Union{Nothing,TypeSystem.Ty}=nothing,
)
    return _typed_analysis(analyze_syntax(tree), ctx; factory, target_ty)
end

"""Analyze a published program while borrowing its read-only tree."""
function analyze_program(
    program::Program,
    ctx::AbstractDict{Symbol,<:TypeSystem.TyLike};
    factory=TypeSystem.make_tyvar_factory(),
    target_ty::Union{Nothing,TypeSystem.Ty}=nothing,
)
    return _typed_analysis(analyze_syntax(program), ctx; factory, target_ty)
end

_empty_self_observation() = (self_references=0, self_calls=0)

_add_self_observations(left, right) = (
    self_references=left.self_references + right.self_references,
    self_calls=left.self_calls + right.self_calls,
)

function _application_spine(index::SyntaxIndex, id::Int)
    arguments = Int[]
    head = id
    while index.nodes[head] isa AppNode
        func_id, arg_id = index.children[head]
        push!(arguments, arg_id)
        head = func_id
    end
    return head, arguments
end

function _functiondef_self_observation(index::SyntaxIndex, id::Int, self_index::Int)
    node = index.nodes[id]
    if node isa IndexNode
        return node.index == self_index ?
            (self_references=1, self_calls=0) : _empty_self_observation()
    elseif node isa AppNode
        head, arguments = _application_spine(index, id)
        head_node = index.nodes[head]
        total = if head_node isa IndexNode && head_node.index == self_index
            (self_references=1, self_calls=1)
        else
            _functiondef_self_observation(index, head, self_index)
        end
        for argument in arguments
            total = _add_self_observations(
                total,
                _functiondef_self_observation(index, argument, self_index),
            )
        end
        return total
    elseif node isa AbsNode || node isa FunctionDefNode
        return _functiondef_self_observation(index, only(index.children[id]), self_index + 1)
    end
    return _empty_self_observation()
end

function _collect_functiondef_observations!(observations, index::SyntaxIndex, id::Int)
    node = index.nodes[id]
    if node isa FunctionDefNode
        body_id = only(index.children[id])
        push!(observations, (
            body_signature=sprint(show, index.nodes[body_id]),
            _functiondef_self_observation(index, body_id, 0)...,
        ))
    end
    for child_id in index.children[id]
        _collect_functiondef_observations!(observations, index, child_id)
    end
    return observations
end

"""
    functiondef_recursion_observations(term) -> Vector{NamedTuple}

Return one preorder observation per common-language `FunctionDefNode`.
`self_references` counts uses of that definition's anonymous binder, while
`self_calls` counts maximal application spines headed by it. Unsupported syntax
is rejected by the common syntax index. External AST inputs
are owned/copied; a published `Program` is analyzed through its read-only tree.
"""
function _functiondef_recursion_observations(index::SyntaxIndex)
    return _collect_functiondef_observations!(NamedTuple[], index, 1)
end

function functiondef_recursion_observations(term::Union{AstNode,AstTree})
    tree = term isa AstTree ? term : AstTree(term)
    return _functiondef_recursion_observations(analyze_syntax(tree))
end

function functiondef_recursion_observations(program::Program)
    return _functiondef_recursion_observations(analyze_syntax(program))
end

"""Summarize genuine recursive calls across common-language FunctionDefs."""
function functiondef_recursion_summary(term::Union{AstNode,AstTree,Program})
    observations = functiondef_recursion_observations(term)
    return (
        definitions=length(observations),
        recursive_definitions=count(item -> item.self_calls > 0, observations),
        self_references=sum(item -> item.self_references, observations; init=0),
        self_calls=sum(item -> item.self_calls, observations; init=0),
    )
end

function _functiondef_obvious_loop_reason(index::SyntaxIndex, id::Int)
    node = index.nodes[id]::FunctionDefNode
    body = only(index.children[id])
    body_node = index.nodes[body]
    body_node isa IndexNode && body_node.index == 0 && return :direct_self

    lambda_count = 0
    current = body
    while index.nodes[current] isa AbsNode
        lambda_count += 1
        current = only(index.children[current])
    end
    current_node = index.nodes[current]
    if lambda_count > 0 && current_node isa AppNode
        head, reverse_arguments = _application_spine(index, current)
        head_node = index.nodes[head]
        if head_node isa IndexNode && head_node.index == lambda_count &&
           length(reverse_arguments) == lambda_count &&
           all(enumerate(reverse_arguments)) do (position, argument)
               argument_node = index.nodes[argument]
               argument_node isa IndexNode && argument_node.index == position - 1
           end
            return :eta_self_forwarding
        end
        head_node isa IndexNode && head_node.index == lambda_count &&
            !isempty(reverse_arguments) && return :unconditional_self_call
    end
    _functiondef_unchanged_self_call(index, id) && return :unchanged_self_call
    return nothing
end

function _has_unchanged_self_call(
    index::SyntaxIndex,
    id::Int,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    node = index.nodes[id]
    if node isa AppNode
        head, reverse_arguments = _application_spine(index, id)
        head_node = index.nodes[head]
        if head_node isa IndexNode && head_node.index == self_index &&
           length(reverse_arguments) == length(parameter_indices) &&
           all(zip(reverse_arguments, parameter_indices)) do (argument, expected)
               argument_node = index.nodes[argument]
               argument_node isa IndexNode && argument_node.index == expected
           end
            return true
        end
        return _has_unchanged_self_call(index, index.children[id][1], self_index, parameter_indices) ||
            _has_unchanged_self_call(index, index.children[id][2], self_index, parameter_indices)
    elseif node isa AbsNode || node isa FunctionDefNode
        return false
    end
    return any(child_id -> _has_unchanged_self_call(
        index, child_id, self_index, parameter_indices,
    ), index.children[id])
end

function _functiondef_unchanged_self_call(index::SyntaxIndex, id::Int)
    current = only(index.children[id])
    lambda_count = 0
    while index.nodes[current] isa AbsNode
        lambda_count += 1
        current = only(index.children[current])
    end
    lambda_count > 0 || return false
    return _has_unchanged_self_call(
        index,
        current,
        lambda_count,
        collect(0:(lambda_count - 1)),
    )
end

function _obvious_functiondef_loop_reason(index::SyntaxIndex, id::Int)
    node = index.nodes[id]
    if node isa FunctionDefNode
        reason = _functiondef_obvious_loop_reason(index, id)
        reason === nothing || return reason
    end
    for child_id in index.children[id]
        reason = _obvious_functiondef_loop_reason(index, child_id)
        reason === nothing || return reason
    end
    return nothing
end

"""
    obvious_functiondef_loop_reason(term) -> Union{Nothing,Symbol}

Return a stable reason for the deliberately small set of syntactically certain
FunctionDef loops. `nothing` means only that this cheap common-syntax filter
found no known shape; it is not a termination proof. External AST inputs are
owned/copied; a published `Program` is analyzed through its read-only tree.
"""
function obvious_functiondef_loop_reason(term::Union{AstNode,AstTree})
    tree = term isa AstTree ? term : AstTree(term)
    index = analyze_syntax(tree)
    return _obvious_functiondef_loop_reason(index, 1)
end

function obvious_functiondef_loop_reason(program::Program)
    return _obvious_functiondef_loop_reason(analyze_syntax(program), 1)
end

@testitem "FunctionDef observations separate self references and recursive calls" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.ProgramAnalysis:
        functiondef_recursion_summary, obvious_functiondef_loop_reason

    returned_self = FunctionDefNode(AbsNode(:x, IndexNode(1)))
    recursive = FunctionDefNode(AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))))
    passed_self = FunctionDefNode(AbsNode(:x, AppNode(IndexNode(0), IndexNode(1))))
    nested = FunctionDefNode(AbsNode(:x, FunctionDefNode(
        AbsNode(:y, AppNode(IndexNode(3), IndexNode(0))),
    )))

    @test functiondef_recursion_summary(returned_self).self_references == 1
    @test functiondef_recursion_summary(returned_self).self_calls == 0
    @test functiondef_recursion_summary(recursive).self_calls == 1
    @test functiondef_recursion_summary(passed_self).self_calls == 0
    @test functiondef_recursion_summary(nested).definitions == 2
    @test functiondef_recursion_summary(nested).self_calls == 1
    published_recursive = Program(AstTree(recursive))
    @test functiondef_recursion_summary(published_recursive) ==
          functiondef_recursion_summary(recursive)
    @test obvious_functiondef_loop_reason(FunctionDefNode(IndexNode(0))) === :direct_self
    published_loop = Program(AstTree(FunctionDefNode(IndexNode(0))))
    @test obvious_functiondef_loop_reason(published_loop) === :direct_self
end

@testitem "compact syntax index records FunctionDef binders and intervals" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode
    using LambdaRegression.ProgramAnalysis:
        analyze_syntax, path, visible_binder_ids, captured_binder_ids
    shared = ConstNode(:same)
    index = analyze_syntax(AstTree(FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )))
    @test index.subtree_end[1] == 6
    @test index.owner_binder[1] != index.owner_binder[2]
    @test index.bindings[index.owner_binder[2]].outer == index.owner_binder[1]
    @test length(visible_binder_ids(index, 3)) == 2
    @test index.reference_binder[4] == index.owner_binder[1]
    @test path(index, 5) == [1, 1, 2]
    @test isempty(captured_binder_ids(index, 1))
    @test length(captured_binder_ids(index, 2)) == 1
    @test index.subtree_end[2] == index.subtree_end[1]
    branched = analyze_syntax(AstTree(AppNode(
        FunctionDefNode(AbsNode(:x, IndexNode(0))), ConstNode(:tail),
    )))
    @test branched.subtree_end[2] < branched.subtree_end[1]
    repeated = analyze_syntax(AstTree(AppNode(shared, shared)))
    @test repeated.nodes[2] === repeated.nodes[3]
    @test repeated.parent[2] == 1 && repeated.parent[3] == 1
    named = analyze_syntax(AstTree(AbsNode(:x, AbsNode(:x, VarNode(:x)))))
    @test named.reference_binder[3] == named.owner_binder[2]
    @test named.reference_binder[3] != named.owner_binder[1]
    sibling_names = analyze_syntax(AstTree(AppNode(
        AbsNode(:a, VarNode(:a)), AbsNode(:b, VarNode(:b)),
    )))
    @test sibling_names.reference_binder[5] == sibling_names.owner_binder[4]

    using LambdaRegression.Programs: Program
    source_tree = AstTree(AbsNode(:x, IndexNode(0)))
    published = Program(source_tree)
    borrowed = analyze_syntax(published)
    owned = analyze_syntax(source_tree)
    @test borrowed.tree.root === published.tree.root
    @test owned.tree.root !== source_tree.root
end

@testitem "typed analysis records let-like schemes and isolates targets" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, TyProd, TyScheme, typekey,
        make_tyvar_factory
    a = make_tyvar_factory()(0)
    ctx = Dict{Symbol,TyLike}(
        :PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy,
        :yes => BoolTy,
        :f => TyFun(a, a),
    )
    body = AppNode(
        AppNode(ConstNode(:PAIR), AppNode(IndexNode(0), ConstNode(:one))),
        AppNode(IndexNode(0), ConstNode(:yes)),
    )
    analysis = analyze_program(
        AstTree(AppNode(AbsNode(:id, body), AbsNode(:x, IndexNode(0)))),
        ctx,
        target_ty=TyProd(IntTy, BoolTy),
    )
    @test analysis isa TypedAnalysis
    analysis isa TypedAnalysis || return
    abs_id = only(findall(n -> n isa AbsNode && n.name === :id, analysis.index.nodes))
    @test !haskey(analysis.metadata, abs_id)
    binder = analysis.index.owner_binder[abs_id]
    @test haskey(analysis.binder_schemes, binder)
    scheme = analysis.binder_schemes[binder]
    @test scheme isa TyScheme && length(scheme.vars) == 1
    @test scheme isa TyScheme && scheme.ty isa TyFun &&
        scheme.ty.arg === scheme.ty.res && scheme.ty.arg === only(scheme.vars)
    use_ids = findall(==(binder), analysis.index.reference_binder)
    @test length(use_ids) == 2
    @test typekey(analysis.metadata[use_ids[1]][1]) == typekey(TyFun(IntTy, IntTy))
    @test typekey(analysis.metadata[use_ids[2]][1]) == typekey(TyFun(BoolTy, BoolTy))

    isolated = analyze_program(
        AstTree(ConstNode(:f)), ctx; target_ty=TyFun(IntTy, IntTy),
    )
    @test isolated isa TypedAnalysis
    @test a.instance === nothing
end

@testitem "typed analysis keeps joint target variables isolated" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, make_tyvar_factory
    a = make_tyvar_factory()(0)
    ctx = Dict{Symbol,TyLike}(:f => TyFun(a, IntTy))
    result = analyze_program(
        AstTree(ConstNode(:f)), ctx; target_ty=TyFun(BoolTy, a),
    )
    @test result isa LambdaRegression.Utils.Err
    @test a.instance === nothing
end

@testitem "typed FunctionDef self is monomorphic and function constrained" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err
    recursive = analyze_program(
        AstTree(FunctionDefNode(AbsNode(
            :x, AppNode(IndexNode(1), IndexNode(0)),
        ))),
        Dict{Symbol,TyLike}(); target_ty=TyFun(FloatTy, FloatTy),
    )
    @test recursive isa TypedAnalysis
    recursive isa TypedAnalysis || return
    self_binder = recursive.index.owner_binder[1]
    @test recursive.binder_schemes[self_binder] isa LambdaRegression.TypeSystem.TyScheme
    @test isempty(recursive.binder_schemes[self_binder].vars)
    self_use = findfirst(==(self_binder), recursive.index.reference_binder)
    @test recursive.metadata[self_use][1] == TyFun(FloatTy, FloatTy)
    rejected = analyze_program(
        AstTree(FunctionDefNode(ConstNode(:one))),
        Dict{Symbol,TyLike}(:one => FloatTy),
    )
    @test rejected isa Err
end

end # module ProgramAnalysis
