"""
    LexicalContexts

Child-specific lexical context for AST traversals. Composite nodes describe the
role of each child and the binders visible while visiting that child. Binder
identities are anonymous and structural, so consumers do not need display names
or knowledge of a node's concrete storage layout.

Extension nodes that introduce binders should add methods for [`child_role`](@ref)
and [`visible_binders`](@ref). Returned binders are ordered innermost first, matching
De Bruijn indices.
"""
module LexicalContexts

using TestItems

import ...Trees: children
using ..NodeASTs:
    AstNode, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode

export BinderId, child_role, visible_binders

"""
    BinderId(kind, owner_path, ordinal)

Stable anonymous identity for a lexical binder. `owner_path` is the path of the
owning node in its containing tree and `ordinal` is one-based within that node.
"""
struct BinderId
    kind::Symbol
    owner_path::Tuple{Vararg{Int}}
    ordinal::Int

    function BinderId(kind::Symbol, owner_path::Tuple{Vararg{Int}}, ordinal::Int)
        kind in (:abs, :function_def) ||
            throw(ArgumentError("unknown lexical binder kind: $kind"))
        all(>(0), owner_path) ||
            throw(ArgumentError("owner paths use positive child indices"))
        ordinal > 0 || throw(ArgumentError("binder ordinal must be positive"))
        return new(kind, owner_path, ordinal)
    end
end

_child_bounds_error(node::AstNode, child_index::Int) =
    throw(BoundsError(children(node), child_index))

"""Describe the lexical role of a composite node's one-based child index."""
function child_role(node::AppNode, child_index::Int)
    child_index == 1 && return (kind=:function, ordinal=nothing)
    child_index == 2 && return (kind=:argument, ordinal=nothing)
    return _child_bounds_error(node, child_index)
end

function child_role(node::AbsNode, child_index::Int)
    child_index == 1 || return _child_bounds_error(node, child_index)
    return (kind=:body, ordinal=nothing)
end

function child_role(node::FunctionDefNode, child_index::Int)
    child_index == 1 || return _child_bounds_error(node, child_index)
    return (kind=:function_body, ordinal=nothing)
end

child_role(node::Union{ConstNode,IndexNode,VarNode}, child_index::Int) =
    _child_bounds_error(node, child_index)

"""
    visible_binders(node, child_index; path=(), outer=())

Return the binders visible to one child, innermost first. `path` identifies the
owning node and `outer` contains the already-visible binders of its parent context.
"""
function visible_binders(
    node::AppNode,
    child_index::Int;
    path::Tuple{Vararg{Int}}=(),
    outer::Tuple=(),
)
    child_role(node, child_index)
    return outer
end

function visible_binders(
    node::AbsNode,
    child_index::Int;
    path::Tuple{Vararg{Int}}=(),
    outer::Tuple=(),
)
    child_role(node, child_index)
    return (BinderId(:abs, path, 1), outer...)
end

function visible_binders(
    node::FunctionDefNode,
    child_index::Int;
    path::Tuple{Vararg{Int}}=(),
    outer::Tuple=(),
)
    child_role(node, child_index)
    return (BinderId(:function_def, path, 1), outer...)
end

@testitem "Core AST nodes expose child-specific lexical contexts" begin
    using Test
    using LambdaRegression.ASTs.LexicalContexts
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode

    outer = (BinderId(:abs, (7,), 1),)
    app = AppNode(ConstNode(:f), ConstNode(:x))
    abstraction = AbsNode(:x, IndexNode(0))
    function_def = FunctionDefNode(IndexNode(0))

    @test child_role(app, 1) == (kind=:function, ordinal=nothing)
    @test child_role(app, 2) == (kind=:argument, ordinal=nothing)
    @test visible_binders(app, 2; path=(1,), outer) === outer
    @test child_role(abstraction, 1) == (kind=:body, ordinal=nothing)
    @test visible_binders(abstraction, 1; path=(2,), outer) ==
        (BinderId(:abs, (2,), 1), outer...)
    @test child_role(function_def, 1) == (kind=:function_body, ordinal=nothing)
    @test visible_binders(function_def, 1; path=(4,), outer) ==
        (BinderId(:function_def, (4,), 1), outer...)
    @test_throws BoundsError child_role(app, 3)
    @test_throws BoundsError child_role(ConstNode(:leaf), 1)
    @test_throws ArgumentError BinderId(:scope, (), 1)
    @test_throws ArgumentError BinderId(:unknown, (), 1)
end

end # module LexicalContexts
