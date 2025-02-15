"""Describe the lexical role of the one-based child index of a procedural node."""
function child_role(node::ScopeNode, child_index::Int)
    n = length(node.bindings)
    if 1 <= child_index <= n
        return (kind=:binding, ordinal=child_index)
    elseif child_index == n + 1
        return (kind=:body, ordinal=nothing)
    end
    throw(BoundsError(children(node), child_index))
end

function child_role(node::FunctionDefNode, child_index::Int)
    child_index == 1 || throw(BoundsError(children(node), child_index))
    return (kind=:function_body, ordinal=nothing)
end

"""
    visible_binders(node, child_index; path=(), outer=())

Return anonymous binders visible to a child, innermost first. Scope RHS `i` sees
scope binders `i-1 ... 1`; the body sees `n ... 1`. A function body sees its single
self binder before all `outer` binders.
"""
function visible_binders(
    node::ScopeNode,
    child_index::Int;
    path::Tuple{Vararg{Int}}=(),
    outer::Tuple=(),
)
    role = child_role(node, child_index)
    visible_count = role.kind === :binding ? child_index - 1 : length(node.bindings)
    local_binders = ntuple(
        offset -> BinderId(:scope, path, visible_count - offset + 1),
        visible_count,
    )
    return (local_binders..., outer...)
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

