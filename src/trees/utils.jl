# directly as included in Trees.jl
"""
    lca(query, tree::AbstractTree, subtree_index)

Find the Lowest Common Ancestor of the nodes where `query(tree, node)` is true in the subtree of index `subtree_index`.
Return the index of the Lowest Common Ancestor, or `nothing` if no such node exists.
"""
function lca(query, tree::T, subtree_index) where T<:AbstractTree
    if query(tree, subtree_index)
        return subtree_index
    end
    I = NodeIndex(T)
    lcas = Union{Nothing, I}[lca(query, tree, child) for child in children(tree, subtree_index)]
    lca_count = count(x->!isnothing(x), lcas)
    if lca_count == 0
        return nothing
    elseif lca_count == 1
        return lcas[findfirst(x->!isnothing(x), lcas)]
    else
        return subtree_index
    end
end

function Base.setindex!(tree::T, new_subtree, index) where T<:AbstractTree
    I = NodeIndex(T)
    new_tree_break::T, _ = traverse(tree, PointerState{I}) do tree, node, bag
        if node == index
            (;parent, child_index) = getstate(bag, PointerState{I})
            new_tree = setchild!(tree, parent, child_index, new_subtree)
            Break(new_tree)
        end
    end
    return new_tree_break
end


