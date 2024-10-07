module ReduceModule

using ..DeBruijnModule: AbstractDBNode, VarNode, AbsNode, AppNode, ProxyNode

apply_on(func::AbsNode, arg::AbstractDBNode) = substitute(func.body, func.index, arg)

find_deepest_scope(node::AbsNode) = find_deepest_scope(node.body, VarNode(node.index), node.index-UInt(1), node.index)

function find_deepest_scope(node::AbstractDBNode, target::VarNode, depth::UInt, max_depth::UInt)
    if node === target
        return depth
    elseif node isa AbsNode
        return find_deepest_scope(node.body, target, depth-UInt(1), max_depth)
    elseif node isa AppNode
        depth1 = find_deepest_scope(node.func, target, depth, max_depth)
        depth2 = find_deepest_scope(node.arg, target, depth, max_depth)
        return min(depth1, depth2)
    elseif node isa ProxyNode
        return max_depth
    else
        throw(ArgumentError("node must be a `VarNode`, `AbsNode`, `AppNode`, or `ProxyNode`, got $(typeof(node))"))
    end
end



end # module ReduceModule