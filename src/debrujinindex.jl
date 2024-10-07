module DeBruijnModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, LambdaToken, ApplyToken, ProxyToken, unfold
using LambdaRegression: to_letter
using AbstractTrees

ALPHABETIC = true

"""
    abstract type AbstractDBNode

Abstract type for De Bruijn index nodes.
"""
abstract type AbstractDBNode <: AbstractNode{Any} end
pretty_string(node::AbstractDBNode) = string(node)
Base.display(io::IO, node::AbstractDBNode) = print(io, pretty_string(node), "\n")
Base.display(node::AbstractDBNode) = display(stdout, node)

struct VarNode <: AbstractDBNode
    index::UInt
end
AbstractTrees.children(node::VarNode) = []
AbstractTrees.nodevalue(node::VarNode) = node.index
AbstractTrees.printnode(io::IO, node::VarNode) = print(io, "#$(node.index)")
function pretty_string(node::VarNode)
    if ALPHABETIC
        return to_letter(node.index)
    else
        return "#$(node.index)"
    end
end
Base.display(io::IO, node::VarNode) = print(io, "#$(node.index)")

struct AbsNode <: AbstractDBNode
    index::UInt
    body::AbstractDBNode
end
AbstractTrees.children(node::AbsNode) = [node.body]
AbstractTrees.nodevalue(node::AbsNode) = node.index
AbstractTrees.printnode(io::IO, node::AbsNode) = print(io, "λ#$(node.index)")
function pretty_string(node::AbsNode)
    args = UInt[]
    while node isa AbsNode
        push!(args, node.index)
        node = node.body
    end
    body_str = pretty_string(node)
    if node isa AppNode
        body_str = "($body_str)"
    end
    if ALPHABETIC
        return "λ$(join([to_letter(arg) for arg in args], ",")).$body_str"
    else
        args = ["#$arg" for arg in args]
        return "λ$(join(args, ",")).$body_str"
    end
end

struct AppNode <: AbstractDBNode
    func::AbstractDBNode
    arg::AbstractDBNode
end
AbstractTrees.children(node::AppNode) = [node.func, node.arg]
AbstractTrees.nodevalue(node::AppNode) = nothing
AbstractTrees.printnode(io::IO, node::AppNode) = print(io, "APPLY")
function pretty_string(node::AppNode)
    func_str = pretty_string(node.func)
    arg_str = pretty_string(node.arg)
    if node.arg isa AppNode
        arg_str = "($arg_str)"
    end
    return "$func_str $arg_str"
end

struct ProxyNode{T} <: AbstractDBNode
    body::T
end
AbstractTrees.children(node::ProxyNode) = []
AbstractTrees.nodevalue(node::ProxyNode) = node.body
pretty_string(node::ProxyNode) = string(node.body)
index_string(node::ProxyNode) = string(node.body)

Base.:*(a::AbstractDBNode, b::AbstractDBNode) = AppNode(a, b)
-->(a::AbstractDBNode, b::AbstractDBNode) = AbsNode(a, b)

function to_derbujin(node::AbstractToken)
    node = unfold(node) # unfold the node
    index_dict = Dict{AbstractVariableToken, UInt}()
    for node in AbstractTrees.PreOrderDFS(node)
        if node isa LambdaToken
            index_dict[node.arg] = depth(node)-UInt(1)
        end
    end
    return _to_debruijn_index(node, index_dict)
end

function _to_debruijn_index(node::AbstractToken, index_dict::Dict{AbstractVariableToken, UInt})
    if node isa VariableToken||node isa DeBruijnToken
        return VarNode(index_dict[node])
    elseif node isa ApplyToken
        return AppNode(_to_debruijn_index(node.func, index_dict), _to_debruijn_index(node.arg, index_dict))
    elseif node isa LambdaToken
        return AbsNode(index_dict[node.arg], _to_debruijn_index(node.body, index_dict))
    elseif node isa ProxyToken
        return ProxyNode(node.body)
    else
        throw(ArgumentError("`node` must be a `VariableToken`, `DeBruijnToken`, `ProxyToken`, `ApplyToken`, or `LambdaToken`, got $(typeof(node))"))
    end
end

function depth(node::AbstractToken)
    if node isa VariableToken || node isa DeBruijnToken || node isa ProxyToken
        return UInt(0)
    elseif node isa ApplyToken
        return max(depth(node.func), depth(node.arg))
    elseif node isa LambdaToken
        return depth(node.body) + UInt(1)
    else
        throw(ArgumentError("`node` must be a `VariableToken`, `DeBruijnToken`, `ProxyToken`, `ApplyToken`, or `LambdaToken`, got $(typeof(node))"))
    end
end

set_alphabetic!(tf::Bool) = global ALPHABETIC = tf



end