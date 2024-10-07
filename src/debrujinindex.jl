module DeBruijnIndexModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, LambdaToken, ApplyToken, ProxyToken, unfold
using LambdaRegression: to_letter
using AbstractTrees

ALPHABETIC = true

"""
    abstract type AbstractDBINode

Abstract type for De Bruijn index nodes.
"""
abstract type AbstractDBINode <: AbstractNode{Any} end
pretty_string(node::AbstractDBINode) = string(node)
Base.display(io::IO, node::AbstractDBINode) = print(io, pretty_string(node), "\n")
Base.display(node::AbstractDBINode) = display(stdout, node)

struct VariableNode <: AbstractDBINode
    index::UInt
end
AbstractTrees.children(node::VariableNode) = []
AbstractTrees.nodevalue(node::VariableNode) = node.index
AbstractTrees.printnode(io::IO, node::VariableNode) = print(io, "#$(node.index)")
function pretty_string(node::VariableNode)
    if ALPHABETIC
        return to_letter(node.index)
    else
        return "#$(node.index)"
    end
end
Base.display(io::IO, node::VariableNode) = print(io, "#$(node.index)")

struct AbstractionNode <: AbstractDBINode
    index::UInt
    body::AbstractDBINode
end
AbstractTrees.children(node::AbstractionNode) = [node.body]
AbstractTrees.nodevalue(node::AbstractionNode) = node.index
AbstractTrees.printnode(io::IO, node::AbstractionNode) = print(io, "λ#$(node.index)")
function pretty_string(node::AbstractionNode)
    args = UInt[]
    while node isa AbstractionNode
        push!(args, node.index)
        node = node.body
    end
    body_str = pretty_string(node)
    if node isa ApplicationNode
        body_str = "($body_str)"
    end
    if ALPHABETIC
        return "λ$(join([to_letter(arg) for arg in args], ",")).$body_str"
    else
        args = ["#$arg" for arg in args]
        return "λ$(join(args, ",")).$body_str"
    end
end

struct ApplicationNode <: AbstractDBINode
    func::AbstractDBINode
    arg::AbstractDBINode
end
AbstractTrees.children(node::ApplicationNode) = [node.func, node.arg]
AbstractTrees.nodevalue(node::ApplicationNode) = nothing
AbstractTrees.printnode(io::IO, node::ApplicationNode) = print(io, "APPLY")
function pretty_string(node::ApplicationNode)
    func_str = pretty_string(node.func)
    arg_str = pretty_string(node.arg)
    if node.arg isa ApplicationNode
        arg_str = "($arg_str)"
    end
    return "$func_str $arg_str"
end

struct ProxyNode{T} <: AbstractDBINode
    body::T
end
AbstractTrees.children(node::ProxyNode) = []
AbstractTrees.nodevalue(node::ProxyNode) = node.body
pretty_string(node::ProxyNode) = string(node.body)
index_string(node::ProxyNode) = string(node.body)

Base.:*(a::AbstractDBINode, b::AbstractDBINode) = ApplicationNode(a, b)
-->(a::AbstractDBINode, b::AbstractDBINode) = AbstractionNode(a, b)

function to_derbujin_index(node::AbstractToken)
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
        return VariableNode(index_dict[node])
    elseif node isa ApplyToken
        return ApplicationNode(_to_debruijn_index(node.func, index_dict), _to_debruijn_index(node.arg, index_dict))
    elseif node isa LambdaToken
        return AbstractionNode(index_dict[node.arg], _to_debruijn_index(node.body, index_dict))
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