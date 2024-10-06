module DeBruijnIndexModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, SingleLambdaToken, LambdaToken, ApplyToken, MultiApplyToken, ProxyToken, unfold
using LambdaRegression: to_letter
using AbstractTrees

"""
    abstract type AbstractDBINode

Abstract type for De Bruijn index nodes.
"""
abstract type AbstractDBINode <: AbstractNode{Any} end
pretty_string(node::AbstractDBINode) = string(node)
index_string(node::AbstractDBINode) = string(node)
Base.display(io::IO, node::AbstractDBINode) = print(io, pretty_string(node), "\n")
Base.display(node::AbstractDBINode) = display(stdout, node)
Base.show(io::IO, node::AbstractDBINode) = print(io, index_string(node))

struct VariableNode <: AbstractDBINode
    index::UInt
end
AbstractTrees.children(node::VariableNode) = []
AbstractTrees.nodevalue(node::VariableNode) = node.index
AbstractTrees.printnode(io::IO, node::VariableNode) = print(io, "#$(node.index)")
pretty_string(node::VariableNode) = to_letter(node.index)
index_string(node::VariableNode) = "#$(node.index)"
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
    return "λ$(join([to_letter(arg) for arg in args], ",")).$body_str"
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

struct DBIProxy{T} <: AbstractDBINode
    body::T
end
AbstractTrees.children(node::DBIProxy) = []
AbstractTrees.nodevalue(node::DBIProxy) = node.body
pretty_string(node::DBIProxy) = string(node.body)
index_string(node::DBIProxy) = string(node.body)

Base.:*(a::AbstractDBINode, b::AbstractDBINode) = ApplicationNode(a, b)
-->(a::AbstractDBINode, b::AbstractDBINode) = AbstractionNode(a, b)


end