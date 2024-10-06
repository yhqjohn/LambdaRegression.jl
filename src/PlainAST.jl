module PlainASTModule

import ..ParseModule: RawVariableNode, RawLambdaNode, RawApplyNode, AbstractRawAstNode, parse_lambda
using AbstractTrees


abstract type AbstractPlainAstNode <: AbstractNode{Any} end

hex(n) = string(n, base=16, pad=2)
hextail(n, r=4) = hex(n)[end-r:end]

struct PlainAST
    root::AbstractPlainAstNode
end

mutable struct VariableNode <: AbstractPlainAstNode end
AbstractTrees.children(node::VariableNode) = []
AbstractTrees.nodevalue(node::VariableNode) = objectid(node)
AbstractTrees.printnode(io::IO, node::VariableNode) = print(io, "v$(hextail(objectid(node)))")

struct LambdaNode <: AbstractPlainAstNode
    arg::VariableNode
    body::AbstractPlainAstNode
end
AbstractTrees.children(node::LambdaNode) = [node.body]
# AbstractTrees.nodevalue(node::LambdaNode) = objectid(node)
AbstractTrees.printnode(io::IO, node::LambdaNode) = print(io, "λ$(hextail(objectid(node.arg)))")

struct ApplyNode <: AbstractPlainAstNode
    func::AbstractPlainAstNode
    arg::AbstractPlainAstNode
end
AbstractTrees.children(node::ApplyNode) = [node.func, node.arg]
AbstractTrees.nodevalue(node::ApplyNode) = nothing
AbstractTrees.printnode(io::IO, node::ApplyNode) = print(io, "APPLY")



function to_plain_ast(node::AbstractRawAstNode, context=nothing)
    if context === nothing
        allnodes = collect(AbstractTrees.PreOrderDFS(node))
        abstractions = filter(n -> n isa RawLambdaNode, allnodes)
        symbols = reduce(vcat, [abstraction.args for abstraction in abstractions])
        symbols = unique(symbols)
        println(symbols)
        context = Dict([symbol => VariableNode() for symbol in symbols])
        println(context)
    end
    if node isa RawVariableNode
        return context[node.name]
    elseif node isa RawLambdaNode
        if length(node.args)==1
            return LambdaNode(context[node.args[1]], to_plain_ast(node.body, context))
        else
            return LambdaNode(context[node.args[1]], to_plain_ast(RawLambdaNode(node.args[2:end], node.body), context))
        end
    elseif node isa RawApplyNode
        return ApplyNode(to_plain_ast(node.func, context), to_plain_ast(node.arg, context))
    else
        throw(ArgumentError("node must be a RawVariableNode, RawLambdaNode, or RawApplyNode"))
    end
end

macro lambda(expr)
    expr = parse_lambda(expr, Symbol[])
    return :(to_plain_ast($expr))
end
    
end