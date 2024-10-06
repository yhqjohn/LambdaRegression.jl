module DeBrujinLevelModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, ApplyToken, MultiApplyToken, ProxyToken
using AbstractTrees

"""
    abstract type AbstractDBLNode

Abstract type for De Bruijn level nodes.
"""
abstract type AbstractDBLNode <: AbstractToken end
pretty_string(node::AbstractDBLNode) = string(node)
Base.display(io::IO, node::AbstractDBLNode) = print(io, pretty_string(node))

repr_letters = "zyxwvutsrqpnmlkjihgfedcba" # skip o to avoid confusion with 0
function to_letter(i::UInt)
    if i<=24
        repr_letters[i+1]
    else
        "x_$(i-24)"
    end
end

struct DBLVariable <: AbstractDBLNode
    index::UInt
end
AbstractTrees.children(node::DBLVariable) = []
AbstractTrees.nodevalue(node::DBLVariable) = node.index
AbstractTrees.printnode(io::IO, node::DBLVariable) = print(io, "#$(node.index)")
pretty_string(node::DBLVariable) = to_letter(node.index)
Base.display(io::IO, node::DBLVariable) = print(io, "#$(node.index)")


struct DBLAbstraction <: AbstractDBLNode
    index::UInt
    body::AbstractDBLNode
end
AbstractTrees.children(node::DBLAbstraction) = [node.body]
AbstractTrees.nodevalue(node::DBLAbstraction) = node.index
AbstractTrees.printnode(io::IO, node::DBLAbstraction) = print(io, "λ#$(node.index)")
function args_body(node::DBLAbstraction)
    args = UInt[]
    while node isa DBLAbstraction
        push!(args, node.index)
        node = node.body
    end
    return args, node
end
function pretty_string(node::DBLAbstraction)
    args, body = args_body(node)
    "λ$(join([to_letter(arg) for arg in args], "")).$(pretty_string(body))"
end


struct DBLApply <: AbstractDBLNode
    func::AbstractDBLNode
    arg::AbstractDBLNode
end
AbstractTrees.children(node::DBLApply) = [node.func, node.arg]
AbstractTrees.nodevalue(node::DBLApply) = nothing
AbstractTrees.printnode(io::IO, node::DBLApply) = print(io, "APPLY")
function pretty_string(node::DBLApply)
    if node.arg isa DBLApply
        "$(pretty_string(node.func))($(pretty_string(node.arg)))"
    else
        "$(pretty_string(node.func)) $(pretty_string(node.arg))"
    end
end

struct DBLProxy{T} <: AbstractDBLNode
    body::T
end
AbstractTrees.children(node::DBLProxy) = []
AbstractTrees.nodevalue(node::DBLProxy) = node.body
pretty_string(node::DBLProxy) = repr(node.body)


function to_debrujin(node::AbstractToken, level=UInt(0), context=Dict{AbstractVariableToken, UInt}())
    if node isa AbstractVariableToken
        idx = get(context, node, nothing)
        isnothing(idx) && throw(ArgumentError("unbound variable $node"))
        return DBLVariable(idx)
    elseif node isa LambdaToken
        if length(node.args)<1
            throw(ArgumentError("lambda must have at least one argument"))
        elseif length(node.args)==1
            arg = node.args[1]
            new_context = copy(context)
            new_context[arg] = level
            return DBLAbstraction(level, to_debrujin(node.body, level+1, new_context))
        else
            arg = node.args[1]
            new_context = copy(context)
            new_context[arg] = level
            return DBLAbstraction(level, to_debrujin(LambdaToken(node.args[2:end], node.body), level+1, new_context))
        end
    elseif node isa ApplyToken
        return DBLApply(to_debrujin(node.func, level, context), to_debrujin(node.arg, level, context))
    elseif node isa MultiApplyToken
        return foldl(node.terms, init=to_debrujin(node.func, level, context)) do (a,b) 
            DBLApply(a, to_debrujin(b, level, context))
        end
    elseif node isa ProxyToken
        return DBLProxy(to_debrujin(node.body, level, context))
    else
        throw(ArgumentError("node must be a VariableToken, LambdaToken, ApplyToken, or ProxyToken"))
    end
end


end