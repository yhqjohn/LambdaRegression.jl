module DeBrujinLevelModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, ApplyToken, MultiApplyToken, ProxyToken
using LambdaRegression: to_letter
using AbstractTrees

"""
    abstract type AbstractDBLNode

Abstract type for De Bruijn level nodes.
"""
abstract type AbstractDBLNode <: AbstractNode{Any} end
pretty_string(node::AbstractDBLNode, d) = string(node)
level_string(node::AbstractDBLNode) = string(node)
Base.display(io::IO, node::AbstractDBLNode) = print(io, pretty_string(node, nothing), "\n")
Base.display(node::AbstractDBLNode) = display(stdout, node)
Base.show(io::IO, node::AbstractDBLNode) = print(io, level_string(node))
depth(node::AbstractDBLNode) = UInt(0)


struct DBLVariable <: AbstractDBLNode
    index::UInt
end
AbstractTrees.children(node::DBLVariable) = []
AbstractTrees.nodevalue(node::DBLVariable) = node.index
AbstractTrees.printnode(io::IO, node::DBLVariable) = print(io, "#$(node.index)")
pretty_string(node::DBLVariable, d) = to_letter(d-node.index)
level_string(node::DBLVariable) = "#$(node.index)"
Base.display(io::IO, node::DBLVariable) = print(io, "#$(node.index)")
depth(node::DBLVariable) = UInt(0)


struct DBLAbstraction <: AbstractDBLNode
    index::UInt
    body::AbstractDBLNode
end
AbstractTrees.children(node::DBLAbstraction) = [node.body]
AbstractTrees.nodevalue(node::DBLAbstraction) = node.index
AbstractTrees.printnode(io::IO, node::DBLAbstraction) = print(io, "λ#$(node.index)")
# note that the args are in reverse order
function args_body(node::DBLAbstraction)
    args = UInt[]
    while node isa DBLAbstraction
        push!(args, node.index)
        node = node.body
    end
    return args, node
end
function pretty_string(node::DBLAbstraction, d=nothing)
    if isnothing(d)
        d = depth(node)
    end
    args, body = args_body(node)
    body_str = pretty_string(body, d)
    if body isa DBLApply
        body_str = "($body_str)"
    end
    "λ$(join([to_letter(d-arg) for arg in args], ",")).$body_str"
end
function level_string(node::DBLAbstraction)
    if node.body isa DBLApply
        "λ#$(node.index).($(level_string(node.body)))"
    else
        "λ#$(node.index).$(level_string(node.body))"
    end
end
depth(node::DBLAbstraction) = max(depth(node.body), node.index)


struct DBLApply <: AbstractDBLNode
    func::AbstractDBLNode
    arg::AbstractDBLNode
end
AbstractTrees.children(node::DBLApply) = [node.func, node.arg]
AbstractTrees.nodevalue(node::DBLApply) = nothing
AbstractTrees.printnode(io::IO, node::DBLApply) = print(io, "APPLY")
function pretty_string(node::DBLApply, d=nothing)
    if isnothing(d)
        d = depth(node)
    end
    if node.arg isa DBLApply
        "$(pretty_string(node.func, d))($(pretty_string(node.arg, d)))"
    else
        "$(pretty_string(node.func, d)) $(pretty_string(node.arg, d))"
    end
end
function level_string(node::DBLApply)
    if node.arg isa DBLApply
        "$(level_string(node.func))($(level_string(node.arg)))"
    else
        "$(level_string(node.func)) $(level_string(node.arg))"
    end
end
depth(node::DBLApply) = max(depth(node.func), depth(node.arg))

struct DBLProxy{T} <: AbstractDBLNode
    body::T
end
AbstractTrees.children(node::DBLProxy) = []
AbstractTrees.nodevalue(node::DBLProxy) = node.body
pretty_string(node::DBLProxy, d) = repr(node.body)
level_string(node::DBLProxy) = level_string(node.body)
depth(node::DBLProxy) = 0


function to_debrujin_level(node::AbstractToken, level=UInt(0), context=Dict{AbstractVariableToken, UInt}())
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
            return DBLAbstraction(level, to_debrujin_level(node.body, level+1, new_context))
        else
            arg = node.args[1]
            new_context = copy(context)
            new_context[arg] = level
            return DBLAbstraction(level, to_debrujin_level(LambdaToken(node.args[2:end], node.body), level+1, new_context))
        end
    elseif node isa ApplyToken
        return DBLApply(to_debrujin_level(node.func, level, context), to_debrujin_level(node.arg, level, context))
    elseif node isa MultiApplyToken
        return foldl(node.terms, init=to_debrujin_level(node.func, level, context)) do (a,b) 
            DBLApply(a, to_debrujin_level(b, level, context))
        end
    elseif node isa ProxyToken
        return DBLProxy(to_debrujin_level(node.body, level, context))
    else
        throw(ArgumentError("node must be a VariableToken, LambdaToken, ApplyToken, or ProxyToken"))
    end
end


end