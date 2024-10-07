module DeBruijnModule

using AbstractTrees
import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, LambdaToken, ApplyToken, ProxyToken, unfold


"""
    abstract type AbstractDBNode

Abstract type for De Bruijn index nodes.
"""
abstract type AbstractDBNode<:AbstractNode{Any} end

struct VarDB <: AbstractDBNode
    index::UInt
end

struct AbsDB <: AbstractDBNode
    body::AbstractDBNode
end

struct AppDB <: AbstractDBNode
    func::AbstractDBNode
    arg::AbstractDBNode
end

struct ProxyDB{T} <: AbstractDBNode
    value::T
end

# tree utilities
children(node::VarDB) = []
children(node::AbsDB) = [node.body]
children(node::AppDB) = [node.func, node.arg]
children(node::ProxyDB) = []
AbstractTrees.nodevalue(node::VarDB) = node.index
AbstractTrees.nodevalue(node::AbsDB) = nothing
AbstractTrees.nodevalue(node::AppDB) = nothing
AbstractTrees.nodevalue(node::ProxyDB) = node.value
AbstractTrees.printnode(io::IO, node::VarDB) = print(io, "#$(node.index)")
AbstractTrees.printnode(io::IO, node::AbsDB) = print(io, "λ")
AbstractTrees.printnode(io::IO, node::AppDB) = print(io, "APPLY")

AllDB{T} = Union{VarDB, AbsDB, AppDB, ProxyDB{T}}

# parse
to_debrujin(node::AbstractToken) = _to_debrujin(unfold(node))

function _to_debrujin(node::AbstractToken, stack=AbstractVariableToken[])
    if node isa AbstractVariableToken
        index = findfirst(==(node), stack)
        isnothing(index) && throw(ArgumentError("variable not found in scope: $node"))
        return VarDB(index)
    elseif node isa LambdaToken
        return AbsDB(_to_debrujin(node.body, [node.arg, stack...]))
    elseif node isa ApplyToken
        return AppDB(_to_debrujin(node.func, stack), _to_debrujin(node.arg, stack))
    else
        throw(ArgumentError("node must be a `VariableToken`, `LambdaToken`, or `ApplyToken`, got $(typeof(node))"))
    end
end

# pretty_print
pretty_string(node::VarDB) = "#$(node.index)"
function pretty_string(node::AbsDB)
    body_str = pretty_string(node.body)
    node.body isa AppDB && (body_str = "($body_str)")
    return "λ.$body_str"
end
function pretty_string(node::AppDB)
    func_str = pretty_string(node.func)
    arg_str = pretty_string(node.arg)
    node.arg isa AppDB && (arg_str = "($arg_str)")
    return "$func_str $arg_str"
end
pretty_string(node::ProxyDB) = string(node.value)
Base.display(io::IO, node::AbstractDBNode) = println(io, pretty_string(node))
Base.display(node::AbstractDBNode) = println(pretty_string(node))

end # module DeBruijnModule