module DeBruijnModule

using AbstractTrees
import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
import LambdaRegression: U1


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
    if node isa VariableToken
        index = findfirst(==(node), stack)
        isnothing(index) && throw(ArgumentError("variable not found in scope: $node"))
        return VarDB(index-1)
    elseif node isa DeBruijnToken
        return VarDB(node.index)
    elseif node isa LambdaToken
        return AbsDB(_to_debrujin(node.body, [node.arg, stack...]))
    elseif node isa DeBruijnLambdaToken
        return AbsDB(_to_debrujin(node.body, stack))
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


# substitution

function substitute(term::AbstractDBNode, newterm::AbstractDBNode, scope::UInt=0)
    if term isa VarDB
        if term.index < scope # bound variable
            return term
        elseif term.index == scope # substitution instance
            newterm = shift(newterm, scope) # shift up
            return newterm
        else # free variable
            return VarDB(term.index-U1) # shift down 
        end
    elseif term isa AbsDB
        return AbsDB(substitute(term.body, newterm, scope+U1))
    elseif term isa AppDB
        return AppDB(substitute(term.func, newterm, scope), substitute(term.arg, newterm, scope))
    elseif term isa ProxyDB
        return term
    else
        throw(ArgumentError("term must be a `VarDB`, `AbsDB`, `AppDB`, or `ProxyDB`, got $(typeof(term))"))
    end
end

function shift(term::AbstractDBNode, offset::UInt, scope::UInt=0)
    if term isa VarDB
        if term.index < scope # bound variabl, e.g. 0 in λ.0 is bound where scope=1, while 0 is free in 0 where scope=0.
            return term
        else # free variable, 
            return VarDB(term.index+offset)
        end
    elseif term isa AbsDB
        return AbsDB(shift(term.body, offset, scope+U1))
    elseif term isa AppDB
        return AppDB(shift(term.func, offset, scope), shift(term.arg, offset, scope))
    elseif term isa ProxyDB
        return term
    else
        throw(ArgumentError("term must be a `VarDB`, `AbsDB`, `AppDB`, or `ProxyDB`, got $(typeof(term))"))
    end
end


end # module DeBruijnModule