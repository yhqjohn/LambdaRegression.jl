module DeBruijnModule

using AbstractTrees
import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
import LambdaRegression: U1, U0


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
    elseif node isa ProxyToken
        return ProxyDB(node.body)
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
struct _Abs end
struct _Right
    value:: AbstractDBNode
end
struct _NewLeft
    value:: AbstractDBNode
end
_Context = Union{_Abs, _Right, _NewLeft}

function traverse(f, node::AbstractDBNode)
    stack = _Context[]
    depth = UInt(0)
    result = nothing
    node = node
    while !isnothing(node) || !isempty(stack)
        if !isnothing(node)
            if node isa AbsDB
                push!(stack, _Abs())
                node = node.body
                depth += 1
            elseif node isa AppDB
                push!(stack, _Right(node.arg))
                node = node.func
            else # leaf node
                result = f(node, depth)
                node = nothing
            end
        else
            context = pop!(stack)
            if context isa _Abs
                result = AbsDB(result)
                depth -= 1
            elseif context isa _Right
                push!(stack, _NewLeft(result))
                node = context.value
            elseif context isa _NewLeft
                result = AppDB(context.value, result)
            end
        end
    end
    return result
end

function substitute(func::AbsDB, arg::AbstractDBNode)
    traverse(func.body) do node, depth # ripped the first lambda, therefore the depth n bounds to variable n, not variable n-1
        if node isa VarDB
            if node.index == depth
                shift_free(arg, depth)
            elseif node.index > depth # shift down free variables
                VarDB(node.index - 1)
            else
                node
            end
        else
            node
        end
    end
end

shift_free(node::AbstractDBNode, n::UInt) = traverse(node) do node, depth
    if node isa VarDB && node.index >= depth
        VarDB(node.index + n)
    else
        node
    end
end


end # module DeBruijnModule