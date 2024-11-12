module DeBruijnModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
import LambdaRegression: Err, @alias, alias_dict
using MLStyle

########################################################
# DeBruijn
########################################################

## data structure
abstract type DBNode{T} end

struct VarDB{T} <: DBNode{T}
    index::UInt
end

struct AbsDB{T} <: DBNode{T}
    body::DBNode{T}
end

struct AppDB{T} <: DBNode{T}
    func::DBNode{T}
    arg::DBNode{T}
end

struct ProxyDB{T} <: DBNode{T}
    value::T
    ProxyDB{T}(x) where {T} = new{T}(convert(T, x))
    ProxyDB(x::S) where {S} = new{S}(x)
end

@as_record VarDB
@as_record AbsDB
@as_record AppDB
@as_record ProxyDB

# type conversion
VarDB(x) = VarDB{Union{}}(x)
AbsDB(x) = AbsDB{Union{}}(x)
AppDB(x, y) = AppDB{Union{}}(x, y)

Base.convert(::Type{VarDB{T}}, x::VarDB{S}) where {T, S} = VarDB{T}(x.index)
Base.convert(::Type{DBNode{T}}, x::VarDB{S}) where {T, S} = convert(VarDB{T}, x)
Base.convert(::Type{AbsDB{T}}, x::AbsDB{S}) where {T, S} = AbsDB{T}(convert(DBNode{T}, x.body))
Base.convert(::Type{DBNode{T}}, x::AbsDB{S}) where {T, S} = convert(AbsDB{T}, x)
Base.convert(::Type{AppDB{T}}, x::AppDB{S}) where {T, S} = AppDB{T}(convert(DBNode{T}, x.func), convert(DBNode{T}, x.arg))
Base.convert(::Type{DBNode{T}}, x::AppDB{S}) where {T, S} = convert(AppDB{T}, x)
Base.convert(::Type{ProxyDB{T}}, x::ProxyDB{S}) where {T, S} = ProxyDB{T}(convert(T, x.value))
Base.convert(::Type{DBNode{T}}, x::ProxyDB{S}) where {T, S} = convert(ProxyDB{T}, x)

## value
value(node::ProxyDB) = node.value
Base.:(==)(p::ProxyDB, q::ProxyDB) = value(p) == value(q)

## parse
DBNode(x) = DBNode{Any}(x)
DBNode{T}(node::AbstractToken) where {T} = _to_debrujin(unfold(node), T)

function _to_debrujin(node::AbstractToken, T, stack=AbstractVariableToken[])
    if node isa VariableToken
        index = findfirst(==(node), stack)
        isnothing(index) && throw(ArgumentError("variable not found in scope: $node"))
        return VarDB{T}(index-1)
    elseif node isa DeBruijnToken
        return VarDB{T}(node.index)
    elseif node isa LambdaToken
        return AbsDB{T}(_to_debrujin(node.body, T, [node.arg, stack...]))
    elseif node isa DeBruijnLambdaToken
        return AbsDB{T}(_to_debrujin(node.body, T, stack))
    elseif node isa ApplyToken
        return AppDB{T}(_to_debrujin(node.func, T, stack), _to_debrujin(node.arg, T, stack))
    elseif node isa ProxyToken
        return ProxyDB{T}(node.body)
    else
        throw(ArgumentError("node must be a `VariableToken`, `LambdaToken`, or `ApplyToken`, got $(typeof(node))"))
    end
end

# pretty_print
pretty_string(node::VarDB) = "#$(node.index)"
function pretty_string(node::AbsDB)
    if haskey(alias_dict, node)
        return alias_dict[node]
    end

    body_str = pretty_string(node.body)
    node.body isa AppDB && (body_str = "($body_str)")
    return "λ.$body_str"
end
function pretty_string(node::AppDB)
    if haskey(alias_dict, node)
        return alias_dict[node]
    end
    func_str = pretty_string(node.func)
    arg_str = pretty_string(node.arg)
    node.arg isa AppDB && (arg_str = "($arg_str)")
    return "$func_str $arg_str"
end
pretty_string(node::ProxyDB) = string(value(node))
Base.display(io::IO, node::DBNode) = println(io, pretty_string(node))
Base.display(node::DBNode) = println(pretty_string(node))

# apply
(n::DBNode{T})(m::T) where {T} = AppDB(n, ProxyDB(m))
(n::DBNode{T})(m::DBNode{T}) where {T} = AppDB(n, m)

@data Context begin
    Abs
    Right(DBNode)
    NewLeft(DBNode)
end

function traverse(f, node::DBNode{T}) where {T}
    stack = Context[]
    depth = UInt(0)
    result = nothing
    node = node
    while !isnothing(node) || !isempty(stack)
        if !isnothing(node)
            result = f(node, depth)
            if isnothing(result)
                node = @match node begin
                    AbsDB(body) =>(push!(stack, Abs); depth += 1; body)
                    AppDB(func, arg) =>(push!(stack, Right(arg)); func)
                    _ => (result= node; node = nothing)
                end
            else
                node = nothing
            end
        else
            result = @match pop!(stack) begin
                Abs => (depth -= 1; AbsDB{T}(result))
                Right(value) => begin
                    push!(stack, NewLeft(result))
                    node = value
                    result
                end
                NewLeft(value) => AppDB{T}(value, result)
            end
        end        
    end
    return result
end

struct Normalized end
const normalized = Err(Normalized())

lambda_substitute(func::AbsDB{T}, arg::DBNode{T}) where {T} = traverse(func.body) do node, depth 
    @match node begin # ripped the first lambda, therefore the depth n bounds to variable n, not variable n-1
        VarDB(index) && if index == depth end => shift_free(arg, depth)
        VarDB(index) && if index > depth end => VarDB{T}(index - 1)
        _ => nothing
    end
end
substitute(func::AbsDB{T}, arg::DBNode{T}) where {T} = lambda_substitute(func, arg)
substitute(func::DBNode{T}, arg::DBNode{T}) where {T} = normalized

shift_free(node::DBNode{T}, n::UInt) where {T} = traverse(node) do node, depth
    @match node begin
        VarDB(index) && if index >= depth end => VarDB{T}(index + n)
        _ => nothing
    end
end

# substitute(func::AbsDB, arg::ProxyDB) = eager_normalize(lambda_substitute(func, arg))

# function eager_normalize(node::DBNode)
#     @match node begin
#         AbsDB(body) => AbsDB(eager_normalize(body))
#         AppDB(func, arg) => @match (eager_normalize(func), eager_normalize(arg)) begin
#             (new_func::ProxyDB, new_arg::ProxyDB) => substitute(new_func, new_arg)
#             (new_func, new_arg) => AppDB(new_func, new_arg)
#         end
#         _ => node
#     end
# end



# show specific patterns
@alias AbsDB
@alias AppDB
@alias ProxyDB


export DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse
export shift_free, substitute, Normalized, normalized

end # module DeBruijnModule