module DeBruijnModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
import LambdaRegression: Err, @alias, alias_dict
using MLStyle

@data DBNode begin
    VarDB(index::UInt)
    AbsDB(body::DBNode)
    AppDB(func::DBNode, arg::DBNode)
    ProxyDB{T}::(value::T)=>DBNode
end
ProxyDB(value::DBNode) = value

value(node::ProxyDB) = node.value
Base.:(==)(p::ProxyDB, q::ProxyDB) = value(p) == value(q)

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
(n::DBNode)(m) = AppDB(n, ProxyDB(m))
(n::DBNode)(m::DBNode) = AppDB(n, m)

@data Context begin
    Abs
    Right(DBNode)
    NewLeft(DBNode)
end

function traverse(f, node::DBNode)
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
                Abs => (depth -= 1; AbsDB(result))
                Right(value) => begin
                    push!(stack, NewLeft(result))
                    node = value
                    result
                end
                NewLeft(value) => AppDB(value, result)
            end
        end        
    end
    return result
end

struct Normalized end
const normalized = Err(Normalized())

lambda_substitute(func::AbsDB, arg::DBNode) = traverse(func.body) do node, depth 
    @match node begin # ripped the first lambda, therefore the depth n bounds to variable n, not variable n-1
        VarDB(index) && if index == depth end => shift_free(arg, depth)
        VarDB(index) && if index > depth end => VarDB(index - 1)
        _ => nothing
    end
end
substitute(func::AbsDB, arg::DBNode) = lambda_substitute(func, arg)

shift_free(node::DBNode, n::UInt) = traverse(node) do node, depth
    @match node begin
        VarDB(index) && if index >= depth end => VarDB(index + n)
        _ => nothing
    end
end

substitute(func::AbsDB, arg::ProxyDB) = eager_normalize(lambda_substitute(func, arg))

function eager_normalize(node::DBNode)
    @match node begin
        AbsDB(body) => AbsDB(eager_normalize(body))
        AppDB(func, arg) => @match (eager_normalize(func), eager_normalize(arg)) begin
            (new_func::ProxyDB, new_arg::ProxyDB) => substitute(new_func, new_arg)
            (new_func, new_arg) => AppDB(new_func, new_arg)
        end
        _ => node
    end
end





# show specific patterns
@alias AbsDB
@alias AppDB
@alias ProxyDB

end # module DeBruijnModule