module DeBruijnModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
using MLStyle

@data DBNode begin
    VarDB(index::UInt)
    AbsDB(body::DBNode)
    AppDB(func::DBNode, arg::DBNode)
    ProxyDB{T}::(value::T)=>DBNode
end

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
            node = @match node begin
                AbsDB(body) =>(push!(stack, Abs); depth += 1; body)
                AppDB(func, arg) =>(push!(stack, Right(arg)); func)
                _ => (result = f(node, depth); node = nothing)
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

substitute(func::AbsDB, arg::DBNode, context=nothing) = traverse(func.body) do node, depth 
    @match node begin # ripped the first lambda, therefore the depth n bounds to variable n, not variable n-1
        VarDB(index) && if index == depth end => shift_free(arg, depth)
        VarDB(index) && if index > depth end => VarDB(index - 1)
        _ => node
    end
end

shift_free(node::DBNode, n::UInt) = traverse(node) do node, depth
    @match node begin
        VarDB(index) && if index >= depth end => VarDB(index + n)
        _ => node
    end
end

# substitution rules for proxy
substitute(::ProxyDB, _, _) = normalized


# show proxy
proxy_alias_dict = Dict{ProxyDB, String}()
set_alias!(p::ProxyDB, alias::String) = proxy_alias_dict[p] = alias

function Base.show(io::IO, ::MIME"text/plain", p::ProxyDB)
    if haskey(proxy_alias_dict, p)
        print(io, proxy_alias_dict[p])
    else
        print(io, "ProxyDB(", p.value, ")")
    end
end

end # module DeBruijnModule