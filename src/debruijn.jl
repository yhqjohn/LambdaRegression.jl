module DeBruijnModule

import ..ParseModule: AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, LambdaToken, DeBruijnLambdaToken, ApplyToken, ProxyToken, unfold
import LambdaRegression: Err, @alias, alias_dict
using MLStyle
using TestItems

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
# AbsDB(x) = AbsDB{Union{}}(x) # Not needed
# AppDB(x, y) = AppDB{Union{}}(x, y) # Not needed
# AppDB(x::DBNode{T}, y::DBNode{T}) where {T} = AppDB{T}(x, y)
AppDB(x::DBNode{T}, y::DBNode{S}) where {T, S<:T} = AppDB{T}(x, y)
AppDB(x::DBNode{T}, y::DBNode{S}) where {S, T<:S} = AppDB{S}(x, y)


@testitem "creating DeBruijn nodes" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
    v1 = VarDB(0)
    @test VarDB(0) === VarDB{Union{}}(0)
    v2 = VarDB{Int}(1)
    l1 = AbsDB(v2)
    @test l1 === AbsDB{Int}(VarDB{Int}(1))
    app1 = AppDB(l1, v1)
    @test app1 === AppDB{Int}(AbsDB{Int}(VarDB{Int}(1)), VarDB{Int}(0))
    app2 = AppDB(v1, l1)
    @test app2 === AppDB{Int}(VarDB{Int}(0), AbsDB{Int}(VarDB{Int}(1)))
    @test ProxyDB(1) === ProxyDB{Int}(1)
    @test ProxyDB{Int}(1.0) === ProxyDB{Int}(1)
end

Base.convert(::Type{VarDB{T}}, x::VarDB{S}) where {T, S} = VarDB{T}(x.index)
Base.convert(::Type{DBNode{T}}, x::VarDB{S}) where {T, S} = convert(VarDB{T}, x)
Base.convert(::Type{AbsDB{T}}, x::AbsDB{S}) where {T, S} = AbsDB{T}(convert(DBNode{T}, x.body))
Base.convert(::Type{DBNode{T}}, x::AbsDB{S}) where {T, S} = convert(AbsDB{T}, x)
Base.convert(::Type{AppDB{T}}, x::AppDB{S}) where {T, S} = AppDB{T}(convert(DBNode{T}, x.func), convert(DBNode{T}, x.arg))
Base.convert(::Type{DBNode{T}}, x::AppDB{S}) where {T, S} = convert(AppDB{T}, x)
Base.convert(::Type{ProxyDB{T}}, x::ProxyDB{S}) where {T, S} = ProxyDB{T}(convert(T, x.value))
Base.convert(::Type{DBNode{T}}, x::ProxyDB{S}) where {T, S} = convert(ProxyDB{T}, x)

## hash
Base.hash(x::VarDB, h::UInt) = hash(x.index, h)
Base.:(==)(x::VarDB, y::VarDB) = x.index == y.index

Base.hash(x::AbsDB, h::UInt) = hash(x.body, h)
Base.:(==)(x::AbsDB, y::AbsDB) = x.body == y.body

Base.hash(x::AppDB, h::UInt) = hash(x.func, hash(x.arg, h))
Base.:(==)(x::AppDB, y::AppDB) = x.func == y.func && x.arg == y.arg

Base.hash(x::ProxyDB, h::UInt) = hash(x.value, h)
Base.:(==)(x::ProxyDB, y::ProxyDB) = x.value == y.value

@testitem "hash and equality" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
    
    # Test VarDB
    @test hash(VarDB{Int}(1)) == hash(VarDB{Float64}(1))
    @test VarDB{Int}(1) == VarDB{Float64}(1)
    @test VarDB{Int}(1) != VarDB{Int}(2)
    
    # Test AbsDB
    v1 = VarDB{Int}(0)
    v2 = VarDB{Float64}(0)
    @test hash(AbsDB{Int}(v1)) == hash(AbsDB{Float64}(v2))
    @test AbsDB{Int}(v1) == AbsDB{Float64}(v2)
    
    # Test AppDB
    app1 = AppDB{Int}(v1, v1)
    app2 = AppDB{Float64}(v2, v2)
    @test hash(app1) == hash(app2)
    @test app1 == app2
    
    # Test ProxyDB
    @test hash(ProxyDB{Int}(1)) == hash(ProxyDB{Float64}(1.0))
    @test ProxyDB{Int}(1) == ProxyDB{Float64}(1.0)
    @test ProxyDB{Int}(1) != ProxyDB{Int}(2)
end

@testitem "type conversion" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
    @test convert(DBNode{Int}, VarDB(0)) == VarDB{Int}(0) # test convert to DBNode and VarDB
    @test convert(DBNode{Int}, AbsDB(VarDB(1))) == AbsDB{Int}(VarDB{Int}(1)) # test convert to DBNode and AbsDB
    @test convert(DBNode{Int}, AppDB(AbsDB(VarDB(1)), VarDB(0))) == AppDB{Int}(AbsDB{Int}(VarDB{Int}(1)), VarDB{Int}(0)) # test convert to DBNode and AppDB
    @test convert(DBNode{Int}, ProxyDB(1.0)) == ProxyDB{Int}(1) # test convert to DBNode and ProxyDB
end

## value
value(node::ProxyDB) = node.value

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

@testitem "parse" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
    using LambdaRegression.ParseModule
    # test default type parse
    @test DBNode(@lambda x->x+1) == AbsDB{Any}(
        AppDB{Any}(
            AppDB{Any}(
                ProxyDB{Any}(+),
                VarDB{Any}(0)
            ),
            ProxyDB{Any}(1)
        )
    )
    y_combinator = DBNode(@lambda f->(x->f(x*x))*(x->f(x*x)))
    y_combinator2 = AbsDB{Any}(
        AppDB{Any}(
            AbsDB{Any}(
                AppDB{Any}(
                    VarDB{Any}(1),
                    AppDB{Any}(
                        VarDB{Any}(0),
                        VarDB{Any}(0)
                    )
                )
            ),
            AbsDB{Any}(
                AppDB{Any}(
                    VarDB{Any}(1),
                    AppDB{Any}(
                        VarDB{Any}(0),
                        VarDB{Any}(0)
                    )
                )
            ),
        )
    )
    display(y_combinator)
    display(y_combinator2)
    @test y_combinator == y_combinator2

    # test parse with type
    @test DBNode{Int}(@lambda x->x*1) == AbsDB{Int}(
        AppDB{Int}(
            VarDB{Int}(0),
            ProxyDB{Int}(1)
        )
    )
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
pretty_string(node::ProxyDB) = "$(node.value)"
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

@testitem "traverse" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse
    using LambdaRegression.ParseModule
    foo = DBNode(@lambda (*)->(*)->(&3 * &1 * ((*)->&0 * &2)))
    new_foo = traverse(foo) do x, n
        if x isa VarDB
            VarDB(x.index + 1)
        else
            nothing
        end
    end
    @test new_foo == DBNode(@lambda (*)->(*)->(&4 * &2 * ((*)->&1 * &3)))
    new_foo2 = traverse(new_foo) do x, n
        if x isa VarDB
            ProxyDB(Int(x.index))
        else
            nothing
        end
    end
    @test new_foo2 == DBNode(@lambda (*)->(*)->(4 * 2 * ((*)->1 * 3)))
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

@testitem "shift_free" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, shift_free
    using LambdaRegression.ParseModule
    foo = DBNode(@lambda (*)->(*)->(&3 * &1 * ((*)->&0 * &2)))
    @test shift_free(foo, UInt(1)) == DBNode(@lambda (*)->(*)->(&4 * &1 * ((*)->&0 * &2)))
end

@testitem "substitute" begin
    import LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute
    using LambdaRegression.ParseModule
    foo = DBNode(@lambda (*)->(*)->(&3 * &1 * ((*)->&0 * &2)))
    bar = DBNode(@lambda (*)->(&4 * &0))
    @test substitute(foo, bar) == DBNode(@lambda (*)->(&2 * ((*)->(&5 * &0)) * ((*)->&0 * (*)->(&6 * &0))))
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