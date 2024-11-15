module TermsModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
using MLStyle
import LambdaRegression: Err, Result
import ..DeBruijnModule: substitute
using TestItems
########################################
# Type system for terms
########################################
@data TypeLabel begin
    LabelType(label::Symbol)
    ArrowType(arg::TypeLabel, ret::TypeLabel)
    UnresolvedType()
    UnknownType()
    AnyType()
end
Base.:<=(::TypeLabel, ::TypeLabel) = false
Base.:<=(a::LabelType, b::LabelType) = a===b
Base.:<=(::UnknownType, ::TypeLabel) = false
Base.:<=(::UnresolvedType, ::TypeLabel) = false
Base.:<=(::TypeLabel, ::AnyType) = true
Base.:<=(::UnknownType, ::AnyType) = true
Base.:<=(::UnresolvedType, ::AnyType) = true

argtype(x::ArrowType) = x.arg
rettype(x::ArrowType) = x.ret

-->(a::TypeLabel, b::TypeLabel) = ArrowType(a, b)
-->(a::Symbol, b::TypeLabel) = LabelType(a) --> b
-->(a::TypeLabel, b::Symbol) = a --> LabelType(b)
-->(a::Symbol, b::Symbol) = LabelType(a) --> LabelType(b)

Base.convert(::Type{LabelType}, x::Symbol) = LabelType(x)
Base.convert(::Type{TypeLabel}, x::Symbol) = LabelType(x)

struct TypeMismatch
    expected::TypeLabel
    actual::TypeLabel
end

@testitem "type system" begin
    import LambdaRegression.TermsModule: TypeLabel, LabelType, ArrowType, UnknownType, UnresolvedType, AnyType, -->, argtype, rettype
    @test LabelType(:Int) <= LabelType(:Int)
    @test !(LabelType(:Int) <= LabelType(:Float64))
    @test !(UnknownType() <= UnknownType())
    @test !(UnresolvedType() <= UnknownType())
    @test UnknownType() <= AnyType()
    @test UnresolvedType() <= AnyType()
    @test (:Int --> :Int) === ArrowType(LabelType(:Int), LabelType(:Int))
    @test (:Int --> :Int) === (LabelType(:Int) --> LabelType(:Int))
    @test (:Int --> :Int) === (LabelType(:Int) --> :Int)
    @test (:Int --> :Int) === (:Int --> LabelType(:Int))
    @test argtype(:Int --> :Int --> :Bool) === LabelType(:Int)
    @test rettype(:Int --> :Int --> :Bool) === (:Int --> :Bool)
end


########################################
# Terms
########################################
abstract type AbstractTerm end

## untyped term
struct Term{T} <: AbstractTerm
    value::T
end
value(term::Term) = term.value
TermProxy{T} = ProxyDB{Term{T}}
TermDB{T} = DBNode{Term{T}}

### coversion
Base.convert(::Type{Term{T}}, x::Term{S}) where {T, S<:T} = Term{T}(x.value)
Base.convert(::Type{Term{T}}, x::S) where {T, S<:T} = Term{T}(x)

@testitem "untyped term and conversion" begin
    import LambdaRegression.TermsModule: Term, TermProxy, TermDB, value, ProxyDB
    term1 = Term(1)
    proxy1 = ProxyDB(term1)
    @test proxy1 isa TermProxy{Int}
    @test proxy1 isa TermDB{Int}
    @test value(term1) === 1
    realterm1 = convert(Term{Real}, term1)
    realterm2 = convert(Term{Real}, 1)
    @test realterm1 === Term{Real}(1)
    @test realterm2 === Term{Real}(1)
end

unpack_to_value(x) = x
unpack_to_value(x::ProxyDB) = unpack_to_value(x.value)
unpack_to_value(x::Term) = x.value

pack_to_termnode(n::DBNode) = n
pack_to_termnode(n::Term) = ProxyDB(n)
pack_to_termnode(x) = ProxyDB(Term(x))

@testitem "pack and unpack" begin
    import LambdaRegression.TermsModule: Term, TermProxy, TermDB, value, ProxyDB, unpack_to_value, pack_to_termnode
    term1 = Term(1)
    proxy1 = ProxyDB(term1)
    @test unpack_to_value(proxy1) === 1
    @test unpack_to_value(term1) === 1
    @test pack_to_termnode(proxy1) === proxy1
    @test pack_to_termnode(term1) === proxy1
    @test pack_to_termnode(1) === proxy1
end

function substitute(func::TermProxy{T}, arg::TermDB{T})::TermDB{T} where {T}
    func = unpack_to_value(func)
    arg = unpack_to_value(arg)
    result = func(arg)
    return pack_to_termnode(result)
end

@testitem "substitute untyped term" begin
    import LambdaRegression.TermsModule: Term, TermProxy, TermDB, value, ProxyDB, substitute
    termincrement = Term{Any}(x -> x + 1)
    proxyincrement = ProxyDB(termincrement)
    term1 = Term{Any}(1)
    proxy1 = ProxyDB(term1)
    @test substitute(proxyincrement, proxy1) === ProxyDB(Term{Any}(2))
end

## typed terms
struct TypedTerm{T} <: AbstractTerm
    value::T
    type::TypeLabel
end
TypedTerm(x, type::Symbol) = TypedTerm(x, LabelType(type))

TypedTermProxy{T} = ProxyDB{TypedTerm{T}}
TypedTermDB{T} = DBNode{TypedTerm{T}}

### conversion
Base.convert(::Type{TypedTerm{T}}, x::TypedTerm{S}) where {T, S<:T} = TypedTerm{T}(x.value, x.type)
Base.convert(::Type{TypedTerm{T}}, x::S) where {T, S<:T} = TypedTerm{T}(x, typelabel(x))

@testitem "typed term and conversion" begin
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, value, ProxyDB, LabelType
    term1 = TypedTerm(1, :Int)
    proxy1 = ProxyDB(term1)
    term2 = convert(TypedTerm{Real}, term1)
    term3 = convert(TypedTerm{Real}, 1)
    @test term2 === TypedTerm{Real}(1, :Int)
    @test term3 === TypedTerm{Real}(1, :Int)
end

unpack_to_value(x::TypedTerm) = x.value

typelabel(n::TypedTermDB) = typelabel(n.value)
typelabel(x::TypedTerm) = x.type
typelabel(::Int) = LabelType(:Int)
typelabel(::Float64) = LabelType(:Float64)
typelabel(::Bool) = LabelType(:Bool)
typelabel(_) = UnknownType()

pack_to_typedtermnode(n::DBNode) = n
pack_to_typedtermnode(n::TypedTerm) = ProxyDB(n)
pack_to_typedtermnode(x) = ProxyDB(TypedTerm(x, typelabel(x)))

@testitem "pack and unpack typed term" begin
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, value, ProxyDB, typelabel, pack_to_typedtermnode, unpack_to_value, LabelType
    term1 = TypedTerm{Int}(1, :Int)
    proxy1 = ProxyDB(term1)
    @test unpack_to_value(proxy1) === 1
    @test unpack_to_value(term1) === 1
    @test pack_to_typedtermnode(proxy1) === proxy1
    @test pack_to_typedtermnode(term1) === proxy1
    @test pack_to_typedtermnode(1) === proxy1
end

# TypedTermDBResult{T} = Union{TypedTermDB{T}, Err{TypeMismatch}}
# Base.convert(::Type{TypedTermDBResult{T}}, x) where {T} = convert(TypedTermDB{T}, x)
TypedTermDBResult{T} = Result{TypedTermDB{T}, TypeMismatch}

function substitute(func::TypedTermProxy{T}, arg::TypedTermDB{T})::TypedTermDBResult{T} where {T}
    if !(typelabel(func) isa ArrowType)
        throw(ArgumentError("Cannot substitute a non-function term"))
    end
    param_type = argtype(typelabel(func))
    ret_type = rettype(typelabel(func))
    arg_type = typelabel(arg)
    if arg_type <= param_type
        func = unpack_to_value(func)
        arg = unpack_to_value(arg)
        result = func(arg)
        return pack_to_typedtermnode(TypedTerm(result, ret_type))
    else
        return Err(TypeMismatch(param_type, arg_type))
    end
end

@testitem "substitute typed term" begin
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, value, ProxyDB, substitute, LabelType, -->, Err, TypeMismatch
    term1 = TypedTerm{Any}(x -> x + 1, :Int --> :Int)
    proxy1 = ProxyDB(term1)
    term2 = TypedTerm{Any}(1, LabelType(:Int))
    proxy2 = ProxyDB(term2)
    @test substitute(proxy1, proxy2) === ProxyDB(TypedTerm{Any}(2, LabelType(:Int)))
    # catch ArgumentError
    @test_throws ArgumentError substitute(proxy2, proxy2)
    @test substitute(proxy1, proxy1) === Err(TypeMismatch(LabelType(:Int), :Int --> :Int))
end


end