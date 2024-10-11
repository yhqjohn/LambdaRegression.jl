module BuiltInTermsModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
using ..DeBruijnModule

typelabel(::ProxyDB{T}) where T = typelabel(T)
typelabel(::Int) = :Int
typelabel(::Float64) = :Float64
typelabel(::Bool) = :Bool
typelabel(_) = :Unknown

struct LabelPair
    a::Union{Symbol, LabelPair}
    b::Union{Symbol, LabelPair}
end

Base.:>>(a::Symbol, b::Symbol) = LabelPair(a, b)
Base.:>>(a::Symbol, b::LabelPair) = LabelPair(a, b)
Base.:>>(a::LabelPair, b::Symbol) = LabelPair(a, b)
Base.show(io::IO, pair::LabelPair) = print(io, pair.a, " -> ", pair.b isa LabelPair ? "($(pair.b))" : pair.b)

isapplicable(a, b) = false
isapplicable(a::ProxyDB, b::ProxyDB) = isapplicable(typelabel(a), typelabel(b))
isapplicable(a::LabelPair, b::Union{Symbol, LabelPair}) = a.a == b

termalias = Dict{Union{FunctionTerm}, String}()
setalias!(term::FunctionTerm, alias::String) = termalias[term] = alias

struct FunctionTerm
    func
    label::LabelPair
end
Base.show(io::IO, ::MIME"text/plain", term::FunctionTerm) = print(io, get(termalias, term, string(term)))
FunctionTerm(func, label, alias::String) = (term = FunctionTerm(func, label); setalias!(term, alias); term)

const SIN = FunctionTerm(sin, :Float64 >> :Float64, "sin")
const COS = FunctionTerm(cos, :Float64 >> :Float64, "cos")
const TAN = FunctionTerm(tan, :Float64 >> :Float64, "tan")
const ADD = FunctionTerm(+, :Float64 >> :Float64 >> :Float64, "+")
const SUB = FunctionTerm(-, :Float64 >> :Float64 >> :Float64, "-")
const MUL = FunctionTerm(*, :Float64 >> :Float64 >> :Float64, "*")
const DIV = FunctionTerm(/, :Float64 >> :Float64 >> :Float64, "/")
const EQ = FunctionTerm(==, :Float64 >> :Float64 >> :Bool, "==")

DebruijnModule.substitute(::ProxyDB{S}, ::ProxyDB{T}) where {S, T} = ProxyDB{S}(T)









    
end