module LabeledProxyModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
using ..DeBruijnModule
using LambdaRegression: Err

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


struct FunctionTerm
    func
    label::LabelPair
end
termalias = Dict{Union{FunctionTerm}, String}()
setalias!(term::FunctionTerm, alias::String) = termalias[term] = alias
Base.show(io::IO, ::MIME"text/plain", term::FunctionTerm) = print(io, get(termalias, term, string(term)))
FunctionTerm(func, label, alias::String) = (term = FunctionTerm(func, label); setalias!(term, alias); term)
(f::FunctionTerm)(x) = f.func(x)

struct TypeMismatch end
const type_mismatch_err = Err(TypeMismatch())

function DeBruijnModule.substitute(a::ProxyDB{S}, b::ProxyDB{T}) where {S, T}
    if isapplicable(S, T)
        return ProxyDB(a.value(b.value))
    else
        return type_mismatch_err
    end
end

end