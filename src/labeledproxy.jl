module LabeledProxyModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, proxy_alias_dict, normalized
using ..DeBruijnModule
using LambdaRegression: Err
using ..ReduceModule:beta_reduce
using MLStyle

abstract type AbstractTypeLabel end

@data SimpleTypeLabel begin
    LabelType(label::Symbol)
    FunctionType(arg::SimpleType, ret::SimpleType)
    FunctionalType(arg::SimpleType, func::Any)
    UnknownType()
end
Base.:>>(a::SimpleTypeLabel, b::SimpleTypeLabel) = FunctionType(a, b)
Base.:>>(a::Symbol, b::Symbol) = FunctionType(LabelType(a), LabelType(b))
Base.:>>(a::SimpleTypeLabel, b::Symbol) = FunctionalType(a, LabelType(b))
Base.:>>(a::Symbol, b::SimpleTypeLabel) = FunctionalType(LabelType(a), b)
argtype(f::FunctionalType) = f.arg
argtype(f::FunctionType) = f.arg
argtype(::LabelType) = nothing
argtype(::UnknownType) = UnknownType()
isapplicable(a::SimpleTypeLabel, b::SimpleTypeLabel) = argtype(a) == typelabel(b) || argtype(a) === UnknownType()

struct TypeMismatch end
typemismatch = Err(TypeMismatch())

rettype(f::FunctionType, arg::SimpleTypeLabel) = f.ret
rettype(f::FunctionalType, arg::SimpleTypeLabel) = f.func(arg)

function typeinfer(a::SimpleTypeLabel, b::SimpleTypeLabel)
    if isapplicable(a, b)
        rettype(b, a)
    else
        typemismatch
    end
end

struct LabeledTerm{V, T<:AbstractTypeLabel}
    value::V
    label::T
end

LabeledProxy{V, T} = ProxyDB{LabeledTerm{V, T}}
function Base.show(io::IO, ::MIME"text/plain", p::LabeledProxy)
    if haskey(proxy_alias_dict, p)
        print(io, proxy_alias_dict[p])
    else
        print(io, p.value.value, "::", p.value.label)
    end
end
argtype(p::LabeledProxy) = argtype(p.value.label)
typelabel(p::LabeledProxy) = p.value.label
value(p::LabeledProxy) = p.value.value
typeinfer(a::LabeledProxy, b::LabeledProxy) = typeinfer(typelabel(a), typelabel(b))


to_labeled_proxy(p::ProxyDB) = ProxyDB(LabeledTerm(p.value, typelabel(p.value)))
labelize(n::DBNode) = traverse(n) do node, depth
    @match node begin
        ProxyDB(value) => to_labeled_proxy(ProxyDB(value))
        _ => node
    end
end

typelabel(_) = UnknownType()
typelabel(::Int64) = LabelType(:Int64)
typelabel(::Float64) = LabelType(:Float64)
typelabel(::Bool) = LabelType(:Bool)

# substitution rules for labeled proxy
function DeBruijnModule.substitute(a::LabeledProxy, b::LabeledProxy, context)
    return_type = @check typeinfer(typelabel(a), typelabel(b))
    if context === nothing
        normalized
    else
        LabeledTerm(value(a)(value(b)), return_type)
    end
end

end