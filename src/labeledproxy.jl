module LabeledProxyModule

using ..ParseModule: AbstractToken
using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, normalized, to_debrujin, traverse
import ..DeBruijnModule
using LambdaRegression: Err, @alias, @check, set_alias!, alias_dict
using ..ReduceModule:beta_reduce
using MLStyle

abstract type AbstractTypeLabel end

@data SimpleTypeLabel <: AbstractTypeLabel begin
    LabelType(label::Symbol)
    FunctionType(arg::SimpleTypeLabel, ret::SimpleTypeLabel)
    FunctionalType(arg::SimpleTypeLabel, func::Any)
    UnknownType()
end
→(a::SimpleTypeLabel, b::SimpleTypeLabel) = FunctionType(a, b)
→(a::Symbol, b::Symbol) = FunctionType(LabelType(a), LabelType(b))
→(a::SimpleTypeLabel, b::Symbol) = FunctionType(a, LabelType(b))
→(a::Symbol, b::SimpleTypeLabel) = FunctionType(LabelType(a), b)
argtype(f::FunctionalType) = f.arg
argtype(f::FunctionType) = f.arg
argtype(::LabelType) = nothing
argtype(::UnknownType) = UnknownType()

isapplicable(a::SimpleTypeLabel, b::SimpleTypeLabel) = argtype(a) == b || argtype(a) === UnknownType()

Base.show(io::IO, label::LabelType) = print(io, label.label)
Base.show(io::IO, label::FunctionType) = print(io, "($(label.arg) → $(label.ret))")
Base.show(io::IO, label::FunctionalType) = print(io, "$(label.func)(:$(label.arg))")
Base.show(io::IO, label::UnknownType) = print(io, "Unknown")

struct TypeMismatch 
    message::Tuple{AbstractTypeLabel, AbstractTypeLabel}
    TypeMismatch(message) = error(new(message))
end

rettype(f::FunctionType, arg::SimpleTypeLabel) = f.ret
rettype(f::FunctionalType, arg::SimpleTypeLabel) = f.func(arg)

function typeinfer(a::SimpleTypeLabel, b::SimpleTypeLabel)
    if isapplicable(a, b)
        rettype(a, b)
    else
        TypeMismatch((a, b))
    end
end

struct LabeledTerm{V, T<:AbstractTypeLabel}
    value::V
    label::T
end
function Base.show(io::IO, term::LabeledTerm)
    if haskey(alias_dict, term)
        print(io, alias_dict[term])
    else
        # print(io, term.value, ":", term.label)
        print(io, term.value)
    end
end

function LabeledTerm(value, label, alias::String)
    term = LabeledTerm(value, label)
    set_alias!(term, alias)
    term
end
to_labeled_term(value, label) = LabeledTerm(value, label)
to_labeled_term(value::LabeledTerm, _) = value

function (f::LabeledTerm)(arg::LabeledTerm)
    return_type = @check typeinfer(f.label, arg.label)
    return to_labeled_term(f.value(arg.value), return_type)
end

LabeledProxy = ProxyDB{<:LabeledTerm}

argtype(p::LabeledProxy) = argtype(p.value.label)
typelabel(p::LabeledProxy) = p.value.label
value(p::LabeledProxy) = p.value.value
typeinfer(a::LabeledProxy, b::LabeledProxy) = typeinfer(typelabel(a), typelabel(b))


to_labeled_proxy(p::ProxyDB) = ProxyDB(LabeledTerm(p.value, typelabel(p.value)))
labelize(n::DBNode) = traverse(n) do node, depth
    @match node begin
        ::LabeledProxy => node
        ProxyDB(value) => to_labeled_proxy(ProxyDB(value))
        _ => node
    end
end
labelize(n::AbstractToken) = labelize(to_debrujin(n))
labelize(x) = ProxyDB(LabeledTerm(x, typelabel(x)))

typelabel(_) = UnknownType()
typelabel(::Int64) = LabelType(:Int64)
typelabel(::Float64) = LabelType(:Float64)
typelabel(::Bool) = LabelType(:Bool)

end