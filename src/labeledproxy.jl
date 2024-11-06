module LabeledProxyModule

using ..ParseModule: AbstractToken
using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, normalized, to_debrujin, traverse
import ..DeBruijnModule
using LambdaRegression: Err, @alias, @check, set_alias!, alias_dict
using ..ReduceModule:beta_reduce
using MLStyle

@data TypeLabel begin
    LabelType(label::Symbol)
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

struct TypeMismatch
    message::Tuple{TypeLabel, TypeLabel}
    # TypeMismatch(message) = error(new(message))
end

struct FunctionTerm
    func
    argtype::TypeLabel
end
FunctionTerm(func, argtype::Symbol) = FunctionTerm(func, LabelType(argtype))
Base.show(io::IO, term::FunctionTerm) = print(io, get(alias_dict, term, term.func))
function FunctionTerm(func, argtype, alias::String)
    term = FunctionTerm(func, argtype)
    set_alias!(term, alias)
    term
end
FunctionProxy = ProxyDB{FunctionTerm}
functionproxy(xs...) = ProxyDB(FunctionTerm(xs...))

struct TypedTerm{V}
    value::V
    label::TypeLabel
end
TypedTerm(value, label::Symbol) = TypedTerm(value, LabelType(label))
Base.:(==)(a::TypedTerm, b::TypedTerm) = a.value == b.value
Base.:(==)(a::TypedTerm, b) = a.value == b
Base.:(==)(a, b::TypedTerm) = a == b.value
Base.show(io::IO, term::TypedTerm) = print(io, term.value)
TypedProxy = ProxyDB{<:TypedTerm}
typedproxy(xs...) = ProxyDB(TypedTerm(xs...))

typelabel(::DBNode) = UnresolvedType()
typelabel(p::TypedProxy) = p.value.label
typelabel(::Int64) = LabelType(:Int64)
typelabel(::Float64) = LabelType(:Float64)
typelabel(::Bool) = LabelType(:Bool)

function DeBruijnModule.substitute(f::FunctionProxy, n::DBNode)
    if typelabel(n) <= f.value.argtype
        return ProxyDB(f.value.func(n))
    elseif typelabel(n) === UnresolvedType()
        normalized
    else
        Err(TypeMismatch((typelabel(n), f.value.argtype)))
    end
end

labelize(x) = ProxyDB(TypedTerm(x, typelabel(x)))
labelize(n::DBNode) = traverse(n) do node, depth
    @match node begin
        ::ProxyDB => labelize(node)
        _ => node
    end
end
labelize(p::ProxyDB) = labelize(p.value)
labelize(p::TypedProxy) = p
labelize(p::FunctionProxy) = p
labelize(n::AbstractToken) = labelize(to_debrujin(n))

end