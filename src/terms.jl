module TermsModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
using MLStyle
import LambdaRegression: Err
import ..DeBruijnModule: substitute

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

struct TypeMismatch
    expected::TypeLabel
    actual::TypeLabel
end


########################################
# Terms
########################################
abstract type AbstractTerm end

# untyped term
struct Term <: AbstractTerm
    value
end
value(term::Term) = term.value
TermProxy = ProxyDB{Term}
TermDB = DBNode{Term}

Base.convert(::Type{Term}, x) = Term(x)
Base.convert(::Type{Term}, x::Term) = x

unpack_to_value(x) = x
unpack_to_value(x::ProxyDB) = unpack_to_value(x.value)
unpack_to_value(x::Term) = x.value

pack_to_termnode(n::DBNode) = n
pack_to_termnode(n::Term) = ProxyDB(n)
pack_to_termnode(x) = ProxyDB(Term(x))


function substitute(func::TermProxy, arg::TermDB)
    func = unpack_to_value(func)
    arg = unpack_to_value(arg)
    result = func(arg)
    return pack_to_termnode(result)
end

## typed terms
struct TypedTerm <: AbstractTerm
    value
    type::TypeLabel
end
TypedTermProxy = ProxyDB{TypedTerm}
TypedTermDB = DBNode{TypedTerm}

Base.convert(::Type{TypedTerm}, x) = TypedTerm(x, typelabel(x))
Base.convert(::Type{TypedTerm}, x::TypedTerm) = x

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

function substitute(func::TypedTermProxy, arg::TypedTermDB)
    if !typelabel(func) <: ArrowType
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



















end