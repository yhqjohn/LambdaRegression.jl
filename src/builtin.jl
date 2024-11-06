module BuiltinTermsModule

import ..LabeledProxyModule: LabelType, UnknownType, AnyType, FunctionTerm, TypedTerm
import ..ParseModule: @lambda
using ..DeBruijnModule: to_debrujin
import LambdaRegression: set_alias!

const SIN = FunctionTerm(x->TypedTerm(sin(x.value.value), :Float64), :Float64, "SIN")
const COS = FunctionTerm(x->TypedTerm(cos(x.value.value), :Float64), :Float64, "COS")
const TAN = FunctionTerm(x->TypedTerm(tan(x.value.value), :Float64), :Float64, "TAN")
const FADD = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value + y.value.value, :Float64), :Float64)
), :Float64, "+")
const FSUB = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value - y.value.value, :Float64), :Float64)
), :Float64, "-")
const FMUL = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value * y.value.value, :Float64), :Float64)
), :Float64, "×")
const FDIV = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value / y.value.value, :Float64), :Float64)
), :Float64, "÷")
const FEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value == y.value.value, :Bool), :Float64)
), :Float64, "==")
const FGT = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value > y.value.value, :Bool), :Float64)
), :Float64, ">")
const FLT = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value < y.value.value, :Bool), :Float64)
), :Float64, "<")
const FGEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value >= y.value.value, :Bool), :Float64)
), :Float64, "≥")
const FLEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value <= y.value.value, :Bool), :Float64)
), :Float64, "≤")

const IADD = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value + y.value.value, :Int64), :Int64)
), :Int64, "+")
const ISUB = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value - y.value.value, :Int64), :Int64)
), :Int64, "-")
const IMUL = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value * y.value.value, :Int64), :Int64)
), :Int64, "×")
const IDIV = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value ÷ y.value.value, :Int64), :Int64)
), :Int64, "÷")
const IEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value == y.value.value, :Bool), :Int64)
), :Int64, "==")
const IGT = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value > y.value.value, :Bool), :Int64)
), :Int64, ">")
const ILT = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value < y.value.value, :Bool), :Int64)
), :Int64, "<")
const IGEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value >= y.value.value, :Bool), :Int64)
), :Int64, "≥")
const ILEQ = FunctionTerm(x->(
    FunctionTerm(y->TypedTerm(x.value.value <= y.value.value, :Bool), :Int64)
), :Int64, "≤")


const K = to_debrujin(@lambda x->(y->x))
set_alias!(K, "K")
const KI = to_debrujin(@lambda x->(y->y))
set_alias!(KI, "KI")

const IFTHENELSE = FunctionTerm(x->(x.value.value ? K : KI), :Bool, "IFTE")

const YCOMBINATOR = to_debrujin(@lambda f->((x->f(x(x)))*(x->f(x(x)))))
set_alias!(YCOMBINATOR, "Y")

FACT = to_debrujin(@lambda f->(n->(IFTHENELSE * (ILEQ * n * 1) * 1 * (IMUL * n * (f * (ISUB * n * 1))))))
set_alias!(FACT, "FACT")

export SIN, COS, TAN
export FADD, FSUB, FMUL, FDIV, FEQ, FGT, FLT, FGEQ, FLEQ
export IADD, ISUB, IMUL, IDIV, IEQ, IGT, ILT, IGEQ, ILEQ
export K, KI, IFTHENELSE, YCOMBINATOR
export FACT

end