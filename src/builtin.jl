module BuiltinTermsModule

import ..LabeledProxyModule: LabeledTerm, →, UnknownType
import ..ParseModule: @lambda
using ..DeBruijnModule: to_debrujin
import LambdaRegression: set_alias!

const SIN = LabeledTerm(sin, :Float64 → :Float64, "sin")
const COS = LabeledTerm(cos, :Float64 → :Float64, "cos")
const TAN = LabeledTerm(tan, :Float64 → :Float64, "tan")
const FADD = LabeledTerm(x->(y->x + y), :Float64 → :Float64 → :Float64, "+")
const FSUB = LabeledTerm(x->(y->x - y), :Float64 → :Float64 → :Float64, "-")
const FMUL = LabeledTerm(x->(y->x * y), :Float64 → :Float64 → :Float64, "*")
const FDIV = LabeledTerm(x->(y->x / y), :Float64 → :Float64 → :Float64, "/")
const FEQ = LabeledTerm(x->(y->x == y), :Float64 → :Float64 → :Bool, "==")
const FGT = LabeledTerm(x->(y->x > y), :Float64 → :Float64 → :Bool, ">")
const FLT = LabeledTerm(x->(y->x < y), :Float64 → :Float64 → :Bool, "<")
const FGEQ = LabeledTerm(x->(y->x >= y), :Float64 → :Float64 → :Bool, ">=")
const FLEQ = LabeledTerm(x->(y->x <= y), :Float64 → :Float64 → :Bool, "<=")

const IADD = LabeledTerm(x->(y->x + y), :Int64 → :Int64 → :Int64, "+")
const ISUB = LabeledTerm(x->(y->x - y), :Int64 → :Int64 → :Int64, "-")
const IMUL = LabeledTerm(x->(y->x * y), :Int64 → :Int64 → :Int64, "*")
const IDIV = LabeledTerm(x->(y->x ÷ y), :Int64 → :Int64 → :Int64, "/")
const IEQ = LabeledTerm(x->(y->x == y), :Int64 → :Int64 → :Bool, "==")
const IGT = LabeledTerm(x->(y->x > y), :Int64 → :Int64 → :Bool, ">")
const ILT = LabeledTerm(x->(y->x < y), :Int64 → :Int64 → :Bool, "<")
const IGEQ = LabeledTerm(x->(y->x >= y), :Int64 → :Int64 → :Bool, ">=")
const ILEQ = LabeledTerm(x->(y->x <= y), :Int64 → :Int64 → :Bool, "<=")

const K = to_debrujin(@lambda x->(y->x))
set_alias!(K, "K")
const KI = to_debrujin(@lambda x->(y->y))
set_alias!(KI, "KI")

const IFTHENELSE = LabeledTerm(x->(x ? K : KI), :Bool → UnknownType(), "IFTHENELSE")

const YCOMBINATOR = to_debrujin(@lambda f->((x->f(x(x)))*(x->f(x(x)))))
set_alias!(YCOMBINATOR, "Y")

FIBB = to_debrujin(@lambda f->(n->(IFTHENELSE * (ILEQ * n * 1) * 1 * (IMUL * n * (f * (ISUB * n * 1))))))
set_alias!(FIBB, "FIBB")


export SIN, COS, TAN
export FADD, FSUB, FMUL, FDIV, FEQ, FGT, FLT, FGEQ, FLEQ
export IADD, ISUB, IMUL, IDIV, IEQ, IGT, ILT, IGEQ, ILEQ
export K, KI, IFTHENELSE, YCOMBINATOR
export FIBB

end