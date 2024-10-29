module BuiltinTermsModule

import ..LabeledProxyModule: FunctionTerm
import ..ParseModule: @lambda
using ..DeBruijnModule: to_debrujin

const SIN = FunctionTerm(sin, :Float64 >> :Float64, "sin")
const COS = FunctionTerm(cos, :Float64 >> :Float64, "cos")
const TAN = FunctionTerm(tan, :Float64 >> :Float64, "tan")
const FADD = FunctionTerm(x->(y->x + y), :Float64 >> :Float64 >> :Float64, "+")
const FSUB = FunctionTerm(x->(y->x - y), :Float64 >> :Float64 >> :Float64, "-")
const FMUL = FunctionTerm(x->(y->x * y), :Float64 >> :Float64 >> :Float64, "*")
const FDIV = FunctionTerm(x->(y->x / y), :Float64 >> :Float64 >> :Float64, "/")
const FEQ = FunctionTerm(x->(y->x == y), :Float64 >> :Float64 >> :Bool, "==")
const FGT = FunctionTerm(x->(y->x > y), :Float64 >> :Float64 >> :Bool, ">")
const FLT = FunctionTerm(x->(y->x < y), :Float64 >> :Float64 >> :Bool, "<")
const FGEQ = FunctionTerm(x->(y->x >= y), :Float64 >> :Float64 >> :Bool, ">=")
const FLEQ = FunctionTerm(x->(y->x <= y), :Float64 >> :Float64 >> :Bool, "<=")

const IADD = FunctionTerm(x->(y->x + y), :Int64 >> :Int64 >> :Int64, "+")
const ISUB = FunctionTerm(x->(y->x - y), :Int64 >> :Int64 >> :Int64, "-")
const IMUL = FunctionTerm(x->(y->x * y), :Int64 >> :Int64 >> :Int64, "*")
const IDIV = FunctionTerm(x->(y->x ÷ y), :Int64 >> :Int64 >> :Int64, "/")
const IEQ = FunctionTerm(x->(y->x == y), :Int64 >> :Int64 >> :Bool, "==")
const IGT = FunctionTerm(x->(y->x > y), :Int64 >> :Int64 >> :Bool, ">")
const ILT = FunctionTerm(x->(y->x < y), :Int64 >> :Int64 >> :Bool, "<")
const IGEQ = FunctionTerm(x->(y->x >= y), :Int64 >> :Int64 >> :Bool, ">=")
const ILEQ = FunctionTerm(x->(y->x <= y), :Int64 >> :Int64 >> :Bool, "<=")

const IFTHENELSE = FunctionTerm(x->(y->(z->x ? y : z)), :Bool >> :Any >> :Any >> :Unknown, "if-then-else")

const YCOMBINATOR = to_debrujin(@lambda f->(x->f(x(x)))(x->f(x(x))))

FIBB = to_debrujin(@lambda f->(n->(IFTHENELSE * (ILEQ * n * 1) * 1 * (IMUL * n * (f * (ISUB * n * 1))))))

end