module BuiltinTermsModule

import ..ParseModule: @lambda
# using ..DeBruijnModule: to_debrujin
import LambdaRegression: set_alias!
import ..TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, -->

using TestItems


# const SIN = FunctionTerm(x->TypedTerm(sin(x.value.value), :Float64), :Float64, "SIN")
# const COS = FunctionTerm(x->TypedTerm(cos(x.value.value), :Float64), :Float64, "COS")
# const TAN = FunctionTerm(x->TypedTerm(tan(x.value.value), :Float64), :Float64, "TAN")

const SIN = TypedTerm(sin, :Float64-->:Float64)
const COS = TypedTerm(cos, :Float64-->:Float64)
const TAN = TypedTerm(tan, :Float64-->:Float64)

set_alias!(SIN, "SIN")
set_alias!(COS, "COS")
set_alias!(TAN, "TAN")

@testitem "unary functions" begin
    import LambdaRegression.BuiltinTermsModule: SIN, COS, TAN, TypedTerm
    import LambdaRegression.TermsModule: TypedTermProxy
    import LambdaRegression.DeBruijnModule: substitute
    @test SIN == convert(TypedTerm{Any}, SIN)
    display(ProxyDB{TypedTerm{Any}}(SIN))
    @test substitute(ProxyDB{TypedTerm{Any}}(SIN), ProxyDB{TypedTerm{Any}}(TypedTerm(1.0, :Float64))) === ProxyDB{TypedTerm{Any}}(TypedTerm(sin(1.0), :Float64))
    @test substitute(ProxyDB{TypedTerm{Any}}(COS), ProxyDB{TypedTerm{Any}}(TypedTerm(1.0, :Float64))) === ProxyDB{TypedTerm{Any}}(TypedTerm(cos(1.0), :Float64))
    @test substitute(ProxyDB{TypedTerm{Any}}(TAN), ProxyDB{TypedTerm{Any}}(TypedTerm(1.0, :Float64))) === ProxyDB{TypedTerm{Any}}(TypedTerm(tan(1.0), :Float64))
end

# const FADD = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value + y.value.value, :Float64), :Float64)
# ), :Float64, "+")
# const FSUB = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value - y.value.value, :Float64), :Float64)
# ), :Float64, "-")
# const FMUL = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value * y.value.value, :Float64), :Float64)
# ), :Float64, "×")
# const FDIV = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value / y.value.value, :Float64), :Float64)
# ), :Float64, "÷")
# const FEQ = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value == y.value.value, :Bool), :Float64)
# ), :Float64, "==")
# const FGT = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value > y.value.value, :Bool), :Float64)
# ), :Float64, ">")
# const FLT = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value < y.value.value, :Bool), :Float64)
# ), :Float64, "<")
# const FGEQ = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value >= y.value.value, :Bool), :Float64)
# ), :Float64, "≥")
# const FLEQ = FunctionTerm(x->(
#     FunctionTerm(y->TypedTerm(x.value.value <= y.value.value, :Bool), :Float64)
# ), :Float64, "≤")

const FADD = TypedTerm(x->(y->x+y), :Float64-->:Float64-->:Float64)
const FSUB = TypedTerm(x->(y->x-y), :Float64-->:Float64-->:Float64)
const FMUL = TypedTerm(x->(y->x*y), :Float64-->:Float64-->:Float64)
const FDIV = TypedTerm(x->(y->x/y), :Float64-->:Float64-->:Float64)
const FEQ = TypedTerm(x->(y->x==y), :Float64-->:Float64-->:Bool)
const FGT = TypedTerm(x->(y->x>y), :Float64-->:Float64-->:Bool)
const FLT = TypedTerm(x->(y->x<y), :Float64-->:Float64-->:Bool)
const FGEQ = TypedTerm(x->(y->x>=y), :Float64-->:Float64-->:Bool)
const FLEQ = TypedTerm(x->(y->x<=y), :Float64-->:Float64-->:Bool)

const IADD = TypedTerm(x->(y->x+y), :Int64-->:Int64-->:Int64)
const ISUB = TypedTerm(x->(y->x-y), :Int64-->:Int64-->:Int64)
const IMUL = TypedTerm(x->(y->x*y), :Int64-->:Int64-->:Int64)
const IDIV = TypedTerm(x->(y->x÷y), :Int64-->:Int64-->:Int64)
const IEQ = TypedTerm(x->(y->x==y), :Int64-->:Int64-->:Bool)
const IGT = TypedTerm(x->(y->x>y), :Int64-->:Int64-->:Bool)
const ILT = TypedTerm(x->(y->x<y), :Int64-->:Int64-->:Bool)
const IGEQ = TypedTerm(x->(y->x>=y), :Int64-->:Int64-->:Bool)
const ILEQ = TypedTerm(x->(y->x<=y), :Int64-->:Int64-->:Bool)

set_alias!(FADD, "+")
set_alias!(FSUB, "-")
set_alias!(FMUL, "×")
set_alias!(FDIV, "÷")
set_alias!(FEQ, "==")
set_alias!(FGT, ">")
set_alias!(FLT, "<")
set_alias!(FGEQ, "≥")
set_alias!(FLEQ, "≤")

set_alias!(IADD, "+")
set_alias!(ISUB, "-")
set_alias!(IMUL, "×")
set_alias!(IDIV, "÷")
set_alias!(IEQ, "==")
set_alias!(IGT, ">")
set_alias!(ILT, "<")
set_alias!(IGEQ, "≥")
set_alias!(ILEQ, "≤")


@testitem "binary functions" begin
    import LambdaRegression.BuiltinTermsModule: FADD, FSUB, FMUL, FDIV, FEQ, FGT, FLT, FGEQ, FLEQ, IADD, ISUB, IMUL, IDIV, IEQ, IGT, ILT, IGEQ, ILEQ
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, ProxyDB, substitute
    term0 = TypedTerm(0.0, :Float64)
    term1 = TypedTerm(1.0, :Float64)
    term2 = TypedTerm(2.0, :Float64)
    termsecond = TypedTerm(0.5, :Float64)
    termtrue = TypedTerm(true, :Bool)
    termfalse = TypedTerm(false, :Bool)

    newincre = substitute(ProxyDB{TypedTerm{Any}}(FADD), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(newincre, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term2)
    newdecre = substitute(ProxyDB{TypedTerm{Any}}(FSUB), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(newdecre, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term0)
    newmul = substitute(ProxyDB{TypedTerm{Any}}(FMUL), ProxyDB{TypedTerm{Any}}(term2))
    @test substitute(newmul, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term2)
    newinv = substitute(ProxyDB{TypedTerm{Any}}(FDIV), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(newinv, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termsecond)
    neweq1 = substitute(ProxyDB{TypedTerm{Any}}(FEQ), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(neweq1, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(termtrue)
    @test substitute(neweq1, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
    new1gt = substitute(ProxyDB{TypedTerm{Any}}(FGT), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(new1gt, ProxyDB{TypedTerm{Any}}(term0)) === ProxyDB{TypedTerm{Any}}(termtrue)
    @test substitute(new1gt, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
    new1lt = substitute(ProxyDB{TypedTerm{Any}}(FLT), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(new1lt, ProxyDB{TypedTerm{Any}}(term0)) === ProxyDB{TypedTerm{Any}}(termfalse)
    @test substitute(new1lt, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termtrue)
    new1geq = substitute(ProxyDB{TypedTerm{Any}}(FGEQ), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(new1geq, ProxyDB{TypedTerm{Any}}(term0)) === ProxyDB{TypedTerm{Any}}(termtrue)
    @test substitute(new1geq, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
    new1leq = substitute(ProxyDB{TypedTerm{Any}}(FLEQ), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(new1leq, ProxyDB{TypedTerm{Any}}(term0)) === ProxyDB{TypedTerm{Any}}(termfalse)
    @test substitute(new1leq, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termtrue)


    term0 = TypedTerm(0, :Int64)
    term1 = TypedTerm(1, :Int64)
    term2 = TypedTerm(2, :Int64)
    term3 = TypedTerm(3, :Int64)

    newincre = substitute(ProxyDB{TypedTerm{Any}}(IADD), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(newincre, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term2)
    newdecre = substitute(ProxyDB{TypedTerm{Any}}(ISUB), ProxyDB{TypedTerm{Any}}(term1))
    @test substitute(newdecre, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term0)
    newmul = substitute(ProxyDB{TypedTerm{Any}}(IMUL), ProxyDB{TypedTerm{Any}}(term2))
    @test substitute(newmul, ProxyDB{TypedTerm{Any}}(term1)) === ProxyDB{TypedTerm{Any}}(term2)
    new3div = substitute(ProxyDB{TypedTerm{Any}}(IDIV), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3div, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(term1)
    new3eq = substitute(ProxyDB{TypedTerm{Any}}(IEQ), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3eq, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
    new3gt = substitute(ProxyDB{TypedTerm{Any}}(IGT), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3gt, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termtrue)
    new3lt = substitute(ProxyDB{TypedTerm{Any}}(ILT), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3lt, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
    new3geq = substitute(ProxyDB{TypedTerm{Any}}(IGEQ), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3geq, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termtrue)
    new3leq = substitute(ProxyDB{TypedTerm{Any}}(ILEQ), ProxyDB{TypedTerm{Any}}(term3))
    @test substitute(new3leq, ProxyDB{TypedTerm{Any}}(term2)) === ProxyDB{TypedTerm{Any}}(termfalse)
end


# const K = to_debrujin(@lambda x->(y->x))
# set_alias!(K, "K")
# const KI = to_debrujin(@lambda x->(y->y))
# set_alias!(KI, "KI")

# const IFTHENELSE = FunctionTerm(x->(x.value.value ? K : KI), :Bool, "IFTE")

# const YCOMBINATOR = to_debrujin(@lambda f->((x->f(x(x)))*(x->f(x(x)))))
# set_alias!(YCOMBINATOR, "Y")

# FACT = to_debrujin(@lambda f->(n->(IFTHENELSE * (ILEQ * n * 1) * 1 * (IMUL * n * (f * (ISUB * n * 1))))))
# set_alias!(FACT, "FACT")

# export SIN, COS, TAN
# export FADD, FSUB, FMUL, FDIV, FEQ, FGT, FLT, FGEQ, FLEQ
# export IADD, ISUB, IMUL, IDIV, IEQ, IGT, ILT, IGEQ, ILEQ
# export K, KI, IFTHENELSE, YCOMBINATOR
# export FACT

end