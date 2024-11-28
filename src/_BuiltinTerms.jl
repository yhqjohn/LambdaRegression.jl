module BuiltinTermsModule

import ..ParseModule: @lambda
using ..DeBruijnModule: DBNode
import LambdaRegression: set_alias!
import ..TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, -->, UnresolvedType

using TestItems

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

const FADD = TypedTerm(x->(y->x+y), :Float64-->:Float64-->:Float64)
const FSUB = TypedTerm(x->(y->x-y), :Float64-->:Float64-->:Float64)
const FMUL = TypedTerm(x->(y->x*y), :Float64-->:Float64-->:Float64)
const FDIV = TypedTerm(x->(y->x/y), :Float64-->:Float64-->:Float64)
const FEQ = TypedTerm(x->(y->x==y), :Float64-->:Float64-->:Bool)
const FGT = TypedTerm(x->(y->x>y), :Float64-->:Float64-->:Bool)
const FLT = TypedTerm(x->(y->x<y), :Float64-->:Float64-->:Bool)
const FGEQ = TypedTerm(x->(y->x>=y), :Float64-->:Float64-->:Bool)
const FLEQ = TypedTerm(x->(y->x<=y), :Float64-->:Float64-->:Bool)

const IADD = TypedTerm(x->(y->x+y), :Int-->:Int-->:Int)
const ISUB = TypedTerm(x->(y->x-y), :Int-->:Int-->:Int)
const IMUL = TypedTerm(x->(y->x*y), :Int-->:Int-->:Int)
const IDIV = TypedTerm(x->(y->x÷y), :Int-->:Int-->:Int)
const IEQ = TypedTerm(x->(y->x==y), :Int-->:Int-->:Bool)
const IGT = TypedTerm(x->(y->x>y), :Int-->:Int-->:Bool)
const ILT = TypedTerm(x->(y->x<y), :Int-->:Int-->:Bool)
const IGEQ = TypedTerm(x->(y->x>=y), :Int-->:Int-->:Bool)
const ILEQ = TypedTerm(x->(y->x<=y), :Int-->:Int-->:Bool)

const OR = TypedTerm(x->(y->x||y), :Bool-->:Bool-->:Bool)
const AND = TypedTerm(x->(y->x&&y), :Bool-->:Bool-->:Bool)

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

set_alias!(OR, "or")
set_alias!(AND, "and")

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


    term0 = TypedTerm(0, :Int)
    term1 = TypedTerm(1, :Int)
    term2 = TypedTerm(2, :Int)
    term3 = TypedTerm(3, :Int)

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

const K = DBNode(@lambda(x->(y->x)))
const KI = DBNode(@lambda(x->(y->y)))
const _IFTE = x->(x ? K : KI)
const IFTHENELSE = TypedTerm(_IFTE, :Bool-->UnresolvedType())
set_alias!(K, "K")
set_alias!(KI, "KI")
set_alias!(IFTHENELSE, "IFTE")

@testitem "if-then-else" begin
    import LambdaRegression.BuiltinTermsModule: K, KI, IFTHENELSE
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, ProxyDB, substitute
    @test substitute(ProxyDB{TypedTerm{Any}}(IFTHENELSE), ProxyDB{TypedTerm{Any}}(TypedTerm(true, :Bool))) === DBNode{TypedTerm{Any}}(K)
    @test substitute(ProxyDB{TypedTerm{Any}}(IFTHENELSE), ProxyDB{TypedTerm{Any}}(TypedTerm(false, :Bool))) === DBNode{TypedTerm{Any}}(KI)
end


const YCOMBINATOR = DBNode(@lambda f->((x->f(x(x)))*(x->f(x(x)))))
set_alias!(YCOMBINATOR, "Y")

@testitem "y-combinator" begin
    import LambdaRegression.BuiltinTermsModule: YCOMBINATOR, SIN
    import LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, ProxyDB, substitute
    @test substitute(DBNode{TypedTerm{Any}}(YCOMBINATOR), ProxyDB{TypedTerm{Any}}(SIN)) === DBNode{TypedTerm{Any}}(@lambda (x->SIN(x(x)))*(x->SIN(x(x))))
end

const FACT = DBNode(@lambda f->(n->(IFTHENELSE * (ILEQ * n * 1) * 1 * (IMUL * n * (f * (ISUB * n * 1))))))
const _type_fact = convert(DBNode{TypedTerm{Any}}, FACT)
set_alias!(FACT, "FACT")
set_alias!(_type_fact, "FACT")

const FIB = DBNode(@lambda f->(n->(IFTHENELSE * (OR * (ILEQ * n * 1) * (ILEQ * n * 2)) * 1 * (IADD * (f * (ISUB * n * 1)) * (f * (ISUB * n * 2))))))
const _type_fib = convert(DBNode{TypedTerm{Any}}, FIB)
set_alias!(FIB, "FIB")
set_alias!(_type_fib, "FIB")

@testitem "factorial" begin
    using LambdaRegression.BuiltinTermsModule: FACT, FIB, YCOMBINATOR
    using LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, ProxyDB, substitute
    using LambdaRegression.ReduceModule: beta_reduce, normalize
    using LambdaRegression: alias_dict, @lambda, set_alias!
    using LambdaRegression.DeBruijnModule: DBNode, normalized

    yf = beta_reduce(DBNode{TypedTerm{Any}}(@lambda YCOMBINATOR * FACT))
    set_alias!(yf, "YF")
    display(yf)

    let fact3 = DBNode{TypedTerm{Any}}(@lambda (YCOMBINATOR * FACT) * 3)
        for i in 1:20
            display(fact3)
            fact3 = beta_reduce(fact3)
        end
    end
    @test normalize(DBNode{TypedTerm{Any}}(@lambda (YCOMBINATOR * FACT) * 5)) == ProxyDB{TypedTerm{Any}}(TypedTerm(120, :Int))

    let fib3 = DBNode{TypedTerm{Any}}(@lambda (YCOMBINATOR * FIB) * 3)
        for i in 1:100
            display(fib3)
            fib3 = beta_reduce(fib3)
            if fib3 === normalized
                break
            end
        end
    end
    @test normalize(DBNode{TypedTerm{Any}}(@lambda (YCOMBINATOR * FIB) * 8), max_iter=10_000) == ProxyDB{TypedTerm{Any}}(TypedTerm(21, :Int))


end

export SIN, COS, TAN
export FADD, FSUB, FMUL, FDIV, FEQ, FGT, FLT, FGEQ, FLEQ
export IADD, ISUB, IMUL, IDIV, IEQ, IGT, ILT, IGEQ, ILEQ
export K, KI, IFTHENELSE, YCOMBINATOR
export FACT, FIB

end