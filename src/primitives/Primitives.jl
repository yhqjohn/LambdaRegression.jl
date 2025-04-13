"""
    Primitives

The concrete host values used by the typed lambda language and their type
signatures.  This layer depends on the common type constructors and syntax
nodes, but the type system does not depend on this catalogue.  `COND` keeps
its lazy branch representation, and `AMAP`/`AFOLDL` preserve the existing
outer logical array semantics.

The functions exported here are the actual function objects used by
generation, typing, and execution; callers should compare them by identity
when classifying primitive heads.
"""
module Primitives

using TestItems
using ..TypeSystem: TyConst, TyFun, TyList, TyArray, TyScheme, BoolTy, IntTy,
    FloatTy, newtyvar, ∀, →, ×
import ..TypeSystem: get_type
using ..ASTs.NodeASTs: AstTree, AbsNode, IndexNode

export TRUE, FALSE, NOT, COND
export INL, INR, PAIR, FST, SND
export CONS, HEAD, TAIL, LENGTH, CASENILERR, UNCONS_CASE, TOLIST, NilErr
export AMAP, AFOLDL, LENF
export FSIN, FCOS, FTAN, FEXP, FLOG, FABS, FSQRT, SQR
export IEQ, ILE, IADD, ISUB, IMUL
export FEQ, FLE, FADD, FSUB, FMUL, FDIV, FPOW, FMAX, FMIN

const TRUE = true
const FALSE = false
NOT(x) = !x

"""A lazy conditional selector represented by ordinary lambda terms."""
const _cond_x = gensym(:x)
const _cond_y = gensym(:y)
const T = AstTree(AbsNode(_cond_x, AbsNode(_cond_y, IndexNode(1))))
const F = AstTree(AbsNode(_cond_x, AbsNode(_cond_y, IndexNode(0))))
COND(x) = x ? T : F

INL(x) = x
INR(x) = x

PAIR(x) = y -> (x, y)
PAIR(x, y) = (x, y)
FST(x) = first(x)
SND(x) = last(x)

struct _NilErr end
const NILERR = _NilErr()
const NilErr = TyConst(:NilErr)
CONS(x) = l -> vcat([x], collect(l))
HEAD(xs) = isempty(xs) ? NILERR : xs[1]
TAIL(xs) = isempty(xs) ? NILERR : xs[2:end]
LENGTH(xs) = length(xs)
CASENILERR(f) = g -> (x -> (x isa _NilErr ? g(x) : f(x)))
UNCONS_CASE(xs) = ok -> fallback -> (isempty(xs) ? fallback : ok(xs[1])(xs[2:end]))
TOLIST(xs) = collect(xs)

"""Curried length-preserving map over the outer logical array layer."""
function _amap_dense_outer(f, xs::AbstractArray)
    mapped = map(f, eachslice(xs; dims=1))
    isempty(mapped) && return mapped
    return first(mapped) isa AbstractArray ? stack(mapped; dims=1) : mapped
end

_amap_outer(f, xs::AbstractVector) = map(f, xs)
_amap_outer(f, xs::AbstractArray) = _amap_dense_outer(f, xs)
AMAP(f) = xs -> _amap_outer(f, xs)

"""Curried left fold over the outer logical array layer."""
_afoldl_outer(op, init, xs::AbstractVector) =
    foldl((acc, x) -> op(acc)(x), xs; init=init)
_afoldl_outer(op, init, xs::AbstractArray) =
    foldl((acc, x) -> op(acc)(x), eachslice(xs; dims=1); init=init)
AFOLDL(op) = init -> xs -> _afoldl_outer(op, init, xs)

"""
Return the current logical array layer's length as a floating-point value.
A dense array represents nested layers, so only its first dimension is counted;
inner dimensions, including empty inner layers, do not change this length.
"""
LENF(xs) = float(size(xs, 1))

FSIN(x) = sin(x)
FCOS(x) = cos(x)
FTAN(x) = tan(x)
FEXP(x) = exp(x)
FLOG(x) = log(x)
FABS(x) = abs(x)
FSQRT(x) = sqrt(x)
SQR(x) = x * x

IEQ(x) = y -> x == y
ILE(x) = y -> x <= y
IADD(x) = y -> x + y
ISUB(x) = y -> x - y
IMUL(x) = y -> x * y

FEQ(x) = y -> x == y
FLE(x) = y -> x <= y
FADD(x) = y -> x + y
FSUB(x) = y -> x - y
FMUL(x) = y -> x * y
FDIV(x) = y -> x / y
FPOW(x) = y -> x ^ y
FMAX(x) = y -> max(x, y)
FMIN(x) = y -> min(x, y)

const _α, _β, _γ = newtyvar(0), newtyvar(0), newtyvar(0)
const IntDiscr = IntTy → IntTy → BoolTy
const IntOp = IntTy → IntTy → IntTy
const FloatDiscr = FloatTy → FloatTy → BoolTy
const FloatOp = FloatTy → FloatTy → FloatTy

get_type(::typeof(COND)) = ∀([_α], BoolTy → _α → _α → _α)
get_type(::typeof(NOT)) = BoolTy → BoolTy
get_type(::typeof(INL)) = ∀([_α, _β], _α → (_α + _β))
get_type(::typeof(INR)) = ∀([_α, _β], _β → (_α + _β))
get_type(::typeof(PAIR)) = ∀([_α, _β], _α → _β → (_α × _β))
get_type(::typeof(FST)) = ∀([_α, _β], (_α × _β) → _α)
get_type(::typeof(SND)) = ∀([_α, _β], (_α × _β) → _β)
get_type(::typeof(CONS)) = ∀([_α], _α → TyList(_α) → TyList(_α))
get_type(::typeof(HEAD)) = ∀([_α], TyList(_α) → (_α + NilErr))
get_type(::typeof(TAIL)) = ∀([_α], TyList(_α) → TyList(_α) + NilErr)
get_type(::typeof(LENGTH)) = ∀([_α], TyList(_α) → IntTy)
get_type(::typeof(CASENILERR)) = ∀([_α, _β], (_α → _β) → (NilErr → _β) → (_α + NilErr) → _β)
get_type(::typeof(UNCONS_CASE)) = ∀([_α, _β], TyList(_α) → (_α → TyList(_α) → _β) → _β → _β)
get_type(::typeof(TOLIST)) = ∀([_α, _β], TyArray(_α, _β) → TyList(_β))

get_type(::typeof(AMAP)) = ∀([_α, _β, _γ], (_β → _γ) → TyArray(_α, _β) → TyArray(_α, _γ))
get_type(::typeof(AFOLDL)) = ∀([_α, _β, _γ], (_γ → _β → _γ) → _γ → TyArray(_α, _β) → _γ)
get_type(::typeof(LENF)) = ∀([_α, _β], TyArray(_α, _β) → FloatTy)

get_type(::typeof(FSIN)) = FloatTy → FloatTy
get_type(::typeof(FCOS)) = FloatTy → FloatTy
get_type(::typeof(FTAN)) = FloatTy → FloatTy
get_type(::typeof(FEXP)) = FloatTy → FloatTy
get_type(::typeof(FLOG)) = FloatTy → FloatTy
get_type(::typeof(FABS)) = FloatTy → FloatTy
get_type(::typeof(FSQRT)) = FloatTy → FloatTy
get_type(::typeof(SQR)) = FloatTy → FloatTy

get_type(::typeof(IEQ)) = IntDiscr
get_type(::typeof(ILE)) = IntDiscr
get_type(::typeof(IADD)) = IntOp
get_type(::typeof(ISUB)) = IntOp
get_type(::typeof(IMUL)) = IntOp
get_type(::typeof(FEQ)) = FloatDiscr
get_type(::typeof(FLE)) = FloatDiscr
get_type(::typeof(FADD)) = FloatOp
get_type(::typeof(FSUB)) = FloatOp
get_type(::typeof(FMUL)) = FloatOp
get_type(::typeof(FDIV)) = FloatOp
get_type(::typeof(FPOW)) = FloatOp
get_type(::typeof(FMAX)) = FloatOp
get_type(::typeof(FMIN)) = FloatOp

@testitem "Primitives preserve identity, laziness, and array layer semantics" begin
    using Test
    using LambdaRegression.Primitives: COND, AMAP, AFOLDL, FADD
    using LambdaRegression.TypeSystem: TyScheme, get_type

    @test COND(true) === COND(true)
    @test COND(false) === COND(false)
    @test COND(true) !== COND(false)
    @test get_type(FADD) isa TyScheme || get_type(FADD) isa LambdaRegression.TypeSystem.Ty

    matrix = [1.0 2.0; 3.0 4.0]
    @test AMAP(x -> x .+ 1)(matrix) == [2.0 3.0; 4.0 5.0]
    @test AFOLDL(acc -> row -> acc + sum(row))(0.0)(matrix) == 10.0
end

@testitem "Primitives LENF observes one logical array layer" begin
    using Test
    using LambdaRegression.Primitives: AMAP, AFOLDL, LENF

    matrix = reshape(collect(1.0:6.0), 2, 3)
    nested = [collect(row) for row in eachslice(matrix; dims=1)]
    @test LENF(matrix) === 2.0
    @test LENF(nested) === 2.0
    @test AMAP(LENF)(matrix) == [3.0, 3.0]
    @test AFOLDL(acc -> row -> acc + LENF(row))(0.0)(matrix) === 6.0
    @test LENF([1.0, 2.0, 3.0]) === 3.0
    @test LENF(Float64[]) === 0.0
    @test LENF(zeros(2, 0)) === 2.0
    @test LENF(zeros(0, 3)) === 0.0

    tensor = reshape(collect(1.0:24.0), 2, 3, 4)
    @test LENF(tensor) === 2.0
    @test AMAP(AMAP(LENF))(tensor) == fill(4.0, 2, 3)

    # Mapping can change inner lengths or remove a layer, but keeps the outer one.
    @test LENF(AMAP(row -> vcat(row, row))(matrix)) === 2.0
    @test LENF(AMAP(row -> sum(row))(matrix)) === 2.0
    @test LENF(AMAP(row -> Float64[])(matrix)) === 2.0
    @test LENF(AMAP(row -> vcat(row, row))(nested)) === 2.0
end

@testitem "Primitives expose the extended real scalar operators" begin
    using Test
    using LambdaRegression.Primitives: FTAN, FEXP, FLOG, FABS, FSQRT, SQR,
        FPOW, FMAX, FMIN
    using LambdaRegression.TypeSystem: FloatTy, TyFun, get_type

    unary = ((FTAN, tan(0.25)), (FEXP, exp(0.25)), (FLOG, log(2.0)),
        (FABS, abs(-2.0)), (FSQRT, sqrt(4.0)), (SQR, 3.0^2))
    for (primitive, expected) in unary
        argument = primitive === FLOG ? 2.0 : primitive === FABS ? -2.0 :
            primitive === FSQRT ? 4.0 : primitive === SQR ? 3.0 : 0.25
        @test primitive(argument) ≈ expected
        @test get_type(primitive) == TyFun(FloatTy, FloatTy)
    end

    @test FPOW(2.0)(3.0) === 8.0
    @test FMAX(2.0)(3.0) === 3.0
    @test FMIN(2.0)(3.0) === 2.0
    for primitive in (FPOW, FMAX, FMIN)
        @test get_type(primitive) ==
            TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    end
end

end # module Primitives
