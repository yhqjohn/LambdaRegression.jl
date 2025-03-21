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
export FSIN, FCOS, FSQRT, SQR
export IEQ, ILE, IADD, ISUB, IMUL
export FEQ, FLE, FADD, FSUB, FMUL, FDIV

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

LENF(xs) = float(length(xs))

FSIN(x) = sin(x)
FCOS(x) = cos(x)
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

end # module Primitives
