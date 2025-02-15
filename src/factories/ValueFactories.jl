"""
    ValueFactories

Typed value factories for generation contexts.

Factories answer three questions: whether they support a concrete closed type,
which available universe types they can provide, and how to build a runtime value
for one supported type. They are ordinary opt-in context/factory tools rather
than privileged generation rules.

Quickstart:

```julia
using Random
using LambdaRegression.TypeSystem: FloatTy, IntTy, BoolTy, TyArray, TyNat

VF = LambdaRegression.ValueFactories
types = [
    FloatTy,
    IntTy,
    TyArray(TyNat(3), FloatTy),
    TyArray(TyNat(2), TyArray(TyNat(3), BoolTy)),
]
ctx = VF.factory_context(VF.default_value_factories(), types; rng=Random.MersenneTwister(1))
```

`ctx.consts` and `ctx.consts_ty` can be merged into a generation context. The
same factory tuple can also be passed to
`CanonicalGeneration.TypedValueFactoryProvider` to materialize values lazily
during canonical generation.

Available factories:

- `NilFactory()` provides empty `List` values.
- `FloatScalarFactory(mean=0, std=1)` samples Gaussian `Float` values.
- `IntScalarFactory(lo=-10, hi=10)` samples inclusive bounded `Int` values.
- `FloatArrayFactory(mean=0, std=1)`, `IntArrayFactory(lo=-10, hi=10)`, and
  `BoolArrayFactory()` provide concrete-length nested `Vector` values for
  `Array[n, ...]` types ending in the matching scalar type.

Users can add new factories by subtyping `AbstractValueFactory` and defining
`supports(factory, ty)`, `make_value(factory, ty, rng)`, and optionally
`factory_name(factory)`. Generation weight and provider routing stay outside
this module.
"""
module ValueFactories

import Random

using ..TypeSystem: Ty, TyConst, TyList, TyArray, TyNat, TyLike, BoolTy, FloatTy, IntTy, semantic_typekey, typekey
using TestItems

export AbstractValueFactory
export NilFactory, FloatScalarFactory, IntScalarFactory
export FloatArrayFactory, IntArrayFactory, BoolArrayFactory
export supports, provided_types, make_value, factory_context
export default_scalar_factories, default_array_factories, default_value_factories
export factory_name

abstract type AbstractValueFactory end

struct NilFactory <: AbstractValueFactory end

struct FloatScalarFactory <: AbstractValueFactory
    mean::Float64
    std::Float64
end
FloatScalarFactory(; mean::Real=0, std::Real=1) = FloatScalarFactory(Float64(mean), _nonnegative_float(:std, std))

struct IntScalarFactory <: AbstractValueFactory
    lo::Int
    hi::Int
end
function IntScalarFactory(; lo::Integer=-10, hi::Integer=10)
    lo <= hi || throw(ArgumentError("lo must be <= hi"))
    return IntScalarFactory(Int(lo), Int(hi))
end

struct FloatArrayFactory <: AbstractValueFactory
    mean::Float64
    std::Float64
end
FloatArrayFactory(; mean::Real=0, std::Real=1) = FloatArrayFactory(Float64(mean), _nonnegative_float(:std, std))

struct IntArrayFactory <: AbstractValueFactory
    lo::Int
    hi::Int
end
function IntArrayFactory(; lo::Integer=-10, hi::Integer=10)
    lo <= hi || throw(ArgumentError("lo must be <= hi"))
    return IntArrayFactory(Int(lo), Int(hi))
end

struct BoolArrayFactory <: AbstractValueFactory end

function _nonnegative_float(name::Symbol, value::Real)
    out = Float64(value)
    isfinite(out) && out >= 0 || throw(ArgumentError("$name must be finite and nonnegative"))
    return out
end

factory_name(::NilFactory) = :NIL
factory_name(::FloatScalarFactory) = :FLOAT
factory_name(::IntScalarFactory) = :INT
factory_name(::FloatArrayFactory) = :FLOAT_ARRAY
factory_name(::IntArrayFactory) = :INT_ARRAY
factory_name(::BoolArrayFactory) = :BOOL_ARRAY

supports(::NilFactory, ty::Ty) = ty isa TyList
supports(::FloatScalarFactory, ty::Ty) = semantic_typekey(ty) == semantic_typekey(FloatTy)
supports(::IntScalarFactory, ty::Ty) = semantic_typekey(ty) == semantic_typekey(IntTy)
supports(::FloatArrayFactory, ty::Ty) = _array_ends_in(ty, FloatTy)
supports(::IntArrayFactory, ty::Ty) = _array_ends_in(ty, IntTy)
supports(::BoolArrayFactory, ty::Ty) = _array_ends_in(ty, BoolTy)
supports(::AbstractValueFactory, ::Ty) = false

function _array_ends_in(ty::Ty, elem::TyConst)
    ty isa TyArray || return false
    ty.len isa TyNat || return false
    tail = ty.elem
    while tail isa TyArray
        tail.len isa TyNat || return false
        tail = tail.elem
    end
    return semantic_typekey(tail) == semantic_typekey(elem)
end

function provided_types(factory::AbstractValueFactory, universe_types)
    out = Ty[]
    seen = Set{Any}()
    for ty in universe_types
        ty isa Ty || continue
        supports(factory, ty) || continue
        key = semantic_typekey(ty)
        key in seen && continue
        push!(seen, key)
        push!(out, ty)
    end
    return out
end

function provided_types(factories, universe_types)
    out = Ty[]
    seen = Set{Any}()
    for factory in factories
        factory isa AbstractValueFactory || continue
        for ty in provided_types(factory, universe_types)
            key = semantic_typekey(ty)
            key in seen && continue
            push!(seen, key)
            push!(out, ty)
        end
    end
    return out
end

make_value(factory::AbstractValueFactory, ty::Ty; rng::Random.AbstractRNG=Random.default_rng()) =
    make_value(factory, ty, rng)

make_value(::NilFactory, ty::Ty, ::Random.AbstractRNG) =
    supports(NilFactory(), ty) ? Any[] : _unsupported(NilFactory(), ty)

function make_value(factory::FloatScalarFactory, ty::Ty, rng::Random.AbstractRNG)
    supports(factory, ty) || return _unsupported(factory, ty)
    return factory.mean + factory.std * randn(rng)
end

function make_value(factory::IntScalarFactory, ty::Ty, rng::Random.AbstractRNG)
    supports(factory, ty) || return _unsupported(factory, ty)
    return rand(rng, factory.lo:factory.hi)
end

function make_value(factory::FloatArrayFactory, ty::Ty, rng::Random.AbstractRNG)
    supports(factory, ty) || return _unsupported(factory, ty)
    return _array_value(ty, rng, () -> begin
        factory.mean + factory.std * randn(rng)
    end)
end

function make_value(factory::IntArrayFactory, ty::Ty, rng::Random.AbstractRNG)
    supports(factory, ty) || return _unsupported(factory, ty)
    return _array_value(ty, rng, () -> begin
        rand(rng, factory.lo:factory.hi)
    end)
end

function make_value(factory::BoolArrayFactory, ty::Ty, rng::Random.AbstractRNG)
    supports(factory, ty) || return _unsupported(factory, ty)
    return _array_value(ty, rng, () -> begin
        rand(rng, Bool)
    end)
end

function _array_value(ty::Ty, rng::Random.AbstractRNG, leaf)
    ty isa TyArray || return leaf()
    n = (ty.len::TyNat).value
    return [_array_value(ty.elem, rng, leaf) for _ in 1:n]
end

function _unsupported(factory::AbstractValueFactory, ty::Ty)
    throw(ArgumentError("$(typeof(factory)) does not support type $ty"))
end

default_scalar_factories() = (FloatScalarFactory(), IntScalarFactory())
default_array_factories() = (FloatArrayFactory(), IntArrayFactory(), BoolArrayFactory())
default_value_factories() = (NilFactory(), default_scalar_factories()..., default_array_factories()...)

function factory_context(
    factories,
    types;
    rng::Random.AbstractRNG=Random.default_rng(),
    prefix::Symbol=:FACTORY,
)
    consts = Dict{Symbol,Any}()
    consts_ty = Dict{Symbol,TyLike}()
    next_index = 1
    for factory in factories
        factory isa AbstractValueFactory || continue
        for ty in provided_types(factory, types)
            name = _fresh_context_name(consts_ty, Symbol(prefix, "_", factory_name(factory), "_", next_index))
            next_index += 1
            consts[name] = make_value(factory, ty, rng)
            consts_ty[name] = ty
        end
    end
    return (consts=consts, consts_ty=consts_ty)
end

factory_context(factory::AbstractValueFactory, types; kwargs...) =
    factory_context((factory,), types; kwargs...)

function _fresh_context_name(consts_ty::Dict{Symbol,TyLike}, base::Symbol)
    haskey(consts_ty, base) || return base
    i = 1
    while true
        name = Symbol(base, "_", i)
        haskey(consts_ty, name) || return name
        i += 1
    end
end

@testitem "ValueFactories expose supported types and deterministic context values" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, TyArray, TyList, TyNat, typekey

    VF = LambdaRegression.ValueFactories
    Vec3F = TyArray(TyNat(3), FloatTy)
    Vec2I = TyArray(TyNat(2), IntTy)
    BoolMat = TyArray(TyNat(2), TyArray(TyNat(3), BoolTy))
    ListF = TyList(FloatTy)
    universe_types = [FloatTy, IntTy, BoolTy, Vec3F, Vec2I, BoolMat, ListF]

    factories = (
        VF.NilFactory(),
        VF.FloatScalarFactory(mean=2, std=0),
        VF.IntScalarFactory(lo=7, hi=7),
        VF.FloatArrayFactory(mean=1.5, std=0),
        VF.IntArrayFactory(lo=-2, hi=-2),
        VF.BoolArrayFactory(),
    )
    provided = VF.provided_types(factories, universe_types)

    @test any(ty -> typekey(ty) == typekey(FloatTy), provided)
    @test any(ty -> typekey(ty) == typekey(IntTy), provided)
    @test any(ty -> typekey(ty) == typekey(Vec3F), provided)
    @test any(ty -> typekey(ty) == typekey(Vec2I), provided)
    @test any(ty -> typekey(ty) == typekey(BoolMat), provided)
    @test any(ty -> typekey(ty) == typekey(ListF), provided)

    ctx = VF.factory_context(factories, universe_types; rng=Random.MersenneTwister(1), prefix=:FACT)
    @test any(value -> value == 2.0, values(ctx.consts))
    @test any(value -> value == 7, values(ctx.consts))
    @test any(value -> value == [1.5, 1.5, 1.5], values(ctx.consts))
    @test any(value -> value == [-2, -2], values(ctx.consts))
    @test any(value -> value == Any[], values(ctx.consts))
    @test any(value -> value isa Vector && length(value) == 2 && all(row -> row isa Vector && length(row) == 3, value), values(ctx.consts))
    @test length(ctx.consts) == length(ctx.consts_ty)
end

end # module ValueFactories
