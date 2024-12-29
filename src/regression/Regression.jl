"""
    Regression

Frontier regression runner interfaces.

This module defines the **top-level API** for running population-based regression
over typed lambda-calculus programs. The implementation is intentionally staged:
we stabilize the interfaces (components/config/context plumbing) before the
end-to-end evolutionary loop.

Design goals
- High composability: configuration is a `NamedTuple` and components are injected
  as a `NamedTuple`.
- Concurrency readiness: SR/regrssion paths must not depend on global factories
  (e.g. shared `newtyvar` counters or `gensym` naming).
- Const injection: inputs and intermediate values flow through a constant pool
  whose types are visible to individuals.

Status
- `fit!` is a stub (returns `Err(NotImplementedErr())`).
- Thread-safe primitives for SR paths live in `ASTs.Contexts.EvalContext` and
  `TypeSystem.make_tyvar_factory`.
"""
module Regression

using Random
using TestItems

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, make_tyvar_factory, get_type_ctx
using ..ASTs.Contexts: EvalContext, ConstSymbolFactory

export Regressor, SupervisedDataset, FitResult, fit!
export default_config, default_components, default_regressor

struct NotImplementedErr end

"""
    SupervisedDataset(X, Y)

Simple supervised dataset container.

`X` and `Y` are vectors of samples; lengths must match.
"""
struct SupervisedDataset{TX,TY}
    X::Vector{TX}
    Y::Vector{TY}
    function SupervisedDataset(X::Vector{TX}, Y::Vector{TY}) where {TX,TY}
        length(X) == length(Y) || throw(ArgumentError("X and Y must have the same length"))
        return new{TX,TY}(X, Y)
    end
end

"""
    Regressor(target_ty; components=..., config=..., rng=...)

Top-level regression runner object.

- `target_ty` is the program type we are trying to fit (e.g. `A → B`).
- `components` is a composable `NamedTuple` of injected strategies, factories
  and contexts.
- `config` is a composable `NamedTuple` of hyperparameters.

This type is intentionally lightweight and mostly acts as a stable interface.
"""
struct Regressor
    target_ty::Ty
    components::NamedTuple
    config::NamedTuple
    rng::AbstractRNG
end

"""
    FitResult

Result container for `fit!`.
"""
struct FitResult
    best
    status::Symbol
    message::String
    history::Vector{Any}
end

"""
    default_config(; kwargs...) -> NamedTuple

Load a default configuration. This function makes strong assumptions and is
intended only as a convenience.
"""
function default_config(;
    population_size::Int=200,
    generations::Int=50,
    error_threshold::Float64=1e-6,
    init_mutation_rounds::Int=1,
    max_reduce_steps::Int=64,
    parallel=(; enabled=true, mode=:individual, schedule=:dynamic),
    kwargs...
)
    base = (;
        population_size,
        generations,
        error_threshold,
        init_mutation_rounds,
        max_reduce_steps,
        parallel,
    )
    return merge(base, (;kwargs...))
end

"""
    default_components(; kwargs...) -> NamedTuple

Load default component wiring for regression.

This is a **loader only**: it does not run any business logic. The returned
components are designed to be easy to override via `merge`.
"""
function default_components(;
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    input_slots::Vector{Symbol}=Symbol[],
    input_types::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(),
    kwargs...
)
    global_consts_ty === nothing && (global_consts_ty = get_type_ctx(global_consts))

    # Thread-safe evaluation context constructor:
    # writes go to `overlay` only; `globals` is assumed read-only.
    make_eval_context = (;
        overlay::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        locals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
        globals::Dict{Symbol,Any}=global_consts,
        symfactory::ConstSymbolFactory=const_symfactory_ctor(),
    ) -> EvalContext(overlay=overlay, locals=locals, globals=globals, symfactory=symfactory)

    base = (;
        global_consts,
        global_consts_ty,
        input_slots,
        input_types,
        tyfactory_ctor,
        const_symfactory_ctor,
        make_eval_context,
    )
    return merge(base, (;kwargs...))
end

"""
    default_regressor(target_ty::Ty; kwargs...) -> Regressor

Convenience constructor: loads `default_config` and `default_components`.
"""
function default_regressor(target_ty::Ty;
    rng::AbstractRNG=MersenneTwister(2000),
    config::NamedTuple=default_config(),
    components::NamedTuple=default_components(),
    kwargs...
)
    reg = Regressor(target_ty, components, config, rng)
    return reg
end

"""
    fit!(reg::Regressor, data::SupervisedDataset; callback=nothing) -> Union{FitResult,Err}

Stub: validates obvious invariants and returns `Err(NotImplementedErr())`.
"""
function fit!(reg::Regressor, data::SupervisedDataset; callback=nothing)
    length(data.X) == length(data.Y) || return Err(ArgumentError("X and Y must have the same length"))
    return Err(NotImplementedErr())
end

@testitem "Regression default_components provides non-global factories" begin
    using LambdaRegression.Regression: default_components
    using LambdaRegression.TypeSystem: make_tyvar_factory
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory

    c = default_components()
    @test c.tyfactory_ctor === make_tyvar_factory
    f = c.tyfactory_ctor()
    @test f(0).id == 1
    sf = c.const_symfactory_ctor()
    @test sf isa ConstSymbolFactory
end

end # module Regression

