"""
    Mutations.Populations

Population-level variation interfaces and baseline typed crossover.

This submodule provides:
- minimal containers for `Individual` and `Population`, and
- a population-level namespace for **binary recombination** (crossover).

Threading / determinism constraints:
- Crossover implementations must not use global RNG state; callers pass `rng`.
- Avoid shared mutable factories (e.g. global type-var counters or `gensym`) in
  SR paths. Concrete crossover operators therefore accept injectable constructors
  like `tyfactory_ctor` and `const_symfactory_ctor`.

Notes
- Some literature distinguishes "mutation" (unary variation) from "crossover"
  (binary recombination). In GP practice both are often grouped under variation
  operators. Here we place crossover in the population-level namespace to avoid
  confusion with `mutate_tree`'s single-individual contract.
- "GC pruning" in this repository currently refers primarily to **per-individual
  constant dictionary cleanup**, implemented by `Mutations.prune_local_consts!`.
  Population-level pruning policies (deduplication, cache eviction, niching)
  remain caller-configurable.
"""
module Populations

using Random

using ...ASTs.NodeASTs: AstTree
using ...TypeSystem: Ty, TyLike
using ...Utils: Err
import ..prune_local_consts!

export Individual, Population
export AbstractCrossover, crossover
export TypedSubtreeSwapCrossover
export prune!

"""
    Individual

Minimal individual container for population-based runs.
"""
struct Individual
    tree::AstTree
    target_ty::Ty
    local_consts::Dict{Symbol,Any}
    local_consts_ty::Dict{Symbol,TyLike}
    fitness::Union{Nothing,Float64}
    meta::NamedTuple
end

function Individual(tree::AstTree, target_ty::Ty;
    local_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    local_consts_ty::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    fitness::Union{Nothing,Float64}=nothing,
    meta::NamedTuple=(;),
)
    return Individual(tree, target_ty, local_consts, local_consts_ty, fitness, meta)
end

"""
    Population(individuals)
"""
struct Population
    individuals::Vector{Individual}
end

"""
    AbstractCrossover

Interface for binary recombination between two individuals.
"""
abstract type AbstractCrossover end

"""
    crossover(op::AbstractCrossover, a::Individual, b::Individual; rng, kwargs...) -> Union{Tuple{Individual,Individual},Err}

Apply crossover to produce (typically) two offspring from parents `a` and `b`.
Implementations must preserve the individuals' `target_ty` constraints.
"""
function crossover(::AbstractCrossover, ::Individual, ::Individual; rng::AbstractRNG=MersenneTwister(2000), kwargs...)
    error("crossover not implemented")
end

"""
    prune!(ind::Individual) -> Nothing

Prune unreferenced local constants for this individual.
"""
function prune!(ind::Individual)
    prune_local_consts!(ind.tree, ind.local_consts, ind.local_consts_ty)
    return nothing
end

include("PopulationCrossover.jl")

end # module Populations
