"""
    Mutations.Populations

Population-level variation interfaces and baseline typed crossover.

This submodule provides minimal `Individual`/`Population` containers and typed
binary recombination. `TypedSubtreeSwapCrossover` analyzes each parent once,
routes a finite set of type- and scope-compatible site pairs without
replacement, and returns two children only when both validate and both are
genuine changes. Parent trees and constant dictionaries are never modified.

The core operator accepts ordinary lambda AST nodes. Extension nodes such as
`FunctionDefNode` return `Err(UnsupportedCrossoverNodeErr(...))`; implement a
separate `AbstractCrossover` adapter when an extension has additional binders or
typing rules.

Site routing is injectable without replacing rewrite logic:

```julia
import LambdaRegression.Mutations.Populations:
    crossover_site_weight, crossover_pair_weight
using LambdaRegression.Mutations.Populations:
    AbstractCrossoverSelectionPolicy, CrossoverSite,
    TypedSubtreeSwapCrossover

struct PreferDeepSites <: AbstractCrossoverSelectionPolicy end
crossover_site_weight(::PreferDeepSites, site::CrossoverSite, side::Symbol) =
    1 + site.depth
crossover_pair_weight(::PreferDeepSites, left::CrossoverSite, right::CrossoverSite) =
    left.ty == right.ty ? 2.0 : 1.0

op = TypedSubtreeSwapCrossover(selection=PreferDeepSites())
```

Callbacks receive borrowed, attempt-local read-only views: do not mutate or
retain their vector fields. Callbacks must return finite, nonnegative weights.
Zero excludes a site or pair; an empty positive-weight pair set returns
`Err(NoEligibleCrossoverPairErr())`. Invalid weights and callback exceptions are
programmer/configuration errors and propagate. Use
`PathCrossoverSelectionPolicy` for exact deterministic routing in tests or
specialized operators.

Positive pair masses are represented in log space and sampled once as a
Gumbel-ranked weighted-without-replacement order. If every finite pair is tried
without a valid child, the operator returns
`Err(CrossoverPairPoolExhaustedErr(attempts))`; reaching `max_try` while pairs
remain uses `CrossoverMaxTryReachedErr`.

Selection policies control routing mass only. Type/scope compatibility,
global/local namespace hygiene, retained application roles, and final W
validation are fixed correctness boundaries. Implement another
`AbstractCrossover` when those semantics must differ instead of weakening the
shared typed-swap kernel.

`require_same_depth=false` permits different lexical depths, but external
captures still map by equal De Bruijn level and every captured level must exist
at the destination. `require_compatible_binders=true` compares only binders the
subtree actually captures, including maximally generalized let schemes. Final W
inference and retained ordinary/let-like application-role checks remain
mandatory regardless of these routing filters.

Threading / determinism constraints:
- Crossover implementations must not use global RNG state; callers pass `rng`.
- Avoid shared mutable factories (e.g. global type-var counters or `gensym`) in
  SR paths. Concrete crossover operators accept injectable constructors such as
  `tyfactory_ctor` and `const_symfactory_ctor`.

Notes
- Some literature distinguishes "mutation" (unary variation) from "crossover"
  (binary recombination). In GP practice both are often grouped under variation
  operators. Here we place crossover in the population-level namespace to avoid
  confusion with `mutate_tree`'s single-individual contract.
- "GC pruning" in this repository currently refers primarily to **per-individual
  constant dictionary cleanup**, implemented by `Mutations.prune_local_consts!`.
  Population-level pruning ideas (deduplication, cache eviction, niching) live
  in the development notes.
"""
module Populations

using Random

using ...ASTs.NodeASTs: AstTree
using ...TypeSystem: Ty, TyLike
using ...Utils: Err
import ..prune_local_consts!

export Individual, Population
export AbstractCrossover, crossover
export AbstractCrossoverSelectionPolicy, UniformCrossoverSelectionPolicy
export CrossoverSite, PathCrossoverSelectionPolicy
export crossover_site_weight, crossover_pair_weight
export TypedSubtreeSwapCrossover, CrossoverPairPoolExhaustedErr
export NoEligibleCrossoverPairErr, UnsupportedCrossoverNodeErr
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
