"""
    Mutations.Populations

Population-level variation interfaces, typed recombination, and donor sampling.

This submodule provides typed binary recombination over immutable `Program`
payloads. `TypedSubtreeSwapCrossover` analyzes each parent once,
routes a finite set of type- and scope-compatible site pairs without
replacement, and returns two children only when both validate and both are
genuine changes. Parent trees and constant dictionaries are never modified.

The directional `recombine` seam uses exact demand-from-offer retrieval,
binder-aware transport, and complete-child validation for ordinary lambda and
first-class `FunctionDef` terms. The binary `crossover` seam validates both
children atomically. Unsupported extension nodes have no implicit compatibility
route.

`DonorPool` is a separate bounded sampling layer. It stores explicit fragment
paths from caller-selected individuals and supports uniform or adaptive
queue-like retention. Pool feedback changes only sampling weights and eviction,
never the retrieval relation.

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

using ...Programs: Program

export Program
export AbstractCrossover, crossover, recombine
export AbstractCrossoverSelectionPolicy, UniformCrossoverSelectionPolicy
export CrossoverSite, PathCrossoverSelectionPolicy
export crossover_site_weight, crossover_pair_weight
export TypedSubtreeSwapCrossover, CrossoverPairPoolExhaustedErr
export NoEligibleCrossoverPairErr
export CrossoverAnswerBudgetExhaustedErr
export DonorPool, DonorPoolConfig, DonorPoolOutcome, DonorPoolSelection
export DonorPoolAttempt, EmptyDonorPoolErr
export admit_donor!, admit_random_donors!, sample_donor, record_outcome!
export donor_pool_stats, donor_weights, recombine_from_pool

"""
    AbstractCrossover

Interface for binary recombination between two individuals.
"""
abstract type AbstractCrossover end

"""
    crossover(op::AbstractCrossover, a::Program, b::Program;
        target_a, target_b, rng, kwargs...) -> Union{Tuple{Program,Program},Err}

Apply crossover to produce (typically) two offspring from programs `a` and `b`.
Root requirements are explicit because they belong to the surrounding task.
"""
function crossover(
    ::AbstractCrossover,
    ::Program,
    ::Program;
    target_a::Ty,
    target_b::Ty,
    rng::AbstractRNG=MersenneTwister(2000),
    kwargs...,
)
    error("crossover not implemented")
end

"""
    recombine(op::AbstractCrossover, recipient::Program, donor::Program;
        recipient_target, donor_target, rng=MersenneTwister(2000), kwargs...) -> Program or Err

Apply the one-way population recombination seam.  `recipient` supplies the
target program and `recipient_target` supplies its root requirement; `donor`
supplies the fragment and `donor_target` its own root requirement. Concrete
operators implement the semantics for the variation family they support.
The existing [`crossover`](@ref) tuple interface remains independent of this
directional seam. This generic fallback does not impose a directional method
on every `AbstractCrossover`; consumers requiring one direction surface an
explicit interface error when an operator has not implemented it.
"""
function recombine(
    op::AbstractCrossover,
    ::Program,
    ::Program;
    recipient_target::Ty,
    donor_target::Ty,
    rng::AbstractRNG=MersenneTwister(2000),
    kwargs...,
)
    error("$(typeof(op)) has no directional recombine method; directional recombine required")
end

include("TypedCrossoverRetrieval.jl")
include("ScopedRetrievalSolver.jl")
include("OpenTypedRetrieval.jl")
include("ProgramAnalysisCrossover.jl")
include("SnapshotOpenRetrieval.jl")
include("TypedCrossoverTransport.jl")
include("PopulationCrossover.jl")
include("TypedCrossoverEngine.jl")
include("DonorPool.jl")

end # module Populations
