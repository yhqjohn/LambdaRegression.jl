"""
    Mutations

Typed, type-stable AST mutation utilities.

This module provides mutation operators for `ASTs.NodeASTs.AstTree` that preserve
the *target type* under `TypeSystem.infer` (up to `TypeSystem.iscompatible`).

Design goals
- Correctness-first: mutations must preserve typing and binding safety.
- Low coupling: optional optimizations (site indexing, eligibility cache, bloat
  control, adaptive weighting) are implemented as swappable providers/policies.
- TestItems: regression tests live next to the implementation.

## Core abstractions

This module models mutation as choosing a pair `(mutation, site)` and applying it:

1. **Site provider** (`AbstractSiteProvider`):
   - Computes and stores a list of mutation sites for a given tree, including
     parent pointers, scope depth, and the inferred type of each node.
   - `IndexedProvider` builds a practical baseline index (`sites_all` + `sites_by_kind`).

2. **Pair chooser** (`AbstractPairChooser`):
   - Selects a mutation operator and a compatible site.
   - `WeightedPairChooser` implements a practical baseline: prefer high-success
     operators (constants) and only select from eligible sites.
   - Optional eligibility caching is scoped to *one* `mutate_tree` call.

3. **Mutation operator** (`AbstractMutation`):
   - Declares which kinds of sites it needs (`required_site_kinds`),
     filters sites (`is_site_eligible`), and performs the mutation (`apply`).

The top-level entry `mutate_tree` runs a retry loop:
- build sites (via provider),
- choose `(mutation, site)` (via chooser),
- apply mutation,
- validate type invariant (`infer` + `iscompatible`),
- optionally prune unreferenced local constants (`prune_local_consts!`).

## Invariants

- **Type stability**: the mutated tree must satisfy `iscompatible(infer(tree, type_ctx), target_ty)`.
- **Binding safety**: mutations must not introduce unbound `IndexNode`s (validated by inference).
- **Constant context consistency**: any local `ConstNode(name)` must have matching
  entries in `local_consts` and `local_consts_ty`.

## Context discipline

`mutate_tree` uses three dictionaries:
- `global_consts_ty::Dict{Symbol,TyLike}`: operator/prelude typing (shared).
- `local_consts::Dict{Symbol,Any}`: per-individual constant values (mutable).
- `local_consts_ty::Dict{Symbol,TyLike}`: types for names in `local_consts`.

By default, `copy_ctx=true` makes mutation transactional: we search using scratch
copies and only commit to the caller dictionaries on success. This avoids the
"shared constants all change together" pitfall in population-based runs.

Quickstart
```julia
using Random
using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
using LambdaRegression.TypeSystem: TyFun, IntTy
using LambdaRegression.Mutations: mutate_tree, ConstMutation, PointMutation
using LambdaRegression.TypeSystem: TyLike

tree = AstTree(AbsNode(:x, IndexNode(0))) # identity
target_ty = TyFun(IntTy, IntTy)

mutated = mutate_tree(
    tree,
    target_ty;
    rng=MersenneTwister(1),
    global_consts_ty=Dict{Symbol,TyLike}(),
    local_consts_ty=Dict{Symbol,TyLike}(),
    local_consts=Dict{Symbol,Any}(),
    mutations=[PointMutation(), ConstMutation()],
)
```

## Generated subtree replacement

`SubtreeReplacementMutation` replaces an eligible subtree `S : A` with a fresh
canonical term `G : A` generated under the lambda binders visible at `S`. It is
not part of the default mutation list; opt in when a caller wants generation-
backed edits.

The public knobs are intentionally small and composable:

- `site_lock`: decides which sites are never replacement roots. The default
  `DefaultSiteLockPolicy(spine_only=true)` avoids De Bruijn leaves,
  application function heads, and non-spine internals. Use
  `DefaultSiteLockPolicy(spine_only=false)` or a custom `is_site_locked` method
  to open more sites.
- `site_selection`: assigns per-site weights. `ExponentialSiteSelectionPolicy`
  is the default; `PathSiteSelectionPolicy(path)` is useful for fixtures. Site
  selection does not override `site_lock`, so pass a permissive lock policy when
  a fixture intentionally targets an otherwise locked site.
- `replacement_generator`: owns replacement construction. The default
  `CanonicalReplacementGenerator(include_local_consts=false)` uses
  `generate_canonical_scoped` and treats old local constants as reserved
  context rather than preferred source heads. Implement
  `generate_replacement(::MyGenerator, site, env, ctx, generation_kwargs)` to
  inject another generator, or wrap any compatible scoped callable with
  `ScopedReplacementGenerator(callable)`.
- `generation_kwargs`: forwarded to `generate_canonical_scoped`. Defaults are
  deliberately small (`sampling_depth_limit=2`, `sampling_node_budget=24`,
  `guarded_expansion_depth=0`) so repeated mutation batches stay fast; callers
  can pass deeper budgets, custom providers, or custom weights explicitly. If
  initialization used a non-default canonical catalog, pass its `providers` and
  `weights` here too; the mutation core does not bake rich catalog definitions into
  its defaults.

Example:
```julia
mutation = SubtreeReplacementMutation(;
    site_lock=DefaultSiteLockPolicy(spine_only=false),
    site_selection=ExponentialSiteSelectionPolicy(scope_bias=-0.5),
    replacement_generator=CanonicalReplacementGenerator(include_local_consts=true),
    generation_kwargs=(; sampling_depth_limit=3, sampling_node_budget=48),
)
```

`mutate_tree(...; type_inference=checker)` replaces the four-argument type
checker used for both site decoration and post-mutation validation. The callable
receives `(tree, type_ctx, tyfactory, meta_or_nothing)`. Omitting it preserves the
ordinary core `infer` path; procedural layers can inject their lowering-aware
checker without coupling this module to that AST extension.

## Direct composition

`UnaryCompositionMutation()` wraps a selected `S : T` with a monomorphic global
endomorphism `H : T → T`. It is the low-cost direct move for forms such as
`SQR(x)` or `TSCALE(head)`. `BinaryCompositionMutation()` preserves `S` as one
argument of a compatible curried binary global and generates the other sibling.
Use their site-selection policies and `WeightedPairChooser` weights to change
routing without changing either rewrite.

## Constant mutation policies

`ScalarConstMutation` and `ArrayConstMutation` keep the legacy `σ` constructor
and field, and also accept a `ConstValueMutationPolicy`. The default keeps the
old small additive/scale-like behavior, while custom policies can select
`:bias`, `:scale`, or `:rerandomize`. Array policies can choose broadcast ranks
over finite `TyArray` shapes whose leaf type matches the runtime array element
type:

```julia
mutation = ArrayConstMutation(;
    policy=ConstValueMutationPolicy(
        modes=(:bias, :scale),
        array_ranks=(:global, :axis, :outer, :full),
        σ_bias=0.1,
        σ_scale=0.05,
    ),
)
```

Scale uses a positive log-normal factor `exp(σ_scale * randn(rng))`. The policy
object is deliberately small so callers can replace value-level behavior without
changing site selection, pair choice, or mutation validation.

## Let-region mutations

Let-like regions are recognized from the core encoding
`AppNode(AbsNode(...), rhs)`; callers do not need a persistent `ScopeNode`.
Three accepted operators share one attempt-local binder-resolved view:

- `LetHoistMutation()` moves one binding across one legal parent. Set
  `min_bindings` and `max_bindings` above one to opt into consecutive,
  dependency-closed block Hoist; block Hoist is not in the default preset.
- `LetSinkMutation()` moves a binding to the deepest region containing every
  use. `allow_trivial=true` also admits otherwise unhelpful local moves.
- `LetPruneMutation()` removes an explicitly unused binding.

They require no mutation-specific context fields. Capture safety, De Bruijn
rebasing, and the distinction between ordinary applications and maximally
generalized let-like applications are checked by the shared rewrite kernel;
the usual final `infer`/`iscompatible` check remains authoritative. An operator
can fail normally when no legal region exists, when a move would alter a
retained application role, or when final validation rejects the result.

The mutation set, routing weights, site provider, and chooser are independent
injection points:

```julia
using LambdaRegression.Mutations

let_mutations = AbstractMutation[
    LetHoistMutation(),
    LetSinkMutation(),
    LetPruneMutation(),
]
let_weights = Dict{Type,Float64}(
    LetHoistMutation => 0.03,
    LetSinkMutation => 0.03,
    LetPruneMutation => 0.02,
)
chooser = WeightedPairChooser(weights=let_weights, cache_eligibility=true)

mutated = mutate_tree(
    tree,
    target_ty;
    mutations=let_mutations,
    chooser,
    provider=IndexedProvider(),
    # rng and constant contexts as in the quickstart above
)
```

Passing a custom `AbstractPairChooser` can replace routing entirely; passing a
custom `AbstractSiteProvider` can replace indexing. Omitting these operators
from `mutations` avoids their scoped analysis and candidate scans.

## Binding insertion and explicit inline

`SourceBindingInsertionMutation()` copies one already typed subtree into an
intentionally unused, maximally generalized let-like binding. It is the only
binding-insertion operator in the ordinary preset, with routing weight `0.015`.
The chooser selects the source site; `host_weight(source_site, host_path,
distance, env, ctx)` then weights its finite capture-safe ancestor hosts. A zero
weight excludes a host. Omitting this operator removes its source/host scan. If
Hoist, Sink, Prune, or another scoped family remains enabled, that family may
still build the shared scoped view; omitting every scoped family avoids it.

`LetInlineMutation()` exposes the inverse let-like β step as an independently
routable, path-authoritative operator. It remains opt-in because the default
`BetaReduceMutation()` already reduces direct-`Abs` applications. Let RHS and
body replacement use `SubtreeReplacementMutation` with its injected site lock,
path-selection policy, and replacement generator.

All three rewrites resolve the selected node from the attempt-local `site.path`;
stale object identity cannot redirect them. A source or generated unused binding
is still a call-by-value application `(λx. body) rhs`: `rhs` may therefore
diverge, time out, or raise a host error even though `x` is unused. These
operators preserve typing and lexical scope, not termination or totality.

`GeneratedBindingInsertionMutation(rhs_types; ...)` is also opt-in. `rhs_types`
must be an explicit finite collection of closed monotypes and/or fully quantified
`TyScheme`s. The operator samples one demand with `rhs_type_weight(rhs_ty, host,
env, ctx)`, asks `replacement_generator` for an RHS, independently checks that
RHS against the selected demand, and only then commits the new let-like binding
and staged local constants. Captured host binders remain rigid during this check;
quantified variables in the selected scheme remain genuinely polymorphic. RHS
generalization follows `FTV(rhs) ∖ FTV(environment)`, and enclosing binder types
are reconstructed from the current tree and authoritative path rather than
trusted from caller-supplied `Site` metadata.

```julia
fresh = LambdaRegression.TypeSystem.make_tyvar_factory()
a = fresh(0)

source = SourceBindingInsertionMutation(
    host_weight=(_, _, distance, _, _) -> exp(-0.4 * distance),
)

generated = GeneratedBindingInsertionMutation(
    TyLike[IntTy, TyScheme([a], TyFun(a, a))];
    replacement_generator=CanonicalReplacementGenerator(),
    rhs_type_weight=(rhs_ty, _, _, _) -> rhs_ty === IntTy ? 2.0 : 1.0,
    generation_kwargs=(;
        sampling_depth_limit=3,
        sampling_node_budget=48,
        guarded_expansion_depth=0,
        providers=my_providers,
        weights=my_generation_weights,
    ),
)

mutations = AbstractMutation[source, generated, LetInlineMutation()]
chooser = WeightedPairChooser(
    weights=Dict{Type,Float64}(
        SourceBindingInsertionMutation => 0.02,
        GeneratedBindingInsertionMutation => 0.01,
        LetInlineMutation => 0.01,
    ),
    cache_eligibility=true,
)

rhs_or_body_edit = SubtreeReplacementMutation(
    site_lock=DefaultSiteLockPolicy(spine_only=false),
    site_selection=PathSiteSelectionPolicy(rhs_or_body_path),
    replacement_generator=CanonicalReplacementGenerator(),
)
```

Here `my_providers` and `my_generation_weights` are caller-owned canonical
generation configuration. The constructor snapshots and validates the type
catalog. `generation_kwargs` cannot override the caller RNG, host scope,
constant dictionaries, or complete generation config. Custom RHS construction
extends `generate_binding_rhs(generator, rhs_ty, host, env, ctx,
generation_kwargs)` on an `AbstractReplacementGenerator`; the mutation kernel
continues to own finite catalog routing, transactional constants, independent
RHS verification, host insertion, and final root validation.

Expected failure is ordinary: a source may have no legal host, all injected
weights may be zero, generation may fail within its finite budget, or either RHS
or root validation may reject the candidate. Negative, non-finite, and otherwise
invalid policy weights are configuration errors returned as failed attempts;
they never silently broaden the candidate set.

## Existing-binding Graft

`GraftMutation()` adds a new use of an existing maximally generalized let-like
binding:

```text
let x = D in B[R]  ↦  let x = D in B[x]
```

It neither creates a binding nor copies/moves the RHS. A legal pair requires the
recipient to lie in the binding body, not already depend on that binder, differ
from the RHS under scope-aware α normalization, preserve retained HM application
roles, and accept a fresh instantiation of the binding scheme at the recipient.
Final whole-tree inference remains authoritative. This is a semantic-changing
operator and may activate recursion or divergence; it is opt-in and absent from
the ordinary preset.

Routing is caller-owned and evaluated once per attempt-local candidate:

```julia
graft = GraftMutation(
    recipient_weight=(site, env, ctx) -> exp(-0.25 * site.depth),
    donor_weight=(binding, site, env, ctx) ->
        binding.name === :preferred ? 3.0 : 1.0,
)

mutated = mutate_tree(
    tree,
    target_ty;
    mutations=AbstractMutation[graft],
    chooser=WeightedPairChooser(
        weights=Dict{Type,Float64}(GraftMutation => 1.0),
        cache_eligibility=true,
    ),
    provider=IndexedProvider(),
    # rng and constant contexts as in the quickstart above
)
```

`GraftSite` exposes the read-only attempt-local recipient path, node, type, and
lexical depth. The likewise read-only `GraftBinding` exposes the complete let
path, RHS path, binder name, and generalized scheme. Zero weight excludes a site or pair; negative,
non-real, or non-finite weights are configuration errors. No extra context field
is required beyond `mutate_tree`'s ordinary typing context. The current resolver
accepts the core lambda AST and rejects unsupported procedural nodes finitely;
procedural-aware Graft belongs to that extension. Omitting `GraftMutation` from
`mutations` avoids its scoped analysis entirely. As with every injected policy,
finite-time claims assume callbacks return and do not mutate these views.

## Product packing and unpacking

`ProductPackMutation(capability)` replaces two adjacent, independent let-like
bindings with one product binding. `ProductUnpackMutation(capability)` is its
strict syntactic inverse: it only accepts a product binding whose uses are
exclusively exact left/right projections. Component order is stable; neither
operator reassociates nested products or performs general product normalization.
They are opt-in and absent from the ordinary preset, so omitting them avoids
their scoped candidate analysis.

This is a type-preserving search mutation, not an evaluator-equivalence pass.
The ordinary host `PAIR` is strict, while an unused let RHS can remain
unevaluated under normal-order beta reduction. Packing can therefore change
termination behavior, for example when an unused component diverges. Use this
family only where that semantic-changing search move is intended.

The required capability supplies the three constant heads used by the rewrite.
The convenience capability below uses the type system's ordinary primitives;
callers must still inject matching types and runtime values into the mutation
context:

```julia
using LambdaRegression.Mutations
using LambdaRegression.TypeSystem: FST, PAIR, SND, TyLike, get_type_ctx

capability = PrimitiveProductCapability(
    pair_head=:MAKE_PAIR,
    left_head=:LEFT_OF,
    right_head=:RIGHT_OF,
)
values = Dict{Symbol,Any}(
    :MAKE_PAIR => PAIR,
    :LEFT_OF => FST,
    :RIGHT_OF => SND,
)
types = Dict{Symbol,TyLike}(get_type_ctx(values))
mutations = AbstractMutation[
    ProductPackMutation(capability),
    ProductUnpackMutation(capability),
]
```

Custom primitive-head mappings subtype `AbstractProductCapability` and define
`product_primitive_heads(::MyCapability)`. The current protocol does not replace
the product AST shape or its evaluation strategy. The shared kernel continues
to own binder resolution, strict matching, materialization, and final type
validation.
"""
module Mutations

using Random
using TestItems

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, TyConst, TyFun, TyList, TyVar, TyArray, TyNat, instantiate, infer, iscompatible, newtyvar, BoolTy, IntTy, FloatTy
using ..ASTs.NodeASTs
using ..ASTs.CSE: canonicalize, eliminate_expression!, eliminate_node!, repeated_nodes
using ..CanonicalGeneration: generate_canonical_scoped, GenerationResult
import ..ASTs.NodeASTs: apply_node!, shift_free_variables!
using ..ASTs.Contexts: new_constant!, ConstSymbolFactory
using ..Trees: traverse, PointerState, StackState, getstate, Break, setroot!, setchild!, children

export mutate_tree, default_mutations
export AbstractMutation, required_site_kinds, is_site_eligible, apply
export AbstractSiteProvider, IndexedProvider
export AbstractPairChooser, WeightedPairChooser
export BloatPolicy, AdaptivePolicy
export MutationAttemptRecorder, mutation_recorder_summary
export prune_local_consts!
export ConstMutation, ScalarConstMutation, BoolConstMutation, ArrayConstMutation, ConstValueMutationPolicy
export ParameterIntroductionMutation
export PointMutation
export SubtreeReplacementMutation, ContextualExpansionMutation
export UnaryCompositionMutation, BinaryCompositionMutation
export AbstractSiteLockPolicy, DefaultSiteLockPolicy
export AbstractSiteSelectionPolicy, ExponentialSiteSelectionPolicy, PathSiteSelectionPolicy
export LocalCallSiteSelectionPolicy
export AbstractReplacementGenerator, CanonicalReplacementGenerator, ScopedReplacementGenerator
export is_site_locked, site_selection_weight, generate_replacement
export BetaReduceMutation, EtaReduceMutation, EtaExpandMutation, CSEMutation
export LetPruneMutation, LetHoistMutation, LetSinkMutation
export SourceBindingInsertionMutation, GeneratedBindingInsertionMutation, LetInlineMutation
export GraftMutation, GraftSite, GraftBinding
export AbstractProductCapability, PrimitiveProductCapability
export ProductPackMutation, ProductUnpackMutation
export product_primitive_heads
export generate_binding_rhs
export CondAppPullMutation, CondAppPushMutation
export Populations

struct MaxTryReachedErr end
struct NoEligibleMutationErr end
struct NoEligibleSiteErr end
struct TypeInvariantErr
    expected::Ty
    got::Ty
end
struct UnsupportedConstTyErr
    ty::Ty
end
struct BloatExceededErr
    max_nodes::Int
    max_nodes_delta::Int
end

"""
    Site

Mutation site metadata derived from a typed traversal of an `AstTree`.

`Site` is the *adapter* between typing/scope information and mutation operators.

# Fields
- `path`: stable 1-based child-index path from root (`Int[]` means root)
- `node`: current AST node
- `parent`: parent AST node (or `nothing` for root)
- `child_index`: 1-based child index in `parent` (from `PointerState`)
- `depth`: number of enclosing lambdas (scope depth)
- `kind`: coarse kind for eligibility routing:
  - `:const_local`, `:const_global`, `:index`, `:app`, `:abs`, `:any`
- `ty`: inferred monotype lower-bound for the node (from `infer(..., meta=...)`)
- `binder_types`: enclosing binder argument types, ordered outer → inner
"""
struct Site
    path::Vector{Int}
    node::AstNode
    parent::Union{Nothing,AstNode}
    child_index::Int                 # 1-based child index in `parent` (PointerState)
    depth::Int                       # number of enclosing lambdas
    kind::Symbol                     # :const_local, :const_global, :index, :app, :abs, :any
    ty::Ty                           # node lower-bound type (monotype)
    binder_types::Vector{Ty}         # enclosing binders (outer -> inner)
end

struct PairChoice
    mutation::Any
    site::Site
end

"""
    AbstractMutation

Abstract base type for typed mutation operators.

## Required interface
- `required_site_kinds(m) -> Vector{Symbol}`
- `is_site_eligible(m, site, env, ctx) -> Bool`
- `apply(m, tree, site, env, ctx) -> Union{AstTree,Err}`

## Optional hooks
- `base_weight(m, ctx) -> Float64`: coarse weight (combined with chooser weights)
- `site_weight(m, site, env, ctx) -> Float64`: per-site weight

Notes
- `env::MutationEnv` carries `rng`, `target_ty`, `type_ctx`, policies and (optional) stats.
- `ctx::NamedTuple` is the runtime context built by `mutate_tree` (constants + generator config).
"""
abstract type AbstractMutation end

"""
    required_site_kinds(m::AbstractMutation) -> Vector{Symbol}

Declare which site kinds an operator wants to consider (coarse routing).
Defaults to `[:any]`.
"""
required_site_kinds(::AbstractMutation) = Symbol[:any]

"""
    is_site_eligible(m::AbstractMutation, site::Site, env::MutationEnv, ctx::NamedTuple) -> Bool

Fine-grained eligibility predicate. This is where operators enforce:
- node type matches (e.g. only `IndexNode`)
- type constraints (`site.ty`, binder compatibility)
- context constraints (e.g. local constant presence)
"""
is_site_eligible(::AbstractMutation, ::Site, ::Any, ::NamedTuple) = true

"""
    apply(m::AbstractMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple) -> Union{AstTree,Err}

Perform the mutation at `site`. Implementations should mutate `tree` in-place and
return the (possibly modified) `AstTree` on success.
"""
apply(::AbstractMutation, ::AstTree, ::Site, ::Any, ::NamedTuple) = error("apply not implemented")

"""
    base_weight(m::AbstractMutation, ctx::NamedTuple) -> Float64

Optional coarse weight hook. `WeightedPairChooser` combines this with its own
type-based weights.
"""
base_weight(::AbstractMutation, ::NamedTuple) = 1.0

"""
    site_weight(m::AbstractMutation, site::Site, env::MutationEnv, ctx::NamedTuple) -> Float64

Optional per-site weight hook. Useful for depth bias, anti-bloat schedules, etc.
Defaults to 1.0.
"""
site_weight(::AbstractMutation, ::Site, ::Any, ::NamedTuple) = 1.0

"""
    AbstractSiteProvider

Abstract base type for site providers.

Site providers are responsible for:
- running `infer(tree, type_ctx; meta=...)` once (if desired),
- traversing the AST and collecting `Site` records,
- (optionally) indexing sites by `kind` for fast access.

## Required interface
- `reset!(provider)`
- `prepare!(provider, tree, type_ctx, env, ctx) -> Union{Nothing,Err}`
- `sites(provider) -> Vector{Site}`

## Optional interface
- `sites_by_kind(provider, kind::Symbol) -> Vector{Site}` (defaults to empty; chooser must fallback)
"""
abstract type AbstractSiteProvider end

reset!(::AbstractSiteProvider) = nothing
prepare!(::AbstractSiteProvider, ::AstTree, ::Dict{Symbol,<:TyLike}, ::Any, ::NamedTuple) = error("prepare! not implemented")
sites(::AbstractSiteProvider) = error("sites not implemented")
sites_by_kind(::AbstractSiteProvider, ::Symbol) = Site[]

"""
    AbstractPairChooser

Abstract base type for choosing `(mutation, site)` pairs.

Choosers may implement:
- operator weighting,
- site weighting,
- eligibility caching,
- (optional) adaptive weighting via `env.adaptive_policy`.

## Required interface
- `choose_pair(chooser, mutations, provider, tree, type_ctx, env, ctx) -> Union{PairChoice,Err}`
"""
abstract type AbstractPairChooser end
choose_pair(::AbstractPairChooser, ::Vector{<:AbstractMutation}, ::AbstractSiteProvider, ::AstTree, ::Dict{Symbol,<:TyLike}, ::Any, ::NamedTuple) = error("choose_pair not implemented")

"""
    AdaptivePolicy(; enabled=false, ewma_alpha=0.2, floor=0.05)

Optional *within-call* adaptive weighting policy.

When `enabled=true`, `WeightedPairChooser` multiplies mutation mass by a success
hint estimated from an EWMA of per-mutation successes/failures. This is scoped to
one `mutate_tree` call (no global learning).

Defaults to disabled to keep coupling low until measurement shows value.
"""
mutable struct AdaptivePolicy
    enabled::Bool
    ewma_alpha::Float64
    floor::Float64
    function AdaptivePolicy(enabled::Bool, ewma_alpha::Float64, floor::Float64)
        isfinite(ewma_alpha) && 0 < ewma_alpha <= 1 ||
            throw(ArgumentError("ewma_alpha must be finite and in (0, 1]"))
        isfinite(floor) && 0 <= floor <= 1 ||
            throw(ArgumentError("adaptive floor must be finite and in [0, 1]"))
        return new(enabled, ewma_alpha, floor)
    end
end
function AdaptivePolicy(; enabled::Bool=false, ewma_alpha::Float64=0.2, floor::Float64=0.05)
    return AdaptivePolicy(enabled, ewma_alpha, floor)
end

"""
    BloatPolicy(; enabled=false, max_nodes=200, max_nodes_delta=40)

Optional bloat control for structure-changing mutations.

When enabled, structure-changing mutations can reject a replacement if:
- the mutated tree exceeds `max_nodes`, or
- the node-count increase exceeds `max_nodes_delta`.

Defaults to disabled to avoid baking performance tradeoffs into the baseline.
"""
mutable struct BloatPolicy
    enabled::Bool
    max_nodes::Int
    max_nodes_delta::Int
    function BloatPolicy(enabled::Bool, max_nodes::Int, max_nodes_delta::Int)
        max_nodes > 0 || throw(ArgumentError("max_nodes must be positive"))
        max_nodes_delta >= 0 || throw(ArgumentError("max_nodes_delta must be non-negative"))
        return new(enabled, max_nodes, max_nodes_delta)
    end
end
function BloatPolicy(; enabled::Bool=false, max_nodes::Int=200, max_nodes_delta::Int=40)
    return BloatPolicy(enabled, max_nodes, max_nodes_delta)
end

mutable struct MutationStat
    ewma_success::Float64
    initialized::Bool
end
MutationStat() = MutationStat(0.0, false)

mutable struct MutationEnv
    rng::AbstractRNG
    target_ty::Ty
    type_ctx::Dict{Symbol,TyLike}
    tree::Union{Nothing,AstTree}
    meta::Union{Nothing,Dict{AstNode,Tuple{TyLike,TyLike}}}
    cache_eligibility::Bool
    eligibility_cache::IdDict{Any,Any}
    adaptive_policy::AdaptivePolicy
    stats::IdDict{DataType,MutationStat}
    bloat_policy::BloatPolicy
end

########################################################
# Low-overhead mutation attempt recording (optional)
########################################################

const MUTATION_REASONS = (
    :success,
    :apply_err,
    :infer_err,
    :incompatible,
    :bloat_exceeded,
    :no_eligible_site,
    :choose_err,
    :prepare_err,
    :max_try_exhausted,
)

const _MUTATION_REASON_INDEX = Dict{Symbol,Int}((r => i for (i, r) in pairs(MUTATION_REASONS))...)

mutable struct MutationAttemptRecorder
    totals::Vector{Int}
    by_mutation::Dict{DataType,Vector{Int}}
    by_error_type::Union{Nothing,Dict{DataType,Int}}
end

MutationAttemptRecorder(; record_error_types::Bool=false) =
    MutationAttemptRecorder(zeros(Int, length(MUTATION_REASONS)), Dict{DataType,Vector{Int}}(), record_error_types ? Dict{DataType,Int}() : nothing)

@inline function _counts!(rec::MutationAttemptRecorder, T::DataType)
    return get!(rec.by_mutation, T) do
        zeros(Int, length(MUTATION_REASONS))
    end
end

@inline function record!(::Nothing, ::Union{Nothing,DataType}, ::Symbol; err_type::Union{Nothing,DataType}=nothing)
    return nothing
end

@inline function record!(rec::MutationAttemptRecorder, mutation_type::Union{Nothing,DataType}, reason::Symbol; err_type::Union{Nothing,DataType}=nothing)
    idx = get(_MUTATION_REASON_INDEX, reason, 0)
    idx == 0 && return nothing
    @inbounds rec.totals[idx] += 1
    if mutation_type !== nothing
        counts = _counts!(rec, mutation_type)
        @inbounds counts[idx] += 1
    end
    if err_type !== nothing && rec.by_error_type !== nothing
        rec.by_error_type[err_type] = get(rec.by_error_type, err_type, 0) + 1
    end
    return nothing
end

"""
    mutation_recorder_summary(rec; topk=8)

Summarize mutation outcomes by operator. When the recorder was constructed with
`record_error_types=true`, `error_types` contains the most frequent concrete
failure types; otherwise it is `nothing`.
"""
function mutation_recorder_summary(rec::MutationAttemptRecorder; topk::Int=8)
    totals = (;
        (r => rec.totals[i] for (i, r) in pairs(MUTATION_REASONS))...,
    )
    items = collect(rec.by_mutation)
    sort!(items; by=kv -> -kv[2][_MUTATION_REASON_INDEX[:success]])
    k = min(topk, length(items))
    per = Any[]
    for j in 1:k
        (T, counts) = items[j]
        push!(per, (;
            mutation_type=T,
            counts=(; (r => counts[i] for (i, r) in pairs(MUTATION_REASONS))...,),
        ))
    end
    error_types = if rec.by_error_type === nothing
        nothing
    else
        errors = collect(rec.by_error_type)
        sort!(errors; by=kv -> (-kv[2], string(kv[1])))
        [
            (; error_type=errors[j][1], count=errors[j][2])
            for j in 1:min(topk, length(errors))
        ]
    end
    return (; totals, per_mutation=per, error_types)
end

function _stat(env::MutationEnv, m::AbstractMutation)
    T = typeof(m)
    return get!(env.stats, T) do
        MutationStat()
    end
end

function _success_hint(env::MutationEnv, m::AbstractMutation)
    env.adaptive_policy.enabled || return 1.0
    st = _stat(env, m)
    st.initialized || return 1.0
    return max(env.adaptive_policy.floor, st.ewma_success)
end

function _update_stat!(env::MutationEnv, m::AbstractMutation, success::Bool)
    env.adaptive_policy.enabled || return nothing
    st = _stat(env, m)
    α = env.adaptive_policy.ewma_alpha
    x = success ? 1.0 : 0.0
    if !st.initialized
        st.ewma_success = x
        st.initialized = true
    else
        st.ewma_success = (1 - α) * st.ewma_success + α * x
    end
    return nothing
end

function _weighted_choice(rng::AbstractRNG, items::Vector, weights::Vector{Float64})
    @assert length(items) == length(weights)
    total = 0.0
    for w in weights
        isfinite(w) && w >= 0 || throw(ArgumentError("selection weights must be finite and non-negative"))
        w > 0 || continue
        total += w
    end
    isfinite(total) || throw(ArgumentError("selection weight sum must be finite"))
    total > 0 || return Err(NoEligibleMutationErr())
    r = rand(rng) * total
    acc = 0.0
    for (i, w) in pairs(weights)
        w > 0 || continue
        acc += w
        if r <= acc
            return items[i]
        end
    end
    return items[end]
end

_is_local_const(name::Symbol, ctx::NamedTuple) = haskey(ctx.local_consts, name) && haskey(ctx.local_consts_ty, name)
function _site_kind(node::AstNode, ctx::NamedTuple)
    if node isa ConstNode
        return _is_local_const(node.name, ctx) ? :const_local : :const_global
    elseif node isa IndexNode
        return :index
    elseif node isa AppNode
        return :app
    elseif node isa AbsNode
        return :abs
    else
        return :any
    end
end

function _binder_types(scope, meta)::Vector{Ty}
    tys = Ty[]
    for abs in scope.abstractions
        haskey(meta, abs) || continue
        raw_abs_ty = meta[abs][1]
        raw_abs_ty isa Ty || continue
        abs_ty = TypeSystem.prune(raw_abs_ty)
        abs_ty isa TyFun || continue
        push!(tys, TypeSystem.prune(abs_ty.arg))
    end
    return tys
end

function _node_at_path(tree::AstTree, path::Vector{Int})::AstNode
    node = tree.root
    for idx in path
        node = children(node)[idx]
    end
    return node
end

function _remap_site(tree::AstTree, site::Site)::Site
    node = _node_at_path(tree, site.path)
    if isempty(site.path)
        return Site(site.path, node, nothing, 1, site.depth, site.kind, site.ty, site.binder_types)
    end
    parent_path = site.path[1:end-1]
    parent = _node_at_path(tree, parent_path)
    return Site(site.path, node, parent, site.path[end], site.depth, site.kind, site.ty, site.binder_types)
end

function _target_refined_binder_types(tree::AstTree, path::Vector{Int}, target_ty::Ty, fallback::Vector{Ty})
    node = tree.root
    expected = target_ty
    tys = Ty[]
    for idx in path
        if node isa AbsNode
            pruned = TypeSystem.prune(expected)
            pruned isa TyFun || return fallback
            push!(tys, pruned.arg)
            expected = pruned.res
        end
        kids = children(node)
        1 <= idx <= length(kids) || return fallback
        node = kids[idx]
    end
    return tys
end

function _target_refined_site_ty(tree::AstTree, path::Vector{Int}, target_ty::Ty, fallback::Ty)
    node = tree.root
    expected = target_ty
    isempty(path) && return TypeSystem.prune(expected)
    for idx in path
        if node isa AbsNode && idx == 1
            pruned = TypeSystem.prune(expected)
            pruned isa TyFun || return fallback
            expected = pruned.res
        else
            return fallback
        end
        kids = children(node)
        1 <= idx <= length(kids) || return fallback
        node = kids[idx]
    end
    return TypeSystem.prune(expected)
end

function _collect_sites(tree::AstTree, meta::Dict{AstNode,Tuple{TyLike,TyLike}}, ctx::NamedTuple, target_ty::Ty)
    sites = Site[]
    traverse(tree, PointerState{AstNode}, ScopeState) do t, node, bag
        pointer = getstate(bag, PointerState{AstNode})
        stack = getstate(bag, StackState{AstNode})
        scope = getstate(bag, ScopeState)
        haskey(meta, node) || return nothing
        meta_node_ty = meta[node][1]
        raw_node_ty = meta_node_ty isa Ty ? TypeSystem.prune(meta_node_ty) : meta_node_ty
        raw_binder_types = _binder_types(scope, meta)
        kind = _site_kind(node, ctx)
        path =
            if length(stack.stack) <= 1
                Int[]
            else
                [stack.stack[i][2] for i in 1:(length(stack.stack)-1)]
            end
        node_ty = raw_node_ty isa Ty ? _target_refined_site_ty(tree, path, target_ty, raw_node_ty) : raw_node_ty
        binder_types = _target_refined_binder_types(tree, path, target_ty, raw_binder_types)
        push!(sites, Site(path, node, pointer.parent, pointer.child_index, length(binder_types), kind, node_ty, binder_types))
        return nothing
    end
    return sites
end

"""
    IndexedProvider() <: AbstractSiteProvider

Indexed site provider (Milestone B).

Runs `infer(..., meta=...)` once and traverses once to build:
- `sites_all`
- `sites_by_kind` index

This is the recommended provider once correctness is established.
"""
mutable struct IndexedProvider <: AbstractSiteProvider
    prepared::Bool
    sites_all::Vector{Site}
    sites_kind::Dict{Symbol,Vector{Site}}
end
IndexedProvider() = IndexedProvider(false, Site[], Dict{Symbol,Vector{Site}}())

function reset!(p::IndexedProvider)
    p.prepared = false
    empty!(p.sites_all)
    empty!(p.sites_kind)
    return p
end

function prepare!(p::IndexedProvider, tree::AstTree, type_ctx::Dict{Symbol,<:TyLike}, env::MutationEnv, ctx::NamedTuple)
    reset!(p)
    meta = Dict{AstNode,Tuple{TyLike,TyLike}}()
    factory = haskey(ctx, :tyfactory) ? ctx.tyfactory : newtyvar
    ty = _run_mutation_inference(ctx, tree, type_ctx, factory, meta)
    ty isa Err && return ty
    if ty isa Ty
        target_copy = TypeSystem._query_copy(
            env.target_ty,
            factory,
            Dict{TypeSystem.TyVar,TypeSystem.TyVar}(),
        )
        matched = TypeSystem.unify!(ty, target_copy)
        matched isa Err && return matched
    end
    env.meta = meta
    p.sites_all = _collect_sites(tree, meta, ctx, env.target_ty)
    for s in p.sites_all
        push!(get!(p.sites_kind, :any, Site[]), s)
        s.kind === :any || push!(get!(p.sites_kind, s.kind, Site[]), s)
    end
    p.prepared = true
    return nothing
end

sites(p::IndexedProvider) = p.sites_all
sites_by_kind(p::IndexedProvider, kind::Symbol) = get(p.sites_kind, kind, Site[])

"""
    WeightedPairChooser(; weights=nothing, cache_eligibility=false) <: AbstractPairChooser

Practical `(mutation, site)` chooser with weights.

Selection model:
1) For each mutation `m`, build eligible sites + per-site weights:
   - candidates come from `required_site_kinds(m)`
   - eligibility is filtered by `is_site_eligible(m, site, env, ctx)`
   - weights are `site_weight(m, site, env, ctx)`
2) Sample a mutation by its *mass*:
   `mass(m) = weight(m) * sum(site_weights) * success_hint(m)`
3) Sample a site within that mutation by its site weights.

## Parameters
- `weights`: per-mutation-type weights. A concrete parameterized type overrides
  its family key, so callers may configure either `typeof(mutation)` or a whole
  family such as `SourceBindingInsertionMutation`. Existing
  `Dict{DataType,Float64}` inputs remain accepted. Defaults:
- `ScalarConstMutation => 0.25`
- `ArrayConstMutation => 0.25`
- `ParameterIntroductionMutation => 0.15`
- `BoolConstMutation => 0.05`
- `PointMutation => 0.10`
- `CSEMutation => 0.08`
- `BetaReduceMutation => 0.04`
- `EtaExpandMutation => 0.03`
- `EtaReduceMutation => 0.03`
- `LetHoistMutation => 0.025`
- `LetSinkMutation => 0.025`
- `LetPruneMutation => 0.02`
- `SourceBindingInsertionMutation => 0.015`
- `cache_eligibility::Bool=false`: cache eligible site lists per mutation within
  one `mutate_tree` call (Milestone C).

Notes
- Uniform selection is intentionally not the default because SR mutation operators
  have wildly different success rates and usefulness (constants often dominate).
"""
struct WeightedPairChooser <: AbstractPairChooser
    weights::Dict{Type,Float64}
    cache_eligibility::Bool
    function WeightedPairChooser(weights::Dict{Type,Float64}, cache_eligibility::Bool)
        for weight in values(weights)
            isfinite(weight) && weight >= 0 ||
                throw(ArgumentError("mutation weights must be finite and non-negative"))
        end
        return new(weights, cache_eligibility)
    end
end

function _normalize_mutation_weights(weights)
    normalized = Dict{Type,Float64}()
    for (mutation_type, weight) in pairs(weights)
        mutation_type isa Type || throw(ArgumentError(
            "mutation weight keys must be types, got $(repr(mutation_type))",
        ))
        weight isa Real || throw(ArgumentError(
            "mutation weights must be real, got $(repr(weight))",
        ))
        normalized_weight = Float64(weight)
        isfinite(normalized_weight) && normalized_weight >= 0 || throw(ArgumentError(
            "mutation weights must be finite and non-negative, got $(repr(weight))",
        ))
        normalized[mutation_type] = normalized_weight
    end
    return normalized
end

function WeightedPairChooser(; weights=nothing, cache_eligibility::Bool=false)
    default = Dict{Type,Float64}(
        # Split constant operators (preferred).
        ScalarConstMutation => 0.25,
        ArrayConstMutation => 0.25,
        ParameterIntroductionMutation => 0.15,
        BoolConstMutation => 0.05,
        # Backwards-compatible wrapper.
        ConstMutation => 0.55,

        PointMutation => 0.10,
        CSEMutation => 0.08,
        BetaReduceMutation => 0.04,
        EtaExpandMutation => 0.03,
        EtaReduceMutation => 0.03,
        LetHoistMutation => 0.025,
        LetSinkMutation => 0.025,
        LetPruneMutation => 0.02,
        SourceBindingInsertionMutation => 0.015,
    )
    return WeightedPairChooser(
        weights === nothing ? default : _normalize_mutation_weights(weights),
        cache_eligibility,
    )
end

WeightedPairChooser(weights::AbstractDict, cache_eligibility::Bool) =
    WeightedPairChooser(_normalize_mutation_weights(weights), cache_eligibility)

function _mutation_weight(
    chooser::WeightedPairChooser,
    mutation::AbstractMutation,
    default::Real,
)
    concrete = typeof(mutation)
    haskey(chooser.weights, concrete) && return chooser.weights[concrete]
    family = Base.typename(concrete).wrapper
    return get(chooser.weights, family, Float64(default))
end

function _eligible_sites(m::AbstractMutation, provider::AbstractSiteProvider, env::MutationEnv, ctx::NamedTuple)
    if env.cache_eligibility && haskey(env.eligibility_cache, m)
        return env.eligibility_cache[m]
    end
    candidates = Site[]
    for kind in required_site_kinds(m)
        append!(candidates, sites_by_kind(provider, kind))
    end
    eligible = Site[]
    weights = Float64[]
    for s in candidates
        is_site_eligible(m, s, env, ctx) || continue
        w = site_weight(m, s, env, ctx)
        w isa Real && isfinite(w) && w >= 0 ||
            throw(ArgumentError("site_weight must return a finite non-negative real"))
        w > 0 || continue
        push!(eligible, s)
        push!(weights, Float64(w))
    end
    res = (eligible, weights)
    env.cache_eligibility && (env.eligibility_cache[m] = res)
    return res
end

function choose_pair(chooser::WeightedPairChooser,
    mutations::Vector{<:AbstractMutation},
    provider::AbstractSiteProvider,
    tree::AstTree,
    type_ctx::Dict{Symbol,<:TyLike},
    env::MutationEnv,
    ctx::NamedTuple,
)
    env.cache_eligibility = chooser.cache_eligibility
    env.tree = tree

    masses = Float64[]
    payloads = Any[]
    for m in mutations
        (eligible, wts) = _eligible_sites(m, provider, env, ctx)
        if isempty(eligible)
            push!(masses, 0.0)
            push!(payloads, (m, eligible, wts))
            continue
        end
        coarse_weight = _mutation_weight(chooser, m, base_weight(m, ctx))
        isfinite(coarse_weight) && coarse_weight >= 0 ||
            throw(ArgumentError("mutation base weight must be finite and non-negative"))
        mass = coarse_weight * sum(wts) * _success_hint(env, m)
        isfinite(mass) && mass >= 0 ||
            throw(ArgumentError("mutation selection mass must be finite and non-negative"))
        push!(masses, mass)
        push!(payloads, (m, eligible, wts))
    end
    choice = _weighted_choice(env.rng, payloads, masses)
    choice isa Err && return choice
    (m, eligible, wts) = choice
    isempty(eligible) && return Err(NoEligibleSiteErr())
    site = _weighted_choice(env.rng, eligible, wts)
    site isa Err && return site
    return PairChoice(m, site)
end

function _replace_site!(tree::AstTree, site::Site, new_subtree::AstTree)
    if site.parent === nothing
        setroot!(tree, new_subtree)
    else
        setchild!(tree, site.parent, site.child_index, new_subtree)
    end
    return tree
end

abstract type AbstractSiteLockPolicy end
abstract type AbstractSiteSelectionPolicy end
abstract type AbstractReplacementGenerator end

"""
    DefaultSiteLockPolicy()

Conservative lock policy for generated subtree replacement.

The default keeps De Bruijn variable leaves and application function positions
out of arbitrary subtree replacement. It also defaults to `spine_only=true`,
which admits only the root and the continuous lambda-body spine. Callers can
disable that or replace the policy when they intentionally want deeper
application internals to participate.
"""
struct DefaultSiteLockPolicy <: AbstractSiteLockPolicy
    spine_only::Bool
end
DefaultSiteLockPolicy(; spine_only::Bool=true) = DefaultSiteLockPolicy(spine_only)

"""
    ExponentialSiteSelectionPolicy(; scope_bias=-0.75, path_bias=-0.04, root_weight=0.35)

Default site weighting policy for subtree replacement. Sites under many
enclosing binders are exponentially damped because scoped generation there has
many more local variables and provider routes. Longer paths are also mildly
damped, while the root is damped by `root_weight`.
"""
struct ExponentialSiteSelectionPolicy <: AbstractSiteSelectionPolicy
    scope_bias::Float64
    path_bias::Float64
    root_weight::Float64
end

function ExponentialSiteSelectionPolicy(;
    scope_bias::Real=-0.75,
    path_bias::Real=-0.04,
    root_weight::Real=0.35,
    depth_bias::Union{Nothing,Real}=nothing,
)
    sb = Float64(depth_bias === nothing ? scope_bias : depth_bias)
    pb = Float64(path_bias)
    rw = Float64(root_weight)
    isfinite(sb) || throw(ArgumentError("scope_bias must be finite"))
    isfinite(pb) || throw(ArgumentError("path_bias must be finite"))
    isfinite(rw) && rw >= 0 || throw(ArgumentError("root_weight must be finite and nonnegative"))
    return ExponentialSiteSelectionPolicy(sb, pb, rw)
end

"""
    PathSiteSelectionPolicy(path)

Test and fixture policy that admits exactly one path.

Selection is still combined with the mutation's site lock policy; this policy
does not force a locked path to become eligible.
"""
struct PathSiteSelectionPolicy <: AbstractSiteSelectionPolicy
    path::Vector{Int}
end
PathSiteSelectionPolicy(path::AbstractVector{<:Integer}) = PathSiteSelectionPolicy(Int[path...])

"""
    CanonicalReplacementGenerator()

Replacement generator adapter that calls `generate_canonical_scoped` with the
selected site's type and visible binder types.
"""
struct CanonicalReplacementGenerator <: AbstractReplacementGenerator
    include_local_consts::Bool
end
CanonicalReplacementGenerator(; include_local_consts::Bool=false) = CanonicalReplacementGenerator(include_local_consts)

"""
    ScopedReplacementGenerator(generate_scoped; include_local_consts=false)

Adapt a callable with the shape `generate_scoped(target_ty; kwargs...)` to the
subtree replacement protocol. This keeps procedural or other alternative
generators injectable without adding a dependency from `Mutations` to them.
"""
struct ScopedReplacementGenerator{F} <: AbstractReplacementGenerator
    generate_scoped::F
    include_local_consts::Bool
end
ScopedReplacementGenerator(generate_scoped; include_local_consts::Bool=false) =
    ScopedReplacementGenerator(generate_scoped, include_local_consts)

is_site_locked(::AbstractSiteLockPolicy, ::Site, ::MutationEnv, ::NamedTuple) = false

function is_site_locked(policy::DefaultSiteLockPolicy, site::Site, ::MutationEnv, ::NamedTuple)
    site.node isa IndexNode && return true
    site.parent isa AppNode && site.child_index == 1 && return true
    policy.spine_only && length(site.path) != site.depth && return true
    return false
end

function site_selection_weight(policy::ExponentialSiteSelectionPolicy, site::Site, ::MutationEnv, ::NamedTuple)
    weight = exp(policy.scope_bias * site.depth + policy.path_bias * length(site.path))
    isempty(site.path) && (weight *= policy.root_weight)
    return weight
end

site_selection_weight(policy::PathSiteSelectionPolicy, site::Site, ::MutationEnv, ::NamedTuple) =
    site.path == policy.path ? 1.0 : 0.0

"""
    LocalCallSiteSelectionPolicy(base=ExponentialSiteSelectionPolicy(); multiplier=16)

Multiply the base weight of an application whose left spine is headed by a
local `IndexNode`. This is a generic higher-order call-result preference: it
does not distinguish ordinary function arguments from recursive self binders.
"""
struct LocalCallSiteSelectionPolicy{P<:AbstractSiteSelectionPolicy} <: AbstractSiteSelectionPolicy
    base::P
    multiplier::Float64
end

function LocalCallSiteSelectionPolicy(
    base::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy();
    multiplier::Real=16.0,
)
    value = Float64(multiplier)
    isfinite(value) && value >= 0 || throw(ArgumentError(
        "local-call multiplier must be finite and non-negative",
    ))
    return LocalCallSiteSelectionPolicy(base, value)
end

function _application_spine_head(node::AstNode)
    current = node
    while current isa AppNode
        current = current.func
    end
    return current
end

function site_selection_weight(
    policy::LocalCallSiteSelectionPolicy,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    weight = site_selection_weight(policy.base, site, env, ctx)
    site.node isa AppNode || return weight
    return _application_spine_head(site.node) isa IndexNode ?
        weight * policy.multiplier : weight
end

site_selection_weight(::AbstractSiteSelectionPolicy, ::Site, ::MutationEnv, ::NamedTuple) = 1.0

function _global_consts(ctx::NamedTuple)
    return haskey(ctx, :global_consts) ? ctx.global_consts : Dict{Symbol,Any}()
end

function _generation_closed_type(ty::Ty)
    pruned = TypeSystem.prune(ty)
    pruned isa TyVar && return false
    pruned isa TyConst && return true
    pruned isa TyFun && return _generation_closed_type(pruned.arg) && _generation_closed_type(pruned.res)
    pruned isa TyList && return _generation_closed_type(pruned.elem)
    pruned isa TyArray && return TypeSystem.prune(pruned.len) isa TyNat && _generation_closed_type(pruned.elem)
    return false
end

function _default_mutation_inference(tree, type_ctx, factory, meta)
    meta === nothing && return infer(tree, type_ctx, factory)
    return infer(tree, type_ctx, factory, meta)
end

function _run_mutation_inference(ctx, tree, type_ctx, factory, meta)
    checker = haskey(ctx, :type_inference) ? ctx.type_inference : _default_mutation_inference
    applicable(checker, tree, type_ctx, factory, meta) || throw(ArgumentError(
        "type_inference must accept (tree, type_ctx, tyfactory, meta_or_nothing)",
    ))
    return checker(tree, type_ctx, factory, meta)
end

function _replacement_generation_kwargs(
    include_local_consts::Bool,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    defaults = (;
        rng=env.rng,
        scope_tys=reverse(site.binder_types),
        global_consts=_global_consts(ctx),
        global_consts_ty=ctx.global_consts_ty,
        local_consts=ctx.local_consts,
        local_consts_ty=ctx.local_consts_ty,
        source_local_consts=include_local_consts,
    )
    return merge(defaults, generation_kwargs)
end

function generate_replacement(
    generator::CanonicalReplacementGenerator,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    kwargs = _replacement_generation_kwargs(
        generator.include_local_consts,
        site,
        env,
        ctx,
        generation_kwargs,
    )
    return generate_canonical_scoped(site.ty; kwargs...)
end

function generate_replacement(
    generator::ScopedReplacementGenerator,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    kwargs = _replacement_generation_kwargs(
        generator.include_local_consts,
        site,
        env,
        ctx,
        generation_kwargs,
    )
    return generator.generate_scoped(site.ty; kwargs...)
end

"""
    SubtreeReplacementMutation(; site_lock, site_selection, replacement_generator, generation_kwargs)

Type-compatible subtree replacement. The selected site `S : A` is replaced by a
fresh canonical term `G : A` generated under the binder scope visible at `S`.

All deep behavior is injectable: `site_lock` controls rejected sites,
`site_selection` controls per-site weights, and `replacement_generator` owns how
the replacement term is sampled.
"""
struct SubtreeReplacementMutation <: AbstractMutation
    site_lock::AbstractSiteLockPolicy
    site_selection::AbstractSiteSelectionPolicy
    replacement_generator::AbstractReplacementGenerator
    generation_kwargs::NamedTuple
end

function SubtreeReplacementMutation(;
    site_lock::AbstractSiteLockPolicy=DefaultSiteLockPolicy(),
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(),
    replacement_generator::AbstractReplacementGenerator=CanonicalReplacementGenerator(),
    generation_kwargs::NamedTuple=(; sampling_depth_limit=2, sampling_node_budget=24, guarded_expansion_depth=0),
)
    return SubtreeReplacementMutation(site_lock, site_selection, replacement_generator, generation_kwargs)
end

required_site_kinds(::SubtreeReplacementMutation) = Symbol[:any]

function is_site_eligible(m::SubtreeReplacementMutation, site::Site, env::MutationEnv, ctx::NamedTuple)
    is_site_locked(m.site_lock, site, env, ctx) && return false
    _generation_closed_type(site.ty) || return false
    return site_selection_weight(m.site_selection, site, env, ctx) > 0
end

site_weight(m::SubtreeReplacementMutation, site::Site, env::MutationEnv, ctx::NamedTuple) =
    site_selection_weight(m.site_selection, site, env, ctx)

"""
    ParameterIntroductionMutation(; σ=1.0, site_lock, site_selection)

Introduce a fresh mutable real parameter at a typed expression site.

The operator anchors on the site's inferred type and supports `Float` plus
fixed-shape nested `TyArray[..., Float]`. It requires `ctx.local_consts`,
`ctx.local_consts_ty`, and `ctx.const_symfactory`; the ordinary mutation driver
provides these fields. Unsupported/open shapes have no eligible site. A chosen
site is replaced transactionally with a new local constant, so final inference
and bloat validation remain owned by `mutate_tree`. The default site policy
does not penalize lexical depth: numeric sites in an ordinary function body
must not become unreachable merely because they are enclosed by lambdas.
"""
struct ParameterIntroductionMutation <: AbstractMutation
    σ::Float64
    site_lock::AbstractSiteLockPolicy
    site_selection::AbstractSiteSelectionPolicy
end

function ParameterIntroductionMutation(;
    σ::Real=1.0,
    site_lock::AbstractSiteLockPolicy=DefaultSiteLockPolicy(spine_only=false),
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(
        scope_bias=0.0,
    ),
)
    scale = Float64(σ)
    isfinite(scale) && scale > 0 || throw(ArgumentError(
        "parameter introduction σ must be finite and positive",
    ))
    return ParameterIntroductionMutation(scale, site_lock, site_selection)
end

required_site_kinds(::ParameterIntroductionMutation) = Symbol[:any]

function _real_parameter_shape(ty::Ty)
    current = TypeSystem.prune(ty)
    current == FloatTy && return ()
    dimensions = Int[]
    while current isa TyArray
        length_ty = TypeSystem.prune(current.len)
        length_ty isa TyNat || return nothing
        push!(dimensions, length_ty.value)
        current = TypeSystem.prune(current.elem)
    end
    current == FloatTy || return nothing
    isempty(dimensions) && return nothing
    return Tuple(dimensions)
end

function is_site_eligible(
    mutation::ParameterIntroductionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    is_site_locked(mutation.site_lock, site, env, ctx) && return false
    _real_parameter_shape(site.ty) === nothing && return false
    return site_selection_weight(mutation.site_selection, site, env, ctx) > 0
end

site_weight(
    mutation::ParameterIntroductionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
) = site_selection_weight(mutation.site_selection, site, env, ctx)

function apply(
    mutation::ParameterIntroductionMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    shape = _real_parameter_shape(site.ty)
    shape === nothing && return Err(UnsupportedConstTyErr(site.ty))
    value = isempty(shape) ?
        randn(env.rng) * mutation.σ :
        randn(env.rng, Float64, shape...) .* mutation.σ
    return _apply_local_const_mutation!(tree, site, ctx, value)
end

function _collect_const_names!(names::Set{Symbol}, node::AstNode)
    node isa ConstNode && push!(names, node.name)
    for child in children(node)
        _collect_const_names!(names, child)
    end
    return names
end

function _same_const_binding(
    name::Symbol,
    result_local_consts::AbstractDict{Symbol,<:Any},
    result_local_const_types::AbstractDict{Symbol,<:TyLike},
    current_local_consts::AbstractDict{Symbol,<:Any},
    current_local_const_types::AbstractDict{Symbol,<:TyLike},
)
    haskey(current_local_const_types, name) || return false
    haskey(result_local_const_types, name) || return false
    TypeSystem.semantic_typekey(current_local_const_types[name]) ==
        TypeSystem.semantic_typekey(result_local_const_types[name]) || return false
    haskey(current_local_consts, name) || return false
    haskey(result_local_consts, name) || return false
    return try
        isequal(current_local_consts[name], result_local_consts[name])
    catch
        current_local_consts[name] === result_local_consts[name]
    end
end

function _fresh_replacement_const_name(occupied::Set{Symbol}, base::Symbol)
    stem = String(base)
    i = 1
    while true
        candidate = Symbol(stem, "_replacement", i)
        candidate in occupied || return candidate
        i += 1
    end
end

function _rename_const_nodes(node::AstNode, renames::Dict{Symbol,Symbol})::AstNode
    if node isa ConstNode
        return ConstNode(get(renames, node.name, node.name))
    end
    for (index, child) in pairs(children(node))
        setchild!(node, index, _rename_const_nodes(child, renames))
    end
    return node
end

function _merge_replacement_consts!(
    replacement::AstTree,
    result_local_consts::Dict{Symbol,Any},
    result_local_const_types::Dict{Symbol,TyLike},
    current_local_consts::Dict{Symbol,Any},
    current_local_const_types::Dict{Symbol,TyLike},
    ;
    reserved_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    reserved_const_types::AbstractDict{Symbol,<:TyLike}=Dict{Symbol,TyLike}(),
)
    used = _collect_const_names!(Set{Symbol}(), replacement.root)
    occupied = union(
        Set(keys(current_local_const_types)),
        Set(keys(result_local_const_types)),
        Set(keys(reserved_const_types)),
    )
    renames = Dict{Symbol,Symbol}()
    for name in sort!(collect(used))
        haskey(result_local_const_types, name) || continue
        haskey(result_local_consts, name) || continue
        target_name = name
        if !haskey(current_local_const_types, name) &&
                haskey(reserved_const_types, name) &&
                _same_const_binding(
                    name,
                    result_local_consts,
                    result_local_const_types,
                    reserved_consts,
                    reserved_const_types,
                )
            continue
        elseif (
            haskey(current_local_const_types, name) &&
            !_same_const_binding(
                name,
                result_local_consts,
                result_local_const_types,
                current_local_consts,
                current_local_const_types,
            )
        ) || (!haskey(current_local_const_types, name) && haskey(reserved_const_types, name))
            target_name = _fresh_replacement_const_name(occupied, name)
            push!(occupied, target_name)
            renames[name] = target_name
        end
        current_local_consts[target_name] = result_local_consts[name]
        current_local_const_types[target_name] = result_local_const_types[name]
    end
    isempty(renames) || (replacement.root = _rename_const_nodes(replacement.root, renames))
    return replacement
end

function apply(m::SubtreeReplacementMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    result = generate_replacement(m.replacement_generator, site, env, ctx, m.generation_kwargs)
    result isa Err && return result
    result isa GenerationResult || return Err(ArgumentError("replacement generator must return GenerationResult or Err"))

    replacement = deepcopy(result.tree)
    _merge_replacement_consts!(
        replacement,
        copy(result.local_consts),
        copy(result.local_const_types),
        ctx.local_consts,
        ctx.local_consts_ty;
        reserved_consts=ctx.global_consts,
        reserved_const_types=ctx.global_consts_ty,
    )

    _replace_site!(tree, site, replacement)
    return tree
end

"""
    ContextualExpansionMutation(; site_selection, replacement_generator, generation_kwargs)

Grow a selected `S : A` into the typed let-like context
`(λseed. C[seed]) S`, where the injected scoped generator constructs
`C[seed] : A` and must use the new innermost binder. This supplies a bounded
composition step that ordinary finite witness generation and plain subtree
replacement do not provide on their own.

The generated context sees `seed` as De Bruijn index zero followed by the
binders already visible at the selected site. Procedural generators remain
injectable, but contexts whose use of `seed` is hidden below a non-core binder
are conservatively rejected; final injected inference is still authoritative.
"""
struct ContextualExpansionMutation <: AbstractMutation
    site_selection::AbstractSiteSelectionPolicy
    replacement_generator::AbstractReplacementGenerator
    generation_kwargs::NamedTuple
end

function ContextualExpansionMutation(;
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(),
    replacement_generator::AbstractReplacementGenerator=CanonicalReplacementGenerator(),
    generation_kwargs::NamedTuple=(;
        sampling_depth_limit=2,
        sampling_node_budget=24,
        guarded_expansion_depth=0,
    ),
)
    return ContextualExpansionMutation(
        site_selection,
        replacement_generator,
        generation_kwargs,
    )
end

required_site_kinds(::ContextualExpansionMutation) = Symbol[:any]
base_weight(::ContextualExpansionMutation, ::NamedTuple) = 0.08

function is_site_eligible(
    mutation::ContextualExpansionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _generation_closed_type(site.ty) || return false
    return site_selection_weight(mutation.site_selection, site, env, ctx) > 0
end

site_weight(
    mutation::ContextualExpansionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
) = site_selection_weight(mutation.site_selection, site, env, ctx)

function _uses_context_binder(node::AstNode, depth::Int=0)::Bool
    if node isa IndexNode
        return node.index == depth
    elseif node isa AbsNode
        return _uses_context_binder(node.body, depth + 1)
    elseif node isa AppNode
        return _uses_context_binder(node.func, depth) ||
               _uses_context_binder(node.arg, depth)
    end
    return false
end

function apply(
    mutation::ContextualExpansionMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    context_site = Site(
        site.path,
        site.node,
        site.parent,
        site.child_index,
        site.depth + 1,
        site.kind,
        site.ty,
        Ty[site.binder_types; site.ty],
    )
    result = generate_replacement(
        mutation.replacement_generator,
        context_site,
        env,
        ctx,
        mutation.generation_kwargs,
    )
    result isa Err && return result
    result isa GenerationResult || return Err(ArgumentError(
        "context generator must return GenerationResult or Err",
    ))

    context = deepcopy(result.tree)
    _uses_context_binder(context.root) || return Err(NoEligibleSiteErr())
    context.root == IndexNode(0) && return Err(NoEligibleSiteErr())
    _merge_replacement_consts!(
        context,
        copy(result.local_consts),
        copy(result.local_const_types),
        ctx.local_consts,
        ctx.local_consts_ty;
        reserved_consts=ctx.global_consts,
        reserved_const_types=ctx.global_consts_ty,
    )

    expanded = AstTree(AppNode(
        AbsNode(:seed, context.root),
        deepcopy(site.node),
    ))
    _replace_site!(tree, site, expanded)
    return tree
end

"""
    UnaryCompositionMutation(; site_selection)

Wrap a selected closed-typed subtree `S : T` with a monomorphic global
endomorphism `H : T → T`, producing `H S`. This is the direct bounded
neighborhood for productive unary operators such as square, square root, or a
shape-specific tensor transform. Polymorphic and local heads remain outside
this operator; final injected inference still validates the complete tree.
"""
struct UnaryCompositionMutation{S} <: AbstractMutation
    site_selection::S
end

UnaryCompositionMutation(;
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(
        scope_bias=0.0,
        path_bias=-0.02,
    ),
) = UnaryCompositionMutation(site_selection)

required_site_kinds(::UnaryCompositionMutation) = Symbol[:any]
base_weight(::UnaryCompositionMutation, ::NamedTuple) = 0.08

function _unary_composition_options(site::Site, ctx::NamedTuple)
    _generation_closed_type(site.ty) || return Symbol[]
    haskey(ctx, :global_consts) || return Symbol[]
    haskey(ctx, :global_consts_ty) || return Symbol[]
    options = Symbol[]
    for name in sort!(collect(keys(ctx.global_consts_ty)))
        haskey(ctx.global_consts, name) || continue
        raw = ctx.global_consts_ty[name]
        raw isa Ty || continue
        arrow = TypeSystem.prune(raw)
        arrow isa TyFun || continue
        argument_ty = TypeSystem.prune(arrow.arg)
        result_ty = TypeSystem.prune(arrow.res)
        argument_ty isa Ty && result_ty isa Ty || continue
        _same_composition_type(argument_ty, site.ty) || continue
        _same_composition_type(result_ty, site.ty) || continue
        push!(options, name)
    end
    return options
end

function is_site_eligible(
    mutation::UnaryCompositionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    site_selection_weight(mutation.site_selection, site, env, ctx) > 0 || return false
    return !isempty(_unary_composition_options(site, ctx))
end

site_weight(
    mutation::UnaryCompositionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
) = site_selection_weight(mutation.site_selection, site, env, ctx)

function apply(
    ::UnaryCompositionMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    options = _unary_composition_options(site, ctx)
    isempty(options) && return Err(NoEligibleSiteErr())
    wrapped = AppNode(ConstNode(rand(env.rng, options)), deepcopy(site.node))
    _replace_site!(tree, site, AstTree(wrapped))
    return tree
end

"""
    BinaryCompositionMutation(; site_selection, replacement_generator, generation_kwargs)

Compose an existing closed-typed subtree `S : T` with one generated sibling
through a monomorphic curried global head. A candidate head has type
`A → B → T`; when `S` matches `A` the rewrite is `H S G`, and when it
matches `B` the rewrite is `H G S`. `G` is generated under the exact lexical
scope visible at `S`.

This is a semantic search move for preserving one useful branch while adding a
second typed branch. It does not recognize operator names or algebraic laws;
noncommutative heads retain independently selectable left/right orientations.
Polymorphic heads are deliberately excluded until their instantiation contract
can be represented without leaking query-local metas.
"""
struct BinaryCompositionMutation{S,G} <: AbstractMutation
    site_selection::S
    replacement_generator::G
    generation_kwargs::NamedTuple
end

function BinaryCompositionMutation(;
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(),
    replacement_generator::AbstractReplacementGenerator=CanonicalReplacementGenerator(),
    generation_kwargs::NamedTuple=(;
        sampling_depth_limit=2,
        sampling_node_budget=24,
        guarded_expansion_depth=0,
    ),
)
    return BinaryCompositionMutation(
        site_selection,
        replacement_generator,
        generation_kwargs,
    )
end

required_site_kinds(::BinaryCompositionMutation) = Symbol[:any]
base_weight(::BinaryCompositionMutation, ::NamedTuple) = 0.08

_same_composition_type(left::Ty, right::Ty) =
    iscompatible(left, right) && iscompatible(right, left)

function _binary_composition_options(site::Site, ctx::NamedTuple)
    _generation_closed_type(site.ty) || return NamedTuple[]
    haskey(ctx, :global_consts) || return NamedTuple[]
    haskey(ctx, :global_consts_ty) || return NamedTuple[]
    options = NamedTuple[]
    for name in sort!(collect(keys(ctx.global_consts_ty)))
        haskey(ctx.global_consts, name) || continue
        raw = ctx.global_consts_ty[name]
        raw isa Ty || continue
        first_arrow = TypeSystem.prune(raw)
        first_arrow isa TyFun || continue
        second_arrow = TypeSystem.prune(first_arrow.res)
        second_arrow isa TyFun || continue
        result_ty = TypeSystem.prune(second_arrow.res)
        result_ty isa Ty || continue
        _same_composition_type(result_ty, site.ty) || continue

        left_ty = TypeSystem.prune(first_arrow.arg)
        right_ty = TypeSystem.prune(second_arrow.arg)
        if left_ty isa Ty && _generation_closed_type(right_ty) &&
                _same_composition_type(left_ty, site.ty)
            push!(options, (; name, sibling_ty=right_ty, seed_first=true))
        end
        if right_ty isa Ty && _generation_closed_type(left_ty) &&
                _same_composition_type(right_ty, site.ty)
            push!(options, (; name, sibling_ty=left_ty, seed_first=false))
        end
    end
    return options
end

function is_site_eligible(
    mutation::BinaryCompositionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    site_selection_weight(mutation.site_selection, site, env, ctx) > 0 || return false
    return !isempty(_binary_composition_options(site, ctx))
end

site_weight(
    mutation::BinaryCompositionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
) = site_selection_weight(mutation.site_selection, site, env, ctx)

function apply(
    mutation::BinaryCompositionMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    options = _binary_composition_options(site, ctx)
    isempty(options) && return Err(NoEligibleSiteErr())
    option = rand(env.rng, options)
    sibling_site = Site(
        site.path,
        site.node,
        site.parent,
        site.child_index,
        site.depth,
        site.kind,
        option.sibling_ty,
        site.binder_types,
    )
    result = generate_replacement(
        mutation.replacement_generator,
        sibling_site,
        env,
        ctx,
        mutation.generation_kwargs,
    )
    result isa Err && return result
    result isa GenerationResult || return Err(ArgumentError(
        "binary composition generator must return GenerationResult or Err",
    ))

    sibling = deepcopy(result.tree)
    _merge_replacement_consts!(
        sibling,
        copy(result.local_consts),
        copy(result.local_const_types),
        ctx.local_consts,
        ctx.local_consts_ty;
        reserved_consts=ctx.global_consts,
        reserved_const_types=ctx.global_consts_ty,
    )
    seed = deepcopy(site.node)
    left, right = option.seed_first ? (seed, sibling.root) : (sibling.root, seed)
    composed = AppNode(AppNode(ConstNode(option.name), left), right)
    _replace_site!(tree, site, AstTree(composed))
    return tree
end

"""
    prune_local_consts!(tree::AstTree, local_consts, local_consts_ty) -> Nothing

Remove unreferenced local constants from the per-individual constant containers.

This is the main mechanism that prevents `ConstMutation` from growing the local
constant dictionaries without bound.

Behavior:
- Collect all `ConstNode(name)` symbols in `tree`.
- Remove any key in `local_consts` / `local_consts_ty` not referenced by the tree.
"""
function prune_local_consts!(tree::AstTree, local_consts::Dict{Symbol,Any}, local_consts_ty::Dict{Symbol,TyLike})
    referenced = Set{Symbol}()
    for node in tree
        node isa ConstNode || continue
        push!(referenced, node.name)
    end
    for key in collect(keys(local_consts))
        key in referenced || delete!(local_consts, key)
    end
    for key in collect(keys(local_consts_ty))
        key in referenced || delete!(local_consts_ty, key)
    end
    return nothing
end

"""
    mutate_const_value(rng, value, ty, args...) -> Union{Any,Err}

Internal, extensible value-level constant mutation protocol.

This function is intentionally implemented via **multiple dispatch** so new
constant categories can be added without editing a central type switch.
"""
mutate_const_value(::AbstractRNG, value, ty::Ty, args...) = Err(UnsupportedConstTyErr(ty))

"""
    ConstValueMutationPolicy(; modes=(:bias, :scale),
        array_ranks=(:global, :axis, :outer, :full),
        σ_bias=0.1, σ_scale=0.1, rerandomize_scale=1.0)

Composable value-level policy for scalar and array constants.

`modes` may contain `:bias`, `:scale`, and `:rerandomize`. Scale uses the
positive log-normal form `exp(σ_scale * randn(rng))`. For arrays, `array_ranks`
selects the broadcast shape for bias/scale perturbations: `:global`, `:axis`,
`:outer`, or `:full`.
"""
struct ConstValueMutationPolicy
    modes::Tuple{Vararg{Symbol}}
    array_ranks::Tuple{Vararg{Symbol}}
    σ_bias::Float64
    σ_scale::Float64
    rerandomize_scale::Float64
end

function ConstValueMutationPolicy(;
    modes::Tuple{Vararg{Symbol}}=(:bias, :scale),
    array_ranks::Tuple{Vararg{Symbol}}=(:global, :axis, :outer, :full),
    σ_bias::Real=0.1,
    σ_scale::Real=0.1,
    rerandomize_scale::Real=1.0,
)
    isempty(modes) && throw(ArgumentError("modes must be non-empty"))
    isempty(array_ranks) && throw(ArgumentError("array_ranks must be non-empty"))
    allowed_modes = (:bias, :scale, :rerandomize)
    allowed_ranks = (:global, :axis, :outer, :full)
    all(m -> m in allowed_modes, modes) || throw(ArgumentError("unsupported const mutation mode"))
    all(r -> r in allowed_ranks, array_ranks) || throw(ArgumentError("unsupported array rank mode"))
    σb = Float64(σ_bias)
    σs = Float64(σ_scale)
    rr = Float64(rerandomize_scale)
    isfinite(σb) && σb >= 0 || throw(ArgumentError("σ_bias must be finite and nonnegative"))
    isfinite(σs) && σs >= 0 || throw(ArgumentError("σ_scale must be finite and nonnegative"))
    isfinite(rr) && rr >= 0 || throw(ArgumentError("rerandomize_scale must be finite and nonnegative"))
    return ConstValueMutationPolicy(modes, array_ranks, σb, σs, rr)
end

ConstValueMutationPolicy(σ::Real) = ConstValueMutationPolicy(; σ_bias=σ, σ_scale=σ)

@inline _policy_mode(rng::AbstractRNG, policy::ConstValueMutationPolicy) = rand(rng, policy.modes)
@inline _policy_rank(rng::AbstractRNG, policy::ConstValueMutationPolicy) = rand(rng, policy.array_ranks)

function mutate_const_value(rng::AbstractRNG, value::AbstractFloat, ty::Ty, σ::Float64)
    ty == FloatTy || return Err(UnsupportedConstTyErr(ty))
    return mutate_const_value(rng, value, ty, ConstValueMutationPolicy(σ))
end

function mutate_const_value(rng::AbstractRNG, value::AbstractFloat, ty::Ty, policy::ConstValueMutationPolicy)
    ty == FloatTy || return Err(UnsupportedConstTyErr(ty))
    T = typeof(value)
    mode = _policy_mode(rng, policy)
    if mode === :bias
        return value + convert(T, randn(rng) * policy.σ_bias)
    elseif mode === :scale
        return value * convert(T, exp(randn(rng) * policy.σ_scale))
    elseif mode === :rerandomize
        return convert(T, randn(rng) * policy.rerandomize_scale)
    end
    return Err(ArgumentError("unsupported const mutation mode: $mode"))
end

function mutate_const_value(rng::AbstractRNG, value::Integer, ty::Ty, ::Float64)
    ty == IntTy || return Err(UnsupportedConstTyErr(ty))
    return value + convert(typeof(value), rand(rng, (-1, 1)))
end

function mutate_const_value(rng::AbstractRNG, value::Integer, ty::Ty, policy::ConstValueMutationPolicy)
    ty == IntTy || return Err(UnsupportedConstTyErr(ty))
    _policy_mode(rng, policy) === :rerandomize && return convert(typeof(value), round(Int, randn(rng) * policy.rerandomize_scale))
    return value + convert(typeof(value), rand(rng, (-1, 1)))
end

function mutate_const_value(::AbstractRNG, value::Bool, ty::Ty, ::Float64)
    ty == BoolTy || return Err(UnsupportedConstTyErr(ty))
    return !value
end

mutate_const_value(rng::AbstractRNG, value::Bool, ty::Ty, ::ConstValueMutationPolicy) =
    mutate_const_value(rng, value, ty, 0.0)

function mutate_const_value(rng::AbstractRNG, value::Complex, ty::Ty, σ::Float64)
    (ty isa TyConst && (ty::TyConst).name == :Complex) || return Err(UnsupportedConstTyErr(ty))
    return mutate_const_value(rng, value, ty, ConstValueMutationPolicy(σ))
end

function mutate_const_value(rng::AbstractRNG, value::Complex, ty::Ty, policy::ConstValueMutationPolicy)
    (ty isa TyConst && (ty::TyConst).name == :Complex) || return Err(UnsupportedConstTyErr(ty))
    R = real(typeof(value))
    mode = _policy_mode(rng, policy)
    if mode === :bias
        return value + complex(convert(R, randn(rng) * policy.σ_bias), convert(R, randn(rng) * policy.σ_bias))
    elseif mode === :scale
        return value * convert(R, exp(randn(rng) * policy.σ_scale))
    elseif mode === :rerandomize
        return complex(convert(R, randn(rng) * policy.rerandomize_scale), convert(R, randn(rng) * policy.rerandomize_scale))
    end
    return Err(ArgumentError("unsupported const mutation mode: $mode"))
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, σ::Float64) where {T<:AbstractFloat}
    _float_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    return mutate_const_value(rng, value, ty, ConstValueMutationPolicy(σ))
end

function mutate_const_value(rng::AbstractRNG, value::AbstractMatrix{T}, ty::Ty, σ::Float64) where {T<:AbstractFloat}
    _float_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    return mutate_const_value(rng, value, ty, ConstValueMutationPolicy(σ))
end

function _array_leaf_ty(ty::Ty)
    ty isa TyArray || return nothing
    elem = (ty::TyArray).elem
    while elem isa TyArray
        elem = (elem::TyArray).elem
    end
    return elem
end

_float_array_ty_supported(ty::Ty) = _array_leaf_ty(ty) == FloatTy
_int_array_ty_supported(ty::Ty) = _array_leaf_ty(ty) == IntTy

function _array_rank_shape(rng::AbstractRNG, dims::Tuple, rank::Symbol)
    if rank === :global
        return ntuple(_ -> 1, length(dims))
    elseif rank === :axis
        axis = rand(rng, 1:length(dims))
        return ntuple(i -> i == axis ? dims[i] : 1, length(dims))
    elseif rank === :outer && length(dims) >= 2
        return ntuple(i -> i <= 2 ? dims[i] : 1, length(dims))
    end
    return dims
end

function _mutate_float_array(rng::AbstractRNG, value::AbstractArray{T}, policy::ConstValueMutationPolicy) where {T<:AbstractFloat}
    mode = _policy_mode(rng, policy)
    rank = mode === :rerandomize ? :full : _policy_rank(rng, policy)
    shape = _array_rank_shape(rng, size(value), rank)
    noise = randn(rng, T, shape)
    if mode === :bias
        return value .+ noise .* convert(T, policy.σ_bias)
    elseif mode === :scale
        return value .* exp.(noise .* convert(T, policy.σ_scale))
    elseif mode === :rerandomize
        return randn(rng, T, size(value)...) .* convert(T, policy.rerandomize_scale)
    end
    return Err(ArgumentError("unsupported const mutation mode: $mode"))
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, policy::ConstValueMutationPolicy) where {T<:AbstractFloat}
    _float_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    return _mutate_float_array(rng, value, policy)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, ::Float64) where {T<:Integer}
    _int_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    delta = rand(rng, (-1, 1), size(value)...)
    return value .+ convert.(T, delta)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractMatrix{T}, ty::Ty, ::Float64) where {T<:Integer}
    _int_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    delta = rand(rng, (-1, 1), size(value)...)
    return value .+ convert.(T, delta)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, policy::ConstValueMutationPolicy) where {T<:Integer}
    _int_array_ty_supported(ty) || return Err(UnsupportedConstTyErr(ty))
    if _policy_mode(rng, policy) === :rerandomize
        return convert.(T, round.(Int, randn(rng, size(value)...) .* policy.rerandomize_scale))
    end
    return mutate_const_value(rng, value, ty, 0.0)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{A}, ty::Ty, σ::Float64) where {A<:AbstractArray}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    (ty::TyArray).elem isa TyArray || return Err(UnsupportedConstTyErr(ty))
    inner_ty = (ty.elem::TyArray)
    out = Vector{Any}(undef, length(value))
    for (j, v0) in enumerate(value)
        v = mutate_const_value(rng, v0, inner_ty, σ)
        v isa Err && return v
        out[j] = v
    end
    return out
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{A}, ty::Ty, policy::ConstValueMutationPolicy) where {A<:AbstractArray}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    (ty::TyArray).elem isa TyArray || return Err(UnsupportedConstTyErr(ty))
    inner_ty = (ty.elem::TyArray)
    out = Vector{Any}(undef, length(value))
    for (j, v0) in enumerate(value)
        v = mutate_const_value(rng, v0, inner_ty, policy)
        v isa Err && return v
        out[j] = v
    end
    return out
end

function _apply_local_const_mutation!(tree::AstTree, site::Site, ctx::NamedTuple, new_value)
    new_value isa Err && return new_value
    factory = haskey(ctx, :const_symfactory) ? ctx.const_symfactory : ConstSymbolFactory()
    new_node = new_constant!(ctx.local_consts, new_value, factory)
    ctx.local_consts_ty[new_node.name] = site.ty
    _replace_site!(tree, site, AstTree(new_node))
    return tree
end

"""
    ScalarConstMutation(; σ=0.1) <: AbstractMutation

Scalar local constant mutation.
"""
struct ScalarConstMutation <: AbstractMutation
    σ::Float64
    policy::ConstValueMutationPolicy
end
ScalarConstMutation(; σ::Real=0.1, policy::ConstValueMutationPolicy=ConstValueMutationPolicy(σ)) =
    ScalarConstMutation(Float64(σ), policy)
ScalarConstMutation(σ::Real) = ScalarConstMutation(; σ)
ScalarConstMutation(policy::ConstValueMutationPolicy) = ScalarConstMutation(; policy)
required_site_kinds(::ScalarConstMutation) = Symbol[:const_local]

function is_site_eligible(::ScalarConstMutation, site::Site, ::MutationEnv, ctx::NamedTuple)
    site.node isa ConstNode || return false
    _is_local_const(site.node.name, ctx) || return false
    value = ctx.local_consts[site.node.name]
    value isa AbstractArray && return false
    value isa Bool && return false
    return value isa Number
end

function apply(m::ScalarConstMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    node = site.node::ConstNode
    _is_local_const(node.name, ctx) || return Err(NoEligibleSiteErr())
    value = ctx.local_consts[node.name]
    new_value = mutate_const_value(env.rng, value, site.ty, m.policy)
    return _apply_local_const_mutation!(tree, site, ctx, new_value)
end

"""
    BoolConstMutation() <: AbstractMutation

Boolean local constant mutation: flips the boolean.
"""
struct BoolConstMutation <: AbstractMutation end
required_site_kinds(::BoolConstMutation) = Symbol[:const_local]

function is_site_eligible(::BoolConstMutation, site::Site, ::MutationEnv, ctx::NamedTuple)
    site.node isa ConstNode || return false
    _is_local_const(site.node.name, ctx) || return false
    return ctx.local_consts[site.node.name] isa Bool
end

function apply(::BoolConstMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    node = site.node::ConstNode
    _is_local_const(node.name, ctx) || return Err(NoEligibleSiteErr())
    value = ctx.local_consts[node.name]
    new_value = mutate_const_value(env.rng, value, site.ty, 0.0)
    return _apply_local_const_mutation!(tree, site, ctx, new_value)
end

"""
    ArrayConstMutation(; σ=0.1) <: AbstractMutation

Array/container local constant mutation.

This is intended for `TyArray`-typed constants, including nested containers
(`TyArray(_, TyArray(_, _))`). Implementations should prefer broadcasted/bulk
operations over scalar loops.
"""
struct ArrayConstMutation <: AbstractMutation
    σ::Float64
    policy::ConstValueMutationPolicy
end
ArrayConstMutation(; σ::Real=0.1, policy::ConstValueMutationPolicy=ConstValueMutationPolicy(σ)) =
    ArrayConstMutation(Float64(σ), policy)
ArrayConstMutation(σ::Real) = ArrayConstMutation(; σ)
ArrayConstMutation(policy::ConstValueMutationPolicy) = ArrayConstMutation(; policy)
required_site_kinds(::ArrayConstMutation) = Symbol[:const_local]

function is_site_eligible(::ArrayConstMutation, site::Site, ::MutationEnv, ctx::NamedTuple)
    site.node isa ConstNode || return false
    _is_local_const(site.node.name, ctx) || return false
    return ctx.local_consts[site.node.name] isa AbstractArray
end

function apply(m::ArrayConstMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    node = site.node::ConstNode
    _is_local_const(node.name, ctx) || return Err(NoEligibleSiteErr())
    value = ctx.local_consts[node.name]
    new_value = mutate_const_value(env.rng, value, site.ty, m.policy)
    return _apply_local_const_mutation!(tree, site, ctx, new_value)
end

"""
    ConstMutation(; σ=0.1) <: AbstractMutation

Compatibility wrapper combining scalar/bool/array constant mutations.
"""
struct ConstMutation <: AbstractMutation
    scalar::ScalarConstMutation
    bool::BoolConstMutation
    array::ArrayConstMutation
end

function ConstMutation(; σ::Float64=0.1)
    return ConstMutation(ScalarConstMutation(σ), BoolConstMutation(), ArrayConstMutation(σ))
end
required_site_kinds(::ConstMutation) = Symbol[:const_local]

_const_delegate(m::ConstMutation, ::Bool) = m.bool
_const_delegate(m::ConstMutation, ::AbstractArray) = m.array
_const_delegate(m::ConstMutation, ::Any) = m.scalar

function is_site_eligible(m::ConstMutation, site::Site, env::MutationEnv, ctx::NamedTuple)
    site.node isa ConstNode || return false
    _is_local_const(site.node.name, ctx) || return false
    value = ctx.local_consts[site.node.name]
    return is_site_eligible(_const_delegate(m, value), site, env, ctx)
end

function apply(m::ConstMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    node = site.node::ConstNode
    _is_local_const(node.name, ctx) || return Err(NoEligibleSiteErr())
    value = ctx.local_consts[node.name]
    return apply(_const_delegate(m, value), tree, site, env, ctx)
end

"""
    PointMutation() <: AbstractMutation

Point-wise structural mutation.

Two sub-operators are implemented:
- `IndexNode` swap: choose another compatible binder index (type-compatible).
- global `ConstNode` replacement: replace a non-local constant symbol with a
  type-compatible symbol from the current `type_ctx`.

Eligibility is strict and checked before attempting the mutation.
"""
struct PointMutation <: AbstractMutation end
required_site_kinds(::PointMutation) = Symbol[:index, :const_global]

function is_site_eligible(::PointMutation, site::Site, env::MutationEnv, ctx::NamedTuple)
    if site.node isa IndexNode
        depth = length(site.binder_types)
        depth > 0 || return false
        cur = site.node.index
        cur < depth || return false
        node_ty = site.ty
        candidates = Int[]
        for idx in 0:(depth-1)
            idx == cur && continue
            binder_ty = site.binder_types[end - idx]
            iscompatible(node_ty, binder_ty) || continue
            push!(candidates, idx)
        end
        return !isempty(candidates)
    elseif site.node isa ConstNode
        node = site.node::ConstNode
        _is_local_const(node.name, ctx) && return false
        node_ty = site.ty
        for (nm, ty) in env.type_ctx
            nm == node.name && continue
            iscompatible(ty, node_ty) || continue
            return true
        end
        return false
    else
        return false
    end
end

function apply(::PointMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    if site.node isa IndexNode
        depth = length(site.binder_types)
        cur = site.node.index
        node_ty = site.ty
        candidates = Int[]
        for idx in 0:(depth-1)
            idx == cur && continue
            binder_ty = site.binder_types[end - idx]
            iscompatible(node_ty, binder_ty) || continue
            push!(candidates, idx)
        end
        isempty(candidates) && return Err(NoEligibleSiteErr())
        new_idx = rand(env.rng, candidates)
        _replace_site!(tree, site, AstTree(IndexNode(new_idx)))
        return tree
    elseif site.node isa ConstNode
        node = site.node::ConstNode
        node_ty = site.ty
        candidates = Symbol[]
        for (nm, ty) in env.type_ctx
            nm == node.name && continue
            _is_local_const(nm, ctx) && continue
            iscompatible(ty, node_ty) || continue
            push!(candidates, nm)
        end
        isempty(candidates) && return Err(NoEligibleSiteErr())
        new_name = rand(env.rng, candidates)
        _replace_site!(tree, site, AstTree(ConstNode(new_name)))
        return tree
    else
        return Err(NoEligibleSiteErr())
    end
end

"""
    BetaReduceMutation() <: AbstractMutation

Single-step beta reduction (β-reduction) at an application site.

Eligibility: `site.node isa AppNode` and `site.node.func isa AbsNode`.

This is semantics-preserving and should keep the inferred type unchanged.
"""
struct BetaReduceMutation <: AbstractMutation end
required_site_kinds(::BetaReduceMutation) = Symbol[:app]

function is_site_eligible(::BetaReduceMutation, site::Site, ::MutationEnv, ::NamedTuple)
    site.node isa AppNode || return false
    return (site.node::AppNode).func isa AbsNode
end

function apply(::BetaReduceMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    site.node isa AppNode || return Err(NoEligibleSiteErr())
    app = site.node::AppNode
    app.func isa AbsNode || return Err(NoEligibleSiteErr())
    abs = app.func::AbsNode
    arg = app.arg
    body_tree = apply_node!(tree, abs, arg)
    _replace_site!(tree, site, body_tree)
    return tree
end

function _occurs_as_eta_binder(f_tree::AstTree)
    found = false
    traverse(f_tree, AbstractionDepthState) do _, node, bag
        node isa IndexNode || return nothing
        depth = getstate(bag, AbstractionDepthState).depth
        if (node::IndexNode).index == depth
            found = true
            return Break()
        end
        return nothing
    end
    return found
end

"""
    EtaReduceMutation() <: AbstractMutation

Safe eta reduction (η-reduction) at an abstraction site.

Eligibility: matches `λx. f #0` and `#0` does not occur free in `f`.

Implementation note: the "x not free in f" check uses De Bruijn indices and
`AbstractionDepthState` to avoid capture issues.
"""
struct EtaReduceMutation <: AbstractMutation end
required_site_kinds(::EtaReduceMutation) = Symbol[:abs]

function is_site_eligible(::EtaReduceMutation, site::Site, ::MutationEnv, ::NamedTuple)
    site.node isa AbsNode || return false
    abs = site.node::AbsNode
    abs.body isa AppNode || return false
    app = abs.body::AppNode
    app.arg isa IndexNode || return false
    (app.arg::IndexNode).index == 0 || return false
    f_tree = AstTree(app.func)
    _occurs_as_eta_binder(f_tree) && return false
    return true
end

function apply(::EtaReduceMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    is_site_eligible(EtaReduceMutation(), site, env, ctx) || return Err(NoEligibleSiteErr())
    abs = site.node::AbsNode
    app = abs.body::AppNode
    f_tree = AstTree(deepcopy(app.func))
    shift_free_variables!(f_tree, -1, 1)
    _replace_site!(tree, site, f_tree)
    return tree
end

"""
    EtaExpandMutation() <: AbstractMutation

Safe eta expansion (η-expansion) at a function-typed site.

Rewrites `f` to `λx. f #0`, preserving the inferred type (up to `iscompatible`),
while increasing structural depth for later CSE/beta moves.

Eligibility: `site.ty` must be (prune-)function-shaped and `site.node` must not be
an `AbsNode` (to avoid introducing an immediate beta-redex).
"""
struct EtaExpandMutation <: AbstractMutation end
required_site_kinds(::EtaExpandMutation) = Symbol[:any]

function is_site_eligible(::EtaExpandMutation, site::Site, ::MutationEnv, ::NamedTuple)
    TypeSystem.prune(site.ty) isa TyFun || return false
    site.node isa AbsNode && return false
    return true
end

function apply(::EtaExpandMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    is_site_eligible(EtaExpandMutation(), site, env, ctx) || return Err(NoEligibleSiteErr())
    f_tree = AstTree(deepcopy(site.node))
    shift_free_variables!(f_tree, 1, 0)
    name = haskey(ctx, :symfactory) ? ctx.symfactory() : SymbolGenerator()()
    new_subtree = AbsNode(name, AppNode(f_tree.root, IndexNode(0)))
    _replace_site!(tree, site, AstTree(new_subtree))
    return tree
end

"""
    CondAppPullMutation() <: AbstractMutation

Semantics-preserving commuting conversion for `COND` (pull application out):

`COND b (f x) (g x)`  ↦  `(COND b f g) x`

This is useful when `COND` is polymorphic (so it can be instantiated at a function type),
allowing later `EtaReduceMutation` to turn `λx. COND b (f x) (g x)` into `COND b f g`.

Eligibility: matches the literal application spine `((COND b) (f x)) (g x)` with the
same argument `x` in both branches.
"""
struct CondAppPullMutation <: AbstractMutation end
required_site_kinds(::CondAppPullMutation) = Symbol[:app]

function is_site_eligible(::CondAppPullMutation, site::Site, ::MutationEnv, ::NamedTuple)
    site.node isa AppNode || return false
    outer = site.node::AppNode # ((COND b then) else)
    outer.func isa AppNode || return false
    mid = outer.func::AppNode  # (COND b then)
    mid.func isa AppNode || return false
    head = mid.func::AppNode   # (COND b)
    head.func isa ConstNode || return false
    (head.func::ConstNode).name == :COND || return false

    then_branch = mid.arg
    else_branch = outer.arg
    then_branch isa AppNode || return false
    else_branch isa AppNode || return false
    (then_branch::AppNode).arg == (else_branch::AppNode).arg || return false
    return true
end

function apply(::CondAppPullMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    is_site_eligible(CondAppPullMutation(), site, env, ctx) || return Err(NoEligibleSiteErr())
    outer = site.node::AppNode
    mid = outer.func::AppNode
    head = mid.func::AppNode
    then_branch = mid.arg::AppNode
    else_branch = outer.arg::AppNode

    b = deepcopy(head.arg)
    f = deepcopy(then_branch.func)
    g = deepcopy(else_branch.func)
    x = deepcopy(then_branch.arg)
    new_subtree = AppNode(AppNode(AppNode(AppNode(ConstNode(:COND), b), f), g), x)
    _replace_site!(tree, site, AstTree(new_subtree))
    return tree
end

"""
    CondAppPushMutation() <: AbstractMutation

Inverse commuting conversion for `COND` (push application into both branches):

`(COND b f g) x`  ↦  `COND b (f x) (g x)`

This can increase tree size; it is *not* enabled by default in `default_mutations()`.

Eligibility: matches the literal application spine `(((COND b) f) g) x`.
"""
struct CondAppPushMutation <: AbstractMutation end
required_site_kinds(::CondAppPushMutation) = Symbol[:app]

function is_site_eligible(::CondAppPushMutation, site::Site, ::MutationEnv, ::NamedTuple)
    site.node isa AppNode || return false
    outer = site.node::AppNode # (cond3 x)
    outer.func isa AppNode || return false
    cond3 = outer.func::AppNode # ((COND b f) g)
    cond3.func isa AppNode || return false
    cond2 = cond3.func::AppNode # (COND b f)
    cond2.func isa AppNode || return false
    head = cond2.func::AppNode  # (COND b)
    head.func isa ConstNode || return false
    (head.func::ConstNode).name == :COND || return false
    return true
end

function apply(::CondAppPushMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    is_site_eligible(CondAppPushMutation(), site, env, ctx) || return Err(NoEligibleSiteErr())
    outer = site.node::AppNode
    x = deepcopy(outer.arg)
    cond3 = outer.func::AppNode
    cond2 = cond3.func::AppNode
    head = cond2.func::AppNode

    b = deepcopy(head.arg)
    f = deepcopy(cond2.arg)
    g = deepcopy(cond3.arg)
    new_subtree = AppNode(
        AppNode(
            AppNode(ConstNode(:COND), b),
            AppNode(f, deepcopy(x)),
        ),
        AppNode(g, x),
    )
    _replace_site!(tree, site, AstTree(new_subtree))
    return tree
end

"""
    CSEMutation(; min_count=1) <: AbstractMutation

Semantics-preserving common subexpression extraction.

This mutation chooses an anchor subexpression, canonicalizes it (alpha-invariant),
and delegates to the node-anchored `ASTs.CSE.eliminate_node!` path, which
compares De Bruijn captures by binder identity before rewriting the tree by
introducing a beta-redex around the LCA of matches.

When `min_count=1`, this acts as a "zero-cost" structural move: it can extract a
single occurrence to increase structural diversity without changing semantics.
Single-lift mode admits non-index sites. Repeated CSE mode (`min_count>1`) keeps
the default candidate filter to applications so it avoids low-value repeated
variables and constants.
"""
struct CSEMutation <: AbstractMutation
    min_count::Int
end
CSEMutation(; min_count::Int=1) = CSEMutation(min_count)
required_site_kinds(::CSEMutation) = Symbol[:any]

function is_site_eligible(m::CSEMutation, site::Site, env::MutationEnv, ::NamedTuple)
    m.min_count <= 1 && return !(site.node isa IndexNode)
    site.node isa AppNode || return false
    env.tree === nothing && return true
    cache_key = Symbol("cse_repeated_nodes_", m.min_count)
    eligible = get!(env.eligibility_cache, cache_key) do
        repeated_nodes(env.tree; min_count=m.min_count)
    end
    eligible isa Err && return true
    return get(eligible, site.node, false)
end

function apply(m::CSEMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    return eliminate_node!(tree, site.node; min_count=m.min_count, factory=ctx.symfactory)
end

function _count_nodes(tree::AstTree)
    c = 0
    for _ in tree
        c += 1
    end
    return c
end

"""Return a fresh authoritative baseline mutation set."""
function default_mutations()
    return AbstractMutation[
        ParameterIntroductionMutation(),
        ScalarConstMutation(),
        BoolConstMutation(),
        ArrayConstMutation(),
        PointMutation(),
        CSEMutation(),
        BetaReduceMutation(),
        EtaExpandMutation(),
        EtaReduceMutation(),
        LetHoistMutation(),
        LetSinkMutation(),
        LetPruneMutation(),
        SourceBindingInsertionMutation(),
    ]
end

function _require_ctx_fields(ctx::NamedTuple, fields::Vector{Symbol})
    for f in fields
        haskey(ctx, f) || throw(ArgumentError("Missing required context field: :$f"))
    end
    return nothing
end

"""
    mutate_tree(tree::AstTree, target_ty::Ty; kwargs...) -> Union{AstTree,Err}

Apply a single mutation step that preserves `target_ty` (up to `iscompatible`).
Returns `Err(MaxTryReachedErr())` if no valid mutation can be found within `max_try` attempts.

# Key keyword arguments
- `mutations`: available mutation operators (default: MVP set)
- `chooser`: how to pick `(mutation, site)` (default: `WeightedPairChooser()`)
- `provider`: how to build sites (default: `IndexedProvider()`)
- `copy_tree`: run mutation on a copy of the input tree (default: `true`)
- `copy_ctx`: transactional mutation over constant dictionaries (default: `true`)
- `prune_consts`: call `prune_local_consts!` after a successful mutation (default: `true`)
- `global_consts_ty`, `local_consts`, `local_consts_ty`: constant/type containers
- `global_consts`: optional runtime values for global constants used by generated replacements
- `const_symfactory`: per-call deterministic local-constant symbol factory (default: `ConstSymbolFactory()`)
- `tyfactory`, `symfactory`: local factories used by type inference and CSE
- `type_inference`: four-argument callable used for site typing and final validation

## Optional policies (off by default)
- `bloat_policy::BloatPolicy`
- `adaptive_policy::AdaptivePolicy`
"""
function mutate_tree(tree::AstTree, target_ty::Ty;
    rng::AbstractRNG=MersenneTwister(2000),
    mutations::Vector{<:AbstractMutation}=default_mutations(),
    chooser::AbstractPairChooser=WeightedPairChooser(),
    provider::AbstractSiteProvider=IndexedProvider(),
    max_try::Int=20,
    copy_tree::Bool=true,
    copy_ctx::Bool=true,
    prune_consts::Bool=true,
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    local_consts_ty::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    local_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    tyfactory=newtyvar,
    type_inference=_default_mutation_inference,
    symfactory=SymbolGenerator(),
    const_symfactory::ConstSymbolFactory=ConstSymbolFactory(),
    bloat_policy::BloatPolicy=BloatPolicy(),
    adaptive_policy::AdaptivePolicy=AdaptivePolicy(),
    recorder::Union{Nothing,MutationAttemptRecorder}=nothing,
)
    current_tree = copy_tree ? deepcopy(tree) : tree
    # Transactional: always search with scratch dicts; on success, commit to caller dicts.
    caller_local_consts = local_consts
    caller_local_consts_ty = local_consts_ty
    current_local_consts = copy(caller_local_consts)
    current_local_consts_ty = copy(caller_local_consts_ty)

    # Prepare sites once (the retry loop works on copies of the same input).
    base_type_ctx = merge(global_consts_ty, current_local_consts_ty)
    base_ctx = (;
        rng=rng,
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts_ty=current_local_consts_ty,
        local_consts=current_local_consts,
        tyfactory=tyfactory,
        symfactory=symfactory,
        const_symfactory=const_symfactory,
        type_inference=type_inference,
    )
    env = MutationEnv(
        rng,
        target_ty,
        base_type_ctx,
        current_tree,
        nothing,
        chooser isa WeightedPairChooser ? (chooser::WeightedPairChooser).cache_eligibility : false,
        IdDict{Any,Any}(),
        adaptive_policy,
        IdDict{DataType,MutationStat}(),
        bloat_policy,
    )

    reset!(provider)
    prep = prepare!(provider, current_tree, base_type_ctx, env, base_ctx)
    if prep isa Err
        record!(recorder, nothing, :prepare_err; err_type=typeof(prep.value))
        return prep
    end

    for _ in 1:max_try
        attempt_tree = deepcopy(current_tree)
        attempt_local_consts = copy(current_local_consts)
        attempt_local_consts_ty = copy(current_local_consts_ty)
        attempt_type_ctx = merge(global_consts_ty, attempt_local_consts_ty)
        attempt_ctx = (;
            rng=rng,
            global_consts=global_consts,
            global_consts_ty=global_consts_ty,
            local_consts_ty=attempt_local_consts_ty,
            local_consts=attempt_local_consts,
            tyfactory=tyfactory,
            symfactory=symfactory,
            const_symfactory=const_symfactory,
            type_inference=type_inference,
        )

        choice = choose_pair(chooser, mutations, provider, current_tree, base_type_ctx, env, base_ctx)
        if choice isa Err
            record!(recorder, nothing, :choose_err; err_type=typeof(choice.value))
            return choice
        end

        m = choice.mutation::AbstractMutation
        attempt_site = _remap_site(attempt_tree, choice.site)
        mutated = apply(m, attempt_tree, attempt_site, env, attempt_ctx)
        if mutated isa Err
            _update_stat!(env, m, false)
            if mutated.value isa NoEligibleSiteErr
                record!(recorder, typeof(m), :no_eligible_site; err_type=typeof(mutated.value))
            elseif mutated.value isa BloatExceededErr
                record!(recorder, typeof(m), :bloat_exceeded; err_type=typeof(mutated.value))
            else
                record!(recorder, typeof(m), :apply_err; err_type=typeof(mutated.value))
            end
            continue
        end

        if bloat_policy.enabled
            before_nodes = length(current_tree)
            after_nodes = length(attempt_tree)
            if after_nodes > bloat_policy.max_nodes ||
               after_nodes - before_nodes > bloat_policy.max_nodes_delta
                _update_stat!(env, m, false)
                record!(recorder, typeof(m), :bloat_exceeded; err_type=BloatExceededErr)
                continue
            end
        end

        prune_consts && prune_local_consts!(attempt_tree, attempt_local_consts, attempt_local_consts_ty)
        attempt_type_ctx = merge(global_consts_ty, attempt_local_consts_ty)
        inferred = _run_mutation_inference(
            attempt_ctx,
            attempt_tree,
            attempt_type_ctx,
            tyfactory,
            nothing,
        )
        if inferred isa Err
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :infer_err; err_type=typeof(inferred.value))
            continue
        end
        if !iscompatible(inferred, target_ty)
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :incompatible)
            continue
        end

        _update_stat!(env, m, true)
        record!(recorder, typeof(m), :success)
        current_tree = attempt_tree
        current_local_consts = attempt_local_consts
        current_local_consts_ty = attempt_local_consts_ty
        if copy_ctx
            empty!(caller_local_consts)
            empty!(caller_local_consts_ty)
            merge!(caller_local_consts, current_local_consts)
            merge!(caller_local_consts_ty, current_local_consts_ty)
        else
            # If the caller wants in-place updates during the search, they can pass mutable dicts
            # that are already per-individual; commit at the end anyway for consistency.
            empty!(caller_local_consts)
            empty!(caller_local_consts_ty)
            merge!(caller_local_consts, current_local_consts)
            merge!(caller_local_consts_ty, current_local_consts_ty)
        end
        return current_tree
    end
    record!(recorder, nothing, :max_try_exhausted)
    return Err(MaxTryReachedErr())
end

include("ScopedRewrite.jl")
include("BindingInsertion.jl")
include("Graft.jl")
include("ProductPacking.jl")
include("Population.jl")

########################################################
# Test Items
########################################################

@testsnippet MutBaseImports begin
    using LambdaRegression.Mutations: prune_local_consts!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using Random
    using LambdaRegression.Mutations: mutate_tree, ScalarConstMutation, WeightedPairChooser, IndexedProvider, ConstValueMutationPolicy
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Mutations: mutate_tree, BoolConstMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.TypeSystem: BoolTy, TyLike
    using LambdaRegression.Mutations: mutate_tree, ArrayConstMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.TypeSystem: TyArray, TyNat, FloatTy, TyLike
    using LambdaRegression.Mutations: mutate_tree, PointMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike
    using LambdaRegression.Utils: Err
    using LambdaRegression.Mutations: WeightedPairChooser, IndexedProvider, ScalarConstMutation
    using LambdaRegression.TypeSystem: FloatTy, TyLike, newtyvar
    using LambdaRegression.Mutations: IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyLike, FloatTy, TyFun, newtyvar
    using LambdaRegression.Mutations: mutate_tree, WeightedPairChooser
    using LambdaRegression.Mutations: mutate_tree, BetaReduceMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, IndexNode, ConstNode
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.Mutations: mutate_tree, EtaReduceMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, IndexNode, ConstNode
    using LambdaRegression.Mutations: mutate_tree, EtaExpandMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.Mutations: mutate_tree, CSEMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike, infer, iscompatible
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, ConstNode
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory
    using LambdaRegression.Mutations: mutate_tree, BoolConstMutation, WeightedPairChooser, IndexedProvider, MutationAttemptRecorder, mutation_recorder_summary
end

@testitem "Mutations prune_local_consts! keeps only referenced" setup=[MutBaseImports] begin
    local_consts = Dict{Symbol,Any}(:a => 1.0, :b => 2.0, :c => 3.0)
    local_consts_ty = Dict{Symbol,TyLike}(:a => FloatTy, :b => FloatTy, :c => FloatTy)
    tree = AstTree(AppNode(ConstNode(:a), ConstNode(:a))) # references only :a
    prune_local_consts!(tree, local_consts, local_consts_ty)
    @test haskey(local_consts, :a)
    @test !haskey(local_consts, :b)
    @test !haskey(local_consts, :c)
    @test haskey(local_consts_ty, :a)
    @test !haskey(local_consts_ty, :b)
    @test !haskey(local_consts_ty, :c)
end

@testitem "Mutations ScalarConstMutation FloatTy single-node replace + type stable" setup=[MutBaseImports] begin
    rng = MersenneTwister(1)
    name = :k
    local_consts = Dict{Symbol,Any}(name => 1.25)
    local_consts_ty = Dict{Symbol,TyLike}(name => FloatTy)
    tree = AstTree(ConstNode(name))
    mutated = mutate_tree(
        tree,
        FloatTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[ScalarConstMutation(σ=0.2)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation=>1.0), cache_eligibility=false),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )
    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    @test mutated.root.name != name
    @test haskey(local_consts, mutated.root.name)
    @test local_consts_ty[mutated.root.name] == FloatTy
    @test local_consts[mutated.root.name] isa AbstractFloat
end

@testitem "Mutations BoolConstMutation flip + type stable" setup=[MutBaseImports] begin
    rng = MersenneTwister(1)
    name = :b
    local_consts = Dict{Symbol,Any}(name => true)
    local_consts_ty = Dict{Symbol,TyLike}(name => BoolTy)
    tree = AstTree(ConstNode(name))
    mutated = mutate_tree(
        tree,
        BoolTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[BoolConstMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(BoolConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )
    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    @test mutated.root.name != name
    @test haskey(local_consts, mutated.root.name)
    @test local_consts_ty[mutated.root.name] == BoolTy
    @test local_consts[mutated.root.name] isa Bool
end

@testitem "Mutations ArrayConstMutation nested float array preserves shape" setup=[MutBaseImports] begin
    rng = MersenneTwister(1111)
    name = :A
    value = [rand(rng, Float64, 4) for _ in 1:3] # vector-of-vector "matrix"
    const_ty = TyArray(TyNat(3), TyArray(TyNat(4), FloatTy))
    local_consts = Dict{Symbol,Any}(name => value)
    local_consts_ty = Dict{Symbol,TyLike}(name => const_ty)
    tree = AstTree(ConstNode(name))
    mutated = mutate_tree(
        tree,
        const_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[ArrayConstMutation(σ=0.1)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ArrayConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )
    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    @test mutated.root.name != name
    new_val = local_consts[mutated.root.name]
    @test new_val isa AbstractVector
    @test length(new_val) == length(value)
    @test all(length.(new_val) .== length.(value))
end

@testitem "Mutations ScalarConstMutation supports lognormal scale policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(42)
    name = :k
    local_consts = Dict{Symbol,Any}(name => 2.0)
    local_consts_ty = Dict{Symbol,TyLike}(name => FloatTy)
    mutation = ScalarConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:scale,), σ_scale=0.4),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        FloatTy;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    @test local_consts[(mutated.root::ConstNode).name] > 0
    @test local_consts[(mutated.root::ConstNode).name] != 2.0
end

@testitem "Mutations ScalarConstMutation supports rerandomize policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(555)
    name = :k
    local_consts = Dict{Symbol,Any}(name => 2.0)
    local_consts_ty = Dict{Symbol,TyLike}(name => FloatTy)
    mutation = ScalarConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:rerandomize,), rerandomize_scale=1.0),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        FloatTy;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    @test local_consts[(mutated.root::ConstNode).name] != 2.0
end

@testitem "Mutations constant mutations preserve legacy sigma constructor access" setup=[MutBaseImports] begin
    scalar = ScalarConstMutation(0.2)
    array = ArrayConstMutation(0.3)

    @test scalar.σ == 0.2
    @test scalar.policy isa ConstValueMutationPolicy
    @test scalar.policy.σ_bias == 0.2
    @test scalar.policy.σ_scale == 0.2
    @test array.σ == 0.3
    @test array.policy isa ConstValueMutationPolicy
    @test array.policy.σ_bias == 0.3
    @test array.policy.σ_scale == 0.3
end

@testitem "Mutations ArrayConstMutation supports global lognormal scale policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(2000)
    name = :A
    value = [1.0 2.0; 3.0 4.0]
    const_ty = TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    local_consts = Dict{Symbol,Any}(name => value)
    local_consts_ty = Dict{Symbol,TyLike}(name => const_ty)
    mutation = ArrayConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:scale,), array_ranks=(:global,), σ_scale=0.4),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        const_ty;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ArrayConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    out = local_consts[(mutated.root::ConstNode).name]
    ratio = out ./ value
    @test all(>(0), ratio)
    @test length(unique(round.(vec(ratio); digits=10))) == 1
end

@testitem "Mutations ArrayConstMutation supports full-rank bias policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(1)
    name = :A
    value = [1.0 2.0; 3.0 4.0]
    const_ty = TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    local_consts = Dict{Symbol,Any}(name => value)
    local_consts_ty = Dict{Symbol,TyLike}(name => const_ty)
    mutation = ArrayConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:bias,), array_ranks=(:full,), σ_bias=0.4),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        const_ty;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ArrayConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    out = local_consts[(mutated.root::ConstNode).name]
    @test size(out) == size(value)
    @test out != value
    @test length(unique(round.(vec(out .- value); digits=10))) > 1
end

@testitem "Mutations ArrayConstMutation supports axis-wise bias policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(1111)
    name = :A
    value = [1.0 2.0 3.0; 4.0 5.0 6.0]
    const_ty = TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    local_consts = Dict{Symbol,Any}(name => value)
    local_consts_ty = Dict{Symbol,TyLike}(name => const_ty)
    mutation = ArrayConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:bias,), array_ranks=(:axis,), σ_bias=0.4),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        const_ty;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ArrayConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    out = local_consts[(mutated.root::ConstNode).name]
    delta = out .- value
    rowwise = all(isapprox(delta[:, j], delta[:, 1]) for j in axes(delta, 2))
    colwise = all(isapprox(delta[i, :], delta[1, :]) for i in axes(delta, 1))
    @test rowwise || colwise
    @test out != value
end

@testitem "Mutations ArrayConstMutation supports outer-rank Float NDArray policy" setup=[MutBaseImports] begin
    rng = MersenneTwister(1234)
    name = :A
    value = reshape(collect(1.0:8.0), 2, 2, 2)
    const_ty = TyArray(TyNat(2), TyArray(TyNat(2), TyArray(TyNat(2), FloatTy)))
    local_consts = Dict{Symbol,Any}(name => value)
    local_consts_ty = Dict{Symbol,TyLike}(name => const_ty)
    mutation = ArrayConstMutation(;
        policy=ConstValueMutationPolicy(modes=(:bias,), array_ranks=(:outer,), σ_bias=0.4),
    )
    mutated = mutate_tree(
        AstTree(ConstNode(name)),
        const_ty;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ArrayConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
    )

    @test mutated isa AstTree
    out = local_consts[(mutated.root::ConstNode).name]
    delta = out .- value
    @test size(out) == size(value)
    @test isapprox(delta[:, :, 1], delta[:, :, 2])
    @test length(unique(round.(vec(delta[:, :, 1]); digits=10))) > 1
end

@testitem "Mutations ParameterIntroductionMutation creates a typed scalar parameter" setup=[MutBaseImports] begin
    using LambdaRegression.Mutations:
        ParameterIntroductionMutation, PathSiteSelectionPolicy
    using LambdaRegression.TypeSystem: FADD, SQR, get_type_ctx, infer, iscompatible

    tree = AstTree(AbsNode(
        :x,
        AppNode(
            AppNode(
                ConstNode(:FADD),
                AppNode(ConstNode(:SQR), IndexNode(0)),
            ),
            IndexNode(0),
        ),
    ))
    target_ty = TyFun(FloatTy, FloatTy)
    local_consts = Dict{Symbol,Any}()
    local_consts_ty = Dict{Symbol,TyLike}()
    mutation = ParameterIntroductionMutation(
        σ=0.5,
        site_selection=PathSiteSelectionPolicy([1, 1, 2]),
    )
    globals = Dict{Symbol,Any}(:FADD => FADD, :SQR => SQR)
    global_types = get_type_ctx(globals)

    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1234),
        global_consts=globals,
        global_consts_ty=global_types,
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(ParameterIntroductionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated isa AstTree
    @test mutated.root.body.func.arg isa ConstNode
    name = mutated.root.body.func.arg.name
    @test local_consts[name] isa Float64
    @test local_consts_ty[name] == FloatTy
    @test tree.root.body.func.arg isa AppNode
    inferred = infer(mutated, merge(global_types, local_consts_ty))
    @test !(inferred isa Err)
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations ParameterIntroductionMutation creates dense tensor blocks" setup=[MutBaseImports] begin
    using LambdaRegression.Mutations: ParameterIntroductionMutation

    tensor_ty = TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    tree = AstTree(ConstNode(:INPUT))
    global_types = Dict{Symbol,TyLike}(:INPUT => tensor_ty)
    local_consts = Dict{Symbol,Any}()
    local_consts_ty = Dict{Symbol,TyLike}()
    mutation = ParameterIntroductionMutation(σ=0.25)

    mutated = mutate_tree(
        tree,
        tensor_ty;
        rng=MersenneTwister(8000),
        global_consts=Dict{Symbol,Any}(:INPUT => zeros(2, 3)),
        global_consts_ty=global_types,
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(ParameterIntroductionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    @test mutated.root.name != :INPUT
    value = local_consts[mutated.root.name]
    @test value isa Matrix{Float64}
    @test size(value) == (2, 3)
    @test any(!iszero, value)
    @test local_consts_ty[mutated.root.name] == tensor_ty
    @test tree.root == ConstNode(:INPUT)
end

@testitem "Mutations PointMutation Index swap respects binder types" setup=[MutBaseImports] begin
    rng = MersenneTwister(1111)
    # λx. λy. #0  (two Int binders so swap is allowed)
    tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(0))))
    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[PointMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(PointMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=20,
    )
    @test mutated isa AstTree
    @test mutated.root.body.body isa IndexNode
    @test mutated.root.body.body.index in (0, 1)
end

@testitem "Mutations SubtreeReplacementMutation uses local binder scope" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationWeights
    using LambdaRegression.Mutations: SubtreeReplacementMutation, PathSiteSelectionPolicy

    rng = MersenneTwister(1234)
    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    target_ty = TyFun(FloatTy, FloatTy)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)

    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[
            SubtreeReplacementMutation(;
                site_selection=PathSiteSelectionPolicy(Int[1]),
                generation_kwargs=(;
                    weights=GenerationWeights(default=0, variable=1),
                    sampling_depth_limit=1,
                    sampling_node_budget=4,
                ),
            ),
        ],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=5,
        prune_consts=true,
    )

    @test mutated isa AstTree
    @test mutated.root isa AbsNode
    @test mutated.root.body == IndexNode(0)
    @test iscompatible(infer(mutated, merge(Dict{Symbol,TyLike}(), local_consts_ty)), target_ty)
end

@testitem "Mutations SubtreeReplacementMutation exposes replaceable generator hook" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations: SubtreeReplacementMutation, PathSiteSelectionPolicy
    using LambdaRegression.Mutations: AbstractReplacementGenerator, generate_replacement

    struct FixedFloatReplacement <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::FixedFloatReplacement,
        site::LambdaRegression.Mutations.Site,
        env::LambdaRegression.Mutations.MutationEnv,
        ctx::NamedTuple,
        ::NamedTuple,
    )
        ctx.local_consts[:replacement] = 2.0
        ctx.local_consts_ty[:replacement] = site.ty
        return GenerationResult(AstTree(ConstNode(:replacement)), site.ty, ctx.local_consts, ctx.local_consts_ty, Dict{Symbol,Any}(:source => :test_fixed))
    end

    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    target_ty = TyFun(FloatTy, FloatTy)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)

    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(42),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[
            SubtreeReplacementMutation(;
                site_selection=PathSiteSelectionPolicy(Int[1]),
                replacement_generator=FixedFloatReplacement(),
            ),
        ],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=5,
        prune_consts=true,
    )

    @test mutated isa AstTree
    @test mutated.root.body == ConstNode(:replacement)
    @test local_consts == Dict{Symbol,Any}(:replacement => 2.0)
    @test local_consts_ty == Dict{Symbol,TyLike}(:replacement => FloatTy)
end

@testitem "Mutations SubtreeReplacementMutation freshens conflicting replacement constants" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations: SubtreeReplacementMutation, PathSiteSelectionPolicy, DefaultSiteLockPolicy
    using LambdaRegression.Mutations: AbstractReplacementGenerator
    using LambdaRegression.TypeSystem: FADD

    struct ConflictingKReplacement <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::ConflictingKReplacement,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(ConstNode(:k)),
            site.ty,
            Dict{Symbol,Any}(:k => 2.0),
            Dict{Symbol,TyLike}(:k => FloatTy),
            Dict{Symbol,Any}(:source => :conflict_fixture),
        )
    end

    tree = AstTree(AppNode(AppNode(ConstNode(:FADD), ConstNode(:k)), ConstNode(:old)))
    local_consts = Dict{Symbol,Any}(:k => 1.0, :old => 3.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy, :old => FloatTy)
    global_consts = Dict{Symbol,Any}(:FADD => FADD)
    global_consts_ty = Dict{Symbol,TyLike}(:FADD => TypeSystem.get_type(FADD))

    mutated = mutate_tree(
        tree,
        FloatTy;
        rng=MersenneTwister(2000),
        global_consts,
        global_consts_ty,
        local_consts,
        local_consts_ty,
        mutations=[
            SubtreeReplacementMutation(;
                site_lock=DefaultSiteLockPolicy(spine_only=false),
                site_selection=PathSiteSelectionPolicy(Int[2]),
                replacement_generator=ConflictingKReplacement(),
            ),
        ],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=5,
        prune_consts=true,
    )

    @test mutated isa AstTree
    @test local_consts[:k] == 1.0
    @test mutated.root.arg isa ConstNode
    @test mutated.root.arg.name != :k
    @test local_consts[mutated.root.arg.name] == 2.0
    @test iscompatible(infer(mutated, merge(global_consts_ty, local_consts_ty)), FloatTy)
end

@testitem "Mutations SubtreeReplacementMutation default lock is conservative and replaceable" setup=[MutBaseImports] begin
    using LambdaRegression.Mutations: DefaultSiteLockPolicy, PathSiteSelectionPolicy
    using LambdaRegression.Mutations: is_site_locked, site_selection_weight

    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))))
    target_ty = TyFun(FloatTy, FloatTy)
    global_consts_ty = Dict{Symbol,TyLike}(:f => TyFun(FloatTy, FloatTy))
    env = LambdaRegression.Mutations.MutationEnv(
        MersenneTwister(42),
        target_ty,
        global_consts_ty,
        nothing,
        nothing,
        false,
        IdDict{Any,Any}(),
        LambdaRegression.Mutations.AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        LambdaRegression.Mutations.BloatPolicy(),
    )
    ctx = (;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        global_consts_ty=global_consts_ty,
        tyfactory=newtyvar,
    )
    provider = IndexedProvider()
    prep = LambdaRegression.Mutations.prepare!(provider, tree, global_consts_ty, env, ctx)
    @test !(prep isa Err)

    f_site = only(filter(s -> s.path == Int[1, 1], LambdaRegression.Mutations.sites(provider)))
    x_site = only(filter(s -> s.path == Int[1, 2], LambdaRegression.Mutations.sites(provider)))
    body_site = only(filter(s -> s.path == Int[1], LambdaRegression.Mutations.sites(provider)))

    @test is_site_locked(DefaultSiteLockPolicy(), f_site, env, ctx)
    @test is_site_locked(DefaultSiteLockPolicy(), x_site, env, ctx)
    @test !is_site_locked(DefaultSiteLockPolicy(), body_site, env, ctx)
    @test site_selection_weight(PathSiteSelectionPolicy(Int[1]), body_site, env, ctx) == 1.0
    @test site_selection_weight(PathSiteSelectionPolicy(Int[1, 1]), body_site, env, ctx) == 0.0
end

@testitem "Mutations SubtreeReplacementMutation is deterministic under fixed seed" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationWeights
    using LambdaRegression.Mutations: SubtreeReplacementMutation, PathSiteSelectionPolicy

    function run_once()
        local_consts = Dict{Symbol,Any}(:k => 1.0)
        local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
        tree = AstTree(AbsNode(:x, ConstNode(:k)))
        out = mutate_tree(
            tree,
            TyFun(FloatTy, FloatTy);
            rng=MersenneTwister(555),
            local_consts=local_consts,
            local_consts_ty=local_consts_ty,
            mutations=[
                SubtreeReplacementMutation(;
                    site_selection=PathSiteSelectionPolicy(Int[1]),
                    generation_kwargs=(;
                        weights=GenerationWeights(default=0, variable=1),
                        sampling_depth_limit=1,
                        sampling_node_budget=4,
                    ),
                ),
            ],
            chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0)),
            provider=IndexedProvider(),
            max_try=5,
            prune_consts=true,
        )
        return out, local_consts, local_consts_ty
    end

    a = run_once()
    b = run_once()
    @test a[1] isa AstTree
    @test a[1] == b[1]
    @test a[2] == b[2]
    @test a[3] == b[3]
end

@testitem "Mutations SubtreeReplacementMutation supports deterministic two-round replacement" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations: SubtreeReplacementMutation
    using LambdaRegression.Mutations: AbstractReplacementGenerator

    mutable struct SequenceReplacement <: AbstractReplacementGenerator
        counter::Int
    end
    function LambdaRegression.Mutations.generate_replacement(
        generator::SequenceReplacement,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        generator.counter += 1
        name = Symbol(:replacement, generator.counter)
        return GenerationResult(
            AstTree(ConstNode(name)),
            site.ty,
            Dict{Symbol,Any}(name => Float64(generator.counter)),
            Dict{Symbol,TyLike}(name => FloatTy),
            Dict{Symbol,Any}(:round => generator.counter),
        )
    end

    function run_two_rounds()
        local_consts = Dict{Symbol,Any}(:seed => 0.0)
        local_consts_ty = Dict{Symbol,TyLike}(:seed => FloatTy)
        tree = AstTree(ConstNode(:seed))
        mutation = SubtreeReplacementMutation(; replacement_generator=SequenceReplacement(0))
        for _ in 1:2
            tree = mutate_tree(
                tree,
                FloatTy;
                rng=MersenneTwister(1),
                local_consts,
                local_consts_ty,
                mutations=[mutation],
                chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0)),
                provider=IndexedProvider(),
                max_try=5,
                prune_consts=true,
            )
            @test tree isa AstTree
        end
        return tree, local_consts, local_consts_ty
    end

    a = run_two_rounds()
    b = run_two_rounds()
    @test a[1] == b[1]
    @test a[2] == b[2]
    @test a[3] == b[3]
    @test a[1].root == ConstNode(:replacement2)
    @test a[2] == Dict{Symbol,Any}(:replacement2 => 2.0)
end

@testitem "Mutations SubtreeReplacementMutation smokes on rich canonical target families" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: generate_canonical, test_catalog_consts, test_catalog_generation_kwargs
    using LambdaRegression.Mutations: SubtreeReplacementMutation

    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    Float4Fn = TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, FloatTy))))
    targets = [
        TyFun(Matrix32, Matrix32),
        TyFun(Matrix32, FloatTy),
        TyFun(FloatTy, FloatTy),
        Float4Fn,
        TyFun(TypeSystem.TyList(FloatTy), FloatTy),
    ]
    consts = test_catalog_consts()
    generation_kwargs = test_catalog_generation_kwargs(profile=:deeper)
    replacement_kwargs = merge(
        generation_kwargs,
        (; sampling_depth_limit=2, sampling_node_budget=24, guarded_expansion_depth=0),
    )

    for (i, target_ty) in enumerate(targets)
        generated = generate_canonical(
            target_ty;
            rng=MersenneTwister(100 + i),
            global_consts=consts,
            generation_kwargs...,
        )
        @test generated isa LambdaRegression.CanonicalGeneration.GenerationResult

        local_consts = copy(generated.local_consts)
        local_consts_ty = copy(generated.local_const_types)
        mutated = mutate_tree(
            generated.tree,
            target_ty;
            rng=MersenneTwister(200 + i),
            global_consts=consts,
            local_consts=local_consts,
            local_consts_ty=local_consts_ty,
            mutations=[
                SubtreeReplacementMutation(; generation_kwargs=replacement_kwargs),
            ],
            chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplacementMutation=>1.0), cache_eligibility=true),
            provider=IndexedProvider(),
            max_try=80,
            prune_consts=true,
        )
        @test mutated isa AstTree
        inferred = infer(mutated, local_consts_ty)
        @test !(inferred isa Err)
        @test iscompatible(inferred, target_ty)
    end
end

@testitem "Mutations choose_pair eligibility_cache is not reset" setup=[MutBaseImports] begin
    rng = MersenneTwister(1)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    global_consts_ty = Dict{Symbol,TyLike}()
    type_ctx = merge(global_consts_ty, local_consts_ty)
    tree = AstTree(ConstNode(:k))

    env = LambdaRegression.Mutations.MutationEnv(
        rng,
        FloatTy,
        type_ctx,
        nothing,
        nothing,
        true,
        IdDict{Any,Any}(),
        LambdaRegression.Mutations.AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        LambdaRegression.Mutations.BloatPolicy(),
    )
    ctx = (; local_consts=local_consts, local_consts_ty=local_consts_ty, tyfactory=newtyvar)

    provider = IndexedProvider()
    LambdaRegression.Mutations.reset!(provider)
    prep = LambdaRegression.Mutations.prepare!(provider, tree, type_ctx, env, ctx)
    @test !(prep isa Err)

    chooser = WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation => 1.0), cache_eligibility=true)
    cache0 = env.eligibility_cache
    choice1 = LambdaRegression.Mutations.choose_pair(chooser, [ScalarConstMutation(σ=0.2)], provider, tree, type_ctx, env, ctx)
    @test !(choice1 isa Err)
    @test env.eligibility_cache === cache0

    choice2 = LambdaRegression.Mutations.choose_pair(chooser, [ScalarConstMutation(σ=0.2)], provider, tree, type_ctx, env, ctx)
    @test !(choice2 isa Err)
    @test env.eligibility_cache === cache0
end

@testitem "Mutations _remap_site preserves path semantics" setup=[MutBaseImports] begin
    # (λx. (k x)) : Float -> Float
    local_consts = Dict{Symbol,Any}(:k => (x -> x))
    local_consts_ty = Dict{Symbol,TyLike}(:k => TyFun(FloatTy, FloatTy))
    global_consts_ty = Dict{Symbol,TyLike}()
    type_ctx = merge(global_consts_ty, local_consts_ty)
    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:k), IndexNode(0))))

    env = LambdaRegression.Mutations.MutationEnv(
        MersenneTwister(1111),
        TyFun(FloatTy, FloatTy),
        type_ctx,
        nothing,
        nothing,
        false,
        IdDict{Any,Any}(),
        LambdaRegression.Mutations.AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        LambdaRegression.Mutations.BloatPolicy(),
    )
    ctx = (; local_consts=local_consts, local_consts_ty=local_consts_ty, tyfactory=newtyvar)

    provider = IndexedProvider()
    LambdaRegression.Mutations.reset!(provider)
    prep = LambdaRegression.Mutations.prepare!(provider, tree, type_ctx, env, ctx)
    @test !(prep isa LambdaRegression.Utils.Err)

    sites = LambdaRegression.Mutations.sites(provider)
    s0 = first(filter(s -> s.node isa IndexNode, sites))
    attempt_tree = deepcopy(tree)
    remapped = LambdaRegression.Mutations._remap_site(attempt_tree, s0)
    expected = LambdaRegression.Mutations._node_at_path(attempt_tree, s0.path)
    @test remapped.node == expected
end

@testitem "Mutations prepare! runs once per mutate_tree call" setup=[MutBaseImports] begin
    struct AlwaysFailMutation <: LambdaRegression.Mutations.AbstractMutation end
    LambdaRegression.Mutations.apply(::AlwaysFailMutation, ::AstTree, ::LambdaRegression.Mutations.Site, ::Any, ::NamedTuple) =
        Err(LambdaRegression.Mutations.NoEligibleSiteErr())

    mutable struct CountingProvider <: LambdaRegression.Mutations.AbstractSiteProvider
        inner::LambdaRegression.Mutations.IndexedProvider
        prepare_calls::Int
    end
    CountingProvider() = CountingProvider(LambdaRegression.Mutations.IndexedProvider(), 0)
    LambdaRegression.Mutations.reset!(p::CountingProvider) = (p.prepare_calls = 0; LambdaRegression.Mutations.reset!(p.inner))
    function LambdaRegression.Mutations.prepare!(p::CountingProvider, tree::AstTree, type_ctx::Dict{Symbol,<:TyLike}, env::Any, ctx::NamedTuple)
        p.prepare_calls += 1
        return LambdaRegression.Mutations.prepare!(p.inner, tree, type_ctx, env, ctx)
    end
    LambdaRegression.Mutations.sites(p::CountingProvider) = LambdaRegression.Mutations.sites(p.inner)
    LambdaRegression.Mutations.sites_by_kind(p::CountingProvider, kind::Symbol) = LambdaRegression.Mutations.sites_by_kind(p.inner, kind)

    rng = MersenneTwister(1234)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    provider = CountingProvider()

    out = mutate_tree(
        AstTree(ConstNode(:k)),
        FloatTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[AlwaysFailMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(AlwaysFailMutation => 1.0), cache_eligibility=true),
        provider=provider,
        max_try=5,
        prune_consts=false,
    )

    @test out isa Err
    @test out.value isa LambdaRegression.Mutations.MaxTryReachedErr
    @test provider.prepare_calls == 1
end

@testitem "Mutations BetaReduceMutation reduces redex + preserves type" setup=[MutBaseImports] begin
    rng = MersenneTwister(8000)
    local_consts = Dict{Symbol,Any}(:k => 7)
    local_consts_ty = Dict{Symbol,TyLike}(:k => IntTy)
    tree = AstTree(AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:k))) # (λx. #0) k
    mutated = mutate_tree(
        tree,
        IntTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[BetaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(BetaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa AstTree
end

@testitem "Mutations BetaReduceMutation reduces non-root redex under binder" setup=[MutBaseImports] begin
    rng = MersenneTwister(1234)
    global_consts_ty = Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy))
    tree = AstTree(AbsNode(:z, AppNode(ConstNode(:f), AppNode(AbsNode(:x, IndexNode(0)), IndexNode(0)))))
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng,
        global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[BetaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(BetaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )

    expected = AstTree(AbsNode(:z, AppNode(ConstNode(:f), IndexNode(0))))
    @test mutated == expected
    @test iscompatible(infer(mutated, global_consts_ty), target_ty)
end

@testitem "Mutations BetaReduceMutation preserves outer binder through nested body" setup=[MutBaseImports] begin
    rng = MersenneTwister(8000)
    tree = AstTree(AbsNode(:z, AppNode(AbsNode(:x, AbsNode(:y, IndexNode(1))), IndexNode(0))))
    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    mutated = mutate_tree(
        tree,
        target_ty;
        rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[BetaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(BetaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )

    expected = AstTree(AbsNode(:z, AbsNode(:y, IndexNode(1))))
    @test mutated == expected
    @test iscompatible(infer(mutated, Dict{Symbol,TyLike}()), target_ty)
end

@testitem "Mutations EtaReduceMutation reduces eta-redex + preserves type" setup=[MutBaseImports] begin
    rng = MersenneTwister(42)
    global_consts_ty = Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy))
    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0)))) # λx. f #0
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[EtaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(EtaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa AstTree
end

@testitem "Mutations EtaReduceMutation reduces non-root eta-redex" setup=[MutBaseImports] begin
    rng = MersenneTwister(555)
    global_consts_ty = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :h => TyFun(TyFun(IntTy, IntTy), IntTy),
    )
    tree = AstTree(AbsNode(:z, AppNode(ConstNode(:h), AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))))))
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng,
        global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[EtaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(EtaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )

    expected = AstTree(AbsNode(:z, AppNode(ConstNode(:h), ConstNode(:f))))
    @test mutated == expected
    @test iscompatible(infer(mutated, global_consts_ty), target_ty)
end

@testitem "Mutations EtaReduceMutation shifts outer binder at non-root site" setup=[MutBaseImports] begin
    rng = MersenneTwister(2000)
    global_consts_ty = Dict{Symbol,TyLike}(:h => TyFun(TyFun(IntTy, IntTy), IntTy))
    tree = AstTree(AbsNode(:z, AppNode(ConstNode(:h), AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))))))
    target_ty = TyFun(TyFun(IntTy, IntTy), IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng,
        global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[EtaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(EtaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )

    expected = AstTree(AbsNode(:z, AppNode(ConstNode(:h), IndexNode(0))))
    @test mutated == expected
    @test iscompatible(infer(mutated, global_consts_ty), target_ty)
end

@testitem "Mutations EtaReduceMutation rejects when binder occurs in f" setup=[MutBaseImports] begin
    rng = MersenneTwister(555)
    global_consts_ty = Dict{Symbol,TyLike}(:g => TyFun(IntTy, TyFun(IntTy, IntTy)))
    # λx. (g #0) #0  -- would match shape but binder occurs in f=(g #0)
    tree = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:g), IndexNode(0)), IndexNode(0))))
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[EtaReduceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(EtaReduceMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa Err
end

@testitem "Mutations EtaExpandMutation expands function constant + preserves type" setup=[MutBaseImports] begin
    rng = MersenneTwister(1111)
    global_consts_ty = Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy))
    tree = AstTree(ConstNode(:f))
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[EtaExpandMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(EtaExpandMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa AstTree
    abs = (mutated::AstTree).root
    @test abs isa AbsNode
    app = (abs::AbsNode).body
    @test app isa AppNode
    @test (app::AppNode).func == ConstNode(:f)
    @test (app::AppNode).arg == IndexNode(0)
    inferred = infer(mutated, global_consts_ty)
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations EtaExpandMutation expands selected non-root function site" setup=[MutBaseImports] begin
    rng = MersenneTwister(8000)
    global_consts_ty = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :h => TyFun(TyFun(IntTy, IntTy), IntTy),
    )
    tree = AstTree(AbsNode(:z, AppNode(ConstNode(:h), ConstNode(:f))))
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = global_consts_ty
    env = LambdaRegression.Mutations.MutationEnv(
        rng,
        target_ty,
        type_ctx,
        tree,
        nothing,
        false,
        IdDict{Any,Any}(),
        LambdaRegression.Mutations.AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        LambdaRegression.Mutations.BloatPolicy(),
    )
    ctx = (;
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        global_consts_ty,
        tyfactory=newtyvar,
        symfactory=LambdaRegression.ASTs.NodeASTs.SymbolGenerator(),
        const_symfactory=LambdaRegression.ASTs.Contexts.ConstSymbolFactory(),
    )
    provider = IndexedProvider()
    prep = LambdaRegression.Mutations.prepare!(provider, tree, type_ctx, env, ctx)
    @test prep === nothing
    site = only([s for s in LambdaRegression.Mutations.sites(provider) if s.path == Int[1, 2]])

    result = LambdaRegression.Mutations.apply(EtaExpandMutation(), tree, site, env, ctx)

    expected = AstTree(AbsNode(:z, AppNode(ConstNode(:h), AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))))))
    @test result isa AstTree
    @test tree == expected
    @test iscompatible(infer(tree, global_consts_ty), target_ty)
end

@testitem "Mutations CondAppPullMutation pulls application through COND + preserves type" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.Contexts: COND
    using LambdaRegression.Mutations: CondAppPullMutation
    using LambdaRegression.TypeSystem: get_type_ctx

    rng = MersenneTwister(2000)
    global_consts_ty = merge(
        get_type_ctx(Dict{Symbol,Any}(:COND => COND, :b => true, :x0 => 1.0)),
        Dict{Symbol,TyLike}(
            :f => TyFun(FloatTy, FloatTy),
            :g => TyFun(FloatTy, FloatTy),
        ),
    )

    # COND b (f x0) (g x0)
    tree = AstTree(
        AppNode(
            AppNode(
                AppNode(ConstNode(:COND), ConstNode(:b)),
                AppNode(ConstNode(:f), ConstNode(:x0)),
            ),
            AppNode(ConstNode(:g), ConstNode(:x0)),
        ),
    )
    target_ty = FloatTy
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[CondAppPullMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(CondAppPullMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa AstTree

    expected = AstTree(
        AppNode(
            AppNode(
                AppNode(
                    AppNode(ConstNode(:COND), ConstNode(:b)),
                    ConstNode(:f),
                ),
                ConstNode(:g),
            ),
            ConstNode(:x0),
        ),
    )
    @test (mutated::AstTree).root == expected.root
    inferred = infer(mutated, global_consts_ty)
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations CSEMutation duplicates extracted preserves type" setup=[MutBaseImports] begin
    rng = MersenneTwister(2000)
    global_consts_ty = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    fx = AppNode(ConstNode(:f), IndexNode(0))
    tree = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), fx), deepcopy(fx))))
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[CSEMutation(min_count=2)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(CSEMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=20,
        prune_consts=false,
    )
    @test mutated isa AstTree
    @test mutated != tree
    inferred = infer(mutated, global_consts_ty)
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations CSEMutation single lift can extract non-application site" setup=[MutBaseImports] begin
    rng = MersenneTwister(1)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    target_ty = TyFun(FloatTy, FloatTy)

    mutated = mutate_tree(
        tree,
        target_ty;
        rng,
        local_consts,
        local_consts_ty,
        mutations=[CSEMutation(min_count=1)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(CSEMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=false,
    )

    @test mutated isa AstTree
    @test mutated != tree
    inferred = infer(mutated, local_consts_ty)
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations integration: matrix->float type stability" setup=[MutBaseImports] begin
    rng = MersenneTwister(42)
    dom = TyArray(TyNat(3), TyArray(TyNat(4), FloatTy))
    target_ty = TyFun(dom, FloatTy)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[ScalarConstMutation(σ=0.3)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
        copy_ctx=true,
    )
    inferred = infer(mutated, merge(Dict{Symbol,TyLike}(), local_consts_ty))
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations uses const_symfactory (no gensym-style names) in SR path" setup=[MutBaseImports] begin
    rng = MersenneTwister(1111)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    tree = AstTree(ConstNode(:k))

    mutated = mutate_tree(
        tree,
        FloatTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[ScalarConstMutation(σ=0.2)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation=>1.0)),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
        const_symfactory=ConstSymbolFactory(prefix=:c),
    )

    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    s = String((mutated.root::ConstNode).name)
    @test startswith(s, "c")
    @test !occursin("#", s)
end

@testitem "Mutations mutate_tree recorder counts apply_err + max_try_exhausted" setup=[MutBaseImports] begin
    # Intentionally inconsistent local const: value is Bool but typed as FloatTy.
    # This makes BoolConstMutation eligible, but its value-level mutation returns UnsupportedConstTyErr(FloatTy),
    # forcing apply() to fail deterministically.
    rng = MersenneTwister(1)
    name = :b
    local_consts = Dict{Symbol,Any}(name => true)
    local_consts_ty = Dict{Symbol,TyLike}(name => FloatTy)
    tree = AstTree(ConstNode(name))

    rec = MutationAttemptRecorder(record_error_types=true)
    mutated = mutate_tree(
        tree,
        FloatTy;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[BoolConstMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(BoolConstMutation => 1.0), cache_eligibility=false),
        provider=IndexedProvider(),
        max_try=7,
        prune_consts=true,
        recorder=rec,
    )

    @test mutated isa LambdaRegression.Utils.Err
    s = mutation_recorder_summary(rec; topk=4)
    @test s.totals.success == 0
    @test s.totals.apply_err == 7
    @test s.totals.max_try_exhausted == 1
    @test length(s.per_mutation) >= 1
    @test s.per_mutation[1].mutation_type == BoolConstMutation
    @test s.per_mutation[1].counts.apply_err == 7
    @test s.error_types == [(
        error_type=LambdaRegression.Mutations.UnsupportedConstTyErr,
        count=7,
    )]
end

@testitem "Mutations ContextualExpansionMutation composes around an existing subtree" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations: ContextualExpansionMutation, PathSiteSelectionPolicy
    using LambdaRegression.Mutations: AbstractReplacementGenerator, generate_replacement
    using LambdaRegression.TypeSystem: FADD, get_type_ctx

    struct AddSeedContext <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::AddSeedContext,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ctx::NamedTuple,
        ::NamedTuple,
    )
        @test site.binder_types == [FloatTy, FloatTy]
        body = AppNode(
            AppNode(ConstNode(:FADD), IndexNode(0)),
            IndexNode(1),
        )
        return GenerationResult(
            AstTree(body),
            site.ty,
            copy(ctx.local_consts),
            copy(ctx.local_consts_ty),
            Dict{Symbol,Any}(),
        )
    end

    tree = AstTree(AbsNode(:x, IndexNode(0)))
    target_ty = TyFun(FloatTy, FloatTy)
    global_consts = Dict{Symbol,Any}(:FADD => FADD)
    global_consts_ty = get_type_ctx(global_consts)
    mutation = ContextualExpansionMutation(
        site_selection=PathSiteSelectionPolicy(Int[1]),
        replacement_generator=AddSeedContext(),
    )
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(555),
        global_consts,
        global_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ContextualExpansionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated isa AstTree
    @test length(mutated) > length(tree)
    @test mutated.root.body == AppNode(
        AbsNode(
            :seed,
            AppNode(
                AppNode(ConstNode(:FADD), IndexNode(0)),
                IndexNode(1),
            ),
        ),
        IndexNode(0),
    )
    @test iscompatible(infer(mutated, global_consts_ty), target_ty)
end

@testitem "Mutations UnaryCompositionMutation wraps a typed endomorphism" setup=[MutBaseImports] begin
    using LambdaRegression.Mutations: PathSiteSelectionPolicy, UnaryCompositionMutation
    using LambdaRegression.TypeSystem: SQR, get_type_ctx

    tree = AstTree(AbsNode(:x, IndexNode(0)))
    target_ty = TyFun(FloatTy, FloatTy)
    globals = Dict{Symbol,Any}(:SQR => SQR)
    mutation = UnaryCompositionMutation(
        site_selection=PathSiteSelectionPolicy(Int[1]),
    )
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(2000),
        global_consts=globals,
        global_consts_ty=get_type_ctx(globals),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(UnaryCompositionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated isa AstTree
    @test mutated.root.body == AppNode(ConstNode(:SQR), IndexNode(0))
    @test iscompatible(infer(mutated, get_type_ctx(globals)), target_ty)
end

@testitem "Mutations BinaryCompositionMutation preserves a typed seed branch" setup=[MutBaseImports] begin
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        BinaryCompositionMutation,
        PathSiteSelectionPolicy
    using LambdaRegression.TypeSystem: FADD, SQR, get_type_ctx

    struct OneSibling <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::OneSibling,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        @test site.ty === FloatTy
        return GenerationResult(
            AstTree(ConstNode(:ONE)),
            FloatTy,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:SQR), IndexNode(0))))
    target_ty = TyFun(FloatTy, FloatTy)
    global_consts = Dict{Symbol,Any}(:FADD => FADD, :SQR => SQR, :ONE => 1.0)
    global_consts_ty = get_type_ctx(global_consts)
    mutation = BinaryCompositionMutation(
        site_selection=PathSiteSelectionPolicy(Int[1]),
        replacement_generator=OneSibling(),
    )
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1111),
        global_consts,
        global_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(BinaryCompositionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated isa AstTree
    names = Symbol[node.name for node in mutated if node isa ConstNode]
    @test :FADD in names
    @test :SQR in names
    @test :ONE in names
    @test length(mutated) == length(tree) + 4
    @test iscompatible(infer(mutated, global_consts_ty), target_ty)
end

@testitem "Mutations target-grounded sites compose inside FunctionDef" setup=[MutBaseImports] begin
    using LambdaRegression: CanonicalGeneration, Procedural
    using LambdaRegression.Mutations:
        BinaryCompositionMutation,
        PathSiteSelectionPolicy,
        ScopedReplacementGenerator
    using LambdaRegression.TypeSystem:
        FADD,
        UNCONS_CASE,
        TyList,
        get_type_ctx

    on_item = AbsNode(
        :head,
        AbsNode(:tail, AppNode(IndexNode(3), IndexNode(0))),
    )
    fallback = AppNode(
        AppNode(ConstNode(:FADD), ConstNode(:ONE)),
        ConstNode(:ONE),
    )
    tree = AstTree(Procedural.FunctionDefNode(AbsNode(
        :series,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                on_item,
            ),
            fallback,
        ),
    )))
    target_ty = TyFun(TyList(FloatTy), FloatTy)
    global_consts = Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :FADD => FADD,
        :ONE => 1.0,
        :ZERO => 0.0,
    )
    global_consts_ty = get_type_ctx(global_consts)
    mutation = BinaryCompositionMutation(
        site_selection=PathSiteSelectionPolicy(Int[1, 1, 1, 2, 1, 1]),
        replacement_generator=ScopedReplacementGenerator(
            Procedural.generate_procedural_scoped;
            include_local_consts=true,
        ),
        generation_kwargs=(;
            providers=(
                CanonicalGeneration.default_head_providers()...,
                CanonicalGeneration.SafeListAccessProvider(),
            ),
            sampling_depth_limit=3,
            sampling_node_budget=24,
            guarded_expansion_depth=1,
        ),
    )
    checker = (candidate, ctx, factory, meta) ->
        Procedural.infer_procedural(candidate, ctx; factory, meta)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1),
        global_consts,
        global_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(BinaryCompositionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=5,
        type_inference=checker,
    )

    @test mutated isa AstTree
    inferred = Procedural.infer_procedural(mutated, global_consts_ty)
    @test !(inferred isa Err)
    @test iscompatible(inferred, target_ty)
    @test count(
        node -> node isa ConstNode && node.name === :FADD,
        mutated,
    ) >= 2
end

_default_mutations() = default_mutations()

@testitem "Mutations bloat policy rejects growth transactionally" setup=[MutBaseImports] begin
    struct DeterministicGrowMutation <: LambdaRegression.Mutations.AbstractMutation end
    function LambdaRegression.Mutations.apply(
        ::DeterministicGrowMutation,
        tree::AstTree,
        ::LambdaRegression.Mutations.Site,
        ::Any,
        ::NamedTuple,
    )
        tree.root = AppNode(AbsNode(:ignored, tree.root), ConstNode(:k))
        return tree
    end

    parent = AstTree(ConstNode(:k))
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    original_consts = copy(local_consts)
    original_types = copy(local_consts_ty)
    recorder = MutationAttemptRecorder()

    result = mutate_tree(
        parent,
        FloatTy;
        rng=MersenneTwister(2000),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        mutations=[DeterministicGrowMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(DeterministicGrowMutation => 1.0)),
        provider=IndexedProvider(),
        max_try=2,
        prune_consts=false,
        bloat_policy=LambdaRegression.Mutations.BloatPolicy(enabled=true, max_nodes=1, max_nodes_delta=0),
        recorder=recorder,
    )

    @test result isa Err
    @test parent == AstTree(ConstNode(:k))
    @test local_consts == original_consts
    @test local_consts_ty == original_types
    summary = mutation_recorder_summary(recorder)
    @test summary.totals.bloat_exceeded == 2
    @test summary.totals.max_try_exhausted == 1
end

@testitem "Mutations policies reject invalid numeric configuration" setup=[MutBaseImports] begin
    @test_throws ArgumentError WeightedPairChooser(
        weights=Dict{DataType,Float64}(ScalarConstMutation => NaN),
    )
    @test_throws ArgumentError LambdaRegression.Mutations.BloatPolicy(max_nodes=0)
    @test_throws ArgumentError LambdaRegression.Mutations.BloatPolicy(max_nodes_delta=-1)
    @test_throws ArgumentError LambdaRegression.Mutations.BloatPolicy(true, 0, 0)
    @test_throws ArgumentError LambdaRegression.Mutations.AdaptivePolicy(ewma_alpha=0.0)
    @test_throws ArgumentError LambdaRegression.Mutations.AdaptivePolicy(floor=Inf)
    @test_throws ArgumentError LambdaRegression.Mutations.AdaptivePolicy(true, 2.0, 0.1)
    @test_throws ArgumentError LambdaRegression.Mutations.WeightedPairChooser(
        Dict{Type,Float64}(ScalarConstMutation => Inf),
        false,
    )
end

end # module Mutations
