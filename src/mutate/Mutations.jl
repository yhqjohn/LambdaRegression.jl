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
   - `IndexedProvider` derives occurrence-backed sites from one root-constrained
     `TypedAnalysis` for the common syntax, with paths and binder slots on demand.
   - Binding insertion and Graft consume that same analysis for the selected
     occurrence and recorded let-like schemes; scoped trees remain rewrite-only.
   - FunctionDef mutations consume that same analysis for exact binder schemes.

2. **Pair chooser** (`AbstractPairChooser`):
   - Selects a mutation operator and a compatible site.
   - `WeightedPairChooser` implements a practical baseline: prefer high-success
     operators (constants) and only select from eligible sites.
   - Optional eligibility caching is scoped to *one* `mutate_tree` call.
   - Provider preparation and chooser/eligibility/weight hooks borrow the
     source tree and caller-owned constant dictionaries as deeply read-only
     inputs; hooks may inspect identity but must not mutate them or reachable
     values. Only `apply` receives a private candidate tree and copied local
     dictionaries; shared global constants remain read-only there.

3. **Mutation operator** (`AbstractMutation`):
   - Declares which kinds of sites it needs (`required_site_kinds`),
     filters sites (`is_site_eligible`), and performs the mutation (`apply`).

The top-level entry `mutate_tree` analyzes and selects against borrowed read-only
inputs, then for each eligible choice copies a private candidate immediately
before `apply`. It validates the type invariant (`infer` + `iscompatible`),
optionally prunes unreferenced local constants (`prune_local_consts!`), and
commits candidate dictionaries only after a successful validation.

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

Provider and chooser hooks borrow the source tree and caller-owned constant
dictionaries through analysis and selection. They may inspect object identity,
but must not mutate those objects or anything reachable from them. `apply`
receives one deep copy of the selected tree and shallow copies of both local
dictionaries; shared global constants remain read-only. Failed attempts discard
their copies, and validated success commits the copied local dictionaries to the
caller. This keeps selection identity cheap without exposing shared constants to
writes.

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
  context rather than preferred source heads. Its neutral-by-default
  `root_constant_weight` changes only direct constants at the complete tree
  root or the output body below a leading `AbsNode` chain; ordinary subtree
  replacements retain their supplied generation weights. A non-neutral value
  cannot be combined with `generation_kwargs.config`, because that config owns
  and overrides `weights`. Implement
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

## Direct composition

`UnaryCompositionMutation()` wraps a selected `S : T` with a monomorphic global
endomorphism `H : T → T`. It is the low-cost direct move for forms such as
`SQR(x)` or `TSCALE(head)`. `BinaryCompositionMutation()` preserves `S` as one
argument of a compatible curried binary global and generates the other sibling.
Their global-head directory is prepared once per `mutate_tree` call, and each
admitted closed target type gets a reusable ordered option menu. Open global
head types remain in the compatibility fallback. Use their site-selection
policies and `WeightedPairChooser` weights to change routing without changing
either rewrite.

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
`AppNode(AbsNode(...), rhs)`; let-like bindings use only the common AST.
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
final whole-tree Algorithm W remains authoritative. An operator
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
stale object identity cannot redirect them. In the default prepared native
evaluator, source-lambda arguments are call-by-need: an unused binding does not
force its RHS. Host calls are strict; an alternative evaluator may use a
different evaluation policy. These operators preserve typing and lexical scope,
not arbitrary effectful-host behavior or resource costs. A subsequently used
RHS may still diverge, time out, or raise a host error.

`GeneratedBindingInsertionMutation(rhs_types; ...)` is also opt-in. `rhs_types`
must be an explicit finite collection of closed monotypes and/or fully quantified
`TyScheme`s. The operator samples one demand with `rhs_type_weight(rhs_ty, host,
env, ctx)`, asks `replacement_generator` for an RHS, independently checks that
RHS against the selected demand, and only then commits the new let-like binding
and staged local constants. Captured host binders remain rigid during this check;
quantified variables in the selected scheme remain genuinely polymorphic. RHS
generalization follows `FTV(rhs) ∖ FTV(environment)`, and enclosing binder types
are reconstructed from the current tree and authoritative path rather than
trusted from caller-supplied `Site` metadata. Its new binder name is made fresh
against named variables and abstractions in the attempt tree.

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
using LambdaRegression.TypeSystem: TyLike, get_type_ctx
using LambdaRegression.Primitives: FST, PAIR, SND

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
validation. Generated binder names avoid all named variables and abstractions
in the attempt tree. Packing/unpacking conservatively reject named references
to the let binders they remove; De Bruijn references remain supported.
"""
module Mutations

using Random
using TestItems

using ..Utils: Err
using ..TypeSystem
using ..Primitives
using ..TypeSystem: Ty, TyLike, TyConst, TyFun, TyList, TyVar, TyArray, TyNat,
    instantiate, infer, iscompatible, _grounded_iscompatible, newtyvar, BoolTy, IntTy, FloatTy,
    _detached_type_ctx
using ..ASTs.NodeASTs
using ..ProgramAnalysis:
    TypedAnalysis, SyntaxIndex, analyze_program, analyze_syntax, path,
    UnsupportedSyntaxError,
    visible_binder_ids
using ..Transforms: eliminate_node!, occurrence_id, repeated_nodes
using ..CanonicalGeneration:
    generate_canonical_scoped, GenerationResult, GenerationWeights
using ..Programs: Program, _publish_program
import ..ASTs.NodeASTs: apply_node!, shift_free_variables!
using ..Programs: new_constant!, ConstSymbolFactory
using ..Trees: traverse, Break, getstate, setroot!, setchild!, children

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
export FunctionDefMutation, FunctionDefScope, generate_functiondef_replacement
export DefaultFunctionDefMutationGenerator
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
- `occurrence`: stable occurrence ID in the shared syntax index (zero for legacy/manual sites)
- `path`: stable 1-based child-index path from root (`Int[]` means root), recovered on demand
- `node`: AST node at this occurrence
- `parent`: parent AST node (or `nothing` for root)
- `child_index`: 1-based child index in `parent`
- `depth`: number of enclosing λ/FunctionDef binders
- `kind`: coarse kind for eligibility routing:
  - `:const_local`, `:const_global`, `:index`, `:app`, `:abs`, `:any`
- `ty`: inferred monotype for the node
- `binder_types`: complete visible binder schemes, ordered outer → inner
"""
mutable struct Site
    analysis::Union{Nothing,TypedAnalysis}
    occurrence::Int
    path_cache::Union{Nothing,Vector{Int}}
    node_cache::Union{Nothing,AstNode}
    parent_cache::Union{Nothing,AstNode}
    child_index_cache::Int
    depth_cache::Int
    kind::Symbol
    ty_cache::Union{Nothing,Ty}
    binder_types_cache::Union{Nothing,Vector{TyLike}}
end

function Site(
    path::Vector{Int},
    node::AstNode,
    parent::Union{Nothing,AstNode},
    child_index::Int,
    depth::Int,
    kind::Symbol,
    ty::Ty,
    binder_types::AbstractVector{<:TyLike},
    occurrence::Int=0,
)
    return Site(
        nothing,
        occurrence,
        path,
        node,
        parent,
        child_index,
        depth,
        kind,
        ty,
        TyLike[binder for binder in binder_types],
    )
end

function Site(analysis::TypedAnalysis, occurrence::Int, kind::Symbol)
    1 <= occurrence <= length(analysis) || throw(BoundsError(analysis.index.nodes, occurrence))
    return Site(analysis, occurrence, nothing, nothing, nothing, 0, -1, kind, nothing, nothing)
end

function Base.getproperty(site::Site, name::Symbol)
    analysis = getfield(site, :analysis)
    occurrence = getfield(site, :occurrence)
    if analysis === nothing
        name === :path && return getfield(site, :path_cache)
        name === :node && return getfield(site, :node_cache)
        name === :parent && return getfield(site, :parent_cache)
        name === :child_index && return getfield(site, :child_index_cache)
        name === :depth && return getfield(site, :depth_cache)
        name === :kind && return getfield(site, :kind)
        name === :ty && return getfield(site, :ty_cache)::Ty
        name === :binder_types && return getfield(site, :binder_types_cache)::Vector{TyLike}
    else
        index = analysis.index
        name === :path && begin
            cached = getfield(site, :path_cache)
            cached === nothing || return cached
            value = path(index, occurrence)
            setfield!(site, :path_cache, value)
            return value
        end
        name === :node && return index.nodes[occurrence]
        name === :parent && begin
            parent_id = index.parent[occurrence]
            return parent_id == 0 ? nothing : index.nodes[parent_id]
        end
        name === :child_index && return index.parent[occurrence] == 0 ? 1 : index.parent_edge[occurrence]
        name === :depth && begin
            cached = getfield(site, :depth_cache)
            cached >= 0 && return cached
            depth = 0
            binder = index.env_head[occurrence]
            while binder != 0
                depth += 1
                binder = index.bindings[binder].outer
            end
            setfield!(site, :depth_cache, depth)
            return depth
        end
        name === :kind && return getfield(site, :kind)
        name === :ty && begin
            cached = getfield(site, :ty_cache)
            cached === nothing || return cached
            raw = analysis.metadata[occurrence][1]
            raw isa Ty || throw(ArgumentError("mutation site $occurrence has non-monotype type metadata"))
            value = TypeSystem.prune(raw)
            setfield!(site, :ty_cache, value)
            return value
        end
        name === :binder_types && begin
            cached = getfield(site, :binder_types_cache)
            cached === nothing || return cached
            ids = reverse(visible_binder_ids(index, occurrence))
            value = TyLike[]
            for binder in ids
                haskey(analysis.binder_schemes, binder) || throw(ArgumentError(
                    "mutation analysis lacks a scheme for visible binder $binder",
                ))
                push!(value, analysis.binder_schemes[binder])
            end
            setfield!(site, :binder_types_cache, value)
            return value
        end
    end
    return getfield(site, name)
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

Site providers are responsible for supplying occurrence-backed `Site` records
and optionally indexing them by `kind` for fast access.

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
    analysis::Union{Nothing,TypedAnalysis}
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

_is_local_const(name::Symbol, ctx::NamedTuple) =
    haskey(ctx, :local_consts) && haskey(ctx, :local_consts_ty) &&
    haskey(ctx.local_consts, name) && haskey(ctx.local_consts_ty, name)
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

function _node_at_path(tree::AstTree, path::Vector{Int})::AstNode
    node = tree.root
    for idx in path
        node = children(node)[idx]
    end
    return node
end

function _analysis_occurrence_at_path(analysis::TypedAnalysis, requested_path)
    occurrence = 1
    for child_index in requested_path
        children_ids = analysis.index.children[occurrence]
        1 <= child_index <= length(children_ids) || return Err(NoEligibleSiteErr())
        occurrence = children_ids[child_index]
    end
    return occurrence
end

function _authoritative_site(
    analysis::TypedAnalysis,
    offered::Site,
    ctx::NamedTuple,
)
    offered_analysis = getfield(offered, :analysis)
    offered_analysis === nothing || offered_analysis === analysis ||
        return Err(NoEligibleSiteErr())

    offered_occurrence = getfield(offered, :occurrence)
    offered_occurrence >= 0 || return Err(NoEligibleSiteErr())
    if offered_analysis === analysis
        offered_occurrence > 0 || return Err(NoEligibleSiteErr())
        offered_occurrence <= length(analysis) || return Err(NoEligibleSiteErr())
        authoritative_path = path(analysis.index, offered_occurrence)
        cached_path = getfield(offered, :path_cache)
        cached_path === nothing || cached_path == authoritative_path ||
            return Err(NoEligibleSiteErr())
    else
        authoritative_path = getfield(offered, :path_cache)
        authoritative_path isa Vector{Int} || return Err(NoEligibleSiteErr())
        authoritative_path = copy(authoritative_path)
    end

    occurrence = _analysis_occurrence_at_path(analysis, authoritative_path)
    occurrence isa Err && return occurrence
    offered_occurrence == 0 || offered_occurrence == occurrence ||
        return Err(NoEligibleSiteErr())
    haskey(analysis.metadata, occurrence) || return Err(NoEligibleSiteErr())
    return Site(analysis, occurrence, _site_kind(analysis.index.nodes[occurrence], ctx))
end

function _remap_authoritative_site(tree::AstTree, site::Site)
    site_path = site.path
    node = _node_at_path(tree, site_path)
    if isempty(site_path)
        return Site(
            site_path, node, nothing, 1, site.depth, site.kind, site.ty,
            site.binder_types, site.occurrence,
        )
    end
    parent_path = site_path[1:end-1]
    parent = _node_at_path(tree, parent_path)
    return Site(
        site_path, node, parent, site_path[end], site.depth, site.kind, site.ty,
        site.binder_types, site.occurrence,
    )
end

function _remap_site(
    tree::AstTree,
    offered::Site,
    analysis::TypedAnalysis,
    ctx::NamedTuple,
)
    site = _authoritative_site(analysis, offered, ctx)
    site isa Err && return site
    return _remap_authoritative_site(tree, site)
end

function _collect_analysis_sites(analysis::TypedAnalysis, ctx::NamedTuple)
    sites = Site[]
    for occurrence in eachindex(analysis.index.nodes)
        haskey(analysis.metadata, occurrence) || continue
        node = analysis.index.nodes[occurrence]
        push!(sites, Site(analysis, occurrence, _site_kind(node, ctx)))
    end
    return sites
end

function _common_mutation_analysis(tree, type_ctx, factory, target_ty)
    return analyze_program(tree, type_ctx; factory, target_ty)
end

function _typed_mutation_analysis(env::MutationEnv, ctx::NamedTuple)
    analysis = get(ctx, :typed_analysis, nothing)
    if analysis isa TypedAnalysis
        env.analysis = analysis
        return analysis
    end
    env.analysis isa TypedAnalysis && return env.analysis
    env.tree === nothing && return Err(NoEligibleSiteErr())
    analysis = _common_mutation_analysis(
        env.tree,
        env.type_ctx,
        get(ctx, :tyfactory, newtyvar),
        env.target_ty,
    )
    analysis isa Err && return analysis
    env.analysis = analysis
    return analysis
end

"""
    IndexedProvider() <: AbstractSiteProvider

Indexed site provider for the common syntax.

Consumes one root-constrained `TypedAnalysis` and builds occurrence-backed
`sites_all` and `sites_by_kind` indexes.
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
    prepared = get(ctx, :typed_analysis, nothing)
    typed_ctx = type_ctx isa Dict{Symbol,TyLike} ? type_ctx : Dict{Symbol,TyLike}(type_ctx)
    if !(prepared isa TypedAnalysis) && (env.tree !== tree || env.type_ctx !== typed_ctx)
        # Cached facts belong to the prior source tree/context. Direct provider
        # callers need not pre-seed MutationEnv with this input.
        env.analysis = nothing
    end
    # A call-local root-constrained analysis, when supplied, remains the
    # authority; otherwise _typed_mutation_analysis builds facts for this input.
    env.tree = tree
    env.type_ctx = typed_ctx
    analysis = _typed_mutation_analysis(env, ctx)
    analysis isa Err && return analysis
    p.sites_all = _collect_analysis_sites(analysis, ctx)
    for site in p.sites_all
        push!(get!(p.sites_kind, :any, Site[]), site)
        site.kind === :any || push!(get!(p.sites_kind, site.kind, Site[]), site)
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
  one `mutate_tree` call.

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
    CanonicalReplacementGenerator(; include_local_consts=false,
                                  root_constant_weight=1.0)

Replacement generator adapter that calls `generate_canonical_scoped` with the
selected site's type and visible binder types. It requires the ordinary
mutation context fields `global_consts_ty`, `local_consts`, and
`local_consts_ty`; `mutate_tree` supplies these. `root_constant_weight` is
applied only at the complete root or a path prefix that stays on the leading
`AbsNode` body chain, so canonical generation can carry the weight through
remaining lambdas to the output root. Non-neutral weighting with an explicit
`generation_kwargs.config` raises `ArgumentError`, since that config owns and
overrides the per-call weights.
"""
struct CanonicalReplacementGenerator <: AbstractReplacementGenerator
    include_local_consts::Bool
    root_constant_weight::Float64
end

function _checked_root_constant_weight(value::Real)
    weight = Float64(value)
    isfinite(weight) && weight >= 0 || throw(ArgumentError(
        "root_constant_weight must be finite and nonnegative",
    ))
    return weight
end

function CanonicalReplacementGenerator(;
    include_local_consts::Bool=false,
    root_constant_weight::Real=1.0,
)
    weight = _checked_root_constant_weight(root_constant_weight)
    return CanonicalReplacementGenerator(include_local_consts, weight)
end

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

"""Whether `site` lies on the complete root's leading lambda chain.

The complete-root path and every prefix following body edges of leading
`AbsNode`s qualify. This includes intermediate lambda nodes whose bounded
canonical generation carries root weights into its output body. Any path that
crosses an application, RHS, callback, or `FunctionDefNode` is excluded even
when its length happens to match binder depth.
"""
function _is_complete_output_root(tree::Union{Nothing,AstTree}, site::Site)
    path = site.path
    isempty(path) && return true
    tree isa AstTree || return false
    node = tree.root
    for edge in path
        node isa AbsNode || return false
        edge == 1 || return false
        node = node.body
    end
    return true
end

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

function _site_binder_monotype(binder::TyLike)
    if binder isa Ty
        return TypeSystem.prune(binder)
    elseif isempty(binder.vars)
        return TypeSystem.prune(binder.ty)
    end
    return nothing
end

_site_scope_is_monomorphic(site::Site) =
    all(binder -> _site_binder_monotype(binder) !== nothing, site.binder_types)

function _site_scope_types(site::Site)
    _site_scope_is_monomorphic(site) || throw(ArgumentError(
        "generation does not support quantified visible binder schemes",
    ))
    return Ty[_site_binder_monotype(binder) for binder in reverse(site.binder_types)]
end

_infer_mutation_type(tree, type_ctx, factory) = infer(tree, type_ctx, factory)

function _replacement_generation_kwargs(
    include_local_consts::Bool,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    defaults = (;
        rng=env.rng,
        scope_tys=_site_scope_types(site),
        global_consts=_global_consts(ctx),
        global_consts_ty=ctx.global_consts_ty,
        local_consts=ctx.local_consts,
        local_consts_ty=ctx.local_consts_ty,
        source_local_consts=include_local_consts,
    )
    return merge(defaults, generation_kwargs)
end

function _canonical_replacement_kwargs(
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
    if generator.root_constant_weight != 1.0 &&
            _is_complete_output_root(env.tree, site)
        haskey(kwargs, :config) && throw(ArgumentError(
            "non-neutral root_constant_weight cannot be combined with generation_kwargs.config",
        ))
        weights = get(kwargs, :weights, GenerationWeights())
        weights isa GenerationWeights || throw(ArgumentError(
            "canonical replacement weights must be a GenerationWeights value",
        ))
        overrides = copy(weights.overrides)
        overrides[:root_constant] = generator.root_constant_weight
        kwargs = merge(kwargs, (; weights=GenerationWeights(
            default=weights.default,
            overrides=overrides,
        )))
    end
    return kwargs
end

function generate_replacement(
    generator::CanonicalReplacementGenerator,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    kwargs = _canonical_replacement_kwargs(generator, site, env, ctx, generation_kwargs)
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
    _site_scope_is_monomorphic(site) || return false
    _generation_closed_type(site.ty) || return false
    return site_selection_weight(m.site_selection, site, env, ctx) > 0
end

site_weight(m::SubtreeReplacementMutation, site::Site, env::MutationEnv, ctx::NamedTuple) =
    site_selection_weight(m.site_selection, site, env, ctx)

"""
    ParameterIntroductionMutation(; σ=1.0, site_lock, site_selection,
                                  root_constant_weight=1.0)

Introduce a fresh mutable real parameter at a typed expression site.

The operator anchors on the site's inferred type and supports `Float` plus
fixed-shape nested `TyArray[..., Float]`. It requires `ctx.local_consts`,
`ctx.local_consts_ty`, and `ctx.const_symfactory`; the ordinary mutation driver
provides these fields. Unsupported/open shapes have no eligible site. A chosen
site is replaced transactionally with a new local constant, so final inference
and bloat validation remain owned by `mutate_tree`. The default site policy
does not penalize lexical depth: numeric sites in an ordinary function body
must not become unreachable merely because they are enclosed by lambdas.
`root_constant_weight` multiplies the site weight only at the complete AST root
or a prefix of the leading `AbsNode` chain. This lets canonical generation
carry the root factor through intermediate lambda nodes; nested body sites do
not receive it.
"""
struct ParameterIntroductionMutation <: AbstractMutation
    σ::Float64
    site_lock::AbstractSiteLockPolicy
    site_selection::AbstractSiteSelectionPolicy
    root_constant_weight::Float64
end

function ParameterIntroductionMutation(;
    σ::Real=1.0,
    site_lock::AbstractSiteLockPolicy=DefaultSiteLockPolicy(spine_only=false),
    site_selection::AbstractSiteSelectionPolicy=ExponentialSiteSelectionPolicy(
        scope_bias=0.0,
    ),
    root_constant_weight::Real=1.0,
)
    scale = Float64(σ)
    isfinite(scale) && scale > 0 || throw(ArgumentError(
        "parameter introduction σ must be finite and positive",
    ))
    return ParameterIntroductionMutation(
        scale,
        site_lock,
        site_selection,
        _checked_root_constant_weight(root_constant_weight),
    )
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
    return site_weight(mutation, site, env, ctx) > 0
end

site_weight(
    mutation::ParameterIntroductionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
) = site_selection_weight(mutation.site_selection, site, env, ctx) *
    (_is_complete_output_root(env.tree, site) ? mutation.root_constant_weight : 1.0)

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
    _site_scope_is_monomorphic(site) || return Err(NoEligibleSiteErr())
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

The actual binder name is chosen fresh against named variables and abstractions
in the generated context, so an injected `VarNode` cannot be captured by the
context wrapper.

The generated context sees `seed` as De Bruijn index zero followed by the
binders already visible at the selected site. FunctionDef-aware generators remain
injectable, but contexts whose use of `seed` is hidden below a non-core binder
are conservatively rejected; final whole-tree Algorithm W remains authoritative.
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
    _site_scope_is_monomorphic(site) || return false
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
    _site_scope_is_monomorphic(site) || return Err(NoEligibleSiteErr())
    context_binders = Ty[_site_binder_monotype(binder) for binder in site.binder_types]
    push!(context_binders, site.ty)
    context_site = Site(
        site.path,
        site.node,
        site.parent,
        site.child_index,
        site.depth + 1,
        site.kind,
        site.ty,
        context_binders,
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

    seed_name = only(_fresh_binder_names(AstTree(context.root), () -> :seed, 1))
    expanded = AstTree(AppNode(
        AbsNode(seed_name, context.root),
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
this operator; final whole-tree Algorithm W still validates the complete tree.
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
    _grounded_iscompatible(left, right) && _grounded_iscompatible(right, left)

struct _CompositionHead
    name::Symbol
    ty::Ty
end

struct _BinaryCompositionOption
    name::Symbol
    sibling_ty::Ty
    seed_first::Bool
end

mutable struct _CompositionMenus
    unary::Union{Nothing,Vector{Symbol}}
    binary::Union{Nothing,Vector{_BinaryCompositionOption}}
end

mutable struct _CompositionDirectory
    heads::Vector{_CompositionHead}
    menus::Dict{Ty,_CompositionMenus}
end

"""
    _prepare_composition_directory(global_consts, global_consts_ty)

Build the ordered global-head view for one mutation call. Both dictionaries and
all reachable type values are borrowed read-only. Open monotypes are retained
for compatibility fallback; only admitted closed site types index menus.
"""
function _prepare_composition_directory(
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
)
    heads = _CompositionHead[]
    for name in sort!(collect(keys(global_consts_ty)))
        name isa Symbol || continue
        haskey(global_consts, name) || continue
        raw = global_consts_ty[name]
        raw isa Ty || continue
        # Normalize each borrowed global descriptor once. The resulting view is
        # read-only; unification still happens only in the detached predicate.
        push!(heads, _CompositionHead(name, TypeSystem.prune(raw)))
    end
    return _CompositionDirectory(heads, Dict{Ty,_CompositionMenus}())
end

function _prepare_composition_directory(ctx::NamedTuple)
    global_consts = get(ctx, :global_consts, nothing)
    global_consts_ty = get(ctx, :global_consts_ty, nothing)
    global_consts isa AbstractDict || return _CompositionDirectory(
        _CompositionHead[], Dict{Ty,_CompositionMenus}(),
    )
    global_consts_ty isa AbstractDict || return _CompositionDirectory(
        _CompositionHead[], Dict{Ty,_CompositionMenus}(),
    )
    return _prepare_composition_directory(global_consts, global_consts_ty)
end

function _composition_directory(ctx::NamedTuple)
    prepared = get(ctx, :composition_directory, nothing)
    prepared isa _CompositionDirectory && return prepared
    # Standalone eligibility/apply calls use this same implementation as a
    # one-shot preparation; mutate_tree passes a shared call-local owner.
    return _prepare_composition_directory(ctx)
end

function _build_unary_composition_options(
    directory::_CompositionDirectory,
    target::Ty,
)
    options = Symbol[]
    for head in directory.heads
        arrow = head.ty
        arrow isa TyFun || continue
        argument_ty = arrow.arg
        result_ty = arrow.res
        _same_composition_type(argument_ty, target) || continue
        _same_composition_type(result_ty, target) || continue
        push!(options, head.name)
    end
    return options
end

function _build_binary_composition_options(
    directory::_CompositionDirectory,
    target::Ty,
)
    options = _BinaryCompositionOption[]
    for head in directory.heads
        first_arrow = head.ty
        first_arrow isa TyFun || continue
        second_arrow = first_arrow.res
        second_arrow isa TyFun || continue
        result_ty = second_arrow.res
        _same_composition_type(result_ty, target) || continue

        left_ty = first_arrow.arg
        right_ty = second_arrow.arg
        if _generation_closed_type(right_ty) &&
                _same_composition_type(left_ty, target)
            push!(options, _BinaryCompositionOption(head.name, right_ty, true))
        end
        if _generation_closed_type(left_ty) &&
                _same_composition_type(right_ty, target)
            push!(options, _BinaryCompositionOption(head.name, left_ty, false))
        end
    end
    return options
end

function _composition_options(kind::Symbol, site::Site, ctx::NamedTuple)
    _generation_closed_type(site.ty) || return nothing
    directory = _composition_directory(ctx)
    target = TypeSystem.prune(site.ty)
    menus = get!(directory.menus, target) do
        _CompositionMenus(nothing, nothing)
    end
    if kind === :unary
        if menus.unary === nothing
            menus.unary = _build_unary_composition_options(directory, target)
        end
        return menus.unary::Vector{Symbol}
    elseif kind === :binary
        if menus.binary === nothing
            menus.binary = _build_binary_composition_options(directory, target)
        end
        return menus.binary::Vector{_BinaryCompositionOption}
    end
    error("unknown composition menu kind: $(repr(kind))")
end

_unary_composition_options(site::Site, ctx::NamedTuple) = begin
    options = _composition_options(:unary, site, ctx)
    options === nothing ? Symbol[] : options
end

_binary_composition_options(site::Site, ctx::NamedTuple) = begin
    options = _composition_options(:binary, site, ctx)
    options === nothing ? _BinaryCompositionOption[] : options
end

function is_site_eligible(
    mutation::BinaryCompositionMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _site_scope_is_monomorphic(site) || return false
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
    _site_scope_is_monomorphic(site) || return Err(NoEligibleSiteErr())
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
an `AbsNode` (to avoid introducing an immediate beta-redex). The new binder name
is adjusted to avoid named references in the attempt tree.
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
    factory = haskey(ctx, :symfactory) ? ctx.symfactory : SymbolGenerator()
    name = only(_fresh_binder_names(tree, factory, 1))
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

Scope-aware common subexpression extraction for pure source programs.

This mutation chooses an anchor occurrence and delegates to the common
`Transforms.eliminate_node!` path.  The transformation consumes the
attempt-local `ProgramAnalysis.SyntaxIndex`, compares captures by binding
identity, and introduces a let-like application around a legal host.

When `min_count=1`, this extracts a single occurrence without changing its
completed pure-program value. It adds a binding and can change scalar-edit
opportunities, tree size, and execution-step costs; it is not a cost-free move.
Single-lift mode admits non-index sites. Repeated CSE mode (`min_count>1`) keeps
the default candidate filter to applications so it avoids low-value repeated
variables and constants.
"""
struct CSEMutation <: AbstractMutation
    min_count::Int
end
CSEMutation(; min_count::Int=1) = CSEMutation(min_count)
required_site_kinds(::CSEMutation) = Symbol[:any]

function is_site_eligible(m::CSEMutation, site::Site, env::MutationEnv, ctx::NamedTuple)
    index = haskey(ctx, :cse_index) ? ctx.cse_index : nothing
    index isa SyntaxIndex || return false
    if m.min_count <= 1
        return !(site.node isa IndexNode)
    end
    site.node isa AppNode || return false
    env.tree === nothing && return true
    cache_key = Symbol("cse_repeated_nodes_", m.min_count)
    cached = get!(env.eligibility_cache, cache_key) do
        (index, repeated_nodes(index; min_count=m.min_count))
    end
    cached_index, eligible = cached
    eligible isa Err && return true
    id = site.occurrence > 0 ? site.occurrence : occurrence_id(cached_index, site.path)
    return id !== nothing && id in eligible
end

function apply(m::CSEMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    haskey(ctx, :cse_index) || return Err(ArgumentError("CSE attempt context lacks its source index"))
    index = ctx.cse_index
    index === nothing && return Err(ArgumentError("CSE does not support this syntax tree"))
    occurrence = site.occurrence > 0 ? site.occurrence : occurrence_id(index, site.path)
    occurrence === nothing && return Err(ArgumentError("CSE site path is not in the attempt tree"))
    return eliminate_node!(
        tree,
        index,
        occurrence;
        min_count=m.min_count,
        factory=ctx.symfactory,
    )
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

function _fresh_binder_names(tree::AstTree, factory, count::Int)
    count >= 0 || throw(ArgumentError("binder name count must be non-negative"))
    used = Set{Symbol}()
    for node in tree
        if node isa Union{AbsNode,VarNode}
            push!(used, node.name)
        end
    end

    names = Symbol[]
    sizehint!(names, count)
    for _ in 1:count
        suggestion = factory()
        suggestion isa Symbol || throw(ArgumentError(
            "symbol factory must return Symbol, got $(typeof(suggestion))",
        ))
        candidate = suggestion
        suffix = 1
        while candidate in used
            candidate = Symbol(string(suggestion), "_", string(suffix))
            suffix += 1
        end
        push!(used, candidate)
        push!(names, candidate)
    end
    return names
end

"""
    mutate_tree(tree::AstTree, target_ty::Ty; kwargs...) -> Union{AstTree,Err}

Apply a single mutation step that preserves `target_ty` (up to `iscompatible`).
Returns `Err(MaxTryReachedErr())` if no valid mutation can be found within `max_try` attempts.

# Key keyword arguments
- `mutations`: available mutation operators (default: MVP set)
- `chooser`: how to pick `(mutation, site)` (default: `WeightedPairChooser()`)
- `provider`: how to build sites (default: `IndexedProvider()`)
- Selection and analysis borrow the input tree and constant dictionaries as
  read-only; each eligible `apply` gets one private tree copy and local-dictionary
  copies, committed only after successful type validation.
- `prune_consts`: call `prune_local_consts!` after a successful mutation (default: `true`)
- `global_consts_ty`, `local_consts`, `local_consts_ty`: constant/type containers
- `global_consts`: optional runtime values for global constants used by generated replacements
- `const_symfactory`: per-call deterministic local-constant symbol factory (default: `ConstSymbolFactory()`)
- `tyfactory`, `symfactory`: local factories used by type inference and CSE

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
    prune_consts::Bool=true,
    global_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    global_consts_ty::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    local_consts_ty::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(),
    local_consts::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    tyfactory=newtyvar,
    symfactory=SymbolGenerator(),
    const_symfactory::ConstSymbolFactory=ConstSymbolFactory(),
    bloat_policy::BloatPolicy=BloatPolicy(),
    adaptive_policy::AdaptivePolicy=AdaptivePolicy(),
    recorder::Union{Nothing,MutationAttemptRecorder}=nothing,
)
    current_tree = tree
    # Selection borrows caller dictionaries; only an eligible apply gets copies.
    caller_local_consts = local_consts
    caller_local_consts_ty = local_consts_ty
    current_local_consts = caller_local_consts
    current_local_consts_ty = caller_local_consts_ty

    # Prepare sites once (the retry loop works on copies of the same input).
    base_type_ctx = merge(global_consts_ty, current_local_consts_ty)
    uses_cse = any(mutation -> mutation isa CSEMutation, mutations)
    typed_analysis = _common_mutation_analysis(
        current_tree,
        base_type_ctx,
        tyfactory,
        target_ty,
    )
    if typed_analysis isa Err
        record!(recorder, nothing, :prepare_err; err_type=typeof(typed_analysis.value))
        return typed_analysis
    end
    cse_index = if uses_cse && typed_analysis isa TypedAnalysis
        typed_analysis.index
    else
        nothing
    end
    uses_composition = any(
        mutation -> mutation isa UnaryCompositionMutation ||
                    mutation isa BinaryCompositionMutation,
        mutations,
    )
    composition_directory = uses_composition ?
        _prepare_composition_directory(global_consts, global_consts_ty) : nothing
    base_ctx = (;
        rng=rng,
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts_ty=current_local_consts_ty,
        local_consts=current_local_consts,
        tyfactory=tyfactory,
        symfactory=symfactory,
        const_symfactory=const_symfactory,
        typed_analysis,
        cse_index,
        composition_directory,
    )
    env = MutationEnv(
        rng,
        target_ty,
        base_type_ctx,
        current_tree,
        typed_analysis,
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
        choice = choose_pair(chooser, mutations, provider, current_tree, base_type_ctx, env, base_ctx)
        if choice isa Err
            record!(recorder, nothing, :choose_err; err_type=typeof(choice.value))
            return choice
        end

        m = choice.mutation::AbstractMutation
        source_site = _authoritative_site(
            base_ctx.typed_analysis,
            choice.site,
            base_ctx,
        )
        if source_site isa Err
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :no_eligible_site; err_type=typeof(source_site.value))
            continue
        end
        required = required_site_kinds(m)
        if !(:any in required || source_site.kind in required) ||
                !is_site_eligible(m, source_site, env, base_ctx)
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :no_eligible_site; err_type=NoEligibleSiteErr)
            continue
        end
        attempt_tree = deepcopy(current_tree)
        attempt_local_consts = copy(current_local_consts)
        attempt_local_consts_ty = copy(current_local_consts_ty)
        attempt_ctx = (;
            rng=rng,
            global_consts=global_consts,
            global_consts_ty=global_consts_ty,
            local_consts_ty=attempt_local_consts_ty,
            local_consts=attempt_local_consts,
            tyfactory=tyfactory,
            symfactory=symfactory,
            const_symfactory=const_symfactory,
            cse_index=base_ctx.cse_index,
            typed_analysis=base_ctx.typed_analysis,
            composition_directory=base_ctx.composition_directory,
        )
        attempt_site = _remap_authoritative_site(attempt_tree, source_site)
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
        query_ctx, detached_targets = TypeSystem._detached_type_inputs(
            attempt_type_ctx,
            target_ty,
        )
        query_target = only(detached_targets)
        inferred = _infer_mutation_type(
            attempt_tree,
            query_ctx,
            tyfactory,
        )
        if inferred isa Err
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :infer_err; err_type=typeof(inferred.value))
            continue
        end
        if !iscompatible(inferred, query_target)
            _update_stat!(env, m, false)
            record!(recorder, typeof(m), :incompatible)
            continue
        end

        _update_stat!(env, m, true)
        record!(recorder, typeof(m), :success)
        current_tree = attempt_tree
        current_local_consts = attempt_local_consts
        current_local_consts_ty = attempt_local_consts_ty
        empty!(caller_local_consts)
        empty!(caller_local_consts_ty)
        merge!(caller_local_consts, current_local_consts)
        merge!(caller_local_consts_ty, current_local_consts_ty)
        return current_tree
    end
    record!(recorder, nothing, :max_try_exhausted)
    return Err(MaxTryReachedErr())
end

include("ScopedRewrite.jl")
include("BindingInsertion.jl")
include("Graft.jl")
include("FunctionDefMutation.jl")
include("ProductPacking.jl")
include("Population.jl")

########################################################
# Test Items
########################################################

@testsnippet MutBaseImports begin
    using LambdaRegression.Mutations: prune_local_consts!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode, FunctionDefNode
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
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.Mutations: mutate_tree, BoolConstMutation, WeightedPairChooser, IndexedProvider, MutationAttemptRecorder, mutation_recorder_summary
end

@testitem "Mutations root constant weight follows only complete leading lambda roots" setup=[MutBaseImports] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.CanonicalGeneration: GenerationConfig, GenerationWeights
    using LambdaRegression.Mutations:
        AdaptivePolicy, BloatPolicy, CanonicalReplacementGenerator,
        ExponentialSiteSelectionPolicy, MutationEnv, MutationStat,
        ParameterIntroductionMutation, Site, _canonical_replacement_kwargs,
        _is_complete_output_root, site_weight
    using LambdaRegression.TypeSystem: FloatTy

    body = AppNode(AppNode(ConstNode(:ADD), IndexNode(1)), IndexNode(0))
    tree = AstTree(AbsNode(:x, AbsNode(:y, body)))
    at(path, node) = Site(
        Int[path...], node, nothing, 1, length(path), :any, FloatTy, TyLike[],
    )
    root_site = at(Int[], tree.root)
    middle_lambda_site = at([1], tree.root.body)
    output_site = at([1, 1], body)
    nested_site = at([1, 1, 1], body.func)

    @test _is_complete_output_root(tree, root_site)
    @test _is_complete_output_root(tree, middle_lambda_site)
    @test _is_complete_output_root(tree, output_site)
    @test !_is_complete_output_root(tree, nested_site)

    callback_tree = AstTree(AbsNode(:x,
        AppNode(ConstNode(:MAP), AbsNode(:y, IndexNode(0)))))
    @test !_is_complete_output_root(callback_tree,
        at([1, 2], callback_tree.root.body.arg))
    rhs_tree = AstTree(AppNode(ConstNode(:ADD), ConstNode(:rhs)))
    @test !_is_complete_output_root(rhs_tree, at([2], rhs_tree.root.arg))
    functiondef_tree = AstTree(AbsNode(:x,
        FunctionDefNode(AbsNode(:f, IndexNode(0)))))
    @test !_is_complete_output_root(functiondef_tree,
        at([1, 1], functiondef_tree.root.body))

    env = MutationEnv(
        MersenneTwister(1), FloatTy, Dict{Symbol,TyLike}(), tree, nothing,
        false, IdDict{Any,Any}(), AdaptivePolicy(),
        IdDict{DataType,MutationStat}(), BloatPolicy(),
    )
    ctx = (
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    weights = GenerationWeights(default=2, variable=3)
    generator = CanonicalReplacementGenerator(root_constant_weight=0.1)
    for site in (root_site, middle_lambda_site, output_site)
        kwargs = _canonical_replacement_kwargs(generator, site, env, ctx, (; weights))
        @test kwargs.weights.overrides[:root_constant] == 0.1
        @test kwargs.weights.default == 2
        @test kwargs.weights.overrides[:variable] == 3
    end
    local_kwargs = _canonical_replacement_kwargs(
        generator, nested_site, env, ctx, (; weights),
    )
    @test local_kwargs.weights === weights
    @test !haskey(weights.overrides, :root_constant)
    neutral_kwargs = _canonical_replacement_kwargs(
        CanonicalReplacementGenerator(), root_site, env, ctx, (; weights),
    )
    @test neutral_kwargs.weights === weights
    @test_throws ArgumentError _canonical_replacement_kwargs(
        generator, root_site, env, ctx, (; config=GenerationConfig()),
    )

    policy = ExponentialSiteSelectionPolicy(
        scope_bias=0, path_bias=0, root_weight=1,
    )
    parameter_mutation = ParameterIntroductionMutation(
        site_selection=policy, root_constant_weight=0.1,
    )
    @test site_weight(parameter_mutation, root_site, env, ctx) == 0.1
    @test site_weight(parameter_mutation, middle_lambda_site, env, ctx) == 0.1
    @test site_weight(parameter_mutation, output_site, env, ctx) == 0.1
    @test site_weight(parameter_mutation, nested_site, env, ctx) == 1.0
    @test site_weight(ParameterIntroductionMutation(site_selection=policy),
        root_site, env, ctx) == 1.0
end

@testitem "Mutations fresh binder names suffix once per suggestion" setup=[MutBaseImports] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: VarNode
    using LambdaRegression.Mutations: _fresh_binder_names

    tree = AstTree(AbsNode(:seed, VarNode(:seed_1)))
    calls = Ref(0)
    factory = () -> (calls[] += 1; :seed)
    names = _fresh_binder_names(tree, factory, 2)

    @test names == [:seed_2, :seed_3]
    @test calls[] == 2
end

@testitem "Mutations FunctionDef sites use shared typed binder schemes" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem: TyFun, IntTy

    self_ty = TyFun(IntTy, IntTy)
    fd = FunctionDefNode(AbsNode(:x, IndexNode(0)))
    tree = AstTree(AppNode(ConstNode(:consume), fd))
    type_ctx = Dict{Symbol,TyLike}(:consume => TyFun(self_ty, IntTy))
    analysis = analyze_program(tree, type_ctx; target_ty=IntTy)
    @test analysis isa TypedAnalysis
    analysis isa TypedAnalysis || return

    sites = LambdaRegression.Mutations._collect_analysis_sites(
        analysis,
        (; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}()),
    )
    functiondef_site = only(filter(site -> site.node isa FunctionDefNode, sites))
    body_site = only(filter(
        site -> site.node isa IndexNode && site.node.index == 0,
        sites,
    ))
    @test functiondef_site.ty == self_ty
    @test map(LambdaRegression.Mutations._site_binder_monotype, body_site.binder_types) ==
        TyLike[self_ty, IntTy]
end

@testitem "Mutations IndexedProvider uses constrained occurrence analysis" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs:
        AppNode, AbsNode, ConstNode, FunctionDefNode, IndexNode, VarNode, AstNode
    using LambdaRegression.Mutations: MutationEnv, AdaptivePolicy, BloatPolicy
    using LambdaRegression.TypeSystem: TyScheme, TyVar

    shared = ConstNode(:x)
    tree = AstTree(AppNode(
        AppNode(ConstNode(:add), shared),
        shared,
    ))
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :x => FloatTy,
    )
    env = MutationEnv(
        MersenneTwister(42), FloatTy, type_ctx, tree, nothing, false,
        IdDict{Any,Any}(), AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        BloatPolicy(),
    )
    ctx = (; local_consts=Dict{Symbol,Any}(), local_consts_ty=Dict{Symbol,TyLike}())
    provider = IndexedProvider()
    @test LambdaRegression.Mutations.prepare!(provider, tree, type_ctx, env, ctx) === nothing

    xsites = filter(
        site -> site.node isa ConstNode && site.node.name === :x,
        LambdaRegression.Mutations.sites(provider),
    )
    @test length(xsites) == 2
    @test xsites[1].occurrence != xsites[2].occurrence
    @test xsites[1].node == xsites[2].node
    @test getfield(xsites[1], :path_cache) === nothing
    @test getfield(xsites[1], :binder_types_cache) === nothing
    @test xsites[1].node isa ConstNode
    @test getfield(xsites[1], :path_cache) === nothing
    first_path = xsites[1].path
    @test first_path === xsites[1].path
    @test [first_path, xsites[2].path] == [[1, 2], [2]]
    @test xsites[1].ty === FloatTy
    @test xsites[1].path === xsites[1].path
    @test xsites[1].binder_types === xsites[1].binder_types

    id_var = newtyvar(0)
    id_ctx = Dict{Symbol,TyLike}(
        :id => TyScheme([id_var], TyFun(id_var, id_var)),
    )
    id_tree = AstTree(ConstNode(:id))
    id_env = MutationEnv(
        MersenneTwister(555), TyFun(IntTy, IntTy), id_ctx, id_tree, nothing,
        false, IdDict{Any,Any}(), AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(), BloatPolicy(),
    )
    id_provider = IndexedProvider()
    @test LambdaRegression.Mutations.prepare!(
        id_provider,
        id_tree,
        id_ctx,
        id_env,
        ctx,
    ) === nothing
    @test iscompatible(id_provider.sites_all[1].ty, TyFun(IntTy, IntTy))

    self_ty = TyFun(FloatTy, FloatTy)
    fd_tree = AstTree(FunctionDefNode(AbsNode(
        :arg, AppNode(IndexNode(1), IndexNode(0)),
    )))
    fd_env = MutationEnv(
        MersenneTwister(1111), self_ty, Dict{Symbol,TyLike}(), fd_tree, nothing,
        false, IdDict{Any,Any}(), AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(), BloatPolicy(),
    )
    fd_provider = IndexedProvider()
    @test LambdaRegression.Mutations.prepare!(
        fd_provider,
        fd_tree,
        Dict{Symbol,TyLike}(),
        fd_env,
        ctx,
    ) === nothing
    self_reference = only(filter(
        site -> site.node isa IndexNode && site.node.index == 1,
        LambdaRegression.Mutations.sites(fd_provider),
    ))
    @test map(LambdaRegression.Mutations._site_binder_monotype, self_reference.binder_types) ==
        TyLike[self_ty, FloatTy]

    struct UnknownMutationNode <: AstNode end
    @test_throws ArgumentError mutate_tree(
        AstTree(UnknownMutationNode()), IntTy; mutations=LambdaRegression.Mutations.AbstractMutation[],
    )
end

@testitem "Mutations preserve polymorphic binder slots and reject quantified generation scopes" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: IndexNode
    using LambdaRegression.Mutations: MutationEnv, AdaptivePolicy, BloatPolicy, PointMutation
    using LambdaRegression.TypeSystem: TyScheme, TyVar, iscompatible

    quantified_var = newtyvar(0)
    poly = TyScheme([quantified_var], quantified_var)
    empty_scheme = TyScheme(TyVar[], FloatTy)
    mono_site = LambdaRegression.Mutations.Site(
        Int[], IndexNode(0), nothing, 0, 1, :index, FloatTy, TyLike[empty_scheme],
    )
    @test LambdaRegression.Mutations._site_scope_is_monomorphic(mono_site)
    @test LambdaRegression.Mutations._site_scope_types(mono_site) == [FloatTy]

    # Index 1 addresses the polymorphic middle slot; compressing that slot would
    # incorrectly make Index 1 refer to the outer Int binder instead.
    site = LambdaRegression.Mutations.Site(
        Int[], IndexNode(0), nothing, 0, 3, :index, BoolTy,
        TyLike[IntTy, poly, BoolTy],
    )
    @test length(site.binder_types) == 3
    @test iscompatible(BoolTy, site.binder_types[2])
    @test !LambdaRegression.Mutations._site_scope_is_monomorphic(site)
    env = MutationEnv(
        MersenneTwister(1), BoolTy, Dict{Symbol,TyLike}(),
        AstTree(IndexNode(0)), nothing, false, IdDict{Any,Any}(),
        AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(), BloatPolicy(),
    )
    changed = LambdaRegression.Mutations.apply(
        PointMutation(), env.tree, site, env, NamedTuple(),
    )
    @test changed isa AstTree
    @test changed.root == IndexNode(1)

    generated = LambdaRegression.Mutations._infer_binding_rhs_scheme(
        AstTree(IndexNode(0)), mono_site, NamedTuple(), Dict{Symbol,TyLike}(), newtyvar,
    )
    @test generated isa TyScheme
    rejected = LambdaRegression.Mutations._infer_binding_rhs_scheme(
        AstTree(IndexNode(0)), site, NamedTuple(), Dict{Symbol,TyLike}(), newtyvar,
    )
    @test rejected isa Err
    @test rejected.value isa LambdaRegression.Mutations.NoEligibleSiteErr
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
    using LambdaRegression.TypeSystem: get_type_ctx, infer, iscompatible
    using LambdaRegression.Primitives: FADD, SQR

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
    using LambdaRegression.Primitives: FADD

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
    using LambdaRegression.ProgramAnalysis: analyze_program
    # (λx. (k x)) : Float -> Float
    local_consts = Dict{Symbol,Any}(:k => (x -> x))
    local_consts_ty = Dict{Symbol,TyLike}(:k => TyFun(FloatTy, FloatTy))
    global_consts_ty = Dict{Symbol,TyLike}()
    type_ctx = merge(global_consts_ty, local_consts_ty)
    tree = AstTree(AbsNode(:x, AppNode(ConstNode(:k), IndexNode(0))))
    stale_tree = AstTree(ConstNode(:k))
    stale_analysis = analyze_program(
        stale_tree, type_ctx; target_ty=TyFun(FloatTy, FloatTy),
    )

    env = LambdaRegression.Mutations.MutationEnv(
        MersenneTwister(1111),
        TyFun(FloatTy, FloatTy),
        type_ctx,
        stale_tree,
        stale_analysis,
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
    @test env.analysis !== stale_analysis
    @test any(node -> node isa IndexNode, env.analysis.index.nodes)

    sites = LambdaRegression.Mutations.sites(provider)
    s0 = first(filter(s -> s.node isa IndexNode, sites))
    attempt_tree = deepcopy(tree)
    remapped = LambdaRegression.Mutations._remap_site(
        attempt_tree,
        s0,
        env.analysis::LambdaRegression.ProgramAnalysis.TypedAnalysis,
        ctx,
    )
    expected = LambdaRegression.Mutations._node_at_path(attempt_tree, s0.path)
    @test remapped.node == expected
end

@testitem "Mutations remap chooser sites through the current typed analysis" setup=[MutBaseImports] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, ConstNode
    using LambdaRegression.Mutations:
        AbstractPairChooser, AbstractSiteProvider, MutationEnv, PairChoice, Site,
        _site_binder_monotype
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, TyFun, TyLike

    struct CaptureSiteMutation <: LambdaRegression.Mutations.AbstractMutation
        seen::Base.RefValue{Any}
        required_kind::Symbol
        required_ty::Union{Nothing,TyLike}
    end
    CaptureSiteMutation(seen) = CaptureSiteMutation(seen, :any, nothing)
    LambdaRegression.Mutations.required_site_kinds(m::CaptureSiteMutation) =
        Symbol[m.required_kind]
    LambdaRegression.Mutations.is_site_eligible(
        m::CaptureSiteMutation,
        site::Site,
        ::Any,
        ::NamedTuple,
    ) = m.required_ty === nothing || site.ty == m.required_ty
    function LambdaRegression.Mutations.apply(
        mutation::CaptureSiteMutation,
        tree::AstTree,
        site::Site,
        ::Any,
        ::NamedTuple,
    )
        mutation.seen[] = site
        return tree
    end

    struct FixedSiteChooser <: AbstractPairChooser
        site::Site
    end
    function LambdaRegression.Mutations.choose_pair(
        chooser::FixedSiteChooser,
        mutations::Vector{<:LambdaRegression.Mutations.AbstractMutation},
        ::AbstractSiteProvider,
        ::AstTree,
        ::Dict{Symbol,<:TyLike},
        ::MutationEnv,
        ::NamedTuple,
    )
        return PairChoice(first(mutations), chooser.site)
    end

    tree = AstTree(AbsNode(:outer, AbsNode(:inner, ConstNode(:v))))
    target_ty = TyFun(IntTy, TyFun(BoolTy, IntTy))
    type_ctx = Dict{Symbol,TyLike}(:v => IntTy)
    current = analyze_program(tree, type_ctx; target_ty)
    @test current isa TypedAnalysis
    current isa TypedAnalysis || return
    occurrence = LambdaRegression.Mutations.occurrence_id(current.index, [1, 1])
    canonical_site = Site(current, occurrence, :const_global)

    # A hand-built proposal may provide only a path. Its other cached claims
    # deliberately disagree with the current analysis.
    offered = Site(
        Int[1, 1], ConstNode(:v), AbsNode(:spoof, ConstNode(:v)), 1,
        99, :abs, IntTy, TyLike[BoolTy], 0,
    )
    accepted = CaptureSiteMutation(Ref{Any}(nothing))
    result = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(2000),
        global_consts_ty=type_ctx,
        mutations=[accepted],
        chooser=FixedSiteChooser(offered),
        provider=IndexedProvider(),
        max_try=1,
        prune_consts=false,
    )
    @test result isa AstTree
    seen = accepted.seen[]
    @test seen isa Site
    @test seen.path == Int[1, 1]
    @test seen.ty == canonical_site.ty == IntTy
    @test map(_site_binder_monotype, seen.binder_types) ==
        map(_site_binder_monotype, canonical_site.binder_types) == TyLike[IntTy, BoolTy]
    @test seen.depth == 2
    @test seen.kind == :const_global
    @test seen.occurrence == occurrence

    # A fake type can satisfy an operator's chooser-time predicate, but the
    # post-choice check runs on the W-projected source site and rejects it.
    fake_float = Site(
        Int[1, 1], ConstNode(:v), AbsNode(:spoof, ConstNode(:v)), 1,
        2, :const_global, FloatTy, TyLike[FloatTy], 0,
    )
    float_only = CaptureSiteMutation(Ref{Any}(nothing), :any, FloatTy)
    result = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1234),
        global_consts_ty=type_ctx,
        mutations=[float_only],
        chooser=FixedSiteChooser(fake_float),
        provider=IndexedProvider(),
        max_try=1,
        prune_consts=false,
    )
    @test result isa Err
    @test float_only.seen[] === nothing

    # A fake kind is likewise not authoritative for required-kind routing.
    fake_abs_kind = Site(
        Int[1, 1], ConstNode(:v), AbsNode(:spoof, ConstNode(:v)), 1,
        2, :abs, IntTy, TyLike[BoolTy], 0,
    )
    abs_only = CaptureSiteMutation(Ref{Any}(nothing), :abs, nothing)
    result = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(8000),
        global_consts_ty=type_ctx,
        mutations=[abs_only],
        chooser=FixedSiteChooser(fake_abs_kind),
        provider=IndexedProvider(),
        max_try=1,
        prune_consts=false,
    )
    @test result isa Err
    @test abs_only.seen[] === nothing

    # An occurrence from another analysis is stale even when the selected
    # node has the same monotype; here the visible binder slots differ.
    stale_target = TyFun(BoolTy, TyFun(FloatTy, IntTy))
    stale = analyze_program(tree, type_ctx; target_ty=stale_target)
    @test stale isa TypedAnalysis
    stale isa TypedAnalysis || return
    stale_site = Site(stale, occurrence, :const_global)
    @test stale_site.ty == canonical_site.ty
    @test stale_site.binder_types != canonical_site.binder_types

    mismatched = Site(current, occurrence, :const_global)
    setfield!(mismatched, :path_cache, Int[1])
    rejected_path = CaptureSiteMutation(Ref{Any}(nothing))
    result = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[rejected_path],
        chooser=FixedSiteChooser(mismatched),
        provider=IndexedProvider(),
        max_try=1,
        prune_consts=false,
    )
    @test result isa Err
    @test rejected_path.seen[] === nothing

    rejected = CaptureSiteMutation(Ref{Any}(nothing))
    original = deepcopy(tree)
    result = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[rejected],
        chooser=FixedSiteChooser(stale_site),
        provider=IndexedProvider(),
        max_try=1,
        prune_consts=false,
    )
    @test result isa Err
    @test rejected.seen[] === nothing
    @test tree == original
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

@testitem "Mutations selection borrows inputs and apply receives a remapped private candidate" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: AbsNode, setroot!
    using LambdaRegression.Mutations:
        AbstractMutation, AbstractPairChooser, AbstractSiteProvider, PairChoice
    using LambdaRegression.TypeSystem: TyFun

    mutable struct CopyBoundaryObservation
        selected_tree::Any
        selected_local_consts::Any
        selected_local_consts_ty::Any
        selected_site::Any
        attempt_tree::Any
        attempt_local_consts::Any
        attempt_local_consts_ty::Any
        attempt_site::Any
        site_matches_candidate::Bool
    end

    struct CopyBoundaryMutation <: AbstractMutation
        observation::CopyBoundaryObservation
    end
    function LambdaRegression.Mutations.apply(
        mutation::CopyBoundaryMutation,
        tree::AstTree,
        site::LambdaRegression.Mutations.Site,
        ::Any,
        ctx::NamedTuple,
    )
        observation = mutation.observation
        observation.attempt_tree = tree
        observation.attempt_local_consts = ctx.local_consts
        observation.attempt_local_consts_ty = ctx.local_consts_ty
        observation.attempt_site = site
        observation.site_matches_candidate =
            site.path == [1] && site.parent == tree.root && site.node == tree.root.body
        setroot!(tree, AbsNode(:x, ConstNode(:m)))
        ctx.local_consts[:k] = 5.0
        return tree
    end

    struct CopyBoundaryChooser <: AbstractPairChooser
        observation::CopyBoundaryObservation
    end
    function LambdaRegression.Mutations.choose_pair(
        chooser::CopyBoundaryChooser,
        mutations::Vector{<:AbstractMutation},
        provider::AbstractSiteProvider,
        tree::AstTree,
        ::Dict{Symbol,<:TyLike},
        ::LambdaRegression.Mutations.MutationEnv,
        ctx::NamedTuple,
    )
        observation = chooser.observation
        observation.selected_tree = tree
        observation.selected_local_consts = ctx.local_consts
        observation.selected_local_consts_ty = ctx.local_consts_ty
        site = only(filter(
            site -> site.node isa ConstNode && site.node.name === :k,
            LambdaRegression.Mutations.sites(provider),
        ))
        observation.selected_site = site
        return PairChoice(first(mutations), site)
    end

    observation = CopyBoundaryObservation(
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false,
    )
    mutation = CopyBoundaryMutation(observation)
    chooser = CopyBoundaryChooser(observation)
    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    local_consts = Dict{Symbol,Any}(:k => 1.0, :m => 2.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy, :m => FloatTy)

    result = mutate_tree(
        tree,
        TyFun(FloatTy, FloatTy);
        mutations=[mutation],
        chooser=chooser,
        provider=IndexedProvider(),
        local_consts,
        local_consts_ty,
        prune_consts=false,
    )

    @test result isa AstTree
    result isa AstTree || return
    @test observation.selected_tree === tree
    @test observation.selected_local_consts === local_consts
    @test observation.selected_local_consts_ty === local_consts_ty
    @test observation.attempt_tree !== observation.selected_tree
    @test observation.attempt_tree !== tree
    @test observation.attempt_local_consts !== local_consts
    @test observation.attempt_local_consts_ty !== local_consts_ty
    @test observation.attempt_site !== observation.selected_site
    @test observation.attempt_site.path == observation.selected_site.path == [1]
    @test observation.site_matches_candidate
    @test (tree.root::AbsNode).body == ConstNode(:k)
    @test (result.root::AbsNode).body == ConstNode(:m)
    @test local_consts[:k] == 5.0
    @test local_consts[:m] == 2.0
end

@testitem "Mutations retries discard ineligible and failed private candidates" setup=[MutBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: AbsNode, IndexNode, setroot!
    using LambdaRegression.Mutations:
        AbstractMutation, AbstractPairChooser, AbstractSiteProvider, PairChoice
    using LambdaRegression.TypeSystem: TyFun

    mutable struct RetryObservation
        selected_trees::Vector{Any}
        selected_local_consts::Vector{Any}
        selected_local_consts_ty::Vector{Any}
        retry_applies::Int
        failed_candidate::Any
        failed_local_consts::Any
        successful_candidate::Any
        success_initial_value::Any
        site_matches_candidate::Bool
    end

    struct NeverEligibleMutation <: AbstractMutation
        calls::Base.RefValue{Int}
    end
    function LambdaRegression.Mutations.is_site_eligible(
        ::NeverEligibleMutation,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
    )
        return false
    end
    function LambdaRegression.Mutations.apply(
        mutation::NeverEligibleMutation,
        ::AstTree,
        ::LambdaRegression.Mutations.Site,
        ::Any,
        ::NamedTuple,
    )
        mutation.calls[] += 1
        error("an ineligible mutation must not be applied")
    end

    struct RetryApplyMutation <: AbstractMutation
        observation::RetryObservation
    end
    function LambdaRegression.Mutations.apply(
        mutation::RetryApplyMutation,
        tree::AstTree,
        site::LambdaRegression.Mutations.Site,
        ::Any,
        ctx::NamedTuple,
    )
        observation = mutation.observation
        observation.retry_applies += 1
        observation.site_matches_candidate =
            site.path == [1] && site.parent == tree.root && site.node == tree.root.body
        if observation.retry_applies == 1
            observation.failed_candidate = tree
            observation.failed_local_consts = ctx.local_consts
            setroot!(tree, AbsNode(:x, ConstNode(:m)))
            ctx.local_consts[:k] = 99.0
            return Err(LambdaRegression.Mutations.NoEligibleSiteErr())
        end

        observation.successful_candidate = tree
        observation.success_initial_value = ctx.local_consts[:k]
        setroot!(tree, AbsNode(:x, ConstNode(:m)))
        return tree
    end

    struct RetryChooser <: AbstractPairChooser
        observation::RetryObservation
        skip::NeverEligibleMutation
        retry::RetryApplyMutation
    end
    function LambdaRegression.Mutations.choose_pair(
        chooser::RetryChooser,
        ::Vector{<:AbstractMutation},
        provider::AbstractSiteProvider,
        tree::AstTree,
        ::Dict{Symbol,<:TyLike},
        ::LambdaRegression.Mutations.MutationEnv,
        ctx::NamedTuple,
    )
        observation = chooser.observation
        push!(observation.selected_trees, tree)
        push!(observation.selected_local_consts, ctx.local_consts)
        push!(observation.selected_local_consts_ty, ctx.local_consts_ty)
        site = only(filter(
            site -> site.node isa ConstNode && site.node.name === :k,
            LambdaRegression.Mutations.sites(provider),
        ))
        mutation = length(observation.selected_trees) == 1 ? chooser.skip : chooser.retry
        return PairChoice(mutation, site)
    end

    skip_calls = Ref(0)
    observation = RetryObservation(Any[], Any[], Any[], 0, nothing, nothing, nothing, nothing, false)
    skip = NeverEligibleMutation(skip_calls)
    retry = RetryApplyMutation(observation)
    chooser = RetryChooser(observation, skip, retry)
    tree = AstTree(AbsNode(:x, ConstNode(:k)))
    local_consts = Dict{Symbol,Any}(:k => 1.0, :m => 2.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy, :m => FloatTy)

    result = mutate_tree(
        tree,
        TyFun(FloatTy, FloatTy);
        mutations=AbstractMutation[skip, retry],
        chooser,
        provider=IndexedProvider(),
        max_try=3,
        local_consts,
        local_consts_ty,
        prune_consts=false,
    )

    @test result isa AstTree
    result isa AstTree || return
    @test length(observation.selected_trees) == 3
    @test all(selected -> selected === tree, observation.selected_trees)
    @test all(selected -> selected === local_consts, observation.selected_local_consts)
    @test all(selected -> selected === local_consts_ty, observation.selected_local_consts_ty)
    @test skip_calls[] == 0
    @test observation.retry_applies == 2
    @test observation.failed_candidate !== observation.successful_candidate
    @test observation.failed_local_consts !== local_consts
    @test observation.failed_candidate.root.body == ConstNode(:m)
    @test observation.failed_local_consts[:k] == 99.0
    @test observation.success_initial_value == 1.0
    @test observation.site_matches_candidate
    @test (tree.root::AbsNode).body == ConstNode(:k)
    @test local_consts[:k] == 1.0
    @test (result.root::AbsNode).body == ConstNode(:m)

    setroot!(tree, AbsNode(:x, IndexNode(0)))
    @test (result.root::AbsNode).body == ConstNode(:m)
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
        const_symfactory=LambdaRegression.Programs.ConstSymbolFactory(),
    )
    provider = IndexedProvider()
    prep = LambdaRegression.Mutations.prepare!(provider, tree, type_ctx, env, ctx)
    @test prep === nothing
    site = LambdaRegression.Mutations._remap_site(
        tree,
        only([s for s in LambdaRegression.Mutations.sites(provider) if s.path == Int[1, 2]]),
        env.analysis::LambdaRegression.ProgramAnalysis.TypedAnalysis,
        ctx,
    )

    result = LambdaRegression.Mutations.apply(EtaExpandMutation(), tree, site, env, ctx)

    expected = AstTree(AbsNode(:z, AppNode(ConstNode(:h), AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))))))
    @test result isa AstTree
    @test tree == expected
    @test iscompatible(infer(tree, global_consts_ty), target_ty)
end

@testitem "Mutations EtaExpandMutation avoids named variable capture" setup=[MutBaseImports] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: VarNode, eliminate_bound_variables!
    using LambdaRegression.Mutations: MutationEnv, AdaptivePolicy, BloatPolicy, apply
    using LambdaRegression.TypeSystem: infer, iscompatible

    tree = AstTree(
        AbsNode(
            :x,
            AppNode(ConstNode(:make), VarNode(:x)),
        ),
    )
    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    type_ctx = Dict{Symbol,TyLike}(:make => TyFun(IntTy, TyFun(IntTy, IntTy)))
    env = MutationEnv(
        MersenneTwister(42), target_ty, type_ctx, tree, nothing, false,
        IdDict{Any,Any}(), AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(), BloatPolicy(),
    )
    ctx = (; symfactory=() -> :x)
    source_site = LambdaRegression.Mutations.Site(
        [1], tree.root.body, tree.root, 1, 1, :any,
        TyFun(IntTy, IntTy), TyLike[TyFun(IntTy, IntTy)],
    )

    result = apply(EtaExpandMutation(), tree, source_site, env, ctx)

    @test result isa AstTree
    @test iscompatible(infer(result, type_ctx), target_ty)
    normalized = deepcopy(result)
    eliminate_bound_variables!(normalized)
    @test normalized.root == AbsNode(
        :x,
        AbsNode(
            :x_1,
            AppNode(
                AppNode(ConstNode(:make), IndexNode(1)),
                IndexNode(0),
            ),
        ),
    )
end

@testitem "Mutations CondAppPullMutation pulls application through COND + preserves type" setup=[MutBaseImports] begin
    using LambdaRegression.Primitives: COND
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
    using LambdaRegression.TypeSystem: get_type_ctx
    using LambdaRegression.Primitives: FADD

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

@testitem "Mutations ContextualExpansionMutation avoids named seed capture" setup=[MutBaseImports] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        ContextualExpansionMutation,
        PathSiteSelectionPolicy,
        MutationEnv,
        Site,
        generate_replacement
    using LambdaRegression.ASTs.NodeASTs: VarNode, eliminate_bound_variables!
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    struct UsesIndexSeedAndNamedOuter <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::UsesIndexSeedAndNamedOuter,
        site::Site,
        ::MutationEnv,
        ctx::NamedTuple,
        ::NamedTuple,
    )
        @test site.binder_types == TyLike[IntTy, IntTy]
        context = AppNode(
            AppNode(ConstNode(:combine), IndexNode(0)),
            VarNode(:seed),
        )
        return GenerationResult(
            AstTree(context),
            site.ty,
            copy(ctx.local_consts),
            copy(ctx.local_consts_ty),
            Dict{Symbol,Any}(),
        )
    end

    original = AstTree(AbsNode(:seed, IndexNode(0)))
    target_ty = TyFun(IntTy, IntTy)
    values = Dict{Symbol,Any}(:combine => ((a, b) -> a + b))
    types = Dict{Symbol,TyLike}(
        :combine => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    mutation = ContextualExpansionMutation(
        site_selection=PathSiteSelectionPolicy(Int[1]),
        replacement_generator=UsesIndexSeedAndNamedOuter(),
    )
    result = mutate_tree(
        original,
        target_ty;
        rng=MersenneTwister(1111),
        global_consts=values,
        global_consts_ty=types,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(ContextualExpansionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa AstTree
    result isa AstTree || return
    @test iscompatible(infer(result, types), target_ty)
    normalized = deepcopy(result)
    eliminate_bound_variables!(normalized)
    @test normalized.root == AbsNode(
        :seed,
        AppNode(
            AbsNode(
                :seed_1,
                AppNode(
                    AppNode(ConstNode(:combine), IndexNode(0)),
                    IndexNode(1),
                ),
            ),
            IndexNode(0),
        ),
    )
end

@testitem "Mutations composition compatibility preserves the original predicate" setup=[MutBaseImports] begin
    using LambdaRegression: TypeSystem
    using LambdaRegression.Mutations: _same_composition_type

    open = TypeSystem.newtyvar(0)
    types = TypeSystem.Ty[
        FloatTy,
        IntTy,
        TypeSystem.TyFun(FloatTy, FloatTy),
        TypeSystem.TyList(FloatTy),
        TypeSystem.TyArray(TypeSystem.TyNat(2), FloatTy),
        TypeSystem.TyProd(FloatTy, IntTy),
        TypeSystem.TyRigid(17),
        TypeSystem.TyFun(open, open),
    ]
    for left in types, right in types
        old = iscompatible(left, right) && iscompatible(right, left)
        @test _same_composition_type(left, right) == old
    end
end

@testitem "Mutations composition menus preserve scans and call-local reuse" setup=[MutBaseImports] begin
    using LambdaRegression: TypeSystem
    using LambdaRegression.Mutations:
        _prepare_composition_directory,
        _unary_composition_options,
        _binary_composition_options,
        _composition_options,
        UnaryCompositionMutation,
        BinaryCompositionMutation,
        AbstractReplacementGenerator,
        is_site_eligible,
        apply
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.TypeSystem:
        IntTy,
        TyArray,
        TyFun,
        TyList,
        TyMu,
        TyNat,
        TyScheme

    function reference_unary_options(site, ctx)
        LambdaRegression.Mutations._generation_closed_type(site.ty) &&
            haskey(ctx, :global_consts) && haskey(ctx, :global_consts_ty) ||
                return Symbol[]
        options = Symbol[]
        for name in sort!(collect(keys(ctx.global_consts_ty)))
            haskey(ctx.global_consts, name) || continue
            raw = ctx.global_consts_ty[name]
            raw isa TypeSystem.Ty || continue
            arrow = TypeSystem.prune(raw)
            arrow isa TyFun || continue
            argument_ty = TypeSystem.prune(arrow.arg)
            result_ty = TypeSystem.prune(arrow.res)
            argument_ty isa TypeSystem.Ty && result_ty isa TypeSystem.Ty || continue
            TypeSystem._grounded_iscompatible(argument_ty, site.ty) &&
                TypeSystem._grounded_iscompatible(site.ty, argument_ty) || continue
            TypeSystem._grounded_iscompatible(result_ty, site.ty) &&
                TypeSystem._grounded_iscompatible(site.ty, result_ty) || continue
            push!(options, name)
        end
        return options
    end

    function reference_binary_options(site, ctx)
        LambdaRegression.Mutations._generation_closed_type(site.ty) &&
            haskey(ctx, :global_consts) && haskey(ctx, :global_consts_ty) ||
                return NamedTuple[]
        options = NamedTuple[]
        for name in sort!(collect(keys(ctx.global_consts_ty)))
            haskey(ctx.global_consts, name) || continue
            raw = ctx.global_consts_ty[name]
            raw isa TypeSystem.Ty || continue
            first_arrow = TypeSystem.prune(raw)
            first_arrow isa TyFun || continue
            second_arrow = TypeSystem.prune(first_arrow.res)
            second_arrow isa TyFun || continue
            result_ty = TypeSystem.prune(second_arrow.res)
            result_ty isa TypeSystem.Ty || continue
            same(left, right) =
                TypeSystem._grounded_iscompatible(left, right) &&
                TypeSystem._grounded_iscompatible(right, left)
            same(result_ty, site.ty) || continue
            left_ty = TypeSystem.prune(first_arrow.arg)
            right_ty = TypeSystem.prune(second_arrow.arg)
            if left_ty isa TypeSystem.Ty &&
                    LambdaRegression.Mutations._generation_closed_type(right_ty) &&
                    same(left_ty, site.ty)
                push!(options, (; name, sibling_ty=right_ty, seed_first=true))
            end
            if right_ty isa TypeSystem.Ty &&
                    LambdaRegression.Mutations._generation_closed_type(left_ty) &&
                    same(right_ty, site.ty)
                push!(options, (; name, sibling_ty=left_ty, seed_first=false))
            end
        end
        return options
    end

    list_ty = TyList(FloatTy)
    array_ty = TyArray(TyNat(2), FloatTy)
    function_ty = TyFun(FloatTy, FloatTy)
    open_shared = TypeSystem.newtyvar(0)
    open_result = TypeSystem.newtyvar(0)
    scheme_var = TypeSystem.newtyvar(0)
    global_consts_ty = Dict{Symbol,TyLike}(
        :U_ARRAY => TyFun(array_ty, array_ty),
        :U_FLOAT => TyFun(FloatTy, FloatTy),
        :U_FUNCTION => TyFun(function_ty, function_ty),
        :U_INT => TyFun(IntTy, IntTy),
        :U_LIST => TyFun(list_ty, list_ty),
        :U_OPEN_END => TyFun(open_shared, open_shared),
        :B_ARRAY => TyFun(array_ty, TyFun(array_ty, array_ty)),
        :B_BOTH => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :B_FLOAT_LEFT => TyFun(FloatTy, TyFun(IntTy, FloatTy)),
        :B_FLOAT_RIGHT => TyFun(IntTy, TyFun(FloatTy, FloatTy)),
        :B_FUNCTION => TyFun(function_ty, TyFun(function_ty, function_ty)),
        :B_INT => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :B_LIST => TyFun(list_ty, TyFun(list_ty, list_ty)),
        :B_OPEN_RESULT => TyFun(IntTy, TyFun(FloatTy, open_result)),
        :B_SHARED => TyFun(open_shared, TyFun(FloatTy, open_shared)),
        :B_SHARED_SIBLING => TyFun(open_shared, TyFun(open_shared, FloatTy)),
        :MISSING_RUNTIME => TyFun(FloatTy, FloatTy),
        :POLYMORPHIC => TyScheme(
            [scheme_var],
            TyFun(scheme_var, scheme_var),
        ),
        :NOT_A_FUNCTION => FloatTy,
    )
    global_consts = Dict{Symbol,Any}(
        name => name for name in keys(global_consts_ty)
        if name !== :MISSING_RUNTIME
    )
    ctx = (; global_consts, global_consts_ty)
    make_site(ty) = LambdaRegression.Mutations.Site(
        Int[],
        ConstNode(:seed),
        nothing,
        0,
        0,
        :const_global,
        ty,
        TyLike[],
    )

    original_type_refs = Dict(name => ty for (name, ty) in global_consts_ty)
    for target in (FloatTy, IntTy, list_ty, array_ty, function_ty)
        site = make_site(target)
        actual_unary = _unary_composition_options(site, ctx)
        legacy_unary = reference_unary_options(site, ctx)
        @test actual_unary == legacy_unary

        actual_binary = [
            (; name=option.name, sibling_ty=option.sibling_ty, seed_first=option.seed_first)
            for option in _binary_composition_options(site, ctx)
        ]
        @test actual_binary == reference_binary_options(site, ctx)
    end

    float_site = make_site(FloatTy)
    float_binary = _binary_composition_options(float_site, ctx)
    @test [(option.name, option.seed_first) for option in float_binary if option.name === :B_BOTH] ==
          [(:B_BOTH, true), (:B_BOTH, false)]
    @test [(option.name, option.seed_first) for option in float_binary if option.name === :B_SHARED] ==
          [(:B_SHARED, true)]
    @test isempty([option for option in float_binary if option.name === :B_SHARED_SIBLING])
    @test :U_OPEN_END in _unary_composition_options(float_site, ctx)
    @test :MISSING_RUNTIME ∉ _unary_composition_options(float_site, ctx)
    @test :POLYMORPHIC ∉ _unary_composition_options(float_site, ctx)

    recursive_var = TypeSystem.newtyvar(0)
    recursive_ty = TyMu(recursive_var, TyFun(FloatTy, recursive_var))
    @test _composition_options(:unary, make_site(recursive_ty), ctx) === nothing

    directory = _prepare_composition_directory(global_consts, global_consts_ty)
    prepared_ctx = (; global_consts, global_consts_ty, composition_directory=directory)
    prepared_site = make_site(FloatTy)
    initial_menu = _unary_composition_options(prepared_site, prepared_ctx)
    @test initial_menu === _unary_composition_options(prepared_site, prepared_ctx)
    for target in (list_ty, array_ty, function_ty)
        original_menu = _unary_composition_options(make_site(target), prepared_ctx)
        copied_menu = _unary_composition_options(make_site(deepcopy(target)), prepared_ctx)
        @test !isempty(original_menu)
        @test copied_menu === original_menu
    end

    tree = AstTree(ConstNode(:seed))
    rng = MersenneTwister(2000)
    env = LambdaRegression.Mutations.MutationEnv(
        rng,
        FloatTy,
        Dict{Symbol,TyLike}(),
        tree,
        nothing,
        false,
        IdDict{Any,Any}(),
        LambdaRegression.Mutations.AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        LambdaRegression.Mutations.BloatPolicy(),
    )
    mutation = UnaryCompositionMutation()
    @test is_site_eligible(mutation, prepared_site, env, prepared_ctx)
    @test initial_menu === _unary_composition_options(prepared_site, prepared_ctx)
    attempt_ctx = (; global_consts, global_consts_ty, composition_directory=directory)
    @test apply(mutation, tree, prepared_site, env, attempt_ctx) === tree
    @test initial_menu === _unary_composition_options(prepared_site, attempt_ctx)

    struct PreparedSiblingGenerator <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_replacement(
        ::PreparedSiblingGenerator,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(ConstNode(:SIBLING)),
            site.ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    binary_mutation = BinaryCompositionMutation(
        replacement_generator=PreparedSiblingGenerator(),
    )
    binary_site = make_site(FloatTy)
    initial_binary_menu = _binary_composition_options(binary_site, prepared_ctx)
    @test is_site_eligible(binary_mutation, binary_site, env, prepared_ctx)
    @test initial_binary_menu === _binary_composition_options(binary_site, prepared_ctx)
    binary_tree = AstTree(ConstNode(:seed))
    binary_local_consts = Dict{Symbol,Any}()
    binary_local_types = Dict{Symbol,TyLike}()
    binary_attempt_ctx = (
        ; global_consts,
        global_consts_ty,
        local_consts=binary_local_consts,
        local_consts_ty=binary_local_types,
        composition_directory=directory,
    )
    @test apply(binary_mutation, binary_tree, binary_site, env, binary_attempt_ctx) === binary_tree
    @test initial_binary_menu === _binary_composition_options(binary_site, binary_attempt_ctx)

    second_directory = _prepare_composition_directory(global_consts, global_consts_ty)
    second_ctx = (; global_consts, global_consts_ty, composition_directory=second_directory)
    second_menu = _unary_composition_options(prepared_site, second_ctx)
    @test second_directory !== directory
    @test second_menu !== initial_menu

    @test Set(keys(global_consts_ty)) == Set(keys(global_consts)) ∪ Set([:MISSING_RUNTIME])
    @test global_consts[:U_OPEN_END] === :U_OPEN_END
    @test all(global_consts_ty[name] === ty for (name, ty) in original_type_refs)
    @test open_shared.instance === nothing
    @test open_result.instance === nothing
    @test scheme_var.instance === nothing
end

@testitem "Mutations UnaryCompositionMutation wraps a typed endomorphism" setup=[MutBaseImports] begin
    using LambdaRegression.Mutations: PathSiteSelectionPolicy, UnaryCompositionMutation
    using LambdaRegression.TypeSystem: get_type_ctx
    using LambdaRegression.Primitives: SQR

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
    using LambdaRegression.TypeSystem: get_type_ctx
    using LambdaRegression.Primitives: FADD, SQR

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
    using LambdaRegression: CanonicalGeneration, TypeSystem
    using LambdaRegression.Mutations:
        BinaryCompositionMutation,
        PathSiteSelectionPolicy,
        ScopedReplacementGenerator
    using LambdaRegression.TypeSystem:
        TyList,
        get_type_ctx
    using LambdaRegression.Primitives: FADD, UNCONS_CASE

    on_item = AbsNode(
        :head,
        AbsNode(:tail, AppNode(IndexNode(3), IndexNode(0))),
    )
    fallback = AppNode(
        AppNode(ConstNode(:FADD), ConstNode(:ONE)),
        ConstNode(:ONE),
    )
    tree = AstTree(FunctionDefNode(AbsNode(
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
            CanonicalGeneration.generate_canonical_scoped;
            include_local_consts=true,
        ),
        generation_kwargs=(;
            providers=(
                CanonicalGeneration.default_head_providers()...,
                CanonicalGeneration.SafeListAccessProvider(),
                CanonicalGeneration.FunctionDefProvider(),
            ),
            sampling_depth_limit=3,
            sampling_node_budget=24,
            guarded_expansion_depth=1,
        ),
    )
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
    )

    @test mutated isa AstTree
    inferred = TypeSystem.infer_term(mutated, global_consts_ty)
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
