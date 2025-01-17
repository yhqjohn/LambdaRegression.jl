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
"""
module Mutations

using Random
using TestItems

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike, TyConst, TyFun, TyArray, TyNat, instantiate, infer, iscompatible, newtyvar, BoolTy, IntTy, FloatTy
using ..ASTs.NodeASTs
using ..ASTs.CSE: canonicalize, eliminate_expression!
import ..ASTs.NodeASTs: apply_node!, shift_free_variables!
using ..ASTs.Contexts: new_constant!, ConstSymbolFactory
using ..Trees: traverse, PointerState, StackState, getstate, Break, setroot!, setchild!, children

export mutate_tree
export AbstractMutation, required_site_kinds, is_site_eligible, apply
export AbstractSiteProvider, IndexedProvider
export AbstractPairChooser, WeightedPairChooser
export BloatPolicy, AdaptivePolicy
export MutationAttemptRecorder, mutation_recorder_summary
export prune_local_consts!
export ConstMutation, ScalarConstMutation, BoolConstMutation, ArrayConstMutation
export PointMutation
export BetaReduceMutation, EtaReduceMutation, EtaExpandMutation, CSEMutation
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
end
AdaptivePolicy(; enabled::Bool=false, ewma_alpha::Float64=0.2, floor::Float64=0.05) = AdaptivePolicy(enabled, ewma_alpha, floor)

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
end
BloatPolicy(; enabled::Bool=false, max_nodes::Int=200, max_nodes_delta::Int=40) = BloatPolicy(enabled, max_nodes, max_nodes_delta)

mutable struct MutationStat
    ewma_success::Float64
    initialized::Bool
end
MutationStat() = MutationStat(0.0, false)

mutable struct MutationEnv
    rng::AbstractRNG
    target_ty::Ty
    type_ctx::Dict{Symbol,TyLike}
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
    return (; totals, per_mutation=per)
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
        w > 0 || continue
        total += w
    end
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
        abs_ty = meta[abs][1]
        abs_ty isa TyFun || continue
        push!(tys, abs_ty.arg)
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

function _collect_sites(tree::AstTree, meta::Dict{AstNode,Tuple{TyLike,TyLike}}, ctx::NamedTuple)
    sites = Site[]
    traverse(tree, PointerState{AstNode}, ScopeState) do t, node, bag
        pointer = getstate(bag, PointerState{AstNode})
        stack = getstate(bag, StackState{AstNode})
        scope = getstate(bag, ScopeState)
        haskey(meta, node) || return nothing
        node_ty = meta[node][1]
        binder_types = _binder_types(scope, meta)
        kind = _site_kind(node, ctx)
        path =
            if length(stack.stack) <= 1
                Int[]
            else
                [stack.stack[i][2] for i in 1:(length(stack.stack)-1)]
            end
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
    ty = infer(tree, type_ctx, factory, meta)
    ty isa Err && return ty
    env.meta = meta
    p.sites_all = _collect_sites(tree, meta, ctx)
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
   `mass(m) = weight(typeof(m)) * sum(site_weights) * success_hint(m)`
3) Sample a site within that mutation by its site weights.

## Parameters
- `weights::Dict{DataType,Float64}`: per-mutation-type weights. Defaults:
- `ScalarConstMutation => 0.25`
- `ArrayConstMutation => 0.25`
- `BoolConstMutation => 0.05`
- `PointMutation => 0.10`
- `CSEMutation => 0.08`
- `BetaReduceMutation => 0.04`
- `EtaExpandMutation => 0.03`
- `EtaReduceMutation => 0.03`
- `cache_eligibility::Bool=false`: cache eligible site lists per mutation within
  one `mutate_tree` call (Milestone C).

Notes
- Uniform selection is intentionally not the default because SR mutation operators
  have wildly different success rates and usefulness (constants often dominate).
"""
struct WeightedPairChooser <: AbstractPairChooser
    weights::Dict{DataType,Float64}
    cache_eligibility::Bool
end

function WeightedPairChooser(; weights::Union{Nothing,Dict{DataType,Float64}}=nothing, cache_eligibility::Bool=false)
	    default = Dict{DataType,Float64}(
	        # Split constant operators (preferred).
	        ScalarConstMutation => 0.25,
	        ArrayConstMutation => 0.25,
	        BoolConstMutation => 0.05,
        # Backwards-compatible wrapper.
        ConstMutation => 0.55,

        PointMutation => 0.10,
	        CSEMutation => 0.08,
	        BetaReduceMutation => 0.04,
	        EtaExpandMutation => 0.03,
	        EtaReduceMutation => 0.03,
	    )
	    return WeightedPairChooser(weights === nothing ? default : weights, cache_eligibility)
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
        w > 0 || continue
        push!(eligible, s)
        push!(weights, w)
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

    masses = Float64[]
    payloads = Any[]
    for m in mutations
        (eligible, wts) = _eligible_sites(m, provider, env, ctx)
        if isempty(eligible)
            push!(masses, 0.0)
            push!(payloads, (m, eligible, wts))
            continue
        end
        mass = get(chooser.weights, typeof(m), base_weight(m, ctx)) * sum(wts) * _success_hint(env, m)
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

function mutate_const_value(rng::AbstractRNG, value::AbstractFloat, ty::Ty, σ::Float64)
    ty == FloatTy || return Err(UnsupportedConstTyErr(ty))
    mode = rand(rng, 1:2)
    if mode == 1
        return value + convert(typeof(value), randn(rng) * σ)
    else
        return value * (one(typeof(value)) + convert(typeof(value), randn(rng) * σ))
    end
end

function mutate_const_value(rng::AbstractRNG, value::Integer, ty::Ty, ::Float64)
    ty == IntTy || return Err(UnsupportedConstTyErr(ty))
    return value + convert(typeof(value), rand(rng, (-1, 1)))
end

function mutate_const_value(::AbstractRNG, value::Bool, ty::Ty, ::Float64)
    ty == BoolTy || return Err(UnsupportedConstTyErr(ty))
    return !value
end

function mutate_const_value(rng::AbstractRNG, value::Complex, ty::Ty, σ::Float64)
    (ty isa TyConst && (ty::TyConst).name == :Complex) || return Err(UnsupportedConstTyErr(ty))
    T = typeof(value)
    δre = convert(real(T), randn(rng) * σ)
    δim = convert(real(T), randn(rng) * σ)
    return value + complex(δre, δim)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, σ::Float64) where {T<:AbstractFloat}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    (ty::TyArray).elem == FloatTy || return Err(UnsupportedConstTyErr(ty))
    noise = randn(rng, T, size(value)...)
    return value .+ noise .* convert(T, σ)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractMatrix{T}, ty::Ty, σ::Float64) where {T<:AbstractFloat}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    if (ty::TyArray).elem == FloatTy
        noise = randn(rng, T, size(value)...)
        return value .+ noise .* convert(T, σ)
    end
    (ty::TyArray).elem isa TyArray || return Err(UnsupportedConstTyErr(ty))
    inner = (ty.elem::TyArray)
    inner.elem == FloatTy || return Err(UnsupportedConstTyErr(ty))
    noise = randn(rng, T, size(value)...)
    return value .+ noise .* convert(T, σ)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractArray{T}, ty::Ty, ::Float64) where {T<:Integer}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    (ty::TyArray).elem == IntTy || return Err(UnsupportedConstTyErr(ty))
    delta = rand(rng, (-1, 1), size(value)...)
    return value .+ convert.(T, delta)
end

function mutate_const_value(rng::AbstractRNG, value::AbstractMatrix{T}, ty::Ty, ::Float64) where {T<:Integer}
    ty isa TyArray || return Err(UnsupportedConstTyErr(ty))
    if (ty::TyArray).elem == IntTy
        delta = rand(rng, (-1, 1), size(value)...)
        return value .+ convert.(T, delta)
    end
    (ty::TyArray).elem isa TyArray || return Err(UnsupportedConstTyErr(ty))
    inner = (ty.elem::TyArray)
    inner.elem == IntTy || return Err(UnsupportedConstTyErr(ty))
    delta = rand(rng, (-1, 1), size(value)...)
    return value .+ convert.(T, delta)
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
end
ScalarConstMutation(; σ::Float64=0.1) = ScalarConstMutation(σ)
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
    new_value = mutate_const_value(env.rng, value, site.ty, m.σ)
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
end
ArrayConstMutation(; σ::Float64=0.1) = ArrayConstMutation(σ)
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
    new_value = mutate_const_value(env.rng, value, site.ty, m.σ)
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

This can increase tree size; it is *not* enabled by default in `_default_mutations()`.

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
and delegates to `ASTs.CSE.eliminate_expression!` which rewrites the tree by
introducing a beta-redex around the LCA of matches.

When `min_count=1`, this acts as a "zero-cost" structural move: it can extract a
single occurrence to increase structural diversity without changing semantics.
"""
struct CSEMutation <: AbstractMutation
    min_count::Int
end
CSEMutation(; min_count::Int=1) = CSEMutation(min_count)
required_site_kinds(::CSEMutation) = Symbol[:any]

function is_site_eligible(::CSEMutation, site::Site, ::MutationEnv, ::NamedTuple)
    return site.node isa AppNode
end

function apply(m::CSEMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    form = canonicalize(tree, site.node)
    form isa Err && return form
    return eliminate_expression!(tree, form; min_count=m.min_count, factory=ctx.symfactory)
end

function _count_nodes(tree::AstTree)
    c = 0
    for _ in tree
        c += 1
    end
    return c
end

function _default_mutations()
    return AbstractMutation[
        ScalarConstMutation(),
        BoolConstMutation(),
        ArrayConstMutation(),
        PointMutation(),
        CSEMutation(),
        BetaReduceMutation(),
        EtaExpandMutation(),
        EtaReduceMutation(),
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
- `const_symfactory`: per-call deterministic local-constant symbol factory (default: `ConstSymbolFactory()`)
- `tyfactory`, `symfactory`: local factories used by type inference and CSE

## Optional policies (off by default)
- `bloat_policy::BloatPolicy`
- `adaptive_policy::AdaptivePolicy`
"""
function mutate_tree(tree::AstTree, target_ty::Ty;
    rng::AbstractRNG=MersenneTwister(2000),
    mutations::Vector{<:AbstractMutation}=_default_mutations(),
    chooser::AbstractPairChooser=WeightedPairChooser(),
    provider::AbstractSiteProvider=IndexedProvider(),
    max_try::Int=20,
    copy_tree::Bool=true,
    copy_ctx::Bool=true,
    prune_consts::Bool=true,
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
        global_consts_ty=global_consts_ty,
        local_consts_ty=current_local_consts_ty,
        local_consts=current_local_consts,
        tyfactory=tyfactory,
        symfactory=symfactory,
        const_symfactory=const_symfactory,
    )
    env = MutationEnv(
        rng,
        target_ty,
        base_type_ctx,
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
            global_consts_ty=global_consts_ty,
            local_consts_ty=attempt_local_consts_ty,
            local_consts=attempt_local_consts,
            tyfactory=tyfactory,
            symfactory=symfactory,
            const_symfactory=const_symfactory,
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

        prune_consts && prune_local_consts!(attempt_tree, attempt_local_consts, attempt_local_consts_ty)
        attempt_type_ctx = merge(global_consts_ty, attempt_local_consts_ty)
        inferred = infer(attempt_tree, attempt_type_ctx, tyfactory)
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

include("Population.jl")

########################################################
# Test Items
########################################################

@testsnippet MutBaseImports begin
    using LambdaRegression.Mutations: prune_local_consts!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using Random
    using LambdaRegression.Mutations: mutate_tree, ScalarConstMutation, WeightedPairChooser, IndexedProvider
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
    inferred = infer(mutated, global_consts_ty)
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

    rec = MutationAttemptRecorder()
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
end

end # module Mutations
