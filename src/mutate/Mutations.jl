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

Quickstart
```julia
using Random
using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
using LambdaRegression.TypeSystem: TyFun, IntTy
using LambdaRegression.GenerateTree: IndexCtor, UniformChooser
using LambdaRegression.Mutations: mutate_tree, ConstMutation, PointMutation, SubtreeReplaceMutation
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
    gen_ctors=[IndexCtor()],
    gen_chooser=UniformChooser(),
    mutations=[SubtreeReplaceMutation(), PointMutation(), ConstMutation()],
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
using ..ASTs.Contexts: new_constant!
using ..Trees: traverse, PointerState, getstate, Break, setroot!, setchild!
using ..GenerateTree

export mutate_tree
export AbstractMutation, required_site_kinds, is_site_eligible, apply
export AbstractSiteProvider, OnDemandProvider, IndexedProvider
export AbstractPairChooser, WeightedPairChooser
export BloatPolicy, AdaptivePolicy
export prune_local_consts!
export ConstMutation, PointMutation, SubtreeReplaceMutation

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
"""
struct Site
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

abstract type AbstractMutation end

required_site_kinds(::AbstractMutation) = Symbol[:any]
is_site_eligible(::AbstractMutation, ::Site, ::Any, ::NamedTuple) = true
apply(::AbstractMutation, ::AstTree, ::Site, ::Any, ::NamedTuple) = error("apply not implemented")

base_weight(::AbstractMutation, ::NamedTuple) = 1.0
site_weight(::AbstractMutation, ::Site, ::Any, ::NamedTuple) = 1.0

abstract type AbstractSiteProvider end

reset!(::AbstractSiteProvider) = nothing
prepare!(::AbstractSiteProvider, ::AstTree, ::Dict{Symbol,<:TyLike}, ::Any, ::NamedTuple) = error("prepare! not implemented")
sites(::AbstractSiteProvider) = error("sites not implemented")
sites_by_kind(::AbstractSiteProvider, ::Symbol) = Site[]

abstract type AbstractPairChooser end
choose_pair(::AbstractPairChooser, ::Vector{<:AbstractMutation}, ::AbstractSiteProvider, ::AstTree, ::Dict{Symbol,<:TyLike}, ::Any, ::NamedTuple) = error("choose_pair not implemented")

mutable struct AdaptivePolicy
    enabled::Bool
    ewma_alpha::Float64
    floor::Float64
end
AdaptivePolicy(; enabled::Bool=false, ewma_alpha::Float64=0.2, floor::Float64=0.05) = AdaptivePolicy(enabled, ewma_alpha, floor)

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

function _collect_sites(tree::AstTree, meta::Dict{AstNode,Tuple{TyLike,TyLike}}, ctx::NamedTuple)
    sites = Site[]
    traverse(tree, PointerState{AstNode}, ScopeState) do t, node, bag
        pointer = getstate(bag, PointerState{AstNode})
        scope = getstate(bag, ScopeState)
        node_ty = meta[node][1]
        binder_types = _binder_types(scope, meta)
        kind = _site_kind(node, ctx)
        push!(sites, Site(node, pointer.parent, pointer.child_index, length(binder_types), kind, node_ty, binder_types))
        return nothing
    end
    return sites
end

mutable struct OnDemandProvider <: AbstractSiteProvider
    sites_all::Vector{Site}
    sites_kind::Dict{Symbol,Vector{Site}}
end
OnDemandProvider() = OnDemandProvider(Site[], Dict{Symbol,Vector{Site}}())

function reset!(p::OnDemandProvider)
    empty!(p.sites_all)
    empty!(p.sites_kind)
    return p
end

function prepare!(p::OnDemandProvider, tree::AstTree, type_ctx::Dict{Symbol,<:TyLike}, env::MutationEnv, ctx::NamedTuple)
    reset!(p)
    meta = Dict{AstNode,Tuple{TyLike,TyLike}}()
    ty = infer(tree, type_ctx, newtyvar, meta)
    ty isa Err && return ty
    env.meta = meta
    p.sites_all = _collect_sites(tree, meta, ctx)
    for s in p.sites_all
        push!(get!(p.sites_kind, :any, Site[]), s)
        s.kind === :any || push!(get!(p.sites_kind, s.kind, Site[]), s)
    end
    return nothing
end

sites(p::OnDemandProvider) = p.sites_all
sites_by_kind(p::OnDemandProvider, kind::Symbol) = get(p.sites_kind, kind, Site[])

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
    ty = infer(tree, type_ctx, newtyvar, meta)
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

struct WeightedPairChooser <: AbstractPairChooser
    weights::Dict{DataType,Float64}
    cache_eligibility::Bool
end

function WeightedPairChooser(; weights::Union{Nothing,Dict{DataType,Float64}}=nothing, cache_eligibility::Bool=false)
    default = Dict{DataType,Float64}(
        ConstMutation => 0.55,
        SubtreeReplaceMutation => 0.25,
        PointMutation => 0.20,
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
    env.eligibility_cache = IdDict{Any,Any}()

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

_is_float_ty(t::Ty) = t == FloatTy
_is_int_ty(t::Ty) = t == IntTy
_is_bool_ty(t::Ty) = t == BoolTy

struct ConstMutation <: AbstractMutation
    σ::Float64
end
ConstMutation(; σ::Float64=0.1) = ConstMutation(σ)
required_site_kinds(::ConstMutation) = Symbol[:const_local]

function _mutate_value(rng::AbstractRNG, value, ty::Ty, σ::Float64)
    if _is_float_ty(ty)
        v = value
        mode = rand(rng, 1:2)
        if mode == 1
            noise = convert(typeof(v), randn(rng) * σ)
            return v + noise
        else
            noise = convert(typeof(v), randn(rng) * σ)
            return v * (one(typeof(v)) + noise)
        end
    elseif _is_int_ty(ty)
        return value + rand(rng, (-1, 1))
    elseif ty isa TyArray && _is_float_ty(ty.elem)
        a = value
        a2 = copy(a)
        for i in eachindex(a2)
            a2[i] = a2[i] + convert(typeof(a2[i]), randn(rng) * σ)
        end
        return a2
    elseif ty isa TyArray && _is_int_ty(ty.elem)
        a = value
        a2 = copy(a)
        for i in eachindex(a2)
            a2[i] = a2[i] + rand(rng, (-1, 1))
        end
        return a2
    elseif ty isa TyArray && ty.elem isa TyArray && _is_float_ty((ty.elem::TyArray).elem)
        a = value
        a2 = [copy(row) for row in a]
        for row in a2
            for i in eachindex(row)
                row[i] = row[i] + convert(typeof(row[i]), randn(rng) * σ)
            end
        end
        return a2
    else
        return Err(UnsupportedConstTyErr(ty))
    end
end

function is_site_eligible(m::ConstMutation, site::Site, env::MutationEnv, ctx::NamedTuple)
    site.node isa ConstNode || return false
    _is_local_const(site.node.name, ctx) || return false
    return true
end

function apply(m::ConstMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    node = site.node::ConstNode
    name = node.name
    _is_local_const(name, ctx) || return Err(NoEligibleSiteErr())
    value = ctx.local_consts[name]
    ty = site.ty
    new_value = _mutate_value(env.rng, value, ty, m.σ)
    new_value isa Err && return new_value
    new_node = new_constant!(ctx.local_consts, new_value)
    ctx.local_consts_ty[new_node.name] = ty
    _replace_site!(tree, site, AstTree(new_node))
    return tree
end

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

struct SubtreeReplaceMutation <: AbstractMutation end
required_site_kinds(::SubtreeReplaceMutation) = Symbol[:any]

is_site_eligible(::SubtreeReplaceMutation, ::Site, ::MutationEnv, ::NamedTuple) = true

function _mk_scope_ctx(site::Site)
    depth = length(site.binder_types)
    scope_stack = Symbol[]
    var_ty = Dict{Symbol,Ty}()
    for i in 1:depth
        name = Symbol("b$(i)")
        push!(scope_stack, name)
        var_ty[name] = site.binder_types[i]
    end
    return (scope_stack, var_ty)
end

function _count_nodes(tree::AstTree)
    c = 0
    for _ in tree
        c += 1
    end
    return c
end

function apply(::SubtreeReplaceMutation, tree::AstTree, site::Site, env::MutationEnv, ctx::NamedTuple)
    if env.bloat_policy.enabled
        old_n = _count_nodes(tree)
        old_n <= env.bloat_policy.max_nodes || return Err(BloatExceededErr(env.bloat_policy.max_nodes, env.bloat_policy.max_nodes_delta))
    end
    scope_stack, var_ty = _mk_scope_ctx(site)
    target_ty = site.ty
    gen = GenerateTree.generate_tree(
        target_ty;
        ctors=ctx.gen_ctors,
        chooser=ctx.gen_chooser,
        max_depth=ctx.gen_max_depth,
        max_try=ctx.gen_max_try,
        tyfactory=ctx.tyfactory,
        symfactory=ctx.symfactory,
        rng=env.rng,
        global_consts_ty=ctx.global_consts_ty,
        local_consts_ty=ctx.local_consts_ty,
        local_consts=ctx.local_consts,
        var_ty=var_ty,
        scope_stack=scope_stack,
    )
    gen isa Err && return gen

    old_n = env.bloat_policy.enabled ? _count_nodes(tree) : 0
    _replace_site!(tree, site, gen)
    if env.bloat_policy.enabled
        new_n = _count_nodes(tree)
        (new_n <= env.bloat_policy.max_nodes && (new_n - old_n) <= env.bloat_policy.max_nodes_delta) || return Err(BloatExceededErr(env.bloat_policy.max_nodes, env.bloat_policy.max_nodes_delta))
    end
    return tree
end

function _default_mutations()
    return AbstractMutation[ConstMutation(), SubtreeReplaceMutation(), PointMutation()]
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
    gen_ctors::Vector{<:GenerateTree.AbstractCtor}=GenerateTree.AbstractCtor[],
    gen_chooser::GenerateTree.AbstractChooser=GenerateTree.UniformChooser(),
    gen_max_depth::Int=8,
    gen_max_try::Int=12,
    tyfactory=newtyvar,
    symfactory=SymbolGenerator(),
    bloat_policy::BloatPolicy=BloatPolicy(),
    adaptive_policy::AdaptivePolicy=AdaptivePolicy(),
)
    current_tree = copy_tree ? deepcopy(tree) : tree
    # Transactional: always search with scratch dicts; on success, commit to caller dicts.
    caller_local_consts = local_consts
    caller_local_consts_ty = local_consts_ty
    current_local_consts = copy(caller_local_consts)
    current_local_consts_ty = copy(caller_local_consts_ty)

    for _ in 1:max_try
        attempt_tree = deepcopy(current_tree)
        attempt_local_consts = copy(current_local_consts)
        attempt_local_consts_ty = copy(current_local_consts_ty)
        type_ctx = merge(global_consts_ty, attempt_local_consts_ty)
        attempt_ctx = (;
            rng=rng,
            global_consts_ty=global_consts_ty,
            local_consts_ty=attempt_local_consts_ty,
            local_consts=attempt_local_consts,
            gen_ctors=gen_ctors,
            gen_chooser=gen_chooser,
            gen_max_depth=gen_max_depth,
            gen_max_try=gen_max_try,
            tyfactory=tyfactory,
            symfactory=symfactory,
        )

        env = MutationEnv(
            rng,
            target_ty,
            type_ctx,
            nothing,
            false,
            IdDict{Any,Any}(),
            adaptive_policy,
            IdDict{DataType,MutationStat}(),
            bloat_policy,
        )

        reset!(provider)
        prep = prepare!(provider, attempt_tree, type_ctx, env, attempt_ctx)
        prep isa Err && return prep

        choice = choose_pair(chooser, mutations, provider, attempt_tree, type_ctx, env, attempt_ctx)
        if choice isa Err
            return choice
        end

        m = choice.mutation::AbstractMutation
        mutated = apply(m, attempt_tree, choice.site, env, attempt_ctx)
        if mutated isa Err
            _update_stat!(env, m, false)
            continue
        end

        prune_consts && prune_local_consts!(attempt_tree, attempt_local_consts, attempt_local_consts_ty)
        attempt_type_ctx = merge(global_consts_ty, attempt_local_consts_ty)
        inferred = infer(attempt_tree, attempt_type_ctx)
        if inferred isa Err
            _update_stat!(env, m, false)
            continue
        end
        if !iscompatible(inferred, target_ty)
            _update_stat!(env, m, false)
            continue
        end

        _update_stat!(env, m, true)
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
    return Err(MaxTryReachedErr())
end

########################################################
# Test Items
########################################################

@testitem "Mutations prune_local_consts! keeps only referenced" begin
    using LambdaRegression.Mutations: prune_local_consts!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike

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

@testitem "Mutations ConstMutation FloatTy single-node replace + type stable" begin
    using Random
    using LambdaRegression.Mutations: mutate_tree, ConstMutation, WeightedPairChooser, OnDemandProvider
    using LambdaRegression.GenerateTree
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike

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
        gen_ctors=GenerateTree.AbstractCtor[],
        mutations=[ConstMutation(σ=0.2)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ConstMutation=>1.0), cache_eligibility=false),
        provider=OnDemandProvider(),
        max_try=10,
        prune_consts=true,
    )
    @test mutated isa AstTree
    @test mutated.root isa ConstNode
    @test mutated.root.name != name
end

@testitem "Mutations PointMutation Index swap respects binder types" begin
    using Random
    using LambdaRegression.Mutations: mutate_tree, PointMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike
    using LambdaRegression.GenerateTree: IndexCtor, UniformChooser

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
        gen_ctors=[IndexCtor()],
        gen_chooser=UniformChooser(),
        mutations=[PointMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(PointMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=20,
    )
    @test mutated isa AstTree
    @test mutated.root.body.body isa IndexNode
    @test mutated.root.body.body.index in (0, 1)
end

@testitem "Mutations SubtreeReplaceMutation under lambda preserves target type" begin
    using Random
    using LambdaRegression.Mutations: mutate_tree, SubtreeReplaceMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike
    using LambdaRegression.GenerateTree: AbsCtor, IndexCtor, UniformChooser

    rng = MersenneTwister(1234)
    tree = AstTree(AbsNode(:x, IndexNode(0))) # identity
    target_ty = TyFun(IntTy, IntTy)
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        gen_ctors=[AbsCtor(), IndexCtor()],
        gen_chooser=UniformChooser(),
        gen_max_depth=3,
        gen_max_try=6,
        mutations=[SubtreeReplaceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplaceMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=10,
    )
    @test mutated isa AstTree
end

@testitem "Mutations integration: matrix->matrix type stability" begin
    using Random
    using LambdaRegression.Mutations: mutate_tree, SubtreeReplaceMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike, infer, iscompatible
    using LambdaRegression.GenerateTree: AbsCtor, IndexCtor, UniformChooser

    rng = MersenneTwister(8000)
    dom = TyArray(TyNat(3), TyArray(TyNat(4), FloatTy))
    target_ty = TyFun(dom, dom)
    tree = AstTree(AbsNode(:x, IndexNode(0))) # identity: dom -> dom
    mutated = mutate_tree(
        tree,
        target_ty;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        gen_ctors=[AbsCtor(), IndexCtor()],
        gen_chooser=UniformChooser(),
        mutations=[SubtreeReplaceMutation()],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(SubtreeReplaceMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=10,
    )
    inferred = infer(mutated, Dict{Symbol,TyLike}())
    @test iscompatible(inferred, target_ty)
end

@testitem "Mutations integration: matrix->float type stability" begin
    using Random
    using LambdaRegression.Mutations: mutate_tree, ConstMutation, WeightedPairChooser, IndexedProvider
    using LambdaRegression.GenerateTree
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, ConstNode
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyNat, FloatTy, TyLike, infer, iscompatible
    using LambdaRegression.GenerateTree: UniformChooser

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
        gen_ctors=GenerateTree.AbstractCtor[],
        gen_chooser=UniformChooser(),
        mutations=[ConstMutation(σ=0.3)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ConstMutation=>1.0), cache_eligibility=true),
        provider=IndexedProvider(),
        max_try=10,
        prune_consts=true,
        copy_ctx=true,
    )
    inferred = infer(mutated, merge(Dict{Symbol,TyLike}(), local_consts_ty))
    @test iscompatible(inferred, target_ty)
end

end # module Mutations
