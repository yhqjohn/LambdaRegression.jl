"""
    Mutations.Populations typed crossover implementations

This file contains **concrete** crossover operators for the population-level
variation layer.

Threading / determinism requirements:
- No global RNG usage: callers must pass `rng`.
- No global type-variable factories: use `tyfactory_ctor()` per call.
- No `gensym` in SR paths: local-constant remapping uses `ConstSymbolFactory`.
"""

using Random
using TestItems

using ...Utils: Err
using ...TypeSystem:
    Ty, TyLike, TyRigid, TyScheme, _inst, collect_free_vars,
    iscompatible, isequal_mu, make_tyvar_factory, semantic_typekey
using ...Primitives
using ...Programs: Program, _publish_program
using ...Programs: ConstSymbolFactory
using ...ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode,
    VarNode, children
using ...Trees: traverse, PointerState, getstate, setroot!, setchild!

export AbstractCrossoverSelectionPolicy, UniformCrossoverSelectionPolicy
export CrossoverSite, PathCrossoverSelectionPolicy
export crossover_site_weight, crossover_pair_weight
export crossover_site_path, crossover_site_depth, crossover_site_type
export crossover_site_binder_types, crossover_site_capture_levels
export crossover_site_offer
export TypedSubtreeSwapCrossover, CrossoverPairPoolExhaustedErr
export NoEligibleCrossoverPairErr
export CrossoverAnswerBudgetExhaustedErr
export recombine

struct TargetTypeMismatchErr
    a::Ty
    b::Ty
end
struct MissingGlobalTypeCtxErr end
struct NoEligibleCrossoverSiteErr end
struct NoEligibleCrossoverPairErr end
struct CrossoverPairPoolExhaustedErr
    attempts::Int
end
struct CrossoverTypeInvariantErr
    expected::Ty
    got::Ty
end
struct CrossoverMaxTryReachedErr
    max_try::Int
end
struct CrossoverAnswerBudgetExhaustedErr
    limit::Int
    function CrossoverAnswerBudgetExhaustedErr(limit::Int)
        limit >= 1 || throw(ArgumentError("answer budget must be positive"))
        return new(limit)
    end
end
"""
    CrossoverSite

Lightweight routing reference `(analysis,id)` for one typed subtree. Paths,
depth, demand, offer, binder schemes, and captured De Bruijn levels are
queried explicitly and are not cached as a second analysis.
"""
struct CrossoverSite
    analysis::TypedAnalysis
    id::Int
end

crossover_site_path(site::CrossoverSite) = collect(Int, typed_analysis_path(site.analysis, site.id))
function crossover_site_depth(site::CrossoverSite)
    head = site.analysis.index.env_head[site.id]
    return head == 0 ? 0 : site.analysis.index.bindings[head].depth
end

function crossover_site_type(site::CrossoverSite)::Ty
    demand = site.analysis.metadata[site.id][1]
    return demand isa TyScheme ? demand.ty : demand
end

function crossover_site_offer(site::CrossoverSite)::TyLike
    return site.analysis.metadata[site.id][2]
end

function crossover_site_binder_types(site::CrossoverSite)
    ids = visible_binder_ids(site.analysis, site.id)
    return Union{Nothing,TyLike}[
        get(site.analysis.binder_schemes, id, nothing)
        for id in reverse(ids)
    ]
end

function crossover_site_capture_levels(site::CrossoverSite)
    depth = crossover_site_depth(site)
    levels = Int[]
    for binder in captured_binder_ids(site.analysis, site.id)
        push!(levels, depth - site.analysis.index.bindings[binder].depth)
    end
    return sort!(unique!(levels))
end

"""
Policy hook for weighting crossover sites and compatible site pairs.

Subtype it and overload `crossover_site_weight` and/or
`crossover_pair_weight`. Callbacks must terminate, treat sites as read-only, and
return finite nonnegative numbers.
"""
abstract type AbstractCrossoverSelectionPolicy end

"""Give every eligible crossover site and pair equal mass."""
struct UniformCrossoverSelectionPolicy <: AbstractCrossoverSelectionPolicy end

"""Select one exact left/right path pair, primarily for deterministic routing."""
struct PathCrossoverSelectionPolicy <: AbstractCrossoverSelectionPolicy
    left_path::Vector{Int}
    right_path::Vector{Int}
end

PathCrossoverSelectionPolicy(
    left_path::AbstractVector{<:Integer},
    right_path::AbstractVector{<:Integer},
) =
    PathCrossoverSelectionPolicy(collect(Int, left_path), collect(Int, right_path))

"""Return the nonnegative sampling weight of `site` on `side`."""
crossover_site_weight(
    ::AbstractCrossoverSelectionPolicy,
    ::CrossoverSite,
    ::Symbol,
) = 1.0

crossover_site_weight(
    policy::PathCrossoverSelectionPolicy,
    site::CrossoverSite,
    side::Symbol,
) = if side === :left
    crossover_site_path(site) == policy.left_path ? 1.0 : 0.0
elseif side === :right
    crossover_site_path(site) == policy.right_path ? 1.0 : 0.0
else
    throw(ArgumentError("crossover side must be :left or :right"))
end

function _crossover_site_weights(
    policy::PathCrossoverSelectionPolicy,
    sites::Vector{CrossoverSite},
    side::Symbol,
)
    isempty(sites) && return Float64[]
    requested_path = if side === :left
        policy.left_path
    elseif side === :right
        policy.right_path
    else
        throw(ArgumentError("crossover side must be :left or :right"))
    end
    analysis = first(sites).analysis
    selected_ids = typed_analysis_site_ids(
        analysis; include_root=true, site_path=requested_path,
    )
    selected_id = isempty(selected_ids) ? 0 : only(selected_ids)
    return Float64[site.id == selected_id ? 1.0 : 0.0 for site in sites]
end

function _crossover_site_weights(
    policy::AbstractCrossoverSelectionPolicy,
    sites::Vector{CrossoverSite},
    side::Symbol,
)
    return Float64[
        _checked_crossover_weight(crossover_site_weight(policy, site, side))
        for site in sites
    ]
end

"""Return the nonnegative extra weight of a compatible site pair."""
crossover_pair_weight(
    ::AbstractCrossoverSelectionPolicy,
    ::CrossoverSite,
    ::CrossoverSite,
) = 1.0

# Built-in policies have factorized pair mass: all pair-specific mass is one,
# while exact paths are expressed by site weights. Custom policies may encode
# arbitrary sparse pair relations and therefore require complete pair-weight
# enumeration before proposal sampling.
_crossover_pair_weights_factorized(::AbstractCrossoverSelectionPolicy) = false
_crossover_pair_weights_factorized(::UniformCrossoverSelectionPolicy) = true
_crossover_pair_weights_factorized(::PathCrossoverSelectionPolicy) = true

function _checked_crossover_weight(weight)::Float64
    value = Float64(weight)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("crossover weights must be finite and nonnegative"))
    return value
end

function _weighted_crossover_order(rng::AbstractRNG, logweights::Vector{Float64})
    scores = Float64[
        logweight - log(randexp(rng))
        for logweight in logweights
    ]
    return sortperm(scores; rev=true)
end

"""Collect transient policy views from the one common typed analysis."""
function _collect_sites(
    tree::AstTree,
    type_ctx::Dict{Symbol,<:TyLike};
    tyfactory,
    target_ty::Union{Nothing,Ty}=nothing,
)
    analysis = prepare_typed_analysis(
        tree,
        type_ctx;
        factory=tyfactory,
        target_ty,
    )
    analysis isa Err && return analysis
    return CrossoverSite[CrossoverSite(analysis, id) for id in typed_analysis_site_ids(analysis)]
end

function _node_at_path(tree::AstTree, path::Vector{Int})::AstNode
    node = tree.root
    for idx in path
        node = children(node)[idx]
    end
    return node
end

function _node_at_path_maybe(tree::AstTree, path::Vector{Int})::Union{Nothing,AstNode}
    node = tree.root
    for idx in path
        kids = children(node)
        (1 <= idx <= length(kids)) || return nothing
        node = kids[idx]
    end
    return node
end

function _replace_at_path!(tree::AstTree, path::Vector{Int}, new_subtree::AstTree)
    if isempty(path)
        setroot!(tree, new_subtree)
        return tree
    end
    parent_path = path[1:end-1]
    parent = _node_at_path(tree, parent_path)
    setchild!(tree, parent, path[end], new_subtree)
    return tree
end

function _replace_at_path_maybe!(tree::AstTree, path::Vector{Int}, new_subtree::AstTree)::Bool
    if isempty(path)
        setroot!(tree, new_subtree)
        return true
    end
    parent_path = path[1:end-1]
    parent = _node_at_path_maybe(tree, parent_path)
    parent === nothing && return false
    kids = children(parent)
    (1 <= path[end] <= length(kids)) || return false
    setchild!(tree, parent, path[end], new_subtree)
    return true
end

function _fresh_const_symbol!(f::ConstSymbolFactory,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    reserved_globals::AbstractDict{Symbol,<:TyLike},
)::Symbol
    while true
        f.counter += 1
        name = Symbol(String(f.prefix), f.counter)
        (
            haskey(local_consts, name) ||
            haskey(local_consts_ty, name) ||
            haskey(reserved_globals, name)
        ) && continue
        return name
    end
end

function _remap_donor_locals!(subtree::AstTree,
    donor_local_consts::Dict{Symbol,Any},
    donor_local_consts_ty::Dict{Symbol,TyLike},
    recipient_local_consts::Dict{Symbol,Any},
    recipient_local_consts_ty::Dict{Symbol,TyLike},
    reserved_globals::AbstractDict{Symbol,<:TyLike},
    factory::ConstSymbolFactory,
)
    mapping = Dict{Symbol,Symbol}()

    traverse(subtree, PointerState{AstNode}) do t, node, bag
        node isa ConstNode || return nothing
        name = (node::ConstNode).name
        (haskey(donor_local_consts, name) && haskey(donor_local_consts_ty, name)) || return nothing

        new_name = get!(mapping, name) do
            fresh = _fresh_const_symbol!(
                factory,
                recipient_local_consts,
                recipient_local_consts_ty,
                reserved_globals,
            )
            recipient_local_consts[fresh] = donor_local_consts[name]
            recipient_local_consts_ty[fresh] = donor_local_consts_ty[name]
            fresh
        end

        pointer = getstate(bag, PointerState{AstNode})
        if pointer.parent === nothing
            setroot!(t, AstTree(ConstNode(new_name)))
        else
            setchild!(t, pointer.parent, pointer.child_index, AstTree(ConstNode(new_name)))
        end
        return nothing
    end

    return nothing
end

function _rename_local_consts!(tree::AstTree, mapping::Dict{Symbol,Symbol})
    isempty(mapping) && return nothing
    traverse(tree, PointerState{AstNode}) do t, node, bag
        node isa ConstNode || return nothing
        new_name = get(mapping, (node::ConstNode).name, nothing)
        new_name === nothing && return nothing
        replacement = AstTree(ConstNode(new_name))
        pointer = getstate(bag, PointerState{AstNode})
        if pointer.parent === nothing
            setroot!(t, replacement)
        else
            setchild!(t, pointer.parent, pointer.child_index, replacement)
        end
        return nothing
    end
    return nothing
end

function _separate_local_global_namespace!(
    recipient_tree::AstTree,
    recipient_local_consts::Dict{Symbol,Any},
    recipient_local_consts_ty::Dict{Symbol,TyLike},
    global_consts_ty::AbstractDict{Symbol,<:TyLike},
    factory::ConstSymbolFactory,
)
    conflicts = Set{Symbol}()
    foreach(name -> haskey(global_consts_ty, name) && push!(conflicts, name), keys(recipient_local_consts))
    foreach(name -> haskey(global_consts_ty, name) && push!(conflicts, name), keys(recipient_local_consts_ty))
    isempty(conflicts) && return nothing

    mapping = Dict{Symbol,Symbol}()
    for name in sort!(collect(conflicts); by=String)
        fresh = _fresh_const_symbol!(
            factory,
            recipient_local_consts,
            recipient_local_consts_ty,
            global_consts_ty,
        )
        haskey(recipient_local_consts, name) &&
            (recipient_local_consts[fresh] = pop!(recipient_local_consts, name))
        haskey(recipient_local_consts_ty, name) &&
            (recipient_local_consts_ty[fresh] = pop!(recipient_local_consts_ty, name))
        mapping[name] = fresh
    end
    _rename_local_consts!(recipient_tree, mapping)
    return nothing
end

function _contextual_node_key(
    node::AstNode,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    slots::Dict{Symbol,Int},
    payloads::Vector{Any},
)
    if node isa ConstNode
        name = (node::ConstNode).name
        if haskey(local_consts, name) && haskey(local_consts_ty, name)
            slot = get!(slots, name) do
                push!(payloads, (local_consts_ty[name], local_consts[name]))
                length(payloads)
            end
            return (:local, slot)
        end
        return (:const, name)
    elseif node isa IndexNode
        return (:index, (node::IndexNode).index)
    elseif node isa VarNode
        return (:var, (node::VarNode).name)
    elseif node isa AbsNode
        return (:abs, _contextual_node_key(
            (node::AbsNode).body,
            local_consts,
            local_consts_ty,
            slots,
            payloads,
        ))
    elseif node isa AppNode
        app = node::AppNode
        return (
            :app,
            _contextual_node_key(app.func, local_consts, local_consts_ty, slots, payloads),
            _contextual_node_key(app.arg, local_consts, local_consts_ty, slots, payloads),
        )
    elseif node isa FunctionDefNode
        return (
            :function_def,
            _contextual_node_key(
                node.body,
                local_consts,
                local_consts_ty,
                slots,
                payloads,
            ),
        )
    end
    throw(ArgumentError("unsupported crossover node $(typeof(node))"))
end

function _contextual_type_keys(types::Vector{TyLike})
    global_rigids = Dict{Any,TyRigid}()
    next_global = Ref(UInt(0))
    global_base = UInt(1) << (8 * sizeof(UInt) - 1)
    globalize(var) = get!(global_rigids, var) do
        next_global[] += UInt(1)
        TyRigid(global_base + next_global[])
    end

    keys = Any[]
    for ty in types
        if ty isa TyScheme
            quantified = Set(ty.vars)
            body = _inst(ty.ty, var -> var in quantified ? var : globalize(var))
            push!(keys, (:scheme, semantic_typekey(TyScheme(ty.vars, body))))
        else
            body = _inst(ty, globalize)
            push!(keys, (:mono, semantic_typekey(body)))
        end
    end
    return Tuple(keys)
end

function _contextual_subtree_key(
    node::AstNode,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
)
    slots = Dict{Symbol,Int}()
    payloads = Any[]
    template = _contextual_node_key(node, local_consts, local_consts_ty, slots, payloads)
    types = TyLike[payload[1] for payload in payloads]
    values = Any[payload[2] for payload in payloads]
    return (template, _contextual_type_keys(types), values)
end

function _same_contextual_subtree(a, b)::Bool
    return a[1] == b[1] && a[2] == b[2] && isequal(a[3], b[3])
end

function _crossover_application_roles_stable(
    original_tree::AstTree,
    child_tree::AstTree,
    path::Vector{Int},
)
    original_node = original_tree.root
    child_node = child_tree.root
    for child_index in path
        if original_node isa AppNode || child_node isa AppNode
            original_node isa AppNode && child_node isa AppNode || return false
            (original_node.func isa AbsNode) ==
                (child_node.func isa AbsNode) || return false
        end
        original_children = children(original_node)
        child_children = children(child_node)
        1 <= child_index <= length(original_children) || return false
        1 <= child_index <= length(child_children) || return false
        original_node = original_children[child_index]
        child_node = child_children[child_index]
    end
    return true
end

function _binding_at_level(site::CrossoverSite, level::Int)
    index = site.analysis.index
    current = index.env_head[site.id]
    while current != 0
        binding = index.bindings[current]
        crossover_site_depth(site) - binding.depth == level && return current
        current = binding.outer
    end
    return nothing
end

function _binder_at_level(site::CrossoverSite, level::Int)
    binder = _binding_at_level(site, level)
    binder === nothing && return nothing
    return get(site.analysis.binder_schemes, binder, nothing)
end

function _captures_fit(op, donor::CrossoverSite, destination::CrossoverSite)::Bool
    for level in crossover_site_capture_levels(donor)
        level < crossover_site_depth(destination) || return false
        donor_binder = _binding_at_level(donor, level)
        destination_binder = _binding_at_level(destination, level)
        donor_binder === nothing && return false
        destination_binder === nothing && return false
        # A FunctionDef self binder is not interchangeable with an ordinary
        # lexical binder even when their types and capture levels coincide.
        donor_is_self = donor.analysis.index.bindings[donor_binder].kind === :function_def
        destination_is_self =
            destination.analysis.index.bindings[destination_binder].kind === :function_def
        donor_is_self == destination_is_self || return false
        op.require_compatible_binders || continue
        donor_ty = _binder_at_level(donor, level)
        destination_ty = _binder_at_level(destination, level)
        donor_ty === nothing && return false
        destination_ty === nothing && return false
        iscompatible(donor_ty, destination_ty) || return false
        iscompatible(destination_ty, donor_ty) || return false
    end
    return true
end

function _sites_match(op, a::CrossoverSite, b::CrossoverSite)::Bool
    iscompatible(crossover_site_type(a), crossover_site_type(b)) || return false
    iscompatible(crossover_site_type(b), crossover_site_type(a)) || return false

    op.require_same_depth &&
        (crossover_site_depth(a) == crossover_site_depth(b)) ||
        (!op.require_same_depth) || return false

    return _captures_fit(op, a, b) && _captures_fit(op, b, a)
end

function _closed_compatibility_key(ty::Ty)
    free_vars = collect_free_vars(ty)
    isempty(free_vars) || return nothing
    return semantic_typekey(TyScheme(collect(free_vars), ty))
end

"""
    TypedSubtreeSwapCrossover(; max_try=40, require_same_depth=true,
        require_compatible_binders=true, min_sites=2, allow_root=true,
        selection=UniformCrossoverSelectionPolicy(),
        max_open_answers_per_pair=8) <: AbstractCrossover

Type-safe subtree swap crossover between two individuals.

This operator:
- collects typed sites in both parents,
- samples a compatible pair of sites,
- swaps the corresponding subtrees,
- remaps donor-local constants into the recipient's local constant pool using a
  per-call `ConstSymbolFactory` (no `gensym`),
- validates the offspring types using `infer` + `iscompatible`.

For contextually open donor sites, `max_open_answers_per_pair` bounds the
number of retrieval witnesses consumed for one site pair (default `8`).

When different depths are allowed, captures retain their De Bruijn level. The
binder compatibility filter checks only actually captured levels. Compatible
positive-weight pairs are attempted without replacement up to `max_try`.

Thread safety:
- requires a task-local `rng`,
- allocates a fresh `tyfactory = tyfactory_ctor()` per call,
- allocates per-offspring constant symbol factories via `const_symfactory_ctor()`.
"""
struct TypedSubtreeSwapCrossover <: AbstractCrossover
    max_try::Int
    max_open_answers_per_pair::Int
    require_same_depth::Bool
    require_compatible_binders::Bool
    min_sites::Int
    allow_root::Bool
    selection::AbstractCrossoverSelectionPolicy

    function TypedSubtreeSwapCrossover(
        max_try::Int,
        require_same_depth::Bool,
        require_compatible_binders::Bool,
        min_sites::Int,
        allow_root::Bool,
        selection::AbstractCrossoverSelectionPolicy,
        max_open_answers_per_pair::Int,
    )
        max_try >= 1 || throw(ArgumentError("max_try must be positive"))
        min_sites >= 1 || throw(ArgumentError("min_sites must be positive"))
        max_open_answers_per_pair >= 1 || throw(ArgumentError(
            "max_open_answers_per_pair must be positive",
        ))
        return new(
            max_try,
            max_open_answers_per_pair,
            require_same_depth,
            require_compatible_binders,
            min_sites,
            allow_root,
            selection,
        )
    end
end

TypedSubtreeSwapCrossover(
    max_try::Int,
    require_same_depth::Bool,
    require_compatible_binders::Bool,
    min_sites::Int,
    allow_root::Bool,
    ) = TypedSubtreeSwapCrossover(
        max_try,
        require_same_depth,
        require_compatible_binders,
        min_sites,
        allow_root,
        UniformCrossoverSelectionPolicy(),
        8,
    )

TypedSubtreeSwapCrossover(
    max_try::Int,
    require_same_depth::Bool,
    require_compatible_binders::Bool,
    min_sites::Int,
    allow_root::Bool,
    selection::AbstractCrossoverSelectionPolicy,
) = TypedSubtreeSwapCrossover(
    max_try,
    require_same_depth,
    require_compatible_binders,
    min_sites,
    allow_root,
    selection,
    8,
)

TypedSubtreeSwapCrossover(;
    max_try::Int=40,
    require_same_depth::Bool=true,
    require_compatible_binders::Bool=true,
    min_sites::Int=2,
    allow_root::Bool=true,
    selection::AbstractCrossoverSelectionPolicy=UniformCrossoverSelectionPolicy(),
    max_open_answers_per_pair::Int=8,
) = TypedSubtreeSwapCrossover(
    max_try,
    require_same_depth,
    require_compatible_binders,
    min_sites,
    allow_root,
    selection,
    max_open_answers_per_pair,
)

function _same_target_type(a::Ty, b::Ty)
    free_a = collect_free_vars(a)
    free_b = collect_free_vars(b)
    isempty(free_a) && isempty(free_b) && return isequal_mu(a, b)
    key_a = semantic_typekey(TyScheme(collect(free_a), a))
    key_b = semantic_typekey(TyScheme(collect(free_b), b))
    return key_a == key_b
end

"""
    crossover(op::TypedSubtreeSwapCrossover, a::Program, b::Program;
        target_a, target_b,
        rng=MersenneTwister(2000),
        global_consts_ty=nothing,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    ) -> Union{Tuple{Program,Program},Err}

Apply type-safe subtree swap crossover.

Required keyword arguments:
- `global_consts_ty`: shared, read-only typing context for global constants/operators.

Thread-safety requirements:
- pass a task-local `rng`;
- the implementation constructs a per-call `tyfactory = tyfactory_ctor()` and uses
  `const_symfactory_ctor()` to generate fresh local-constant symbols deterministically.
"""
function crossover(op::TypedSubtreeSwapCrossover,
    a::Program,
    b::Program;
    target_a::Ty,
    target_b::Ty,
    rng::AbstractRNG=MersenneTwister(2000),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    prune::Bool=true,
)
    _same_target_type(target_a, target_b) ||
        return Err(TargetTypeMismatchErr(target_a, target_b))
    global_consts_ty === nothing && return Err(MissingGlobalTypeCtxErr())

    tyfactory = tyfactory_ctor()

    # Pre-collect sites using the parents' own local type contexts.
    sites_a = _collect_sites(
        a.tree,
        merge(global_consts_ty, a.local_consts_ty);
        tyfactory=tyfactory,
        target_ty=target_a,
    )
    sites_a isa Err && return sites_a
    sites_b = _collect_sites(
        b.tree,
        merge(global_consts_ty, b.local_consts_ty);
        tyfactory=tyfactory,
        target_ty=target_b,
    )
    sites_b isa Err && return sites_b

    allow_root = op.allow_root
    sites_a2 = allow_root ? sites_a : filter(s -> s.id != 1, sites_a)
    sites_b2 = allow_root ? sites_b : filter(s -> s.id != 1, sites_b)

    length(sites_a2) >= op.min_sites || return Err(NoEligibleCrossoverSiteErr())
    length(sites_b2) >= op.min_sites || return Err(NoEligibleCrossoverSiteErr())

    weights_a = _crossover_site_weights(op.selection, sites_a2, :left)
    weights_b = _crossover_site_weights(op.selection, sites_b2, :right)
    compatibility_keys_a = _closed_compatibility_key.(crossover_site_type.(sites_a2))
    compatibility_keys_b = _closed_compatibility_key.(crossover_site_type.(sites_b2))
    pair_pool = Tuple{CrossoverSite,CrossoverSite}[]
    pair_logweights = Float64[]
    for (index_a, site_a) in pairs(sites_a2)
        weights_a[index_a] > 0 || continue
        for (index_b, site_b) in pairs(sites_b2)
            weights_b[index_b] > 0 || continue
            key_a = compatibility_keys_a[index_a]
            key_b = compatibility_keys_b[index_b]
            key_a !== nothing && key_b !== nothing && key_a != key_b && continue
            _sites_match(op, site_a, site_b) || continue
            pair_weight = _checked_crossover_weight(
                crossover_pair_weight(op.selection, site_a, site_b),
            )
            pair_weight > 0 || continue
            push!(pair_pool, (site_a, site_b))
            push!(
                pair_logweights,
                log(weights_a[index_a]) + log(weights_b[index_b]) + log(pair_weight),
            )
        end
    end
    isempty(pair_pool) && return Err(NoEligibleCrossoverPairErr())

    pair_order = _weighted_crossover_order(rng, pair_logweights)
    attempts = 0
    for order_index in 1:min(op.max_try, length(pair_order))
        attempts += 1
        pair_index = pair_order[order_index]
        site_a, site_b = pair_pool[pair_index]

        site_a_path = crossover_site_path(site_a)
        site_b_path = crossover_site_path(site_b)
        source_a = _node_at_path_maybe(a.tree, site_a_path)
        source_a === nothing && continue
        source_b = _node_at_path_maybe(b.tree, site_b_path)
        source_b === nothing && continue
        key_a = _contextual_subtree_key(source_a, a.local_consts, a.local_consts_ty)
        key_b = _contextual_subtree_key(source_b, b.local_consts, b.local_consts_ty)
        _same_contextual_subtree(key_a, key_b) && continue

        # Build offspring from fresh copies (transactional).
        tree_a = deepcopy(a.tree)
        tree_b = deepcopy(b.tree)
        local_a = copy(a.local_consts)
        local_a_ty = copy(a.local_consts_ty)
        local_b = copy(b.local_consts)
        local_b_ty = copy(b.local_consts_ty)

        node_a = _node_at_path_maybe(tree_a, site_a_path)
        node_a === nothing && continue
        node_b = _node_at_path_maybe(tree_b, site_b_path)
        node_b === nothing && continue
        sub_a = AstTree(deepcopy(node_a))
        sub_b = AstTree(deepcopy(node_b))

        # Preserve both sides of the global/local namespace boundary before splicing.
        factory_a = const_symfactory_ctor()
        factory_b = const_symfactory_ctor()
        _separate_local_global_namespace!(
            tree_a,
            local_a,
            local_a_ty,
            global_consts_ty,
            factory_a,
        )
        _separate_local_global_namespace!(
            tree_b,
            local_b,
            local_b_ty,
            global_consts_ty,
            factory_b,
        )
        _remap_donor_locals!(
            sub_b,
            b.local_consts,
            b.local_consts_ty,
            local_a,
            local_a_ty,
            global_consts_ty,
            factory_a,
        )
        _remap_donor_locals!(
            sub_a,
            a.local_consts,
            a.local_consts_ty,
            local_b,
            local_b_ty,
            global_consts_ty,
            factory_b,
        )

        _replace_at_path_maybe!(tree_a, site_a_path, sub_b) || continue
        _replace_at_path_maybe!(tree_b, site_b_path, sub_a) || continue

        _crossover_application_roles_stable(a.tree, tree_a, site_a_path) || continue
        _crossover_application_roles_stable(b.tree, tree_b, site_b_path) || continue

        if prune
            prune_local_consts!(tree_a, local_a, local_a_ty)
            prune_local_consts!(tree_b, local_b, local_b_ty)
        end

        # Validate type invariants.
        type_ctx_a = merge(global_consts_ty, local_a_ty)
        inf_a = prepare_typed_analysis(
            tree_a,
            type_ctx_a;
            factory=tyfactory,
            target_ty=target_a,
        )
        inf_a isa Err && continue

        type_ctx_b = merge(global_consts_ty, local_b_ty)
        inf_b = prepare_typed_analysis(
            tree_b,
            type_ctx_b;
            factory=tyfactory,
            target_ty=target_b,
        )
        inf_b isa Err && continue

        child_a = _publish_program(tree_a, local_a, local_a_ty)
        child_b = _publish_program(tree_b, local_b, local_b_ty)
        return (child_a, child_b)
    end

    attempts == length(pair_pool) && return Err(CrossoverPairPoolExhaustedErr(attempts))
    return Err(CrossoverMaxTryReachedErr(op.max_try))
end

"""
    IndependentCrossover(operator)

Compose the two directed recombination calls used by the default regression
path.  The first child is produced from `(recipient=a, donor=b)` and the
second from `(recipient=b, donor=a)`; each direction keeps its own target and
selection role.  This is deliberately separate from `crossover` on
`TypedSubtreeSwapCrossover`, whose contract is one reciprocal site-pair swap
with atomic two-child validation.
"""
struct IndependentCrossover <: AbstractCrossover
    operator::TypedSubtreeSwapCrossover
end

IndependentCrossover(; kwargs...) =
    IndependentCrossover(TypedSubtreeSwapCrossover(; kwargs...))

function crossover(
    op::IndependentCrossover,
    a::Program,
    b::Program;
    target_a::Ty,
    target_b::Ty,
    rng::AbstractRNG=MersenneTwister(2000),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    tyfactory_ctor=make_tyvar_factory,
    kwargs...,
)
    global_consts_ty === nothing && return Err(MissingGlobalTypeCtxErr())
    prepared_a = _prepare_crossover_parent(
        a,
        target_a,
        global_consts_ty;
        tyfactory_ctor,
    )
    prepared_a isa Err && return prepared_a
    prepared_b = _prepare_crossover_parent(
        b,
        target_b,
        global_consts_ty;
        tyfactory_ctor,
    )
    prepared_b isa Err && return prepared_b

    child_a = recombine(
        op.operator,
        a,
        b;
        recipient_target=target_a,
        donor_target=target_b,
        rng,
        global_consts_ty,
        tyfactory_ctor,
        selection_roles=(:left, :right),
        _prepared_recipient=prepared_a,
        _prepared_donor=prepared_b,
        kwargs...,
    )
    child_a isa Err && return child_a

    child_b = recombine(
        op.operator,
        b,
        a;
        recipient_target=target_b,
        donor_target=target_a,
        rng,
        global_consts_ty,
        tyfactory_ctor,
        selection_roles=(:right, :left),
        _prepared_recipient=prepared_b,
        _prepared_donor=prepared_a,
        kwargs...,
    )
    child_b isa Err && return child_b
    return (child_a, child_b)
end

@testsnippet PopCrossBaseImports begin
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, TypedSubtreeSwapCrossover, crossover, prepare_typed_analysis
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike, infer, iscompatible
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.TypeSystem: make_tyvar_factory
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike
    using LambdaRegression.Utils: Err
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
end

@testitem "Populations recombine is a directed closed public seam" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program,
        PathCrossoverSelectionPolicy,
        TypedSubtreeSwapCrossover,
        recombine
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode,
        AppNode,
        AstTree,
        ConstNode,
        IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    int_to_int = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :USEF => TyFun(int_to_int, IntTy),
        :INC => int_to_int,
    )
    recipient = Program(
        AstTree(AppNode(ConstNode(:USEF), ConstNode(:INC))),
    )
    donor = Program(
        AstTree(AbsNode(:x, IndexNode(0))),
    )
    before_recipient = deepcopy(recipient)
    before_donor = deepcopy(donor)
    op = TypedSubtreeSwapCrossover(
        max_try=1,
        min_sites=1,
        selection=PathCrossoverSelectionPolicy([2], Int[]),
    )

    run_once() = recombine(
        op,
        recipient,
        donor;
        rng=MersenneTwister(1111),
        global_consts_ty=global_types,
        recipient_target=IntTy,
        donor_target=int_to_int,
    )
    first = run_once()
    second = run_once()
    @test !(first isa Err)
    @test !(second isa Err)
    if !(first isa Err) && !(second isa Err)
        @test first.tree.root == AppNode(
            ConstNode(:USEF),
            AbsNode(:x, IndexNode(0)),
        )
        @test first.tree == second.tree
        @test infer(first.tree, global_types) == IntTy
    end
    @test recipient.tree == before_recipient.tree
    @test recipient.local_consts == before_recipient.local_consts
    @test recipient.local_consts_ty == before_recipient.local_consts_ty
    @test donor.tree == before_donor.tree
    @test donor.local_consts == before_donor.local_consts
    @test donor.local_consts_ty == before_donor.local_consts_ty

    # Site analysis is observational. A monomorphic variable owned by the
    # caller's context must not be linked while the recipient is inferred.
    factory = LambdaRegression.TypeSystem.make_tyvar_factory()
    caller_owned = factory(0)
    mutable_global_types = Dict{Symbol,TyLike}(
        :APPLY => TyFun(caller_owned, IntTy),
        :ONE => IntTy,
    )
    mutable_recipient = Program(
        AstTree(AppNode(ConstNode(:APPLY), ConstNode(:ONE))),
    )
    recombine(
        TypedSubtreeSwapCrossover(min_sites=1),
        mutable_recipient,
        donor;
        rng=MersenneTwister(1234),
        global_consts_ty=mutable_global_types,
        recipient_target=IntTy,
        donor_target=int_to_int,
    )
    @test caller_owned.instance === nothing
end

@testitem "Populations recombine skips excluded mu sites when a finite pair exists" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations.Populations:
        Program, TypedSubtreeSwapCrossover, recombine
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyLike, TyMu, infer, isequal_mu, make_tyvar_factory
    using LambdaRegression.Utils: Err

    factory = make_tyvar_factory()
    recursive = factory(0)
    mu_type = TyMu(recursive, TyFun(IntTy, recursive))
    global_types = Dict{Symbol,TyLike}(
        :WRAP => TyFun(IntTy, mu_type),
        :ONE => IntTy,
        :TWO => IntTy,
    )
    recipient = Program(
        AstTree(AppNode(ConstNode(:WRAP), ConstNode(:ONE))),
    )
    donor = Program(AstTree(ConstNode(:TWO)))

    child = recombine(
        TypedSubtreeSwapCrossover(max_try=8, min_sites=1),
        recipient,
        donor;
        rng=MersenneTwister(8000),
        global_consts_ty=global_types,
        recipient_target=mu_type,
        donor_target=IntTy,
    )
    @test !(child isa Err)
    if !(child isa Err)
        @test child.tree.root == AppNode(ConstNode(:WRAP), ConstNode(:TWO))
        @test isequal_mu(infer(child.tree, global_types), mu_type)
    end
end

@testitem "Populations TypedSubtreeSwapCrossover: basic success + type stable (TyFun)" setup=[PopCrossBaseImports] begin
    rng = MersenneTwister(1)
    target_ty = TyFun(IntTy, IntTy)
    global_consts_ty = Dict{Symbol,TyLike}(:add => TyFun(IntTy, TyFun(IntTy, IntTy)))

    tree_a = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:k))))
    tree_b = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), ConstNode(:k)), IndexNode(0))))

    ind_a = Program(tree_a; local_consts=Dict{Symbol,Any}(:k => 1), local_consts_ty=Dict{Symbol,TyLike}(:k => IntTy))
    ind_b = Program(tree_b; local_consts=Dict{Symbol,Any}(:k => 99), local_consts_ty=Dict{Symbol,TyLike}(:k => IntTy))

    op = TypedSubtreeSwapCrossover(max_try=20)
    (c1, c2) = crossover(
        op,
        ind_a,
        ind_b;
        rng=rng,
        global_consts_ty=global_consts_ty,
        target_a=target_ty,
        target_b=target_ty,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    tyf = make_tyvar_factory()
    t1 = infer(c1.tree, merge(global_consts_ty, c1.local_consts_ty), tyf)
    @test iscompatible(t1, target_ty)
    tyf2 = make_tyvar_factory()
    t2 = infer(c2.tree, merge(global_consts_ty, c2.local_consts_ty), tyf2)
    @test iscompatible(t2, target_ty)
end

@testitem "Populations _collect_sites paths are structurally valid" setup=[PopCrossBaseImports] begin
    # λx. add (#0) (add k #0)
    tree = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), IndexNode(0)), AppNode(AppNode(ConstNode(:add), ConstNode(:k)), IndexNode(0)))))
    global_consts_ty = Dict{Symbol,TyLike}(:add => TyFun(IntTy, TyFun(IntTy, IntTy)))
    local_consts_ty = Dict{Symbol,TyLike}(:k => IntTy)
    tyf = make_tyvar_factory()
    sites = LambdaRegression.Mutations.Populations._collect_sites(tree, merge(global_consts_ty, local_consts_ty); tyfactory=tyf)
    @test !(sites isa Err)
    for s in sites
        @test LambdaRegression.Mutations.Populations._node_at_path_maybe(
            tree,
            LambdaRegression.Mutations.Populations.crossover_site_path(s),
        ) !== nothing
    end
end

@testitem "Populations TypedSubtreeSwapCrossover: local-const conflict renamed" setup=[PopCrossBaseImports] begin
    rng = MersenneTwister(1111)
    global_consts_ty = Dict{Symbol,TyLike}()

    # Both parents use the same local constant name but different values.
    ind_a = Program(AstTree(ConstNode(:c1)); local_consts=Dict{Symbol,Any}(:c1 => 1), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))
    ind_b = Program(AstTree(ConstNode(:c1)); local_consts=Dict{Symbol,Any}(:c1 => 2), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))

    op = TypedSubtreeSwapCrossover(max_try=5, min_sites=1, allow_root=true)
    (c1, c2) = crossover(
        op,
        ind_a,
        ind_b;
        rng=rng,
        global_consts_ty=global_consts_ty,
        target_a=IntTy,
        target_b=IntTy,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    @test c1.tree.root isa ConstNode
    @test (c1.tree.root::ConstNode).name != :c1
    @test haskey(c1.local_consts, (c1.tree.root::ConstNode).name)
    @test c1.local_consts[(c1.tree.root::ConstNode).name] == 2

    @test c2.tree.root isa ConstNode
    @test (c2.tree.root::ConstNode).name != :c1
    @test haskey(c2.local_consts, (c2.tree.root::ConstNode).name)
    @test c2.local_consts[(c2.tree.root::ConstNode).name] == 1
end

@testitem "Populations crossover preserves donor globals against recipient locals" setup=[PopCrossBaseImports] begin
    global_consts_ty = Dict{Symbol,TyLike}(:a => IntTy, :g => IntTy)
    ind_a = Program(
        AstTree(ConstNode(:a)),
        local_consts=Dict{Symbol,Any}(:g => 99),
        local_consts_ty=Dict{Symbol,TyLike}(:g => IntTy),
    )
    ind_b = Program(AstTree(ConstNode(:g)))

    result = crossover(
        TypedSubtreeSwapCrossover(max_try=1, min_sites=1, allow_root=true),
        ind_a,
        ind_b;
        rng=MersenneTwister(2000),
        global_consts_ty=global_consts_ty,
        target_a=IntTy,
        target_b=IntTy,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, _ = result
        @test child_a.tree.root == ConstNode(:g)
        @test !haskey(child_a.local_consts, :g)
        @test iscompatible(infer(child_a.tree, merge(global_consts_ty, child_a.local_consts_ty)), IntTy)
    end
end

@testitem "Populations crossover rejects local alpha-renaming as a no-op" setup=[PopCrossBaseImports] begin
    ind_a = Program(
        AstTree(ConstNode(:left)),
        local_consts=Dict{Symbol,Any}(:left => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:left => IntTy),
    )
    ind_b = Program(
        AstTree(ConstNode(:right)),
        local_consts=Dict{Symbol,Any}(:right => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:right => IntTy),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(max_try=20, min_sites=1, allow_root=true),
        ind_a,
        ind_b;
        rng=MersenneTwister(1),
        global_consts_ty=Dict{Symbol,TyLike}(),
        target_a=IntTy,
        target_b=IntTy,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    @test result isa Err
    if result isa Err
        @test nameof(typeof(result.value)) == :CrossoverPairPoolExhaustedErr
        nameof(typeof(result.value)) == :CrossoverPairPoolExhaustedErr &&
            @test result.value.attempts == 1
    end
end

@testitem "Populations crossover samples huge finite weights stably" begin
    using Test
    using Random

    logweights = [log(floatmax(Float64)), log(floatmax(Float64))]
    orders = [
        LambdaRegression.Mutations.Populations._weighted_crossover_order(
            MersenneTwister(seed),
            logweights,
        ) for seed in 1:64
    ]
    @test all(order -> sort(order) == [1, 2], orders)
    @test LambdaRegression.Mutations.Populations._weighted_crossover_order(
        MersenneTwister(555),
        logweights,
    ) == LambdaRegression.Mutations.Populations._weighted_crossover_order(
        MersenneTwister(555),
        logweights,
    )
    observed = Set(first(order) for order in orders)
    @test observed == Set([1, 2])
end

@testitem "Populations crossover preserves positive pair mass across extreme scales" begin
    using Test
    using Random
    import LambdaRegression.Mutations.Populations:
        crossover_pair_weight, crossover_site_weight, crossover_site_path
    using LambdaRegression.Mutations.Populations:
        AbstractCrossoverSelectionPolicy, CrossoverSite, Program,
        TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    struct ExtremeScalePolicy <: AbstractCrossoverSelectionPolicy end
    crossover_site_weight(
        ::ExtremeScalePolicy,
        site::CrossoverSite,
        ::Symbol,
    ) = crossover_site_path(site) == [1] ? 1.0e-300 : 1.0
    crossover_pair_weight(
        ::ExtremeScalePolicy,
        left::CrossoverSite,
        right::CrossoverSite,
    ) = crossover_site_path(left) == [1] && crossover_site_path(right) == [1] ? 1.0e300 : 0.0

    unary_int = TyFun(IntTy, IntTy)
    globals = Dict{Symbol,TyLike}(:inc => unary_int, :dec => unary_int)
    parent_a = Program(
        AstTree(AbsNode(:x, AppNode(ConstNode(:inc), IndexNode(0)))),
    )
    parent_b = Program(
        AstTree(AbsNode(:x, AppNode(ConstNode(:dec), IndexNode(0)))),
    )
    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=ExtremeScalePolicy(),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(8000),
        global_consts_ty=globals,
        target_a=unary_int,
        target_b=unary_int,
    )

    @test !(result isa Err)
end

@testitem "Populations crossover reports an empty positive-weight pair set" setup=[PopCrossBaseImports] begin
    using LambdaRegression.Mutations.Populations: PathCrossoverSelectionPolicy

    global_consts_ty = Dict{Symbol,TyLike}(:a => IntTy, :b => IntTy)
    ind_a = Program(AstTree(ConstNode(:a)))
    ind_b = Program(AstTree(ConstNode(:b)))
    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            allow_root=true,
            selection=PathCrossoverSelectionPolicy([1], [1]),
        ),
        ind_a,
        ind_b;
        rng=MersenneTwister(1111),
        global_consts_ty=global_consts_ty,
        target_a=IntTy,
        target_b=IntTy,
    )

    @test result isa Err
    @test nameof(typeof(result.value)) == :NoEligibleCrossoverPairErr
end

@testitem "Populations path weights preserve side roles and prefilter min_sites" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    function_type = TyFun(IntTy, IntTy)
    globals = Dict{Symbol,TyLike}(
        :INC => function_type,
        :ONE => IntTy,
    )
    left = Program(AstTree(AbsNode(
        :x, AppNode(ConstNode(:INC), IndexNode(0)),
    )))
    right = Program(AstTree(AbsNode(
        :y, AppNode(ConstNode(:INC), IndexNode(0)),
    )))
    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=2,
            allow_root=true,
            require_compatible_binders=false,
            selection=PathCrossoverSelectionPolicy([1], [1, 2]),
        ),
        left,
        right;
        rng=MersenneTwister(2000),
        global_consts_ty=globals,
        target_a=function_type,
        target_b=function_type,
    )
    @test !(result isa Err)
    result isa Err && return
    child_left, child_right = result
    @test child_left.tree != left.tree
    @test child_right.tree != right.tree
    @test iscompatible(infer(child_left.tree, globals), function_type)
    @test iscompatible(infer(child_right.tree, globals), function_type)

    root_excluded = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=2,
            allow_root=false,
            require_compatible_binders=false,
            selection=PathCrossoverSelectionPolicy(Int[], [1, 2]),
        ),
        left,
        right;
        rng=MersenneTwister(1),
        global_consts_ty=globals,
        target_a=function_type,
        target_b=function_type,
    )
    @test root_excluded isa Err
    if root_excluded isa Err
        @test nameof(typeof(root_excluded.value)) == :NoEligibleCrossoverPairErr
    end
end

@testitem "Populations crossover validates routing sides and budgets" setup=[PopCrossBaseImports] begin
    using LambdaRegression.Mutations.Populations:
        CrossoverSite, PathCrossoverSelectionPolicy, crossover_site_weight

    analysis = prepare_typed_analysis(
        AstTree(ConstNode(:one)),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    site = CrossoverSite(analysis, 1)
    policy = PathCrossoverSelectionPolicy(Int[], Int[])
    @test_throws ArgumentError crossover_site_weight(policy, site, :middle)
    @test_throws ArgumentError TypedSubtreeSwapCrossover(max_try=0)
    @test_throws ArgumentError TypedSubtreeSwapCrossover(min_sites=0)
    @test_throws ArgumentError TypedSubtreeSwapCrossover(
        max_open_answers_per_pair=0,
    )
    @test_throws ArgumentError TypedSubtreeSwapCrossover(
        0,
        true,
        true,
        1,
        true,
    )
end

@testitem "Populations TypedSubtreeSwapCrossover: target_ty mismatch fails" setup=[PopCrossBaseImports] begin
    rng = MersenneTwister(1234)
    global_consts_ty = Dict{Symbol,TyLike}()

    ind_a = Program(AstTree(ConstNode(:c1)); local_consts=Dict{Symbol,Any}(:c1 => 1), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))
    ind_b = Program(AstTree(AbsNode(:x, IndexNode(0))))

    op = TypedSubtreeSwapCrossover(max_try=5, min_sites=1, allow_root=true)
    res = crossover(
        op,
        ind_a,
        ind_b;
        rng=rng,
        global_consts_ty=global_consts_ty,
        target_a=IntTy,
        target_b=TyFun(IntTy, IntTy),
    )
    @test res isa Err
end

@testitem "Populations TypedSubtreeSwapCrossover accepts equirecursive target equality" setup=[PopCrossBaseImports] begin
    fresh = make_tyvar_factory()
    one_binder = fresh(0)
    period_one = LambdaRegression.TypeSystem.TyMu(
        one_binder,
        TyFun(IntTy, one_binder),
    )
    two_binder = fresh(0)
    period_two = LambdaRegression.TypeSystem.TyMu(
        two_binder,
        TyFun(IntTy, TyFun(IntTy, two_binder)),
    )

    ind_a = Program(AstTree(ConstNode(:a)))
    ind_b = Program(AstTree(ConstNode(:b)))
    result = crossover(
        TypedSubtreeSwapCrossover(max_try=1, min_sites=1, allow_root=true),
        ind_a,
        ind_b;
        global_consts_ty=Dict{Symbol,TyLike}(:a => period_one, :b => period_two),
        target_a=period_one,
        target_b=period_two,
    )
    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree == AstTree(ConstNode(:b))
        @test child_b.tree == AstTree(ConstNode(:a))
    end
end

@testitem "Populations TypedSubtreeSwapCrossover handles open target equality" setup=[PopCrossBaseImports] begin
    fresh = make_tyvar_factory()
    target_var = fresh(0)
    open_target = TyFun(target_var, target_var)
    ind_a = Program(AstTree(ConstNode(:a)))
    ind_b = Program(AstTree(ConstNode(:b)))
    result = crossover(
        TypedSubtreeSwapCrossover(max_try=1, min_sites=1, allow_root=true),
        ind_a,
        ind_b;
        global_consts_ty=Dict{Symbol,TyLike}(
            :a => open_target,
            :b => open_target,
        ),
        target_a=open_target,
        target_b=open_target,
    )
    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree == AstTree(ConstNode(:b))
        @test child_b.tree == AstTree(ConstNode(:a))
    end
end

@testitem "Populations crossover contextual no-op keys support open local types" setup=[PopCrossBaseImports] begin
    fresh = make_tyvar_factory()
    left_ty = fresh(0)
    right_ty = fresh(0)
    parent_a = Program(
        AstTree(ConstNode(:left)),
        local_consts=Dict{Symbol,Any}(:left => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:left => left_ty),
    )
    parent_b = Program(
        AstTree(ConstNode(:right)),
        local_consts=Dict{Symbol,Any}(:right => 2),
        local_consts_ty=Dict{Symbol,TyLike}(:right => right_ty),
    )

    result = try
        crossover(
            TypedSubtreeSwapCrossover(max_try=1, min_sites=1, allow_root=true),
            parent_a,
            parent_b;
            rng=MersenneTwister(42),
            global_consts_ty=Dict{Symbol,TyLike}(),
            target_a=left_ty,
            target_b=right_ty,
        )
    catch exception
        exception
    end

    @test !(result isa Exception)
    @test !(result isa Err)
end

@testitem "Populations crossover contextual keys preserve joint open-local relations" setup=[PopCrossBaseImports] begin
    using LambdaRegression.TypeSystem: TyScheme

    pair_tree = AstTree(
        AppNode(AppNode(ConstNode(:PAIR), ConstNode(:a)), ConstNode(:b)),
    )
    values = Dict{Symbol,Any}(:a => 1, :b => 2)

    shared_factory = make_tyvar_factory()
    shared = shared_factory(0)
    shared_types = Dict{Symbol,TyLike}(:a => shared, :b => shared)

    independent_factory = make_tyvar_factory()
    independent_types = Dict{Symbol,TyLike}(
        :a => independent_factory(0),
        :b => independent_factory(0),
    )
    shared_key = LambdaRegression.Mutations.Populations._contextual_subtree_key(
        pair_tree.root,
        values,
        shared_types,
    )
    independent_key = LambdaRegression.Mutations.Populations._contextual_subtree_key(
        pair_tree.root,
        values,
        independent_types,
    )
    @test !LambdaRegression.Mutations.Populations._same_contextual_subtree(
        shared_key,
        independent_key,
    )

    mono_factory = make_tyvar_factory()
    mono = mono_factory(0)
    scheme_factory = make_tyvar_factory()
    quantified = scheme_factory(0)
    mono_key = LambdaRegression.Mutations.Populations._contextual_subtree_key(
        ConstNode(:a),
        Dict{Symbol,Any}(:a => 1),
        Dict{Symbol,TyLike}(:a => mono),
    )
    scheme_key = LambdaRegression.Mutations.Populations._contextual_subtree_key(
        ConstNode(:a),
        Dict{Symbol,Any}(:a => 1),
        Dict{Symbol,TyLike}(:a => TyScheme([quantified], quantified)),
    )
    @test !LambdaRegression.Mutations.Populations._same_contextual_subtree(
        mono_key,
        scheme_key,
    )
end

@testitem "Populations TypedSubtreeSwapCrossover handles open equirecursive targets" setup=[PopCrossBaseImports] begin
    fresh = make_tyvar_factory()
    atom_a = fresh(0)
    binder_a = fresh(0)
    target_a = LambdaRegression.TypeSystem.TyMu(
        binder_a,
        TyFun(atom_a, binder_a),
    )
    atom_b = fresh(0)
    binder_b = fresh(0)
    target_b = LambdaRegression.TypeSystem.TyMu(
        binder_b,
        TyFun(atom_b, TyFun(atom_b, binder_b)),
    )
    ind_a = Program(AstTree(ConstNode(:a)))
    ind_b = Program(AstTree(ConstNode(:b)))
    result = crossover(
        TypedSubtreeSwapCrossover(max_try=1, min_sites=1, allow_root=true),
        ind_a,
        ind_b;
        target_a=target_a,
        target_b=target_b,
        global_consts_ty=Dict{Symbol,TyLike}(:a => target_a, :b => target_b),
    )
    @test !(
        result isa Err &&
        result.value isa LambdaRegression.Mutations.Populations.TargetTypeMismatchErr
    )
end

@testitem "Populations TypedSubtreeSwapCrossover rejects alpha-equivalent no-op swaps" setup=[PopCrossBaseImports] begin
    parent_a = Program(
        AstTree(AbsNode(:left_name, IndexNode(0))),
    )
    parent_b = Program(
        AstTree(AbsNode(:right_name, IndexNode(0))),
    )
    before_a = deepcopy(parent_a.tree)
    before_b = deepcopy(parent_b.tree)

    result = crossover(
        TypedSubtreeSwapCrossover(max_try=8, min_sites=1, allow_root=true),
        parent_a,
        parent_b;
        rng=MersenneTwister(42),
        global_consts_ty=Dict{Symbol,TyLike}(),
        target_a=TyFun(IntTy, IntTy),
        target_b=TyFun(IntTy, IntTy),
    )

    @test result isa Err
    @test parent_a.tree == before_a
    @test parent_b.tree == before_b
end

@testitem "Populations crossover analyzes maximally generalized let bodies" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _collect_sites, crossover_site_path, crossover_site_type,
        crossover_site_binder_types
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyLike, TyProd, TyScheme, get_type_ctx,
        make_tyvar_factory
    using LambdaRegression.Primitives: PAIR, TRUE
    using LambdaRegression.Utils: Err

    pair(left, right) = AppNode(AppNode(ConstNode(:PAIR), left), right)
    tree = AstTree(
        AppNode(
            AbsNode(
                :id,
                pair(
                    AppNode(IndexNode(0), ConstNode(:one)),
                    AppNode(IndexNode(0), ConstNode(:truth)),
                ),
            ),
            AbsNode(:x, IndexNode(0)),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(get_type_ctx(Dict{Symbol,Any}(
        :PAIR => PAIR,
        :one => 1,
        :truth => TRUE,
    )))
    sites = _collect_sites(tree, type_ctx; tyfactory=make_tyvar_factory())

    @test !(sites isa Err)
    sites isa Err && return
    @test any(site -> length(crossover_site_path(site)) >= 2 &&
        crossover_site_path(site)[1:2] == [1, 1], sites)
    @test any(site -> crossover_site_type(site) == TyProd(IntTy, BoolTy), sites)
    @test any(
        site -> any(
            binder_ty -> binder_ty isa TyScheme && !isempty(binder_ty.vars),
            crossover_site_binder_types(site),
        ),
        sites,
    )
end

@testitem "Populations crossover rejects an ordinary-to-let application role change" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    int_to_int = TyFun(IntTy, IntTy)
    higher = TyFun(int_to_int, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :one => IntTy,
        :use_higher => TyFun(higher, IntTy),
        :head => higher,
    )
    parent_a = Program(
        AstTree(AppNode(ConstNode(:use_higher), AbsNode(:f, ConstNode(:one)))),
    )
    parent_b = Program(
        AstTree(AppNode(ConstNode(:head), AbsNode(:x, IndexNode(0)))),
    )
    before_a = deepcopy(parent_a.tree)
    before_b = deepcopy(parent_b.tree)
    policy = PathCrossoverSelectionPolicy([2], [1])

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=false,
            selection=policy,
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(2000),
        global_consts_ty=global_types,
        target_a=IntTy,
        target_b=IntTy,
    )

    @test result isa Err
    @test parent_a.tree == before_a
    @test parent_b.tree == before_b
end

@testitem "Populations crossover permits closed sites at different depths" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :one => IntTy,
        :two => IntTy,
        :consume => TyFun(TyFun(IntTy, IntTy), IntTy),
    )
    parent_a = Program(AstTree(AbsNode(:x, ConstNode(:one))))
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(ConstNode(:consume), AbsNode(:y, ConstNode(:two))),
            ),
        ),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=true,
            selection=PathCrossoverSelectionPolicy([1], [1, 2, 1]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree == AstTree(AbsNode(:x, ConstNode(:two)))
        @test child_b.tree == AstTree(
            AbsNode(:x, AppNode(ConstNode(:consume), AbsNode(:y, ConstNode(:one)))),
        )
        @test iscompatible(infer(child_a.tree, global_types), target_ty)
        @test iscompatible(infer(child_b.tree, global_types), target_ty)
    end
end

@testitem "Populations crossover maps actual captures by De Bruijn level" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    binary_int = TyFun(IntTy, TyFun(IntTy, IntTy))
    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :add => binary_int,
        :mul => binary_int,
        :consume => TyFun(TyFun(IntTy, IntTy), IntTy),
        :one => IntTy,
        :two => IntTy,
    )
    add_x = AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:one))
    mul_y = AppNode(AppNode(ConstNode(:mul), IndexNode(0)), ConstNode(:two))
    parent_a = Program(AstTree(AbsNode(:x, add_x)))
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(ConstNode(:consume), AbsNode(:y, mul_y)),
            ),
        ),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=true,
            selection=PathCrossoverSelectionPolicy([1], [1, 2, 1]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1111),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree.root.body == mul_y
        @test child_b.tree.root.body.arg.body == add_x
        @test iscompatible(infer(child_a.tree, global_types), target_ty)
        @test iscompatible(infer(child_b.tree, global_types), target_ty)
    end
end

@testitem "Populations crossover rebases an internal lambda with a level-one capture" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    binary_int = TyFun(IntTy, TyFun(IntTy, IntTy))
    unary_int = TyFun(IntTy, IntTy)
    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    global_types = Dict{Symbol,TyLike}(
        :add => binary_int,
        :mul => binary_int,
        :consume => TyFun(unary_int, IntTy),
        :consume_outer => TyFun(unary_int, IntTy),
        :one => IntTy,
        :two => IntTy,
    )
    captured_a = AbsNode(
        :z,
        AppNode(AppNode(ConstNode(:add), IndexNode(2)), ConstNode(:one)),
    )
    captured_b = AbsNode(
        :z,
        AppNode(AppNode(ConstNode(:mul), IndexNode(2)), ConstNode(:two)),
    )
    parent_a = Program(
        AstTree(AbsNode(:x, AbsNode(:y, AppNode(ConstNode(:consume), captured_a)))),
    )
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AbsNode(
                    :y,
                    AppNode(
                        ConstNode(:consume_outer),
                        AbsNode(:u, AppNode(ConstNode(:consume), captured_b)),
                    ),
                ),
            ),
        ),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=true,
            selection=PathCrossoverSelectionPolicy([1, 1, 2], [1, 1, 2, 1, 2]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1234),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree.root.body.body.arg == captured_b
        @test child_b.tree.root.body.body.arg.body.arg == captured_a
        @test iscompatible(infer(child_a.tree, global_types), target_ty)
        @test iscompatible(infer(child_b.tree, global_types), target_ty)
    end
end

@testitem "Populations crossover rejects a capture absent at the destination" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    binary_int = TyFun(IntTy, TyFun(IntTy, IntTy))
    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :add => binary_int,
        :consume => TyFun(TyFun(IntTy, IntTy), IntTy),
        :one => IntTy,
        :two => IntTy,
    )
    parent_a = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:one)),
            ),
        ),
    )
    escaped = AppNode(AppNode(ConstNode(:add), IndexNode(1)), ConstNode(:two))
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(ConstNode(:consume), AbsNode(:y, escaped)),
            ),
        ),
    )
    before_a = deepcopy(parent_a.tree)
    before_b = deepcopy(parent_b.tree)

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=true,
            selection=PathCrossoverSelectionPolicy([1], [1, 2, 1]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1234),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test result isa Err
    @test parent_a.tree == before_a
    @test parent_b.tree == before_b
end

@testitem "Populations crossover preserves polymorphic let captures" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyLike, TyProd, get_type_ctx, infer
    using LambdaRegression.Primitives: FALSE, PAIR, TRUE
    using LambdaRegression.Utils: Err

    pair(left, right) = AppNode(AppNode(ConstNode(:PAIR), left), right)
    apply_id(value) = AppNode(IndexNode(0), ConstNode(value))
    make_parent(int_name, bool_name) = Program(
        AstTree(
            AppNode(
                AbsNode(:id, pair(apply_id(int_name), apply_id(bool_name))),
                AbsNode(:x, IndexNode(0)),
            ),
        ),
    )
    global_types = Dict{Symbol,TyLike}(get_type_ctx(Dict{Symbol,Any}(
        :PAIR => PAIR,
        :one => 1,
        :two => 2,
        :truth => TRUE,
        :falsehood => FALSE,
    )))

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy([1, 1, 1, 2], [1, 1, 1, 2]),
        ),
        make_parent(:one, :truth),
        make_parent(:two, :falsehood);
        rng=MersenneTwister(8000),
        global_consts_ty=global_types,
        target_a=TyProd(IntTy, BoolTy),
        target_b=TyProd(IntTy, BoolTy),
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test infer(child_a.tree, global_types) == TyProd(IntTy, BoolTy)
        @test infer(child_b.tree, global_types) == TyProd(IntTy, BoolTy)
    end
end

@testitem "Populations crossover local remapping reserves global symbols" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    binary_int = TyFun(IntTy, TyFun(IntTy, IntTy))
    global_types = Dict{Symbol,TyLike}(
        :add => binary_int,
        :c1 => IntTy,
        :one => IntTy,
        :two => IntTy,
    )
    parent_a = Program(
        AstTree(
            AppNode(
                AppNode(ConstNode(:add), ConstNode(:c1)),
                ConstNode(:one),
            ),
        ),
    )
    parent_b = Program(
        AstTree(
            AppNode(
                AppNode(ConstNode(:add), ConstNode(:donor)),
                ConstNode(:two),
            ),
        ),
        local_consts=Dict{Symbol,Any}(:donor => 9),
        local_consts_ty=Dict{Symbol,TyLike}(:donor => IntTy),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy([2], [1, 2]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(42),
        global_consts_ty=global_types,
        target_a=IntTy,
        target_b=IntTy,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test !haskey(child_a.local_consts, :c1)
        @test child_a.tree.root.func.arg == ConstNode(:c1)
        inserted = child_a.tree.root.arg
        @test inserted isa ConstNode
        @test haskey(child_a.local_consts, inserted.name)
        @test child_a.local_consts[inserted.name] == 9
        @test infer(child_a.tree, merge(global_types, child_a.local_consts_ty)) == IntTy
        @test infer(child_b.tree, merge(global_types, child_b.local_consts_ty)) == IntTy
    end
end

@testitem "Populations crossover builds finite pairs once and rejects no-ops before copying" begin
    using Test
    using Random
    import LambdaRegression.Mutations.Populations:
        crossover_pair_weight, crossover_site_weight, crossover_site_path
    using LambdaRegression.Mutations.Populations:
        AbstractCrossoverSelectionPolicy, CrossoverSite, Program,
        TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    mutable struct CountingRootPolicy <: AbstractCrossoverSelectionPolicy
        pair_calls::Int
    end
    crossover_site_weight(
        ::CountingRootPolicy,
        site::CrossoverSite,
        ::Symbol,
    ) = (isempty(crossover_site_path(site)) || crossover_site_path(site) == [1]) ? 1.0 : 0.0
    function crossover_pair_weight(
        policy::CountingRootPolicy,
        left::CrossoverSite,
        right::CrossoverSite,
    )
        policy.pair_calls += 1
        return crossover_site_path(left) == crossover_site_path(right) ? 1.0 : 0.0
    end

    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
    )
    body = AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:one))
    parent_a = Program(AstTree(AbsNode(:left, deepcopy(body))))
    parent_b = Program(AstTree(AbsNode(:right, deepcopy(body))))
    policy = CountingRootPolicy(0)
    factory_calls = Ref(0)
    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=20,
            min_sites=1,
            selection=policy,
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(555),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
        const_symfactory_ctor=() -> begin
            factory_calls[] += 1
            ConstSymbolFactory(prefix=:c)
        end,
    )

    @test result isa Err
    @test policy.pair_calls == 2
    @test factory_calls[] == 0
end

@testitem "Populations crossover rejects both children transactionally" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :inc => TyFun(IntTy, IntTy),
        :consume_bool => TyFun(TyFun(BoolTy, IntTy), IntTy),
        :one => IntTy,
    )
    parent_a = Program(
        AstTree(AbsNode(:x, AppNode(ConstNode(:inc), IndexNode(0)))),
    )
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(
                    AppNode(ConstNode(:add), IndexNode(0)),
                    AppNode(
                        ConstNode(:consume_bool),
                        AbsNode(:flag, ConstNode(:one)),
                    ),
                ),
            ),
        ),
    )
    before_a = deepcopy(parent_a)
    before_b = deepcopy(parent_b)

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            require_compatible_binders=false,
            selection=PathCrossoverSelectionPolicy([1], [1, 2, 2, 1]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(2000),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test result isa Err
    @test parent_a.tree == before_a.tree
    @test parent_b.tree == before_b.tree
    @test parent_a.local_consts == before_a.local_consts
    @test parent_b.local_consts == before_b.local_consts
    @test parent_a.local_consts_ty == before_a.local_consts_ty
    @test parent_b.local_consts_ty == before_b.local_consts_ty
end

@testitem "Populations crossover is deterministic for a fixed seed" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    target_ty = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    parent_a = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(
                    AppNode(ConstNode(:add), IndexNode(0)),
                    ConstNode(:left),
                ),
            ),
        ),
        local_consts=Dict{Symbol,Any}(:left => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:left => IntTy),
    )
    parent_b = Program(
        AstTree(
            AbsNode(
                :x,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:right)),
                    IndexNode(0),
                ),
            ),
        ),
        local_consts=Dict{Symbol,Any}(:right => 2),
        local_consts_ty=Dict{Symbol,TyLike}(:right => IntTy),
    )
    op = TypedSubtreeSwapCrossover(max_try=20)
    run_once() = crossover(
        op,
        parent_a,
        parent_b;
        rng=MersenneTwister(1),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    first = run_once()
    second = run_once()
    @test !(first isa Err)
    @test !(second isa Err)
    if !(first isa Err) && !(second isa Err)
        @test first[1].tree == second[1].tree
        @test first[2].tree == second[2].tree
        @test first[1].local_consts == second[1].local_consts
        @test first[2].local_consts == second[2].local_consts
        @test first[1].local_consts_ty == second[1].local_consts_ty
        @test first[2].local_consts_ty == second[2].local_consts_ty
    end
end

@testitem "Populations crossover exchanges higher-order captured subtrees" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible
    using LambdaRegression.Utils: Err

    int_function = TyFun(IntTy, IntTy)
    target_ty = TyFun(int_function, int_function)
    global_types = Dict{Symbol,TyLike}(:inc => int_function)
    parent_a = Program(
        AstTree(
            AbsNode(
                :f,
                AbsNode(
                    :x,
                    AppNode(
                        IndexNode(1),
                        AppNode(ConstNode(:inc), IndexNode(0)),
                    ),
                ),
            ),
        ),
    )
    parent_b = Program(
        AstTree(
            AbsNode(
                :f,
                AbsNode(
                    :x,
                    AppNode(
                        ConstNode(:inc),
                        AppNode(IndexNode(1), IndexNode(0)),
                    ),
                ),
            ),
        ),
    )

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy([1, 1, 2], [1, 1, 2]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1111),
        global_consts_ty=global_types,
        target_a=target_ty,
        target_b=target_ty,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test child_a.tree != parent_a.tree
        @test child_b.tree != parent_b.tree
        @test iscompatible(infer(child_a.tree, global_types), target_ty)
        @test iscompatible(infer(child_b.tree, global_types), target_ty)
    end
end

@testitem "Populations crossover accepts equirecursive site exchange" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations:
        Program, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyLike, TyMu, infer, iscompatible, make_tyvar_factory
    using LambdaRegression.Utils: Err

    fresh = make_tyvar_factory()
    one_binder = fresh(0)
    period_one = TyMu(one_binder, TyFun(IntTy, one_binder))
    two_binder = fresh(0)
    period_two = TyMu(
        two_binder,
        TyFun(IntTy, TyFun(IntTy, two_binder)),
    )
    global_types = Dict{Symbol,TyLike}(
        :period_one => period_one,
        :period_two => period_two,
    )
    parent_a = Program(AstTree(ConstNode(:period_one)))
    parent_b = Program(AstTree(ConstNode(:period_two)))

    result = crossover(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy(Int[], Int[]),
        ),
        parent_a,
        parent_b;
        rng=MersenneTwister(1234),
        global_consts_ty=global_types,
        target_a=period_one,
        target_b=period_two,
    )

    @test !(result isa Err)
    if !(result isa Err)
        child_a, child_b = result
        @test iscompatible(infer(child_a.tree, global_types), period_one)
        @test iscompatible(infer(child_b.tree, global_types), period_two)
    end
end

@testitem "Populations crossover keeps FunctionDef self captures role-safe" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, FunctionDefNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        Program, CrossoverSite, TypedSubtreeSwapCrossover,
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_path
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    target = TyFun(TyFun(IntTy, IntTy), TyFun(IntTy, IntTy))
    self_parent = Program(AstTree(AbsNode(
        :h,
        FunctionDefNode(AbsNode(:x, AppNode(IndexNode(1), IndexNode(0)))),
    )))
    ordinary_parent = Program(AstTree(AbsNode(
        :h,
        AbsNode(:k, AppNode(IndexNode(1), IndexNode(0))),
    )))
    empty = Dict{Symbol,TyLike}()
    self_analysis = prepare_typed_analysis(self_parent, empty; target_ty=target)
    ordinary_analysis = prepare_typed_analysis(ordinary_parent, empty; target_ty=target)
    @test !(self_analysis isa Err)
    @test !(ordinary_analysis isa Err)
    if !(self_analysis isa Err || ordinary_analysis isa Err)
        self_id = only(filter(
            id -> typed_analysis_path(self_analysis, id) == (1, 1, 1, 1),
            typed_analysis_site_ids(self_analysis),
        ))
        ordinary_id = only(filter(
            id -> typed_analysis_path(ordinary_analysis, id) == (1, 1, 1),
            typed_analysis_site_ids(ordinary_analysis),
        ))
        self_site = CrossoverSite(self_analysis, self_id)
        ordinary_site = CrossoverSite(ordinary_analysis, ordinary_id)
        for require_compatible_binders in (false, true)
            op = TypedSubtreeSwapCrossover(
                require_same_depth=false,
                require_compatible_binders=require_compatible_binders,
            )
            @test !LambdaRegression.Mutations.Populations._captures_fit(
                op, self_site, ordinary_site,
            )
            @test !LambdaRegression.Mutations.Populations._captures_fit(
                op, ordinary_site, self_site,
            )
        end
    end
end

@testitem "Populations IndependentCrossover returns both directed children" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Mutations.Populations:
        Program, IndependentCrossover, PathCrossoverSelectionPolicy,
        TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.Utils: Err

    left = Program(AstTree(ConstNode(:left)))
    right = Program(AstTree(ConstNode(:right)))
    op = IndependentCrossover(TypedSubtreeSwapCrossover(
        max_try=1,
        min_sites=1,
        allow_root=true,
        selection=PathCrossoverSelectionPolicy(Int[], Int[]),
    ))
    result = crossover(
        op,
        left,
        right;
        rng=MersenneTwister(42),
        global_consts_ty=Dict{Symbol,TyLike}(:left => IntTy, :right => IntTy),
        target_a=IntTy,
        target_b=IntTy,
    )
    @test !(result isa Err)
    if !(result isa Err)
        child_left, child_right = result
        @test child_left.tree == right.tree
        @test child_right.tree == left.tree
        @test left.tree == AstTree(ConstNode(:left))
        @test right.tree == AstTree(ConstNode(:right))
    end
end
