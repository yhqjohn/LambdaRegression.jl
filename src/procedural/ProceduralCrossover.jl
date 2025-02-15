using Random

const Populations = Mutations.Populations

"""
    ProceduralTypedSubtreeSwapCrossover(; max_try=40, min_sites=2, allow_root=true)

Binder-aware typed subtree swap for trees containing `ScopeNode` or
`FunctionDefNode`. Sites are analyzed directly with procedural inference. A
captured De Bruijn position may move only to a binder of the same lexical kind;
both offspring are then re-inferred transactionally.
"""
struct ProceduralTypedSubtreeSwapCrossover <: Populations.AbstractCrossover
    max_try::Int
    min_sites::Int
    allow_root::Bool

    function ProceduralTypedSubtreeSwapCrossover(
        max_try::Int,
        min_sites::Int,
        allow_root::Bool,
    )
        max_try >= 1 || throw(ArgumentError("max_try must be positive"))
        min_sites >= 1 || throw(ArgumentError("min_sites must be positive"))
        return new(max_try, min_sites, allow_root)
    end
end

ProceduralTypedSubtreeSwapCrossover(;
    max_try::Int=40,
    min_sites::Int=2,
    allow_root::Bool=true,
) = ProceduralTypedSubtreeSwapCrossover(max_try, min_sites, allow_root)

"""
    AutoTypedCrossover(; core, procedural)

Preserve the core `TypedSubtreeSwapCrossover` contract while automatically using
the procedural operator whenever either parent contains a procedural binder.
"""
struct AutoTypedCrossover{C<:Populations.AbstractCrossover,P<:Populations.AbstractCrossover} <:
       Populations.AbstractCrossover
    core::C
    procedural::P
end

AutoTypedCrossover(;
    core=Populations.TypedSubtreeSwapCrossover(),
    procedural=ProceduralTypedSubtreeSwapCrossover(),
) = AutoTypedCrossover(core, procedural)

struct _ProceduralCrossoverSite
    path::Vector{Int}
    node::AstNode
    ty::Ty
    binders::Tuple
    captures::Vector{Int}
end

function _contains_procedural_node(node::AstNode)
    node isa Procedural.ScopeNode && return true
    node isa Procedural.FunctionDefNode && return true
    return any(_contains_procedural_node, children(node))
end

function _captured_positions!(
    result::Set{Int},
    node::AstNode,
    local_depth::Int=0,
)
    if node isa IndexNode
        node.index >= local_depth && push!(result, node.index - local_depth)
        return result
    end
    for (child_index, child) in enumerate(children(node))
        introduced = length(Procedural.visible_binders(
            node,
            child_index;
            path=(),
            outer=(),
        ))
        _captured_positions!(result, child, local_depth + introduced)
    end
    return result
end

function _collect_procedural_sites(
    tree::AstTree,
    type_ctx::AbstractDict{Symbol,<:TyLike},
    tyfactory,
)
    meta = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    inferred = Procedural.infer_procedural(tree, type_ctx; factory=tyfactory, meta)
    inferred isa Err && return inferred
    sites = _ProceduralCrossoverSite[]

    function visit(node::AstNode, path::Vector{Int}, binders::Tuple)
        if haskey(meta, node)
            ty = TypeSystem.prune(meta[node][1])
            if ty isa Ty
                captures = sort!(collect(_captured_positions!(Set{Int}(), node)))
                all(position -> position < length(binders), captures) && push!(
                    sites,
                    _ProceduralCrossoverSite(copy(path), node, ty, binders, captures),
                )
            end
        end
        for (child_index, child) in enumerate(children(node))
            child_binders = Procedural.visible_binders(
                node,
                child_index;
                path=Tuple(path),
                outer=binders,
            )
            visit(child, [path; child_index], child_binders)
        end
    end

    visit(tree.root, Int[], ())
    return sites
end

function _captured_kinds_fit(
    donor::_ProceduralCrossoverSite,
    destination::_ProceduralCrossoverSite,
)
    for position in donor.captures
        position < length(destination.binders) || return false
        donor.binders[position + 1].kind ===
            destination.binders[position + 1].kind || return false
    end
    return true
end

function _procedural_sites_match(left::_ProceduralCrossoverSite, right::_ProceduralCrossoverSite)
    length(left.binders) == length(right.binders) || return false
    TypeSystem.iscompatible(left.ty, right.ty) || return false
    TypeSystem.iscompatible(right.ty, left.ty) || return false
    return _captured_kinds_fit(left, right) && _captured_kinds_fit(right, left)
end

function _used_local_payloads(node, values, types)
    result = Dict{Symbol,Tuple{Any,TyLike}}()
    function visit(current)
        if current isa ConstNode
            name = current.name
            haskey(values, name) && haskey(types, name) &&
                (result[name] = (values[name], types[name]))
        end
        foreach(visit, children(current))
    end
    visit(node)
    return result
end

function _same_procedural_contextual_subtree(site_a, a, site_b, b)
    site_a.node == site_b.node || return false
    payloads_a = _used_local_payloads(site_a.node, a.local_consts, a.local_consts_ty)
    payloads_b = _used_local_payloads(site_b.node, b.local_consts, b.local_consts_ty)
    keys(payloads_a) == keys(payloads_b) || return false
    for name in keys(payloads_a)
        value_a, ty_a = payloads_a[name]
        value_b, ty_b = payloads_b[name]
        isequal(value_a, value_b) || return false
        TypeSystem.iscompatible(ty_a, ty_b) || return false
        TypeSystem.iscompatible(ty_b, ty_a) || return false
    end
    return true
end

function Populations.crossover(
    op::ProceduralTypedSubtreeSwapCrossover,
    a::Populations.Individual,
    b::Populations.Individual;
    rng::AbstractRNG=MersenneTwister(2000),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    tyfactory_ctor=TypeSystem.make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    prune::Bool=true,
)
    Populations._same_target_type(a.target_ty, b.target_ty) ||
        return Err(Populations.TargetTypeMismatchErr(a.target_ty, b.target_ty))
    global_consts_ty === nothing && return Err(Populations.MissingGlobalTypeCtxErr())

    sites_a = _collect_procedural_sites(
        a.tree,
        merge(global_consts_ty, a.local_consts_ty),
        tyfactory_ctor(),
    )
    sites_a isa Err && return sites_a
    sites_b = _collect_procedural_sites(
        b.tree,
        merge(global_consts_ty, b.local_consts_ty),
        tyfactory_ctor(),
    )
    sites_b isa Err && return sites_b
    op.allow_root || (sites_a = filter(site -> !isempty(site.path), sites_a))
    op.allow_root || (sites_b = filter(site -> !isempty(site.path), sites_b))
    length(sites_a) >= op.min_sites || return Err(Populations.NoEligibleCrossoverSiteErr())
    length(sites_b) >= op.min_sites || return Err(Populations.NoEligibleCrossoverSiteErr())

    pairs = Tuple{_ProceduralCrossoverSite,_ProceduralCrossoverSite}[]
    for site_a in sites_a, site_b in sites_b
        _procedural_sites_match(site_a, site_b) && push!(pairs, (site_a, site_b))
    end
    isempty(pairs) && return Err(Populations.NoEligibleCrossoverPairErr())
    shuffle!(rng, pairs)

    attempts = 0
    for (site_a, site_b) in Iterators.take(pairs, op.max_try)
        attempts += 1
        _same_procedural_contextual_subtree(site_a, a, site_b, b) && continue

        tree_a, tree_b = deepcopy(a.tree), deepcopy(b.tree)
        local_a, local_b = copy(a.local_consts), copy(b.local_consts)
        local_a_ty, local_b_ty = copy(a.local_consts_ty), copy(b.local_consts_ty)
        node_a = Populations._node_at_path_maybe(tree_a, site_a.path)
        node_b = Populations._node_at_path_maybe(tree_b, site_b.path)
        node_a === nothing && continue
        node_b === nothing && continue
        subtree_a, subtree_b = AstTree(deepcopy(node_a)), AstTree(deepcopy(node_b))

        factory_a, factory_b = const_symfactory_ctor(), const_symfactory_ctor()
        Populations._separate_local_global_namespace!(
            tree_a, local_a, local_a_ty, global_consts_ty, factory_a,
        )
        Populations._separate_local_global_namespace!(
            tree_b, local_b, local_b_ty, global_consts_ty, factory_b,
        )
        Populations._remap_donor_locals!(
            subtree_b, b.local_consts, b.local_consts_ty,
            local_a, local_a_ty, global_consts_ty, factory_a,
        )
        Populations._remap_donor_locals!(
            subtree_a, a.local_consts, a.local_consts_ty,
            local_b, local_b_ty, global_consts_ty, factory_b,
        )
        Populations._replace_at_path_maybe!(tree_a, site_a.path, subtree_b) || continue
        Populations._replace_at_path_maybe!(tree_b, site_b.path, subtree_a) || continue

        if prune
            Mutations.prune_local_consts!(tree_a, local_a, local_a_ty)
            Mutations.prune_local_consts!(tree_b, local_b, local_b_ty)
        end
        inferred_a = Procedural.infer_procedural(
            tree_a,
            merge(global_consts_ty, local_a_ty);
            factory=tyfactory_ctor(),
        )
        inferred_a isa Err && continue
        TypeSystem.iscompatible(inferred_a, a.target_ty) || continue
        inferred_b = Procedural.infer_procedural(
            tree_b,
            merge(global_consts_ty, local_b_ty);
            factory=tyfactory_ctor(),
        )
        inferred_b isa Err && continue
        TypeSystem.iscompatible(inferred_b, b.target_ty) || continue

        child_a = Populations.Individual(
            tree_a, a.target_ty;
            local_consts=local_a, local_consts_ty=local_a_ty,
        )
        child_b = Populations.Individual(
            tree_b, b.target_ty;
            local_consts=local_b, local_consts_ty=local_b_ty,
        )
        return (child_a, child_b)
    end
    attempts == length(pairs) &&
        return Err(Populations.CrossoverPairPoolExhaustedErr(attempts))
    return Err(Populations.CrossoverMaxTryReachedErr(op.max_try))
end

function Populations.crossover(
    op::AutoTypedCrossover,
    a::Populations.Individual,
    b::Populations.Individual;
    kwargs...,
)
    backend = _contains_procedural_node(a.tree.root) ||
        _contains_procedural_node(b.tree.root) ? op.procedural : op.core
    return Populations.crossover(backend, a, b; kwargs...)
end
