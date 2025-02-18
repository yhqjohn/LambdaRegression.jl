using Random

const Populations = Mutations.Populations

"""
    FunctionDefTypedSubtreeSwapCrossover(; max_try=40, min_sites=2,
        allow_root=true, allow_capture_rebinding=true, prefer_closed=true)

Typed subtree swap for trees containing `FunctionDefNode`. Closed subtrees may
move across arbitrary lexical depths. An open subtree may be rebound only when
each captured De Bruijn position exists at the destination with the same binder
kind; both offspring are then re-inferred transactionally.

This is the compatibility backend used only when callers inject a custom
procedural operator. Calls require `global_consts_ty`; missing context,
insufficient eligible sites, exhausted finite pairs, and the `max_try` budget
return typed `Err` values. Invalid constructor budgets throw `ArgumentError`.

`ScopeNode` is a transient IR, not a population crossover representation, and
deliberately has no child-context method in this operator.
"""
struct FunctionDefTypedSubtreeSwapCrossover <: Populations.AbstractCrossover
    max_try::Int
    min_sites::Int
    allow_root::Bool
    allow_capture_rebinding::Bool
    prefer_closed::Bool

    function FunctionDefTypedSubtreeSwapCrossover(
        max_try::Int,
        min_sites::Int,
        allow_root::Bool,
        allow_capture_rebinding::Bool,
        prefer_closed::Bool,
    )
        max_try >= 1 || throw(ArgumentError("max_try must be positive"))
        min_sites >= 1 || throw(ArgumentError("min_sites must be positive"))
        return new(
            max_try,
            min_sites,
            allow_root,
            allow_capture_rebinding,
            prefer_closed,
        )
    end
end

FunctionDefTypedSubtreeSwapCrossover(;
    max_try::Int=40,
    min_sites::Int=2,
    allow_root::Bool=true,
    allow_capture_rebinding::Bool=true,
    prefer_closed::Bool=true,
) = FunctionDefTypedSubtreeSwapCrossover(
    max_try,
    min_sites,
    allow_root,
    allow_capture_rebinding,
    prefer_closed,
)

"""
    AutoTypedCrossover(; core, procedural)

Binary adapter over directional typed recombination. The default path uses the
same authoritative closed/open retrieval engine for core and `FunctionDef`
trees, while retaining `procedural` as a compatibility backend for custom
operators. It produces two independently sampled directed children; callers
that require one reciprocal site pair can still use the concrete binary
`crossover` operator directly. Calls require the same `global_consts_ty`
context as the selected backend, and forward its classified finite-search
failures. `ScopeNode` has no population route.
"""
struct AutoTypedCrossover{C<:Populations.AbstractCrossover,P<:Populations.AbstractCrossover} <:
       Populations.AbstractCrossover
    core::C
    procedural::P
end

AutoTypedCrossover(;
    core=Populations.TypedSubtreeSwapCrossover(require_same_depth=false),
    procedural=FunctionDefTypedSubtreeSwapCrossover(),
) = AutoTypedCrossover(core, procedural)

function _functiondef_directional_operator(
    core::Populations.TypedSubtreeSwapCrossover,
    procedural::FunctionDefTypedSubtreeSwapCrossover,
)
    return Populations.TypedSubtreeSwapCrossover(
        max_try=procedural.max_try,
        max_open_answers_per_pair=core.max_open_answers_per_pair,
        require_same_depth=core.require_same_depth,
        require_compatible_binders=core.require_compatible_binders,
        min_sites=procedural.min_sites,
        allow_root=procedural.allow_root,
        selection=core.selection,
    )
end

struct _FunctionDefCrossoverSite
    path::Vector{Int}
    node::AstNode
    ty::Ty
    binders::Tuple
    captures::Vector{Int}
end

function _contains_functiondef_node(node::AstNode)
    node isa Procedural.FunctionDefNode && return true
    return any(_contains_functiondef_node, children(node))
end

# Crossover owns a deliberately narrower lexical protocol than the procedural
# IR. Core binders and FunctionDef participate; ScopeNode intentionally has no
# method and therefore cannot silently enter the population representation.
function _crossover_child_binders(
    node::AppNode,
    child_index::Int;
    path::Tuple=(),
    outer::Tuple=(),
)
    child_index in (1, 2) || throw(BoundsError(children(node), child_index))
    return outer
end

function _crossover_child_binders(
    node::AbsNode,
    child_index::Int;
    path::Tuple=(),
    outer::Tuple=(),
)
    child_index == 1 || throw(BoundsError(children(node), child_index))
    return (Procedural.BinderId(:abs, path, 1), outer...)
end

function _crossover_child_binders(
    node::LetNode,
    child_index::Int;
    path::Tuple=(),
    outer::Tuple=(),
)
    child_index in (1, 2) || throw(BoundsError(children(node), child_index))
    child_index == 1 && return outer
    return (Procedural.BinderId(:let, path, 1), outer...)
end

function _crossover_child_binders(
    node::Procedural.FunctionDefNode,
    child_index::Int;
    path::Tuple=(),
    outer::Tuple=(),
)
    child_index == 1 || throw(BoundsError(children(node), child_index))
    return (Procedural.BinderId(:function_def, path, 1), outer...)
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
        introduced = length(_crossover_child_binders(
            node,
            child_index;
            path=(),
            outer=(),
        ))
        _captured_positions!(result, child, local_depth + introduced)
    end
    return result
end

function _collect_functiondef_sites(
    tree::AstTree,
    type_ctx::AbstractDict{Symbol,<:TyLike},
    tyfactory,
)
    meta = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    inferred = Procedural.infer_procedural(tree, type_ctx; factory=tyfactory, meta)
    inferred isa Err && return inferred
    sites = _FunctionDefCrossoverSite[]

    function visit(node::AstNode, path::Vector{Int}, binders::Tuple)
        if haskey(meta, node)
            ty = TypeSystem.prune(meta[node][1])
            if ty isa Ty
                captures = sort!(collect(_captured_positions!(Set{Int}(), node)))
                all(position -> position < length(binders), captures) && push!(
                    sites,
                    _FunctionDefCrossoverSite(copy(path), node, ty, binders, captures),
                )
            end
        end
        for (child_index, child) in enumerate(children(node))
            child_binders = _crossover_child_binders(
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
    donor::_FunctionDefCrossoverSite,
    destination::_FunctionDefCrossoverSite,
)
    for position in donor.captures
        position < length(destination.binders) || return false
        donor.binders[position + 1].kind ===
            destination.binders[position + 1].kind || return false
    end
    return true
end

function _functiondef_sites_match(
    op::FunctionDefTypedSubtreeSwapCrossover,
    left::_FunctionDefCrossoverSite,
    right::_FunctionDefCrossoverSite,
)
    TypeSystem.iscompatible(left.ty, right.ty) || return false
    TypeSystem.iscompatible(right.ty, left.ty) || return false
    if !op.allow_capture_rebinding
        isempty(left.captures) && isempty(right.captures) || return false
    end
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
    op::FunctionDefTypedSubtreeSwapCrossover,
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

    sites_a = _collect_functiondef_sites(
        a.tree,
        merge(global_consts_ty, a.local_consts_ty),
        tyfactory_ctor(),
    )
    sites_a isa Err && return sites_a
    sites_b = _collect_functiondef_sites(
        b.tree,
        merge(global_consts_ty, b.local_consts_ty),
        tyfactory_ctor(),
    )
    sites_b isa Err && return sites_b
    op.allow_root || (sites_a = filter(site -> !isempty(site.path), sites_a))
    op.allow_root || (sites_b = filter(site -> !isempty(site.path), sites_b))
    length(sites_a) >= op.min_sites || return Err(Populations.NoEligibleCrossoverSiteErr())
    length(sites_b) >= op.min_sites || return Err(Populations.NoEligibleCrossoverSiteErr())

    pairs = Tuple{_FunctionDefCrossoverSite,_FunctionDefCrossoverSite}[]
    for site_a in sites_a, site_b in sites_b
        _functiondef_sites_match(op, site_a, site_b) && push!(pairs, (site_a, site_b))
    end
    isempty(pairs) && return Err(Populations.NoEligibleCrossoverPairErr())
    if op.prefer_closed
        closed = filter(pair -> isempty(pair[1].captures) && isempty(pair[2].captures), pairs)
        open = filter(pair -> !isempty(pair[1].captures) || !isempty(pair[2].captures), pairs)
        shuffle!(rng, closed)
        shuffle!(rng, open)
        pairs = vcat(closed, open)
    else
        shuffle!(rng, pairs)
    end

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
    has_functiondef = _contains_functiondef_node(a.tree.root) ||
        _contains_functiondef_node(b.tree.root)
    if op.core isa Populations.TypedSubtreeSwapCrossover &&
       (!has_functiondef ||
        op.procedural isa FunctionDefTypedSubtreeSwapCrossover)
        Populations._same_target_type(a.target_ty, b.target_ty) ||
            return Err(Populations.TargetTypeMismatchErr(a.target_ty, b.target_ty))

        core = op.core::Populations.TypedSubtreeSwapCrossover
        formal = if has_functiondef
            procedural = op.procedural::FunctionDefTypedSubtreeSwapCrossover
            _functiondef_directional_operator(core, procedural)
        else
            core
        end
        child_a = Populations.recombine(
            formal,
            a,
            b;
            selection_roles=(:left, :right),
            kwargs...,
        )
        child_a isa Err && return child_a
        child_b = Populations.recombine(
            formal,
            b,
            a;
            selection_roles=(:right, :left),
            kwargs...,
        )
        child_b isa Err && return child_b
        return (child_a, child_b)
    end

    backend = has_functiondef ? op.procedural : op.core
    return Populations.crossover(backend, a, b; kwargs...)
end
