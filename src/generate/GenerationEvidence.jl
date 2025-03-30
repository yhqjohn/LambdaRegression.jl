"""
    generation_catalog_scope_invariant(provider) -> Bool

Declare that a provider's core catalogue depends on the fixed type universe and
external context, not the order/multiplicity of concrete scoped binders or the
current target. The finite evidence engine prepares that catalogue once and
keeps actual binder choices in replay/sampling. Custom providers default to
`false`; declaring `true` requires finite, closed catalogue construction and
the usual context/capability protocol. This is not a declaration about extension
options, mutable cache safety, or recursive-type support.
"""
generation_catalog_scope_invariant(::AbstractHeadProvider) = false
generation_catalog_scope_invariant(::ContextHeadProvider) = true
generation_catalog_scope_invariant(::NilFactoryProvider) = true
generation_catalog_scope_invariant(::TypedValueFactoryProvider) = true
generation_catalog_scope_invariant(::ToListProvider) = true
generation_catalog_scope_invariant(::AMapProvider) = true
generation_catalog_scope_invariant(::GuardedExpansionProvider) = true
generation_catalog_scope_invariant(::SafeListAccessProvider) = true
generation_catalog_scope_invariant(::WeightTagProvider) = true

# Keys contain only runtime integers. Source limit -1 denotes an ordinary fact;
# a nonnegative limit denotes the existing anchored source-array policy.
# The last coordinate bounds preparation along scheme-instance paths only;
# -1 keeps the fixed monomorphic world (and the type-witness contract) unbounded.
const _EvidenceFactKey = NTuple{5,Int}

struct _EvidenceEdge
    parent::Int
    kind::Symbol
    value::Any
    children::Vector{Int}
    charge::Int
end

struct _EvidencePlan
    kind::Symbol
    value::Any
    children::Vector{Int}
    charge::Int
    weight::Float64
end

# A syntactic option, not a proof. Child facts are registered and saturated
# only after selection, or when a consumer explicitly requests a certified menu.
struct _EvidenceStep
    kind::Symbol
    value::Any
    children::Vector{_EvidenceFactKey}
    charge::Int
    weight::Float64
end

struct _UnsupportedEvidenceSampling end

struct _EvidenceCatalogState
    heads::Vector{HeadEntry}
    rules::Vector{CoreRule}
end

mutable struct _GenerationEvidence
    types::Vector{Ty}
    type_ids::Dict{_StructuralKey,Int}
    object_ids::IdDict{Ty,Int}
    contexts::Vector{Vector{Int}}
    context_ids::Dict{_StructuralKey,Int}
    catalog::Union{Nothing,CoreCatalog}
    root_target::Ty
    root_scope::Vector{Ty}
    heads_by_type::Vector{Vector{HeadEntry}}
    rules_by_type::Vector{Vector{CoreRule}}
    fact_ids::Dict{_EvidenceFactKey,Int}
    facts::Vector{_EvidenceFactKey}
    costs::Vector{Int}
    certificates::Vector{Int}
    edges::Vector{_EvidenceEdge}
    dependents::Vector{Vector{Int}}
    expanded::BitVector
    pending::Vector{Int}
    pending_cursor::Int
    updates::Vector{Int}
    rule_checks::Int
    menus::Dict{Tuple{Int,Bool},Vector{_EvidencePlan}}
    steps::Dict{Tuple{Int,Bool},Vector{_EvidenceStep}}
    sampling::Bool
    layered::Bool
    root_scope_types::Vector{Int}
    catalogs::Dict{Tuple{Int,Int},_EvidenceCatalogState}
    obligations::Dict{_StructuralKey,Any}
    unsupported_sampling::Bool
end

_evidence_has_schemes(relation::CoreRelation) = any(
    context -> any(ty -> ty isa TyScheme, values(context)),
    (relation.global_consts_ty, relation.local_consts_ty))

function _finite_evidence_request_supported(relation::CoreRelation, query::GenerationQuery)
    _in_closed_fragment(query.target_ty) || return false
    _scope_all(_in_closed_fragment, query.scope_tys) || return false
    all(generation_catalog_scope_invariant, relation.providers) || return false
    for context in (relation.global_consts_ty, relation.local_consts_ty)
        for ty in values(context)
            # Budget-layered child catalogues preserve newly admitted closed
            # instances without repeatedly closing all schemes over the type
            # registry. The unbounded scheme case has no such finite proof.
            if ty isa TyScheme
                query.depth_left >= 0 || return false
                continue
            end
            ty isa Ty && _in_closed_fragment(ty) || return false
        end
    end
    isempty(relation.presets) && isempty(relation.preset_types) || return false
    return true
end

function _new_generation_evidence(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty}; sampling::Bool=false)
    # AMap's target-specific catalogue is an old preparation shortcut. Preparing
    # all instances over this one finite universe keeps child mapper/source
    # demands available without extending the universe at every lambda.
    layered = _evidence_has_schemes(relation)
    universe = CoreUniverse(target_ty;
        global_consts_ty=relation.global_consts_ty, local_consts_ty=relation.local_consts_ty,
        presets=relation.presets, preset_types=relation.preset_types,
        scope_source_tys=scope_tys)
    catalog = layered ? nothing : _build_core_catalog(universe;
        _precollected_inputs=true,
        global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty,
        local_consts=relation.local_consts, local_consts_ty=relation.local_consts_ty,
        presets=relation.presets, preset_types=relation.preset_types,
        scope_source_tys=scope_tys, providers=_core_catalog_providers(relation.providers),
        catalog_target_ty=layered ? target_ty : nothing)
    world = _GenerationEvidence(Ty[], Dict{_StructuralKey,Int}(), IdDict{Ty,Int}(),
        Vector{Int}[], Dict{_StructuralKey,Int}(), catalog, target_ty, copy(scope_tys),
        Vector{HeadEntry}[], Vector{CoreRule}[], Dict{_EvidenceFactKey,Int}(),
        _EvidenceFactKey[], Int[], Int[], _EvidenceEdge[], Vector{Int}[],
        BitVector(), Int[], 1, Int[], 0, Dict{Tuple{Int,Bool},Vector{_EvidencePlan}}(),
        Dict{Tuple{Int,Bool},Vector{_EvidenceStep}}(), sampling,
        layered, Int[], Dict{Tuple{Int,Int},_EvidenceCatalogState}(),
        Dict{_StructuralKey,Any}(), false)
    for ty in (layered ? universe.types : catalog.universe.types)
        _evidence_register_type!(world, ty)
    end
    # Register the original lexical domain before any fact is interned. In a
    # layered world the registry is seeded, not closed over scheme instances.
    _evidence_register_type!(world, target_ty)
    for ty in scope_tys
        _evidence_register_type!(world, ty)
    end
    append!(world.root_scope_types, unique!([_evidence_type_id(world, ty) for ty in scope_tys]))
    # Every layered fact prepares its own exact instance domain below. An
    # unused whole-root catalogue would duplicate that work; legacy consumers
    # explicitly request it through _evidence_root_catalog! instead.
    layered && return world
    # Finite head types also contain residual arrow types and scoped domains.
    for head in catalog.heads
        head.type isa Ty && _evidence_register_type!(world, head.type)
    end
    # A provider may introduce a closed premise absent from the initial
    # universe. Prepare leaf factories for those already finite premise types.
    # Layered scheme worlds instead obtain their leaves in each demand's
    # catalogue; neither path closes schemes over the cumulative registry.
    leaf_universe = CoreUniverse(copy(world.types),
        Set{Any}(keys(world.type_ids)), sort!(collect(_collect_lengths(world.types))))
    leaf_heads = HeadEntry[]
    for provider in relation.providers
        provider isa Union{NilFactoryProvider,TypedValueFactoryProvider} || continue
        add_heads!(provider, leaf_heads, leaf_universe)
    end
    _apply_weight_tag_providers!(leaf_heads, leaf_universe, relation.providers)
    _filter_rigid_heads!(leaf_heads)
    all_heads = [catalog.heads; leaf_heads]
    for head in leaf_heads
        head.type isa Ty && _evidence_register_type!(world, head.type)
    end
    for head in all_heads
        head.type isa Ty || continue
        head.rule_class isa CoreProductive || continue
        _known_head_admitted(relation, head) || continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        sampling && _head_weight(relation.weights, head) <= 0 && continue
        id = _evidence_type_id(world, head.type)
        id == 0 || push!(world.heads_by_type[id], head)
    end
    for head in all_heads
        head.type isa Ty && head.rule_class isa CoreProductive || continue
        _known_head_admitted(relation, head) || continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        sampling && _head_weight(relation.weights, head) <= 0 && continue
        args, result = _split_fun(head.type)
        isempty(args) && continue
        id = _evidence_type_id(world, result)
        id == 0 || push!(world.rules_by_type[id],
            CoreRule(head, result, args, [_argument_class(arg) for arg in args]))
    end
    return world
end

"""Prepare the original root catalogue, not a discovered child layer's domain.

Only preparation-time consumers may call this. Root scope includes forbidden
binders; the cumulative type registry is never an instance domain.
"""
function _evidence_root_catalog!(world::_GenerationEvidence, relation::CoreRelation)
    catalog = world.catalog
    catalog === nothing || return catalog
    universe = CoreUniverse(world.root_target;
        global_consts_ty=relation.global_consts_ty, local_consts_ty=relation.local_consts_ty,
        presets=relation.presets, preset_types=relation.preset_types,
        scope_source_tys=world.root_scope)
    catalog = _build_core_catalog(universe;
        _precollected_inputs=true,
        global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty,
        local_consts=relation.local_consts, local_consts_ty=relation.local_consts_ty,
        presets=relation.presets, preset_types=relation.preset_types,
        scope_source_tys=world.root_scope, providers=_core_catalog_providers(relation.providers),
        catalog_target_ty=world.layered ? world.root_target : nothing)
    world.catalog = catalog
    return catalog
end

# Catalogue scope invariance does not exclude a provider-local argument policy.
# Known built-ins cannot attach one; unknown providers retain the full original
# root preflight, even if an unrelated head carries the policy.
_evidence_heads_policy_free(::AbstractHeadProvider) = false
_evidence_heads_policy_free(::Union{ContextHeadProvider,NilFactoryProvider,
    TypedValueFactoryProvider,ToListProvider,AMapProvider,GuardedExpansionProvider,
    SafeListAccessProvider,WeightTagProvider}) = true

# Early certificate preparation additionally needs no late catalogue-wide
# veto (for example an unsupported custom multi-array source rule). This is
# stronger than merely having no local argument policy.
_evidence_early_certificate_supported(::AbstractHeadProvider) = false
_evidence_early_certificate_supported(::Union{ContextHeadProvider,NilFactoryProvider,
    TypedValueFactoryProvider,ToListProvider,AMapProvider,GuardedExpansionProvider,
    SafeListAccessProvider,WeightTagProvider}) = true

function _evidence_has_local_argument_policy(world::_GenerationEvidence, relation::CoreRelation)
    all(_evidence_heads_policy_free, relation.providers) && return false
    return any(head -> _metadata_get(head.metadata, :argument_planner, nothing) !== nothing,
        _evidence_root_catalog!(world, relation).heads)
end

_evidence_catalog_domain_sharing(::AbstractHeadProvider) = false
_evidence_catalog_domain_sharing(::Union{ContextHeadProvider,NilFactoryProvider,
    ToListProvider,AMapProvider,GuardedExpansionProvider,SafeListAccessProvider,
    WeightTagProvider}) = true
_evidence_catalog_domain_sharing(provider::TypedValueFactoryProvider) =
    all(factory -> factory isa Union{ValueFactories.NilFactory,
        ValueFactories.FloatScalarFactory,ValueFactories.IntScalarFactory,
        ValueFactories.FloatArrayFactory,ValueFactories.IntArrayFactory,
        ValueFactories.BoolArrayFactory}, provider.factories)

function _evidence_catalog_context!(world::_GenerationEvidence, context::Int)
    available = world.contexts[context]
    all(id -> id in available, world.root_scope_types) && return context
    return _evidence_context!(world, Int[world.root_scope_types; available])
end

function _evidence_catalog_state(world::_GenerationEvidence, relation::CoreRelation,
    context::Int, ty_id::Int)
    world.layered || return _EvidenceCatalogState(world.heads_by_type[ty_id], world.rules_by_type[ty_id])
    # The catalogue domain includes original forbidden binders. Canonicalise
    # that exact domain, not the fact's usable variables or the type registry.
    # This shares preparation only; facts and concrete obligations stay apart.
    catalog_context = all(_evidence_catalog_domain_sharing, relation.providers) ?
        _evidence_catalog_context!(world, context) : context
    return get!(world.catalogs, (catalog_context, ty_id)) do
        # The registry is NOT an instantiation domain. Each local catalogue
        # receives just this demand, its lexical type availability, and the
        # fixed external context. Original forbidden binders remain in the
        # catalogue domain, but not in the usable variable set.
        ids = sort!(unique!(Int[world.root_scope_types; world.contexts[context]]))
        scope = Ty[world.types[id] for id in ids]
        target = world.types[ty_id]
        # Do not borrow relation.catalogs: its legacy key has no external
        # context version. This world is already versioned (or query-private),
        # so its own catalogue cache is the correct ownership boundary.
        universe = CoreUniverse(target;
            global_consts_ty=relation.global_consts_ty,
            local_consts_ty=relation.local_consts_ty,
            presets=relation.presets, preset_types=relation.preset_types,
            scope_source_tys=scope)
        prepared_universe, prepared_heads = _build_core_heads(universe;
            _precollected_inputs=true,
            global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty,
            local_consts=relation.local_consts, local_consts_ty=relation.local_consts_ty,
            presets=relation.presets, preset_types=relation.preset_types,
            scope_source_tys=scope, providers=_core_catalog_providers(relation.providers),
            catalog_target_ty=target)
        for ty in prepared_universe.types
            _evidence_register_type!(world, ty)
        end
        heads = HeadEntry[]
        rules = CoreRule[]
        for head in prepared_heads
            head.type isa Ty || continue
            _evidence_register_type!(world, head.type)
            head.rule_class isa CoreProductive || continue
            _known_head_admitted(relation, head) || continue
            _bucket_enabled(relation.weights, head.weight_bucket) || continue
            _head_has_value(head, relation) || continue
            world.sampling && _head_weight(relation.weights, head) <= 0 && continue
            _metadata_get(head.metadata, :argument_planner, nothing) === nothing ||
                (world.unsupported_sampling = true)
            _evidence_type_id(world, head.type) == ty_id && push!(heads, head)
            args, result = _split_fun(head.type)
            isempty(args) && continue
            _evidence_type_id(world, result) == ty_id || continue
            push!(rules, CoreRule(head, result, args, [_argument_class(arg) for arg in args]))
        end
        _EvidenceCatalogState(heads, rules)
    end
end

function _evidence_child_source_depth(world::_GenerationEvidence, rule::CoreRule,
    source_depth::Int, source_index::Int, index::Int)
    array_arg = rule.argument_classes[index] == :source_array
    if source_depth < 0
        return array_arg ? _source_array_depth_limit(rule) : -1
    end
    if world.sampling
        # The existing default source sampler filters children by this head's
        # policy after its raw recursion allowance. This equivalence covers
        # the supported single-array source chain, not arbitrary multi-sources.
        return array_arg ? min(source_depth - 1, _source_array_depth_limit(rule)) : -1
    end
    # The type-witness interface uses its existing designated-source contract.
    return index == source_index && array_arg ? source_depth - 1 : -1
end

function _evidence_source_rule_supported!(world::_GenerationEvidence, rule::CoreRule)
    world.sampling || return true
    source_index = _source_array_argument_index(rule.head, length(rule.argument_tys))
    if count(==(:source_array), rule.argument_classes) != 1 ||
       rule.argument_classes[source_index] != :source_array
        world.unsupported_sampling = true
        return false
    end
    return true
end

_evidence_child_budget(budget::Int, charge::Int) = budget < 0 ? -1 : budget - charge

function _evidence_child_depth(world::_GenerationEvidence, ordinary_depth::Int,
    source_depth::Int, budget::Int)
    # Entering source syntax is independent of the parent's ordinary depth.
    # Scheme paths still decrease the original available node balance at every
    # constructor, even if ordinary/source syntax alternates repeatedly.
    if world.sampling && source_depth >= 0
        return world.layered ? budget : -1
    end
    return ordinary_depth
end

function _evidence_register_type!(world::_GenerationEvidence, ty::Ty)
    _in_closed_fragment(ty) || return 0
    get(world.object_ids, ty, 0) != 0 && return world.object_ids[ty]
    key = _StructuralKey(typekey(ty))
    id = get(world.type_ids, key, 0)
    if id == 0
        push!(world.types, ty)
        push!(world.heads_by_type, HeadEntry[])
        push!(world.rules_by_type, CoreRule[])
        id = length(world.types)
        world.type_ids[key] = id
        if ty isa TyFun
            _evidence_register_type!(world, ty.arg)
            _evidence_register_type!(world, ty.res)
        elseif ty isa Union{TyList,TyArray}
            _evidence_register_type!(world, ty.elem)
        end
    end
    world.object_ids[ty] = id
    return id
end

function _evidence_type_id(world::_GenerationEvidence, ty::Ty)
    id = get(world.object_ids, ty, 0)
    id != 0 && return id
    id = get(world.type_ids, _StructuralKey(typekey(ty)), 0)
    id != 0 && (world.object_ids[ty] = id)
    return id
end

function _evidence_context!(world::_GenerationEvidence, ids::Vector{Int})
    ids = sort!(unique!(copy(ids)))
    key = _StructuralKey(Tuple(ids))
    return get!(world.context_ids, key) do
        push!(world.contexts, ids)
        length(world.contexts)
    end
end

function _evidence_fact!(world::_GenerationEvidence, context::Int, ty::Int,
    depth::Int, source_depth::Int=-1, budget::Int=-1)
    ty == 0 && return 0
    budget < -1 && return 0
    key = (context, ty, depth, source_depth, budget)
    return get!(world.fact_ids, key) do
        push!(world.facts, key)
        push!(world.costs, typemax(Int))
        push!(world.certificates, 0)
        push!(world.dependents, Int[])
        push!(world.expanded, false)
        id = length(world.facts)
        push!(world.pending, id)
        id
    end
end

function _evidence_add_edge!(world::_GenerationEvidence, parent::Int,
    kind::Symbol, value, children::Vector{Int}, charge::Int)
    any(==(0), children) && return
    push!(world.edges, _EvidenceEdge(parent, kind, value, children, charge))
    edge_id = length(world.edges)
    for child in children
        push!(world.dependents[child], edge_id)
    end
    _evidence_relax!(world, edge_id)
end

function _evidence_relax!(world::_GenerationEvidence, edge_id::Int)
    world.rule_checks += 1
    edge = world.edges[edge_id]
    cost = edge.charge
    for child in edge.children
        child_cost = world.costs[child]
        child_cost == typemax(Int) && return
        child_cost > typemax(Int) - cost && return
        cost += child_cost
    end
    budget = world.facts[edge.parent][5]
    budget >= 0 && cost > budget && return
    cost < world.costs[edge.parent] || return
    world.costs[edge.parent] = cost
    world.certificates[edge.parent] = edge_id
    push!(world.updates, edge.parent)
end

function _evidence_expand!(world::_GenerationEvidence, relation::CoreRelation, fact::Int)
    world.expanded[fact] && return
    world.expanded[fact] = true
    context, ty_id, depth, source_depth, budget = world.facts[fact]
    budget == 0 && return
    ty = world.types[ty_id]
    available = world.contexts[context]
    source = source_depth >= 0
    if _bucket_enabled(relation.weights, _VARIABLE_WEIGHT) || (source && !world.sampling)
        if ty_id in available
            _evidence_add_edge!(world, fact, :variable, ty_id, Int[], 1)
        end
    end
    world.costs[fact] == 1 && return
    catalog_state = _evidence_catalog_state(world, relation, context, ty_id)
    for head in catalog_state.heads
        source && _is_instance_factory_head(head) && continue
        _evidence_add_edge!(world, fact, :constant, head, Int[], 1)
    end
    # A direct witness is minimum-cost. Alternative expression routes belong
    # to the sampler, not to this existence/cost judgement.
    world.costs[fact] == 1 && return
    depth == 0 && return
    next_depth = depth < 0 ? depth : depth - 1
    if !source && ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        arg_id = _evidence_type_id(world, ty.arg)
        body_context = _evidence_context!(world, Int[available; arg_id])
        child = _evidence_fact!(world, body_context, _evidence_type_id(world, ty.res),
            next_depth, -1, _evidence_child_budget(budget, 1))
        _evidence_add_edge!(world, fact, :lambda, ty.arg, Int[child], 1)
    end
    if !source && _scope_application_enabled(relation.weights)
        scope = Ty[world.types[id] for id in available]
        limit = depth < 0 ? maximum(length(_split_fun(t)[1]) for t in scope; init=0) : depth
        for rule in _scope_application_rules(ty, scope, limit)
            _known_type_admitted(relation, rule.head_ty) || continue
            argument_depth = depth < 0 ? depth :
                depth - (world.sampling ? 1 : length(rule.argument_tys))
            argument_depth < 0 && depth >= 0 && continue
            child_budget = _evidence_child_budget(budget, length(rule.argument_tys))
            child_budget < 0 && budget >= 0 && continue
            children = Int[]
            for (arg_ty, arg_class) in zip(rule.argument_tys, rule.argument_classes)
                array_arg = arg_class == :source_array
                array_arg && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && break
                child_source = array_arg ? 0 : -1
                child_depth = _evidence_child_depth(world, argument_depth, child_source, child_budget)
                push!(children, _evidence_fact!(world, context,
                    _evidence_type_id(world, arg_ty), child_depth, child_source, child_budget))
            end
            length(children) == length(rule.argument_tys) || continue
            _evidence_add_edge!(world, fact, :scope_application, rule.head_ty,
                children, length(children))
        end
    end
    source && source_depth == 0 && return
    for rule in catalog_state.rules
        isempty(rule.argument_tys) && continue
        source && !_is_source_array_constructor(rule.head) && continue
        source && !_evidence_source_rule_supported!(world, rule) && continue
        children = Int[]
        source_index = source ? _source_array_argument_index(rule.head, length(rule.argument_tys)) : 0
        for (index, arg_ty) in enumerate(rule.argument_tys)
            array_arg = rule.argument_classes[index] == :source_array
            array_arg && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && break
            child_source = _evidence_child_source_depth(world, rule, source_depth, source_index, index)
            child_budget = _evidence_child_budget(budget, 1)
            child_depth = _evidence_child_depth(world, next_depth, child_source, child_budget)
            push!(children, _evidence_fact!(world, context,
                _evidence_type_id(world, arg_ty), child_depth, child_source, child_budget))
        end
        length(children) == length(rule.argument_tys) || continue
        _evidence_add_edge!(world, fact, :head, rule.head, children, 1)
    end
end

function _evidence_drain_updates!(world::_GenerationEvidence, graph=nothing)
    index = 1
    while index <= length(world.updates)
        fact = world.updates[index]
        for edge_id in world.dependents[fact]
            _evidence_relax!(world, edge_id)
        end
        graph === nothing || _obligation_sync_ordinary!(graph, fact)
        index += 1
    end
    empty!(world.updates)
end

function _evidence_saturate!(world::_GenerationEvidence, relation::CoreRelation)
    # Discover dependencies without DFS failure memoisation. A cycle is a set
    # of pending facts until independently grounded; it is not a negative fact.
    while world.pending_cursor <= length(world.pending)
        fact = world.pending[world.pending_cursor]
        world.pending_cursor += 1
        _evidence_expand!(world, relation, fact)
    end
    empty!(world.pending)
    world.pending_cursor = 1
    _evidence_drain_updates!(world)
    # A full ordinary consumer may resume a world suspended by a joint replay
    # consumer. Refresh its existing zero-mask subscriptions before reuse.
    for (graph, _) in values(world.obligations)
        _obligation_refresh_zero!(graph)
        _obligation_drain_updates!(graph)
    end
    return world
end

function _evidence_replay(world::_GenerationEvidence, fact::Int,
    scope::Vector{Ty}, forbidden::Tuple=())
    edge_id = world.certificates[fact]
    edge_id == 0 && return nothing
    edge = world.edges[edge_id]
    binder = -1
    if edge.kind == :variable || edge.kind == :scope_application
        binder_ty = edge.kind == :variable ? world.types[edge.value] : edge.value
        index = findfirst(eachindex(scope)) do slot
            slot - 1 in forbidden && return false
            _evidence_type_id(world, scope[slot]) == _evidence_type_id(world, binder_ty)
        end
        index === nothing && error("finite evidence replay lost its concrete binder")
        binder = index - 1
    end
    return _evidence_replay_step(world, edge, binder, scope, forbidden)
end

"""Replay one selected, concrete root step and its already certified children."""
function _evidence_replay_step(world::_GenerationEvidence, step, binder::Int,
    scope::Vector{Ty}, forbidden::Tuple)
    step.kind == :variable && return VariableDerivation(binder, scope[binder + 1])
    step.kind == :constant && return ConstantDerivation(step.value)
    lambda = step.kind == :lambda
    child_scope = lambda ? Ty[step.value; scope] : scope
    child_forbidden = lambda ? _shift_scope_binders(forbidden, 1) : forbidden
    args = CoreDerivation[_evidence_replay(world, child, child_scope, child_forbidden)
        for child in step.children]
    lambda && return LambdaDerivation(step.value, only(args))
    step.kind == :scope_application &&
        return ScopeApplicationDerivation(binder, scope[binder + 1], args)
    return HeadApplicationDerivation(step.value, args)
end

function _finite_evidence_context_key(relation::CoreRelation)
    isempty(relation.presets) && isempty(relation.preset_types) || return nothing
    global_key = _cacheable_global_consts_key(relation.global_consts, relation.global_consts_ty)
    global_key === nothing && return nothing
    local_key = _cacheable_local_context_key(relation.local_consts, relation.local_consts_ty)
    local_key === nothing && return nothing
    provider_key = _cacheable_providers_key(relation.providers)
    provider_key === nothing && return nothing
    return (global_key, local_key, provider_key, _weights_key(relation.weights),
        relation.capability_mask)
end

function _finite_evidence_world(relation::CoreRelation, target_ty::Ty,
    scope::Vector{Ty}; reuse::Bool=true, sampling::Bool=false)
    # Check this mutable context once, here, not first in each caller and then
    # again to form the world key. A missing version is not a reusable identity.
    # Private calls need neither a context fingerprint nor a scope cache key.
    context_version = reuse ? _finite_evidence_context_key(relation) : nothing
    context_version === nothing && return _new_generation_evidence(
        relation, target_ty, scope; sampling)
    scope_keys = sort!(unique!([_StructuralKey(typekey(ty)) for ty in scope]); by=key -> repr(key.value))
    world_key = _StructuralKey((context_version, typekey(target_ty), Tuple(scope_keys), sampling))
    world = get!(relation.evidence_worlds, world_key) do
        _new_generation_evidence(relation, target_ty, scope; sampling)
    end
    return world
end

function _finite_evidence_witness(relation::CoreRelation, query::GenerationQuery; reuse::Bool=true)
    scope = _scope_type_vector(query.scope_tys)
    forbidden = _forbidden_scope_binders(query)
    if _bucket_enabled(relation.weights, _VARIABLE_WEIGHT)
        for (slot, ty) in enumerate(scope)
            slot - 1 in forbidden && continue
            _grounded_iscompatible(ty, query.target_ty) || continue
            return VariableDerivation(slot - 1, ty)
        end
    end
    world = _finite_evidence_world(relation, query.target_ty, scope; reuse)
    available = Int[_evidence_type_id(world, ty) for (slot, ty) in enumerate(scope)
        if slot - 1 ∉ forbidden]
    context = _evidence_context!(world, available)
    fact = _evidence_fact!(world, context, _evidence_type_id(world, query.target_ty), query.depth_left)
    _evidence_saturate!(world, relation)
    # A budget miss is not non-inhabitation. The public wrapper caches this
    # minimum-cost witness with its cost, then applies each caller's budget.
    world.certificates[fact] == 0 && return nothing
    return _evidence_replay(world, fact, scope, forbidden)
end

function _evidence_steps!(world::_GenerationEvidence, relation::CoreRelation,
    fact::Int; allow_heads::Bool=true)
    key = (fact, allow_heads)
    haskey(world.steps, key) && return world.steps[key]
    context, ty_id, depth, source_depth, budget = world.facts[fact]
    budget == 0 && return _EvidenceStep[]
    ty = world.types[ty_id]
    available = world.contexts[context]
    source = source_depth >= 0
    plans = _EvidenceStep[]
    catalog_state = _evidence_catalog_state(world, relation, context, ty_id)
    if ty_id in available && _bucket_enabled(relation.weights, _VARIABLE_WEIGHT)
        push!(plans, _EvidenceStep(:variable, ty_id, _EvidenceFactKey[], 1,
            _bucket_weight(relation.weights, _VARIABLE_WEIGHT)))
    end
    for head in catalog_state.heads
        source && _is_instance_factory_head(head) && continue
        push!(plans, _EvidenceStep(:constant, head, _EvidenceFactKey[], 1, _head_weight(relation.weights, head)))
    end
    if depth != 0
        next_depth = depth < 0 ? depth : depth - 1
        if !source && ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
            body_context = _evidence_context!(world,
                Int[available; _evidence_type_id(world, ty.arg)])
            child = (body_context, _evidence_type_id(world, ty.res),
                next_depth, -1, _evidence_child_budget(budget, 1))
            push!(plans, _EvidenceStep(:lambda, ty.arg, _EvidenceFactKey[child], 1,
                _bucket_weight(relation.weights, _LAMBDA_WEIGHT)))
        end
        if allow_heads && !source && _scope_application_enabled(relation.weights)
            scope = Ty[world.types[id] for id in available]
            limit = depth < 0 ? maximum(length(_split_fun(t)[1]) for t in scope; init=0) : depth
            for rule in _scope_application_rules(ty, scope, limit)
                _known_type_admitted(relation, rule.head_ty) || continue
                child_depth = depth < 0 ? depth :
                    depth - (world.sampling ? 1 : length(rule.argument_tys))
                child_depth < 0 && depth >= 0 && continue
                child_budget = _evidence_child_budget(budget, length(rule.argument_tys))
                child_budget < 0 && budget >= 0 && continue
                children = _EvidenceFactKey[]
                for (arg_ty, arg_class) in zip(rule.argument_tys, rule.argument_classes)
                    array_arg = arg_class == :source_array
                    array_arg && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && break
                    child_source = array_arg ? 0 : -1
                    argument_depth = _evidence_child_depth(world, child_depth, child_source, child_budget)
                    push!(children, (context,
                        _evidence_type_id(world, arg_ty), argument_depth, child_source, child_budget))
                end
                length(children) == length(rule.argument_tys) || continue
                push!(plans, _EvidenceStep(:scope_application, rule.head_ty, children,
                    length(children), _scope_application_option_weight(relation, rule)))
            end
        end
        if allow_heads && (!source || source_depth > 0)
            for rule in catalog_state.rules
                source && !_is_source_array_constructor(rule.head) && continue
                source && !_evidence_source_rule_supported!(world, rule) && continue
                children = _EvidenceFactKey[]
                source_index = source ? _source_array_argument_index(rule.head, length(rule.argument_tys)) : 0
                for (index, arg_ty) in enumerate(rule.argument_tys)
                    array_arg = rule.argument_classes[index] == :source_array
                    array_arg && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && break
                    child_source = _evidence_child_source_depth(world, rule, source_depth, source_index, index)
                    child_budget = _evidence_child_budget(budget, 1)
                    child_depth = _evidence_child_depth(world, next_depth, child_source, child_budget)
                    push!(children, (context,
                        _evidence_type_id(world, arg_ty), child_depth, child_source, child_budget))
                end
                length(children) == length(rule.argument_tys) || continue
                push!(plans, _EvidenceStep(:head, rule.head, children, 1, _head_option_weight(relation, rule)))
            end
        end
    end
    filter!(plans) do plan
        plan.weight > 0 && all(child -> child[2] != 0 && child[5] >= -1, plan.children)
    end
    world.steps[key] = plans
    return plans
end

function _evidence_prepare_step!(world::_GenerationEvidence, relation::CoreRelation,
    step::_EvidenceStep)
    children = Int[_evidence_fact!(world, child...) for child in step.children]
    _evidence_saturate!(world, relation)
    return _EvidencePlan(step.kind, step.value, children, step.charge, step.weight)
end

# Joint obligations and policy-restricted completion need a certified menu.
# Ordinary exploration instead prepares only its selected _EvidenceStep.
function _evidence_menu!(world::_GenerationEvidence, relation::CoreRelation,
    fact::Int; allow_heads::Bool=true)
    key = (fact, allow_heads)
    haskey(world.menus, key) && return world.menus[key]
    steps = _evidence_steps!(world, relation, fact; allow_heads)
    plans = _EvidencePlan[_EvidencePlan(step.kind, step.value,
        Int[_evidence_fact!(world, child...) for child in step.children],
        step.charge, step.weight) for step in steps]
    _evidence_saturate!(world, relation)
    budget = world.facts[fact][5]
    filter!(plans) do plan
        all(child -> child != 0 && world.certificates[child] != 0, plan.children) &&
            (budget < 0 || _evidence_plan_cost(world, plan) <= budget)
    end
    world.menus[key] = plans
    return plans
end

function _evidence_plan_cost(world::_GenerationEvidence, plan::_EvidencePlan)
    cost = plan.charge
    for child in plan.children
        child_cost = world.costs[child]
        child_cost > typemax(Int) - cost && return typemax(Int)
        cost += child_cost
    end
    return cost
end

function _completion_boundary!(state::DerivationSamplingState)
    mode = state.completion
    mode.completing && return true
    mode.boundaries += 1
    if mode.grace_left > 0
        mode.grace_left -= 1
    elseif mode.probability > 0 && rand(state.rng) < mode.probability
        mode.completing = true
    end
    return mode.completing
end

function _evidence_concrete_options(world::_GenerationEvidence, plans,
    scope::Vector{Ty}, forbidden::Tuple)
    options = NamedTuple[]
    for plan in plans
        if plan.kind == :variable || plan.kind == :scope_application
            ty_id = plan.kind == :variable ? plan.value : _evidence_type_id(world, plan.value)
            for (slot, ty) in enumerate(scope)
                slot - 1 in forbidden && continue
                _evidence_type_id(world, ty) == ty_id || continue
                push!(options, (plan=plan, binder=slot - 1, weight=plan.weight))
            end
        else
            push!(options, (plan=plan, binder=-1, weight=plan.weight))
        end
    end
    return options
end

function _sample_evidence_fact!(state::DerivationSamplingState, world::_GenerationEvidence,
    relation::CoreRelation, fact::Int, scope::Vector{Ty}, forbidden::Tuple=();
    allow_heads::Bool=true, certificate_ready::Bool=false)
    certificate_ready && world.costs[fact] > state.nodes_left && return nothing
    # All recursive children and the public root have already been saturated.
    # Replay must not resume proof search after the irreversible switch.
    if state.completion.completing && certificate_ready && allow_heads
        _consume_sampling_nodes!(state, world.costs[fact]) || return nothing
        return _evidence_replay(world, fact, scope, forbidden)
    end
    if state.completion.completing
        # An inherited completed mode cannot prepare a missing policy proof.
        # This explicit unsupported boundary still needs root-level integration
        # in consumers that enter with no compatible certificate prepared.
        plans = get(world.menus, (fact, allow_heads), nothing)
        plans === nothing && return _UnsupportedEvidenceSampling()
        options = _evidence_concrete_options(world, plans, scope, forbidden)
        filter!(option -> _evidence_plan_cost(world, option.plan) <= state.nodes_left, options)
        isempty(options) && return nothing
        chosen = argmin(option -> _evidence_plan_cost(world, option.plan), options)
        return _sample_evidence_plan!(state, world, relation, chosen.plan,
            chosen.binder, scope, forbidden)
    end
    steps = _evidence_steps!(world, relation, fact; allow_heads)
    world.unsupported_sampling && return _UnsupportedEvidenceSampling()
    options = _evidence_concrete_options(world, steps, scope, forbidden)
    # A necessary lower bound only; no unproved option is labelled negative.
    filter!(option -> option.plan.charge + length(option.plan.children) <= state.nodes_left, options)
    options, total_weight = _positive_option_pool(options)
    isempty(options) && return nothing

    # The ordinary root certificate may use a head excluded by this call's
    # policy. Prepare a policy-compatible certificate BEFORE switching modes.
    completion_options = NamedTuple[]
    mode = state.completion
    if !(certificate_ready && allow_heads) &&
        (mode.completing || (mode.probability > 0 && mode.grace_left == 0))
        plans = _evidence_menu!(world, relation, fact; allow_heads)
        world.unsupported_sampling && return _UnsupportedEvidenceSampling()
        completion_options = _evidence_concrete_options(world, plans, scope, forbidden)
        filter!(option -> _evidence_plan_cost(world, option.plan) <= state.nodes_left, completion_options)
        isempty(completion_options) && return nothing
    end
    if _completion_boundary!(state)
        if certificate_ready && allow_heads
            _consume_sampling_nodes!(state, world.costs[fact]) || return nothing
            return _evidence_replay(world, fact, scope, forbidden)
        end
        chosen = argmin(option -> _evidence_plan_cost(world, option.plan), completion_options)
        return _sample_evidence_plan!(state, world, relation, chosen.plan,
            chosen.binder, scope, forbidden)
    end

    # Rejections modify only this request's finite option pool. Saturating the
    # selected premises obtains minimum costs (not merely a first expensive
    # witness), so small legal terms cannot be rejected by an overestimate.
    while !isempty(options)
        index = _weighted_option_index(state.rng, options, total_weight)
        chosen = options[index]
        plan = _evidence_prepare_step!(world, relation, chosen.plan)
        world.unsupported_sampling && return _UnsupportedEvidenceSampling()
        if all(child -> child != 0 && world.certificates[child] != 0, plan.children) &&
            _evidence_plan_cost(world, plan) <= state.nodes_left
            return _sample_evidence_plan!(state, world, relation, plan,
                chosen.binder, scope, forbidden)
        end
        deleteat!(options, index)
        total_weight = sum(option -> option.weight, options; init=0.0)
    end
    return nothing
end

function _sample_evidence_plan!(state::DerivationSamplingState, world::_GenerationEvidence,
    relation::CoreRelation, plan::_EvidencePlan, binder::Int, scope::Vector{Ty}, forbidden::Tuple)
    _consume_sampling_nodes!(state, plan.charge) || return nothing
    if plan.kind == :variable
        return VariableDerivation(binder, scope[binder + 1])
    elseif plan.kind == :constant
        return ConstantDerivation(plan.value)
    end
    args = CoreDerivation[]
    reserved = sum(child -> world.costs[child], plan.children; init=0)
    for child in plan.children
        reserved -= world.costs[child]
        state.nodes_left -= reserved
        child_scope = plan.kind == :lambda ? Ty[plan.value; scope] : scope
        child_forbidden = plan.kind == :lambda ? _shift_scope_binders(forbidden, 1) : forbidden
        arg = _sample_evidence_fact!(state, world, relation, child, child_scope,
            child_forbidden; certificate_ready=true)
        state.nodes_left += reserved
        arg isa _UnsupportedEvidenceSampling && return arg
        arg === nothing && error("certified core plan failed to fill a budget-reserved child")
        push!(args, arg)
    end
    if plan.kind == :lambda
        return LambdaDerivation(plan.value, only(args))
    elseif plan.kind == :scope_application
        return ScopeApplicationDerivation(binder, scope[binder + 1], args)
    end
    return HeadApplicationDerivation(plan.value, args)
end

function _prepare_finite_core_certificate(relation::CoreRelation, target_ty::Ty,
    scope::Vector{Ty}, depth::Int, node_budget::Int;
    source_depth::Int=-1, forbidden::Tuple=())
    _evidence_has_schemes(relation) && depth < 0 && return _UnsupportedEvidenceSampling()
    world = _finite_evidence_world(relation, target_ty, scope; sampling=true)
    # Scope-invariant catalogue preparation is not permission to ignore a
    # provider's local argument policy. Keep that separate bounded interface
    # until a plan-aware replacement has been explicitly provided.
    _evidence_has_local_argument_policy(world, relation) && return _UnsupportedEvidenceSampling()
    available = Int[_evidence_type_id(world, ty) for (slot, ty) in enumerate(scope)
        if slot - 1 ∉ forbidden]
    context = _evidence_context!(world, available)
    fact = _evidence_fact!(world, context, _evidence_type_id(world, target_ty), depth, source_depth,
        world.layered ? node_budget : -1)
    _evidence_saturate!(world, relation)
    world.unsupported_sampling && return _UnsupportedEvidenceSampling()
    world.certificates[fact] == 0 && return nothing
    return (; world, fact)
end

function _sample_finite_core_derivation(state::DerivationSamplingState,
    relation::CoreRelation, target_ty::Ty, scope::Vector{Ty}, depth::Int;
    allow_heads::Bool=true, source_depth::Int=-1, forbidden::Tuple=())
    prepared = _prepare_finite_core_certificate(relation, target_ty, scope,
        depth, state.nodes_left; source_depth, forbidden)
    prepared isa _UnsupportedEvidenceSampling && return prepared
    prepared === nothing && return nothing
    (; world, fact) = prepared
    saved_nodes = state.nodes_left
    sampled = _sample_evidence_fact!(state, world, relation, fact, scope, forbidden;
        allow_heads, certificate_ready=true)
    sampled isa _UnsupportedEvidenceSampling && (state.nodes_left = saved_nodes)
    return sampled
end

"""Prepare certified finite-core root plans excluding one bare variable.

The anchoring algorithm reuses the finite evidence world and certified one-step
menu; only a root `VariableDerivation(excluded_root)` is removed. The supplied
closed target, ordered scope, depth, node budget and zero-based root slot define
the request. `forbidden` remains a separate global binder policy, so the
excluded binder may still occur in a child. A ready result contains every
budget-compatible certified root plan with its proven node cost. A `:budget`
result reports a certified plan that does not fit; `:unavailable` means no
compatible finite-core root certificate was prepared within this request, not
that a mixed language request is impossible. Unsupported core policies retain
the usual `_UnsupportedEvidenceSampling` marker.
"""
function _prepare_finite_core_nonbare_certificate(relation::CoreRelation,
    target_ty::Ty, scope::Vector{Ty}, depth::Int, node_budget::Int,
    excluded_root::Int; allow_heads::Bool=true, source_depth::Int=-1,
    forbidden::Tuple=())
    node_budget >= -1 || throw(ArgumentError("node budget must be -1 or nonnegative"))
    0 <= excluded_root < length(scope) || throw(BoundsError(scope, excluded_root + 1))

    prepared = _prepare_finite_core_certificate(relation, target_ty, scope,
        depth, node_budget; source_depth, forbidden)
    prepared isa _UnsupportedEvidenceSampling && return prepared
    prepared === nothing && return (; status=:unavailable, reason=:no_core_certificate)
    (; world, fact) = prepared

    plans = _evidence_menu!(world, relation, fact; allow_heads)
    world.unsupported_sampling && return _UnsupportedEvidenceSampling()
    options = _evidence_concrete_options(world, plans, scope, forbidden)
    filter!(option -> option.plan.kind != :variable || option.binder != excluded_root,
        options)
    isempty(options) && return (; status=:unavailable, reason=:no_compatible_root_plan)

    candidates = NamedTuple[]
    for option in options
        push!(candidates, (plan=option.plan, binder=option.binder,
            weight=option.weight, cost=_evidence_plan_cost(world, option.plan)))
    end
    eligible = filter(option -> node_budget < 0 || option.cost <= node_budget, candidates)
    isempty(eligible) && return (; status=:budget,
        proven_cost=minimum(option.cost for option in candidates), node_budget)
    return (; status=:ready, world, fact, options=eligible, scope=copy(scope),
        forbidden, excluded_root, allow_heads, node_budget)
end

"""Choose a root step before its parent commits a node lease.

The returned cost is the certified minimum for this selected root and its
children, not necessarily the cost of a subsequent exploratory sample. A
parent reserves at least this amount, then charges the actual constructed
term; it must not substitute the ordinary bare variable's cheaper certificate.
This choice does not construct or mutate a term.
"""
function _select_finite_core_nonbare_root(state::DerivationSamplingState, prepared)
    prepared isa _UnsupportedEvidenceSampling && return prepared
    prepared.status === :ready || return nothing
    options = filter(option -> option.cost <= state.nodes_left, prepared.options)
    isempty(options) && return nothing

    if _completion_boundary!(state)
        return argmin(option -> option.cost, options)
    else
        pool, total_weight = _positive_option_pool(options)
        isempty(pool) && return nothing
        return pool[_weighted_option_index(state.rng, pool, total_weight)]
    end
end

"""Replay a core certificate under the actual sampling policy, without a mode switch.

The query supplies closed types, capability masks, depth/node bounds and
forbidden bindings. Positive occurrence requirements are deliberately NOT
discharged here: the composing caller must account for the returned term's
actual occurrences. Only the default core policy is supported. Returns an
admitted certificate, budget failure, or an unsupported-policy marker; no
negative answer decides a mixed extension request.
"""
function _replay_finite_core_certificate(state::DerivationSamplingState,
    relation::CoreRelation, query::GenerationQuery)
    relation, query = _relation_for_query(relation, query)
    relation.argument_planner isa DefaultArgumentPlanner || return _UnsupportedEvidenceSampling()
    _finite_evidence_request_supported(relation, query) || return _UnsupportedEvidenceSampling()
    any(query.requirements) do requirement
        requirement isa Union{UsesScopeBinder,CallsScopeBinder,ForbidsScopeBinder} && return false
        generation_requirement_affects_core_search(requirement)
    end && return _UnsupportedEvidenceSampling()
    scope = _scope_type_vector(query.scope_tys)
    forbidden = _forbidden_scope_binders(query)
    available = min(state.nodes_left, query.node_budget)
    prepared = _prepare_finite_core_certificate(relation, query.target_ty, scope,
        query.depth_left, available; forbidden)
    prepared isa _UnsupportedEvidenceSampling && return prepared
    prepared === nothing && return nothing
    (; world, fact) = prepared
    cost = world.costs[fact]
    cost <= available || return nothing
    _consume_sampling_nodes!(state, cost) || return nothing
    return _evidence_replay(world, fact, scope, forbidden)
end

"""Sample an anchored array source without enumerating its complete witnesses.

The relation supplies the fixed constants, closed catalogue providers and
default argument policy. `source_depth` is the caller's existing source
allowance; `depth` is the parent's remaining ordinary depth, not a new bound
on the historically independent source grammar. Its finite preparation uses
the fixed monomorphic fact world or the existing available node budget for
scheme-layered preparation; actual sampling still consumes those same nodes.
The parent contributes capability and forbidden-binder constraints, rebased to
the supplied scope. Returns a certified source or `nothing` for an unavailable
or over-budget source. Unknown policies return `_UnsupportedEvidenceSampling`
without retaining partial node charges, so their explicit old route survives.
"""
function _sample_finite_source_array(state::DerivationSamplingState,
    relation::CoreRelation, target::Ty, scope::Vector{Ty}, depth::Int,
    source_depth::Int, parent::Union{Nothing,GenerationQuery}=nothing)
    relation.argument_planner isa DefaultArgumentPlanner || return _UnsupportedEvidenceSampling()
    query = parent === nothing ? GenerationQuery(target; scope_tys=scope,
        depth_left=depth, node_budget=state.nodes_left, capability_mask=relation.capability_mask) :
        _nested_generation_query(state, parent, target, scope, depth, parent.guarded_left)
    restricted, effective = _relation_for_query(relation, query)
    _finite_evidence_request_supported(restricted, effective) || return _UnsupportedEvidenceSampling()
    _bucket_enabled(restricted.weights, _ARRAY_SOURCE_WEIGHT) || return nothing
    available = min(state.nodes_left, effective.node_budget)
    outside_reserve = state.nodes_left - available
    state.nodes_left = available
    try
        preparation_depth = _evidence_has_schemes(restricted) ? available : -1
        return _sample_finite_core_derivation(state, restricted, target, scope,
            preparation_depth; source_depth, forbidden=_forbidden_scope_binders(effective))
    finally
        state.nodes_left += outside_reserve
    end
end

@testitem "CanonicalGeneration anchored source sampling selects plans without witness enumeration" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
        weights=CG.GenerationWeights(amap=8, instance_factory=0))
    source_rule_head = CG.HeadEntry(:TOLIST, :primitive, TyFun(array, TyList(FloatTy)),
        CG.CoreProductive(), CG.WeightBucket(:tolist);
        metadata=(value=LambdaRegression.Primitives.TOLIST, source_array_depth_limit=1))
    rule = CG.CoreRule(source_rule_head, TyList(FloatTy), Ty[array], [:source_array])
    maps = Ref(0)
    for seed in 1:16
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 32)
        source = CG._sample_head_argument(state, relation, rule, 1, Ty[array], 4, 0, false, false)
        @test source !== nothing
        @test CG._array_source_depth(source) <= 1
        @test CG.derivation_node_count(source) == 32 - state.nodes_left
        maps[] += source isa CG.HeadApplicationDerivation
    end
    @test maps[] > 0
    @test isempty(relation.derivations)
    @test isempty(relation.source_derivations)
    query = CG.GenerationQuery(array; scope_tys=Ty[array, array], depth_left=4,
        node_budget=32, requirements=(CG.ForbidsScopeBinder(0),))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 32)
    source = CG._sample_finite_source_array(state, relation, array, Ty[array, array], 4, 1, query)
    @test source !== nothing
    @test !CG.derivation_uses_scope_binder(source, 0)
    @test CG.derivation_uses_scope_binder(source, 1)
end

@testitem "CanonicalGeneration evidence sampling preserves independent source and scoped depths" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyConst, TyFun, TyList, TyArray, TyNat, TyScheme, FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    input = TyArray(TyNat(4), FloatTy)
    output = TyArray(TyNat(4), BoolTy)
    mapper = TyFun(FloatTy, BoolTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
        global_consts=Dict{Symbol,Any}(:predicate => iszero),
        global_consts_ty=Dict{Symbol,Any}(:predicate => mapper))
    parent_head = CG.HeadEntry(:TOLIST, :primitive, TyFun(output, TyList(BoolTy)),
        CG.CoreProductive(), CG.WeightBucket(:tolist);
        metadata=(value=LambdaRegression.Primitives.TOLIST, source_array_depth_limit=1))
    parent_rule = CG.CoreRule(parent_head, TyList(BoolTy), Ty[output], [:source_array])
    # The old source grammar is independent of the parent's ordinary depth.
    # A non-same-type map is forced, and its three-node charge is exact.
    for budget in (2, 3)
        state = CG.DerivationSamplingState(Random.Xoshiro(2000), budget)
        query = CG.GenerationQuery(output; scope_tys=Ty[input], depth_left=0, node_budget=budget)
        result = CG._sample_head_argument(state, relation, parent_rule, 1,
            Ty[input], 0, 0, false, false, query)
        if budget == 2
            @test result === nothing
            @test state.nodes_left == budget
        else
            @test result isa CG.HeadApplicationDerivation
            @test result.head.name == :AMAP
            @test CG.derivation_result_type(result) == output
            @test CG.derivation_node_count(result) == 3
            @test state.nodes_left == 0
            @test CG._array_source_depth(result) == 1
        end
    end
    @test isempty(relation.source_derivations)
    # External schemes need a preparation bound, but must not inherit the
    # parent's depth zero. Use exactly the still-available node balance.
    len, elem = newtyvar(0), newtyvar(0)
    layered = CG.CoreRelation(providers=relation.providers,
        global_consts=Dict{Symbol,Any}(:predicate => iszero,
            :LENF => LambdaRegression.Primitives.LENF),
        global_consts_ty=Dict{Symbol,Any}(:predicate => mapper,
            :LENF => TyScheme([len, elem], TyFun(TyArray(len, elem), FloatTy))))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 3)
    query = CG.GenerationQuery(output; scope_tys=Ty[input], depth_left=0, node_budget=3)
    result = CG._sample_finite_source_array(state, layered, output, Ty[input], 0, 1, query)
    @test result isa CG.HeadApplicationDerivation
    @test CG.derivation_node_count(result) == 3
    @test state.nodes_left == 0
    @test all(world -> all(fact -> fact[3] >= 0, world.facts), values(layered.evidence_worlds))
    @test all(world -> all(fact -> fact[5] >= 0, world.facts), values(layered.evidence_worlds))
    # All instance-preparation edges decrease the inherited node ceiling,
    # including ordinary/source crossings; zero cannot become the -1 sentinel.
    for world in values(layered.evidence_worlds)
        @test all(edge -> all(child -> world.facts[child][5] < world.facts[edge.parent][5],
            edge.children), world.edges)
    end
    zero = CG.DerivationSamplingState(Random.Xoshiro(2000), 0)
    @test CG._sample_finite_core_derivation(zero, layered, output, Ty[output], 0) === nothing
    @test zero.nodes_left == 0

    a, b, c = TyConst(:A), TyConst(:B), TyConst(:C)
    head = TyFun(TyFun(a, a), TyFun(b, c))
    scope = Ty[head, b]
    scoped = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 5)
    result = CG._sample_finite_core_derivation(state, scoped, c, scope, 2)
    @test result isa CG.ScopeApplicationDerivation
    @test result.index == 0
    @test first(result.args) isa CG.LambdaDerivation
    @test last(result.args) isa CG.VariableDerivation
    @test last(result.args).index == 1
    @test CG.derivation_node_count(result) == 5
    @test state.nodes_left == 0
    # The existence helper retains its own historical arity-depth contract.
    @test CG.known_generation_witness(scoped,
        CG.GenerationQuery(c; scope_tys=scope, depth_left=2, node_budget=5)) === nothing
    # With a scheme present, the arity charge must not turn b=1 into the
    # unbounded sentinel b=-1 before child preparation.
    alpha = newtyvar(0)
    capped_scoped = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:unused => identity),
        global_consts_ty=Dict{Symbol,Any}(:unused => TyScheme([alpha], TyFun(alpha, a))))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 1)
    @test CG._sample_finite_core_derivation(state, capped_scoped, c, scope, 2) === nothing
    @test state.nodes_left == 1
    @test all(world -> all(fact -> fact[5] >= 0, world.facts), values(capped_scoped.evidence_worlds))
    # The internal ordinary-to-source edge must implement the same rule; the
    # source helper alone is not enough to cover this list request.
    list_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.ToListProvider(), CG.AMapProvider()),
        global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 4)
    result = CG._sample_finite_core_derivation(state, list_relation, TyList(BoolTy), Ty[input], 1)
    @test result isa CG.HeadApplicationDerivation
    @test result.head.name == :TOLIST
    @test only(result.args) isa CG.HeadApplicationDerivation
    @test only(result.args).head.name == :AMAP
    @test CG.derivation_node_count(result) == 4
    @test state.nodes_left == 0
    limited = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    limited_query = CG.GenerationQuery(output; scope_tys=Ty[input], depth_left=0, node_budget=2)
    @test CG._sample_finite_source_array(limited, relation, output, Ty[input], 0, 1, limited_query) === nothing
    @test limited.nodes_left == 9
end

@testitem "CanonicalGeneration evidence source fallback retains multi-array policies" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    input = TyArray(TyNat(4), FloatTy)
    output = TyArray(TyNat(4), BoolTy)
    constructor = CG.HeadEntry(:JOIN_ARRAY, :test, TyFun(input, TyFun(input, output)),
        CG.CoreProductive(), CG.WeightBucket(:test);
        metadata=(value=((a,b) -> map(iszero, a)), source_array_constructor=true,
            source_array_argument_index=1, source_array_depth_limit=0))
    struct MultipleSourceEvidenceFixture <: CG.AbstractHeadProvider
        head::CG.HeadEntry
    end
    CG.generation_catalog_scope_invariant(::MultipleSourceEvidenceFixture) = true
    CG.add_heads!(provider::MultipleSourceEvidenceFixture, heads, universe; kwargs...) =
        push!(heads, provider.head)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),
        MultipleSourceEvidenceFixture(constructor)))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 3)
    result = CG._sample_finite_source_array(state, relation, output, Ty[input], 0, 1)
    @test result isa CG._UnsupportedEvidenceSampling
    @test state.nodes_left == 3
    parent = CG.HeadEntry(:CONSUME_ARRAY, :test, TyFun(output, BoolTy),
        CG.CoreProductive(), CG.WeightBucket(:test);
        metadata=(value=identity, source_array_depth_limit=1))
    rule = CG.CoreRule(parent, BoolTy, Ty[output], [:source_array])
    result = CG._sample_head_argument(state, relation, rule, 1, Ty[input], 0, 0, false, false)
    @test result isa CG.HeadApplicationDerivation
    @test result.head.name == :JOIN_ARRAY
    @test CG.derivation_node_count(result) == 3
    @test state.nodes_left == 0
    @test !isempty(relation.source_derivations)
end

@testitem "CanonicalGeneration evidence migration preserves polymorphic and local-policy boundaries" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyScheme, FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    alpha, beta = newtyvar(0), newtyvar(0)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:h=>identity, :g=>identity),
        global_consts_ty=Dict{Symbol,Any}(
            :h=>TyScheme([alpha], TyFun(TyFun(TyList(alpha), alpha), BoolTy)),
            :g=>TyScheme([beta], TyFun(beta, BoolTy))))
    query = CG.GenerationQuery(BoolTy; depth_left=4, node_budget=16)
    @test CG._finite_evidence_request_supported(relation, query)
    @test CG.known_generation_witness(relation, query) !== nothing

    # A child binder admits g at List Bool. That instance was absent from the
    # root catalogue and must not leak back into the root's instantiation set.
    world = CG._finite_evidence_world(relation, BoolTy, Ty[])
    root_context = CG._evidence_context!(world, Int[])
    root_type = CG._evidence_type_id(world, BoolTy)
    root_catalog = CG._evidence_catalog_state(world, relation, root_context, root_type)
    g_list = TyFun(TyList(BoolTy), BoolTy)
    @test !any(r -> r.head.name == :g &&
        LambdaRegression.TypeSystem.iscompatible(r.head.type, g_list), root_catalog.rules)
    child_context = CG._evidence_context!(world,
        Int[CG._evidence_type_id(world, TyList(BoolTy))])
    child_catalog = CG._evidence_catalog_state(world, relation, child_context, root_type)
    @test any(r -> r.head.name == :g &&
        LambdaRegression.TypeSystem.iscompatible(r.head.type, g_list), child_catalog.rules)
    @test CG._evidence_catalog_state(world, relation, root_context, root_type) === root_catalog
    @test !any(r -> r.head.name == :g &&
        LambdaRegression.TypeSystem.iscompatible(r.head.type, g_list), root_catalog.rules)
    @test CG._sample_finite_core_derivation(CG.DerivationSamplingState(Random.Xoshiro(1234), 16),
        relation, BoolTy, Ty[], -1) isa CG._UnsupportedEvidenceSampling

    struct RefusingLocalPlanner <: CG.AbstractHeadProvider end
    CG.generation_catalog_scope_invariant(::RefusingLocalPlanner) = true
    function CG.add_heads!(::RefusingLocalPlanner, heads, universe; kwargs...)
        push!(heads, CG.HeadEntry(:seed, :local_policy_test, FloatTy,
            CG.CoreProductive(), CG.WeightBucket(:constant); metadata=(value=1.0,)))
        push!(heads, CG.HeadEntry(:h, :local_policy_test, TyFun(FloatTy, BoolTy),
            CG.CoreProductive(), CG.WeightBucket(:constant);
            metadata=(value=(x->true), argument_planner=(ds,args...)->empty(ds))))
        return heads
    end
    refusing = CG.CoreRelation(providers=(RefusingLocalPlanner(),))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
    @test CG._sample_finite_core_derivation(state, refusing, BoolTy, Ty[], 3) isa
        CG._UnsupportedEvidenceSampling
    @test CG._sample_known_derivation(state, refusing, BoolTy, Ty[], 3) === nothing
end

@testitem "CanonicalGeneration layered scheme evidence has finite preparation and source budgets" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyArray, TyNat, TyScheme, FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    alpha = newtyvar(0)
    loop = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:h=>identity),
        global_consts_ty=Dict{Symbol,Any}(:h=>TyScheme([alpha],
            TyFun(TyFun(TyList(alpha), BoolTy), BoolTy))))
    query = CG.GenerationQuery(BoolTy; depth_left=4, node_budget=64)
    @test CG.known_generation_witness(loop, query) === nothing
    world = CG._finite_evidence_world(loop, BoolTy, Ty[])
    @test world.layered
    @test all(fact -> fact[3] >= 0, world.facts)
    @test length(world.facts) < 100
    list_depth(ty::Ty) = ty isa TyList ? 1 + list_depth(ty.elem) :
        ty isa TyFun ? max(list_depth(ty.arg), list_depth(ty.res)) : 0
    @test maximum(list_depth, world.types) <= 3
    # Registry growth from this negative search does not widen the old root
    # catalogue on a later lookup, regardless of the lookup order.
    context = CG._evidence_context!(world, Int[])
    root = CG._evidence_catalog_state(world, loop, context,
        CG._evidence_type_id(world, BoolTy))
    @test all(r -> list_depth(r.head.type) <= 1, root.rules)

    len, elem = newtyvar(0), newtyvar(0)
    array = TyArray(TyNat(4), BoolTy)
    len_type = TyScheme([len, elem], TyFun(TyArray(len, elem), FloatTy))
    lengths = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:LENF=>LambdaRegression.Primitives.LENF),
        global_consts_ty=Dict{Symbol,Any}(:LENF=>len_type))
    small = CG.GenerationQuery(FloatTy; scope_tys=Ty[array], depth_left=2, node_budget=1)
    large = CG.GenerationQuery(FloatTy; scope_tys=Ty[array], depth_left=2, node_budget=2)
    @test CG.known_generation_witness(lengths, small) === nothing
    witness = CG.known_generation_witness(lengths, large)
    @test witness !== nothing
    @test CG.derivation_node_count(witness) == 2
    @test CG.derivation_uses_scope_binder(witness, 0)
    @test CG.known_generation_witness(lengths,
        CG.GenerationQuery(FloatTy; depth_left=2, node_budget=8)) === nothing
    no_source = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=lengths.global_consts, global_consts_ty=lengths.global_consts_ty,
        weights=CG.GenerationWeights(array_source=0))
    @test CG.known_generation_witness(no_source, large) === nothing
end

@testitem "CanonicalGeneration layered evidence versions mutable constant contexts" begin
    using Test
    using LambdaRegression.TypeSystem: TyFun, TyArray, TyScheme, FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    len, elem = newtyvar(0), newtyvar(0)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:LENF=>LambdaRegression.Primitives.LENF),
        global_consts_ty=Dict{Symbol,Any}(:LENF=>TyScheme([len, elem],
            TyFun(TyArray(len, elem), FloatTy))))
    query = CG.GenerationQuery(BoolTy; depth_left=3, node_budget=16)
    @test CG.known_generation_witness(relation, query) === nothing
    # Deliberately prime the old cache. A new versioned world must not read it.
    CG._relation_catalog(relation, BoolTy, LambdaRegression.TypeSystem.Ty[])
    relation.local_consts[:b] = true
    relation.local_consts_ty[:b] = BoolTy
    witness = CG.known_generation_witness(relation, query)
    @test witness !== nothing
    @test CG.derivation_node_count(witness) == 1
    @test witness.head.name == :b
    delete!(relation.local_consts, :b)
    delete!(relation.local_consts_ty, :b)
    @test CG.known_generation_witness(relation, query) === nothing
end

@testitem "CanonicalGeneration evidence world owns one current cacheability check" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    struct CountedEvidenceProvider <: CG.AbstractHeadProvider
        checks::Base.RefValue{Int}
        cacheable::Bool
    end
    CG.generation_catalog_scope_invariant(::CountedEvidenceProvider) = true
    function CG.generation_provider_cache_key(provider::CountedEvidenceProvider)
        provider.checks[] += 1
        return provider.cacheable ? (:counted_evidence_provider,) : nothing
    end
    CG.add_heads!(::CountedEvidenceProvider, heads, universe; kwargs...) =
        CG.add_heads!(CG.ContextHeadProvider(), heads, universe; kwargs...)

    checks = Ref(0)
    relation = CG.CoreRelation(providers=(CountedEvidenceProvider(checks, true),),
        global_consts=Dict{Symbol,Any}(:flag=>true),
        global_consts_ty=Dict{Symbol,Any}(:flag=>BoolTy))
    first = CG._prepare_finite_core_certificate(relation, BoolTy, Ty[], 0, 1)
    @test first !== nothing
    @test checks[] == 1
    checks[] = 0
    repeated = CG._prepare_finite_core_certificate(relation, BoolTy, Ty[], 0, 1)
    @test repeated.world === first.world
    @test checks[] == 1

    relation.global_consts[:flag] = false
    changed_value = CG._prepare_finite_core_certificate(relation, BoolTy, Ty[], 0, 1)
    @test changed_value.world !== first.world
    relation.weights.overrides[:constant] = 2.0
    changed_weights = CG._prepare_finite_core_certificate(relation, BoolTy, Ty[], 0, 1)
    @test changed_weights.world !== changed_value.world
    local_ty = TyFun(FloatTy, BoolTy)
    relation.local_consts[:mapper] = identity
    relation.local_consts_ty[:mapper] = local_ty
    with_local = CG._finite_evidence_world(relation, BoolTy, Ty[]; sampling=true)
    setfield!(local_ty, :res, FloatTy)
    @test CG._finite_evidence_world(relation, BoolTy, Ty[]; sampling=true) !== with_local

    checks[] = 0
    count_before = length(relation.evidence_worlds)
    private = CG._finite_evidence_world(relation, BoolTy, Ty[]; reuse=false, sampling=true)
    @test private !== with_local
    @test checks[] == 0
    @test length(relation.evidence_worlds) == count_before

    opaque = CG.CoreRelation(providers=(CountedEvidenceProvider(checks, false),),
        global_consts=Dict{Symbol,Any}(:flag=>true),
        global_consts_ty=Dict{Symbol,Any}(:flag=>BoolTy))
    a = CG._finite_evidence_world(opaque, BoolTy, Ty[])
    @test CG._finite_evidence_world(opaque, BoolTy, Ty[]) !== a
    @test isempty(opaque.evidence_worlds)
    # A provider without a declared key must also stay private on the witness
    # entry, whose default `reuse=true` only permits, never asserts, cacheability.
    query = CG.GenerationQuery(BoolTy; depth_left=0, node_budget=1)
    @test CG._finite_evidence_witness(opaque, query) !== nothing
    @test isempty(opaque.evidence_worlds)

    uncacheable_value = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:number=>1.0),
        global_consts_ty=Dict{Symbol,Any}(:number=>FloatTy))
    b = CG._finite_evidence_world(uncacheable_value, FloatTy, Ty[])
    @test CG._finite_evidence_world(uncacheable_value, FloatTy, Ty[]) !== b
    @test isempty(uncacheable_value.evidence_worlds)
end

@testitem "CanonicalGeneration layered worlds defer unused root catalogues" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyScheme, FloatTy, BoolTy, newtyvar, typekey
    CG = LambdaRegression.CanonicalGeneration
    alpha = newtyvar(0)
    list = TyList(FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.NilFactoryProvider()),
        global_consts=Dict{Symbol,Any}(:consume=>identity),
        global_consts_ty=Dict{Symbol,Any}(:consume=>TyScheme([alpha], TyFun(alpha, BoolTy))))
    original_scope = Ty[list, FloatTy]
    world = CG._new_generation_evidence(relation, BoolTy, original_scope; sampling=true)
    @test world.layered
    @test world.catalog === nothing
    @test all(ty -> CG._evidence_type_id(world, ty) != 0,
        (BoolTy, list, FloatTy))
    empty!(original_scope) # The root owns the original domain vector.
    @test length(world.root_scope) == 2
    root_context = CG._evidence_context!(world, Int[]) # All original binders forbidden.
    root_type = CG._evidence_type_id(world, BoolTy)
    root = CG._evidence_catalog_state(world, relation, root_context, root_type)
    rulekeys(state) = Set(typekey(rule.head.type) for rule in state.rules)
    @test typekey(TyFun(list, BoolTy)) in rulekeys(root)
    child_list = TyList(BoolTy)
    CG._evidence_register_type!(world, child_list)
    child_context = CG._evidence_context!(world, Int[CG._evidence_type_id(world, child_list)])
    child = CG._evidence_catalog_state(world, relation, child_context, root_type)
    @test typekey(TyFun(child_list, BoolTy)) in rulekeys(child)
    @test !(typekey(TyFun(child_list, BoolTy)) in rulekeys(root))
    @test world.catalog === nothing
    @test CG._evidence_catalog_state(world, relation, root_context, root_type) === root
    # A direct legacy consumer gets the exact original root domain, not the
    # first/last visited child catalogue or the now-larger registry.
    owned = CG._evidence_root_catalog!(world, relation)
    reference = CG._build_core_catalog(CG.CoreUniverse(BoolTy;
        global_consts_ty=relation.global_consts_ty, scope_source_tys=world.root_scope);
        global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty,
        scope_source_tys=world.root_scope, providers=relation.providers, catalog_target_ty=BoolTy)
    headkeys(catalog) = [(h.name, h.source, typekey(h.type)) for h in catalog.heads]
    @test headkeys(owned) == headkeys(reference)
    @test CG._evidence_root_catalog!(world, relation) === owned
    @test !any(h -> typekey(h.type) == typekey(TyFun(child_list, BoolTy)), owned.heads)

    prepared = CG._prepare_finite_core_certificate(relation, BoolTy,
        Ty[list, FloatTy], 1, 2; forbidden=(0, 1))
    @test prepared !== nothing
    @test prepared.world.catalog === nothing # No local-policy preflight for built-ins.
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 2)
    witness = CG._sample_evidence_fact!(state, prepared.world, relation, prepared.fact,
        Ty[list, FloatTy], (0, 1); certificate_ready=true)
    @test witness isa CG.HeadApplicationDerivation
    @test witness.head.name == :consume
    @test only(witness.args) isa CG.ConstantDerivation
    @test only(witness.args).head.name == :NIL
    @test CG.derivation_node_count(witness) == 2
    @test !CG.derivation_uses_scope_binder(witness, 0)
    @test !CG.derivation_uses_scope_binder(witness, 1)
    @test state.nodes_left == 0
end

@testitem "CanonicalGeneration layered catalogue domains share preparation not available facts" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyScheme,
        FloatTy, BoolTy, newtyvar, typekey
    CG = LambdaRegression.CanonicalGeneration
    alpha = newtyvar(0)
    list = TyList(FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.NilFactoryProvider()),
        global_consts=Dict{Symbol,Any}(:consume => identity),
        global_consts_ty=Dict{Symbol,Any}(:consume => TyScheme([alpha], TyFun(alpha, BoolTy))))
    world = CG._new_generation_evidence(relation, BoolTy, Ty[list, FloatTy]; sampling=true)
    absent = CG._evidence_context!(world, Int[])
    present = CG._evidence_context!(world, Int[CG._evidence_type_id(world, FloatTy)])
    target = CG._evidence_type_id(world, BoolTy)
    first = CG._evidence_catalog_state(world, relation, absent, target)
    second = CG._evidence_catalog_state(world, relation, present, target)
    @test first === second
    @test length(world.catalogs) == 1
    @test absent != present
    floating = CG._evidence_type_id(world, FloatTy)
    missing_fact = CG._evidence_fact!(world, absent, floating, 0, -1, 1)
    variable_fact = CG._evidence_fact!(world, present, floating, 0, -1, 1)
    CG._evidence_saturate!(world, relation)
    @test missing_fact != variable_fact
    @test world.certificates[missing_fact] == 0
    @test world.costs[variable_fact] == 1
    @test CG._evidence_replay(world, variable_fact, Ty[FloatTy], ()) isa CG.VariableDerivation
    @test CG._evidence_replay(world, variable_fact, Ty[FloatTy], ()).index == 0

    # Registering a type alone must not enlarge the original catalogue domain.
    child_list = TyList(BoolTy)
    CG._evidence_register_type!(world, child_list)
    child = CG._evidence_context!(world, Int[CG._evidence_type_id(world, child_list)])
    child_state = CG._evidence_catalog_state(world, relation, child, target)
    @test child_state !== first
    rulekeys(state) = Set(typekey(rule.head.type) for rule in state.rules)
    @test typekey(TyFun(child_list, BoolTy)) in rulekeys(child_state)
    @test !(typekey(TyFun(child_list, BoolTy)) in rulekeys(first))
    @test CG._evidence_catalog_state(world, relation, absent, target) === first

    universe = CG.CoreUniverse(BoolTy; global_consts_ty=relation.global_consts_ty,
        scope_source_tys=Ty[list, FloatTy])
    full = CG._build_core_catalog(universe; global_consts=relation.global_consts,
        global_consts_ty=relation.global_consts_ty, scope_source_tys=Ty[list, FloatTy],
        providers=relation.providers, catalog_target_ty=BoolTy)
    prepared, heads = CG._build_core_heads(universe; global_consts=relation.global_consts,
        global_consts_ty=relation.global_consts_ty, scope_source_tys=Ty[list, FloatTy],
        providers=relation.providers, catalog_target_ty=BoolTy)
    headkeys(heads) = [(h.name, h.source, typekey(h.type)) for h in heads]
    @test prepared.typekeys == full.universe.typekeys
    @test headkeys(heads) == headkeys(full.heads)
    @test [headkeys([r.head]) for r in CG._core_rules(heads, prepared)] ==
        [headkeys([r.head]) for r in full.rules]

    # Unknown callbacks retain their old per-availability catalogue boundary,
    # even if they declare scope invariance for the finite-fact engine.
    struct CountedDomainProvider <: CG.AbstractHeadProvider
        builds::Base.RefValue{Int}
    end
    CG.generation_catalog_scope_invariant(::CountedDomainProvider) = true
    function CG.add_heads!(provider::CountedDomainProvider, heads, universe; kwargs...)
        provider.builds[] += 1
        CG.add_heads!(CG.ContextHeadProvider(), heads, universe; kwargs...)
    end
    builds = Ref(0)
    custom = CG.CoreRelation(providers=(CountedDomainProvider(builds),),
        global_consts=relation.global_consts, global_consts_ty=relation.global_consts_ty)
    private = CG._new_generation_evidence(custom, BoolTy, Ty[FloatTy]; sampling=true)
    absent = CG._evidence_context!(private, Int[])
    present = CG._evidence_context!(private, Int[CG._evidence_type_id(private, FloatTy)])
    target = CG._evidence_type_id(private, BoolTy)
    one = CG._evidence_catalog_state(private, custom, absent, target)
    two = CG._evidence_catalog_state(private, custom, present, target)
    @test one !== two
    @test builds[] == 2
    @test CG._evidence_catalog_state(private, custom, absent, target) === one
    @test builds[] == 2
end

@testitem "CanonicalGeneration lazy layered worlds retain unknown local policy preflight" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyScheme, FloatTy, BoolTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    struct LayeredLocalPolicyProbe <: CG.AbstractHeadProvider
        preparations::Base.RefValue{Int}
    end
    CG.generation_catalog_scope_invariant(::LayeredLocalPolicyProbe) = true
    function CG.add_heads!(provider::LayeredLocalPolicyProbe, heads, universe; kwargs...)
        provider.preparations[] += 1
        push!(heads, CG.HeadEntry(:unused, :policy_probe, TyFun(FloatTy, FloatTy),
            CG.CoreProductive(), CG.WeightBucket(:constant);
            metadata=(value=identity, argument_planner=(ds,args...)->empty(ds))))
        return heads
    end
    alpha = newtyvar(0)
    count = Ref(0)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), LayeredLocalPolicyProbe(count)),
        global_consts=Dict{Symbol,Any}(:flag=>true, :id=>identity),
        global_consts_ty=Dict{Symbol,Any}(:flag=>BoolTy,
            :id=>TyScheme([alpha], TyFun(alpha, alpha))))
    world = CG._new_generation_evidence(relation, BoolTy, Ty[]; sampling=true)
    @test count[] == 0
    @test world.catalog === nothing
    @test CG._evidence_has_local_argument_policy(world, relation)
    @test count[] == 1
    @test world.catalog isa CG.CoreCatalog
    @test CG._evidence_has_local_argument_policy(world, relation)
    @test count[] == 1
    # Even an unrelated policy head preserves the original unsupported cut.
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 4)
    @test CG._sample_finite_core_derivation(state, relation, BoolTy, Ty[], 1) isa
        CG._UnsupportedEvidenceSampling
    @test state.nodes_left == 4
end

@testitem "CanonicalGeneration ordinary steps defer unselected child proofs" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:seed=>1.0, :b=>true, :from_bool=>(x->1.0)),
        global_consts_ty=Dict{Symbol,Any}(:seed=>FloatTy, :b=>BoolTy,
            :from_bool=>TyFun(BoolTy, FloatTy)))
    world = CG._new_generation_evidence(relation, FloatTy, Ty[]; sampling=true)
    context = CG._evidence_context!(world, Int[])
    fact = CG._evidence_fact!(world, context, CG._evidence_type_id(world, FloatTy), -1)
    CG._evidence_saturate!(world, relation)
    @test length(world.facts) == 1
    @test world.costs[fact] == 1
    steps = CG._evidence_steps!(world, relation, fact)
    @test any(p -> p.kind == :head && p.value.name == :from_bool, steps)
    @test length(world.facts) == 1
    @test isempty(world.menus)
    @test CG._evidence_steps!(world, relation, fact) === steps

    # Completion of an already proved leaf does not prepare unselected heads.
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 8,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._sample_evidence_fact!(state, world, relation, fact, Ty[];
        certificate_ready=true)
    @test result isa CG.ConstantDerivation
    @test length(world.facts) == 1
    @test isempty(world.menus)
    @test state.nodes_left == 7
    @test state.completion.boundaries == 1

    step = only(filter(p -> p.kind == :head && p.value.name == :from_bool, steps))
    plan = CG._evidence_prepare_step!(world, relation, step)
    @test length(world.facts) == 2
    @test world.costs[only(plan.children)] == 1
    @test CG._evidence_plan_cost(world, plan) == 2
    @test isempty(world.menus)
    @test any(p -> p.kind == :head && p.value.name == :from_bool,
        CG._evidence_menu!(world, relation, fact))

    # Selection must sometimes prove and reject an expensive child, not just
    # filter by child count. Both root options pass the cheap size lower bound.
    small = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:seed=>1.0, :make_bool=>(x->true),
            :costly=>(b->1.0)),
        global_consts_ty=Dict{Symbol,Any}(:seed=>FloatTy, :make_bool=>TyFun(FloatTy, BoolTy),
            :costly=>TyFun(BoolTy, FloatTy)))
    small_world = CG._new_generation_evidence(small, FloatTy, Ty[]; sampling=true)
    small_ctx = CG._evidence_context!(small_world, Int[])
    small_fact = CG._evidence_fact!(small_world, small_ctx,
        CG._evidence_type_id(small_world, FloatTy), -1)
    CG._evidence_saturate!(small_world, small)
    @test length(small_world.facts) == 1
    for seed in 1:12
        s = CG.DerivationSamplingState(Random.Xoshiro(seed), 2)
        d = CG._sample_evidence_fact!(s, small_world, small, small_fact, Ty[];
            certificate_ready=true)
        @test d isa CG.ConstantDerivation
        @test d.head.name == :seed
        @test CG.derivation_node_count(d) == 1
    end
    bool_fact = CG._evidence_fact!(small_world, small_ctx,
        CG._evidence_type_id(small_world, BoolTy), -1)
    @test small_world.certificates[bool_fact] != 0 # Selected, proved, rejected.
    @test small_world.costs[bool_fact] == 2

    # A head-only ordinary proof must not satisfy allow_heads=false, even in
    # completion mode; failure neither charges nodes nor flips an idle root.
    denied = CG.DerivationSamplingState(Random.Xoshiro(1111), 8,
        CG.GenerationCompletionState(probability=1.0))
    @test CG._sample_finite_core_derivation(denied, small, BoolTy, Ty[], -1;
        allow_heads=false) === nothing
    @test denied.nodes_left == 8
    @test !denied.completion.completing

    # Policy-specific completion uses only a prepared menu. A new inherited
    # completed request with no such evidence declares the explicit boundary.
    frozen = CG.DerivationSamplingState(Random.Xoshiro(1111), 8,
        CG.GenerationCompletionState(probability=1.0))
    frozen.completion.completing = true
    before_facts, before_checks = length(small_world.facts), small_world.rule_checks
    missing = CG._sample_evidence_fact!(frozen, small_world, small, small_fact, Ty[];
        allow_heads=false, certificate_ready=true)
    @test missing isa CG._UnsupportedEvidenceSampling
    @test length(small_world.facts) == before_facts
    @test small_world.rule_checks == before_checks
    @test frozen.nodes_left == 8

    CG._evidence_menu!(small_world, small, small_fact; allow_heads=false)
    before_facts, before_checks = length(small_world.facts), small_world.rule_checks
    replayed = CG._sample_evidence_fact!(frozen, small_world, small, small_fact, Ty[];
        allow_heads=false, certificate_ready=true)
    @test replayed isa CG.ConstantDerivation
    @test length(small_world.facts) == before_facts
    @test small_world.rule_checks == before_checks
    @test frozen.completion.boundaries == 0
    @test frozen.nodes_left == 7
end

@testitem "CanonicalGeneration evidence plans preserve support and shared completion" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:seed=>1.0, :wrap=>identity),
        global_consts_ty=Dict{Symbol,Any}(:seed=>FloatTy, :wrap=>TyFun(FloatTy, FloatTy)))
    world = CG._new_generation_evidence(relation, FloatTy, Ty[]; sampling=true)
    context = CG._evidence_context!(world, Int[])
    fact = CG._evidence_fact!(world, context, CG._evidence_type_id(world, FloatTy), -1)
    CG._evidence_saturate!(world, relation)
    menu = CG._evidence_menu!(world, relation, fact)
    # First proof is a leaf, but normal exploration retains wrap's cycle.
    @test any(p -> p.kind == :constant, menu)
    @test any(p -> p.kind == :head && p.value.name == :wrap && p.children == [fact], menu)

    immediate = CG.GenerationCompletionState(probability=1.0)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 20, immediate)
    completed = CG._sample_finite_core_derivation(state, relation, FloatTy, Ty[], -1)
    @test completed isa CG.ConstantDerivation
    @test immediate.completing
    @test state.nodes_left == 19
    state.nodes_left = 20 # Fuel rollback must not undo completion.
    @test immediate.completing

    duplicate_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    chosen_indices = Set{Int}()
    for seed in 1:24
        s = CG.DerivationSamplingState(Random.Xoshiro(seed), 4)
        d = CG._sample_finite_core_derivation(s, duplicate_relation,
            FloatTy, Ty[FloatTy, FloatTy], 0)
        push!(chosen_indices, d.index)
    end
    @test chosen_indices == Set([0, 1])

    # The first selected sibling may explore only while reserving the second
    # sibling's certificate. Both inherit the same irreversible root mode.
    binary_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:seed=>1.0, :pair=>((x,y)->true), :wrap=>identity),
        global_consts_ty=Dict{Symbol,Any}(:seed=>FloatTy,
            :pair=>TyFun(FloatTy, TyFun(FloatTy, BoolTy)), :wrap=>TyFun(FloatTy, FloatTy)))
    root_mode = CG.GenerationCompletionState(probability=1.0, grace_steps=1)
    sibling_state = CG.DerivationSamplingState(Random.Xoshiro(1111), 8, root_mode)
    binary = CG._sample_finite_core_derivation(sibling_state, binary_relation, BoolTy, Ty[], -1)
    @test binary isa CG.HeadApplicationDerivation
    @test root_mode.completing
    @test length(binary.args) == 2
    @test all(arg -> arg isa CG.ConstantDerivation, binary.args)
    @test root_mode.boundaries == 2
    @test CG.derivation_node_count(binary) == 3

    # Positive weights plus a nonzero switch retain long finite routes.
    lengths = Int[]
    for seed in 1:40
        m = CG.GenerationCompletionState(probability=0.1)
        s = CG.DerivationSamplingState(Random.Xoshiro(seed), 64, m)
        d = CG._sample_finite_core_derivation(s, relation, FloatTy, Ty[], -1)
        @test d !== nothing
        push!(lengths, CG.derivation_node_count(d))
    end
    @test maximum(lengths) > 2
end

@testitem "CanonicalGeneration non-bare evidence preserves root identity and budget" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration

    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    only_x = CG._prepare_finite_core_nonbare_certificate(relation,
        FloatTy, Ty[FloatTy], 0, 1, 0)
    @test only_x.status == :unavailable
    @test only_x.reason == :no_compatible_root_plan

    duplicate = CG._prepare_finite_core_nonbare_certificate(relation,
        FloatTy, Ty[FloatTy, FloatTy], 0, 1, 0)
    @test duplicate.status == :ready
    @test [(option.plan.kind, option.binder) for option in duplicate.options] == [(:variable, 1)]
    duplicate_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 1,
        CG.GenerationCompletionState(probability=1.0))
    duplicate_root = CG._select_finite_core_nonbare_root(duplicate_state, duplicate)
    other_x = CG._sample_evidence_plan!(duplicate_state, duplicate.world, relation,
        duplicate_root.plan, duplicate_root.binder, duplicate.scope,
        duplicate.forbidden)
    @test other_x isa CG.VariableDerivation
    @test other_x.index == 1
    @test duplicate_state.nodes_left == 0

    function_ty = TyFun(FloatTy, BoolTy)
    scope = Ty[FloatTy, function_ty, function_ty] # x, f, g
    wrapped_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    wrapped = CG._prepare_finite_core_nonbare_certificate(wrapped_relation,
        BoolTy, scope, 2, 2, 0)
    @test wrapped.status == :ready
    root_options = filter(option -> option.plan.kind == :scope_application,
        wrapped.options)
    @test Set(option.binder for option in root_options) == Set([1, 2])
    @test all(option -> option.cost == 2 && option.weight > 0, root_options)

    # Completion uses the saved root binder and ordinary child replay. The
    # excluded x is still available inside f(x), and completion terminates.
    completion = CG.GenerationCompletionState(probability=1.0)
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 2, completion)
    wrapped_root = CG._select_finite_core_nonbare_root(state, wrapped)
    witness = CG._sample_evidence_plan!(state, wrapped.world, wrapped_relation,
        wrapped_root.plan, wrapped_root.binder, wrapped.scope,
        wrapped.forbidden)
    @test witness isa CG.ScopeApplicationDerivation
    @test witness.index in (1, 2)
    @test only(witness.args) isa CG.VariableDerivation
    @test only(witness.args).index == 0
    @test CG.derivation_node_count(witness) == 2
    @test CG.derivation_result_type(witness) == BoolTy
    @test completion.completing
    @test state.nodes_left == 0

    too_small = CG._prepare_finite_core_nonbare_certificate(wrapped_relation,
        BoolTy, scope, 2, 1, 0)
    @test too_small.status == :budget
    @test too_small.proven_cost == 2
    budget_state = CG.DerivationSamplingState(Random.Xoshiro(1), 1)
    @test CG._select_finite_core_nonbare_root(budget_state, too_small) === nothing
    @test budget_state.nodes_left == 1
end

@testitem "CanonicalGeneration finite evidence closure and concrete replay" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyArray, TyNat, TyScheme, newtyvar, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration

    # A positive cycle cannot establish its own first fact; adding an
    # independent base establishes it without enumerating the cycle's terms.
    loop_ty = TyFun(BoolTy, BoolTy)
    negative = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    world = CG._new_generation_evidence(negative, BoolTy, Ty[loop_ty])
    ctx = CG._evidence_context!(world, Int[CG._evidence_type_id(world, loop_ty)])
    fact = CG._evidence_fact!(world, ctx, CG._evidence_type_id(world, BoolTy), -1)
    CG._evidence_saturate!(world, negative)
    @test world.certificates[fact] == 0
    @test length(world.facts) == 1

    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    q = CG.GenerationQuery(BoolTy; scope_tys=Ty[loop_ty, BoolTy],
        depth_left=8, node_budget=16)
    @test CG.known_generation_witness(relation, q) isa CG.VariableDerivation
    forbidden = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy, FloatTy],
        depth_left=0, node_budget=1, requirements=(CG.ForbidsScopeBinder(0),))
    witness = CG.known_generation_witness(relation, forbidden)
    @test witness isa CG.VariableDerivation
    @test witness.index == 1
    @test !CG.derivation_uses_scope_binder(witness, 0)

    function_ty = TyFun(FloatTy, FloatTy)
    under_lambda = CG.GenerationQuery(function_ty; scope_tys=Ty[function_ty],
        depth_left=1, node_budget=2, requirements=(CG.ForbidsScopeBinder(0),))
    lambda = CG.known_generation_witness(relation, under_lambda)
    @test lambda isa CG.LambdaDerivation
    @test lambda.body isa CG.VariableDerivation
    @test lambda.body.index == 0

    applied = CG.GenerationQuery(BoolTy;
        scope_tys=Ty[FloatTy, TyFun(FloatTy, BoolTy)], depth_left=2, node_budget=2)
    application = CG.known_generation_witness(relation, applied)
    @test application isa CG.ScopeApplicationDerivation
    @test application.index == 1
    @test only(application.args).index == 0
    @test CG.derivation_node_count(application) == 2

    # Joint argument costs matter: a first larger route cannot hide a smaller
    # route which fits the public node-budget contract.
    minimum_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:seed=>1.0, :large=>identity,
            :small=>identity, :finish=>identity),
        global_consts_ty=Dict{Symbol,Any}(:seed=>FloatTy,
            :large=>TyFun(FloatTy, TyFun(FloatTy, BoolTy)),
            :small=>TyFun(FloatTy, BoolTy), :finish=>TyFun(BoolTy, BoolTy)))
    minimum_query = CG.GenerationQuery(BoolTy; depth_left=3, node_budget=2)
    minimum = CG.known_generation_witness(minimum_relation, minimum_query)
    @test minimum isa CG.HeadApplicationDerivation
    @test minimum.head.name == :small
    @test CG.derivation_node_count(minimum) == 2

    small_budget = CG.GenerationQuery(BoolTy; depth_left=3, node_budget=1)
    fresh = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=minimum_relation.global_consts,
        global_consts_ty=minimum_relation.global_consts_ty)
    @test CG.known_generation_witness(fresh, small_budget) === nothing
    @test CG.known_generation_witness(fresh, minimum_query) !== nothing

    alpha = newtyvar(0)
    scheme_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.NilFactoryProvider()),
        global_consts=Dict{Symbol,Any}(:consume=>identity),
        global_consts_ty=Dict{Symbol,Any}(:consume=>TyScheme([alpha], TyFun(TyList(alpha), BoolTy))))
    scheme_query = CG.GenerationQuery(BoolTy; depth_left=2, node_budget=2)
    scheme_witness = CG.known_generation_witness(scheme_relation, scheme_query)
    @test scheme_witness isa CG.HeadApplicationDerivation
    @test scheme_witness.head.name == :consume
    @test only(scheme_witness.args) isa CG.ConstantDerivation
    @test only(scheme_witness.args).head.name == :NIL
end

@testitem "CanonicalGeneration finite evidence handles context growth without witness enumeration" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, TyArray, TyNat, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    list = TyList(FloatTy)
    consumer = TyFun(list, TyFun(TyFun(FloatTy, TyFun(list, FloatTy)),
        TyFun(FloatTy, FloatTy)))
    relation = CG.CoreRelation(local_consts=Dict{Symbol,Any}(
        :consume=>LambdaRegression.Primitives.UNCONS_CASE),
        local_consts_ty=Dict{Symbol,TyLike}(:consume=>consumer))
    world = CG._new_generation_evidence(relation, list, Ty[array])
    ctx = CG._evidence_context!(world, Int[CG._evidence_type_id(world, array)])
    fact = CG._evidence_fact!(world, ctx, CG._evidence_type_id(world, list), -1)
    CG._evidence_saturate!(world, relation)
    @test world.certificates[fact] != 0
    proof = CG._evidence_replay(world, fact, Ty[array])
    @test CG.derivation_result_type(proof) == list
    @test isempty(world.pending)
    @test isempty(world.updates)
    @test isempty(relation.derivations)

    # Repeating a type in the actual scope does not create a new abstract
    # state, yet replay may select a different non-forbidden concrete binder.
    singleton = CG._evidence_context!(world, Int[CG._evidence_type_id(world, FloatTy)])
    duplicate = CG._evidence_context!(world,
        Int[CG._evidence_type_id(world, FloatTy), CG._evidence_type_id(world, FloatTy)])
    @test singleton == duplicate
    leaf = CG._evidence_fact!(world, singleton, CG._evidence_type_id(world, FloatTy), -1)
    CG._evidence_saturate!(world, relation)
    @test CG._evidence_replay(world, leaf, Ty[FloatTy, FloatTy], (0,)).index == 1
end
