struct ListAccessSource
    ty::TyList
    derivation::CoreDerivation
end

"""An anchored source family, not a concrete list-access identity.

The closed array type comes from the original scope/constant context. Selection
uses the default finite source grammar at `depth`; only the selected concrete
array witness becomes a `ListAccessSource`, before any proxy body is generated.
Custom argument/source grammars retain their existing bounded sampling path.
"""
struct _ArrayListAccessSourcePlan
    array_ty::TyArray
    depth::Int
end

struct SafeListAccessDerivation <: CoreDerivation
    source::CoreDerivation
    source_ty::TyList
    body::CoreDerivation
    fallback::CoreDerivation
    result_ty::Ty
end

const _RequiredScopeExactChoice = NamedTuple{(:index, :arguments),Tuple{Int,Vector{Int}}}

const _SAFE_LIST_ACCESS_WEIGHT = WeightBucket(:safe_list_access)

const _REQUIRED_SCOPE_ROUTE_DIAGNOSTIC_KEY = gensym(:required_scope_route_observer)

function _record_required_scope_route!(derivation, route::Symbol)
    derivation isa CoreDerivation || return derivation
    observer = get(task_local_storage(), _REQUIRED_SCOPE_ROUTE_DIAGNOSTIC_KEY, nothing)
    observer === nothing || observer(derivation, route)
    return derivation
end

function _safe_list_access_provider(providers::Tuple)
    for provider in providers
        provider isa SafeListAccessProvider && return provider
    end
    return nothing
end

_safe_list_access_enabled(relation::CoreRelation) =
    _safe_list_access_provider(relation.providers) !== nothing &&
    _bucket_enabled(relation.weights, _SAFE_LIST_ACCESS_WEIGHT)

function _safe_list_access_generation_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    ; evidence_only::Bool=false,
)
    _safe_list_access_enabled(relation) || return NamedTuple[]
    _may_have_list_access_source(relation, scope_tys) || return NamedTuple[]
    # A materializer may add local constants to this relation. A source menu
    # must not survive a changed external context or a mutable weight policy.
    context_version = (
        _cacheable_global_consts_key(relation.global_consts, relation.global_consts_ty),
        _cacheable_local_context_key(relation.local_consts, relation.local_consts_ty),
        _cacheable_providers_key(relation.providers),
        _weights_key(relation.weights), relation.capability_mask,
    )
    cacheable = all(!isnothing, context_version[1:3])
    key = _StructuralKey((:safe_list_access_plans, _known_key(scope_tys, target_ty), context_version, evidence_only))
    if cacheable
        cached = get(relation.sampling_options, key, nothing)
        cached === nothing || return cached
    end
    sources = _list_access_sampling_sources(relation, scope_tys; evidence_only)
    sources isa _UnsupportedEvidenceSampling && return sources
    if isempty(sources)
        cacheable && (relation.sampling_options[key] = NamedTuple[])
        return NamedTuple[]
    end
    safe_weight = _bucket_weight(relation.weights, _SAFE_LIST_ACCESS_WEIGHT)
    options = [
        (
            kind=:safe_list_access,
            value=(source=source, target_ty=target_ty),
            weight=safe_weight * _list_access_source_weight(source, relation.weights),
        )
        for source in sources
    ]
    cacheable && (relation.sampling_options[key] = options)
    return options
end

_list_access_source_weight(source::ListAccessSource, weights::GenerationWeights) =
    _derivation_weight(source.derivation, weights)

# Family selection changes probabilities, not support: every admitted concrete
# source still has a positive conditional route. Do not pretend this is the
# sum of the weights of all witnesses (which would require their enumeration).
_list_access_source_weight(::_ArrayListAccessSourcePlan, weights::GenerationWeights) =
    _bucket_weight(weights, _TOLIST_WEIGHT) * _bucket_weight(weights, _ARRAY_SOURCE_WEIGHT)

function _may_have_list_access_source(relation::CoreRelation, scope_tys::Vector{Ty})
    provider = _safe_list_access_provider(relation.providers)
    provider === nothing && return false
    isempty(provider.source_collectors) || return true
    any(ty -> ty isa Union{TyList,TyArray}, scope_tys) && return true
    for tyctx in (relation.local_consts_ty, relation.global_consts_ty)
        any(ty -> ty isa Union{TyList,TyArray}, values(tyctx)) && return true
    end
    return false
end

function _safe_list_access_candidates(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    fallback_derivations::Vector{CoreDerivation},
)
    _safe_list_access_enabled(relation) || return CoreDerivation[]
    isempty(fallback_derivations) && return CoreDerivation[]
    out = CoreDerivation[]
    signatures = Set{Any}()
    for source in _list_access_sources(relation, scope_tys)
        success_scope = _list_access_success_scope(source, scope_tys)
        body_core = _known_derivations!(relation, target_ty, success_scope)
        for body in _generation_candidates(
            relation,
            target_ty,
            success_scope,
            body_core;
            allow_safe_list_access=false,
        )
            _uses_scope_prefix(body, 2) || continue
            for fallback in fallback_derivations
                _push_derivation_candidate!(
                    out,
                    signatures,
                    SafeListAccessDerivation(source.derivation, source.ty, body, fallback, target_ty),
                )
            end
        end
    end
    return out
end

function _sample_safe_list_access(
    state::DerivationSamplingState,
    relation::CoreRelation,
    source::Union{ListAccessSource,_ArrayListAccessSourcePlan},
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
    ; body_sampler=_sample_safe_list_access_body, evidence_only::Bool=false,
)
    # evidence_only keeps finite preparation separate from the old explicit
    # source-product fallback. The latter remains this sampler's default.
    # Lease the entire access budget, including lowering/body/fallback. The
    # state may include a caller's unleased surplus; array-only clamping is
    # insufficient because all the other charges belong to this request too.
    available = generation_query === nothing ? state.nodes_left :
        min(state.nodes_left, generation_query.node_budget)
    outside = state.nodes_left - available
    state.nodes_left = available
    succeeded = false
    try
        result = _sample_safe_list_access_impl(state, relation, source, target_ty,
            scope_tys, depth_left, guarded_left, generation_query; body_sampler, evidence_only)
        succeeded = result !== nothing
        return result
    finally
        succeeded || (state.nodes_left = available)
        state.nodes_left += outside
    end
end

function _sample_safe_list_access_impl(
    state::DerivationSamplingState,
    relation::CoreRelation,
    source::ListAccessSource,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
    ; body_sampler=_sample_safe_list_access_body, evidence_only::Bool=false,
)
    depth_left > 0 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 3 + _derivation_node_count(source.derivation)) || return nothing
    return body_sampler(state, relation, source, target_ty, scope_tys,
        depth_left, guarded_left, generation_query, saved_nodes)
end

function _sample_safe_list_access_impl(
    state::DerivationSamplingState,
    relation::CoreRelation,
    plan::_ArrayListAccessSourcePlan,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
    ; body_sampler=_sample_safe_list_access_body, evidence_only::Bool=false,
)
    depth_left > 0 || return nothing
    saved_nodes = state.nodes_left
    # UNCONS lowering has the existing three-node charge; TOLIST costs one.
    # Every successful body and fallback consumes at least one node each, so
    # this reservation excludes no previously successful bounded access.
    state.nodes_left >= 7 || return nothing
    _consume_sampling_nodes!(state, 4) || return nothing
    state.nodes_left -= 2
    array = try
        _sample_finite_source_array(state, relation, plan.array_ty, scope_tys,
            depth_left, plan.depth, generation_query)
    finally
        state.nodes_left += 2
    end
    if array isa _UnsupportedEvidenceSampling
        # Finite certificate preparation must not escape into the old complete
        # witness product. This is unsupported preparation, not a negative
        # answer to the old consumer, whose default fallback remains below.
        evidence_only && return nothing
        # Unknown/custom source policies remain explicit old-path boundaries.
        # Reset the source reservation before selecting their concrete witness.
        state.nodes_left = saved_nodes
        sources = _bounded_source_array_derivations(relation, plan.array_ty, scope_tys, plan.depth)
        state.nodes_left -= 4 + 2
        array = _sample_materialized_derivation(state, relation, sources)
        state.nodes_left += 2
    end
    if array === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    list_ty = TyList(plan.array_ty.elem)
    source = ListAccessSource(list_ty, HeadApplicationDerivation(
        _safe_tolist_head(plan.array_ty, list_ty), CoreDerivation[array]))
    if _contains_rigid_dummy(source.ty) && !_rigid_derivation_allowed(source.derivation)
        state.nodes_left = saved_nodes
        return nothing
    end
    return body_sampler(state, relation, source, target_ty, scope_tys,
        depth_left, guarded_left, generation_query, saved_nodes)
end

function _sample_safe_list_access_body(
    state::DerivationSamplingState,
    relation::CoreRelation,
    source::ListAccessSource,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
    saved_nodes::Int,
)
    fallback = _sample_short_nonrecursive_list_fallback(state, relation, target_ty,
        scope_tys, depth_left - 1, guarded_left, generation_query)
    fallback === nothing && (fallback =
        _sample_known_derivation(state, relation, target_ty, scope_tys, depth_left - 1))
    if fallback === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    success_scope = _list_access_success_scope(source, scope_tys)
    body_parent = _without_satisfied_scope_binder_requirements(
        generation_query,
        source.derivation,
    )
    body_parent = _without_satisfied_scope_binder_requirements(
        body_parent,
        fallback,
    )
    body_query = _nested_generation_query(
        state,
        body_parent,
        target_ty,
        success_scope,
        depth_left - 1,
        guarded_left,
    )
    body_query === nothing || (body_query = _derive_generation_query(
        body_query;
        capability_mask=(:safe_list_access,),
    ))
    body_query = _with_list_recursion_hints(relation, source, generation_query, body_query)
    body = _sample_list_recursive_body(state, relation, target_ty,
        success_scope, depth_left - 1, guarded_left, body_query)
    body === nothing && (body = _sample_proxy_body_derivation(
        state,
        relation,
        target_ty,
        success_scope,
        depth_left - 1,
        guarded_left,
        body_query,
    ))
    if body === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    return SafeListAccessDerivation(source.derivation, source.ty, body, fallback, target_ty)
end

function _sample_proxy_body_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    return _sample_proxy_using_derivation(
        state,
        relation,
        target_ty,
        scope_tys,
        depth_left,
        guarded_left,
        2,
        0;
        generation_query,
    )
end

"""Construct a safe-list candidate with joint core evidence for its success body.

The selected source is already concrete. `generation_query` supplies exact
use/call and forbidden slots; source/fallback discharge only actual occurrences.
The two new proxies are alternatives, not two conjunctive requirements. This
bounded candidate route neither opens the root completion switch nor certifies
an enclosing FunctionDef post-filter. Missing core evidence means route failure,
not mixed non-inhabitation.
"""
function _sample_certified_safe_list_body(
    state::DerivationSamplingState, relation::CoreRelation, source::ListAccessSource,
    target_ty::Ty, scope_tys::Vector{Ty}, depth_left::Int, guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}, saved_nodes::Int,
)
    generation_query === nothing && return nothing
    forbidden = _forbidden_scope_binders(generation_query)
    any(index -> derivation_uses_scope_binder(source.derivation, index), forbidden) && return nothing
    fallback_query = _nested_generation_query(state, generation_query, target_ty,
        scope_tys, depth_left - 1, guarded_left)
    fallback = _sample_short_nonrecursive_list_fallback(state, relation, target_ty,
        scope_tys, depth_left - 1, guarded_left, generation_query)
    fallback === nothing && (fallback = _replay_finite_core_certificate(state, relation, fallback_query))
    fallback isa _UnsupportedEvidenceSampling && return nothing
    fallback === nothing && return nothing
    # A type certificate is not a proof of the enclosing occurrence/filter
    # request. Only its actual occurrences may discharge those obligations.
    any(index -> derivation_uses_scope_binder(fallback, index), forbidden) && return nothing
    body_parent = _without_satisfied_scope_binder_requirements(generation_query, source.derivation)
    body_parent = _without_satisfied_scope_binder_requirements(body_parent, fallback)
    success_scope = _list_access_success_scope(source, scope_tys)
    body_query = _nested_generation_query(state, body_parent, target_ty, success_scope,
        depth_left - 1, guarded_left)
    body_query = _derive_generation_query(body_query; capability_mask=(:safe_list_access,))
    # This is a bounded complete-candidate builder even when called by the
    # root frontier's preparer. The root controller is not opened/reset here.
    # A failed preference leaves the original proof-backed OR replay intact.
    if state.completion.probability == 0 && !state.completion.completing &&
       _list_recursion_bias_enabled(relation.weights)
        preferred_query = _with_list_recursion_hints(relation, source, generation_query, body_query)
        body = _sample_list_recursive_body(state, relation, target_ty, success_scope,
            depth_left - 1, guarded_left, preferred_query)
        if body !== nothing
            result = SafeListAccessDerivation(source.derivation, source.ty, body, fallback, target_ty)
            _generation_requirements_satisfied(result, generation_query) ||
                error("biased safe-list composition lost an occurrence obligation")
            return result
        end
    end
    # Select an OR branch before proving it. A positive first certificate does
    # not need the other proxy's graph; failure may try the other branch once.
    first_proxy = rand(state.rng, Bool) ? 0 : 1
    for proxy in (first_proxy, 1 - first_proxy)
        query = _derive_generation_query(body_query; requirements=(UsesScopeBinder(proxy),))
        prepared = _prepare_generation_obligations(relation, query; certificate_only=true)
        prepared isa _UnsupportedEvidenceSampling && continue
        prepared === nothing && continue
        cost = prepared.graph.costs[prepared.root]
        _consume_sampling_nodes!(state, cost) || return nothing
        body = _obligation_replay(prepared.graph, prepared.root, prepared.scope, prepared.forbidden)
        result = SafeListAccessDerivation(source.derivation, source.ty, body, fallback, target_ty)
        _generation_requirements_satisfied(result, generation_query) ||
            error("certified safe-list composition lost an occurrence obligation")
        return result
    end
    return nothing
end

"""Try a complete constrained mixed candidate without preparing old head menus.

Uses the default fixed source catalogue/type families and existing depth/node
charges. Lambda prefixes rebase the exact request once per binder. Ordinary
heads compose selected complete child packets; at least one child contains a
safe-list access. Body evidence is a positive core sublanguage; its absence or
expense never prunes the caller's legacy mixed route. This structural condition
does not certify a FunctionDef filter. Custom planners/collectors, unsupported
type worlds and active completion modes retain their existing boundary. Source
menus/materialization in this preparer never fall back to complete legacy
source-witness enumeration. It may run between root boundaries with a temporary
zero-probability state: it is finite preparation, not a new root exploration.
"""
function _sample_certified_required_candidate(
    state::DerivationSamplingState, relation::CoreRelation, query::GenerationQuery,
    allow_safe_list_access::Bool,
)
    relation, query = _relation_for_query(relation, query)
    allow_safe_list_access && _safe_list_access_enabled(relation) || return nothing
    provider = _safe_list_access_provider(relation.providers)
    isempty(provider.source_collectors) || return nothing
    relation.argument_planner isa DefaultArgumentPlanner || return nothing
    # This is a bounded mixed candidate, not a prepared pending-frontier step.
    state.completion.probability == 0 && !state.completion.completing || return nothing
    _finite_evidence_request_supported(relation, query) || return nothing
    available = min(state.nodes_left, query.node_budget)
    outside = state.nodes_left - available
    state.nodes_left = available
    succeeded = false
    try
        target = query.target_ty
        scope = _scope_type_vector(query.scope_tys)
        if target isa TyFun && query.depth_left > 0 && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
            _consume_sampling_nodes!(state, 1) || return nothing
            body_query = _derive_generation_query(query; target_ty=target.res,
                added_scope_tys=(target.arg,), depth_left=query.depth_left - 1,
                node_budget=state.nodes_left)
            body_query = _advance_functiondef_prefix(body_query)
            body = _sample_certified_required_candidate(state, relation, body_query, allow_safe_list_access)
            body === nothing && return nothing
            succeeded = true
            return LambdaDerivation(target.arg, body)
        end
        query.depth_left > 1 || return nothing
        choice_query = query
        query = _close_functiondef_prefix(query)
        # These are step descriptors, not a menu of recursively prepared
        # witnesses. A selected head prepares just one obligation allocation.
        # Source options can have one concrete NamedTuple schema. Ordinary
        # selected heads use another: keep the mixed menu's element type at
        # the runtime descriptor boundary, not the first source's schema.
        options = _certified_mixed_candidate_options(relation, query, scope)
        options isa _UnsupportedEvidenceSampling && return nothing
        options = _prefer_state_helper_body_options(options, relation, choice_query)
        pool, total = _positive_option_pool(options)
        while !isempty(pool) && total > 0
            index = _weighted_option_index(state.rng, pool, total)
            option = pool[index]
            state.nodes_left = available
            result = if option.kind == :safe_list_access
                _sample_safe_list_access(state, relation, option.value.source, target,
                    scope, query.depth_left - 1, query.guarded_left, query;
                    body_sampler=_sample_certified_safe_list_body, evidence_only=true)
            else
                _sample_certified_productive_head(state, relation, option.plan.rule,
                    query, query.depth_left - 1, allow_safe_list_access)
            end
            if option.kind == :productive_head
                # Do not recursively retry every ordinary head. Even a finite
                # depth would make b failing heads expand into about b^depth
                # failed packets. One selected structure/allocation is a
                # bounded attempt; its failure leaves the old caller intact.
                succeeded = result !== nothing
                return result
            end
            if result !== nothing
                succeeded = true
                return result
            end
            total -= option.weight
            deleteat!(pool, index)
        end
        return nothing
    finally
        succeeded || (state.nodes_left = available)
        state.nodes_left += outside
    end
end

"""List mixed step descriptors without preparing child witnesses.

Sources are anchored in the relation's fixed source catalogue and concrete
scope; ordinary heads use the current owned evidence world. `query` provides
the closed target, context, capabilities and bounds. Cached source menus are
borrowed read-only and copied into a runtime-schema vector before adding head
descriptors. Unsupported worlds return the marker; no failed query is cached.
"""
function _certified_mixed_candidate_options(relation::CoreRelation,
    query::GenerationQuery, scope::Vector{Ty})
    options = NamedTuple[]
    sources = _safe_list_access_generation_options(relation, query.target_ty, scope; evidence_only=true)
    sources isa _UnsupportedEvidenceSampling && return sources
    append!(options, sources)
    rules = _certified_mixed_head_rules(relation, query)
    rules isa _UnsupportedEvidenceSampling && return rules
    for rule in rules
        isempty(rule.argument_tys) && continue
        _certified_mixed_rule_supported(relation, rule) || continue
        push!(options, (kind=:productive_head, plan=(rule=rule,),
            weight=_head_option_weight(relation, rule)))
    end
    return options
end

_certified_mixed_rule_supported(relation::CoreRelation, rule::CoreRule) =
    relation.argument_planner isa DefaultArgumentPlanner &&
    _metadata_get(rule.head.metadata, :argument_planner, nothing) === nothing &&
    !any(==(:source_array), rule.argument_classes)

function _certified_mixed_head_rules(relation::CoreRelation, query::GenerationQuery)
    scope = _scope_type_vector(query.scope_tys)
    world = _finite_evidence_world(relation, query.target_ty, scope; sampling=true)
    forbidden = _forbidden_scope_binders(query)
    context = _evidence_context!(world, Int[_evidence_type_id(world, ty)
        for (slot, ty) in enumerate(scope) if slot - 1 ∉ forbidden])
    catalog = _evidence_catalog_state(world, relation, context,
        _evidence_type_id(world, query.target_ty))
    world.unsupported_sampling && return _UnsupportedEvidenceSampling()
    # Neither legacy catalogue nor its derived-rule cache has the checked
    # external-context version. Borrow descriptors only from this owned world.
    return catalog.rules
end

"""Choose one placement, then build a complete positive child packet.

The exact query supplies scope identities, use/call and forbidden obligations,
capabilities and original bounds. One child is selected to contain safe-list
access; each positive binder requirement is assigned to one argument without
independent reachability preparation. The packet helper is also used by the
legacy selected-head consumer as a complete-packet proposal; the latter no
longer prepares an independent reachability/base-witness matrix.
`require_safe_child` is a structural condition for the dedicated safe candidate,
not a parent's use/call requirement. The real selected-head consumer disables
that extra condition; failure of a core child may still try a mixed certificate.
Source-array arguments and custom argument policies remain unsupported here,
not excluded from the legacy language. Failure means only this selected packet
was not obtained; it is neither a query-negative fact nor a mixed cost bound.
"""
function _sample_certified_productive_head(state::DerivationSamplingState,
    relation::CoreRelation, rule::CoreRule, query::GenerationQuery,
    child_depth::Int, allow_safe_list_access::Bool; require_safe_child::Bool=true)
    count = length(rule.argument_tys)
    count > 0 || return nothing
    _certified_mixed_rule_supported(relation, rule) || return nothing
    assigned = _distribute_scope_requirements(
        _required_scope_binder_indices(query, length(query.scope_tys)), count, state.rng)
    return _materialize_selected_mixed_head_packet(state, relation, rule, query,
        child_depth, allow_safe_list_access, assigned,
        require_safe_child ? rand(state.rng, 1:count) : 0)
end

"""Materialize one exact step/allocation packet, refunding it atomically on failure.

`assigned` distributes only the parent's positive use/call identities;
`safe_child` marks one argument required to contain safe-list access; zero adds
no such structural condition, without prohibiting access. Other
children replay joint or ordinary core certificates and may fall back to the
same positive mixed builder. No complete-witness Cartesian product, root-mode
restart, query-failure cache or minimum-mixed-cost inference is involved. The
original completion object is shared; this route is only admitted in bounded
exploration with zero switch probability. Complete finite child terms are the
packet's replay evidence. Successful charges are retained exactly once.
"""
function _materialize_selected_mixed_head_packet(state::DerivationSamplingState,
    relation::CoreRelation, rule::CoreRule, query::GenerationQuery,
    child_depth::Int, allow_safe_list_access::Bool,
    assigned::Vector{Vector{Int}}, safe_child::Int)
    relation, query = _relation_for_query(relation, query)
    allow_safe_list_access && _safe_list_access_enabled(relation) || return nothing
    state.completion.probability == 0 && !state.completion.completing || return nothing
    _certified_mixed_rule_supported(relation, rule) || return nothing
    _known_head_admitted(relation, rule.head) || return nothing
    _bucket_enabled(relation.weights, rule.head.weight_bucket) || return nothing
    _head_has_value(rule.head, relation) || return nothing
    _finite_evidence_request_supported(relation, query) || return nothing
    child_depth >= 0 || return nothing
    child_depth < query.depth_left || return nothing
    current_rules = _certified_mixed_head_rules(relation, query)
    current_rules isa _UnsupportedEvidenceSampling && return nothing
    current = findfirst(current_rules) do candidate
        candidate.head.source == rule.head.source && candidate.head.name == rule.head.name &&
            _cache_type_key(candidate.head.type) == _cache_type_key(rule.head.type)
    end
    current === nothing && return nothing
    # A legacy selected descriptor may predate a mutable constant/provider
    # update. Use the current owned rule/value/policy, or decline this packet.
    rule = current_rules[current]
    _certified_mixed_rule_supported(relation, rule) || return nothing
    length(assigned) == length(rule.argument_tys) || throw(ArgumentError("one allocation is required per argument"))
    0 <= safe_child <= length(assigned) || throw(ArgumentError("safe child is outside the selected head"))
    available = min(state.nodes_left, query.node_budget)
    outside = state.nodes_left - available
    state.nodes_left = available
    succeeded = false
    try
        _consume_sampling_nodes!(state, 1) || return nothing
        args = Vector{CoreDerivation}(undef, length(assigned))
        scope = _scope_type_vector(query.scope_tys)
        for i in eachindex(assigned)
            parent = _scope_binder_requirement_subset(query, assigned[i])
            child = _nested_generation_query(state, parent, rule.argument_tys[i],
                scope, child_depth, query.guarded_left)
            saved_nodes = state.nodes_left
            arg = if i == safe_child
                _sample_certified_required_candidate(state, relation, child, allow_safe_list_access)
            elseif isempty(assigned[i])
                _replay_finite_core_certificate(state, relation, child)
            else
                progress = _sample_preplanned_functiondef_call!(state, relation, child)
                if progress !== nothing
                    progress
                else
                    prepared = _prepare_generation_obligations(relation, child;
                        certificate_only=true)
                    if prepared isa _UnsupportedEvidenceSampling || prepared === nothing
                        nothing
                    elseif _consume_sampling_nodes!(state, prepared.graph.costs[prepared.root])
                        _obligation_replay(prepared.graph, prepared.root, prepared.scope,
                            prepared.forbidden)
                    else
                        nothing
                    end
                end
            end
            arg isa _UnsupportedEvidenceSampling && (arg = nothing)
            if arg === nothing && i != safe_child
                state.nodes_left = saved_nodes
                arg = _sample_certified_required_candidate(state, relation, child, allow_safe_list_access)
            end
            arg === nothing && return nothing
            _fast_forced_argument_admitted(relation, rule, i, arg, scope) || return nothing
            args[i] = arg
        end
        result = HeadApplicationDerivation(rule.head, args)
        _generation_requirements_satisfied(result, query) || return nothing
        succeeded = true
        return result
    finally
        succeeded || (state.nodes_left = available)
        state.nodes_left += outside
    end
end

function _sample_proxy_using_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    ;
    allow_head_application::Bool=true,
    allow_safe_list_access::Bool=false,
    allow_direct_result::Bool=true,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
    accept_derivation=nothing,
)
    depth_left >= 0 || return nothing
    # Proxy-enforcing generation must remain constructive. Guarded expansions
    # are still available outside the safe-list body, but opening them here can
    # recursively search a very wide space before discovering a proxy use.
    fast_weight = get(relation.weights.overrides, :proxy_constructive, 9.0)
    general_weight = get(relation.weights.overrides, :proxy_general, 1.0)
    fast_first = fast_weight > 0 &&
        (general_weight <= 0 || rand(state.rng) * (fast_weight + general_weight) < fast_weight)

    if !fast_first && general_weight > 0
        saved_nodes = state.nodes_left
        general = _sample_from_proxy_using_options(
            state,
            relation,
            _proxy_using_general_options(
                relation,
                target_ty,
                scope_tys,
                depth_left,
                prefix_count,
                prefix_offset,
                allow_head_application,
                allow_safe_list_access,
                allow_direct_result,
                generation_query,
            ),
            scope_tys,
            depth_left,
            guarded_left,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
        )
        if general !== nothing
            if _proxy_derivation_accepted(general, accept_derivation)
                return general
            end
            state.nodes_left = saved_nodes
            return nothing
        end
        state.nodes_left = saved_nodes
    end

    if fast_weight > 0
        saved_nodes = state.nodes_left
        fast = _sample_fast_proxy_insertion_derivation(
            state,
            relation,
            target_ty,
            scope_tys,
            depth_left,
            guarded_left,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
            ;
            allow_direct_result,
            allow_head_application,
        )
        if fast !== nothing &&
           _generation_requirements_satisfied(fast, generation_query)
            if _proxy_derivation_accepted(fast, accept_derivation)
                return fast
            end
            state.nodes_left = saved_nodes
            return nothing
        end
        state.nodes_left = saved_nodes
    end

    fast_first || return nothing
    general_weight > 0 || return nothing
    saved_nodes = state.nodes_left
    general = _sample_from_proxy_using_options(
        state,
        relation,
        _proxy_using_general_options(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            prefix_count,
            prefix_offset,
            allow_head_application,
            allow_safe_list_access,
            allow_direct_result,
            generation_query,
        ),
        scope_tys,
        depth_left,
        guarded_left,
        prefix_count,
        prefix_offset,
        allow_safe_list_access,
        generation_query,
    )
    if general !== nothing
        if _proxy_derivation_accepted(general, accept_derivation)
            return general
        end
        state.nodes_left = saved_nodes
        return nothing
    end
    state.nodes_left = saved_nodes
    return nothing
end

function _proxy_using_general_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_head_application::Bool,
    allow_safe_list_access::Bool,
    allow_direct_result::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    options = copy(_sampling_options(
        relation,
        target_ty,
        scope_tys,
        depth_left;
        allow_head_application,
        lambda_kind=:lambda_generation,
    ))
    generation_query === nothing || append!(
        options,
        _generation_extension_options(relation, generation_query),
    )
    if depth_left > 0 && allow_safe_list_access
        append!(options, _safe_list_access_generation_options(relation, target_ty, scope_tys))
    end
    if !allow_direct_result
        filter!(options) do option
            !(option.kind == :variable &&
                _uses_scope_prefix(option.value, prefix_count, prefix_offset))
        end
    end
    return options
end

function _proxy_derivation_accepted(derivation::CoreDerivation, acceptance)
    acceptance === nothing && return true
    applicable(acceptance, derivation) || throw(ArgumentError(
        "accept_derivation must accept one CoreDerivation",
    ))
    accepted = acceptance(derivation)
    accepted isa Bool || throw(ArgumentError("accept_derivation must return Bool"))
    return accepted
end

function _sample_fast_proxy_insertion_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
    ;
    allow_direct_result::Bool=true,
    allow_head_application::Bool=true,
    accept_derivation=nothing,
)
    if allow_direct_result
        direct = CoreDerivation[
            derivation
            for derivation in _direct_proxy_derivations(
                target_ty,
                scope_tys,
                prefix_count,
                prefix_offset,
            )
            if _proxy_prefix_satisfied(derivation, prefix_count, prefix_offset) &&
               _generation_requirements_satisfied(derivation, generation_query)
        ]
        for index in _required_scope_binder_indices(generation_query, length(scope_tys))
            scope_ty = scope_tys[index + 1]
            iscompatible(scope_ty, target_ty) || continue
            derivation = VariableDerivation(index, scope_ty)
            _proxy_prefix_satisfied(derivation, prefix_count, prefix_offset) || continue
            _generation_requirements_satisfied(derivation, generation_query) || continue
            any(existing -> derivation_signature(existing) == derivation_signature(derivation), direct) ||
                push!(direct, derivation)
        end
        if !isempty(direct)
            saved_nodes = state.nodes_left
            derivation = _sample_materialized_derivation(state, relation, direct)
            if derivation !== nothing &&
               _proxy_derivation_accepted(derivation, accept_derivation)
                return derivation
            end
            state.nodes_left = saved_nodes
            return nothing
        end
    end

    if depth_left > 0 &&
       target_ty isa TyFun &&
       _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        saved_nodes = state.nodes_left
        _consume_sampling_nodes!(state, 1) || return nothing
        body_scope = Ty[target_ty.arg; scope_tys]
        body_query = _nested_generation_query(
            state,
            generation_query,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
        )
        body_query === nothing || (body_query = _advance_functiondef_prefix(body_query))
        body = _sample_fast_proxy_insertion_derivation(
            state,
            relation,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
            prefix_count,
            prefix_offset + 1,
            allow_safe_list_access,
            body_query,
            ;
            allow_direct_result=true,
            allow_head_application,
        )
        if body === nothing
            state.nodes_left = saved_nodes
            return nothing
        end
        derivation = LambdaDerivation(target_ty.arg, body)
        if _proxy_derivation_accepted(derivation, accept_derivation)
            return derivation
        end
        state.nodes_left = saved_nodes
        return nothing
    end

    generation_query === nothing ||
        (generation_query = _close_functiondef_prefix(generation_query))
    depth_left > 0 || return nothing
    safe_scale = get(relation.weights.overrides, :proxy_safe_list, 1.0)
    application_scale = get(relation.weights.overrides, :proxy_scope_application, 1.0)
    head_scale = get(relation.weights.overrides, :proxy_head, 1.0)
    required_head_scale = get(relation.weights.overrides, :proxy_required_head, 1.0)
    split_scale = get(relation.weights.overrides, :proxy_split, 1.0)
    safe_options = allow_safe_list_access && safe_scale > 0 ?
        _fast_required_safe_list_options(
            relation,
            target_ty,
            scope_tys,
            true,
            generation_query,
        ) :
        NamedTuple[]
    _, safe_weight = _positive_option_pool(safe_options)
    application_plans = allow_head_application &&
        application_scale > 0 &&
        _scope_application_weight(relation.weights) > 0 ?
        _fast_required_scope_application_plans(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            prefix_count,
            prefix_offset,
            generation_query,
        ) :
        NamedTuple[]
    _, application_weight = _positive_option_pool(application_plans)
    split_plans = allow_head_application && split_scale > 0 ?
        _fast_split_requirement_head_plans(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            prefix_count,
            prefix_offset,
            generation_query,
        ) :
        NamedTuple[]
    _, split_weight = _positive_option_pool(split_plans)
    head_routes = allow_head_application && head_scale > 0 ?
        _fast_proxy_head_routes(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            prefix_count,
            prefix_offset,
        ) :
        NamedTuple[]
    _, head_weight = _positive_option_pool(head_routes)
    routes = NamedTuple[
        (kind=:safe_list, weight=safe_weight * safe_scale),
        (kind=:scope_application, weight=application_weight * application_scale),
        (
            kind=:required_head,
            weight=allow_head_application ? head_scale * required_head_scale : 0.0,
        ),
        (kind=:split_head, weight=split_weight * split_scale),
        (kind=:head, weight=head_weight * head_scale),
    ]

    pool, total_weight = _positive_option_pool(routes)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        route = pool[chosen]
        saved_nodes = state.nodes_left
        derivation = if route.kind == :safe_list
            _sample_fast_required_safe_list_derivation(
                state,
                relation,
                safe_options,
                target_ty,
                scope_tys,
                depth_left,
                guarded_left,
                prefix_count,
                prefix_offset,
                generation_query,
            )
        elseif route.kind == :scope_application
            _sample_fast_required_scope_application_derivation(
                state,
                relation,
                target_ty,
                scope_tys,
                depth_left,
                guarded_left,
                prefix_count,
                prefix_offset,
                allow_safe_list_access,
                generation_query,
            )
        elseif route.kind == :required_head
            required_head_plans = _fast_required_productive_head_plans(
                relation,
                target_ty,
                scope_tys,
                depth_left,
                generation_query,
            )
            _sample_fast_required_productive_head_derivation(
                state,
                relation,
                required_head_plans,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
                generation_query,
            )
        elseif route.kind == :split_head
            _sample_fast_split_requirement_head_derivation(
                state,
                relation,
                target_ty,
                scope_tys,
                depth_left,
                guarded_left,
                prefix_count,
                prefix_offset,
                allow_safe_list_access,
                generation_query,
            )
        else
            _sample_fast_proxy_head_derivation(
                state,
                relation,
                head_routes,
                scope_tys,
                depth_left - 1,
                prefix_count,
                prefix_offset,
            )
        end
        if derivation !== nothing &&
           _proxy_prefix_satisfied(derivation, prefix_count, prefix_offset) &&
           _generation_requirements_satisfied(derivation, generation_query)
            if _proxy_derivation_accepted(derivation, accept_derivation)
                return derivation
            end
            state.nodes_left = saved_nodes
            return nothing
        end
        state.nodes_left = saved_nodes
        total_weight -= route.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

_proxy_prefix_satisfied(derivation::CoreDerivation, prefix_count::Int, prefix_offset::Int) =
    prefix_count <= 0 || _uses_scope_prefix(derivation, prefix_count, prefix_offset)

"""Build one-step constrained candidates, not a feasibility certificate.

The relation/query supply the exact scoped use/call obligations, capability and
provider policies and original bounds. Productive heads are cheap descriptors:
requirements are allocated after selection, without a preliminary independent
reachability/base-witness matrix. Actual children may fail finitely.
Illegal slots are rejected here; a nonempty menu may still fail materialization.
The caller makes one weighted construction proposal and refunds its failure,
not completion mode. A candidate descriptor is not a feasibility certificate.
"""
function _required_scope_certificate_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_indices::Tuple,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
    ;
    allow_safe_list_access::Bool=true,
)
    required = Tuple(sort!(unique!(Int[index for index in required_indices])))
    all(index -> 0 <= index < length(scope_tys), required) || return NamedTuple[]
    forbidden = generation_query === nothing ? () : _forbidden_scope_binders(generation_query)
    return _required_scope_lazy_certificate_options(
        relation,
        target_ty,
        scope_tys,
        depth_left,
        required,
        forbidden,
        generation_query,
        allow_safe_list_access,
    )
end

function _required_scope_query_cache_key(
    relation::CoreRelation,
    generation_query::Union{Nothing,GenerationQuery},
)
    generation_query === nothing && return :no_generation_query
    admitted = _admitted_providers(
        relation.generation_providers,
        generation_query.capability_mask,
    )
    return _generation_query_cache_key(generation_query, admitted)
end

function _required_scope_lazy_certificate_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required::Tuple{Vararg{Int}},
    forbidden::Tuple,
    generation_query::Union{Nothing,GenerationQuery},
    allow_safe_list_access::Bool,
)
    call_indices = Tuple(_required_scope_call_indices(generation_query, length(scope_tys)))
    query_cache_key = _required_scope_query_cache_key(relation, generation_query)
    cacheable = query_cache_key !== nothing
    key = (
        :required_scope_certificate,
        _known_key(scope_tys, target_ty),
        depth_left,
        required,
        call_indices,
        forbidden,
        allow_safe_list_access,
        query_cache_key,
    )
    cache_key = cacheable ? _StructuralKey(key) : nothing
    cached = cacheable ? get(relation.sampling_options, cache_key, nothing) : nothing
    cached === nothing || return cached

    options = NamedTuple[]
    if length(required) == 1 && _bucket_enabled(relation.weights, _VARIABLE_WEIGHT)
        index = only(required)
        if !(index in call_indices) &&
           !(index in forbidden) &&
           iscompatible(scope_tys[index + 1], target_ty)
            push!(options, (
                kind=:variable,
                value=VariableDerivation(index, scope_tys[index + 1]),
                weight=_bucket_weight(relation.weights, _VARIABLE_WEIGHT),
            ))
        end
    end

    if depth_left > 0 && target_ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        push!(options, (
            kind=:lambda,
            value=target_ty,
            weight=_bucket_weight(relation.weights, _LAMBDA_WEIGHT),
        ))
    end

    if depth_left > 0 && allow_safe_list_access
        for option in _fast_required_safe_list_options(
            relation,
            target_ty,
            scope_tys,
            true,
            generation_query,
        )
            push!(options, (
                kind=:safe_list_access,
                value=option.value,
                weight=option.weight,
            ))
        end
    end

    if depth_left > 0 &&
       generation_query !== nothing &&
       generation_query.provider_fuel > 0
        append!(options, _generation_extension_options(relation, generation_query))
    end

    guarded_weight = _bucket_weight(relation.weights, _GUARDED_EXPANSION_WEIGHT)
    if depth_left > 0 &&
       generation_query !== nothing &&
       generation_query.guarded_left > 0 &&
       guarded_weight > 0 &&
       any(_may_supply_guarded_heads, relation.providers)
        # Do not offer a guaranteed-empty group when no provider can add such
        # heads. Keep target-specific menus lazy, and unknown producers eligible:
        # a possible group may still fail its one bounded construction attempt.
        push!(options, (kind=:guarded_route, value=target_ty, weight=guarded_weight))
    end

    if depth_left > 0 && _scope_application_enabled(relation.weights)
        for plan in _fast_required_scope_application_plans(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            0,
            0,
            generation_query,
        )
            push!(options, (kind=:scope_application, plan, weight=plan.weight))
        end
    end

    if depth_left > 0
        for plan in _fast_required_productive_head_plans(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            generation_query,
        )
            # Allocate obligations only after selection, at depth_left - 1.
            # The recursive caller makes one proposal, not a failed-head tree.
            push!(options, (kind=:productive_head, plan=plan, weight=plan.weight))
        end
    end

    cacheable && (relation.sampling_options[cache_key] = options)
    return options
end

@testitem "CanonicalGeneration required menu excludes absent guarded producers lazily" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    struct UnknownGuardedProducer <: CG.AbstractHeadProvider end
    @test CG._may_supply_guarded_heads(UnknownGuardedProducer())
    target = TyFun(FloatTy, FloatTy)
    scope = Ty[target]
    query = CG.GenerationQuery(target; scope_tys=scope, depth_left=2,
        guarded_left=1, node_budget=11, requirements=(CG.CallsScopeBinder(0),))
    core = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    @test isempty(CG._guarded_sampling_options(core, target, scope))
    options = CG._required_scope_certificate_options(core, target, scope, 2, (0,), query;
        allow_safe_list_access=false)
    @test any(option -> option.kind == :lambda, options)
    @test !any(option -> option.kind == :guarded_route, options)
    @test !CG._may_supply_guarded_heads(CG.FunctionDefProvider())
    guarded = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.GuardedExpansionProvider()))
    # Bool -> Bool brings the condition type into the original universe but
    # cannot first supply a Bool value. COND still has a valid descriptor.
    # Descriptor admission must not become a child-feasibility preflight.
    scalar_scope = Ty[FloatTy, TyFun(BoolTy, BoolTy)]
    scalar_query = CG.GenerationQuery(FloatTy; scope_tys=scalar_scope, depth_left=2,
        guarded_left=1, node_budget=11, requirements=(CG.UsesScopeBinder(0),))
    admitted = CG._required_scope_certificate_options(guarded, FloatTy, scalar_scope, 2, (0,), scalar_query;
        allow_safe_list_access=false)
    @test any(option -> option.kind == :guarded_route, admitted)
    @test !any(key -> key isa Tuple && !isempty(key) && first(key) == :guarded,
        keys(guarded.sampling_options))
    @test any(option -> option.value.head.name == :COND,
        CG._guarded_sampling_options(guarded, FloatTy, scalar_scope))
    @test CG._finite_evidence_witness(guarded,
        CG.GenerationQuery(BoolTy; scope_tys=scalar_scope, depth_left=2, node_budget=11)) === nothing
end

"""Allocate each exact positive identity to one selected head argument.

This is a bounded construction choice, not a reachability proof. Every finite
placement has positive probability, including several identities in one child
and indirect use through a differently typed child. `indices` are concrete
query slots; their Use/Call meanings and forbids stay in the child queries.
No context/planner or base-witness preparation is performed here. A zero-arity
head can only take an empty obligation set; nonempty input is a caller error.
"""
function _distribute_scope_requirements(indices, count::Int, rng)
    count >= 0 || throw(ArgumentError("argument count must be nonnegative"))
    assigned = [Int[] for _ in 1:count]
    for index in indices
        count > 0 || throw(ArgumentError("a positive obligation needs an argument"))
        push!(assigned[rand(rng, 1:count)], index)
    end
    return assigned
end

@testitem "CanonicalGeneration required head candidates omit independent proof matrices" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    fun = TyFun(FloatTy, FloatTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, lambda=1,
            scope_application=1, constant=1))
    query = CG.GenerationQuery(fun; scope_tys=Ty[fun], depth_left=3,
        node_budget=8, requirements=(CG.CallsScopeBinder(0),))
    options = CG._required_scope_certificate_options(relation, fun, Ty[fun],
        3, (0,), query; allow_safe_list_access=false)
    @test any(o -> o.kind == :lambda, options)
    @test isempty(relation.known_witnesses)
    prepared_reachability = any(keys(relation.sampling_options)) do key
        value = key isa CG._StructuralKey ? key.value : key
        value isa Tuple && !isempty(value) && first(value) in
            (:required_scope_call_reachable, :required_scope_binder_reachable)
    end
    @test !prepared_reachability

    # Keep the direct forced route independent from generic allocation. The
    # former already admitted a direct parameter when variable weight is zero.
    scope = Ty[fun, FloatTy]
    direct = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, scope_application=1))
    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=1,
        node_budget=2, requirements=(CG.CallsScopeBinder(0), CG.UsesScopeBinder(1)))
    plans = CG._fast_required_scope_application_plans(direct, FloatTy, scope, 1, 0, 0, q)
    @test length(plans) == 2
    forced = only(filter(p -> p.forced == [1], plans))
    generic = only(filter(p -> p.forced == [-1], plans))
    @test isempty(forced.exact_choices)
    @test generic.exact_choices == [(index=1, arguments=[1])]
    @test all(p -> isempty(p.prefix_choices), plans) # no empty-set proxy OR
    @test all(p -> !haskey(p, :base_available) && !haskey(p, :required_arguments), plans)
    @test isempty(direct.known_witnesses)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 2)
    result = CG._materialize_fast_required_scope_application(state, direct,
        forced, scope, 0, 0, 0, 0, false, q)
    @test result isa CG.ScopeApplicationDerivation
    @test CG.derivation_calls_scope_binder(result, 0)
    @test CG.derivation_uses_scope_binder(result, 1)
    @test state.nodes_left == 0

    # A forbidden function head is not a legal candidate, even when its type
    # equals that of the permitted exact head.
    scope = Ty[fun, FloatTy, fun]
    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=1,
        node_budget=2, requirements=(CG.CallsScopeBinder(0), CG.ForbidsScopeBinder(2)))
    plans = CG._fast_required_scope_application_plans(direct, FloatTy, scope, 1, 0, 0, q)
    @test !isempty(plans)
    @test all(p -> p.rule.index != 2, plans)
end

@testitem "CanonicalGeneration selected head allocation preserves indirect joint and proxy uses" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    scope = Ty[FloatTy, FloatTy, FloatTy]
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:WRAP => identity, :ADD => ((x,y) -> x+y),
            :PRED => iszero, :FROM_BOOL => (b -> b ? 1.0 : 0.0)),
        global_consts_ty=Dict{Symbol,Any}(:WRAP => TyFun(FloatTy, FloatTy),
            :ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
            :PRED => TyFun(FloatTy, BoolTy), :FROM_BOOL => TyFun(BoolTy, FloatTy)),
        weights=CG.GenerationWeights(default=0, variable=1, constant=1))
    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=2,
        node_budget=4, requirements=(CG.UsesScopeBinder(0), CG.UsesScopeBinder(1),
            CG.ForbidsScopeBinder(2)))
    plans = CG._fast_required_productive_head_plans(relation, FloatTy, scope, 2, q)
    wrap = only(filter(p -> p.rule.head.name == :WRAP, plans))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 4,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._materialize_fast_required_productive_head(state, relation,
        wrap, scope, 1, 0, false, q)
    @test result !== nothing
    @test CG.derivation_uses_scope_binder(result, 0)
    @test CG.derivation_uses_scope_binder(result, 1)
    @test !CG.derivation_uses_scope_binder(result, 2)
    @test CG.derivation_node_count(result) == 4
    @test state.nodes_left == 0

    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=2,
        node_budget=3, requirements=(CG.UsesScopeBinder(0), CG.ForbidsScopeBinder(2)))
    plans = CG._fast_required_productive_head_plans(relation, FloatTy, scope, 2, q)
    from_bool = only(filter(p -> p.rule.head.name == :FROM_BOOL, plans))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 3,
        CG.GenerationCompletionState(probability=1.0))
    result = CG._materialize_fast_required_productive_head(state, relation,
        from_bool, scope, 1, 0, false, q)
    @test result !== nothing # Float identity is used through a Bool predicate
    @test CG.derivation_uses_scope_binder(result, 0)
    @test !CG.derivation_uses_scope_binder(result, 2)
    @test CG.derivation_node_count(result) == 3
    @test state.nodes_left == 0

    # One OR request chooses at least one of the two proxies, not both.
    scope = Ty[FloatTy, FloatTy, TyFun(FloatTy, FloatTy)]
    q = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=2,
        node_budget=8, requirements=(CG.CallsScopeBinder(2),))
    proxy = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=1))
    plans = CG._fast_required_scope_application_plans(proxy, FloatTy, scope, 2, 2, 0, q)
    @test any(p -> !isempty(p.prefix_choices), plans)
    for seed in 1:8
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 8)
        result = CG._sample_fast_required_scope_application_derivation(state,
            proxy, FloatTy, scope, 2, 0, 2, 0, false, q)
        @test result !== nothing
        @test CG.derivation_calls_scope_binder(result, 2)
        @test CG.derivation_uses_scope_binder(result, 0) || CG.derivation_uses_scope_binder(result, 1)
        @test CG.derivation_node_count(result) == 8 - state.nodes_left
    end
end

@testitem "CanonicalGeneration unproved required proposals do not enumerate failed head trees" begin
    using Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    using LambdaRegression.CanonicalGeneration
    const CG = LambdaRegression.CanonicalGeneration

    # Count descriptor draws, not elapsed time or cache entries. Reusing a
    # cached menu must not conceal repeated recursive materialization.
    mutable struct RequiredProposalRNG <: Random.AbstractRNG
        inner::Random.Xoshiro
        choices::Int
    end
    function Random.rand(rng::RequiredProposalRNG)
        rng.choices += 1
        return rand(rng.inner)
    end
    Random.rand(rng::RequiredProposalRNG, range::UnitRange{Int}) = rand(rng.inner, range)
    Random.rand(rng::RequiredProposalRNG, values::AbstractVector{Int}) = rand(rng.inner, values)

    for head_count in (1, 3, 7), depth in (1, 3, 6)
        constants = Dict{Symbol,Any}(Symbol("ID", i) => identity for i in 1:head_count)
        types = Dict{Symbol,Any}(name => TyFun(FloatTy, FloatTy) for name in keys(constants))
        relation = CG.CoreRelation(; global_consts=constants, global_consts_ty=types,
            providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
            weights=CG.GenerationWeights(default=0, variable=1, constant=1,
                safe_list_access=1, constraint_evidence=0))
        scope = Ty[BoolTy]
        query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=depth,
            node_budget=32, requirements=(CG.UsesScopeBinder(0),))
        rng = RequiredProposalRNG(Random.Xoshiro(800 + head_count + depth), 0)
        state = CG.DerivationSamplingState(rng, 32)
        # No Float base, no Bool-to-Float bridge, and no list source. Mixed
        # admission prevents pretending that core-negative proves mixed-negative.
        @test CG._sample_required_scope_certificate_derivation(state, relation,
            FloatTy, scope, depth, 0, true, query) === nothing
        @test rng.choices == depth
        @test state.nodes_left == 32
    end

    # The separate proxy/scoped entry must also draw only one unproved plan.
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=1,
            safe_list_access=1, constraint_evidence=0))
    scope = Ty[TyFun(FloatTy, FloatTy), TyFun(FloatTy, FloatTy), BoolTy]
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        node_budget=32, requirements=(CG.UsesScopeBinder(2),))
    rng = RequiredProposalRNG(Random.Xoshiro(1234), 0)
    state = CG.DerivationSamplingState(rng, 32)
    @test CG._sample_fast_required_scope_application_derivation(state, relation,
        FloatTy, scope, 5, 0, 0, 0, true, query) === nothing
    @test rng.choices == 5
    @test state.nodes_left == 32
end

function _sample_required_scope_certificate_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    required = Tuple(_required_scope_binder_indices(generation_query, length(scope_tys)))
    isempty(required) && return nothing
    # A bounded preference is not a core occurrence certificate. Keep its
    # concrete argument-source choices in the existing constructive planner.
    if _has_list_recursion_hint(generation_query)
        return _sample_required_scope_legacy_derivation(state, relation, target_ty,
            scope_tys, depth_left, guarded_left, allow_safe_list_access, generation_query)
    end
    route_observer = get(task_local_storage(), _REQUIRED_SCOPE_ROUTE_DIAGNOSTIC_KEY, nothing)
    if generation_query !== nothing
        nested = _nested_generation_query(state, generation_query, target_ty,
            scope_tys, depth_left, guarded_left)
        core_relation, core_query = _relation_for_query(relation, nested)
        if _pure_evidence_requirement_query(core_relation, core_query, allow_safe_list_access)
            core = _sample_finite_required_derivation(state, core_relation, core_query)
            if !(core isa _UnsupportedEvidenceSampling)
                return _record_required_scope_route!(core, :direct_evidence)
            end
        elseif !state.completion.completing && state.completion.probability == 0 &&
               allow_safe_list_access && _safe_list_access_enabled(core_relation)
            # Choose routes BEFORE preparing the legacy scope/head menus. The
            # retry pool is this call only; a core-negative certificate never
            # removes the legacy mixed route. The candidate builder itself
            # does not re-enter this dispatcher.
            evidence_weight = get(relation.weights.overrides, :constraint_evidence, 1.0)
            evidence_first = evidence_weight > 0 &&
                rand(state.rng) * (1.0 + evidence_weight) < evidence_weight
            if evidence_first
                candidate = _sample_certified_required_candidate(state, core_relation,
                    core_query, allow_safe_list_access)
                candidate === nothing || return _record_required_scope_route!(
                    candidate, :direct_evidence)
            end
            route_out = route_observer === nothing ? nothing : Ref(:legacy_mixed)
            derivation = _sample_required_scope_legacy_derivation(state, relation, target_ty,
                scope_tys, depth_left, guarded_left, allow_safe_list_access, generation_query;
                evidence_query=evidence_first || evidence_weight <= 0 ? nothing : core_query,
                evidence_relation=core_relation, route_out)
            derivation === nothing && return nothing
            route = route_out === nothing ? :legacy_mixed : route_out[]
            return _record_required_scope_route!(derivation, route)
        end
    end
    route_out = route_observer === nothing ? nothing : Ref(:legacy_mixed)
    derivation = _sample_required_scope_legacy_derivation(state, relation, target_ty,
        scope_tys, depth_left, guarded_left, allow_safe_list_access, generation_query;
        route_out)
    derivation === nothing && return nothing
    route = route_out === nothing ? :legacy_mixed : route_out[]
    return _record_required_scope_route!(derivation, route)
end

"""Try one weighted constrained descriptor, then the optional certified route.

The relation/query supply exact Use/Call/Forbid identities and the original
depth/node/provider policy. Allocations are sampled after selection. Failed
construction refunds nodes but not provider attempts or completion state.
Without preproved children, exhausting every head recursively can expand k
failing unary heads to k^depth attempts. One proposal retains every finite
successful route with positive probability under the built-in rule contract;
it may fail even when another descriptor would succeed. This is bounded
candidate construction, not a complete mixed decision or successful sampler.
"""
function _sample_required_scope_legacy_derivation(
    state::DerivationSamplingState, relation::CoreRelation, target_ty::Ty,
    scope_tys::Vector{Ty}, depth_left::Int, guarded_left::Int,
    allow_safe_list_access::Bool, generation_query::Union{Nothing,GenerationQuery};
    evidence_query=nothing, evidence_relation=relation, route_out=nothing,
)
    required = Tuple(_required_scope_binder_indices(generation_query, length(scope_tys)))
    options = _required_scope_certificate_options(
        relation,
        target_ty,
        scope_tys,
        depth_left,
        required,
        generation_query,
        ;
        allow_safe_list_access,
    )
    options = _prefer_state_helper_body_options(options, relation, generation_query)
    pool, total_weight = _positive_option_pool(options)
    saved_nodes = state.nodes_left
    if !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
        state.nodes_left = saved_nodes
        derivation = _materialize_required_scope_certificate(
            state,
            relation,
            option,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
        if derivation !== nothing &&
           _generation_requirements_satisfied(derivation, generation_query)
            if route_out !== nothing
                route_out[] = :legacy_mixed
            end
            return derivation
        end
    end
    state.nodes_left = saved_nodes
    if evidence_query !== nothing
        derivation = _sample_certified_required_candidate(state, evidence_relation,
            evidence_query, allow_safe_list_access)
        if derivation !== nothing && route_out !== nothing
            route_out[] = :evidence_fallback
        end
        return derivation
    end
    return nothing
end

function _materialize_required_scope_certificate(
    state::DerivationSamplingState,
    relation::CoreRelation,
    option::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    if option.kind == :variable
        _consume_sampling_nodes!(state, 1) || return nothing
        return option.value
    elseif option.kind == :lambda
        depth_left > 0 || return nothing
        _consume_sampling_nodes!(state, 1) || return nothing
        target_ty = option.value::TyFun
        body_scope = Ty[target_ty.arg; scope_tys]
        body_query = _nested_generation_query(
            state,
            generation_query,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
        )
        body_query === nothing || (body_query = _advance_functiondef_prefix(body_query))
        body = _sample_required_scope_certificate_derivation(
            state,
            relation,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
            allow_safe_list_access,
            body_query,
        )
        body === nothing && return nothing
        return LambdaDerivation(target_ty.arg, body)
    end

    generation_query === nothing ||
        (generation_query = _close_functiondef_prefix(generation_query))

    if option.kind == :scope_application
        return _materialize_fast_required_scope_application(
            state,
            relation,
            option.plan,
            scope_tys,
            depth_left - 1,
            guarded_left,
            0,
            0,
            allow_safe_list_access,
            generation_query,
        )
    elseif option.kind == :safe_list_access
        return _sample_safe_list_access(
            state,
            relation,
            option.value.source,
            option.value.target_ty,
            scope_tys,
            depth_left - 1,
            guarded_left,
            generation_query,
        )
    elseif option.kind == :guarded_route
        guarded_options = _guarded_sampling_options(relation, option.value, scope_tys)
        return _sample_from_options(
            state,
            relation,
            guarded_options,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
    elseif option.kind == :generation_extension || option.kind == :guarded_head
        return _sample_option(
            state,
            relation,
            option,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
    elseif option.kind == :productive_head
        return _materialize_fast_required_productive_head(
            state,
            relation,
            option.plan,
            scope_tys,
            depth_left - 1,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
    end
    return nothing
end

function _fast_required_productive_head_plans(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    depth_left > 0 || return NamedTuple[]
    exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
    isempty(exact_indices) && return NamedTuple[]
    key = (
        :fast_required_productive_head_plans,
        _known_key(scope_tys, target_ty),
        depth_left,
        Tuple(exact_indices),
    )
    cache_key = _StructuralKey(key)
    cached = get(relation.sampling_options, cache_key, nothing)
    cached === nothing || return cached

    plans = NamedTuple[]
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    catalog_key = _known_key(scope_tys, target_ty)
    for entry in _productive_rules_for_target(
        relation,
        catalog,
        catalog_key,
        target_ty,
    )
        rule = entry.rule
        isempty(rule.argument_tys) && continue

        push!(plans, (
            rule,
            weight=_head_option_weight(relation, rule),
        ))
    end
    relation.sampling_options[cache_key] = plans
    return plans
end

function _productive_rules_for_target(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
)
    key = (:productive_rules_for_target, catalog_key, typekey(target_ty))
    cache_key = _StructuralKey(key)
    cached = get(relation.sampling_options, cache_key, nothing)
    cached === nothing || return cached
    rules = NamedTuple[]
    for rule in catalog.rules
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        push!(rules, (rule=rule,))
    end
    relation.sampling_options[cache_key] = rules
    return rules
end

function _sample_fast_required_productive_head_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    plans::AbstractVector{<:NamedTuple},
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    depth_left > 0 || return nothing
    pool, total_weight = _positive_option_pool(plans)
    isempty(pool) && return nothing
    total_weight > 0 || return nothing
    plan = pool[_weighted_option_index(state.rng, pool, total_weight)]
    saved_nodes = state.nodes_left
    derivation = _materialize_fast_required_productive_head(
        state,
        relation,
        plan,
        scope_tys,
        depth_left - 1,
        guarded_left,
        allow_safe_list_access,
        generation_query,
    )
    if derivation === nothing
        state.nodes_left = saved_nodes
    end
    return derivation
end

function _materialize_fast_required_productive_head(
    state::DerivationSamplingState,
    relation::CoreRelation,
    plan::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    # A complete-packet proposal may finish directly. Otherwise the general
    # child constructor uses a sampled allocation, not another preliminary
    # reachability/base-witness matrix. Neither route proves mixed absence.
    if generation_query !== nothing && state.completion.probability == 0 &&
       !state.completion.completing && allow_safe_list_access && _safe_list_access_enabled(relation)
        evidence_weight = get(relation.weights.overrides, :constraint_evidence, 1.0)
        if evidence_weight > 0 && rand(state.rng) * (1.0 + evidence_weight) < evidence_weight
            packet = _sample_certified_productive_head(state, relation, plan.rule,
                generation_query, depth_left, allow_safe_list_access; require_safe_child=false)
            packet === nothing || return packet
        end
    end
    exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
    isempty(exact_indices) && return nothing
    rule = plan.rule
    isempty(rule.argument_tys) && return nothing
    assigned_exact = _distribute_scope_requirements(exact_indices,
        length(rule.argument_tys), state.rng)
    _consume_sampling_nodes!(state, 1) || return nothing
    grounded_parent = _without_scope_binder_requirements(generation_query)
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    for i in _argument_resolution_order(rule)
        assigned = assigned_exact[i]
        arg = if isempty(assigned)
            _sample_required_productive_free_argument(
                state,
                relation,
                rule,
                i,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
                grounded_parent,
                generation_query,
            )
        else
            child_parent = _scope_binder_requirement_subset(
                generation_query,
                assigned,
            )
            child_query = _nested_generation_query(
                state,
                child_parent,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                guarded_left,
            )
            candidate = _sample_required_scope_certificate_derivation(
                state,
                relation,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
                child_query,
            )
            if candidate !== nothing && !_fast_forced_argument_admitted(
                    relation,
                    rule,
                    i,
                    candidate,
                    scope_tys,
                )
                candidate = nothing
            end
            candidate
        end
        arg === nothing && return nothing
        _required_source_argument_admitted(rule, i, arg) || return nothing
        args[i] = arg
    end
    return HeadApplicationDerivation(rule.head, args)
end

function _sample_required_productive_free_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    grounded_parent::Union{Nothing,GenerationQuery},
    generation_query::Union{Nothing,GenerationQuery},
)
    saved_nodes = state.nodes_left
    candidate = _sample_nested_generation_derivation(
        state,
        relation,
        rule.argument_tys[i],
        scope_tys,
        depth_left,
        guarded_left,
        allow_safe_list_access,
        grounded_parent,
    )
    if candidate !== nothing &&
       _fast_forced_argument_admitted(relation, rule, i, candidate, scope_tys)
        return candidate
    end

    state.nodes_left = saved_nodes
    witness = _fast_required_head_base_witness(
        relation,
        rule,
        i,
        scope_tys,
        depth_left,
        generation_query,
    )
    if witness === nothing ||
       !_consume_sampling_nodes!(state, derivation_node_count(witness))
        return nothing
    end
    return witness
end

function _fast_required_head_base_witness(
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    grounded = _first_fast_proxy_other_argument_derivation(relation, rule, i, scope_tys)
    grounded === nothing || return grounded
    depth_left >= 0 || return nothing
    parent = _without_scope_binder_requirements(generation_query)
    query = if parent === nothing
        GenerationQuery(
            rule.argument_tys[i];
            scope_tys,
            depth_left,
            provider_fuel=depth_left,
        )
    else
        _derive_generation_query(
            parent;
            target_ty=rule.argument_tys[i],
            depth_left=min(depth_left, parent.depth_left),
            provider_fuel=min(depth_left, parent.provider_fuel),
        )
    end
    witness = _borrow_known_generation_witness(relation, query)
    witness === nothing && return nothing
    return _fast_forced_argument_admitted(relation, rule, i, witness, scope_tys) ?
        witness : nothing
end

function _fast_required_safe_list_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    allow_safe_list_access || return NamedTuple[]
    isempty(_required_scope_binder_indices(generation_query, length(scope_tys))) &&
        return NamedTuple[]
    return _safe_list_access_generation_options(relation, target_ty, scope_tys)
end

function _sample_fast_required_safe_list_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    options::AbstractVector{<:NamedTuple},
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    options = _prefer_state_helper_body_options(options, relation, generation_query)
    pool, total_weight = _positive_option_pool(options)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
        saved_nodes = state.nodes_left
        derivation = _sample_safe_list_access(
            state,
            relation,
            option.value.source,
            target_ty,
            scope_tys,
            depth_left - 1,
            guarded_left,
            generation_query,
        )
        if derivation !== nothing &&
           _uses_scope_prefix(derivation, prefix_count, prefix_offset) &&
           _generation_requirements_satisfied(derivation, generation_query)
            return derivation
        end
        state.nodes_left = saved_nodes
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

"""Try one scoped-head/placement proposal under the original child bounds.

The query retains exact binding requirements and the optional proxy prefix.
Forced and generic placements remain independent positive choices. A selected
plan may fail; refund nodes without retrying every unproved plan recursively
or resetting the provider/completion state. No new type demand is introduced.
"""
function _sample_fast_required_scope_application_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    depth_left > 0 || return nothing
    plans = _fast_required_scope_application_plans(
        relation,
        target_ty,
        scope_tys,
        depth_left,
        prefix_count,
        prefix_offset,
        generation_query,
    )
    pool, total_weight = _positive_option_pool(plans)
    if !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        plan = pool[chosen]
        saved_nodes = state.nodes_left
        exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
        required_head = plan.rule.index in exact_indices
        derivation = _materialize_fast_required_scope_application(
            state,
            relation,
            plan,
            scope_tys,
            depth_left - 1,
            guarded_left,
            prefix_count,
            prefix_offset,
            allow_safe_list_access && !required_head,
            generation_query,
        )
        derivation === nothing || return derivation
        state.nodes_left = saved_nodes
    end
    return nothing
end

function _fast_required_scope_application_plans(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    depth_left > 0 || return NamedTuple[]
    prefix_indices = _proxy_prefix_indices(scope_tys, prefix_count, prefix_offset)
    exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
    call_indices = _required_scope_call_indices(generation_query, length(scope_tys))
    isempty(prefix_indices) && isempty(exact_indices) && return NamedTuple[]
    query_cache_key = _required_scope_query_cache_key(relation, generation_query)
    cacheable = query_cache_key !== nothing
    key = (
        :fast_required_scope_application_plans,
        _known_key(scope_tys, target_ty),
        depth_left,
        Tuple(prefix_indices),
        Tuple(exact_indices),
        Tuple(call_indices),
        query_cache_key,
    )
    cache_key = cacheable ? _StructuralKey(key) : nothing
    cached = cacheable ? get(relation.sampling_options, cache_key, nothing) : nothing
    cached === nothing || return cached

    plans = NamedTuple[]
    for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
        _known_type_admitted(relation, rule.head_ty) || continue
        generation_query !== nothing &&
            rule.index in _forbidden_scope_binders(generation_query) && continue
        rule_plans = NamedTuple[]
        forced = _required_scope_argument_plan(
            rule,
            scope_tys,
            prefix_indices,
            exact_indices,
            call_indices,
        )
        forced === nothing || push!(
            rule_plans,
            (
                forced=forced,
                exact_choices=_RequiredScopeExactChoice[],
                prefix_choices=Int[],
            ),
        )

        remaining_exact = Int[index for index in exact_indices if index != rule.index]
        # Prefix is one OR obligation, not all proxy identities and not an
        # obligation when there is no prefix. A required identity in that
        # prefix already guarantees its occurrence wherever it is assigned.
        prefix_required = !isempty(prefix_indices) && rule.index ∉ prefix_indices &&
            !any(index -> index in prefix_indices, remaining_exact)
        if !isempty(remaining_exact) || prefix_required
            arguments = collect(eachindex(rule.argument_tys))
            exact_choices = _RequiredScopeExactChoice[
                (index=index, arguments=arguments) for index in remaining_exact]
            push!(rule_plans, (
                forced=fill(-1, length(rule.argument_tys)),
                exact_choices,
                prefix_choices=prefix_required ? arguments : Int[],
            ))
        end

        isempty(rule_plans) && continue
        plan_weight = _scope_application_option_weight(relation, rule) / length(rule_plans)
        for plan in rule_plans
            # These are candidates, not proof menus. The direct forced path
            # stays separate (including variable-weight-zero cases); generic
            # allocations are chosen only when this candidate is materialized.
            push!(plans, (rule=rule, plan..., weight=plan_weight))
        end
    end
    cacheable && (relation.sampling_options[cache_key] = plans)
    return plans
end

function _required_scope_application_base_witness(
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    if rule.argument_classes[i] == :source_array
        _bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) || return nothing
        return _first_grounded_derivation(relation, rule.argument_tys[i], scope_tys)
    end
    grounded = _first_fast_proxy_other_argument_derivation(relation, rule, i, scope_tys)
    grounded === nothing || return grounded
    depth_left >= 0 || return nothing
    parent = _without_scope_binder_requirements(generation_query)
    query = parent === nothing ?
        GenerationQuery(
            rule.argument_tys[i];
            scope_tys,
            depth_left,
            provider_fuel=depth_left,
        ) :
        _derive_generation_query(
            parent;
            target_ty=rule.argument_tys[i],
            depth_left=min(depth_left, parent.depth_left),
            provider_fuel=min(depth_left, parent.provider_fuel),
        )
    return _borrow_known_generation_witness(relation, query)
end

function _required_scope_binder_reachable(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    catalog_key = _known_key(scope_tys, target_ty)
    return _required_scope_binder_reachable_in_catalog(
        relation,
        catalog,
        catalog_key,
        target_ty,
        scope_tys,
        depth_left,
        required_index,
    )
end

function _required_scope_binder_reachable_in_catalog(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    key = (
        :required_scope_binder_reachable,
        catalog_key,
        _known_key(scope_tys, target_ty),
        depth_left,
        required_index,
    )
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return !isempty(cached)
    reachable = _compute_required_scope_binder_reachable(
        relation,
        catalog,
        catalog_key,
        target_ty,
        scope_tys,
        depth_left,
        required_index,
    )
    relation.sampling_options[key] = reachable ? [(reachable=true,)] : NamedTuple[]
    return reachable
end

function _compute_required_scope_binder_reachable(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    0 <= required_index < length(scope_tys) || return false
    required_ty = scope_tys[required_index + 1]
    _known_type_admitted(relation, required_ty) || return false
    iscompatible(required_ty, target_ty) && return true
    depth_left > 0 || return false
    if _scope_application_enabled(relation.weights)
        any(_scope_application_rules(target_ty, scope_tys, depth_left)) do rule
            rule.index == required_index && _known_type_admitted(relation, rule.head_ty)
        end && return true
    end
    if target_ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        _required_scope_binder_reachable_in_catalog(
            relation,
            catalog,
            catalog_key,
            target_ty.res,
            Ty[target_ty.arg; scope_tys],
            depth_left - 1,
            required_index + 1,
        ) && return true
    end

    for entry in _productive_rules_for_target(
        relation,
        catalog,
        catalog_key,
        target_ty,
    )
        rule = entry.rule
        for i in eachindex(rule.argument_tys)
            _required_scope_binder_reachable_in_catalog(
                relation,
                catalog,
                catalog_key,
                rule.argument_tys[i],
                scope_tys,
                depth_left - 1,
                required_index,
            ) && return true
        end
    end
    return false
end

function _required_scope_call_reachable(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    catalog_key = _known_key(scope_tys, target_ty)
    return _required_scope_call_reachable_in_catalog(
        relation,
        catalog,
        catalog_key,
        target_ty,
        scope_tys,
        depth_left,
        required_index,
    )
end

function _required_scope_call_reachable_in_catalog(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    key = (
        :required_scope_call_reachable,
        catalog_key,
        _known_key(scope_tys, target_ty),
        depth_left,
        required_index,
    )
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return !isempty(cached)
    reachable = _compute_required_scope_call_reachable(
        relation,
        catalog,
        catalog_key,
        target_ty,
        scope_tys,
        depth_left,
        required_index,
    )
    relation.sampling_options[key] = reachable ? [(reachable=true,)] : NamedTuple[]
    return reachable
end

function _compute_required_scope_call_reachable(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    required_index::Int,
)
    0 <= required_index < length(scope_tys) || return false
    required_ty = scope_tys[required_index + 1]
    _known_type_admitted(relation, required_ty) || return false
    depth_left > 0 || return false
    if _scope_application_enabled(relation.weights)
        for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
            _known_type_admitted(relation, rule.head_ty) || continue
            if rule.index == required_index && _scope_application_other_arguments_available(
                    relation,
                    rule,
                    0,
                    scope_tys,
                    depth_left - 1,
                )
                return true
            end
            for i in eachindex(rule.argument_tys)
                _required_scope_call_reachable_in_catalog(
                    relation,
                    catalog,
                    catalog_key,
                    rule.argument_tys[i],
                    scope_tys,
                    depth_left - 1,
                    required_index,
                ) || continue
                _scope_application_other_arguments_available(
                    relation,
                    rule,
                    i,
                    scope_tys,
                    depth_left - 1,
                ) && return true
            end
        end
    end
    if target_ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        _required_scope_call_reachable_in_catalog(
            relation,
            catalog,
            catalog_key,
            target_ty.res,
            Ty[target_ty.arg; scope_tys],
            depth_left - 1,
            required_index + 1,
        ) && return true
    end
    for entry in _productive_rules_for_target(
        relation,
        catalog,
        catalog_key,
        target_ty,
    )
        rule = entry.rule
        for argument_ty in rule.argument_tys
            _required_scope_call_reachable_in_catalog(
                relation,
                catalog,
                catalog_key,
                argument_ty,
                scope_tys,
                depth_left - 1,
                required_index,
            ) && return true
        end
    end
    return false
end

function _scope_application_other_arguments_available(
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    selected::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
)
    for i in eachindex(rule.argument_tys)
        i == selected && continue
        _required_scope_application_base_witness(
            relation,
            rule,
            i,
            scope_tys,
            depth_left,
            nothing,
        ) === nothing && return false
    end
    return true
end

function _proxy_prefix_indices(scope_tys, prefix_count::Int, prefix_offset::Int)
    first_index = max(prefix_offset, 0)
    last_index = min(length(scope_tys) - 1, prefix_offset + prefix_count - 1)
    first_index <= last_index || return Int[]
    return collect(first_index:last_index)
end

function _required_scope_binder_indices(
    generation_query::Union{Nothing,GenerationQuery},
    scope_length::Int,
)
    generation_query === nothing && return Int[]
    indices = Int[]
    for requirement in generation_query.requirements
        requirement isa Union{UsesScopeBinder,CallsScopeBinder} || continue
        requirement.index < scope_length || return Int[]
        requirement.index in indices || push!(indices, requirement.index)
    end
    return indices
end

function _required_scope_call_indices(
    generation_query::Union{Nothing,GenerationQuery},
    scope_length::Int,
)
    generation_query === nothing && return Int[]
    indices = Int[]
    for requirement in generation_query.requirements
        requirement isa CallsScopeBinder || continue
        requirement.index < scope_length || return Int[]
        requirement.index in indices || push!(indices, requirement.index)
    end
    return indices
end

function _required_scope_argument_plan(
    rule::ScopeApplicationRule,
    scope_tys::Vector{Ty},
    prefix_indices::Vector{Int},
    exact_indices::Vector{Int},
    call_indices::Vector{Int},
)
    any(index -> index != rule.index, call_indices) && return nothing
    forced = fill(-1, length(rule.argument_tys))
    exact_remaining = Int[
        index for index in exact_indices
        if index != rule.index && !(index in call_indices)
    ]
    _assign_exact_scope_arguments!(forced, exact_remaining, rule.argument_tys, scope_tys, 1) ||
        return nothing

    prefix_satisfied = rule.index in prefix_indices || any(index -> index in prefix_indices, forced)
    if !isempty(prefix_indices) && !prefix_satisfied
        pairs = Tuple{Int,Int}[]
        for argument_index in eachindex(rule.argument_tys)
            forced[argument_index] < 0 || continue
            for binder_index in prefix_indices
                iscompatible(scope_tys[binder_index + 1], rule.argument_tys[argument_index]) || continue
                push!(pairs, (argument_index, binder_index))
            end
        end
        isempty(pairs) && return nothing
        argument_index, binder_index = first(pairs)
        forced[argument_index] = binder_index
    end
    return forced
end

function _assign_exact_scope_arguments!(
    forced::Vector{Int},
    required::Vector{Int},
    argument_tys::Vector{Ty},
    scope_tys::Vector{Ty},
    position::Int,
)
    position > length(required) && return true
    binder_index = required[position]
    for argument_index in eachindex(argument_tys)
        forced[argument_index] < 0 || continue
        iscompatible(scope_tys[binder_index + 1], argument_tys[argument_index]) || continue
        forced[argument_index] = binder_index
        _assign_exact_scope_arguments!(
            forced,
            required,
            argument_tys,
            scope_tys,
            position + 1,
        ) && return true
        forced[argument_index] = -1
    end
    return false
end

function _materialize_fast_required_scope_application(
    state::DerivationSamplingState,
    relation::CoreRelation,
    plan::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    rule = plan.rule
    forced = plan.forced
    if generation_query !== nothing &&
       ((prefix_count == 0 &&
         _required_scope_binder_indices(generation_query, length(scope_tys)) == [rule.index]) ||
        _state_update_request(generation_query, rule.index) !== nothing)
        before_preferred = state.nodes_left
        preferred = _sample_preferred_list_recursive_call!(state, relation, rule,
            scope_tys, depth_left, guarded_left, allow_safe_list_access, generation_query)
        if preferred !== nothing
            _proxy_prefix_satisfied(preferred, prefix_count, prefix_offset) && return preferred
            state.nodes_left = before_preferred
        end
    end
    if generation_query !== nothing &&
       _required_full_self_marker(generation_query, rule.index,
           length(rule.argument_tys)) !== nothing &&
       _required_scope_binder_indices(generation_query, length(scope_tys)) == [rule.index] &&
       isempty(plan.exact_choices) && isempty(plan.prefix_choices) &&
       all(index -> index < 0, forced)
        progress = _sample_preplanned_functiondef_call!(state, relation, generation_query)
        progress === nothing || return progress
    end
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    assigned_exact = [Int[] for _ in rule.argument_tys]
    # Borrow read-only placement descriptors. Each exact identity and the
    # optional proxy OR choose one child independently, allowing co-location.
    for obligation in plan.exact_choices
        argument_index = rand(state.rng, obligation.arguments)
        push!(assigned_exact[argument_index], obligation.index)
    end
    prefix_argument = isempty(plan.prefix_choices) ? 0 : rand(state.rng, plan.prefix_choices)
    _consume_sampling_nodes!(state, length(rule.argument_tys)) || return nothing
    head_parent = _without_satisfied_scope_binder_requirements(
        generation_query,
        ScopeApplicationDerivation(
            rule.index,
            rule.head_ty,
            CoreDerivation[VariableDerivation(rule.index, rule.head_ty)],
        ),
    )
    grounded_parent = _without_scope_binder_requirements(generation_query)
    for i in _argument_resolution_order(rule)
        binder_index = forced[i]
        assigned = assigned_exact[i]
        arg = if !isempty(assigned) || i == prefix_argument
            child_parent = _scope_binder_requirement_subset(head_parent, assigned)
            child_query = _nested_generation_query(
                state,
                child_parent,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                guarded_left,
            )
            if i == prefix_argument
                _sample_fast_proxy_insertion_derivation(
                    state,
                    relation,
                    rule.argument_tys[i],
                    scope_tys,
                    depth_left,
                    guarded_left,
                    prefix_count,
                    prefix_offset,
                    allow_safe_list_access,
                    child_query,
                )
            else
                _sample_required_scope_certificate_derivation(
                    state,
                    relation,
                    rule.argument_tys[i],
                    scope_tys,
                    depth_left,
                    guarded_left,
                    allow_safe_list_access,
                    child_query,
                )
            end
        elseif binder_index >= 0
            _consume_sampling_nodes!(state, 1) || return nothing
            VariableDerivation(binder_index, scope_tys[binder_index + 1])
        else
            _sample_required_scope_free_argument(
                state,
                relation,
                rule,
                i,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
                grounded_parent,
                generation_query,
            )
        end
        arg === nothing && return nothing
        _required_source_argument_admitted(rule, i, arg) || return nothing
        args[i] = arg
    end
    return ScopeApplicationDerivation(rule.index, rule.head_ty, args)
end

function _required_source_argument_admitted(
    rule::CoreRule,
    i::Int,
    argument::CoreDerivation,
)
    rule.argument_classes[i] == :source_array || return true
    return _array_source_depth(argument) <= _source_array_depth_limit(rule)
end

function _required_source_argument_admitted(
    rule::ScopeApplicationRule,
    i::Int,
    argument::CoreDerivation,
)
    rule.argument_classes[i] == :source_array || return true
    return _array_source_depth(argument) == 0
end

function _sample_required_scope_free_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    grounded_parent::Union{Nothing,GenerationQuery},
    generation_query::Union{Nothing,GenerationQuery},
)
    if rule.argument_classes[i] == :source_array
        _bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) || return nothing
        return _sample_materialized_derivation(
            state,
            relation,
            _bounded_source_array_derivations(
                relation,
                rule.argument_tys[i],
                scope_tys,
                0,
            ),
        )
    end
    saved_nodes = state.nodes_left
    candidate = _sample_nested_generation_derivation(
        state,
        relation,
        rule.argument_tys[i],
        scope_tys,
        depth_left,
        guarded_left,
        allow_safe_list_access,
        grounded_parent,
    )
    candidate !== nothing && return candidate

    state.nodes_left = saved_nodes
    witness = _required_scope_application_base_witness(
        relation,
        rule,
        i,
        scope_tys,
        depth_left,
        generation_query,
    )
    if witness === nothing ||
       !_consume_sampling_nodes!(state, derivation_node_count(witness))
        return nothing
    end
    return witness
end

Base.@noinline Base.@nospecializeinfer function _scope_binder_requirement_subset(
    query::Union{Nothing,GenerationQuery},
    assigned_indices::Vector{Int},
)
    query === nothing && return nothing
    requirements = Any[]
    sizehint!(requirements, length(query.requirements))
    for index in 1:length(query.requirements)
        requirement = query.requirements[index]
        keep = !(requirement isa Union{UsesScopeBinder,CallsScopeBinder}) ||
            requirement.index in assigned_indices
        keep && push!(requirements, requirement)
    end
    return _checked_generation_query(
        query.target_ty,
        query.scope_tys,
        query.depth_left,
        query.guarded_left,
        query.provider_fuel,
        query.node_budget,
        query.capability_mask,
        requirements,
    )
end

function _sample_fast_split_requirement_head_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    plans = _fast_split_requirement_head_plans(
        relation,
        target_ty,
        scope_tys,
        depth_left,
        prefix_count,
        prefix_offset,
        generation_query,
    )

    pool, total_weight = _positive_option_pool(plans)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        plan = pool[chosen]
        saved_nodes = state.nodes_left
        derivation = _materialize_fast_split_requirement_head(
            state,
            relation,
            plan.rule,
            plan.proxy_i,
            plan.exact_i,
            scope_tys,
            depth_left - 1,
            guarded_left,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
        )
        if derivation !== nothing &&
           _uses_scope_prefix(derivation, prefix_count, prefix_offset) &&
           _generation_requirements_satisfied(derivation, generation_query)
            return derivation
        end
        state.nodes_left = saved_nodes
        total_weight -= plan.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

function _fast_split_requirement_head_plans(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    depth_left > 1 || return NamedTuple[]
    prefix_indices = _proxy_prefix_indices(scope_tys, prefix_count, prefix_offset)
    isempty(prefix_indices) && return NamedTuple[]
    exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
    isempty(exact_indices) && return NamedTuple[]
    any(index -> index in prefix_indices, exact_indices) && return NamedTuple[]
    key = (
        :fast_split_requirement_head_plans,
        _known_key(scope_tys, target_ty),
        depth_left,
        Tuple(prefix_indices),
        Tuple(exact_indices),
    )
    cache_key = _StructuralKey(key)
    cached = get(relation.sampling_options, cache_key, nothing)
    cached === nothing || return cached

    plans = NamedTuple[]
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        nargs = length(rule.argument_tys)
        nargs >= 2 || continue
        plan_weight = _head_option_weight(relation, rule) / (nargs * (nargs - 1))
        for proxy_i in eachindex(rule.argument_tys)
            for exact_i in eachindex(rule.argument_tys)
                proxy_i == exact_i && continue
                push!(
                    plans,
                    (
                        rule=rule,
                        proxy_i=proxy_i,
                        exact_i=exact_i,
                        weight=plan_weight,
                    ),
                )
            end
        end
    end
    relation.sampling_options[cache_key] = plans
    return plans
end

function _materialize_fast_split_requirement_head(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    proxy_i::Int,
    exact_i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool,
    generation_query::Union{Nothing,GenerationQuery},
)
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 1) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    proxy_parent = _without_scope_binder_requirements(generation_query)
    proxy_query = _nested_generation_query(
        state,
        proxy_parent,
        rule.argument_tys[proxy_i],
        scope_tys,
        depth_left,
        guarded_left,
    )
    exact_query = _nested_generation_query(
        state,
        generation_query,
        rule.argument_tys[exact_i],
        scope_tys,
        depth_left,
        guarded_left,
    )

    for i in _argument_resolution_order(rule)
        arg = if i == proxy_i
            _sample_fast_proxy_insertion_derivation(
                state,
                relation,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                guarded_left,
                prefix_count,
                prefix_offset,
                allow_safe_list_access,
                proxy_query,
            )
        elseif i == exact_i
            _sample_fast_required_scope_application_derivation(
                state,
                relation,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                guarded_left,
                0,
                0,
                allow_safe_list_access,
                exact_query,
            )
        else
            _sample_fast_proxy_other_argument(state, relation, rule, i, scope_tys)
        end
        if arg === nothing
            state.nodes_left = saved_nodes
            return nothing
        end
        if (i == proxy_i || i == exact_i) &&
           !_fast_forced_argument_admitted(relation, rule, i, arg, scope_tys)
            state.nodes_left = saved_nodes
            return nothing
        end
        args[i] = arg
    end
    return HeadApplicationDerivation(rule.head, args)
end

Base.@noinline Base.@nospecializeinfer function _without_scope_binder_requirements(
    query::Union{Nothing,GenerationQuery},
)
    query === nothing && return nothing
    requirements = Any[]
    sizehint!(requirements, length(query.requirements))
    for index in 1:length(query.requirements)
        requirement = query.requirements[index]
        keep = !(requirement isa Union{UsesScopeBinder,CallsScopeBinder})
        keep && push!(requirements, requirement)
    end
    return _checked_generation_query(
        query.target_ty,
        query.scope_tys,
        query.depth_left,
        query.guarded_left,
        query.provider_fuel,
        query.node_budget,
        query.capability_mask,
        requirements,
    )
end

Base.@noinline Base.@nospecializeinfer function _without_satisfied_scope_binder_requirements(
    query::Union{Nothing,GenerationQuery},
    derivation::CoreDerivation,
)
    query === nothing && return nothing
    requirements = Any[]
    sizehint!(requirements, length(query.requirements))
    removed_satisfied_requirement = false
    for index in 1:length(query.requirements)
        requirement = query.requirements[index]
        satisfied = if requirement isa UsesScopeBinder
            derivation_uses_scope_binder(derivation, requirement.index)
        elseif requirement isa CallsScopeBinder
            derivation_calls_scope_binder(derivation, requirement.index)
        else
            false
        end
        if satisfied
            removed_satisfied_requirement = true
        else
            push!(requirements, requirement)
        end
    end
    removed_satisfied_requirement || return query
    return _checked_generation_query(
        query.target_ty,
        query.scope_tys,
        query.depth_left,
        query.guarded_left,
        query.provider_fuel,
        query.node_budget,
        query.capability_mask,
        requirements,
    )
end

function _generation_requirements_satisfied(
    derivation::CoreDerivation,
    query::Union{Nothing,GenerationQuery},
)
    query === nothing && return true
    for requirement in query.requirements
        if requirement isa UsesScopeBinder
            derivation_uses_scope_binder(derivation, requirement.index) || return false
        elseif requirement isa CallsScopeBinder
            derivation_calls_scope_binder(derivation, requirement.index) || return false
        end
    end
    return true
end

function _fast_proxy_head_routes(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    key = (:fast_proxy_routes, _known_key(scope_tys, target_ty), depth_left, prefix_count, prefix_offset)
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return cached
    routes = NamedTuple[]
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        isempty(rule.argument_tys) && !_is_instance_factory_head(rule.head) && continue
        forced_indices = Int[]
        for i in eachindex(rule.argument_tys)
            _fast_proxy_lambda_or_direct_possible(
                relation,
                rule.argument_tys[i],
                scope_tys,
                depth_left - 1,
                prefix_count,
                prefix_offset,
            ) || continue
            _fast_proxy_other_arguments_available(relation, rule, scope_tys, i) || continue
            push!(forced_indices, i)
        end
        isempty(forced_indices) && continue
        route_weight = _head_option_weight(relation, rule) / length(forced_indices)
        for i in forced_indices
            push!(
                routes,
                (
                    kind=:fast_proxy_head,
                    value=(rule=rule, force_i=i),
                    weight=route_weight,
                ),
            )
        end
    end
    relation.sampling_options[key] = routes
    return routes
end

function _sample_fast_proxy_head_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    routes::AbstractVector{<:NamedTuple},
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    pool, total_weight = _positive_option_pool(routes)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        route = pool[chosen]
        saved_nodes = state.nodes_left
        derivation = _sample_fast_proxy_head_route(
            state,
            relation,
            route.value.rule,
            route.value.force_i,
            scope_tys,
            depth_left,
            prefix_count,
            prefix_offset,
        )
        derivation === nothing || return derivation
        state.nodes_left = saved_nodes
        total_weight -= route.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

function _sample_fast_proxy_head_route(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    force_i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    depth_left >= 0 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 1) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    for i in _argument_resolution_order(rule)
        arg =
            i == force_i ?
            _sample_fast_proxy_lambda_or_direct(
                state,
                relation,
                rule.argument_tys[i],
                scope_tys,
                depth_left,
                prefix_count,
                prefix_offset,
            ) :
            _sample_fast_proxy_other_argument(state, relation, rule, i, scope_tys)
        if arg === nothing
            state.nodes_left = saved_nodes
            return nothing
        end
        if i == force_i &&
           !_fast_forced_argument_admitted(relation, rule, i, arg, scope_tys)
            state.nodes_left = saved_nodes
            return nothing
        end
        args[i] = arg
    end
    return HeadApplicationDerivation(rule.head, args)
end

function _sample_fast_proxy_lambda_or_direct(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    direct = _direct_proxy_derivations(target_ty, scope_tys, prefix_count, prefix_offset)
    if !isempty(direct)
        return _sample_materialized_derivation(state, relation, direct)
    end
    depth_left > 0 &&
        target_ty isa TyFun &&
        _bucket_enabled(relation.weights, _LAMBDA_WEIGHT) ||
        return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 1) || return nothing
    body = _sample_fast_proxy_lambda_or_direct(
        state,
        relation,
        target_ty.res,
        Ty[target_ty.arg; scope_tys],
        depth_left - 1,
        prefix_count,
        prefix_offset + 1,
    )
    if body === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    return LambdaDerivation(target_ty.arg, body)
end

function _fast_proxy_lambda_or_direct_possible(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    _has_direct_proxy_derivation(target_ty, scope_tys, prefix_count, prefix_offset) && return true
    depth_left > 0 &&
        target_ty isa TyFun &&
        _bucket_enabled(relation.weights, _LAMBDA_WEIGHT) ||
        return false
    return _fast_proxy_lambda_or_direct_possible(
        relation,
        target_ty.res,
        Ty[target_ty.arg; scope_tys],
        depth_left - 1,
        prefix_count,
        prefix_offset + 1,
    )
end

function _fast_proxy_other_arguments_available(
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    force_i::Int,
)
    for i in eachindex(rule.argument_tys)
        i == force_i && continue
        isempty(_fast_proxy_other_argument_derivations(relation, rule, i, scope_tys)) && return false
    end
    return true
end

function _sample_fast_proxy_other_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
)
    derivations = _fast_proxy_other_argument_derivations(relation, rule, i, scope_tys)
    return _sample_materialized_derivation(state, relation, derivations)
end

function _fast_forced_argument_admitted(
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    argument::CoreDerivation,
    scope_tys::Vector{Ty},
)
    if rule.argument_classes[i] == :source_array &&
       _array_source_depth(argument) > _source_array_depth_limit(rule)
        return false
    end
    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    planner === nothing && (planner = relation.argument_planner)
    planner isa DefaultArgumentPlanner && return true
    signature = derivation_signature(argument)
    candidates = copy(_raw_argument_derivations(
        relation,
        rule.argument_tys[i],
        rule.argument_classes[i],
        rule,
        scope_tys,
    ))
    any(candidate -> derivation_signature(candidate) == signature, candidates) ||
        push!(candidates, argument)
    planned = _plan_argument_derivations(
        relation,
        rule,
        candidates,
        rule.argument_tys[i],
        rule.argument_classes[i],
    )
    return any(candidate -> derivation_signature(candidate) == signature, planned)
end

"""Select the first grounded argument only for the original pointwise policy.

The relation provides the default planner, constants and catalogue; the rule
supplies its argument class and source-depth policy. Whole-vector or local
planners retain the original complete candidate vector, including reordering
and failures. Source capability and invalid depth metadata are still checked.
"""
function _first_fast_proxy_other_argument_derivation(
    relation::CoreRelation, rule::CoreRule, i::Int, scope_tys::Vector{Ty})
    source = rule.argument_classes[i] == :source_array
    source && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && return nothing
    if relation.argument_planner isa DefaultArgumentPlanner &&
        _metadata_get(rule.head.metadata, :argument_planner, nothing) === nothing
        if source
            _source_array_depth_limit(rule) # Validate even for leaf sources.
        end
        return _first_grounded_derivation(relation, rule.argument_tys[i], scope_tys)
    end
    candidates = _fast_proxy_other_argument_derivations(relation, rule, i, scope_tys)
    return isempty(candidates) ? nothing : first(candidates)
end

function _first_fast_proxy_other_argument_derivation(
    relation::CoreRelation, rule::ScopeApplicationRule, i::Int, scope_tys::Vector{Ty})
    rule.argument_classes[i] == :source_array &&
        !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) && return nothing
    return _first_grounded_derivation(relation, rule.argument_tys[i], scope_tys)
end

function _fast_proxy_other_argument_derivations(
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
)
    arg_ty = rule.argument_tys[i]
    arg_class = rule.argument_classes[i]
    if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
        return CoreDerivation[]
    end
    raw = _grounded_derivations(relation, arg_ty, scope_tys)
    return _plan_argument_derivations(relation, rule, raw, arg_ty, arg_class)
end

function _fast_proxy_other_argument_derivations(
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    i::Int,
    scope_tys::Vector{Ty},
)
    rule.argument_classes[i] == :source_array &&
        !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) &&
        return CoreDerivation[]
    return _grounded_derivations(relation, rule.argument_tys[i], scope_tys)
end

function _sample_from_proxy_using_options(
    state::DerivationSamplingState,
    relation::CoreRelation,
    options::Vector{<:NamedTuple},
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    options = _prefer_state_helper_body_options(options, relation, generation_query)
    pool, total_weight = _positive_option_pool(options)
    (isempty(pool) || total_weight <= 0) && return nothing
    option = pool[_weighted_option_index(state.rng, pool, total_weight)]
    saved_nodes = state.nodes_left
    derivation = _sample_proxy_using_option(
        state,
        relation,
        option,
        scope_tys,
        depth_left,
        guarded_left,
        prefix_count,
        prefix_offset,
        allow_safe_list_access,
        generation_query,
    )
    if derivation !== nothing && _generation_requirements_satisfied(derivation, generation_query)
        return derivation
    end
    # This is one stochastic proposal. The caller's constructive route provides
    # fallback without recursively enumerating the rejected proof frontier.
    state.nodes_left = saved_nodes
    return nothing
end

function _sample_proxy_using_option(
    state::DerivationSamplingState,
    relation::CoreRelation,
    option::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    if generation_query !== nothing &&
       option.kind != :lambda_known && option.kind != :lambda_generation
        generation_query = _close_functiondef_prefix(generation_query)
    end
    if option.kind == :variable
        _uses_scope_prefix(option.value, prefix_count, prefix_offset) || return nothing
        _consume_sampling_nodes!(state, 1) || return nothing
        return option.value
    elseif option.kind == :constant
        return nothing
    elseif option.kind == :lambda_known || option.kind == :lambda_generation
        depth_left > 0 || return nothing
        _consume_sampling_nodes!(state, 1) || return nothing
        target_ty = option.value::TyFun
        body_scope = Ty[target_ty.arg; scope_tys]
        body_query = _nested_generation_query(
            state,
            generation_query,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
        )
        body_query === nothing || (body_query = _advance_functiondef_prefix(body_query))
        body = _sample_proxy_using_derivation(
            state,
            relation,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
            prefix_count,
            prefix_offset + 1;
            allow_safe_list_access,
            generation_query=body_query,
        )
        body === nothing && return nothing
        return LambdaDerivation(target_ty.arg, body)
    elseif option.kind == :productive_head
        return _sample_proxy_using_head_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left,
            false,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
        )
    elseif option.kind == :scope_application
        derivation = _sample_scope_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
        derivation === nothing && return nothing
        _uses_scope_prefix(derivation, prefix_count, prefix_offset) || return nothing
        return derivation
    elseif option.kind == :guarded_head
        guarded_left > 0 || return nothing
        return _sample_proxy_using_head_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left - 1,
            true,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
        )
    elseif option.kind == :safe_list_access
        derivation = _sample_option(
            state,
            relation,
            option,
            scope_tys,
            depth_left,
            guarded_left,
            true,
            generation_query,
        )
        derivation === nothing && return nothing
        _uses_scope_prefix(derivation, prefix_count, prefix_offset) || return nothing
        return derivation
    elseif option.kind == :generation_extension
        generation_query === nothing && return nothing
        derivation = _sample_option(
            state,
            relation,
            option,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
            generation_query,
        )
        derivation === nothing && return nothing
        _uses_scope_prefix(derivation, prefix_count, prefix_offset) || return nothing
        return derivation
    end
    return nothing
end

function _sample_proxy_using_head_application(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    guarded_head::Bool,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    depth_left >= 0 || return nothing
    isempty(rule.argument_tys) && return nothing
    direct_indices = [
        i for i in eachindex(rule.argument_tys)
        if _has_direct_proxy_derivation(rule.argument_tys[i], scope_tys, prefix_count, prefix_offset)
    ]
    fallback_indices = [
        i for i in eachindex(rule.argument_tys)
        if !(i in direct_indices)
    ]
    force_indices = isempty(direct_indices) ? fallback_indices : direct_indices
    saved_nodes = state.nodes_left
    derivation = _sample_proxy_using_head_application_from_indices(
        state,
        relation,
        rule,
        scope_tys,
        depth_left,
        guarded_left,
        guarded_head,
        prefix_count,
        prefix_offset,
        force_indices,
        saved_nodes,
        allow_safe_list_access,
        generation_query,
    )
    derivation === nothing || return derivation
    if !isempty(direct_indices)
        return _sample_proxy_using_head_application_from_indices(
            state,
            relation,
            rule,
            scope_tys,
            depth_left,
            guarded_left,
            guarded_head,
            prefix_count,
            prefix_offset,
            fallback_indices,
            saved_nodes,
            allow_safe_list_access,
            generation_query,
        )
    end
    state.nodes_left = saved_nodes
    return nothing
end

function _sample_proxy_using_head_application_from_indices(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    guarded_head::Bool,
    prefix_count::Int,
    prefix_offset::Int,
    force_indices::Vector{Int},
    saved_nodes::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    while !isempty(force_indices)
        force_choice = rand(state.rng, eachindex(force_indices))
        force_i = force_indices[force_choice]
        state.nodes_left = saved_nodes
        _consume_sampling_nodes!(state, 1) || return nothing
        args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
        ok = true
        for i in _argument_resolution_order(rule)
            arg =
                i == force_i ?
                _sample_proxy_using_head_argument(
                    state,
                    relation,
                    rule,
                    i,
                    scope_tys,
                    depth_left,
                    guarded_left,
                    guarded_head,
                    prefix_count,
                    prefix_offset,
                    allow_safe_list_access,
                    generation_query,
                ) :
                _sample_head_argument(
                    state,
                    relation,
                    rule,
                    i,
                    scope_tys,
                    depth_left,
                    guarded_left,
                    guarded_head,
                    allow_safe_list_access,
                    generation_query,
                )
            if arg === nothing
                ok = false
                break
            end
            args[i] = arg
        end
        ok && return HeadApplicationDerivation(rule.head, args)
        deleteat!(force_indices, force_choice)
    end
    state.nodes_left = saved_nodes
    return nothing
end

"""Sample an anchored source containing at least one specified proxy.

Uses the ordinary joint occurrence evidence, with the caller's fixed source
depth and original node lease. Each concrete proxy is a positive OR route;
failed branches do not renew root mode or provider attempts. The parent query
supplies forbidden binders, capabilities and extension requirements. Parent
positive Use/Call obligations are checked on the completed head, not copied
into this forced argument (another sibling may satisfy them). Unknown source
policies return `_UnsupportedEvidenceSampling` without node charges so the
caller's explicit legacy policy still applies.
"""
function _sample_proxy_using_source_array(state::DerivationSamplingState,
    relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty},
    depth_left::Int, guarded_left::Int, source_depth::Int,
    prefix_count::Int, prefix_offset::Int,
    generation_query::Union{Nothing,GenerationQuery})
    parent = _without_scope_binder_requirements(generation_query)
    query = _nested_generation_query(state, parent, target_ty, scope_tys,
        depth_left, guarded_left)
    query === nothing && (query = GenerationQuery(target_ty; scope_tys,
        depth_left, guarded_left, node_budget=state.nodes_left,
        capability_mask=relation.capability_mask))
    forbidden = _forbidden_scope_binders(query)
    proxies = Int[index for index in prefix_offset:(prefix_offset + prefix_count - 1)
        if index < length(scope_tys) && index ∉ forbidden]
    Random.shuffle!(state.rng, proxies)
    for proxy in proxies
        required = _derive_generation_query(query; requirements=(UsesScopeBinder(proxy),))
        result = _sample_finite_required_derivation(state, relation, required; source_depth)
        result isa _UnsupportedEvidenceSampling && return result
        result === nothing || return result
    end
    return nothing
end

function _sample_proxy_using_head_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    guarded_head::Bool,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    arg_ty = rule.argument_tys[i]
    arg_class = rule.argument_classes[i]
    if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
        return nothing
    end
    direct_proxy_derivations = _direct_proxy_derivations(arg_ty, scope_tys, prefix_count, prefix_offset)
    if !isempty(direct_proxy_derivations)
        return _sample_materialized_derivation(state, relation, direct_proxy_derivations)
    end

    if guarded_head && _afoldl_step_argument(rule, i)
        return _sample_proxy_using_afoldl_step_argument(
            state,
            relation,
            rule,
            scope_tys,
            depth_left,
            guarded_left,
            prefix_count,
            prefix_offset,
            allow_safe_list_access,
            generation_query,
        )
    end

    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    if planner === nothing && arg_class == :source_array &&
       !(guarded_head && _afoldl_grounded_argument(rule, i))
        source = _sample_proxy_using_source_array(state, relation, arg_ty, scope_tys,
            depth_left, guarded_left, _source_array_depth_limit(rule),
            prefix_count, prefix_offset, generation_query)
        source isa _UnsupportedEvidenceSampling || return source
    end
    if planner !== nothing || arg_class == :source_array || (guarded_head && _afoldl_grounded_argument(rule, i))
        raw_derivations =
            guarded_head && _afoldl_grounded_argument(rule, i) ?
            _grounded_derivations(relation, arg_ty, scope_tys) :
            _raw_argument_derivations(relation, arg_ty, arg_class, rule, scope_tys)
        derivs =
            guarded_head ?
            _guarded_argument_derivations(relation, raw_derivations, arg_ty, arg_class, rule) :
            _plan_argument_derivations(relation, rule, raw_derivations, arg_ty, arg_class)
        proxy_derivs = _proxy_using_materialized_derivations(derivs, prefix_count, prefix_offset)
        return _sample_materialized_derivation(state, relation, proxy_derivs)
    end

    allow_head_application = !(guarded_head && typekey(arg_ty) != typekey(rule.result_ty))
    return _sample_proxy_using_derivation(
        state,
        relation,
        arg_ty,
        scope_tys,
        depth_left,
        guarded_left,
        prefix_count,
        prefix_offset;
        allow_head_application,
        allow_safe_list_access,
        generation_query,
    )
end

@testitem "CanonicalGeneration proxy source arguments use finite occurrence evidence" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, TyList, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    input = TyArray(TyNat(4), FloatTy)
    output = TyArray(TyNat(4), BoolTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.AMapProvider()),
        global_consts=Dict{Symbol,Any}(:predicate => iszero),
        global_consts_ty=Dict{Symbol,Any}(:predicate => TyFun(FloatTy, BoolTy)))
    head = CG.HeadEntry(:TOLIST, :primitive, TyFun(output, TyList(BoolTy)),
        CG.CoreProductive(), CG.WeightBucket(:tolist);
        metadata=(value=LambdaRegression.Primitives.TOLIST, source_array_depth_limit=1))
    rule = CG.CoreRule(head, TyList(BoolTy), Ty[output], [:source_array])
    scope = Ty[FloatTy, FloatTy, input, TyFun(FloatTy, FloatTy)]
    # The unrelated Call may be satisfied by a sibling. Forbid is global;
    # source generation must neither copy the Call nor erase the Forbid.
    parent = CG.GenerationQuery(output; scope_tys=scope, depth_left=0,
        node_budget=5, requirements=(CG.CallsScopeBinder(3), CG.ForbidsScopeBinder(0)))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    result = CG._sample_proxy_using_head_argument(state, relation, rule, 1, scope,
        0, 0, false, 2, 0, false, parent)
    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :AMAP
        @test CG.derivation_result_type(result) == output
        @test CG.derivation_uses_scope_binder(result, 1)
        @test !CG.derivation_uses_scope_binder(result, 0)
        @test !CG.derivation_calls_scope_binder(result, 3)
        @test CG.derivation_node_count(result) == 5
        @test CG._array_source_depth(result) == 1
    end
    @test state.nodes_left == 4
    # Both concrete OR branches remain available; this is not a conjunction
    # or same-type proxy quotient. Do not rely on a seed frequency assertion.
    for proxy in 0:1
        state = CG.DerivationSamplingState(Random.Xoshiro(1111), 5)
        source = CG._sample_proxy_using_head_argument(state, relation, rule, 1,
            scope, 0, 0, false, 1, proxy)
        @test source !== nothing
        @test source !== nothing && CG.derivation_uses_scope_binder(source, proxy)
        @test state.nodes_left == 0
    end
    blocked = CG._derive_generation_query(parent; requirements=(CG.ForbidsScopeBinder(1),))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    @test CG._sample_proxy_using_head_argument(state, relation, rule, 1,
        scope, 0, 0, false, 2, 0, false, blocked) === nothing
    @test state.nodes_left == 9
    @test isempty(relation.derivations)
    @test isempty(relation.source_derivations)
end

function _sample_proxy_using_afoldl_step_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    joint = _sample_joint_afoldl_step(state, relation, rule, scope_tys,
        depth_left, guarded_left, allow_safe_list_access, generation_query;
        prefix_count, prefix_offset)
    joint === nothing || return joint
    options = NamedTuple[]
    grounded = _proxy_using_materialized_derivations(
        _grounded_derivations(relation, first(rule.argument_tys), scope_tys),
        prefix_count,
        prefix_offset,
    )
    isempty(grounded) || push!(options, (kind=:grounded_step, value=grounded, weight=1.0))
    depth_left >= 2 && push!(options, (kind=:synthesized_step, value=rule, weight=1.0))

    pool, total_weight = _positive_option_pool(options)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
        saved_nodes = state.nodes_left
        derivation =
            option.kind == :grounded_step ?
            _sample_materialized_derivation(state, relation, option.value) :
            _sample_proxy_using_synthesized_afoldl_step(
                state,
                relation,
                option.value,
                scope_tys,
                depth_left,
                guarded_left,
                prefix_count,
                prefix_offset,
                allow_safe_list_access,
                generation_query,
            )
        derivation !== nothing && return derivation
        state.nodes_left = saved_nodes
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

function _sample_proxy_using_synthesized_afoldl_step(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
    allow_safe_list_access::Bool=false,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    parts = _afoldl_step_parts(rule)
    parts === nothing && return nothing
    depth_left >= 2 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 2) || return nothing
    body_scope = _afoldl_step_scope(rule, scope_tys)
    body_query = _nested_generation_query(
        state,
        generation_query === nothing ? nothing : _close_functiondef_prefix(generation_query),
        parts.body_ty,
        body_scope,
        depth_left - 2,
        guarded_left,
    )
    body = _sample_proxy_using_derivation(
        state,
        relation,
        parts.body_ty,
        body_scope,
        depth_left - 2,
        guarded_left,
        prefix_count,
        prefix_offset + 2;
        allow_safe_list_access,
        generation_query=body_query,
    )
    if body === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    return LambdaDerivation(parts.acc_ty, LambdaDerivation(parts.elem_ty, body))
end

function _proxy_using_materialized_derivations(
    derivations::Vector{<:CoreDerivation},
    prefix_count::Int,
    prefix_offset::Int,
)
    return CoreDerivation[
        derivation for derivation in derivations
        if _uses_scope_prefix(derivation, prefix_count, prefix_offset)
    ]
end

function _direct_proxy_derivations(target_ty::Ty, scope_tys::Vector{Ty}, prefix_count::Int, prefix_offset::Int)
    out = CoreDerivation[]
    last_index = min(length(scope_tys) - 1, prefix_offset + prefix_count - 1)
    for index in prefix_offset:last_index
        index >= 0 || continue
        proxy_ty = scope_tys[index + 1]
        iscompatible(proxy_ty, target_ty) || continue
        push!(out, VariableDerivation(index, proxy_ty))
    end
    return out
end

function _has_direct_proxy_derivation(target_ty::Ty, scope_tys::Vector{Ty}, prefix_count::Int, prefix_offset::Int)
    last_index = min(length(scope_tys) - 1, prefix_offset + prefix_count - 1)
    for index in prefix_offset:last_index
        index >= 0 || continue
        iscompatible(scope_tys[index + 1], target_ty) && return true
    end
    return false
end

function _list_access_sources(relation::CoreRelation, scope_tys::Vector{Ty})
    provider = _safe_list_access_provider(relation.providers)
    provider === nothing && return ListAccessSource[]
    key = (_scope_typekeys(scope_tys), provider.source_array_depth_limit)
    cache_key = _StructuralKey(key)
    cached = get(relation.list_access_sources, cache_key, nothing)
    cached === nothing || return cached
    out = ListAccessSource[]
    signatures = Set{Any}()
    _append_grounded_list_access_sources!(out, signatures, relation, scope_tys)

    for array_ty in _array_context_source_types(relation, scope_tys)
        list_ty = TyList(array_ty.elem)
        for array_source in _bounded_source_array_derivations(
            relation,
            array_ty,
            scope_tys,
            provider.source_array_depth_limit,
        )
            tolist_head = _safe_tolist_head(array_ty, list_ty)
            derivation = HeadApplicationDerivation(tolist_head, CoreDerivation[array_source])
            _push_list_access_source!(out, signatures, ListAccessSource(list_ty, derivation))
        end
    end
    _append_collected_list_access_sources!(out, signatures, relation, scope_tys, provider)
    relation.list_access_sources[cache_key] = out
    return out
end

function _append_grounded_list_access_sources!(out, signatures,
    relation::CoreRelation, scope_tys::Vector{Ty}; sampling::Bool=false)
    for (idx0, scope_ty) in enumerate(scope_tys)
        if scope_ty isa TyList
            _push_list_access_source!(out, signatures, ListAccessSource(scope_ty, VariableDerivation(idx0 - 1, scope_ty)))
        end
    end

    for list_ty in _list_context_source_types(relation)
        query = GenerationQuery(list_ty; scope_tys, depth_left=0)
        derivations = if sampling && _finite_evidence_request_supported(relation, query)
            # The outer menu version cannot repair a stale inner catalogue.
            # Use exactly the evidence world's versioned ownership boundary.
            world = _finite_evidence_world(relation, list_ty, scope_tys)
            _grounded_derivations(relation, list_ty, scope_tys,
                _evidence_root_catalog!(world, relation))
        else
            _grounded_derivations(relation, list_ty, scope_tys)
        end
        for derivation in derivations
            _push_list_access_source!(out, signatures, ListAccessSource(list_ty, derivation))
        end
    end

    return out
end

function _append_collected_list_access_sources!(out, signatures,
    relation::CoreRelation, scope_tys::Vector{Ty}, provider::SafeListAccessProvider)
    for collector in provider.source_collectors
        for raw_source in collector(relation, scope_tys)
            source = _normalize_list_access_source(raw_source, relation)
            source === nothing && continue
            _push_list_access_source!(out, signatures, source)
        end
    end
    return out
end

"""Prepare finite source descriptors without preparing array witness products.

This is a sampling menu, not the fixed concrete catalogue used by lowering.
It needs a closed scope/constant context, invariant head catalogues and the
default argument policy for deferred array families. Unsupported preparation
uses the original explicit source catalogue; custom collectors still supply
their own concrete identities. `evidence_only=true` instead declines unsupported
preparation, including custom collectors, without enumerating that catalogue.
It is used by finite certificate preparers, not as a language-negative result.
No mixed body is replaced by this source step.
"""
function _list_access_sampling_sources(relation::CoreRelation, scope_tys::Vector{Ty}; evidence_only::Bool=false)
    provider = _safe_list_access_provider(relation.providers)
    provider === nothing && return ListAccessSource[]
    evidence_only && !isempty(provider.source_collectors) && return _UnsupportedEvidenceSampling()
    array_types = _array_context_source_types(relation, scope_tys)
    if !(relation.argument_planner isa DefaultArgumentPlanner) ||
       any(array -> !_finite_evidence_request_supported(relation,
            GenerationQuery(array; scope_tys, depth_left=0)), array_types)
        return evidence_only ? _UnsupportedEvidenceSampling() : _list_access_sources(relation, scope_tys)
    end
    if evidence_only && any(list -> !_finite_evidence_request_supported(relation,
            GenerationQuery(list; scope_tys, depth_left=0)), _list_context_source_types(relation))
        return _UnsupportedEvidenceSampling()
    end
    concrete = ListAccessSource[]
    signatures = Set{Any}()
    _append_grounded_list_access_sources!(concrete, signatures, relation, scope_tys; sampling=true)
    _append_collected_list_access_sources!(concrete, signatures, relation, scope_tys, provider)
    out = Union{ListAccessSource,_ArrayListAccessSourcePlan}[concrete...]
    for array in array_types
        push!(out, _ArrayListAccessSourcePlan(array, provider.source_array_depth_limit))
    end
    return out
end

_normalize_list_access_source(source::ListAccessSource, relation::CoreRelation) = source

function _normalize_list_access_source(source::HeadEntry, relation::CoreRelation)
    source.type isa TyList || return nothing
    _head_has_value(source, relation) || return nothing
    return ListAccessSource(source.type, ConstantDerivation(source))
end

function _normalize_list_access_source(source::Pair, relation::CoreRelation)
    return _normalize_list_access_source((source.first, source.second), relation)
end

function _normalize_list_access_source(source::Tuple{<:TyList,<:CoreDerivation}, relation::CoreRelation)
    return ListAccessSource(source[1], source[2])
end

_normalize_list_access_source(source, relation::CoreRelation) =
    throw(ArgumentError("safe list source collectors must return HeadEntry, ListAccessSource, or (TyList, CoreDerivation) values; got $(typeof(source))"))

function _push_list_access_source!(
    sources::Vector{ListAccessSource},
    signatures::Set{Any},
    source::ListAccessSource,
)
    _contains_rigid_dummy(source.ty) && !_rigid_derivation_allowed(source.derivation) && return false
    signature = _derivation_signature(source.derivation)
    signature in signatures && return false
    push!(signatures, signature)
    push!(sources, source)
    return true
end

function _list_context_source_types(relation::CoreRelation)
    types = TyList[]
    keys = Set{Any}()
    for tyctx in (relation.local_consts_ty, relation.global_consts_ty)
        for ty in values(tyctx)
            ty isa TyList || continue
            key = typekey(ty)
            key in keys && continue
            push!(keys, key)
            push!(types, ty)
        end
    end
    return types
end

function _array_context_source_types(relation::CoreRelation, scope_tys::Vector{Ty})
    types = TyArray[]
    keys = Set{Any}()
    for ty in scope_tys
        _push_array_context_source_type!(types, keys, ty)
    end
    for tyctx in (relation.local_consts_ty, relation.global_consts_ty)
        for ty in values(tyctx)
            _push_array_context_source_type!(types, keys, ty)
        end
    end
    return types
end

function _push_array_context_source_type!(types::Vector{TyArray}, keys::Set{Any}, ty)
    ty isa TyArray || return false
    key = typekey(ty)
    key in keys && return false
    push!(keys, key)
    push!(types, ty)
    return true
end

function _list_access_success_scope(source::ListAccessSource, scope_tys::Vector{Ty})
    scope = Ty[source.ty, source.ty.elem]
    append!(scope, scope_tys)
    return scope
end

function _safe_tolist_head(array_ty::TyArray, list_ty::TyList)
    return HeadEntry(
        :TOLIST,
        :primitive,
        TyFun(array_ty, list_ty),
        CoreProductive(),
        _TOLIST_WEIGHT;
        metadata=(value=Primitives.TOLIST, source_array_depth_limit=0),
    )
end

function _safe_uncons_case_head(source_ty::TyList, result_ty::Ty)
    ok_ty = TyFun(source_ty.elem, TyFun(source_ty, result_ty))
    head_ty = TyFun(source_ty, TyFun(ok_ty, TyFun(result_ty, result_ty)))
    return HeadEntry(
        :UNCONS_CASE,
        :safe_list_access,
        head_ty,
        Decoration(),
        _SAFE_LIST_ACCESS_WEIGHT;
        metadata=(value=Primitives.UNCONS_CASE,),
    )
end

_uses_scope_prefix(derivation::CoreDerivation, prefix_count::Int) =
    _uses_scope_prefix(derivation, prefix_count, 0)

function _uses_scope_prefix(derivation::CoreDerivation, prefix_count::Int, offset::Int)
    prefix_count > 0 || return false
    return any(index -> derivation_uses_scope_binder(derivation, index), offset:(offset + prefix_count - 1))
end

function _uses_scope_prefix(derivation::VariableDerivation, prefix_count::Int, offset::Int)
    return offset <= derivation.index < offset + prefix_count
end

_uses_scope_prefix(::ConstantDerivation, ::Int, ::Int) = false

_uses_scope_prefix(derivation::LambdaDerivation, prefix_count::Int, offset::Int) =
    _uses_scope_prefix(derivation.body, prefix_count, offset + 1)

function _uses_scope_prefix(derivation::HeadApplicationDerivation, prefix_count::Int, offset::Int)
    return any(arg -> _uses_scope_prefix(arg, prefix_count, offset), derivation.args)
end

function _uses_scope_prefix(derivation::ScopeApplicationDerivation, prefix_count::Int, offset::Int)
    if offset <= derivation.index < offset + prefix_count
        return true
    end
    return any(arg -> _uses_scope_prefix(arg, prefix_count, offset), derivation.args)
end

function _uses_scope_prefix(derivation::SafeListAccessDerivation, prefix_count::Int, offset::Int)
    return _uses_scope_prefix(derivation.source, prefix_count, offset) ||
        _uses_scope_prefix(derivation.body, prefix_count, offset + 2) ||
        _uses_scope_prefix(derivation.fallback, prefix_count, offset)
end

@testitem "CanonicalGeneration first grounded witnesses preserve collection policy" begin
    using Test
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, TyMu, TyVar, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    values = Dict{Symbol,Any}(:seed=>1.0, :b=>true)
    types = Dict{Symbol,Any}(:seed=>FloatTy, :b=>BoolTy)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=values, global_consts_ty=types,
        weights=CG.GenerationWeights(variable=0))
    compare_first(rel, target, scope) = begin
        all = CG._grounded_derivations(rel, target, scope)
        first = CG._first_grounded_derivation(rel, target, scope)
        isempty(all) ? first === nothing :
            CG.derivation_signature(first) == CG.derivation_signature(all[1])
    end
    for (target, scope) in ((FloatTy, Ty[FloatTy, FloatTy]), (FloatTy, Ty[]),
            (BoolTy, Ty[FloatTy]), (TyFun(FloatTy, BoolTy), Ty[]),
            (TyFun(FloatTy, BoolTy), Ty[TyFun(FloatTy, BoolTy)]))
        @test compare_first(relation, target, scope)
    end
    binder = TyVar(9191, 0, nothing)
    recursive = TyMu(binder, TyFun(FloatTy, binder))
    @test compare_first(relation, recursive, Ty[recursive])
    @test CG._first_grounded_derivation(relation, FloatTy, Ty[FloatTy, FloatTy]).index == 0

    # The common variable witness does not need a catalogue at all.
    fresh = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    @test CG._first_grounded_derivation(fresh, FloatTy, Ty[FloatTy]) isa CG.VariableDerivation
    @test isempty(fresh.catalogs)
    head = CG.HeadEntry(:accept, :primitive, TyFun(FloatTy, BoolTy), CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=(x->true),))
    rule = CG.CoreRule(head, BoolTy, Ty[FloatTy], [:ordinary])
    scope = Ty[FloatTy, FloatTy]
    expected = first(CG._fast_proxy_other_argument_derivations(relation, rule, 1, scope))
    actual = CG._first_fast_proxy_other_argument_derivation(relation, rule, 1, scope)
    @test CG.derivation_signature(actual) == CG.derivation_signature(expected)
    @test CG._fast_required_head_base_witness(relation, rule, 1, scope, 0, nothing).index == 0

    # Whole-vector policies may choose the last entry, not the first leaf.
    seen = Int[]
    reverse_planner = (candidates, args...) -> (push!(seen, length(candidates)); reverse(candidates))
    custom = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=values, global_consts_ty=types, argument_planner=reverse_planner)
    actual = CG._first_fast_proxy_other_argument_derivation(custom, rule, 1, scope)
    expected = first(CG._fast_proxy_other_argument_derivations(custom, rule, 1, scope))
    @test CG.derivation_signature(actual) == CG.derivation_signature(expected)
    @test all(n -> n >= 2, seen)
    local_head = CG.HeadEntry(:accept, :primitive, TyFun(FloatTy, BoolTy), CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=(x->true), argument_planner=reverse_planner))
    local_rule = CG.CoreRule(local_head, BoolTy, Ty[FloatTy], [:ordinary])
    actual = CG._first_fast_proxy_other_argument_derivation(relation, local_rule, 1, scope)
    expected = first(CG._fast_proxy_other_argument_derivations(relation, local_rule, 1, scope))
    @test CG.derivation_signature(actual) == CG.derivation_signature(expected)

    array = TyArray(TyNat(4), FloatTy)
    array_head = CG.HeadEntry(:use_array, :primitive, TyFun(array, FloatTy), CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=sum, source_array_depth_limit=0))
    array_rule = CG.CoreRule(array_head, FloatTy, Ty[array], [:source_array])
    array_scope = Ty[array, array]
    actual = CG._first_fast_proxy_other_argument_derivation(relation, array_rule, 1, array_scope)
    expected = first(CG._fast_proxy_other_argument_derivations(relation, array_rule, 1, array_scope))
    @test CG.derivation_signature(actual) == CG.derivation_signature(expected)
    scoped_rule = CG.ScopeApplicationRule(2, TyFun(array, FloatTy), FloatTy, Ty[array], [:source_array])
    @test CG._required_scope_application_base_witness(relation, scoped_rule, 1,
        array_scope, 0, nothing).index == 0
    disabled = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(array_source=0))
    @test CG._first_fast_proxy_other_argument_derivation(disabled, array_rule, 1, array_scope) === nothing
    @test CG._required_scope_application_base_witness(disabled, scoped_rule, 1,
        array_scope, 0, nothing) === nothing
    invalid_head = CG.HeadEntry(:bad, :primitive, TyFun(array, FloatTy), CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=sum, source_array_depth_limit=-1))
    invalid_rule = CG.CoreRule(invalid_head, FloatTy, Ty[array], [:source_array])
    @test_throws ArgumentError CG._first_fast_proxy_other_argument_derivation(
        relation, invalid_rule, 1, array_scope)
    @test_throws ArgumentError CG._fast_proxy_other_argument_derivations(
        relation, invalid_rule, 1, array_scope)
end

@testitem "CanonicalGeneration constrained head menus defer unselected preparation" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, FloatTy, BoolTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode
    CG = LambdaRegression.CanonicalGeneration
    calls = Ref(0)
    planner = (candidates, args...) -> (calls[] += 1; candidates)
    values = Dict{Symbol,Any}(Symbol(:HEAD_, i) => (x -> y -> x < y) for i in 1:16)
    types = Dict{Symbol,TyLike}(name => TyFun(FloatTy, TyFun(FloatTy, BoolTy))
        for name in keys(values))
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, scope_application=1)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        global_consts=values, global_consts_ty=types, argument_planner=planner, weights=weights)
    scope = Ty[TyFun(FloatTy, FloatTy), FloatTy]
    query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=3,
        guarded_left=0, provider_fuel=3, node_budget=64,
        requirements=(CG.CallsScopeBinder(0),))
    options = CG._required_scope_certificate_options(relation, BoolTy, scope, 3, (0,), query)
    @test length(options) == 16
    @test calls[] == 0
    @test all(option -> option.kind == :productive_head &&
        !haskey(option.plan, :obligation_plan), options)
    @test isempty(relation.known_witnesses)
    @test isempty(relation.evidence_worlds)
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 64)
    result = CG._materialize_required_scope_certificate(state, relation,
        first(options), scope, 3, 0, false, query)
    @test result isa CG.HeadApplicationDerivation
    @test CG.derivation_calls_scope_binder(result, 0)
    @test 0 < calls[] < 16 # only the chosen head's arguments, not all 16 heads
    @test state.nodes_left == 64 - CG.derivation_node_count(result)
    node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(), result,
        relation.local_consts, relation.local_consts_ty, relation, Random.Xoshiro(2000))
    closed = AstTree(AbsNode(:outer, AbsNode(:inner, node)))
    @test iscompatible(infer(closed, merge(types, relation.local_consts_ty)),
        TyFun(FloatTy, TyFun(scope[1], BoolTy)))

    # A dead descriptor is retained without pretending it is a certificate.
    # The mixed retry path fails finitely, owns its pool and restores its charge.
    fun = scope[1]
    dead = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(:ISFUNC => (_ -> true)),
        global_consts_ty=Dict{Symbol,TyLike}(:ISFUNC => TyFun(fun, BoolTy)),
        weights=CG.GenerationWeights(default=0, variable=1, constant=1,
            scope_application=1, safe_list_access=1))
    impossible = CG.GenerationQuery(BoolTy; scope_tys=Ty[fun], depth_left=3,
        guarded_left=0, provider_fuel=3, node_budget=12,
        requirements=(CG.CallsScopeBinder(0),))
    menu = CG._required_scope_certificate_options(dead, BoolTy, Ty[fun], 3, (0,), impossible)
    @test length(menu) == 1
    @test !CG._pure_evidence_requirement_query(dead, impossible, true)
    failed = CG.DerivationSamplingState(Random.Xoshiro(1111), 12,
        CG.GenerationCompletionState(probability=0.0))
    failed.completion.completing = true
    @test CG._sample_required_scope_certificate_derivation(failed, dead, BoolTy,
        Ty[fun], 3, 0, true, impossible) === nothing
    @test failed.nodes_left == 12
    @test failed.completion.completing
    @test failed.completion.boundaries == 0
    @test length(menu) == 1
    @test CG._required_scope_certificate_options(dead, BoolTy, Ty[fun], 3, (0,), impossible) === menu
end

@testitem "CanonicalGeneration list source plans preserve concrete identities and budget" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyArray, TyList, TyNat, TyFun, FloatTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode
    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    scope = Ty[array, array]
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.AMapProvider(), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(:zero=>0.0),
        global_consts_ty=Dict{Symbol,TyLike}(:zero=>FloatTy))
    options = CG._safe_list_access_generation_options(relation, FloatTy, scope)
    @test length(options) == 1
    @test only(options).value.source isa CG._ArrayListAccessSourcePlan
    @test isempty(relation.source_derivations)
    plan = only(options).value.source
    selected = Set{Int}()
    for seed in 1:20
        state = CG.DerivationSamplingState(Random.Xoshiro(seed), 7)
        result = CG._sample_safe_list_access(state, relation, plan, FloatTy, scope, 3, 0)
        @test result isa CG.SafeListAccessDerivation
        @test state.nodes_left == 0
        @test CG.derivation_node_count(result) == 7
        @test result.source.head.name == :TOLIST
        @test only(result.source.args) isa CG.VariableDerivation
        push!(selected, only(result.source.args).index)
        node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(), result,
            relation.local_consts, relation.local_consts_ty, relation, Random.Xoshiro(seed))
        closed = AstTree(AbsNode(:outer, AbsNode(:inner, node)))
        expected = TyFun(array, TyFun(array, FloatTy))
        @test iscompatible(infer(closed,
            merge(relation.global_consts_ty, relation.local_consts_ty)), expected)
    end
    @test selected == Set([0, 1])
    @test isempty(relation.source_derivations)
    short = CG.DerivationSamplingState(Random.Xoshiro(2000), 6)
    @test CG._sample_safe_list_access(short, relation, plan, FloatTy, scope, 3, 0) === nothing
    @test short.nodes_left == 6
    parent = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
        node_budget=7, requirements=(CG.ForbidsScopeBinder(0), CG.UsesScopeBinder(1)))
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 7)
    result = CG._sample_safe_list_access(state, relation, plan, FloatTy, scope, 3, 0, parent)
    @test result !== nothing
    @test !CG.derivation_uses_scope_binder(result, 0)
    @test CG.derivation_uses_scope_binder(result, 1)
    for cap in (6, 7)
        larger_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 20)
        bounded = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3, node_budget=cap)
        access = CG._sample_safe_list_access(larger_state, relation, plan,
            FloatTy, scope, 3, 0, bounded)
        @test (access !== nothing) == (cap == 7)
        @test larger_state.nodes_left == (cap == 7 ? 13 : 20)
    end
end

@testitem "CanonicalGeneration list source menus retain policy and mutable context" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyArray, TyList, TyNat, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    array = TyArray(TyNat(4), FloatTy)
    providers = (CG.ContextHeadProvider(), CG.AMapProvider(),
        CG.SafeListAccessProvider(source_array_depth_limit=1))
    for weights in (CG.GenerationWeights(tolist=0), CG.GenerationWeights(array_source=0))
        relation = CG.CoreRelation(;providers, weights)
        options = CG._safe_list_access_generation_options(relation, FloatTy, Ty[array])
        @test all(option -> option.weight == 0, options)
        @test isempty(relation.source_derivations)
    end
    locals = Dict{Symbol,Any}()
    local_types = Dict{Symbol,TyLike}()
    relation = CG.CoreRelation(;providers, local_consts=locals, local_consts_ty=local_types)
    @test isempty(CG._safe_list_access_generation_options(relation, FloatTy, Ty[]))
    # CoreRelation owns copies of its external dictionaries.
    relation.local_consts[:input] = zeros(4)
    relation.local_consts_ty[:input] = array
    after = CG._safe_list_access_generation_options(relation, FloatTy, Ty[])
    @test length(after) == 1
    @test only(after).value.source isa CG._ArrayListAccessSourcePlan
    @test isempty(relation.source_derivations)
    # Populate the old query-only catalogue first: rebuilding the outer menu
    # must not accidentally reuse it after a same-type constant is added.
    list = TyList(FloatTy)
    changed = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        local_consts=Dict{Symbol,Any}(:a=>[1.0]),
        local_consts_ty=Dict{Symbol,TyLike}(:a=>list))
    CG._list_access_sources(changed, Ty[])
    before = CG._safe_list_access_generation_options(changed, FloatTy, Ty[])
    @test Set(option.value.source.derivation.head.name for option in before) == Set([:a])
    changed.local_consts[:b] = [2.0]
    changed.local_consts_ty[:b] = list
    added = CG._safe_list_access_generation_options(changed, FloatTy, Ty[])
    @test Set(option.value.source.derivation.head.name for option in added) == Set([:a, :b])
    changed.local_consts[:a] = zeros(4)
    changed.local_consts_ty[:a] = array
    moved = CG._safe_list_access_generation_options(changed, FloatTy, Ty[])
    @test count(option -> option.value.source isa CG._ArrayListAccessSourcePlan, moved) == 1
    @test Set(option.value.source.derivation.head.name for option in moved
        if option.value.source isa CG.ListAccessSource) == Set([:b])
    # An explicitly unknown argument policy keeps its concrete old catalogue.
    custom = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        argument_planner=(ds, args...)->ds)
    sources = CG._list_access_sampling_sources(custom, Ty[array])
    @test !isempty(sources)
    @test all(source -> source isa CG.ListAccessSource, sources)
end

@testitem "CanonicalGeneration certificate preparation declines legacy source enumeration" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration
    input = TyArray(TyNat(4), FloatTy)
    output = TyArray(TyNat(4), BoolTy)
    constructor = CG.HeadEntry(:JOIN_ARRAY, :test, TyFun(input, TyFun(input, output)),
        CG.CoreProductive(), CG.WeightBucket(:test);
        metadata=(value=((a, b) -> map(iszero, a)), source_array_constructor=true,
            source_array_argument_index=1, source_array_depth_limit=0))
    struct CertificateMultipleSourceFixture <: CG.AbstractHeadProvider
        head::CG.HeadEntry
    end
    CG.generation_catalog_scope_invariant(::CertificateMultipleSourceFixture) = true
    CG.add_heads!(provider::CertificateMultipleSourceFixture, heads, universe; kwargs...) =
        push!(heads, provider.head)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),
            CertificateMultipleSourceFixture(constructor), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(:TRUE => true),
        global_consts_ty=Dict{Symbol,Any}(:TRUE => BoolTy))
    scope = Ty[input]
    plan = CG._ArrayListAccessSourcePlan(output, 1)
    parent = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=4, node_budget=9)
    certified = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    @test CG._sample_safe_list_access(certified, relation, plan, BoolTy, scope,
        3, 0, parent; evidence_only=true) === nothing
    @test certified.nodes_left == 9
    @test isempty(relation.source_derivations)
    @test !certified.completion.completing
    # The original caller still has the same concrete multi-array route.
    old = CG.DerivationSamplingState(Random.Xoshiro(2000), 9)
    result = CG._sample_safe_list_access(old, relation, plan, BoolTy, scope, 3, 0, parent)
    @test result isa CG.SafeListAccessDerivation
    @test only(result.source.args).head.name == :JOIN_ARRAY
    @test CG.derivation_node_count(result) == 9
    @test old.nodes_left == 0
    @test !isempty(relation.source_derivations)

    # A separately cached old catalogue must not enter the evidence-only menu.
    custom = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        argument_planner=(ds, args...) -> ds)
    # Insertion into Vector{NamedTuple} storage may convert the first typed
    # vector. Borrow the stored menu on the subsequent hit, not that first value.
    CG._safe_list_access_generation_options(custom, FloatTy, scope)
    borrowed = CG._safe_list_access_generation_options(custom, FloatTy, scope)
    @test !isempty(borrowed)
    @test all(option -> option.value.source isa CG.ListAccessSource, borrowed)
    declined = CG._safe_list_access_generation_options(custom, FloatTy, scope; evidence_only=true)
    @test declined isa CG._UnsupportedEvidenceSampling
    @test CG._safe_list_access_generation_options(custom, FloatTy, scope) === borrowed
end

@testitem "FunctionDef certified public root reaches guarded self tail before switching" begin
    using Test, Random
    using LambdaRegression.TypeSystem: TyLike, TyFun, TyList, FloatTy, infer, iscompatible
    using LambdaRegression.Utils: Err
    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(TyList(FloatTy), FloatTy)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, lambda=1,
        scope_application=1, safe_list_access=1, constraint_evidence=1, function_def=1)
    providers = (CG.ContextHeadProvider(), CG.SafeListAccessProvider())
    constants = Dict{Symbol,Any}(:ZERO => 0.0)
    constant_types = Dict{Symbol,TyLike}(:ZERO => FloatTy)
    successes, rejected = Ref(0), Ref(0)
    for seed in 1:16
        calls = Ref(0)
        policy = candidate -> (calls[] += 1; CG.GenuineSelfCallFilter()(candidate))
        provider = CG.FunctionDefProvider(max_nesting=1, body_attempts=8, admissibility=policy)
        config = CG.GenerationConfig(rng=Random.Xoshiro(seed), providers=providers,
            weights=weights, sampling_depth_limit=8, sampling_node_budget=32)
        generated = CG.generate_functiondef_root_scoped(target;
            functiondef_provider=provider, config=config, global_consts=constants,
            global_consts_ty=constant_types, completion_probability=1)
        @test calls[] == 1 # one original final filter; no retry after rejection
        @test generated isa Union{CG.GenerationResult,Err}
        if generated isa CG.GenerationResult
            successes[] += 1
            signature = generated.metadata[:derivation]
            @test CG._signature_contains(signature, :safe_list_access)
            @test CG._signature_calls_scope_binder(signature[4], 0)
            @test generated.metadata[:certified_completion].switched
            @test generated.metadata[:certified_completion].boundaries == 1
            @test generated.metadata[:sampling_nodes_used] == 9
            @test iscompatible(infer(generated.tree,
                merge(constant_types, generated.local_const_types)), target)
        else
            rejected[] += 1
        end
    end
    @test successes[] > 0 # existing guarded self(tail), not an injected teacher
    @test rejected[] > 0 # core eta-forwarding remains a genuine failed attempt
end

@testitem "CanonicalGeneration certified safe-list body calls self on the tail" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, BoolTy, infer, iscompatible
    CG = LambdaRegression.CanonicalGeneration
    ListFloat = TyList(FloatTy)
    target = TyFun(ListFloat, FloatTy)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, lambda=1,
        scope_application=1, safe_list_access=1, constraint_evidence=1,
        function_def=1)
    providers = (CG.ContextHeadProvider(), CG.SafeListAccessProvider())
    constants = Dict{Symbol,Any}(:ZERO => 0.0)
    constant_types = Dict{Symbol,TyLike}(:ZERO => FloatTy)
    relation = CG.CoreRelation(; providers=providers, global_consts=constants,
        global_consts_ty=constant_types, weights=weights)
    query = CG.GenerationQuery(target; scope_tys=Ty[target, BoolTy], depth_left=6,
        guarded_left=0, provider_fuel=6, node_budget=8,
        requirements=(CG.CallsScopeBinder(0), CG.ForbidsScopeBinder(1)))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
    body = CG._sample_certified_required_candidate(state, relation, query, true)

    @test body isa CG.LambdaDerivation
    if body isa CG.LambdaDerivation
        access = body.body
        @test access isa CG.SafeListAccessDerivation
        if access isa CG.SafeListAccessDerivation
            @test access.body isa CG.ScopeApplicationDerivation
            if access.body isa CG.ScopeApplicationDerivation
                @test access.body.index == 3 # self, after the list/head proxies
                @test length(access.body.args) == 1
                @test only(access.body.args) isa CG.VariableDerivation
                @test only(access.body.args).index == 0 # the tail proxy
                @test !CG.derivation_uses_scope_binder(access.body, 1)
            end
        end
        @test CG.derivation_calls_scope_binder(body, 0)
        @test !CG.derivation_uses_scope_binder(body, 1)
        @test CG.derivation_node_count(body) == 8
        @test state.nodes_left == 16 - CG.derivation_node_count(body)

        zero_head = CG.HeadEntry(:ZERO, :test, FloatTy, CG.CoreProductive(),
            CG.WeightBucket(:constant); metadata=(value=0.0,))
        base = CG.LambdaDerivation(ListFloat, CG.ConstantDerivation(zero_head))
        candidate = CG.FunctionDefCandidate(target, body, base, 0)
        @test CG.DirectKnotFilter()(candidate)
        @test CG.GenuineSelfCallFilter()(candidate)
    end

    # A Calls certificate can come only from the empty-list fallback. It is
    # still rejected as nonproductive because it calls self on the unchanged
    # original list rather than the structurally smaller tail.
    manual_access = CG.SafeListAccessDerivation(
        CG.VariableDerivation(0, ListFloat),
        ListFloat,
        CG.VariableDerivation(1, FloatTy),
        CG.ScopeApplicationDerivation(1, target,
            CG.CoreDerivation[CG.VariableDerivation(0, ListFloat)]),
        FloatTy,
    )
    manual_body = CG.LambdaDerivation(ListFloat, manual_access)
    manual_zero = CG.HeadEntry(:ZERO, :test, FloatTy, CG.CoreProductive(),
        CG.WeightBucket(:constant); metadata=(value=0.0,))
    manual_base = CG.LambdaDerivation(ListFloat, CG.ConstantDerivation(manual_zero))
    manual_candidate = CG.FunctionDefCandidate(target, manual_body, manual_base, 0)
    @test CG.derivation_calls_scope_binder(manual_body, 0)
    @test !CG.DirectKnotFilter()(manual_candidate)
    @test !CG.GenuineSelfCallFilter()(manual_candidate)

    # Exercise the public forced-root path with the provider passed explicitly:
    # generate_functiondef_root_scoped otherwise replaces providers in config.
    provider = CG.FunctionDefProvider(max_nesting=1, body_attempts=8,
        admissibility=CG.GenuineSelfCallFilter())
    config = CG.GenerationConfig(rng=Random.Xoshiro(1234), providers=providers,
        weights=weights,
        sampling_depth_limit=8,
        sampling_node_budget=32)
    generated = CG.generate_functiondef_root_scoped(target;
        functiondef_provider=provider, config=config, global_consts=constants,
        global_consts_ty=constant_types)
    @test generated isa CG.GenerationResult
    if generated isa CG.GenerationResult
        signature = generated.metadata[:derivation]
        @test CG._signature_contains(signature, :safe_list_access)
        @test CG._signature_calls_scope_binder(signature[4], 0)
        @test iscompatible(infer(generated.tree,
            merge(constant_types, generated.local_const_types)), target)
    end
end

@testitem "CanonicalGeneration certified safe-list candidates respect forbids and root state" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    ListFloat = TyList(FloatTy)
    target = TyFun(ListFloat, FloatTy)
    providers = (CG.ContextHeadProvider(), CG.SafeListAccessProvider())
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        lambda=1, scope_application=1, safe_list_access=1)
    relation = CG.CoreRelation(; providers=providers, global_consts=Dict{Symbol,Any}(:ZERO=>0.0),
        global_consts_ty=Dict{Symbol,Any}(:ZERO=>FloatTy), weights=weights)
    source = CG.ListAccessSource(ListFloat, CG.VariableDerivation(0, ListFloat))

    source_forbidden = CG.GenerationQuery(FloatTy; scope_tys=Ty[ListFloat, target],
        depth_left=4, node_budget=12,
        requirements=(CG.CallsScopeBinder(1), CG.ForbidsScopeBinder(0)))
    source_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 12)
    @test CG._sample_safe_list_access(source_state, relation, source, FloatTy,
        Ty[ListFloat, target], 4, 0, source_forbidden;
        body_sampler=CG._sample_certified_safe_list_body) === nothing
    @test source_state.nodes_left == 12

    # With no Float constant and scope application disabled, the only fallback
    # is the forbidden Float binder; the candidate must be rejected and refunded.
    fallback_scope = Ty[ListFloat, FloatTy, target]
    fallback_relation = CG.CoreRelation(; providers=providers, weights=CG.GenerationWeights(
        default=0, variable=1, scope_application=0, safe_list_access=1))
    fallback_query = CG.GenerationQuery(FloatTy; scope_tys=fallback_scope,
        depth_left=4, node_budget=12,
        requirements=(CG.CallsScopeBinder(2), CG.ForbidsScopeBinder(1)))
    fallback_state = CG.DerivationSamplingState(Random.Xoshiro(42), 12)
    @test CG._sample_safe_list_access(fallback_state, fallback_relation, source,
        FloatTy, fallback_scope, 4, 0, fallback_query;
        body_sampler=CG._sample_certified_safe_list_body) === nothing
    @test fallback_state.nodes_left == 12

    exact_query = CG.GenerationQuery(target; scope_tys=Ty[target], depth_left=6,
        node_budget=7, requirements=(CG.CallsScopeBinder(0),))
    short_state = CG.DerivationSamplingState(Random.Xoshiro(42), 16)
    @test CG._sample_certified_required_candidate(short_state,
        CG.CoreRelation(; providers=providers, global_consts=Dict{Symbol,Any}(:ZERO=>0.0),
            global_consts_ty=Dict{Symbol,Any}(:ZERO=>FloatTy), weights=weights),
        exact_query, true) === nothing
    @test short_state.nodes_left == 16

    probability_completion = CG.GenerationCompletionState(probability=0.25)
    active_completion = CG.GenerationCompletionState()
    active_completion.completing = true
    for completion in (probability_completion, active_completion)
        guarded_relation = CG.CoreRelation(; providers=providers, global_consts=Dict{Symbol,Any}(:ZERO=>0.0),
            global_consts_ty=Dict{Symbol,Any}(:ZERO=>FloatTy), weights=weights)
        guarded_state = CG.DerivationSamplingState(Random.Xoshiro(555), 16, completion)
        @test CG._sample_certified_required_candidate(guarded_state,
            guarded_relation, exact_query, true) === nothing
        @test guarded_state.nodes_left == 16
        @test isempty(guarded_relation.evidence_worlds)
    end
end

@testitem "CanonicalGeneration mixed packet keeps root use in another child" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, BoolTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode
    CG = LambdaRegression.CanonicalGeneration

    ListFloat = TyList(FloatTy)
    scope = Ty[FloatTy, FloatTy, ListFloat] # same-type binders remain distinct
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        safe_list_access=1)
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(:LT => (x -> y -> x < y), :ZERO => 0.0),
        global_consts_ty=Dict{Symbol,TyLike}(
            :LT => TyFun(FloatTy, TyFun(FloatTy, BoolTy)),
            :ZERO => FloatTy,
        ),
        weights=weights,
    )
    query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=5,
        guarded_left=0, provider_fuel=5, node_budget=8,
        requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    catalog = CG._relation_catalog(relation, BoolTy, scope)
    choices = CG._productive_rules_for_target(relation, catalog,
        CG._known_key(scope, BoolTy), BoolTy)
    rule = only(filter(choice -> choice.rule.head.name == :LT, choices)).rule

    # Child one receives the exact positive binder identity; child two is the
    # selected safe-list child. A same-typed forbidden sibling cannot substitute.
    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 20)
    result = CG._materialize_selected_mixed_head_packet(state, relation, rule,
        query, 4, true, [Int[1], Int[]], 2)

    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :LT
        @test length(result.args) == 2
        @test result.args[1] isa CG.VariableDerivation
        @test (result.args[1]::CG.VariableDerivation).index == 1
        @test result.args[2] isa CG.SafeListAccessDerivation
        access = result.args[2]
        if access isa CG.SafeListAccessDerivation
            @test access.source isa CG.VariableDerivation
            @test (access.source::CG.VariableDerivation).index == 2
            @test access.body isa CG.VariableDerivation
            @test access.body isa CG.VariableDerivation &&
                (access.body::CG.VariableDerivation).index in (0, 1)
        end
        @test CG._generation_requirements_satisfied(result, query)
        @test !CG.derivation_uses_scope_binder(result, 0)
        @test CG.derivation_node_count(result) == 8
        @test state.nodes_left == 20 - CG.derivation_node_count(result)
        local_consts = Dict{Symbol,Any}()
        local_types = Dict{Symbol,TyLike}()
        node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
            result, local_consts, local_types, relation, Random.Xoshiro(1234))
        closed = AstTree(AbsNode(:list, AbsNode(:second, AbsNode(:first, node))))
        expected = TyFun(ListFloat, TyFun(FloatTy, TyFun(FloatTy, BoolTy)))
        @test iscompatible(infer(closed,
            merge(relation.global_consts_ty, local_types)), expected)
    end
end

@testitem "CanonicalGeneration mixed packet keeps call in safe body and proxy alternatives" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, BoolTy, infer, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode
    CG = LambdaRegression.CanonicalGeneration

    ListFloat = TyList(FloatTy)
    function_ty = TyFun(FloatTy, FloatTy)
    scope = Ty[function_ty, ListFloat]
    values = Dict{Symbol,Any}(
        :H => (x -> x > 0.0),
        :ZERO => 0.0,
    )
    types = Dict{Symbol,TyLike}(
        :H => TyFun(FloatTy, BoolTy),
        :ZERO => FloatTy,
    )
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        global_consts=values, global_consts_ty=types,
        weights=CG.GenerationWeights(default=0, variable=1, constant=1,
            scope_application=1, safe_list_access=1))
    query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=5,
        guarded_left=0, provider_fuel=5, node_budget=8,
        requirements=(CG.CallsScopeBinder(0),))
    rules = CG._certified_mixed_head_rules(relation, query)
    rule = only(filter(candidate -> candidate.head.name == :H, rules))
    state = CG.DerivationSamplingState(Random.Xoshiro(42), 8)
    result = CG._materialize_selected_mixed_head_packet(state, relation, rule,
        query, 4, true, [Int[0]], 1)

    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :H
        @test length(result.args) == 1
        access = only(result.args)
        @test access isa CG.SafeListAccessDerivation
        if access isa CG.SafeListAccessDerivation
            @test access.source isa CG.VariableDerivation
            @test (access.source::CG.VariableDerivation).index == 1
            @test access.fallback isa CG.ConstantDerivation
            @test (access.fallback::CG.ConstantDerivation).head.name == :ZERO
            @test CG.derivation_calls_scope_binder(access.body, 2)
            @test (CG.derivation_uses_scope_binder(access.body, 0) ||
                CG.derivation_uses_scope_binder(access.body, 1))
            @test CG._generation_requirements_satisfied(result, query)
        end
        @test CG.derivation_node_count(result) == 8
        @test state.nodes_left == 0
        local_consts = Dict{Symbol,Any}()
        local_types = Dict{Symbol,TyLike}()
        node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
            result, local_consts, local_types, relation, Random.Xoshiro(42))
        closed = AstTree(AbsNode(:list, AbsNode(:function, node)))
        expected = TyFun(ListFloat, TyFun(function_ty, BoolTy))
        @test iscompatible(infer(closed, merge(types, local_types)), expected)
    end
end

@testitem "CanonicalGeneration mixed packet refunds budget and preserves unsupported legacy route" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration

    ListFloat = TyList(FloatTy)
    scope = Ty[FloatTy, ListFloat]
    providers = (CG.ContextHeadProvider(), CG.SafeListAccessProvider())
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        safe_list_access=1)
    relation = CG.CoreRelation(providers=providers,
        global_consts=Dict{Symbol,Any}(:LT => (x -> y -> x < y), :ZERO => 0.0),
        global_consts_ty=Dict{Symbol,TyLike}(
            :LT => TyFun(FloatTy, TyFun(FloatTy, BoolTy)), :ZERO => FloatTy),
        weights=weights)
    catalog = CG._relation_catalog(relation, BoolTy, scope)
    rule = only(filter(choice -> choice.rule.head.name == :LT,
        CG._productive_rules_for_target(relation, catalog,
            CG._known_key(scope, BoolTy), BoolTy))).rule

    for (budget, expected_left) in ((7, 11), (8, 3))
        query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=5,
            guarded_left=0, provider_fuel=5, node_budget=budget,
            requirements=(CG.UsesScopeBinder(0),))
        state = CG.DerivationSamplingState(Random.Xoshiro(2000), 11)
        result = CG._materialize_selected_mixed_head_packet(state, relation, rule,
            query, 4, true, [Int[0], Int[]], 2)
        @test (result !== nothing) == (budget == 8)
        @test state.nodes_left == expected_left
        if result !== nothing
            @test CG.derivation_node_count(result) == 8
            @test CG._generation_requirements_satisfied(result, query)
        end
    end

    # A query capability mask disables the safe-list provider before any lease.
    blocked = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=5,
        guarded_left=0, provider_fuel=5, node_budget=8,
        capability_mask=(:safe_list_access,),
        requirements=(CG.UsesScopeBinder(0),))
    blocked_state = CG.DerivationSamplingState(Random.Xoshiro(42), 11)
    @test CG._materialize_selected_mixed_head_packet(blocked_state, relation, rule,
        blocked, 4, true, [Int[0], Int[]], 2) === nothing
    @test blocked_state.nodes_left == 11

    # Custom source collectors are outside the certified mixed sublanguage.
    # The public constrained sampler still reaches its explicit legacy route.
    collector = (rel, scoped) -> ((ListFloat,
        CG.VariableDerivation(0, ListFloat)),)
    custom_provider = CG.SafeListAccessProvider(source_collectors=(collector,))
    custom = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), custom_provider),
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
        global_consts_ty=Dict{Symbol,TyLike}(:ZERO => FloatTy),
        weights=CG.GenerationWeights(default=0, variable=1, constant=1,
            safe_list_access=1, constraint_evidence=0),
    )
    legacy_query = CG.GenerationQuery(FloatTy; scope_tys=Ty[ListFloat],
        depth_left=5, guarded_left=0, provider_fuel=5, node_budget=16,
        requirements=(CG.UsesScopeBinder(0),))
    certified_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 16)
    @test CG._sample_certified_required_candidate(certified_state, custom,
        legacy_query, true) === nothing
    @test certified_state.nodes_left == 16

    legacy_state = CG.DerivationSamplingState(Random.Xoshiro(1), 16)
    legacy = CG._sample_required_scope_certificate_derivation(legacy_state,
        custom, FloatTy, Ty[ListFloat], 5, 0, true, legacy_query)
    @test legacy isa CG.SafeListAccessDerivation
    if legacy isa CG.SafeListAccessDerivation
        @test legacy.source isa CG.VariableDerivation
        @test (legacy.source::CG.VariableDerivation).index == 0
        @test CG._generation_requirements_satisfied(legacy, legacy_query)
        @test legacy_state.nodes_left == 16 - CG.derivation_node_count(legacy)
    end
end

@testitem "CanonicalGeneration selected mixed packets refresh versioned head rules" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyList, FloatTy, IntTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration

    struct MixedPacketThreshold
        cutoff::Float64
    end
    (f::MixedPacketThreshold)(x) = x > f.cutoff

    ListFloat = TyList(FloatTy)
    relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        global_consts=Dict{Symbol,Any}(
            :H => MixedPacketThreshold(0.0), :XS => [0.0], :ZERO => 0.0),
        global_consts_ty=Dict{Symbol,TyLike}(
            :H => TyFun(IntTy, BoolTy), :XS => ListFloat, :ZERO => FloatTy),
        weights=CG.GenerationWeights(default=0, variable=1, constant=1,
            safe_list_access=1),
    )
    old_scope = Ty[IntTy]
    old_catalog = CG._relation_catalog(relation, BoolTy, old_scope)
    old_choices = CG._productive_rules_for_target(relation, old_catalog,
        CG._known_key(old_scope, BoolTy), BoolTy)
    old_rule = only(filter(choice -> choice.rule.head.name == :H, old_choices)).rule
    @test old_rule.argument_tys == Ty[IntTy]

    # Mutating a same-name context head to a new closed type and adding another
    # same-typed head must be visible to the query-owned catalogue, not the old
    # relation catalogue/rule cache.
    relation.global_consts[:H] = MixedPacketThreshold(1.0)
    relation.global_consts_ty[:H] = TyFun(FloatTy, BoolTy)
    relation.global_consts[:H2] = MixedPacketThreshold(-1.0)
    relation.global_consts_ty[:H2] = TyFun(FloatTy, BoolTy)
    scope = old_scope
    query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=5,
        guarded_left=0, provider_fuel=5, node_budget=16)
    owned = CG._certified_mixed_head_rules(relation, query)
    @test owned isa Vector{CG.CoreRule}
    @test any(rule -> rule.head.name == :H && rule.argument_tys == Ty[FloatTy], owned)
    @test any(rule -> rule.head.name == :H2 && rule.argument_tys == Ty[FloatTy], owned)

    stale_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 16)
    @test CG._materialize_selected_mixed_head_packet(stale_state, relation,
        old_rule, query, 4, true, [Int[]], 1) === nothing
    @test stale_state.nodes_left == 16

    current_rule = only(filter(rule -> rule.head.name == :H, owned))
    replacement = MixedPacketThreshold(-1.0)
    relation.global_consts[:H] = replacement # same TyFun(Float, Bool), new runtime value
    refreshed = CG._certified_mixed_head_rules(relation, query)
    @test any(rule -> rule.head.name == :H && rule.argument_tys == Ty[FloatTy], refreshed)
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 16)
    result = CG._materialize_selected_mixed_head_packet(state, relation,
        current_rule, query, 4, true, [Int[]], 1)
    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :H
        @test result.head.type == TyFun(FloatTy, BoolTy)
        @test only(result.args) isa CG.SafeListAccessDerivation
        local_consts = Dict{Symbol,Any}()
        local_types = Dict{Symbol,TyLike}()
        _ = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
            result, local_consts, local_types, relation, Random.Xoshiro(1111))
        @test any(value -> value === replacement, values(local_consts))
        @test state.nodes_left == 16 - CG.derivation_node_count(result)
    end
end

@testitem "CanonicalGeneration mixed source and ordinary head menus keep both schemas" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyArray, TyNat, TyList,
        FloatTy, get_type_ctx, infer_term, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree
    CG = LambdaRegression.CanonicalGeneration
    globals = CG.test_catalog_consts()
    providers = (CG.default_head_providers()..., CG.SafeListAccessProvider(),
        CG.TypedValueFactoryProvider(LambdaRegression.ValueFactories.FloatScalarFactory()))
    relation = CG.CoreRelation(global_consts=globals, global_consts_ty=get_type_ctx(globals),
        providers=providers, guarded_expansion_depth=1)
    for source_type in (TyList(FloatTy), TyArray(TyNat(4), FloatTy))
        scope = Ty[FloatTy, TyFun(FloatTy, FloatTy), source_type]
        query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
            guarded_left=1, provider_fuel=3, node_budget=96,
            requirements=(CG.UsesScopeBinder(0), CG.CallsScopeBinder(1)))
        borrowed = CG._safe_list_access_generation_options(relation, FloatTy, scope)
        @test !isempty(borrowed)
        source_count = length(borrowed)
        for seed in (31, 47) # first construction and source-menu cache hit
            options = CG._certified_mixed_candidate_options(relation, query, scope)
            @test options isa Vector{NamedTuple}
            @test count(option -> option.kind == :safe_list_access, options) == source_count
            @test any(option -> option.kind == :productive_head &&
                option.plan.rule.head.name == :FADD, options)
            @test length(borrowed) == source_count
            state = CG.DerivationSamplingState(Random.Xoshiro(seed), 96)
            result = CG._sample_certified_required_candidate(state, relation, query, true)
            @test result === nothing || result isa CG.CoreDerivation
            if result === nothing
                @test state.nodes_left == 96
            else
                @test CG._generation_requirements_satisfied(result, query)
                @test state.nodes_left == 96 - CG.derivation_node_count(result)
                locals = Dict{Symbol,Any}(); local_types = Dict{Symbol,TyLike}()
                node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
                    result, locals, local_types, relation, Random.Xoshiro(seed))
                closed, expected = CG._close_scoped_verification(AstTree(node), FloatTy, scope)
                inferred = infer_term(closed, merge(relation.global_consts_ty, local_types))
                @test inferred isa Ty && iscompatible(inferred, expected)
            end
        end
    end
end

@testitem "CanonicalGeneration ordinary child packets do not require a safe child" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyLike, TyFun, TyArray, TyNat,
        FloatTy, get_type_ctx, infer_term, iscompatible
    using LambdaRegression.ASTs.NodeASTs: AstTree
    CG = LambdaRegression.CanonicalGeneration
    globals = CG.test_catalog_consts()
    relation = CG.CoreRelation(global_consts=globals, global_consts_ty=get_type_ctx(globals),
        providers=(CG.default_head_providers()..., CG.SafeListAccessProvider(),
            CG.TypedValueFactoryProvider(LambdaRegression.ValueFactories.FloatScalarFactory())),
        guarded_expansion_depth=1)
    scope = Ty[FloatTy, TyFun(FloatTy, FloatTy), TyArray(TyNat(4), FloatTy)]
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=3,
        guarded_left=1, provider_fuel=3, node_budget=96,
        requirements=(CG.UsesScopeBinder(0), CG.CallsScopeBinder(1)))
    rule = only(filter(rule -> rule.head.name == :FADD && rule.result_ty === FloatTy,
        CG._certified_mixed_head_rules(relation, query)))
    assigned = [Int[0], Int[1]]
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 100)
    result = CG._materialize_selected_mixed_head_packet(state, relation, rule,
        query, 2, true, assigned, 0)
    @test result isa CG.HeadApplicationDerivation
    if result isa CG.HeadApplicationDerivation
        @test result.head.name == :FADD
        @test CG.derivation_uses_scope_binder(result.args[1], 0)
        @test CG.derivation_calls_scope_binder(result.args[2], 1)
        @test CG._generation_requirements_satisfied(result, query)
        @test state.nodes_left == 100 - CG.derivation_node_count(result)
        locals = Dict{Symbol,Any}(); local_types = Dict{Symbol,TyLike}()
        node = CG.materialize_generation_derivation(CG.DefaultAstMaterializer(),
            result, locals, local_types, relation, Random.Xoshiro(2000))
        closed, expected = CG._close_scoped_verification(AstTree(node), FloatTy, scope)
        inferred = infer_term(closed, merge(relation.global_consts_ty, local_types))
        @test inferred isa Ty && iscompatible(inferred, expected)
    end
    # At the original child depth the dedicated safe branch has no depth left
    # for its called-binder body. Do not relax that bound to make it succeed.
    required_safe = CG.DerivationSamplingState(Random.Xoshiro(2000), 100)
    @test CG._materialize_selected_mixed_head_packet(required_safe, relation,
        rule, query, 2, true, assigned, 2) === nothing
    @test required_safe.nodes_left == 100
    forbidden = CG._derive_generation_query(query;
        requirements=(CG.ForbidsScopeBinder(1),))
    rejected = CG.DerivationSamplingState(Random.Xoshiro(2000), 100)
    @test CG._materialize_selected_mixed_head_packet(rejected, relation,
        rule, forbidden, 2, true, assigned, 0) === nothing
    @test rejected.nodes_left == 100
end

@testitem "CanonicalGeneration scoped plans retain proxy prefixes" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy, BoolTy
    CG = LambdaRegression.CanonicalGeneration

    scope = Ty[TyFun(FloatTy, BoolTy), TyFun(FloatTy, FloatTy), FloatTy,
        TyFun(BoolTy, BoolTy)]
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=0, variable=1, constant=0,
            lambda=1, scope_application=1))
    query = CG.GenerationQuery(BoolTy; scope_tys=scope, depth_left=3,
        guarded_left=0, provider_fuel=3, node_budget=64,
        requirements=(CG.CallsScopeBinder(1),))
    prefix_count = 1
    prefix_offset = 2
    # This invokes the proxy prefix_required branch for rule 0: its own head
    # does not satisfy the prefix, while the Call binder can be placed in the
    # same Float child as the prefixed binder.
    old_plans = CG._fast_required_scope_application_plans(relation, BoolTy,
        scope, 3, prefix_count, prefix_offset, query)
    @test any(plan -> plan.rule.index == 0 && plan.prefix_choices == [1] &&
        any(choice -> choice.index == 1 && choice.arguments == [1],
            plan.exact_choices), old_plans)
    plan = only(filter(plan -> plan.rule.index == 0 &&
        plan.prefix_choices == [1] &&
        any(choice -> choice.index == 1 && choice.arguments == [1],
            plan.exact_choices), old_plans))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 64)
    result = CG._materialize_fast_required_scope_application(
        state, relation, plan, scope, 2, 0, 1, 2, false, query)
    @test result isa CG.ScopeApplicationDerivation
    if result isa CG.ScopeApplicationDerivation
        @test result.index == 0
        @test CG.derivation_calls_scope_binder(result, 1)
        @test CG.derivation_uses_scope_binder(result, 2)
        @test CG._generation_requirements_satisfied(result, query)
        @test state.nodes_left == 64 - CG.derivation_node_count(result)
    end
end

@testitem "CanonicalGeneration required constraints preserve binder identity and refund failures" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    weights = CG.GenerationWeights(default=0, variable=1, constant=0,
        lambda=1, scope_application=1)
    relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),), weights=weights)

    # Same-typed slots remain distinct, and Forbid applies to the concrete
    # binder rather than erasing all binders of that type.
    duplicate_scope = Ty[FloatTy, FloatTy]
    use_query = CG.GenerationQuery(FloatTy; scope_tys=duplicate_scope,
        depth_left=2, guarded_left=0, provider_fuel=2, node_budget=16,
        requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    use_options = CG._required_scope_certificate_options(relation, FloatTy,
        duplicate_scope, 2, (1,), use_query; allow_safe_list_access=false)
    use_result = CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(1), 16), relation,
        only(use_options), duplicate_scope, 2, 0, false, use_query)
    @test use_result isa CG.VariableDerivation
    if use_result isa CG.VariableDerivation
        @test use_result.index == 1
        @test CG.derivation_uses_scope_binder(use_result, 1)
        @test !CG.derivation_uses_scope_binder(use_result, 0)
    end

    # A function binder can be Used as a value without being Called. Requiring
    # Call excludes that terminal-variable route and reaches it under a lambda.
    fun = TyFun(FloatTy, FloatTy)
    function_scope = Ty[fun]
    value_query = CG.GenerationQuery(fun; scope_tys=function_scope,
        depth_left=3, guarded_left=0, provider_fuel=3, node_budget=32,
        requirements=(CG.UsesScopeBinder(0),))
    value_options = CG._required_scope_certificate_options(relation, fun,
        function_scope, 3, (0,), value_query; allow_safe_list_access=false)
    value_option = only(filter(option -> option.kind == :variable, value_options))
    value_result = CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(2000), 32), relation,
        value_option, function_scope, 3, 0, false, value_query)
    @test value_result isa CG.VariableDerivation
    @test value_result isa CG.VariableDerivation &&
        CG.derivation_uses_scope_binder(value_result, 0) &&
        !CG.derivation_calls_scope_binder(value_result, 0)

    call_query = CG.GenerationQuery(fun; scope_tys=function_scope,
        depth_left=3, guarded_left=0, provider_fuel=3, node_budget=32,
        requirements=(CG.CallsScopeBinder(0),))
    call_options = CG._required_scope_certificate_options(relation, fun,
        function_scope, 3, (0,), call_query; allow_safe_list_access=false)
    @test !any(option -> option.kind == :variable, call_options)
    lambda_option = only(filter(option -> option.kind == :lambda, call_options))
    call_result = CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(1111), 32), relation,
        lambda_option, function_scope, 3, 0, false, call_query)
    @test call_result isa CG.LambdaDerivation
    if call_result isa CG.LambdaDerivation
        @test CG.derivation_calls_scope_binder(call_result, 0)
        @test CG._generation_requirements_satisfied(call_result, call_query)
    end

    # One node is insufficient for application plus its variable argument;
    # failed selected-rule materialization refunds the entire local charge and
    # does not reverse the root's completion state.
    refund_scope = Ty[FloatTy, fun]
    refund_query = CG.GenerationQuery(FloatTy; scope_tys=refund_scope,
        depth_left=3, guarded_left=0, provider_fuel=3, node_budget=1,
        requirements=(CG.CallsScopeBinder(1),))
    refund_options = CG._required_scope_certificate_options(relation, FloatTy,
        refund_scope, 3, (1,), refund_query; allow_safe_list_access=false)
    @test any(option -> option.kind == :scope_application, refund_options)
    completion = CG.GenerationCompletionState(probability=0.5)
    completion.completing = true
    completion.boundaries = 4
    refund_state = CG.DerivationSamplingState(Random.Xoshiro(1), 1, completion)
    @test CG._sample_required_scope_legacy_derivation(refund_state, relation,
        FloatTy, refund_scope, 3, 0, false, refund_query) === nothing
    @test refund_state.nodes_left == 1
    @test refund_state.completion.completing
    @test refund_state.completion.boundaries == 4
end
