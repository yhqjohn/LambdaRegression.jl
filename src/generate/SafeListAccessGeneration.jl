struct ListAccessSource
    ty::TyList
    derivation::CoreDerivation
end

struct SafeListAccessDerivation <: CoreDerivation
    source::CoreDerivation
    source_ty::TyList
    body::CoreDerivation
    fallback::CoreDerivation
    result_ty::Ty
end

const _SAFE_LIST_ACCESS_WEIGHT = WeightBucket(:safe_list_access)

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
)
    _safe_list_access_enabled(relation) || return NamedTuple[]
    key = (:safe_list_access, _known_key(scope_tys, target_ty))
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return cached
    sources = _list_access_sources(relation, scope_tys)
    if isempty(sources)
        relation.sampling_options[key] = NamedTuple[]
        return NamedTuple[]
    end
    safe_weight = _bucket_weight(relation.weights, _SAFE_LIST_ACCESS_WEIGHT)
    options = [
        (
            kind=:safe_list_access,
            value=(source=source, target_ty=target_ty),
            weight=safe_weight * _derivation_weight(source.derivation, relation.weights),
        )
        for source in sources
    ]
    relation.sampling_options[key] = options
    return options
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
    source::ListAccessSource,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
)
    depth_left > 0 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 3 + _derivation_node_count(source.derivation)) || return nothing
    fallback = _sample_known_derivation(state, relation, target_ty, scope_tys, depth_left - 1)
    if fallback === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    body = _sample_proxy_body_derivation(
        state,
        relation,
        target_ty,
        _list_access_success_scope(source, scope_tys),
        depth_left - 1,
        guarded_left,
    )
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
)
    return _sample_proxy_using_derivation(state, relation, target_ty, scope_tys, depth_left, guarded_left, 2, 0)
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
)
    depth_left >= 0 || return nothing
    # In the common safe-list case, proxy variables should behave like fresh
    # binders. Build that path directly before falling back to full proof-space
    # sampling; otherwise guarded folds can spend seconds searching for a body
    # that merely needs to contain `h` or `t`.
    fast = _sample_fast_proxy_insertion_derivation(
        state,
        relation,
        target_ty,
        scope_tys,
        depth_left,
        prefix_count,
        prefix_offset,
    )
    fast === nothing || return fast

    options = copy(_sampling_options(
        relation,
        target_ty,
        scope_tys,
        depth_left;
        allow_head_application,
        lambda_kind=:lambda_generation,
    ))
    # Proxy-enforcing generation must remain constructive. Guarded expansions
    # are still available outside the safe-list body, but opening them here can
    # recursively search a very wide space before discovering a proxy use.
    return _sample_from_proxy_using_options(
        state,
        relation,
        options,
        scope_tys,
        depth_left,
        guarded_left,
        prefix_count,
        prefix_offset,
    )
end

function _sample_fast_proxy_insertion_derivation(
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

    if depth_left > 0 && target_ty isa TyFun
        saved_nodes = state.nodes_left
        _consume_sampling_nodes!(state, 1) || return nothing
        body = _sample_fast_proxy_insertion_derivation(
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

    depth_left > 0 || return nothing
    routes = _fast_proxy_head_routes(relation, target_ty, scope_tys, depth_left, prefix_count, prefix_offset)
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
            depth_left - 1,
            prefix_count,
            prefix_offset,
        )
        derivation !== nothing && return derivation
        state.nodes_left = saved_nodes
        total_weight -= route.weight
        deleteat!(pool, chosen)
    end
    return nothing
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
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        isempty(rule.argument_tys) && !_is_instance_factory_head(rule.head) && continue
        for i in eachindex(rule.argument_tys)
            _fast_proxy_lambda_or_direct_possible(
                rule.argument_tys[i],
                scope_tys,
                depth_left - 1,
                prefix_count,
                prefix_offset,
            ) || continue
            _fast_proxy_other_arguments_available(relation, rule, scope_tys, i) || continue
            push!(
                routes,
                (
                    kind=:fast_proxy_head,
                    value=(rule=rule, force_i=i),
                    weight=_head_option_weight(relation, rule),
                ),
            )
        end
    end
    relation.sampling_options[key] = routes
    return routes
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
    depth_left > 0 && target_ty isa TyFun || return nothing
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
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    prefix_count::Int,
    prefix_offset::Int,
)
    _has_direct_proxy_derivation(target_ty, scope_tys, prefix_count, prefix_offset) && return true
    depth_left > 0 && target_ty isa TyFun || return false
    return _fast_proxy_lambda_or_direct_possible(
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
    return relation.argument_planner(raw, arg_ty, arg_class, rule, relation)
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
)
    pool, total_weight = _positive_option_pool(options)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
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
        )
        derivation !== nothing && return derivation
        state.nodes_left = saved_nodes
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
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
)
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
        body = _sample_proxy_using_derivation(
            state,
            relation,
            target_ty.res,
            body_scope,
            depth_left - 1,
            guarded_left,
            prefix_count,
            prefix_offset + 1,
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
        )
    elseif option.kind == :scope_application
        derivation = _sample_scope_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left,
            false,
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
        )
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
                    false,
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
        )
    end

    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    if planner !== nothing || arg_class == :source_array || (guarded_head && _afoldl_grounded_argument(rule, i))
        raw_derivations =
            guarded_head && _afoldl_grounded_argument(rule, i) ?
            _grounded_derivations(relation, arg_ty, scope_tys) :
            _raw_argument_derivations(relation, arg_ty, arg_class, rule, scope_tys)
        derivs =
            guarded_head ?
            _guarded_argument_derivations(relation, raw_derivations, arg_ty, arg_class, rule) :
            relation.argument_planner(raw_derivations, arg_ty, arg_class, rule, relation)
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
    )
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
)
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
)
    parts = _afoldl_step_parts(rule)
    parts === nothing && return nothing
    depth_left >= 2 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 2) || return nothing
    body = _sample_proxy_using_derivation(
        state,
        relation,
        parts.body_ty,
        _afoldl_step_scope(rule, scope_tys),
        depth_left - 2,
        guarded_left,
        prefix_count,
        prefix_offset + 2,
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
    key = (Tuple(typekey.(scope_tys)), provider.source_array_depth_limit)
    cached = get(relation.list_access_sources, key, nothing)
    cached === nothing || return cached
    out = ListAccessSource[]
    signatures = Set{Any}()

    for (idx0, scope_ty) in enumerate(scope_tys)
        if scope_ty isa TyList
            _push_list_access_source!(out, signatures, ListAccessSource(scope_ty, VariableDerivation(idx0 - 1, scope_ty)))
        end
    end

    for list_ty in _list_context_source_types(relation)
        for derivation in _grounded_derivations(relation, list_ty, scope_tys)
            _push_list_access_source!(out, signatures, ListAccessSource(list_ty, derivation))
        end
    end

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

    for collector in provider.source_collectors
        for raw_source in collector(relation, scope_tys)
            source = _normalize_list_access_source(raw_source, relation)
            source === nothing && continue
            _push_list_access_source!(out, signatures, source)
        end
    end
    relation.list_access_sources[key] = out
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

_list_access_success_scope(source::ListAccessSource, scope_tys::Vector{Ty}) =
    Ty[source.ty, source.ty.elem, scope_tys...]

function _safe_tolist_head(array_ty::TyArray, list_ty::TyList)
    return HeadEntry(
        :TOLIST,
        :primitive,
        TyFun(array_ty, list_ty),
        CoreProductive(),
        _TOLIST_WEIGHT;
        metadata=(value=TypeSystem.TOLIST, source_array_depth_limit=0),
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
        metadata=(value=TypeSystem.UNCONS_CASE,),
    )
end

_uses_scope_prefix(derivation::CoreDerivation, prefix_count::Int) =
    _uses_scope_prefix(derivation, prefix_count, 0)

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
