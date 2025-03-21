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
    _may_have_list_access_source(relation, scope_tys) || return NamedTuple[]
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
    source::ListAccessSource,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    depth_left > 0 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 3 + _derivation_node_count(source.derivation)) || return nothing
    fallback = _sample_known_derivation(state, relation, target_ty, scope_tys, depth_left - 1)
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
    body = _sample_proxy_body_derivation(
        state,
        relation,
        target_ty,
        success_scope,
        depth_left - 1,
        guarded_left,
        body_query,
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
    cached = cacheable ? get(relation.sampling_options, key, nothing) : nothing
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
        body_scope = Ty[target_ty.arg; scope_tys]
        if all(required) do index
            body_index = index + 1
            index in call_indices ?
                _required_scope_call_reachable(
                    relation,
                    target_ty.res,
                    body_scope,
                    depth_left - 1,
                    body_index,
                ) :
                _required_scope_binder_reachable(
                    relation,
                    target_ty.res,
                    body_scope,
                    depth_left - 1,
                    body_index,
                )
        end
            push!(options, (
                kind=:lambda,
                value=target_ty,
                weight=_bucket_weight(relation.weights, _LAMBDA_WEIGHT),
            ))
        end
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
       guarded_weight > 0
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
            obligation_plan = _required_productive_obligation_plan(
                relation,
                plan,
                scope_tys,
                depth_left - 1,
                generation_query,
            )
            obligation_plan === nothing && continue
            planned = (; plan..., obligation_plan)
            push!(options, (kind=:productive_head, plan=planned, weight=plan.weight))
        end
    end

    cacheable && (relation.sampling_options[key] = options)
    return options
end

function _required_productive_obligation_plan(
    relation::CoreRelation,
    plan::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    exact_indices = _required_scope_binder_indices(generation_query, length(scope_tys))
    call_indices = _required_scope_call_indices(generation_query, length(scope_tys))
    isempty(exact_indices) && return nothing
    exact_choices = NamedTuple[]
    for index in exact_indices
        arguments = Int[]
        for argument_index in eachindex(plan.rule.argument_tys)
            reachable = index in call_indices ?
                _required_scope_call_reachable(
                    relation,
                    plan.rule.argument_tys[argument_index],
                    scope_tys,
                    depth_left,
                    index,
                ) :
                _required_scope_binder_reachable(
                    relation,
                    plan.rule.argument_tys[argument_index],
                    scope_tys,
                    depth_left,
                    index,
                )
            reachable || continue
            push!(arguments, argument_index)
        end
        isempty(arguments) && return nothing
        push!(exact_choices, (index=index, arguments=arguments))
    end
    base_available = Tuple(
        _fast_required_head_base_witness(
            relation,
            plan.rule,
            argument_index,
            scope_tys,
            depth_left,
            generation_query,
        ) !== nothing
        for argument_index in eachindex(plan.rule.argument_tys)
    )
    required_arguments = Tuple(
        argument_index
        for argument_index in eachindex(plan.rule.argument_tys)
        if !base_available[argument_index]
    )
    _required_head_assignment_exists(exact_choices, required_arguments) || return nothing
    return (; exact_choices=Tuple(exact_choices), base_available, required_arguments)
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
    pool, total_weight = _positive_option_pool(options)
    saved_nodes = state.nodes_left
    while !isempty(pool) && total_weight > 0
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
            return derivation
        end
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
    state.nodes_left = saved_nodes
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
    cached = get(relation.sampling_options, key, nothing)
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
    relation.sampling_options[key] = plans
    return plans
end

function _productive_rules_for_target(
    relation::CoreRelation,
    catalog::CoreCatalog,
    catalog_key,
    target_ty::Ty,
)
    key = (:productive_rules_for_target, catalog_key, typekey(target_ty))
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return cached
    rules = NamedTuple[]
    for rule in catalog.rules
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        push!(rules, (rule=rule,))
    end
    relation.sampling_options[key] = rules
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
    obligation_plan = haskey(plan, :obligation_plan) ?
        plan.obligation_plan :
        _required_productive_obligation_plan(
            relation,
            plan,
            scope_tys,
            depth_left,
            generation_query,
        )
    obligation_plan === nothing && return nothing
    exact_choices = obligation_plan.exact_choices

    _consume_sampling_nodes!(state, 1) || return nothing
    rule = plan.rule
    assigned_exact = [Int[] for _ in rule.argument_tys]
    required_arguments = obligation_plan.required_arguments
    assignment = _required_head_argument_assignment(
        exact_choices,
        required_arguments,
        state.rng,
    )
    assignment === nothing && return nothing
    for (obligation_index, obligation) in enumerate(exact_choices)
        argument_index = assignment[obligation_index]
        argument_index == 0 && (argument_index = rand(state.rng, obligation.arguments))
        push!(assigned_exact[argument_index], obligation.index)
    end
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
    grounded = _fast_proxy_other_argument_derivations(relation, rule, i, scope_tys)
    isempty(grounded) || return first(grounded)
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

function _required_head_assignment_exists(exact_choices, required_arguments)
    return _required_head_argument_assignment(
        exact_choices,
        required_arguments,
        nothing,
    ) !== nothing
end

function _required_head_argument_assignment(exact_choices, required_arguments, rng)
    assignment = fill(0, length(exact_choices))
    arguments = collect(required_arguments)
    rng === nothing || Random.shuffle!(rng, arguments)
    for argument_index in arguments
        seen = falses(length(exact_choices))
        _match_required_head_argument!(
            argument_index,
            exact_choices,
            assignment,
            seen,
            rng,
        ) || return nothing
    end
    return assignment
end

function _match_required_head_argument!(
    argument_index::Int,
    exact_choices,
    assignment::Vector{Int},
    seen::BitVector,
    rng,
)
    obligations = Int[
        obligation_index
        for obligation_index in eachindex(exact_choices)
        if argument_index in exact_choices[obligation_index].arguments
    ]
    rng === nothing || Random.shuffle!(rng, obligations)
    for obligation_index in obligations
        seen[obligation_index] && continue
        seen[obligation_index] = true
        previous = assignment[obligation_index]
        if previous == 0 || _match_required_head_argument!(
                previous,
                exact_choices,
                assignment,
                seen,
                rng,
            )
            assignment[obligation_index] = argument_index
            return true
        end
    end
    return false
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
    while !isempty(pool) && total_weight > 0
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
        total_weight -= plan.weight
        deleteat!(pool, chosen)
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
    cached = cacheable ? get(relation.sampling_options, key, nothing) : nothing
    cached === nothing || return cached

    plans = NamedTuple[]
    for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
        _known_type_admitted(relation, rule.head_ty) || continue
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
            (forced=forced, exact_choices=(), prefix_choices=Int[]),
        )

        head_satisfies_prefix = rule.index in prefix_indices
        head_satisfies_exact = rule.index in exact_indices
        if head_satisfies_prefix || head_satisfies_exact || !isempty(call_indices)
            remaining_exact = Int[index for index in exact_indices if index != rule.index]
            # Cache only deterministic placement choices. Sampling assigns each
            # remaining binder obligation to one compatible child query later.
            exact_choices = NamedTuple[]
            partition_possible = true
            for index in remaining_exact
                arguments = Int[]
                for argument_index in eachindex(rule.argument_tys)
                    reachable = index in call_indices ?
                        _required_scope_call_reachable(
                            relation,
                            rule.argument_tys[argument_index],
                            scope_tys,
                            depth_left - 1,
                            index,
                        ) :
                        _required_scope_binder_reachable(
                            relation,
                            rule.argument_tys[argument_index],
                            scope_tys,
                            depth_left - 1,
                            index,
                        )
                    reachable || continue
                    push!(arguments, argument_index)
                end
                if isempty(arguments)
                    partition_possible = false
                    break
                end
                push!(exact_choices, (index=index, arguments=arguments))
            end

            prefix_required = !head_satisfies_prefix &&
                !any(index -> index in prefix_indices, remaining_exact)
            prefix_choices = Int[]
            if partition_possible && prefix_required
                for argument_index in eachindex(rule.argument_tys)
                    _fast_proxy_lambda_or_direct_possible(
                        relation,
                        rule.argument_tys[argument_index],
                        scope_tys,
                        depth_left - 1,
                        prefix_count,
                        prefix_offset,
                    ) || continue
                    push!(prefix_choices, argument_index)
                end
                isempty(prefix_choices) && (partition_possible = false)
            end
            if forced === nothing &&
               partition_possible &&
               (!isempty(exact_choices) || prefix_required)
                push!(
                    rule_plans,
                    (
                        forced=fill(-1, length(rule.argument_tys)),
                        exact_choices=Tuple(exact_choices),
                        prefix_choices,
                    ),
                )
            end
        end

        isempty(rule_plans) && continue
        plan_weight = _scope_application_option_weight(relation, rule) / length(rule_plans)
        for plan in rule_plans
            completed = _complete_required_scope_application_plan(
                relation,
                rule,
                plan,
                scope_tys,
                depth_left - 1,
                generation_query,
            )
            completed === nothing && continue
            push!(plans, (rule=rule, completed..., weight=plan_weight))
        end
    end
    cacheable && (relation.sampling_options[key] = plans)
    return plans
end

function _complete_required_scope_application_plan(
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    plan::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    generation_query::Union{Nothing,GenerationQuery},
)
    base_available = Tuple(
        plan.forced[i] >= 0 ||
        _required_scope_application_base_witness(
            relation,
            rule,
            i,
            scope_tys,
            depth_left,
            generation_query,
        ) !== nothing
        for i in eachindex(rule.argument_tys)
    )
    required_arguments = Tuple(
        i for i in eachindex(rule.argument_tys) if !base_available[i]
    )
    choices = NamedTuple[plan.exact_choices...]
    isempty(plan.prefix_choices) || push!(
        choices,
        (index=-1, arguments=plan.prefix_choices),
    )
    _required_head_assignment_exists(choices, required_arguments) || return nothing
    return (; plan..., base_available, required_arguments)
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
        sources = _bounded_source_array_derivations(
            relation,
            rule.argument_tys[i],
            scope_tys,
            0,
        )
        return isempty(sources) ? nothing : first(sources)
    end
    grounded = _fast_proxy_other_argument_derivations(relation, rule, i, scope_tys)
    isempty(grounded) || return first(grounded)
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
    _consume_sampling_nodes!(state, length(rule.argument_tys)) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    assigned_exact = [Int[] for _ in rule.argument_tys]
    choices = NamedTuple[plan.exact_choices...]
    prefix_choice_index = 0
    if !isempty(plan.prefix_choices)
        push!(choices, (index=-1, arguments=plan.prefix_choices))
        prefix_choice_index = length(choices)
    end
    assignment = _required_head_argument_assignment(
        choices,
        plan.required_arguments,
        state.rng,
    )
    assignment === nothing && return nothing
    for (obligation_index, obligation) in enumerate(plan.exact_choices)
        argument_index = assignment[obligation_index]
        argument_index == 0 && (argument_index = rand(state.rng, obligation.arguments))
        push!(assigned_exact[argument_index], obligation.index)
    end
    prefix_argument = if prefix_choice_index == 0
        0
    else
        selected = assignment[prefix_choice_index]
        selected == 0 ? rand(state.rng, plan.prefix_choices) : selected
    end
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

function _scope_binder_requirement_subset(
    query::Union{Nothing,GenerationQuery},
    assigned_indices::Vector{Int},
)
    query === nothing && return nothing
    requirements = Tuple(
        requirement for requirement in query.requirements
        if !(requirement isa Union{UsesScopeBinder,CallsScopeBinder}) ||
           requirement.index in assigned_indices
    )
    return GenerationQuery(
        query.target_ty;
        scope_tys=query.scope_tys,
        depth_left=query.depth_left,
        guarded_left=query.guarded_left,
        provider_fuel=query.provider_fuel,
        node_budget=query.node_budget,
        capability_mask=query.capability_mask,
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
    cached = get(relation.sampling_options, key, nothing)
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
    relation.sampling_options[key] = plans
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

function _without_scope_binder_requirements(
    query::Union{Nothing,GenerationQuery},
)
    query === nothing && return nothing
    requirements = Tuple(
        requirement for requirement in query.requirements
        if !(requirement isa Union{UsesScopeBinder,CallsScopeBinder})
    )
    return GenerationQuery(
        query.target_ty;
        scope_tys=query.scope_tys,
        depth_left=query.depth_left,
        guarded_left=query.guarded_left,
        provider_fuel=query.provider_fuel,
        node_budget=query.node_budget,
        capability_mask=query.capability_mask,
        requirements,
    )
end

function _without_satisfied_scope_binder_requirements(
    query::Union{Nothing,GenerationQuery},
    derivation::CoreDerivation,
)
    query === nothing && return nothing
    requirements = Tuple(
        requirement for requirement in query.requirements
        if (!(requirement isa UsesScopeBinder) ||
            !derivation_uses_scope_binder(derivation, requirement.index)) &&
           (!(requirement isa CallsScopeBinder) ||
            !derivation_calls_scope_binder(derivation, requirement.index))
    )
    length(requirements) == length(query.requirements) && return query
    return GenerationQuery(
        query.target_ty;
        scope_tys=query.scope_tys,
        depth_left=query.depth_left,
        guarded_left=query.guarded_left,
        provider_fuel=query.provider_fuel,
        node_budget=query.node_budget,
        capability_mask=query.capability_mask,
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
        generation_query,
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
