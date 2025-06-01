"""Finite List state-helper proposals over an existing scoped list identity.

Set `GenerationWeights(list_state_helper=w)` to mix `C → List A → C`
helpers into a closed non-function `C` request. Initial values use ordinary
non-recursive generation. Bodies, guards and update expressions remain free;
acceptance requires one real same-source tail call whose state operand uses
both old state and item. The existing body/tail probabilities guide that operand.

No new ctx fields are needed. This bounded extension never seeds core evidence,
reopens itself in children, or owns a fresh FunctionDef ledger. Missing finite
initial evidence, source identity, depth, or at least three shared attempts
declines the proposal. Failure refunds nodes only; the ordinary menu remains.
"""

derivation_node_count(d::ComputedApplicationDerivation) =
    length(d.args) + derivation_node_count(d.head) + sum(derivation_node_count, d.args; init=0)
derivation_result_type(d::ComputedApplicationDerivation) = d.result_ty
derivation_signature(d::ComputedApplicationDerivation) =
    (:computed_application, typekey(d.result_ty), derivation_signature(d.head),
        _derivation_signature_tuple(d.args))
derivation_uses_scope_binder(d::ComputedApplicationDerivation, i::Integer, offset::Integer=0) =
    derivation_uses_scope_binder(d.head, i, offset) ||
    any(a -> derivation_uses_scope_binder(a, i, offset), d.args)
derivation_calls_scope_binder(d::ComputedApplicationDerivation, i::Integer, offset::Integer=0) =
    (d.head isa VariableDerivation && d.head.index == i + offset) ||
    derivation_calls_scope_binder(d.head, i, offset) ||
    any(a -> derivation_calls_scope_binder(a, i, offset), d.args)
rigid_safe_derivation(d::ComputedApplicationDerivation) =
    rigid_safe_derivation(d.head) && all(rigid_safe_derivation, d.args)
derivation_capabilities(d::ComputedApplicationDerivation) =
    _merge_derivation_capabilities(derivation_capabilities(d.head), d.args)
derivation_metadata(d::ComputedApplicationDerivation) =
    _merge_derivation_metadata((d.head, d.args...))
_copy_generation_witness(d::ComputedApplicationDerivation) =
    ComputedApplicationDerivation(_copy_generation_witness(d.head),
        CoreDerivation[_copy_generation_witness(a) for a in d.args])
_copy_generation_witness(d::FunctionDefDerivation) =
    FunctionDefDerivation(d.target_ty, _copy_generation_witness(d.body),
        _copy_generation_witness(d.base_witness), d.self_index, d.capabilities, d.remaining)
_guarded_depth(d::ComputedApplicationDerivation) =
    max(_guarded_depth(d.head), maximum(_guarded_depth, d.args; init=0))
_array_source_depth(::ComputedApplicationDerivation) = typemax(Int)
_derivation_weight(d::ComputedApplicationDerivation, weights::GenerationWeights) =
    (d.head isa FunctionDefDerivation ? _list_state_helper_weight(weights) :
        _derivation_weight(d.head, weights)) * prod(a -> _derivation_weight(a, weights), d.args)

function materialize_generation_derivation(materializer::DefaultAstMaterializer,
    d::ComputedApplicationDerivation, values, types, relation, rng)
    node = materialize_generation_derivation(materializer, d.head, values, types, relation, rng)
    for arg in d.args
        node = AppNode(node,
            materialize_generation_derivation(materializer, arg, values, types, relation, rng))
    end
    return node
end

_contains_safe_list_access(d::ComputedApplicationDerivation) =
    _contains_safe_list_access(d.head) || any(_contains_safe_list_access, d.args)
function _computed_scope_signature(signature)
    signature isa Tuple && length(signature) >= 4 && signature[1] === :computed_application || return signature
    head = signature[3]
    head isa Tuple && length(head) >= 3 && head[1] === :var || return signature
    return (:scope_application, head[2], head[3], signature[4])
end
_computed_scope_call(d::ComputedApplicationDerivation) =
    ScopeApplicationDerivation(d.head.index, d.head.ty, d.args)
function _has_unchanged_self_call_derivation(d::ComputedApplicationDerivation, self::Int, parameters::Vector{Int})
    d.head isa VariableDerivation && return _has_unchanged_self_call_derivation(_computed_scope_call(d), self, parameters)
    return _has_unchanged_self_call_derivation(d.head, self, parameters) ||
        any(a -> _has_unchanged_self_call_derivation(a, self, parameters), d.args)
end
function _has_unguarded_safe_list_recursive_derivation(d::ComputedApplicationDerivation, self::Int, depth::Int=0)
    d.head isa VariableDerivation && return _has_unguarded_safe_list_recursive_derivation(_computed_scope_call(d), self, depth)
    return _has_unguarded_safe_list_recursive_derivation(d.head, self, depth) ||
        any(a -> _has_unguarded_safe_list_recursive_derivation(a, self, depth), d.args)
end
function _strict_path_self_call(d::ComputedApplicationDerivation, self, arity, target, relation)
    d.head isa VariableDerivation && return _strict_path_self_call(_computed_scope_call(d), self, arity, target, relation)
    return _strict_path_self_call(d.head, self, arity, target, relation)
end

struct _ListStateHelperMask <: AbstractGenerationRequirement end
rebase_generation_requirement(mask::_ListStateHelperMask, ::Integer) = mask
generation_requirement_affects_core_search(::_ListStateHelperMask) = false

struct _ListStateHelperSelf <: AbstractGenerationRequirement
    self_index::Int
    helper_ty::Ty
end
rebase_generation_requirement(h::_ListStateHelperSelf, n::Integer) =
    _ListStateHelperSelf(h.self_index + _checked_nonnegative_int(:added_binders, n), h.helper_ty)
generation_requirement_affects_core_search(::_ListStateHelperSelf) = false
generation_requirement_cache_key(h::_ListStateHelperSelf) = (:list_state_helper_self, h.self_index, typekey(h.helper_ty))

struct _ListStateUpdateRequest <: AbstractGenerationRequirement
    self_index::Int
    state_index::Int
    item_index::Int
end
rebase_generation_requirement(h::_ListStateUpdateRequest, n::Integer) =
    _ListStateUpdateRequest(h.self_index + n, h.state_index + n, h.item_index + n)
generation_requirement_affects_core_search(::_ListStateUpdateRequest) = false
generation_requirement_cache_key(h::_ListStateUpdateRequest) =
    (:list_state_update, h.self_index, h.state_index, h.item_index)

_functiondef_helper_requirements(option) =
    hasproperty(option, :state_helper) && option.state_helper ?
        (_ListStateHelperSelf(0, option.target_ty),) : ()
_list_state_helper_weight(weights) = get(weights.overrides, :list_state_helper, 0.0)
_functiondef_or_helper_enabled(provider, weights) =
    _bucket_enabled(weights, provider.weight_bucket) || _list_state_helper_weight(weights) > 0
_has_list_state_helper_context(query) =
    any(r -> r isa _ListStateHelperSelf, query.requirements)

"""Recognize only the first scalar choice after the helper's two leading lambdas.

The existing FunctionDef prefix closes on the selected non-lambda constructor.
Rebasing alone cannot reopen it, including in later scalar children/callbacks.
"""
function _at_state_helper_body_entry(query)
    query === nothing && return false
    for h in query.requirements
        h isa _ListStateHelperSelf && 2 <= h.self_index < length(query.scope_tys) || continue
        typekey(query.target_ty) == typekey(h.helper_ty.res.res) || continue
        typekey(query.scope_tys[h.self_index + 1]) == typekey(h.helper_ty) || continue
        typekey(query.scope_tys[h.self_index]) == typekey(h.helper_ty.arg) || continue
        typekey(query.scope_tys[h.self_index - 1]) == typekey(h.helper_ty.res.arg) || continue
        any(r -> r isa FunctionDefSelfBinder && r.index == h.self_index &&
            r.leading_parameters == 2 && r.prefix_open, query.requirements) && return true
    end
    return false
end

"""Mix the first helper-body menu with its already available safe-operation family.

The body probability allocates `p*total` to existing safe choices proportionally
and leaves `(1-p)` of every old choice. Total menu mass and positive support are
preserved, with no extra draw, attempt, constructor, or cached-menu mutation.
The separate source preference then acts within that family's resulting mass.
"""
function _prefer_state_helper_body_options(options, relation, query)
    p = get(relation.weights.overrides, :list_recursive_body_probability, 0.0)
    if p > 0 && _at_state_helper_body_entry(query)
        total = sum(o.weight for o in options; init=0.0)
        safe = sum(o.weight for o in options if o.kind === :safe_list_access; init=0.0)
        if total > 0 && 0 < safe < total
            options = [merge(o, (; weight=o.weight * ((1-p) +
                (o.kind === :safe_list_access ? p * total / safe : 0.0)))) for o in options]
        end
    end
    return _prefer_state_helper_list_sources(options, relation, query)
end

"""Softly favour a helper's own List formal within an existing safe-source menu.

Only exact self/type/leading-parameter identity is accepted. The body-proposal
probability mixes old source weights with the formal-only allocation, retaining
all old positive sources and the same safe-source total. Cached menus are never
mutated; no randomness or provider attempt is spent here.
"""
function _prefer_state_helper_list_sources(options, relation, query)
    query === nothing && return options
    p = get(relation.weights.overrides, :list_recursive_body_probability, 0.0)
    p > 0 || return options
    formal = nothing
    forbidden = _forbidden_scope_binders(query)
    for h in query.requirements
        h isa _ListStateHelperSelf && h.self_index >= 2 || continue
        any(r -> r isa FunctionDefSelfBinder && r.index == h.self_index &&
            r.leading_parameters == 2, query.requirements) || continue
        h.self_index < length(query.scope_tys) || continue
        typekey(query.scope_tys[h.self_index + 1]) == typekey(h.helper_ty) || continue
        index = h.self_index - 2
        index in forbidden && continue
        typekey(query.scope_tys[index + 1]) == typekey(h.helper_ty.res.arg) || continue
        formal = index
        break
    end
    formal === nothing && return options
    safe = [o.kind === :safe_list_access for o in options]
    preferred = [safe[i] && o.value.source isa ListAccessSource &&
        o.value.source.derivation isa VariableDerivation &&
        o.value.source.derivation.index == formal for (i, o) in enumerate(options)]
    total = sum(o.weight for (i, o) in enumerate(options) if safe[i]; init=0.0)
    selected = sum(o.weight for (i, o) in enumerate(options) if preferred[i]; init=0.0)
    total > 0 && selected > 0 || return options
    return [safe[i] ? merge(o, (; weight=o.weight *
        ((1-p) + (preferred[i] ? p * total / selected : 0.0)))) : o
        for (i, o) in enumerate(options)]
end

function _state_helper_hint_admitted(query, hint)
    helpers = [r for r in query.requirements if r isa _ListStateHelperSelf && r.self_index == hint.self_index]
    return isempty(helpers) || hint.argument_position == 2
end

function _state_helper_body_requirements(query, hint)
    for helper in query.requirements
        helper isa _ListStateHelperSelf && helper.self_index == hint.self_index || continue
        hint.argument_position == 2 || continue
        any(r -> r isa FunctionDefSelfBinder && r.index == helper.self_index &&
            r.leading_parameters == 2, query.requirements) || continue
        return (_ListStateUpdateRequest(helper.self_index, helper.self_index - 1, hint.head_index),)
    end
    return ()
end

function _state_update_request(query, self)
    for r in query.requirements
        r isa _ListStateUpdateRequest && r.self_index == self && return r
    end
    return nothing
end

function _sample_joint_list_state_argument(state, relation, rule, scope,
    depth, guarded, allow_safe, query, update, hint)
    length(rule.argument_tys) == 2 || return nothing
    generation_function_view(rule.argument_tys[1]) === nothing || return nothing
    free = _without_scope_binder_requirements(query)
    tail = VariableDerivation(hint.tail_index, scope[hint.tail_index + 1])
    remaining = _without_satisfied_scope_binder_requirements(query, tail)
    positives = Tuple(r for r in remaining.requirements if
        r isa Union{UsesScopeBinder,CallsScopeBinder} && r.index != rule.index)
    argument_query = _derive_generation_query(free; target_ty=rule.argument_tys[1],
        depth_left=min(query.depth_left, depth), guarded_left=min(query.guarded_left, guarded),
        node_budget=min(query.node_budget, state.nodes_left),
        requirements=(positives..., UsesScopeBinder(update.state_index),
            UsesScopeBinder(update.item_index), ForbidsScopeBinder(rule.index), _ListStateHelperMask()))
    result = _sample_required_scope_certificate_derivation(state, relation,
        rule.argument_tys[1], scope, depth, guarded, allow_safe, argument_query)
    result === nothing && return nothing
    _generation_requirements_satisfied(result, argument_query) || return nothing
    any(i -> derivation_uses_scope_binder(result, i), _forbidden_scope_binders(argument_query)) && return nothing
    return result
end

function _list_state_helper_sources(query)
    forbidden = _forbidden_scope_binders(query)
    return [i - 1 for (i, ty) in enumerate(query.scope_tys) if
        ty isa TyList && _in_closed_fragment(ty) && !_contains_rigid_dummy(ty) &&
        !(i - 1 in forbidden)]
end

function _list_state_helper_applicable(provider, relation, query)
    _list_state_helper_weight(relation.weights) > 0 || return false
    any(r -> r isa _ListStateHelperMask, query.requirements) && return false
    _in_closed_fragment(query.target_ty) && !(query.target_ty isa TyFun) &&
        !_contains_rigid_dummy(query.target_ty) || return false
    _scope_all(_in_closed_fragment, query.scope_tys) || return false
    query.depth_left >= 5 && query.node_budget >= 8 || return false
    _functiondef_attempts_remaining(query, provider) >= 3 || return false
    any(r -> r isa FunctionDefAttemptBudget && r.route_token === provider.route_token,
        query.requirements) || return false
    return !isempty(_list_state_helper_sources(query))
end

function _add_list_state_helper_options!(provider, options, relation, query)
    _list_state_helper_applicable(provider, relation, query) || return options
    sources = _list_state_helper_sources(query)
    for i in sources
        push!(options, (; kind=:list_state_helper, target_ty=query.target_ty,
            helper_ty=TyFun(query.target_ty, TyFun(query.scope_tys[i + 1], query.target_ty)),
            source_index=i, weight=_list_state_helper_weight(relation.weights) / length(sources)))
    end
    return options
end

struct _StateHelperAdmissibility{A}
    inner::A
end
function (filter::_StateHelperAdmissibility)(candidate::FunctionDefCandidate, args...)
    _inner_functiondef_admissible(filter.inner, candidate, args...) || return false
    return _has_state_helper_update(candidate)
end
_functiondef_sampling_admissible(filter::_StateHelperAdmissibility, candidate::FunctionDefCandidate) =
    _functiondef_sampling_admissible(filter.inner, candidate) && _has_state_helper_update(candidate)

function _has_state_helper_update(candidate::FunctionDefCandidate)
    ty = candidate.target_ty
    ty isa TyFun && ty.res isa TyFun && ty.res.arg isa TyList || return false
    typekey(ty.arg) == typekey(ty.res.res) || return false
    body = candidate.body
    body isa LambdaDerivation && body.body isa LambdaDerivation || return false
    typekey(body.arg_ty) == typekey(ty.arg) &&
        typekey(body.body.arg_ty) == typekey(ty.res.arg) || return false
    return _find_state_helper_access(body.body.body, candidate.self_index + 2,
        candidate.self_index + 1, candidate.self_index, ty)
end
_find_state_helper_access(::CoreDerivation, self, old, list, ty) = false
function _find_state_helper_access(d::SafeListAccessDerivation, self, old, list, ty)
    if d.source isa VariableDerivation && d.source.index == list &&
       typekey(d.source_ty) == typekey(ty.res.arg) &&
       _find_state_update_call(d.body, self + 2, old + 2, 1, 0, ty)
        return true
    end
    return _find_state_helper_access(d.source, self, old, list, ty) ||
        _find_state_helper_access(d.body, self + 2, old + 2, list + 2, ty) ||
        _find_state_helper_access(d.fallback, self, old, list, ty)
end
_find_state_helper_access(d::LambdaDerivation, self, old, list, ty) =
    _find_state_helper_access(d.body, self + 1, old + 1, list + 1, ty)
_find_state_helper_access(d::HeadApplicationDerivation, self, old, list, ty) =
    any(a -> _find_state_helper_access(a, self, old, list, ty), d.args)
_find_state_helper_access(d::ScopeApplicationDerivation, self, old, list, ty) =
    any(a -> _find_state_helper_access(a, self, old, list, ty), d.args)
_find_state_helper_access(d::ComputedApplicationDerivation, self, old, list, ty) =
    _find_state_helper_access(d.head, self, old, list, ty) ||
    any(a -> _find_state_helper_access(a, self, old, list, ty), d.args)
# A nested definition's update is not evidence that this helper updates itself.
_find_state_update_call(::CoreDerivation, self, old, item, tail, ty) = false
function _find_state_update_call(d::ScopeApplicationDerivation, self, old, item, tail, ty)
    if d.index == self && typekey(d.head_ty) == typekey(ty) && length(d.args) == 2
        state, source = d.args
        source isa VariableDerivation && source.index == tail &&
            derivation_uses_scope_binder(state, old) &&
            derivation_uses_scope_binder(state, item) && return true
    end
    return any(a -> _find_state_update_call(a, self, old, item, tail, ty), d.args)
end
_find_state_update_call(d::HeadApplicationDerivation, self, old, item, tail, ty) =
    any(a -> _find_state_update_call(a, self, old, item, tail, ty), d.args)
_find_state_update_call(d::ComputedApplicationDerivation, self, old, item, tail, ty) =
    _find_state_update_call(d.head, self, old, item, tail, ty) ||
    any(a -> _find_state_update_call(a, self, old, item, tail, ty), d.args)
_find_state_update_call(d::LambdaDerivation, self, old, item, tail, ty) =
    _find_state_update_call(d.body, self + 1, old + 1, item + 1, tail + 1, ty)
_find_state_update_call(d::SafeListAccessDerivation, self, old, item, tail, ty) =
    _find_state_update_call(d.source, self, old, item, tail, ty) ||
    _find_state_update_call(d.body, self + 2, old + 2, item + 2, tail + 2, ty) ||
    _find_state_update_call(d.fallback, self, old, item, tail, ty)

function _sample_list_state_helper(provider, state, relation, option, query)
    _list_state_helper_applicable(provider, relation, query) || return nothing
    state.completion.probability == 0 && !state.completion.completing &&
        relation.argument_planner isa DefaultArgumentPlanner || throw(ArgumentError(
            "list_state_helper requires bounded default sampling/planning"))
    i = option.source_index
    i in _list_state_helper_sources(query) || return nothing
    saved = state.nodes_left
    result = _construct_list_state_helper(provider, state, relation, option, query)
    result === nothing && (state.nodes_left = saved)
    return result
end

function _construct_list_state_helper(provider, state, relation, option, query)
    _consume_sampling_nodes!(state, 3) || return nothing # two App nodes and exact source
    source = VariableDerivation(option.source_index, query.scope_tys[option.source_index + 1])
    free = _without_scope_binder_requirements(_close_functiondef_prefix(query))
    masked = _derive_generation_query(free; depth_left=query.depth_left - 2,
        node_budget=min(query.node_budget, state.nodes_left), requirements=(_ListStateHelperMask(),))
    self_forbids = Tuple(ForbidsScopeBinder(r.index) for r in query.requirements if r isa FunctionDefSelfBinder)
    initial_query = _derive_generation_query(masked;
        capability_mask=(:function_def, :fixed_point), requirements=self_forbids)
    initial_relation, initial_query = _relation_for_query(relation, initial_query)
    witness = _borrow_known_generation_witness(initial_relation, initial_query)
    witness === nothing && return nothing
    any(i -> derivation_uses_scope_binder(witness, i),
        _forbidden_scope_binders(initial_query)) && return nothing
    initial_saved = state.nodes_left
    initial = _sample_generation_derivation(state, initial_relation, initial_query)
    if initial !== nothing && any(i -> derivation_uses_scope_binder(initial, i),
        _forbidden_scope_binders(initial_query))
        initial = nothing
    end
    if initial === nothing
        state.nodes_left = initial_saved
        _consume_sampling_nodes!(state, derivation_node_count(witness)) || return nothing
        initial = _copy_generation_witness(witness)
    end
    remaining = _without_satisfied_scope_binder_requirements(query, source)
    _functiondef_attempts_remaining(query, provider) >= 3 || return nothing
    positives = Tuple(r for r in remaining.requirements if r isa Union{UsesScopeBinder,CallsScopeBinder})
    helper_query = _derive_generation_query(masked; target_ty=option.helper_ty,
        node_budget=min(masked.node_budget, state.nodes_left), requirements=positives)
    wrapper = FunctionDefProvider(weight_bucket=provider.weight_bucket,
        base_capability_mask=provider.base_capability_mask, base_witness=provider.base_witness,
        admissibility=_StateHelperAdmissibility(provider.admissibility),
        max_nesting=provider.max_nesting, body_attempts=provider.body_attempts,
        _route_token=provider.route_token)
    helper_option = (; target_ty=option.helper_ty, state_helper=true,
        remaining=_functiondef_remaining(helper_query, wrapper),
        base_depth_left=helper_query.depth_left, base_node_budget=helper_query.node_budget)
    helper = sample_generation_extension(wrapper, state, relation, helper_option, helper_query)
    helper === nothing && return nothing
    result = ComputedApplicationDerivation(helper, CoreDerivation[initial, source])
    _generation_requirements_satisfied(result, query) || return nothing
    any(i -> derivation_uses_scope_binder(result, i), _forbidden_scope_binders(query)) && return nothing
    return result
end
