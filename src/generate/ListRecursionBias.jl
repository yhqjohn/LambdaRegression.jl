"""Bounded sampling hint for a safe tail of one original recursive parameter.

The identity comes only from `FunctionDefSelfBinder` and a concrete variable
source, never from types or names alone. This is sampling policy, not a new
core requirement, recursion certificate, or termination claim. Both binding
identities move together when generation enters another lexical scope.
"""
struct _ListRecursionHint <: AbstractGenerationRequirement
    self_index::Int
    argument_position::Int
    tail_index::Int
    head_index::Int
end

@testitem "CanonicalGeneration list recursion bias preserves source identities and budgets" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyMu, FloatTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    fun = TyFun(list, FloatTy)
    p = 1 - eps(Float64)
    weights = CG.GenerationWeights(default=0, variable=1, scope_application=1,
        constant=1, proxy_constructive=1, proxy_general=0, proxy_head=0,
        proxy_scope_application=0, proxy_split=1, constraint_evidence=0,
        list_recursive_body_probability=p, list_recursive_tail_probability=p)
    relation = CG.CoreRelation(weights=weights,
        global_consts=Dict{Symbol,Any}(:ADD => (x -> y -> x + y)),
        global_consts_ty=Dict{Symbol,Any}(:ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    # Two equal-typed lists are distinct: only slot 0 is self's original parameter.
    scope = Ty[list, fun, list]
    parent = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=5,
        node_budget=16, requirements=(CG.FunctionDefSelfBinder(1, 1, false),))
    success = Ty[list, FloatTy, scope...]
    bodyq = CG._derive_generation_query(parent; added_scope_tys=(list, FloatTy), depth_left=4)
    source = CG.ListAccessSource(list, CG.VariableDerivation(0, list))
    q = CG._with_list_recursion_hints(relation, source, parent, bodyq)
    hint = only(r for r in q.requirements if r isa CG._ListRecursionHint)
    @test (hint.self_index, hint.argument_position, hint.tail_index, hint.head_index) == (3, 1, 0, 1)
    unrelated = CG.ListAccessSource(list, CG.VariableDerivation(2, list))
    @test CG._with_list_recursion_hints(relation, unrelated, parent, bodyq) === bodyq
    @test CG._with_list_recursion_hints(relation, source, nothing, bodyq) === bodyq
    shifted = CG._derive_generation_query(q; added_scope_tys=(FloatTy,))
    moved = only(r for r in shifted.requirements if r isa CG._ListRecursionHint)
    @test (moved.self_index, moved.argument_position, moved.tail_index, moved.head_index) == (4, 1, 1, 2)
    @test CG.generation_requirement_cache_key(moved) != CG.generation_requirement_cache_key(hint)
    @test !CG.generation_requirement_affects_core_search(hint)
    @test !CG._has_list_recursion_hint(q)
    @test CG._has_list_recursion_hint(CG._derive_generation_query(q;
        requirements=(CG.CallsScopeBinder(3),)))
    @test !CG._has_list_recursion_hint(CG._derive_generation_query(q;
        requirements=(CG.UsesScopeBinder(1),)))

    state = CG.DerivationSamplingState(Random.Xoshiro(8000), 16)
    body = CG._sample_list_recursive_body(state, relation, FloatTy, success, 4, 0, q)
    @test body isa CG.HeadApplicationDerivation
    if body isa CG.HeadApplicationDerivation
        @test CG.derivation_uses_scope_binder(body, 1)
        @test CG.derivation_calls_scope_binder(body, 3)
        calls = filter(d -> d isa CG.ScopeApplicationDerivation && d.index == 3, body.args)
        @test length(calls) == 1
        if length(calls) == 1
            @test only(only(calls).args) == CG.VariableDerivation(0, list)
        end
        @test 16 - state.nodes_left == CG.derivation_node_count(body)
    end
    short = CG.DerivationSamplingState(Random.Xoshiro(8000), 1)
    @test CG._sample_list_recursive_body(short, relation, FloatTy, success, 4, 0, q) === nothing
    @test short.nodes_left == 1
    blocked = CG._derive_generation_query(q; requirements=(CG.ForbidsScopeBinder(1),))
    @test CG._sample_list_recursive_body(CG.DerivationSamplingState(Random.Xoshiro(8000), 16),
        relation, FloatTy, success, 4, 0, blocked) === nothing

    # Existing stateful signatures use the right argument position, not the
    # nearest same-typed list. No helper definition is injected by this test.
    statefun = TyFun(FloatTy, TyFun(list, FloatTy))
    statescope = Ty[list, FloatTy, list, FloatTy, statefun]
    stateq = CG.GenerationQuery(FloatTy; scope_tys=statescope, depth_left=3,
        node_budget=12, requirements=(CG.FunctionDefSelfBinder(4, 2, false),
            CG.CallsScopeBinder(4), CG._ListRecursionHint(4, 2, 0, 1)))
    rule = only(filter(r -> r.index == 4,
        CG._scope_application_rules(FloatTy, statescope, 3)))
    callstate = CG.DerivationSamplingState(Random.Xoshiro(555), 12)
    call = CG._sample_preferred_list_recursive_call!(callstate, relation, rule,
        statescope, 2, 0, false, stateq)
    @test call isa CG.ScopeApplicationDerivation
    if call isa CG.ScopeApplicationDerivation
        @test call.args[2] == CG.VariableDerivation(0, list)
        @test 12 - callstate.nodes_left == CG.derivation_node_count(call)
    end
    exhausted = CG.DerivationSamplingState(Random.Xoshiro(555), 2)
    @test CG._sample_preferred_list_recursive_call!(exhausted, relation, rule,
        statescope, 2, 0, false, stateq) === nothing
    @test exhausted.nodes_left == 2
    # A separate state-use obligation must never vanish in a tail shortcut.
    requiredstate = CG._derive_generation_query(stateq; requirements=(CG.UsesScopeBinder(3),))
    constrainedrng = Random.Xoshiro(2000)
    before = copy(constrainedrng)
    @test CG._sample_preferred_list_recursive_call!(CG.DerivationSamplingState(constrainedrng, 12),
        relation, rule, statescope, 2, 0, false, requiredstate) === nothing
    @test rand(constrainedrng) == rand(before)

    # An opaque recursive type does not become a list operand merely because
    # it exposes a function view elsewhere in the generator.
    v = newtyvar(0)
    mu = TyMu(v, TyFun(FloatTy, v))
    murule = CG.ScopeApplicationRule(1, TyFun(mu, FloatTy), FloatTy, Ty[mu], [:value])
    muscope = Ty[mu, murule.head_ty]
    muq = CG.GenerationQuery(FloatTy; scope_tys=muscope, depth_left=2,
        requirements=(CG._ListRecursionHint(1, 1, 0, 0),))
    rng = Random.Xoshiro(1)
    control = copy(rng)
    @test CG._sample_preferred_list_recursive_call!(CG.DerivationSamplingState(rng, 8),
        relation, murule, muscope, 1, 0, false, muq) === nothing
    @test rand(rng) == rand(control)
end

@testitem "CanonicalGeneration list recursion bias has neutral RNG and explicit mode bounds" begin
    using Test, Random
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    for key in (:list_recursive_body_probability, :list_recursive_tail_probability,
                :functiondef_self_call_probability, :safe_list_short_fallback_probability)
        for value in (-0.1, 1.0, Inf, NaN)
            @test_throws ArgumentError CG.GenerationWeights(; (key => value,)...)
        end
        weights = CG.GenerationWeights(; (key => 0.5,)...)
        @test CG._uses_lazy_sampling(CG.GenerationConfig(; weights))
        @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(; weights,
            sampler=CG.WeightedDerivationSampler(completion_probability=0.1)))
        @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(; weights,
            argument_planner=(args...) -> nothing))
        @test_throws ArgumentError CG._uses_lazy_sampling(CG.GenerationConfig(; weights,
            sampler=(args...) -> nothing))
        @test_throws ArgumentError CG._validate_functiondef_sampling_config(CG.GenerationConfig(; weights,
            sampler=CG.WeightedDerivationSampler(completion_probability=0.1)))
    end
    providers = (CG.ContextHeadProvider(), CG.SafeListAccessProvider(), CG.FunctionDefProvider())
    rng1, rng2 = Random.Xoshiro(1111), Random.Xoshiro(1111)
    target = TyFun(TyList(FloatTy), FloatTy)
    globals = Dict{Symbol,Any}(:ZERO => 0.0)
    result1 = CG.generate_canonical(target; global_consts=globals,
        config=CG.GenerationConfig(rng=rng1, providers=providers, sampling_depth_limit=4))
    result2 = CG.generate_canonical(target; global_consts=globals,
        config=CG.GenerationConfig(rng=rng2, providers=providers, sampling_depth_limit=4,
            weights=CG.GenerationWeights(list_recursive_body_probability=0,
                list_recursive_tail_probability=0, functiondef_self_call_probability=0,
                safe_list_short_fallback_probability=0)))
    @test typeof(result1) == typeof(result2)
    if result1 isa Err
        @test typeof(result1.value) == typeof(result2.value)
    else
        @test result1.tree == result2.tree
        @test result1.local_consts == result2.local_consts
    end
    @test rand(rng1) == rand(rng2)
end

@testitem "CanonicalGeneration bounded certified safe body applies list recursion bias" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    list = TyList(FloatTy)
    fun = TyFun(list, FloatTy)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, scope_application=1,
        proxy_constructive=1, proxy_general=0, proxy_head=0, proxy_scope_application=0,
        proxy_split=1, constraint_evidence=0, list_recursive_body_probability=1-eps(Float64),
        list_recursive_tail_probability=1-eps(Float64))
    relation = CG.CoreRelation(weights=weights,
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0, :ADD => (x -> y -> x + y)),
        global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy,
            :ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    scope = Ty[list, fun]
    ledger = CG.FunctionDefAttemptBudget(CG.FunctionDefRouteToken(), [3, 4])
    query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=6, node_budget=20,
        requirements=(CG.FunctionDefSelfBinder(1, 1, false), CG.CallsScopeBinder(1), ledger))
    source = CG.ListAccessSource(list, CG.VariableDerivation(0, list))
    # Call the evidence-body entry itself, with the source/wrapper's four
    # emitted nodes already reserved. The ordinary safe-body hook is bypassed.
    state = CG.DerivationSamplingState(Random.Xoshiro(8000), 16)
    mode = state.completion
    result = CG._sample_certified_safe_list_body(state, relation, source,
        FloatTy, scope, 6, 0, query, 20)
    @test result isa CG.SafeListAccessDerivation
    if result isa CG.SafeListAccessDerivation
        @test CG.derivation_uses_scope_binder(result.body, 1)
        @test CG.derivation_calls_scope_binder(result.body, 3)
        @test CG._generation_requirements_satisfied(result, query)
        @test result.body isa CG.HeadApplicationDerivation
        if result.body isa CG.HeadApplicationDerivation
            calls = filter(d -> d isa CG.ScopeApplicationDerivation && d.index == 3,
                result.body.args)
            @test length(calls) == 1
            @test length(calls) != 1 || only(only(calls).args) == CG.VariableDerivation(0, list)
        end
        @test 20 - state.nodes_left == CG.derivation_node_count(result)
    end
    @test ledger.remaining_by_level == [3, 4]
    @test state.completion === mode
    @test !mode.completing && mode.boundaries == 0

    # No combining head exists: the joint proposal fails, refunds its nodes,
    # and the original OR certificate may still return self(tail).
    no_combine = CG.CoreRelation(weights=weights,
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    fallback_state = CG.DerivationSamplingState(Random.Xoshiro(42), 16)
    fallback = CG._sample_certified_safe_list_body(fallback_state, no_combine, source,
        FloatTy, scope, 6, 0, query, 20)
    @test fallback isa CG.SafeListAccessDerivation
    if fallback isa CG.SafeListAccessDerivation
        @test CG._generation_requirements_satisfied(fallback, query)
        @test !CG.derivation_uses_scope_binder(fallback.body, 1)
        @test 20 - fallback_state.nodes_left == CG.derivation_node_count(fallback)
    end
    @test ledger.remaining_by_level == [3, 4]
end

function rebase_generation_requirement(hint::_ListRecursionHint, added::Integer)
    shift = _checked_nonnegative_int(:added_binders, added)
    return _ListRecursionHint(hint.self_index + shift, hint.argument_position,
        hint.tail_index + shift, hint.head_index + shift)
end
generation_requirement_affects_core_search(::_ListRecursionHint) = false
generation_requirement_cache_key(hint::_ListRecursionHint) =
    (:list_recursion_hint, hint.self_index, hint.argument_position,
        hint.tail_index, hint.head_index)

_list_recursion_bias_enabled(weights::GenerationWeights) =
    get(weights.overrides, :list_recursive_body_probability, 0.0) > 0 ||
    get(weights.overrides, :list_recursive_tail_probability, 0.0) > 0

_has_list_recursion_hint(::Nothing) = false
function _has_list_recursion_hint(query::GenerationQuery)
    for hint in query.requirements
        hint isa _ListRecursionHint || continue
        any(r -> r isa CallsScopeBinder && r.index == hint.self_index,
            query.requirements) && return true
    end
    return false
end

"""Attach finite same-source identities at one safe-list success boundary.

Requires the enclosing query's real FunctionDef marker and an original list
parameter as the selected source. Computed sources and unrelated same-typed
variables are deliberately ineligible. This does not register the tail as a
new access source, introduce types, draw randomness, or replenish any budget.
"""
function _with_list_recursion_hints(relation::CoreRelation, source::ListAccessSource,
    parent::Union{Nothing,GenerationQuery}, body::Union{Nothing,GenerationQuery})
    _list_recursion_bias_enabled(relation.weights) || return body
    parent === nothing && return body
    body === nothing && return body
    source.derivation isa VariableDerivation || return body
    source_index = source.derivation.index
    hints = _ListRecursionHint[]
    for marker in parent.requirements
        marker isa FunctionDefSelfBinder || continue
        position = marker.index - source_index
        1 <= position <= marker.leading_parameters || continue
        source_index < length(parent.scope_tys) || continue
        source.ty isa TyList || continue
        iscompatible(parent.scope_tys[source_index + 1], source.ty) || continue
        push!(hints, _ListRecursionHint(marker.index + 2, position, 0, 1))
    end
    isempty(hints) && return body
    return _derive_generation_query(_close_functiondef_prefix(body); requirements=Tuple(hints))
end

"""Try one joint element-use/self-call body with the existing proxy planner.

`query` carries exact safe-source identities and the original decreasing
bounds. A single concrete head is the proxy obligation; the genuine self call
is a separate conjunctive request. No operator or complete traversal is fixed.
One failed bounded attempt refunds nodes and leaves the caller's ordinary
route available; shared FunctionDef attempt balances are never reset.
"""
function _sample_list_recursive_body(state::DerivationSamplingState,
    relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty}, depth_left::Int,
    guarded_left::Int, query::Union{Nothing,GenerationQuery})
    probability = get(relation.weights.overrides, :list_recursive_body_probability, 0.0)
    probability > 0 && query !== nothing || return nothing
    hints = _ListRecursionHint[r for r in query.requirements if r isa _ListRecursionHint &&
        r.head_index == 1 && r.tail_index == 0 && _state_helper_hint_admitted(query, r)]
    isempty(hints) && return nothing
    state.completion.probability == 0 && !state.completion.completing ||
        throw(ArgumentError("list recursion bias requires bounded sampling"))
    rand(state.rng) < probability || return nothing
    hint = length(hints) == 1 ? only(hints) : rand(state.rng, hints)
    forbidden = _forbidden_scope_binders(query)
    (hint.head_index in forbidden || hint.self_index in forbidden) && return nothing
    joint = _derive_generation_query(query; requirements=(CallsScopeBinder(hint.self_index),
        _state_helper_body_requirements(query, hint)...))
    saved = state.nodes_left
    body = _sample_proxy_using_derivation(state, relation, target_ty, scope_tys,
        depth_left, guarded_left, 1, hint.head_index; generation_query=joint)
    if body !== nothing && derivation_uses_scope_binder(body, hint.head_index) &&
       _generation_requirements_satisfied(body, joint)
        return body
    end
    state.nodes_left = saved
    return nothing
end

"""Try the chosen recursive head with its same-source strict tail argument.

The existing rule determines arity/types; the hint fixes only the matching
list argument. Other operands retain ordinary bounded generation. Replacing a
child cannot discharge unrelated positive requirements: this entry is used
only for an unconstrained call or a request whose sole positive identity is
the selected head. A rejected attempt refunds emitted nodes, not provider fuel.
"""
function _sample_preferred_list_recursive_call!(state::DerivationSamplingState,
    relation::CoreRelation, rule::ScopeApplicationRule, scope_tys::Vector{Ty},
    depth_left::Int, guarded_left::Int, allow_safe_list_access::Bool,
    query::Union{Nothing,GenerationQuery})
    probability = get(relation.weights.overrides, :list_recursive_tail_probability, 0.0)
    probability > 0 && query !== nothing && depth_left >= 0 || return nothing
    update = _state_update_request(query, rule.index)
    positives = _required_scope_binder_indices(query, length(scope_tys))
    update !== nothing || all(==(rule.index), positives) || return nothing
    forbidden = _forbidden_scope_binders(query)
    hints = _ListRecursionHint[]
    for hint in query.requirements
        hint isa _ListRecursionHint && hint.self_index == rule.index || continue
        update === nothing || hint.argument_position == 2 || continue
        1 <= hint.argument_position <= length(rule.argument_tys) || continue
        0 <= hint.tail_index < length(scope_tys) || continue
        hint.tail_index in forbidden && continue
        rule.index in forbidden && continue
        argument_ty = rule.argument_tys[hint.argument_position]
        argument_ty isa TyList || continue
        iscompatible(scope_tys[hint.tail_index + 1], argument_ty) || continue
        push!(hints, hint)
    end
    isempty(hints) && return nothing
    state.completion.probability == 0 && !state.completion.completing ||
        throw(ArgumentError("list recursion bias requires bounded sampling"))
    rand(state.rng) < probability || return nothing
    hint = length(hints) == 1 ? only(hints) : rand(state.rng, hints)
    saved = state.nodes_left
    _consume_sampling_nodes!(state, length(rule.argument_tys)) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    free = _without_scope_binder_requirements(query)
    for i in _argument_resolution_order(rule)
        argument = if i == hint.argument_position
            _consume_sampling_nodes!(state, 1) ?
                VariableDerivation(hint.tail_index, scope_tys[hint.tail_index + 1]) : nothing
        elseif update !== nothing && i == 1
            _sample_joint_list_state_argument(state, relation, rule, scope_tys,
                depth_left, guarded_left, allow_safe_list_access, query, update, hint)
        else
            _sample_required_scope_free_argument(state, relation, rule, i,
                scope_tys, depth_left, guarded_left, allow_safe_list_access, free, query)
        end
        if argument === nothing || !_required_source_argument_admitted(rule, i, argument)
            state.nodes_left = saved
            return nothing
        end
        args[i] = argument
    end
    result = ScopeApplicationDerivation(rule.index, rule.head_ty, args)
    _generation_requirements_satisfied(result, query) || error("preferred tail call lost an obligation")
    return result
end
