"""Whether any bounded-only recursive proposal policy is active."""
_bounded_recursion_bias_enabled(weights::GenerationWeights) =
    _list_recursion_bias_enabled(weights) ||
    get(weights.overrides, :functiondef_self_call_probability, 0.0) > 0 ||
    get(weights.overrides, :safe_list_short_fallback_probability, 0.0) > 0

"""Try one genuine self-call body before ordinary FunctionDef body sampling.

The provider has already anchored its acyclic base and charged its wrapper.
`query` supplies the enclosing bounds and shared attempt ledger; `body_parent`
has the original leading prefix closed. At least two attempts must remain at
this definition's level: this proposal consumes one, reserving one for the
ordinary route if it fails. Nested definitions keep their lower-level balances.
The existing exact Call planner selects the body, with no prescribed operator,
list case, argument or recursion tree. Every result passes the provider's
original admissibility. Failure refunds emitted derivation nodes only.
"""
function _sample_preferred_functiondef_body(state::DerivationSamplingState,
    relation::CoreRelation, provider::FunctionDefProvider, option,
    query::GenerationQuery, body_parent::GenerationQuery, base_witness::CoreDerivation;
    requirements=())
    probability = get(relation.weights.overrides, :functiondef_self_call_probability, 0.0)
    probability > 0 || return nothing
    provider.admissibility isa GenuineSelfCallFilter && return nothing
    _functiondef_attempts_remaining(query, provider) >= 2 || return nothing
    state.completion.probability == 0 && !state.completion.completing ||
        throw(ArgumentError("FunctionDef self-call proposal requires bounded sampling"))
    rand(state.rng) < probability || return nothing
    saved = state.nodes_left
    _consume_functiondef_attempt!(query, provider) || return nothing
    accept_body = body -> _functiondef_sampling_admissible(provider.admissibility,
        FunctionDefCandidate(query.target_ty, body, base_witness, 0))
    body = sample_generation_derivation_using_scope_binder(state, relation,
        body_parent, query.target_ty; node_budget=min(query.node_budget, state.nodes_left),
        requirements=(FunctionDefFuel(option.remaining - 1),
            _functiondef_attempt_budget(query, provider), FunctionDefSelfBinder(0), requirements...),
        allow_safe_list_access=true, allow_direct_result=false,
        require_scope_application=true, accept_derivation=accept_body)
    if body !== nothing
        candidate = FunctionDefCandidate(query.target_ty, body, base_witness, 0)
        _derivation_has_self_call(body, 0) || error("preferred FunctionDef body lost its self call")
        _functiondef_admissible(provider, candidate, relation, query) && return body
    end
    state.nodes_left = saved
    return nothing
end

@testitem "CanonicalGeneration preferred self-call body shares finite attempts and ordinary fallback" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, FloatTy
    CG = LambdaRegression.CanonicalGeneration
    probability = 1 - eps(Float64)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, lambda=1,
        scope_application=1, constraint_evidence=0,
        functiondef_self_call_probability=probability)
    relation = CG.CoreRelation(weights=weights,
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy),
        providers=(CG.ContextHeadProvider(),))
    fun = TyFun(FloatTy, FloatTy)
    provider = CG.FunctionDefProvider(max_nesting=1, body_attempts=4)
    function request(provider; attempts=4, nodes=24)
        ledger = CG.FunctionDefAttemptBudget(provider.route_token, [attempts])
        query = CG.GenerationQuery(fun; depth_left=4, node_budget=nodes,
            requirements=(ledger,))
        option = (; remaining=1, target_ty=fun, base_depth_left=4, base_node_budget=nodes)
        return query, option, ledger
    end
    query, option, ledger = request(provider)
    base = CG.LambdaDerivation(FloatTy, CG.VariableDerivation(0, FloatTy))
    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 24)
    mode = state.completion
    body = CG._sample_preferred_functiondef_body(state, relation, provider, option,
        query, CG._close_functiondef_prefix(query), base)
    @test body isa CG.CoreDerivation
    if body isa CG.CoreDerivation
        @test CG.derivation_calls_scope_binder(body, 0)
        @test 24 - state.nodes_left == CG.derivation_node_count(body)
        # The permissive provider above does not accidentally acquire a new
        # direct-loop policy. The real filter still rejects this unguarded body.
        @test !CG.DirectKnotFilter()(CG.FunctionDefCandidate(fun, body, base))
    end
    @test ledger.remaining_by_level == [3]
    # Ordinary core sampling may advance its diagnostic boundary counter even
    # at probability zero; the shared mode itself must never be replaced.
    @test state.completion === mode && !mode.completing && mode.probability == 0

    # A real rejecting policy spends its preferred attempt, refunds nodes,
    # and leaves the final ordinary attempt available without replenishment.
    filtered = CG.FunctionDefProvider(max_nesting=1, body_attempts=4,
        admissibility=CG.DirectKnotFilter())
    rejectedq, rejected_option, rejected_ledger = request(filtered)
    rejected = CG.DerivationSamplingState(Random.Xoshiro(2000), 24)
    @test CG._sample_preferred_functiondef_body(rejected, relation, filtered,
        rejected_option, rejectedq, CG._close_functiondef_prefix(rejectedq), base) === nothing
    @test rejected.nodes_left == 24
    @test rejected_ledger.remaining_by_level == [3]
    @test CG._consume_functiondef_attempt!(rejectedq, filtered)
    @test rejected_ledger.remaining_by_level == [2]

    # Integration: a user policy rejects the preferred candidate once, then
    # accepts the ordinary route. There are exactly two ledger charges.
    mutable struct FirstCandidateRejected
        calls::Int
    end
    function (policy::FirstCandidateRejected)(candidate)
        policy.calls += 1
        return policy.calls > 1
    end
    policy = FirstCandidateRejected(0)
    mixed = CG.FunctionDefProvider(max_nesting=1, body_attempts=4, admissibility=policy)
    mixedq, mixed_option, mixed_ledger = request(mixed)
    mixed_state = CG.DerivationSamplingState(Random.Xoshiro(1), 24)
    result = CG.sample_generation_extension(mixed, mixed_state, relation, mixed_option, mixedq)
    @test result isa CG.FunctionDefDerivation
    @test policy.calls == 2
    @test mixed_ledger.remaining_by_level == [2]
    if result isa CG.FunctionDefDerivation
        @test 24 - mixed_state.nodes_left == CG.derivation_node_count(result)
    end

    # With one remaining attempt there is no new coin flip or ledger use.
    lastq, last_option, last_ledger = request(provider; attempts=1)
    lastrng = Random.Xoshiro(1111)
    control = copy(lastrng)
    @test CG._sample_preferred_functiondef_body(CG.DerivationSamplingState(lastrng, 24),
        relation, provider, last_option, lastq, CG._close_functiondef_prefix(lastq), base) === nothing
    @test last_ledger.remaining_by_level == [1]
    @test rand(lastrng) == rand(control)
    forced = CG._require_genuine_self_call(provider)
    forcedq, forced_option, forced_ledger = request(forced)
    forcedrng = Random.Xoshiro(1234)
    forced_control = copy(forcedrng)
    @test CG._sample_preferred_functiondef_body(CG.DerivationSamplingState(forcedrng, 24),
        relation, forced, forced_option, forcedq, CG._close_functiondef_prefix(forcedq), base) === nothing
    @test forced_ledger.remaining_by_level == [4]
    @test rand(forcedrng) == rand(forced_control)
end

@testitem "CanonicalGeneration short recursive-context fallback preserves binding separation" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyList, TyMu, FloatTy, newtyvar
    CG = LambdaRegression.CanonicalGeneration
    probability = 1 - eps(Float64)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1,
        lambda=1, scope_application=1, proxy_constructive=1, proxy_general=0,
        proxy_head=0, proxy_scope_application=0, proxy_split=1, constraint_evidence=0,
        list_recursive_body_probability=probability, list_recursive_tail_probability=probability,
        safe_list_short_fallback_probability=probability)
    relation = CG.CoreRelation(weights=weights,
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0, :ADD => (x -> y -> x+y)),
        global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy,
            :ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    list, fun = TyList(FloatTy), TyFun(FloatTy, FloatTy)
    # Equal-typed ordinary function at #1 remains eligible; marked selves at
    # #0 and #2 are both excluded. A supplied source name is never inspected.
    scope = Ty[fun, fun, fun]
    query = CG.GenerationQuery(fun; scope_tys=scope, depth_left=4, node_budget=20,
        requirements=(CG.FunctionDefSelfBinder(0), CG.FunctionDefSelfBinder(2),
            CG.CallsScopeBinder(0)))
    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 20)
    fallback = CG._sample_short_nonrecursive_list_fallback(state, relation, fun,
        scope, 3, 0, query)
    @test fallback == CG.VariableDerivation(1, fun)
    @test state.nodes_left == 19
    @test isempty(CG._forbidden_scope_binders(query))
    @test CG.CallsScopeBinder(0) in query.requirements
    # Entering a higher-order lexical binder rebases both forbidden selves,
    # while the new lambda argument remains an ordinary candidate.
    higher = TyFun(fun, fun)
    higherq = CG._derive_generation_query(query; added_scope_tys=(higher,))
    higher_scope = Ty[higher, scope...]
    higher_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 20)
    higher_fallback = CG._sample_short_nonrecursive_list_fallback(higher_state,
        relation, fun, higher_scope, 3, 0, higherq)
    @test higher_fallback == CG.VariableDerivation(2, fun)

    # No real self marker means no policy action, even at an identical type.
    plain = CG.GenerationQuery(fun; scope_tys=scope, depth_left=4, node_budget=20)
    rng = Random.Xoshiro(42)
    control = copy(rng)
    @test CG._sample_short_nonrecursive_list_fallback(CG.DerivationSamplingState(rng, 20),
        relation, fun, scope, 3, 0, plain) === nothing
    @test rand(rng) == rand(control)
    zero = CG.DerivationSamplingState(Random.Xoshiro(555), 0)
    @test CG._sample_short_nonrecursive_list_fallback(zero, relation, fun,
        scope, 3, 0, query) === nothing
    @test zero.nodes_left == 0
    recursive_var = newtyvar(0)
    recursive = TyMu(recursive_var, TyFun(FloatTy, recursive_var))
    muq = CG.GenerationQuery(FloatTy; scope_tys=Ty[recursive], depth_left=2,
        node_budget=8, requirements=(CG.FunctionDefSelfBinder(0),))
    mustate = CG.DerivationSamplingState(Random.Xoshiro(2000), 8)
    @test CG._sample_short_nonrecursive_list_fallback(mustate, relation, FloatTy,
        Ty[recursive], 1, 0, muq) === nothing
    @test mustate.nodes_left == 8

    # Both safe-body entries keep the Call obligation in success while their
    # independent fallback uses a one-node core leaf, with exact charges.
    listfun = TyFun(list, FloatTy)
    listscope = Ty[list, listfun]
    listq = CG.GenerationQuery(FloatTy; scope_tys=listscope, depth_left=6,
        node_budget=20, requirements=(CG.FunctionDefSelfBinder(1, 1, false),
            CG.CallsScopeBinder(1)))
    source = CG.ListAccessSource(list, CG.VariableDerivation(0, list))
    for sampler in (CG._sample_safe_list_access_body, CG._sample_certified_safe_list_body)
        candidate_state = CG.DerivationSamplingState(Random.Xoshiro(1), 20)
        candidate = CG._sample_safe_list_access(candidate_state, relation, source,
            FloatTy, listscope, 6, 0, listq; body_sampler=sampler)
        @test candidate isa CG.SafeListAccessDerivation
        if candidate isa CG.SafeListAccessDerivation
            @test CG.derivation_node_count(candidate.fallback) == 1
            @test !CG.derivation_uses_scope_binder(candidate.fallback, 1)
            @test CG.derivation_calls_scope_binder(candidate.body, 3)
            @test CG.derivation_uses_scope_binder(candidate.body, 1)
            @test CG._generation_requirements_satisfied(candidate, listq)
            @test 20 - candidate_state.nodes_left == CG.derivation_node_count(candidate)
            @test isempty(CG._forbidden_scope_binders(listq))
        end
    end
end

"""Prefer one short core fallback excluding all visible recursive self binders.

This local safe-list fallback proposal requires real `FunctionDefSelfBinder`
markers in `parent`; same-typed ordinary functions are not treated as self.
The finite core certificate records a minimum-cost witness in its own grammar;
replay does not explore arithmetic wrappers. A standalone child request drops
positive Use/Call obligations and forbids the marked identities, since only
the complete safe expression must discharge the parent's positive obligations.
That child never becomes the success body's parent. Failure or unsupported
evidence refunds nodes and leaves the original fallback route available.
This is a structural preference, not a numeric-domain or totality proof.
"""
function _sample_short_nonrecursive_list_fallback(state::DerivationSamplingState,
    relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty}, depth_left::Int,
    guarded_left::Int, parent::Union{Nothing,GenerationQuery})
    probability = get(relation.weights.overrides, :safe_list_short_fallback_probability, 0.0)
    probability > 0 && parent !== nothing || return nothing
    indices = unique(Int[r.index for r in parent.requirements if r isa FunctionDefSelfBinder])
    isempty(indices) && return nothing
    state.completion.probability == 0 && !state.completion.completing ||
        throw(ArgumentError("short recursive-context fallback requires bounded sampling"))
    rand(state.rng) < probability || return nothing
    saved = state.nodes_left
    child = _nested_generation_query(state, _without_scope_binder_requirements(parent),
        target_ty, scope_tys, depth_left, guarded_left)
    # Adding binders is supported by the existing nested-query protocol, so
    # rebase the source identities by the same concrete lexical shift.
    added = length(scope_tys) - length(parent.scope_tys)
    child = _derive_generation_query(child;
        requirements=Tuple(ForbidsScopeBinder(index + added) for index in indices))
    fallback = _replay_finite_core_certificate(state, relation, child)
    if fallback isa _UnsupportedEvidenceSampling || fallback === nothing
        state.nodes_left = saved
        return nothing
    end
    any(index -> derivation_uses_scope_binder(fallback, index + added), indices) &&
        error("short fallback replay used a forbidden recursive binder")
    return fallback
end
