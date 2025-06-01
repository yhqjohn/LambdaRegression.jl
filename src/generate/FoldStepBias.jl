"""Whether a bounded-only proposal policy requires the default sampler/planner."""
_bounded_proposal_bias_enabled(weights::GenerationWeights) =
    _bounded_recursion_bias_enabled(weights) ||
    get(weights.overrides, :fold_step_joint_use_probability, 0.0) > 0 ||
    get(weights.overrides, :list_state_helper, 0.0) > 0

"""Try one AFOLDL callback using the exact fresh item and old-state identities.

The existing callback grammar supplies two lambdas and the joint occurrence
planner supplies their body, without prescribing an update operator. A
function-valued old state must be called, not merely passed or captured. The
parent query contributes all inherited requirements and finite budgets; when
`prefix_count > 0`, one of its existing proxy alternatives must still occur.
No self marker is introduced for these ordinary callback parameters.

No new `ctx` fields are required. Zero probability returns before RNG use;
insufficient depth/fuel, impossible joint use, or inherited forbids fail back
to the caller's original menu. Failed proposals refund nodes, never the shared
FunctionDef attempt ledger or completion state. Nonzero use requires bounded
default sampling/planning, validated by the public generation config boundary.
"""
function _sample_joint_afoldl_step(state::DerivationSamplingState,
    relation::CoreRelation, rule::CoreRule, scope_tys::Vector{Ty}, depth_left::Int,
    guarded_left::Int, allow_safe_list_access::Bool,
    parent::Union{Nothing,GenerationQuery}; prefix_count::Int=0, prefix_offset::Int=0)
    probability = get(relation.weights.overrides, :fold_step_joint_use_probability, 0.0)
    probability > 0 || return nothing
    state.completion.probability == 0 && !state.completion.completing &&
        relation.argument_planner isa DefaultArgumentPlanner || throw(ArgumentError(
            "fold_step_joint_use_probability requires bounded default sampling/planning"))
    parts = _afoldl_step_parts(rule)
    parts === nothing && return nothing
    depth_left >= 2 && state.nodes_left >= 3 || return nothing
    rand(state.rng) < probability || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 2) || return nothing
    body_scope = _afoldl_step_scope(rule, scope_tys)
    inherited = parent === nothing ? GenerationQuery(parts.body_ty;
        scope_tys, depth_left, guarded_left, node_budget=saved_nodes) :
        _close_functiondef_prefix(parent)
    body_query = _nested_generation_query(state, inherited, parts.body_ty,
        body_scope, depth_left - 2, guarded_left)
    state_use = generation_function_view(parts.acc_ty) === nothing ?
        UsesScopeBinder(1) : CallsScopeBinder(1)
    joint = _derive_generation_query(body_query;
        requirements=(UsesScopeBinder(0), state_use))
    # In the ordinary entry the item is a single mandatory proxy. The other
    # entry keeps its original OR-prefix, shifted past the two fresh binders;
    # item and state remain separate conjunctive query requirements.
    count = prefix_count > 0 ? prefix_count : 1
    offset = prefix_count > 0 ? prefix_offset + 2 : 0
    body = _sample_proxy_using_derivation(state, relation, parts.body_ty,
        body_scope, depth_left - 2, guarded_left, count, offset;
        allow_safe_list_access, generation_query=joint)
    if body !== nothing && _generation_requirements_satisfied(body, joint)
        return LambdaDerivation(parts.acc_ty, LambdaDerivation(parts.elem_ty, body))
    end
    state.nodes_left = saved_nodes
    return nothing
end

@testitem "CanonicalGeneration fold callback joint proposal preserves roles and inherited requirements" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, TyMu, FloatTy, newtyvar, typekey
    using LambdaRegression.Primitives: AFOLDL, FADD
    CG = LambdaRegression.CanonicalGeneration
    probability = 1 - eps(Float64)
    weights = CG.GenerationWeights(default=0, variable=1, constant=1, lambda=1,
        scope_application=1, proxy_constructive=1, proxy_general=0,
        fold_step_joint_use_probability=probability)
    relation = CG.CoreRelation(; weights, providers=(CG.ContextHeadProvider(),),
        global_consts=Dict{Symbol,Any}(:ADD => FADD))
    array = TyArray(TyNat(2), FloatTy)
    function fold_rule(state_ty)
        step = TyFun(state_ty, TyFun(FloatTy, state_ty))
        head = CG.HeadEntry(:AFOLDL, :guarded_primitive,
            TyFun(step, TyFun(state_ty, TyFun(array, state_ty))),
            CG.Decoration(), CG.WeightBucket(:guarded_expansion);
            metadata=(value=AFOLDL, guarded=true))
        return CG.CoreRule(head, state_ty, Ty[step, state_ty, array],
            [:value, :value, :source_array])
    end
    rule = fold_rule(FloatTy)
    scope = Ty[array]
    parent = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=6, node_budget=24)
    state = CG.DerivationSamplingState(Random.Xoshiro(1111), 24)
    mode = state.completion
    step = CG._sample_afoldl_step_argument(state, relation, rule, scope, 6, 0, false, parent)
    @test step isa CG.LambdaDerivation
    if step isa CG.LambdaDerivation
        body = step.body.body
        @test CG.derivation_uses_scope_binder(body, 0) # item
        @test CG.derivation_uses_scope_binder(body, 1) # acc, despite identical type
        @test body isa CG.HeadApplicationDerivation
        @test CG.derivation_node_count(step) == 24 - state.nodes_left
        @test typekey(CG.derivation_result_type(step)) == typekey(first(rule.argument_tys))
    end
    @test state.completion === mode
    @test isempty(parent.requirements)

    # Keep an inherited OR-proxy while a separate exact use and forbid select
    # its second member. The fresh same-typed callback binders cannot satisfy
    # this outer requirement by accidental slot aliasing.
    outer = Ty[FloatTy, FloatTy, array]
    required = CG.GenerationQuery(FloatTy; scope_tys=outer, depth_left=7,
        node_budget=32, requirements=(CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0)))
    required_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 32)
    required_step = CG._sample_proxy_using_afoldl_step_argument(required_state,
        relation, rule, outer, 7, 0, 2, 0, false, required)
    @test required_step isa CG.LambdaDerivation
    if required_step isa CG.LambdaDerivation
        body = required_step.body.body
        @test CG.derivation_uses_scope_binder(body, 0)
        @test CG.derivation_uses_scope_binder(body, 1)
        @test CG.derivation_uses_scope_binder(body, 3)
        @test !CG.derivation_uses_scope_binder(body, 2)
        @test CG._generation_requirements_satisfied(required_step, required)
        @test 32 - required_state.nodes_left == CG.derivation_node_count(required_step)
    end
    @test required.requirements == (CG.UsesScopeBinder(1), CG.ForbidsScopeBinder(0))

    # Function-valued state is an ordinary callback argument, with no recursive
    # provider or self marker in this world. Passing/capturing acc is insufficient.
    function_state = TyFun(FloatTy, FloatTy)
    higher_rule = fold_rule(function_state)
    higher = CG.DerivationSamplingState(Random.Xoshiro(8000), 32)
    higher_step = CG._sample_afoldl_step_argument(higher, relation, higher_rule,
        scope, 7, 0, false)
    @test higher_step isa CG.LambdaDerivation
    if higher_step isa CG.LambdaDerivation
        body = higher_step.body.body
        @test CG.derivation_uses_scope_binder(body, 0)
        @test CG.derivation_calls_scope_binder(body, 1)
        @test 32 - higher.nodes_left == CG.derivation_node_count(higher_step)
    end

    # An impossible joint proposal leaves all node fuel for the original menu,
    # where a grounded callback or a one-parameter synthesized body is legal.
    plain = CG.CoreRelation(; weights, providers=(CG.ContextHeadProvider(),))
    callback_scope = Ty[first(rule.argument_tys), array]
    failed = CG.DerivationSamplingState(Random.Xoshiro(42), 12)
    @test CG._sample_joint_afoldl_step(failed, plain, rule, callback_scope,
        2, 0, false, nothing) === nothing
    @test failed.nodes_left == 12
    fallback = CG.DerivationSamplingState(Random.Xoshiro(42), 12)
    ordinary = CG._sample_afoldl_step_argument(fallback, plain, rule,
        callback_scope, 2, 0, false)
    @test ordinary isa CG.CoreDerivation
    @test 12 - fallback.nodes_left == CG.derivation_node_count(ordinary)
    short = CG.DerivationSamplingState(Random.Xoshiro(555), 2)
    rng_copy = copy(short.rng)
    @test CG._sample_joint_afoldl_step(short, relation, rule, scope,
        6, 0, false, parent) === nothing
    @test short.nodes_left == 2
    @test rand(short.rng) == rand(rng_copy)

    # The finite core is not expanded to infer arbitrary mu demands. This
    # opaque higher-order boundary must either fail with a full refund or emit
    # one finite, correctly charged role-preserving derivation.
    variable = newtyvar(0)
    recursive_state = TyMu(variable, TyFun(FloatTy, variable))
    recursive_rule = fold_rule(recursive_state)
    recursive_budget = CG.DerivationSamplingState(Random.Xoshiro(1), 8)
    recursive_step = CG._sample_joint_afoldl_step(recursive_budget, relation,
        recursive_rule, scope, 4, 0, false, nothing)
    @test recursive_step === nothing || recursive_step isa CG.LambdaDerivation
    if recursive_step === nothing
        @test recursive_budget.nodes_left == 8
    else
        @test CG.derivation_uses_scope_binder(recursive_step.body.body, 0)
        @test CG.derivation_calls_scope_binder(recursive_step.body.body, 1)
        @test 8 - recursive_budget.nodes_left == CG.derivation_node_count(recursive_step)
    end
end

@testitem "CanonicalGeneration fold callback bias is neutral at zero and rejects unsupported modes" begin
    using Test, Random
    using LambdaRegression.TypeSystem: Ty, TyFun, TyArray, TyNat, FloatTy
    using LambdaRegression.Primitives: AFOLDL, FADD
    CG = LambdaRegression.CanonicalGeneration
    for invalid in (-0.1, 1.0, Inf, NaN)
        @test_throws ArgumentError CG.GenerationWeights(fold_step_joint_use_probability=invalid)
    end
    positive = CG.GenerationWeights(fold_step_joint_use_probability=0.5)
    @test CG._uses_lazy_sampling(CG.GenerationConfig(weights=positive))
    for config in (
        CG.GenerationConfig(weights=positive,
            sampler=CG.WeightedDerivationSampler(completion_probability=0.1)),
        CG.GenerationConfig(weights=positive, sampler=(args...) -> nothing),
        CG.GenerationConfig(weights=positive, argument_planner=(args...) -> nothing),
    )
        @test_throws ArgumentError CG._uses_lazy_sampling(config)
        @test_throws ArgumentError CG._validate_functiondef_sampling_config(config)
    end

    array = TyArray(TyNat(2), FloatTy)
    step_ty = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    head = CG.HeadEntry(:AFOLDL, :guarded_primitive,
        TyFun(step_ty, TyFun(FloatTy, TyFun(array, FloatTy))), CG.Decoration(),
        CG.WeightBucket(:guarded_expansion); metadata=(value=AFOLDL, guarded=true))
    rule = CG.CoreRule(head, FloatTy, Ty[step_ty, FloatTy, array],
        [:value, :value, :source_array])
    scope = Ty[FloatTy, array]
    function neutral(explicit, proxy)
        weights = explicit ? CG.GenerationWeights(fold_step_joint_use_probability=0) :
            CG.GenerationWeights()
        relation = CG.CoreRelation(; weights, providers=(CG.ContextHeadProvider(),),
            global_consts=Dict{Symbol,Any}(:ADD => FADD))
        query = CG.GenerationQuery(FloatTy; scope_tys=scope, depth_left=4, node_budget=16)
        state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
        result = proxy ? CG._sample_proxy_using_afoldl_step_argument(state,
            relation, rule, scope, 4, 0, 1, 0, false, query) :
            CG._sample_afoldl_step_argument(state, relation, rule, scope, 4, 0, false, query)
        return (; signature=result === nothing ? nothing : CG.derivation_signature(result),
            nodes=state.nodes_left, next=rand(state.rng, UInt64))
    end
    @test neutral(false, false) == neutral(true, false)
    @test neutral(false, true) == neutral(true, true)
end
