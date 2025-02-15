const CG = CanonicalGeneration

"""Finite candidate passed to an injected FunctionDef admissibility policy."""
struct FunctionDefCandidate
    target_ty::Ty
    body::CG.CoreDerivation
    base_witness::CG.CoreDerivation
    self_index::Int

    function FunctionDefCandidate(target_ty, body, base_witness, self_index::Integer=0)
        target_ty isa Ty || throw(ArgumentError("FunctionDef target must subtype Ty"))
        body isa CG.CoreDerivation || throw(ArgumentError("FunctionDef body must be a CoreDerivation"))
        base_witness isa CG.CoreDerivation ||
            throw(ArgumentError("FunctionDef base witness must be a CoreDerivation"))
        self_index >= 0 || throw(ArgumentError("FunctionDef self index must be nonnegative"))
        return new(target_ty, body, base_witness, Int(self_index))
    end
end

"""
    DirectKnotFilter()

Reject a body that is exactly `self`, a finite eta-forwarding prefix such as
`λx. self x`, or a recursive call whose argument hides a safe-list case outside
its success guard. The filter is deliberately syntactic and total; richer loop
analysis and runtime limits remain separate injectable policies.
"""
struct DirectKnotFilter end

"""Default FunctionDef policy: accept every finite, well-typed sampled body."""
struct AcceptFunctionDef end
(::AcceptFunctionDef)(::FunctionDefCandidate, args...) = true

"""
    GenuineSelfCallFilter(inner=DirectKnotFilter())

Compose an existing FunctionDef admissibility policy with the stronger demand
that the selected self binder heads at least one application spine. Merely
returning self or passing it as an ordinary argument does not satisfy the filter.
"""
struct GenuineSelfCallFilter{A}
    inner::A
end
GenuineSelfCallFilter() = GenuineSelfCallFilter(DirectKnotFilter())

"""Default finite acyclic base-witness strategy for FunctionDef generation."""
struct FiniteBaseWitness end

(::FiniteBaseWitness)(relation, query, capability_mask) =
    CG._borrow_known_generation_witness(relation, query; capability_mask)

function _valid_functiondef_base_type(strategy, witness, target_ty)
    result_ty = CG.derivation_result_type(witness)
    return result_ty === target_ty || TypeSystem.iscompatible(result_ty, target_ty)
end

function (filter::DirectKnotFilter)(candidate::FunctionDefCandidate)
    body = candidate.body
    _is_direct_self_derivation(body, candidate.self_index) && return false
    _is_eta_forwarding_derivation(body, candidate.self_index) && return false
    _has_unguarded_safe_list_recursive_derivation(body, candidate.self_index) && return false
    return true
end

function _inner_functiondef_admissible(policy, candidate, args...)
    accepted = if applicable(policy, candidate, args...)
        policy(candidate, args...)
    elseif applicable(policy, candidate)
        policy(candidate)
    else
        throw(ArgumentError(
            "FunctionDef admissibility must accept candidate or " *
            "(candidate, relation, query)",
        ))
    end
    accepted isa Bool || throw(ArgumentError("FunctionDef admissibility must return Bool"))
    return accepted
end

function (filter::GenuineSelfCallFilter)(candidate::FunctionDefCandidate, args...)
    _inner_functiondef_admissible(filter.inner, candidate, args...) || return false
    return _derivation_has_self_call(candidate.body, candidate.self_index)
end

_derivation_has_self_call(
    ::Union{CG.VariableDerivation,CG.ConstantDerivation},
    self_index::Int,
) = false

_derivation_has_self_call(derivation::CG.LambdaDerivation, self_index::Int) =
    _derivation_has_self_call(derivation.body, self_index + 1)

function _derivation_has_self_call(
    derivation::CG.ScopeApplicationDerivation,
    self_index::Int,
)
    derivation.index == self_index && !isempty(derivation.args) && return true
    return any(argument -> _derivation_has_self_call(argument, self_index), derivation.args)
end

_derivation_has_self_call(derivation::CG.HeadApplicationDerivation, self_index::Int) =
    any(argument -> _derivation_has_self_call(argument, self_index), derivation.args)

function _derivation_has_self_call(
    derivation::CG.SafeListAccessDerivation,
    self_index::Int,
)
    return _derivation_has_self_call(derivation.source, self_index) ||
        _derivation_has_self_call(derivation.body, self_index + 2) ||
        _derivation_has_self_call(derivation.fallback, self_index)
end

_derivation_has_self_call(derivation::CG.CoreDerivation, self_index::Int) =
    _signature_has_self_call(CG.derivation_signature(derivation), self_index)

function _signature_has_self_call(signature, self_index::Int)
    signature isa Tuple || return false
    isempty(signature) && return false
    tag = signature[1]
    if tag === :lambda
        return length(signature) >= 3 &&
            _signature_has_self_call(signature[3], self_index + 1)
    elseif tag === :scope_application
        length(signature) >= 4 || return false
        arguments = signature[4]
        signature[2] == self_index && !isempty(arguments) && return true
        return any(argument -> _signature_has_self_call(argument, self_index), arguments)
    elseif tag === :head
        length(signature) >= 5 || return false
        return any(argument -> _signature_has_self_call(argument, self_index), signature[5])
    elseif tag === :safe_list_access
        length(signature) >= 7 || return false
        return _signature_has_self_call(signature[5], self_index) ||
            _signature_has_self_call(signature[6], self_index + 2) ||
            _signature_has_self_call(signature[7], self_index)
    elseif tag === :function_def
        length(signature) >= 4 || return false
        return _signature_has_self_call(signature[4], self_index + 1)
    end
    return false
end

_is_direct_self_derivation(derivation::CG.VariableDerivation, self_index::Int) =
    derivation.index == self_index
_is_direct_self_derivation(derivation::CG.CoreDerivation, self_index::Int) =
    _is_direct_self_signature(CG.derivation_signature(derivation), self_index)

_is_eta_forwarding_derivation(
    ::Union{
        CG.VariableDerivation,
        CG.ConstantDerivation,
        CG.HeadApplicationDerivation,
        CG.ScopeApplicationDerivation,
        CG.SafeListAccessDerivation,
    },
    self_index::Int,
) = false

function _is_eta_forwarding_derivation(derivation::CG.LambdaDerivation, self_index::Int)
    current = derivation
    lambda_count = 0
    while current isa CG.LambdaDerivation
        lambda_count += 1
        current = current.body
    end
    lambda_count > 0 || return false
    if !(current isa CG.ScopeApplicationDerivation)
        current isa Union{
            CG.VariableDerivation,
            CG.ConstantDerivation,
            CG.HeadApplicationDerivation,
            CG.SafeListAccessDerivation,
        } && return false
        return _is_eta_forwarding_signature(CG.derivation_signature(derivation), self_index)
    end
    current.index == self_index + lambda_count || return false
    length(current.args) == lambda_count || return false
    return all(enumerate(current.args)) do (position, argument)
        expected = lambda_count - position
        argument isa CG.VariableDerivation && argument.index == expected
    end
end

_is_eta_forwarding_derivation(derivation::CG.CoreDerivation, self_index::Int) =
    _is_eta_forwarding_signature(CG.derivation_signature(derivation), self_index)

_contains_safe_list_access(::CG.VariableDerivation) = false
_contains_safe_list_access(::CG.ConstantDerivation) = false
_contains_safe_list_access(derivation::CG.LambdaDerivation) =
    _contains_safe_list_access(derivation.body)
_contains_safe_list_access(derivation::CG.HeadApplicationDerivation) =
    any(_contains_safe_list_access, derivation.args)
_contains_safe_list_access(derivation::CG.ScopeApplicationDerivation) =
    any(_contains_safe_list_access, derivation.args)
_contains_safe_list_access(::CG.SafeListAccessDerivation) = true
_contains_safe_list_access(derivation::CG.CoreDerivation) =
    :safe_list_access in CG.derivation_capabilities(derivation) ||
    CG._signature_contains(CG.derivation_signature(derivation), :safe_list_access)

_has_unguarded_safe_list_recursive_derivation(
    ::Union{CG.VariableDerivation,CG.ConstantDerivation},
    self_index::Int,
    safe_list_depth::Int=0,
) = false

function _has_unguarded_safe_list_recursive_derivation(
    derivation::CG.LambdaDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    return _has_unguarded_safe_list_recursive_derivation(
        derivation.body,
        self_index + 1,
        safe_list_depth,
    )
end

function _has_unguarded_safe_list_recursive_derivation(
    derivation::CG.ScopeApplicationDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    if safe_list_depth == 0 && derivation.index == self_index &&
       any(_contains_safe_list_access, derivation.args)
        return true
    end
    return any(derivation.args) do argument
        _has_unguarded_safe_list_recursive_derivation(
            argument,
            self_index,
            safe_list_depth,
        )
    end
end

function _has_unguarded_safe_list_recursive_derivation(
    derivation::CG.HeadApplicationDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    return any(derivation.args) do argument
        _has_unguarded_safe_list_recursive_derivation(
            argument,
            self_index,
            safe_list_depth,
        )
    end
end

function _has_unguarded_safe_list_recursive_derivation(
    derivation::CG.SafeListAccessDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    return _has_unguarded_safe_list_recursive_derivation(
        derivation.source,
        self_index,
        safe_list_depth,
    ) || _has_unguarded_safe_list_recursive_derivation(
        derivation.body,
        self_index + 2,
        safe_list_depth + 1,
    ) || _has_unguarded_safe_list_recursive_derivation(
        derivation.fallback,
        self_index,
        safe_list_depth,
    )
end

function _has_unguarded_safe_list_recursive_derivation(
    derivation::CG.CoreDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    return _has_unguarded_safe_list_recursive_argument(
        CG.derivation_signature(derivation),
        self_index,
        safe_list_depth,
    )
end

function _is_direct_self_signature(signature, self_index::Int)
    return signature isa Tuple && length(signature) >= 2 &&
        signature[1] === :var && signature[2] == self_index
end

function _has_unguarded_safe_list_recursive_argument(
    signature,
    self_index::Int,
    safe_list_depth::Int=0,
)
    signature isa Tuple || return false
    isempty(signature) && return false
    tag = signature[1]
    if tag === :lambda
        return length(signature) >= 3 && _has_unguarded_safe_list_recursive_argument(
            signature[3],
            self_index + 1,
            safe_list_depth,
        )
    elseif tag === :scope_application
        length(signature) >= 4 || return false
        arguments = signature[4]
        if safe_list_depth == 0 && signature[2] == self_index &&
           any(argument -> CG._signature_contains(argument, :safe_list_access), arguments)
            return true
        end
        return any(
            argument -> _has_unguarded_safe_list_recursive_argument(
                argument,
                self_index,
                safe_list_depth,
            ),
            arguments,
        )
    elseif tag === :head
        length(signature) >= 5 || return false
        return any(
            argument -> _has_unguarded_safe_list_recursive_argument(
                argument,
                self_index,
                safe_list_depth,
            ),
            signature[5],
        )
    elseif tag === :safe_list_access
        length(signature) >= 7 || return false
        return _has_unguarded_safe_list_recursive_argument(
            signature[5],
            self_index,
            safe_list_depth,
        ) || _has_unguarded_safe_list_recursive_argument(
            signature[6],
            self_index + 2,
            safe_list_depth + 1,
        ) || _has_unguarded_safe_list_recursive_argument(
            signature[7],
            self_index,
            safe_list_depth,
        )
    elseif tag === :function_def
        length(signature) >= 4 || return false
        return _has_unguarded_safe_list_recursive_argument(
            signature[4],
            self_index + 1,
            safe_list_depth,
        )
    end
    return false
end

function _is_eta_forwarding_signature(signature, self_index::Int)
    current = signature
    lambda_count = 0
    while current isa Tuple && length(current) >= 3 && current[1] === :lambda
        lambda_count += 1
        current = current[3]
    end
    lambda_count > 0 || return false
    current isa Tuple && length(current) >= 4 || return false
    current[1] === :scope_application || return false
    current[2] == self_index + lambda_count || return false
    arguments = current[4]
    arguments isa Tuple && length(arguments) == lambda_count || return false
    return all(enumerate(arguments)) do (position, argument)
        expected = lambda_count - position
        argument isa Tuple && length(argument) >= 2 &&
            argument[1] === :var && argument[2] == expected
    end
end

struct FunctionDefFuel <: CG.AbstractGenerationRequirement
    remaining::Int
    function FunctionDefFuel(remaining::Integer)
        remaining >= 0 || throw(ArgumentError("FunctionDef fuel must be nonnegative"))
        return new(Int(remaining))
    end
end

struct FunctionDefSelfBinder <: CG.AbstractGenerationRequirement
    index::Int
    function FunctionDefSelfBinder(index::Integer)
        index >= 0 || throw(ArgumentError("FunctionDef self binder index must be nonnegative"))
        return new(Int(index))
    end
end

const _FUNCTIONDEF_ROUTE_IDS = Threads.Atomic{UInt64}(0)

struct FunctionDefRouteToken
    id::UInt64
end

FunctionDefRouteToken() =
    FunctionDefRouteToken(Threads.atomic_add!(_FUNCTIONDEF_ROUTE_IDS, UInt64(1)) + UInt64(1))

mutable struct FunctionDefAttemptBudget <: CG.AbstractGenerationRequirement
    route_token::FunctionDefRouteToken
    remaining_by_level::Vector{Int}
end

CG.rebase_generation_requirement(fuel::FunctionDefFuel, ::Integer) = fuel
CG.rebase_generation_requirement(budget::FunctionDefAttemptBudget, ::Integer) = budget
CG.rebase_generation_requirement(marker::FunctionDefSelfBinder, added::Integer) =
    FunctionDefSelfBinder(marker.index + CG._checked_nonnegative_int(:added_binders, added))
CG.generation_requirement_affects_core_search(::FunctionDefFuel) = false
CG.generation_requirement_affects_core_search(::FunctionDefAttemptBudget) = false
CG.generation_requirement_affects_core_search(::FunctionDefSelfBinder) = false
CG.generation_requirement_cache_key(fuel::FunctionDefFuel) =
    (:function_def_fuel, fuel.remaining)
CG.generation_requirement_cache_key(budget::FunctionDefAttemptBudget) =
    (:function_def_attempt_budget, budget.route_token.id)
CG.generation_requirement_cache_key(marker::FunctionDefSelfBinder) =
    (:function_def_self_binder, marker.index)

"""
    FunctionDefProvider(; weight_bucket=:function_def,
                        base_capability_mask=(),
                        base_witness=FiniteBaseWitness(),
                        admissibility=AcceptFunctionDef(),
                        max_nesting=2, body_attempts=4)

Opt-in FunctionDef provider. When its route is sampled, it first establishes a
finite acyclic base witness, then adds the monomorphic self binder and performs
one ordinary bounded body sample. `body_attempts` bounds the shared number of
FunctionDef route attempts at each permitted nesting level; only an explicitly
forced `GenuineSelfCallFilter` route may spend more than one attempt in a single
call. `base_witness` is an injectable finite callable accepting
`(relation, query, capability_mask)`.
"""
struct FunctionDefProvider{W,B,A} <: CG.AbstractHeadProvider
    weight_bucket::W
    base_capability_mask::Tuple{Vararg{Symbol}}
    base_witness::B
    admissibility::A
    max_nesting::Int
    body_attempts::Int
    route_token::FunctionDefRouteToken
end

function FunctionDefProvider(;
    weight_bucket=:function_def,
    base_capability_mask=(),
    base_witness=FiniteBaseWitness(),
    admissibility=AcceptFunctionDef(),
    max_nesting::Integer=2,
    body_attempts::Integer=4,
    _route_token::FunctionDefRouteToken=FunctionDefRouteToken(),
)
    bucket = weight_bucket isa Symbol ? CG.WeightBucket(weight_bucket) : weight_bucket
    bucket isa CG.CoreWeightBucket ||
        throw(ArgumentError("FunctionDef weight_bucket must be a symbol or CoreWeightBucket"))
    max_nesting >= 0 || throw(ArgumentError("max_nesting must be nonnegative"))
    body_attempts >= 0 || throw(ArgumentError("body_attempts must be nonnegative"))
    mask = _functiondef_capability_tuple((
        :function_def,
        :fixed_point,
        :opaque,
        base_capability_mask...,
    ))
    return FunctionDefProvider(
        bucket,
        mask,
        base_witness,
        admissibility,
        Int(max_nesting),
        Int(body_attempts),
        _route_token,
    )
end

function _functiondef_capability_tuple(capabilities)
    out = Symbol[]
    for capability in capabilities
        capability isa Symbol || throw(ArgumentError("FunctionDef capabilities must be symbols"))
        capability in out || push!(out, capability)
    end
    return Tuple(out)
end

CG.add_heads!(::FunctionDefProvider, heads, universe; kwargs...) = heads
CG.provider_capabilities(::FunctionDefProvider) =
    (:function_def, :fixed_point, :recursive_type_query)
CG.provides_generation_options(::FunctionDefProvider) = true

function CG.generation_provider_applicable(
    provider::FunctionDefProvider,
    relation,
    query::CG.GenerationQuery,
)
    _functiondef_remaining(query, provider) > 0 || return false
    provider.body_attempts > 0 || return false
    query.node_budget >= 2 || return false
    CG._bucket_enabled(relation.weights, provider.weight_bucket) || return false
    _functiondef_attempts_remaining(query, provider) > 0 || return false
    return CG.generation_function_view(query.target_ty) !== nothing
end

function CG.generation_provider_exhausted(
    provider::FunctionDefProvider,
    relation,
    query::CG.GenerationQuery,
)
    provider.max_nesting > 0 || return true
    provider.body_attempts > 0 || return true
    query.depth_left > 0 && query.provider_fuel > 0 || return true
    query.node_budget >= 2 || return true
    CG._bucket_enabled(relation.weights, provider.weight_bucket) || return true
    _functiondef_remaining(query, provider) > 0 || return true
    return _functiondef_attempts_remaining(query, provider) <= 0
end

function CG.generation_provider_cache_key(provider::FunctionDefProvider)
    Base.issingletontype(typeof(provider.base_witness)) || return nothing
    Base.issingletontype(typeof(provider.admissibility)) || return nothing
    return (
        :function_def_provider,
        provider.weight_bucket,
        provider.base_capability_mask,
        typeof(provider.base_witness),
        typeof(provider.admissibility),
        provider.max_nesting,
        provider.body_attempts,
    )
end

function CG.generation_root_requirements(
    provider::FunctionDefProvider,
    relation,
    query,
)
    provider.max_nesting > 0 || return ()
    provider.body_attempts > 0 || return ()
    query.provider_fuel > 0 || return ()
    CG._bucket_enabled(relation.weights, provider.weight_bucket) || return ()
    return (FunctionDefAttemptBudget(
        provider.route_token,
        fill(provider.body_attempts, provider.max_nesting),
    ),)
end

function _functiondef_remaining(query::CG.GenerationQuery, provider::FunctionDefProvider)
    remaining = min(provider.max_nesting, query.provider_fuel)
    for requirement in query.requirements
        requirement isa FunctionDefFuel || continue
        remaining = min(remaining, requirement.remaining)
    end
    return remaining
end

function _functiondef_attempt_budget(query::CG.GenerationQuery, provider::FunctionDefProvider)
    for requirement in query.requirements
        requirement isa FunctionDefAttemptBudget || continue
        requirement.route_token === provider.route_token && return requirement
    end
    return FunctionDefAttemptBudget(
        provider.route_token,
        fill(provider.body_attempts, provider.max_nesting),
    )
end

function _functiondef_attempts_remaining(
    query::CG.GenerationQuery,
    provider::FunctionDefProvider,
)
    level = _functiondef_remaining(query, provider)
    level > 0 || return 0
    budget = _functiondef_attempt_budget(query, provider)
    level <= length(budget.remaining_by_level) || return 0
    return budget.remaining_by_level[level]
end

function _consume_functiondef_attempt!(
    query::CG.GenerationQuery,
    provider::FunctionDefProvider,
)
    level = _functiondef_remaining(query, provider)
    level > 0 || return false
    budget = _functiondef_attempt_budget(query, provider)
    level <= length(budget.remaining_by_level) || return false
    budget.remaining_by_level[level] > 0 || return false
    budget.remaining_by_level[level] -= 1
    return true
end

function CG.add_generation_options!(
    provider::FunctionDefProvider,
    options,
    relation,
    query::CG.GenerationQuery,
)
    CG.generation_provider_applicable(provider, relation, query) || return options
    push!(
        options,
        (
            weight_bucket=provider.weight_bucket,
            target_ty=query.target_ty,
            base_depth_left=query.depth_left,
            base_node_budget=query.node_budget,
            remaining=_functiondef_remaining(query, provider),
        ),
    )
    return options
end

function _functiondef_base_witness(provider, relation, option, query)
    forbidden = Tuple(
        requirement isa FunctionDefSelfBinder ?
        CG.ForbidsScopeBinder(requirement.index) :
        requirement
        for requirement in query.requirements
        if requirement isa Union{FunctionDefSelfBinder,CG.ForbidsScopeBinder}
    )
    base_query = CG.GenerationQuery(
        option.target_ty;
        scope_tys=query.scope_tys,
        depth_left=option.base_depth_left,
        guarded_left=query.guarded_left,
        provider_fuel=query.provider_fuel,
        node_budget=option.base_node_budget,
        capability_mask=query.capability_mask,
        requirements=forbidden,
    )
    witness = provider.base_witness(
        relation,
        base_query,
        provider.base_capability_mask,
    )
    witness === nothing && return nothing
    witness isa CG.CoreDerivation || throw(ArgumentError(
        "FunctionDef base_witness must return CoreDerivation or nothing",
    ))
    CG.derivation_node_count(witness) <= option.base_node_budget || return nothing
    _valid_functiondef_base_type(provider.base_witness, witness, option.target_ty) ||
        return nothing
    return CG._copy_generation_witness(witness)
end

struct FunctionDefDerivation <: CG.CoreDerivation
    target_ty::Ty
    body::CG.CoreDerivation
    base_witness::CG.CoreDerivation
    self_index::Int
    capabilities::Tuple{Vararg{Symbol}}
    remaining::Int
end

_is_eta_forwarding_derivation(::FunctionDefDerivation, self_index::Int) = false
_contains_safe_list_access(derivation::FunctionDefDerivation) =
    _contains_safe_list_access(derivation.body)

function _has_unguarded_safe_list_recursive_derivation(
    derivation::FunctionDefDerivation,
    self_index::Int,
    safe_list_depth::Int=0,
)
    return _has_unguarded_safe_list_recursive_derivation(
        derivation.body,
        self_index + 1,
        safe_list_depth,
    )
end

function _functiondef_admissible(provider, candidate, relation, query)
    policy = provider.admissibility
    accepted = if applicable(policy, candidate, relation, query)
        policy(candidate, relation, query)
    elseif applicable(policy, candidate)
        policy(candidate)
    else
        throw(ArgumentError(
            "FunctionDef admissibility must accept candidate or " *
            "(candidate, relation, query)",
        ))
    end
    accepted isa Bool || throw(ArgumentError("FunctionDef admissibility must return Bool"))
    return accepted
end

_functiondef_sampling_admissible(::Any, ::FunctionDefCandidate) = true
_functiondef_sampling_admissible(::DirectKnotFilter, candidate::FunctionDefCandidate) =
    DirectKnotFilter()(candidate)
function _functiondef_sampling_admissible(
    filter::GenuineSelfCallFilter,
    candidate::FunctionDefCandidate,
)
    _functiondef_sampling_admissible(filter.inner, candidate) || return false
    return _derivation_has_self_call(candidate.body, candidate.self_index)
end

function CG.sample_generation_extension(
    provider::FunctionDefProvider,
    state,
    relation,
    option,
    query::CG.GenerationQuery,
)
    option.remaining > 0 || return nothing
    (option.target_ty === query.target_ty ||
        TypeSystem.iscompatible(option.target_ty, query.target_ty)) || return nothing
    _functiondef_attempts_remaining(query, provider) > 0 || return nothing
    base_witness = _functiondef_base_witness(provider, relation, option, query)
    base_witness === nothing && return nothing
    CG.consume_generation_nodes!(state, 1) || return nothing
    wrapper_nodes_left = state.nodes_left
    nested_fuel = FunctionDefFuel(option.remaining - 1)
    self_binder = FunctionDefSelfBinder(0)

    if !(provider.admissibility isa GenuineSelfCallFilter)
        _consume_functiondef_attempt!(query, provider) || return nothing
        body = CG.sample_generation_extension(
            state,
            relation,
            query;
            target_ty=query.target_ty,
            added_scope_tys=(query.target_ty,),
            node_budget=min(query.node_budget, state.nodes_left),
            requirements=(
                nested_fuel,
                _functiondef_attempt_budget(query, provider),
                self_binder,
            ),
            allow_safe_list_access=true,
        )
        if body === nothing
            state.nodes_left = wrapper_nodes_left
            return nothing
        end
        candidate = FunctionDefCandidate(query.target_ty, body, base_witness, 0)
        if !_functiondef_admissible(provider, candidate, relation, query)
            state.nodes_left = wrapper_nodes_left
            return nothing
        end
        return FunctionDefDerivation(
            query.target_ty,
            body,
            base_witness,
            0,
            CG.provider_capabilities(provider),
            nested_fuel.remaining,
        )
    end

    for _ in 1:provider.body_attempts
        _consume_functiondef_attempt!(query, provider) || break
        state.nodes_left = wrapper_nodes_left
        accept_body = body -> _functiondef_sampling_admissible(
            provider.admissibility,
            FunctionDefCandidate(query.target_ty, body, base_witness, 0),
        )
        body = CG.sample_generation_derivation_using_scope_binder(
            state,
            relation,
            query,
            query.target_ty;
            target_ty=query.target_ty,
            node_budget=min(query.node_budget, state.nodes_left),
            requirements=(nested_fuel, _functiondef_attempt_budget(query, provider), self_binder),
            allow_safe_list_access=true,
            allow_direct_result=false,
            require_scope_application=true,
            accept_derivation=accept_body,
        )
        body === nothing && continue
        CG.derivation_uses_scope_binder(body, 0) || continue
        candidate = FunctionDefCandidate(query.target_ty, body, base_witness, 0)
        _functiondef_admissible(provider, candidate, relation, query) || continue
        return FunctionDefDerivation(
            query.target_ty,
            body,
            base_witness,
            0,
            CG.provider_capabilities(provider),
            nested_fuel.remaining,
        )
    end
    state.nodes_left = wrapper_nodes_left
    return nothing
end

CG.derivation_node_count(derivation::FunctionDefDerivation) =
    1 + CG.derivation_node_count(derivation.body)
CG.derivation_result_type(derivation::FunctionDefDerivation) = derivation.target_ty
CG.derivation_signature(derivation::FunctionDefDerivation) = (
    :function_def,
    TypeSystem.typekey(derivation.target_ty),
    derivation.self_index,
    CG.derivation_signature(derivation.body),
    CG.derivation_signature(derivation.base_witness),
)
CG.derivation_uses_scope_binder(
    derivation::FunctionDefDerivation,
    index::Integer,
    offset::Integer=0,
) = CG.derivation_uses_scope_binder(derivation.body, index, offset + 1)
CG.rigid_safe_derivation(derivation::FunctionDefDerivation) =
    CG.rigid_safe_derivation(derivation.body) &&
    CG.rigid_safe_derivation(derivation.base_witness)

function CG.derivation_metadata(derivation::FunctionDefDerivation)
    metadata = deepcopy(CG.derivation_metadata(derivation.body))
    nested = get(metadata, :function_defs, Any[])
    nested isa AbstractVector || (nested = Any[nested])
    entry = Dict{Symbol,Any}(
        :target => TypeSystem.typekey(derivation.target_ty),
        :self_index => derivation.self_index,
        :self_binder => :debruijn,
        :base_signature => CG.derivation_signature(derivation.base_witness),
        :body_signature => CG.derivation_signature(derivation.body),
        :provider_capabilities => derivation.capabilities,
        :used_capabilities => _used_functiondef_capabilities(derivation),
        :nesting_remaining => derivation.remaining,
    )
    metadata[:function_defs] = Any[entry; nested...]
    return metadata
end

function _used_functiondef_capabilities(derivation::FunctionDefDerivation)
    used = Symbol[:function_def, :fixed_point]
    CG.generation_function_view(derivation.target_ty) !== derivation.target_ty &&
        push!(used, :recursive_type_query)
    for capability in CG.derivation_capabilities(derivation.body)
        capability in used || push!(used, capability)
    end
    return Tuple(used)
end

CG.derivation_capabilities(derivation::FunctionDefDerivation) =
    _used_functiondef_capabilities(derivation)

CG._guarded_depth(derivation::FunctionDefDerivation) = CG._guarded_depth(derivation.body)
CG._array_source_depth(::FunctionDefDerivation) = typemax(Int)

function CG.materialize_generation_derivation(
    materializer::CG.DefaultAstMaterializer,
    derivation::FunctionDefDerivation,
    local_consts,
    local_const_types,
    relation,
    rng,
)
    body = CG.materialize_generation_derivation(
        materializer,
        derivation.body,
        local_consts,
        local_const_types,
        relation,
        rng,
    )
    return FunctionDefNode(body)
end

"""Post-generation failure when public procedural inference disagrees with the demand."""
struct ProceduralGenerationVerificationError
    expected::TyLike
    inferred
end

function _generation_config_from_kwargs(kwargs)
    rng = get(kwargs, :rng, Random.default_rng())
    weights = get(kwargs, :weights, CG.GenerationWeights())
    providers = get(kwargs, :providers, CG.default_head_providers())
    guarded = get(kwargs, :guarded_expansion_depth, 1)
    depth = get(kwargs, :sampling_depth_limit, 10)
    nodes = get(kwargs, :sampling_node_budget, 128)
    return CG.GenerationConfig(;
        rng,
        weights,
        providers,
        guarded_expansion_depth=guarded,
        sampling_depth_limit=depth,
        sampling_node_budget=nodes,
    )
end

struct _DefaultFunctionDefProvider end
const _DEFAULT_FUNCTIONDEF_PROVIDER = _DefaultFunctionDefProvider()

function _with_functiondef_provider(config::CG.GenerationConfig, selection)
    existing = Tuple(provider for provider in config.providers if provider isa FunctionDefProvider)
    retained = Tuple(provider for provider in config.providers if !(provider isa FunctionDefProvider))
    providers = if selection isa _DefaultFunctionDefProvider
        isempty(existing) ? (retained..., FunctionDefProvider()) : config.providers
    elseif selection === nothing
        retained
    else
        selection isa FunctionDefProvider ||
            throw(ArgumentError("functiondef_provider must be FunctionDefProvider or nothing"))
        (retained..., selection)
    end
    providers === config.providers && return config
    return CG.GenerationConfig(;
        rng=config.rng,
        weights=config.weights,
        providers,
        argument_planner=config.argument_planner,
        sampler=config.sampler,
        materializer=config.materializer,
        guarded_expansion_depth=config.guarded_expansion_depth,
        sampling_depth_limit=config.sampling_depth_limit,
        sampling_node_budget=config.sampling_node_budget,
    )
end

function _require_genuine_self_call(provider::FunctionDefProvider)
    provider.admissibility isa GenuineSelfCallFilter && return provider
    return FunctionDefProvider(
        weight_bucket=provider.weight_bucket,
        base_capability_mask=provider.base_capability_mask,
        base_witness=provider.base_witness,
        admissibility=GenuineSelfCallFilter(provider.admissibility),
        max_nesting=provider.max_nesting,
        body_attempts=provider.body_attempts,
        _route_token=provider.route_token,
    )
end

function _verification_context(result, kwargs)
    context = Dict{Symbol,TyLike}(result.local_const_types)
    for key in (:global_consts_ty, :local_consts_ty, :preset_types)
        source = get(kwargs, key, nothing)
        source === nothing && continue
        for (name, ty) in pairs(source)
            ty isa TyLike && (context[name] = ty)
        end
    end
    return context
end

function _close_scoped_verification(tree::AstTree, target_ty::TyLike, scope_tys)
    root = tree.root
    expected = target_ty isa TyScheme ? target_ty.ty : target_ty
    for (index, scope_ty) in enumerate(scope_tys)
        root = AbsNode(Symbol("##verify_scope", index), root)
        expected = TyFun(scope_ty, expected)
    end
    if target_ty isa TyScheme
        expected = TyScheme(target_ty.vars, expected)
    end
    return AstTree(root), expected
end

function _verify_procedural_generation(
    result,
    target_ty;
    fixed_point,
    kwargs,
    scope_tys=Ty[],
)
    result isa CG.GenerationResult || return result
    result.metadata[:procedural_generation] = true
    result.metadata[:functiondef_observations] =
        functiondef_recursion_observations(result.tree)
    if !_contains_procedural_node(result.tree.root)
        result.metadata[:procedural_verification] = :not_needed
        return result
    end
    verification_tree, expected = _close_scoped_verification(result.tree, target_ty, scope_tys)
    inferred = infer_procedural(
        verification_tree,
        _verification_context(result, kwargs);
        fixed_point,
    )
    inferred isa Err && return inferred
    TypeSystem.iscompatible(inferred, expected) ||
        return Err(ProceduralGenerationVerificationError(expected, inferred))
    result.metadata[:procedural_verification] = :verified
    return result
end

function _contains_procedural_node(node::AstNode)
    node isa Union{ScopeNode,FunctionDefNode} && return true
    return any(_contains_procedural_node, children(node))
end

"""Generate with an opt-in guarded FunctionDef provider and public post-verification."""
function generate_procedural(
    target_ty::TyLike;
    functiondef_provider=_DEFAULT_FUNCTIONDEF_PROVIDER,
    fixed_point=fixed_point_witness,
    config=nothing,
    kwargs...,
)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    effective = _with_functiondef_provider(base_config, functiondef_provider)
    _validate_functiondef_sampling_config(effective)
    result = CG.generate_canonical(target_ty; kwargs..., config=effective)
    return _verify_procedural_generation(result, target_ty; fixed_point, kwargs)
end

"""Scoped companion to [`generate_procedural`](@ref); scope is inner to outer."""
function generate_procedural_scoped(
    target_ty::TyLike;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    functiondef_provider=_DEFAULT_FUNCTIONDEF_PROVIDER,
    fixed_point=fixed_point_witness,
    config=nothing,
    kwargs...,
)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    effective = _with_functiondef_provider(base_config, functiondef_provider)
    _validate_functiondef_sampling_config(effective)
    result = CG.generate_canonical_scoped(
        target_ty;
        scope_tys,
        kwargs...,
        config=effective,
    )
    return _verify_procedural_generation(result, target_ty; fixed_point, kwargs, scope_tys)
end

"""
    generate_finite_procedural_scoped(target_ty::Ty; scope_tys=Ty[], kwargs...)

Return one finite core witness while retaining equirecursive target and scope
support. The witness query masks `:function_def`, `:fixed_point`, and `:opaque`
capabilities transitively, so this entry can observe a finite `μ` graph without
constructing a recursive definition. It is primarily useful as the finite side
of FunctionDef removal and body replacement.
"""
function generate_finite_procedural_scoped(
    target_ty::Ty;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    fixed_point=fixed_point_witness,
    config=nothing,
    kwargs...,
)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    effective = _with_functiondef_provider(base_config, nothing)
    scope = Ty[scope_tys...]

    local_consts = get(kwargs, :local_consts, Dict{Symbol,Any}())
    local_consts_ty = get(kwargs, :local_consts_ty, Dict{Symbol,Any}())
    result_local_consts = CG._anydict(local_consts)
    result_local_const_types = CG._typed_context(local_consts, local_consts_ty)
    source_local_consts = get(kwargs, :source_local_consts, true)
    relation_local_consts = source_local_consts ? result_local_consts : Dict{Symbol,Any}()
    relation_local_const_types = source_local_consts ?
        result_local_const_types : Dict{Symbol,TyLike}()
    relation = CG.CoreRelation(
        global_consts=get(kwargs, :global_consts, Dict{Symbol,Any}()),
        global_consts_ty=get(kwargs, :global_consts_ty, Dict{Symbol,Any}()),
        local_consts=relation_local_consts,
        local_consts_ty=relation_local_const_types,
        presets=get(kwargs, :presets, Dict{Symbol,Any}()),
        preset_types=get(kwargs, :preset_types, Dict{Symbol,Any}()),
        weights=effective.weights,
        providers=effective.providers,
        argument_planner=effective.argument_planner,
        guarded_expansion_depth=effective.guarded_expansion_depth,
    )
    query = CG.GenerationQuery(
        target_ty;
        scope_tys=scope,
        depth_left=effective.sampling_depth_limit,
        guarded_left=effective.guarded_expansion_depth,
        provider_fuel=effective.sampling_depth_limit,
        node_budget=effective.sampling_node_budget,
        capability_mask=relation.capability_mask,
    )
    witness = CG.known_generation_witness(
        relation,
        query;
        capability_mask=(:function_def, :fixed_point, :opaque),
    )
    witness === nothing && return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))
    node = CG.materialize_generation_derivation(
        effective.materializer,
        witness,
        result_local_consts,
        result_local_const_types,
        relation,
        effective.rng,
    )
    result = CG.GenerationResult(
        AstTree(node),
        target_ty,
        result_local_consts,
        result_local_const_types,
        Dict{Symbol,Any}(
            :generator => :canonical_finite_witness,
            :rng_type => typeof(effective.rng),
            :sampling_nodes_used => CG.derivation_node_count(witness),
            :scope_tys => Tuple(TypeSystem.typekey.(scope)),
            :derivation => CG.derivation_signature(witness),
            :forced_finite_witness => true,
        ),
    )
    return _verify_procedural_generation(
        result,
        target_ty;
        fixed_point,
        kwargs,
        scope_tys=scope,
    )
end

"""
    generate_recursive_functiondef_scoped(target_ty::Ty; scope_tys=Ty[], kwargs...)

Generate one guarded, genuinely self-using [`FunctionDefNode`](@ref) at the
root. Unlike [`generate_procedural_scoped`](@ref), this entry does not route
between ordinary core terms and FunctionDef: selecting it explicitly forces the
existing [`FunctionDefProvider`](@ref) protocol, including its finite base
witness, nesting fuel, shared attempt budget, and admissibility check.

`scope_tys` remains inner → outer. The provider and all ordinary body-generation
configuration are injectable; caller-owned RNG and constant containers are
forwarded unchanged.
"""
function generate_recursive_functiondef_scoped(
    target_ty::Ty;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    functiondef_provider::FunctionDefProvider=FunctionDefProvider(),
    fixed_point=fixed_point_witness,
    config=nothing,
    kwargs...,
)
    functiondef_provider = _require_genuine_self_call(functiondef_provider)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    effective = _with_functiondef_provider(base_config, functiondef_provider)
    _validate_functiondef_sampling_config(effective)
    CG._generation_target_supported(target_ty, effective) ||
        return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))

    scope = Ty[scope_tys...]
    CG._generation_scope_supported(scope, effective) ||
        return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))

    local_consts = get(kwargs, :local_consts, Dict{Symbol,Any}())
    local_consts_ty = get(kwargs, :local_consts_ty, Dict{Symbol,Any}())
    result_local_consts = CG._anydict(local_consts)
    result_local_const_types = CG._typed_context(local_consts, local_consts_ty)
    source_local_consts = get(kwargs, :source_local_consts, true)
    relation_local_consts = source_local_consts ? result_local_consts : Dict{Symbol,Any}()
    relation_local_const_types = source_local_consts ?
        result_local_const_types : Dict{Symbol,TyLike}()
    global_consts = get(kwargs, :global_consts, Dict{Symbol,Any}())
    global_consts_ty = get(kwargs, :global_consts_ty, Dict{Symbol,Any}())
    presets = get(kwargs, :presets, Dict{Symbol,Any}())
    preset_types = get(kwargs, :preset_types, Dict{Symbol,Any}())
    cache_key = CG._uses_lazy_sampling(effective) ? CG._scoped_lazy_relation_cache_key(
        effective.weights,
        global_consts,
        global_consts_ty,
        relation_local_consts,
        relation_local_const_types,
        presets,
        preset_types,
        effective.providers,
        effective.argument_planner,
        effective.guarded_expansion_depth,
    ) : nothing
    if cache_key !== nothing
        entry = CG._get_scoped_lazy_relation_cache_entry(
            cache_key,
            effective.weights,
            effective.providers,
            effective.argument_planner,
            effective.guarded_expansion_depth,
            global_consts,
            global_consts_ty,
            relation_local_consts,
            relation_local_const_types,
        )
        lock(entry.lock)
        try
            entry.relation.local_consts = relation_local_consts
            entry.relation.local_consts_ty = relation_local_const_types
            return _generate_forced_functiondef_result(
                entry.relation,
                effective,
                functiondef_provider,
                target_ty,
                scope,
                result_local_consts,
                result_local_const_types,
                fixed_point,
                kwargs,
            )
        finally
            unlock(entry.lock)
        end
    end

    relation = CG.CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts=relation_local_consts,
        local_consts_ty=relation_local_const_types,
        presets=presets,
        preset_types=preset_types,
        weights=effective.weights,
        providers=effective.providers,
        argument_planner=effective.argument_planner,
        guarded_expansion_depth=effective.guarded_expansion_depth,
    )
    return _generate_forced_functiondef_result(
        relation,
        effective,
        functiondef_provider,
        target_ty,
        scope,
        result_local_consts,
        result_local_const_types,
        fixed_point,
        kwargs,
    )
end

function _generate_forced_functiondef_result(
    relation,
    effective,
    functiondef_provider,
    target_ty,
    scope,
    result_local_consts,
    result_local_const_types,
    fixed_point,
    kwargs,
)
    state = CG.DerivationSamplingState(effective.rng, effective.sampling_node_budget)
    query = CG.GenerationQuery(
        target_ty;
        scope_tys=scope,
        depth_left=effective.sampling_depth_limit,
        guarded_left=effective.guarded_expansion_depth,
        provider_fuel=effective.sampling_depth_limit,
        node_budget=state.nodes_left,
        capability_mask=relation.capability_mask,
    )
    root_requirements = CG._root_generation_requirements(relation, query)
    isempty(root_requirements) ||
        (query = CG._derive_generation_query(query; requirements=root_requirements))

    provider_options = NamedTuple[]
    CG.add_generation_options!(
        functiondef_provider,
        provider_options,
        relation,
        query,
    )
    wrapped_options = NamedTuple[
        (
            kind=:generation_extension,
            value=(provider=functiondef_provider, option=option),
            weight=1.0,
        ) for option in provider_options
    ]
    derivation = CG._sample_from_options(
        state,
        relation,
        wrapped_options,
        scope,
        query.depth_left,
        query.guarded_left,
        true,
        query,
    )
    derivation === nothing && return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))
    derivation isa FunctionDefDerivation || throw(ArgumentError(
        "FunctionDefProvider must materialize FunctionDefDerivation at the forced root",
    ))

    node = CG.materialize_generation_derivation(
        effective.materializer,
        derivation,
        result_local_consts,
        result_local_const_types,
        relation,
        effective.rng,
    )
    metadata = Dict{Symbol,Any}(
        :generator => :canonical_finite_core,
        :rng_type => typeof(effective.rng),
        :sampling_nodes_used => effective.sampling_node_budget - state.nodes_left,
        :scope_tys => Tuple(TypeSystem.typekey.(scope)),
        :derivation => CG.derivation_signature(derivation),
        :forced_functiondef_root => true,
    )
    merge!(metadata, CG.derivation_metadata(derivation))
    result = CG.GenerationResult(
        AstTree(node),
        target_ty,
        result_local_consts,
        result_local_const_types,
        metadata,
    )
    return _verify_procedural_generation(
        result,
        target_ty;
        fixed_point,
        kwargs,
        scope_tys=scope,
    )
end

function _validate_functiondef_sampling_config(config::CG.GenerationConfig)
    any(provider -> provider isa FunctionDefProvider, config.providers) || return config
    config.sampler isa CG.WeightedDerivationSampler || throw(ArgumentError(
        "FunctionDef generation currently requires WeightedDerivationSampler",
    ))
    config.argument_planner isa CG.DefaultArgumentPlanner || throw(ArgumentError(
        "FunctionDef generation currently requires DefaultArgumentPlanner",
    ))
    return config
end
