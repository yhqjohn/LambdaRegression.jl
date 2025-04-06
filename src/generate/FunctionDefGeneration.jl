const CG = @__MODULE__

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
`λx. self x`, an unconditional recursive call with every function argument
unchanged in a strict expression context, or a recursive call whose argument
hides a safe-list case outside its success guard. Nested lambda/FunctionDef
values are not assumed to execute. A self call in a safe-list fallback is also
rejected because the
fallback is selected only after the source is empty and cannot establish a
structurally smaller recursive argument. The filter is deliberately syntactic
and total; richer loop analysis and runtime limits remain separate injectable
policies.
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
    _is_unconditional_self_call_derivation(body, candidate.self_index) && return false
    _has_unchanged_self_call_derivation(body, candidate.self_index) && return false
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
    CG._signature_calls_scope_binder(CG.derivation_signature(derivation), self_index)

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

function _is_unconditional_self_call_derivation(
    derivation::CG.CoreDerivation,
    self_index::Int,
)
    current = derivation
    lambda_count = 0
    while current isa CG.LambdaDerivation
        lambda_count += 1
        current = current.body
    end
    lambda_count > 0 || return false
    if current isa CG.ScopeApplicationDerivation
        return current.index == self_index + lambda_count && !isempty(current.args)
    end
    return _is_unconditional_self_call_signature(
        CG.derivation_signature(derivation),
        self_index,
    )
end

function _has_unchanged_self_call_derivation(
    derivation::CG.CoreDerivation,
    self_index::Int,
)
    current = derivation
    lambda_count = 0
    while current isa CG.LambdaDerivation
        lambda_count += 1
        current = current.body
    end
    lambda_count > 0 || return false
    parameter_indices = collect((lambda_count - 1):-1:0)
    return _has_unchanged_self_call_derivation(
        current,
        self_index + lambda_count,
        parameter_indices,
    )
end

_has_unchanged_self_call_derivation(
    ::Union{CG.VariableDerivation,CG.ConstantDerivation},
    self_index::Int,
    parameter_indices::Vector{Int},
) = false

function _has_unchanged_self_call_derivation(
    ::CG.LambdaDerivation,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    return false
end

function _has_unchanged_self_call_derivation(
    derivation::CG.ScopeApplicationDerivation,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    if derivation.index == self_index &&
       length(derivation.args) == length(parameter_indices) &&
       all(zip(derivation.args, parameter_indices)) do (argument, expected)
           argument isa CG.VariableDerivation && argument.index == expected
       end
        return true
    end
    return any(derivation.args) do argument
        _has_unchanged_self_call_derivation(
            argument,
            self_index,
            parameter_indices,
        )
    end
end

function _has_unchanged_self_call_derivation(
    derivation::CG.HeadApplicationDerivation,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    return any(derivation.args) do argument
        _has_unchanged_self_call_derivation(
            argument,
            self_index,
            parameter_indices,
        )
    end
end

function _has_unchanged_self_call_derivation(
    derivation::CG.SafeListAccessDerivation,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    return _has_unchanged_self_call_derivation(
        derivation.source,
        self_index,
        parameter_indices,
    ) || _has_unchanged_self_call_derivation(
        derivation.body,
        self_index + 2,
        parameter_indices .+ 2,
    ) || _has_unchanged_self_call_derivation(
        derivation.fallback,
        self_index,
        parameter_indices,
    )
end

function _has_unchanged_self_call_derivation(
    derivation::CG.CoreDerivation,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    return _has_unchanged_self_call_signature(
        CG.derivation_signature(derivation),
        self_index,
        parameter_indices,
    )
end

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
    ) || _derivation_has_self_call(
        derivation.fallback,
        self_index,
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

Base.@noinline Base.@nospecializeinfer function _has_unguarded_safe_list_recursive_argument(
    signature,
    self_index::Int,
    safe_list_depth::Int=0,
)
    Base.@nospecialize signature
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
           CG._scope_any(argument -> CG._signature_contains(argument, :safe_list_access), arguments)
            return true
        end
        return CG._scope_any(
            argument -> _has_unguarded_safe_list_recursive_argument(
                argument,
                self_index,
                safe_list_depth,
            ),
            arguments,
        )
    elseif tag === :head
        length(signature) >= 5 || return false
        return CG._scope_any(
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
        ) || CG._signature_calls_scope_binder(
            signature[7],
            self_index,
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

Base.@noinline Base.@nospecializeinfer function _has_unchanged_self_call_signature(
    signature,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    Base.@nospecialize signature
    signature isa Tuple || return false
    isempty(signature) && return false
    tag = signature[1]
    if tag === :lambda
        return false
    elseif tag === :scope_application
        length(signature) >= 4 || return false
        arguments = signature[4]
        if signature[2] == self_index &&
           length(arguments) == length(parameter_indices)
            unchanged = if arguments isa Tuple
                matches = true
                for position in 1:length(arguments)
                    argument = arguments[position]
                    if !(argument isa Tuple && length(argument) >= 2 &&
                         argument[1] === :var && argument[2] == parameter_indices[position])
                        matches = false
                        break
                    end
                end
                matches
            else
                all(zip(arguments, parameter_indices)) do (argument, expected)
                    argument isa Tuple && length(argument) >= 2 &&
                        argument[1] === :var && argument[2] == expected
                end
            end
            unchanged && return true
        end
        return CG._scope_any(
            argument -> _has_unchanged_self_call_signature(
                argument,
                self_index,
                parameter_indices,
            ),
            arguments,
        )
    elseif tag === :head
        length(signature) >= 5 || return false
        return CG._scope_any(
            argument -> _has_unchanged_self_call_signature(
                argument,
                self_index,
                parameter_indices,
            ),
            signature[5],
        )
    elseif tag === :safe_list_access
        length(signature) >= 7 || return false
        return _has_unchanged_self_call_signature(
            signature[5],
            self_index,
            parameter_indices,
        ) || _has_unchanged_self_call_signature(
            signature[6],
            self_index + 2,
            parameter_indices .+ 2,
        ) || _has_unchanged_self_call_signature(
            signature[7],
            self_index,
            parameter_indices,
        )
    elseif tag === :function_def
        return false
    end
    return false
end

Base.@noinline Base.@nospecializeinfer function _is_eta_forwarding_signature(signature, self_index::Int)
    Base.@nospecialize signature
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
    for position in 1:lambda_count
        argument = arguments[position]
        expected = lambda_count - position
        argument isa Tuple && length(argument) >= 2 &&
            argument[1] === :var && argument[2] == expected || return false
    end
    return true
end

Base.@noinline Base.@nospecializeinfer function _is_unconditional_self_call_signature(
    signature,
    self_index::Int,
)
    Base.@nospecialize signature
    current = signature
    lambda_count = 0
    while current isa Tuple && length(current) >= 3 && current[1] === :lambda
        lambda_count += 1
        current = current[3]
    end
    lambda_count > 0 || return false
    return current isa Tuple && length(current) >= 4 &&
        current[1] === :scope_application &&
        current[2] == self_index + lambda_count &&
        !isempty(current[4])
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
    leading_parameters::Int
    prefix_open::Bool
    function FunctionDefSelfBinder(index::Integer, leading_parameters::Integer=0,
        prefix_open::Bool=true)
        index >= 0 || throw(ArgumentError("FunctionDef self binder index must be nonnegative"))
        leading_parameters >= 0 || throw(ArgumentError(
            "FunctionDef leading parameter count must be nonnegative"))
        leading_parameters <= index || throw(ArgumentError(
            "FunctionDef leading parameters cannot exceed self index"))
        return new(Int(index), Int(leading_parameters), prefix_open)
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
    FunctionDefSelfBinder(marker.index + CG._checked_nonnegative_int(:added_binders, added),
        marker.leading_parameters, marker.prefix_open)
CG.generation_requirement_affects_core_search(::FunctionDefFuel) = false
CG.generation_requirement_affects_core_search(::FunctionDefAttemptBudget) = false
CG.generation_requirement_affects_core_search(::FunctionDefSelfBinder) = false
CG.generation_requirement_cache_key(fuel::FunctionDefFuel) =
    (:function_def_fuel, fuel.remaining)
CG.generation_requirement_cache_key(budget::FunctionDefAttemptBudget) =
    # Menus include extension admission, which depends on mutable attempt
    # balances. Cache the current immutable snapshot, never the ledger itself.
    (:function_def_attempt_budget, budget.route_token.id, Tuple(budget.remaining_by_level))
CG.generation_requirement_cache_key(marker::FunctionDefSelfBinder) =
    (:function_def_self_binder, marker.index, marker.leading_parameters, marker.prefix_open)

"""Record one confirmed *leading* lambda parameter of the marked FunctionDef.

Only a constructor which actually chooses a lambda in the still-open prefix may
call this. A lambda nested later in an argument is not an original parameter.
"""
function _advance_functiondef_prefix(query::CG.GenerationQuery)
    return _map_functiondef_markers(query) do marker
        marker.prefix_open || return marker
        FunctionDefSelfBinder(marker.index, marker.leading_parameters + 1, true)
    end
end

"""Close the leading-lambda prefix before entering a non-lambda constructor."""
function _close_functiondef_prefix(query::CG.GenerationQuery)
    return _map_functiondef_markers(query) do marker
        marker.prefix_open || return marker
        FunctionDefSelfBinder(marker.index, marker.leading_parameters, false)
    end
end

function _map_functiondef_markers(transform, query::CG.GenerationQuery)
    any(r -> r isa FunctionDefSelfBinder, query.requirements) || return query
    mapped = map(query.requirements) do requirement
        requirement isa FunctionDefSelfBinder ? transform(requirement) : requirement
    end
    all(isequal(old, new) for (old, new) in zip(query.requirements, mapped)) && return query
    return CG.GenerationQuery(query.target_ty, query.scope_tys, query.depth_left,
        query.guarded_left, query.provider_fuel, query.node_budget,
        query.capability_mask, mapped)
end

function _required_full_self_marker(query::CG.GenerationQuery,
    head_index::Int, arity::Int)
    marker = findfirst(query.requirements) do requirement
        requirement isa FunctionDefSelfBinder && !requirement.prefix_open &&
            requirement.index == head_index &&
            requirement.leading_parameters == arity
    end
    marker === nothing && return nothing
    self = query.requirements[marker]::FunctionDefSelfBinder
    any(r -> r isa CG.CallsScopeBinder && r.index == self.index,
        query.requirements) || return nothing
    return self
end

"""Sample one complete, proof-backed progress call for a pure Call child.

The chosen scoped head and every child have finite evidence before the shared
node balance is charged. One argument root excludes its original leading
parameter; other child facts keep their true cheaper costs. Certified minima
reserve the entire packet, while ordinary child sampling explores proven steps
within each leased budget; the minimum certificate is not forced as output.
The scoped head's node charge equals its applied arity, matching canonical
derivation cost.
The old joint obligation route remains selectable outside this finite core.
This is deliberately not a negative result for queries with additional use or
call obligations, custom argument policies, or mixed-only children.
"""
function _sample_preplanned_functiondef_call!(state::CG.DerivationSamplingState,
    relation::CG.CoreRelation, query::CG.GenerationQuery)
    state.completion.probability == 0 && !state.completion.completing || return nothing
    scope = CG._scope_type_vector(query.scope_tys)
    positive = CG._required_scope_binder_indices(query, length(scope))
    length(positive) == 1 || return nothing
    self_index = only(positive)
    forbidden = CG._forbidden_scope_binders(query)
    self_index in forbidden && return nothing
    CG._finite_evidence_request_supported(relation, query) || return nothing
    relation.argument_planner isa CG.DefaultArgumentPlanner || return nothing
    evidence_weight = get(relation.weights.overrides, :constraint_evidence, 1.0)
    evidence_weight > 0 &&
        rand(state.rng) * (1.0 + evidence_weight) < evidence_weight || return nothing

    options = NamedTuple[]
    for rule in CG._scope_application_rules(query.target_ty, scope, query.depth_left)
        rule.index == self_index || continue
        CG._known_type_admitted(relation, rule.head_ty) || continue
        self = _required_full_self_marker(query, rule.index, length(rule.argument_tys))
        self === nothing && continue
        weight = CG._scope_application_option_weight(relation, rule)
        weight > 0 && push!(options, (; rule, self, weight))
    end
    pool, total = CG._positive_option_pool(options)
    isempty(pool) && return nothing
    chosen = pool[CG._weighted_option_index(state.rng, pool, total)]
    rule, self = chosen.rule, chosen.self
    isempty(rule.argument_tys) && return nothing # progress needs a changed argument
    head_cost = length(rule.argument_tys)
    changed = rand(state.rng, eachindex(rule.argument_tys))
    available = min(state.nodes_left, query.node_budget)
    available > head_cost || return nothing
    child_depth = query.depth_left - 1
    # Prepare every child before construction, so their proven minimum costs
    # reserve a complete packet. A certificate is a completion guarantee, not
    # the ordinary exploration distribution of its descendants.
    children = Any[]
    lower_bounds = Int[]
    max_child_budget = available - head_cost
    max_child_budget >= 1 || return nothing
    for i in eachindex(rule.argument_tys)
        if i == changed
            prepared = CG._prepare_finite_core_nonbare_certificate(relation,
                rule.argument_tys[i], scope, child_depth, max_child_budget,
                self.index - i; forbidden)
            prepared isa CG._UnsupportedEvidenceSampling && return nothing
            prepared.status === :ready || return nothing
            push!(children, prepared)
            push!(lower_bounds, minimum(option.cost for option in prepared.options))
        else
            prepared = CG._prepare_finite_core_certificate(relation,
                rule.argument_tys[i], scope, child_depth, max_child_budget; forbidden)
            prepared isa CG._UnsupportedEvidenceSampling && return nothing
            prepared === nothing && return nothing
            child_cost = prepared.world.costs[prepared.fact]
            child_cost <= max_child_budget || return nothing
            push!(children, prepared)
            push!(lower_bounds, child_cost)
        end
    end
    head_cost + sum(lower_bounds) <= available || return nothing
    changed_budget = available - head_cost - sum(lower_bounds) + lower_bounds[changed]
    selection_state = CG.DerivationSamplingState(state.rng, changed_budget,
        state.completion)
    selected = CG._select_finite_core_nonbare_root(selection_state, children[changed])
    selected === nothing && return nothing
    lower_bounds[changed] = selected.cost
    args = CG.CoreDerivation[]
    cost = head_cost
    for (i, prepared) in enumerate(children)
        reserved = sum(lower_bounds[j] for j in (i + 1):length(children); init=0)
        child_budget = available - cost - reserved
        child_budget >= lower_bounds[i] || return nothing
        child_state = CG.DerivationSamplingState(state.rng, child_budget,
            state.completion)
        argument = if i == changed
            CG._sample_evidence_plan!(child_state, prepared.world, relation,
                selected.plan, selected.binder, prepared.scope,
                prepared.forbidden)
        else
            CG._sample_evidence_fact!(child_state, prepared.world, relation,
                prepared.fact, scope, forbidden; certificate_ready=true)
        end
        argument isa CG._UnsupportedEvidenceSampling && return nothing
        argument === nothing && return nothing
        CG._required_source_argument_admitted(rule, i, argument) || return nothing
        push!(args, argument)
        charged = child_budget - child_state.nodes_left
        CG.derivation_node_count(argument) == charged ||
            error("finite self-call child charge does not match its derivation")
        cost += charged
    end
    result = CG.ScopeApplicationDerivation(rule.index, rule.head_ty, args)
    CG.derivation_node_count(result) == cost ||
        error("preplanned self-call certificate cost does not match its derivation")
    CG._generation_requirements_satisfied(result, query) ||
        error("preplanned self call lost its occurrence obligation")
    CG._consume_sampling_nodes!(state, cost) || return nothing
    return CG._record_required_scope_route!(result, :preplanned_self_call)
end

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

# This provider contributes query options, not target/scope-sensitive core heads.
CG.generation_catalog_scope_invariant(::FunctionDefProvider) = true
CG._evidence_catalog_domain_sharing(::FunctionDefProvider) = true
CG._evidence_heads_policy_free(::FunctionDefProvider) = true
CG._evidence_early_certificate_supported(::FunctionDefProvider) = true
CG._may_supply_guarded_heads(::FunctionDefProvider) = false

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
    body_parent = _close_functiondef_prefix(query)

    if !(provider.admissibility isa GenuineSelfCallFilter)
        _consume_functiondef_attempt!(query, provider) || return nothing
        body = CG.sample_generation_extension(
            state,
            relation,
            body_parent;
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
            body_parent,
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
    metadata = CG._copy_derivation_metadata(CG.derivation_metadata(derivation.body))
    nested = get(metadata, :function_defs, Any[])
    nested isa AbstractVector || (nested = Any[nested])
    entry = Dict{Symbol,Any}(
        Pair{Symbol,Any}(:target, TypeSystem.typekey(derivation.target_ty)),
        :self_index => derivation.self_index,
        :self_binder => :debruijn,
        Pair{Symbol,Any}(:base_signature, CG.derivation_signature(derivation.base_witness)),
        Pair{Symbol,Any}(:body_signature, CG.derivation_signature(derivation.body)),
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

"""Post-generation failure when public type inference disagrees with the demand."""
struct FunctionDefGenerationVerificationError
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

function with_functiondef_provider(
    config::CG.GenerationConfig,
    selection=_DEFAULT_FUNCTIONDEF_PROVIDER,
)
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
    empty_values = Dict{Symbol,Any}()
    empty_types = Dict{Symbol,Any}()
    global_types = CG._typed_context(
        get(kwargs, :global_consts, empty_values),
        get(kwargs, :global_consts_ty, empty_types),
    )
    local_types = CG._typed_context(
        get(kwargs, :local_consts, empty_values),
        get(kwargs, :local_consts_ty, empty_types),
    )
    preset_types = CG._typed_presets(
        get(kwargs, :presets, empty_values),
        get(kwargs, :preset_types, empty_types);
        global_consts_ty=global_types,
        local_consts_ty=local_types,
    )
    context = merge(global_types, local_types, preset_types)
    # Explicit declarations retain their precedence over inferred contexts,
    # while names materialized into this result remain authoritative.
    for key in (:global_consts_ty, :local_consts_ty, :preset_types)
        source = get(kwargs, key, nothing)
        source === nothing && continue
        for (name, ty) in pairs(source)
            ty isa TyLike && (context[name] = ty)
        end
    end
    merge!(context, result.local_const_types)
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

function _verify_functiondef_result(
    result,
    target_ty;
    kwargs,
    scope_tys=Ty[],
)
    result isa CG.GenerationResult || return result
    if !_contains_functiondef(result.tree.root)
        result.metadata[:functiondef_verification] = :not_needed
        return result
    end
    result.metadata[:functiondef_observations] =
        functiondef_recursion_observations(result.tree)
    verification_tree, expected = _close_scoped_verification(result.tree, target_ty, scope_tys)
    inferred = TypeSystem.infer_term(
        verification_tree,
        _verification_context(result, kwargs);
    )
    inferred isa Err && return inferred
    TypeSystem.iscompatible(inferred, expected) ||
        return Err(FunctionDefGenerationVerificationError(expected, inferred))
    result.metadata[:functiondef_verification] = :verified
    return result
end

function _functiondef_verification_required(config::CG.GenerationConfig, presets)
    config.materializer isa CG.DefaultAstMaterializer || return true
    config.sampler isa CG.WeightedDerivationSampler || return true
    config.argument_planner isa CG.DefaultArgumentPlanner || return true
    all(_known_core_provider_without_functiondef, config.providers) || return true
    for preset in values(presets)
        node = preset isa AstTree ? preset.root : preset
        if node isa AstNode
            _contains_functiondef(node) && return true
        elseif !(node isa Union{Number,Symbol,AbstractString,Char,Bool,Nothing,Missing})
            # An unfamiliar preset wrapper may carry an AST that can be
            # materialized by a provider, so do not use the no-scan shortcut.
            return true
        end
    end
    return false
end

_known_core_provider_without_functiondef(provider::CG.SafeListAccessProvider) =
    isempty(provider.source_collectors)

_known_core_provider_without_functiondef(provider) = provider isa Union{
    CG.ContextHeadProvider,
    CG.NilFactoryProvider,
    CG.TypedValueFactoryProvider,
    CG.ToListProvider,
    CG.AMapProvider,
    CG.GuardedExpansionProvider,
    CG.WeightTagProvider,
}

function _finalize_functiondef_result(
    result,
    target_ty;
    kwargs,
    scope_tys=Ty[],
    verification_required::Bool=true,
)
    result isa CG.GenerationResult || return result
    if !verification_required
        result.metadata[:functiondef_verification] = :not_needed
        return result
    end
    return _verify_functiondef_result(result, target_ty; kwargs, scope_tys)
end

function _contains_functiondef(node::AstNode)
    node isa FunctionDefNode && return true
    return any(_contains_functiondef, children(node))
end

"""
    generate_finite_canonical_scoped(target_ty::Ty; scope_tys=Ty[], kwargs...)

Return one finite core witness while retaining equirecursive target and scope
support. The witness query masks `:function_def`, `:fixed_point`, and `:opaque`
capabilities transitively, so this entry can observe a finite `μ` graph without
constructing a recursive definition. It is primarily useful as the finite side
of FunctionDef removal and body replacement.
"""
function generate_finite_canonical_scoped(
    target_ty::Ty;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    config=nothing,
    kwargs...,
)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    effective = with_functiondef_provider(base_config, nothing)
    scope = CG._scope_type_vector(scope_tys)

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
            Pair{Symbol,Any}(:scope_tys, CG._scope_typekeys(scope)),
            Pair{Symbol,Any}(:derivation, CG.derivation_signature(witness)),
            :forced_finite_witness => true,
        ),
    )
    return _finalize_functiondef_result(
        result,
        target_ty;
        kwargs,
        scope_tys=scope,
        verification_required=_functiondef_verification_required(
            effective,
            get(kwargs, :presets, Dict{Symbol,Any}()),
        ),
    )
end

"""
    generate_functiondef_root_scoped(target_ty::Ty; scope_tys=Ty[], kwargs...)

Generate one guarded, genuinely self-using [`FunctionDefNode`](@ref) at the
root. Unlike [`generate_canonical_scoped`](@ref), this entry does not route
between ordinary core terms and FunctionDef: selecting it explicitly forces the
existing [`FunctionDefProvider`](@ref) protocol, including its finite base
witness, nesting fuel, shared attempt budget, and admissibility check.

`scope_tys` remains inner → outer. The provider and all ordinary body-generation
configuration are injectable; caller-owned RNG and constant containers are
forwarded unchanged. `completion_probability=0` retains the existing bounded
provider route. A positive probability enables an incremental certified root
frontier with one irreversible completion switch and optional finite
`completion_grace_steps`. This route currently covers ordinary/joint core and
positive safe-list evidence, not all mixed extensions. The original provider
filter still runs once after construction and may reject; there is no promise
of successful sampling for an arbitrary filter.
Omitted completion keywords inherit `WeightedDerivationSampler` settings from
`config`; explicit keywords override them independently. The default sampler
still supplies probability zero and no grace.
"""
function generate_functiondef_root_scoped(
    target_ty::Ty;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    functiondef_provider::FunctionDefProvider=FunctionDefProvider(),
    config=nothing,
    completion_probability::Union{Nothing,Real}=nothing,
    completion_grace_steps::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    functiondef_provider = _require_genuine_self_call(functiondef_provider)
    base_config = config === nothing ? _generation_config_from_kwargs(kwargs) : config
    base_config isa CG.GenerationConfig || throw(ArgumentError("config must be GenerationConfig"))
    sampler = base_config.sampler
    probability = completion_probability === nothing ?
        (sampler isa WeightedDerivationSampler ? sampler.completion_probability : 0.0) :
        completion_probability
    grace = completion_grace_steps === nothing ?
        (sampler isa WeightedDerivationSampler ? sampler.completion_grace_steps : 0) :
        completion_grace_steps
    completion = GenerationCompletionState(; probability, grace_steps=grace)
    effective = with_functiondef_provider(base_config, functiondef_provider)
    _validate_functiondef_sampling_config(effective)
    CG._generation_target_supported(target_ty, effective) ||
        return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))

    scope = CG._scope_type_vector(scope_tys)
    CG._generation_scope_supported(scope, effective) ||
        return Err(CG.UnsupportedCanonicalGenerationTarget(target_ty))

    return CG._with_scoped_generation_materials(
        effective;
        global_consts=get(kwargs, :global_consts, Dict{Symbol,Any}()),
        global_consts_ty=get(kwargs, :global_consts_ty, Dict{Symbol,Any}()),
        local_consts=get(kwargs, :local_consts, Dict{Symbol,Any}()),
        local_consts_ty=get(kwargs, :local_consts_ty, Dict{Symbol,Any}()),
        presets=get(kwargs, :presets, Dict{Symbol,Any}()),
        preset_types=get(kwargs, :preset_types, Dict{Symbol,Any}()),
        source_local_consts=get(kwargs, :source_local_consts, true),
    ) do relation, result_local_consts, result_local_const_types
        owned_provider = _owned_forced_functiondef_provider(relation, functiondef_provider)
        _generate_forced_functiondef_result(
            relation, effective, owned_provider, target_ty, scope,
            result_local_consts, result_local_const_types, kwargs, completion,
        )
    end
end

"""Select the cache-owned instance of a semantically cacheable root provider.

Scoped relation caching interns singleton-policy providers by their declared
cache key. A fresh equivalent caller instance must not fail the extension's
identity admission or use a different route token from the root attempt ledger.
Noncacheable policies take the uncached public path and keep caller identity.
"""
function _owned_forced_functiondef_provider(relation, requested::FunctionDefProvider)
    any(p -> p === requested, relation.providers) && return requested
    key = generation_provider_cache_key(requested)
    key === nothing && throw(ArgumentError("a noncacheable FunctionDef provider cannot use a shared relation"))
    slot = findfirst(relation.providers) do provider
        provider isa FunctionDefProvider && isequal(generation_provider_cache_key(provider), key)
    end
    slot === nothing && throw(ArgumentError("cached relation has no matching FunctionDef provider"))
    return relation.providers[slot]
end

function _generate_forced_functiondef_result(
    relation,
    effective,
    functiondef_provider,
    target_ty,
    scope,
    result_local_consts,
    result_local_const_types,
    kwargs,
    completion::GenerationCompletionState=GenerationCompletionState(),
)
    state = CG.DerivationSamplingState(effective.rng, effective.sampling_node_budget, completion)
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
    derivation = if completion.probability > 0
        _sample_certified_functiondef_root(state, relation, functiondef_provider,
            provider_options, query)
    else
        CG._sample_from_options(state, relation, wrapped_options, scope,
            query.depth_left, query.guarded_left, true, query)
    end
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
        Pair{Symbol,Any}(:scope_tys, CG._scope_typekeys(scope)),
        Pair{Symbol,Any}(:derivation, CG.derivation_signature(derivation)),
        :forced_functiondef_root => true,
    )
    if completion.probability > 0
        metadata[:certified_completion] = (
            probability=completion.probability,
            switched=completion.completing,
            boundaries=completion.boundaries,
            coverage=:incremental_mixed_frontier,
        )
    end
    merge!(metadata, CG.derivation_metadata(derivation))
    result = CG.GenerationResult(
        AstTree(node),
        target_ty,
        result_local_consts,
        result_local_const_types,
        metadata,
    )
    return _verify_functiondef_result(
        result,
        target_ty;
        kwargs,
        scope_tys=scope,
    )
end

"""Prepare one complete forced-root certificate, then run its owned frontier.

The provider anchors a finite base and the query supplies exact outer scope,
capabilities and a shared FunctionDef attempt ledger. Preparation consumes one
attempt, never emitted node fuel; body evidence adds the exact self Call.
Initial preparation chooses among the finite core and positive mixed routes;
core forwarding does not make the mixed route unreachable. Missing evidence
or a final provider-filter rejection
returns `nothing`; neither reopens the root mode nor proves mixed absence.
"""
function _sample_certified_functiondef_root(state, relation, provider, options, query)
    state.completion.completing && return nothing # no initial search in inherited completion
    query.depth_left > 0 && query.provider_fuel > 0 || return nothing
    # The legacy forced root passes through :generation_extension before the
    # provider is entered. Certification must pay that same dispatch depth and
    # fuel, not silently give the body an extra construction layer.
    query = _derive_generation_query(query; depth_left=query.depth_left - 1,
        provider_fuel=query.provider_fuel - 1,
        node_budget=min(state.nodes_left, query.node_budget))
    relation, query = _relation_for_query(relation, query)
    any(p -> p === provider, relation.providers) || return nothing
    isempty(options) && return nothing
    option = first(options)
    option.remaining > 0 || return nothing
    _functiondef_attempts_remaining(query, provider) > 0 || return nothing
    base = _functiondef_base_witness(provider, relation, option, query)
    base === nothing && return nothing
    query.node_budget > 1 || return nothing
    _consume_functiondef_attempt!(query, provider) || return nothing
    childq = _derive_generation_query(_close_functiondef_prefix(query);
        added_scope_tys=(query.target_ty,),
        node_budget=query.node_budget - 1,
        requirements=(FunctionDefFuel(option.remaining - 1),
            _functiondef_attempt_budget(query, provider), FunctionDefSelfBinder(0), CallsScopeBinder(0)))
    _frontier_prepare_supported(relation, childq) || return nothing
    # A cheap core forwarding certificate may be rejected by the unchanged
    # recursion filter. Give the already admitted finite mixed preparer a
    # positive initial route, rather than reaching it only on core absence.
    body = _prepare_frontier_candidate(relation, childq, state.rng)
    body === nothing && return nothing
    certificate = FunctionDefDerivation(query.target_ty, body, base, 0,
        provider_capabilities(provider), option.remaining - 1)
    nodes_before = state.nodes_left
    result = _sample_certified_frontier!(state, relation, query, certificate; functiondef_root=true)
    result === nothing && return nothing
    candidate = FunctionDefCandidate(query.target_ty, result.body, base, 0)
    _functiondef_admissible(provider, candidate, relation, query) || return nothing
    derivation_node_count(result) <= nodes_before - state.nodes_left || return nothing
    result_ty = derivation_result_type(result)
    (result_ty === query.target_ty || TypeSystem.iscompatible(result_ty, query.target_ty)) || return nothing
    if _contains_rigid_dummy(query.target_ty) || _scope_any(_contains_rigid_dummy, query.scope_tys)
        safe = rigid_safe_derivation(result)
        safe isa Bool || throw(ArgumentError("rigid_safe_derivation must return Bool"))
        safe || return nothing
    end
    return result
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
