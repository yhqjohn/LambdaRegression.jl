using ..Primitives: AMAP, AFOLDL, LENF, FSIN, FCOS, FSQRT, SQR,
    IEQ, ILE, IADD, ISUB, IMUL, FEQ, FLE, FADD, FSUB, FMUL, FDIV
using ..TypeSystem: Ty, TyConst, TyFun, TyArray, TyScheme, TyLike, IntTy, FloatTy,
    collect_free_vars, isequal_mu, islesseq

struct _FunctionDependencyPattern end
struct _ArrayDependencyPattern end
struct _ClosedDependencyPattern end
struct _ScalarDependencyPattern
    project_type::Ty
    host_type::Any
end

const _FUNCTION_DEPENDENCY_PATTERN = _FunctionDependencyPattern()
const _ARRAY_DEPENDENCY_PATTERN = _ArrayDependencyPattern()
const _CLOSED_DEPENDENCY_PATTERN = _ClosedDependencyPattern()
const _FLOAT_DEPENDENCY_PATTERN = _ScalarDependencyPattern(FloatTy, AbstractFloat)
const _INT_DEPENDENCY_PATTERN = _ScalarDependencyPattern(IntTy, Integer)

"""
    DependencySummary(given_arity; expected_arity=given_arity, ...)

Finite, conservative metadata about how a call may depend on its inputs.
Argument positions are one-based.
`value_inputs` marks argument values that may affect returned values;
`shape_inputs` marks argument shapes or lengths that may affect returned values
or result shape. Shape relations are coarse and do not distinguish logical array
layers; a nested-shape dependency need not affect an outer-length observer.
`callback_inputs` marks positions whose callable values may be
invoked. `callback_result_values` and `callback_result_shapes` separately mark
whether callback results may affect result values or result shape. These are
all may-relations: the summary cannot express
that an argument is always used or a callback is always called.

`unknown` and `reason` explain when the summary is a fallback. Assumptions are
stable symbols describing the scope of a known summary. A callee dependency
includes captured environment as well as the callee value itself.
"""
struct DependencySummary
    given_arity::Int
    expected_arity::Union{Nothing,Int}
    value_inputs::Tuple{Vararg{Int}}
    shape_inputs::Tuple{Vararg{Int}}
    callback_inputs::Tuple{Vararg{Int}}
    callback_result_values::Tuple{Vararg{Int}}
    callback_result_shapes::Tuple{Vararg{Int}}
    callee_dependency::Bool
    unknown::Bool
    reason::Symbol
    assumptions::Tuple{Vararg{Symbol}}

    function DependencySummary(given_arity::Integer;
            expected_arity::Union{Nothing,Integer}=given_arity,
            value_inputs=(), shape_inputs=(), callback_inputs=(),
            callback_result_values=(), callback_result_shapes=(),
            callee_dependency::Bool=false,
            unknown::Bool=false, reason::Symbol=(unknown ? :unknown : :known),
            assumptions=())
        given = Int(given_arity)
        expected = isnothing(expected_arity) ? nothing : Int(expected_arity)
        given >= 0 || throw(ArgumentError("given_arity must be nonnegative"))
        !isnothing(expected) && expected < 0 &&
            throw(ArgumentError("expected_arity must be nonnegative or nothing"))

        values = _dependency_positions(value_inputs, given, :value_inputs)
        shapes = _dependency_positions(shape_inputs, given, :shape_inputs)
        callbacks = _dependency_positions(callback_inputs, given, :callback_inputs)
        callback_values = _dependency_positions(
            callback_result_values, given, :callback_result_values)
        callback_shapes = _dependency_positions(
            callback_result_shapes, given, :callback_result_shapes)
        assumption_symbols = _dependency_assumptions(assumptions)
        return new(given, expected, values, shapes, callbacks, callback_values,
            callback_shapes, callee_dependency, unknown, reason, assumption_symbols)
    end
end

function _dependency_positions(positions, arity::Int, label::Symbol)
    result = Int[]
    for position in positions
        position isa Integer || throw(ArgumentError("$label must contain integer positions"))
        index = Int(position)
        1 <= index <= arity || throw(ArgumentError(
            "$label position $index is outside 1:$arity"))
        index in result || push!(result, index)
    end
    return Tuple(result)
end

function _dependency_assumptions(assumptions)
    result = Symbol[]
    for assumption in assumptions
        assumption isa Symbol || throw(ArgumentError("assumptions must contain Symbols"))
        assumption in result || push!(result, assumption)
    end
    return Tuple(result)
end

"""One identity-keyed rule. Type constraints use `nothing` as a wildcard."""
struct PrimitiveDependencyRule
    actual_value::Any
    summary::DependencySummary
    argument_types::Union{Nothing,Tuple}
    applicability::Any

    function PrimitiveDependencyRule(actual_value, summary::DependencySummary;
            argument_types=nothing, applicability=nothing)
        summary.unknown && throw(ArgumentError("rules must provide known summaries"))
        summary.given_arity == summary.expected_arity || throw(ArgumentError(
            "a rule summary must have matching given and expected arity"))
        constraints = isnothing(argument_types) ? nothing :
            Tuple(_owned_dependency_type_constraint(value) for value in argument_types)
        !isnothing(constraints) && length(constraints) != summary.given_arity &&
            throw(ArgumentError("argument_types must have one constraint per argument"))
        return new(actual_value, summary, constraints, applicability)
    end
end

_owned_dependency_type_constraint(value::TyScheme) = TyScheme(copy(value.vars), value.ty)
_owned_dependency_type_constraint(value) = value

"""
    PrimitiveDependencyProfile(rules=(); include_builtins=true)

An immutable collection of identity-keyed dependency rules. Rule order has no
semantic effect: among matching rules, the most constrained rule wins, and an
equally specific overlap returns an unknown summary. Custom applicability
values are opaque stable tokens compared with `===`; no predicate or user
function is executed. Keep function identities, type constraints, and tokens
stable for the lifetime of a profile. The profile owns an immutable tuple of
rules; rule summaries and their dependency sets are also immutable. A `nothing`
argument-type constraint is the only wildcard. Project `Ty` constraints must
be closed and match by the TypeSystem's recursive equality; schemes match by
mutual instance containment after their quantified variables are validated.
"""
struct PrimitiveDependencyProfile
    rules::Tuple{Vararg{PrimitiveDependencyRule}}

    function PrimitiveDependencyProfile(rules=(); include_builtins::Bool=true)
        supplied = Tuple(rules)
        all(rule -> rule isa PrimitiveDependencyRule, supplied) ||
            throw(ArgumentError("rules must contain PrimitiveDependencyRule values"))
        builtins = include_builtins ? _BUILTIN_DEPENDENCY_RULES : ()
        return new((builtins..., supplied...))
    end
end

export DependencySummary, PrimitiveDependencyRule, PrimitiveDependencyProfile,
    default_dependency_profile, query_dependency

function _unknown_dependency_summary(given_arity::Int, expected_arity, reason::Symbol)
    every_argument = 1:given_arity
    return DependencySummary(given_arity;
        expected_arity=expected_arity,
        value_inputs=every_argument,
        shape_inputs=every_argument,
        callback_inputs=every_argument,
        callback_result_values=every_argument,
        callback_result_shapes=every_argument,
        callee_dependency=true,
        unknown=true,
        reason=reason,
        assumptions=(:all_arguments_and_callee_may_affect_result,))
end

"""
    query_dependency(profile, actual_value, argument_types; condition=nothing)

Return a finite dependency summary for a fully collected application. Rules
match only when the primitive value is identical (`===`), the supplied arity
matches, every declared argument type constraint matches, and any explicit
applicability token is identical. `argument_types` is a finite iterable of type
metadata, not argument values. This function never evaluates the callee or its
arguments.
Partial applications, wrong arity, unknown functions, and unmatched conditions
return a conservative unknown summary that includes every supplied argument and
the callee/captured environment as possible dependencies.
"""
function query_dependency(profile::PrimitiveDependencyProfile, actual_value,
        argument_types; condition=nothing)
    # Runtime analysis metadata is commonly a Vector. Do not turn its length or
    # contents into a Tuple specialization; collect only generic iterators.
    types = argument_types isa AbstractVector ? argument_types : collect(argument_types)
    given_arity = length(types)
    identity_rules = PrimitiveDependencyRule[]
    for rule in profile.rules
        rule.actual_value === actual_value && push!(identity_rules, rule)
    end
    isempty(identity_rules) &&
        return _unknown_dependency_summary(given_arity, nothing, :unregistered)

    expected_arities = unique!(Int[rule.summary.expected_arity for rule in identity_rules])
    arity_rules = PrimitiveDependencyRule[]
    for rule in identity_rules
        rule.summary.expected_arity == given_arity && push!(arity_rules, rule)
    end
    isempty(arity_rules) && begin
        expected = length(expected_arities) == 1 ? expected_arities[1] : nothing
        reason = any(arity -> given_arity < arity, expected_arities) ?
            :partial_application : :wrong_arity
        return _unknown_dependency_summary(given_arity, expected, reason)
    end

    matching = PrimitiveDependencyRule[]
    for rule in arity_rules
        _dependency_types_match(rule.argument_types, types) &&
            _dependency_condition_matches(rule.applicability, condition) &&
            push!(matching, rule)
    end
    isempty(matching) && return _unknown_dependency_summary(
        given_arity, given_arity, :type_or_condition_mismatch)

    best_specificity = _dependency_specificity(matching[1])
    best_rule = matching[1]
    best_count = 1
    for rule in Iterators.drop(matching, 1)
        specificity = _dependency_specificity(rule)
        if isless(best_specificity, specificity)
            best_specificity = specificity
            best_rule = rule
            best_count = 1
        elseif specificity == best_specificity
            best_count += 1
        end
    end
    best_count == 1 || return _unknown_dependency_summary(
        given_arity, given_arity, :ambiguous_rules)
    return best_rule.summary
end

function _dependency_types_match(constraints::Nothing, argument_types)
    return true
end

function _dependency_types_match(constraints::Tuple, argument_types)
    return all(eachindex(constraints)) do index
        expected = constraints[index]
        isnothing(expected) || _dependency_type_matches(argument_types[index], expected)
    end
end

function _dependency_type_matches(actual, expected)
    expected isa _FunctionDependencyPattern &&
        return (actual isa TyFun && _is_closed_monotype(actual)) ||
            (_is_type_object(actual) && actual <: Function)
    expected isa _ArrayDependencyPattern &&
        return (actual isa TyArray && _is_closed_monotype(actual)) ||
            (_is_type_object(actual) && actual <: AbstractArray)
    expected isa _ClosedDependencyPattern && return _is_closed_dependency_type(actual)
    if expected isa _ScalarDependencyPattern
        actual isa Ty && return _closed_type_equal(actual, expected.project_type)
        return _is_type_object(actual) && actual <: expected.host_type
    end
    if actual isa TyLike || expected isa TyLike
        (actual isa TyLike && expected isa TyLike) || return false
        return _closed_type_equal(actual, expected)
    end
    if _is_type_object(actual) && _is_type_object(expected)
        return actual <: expected
    end
    # Opaque type-system objects are compared by identity/value identity only;
    # never call an overloaded equality method supplied by an analysis client.
    return actual === expected
end

function _closed_type_equal(left::Ty, right::Ty)
    try
        return isequal_mu(left, right)
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
end

function _is_closed_monotype(ty::Ty)
    return _closed_type_equal(ty, ty)
end

function _closed_type_equal(left::TyScheme, right::TyScheme)
    _closed_scheme(left) && _closed_scheme(right) || return false
    try
        # Mutual instance containment is the trusted TypeSystem equivalence for
        # complete polymorphic schemes. It avoids large structural tuple keys.
        return islesseq(left, right) && islesseq(right, left)
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
end

function _closed_scheme(scheme::TyScheme)
    vars = scheme.vars
    length(unique(vars)) == length(vars) || return false
    all(var -> var.instance === nothing, vars) || return false
    free_vars = collect_free_vars(scheme.ty)
    length(free_vars) == length(vars) || return false
    all(var -> var in free_vars, vars) || return false
    try
        return islesseq(scheme, scheme)
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
end

_is_closed_dependency_type(ty::Ty) = _is_closed_monotype(ty)
_is_closed_dependency_type(scheme::TyScheme) = _closed_scheme(scheme)
_is_closed_dependency_type(ty) = _is_type_object(ty) && ty !== Any

_is_type_object(value) = value isa Type || value isa UnionAll || value isa Union

function _dependency_condition_matches(rule_condition, query_condition)
    return isnothing(rule_condition) ||
        (!isnothing(query_condition) && rule_condition === query_condition)
end

function _dependency_specificity(rule::PrimitiveDependencyRule)
    type_specificity = isnothing(rule.argument_types) ? 0 :
        count(constraint -> !isnothing(constraint), rule.argument_types)
    return (!isnothing(rule.applicability), type_specificity)
end

const _BUILTIN_DEPENDENCY_RULES = (
    PrimitiveDependencyRule(AMAP, DependencySummary(2;
        value_inputs=(1, 2), shape_inputs=(2,), callback_inputs=(1,),
        callback_result_values=(1,), callback_result_shapes=(1,), assumptions=(
            :maps_outer_logical_layer,
            :preserves_outer_logical_length,
            :callback_result_may_affect_values_and_nested_shapes,
            :callback_may_be_invoked_zero_or_more_times));
        argument_types=(_FUNCTION_DEPENDENCY_PATTERN, _ARRAY_DEPENDENCY_PATTERN)),
    PrimitiveDependencyRule(AFOLDL, DependencySummary(3;
        value_inputs=(1, 2, 3), shape_inputs=(2, 3), callback_inputs=(1,),
        callback_result_values=(1,), callback_result_shapes=(1,), assumptions=(
            :left_fold_over_outer_logical_layer,
            :callback_may_be_invoked_zero_or_more_times));
        argument_types=(_FUNCTION_DEPENDENCY_PATTERN, _CLOSED_DEPENDENCY_PATTERN,
            _ARRAY_DEPENDENCY_PATTERN)),
    PrimitiveDependencyRule(LENF, DependencySummary(1;
        shape_inputs=(1,), assumptions=(
            :outer_logical_length_affects_numeric_result,
            :input_elements_and_inner_shapes_do_not_affect_length));
        argument_types=(_ARRAY_DEPENDENCY_PATTERN,)),
    (PrimitiveDependencyRule(primitive, DependencySummary(1;
        value_inputs=(1,), assumptions=(:scalar_value_operation,));
        argument_types=(_FLOAT_DEPENDENCY_PATTERN,))
        for primitive in (FSIN, FCOS, FSQRT, SQR))...,
    (PrimitiveDependencyRule(primitive, DependencySummary(2;
        value_inputs=(1, 2), assumptions=(:scalar_value_operation,));
        argument_types=(_INT_DEPENDENCY_PATTERN, _INT_DEPENDENCY_PATTERN))
        for primitive in (IEQ, ILE, IADD, ISUB, IMUL))...,
    (PrimitiveDependencyRule(primitive, DependencySummary(2;
        value_inputs=(1, 2), assumptions=(:scalar_value_operation,));
        argument_types=(_FLOAT_DEPENDENCY_PATTERN, _FLOAT_DEPENDENCY_PATTERN))
        for primitive in (FEQ, FLE, FADD, FSUB, FMUL, FDIV))...,
)

"""Return a fresh immutable profile containing the built-in safe summaries."""
default_dependency_profile() = PrimitiveDependencyProfile()

using TestItems

@testitem "Structure diversity dependency profiles are identity keyed and conservative" begin
    using Test
    using LambdaRegression.StructureDiversity: DependencySummary,
        PrimitiveDependencyRule, PrimitiveDependencyProfile, query_dependency
    using LambdaRegression.Primitives: AMAP, AFOLDL, LENF, FSIN, SQR
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyArray, TyNat,
        IntTy, FloatTy, newtyvar

    summary = DependencySummary(1; value_inputs=(1,))
    shadow = x -> x
    source_rules = [PrimitiveDependencyRule(shadow, summary)]
    profile = PrimitiveDependencyProfile(source_rules; include_builtins=false)
    empty!(source_rules)

    @test query_dependency(profile, shadow, (Float64,)) === summary
    @test isimmutable(query_dependency(profile, shadow, (Float64,)))
    @test query_dependency(profile, identity, (Float64,)).unknown
    @test query_dependency(profile, shadow, ()).reason == :partial_application
    @test query_dependency(profile, shadow, (Float64, Int)).reason == :wrong_arity
    @test query_dependency(profile, shadow, (Float64,)).given_arity == 1
    @test query_dependency(profile, shadow, (Float64,)).expected_arity == 1

    condition = :array_case
    conditional_profile = PrimitiveDependencyProfile((
        PrimitiveDependencyRule(shadow, summary; applicability=condition),);
        include_builtins=false)
    @test query_dependency(conditional_profile, shadow, (Float64,)).unknown
    @test query_dependency(conditional_profile, shadow, (Float64,);
        condition=condition) === summary
    typed_profile = PrimitiveDependencyProfile((
        PrimitiveDependencyRule(shadow, summary; argument_types=(Real,)),);
        include_builtins=false)
    @test query_dependency(typed_profile, shadow, (Float64,)) === summary
    @test query_dependency(typed_profile, shadow, (String,)).unknown

    closed_type = TyFun(TyConst(:Float), TyConst(:Float))
    type_summary = DependencySummary(1; value_inputs=(1,))
    project_type_profile = PrimitiveDependencyProfile((
        PrimitiveDependencyRule(shadow, type_summary;
            argument_types=(closed_type,)),); include_builtins=false)
    fresh_closed_type = TyFun(TyConst(:Float), TyConst(:Float))
    @test query_dependency(project_type_profile, shadow, [fresh_closed_type]) === type_summary
    open_var = newtyvar(0)
    open_type = TyFun(open_var, open_var)
    @test query_dependency(project_type_profile, shadow, [open_type]).unknown

    builtins = PrimitiveDependencyProfile()
    array_type = TyArray(TyNat(4), FloatTy)
    mapper_type = TyFun(FloatTy, FloatTy)
    map_summary = query_dependency(builtins, AMAP, [mapper_type, array_type])
    length_summary = query_dependency(builtins, LENF, [array_type])
    @test !map_summary.unknown
    @test map_summary.callback_inputs == (1,)
    @test map_summary.callback_result_values == (1,)
    @test map_summary.callback_result_shapes == (1,) # inner shapes, not outer length
    @test :preserves_outer_logical_length in map_summary.assumptions
    @test length_summary.shape_inputs == (1,)
    @test isempty(length_summary.value_inputs)
    @test :input_elements_and_inner_shapes_do_not_affect_length in length_summary.assumptions
    @test query_dependency(builtins, LENF, [IntTy]).unknown
    fold_type = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    @test !query_dependency(builtins, AFOLDL, [fold_type, FloatTy, array_type]).unknown
    @test query_dependency(builtins, AFOLDL, [fold_type, Any, array_type]).unknown
    @test query_dependency(builtins, AMAP, [mapper_type, array_type, FloatTy]).reason == :wrong_arity
    @test query_dependency(builtins, FSIN, [FloatTy]).value_inputs == (1,)
    @test query_dependency(builtins, SQR, [FloatTy]).value_inputs == (1,)
    @test query_dependency(builtins, AMAP, [mapper_type]).reason == :partial_application

    unknown = query_dependency(builtins, shadow, (Float64, Bool))
    @test unknown.unknown
    @test unknown.value_inputs == (1, 2)
    @test unknown.shape_inputs == (1, 2)
    @test unknown.callback_inputs == (1, 2)
    @test unknown.callback_result_values == (1, 2)
    @test unknown.callback_result_shapes == (1, 2)
    @test unknown.callee_dependency
end
