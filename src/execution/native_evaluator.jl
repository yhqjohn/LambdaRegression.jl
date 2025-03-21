abstract type NativeEvaluationError <: Exception end

"""Native evaluation exhausted its shared reduction budget."""
struct NativeStepLimitError <: NativeEvaluationError
    steps::Int
end

"""A native program finished as syntax or an unresolved value."""
struct NativeNonValueError <: NativeEvaluationError
    value_kind::Symbol
end

"""A De Bruijn reference escaped the lexical environment."""
struct NativeUnboundIndexError <: NativeEvaluationError
    index::Int
end

"""A constant required at runtime was absent from the evaluation context."""
struct NativeMissingConstantError <: NativeEvaluationError
    name::Symbol
end

"""A call-by-need thunk recursively demanded itself before producing a value."""
struct NativeEvaluationCycleError <: NativeEvaluationError end

"""
    PreparedNativeEvaluator

Prepared immutable-by-convention program handle for repeated native evaluation.
Use [`prepare_native_evaluator`](@ref) once per individual, then [`run_native_evaluator`](@ref)
for each sample. Runtime environments and budgets are created per run.
"""
struct PreparedNativeEvaluator
    root::AstNode
end

abstract type _NativeValue end

struct _HostValue <: _NativeValue
    value
end

mutable struct _EvalThunk
    node::Union{Nothing,AstNode}
    env::Vector{_EvalThunk}
    names::Vector{Union{Nothing,Symbol}}
    value::Union{Nothing,_NativeValue}
    state::UInt8
end

struct _ClosureValue <: _NativeValue
    name::Symbol
    body::AstNode
    env::Vector{_EvalThunk}
    names::Vector{Union{Nothing,Symbol}}
end

struct _FixValue <: _NativeValue
    body::AstNode
    env::Vector{_EvalThunk}
    names::Vector{Union{Nothing,Symbol}}
end

mutable struct _EvaluationBudget
    remaining::Int
    initial::Int
end

struct _EvaluationState
    context
    budget::_EvaluationBudget
end

_node_thunk(
    node::AstNode,
    env::Vector{_EvalThunk},
    names::Vector{Union{Nothing,Symbol}},
) = _EvalThunk(node, env, names, nothing, 0x00)
_value_thunk(value::_NativeValue) =
    _EvalThunk(nothing, _EvalThunk[], Union{Nothing,Symbol}[], value, 0x02)

function _tick!(state::_EvaluationState)
    state.budget.remaining > 0 || throw(NativeStepLimitError(state.budget.initial))
    state.budget.remaining -= 1
    return nothing
end

function _force!(thunk::_EvalThunk, state::_EvaluationState)
    thunk.state == 0x02 && return thunk.value::_NativeValue
    thunk.state == 0x01 && throw(NativeEvaluationCycleError())
    _tick!(state)
    thunk.state = 0x01
    try
        value = _evaluate_node!(thunk.node::AstNode, thunk.env, thunk.names, state)
        thunk.value = value
        thunk.node = nothing
        empty!(thunk.env)
        empty!(thunk.names)
        thunk.state = 0x02
        return value
    catch
        thunk.state = 0x00
        rethrow()
    end
end

function _lookup_index(env::Vector{_EvalThunk}, index::Int)
    position = length(env) - index
    1 <= position <= length(env) || throw(NativeUnboundIndexError(index))
    return env[position]
end

function _lookup_name(
    env::Vector{_EvalThunk},
    names::Vector{Union{Nothing,Symbol}},
    name::Symbol,
)
    position = findlast(==(name), names)
    position === nothing && throw(NativeNonValueError(:free_variable))
    return env[position]
end

struct _NativeCallable
    value::_NativeValue
    state::_EvaluationState
    site_env::Vector{_EvalThunk}
    site_names::Vector{Union{Nothing,Symbol}}
end

function (callable::_NativeCallable)(argument)
    value = _apply_value!(
        callable.value,
        _value_thunk(_HostValue(argument)),
        callable.site_env,
        callable.site_names,
        callable.state,
    )
    return _to_host(value, callable.state, callable.site_env, callable.site_names)
end

_to_host(value::_HostValue, state, site_env, site_names) = value.value
_to_host(value::Union{_ClosureValue,_FixValue}, state, site_env, site_names) =
    _NativeCallable(value, state, copy(site_env), copy(site_names))

function _from_host(
    value,
    env::Vector{_EvalThunk},
    names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    value isa _NativeValue && return value
    value isa AstTree && return _evaluate_node!(value.root, env, names, state)
    value isa AstNode && return _evaluate_node!(value, env, names, state)
    return _HostValue(value)
end

function _apply_value!(
    function_value::_ClosureValue,
    argument::_EvalThunk,
    site_env::Vector{_EvalThunk},
    site_names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    _tick!(state)
    body_env = copy(function_value.env)
    body_names = copy(function_value.names)
    push!(body_env, argument)
    push!(body_names, function_value.name)
    return _evaluate_node!(function_value.body, body_env, body_names, state)
end

function _apply_value!(
    function_value::_FixValue,
    argument::_EvalThunk,
    site_env::Vector{_EvalThunk},
    site_names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    _tick!(state)
    body_env = copy(function_value.env)
    body_names = copy(function_value.names)
    push!(body_env, _value_thunk(function_value))
    push!(body_names, nothing)
    callable = _evaluate_node!(function_value.body, body_env, body_names, state)
    return _apply_value!(callable, argument, body_env, body_names, state)
end

function _apply_value!(
    function_value::_HostValue,
    argument::_EvalThunk,
    site_env::Vector{_EvalThunk},
    site_names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    _tick!(state)
    argument_value = _force!(argument, state)
    host_argument = _to_host(argument_value, state, site_env, site_names)
    result = function_value.value(host_argument)
    return _from_host(result, site_env, site_names, state)
end

function _apply_value!(
    function_value::_NativeValue,
    argument::_EvalThunk,
    site_env::Vector{_EvalThunk},
    site_names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    throw(NativeNonValueError(:non_callable))
end

function _evaluate_node!(
    node::AstNode,
    env::Vector{_EvalThunk},
    names::Vector{Union{Nothing,Symbol}},
    state::_EvaluationState,
)
    _tick!(state)
    if node isa ConstNode
        haskey(state.context, node.name) ||
            throw(NativeMissingConstantError(node.name))
        return _HostValue(state.context[node.name])
    elseif node isa VarNode
        return _force!(_lookup_name(env, names, node.name), state)
    elseif node isa IndexNode
        return _force!(_lookup_index(env, node.index), state)
    elseif node isa AbsNode
        return _ClosureValue(node.name, node.body, copy(env), copy(names))
    elseif node isa AppNode
        function_value = _evaluate_node!(node.func, env, names, state)
        argument = _node_thunk(node.arg, copy(env), copy(names))
        return _apply_value!(function_value, argument, env, names, state)
    elseif node isa FunctionDefNode
        return _FixValue(node.body, copy(env), copy(names))
    end
    throw(NativeNonValueError(:unsupported_node))
end

"""
    prepare_native_evaluator(term; copy_tree=true) -> PreparedNativeEvaluator

Prepare a FunctionDef or core AST for repeated native evaluation. The default owns
a structural copy so later mutation of the source individual cannot change an
already-prepared program. Internal hot paths may pass `copy_tree=false` when the
individual is immutable for the duration of the run.
"""
function prepare_native_evaluator(term::Union{AstNode,AstTree}; copy_tree::Bool=true)
    root = term isa AstTree ? term.root : term
    return PreparedNativeEvaluator(copy_tree ? deepcopy(root) : root)
end

"""
    run_native_evaluator(prepared, context; args=(), max_steps=256)

Run a prepared program with positional host arguments. `args` should be a tuple;
a non-tuple value is treated as one argument. The step budget is shared by all
recursive calls, thunks, and higher-order callbacks in this run. Expected
resource and malformed-program failures from the run return `Err`. A callable
returned by the run retains its state and budget; later host calls may throw a
native evaluation error directly when that shared budget is exhausted.
"""
function run_native_evaluator(
    prepared::PreparedNativeEvaluator,
    context;
    args=(),
    max_steps::Int=256,
)
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    arguments = args isa Tuple ? args : (args,)
    state = _EvaluationState(context, _EvaluationBudget(max_steps, max_steps))
    try
        empty_names = Union{Nothing,Symbol}[]
        value = _evaluate_node!(prepared.root, _EvalThunk[], empty_names, state)
        for argument in arguments
            value = _apply_value!(
                value,
                _value_thunk(_HostValue(argument)),
                _EvalThunk[],
                empty_names,
                state,
            )
        end
        return _to_host(value, state, _EvalThunk[], empty_names)
    catch error
        error isa NativeEvaluationError && return Err(error)
        rethrow()
    end
end

"""Prepare and run one term natively. Prefer `prepare_native_evaluator` for batches."""
function evaluate_native(term::Union{AstNode,AstTree}, context; kwargs...)
    return run_native_evaluator(prepare_native_evaluator(term), context; kwargs...)
end
