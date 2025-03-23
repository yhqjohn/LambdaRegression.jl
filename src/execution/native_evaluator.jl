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

"""Nested source calls exceeded the basic host-stack safety limit."""
struct NativeCallDepthLimitError <: NativeEvaluationError
    depth::Int
    limit::Int
end

const _NATIVE_CONST = 0x01
const _NATIVE_VAR = 0x02
const _NATIVE_INDEX = 0x03
const _NATIVE_ABS = 0x04
const _NATIVE_APP = 0x05
const _NATIVE_FUNCTIONDEF = 0x06
const _NATIVE_UNSUPPORTED = 0x07

struct _NativeInstruction
    opcode::UInt8
    first::Int
    second::Int
    name::Union{Nothing,Symbol}
end

const _EMPTY_NATIVE_INSTRUCTION = _NativeInstruction(0x00, 0, 0, nothing)

"""
    PreparedNativeEvaluator

Prepared source program and compact indexed execution code. Use
[`prepare_native_evaluator`](@ref) once per individual, then
[`run_native_evaluator`](@ref) for each sample. The source AST remains available
as the authoritative program; the derived instruction vector is used at runtime.
Runtime environments and budgets are created per run.
"""
struct PreparedNativeEvaluator
    root::AstNode
    code::Vector{_NativeInstruction}
end

abstract type _NativeValue end

struct _HostValue{T} <: _NativeValue
    value::T
end

mutable struct _EvalThunk
    instruction::Int
    env
    value::Union{Nothing,_NativeValue}
    state::UInt8
end

# Frames are persistent: fields are written only at construction. New binders
# point to an existing frame rather than copying all enclosing bindings.
mutable struct _EvalEnv
    parent::Union{Nothing,_EvalEnv}
    thunk::_EvalThunk
    name::Union{Nothing,Symbol}
end

struct _ClosureValue <: _NativeValue
    name::Symbol
    body::Int
    env::Union{Nothing,_EvalEnv}
end

struct _FixValue <: _NativeValue
    body::Int
    env::Union{Nothing,_EvalEnv}
end

mutable struct _EvaluationBudget
    remaining::Int
    initial::Int
end

mutable struct _EvaluationState
    context
    budget::_EvaluationBudget
    prepared_code::Vector{_NativeInstruction}
    dynamic_code::Vector{_NativeInstruction}
    call_depth::Int
    max_call_depth::Int
end

_bind(env, thunk, name) = _EvalEnv(env, thunk, name)
_node_thunk(instruction::Int, env) = _EvalThunk(instruction, env, nothing, 0x00)
_value_thunk(value::_NativeValue) = _EvalThunk(0, nothing, value, 0x02)

function _tick!(state::_EvaluationState)
    state.budget.remaining > 0 || throw(NativeStepLimitError(state.budget.initial))
    state.budget.remaining -= 1
    return nothing
end

function _with_native_call(f, state::_EvaluationState)
    depth = state.call_depth + 1
    depth <= state.max_call_depth ||
        throw(NativeCallDepthLimitError(depth, state.max_call_depth))
    state.call_depth = depth
    try
        return f()
    finally
        state.call_depth -= 1
    end
end

function _instruction!(code::Vector{_NativeInstruction}, node::AstNode, offset::Int)
    instruction = offset + length(code) + 1
    push!(code, _EMPTY_NATIVE_INSTRUCTION)
    if node isa ConstNode
        code[instruction - offset] = _NativeInstruction(_NATIVE_CONST, 0, 0, node.name)
    elseif node isa VarNode
        code[instruction - offset] = _NativeInstruction(_NATIVE_VAR, 0, 0, node.name)
    elseif node isa IndexNode
        code[instruction - offset] = _NativeInstruction(_NATIVE_INDEX, node.index, 0, nothing)
    elseif node isa AbsNode
        body = _instruction!(code, node.body, offset)
        code[instruction - offset] = _NativeInstruction(_NATIVE_ABS, body, 0, node.name)
    elseif node isa AppNode
        func = _instruction!(code, node.func, offset)
        arg = _instruction!(code, node.arg, offset)
        code[instruction - offset] = _NativeInstruction(_NATIVE_APP, func, arg, nothing)
    elseif node isa FunctionDefNode
        body = _instruction!(code, node.body, offset)
        code[instruction - offset] = _NativeInstruction(_NATIVE_FUNCTIONDEF, body, 0, nothing)
    else
        code[instruction - offset] = _NativeInstruction(_NATIVE_UNSUPPORTED, 0, 0, nothing)
    end
    return instruction
end

function _native_instruction(state::_EvaluationState, instruction::Int)
    prepared_length = length(state.prepared_code)
    if instruction <= prepared_length
        return state.prepared_code[instruction]
    end
    return state.dynamic_code[instruction - prepared_length]
end

function _force!(thunk::_EvalThunk, state::_EvaluationState)
    thunk.state == 0x02 && return thunk.value::_NativeValue
    thunk.state == 0x01 && throw(NativeEvaluationCycleError())
    _tick!(state)
    thunk.state = 0x01
    try
        value = _evaluate_node!(thunk.instruction, thunk.env, state)
        thunk.value = value
        thunk.instruction = 0
        thunk.env = nothing
        thunk.state = 0x02
        return value
    catch
        thunk.state = 0x00
        rethrow()
    end
end

function _lookup_index(env::Union{Nothing,_EvalEnv}, index::Int)
    index >= 0 || throw(NativeUnboundIndexError(index))
    frame = env
    for _ in 1:index
        frame === nothing && throw(NativeUnboundIndexError(index))
        frame = frame.parent
    end
    frame === nothing && throw(NativeUnboundIndexError(index))
    return frame.thunk
end

function _lookup_name(env::Union{Nothing,_EvalEnv}, name::Symbol)
    frame = env
    while frame !== nothing
        frame.name === name && return frame.thunk
        frame = frame.parent
    end
    throw(NativeNonValueError(:free_variable))
end

struct _NativeCallable
    value::_NativeValue
    state::_EvaluationState
    site_env::Union{Nothing,_EvalEnv}
end

function (callable::_NativeCallable)(argument)
    value = _apply_value!(
        callable.value,
        _value_thunk(_HostValue(argument)),
        callable.site_env,
        callable.state,
    )
    return _to_host(value, callable.state, callable.site_env)
end

_to_host(value::_HostValue, state, site_env) = value.value
_to_host(value::Union{_ClosureValue,_FixValue}, state, site_env) =
    _NativeCallable(value, state, site_env)

function _from_host(value, env, state::_EvaluationState)
    value isa _NativeValue && return value
    if value isa AstTree || value isa AstNode
        node = value isa AstTree ? value.root : value
        offset = length(state.prepared_code)
        instruction = _instruction!(state.dynamic_code, node, offset)
        return _evaluate_node!(instruction, env, state)
    end
    return _HostValue(value)
end

function _apply_value!(
    function_value::_ClosureValue,
    argument::_EvalThunk,
    site_env,
    state::_EvaluationState,
)
    return _with_native_call(state) do
        _tick!(state)
        body_env = _bind(function_value.env, argument, function_value.name)
        _evaluate_node!(function_value.body, body_env, state)
    end
end

function _apply_value!(
    function_value::_FixValue,
    argument::_EvalThunk,
    site_env,
    state::_EvaluationState,
)
    return _with_native_call(state) do
        _tick!(state)
        body_env = _bind(function_value.env, _value_thunk(function_value), nothing)
        callable = _evaluate_node!(function_value.body, body_env, state)
        _apply_value!(callable, argument, body_env, state)
    end
end

function _apply_value!(
    function_value::_HostValue,
    argument::_EvalThunk,
    site_env,
    state::_EvaluationState,
)
    _tick!(state)
    argument_value = argument.state == 0x02 ? argument.value::_NativeValue :
        _force!(argument, state)
    return _invoke_host!(function_value, argument_value, site_env, state)
end

function _invoke_host!(
    function_value::_HostValue{F},
    argument_value::_NativeValue,
    site_env,
    state,
) where {F}
    host_argument = _to_host(argument_value, state, site_env)
    result = function_value.value(host_argument)
    return _from_host(result, site_env, state)
end

function _apply_value!(
    function_value::_NativeValue,
    argument::_EvalThunk,
    site_env,
    state::_EvaluationState,
)
    throw(NativeNonValueError(:non_callable))
end

function _evaluate_node!(instruction::Int, env, state::_EvaluationState)
    _tick!(state)
    node = _native_instruction(state, instruction)
    if node.opcode == _NATIVE_CONST
        name = node.name::Symbol
        haskey(state.context, name) || throw(NativeMissingConstantError(name))
        return _HostValue(state.context[name])
    elseif node.opcode == _NATIVE_VAR
        return _force!(_lookup_name(env, node.name::Symbol), state)
    elseif node.opcode == _NATIVE_INDEX
        return _force!(_lookup_index(env, node.first), state)
    elseif node.opcode == _NATIVE_ABS
        return _ClosureValue(node.name::Symbol, node.first, env)
    elseif node.opcode == _NATIVE_APP
        function_value = _evaluate_node!(node.first, env, state)
        if function_value isa _HostValue
            # Host calls are strict in this evaluator, so evaluate the operand
            # directly instead of allocating a thunk that is forced immediately.
            _tick!(state)
            argument = _evaluate_node!(node.second, env, state)
            return _invoke_host!(function_value, argument, env, state)
        end
        argument = _node_thunk(node.second, env)
        return _apply_value!(function_value, argument, env, state)
    elseif node.opcode == _NATIVE_FUNCTIONDEF
        return _FixValue(node.first, env)
    elseif node.opcode == _NATIVE_UNSUPPORTED
        throw(NativeNonValueError(:unsupported_node))
    end
    throw(NativeNonValueError(:unsupported_node))
end

"""
    prepare_native_evaluator(term; copy_tree=true) -> PreparedNativeEvaluator

Prepare a FunctionDef or core AST for repeated native evaluation. The default owns
a structural copy so later mutation of the source individual cannot change an
already-prepared program. Preparation flattens that source into ordinary indexed
instructions; runtime evaluation does not traverse the AST. Internal hot paths
may pass `copy_tree=false` when the individual is immutable for the duration of
the run.
"""
function prepare_native_evaluator(term::Union{AstNode,AstTree}; copy_tree::Bool=true)
    source_root = term isa AstTree ? term.root : term
    root = copy_tree ? deepcopy(source_root) : source_root
    code = _NativeInstruction[]
    _instruction!(code, root, 0)
    return PreparedNativeEvaluator(root, code)
end

# Keep the exported one-argument constructor usable while routing it through
# the same preparation path as the named API.
PreparedNativeEvaluator(root::AstNode) =
    prepare_native_evaluator(root; copy_tree=false)

"""
    run_native_evaluator(prepared, context; args=(), max_steps=256,
                         max_call_depth=512)

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
    max_call_depth::Int=512,
)
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    max_call_depth > 0 || throw(ArgumentError("max_call_depth must be positive"))
    arguments = args isa Tuple ? args : (args,)
    state = _EvaluationState(
        context,
        _EvaluationBudget(max_steps, max_steps),
        prepared.code,
        _NativeInstruction[],
        0,
        max_call_depth,
    )
    try
        value = _evaluate_node!(1, nothing, state)
        for argument in arguments
            value = _apply_value!(
                value,
                _value_thunk(_HostValue(argument)),
                nothing,
                state,
            )
        end
        return _to_host(value, state, nothing)
    catch error
        error isa NativeEvaluationError && return Err(error)
        rethrow()
    end
end

"""Prepare and run one term natively. Prefer `prepare_native_evaluator` for batches."""
function evaluate_native(term::Union{AstNode,AstTree}, context; kwargs...)
    return run_native_evaluator(prepare_native_evaluator(term), context; kwargs...)
end

@testitem "Execution native indexed evaluator preserves dynamic scope and general FunctionDef bodies" begin
    using Test
    using LambdaRegression.Execution: evaluate_native, NativeNonValueError
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstNode, ConstNode, FunctionDefNode, IndexNode, VarNode
    using LambdaRegression.Utils: Err

    struct UnsupportedNativeNode <: AstNode end

    context = Dict{Symbol,Any}(
        :answer => 41,
        :return_named => _ -> VarNode(:x),
        :make_abstraction => _ -> AbsNode(:unused, IndexNode(0)),
    )
    named = AppNode(
        AbsNode(:x, AppNode(ConstNode(:return_named), IndexNode(0))),
        ConstNode(:answer),
    )
    @test evaluate_native(named, context) == 41

    # The fixed body computes a callable instead of being a lambda itself.
    fixed = FunctionDefNode(
        AppNode(ConstNode(:make_abstraction), IndexNode(0)),
    )
    @test evaluate_native(fixed, context; args=(43,)) == 43

    unsupported = evaluate_native(UnsupportedNativeNode(), context)
    @test unsupported isa Err
    @test unsupported.value isa NativeNonValueError
end
