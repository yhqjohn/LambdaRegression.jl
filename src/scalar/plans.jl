"""Expected shape or parameter mismatch in a direct collection plan."""
struct ScalarPlanShapeErr
    reason::Symbol
end

abstract type _ScalarKernel end
struct _InputKernel{I} <: _ScalarKernel end
struct _ConstantKernel <: _ScalarKernel
    value::Float64
end
struct _UnaryKernel{OP,A<:_ScalarKernel} <: _ScalarKernel
    arg::A
end
struct _BinaryKernel{OP,L<:_ScalarKernel,R<:_ScalarKernel} <: _ScalarKernel
    left::L
    right::R
end

_scalar_kernel(expr::_ScalarInput) = _InputKernel{expr.index}()
_scalar_kernel(expr::_ScalarConstant) = _ConstantKernel(expr.value)
_scalar_kernel(expr::_ScalarUnary) = _UnaryKernel{expr.op,typeof(_scalar_kernel(expr.arg))}(
    _scalar_kernel(expr.arg))
function _scalar_kernel(expr::_ScalarBinary)
    left, right = _scalar_kernel(expr.left), _scalar_kernel(expr.right)
    return _BinaryKernel{expr.op,typeof(left),typeof(right)}(left, right)
end

struct _MapArguments{E,P}
    element::E
    parameters::P
end
struct _FoldArguments{A,V,P}
    accumulator::A
    value::V
    parameters::P
end

@inline _kernel_feature(args::_MapArguments, ::Val{1}) = args.element
@inline _kernel_feature(args::_MapArguments, ::Val{I}) where {I} = args.parameters[I - 1]
@inline _kernel_feature(args::_FoldArguments, ::Val{1}) = args.accumulator
@inline _kernel_feature(args::_FoldArguments, ::Val{2}) = args.value
@inline _kernel_feature(args::_FoldArguments, ::Val{I}) where {I} = args.parameters[I - 2]

@inline _kernel_value(::_InputKernel{I}, args) where {I} = _kernel_feature(args, Val(I))
@inline _kernel_value(kernel::_ConstantKernel, args) = kernel.value
@inline function _kernel_value(kernel::_UnaryKernel{OP}, args) where {OP}
    value = _kernel_value(kernel.arg, args)
    OP === :sin && return sin(value)
    OP === :cos && return cos(value)
    OP === :tan && return _scalar_unary_value(:tan, value)
    OP === :exp && return exp(value)
    if OP === :log
        try
            return log(value)
        catch error
            error isa DomainError || rethrow()
            return NaN
        end
    end
    OP === :abs && return abs(value)
    OP === :sqrt && return value < 0 ? NaN : sqrt(value)
    return value * value
end
@inline function _kernel_value(kernel::_BinaryKernel{OP}, args) where {OP}
    left = _kernel_value(kernel.left, args)
    right = _kernel_value(kernel.right, args)
    OP === :+ && return left + right
    OP === :- && return left - right
    OP === :* && return left * right
    if OP === :^
        try
            return left ^ right
        catch error
            error isa DomainError || rethrow()
            return NaN
        end
    end
    OP === :max && return max(left, right)
    OP === :min && return min(left, right)
    return left / right
end

"""Prepared shape-preserving scalar map; feature 1 is the current element."""
struct ScalarMapPlan{N,P<:PreparedScalarProgram,K<:_ScalarKernel}
    program::P
    kernel::K
    parameter_count::Int
end

"""Reusable caller-owned buffers for dense map execution."""
mutable struct ScalarMapWorkspace
    features::Vector{Float64}
    values::Vector{Float64}
end

ScalarMapWorkspace() = ScalarMapWorkspace(Float64[], Float64[])

"""
    prepare_scalar_map(expr; parameter_count=0)

Prepare a direct scalar map. Input feature 1 denotes each scalar leaf and the
remaining features denote a repeated parameter block. General AST lowering is
not performed by this constructor.
"""
function prepare_scalar_map(expr::ScalarExpr; parameter_count::Int=0)
    parameter_count >= 0 || throw(ArgumentError("parameter_count cannot be negative"))
    prepared = prepare_scalar(expr)
    prepared.required_features <= 1 + parameter_count ||
        throw(ArgumentError("map expression requests more features than element + parameters"))
    kernel = _scalar_kernel(expr)
    return ScalarMapPlan{parameter_count,typeof(prepared),typeof(kernel)}(
        prepared, kernel, parameter_count)
end

function _check_parameters(parameters, expected)
    length(parameters) == expected || return Err(ScalarPlanShapeErr(:parameter_count))
    all(parameter -> parameter isa Real && isfinite(parameter), parameters) ||
        return Err(ScalarPlanShapeErr(:nonfinite_parameter))
    return nothing
end

"""
    execute_scalar_map!(output, plan, input, parameters, workspace)

Execute a dense shape-preserving map. Once the output and workspace have been
sized by a warm-up call, the finite hot path performs no per-element heap
allocation. Plans are shareable; workspaces are not thread-safe.
"""
function execute_scalar_map!(output::AbstractArray{Float64}, plan::ScalarMapPlan{N},
        input::AbstractArray{<:Real}, parameters=(), workspace::ScalarMapWorkspace=ScalarMapWorkspace()) where {N}
    axes(output) == axes(input) || return Err(ScalarPlanShapeErr(:map_output_shape))
    parameter_check = _check_parameters(parameters, plan.parameter_count)
    parameter_check isa Err && return parameter_check
    converted_parameters = ntuple(index -> Float64(parameters[index]), Val(N))
    finite = true
    @inbounds @simd for index in eachindex(input, output)
        value = _kernel_value(plan.kernel,
            _MapArguments(Float64(input[index]), converted_parameters))
        output[index] = value
        finite &= isfinite(value)
    end
    if !finite
        @inbounds for index in eachindex(output, input)
            isfinite(output[index]) && continue
            feature_count = 1 + plan.parameter_count
            length(workspace.features) < feature_count && resize!(workspace.features, feature_count)
            workspace.features[1] = Float64(input[index])
            for parameter_index in 1:plan.parameter_count
                workspace.features[parameter_index + 1] = converted_parameters[parameter_index]
            end
            return _evaluate_prepared_point!(
                plan.program, workspace.features, workspace.values, index)
        end
    end
    return output
end

function execute_scalar_map(plan::ScalarMapPlan, input::AbstractArray{<:Real}, parameters=())
    output = similar(input, Float64)
    return execute_scalar_map!(output, plan, input, parameters, ScalarMapWorkspace())
end

function _flatten_scalar_leaves!(output::Vector{Float64}, value)
    if value isa Real
        push!(output, Float64(value))
    elseif value isa AbstractArray
        for child in value
            result = _flatten_scalar_leaves!(output, child)
            result isa Err && return result
        end
    else
        return Err(ScalarPlanShapeErr(:nonreal_map_leaf))
    end
    return output
end

function _rebuild_scalar_shape(value::Real, flat, position::Base.RefValue{Int})
    position[] += 1
    return flat[position[]]
end

function _rebuild_scalar_shape(value::AbstractArray, flat, position::Base.RefValue{Int})
    return map(child -> _rebuild_scalar_shape(child, flat, position), value)
end

function execute_scalar_map(plan::ScalarMapPlan, input::AbstractArray, parameters=())
    flat = Float64[]
    flattened = _flatten_scalar_leaves!(flat, input)
    flattened isa Err && return flattened
    mapped = execute_scalar_map(plan, flat, parameters)
    mapped isa Err && return mapped
    return _rebuild_scalar_shape(input, mapped, Ref(0))
end

"""Prepared ordered scalar fold; features 1 and 2 are accumulator and item."""
struct ScalarFoldPlan{N,P<:PreparedScalarProgram,K<:_ScalarKernel}
    program::P
    kernel::K
    parameter_count::Int
end

"""Reusable caller-owned scratch for ordered fold execution."""
mutable struct ScalarFoldWorkspace
    features::Vector{Float64}
    values::Vector{Float64}
end

ScalarFoldWorkspace() = ScalarFoldWorkspace(Float64[], Float64[])

"""
    prepare_scalar_fold(expr; parameter_count=0)

Prepare an ordered left fold. Features 1 and 2 denote the accumulator and the
current item; later features denote repeated parameters. Associativity is not
assumed and execution is never reordered.
"""
function prepare_scalar_fold(expr::ScalarExpr; parameter_count::Int=0)
    parameter_count >= 0 || throw(ArgumentError("parameter_count cannot be negative"))
    prepared = prepare_scalar(expr)
    prepared.required_features <= 2 + parameter_count ||
        throw(ArgumentError("fold expression requests more features than accumulator + item + parameters"))
    kernel = _scalar_kernel(expr)
    return ScalarFoldPlan{parameter_count,typeof(prepared),typeof(kernel)}(
        prepared, kernel, parameter_count)
end

function _evaluate_prepared_point!(program::PreparedScalarProgram, features, values, step)
    instructions = program.instructions
    length(values) < length(instructions) && resize!(values, length(instructions))
    @inbounds for (slot, instruction) in pairs(instructions)
        result = if instruction.opcode === _OP_INPUT
            features[instruction.left]
        elseif instruction.opcode === _OP_CONSTANT
            instruction.value
        elseif instruction.opcode === _OP_SIN || instruction.opcode === _OP_COS ||
                instruction.opcode === _OP_SQRT || instruction.opcode === _OP_SQUARE ||
                instruction.opcode === _OP_TAN || instruction.opcode === _OP_EXP ||
                instruction.opcode === _OP_LOG || instruction.opcode === _OP_ABS
            _scalar_unary_value(instruction.opcode, values[instruction.left])
        else
            _scalar_binary_value(instruction.opcode, values[instruction.left],
                values[instruction.right])
        end
        isfinite(result) || return Err(ScalarDomainErr(_opcode_symbol(instruction.opcode), step))
        values[slot] = result
    end
    return values[length(instructions)]
end

function _prepare_fold_features!(workspace, parameter_count, parameters)
    parameter_check = _check_parameters(parameters, parameter_count)
    parameter_check isa Err && return parameter_check
    feature_count = 2 + parameter_count
    length(workspace.features) < feature_count && resize!(workspace.features, feature_count)
    @inbounds for index in 1:parameter_count
        workspace.features[index + 2] = Float64(parameters[index])
    end
    return workspace.features
end

function execute_scalar_fold(plan::ScalarFoldPlan{N}, input::AbstractVector{<:Real},
        initial::Real, parameters=(); workspace::ScalarFoldWorkspace=ScalarFoldWorkspace()) where {N}
    parameter_check = _check_parameters(parameters, plan.parameter_count)
    parameter_check isa Err && return parameter_check
    converted_parameters = ntuple(index -> Float64(parameters[index]), Val(N))
    accumulator = Float64(initial)
    isfinite(accumulator) || return Err(ScalarDomainErr(:initial, 0))
    @inbounds for (step, item) in pairs(input)
        value = _kernel_value(plan.kernel,
            _FoldArguments(accumulator, Float64(item), converted_parameters))
        if !isfinite(value)
            features = _prepare_fold_features!(workspace, plan.parameter_count, parameters)
            features isa Err && return features
            features[1], features[2] = accumulator, Float64(item)
            return _evaluate_prepared_point!(plan.program, features, workspace.values, step)
        end
        accumulator = value
    end
    return accumulator
end

function execute_scalar_fold(plan::ScalarFoldPlan{N}, input::AbstractVector,
        initial::AbstractArray{<:Real}, parameters=(); workspace::ScalarFoldWorkspace=ScalarFoldWorkspace()) where {N}
    parameter_check = _check_parameters(parameters, plan.parameter_count)
    parameter_check isa Err && return parameter_check
    converted_parameters = ntuple(index -> Float64(parameters[index]), Val(N))
    accumulator = Float64.(initial)
    all(isfinite, accumulator) || return Err(ScalarDomainErr(:initial, 0))
    for (step, item) in pairs(input)
        item isa AbstractArray || return Err(ScalarPlanShapeErr(:fold_item_shape))
        axes(item) == axes(accumulator) || return Err(ScalarPlanShapeErr(:fold_item_shape))
        @inbounds for index in eachindex(accumulator, item)
            value = _kernel_value(plan.kernel,
                _FoldArguments(accumulator[index], Float64(item[index]), converted_parameters))
            if !isfinite(value)
                features = _prepare_fold_features!(workspace, plan.parameter_count, parameters)
                features isa Err && return features
                features[1], features[2] = accumulator[index], Float64(item[index])
                return _evaluate_prepared_point!(plan.program, features, workspace.values, step)
            end
            accumulator[index] = value
        end
    end
    return accumulator
end
