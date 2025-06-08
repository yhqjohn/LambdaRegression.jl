"""
    ScalarSearch

Derived real-valued scalar expressions and conventional symbolic-regression
search inside the ordinary typed program system. The authoritative
representation remains an AST; scalar expressions are rebuildable caches that
can be converted from supported AST regions and materialised transactionally
into fixed program scaffolds.

The module provides reference and prepared batched evaluation, a
DynamicExpressions bridge, bounded generational search, and node-native
regularized evolution with age replacement, rank tournaments, annealing,
adaptive complexity frequencies, and a best-per-complexity archive. The
`search_multi_scalar` coordinator advances persistent regularized islands with
bounded inter-island migration. Its `initialize_multi_scalar`,
`advance_multi_scalar!`, and `multi_scalar_result` APIs retain a fixed-context
session across complete coordinator rounds. Its default `hof_donor_policy=:local_best`
draws from bounded per-island archive leaders and excludes the destination;
`:global_pareto` instead draws from the global loss-complexity frontier,
including the destination's own history. Both retain the same migration clock,
without-replacement donor sampling and random-slot replacement with fresh ages.
The public strategies share scalar configuration and independently bounded
constant polishing while retaining their own search loops. Constant polishing
stops at its evaluation cap and keeps the best valid trial already scored;
resource exhaustion is not treated as an infinite objective value. Initial
candidates are not deeply polished. After structural search, the generational
strategy may spend its remaining refinement budget on evolved candidates from
the training loss–complexity frontier. The regularized strategy defaults to
probabilistic cycle-end refinements of live members and can instead refine
scored offspring before archive/admission. Its optional `max_total_evaluations`
cap counts structural and constant-refinement scores together while excluding
each island's baseline callback. Its best-per-complexity archive is retained and
returned within that search state, not reused by a later search call. Results
retain scored expression forms without implicit report-time simplification.
Regularized cycle simplification proposes a new expression and scores it against
the current structural and total budgets; its finite score may enter the archive,
and it replaces a live member only when its loss is no worse, retaining its age.
The module also provides direct shape-preserving `MapPlan` and
ordered `FoldPlan` adapters. A minimal evaluation path is:

```julia
expr = scalar_binary(:*, scalar_input(1),
    scalar_unary(:sin, scalar_binary(:-, scalar_input(2), scalar_input(1))))
y = evaluate_scalar(expr, X) # X is features × samples
```

Use `search_scalar(X, y; config=ScalarSearchConfig(...))` for independent
scalar teachers, set `evolution_strategy=:regularized` for the steady-state
path, or pass a contextual objective for a fixed program scaffold. Set
`constant_refinement_timing=:offspring` to compare pre-admission polishing.
Expected input-shape, numeric-domain, conversion, scaffold, and plan-shape
failures are returned as `Err`; unsupported AST regions remain on the general
program path.
"""
module ScalarSearch

using TestItems
using Random: AbstractRNG, MersenneTwister, default_rng, rand, randn, shuffle!
using Optim
using ..Utils: Err
using ..ASTs.NodeASTs: AstNode, AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode
using ..TypeSystem: FloatTy, TyLike
using ..SearchPolicies: _RegularizedFrequency, _regularized_move_window!,
    _regularized_normalize!, _regularized_frequency,
    _regularized_acceptance_probability

export ScalarExpr, scalar_input, scalar_constant, scalar_unary, scalar_binary
export evaluate_scalar_reference, scalar_from_ast, scalar_to_ast
export PreparedScalarProgram, ScalarWorkspace, prepare_scalar, evaluate_scalar, evaluate_scalar!
export ScalarSearchConfig, ScalarSearchResult, scalar_complexity
export scalar_structure_similarity, search_scalar
export MultiScalarSession, initialize_multi_scalar, advance_multi_scalar!, multi_scalar_result
export multi_scalar_progress
export search_multi_scalar
export FixedScalarCall, FixedScalarScaffold, ScalarMaterialization, materialize_scalar_scaffold
export ScalarScaffoldErr
export ScalarMapPlan, ScalarMapWorkspace, prepare_scalar_map, execute_scalar_map, execute_scalar_map!
export ScalarFoldPlan, ScalarFoldWorkspace, prepare_scalar_fold, execute_scalar_fold
export ScalarPlanShapeErr
export ScalarInputShapeErr, ScalarDomainErr, ScalarConversionErr

"""Base type for rebuildable real-valued scalar expressions."""
abstract type ScalarExpr end

struct _ScalarInput <: ScalarExpr
    index::Int
end

struct _ScalarConstant <: ScalarExpr
    value::Float64
end

struct _ScalarUnary <: ScalarExpr
    op::Symbol
    arg::ScalarExpr
end

struct _ScalarBinary <: ScalarExpr
    op::Symbol
    left::ScalarExpr
    right::ScalarExpr
end

_same_scalar_expression(::_ScalarInput, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarConstant, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarUnary, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarBinary, ::ScalarExpr) = false
_same_scalar_expression(left::_ScalarInput, right::_ScalarInput) =
    left.index == right.index
_same_scalar_expression(left::_ScalarConstant, right::_ScalarConstant) =
    left.value == right.value
_same_scalar_expression(left::_ScalarUnary, right::_ScalarUnary) =
    left.op === right.op && _same_scalar_expression(left.arg, right.arg)
_same_scalar_expression(left::_ScalarBinary, right::_ScalarBinary) =
    left.op === right.op &&
    _same_scalar_expression(left.left, right.left) &&
    _same_scalar_expression(left.right, right.right)

const _UNARY_AST = Dict(
    :sin => :FSIN, :cos => :FCOS, :sqrt => :FSQRT, :square => :SQR,
    :tan => :FTAN, :exp => :FEXP, :log => :FLOG, :abs => :FABS,
)
const _BINARY_AST = Dict(
    :+ => :FADD, :- => :FSUB, :* => :FMUL, :/ => :FDIV,
    :^ => :FPOW, :max => :FMAX, :min => :FMIN,
)
const _AST_UNARY = Dict(value => key for (key, value) in _UNARY_AST)
const _AST_BINARY = Dict(value => key for (key, value) in _BINARY_AST)

"""The feature matrix does not contain every feature requested by the expression."""
struct ScalarInputShapeErr
    required_features::Int
    actual_features::Int
end

"""A scalar operator failed or produced a non-finite value at one sample."""
struct ScalarDomainErr
    op::Symbol
    sample::Int
end

"""An AST region is outside the scalar language, or cannot be materialised."""
struct ScalarConversionErr
    reason::Symbol
end

"""Create a one-based scalar feature reference."""
function scalar_input(index::Integer)
    index >= 1 || throw(ArgumentError("scalar input indices are one-based and must be positive"))
    return _ScalarInput(Int(index))
end

"""Create an ephemeral finite `Float64` scalar constant."""
function scalar_constant(value::Real)
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("scalar constants must be finite"))
    return _ScalarConstant(converted)
end

"""Create a unary expression for the supported real scalar functions."""
function scalar_unary(op::Symbol, arg::ScalarExpr)
    haskey(_UNARY_AST, op) || throw(ArgumentError("unsupported scalar unary operator: $op"))
    return _ScalarUnary(op, arg)
end

"""Create a binary expression for arithmetic, power, max, or min."""
function scalar_binary(op::Symbol, left::ScalarExpr, right::ScalarExpr)
    haskey(_BINARY_AST, op) || throw(ArgumentError("unsupported scalar binary operator: $op"))
    return _ScalarBinary(op, left, right)
end

function _required_features(expr::_ScalarInput)
    return expr.index
end
_required_features(::_ScalarConstant) = 0
_required_features(expr::_ScalarUnary) = _required_features(expr.arg)
_required_features(expr::_ScalarBinary) = max(_required_features(expr.left), _required_features(expr.right))

function _eval_reference(expr::_ScalarInput, X, sample)
    value = Float64(X[expr.index, sample])
    return isfinite(value) ? value : Err(ScalarDomainErr(:input, sample))
end
_eval_reference(expr::_ScalarConstant, X, sample) = expr.value

@inline function _scalar_unary_value(op::Symbol, value::Float64)
    try
        return op === :sin ? sin(value) :
            op === :cos ? cos(value) :
            op === :tan ? tan(value) :
            op === :exp ? exp(value) :
            op === :log ? log(value) :
            op === :abs ? abs(value) :
            op === :sqrt ? sqrt(value) :
            value * value
    catch error
        error isa DomainError || rethrow()
        return NaN
    end
end

@inline function _scalar_binary_value(op::Symbol, left::Float64, right::Float64)
    try
        return op === :+ ? left + right :
            op === :- ? left - right :
            op === :* ? left * right :
            op === :/ ? left / right :
            op === :^ ? left ^ right :
            op === :max ? max(left, right) : min(left, right)
    catch error
        error isa DomainError || rethrow()
        return NaN
    end
end

function _eval_reference(expr::_ScalarUnary, X, sample)
    value = _eval_reference(expr.arg, X, sample)
    value isa Err && return value
    result = _scalar_unary_value(expr.op, value)
    return isfinite(result) ? result : Err(ScalarDomainErr(expr.op, sample))
end

function _eval_reference(expr::_ScalarBinary, X, sample)
    left = _eval_reference(expr.left, X, sample)
    left isa Err && return left
    right = _eval_reference(expr.right, X, sample)
    right isa Err && return right
    result = _scalar_binary_value(expr.op, left, right)
    return isfinite(result) ? result : Err(ScalarDomainErr(expr.op, sample))
end

"""
    evaluate_scalar_reference(expr, X) -> Vector{Float64} | Err

Evaluate `expr` sample by sample over a `features × samples` real matrix. This
path favours transparent semantics over speed and reports the first domain or
non-finite failure with its one-based sample index.
"""
function evaluate_scalar_reference(expr::ScalarExpr, X::AbstractMatrix{<:Real})
    required = _required_features(expr)
    size(X, 1) >= required || return Err(ScalarInputShapeErr(required, size(X, 1)))
    output = Vector{Float64}(undef, size(X, 2))
    for sample in axes(X, 2)
        value = _eval_reference(expr, X, sample)
        value isa Err && return value
        output[sample] = value
    end
    return output
end

function _dependency_index(node::AstNode, dependencies::AbstractVector{<:AstNode})
    return findfirst(dependency -> dependency == node, dependencies)
end

function _scalar_from_ast(node::AstNode, dependencies, constant_values)
    dependency = _dependency_index(node, dependencies)
    dependency === nothing || return scalar_input(dependency)

    if node isa ConstNode
        if haskey(constant_values, node.name)
            value = constant_values[node.name]
            value isa Real || return Err(ScalarConversionErr(:nonreal_constant))
            converted = Float64(value)
            isfinite(converted) || return Err(ScalarConversionErr(:nonfinite_constant))
            return scalar_constant(converted)
        end
        return Err(ScalarConversionErr(:unsupported_constant))
    elseif node isa AppNode && node.func isa ConstNode && haskey(_AST_UNARY, node.func.name)
        arg = _scalar_from_ast(node.arg, dependencies, constant_values)
        arg isa Err && return arg
        return scalar_unary(_AST_UNARY[node.func.name], arg)
    elseif node isa AppNode && node.func isa AppNode &&
            node.func.func isa ConstNode && haskey(_AST_BINARY, node.func.func.name)
        left = _scalar_from_ast(node.func.arg, dependencies, constant_values)
        left isa Err && return left
        right = _scalar_from_ast(node.arg, dependencies, constant_values)
        right isa Err && return right
        return scalar_binary(_AST_BINARY[node.func.func.name], left, right)
    end
    return Err(ScalarConversionErr(:unsupported_ast))
end

"""
    scalar_from_ast(node, dependencies; constant_values=Dict()) -> ScalarExpr | Err

Recognise a supported real scalar region. `dependencies` defines the ordered
feature boundary; named finite real constants may be supplied separately.
"""
function scalar_from_ast(node::AstNode, dependencies::AbstractVector{<:AstNode};
        constant_values=Dict{Symbol,Float64}())
    return _scalar_from_ast(node, dependencies, constant_values)
end

_default_constant_materializer(value) = nothing

function _scalar_to_ast(expr::_ScalarInput, dependencies, materialize_constant)
    expr.index <= length(dependencies) ||
        return Err(ScalarConversionErr(:missing_dependency))
    return deepcopy(dependencies[expr.index])
end

function _scalar_to_ast(expr::_ScalarConstant, dependencies, materialize_constant)
    node = materialize_constant(expr.value)
    return node isa AstNode ? node : Err(ScalarConversionErr(:unmaterialized_constant))
end

function _scalar_to_ast(expr::_ScalarUnary, dependencies, materialize_constant)
    arg = _scalar_to_ast(expr.arg, dependencies, materialize_constant)
    arg isa Err && return arg
    return AppNode(ConstNode(_UNARY_AST[expr.op]), arg)
end

function _scalar_to_ast(expr::_ScalarBinary, dependencies, materialize_constant)
    left = _scalar_to_ast(expr.left, dependencies, materialize_constant)
    left isa Err && return left
    right = _scalar_to_ast(expr.right, dependencies, materialize_constant)
    right isa Err && return right
    return AppNode(AppNode(ConstNode(_BINARY_AST[expr.op]), left), right)
end

"""
    scalar_to_ast(expr, dependencies; materialize_constant=value -> nothing)

Materialise a scalar expression as an ordinary AST. Feature dependencies are
deep-copied. An expression containing ephemeral constants requires a callback
that owns their eventual names and returns a `ConstNode` (or another AST node).
"""
function scalar_to_ast(expr::ScalarExpr, dependencies::AbstractVector{<:AstNode};
        materialize_constant=_default_constant_materializer)
    return _scalar_to_ast(expr, dependencies, materialize_constant)
end

@enum _ScalarOpcode::UInt8 begin
    _OP_INPUT
    _OP_CONSTANT
    _OP_SIN
    _OP_COS
    _OP_SQRT
    _OP_SQUARE
    _OP_ADD
    _OP_SUBTRACT
    _OP_MULTIPLY
    _OP_DIVIDE
    _OP_TAN
    _OP_EXP
    _OP_LOG
    _OP_ABS
    _OP_POWER
    _OP_MAX
    _OP_MIN
end

@inline function _opcode_symbol(opcode::_ScalarOpcode)
    opcode === _OP_INPUT && return :input
    opcode === _OP_CONSTANT && return :constant
    opcode === _OP_SIN && return :sin
    opcode === _OP_COS && return :cos
    opcode === _OP_SQRT && return :sqrt
    opcode === _OP_SQUARE && return :square
    opcode === _OP_ADD && return :+
    opcode === _OP_SUBTRACT && return :-
    opcode === _OP_MULTIPLY && return :*
    opcode === _OP_DIVIDE && return :/
    opcode === _OP_TAN && return :tan
    opcode === _OP_EXP && return :exp
    opcode === _OP_LOG && return :log
    opcode === _OP_ABS && return :abs
    opcode === _OP_POWER && return :^
    opcode === _OP_MAX && return :max
    opcode === _OP_MIN && return :min
    return :unknown
end

@inline function _scalar_unary_value(opcode::_ScalarOpcode, value::Float64)
    try
        return opcode === _OP_SIN ? sin(value) :
            opcode === _OP_COS ? cos(value) :
            opcode === _OP_TAN ? tan(value) :
            opcode === _OP_EXP ? exp(value) :
            opcode === _OP_LOG ? log(value) :
            opcode === _OP_ABS ? abs(value) :
            opcode === _OP_SQRT ? sqrt(value) : value * value
    catch error
        error isa DomainError || rethrow()
        return NaN
    end
end

@inline function _scalar_binary_value(opcode::_ScalarOpcode,
        left::Float64, right::Float64)
    try
        return opcode === _OP_ADD ? left + right :
            opcode === _OP_SUBTRACT ? left - right :
            opcode === _OP_MULTIPLY ? left * right :
            opcode === _OP_POWER ? left ^ right :
            opcode === _OP_MAX ? max(left, right) :
            opcode === _OP_MIN ? min(left, right) : left / right
    catch error
        error isa DomainError || rethrow()
        return NaN
    end
end

struct _ScalarInstruction
    opcode::_ScalarOpcode
    left::Int
    right::Int
    value::Float64
end

"""
    PreparedScalarProgram

Immutable, thread-safe instruction plan compiled from a `ScalarExpr`. The plan
contains no dataset or mutable scratch state and may therefore be shared across
workers. Use one [`ScalarWorkspace`](@ref) per concurrent evaluation.
"""
struct PreparedScalarProgram
    instructions::Tuple{Vararg{_ScalarInstruction}}
    required_features::Int
    source::ScalarExpr
end

"""Reusable, caller-owned scratch storage for prepared scalar evaluation."""
mutable struct ScalarWorkspace
    values::Vector{Float64}
end

ScalarWorkspace() = ScalarWorkspace(Float64[])

function _compile_scalar!(instructions, expr::_ScalarInput)
    push!(instructions, _ScalarInstruction(_OP_INPUT, expr.index, 0, 0.0))
    return length(instructions)
end

function _compile_scalar!(instructions, expr::_ScalarConstant)
    push!(instructions, _ScalarInstruction(_OP_CONSTANT, 0, 0, expr.value))
    return length(instructions)
end

function _compile_scalar!(instructions, expr::_ScalarUnary)
    arg = _compile_scalar!(instructions, expr.arg)
    opcode = expr.op === :sin ? _OP_SIN :
        expr.op === :cos ? _OP_COS :
        expr.op === :tan ? _OP_TAN :
        expr.op === :exp ? _OP_EXP :
        expr.op === :log ? _OP_LOG :
        expr.op === :abs ? _OP_ABS :
        expr.op === :sqrt ? _OP_SQRT : _OP_SQUARE
    push!(instructions, _ScalarInstruction(opcode, arg, 0, 0.0))
    return length(instructions)
end

function _compile_scalar!(instructions, expr::_ScalarBinary)
    left = _compile_scalar!(instructions, expr.left)
    right = _compile_scalar!(instructions, expr.right)
    opcode = expr.op === :+ ? _OP_ADD :
        expr.op === :- ? _OP_SUBTRACT :
        expr.op === :* ? _OP_MULTIPLY :
        expr.op === :^ ? _OP_POWER :
        expr.op === :max ? _OP_MAX :
        expr.op === :min ? _OP_MIN : _OP_DIVIDE
    push!(instructions, _ScalarInstruction(opcode, left, right, 0.0))
    return length(instructions)
end

"""Compile a scalar expression into a rebuildable, dataset-independent plan."""
function prepare_scalar(expr::ScalarExpr)
    instructions = _ScalarInstruction[]
    _compile_scalar!(instructions, expr)
    return PreparedScalarProgram(Tuple(instructions), _required_features(expr), expr)
end

@inline _slot_start(slot::Int, samples::Int) = (slot - 1) * samples

"""
    evaluate_scalar!(output, prepared, X, workspace) -> output | Err

Execute a prepared plan over a `features × samples` matrix without recursive
per-sample tree dispatch. `output` must have one entry per sample. The workspace
is reusable but deliberately not thread-safe; concurrent calls use distinct
workspaces.
"""
function evaluate_scalar!(output::AbstractVector{Float64}, prepared::PreparedScalarProgram,
        X::AbstractMatrix{<:Real}, workspace::ScalarWorkspace)
    samples = size(X, 2)
    length(output) == samples ||
        throw(DimensionMismatch("scalar output length must equal the sample count"))
    size(X, 1) >= prepared.required_features ||
        return Err(ScalarInputShapeErr(prepared.required_features, size(X, 1)))

    required_values = length(prepared.instructions) * samples
    length(workspace.values) < required_values && resize!(workspace.values, required_values)
    values = workspace.values
    invalid = false

    for (slot, instruction) in pairs(prepared.instructions)
        destination = _slot_start(slot, samples)
        if instruction.opcode === _OP_INPUT
            feature = instruction.left
            @inbounds for sample in 1:samples
                value = Float64(X[feature, sample])
                if isfinite(value)
                    values[destination + sample] = value
                else
                    invalid = true
                    values[destination + sample] = NaN
                end
            end
        elseif instruction.opcode === _OP_CONSTANT
            @inbounds for sample in 1:samples
                values[destination + sample] = instruction.value
            end
        elseif instruction.opcode === _OP_SIN || instruction.opcode === _OP_COS ||
                instruction.opcode === _OP_SQRT ||
                instruction.opcode === _OP_SQUARE ||
                instruction.opcode === _OP_TAN || instruction.opcode === _OP_EXP ||
                instruction.opcode === _OP_LOG || instruction.opcode === _OP_ABS
            source = _slot_start(instruction.left, samples)
            @inbounds for sample in 1:samples
                value = values[source + sample]
                result = _scalar_unary_value(instruction.opcode, value)
                if isfinite(result)
                    values[destination + sample] = result
                else
                    invalid = true
                    values[destination + sample] = NaN
                end
            end
        else
            left = _slot_start(instruction.left, samples)
            right = _slot_start(instruction.right, samples)
            @inbounds for sample in 1:samples
                left_value = values[left + sample]
                right_value = values[right + sample]
                result = _scalar_binary_value(instruction.opcode, left_value,
                    right_value)
                if isfinite(result)
                    values[destination + sample] = result
                else
                    invalid = true
                    values[destination + sample] = NaN
                end
            end
        end
    end

    invalid && return evaluate_scalar_reference(prepared.source, X)
    root = _slot_start(length(prepared.instructions), samples)
    copyto!(output, 1, values, root + 1, samples)
    return output
end

"""Allocate the output and scratch storage for one prepared scalar evaluation."""
function evaluate_scalar(prepared::PreparedScalarProgram, X::AbstractMatrix{<:Real})
    output = Vector{Float64}(undef, size(X, 2))
    return evaluate_scalar!(output, prepared, X, ScalarWorkspace())
end

evaluate_scalar(expr::ScalarExpr, X::AbstractMatrix{<:Real}) =
    evaluate_scalar(prepare_scalar(expr), X)

include("search.jl")
include("dynamic_expressions.jl")
include("node_regularized.jl")
include("node_composition_cleanup.jl")
include("node_constant_refinement.jl")
include("node_regularized_search.jl")
include("node_regularized_state.jl")
include("multi_island.jl")
include("scaffolds.jl")
include("plans.jl")

@testsnippet ScalarSearchBaseImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
    using LambdaRegression.ASTs.NodeASTs: AppNode, ConstNode
end

@testitem "Scalar expression equality preserves scalar value semantics" setup=[ScalarSearchBaseImports] begin
    left = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    same = scalar_binary(:+, scalar_input(1), scalar_constant(-0.0))
    different = scalar_binary(:+, scalar_input(2), scalar_constant(0.0))
    @test ScalarSearch._same_scalar_expression(left, same)
    @test !ScalarSearch._same_scalar_expression(left, different)
end

@testitem "Scalar reference evaluator agrees with a worked batch and round trips AST" setup=[ScalarSearchBaseImports] begin
    expr = scalar_binary(:*, scalar_input(1),
        scalar_unary(:sin, scalar_binary(:-, scalar_input(2), scalar_input(1))))
    X = [1.0 2.0 3.0; 4.0 6.0 8.0]

    @test evaluate_scalar_reference(expr, X) ≈ X[1, :] .* sin.(X[2, :] .- X[1, :])

    dependencies = [ConstNode(:x1), ConstNode(:x2)]
    ast = AppNode(
        AppNode(ConstNode(:FMUL), dependencies[1]),
        AppNode(ConstNode(:FSIN),
            AppNode(AppNode(ConstNode(:FSUB), dependencies[2]), dependencies[1])),
    )
    parsed = scalar_from_ast(ast, dependencies)
    @test parsed isa ScalarExpr
    @test scalar_to_ast(parsed, dependencies) == ast
end

@testitem "Scalar reference evaluator classifies domain failures" setup=[ScalarSearchBaseImports] begin
    result = evaluate_scalar_reference(scalar_unary(:sqrt, scalar_input(1)),
        reshape([-1.0, 4.0], 1, :))

    @test result == LambdaRegression.Utils.Err(ScalarDomainErr(:sqrt, 1))
end

@testitem "Scalar cosine is an authoritative AST primitive" setup=[ScalarSearchBaseImports] begin
    expr = scalar_unary(:cos, scalar_input(1))
    X = reshape([0.0, 0.5, 1.0], 1, :)
    ast = AppNode(ConstNode(:FCOS), ConstNode(:x))

    @test evaluate_scalar(expr, X) ≈ cos.(X[1, :])
    @test LambdaRegression.TypeSystem.get_type(LambdaRegression.LambdaRegression.Primitives.FCOS) ==
        LambdaRegression.TypeSystem.TyFun(
            LambdaRegression.TypeSystem.FloatTy,
            LambdaRegression.TypeSystem.FloatTy,
        )
    @test LambdaRegression.CanonicalGeneration.test_catalog_consts()[:FCOS] ===
        LambdaRegression.LambdaRegression.Primitives.FCOS
    @test scalar_from_ast(ast, [ConstNode(:x)]) isa ScalarExpr
    @test scalar_to_ast(expr, [ConstNode(:x)]) == ast
end

@testitem "Extended scalar operators round-trip through AST and node backends" setup=[ScalarSearchBaseImports] begin
    unary_cases = (
        (:tan, :FTAN, [0.25, 0.5], tan.([0.25, 0.5])),
        (:exp, :FEXP, [0.0, 1.0], exp.([0.0, 1.0])),
        (:log, :FLOG, [0.5, 2.0], log.([0.5, 2.0])),
        (:abs, :FABS, [-0.5, 2.0], abs.([-0.5, 2.0])),
    )
    binary_cases = (
        (:^, :FPOW, [2.0, 3.0], [3.0, 2.0], [8.0, 9.0]),
        (:max, :FMAX, [2.0, 4.0], [3.0, 1.0], [3.0, 4.0]),
        (:min, :FMIN, [2.0, 4.0], [3.0, 1.0], [2.0, 1.0]),
    )
    dependencies = [ConstNode(:x), ConstNode(:y)]

    for op in (:sinh, :cosh, :tanh, :log10, :sign, :floor, :ceil)
        @test_throws ArgumentError scalar_unary(op, scalar_input(1))
    end

    for (op, primitive_name, values, expected) in unary_cases
        expr = scalar_unary(op, scalar_input(1))
        X = reshape(values, 1, :)
        ast = AppNode(ConstNode(primitive_name), dependencies[1])
        @test evaluate_scalar_reference(expr, X) ≈ expected
        @test evaluate_scalar(expr, X) ≈ expected
        @test evaluate_dynamic_scalar(expr, X) ≈ expected
        node = scalar_to_dynamic(expr)
        @test ScalarSearch.scalar_from_dynamic(node) isa ScalarExpr
        @test evaluate_dynamic_scalar(node, X) ≈ expected
        materialized = scalar_to_ast(expr, dependencies)
        @test materialized == ast
        rebuilt = scalar_from_ast(materialized, dependencies)
        @test ScalarSearch._same_scalar_expression(rebuilt, expr)
    end

    for (op, primitive_name, left, right, expected) in binary_cases
        expr = scalar_binary(op, scalar_input(1), scalar_input(2))
        X = [left'; right']
        ast = AppNode(AppNode(ConstNode(primitive_name), dependencies[1]), dependencies[2])
        @test evaluate_scalar_reference(expr, X) ≈ expected
        @test evaluate_scalar(expr, X) ≈ expected
        @test evaluate_dynamic_scalar(expr, X) ≈ expected
        node = scalar_to_dynamic(expr)
        @test ScalarSearch.scalar_from_dynamic(node) isa ScalarExpr
        @test evaluate_dynamic_scalar(node, X) ≈ expected
        materialized = scalar_to_ast(expr, dependencies)
        @test materialized == ast
        rebuilt = scalar_from_ast(materialized, dependencies)
        @test ScalarSearch._same_scalar_expression(rebuilt, expr)
    end
end

@testitem "Extended scalar operators classify real-domain and non-finite failures" setup=[ScalarSearchBaseImports] begin
    cases = (
        (scalar_unary(:log, scalar_input(1)), reshape([-1.0, 2.0], 1, :), :log),
        (scalar_unary(:exp, scalar_input(1)), reshape([1000.0, 0.0], 1, :), :exp),
        (scalar_binary(:^, scalar_input(1), scalar_input(2)),
            [-2.0 2.0; 0.5 3.0], :^),
    )
    for (expr, X, op) in cases
        expected = LambdaRegression.Utils.Err(ScalarDomainErr(op, 1))
        @test evaluate_scalar_reference(expr, X) == expected
        @test evaluate_scalar(expr, X) == expected
        @test evaluate_dynamic_scalar(expr, X) == expected
    end

    # A non-finite intermediate must remain an expected error in the node path.
    nested = scalar_unary(:tan, scalar_unary(:exp, scalar_input(1)))
    @test evaluate_dynamic_scalar(nested, reshape([1000.0], 1, :)) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:exp, 1))
end

@testitem "Prepared scalar evaluator agrees with the reference corpus" setup=[ScalarSearchBaseImports] begin
    x1, x2 = scalar_input(1), scalar_input(2)
    expressions = ScalarExpr[
        x1,
        scalar_constant(2.5),
        scalar_unary(:sin, scalar_binary(:-, x2, x1)),
        scalar_unary(:sqrt,
            scalar_binary(:+, scalar_unary(:square, x1), scalar_constant(0.25))),
        scalar_binary(:/,
            scalar_binary(:*, x1, scalar_unary(:sin, x2)),
            scalar_binary(:+, scalar_unary(:square, x2), scalar_constant(1.0))),
    ]
    X = randn(MersenneTwister(1111), 2, 257)
    workspace = ScalarWorkspace()

    for expr in expressions
        reference = evaluate_scalar_reference(expr, X)
        prepared = prepare_scalar(expr)
        output = zeros(size(X, 2))
        @test evaluate_scalar!(output, prepared, X, workspace) ≈ reference
        @test evaluate_scalar(prepared, X) ≈ reference
    end

    failing = prepare_scalar(scalar_unary(:sqrt, x1))
    bad_X = reshape([1.0, -1.0], 1, :)
    @test evaluate_scalar(failing, bad_X) ==
        evaluate_scalar_reference(scalar_unary(:sqrt, x1), bad_X)

    competing_failures = scalar_binary(:+,
        scalar_unary(:sqrt, x1), scalar_unary(:sqrt, x2))
    crossed_X = [1.0 -1.0; -1.0 1.0]
    @test evaluate_scalar(prepare_scalar(competing_failures), crossed_X) ==
        evaluate_scalar_reference(competing_failures, crossed_X)
end

@testitem "Prepared scalar plans erase instruction length from their type" setup=[ScalarSearchBaseImports] begin
    short = prepare_scalar(scalar_binary(:+, scalar_input(1), scalar_input(2)))
    long = prepare_scalar(scalar_binary(:*,
        scalar_binary(:+, scalar_input(1), scalar_constant(1.0)),
        scalar_unary(:sin, scalar_input(2))))

    @test length(short.instructions) != length(long.instructions)
    @test typeof(short) === typeof(long)
end

@testitem "Scalar structural similarity distinguishes equal-size topology" setup=[ScalarSearchBaseImports] begin
    branched = scalar_binary(:+, scalar_input(1), scalar_constant(1.0))
    nested = scalar_unary(:sin, scalar_unary(:square, scalar_input(1)))

    @test scalar_complexity(branched) == scalar_complexity(nested)
    @test scalar_structure_similarity(branched, nested) < 1.0
    @test scalar_structure_similarity(branched, branched) == 1.0
end

@testitem "Prepared scalar evaluator matches generated finite expressions" setup=[ScalarSearchBaseImports] begin
    rng = MersenneTwister(1234)
    function finite_expression(depth)
        if depth == 0
            return rand(rng, Bool) ? scalar_input(rand(rng, 1:3)) :
                scalar_constant(rand(rng) * 2 - 1)
        end
        choice = rand(rng, 1:7)
        if choice == 1
            return scalar_unary(:sin, finite_expression(depth - 1))
        elseif choice == 2
            return scalar_unary(:square, finite_expression(depth - 1))
        elseif choice == 3
            child = finite_expression(depth - 1)
            return scalar_unary(:sqrt,
                scalar_binary(:+, scalar_unary(:square, child), scalar_constant(0.25)))
        elseif choice <= 6
            operator = (:+, :-, :*)[choice - 3]
            return scalar_binary(operator,
                finite_expression(depth - 1), finite_expression(depth - 1))
        end
        denominator = finite_expression(depth - 1)
        return scalar_binary(:/, finite_expression(depth - 1),
            scalar_binary(:+, scalar_unary(:square, denominator), scalar_constant(1.0)))
    end

    X = rand(rng, 3, 129) .* 2 .- 1
    workspace = ScalarWorkspace()
    for _ in 1:64
        expr = finite_expression(4)
        expected = evaluate_scalar_reference(expr, X)
        actual = zeros(size(X, 2))
        @test evaluate_scalar!(actual, prepare_scalar(expr), X, workspace) ≈ expected
    end

    empty_X = Matrix{Float64}(undef, 3, 0)
    @test evaluate_scalar(prepare_scalar(finite_expression(2)), empty_X) == Float64[]
end

@testitem "Scalar population search discovers a multistep polynomial" setup=[ScalarSearchBaseImports] begin
    rng = MersenneTwister(1234)
    X = rand(rng, 2, 128) .* 4 .- 2
    y = @. X[1, :] * X[2, :] + X[1, :]
    config = ScalarSearchConfig(
        population_size=48,
        max_evaluations=4_000,
        max_complexity=15,
        initial_depth=3,
        unary_operators=[:sin, :square],
        binary_operators=[:+, :-, :*],
        constant_probability=0.1,
        constant_refinement_iterations=0,
        initial_affine_basis=false,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(1234))

    @test result.best_loss <= 1.0e-12
    @test result.evaluations <= config.max_evaluations
    @test length(result.improvements) >= 2
    @test result.generations >= 2
    @test scalar_complexity(result.best) >= 5
    @test evaluate_scalar(result.best, X) ≈ y atol=1.0e-6

    repeated = search_scalar(X, y; config, rng=MersenneTwister(1234))
    @test repeated.best_loss == result.best_loss
    @test repeated.evaluations == result.evaluations
    @test repeated.generations == result.generations
    @test scalar_to_ast(repeated.best, [ConstNode(:x1), ConstNode(:x2)]) ==
        scalar_to_ast(result.best, [ConstNode(:x1), ConstNode(:x2)])
end

@testitem "Scalar search defaults keep structure ahead of constant polishing" setup=[ScalarSearchBaseImports] begin
    config = ScalarSearchConfig()

    @test config.evolution_strategy === :generational
    @test config.constant_refinement_iterations == 8
    @test config.constant_refinement_elites == 2
    @test config.constant_refinement_interval == 25
    @test config.constant_refinement_start_generation == 25
    @test config.constant_refinement_probability == 0.0
    @test config.constant_refinement_timing === :cycle
    @test config.max_constant_refinement_evaluations == 10_000
    @test config.max_total_evaluations === nothing
    @test config.constant_refinement_optimizer == :bfgs
    @test config.complexity_balanced_tournament
    @test config.mutation_probability == 0.9
    @test config.crossover_probability == 0.05
    @test config.random_restart_probability == 0.0
    @test !config.initial_affine_basis
end

@testitem "Scalar search selects regularized evolution through its public configuration" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=24)), 1, :)
    y = @. X[1, :]^2 + X[1, :]
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=40,
        max_complexity=10,
        initial_depth=2,
        tournament_size=3,
        elite_count=2,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        constant_refinement_iterations=0,
        initial_pool_multiplier=1,
        evolution_strategy=:regularized,
    )

    first_result = search_scalar(
        X, y; config, rng=MersenneTwister(8000), stop_loss=nothing)
    second_result = search_scalar(
        X, y; config, rng=MersenneTwister(8000), stop_loss=nothing)

    @test first_result.evaluations == config.max_evaluations
    @test first_result.termination === :evaluation_cap
    @test first_result.variation.oldest_replacements > 0
    @test first_result.best_loss == second_result.best_loss
    @test scalar_to_ast(first_result.best, [ConstNode(:x)]) ==
        scalar_to_ast(second_result.best, [ConstNode(:x)])
end

@testitem "Scalar search can defer early stopping for archive selection" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=24)), 1, :)
    y = copy(X[1, :])
    config = ScalarSearchConfig(
        population_size=6,
        max_evaluations=24,
        max_complexity=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=1,
        unary_operators=[:square],
        binary_operators=[:+],
        constant_refinement_iterations=0,
        initial_pool_multiplier=1,
    )

    result = search_scalar(X, y;
        config, rng=MersenneTwister(1), stop_loss=nothing)

    @test result.evaluations == config.max_evaluations
    @test result.termination === :evaluation_cap
    @test result.best_loss <= 1.0e-12
end

@testitem "Scalar search exposes random restart mutation" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=24)), 1, :)
    y = @. X[1, :]^3 + 0.17
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=32,
        max_complexity=8,
        initial_depth=1,
        tournament_size=3,
        elite_count=1,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        mutation_probability=1.0,
        crossover_probability=0.0,
        random_restart_probability=1.0,
        constant_refinement_iterations=0,
        initial_pool_multiplier=1,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(42))

    @test result.variation.mutation_attempts > 0
    @test result.variation.mutation_kinds.restart ==
        result.variation.mutation_attempts
end

@testitem "Scalar search exposes subtree graft mutation" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=32)), 1, :)
    y = @. X[1, :] + sin(X[1, :])
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=96,
        max_complexity=14,
        initial_depth=2,
        tournament_size=3,
        elite_count=1,
        unary_operators=[:sin],
        binary_operators=[:+, :*],
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        initial_pool_multiplier=1,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(1234))

    @test result.variation.mutation_kinds.graft > 0
    @test sum(values(result.variation.mutation_kinds)) ==
        result.variation.mutation_attempts
end

@testitem "Scalar search exposes tree rotation mutation" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=32)), 1, :)
    y = @. 0.65 * X[1, :] + 0.2 * sin(X[1, :]) - 0.1
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=96,
        max_complexity=14,
        initial_depth=3,
        tournament_size=3,
        elite_count=1,
        unary_operators=[:sin, :square],
        binary_operators=[:+, :*],
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        initial_pool_multiplier=1,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(42))

    @test result.variation.mutation_kinds.rotate > 0
    @test sum(values(result.variation.mutation_kinds)) ==
        result.variation.mutation_attempts
end

@testitem "Scalar search refines constants with separate accounting" setup=[ScalarSearchBaseImports] begin
    X = [
        -1.0 -0.5 0.0 0.5 1.0 -0.75 0.25 0.8;
         0.2  0.9 -0.4 1.0 -0.7 -0.25 0.65 -0.9;
    ]
    y = @. 0.75 * X[1, :] - 1.25 * X[2, :] + 0.5
    approximate = scalar_binary(:+,
        scalar_binary(:+,
            scalar_binary(:*, scalar_constant(0.2), scalar_input(1)),
            scalar_binary(:*, scalar_constant(-0.3), scalar_input(2))),
        scalar_constant(1.1))
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=4,
        max_complexity=16,
        initial_depth=1,
        tournament_size=1,
        elite_count=1,
        unary_operators=[:square],
        binary_operators=[:+, :-, :*],
        constant_probability=0.2,
        constant_refinement_iterations=160,
        constant_refinement_elites=1,
        constant_refinement_interval=1,
        constant_refinement_start_generation=1,
        constant_refinement_probability=0.0,
        max_constant_refinement_evaluations=1_000,
        initial_pool_multiplier=1,
    )

    result = search_scalar(
        X,
        y;
        config,
        rng=MersenneTwister(8000),
        initial_population=[approximate, approximate],
    )

    @test result.best_loss <= 1.0e-5
    @test result.evaluations <= config.max_evaluations
    @test result.constant_refinement_evaluations > 0
    @test result.constant_refinement_evaluations <=
        config.max_constant_refinement_evaluations
    @test result.constant_refinement_invalid_evaluations == 0
end

@testitem "Scalar search does not polish the unevolved initial pool" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=24)), 1, :)
    y = @. 0.7 * X[1, :] + 0.2
    approximate = scalar_binary(:+,
        scalar_binary(:*, scalar_constant(0.4), scalar_input(1)),
        scalar_constant(0.0))
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=8,
        max_complexity=8,
        initial_depth=1,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        constant_refinement_iterations=32,
        constant_refinement_elites=2,
        constant_refinement_interval=2,
        initial_pool_multiplier=1,
        initial_affine_basis=false,
    )

    result = search_scalar(
        X,
        y;
        config,
        rng=MersenneTwister(555),
        initial_population=[approximate],
    )

    @test result.generations == 0
    @test result.constant_refinement_evaluations == 0

    delayed = ScalarSearchConfig(
        population_size=8,
        max_evaluations=16,
        max_complexity=8,
        initial_depth=1,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        constant_refinement_iterations=32,
        constant_refinement_elites=2,
        constant_refinement_interval=1,
        constant_refinement_start_generation=3,
        constant_refinement_probability=1.0,
        initial_pool_multiplier=1,
    )
    delayed_result = search_scalar(
        X,
        y;
        config=delayed,
        rng=MersenneTwister(555),
        initial_population=fill(approximate, delayed.population_size),
    )

    @test delayed_result.generations < delayed.constant_refinement_start_generation
    @test delayed_result.constant_refinement_evaluations == 0
end

@testitem "Scalar search polishes evolved Pareto candidates at the final boundary" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=24)), 1, :)
    y = @. 0.7 * X[1, :] + 0.2
    approximate = scalar_binary(:+,
        scalar_binary(:*, scalar_constant(0.4), scalar_input(1)),
        scalar_constant(0.0))
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=4,
        max_complexity=8,
        initial_depth=1,
        tournament_size=1,
        elite_count=1,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=64,
        constant_refinement_elites=1,
        constant_refinement_interval=100,
        constant_refinement_start_generation=1,
        constant_refinement_probability=0.0,
        max_constant_refinement_evaluations=500,
        initial_pool_multiplier=1,
    )

    result = search_scalar(
        X,
        y;
        config,
        rng=MersenneTwister(2000),
        initial_population=fill(approximate, config.population_size),
        stop_loss=nothing,
    )

    @test result.evaluations == config.max_evaluations
    @test result.variation.mutation_kinds.constant > 0
    @test result.constant_refinement_evaluations > 0
    @test result.constant_refinement_evaluations <=
        config.max_constant_refinement_evaluations
    @test result.best_loss <= 1.0e-8
end

@testitem "Scalar search reports diverse leaders and variation outcomes" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=32)), 1, :)
    y = @. X[1, :] * X[1, :] + X[1, :] + 0.001 * sin(7 * X[1, :])
    config = ScalarSearchConfig(
        population_size=12,
        max_evaluations=125,
        max_complexity=12,
        initial_depth=2,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        constant_refinement_iterations=8,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(1234))

    @test !isempty(result.leaders)
    @test length(result.leaders) <= 8
    @test all(leader -> leader.loss >= result.best_loss, result.leaders)
    @test result.variation.attempts == result.variation.changed + result.variation.unchanged
    @test sum(values(result.variation.mutation_kinds)) ==
        result.variation.mutation_attempts
    @test sum(entry.count for entry in result.variation.final_complexity_histogram) ==
        result.variation.final_population_size
    @test result.variation.final_population_size == config.population_size
    @test 1 <= result.variation.final_unique_structures <=
        result.variation.final_population_size
    @test issorted(entry.complexity for entry in result.complexity_archive)
    @test allunique(entry.complexity for entry in result.complexity_archive)
    @test minimum(entry.loss for entry in result.complexity_archive) == result.best_loss
    @test 0.0 <= scalar_structure_similarity(result.leaders[1].expr,
        result.leaders[end].expr) <= 1.0
    @test scalar_structure_similarity(
        scalar_binary(:+, scalar_input(1), scalar_constant(1.0)),
        scalar_binary(:+, scalar_input(1), scalar_constant(2.0)),
    ) == 1.0
end

@testitem "Scalar search discovers a coefficient-bearing analytic teacher" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=96)), 1, :)
    y = @. sin(X[1, :]) + 0.3 * X[1, :] * X[1, :] - 0.2
    config = ScalarSearchConfig(
        population_size=32,
        max_evaluations=12_000,
        max_complexity=16,
        initial_depth=3,
        unary_operators=[:sin, :square],
        binary_operators=[:+, :*],
        constant_probability=0.2,
        mutation_probability=0.9,
        crossover_probability=0.05,
        constant_refinement_iterations=16,
        constant_refinement_elites=2,
        constant_refinement_interval=10,
        constant_refinement_start_generation=1,
        constant_refinement_probability=0.01,
        initial_affine_basis=false,
    )

    result = search_scalar(X, y; config, rng=MersenneTwister(8000))

    @test result.best_loss <= 1.0e-5
    @test result.generations > 0
    @test result.variation.attempts > 0
    @test result.constant_refinement_evaluations > 0
    @test result.variation.mutation_kinds.scale > 0
    @test result.variation.mutation_kinds.shift > 0
end

@testitem "Scalar search accepts a contextual repeated-call objective" setup=[ScalarSearchBaseImports] begin
    X = reshape(collect(range(-1.0, 1.0; length=48)), 1, :)
    triple_target = @. 3 * (X[1, :] * X[1, :] + X[1, :])
    objective_calls = Ref(0)
    function repeated_objective(expr)
        objective_calls[] += 1
        prediction = evaluate_scalar(expr, X)
        prediction isa LambdaRegression.Utils.Err && return prediction
        return sum(abs2, 3 .* prediction .- triple_target) / length(triple_target)
    end
    config = ScalarSearchConfig(
        population_size=16,
        max_evaluations=512,
        max_complexity=12,
        initial_depth=2,
        unary_operators=[:square],
        binary_operators=[:+, :*],
        constant_refinement_iterations=16,
    )

    result = search_scalar(
        X,
        zeros(size(X, 2));
        objective=repeated_objective,
        config,
        rng=MersenneTwister(2000),
    )

    @test result.best_loss <= 1.0e-10
    @test objective_calls[] >= result.evaluations
end

@testitem "Fixed scalar scaffold materializes shared constants transactionally" setup=[ScalarSearchBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode

    apply2(op, left, right) = AppNode(AppNode(ConstNode(op), left), right)
    template = AstTree(AbsNode(:a, AbsNode(:b, AbsNode(:c,
        apply2(:FADD,
            apply2(:FADD, ConstNode(:PHI_A), ConstNode(:PHI_B)),
            ConstNode(:PHI_C)),
    ))))
    original = deepcopy(template)
    scaffold = FixedScalarScaffold(template, [
        FixedScalarCall(:PHI_A, [IndexNode(2)]),
        FixedScalarCall(:PHI_B, [IndexNode(1)]),
        FixedScalarCall(:PHI_C, [IndexNode(0)]),
    ])
    x = scalar_input(1)
    candidate = scalar_binary(:+, scalar_unary(:square, x), scalar_constant(0.5))

    materialized = materialize_scalar_scaffold(scaffold, candidate)

    @test materialized isa ScalarMaterialization
    @test template == original
    @test length(materialized.local_consts) == 1
    @test length(materialized.local_consts_ty) == 1
    second_materialized = materialize_scalar_scaffold(scaffold, candidate)
    @test second_materialized.local_consts !== materialized.local_consts
    @test second_materialized.tree !== materialized.tree
    prepared = LambdaRegression.Execution.prepare_native_evaluator(materialized.tree)
    context = merge(
        Dict{Symbol,Any}(:FADD => LambdaRegression.LambdaRegression.Primitives.FADD,
            :SQR => LambdaRegression.LambdaRegression.Primitives.SQR),
        materialized.local_consts,
    )
    @test LambdaRegression.Execution.run_native_evaluator(
        prepared, context; args=(1.0, 2.0, 3.0), max_steps=256) ≈ 15.5
    inferred = LambdaRegression.TypeSystem.infer(
        materialized.tree,
        merge(LambdaRegression.TypeSystem.get_type_ctx(Dict{Symbol,Any}(
            :FADD => LambdaRegression.LambdaRegression.Primitives.FADD,
            :SQR => LambdaRegression.LambdaRegression.Primitives.SQR,
        )), materialized.local_consts_ty),
    )
    expected = LambdaRegression.TypeSystem.TyFun(
        LambdaRegression.TypeSystem.FloatTy,
        LambdaRegression.TypeSystem.TyFun(
            LambdaRegression.TypeSystem.FloatTy,
            LambdaRegression.TypeSystem.TyFun(
                LambdaRegression.TypeSystem.FloatTy,
                LambdaRegression.TypeSystem.FloatTy,
            ),
        ),
    )
    @test LambdaRegression.TypeSystem.iscompatible(inferred, expected)
end

@testitem "Fixed scalar scaffold resolves nested shared calls" setup=[ScalarSearchBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode

    template = AstTree(AbsNode(:x, ConstNode(:PHI_3)))
    scaffold = FixedScalarScaffold(template, [
        FixedScalarCall(:PHI_1, [IndexNode(0)]),
        FixedScalarCall(:PHI_2, [ConstNode(:PHI_1)]),
        FixedScalarCall(:PHI_3, [ConstNode(:PHI_2)]),
    ])
    candidate = scalar_binary(:+, scalar_input(1), scalar_constant(0.5))

    materialized = materialize_scalar_scaffold(scaffold, candidate)

    @test materialized isa ScalarMaterialization
    @test length(materialized.local_consts) == 1
    prepared = LambdaRegression.Execution.prepare_native_evaluator(materialized.tree)
    context = merge(
        Dict{Symbol,Any}(:FADD => LambdaRegression.LambdaRegression.Primitives.FADD),
        materialized.local_consts,
    )
    @test LambdaRegression.Execution.run_native_evaluator(
        prepared, context; args=(1.0,), max_steps=256) ≈ 2.5

    cyclic = FixedScalarScaffold(
        AstTree(AbsNode(:x, ConstNode(:PHI_A))),
        [
            FixedScalarCall(:PHI_A, [ConstNode(:PHI_B)]),
            FixedScalarCall(:PHI_B, [ConstNode(:PHI_A)]),
        ],
    )
    issue = materialize_scalar_scaffold(cyclic, candidate)
    @test issue isa LambdaRegression.Utils.Err
    @test issue.value == ScalarScaffoldErr(:cyclic_dependency, :PHI_A)
end

@testitem "Fixed scalar scaffold preserves a let-like application template" setup=[ScalarSearchBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode

    scaffold = FixedScalarScaffold(
        AstTree(AppNode(
            AbsNode(:x, ConstNode(:PHI)),
            ConstNode(:two),
        )),
        [FixedScalarCall(:PHI, [IndexNode(0)])],
    )
    materialized = materialize_scalar_scaffold(scaffold, scalar_input(1))

    @test materialized isa ScalarMaterialization
    @test materialized.tree.root isa AppNode
    @test materialized.tree.root.func isa AbsNode
    @test materialized.tree.root.func.name == :x
    @test materialized.tree.root.func.body == IndexNode(0)
    @test materialized.tree.root.arg == ConstNode(:two)
    @test LambdaRegression.Execution.evaluate_native(
        materialized.tree, Dict{Symbol,Any}(:two => 2.0)) == 2.0
end

@testitem "Fixed scalar scaffold preserves polymorphic let-like RHS" setup=[ScalarSearchBaseImports] begin
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, TyProd, infer

    template = AstTree(AppNode(
        AbsNode(:id, AppNode(
            AppNode(ConstNode(:PAIR), ConstNode(:PHI)),
            AppNode(IndexNode(0), ConstNode(:yes)),
        )),
        AbsNode(:y, IndexNode(0)),
    ))
    scaffold = FixedScalarScaffold(template, [FixedScalarCall(
        :PHI, [AppNode(IndexNode(0), ConstNode(:one))])])
    materialized = materialize_scalar_scaffold(scaffold, scalar_input(1))
    context = Dict{Symbol,TyLike}(
        :PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy,
        :yes => BoolTy,
    )

    @test materialized isa ScalarMaterialization
    @test materialized.tree.root isa AppNode
    @test materialized.tree.root.func isa AbsNode
    @test infer(materialized.tree, context) == TyProd(IntTy, BoolTy)
end

@testitem "Direct scalar map preserves dense and nested shapes" setup=[ScalarSearchBaseImports] begin
    expr = scalar_binary(:+, scalar_unary(:square, scalar_input(1)), scalar_input(2))
    plan = prepare_scalar_map(expr; parameter_count=1)
    input = [1.0 2.0 3.0; 4.0 5.0 6.0]
    output = similar(input)
    workspace = ScalarMapWorkspace()

    @test execute_scalar_map!(output, plan, input, (0.5,), workspace) === output
    @test output ≈ input .^ 2 .+ 0.5

    nested = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    nested_output = execute_scalar_map(plan, nested, (0.5,))
    @test length.(nested_output) == [2, 3]
    @test nested_output == [[1.5, 4.5], [9.5, 16.5, 25.5]]

    domain_plan = prepare_scalar_map(scalar_unary(:sqrt, scalar_input(1)))
    @test execute_scalar_map(domain_plan, [-1.0, 4.0]) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:sqrt, 1))
end

@testitem "Direct scalar fold is ordered and supports array accumulators" setup=[ScalarSearchBaseImports] begin
    subtract = prepare_scalar_fold(
        scalar_binary(:-, scalar_input(1), scalar_input(2)))
    @test execute_scalar_fold(subtract, [1.0, 2.0, 3.0], 10.0) == 4.0
    @test execute_scalar_fold(subtract, Float64[], 10.0) == 10.0

    add = prepare_scalar_fold(scalar_binary(:+, scalar_input(1), scalar_input(2)))
    sequence = [[1.0, 2.0], [3.0, 4.0]]
    result = execute_scalar_fold(add, sequence, [10.0, 20.0])
    @test result == [14.0, 26.0]

    bad = execute_scalar_fold(add, [[1.0], [2.0, 3.0]], [0.0])
    @test bad == LambdaRegression.Utils.Err(ScalarPlanShapeErr(:fold_item_shape))
end

@testitem "Scalar collection plans evaluate extended operators and report their domains" setup=[ScalarSearchBaseImports] begin
    map_expr = scalar_binary(:max, scalar_unary(:exp, scalar_input(1)), scalar_input(2))
    map_plan = prepare_scalar_map(map_expr; parameter_count=1)
    @test execute_scalar_map(map_plan, [-1.0, 0.0, 1.0], (0.5,)) ≈
        [0.5, 1.0, exp(1.0)]

    fold_expr = scalar_binary(:^, scalar_input(1), scalar_input(2))
    fold_plan = prepare_scalar_fold(fold_expr)
    @test execute_scalar_fold(fold_plan, [2.0, 2.0], 2.0) == 16.0

    # Invalid values take the prepared-point fallback and retain the operator.
    log_plan = prepare_scalar_map(scalar_unary(:log, scalar_input(1)))
    @test execute_scalar_map(log_plan, [-1.0]) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:log, 1))
    @test execute_scalar_fold(fold_plan, [0.5], -2.0) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:^, 1))

    sqrt_plan = prepare_scalar_map(scalar_unary(:sqrt, scalar_input(1)))
    @test execute_scalar_map(sqrt_plan, [-1.0]) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:sqrt, 1))

    # tan must route non-finite intermediate values through the point evaluator.
    tan_division = scalar_unary(:tan,
        scalar_binary(:/, scalar_input(1), scalar_constant(0.0)))
    tan_plan = prepare_scalar_map(tan_division)
    @test execute_scalar_map(tan_plan, [1.0]) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:/, 1))
end

end # module ScalarSearch
