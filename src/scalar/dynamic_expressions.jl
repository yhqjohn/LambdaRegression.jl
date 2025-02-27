"""
    DynamicExpressions scalar bridge

Small, deliberately local adapter between the project's rebuildable `ScalarExpr`
language and `DynamicExpressions.Node`.  The adapter owns neither the scalar
search state nor a backend interface: it only provides a convenient
representation and batched evaluator for a scalar island.  The
`ScalarExpr`-to-AST boundary remains authoritative.

`DynamicExpressions` is intentionally imported here, rather than from the
search coordinator.  Keeping the dependency in this file makes the bridge
easy to remove if the scalar search later uses another execution tree.
"""

import DynamicExpressions

export DYNAMIC_SCALAR_OPERATORS
export scalar_to_dynamic
export scalar_from_dynamic
export evaluate_dynamic_scalar
export DynamicScalarWorkspace
export evaluate_dynamic_scalar!

"""Square primitive kept as a named function for stable operator identity."""
@inline _dynamic_square(x) = x * x

"""Domain-safe square root matching `evaluate_scalar_reference`."""
@inline _dynamic_sqrt(x) = x < 0 ? NaN : sqrt(x)

"""
Operator table for the scalar world.  The order is part of the local
Node encoding: unary indices are `sin`, `cos`, `sqrt`, `square`, and binary
indices are `+`, `-`, `*`, `/`.

Helper methods are disabled because this bridge always passes the table
explicitly to DynamicExpressions and must not mutate its process-wide default
operator state.
"""
const DYNAMIC_SCALAR_OPERATORS = DynamicExpressions.OperatorEnum(
    binary_operators=(+, -, *, /),
    unary_operators=(sin, cos, _dynamic_sqrt, _dynamic_square),
    define_helper_functions=false,
    empty_old_operators=false,
)

@inline function _dynamic_unary_index(op::Symbol)
    op === :sin && return UInt8(1)
    op === :cos && return UInt8(2)
    op === :sqrt && return UInt8(3)
    op === :square && return UInt8(4)
    throw(ArgumentError("unsupported scalar unary operator: $op"))
end

@inline function _dynamic_binary_index(op::Symbol)
    op === :+ && return UInt8(1)
    op === :- && return UInt8(2)
    op === :* && return UInt8(3)
    op === :/ && return UInt8(4)
    throw(ArgumentError("unsupported scalar binary operator: $op"))
end

function _scalar_to_dynamic(expr::_ScalarInput)
    return DynamicExpressions.Node(Float64; feature=expr.index)
end

function _scalar_to_dynamic(expr::_ScalarConstant)
    return DynamicExpressions.Node(Float64; val=expr.value)
end

function _scalar_to_dynamic(expr::_ScalarUnary)
    arg = _scalar_to_dynamic(expr.arg)
    return DynamicExpressions.Node(Float64; op=_dynamic_unary_index(expr.op), l=arg)
end

function _scalar_to_dynamic(expr::_ScalarBinary)
    left = _scalar_to_dynamic(expr.left)
    right = _scalar_to_dynamic(expr.right)
    return DynamicExpressions.Node(Float64;
        op=_dynamic_binary_index(expr.op), l=left, r=right)
end

"""
    scalar_to_dynamic(expr::ScalarExpr) -> DynamicExpressions.Node{Float64}

Convert a supported `ScalarExpr` to the local DynamicExpressions node
encoding.  Constants and feature indices retain their `Float64`/one-based
semantics; no simplification or canonicalisation is performed.
"""
scalar_to_dynamic(expr::ScalarExpr) = _scalar_to_dynamic(expr)

function _dynamic_unary_symbol(op)
    op === sin && return :sin
    op === cos && return :cos
    op === _dynamic_sqrt && return :sqrt
    op === _dynamic_square && return :square
    return nothing
end

function _dynamic_binary_symbol(op)
    op === (+) && return :+
    op === (-) && return :-
    op === (*) && return :*
    op === (/) && return :/
    return nothing
end

function _scalar_from_dynamic(node::DynamicExpressions.AbstractExpressionNode,
        operators::DynamicExpressions.AbstractOperatorEnum)
    node.degree == 0 && return node.constant ? scalar_constant(node.val) :
        scalar_input(Int(node.feature))

    if node.degree == 1
        op = node.op >= 1 && node.op <= length(operators.unaops) ?
            _dynamic_unary_symbol(operators.unaops[node.op]) : nothing
        op === nothing && return Err(ScalarConversionErr(:unsupported_operator))
        arg = _scalar_from_dynamic(node.l, operators)
        arg isa Err && return arg
        return scalar_unary(op, arg)
    elseif node.degree == 2
        op = node.op >= 1 && node.op <= length(operators.binops) ?
            _dynamic_binary_symbol(operators.binops[node.op]) : nothing
        op === nothing && return Err(ScalarConversionErr(:unsupported_operator))
        left = _scalar_from_dynamic(node.l, operators)
        left isa Err && return left
        right = _scalar_from_dynamic(node.r, operators)
        right isa Err && return right
        return scalar_binary(op, left, right)
    end

    return Err(ScalarConversionErr(:malformed_dynamic_node))
end

"""
    scalar_from_dynamic(node; operators=DYNAMIC_SCALAR_OPERATORS) -> ScalarExpr | Err

Rebuild a `ScalarExpr` from a DynamicExpressions node.  The conversion is
strict about the four unary and four binary functions in the local operator
table; unsupported operator tables remain on the caller's general path.
"""
function scalar_from_dynamic(node::DynamicExpressions.AbstractExpressionNode;
        operators::DynamicExpressions.AbstractOperatorEnum=DYNAMIC_SCALAR_OPERATORS)
    return _scalar_from_dynamic(node, operators)
end

"""Return a finite output vector or the first scalar-domain classification."""
function _evaluate_dynamic_node(node::DynamicExpressions.AbstractExpressionNode,
        X::AbstractMatrix{<:Real}, operators)
    required = _dynamic_required_features(node)
    size(X, 1) >= required || return Err(ScalarInputShapeErr(required, size(X, 1)))
    Xf = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    output, complete = DynamicExpressions.eval_tree_array(node, Xf, operators)
    if complete && all(isfinite, output)
        return Vector{Float64}(output)
    end
    # DynamicExpressions reports only a completion bit.  The caller that owns
    # the ScalarExpr can recover the project's precise (operator, sample)
    # classification; a bare node has no such provenance.
    sample = findfirst(value -> !isfinite(value), output)
    return Err(ScalarDomainErr(:dynamic, something(sample, 1)))
end

@testsnippet DynamicExpressionsBufferedImports begin
    using Test
    using LambdaRegression.ScalarSearch
    using LambdaRegression.ScalarSearch: DynamicExpressions
end

@testitem "DynamicExpressions buffered evaluation matches the allocating path" setup=[DynamicExpressionsBufferedImports] begin
    expr = scalar_binary(:+, scalar_unary(:sin, scalar_input(1)),
        scalar_binary(:*, scalar_input(2), scalar_constant(2.0)))
    node = scalar_to_dynamic(expr)
    X = [0.0 0.5 1.0; 1.0 2.0 3.0]
    workspace = DynamicScalarWorkspace()

    expected = evaluate_dynamic_scalar(node, X)
    actual = evaluate_dynamic_scalar!(workspace, node, X)
    @test actual ≈ expected
    saved = copy(actual)

    # The result is a workspace view, but the copied value remains stable
    # across the next call on that workspace.
    repeated = evaluate_dynamic_scalar!(workspace, node, X)
    @test repeated ≈ saved
    @test saved ≈ expected
end

@testitem "DynamicExpressions buffered workspace expands safely" setup=[DynamicExpressionsBufferedImports] begin
    node = scalar_to_dynamic(scalar_binary(:+, scalar_input(1),
        scalar_unary(:square, scalar_input(2))))
    workspace = DynamicScalarWorkspace()
    small = [1.0 2.0; 2.0 3.0]
    large = [collect(1.0:12.0)'; collect(2.0:13.0)']
    @test evaluate_dynamic_scalar!(workspace, node, small) ≈
        evaluate_dynamic_scalar(node, small)
    @test evaluate_dynamic_scalar!(workspace, node, large) ≈
        evaluate_dynamic_scalar(node, large)
    @test workspace.sample_capacity == size(large, 2)
    small_again = large[:, 1:3]
    buffered_small = evaluate_dynamic_scalar!(workspace, node, small_again)
    @test length(buffered_small) == size(small_again, 2)
    @test buffered_small ≈ evaluate_dynamic_scalar(node, small_again)
    @test workspace.sample_capacity == size(small_again, 2)
    @test workspace.node_capacity >= DynamicExpressions.count_nodes(node)
end

@testitem "DynamicExpressions buffered evaluation preserves domain failure flag" setup=[DynamicExpressionsBufferedImports] begin
    node = scalar_to_dynamic(scalar_unary(:sqrt, scalar_input(1)))
    X = reshape([-1.0, 4.0], 1, :)
    workspace = DynamicScalarWorkspace()
    @test evaluate_dynamic_scalar!(workspace, node, X) ==
        evaluate_dynamic_scalar(node, X)
end

function _dynamic_required_features(node::DynamicExpressions.AbstractExpressionNode)
    node.degree == 0 && return node.constant ? 0 : Int(node.feature)
    node.degree == 1 && return _dynamic_required_features(node.l)
    node.degree == 2 && return max(_dynamic_required_features(node.l),
        _dynamic_required_features(node.r))
    return 0
end

"""
    DynamicScalarWorkspace(; node_capacity=1, sample_capacity=1)

Reusable storage for `evaluate_dynamic_scalar!`.  The returned vector from a
buffered evaluation is a view into this workspace and remains valid only until
the next call using the same workspace.  A workspace owns mutable state and
must therefore not be shared by concurrent evaluations.  Node capacity grows
as needed; the sample width is kept exact because DynamicExpressions' buffer
rows are not truncated to the input axes.
"""
mutable struct DynamicScalarWorkspace{B,O}
    buffer::B
    eval_options::O
    node_capacity::Int
    sample_capacity::Int
end

function DynamicScalarWorkspace(; node_capacity::Int=1, sample_capacity::Int=1)
    node_capacity >= 1 || throw(ArgumentError("node_capacity must be positive"))
    sample_capacity >= 1 || throw(ArgumentError("sample_capacity must be positive"))
    buffer = DynamicExpressions.ArrayBuffer(
        Matrix{Float64}(undef, node_capacity, sample_capacity), Ref(0))
    options = DynamicExpressions.EvalOptions(; buffer)
    return DynamicScalarWorkspace(buffer, options, node_capacity, sample_capacity)
end

function _ensure_dynamic_workspace!(workspace::DynamicScalarWorkspace,
        node::DynamicExpressions.AbstractExpressionNode, samples::Int)
    samples >= 1 || throw(ArgumentError("dynamic evaluation requires at least one sample"))
    # Count repeated shared subtrees too: the evaluator may visit a shared
    # node once per parent path, so unique-node counting is not a safe buffer
    # bound for arbitrary DynamicExpressions nodes.
    required_nodes = max(1,
        DynamicExpressions.count_nodes(node; break_sharing=Val(true)))
    # DynamicExpressions.ArrayBuffer returns the whole row and does not slice
    # it to the input axes.  The sample width must therefore match exactly;
    # retaining a larger row would evaluate uninitialized tail values.
    required_nodes <= workspace.node_capacity &&
        samples == workspace.sample_capacity && return workspace

    node_capacity = max(required_nodes, workspace.node_capacity)
    while node_capacity < required_nodes
        node_capacity = max(required_nodes, 2 * node_capacity)
    end
    sample_capacity = samples
    buffer = DynamicExpressions.ArrayBuffer(
        Matrix{Float64}(undef, node_capacity, sample_capacity), Ref(0))
    workspace.buffer = buffer
    workspace.eval_options = DynamicExpressions.EvalOptions(; buffer)
    workspace.node_capacity = node_capacity
    workspace.sample_capacity = sample_capacity
    return workspace
end

"""
    evaluate_dynamic_scalar!(workspace, node, X; operators=DYNAMIC_SCALAR_OPERATORS)

Evaluate a converted node using reusable `ArrayBuffer` storage.  The result is
a view backed by `workspace`; callers that need to retain it across another
call must copy it.  A failed evaluation returns the same generic dynamic
`Err` classification as `evaluate_dynamic_scalar(node, X)`.
"""
function evaluate_dynamic_scalar!(workspace::DynamicScalarWorkspace,
        node::DynamicExpressions.AbstractExpressionNode,
        X::AbstractMatrix{<:Real};
        operators::DynamicExpressions.AbstractOperatorEnum=DYNAMIC_SCALAR_OPERATORS)
    required = _dynamic_required_features(node)
    size(X, 1) >= required || return Err(ScalarInputShapeErr(required, size(X, 1)))
    Xf = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    _ensure_dynamic_workspace!(workspace, node, size(Xf, 2))
    output, complete = DynamicExpressions.eval_tree_array(
        node, Xf, operators; eval_options=workspace.eval_options)
    if complete && all(isfinite, output)
        return output
    end
    sample = findfirst(value -> !isfinite(value), output)
    return Err(ScalarDomainErr(:dynamic, something(sample, 1)))
end

"""
    evaluate_dynamic_scalar!(workspace, expr, X; operators=DYNAMIC_SCALAR_OPERATORS)

Convert and evaluate a `ScalarExpr` with reusable storage. On a numeric-domain
failure the reference evaluator is used only to recover the precise operator
and sample classification, matching the allocating expression overload.
"""
function evaluate_dynamic_scalar!(workspace::DynamicScalarWorkspace,
        expr::ScalarExpr, X::AbstractMatrix{<:Real};
        operators::DynamicExpressions.AbstractOperatorEnum=DYNAMIC_SCALAR_OPERATORS)
    required = _required_features(expr)
    size(X, 1) >= required || return Err(ScalarInputShapeErr(required, size(X, 1)))
    result = evaluate_dynamic_scalar!(workspace, scalar_to_dynamic(expr), X;
        operators)
    result isa Err || return result
    return evaluate_scalar_reference(expr, X)
end

"""
    evaluate_dynamic_scalar(expr, X; operators=DYNAMIC_SCALAR_OPERATORS)

Evaluate a `ScalarExpr` using DynamicExpressions over a `features × samples`
matrix.  Valid batches use the DynamicExpressions fast path.  If it reports a
domain/non-finite result, the authoritative reference evaluator is called only
for error classification, preserving the existing `ScalarDomainErr` contract.
"""
function evaluate_dynamic_scalar(expr::ScalarExpr, X::AbstractMatrix{<:Real};
        operators::DynamicExpressions.AbstractOperatorEnum=DYNAMIC_SCALAR_OPERATORS)
    required = _required_features(expr)
    size(X, 1) >= required || return Err(ScalarInputShapeErr(required, size(X, 1)))
    node = scalar_to_dynamic(expr)
    result = _evaluate_dynamic_node(node, X, operators)
    result isa Err || return result
    return evaluate_scalar_reference(expr, X)
end

"""
    evaluate_dynamic_scalar(node, X; operators=DYNAMIC_SCALAR_OPERATORS)

Evaluate an already-converted node.  A bare node can report only the generic
`:dynamic` domain tag because it does not carry the originating `ScalarExpr`.
"""
function evaluate_dynamic_scalar(node::DynamicExpressions.AbstractExpressionNode,
        X::AbstractMatrix{<:Real};
        operators::DynamicExpressions.AbstractOperatorEnum=DYNAMIC_SCALAR_OPERATORS)
    return _evaluate_dynamic_node(node, X, operators)
end

@testsnippet DynamicExpressionsScalarImports begin
    using Test
    using LambdaRegression.ScalarSearch
end

@testitem "DynamicExpressions scalar roundtrip preserves batch semantics" setup=[DynamicExpressionsScalarImports] begin
    expr = scalar_binary(:+, scalar_unary(:sqrt,
            scalar_unary(:square, scalar_input(1))),
        scalar_binary(:*, scalar_input(2), scalar_constant(2.0)))
    X = [0.0 0.5 1.0; 1.0 2.0 3.0]

    node = scalar_to_dynamic(expr)
    rebuilt = scalar_from_dynamic(node)
    @test rebuilt isa ScalarExpr
    @test scalar_complexity(rebuilt) == scalar_complexity(expr)
    @test scalar_structure_similarity(rebuilt, expr) == 1.0
    @test evaluate_dynamic_scalar(expr, X) ≈ evaluate_scalar(expr, X)
    @test evaluate_dynamic_scalar(node, X) ≈ evaluate_scalar(expr, X)
end

@testitem "DynamicExpressions scalar bridge preserves domain classification" setup=[DynamicExpressionsScalarImports] begin
    expr = scalar_unary(:sqrt, scalar_input(1))
    X = reshape([-1.0, 4.0], 1, :)
    workspace = DynamicScalarWorkspace()

    @test evaluate_dynamic_scalar(expr, X) == evaluate_scalar_reference(expr, X)
    @test evaluate_dynamic_scalar!(workspace, expr, X) ==
        evaluate_scalar_reference(expr, X)
    @test evaluate_dynamic_scalar(expr, X) ==
        LambdaRegression.Utils.Err(ScalarDomainErr(:sqrt, 1))
end
