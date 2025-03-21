"""
    Execution

    Execution support for AST reduction, host-value application, and evaluation
    of `FunctionDefNode` and complete `Program` values.
`EvalContext` provides overlay/locals/globals lookup with writes confined to
the overlay.  `beta_reduce!` keeps the existing normal-order one-step contract
and delegates host application through the extensible `isapplicable` and
`apply_node!` methods. `prepare_native_evaluator` prepares a tree once for
repeated call-by-need evaluation; `run_native_evaluator` creates a fresh shared
step budget and runtime environment for each run. `prepare_default_evaluator`
prepares a complete program for repeated core-reduction or native evaluation
using explicit input, global, context-factory, and budget keywords. Each sample
gets a fresh context; synchronous host callbacks and lifted closures share that
sample's core budget. FunctionDef lowering remains the independent
`Transforms.lower_functiondefs` operation.

Quickstart for one complete program:
```julia
prepared = prepare_default_evaluator(program;
    input_slots=[:x], global_consts=globals)
prediction = prepared(1.25)
```
Task-specific roots and losses remain outside this module. Non-finite outputs,
known native failures, and incomplete top-level core reductions are returned as
`Err`; expected failures raised inside lifted-closure calls and unexpected host
exceptions retain their existing thrown-error behavior.
"""
module Execution

using TestItems

using ..ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, VarNode, IndexNode, AppNode, ConstNode,
    FunctionDefNode,
    eliminate_bound_variables!
using ..Trees: traverse, PointerState, getstate, Break, setroot!, setchild!
using ..Programs: ConstSymbolFactory, _next_const_symbol!, Program
import ..ASTs.NodeASTs: apply_node!
import ..Programs: new_constant!
using ..Utils: Err

export EvalContext, beta_reduce!, prepare, isapplicable
export PreparedNativeEvaluator, prepare_native_evaluator
export run_native_evaluator, evaluate_native
export NativeEvaluationError, NativeStepLimitError, NativeNonValueError
export NativeUnboundIndexError, NativeMissingConstantError
export NativeEvaluationCycleError
export PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
export MaxReduceStepsErr, ClosureDepthLimitErr, NonConstOutputErr
export MissingConstValueErr, NonClosedLambdaErr, NonFiniteOutputErr

struct MaxReduceStepsErr
    steps::Int
end
struct ClosureDepthLimitErr
    depth::Int
    limit::Int
end
struct NonConstOutputErr end
struct MissingConstValueErr
    name::Symbol
end
struct NonClosedLambdaErr
    f::Symbol
end
struct NonFiniteOutputErr end

include("native_evaluator.jl")

mutable struct EvalContext
    overlay::Dict{Symbol,Any}
    locals::Dict{Symbol,Any}
    globals::Dict{Symbol,Any}
    symfactory::ConstSymbolFactory
end

function EvalContext(;
    overlay::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    locals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    globals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    symfactory::ConstSymbolFactory=ConstSymbolFactory(),
)
    return EvalContext(overlay, locals, globals, symfactory)
end

Base.haskey(ctx::EvalContext, key::Symbol) =
    haskey(ctx.overlay, key) || haskey(ctx.locals, key) || haskey(ctx.globals, key)

function Base.getindex(ctx::EvalContext, key::Symbol)
    haskey(ctx.overlay, key) && return ctx.overlay[key]
    haskey(ctx.locals, key) && return ctx.locals[key]
    haskey(ctx.globals, key) && return ctx.globals[key]
    throw(KeyError(key))
end

Base.setindex!(ctx::EvalContext, value, key::Symbol) = (ctx.overlay[key] = value)

function Base.get(ctx::EvalContext, key::Symbol, default)
    haskey(ctx.overlay, key) && return ctx.overlay[key]
    haskey(ctx.locals, key) && return ctx.locals[key]
    haskey(ctx.globals, key) && return ctx.globals[key]
    return default
end

function new_constant!(ctx::EvalContext, value)
    while true
        name = _next_const_symbol!(ctx.symfactory)
        haskey(ctx, name) && continue
        ctx.overlay[name] = value
        return ConstNode(name)
    end
end

# The foundation intentionally remains open to specialized host application.
apply_node!(tree, f, x, args...) = apply_node!(tree, f, x)
isapplicable(::AstNode, ::AstNode) = false
isapplicable(::AbsNode, ::AstNode) = true
isapplicable(a::AstNode, b::AstNode, args...) = isapplicable(a, b)
isapplicable(::ConstNode, ::ConstNode, ::EvalContext) = true
isapplicable(::ConstNode, ::ConstNode, ::AbstractDict) = true

function _apply_const_application(tree::AstTree, f::ConstNode, x::ConstNode, context)
    result = context[f.name](context[x.name])
    if result isa AstTree
        return result
    elseif result isa AstNode
        return AstTree(result)
    else
        return AstTree(new_constant!(context, result))
    end
end

apply_node!(tree::AstTree, f::ConstNode, x::ConstNode, context::EvalContext) =
    _apply_const_application(tree, f, x, context)
apply_node!(tree::AstTree, f::ConstNode, x::ConstNode, context::AbstractDict) =
    _apply_const_application(tree, f, x, context)

"""Perform one normal-order reduction step; return `true` iff no step occurred."""
function beta_reduce!(tree::AstTree, args...)::Bool
    reduction_occurred = Ref(false)
    traverse(tree, PointerState) do t, node, bag
        if node isa AppNode && isapplicable(node.func, node.arg, args...)
            reduced = apply_node!(t, node.func, node.arg, args...)
            pointer = getstate(bag, PointerState{AstNode})
            if pointer.parent === nothing
                setroot!(t, reduced)
            else
                setchild!(t, pointer.parent, pointer.child_index, reduced)
            end
            reduction_occurred[] = true
            return Break()
        end
        return nothing
    end
    return !reduction_occurred[]
end

function beta_reduce!(tree::AstTree, steps::Int, args...)::Bool
    steps < 0 && throw(ArgumentError("steps must be non-negative"))
    last_result = true
    for _ in 1:steps
        last_result = beta_reduce!(tree, args...)
        last_result && return true
    end
    return last_result
end

function prepare(tree::AstTree)
    prepared = deepcopy(tree)
    eliminate_bound_variables!(prepared)
    for node in prepared
        node isa AbsNode && (node.name = gensym(node.name))
    end
    return prepared
end

include("default_evaluator.jl")

@testitem "Execution EvalContext overlay and deterministic constants" begin
    using Test
    using LambdaRegression.Execution: EvalContext
    using LambdaRegression.Programs: ConstSymbolFactory, new_constant!

    globals = Dict{Symbol,Any}(:g => 1)
    overlay = Dict{Symbol,Any}()
    ctx = EvalContext(overlay=overlay, globals=globals,
        symfactory=ConstSymbolFactory(prefix=:c))
    ctx[:k] = 10
    @test overlay[:k] == 10
    @test !haskey(globals, :k)
    node = new_constant!(ctx, 42)
    @test node.name == :c1
    @test ctx.overlay[:c1] == 42
end

@testitem "Execution beta reduction applies ConstNode values" begin
    using Test
    using LambdaRegression.Execution: EvalContext, beta_reduce!
    using LambdaRegression.Programs: ConstSymbolFactory
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode

    ctx = EvalContext(globals=Dict{Symbol,Any}(:double => (x -> 2x), :three => 3),
        symfactory=ConstSymbolFactory(prefix=:c))
    tree = AstTree(AppNode(ConstNode(:double), ConstNode(:three)))
    @test beta_reduce!(tree, ctx) == false
    @test tree.root isa ConstNode
    @test ctx[tree.root.name] == 6
end

@testitem "Execution normal-order reduction remains deterministic" begin
    using Test
    using LambdaRegression.Execution: beta_reduce!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, VarNode

    tree = AstTree(AppNode(AbsNode(:x, VarNode(:x)), VarNode(:y)))
    @test beta_reduce!(tree) == false
    @test tree.root isa VarNode
    @test tree.root.name == :y
end

@testitem "Execution applies host constants through ordinary Dict contexts" begin
    using Test
    using LambdaRegression.Execution: beta_reduce!, isapplicable
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode

    context = Dict{Symbol,Any}(:double => (x -> 2x), :three => 3)
    tree = AstTree(AppNode(ConstNode(:double), ConstNode(:three)))
    @test isapplicable(tree.root.func, tree.root.arg, context)
    @test beta_reduce!(tree, context) == false
    @test context[tree.root.name] == 6
end

@testitem "Execution native evaluator preserves laziness, host AST returns, and prepared ownership" begin
    using Test
    using LambdaRegression.Execution: evaluate_native, prepare_native_evaluator,
        run_native_evaluator
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode

    lazy = AstTree(AppNode(
        AbsNode(:unused, ConstNode(:answer)),
        AppNode(ConstNode(:explode), ConstNode(:unit)),
    ))
    context = Dict{Symbol,Any}(
        :answer => 17,
        :unit => nothing,
        :explode => _ -> error("a lazy argument was forced"),
        :return_ast => _ -> AstTree(ConstNode(:answer)),
    )
    @test evaluate_native(lazy, context) == 17
    @test evaluate_native(
        AstTree(AppNode(ConstNode(:return_ast), ConstNode(:unit))), context,
    ) == 17

    source = AstTree(AbsNode(:x, IndexNode(0)))
    prepared = prepare_native_evaluator(source)
    source.root = ConstNode(:answer)
    @test run_native_evaluator(prepared, context; args=(23,), max_steps=5) == 23
    @test run_native_evaluator(prepared, context; args=(29,), max_steps=5) == 29

    escaped = run_native_evaluator(
        prepare_native_evaluator(AbsNode(:x, IndexNode(0))),
        context;
        max_steps=5,
    )
    @test escaped(31) == 31
    @test escaped(37) == 37
    @test_throws LambdaRegression.Execution.NativeStepLimitError escaped(41)
end

@testitem "Execution preserves lazy COND branches and context isolation" begin
    using Test
    using LambdaRegression.Execution: EvalContext, beta_reduce!
    using LambdaRegression.Programs: ConstSymbolFactory, new_constant!
    using LambdaRegression.Primitives: COND
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode

    boom = x -> error("unselected branch evaluated")
    function run_conditional(condition, selected_name, rejected_name)
        rejected = AppNode(ConstNode(:boom), ConstNode(:safe))
        selected = ConstNode(selected_name)
        branches = condition ? (selected, rejected) : (rejected, selected)
        tree = AstTree(AppNode(
            AppNode(
                AppNode(ConstNode(:COND), ConstNode(:condition)),
                branches[1],
            ),
            branches[2],
        ))
        context = Dict{Symbol,Any}(
            :COND => COND,
            :condition => condition,
            :boom => boom,
            :safe => 7,
            selected_name => 7,
        )
        for _ in 1:3
            beta_reduce!(tree, context)
        end
        return tree, context
    end

    true_tree, true_context = run_conditional(true, :safe, :boom)
    false_tree, false_context = run_conditional(false, :safe, :boom)
    @test true_context[true_tree.root.name] == 7
    @test false_context[false_tree.root.name] == 7

    true_term = COND(true)
    false_term = COND(false)
    true_snapshot = deepcopy(true_term.root)
    reduction = AstTree(AppNode(true_term.root, ConstNode(:safe)))
    @test beta_reduce!(reduction) == false
    @test true_term.root == true_snapshot
    @test COND(true) === true_term
    @test COND(false) === false_term

    globals = Dict{Symbol,Any}(:c1 => :global)
    locals = Dict{Symbol,Any}(:c2 => :local)
    overlay = Dict{Symbol,Any}(:c3 => :overlay)
    context = EvalContext(
        globals=globals,
        locals=locals,
        overlay=overlay,
        symfactory=ConstSymbolFactory(prefix=:c),
    )
    generated = new_constant!(context, :value)
    @test generated.name == :c4
    @test context.overlay[:c4] == :value
    @test !haskey(globals, :c4)
    other = EvalContext(symfactory=ConstSymbolFactory(prefix=:c))
    @test new_constant!(other, :other).name == :c1
end

end # module Execution
