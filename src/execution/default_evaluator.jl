"""Prepared complete-program scoring with a fresh context and budget per sample."""
struct PreparedDefaultEvaluator{P<:Program,I<:AbstractVector{Symbol},G<:AbstractDict{Symbol},M,S}
    program::P
    input_slots::I
    global_consts::G
    make_eval_context::M
    const_symfactory_ctor::S
    max_steps::Int
    max_call_depth::Int
    native::PreparedNativeEvaluator
end

function _finite_prediction(value)::Bool
    if value isa Real
        return isfinite(value)
    elseif value isa AbstractArray || value isa Tuple
        return all(_finite_prediction, value)
    end
    return false
end

"""Prepare a complete program for repeated sample evaluation.

The returned callable owns indexed execution code (unless `copy_tree=false`),
but reads the program's current local values and supplied global values when a
sample is run. `max_steps` and `max_call_depth` are independent per-sample
engineering guards, not historical β-reduction counts.
"""
function prepare_default_evaluator(
    program::Program;
    input_slots::AbstractVector{Symbol},
    global_consts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
    make_eval_context=EvalContext,
    const_symfactory_ctor=ConstSymbolFactory,
    max_steps::Int=256,
    max_call_depth::Int=512,
    copy_tree::Bool=true,
)
    native = prepare_native_evaluator(program.tree; copy_tree)
    return PreparedDefaultEvaluator(
        program, input_slots, global_consts, make_eval_context,
        const_symfactory_ctor, max_steps, max_call_depth, native,
    )
end

function _evaluate_default_prepared(prepared::PreparedDefaultEvaluator, x)
    input_slots = prepared.input_slots
    isempty(input_slots) && return Err(ArgumentError(
        "input_slots must be non-empty for evaluate_default",
    ))

    overlay = Dict{Symbol,Any}()
    if length(input_slots) == 1
        overlay[input_slots[1]] = x
    else
        x isa Tuple || return Err(ArgumentError("multi-input expects x::Tuple"))
        length(x) == length(input_slots) ||
            return Err(ArgumentError("input tuple length mismatch"))
        for (slot, value) in zip(input_slots, x)
            overlay[slot] = value
        end
    end

    ctx = prepared.make_eval_context(
        overlay=overlay,
        locals=prepared.program.local_consts,
        globals=prepared.global_consts,
        symfactory=prepared.const_symfactory_ctor(),
    )
    arguments = length(input_slots) == 1 ? (x,) : x
    result = run_native_evaluator(
        prepared.native, ctx;
        args=arguments,
        max_steps=prepared.max_steps,
        max_call_depth=prepared.max_call_depth,
    )
    if result isa Err
        issue = result.value
        issue isa NativeStepLimitError && return Err(MaxReduceStepsErr(prepared.max_steps))
        issue isa NativeEvaluationCycleError && return Err(MaxReduceStepsErr(prepared.max_steps))
        issue isa NativeCallDepthLimitError &&
            return Err(ClosureDepthLimitErr(issue.depth, issue.limit))
        issue isa NativeMissingConstantError && return Err(MissingConstValueErr(issue.name))
        issue isa NativeEvaluationError && return Err(NonConstOutputErr())
        return result
    end
    _finite_prediction(result) || return Err(NonFiniteOutputErr())
    return result
end

(prepared::PreparedDefaultEvaluator)(x) = _evaluate_default_prepared(prepared, x)

"""Evaluate one sample using explicit task-independent execution inputs."""
function evaluate_default(program::Program, x; kwargs...)
    return prepare_default_evaluator(program; copy_tree=false, kwargs...)(x)
end

@testitem "Execution default evaluator isolates sample contexts and preserves array layers" begin
    using Test
    using LambdaRegression.Execution: EvalContext, evaluate_default,
        prepare_default_evaluator
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode

    contexts = EvalContext[]
    make_context = (; kwargs...) -> begin
        context = EvalContext(; kwargs...)
        push!(contexts, context)
        context
    end
    program = Program(AstTree(AbsNode(:xs, IndexNode(0))))
    prepared = prepare_default_evaluator(
        program;
        input_slots=[:xs],
        global_consts=Dict{Symbol,Any}(),
        make_eval_context=make_context,
    )
    first = [[1.0, 2.0], [3.0, 4.0]]
    second = [[5.0], [6.0, 7.0]]
    @test prepared(first) == first
    @test prepared(second) == second
    @test length(contexts) == 2
    @test contexts[1] !== contexts[2]
    @test contexts[1].overlay[:xs] === first
    @test contexts[2].overlay[:xs] === second
end

@testitem "Execution default evaluator accepts an upstream custom primitive" begin
    using Test
    using LambdaRegression.Execution: prepare_default_evaluator
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode

    tree = AstTree(AbsNode(:x,
        AppNode(AppNode(ConstNode(:combine), ConstNode(:bias)), IndexNode(0))))
    program = Program(tree;
        local_consts=Dict{Symbol,Any}(:bias => 1.25),
        local_consts_ty=Dict{Symbol,TyLike}(:bias => FloatTy))
    prepared = prepare_default_evaluator(program;
        input_slots=[:x],
        global_consts=Dict{Symbol,Any}(:combine => (a -> b -> a + b)))

    @test prepared(2.0) == 3.25
    @test prepared(4.0) == 5.25
end

@testitem "Execution default evaluator runs native FunctionDef and rejects non-finite output" begin
    using Test
    using LambdaRegression.Execution: NonFiniteOutputErr, evaluate_default
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AstTree, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.Utils: Err

    identity = Program(AstTree(FunctionDefNode(
        AbsNode(:x, IndexNode(0)),
    )))
    kwargs = (; input_slots=[:x], global_consts=Dict{Symbol,Any}())
    @test evaluate_default(identity, 3.5; kwargs...) == 3.5

    non_finite = Program(AstTree(AbsNode(:x, ConstNode(:NAN))))
    output = evaluate_default(
        non_finite,
        1.0;
        input_slots=[:x],
        global_consts=Dict{Symbol,Any}(:NAN => NaN),
    )
    @test output isa Err
    @test output.value isa NonFiniteOutputErr
end

@testitem "Execution default evaluator reports bounded native execution" begin
    using Test
    using LambdaRegression.Execution:
        MaxReduceStepsErr, prepare_default_evaluator
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, IndexNode, VarNode
    using LambdaRegression.Utils: Err

    nested_identity = AbsNode(
        :x,
        AppNode(
            AbsNode(:y, VarNode(:y)),
            AppNode(AbsNode(:z, VarNode(:z)), IndexNode(0)),
        ),
    )
    program = Program(AstTree(nested_identity))
    prepared = prepare_default_evaluator(
        program;
        input_slots=[:x],
        global_consts=Dict{Symbol,Any}(),
        max_steps=1,
    )

    result = prepared(7.0)
    @test result isa Err
    @test result.value isa MaxReduceStepsErr
    @test result.value.steps == 1
end
