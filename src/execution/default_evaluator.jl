"""A prepared default evaluator for one program and execution world.

The object deliberately contains execution inputs only: the program, input
slots, global values, context/symbol factories, execution budgets, and optional
native preparation. Task types and Regression components/configuration do not
belong here.
"""
struct PreparedDefaultEvaluator{P<:Program,I<:AbstractVector{Symbol},G<:AbstractDict{Symbol},M,S}
    program::P
    input_slots::I
    global_consts::G
    make_eval_context::M
    const_symfactory_ctor::S
    max_reduce_steps::Int
    native_max_steps::Int
    closure_max_reduce_steps::Int
    closure_lift_max_per_pass::Int
    closure_max_depth::Int
    native::Union{Nothing,PreparedNativeEvaluator}
end

function _execution_contains_functiondef(tree::AstTree)
    return any(node -> node isa FunctionDefNode, tree)
end

function _finite_prediction(value)::Bool
    if value isa Real
        return isfinite(value)
    elseif value isa AbstractArray || value isa Tuple
        return all(_finite_prediction, value)
    end
    return false
end

function prepare_default_evaluator(
    program::Program;
    input_slots::AbstractVector{Symbol},
    global_consts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
    make_eval_context=EvalContext,
    const_symfactory_ctor=ConstSymbolFactory,
    max_reduce_steps::Int=64,
    native_max_steps::Int=max(256, 4 * max_reduce_steps),
    closure_max_reduce_steps::Int=max_reduce_steps,
    closure_lift_max_per_pass::Int=64,
    closure_max_depth::Int=64,
    copy_tree::Bool=true,
)
    native = _execution_contains_functiondef(program.tree) ?
        prepare_native_evaluator(program.tree; copy_tree) : nothing
    return PreparedDefaultEvaluator(
        program,
        input_slots,
        global_consts,
        make_eval_context,
        const_symfactory_ctor,
        max_reduce_steps,
        native_max_steps,
        closure_max_reduce_steps,
        closure_lift_max_per_pass,
        closure_max_depth,
        native,
    )
end

function _execution_apply_inputs(program::AstTree, input_slots::AbstractVector{Symbol})::AstTree
    node = deepcopy(program.root)
    for slot in input_slots
        node = AppNode(node, ConstNode(slot))
    end
    return AstTree(node)
end

function _execution_is_closed(node::AstNode, depth::Int)::Bool
    if node isa IndexNode
        return (node::IndexNode).index < depth
    elseif node isa AbsNode
        return _execution_is_closed((node::AbsNode).body, depth + 1)
    elseif node isa AppNode
        n = node::AppNode
        return _execution_is_closed(n.func, depth) &&
               _execution_is_closed(n.arg, depth)
    else
        return node isa ConstNode
    end
end

_execution_is_closed_abs(abs::AbsNode)::Bool = _execution_is_closed(abs.body, 1)

function _capture_eval_env(ctx::EvalContext)::Dict{Symbol,Any}
    env = Dict{Symbol,Any}()
    merge!(env, ctx.globals)
    merge!(env, ctx.locals)
    merge!(env, ctx.overlay)
    return env
end

# This key and the reuse behavior are deliberate: synchronous host callbacks
# (AMAP/AFOLDL, for example) must spend the active sample's budget rather than
# starting a fresh one. An escaped LiftedClosure establishes a state only when
# it is called outside such an active sample.
const _CORE_EVAL_STATE_KEY = :LambdaRegression_core_eval_state

mutable struct _CoreEvalState
    remaining::Int
    initial::Int
    closure_depth::Int
    max_closure_depth::Int
end

function _current_core_eval_state()
    state = get(Base.task_local_storage(), _CORE_EVAL_STATE_KEY, nothing)
    return state isa _CoreEvalState ? state : nothing
end

function _with_core_eval_state(f::F, steps::Int, max_closure_depth::Int) where {F}
    state = _current_core_eval_state()
    state === nothing || return f()
    local_state = _CoreEvalState(steps, steps, 0, max_closure_depth)
    return Base.task_local_storage(() -> f(), _CORE_EVAL_STATE_KEY, local_state)
end

function _consume_core_eval_step!(state::_CoreEvalState)::Bool
    state.remaining > 0 || throw(MaxReduceStepsErr(state.initial))
    state.remaining -= 1
    return true
end

function _enter_core_closure!(state::_CoreEvalState)
    attempted_depth = state.closure_depth + 1
    attempted_depth <= state.max_closure_depth ||
        throw(ClosureDepthLimitErr(attempted_depth, state.max_closure_depth))
    state.closure_depth = attempted_depth
    return nothing
end

function _leave_core_closure!(state::_CoreEvalState)
    state.closure_depth > 0 || throw(AssertionError("core closure depth underflow"))
    state.closure_depth -= 1
    return nothing
end

struct LiftedClosure{M,S}
    ast::AstTree
    captured_env::Dict{Symbol,Any}
    make_eval_context::M
    const_symfactory_ctor::S
    max_steps::Int
    closure_lift_max_per_pass::Int
    max_closure_depth::Int
end

function _closure_lift!(
    tree::AstTree,
    ctx::EvalContext,
    make_eval_context,
    const_symfactory_ctor,
    closure_max_reduce_steps::Int,
    closure_lift_max_per_pass::Int,
    closure_max_depth::Int,
)::Bool
    max_per_pass = closure_lift_max_per_pass
    lifted = Ref(0)

    function visit(node::AstNode, under_abs::Bool)
        under_abs && return
        node isa AbsNode && return
        if node isa AppNode
            app = node::AppNode
            if app.func isa ConstNode && app.arg isa AbsNode && lifted[] < max_per_pass
                f = (app.func::ConstNode).name
                abs = app.arg::AbsNode
                _execution_is_closed_abs(abs) || throw(NonClosedLambdaErr(f))
                cl = LiftedClosure(
                    AstTree(deepcopy(abs)),
                    _capture_eval_env(ctx),
                    make_eval_context,
                    const_symfactory_ctor,
                    closure_max_reduce_steps,
                    max_per_pass,
                    closure_max_depth,
                )
                app.arg = new_constant!(ctx, cl)
                lifted[] += 1
            end
            visit(app.func, under_abs)
            visit(app.arg, under_abs)
        end
        return
    end

    visit(tree.root, false)
    return lifted[] > 0
end

function _normalize_with_lift!(
    tree::AstTree,
    ctx::EvalContext,
    make_eval_context,
    const_symfactory_ctor,
    closure_max_reduce_steps::Int,
    closure_lift_max_per_pass::Int,
    closure_max_depth::Int,
    steps::Int,
)::Bool
    state = _current_core_eval_state()
    state === nothing && throw(AssertionError("core evaluation budget is not bound"))
    for _ in 1:steps
        _consume_core_eval_step!(state) || return false
        normalized = beta_reduce!(tree, ctx)
        if normalized
            progressed = _closure_lift!(
                tree,
                ctx,
                make_eval_context,
                const_symfactory_ctor,
                closure_max_reduce_steps,
                closure_lift_max_per_pass,
                closure_max_depth,
            )
            progressed || return true
        end
    end
    return false
end

function _call_lifted_closure(cl::LiftedClosure, x)
    state = _current_core_eval_state()
    state === nothing && throw(AssertionError("core evaluation budget is not bound"))
    _enter_core_closure!(state)
    try
        call_ctx = cl.make_eval_context(
            overlay=Dict{Symbol,Any}(),
            locals=Dict{Symbol,Any}(),
            globals=cl.captured_env,
            symfactory=cl.const_symfactory_ctor(),
        )
        arg = new_constant!(call_ctx, x)
        call_tree = AstTree(AppNode(deepcopy(cl.ast.root), arg))
        # A nested closure receives its own configured loop cap, but consumes
        # the active sample's shared task-local step budget.
        ok = _normalize_with_lift!(
            call_tree,
            call_ctx,
            cl.make_eval_context,
            cl.const_symfactory_ctor,
            cl.max_steps,
            cl.closure_lift_max_per_pass,
            cl.max_closure_depth,
            cl.max_steps,
        )
        ok || throw(MaxReduceStepsErr(cl.max_steps))
        if call_tree.root isa ConstNode
            name = (call_tree.root::ConstNode).name
            haskey(call_ctx, name) || throw(MissingConstValueErr(name))
            return call_ctx[name]
        elseif call_tree.root isa AbsNode
            return LiftedClosure(
                AstTree(deepcopy(call_tree.root)),
                _capture_eval_env(call_ctx),
                cl.make_eval_context,
                cl.const_symfactory_ctor,
                cl.max_steps,
                cl.closure_lift_max_per_pass,
                cl.max_closure_depth,
            )
        else
            throw(NonConstOutputErr())
        end
    finally
        _leave_core_closure!(state)
    end
end

function (cl::LiftedClosure)(x)
    state = _current_core_eval_state()
    state === nothing && return _with_core_eval_state(
        () -> _call_lifted_closure(cl, x),
        cl.max_steps,
        cl.max_closure_depth,
    )
    return _call_lifted_closure(cl, x)
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
        length(x) == length(input_slots) || return Err(ArgumentError("input tuple length mismatch"))
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

    if prepared.native !== nothing
        arguments = length(input_slots) == 1 ? (x,) : x
        result = run_native_evaluator(
            prepared.native,
            ctx;
            args=arguments,
            max_steps=prepared.native_max_steps,
        )
        if result isa Err
            issue = result.value
            issue isa NativeStepLimitError && return Err(MaxReduceStepsErr(prepared.native_max_steps))
            issue isa NativeEvaluationCycleError && return Err(MaxReduceStepsErr(prepared.native_max_steps))
            issue isa NativeMissingConstantError && return Err(MissingConstValueErr(issue.name))
            issue isa NativeEvaluationError && return Err(NonConstOutputErr())
        end
        _finite_prediction(result) || return Err(NonFiniteOutputErr())
        return result
    end

    steps = prepared.max_reduce_steps
    return _with_core_eval_state(steps, prepared.closure_max_depth) do
        tree = _execution_apply_inputs(prepared.program.tree, input_slots)
        normalized = _normalize_with_lift!(
            tree,
            ctx,
            prepared.make_eval_context,
            prepared.const_symfactory_ctor,
            prepared.closure_max_reduce_steps,
            prepared.closure_lift_max_per_pass,
            prepared.closure_max_depth,
            steps,
        )
        normalized || return Err(MaxReduceStepsErr(steps))

        tree.root isa ConstNode || return Err(NonConstOutputErr())
        name = (tree.root::ConstNode).name
        haskey(ctx, name) || return Err(MissingConstValueErr(name))
        result = ctx[name]
        _finite_prediction(result) || return Err(NonFiniteOutputErr())
        return result
    end
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

@testitem "Execution default evaluator returns Err for incomplete core reduction" begin
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
        max_reduce_steps=1,
    )

    result = prepared(7.0)
    @test result isa Err
    @test result.value isa MaxReduceStepsErr
    @test result.value.steps == 1
end
