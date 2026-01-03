import ..ScalarSearch
using ..ASTs.NodeASTs: VarNode

# A read-only feature matrix without repeating captured parameters at every leaf.
struct _MapFeatures{A<:Array{Float64}} <: AbstractMatrix{Float64}
    input::A
    parameters::Vector{Float64}
end
Base.size(x::_MapFeatures) = (1 + length(x.parameters), length(x.input))
Base.getindex(x::_MapFeatures, i::Int, j::Int) = i == 1 ? x.input[j] : x.parameters[i - 1]

struct _BatchedMapEvaluator{E}
    fallback::E
    plan::ScalarSearch.PreparedScalarProgram
    parameters::Vector{Symbol}
    operators::Vector{Pair{Symbol,Any}}
    setup_steps::Int
    level_steps::Vector{Int}
end

_map_value(prepared, name) = haskey(prepared.program.local_consts, name) ?
    prepared.program.local_consts[name] : get(prepared.global_consts, name, nothing)

function _map_operator!(operators, prepared, node, expected)
    node isa ConstNode || return false
    node.name in prepared.input_slots && return false
    _map_value(prepared, node.name) === expected || return false
    pair = Pair{Symbol,Any}(node.name, expected)
    pair in operators || push!(operators, pair)
    return true
end

_map_argument(node, name) = node isa IndexNode && node.index == 0 ||
    node isa VarNode && node.name == name

# Only closed, registered scalar arithmetic is admitted. Values remain inputs.
function _map_scalar(node, name, prepared, parameters, operators)
    _map_argument(node, name) && return (IndexNode(0), 1)
    if node isa ConstNode
        node.name in prepared.input_slots && return nothing
        _map_value(prepared, node.name) isa Float64 || return nothing
        node.name in parameters || push!(parameters, node.name)
        return (node, 1)
    elseif node isa AppNode
        for (symbol, _) in ScalarSearch._AST_UNARY
            if _map_operator!(operators, prepared, node.func, getfield(Primitives, symbol))
                child = _map_scalar(node.arg, name, prepared, parameters, operators)
                child === nothing && return nothing
                return (AppNode(ConstNode(symbol), child[1]), 3 + child[2])
            end
        end
        if node.func isa AppNode
            for (symbol, _) in ScalarSearch._AST_BINARY
                if _map_operator!(operators, prepared, node.func.func, getfield(Primitives, symbol))
                    left = _map_scalar(node.func.arg, name, prepared, parameters, operators)
                    right = _map_scalar(node.arg, name, prepared, parameters, operators)
                    (left === nothing || right === nothing) && return nothing
                    return (AppNode(AppNode(ConstNode(symbol), left[1]), right[1]),
                        5 + left[2] + right[2])
                end
            end
        end
    end
    return nothing
end

function _map_callable(node, prepared, parameters, operators)
    node isa AppNode && _map_operator!(operators, prepared, node.func, AMAP) || return nothing
    callback = node.arg
    if callback isa AbsNode
        body = callback.body
        if body isa AppNode && _map_argument(body.arg, callback.name)
            inner = _map_callable(body.func, prepared, parameters, operators)
            inner === nothing || return (; body=inner.body, setup=4,
                levels=[inner.setup + 4; inner.levels])
        end
        scalar = _map_scalar(body, callback.name, prepared, parameters, operators)
        scalar === nothing && return nothing
        return (; body=scalar[1], setup=4, levels=[1 + scalar[2]])
    elseif callback isa AppNode
        inner = _map_callable(callback, prepared, parameters, operators)
        inner === nothing && return nothing
        return (; body=inner.body, setup=3 + inner.setup, levels=[0; inner.levels])
    elseif callback isa ConstNode
        for (symbol, _) in ScalarSearch._AST_UNARY
            if _map_operator!(operators, prepared, callback, getfield(Primitives, symbol))
                return (; body=AppNode(ConstNode(symbol), IndexNode(0)), setup=4, levels=[0])
            end
        end
    end
    return nothing
end

"""Derive a batch plan for a closed nested map, otherwise retain native execution.

Recognises registered primitive identities and scalar bodies using only their
item and Float64 constants. Dense Float64 rank must match the map depth at run
time. Custom contexts, other functions/bindings and general programs fall back.
Source syntax is unchanged. Each invocation owns output and workspace; captured
values are read afresh. No candidate-topology-specific kernel is compiled.
"""
function _prepare_batched_map(prepared, components)
    length(prepared.input_slots) == 1 || return prepared
    (prepared.make_eval_context === EvalContext ||
        prepared.make_eval_context === get(components, :_parallel_default_eval_context, nothing)) || return prepared
    prepared.const_symfactory_ctor in (_default_const_symfactory, ConstSymbolFactory) || return prepared
    root = prepared.native.root
    extra_steps = 1
    if root isa AbsNode
        root.body isa AppNode && _map_argument(root.body.arg, root.name) || return prepared
        root = root.body.func
        extra_steps = 5
    end
    parameters = Symbol[]
    operators = Pair{Symbol,Any}[]
    spec = _map_callable(root, prepared, parameters, operators)
    spec === nothing && return prepared
    dependencies = AstNode[IndexNode(0), (ConstNode(name) for name in parameters)...]
    expression = ScalarSearch.scalar_from_ast(spec.body, dependencies)
    expression isa Err && return prepared
    return _BatchedMapEvaluator(prepared, ScalarSearch.prepare_scalar(expression),
        parameters, operators, spec.setup + extra_steps, spec.levels)
end

function _map_budget_fits(evaluator::_BatchedMapEvaluator, input)
    remaining = evaluator.fallback.max_steps - evaluator.setup_steps
    remaining >= 0 || return false
    # Conservative for partially applied host maps, exact for admitted lambda bodies.
    evaluator.fallback.max_call_depth >= length(evaluator.level_steps) + 1 || return false
    calls = 1
    for (axis, cost) in enumerate(evaluator.level_steps)
        calls *= size(input, axis)
        if cost > 0
            calls <= remaining ÷ cost || return false
            remaining -= calls * cost
        end
    end
    return true
end

function (evaluator::_BatchedMapEvaluator)(input)
    fallback = evaluator.fallback
    input isa Array{Float64} && ndims(input) == length(evaluator.level_steps) &&
        !isempty(input) && _map_budget_fits(evaluator, input) || return fallback(input)
    all(pair -> _map_value(fallback, pair.first) === pair.second, evaluator.operators) || return fallback(input)
    parameters = Float64[]
    for name in evaluator.parameters
        value = _map_value(fallback, name)
        value isa Float64 && isfinite(value) || return fallback(input)
        push!(parameters, value)
    end
    output = Vector{Float64}(undef, length(input))
    result = ScalarSearch.evaluate_scalar!(output, evaluator.plan,
        _MapFeatures(input, parameters), ScalarSearch.ScalarWorkspace())
    # Native execution defines domain/error order and permits recoverable Inf intermediates.
    result isa Err && return fallback(input)
    return reshape(output, size(input))
end

@testitem "Regression batched maps preserve complete-program scoring semantics" begin
    using Test
    import LambdaRegression.Regression as R
    import LambdaRegression.Execution as E
    import LambdaRegression.Primitives as P
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode, FunctionDefNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Utils: Err
    globals = Dict{Symbol,Any}(:map => P.AMAP, :square => P.SQR, :plus => P.FADD,
        :exp => P.FEXP, :divide => P.FDIV, :sqrt => P.FSQRT, :c => 2.0)
    components = R.default_components(global_consts=globals, input_slots=[:x])
    cfg = R.default_config(procedural_max_steps=100000)
    scalar = AbsNode(:v, AppNode(AppNode(ConstNode(:plus),
        AppNode(ConstNode(:square), IndexNode(0))), ConstNode(:c)))
    nested(fn) = AppNode(ConstNode(:map), AppNode(ConstNode(:map), fn))
    program = Program(AstTree(AbsNode(:x, AppNode(nested(scalar), IndexNode(0)))))
    fast = R.prepare_regression_evaluator(program, components, cfg)
    native = R.prepare_regression_evaluator(program, components, merge(cfg, (; batch_scalar_maps=false)))
    @test fast isa R._BatchedMapEvaluator
    x = reshape(collect(1.0:12.0), 3, 4)
    @test fast(x) == native(x) == x.^2 .+ 2
    saved = fast(x)
    @test fast(2x) == native(2x)
    @test saved == x.^2 .+ 2
    @test x == reshape(collect(1.0:12.0), 3, 4)
    globals[:c] = 3.0
    @test fast(x) == native(x) == x.^2 .+ 3
    globals[:square] = v -> v + 10
    @test fast(x) == native(x) == x .+ 13
    globals[:square] = P.SQR
    @test fast(Float32.(x)) == native(Float32.(x))
    @test fast(view(x, :, :)) == native(view(x, :, :))
    @test fast([collect(row) for row in eachrow(x)]) == native([collect(row) for row in eachrow(x)])
    @test fast(zeros(0, 4)) == native(zeros(0, 4))
    @test R.score_samples(R.SupervisedDataset([x, 2x], [native(x), native(2x)]), fast, R.mse;
        failure_mode=:return_error).score == 0

    expanded = Program(AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:map),
        AbsNode(:row, AppNode(AppNode(ConstNode(:map), scalar), VarNode(:row)))), IndexNode(0)))))
    for p in (program, expanded, Program(AstTree(nested(scalar))),
              Program(AstTree(AbsNode(:x, AppNode(nested(ConstNode(:square)), IndexNode(0))))))
        accelerated = R.prepare_regression_evaluator(p, components, cfg)
        @test accelerated isa R._BatchedMapEvaluator
        for budget in (0, 1, 8, 15, 50, 100, 112, 150, 200, 1000)
            limits = merge(cfg, (; procedural_max_steps=budget))
            a = R.prepare_regression_evaluator(p, components, limits)(x)
            b = R.prepare_regression_evaluator(p, components, merge(limits, (; batch_scalar_maps=false)))(x)
            @test a == b
        end
    end
    shallow = merge(cfg, (; closure_max_depth=1))
    @test R.prepare_regression_evaluator(expanded, components, shallow)(x) ==
        R.prepare_regression_evaluator(expanded, components, merge(shallow, (; batch_scalar_maps=false)))(x)

    body = AppNode(AppNode(ConstNode(:divide), ConstNode(:c)), AppNode(ConstNode(:exp), IndexNode(0)))
    recovering = Program(AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:map), AbsNode(:v, body)), IndexNode(0)))))
    @test R.prepare_regression_evaluator(recovering, components, cfg)([1000.0]) == [0.0]
    invalid = Program(AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:map),
        AbsNode(:v, AppNode(ConstNode(:sqrt), IndexNode(0)))), IndexNode(0)))))
    @test_throws DomainError R.prepare_regression_evaluator(invalid, components, cfg)([-1.0])
    @test R.prepare_regression_evaluator(program, components, cfg)(fill(Inf, 3, 4)) isa Err
    globals[:plus] = (a -> b -> a - b)
    @test R.prepare_regression_evaluator(program, components, cfg) isa E.PreparedDefaultEvaluator
    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    @test R.prepare_regression_evaluator(identity, components, cfg) isa E.PreparedDefaultEvaluator
    custom = merge(components, (; make_eval_context=(; kwargs...) -> E.EvalContext(; kwargs...)))
    @test R.prepare_regression_evaluator(program, custom, cfg) isa E.PreparedDefaultEvaluator
    injected = merge(components, (; evaluate=(args...) -> :custom))
    @test R.prepare_task_evaluator(program, R.TensorSchema((3,4)).ty, injected, cfg)(x) === :custom
end

@testitem "Regression batched maps integrate with collection fit and prediction" begin
    using Test
    import LambdaRegression.Regression as R
    import LambdaRegression.FrontendAdapter as A
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.Programs: Program
    body = AbsNode(:v, AppNode(ConstNode(:SQR), IndexNode(0)))
    mapped = AppNode(ConstNode(:AMAP), AppNode(ConstNode(:AMAP), body))
    seed = Program(AstTree(AbsNode(:x, AppNode(mapped, IndexNode(0)))))
    X = reshape(collect(1.0:24.0), 2, 3, 4)
    fitted = A.fit_collection(X, X.^2; task_kind=:array, initial_programs=[seed],
        population_size=2, generations=0, operator_set=[:square], random_state=42,
        regressor_config=(procedural_max_steps=10000,))
    model = fitted["model"]
    @test R.prepare_task_evaluator(model.program, model.target_ty, model.components,
        model.config) isa R._BatchedMapEvaluator
    @test A.predict(model, X) == X.^2
    @test model.sample_executions == model.objective_calls * size(X, 1)
end
