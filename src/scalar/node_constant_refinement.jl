"""Private constant refinement for owned node-native scalar candidates.

This seam deliberately owns no search state.  A trial receives a fresh
`DynamicExpressions.Node{Float64}` copy, constants are written into that copy,
and only a strictly improving finite score is returned.  The authoritative
candidate and its structure are never mutated.
"""

"""One island's direct MSE scorer and optional constant-Jacobian callback.

Inputs are borrowed read-only; evaluation buffers and diagnostic counters are
island-owned. The callback is synchronous and must not mutate its node or X.
It returns `(prediction, jacobian, complete)` in depth-first constant-leaf
order, with a constants-by-samples Jacobian. It never supplies the authoritative
loss. Invalid derivatives fall back to charged central differences.
"""
mutable struct _NodeMSEScore{X,Y,J,W}
    X::X
    y::Y
    workspace::W
    jacobian::J
    derivative_evaluations::Int
    invalid_derivatives::Int
    fallback_gradients::Int
end

_NodeMSEScore(X, y, jacobian) =
    _NodeMSEScore(X, y, DynamicScalarWorkspace(), jacobian, 0, 0, 0)

function _node_mse_score(score::_NodeMSEScore, node)
    prediction = evaluate_dynamic_scalar!(score.workspace, node, score.X)
    prediction isa Err && return (Inf, true, nothing)
    total = 0.0
    @inbounds @simd for index in eachindex(score.y)
        residual = prediction[index] - score.y[index]
        total += residual * residual
    end
    loss = total / length(score.y)
    return isfinite(loss) ? (loss, false, prediction) : (Inf, true, nothing)
end

(score::_NodeMSEScore)(value::_NodeScalarCandidate) =
    _node_mse_score(score, value.node)[1:2]

_node_constant_work(score, count) = (forward_evaluations=count,
    derivative_evaluations=0, invalid_derivatives=0, fallback_gradients=0)
_node_constant_work(score::_NodeMSEScore, count) = (
    forward_evaluations=count - score.derivative_evaluations,
    derivative_evaluations=score.derivative_evaluations,
    invalid_derivatives=score.invalid_derivatives,
    fallback_gradients=score.fallback_gradients)

"""Bounded BFGS using an explicit direct-MSE constant Jacobian.

Every canonical forward evaluation and every Jacobian callback consumes one
unit of the same finite allowance. Counters separate these unequal-cost units.
Only canonical forward evaluations can publish a best trial. Each trial owns
its tree; derivative singularities use charged central differences, including
retention of a better perturbation if the allowance expires mid-gradient.
"""
function _node_refine_mse_jacobian(candidate::_NodeScalarCandidate, loss::Real,
        score::_NodeMSEScore, config::ScalarSearchConfig, cap::Integer;
        rng::AbstractRNG=default_rng(), restarts::Int=2)
    config.constant_refinement_optimizer === :bfgs ||
        throw(ArgumentError("constant Jacobians require BFGS"))
    restarts >= 0 || throw(ArgumentError("constant-refinement restarts cannot be negative"))
    paths = _node_constant_paths(candidate.node)
    initial = Float64[_node_at_path(candidate.node, p).val for p in paths]
    (cap <= 0 || isempty(initial) || config.constant_refinement_iterations == 0) &&
        return (candidate, Float64(loss), 0, 0)
    work = Ref(0); invalid = Ref(0)
    best_loss = Ref(Float64(loss)); best_values = copy(initial)
    function charge!()
        work[] >= cap && throw(_ConstantRefinementBudgetExhausted())
        work[] += 1
    end
    function owned(values)
        node = DynamicExpressions.copy_node(candidate.node; break_sharing=Val(true))
        _node_set_constants!(node, paths, values) || error("invalid refinement parameters")
        return node
    end
    function forward(node, values)
        charge!()
        value, failed, prediction = _node_mse_score(score, node)
        invalid[] += failed
        if !failed && value < best_loss[]
            best_loss[] = value
            copyto!(best_values, values)
        end
        return value, prediction
    end
    function trial(values)
        all(isfinite, values) || return Inf
        return first(forward(owned(values), values))
    end
    function fg!(F, G, values)
        G === nothing && return trial(values)
        if !all(isfinite, values)
            fill!(G, NaN)
            return F === nothing ? nothing : Inf
        end
        node = owned(values)
        value, prediction = forward(node, values)
        if isfinite(value)
            charge!()
            score.derivative_evaluations += 1
            _, jacobian, complete = score.jacobian(node, score.X)
            if complete
                size(jacobian) == (length(values), length(score.y)) ||
                    throw(DimensionMismatch("constant Jacobian must be constants by samples"))
                for j in eachindex(G)
                    total = 0.0
                    @inbounds @simd for i in eachindex(score.y)
                        total += (prediction[i] - score.y[i]) * jacobian[j, i]
                    end
                    G[j] = 2total / length(score.y)
                end
                complete = all(isfinite, G)
            end
            if complete
                return F === nothing ? nothing : value
            end
            score.invalid_derivatives += 1
        end
        score.fallback_gradients += 1
        for j in eachindex(values)
            h = cbrt(eps(Float64)) * max(1.0, abs(values[j]))
            plus, minus = copy(values), copy(values)
            plus[j] += h
            minus[j] -= h
            G[j] = (trial(plus) - trial(minus)) / (2h)
        end
        return F === nothing ? nothing : value
    end
    options = Optim.Options(iterations=config.constant_refinement_iterations,
        show_trace=false, store_trace=false)
    method = Optim.BFGS(linesearch=Optim.LineSearches.BackTracking())
    try
        Optim.optimize(Optim.NLSolversBase.only_fg!(fg!), initial, method, options)
        for _ in 1:restarts
            work[] >= cap && break
            restart = initial .* (1 .+ 0.5 .* randn(rng, length(initial)))
            Optim.optimize(Optim.NLSolversBase.only_fg!(fg!), restart, method, options)
        end
    catch exception
        exception isa _ConstantRefinementBudgetExhausted || rethrow()
    end
    best_loss[] < loss || return (candidate, Float64(loss), work[], invalid[])
    return (_NodeScalarCandidate(owned(best_values), candidate.complexity),
        best_loss[], work[], invalid[])
end

function _node_constant_paths!(paths::Vector{Tuple{Vararg{Int}}}, node,
        path::Tuple=())
    node.degree == 0 && node.constant && push!(paths, path)
    node.degree >= 1 && _node_constant_paths!(paths, node.l, (path..., 1))
    node.degree == 2 && _node_constant_paths!(paths, node.r, (path..., 2))
    return paths
end

_node_constant_paths(node::DynamicExpressions.Node{Float64}) =
    _node_constant_paths!(Tuple{Vararg{Int}}[], node)

function _node_at_path(node::DynamicExpressions.Node{Float64},
        path::Tuple{Vararg{Int}})
    current = node
    for side in path
        current = side == 1 ? current.l : side == 2 ? current.r :
            throw(ArgumentError("invalid node constant path"))
    end
    return current
end

function _node_set_constants!(node::DynamicExpressions.Node{Float64},
        paths, values)
    for (path, value) in zip(paths, values)
        isfinite(value) || return false
        leaf = _node_at_path(node, path)
        leaf.degree == 0 && leaf.constant ||
            throw(ArgumentError("constant path changed during node refinement"))
        leaf.val = Float64(value)
    end
    return true
end

function _node_refinement_score(score_node, node)
    result = score_node(node)
    if result isa Tuple
        length(result) >= 2 || throw(ArgumentError(
            "score_node tuple result must contain (loss, invalid)"))
        loss, invalid = result[1], result[2]
        return (Float64(loss), Bool(invalid))
    end
    result isa Err && return (Inf, true)
    loss = Float64(result)
    return (loss, !isfinite(loss))
end

"""
    _node_refine_constants(candidate, loss, score_node, config, max_evaluations)

Refine all finite constant leaves with the configured Optim BFGS or Nelder–
Mead method. BFGS uses Optim's BackTracking line search and central finite
differences. `score_node` receives a freshly copied node and returns either a
loss or `(loss, invalid)`. Evaluation calls are bounded by `max_evaluations`;
invalid callback outcomes are counted separately. Budget exhaustion stops the
optimizer with a private signal rather than changing its objective to infinity.
The best actually scored finite valid trial is retained, including when
exhaustion interrupts a finite-difference gradient. A candidate is committed
only when its loss is strictly lower than the supplied `loss`.
"""
function _node_refine_constants(candidate::_NodeScalarCandidate, loss::Real,
        score_node, config::ScalarSearchConfig, max_evaluations::Integer;
        rng::AbstractRNG=default_rng(), restarts::Int=2)
    paths = _node_constant_paths(candidate.node)
    initial = Float64[_node_at_path(candidate.node, path).val for path in paths]
    function score_parameters(values)
        node = DynamicExpressions.copy_node(candidate.node; break_sharing=Val(true))
        _node_set_constants!(node, paths, values) || return (Inf, true)
        return _node_refinement_score(score_node, node)
    end
    parameters, refined_loss, count, invalid = _refine_parameter_vector(initial,
        loss, score_parameters, config, max_evaluations; rng, restarts)
    refined_loss < loss || return (candidate, Float64(loss), count, invalid)
    node = DynamicExpressions.copy_node(candidate.node; break_sharing=Val(true))
    _node_set_constants!(node, paths, parameters) || error("invalid retained refinement")
    return (_NodeScalarCandidate(node, candidate.complexity), refined_loss, count, invalid)
end

"""Bounded Optim adaptation shared by scalar nodes and complete configurations.

`initial` is one entry per independent parameter. The synchronous callback
returns `(loss, invalid)`. Keep the best actually scored vector on exhaustion;
unexpected errors escape. This helper owns no syntax, aliases or search state.
"""
function _refine_parameter_vector(initial::Vector{Float64}, loss::Real,
        score_parameters, config::ScalarSearchConfig, max_evaluations::Integer;
        rng::AbstractRNG=default_rng(), restarts::Int=2)
    evaluations = Ref(0)
    invalid_evaluations = Ref(0)
    max_evaluations <= 0 && return (initial, Float64(loss), 0, 0)
    isempty(initial) && return (initial, Float64(loss), 0, 0)
    config.constant_refinement_iterations == 0 &&
        return (initial, Float64(loss), 0, 0)
    restarts >= 0 || throw(ArgumentError("constant-refinement restarts cannot be negative"))

    best_trial_values = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_trial_loss = Ref(Inf)
    function trial(values)
        evaluations[] >= max_evaluations &&
            throw(_ConstantRefinementBudgetExhausted())
        all(isfinite, values) || return Inf
        candidate_loss, invalid = score_parameters(values)
        evaluations[] += 1
        invalid_evaluations[] += invalid
        if !invalid && isfinite(candidate_loss) &&
                candidate_loss < best_trial_loss[]
            best_trial_values[] = copy(values)
            best_trial_loss[] = candidate_loss
        end
        return !invalid && isfinite(candidate_loss) ? candidate_loss : Inf
    end

    options = Optim.Options(iterations=config.constant_refinement_iterations,
        show_trace=false, store_trace=false)
    function optimize_from(start)
        if config.constant_refinement_optimizer === :bfgs
            method = Optim.BFGS(linesearch=Optim.LineSearches.BackTracking())
            return Optim.optimize(trial, start, method, options;
                autodiff=Optim.ADTypes.AutoFiniteDiff(fdtype=Val(:central)))
        end
        return Optim.optimize(trial, start, Optim.NelderMead(), options)
    end
    try
        optimize_from(initial)
        for _ in 1:restarts
            evaluations[] >= max_evaluations && break
            restart = initial .* (1 .+ 0.5 .* randn(rng, length(initial)))
            optimize_from(restart)
        end
    catch exception
        exception isa _ConstantRefinementBudgetExhausted || rethrow()
    end
    if best_trial_values[] === nothing || !(best_trial_loss[] < loss)
        return (initial, Float64(loss), evaluations[], invalid_evaluations[])
    end
    return (best_trial_values[], best_trial_loss[], evaluations[], invalid_evaluations[])
end

@testsnippet NodeConstantRefinementImports begin
    using Test
    using LambdaRegression.ScalarSearch
end

@testitem "direct MSE Jacobian refinement charges work and preserves publication" setup=[NodeConstantRefinementImports] begin
    using Random: MersenneTwister
    S = ScalarSearch
    X = reshape([0.2, 0.5, 0.9, 1.3], 1, :)
    y = 2vec(X) .+ 0.7
    expr = scalar_binary(:+, scalar_binary(:*, scalar_constant(0.0),
        scalar_input(1)), scalar_constant(0.0))
    parent = S._node_scalar_candidate(expr)
    before = copy(parent.node; break_sharing=Val(true))
    config = ScalarSearchConfig(constant_refinement_iterations=24)
    callbacks = Ref(0)
    jacobian = (node, data) -> begin
        callbacks[] += 1
        (nothing, vcat(data, ones(1, size(data, 2))), true)
    end
    for cap in (0, 1, 2, 3, 5, 100)
        callbacks[] = 0
        score = S._NodeMSEScore(X, y, jacobian)
        initial = first(score(parent))
        fitted, value, work, invalid = S._regularized_refine_value(nothing,
            parent, initial, score, config, cap; rng=MersenneTwister(1111), restarts=0)
        counts = S._node_constant_work(score, work)
        @test work <= cap
        @test counts.forward_evaluations + counts.derivative_evaluations == work
        @test counts.derivative_evaluations == callbacks[]
        @test invalid == 0
        @test first(score(fitted)) == value <= initial
        @test parent.node == before
        cap == 100 && @test value < 1e-10
    end
    # A finite forward with a singular derivative keeps its finite-difference
    # improvement even when the cap expires midway through the fallback.
    singular = (node, data) -> (nothing, fill(NaN, 2, size(data, 2)), true)
    score = S._NodeMSEScore(X, y, singular)
    initial = first(score(parent))
    fitted, value, work, invalid = S._regularized_refine_value(nothing,
        parent, initial, score, config, 3; rng=MersenneTwister(1111), restarts=0)
    @test work == 3
    @test score.invalid_derivatives == score.fallback_gradients == 1
    @test value == first(score(fitted)) < initial
    @test invalid == 0
    bad_shape = S._NodeMSEScore(X, y, (n, x) -> (nothing, zeros(1, 1), true))
    @test_throws DimensionMismatch S._regularized_refine_value(nothing,
        parent, initial, bad_shape, config, 8; rng=MersenneTwister(1111))
    crashing = S._NodeMSEScore(X, y, (n, x) -> error("derivative bug"))
    @test_throws ErrorException S._regularized_refine_value(nothing,
        parent, initial, crashing, config, 8; rng=MersenneTwister(1111))
    no_constants = S._node_scalar_candidate(scalar_input(1))
    @test S._regularized_refine_value(nothing, no_constants, 1.0, crashing,
        config, 8; rng=MersenneTwister(1111)) == (no_constants, 1.0, 0, 0)
end

@testitem "node constant refinement preserves parent and obeys cap" setup=[NodeConstantRefinementImports] begin
    config = ScalarSearchConfig(constant_refinement_iterations=24,
        constant_refinement_optimizer=:bfgs)
    parent = ScalarSearch._node_scalar_candidate(scalar_constant(0.0))
    before = copy(parent.node; break_sharing=Val(true))
    calls = Ref(0)
    seen_losses = Float64[]
    score = node -> begin
        calls[] += 1
        score_loss = (node.val - 2.0)^2
        push!(seen_losses, score_loss)
        return (score_loss, false)
    end
    one_call, one_loss, one_evaluation, one_invalid =
        ScalarSearch._node_refine_constants(parent, 4.0, score, config, 1;
            restarts=0)
    @test one_call === parent
    @test one_loss == 4.0
    @test one_evaluation == calls[] == 1
    @test one_invalid == 0

    calls[] = 0
    empty!(seen_losses)
    capped, capped_loss, capped_evaluations, capped_invalid =
        ScalarSearch._node_refine_constants(parent, 4.0, score, config, 3)
    @test capped_loss == min(4.0, minimum(seen_losses))
    @test (capped.node.val - 2.0)^2 == capped_loss
    @test capped_evaluations <= 3
    @test calls[] == capped_evaluations
    @test capped_invalid == 0
    calls[] = 0
    empty!(seen_losses)
    refined, refined_loss, evaluations, invalid =
        ScalarSearch._node_refine_constants(parent, 4.0, score, config, 16)
    @test refined_loss < 4.0
    @test refined_loss == minimum(seen_losses)
    @test 0 < evaluations <= 16
    @test calls[] == evaluations
    @test invalid == 0
    @test parent.node == before
    @test refined.node.val != parent.node.val

    no_improvement = node -> (0.0, false)
    unchanged, unchanged_loss, _, _ = ScalarSearch._node_refine_constants(
        parent, 0.0, no_improvement, config, 8; restarts=0)
    @test unchanged === parent
    @test unchanged_loss == 0.0

    failing_score = _ -> error("node refinement callback failure")
    @test_throws ErrorException ScalarSearch._node_refine_constants(
        parent, 4.0, failing_score, config, 3; restarts=0)
end

@testitem "node constant refinement has zero cost without constants" setup=[NodeConstantRefinementImports] begin
    config = ScalarSearchConfig(constant_refinement_iterations=24)
    parent = ScalarSearch._node_scalar_candidate(scalar_input(1))
    calls = Ref(0)
    score = node -> (calls[] += 1; 0.0)
    result = ScalarSearch._node_refine_constants(parent, 1.0, score, config, 10)
    @test result == (parent, 1.0, 0, 0)
    @test calls[] == 0
end

@testitem "C skeleton central BFGS fits coupled constants within a bounded call count" setup=[NodeConstantRefinementImports] begin
    # This fixed-skeleton diagnostic uses known structure and constants only
    # to test numerical polishing; it is not a symbolic discovery trial.
    function c_skeleton(constants)
        c0, c1, c2 = constants
        x, a, b = scalar_input(1), scalar_input(2), scalar_input(3)
        denominator = scalar_binary(:+, scalar_constant(c2),
            scalar_binary(:*, b, x))
        quotient = scalar_binary(:/, scalar_constant(c1), denominator)
        scalar_binary(:-, scalar_binary(:+, a, scalar_constant(c0)), quotient)
    end

    xs = Float64[
        0.10 0.25 0.40 0.55 0.70 0.85 1.00 1.15 0.17 0.32 0.47 0.62;
        0.22 0.37 0.52 0.67 0.82 0.97 1.12 0.12 0.29 0.44 0.59 0.74;
        0.34 0.49 0.64 0.79 0.94 1.09 0.19 0.36 0.51 0.66 0.81 0.96;
        0.46 0.61 0.76 0.91 1.06 0.16 0.33 0.48 0.63 0.78 0.93 1.08;
    ]
    b = Float64[0.4, 1.1, 0.7, 1.4, 0.6, 1.3, 0.9, 0.5, 1.2, 0.8, 1.5, 0.65]
    function fold_result(expr)
        accumulator = zeros(Float64, length(b))
        for step in axes(xs, 1)
            X = Matrix{Float64}(undef, 3, length(b))
            X[1, :] .= xs[step, :]
            X[2, :] .= accumulator
            X[3, :] .= b
            accumulator = evaluate_scalar_reference(expr, X)
        end
        return accumulator
    end

    target = fold_result(c_skeleton((0.75, 0.8, 0.55)))
    loss_for(expr) = sum(abs2, fold_result(expr) .- target) / length(target)
    initial = c_skeleton((1.17, 1.001, 0.77))
    initial_loss = loss_for(initial)
    config = ScalarSearchConfig(constant_refinement_iterations=64,
        constant_refinement_optimizer=:bfgs,
        max_constant_refinement_evaluations=2_000)

    scalar_calls = Ref(0)
    scalar_score = expr -> (scalar_calls[] += 1; (loss_for(expr), false))
    scalar_refined, scalar_loss, scalar_evaluations, scalar_invalid =
        ScalarSearch._refine_constants(initial, initial_loss, scalar_score,
            config, 2_000)
    @test scalar_loss <= 1.0e-8
    @test scalar_evaluations == scalar_calls[]
    @test 0 < scalar_evaluations <= 2_000
    @test scalar_invalid == 0
    @test loss_for(scalar_refined) == scalar_loss

    node_candidate = ScalarSearch._node_scalar_candidate(initial)
    node_calls = Ref(0)
    node_score = node -> begin
        node_calls[] += 1
        return (loss_for(ScalarSearch.scalar_from_dynamic(node)), false)
    end
    node_refined, node_loss, node_evaluations, node_invalid =
        ScalarSearch._node_refine_constants(node_candidate, initial_loss,
            node_score, config, 2_000; restarts=0)
    @test node_loss <= 1.0e-8
    @test node_evaluations == node_calls[]
    @test 0 < node_evaluations <= 2_000
    @test node_invalid == 0
    @test loss_for(ScalarSearch.scalar_from_dynamic(node_refined.node)) == node_loss
end
