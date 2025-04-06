"""Private constant refinement for owned node-native scalar candidates.

This seam deliberately owns no search state.  A trial receives a fresh
`DynamicExpressions.Node{Float64}` copy, constants are written into that copy,
and only a strictly improving finite score is returned.  The authoritative
candidate and its structure are never mutated.
"""

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
    evaluations = Ref(0)
    invalid_evaluations = Ref(0)
    max_evaluations <= 0 && return (candidate, Float64(loss), 0, 0)
    isempty(paths) && return (candidate, Float64(loss), 0, 0)
    config.constant_refinement_iterations == 0 &&
        return (candidate, Float64(loss), 0, 0)
    restarts >= 0 || throw(ArgumentError("constant-refinement restarts cannot be negative"))

    initial = Float64[_node_at_path(candidate.node, path).val for path in paths]
    best_trial_node = Ref{Union{Nothing,DynamicExpressions.Node{Float64}}}(nothing)
    best_trial_loss = Ref(Inf)
    function trial(values)
        evaluations[] >= max_evaluations &&
            throw(_ConstantRefinementBudgetExhausted())
        node = DynamicExpressions.copy_node(candidate.node;
            break_sharing=Val(true))
        _node_set_constants!(node, paths, values) || return Inf
        candidate_loss, invalid = _node_refinement_score(score_node, node)
        evaluations[] += 1
        invalid_evaluations[] += invalid
        if !invalid && isfinite(candidate_loss) &&
                candidate_loss < best_trial_loss[]
            best_trial_node[] = DynamicExpressions.copy_node(node;
                break_sharing=Val(true))
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
    if best_trial_node[] === nothing || !(best_trial_loss[] < loss)
        return (candidate, Float64(loss), evaluations[], invalid_evaluations[])
    end
    refined = _NodeScalarCandidate(best_trial_node[], candidate.complexity)
    return (refined, best_trial_loss[], evaluations[], invalid_evaluations[])
end

@testsnippet NodeConstantRefinementImports begin
    using Test
    using LambdaRegression.ScalarSearch
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
