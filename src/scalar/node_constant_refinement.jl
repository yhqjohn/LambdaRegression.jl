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
Mead method.  `score_node` receives a freshly copied node and returns either a
loss or `(loss, invalid)`.  Evaluation calls are bounded by
`max_evaluations`; invalid callback outcomes are counted separately.  A
candidate is committed only when its finite loss is strictly lower than the
supplied `loss`.
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
    function trial(values)
        evaluations[] >= max_evaluations && return Inf
        node = DynamicExpressions.copy_node(candidate.node;
            break_sharing=Val(true))
        _node_set_constants!(node, paths, values) || return Inf
        candidate_loss, invalid = _node_refinement_score(score_node, node)
        evaluations[] += 1
        invalid_evaluations[] += invalid
        return isfinite(candidate_loss) ? candidate_loss : Inf
    end

    options = Optim.Options(iterations=config.constant_refinement_iterations,
        show_trace=false, store_trace=false)
    function optimize_from(start)
        if config.constant_refinement_optimizer === :bfgs
            return Optim.optimize(trial, start, Optim.BFGS(), options;
                autodiff=Optim.ADTypes.AutoFiniteDiff())
        end
        return Optim.optimize(trial, start, Optim.NelderMead(), options)
    end
    result = optimize_from(initial)
    for _ in 1:restarts
        evaluations[] >= max_evaluations && break
        restart = initial .* (1 .+ 0.5 .* randn(rng, length(initial)))
        candidate_result = optimize_from(restart)
        Optim.minimum(candidate_result) < Optim.minimum(result) &&
            (result = candidate_result)
    end
    refined_loss = try
        Float64(Optim.minimum(result))
    catch
        Inf
    end
    values = Optim.minimizer(result)
    refined_node = DynamicExpressions.copy_node(candidate.node;
        break_sharing=Val(true))
    valid_values = _node_set_constants!(refined_node, paths, values)
    if !valid_values || !isfinite(refined_loss) || !(refined_loss < loss)
        return (candidate, Float64(loss), evaluations[], invalid_evaluations[])
    end
    refined = _NodeScalarCandidate(refined_node, candidate.complexity)
    return (refined, refined_loss, evaluations[], invalid_evaluations[])
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
    score = node -> begin
        calls[] += 1
        return ((node.val - 2.0)^2, false)
    end
    capped, capped_loss, capped_evaluations, capped_invalid =
        ScalarSearch._node_refine_constants(parent, 4.0, score, config, 3)
    @test capped === parent
    @test capped_loss == 4.0
    @test capped_evaluations <= 3
    @test capped_invalid == 0
    calls[] = 0
    refined, refined_loss, evaluations, invalid =
        ScalarSearch._node_refine_constants(parent, 4.0, score, config, 16)
    @test refined_loss < 4.0
    @test 0 < evaluations <= 16
    @test calls[] == evaluations
    @test invalid == 0
    @test parent.node == before
    @test refined.node.val != parent.node.val
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
