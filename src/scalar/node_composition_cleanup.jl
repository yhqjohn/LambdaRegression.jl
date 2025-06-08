"""Return one small, nonzero coefficient seed for a composition proposal."""
const _NODE_COMPOSITION_COEFFICIENT_SEEDS =
    (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

"""Build an owned local `f + α*g` or `α*f + β*g` proposal.

One subtree is selected from the parent and one from its donor; they share the
same region-local feature numbering. The replacement is grafted at the parent
subtree, preserving its surrounding context. The parent supplies `f` in either
form. Only configured `+` and `*` operators are used. This function constructs
syntax only; regularized search scores the complete value and may then use its
existing bounded constant refinement.
"""
function _node_coefficient_proposal(rng::AbstractRNG,
        parent::_NodeScalarCandidate, donor::_NodeScalarCandidate,
        feature_count::Int, config::ScalarSearchConfig)
    :+ in config.binary_operators && :* in config.binary_operators ||
        return (parent, false)

    root = copy(parent.node; break_sharing=Val(true))
    parent_nodes = _node_anchors(root)
    donor_nodes = _node_anchors(donor.node)
    plus = _dynamic_binary_index(:+)
    multiply = _dynamic_binary_index(:*)
    for _ in 1:4
        target = rand(rng, parent_nodes)
        donor_anchor = rand(rng, donor_nodes)
        use_two_coefficients = rand(rng, Bool)
        target_size = DynamicExpressions.count_nodes(target.node)
        donor_size = DynamicExpressions.count_nodes(donor_anchor.node)
        replacement_size = target_size + donor_size +
            (use_two_coefficients ? 5 : 3)
        parent_size = parent.complexity - target_size + replacement_size
        parent_size <= config.max_complexity || continue

        f, g = target.node, donor_anchor.node
        if use_two_coefficients
            alpha = rand(rng, _NODE_COMPOSITION_COEFFICIENT_SEEDS)
            beta = rand(rng, _NODE_COMPOSITION_COEFFICIENT_SEEDS)
            left = DynamicExpressions.Node(Float64; op=multiply,
                l=DynamicExpressions.Node(Float64; val=alpha),
                r=copy(f; break_sharing=Val(true)))
            right = DynamicExpressions.Node(Float64; op=multiply,
                l=DynamicExpressions.Node(Float64; val=beta),
                r=copy(g; break_sharing=Val(true)))
            replacement = DynamicExpressions.Node(Float64; op=plus,
                l=left, r=right)
        else
            alpha = rand(rng, _NODE_COMPOSITION_COEFFICIENT_SEEDS)
            scaled_g = DynamicExpressions.Node(Float64; op=multiply,
                l=DynamicExpressions.Node(Float64; val=alpha),
                r=copy(g; break_sharing=Val(true)))
            replacement = DynamicExpressions.Node(Float64; op=plus,
                l=copy(f; break_sharing=Val(true)), r=scaled_g)
        end
        proposed = _node_replace_anchor(root, target, replacement)
        return _node_finalize_proposal(proposed, parent, config)
    end
    return (parent, false)
end

"""Return the nonconstant operand of a multiplication by a scalar leaf."""
function _node_scaled_term_payload(node::DynamicExpressions.Node{Float64},
        multiply::UInt8)
    node.degree == 2 && node.op == multiply || return nothing
    if node.l.degree == 0 && node.l.constant
        return node.r
    elseif node.r.degree == 0 && node.r.constant
        return node.l
    end
    return nothing
end

function _node_collect_prune_sites!(sites::Vector{Tuple{Tuple,UInt8}}, node,
        multiply::UInt8, plus::UInt8, path::Tuple=())
    if node.degree == 2 && node.op == plus
        _node_scaled_term_payload(node.l, multiply) !== nothing &&
            push!(sites, (path, 0x02))
        _node_scaled_term_payload(node.r, multiply) !== nothing &&
            push!(sites, (path, 0x01))
    end
    node.degree >= 1 && _node_collect_prune_sites!(
        sites, node.l, multiply, plus, (path..., 1))
    node.degree == 2 && _node_collect_prune_sites!(
        sites, node.r, multiply, plus, (path..., 2))
    return sites
end

function _node_composition_path_node(node, path::Tuple)
    current = node
    for side in path
        current = side == 1 ? current.l : side == 2 ? current.r :
            throw(ArgumentError("invalid composition proposal path"))
    end
    return current
end

function _node_composition_replace_path(node, path::Tuple, replacement)
    isempty(path) && return replacement
    side = first(path)
    rest = Base.tail(path)
    if side == 1
        node.l = _node_composition_replace_path(node.l, rest, replacement)
    elseif side == 2
        node.r = _node_composition_replace_path(node.r, rest, replacement)
    else
        throw(ArgumentError("invalid composition proposal path"))
    end
    return node
end

"""Propose deleting one explicitly coefficient-scaled additive term.

For `f + c*g`, the proposal returns `f` (and symmetrically `g` when the
scaled term is on the left). No coefficient threshold or algebraic identity is
assumed. The caller sends the resulting complete candidate through ordinary
scoring, so domain failures and objective changes determine whether it survives.
"""
function _node_prune_proposal(rng::AbstractRNG,
        parent::_NodeScalarCandidate, donor::_NodeScalarCandidate,
        feature_count::Int, config::ScalarSearchConfig)
    :+ in config.binary_operators && :* in config.binary_operators ||
        return (parent, false)

    plus = _dynamic_binary_index(:+)
    multiply = _dynamic_binary_index(:*)
    choices = _node_collect_prune_sites!(Tuple{Tuple,UInt8}[], parent.node,
        multiply, plus)
    isempty(choices) && return (parent, false)
    path, keep_side = rand(rng, choices)
    root = copy(parent.node; break_sharing=Val(true))
    sum_node = _node_composition_path_node(root, path)
    keep = keep_side == 0x01 ? sum_node.l : sum_node.r
    root = _node_composition_replace_path(root, path,
        copy(keep; break_sharing=Val(true)))
    return _node_finalize_proposal(root, parent, config)
end

@testsnippet NodeCompositionCleanupImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
end

@testitem "coefficient composition builds an owned bounded proposal" setup=[NodeCompositionCleanupImports] begin
    config = ScalarSearchConfig(population_size=2, max_evaluations=2,
        max_complexity=12, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(), binary_operators=(:+, :*),
        constant_refinement_iterations=8,
        max_constant_refinement_evaluations=128,
        evolution_strategy=:regularized)
    parent = ScalarSearch._node_scalar_candidate(scalar_binary(:+, scalar_input(1), scalar_input(2)))
    donor = ScalarSearch._node_scalar_candidate(scalar_binary(:*, scalar_input(1), scalar_input(2)))
    parent_before = copy(parent.node; break_sharing=Val(true))
    donor_before = copy(donor.node; break_sharing=Val(true))

    proposal, changed = ScalarSearch._node_coefficient_proposal(
        MersenneTwister(1234), parent, donor, 2, config)

    @test changed
    @test proposal isa ScalarSearch._NodeScalarCandidate
    @test 1 < proposal.complexity <= config.max_complexity
    @test proposal.node !== parent.node
    @test parent.node == parent_before
    @test donor.node == donor_before
    parameters = [ScalarSearch._node_at_path(proposal.node, path).val
        for path in ScalarSearch._node_constant_paths(proposal.node)]
    @test length(parameters) in (1, 2)
    @test all(value -> !iszero(value), parameters)

    seen_parameter_counts = Int[]
    score_node = node -> begin
        values = [ScalarSearch._node_at_path(node, path).val
            for path in ScalarSearch._node_constant_paths(node)]
        push!(seen_parameter_counts, length(values))
        return (sum(abs2(value - 3.0) for value in values), false)
    end
    refined, loss, evaluations, _ = ScalarSearch._node_refine_constants(
        proposal, 1.0e6, score_node, config, 128; rng=MersenneTwister(2000))
    @test evaluations > 0
    @test loss < 1.0e6
    @test all(==(length(parameters)), seen_parameter_counts)
    @test refined.node !== proposal.node

    ids = IdDict{Any,Bool}()
    for candidate in (parent, donor), anchor in ScalarSearch._node_anchors(candidate.node)
        ids[anchor.node] = true
    end
    @test all(anchor -> !haskey(ids, anchor.node),
        ScalarSearch._node_anchors(proposal.node))
end

@testitem "coefficient composition respects operators and size limits" setup=[NodeCompositionCleanupImports] begin
    parent = ScalarSearch._node_scalar_candidate(scalar_input(1))
    donor = ScalarSearch._node_scalar_candidate(scalar_input(2))
    no_multiply = ScalarSearch.ScalarSearchConfig(population_size=2,
        max_evaluations=2, max_complexity=12, initial_depth=0,
        tournament_size=1, elite_count=1, unary_operators=(),
        binary_operators=(:+,), evolution_strategy=:regularized)
    too_small = ScalarSearch.ScalarSearchConfig(population_size=2,
        max_evaluations=2, max_complexity=3, initial_depth=0,
        tournament_size=1, elite_count=1, unary_operators=(),
        binary_operators=(:+, :*), evolution_strategy=:regularized)

    unchanged, changed = ScalarSearch._node_coefficient_proposal(
        MersenneTwister(1), parent, donor, 2, no_multiply)
    @test unchanged === parent
    @test !changed
    unchanged, changed = ScalarSearch._node_coefficient_proposal(
        MersenneTwister(1), parent, donor, 2, too_small)
    @test unchanged === parent
    @test !changed
end

@testitem "coefficient pruning deletes a scaled additive term as a rescored proposal" setup=[NodeCompositionCleanupImports] begin
    config = ScalarSearch.ScalarSearchConfig(population_size=2,
        max_evaluations=2, max_complexity=16, initial_depth=0,
        tournament_size=1, elite_count=1, unary_operators=(),
        binary_operators=(:+, :*), evolution_strategy=:regularized)
    expression = scalar_binary(:+, scalar_input(1),
        scalar_binary(:*, scalar_constant(2.0), scalar_input(2)))
    parent = ScalarSearch._node_scalar_candidate(expression)
    before = copy(parent.node; break_sharing=Val(true))

    proposal, changed = ScalarSearch._node_prune_proposal(
        MersenneTwister(1111), parent, parent, 2, config)

    @test changed
    @test proposal.complexity < parent.complexity
    @test proposal.node == ScalarSearch.scalar_to_dynamic(scalar_input(1)) ||
        proposal.node == ScalarSearch.scalar_to_dynamic(scalar_input(2))
    @test parent.node == before
    @test proposal.node !== parent.node

    no_multiply = ScalarSearch.ScalarSearchConfig(population_size=2,
        max_evaluations=2, max_complexity=16, initial_depth=0,
        tournament_size=1, elite_count=1, unary_operators=(),
        binary_operators=(:+,), evolution_strategy=:regularized)
    unchanged, changed = ScalarSearch._node_prune_proposal(
        MersenneTwister(1111), parent, parent, 2, no_multiply)
    @test unchanged === parent
    @test !changed
end
