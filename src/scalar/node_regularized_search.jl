"""Node-native steady-state search behind the existing scalar-search interface.

This is the node-native loop for the internal representation change. It owns
mutable DynamicExpressions trees and one evaluation workspace, while returning
the established `ScalarSearchResult` with rebuildable `ScalarExpr` values. The
public objective contract remains `ScalarExpr`-compatible; objectives that also
accept `(ScalarExpr, node)` can reuse the owned node without converting it back.
"""

struct _NodeSearchCandidate
    value::_NodeScalarCandidate
    loss::Float64
    serial::Int
    evolved::Bool
end

mutable struct _NodeRegularizedMember
    candidate::_NodeSearchCandidate
    born::Int
end

_node_candidate_key(candidate::_NodeSearchCandidate) =
    (candidate.loss, candidate.value.complexity, candidate.serial)

function _node_pareto_candidates(archive::Dict{Int,_NodeSearchCandidate})
    result = _NodeSearchCandidate[]
    best_loss = Inf
    for complexity in sort!(collect(keys(archive)))
        candidate = archive[complexity]
        candidate.loss < best_loss || continue
        push!(result, candidate)
        best_loss = candidate.loss
    end
    return result
end

function _node_oldest_index(population)
    best = firstindex(population)
    best_key = (population[best].born, population[best].candidate.serial)
    for index in (firstindex(population) + 1):lastindex(population)
        key = (population[index].born, population[index].candidate.serial)
        key < best_key || continue
        best, best_key = index, key
    end
    return best
end

function _node_observe_population!(frequency::_RegularizedFrequency, population)
    for member in population
        complexity = member.candidate.value.complexity
        0 < complexity <= length(frequency.frequencies) || continue
        frequency.frequencies[complexity] += 1.0
    end
    _regularized_move_window!(frequency)
    return _regularized_normalize!(frequency)
end

"""Choose one candidate by rank tournament without replacement.

Candidate loss/complexity, the configured tournament size, and the shared
frequency snapshot anchor ranking. Empty candidate sets fail at sampling;
malformed selection probabilities are rejected by the state initializer.
"""
function _node_tournament(rng, candidates, config, frequency;
        loss_normalization, selection_p, use_frequency_in_tournament,
        adaptive_parsimony_scaling)
    n = min(config.tournament_size, length(candidates))
    available = collect(eachindex(candidates))
    selected = Vector{Int}(undef, n)
    for index in 1:n
        slot = rand(rng, eachindex(available))
        selected[index] = available[slot]
        deleteat!(available, slot)
    end
    costs = Vector{Float64}(undef, n)
    for index in 1:n
        candidate = candidates[selected[index]]
        complexity = candidate.value.complexity
        cost = _regularized_cost(candidate.loss, complexity,
            loss_normalization, config.parsimony_coefficient)
        if use_frequency_in_tournament
            cost *= exp(adaptive_parsimony_scaling *
                _regularized_frequency(frequency, complexity))
        end
        costs[index] = cost
    end
    order = sortperm(costs)
    draw = rand(rng)
    cumulative = 0.0
    chosen = n
    normalization = sum(selection_p * (1 - selection_p)^(rank - 1) for rank in 1:n)
    for rank in 1:n
        cumulative += selection_p * (1 - selection_p)^(rank - 1) / normalization
        if draw < cumulative
            chosen = rank
            break
        end
    end
    return candidates[selected[order[chosen]]]
end

function _node_to_scalar(candidate::_NodeSearchCandidate)
    expr = _node_scalar_materialize(candidate.value)
    expr isa Err && error("node-native candidate failed strict ScalarExpr rebuilding")
    return _ScalarCandidate(expr, candidate.loss, candidate.serial, candidate.evolved)
end

function _node_regularized_report(population, archive, best, evaluations,
        invalid_evaluations, constant_refinement_evaluations,
        constant_refinement_invalid_evaluations,
        generations, termination, improvements,
        mutation_attempts, crossover_attempts, composition_attempts,
        changed_variations, mutation_kind_counts, accepted_offspring,
        rejected_offspring, skipped_variations, oldest_replacements,
        hof_migrations, annealing, use_frequency, frequency_window)
    scalar_population = [_RegularizedMember(_node_to_scalar(member.candidate), member.born)
        for member in population]
    scalar_archive = Dict(complexity => _node_to_scalar(candidate)
        for (complexity, candidate) in archive)
    best_expr = _node_to_scalar(best).expr
    return _regularized_report(scalar_population, scalar_archive, best_expr,
        best.loss, evaluations, invalid_evaluations,
        constant_refinement_evaluations,
        constant_refinement_invalid_evaluations, generations,
        termination, improvements, mutation_attempts, crossover_attempts,
        composition_attempts, changed_variations, mutation_kind_counts,
        accepted_offspring, rejected_offspring, skipped_variations,
        oldest_replacements, hof_migrations,
        annealing, use_frequency, frequency_window)
end

"""Propose one named variation under the conventional retry boundary.

The owned parents, feature count, configured operator world, and explicit
mutation kind fully anchor the proposal. Structural kinds retry at most ten
times; a no-op, simplification, optimisation marker, or composition is tried
once. Exhaustion returns the unchanged first parent with `false`; unexpected
operator kinds fail in `_node_scalar_propose`.
"""
function _node_regularized_named_child(rng::AbstractRNG,
        first_parent::_NodeScalarCandidate, second_parent::_NodeScalarCandidate,
        feature_count::Int, config::ScalarSearchConfig, kind::Symbol;
        temperature::Real=1.0)
    tries = kind in (:do_nothing, :simplify, :optimize, :compose) ? 1 : 10
    for _ in 1:tries
        child, changed = _node_scalar_propose(
            rng, first_parent, second_parent, feature_count, config, kind;
            temperature)
        changed && return (child, true)
    end
    return (first_parent, false)
end

@testsnippet NodeRegularizedSearchImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
end

@testitem "node regularized search spends its budget and returns ScalarExpr" setup=[NodeRegularizedSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=12,
        max_complexity=12,
        initial_depth=2,
        tournament_size=2,
        elite_count=2,
        unary_operators=(:sin,),
        binary_operators=(:+, :*),
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        constant_refinement_probability=0.0,
        evolution_strategy=:regularized,
    )
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = @. X[1, :] * X[1, :] + X[1, :]

    result = ScalarSearch._search_scalar_regularized_node(
        X, y;
        config,
        rng=MersenneTwister(1234),
        initial_population=[scalar_input(1)],
        stop_loss=nothing,
        annealing=false,
        use_frequency=false,
        ncycles_per_iteration=2,
    )

    @test result.best isa ScalarExpr
    @test result.evaluations == config.max_evaluations
    @test result.termination == :evaluation_cap
    @test isfinite(result.best_loss)
    @test all(isfinite, evaluate_scalar_reference(result.best, X))
    @test result.variation.final_population_size == config.population_size
end
