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

function _regularized_frequency_ratio(frequency::_RegularizedFrequency,
        old_complexity::Int, new_complexity::Int)
    return _regularized_frequency(frequency, old_complexity),
        _regularized_frequency(frequency, new_complexity)
end

"""Return the baseline-normalized cost used by node search selection and annealing."""
function _regularized_cost(loss::Real, complexity::Int, normalization::Real,
        parsimony::Real)
    normalization > 0 || throw(ArgumentError("loss normalization must be positive"))
    return Float64(loss) / Float64(normalization) +
        complexity * Float64(parsimony)
end

function _regularized_cycle_temperature(attempt::Int, population_size::Int,
        tournament_size::Int, ncycles_per_iteration::Int)
    attempt >= 1 || throw(ArgumentError("attempt must be positive"))
    ncycles_per_iteration >= 1 ||
        throw(ArgumentError("ncycles_per_iteration must be positive"))
    sweeps = cld(population_size, tournament_size)
    period = sweeps * ncycles_per_iteration
    temperature_index = div(mod(attempt - 1, period), sweeps)
    temperature = ncycles_per_iteration == 1 ? 1.0 :
        1.0 - temperature_index / (ncycles_per_iteration - 1)
    return (max(eps(Float64), temperature), period)
end

function _regularized_poisson(rng::AbstractRNG, mean::Real)
    mean >= 0 || throw(ArgumentError("Poisson mean must be nonnegative"))
    iszero(mean) && return 0
    threshold = exp(-Float64(mean))
    product = 1.0
    count = 0
    while product > threshold
        count += 1
        product *= rand(rng)
    end
    return count - 1
end

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
    best_expr = _simplify_scalar(_node_to_scalar(best).expr)
    population_candidates = [_node_to_scalar(member.candidate) for member in population]
    report_candidates = copy(population_candidates)
    push!(report_candidates, _ScalarCandidate(best_expr, best.loss, 0, true))
    sort!(report_candidates; by=_candidate_key)
    leaders = NamedTuple[]
    seen_structures = Set{_ScalarSignature}()
    for candidate in report_candidates
        signature = _structure_signature(candidate.expr)
        signature in seen_structures && continue
        push!(seen_structures, signature)
        push!(leaders, (expr=candidate.expr, loss=candidate.loss,
            complexity=scalar_complexity(candidate.expr)))
        length(leaders) >= 8 && break
    end

    complexity_archive = [
        (expr=_node_to_scalar(archive[complexity]).expr,
            loss=archive[complexity].loss, complexity)
        for complexity in sort!(collect(keys(archive)))
    ]
    complexity_counts = Dict{Int,Int}()
    structures = Set{_ScalarSignature}()
    for candidate in population_candidates
        complexity = scalar_complexity(candidate.expr)
        complexity_counts[complexity] = get(complexity_counts, complexity, 0) + 1
        push!(structures, _structure_signature(candidate.expr))
    end
    final_complexity_histogram = [
        (complexity=complexity, count=complexity_counts[complexity])
        for complexity in sort!(collect(keys(complexity_counts)))
    ]
    variation_attempts = mutation_attempts + crossover_attempts + composition_attempts
    mutation_kinds = (; (kind => get(mutation_kind_counts, kind, 0)
        for kind in (:constant, :operator, :swap, :rotate, :add, :insert, :delete,
            :simplify, :restart, :do_nothing, :optimize,
            :graft, :scale, :shift, :subtree))...)
    variation = (
        attempts=variation_attempts,
        changed=changed_variations,
        unchanged=variation_attempts - changed_variations,
        mutation_attempts,
        mutation_kinds,
        crossover_attempts,
        composition_attempts,
        accepted_offspring,
        rejected_offspring,
        skipped_variations,
        oldest_replacements,
        hof_migrations,
        annealing,
        use_frequency,
        frequency_window,
        final_population_size=length(population),
        final_unique_structures=length(structures),
        final_complexity_histogram,
    )
    return ScalarSearchResult(best_expr, best.loss, evaluations, invalid_evaluations,
        constant_refinement_evaluations, constant_refinement_invalid_evaluations,
        generations, termination, improvements, leaders, complexity_archive, variation)
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

@testitem "node regularized policy helpers preserve cost, cycle, and migration behavior" setup=[NodeRegularizedSearchImports] begin
    @test ScalarSearch._regularized_cost(20.0, 3, 100.0, 0.01) ≈ 0.23
    @test ScalarSearch._regularized_cycle_temperature(1, 27, 15, 4) == (1.0, 8)
    @test ScalarSearch._regularized_cycle_temperature(8, 27, 15, 4)[1] ==
        eps(Float64)
    @test ScalarSearch._regularized_cycle_temperature(9, 27, 15, 4) == (1.0, 8)
    @test ScalarSearch._regularized_acceptance_probability(1.0, 2.0) == 1.0
    @test ScalarSearch._regularized_acceptance_probability(1.0, 2.0;
        annealing=true, temperature=1.0, alpha=1.0) ≈ exp(-1.0)
    @test ScalarSearch._regularized_acceptance_probability(2.0, 1.0;
        annealing=true, temperature=1.0, alpha=1.0) == 1.0
    @test ScalarSearch._regularized_poisson(MersenneTwister(8000), 0.0) == 0
    @test ScalarSearch._regularized_poisson(MersenneTwister(1111), 3.0) ==
        ScalarSearch._regularized_poisson(MersenneTwister(1111), 3.0)
end

@testitem "node regularized report preserves archive, leaders, and counters" setup=[NodeRegularizedSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4, max_evaluations=4, max_complexity=10,
        initial_depth=1, tournament_size=2, elite_count=1,
        unary_operators=(:sin,), binary_operators=(:+, :*),
        constant_refinement_iterations=0,
        constant_refinement_probability=0.0,
        evolution_strategy=:regularized,
    )
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = vec(X)
    initial = ScalarExpr[
        scalar_input(1),
        scalar_unary(:sin, scalar_input(1)),
        scalar_binary(:+, scalar_input(1), scalar_constant(1.0)),
        scalar_binary(:*, scalar_input(1), scalar_input(1)),
    ]
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, initial_population=initial,
        rng=MersenneTwister(2000), stop_loss=nothing,
        annealing=false, use_frequency=false)
    result = ScalarSearch._node_regularized_result(state)

    expected_archive = [
        (ScalarSearch._structure_signature(
            ScalarSearch._node_to_scalar(state.archive[complexity]).expr),
            state.archive[complexity].loss, complexity)
        for complexity in sort!(collect(keys(state.archive)))
    ]
    actual_archive = [
        (ScalarSearch._structure_signature(entry.expr), entry.loss, entry.complexity)
        for entry in result.complexity_archive
    ]
    @test actual_archive == expected_archive

    scalar_population = [ScalarSearch._node_to_scalar(member.candidate)
        for member in state.population]
    expected_best = ScalarSearch._simplify_scalar(
        ScalarSearch._node_to_scalar(state.best).expr)
    report_candidates = copy(scalar_population)
    push!(report_candidates,
        ScalarSearch._ScalarCandidate(expected_best, state.best.loss, 0, true))
    sort!(report_candidates; by=ScalarSearch._candidate_key)
    expected_leaders = NamedTuple[]
    seen = Set{typeof(ScalarSearch._structure_signature(expected_best))}()
    for candidate in report_candidates
        signature = ScalarSearch._structure_signature(candidate.expr)
        signature in seen && continue
        push!(seen, signature)
        push!(expected_leaders, (signature=ScalarSearch._structure_signature(candidate.expr),
            loss=candidate.loss,
            complexity=ScalarSearch.scalar_complexity(candidate.expr)))
        length(expected_leaders) >= 8 && break
    end
    actual_leaders = [(signature=ScalarSearch._structure_signature(leader.expr),
        loss=leader.loss, complexity=leader.complexity) for leader in result.leaders]
    @test actual_leaders == expected_leaders

    @test result.evaluations == state.evaluations == config.max_evaluations
    @test result.invalid_evaluations == state.invalid_evaluations
    @test result.constant_refinement_evaluations == state.constant_refinement_evaluations
    @test result.constant_refinement_invalid_evaluations ==
        state.constant_refinement_invalid_evaluations
    @test result.generations == 0
    @test result.termination == :evaluation_cap
    @test result.variation.attempts == state.mutation_attempts +
        state.crossover_attempts + state.composition_attempts
    @test result.variation.accepted_offspring == state.accepted_offspring
    @test result.variation.rejected_offspring == state.rejected_offspring
    @test result.variation.skipped_variations == state.skipped_variations
    @test result.variation.oldest_replacements == state.oldest_replacements
    @test result.variation.hof_migrations == state.hof_migrations
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
