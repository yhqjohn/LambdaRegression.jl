"""Shared steady-state regularized search policies and scalar-domain adapters.

The scalar caller retains its established `ScalarSearchResult` and rebuildable
`ScalarExpr` values. Internal value hooks let a domain reuse the same persistent
population, archive, and variation lifecycle with its own complete candidate.
"""

struct _NodeSearchCandidate{V}
    value::V
    loss::Float64
    serial::Int
    evolved::Bool
end

mutable struct _NodeRegularizedMember{V}
    candidate::_NodeSearchCandidate{V}
    born::Int
end

mutable struct _NodeTournamentWorkspace
    available::Vector{Int}
    selected::Vector{Int}
    costs::Vector{Float64}
    order::Vector{Int}
end

"""Allocate reusable sampling and ranking buffers for one regularized state.

Buffers are sized to the state's population and tournament. The workspace owns
only indices and costs, never candidate references, and must not be shared
between states. Invalid sizes are left to the vector constructors to reject.
"""
_NodeTournamentWorkspace(population_size::Int, tournament_size::Int) =
    _NodeTournamentWorkspace(collect(1:population_size),
        Vector{Int}(undef, tournament_size),
        Vector{Float64}(undef, tournament_size),
        Vector{Int}(undef, tournament_size))

_node_tournament_candidate(candidate::_NodeSearchCandidate) = candidate
_node_tournament_candidate(member::_NodeRegularizedMember) = member.candidate

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

function _node_pareto_candidates(archive::AbstractDict{Int,<:_NodeSearchCandidate})
    result = collect(values(archive))
    empty!(result)
    best_loss = Inf
    for complexity in sort!(collect(keys(archive)))
        candidate = archive[complexity]
        candidate.loss < best_loss || continue
        push!(result, candidate)
        best_loss = candidate.loss
    end
    return result
end

_regularized_copy_value(::Nothing, value::_NodeScalarCandidate) =
    _node_scalar_candidate(copy(value.node; break_sharing=Val(true)))

_regularized_has_constants(::Nothing, value::_NodeScalarCandidate) =
    !isempty(_node_constant_paths(value.node))

function _regularized_refine_value(::Nothing, value::_NodeScalarCandidate,
        loss::Real, score, config::ScalarSearchConfig, remaining::Int;
        rng::AbstractRNG, restarts::Int=2)
    score_node = node -> score(_NodeScalarCandidate(node, value.complexity))
    return _node_refine_constants(value, loss, score_node, config, remaining;
        rng, restarts)
end

function _regularized_refine_value(::Nothing, value::_NodeScalarCandidate,
        loss::Real, score::_NodeMSEScore, config::ScalarSearchConfig, remaining::Int;
        rng::AbstractRNG, restarts::Int=2)
    if score.jacobian === nothing
        score_node = node -> score(_NodeScalarCandidate(node, value.complexity))
        return _node_refine_constants(value, loss, score_node, config, remaining;
            rng, restarts)
    end
    return _node_refine_mse_jacobian(value, loss, score, config, remaining;
        rng, restarts)
end

function _regularized_simplify_value(::Nothing, value::_NodeScalarCandidate, rng)
    expr = _node_scalar_materialize(value)
    expr isa Err && return nothing
    simplified = _simplify_scalar(expr)
    _same_scalar_expression(simplified, expr) && return nothing
    return _node_scalar_candidate(simplified)
end

_regularized_sample_mutation(::Nothing, rng::AbstractRNG,
        value::_NodeScalarCandidate, config::ScalarSearchConfig) =
    _node_scalar_sample_mutation(rng, value, config)

_regularized_crossover_pair(::Nothing, rng::AbstractRNG,
        first::_NodeScalarCandidate, second::_NodeScalarCandidate,
        config::ScalarSearchConfig) =
    _node_scalar_crossover_pair(rng, first, second, config)

function _regularized_named_child(::Nothing, rng::AbstractRNG,
        first::_NodeScalarCandidate, second::_NodeScalarCandidate,
        feature_count::Int, config::ScalarSearchConfig, kind::Symbol;
        temperature::Real=1.0)
    return _node_regularized_named_child(rng, first, second, feature_count,
        config, kind; temperature)
end

_regularized_replacement_index(domain, state, parent, child) =
    _node_oldest_index(state.population)

_regularized_observe(domain, state, event, parent, child, admitted;
    new_offspring::Bool=false) = nothing

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

"""Sample tournament slots and its rank draw without ranking candidates.

Population size and `config.tournament_size` anchor sampling without replacement.
The state-owned workspace keeps the sampled indices until the next draw. The
returned random draw can be ranked later against the unchanged population and
frequency snapshot, or discarded while preserving RNG consumption. The caller
supplies a nonempty candidate set; no candidate references are retained.
"""
function _node_draw_tournament!(rng, candidates,
        workspace::_NodeTournamentWorkspace, config)
    n = min(config.tournament_size, length(candidates))
    available = workspace.available
    resize!(available, length(candidates))
    for (position, index) in enumerate(eachindex(candidates))
        available[position] = index
    end
    selected = workspace.selected
    resize!(selected, n)
    for index in 1:n
        slot = rand(rng, eachindex(available))
        selected[index] = available[slot]
        deleteat!(available, slot)
    end
    return rand(rng)
end

"""Resolve previously sampled tournament slots against the current ranking.

The caller must preserve the population, frequency snapshot and workspace's
selected indices since `_node_draw_tournament!`. Candidate loss/complexity and
the supplied pressure parameters anchor costs. This step consumes no random
numbers; selection probability is validated by the state initializer.
"""
function _node_rank_tournament!(draw, candidates,
        workspace::_NodeTournamentWorkspace, config, frequency;
        loss_normalization, selection_p, use_frequency_in_tournament,
        adaptive_parsimony_scaling)
    selected = workspace.selected
    n = length(selected)
    costs = workspace.costs
    resize!(costs, n)
    for index in 1:n
        candidate = _node_tournament_candidate(candidates[selected[index]])
        complexity = candidate.value.complexity
        cost = _regularized_cost(candidate.loss, complexity,
            loss_normalization, config.parsimony_coefficient)
        if use_frequency_in_tournament
            pressure = adaptive_parsimony_scaling *
                _regularized_frequency(frequency, complexity)
            cost *= exp(cost < 0 ? -pressure : pressure)
        end
        costs[index] = cost
    end
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
    if chosen == 1
        # Match stable sortperm ordering, including ties, signed zero and NaN.
        # Ordinary `<` or an unordered reduction can change these boundaries.
        best = 1
        for index in 2:n
            isless(costs[index], costs[best]) && (best = index)
        end
        return _node_tournament_candidate(candidates[selected[best]])
    end
    order = workspace.order
    resize!(order, n)
    sortperm!(order, costs)
    return _node_tournament_candidate(candidates[selected[order[chosen]]])
end

"""Choose one candidate by rank tournament without replacement.

Candidate loss/complexity, the configured tournament size, and the shared
frequency snapshot anchor ranking. Empty candidate sets fail at sampling;
malformed selection probabilities are rejected by the state initializer.
"""
function _node_tournament(rng, candidates, workspace::_NodeTournamentWorkspace,
        config, frequency; kwargs...)
    draw = _node_draw_tournament!(rng, candidates, workspace, config)
    return _node_rank_tournament!(draw, candidates, workspace, config,
        frequency; kwargs...)
end

# Ordinary scalar mutations below never read their donor. Keep all composition,
# extension and unknown kinds conservative; joint-domain mutation has separate
# donor/provenance semantics and does not use this shortcut.
_node_mutation_uses_donor(kind::Symbol) = !(kind in
    (:constant, :operator, :swap, :rotate, :add, :insert, :delete,
     :simplify, :restart, :do_nothing, :optimize, :rotation_copy))

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
        hof_migrations, annealing, use_frequency, frequency_window,
        cycle_refinement_calls, offspring_refinement_calls,
        total_evaluation_budget, remaining_total_evaluations)
    best_expr = _node_to_scalar(best).expr
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
            :rotation_copy, :simplify, :restart, :do_nothing, :optimize,
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
        constant_refinement_calls=
            cycle_refinement_calls + offspring_refinement_calls,
        cycle_refinement_calls,
        offspring_refinement_calls,
        total_evaluation_budget,
        remaining_total_evaluations,
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
    tries = kind in (:do_nothing, :simplify, :optimize, :compose,
        :rotation_copy) ? 1 : 10
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

@testitem "frequency pressure penalizes common negative-loss candidates" setup=[NodeRegularizedSearchImports] begin
    config = ScalarSearchConfig(population_size=2, max_evaluations=2,
        max_complexity=2, initial_depth=0, tournament_size=2, elite_count=1,
        unary_operators=(:sin,), binary_operators=(),
        parsimony_coefficient=0.1, constant_refinement_iterations=0,
        evolution_strategy=:regularized)
    simple = ScalarSearch._node_scalar_candidate(scalar_constant(0.0))
    larger = ScalarSearch._node_scalar_candidate(
        scalar_unary(:sin, scalar_constant(0.0)))
    frequency = ScalarSearch._RegularizedFrequency(10, 2)
    frequency.normalized_frequencies .= [0.9, 0.1]
    workspace = ScalarSearch._NodeTournamentWorkspace(2, config.tournament_size)
    choose = candidates -> ScalarSearch._node_tournament(
        MersenneTwister(1111), candidates, workspace, config, frequency;

        loss_normalization=1.0, selection_p=1.0,
        use_frequency_in_tournament=true, adaptive_parsimony_scaling=1.0)

    negative = [
        ScalarSearch._NodeSearchCandidate(simple, -1.0, 1, true),
        ScalarSearch._NodeSearchCandidate(larger, -1.0, 2, true),
    ]
    @test choose(negative).value.complexity == 2

    positive = [
        ScalarSearch._NodeSearchCandidate(simple, 1.0, 1, true),
        ScalarSearch._NodeSearchCandidate(larger, 1.0, 2, true),
    ]
    @test choose(positive).value.complexity == 2
end

@testitem "node tournament workspace preserves selection and RNG stream" setup=[NodeRegularizedSearchImports] begin
    candidate_of(value) = value isa ScalarSearch._NodeRegularizedMember ?
        value.candidate : value
    function reference_tournament(rng, candidates, config, frequency;
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
            candidate = candidate_of(candidates[selected[index]])
            complexity = candidate.value.complexity
            cost = ScalarSearch._regularized_cost(candidate.loss, complexity,
                loss_normalization, config.parsimony_coefficient)
            if use_frequency_in_tournament
                pressure = adaptive_parsimony_scaling *
                    ScalarSearch._regularized_frequency(frequency, complexity)
                cost *= exp(cost < 0 ? -pressure : pressure)
            end
            costs[index] = cost
        end
        order = sortperm(costs)
        draw = rand(rng)
        cumulative = 0.0
        chosen = n
        normalization = sum(selection_p * (1 - selection_p)^(rank - 1)
            for rank in 1:n)
        for rank in 1:n
            cumulative += selection_p * (1 - selection_p)^(rank - 1) /
                normalization
            if draw < cumulative
                chosen = rank
                break
            end
        end
        return candidate_of(candidates[selected[order[chosen]]])
    end

    values = [
        ScalarSearch._node_scalar_candidate(scalar_constant(0.0)),
        ScalarSearch._node_scalar_candidate(
            scalar_unary(:sin, scalar_constant(0.0))),
        ScalarSearch._node_scalar_candidate(scalar_unary(:sin,
            scalar_unary(:sin, scalar_constant(0.0)))),
    ]
    frequency = ScalarSearch._RegularizedFrequency(20, 3)
    frequency.normalized_frequencies .= [0.8, 0.15, 0.05]

    for population_size in (2, 3, 7)
        candidates = [
            ScalarSearch._NodeSearchCandidate(values[mod1(index, 3)],
                index == 4 ? Inf : (isodd(index) ? -1.0 : 0.5), index, true)
            for index in 1:population_size
        ]
        tied = [ScalarSearch._NodeSearchCandidate(values[1], 0.0,
            100 + index, true) for index in 1:population_size]
        members = [ScalarSearch._NodeRegularizedMember(candidate, index)
            for (index, candidate) in enumerate(candidates)]
        for population in (candidates, tied, members)
            n = length(population)
            for tournament_size in unique((1, min(2, n), n)),
                    use_frequency in (false, true), selection_p in (1.0, 0.37)
                config = ScalarSearchConfig(population_size=n,
                    max_evaluations=max(n, 1), max_complexity=4,
                    initial_depth=0, tournament_size=tournament_size,
                    elite_count=1, unary_operators=(:sin,), binary_operators=(),
                    parsimony_coefficient=0.2,
                    constant_refinement_iterations=0,
                    evolution_strategy=:regularized)
                actual_rng = MersenneTwister(1234)
                reference_rng = MersenneTwister(1234)
                workspace = ScalarSearch._NodeTournamentWorkspace(n,
                    tournament_size)
                for _ in 1:4
                    kwargs = (; loss_normalization=1.0,
                        selection_p, use_frequency_in_tournament=use_frequency,
                        adaptive_parsimony_scaling=0.7)
                    actual = ScalarSearch._node_tournament(actual_rng,
                        population, workspace, config, frequency; kwargs...)
                    expected = reference_tournament(reference_rng, population,
                        config, frequency; kwargs...)
                    @test actual.serial == expected.serial
                    @test rand(actual_rng, UInt, 4) == rand(reference_rng, UInt, 4)
                end
                # A discarded donor must still consume the historical draws;
                # resolving a retained donor later must not consume any more.
                kwargs = (; loss_normalization=1.0, selection_p,
                    use_frequency_in_tournament=use_frequency,
                    adaptive_parsimony_scaling=0.7)
                ScalarSearch._node_draw_tournament!(actual_rng, population,
                    workspace, config)
                reference_tournament(reference_rng, population, config,
                    frequency; kwargs...)
                @test rand(actual_rng, UInt, 4) == rand(reference_rng, UInt, 4)
                draw = ScalarSearch._node_draw_tournament!(actual_rng,
                    population, workspace, config)
                expected = reference_tournament(reference_rng, population,
                    config, frequency; kwargs...)
                @test rand(actual_rng, UInt, 4) == rand(reference_rng, UInt, 4)
                actual = ScalarSearch._node_rank_tournament!(draw, population,
                    workspace, config, frequency; kwargs...)
                @test actual.serial == expected.serial
                @test rand(actual_rng, UInt, 4) == rand(reference_rng, UInt, 4)
            end
        end
    end
end

@testitem "node tournament minimum preserves stable float ordering" setup=[NodeRegularizedSearchImports] begin
    value = ScalarSearch._node_scalar_candidate(scalar_constant(0.0))
    losses = [NaN, Inf, -Inf, -0.0, 0.0, 1.0, 1.0, NaN]
    candidates = [ScalarSearch._NodeSearchCandidate(value, loss, i, true)
        for (i, loss) in enumerate(losses)]
    config = ScalarSearchConfig(population_size=8, max_evaluations=8,
        tournament_size=8, elite_count=1, parsimony_coefficient=-0.0,
        constant_refinement_iterations=0, evolution_strategy=:regularized)
    frequency = ScalarSearch._RegularizedFrequency(20, config.max_complexity)
    workspace = ScalarSearch._NodeTournamentWorkspace(8, 8)
    # Explicit rank draws exercise the shortcut and every fallback rank, even
    # when production selection pressure makes non-first ranks very rare.
    for selected in (collect(1:8), collect(8:-1:1), [1,8,2], [6,7], [5,4])
        workspace.selected = copy(selected)
        n = length(selected)
        for rank in 1:n
            expected_costs = [ScalarSearch._regularized_cost(losses[i], 1,
                1.0, -0.0) for i in selected]
            p = 0.2
            weights = [p * (1-p)^(k-1) for k in 1:n]
            draw = (sum(weights[1:rank-1]) + weights[rank]/2) / sum(weights)
            actual = ScalarSearch._node_rank_tournament!(draw, candidates,
                workspace, config, frequency; loss_normalization=1.0,
                selection_p=p, use_frequency_in_tournament=false,
                adaptive_parsimony_scaling=0.0)
            @test actual.serial == selected[sortperm(expected_costs)[rank]]
        end
    end
end

@testitem "scalar reports retain scored trees under floating-point cancellation" setup=[NodeRegularizedSearchImports] begin
    input = scalar_input(1)
    original = scalar_binary(:-, input,
        scalar_binary(:+, scalar_constant(1.0), input))
    for strategy in (:regularized, :generational), x in (1.0e16, 2.0)
        X = reshape([x], 1, 1)
        y = [0.0]
        config = ScalarSearchConfig(population_size=2, max_evaluations=2,
            max_complexity=5, initial_depth=0, tournament_size=1, elite_count=1,
            unary_operators=(), binary_operators=(:+, :-), initial_pool_multiplier=1,
            constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
            parsimony_coefficient=0.0, evolution_strategy=strategy)
        result = search_scalar(X, y; config, rng=MersenneTwister(1234),
            initial_population=[deepcopy(original), deepcopy(original)], stop_loss=nothing)
        @test ScalarSearch._same_scalar_expression(result.best, original)
        expected = abs2(only(evaluate_scalar_reference(original, X)))
        @test result.best_loss == expected == (x == 1.0e16 ? 0.0 : 1.0)
        @test abs2(only(evaluate_scalar_reference(result.best, X))) == result.best_loss
        @test result.evaluations == 2
        @test result.constant_refinement_evaluations == 0
        @test result.variation.attempts == 0
        for leader in result.leaders
            @test abs2(only(evaluate_scalar_reference(leader.expr, X))) == leader.loss
        end
    end
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
    expected_best = ScalarSearch._node_to_scalar(state.best).expr
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
