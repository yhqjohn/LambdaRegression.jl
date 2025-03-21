"""
    search_multi_scalar

Coordinates persistent node-native regularized scalar islands, advancing
each through synchronous round budgets and applying bounded HOF/top migration
between rounds. The coordinator does not select a validation or test
expression: its only objective is the callback supplied by the caller.

`search_multi_scalar` returns the `ScalarSearch` result fields expected by the
program/scalar bridge.  Its matrix arguments are the normal scalar backend
contract; a contextual objective may deliberately ignore them when the
caller scores a materialised whole program.
"""
function _copy_config(config::ScalarSearchConfig, structural::Int,
        refinement::Int)
    structural >= config.population_size || throw(ArgumentError(
        "each scalar island budget must cover population_size"))
    return ScalarSearchConfig(
        population_size=config.population_size,
        max_evaluations=structural,
        max_complexity=config.max_complexity,
        initial_depth=config.initial_depth,
        tournament_size=config.tournament_size,
        complexity_balanced_tournament=config.complexity_balanced_tournament,
        elite_count=config.elite_count,
        unary_operators=config.unary_operators,
        binary_operators=config.binary_operators,
        constant_probability=config.constant_probability,
        mutation_probability=config.mutation_probability,
        crossover_probability=config.crossover_probability,
        random_restart_probability=config.random_restart_probability,
        constant_scale=config.constant_scale,
        constant_refinement_iterations=config.constant_refinement_iterations,
        constant_refinement_elites=config.constant_refinement_elites,
        constant_refinement_interval=config.constant_refinement_interval,
        constant_refinement_start_generation=config.constant_refinement_start_generation,
        constant_refinement_probability=config.constant_refinement_probability,
        max_constant_refinement_evaluations=refinement,
        constant_refinement_optimizer=config.constant_refinement_optimizer,
        parsimony_coefficient=config.parsimony_coefficient,
        initial_pool_multiplier=config.initial_pool_multiplier,
        initial_affine_basis=config.initial_affine_basis,
        evolution_strategy=config.evolution_strategy,
    )
end

function _split_budget(total::Int, parts::Int)
    base, remainder = divrem(total, parts)
    return [base + (index <= remainder) for index in 1:parts]
end

function _validated_initial_population(initial_population, config)
    initial = ScalarExpr[]
    initial_population === nothing && return initial
    for expression in initial_population
        expression isa ScalarExpr || throw(ArgumentError(
            "initial_population must contain ScalarExpr values"))
        scalar_complexity(expression) <= config.max_complexity ||
            throw(ArgumentError(
                "initial_population exceeds max_complexity"))
        push!(initial, deepcopy(expression))
    end
    return initial
end

"""Collect finite HOF or top-population node candidates from each island."""
function _donor_sources(states, source::Symbol; limit::Int=12)
    source in (:hof, :top) || throw(ArgumentError(
        "scalar migration source must be :hof or :top"))
    entries = NamedTuple[]
    for (island, state) in enumerate(states)
        candidates = source === :hof ?
            collect(values(state.archive)) :
            [member.candidate for member in state.population]
        sort!(candidates; by=candidate -> (
            candidate.loss, candidate.value.complexity, candidate.serial))
        finite_candidates = (
            candidate for candidate in candidates if isfinite(candidate.loss))
        for candidate in Iterators.take(finite_candidates, limit)
            push!(entries, (; island, candidate))
        end
    end
    return entries
end

"""Inject same-objective copies and report destination counts by source tier."""
function _inject_migrants!(states, rngs, hof_count::Int,
        top_count::Union{Nothing,Int}, top_migration_rate::Float64,
        donor_limit::Int)
    hof_count >= 0 || throw(ArgumentError("migrant_count cannot be negative"))
    top_count === nothing || top_count >= 0 || throw(ArgumentError(
        "top_migrant_count cannot be negative"))
    hof_sources = _donor_sources(states, :hof; limit=donor_limit)
    top_sources = _donor_sources(states, :top; limit=donor_limit)
    incoming_hof = zeros(Int, length(states))
    incoming_top = zeros(Int, length(states))
    for island in eachindex(states)
        desired_top = top_count === nothing ?
            _regularized_poisson(
                rngs[island], length(states[island].population) * top_migration_rate) :
            top_count
        eligible_hof = [entry for entry in hof_sources if entry.island != island]
        eligible_top = [entry for entry in top_sources if entry.island != island]
        shuffle!(rngs[island], eligible_hof)
        shuffle!(rngs[island], eligible_top)
        for entry in Iterators.take(
                eligible_hof, min(hof_count, length(eligible_hof)))
            _node_inject_candidate!(states[island], entry.candidate)
            incoming_hof[island] += 1
        end
        for entry in Iterators.take(
                eligible_top, min(desired_top, length(eligible_top)))
            _node_inject_candidate!(states[island], entry.candidate)
            incoming_top[island] += 1
        end
    end
    return incoming_hof, incoming_top
end

function _best_report(reports)
    best_index = 1
    for index in 2:length(reports)
        candidate = reports[index]
        incumbent = reports[best_index]
        candidate_key = (
            Float64(candidate.best_loss), scalar_complexity(candidate.best), index)
        incumbent_key = (
            Float64(incumbent.best_loss), scalar_complexity(incumbent.best), best_index)
        candidate_key < incumbent_key && (best_index = index)
    end
    return reports[best_index], best_index
end

function _merge_archives(reports)
    by_complexity = Dict{Int,NamedTuple}()
    for report in reports
        for entry in report.complexity_archive
            isfinite(entry.loss) || continue
            complexity = Int(entry.complexity)
            incumbent = get(by_complexity, complexity, nothing)
            if incumbent === nothing ||
                    (Float64(entry.loss),) < (Float64(incumbent.loss),)
                by_complexity[complexity] = (
                    expr=deepcopy(entry.expr),
                    loss=Float64(entry.loss),
                    complexity,
                )
            end
        end
    end
    return [by_complexity[complexity]
        for complexity in sort!(collect(keys(by_complexity)))]
end

"""
    search_multi_scalar(X, y; config, rng, initial_population, objective,
        stop_loss, island_count=2, rounds=2, migrant_count=2,
        top_migrant_count=nothing, ncycles_per_iteration=380,
        frequency_window=100_000, top_migration_rate=0.00036,
        donor_limit=12) -> NamedTuple

Run the existing node-native regularized search over persistent islands.  The
configured structural and constant-refinement budgets are split independently
across islands; the structural split is then advanced over `rounds` without
reinitialising populations.  A native initializer calls the supplied
contextual objective once for its baseline loss per island; that call is not
part of `evaluations`, so `diagnostics.objective_calls` reports it separately
from structural and constant-refinement counts.  Objective exceptions are
intentionally not caught.  `ncycles_per_iteration` is passed unchanged to
each native island; its default preserves the native 380-cycle behavior.
`frequency_window`, `top_migration_rate`, and `donor_limit` expose the existing
cross-island coordinator settings without changing their defaults. The
frequency window is shared across islands; the migration-rate factor scales
the per-destination Poisson top-tier request by population size, while the
donor limit caps eligible finite candidates from each source tier and island.
"""
function search_multi_scalar(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
    rng::AbstractRNG=default_rng(),
    initial_population=nothing,
    objective=nothing,
    stop_loss=nothing,
    island_count=2,
    rounds=2,
    migrant_count=2,
    top_migrant_count=nothing,
    ncycles_per_iteration::Int=380,
    frequency_window=100_000,
    top_migration_rate=0.00036,
    donor_limit=12,
)
    island_count isa Integer || throw(ArgumentError(
        "island_count must be an integer"))
    rounds isa Integer || throw(ArgumentError("rounds must be an integer"))
    migrant_count isa Integer || throw(ArgumentError(
        "migrant_count must be an integer"))
    island_count = Int(island_count)
    rounds = Int(rounds)
    migrant_count = Int(migrant_count)
    ncycles_per_iteration >= 1 || throw(ArgumentError(
        "ncycles_per_iteration must be positive"))
    island_count >= 1 || throw(ArgumentError("island_count must be positive"))
    rounds >= 1 || throw(ArgumentError("rounds must be positive"))
    migrant_count >= 0 || throw(ArgumentError("migrant_count cannot be negative"))
    top_migrant_count === nothing || top_migrant_count isa Integer ||
        throw(ArgumentError("top_migrant_count must be an integer or nothing"))
    top_migrant_count === nothing || (top_migrant_count = Int(top_migrant_count))
    top_migrant_count === nothing || top_migrant_count >= 0 ||
        throw(ArgumentError("top_migrant_count cannot be negative"))
    frequency_window isa Integer || throw(ArgumentError(
        "frequency_window must be an integer"))
    frequency_window = Int(frequency_window)
    frequency_window >= 1 || throw(ArgumentError(
        "frequency_window must be positive"))
    top_migration_rate isa Real && isfinite(Float64(top_migration_rate)) ||
        throw(ArgumentError("top_migration_rate must be finite and real"))
    top_migration_rate = Float64(top_migration_rate)
    top_migration_rate >= 0 || throw(ArgumentError(
        "top_migration_rate cannot be negative"))
    donor_limit isa Integer || throw(ArgumentError(
        "donor_limit must be an integer"))
    donor_limit = Int(donor_limit)
    donor_limit >= 0 || throw(ArgumentError("donor_limit cannot be negative"))
    rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    config.evolution_strategy === :regularized || throw(ArgumentError(
        "search_multi_scalar requires evolution_strategy=:regularized"))
    stop_loss === nothing || (stop_loss isa Real &&
        isfinite(Float64(stop_loss)) && Float64(stop_loss) >= 0) ||
        throw(ArgumentError("stop_loss must be nothing or finite and nonnegative"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))

    initial = _validated_initial_population(initial_population, config)
    structural_budgets = _split_budget(config.max_evaluations, island_count)
    all(budget >= config.population_size for budget in structural_budgets) ||
        throw(ArgumentError(
            "config.max_evaluations cannot fill population_size on every island"))
    refinement_budgets = _split_budget(
        config.max_constant_refinement_evaluations, island_count)
    island_configs = [
        _copy_config(config, structural_budgets[index], refinement_budgets[index])
        for index in 1:island_count
    ]

    island_rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:island_count]
    shared_frequency = _RegularizedFrequency(frequency_window, config.max_complexity)
    objective_calls = Ref(0)
    function tracked_objective(expression, extra...)
        objective_calls[] += 1
        if !isempty(extra) && applicable(objective, expression, extra...)
            return objective(expression, extra...)
        end
        return objective(expression)
    end
    state_objective = objective === nothing ? nothing : tracked_objective
    states = Vector{Any}(undef, island_count)
    for island in 1:island_count
        states[island] = _initialize_node_regularized_state(
            X, y;
            config=island_configs[island],
            rng=island_rngs[island],
            initial_population=[deepcopy(expression) for expression in initial],
            objective=state_objective,
            stop_loss=stop_loss,
            ncycles_per_iteration=ncycles_per_iteration,
            # Coordinator migration is the only cross-island tier.
            hof_migration_fraction=0.0,
            frequency=shared_frequency,
        )
    end

    # The initial population has already consumed population_size structural
    # attempts on every island. Spread only each island's remaining budget
    # over rounds so a round target never falls below that initialized count.
    remaining_budgets = [
        structural_budgets[index] - config.population_size
        for index in 1:island_count
    ]
    round_increments = [_split_budget(remaining_budgets[index], rounds)
        for index in 1:island_count]
    targets = fill(config.population_size, island_count)
    previous_evaluations = fill(config.population_size, island_count)
    previous_invalid = zeros(Int, island_count)
    previous_refinement = zeros(Int, island_count)
    previous_generations = zeros(Int, island_count)
    incoming_hof_total = zeros(Int, island_count)
    incoming_top_total = zeros(Int, island_count)
    round_ledger = NamedTuple[]

    for round in 1:rounds
        for island in 1:island_count
            targets[island] += round_increments[island][round]
            state = states[island]
            _step_node_regularized!(state;
                evaluation_limit=targets[island])
            report = _node_regularized_result(state)
            structural_delta = state.evaluations - previous_evaluations[island]
            invalid_delta = state.invalid_evaluations - previous_invalid[island]
            refinement_delta = state.constant_refinement_evaluations -
                previous_refinement[island]
            generation_delta = report.generations - previous_generations[island]
            push!(round_ledger, (;
                round,
                island,
                configured_structural_budget=round_increments[island][round],
                structural_evaluations=structural_delta,
                invalid_evaluations=invalid_delta,
                constant_refinement_evaluations=refinement_delta,
                generations=generation_delta,
                underfilled=state.evaluations < targets[island],
                attempt_limited=state.evaluations < targets[island] &&
                    !(stop_loss !== nothing && state.best.loss <= Float64(stop_loss)),
                objective_callbacks=objective === nothing ? 0 :
                    structural_delta + refinement_delta,
            ))
            previous_evaluations[island] = state.evaluations
            previous_invalid[island] = state.invalid_evaluations
            previous_refinement[island] = state.constant_refinement_evaluations
            previous_generations[island] = report.generations
        end
        if round < rounds
            incoming_hof, incoming_top = _inject_migrants!(
                states, island_rngs, migrant_count, top_migrant_count,
                top_migration_rate, donor_limit)
            incoming_hof_total .+= incoming_hof
            incoming_top_total .+= incoming_top
        end
    end

    reports = [
        _node_regularized_result(state)
        for state in states
    ]
    best_report, best_island = _best_report(reports)
    complexity_archive = _merge_archives(reports)
    evaluations = sum(report.evaluations for report in reports)
    invalid_evaluations = sum(report.invalid_evaluations for report in reports)
    refinement_evaluations = sum(
        report.constant_refinement_evaluations for report in reports)
    refinement_invalid = sum(
        report.constant_refinement_invalid_evaluations for report in reports)
    generations = sum(report.generations for report in reports)
    total_migrations = sum(incoming_hof_total) + sum(incoming_top_total)
    solved = stop_loss !== nothing && any(
        report.best_loss <= Float64(stop_loss) for report in reports)
    termination = solved ? :solved :
        (evaluations >= config.max_evaluations ? :evaluation_cap : :paused)
    per_island = [
        (
            island=index,
            configured_structural_budget=structural_budgets[index],
            configured_constant_refinement_budget=refinement_budgets[index],
            initialization_count=1,
            initialization_structural_evaluations=config.population_size,
            structural_evaluations=reports[index].evaluations,
            invalid_evaluations=reports[index].invalid_evaluations,
            constant_refinement_evaluations=
                reports[index].constant_refinement_evaluations,
            constant_refinement_invalid_evaluations=
                reports[index].constant_refinement_invalid_evaluations,
            objective_callbacks=objective === nothing ? 0 :
                1 + reports[index].evaluations +
                reports[index].constant_refinement_evaluations,
            baseline_objective_calls=objective === nothing ? 0 : 1,
            hof_migrations=incoming_hof_total[index],
            top_migrations=incoming_top_total[index],
            generations=reports[index].generations,
            offspring_attempts=states[index].offspring_attempts,
            accepted_offspring=states[index].accepted_offspring,
            rejected_offspring=states[index].rejected_offspring,
            skipped_variations=states[index].skipped_variations,
        )
        for index in 1:island_count
    ]
    diagnostics = (;
        island_count,
        rounds,
        rounds_completed=rounds,
        initialization_count=island_count,
        objective_calls=objective_calls[],
        total_objective_callbacks=objective_calls[],
        baseline_objective_calls=objective === nothing ? 0 : island_count,
        structural_objective_calls=evaluations,
        constant_refinement_objective_calls=refinement_evaluations,
        structural_evaluations=evaluations,
        invalid_evaluations,
        constant_refinement_evaluations=refinement_evaluations,
        constant_refinement_invalid_evaluations=refinement_invalid,
        migration_events=total_migrations,
        hof_migrations=sum(incoming_hof_total),
        top_migrations=sum(incoming_top_total),
        offspring_attempts=sum(state.offspring_attempts for state in states),
        shared_frequency=true,
        single_island_hof_migration_fraction=0.0,
        best_island,
        per_island,
        round_ledger,
    )
    variation = (
        attempts=sum(report.variation.attempts for report in reports),
        changed=sum(report.variation.changed for report in reports),
        accepted_offspring=sum(report.variation.accepted_offspring
            for report in reports),
        rejected_offspring=sum(report.variation.rejected_offspring
            for report in reports),
        skipped_variations=sum(report.variation.skipped_variations
            for report in reports),
        final_population_size=sum(report.variation.final_population_size
            for report in reports),
    )
    return (
        best=deepcopy(best_report.best),
        best_loss=Float64(best_report.best_loss),
        evaluations,
        invalid_evaluations,
        constant_refinement_evaluations=refinement_evaluations,
        constant_refinement_invalid_evaluations=refinement_invalid,
        generations,
        termination,
        complexity_archive,
        variation,
        diagnostics,
    )
end

@testsnippet MultiIslandSearchImports begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ScalarSearch:
        ScalarSearchConfig, scalar_complexity, scalar_input, search_multi_scalar
end

@testitem "Multi-island scalar coordinator reports layered work" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=8,
        max_complexity=6,
        initial_depth=0,
        tournament_size=1,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_probability=0.0,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        evolution_strategy=:regularized,
    )
    objective_calls = Ref(0)
    objective = expression -> begin
        objective_calls[] += 1
        return Float64(scalar_complexity(expression))
    end
    result = search_multi_scalar(X, y;
        config,
        rng=MersenneTwister(2000),
        initial_population=[scalar_input(1)],
        objective,
        island_count=2,
        rounds=2,
        migrant_count=1,
        top_migrant_count=1)

    @test result.evaluations == 8
    @test result.constant_refinement_evaluations == 0
    @test result.diagnostics.objective_calls == objective_calls[]
    @test result.diagnostics.baseline_objective_calls == 2
    @test result.diagnostics.structural_objective_calls == result.evaluations
    @test result.diagnostics.initialization_count == 2
    @test result.diagnostics.rounds_completed == 2
    @test length(result.diagnostics.round_ledger) == 4
    @test sum(item.structural_evaluations for item in result.diagnostics.per_island) == 8
    @test result.diagnostics.hof_migrations > 0
    @test result.diagnostics.top_migrations > 0
end

@testitem "Multi-island scalar coordinator exposes bounded policy settings" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=8,
        max_complexity=6,
        initial_depth=0,
        tournament_size=1,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_probability=0.0,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        evolution_strategy=:regularized,
    )
    args = (; config, rng=MersenneTwister(1111),
        initial_population=[scalar_input(1)], objective=scalar_complexity,
        island_count=2, rounds=2, migrant_count=1)
    result = search_multi_scalar(X, y; args...,
        frequency_window=7, top_migration_rate=0.0, donor_limit=1)
    @test result.evaluations == 8
    @test result.diagnostics.shared_frequency
    @test result.diagnostics.hof_migrations > 0
    @test result.diagnostics.top_migrations == 0
    @test result.diagnostics.migration_events == result.diagnostics.hof_migrations

    no_donors = search_multi_scalar(X, y; args...,
        frequency_window=7, top_migration_rate=0.0, donor_limit=0)
    @test no_donors.diagnostics.migration_events == 0
    @test no_donors.diagnostics.hof_migrations == 0

    @test_throws ArgumentError search_multi_scalar(X, y; args..., frequency_window=0)
    @test_throws ArgumentError search_multi_scalar(X, y; args..., frequency_window=1.5)
    @test_throws ArgumentError search_multi_scalar(X, y; args..., top_migration_rate=-0.1)
    @test_throws ArgumentError search_multi_scalar(X, y; args..., top_migration_rate=Inf)
    @test_throws ArgumentError search_multi_scalar(X, y; args..., donor_limit=-1)
    @test_throws ArgumentError search_multi_scalar(X, y; args..., donor_limit=1.5)
end
