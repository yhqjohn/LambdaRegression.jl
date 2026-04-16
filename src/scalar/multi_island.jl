"""
    search_multi_scalar

Coordinates persistent node-native regularized scalar islands, advancing
each through synchronous round budgets and applying bounded HOF/top migration
between rounds. The coordinator does not select a validation or test
expression: its only objective is the callback supplied by the caller.

`rounds` is an upper bound. The session completes at that bound, at
initialization, or after a coordinator round when every island is terminal by
stop loss, structural budget, or shared total budget. A round observer still
sees the actual round that reaches this boundary, but no migration follows it.
Independent constant-refinement-budget exhaustion alone does not make an island
terminal.

`search_multi_scalar` returns the `ScalarSearch` result fields expected by the
program/scalar bridge.  Its matrix arguments are the normal scalar backend
contract; a contextual objective may deliberately ignore them when the
caller scores a materialised whole program.
"""
function _copy_config(config::ScalarSearchConfig, structural::Int,
        refinement::Int, total::Union{Nothing,Int})
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
        constant_refinement_timing=config.constant_refinement_timing,
        constant_refinement_resets_age=config.constant_refinement_resets_age,
        rotation_noop_reproduction=config.rotation_noop_reproduction,
        deletion_policy=config.deletion_policy,
        max_constant_refinement_evaluations=refinement,
        max_total_evaluations=total,
        constant_refinement_optimizer=config.constant_refinement_optimizer,
        parsimony_coefficient=config.parsimony_coefficient,
        initial_pool_multiplier=config.initial_pool_multiplier,
        initial_affine_basis=config.initial_affine_basis,
        evolution_strategy=config.evolution_strategy,
        proposal_biases=config.proposal_biases,
    )
end

function _split_budget(total::Int, parts::Int)
    base, remainder = divrem(total, parts)
    return [base + (index <= remainder) for index in 1:parts]
end

function _validated_initial_population(initial_population, config, feature_count)
    initial = ScalarExpr[]
    initial_population === nothing && return initial
    for expression in initial_population
        expression isa ScalarExpr || throw(ArgumentError(
            "initial_population must contain ScalarExpr values"))
        scalar_complexity(expression) <= config.max_complexity ||
            throw(ArgumentError(
                "initial_population exceeds max_complexity"))
        _required_features(expression) <= feature_count ||
            throw(ArgumentError(
                "initial_population references a feature outside the input matrix"))
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

"""Collect the global loss-complexity frontier, including each destination's own archive."""
function _global_pareto_donor_sources(states)
    by_complexity = Dict{Int,NamedTuple}()
    for (island, state) in enumerate(states), candidate in values(state.archive)
        isfinite(candidate.loss) || continue
        complexity = candidate.value.complexity
        entry = (; island, candidate)
        incumbent = get(by_complexity, complexity, nothing)
        if incumbent === nothing ||
                (candidate.loss, island, candidate.serial) <
                (incumbent.candidate.loss, incumbent.island,
                    incumbent.candidate.serial)
            by_complexity[complexity] = entry
        end
    end
    entries = NamedTuple[]
    best_loss = Inf
    for complexity in sort!(collect(keys(by_complexity)))
        entry = by_complexity[complexity]
        entry.candidate.loss < best_loss || continue
        push!(entries, entry)
        best_loss = entry.candidate.loss
    end
    return entries
end

"""Inject same-objective copies and report destination counts by source tier."""
function _inject_migrants!(states, rngs, hof_count::Int,
        top_count::Union{Nothing,Int}, top_migration_rate::Float64,
        donor_limit::Int; skip_terminal_destinations::Bool=false,
        hof_donor_policy::Symbol=:local_best)
    hof_count >= 0 || throw(ArgumentError("migrant_count cannot be negative"))
    top_count === nothing || top_count >= 0 || throw(ArgumentError(
        "top_migrant_count cannot be negative"))
    hof_sources = hof_donor_policy === :global_pareto ?
        _global_pareto_donor_sources(states) :
        _donor_sources(states, :hof; limit=donor_limit)
    top_sources = _donor_sources(states, :top; limit=donor_limit)
    incoming_hof = zeros(Int, length(states))
    incoming_top = zeros(Int, length(states))
    for island in eachindex(states)
        skip_terminal_destinations && _node_state_terminal(states[island]) &&
            continue
        desired_top = top_count === nothing ?
            _regularized_poisson(
                rngs[island], length(states[island].population) * top_migration_rate) :
            top_count
        eligible_hof = hof_donor_policy === :global_pareto ? copy(hof_sources) :
            [entry for entry in hof_sources if entry.island != island]
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

function _node_state_terminal(state::_NodeRegularizedState)
    solved = state.flow.stop_loss !== nothing &&
        state.best.loss <= state.flow.stop_loss
    structural_cap = state.evaluations >= state.config.max_evaluations
    total_cap = state.config.max_total_evaluations !== nothing &&
        _node_state_total_remaining(state) <= 0
    return solved || structural_cap || total_cap
end

function _multi_scalar_total_remaining(session)
    total = session.config.max_total_evaluations
    total === nothing && return nothing
    spent = sum(state.evaluations + state.constant_refinement_evaluations
        for state in session.progress.states)
    return max(0, total - spent)
end

function _multi_scalar_completed(session)
    progress = session.progress
    progress.rounds_completed >= session.rounds ||
        all(_node_state_terminal, progress.states)
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

mutable struct _MultiScalarProgress
    states::Vector{Any}
    island_rngs::Vector{MersenneTwister}
    shared_frequency::_RegularizedFrequency
    objective_calls::Base.RefValue{Int}
    round_increments::Vector{Vector{Int}}
    targets::Vector{Int}
    previous_evaluations::Vector{Int}
    previous_invalid::Vector{Int}
    previous_refinement::Vector{Int}
    previous_generations::Vector{Int}
    incoming_hof_total::Vector{Int}
    incoming_top_total::Vector{Int}
    round_ledger::Vector{NamedTuple}
    rounds_completed::Int
end

"""Session state for one fixed-context, process-local scalar task."""
struct MultiScalarSession
    X::AbstractMatrix{<:Real}
    y::AbstractVector{<:Real}
    config::ScalarSearchConfig
    objective::Any
    state_objective::Any
    stop_loss::Any
    island_count::Int
    rounds::Int
    migrant_count::Int
    top_migrant_count::Union{Nothing,Int}
    ncycles_per_iteration::Int
    frequency_window::Int
    top_migration_rate::Float64
    donor_limit::Int
    hof_donor_policy::Symbol
    round_observer::Any
    structural_budgets::Vector{Int}
    refinement_budgets::Vector{Int}
    island_configs::Vector{ScalarSearchConfig}
    progress::_MultiScalarProgress
end

"""
    initialize_multi_scalar(X, y; config, rng, initial_population, objective,
        stop_loss, island_count=2, rounds=2, migrant_count=2,
        top_migrant_count=nothing, ncycles_per_iteration=380,
        use_frequency=true, use_frequency_in_tournament=true,
        frequency_window=100_000, top_migration_rate=0.00036,
        donor_limit=12, hof_donor_policy=:local_best,
        round_observer=nothing) -> MultiScalarSession

Create an in-memory session for one fixed scalar task and objective. The
configured structural and constant-refinement budgets are split
independently across islands and remain fixed for the session. When
`max_total_evaluations` is configured, its combined structural/refinement
allowance is also split across islands; baseline callbacks are excluded. `X`
and `y` are borrowed
read-only inputs; keep them and any state captured by `objective` unchanged
while the session is live. A session must not be reused for another task or
context. Session fields are internal and must not be mutated; use only
`advance_multi_scalar!` and `multi_scalar_result` to operate on the session. A
native initializer calls the supplied objective once for its baseline loss per
island; that call is not part of `evaluations`, so the result
reports it separately from structural and constant-refinement counts.
The optional `constant_jacobian(node, X)` callback enables direct-MSE BFGS
derivatives; it is rejected with a custom objective or a different optimizer.
It returns `(prediction, jacobian, complete)` with a constants-by-samples
Jacobian in depth-first constant-leaf order, using the native operators and
domains. The canonical evaluator still owns the forward loss. The callback
must not mutate its inputs and must be safe for serial use by all islands;
unexpected errors propagate. A non-finite derivative or `complete=false` uses
charged central differences. Default `nothing` preserves finite differences
and requires no automatic-differentiation dependency.
With this callback, `constant_refinement_evaluations` and its budget count
forward evaluations plus Jacobian calls, including their internal forward
work, as one unit each. These units do not imply equal cost. The separate
`diagnostics.constant_refinement_work` and per-island fields report forward,
derivative, invalid-derivative and fallback counts; invalid evaluation counts
still mean forward failures. Total budgets use the same combined units.
`ncycles_per_iteration` is passed unchanged to each native island; its default
preserves the native 380-cycle behavior. `frequency_window`,
`top_migration_rate`, and `donor_limit` expose the existing cross-island
coordinator settings without changing their defaults. `use_frequency` controls
the acceptance-probability correction by old/new complexity frequency;
it may be one `Bool` shared by all islands or an `AbstractVector{Bool}` with
one entry per island, copied during initialization. A vector with the wrong
length is rejected before any objective calls. `use_frequency_in_tournament`
independently controls frequency pressure in tournament ranking and remains a
shared `Bool`; both settings default to `true`. The shared frequency histogram
continues to update at island maintenance boundaries and remains available to
observers regardless of these switches.

`hof_donor_policy=:local_best` uses each island's lowest-loss archive entries,
limited by `donor_limit`, and excludes that destination's own entries. Set
`:global_pareto` to merge all historical island archives into one best member
per complexity (loss ties use island index, then serial), and retain only
strict loss improvements as complexity increases. This global pool includes
the destination's own history and ignores `donor_limit`; the top-population
pool is still limited by it and excludes the destination. Both policies sample
archive donors without replacement, inject at most `migrant_count` copies into
random population slots with fresh ages, and add no scoring calls. Set
`migrant_count=0` to disable archive injection. All migration clocks and budgets
are unchanged.

Optional `round_observer(round, snapshot)` runs after all islands complete that
round, before migration. The snapshot is a zero-argument function: call it only
at wanted boundaries to obtain detached scalar population/archive records,
age clocks, cumulative budgets, and a copy of shared complexity frequencies.
Neither it nor the coordinator draws randomness or scores candidates for
observation. Call the snapshot synchronously; retaining it would observe later
mutable state. Observer exceptions propagate. No snapshot work occurs by
default. Scorer and observer exceptions propagate; an advance that throws may
have changed part of a round, so discard that session instead of retrying it.

`rounds` bounds coordinator rounds rather than requiring all of them to run.
If every island is already terminal after initialization, the session completes
with zero rounds and no observer or migration. If a round makes every island
terminal, its observer runs after island advancement and migration is skipped.
Partially terminal islands retain the existing coordinator behavior.
"""
function initialize_multi_scalar(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
    rng::AbstractRNG=default_rng(),
    initial_population=nothing,
    objective=nothing,
    constant_jacobian=nothing,
    stop_loss=nothing,
    island_count=2,
    rounds=2,
    migrant_count=2,
    top_migrant_count=nothing,
    ncycles_per_iteration::Int=380,
    use_frequency=true,
    use_frequency_in_tournament::Bool=true,
    frequency_window=100_000,
    top_migration_rate=0.00036,
    donor_limit=12,
    hof_donor_policy=:local_best,
    round_observer=nothing,
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
    acceptance_frequency = if use_frequency isa Bool
        fill(use_frequency, island_count)
    elseif use_frequency isa AbstractVector{Bool}
        length(use_frequency) == island_count || throw(ArgumentError(
            "use_frequency vector length must equal island_count"))
        Bool[flag for flag in use_frequency]
    else
        throw(ArgumentError(
            "use_frequency must be a Bool or AbstractVector{Bool}"))
    end
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
    hof_donor_policy in (:local_best, :global_pareto) || throw(ArgumentError(
        "hof_donor_policy must be :local_best or :global_pareto"))
    rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    config.evolution_strategy === :regularized || throw(ArgumentError(
        "search_multi_scalar requires evolution_strategy=:regularized"))
    stop_loss === nothing || (stop_loss isa Real &&
        isfinite(Float64(stop_loss))) ||
        throw(ArgumentError("stop_loss must be nothing or finite"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))
    constant_jacobian === nothing || (objective === nothing &&
        config.constant_refinement_optimizer === :bfgs) || throw(ArgumentError(
        "constant_jacobian requires direct MSE and BFGS refinement"))

    initial = _validated_initial_population(
        initial_population, config, size(X, 1))
    structural_budgets = _split_budget(config.max_evaluations, island_count)
    all(budget >= config.population_size for budget in structural_budgets) ||
        throw(ArgumentError(
            "config.max_evaluations cannot fill population_size on every island"))
    refinement_budgets = _split_budget(
        config.max_constant_refinement_evaluations, island_count)
    total_budgets = config.max_total_evaluations === nothing ?
        Union{Nothing,Int}[nothing for _ in 1:island_count] :
        Union{Nothing,Int}[_split_budget(
            config.max_total_evaluations, island_count)...]
    all(budget === nothing || budget >= config.population_size
        for budget in total_budgets) || throw(ArgumentError(
            "config.max_total_evaluations cannot fill population_size on every island"))
    island_configs = [
        _copy_config(config, structural_budgets[index], refinement_budgets[index],
            total_budgets[index])
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
            constant_jacobian,
            stop_loss=stop_loss,
            ncycles_per_iteration=ncycles_per_iteration,
            use_frequency=acceptance_frequency[island], use_frequency_in_tournament,
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
    progress = _MultiScalarProgress(
        states, island_rngs, shared_frequency, objective_calls,
        round_increments, targets, previous_evaluations, previous_invalid,
        previous_refinement, previous_generations, incoming_hof_total,
        incoming_top_total, NamedTuple[], 0)
    return MultiScalarSession(
        X, y, config, objective, state_objective, stop_loss,
        island_count, rounds, migrant_count, top_migrant_count,
        ncycles_per_iteration, frequency_window, top_migration_rate,
        donor_limit, hof_donor_policy, round_observer, structural_budgets,
        refinement_budgets, island_configs, progress)
end

function _advance_multi_scalar_round!(session::MultiScalarSession)
    progress = session.progress
    round = progress.rounds_completed + 1
    for island in 1:session.island_count
        state = progress.states[island]
        session.config.max_total_evaluations !== nothing &&
            _node_state_terminal(state) && continue
        progress.targets[island] += progress.round_increments[island][round]
        _step_node_regularized!(state;
            evaluation_limit=progress.targets[island])
        generations = _node_state_generations(state)
        structural_delta = state.evaluations - progress.previous_evaluations[island]
        invalid_delta = state.invalid_evaluations - progress.previous_invalid[island]
        refinement_delta = state.constant_refinement_evaluations -
            progress.previous_refinement[island]
        generation_delta = generations - progress.previous_generations[island]
        total_cap_hit = session.config.max_total_evaluations !== nothing &&
            _node_state_total_remaining(state) <= 0
        underfilled = state.evaluations < progress.targets[island]
        push!(progress.round_ledger, (;
            round,
            island,
            configured_structural_budget=progress.round_increments[island][round],
            structural_evaluations=structural_delta,
            invalid_evaluations=invalid_delta,
            constant_refinement_evaluations=refinement_delta,
            generations=generation_delta,
            underfilled,
            total_cap_hit,
            attempt_limited=underfilled && !total_cap_hit &&
                !(session.stop_loss !== nothing &&
                    state.best.loss <= Float64(session.stop_loss)),
            objective_callbacks=session.objective === nothing ? 0 :
                structural_delta + refinement_delta,
        ))
        progress.previous_evaluations[island] = state.evaluations
        progress.previous_invalid[island] = state.invalid_evaluations
        progress.previous_refinement[island] =
            state.constant_refinement_evaluations
        progress.previous_generations[island] = generations
    end
    if session.round_observer !== nothing
        snapshot = () -> (;
            frequencies=copy(progress.shared_frequency.normalized_frequencies),
            islands=[(;
                island,
                evaluations=state.evaluations,
                constant_evaluations=state.constant_refinement_evaluations,
                birth_clock=state.birth_clock,
                population=[(;
                    expr=_node_to_scalar(member.candidate).expr,
                    loss=member.candidate.loss,
                    complexity=member.candidate.value.complexity,
                    serial=member.candidate.serial, born=member.born,
                ) for member in state.population],
                archive=[(;
                    expr=_node_to_scalar(candidate).expr,
                    loss=candidate.loss, complexity,
                    serial=candidate.serial,
                ) for (complexity, candidate) in
                    sort!(collect(state.archive); by=first)],
            ) for (island, state) in enumerate(progress.states)],
        )
        session.round_observer(round, snapshot)
    end
    if round < session.rounds &&
            !all(_node_state_terminal, progress.states)
        incoming_hof, incoming_top = _inject_migrants!(
            progress.states, progress.island_rngs, session.migrant_count,
            session.top_migrant_count, session.top_migration_rate,
            session.donor_limit;
            hof_donor_policy=session.hof_donor_policy,
            skip_terminal_destinations=
                session.config.max_total_evaluations !== nothing)
        progress.incoming_hof_total .+= incoming_hof
        progress.incoming_top_total .+= incoming_top
    end
    progress.rounds_completed = round
    return session
end

"""
    advance_multi_scalar!(session; max_rounds=1) -> session

Advance at most `max_rounds` coordinator rounds. This slices the session's
fixed structural and refinement budgets; it cannot add budget. A round that
leaves every island terminal completes its observer boundary without a final
migration. A session terminal after initialization runs no rounds. Zero is a
no-op, completed sessions are unchanged, and negative values throw
`ArgumentError`.
"""
function advance_multi_scalar!(session::MultiScalarSession; max_rounds::Int=1)
    max_rounds >= 0 || throw(ArgumentError("max_rounds cannot be negative"))
    progress = session.progress
    requested = min(max_rounds, session.rounds - progress.rounds_completed)
    for _ in 1:requested
        _multi_scalar_completed(session) && break
        _advance_multi_scalar_round!(session)
    end
    return session
end

"""
    multi_scalar_result(session) -> NamedTuple

Return a detached report for the current session boundary. Projection performs
no scoring or random draws. `diagnostics.rounds_completed` reports actual
progress. Any session completes early when every island has reached a
structural, applicable total-evaluation, or stop-loss boundary; no empty rounds
are counted. Shared total-cap exhaustion retains priority in termination
classification. Termination identifies the boundary that ended the session.
"""
function multi_scalar_result(session::MultiScalarSession)
    progress = session.progress
    config = session.config
    states = progress.states
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
    work = [_node_constant_work(state.score, state.constant_refinement_evaluations)
        for state in states]
    constant_refinement_work = (;
        (key => sum(getproperty(row, key) for row in work)
            for key in keys(first(work)))...)
    generations = sum(report.generations for report in reports)
    total_migrations = sum(progress.incoming_hof_total) +
        sum(progress.incoming_top_total)
    solved = session.stop_loss !== nothing && any(
        report.best_loss <= Float64(session.stop_loss) for report in reports)
    total_remaining = _multi_scalar_total_remaining(session)
    all_terminal = all(_node_state_terminal, states)
    termination = if config.max_total_evaluations !== nothing &&
            total_remaining === 0
        :total_evaluation_cap
    elseif all_terminal
        if config.max_total_evaluations === nothing
            # Preserve legacy no-total-cap solved precedence at the new early
            # terminal boundary: any solved island takes precedence over the
            # structural cap of another island.
            solved ? :solved :
                (evaluations >= config.max_evaluations ? :evaluation_cap : :paused)
        else
            all(state -> session.stop_loss !== nothing &&
                state.best.loss <= Float64(session.stop_loss), states) ? :solved :
                :evaluation_cap
        end
    elseif progress.rounds_completed < session.rounds
        :paused
    else
        solved ? :solved :
            (evaluations >= config.max_evaluations ? :evaluation_cap : :paused)
    end
    per_island = [
        (
            island=index,
            configured_structural_budget=session.structural_budgets[index],
            configured_constant_refinement_budget=session.refinement_budgets[index],
            configured_total_evaluation_budget=
                session.island_configs[index].max_total_evaluations,
            remaining_total_evaluations=config.max_total_evaluations === nothing ?
                nothing : max(0, _node_state_total_remaining(states[index])),
            initialization_count=1,
            initialization_structural_evaluations=config.population_size,
            structural_evaluations=reports[index].evaluations,
            invalid_evaluations=reports[index].invalid_evaluations,
            constant_refinement_evaluations=
                reports[index].constant_refinement_evaluations,
            constant_refinement_invalid_evaluations=
                reports[index].constant_refinement_invalid_evaluations,
            constant_refinement_work=work[index],
            cycle_refinement_calls=
                reports[index].variation.cycle_refinement_calls,
            offspring_refinement_calls=
                reports[index].variation.offspring_refinement_calls,
            objective_callbacks=session.objective === nothing ? 0 :
                1 + reports[index].evaluations +
                reports[index].constant_refinement_evaluations,
            baseline_objective_calls=session.objective === nothing ? 0 : 1,
            hof_migrations=progress.incoming_hof_total[index],
            top_migrations=progress.incoming_top_total[index],
            generations=reports[index].generations,
            offspring_attempts=states[index].offspring_attempts,
            accepted_offspring=states[index].accepted_offspring,
            rejected_offspring=states[index].rejected_offspring,
            skipped_variations=states[index].skipped_variations,
            rotation_copy_attempts=get(
                states[index].mutation_kind_counts, :rotation_copy, 0),
            rotation_copy_evaluations=states[index].rotation_copy_evaluations,
            rotation_copy_accepted=states[index].rotation_copy_accepted,
            changed_variations=states[index].changed_variations,
        )
        for index in 1:session.island_count
    ]
    diagnostics = (;
        island_count=session.island_count,
        rounds=session.rounds,
        rounds_completed=progress.rounds_completed,
        initialization_count=session.island_count,
        objective_calls=progress.objective_calls[],
        total_objective_callbacks=progress.objective_calls[],
        baseline_objective_calls=session.objective === nothing ? 0 :
            session.island_count,
        structural_objective_calls=evaluations,
        constant_refinement_objective_calls=constant_refinement_work.forward_evaluations,
        constant_refinement_work,
        structural_evaluations=evaluations,
        invalid_evaluations,
        constant_refinement_evaluations=refinement_evaluations,
        constant_refinement_invalid_evaluations=refinement_invalid,
        cycle_refinement_calls=sum(
            report.variation.cycle_refinement_calls for report in reports),
        offspring_refinement_calls=sum(
            report.variation.offspring_refinement_calls for report in reports),
        total_evaluation_budget=config.max_total_evaluations,
        remaining_total_evaluations=total_remaining,
        migration_events=total_migrations,
        hof_migrations=sum(progress.incoming_hof_total),
        hof_donor_policy=session.hof_donor_policy,
        top_migrations=sum(progress.incoming_top_total),
        offspring_attempts=sum(state.offspring_attempts for state in states),
        shared_frequency=true,
        single_island_hof_migration_fraction=0.0,
        best_island,
        per_island,
        round_ledger=copy(progress.round_ledger),
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
        cycle_refinement_calls=sum(
            report.variation.cycle_refinement_calls for report in reports),
        offspring_refinement_calls=sum(
            report.variation.offspring_refinement_calls for report in reports),
        total_evaluation_budget=config.max_total_evaluations,
        remaining_total_evaluations=total_remaining,
        final_population_size=sum(report.variation.final_population_size
            for report in reports),
    )
    # A bounded real-member projection for downstream material publication.
    # Reports already own these scalar expressions; selection adds no score
    # calls or RNG draws. It does not expose complete live populations.
    leaders = NamedTuple[]
    seen_leaders = Set{_ScalarSignature}()
    for report in reports, member in report.leaders
        isfinite(member.loss) || continue
        signature = _expression_signature(member.expr)
        signature in seen_leaders && continue
        push!(seen_leaders, signature)
        push!(leaders, (; expr=deepcopy(member.expr), loss=member.loss,
            complexity=member.complexity))
        length(leaders) >= 8 && break
    end
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
        leaders,
        variation,
        diagnostics,
    )
end

"""Read detached counters without publishing expressions, scoring or random draws.
`completed` becomes true at the configured round count or earlier when all
islands reach a structural, applicable total-evaluation, or stop-loss boundary.
Independent constant-refinement-budget exhaustion alone is not terminal.
`rounds_completed` always reports actual rounds.
"""
function multi_scalar_progress(session::MultiScalarSession)
    progress = session.progress
    structural = sum(state.evaluations for state in progress.states)
    constants = sum(state.constant_refinement_evaluations for state in progress.states)
    derivatives = sum(_node_constant_work(state.score,
        state.constant_refinement_evaluations).derivative_evaluations
        for state in progress.states)
    return (; rounds=session.rounds, rounds_completed=progress.rounds_completed,
        completed=_multi_scalar_completed(session),
        constant_refinement_work=(forward_evaluations=constants - derivatives,
            derivative_evaluations=derivatives),
        initialization_count=session.island_count,
        budget=(; structural_evaluations=session.config.max_evaluations,
            constant_refinement_evaluations=session.config.max_constant_refinement_evaluations,
            total_evaluations=session.config.max_total_evaluations),
        remaining=(; rounds=session.rounds - progress.rounds_completed,
            structural_evaluations=session.config.max_evaluations - structural,
            constant_refinement_evaluations=session.config.max_constant_refinement_evaluations - constants,
            total_evaluations=_multi_scalar_total_remaining(session)),
        cumulative=(; objective_calls=progress.objective_calls[],
            structural_evaluations=structural,
            constant_refinement_evaluations=constants,
            total_evaluations=structural + constants,
            baseline_objective_calls=session.objective === nothing ? 0 : session.island_count))
end

"""Run up to the configured round count, stopping early when every island is
terminal, and return the existing multi-island report. See
`initialize_multi_scalar` for options, borrowed-input ownership, and the
fixed-context session contract.
"""
function search_multi_scalar(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
    rng::AbstractRNG=default_rng(),
    initial_population=nothing,
    objective=nothing,
    constant_jacobian=nothing,
    stop_loss=nothing,
    island_count=2,
    rounds=2,
    migrant_count=2,
    top_migrant_count=nothing,
    ncycles_per_iteration::Int=380,
    use_frequency=true,
    use_frequency_in_tournament::Bool=true,
    frequency_window=100_000,
    top_migration_rate=0.00036,
    donor_limit=12,
    hof_donor_policy=:local_best,
    round_observer=nothing,
)
    session = initialize_multi_scalar(X, y;
        config, rng, initial_population, objective, constant_jacobian, stop_loss, island_count,
        rounds, migrant_count, top_migrant_count, ncycles_per_iteration,
        use_frequency, use_frequency_in_tournament,
        frequency_window, top_migration_rate, donor_limit, hof_donor_policy,
        round_observer)
    advance_multi_scalar!(session; max_rounds=session.rounds)
    return multi_scalar_result(session)
end

@testsnippet MultiIslandSearchImports begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.ScalarSearch:
        ScalarSearchConfig, scalar_complexity, scalar_input, scalar_constant,
        scalar_binary, evaluate_scalar_reference, MultiScalarSession,
        initialize_multi_scalar, advance_multi_scalar!, multi_scalar_result,
        multi_scalar_progress, search_multi_scalar
end

@testitem "multi scalar explicit Jacobian capability and island accounting" setup=[MultiIslandSearchImports] begin
    import LambdaRegression.ScalarSearch
    S = ScalarSearch
    X = reshape([0.2, 0.5, 0.9, 1.3], 1, :)
    y = 2vec(X)
    config = ScalarSearchConfig(population_size=4, tournament_size=2, elite_count=1,
        constant_refinement_elites=1,
        max_evaluations=40, max_constant_refinement_evaluations=12,
        max_total_evaluations=24, constant_refinement_iterations=8,
        constant_refinement_start_generation=1,
        evolution_strategy=:regularized)
    jacobian = (node, data) -> (nothing, copy(data), true)
    @test_throws ArgumentError initialize_multi_scalar(X, y; config,
        objective=scalar_complexity, constant_jacobian=jacobian)
    other = ScalarSearchConfig(evolution_strategy=:regularized,
        constant_refinement_optimizer=:nelder_mead)
    @test_throws ArgumentError initialize_multi_scalar(X, y; config=other,
        constant_jacobian=jacobian)
    expr = scalar_binary(:*, scalar_constant(0.2), scalar_input(1))
    session = initialize_multi_scalar(X, y; config, constant_jacobian=jacobian,
        initial_population=fill(expr, 4), island_count=2, rounds=2,
        rng=MersenneTwister(1234))
    for state in session.progress.states
        S._node_state_refine_candidate!(state, first(state.population).candidate,
            1, :cycle)
        @test state.constant_refinement_evaluations <= 6
    end
    @test session.progress.states[1].score !== session.progress.states[2].score
    report = multi_scalar_result(session)
    counts = report.diagnostics.constant_refinement_work
    @test counts.derivative_evaluations > 0
    @test counts.forward_evaluations + counts.derivative_evaluations ==
        report.constant_refinement_evaluations
    @test report.diagnostics.constant_refinement_objective_calls == counts.forward_evaluations
    @test sum(row.constant_refinement_work.derivative_evaluations
        for row in report.diagnostics.per_island) == counts.derivative_evaluations
    progress = multi_scalar_progress(session)
    @test progress.constant_refinement_work.derivative_evaluations == counts.derivative_evaluations
    @test progress.remaining.total_evaluations == 24 - report.evaluations -
        report.constant_refinement_evaluations
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
    @test_throws ArgumentError search_multi_scalar(X, y; args...,
        hof_donor_policy=:unknown)
    @test_throws ArgumentError search_multi_scalar(X, y; args...,
        hof_donor_policy="global_pareto")
end

@testitem "Multi-island frequency switches preserve defaults and tunnel to islands" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=2, max_evaluations=12,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_probability=0.0, mutation_probability=1.0,
        crossover_probability=0.0, constant_refinement_iterations=0,
        constant_refinement_probability=0.0,
        max_constant_refinement_evaluations=0, evolution_strategy=:regularized)
    function run_with_frequency(log; kwargs...)
        objective = expression -> begin
            push!(log, repr(expression))
            Float64(scalar_complexity(expression))
        end
        return search_multi_scalar(X, y; config, rng=MersenneTwister(8000),
            initial_population=[scalar_input(1)], objective, island_count=2,
            rounds=1, migrant_count=1, top_migrant_count=0,
            ncycles_per_iteration=1, kwargs...)
    end

    default_log = String[]
    default_result = run_with_frequency(default_log)
    enabled_log = String[]
    enabled_result = run_with_frequency(enabled_log;
        use_frequency=true, use_frequency_in_tournament=true)
    @test enabled_log == default_log
    @test repr(enabled_result) == repr(default_result)

    enabled_vector_log = String[]
    enabled_vector_result = run_with_frequency(enabled_vector_log;
        use_frequency=fill(true, 2), use_frequency_in_tournament=true)
    @test enabled_vector_log == default_log
    @test repr(enabled_vector_result) == repr(default_result)

    disabled_log = String[]
    disabled_result = run_with_frequency(disabled_log;
        use_frequency=false, use_frequency_in_tournament=false)
    disabled_vector_log = String[]
    disabled_vector_result = run_with_frequency(disabled_vector_log;
        use_frequency=fill(false, 2), use_frequency_in_tournament=false)
    @test disabled_vector_log == disabled_log
    @test repr(disabled_vector_result) == repr(disabled_result)
    disabled = initialize_multi_scalar(X, y; config,
        rng=MersenneTwister(8000), initial_population=[scalar_input(1)],
        objective=scalar_complexity, island_count=2, rounds=1,
        migrant_count=1, top_migrant_count=0, ncycles_per_iteration=1,
        frequency_window=100, use_frequency=false,
        use_frequency_in_tournament=false)
    @test !multi_scalar_progress(disabled).completed
    @test default_result.evaluations == config.max_evaluations
    shared_frequency = disabled.progress.shared_frequency
    before = copy(shared_frequency.frequencies)
    @test all(state -> state.frequency === shared_frequency,
        disabled.progress.states)
    @test all(state -> !state.flow.use_frequency &&
        !state.flow.use_frequency_in_tournament, disabled.progress.states)
    advance_multi_scalar!(disabled; max_rounds=1)
    after = multi_scalar_result(disabled)
    @test after.diagnostics.rounds_completed == 1
    @test repr(after) == repr(disabled_result)
    @test sum(shared_frequency.frequencies) > sum(before)
    @test all(state -> state.frequency === shared_frequency,
        disabled.progress.states)

    acceptance_flags = Bool[true, false]
    scored = Ref(0)
    counting_objective = expression -> begin
        scored[] += 1
        Float64(scalar_complexity(expression))
    end
    heterogeneous = initialize_multi_scalar(X, y; config,
        rng=MersenneTwister(42), initial_population=[scalar_input(1)],
        objective=counting_objective, island_count=2, rounds=1,
        migrant_count=1, top_migrant_count=0, ncycles_per_iteration=1,
        use_frequency=acceptance_flags)
    @test scored[] > 0
    states = heterogeneous.progress.states
    acceptance_flags .= Bool[false, true]
    @test [state.flow.use_frequency for state in states] == [true, false]
    @test all(state -> state.flow.use_frequency isa Bool, states)
    @test all(state -> state.frequency === heterogeneous.progress.shared_frequency,
        states)

    rejected_scores = Ref(0)
    rejected_objective = expression -> begin
        rejected_scores[] += 1
        Float64(scalar_complexity(expression))
    end
    @test_throws ArgumentError initialize_multi_scalar(X, y; config,
        rng=MersenneTwister(555), objective=rejected_objective,
        island_count=2, use_frequency=Bool[true])
    @test rejected_scores[] == 0
end

@testitem "Multi-island sessions stop when all islands are terminal at initialization" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    common = (; population_size=2, max_complexity=6, initial_depth=0,
        tournament_size=1, elite_count=1, unary_operators=(:square,),
        binary_operators=(:+,), constant_probability=0.0,
        mutation_probability=1.0, crossover_probability=0.0,
        constant_refinement_iterations=0,
        max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    initial = [scalar_input(1)]
    rng_preview(session) = [rand(deepcopy(rng), UInt64, 16)
        for rng in session.progress.island_rngs]

    structural_config = ScalarSearchConfig(; common..., max_evaluations=4)
    structural_observed = Int[]
    structural = initialize_multi_scalar(X, y; config=structural_config,
        rng=MersenneTwister(8000), initial_population=initial,
        objective=scalar_complexity, island_count=2, rounds=3,
        migrant_count=1,
        round_observer=(round, snapshot) -> push!(structural_observed, round))
    structural_before = multi_scalar_result(structural)
    structural_rng = rng_preview(structural)
    structural_calls = structural.progress.objective_calls[]
    @test multi_scalar_progress(structural).completed
    @test structural_before.diagnostics.rounds_completed == 0
    @test structural_before.termination == :evaluation_cap
    advance_multi_scalar!(structural; max_rounds=3)
    @test repr(multi_scalar_result(structural)) == repr(structural_before)
    @test rng_preview(structural) == structural_rng
    @test structural.progress.objective_calls[] == structural_calls
    @test isempty(structural_observed)
    @test multi_scalar_result(structural).diagnostics.migration_events == 0

    solved_config = ScalarSearchConfig(; common..., max_evaluations=8)
    zero_objective = expression -> 0.0
    solved_observed = Int[]
    solved = initialize_multi_scalar(X, y; config=solved_config,
        rng=MersenneTwister(42), initial_population=initial,
        objective=zero_objective, stop_loss=0.0, island_count=2, rounds=3,
        migrant_count=1,
        round_observer=(round, snapshot) -> push!(solved_observed, round))
    solved_before = multi_scalar_result(solved)
    @test multi_scalar_progress(solved).completed
    @test solved_before.diagnostics.rounds_completed == 0
    @test solved_before.termination == :solved
    advance_multi_scalar!(solved; max_rounds=3)
    @test repr(multi_scalar_result(solved)) == repr(solved_before)
    @test isempty(solved_observed)
    @test multi_scalar_result(solved).diagnostics.migration_events == 0

    total_config = ScalarSearchConfig(; common..., max_evaluations=8,
        max_total_evaluations=4)
    total = initialize_multi_scalar(X, y; config=total_config,
        rng=MersenneTwister(555), initial_population=initial,
        objective=zero_objective, stop_loss=0.0, island_count=2, rounds=3,
        migrant_count=1)
    total_result = multi_scalar_result(total)
    @test multi_scalar_progress(total).completed
    @test total_result.diagnostics.rounds_completed == 0
    @test total_result.diagnostics.remaining_total_evaluations == 0
    @test total_result.termination == :total_evaluation_cap
end

@testitem "Multi-island sessions stop after the all-terminal round boundary" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = vec(X)
    # Seven structural scores split as 4/3: one island naturally finishes
    # after the first round while the other still has a real score to spend.
    config = ScalarSearchConfig(population_size=2, max_evaluations=7,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_probability=0.0, mutation_probability=1.0,
        crossover_probability=0.0, constant_refinement_iterations=0,
        max_constant_refinement_evaluations=0, evolution_strategy=:regularized)
    function make_session(seed, objective_log, observed_rounds)
        objective = expression -> begin
            push!(objective_log, repr(expression))
            return Float64(scalar_complexity(expression))
        end
        observer = (round, snapshot) ->
            push!(observed_rounds, (round=round, state=snapshot()))
        return initialize_multi_scalar(X, y; config,
            rng=MersenneTwister(seed), initial_population=[scalar_input(1)],
            objective, stop_loss=0.0, island_count=2, rounds=4,
            migrant_count=1, top_migrant_count=0,
            ncycles_per_iteration=1, round_observer=observer)
    end
    rng_preview(session) = [rand(deepcopy(rng), UInt64, 16)
        for rng in session.progress.island_rngs]

    whole_log = String[]
    whole_rounds = NamedTuple[]
    whole = make_session(73, whole_log, whole_rounds)
    @test !multi_scalar_progress(whole).completed
    advance_multi_scalar!(whole; max_rounds=4)
    whole_result = multi_scalar_result(whole)
    @test multi_scalar_progress(whole).completed
    @test whole_result.diagnostics.rounds_completed == 2
    @test whole_result.termination == :evaluation_cap
    @test [row.round for row in whole_rounds] == [1, 2]
    @test whole_result.diagnostics.per_island[1].structural_evaluations == 4
    @test whole_result.diagnostics.per_island[2].structural_evaluations == 3
    @test whole_result.diagnostics.hof_migrations > 0

    split_log = String[]
    split_rounds = NamedTuple[]
    split = make_session(73, split_log, split_rounds)
    advance_multi_scalar!(split; max_rounds=1)
    partial = multi_scalar_result(split)
    @test !multi_scalar_progress(split).completed
    @test partial.diagnostics.rounds_completed == 1
    @test partial.diagnostics.hof_migrations > 0
    @test any(state -> state.evaluations == state.config.max_evaluations,
        split.progress.states)
    @test any(state -> state.evaluations < state.config.max_evaluations &&
        state.best.loss > 0.0, split.progress.states)
    first_migrations = partial.diagnostics.hof_migrations
    advance_multi_scalar!(split; max_rounds=1)
    split_result = multi_scalar_result(split)
    @test split_result.diagnostics.rounds_completed == 2
    @test multi_scalar_progress(split).completed
    @test [row.round for row in split_rounds] == [1, 2]
    @test split_result.diagnostics.hof_migrations == first_migrations
    @test repr(split_result) == repr(whole_result)
    @test split_log == whole_log
    @test repr(split_rounds) == repr(whole_rounds)
    @test rng_preview(split) == rng_preview(whole)

    completed_rng = rng_preview(split)
    completed_calls = length(split_log)
    completed_rounds = length(split_rounds)
    advance_multi_scalar!(split; max_rounds=4)
    @test repr(multi_scalar_result(split)) == repr(split_result)
    @test rng_preview(split) == completed_rng
    @test length(split_log) == completed_calls
    @test length(split_rounds) == completed_rounds
end

@testitem "Global Pareto donors merge archives before dominance filtering" setup=[MultiIslandSearchImports] begin
    using LambdaRegression: ScalarSearch
    candidate(complexity, loss, serial) =
        (; value=(; complexity), loss, serial)
    states = [
        (archive=Dict(
            1 => candidate(1, 10.0, 9),
            3 => candidate(3, 8.0, 8),
            4 => candidate(4, 6.0, 7),
            7 => candidate(7, Inf, 6)),),
        (archive=Dict(
            1 => candidate(1, 10.0, 1),
            2 => candidate(2, 9.0, 2),
            3 => candidate(3, 5.0, 3),
            5 => candidate(5, 5.0, 4),
            6 => candidate(6, 4.0, 5),
            8 => candidate(8, NaN, 6),
            9 => candidate(9, -Inf, 7)),),
    ]
    donors = ScalarSearch._global_pareto_donor_sources(states)
    @test [(entry.island, entry.candidate.value.complexity,
        entry.candidate.loss, entry.candidate.serial) for entry in donors] == [
            (1, 1, 10.0, 9), (2, 2, 9.0, 2),
            (2, 3, 5.0, 3), (2, 6, 4.0, 5)]
    @test isempty(ScalarSearch._global_pareto_donor_sources([
        (archive=Dict(1 => candidate(1, Inf, 1)),)]))
end

@testitem "Global Pareto archive migration includes self and ignores donor limit" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_complexity=1, initial_depth=0, tournament_size=1, elite_count=1,
        constant_refinement_iterations=0, evolution_strategy=:regularized)
    make_session(policy; migrant_count=2, round_observer=nothing) = initialize_multi_scalar(X, y;
        config, rng=MersenneTwister(1234), initial_population=[scalar_input(1)],
        objective=scalar_complexity, island_count=1, rounds=2,
        migrant_count, top_migrant_count=0, donor_limit=0,
        hof_donor_policy=policy, round_observer)

    local_session = make_session(:local_best)
    advance_multi_scalar!(local_session; max_rounds=2)
    @test multi_scalar_result(local_session).diagnostics.hof_migrations == 0

    pre_migration_clock = Ref(0)
    global_session = make_session(:global_pareto;
        round_observer=(_, snapshot) ->
            (pre_migration_clock[] = only(snapshot().islands).birth_clock))
    before = multi_scalar_result(global_session)
    advance_multi_scalar!(global_session; max_rounds=1)
    after = multi_scalar_result(global_session)
    @test after.diagnostics.hof_donor_policy === :global_pareto
    @test after.diagnostics.hof_migrations == 1 # Only one frontier member.
    @test after.diagnostics.top_migrations == 0
    # This one-node language skips unchanged proposals; migration itself scores nothing.
    @test after.diagnostics.objective_calls == before.diagnostics.objective_calls
    @test before.evaluations == 2
    @test after.evaluations == 2
    @test !multi_scalar_progress(global_session).completed
    @test global_session.progress.states[1].birth_clock == pre_migration_clock[] + 1
    @test any(member.born == pre_migration_clock[] + 1
        for member in global_session.progress.states[1].population)

    disabled = make_session(:global_pareto; migrant_count=0)
    advance_multi_scalar!(disabled; max_rounds=2)
    @test multi_scalar_result(disabled).diagnostics.hof_migrations == 0
end

@testitem "Global Pareto multi-island sessions preserve split parity" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=2, max_evaluations=12,
        max_complexity=4, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_refinement_iterations=0, evolution_strategy=:regularized)
    make_session(log) = initialize_multi_scalar(X, y;
        config, rng=MersenneTwister(8000), initial_population=[scalar_input(1)],
        objective=expression -> begin
            push!(log, repr(expression))
            Float64(scalar_complexity(expression))
        end,
        island_count=2, rounds=3, migrant_count=2, top_migrant_count=1,
        hof_donor_policy=:global_pareto)
    whole_log = String[]
    whole = make_session(whole_log)
    advance_multi_scalar!(whole; max_rounds=3)
    split_log = String[]
    split = make_session(split_log)
    advance_multi_scalar!(split; max_rounds=1)
    advance_multi_scalar!(split; max_rounds=2)
    @test repr(multi_scalar_result(split)) == repr(multi_scalar_result(whole))
    @test split_log == whole_log
    @test [rand(deepcopy(rng), UInt64, 16) for rng in split.progress.island_rngs] ==
        [rand(deepcopy(rng), UInt64, 16) for rng in whole.progress.island_rngs]
end

@testitem "Multi-island observation leaves search and ownership unchanged" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=3, max_evaluations=40,
        max_complexity=8, initial_depth=1, tournament_size=2, elite_count=1,
        constant_refinement_iterations=0, evolution_strategy=:regularized)
    run(; kwargs...) = search_multi_scalar(X, y; config, rng=MersenneTwister(2000),
        island_count=2, rounds=3, migrant_count=1, kwargs...)
    baseline = run()
    rounds = Int[]
    function observe(round, capture)
        push!(rounds, round)
        round == 2 || return
        snapshot = capture()
        @test length(snapshot.islands) == 2
        @test all(length(row.population) == 3 for row in snapshot.islands)
        @test all(member.born <= row.birth_clock
            for row in snapshot.islands for member in row.population)
        # The observer owns detached collections; no mutable search state escapes.
        fill!(snapshot.frequencies, 0.0)
        empty!(snapshot.islands[1].population)
        empty!(snapshot.islands[1].archive)
    end
    observed = run(round_observer=observe)
    @test rounds == [1, 2, 3]
    @test observed.best_loss == baseline.best_loss
    @test repr(observed.best) == repr(baseline.best)
    @test observed.diagnostics == baseline.diagnostics
    @test repr(observed.complexity_archive) == repr(baseline.complexity_archive)
end

@testitem "Multi-island wrapper preserves the pre-resume coordinator outcome" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=2, max_evaluations=8,
        max_complexity=6, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_probability=0.0, mutation_probability=1.0,
        crossover_probability=0.0, constant_refinement_iterations=0,
        evolution_strategy=:regularized)
    objective_log = String[]
    objective = expression -> begin
        push!(objective_log, repr(expression))
        Float64(scalar_complexity(expression))
    end
    observed_rounds = NamedTuple[]
    observer = (round, snapshot) ->
        push!(observed_rounds, (round=round, state=snapshot()))
    result = search_multi_scalar(X, y; config, rng=MersenneTwister(2000),
        initial_population=[scalar_input(1)], objective, island_count=2,
        rounds=2, migrant_count=1, top_migrant_count=1,
        round_observer=observer)

    input_repr = repr(scalar_input(1))
    constant_repr = repr(scalar_constant(0.0))
    plus_expr = scalar_binary(:+, scalar_input(1), scalar_input(1))
    plus_repr = repr(plus_expr)
    nested_repr = repr(scalar_binary(:+, plus_expr, scalar_input(1)))
    @test repr(result.best) == input_repr
    @test result.best_loss == 1.0
    @test result.evaluations == 8
    @test result.invalid_evaluations == 0
    @test result.constant_refinement_evaluations == 0
    @test result.generations == 2
    @test result.termination == :evaluation_cap
    @test [(repr(item.expr), item.loss, item.complexity)
        for item in result.complexity_archive] == [
            (input_repr, 1.0, 1), (plus_repr, 3.0, 3),
            (nested_repr, 5.0, 5)]
    @test [(repr(item.expr), item.loss, item.complexity)
        for item in result.leaders] == [(input_repr, 1.0, 1), (plus_repr, 3.0, 3)]
    @test result.variation == (attempts=4, changed=4,
        accepted_offspring=3, rejected_offspring=1, skipped_variations=0,
        cycle_refinement_calls=0, offspring_refinement_calls=0,
        total_evaluation_budget=nothing, remaining_total_evaluations=nothing,
        final_population_size=4)
    diagnostics = result.diagnostics
    @test diagnostics.hof_donor_policy === :local_best
    @test diagnostics.objective_calls == 10 == length(objective_log)
    @test diagnostics.baseline_objective_calls == 2
    @test diagnostics.rounds == diagnostics.rounds_completed == 2
    @test diagnostics.migration_events == 4
    @test diagnostics.hof_migrations == 2
    @test diagnostics.top_migrations == 2
    @test [(row.structural_evaluations, row.invalid_evaluations,
        row.constant_refinement_evaluations, row.generations,
        row.hof_migrations, row.top_migrations, row.offspring_attempts,
        row.accepted_offspring, row.rejected_offspring,
        row.rotation_copy_attempts, row.rotation_copy_evaluations,
        row.rotation_copy_accepted, row.changed_variations)
        for row in diagnostics.per_island] == [
            (4, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0, 2),
            (4, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 2)]
    @test [(row.round, row.island, row.configured_structural_budget,
        row.structural_evaluations, row.invalid_evaluations,
        row.constant_refinement_evaluations, row.generations,
        row.underfilled, row.attempt_limited, row.objective_callbacks)
        for row in diagnostics.round_ledger] == [
            (1, 1, 1, 1, 0, 0, 1, false, false, 1),
            (1, 2, 1, 1, 0, 0, 1, false, false, 1),
            (2, 1, 1, 1, 0, 0, 0, false, false, 1),
            (2, 2, 1, 1, 0, 0, 0, false, false, 1)]
    @test objective_log == [constant_repr, input_repr, input_repr,
        constant_repr, input_repr, input_repr, plus_repr, plus_repr,
        nested_repr, input_repr]

    @test [row.round for row in observed_rounds] == [1, 2]
    @test all(row.state.frequencies == fill(1 / 6, 6)
        for row in observed_rounds)
    @test [[island.evaluations for island in row.state.islands]
        for row in observed_rounds] == [[3, 3], [4, 4]]
    @test [[island.birth_clock for island in row.state.islands]
        for row in observed_rounds] == [[3, 3], [5, 6]]
    @test [[(member.serial, member.born)
        for island in row.state.islands for member in island.population]
        for row in observed_rounds] == [
            [(3, 3), (2, 2), (3, 3), (2, 2)],
            [(3, 3), (5, 5), (6, 6), (5, 5)]]
    @test [[repr(member.expr) for island in row.state.islands
        for member in island.population] for row in observed_rounds] == [
            [plus_repr, input_repr, plus_repr, input_repr],
            [plus_repr, plus_repr, input_repr, plus_repr]]
    @test [[member.serial for island in row.state.islands
        for member in island.archive] for row in observed_rounds] == [
            [1, 3, 1, 3], [1, 3, 6, 1, 3]]
end

@testitem "Multi-island sessions resume at exact round boundaries" setup=[MultiIslandSearchImports] begin
    X = zeros(0, 4)
    y = fill(3.0, 4)
    config = ScalarSearchConfig(population_size=2, max_evaluations=8,
        max_complexity=1, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(), constant_probability=0.0,
        mutation_probability=0.0, crossover_probability=1.0,
        constant_refinement_iterations=8, constant_refinement_elites=1,
        constant_refinement_interval=1, constant_refinement_start_generation=1,
        constant_refinement_probability=1.0,
        max_constant_refinement_evaluations=8,
        max_total_evaluations=12,
        constant_refinement_optimizer=:nelder_mead,
        evolution_strategy=:regularized)
    initial = [scalar_constant(0.0), scalar_constant(0.0)]
    make_session(seed, objective_log, observed_rounds) = begin
        objective = expression -> begin
            push!(objective_log, repr(expression))
            prediction = evaluate_scalar_reference(expression, X)
            return sum(abs2, prediction .- y) / length(y)
        end
        observer = (round, snapshot) ->
            push!(observed_rounds, (round=round, state=snapshot()))
        initialize_multi_scalar(X, y; config, rng=MersenneTwister(seed),
            initial_population=initial, objective, stop_loss=nothing,
            island_count=2, rounds=4, migrant_count=1, top_migrant_count=1,
            ncycles_per_iteration=10, round_observer=observer)
    end
    rng_preview(session) = [rand(deepcopy(rng), UInt64, 16)
        for rng in session.progress.island_rngs]

    whole_log = String[]
    whole_rounds = NamedTuple[]
    whole = make_session(211, whole_log, whole_rounds)
    @test whole isa MultiScalarSession
    advance_multi_scalar!(whole; max_rounds=10)
    whole_result = multi_scalar_result(whole)
    @test whole_result.diagnostics.rounds == 4
    @test whole_result.diagnostics.rounds_completed == 2
    @test whole_result.termination == :total_evaluation_cap
    @test whole_result.diagnostics.remaining_total_evaluations == 0

    split_log = String[]
    split_rounds = NamedTuple[]
    split = make_session(211, split_log, split_rounds)
    initial_preview = rng_preview(split)
    initial_log_length = length(split_log)
    @test_throws ArgumentError advance_multi_scalar!(split; max_rounds=-1)
    advance_multi_scalar!(split; max_rounds=0)
    initial_result = multi_scalar_result(split)
    @test initial_result.diagnostics.rounds_completed == 0
    @test initial_result.termination == :paused
    @test isempty(initial_result.diagnostics.round_ledger)
    @test rng_preview(split) == initial_preview
    @test length(split_log) == initial_log_length

    advance_multi_scalar!(split; max_rounds=1)
    first_round_result = multi_scalar_result(split)
    first_round_ledger = repr(first_round_result.diagnostics.round_ledger)
    @test first_round_result.diagnostics.rounds_completed == 1
    @test first_round_result.termination == :paused
    @test length(first_round_result.diagnostics.round_ledger) == 2
    @test first_round_result.constant_refinement_evaluations == 0
    projection_rng = rng_preview(split)
    projection_log_length = length(split_log)
    detached = multi_scalar_result(split)
    empty!(detached.diagnostics.round_ledger)
    @test rng_preview(split) == projection_rng
    @test length(split_log) == projection_log_length
    @test length(multi_scalar_result(split).diagnostics.round_ledger) == 2

    advance_multi_scalar!(split; max_rounds=1)
    split_result = multi_scalar_result(split)
    @test first_round_result.diagnostics.rounds_completed == 1
    @test repr(first_round_result.diagnostics.round_ledger) == first_round_ledger
    @test split_result.diagnostics.rounds_completed == 2
    @test split_result.termination == :total_evaluation_cap
    @test split_result.constant_refinement_evaluations > 0
    @test split_result.diagnostics.cycle_refinement_calls > 0
    @test split_result.diagnostics.offspring_refinement_calls == 0
    @test repr(split_result) == repr(whole_result)
    @test split_log == whole_log
    @test repr(split_rounds) == repr(whole_rounds)
    @test rng_preview(split) == rng_preview(whole)
    @test split_result.diagnostics.objective_calls == length(split_log)
    @test split_result.diagnostics.initialization_count == 2
    @test split_result.diagnostics.baseline_objective_calls == 2
    @test split_result.diagnostics.objective_calls ==
        split_result.diagnostics.structural_objective_calls +
        split_result.diagnostics.constant_refinement_objective_calls +
        split_result.diagnostics.baseline_objective_calls

    completed_preview = rng_preview(split)
    completed_log_length = length(split_log)
    advance_multi_scalar!(split; max_rounds=1)
    @test rng_preview(split) == completed_preview
    @test length(split_log) == completed_log_length
end

@testitem "Multi-island split parity with 20 islands and population 27" setup=[MultiIslandSearchImports] begin
    X = reshape(Float64[-1, 0, 1], 1, :)
    y = vec(X)
    config = ScalarSearchConfig(population_size=27, max_evaluations=580,
        max_complexity=3, initial_depth=0, tournament_size=15, elite_count=4,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_probability=0.0, mutation_probability=0.65,
        crossover_probability=0.35, constant_refinement_iterations=0,
        evolution_strategy=:regularized)
    make_session(log) = begin
        objective = expression -> begin
            push!(log, repr(expression))
            Float64(scalar_complexity(expression))
        end
        initialize_multi_scalar(X, y; config, rng=MersenneTwister(1),
            objective, island_count=20, rounds=2, migrant_count=1,
            top_migrant_count=1, ncycles_per_iteration=1)
    end
    rng_preview(session) = [rand(deepcopy(rng), UInt64, 16)
        for rng in session.progress.island_rngs]

    whole_log = String[]
    whole = make_session(whole_log)
    advance_multi_scalar!(whole; max_rounds=2)
    whole_result = multi_scalar_result(whole)

    split_log = String[]
    split = make_session(split_log)
    advance_multi_scalar!(split; max_rounds=1)
    first_result = multi_scalar_result(split)
    @test first_result.diagnostics.rounds_completed == 1
    @test first_result.termination == :paused
    advance_multi_scalar!(split; max_rounds=1)
    split_result = multi_scalar_result(split)

    @test split_result.diagnostics.rounds_completed == 2
    @test split_result.diagnostics.structural_evaluations <= 580
    @test length(split_result.diagnostics.per_island) == 20
    @test repr(split_result) == repr(whole_result)
    @test split_log == whole_log
    @test rng_preview(split) == rng_preview(whole)
end

@testitem "multi-island accepts finite negative losses and thresholds" setup=[MultiIslandSearchImports] begin
    config = ScalarSearchConfig(evolution_strategy=:regularized,
        population_size=2, max_evaluations=2, max_complexity=1,
        initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(), binary_operators=(:+,),
        constant_refinement_iterations=0,
        max_constant_refinement_evaluations=0)
    X = zeros(Float64, 0, 2)
    y = zeros(2)
    initial = [scalar_constant(0.0), scalar_constant(1.0)]
    objective = _ -> -2.0
    make_session(; stop_loss=nothing) = initialize_multi_scalar(
        X, y; config, rng=MersenneTwister(1111), initial_population=initial,
        objective, stop_loss, island_count=1, rounds=1, migrant_count=0)

    default_session = make_session()
    negative_threshold_session = make_session(stop_loss=-1.0)
    state = only(negative_threshold_session.progress.states)

    @test default_session.stop_loss === nothing
    @test negative_threshold_session.stop_loss == -1.0
    @test state.invalid_evaluations == 0
    @test state.best.loss == -2.0
    @test state.loss_normalization == 2.0
end
