"""Persistent regularized island state for scalar and domain-backed search."""

struct _NodeRegularizedFlow
    stop_loss::Union{Nothing,Float64}
    annealing::Bool
    temperature::Float64
    alpha::Float64
    ncycles_per_iteration::Int
    use_frequency::Bool
    hof_migration_fraction::Float64
    tournament_selection_p::Float64
    use_frequency_in_tournament::Bool
    adaptive_parsimony_scaling::Float64
end

mutable struct _NodeRegularizedState{R<:AbstractRNG,F,V,D}
    config::ScalarSearchConfig
    rng::R
    feature_count::Int
    score::F
    flow::_NodeRegularizedFlow
    population::Vector{_NodeRegularizedMember{V}}
    tournament_workspace::_NodeTournamentWorkspace
    archive::Dict{Int,_NodeSearchCandidate{V}}
    frequency::_RegularizedFrequency
    domain::D
    evaluations::Int
    invalid_evaluations::Int
    constant_refinement_evaluations::Int
    constant_refinement_invalid_evaluations::Int
    serial::Int
    birth_clock::Int
    best::_NodeSearchCandidate{V}
    improvements::Vector{NamedTuple}
    mutation_attempts::Int
    crossover_attempts::Int
    composition_attempts::Int
    changed_variations::Int
    accepted_offspring::Int
    rejected_offspring::Int
    skipped_variations::Int
    oldest_replacements::Int
    hof_migrations::Int
    mutation_kind_counts::Dict{Symbol,Int}
    rotation_copy_evaluations::Int
    rotation_copy_accepted::Int
    offspring_attempts::Int
    loss_normalization::Float64
    cycle_refinement_calls::Int
    offspring_refinement_calls::Int
end

_node_state_total_remaining(state::_NodeRegularizedState) =
    state.config.max_total_evaluations === nothing ? typemax(Int) :
    state.config.max_total_evaluations - state.evaluations -
        state.constant_refinement_evaluations

function _node_state_record!(state::_NodeRegularizedState,
        candidate::_NodeSearchCandidate; source::Union{Nothing,Symbol}=nothing)
    isfinite(candidate.loss) || return candidate
    complexity = candidate.value.complexity
    incumbent = get(state.archive, complexity, nothing)
    if incumbent === nothing || _node_candidate_key(candidate) <
            _node_candidate_key(incumbent)
        state.archive[complexity] = candidate
    end
    if _node_candidate_key(candidate) < _node_candidate_key(state.best)
        state.best = candidate
        entry = (evaluation=state.evaluations,
            generation=state.offspring_attempts,
            loss=candidate.loss, complexity)
        push!(state.improvements,
            source === nothing ? entry : merge(entry, (; source)))
    end
    return candidate
end

function _node_state_score_candidate!(state::_NodeRegularizedState,
        value, evolved::Bool, evaluation_limit::Int)
    state.evaluations >= evaluation_limit && return nothing
    _node_state_total_remaining(state) <= 0 && return nothing
    loss, invalid = state.score(value)
    state.evaluations += 1
    state.invalid_evaluations += invalid
    state.serial += 1
    return _NodeSearchCandidate(value, loss, state.serial, evolved)
end

function _node_state_refine_candidate!(state::_NodeRegularizedState,
        candidate::_NodeSearchCandidate, generation::Int, timing::Symbol)
    config = state.config
    config.constant_refinement_iterations == 0 && return candidate
    generation < config.constant_refinement_start_generation && return candidate
    remaining = min(
        config.max_constant_refinement_evaluations -
            state.constant_refinement_evaluations,
        _node_state_total_remaining(state),
    )
    remaining <= 0 && return candidate
    _regularized_has_constants(state.domain, candidate.value) || return candidate
    refined, refined_loss, count, invalid = _regularized_refine_value(
        state.domain, candidate.value, candidate.loss, state.score,
        config, remaining; rng=state.rng, restarts=2)
    state.constant_refinement_evaluations += count
    state.constant_refinement_invalid_evaluations += invalid
    if count > 0
        if timing === :cycle
            state.cycle_refinement_calls += 1
        elseif timing === :offspring
            state.offspring_refinement_calls += 1
        else
            throw(ArgumentError("unknown constant refinement timing: $timing"))
        end
    end
    refined_loss < candidate.loss || return candidate
    return _NodeSearchCandidate(refined, refined_loss,
        candidate.serial, candidate.evolved)
end

function _node_state_refine!(state::_NodeRegularizedState,
        member::_NodeRegularizedMember, generation::Int)
    previous = member.candidate
    refined = _node_state_refine_candidate!(
        state, previous, generation, :cycle)
    refined === previous && return member
    member.candidate = refined
    if state.config.constant_refinement_resets_age
        state.birth_clock += 1
        member.born = state.birth_clock
    end
    _node_state_record!(state, member.candidate; source=:constant_refinement)
    _regularized_observe(state.domain, state, :constant_refinement,
        previous, refined, true)
    return member
end

function _node_state_refine_offspring!(state::_NodeRegularizedState,
        candidate::_NodeSearchCandidate)
    config = state.config
    config.constant_refinement_timing === :offspring || return candidate
    config.constant_refinement_probability > 0 || return candidate
    isfinite(candidate.loss) || return candidate
    generation = cld(state.offspring_attempts,
        _regularized_cycle_temperature(state.offspring_attempts,
            config.population_size, config.tournament_size,
            state.flow.ncycles_per_iteration)[2])
    generation >= config.constant_refinement_start_generation || return candidate
    config.constant_refinement_iterations > 0 || return candidate
    config.max_constant_refinement_evaluations -
        state.constant_refinement_evaluations > 0 || return candidate
    _node_state_total_remaining(state) > 0 || return candidate
    _regularized_has_constants(state.domain, candidate.value) || return candidate
    rand(state.rng) < config.constant_refinement_probability || return candidate
    refined = _node_state_refine_candidate!(
        state, candidate, generation, :offspring)
    refined === candidate || _regularized_observe(state.domain, state,
        :constant_refinement, candidate, refined, true)
    return refined
end

function _node_state_maintain_cycle!(state::_NodeRegularizedState,
        evaluation_limit::Int)
    _node_observe_population!(state.frequency, state.population)
    generation = cld(state.offspring_attempts,
        _regularized_cycle_temperature(max(1, state.offspring_attempts),
            state.config.population_size, state.config.tournament_size,
            state.flow.ncycles_per_iteration)[2])
    for member_index in eachindex(state.population)
        member = state.population[member_index]
        state.config.max_total_evaluations !== nothing &&
            _node_state_total_remaining(state) <= 0 && return state
        simplified = _regularized_simplify_value(
            state.domain, member.candidate.value, state.rng)
        if simplified !== nothing
            candidate = _node_state_score_candidate!(
                state, simplified, member.candidate.evolved, evaluation_limit)
            if candidate !== nothing
                _node_state_record!(state, candidate; source=:simplification)
                parent = member.candidate
                if isfinite(candidate.loss) && candidate.loss <= parent.loss
                    state.population[member_index].candidate = candidate
                    _regularized_observe(state.domain, state, :simplification,
                        parent, candidate, true)
                else
                    _regularized_observe(state.domain, state, :simplification,
                        parent, candidate, false)
                end
            end
        end
        state.config.constant_refinement_timing === :cycle || continue
        state.config.constant_refinement_probability > 0 || continue
        rand(state.rng) < state.config.constant_refinement_probability || continue
        _node_state_refine!(state, member, generation)
    end
    migrants = _node_pareto_candidates(state.archive)
    count = min(length(state.population), length(migrants),
        _regularized_poisson(state.rng,
            length(state.population) * state.flow.hof_migration_fraction))
    for _ in 1:count
        _node_inject_candidate!(state, rand(state.rng, migrants))
    end
    return state
end

"""Copy one same-objective training candidate into a persistent island.

The source loss is reusable only because the coordinator anchors all islands
to the same training data/objective. The copied value receives fresh age and
serial metadata. An invalid destination index fails at the vector boundary.
"""
function _node_inject_candidate!(state::_NodeRegularizedState,
        source::_NodeSearchCandidate; destination=nothing)
    state.serial += 1
    state.birth_clock += 1
    copied_value = _regularized_copy_value(state.domain, source.value)
    copied = _NodeSearchCandidate(
        copied_value, source.loss, state.serial, true)
    index = destination === nothing ?
        rand(state.rng, eachindex(state.population)) : destination
    parent = state.population[index].candidate
    state.population[index] = _NodeRegularizedMember(copied, state.birth_clock)
    state.hof_migrations += 1
    _regularized_observe(state.domain, state, :migration, parent, copied, true)
    return copied
end

function _node_state_accept!(state::_NodeRegularizedState,
        parent::_NodeSearchCandidate, child::_NodeSearchCandidate)
    old_complexity = parent.value.complexity
    new_complexity = child.value.complexity
    old_frequency, new_frequency = state.flow.use_frequency ?
        _regularized_frequency_ratio(
            state.frequency, old_complexity, new_complexity) : (1.0, 1.0)
    cycle_temperature, _ = _regularized_cycle_temperature(
        state.offspring_attempts, state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    parent_cost = _regularized_cost(parent.loss, old_complexity,
        state.loss_normalization, state.config.parsimony_coefficient)
    child_cost = _regularized_cost(child.loss, new_complexity,
        state.loss_normalization, state.config.parsimony_coefficient)
    probability = _regularized_acceptance_probability(parent_cost, child_cost;
        annealing=state.flow.annealing,
        temperature=state.flow.temperature * cycle_temperature,
        alpha=state.flow.alpha, old_frequency, new_frequency)
    return rand(state.rng) < probability
end

function _node_state_replace_offspring!(state::_NodeRegularizedState,
        parent::_NodeSearchCandidate, child::_NodeSearchCandidate)
    oldest = _node_oldest_index(state.population)
    index = _regularized_replacement_index(
        state.domain, state, parent, child)
    if index === nothing
        state.rejected_offspring += 1
        return false
    end
    state.birth_clock += 1
    state.population[index] =
        _NodeRegularizedMember(child, state.birth_clock)
    state.accepted_offspring += 1
    index == oldest && (state.oldest_replacements += 1)
    _node_state_record!(state, child)
    return true
end

function _node_state_offspring_variation!(state::_NodeRegularizedState,
        evaluation_limit::Int)
    tournament() = _node_tournament(state.rng, state.population,
        state.tournament_workspace, state.config, state.frequency;
        loss_normalization=state.loss_normalization,
        selection_p=state.flow.tournament_selection_p,
        use_frequency_in_tournament=state.flow.use_frequency_in_tournament,
        adaptive_parsimony_scaling=state.flow.adaptive_parsimony_scaling)
    first_parent = tournament()
    second_draw = _node_draw_tournament!(state.rng, state.population,
        state.tournament_workspace, state.config)
    rank_second() = _node_rank_tournament!(second_draw, state.population,
        state.tournament_workspace, state.config, state.frequency;
        loss_normalization=state.loss_normalization,
        selection_p=state.flow.tournament_selection_p,
        use_frequency_in_tournament=state.flow.use_frequency_in_tournament,
        adaptive_parsimony_scaling=state.flow.adaptive_parsimony_scaling)
    # Always consume both tournaments' random draws. Native one-parent mutation
    # can omit the second ranking; other domains retain their donor semantics.
    deferred_second = state.domain === nothing
    second_parent = deferred_second ? first_parent : rank_second()
    draw = rand(state.rng)
    if draw >= state.config.mutation_probability &&
            draw < state.config.mutation_probability +
                state.config.crossover_probability
        state.crossover_attempts += 1
        min(evaluation_limit - state.evaluations,
            _node_state_total_remaining(state)) >= 2 ||
            (state.skipped_variations += 1; return state)
        deferred_second && (second_parent = rank_second())
        first_value, second_value, changed =
            first_parent.value, second_parent.value, false
        for _ in 1:10
            first_value, second_value, changed = _regularized_crossover_pair(
                state.domain, state.rng, first_parent.value,
                second_parent.value, state.config)
            changed && break
        end
        state.changed_variations += changed
        if !changed
            state.skipped_variations += 1
            return state
        end
        first_child = _node_state_score_candidate!(
            state, first_value, true, evaluation_limit)
        second_child = _node_state_score_candidate!(
            state, second_value, true, evaluation_limit)
        first_child === nothing ||
            (first_child = _node_state_refine_offspring!(state, first_child))
        second_child === nothing ||
            (second_child = _node_state_refine_offspring!(state, second_child))
        for (parent, child) in ((first_parent, first_child),
                (second_parent, second_child))
            child === nothing && continue
            _node_state_record!(state, child)
            admitted = _node_state_replace_offspring!(state, parent, child)
            _regularized_observe(state.domain, state, :crossover,
                parent, child, admitted; new_offspring=true)
        end
        return state
    end

    kind = draw < state.config.mutation_probability ?
        _regularized_sample_mutation(state.domain, state.rng,
            first_parent.value, state.config) :
        :compose
    if deferred_second && _node_mutation_uses_donor(kind)
        second_parent = rank_second()
    end
    cycle_temperature, _ = _regularized_cycle_temperature(
        state.offspring_attempts, state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    child_value, changed = _regularized_named_child(
        state.domain, state.rng, first_parent.value, second_parent.value,
        state.feature_count, state.config, kind;
        temperature=cycle_temperature)
    if kind === :compose
        state.composition_attempts += 1
    else
        state.mutation_attempts += 1
        state.mutation_kind_counts[kind] =
            get(state.mutation_kind_counts, kind, 0) + 1
    end
    state.changed_variations += changed
    if kind in (:do_nothing, :simplify)
        state.serial += 1
        child = _NodeSearchCandidate(
            first_parent.value, first_parent.loss, state.serial, true)
        admitted = _node_state_replace_offspring!(state, first_parent, child)
        _regularized_observe(state.domain, state, kind,
            first_parent, child, admitted)
    elseif !changed && kind !== :rotation_copy
        state.skipped_variations += 1
    else
        child = _node_state_score_candidate!(state, child_value, true, evaluation_limit)
        child === nothing && return state
        kind === :rotation_copy && (state.rotation_copy_evaluations += 1)
        child = _node_state_refine_offspring!(state, child)
        # Archive every finite scored mutation before applying the live-island
        # admission policy. A rejected child can still be the best candidate
        # at its complexity and the global loss best.
        _node_state_record!(state, child)
        if _node_state_accept!(state, first_parent, child)
            admitted = _node_state_replace_offspring!(state, first_parent, child)
            kind === :rotation_copy && admitted &&
                (state.rotation_copy_accepted += 1)
        else
            state.rejected_offspring += 1
            admitted = false
        end
        _regularized_observe(state.domain, state, kind,
            first_parent, child, admitted; new_offspring=true)
    end
    return state
end

"""Build and score one persistent regularized population from ready values.

`score(value)` returns `(loss, invalid)`. Values must share one concrete type
and expose positive integer `complexity`; a supplied frequency object controls
its own complexity capacity, while the default covers both configured and
initial-value complexity. Domain hooks own local proposal limits.
"""
function _initialize_regularized_values(values, score;
        config::ScalarSearchConfig, rng::AbstractRNG,
        flow::_NodeRegularizedFlow, feature_count::Int=0,
        frequency=nothing, frequency_window::Int=100000,
        loss_normalization::Real=0.01, domain=nothing)
    config.rotation_noop_reproduction && domain !== nothing &&
        throw(ArgumentError(
            "rotation_noop_reproduction is supported only for native scalar search (domain=nothing)"))
    config.deletion_policy !== :operator && domain !== nothing &&
        throw(ArgumentError("deletion_policy=:child_edge is supported only for native scalar search"))
    initial_values = collect(values)
    length(initial_values) > config.population_size &&
        resize!(initial_values, config.population_size)
    length(initial_values) == config.population_size ||
        throw(ArgumentError("initial values must fill population_size"))
    all(value -> hasproperty(value, :complexity), initial_values) ||
        throw(ArgumentError("regularized values must expose complexity"))
    value_type = typeof(first(initial_values))
    all(value -> typeof(value) === value_type, initial_values) ||
        throw(ArgumentError("regularized values must share one concrete type"))
    all(value -> value.complexity isa Integer && value.complexity > 0,
        initial_values) ||
        throw(ArgumentError("regularized value complexity must be a positive integer"))
    loss_normalization > 0 ||
        throw(ArgumentError("loss_normalization must be positive"))
    positive_complexity = maximum(value.complexity for value in initial_values)
    shared_frequency = frequency === nothing ?
        _RegularizedFrequency(frequency_window,
            max(config.max_complexity, positive_complexity)) : frequency
    first_value = first(initial_values)
    candidate_type = typeof(_NodeSearchCandidate(first_value, Inf, 0, false))
    member_type = typeof(_NodeRegularizedMember(
        _NodeSearchCandidate(first_value, Inf, 0, false), 0))
    state = _NodeRegularizedState(config, rng, feature_count, score, flow,
        member_type[],
        _NodeTournamentWorkspace(config.population_size, config.tournament_size),
        Dict{Int,candidate_type}(), shared_frequency, domain,
        0, 0, 0, 0, 0, 0,
        _NodeSearchCandidate(first_value, Inf, 0, false), NamedTuple[],
        0, 0, 0, 0, 0, 0, 0, 0, 0, Dict{Symbol,Int}(), 0,
        0, 0, Float64(loss_normalization), 0, 0)
    for value in initial_values
        candidate = _node_state_score_candidate!(
            state, value, false, config.max_evaluations)
        candidate === nothing && break
        _node_state_record!(state, candidate)
        state.birth_clock += 1
        push!(state.population,
            _NodeRegularizedMember(candidate, state.birth_clock))
        _regularized_observe(state.domain, state, :initial_population,
            nothing, candidate, true)
    end
    length(state.population) == config.population_size ||
        throw(ArgumentError("initial population did not fill population_size"))
    return state
end

function _node_state_one_offspring!(state::_NodeRegularizedState,
        evaluation_limit::Int)
    state.offspring_attempts += 1
    _node_state_offspring_variation!(state, evaluation_limit)
    # A skipped proposal still completes an attempt. Maintenance belongs to the
    # attempt clock, irrespective of whether its final proposal changed/scored.
    # Do not use finally: unexpected errors must escape without more search.
    _, period = _regularized_cycle_temperature(
        state.offspring_attempts, state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    state.offspring_attempts % period == 0 &&
        (state.config.max_total_evaluations === nothing ||
            _node_state_total_remaining(state) > 0) &&
        _node_state_maintain_cycle!(state, evaluation_limit)
    return state
end

"""Initialize one owned persistent island from a scalar training context.

`X`, `y`, `objective`, and the operator/configuration world anchor every later
score and variation. A supplied `frequency` may be shared only by serial
islands for the same task. Invalid shapes, non-finite data, incompatible
objectives, or an unfillable initial population fail before the state escapes.
"""
function _initialize_node_regularized_state(X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real}; config::ScalarSearchConfig=ScalarSearchConfig(),
        rng::AbstractRNG=default_rng(), initial_population=nothing,
        objective=nothing, constant_jacobian=nothing, stop_loss::Union{Nothing,Real}=
            objective === nothing ? 1.0e-12 : nothing,
        annealing::Bool=true, temperature::Real=1.0, alpha::Real=3.17,
        ncycles_per_iteration::Int=380, use_frequency::Bool=true,
        frequency_window::Int=100000, hof_migration_fraction::Real=0.0614,
        tournament_selection_p::Real=0.982,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0,
        frequency=nothing)
    feature_count = size(X, 1)
    size(X, 2) == length(y) || throw(DimensionMismatch("target length must match samples"))
    isempty(y) && throw(ArgumentError("scalar search requires at least one sample"))
    all(isfinite, X) || throw(ArgumentError("scalar search inputs must be finite"))
    all(isfinite, y) || throw(ArgumentError("scalar search targets must be finite"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))
    constant_jacobian === nothing || (objective === nothing &&
        config.constant_refinement_optimizer === :bfgs) || throw(ArgumentError(
        "constant_jacobian requires direct MSE and BFGS refinement"))
    flow = _NodeRegularizedFlow(
        stop_loss === nothing ? nothing : Float64(stop_loss), annealing,
        Float64(temperature), Float64(alpha), ncycles_per_iteration,
        use_frequency, Float64(hof_migration_fraction),
        Float64(tournament_selection_p), use_frequency_in_tournament,
        Float64(adaptive_parsimony_scaling))
    function custom_score(value::_NodeScalarCandidate)
        expr = _node_scalar_materialize(value)
        expr isa Err && return (Inf, true)
        result = applicable(objective, expr, value.node) ?
            objective(expr, value.node) : objective(expr)
        result isa Err && return (Inf, true)
        result isa Real || throw(ArgumentError("objective must return a real loss or Err"))
        loss = Float64(result)
        return isfinite(loss) ? (loss, false) : (Inf, true)
    end
    score = objective === nothing ? _NodeMSEScore(X, y, constant_jacobian) : custom_score
    values = _NodeScalarCandidate[]
    if initial_population !== nothing
        all(expr -> expr isa ScalarExpr, initial_population) ||
            throw(ArgumentError("initial_population must contain ScalarExpr values"))
        all(expr -> scalar_complexity(expr) <= config.max_complexity,
            initial_population) ||
            throw(ArgumentError("initial_population exceeds max_complexity"))
        all(expr -> _required_features(expr) <= feature_count,
            initial_population) ||
            throw(ArgumentError(
                "initial_population references a feature outside the input matrix"))
        append!(values, (_node_scalar_candidate(expr) for expr in initial_population))
    end
    length(values) > config.population_size &&
        resize!(values, config.population_size)
    while length(values) < config.population_size
        node = _node_random_grow(
            rng, feature_count, config, config.initial_depth)
        candidate = _node_scalar_candidate(node)
        if candidate.complexity > config.max_complexity
            node = _node_random_fixed_size(
                rng, feature_count, config,
                rand(rng, 1:config.max_complexity))
            candidate = _node_scalar_candidate(node)
        end
        candidate.complexity <= config.max_complexity && push!(values, candidate)
    end
    shared_frequency = frequency === nothing ?
        _RegularizedFrequency(frequency_window, config.max_complexity) : frequency
    baseline = _node_scalar_candidate(scalar_constant(sum(y) / length(y)))
    baseline_loss, baseline_invalid = score(baseline)
    loss_normalization = baseline_invalid || !isfinite(baseline_loss) ?
        0.01 : max(0.01, abs(baseline_loss))
    return _initialize_regularized_values(values, score; config, rng, flow,
        feature_count, frequency=shared_frequency, frequency_window,
        loss_normalization,
        domain=nothing)
end

"""Advance an island without rebuilding its population or lifecycle state.

The cumulative evaluation boundary and optional offspring-attempt boundary
anchor the step. Limits below current accounting or above the configured cap
raise `ArgumentError`; expected invalid proposals remain search observations.
"""
function _step_node_regularized!(state::_NodeRegularizedState;
        evaluation_limit::Int=state.config.max_evaluations,
        offspring_limit::Int=typemax(Int))
    state.evaluations <= evaluation_limit <= state.config.max_evaluations ||
        throw(ArgumentError("evaluation_limit must be between current and configured evaluations"))
    attempted = 0
    max_variation_attempts = max(1,
        10 * max(0, evaluation_limit - state.evaluations))
    while state.evaluations < evaluation_limit && attempted < offspring_limit &&
            attempted < max_variation_attempts &&
            _node_state_total_remaining(state) > 0 &&
            !(state.flow.stop_loss !== nothing &&
                state.best.loss <= state.flow.stop_loss)
        _node_state_one_offspring!(state, evaluation_limit)
        attempted += 1
    end
    return state
end

"""Read the completed-or-partial maintenance cycle count without publishing candidates."""
function _node_state_generations(state::_NodeRegularizedState)
    _, period = _regularized_cycle_temperature(
        max(1, state.offspring_attempts), state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    return state.offspring_attempts == 0 ? 0 :
        cld(state.offspring_attempts, period)
end

function _node_regularized_result(state::_NodeRegularizedState)
    solved = state.flow.stop_loss !== nothing &&
        state.best.loss <= state.flow.stop_loss
    termination = solved ? :solved :
        (_node_state_total_remaining(state) <= 0 ? :total_evaluation_cap :
            (state.evaluations >= state.config.max_evaluations ?
                :evaluation_cap : :paused))
    generations = _node_state_generations(state)
    return _node_regularized_report(state.population, state.archive, state.best,
        state.evaluations, state.invalid_evaluations,
        state.constant_refinement_evaluations,
        state.constant_refinement_invalid_evaluations,
        generations, termination, state.improvements,
        state.mutation_attempts, state.crossover_attempts,
        state.composition_attempts, state.changed_variations,
        state.mutation_kind_counts, state.accepted_offspring,
        state.rejected_offspring, state.skipped_variations,
        state.oldest_replacements, state.hof_migrations,
        state.flow.annealing, state.flow.use_frequency,
        state.frequency.window, state.cycle_refinement_calls,
        state.offspring_refinement_calls,
        state.config.max_total_evaluations,
        state.config.max_total_evaluations === nothing ? nothing :
            max(0, _node_state_total_remaining(state)))
end

function _search_scalar_regularized_node(X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        config::ScalarSearchConfig=ScalarSearchConfig(),
        rng::AbstractRNG=default_rng(), initial_population=nothing,
        objective=nothing, stop_loss::Union{Nothing,Real}=
            objective === nothing ? 1.0e-12 : nothing,
        annealing::Bool=true, temperature::Real=1.0, alpha::Real=3.17,
        ncycles_per_iteration::Int=380, use_frequency::Bool=true,
        frequency_window::Int=100000, hof_migration_fraction::Real=0.0614,
        tournament_selection_p::Real=0.982,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0)
    state = _initialize_node_regularized_state(X, y;
        config, rng, initial_population, objective, stop_loss,
        annealing, temperature, alpha, ncycles_per_iteration,
        use_frequency, frequency_window, hof_migration_fraction,
        tournament_selection_p, use_frequency_in_tournament,
        adaptive_parsimony_scaling)
    _step_node_regularized!(state)
    return _node_regularized_result(state)
end

@testsnippet NodeRegularizedStateImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
end

@testsnippet RefinementAgePolicySetup begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.ScalarSearch

    struct RefinementAgeMockValue
        complexity::Int
        loss::Float64
        improvable::Bool
    end
    struct RefinementAgeMockDomain
        improve::Bool
    end

    ScalarSearch._regularized_has_constants(::RefinementAgeMockDomain,
        ::RefinementAgeMockValue) = true
    ScalarSearch._regularized_simplify_value(::RefinementAgeMockDomain,
        value::RefinementAgeMockValue, rng) = nothing
    function ScalarSearch._regularized_refine_value(
            domain::RefinementAgeMockDomain, value::RefinementAgeMockValue,
            loss::Real, score, config::ScalarSearchConfig, remaining::Int;
            rng, restarts::Int=2)
        remaining <= 0 && return (value, Float64(loss), 0, 0)
        refined = domain.improve && value.improvable ?
            RefinementAgeMockValue(value.complexity, Float64(loss) - 1.0, false) :
            value
        refined_loss, invalid = score(refined)
        return refined, refined_loss, 1, invalid
    end

    function refinement_age_state(; reset_age=nothing, improve=true,
            improvable=true, probability=1.0, budget=8, seed=71)
        options = (
            population_size=2,
            max_evaluations=8,
            max_complexity=2,
            initial_depth=0,
            tournament_size=1,
            elite_count=1,
            unary_operators=(:square,),
            binary_operators=(:+,),
            constant_refinement_iterations=1,
            constant_refinement_start_generation=1,
            constant_refinement_probability=probability,
            max_constant_refinement_evaluations=budget,
            evolution_strategy=:regularized,
        )
        reset_options = reset_age === nothing ? (;) :
            (; constant_refinement_resets_age=reset_age)
        config = ScalarSearchConfig(; merge(options, reset_options)...)
        values = [RefinementAgeMockValue(1, 3.0, improvable),
            RefinementAgeMockValue(1, 2.0, false)]
        score = value -> (value.loss, 0)
        flow = ScalarSearch._NodeRegularizedFlow(nothing, false, 1.0, 3.17,
            1, false, 0.0, 0.982, false, 0.0)
        state = ScalarSearch._initialize_regularized_values(values, score;
            config, rng=MersenneTwister(seed), flow, domain=RefinementAgeMockDomain(improve))
        state.offspring_attempts = 1
        return state
    end
end

@testitem "cycle refinement age reset requires strict improvement and opt-in" setup=[RefinementAgePolicySetup] begin
    enabled = refinement_age_state(reset_age=true)
    original = [member.candidate for member in enabled.population]
    ScalarSearch._node_state_maintain_cycle!(enabled, enabled.config.max_evaluations)
    @test enabled.population[1].candidate.loss == original[1].loss - 1.0
    @test enabled.population[1].candidate.serial == original[1].serial
    @test enabled.population[1].born == enabled.birth_clock
    @test enabled.population[1].born > enabled.population[2].born
    @test enabled.birth_clock == 3

    unchanged = refinement_age_state(reset_age=true, improve=false)
    unchanged_ages = [member.born for member in unchanged.population]
    unchanged_clock = unchanged.birth_clock
    ScalarSearch._node_state_maintain_cycle!(unchanged,
        unchanged.config.max_evaluations)
    @test [member.born for member in unchanged.population] == unchanged_ages
    @test unchanged.birth_clock == unchanged_clock

    skipped = refinement_age_state(reset_age=true, probability=0.0)
    skipped_ages = [member.born for member in skipped.population]
    ScalarSearch._node_state_maintain_cycle!(skipped,
        skipped.config.max_evaluations)
    @test skipped.constant_refinement_evaluations == 0
    @test [member.born for member in skipped.population] == skipped_ages

    exhausted = refinement_age_state(reset_age=true, budget=0)
    exhausted_ages = [member.born for member in exhausted.population]
    exhausted_clock = exhausted.birth_clock
    ScalarSearch._node_state_maintain_cycle!(exhausted,
        exhausted.config.max_evaluations)
    @test exhausted.constant_refinement_evaluations == 0
    @test [member.born for member in exhausted.population] == exhausted_ages
    @test exhausted.birth_clock == exhausted_clock

    default = refinement_age_state()
    explicit_off = refinement_age_state(reset_age=false)
    @test default.config.constant_refinement_resets_age === false
    ScalarSearch._node_state_maintain_cycle!(default, default.config.max_evaluations)
    ScalarSearch._node_state_maintain_cycle!(explicit_off,
        explicit_off.config.max_evaluations)
    @test [member.born for member in default.population] == [1, 2]
    @test default.birth_clock == explicit_off.birth_clock == 2
    @test default.constant_refinement_evaluations ==
        explicit_off.constant_refinement_evaluations == 2
    @test [member.candidate.loss for member in default.population] ==
        [member.candidate.loss for member in explicit_off.population]
    @test rand(default.rng) == rand(explicit_off.rng)

    copied = ScalarSearch._copy_config(enabled.config,
        enabled.config.max_evaluations,
        enabled.config.max_constant_refinement_evaluations,
        enabled.config.max_total_evaluations)
    @test copied.constant_refinement_resets_age
end

@testitem "cycle refinement age state is identical across segmented advancement" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=8, max_complexity=4,
        initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        mutation_probability=1.0, crossover_probability=0.0,
        constant_refinement_iterations=3,
        constant_refinement_start_generation=1,
        constant_refinement_probability=1.0,
        constant_refinement_timing=:cycle,
        constant_refinement_resets_age=true,
        max_constant_refinement_evaluations=12,
        constant_refinement_optimizer=:nelder_mead,
        evolution_strategy=:regularized)
    X = zeros(Float64, 0, 2)
    objective = expr -> (first(evaluate_scalar_reference(expr, X)) - 2.0)^2
    initial = [scalar_constant(0.0), scalar_constant(1.0)]
    function make_state()
        return ScalarSearch._initialize_node_regularized_state(
            X, [2.0, 2.0]; config, rng=MersenneTwister(42), objective,
            initial_population=initial, stop_loss=nothing, annealing=false,
            use_frequency=false, ncycles_per_iteration=1,
            hof_migration_fraction=0.0)
    end

    segmented = make_state()
    continuous = make_state()
    ScalarSearch._step_node_regularized!(segmented; evaluation_limit=5)
    ScalarSearch._step_node_regularized!(segmented; evaluation_limit=8)
    ScalarSearch._step_node_regularized!(continuous; evaluation_limit=8)

    @test segmented.evaluations == continuous.evaluations == 8
    @test segmented.constant_refinement_evaluations ==
        continuous.constant_refinement_evaluations
    @test segmented.serial == continuous.serial
    @test segmented.birth_clock == continuous.birth_clock
    @test segmented.offspring_attempts == continuous.offspring_attempts
    @test [(member.candidate.loss, member.candidate.serial, member.born) for
        member in segmented.population] ==
        [(member.candidate.loss, member.candidate.serial, member.born) for
            member in continuous.population]
    @test all(ScalarSearch._same_scalar_expression(
        ScalarSearch._node_scalar_materialize(left.candidate.value),
        ScalarSearch._node_scalar_materialize(right.candidate.value))
        for (left, right) in zip(segmented.population, continuous.population))
    @test rand(segmented.rng) == rand(continuous.rng)
end

@testitem "regularized search accepts finite negative objective losses" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_complexity=1, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(), binary_operators=(:+,), mutation_probability=0.0,
        crossover_probability=0.0, constant_refinement_iterations=0,
        constant_refinement_probability=0.0,
        max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized)
    objective = _ -> -2.0
    X = zeros(Float64, 0, 2)
    y = zeros(2)
    initial = [scalar_constant(0.0), scalar_constant(1.0)]
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(1111), initial_population=initial,
        objective)

    @test state.evaluations == 2
    @test state.invalid_evaluations == 0
    @test state.best.loss == -2.0
    @test state.loss_normalization == 2.0
    @test state.flow.stop_loss === nothing

    threshold_state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(1111),
        initial_population=initial, objective, stop_loss=-1.0)
    @test threshold_state.flow.stop_loss == -1.0
    @test state.tournament_workspace !== threshold_state.tournament_workspace
    @test state.tournament_workspace.available !==
        threshold_state.tournament_workspace.available
    @test state.tournament_workspace.selected !==
        threshold_state.tournament_workspace.selected
    @test state.tournament_workspace.costs !==
        threshold_state.tournament_workspace.costs
    @test state.tournament_workspace.order !==
        threshold_state.tournament_workspace.order
end

@testsnippet SimplificationScoreContractSetup begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ScalarSearch
    using LambdaRegression.Utils: Err

    function simplification_state(expression, objective;
            X=reshape([2.0], 1, 1), max_evaluations=8,
            max_total_evaluations=nothing, refinement_budget=0,
            reset_age=false)
        config = ScalarSearchConfig(; population_size=2, max_evaluations,
            max_total_evaluations, max_complexity=5, initial_depth=0,
            tournament_size=1, elite_count=1, unary_operators=(),
            binary_operators=(:+, :-), constant_refinement_iterations=refinement_budget > 0 ? 1 : 0,
            constant_refinement_start_generation=1,
            constant_refinement_probability=refinement_budget > 0 ? 1.0 : 0.0,
            max_constant_refinement_evaluations=refinement_budget,
            constant_refinement_resets_age=reset_age,
            evolution_strategy=:regularized)
        return ScalarSearch._initialize_node_regularized_state(X, [0.0];
            config, objective, rng=MersenneTwister(1234),
            initial_population=[deepcopy(expression), deepcopy(expression)],
            stop_loss=nothing, use_frequency=false, ncycles_per_iteration=1,
            hof_migration_fraction=0.0)
    end
end

@testitem "cycle simplification scores structure and signed-zero objectives" setup=[SimplificationScoreContractSetup] begin
    original = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    X = reshape([-0.0], 1, 1)
    structural = expression -> expression isa ScalarSearch._ScalarInput ? 1.0 : 0.0
    signed_zero = expression -> signbit(only(evaluate_scalar_reference(expression, X))) ? 1.0 : 0.0
    for objective in (structural, signed_zero)
        state = simplification_state(original, objective; X)
        ages = [member.born for member in state.population]
        birth_clock = state.birth_clock
        ScalarSearch._node_state_maintain_cycle!(state, 8)
        @test state.evaluations == 4
        @test all(member.candidate.value.complexity == 3 for member in state.population)
        @test all(member.candidate.loss == 0.0 for member in state.population)
        @test state.archive[1].loss == 1.0
        @test state.best.loss == 0.0
        @test [member.born for member in state.population] == ages
        @test state.birth_clock == birth_clock
        for candidate in values(state.archive)
            expression = ScalarSearch._node_scalar_materialize(candidate.value)
            @test objective(expression) == candidate.loss
        end
    end
end

@testitem "cycle simplification accepts scored improvements without refreshing age" setup=[SimplificationScoreContractSetup] begin
    original = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    objective = expression -> Float64(scalar_complexity(expression))
    state = simplification_state(original, objective; reset_age=true)
    ages = [member.born for member in state.population]
    birth_clock = state.birth_clock
    ScalarSearch._node_state_maintain_cycle!(state, 8)
    @test state.evaluations == 4
    @test all(member.candidate.value.complexity == 1 for member in state.population)
    @test all(member.candidate.loss == 1.0 for member in state.population)
    @test state.archive[1].loss == state.best.loss == 1.0
    @test [member.born for member in state.population] == ages
    @test state.birth_clock == birth_clock
end

@testitem "cycle simplification rejects invalid scores and propagates unexpected failures" setup=[SimplificationScoreContractSetup] begin
    original = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    objective = expression -> expression isa ScalarSearch._ScalarInput ? Err(:rejected_input) : 0.0
    state = simplification_state(original, objective)
    previous = [member.candidate for member in state.population]
    ScalarSearch._node_state_maintain_cycle!(state, 8)
    @test state.evaluations == 4
    @test state.invalid_evaluations == 2
    @test !haskey(state.archive, 1)
    @test [member.candidate for member in state.population] == previous

    failing = expression -> expression isa ScalarSearch._ScalarInput ? error("unexpected simplification score") : 0.0
    failed = simplification_state(original, failing)
    previous = [member.candidate for member in failed.population]
    @test_throws ErrorException ScalarSearch._node_state_maintain_cycle!(failed, 8)
    @test failed.evaluations == 2
    @test [member.candidate for member in failed.population] == previous
    @test !haskey(failed.archive, 1)
end

@testitem "cycle simplification respects current structural and shared total limits" setup=[SimplificationScoreContractSetup] begin
    original = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    for (limit, total, expected) in ((2, nothing, 2), (3, nothing, 3), (8, 2, 2), (8, 3, 3))
        calls = Ref(0)
        objective = expression -> begin
            calls[] += 1
            Float64(scalar_complexity(expression))
        end
        state = simplification_state(original, objective; max_total_evaluations=total)
        previous = [member.candidate for member in state.population]
        ages = [member.born for member in state.population]
        birth_clock = state.birth_clock
        serial = state.serial
        ScalarSearch._node_state_maintain_cycle!(state, limit)
        @test state.evaluations == expected
        @test calls[] == expected + 1 # One initialization baseline is outside the cap.
        @test state.serial == serial + expected - 2
        @test count(member -> member.candidate.value.complexity == 1, state.population) == expected - 2
        @test [member.born for member in state.population] == ages
        @test state.birth_clock == birth_clock
        total === nothing || @test state.evaluations + state.constant_refinement_evaluations <= total
        if expected == 2
            @test [member.candidate for member in state.population] == previous
            @test !haskey(state.archive, 1)
        else
            @test state.archive[1].loss == 1.0
        end
    end
end

@testitem "exhausted structural slice still permits bounded cycle constant refinement" setup=[SimplificationScoreContractSetup] begin
    X = reshape([2.0], 1, 1)
    input = scalar_input(1)
    original = scalar_binary(:-, input,
        scalar_binary(:+, scalar_constant(1.0), input))
    objective = expression -> abs2(only(evaluate_scalar_reference(expression, X)))
    state = simplification_state(original, objective; X, max_total_evaluations=6,
        refinement_budget=4)
    state.offspring_attempts = 1
    ages = [member.born for member in state.population]
    birth_clock = state.birth_clock
    ScalarSearch._node_state_maintain_cycle!(state, 2)
    @test state.evaluations == 2
    @test 0 < state.constant_refinement_evaluations <= 4
    @test state.evaluations + state.constant_refinement_evaluations <= 6
    @test state.cycle_refinement_calls > 0
    @test all(member.candidate.value.complexity == 5 for member in state.population)
    @test !haskey(state.archive, 1)
    @test [member.born for member in state.population] == ages
    @test state.birth_clock == birth_clock
end

@testitem "node regularized skipped proposals still finish cycle maintenance" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=4, max_complexity=1,
        initial_depth=0, tournament_size=2, elite_count=1,
        mutation_probability=0.0, crossover_probability=1.0,
        constant_refinement_iterations=8, constant_refinement_probability=1.0,
        constant_refinement_start_generation=1,
        max_constant_refinement_evaluations=64, evolution_strategy=:regularized)
    # A pair of identical leaves cannot change by crossover. The second limit
    # instead skips before crossover because it cannot score both children.
    for limit in (4, 3)
        state = ScalarSearch._initialize_node_regularized_state(
            zeros(0, 4), fill(3.0, 4); config, rng=MersenneTwister(42),
            initial_population=[scalar_constant(0.0), scalar_constant(0.0)],
            stop_loss=nothing, ncycles_per_iteration=1, hof_migration_fraction=0.0)
        ScalarSearch._node_state_one_offspring!(state, limit)
        @test state.offspring_attempts == 1
        @test state.skipped_variations == 1
        @test state.evaluations == 2
        @test state.frequency.frequencies[1] == 3.0
        @test 0 < state.constant_refinement_evaluations <= 64
        @test state.best.loss < 1e-8
    end
end

@testitem "rotation no-op is scored and accepted as an unchanged regularized offspring" setup=[NodeRegularizedStateImports] begin
    S = ScalarSearch
    config = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_complexity=1, initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:sin,), binary_operators=(:+,),
        mutation_probability=1.0, crossover_probability=0.0,
        constant_refinement_iterations=0,
        rotation_noop_reproduction=true,
        evolution_strategy=:regularized)
    X = reshape(Float64[0, 1], 1, :)
    initial = [scalar_input(1), scalar_input(1)]
    calls = Ref(0)
    objective = expression -> (calls[] += 1; 1.0)
    state = S._initialize_node_regularized_state(X, [0.0, 1.0]; config,
        rng=MersenneTwister(1), initial_population=initial, objective,
        stop_loss=nothing, annealing=false, use_frequency=false,
        ncycles_per_iteration=1, hof_migration_fraction=0.0)
    S._node_state_one_offspring!(state, config.max_evaluations)
    @test get(state.mutation_kind_counts, :rotation_copy, 0) == 1
    @test calls[] == 4 # baseline, two members, and the scored copy
    @test state.evaluations == 3
    @test state.mutation_attempts == 1
    @test state.changed_variations == 0
    @test state.skipped_variations == 0
    @test state.accepted_offspring == 1
    @test state.rejected_offspring == 0
    @test state.rotation_copy_evaluations == 1
    @test state.rotation_copy_accepted == 1
    @test state.serial == 3
    @test state.birth_clock == 3
    @test maximum(member.born for member in state.population) == 3
    @test any(member -> member.candidate.serial == 3, state.population)
    @test all(S._node_scalar_materialize(member.candidate.value) ==
        scalar_input(1) for member in state.population)

    rejected_calls = Ref(0)
    reject_objective = expression -> begin
        rejected_calls[] += 1
        rejected_calls[] <= 3 ? 1.0 : Inf
    end
    rejected = S._initialize_node_regularized_state(X, [0.0, 1.0]; config,
        rng=MersenneTwister(1), initial_population=initial,
        objective=reject_objective, stop_loss=nothing, annealing=false,
        use_frequency=false, ncycles_per_iteration=1,
        hof_migration_fraction=0.0)
    S._node_state_one_offspring!(rejected, config.max_evaluations)
    @test get(rejected.mutation_kind_counts, :rotation_copy, 0) == 1
    @test rejected.rotation_copy_evaluations == 1
    @test rejected.rotation_copy_accepted == 0
    @test rejected.invalid_evaluations == 1
    @test rejected.rejected_offspring == 1
    @test rejected.changed_variations == 0
    @test rejected.birth_clock == 2
    @test [member.born for member in rejected.population] == [1, 2]
end

@testitem "rotation no-op rejects domain-backed states and honors exhausted budgets" setup=[NodeRegularizedStateImports] begin
    S = ScalarSearch
    config = ScalarSearchConfig(population_size=2, max_evaluations=4,
        max_total_evaluations=2, max_complexity=1, initial_depth=0,
        tournament_size=1, elite_count=1,
        rotation_noop_reproduction=true,
        evolution_strategy=:regularized)
    flow = S._NodeRegularizedFlow(nothing, false, 1.0, 3.17, 1,
        false, 0.0, 0.982, false, 0.0)
    values = [(complexity=1,), (complexity=1,)]
    @test_throws ArgumentError S._initialize_regularized_values(values,
        _ -> (1.0, 0); config, rng=MersenneTwister(1111), flow,
        domain=NamedTuple())

    calls = Ref(0)
    objective = expression -> (calls[] += 1; 1.0)
    exhausted = S._initialize_node_regularized_state(
        reshape(Float64[0, 1], 1, :), [0.0, 1.0]; config,
        rng=MersenneTwister(2000),
        initial_population=[scalar_input(1), scalar_input(1)], objective,
        stop_loss=nothing, annealing=false, use_frequency=false,
        ncycles_per_iteration=1, hof_migration_fraction=0.0)
    S._step_node_regularized!(exhausted;
        evaluation_limit=config.max_evaluations, offspring_limit=1)
    @test calls[] == 3 # baseline and two initial members; no offspring score
    @test exhausted.evaluations == 2
    @test exhausted.offspring_attempts == 0
    @test isempty(exhausted.mutation_kind_counts)
    @test exhausted.rotation_copy_evaluations == 0
    @test exhausted.rotation_copy_accepted == 0
end

@testitem "node regularized state resumes without resetting island history" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=20,
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
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(8000), stop_loss=nothing,
        annealing=false, use_frequency=false, ncycles_per_iteration=2)

    @test state.evaluations == config.population_size
    ScalarSearch._step_node_regularized!(state; evaluation_limit=10)
    first_serial = state.serial
    first_birth = state.birth_clock
    first_population = objectid(state.population)
    @test state.evaluations == 10

    ScalarSearch._step_node_regularized!(state; evaluation_limit=20)
    @test state.evaluations == config.max_evaluations
    @test state.serial > first_serial
    @test state.birth_clock > first_birth
    @test objectid(state.population) == first_population
    result = ScalarSearch._node_regularized_result(state)
    @test result.evaluations == 20
    @test result.best isa ScalarExpr
end

@testitem "node regularized search refines constants with zero input features" setup=[NodeRegularizedStateImports] begin
    X = zeros(Float64, 0, 4)
    y = fill(3.0, 4)
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=4,
        max_complexity=1,
        initial_depth=0,
        tournament_size=2,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_probability=0.0,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=32,
        constant_refinement_probability=1.0,
        constant_refinement_start_generation=1,
        max_constant_refinement_evaluations=24,
        constant_refinement_optimizer=:nelder_mead,
        evolution_strategy=:regularized,
    )
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config,
        rng=MersenneTwister(42),
        initial_population=[scalar_constant(0.0), scalar_constant(-1.0)],
        stop_loss=nothing, annealing=false, use_frequency=false,
        ncycles_per_iteration=1)
    initial_best_loss = state.best.loss
    ScalarSearch._step_node_regularized!(state; evaluation_limit=4)
    result = ScalarSearch._node_regularized_result(state)

    @test result.best_loss < initial_best_loss
    @test result.evaluations <= config.max_evaluations
    @test result.constant_refinement_evaluations > 0
    @test result.constant_refinement_evaluations <=
        config.max_constant_refinement_evaluations
    @test ScalarSearch._required_features(result.best) == 0
end

@testitem "offspring refinement can rescue a child before live admission" setup=[NodeRegularizedStateImports] begin
    X = zeros(Float64, 0, 1)
    y = fill(0.1, 1)
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=12, max_complexity=1,
        initial_depth=0, tournament_size=2, elite_count=1,
        unary_operators=(:square,), binary_operators=(),
        constant_probability=0.0, mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=8,
        constant_refinement_start_generation=1,
        constant_refinement_probability=1.0,
        constant_refinement_timing=:offspring,
        max_constant_refinement_evaluations=40,
        constant_refinement_optimizer=:bfgs,
        max_total_evaluations=40,
        evolution_strategy=:regularized)
    objective = expr -> (only(evaluate_scalar_reference(expr, X)) - 0.1)^2
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(1234), objective,
        initial_population=fill(scalar_constant(0.0), 2),
        stop_loss=nothing, annealing=true, temperature=1.0e-12,
        use_frequency=false)
    parent = state.population[1].candidate
    raw = ScalarSearch._node_state_score_candidate!(state,
        ScalarSearch._node_scalar_candidate(scalar_constant(2.0)), true,
        config.max_evaluations)
    state.offspring_attempts = 1

    @test raw.loss > parent.loss
    @test !ScalarSearch._node_state_accept!(state, parent, raw)
    refined = ScalarSearch._node_state_refine_offspring!(state, raw)
    @test refined.loss < parent.loss
    @test ScalarSearch._node_state_accept!(state, parent, refined)
    @test state.offspring_refinement_calls == 1
    @test state.cycle_refinement_calls == 0
end

@testitem "total evaluation cap reserves crossover scores and bounds refinement" setup=[NodeRegularizedStateImports] begin
    X = reshape(Float64[-1, 1], 1, :)
    y = vec(X)
    callbacks = Ref(0)
    objective = expr -> begin
        callbacks[] += 1
        sum(abs2, evaluate_scalar_reference(expr, X) .- y) / length(y)
    end
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=8, max_complexity=3,
        initial_depth=0, tournament_size=2, elite_count=1,
        unary_operators=(:square,), binary_operators=(:+,),
        constant_probability=0.0, mutation_probability=0.0,
        crossover_probability=1.0,
        constant_refinement_iterations=8,
        constant_refinement_start_generation=1,
        constant_refinement_probability=1.0,
        constant_refinement_timing=:offspring,
        max_constant_refinement_evaluations=20,
        max_total_evaluations=3,
        evolution_strategy=:regularized)
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(8000), objective,
        initial_population=[scalar_input(1),
            scalar_binary(:+, scalar_input(1), scalar_constant(1.0))],
        stop_loss=nothing)
    @test callbacks[] == 1 + state.evaluations
    @test ScalarSearch._node_state_total_remaining(state) == 1

    ScalarSearch._node_state_one_offspring!(state, config.max_evaluations)
    @test state.evaluations == config.population_size
    @test state.constant_refinement_evaluations == 0
    @test state.skipped_variations == 1
    @test callbacks[] == 1 + state.evaluations

    candidate = state.population[2].candidate
    refined = ScalarSearch._node_state_refine_candidate!(
        state, candidate, 1, :offspring)
    @test state.constant_refinement_evaluations == 1
    @test state.evaluations + state.constant_refinement_evaluations <=
        config.max_total_evaluations
    @test ScalarSearch._node_state_total_remaining(state) ==
        config.max_total_evaluations - state.evaluations -
        state.constant_refinement_evaluations
    @test callbacks[] == 1 + state.evaluations +
        state.constant_refinement_evaluations

    @test ScalarSearch._node_state_score_candidate!(state,
        candidate.value, true, config.max_evaluations) === nothing
    @test callbacks[] == 1 + state.evaluations +
        state.constant_refinement_evaluations
    @test_throws ArgumentError ScalarSearchConfig(
        population_size=2, max_evaluations=8, max_total_evaluations=1,
        evolution_strategy=:regularized)
    @test_throws ArgumentError ScalarSearchConfig(
        population_size=2, max_evaluations=8, max_total_evaluations=8,
        constant_refinement_timing=:offspring,
        evolution_strategy=:generational)
end

@testitem "node regularized crossover evaluates and replaces two children" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(
        population_size=4, max_evaluations=6, max_complexity=12,
        initial_depth=1, tournament_size=2, elite_count=1,
        unary_operators=(:sin,), binary_operators=(:+, :*),
        mutation_probability=0.0, crossover_probability=1.0,
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
    ScalarSearch._step_node_regularized!(state; evaluation_limit=6)
    @test state.evaluations == 6
    @test state.crossover_attempts == 1
    @test state.accepted_offspring == 2
    @test state.oldest_replacements == 2
end

@testitem "node regularized archive retains scored rejected mutation children" setup=[NodeRegularizedStateImports] begin
    config = ScalarSearchConfig(
        population_size=4, max_evaluations=5, max_complexity=4,
        initial_depth=0, tournament_size=2, elite_count=1,
        unary_operators=(:square,), binary_operators=(),
        constant_probability=0.0, mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_iterations=0,
        constant_refinement_probability=0.0,
        evolution_strategy=:regularized,
    )
    X = reshape(Float64[0, 1, 2, 3], 1, :)
    y = ones(Float64, 4)
    objective = expr -> scalar_complexity(expr) == 1 ? 1.0 : 0.0
    state = ScalarSearch._initialize_node_regularized_state(
        X, y; config, rng=MersenneTwister(1),
        initial_population=fill(scalar_input(1), config.population_size),
        objective, stop_loss=nothing, annealing=false, use_frequency=true,
        use_frequency_in_tournament=false, ncycles_per_iteration=1_000_000,
        hof_migration_fraction=0.0)

    # Force the regularized live-admission probability of a structural child
    # from a size-one parent to zero, without changing the mutation path.
    state.frequency.normalized_frequencies .= [0.0, 1 / 3, 1 / 3, 1 / 3]
    for _ in 1:40
        state.evaluations > config.population_size && break
        ScalarSearch._step_node_regularized!(state;
            evaluation_limit=config.max_evaluations, offspring_limit=1)
    end

    @test state.evaluations == config.max_evaluations
    @test state.rejected_offspring == 1
    @test state.accepted_offspring == 0
    @test all(member.candidate.value.complexity == 1 for member in state.population)
    @test state.best.loss == 0.0
    @test state.best.value.complexity > 1

    archived = state.archive[state.best.value.complexity]
    @test archived.loss == 0.0
    migrants = ScalarSearch._node_pareto_candidates(state.archive)
    migrant = only(filter(candidate -> candidate.loss == 0.0, migrants))
    copied = ScalarSearch._node_inject_candidate!(state, migrant; destination=1)
    @test copied.loss == 0.0
    @test state.population[1].candidate.loss == 0.0

    result = ScalarSearch._node_regularized_result(state)
    @test result.best_loss == 0.0
    @test any(candidate -> candidate.complexity == archived.value.complexity &&
        candidate.loss == 0.0, result.complexity_archive)
    @test result.variation.hof_migrations == 1
end
