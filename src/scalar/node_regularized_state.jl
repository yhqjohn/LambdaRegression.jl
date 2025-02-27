"""Persistent node-native island state for SR-style multi-population flow."""

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

mutable struct _NodeRegularizedState{R<:AbstractRNG,F}
    config::ScalarSearchConfig
    rng::R
    feature_count::Int
    score::F
    flow::_NodeRegularizedFlow
    population::Vector{_NodeRegularizedMember}
    archive::Dict{Int,_NodeSearchCandidate}
    frequency::_RegularizedFrequency
    evaluations::Int
    invalid_evaluations::Int
    constant_refinement_evaluations::Int
    constant_refinement_invalid_evaluations::Int
    serial::Int
    birth_clock::Int
    best::_NodeSearchCandidate
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
    offspring_attempts::Int
    loss_normalization::Float64
end

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
        value::_NodeScalarCandidate, evolved::Bool, evaluation_limit::Int)
    state.evaluations >= evaluation_limit && return nothing
    loss, invalid = state.score(value)
    state.evaluations += 1
    state.invalid_evaluations += invalid
    state.serial += 1
    return _NodeSearchCandidate(value, loss, state.serial, evolved)
end

function _node_state_refine!(state::_NodeRegularizedState,
        member::_NodeRegularizedMember, generation::Int)
    config = state.config
    config.constant_refinement_iterations == 0 && return member
    generation < config.constant_refinement_start_generation && return member
    remaining = config.max_constant_refinement_evaluations -
        state.constant_refinement_evaluations
    remaining <= 0 && return member
    complexity = member.candidate.value.complexity
    refined, refined_loss, count, invalid = _node_refine_constants(
        member.candidate.value, member.candidate.loss,
        node -> state.score(_NodeScalarCandidate(node, complexity)),
        config, remaining; rng=state.rng, restarts=2)
    state.constant_refinement_evaluations += count
    state.constant_refinement_invalid_evaluations += invalid
    refined_loss < member.candidate.loss || return member
    member.candidate = _NodeSearchCandidate(refined, refined_loss,
        member.candidate.serial, member.candidate.evolved)
    _node_state_record!(state, member.candidate; source=:constant_refinement)
    return member
end

function _node_state_maintain_cycle!(state::_NodeRegularizedState)
    _node_observe_population!(state.frequency, state.population)
    generation = cld(state.offspring_attempts,
        _regularized_cycle_temperature(max(1, state.offspring_attempts),
            state.config.population_size, state.config.tournament_size,
            state.flow.ncycles_per_iteration)[2])
    for member in state.population
        expr = _node_scalar_materialize(member.candidate.value)
        expr isa Err && continue
        simplified = _simplify_scalar(expr)
        if !_same_scalar_expression(simplified, expr)
            member.candidate = _NodeSearchCandidate(
                _node_scalar_candidate(simplified), member.candidate.loss,
                member.candidate.serial, member.candidate.evolved)
            _node_state_record!(state, member.candidate)
        end
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
to the same training data/objective. The copied node receives fresh age and
serial metadata. An invalid destination index fails at the vector boundary.
"""
function _node_inject_candidate!(state::_NodeRegularizedState,
        source::_NodeSearchCandidate; destination=nothing)
    state.serial += 1
    state.birth_clock += 1
    copied_value = _node_scalar_candidate(
        copy(source.value.node; break_sharing=Val(true)))
    copied = _NodeSearchCandidate(
        copied_value, source.loss, state.serial, true)
    index = destination === nothing ?
        rand(state.rng, eachindex(state.population)) : destination
    state.population[index] = _NodeRegularizedMember(copied, state.birth_clock)
    state.hof_migrations += 1
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

function _node_state_one_offspring!(state::_NodeRegularizedState,
        evaluation_limit::Int)
    state.offspring_attempts += 1
    candidates = [member.candidate for member in state.population]
    tournament() = _node_tournament(state.rng, candidates, state.config,
        state.frequency; loss_normalization=state.loss_normalization,
        selection_p=state.flow.tournament_selection_p,
        use_frequency_in_tournament=state.flow.use_frequency_in_tournament,
        adaptive_parsimony_scaling=state.flow.adaptive_parsimony_scaling)
    first_parent, second_parent = tournament(), tournament()
    draw = rand(state.rng)
    if draw >= state.config.mutation_probability &&
            draw < state.config.mutation_probability +
                state.config.crossover_probability
        state.crossover_attempts += 1
        evaluation_limit - state.evaluations >= 2 ||
            (state.skipped_variations += 1; return state)
        first_value, second_value, changed =
            first_parent.value, second_parent.value, false
        for _ in 1:10
            first_value, second_value, changed = _node_scalar_crossover_pair(
                state.rng, first_parent.value, second_parent.value, state.config)
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
        for child in (first_child, second_child)
            child === nothing && continue
            state.accepted_offspring += 1
            index = _node_oldest_index(state.population)
            state.birth_clock += 1
            state.population[index] =
                _NodeRegularizedMember(child, state.birth_clock)
            state.oldest_replacements += 1
            _node_state_record!(state, child)
        end
        _, period = _regularized_cycle_temperature(
            max(1, state.offspring_attempts), state.config.population_size,
            state.config.tournament_size, state.flow.ncycles_per_iteration)
        state.offspring_attempts % period == 0 &&
            _node_state_maintain_cycle!(state)
        return state
    end

    kind = draw < state.config.mutation_probability ?
        _node_scalar_sample_mutation(state.rng, first_parent.value, state.config) :
        :compose
    cycle_temperature, _ = _regularized_cycle_temperature(
        state.offspring_attempts, state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    child_value, changed = _node_regularized_named_child(
        state.rng, first_parent.value, second_parent.value,
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
        state.accepted_offspring += 1
        index = _node_oldest_index(state.population)
        state.birth_clock += 1
        state.population[index] = _NodeRegularizedMember(child, state.birth_clock)
        state.oldest_replacements += 1
        _node_state_record!(state, child)
    elseif !changed
        state.skipped_variations += 1
    else
        child = _node_state_score_candidate!(state, child_value, true, evaluation_limit)
        child === nothing && return state
        if _node_state_accept!(state, first_parent, child)
            state.accepted_offspring += 1
            index = _node_oldest_index(state.population)
            state.birth_clock += 1
            state.population[index] = _NodeRegularizedMember(child, state.birth_clock)
            state.oldest_replacements += 1
            _node_state_record!(state, child)
        else
            state.rejected_offspring += 1
        end
    end
    _, period = _regularized_cycle_temperature(
        max(1, state.offspring_attempts), state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    state.offspring_attempts % period == 0 && _node_state_maintain_cycle!(state)
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
        objective=nothing, stop_loss::Union{Nothing,Real}=1.0e-12,
        annealing::Bool=true, temperature::Real=1.0, alpha::Real=3.17,
        ncycles_per_iteration::Int=380, use_frequency::Bool=true,
        frequency_window::Int=100000, hof_migration_fraction::Real=0.0614,
        tournament_selection_p::Real=0.982,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0,
        frequency=nothing)
    feature_count = size(X, 1)
    feature_count >= 1 || throw(ArgumentError("scalar search requires at least one feature"))
    size(X, 2) == length(y) || throw(DimensionMismatch("target length must match samples"))
    isempty(y) && throw(ArgumentError("scalar search requires at least one sample"))
    all(isfinite, X) || throw(ArgumentError("scalar search inputs must be finite"))
    all(isfinite, y) || throw(ArgumentError("scalar search targets must be finite"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))
    flow = _NodeRegularizedFlow(
        stop_loss === nothing ? nothing : Float64(stop_loss), annealing,
        Float64(temperature), Float64(alpha), ncycles_per_iteration,
        use_frequency, Float64(hof_migration_fraction),
        Float64(tournament_selection_p), use_frequency_in_tournament,
        Float64(adaptive_parsimony_scaling))
    workspace = DynamicScalarWorkspace()
    function score(value::_NodeScalarCandidate)
        if objective === nothing
            prediction = evaluate_dynamic_scalar!(workspace, value.node, X)
            prediction isa Err && return (Inf, true)
            total = 0.0
            @inbounds @simd for index in eachindex(y)
                residual = prediction[index] - y[index]
                total += residual * residual
            end
            loss = total / length(y)
            return isfinite(loss) ? (loss, false) : (Inf, true)
        end
        expr = _node_scalar_materialize(value)
        expr isa Err && return (Inf, true)
        result = applicable(objective, expr, value.node) ?
            objective(expr, value.node) : objective(expr)
        result isa Err && return (Inf, true)
        result isa Real || throw(ArgumentError("objective must return a real loss or Err"))
        loss = Float64(result)
        return isfinite(loss) && loss >= 0 ? (loss, false) : (Inf, true)
    end
    values = _NodeScalarCandidate[]
    initial_population !== nothing && append!(values,
        (_node_scalar_candidate(expr) for expr in initial_population))
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
    first_value = first(values)
    dummy = _NodeSearchCandidate(first_value, Inf, 0, false)
    state = _NodeRegularizedState(config, rng, feature_count, score, flow,
        _NodeRegularizedMember[], Dict{Int,_NodeSearchCandidate}(),
        shared_frequency, 0, 0, 0, 0, 0, 0, dummy, NamedTuple[],
        0, 0, 0, 0, 0, 0, 0, 0, 0, Dict{Symbol,Int}(), 0, 0.01)
    baseline = _node_scalar_candidate(scalar_constant(sum(y) / length(y)))
    baseline_loss, baseline_invalid = score(baseline)
    state.loss_normalization = baseline_invalid || !isfinite(baseline_loss) ?
        0.01 : max(0.01, baseline_loss)
    for value in values
        candidate = _node_state_score_candidate!(state,
            value, false, config.max_evaluations)
        candidate === nothing && break
        _node_state_record!(state, candidate)
        state.birth_clock += 1
        push!(state.population,
            _NodeRegularizedMember(candidate, state.birth_clock))
    end
    length(state.population) == config.population_size ||
        throw(ArgumentError("initial population did not fill population_size"))
    return state
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
            !(state.flow.stop_loss !== nothing &&
                state.best.loss <= state.flow.stop_loss)
        _node_state_one_offspring!(state, evaluation_limit)
        attempted += 1
    end
    return state
end

function _node_regularized_result(state::_NodeRegularizedState)
    solved = state.flow.stop_loss !== nothing &&
        state.best.loss <= state.flow.stop_loss
    termination = solved ? :solved :
        (state.evaluations >= state.config.max_evaluations ?
            :evaluation_cap : :paused)
    _, period = _regularized_cycle_temperature(
        max(1, state.offspring_attempts), state.config.population_size,
        state.config.tournament_size, state.flow.ncycles_per_iteration)
    generations = state.offspring_attempts == 0 ? 0 :
        cld(state.offspring_attempts, period)
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
        state.frequency.window)
end

function _search_scalar_regularized_node(X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        config::ScalarSearchConfig=ScalarSearchConfig(),
        rng::AbstractRNG=default_rng(), initial_population=nothing,
        objective=nothing, stop_loss::Union{Nothing,Real}=1.0e-12,
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
