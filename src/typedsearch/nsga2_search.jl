"""
    search_nsga2(initial, language, objective;
                 rng=default_rng(), group_count=1, proposal_policy=nothing,
                 diverse_initialization=false, material_probability=0.0,
                 group_mating=true, group_survival=true, group_material=true,
                 classifier=nothing, family_capacity=2, evaluation_state=nothing,
                 trace=nothing,
                 search_options...)

An opt-in generational NSGA-II typed search.  The first objective is the
finite training loss returned by `objective`; the second is the existing
written-complexity measure.  Each island owns a private population and RNG.
Initial terms are scored once and copied into every island, while each full
generation makes a fresh scored offspring batch and applies NSGA-II survivor
selection to `P ∪ Q`.  Proposal failures consume raw attempts and invalid
objective calls consume evaluation budget, but neither enters a survivor
set.  If a budget ends with a non-empty partial offspring batch, that batch is
still included in one final survival step.

This entry point deliberately does not combine frequency penalties or
Metropolis acceptance with Pareto selection: nonzero values for those two
configuration options are rejected.  `material_probability` is a separate,
opt-in historical-material source for parent and crossover-donor selection;
it samples occupied best-per-written-size archive entries but never registers
or migrates them.  With `classifier` supplied, the objective must return
`(loss=finite_real, outputs=finite_nonempty_vector)` on one fixed batch.  The
output vector is copied once after the charged objective call. The supplied
classifier partitions observations only within this invocation;
`ExactOutputClassifier()` compares Float64 bit patterns (including signed zero).
Families are an empirical search heuristic, not equivalence certificates.
`diverse_initialization` independently fills each island's unseeded slots with
at most 16 charged generation attempts per slot; copying an admitted seed as a
fallback has no score or attempt fee.  `proposal_policy` is an opt-in proposal
route.  The returned
statistics use `algorithm=:nsga2` and report generations and partial
generations rather than age-replacement acceptance counts.
"""

mutable struct _TypedNSGA2Run
    attempts::Int
    evaluations::Int
    invalid_evaluations::Int
    score_time_seconds::Float64
    selection_time_seconds::Float64
    global_best::Float64
    observer_calls::Int
    improvement_calls::Int
    next_observer_attempt::Int
    birth::Int
end

mutable struct _TypedNSGA2Island
    population::Vector{ScoredTerm}
    outputs::Vector{Vector{Float64}}
    behavior_selection::Any
    rng::AbstractRNG
    ranks::Vector{Int}
    crowding::Vector{Float64}
    attempts::Int
    generations::Int
    partial_generations::Int
    offspring::Int
    evaluations::Int
    invalid_evaluations::Int
    proposal_failures::Int
    changed::Int
    unchanged::Int
    initialization_attempts::Int
    initialization_evaluations::Int
    initialization_invalid_evaluations::Int
    material_parent_selected::Int
    material_donor_selected::Int
    edit_count::Int
    kind_attempts::Dict{Symbol,Int}
    kind_failures::Dict{Symbol,Int}
    kind_evaluations::Dict{Symbol,Int}
    kind_invalid::Dict{Symbol,Int}
    kind_unchanged::Dict{Symbol,Int}
    kind_edit_counts::Dict{Symbol,Int}
end

function _typed_nsga2_island(seeds, config,
                             language::Language, rng::AbstractRNG;
                             seed_outputs=nothing,
                             run=nothing, objective=nothing,
                             archive=nothing, evaluation_state=nothing,
                             output_length=nothing,
                             classifier=nothing,
                             group_mating::Bool=true,
                             group_survival::Bool=true,
                             group_material::Bool=true,
                             diverse_initialization::Bool=false,
                             proposal_policy=nothing,
                             family_capacity::Int=2)
    population = ScoredTerm[]
    outputs = Vector{Vector{Float64}}()
    grouping = classifier !== nothing
    seed_count = diverse_initialization ? min(length(seeds), config.population_size) :
        config.population_size
    grouping && (seed_outputs isa AbstractVector &&
        length(seed_outputs) >= length(seeds) || throw(ArgumentError(
            "behaviour grouping requires seed output vectors")))
    while length(population) < seed_count
        seed = seeds[mod1(length(population) + 1, length(seeds))]
        # Every island receives independent node buffers.  The repeated seed
        # score is copied, never re-evaluated.
        member = ScoredTerm(TypedTerm(copy(seed.term.nodes)),
            seed.loss, length(population) + 1)
        push!(population, member)
        if seed_outputs isa AbstractVector && !isempty(seed_outputs)
            output = seed_outputs[mod1(length(population), length(seed_outputs))]
            grouping && push!(outputs, output)
            _typed_evaluation_alias!(evaluation_state, member, output)
        end
    end
    # Optional independent material generation is deliberately isolated from
    # the baseline constructor path.  It receives the global budget so an
    # island cannot overrun the caller's attempt/evaluation limits.
    initialization_attempts = 0
    initialization_evaluations = 0
    initialization_invalid_evaluations = 0
    if diverse_initialization
        run isa _TypedNSGA2Run || throw(ArgumentError(
            "diverse initialization requires the NSGA-II run state"))
        objective === nothing && throw(ArgumentError(
            "diverse initialization requires an objective"))
        archive === nothing && throw(ArgumentError(
            "diverse initialization requires an archive"))
        evaluation_state === nothing && grouping && throw(ArgumentError(
            "diverse grouping initialization requires evaluation state"))
        targetid = seeds[1].term.nodes[end].typeid
        while length(population) < config.population_size
            generated = false
            for _ in 1:16
                run.attempts < config.attempts && run.evaluations < config.evaluations || break
                run.attempts += 1
                initialization_attempts += 1
                candidate = if proposal_policy === nothing
                    generate_term(rng, language, targetid;
                        context=Int32[], max_nodes=config.max_nodes,
                        max_depth=config.max_depth, max_attempts=1)
                else
                    generate_toolbox(rng, language, targetid;
                        context=Int32[], policy=proposal_policy,
                        max_nodes=config.max_nodes, max_depth=config.max_depth)
                end
                candidate isa Err && continue
                candidate_size = _typed_written_size(candidate, language)
                candidate_size <= config.max_nodes || continue
                candidate_depth = _search_term_depth(candidate)
                candidate_depth isa Err && continue
                candidate_depth <= config.max_depth || continue
                candidate isa TypedTerm || throw(ArgumentError(
                    "initialization generator must return TypedTerm or Err"))
                score_started = time_ns()
                scored = _typed_nsga2_objective_score(objective, candidate,
                    grouping, evaluation_state === nothing ? nothing : output_length[],
                    evaluation_state)
                run.score_time_seconds += (time_ns() - score_started) / 1.0e9
                if evaluation_state !== nothing && scored.outputs !== nothing &&
                   output_length[] === nothing
                    output_length[] = length(scored.outputs)
                end
                run.evaluations += 1
                initialization_evaluations += 1
                if scored.loss isa Err
                    run.invalid_evaluations += 1
                    initialization_invalid_evaluations += 1
                    continue
                end
                run.birth += 1
                member = ScoredTerm(TypedTerm(copy(candidate.nodes)),
                    scored.loss, run.birth)
                push!(population, member)
                if scored.outputs !== nothing
                    _typed_evaluation_register!(evaluation_state, member,
                        scored.outputs)
                    grouping && push!(outputs, scored.outputs)
                end
                # A successfully generated initialization member participates
                # in the run-wide best-so-far state, but initialization is not
                # a mutation event and must not invoke the improvement hook.
                run.global_best = min(run.global_best, member.loss)
                _record_typed!(archive, member, language; archive_ties=:first)
                generated = true
                break
            end
            generated && continue
            # An admitted scored seed is an owned fallback: no objective call
            # and no raw-attempt fee is invented when finite budgets close.
            seed = seeds[mod1(length(population) + 1, length(seeds))]
            member = ScoredTerm(TypedTerm(copy(seed.term.nodes)),
                seed.loss, length(population) + 1)
            push!(population, member)
            if seed_outputs isa AbstractVector && !isempty(seed_outputs)
                output = seed_outputs[mod1(length(population), length(seed_outputs))]
                grouping && push!(outputs, output)
                _typed_evaluation_alias!(evaluation_state, member, output)
            end
        end
    end
    mating = grouping && group_mating ? _typed_behavior_selection(population, outputs,
        language, family_capacity; rng, classifier) :
        _typed_nsga2_rank_crowding(population, language)
    return _TypedNSGA2Island(population, outputs,
        grouping && group_mating ? mating : nothing, rng, mating.ranks, mating.crowding,
        initialization_attempts, 0, 0, 0, initialization_evaluations,
        initialization_invalid_evaluations, 0, 0, 0,
        initialization_attempts, initialization_evaluations,
        initialization_invalid_evaluations, 0, 0, 0,
        Dict{Symbol,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}(),
        Dict{Symbol,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}())
end

function _typed_nsga2_fitness(population, language::Language)
    fitness = Vector{Tuple{Float64,Float64}}(undef, length(population))
    for (index, member) in enumerate(population)
        size = _typed_written_size(member.term, language)
        fitness[index] = (member.loss, Float64(size))
    end
    return fitness
end

"""Score one candidate for scalar or empirical-output NSGA mode."""
function _typed_nsga2_objective_score(objective, term::TypedTerm,
                                      behavior_grouping::Bool,
                                      expected_output_length,
                                      evaluation_state=nothing)
    !behavior_grouping && evaluation_state === nothing &&
        return (; loss=_typed_loss(objective, term), outputs=nothing)
    raw = objective(term)
    raw isa Err && return (; loss=raw, outputs=nothing)
    raw isa NamedTuple && hasproperty(raw, :loss) ||
        throw(ArgumentError(
            "grouped objective must return (loss=..., outputs=vector) or Err"))
    loss = getproperty(raw, :loss)
    loss isa Real || throw(ArgumentError(
        "behaviour grouping loss must be a real number"))
    loss_value = Float64(loss)
    isfinite(loss_value) || return (; loss=Err(TypedSearchError(:nonfinite_loss)),
        outputs=nothing)
    hasproperty(raw, :outputs) || throw(ArgumentError(
        "grouped objective must return (loss=..., outputs=vector) or Err"))
    raw_outputs = getproperty(raw, :outputs)
    raw_outputs isa AbstractVector || throw(ArgumentError(
        "behaviour grouping outputs must be a vector"))
    expected_output_length === nothing || length(raw_outputs) == expected_output_length ||
        throw(ArgumentError(
            "behaviour output length changed within one fixed batch"))
    output = _typed_validate_output_vector(raw_outputs)
    output isa Err && return (; loss=output, outputs=nothing)
    return (; loss=loss_value, outputs=output)
end

function _typed_nsga2_rank_crowding(population, language::Language)
    fitness = _typed_nsga2_fitness(population, language)
    selected = _typed_nsga2_select(fitness, length(population))
    return (indices=collect(eachindex(population)), ranks=selected.ranks,
        crowding=selected.crowding)
end

function _typed_nsga2_parent_timed(run::_TypedNSGA2Run,
                                   island::_TypedNSGA2Island)
    started = time_ns()
    index = if island.behavior_selection === nothing
        _typed_nsga2_tournament(island.rng, eachindex(island.population),
            island.ranks, island.crowding)
    else
        _typed_behavior_parent(island.rng, island.behavior_selection)
    end
    run.selection_time_seconds += (time_ns() - started) / 1.0e9
    return index
end

function _typed_nsga2_material_choice(rng::AbstractRNG, material_archive)
    material_archive === nothing && return nothing
    groups = material_archive.groups
    isempty(groups) && return nothing
    group = groups[rand(rng, eachindex(groups))]
    members = group.member_indices
    index = length(members) == 1 ? 1 : rand(rng, eachindex(members))
    return material_archive.members[members[index]]
end

function _typed_nsga2_survive(pool::Vector{ScoredTerm}, capacity::Int,
                              language::Language, run::_TypedNSGA2Run;
                              outputs=nothing, rng=nothing, family_capacity::Int=2,
                              classifier=ExactOutputClassifier())
    if outputs !== nothing
        rng isa AbstractRNG || throw(ArgumentError(
            "behaviour survivor selection requires an RNG"))
        started = time_ns()
        result = _typed_behavior_survive(pool, outputs, capacity, language,
            family_capacity, rng; classifier)
        run.selection_time_seconds += (time_ns() - started) / 1.0e9
        return result
    end
    started = time_ns()
    selected = _typed_nsga2_select(_typed_nsga2_fitness(pool, language), capacity)
    run.selection_time_seconds += (time_ns() - started) / 1.0e9
    return (; population=[pool[index] for index in selected.indices],
        ranks=selected.ranks[selected.indices],
        crowding=selected.crowding[selected.indices])
end

function _typed_nsga2_increment!(counts::Dict{Symbol,Int}, kind::Symbol)
    counts[kind] = get(counts, kind, 0) + 1
    return nothing
end

function _typed_nsga2_merge_counts(islands, field::Symbol)
    merged = Dict{Symbol,Int}()
    for island in islands
        for (kind, count) in getfield(island, field)
            merged[kind] = get(merged, kind, 0) + count
        end
    end
    return merged
end

function _typed_nsga2_observe!(run::_TypedNSGA2Run,
                               island::_TypedNSGA2Island,
                               island_id::Int, generation::Int, partial::Bool,
                               archive, language::Language,
                               config)
    config.observer === nothing && return nothing
    config.observer_interval > 0 || return nothing
    run.observer_calls < config.observer_limit || return nothing
    run.attempts >= run.next_observer_attempt || return nothing
    run.next_observer_attempt = run.attempts + config.observer_interval
    snapshot = _typed_snapshot(run.attempts, run.evaluations,
        island.population, archive, language)
    config.observer((; algorithm=:nsga2, selection=:nsga2,
        population_id=island_id, generation, partial_generation=partial,
        snapshot...))
    run.observer_calls += 1
    return nothing
end

function _typed_nsga2_improvement!(run::_TypedNSGA2Run,
                                   parent::ScoredTerm, donor::ScoredTerm,
                                   child::TypedTerm, child_loss::Float64,
                                   child_size::Int, parent_size::Int,
                                   kind::Symbol, island_id::Int,
                                   config;
                                   parent_source::Symbol=:population,
                                   donor_source::Symbol=:population)
    child_loss < run.global_best || return nothing
    previous = run.global_best
    run.global_best = child_loss
    callback = config.improvement_callback
    callback === nothing && return nothing
    run.improvement_calls < config.improvement_limit || return nothing
    event = (; algorithm=:nsga2, population_id=island_id,
        attempt=run.attempts, evaluation_id=run.evaluations, operator=kind,
        parent_source, donor_source,
        parent=TypedTerm(copy(parent.term.nodes)),
        donor=TypedTerm(copy(donor.term.nodes)),
        child=TypedTerm(copy(child.nodes)), parent_loss=parent.loss,
        donor_loss=donor.loss, child_loss, parent_size, child_size,
        evicted_size=nothing, size_delta=child_size - parent_size,
        replacement_size_delta=nothing,
        loss_improvement=parent.loss - child_loss,
        global_improvement=true, global_loss_improvement=previous - child_loss)
    callback(event)
    run.improvement_calls += 1
    return nothing
end

function _typed_nsga2_propose(rng::AbstractRNG, parent::ScoredTerm,
                              donor::ScoredTerm, language::Language,
                              config, kind::Symbol;
                              proposal_policy=nothing)
    if proposal_policy !== nothing
        return propose_toolbox(rng, parent.term, donor.term, language;
            policy=proposal_policy, max_nodes=config.max_nodes,
            max_depth=config.max_depth)
    elseif kind === :simplify
        term = simplify_term(rng, parent.term, language;
            max_nodes=config.max_nodes, max_depth=config.max_depth)
        term isa Err && return term
        return (; term, kind=:simplify, edit_count=1)
    elseif kind === :simplify_atoms
        term = simplify_term(rng, parent.term, language;
            max_nodes=config.max_nodes, max_depth=config.max_depth,
            atomic_beta=true)
        term isa Err && return term
        return (; term, kind=:simplify_atoms, edit_count=1)
    end
    term = propose_term(rng, parent.term, donor.term, language;
        kind, max_nodes=config.max_nodes, max_depth=config.max_depth)
    term isa Err && return term
    return (; term, kind, edit_count=1)
end

"""Score supplied initial seeds with one copied fixed-batch output vector."""
function _typed_nsga2_score_initial(initial, language::Language, objective,
                                    config,
                                    behavior_grouping::Bool;
                                    evaluation_state=nothing)
    seeds = ScoredTerm[]
    seed_outputs = Vector{Vector{Float64}}()
    archive = Dict{Int,ScoredTerm}()
    output_length = Ref{Any}(nothing)
    evaluations = 0
    invalid_evaluations = 0
    root_type = Int32(0)
    attempted_terms = Set{Any}()
    for term in initial
        length(seeds) >= config.population_size && break
        evaluations >= config.evaluations && break
        length(term.nodes) <= config.max_nodes || throw(ArgumentError(
            "seed exceeds node budget"))
        _typed_written_size(term, language) <= config.max_nodes || throw(ArgumentError(
            "seed exceeds written complexity budget"))
        depth = _search_term_depth(term)
        depth isa Err && return depth
        depth <= config.max_depth || throw(ArgumentError("seed exceeds depth budget"))
        checked = validate(term, language)
        checked isa Err && return checked
        root_type == 0 && (root_type = term.nodes[end].typeid)
        term.nodes[end].typeid == root_type || throw(ArgumentError(
            "seed root types differ"))
        syntax_key = term.nodes
        syntax_key in attempted_terms && continue
        push!(attempted_terms, syntax_key)
        scored = _typed_nsga2_objective_score(objective, term,
            behavior_grouping, evaluation_state === nothing ? nothing :
                output_length[], evaluation_state)
        if evaluation_state !== nothing && scored.outputs !== nothing &&
           output_length[] === nothing
            output_length[] = length(scored.outputs)
        end
        evaluations += 1
        if scored.loss isa Err
            invalid_evaluations += 1
            continue
        end
        member = ScoredTerm(TypedTerm(copy(term.nodes)), Float64(scored.loss),
            length(seeds) + 1)
        push!(seeds, member)
        if scored.outputs !== nothing
            _typed_evaluation_register!(evaluation_state, member, scored.outputs)
            push!(seed_outputs, scored.outputs)
        end
        _record_typed!(archive, member, language; archive_ties=:first)
    end
    isempty(seeds) && return Err(TypedSearchError(:no_scored_seed))
    return (; seeds, seed_outputs, archive, evaluations, invalid_evaluations,
        output_length=output_length[])
end

"""
    search_nsga2(initial, language, objective;
                 rng=default_rng(), group_count=1, classifier=nothing,
                 group_mating=true, group_survival=true, group_material=true,
                 evaluation_state=nothing, search_options...)

Run a bounded typed search using generational NSGA-II.  With grouping disabled
and no `evaluation_state`, `objective(term)` must return a finite real loss or
`Err`; the second NSGA-II objective is written complexity.  A classifier
requires
`(loss=finite_real, outputs=finite_nonempty_vector)` or `Err`, with one fixed
output-vector length for this invocation.  Supplying `evaluation_state` enables
the same owned observation capture without changing grouping consumers.  Raw
proposal attempts and objective calls share the same global limits in the
resolved search options.  Every objective call made for an invalid or unchanged
offspring still consumes the evaluation budget; no score is borrowed from a
syntactically equal term.
When a toolbox policy is supplied, its returned `kind` and `edit_count` drive
operator statistics and provenance.  `material_probability` independently
offers parent and (when a donor is used) donor selections from current
best-per-written-size archive material.  A disabled option performs no extra
RNG draw.
"""
function search_nsga2(initial::AbstractVector{<:TypedTerm}, language::Language,
                      objective;
                      rng::AbstractRNG=default_rng(), group_count::Int=1,
                      proposal_policy=nothing,
                      diverse_initialization::Bool=false,
                      material_probability::Real=0.0,
                      family_capacity::Int=2,
                      classifier=nothing,
                      group_mating::Bool=true,
                      group_survival::Bool=true,
                      group_material::Bool=true,
                      evaluation_state=nothing,
                      trace=nothing,
                      search_options...)
    config = _typed_search_options(; search_options...)
    _check_search_config(config, :tournament)
    iszero(config.archive_probability) || throw(ArgumentError(
        "NSGA-II does not support archive_probability; use material_probability"))
    iszero(config.frequency_strength) || throw(ArgumentError(
        "NSGA-II does not support frequency tournaments; frequency_strength must be zero"))
    iszero(config.acceptance_temperature) || throw(ArgumentError(
        "NSGA-II does not support Metropolis acceptance; acceptance_temperature must be zero"))
    isempty(initial) && throw(ArgumentError("typed NSGA-II requires supplied initial material"))
    group_count > 0 || throw(ArgumentError("group_count must be positive"))
    0.0 <= Float64(material_probability) <= 1.0 &&
        isfinite(Float64(material_probability)) || throw(ArgumentError(
            "material_probability must be finite and in [0, 1]"))
    family_capacity >= 1 || throw(ArgumentError("family_capacity must be positive"))
    (classifier === nothing || applicable(classifier, Vector{Float64}())) ||
        throw(ArgumentError("classifier must be callable on a Float64 output vector"))
    grouping = classifier !== nothing
    evaluation_state === nothing && grouping && (evaluation_state = EvaluationState())
    evaluation_state === nothing || evaluation_state isa EvaluationState ||
        throw(ArgumentError("evaluation_state must be EvaluationState or nothing"))
    evaluation_state !== nothing && _typed_evaluation_begin!(evaluation_state)
    _typed_trace_start!(trace)

    score_started = time_ns()
    scored = grouping || evaluation_state !== nothing ?
        _typed_nsga2_score_initial(initial, language, objective, config, grouping;
            evaluation_state) :
        _typed_score_initial(initial, language, objective, config, :tournament;
            archive_ties=:first)
    score_initial_seconds = (time_ns() - score_started) / 1.0e9
    scored isa Err && return scored

    seed_outputs = hasproperty(scored, :seed_outputs) ? scored.seed_outputs : nothing
    output_length = Ref{Any}(hasproperty(scored, :output_length) ?
        scored.output_length : nothing)
    if grouping
        for (member, output) in zip(scored.seeds, seed_outputs)
            trace !== nothing && record_trace_outputs!(trace, member.term, output)
        end
    end
    run = _TypedNSGA2Run(0, scored.evaluations, scored.invalid_evaluations,
        score_initial_seconds, 0.0,
        minimum(member.loss for member in scored.seeds), 0, 0,
        config.observer_interval > 0 ? config.observer_interval : typemax(Int),
        maximum(member.birth for member in scored.seeds))

    rngs = AbstractRNG[]
    if group_count == 1
        push!(rngs, rng)
    else
        for _ in 1:group_count
            push!(rngs, MersenneTwister(rand(rng, UInt)))
        end
    end
    islands = _TypedNSGA2Island[]
    for index in 1:group_count
        selection_started = time_ns()
        push!(islands, _typed_nsga2_island(scored.seeds, config, language,
            rngs[index]; seed_outputs, run, objective, archive=scored.archive,
            evaluation_state, output_length, classifier, group_mating,
            group_survival, group_material, diverse_initialization,
            proposal_policy, family_capacity))
        run.selection_time_seconds += (time_ns() - selection_started) / 1.0e9
    end
    archive = scored.archive
    population_size = config.population_size
    _typed_trace_checkpoint!(trace, :initial, islands; archive, language,
        attempt=run.attempts, evaluation=run.evaluations)

    while run.attempts < config.attempts && run.evaluations < config.evaluations
        made_progress = false
        material_archive = grouping && group_material && material_probability > 0 ?
            _typed_behavior_archive(archive, evaluation_state, language,
                family_capacity, rng; classifier) : nothing
        for (island_id, island) in enumerate(islands)
            run.attempts < config.attempts && run.evaluations < config.evaluations || break

            offspring = ScoredTerm[]
            offspring_outputs = Vector{Vector{Float64}}()
            while length(offspring) < population_size &&
                  run.attempts < config.attempts &&
                  run.evaluations < config.evaluations
                run.attempts += 1
                island.attempts += 1
                parent_index = _typed_nsga2_parent_timed(run, island)
                parent = island.population[parent_index]
                kind = if proposal_policy !== nothing
                    :toolbox
                else
                    _typed_kind(island.rng, config)
                end
                donor = parent
                parent_source = :population
                donor_source = :population
                if material_probability > 0 && rand(island.rng) < material_probability
                    archived = material_archive === nothing ?
                        _typed_archive_sample(island.rng, archive) :
                        _typed_nsga2_material_choice(island.rng, material_archive)
                    if archived !== nothing
                        parent = archived
                        parent_source = :archive
                        island.material_parent_selected += 1
                    end
                end
                if kind === :crossover || kind === :toolbox
                    donor_index = _typed_nsga2_parent_timed(run, island)
                    donor = island.population[donor_index]
                    if material_probability > 0 && rand(island.rng) < material_probability
                        archived = material_archive === nothing ?
                            _typed_archive_sample(island.rng, archive) :
                            _typed_nsga2_material_choice(island.rng, material_archive)
                        if archived !== nothing
                            donor = archived
                            donor_source = :archive
                            island.material_donor_selected += 1
                        end
                    end
                end

                proposed = _typed_nsga2_propose(island.rng, parent, donor,
                    language, config, kind; proposal_policy)
                if proposed isa Err
                    island.proposal_failures += 1
                    failure_kind = kind === :toolbox ? :toolbox_failed : kind
                    _typed_nsga2_increment!(island.kind_attempts, failure_kind)
                    _typed_nsga2_increment!(island.kind_failures, failure_kind)
                    trace !== nothing && _typed_trace_record!(trace;
                        status=:generation_failure, island=island_id,
                        generation=island.generations + island.partial_generations,
                        attempt=run.attempts, evaluation=run.evaluations,
                        operator=failure_kind, parent, donor,
                        parent_source, donor_source,
                        parent_size=_typed_written_size(parent.term, language),
                        donor_size=_typed_written_size(donor.term, language))
                    continue
                end
                proposed isa NamedTuple ||
                    throw(ArgumentError("typed proposal must return a term record"))
                child = proposed.term
                child isa TypedTerm || throw(ArgumentError(
                    "typed proposal must contain a TypedTerm"))
                actual_kind = Symbol(proposed.kind)
                edit_count = Int(proposed.edit_count)
                edit_count >= 0 || throw(ArgumentError(
                    "toolbox edit_count must be nonnegative"))
                _typed_nsga2_increment!(island.kind_attempts, actual_kind)
                island.edit_count += edit_count
                island.kind_edit_counts[actual_kind] =
                    get(island.kind_edit_counts, actual_kind, 0) + edit_count
                child_size = _typed_written_size(child, language)
                if child_size > config.max_nodes
                    island.proposal_failures += 1
                    _typed_nsga2_increment!(island.kind_failures, actual_kind)
                    trace !== nothing && _typed_trace_record!(trace;
                        status=:generation_failure, island=island_id,
                        generation=island.generations + island.partial_generations,
                        attempt=run.attempts, evaluation=run.evaluations,
                        operator=actual_kind, parent, donor, child,
                        parent_source, donor_source,
                        parent_size=_typed_written_size(parent.term, language),
                        donor_size=_typed_written_size(donor.term, language),
                        child_size=child_size)
                    continue
                end
                unchanged_child = _same_term(child, parent.term)
                if unchanged_child
                    island.unchanged += 1
                    _typed_nsga2_increment!(island.kind_unchanged, actual_kind)
                else
                    island.changed += 1
                end

                score_started = time_ns()
                scored_child = _typed_nsga2_objective_score(objective, child,
                    grouping, evaluation_state === nothing ? nothing : output_length[],
                    evaluation_state)
                run.score_time_seconds += (time_ns() - score_started) / 1.0e9
                if evaluation_state !== nothing && scored_child.outputs !== nothing &&
                   output_length[] === nothing
                    output_length[] = length(scored_child.outputs)
                end
                run.evaluations += 1
                island.evaluations += 1
                _typed_nsga2_increment!(island.kind_evaluations, actual_kind)
                if scored_child.loss isa Err
                    run.invalid_evaluations += 1
                    island.invalid_evaluations += 1
                    _typed_nsga2_increment!(island.kind_invalid, actual_kind)
                    trace !== nothing && _typed_trace_record!(trace;
                        status=:evaluation_failure, island=island_id,
                        generation=island.generations + island.partial_generations,
                        attempt=run.attempts, evaluation=run.evaluations,
                        operator=actual_kind, parent, donor, child,
                        parent_source, donor_source,
                        parent_size=_typed_written_size(parent.term, language),
                        donor_size=_typed_written_size(donor.term, language),
                        child_size=child_size)
                    continue
                end

                run.birth += 1
                member = ScoredTerm(TypedTerm(copy(child.nodes)),
                    Float64(scored_child.loss), run.birth)
                push!(offspring, member)
                if scored_child.outputs !== nothing
                    _typed_evaluation_register!(evaluation_state, member,
                        scored_child.outputs)
                    grouping && push!(offspring_outputs, scored_child.outputs)
                    trace !== nothing && record_trace_outputs!(trace, child,
                        scored_child.outputs)
                end
                trace !== nothing && _typed_trace_record!(trace;
                    status=unchanged_child ? :unchanged : :scored,
                    island=island_id,
                    generation=island.generations + island.partial_generations,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=actual_kind, parent, donor, child,
                    child_member=member, parent_source, donor_source,
                    parent_size=_typed_written_size(parent.term, language),
                    donor_size=_typed_written_size(donor.term, language),
                    child_size=child_size, parent_loss=parent.loss,
                    donor_loss=donor.loss, child_loss=member.loss)
                island.offspring += 1
                _record_typed!(archive, member, language; archive_ties=:first)
                _typed_nsga2_improvement!(run, parent, donor, child, member.loss,
                    child_size, _typed_written_size(parent.term, language),
                    actual_kind, island_id, config; parent_source, donor_source)
            end

            isempty(offspring) && continue
            pool = vcat(island.population, offspring)
            pool_outputs = grouping && group_survival ?
                vcat(island.outputs, offspring_outputs) : nothing
            partial = length(offspring) < population_size
            survivors = _typed_nsga2_survive(pool, population_size,
                language, run; outputs=pool_outputs, rng=island.rng,
                family_capacity, classifier)
            island.population = survivors.population
            if grouping
                island.outputs = group_survival ? survivors.outputs :
                    [_typed_evaluation_borrow(evaluation_state, member)
                        for member in survivors.population]
            end
            island.ranks = survivors.ranks
            island.crowding = survivors.crowding
            island.behavior_selection = grouping && group_mating ?
                _typed_behavior_selection(island.population, island.outputs,
                    language, family_capacity; rng=island.rng, classifier) : nothing
            if partial
                island.partial_generations += 1
            else
                island.generations += 1
            end
            made_progress = true
            _typed_nsga2_observe!(run, island, island_id,
                island.generations + island.partial_generations, partial,
                archive, language, config)
            _typed_trace_checkpoint_island!(trace, :generation, island, island_id;
                language, attempt=run.attempts, evaluation=run.evaluations,
                generation=island.generations + island.partial_generations)
            if _typed_trace_due!(trace, run.attempts)
                _typed_trace_checkpoint!(trace, :periodic, islands; archive,
                    language, attempt=run.attempts, evaluation=run.evaluations,
                    generation=island.generations + island.partial_generations)
            end
        end
        evaluation_state !== nothing &&
            _typed_evaluation_prune!(evaluation_state, islands, archive)
        made_progress || break
    end

    island_stats = [
        (; island_id=index, attempts=island.attempts,
            generations=island.generations,
            partial_generations=island.partial_generations,
            offspring=island.offspring, evaluations=island.evaluations,
            invalid_evaluations=island.invalid_evaluations,
            proposal_failures=island.proposal_failures,
            changed=island.changed, unchanged=island.unchanged,
            initialization_attempts=island.initialization_attempts,
            initialization_evaluations=island.initialization_evaluations,
            initialization_invalid_evaluations=island.initialization_invalid_evaluations,
            material_parent_selected=island.material_parent_selected,
            material_donor_selected=island.material_donor_selected,
            edit_count=island.edit_count,
            kind_edit_counts=copy(island.kind_edit_counts),
            kind_attempts=copy(island.kind_attempts),
            kind_failures=copy(island.kind_failures),
            kind_evaluations=copy(island.kind_evaluations),
            kind_invalid=copy(island.kind_invalid),
            kind_unchanged=copy(island.kind_unchanged))
        for (index, island) in enumerate(islands)]
    stats = (; algorithm=:nsga2, selection=:nsga2,
        attempts=run.attempts, evaluations=run.evaluations,
        invalid_evaluations=run.invalid_evaluations,
        initial_evaluations=scored.evaluations,
        initial_invalid_evaluations=scored.invalid_evaluations,
        scored_seeds=length(scored.seeds),
        # Generated diverse-initialization members occupy slots just like
        # scored seeds; only the remaining slots are copied/replicated seeds.
        replicated_seeds=group_count * population_size - length(scored.seeds) -
            sum(island.initialization_evaluations -
                island.initialization_invalid_evaluations for island in islands),
        offspring=sum(island.offspring for island in islands),
        generations=sum(island.generations for island in islands),
        partial_generations=sum(island.partial_generations for island in islands),
        proposal_failures=sum(island.proposal_failures for island in islands),
        changed=sum(island.changed for island in islands),
        unchanged=sum(island.unchanged for island in islands),
        initialization_attempts=sum(island.initialization_attempts for island in islands),
        initialization_evaluations=sum(island.initialization_evaluations for island in islands),
        initialization_invalid_evaluations=sum(
            island.initialization_invalid_evaluations for island in islands),
        material_parent_selected=sum(island.material_parent_selected for island in islands),
        material_donor_selected=sum(island.material_donor_selected for island in islands),
        edit_count=sum(island.edit_count for island in islands),
        kind_edit_counts=_typed_nsga2_merge_counts(islands, :kind_edit_counts),
        score_time_seconds=run.score_time_seconds,
        selection_time_seconds=run.selection_time_seconds,
        budget=(; attempts=config.attempts, evaluations=config.evaluations),
        population_size, group_count, archive_size=length(archive),
        features=(; proposal_toolbox=proposal_policy !== nothing,
            diverse_initialization, material_probability=Float64(material_probability),
            behavior_grouping=grouping, family_capacity,
            group_mating, group_survival, group_material),
        kind_attempts=_typed_nsga2_merge_counts(islands, :kind_attempts),
        kind_failures=_typed_nsga2_merge_counts(islands, :kind_failures),
        kind_evaluations=_typed_nsga2_merge_counts(islands, :kind_evaluations),
        kind_invalid=_typed_nsga2_merge_counts(islands, :kind_invalid),
        kind_unchanged=_typed_nsga2_merge_counts(islands, :kind_unchanged),
        observer_calls=run.observer_calls,
        improvement_calls=run.improvement_calls, island_stats)
    _typed_trace_checkpoint!(trace, :final, islands; archive, language,
        attempt=run.attempts, evaluation=run.evaluations,
        generation=sum(island.generations + island.partial_generations
            for island in islands))
    _typed_trace_finish!(trace)
    return TypedPopulationResult([island.population for island in islands],
        _typed_frontier(archive), stats)
end

export search_nsga2

@testitem "TypedSearch NSGA-II completes and partially commits generations" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    language = Language(table)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    objective = function (term)
        calls[] += 1
        return 0.0
    end
    result = search_nsga2([seed], language, objective;
        (; _typed_search_options(population_size=4, attempts=2, evaluations=3,
            kinds=(:point,), weights=(1.0,))...)..., rng=MersenneTwister(2000))
    @test result isa TypedPopulationResult
    @test calls[] == 3
    @test result.stats.attempts == 2
    @test result.stats.evaluations == 3
    @test result.stats.offspring == 2
    @test result.stats.generations == 0
    @test result.stats.partial_generations == 1
    @test length(result.populations) == 1
    @test length(result.populations[1]) == 4
end

@testitem "TypedSearch NSGA-II scores duplicate offspring and shares seeds" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    objective = function (term)
        calls[] += 1
        return 0.0
    end
    result = search_nsga2([seed], language, objective;
        (; _typed_search_options(population_size=2, attempts=2, evaluations=3,
            kinds=(:point,), weights=(1.0,))...)..., rng=MersenneTwister(1))
    @test result.stats.generations == 1
    @test result.stats.offspring == 2
    @test result.stats.kind_evaluations[:point] == 2
    @test calls[] == 3  # initial plus both duplicate valid proposals

    shared_calls = Ref(0)
    shared = search_nsga2([seed], language, _ -> begin
        shared_calls[] += 1
        0.0
    end; (; _typed_search_options(population_size=2, attempts=0, evaluations=4)...)...,
        rng=MersenneTwister(1111), group_count=3)
    @test shared_calls[] == 1
    @test shared.stats.scored_seeds == 1
    @test shared.stats.replicated_seeds == 5
    @test all(length(population) == 2 for population in shared.populations)
    @test shared.populations[1][1].term.nodes !==
        shared.populations[2][1].term.nodes

    @test_throws ArgumentError search_nsga2([seed], language, _ -> 0.0;
        (; _typed_search_options(population_size=1, attempts=0, evaluations=1,
            archive_probability=0.1)...)...)
end

@testitem "TypedSearch NSGA-II counts invalid offspring without selecting them" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    objective = function (term)
        calls[] += 1
        term.nodes[end].x == one && return 0.0
        return Err(TypedSearchError(:synthetic_invalid))
    end
    result = search_nsga2([seed], language, objective;
        (; _typed_search_options(population_size=2, attempts=2, evaluations=4,
            kinds=(:point,), weights=(1.0,))...)..., rng=MersenneTwister(1234))
    @test calls[] == 3
    @test result.stats.attempts == 2
    @test result.stats.evaluations == 3
    @test result.stats.invalid_evaluations == 2
    @test result.stats.offspring == 0
    @test result.stats.generations == 0
    @test result.stats.partial_generations == 0
    @test all(member.term.nodes[end].x == one for member in result.populations[1])
end

@testitem "TypedSearch NSGA-II accepts higher-order lambda seeds" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    function_type = arrow!(table, scalar, scalar)
    higher_type = arrow!(table, function_type, function_type)
    language = Language(table)
    higher = TypedTerm(TermNode[
        var_node(function_type, 1),
        var_node(scalar, 0),
        app_node(scalar, 1, 2),
        abs_node(function_type, 3),
        abs_node(higher_type, 4),
    ])
    @test validate(higher, language) === true
    result = search_nsga2([higher], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=0, evaluations=1)...)...)
    @test result isa TypedPopulationResult
    @test all(validate(member.term, language) === true
        for member in result.populations[1])
    @test all(member.term.nodes[end].kind == ABS
        for member in result.populations[1])
end

@testitem "TypedSearch NSGA-II retains union crowding for mating" begin
    using Test
    using LambdaRegression.TypedSearch: ScoredTerm,
        _TypedNSGA2Run, _typed_nsga2_fitness, _typed_nsga2_select,
        _typed_nsga2_survive
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    function_type = arrow!(table, scalar, scalar)
    language = Language(table)
    f = add_leaf!(language, function_type, :f)
    a = add_leaf!(language, scalar, :a)
    make_term = function (depth)
        nodes = TermNode[]
        for _ in 1:depth
            push!(nodes, leaf_node(function_type, f))
        end
        push!(nodes, leaf_node(scalar, a))
        argument = length(nodes)
        for level in 1:depth
            function_index = depth - level + 1
            push!(nodes, app_node(scalar, function_index, argument))
            argument = length(nodes)
        end
        return TypedTerm(nodes)
    end
    pool = [ScoredTerm(make_term(depth), Float64(5 - depth), depth + 1)
        for depth in 0:5]
    @test all(validate(member.term, language) === true for member in pool)
    run = _TypedNSGA2Run(0, 0, 0, 0.0, 0.0, Inf, 0, 0, typemax(Int), 0)
    survivors = _typed_nsga2_survive(pool, 3, language, run)
    @test [member.loss for member in survivors.population] == [5.0, 0.0, 4.0]

    union_fitness = _typed_nsga2_fitness(pool, language)
    union_selected = _typed_nsga2_select(union_fitness, 3)
    @test union_selected.indices == [1, 6, 2]
    recomputed = _typed_nsga2_select(
        _typed_nsga2_fitness(survivors.population, language), 3)
    # The third survivor's crowding is computed in the six-member union.  A
    # fresh ranking of P alone would make it an endpoint-normalized interior
    # with a different distance.
    @test survivors.crowding[3] ≈ 0.4
    @test recomputed.crowding[3] ≈ 1.0
    @test survivors.crowding[3] != recomputed.crowding[3]
end

@testitem "TypedSearch NSGA-II feature switches preserve baseline and charge diverse initialization" begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    config = _typed_search_options(population_size=4, attempts=0, evaluations=4,
        max_nodes=1, max_depth=0)
    left_rng = MersenneTwister(8000)
    right_rng = MersenneTwister(8000)
    left = search_nsga2([seed], language, _ -> 0.0; (; config...)..., rng=left_rng)
    right = search_nsga2([seed], language, _ -> 0.0; (; config...)..., rng=right_rng,
        proposal_policy=nothing, diverse_initialization=false,
        material_probability=0.0, family_capacity=2)
    @test [member.term.nodes for member in left.populations[1]] ==
        [member.term.nodes for member in right.populations[1]]
    @test rand(left_rng) == rand(right_rng)
    @test length(left.populations[1]) == 4

    diverse = search_nsga2([seed], language, _ -> 0.0;
        (; _typed_search_options(population_size=3, attempts=2, evaluations=4,
            max_nodes=1, max_depth=0)...)..., rng=MersenneTwister(42),
        diverse_initialization=true)
    @test diverse.stats.attempts == 2
    @test diverse.stats.initialization_attempts == 2
    @test diverse.stats.initialization_evaluations == 2
    @test diverse.stats.evaluations == 3
    @test length(diverse.populations[1]) == 3
    @test diverse.stats.replicated_seeds == 0
end

@testitem "TypedSearch NSGA-II diverse initialization updates global best" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    scored_seed = ScoredTerm(seed, 1.0, 1)
    archive = Dict{Int,ScoredTerm}(1 => scored_seed)
    run = LambdaRegression.TypedSearch._TypedNSGA2Run(
        0, 1, 0, 0.0, 0.0, 1.0, 0, 0, typemax(Int), 1)
    config = _typed_search_options(population_size=2, attempts=1, evaluations=2,
        max_nodes=1, max_depth=0)
    island = LambdaRegression.TypedSearch._typed_nsga2_island(
        [scored_seed], config, language, MersenneTwister(555);
        run, objective=(_ -> 0.0), archive, diverse_initialization=true)
    @test island.initialization_evaluations == 1
    @test run.global_best == 0.0
end

@testitem "TypedSearch NSGA-II material H is scored once and reports source counts" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    result = search_nsga2([seed], language, term -> begin
        calls[] += 1
        0.0
    end; (; _typed_search_options(population_size=1, attempts=1,
        evaluations=2, kinds=(:point,), weights=(1.0,))...)...,
        rng=MersenneTwister(555), material_probability=1.0)
    @test calls[] == 2
    @test result.stats.evaluations == 2
    @test result.stats.material_parent_selected == 1
    @test result.stats.material_donor_selected == 0
    @test result.stats.kind_attempts[:point] == 1
end

@testitem "TypedSearch NSGA-II grouped objective keeps fixed output contract" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed_one = TypedTerm(TermNode[leaf_node(scalar, one)])
    seed_zero = TypedTerm(TermNode[leaf_node(scalar, zero)])
    calls = Ref(0)
    objective = term -> begin
        calls[] += 1
        value = term.nodes[end].x == one ? 1.0 : 2.0
        (loss=0.0, outputs=[value, value + 1.0])
    end
    result = search_nsga2([seed_one, seed_zero], language, objective;
        (; _typed_search_options(population_size=2, attempts=0, evaluations=2)...)...,
        rng=MersenneTwister(2000), classifier=ExactOutputClassifier(),
        family_capacity=1)
    @test calls[] == 2
    @test result.stats.initial_evaluations == 2
    @test length(result.populations[1]) == 2
    @test length(unique(member.term.nodes for member in result.populations[1])) == 2
    independent = search_nsga2([seed_one, seed_zero], language, objective;
        (; _typed_search_options(population_size=2, attempts=0, evaluations=2)...)...,
        rng=MersenneTwister(1), classifier=_ -> :one,
        group_mating=false, group_survival=false, group_material=false)
    @test independent.stats.initial_evaluations == 2
    @test independent.stats.features.behavior_grouping
    @test !independent.stats.features.group_mating
    @test !independent.stats.features.group_survival
    @test !independent.stats.features.group_material
    missing_calls = Ref(0)
    @test_throws ArgumentError search_nsga2([seed_one, seed_zero], language,
        _ -> begin
            missing_calls[] += 1
            missing_calls[] == 1 ?
                (loss=0.0, outputs=[1.0]) : (loss=0.0, outputs=nothing)
        end;
        (; _typed_search_options(population_size=2, attempts=0, evaluations=2)...)...,
        classifier=ExactOutputClassifier())
    mismatch_calls = Ref(0)
    @test_throws ArgumentError search_nsga2([seed_one], language,
        _ -> begin
            mismatch_calls[] += 1
            mismatch_calls[] == 1 ?
                (loss=0.0, outputs=[1.0, 2.0]) :
                (loss=0.0, outputs=[1.0])
        end;
        (; _typed_search_options(population_size=1, attempts=1, evaluations=2,
            kinds=(:point,), weights=(1.0,))...)...,
        classifier=ExactOutputClassifier())
end

@testitem "TypedSearch cache-only observations do not change scoring" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaf = add_leaf!(language, scalar, :x)
    seed = TypedTerm(TermNode[leaf_node(scalar, leaf)])
    calls = Ref(0)
    buffers = Vector{Vector{Float64}}()
    objective = term -> begin
        calls[] += 1
        buffer = [Float64(length(term.nodes)), -0.0]
        push!(buffers, buffer)
        (loss=0.0, outputs=buffer)
    end
    state = EvaluationState()
    result = search_nsga2([seed], language, objective;
        (; _typed_search_options(population_size=2, attempts=0, evaluations=1)...)...,
        rng=MersenneTwister(42),
        evaluation_state=state)
    @test calls[] == result.stats.evaluations == 1
    member = result.populations[1][1]
    cached = evaluation_outputs(state, member)
    buffers[1][1] = 88.0
    @test evaluation_outputs(state, member)[1] != 88.0
    cached[1] = 99.0
    @test evaluation_outputs(state, member)[1] != 99.0
    @test_throws ArgumentError search_nsga2([seed], language, objective;
        (; _typed_search_options(population_size=1, attempts=0, evaluations=1)...)...,
        evaluation_state=state)
end

@testitem "TypedSearch NSGA-II composes toolbox, H, empirical G, and eta" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    function_type = arrow!(language.types, scalar, scalar)
    f = add_leaf!(language, function_type, :f)
    seed = TypedTerm(TermNode[leaf_node(function_type, f)])
    policy = TypedProposalPolicy(weights=(generate=0.0, crossover=1.0,
        atomic_rewire=0.0, operator=0.0, insert=0.0, hoist=0.0,
        point=0.0, simplify=0.0), double_edit_probability=0.0)
    toolbox_result = search_nsga2([seed], language,
        term -> (loss=0.0, outputs=[Float64(length(term.nodes)), 1.0]);
        (; _typed_search_options(population_size=1, attempts=1, evaluations=2)...)...,
        rng=MersenneTwister(1), proposal_policy=policy,
        material_probability=1.0, classifier=ExactOutputClassifier(),
        family_capacity=2)
    @test toolbox_result isa TypedPopulationResult
    @test toolbox_result.stats.attempts == 1
    @test toolbox_result.stats.evaluations == 2
    @test toolbox_result.stats.material_parent_selected == 1
    @test toolbox_result.stats.material_donor_selected == 1
    @test toolbox_result.stats.kind_attempts[:crossover] == 1
    @test validate(toolbox_result.populations[1][1].term, language) === true

    eta_result = search_nsga2([seed], language,
        term -> (loss=0.0, outputs=[Float64(length(term.nodes)), 1.0]);
        (; _typed_search_options(population_size=1, attempts=1, evaluations=2,
            kinds=(:eta_expand,), weights=(1.0,))...)...,
        rng=MersenneTwister(1111), proposal_policy=nothing,
        material_probability=1.0, classifier=ExactOutputClassifier(),
        family_capacity=2)
    @test eta_result isa TypedPopulationResult
    @test eta_result.stats.attempts == 1
    @test eta_result.stats.evaluations == 2
    @test eta_result.stats.kind_attempts[:eta_expand] == 1
    @test validate(eta_result.populations[1][1].term, language) === true
end
