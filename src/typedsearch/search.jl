using Random: AbstractRNG, MersenneTwister, default_rng, rand

"""
    _typed_search_options(; population_size=32, attempts=1000, evaluations=1000, ...)

A bounded regularized-evolution pilot: rank tournaments select parents, every
valid changed child replaces the oldest member, and a best-per-size archive
retains tradeoffs. This follows the age-replacement/population pattern already
used by ScalarSearch, not a new task-specific optimiser. The first slice uses
one population, no constant refinement, and no hidden teacher initialisation.
Budgets include initial objective calls; proposal failures also consume attempts.
The default `kinds` and weights retain the four ordinary variation operators;
callers may explicitly add `:simplify`, `:simplify_atoms`, `:insert`,
`:operator`, `:eta_reduce`, or `:eta_expand` to request those bounded opt-in
proposals; the default tuple and weights remain unchanged.  `:simplify_atoms`
enables the opt-in
atomic-argument β rule while retaining the ordinary identity and unused-binder
rules.  `:operator` replaces one applied atomic function head with an exact
same-`typeid` leaf or visible local while preserving its argument subtrees.
`archive_probability` is opt-in: when nonzero, each of the parent and
donor can be replaced by a uniformly sampled occupied size bucket from the
best-per-size archive. The live population remains age-replaced; the archive
is not copied wholesale into it.

`frequency_strength=0` preserves this trajectory and accepts arbitrary finite
real losses.  A positive strength enables a bounded written-complexity
frequency penalty in breeding tournaments only; it requires finite nonnegative
losses and `frequency_window >= max_nodes`.  The tournament key is
`log(loss) + frequency_strength * f`, with zero loss ordered separately.
Raw losses, archive ranking, and age replacement are unchanged.

Setting `observer_interval` enables sparse population snapshots in the result;
`observer`, when supplied, receives the same read-only-by-convention snapshots.
Both are bounded by `observer_limit`.
`improvement_callback` receives at most `improvement_limit` copied
parent/donor/child records when an evaluated child strictly improves the
global best. Each record includes the operator, evaluation id,
binding-preserving terms, source (`:population` or `:archive`), written sizes,
and loss deltas.
Neither callback receives the search RNG, and telemetry itself makes no random
draws.
Per-operator statistics retain accepted-child loss transitions: improvement,
degradation, and equality counts. Nonnegative magnitude sums may overflow to
`Inf`; overflow counts concern individual loss differences, not accumulation.
When enabled, `acceptance_rejected` counts valid evaluated proposals rejected by
the fixed-temperature gate; accepted loss transitions remain an accepted-only
partition.

`search_terms` and `search_populations` accept `selection=:tournament` (the
default path) or the opt-in `selection=:automatic_epsilon_lexicase`.  The
lexicase objective must return one finite, non-negative error per case; its
mean is the ordinary archive/population loss, while the private case matrix is
used only for live parent and crossover-donor selection.  Case vectors are
scored once per objective call and copied with replicated initial seeds.  The
lexicase mode requires `frequency_strength=0`; it does not combine two
selection rankings.  `archive_ties=:first` preserves the first member at an
occupied written-size bucket when losses are exactly equal.  The opt-in
`archive_ties=:newest` replaces that bucket only for exactly equal finite
losses; strict improvements retain their existing behavior and worse losses
are ignored.  This choice does not affect population replacement or RNG use.
`hof_migration_interval=0` disables periodic Pareto-front migration. When
enabled, each persistent population counts its own raw attempts and, at every
interval, samples a bounded Poisson number of fresh copies from the shared
training-loss frontier into uniformly selected live destinations. Migration
does not consume objective calls or mutation-acceptance counts. Its optional
`hof_migration_fraction` controls the Poisson mean as a fraction of population
size; direct migration and selection counters are reported per population and
in aggregate statistics.
"""
function _typed_search_options(;
                               population_size::Integer=32,
                               attempts::Integer=1000,
                               evaluations::Integer=1000,
                               tournament_size::Integer=5,
                               selection_probability::Real=0.98,
                               max_nodes::Integer=128,
                               max_depth::Integer=32,
                               kinds::Tuple{Vararg{Symbol}}=(:generate, :crossover,
                                   :hoist, :point),
                               weights::Tuple=(0.35, 0.35,
                                   0.15, 0.15),
                               archive_probability::Real=0.0,
                               frequency_strength::Real=0.0,
                               frequency_window::Integer=4096,
                               acceptance_temperature::Real=0.0,
                               loss_normalization::Real=1.0,
                               observer::Union{Nothing,Function}=nothing,
                               observer_interval::Integer=0,
                               observer_limit::Integer=64,
                               improvement_callback::Union{Nothing,Function}=nothing,
                               improvement_limit::Integer=32)
    return (; population_size=Int(population_size), attempts=Int(attempts),
        evaluations=Int(evaluations), tournament_size=Int(tournament_size),
        selection_probability=Float64(selection_probability),
        max_nodes=Int(max_nodes), max_depth=Int(max_depth), kinds,
        weights=Tuple(Float64(weight) for weight in weights),
        archive_probability=Float64(archive_probability),
        frequency_strength=Float64(frequency_strength),
        frequency_window=Int(frequency_window),
        acceptance_temperature=Float64(acceptance_temperature),
        loss_normalization=Float64(loss_normalization), observer,
        observer_interval=Int(observer_interval), observer_limit=Int(observer_limit),
        improvement_callback, improvement_limit=Int(improvement_limit))
end

struct ScoredTerm
    term::TypedTerm
    loss::Float64
    birth::Int
end

struct TypedSearchResult
    population::Vector{ScoredTerm}
    frontier::Vector{ScoredTerm}
    stats::NamedTuple
end

struct TypedPopulationResult
    populations::Vector{Vector{ScoredTerm}}
    frontier::Vector{ScoredTerm}
    stats::NamedTuple
end


function _check_search_config(c, selection::Symbol=:tournament)
    c.population_size > 0 && c.attempts >= 0 && c.evaluations > 0 &&
        c.tournament_size > 0 && c.max_nodes > 0 && c.max_depth >= 0 ||
        throw(ArgumentError("invalid typed-search budgets"))
    0 < c.selection_probability <= 1 || throw(ArgumentError("invalid tournament probability"))
    0 <= c.archive_probability <= 1 ||
        throw(ArgumentError("invalid archive probability"))
    isfinite(c.frequency_strength) && c.frequency_strength >= 0 ||
        throw(ArgumentError("invalid frequency strength"))
    c.frequency_window >= 1 || throw(ArgumentError("invalid frequency window"))
    c.frequency_strength == 0 || c.frequency_window >= c.max_nodes ||
        throw(ArgumentError("frequency_window must be at least max_nodes when frequency is enabled"))
    isfinite(c.acceptance_temperature) && c.acceptance_temperature >= 0 ||
        throw(ArgumentError("invalid acceptance temperature"))
    isfinite(c.loss_normalization) && c.loss_normalization > 0 ||
        throw(ArgumentError("invalid loss normalization"))
    c.observer_interval >= 0 || throw(ArgumentError("invalid observer interval"))
    c.observer_limit >= 0 || throw(ArgumentError("invalid observer limit"))
    c.improvement_limit >= 0 || throw(ArgumentError("invalid improvement limit"))
    length(c.kinds) == length(c.weights) && !isempty(c.kinds) ||
        throw(ArgumentError("mutation kinds and weights must match"))
    all(k -> k in (:generate, :crossover, :hoist, :point, :insert, :operator,
                   :simplify, :simplify_atoms, :eta_reduce, :eta_expand), c.kinds) ||
        throw(ArgumentError("unknown typed mutation kind"))
    all(w -> isfinite(w) && w >= 0, c.weights) && sum(c.weights) > 0 ||
        throw(ArgumentError("invalid mutation weights"))
    selection in (:tournament, :automatic_epsilon_lexicase) ||
        throw(ArgumentError("unknown typed selection mode"))
    selection === :automatic_epsilon_lexicase && c.frequency_strength > 0 &&
        throw(ArgumentError(
            "automatic epsilon lexicase requires frequency_strength=0",
        ))
    return nothing
end

function _check_archive_ties(archive_ties::Symbol)
    archive_ties in (:first, :newest) || throw(ArgumentError(
        "archive_ties must be :first or :newest"))
    return archive_ties
end

function _check_hof_migration(hof_migration_interval::Integer,
                              hof_migration_fraction::Real)
    hof_migration_interval >= 0 || throw(ArgumentError(
        "hof_migration_interval must be nonnegative"))
    hof_migration_interval <= typemax(Int) || throw(ArgumentError(
        "hof_migration_interval is too large"))
    fraction = Float64(hof_migration_fraction)
    isfinite(fraction) && 0.0 <= fraction <= 1.0 || throw(ArgumentError(
        "hof_migration_fraction must be in [0, 1]"))
    return (Int(hof_migration_interval), fraction)
end

_term_key(member::ScoredTerm, language::Language) =
    (member.loss, _typed_written_size(member.term, language), length(member.term.nodes))
_same_term(a::TypedTerm, b::TypedTerm) = a.nodes == b.nodes

# Donor-backed terms may carry a written complexity that deliberately differs
# from their compact node count. The donor module owns this authoritative
# measure; ordinary compact terms still cost one per node.
function _typed_written_size(term::TypedTerm, language::Language)
    # Ordinary compact nodes have exactly one written unit each. Avoid making
    # the donor helper revalidate long syntax chains in the hot loop; only a
    # SourceProgram leaf can hide recursively expanded written material.
    has_program_leaf = any(node -> node.kind == LEAF_KIND &&
        language.leaves[Int(node.x)].payload isa SourceProgram, term.nodes)
    return has_program_leaf ? Int(written_complexity(term, language)) : length(term.nodes)
end

function _search_term_depth(term)
    isempty(term.nodes) && return Err(TypedSearchError(:empty_term))
    depths = zeros(Int, length(term.nodes))
    for (i, node) in enumerate(term.nodes)
        if node.kind == ABS || node.kind == APP
            1 <= node.x < i || return Err(TypedSearchError(:bad_child_index, (i, node.x)))
        end
        if node.kind == APP
            1 <= node.y < i || return Err(TypedSearchError(:bad_child_index, (i, node.y)))
        end
        depths[i] = node.kind == ABS ? 1 + depths[node.x] :
            node.kind == APP ? 1 + max(depths[node.x], depths[node.y]) : 0
    end
    return depths[end]
end

function _typed_loss(objective, term)
    loss = objective(term)
    loss isa Err && return loss
    loss isa Real || throw(ArgumentError("typed objective must return a real loss or Err"))
    value = Float64(loss)
    return isfinite(value) ? value : Err(TypedSearchError(:nonfinite_loss))
end

function _typed_case_score(objective, term, case_count::Int)
    raw = objective(term)
    raw isa Err && return (raw, nothing, case_count)
    raw isa AbstractVector || throw(ArgumentError(
        "automatic epsilon lexicase objective must return an error vector or Err",
    ))
    observed = length(raw)
    observed > 0 || throw(ArgumentError(
        "automatic epsilon lexicase objective returned an empty error vector",
    ))
    case_count == 0 || observed == case_count || throw(ArgumentError(
        "automatic epsilon lexicase case count changed: expected $case_count, got $observed",
    ))
    errors = Vector{Float64}(undef, observed)
    for (index, error) in enumerate(raw)
        error isa Real || throw(ArgumentError(
            "automatic epsilon lexicase errors must be real, got $(typeof(error))",
        ))
        value = Float64(error)
        isfinite(value) || return (
            Err(TypedSearchError(:nonfinite_case_error, (index, value))),
            nothing,
            observed,
        )
        value >= 0.0 || throw(ArgumentError(
            "automatic epsilon lexicase errors must be non-negative",
        ))
        errors[index] = value
    end
    total = sum(errors)
    isfinite(total) || return (
        Err(TypedSearchError(:nonfinite_case_error, total)), nothing, observed,
    )
    return (total / observed, errors, observed)
end

function _typed_objective_score(objective, term, selection::Symbol,
                                case_count::Int)
    selection === :tournament && return (_typed_loss(objective, term), nothing, 0)
    return _typed_case_score(objective, term, case_count)
end

function _typed_frequency_key(member::ScoredTerm, language::Language,
                               frequency::_RegularizedFrequency, strength::Float64)
    loss = member.loss
    size = _typed_written_size(member.term, language)
    # Zero is a distinct best-loss case. Positive losses use log-space, so the
    # optional frequency term never needs an exponential or a loss shift.
    rank = loss == 0.0 ? 0 : 1
    score = rank == 0 ? 0.0 : log(loss) + strength *
        _regularized_frequency(frequency, size)
    return (rank, score, size, length(member.term.nodes))
end

function _typed_tournament(rng, population, config, language::Language,
                           frequency=nothing)
    sample = rand(rng, eachindex(population), config.tournament_size)
    if frequency === nothing
        sort!(sample; by=i -> _term_key(population[i], language))
    else
        sort!(sample; by=i -> _typed_frequency_key(
            population[i], language, frequency, config.frequency_strength))
    end
    for i in sample[1:end-1]
        rand(rng) < config.selection_probability && return population[i]
    end
    return population[sample[end]]
end

function _typed_select_parent(rng, population, config, language::Language,
                              frequency, selection::Symbol, case_errors)
    if selection === :automatic_epsilon_lexicase
        case_errors isa Matrix{Float64} || throw(ArgumentError(
            "automatic epsilon lexicase requires live case errors",
        ))
        return population[_typed_epsilon_lexicase_index(rng, case_errors)]
    end
    return _typed_tournament(rng, population, config, language, frequency)
end

"""Observe live written-complexity occupancy in the shared bounded window."""
function _typed_observe_frequency!(frequency::_RegularizedFrequency,
                                   population, language::Language)
    for member in population
        complexity = _typed_written_size(member.term, language)
        0 < complexity <= length(frequency.frequencies) || continue
        frequency.frequencies[complexity] += 1.0
    end
    _regularized_move_window!(frequency)
    return _regularized_normalize!(frequency)
end

function _typed_frequency_summary(frequency, strength::Float64)
    frequency === nothing && return nothing
    observed_sizes = Int[]
    for (size, value) in enumerate(frequency.frequencies)
        value > 1.0 + 1.0e-12 && push!(observed_sizes, size)
    end
    peak = argmax(frequency.normalized_frequencies)
    return (; enabled=true, strength, window=frequency.window,
        total=sum(frequency.frequencies), observed_sizes,
        peak_size=peak, peak_frequency=frequency.normalized_frequencies[peak])
end

function _typed_acceptance_probability(parent_loss::Float64, child_loss::Float64,
                                      config, kind::Symbol)
    enabled = config.acceptance_temperature > 0 &&
        (kind === :generate || kind === :point || kind === :hoist ||
         kind === :insert || kind === :operator)
    enabled || return 1.0
    isfinite(parent_loss) && isfinite(child_loss) || throw(ArgumentError(
        "Metropolis acceptance requires finite parent and child losses"))
    delta = child_loss - parent_loss
    isnan(delta) && throw(ArgumentError("Metropolis loss difference is NaN"))
    delta <= 0 && return 1.0
    normalized_delta = delta / config.loss_normalization /
        config.acceptance_temperature
    if !isfinite(normalized_delta)
        # Float64 subtraction/division may overflow even when the final
        # dimensionless exponent is small (for example, ±floatmax divided by
        # floatmax).  BigFloat is only an exceptional-path recovery here.
        normalized_delta = setprecision(BigFloat, 256) do
            Float64((BigFloat(child_loss) - BigFloat(parent_loss)) /
                BigFloat(config.loss_normalization) /
                BigFloat(config.acceptance_temperature))
        end
    end
    return _regularized_acceptance_probability(0.0, normalized_delta;
        annealing=true, temperature=1.0, alpha=1.0)
end

function _typed_kind(rng, config)
    threshold = rand(rng) * sum(config.weights)
    for (kind, weight) in zip(config.kinds, config.weights)
        threshold -= weight
        threshold < 0 && return kind
    end
    return config.kinds[end]
end

function _record_typed!(archive, member, language::Language;
                        archive_ties::Symbol=:first,
                        archive_case_errors=nothing, case_errors=nothing)
    size = _typed_written_size(member.term, language)
    old = get(archive, size, nothing)
    equal_finite = old !== nothing && isfinite(member.loss) &&
        isfinite(old.loss) && member.loss == old.loss
    if old === nothing || member.loss < old.loss ||
       (archive_ties === :newest && equal_finite)
        archive[size] = member
        if archive_case_errors !== nothing
            case_errors isa Vector{Float64} || throw(ArgumentError(
                "lexicase archive migration requires a case vector"))
            archive_case_errors[size] = copy(case_errors)
        end
    end
    return member
end

"""Sample `min(Poisson(mean), upper)` with bounded Poisson arrivals.

The arrival-clock form avoids `exp(-mean)` underflow in the usual Knuth loop
and is bounded by `upper`, already the maximum number of migrants that can be
inserted into one population. It consumes no draws when either bound is zero.
"""
function _typed_hof_poisson(rng::AbstractRNG, mean::Float64, upper::Int)
    isfinite(mean) && mean >= 0.0 || throw(ArgumentError(
        "HOF migration mean must be finite and nonnegative"))
    upper >= 0 || throw(ArgumentError("HOF migration bound must be nonnegative"))
    (iszero(mean) || iszero(upper)) && return 0
    elapsed = 0.0
    for count in 1:upper
        elapsed += -log(rand(rng)) / mean
        elapsed > 1.0 && return count - 1
    end
    return upper
end

function _typed_archive_sample(rng::AbstractRNG, archive)
    isempty(archive) && return nothing
    # Sorting makes a fixed RNG stream reproducible across Dict iteration
    # orders. Sampling is by occupied size bucket, not by loss-first rank.
    sizes = sort!(collect(keys(archive)))
    return archive[sizes[rand(rng, 1:length(sizes))]]
end

function _typed_frontier(archive)
    frontier = ScoredTerm[]
    best_loss = Inf
    for size in sort!(collect(keys(archive)))
        member = archive[size]
        member.loss < best_loss || continue
        push!(frontier, member)
        best_loss = member.loss
    end
    return frontier
end

mutable struct _TypedOperatorStats
    attempts::Int
    failures::Int
    changed::Int
    evaluated::Int
    accepted::Int
    size_delta_sum::Int
    size_delta_count::Int
    parent_size_sum::Int
    parent_size_count::Int
    evicted_size_sum::Int
    evicted_size_count::Int
    replacement_delta_sum::Int
    replacement_delta_count::Int
    loss_improvement_sum::Float64
    loss_improvement_count::Int
    loss_improvement_overflow_count::Int
    loss_degradation_sum::Float64
    loss_degradation_count::Int
    loss_degradation_overflow_count::Int
    loss_equal_count::Int
    acceptance_rejected::Int
end

_TypedOperatorStats() = _TypedOperatorStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0,
    0, 0.0, 0, 0, 0, 0)

mutable struct _TypedPopulationState
    population::Vector{ScoredTerm}
    rng::AbstractRNG
    frequency::Union{Nothing,_RegularizedFrequency}
    birth::Int
    accepted_since_frequency::Int
    attempts::Int
    score_requests::Int
    initialization_attempts::Int
    initialization_evaluations::Int
    initialization_invalid_evaluations::Int
    initialization_fallbacks::Int
    evaluations::Int
    invalid_evaluations::Int
    changed::Int
    failures::Int
    unchanged::Int
    repeats::Int
    accepted::Int
    acceptance_rejected::Int
    archive_parent_feedback::Int
    archive_donor_feedback::Int
    hof_migration_events::Int
    hof_migration_insertions::Int
    hof_migration_parent_selected::Int
    hof_migration_donor_selected::Int
    migrant_births::Set{Int}
    kind_counts::Dict{Symbol,Int}
    errors::Dict{Symbol,Int}
    operator_stats::Dict{Symbol,_TypedOperatorStats}
    telemetry::Vector{NamedTuple}
    case_errors::Union{Nothing,Matrix{Float64}}
end

mutable struct _TypedPopulationRun
    evaluations::Int
    invalid_evaluations::Int
    attempts::Int
    global_best::Float64
    observer_calls::Int
    improvement_calls::Int
    last_improvement_event::Any
    last_improvement_emitted::Int
    score_requests::Int
    budget_mode::Symbol
end

"""Record one selected proposal kind and return its mutable counters.

Toolbox dispatch does not know its concrete kind until a proposal succeeds;
callers therefore record `:toolbox` only for a dispatch failure and record the
returned concrete kind exactly once for a successful proposal.  Keeping this
small bookkeeping seam separate from proposal construction also lets other
search front ends use the same no-phantom-bucket rule.
"""
@inline function _typed_record_proposal!(state::_TypedPopulationState,
                                         kind::Symbol)
    state.kind_counts[kind] = get(state.kind_counts, kind, 0) + 1
    counters = get!(state.operator_stats, kind, _TypedOperatorStats())
    counters.attempts += 1
    return counters
end

function _check_budget_mode(budget_mode::Symbol)
    budget_mode in (:objective_calls, :candidate_requests) || throw(ArgumentError(
        "budget_mode must be :objective_calls or :candidate_requests"))
    return budget_mode
end

@inline function _typed_budget_available(run::_TypedPopulationRun,
                                         config)
    return run.attempts < config.attempts &&
        (run.budget_mode === :objective_calls ?
         run.evaluations < config.evaluations :
         run.score_requests < config.evaluations)
end

@inline function _typed_request!(run::_TypedPopulationRun,
                                 state::_TypedPopulationState)
    run.score_requests += 1
    state.score_requests += 1
    return nothing
end

function _typed_operator_stats(stats)
    return Dict(kind => (; attempts=s.attempts, failures=s.failures,
        changed=s.changed, evaluated=s.evaluated, accepted=s.accepted,
        size_delta_sum=s.size_delta_sum, size_delta_count=s.size_delta_count,
        size_delta_mean=s.size_delta_count == 0 ? 0.0 :
            s.size_delta_sum / s.size_delta_count,
        parent_size_sum=s.parent_size_sum, parent_size_count=s.parent_size_count,
        parent_size_mean=s.parent_size_count == 0 ? 0.0 :
            s.parent_size_sum / s.parent_size_count,
        evicted_size_sum=s.evicted_size_sum, evicted_size_count=s.evicted_size_count,
        evicted_size_mean=s.evicted_size_count == 0 ? 0.0 :
            s.evicted_size_sum / s.evicted_size_count,
        replacement_delta_sum=s.replacement_delta_sum,
        replacement_delta_count=s.replacement_delta_count,
        replacement_delta_mean=s.replacement_delta_count == 0 ? 0.0 :
            s.replacement_delta_sum / s.replacement_delta_count,
        loss_improvement_sum=s.loss_improvement_sum,
        loss_improvement_count=s.loss_improvement_count,
        loss_improvement_overflow_count=s.loss_improvement_overflow_count,
        loss_improvement_mean=s.loss_improvement_count == 0 ? 0.0 :
            s.loss_improvement_sum / s.loss_improvement_count,
        loss_degradation_sum=s.loss_degradation_sum,
        loss_degradation_count=s.loss_degradation_count,
        loss_degradation_overflow_count=s.loss_degradation_overflow_count,
        loss_equal_count=s.loss_equal_count,
        acceptance_rejected=s.acceptance_rejected)
        for (kind, s) in stats)
end

function _typed_distinct_count(population)
    # The set is deliberately ephemeral: callers do not mutate member terms
    # during this observation, and no mutable key state escapes the call.
    seen = Set{Vector{TermNode}}()
    for member in population
        push!(seen, member.term.nodes)
    end
    return length(seen)
end

function _typed_snapshot(attempt, evaluations, population, archive, language;
                         snapshot_archive_terms::Bool=false)
    sizes = [_typed_written_size(member.term, language) for member in population]
    size_summary = (; min=minimum(sizes), max=maximum(sizes),
        mean=sum(sizes) / length(sizes))
    best_loss_by_size = Dict(size => archive[size].loss
        for size in sort!(collect(keys(archive))))
    snapshot = (; attempt, evaluations, population_size=length(population),
        distinct_count=_typed_distinct_count(population), size_summary,
        best_loss_by_size)
    snapshot_archive_terms || return snapshot
    # Copying every occupied archive bucket is opt-in; disabled snapshots do
    # not allocate term copies.  The bounded worst case is
    # O(observer_limit * max_nodes^2) space across retained snapshots.
    archive_members = [(; size, loss=archive[size].loss,
        term=TypedTerm(copy(archive[size].term.nodes)))
        for size in sort!(collect(keys(archive)))]
    return (; snapshot..., archive_members)
end

function _typed_score_initial(initial, language::Language, objective,
                              config, selection::Symbol;
                              archive_ties::Symbol=:first,
                              archive_case_vectors::Bool=false,
                              budget_mode::Symbol=:objective_calls)
    _check_budget_mode(budget_mode)
    frequency_enabled = config.frequency_strength > 0
    seeds = ScoredTerm[]
    seed_case_errors = Vector{Vector{Float64}}()
    archive = Dict{Int,ScoredTerm}()
    archive_case_errors = archive_case_vectors ? Dict{Int,Vector{Float64}}() : nothing
    evaluations = 0
    score_requests = 0
    invalid_evaluations = 0
    case_count = 0
    root_type = Int32(0)
    attempted_terms = Set{Any}()
    for term in initial
        length(seeds) >= config.population_size && break
        score_requests >= config.evaluations && break
        length(term.nodes) <= config.max_nodes ||
            throw(ArgumentError("seed exceeds node budget"))
        _typed_written_size(term, language) <= config.max_nodes ||
            throw(ArgumentError("seed exceeds written complexity budget"))
        depth = _search_term_depth(term)
        depth isa Err && return depth
        depth <= config.max_depth || throw(ArgumentError("seed exceeds depth budget"))
        checked = validate(term, language)
        checked isa Err && return checked
        root_type == 0 && (root_type = term.nodes[end].typeid)
        term.nodes[end].typeid == root_type || throw(ArgumentError("seed root types differ"))
        syntax_key = Tuple(term.nodes)
        syntax_key in attempted_terms && continue
        push!(attempted_terms, syntax_key)
        loss, errors, observed_case_count = _typed_objective_score(
            objective, term, selection, case_count)
        if selection === :automatic_epsilon_lexicase && observed_case_count > 0
            case_count == 0 && (case_count = observed_case_count)
            case_count == observed_case_count || throw(ArgumentError(
                "automatic epsilon lexicase case count changed: expected $case_count, got $observed_case_count",
            ))
        end
        evaluations += 1
        score_requests += 1
        if loss isa Err
            invalid_evaluations += 1
            continue
        end
        frequency_enabled && loss < 0 &&
            throw(ArgumentError("frequency tournament requires nonnegative losses"))
        member = ScoredTerm(TypedTerm(copy(term.nodes)), Float64(loss), length(seeds) + 1)
        push!(seeds, member)
        selection === :automatic_epsilon_lexicase &&
            push!(seed_case_errors, errors::Vector{Float64})
        _record_typed!(archive, member, language; archive_ties,
            archive_case_errors=archive_case_errors,
            case_errors=(selection === :automatic_epsilon_lexicase ?
                errors : nothing))
    end
    isempty(seeds) && return Err(TypedSearchError(:no_scored_seed))
    return (; seeds, seed_case_errors, case_count, archive, archive_case_errors,
        evaluations, score_requests, invalid_evaluations)
end

function _typed_make_population_state(seeds, seed_case_errors, case_count::Int,
                                      config,
                                      language::Language, rng::AbstractRNG,
                                      selection::Symbol;
                                      diverse_initialization::Bool=false)
    population = ScoredTerm[]
    scored_seeds = length(seeds)
    seed_limit = diverse_initialization ?
        min(scored_seeds, config.population_size) : config.population_size
    case_errors = selection === :automatic_epsilon_lexicase ?
        Matrix{Float64}(undef, case_count, config.population_size) : nothing
    while length(population) < seed_limit
        population_index = length(population) + 1
        seed_index = mod1(population_index, scored_seeds)
        seed = seeds[seed_index]
        push!(population, ScoredTerm(TypedTerm(copy(seed.term.nodes)), seed.loss,
            population_index))
        case_errors === nothing || (case_errors[:, population_index] =
            seed_case_errors[seed_index])
    end
    frequency = config.frequency_strength > 0 ?
        _RegularizedFrequency(config.frequency_window, config.max_nodes) : nothing
    frequency === nothing || diverse_initialization ||
        _typed_observe_frequency!(frequency, population, language)
    operator_stats = Dict{Symbol,_TypedOperatorStats}(kind => _TypedOperatorStats()
        for kind in config.kinds)
    return _TypedPopulationState(
        population, rng, frequency,
        length(population), # birth
        0, # accepted_since_frequency
        0, # attempts
        0, # score_requests
        0, # initialization_attempts
        0, # initialization_evaluations
        0, # initialization_invalid_evaluations
        0, # initialization_fallbacks
        0, # evaluations
        0, # invalid_evaluations
        0, # changed
        0, # failures
        0, # unchanged
        0, # repeats
        0, # accepted
        0, # acceptance_rejected
        0, # archive_parent_feedback
        0, # archive_donor_feedback
        0, # hof_migration_events
        0, # hof_migration_insertions
        0, # hof_migration_parent_selected
        0, # hof_migration_donor_selected
        Set{Int}(), # migrant_births
        Dict{Symbol,Int}(), Dict{Symbol,Int}(), operator_stats, NamedTuple[],
        case_errors)
end

"""Fill the unseeded part of one age-replacement island independently.

Each slot receives at most sixteen generator attempts.  Structural and
written-budget failures consume only the shared raw-attempt budget; a
candidate is charged as a logical score request immediately before the
objective seam.  If the finite budgets close, an admitted seed is copied into
the slot without an attempt or score.  Initialization updates the shared
archive and best-so-far loss but deliberately emits no improvement callback.
"""
function _typed_diverse_initialize!(state::_TypedPopulationState,
                                    run::_TypedPopulationRun, seeds,
                                    seed_case_errors, case_count::Int,
                                    archive, archive_case_errors,
                                    language::Language, objective,
                                    config,
                                    selection::Symbol;
                                    proposal_policy=nothing,
                                    archive_ties::Symbol=:first,
                                    trace=nothing, population_id::Int=0)
    targetid = seeds[1].term.nodes[end].typeid
    while length(state.population) < config.population_size
        admitted = false
        for _ in 1:16
            _typed_budget_available(run, config) || break
            run.attempts += 1
            state.attempts += 1
            state.initialization_attempts += 1
            candidate = proposal_policy === nothing ?
                generate_term(state.rng, language, targetid;
                    context=Int32[], max_nodes=config.max_nodes,
                    max_depth=config.max_depth, max_attempts=1) :
                generate_toolbox(state.rng, language, targetid;
                    context=Int32[], policy=proposal_policy,
                    max_nodes=config.max_nodes, max_depth=config.max_depth)
            if candidate isa Err
                trace !== nothing && _typed_trace_record!(trace;
                    status=:generation_failure, island=population_id,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=:initialization)
                continue
            end
            candidate isa TypedTerm || throw(ArgumentError(
                "initialization generator must return TypedTerm or Err"))
            candidate_size = _typed_written_size(candidate, language)
            if candidate_size > config.max_nodes
                trace !== nothing && _typed_trace_record!(trace;
                    status=:generation_failure, island=population_id,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=:initialization)
                continue
            end
            depth = _search_term_depth(candidate)
            if depth isa Err || depth > config.max_depth
                trace !== nothing && _typed_trace_record!(trace;
                    status=:generation_failure, island=population_id,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=:initialization)
                continue
            end
            if validate(candidate, language) !== true
                trace !== nothing && _typed_trace_record!(trace;
                    status=:generation_failure, island=population_id,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=:initialization)
                continue
            end

            # This is the objective seam: in candidate-request mode the
            # request is charged even when an implementation later chooses
            # not to retain a candidate.  Initialization scores every legal
            # generated candidate, matching NSGA-II's recipe.
            _typed_request!(run, state)
            loss, errors, observed_case_count = _typed_objective_score(
                objective, candidate, selection, case_count)
            selection === :automatic_epsilon_lexicase &&
                observed_case_count != case_count && throw(ArgumentError(
                    "automatic epsilon lexicase case count changed: expected $case_count, got $observed_case_count"))
            run.evaluations += 1
            state.evaluations += 1
            state.initialization_evaluations += 1
            if loss isa Err
                trace !== nothing && _typed_trace_record!(trace;
                    status=:evaluation_failure, island=population_id,
                    attempt=run.attempts, evaluation=run.evaluations,
                    operator=:initialization, child=candidate,
                    child_size=candidate_size)
                run.invalid_evaluations += 1
                state.invalid_evaluations += 1
                state.initialization_invalid_evaluations += 1
                continue
            end
            frequency = state.frequency
            frequency === nothing || loss >= 0 || throw(ArgumentError(
                "frequency tournament requires nonnegative losses"))
            state.birth += 1
            member = ScoredTerm(TypedTerm(copy(candidate.nodes)), Float64(loss),
                state.birth)
            push!(state.population, member)
            state.case_errors === nothing || (state.case_errors[:, length(
                state.population)] = errors::Vector{Float64})
            _record_typed!(archive, member, language; archive_ties,
                archive_case_errors, case_errors=errors)
            trace !== nothing && _typed_trace_record!(trace;
                status=:scored, island=population_id, attempt=run.attempts,
                evaluation=run.evaluations, operator=:initialization,
                child=candidate, child_size=candidate_size,
                child_loss=member.loss, child_slot=length(state.population))
            run.global_best = min(run.global_best, member.loss)
            admitted = true
            break
        end
        admitted && continue
        # Finite fallback is free and uses only already-admitted scored seeds.
        seed = seeds[mod1(length(state.population) + 1, length(seeds))]
        state.birth += 1
        push!(state.population, ScoredTerm(TypedTerm(copy(seed.term.nodes)),
            seed.loss, state.birth))
        state.case_errors === nothing || (state.case_errors[:, length(
            state.population)] = seed_case_errors[mod1(length(state.population),
                length(seed_case_errors))])
        state.initialization_fallbacks += 1
    end
    state.frequency === nothing || _typed_observe_frequency!(state.frequency,
        state.population, language)
    return nothing
end

"""Inject a bounded Pareto-front sample into one persistent population.

Sources and destinations are sampled uniformly with replacement. Each copied
term receives a fresh birth clock and, in lexicase mode, the case vector stored
with the exact archive size-bucket representative. Migration is a population
change only: it performs no objective call and does not increment mutation
acceptance counters. The live frequency state is observed after insertions so
the next tournament sees the copied material immediately.
"""
function _typed_hof_migrate!(state::_TypedPopulationState, archive,
                             archive_case_errors, language::Language,
                             fraction::Float64; trace=nothing,
                             island_id::Int=0, attempt::Int=0,
                             evaluation::Int=0)
    state.hof_migration_events += 1
    frontier = _typed_frontier(archive)
    limit = min(length(state.population), length(frontier))
    count = _typed_hof_poisson(state.rng,
        Float64(length(state.population)) * fraction, limit)
    for _ in 1:count
        source = frontier[rand(state.rng, eachindex(frontier))]
        destination = rand(state.rng, eachindex(state.population))
        old = state.population[destination]
        delete!(state.migrant_births, old.birth)
        state.birth += 1
        copied = ScoredTerm(TypedTerm(copy(source.term.nodes)), source.loss,
            state.birth)
        state.population[destination] = copied
        push!(state.migrant_births, copied.birth)
        trace !== nothing && _typed_trace_record!(trace; kind=:migration,
            status=:scored, island=island_id, attempt, evaluation,
            operator=:hof_migration, parent=source, parent_source=:archive,
            parent_size=_typed_written_size(source.term, language),
            parent_loss=source.loss, child=copied.term, child_member=copied,
            child_slot=destination, evicted=old, evicted_slot=destination,
            child_size=_typed_written_size(copied.term, language),
            child_loss=copied.loss)
        if state.case_errors !== nothing
            archive_case_errors isa Dict{Int,Vector{Float64}} || throw(
                ArgumentError("lexicase HOF migration lacks archive case vectors"))
            source_size = _typed_written_size(source.term, language)
            source_errors = get(archive_case_errors, source_size, nothing)
            source_errors isa Vector{Float64} || throw(ArgumentError(
                "lexicase HOF migration lacks the selected archive case vector"))
            state.case_errors[:, destination] = source_errors
        end
    end
    state.hof_migration_insertions += count
    count == 0 || state.frequency === nothing ||
        _typed_observe_frequency!(state.frequency, state.population, language)
    return count
end

function _typed_attempt!(state::_TypedPopulationState, run::_TypedPopulationRun,
                         archive, language::Language, objective, config,
                         population_id::Int, attempt_id::Int,
                         selection::Symbol, case_count::Int;
                         archive_ties::Symbol=:first,
                         archive_case_errors=nothing,
                         proposal_policy=nothing,
                         trace=nothing,
                         generation::Int=0)
    population = state.population
    rng = state.rng
    frequency = state.frequency
    parent = _typed_select_parent(rng, population, config, language, frequency,
        selection, state.case_errors)
    toolbox_enabled = proposal_policy !== nothing
    kind = toolbox_enabled ? :toolbox : _typed_kind(rng, config)
    donor = parent
    donor_source = :population
    # The toolbox may choose crossover internally, so it receives an actual
    # donor up front just as the NSGA-II route does. The ordinary proposal
    # mixture draws a donor only for crossover.
    if kind === :crossover || toolbox_enabled
        donor = _typed_select_parent(rng, population, config, language, frequency,
            selection, state.case_errors)
    end
    parent_source = :population
    if config.archive_probability > 0
        if rand(rng) < config.archive_probability
            archived = _typed_archive_sample(rng, archive)
            if archived !== nothing
                parent = archived
                parent_source = :archive
                state.archive_parent_feedback += 1
            end
        end
        if (kind === :crossover || toolbox_enabled) &&
           rand(rng) < config.archive_probability
            archived = _typed_archive_sample(rng, archive)
            if archived !== nothing
                donor = archived
                donor_source = :archive
                state.archive_donor_feedback += 1
            end
        end
    end
    if parent_source === :population && parent.birth in state.migrant_births
        state.hof_migration_parent_selected += 1
    end
    if (kind === :crossover || toolbox_enabled) &&
       donor_source === :population &&
       donor.birth in state.migrant_births
        state.hof_migration_donor_selected += 1
    end
    child_size = 0
    child = nothing
    if toolbox_enabled
        proposed = propose_toolbox(rng, parent.term, donor.term, language;
            policy=proposal_policy, max_nodes=config.max_nodes,
            max_depth=config.max_depth)
        if proposed isa Err
            # A failed toolbox dispatch has no concrete operation to report;
            # retain the historical :toolbox failure bucket.
            kind = :toolbox
            opstats = _typed_record_proposal!(state, kind)
            child = proposed
        else
            proposed isa NamedTuple ||
                throw(ArgumentError("toolbox proposal must return a term record"))
            actual_kind = Symbol(proposed.kind)
            # Successful toolbox proposals carry their concrete selected kind;
            # record it directly so successful runs have no phantom bucket.
            kind = actual_kind
            opstats = _typed_record_proposal!(state, kind)
            child = proposed.term
        end
    else
        opstats = _typed_record_proposal!(state, kind)
        child = kind === :simplify ?
            simplify_term(rng, parent.term, language;
                max_nodes=config.max_nodes, max_depth=config.max_depth) :
            kind === :simplify_atoms ?
            simplify_term(rng, parent.term, language;
                max_nodes=config.max_nodes, max_depth=config.max_depth,
                atomic_beta=true) :
            propose_term(rng, parent.term, donor.term, language;
                kind, max_nodes=config.max_nodes, max_depth=config.max_depth)
    end
    child_errors = nothing
    trace_status = :unknown
    child_loss = NaN
    parent_size = 0
    evicted_size = 0
    replacement_slot = 0
    if child isa Err
        trace_status = :generation_failure
        state.failures += 1
        opstats.failures += 1
        reason = child.value isa TypedSearchError ? child.value.reason : :proposal_error
        state.errors[reason] = get(state.errors, reason, 0) + 1
    elseif !(child isa TypedTerm)
        throw(ArgumentError("typed proposal must return TypedTerm or Err"))
    elseif ((child_size = _typed_written_size(child, language)) > config.max_nodes)
        trace_status = :generation_failure
        state.failures += 1
        opstats.failures += 1
        reason = :written_size_budget_exceeded
        state.errors[reason] = get(state.errors, reason, 0) + 1
    elseif (_typed_request!(run, state); _same_term(child, parent.term))
        trace_status = :unchanged
        state.unchanged += 1
    else
        state.changed += 1
        opstats.changed += 1
        parent_size = _typed_written_size(parent.term, language)
        opstats.size_delta_sum += child_size - parent_size
        opstats.size_delta_count += 1
        if any(m -> _same_term(m.term, child), population)
            trace_status = :live_duplicate
            state.repeats += 1
        else
            loss, child_errors, observed_case_count = _typed_objective_score(
                objective, child, selection, case_count)
            if selection === :automatic_epsilon_lexicase &&
               observed_case_count != case_count
                throw(ArgumentError(
                    "automatic epsilon lexicase case count changed: expected $case_count, got $observed_case_count",
                ))
            end
            run.evaluations += 1
            state.evaluations += 1
            opstats.evaluated += 1
            if loss isa Err
                trace_status = :evaluation_failure
                run.invalid_evaluations += 1
                state.invalid_evaluations += 1
            else
                child_loss = Float64(loss)
                frequency === nothing || child_loss >= 0 ||
                    throw(ArgumentError("frequency tournament requires nonnegative losses"))
                if config.acceptance_temperature > 0 &&
                   (kind === :generate || kind === :point || kind === :hoist ||
                    kind === :insert || kind === :operator)
                    probability = _typed_acceptance_probability(
                        parent.loss, child_loss, config, kind)
                    if rand(rng) >= probability
                        trace_status = :rejected
                        state.acceptance_rejected += 1
                        opstats.acceptance_rejected += 1
                        trace !== nothing && _typed_trace_record!(trace;
                            status=trace_status, island=population_id,
                            generation, attempt=attempt_id,
                            evaluation=run.evaluations, operator=kind,
                            parent, donor,
                            child=child isa TypedTerm ? child : nothing,
                            parent_source, donor_source,
                            parent_slot=0, donor_slot=0, child_slot=0,
                            parent_size, child_size,
                            donor_size=_typed_written_size(donor.term, language),
                            parent_loss=parent.loss, donor_loss=donor.loss,
                            child_loss)
                        return nothing
                    end
                end
                improvement = parent.loss - child_loss
                if improvement > 0
                    opstats.loss_improvement_count += 1
                    isfinite(improvement) || (opstats.loss_improvement_overflow_count += 1)
                    opstats.loss_improvement_sum += improvement
                elseif improvement < 0
                    opstats.loss_degradation_count += 1
                    degradation = child_loss - parent.loss
                    isfinite(degradation) || (opstats.loss_degradation_overflow_count += 1)
                    opstats.loss_degradation_sum += degradation
                else
                    opstats.loss_equal_count += 1
                end
                global_improvement = child_loss < run.global_best
                global_loss_improvement = run.global_best - child_loss
                oldest = argmin(map(m -> m.birth, population))
                replacement_slot = oldest
                evicted_member = population[oldest]
                evicted_size = _typed_written_size(population[oldest].term, language)
                opstats.parent_size_sum += parent_size
                opstats.parent_size_count += 1
                opstats.evicted_size_sum += evicted_size
                opstats.evicted_size_count += 1
                opstats.replacement_delta_sum += child_size - evicted_size
                opstats.replacement_delta_count += 1
                delete!(state.migrant_births, population[oldest].birth)
                state.birth += 1
                member = ScoredTerm(child, child_loss, state.birth)
                population[oldest] = member
                state.case_errors === nothing || (state.case_errors[:, oldest] =
                    child_errors::Vector{Float64})
                state.accepted += 1
                opstats.accepted += 1
                run.global_best = min(run.global_best, child_loss)
                _record_typed!(archive, member, language; archive_ties,
                    archive_case_errors=archive_case_errors,
                    case_errors=child_errors)
                if frequency !== nothing
                    state.accepted_since_frequency += 1
                    if state.accepted_since_frequency >= config.population_size
                        _typed_observe_frequency!(frequency, population, language)
                        state.accepted_since_frequency = 0
                    end
                end
                if config.improvement_callback !== nothing && global_improvement
                    event = (; population_id, attempt=attempt_id,
                        evaluation_id=run.evaluations,
                        score_request_id=run.score_requests, operator=kind,
                        parent_source, donor_source,
                        parent=TypedTerm(copy(parent.term.nodes)),
                        donor=TypedTerm(copy(donor.term.nodes)),
                        child=TypedTerm(copy(child.nodes)),
                        parent_loss=parent.loss, donor_loss=donor.loss,
                        child_loss, parent_size, child_size, evicted_size,
                        size_delta=child_size - parent_size,
                        replacement_size_delta=child_size - evicted_size,
                        loss_improvement=improvement,
                        global_improvement, global_loss_improvement)
                    run.last_improvement_event = event
                    early_limit = max(0, config.improvement_limit - 1)
                    if run.improvement_calls < early_limit
                        config.improvement_callback(event)
                        run.improvement_calls += 1
                        run.last_improvement_emitted = run.evaluations
                    end
                end
                trace_status = :scored
                trace !== nothing && _typed_trace_record!(trace;
                    status=trace_status, island=population_id,
                    generation, attempt=attempt_id, evaluation=run.evaluations,
                    operator=kind, parent, donor, child=child,
                    child_member=member, parent_source, donor_source,
                    parent_slot=0, donor_slot=0, child_slot=replacement_slot,
                    evicted=evicted_member, evicted_slot=replacement_slot,
                    parent_size, donor_size=_typed_written_size(donor.term, language),
                    child_size, parent_loss=parent.loss, donor_loss=donor.loss,
                    child_loss)
            end
        end
    end
    if trace !== nothing && trace_status !== :scored
        parent_slot = findfirst(m -> m.term.nodes === parent.term.nodes, population)
        donor_slot = findfirst(m -> m.term.nodes === donor.term.nodes, population)
        _typed_trace_record!(trace; status=trace_status, island=population_id,
            generation, attempt=attempt_id, evaluation=run.evaluations,
            operator=kind, parent, donor,
            child=child isa TypedTerm ? child : nothing, parent_source,
            donor_source, parent_slot=parent_slot === nothing ? 0 : parent_slot,
            donor_slot=donor_slot === nothing ? 0 : donor_slot,
            child_slot=replacement_slot, parent_size,
            donor_size=_typed_written_size(donor.term, language), child_size,
            parent_loss=parent.loss, donor_loss=donor.loss, child_loss)
    end
    return nothing
end

function _typed_emit_snapshot!(state::_TypedPopulationState, run::_TypedPopulationRun,
                               archive, language::Language, config,
                               population_id::Int,
                               snapshot_archive_terms::Bool=false)
    run.observer_calls >= config.observer_limit && return false
    snapshot0 = _typed_snapshot(run.attempts, run.evaluations,
        state.population, archive, language; snapshot_archive_terms)
    snapshot = (; population_id, snapshot0...)
    config.observer === nothing || config.observer(snapshot)
    push!(state.telemetry, snapshot)
    run.observer_calls += 1
    return true
end

function _typed_merge_operator_stats(states)
    merged = Dict{Symbol,_TypedOperatorStats}()
    for state in states
        for (kind, source) in state.operator_stats
            target = get!(merged, kind, _TypedOperatorStats())
            for field in fieldnames(_TypedOperatorStats)
                setfield!(target, field, getfield(target, field) + getfield(source, field))
            end
        end
    end
    return _typed_operator_stats(merged)
end

function _typed_merge_counts(states, field::Symbol)
    result = Dict{Symbol,Int}()
    for state in states
        for (key, value) in getfield(state, field)
            result[key] = get(result, key, 0) + value
        end
    end
    return result
end

function _typed_population_stats(state::_TypedPopulationState, language::Language,
                                 config, population_id::Int,
                                 selection::Symbol, case_count::Int,
                                 budget_mode::Symbol)
    return (; population_id, attempts=state.attempts, changed=state.changed,
        failures=state.failures, unchanged=state.unchanged,
        live_population_repeats=state.repeats, accepted=state.accepted,
        acceptance_rejected=state.acceptance_rejected,
        score_requests=state.score_requests,
        evaluations=state.evaluations, invalid_evaluations=state.invalid_evaluations,
        initialization_attempts=state.initialization_attempts,
        initialization_evaluations=state.initialization_evaluations,
        initialization_invalid_evaluations=state.initialization_invalid_evaluations,
        initialization_fallbacks=state.initialization_fallbacks,
        kind_counts=state.kind_counts, errors=state.errors,
        archive_parent_feedback=state.archive_parent_feedback,
        archive_donor_feedback=state.archive_donor_feedback,
        hof_migration_events=state.hof_migration_events,
        hof_migration_insertions=state.hof_migration_insertions,
        hof_migration_parent_selected=state.hof_migration_parent_selected,
        hof_migration_donor_selected=state.hof_migration_donor_selected,
        operator_stats=_typed_operator_stats(state.operator_stats),
        frequency_summary=_typed_frequency_summary(state.frequency,
        config.frequency_strength), selection, case_count, budget_mode)
end

function _typed_population_run_stats(states, run, scored_seeds, initial_evaluations,
                                     initial_invalid_evaluations,
                                     language::Language, config,
                                     selection::Symbol, case_count::Int,
                                     diverse_initialization::Bool=false)
    telemetry = NamedTuple[]
    for state in states
        append!(telemetry, state.telemetry)
    end
    sort!(telemetry; by=s -> (s.attempt, s.population_id))
    population_stats = [_typed_population_stats(state, language, config, i,
        selection, case_count, run.budget_mode)
        for (i, state) in enumerate(states)]
    frequencies = [stats.frequency_summary for stats in population_stats]
    kind_counts = _typed_merge_counts(states, :kind_counts)
    errors = _typed_merge_counts(states, :errors)
    return (; attempts=run.attempts, changed=sum(s.changed for s in states),
        failures=sum(s.failures for s in states), unchanged=sum(s.unchanged for s in states),
        live_population_repeats=sum(s.repeats for s in states),
        score_requests=run.score_requests,
        evaluations=run.evaluations, invalid_evaluations=run.invalid_evaluations,
        accepted=sum(s.accepted for s in states),
        acceptance_rejected=sum(s.acceptance_rejected for s in states), scored_seeds,
        replicated_seeds=diverse_initialization ?
            length(states) * config.population_size - scored_seeds -
                sum(s.initialization_evaluations -
                    s.initialization_invalid_evaluations for s in states) :
            length(states) * config.population_size - scored_seeds,
        initial_evaluations, initial_invalid_evaluations, kind_counts, errors,
        budget_mode=run.budget_mode,
        initialization_attempts=sum(s.initialization_attempts for s in states),
        initialization_evaluations=sum(s.initialization_evaluations for s in states),
        initialization_invalid_evaluations=sum(
            s.initialization_invalid_evaluations for s in states),
        initialization_fallbacks=sum(s.initialization_fallbacks for s in states),
        archive_parent_feedback=sum(s.archive_parent_feedback for s in states),
        archive_donor_feedback=sum(s.archive_donor_feedback for s in states),
        hof_migration_events=sum(s.hof_migration_events for s in states),
        hof_migration_insertions=sum(s.hof_migration_insertions for s in states),
        hof_migration_parent_selected=sum(
            s.hof_migration_parent_selected for s in states),
        hof_migration_donor_selected=sum(
            s.hof_migration_donor_selected for s in states),
        operator_stats=_typed_merge_operator_stats(states), telemetry,
        observer_calls=run.observer_calls, improvement_calls=run.improvement_calls,
        frequency_summary=length(states) == 1 ? frequencies[1] : nothing,
        frequency_summaries=frequencies, population_stats, selection, case_count,
        group_count=length(states))
end

"""
    search_populations(initial, language, objective;
                       rng=default_rng(), group_count=1, chunk_attempts=1,
                       selection=:tournament, archive_ties=:first,
                       hof_migration_interval=0, hof_migration_fraction=0.0614,
                       proposal_policy=nothing, diverse_initialization=false,
                       budget_mode=:objective_calls, trace=nothing,
                       search_options...)

Run one shared-budget search across persistent age-replacement populations.
Initial seeds are scored once and copied explicitly into each population. Raw
proposal attempts are assigned round-robin in chunks; objective calls are
counted globally, including invalid calls, and never exceed the evaluation
budget supplied through `search_options...`.
Each population retains private birth, RNG, frequency, and accepted-remainder
state. A shared best-per-written-size archive supplies optional feedback, and
the returned frontier is computed once from that complete archive. With
`selection=:automatic_epsilon_lexicase`, each population additionally retains
one bounded case-error matrix for live parent/donor selection; archive members
are still ranked by mean loss and written size. The public result exposes
members and per-population statistics, not mutable RNG state. Passing
`snapshot_archive_terms=true` adds an `archive_members` field to each emitted
snapshot, containing detached `(size, loss, term)` records for occupied archive
size buckets. This is intentionally opt-in because copying terms is bounded by
the observer limit and can require O(observer_limit * max_nodes²) space; the
default snapshot shape and trajectory are unchanged.
`archive_ties=:newest` is an independent opt-in archive policy: an exactly
equal finite loss replaces the member already occupying that written-size
bucket. The default `:first` policy keeps the existing member; strict
improvements and worse losses are handled identically by both policies.
`hof_migration_interval` is measured separately for each persistent population
in raw attempts. At each positive interval, a bounded Poisson number with mean
`population_size * hof_migration_fraction` copies members from the shared
training-loss frontier into uniformly selected destinations. Copies have fresh
birth clocks and do not consume objective calls, attempts, or accepted-child
statistics. With lexicase selection, the exact archived case vector follows
each size-bucket representative, including `archive_ties=:newest` replacement.
When `proposal_policy` is supplied, parent and donor selection follows the
same toolbox route as NSGA-II, including an up-front donor for every toolbox
request. `diverse_initialization=true` independently fills each island's
unseeded slots with at most sixteen bounded generation attempts. The default
`budget_mode=:objective_calls` bounds physical objective calls; opt-in
`:candidate_requests` bounds logical requests at the objective seam and
reports both counters. All search controls are direct keywords and are
resolved once at this entry point.
"""
function search_populations(initial::AbstractVector{<:TypedTerm}, language::Language,
                            objective;
                            rng::AbstractRNG=default_rng(), group_count::Int=1,
                            chunk_attempts::Int=1,
                            selection::Symbol=:tournament,
                            snapshot_archive_terms::Bool=false,
                            archive_ties::Symbol=:first,
                            hof_migration_interval::Integer=0,
                            hof_migration_fraction::Real=0.0614,
                            proposal_policy=nothing,
                            diverse_initialization::Bool=false,
                            budget_mode::Symbol=:objective_calls,
                            trace=nothing,
                            search_options...)
    config = _typed_search_options(; search_options...)
    _check_search_config(config, selection)
    _check_budget_mode(budget_mode)
    _check_archive_ties(archive_ties)
    hof_interval, hof_fraction = _check_hof_migration(
        hof_migration_interval, hof_migration_fraction)
    isempty(initial) && throw(ArgumentError("typed search requires supplied initial material"))
    group_count > 0 || throw(ArgumentError("group_count must be positive"))
    chunk_attempts > 0 || throw(ArgumentError("chunk_attempts must be positive"))
    _typed_trace_start!(trace)
    scored = _typed_score_initial(initial, language, objective, config, selection;
        archive_ties, archive_case_vectors=(selection ===
            :automatic_epsilon_lexicase && hof_interval > 0), budget_mode)
    scored isa Err && return scored
    archive = scored.archive
    scored_seeds = length(scored.seeds)
    rngs = AbstractRNG[]
    if group_count == 1
        push!(rngs, rng)
    else
        for _ in 1:group_count
            push!(rngs, MersenneTwister(rand(rng, UInt)))
        end
    end
    states = [_typed_make_population_state(scored.seeds, scored.seed_case_errors,
        scored.case_count, config, language, rngs[i], selection;
        diverse_initialization)
        for i in 1:group_count]
    run = _TypedPopulationRun(scored.evaluations, scored.invalid_evaluations, 0,
        minimum(member.loss for member in scored.seeds), 0, 0, nothing, 0,
        scored.score_requests, budget_mode)
    if diverse_initialization
        for (population_id, state) in enumerate(states)
            _typed_diverse_initialize!(state, run, scored.seeds,
                scored.seed_case_errors, scored.case_count, archive,
                scored.archive_case_errors, language, objective, config,
                selection; proposal_policy, archive_ties, trace, population_id)
        end
    end
    _typed_trace_checkpoint!(trace, :initial, states; archive, language,
        attempt=run.attempts, evaluation=run.evaluations)
    while _typed_budget_available(run, config)
        population_id = mod1(fld(run.attempts, chunk_attempts) + 1, group_count)
        state = states[population_id]
        for _ in 1:chunk_attempts
            _typed_budget_available(run, config) || break
            run.attempts += 1
            state.attempts += 1
            _typed_attempt!(state, run, archive, language, objective, config,
                population_id, run.attempts, selection, scored.case_count;
                archive_ties, archive_case_errors=scored.archive_case_errors,
                proposal_policy, trace)
            if hof_interval > 0 && state.attempts % hof_interval == 0
                _typed_hof_migrate!(state, archive, scored.archive_case_errors,
                    language, hof_fraction; trace, island_id=population_id,
                    attempt=run.attempts, evaluation=run.evaluations)
            end
            if _typed_trace_due!(trace, run.attempts)
                _typed_trace_checkpoint!(trace, :periodic, states; archive, language,
                    attempt=run.attempts, evaluation=run.evaluations)
            end
            if config.observer_interval > 0 &&
               run.attempts % config.observer_interval == 0
                _typed_emit_snapshot!(state, run, archive, language, config,
                    population_id, snapshot_archive_terms)
            end
        end
    end
    if config.observer_interval > 0 && run.attempts > 0 &&
       run.attempts % config.observer_interval != 0
        for (population_id, state) in enumerate(states)
            run.observer_calls >= config.observer_limit && break
            _typed_emit_snapshot!(state, run, archive, language, config,
                population_id, snapshot_archive_terms)
        end
    end
    if config.improvement_callback !== nothing && config.improvement_limit > 0 &&
       run.last_improvement_event !== nothing &&
       run.last_improvement_event.evaluation_id != run.last_improvement_emitted
        config.improvement_callback(run.last_improvement_event)
        run.improvement_calls += 1
    end
    frontier = _typed_frontier(archive)
    _typed_trace_checkpoint!(trace, :final, states; archive, language,
        attempt=run.attempts, evaluation=run.evaluations)
    _typed_trace_finish!(trace)
    stats = _typed_population_run_stats(states, run, scored_seeds,
        scored.evaluations, scored.invalid_evaluations, language, config,
        selection, scored.case_count, diverse_initialization)
    return TypedPopulationResult([state.population for state in states], frontier, stats)
end

"""
    search_terms(initial, language, objective;
                 rng=default_rng(), selection=:tournament, archive_ties=:first,
                 hof_migration_interval=0, hof_migration_fraction=0.0614,
                 proposal_policy=nothing, diverse_initialization=false,
                 budget_mode=:objective_calls, trace=nothing,
                 search_options...)

Run a finite, single-objective population phase over supplied typed terms.
`objective(term)` returns a finite real loss or `Err`; unexpected exceptions
propagate. With `selection=:automatic_epsilon_lexicase`, it instead returns
one finite, non-negative error per case; the mean is used as the ordinary
archive loss. Archive feedback, sparse observer snapshots, bounded provenance,
and frequency tournaments are controlled by direct keywords. This wrapper is
exactly one `search_populations` group and retains the `TypedSearchResult`
shape. The returned frontier is the shared best-per-size archive frontier for
that one group. The `snapshot_archive_terms` keyword is forwarded unchanged;
when true, only emitted snapshots gain detached archive-member term copies.
`archive_ties` is forwarded unchanged and accepts the same `:first`/`:newest`
archive tie policy.
"""
function search_terms(initial::AbstractVector{<:TypedTerm}, language::Language, objective;
                      rng::AbstractRNG=default_rng(),
                      selection::Symbol=:tournament,
                      snapshot_archive_terms::Bool=false,
                      archive_ties::Symbol=:first,
                      hof_migration_interval::Integer=0,
                      hof_migration_fraction::Real=0.0614,
                      proposal_policy=nothing,
                      diverse_initialization::Bool=false,
                      budget_mode::Symbol=:objective_calls,
                      trace=nothing,
                      search_options...)
    result = search_populations(initial, language, objective;
        rng=rng, group_count=1, chunk_attempts=1,
        selection=selection, snapshot_archive_terms=snapshot_archive_terms,
        archive_ties=archive_ties,
        hof_migration_interval=hof_migration_interval,
        hof_migration_fraction=hof_migration_fraction,
        proposal_policy=proposal_policy,
        diverse_initialization=diverse_initialization,
        budget_mode=budget_mode, trace=trace, search_options...)
    result isa Err && return result
    return TypedSearchResult(result.populations[1], result.frontier, result.stats)
end

@testitem "TypedSearch lexicase keeps case columns aligned on replacement" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch: _typed_attempt!, _TypedPopulationRun,
        _typed_make_population_state, _typed_score_initial
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    objective = function(term)
        calls[] += 1
        leaf = term.nodes[end]
        leaf.x == one && return [1.0, 0.0]
        leaf.x == zero && return [0.0, 1.0]
        return Err(TypedSearchError(:unexpected_test_term))
    end
    config = _typed_search_options(population_size=2, attempts=2, evaluations=8,
        kinds=(:point,), weights=(1.0,))
    scored = _typed_score_initial([seed], language, objective, config,
        :automatic_epsilon_lexicase)
    state = _typed_make_population_state(scored.seeds, scored.seed_case_errors,
        scored.case_count, config, language, MersenneTwister(42),
        :automatic_epsilon_lexicase)
    run = _TypedPopulationRun(scored.evaluations, scored.invalid_evaluations, 0,
        minimum(member.loss for member in scored.seeds), 0, 0, nothing, 0,
        scored.evaluations, :objective_calls)
    @test calls[] == 1
    @test state.case_errors == [1.0 1.0; 0.0 0.0]

    _typed_attempt!(state, run, scored.archive, language, objective, config,
        1, 1, :automatic_epsilon_lexicase, scored.case_count)
    @test state.accepted == 1
    @test calls[] == 2
    @test state.population[1].term.nodes[end].x == zero
    @test state.case_errors[:, 1] == [0.0, 1.0]
    @test state.case_errors[:, 2] == [1.0, 0.0]

    after_replacement = copy(state.case_errors)
    _typed_attempt!(state, run, scored.archive, language, objective, config,
        1, 2, :automatic_epsilon_lexicase, scored.case_count)
    @test state.repeats == 1
    @test calls[] == 2
    @test state.case_errors == after_replacement
end

@testitem "TypedSearch lexicase invalid and rejected children preserve columns" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch: _typed_attempt!, _TypedPopulationRun,
        _typed_make_population_state, _typed_score_initial
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])

    function make_state(objective, config, rng)
        scored = _typed_score_initial([seed], language, objective, config,
            :automatic_epsilon_lexicase)
        state = _typed_make_population_state(scored.seeds, scored.seed_case_errors,
            scored.case_count, config, language, rng, :automatic_epsilon_lexicase)
        run = _TypedPopulationRun(scored.evaluations, scored.invalid_evaluations, 0,
            minimum(member.loss for member in scored.seeds), 0, 0, nothing, 0,
            scored.evaluations, :objective_calls)
        return scored, state, run
    end

    rejected_calls = Ref(0)
    rejected_objective = function(term)
        rejected_calls[] += 1
        return term.nodes[end].x == one ? [0.0, 0.0] : [1.0, 1.0]
    end
    rejected_config = _typed_search_options(population_size=1, attempts=1,
        evaluations=4, kinds=(:point,), weights=(1.0,),
        acceptance_temperature=1.0e-9)
    rejected_scored, rejected_state, rejected_run = make_state(
        rejected_objective, rejected_config, MersenneTwister(555))
    rejected_columns = copy(rejected_state.case_errors)
    _typed_attempt!(rejected_state, rejected_run, rejected_scored.archive, language,
        rejected_objective, rejected_config, 1, 1,
        :automatic_epsilon_lexicase, rejected_scored.case_count)
    @test rejected_state.acceptance_rejected == 1
    @test rejected_calls[] == 2
    @test rejected_state.case_errors == rejected_columns
    @test rejected_state.population[1].term.nodes[end].x == one

    invalid_calls = Ref(0)
    invalid_objective = function(term)
        invalid_calls[] += 1
        return term.nodes[end].x == one ? [0.0, 0.0] :
            Err(TypedSearchError(:synthetic_invalid))
    end
    invalid_config = _typed_search_options(population_size=1, attempts=1,
        evaluations=4, kinds=(:point,), weights=(1.0,))
    invalid_scored, invalid_state, invalid_run = make_state(
        invalid_objective, invalid_config, MersenneTwister(2000))
    invalid_columns = copy(invalid_state.case_errors)
    _typed_attempt!(invalid_state, invalid_run, invalid_scored.archive, language,
        invalid_objective, invalid_config, 1, 1,
        :automatic_epsilon_lexicase, invalid_scored.case_count)
    @test invalid_state.invalid_evaluations == 1
    @test invalid_run.invalid_evaluations == 1
    @test invalid_calls[] == 2
    @test invalid_state.case_errors == invalid_columns
    @test invalid_state.population[1].term.nodes[end].x == one
end

@testitem "TypedSearch lexicase shares seeds and preserves tournament trajectory" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    shared_calls = Ref(0)
    shared = search_populations([seed], language, _ -> begin
        shared_calls[] += 1
        0.0
    end; (; _typed_search_options(population_size=3, attempts=0, evaluations=8)...)...,
        rng=MersenneTwister(1), group_count=20)
    @test shared_calls[] == 1
    @test shared.stats.initial_evaluations == 1
    @test shared.stats.evaluations == 1
    @test shared.stats.scored_seeds == 1
    @test shared.stats.replicated_seeds == 59
    @test shared.stats.selection === :tournament
    @test shared.stats.case_count == 0

    lex_calls = Ref(0)
    lex_shared = search_populations([seed], language, _ -> begin
        lex_calls[] += 1
        [0.0, 1.0]
    end; (; _typed_search_options(population_size=3, attempts=0, evaluations=8)...)...,
        rng=MersenneTwister(1), group_count=20,
        selection=:automatic_epsilon_lexicase)
    @test lex_calls[] == 1
    @test lex_shared.stats.initial_evaluations == 1
    @test lex_shared.stats.evaluations == 1
    @test lex_shared.stats.scored_seeds == 1
    @test lex_shared.stats.replicated_seeds == 59
    @test lex_shared.stats.selection === :automatic_epsilon_lexicase
    @test lex_shared.stats.case_count == 2

    higher = TypedTerm(TermNode[var_node(scalar, 0),
        abs_node(arrow!(language.types, scalar, scalar), 1)])
    @test validate(higher, language) === true
    higher_result = search_terms([higher], language, _ -> [0.0, 1.0];
        (; _typed_search_options(population_size=2, attempts=0, evaluations=4)...)...,
        rng=MersenneTwister(1111), selection=:automatic_epsilon_lexicase)
    @test higher_result isa TypedSearchResult
    @test higher_result.stats.selection === :automatic_epsilon_lexicase
    @test higher_result.stats.case_count == 2

    config = _typed_search_options(population_size=3, attempts=12, evaluations=20,
        kinds=(:point,), weights=(1.0,))
    default = search_terms([seed], language, _ -> 0.0;
        (; config...)..., rng=MersenneTwister(2000))
    explicit = search_terms([seed], language, _ -> 0.0;
        (; config...)..., rng=MersenneTwister(2000), selection=:tournament)
    @test [member.term.nodes for member in default.population] ==
        [member.term.nodes for member in explicit.population]
    @test [member.loss for member in default.population] ==
        [member.loss for member in explicit.population]
    @test [member.birth for member in default.population] ==
        [member.birth for member in explicit.population]
    @test [member.term.nodes for member in default.frontier] ==
        [member.term.nodes for member in explicit.frontier]
    @test default.stats.attempts == explicit.stats.attempts
    @test default.stats.evaluations == explicit.stats.evaluations
    @test default.stats.accepted == explicit.stats.accepted
    @test default.stats.selection === :tournament
    @test explicit.stats.selection === :tournament
end

@testitem "TypedSearch lexicase validates case objective contracts" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    base = _typed_search_options(population_size=1, attempts=0, evaluations=4)
    lex = :automatic_epsilon_lexicase
    @test_throws ArgumentError search_terms([seed], language, _ -> Float64[];
        (; base...)..., rng=MersenneTwister(1), selection=lex)
    @test_throws ArgumentError search_terms([seed], language, _ -> 0.0;
        (; base...)..., rng=MersenneTwister(1), selection=lex)
    @test_throws ArgumentError search_terms([seed], language, _ -> [0.0, -1.0];
        (; base...)..., rng=MersenneTwister(1), selection=lex)

    nonfinite = search_populations([seed, TypedTerm(TermNode[
        leaf_node(scalar, zero)])], language, term ->
        term.nodes[end].x == one ? [0.0, Inf] : [0.0, 0.0];
        (; _typed_search_options(population_size=2, attempts=0, evaluations=4)...)...,
        rng=MersenneTwister(1111), group_count=1, selection=lex)
    @test nonfinite.stats.evaluations == 2
    @test nonfinite.stats.invalid_evaluations == 1
    @test nonfinite.stats.initial_invalid_evaluations == 1
    @test nonfinite.stats.scored_seeds == 1
    @test nonfinite.stats.case_count == 2

    @test_throws ArgumentError search_terms([seed,
        TypedTerm(TermNode[leaf_node(scalar, zero)])], language,
        term -> term.nodes[end].x == one ? [0.0] : [0.0, 0.0];
        (; _typed_search_options(population_size=2, attempts=0, evaluations=4)...)...,
        rng=MersenneTwister(1234), selection=lex)
end

@testitem "TypedSearch rejects oversized seeds before recursive checking" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy, TyFun
    using LambdaRegression.Utils: Err
    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    fn = intern!(language.types, TyFun(FloatTy, FloatTy))
    leaf = add_leaf!(language, scalar, :one)
    nodes = TermNode[leaf_node(scalar, leaf)]
    # Deliberately invalid types are irrelevant: the resource guard must run
    # first, without descending this long syntax chain or calling the objective.
    for i in 2:10_000
        push!(nodes, abs_node(fn, i - 1))
    end
    calls = Ref(0)
    objective = _ -> (calls[] += 1; 0.0)
    @test_throws ArgumentError search_terms([TypedTerm(nodes)], language, objective)
    @test_throws ArgumentError search_terms([TypedTerm(nodes)], language, objective;
        (; _typed_search_options(max_nodes=10_000, max_depth=32)...)...)
    cycle = TypedTerm(TermNode[abs_node(fn, 1)])
    @test search_terms([cycle], language, objective) isa Err
    @test calls[] == 0
end

@testitem "TypedSearch bounded population accounting and source return" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Procedural: evaluate_procedural
    using Random: MersenneTwister
    ctx = Dict{Symbol,TyLike}(:one => FloatTy)
    values = Dict{Symbol,Any}(:one => 1.0)
    target = TyFun(FloatTy, FloatTy)
    source = prepare_source(AstTree(AbsNode(:x, IndexNode(0))), ctx, target)
    hole = source_hole(source)
    language = Language(TypeTable())
    seed = import_hole(hole, language)
    # This is a labelled unit fixture, not autonomous initial material.
    constant = prepare_source(AstTree(ConstNode(:one)), ctx, FloatTy)
    import_hole(source_hole(constant), language)
    calls = Ref(0)
    objective = function(term)
        calls[] += 1
        tree = export_hole(term, language, hole)
        value = evaluate_procedural(tree, values; args=(2.0,))
        return (value - 1.0)^2
    end
    config = _typed_search_options(population_size=4, attempts=40, evaluations=20,
        kinds=(:point,), weights=(1.0,))
    result = search_terms([seed], language, objective;
        (; config...)..., rng=MersenneTwister(1111))
    @test result isa TypedSearchResult
    @test result.stats.evaluations == calls[] <= 20
    @test result.stats.attempts <= 40
    @test result.stats.scored_seeds == 1
    @test result.stats.replicated_seeds == 3
    @test length(result.population) == 4
    @test result.stats.attempts == result.stats.changed + result.stats.failures + result.stats.unchanged
    @test !isempty(result.frontier)
    @test all(m -> validate(m.term, language) === true, result.population)
    @test all(m -> prepare_source(export_hole(m.term, language, hole), ctx, target) isa SourceAnalysis,
        result.frontier)
end

@testitem "TypedSearch archive feedback and telemetry are opt-in" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    objective = term -> any(node -> node.kind == LEAF_KIND && node.x == zero, term.nodes) ? 0.0 : 1.0
    base_config = _typed_search_options(population_size=4, attempts=24, evaluations=32,
        kinds=(:point,), weights=(1.0,))
    plain = search_terms([seed], language, objective;
        (; base_config...)..., rng=MersenneTwister(42))
    snapshots = NamedTuple[]
    events = NamedTuple[]
    observed_config = _typed_search_options(population_size=4, attempts=24, evaluations=32,
        kinds=(:point,), weights=(1.0,), observer=s -> push!(snapshots, s),
        observer_interval=3, improvement_callback=e -> push!(events, e),
        improvement_limit=2)
    observed = search_terms([seed], language, objective;
        (; observed_config...)..., rng=MersenneTwister(42))

    # The observer and trace do not receive the RNG and therefore cannot alter
    # the default trajectory or objective budget.
    @test observed.stats.evaluations == plain.stats.evaluations
    @test observed.stats.attempts == plain.stats.attempts
    @test [m.loss for m in observed.population] == [m.loss for m in plain.population]
    @test [m.term.nodes for m in observed.population] == [m.term.nodes for m in plain.population]
    @test observed.stats.observer_calls == length(snapshots) <= 8
    @test all(s -> s.population_size == 4 && s.distinct_count >= 1 &&
        haskey(s.best_loss_by_size, 1), snapshots)
    @test observed.stats.improvement_calls == length(events) <= 2
    @test all(e -> e.evaluation_id <= observed.stats.evaluations &&
        e.global_improvement && e.parent isa TypedTerm &&
        e.donor isa TypedTerm && e.child isa TypedTerm, events)

    feedback = search_terms([seed], language, objective;
        (; _typed_search_options(population_size=4, attempts=8, evaluations=12,
            kinds=(:crossover,), weights=(1.0,), archive_probability=1.0)...)...,
        rng=MersenneTwister(42))
    @test feedback.stats.archive_parent_feedback > 0
    @test feedback.stats.archive_donor_feedback > 0
    @test feedback.stats.evaluations <= 12
end

@testitem "TypedSearch archive snapshots copy terms only when requested" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _typed_frontier, _typed_snapshot
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    objective = function(term, calls)
        calls[] += 1
        return any(node -> node.kind == LEAF_KIND && node.x == zero, term.nodes) ?
            0.0 : 1.0
    end
    plain_calls = Ref(0)
    archive_calls = Ref(0)
    config = _typed_search_options(population_size=4, attempts=24, evaluations=32,
        kinds=(:point,), weights=(1.0,), observer_interval=3, observer_limit=8)
    plain_rng = MersenneTwister(42)
    archive_rng = MersenneTwister(42)
    plain = search_terms([seed], language, term -> objective(term, plain_calls);
        (; config...)..., rng=plain_rng)
    snapshots = NamedTuple[]
    observer = function(snapshot)
        push!(snapshots, snapshot)
        # Mutating an observed copy during the run must not mutate the archive,
        # later snapshots, or the search trajectory.
        if length(snapshots) == 1
            original_x = snapshot.archive_members[1].term.nodes[1].x
            replacement_x = original_x == one ? zero : one
            snapshot.archive_members[1].term.nodes[1] =
                leaf_node(scalar, replacement_x)
            @assert original_x != snapshot.archive_members[1].term.nodes[1].x
        end
    end
    archived = search_terms([seed], language, term -> objective(term, archive_calls);
        (; _typed_search_options(population_size=4, attempts=24, evaluations=32,
            kinds=(:point,), weights=(1.0,), observer_interval=3, observer_limit=8,
            observer=observer)...)..., rng=archive_rng,
        snapshot_archive_terms=true)

    # The opt-in materialisation must not affect the trajectory or objective
    # budget.  Telemetry is the sole intentionally different result field.
    for field in fieldnames(typeof(plain.stats))
        field === :telemetry && continue
        @test getproperty(archived.stats, field) == getproperty(plain.stats, field)
    end
    @test plain_calls[] == archive_calls[] == plain.stats.evaluations
    @test rand(plain_rng) == rand(archive_rng)
    @test length(archived.population) == length(plain.population)
    @test length(archived.frontier) == length(plain.frontier)
    @test all(zip(archived.population, plain.population)) do members
        a, b = members
        a.loss == b.loss && a.birth == b.birth && a.term.nodes == b.term.nodes
    end
    @test all(zip(archived.frontier, plain.frontier)) do members
        a, b = members
        a.loss == b.loss && a.birth == b.birth && a.term.nodes == b.term.nodes
    end
    @test !isempty(snapshots)
    @test all(s -> !hasproperty(s, :archive_members), plain.stats.telemetry)
    @test all(s -> hasproperty(s, :archive_members) &&
        length(s.archive_members) == length(s.best_loss_by_size) &&
        all(member -> member.loss == s.best_loss_by_size[member.size],
            s.archive_members), snapshots)

    # A larger equal-loss bucket is Pareto-dominated, but remains in the full
    # archive snapshot rather than being reduced to the returned frontier.
    fn = arrow!(language.types, scalar, scalar)
    short = TypedTerm(TermNode[leaf_node(scalar, one)])
    longer = TypedTerm(TermNode[leaf_node(scalar, one), abs_node(fn, 1)])
    archive = Dict{Int,ScoredTerm}(1 => ScoredTerm(short, 0.0, 1),
        2 => ScoredTerm(longer, 0.0, 2))
    direct = _typed_snapshot(1, 1, [ScoredTerm(short, 0.0, 1)], archive, language;
        snapshot_archive_terms=true)
    @test length(_typed_frontier(archive)) == 1
    @test [member.size for member in direct.archive_members] == [1, 2]
    @test all(member -> member.loss == direct.best_loss_by_size[member.size],
        direct.archive_members)
    original_nodes = copy(archive[1].term.nodes)
    direct.archive_members[1].term.nodes[1] = leaf_node(scalar, zero)
    @test archive[1].term.nodes == original_nodes
    if length(snapshots) > 1
        later_nodes = copy(snapshots[2].archive_members[1].term.nodes)
        snapshots[1].archive_members[1].term.nodes[1] = leaf_node(scalar, zero)
        @test snapshots[2].archive_members[1].term.nodes == later_nodes
    end

    no_limit_seen = NamedTuple[]
    no_limit = search_terms([seed], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
            kinds=(:point,), weights=(1.0,), observer=s -> push!(no_limit_seen, s),
            observer_interval=1, observer_limit=0)...)...,
        rng=MersenneTwister(555), snapshot_archive_terms=true)
    no_interval_seen = NamedTuple[]
    no_interval = search_terms([seed], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
            kinds=(:point,), weights=(1.0,), observer=s -> push!(no_interval_seen, s),
            observer_interval=0, observer_limit=8)...)...,
        rng=MersenneTwister(555), snapshot_archive_terms=true)
    @test isempty(no_limit.stats.telemetry)
    @test isempty(no_interval.stats.telemetry)
    @test isempty(no_limit_seen)
    @test isempty(no_interval_seen)

    # Exercise the same keyword through the multi-population lexicase path.
    lexicase = search_populations([seed], language,
        term -> any(node -> node.kind == LEAF_KIND && node.x == zero, term.nodes) ?
            [0.0, 1.0] : [1.0, 0.0];
        (; _typed_search_options(population_size=2, attempts=8, evaluations=8,
            kinds=(:point,), weights=(1.0,), observer_interval=1, observer_limit=4)...)...,
        rng=MersenneTwister(2000), group_count=2, chunk_attempts=1,
        selection=:automatic_epsilon_lexicase, snapshot_archive_terms=true)
    @test lexicase isa TypedPopulationResult
    @test length(lexicase.populations) == 2
    @test !isempty(lexicase.stats.telemetry)
    @test all(hasproperty(s, :archive_members) for s in lexicase.stats.telemetry)
end

@testitem "TypedSearch archive tie policy is opt-in" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _record_typed!
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed_one = TypedTerm(TermNode[leaf_node(scalar, one)])
    seed_zero = TypedTerm(TermNode[leaf_node(scalar, zero)])

    # Initial seed recording uses the same tie policy as accepted children.
    seed_config = _typed_search_options(population_size=2, attempts=0, evaluations=4,
        kinds=(:point,), weights=(1.0,))
    first_seed = search_terms([seed_one, seed_zero], language, _ -> 1.0;
        (; seed_config...)..., rng=MersenneTwister(42))
    newest_seed = search_terms([seed_one, seed_zero], language, _ -> 1.0;
        (; seed_config...)..., rng=MersenneTwister(42), archive_ties=:newest)
    @test first_seed.frontier[1].term.nodes == seed_one.nodes
    @test newest_seed.frontier[1].term.nodes == seed_zero.nodes

    first_archive = Dict{Int,ScoredTerm}()
    newest_archive = Dict{Int,ScoredTerm}()
    old = ScoredTerm(seed_one, 1.0, 1)
    tie = ScoredTerm(seed_zero, 1.0, 2)
    better = ScoredTerm(seed_zero, 0.5, 3)
    worse = ScoredTerm(seed_one, 2.0, 4)
    _record_typed!(first_archive, old, language)
    _record_typed!(first_archive, tie, language)
    _record_typed!(first_archive, better, language)
    _record_typed!(first_archive, worse, language)
    _record_typed!(newest_archive, old, language; archive_ties=:newest)
    _record_typed!(newest_archive, tie, language; archive_ties=:newest)
    _record_typed!(newest_archive, better, language; archive_ties=:newest)
    _record_typed!(newest_archive, worse, language; archive_ties=:newest)
    @test first_archive[1] === better
    @test newest_archive[1] === better
    @test first_archive[1].loss == 0.5
    @test newest_archive[1].loss == 0.5

    first_tie = Dict{Int,ScoredTerm}()
    newest_tie = Dict{Int,ScoredTerm}()
    _record_typed!(first_tie, old, language)
    _record_typed!(first_tie, tie, language)
    _record_typed!(newest_tie, old, language; archive_ties=:newest)
    _record_typed!(newest_tie, tie, language; archive_ties=:newest)
    @test first_tie[1].term.nodes == old.term.nodes
    @test newest_tie[1].term.nodes == tie.term.nodes

    config = _typed_search_options(population_size=2, attempts=8, evaluations=8,
        kinds=(:point,), weights=(1.0,))
    default_rng = MersenneTwister(555)
    first_rng = MersenneTwister(555)
    default = search_terms([seed_one], language, _ -> 1.0;
        (; config...)..., rng=default_rng)
    explicit_first = search_terms([seed_one], language, _ -> 1.0;
        (; config...)..., rng=first_rng, archive_ties=:first)
    @test [member.term.nodes for member in default.population] ==
        [member.term.nodes for member in explicit_first.population]
    @test [member.term.nodes for member in default.frontier] ==
        [member.term.nodes for member in explicit_first.frontier]
    @test default.stats == explicit_first.stats
    @test rand(default_rng) == rand(first_rng)

    # A real accepted point child has the same written size and equal loss;
    # only :newest may move that archive bucket to the child.
    evolution_config = _typed_search_options(population_size=1, attempts=1,
        evaluations=2, kinds=(:point,), weights=(1.0,))
    first_evolution = search_terms([seed_one], language, _ -> 1.0;
        (; evolution_config...)..., rng=MersenneTwister(2000), archive_ties=:first)
    newest_evolution = search_terms([seed_one], language, _ -> 1.0;
        (; evolution_config...)..., rng=MersenneTwister(2000), archive_ties=:newest)
    @test first_evolution.stats.accepted == newest_evolution.stats.accepted == 1
    @test first_evolution.frontier[1].term.nodes == seed_one.nodes
    @test newest_evolution.frontier[1].term.nodes == newest_evolution.population[1].term.nodes
    @test newest_evolution.frontier[1].term.nodes != seed_one.nodes

    newest_population = search_populations([seed_one], language, _ -> 1.0;
        (; config...)..., rng=MersenneTwister(1), group_count=2, chunk_attempts=1,
        archive_ties=:newest)
    @test newest_population isa TypedPopulationResult
    @test length(newest_population.populations) == 2
    lexicase = search_populations([seed_one], language, _ -> [1.0, 1.0];
        (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
            kinds=(:point,), weights=(1.0,))...)..., rng=MersenneTwister(1),
        selection=:automatic_epsilon_lexicase, archive_ties=:newest)
    @test lexicase isa TypedPopulationResult
    @test lexicase.stats.selection === :automatic_epsilon_lexicase
    @test_throws ArgumentError search_terms([seed_one], language, _ -> 1.0;
        (; config...)..., archive_ties=:invalid)
    @test_throws ArgumentError search_populations([seed_one], language, _ -> 1.0;
        (; config...)..., archive_ties=:invalid)
end

@testitem "TypedSearch telemetry retains higher-order terms" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    fn = arrow!(language.types, scalar, scalar)
    target = arrow!(language.types, fn, fn)
    # λf.λx.f x, a higher-order seed whose captured function is a local.
    higher = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    @test validate(higher, language) === true
    snapshots = NamedTuple[]
    result = search_terms([higher], language, term -> 0.0;
        (; _typed_search_options(population_size=3, attempts=4, evaluations=8,
            kinds=(:point,), weights=(1.0,), observer=s -> push!(snapshots, s),
            observer_interval=1, archive_probability=1.0)...)...,
        rng=MersenneTwister(1111))
    @test result isa TypedSearchResult
    @test all(m -> validate(m.term, language) === true, result.population)
    @test any(any(node -> node.kind == ABS_KIND, m.term.nodes) for m in result.population)
    @test !isempty(snapshots)
end

@testitem "TypedSearch distinct count uses node-vector content" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _typed_distinct_count
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    fn = arrow!(language.types, scalar, scalar)
    short_one = TypedTerm(TermNode[leaf_node(scalar, one)])
    equal_copy = TypedTerm(TermNode[leaf_node(scalar, one)])
    short_zero = TypedTerm(TermNode[leaf_node(scalar, zero)])
    longer = TypedTerm(TermNode[leaf_node(scalar, one), abs_node(fn, 1)])
    @test validate(short_one, language) === true
    @test validate(equal_copy, language) === true
    @test validate(short_zero, language) === true
    @test validate(longer, language) === true
    population = [ScoredTerm(short_one, 0.0, 1),
                  ScoredTerm(equal_copy, 0.0, 2),
                  ScoredTerm(short_zero, 0.0, 3),
                  ScoredTerm(longer, 0.0, 4)]
    @test _typed_distinct_count(population) == 3
    # The previous call's ephemeral set must not retain mutable keys.
    population[3].term.nodes[1] = leaf_node(scalar, one)
    @test _typed_distinct_count(population) == 2
    @test _typed_distinct_count(ScoredTerm[]) == 0
end

@testitem "TypedSearch written-size frequency tournament is opt-in" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _RegularizedFrequency,
        _typed_frequency_key, _typed_observe_frequency!
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    fn = arrow!(language.types, scalar, scalar)
    target = arrow!(language.types, fn, fn)
    higher = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    @test validate(higher, language) === true

    base = _typed_search_options(population_size=3, attempts=4, evaluations=8,
        kinds=(:point,), weights=(1.0,))
    zero = _typed_search_options(population_size=3, attempts=4, evaluations=8,
        kinds=(:point,), weights=(1.0,), frequency_strength=0.0)
    plain = search_terms([higher], language, _ -> 0.0;
        (; base...)..., rng=MersenneTwister(1234))
    disabled = search_terms([higher], language, _ -> 0.0;
        (; zero...)..., rng=MersenneTwister(1234))
    @test [m.term.nodes for m in disabled.population] == [m.term.nodes for m in plain.population]
    @test disabled.stats.frequency_summary === nothing

    enabled = search_terms([higher], language, _ -> 0.0;
        (; _typed_search_options(population_size=3, attempts=4, evaluations=8,
            kinds=(:point,), weights=(1.0,), max_nodes=8,
            frequency_strength=0.25, frequency_window=32)...)...,
        rng=MersenneTwister(1234))
    summary = enabled.stats.frequency_summary
    @test summary.enabled && summary.window == 32 && !isempty(summary.observed_sizes)
    @test summary.total <= 32 + 1.0e-8

    degenerate = search_terms([higher], language, _ -> 0.0;
        (; _typed_search_options(population_size=3, attempts=1, evaluations=8,
            kinds=(:point,), weights=(1.0,), max_nodes=8,
            frequency_strength=0.25, frequency_window=8)...)...,
        rng=MersenneTwister(555))
    @test degenerate.stats.frequency_summary.window == 8
    @test isempty(degenerate.stats.frequency_summary.observed_sizes)

    @test_throws ArgumentError search_terms([higher], language, _ -> -1.0;
        (; _typed_search_options(population_size=3, attempts=1, evaluations=4,
            kinds=(:point,), weights=(1.0,), max_nodes=8,
            frequency_strength=0.25, frequency_window=32)...)..., rng=MersenneTwister(8000))
    negative_disabled = search_terms([higher], language, _ -> -1.0;
        (; _typed_search_options(population_size=3, attempts=0, evaluations=4,
            kinds=(:point,), weights=(1.0,), max_nodes=8)...)..., rng=MersenneTwister(2000))
    @test negative_disabled isa TypedSearchResult
    @test_throws ArgumentError search_terms([higher], language, _ -> 0.0;
        (; _typed_search_options(max_nodes=8, frequency_strength=0.25,
            frequency_window=7)...)..., rng=MersenneTwister(42))

    # The key penalizes a more frequent written-size bucket, while preserving
    # zero loss as the separate best-loss case.
    freq = _RegularizedFrequency(32, 4)
    one = add_leaf!(language, scalar, :one)
    t1 = TypedTerm(TermNode[leaf_node(scalar, one)])
    t2 = TypedTerm(TermNode[var_node(scalar, 0), abs_node(fn, 1)])
    occupancy = [ScoredTerm(t1, 1.0, i) for i in 1:8]
    push!(occupancy, ScoredTerm(t2, 1.0, 9))
    _typed_observe_frequency!(freq, occupancy, language)
    key1 = _typed_frequency_key(ScoredTerm(t1, 1.0, 1), language, freq, 0.25)
    key2 = _typed_frequency_key(ScoredTerm(t2, 1.0, 2), language, freq, 0.25)
    @test key1[2] > key2[2]
    @test _typed_frequency_key(ScoredTerm(t2, 0.0, 3), language, freq, 0.25) <
        _typed_frequency_key(ScoredTerm(t2, 1.0e-12, 4), language, freq, 0.25)
end

@testitem "TypedSearch persistent populations share budget and HOF" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    two = add_leaf!(language, scalar, :two)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    calls = Ref(0)
    objective = term -> begin
        calls[] += 1
        any(node -> node.kind == LEAF_KIND && node.x == two, term.nodes) ? 0.0 : 1.0
    end
    config = _typed_search_options(population_size=3, attempts=10, evaluations=12,
        kinds=(:point,), weights=(1.0,), archive_probability=1.0,
        max_nodes=8, frequency_strength=0.2, frequency_window=32)
    snapshots = NamedTuple[]
    result = search_populations([seed], language, objective; (; config...)...,
        rng=MersenneTwister(555), group_count=2, chunk_attempts=1)
    @test result isa TypedPopulationResult
    @test length(result.populations) == 2
    @test all(length(population) == 3 for population in result.populations)
    @test result.stats.evaluations == calls[] <= config.evaluations
    @test result.stats.evaluations == result.stats.initial_evaluations +
        sum(stats.evaluations for stats in result.stats.population_stats)
    @test result.stats.invalid_evaluations ==
        sum(stats.invalid_evaluations for stats in result.stats.population_stats)
    @test result.stats.attempts == sum(stats.attempts for stats in result.stats.population_stats)
    @test result.stats.accepted == sum(stats.accepted for stats in result.stats.population_stats)
    @test all(stats -> begin
        point = stats.operator_stats[:point]
        point.accepted == point.loss_improvement_count +
            point.loss_degradation_count + point.loss_equal_count
    end, result.stats.population_stats)
    @test result.stats.accepted > 0
    @test result.stats.archive_parent_feedback > 0
    @test result.stats.frequency_summary === nothing
    @test length(result.stats.frequency_summaries) == 2
    @test all(summary -> summary !== nothing && summary.total <= 32 + 1.0e-8,
        result.stats.frequency_summaries)
    @test all(stats -> stats.population_id in (1, 2), result.stats.population_stats)
    @test all(pair -> all(member -> member.birth <= config.population_size + pair[2].accepted,
            pair[1]), zip(result.populations, result.stats.population_stats))
    @test issorted(result.stats.telemetry; by=s -> (s.attempt, s.population_id))

    push!(snapshots, (;))
    observed_config = _typed_search_options(population_size=3, attempts=10, evaluations=12,
        kinds=(:point,), weights=(1.0,), observer=s -> push!(snapshots, s),
        observer_interval=1, observer_limit=8)
    observed = search_populations([seed], language, objective; (; observed_config...)...,
        rng=MersenneTwister(555), group_count=2, chunk_attempts=1)
    plain = search_populations([seed], language, objective; (; _typed_search_options(
        population_size=3, attempts=10, evaluations=12, kinds=(:point,), weights=(1.0,))...)...,
        rng=MersenneTwister(555), group_count=2, chunk_attempts=1)
    @test all(zip(observed.populations, plain.populations)) do pair
        all(zip(pair[1], pair[2])) do members
            a, b = members
            a.loss == b.loss && a.birth == b.birth && a.term.nodes == b.term.nodes
        end
    end
    @test observed.stats.attempts == plain.stats.attempts
    @test observed.stats.evaluations == plain.stats.evaluations
    @test length(snapshots) > 1
    @test all(hasproperty(snapshot, :population_id) for snapshot in snapshots[2:end])

    serial_config = _typed_search_options(population_size=3, attempts=8, evaluations=12,
        kinds=(:point,), weights=(1.0,))
    serial_one = search_populations([seed], language, objective; (; serial_config...)...,
        rng=MersenneTwister(1), group_count=1, chunk_attempts=1)
    serial_two = search_populations([seed], language, objective; (; serial_config...)...,
        rng=MersenneTwister(1), group_count=1, chunk_attempts=2)
    @test serial_one.stats.attempts == serial_two.stats.attempts
    @test serial_one.stats.evaluations == serial_two.stats.evaluations
    @test all(zip(serial_one.populations[1], serial_two.populations[1])) do members
        a, b = members
        a.loss == b.loss && a.birth == b.birth && a.term.nodes == b.term.nodes
    end
end

@testitem "TypedSearch periodic HOF migration is bounded and case-aligned" begin
    using Test
    using Random
    using LambdaRegression.TypedSearch: _typed_hof_migrate!,
        _typed_hof_poisson, _typed_make_population_state, _record_typed!,
        _typed_score_initial, _typed_attempt!, _TypedPopulationRun,
        _typed_frontier
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    fn = arrow!(language.types, scalar, scalar)
    target = arrow!(language.types, fn, fn)
    higher = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    higher_one = TypedTerm(TermNode[
        var_node(fn, 1), leaf_node(scalar, one), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    dominated = TypedTerm(TermNode[
        var_node(fn, 1), var_node(fn, 1), var_node(scalar, 0),
        app_node(scalar, 2, 3), app_node(scalar, 1, 4),
        abs_node(fn, 5), abs_node(target, 6)])
    @test validate(higher, language) === true
    @test validate(higher_one, language) === true
    @test validate(dominated, language) === true

    lex_objective = _ -> [0.0, 1.0]
    config = _typed_search_options(population_size=8, attempts=0, evaluations=4,
        max_nodes=8)
    scored = _typed_score_initial([higher], language, lex_objective, config,
        :automatic_epsilon_lexicase; archive_ties=:newest,
        archive_case_vectors=true)
    @test scored.archive_case_errors isa Dict{Int,Vector{Float64}}
    state = _typed_make_population_state(scored.seeds, scored.seed_case_errors,
        scored.case_count, config, language, MersenneTwister(1),
        :automatic_epsilon_lexicase)

    # Replace the exact archive representative under :newest and verify that
    # its replacement vector, not the stale seed vector, is migrated.
    size = first(keys(scored.archive))
    replacement = ScoredTerm(TypedTerm(copy(higher_one.nodes)), 0.5, 99)
    _record_typed!(scored.archive, replacement, language;
        archive_ties=:newest, archive_case_errors=scored.archive_case_errors,
        case_errors=[1.0, 0.0])
    _record_typed!(scored.archive, ScoredTerm(TypedTerm(copy(dominated.nodes)),
        2.0, 100), language; archive_ties=:newest,
        archive_case_errors=scored.archive_case_errors, case_errors=[2.0, 2.0])
    frontier = _typed_frontier(scored.archive)
    @test length(frontier) == 1
    @test frontier[1].term.nodes == higher_one.nodes
    source = scored.archive[size]
    source_nodes = copy(source.term.nodes)
    before_birth = state.birth
    inserted = _typed_hof_migrate!(state, scored.archive,
        scored.archive_case_errors, language, 1.0)
    @test inserted == 1
    @test state.hof_migration_events == 1
    @test state.hof_migration_insertions == 1
    @test state.birth == before_birth + 1
    destination = findfirst(member -> member.birth == state.birth, state.population)
    @test destination !== nothing
    @test state.case_errors[:, destination] == [1.0, 0.0]
    @test state.population[destination].term.nodes == source.term.nodes
    @test state.population[destination].term.nodes !== source.term.nodes
    @test validate(state.population[destination].term, language) === true
    state.population[destination].term.nodes[1] = leaf_node(scalar, one)
    @test source.term.nodes == source_nodes

    # A directly migrated member is counted in both roles for crossover, but
    # an archive override suppresses both direct-population counts.
    selected_config = _typed_search_options(population_size=1, attempts=1,
        evaluations=4, tournament_size=1, kinds=(:crossover,), weights=(1.0,),
        max_nodes=8)
    selected_state = _typed_make_population_state([source], [[1.0, 0.0]], 2,
        selected_config, language, MersenneTwister(1111),
        :automatic_epsilon_lexicase)
    selected_state.birth += 1
    selected_state.population[1] = ScoredTerm(
        TypedTerm(copy(source.term.nodes)), source.loss, selected_state.birth)
    push!(selected_state.migrant_births, selected_state.birth)
    selected_run = _TypedPopulationRun(1, 0, 0, source.loss, 0, 0, nothing, 0,
        1, :objective_calls)
    _typed_attempt!(selected_state, selected_run, scored.archive, language,
        _ -> [1.0, 0.0], selected_config, 1, 1,
        :automatic_epsilon_lexicase, 2;
        archive_ties=:newest, archive_case_errors=scored.archive_case_errors)
    @test selected_state.hof_migration_parent_selected == 1
    @test selected_state.hof_migration_donor_selected == 1

    override_config = _typed_search_options(population_size=1, attempts=1,
        evaluations=4, tournament_size=1, kinds=(:crossover,), weights=(1.0,),
        max_nodes=8, archive_probability=1.0)
    override_state = _typed_make_population_state([source], [[1.0, 0.0]], 2,
        override_config, language, MersenneTwister(1234),
        :automatic_epsilon_lexicase)
    override_state.birth += 1
    override_state.population[1] = ScoredTerm(
        TypedTerm(copy(source.term.nodes)), source.loss, override_state.birth)
    push!(override_state.migrant_births, override_state.birth)
    override_run = _TypedPopulationRun(1, 0, 0, source.loss, 0, 0, nothing, 0,
        1, :objective_calls)
    _typed_attempt!(override_state, override_run, scored.archive, language,
        _ -> [1.0, 0.0], override_config, 1, 1,
        :automatic_epsilon_lexicase, 2;
        archive_ties=:newest, archive_case_errors=scored.archive_case_errors)
    @test override_state.hof_migration_parent_selected == 0
    @test override_state.hof_migration_donor_selected == 0
    @test override_state.archive_parent_feedback > 0
    @test override_state.archive_donor_feedback > 0

    # Disabled migration is trajectory/RNG-neutral; enabled migration keeps
    # objective accounting separate from per-population raw-attempt periods.
    plain_rng = MersenneTwister(555)
    off_rng = MersenneTwister(555)
    plain_calls = Ref(0)
    off_calls = Ref(0)
    plain = search_populations([higher], language, _ -> begin
        plain_calls[] += 1
        [0.0, 1.0]
    end; (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
        max_nodes=8, kinds=(:point,), weights=(1.0,))...)...,
        rng=plain_rng, group_count=2, chunk_attempts=1,
        selection=:automatic_epsilon_lexicase)
    off = search_populations([higher], language, _ -> begin
        off_calls[] += 1
        [0.0, 1.0]
    end;
        (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
            max_nodes=8, kinds=(:point,), weights=(1.0,))...)...,
        rng=off_rng, group_count=2, chunk_attempts=1,
        selection=:automatic_epsilon_lexicase, hof_migration_interval=0,
        hof_migration_fraction=0.0614)
    @test [m.term.nodes for m in plain.frontier] == [m.term.nodes for m in off.frontier]
    @test all(zip(plain.populations, off.populations)) do populations
        all(zip(populations[1], populations[2])) do members
            a, b = members
            a.loss == b.loss && a.birth == b.birth &&
                a.term.nodes == b.term.nodes
        end
    end
    @test plain.stats.evaluations == off.stats.evaluations == plain_calls[] == off_calls[]
    @test rand(plain_rng) == rand(off_rng)
    @test all(s -> s.hof_migration_events == 0, off.stats.population_stats)

    calls = Ref(0)
    grouped = search_populations([higher], language, term -> begin
        calls[] += 1
        [0.0, 1.0]
    end; (; _typed_search_options(population_size=2, attempts=4, evaluations=8,
        max_nodes=8, kinds=(:point,), weights=(1.0,))...)...,
        rng=MersenneTwister(2000), group_count=2, chunk_attempts=1,
        selection=:automatic_epsilon_lexicase, hof_migration_interval=2,
        hof_migration_fraction=1.0)
    @test [s.attempts for s in grouped.stats.population_stats] == [2, 2]
    @test [s.hof_migration_events for s in grouped.stats.population_stats] == [1, 1]
    @test grouped.stats.hof_migration_insertions <= 4
    @test grouped.stats.evaluations == calls[]
    @test grouped.stats.evaluations == grouped.stats.initial_evaluations +
        sum(s.evaluations for s in grouped.stats.population_stats)
    @test grouped.stats.hof_migration_parent_selected >= 0
    @test grouped.stats.hof_migration_donor_selected >= 0
    @test all(population -> all(member -> validate(member.term, language) === true,
        population), grouped.populations)

    @test_throws ArgumentError search_populations([higher], language, _ -> 0.0;
        (; config...)..., hof_migration_interval=-1)
    @test_throws ArgumentError search_populations([higher], language, _ -> 0.0;
        (; config...)..., hof_migration_fraction=1.1)
    @test_throws ArgumentError search_populations([higher], language, _ -> 0.0;
        (; config...)..., hof_migration_fraction=NaN)
    @test _typed_hof_poisson(MersenneTwister(8000), 1.0e300, 3) <= 3
end

@testitem "TypedSearch populations retain higher-order capture terms" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    fn = arrow!(language.types, scalar, scalar)
    target = arrow!(language.types, fn, fn)
    higher = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    @test validate(higher, language) === true
    result = search_populations([higher], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=6, evaluations=8,
            kinds=(:point,), weights=(1.0,), archive_probability=1.0)...)...,
        rng=MersenneTwister(2000), group_count=2, chunk_attempts=2)
    @test result isa TypedPopulationResult
    @test all(population -> all(member -> validate(member.term, language) === true,
        population), result.populations)
    @test all(population -> any(member ->
        any(node -> node.kind == ABS_KIND, member.term.nodes), population), result.populations)
    @test all(stats -> stats.evaluations >= 0, result.stats.population_stats)
end

@testitem "TypedSearch accepted loss transition statistics preserve trajectory" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    two = add_leaf!(language, scalar, :two)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    objective = term -> any(node -> node.kind == LEAF_KIND && node.x == two,
        term.nodes) ? -2.0 : -1.0
    config = _typed_search_options(population_size=1, attempts=24, evaluations=32,
        kinds=(:point,), weights=(1.0,))
    plain = search_terms([seed], language, objective;
        (; config...)..., rng=MersenneTwister(8000))
    observed = search_terms([seed], language, objective;
        (; _typed_search_options(population_size=1, attempts=24, evaluations=32,
            kinds=(:point,), weights=(1.0,), observer=s -> nothing,
            observer_interval=1)...)..., rng=MersenneTwister(8000))
    point = plain.stats.operator_stats[:point]
    @test point.accepted == point.loss_improvement_count +
        point.loss_degradation_count + point.loss_equal_count
    @test point.loss_degradation_sum >= 0.0
    @test point.loss_degradation_overflow_count == 0
    @test point.loss_improvement_overflow_count == 0
    @test point.loss_improvement_count > 0
    @test point.loss_degradation_count > 0
    @test plain.stats.attempts == observed.stats.attempts
    @test plain.stats.evaluations == observed.stats.evaluations
    @test all(zip(plain.population, observed.population)) do members
        a, b = members
        a.loss == b.loss && a.birth == b.birth && a.term.nodes == b.term.nodes
    end

    equal = search_terms([seed], language, _ -> -1.0;
        (; config...)..., rng=MersenneTwister(555))
    equal_point = equal.stats.operator_stats[:point]
    @test equal_point.loss_equal_count > 0
    @test equal_point.accepted == equal_point.loss_equal_count

    extreme = search_terms([seed], language,
        term -> any(node -> node.kind == LEAF_KIND && node.x == two, term.nodes) ?
            floatmax(Float64) : -floatmax(Float64);
        (; _typed_search_options(population_size=1, attempts=24, evaluations=32,
            kinds=(:point,), weights=(1.0,))...)..., rng=MersenneTwister(42))
    extreme_point = extreme.stats.operator_stats[:point]
    @test extreme_point.accepted == extreme_point.loss_improvement_count +
        extreme_point.loss_degradation_count + extreme_point.loss_equal_count
    @test extreme_point.loss_degradation_overflow_count > 0
    @test isinf(extreme_point.loss_degradation_sum)
end

@testitem "TypedSearch insertion is opt-in and follows the acceptance gate" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _typed_acceptance_probability
    using LambdaRegression.TypeSystem: FloatTy
    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    fn = arrow!(language.types, scalar, scalar)
    add_leaf!(language, scalar, :one)
    add_leaf!(language, fn, :identity)
    seed = TypedTerm(TermNode[leaf_node(scalar, 1)])
    @test _typed_search_options().kinds == (:generate, :crossover, :hoist, :point)
    integer_budget_child = propose_term(MersenneTwister(1234), seed, seed, language;
        kind=:insert, max_nodes=Int32(8), max_depth=Int32(8))
    @test integer_budget_child isa TypedTerm
    @test validate(integer_budget_child, language) === true
    @test length(integer_budget_child.nodes) == 3
    config = _typed_search_options(population_size=1, attempts=1, evaluations=2,
        max_nodes=8, max_depth=8, kinds=(:insert,), weights=(1.0,))
    result = search_terms([seed], language, _ -> 0.0; (; config...)...,
        rng=MersenneTwister(1111))
    @test result isa TypedSearchResult
    @test result.stats.operator_stats[:insert].changed == 1
    @test result.stats.operator_stats[:insert].accepted == 1
    @test validate(result.population[1].term, language) === true
    @test length(result.population[1].term.nodes) == 3
    annealed = _typed_search_options(acceptance_temperature=1.0,
        loss_normalization=1.0)
    @test _typed_acceptance_probability(0.0, 1.0, annealed, :insert) ≈ exp(-1.0)
end

@testitem "TypedSearch fixed-temperature acceptance is opt-in and bounded" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _typed_acceptance_probability
    using LambdaRegression.TypeSystem: FloatTy
    using Random: MersenneTwister

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seed = TypedTerm(TermNode[leaf_node(scalar, zero)])
    objective = term -> any(node -> node.kind == LEAF_KIND && node.x == zero,
        term.nodes) ? 0.0 : 1.0

    plain_config = _typed_search_options(population_size=1, attempts=24,
        evaluations=32, kinds=(:point,), weights=(1.0,))
    explicit_zero = _typed_search_options(population_size=1, attempts=24,
        evaluations=32, kinds=(:point,), weights=(1.0,),
        acceptance_temperature=0.0, loss_normalization=3.0)
    plain_rng = MersenneTwister(8000)
    zero_rng = MersenneTwister(8000)
    plain = search_terms([seed], language, objective; (; plain_config...)...,
        rng=plain_rng)
    zero_disabled = search_terms([seed], language, objective; (; explicit_zero...)...,
        rng=zero_rng)
    @test [member.term.nodes for member in zero_disabled.population] ==
        [member.term.nodes for member in plain.population]
    @test zero_disabled.stats.acceptance_rejected == 0
    @test rand(plain_rng) == rand(zero_rng)

    rejecting = search_terms([seed], language, objective;
        (; _typed_search_options(population_size=1, attempts=24, evaluations=32,
            kinds=(:point,), weights=(1.0,), max_nodes=8,
            frequency_strength=0.2, frequency_window=32,
            acceptance_temperature=1.0e-12)...)...,
        rng=MersenneTwister(42))
    @test rejecting.stats.acceptance_rejected > 0
    @test rejecting.stats.accepted == 0
    @test rejecting.population[1].birth == 1
    # One population observation is added to the scalar helper's unit
    # pseudocounts; a rejected child must not add another observation.
    @test rejecting.stats.frequency_summary.total ≈ 9.0
    @test length(rejecting.frontier) == 1
    point = rejecting.stats.operator_stats[:point]
    @test point.accepted == point.loss_improvement_count +
        point.loss_degradation_count + point.loss_equal_count
    @test rejecting.stats.evaluations <= 32

    enabled = _typed_search_options(acceptance_temperature=2.0,
        loss_normalization=3.0)
    @test _typed_acceptance_probability(-2.0, -3.0, enabled, :point) == 1.0
    @test _typed_acceptance_probability(-2.0, 1.0, enabled, :point) ≈ exp(-0.5)
    @test _typed_acceptance_probability(-2.0, 1.0, enabled, :crossover) == 1.0
    @test _typed_acceptance_probability(-2.0, 1.0, enabled, :simplify) == 1.0
    @test _typed_acceptance_probability(-2.0, 1.0, enabled, :simplify_atoms) == 1.0
    @test _typed_acceptance_probability(-floatmax(Float64), floatmax(Float64),
        _typed_search_options(acceptance_temperature=1.0,
            loss_normalization=floatmax(Float64)), :point) ≈ exp(-2.0)
    @test _typed_acceptance_probability(floatmax(Float64), -floatmax(Float64),
        enabled, :point) == 1.0
    @test_throws ArgumentError _typed_acceptance_probability(NaN, 0.0, enabled, :point)
    @test_throws ArgumentError search_terms([seed], language, objective;
        (; _typed_search_options(acceptance_temperature=-1.0)...)...)
    @test_throws ArgumentError search_terms([seed], language, objective;
        (; _typed_search_options(loss_normalization=Inf)...)...)

    # The acceptance gate does not weaken higher-order type preservation.
    fn = arrow!(language.types, scalar, scalar)
    target = arrow!(language.types, fn, fn)
    higher = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(target, 4)])
    @test validate(higher, language) === true
    higher_result = search_terms([higher], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=6, evaluations=8,
            kinds=(:point,), weights=(1.0,), acceptance_temperature=0.5)...)...,
        rng=MersenneTwister(555))
    @test higher_result isa TypedSearchResult
    @test all(member -> validate(member.term, language) === true,
        higher_result.population)

    grouped = search_populations([seed], language, objective;
        (; _typed_search_options(population_size=1, attempts=12, evaluations=16,
            kinds=(:point,), weights=(1.0,), acceptance_temperature=1.0e-12)...)...,
        rng=MersenneTwister(2000), group_count=2, chunk_attempts=1)
    @test grouped.stats.acceptance_rejected ==
        sum(stats.acceptance_rejected for stats in grouped.stats.population_stats)
    @test grouped.stats.evaluations == grouped.stats.initial_evaluations +
        sum(stats.evaluations for stats in grouped.stats.population_stats)
end

@testitem "TypedSearch operator is opt-in, scored, and written-budget guarded" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypedSearch: _typed_acceptance_probability

    # A successful operator proposal is recorded and passes the generic
    # acceptance gate, while the default operator tuple remains untouched.
    simple = Language(TypeTable())
    scalar = intern!(simple.types, FloatTy)
    fn = arrow!(simple.types, scalar, scalar)
    f = add_leaf!(simple, fn, :f)
    g = add_leaf!(simple, fn, :g)
    argument = add_leaf!(simple, scalar, :argument)
    seed = TypedTerm(TermNode[
        leaf_node(fn, f), leaf_node(scalar, argument), app_node(scalar, 1, 2),
    ])
    @test validate(seed, simple) === true
    config = _typed_search_options(population_size=1, attempts=1, evaluations=2,
        max_nodes=3, kinds=(:operator,), weights=(1.0,),
        acceptance_temperature=1.0, loss_normalization=1.0)
    result = search_terms([seed], simple, _ -> 0.0; (; config...)...,
        rng=MersenneTwister(8000))
    @test result isa TypedSearchResult
    operator = result.stats.operator_stats[:operator]
    @test operator.attempts == 1
    @test operator.changed == 1
    @test operator.evaluated == 1
    @test operator.accepted == 1
    @test operator.loss_equal_count == 1
    @test result.population[1].term.nodes[1] == leaf_node(fn, g)
    @test _typed_acceptance_probability(0.0, 1.0, config, :operator) ≈ exp(-1.0)
    @test _typed_search_options().kinds == (:generate, :crossover, :hoist, :point)

    # One admitted donor makes compact replacement look small while the
    # written-size guard rejects its SourceProgram head alternative.
    source_language = Language(TypeTable())
    source_scalar = intern!(source_language.types, FloatTy)
    source_fn = arrow!(source_language.types, source_scalar, source_scalar)
    source_ctx = Dict{Symbol,TyLike}(:one => FloatTy)
    mid = register_program!(source_language, :mid,
        AstTree(AbsNode(:x, ConstNode(:one))), source_ctx, TyFun(FloatTy, FloatTy))
    @test mid isa Integer
    source_f = add_leaf!(source_language, source_fn, :f)
    source_argument = add_leaf!(source_language, source_scalar, :argument)
    source_seed = TypedTerm(TermNode[
        leaf_node(source_fn, source_f),
        leaf_node(source_scalar, source_argument),
        app_node(source_scalar, 1, 2),
    ])
    @test validate(source_seed, source_language) === true
    @test written_complexity(TypedTerm(TermNode[
        leaf_node(source_fn, mid)]), source_language) == 2
    guarded = search_terms([source_seed], source_language, _ -> 0.0;
        (; _typed_search_options(population_size=1, attempts=1, evaluations=4,
            max_nodes=3, kinds=(:operator,), weights=(1.0,))...)...,
        rng=MersenneTwister(42))
    @test guarded isa TypedSearchResult
    @test guarded.stats.evaluations == 1
    @test guarded.stats.accepted == 0
    @test guarded.stats.operator_stats[:operator].attempts == 1
    @test guarded.stats.operator_stats[:operator].failures == 1
    @test guarded.stats.operator_stats[:operator].evaluated == 0
    @test guarded.stats.errors[:written_size_budget_exceeded] == 1
    @test guarded.population[1].term.nodes == source_seed.nodes
end

@testitem "TypedSearch age fair candidate requests charge legal repeats" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    policy = TypedProposalPolicy(
        weights=(generate=1.0, crossover=0.0, atomic_rewire=0.0,
            operator=0.0, insert=0.0, hoist=0.0, point=0.0, simplify=0.0),
        generation_mode=:grow, depth_tiers=(0,), size_tiers=(1,),
        generation_attempts=1, double_edit_probability=0.0)
    calls = Ref(0)
    config = _typed_search_options(population_size=2, attempts=8, evaluations=5,
        kinds=(:point,), weights=(1.0,))
    result = search_terms([seed], language, _ -> (calls[] += 1; 0.0);
        (; config...)..., rng=MersenneTwister(555), proposal_policy=policy,
        budget_mode=:candidate_requests)
    @test result isa TypedSearchResult
    @test result.stats.budget_mode === :candidate_requests
    @test result.stats.score_requests == 5
    @test result.stats.evaluations == calls[] == 1
    @test result.stats.unchanged == 4
    @test result.stats.score_requests >= result.stats.evaluations
    @test result.stats.population_stats[1].score_requests == 4

    # Explicitly spelling all new defaults preserves the historical trajectory
    # and result members exactly.
    plain = search_terms([seed], language, _ -> 0.0; (; config...)...,
        rng=MersenneTwister(1111))
    explicit = search_terms([seed], language, _ -> 0.0; (; config...)...,
        rng=MersenneTwister(1111), proposal_policy=nothing,
        diverse_initialization=false, budget_mode=:objective_calls)
    @test explicit.stats == plain.stats
    @test [m.term.nodes for m in explicit.population] ==
        [m.term.nodes for m in plain.population]
    @test [m.birth for m in explicit.population] ==
        [m.birth for m in plain.population]

    # The same opt-in accounting and toolbox route are available through the
    # shared multi-population entry point; requests remain globally bounded.
    calls[] = 0
    populations = search_populations([seed], language,
        _ -> (calls[] += 1; 0.0); (; config...)..., rng=MersenneTwister(555),
        group_count=2, chunk_attempts=1, proposal_policy=policy,
        budget_mode=:candidate_requests)
    @test populations isa TypedPopulationResult
    @test populations.stats.score_requests == 5
    @test populations.stats.evaluations == calls[] == 1
    @test sum(s.score_requests for s in populations.stats.population_stats) == 4
    @test populations.stats.score_requests <= config.evaluations
end

@testitem "TypedSearch age diverse initialization shares budget and records fallback" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    policy = TypedProposalPolicy(
        weights=(generate=1.0, crossover=0.0, atomic_rewire=0.0,
            operator=0.0, insert=0.0, hoist=0.0, point=0.0, simplify=0.0),
        generation_mode=:grow, depth_tiers=(0,), size_tiers=(1,),
        generation_attempts=1, double_edit_probability=0.0)
    calls = Ref(0)
    result = search_populations([seed], language, _ -> (calls[] += 1; 0.0);
        (; _typed_search_options(population_size=3, attempts=2, evaluations=3)...)...,
        rng=MersenneTwister(2000), group_count=2,
        proposal_policy=policy, diverse_initialization=true,
        budget_mode=:candidate_requests)
    @test result isa TypedPopulationResult
    @test calls[] == result.stats.evaluations == 3
    @test result.stats.score_requests == 3
    @test result.stats.attempts == 2
    @test result.stats.initialization_attempts == 2
    @test result.stats.initialization_evaluations == 2
    @test result.stats.initialization_invalid_evaluations == 0
    @test result.stats.initialization_fallbacks == 2
    @test result.stats.replicated_seeds == 3
    @test all(length(population) == 3 for population in result.populations)
    @test all(validate(member.term, language) === true for population in result.populations
        for member in population)
    @test all(s.initialization_fallbacks >= 0 for s in result.stats.population_stats)
end

@testitem "TypedSearch age toolbox uses ordinary donor and archive H" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    seed = TypedTerm(TermNode[leaf_node(scalar, one)])
    policy = TypedProposalPolicy(
        weights=(generate=1.0, crossover=0.0, atomic_rewire=0.0,
            operator=0.0, insert=0.0, hoist=0.0, point=0.0, simplify=0.0),
        generation_mode=:grow, depth_tiers=(0,), size_tiers=(1,),
        generation_attempts=1, double_edit_probability=0.0)
    result = search_terms([seed], language, _ -> 0.0;
        (; _typed_search_options(population_size=2, attempts=1, evaluations=2,
            archive_probability=1.0, kinds=(:point,), weights=(1.0,))...)...,
        rng=MersenneTwister(1), proposal_policy=policy)
    @test result isa TypedSearchResult
    @test result.stats.archive_parent_feedback == 1
    @test result.stats.archive_donor_feedback == 1
    @test result.stats.score_requests == 2
    @test result.stats.evaluations == 1
    @test result.stats.operator_stats[:generate].attempts == 1
    @test result.stats.operator_stats[:generate].accepted == 0
    @test sum(values(result.stats.kind_counts)) == result.stats.attempts
    @test !haskey(result.stats.kind_counts, :toolbox)
    @test !haskey(result.stats.operator_stats, :toolbox)
    @test all(validate(member.term, language) === true for member in result.population)
end

@testitem "TypedSearch age records ordinary eta kind without toolbox phantom" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(fn, :identity)])
    seed = TypedTerm(TermNode[
        leaf_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3),
    ])
    config = _typed_search_options(population_size=1, attempts=1, evaluations=2,
        kinds=(:eta_reduce,), weights=(1.0,))
    result = search_terms([seed], language, _ -> 0.0;
        (; config...)..., rng=MersenneTwister(42))
    @test result isa TypedSearchResult
    @test result.stats.attempts == 1
    @test result.stats.attempts == result.stats.changed + result.stats.failures +
        result.stats.unchanged
    @test result.stats.kind_counts[:eta_reduce] == 1
    @test !haskey(result.stats.kind_counts, :toolbox)
    @test result.stats.operator_stats[:eta_reduce].attempts == 1
    @test !haskey(result.stats.operator_stats, :toolbox)
    @test result.population[1].term.nodes == TermNode[leaf_node(fn, 1)]
end

@testitem "TypedSearch NSGA-II grouped switches cover real generations and owned outputs" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_search_options

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    seeds = [
        TypedTerm(TermNode[leaf_node(scalar, one)]),
        TypedTerm(TermNode[leaf_node(scalar, zero)]),
    ]
    config = _typed_search_options(population_size=2, attempts=4, evaluations=6,
        max_nodes=1, max_depth=0, kinds=(:generate,), weights=(1.0,))

    for (index, (group_mating, group_survival, group_material)) in enumerate(
            Iterators.product((false, true), (false, true), (false, true)))
        calls = Ref(0)
        objective = term -> begin
            calls[] += 1
            # The result is an actual objective payload, and every call owns a
            # fresh vector that the search must copy before retaining it.
            (loss=0.0, outputs=[Float64(calls[]), Float64(length(term.nodes))])
        end
        state = EvaluationState()
        result = search_nsga2(seeds, language, objective;
            (; config...)..., rng=MersenneTwister(1950 + index), classifier=_ -> :one,
            material_probability=1.0, group_mating, group_survival,
            group_material, evaluation_state=state)

        returned = vcat(result.populations...)
        append!(returned, result.frontier)
        @test calls[] == result.stats.evaluations == 6
        @test result.stats.attempts == 4
        @test result.stats.offspring == 4
        @test all(member -> validate(member.term, language) === true, returned)
        @test all(member -> begin
            output = evaluation_outputs(state, member)
            output isa Vector{Float64} && !isempty(output) &&
                all(isfinite, output)
        end, returned)
        @test result.stats.features.behavior_grouping
        @test result.stats.features.group_mating === group_mating
        @test result.stats.features.group_survival === group_survival
        @test result.stats.features.group_material === group_material
        @test result.stats.features.material_probability == 1.0
        live_count = sum(length, result.populations)
        @test length(state.outputs) <= live_count + length(result.frontier)
    end

    # An evaluation state can retain strict output payloads without enabling
    # grouping.  Its output copies must not alter the scalar search trajectory.
    cache_calls = Ref(0)
    source_buffers = Vector{Vector{Float64}}()
    cache_objective = term -> begin
        cache_calls[] += 1
        buffer = [Float64(cache_calls[]), Float64(length(term.nodes))]
        push!(source_buffers, buffer)
        (loss=0.0, outputs=buffer)
    end
    cache_state = EvaluationState()
    cache = search_nsga2(seeds, language, cache_objective;
        (; config...)..., rng=MersenneTwister(1), evaluation_state=cache_state,
        group_mating=false, group_survival=false, group_material=false)

    scalar_calls = Ref(0)
    scalar_result = search_nsga2(seeds, language, _ -> begin
        scalar_calls[] += 1
        0.0
    end; (; config...)..., rng=MersenneTwister(1), group_mating=false,
        group_survival=false, group_material=false)
    @test cache_calls[] == scalar_calls[] == cache.stats.evaluations ==
        scalar_result.stats.evaluations == 6
    @test cache.stats.attempts == scalar_result.stats.attempts == 4
    @test cache.stats.offspring == scalar_result.stats.offspring == 4
    @test cache.stats.generations == scalar_result.stats.generations
    same_members = function (left, right)
        length(left) == length(right) || return false
        return all(zip(left, right)) do pair
            a, b = pair
            a.loss == b.loss && a.birth == b.birth &&
                a.term.nodes == b.term.nodes
        end
    end
    @test same_members(cache.populations[1], scalar_result.populations[1])
    @test same_members(cache.frontier, scalar_result.frontier)

    retained = vcat(cache.populations...)
    append!(retained, cache.frontier)
    owned = [(member, evaluation_outputs(cache_state, member))
        for member in retained]
    @test all(pair -> pair[2] isa Vector{Float64} &&
        all(isfinite, pair[2]), owned)
    for buffer in source_buffers
        fill!(buffer, -1000.0)
    end
    @test all(pair -> evaluation_outputs(cache_state, pair[1]) == pair[2], owned)
    @test all(pair -> begin
        retained_output = get(cache_state.outputs, pair[1].term.nodes, nothing)
        retained_output isa Vector{Float64} &&
            all(buffer -> retained_output !== buffer, source_buffers)
    end, owned)
end
