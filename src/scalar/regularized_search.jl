"""
Minimal regularized-evolution scalar search.

This file implements regularized-evolution scalar search following the
SymbolicRegression.jl algorithm.  It deliberately
reuses this package's `ScalarExpr`, evaluator, mutation palette, and accounting
types; it is not a source translation and it does not define a compatibility
layer.  The file is included after `search.jl` so that the implementation can
remain a narrow, replaceable seam.
"""

"""A member keeps its insertion clock separately from its fitness."""
mutable struct _RegularizedMember
    candidate::_ScalarCandidate
    born::Int
end

_regularized_candidate(member::_RegularizedMember) = member.candidate

# These are the default scalar-tree weights in SymbolicRegression.jl v1.13.4.
# `form_connection` and `break_connection` are GraphNode-only and therefore
# intentionally have no ScalarExpr analogue.  `optimize` remains zero here:
# constant refinement keeps its separate budget and accounting in this package.
const _REGULARIZED_MUTATION_WEIGHTS = (
    # SymbolicRegression.jl v1.13.4 Options() defaults (normalized only when
    # sampled; these values are intentionally kept in their source scale).
    mutate_constant=0.0346,
    mutate_operator=0.293,
    swap_operands=0.198,
    rotate_tree=4.26,
    add_node=2.47,
    insert_node=0.0112,
    delete_node=0.870,
    simplify=0.00209,
    randomize=0.000502,
    do_nothing=0.273,
    optimize=0.0,
)

const _REGULARIZED_MUTATION_NAMES = (
    :constant, :operator, :swap, :rotate, :add, :insert, :delete,
    :simplify, :restart, :do_nothing, :optimize,
)

"""One occurrence in a scalar tree and the eligibility facts used by mutation."""
struct _RegularizedMutationAnchor
    path::Tuple{Vararg{Int}}
    node::ScalarExpr
    has_children::Bool
    is_rotatable::Bool
    is_constant::Bool
    is_binary::Bool
    has_alternate_operator::Bool
end

"""Single traversal of a parent reused by mutation selection and application."""
struct _RegularizedMutationScan
    anchors::Vector{_RegularizedMutationAnchor}
    has_constant::Bool
    has_alternate_operator::Bool
    has_binary::Bool
    has_children::Bool
    has_rotation::Bool
    complexity::Int
end

mutable struct _RegularizedMutationFacts
    has_constant::Bool
    has_alternate_operator::Bool
    has_binary::Bool
    has_children::Bool
    has_rotation::Bool
end

function _regularized_collect_mutation_anchors!(anchors,
        expr::ScalarExpr, path, unary_operators, binary_operators,
        facts::_RegularizedMutationFacts)
    has_children = _has_children(expr)
    is_rotatable = _is_rotatable(expr)
    is_constant = expr isa _ScalarConstant
    is_binary = expr isa _ScalarBinary
    has_alternate_operator = expr isa _ScalarUnary ?
        any(!=(expr.op), unary_operators) :
        expr isa _ScalarBinary ? any(!=(expr.op), binary_operators) : false
    push!(anchors, _RegularizedMutationAnchor(path, expr, has_children,
        is_rotatable, is_constant, is_binary, has_alternate_operator))
    facts.has_constant |= is_constant
    facts.has_alternate_operator |= has_alternate_operator
    facts.has_binary |= is_binary
    facts.has_children |= has_children
    facts.has_rotation |= is_rotatable
    if expr isa _ScalarUnary
        _regularized_collect_mutation_anchors!(anchors, expr.arg, (path..., 1),
            unary_operators, binary_operators, facts)
    elseif expr isa _ScalarBinary
        _regularized_collect_mutation_anchors!(anchors, expr.left, (path..., 1),
            unary_operators, binary_operators, facts)
        _regularized_collect_mutation_anchors!(anchors, expr.right, (path..., 2),
            unary_operators, binary_operators, facts)
    end
    return anchors
end

function _regularized_mutation_scan(parent::ScalarExpr,
        config::ScalarSearchConfig)
    anchors = _RegularizedMutationAnchor[]
    facts = _RegularizedMutationFacts(false, false, false, false, false)
    _regularized_collect_mutation_anchors!(anchors, parent, (),
        config.unary_operators, config.binary_operators, facts)
    return _RegularizedMutationScan(anchors,
        facts.has_constant,
        facts.has_alternate_operator,
        facts.has_binary,
        facts.has_children,
        facts.has_rotation,
        length(anchors))
end

_same_scalar_expression(::_ScalarInput, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarConstant, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarUnary, ::ScalarExpr) = false
_same_scalar_expression(::_ScalarBinary, ::ScalarExpr) = false
_same_scalar_expression(left::_ScalarInput, right::_ScalarInput) =
    left.index == right.index
_same_scalar_expression(left::_ScalarConstant, right::_ScalarConstant) =
    left.value == right.value
_same_scalar_expression(left::_ScalarUnary, right::_ScalarUnary) =
    left.op === right.op && _same_scalar_expression(left.arg, right.arg)
_same_scalar_expression(left::_ScalarBinary, right::_ScalarBinary) =
    left.op === right.op &&
    _same_scalar_expression(left.left, right.left) &&
    _same_scalar_expression(left.right, right.right)

"""Choose one parent-anchored mutation using conditioned SR reference weights.

Eligibility is derived from the parent's node kinds and
`config.unary_operators`, `config.binary_operators`, and `max_complexity`.
When no weighted operation is eligible, the chooser returns `:do_nothing`;
invalid configuration is rejected by `ScalarSearchConfig` before this seam.
"""
function _regularized_sample_mutation(rng::AbstractRNG, parent::ScalarExpr,
        config::ScalarSearchConfig; scan=nothing)
    names = _REGULARIZED_MUTATION_NAMES
    values = (
        _REGULARIZED_MUTATION_WEIGHTS.mutate_constant,
        _REGULARIZED_MUTATION_WEIGHTS.mutate_operator,
        _REGULARIZED_MUTATION_WEIGHTS.swap_operands,
        _REGULARIZED_MUTATION_WEIGHTS.rotate_tree,
        _REGULARIZED_MUTATION_WEIGHTS.add_node,
        _REGULARIZED_MUTATION_WEIGHTS.insert_node,
        _REGULARIZED_MUTATION_WEIGHTS.delete_node,
        _REGULARIZED_MUTATION_WEIGHTS.simplify,
        _REGULARIZED_MUTATION_WEIGHTS.randomize,
        _REGULARIZED_MUTATION_WEIGHTS.do_nothing,
        _REGULARIZED_MUTATION_WEIGHTS.optimize,
    )
    scan === nothing && (scan = _regularized_mutation_scan(parent, config))
    eligible = Bool[
        scan.has_constant,
        scan.has_alternate_operator,
        scan.has_binary,
        scan.has_rotation,
        scan.complexity < config.max_complexity,
        scan.complexity < config.max_complexity,
        scan.has_children,
        true,
        true,
        true,
        false, # optimize remains zero; refinement has its independent budget
    ]
    values = [value * eligible[index] for (index, value) in enumerate(values)]
    total = sum(values)
    total > 0 || return :do_nothing
    draw = rand(rng) * total
    cumulative = 0.0
    for (name, weight) in zip(names, values)
        cumulative += weight
        draw < cumulative && return name
    end
    return last(names)
end

"""Apply one named SR mutation at a valid subtree anchor.

The parent supplies all candidate anchors; `feature_count` and `config` supply
the input/operator context. Inapplicable or complexity-overflowing operations
return the unchanged parent and are classified as skipped by the search loop;
unexpected mutation names also leave the parent unchanged.
"""
function _regularized_forced_mutation(rng::AbstractRNG, parent::ScalarExpr,
        feature_count::Int, config::ScalarSearchConfig, kind::Symbol;
        scan=nothing)
    kind === :do_nothing && return (parent, kind)
    kind === :optimize && return (parent, kind)
    kind === :simplify && return (_simplify_scalar(parent), kind)
    if kind === :restart
        candidate = _random_expression(rng, feature_count, config, config.initial_depth)
        return (_bounded_candidate(candidate, parent, config), kind)
    end
    scan === nothing && (scan = _regularized_mutation_scan(parent, config))
    anchors = scan.anchors
    if kind === :add
        candidate = if rand(rng, Bool)
            leaf_anchors = filter(anchor -> !anchor.has_children, anchors)
            anchor = rand(rng, leaf_anchors)
            _replace_subexpression(parent, anchor.path,
                _regularized_fresh_operation(rng, feature_count, config))
        else
            _regularized_wrap_operation(rng, parent, feature_count, config)
        end
        return (_bounded_candidate(candidate, parent, config), kind)
    end
    compatible_anchors = if kind === :operator
        filter(anchor -> anchor.has_alternate_operator, anchors)
    elseif kind === :swap
        filter(anchor -> anchor.is_binary, anchors)
    elseif kind === :constant
        filter(anchor -> anchor.is_constant, anchors)
    elseif kind === :rotate
        filter(anchor -> anchor.is_rotatable, anchors)
    elseif kind === :delete
        filter(anchor -> anchor.has_children, anchors)
    else
        anchors
    end
    isempty(compatible_anchors) && return (parent, kind)
    anchor = rand(rng, compatible_anchors)
    path = anchor.path
    subtree = anchor.node
    replacement = if kind === :insert
        _regularized_wrap_operation(rng, subtree, feature_count, config)
    elseif kind === :rotate
        _rotate_expression(rng, subtree)
    elseif kind === :delete
        subtree isa _ScalarUnary ? subtree.arg :
            (rand(rng, Bool) ? subtree.left : subtree.right)
    elseif kind === :operator
        if subtree isa _ScalarUnary
            operators = filter(!=(subtree.op), collect(config.unary_operators))
            isempty(operators) && return (parent, kind)
            scalar_unary(rand(rng, operators), subtree.arg)
        else
            operators = filter(!=(subtree.op), collect(config.binary_operators))
            isempty(operators) && return (parent, kind)
            scalar_binary(rand(rng, operators), subtree.left, subtree.right)
        end
    elseif kind === :swap
        scalar_binary(subtree.op, subtree.right, subtree.left)
    elseif kind === :constant
        scalar_constant(subtree.value + randn(rng) * 0.25 * config.constant_scale)
    else
        # A sampled operation with no ScalarExpr analogue is a no-op rather
        # than an invented mutation.  This currently only covers :optimize,
        # whose independent refinement budget is handled by the caller.
        return (parent, kind)
    end
    candidate = _replace_subexpression(parent, path, replacement)
    return (_bounded_candidate(candidate, parent, config), kind)
end

"""Choose unary versus binary construction from the configured operator world.

The choice is anchored only to the number of enabled operators. It requires at
least one configured operator (validated by `ScalarSearchConfig`) and cannot
fail after configuration validation.
"""
function _regularized_choose_binary(rng::AbstractRNG, config::ScalarSearchConfig)
    isempty(config.binary_operators) && return false
    isempty(config.unary_operators) && return true
    total = length(config.binary_operators) + length(config.unary_operators)
    return rand(rng) < length(config.binary_operators) / total
end

"""Create a fresh one-level operation anchored to valid feature leaves.

`feature_count` defines the available input context and `config` supplies the
operator world and constant-leaf probability. Invalid feature/operator
contexts are rejected by the public search/configuration boundary.
"""
function _regularized_fresh_operation(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig)
    if _regularized_choose_binary(rng, config)
        return scalar_binary(rand(rng, config.binary_operators),
            _random_leaf(rng, feature_count, config),
            _random_leaf(rng, feature_count, config))
    end
    return scalar_unary(rand(rng, config.unary_operators),
        _random_leaf(rng, feature_count, config))
end

"""Wrap `expr` with one configured operation and a fresh leaf when required.

The existing expression remains one anchored child; the other child is drawn
from the configured feature/constant context. Complexity overflow is handled
by the caller's bounded-candidate rule rather than by this builder.
"""
function _regularized_wrap_operation(rng::AbstractRNG, expr::ScalarExpr,
        feature_count::Int, config::ScalarSearchConfig)
    if _regularized_choose_binary(rng, config)
        return scalar_binary(rand(rng, config.binary_operators), expr,
            _random_leaf(rng, feature_count, config))
    end
    return scalar_unary(rand(rng, config.unary_operators), expr)
end

"""Sample an eligible mutation and dispatch it against the same parent.

Eligibility and failure behavior are those of `_regularized_sample_mutation`
and `_regularized_forced_mutation`; this wrapper adds no fallback search or
formula-specific proposal.
"""
function _regularized_mutation(rng::AbstractRNG, parent::ScalarExpr,
        feature_count::Int, config::ScalarSearchConfig)
    scan = _regularized_mutation_scan(parent, config)
    kind = _regularized_sample_mutation(rng, parent, config; scan=scan)
    return _regularized_forced_mutation(rng, parent, feature_count, config, kind;
        scan=scan)
end

function _regularized_initial_expressions(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig, initial_population)
    expressions = ScalarExpr[]
    if initial_population !== nothing
        append!(expressions, initial_population)
    end
    if length(expressions) > config.population_size
        resize!(expressions, config.population_size)
    end
    while length(expressions) < config.population_size
        candidate = _random_expression(rng, feature_count, config, config.initial_depth)
        scalar_complexity(candidate) <= config.max_complexity && push!(expressions, candidate)
    end
    return expressions
end

function _regularized_oldest_index(population::Vector{_RegularizedMember})
    isempty(population) && throw(ArgumentError("regularized population cannot be empty"))
    best_index = firstindex(population)
    best_key = (population[best_index].born, population[best_index].candidate.serial)
    for index in (firstindex(population) + 1):lastindex(population)
        member = population[index]
        key = (member.born, member.candidate.serial)
        key < best_key || continue
        best_index, best_key = index, key
    end
    return best_index
end

"""Running complexity frequencies used by SR's adaptive parsimony."""
mutable struct _RegularizedFrequency
    window::Int
    frequencies::Vector{Float64}
    normalized_frequencies::Vector{Float64}
end

function _RegularizedFrequency(window::Int, max_complexity::Int)
    window >= 1 || throw(ArgumentError("frequency_window must be positive"))
    max_complexity >= 1 || throw(ArgumentError("max_complexity must be positive"))
    frequencies = ones(Float64, max_complexity)
    return _RegularizedFrequency(window, frequencies, copy(frequencies) ./ sum(frequencies))
end

function _regularized_move_window!(frequency::_RegularizedFrequency)
    difference = sum(frequency.frequencies) - frequency.window
    loops = 0
    while difference > 0 && loops < length(frequency.frequencies)
        indices = findall(value -> value > 1.0, frequency.frequencies)
        isempty(indices) && break
        amount = min(difference / length(indices),
            minimum(frequency.frequencies[indices]) - 1.0)
        # Even a one-ulp excess above the pseudocount floor must be removed.
        # Otherwise that smallest bucket blocks trimming every larger bucket
        # on all later observations. Each floor-limited pass removes at least
        # one active bucket, so the number of buckets bounds the passes.
        amount <= 0.0 && break
        frequency.frequencies[indices] .-= amount
        difference -= amount * length(indices)
        loops += 1
    end
    return frequency
end

@testitem "Scalar frequency window trims near-floor buckets" begin
    using LambdaRegression.ScalarSearch: _RegularizedFrequency,
        _regularized_move_window!, _regularized_normalize!
    using Random: MersenneTwister, rand

    state = _RegularizedFrequency(10, 2)
    state.frequencies .= [nextfloat(1.0), 100.0]
    _regularized_move_window!(state)
    @test sum(state.frequencies) <= 10 + 1e-9
    @test all(>=(1.0), state.frequencies)

    state = _RegularizedFrequency(4096, 254)
    rng = MersenneTwister(1234)
    for _ in 1:200
        for size in rand(rng, 1:254, 64)
            state.frequencies[size] += 1.0
        end
        _regularized_move_window!(state)
        @test sum(state.frequencies) <= 4096 + 1e-8
        @test all(>=(1.0), state.frequencies)
    end
    _regularized_normalize!(state)
    @test sum(state.normalized_frequencies) ≈ 1.0
end

function _regularized_normalize!(frequency::_RegularizedFrequency)
    total = sum(frequency.frequencies)
    total > 0 || throw(ArgumentError("running complexity frequencies must be positive"))
    frequency.normalized_frequencies .= frequency.frequencies ./ total
    return frequency
end

function _regularized_observe_population!(frequency::_RegularizedFrequency,
        population::Vector{_RegularizedMember})
    for member in population
        complexity = scalar_complexity(member.candidate.expr)
        0 < complexity <= length(frequency.frequencies) || continue
        frequency.frequencies[complexity] += 1.0
    end
    _regularized_move_window!(frequency)
    return _regularized_normalize!(frequency)
end

function _regularized_frequency(frequency::_RegularizedFrequency, complexity::Int)
    0 < complexity <= length(frequency.normalized_frequencies) || return 1.0
    return frequency.normalized_frequencies[complexity]
end

"""
    _regularized_acceptance_probability(parent_loss, child_loss; ...)

Compute the probability used by the regularized loop.  With annealing and
frequency correction both disabled, every finite child is accepted (including
a worse child); this is the regularized-evolution replacement rule rather than
elitist hill climbing.  Frequency correction is intentionally finite-window
and uses the running, normalized complexity counts maintained once per cycle.
"""
function _regularized_acceptance_probability(parent_loss::Real, child_loss::Real;
        annealing::Bool=false, temperature::Real=1.0, alpha::Real=1.0,
        old_frequency::Real=1.0, new_frequency::Real=1.0)
    isfinite(child_loss) || return 0.0
    probability = 1.0
    if annealing
        temperature > 0 || throw(ArgumentError("temperature must be positive"))
        alpha > 0 || throw(ArgumentError("alpha must be positive"))
        delta = Float64(child_loss) - Float64(parent_loss)
        delta > 0 && (probability *= exp(-delta / (Float64(temperature) * Float64(alpha))))
    end
    if old_frequency != 1.0 || new_frequency != 1.0
        old_frequency >= 0 || throw(ArgumentError("old_frequency must be nonnegative"))
        new_frequency >= 0 || throw(ArgumentError("new_frequency must be nonnegative"))
        probability *= old_frequency / max(new_frequency, eps(Float64))
    end
    return clamp(probability, 0.0, 1.0)
end

function _regularized_frequency_ratio(frequency::_RegularizedFrequency,
        old_complexity::Int, new_complexity::Int)
    return _regularized_frequency(frequency, old_complexity),
        _regularized_frequency(frequency, new_complexity)
end

"""Return the baseline-normalized cost used by SR selection and annealing."""
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

function _regularized_pareto_candidates(
        archive_by_complexity::Dict{Int,_ScalarCandidate})
    candidates = _ScalarCandidate[]
    best_loss = Inf
    for complexity in sort!(collect(keys(archive_by_complexity)))
        candidate = archive_by_complexity[complexity]
        candidate.loss < best_loss || continue
        push!(candidates, candidate)
        best_loss = candidate.loss
    end
    return candidates
end

"""Build one offspring from subtree-anchored mutation or crossover.

The operator world and size bound come from `config`. Mutation/crossover
retries are bounded at ten; exhaustion returns the first parent unchanged so
the caller can classify a skipped variation without spending a structural
evaluation. An injected test builder with an unsupported signature or return
type raises `ArgumentError`.
"""
function _regularized_child(rng::AbstractRNG, first_parent::_ScalarCandidate,
        second_parent::_ScalarCandidate, feature_count::Int, config::ScalarSearchConfig,
        offspring_builder)
    if offspring_builder !== nothing
        child = if applicable(offspring_builder, rng, first_parent.expr,
                second_parent.expr, feature_count, config)
            offspring_builder(rng, first_parent.expr, second_parent.expr,
                feature_count, config)
        elseif applicable(offspring_builder, rng, first_parent.expr,
                second_parent.expr, config)
            offspring_builder(rng, first_parent.expr, second_parent.expr, config)
        else
            throw(ArgumentError("offspring_builder has an unsupported signature"))
        end
        child isa ScalarExpr || throw(ArgumentError("offspring_builder must return ScalarExpr"))
        return (child, :injected)
    end
    draw = rand(rng)
    if draw < config.mutation_probability
        scan = _regularized_mutation_scan(first_parent.expr, config)
        kind = _regularized_sample_mutation(rng, first_parent.expr, config; scan=scan)
        kind in (:do_nothing, :simplify, :optimize) &&
            return _regularized_forced_mutation(
                rng, first_parent.expr, feature_count, config, kind; scan=scan)
        for _ in 1:10
            child, _ = _regularized_forced_mutation(
                rng, first_parent.expr, feature_count, config, kind; scan=scan)
            !_same_scalar_expression(child, first_parent.expr) &&
                return (child, kind)
        end
        return (first_parent.expr, kind)
    elseif draw < config.mutation_probability + config.crossover_probability
        for _ in 1:10
            child = _crossover_expression(
                rng, first_parent.expr, second_parent.expr, config)
            !_same_scalar_expression(child, first_parent.expr) &&
                return (child, :crossover)
        end
        return (first_parent.expr, :crossover)
    end
    return (_compose_expression(rng, first_parent.expr, second_parent.expr, config), :compose)
end

"""Select by an un-replaced rank tournament over the current population.

Costs use candidate loss, structural complexity, `config.parsimony_coefficient`,
and the supplied running complexity frequencies. Empty populations and an
invalid selection probability raise `ArgumentError`.
"""
function _regularized_tournament(rng::AbstractRNG,
        candidates::Vector{_ScalarCandidate}, config::ScalarSearchConfig,
        frequency::_RegularizedFrequency; loss_normalization::Real,
        selection_p::Real=0.982,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0)
    isempty(candidates) && throw(ArgumentError("regularized population cannot be empty"))
    selection_p = Float64(selection_p)
    0 < selection_p <= 1 || throw(ArgumentError("selection_p must be in (0, 1]"))
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
        complexity = scalar_complexity(candidate.expr)
        cost = _regularized_cost(candidate.loss, complexity,
            loss_normalization, config.parsimony_coefficient)
        if use_frequency_in_tournament
            normalized_frequency = _regularized_frequency(
                frequency, complexity)
            cost *= exp(Float64(adaptive_parsimony_scaling) * normalized_frequency)
        end
        costs[index] = cost
    end
    order = sortperm(costs)
    probabilities = [selection_p * (1 - selection_p)^(rank - 1) for rank in 1:n]
    draw = rand(rng) * sum(probabilities)
    cumulative = 0.0
    chosen_rank = n
    for rank in 1:n
        cumulative += probabilities[rank]
        draw < cumulative && (chosen_rank = rank; break)
    end
    return candidates[selected[order[chosen_rank]]]
end

"""Build a finite, loss-ordered report from regularized population state."""
function _regularized_report(population::Vector{_RegularizedMember},
        archive_by_complexity::Dict{Int,_ScalarCandidate}, best_expr::ScalarExpr,
        best_loss::Float64, evaluations::Int, invalid_evaluations::Int,
        constant_refinement_evaluations::Int, constant_refinement_invalid_evaluations::Int,
        generations::Int, termination::Symbol, improvements::Vector{NamedTuple},
        mutation_attempts::Int, crossover_attempts::Int, composition_attempts::Int,
        changed_variations::Int, mutation_kind_counts::Dict{Symbol,Int},
        accepted_offspring::Int, rejected_offspring::Int, skipped_variations::Int,
        oldest_replacements::Int, hof_migrations::Int,
        annealing::Bool, use_frequency::Bool, frequency_window::Int)
    best_expr = _simplify_scalar(best_expr)
    population_candidates = [_regularized_candidate(member) for member in population]
    report_candidates = copy(population_candidates)
    push!(report_candidates, _ScalarCandidate(best_expr, best_loss, 0, true))
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
        (expr=archive_by_complexity[complexity].expr,
            loss=archive_by_complexity[complexity].loss, complexity)
        for complexity in sort!(collect(keys(archive_by_complexity)))
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
            # Existing ScalarExpr-only mutations retained as zero-valued
            # compatibility fields for callers that inspect variation records.
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
    return ScalarSearchResult(best_expr, best_loss, evaluations, invalid_evaluations,
        constant_refinement_evaluations, constant_refinement_invalid_evaluations,
        generations, termination, improvements, leaders, complexity_archive, variation)
end

"""
    _search_scalar_regularized(X, y; kwargs...) -> ScalarSearchResult

Run the small coherent steady-state scalar search used for comparison with
SymbolicRegression.jl.  The initial population is evaluated once; each later
finite child is drawn by tournament, optionally accepted using annealing and a
complexity-frequency correction, and replaces the oldest member.  Structural
evaluation attempts consume `config.max_evaluations`; constant refinement uses
the independent budget already defined by `ScalarSearchConfig`.

Keyword arguments `annealing`, `temperature`, `alpha`,
`ncycles_per_iteration`, `use_frequency`, and `frequency_window` follow the
current SymbolicRegression.jl Options defaults.  Each regularized iteration
restarts its annealing schedule, simplifies/refines the population, and admits
a small Poisson sample from the complexity-stratified Pareto archive;
selection probability and adaptive-parsimony parameters are likewise exposed
only for comparison.  They remain deliberately internal and provisional.  The
`offspring_builder` seam is for deterministic narrow tests and is not a public
search API.
"""
function _search_scalar_regularized(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        config::ScalarSearchConfig=ScalarSearchConfig(), rng::AbstractRNG=default_rng(),
        initial_population=nothing, objective=nothing,
        stop_loss::Union{Nothing,Real}=1.0e-12,
        annealing::Bool=true, temperature::Real=1.0, alpha::Real=3.17,
        ncycles_per_iteration::Int=380,
        use_frequency::Bool=true, frequency_window::Int=100000,
        hof_migration_fraction::Real=0.0614,
        tournament_selection_p::Real=0.982,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0,
        initializer=nothing,
        offspring_builder=nothing)
    feature_count = size(X, 1)
    feature_count >= 1 || throw(ArgumentError("scalar search requires at least one feature"))
    size(X, 2) == length(y) || throw(DimensionMismatch("target length must match samples"))
    isempty(y) && throw(ArgumentError("scalar search requires at least one sample"))
    all(isfinite, X) || throw(ArgumentError("scalar search inputs must be finite"))
    all(isfinite, y) || throw(ArgumentError("scalar search targets must be finite"))
    stop_loss === nothing || (isfinite(stop_loss) && stop_loss >= 0) ||
        throw(ArgumentError("stop_loss must be nothing or finite and nonnegative"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    ncycles_per_iteration >= 1 ||
        throw(ArgumentError("ncycles_per_iteration must be positive"))
    frequency_window >= 1 || throw(ArgumentError("frequency_window must be positive"))
    0 <= hof_migration_fraction <= 1 ||
        throw(ArgumentError("hof_migration_fraction must be in [0, 1]"))
    0 < tournament_selection_p <= 1 ||
        throw(ArgumentError("tournament_selection_p must be in (0, 1]"))
    adaptive_parsimony_scaling >= 0 ||
        throw(ArgumentError("adaptive_parsimony_scaling must be nonnegative"))
    initial_population === nothing || all(expr -> expr isa ScalarExpr, initial_population) ||
        throw(ArgumentError("initial_population must contain ScalarExpr values"))
    initial_population === nothing ||
        all(expr -> scalar_complexity(expr) <= config.max_complexity, initial_population) ||
        throw(ArgumentError("initial_population exceeds max_complexity"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))

    expressions = if initializer === nothing
        _regularized_initial_expressions(rng, feature_count, config, initial_population)
    elseif applicable(initializer, rng, feature_count, config, initial_population)
        initializer(rng, feature_count, config, initial_population)
    else
        throw(ArgumentError("initializer has an unsupported signature"))
    end
    expressions isa AbstractVector || throw(ArgumentError("initializer must return a vector"))
    length(expressions) >= config.population_size ||
        throw(ArgumentError("initializer must fill population_size expressions"))
    all(expr -> expr isa ScalarExpr, expressions) ||
        throw(ArgumentError("initializer must return ScalarExpr values"))
    all(expr -> scalar_complexity(expr) <= config.max_complexity, expressions) ||
        throw(ArgumentError("initializer returned an expression above max_complexity"))
    population = _RegularizedMember[]
    archive_by_complexity = Dict{Int,_ScalarCandidate}()
    frequency = _RegularizedFrequency(frequency_window, config.max_complexity)
    evaluations = 0
    invalid_evaluations = 0
    constant_refinement_evaluations = 0
    constant_refinement_invalid_evaluations = 0
    serial = 0
    birth_clock = 0
    best_loss = Inf
    best_expr = first(expressions)
    improvements = NamedTuple[]
    mutation_attempts = 0
    crossover_attempts = 0
    composition_attempts = 0
    changed_variations = 0
    accepted_offspring = 0
    rejected_offspring = 0
    skipped_variations = 0
    oldest_replacements = 0
    hof_migrations = 0
    mutation_kind_counts = Dict{Symbol,Int}()

    function record_archive!(candidate::_ScalarCandidate)
        isfinite(candidate.loss) || return candidate
        complexity = scalar_complexity(candidate.expr)
        incumbent = get(archive_by_complexity, complexity, nothing)
        incumbent === nothing || _candidate_key(candidate) >= _candidate_key(incumbent) ||
            (archive_by_complexity[complexity] = candidate; return candidate)
        incumbent === nothing && (archive_by_complexity[complexity] = candidate)
        return candidate
    end

    function admit_candidate!(candidate::_ScalarCandidate, generation::Int;
            source::Union{Nothing,Symbol}=nothing)
        record_archive!(candidate)
        if _candidate_key(candidate) <
                (best_loss, scalar_complexity(best_expr), typemax(Int))
            best_loss = candidate.loss
            best_expr = candidate.expr
            entry = (evaluation=evaluations, generation,
                loss=candidate.loss, complexity=scalar_complexity(candidate.expr))
            push!(improvements, source === nothing ? entry : merge(entry, (; source)))
        end
        return candidate
    end

    function score_expression(expr)
        if objective === nothing
            prediction = evaluate_dynamic_scalar(expr, X)
            prediction isa Err && return (Inf, true)
            total = 0.0
            @inbounds @simd for index in eachindex(y)
                residual = prediction[index] - y[index]
                total += residual * residual
            end
            loss = total / length(y)
            return isfinite(loss) ? (loss, false) : (Inf, true)
        end
        value = if applicable(objective, expr)
            objective(expr)
        else
            dynamic = scalar_to_dynamic(expr)
            if applicable(objective, expr, dynamic)
                objective(expr, dynamic)
            else
                prepared = prepare_scalar(expr)
                applicable(objective, expr, prepared) || throw(ArgumentError(
                    "objective does not accept ScalarExpr, dynamic node, or prepared plan"))
                objective(expr, prepared)
            end
        end
        value isa Err && return (Inf, true)
        value isa Real || throw(ArgumentError("objective must return a real loss or Err"))
        loss = Float64(value)
        return isfinite(loss) && loss >= 0 ? (loss, false) : (Inf, true)
    end


    baseline_loss, baseline_invalid = score_expression(scalar_constant(0.0))
    loss_normalization = baseline_invalid || !isfinite(baseline_loss) ? 0.01 :
        max(0.01, baseline_loss)

    function score_candidate(expr, evolved::Bool)
        evaluations >= config.max_evaluations && return nothing
        loss, invalid = score_expression(expr)
        evaluations += 1
        invalid_evaluations += invalid
        serial += 1
        return _ScalarCandidate(expr, loss, serial, evolved)
    end

    initial_limit = min(config.population_size, length(expressions))
    for expr in expressions[1:initial_limit]
        candidate = score_candidate(expr, false)
        candidate === nothing && break
        admit_candidate!(candidate, evaluations)
        birth_clock += 1
        push!(population, _RegularizedMember(candidate, birth_clock))
    end
    length(population) == config.population_size ||
        throw(ArgumentError("initial population did not fill population_size"))

    function refine_candidate!(member::_RegularizedMember, generation::Int)
        config.constant_refinement_iterations == 0 && return member
        generation < config.constant_refinement_start_generation && return member
        remaining = config.max_constant_refinement_evaluations - constant_refinement_evaluations
        remaining <= 0 && return member
        refined_expr, refined_loss, count, invalid = _refine_constants(
            member.candidate.expr, member.candidate.loss, score_expression, config, remaining)
        constant_refinement_evaluations += count
        constant_refinement_invalid_evaluations += invalid
        refined_loss < member.candidate.loss || return member
        member.candidate = _ScalarCandidate(refined_expr, refined_loss,
            member.candidate.serial, member.candidate.evolved)
        admit_candidate!(member.candidate, generation; source=:constant_refinement)
        return member
    end

    function maintain_cycle!()
        _, iteration_length = _regularized_cycle_temperature(offspring_attempts,
            config.population_size, config.tournament_size, ncycles_per_iteration)
        if offspring_attempts % iteration_length == 0
            _regularized_observe_population!(frequency, population)
            for member in population
                simplified = _simplify_scalar(member.candidate.expr)
                if !_same_scalar_expression(simplified, member.candidate.expr)
                    member.candidate = _ScalarCandidate(simplified,
                        member.candidate.loss, member.candidate.serial,
                        member.candidate.evolved)
                    record_archive!(member.candidate)
                end
                config.constant_refinement_probability > 0 &&
                    rand(rng) < config.constant_refinement_probability &&
                    refine_candidate!(member, cld(offspring_attempts, iteration_length))
            end
            migrants = _regularized_pareto_candidates(archive_by_complexity)
            number_to_replace = min(length(population), length(migrants),
                _regularized_poisson(
                    rng, length(population) * Float64(hof_migration_fraction)))
            for _ in 1:number_to_replace
                destination = rand(rng, eachindex(population))
                migrant = rand(rng, migrants)
                serial += 1
                birth_clock += 1
                copied = _ScalarCandidate(migrant.expr, migrant.loss, serial, true)
                population[destination] = _RegularizedMember(copied, birth_clock)
                hof_migrations += 1
            end
        end
        return nothing
    end

    offspring_attempts = 0
    should_stop() = stop_loss !== nothing && best_loss <= stop_loss
    max_offspring = max(0, config.max_evaluations - config.population_size)
    max_variation_attempts = max(1, 10 * max_offspring)
    while evaluations < config.max_evaluations &&
            offspring_attempts < max_variation_attempts && !should_stop()
        offspring_attempts += 1
        candidates = [_regularized_candidate(member) for member in population]
        first_parent = _regularized_tournament(rng, candidates, config, frequency;
            loss_normalization=loss_normalization,
            selection_p=tournament_selection_p,
            use_frequency_in_tournament=use_frequency_in_tournament,
            adaptive_parsimony_scaling=adaptive_parsimony_scaling)
        second_parent = _regularized_tournament(rng, candidates, config, frequency;
            loss_normalization=loss_normalization,
            selection_p=tournament_selection_p,
            use_frequency_in_tournament=use_frequency_in_tournament,
            adaptive_parsimony_scaling=adaptive_parsimony_scaling)
        child_expr, kind = _regularized_child(rng, first_parent, second_parent,
            feature_count, config, offspring_builder)
        if kind === :crossover
            crossover_attempts += 1
        elseif kind === :compose
            composition_attempts += 1
        elseif kind === :injected
            composition_attempts += 1
        else
            mutation_attempts += 1
            mutation_kind_counts[kind] = get(mutation_kind_counts, kind, 0) + 1
        end
        changed_variations += !_same_scalar_expression(
            child_expr, first_parent.expr)
        if kind in (:do_nothing, :simplify)
            serial += 1
            child = _ScalarCandidate(child_expr, first_parent.loss, serial, true)
            accepted_offspring += 1
            index = _regularized_oldest_index(population)
            birth_clock += 1
            population[index] = _RegularizedMember(child, birth_clock)
            oldest_replacements += 1
            admit_candidate!(child, offspring_attempts)
            maintain_cycle!()
            continue
        elseif _same_scalar_expression(child_expr, first_parent.expr)
            skipped_variations += 1
            maintain_cycle!()
            continue
        end
        child = score_candidate(child_expr, true)
        child === nothing && break
        old_complexity = scalar_complexity(first_parent.expr)
        new_complexity = scalar_complexity(child.expr)
        old_frequency, new_frequency = use_frequency ?
            _regularized_frequency_ratio(frequency, old_complexity, new_complexity) : (1.0, 1.0)
        cycle_temperature, _ = _regularized_cycle_temperature(offspring_attempts,
            config.population_size, config.tournament_size, ncycles_per_iteration)
        parent_cost = _regularized_cost(first_parent.loss, old_complexity,
            loss_normalization, config.parsimony_coefficient)
        child_cost = _regularized_cost(child.loss, new_complexity,
            loss_normalization, config.parsimony_coefficient)
        probability = _regularized_acceptance_probability(parent_cost, child_cost;
            annealing=annealing,
            temperature=Float64(temperature) * cycle_temperature,
            alpha=alpha,
            old_frequency=old_frequency, new_frequency=new_frequency)
        if rand(rng) < probability
            accepted_offspring += 1
            index = _regularized_oldest_index(population)
            birth_clock += 1
            population[index] = _RegularizedMember(child, birth_clock)
            oldest_replacements += 1
            admit_candidate!(child, offspring_attempts)
        else
            rejected_offspring += 1
        end
        maintain_cycle!()
    end

    termination = should_stop() ? :solved :
        (evaluations >= config.max_evaluations ? :evaluation_cap : :variation_cap)
    _, iteration_length = _regularized_cycle_temperature(max(1, offspring_attempts),
        config.population_size, config.tournament_size, ncycles_per_iteration)
    generations = offspring_attempts == 0 ? 0 : cld(offspring_attempts, iteration_length)
    return _regularized_report(population, archive_by_complexity, best_expr, best_loss,
        evaluations, invalid_evaluations, constant_refinement_evaluations,
        constant_refinement_invalid_evaluations, generations, termination,
        improvements, mutation_attempts, crossover_attempts, composition_attempts,
        changed_variations, mutation_kind_counts, accepted_offspring, rejected_offspring,
        skipped_variations, oldest_replacements, hof_migrations,
        annealing, use_frequency,
        frequency_window)
end

@testsnippet RegularizedScalarSearchImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
end

@testitem "regularized acceptance follows the non-elitist annealing contract" setup=[RegularizedScalarSearchImports] begin
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
end

@testitem "regularized change detection preserves scalar value equality" setup=[RegularizedScalarSearchImports] begin
    left = scalar_binary(:+, scalar_input(1), scalar_constant(0.0))
    same = scalar_binary(:+, scalar_input(1), scalar_constant(-0.0))
    different = scalar_binary(:+, scalar_input(2), scalar_constant(0.0))
    @test ScalarSearch._same_scalar_expression(left, same)
    @test !ScalarSearch._same_scalar_expression(left, different)
end

@testitem "regularized search consumes the structural budget and replaces oldest members" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_probability=0.0,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = zeros(4)
    child_builder = (rng, first, second, feature_count, config) ->
        scalar_binary(:+, scalar_input(1), scalar_constant(1.0))
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(1111), stop_loss=nothing,
        annealing=false, use_frequency=false,
        initial_population=[scalar_input(1)], offspring_builder=child_builder)
    @test result.evaluations == config.max_evaluations
    @test result.variation.accepted_offspring == config.max_evaluations - config.population_size
    @test result.variation.rejected_offspring == 0
    @test result.variation.oldest_replacements == result.variation.accepted_offspring
    @test result.variation.final_population_size == config.population_size
    @test result.generations == 1
end

@testitem "regularized archive has at most one leader per complexity" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_probability=0.0,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = [1.0, 2.0, 3.0, 4.0]
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(1), stop_loss=nothing,
        ncycles_per_iteration=1, hof_migration_fraction=1.0)
    complexities = [entry.complexity for entry in result.complexity_archive]
    @test length(complexities) == length(unique(complexities))
    @test result.variation.hof_migrations > 0
end

@testitem "regularized mutation weights keep the main SR operations reachable" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=512,
        initial_depth=2,
        tournament_size=3,
        elite_count=2,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_probability=0.0,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = zeros(4)
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(2000), stop_loss=nothing)
    kinds = result.variation.mutation_kinds
    @test kinds.operator > 0
    @test kinds.rotate > 0
    @test kinds.optimize == 0
    @test sum(values(kinds)) == result.variation.mutation_attempts
end

@testitem "regularized mutation traversal is reusable without changing proposals" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=16,
        max_complexity=12,
        initial_depth=2,
        tournament_size=3,
        elite_count=2,
        unary_operators=(:sin, :cos),
        binary_operators=(:+, :-, :*),
    )
    parent = scalar_binary(:+,
        scalar_unary(:sin, scalar_input(1)),
        scalar_binary(:*, scalar_constant(0.25), scalar_input(1)))
    scan = ScalarSearch._regularized_mutation_scan(parent, config)
    @test length(scan.anchors) == scalar_complexity(parent)
    @test length(scan.anchors) == length(ScalarSearch._paths(parent))
    @test scan.has_constant
    @test scan.has_alternate_operator
    @test scan.has_binary
    @test scan.has_children
    @test scan.has_rotation
    for (index, anchor) in enumerate(scan.anchors)
        expected = ScalarSearch._subexpression(parent, anchor.path)
        @test ScalarSearch._same_scalar_expression(anchor.node, expected)
        @test anchor.path == ScalarSearch._paths(parent)[index]
    end

    for (index, kind) in enumerate((:constant, :operator, :swap, :rotate,
            :add, :insert, :delete))
        reference = ScalarSearch._regularized_forced_mutation(
            MersenneTwister(100 + index), parent, 2, config, kind)
        reused = ScalarSearch._regularized_forced_mutation(
            MersenneTwister(100 + index), parent, 2, config, kind; scan=scan)
        @test reference[2] == reused[2] == kind
        @test ScalarSearch._same_scalar_expression(reference[1], reused[1])
    end

    reference_kind = ScalarSearch._regularized_sample_mutation(
        MersenneTwister(1234), parent, config)
    reused_kind = ScalarSearch._regularized_sample_mutation(
        MersenneTwister(1234), parent, config; scan=scan)
    @test reference_kind == reused_kind
end

@testitem "regularized leaf deletion skips and binary insertion stays reachable" setup=[RegularizedScalarSearchImports] begin
    constants = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        max_complexity=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        unary_operators=(),
        binary_operators=(:+,),
        constant_probability=1.0,
    )
    deleted, delete_kind = ScalarSearch._regularized_forced_mutation(
        MersenneTwister(1234), scalar_input(1), 2, constants, :delete)
    @test delete_kind == :delete
    @test ScalarSearch._same_scalar_expression(deleted, scalar_input(1))

    inserted, insert_kind = ScalarSearch._regularized_forced_mutation(
        MersenneTwister(42), scalar_input(1), 2, constants, :insert)
    @test insert_kind == :insert
    @test scalar_complexity(inserted) == 3
end

@testitem "regularized search accepts an injected random-fill initializer" setup=[RegularizedScalarSearchImports] begin
    called = Ref(false)
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=4,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
    )
    initializer = (rng, feature_count, config, provided) -> begin
        called[] = true
        [scalar_unary(:cos, scalar_input(1)) for _ in 1:config.population_size]
    end
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = cos.(vec(X))
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(42), stop_loss=nothing,
        initializer=initializer)
    @test called[]
    @test result.evaluations == config.population_size
    @test any(entry.complexity == 2 for entry in result.complexity_archive)
end

@testitem "regularized failed variations do not spend structural evaluations" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_probability=0.0,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = zeros(4)
    unchanged = (rng, first, second, feature_count, config) -> first
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(2000), stop_loss=nothing,
        annealing=false, use_frequency=false,
        initial_population=[scalar_input(1)], offspring_builder=unchanged)
    @test result.evaluations == config.population_size
    @test result.termination == :variation_cap
    @test result.variation.skipped_variations > 0
    @test result.variation.oldest_replacements == 0
end

@testitem "regularized rejected offspring stay out of the archive" setup=[RegularizedScalarSearchImports] begin
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        mutation_probability=1.0,
        crossover_probability=0.0,
        constant_refinement_probability=0.0,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = vec(X)
    initializer = (rng, feature_count, config, provided) ->
        [scalar_input(1) for _ in 1:config.population_size]
    worse = (rng, first, second, feature_count, config) ->
        scalar_binary(:+, first, scalar_constant(1.0))
    result = ScalarSearch._search_scalar_regularized(X, y;
        config=config, rng=MersenneTwister(1111), stop_loss=nothing,
        annealing=true, temperature=eps(Float64), alpha=1.0,
        use_frequency=false, initializer=initializer,
        offspring_builder=worse)
    @test result.variation.rejected_offspring ==
        config.max_evaluations - config.population_size
    @test all(entry.complexity == 1 for entry in result.complexity_archive)
    @test scalar_complexity(result.best) == 1
end
