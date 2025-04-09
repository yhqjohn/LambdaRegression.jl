"""Count scalar expression nodes; this is the scalar structural complexity."""
scalar_complexity(::_ScalarInput) = 1
scalar_complexity(::_ScalarConstant) = 1
scalar_complexity(expr::_ScalarUnary) = 1 + scalar_complexity(expr.arg)
scalar_complexity(expr::_ScalarBinary) =
    1 + scalar_complexity(expr.left) + scalar_complexity(expr.right)

struct _ScalarSignatureToken
    kind::Symbol
    payload::Int
    value::Float64
end

struct _ScalarSignature
    tokens::Tuple{Vararg{_ScalarSignatureToken}}
end

Base.:(==)(left::_ScalarSignature, right::_ScalarSignature) =
    left.tokens == right.tokens
Base.isequal(left::_ScalarSignature, right::_ScalarSignature) =
    isequal(left.tokens, right.tokens)
Base.hash(signature::_ScalarSignature, seed::UInt) = hash(signature.tokens, seed)

function _signature_tokens!(tokens, expr::_ScalarInput, include_constant_values)
    push!(tokens, _ScalarSignatureToken(:input, expr.index, 0.0))
    return tokens
end

function _signature_tokens!(tokens, expr::_ScalarConstant, include_constant_values)
    value = include_constant_values ? expr.value : 0.0
    push!(tokens, _ScalarSignatureToken(:constant, 0, value))
    return tokens
end


function _signature_tokens!(tokens, expr::_ScalarUnary, include_constant_values)
    push!(tokens, _ScalarSignatureToken(expr.op, 1, 0.0))
    _signature_tokens!(tokens, expr.arg, include_constant_values)
    return tokens
end

function _signature_tokens!(tokens, expr::_ScalarBinary, include_constant_values)
    push!(tokens, _ScalarSignatureToken(expr.op, 2, 0.0))
    _signature_tokens!(tokens, expr.left, include_constant_values)
    _signature_tokens!(tokens, expr.right, include_constant_values)
    return tokens
end

function _scalar_signature(expr::ScalarExpr, include_constant_values::Bool)
    tokens = _ScalarSignatureToken[]
    _signature_tokens!(tokens, expr, include_constant_values)
    return _ScalarSignature(Tuple(tokens))
end

_structure_signature(expr::ScalarExpr) = _scalar_signature(expr, false)
_expression_signature(expr::ScalarExpr) = _scalar_signature(expr, true)

function _collect_structure_counts!(counts, expr::ScalarExpr)
    signature = _structure_signature(expr)
    counts[signature] = get(counts, signature, 0) + 1
    if expr isa _ScalarUnary
        _collect_structure_counts!(counts, expr.arg)
    elseif expr isa _ScalarBinary
        _collect_structure_counts!(counts, expr.left)
        _collect_structure_counts!(counts, expr.right)
    end
    return counts
end

"""Weighted subtree-overlap similarity in `[0, 1]`, ignoring constant values."""
function scalar_structure_similarity(left::ScalarExpr, right::ScalarExpr)
    left_counts = _collect_structure_counts!(Dict{_ScalarSignature,Int}(), left)
    right_counts = _collect_structure_counts!(Dict{_ScalarSignature,Int}(), right)
    keys_union = union(keys(left_counts), keys(right_counts))
    intersection_weight = sum(min(get(left_counts, key, 0), get(right_counts, key, 0))
        for key in keys_union)
    union_weight = sum(max(get(left_counts, key, 0), get(right_counts, key, 0))
        for key in keys_union)
    return union_weight == 0 ? 1.0 : intersection_weight / union_weight
end

"""
    ScalarSearchConfig(; ...)

Configuration for bounded real scalar population search. The search anchors on
the supplied feature count, uses caller-owned randomness, and stops before
scoring more than `max_evaluations` candidates. Expected domain failures count
as invalid evaluations. Constant-polishing objective calls have an independent
hard budget and are reported separately; BFGS uses Optim's BackTracking line
search with central finite differences, with Nelder–Mead retained as an
explicit option. `parsimony_coefficient`
affects tournament and elite ordering without changing the reported raw MSE,
while complexity-balanced tournament sampling controls which size classes get
to compete. The loss-complexity archive is observational and supports held-out
model selection; it is not injected into breeding by this baseline.
`evolution_strategy=:generational` selects the established bounded generational
search. `:regularized` selects the steady-state, age-replacement SR workflow;
it uses the same feature matrix, operator world, budgets, and expected-domain
failure classification, but interprets a reported generation as one completed
annealing/search cycle rather than one offspring proposal. Unsupported strategy
names and malformed operator worlds are rejected before search starts.
For regularized search, `constant_refinement_timing` chooses between the
existing probabilistic cycle-end polishing and probabilistic pre-admission
polishing of scored offspring. `max_total_evaluations`, when set, caps the sum
of structural and constant-refinement objective calls; multi-island sessions
split that cap across islands. Per-island baseline callbacks are reported
separately and are excluded from the cap.
Malformed configuration is rejected immediately.
"""
struct ScalarSearchConfig
    population_size::Int
    max_evaluations::Int
    max_complexity::Int
    initial_depth::Int
    tournament_size::Int
    complexity_balanced_tournament::Bool
    elite_count::Int
    unary_operators::Tuple{Vararg{Symbol}}
    binary_operators::Tuple{Vararg{Symbol}}
    constant_probability::Float64
    mutation_probability::Float64
    crossover_probability::Float64
    random_restart_probability::Float64
    constant_scale::Float64
    constant_refinement_iterations::Int
    constant_refinement_elites::Int
    constant_refinement_interval::Int
    constant_refinement_start_generation::Int
    constant_refinement_probability::Float64
    constant_refinement_timing::Symbol
    max_constant_refinement_evaluations::Int
    max_total_evaluations::Union{Nothing,Int}
    constant_refinement_optimizer::Symbol
    parsimony_coefficient::Float64
    initial_pool_multiplier::Int
    initial_affine_basis::Bool
    evolution_strategy::Symbol
end

function ScalarSearchConfig(;
        population_size::Int=64,
        max_evaluations::Int=50_000,
        max_complexity::Int=24,
        initial_depth::Int=3,
        tournament_size::Int=5,
        complexity_balanced_tournament::Bool=true,
        elite_count::Int=4,
        unary_operators=(:sin, :sqrt, :square),
        binary_operators=(:+, :-, :*, :/),
        constant_probability::Real=0.2,
        mutation_probability::Real=0.9,
        crossover_probability::Real=0.05,
        random_restart_probability::Real=0.0,
        constant_scale::Real=2.0,
        constant_refinement_iterations::Int=8,
        constant_refinement_elites::Int=2,
        constant_refinement_interval::Int=25,
        constant_refinement_start_generation::Int=25,
        constant_refinement_probability::Real=0.0,
        constant_refinement_timing::Symbol=:cycle,
        max_constant_refinement_evaluations::Int=max(0, max_evaluations ÷ 5),
        max_total_evaluations::Union{Nothing,Int}=nothing,
        constant_refinement_optimizer::Symbol=:bfgs,
        parsimony_coefficient::Real=1.0e-4,
        initial_pool_multiplier::Int=4,
        initial_affine_basis::Bool=false,
        evolution_strategy::Symbol=:generational)
    population_size >= 2 || throw(ArgumentError("population_size must be at least 2"))
    max_evaluations >= population_size ||
        throw(ArgumentError("max_evaluations must cover the initial population"))
    max_complexity >= 1 || throw(ArgumentError("max_complexity must be positive"))
    initial_depth >= 0 || throw(ArgumentError("initial_depth cannot be negative"))
    1 <= tournament_size <= population_size ||
        throw(ArgumentError("tournament_size must be within the population"))
    1 <= elite_count < population_size ||
        throw(ArgumentError("elite_count must be positive and below population_size"))
    unary = Tuple(Symbol(op) for op in unary_operators)
    binary = Tuple(Symbol(op) for op in binary_operators)
    all(haskey(_UNARY_AST, op) for op in unary) ||
        throw(ArgumentError("unary_operators contains an unsupported operator"))
    all(haskey(_BINARY_AST, op) for op in binary) ||
        throw(ArgumentError("binary_operators contains an unsupported operator"))
    isempty(unary) && isempty(binary) &&
        throw(ArgumentError("at least one scalar operator is required"))
    0 <= constant_probability <= 1 ||
        throw(ArgumentError("constant_probability must be in [0, 1]"))
    0 <= mutation_probability <= 1 ||
        throw(ArgumentError("mutation_probability must be in [0, 1]"))
    0 <= crossover_probability <= 1 ||
        throw(ArgumentError("crossover_probability must be in [0, 1]"))
    0 <= random_restart_probability <= 1 || throw(ArgumentError(
        "random_restart_probability must be in [0, 1]"))
    mutation_probability + crossover_probability <= 1 ||
        throw(ArgumentError("mutation_probability + crossover_probability must be <= 1"))
    constant_scale > 0 || throw(ArgumentError("constant_scale must be positive"))
    constant_refinement_iterations >= 0 ||
        throw(ArgumentError("constant_refinement_iterations cannot be negative"))
    0 <= constant_refinement_elites <= population_size ||
        throw(ArgumentError("constant_refinement_elites must be within the population"))
    constant_refinement_interval >= 1 ||
        throw(ArgumentError("constant_refinement_interval must be positive"))
    constant_refinement_start_generation >= 1 || throw(ArgumentError(
        "constant_refinement_start_generation must be positive"))
    0 <= constant_refinement_probability <= 1 ||
        throw(ArgumentError("constant_refinement_probability must be in [0, 1]"))
    max_constant_refinement_evaluations >= 0 ||
        throw(ArgumentError("max_constant_refinement_evaluations cannot be negative"))
    constant_refinement_timing in (:cycle, :offspring) ||
        throw(ArgumentError(
            "constant_refinement_timing must be :cycle or :offspring"))
    constant_refinement_optimizer in (:bfgs, :nelder_mead) ||
        throw(ArgumentError(
            "constant_refinement_optimizer must be :bfgs or :nelder_mead"))
    parsimony_coefficient >= 0 ||
        throw(ArgumentError("parsimony_coefficient cannot be negative"))
    initial_pool_multiplier >= 1 ||
        throw(ArgumentError("initial_pool_multiplier must be positive"))
    evolution_strategy in (:generational, :regularized) || throw(ArgumentError(
        "evolution_strategy must be :generational or :regularized"))
    if evolution_strategy === :generational
        constant_refinement_timing === :cycle || throw(ArgumentError(
            "constant_refinement_timing=:offspring requires evolution_strategy=:regularized"))
        max_total_evaluations === nothing || throw(ArgumentError(
            "max_total_evaluations requires evolution_strategy=:regularized"))
    end
    max_total_evaluations === nothing || max_total_evaluations >= population_size ||
        throw(ArgumentError(
            "max_total_evaluations must cover the initial population"))
    return ScalarSearchConfig(
        population_size,
        max_evaluations,
        max_complexity,
        initial_depth,
        tournament_size,
        complexity_balanced_tournament,
        elite_count,
        unary,
        binary,
        Float64(constant_probability),
        Float64(mutation_probability),
        Float64(crossover_probability),
        Float64(random_restart_probability),
        Float64(constant_scale),
        constant_refinement_iterations,
        constant_refinement_elites,
        constant_refinement_interval,
        constant_refinement_start_generation,
        Float64(constant_refinement_probability),
        constant_refinement_timing,
        max_constant_refinement_evaluations,
        max_total_evaluations,
        constant_refinement_optimizer,
        Float64(parsimony_coefficient),
        initial_pool_multiplier,
        initial_affine_basis,
        evolution_strategy,
    )
end

"""Result and accounting for one bounded scalar population search."""
struct ScalarSearchResult
    best::ScalarExpr
    best_loss::Float64
    evaluations::Int
    invalid_evaluations::Int
    constant_refinement_evaluations::Int
    constant_refinement_invalid_evaluations::Int
    generations::Int
    termination::Symbol
    improvements::Vector{NamedTuple}
    leaders::Vector{NamedTuple}
    complexity_archive::Vector{NamedTuple}
    variation::NamedTuple
end

struct _ScalarCandidate
    expr::ScalarExpr
    loss::Float64
    serial::Int
    evolved::Bool
end

_candidate_key(candidate::_ScalarCandidate) =
    (candidate.loss, scalar_complexity(candidate.expr), candidate.serial)

_selection_key(candidate::_ScalarCandidate, config::ScalarSearchConfig) = (
    candidate.loss + config.parsimony_coefficient * scalar_complexity(candidate.expr),
    candidate.loss,
    scalar_complexity(candidate.expr),
    candidate.serial,
)

function _random_leaf(rng::AbstractRNG, feature_count::Int, config::ScalarSearchConfig)
    feature_count == 0 &&
        return scalar_constant(randn(rng) * config.constant_scale)
    if rand(rng) < config.constant_probability
        return scalar_constant(randn(rng) * config.constant_scale)
    end
    return scalar_input(rand(rng, 1:feature_count))
end

function _random_expression(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig, depth::Int)
    depth <= 0 && return _random_leaf(rng, feature_count, config)
    leaf_probability = 1 / (depth + 2)
    rand(rng) < leaf_probability && return _random_leaf(rng, feature_count, config)
    choose_unary = !isempty(config.unary_operators) &&
        (isempty(config.binary_operators) || rand(rng, Bool))
    if choose_unary
        op = rand(rng, config.unary_operators)
        arg = _random_expression(rng, feature_count, config, depth - 1)
        return scalar_unary(op, arg)
    end
    op = rand(rng, config.binary_operators)
    left = _random_expression(rng, feature_count, config, depth - 1)
    right = _random_expression(rng, feature_count, config, depth - 1)
    return scalar_binary(op, left, right)
end

function _paths!(paths::Vector{Tuple{Vararg{Int}}}, expr::ScalarExpr, path=())
    push!(paths, path)
    if expr isa _ScalarUnary
        _paths!(paths, expr.arg, (path..., 1))
    elseif expr isa _ScalarBinary
        _paths!(paths, expr.left, (path..., 1))
        _paths!(paths, expr.right, (path..., 2))
    end
    return paths
end

_paths(expr::ScalarExpr) = _paths!(Tuple{Vararg{Int}}[], expr)

function _subexpression(expr::ScalarExpr, path, offset=1)
    offset > length(path) && return expr
    child = path[offset]
    expr isa _ScalarUnary && child == 1 &&
        return _subexpression(expr.arg, path, offset + 1)
    if expr isa _ScalarBinary
        child == 1 && return _subexpression(expr.left, path, offset + 1)
        child == 2 && return _subexpression(expr.right, path, offset + 1)
    end
    throw(ArgumentError("invalid scalar expression path"))
end

function _replace_subexpression(expr::ScalarExpr, path, replacement::ScalarExpr, offset=1)
    offset > length(path) && return replacement
    child = path[offset]
    if expr isa _ScalarUnary && child == 1
        return scalar_unary(expr.op,
            _replace_subexpression(expr.arg, path, replacement, offset + 1))
    elseif expr isa _ScalarBinary
        child == 1 && return scalar_binary(expr.op,
            _replace_subexpression(expr.left, path, replacement, offset + 1), expr.right)
        child == 2 && return scalar_binary(expr.op, expr.left,
            _replace_subexpression(expr.right, path, replacement, offset + 1))
    end
    throw(ArgumentError("invalid scalar expression path"))
end

function _fold_constant_unary(op::Symbol, value::Float64)
    result = try
        op === :sin ? sin(value) :
        op === :cos ? cos(value) :
        op === :sqrt ? sqrt(value) :
        value * value
    catch error
        error isa DomainError || rethrow()
        return nothing
    end
    return isfinite(result) ? result : nothing
end

function _fold_constant_binary(op::Symbol, left::Float64, right::Float64)
    result = op === :+ ? left + right :
        op === :- ? left - right :
        op === :* ? left * right : left / right
    return isfinite(result) ? result : nothing
end

_fold_constants(expr::_ScalarInput) = expr
_fold_constants(expr::_ScalarConstant) = expr

function _fold_constants(expr::_ScalarUnary)
    arg = _fold_constants(expr.arg)
    if arg isa _ScalarConstant
        value = _fold_constant_unary(expr.op, arg.value)
        value === nothing || return scalar_constant(value)
    end
    return scalar_unary(expr.op, arg)
end

function _fold_constants(expr::_ScalarBinary)
    left = _fold_constants(expr.left)
    right = _fold_constants(expr.right)
    if left isa _ScalarConstant && right isa _ScalarConstant
        value = _fold_constant_binary(expr.op, left.value, right.value)
        value === nothing || return scalar_constant(value)
    end
    return scalar_binary(expr.op, left, right)
end

_simplify_scalar(expr::_ScalarInput) = expr
_simplify_scalar(expr::_ScalarConstant) = expr

function _simplify_scalar(expr::_ScalarUnary)
    return _fold_constants(scalar_unary(expr.op, _simplify_scalar(expr.arg)))
end

function _simplify_scalar(expr::_ScalarBinary)
    left = _simplify_scalar(expr.left)
    right = _simplify_scalar(expr.right)
    folded = _fold_constants(scalar_binary(expr.op, left, right))
    folded isa _ScalarBinary || return folded
    left, right = folded.left, folded.right
    if expr.op === :+
        left isa _ScalarConstant && iszero(left.value) && return right
        right isa _ScalarConstant && iszero(right.value) && return left
    elseif expr.op === :-
        right isa _ScalarConstant && iszero(right.value) && return left
        if left isa _ScalarInput && right isa _ScalarBinary && right.op === :+
            if right.left isa _ScalarConstant && right.right isa _ScalarInput &&
                    left.index == right.right.index
                return scalar_constant(-right.left.value)
            elseif right.right isa _ScalarConstant && right.left isa _ScalarInput &&
                    left.index == right.left.index
                return scalar_constant(-right.right.value)
            end
        end
    elseif expr.op === :*
        left isa _ScalarConstant && isone(left.value) && return right
        right isa _ScalarConstant && isone(right.value) && return left
    elseif expr.op === :/
        right isa _ScalarConstant && isone(right.value) && return left
    end
    return folded
end

function _bounded_candidate(candidate::ScalarExpr, parent::ScalarExpr, config)
    folded = _fold_constants(candidate)
    return scalar_complexity(folded) <= config.max_complexity ? folded : parent
end

_has_children(expr::ScalarExpr) = expr isa Union{_ScalarUnary,_ScalarBinary}

function _is_rotatable(expr::ScalarExpr)
    expr isa _ScalarUnary && return _has_children(expr.arg)
    expr isa _ScalarBinary || return false
    return _has_children(expr.left) || _has_children(expr.right)
end

function _rotate_expression(rng::AbstractRNG, root::_ScalarUnary)
    child = root.arg
    if child isa _ScalarUnary
        return scalar_unary(child.op, scalar_unary(root.op, child.arg))
    end
    child isa _ScalarBinary || return root
    return scalar_binary(
        child.op, child.left, scalar_unary(root.op, child.right))
end

function _rotate_expression(rng::AbstractRNG, root::_ScalarBinary)
    right_rotation = _has_children(root.left) &&
        (!_has_children(root.right) || rand(rng, Bool))
    child = right_rotation ? root.left : root.right
    if right_rotation && child isa _ScalarUnary
        return scalar_unary(
            child.op, scalar_binary(root.op, child.arg, root.right))
    elseif right_rotation && child isa _ScalarBinary
        return scalar_binary(child.op, child.left,
            scalar_binary(root.op, child.right, root.right))
    elseif child isa _ScalarUnary
        return scalar_unary(
            child.op, scalar_binary(root.op, root.left, child.arg))
    end
    child isa _ScalarBinary || return root
    return scalar_binary(child.op,
        scalar_binary(root.op, root.left, child.left), child.right)
end

"""
Choose one eligible scalar mutation anchored at a path in `parent`.

Eligibility is derived from the current node kinds and configured operator
world; no evaluation context is required. The chooser always returns an
implemented mutation kind, while later size bounding may conservatively return
the unchanged parent when a generated replacement exceeds `max_complexity`.
"""
function _mutation_kind(rng::AbstractRNG, parent::ScalarExpr, paths, config)
    config.random_restart_probability > 0 &&
        rand(rng) < config.random_restart_probability && return :restart
    nodes = [_subexpression(parent, path) for path in paths]
    has_branch = any(node -> node isa Union{_ScalarUnary,_ScalarBinary}, nodes)
    has_binary = any(node -> node isa _ScalarBinary, nodes)
    has_rotation = any(_is_rotatable, nodes)
    has_constant = any(node -> node isa _ScalarConstant, nodes)
    has_alternate_operator = any(nodes) do node
        node isa _ScalarUnary && any(!=(node.op), config.unary_operators) ||
            node isa _ScalarBinary && any(!=(node.op), config.binary_operators)
    end

    # Repetition encodes a deliberately small conventional GP mixture while
    # keeping the physical mutation sampler private and replaceable.
    choices = Symbol[:subtree]
    !isempty(config.unary_operators) && append!(choices, (:insert, :insert))
    !isempty(config.binary_operators) && append!(choices, (:add, :add))
    !isempty(config.binary_operators) && append!(choices, (:graft, :graft))
    has_rotation && append!(choices, (:rotate, :rotate, :rotate, :rotate))
    config.constant_probability > 0 && :* in config.binary_operators &&
        push!(choices, :scale)
    config.constant_probability > 0 && :+ in config.binary_operators &&
        push!(choices, :shift)
    has_branch && append!(choices, (:delete, :delete))
    has_alternate_operator && append!(choices, (:operator, :operator))
    has_binary && push!(choices, :swap)
    has_constant && push!(choices, :constant)
    return rand(rng, choices)
end

function _mutate_expression(rng::AbstractRNG, parent::ScalarExpr, feature_count, config)
    paths = _paths(parent)
    kind = _mutation_kind(rng, parent, paths, config)
    if kind === :restart
        candidate = _random_expression(rng, feature_count, config, config.initial_depth)
        return (_bounded_candidate(candidate, parent, config), kind)
    end
    compatible_paths = if kind === :delete
        filter(path -> _subexpression(parent, path) isa Union{_ScalarUnary,_ScalarBinary}, paths)
    elseif kind === :operator
        filter(paths) do path
            node = _subexpression(parent, path)
            node isa _ScalarUnary && any(!=(node.op), config.unary_operators) ||
                node isa _ScalarBinary && any(!=(node.op), config.binary_operators)
        end
    elseif kind === :swap
        filter(path -> _subexpression(parent, path) isa _ScalarBinary, paths)
    elseif kind === :constant
        filter(path -> _subexpression(parent, path) isa _ScalarConstant, paths)
    elseif kind === :rotate
        filter(path -> _is_rotatable(_subexpression(parent, path)), paths)
    else
        paths
    end
    path = rand(rng, compatible_paths)
    subtree = _subexpression(parent, path)
    replacement = if kind === :subtree
        _random_expression(rng, feature_count, config, min(config.initial_depth, 2))
    elseif kind === :insert
        scalar_unary(rand(rng, config.unary_operators), subtree)
    elseif kind === :add
        leaf = _random_leaf(rng, feature_count, config)
        op = rand(rng, config.binary_operators)
        rand(rng, Bool) ? scalar_binary(op, subtree, leaf) : scalar_binary(op, leaf, subtree)
    elseif kind === :graft
        branch = _random_expression(
            rng, feature_count, config, min(config.initial_depth, 2))
        op = rand(rng, config.binary_operators)
        rand(rng, Bool) ? scalar_binary(op, subtree, branch) :
            scalar_binary(op, branch, subtree)
    elseif kind === :rotate
        _rotate_expression(rng, subtree)
    elseif kind === :scale
        coefficient = rand(rng, Bool) ?
            1.0 + 0.25 * randn(rng) : 0.5 * randn(rng)
        scalar_binary(:*, scalar_constant(coefficient), subtree)
    elseif kind === :shift
        offset = 0.25 * config.constant_scale * randn(rng)
        scalar_binary(:+, subtree, scalar_constant(offset))
    elseif kind === :delete
        subtree isa _ScalarUnary ? subtree.arg :
            (rand(rng, Bool) ? subtree.left : subtree.right)
    elseif kind === :operator
        if subtree isa _ScalarUnary
            operators = filter(!=(subtree.op), collect(config.unary_operators))
            scalar_unary(rand(rng, operators), subtree.arg)
        else
            operators = filter(!=(subtree.op), collect(config.binary_operators))
            scalar_binary(rand(rng, operators), subtree.left, subtree.right)
        end
    elseif kind === :swap
        scalar_binary(subtree.op, subtree.right, subtree.left)
    else
        scalar_constant(subtree.value + randn(rng) * 0.25 * config.constant_scale)
    end
    candidate = _replace_subexpression(parent, path, replacement)
    return (_bounded_candidate(candidate, parent, config), kind)
end

function _crossover_expression(rng::AbstractRNG, parent::ScalarExpr,
        donor::ScalarExpr, config)
    target_path = rand(rng, _paths(parent))
    donor_path = rand(rng, _paths(donor))
    child = _replace_subexpression(parent, target_path, _subexpression(donor, donor_path))
    return _bounded_candidate(child, parent, config)
end

function _compose_expression(rng::AbstractRNG, left::ScalarExpr,
        right::ScalarExpr, config)
    isempty(config.binary_operators) && return left
    child = scalar_binary(rand(rng, config.binary_operators), left, right)
    return _bounded_candidate(child, left, config)
end

function _initial_expressions(rng, feature_count, config, initial_population)
    target_count = min(
        config.max_evaluations,
        config.initial_pool_multiplier * config.population_size,
    )
    expressions = ScalarExpr[]
    primitive_basis = ScalarExpr[]
    if initial_population !== nothing
        append!(expressions, initial_population)
    end
    for feature in 1:feature_count
        length(expressions) >= target_count && break
        input = scalar_input(feature)
        push!(expressions, input)
        push!(primitive_basis, input)
    end
    if config.constant_probability > 0
        for value in (0.0, 1.0, -1.0)
            length(expressions) >= target_count && break
            push!(expressions, scalar_constant(value))
        end
    end
    for feature in 1:feature_count, op in config.unary_operators
        length(expressions) >= target_count && break
        unary = scalar_unary(op, scalar_input(feature))
        push!(expressions, unary)
        push!(primitive_basis, unary)
    end
    if :/ in config.binary_operators
        for feature in 1:feature_count
            length(expressions) >= target_count && break
            reciprocal = scalar_binary(
                :/, scalar_constant(1.0), scalar_input(feature))
            push!(expressions, reciprocal)
            push!(primitive_basis, reciprocal)
        end
    end
    structured_limit = max(length(expressions), fld(3 * target_count, 4))
    if config.initial_affine_basis && :+ in config.binary_operators &&
            :* in config.binary_operators
        for left in eachindex(primitive_basis)
            for right in (left + 1):length(primitive_basis)
                length(expressions) >= structured_limit && break
                affine = scalar_binary(:+,
                    scalar_binary(:*, scalar_constant(1.0), primitive_basis[left]),
                    scalar_binary(:+,
                        scalar_binary(:*, scalar_constant(1.0), primitive_basis[right]),
                        scalar_constant(0.0)))
                push!(expressions, affine)
            end
            length(expressions) >= structured_limit && break
        end
    end
    if length(expressions) < structured_limit && !isempty(config.binary_operators)
        for right in eachindex(primitive_basis), left in eachindex(primitive_basis),
                op in config.binary_operators
            length(expressions) >= structured_limit && break
            candidate = scalar_binary(op, primitive_basis[left], primitive_basis[right])
            if scalar_complexity(candidate) <= config.max_complexity
                push!(expressions, candidate)
            end
        end
    end
    while length(expressions) < target_count
        candidate = _random_expression(rng, feature_count, config, config.initial_depth)
        scalar_complexity(candidate) <= config.max_complexity && push!(expressions, candidate)
    end
    return _fold_constants.(expressions[1:target_count])
end

function _score_expression!(expr, X, y, output, workspace)
    prediction = evaluate_scalar!(output, prepare_scalar(expr), X, workspace)
    prediction isa Err && return (Inf, true)
    total = 0.0
    @inbounds @simd for index in eachindex(y)
        residual = output[index] - y[index]
        total += residual * residual
    end
    loss = total / length(y)
    return isfinite(loss) ? (loss, false) : (Inf, true)
end

function _constant_paths!(paths::Vector{Tuple{Vararg{Int}}}, expr::ScalarExpr, path=())
    if expr isa _ScalarConstant
        push!(paths, path)
    elseif expr isa _ScalarUnary
        _constant_paths!(paths, expr.arg, (path..., 1))
    elseif expr isa _ScalarBinary
        _constant_paths!(paths, expr.left, (path..., 1))
        _constant_paths!(paths, expr.right, (path..., 2))
    end
    return paths
end

_constant_paths(expr::ScalarExpr) = _constant_paths!(Tuple{Vararg{Int}}[], expr)

# Resource exhaustion is not an objective value; only our optimizer boundary
# catches this signal. Callback and numerical-programming errors still propagate.
struct _ConstantRefinementBudgetExhausted <: Exception end

function _refine_constants(expr::ScalarExpr, loss::Float64, score_expression,
        config::ScalarSearchConfig, max_evaluations::Int)
    paths = _constant_paths(expr)
    isempty(paths) && return (expr, loss, 0, 0)
    config.constant_refinement_iterations == 0 && return (expr, loss, 0, 0)
    max_evaluations <= 0 && return (expr, loss, 0, 0)
    initial = Float64[]
    for path in paths
        constant = _subexpression(expr, path)
        constant isa _ScalarConstant || error("constant path changed during refinement")
        push!(initial, constant.value)
    end
    evaluations = Ref(0)
    invalid_evaluations = Ref(0)
    best_expr = Ref(expr)
    best_loss = Ref(loss)
    best_trial_valid = Ref(false)
    function rebuild(values)
        candidate = expr
        for (path, value) in zip(paths, values)
            isfinite(value) || return nothing
            candidate = _replace_subexpression(candidate, path, scalar_constant(value))
        end
        return candidate
    end
    function objective(values)
        evaluations[] >= max_evaluations &&
            throw(_ConstantRefinementBudgetExhausted())
        candidate = rebuild(values)
        candidate === nothing && return Inf
        candidate_loss, invalid = score_expression(candidate)
        evaluations[] += 1
        invalid_evaluations[] += invalid
        if !invalid && isfinite(candidate_loss) &&
                (!best_trial_valid[] || candidate_loss < best_loss[])
            best_expr[] = candidate
            best_loss[] = candidate_loss
            best_trial_valid[] = true
        end
        return !invalid && isfinite(candidate_loss) ? candidate_loss : Inf
    end
    options = Optim.Options(
        iterations=config.constant_refinement_iterations,
        show_trace=false,
        store_trace=false,
    )
    function optimize_from(start)
        if config.constant_refinement_optimizer === :bfgs
            method = Optim.BFGS(linesearch=Optim.LineSearches.BackTracking())
            Optim.optimize(
                objective,
                start,
                method,
                options;
                autodiff=Optim.ADTypes.AutoFiniteDiff(fdtype=Val(:central)),
            )
        else
            Optim.optimize(objective, start, Optim.NelderMead(), options)
        end
    end
    try
        optimize_from(initial)
    catch exception
        exception isa _ConstantRefinementBudgetExhausted || rethrow()
    end
    if best_trial_valid[] && best_loss[] < loss
        return (best_expr[], best_loss[], evaluations[], invalid_evaluations[])
    end
    return (expr, loss, evaluations[], invalid_evaluations[])
end

function _complexity_buckets(population)
    by_complexity = Dict{Int,Vector{_ScalarCandidate}}()
    for candidate in population
        push!(get!(Vector{_ScalarCandidate}, by_complexity,
            scalar_complexity(candidate.expr)), candidate)
    end
    return [by_complexity[complexity] for complexity in sort!(collect(keys(by_complexity)))]
end

function _tournament(rng, population, size, config, complexity_buckets)
    sample_candidate() = config.complexity_balanced_tournament ?
        rand(rng, rand(rng, complexity_buckets)) :
        population[rand(rng, eachindex(population))]
    best = sample_candidate()
    for _ in 2:size
        candidate = sample_candidate()
        _selection_key(candidate, config) < _selection_key(best, config) &&
            (best = candidate)
    end
    return best
end

"""
    search_scalar(X, y; config=ScalarSearchConfig(), rng=default_rng(),
                  initial_population=nothing, objective=nothing,
                  stop_loss=objective === nothing ? 1e-12 : nothing) -> ScalarSearchResult

Run a bounded multi-generation scalar genetic search on `features × samples`
data. A zero-row feature matrix defines a constant-only search domain. Initial
populations may be supplied for calibration or scaffold reuse; callers
may leave them unset. Candidate evaluation attempts,
including domain failures, consume the evaluation budget. A contextual
`objective(expr)` may replace row-wise MSE for fixed-scaffold/mystery tasks; it
must return a finite real loss or `Err`, and is also used by constant refinement.
The default objective stops at loss `1e-12`; a custom objective has no default
stop threshold because its valid losses may be negative. Pass `stop_loss=nothing`
when the caller must spend the declared structural budget before selecting from
the loss-complexity archive.
"""
function search_scalar(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        config::ScalarSearchConfig=ScalarSearchConfig(),
        rng::AbstractRNG=default_rng(),
        initial_population=nothing,
        objective=nothing,
        stop_loss::Union{Nothing,Real}=
            objective === nothing ? 1.0e-12 : nothing)
    feature_count = size(X, 1)
    size(X, 2) == length(y) || throw(DimensionMismatch("target length must match samples"))
    isempty(y) && throw(ArgumentError("scalar search requires at least one sample"))
    all(isfinite, X) || throw(ArgumentError("scalar search inputs must be finite"))
    all(isfinite, y) || throw(ArgumentError("scalar search targets must be finite"))
    stop_loss === nothing || isfinite(stop_loss) ||
        throw(ArgumentError("stop_loss must be nothing or finite"))
    objective === nothing || applicable(objective, scalar_input(1)) ||
        throw(ArgumentError("objective must accept one ScalarExpr"))
    if initial_population !== nothing
        all(expr -> expr isa ScalarExpr, initial_population) ||
            throw(ArgumentError("initial_population must contain ScalarExpr values"))
        all(expr -> scalar_complexity(expr) <= config.max_complexity, initial_population) ||
            throw(ArgumentError("initial_population exceeds max_complexity"))
        all(expr -> _required_features(expr) <= feature_count,
            initial_population) || throw(ArgumentError(
                "initial_population references a feature outside the input matrix"))
    end

    if config.evolution_strategy === :regularized
        return _search_scalar_regularized_node(
            X,
            y;
            config,
            rng,
            initial_population,
            objective,
            stop_loss,
        )
    end

    expressions = _initial_expressions(rng, feature_count, config, initial_population)
    output = Vector{Float64}(undef, length(y))
    workspace = ScalarWorkspace()
    population = _ScalarCandidate[]
    evaluations = 0
    invalid_evaluations = 0
    constant_refinement_evaluations = 0
    constant_refinement_invalid_evaluations = 0
    mutation_attempts = 0
    crossover_attempts = 0
    composition_attempts = 0
    mutation_kind_counts = Dict(
        kind => 0 for kind in
            (:constant, :operator, :swap, :insert, :add, :graft, :rotate, :scale,
                :shift, :restart, :delete, :subtree)
    )
    changed_variations = 0
    serial = 0
    improvements = NamedTuple[]
    archive_by_complexity = Dict{Int,_ScalarCandidate}()
    best_loss = Inf
    best_expr = first(expressions)

    function record_archive!(candidate)
        complexity = scalar_complexity(candidate.expr)
        incumbent = get(archive_by_complexity, complexity, nothing)
        if incumbent === nothing || _candidate_key(candidate) < _candidate_key(incumbent)
            archive_by_complexity[complexity] = candidate
        end
        return candidate
    end

    function score_expression(expr)
        if objective === nothing
            return _score_expression!(expr, X, y, output, workspace)
        end
        prepared = prepare_scalar(expr)
        value = applicable(objective, expr, prepared) ?
            objective(expr, prepared) : objective(expr)
        value isa Err && return (Inf, true)
        value isa Real || throw(ArgumentError("objective must return a real loss or Err"))
        loss = Float64(value)
        return isfinite(loss) ? (loss, false) : (Inf, true)
    end

    function score_and_record(expr, generation)
        evaluations >= config.max_evaluations && return nothing
        loss, invalid = score_expression(expr)
        evaluations += 1
        invalid_evaluations += invalid
        serial += 1
        candidate = _ScalarCandidate(expr, loss, serial, generation > 0)
        record_archive!(candidate)
        if _candidate_key(candidate) < (best_loss, scalar_complexity(best_expr), typemax(Int))
            best_loss = loss
            best_expr = expr
            push!(improvements, (
                evaluation=evaluations,
                generation=generation,
                loss=loss,
                complexity=scalar_complexity(expr),
            ))
        end
        return candidate
    end

    for expr in expressions
        candidate = score_and_record(expr, 0)
        candidate === nothing && break
        push!(population, candidate)
    end
    sort!(population; by=candidate -> _selection_key(candidate, config))

    function refine_candidate(candidate, generation; source=:constant_refinement)
        config.constant_refinement_iterations == 0 && return candidate
        generation < config.constant_refinement_start_generation && return candidate
        remaining = config.max_constant_refinement_evaluations -
            constant_refinement_evaluations
        remaining <= 0 && return candidate
        refined_expr, refined_loss, refinement_count, refinement_invalid =
            _refine_constants(candidate.expr, candidate.loss, score_expression,
                config, remaining)
        constant_refinement_evaluations += refinement_count
        constant_refinement_invalid_evaluations += refinement_invalid
        refined_loss < candidate.loss || return candidate
        serial += 1
        refined = _ScalarCandidate(
            refined_expr, refined_loss, serial, candidate.evolved)
        record_archive!(refined)
        if _candidate_key(refined) <
                (best_loss, scalar_complexity(best_expr), typemax(Int))
            best_loss = refined_loss
            best_expr = refined_expr
            push!(improvements, (
                evaluation=evaluations,
                generation=generation,
                loss=refined_loss,
                complexity=scalar_complexity(refined_expr),
                source=source,
            ))
        end
        return refined
    end

    function refine_leaders!(candidates, generation;
            limit::Int=config.constant_refinement_elites)
        count = min(limit, length(candidates))
        for index in 1:count
            candidates[index] = refine_candidate(candidates[index], generation)
        end
        sort!(candidates; by=candidate -> _selection_key(candidate, config))
        return candidates
    end

    length(population) > config.population_size &&
        resize!(population, config.population_size)
    generations = 0
    should_stop() = stop_loss !== nothing && best_loss <= stop_loss

    function breed_child(first_parent, second_parent)
        draw = rand(rng)
        child = if draw < config.mutation_probability
            mutation_attempts += 1
            mutated, kind = _mutate_expression(
                rng, first_parent.expr, feature_count, config)
            mutation_kind_counts[kind] += 1
            mutated
        elseif draw < config.mutation_probability + config.crossover_probability
            crossover_attempts += 1
            _crossover_expression(rng, first_parent.expr, second_parent.expr, config)
        else
            composition_attempts += 1
            _compose_expression(rng, first_parent.expr, second_parent.expr, config)
        end
        changed_variations += _expression_signature(child) !=
            _expression_signature(first_parent.expr)
        return child
    end

    while evaluations < config.max_evaluations && !should_stop()
        generations += 1
        next_population = _ScalarCandidate[]
        retained_serials = Set{Int}()
        for candidate in Iterators.take(population, config.elite_count)
            push!(next_population, candidate)
            push!(retained_serials, candidate.serial)
        end
        complexity_buckets = _complexity_buckets(population)
        while length(next_population) < config.population_size &&
                evaluations < config.max_evaluations
            first_parent = _tournament(rng, population, config.tournament_size,
                config, complexity_buckets)
            second_parent = _tournament(rng, population, config.tournament_size,
                config, complexity_buckets)
            child = breed_child(first_parent, second_parent)
            candidate = score_and_record(child, generations)
            candidate === nothing && break
            if config.constant_refinement_iterations > 0 &&
                    config.constant_refinement_probability > 0 &&
                    rand(rng) < config.constant_refinement_probability
                candidate = refine_candidate(candidate, generations)
            end
            push!(next_population, candidate)
        end
        sort!(next_population; by=candidate -> _selection_key(candidate, config))
        if length(next_population) < config.population_size
            retained_serials = Set(candidate.serial for candidate in next_population)
            for candidate in population
                candidate.serial in retained_serials && continue
                push!(next_population, candidate)
                push!(retained_serials, candidate.serial)
                length(next_population) >= config.population_size && break
            end
            sort!(next_population; by=candidate -> _selection_key(candidate, config))
            population = next_population
            break
        end
        generations % config.constant_refinement_interval == 0 &&
            refine_leaders!(next_population, generations)
        population = next_population
    end

    if generations >= config.constant_refinement_start_generation &&
            config.constant_refinement_iterations > 0
        pareto_candidates = _ScalarCandidate[]
        best_frontier_loss = Inf
        for complexity in sort!(collect(keys(archive_by_complexity)))
            candidate = archive_by_complexity[complexity]
            if candidate.loss < best_frontier_loss
                push!(pareto_candidates, candidate)
                best_frontier_loss = candidate.loss
            end
        end
        sort!(pareto_candidates; by=_candidate_key)
        for candidate in pareto_candidates
            candidate.evolved || continue
            constant_refinement_evaluations >=
                config.max_constant_refinement_evaluations && break
            refine_candidate(
                candidate, generations; source=:final_archive_refinement)
        end
    end
    termination = should_stop() ? :solved : :evaluation_cap
    report_candidates = copy(population)
    push!(report_candidates, _ScalarCandidate(best_expr, best_loss, 0, true))
    sort!(report_candidates; by=_candidate_key)
    leaders = NamedTuple[]
    seen_structures = Set{_ScalarSignature}()
    for candidate in report_candidates
        signature = _structure_signature(candidate.expr)
        signature in seen_structures && continue
        push!(seen_structures, signature)
        push!(leaders, (
            expr=candidate.expr,
            loss=candidate.loss,
            complexity=scalar_complexity(candidate.expr),
        ))
        length(leaders) >= 8 && break
    end
    complexity_archive = [
        (
            expr=archive_by_complexity[complexity].expr,
            loss=archive_by_complexity[complexity].loss,
            complexity,
        )
        for complexity in sort!(collect(keys(archive_by_complexity)))
    ]
    variation_attempts = mutation_attempts + crossover_attempts + composition_attempts
    complexity_counts = Dict{Int,Int}()
    structures = Set{_ScalarSignature}()
    for candidate in population
        complexity = scalar_complexity(candidate.expr)
        complexity_counts[complexity] = get(complexity_counts, complexity, 0) + 1
        push!(structures, _structure_signature(candidate.expr))
    end
    final_complexity_histogram = [
        (complexity=complexity, count=complexity_counts[complexity])
        for complexity in sort!(collect(keys(complexity_counts)))
    ]
    variation = (
        attempts=variation_attempts,
        changed=changed_variations,
        unchanged=variation_attempts - changed_variations,
        mutation_attempts,
        mutation_kinds=(
            constant=mutation_kind_counts[:constant],
            operator=mutation_kind_counts[:operator],
            swap=mutation_kind_counts[:swap],
            insert=mutation_kind_counts[:insert],
            add=mutation_kind_counts[:add],
            graft=mutation_kind_counts[:graft],
            rotate=mutation_kind_counts[:rotate],
            scale=mutation_kind_counts[:scale],
            shift=mutation_kind_counts[:shift],
            restart=mutation_kind_counts[:restart],
            delete=mutation_kind_counts[:delete],
            subtree=mutation_kind_counts[:subtree],
        ),
        crossover_attempts,
        composition_attempts,
        final_population_size=length(population),
        final_unique_structures=length(structures),
        final_complexity_histogram,
    )
    return ScalarSearchResult(
        best_expr,
        best_loss,
        evaluations,
        invalid_evaluations,
        constant_refinement_evaluations,
        constant_refinement_invalid_evaluations,
        generations,
        termination,
        improvements,
        leaders,
        complexity_archive,
        variation,
    )
end

@testitem "scalar backends support zero-input matrices and check supplied feature ids" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ScalarSearch

    Xzero = zeros(Float64, 0, 3)
    yzero = fill(2.0, 3)
    Xpositive = reshape(Float64[-1.0, 0.0, 1.0], 1, :)
    ypositive = vec(Xpositive)
    for strategy in (:generational, :regularized)
        config = ScalarSearchConfig(
            population_size=2,
            max_evaluations=2,
            max_complexity=4,
            initial_depth=1,
            tournament_size=1,
            elite_count=1,
            unary_operators=(:square,),
            binary_operators=(:+,),
            constant_probability=0.0,
            constant_refinement_iterations=0,
            evolution_strategy=strategy,
        )
        zero_result = search_scalar(Xzero, yzero; config,
            rng=MersenneTwister(42),
            initial_population=[scalar_constant(0.5)], stop_loss=nothing)
        @test zero_result isa ScalarSearchResult
        @test zero_result.evaluations <= config.max_evaluations
        @test isfinite(zero_result.best_loss)
        @test ScalarSearch._required_features(zero_result.best) == 0
        @test_throws ArgumentError search_scalar(Xzero, yzero; config,
            rng=MersenneTwister(42),
            initial_population=[scalar_input(1)], stop_loss=nothing)

        positive_result = search_scalar(Xpositive, ypositive; config,
            rng=MersenneTwister(42),
            initial_population=[scalar_input(1)], stop_loss=nothing)
        @test isfinite(positive_result.best_loss)
        @test positive_result.best_loss <= 1.0e-12
        @test ScalarSearch._required_features(positive_result.best) <= 1
    end
end

@testitem "scalar constant refinement stops cleanly at budget boundary" begin
    using Test
    using LambdaRegression.ScalarSearch

    config = ScalarSearchConfig(constant_refinement_iterations=24,
        constant_refinement_optimizer=:bfgs)
    source = ScalarSearch.scalar_constant(0.0)
    score_calls = Ref(0)
    seen_losses = Float64[]
    score = expr -> begin
        score_calls[] += 1
        score_loss = (expr.value - 2.0)^2
        push!(seen_losses, score_loss)
        return (score_loss, false)
    end

    cap_one, loss_one, evals_one, invalid_one =
        ScalarSearch._refine_constants(source, 4.0, score, config, 1)
    @test cap_one === source
    @test loss_one == 4.0
    @test evals_one == score_calls[] == 1
    @test invalid_one == 0

    score_calls[] = 0
    empty!(seen_losses)
    cap_three, loss_three, evals_three, invalid_three =
        ScalarSearch._refine_constants(source, 4.0, score, config, 3)
    @test loss_three == min(4.0, minimum(seen_losses))
    @test (cap_three.value - 2.0)^2 == loss_three
    @test evals_three == score_calls[] <= 3
    @test invalid_three == 0

    empty!(seen_losses)
    improved, improved_loss, _, _ =
        ScalarSearch._refine_constants(source, 4.0, score, config, 16)
    @test improved_loss < 4.0
    @test improved_loss == minimum(seen_losses)
    @test (improved.value - 2.0)^2 == improved_loss

    no_improvement = _ -> (0.0, false)
    unchanged, unchanged_loss, _, _ = ScalarSearch._refine_constants(
        source, 0.0, no_improvement, config, 8)
    @test unchanged === source
    @test unchanged_loss == 0.0

    failing_score = _ -> error("scalar refinement callback failure")
    @test_throws ErrorException ScalarSearch._refine_constants(
        source, 4.0, failing_score, config, 3)
end

@testitem "finite negative objectives and stop thresholds work on both scalar backends" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ScalarSearch

    X = zeros(Float64, 0, 2)
    y = zeros(2)
    initial = [scalar_constant(0.0), scalar_constant(1.0)]
    objective = _ -> -2.0
    for strategy in (:generational, :regularized)
        config = ScalarSearchConfig(evolution_strategy=strategy,
            population_size=2, max_evaluations=4, max_complexity=1,
            initial_depth=0, tournament_size=1, elite_count=1,
            unary_operators=(), binary_operators=(:+,), constant_probability=0.0,
            initial_pool_multiplier=1, mutation_probability=0.0,
            crossover_probability=0.0, constant_refinement_iterations=0,
            max_constant_refinement_evaluations=0)
        default_result = search_scalar(X, y; config,
            rng=MersenneTwister(1111), initial_population=initial, objective)
        @test default_result.best_loss == -2.0
        @test default_result.invalid_evaluations == 0
        @test default_result.termination != :solved
        if strategy === :generational
            @test default_result.evaluations == config.max_evaluations
        end

        negative_threshold = search_scalar(X, y; config,
            rng=MersenneTwister(1111), initial_population=initial,
            objective, stop_loss=-1.0)
        @test negative_threshold.best_loss == -2.0
        @test negative_threshold.termination == :solved
        @test negative_threshold.invalid_evaluations == 0
    end
end
