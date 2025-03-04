"""Private node-native candidate operations for the regularized scalar island.

The module seam is deliberately narrow: candidates own unshared mutable
`DynamicExpressions.Node{Float64}` trees, cache their structural complexity,
and cross back to `ScalarExpr` only at the authoritative project boundary.
Every proposal copies its parent before mutation; callers may therefore retain
population, archive, and donor candidates without aliasing.
"""

struct _NodeScalarCandidate
    node::DynamicExpressions.Node{Float64}
    complexity::Int
end

struct _NodeScalarAnchor
    node::DynamicExpressions.Node{Float64}
    parent::Union{Nothing,DynamicExpressions.Node{Float64}}
    side::UInt8
end

function _node_scalar_candidate(expr::ScalarExpr)
    node = scalar_to_dynamic(expr)
    return _NodeScalarCandidate(node, DynamicExpressions.count_nodes(node))
end

"""Adopt an already-owned unshared node as an internal scalar candidate.

The caller supplies ownership and a node using `DYNAMIC_SCALAR_OPERATORS`;
complexity is counted immediately. Malformed operator indices fail later at
the strict materialisation/evaluation boundary rather than being normalised.
"""
function _node_scalar_candidate(node::DynamicExpressions.Node{Float64})
    return _NodeScalarCandidate(node, DynamicExpressions.count_nodes(node))
end

_node_scalar_complexity(candidate::_NodeScalarCandidate) = candidate.complexity
_node_scalar_materialize(candidate::_NodeScalarCandidate) =
    scalar_from_dynamic(candidate.node)
_node_scalar_evaluate(candidate::_NodeScalarCandidate, X) =
    evaluate_dynamic_scalar(candidate.node, X)

function _node_collect_anchors!(anchors, node, parent=nothing, side::UInt8=0x00)
    push!(anchors, _NodeScalarAnchor(node, parent, side))
    node.degree >= 1 && _node_collect_anchors!(anchors, node.l, node, 0x01)
    node.degree == 2 && _node_collect_anchors!(anchors, node.r, node, 0x02)
    return anchors
end

_node_anchors(node::DynamicExpressions.Node{Float64}) =
    _node_collect_anchors!(_NodeScalarAnchor[], node)

@inline _node_has_children(node) = node.degree > 0

function _node_is_rotatable(node)
    node.degree == 1 && return _node_has_children(node.l)
    node.degree == 2 || return false
    return _node_has_children(node.l) || _node_has_children(node.r)
end

function _node_replace_anchor(root, anchor::_NodeScalarAnchor, replacement)
    anchor.parent === nothing && return replacement
    if anchor.side == 0x01
        anchor.parent.l = replacement
    elseif anchor.side == 0x02
        anchor.parent.r = replacement
    else
        error("non-root node anchor has an invalid side")
    end
    return root
end

"""Choose unary or binary construction from the configured operator world.

The operator counts are the only anchoring information. `config` must contain
at least one scalar operator, as enforced by `ScalarSearchConfig`; after that
validation the chooser cannot fail.
"""
function _node_choose_binary(rng::AbstractRNG, config::ScalarSearchConfig)
    isempty(config.binary_operators) && return false
    isempty(config.unary_operators) && return true
    total = length(config.binary_operators) + length(config.unary_operators)
    return rand(rng) < length(config.binary_operators) / total
end

"""Construct one finite node leaf from the feature and constant context.

`feature_count` supplies one-based feature ids and `config` supplies the
constant probability/scale. Invalid feature counts are rejected by the public
search interface; generated constants are finite for the supported RNGs.
"""
function _node_random_leaf(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig)
    if rand(rng) < config.constant_probability
        return DynamicExpressions.Node(Float64;
            val=randn(rng) * config.constant_scale)
    end
    return DynamicExpressions.Node(Float64; feature=rand(rng, 1:feature_count))
end

"""Construct a bounded-depth node tree in the configured scalar grammar.

Depth and the operator world anchor the recursion. The public configuration
and feature checks supply the required context; no expected failure remains
after those checks.
"""
function _node_random_expression(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig, depth::Int)
    depth <= 0 && return _node_random_leaf(rng, feature_count, config)
    rand(rng) < 1 / (depth + 2) &&
        return _node_random_leaf(rng, feature_count, config)
    if _node_choose_binary(rng, config)
        return DynamicExpressions.Node(Float64;
            op=_dynamic_binary_index(rand(rng, config.binary_operators)),
            l=_node_random_expression(rng, feature_count, config, depth - 1),
            r=_node_random_expression(rng, feature_count, config, depth - 1))
    end
    return DynamicExpressions.Node(Float64;
        op=_dynamic_unary_index(rand(rng, config.unary_operators)),
        l=_node_random_expression(rng, feature_count, config, depth - 1))
end

"""Grow from one leaf by replacing uniformly sampled leaves.

The feature count and configured operator world are the complete grammar
context. It fails through the underlying leaf/operator chooser when that world
cannot construct a finite operation.
"""
function _node_random_grow(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig, operations::Int)
    root = _node_random_leaf(rng, feature_count, config)
    for _ in 1:operations
        anchors = _node_anchors(root)
        leaf = rand(rng, filter(anchor -> anchor.node.degree == 0, anchors))
        root = _node_replace_anchor(root, leaf,
            _node_fresh_operation(rng, feature_count, config))
    end
    return root
end

"""Construct a finite tree anchored to an approximate requested node count.

The configured unary/binary world and feature count supply all children. When
the grammar cannot realise the exact parity/arity, the helper returns the
largest smaller positive tree; nonpositive requests fail with `ArgumentError`.
"""
function _node_random_fixed_size(rng::AbstractRNG, feature_count::Int,
        config::ScalarSearchConfig, nodes::Int)
    nodes >= 1 || throw(ArgumentError("random node size must be positive"))
    nodes == 1 && return _node_random_leaf(rng, feature_count, config)
    can_unary = !isempty(config.unary_operators)
    can_binary = !isempty(config.binary_operators) && nodes >= 3
    if can_unary && (!can_binary || rand(rng, Bool))
        return DynamicExpressions.Node(Float64;
            op=_dynamic_unary_index(rand(rng, config.unary_operators)),
            l=_node_random_fixed_size(rng, feature_count, config, nodes - 1))
    end
    can_binary || return _node_random_fixed_size(
        rng, feature_count, config, nodes - 1)
    left_nodes = rand(rng, 1:(nodes - 2))
    right_nodes = nodes - 1 - left_nodes
    return DynamicExpressions.Node(Float64;
        op=_dynamic_binary_index(rand(rng, config.binary_operators)),
        l=_node_random_fixed_size(rng, feature_count, config, left_nodes),
        r=_node_random_fixed_size(rng, feature_count, config, right_nodes))
end

"""Choose one configured operation and anchor it over fresh scalar leaves.

The chooser uses only `rng`, `feature_count`, and the unary/binary operator
fields of `config`. An empty operator world fails at the shared arity chooser.
"""
function _node_fresh_operation(rng, feature_count, config)
    if _node_choose_binary(rng, config)
        return DynamicExpressions.Node(Float64;
            op=_dynamic_binary_index(rand(rng, config.binary_operators)),
            l=_node_random_leaf(rng, feature_count, config),
            r=_node_random_leaf(rng, feature_count, config))
    end
    return DynamicExpressions.Node(Float64;
        op=_dynamic_unary_index(rand(rng, config.unary_operators)),
        l=_node_random_leaf(rng, feature_count, config))
end

"""Choose one configured operation anchored around an owned child node.

Binary choices place `node` on the left and create one fresh right leaf; unary
choices wrap it directly. An empty operator world fails at the shared chooser.
"""
function _node_wrap_operation(rng, node, feature_count, config)
    if _node_choose_binary(rng, config)
        return DynamicExpressions.Node(Float64;
            op=_dynamic_binary_index(rand(rng, config.binary_operators)),
            l=node,
            r=_node_random_leaf(rng, feature_count, config))
    end
    return DynamicExpressions.Node(Float64;
        op=_dynamic_unary_index(rand(rng, config.unary_operators)), l=node)
end

"""List configured same-arity operator indices excluding the current one.

The node arity and configured operator world anchor the list. Unsupported node
degrees follow the binary branch and fail when converted or proposed.
"""
function _node_allowed_operator_indices(node, config)
    symbols = node.degree == 1 ? config.unary_operators : config.binary_operators
    indices = node.degree == 1 ?
        UInt8[_dynamic_unary_index(op) for op in symbols] :
        UInt8[_dynamic_binary_index(op) for op in symbols]
    return filter(!=(node.op), indices)
end

"""Choose one local arity-preserving rotation of an owned operation root.

Only the supplied root and caller RNG are consulted. Leaves and unsupported
child shapes return the original root; malformed operator indices remain an
unexpected programmer error at evaluation/materialisation.
"""
function _node_rotate(rng, root)
    if root.degree == 1
        child = root.l
        child.degree == 1 && return DynamicExpressions.Node(Float64;
            op=child.op,
            l=DynamicExpressions.Node(Float64; op=root.op, l=child.l))
        child.degree == 2 || return root
        return DynamicExpressions.Node(Float64; op=child.op, l=child.l,
            r=DynamicExpressions.Node(Float64; op=root.op, l=child.r))
    end
    root.degree == 2 || return root
    rotate_left_child = _node_has_children(root.l) &&
        (!_node_has_children(root.r) || rand(rng, Bool))
    child = rotate_left_child ? root.l : root.r
    if rotate_left_child && child.degree == 1
        return DynamicExpressions.Node(Float64; op=child.op,
            l=DynamicExpressions.Node(Float64; op=root.op,
                l=child.l, r=root.r))
    elseif rotate_left_child && child.degree == 2
        return DynamicExpressions.Node(Float64; op=child.op, l=child.l,
            r=DynamicExpressions.Node(Float64; op=root.op,
                l=child.r, r=root.r))
    elseif child.degree == 1
        return DynamicExpressions.Node(Float64; op=child.op,
            l=DynamicExpressions.Node(Float64; op=root.op,
                l=root.l, r=child.l))
    elseif child.degree == 2
        return DynamicExpressions.Node(Float64; op=child.op,
            l=DynamicExpressions.Node(Float64; op=root.op,
                l=root.l, r=child.l), r=child.r)
    end
    return root
end

function _node_finalize_proposal(root, parent::_NodeScalarCandidate,
        config::ScalarSearchConfig)
    complexity = DynamicExpressions.count_nodes(root)
    complexity <= config.max_complexity || return (parent, false)
    root == parent.node && return (parent, false)
    return (_NodeScalarCandidate(root, complexity), true)
end

"""Build one copy-on-propose node candidate.

Mutation anchors are selected uniformly from the copied parent. The donor,
feature count, and `config` provide crossover and grammar context. Unsupported
or inapplicable kinds, unchanged results, and complexity overflow return the
original candidate with `false`; programmer errors still surface at source.
"""
function _node_scalar_propose(rng::AbstractRNG, parent::_NodeScalarCandidate,
        donor::_NodeScalarCandidate, feature_count::Int,
        config::ScalarSearchConfig, kind::Symbol; temperature::Real=1.0)
    kind in (:do_nothing, :simplify, :optimize) && return (parent, false)
    if kind === :restart
        root = _node_random_fixed_size(
            rng, feature_count, config, rand(rng, 1:config.max_complexity))
        return _node_finalize_proposal(root, parent, config)
    elseif kind === :compose
        isempty(config.binary_operators) && return (parent, false)
        root = DynamicExpressions.Node(Float64;
            op=_dynamic_binary_index(rand(rng, config.binary_operators)),
            l=copy(parent.node; break_sharing=Val(true)),
            r=copy(donor.node; break_sharing=Val(true)))
        return _node_finalize_proposal(root, parent, config)
    end

    root = copy(parent.node; break_sharing=Val(true))
    anchors = _node_anchors(root)
    if kind === :crossover
        target = rand(rng, anchors)
        donor_anchor = rand(rng, _node_anchors(donor.node))
        replacement = copy(donor_anchor.node; break_sharing=Val(true))
        root = _node_replace_anchor(root, target, replacement)
        return _node_finalize_proposal(root, parent, config)
    elseif kind === :add
        if rand(rng, Bool)
            leaves = filter(anchor -> !_node_has_children(anchor.node), anchors)
            target = rand(rng, leaves)
            root = _node_replace_anchor(root, target,
                _node_fresh_operation(rng, feature_count, config))
        else
            root = _node_wrap_operation(rng, root, feature_count, config)
        end
        return _node_finalize_proposal(root, parent, config)
    end

    compatible = if kind === :operator
        filter(anchor -> anchor.node.degree > 0 &&
            !isempty(_node_allowed_operator_indices(anchor.node, config)), anchors)
    elseif kind === :swap
        filter(anchor -> anchor.node.degree == 2, anchors)
    elseif kind === :constant
        filter(anchor -> anchor.node.degree == 0 && anchor.node.constant, anchors)
    elseif kind === :rotate
        filter(anchor -> _node_is_rotatable(anchor.node), anchors)
    elseif kind === :delete
        filter(anchor -> _node_has_children(anchor.node), anchors)
    else
        anchors
    end
    isempty(compatible) && return (parent, false)
    target = rand(rng, compatible)
    node = target.node
    if kind === :insert
        root = _node_replace_anchor(root, target,
            _node_wrap_operation(rng, node, feature_count, config))
    elseif kind === :rotate
        root = _node_replace_anchor(root, target, _node_rotate(rng, node))
    elseif kind === :delete
        replacement = node.degree == 1 ? node.l :
            (rand(rng, Bool) ? node.l : node.r)
        root = _node_replace_anchor(root, target, replacement)
    elseif kind === :operator
        node.op = rand(rng, _node_allowed_operator_indices(node, config))
    elseif kind === :swap
        node.l, node.r = node.r, node.l
    elseif kind === :constant
        max_change = 1.1 + 0.129 * Float64(temperature)
        factor = max_change^rand(rng)
        node.val *= rand(rng, Bool) ? factor : inv(factor)
        rand(rng) < 0.00743 && (node.val = -node.val)
    else
        return (parent, false)
    end
    return _node_finalize_proposal(root, parent, config)
end

"""Exchange one uniformly selected subtree in both owned node parents.

Both parents are copied before rewiring. The pair is accepted only when both
children are changed and satisfy `config.max_complexity`; otherwise the two
original candidates and `false` are returned. Evaluation and replacement are
left to the regularized island.
"""
function _node_scalar_crossover_pair(rng::AbstractRNG,
        first::_NodeScalarCandidate, second::_NodeScalarCandidate,
        config::ScalarSearchConfig)
    first_root = copy(first.node; break_sharing=Val(true))
    second_root = copy(second.node; break_sharing=Val(true))
    first_anchor = rand(rng, _node_anchors(first_root))
    second_anchor = rand(rng, _node_anchors(second_root))
    first_subtree = copy(first_anchor.node; break_sharing=Val(true))
    second_subtree = copy(second_anchor.node; break_sharing=Val(true))
    first_root = _node_replace_anchor(first_root, first_anchor, second_subtree)
    second_root = _node_replace_anchor(second_root, second_anchor, first_subtree)
    first_child, first_changed = _node_finalize_proposal(first_root, first, config)
    second_child, second_changed = _node_finalize_proposal(second_root, second, config)
    first_changed && second_changed || return (first, second, false)
    return (first_child, second_child, true)
end

"""
    _node_scalar_sample_mutation(rng, candidate, config) -> Symbol

Sample one regularized-evolution mutation directly from a node candidate.
Eligibility is computed from one anchor traversal and conditions the existing
`_REGULARIZED_MUTATION_WEIGHTS`; in particular `:optimize` remains disabled
because constant refinement has a separate budget.  The chooser only returns
operation names and never rebuilds the candidate through `ScalarExpr`.
"""
function _node_scalar_sample_mutation(rng::AbstractRNG,
        candidate::_NodeScalarCandidate, config::ScalarSearchConfig)
    anchors = _node_anchors(candidate.node)
    has_constant = any(a -> a.node.degree == 0 && a.node.constant, anchors)
    has_alternate_operator = any(a -> a.node.degree == 1 || a.node.degree == 2 ?
        !isempty(_node_allowed_operator_indices(a.node, config)) : false, anchors)
    has_binary = any(a -> a.node.degree == 2, anchors)
    has_children = any(a -> a.node.degree > 0, anchors)
    has_rotation = any(a -> _node_is_rotatable(a.node), anchors)
    under_cap = candidate.complexity < config.max_complexity

    names = _REGULARIZED_MUTATION_NAMES
    weights = (
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
        0.0, # optimize: independent constant-refinement budget
    )
    eligible = (
        has_constant,
        has_alternate_operator,
        has_binary,
        has_rotation,
        under_cap,
        under_cap,
        has_children,
        true,
        true,
        true,
        false,
    )
    conditioned = ntuple(i -> eligible[i] ? weights[i] : 0.0, length(weights))
    total = sum(conditioned)
    total > 0 || return :do_nothing
    draw = rand(rng) * total
    cumulative = 0.0
    for i in eachindex(names)
        cumulative += conditioned[i]
        draw < cumulative && return names[i]
    end
    return last(names)
end

@testsnippet NodeRegularizedImports begin
    using Test
    using Random
    using LambdaRegression.ScalarSearch
end

@testitem "node-native scalar candidates preserve the authoritative boundary" setup=[NodeRegularizedImports] begin
    expr = scalar_binary(:+,
        scalar_unary(:sin, scalar_input(1)),
        scalar_binary(:*, scalar_constant(0.25), scalar_input(2)))
    candidate = ScalarSearch._node_scalar_candidate(expr)
    X = [0.1 0.2 0.3; -1.0 0.5 2.0]

    @test ScalarSearch._node_scalar_complexity(candidate) == scalar_complexity(expr)
    @test ScalarSearch._node_scalar_evaluate(candidate, X) ≈
        evaluate_scalar_reference(expr, X)
    rebuilt = ScalarSearch._node_scalar_materialize(candidate)
    @test rebuilt isa ScalarExpr
    @test evaluate_scalar_reference(rebuilt, X) ≈ evaluate_scalar_reference(expr, X)
end

@testitem "node-native proposals preserve parents and remain materialisable" setup=[NodeRegularizedImports] begin
    config = ScalarSearchConfig(
        population_size=8,
        max_evaluations=16,
        max_complexity=16,
        initial_depth=2,
        tournament_size=3,
        elite_count=2,
        unary_operators=(:sin, :cos, :sqrt, :square),
        binary_operators=(:+, :-, :*, :/),
        evolution_strategy=:regularized,
    )
    parent_expr = scalar_binary(:+,
        scalar_unary(:sin, scalar_input(1)),
        scalar_binary(:*, scalar_constant(0.25), scalar_input(1)))
    donor_expr = scalar_unary(:cos, scalar_input(2))
    parent = ScalarSearch._node_scalar_candidate(parent_expr)
    donor = ScalarSearch._node_scalar_candidate(donor_expr)
    X = [0.1 0.2 0.3; 2.0 1.0 -1.0]
    parent_before = ScalarSearch._node_scalar_evaluate(parent, X)

    changed = Ref(0)
    for (offset, kind) in enumerate((
            :constant, :operator, :swap, :rotate, :add, :insert, :delete, :restart))
        child, did_change = ScalarSearch._node_scalar_propose(
            MersenneTwister(200 + offset), parent, donor, 2, config, kind)
        changed[] += did_change
        @test ScalarSearch._node_scalar_complexity(child) <= config.max_complexity
        @test ScalarSearch._node_scalar_materialize(child) isa ScalarExpr
        @test ScalarSearch._node_scalar_evaluate(parent, X) == parent_before
    end
    @test changed[] >= 5

    crossed, did_cross = ScalarSearch._node_scalar_propose(
        MersenneTwister(1234), parent, donor, 2, config, :crossover)
    @test did_cross
    @test ScalarSearch._node_scalar_complexity(crossed) <= config.max_complexity
    @test ScalarSearch._node_scalar_materialize(crossed) isa ScalarExpr
    @test ScalarSearch._node_scalar_evaluate(parent, X) == parent_before
end

@testitem "node-native mutation sampling conditions the regularized palette" setup=[NodeRegularizedImports] begin
    config = ScalarSearchConfig(
        population_size=8, max_evaluations=16, max_complexity=24,
        initial_depth=2, tournament_size=3, elite_count=2,
        unary_operators=(:sin, :cos, :sqrt, :square),
        binary_operators=(:+, :-, :*, :/), evolution_strategy=:regularized,
    )
    # Rich enough to make every positive-weight structural eligibility true.
    expr = scalar_binary(:+,
        scalar_binary(:*, scalar_unary(:sin, scalar_input(1)), scalar_input(2)),
        scalar_binary(:-, scalar_constant(0.5), scalar_input(1)))
    candidate = ScalarSearch._node_scalar_candidate(expr)
    allowed = Set(ScalarSearch._REGULARIZED_MUTATION_NAMES)
    counts = Dict{Symbol,Int}(kind => 0 for kind in allowed)
    rng = MersenneTwister(2000)
    draws = 2_000
    # This locks the conditioned palette and accounting contract, not the
    # reachability of every rare event in the reference distribution.
    for _ in 1:draws
        kind = ScalarSearch._node_scalar_sample_mutation(rng, candidate, config)
        @test kind in allowed
        counts[kind] = get(counts, kind, 0) + 1
    end
    @test sum(values(counts)) == draws
    @test counts[:constant] > 0
    @test counts[:operator] > 0
    @test counts[:swap] > 0
    @test counts[:rotate] > 0
    @test counts[:add] > 0
    @test counts[:delete] > 0
    @test counts[:do_nothing] > 0
    @test counts[:optimize] == 0
end
