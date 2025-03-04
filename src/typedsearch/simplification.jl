"""Bounded local simplification of typed compact λ terms.

`simplify_term(rng, term, language)` scans the supplied term for ordinary
application redexes and uniformly selects one of the two deliberately small
rules supported here: `(λx.x) a → a`, or `(λx.b) a → b` when `x` has no free
occurrence in `b`.  With `atomic_beta=true`, it also admits one ordinary
β-contraction when `a` is a variable or a non-program language leaf.  The
selected subtree is copied with the required De Bruijn lowering; no general
β-substitution, donor expansion, or score reuse is performed.  In particular,
an atomic argument is copied at each bound occurrence, while an admitted
`SourceProgram` leaf remains ineligible for this opt-in rule.  An input with no
eligible redex returns `Err(TypedSearchError(:no_simplifiable_redex))` without
consuming `rng`.
"""

using Random: AbstractRNG, rand

export simplify_term

struct _SimplifiableRedex
    root::Int
    rule::Symbol
end

@inline _simplify_fail(reason::Symbol, detail=nothing) =
    Err(TypedSearchError(reason, detail))

"""Whether `body` contains the λ binder at relative De Bruijn depth `depth`."""
function _simplify_binder_used(nodes::Vector{TermNode}, root::Int, depth::Int=0)
    node = nodes[root]
    if node.kind == VAR_KIND
        return Int(node.x) == depth
    elseif node.kind == ABS_KIND
        return _simplify_binder_used(nodes, Int(node.x), depth + 1)
    elseif node.kind == APP_KIND
        return _simplify_binder_used(nodes, Int(node.x), depth) ||
            _simplify_binder_used(nodes, Int(node.y), depth)
    end
    return false
end

function _simplify_redex_rule(nodes::Vector{TermNode}, root::Int,
                              language::Union{Nothing,Language}=nothing;
                              atomic_beta::Bool=false)
    app = nodes[root]
    app.kind == APP_KIND || return nothing
    fn = nodes[Int(app.x)]
    fn.kind == ABS_KIND || return nothing
    body = nodes[Int(fn.x)]
    # This rule is intentionally checked before the unused-binder rule: the
    # variable is used, but identity has its own capture-free reduction.
    body.kind == VAR_KIND && body.x == 0 && return :identity
    !_simplify_binder_used(nodes, Int(fn.x)) && return :unused_binder
    if atomic_beta
        language isa Language || throw(ArgumentError(
            "atomic simplification requires a language"))
        argument = nodes[Int(app.y)]
        if argument.kind == VAR_KIND
            return :atomic_beta
        elseif argument.kind == LEAF_KIND
            payload = language.leaves[Int(argument.x)].payload
            payload isa SourceProgram || return :atomic_beta
        end
    end
    return nothing
end

function _simplify_collect!(out::Vector{_SimplifiableRedex},
                            nodes::Vector{TermNode}, root::Int,
                            language::Union{Nothing,Language}=nothing;
                            atomic_beta::Bool=false)
    node = nodes[root]
    if node.kind == ABS_KIND
        _simplify_collect!(out, nodes, Int(node.x), language;
            atomic_beta=atomic_beta)
    elseif node.kind == APP_KIND
        rule = _simplify_redex_rule(nodes, root, language;
            atomic_beta=atomic_beta)
        rule === nothing || push!(out, _SimplifiableRedex(root, rule))
        _simplify_collect!(out, nodes, Int(node.x), language;
            atomic_beta=atomic_beta)
        _simplify_collect!(out, nodes, Int(node.y), language;
            atomic_beta=atomic_beta)
    end
    return out
end

function _simplify_depth(term::TypedTerm)
    depths = zeros(Int, length(term.nodes))
    for (i, node) in enumerate(term.nodes)
        if node.kind == ABS_KIND
            depths[i] = 1 + depths[Int(node.x)]
        elseif node.kind == APP_KIND
            depths[i] = 1 + max(depths[Int(node.x)], depths[Int(node.y)])
        end
    end
    return depths[end]
end

"""Copy a subtree, optionally lowering free De Bruijn indices by `shift`.

`local_depth` counts binders introduced inside the copied subtree.  Variables
`db < local_depth` are bound there and remain unchanged; all other variables
are residual references to the subtree's enclosing context and receive the
requested shift.  The recursive call used for a selected redex disables the
selection so a shared compact index cannot be rewritten twice.
"""
function _simplify_copy!(out::Vector{TermNode}, nodes::Vector{TermNode}, root::Int,
                         selected_root::Int, selected_rule::Symbol,
                         shift::Int=0, local_depth::Int=0)
    if root == selected_root
        app = nodes[root]
        fn = nodes[Int(app.x)]
        if selected_rule === :atomic_beta
            return _simplify_copy_atomic!(out, nodes, Int(fn.x), Int(app.y))
        end
        replacement = selected_rule === :identity ? Int(app.y) : Int(fn.x)
        replacement_shift = selected_rule === :identity ? 0 : -1
        return _simplify_copy!(out, nodes, replacement, 0, :none,
                               replacement_shift, 0)
    end

    node = nodes[root]
    if node.kind == VAR_KIND
        db = Int(node.x)
        shifted = db >= local_depth ? db + shift : db
        shifted >= 0 || return _simplify_fail(:invalid_binder_shift, (root, db, shift))
        push!(out, var_node(node.typeid, shifted))
    elseif node.kind == LEAF_KIND
        push!(out, leaf_node(node.typeid, node.x))
    elseif node.kind == ABS_KIND
        body = _simplify_copy!(out, nodes, Int(node.x), selected_root,
            selected_rule, shift, local_depth + 1)
        body isa Err && return body
        push!(out, abs_node(node.typeid, body))
    elseif node.kind == APP_KIND
        fn = _simplify_copy!(out, nodes, Int(node.x), selected_root,
            selected_rule, shift, local_depth)
        fn isa Err && return fn
        arg = _simplify_copy!(out, nodes, Int(node.y), selected_root,
            selected_rule, shift, local_depth)
        arg isa Err && return arg
        push!(out, app_node(node.typeid, fn, arg))
    else
        return _simplify_fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

"""Copy `body`, substituting one atomic argument for its redex binder.

`local_depth` is the number of binders introduced while descending into the
body.  A reference at that depth points at the removed binder and is replaced
by the argument atom; a variable argument is shifted under those local
binders.  References below that binder are lowered by one.  The helper never
accepts an application or abstraction argument, and reads the input nodes
without mutating them.
"""
function _simplify_copy_atomic!(out::Vector{TermNode}, nodes::Vector{TermNode},
                                root::Int, argument_root::Int,
                                local_depth::Int=0)
    node = nodes[root]
    if node.kind == VAR_KIND
        db = Int(node.x)
        if db < local_depth
            push!(out, var_node(node.typeid, db))
        elseif db == local_depth
            argument = nodes[argument_root]
            if argument.kind == LEAF_KIND
                push!(out, leaf_node(node.typeid, argument.x))
            elseif argument.kind == VAR_KIND
                replacement_db = Int(argument.x) + local_depth
                replacement_db >= 0 || return _simplify_fail(
                    :invalid_binder_shift, (root, argument.x, local_depth))
                push!(out, var_node(node.typeid, replacement_db))
            else
                return _simplify_fail(:non_atomic_argument, argument.kind)
            end
        else
            push!(out, var_node(node.typeid, db - 1))
        end
    elseif node.kind == LEAF_KIND
        push!(out, leaf_node(node.typeid, node.x))
    elseif node.kind == ABS_KIND
        body = _simplify_copy_atomic!(out, nodes, Int(node.x), argument_root,
            local_depth + 1)
        body isa Err && return body
        push!(out, abs_node(node.typeid, body))
    elseif node.kind == APP_KIND
        fn = _simplify_copy_atomic!(out, nodes, Int(node.x), argument_root,
            local_depth)
        fn isa Err && return fn
        arg = _simplify_copy_atomic!(out, nodes, Int(node.y), argument_root,
            local_depth)
        arg isa Err && return arg
        push!(out, app_node(node.typeid, fn, arg))
    else
        return _simplify_fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

"""
    simplify_term(rng, term, language; context=Int32[], max_nodes=128,
                  max_depth=32, atomic_beta=false)

Return one new term after uniformly selecting an eligible local redex.  The
operation is a normal proposal: callers must score the returned term and may
replace the original independently.  `context` supplies visible outer
locals for open terms.  Input and output are validated against the supplied
language; malformed terms and resource violations return `Err`.  The bounded
scan is deterministic apart from one uniform occurrence draw, with a worst
case quadratic cost when many candidate bodies contain one another.

When `atomic_beta=true`, one additional eligible occurrence is selected from
the same uniform candidate pool.  Its argument must be a VAR or a language
LEAF whose payload is not `SourceProgram`; applications and abstractions are
not substituted.  This is a proposal only: the caller re-evaluates the child
under its normal objective and execution budget.  The compact result is
checked against both node and structural-depth bounds.  Because the admitted
argument is one compact atom, the selected strict redex removes its APP, ABS,
and argument nodes (three compact nodes) even when the binder occurs more than
once; each occurrence receives one copied atom.
"""
function simplify_term(rng::AbstractRNG, term::TypedTerm, language::Language;
                       context::AbstractVector{<:Integer}=Int32[],
                       max_nodes::Integer=128, max_depth::Integer=32,
                       atomic_beta::Bool=false)
    max_nodes >= 1 || return _simplify_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _simplify_fail(:invalid_depth_budget, max_depth)
    length(term.nodes) <= max_nodes ||
        return _simplify_fail(:node_budget_exceeded, length(term.nodes))
    checked = validate(term, language; context=context)
    checked === true || return checked
    depth = _simplify_depth(term)
    depth <= max_depth || return _simplify_fail(:depth_budget_exceeded, depth)

    candidates = _SimplifiableRedex[]
    _simplify_collect!(candidates, term.nodes, length(term.nodes), language;
        atomic_beta=atomic_beta)
    isempty(candidates) && return _simplify_fail(:no_simplifiable_redex)
    selected = candidates[rand(rng, 1:length(candidates))]

    out = TermNode[]
    sizehint!(out, max(1, length(term.nodes) - 2))
    copied = _simplify_copy!(out, term.nodes, length(term.nodes),
        selected.root, selected.rule)
    copied isa Err && return copied
    result = TypedTerm(out)
    length(result.nodes) <= max_nodes ||
        return _simplify_fail(:node_budget_exceeded, length(result.nodes))
    checked = validate(result, language; context=context)
    checked === true || return checked
    result_depth = _simplify_depth(result)
    result_depth <= max_depth || return _simplify_fail(
        :depth_budget_exceeded, result_depth)
    return result
end

@testitem "TypedSearch bounded simplification preserves binders" begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(scalar, :arg),
                                LeafSpec(fn, :f)])

    # (λx.x) a -> a.
    identity = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), leaf_node(scalar, 1),
        app_node(scalar, 2, 3),
    ])
    reduced = simplify_term(MersenneTwister(1), identity, language)
    @test reduced isa TypedTerm
    @test reduced.nodes == TermNode[leaf_node(scalar, 1)]
    @test validate(reduced, language) === true
    over_nodes = simplify_term(MersenneTwister(1), identity, language; max_nodes=3)
    @test over_nodes isa Err
    @test over_nodes.value.reason == :node_budget_exceeded
    over_depth = simplify_term(MersenneTwister(1), identity, language; max_depth=1)
    @test over_depth isa Err
    @test over_depth.value.reason == :depth_budget_exceeded

    # (λx. λy. outer) arg -> λy. outer, with outer DB lowered from 2 to 1.
    nested = TypedTerm(TermNode[
        var_node(scalar, 2), abs_node(fn, 1), abs_node(arrow!(table, scalar, fn), 2),
        leaf_node(scalar, 2), app_node(fn, 3, 4),
    ])
    @test validate(nested, language; context=Int32[scalar]) === true
    nested_reduced = simplify_term(MersenneTwister(1111), nested, language;
        context=Int32[scalar])
    @test nested_reduced isa TypedTerm
    @test nested_reduced.nodes == TermNode[
        var_node(scalar, 1), abs_node(fn, 1),
    ]
    @test validate(nested_reduced, language; context=Int32[scalar]) === true

    # A higher-order binder used in a non-identity body is not simplified.
    higher = TypedTerm(TermNode[
        var_node(fn, 0), var_node(scalar, 1), app_node(scalar, 1, 2),
        abs_node(arrow!(table, fn, scalar), 3), leaf_node(fn, 3),
        app_node(scalar, 4, 5),
    ])
    @test validate(higher, language; context=Int32[scalar]) === true
    blocked = simplify_term(MersenneTwister(1234), higher, language;
        context=Int32[scalar])
    @test blocked isa Err
    @test blocked.value.reason == :no_simplifiable_redex
    untouched_rng = MersenneTwister(8000)
    expected_next = rand(MersenneTwister(8000))
    @test simplify_term(untouched_rng, higher, language;
        context=Int32[scalar]) isa Err
    @test rand(untouched_rng) == expected_next
end

@testitem "TypedSearch search dispatches simplification proposals" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(scalar, :a)])
    seed = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), leaf_node(scalar, 1),
        app_node(scalar, 2, 3),
    ])
    result = search_terms([seed], language, term -> length(term.nodes) == 1 ? 0.0 : 1.0;
        population_size=2, attempts=4, evaluations=5,
        kinds=(:simplify,), weights=(1.0,), rng=MersenneTwister(42))
    @test result isa TypedSearchResult
    @test result.stats.kind_counts[:simplify] == result.stats.attempts
    @test result.stats.evaluations <= 5
    @test result.stats.operator_stats[:simplify].failures > 0
    @test all(member -> validate(member.term, language) === true, result.population)
end

@testitem "TypedSearch simplification keeps donor leaves closed" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    donor_leaf = register_program!(language, :donor,
        AstTree(ConstNode(:primitive)), Dict(:primitive => FloatTy), FloatTy;
        provenance=:simplification_test)
    @test donor_leaf isa Integer
    fn = arrow!(language.types, scalar, scalar)
    term = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), leaf_node(scalar, donor_leaf),
        app_node(scalar, 2, 3),
    ])
    reduced = simplify_term(MersenneTwister(8000), term, language)
    @test reduced isa TypedTerm
    @test length(reduced.nodes) == 1
    @test language.leaves[Int(reduced.nodes[1].x)].payload isa SourceProgram
    @test validate(reduced, language) === true
    reduced_atomic = simplify_term(MersenneTwister(8000), term, language;
        atomic_beta=true)
    @test reduced_atomic isa TypedTerm
    @test reduced_atomic.nodes == reduced.nodes
    @test language.leaves[Int(reduced_atomic.nodes[1].x)].payload isa SourceProgram
end

@testitem "TypedSearch opt-in atomic beta preserves typed binding" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch: SourceConstant, SourceExternal
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    curried = arrow!(table, scalar, fn)
    higher = arrow!(table, fn, fn)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(fn, :f),
        LeafSpec(curried, :add), LeafSpec(higher, :g)])

    # (λx.f x) a -> f a.  The old rules do not admit this redex.
    function scalar_apply(argument)
        return TypedTerm(TermNode[
            leaf_node(fn, 2), var_node(scalar, 0), app_node(scalar, 1, 2),
            abs_node(fn, 3), leaf_node(scalar, argument),
            app_node(scalar, 4, 5),
        ])
    end
    ordinary = scalar_apply(1)
    @test validate(ordinary, language) === true
    @test simplify_term(MersenneTwister(1), ordinary, language) isa Err
    before = copy(ordinary.nodes)
    reduced = simplify_term(MersenneTwister(1), ordinary, language;
        atomic_beta=true)
    @test reduced isa TypedTerm
    @test reduced.nodes == TermNode[
        leaf_node(fn, 2), leaf_node(scalar, 1), app_node(scalar, 1, 2),
    ]
    @test length(reduced.nodes) == length(ordinary.nodes) - 3
    @test validate(reduced, language) === true
    @test ordinary.nodes == before

    # An outer variable must remain outside a nested binder after substitution.
    capture = TypedTerm(TermNode[
        var_node(scalar, 1), abs_node(fn, 1),
        abs_node(arrow!(table, scalar, fn), 2), var_node(scalar, 0),
        app_node(fn, 3, 4),
    ])
    @test validate(capture, language; context=Int32[scalar]) === true
    capture_reduced = simplify_term(MersenneTwister(1111), capture, language;
        context=Int32[scalar], atomic_beta=true)
    @test capture_reduced isa TypedTerm
    @test capture_reduced.nodes == TermNode[
        var_node(scalar, 1), abs_node(fn, 1),
    ]
    @test validate(capture_reduced, language; context=Int32[scalar]) === true

    # Repeated use copies the one-node argument at each occurrence, while an
    # outer reference is lowered under the removed binder.  The body is
    # `λy. add x (add x outer)`.
    repeated = TypedTerm(TermNode[
        leaf_node(curried, 3), var_node(scalar, 1), app_node(fn, 1, 2),
        leaf_node(curried, 3), var_node(scalar, 1), app_node(fn, 4, 5),
        var_node(scalar, 2), app_node(scalar, 6, 7), app_node(scalar, 3, 8),
        abs_node(fn, 9), abs_node(arrow!(table, scalar, fn), 10),
        leaf_node(scalar, 1), app_node(fn, 11, 12),
    ])
    @test validate(repeated, language; context=Int32[scalar]) === true
    repeated_reduced = simplify_term(MersenneTwister(1234), repeated, language;
        context=Int32[scalar], atomic_beta=true)
    @test repeated_reduced isa TypedTerm
    @test repeated_reduced.nodes == TermNode[
        leaf_node(curried, 3), leaf_node(scalar, 1), app_node(fn, 1, 2),
        leaf_node(curried, 3), leaf_node(scalar, 1), app_node(fn, 4, 5),
        var_node(scalar, 1), app_node(scalar, 6, 7), app_node(scalar, 3, 8),
        abs_node(fn, 9),
    ]
    @test validate(repeated_reduced, language; context=Int32[scalar]) === true
    @test length(repeated_reduced.nodes) == length(repeated.nodes) - 3

    # Higher-order variables are admissible atoms and retain their binding.
    higher_order = TypedTerm(TermNode[
        leaf_node(higher, 4), var_node(fn, 0), app_node(fn, 1, 2),
        abs_node(higher, 3), var_node(fn, 0), app_node(fn, 4, 5),
    ])
    @test validate(higher_order, language; context=Int32[fn]) === true
    higher_reduced = simplify_term(MersenneTwister(8000), higher_order, language;
        context=Int32[fn], atomic_beta=true)
    @test higher_reduced isa TypedTerm
    @test higher_reduced.nodes == TermNode[
        leaf_node(higher, 4), var_node(fn, 0), app_node(fn, 1, 2),
    ]
    @test validate(higher_reduced, language; context=Int32[fn]) === true

    # Pre-bound constants and anchored external leaves are ordinary atoms.
    constant = add_leaf!(language, scalar, SourceConstant(:constant))
    external = add_leaf!(language, scalar, SourceExternal(Ref{Nothing}(nothing), 0))
    for argument in (constant, external)
        term = scalar_apply(argument)
        result = simplify_term(MersenneTwister(42), term, language;
            atomic_beta=true)
        @test result isa TypedTerm
        @test result.nodes[2] == leaf_node(scalar, argument)
        @test validate(result, language) === true
    end

    # A registered source program is intentionally not substituted by this
    # opt-in rule; its existing identity rule remains separately valid.
    donor_leaf = register_program!(language, :program_atom,
        AstTree(ConstNode(:primitive)), Dict(:primitive => FloatTy), FloatTy;
        provenance=:atomic_beta_test)
    @test donor_leaf isa Integer
    program_term = scalar_apply(donor_leaf)
    blocked_program = simplify_term(MersenneTwister(555), program_term, language;
        atomic_beta=true)
    @test blocked_program isa Err
    @test blocked_program.value.reason == :no_simplifiable_redex

    # Applications and abstractions are not atomic arguments, and no nested
    # redex is present in either argument here.
    application_argument = TypedTerm(TermNode[
        leaf_node(fn, 2), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), leaf_node(fn, 2), leaf_node(scalar, 1),
        app_node(scalar, 5, 6), app_node(scalar, 4, 7),
    ])
    @test validate(application_argument, language) === true
    blocked_application = simplify_term(MersenneTwister(2000), application_argument,
        language; atomic_beta=true)
    @test blocked_application isa Err
    @test blocked_application.value.reason == :no_simplifiable_redex

    abstraction_argument = TypedTerm(TermNode[
        leaf_node(higher, 4), var_node(fn, 0), app_node(fn, 1, 2),
        abs_node(higher, 3), leaf_node(fn, 2), var_node(scalar, 0),
        app_node(scalar, 5, 6), abs_node(fn, 7), app_node(fn, 4, 8),
    ])
    @test validate(abstraction_argument, language) === true
    blocked_abstraction = simplify_term(MersenneTwister(1), abstraction_argument,
        language; atomic_beta=true)
    @test blocked_abstraction isa Err
    @test blocked_abstraction.value.reason == :no_simplifiable_redex

    # Existing identity/unused rules remain active in opt-in mode, and the
    # ordinary input/depth budgets are still enforced before copying.
    identity = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), leaf_node(scalar, 1),
        app_node(scalar, 2, 3),
    ])
    @test simplify_term(MersenneTwister(1111), identity, language;
        atomic_beta=true).nodes == TermNode[leaf_node(scalar, 1)]
    bounded = simplify_term(MersenneTwister(1111), ordinary, language;
        atomic_beta=true, max_nodes=Int32(length(ordinary.nodes)))
    @test bounded isa TypedTerm
    over_nodes = simplify_term(MersenneTwister(1111), ordinary, language;
        atomic_beta=true, max_nodes=length(ordinary.nodes) - 1)
    @test over_nodes isa Err
    @test over_nodes.value.reason == :node_budget_exceeded
    depth_error = simplify_term(MersenneTwister(1111), ordinary, language;
        atomic_beta=true, max_depth=1)
    @test depth_error isa Err
    @test depth_error.value.reason == :depth_budget_exceeded
end

@testitem "TypedSearch dispatches atomic simplification proposals" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(fn, :f)])
    seed = TypedTerm(TermNode[
        leaf_node(fn, 2), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), leaf_node(scalar, 1), app_node(scalar, 4, 5),
    ])
    result = search_terms([seed], language,
        term -> length(term.nodes) == 3 ? 0.0 : 1.0;
        population_size=1, attempts=2, evaluations=4,
        kinds=(:simplify_atoms,), weights=(1.0,), rng=MersenneTwister(1111))
    @test result isa TypedSearchResult
    @test result.stats.kind_counts[:simplify_atoms] == result.stats.attempts
    @test result.stats.operator_stats[:simplify_atoms].changed == 1
    @test result.stats.operator_stats[:simplify_atoms].failures >= 1
    @test result.stats.acceptance_rejected == 0
    @test all(member -> validate(member.term, language) === true, result.population)
end
