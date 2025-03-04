"""Bounded η expansion and contraction for compact typed λ terms.

`eta_move` is a normal, scored proposal.  It chooses one eligible occurrence
from the requested direction and rebuilds the compact tree around it.  For a
function occurrence `t : a → b`, expansion constructs
`λu : a. (t ↑ 1) u`; contraction accepts only the explicit shape
`λu. v u` when `u` has no free occurrence in `v`.  The shift and drop walks
are lexical (De Bruijn) operations, so outer references are adjusted while
nested binders and their shadowing remain unchanged.  No β-reduction, CSE, or
semantic-equivalence claim is made here; callers must re-score the result.
"""

using Random: AbstractRNG, rand

export eta_move, eta_move_at

struct _EtaCandidate
    site::Int
    direction::Symbol
end

@inline _eta_fail(reason::Symbol, detail=nothing) =
    Err(TypedSearchError(reason, detail))

@inline function _eta_direction_ok(direction::Symbol)
    return direction === :either || direction === :expand || direction === :shrink
end

"""Return whether a subtree contains the binder at its current depth."""
function _eta_binder_used(nodes::Vector{TermNode}, root::Int, depth::Int=0)
    node = nodes[root]
    if node.kind == VAR_KIND
        return Int(node.x) == depth
    elseif node.kind == ABS_KIND
        return _eta_binder_used(nodes, Int(node.x), depth + 1)
    elseif node.kind == APP_KIND
        return _eta_binder_used(nodes, Int(node.x), depth) ||
            _eta_binder_used(nodes, Int(node.y), depth)
    end
    return false
end

"""Copy a subtree while shifting references to its enclosing context by `shift`.

`local_depth` counts binders introduced within the copied subtree.  References
    below that depth are local and unchanged; all other references belong to
    the enclosing context and receive `shift`.
"""
function _eta_copy_shift!(out::Vector{TermNode}, nodes::Vector{TermNode}, root::Int,
                          shift::Int, local_depth::Int=0)
    node = nodes[root]
    if node.kind == VAR_KIND
        db = Int(node.x)
        newdb = db < local_depth ? db : db + shift
        newdb >= 0 || return _eta_fail(:invalid_binder_shift, (root, db, shift))
        push!(out, var_node(node.typeid, newdb))
    elseif node.kind == LEAF_KIND
        push!(out, leaf_node(node.typeid, node.x))
    elseif node.kind == ABS_KIND
        body = _eta_copy_shift!(out, nodes, Int(node.x), shift, local_depth + 1)
        body isa Err && return body
        push!(out, abs_node(node.typeid, body))
    elseif node.kind == APP_KIND
        fn = _eta_copy_shift!(out, nodes, Int(node.x), shift, local_depth)
        fn isa Err && return fn
        arg = _eta_copy_shift!(out, nodes, Int(node.y), shift, local_depth)
        arg isa Err && return arg
        push!(out, app_node(node.typeid, fn, arg))
    else
        return _eta_fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

"""Copy a subtree out of one removed binder, lowering enclosing references.

The removed binder is at `local_depth == 0` at the call site.  Encountering a
reference exactly at that depth is a failed contraction; callers check this
condition before starting the walk, but retaining the error here makes the
helper capture-safe if reused.
"""
function _eta_copy_drop!(out::Vector{TermNode}, nodes::Vector{TermNode}, root::Int,
                         local_depth::Int=0)
    node = nodes[root]
    if node.kind == VAR_KIND
        db = Int(node.x)
        if db < local_depth
            newdb = db
        elseif db == local_depth
            return _eta_fail(:eta_binder_used, root)
        else
            newdb = db - 1
        end
        newdb >= 0 || return _eta_fail(:invalid_binder_drop, (root, db))
        push!(out, var_node(node.typeid, newdb))
    elseif node.kind == LEAF_KIND
        push!(out, leaf_node(node.typeid, node.x))
    elseif node.kind == ABS_KIND
        body = _eta_copy_drop!(out, nodes, Int(node.x), local_depth + 1)
        body isa Err && return body
        push!(out, abs_node(node.typeid, body))
    elseif node.kind == APP_KIND
        fn = _eta_copy_drop!(out, nodes, Int(node.x), local_depth)
        fn isa Err && return fn
        arg = _eta_copy_drop!(out, nodes, Int(node.y), local_depth)
        arg isa Err && return arg
        push!(out, app_node(node.typeid, fn, arg))
    else
        return _eta_fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

"""Return the `v` root for a shrinkable `ABS` occurrence, or `nothing`.

The supplied term has already passed `validate`; this predicate therefore only
checks the syntactic η shape and the lexical side condition on `v`.
"""
function _eta_shrink_body(nodes::Vector{TermNode}, root::Int)
    binder = nodes[root]
    binder.kind == ABS_KIND || return nothing
    body = nodes[Int(binder.x)]
    body.kind == APP_KIND || return nothing
    arg = nodes[Int(body.y)]
    arg.kind == VAR_KIND && arg.x == 0 || return nothing
    vroot = Int(body.x)
    _eta_binder_used(nodes, vroot) && return nothing
    return vroot
end

function _eta_collect(term::TypedTerm, language::Language, direction::Symbol)
    candidates = _EtaCandidate[]
    table = language.types
    for site in eachindex(term.nodes)
        node = term.nodes[site]
        if direction === :either || direction === :expand
            _arrow_parts(table, node.typeid) === nothing ||
                push!(candidates, _EtaCandidate(site, :expand))
        end
        if direction === :either || direction === :shrink
            _eta_shrink_body(term.nodes, site) === nothing ||
                push!(candidates, _EtaCandidate(site, :shrink))
        end
    end
    return candidates
end

"""Rebuild `parent` while replacing exactly `site` by one η move."""
function _eta_replace!(out::Vector{TermNode}, nodes::Vector{TermNode}, root::Int,
                        site::Int, direction::Symbol, language::Language)
    if root == site
        if direction === :expand
            original = nodes[root]
            parts = _arrow_parts(language.types, original.typeid)
            parts === nothing && return _eta_fail(:eta_not_function, original.typeid)
            argid, resid = parts
            shifted = _eta_copy_shift!(out, nodes, root, 1, 0)
            shifted isa Err && return shifted
            argument = push_var!(out, argid, 0)
            application = push_app!(out, resid, shifted, argument)
            push!(out, abs_node(original.typeid, application))
            return Int32(length(out))
        end

        vroot = _eta_shrink_body(nodes, root)
        vroot === nothing && return _eta_fail(:eta_not_shrinkable, root)
        dropped = _eta_copy_drop!(out, nodes, vroot, 0)
        dropped isa Err && return dropped
        return dropped
    end

    node = nodes[root]
    if node.kind == VAR_KIND
        push!(out, var_node(node.typeid, node.x))
    elseif node.kind == LEAF_KIND
        push!(out, leaf_node(node.typeid, node.x))
    elseif node.kind == ABS_KIND
        body = _eta_replace!(out, nodes, Int(node.x), site, direction, language)
        body isa Err && return body
        push!(out, abs_node(node.typeid, body))
    elseif node.kind == APP_KIND
        fn = _eta_replace!(out, nodes, Int(node.x), site, direction, language)
        fn isa Err && return fn
        arg = _eta_replace!(out, nodes, Int(node.y), site, direction, language)
        arg isa Err && return arg
        push!(out, app_node(node.typeid, fn, arg))
    else
        return _eta_fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

function _eta_depth(nodes::Vector{TermNode})
    depths = zeros(Int, length(nodes))
    for (i, node) in enumerate(nodes)
        depths[i] = if node.kind == ABS_KIND
            1 + depths[Int(node.x)]
        elseif node.kind == APP_KIND
            1 + max(depths[Int(node.x)], depths[Int(node.y)])
        else
            0
        end
    end
    return depths[end]
end

function _eta_apply(term::TypedTerm, site::Int, direction::Symbol, language::Language;
                    context::AbstractVector{<:Integer}=Int32[],
                    max_nodes::Integer=128, max_depth::Integer=32)
    max_nodes >= 1 || return _eta_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _eta_fail(:invalid_depth_budget, max_depth)
    1 <= site <= length(term.nodes) || return _eta_fail(:bad_site, site)
    length(term.nodes) <= max_nodes ||
        return _eta_fail(:node_budget_exceeded, length(term.nodes))
    checked = validate(term, language; context=context)
    checked === true || return checked
    _eta_depth(term.nodes) <= max_depth ||
        return _eta_fail(:depth_budget_exceeded, _eta_depth(term.nodes))

    out = TermNode[]
    # A contraction can remove a whole subtree; expansion is bounded below by
    # the input plus two nodes.  The exact size is checked after rebuilding.
    sizehint!(out, max(1, length(term.nodes) + (direction === :expand ? 2 : -1)))
    copied = _eta_replace!(out, term.nodes, length(term.nodes), site, direction, language)
    copied isa Err && return copied
    result = TypedTerm(out)
    length(result.nodes) <= max_nodes ||
        return _eta_fail(:node_budget_exceeded, length(result.nodes))
    result_depth = _eta_depth(result.nodes)
    result_depth <= max_depth ||
        return _eta_fail(:depth_budget_exceeded, result_depth)
    checked = validate(result, language; context=context)
    checked === true || return checked
    return result
end

"""
    eta_move_at(term, site, language; context=Int32[], direction=:either,
                max_nodes=128, max_depth=32)

Apply a deterministic η expansion or contraction at one compact-tree
occurrence.  `direction=:either` tries expansion first and then contraction at
the requested site.  If the selected direction is not legal, an explicit
`Err(TypedSearchError(:eta_not_shrinkable, ...))` (or
`:eta_not_function`) is returned; no random state is consumed.
"""
function eta_move_at(term::TypedTerm, site::Integer, language::Language;
                     context::AbstractVector{<:Integer}=Int32[],
                     direction::Symbol=:either,
                     max_nodes::Integer=128, max_depth::Integer=32)
    _eta_direction_ok(direction) || return _eta_fail(:unknown_eta_direction, direction)
    site_int = Int(site)
    1 <= site_int <= length(term.nodes) || return _eta_fail(:bad_site, site)
    checked = validate(term, language; context=context)
    checked === true || return checked
    node = term.nodes[site_int]
    if direction === :either
        if _arrow_parts(language.types, node.typeid) !== nothing
            return _eta_apply(term, site_int, :expand, language;
                context=context, max_nodes=max_nodes, max_depth=max_depth)
        end
        _eta_shrink_body(term.nodes, site_int) !== nothing ||
            return _eta_fail(:eta_not_shrinkable, site)
        return _eta_apply(term, site_int, :shrink, language;
            context=context, max_nodes=max_nodes, max_depth=max_depth)
    elseif direction === :expand
        _arrow_parts(language.types, node.typeid) === nothing &&
            return _eta_fail(:eta_not_function, node.typeid)
    else
        _eta_shrink_body(term.nodes, site_int) === nothing &&
            return _eta_fail(:eta_not_shrinkable, site)
    end
    return _eta_apply(term, site_int, direction, language;
        context=context, max_nodes=max_nodes, max_depth=max_depth)
end

"""
    eta_move(rng, term, language; context=Int32[], direction=:either,
             max_nodes=128, max_depth=32)

Select one eligible occurrence uniformly and apply one bounded η move.  The
input is validated before scanning; a term with no candidate returns
`Err(TypedSearchError(:no_eta_candidate))` without consuming `rng`.
"""
function eta_move(rng::AbstractRNG, term::TypedTerm, language::Language;
                  context::AbstractVector{<:Integer}=Int32[],
                  direction::Symbol=:either,
                  max_nodes::Integer=128, max_depth::Integer=32)
    _eta_direction_ok(direction) || return _eta_fail(:unknown_eta_direction, direction)
    max_nodes >= 1 || return _eta_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _eta_fail(:invalid_depth_budget, max_depth)
    length(term.nodes) <= max_nodes ||
        return _eta_fail(:node_budget_exceeded, length(term.nodes))
    checked = validate(term, language; context=context)
    checked === true || return checked
    depth = _eta_depth(term.nodes)
    depth <= max_depth || return _eta_fail(:depth_budget_exceeded, depth)
    candidates = _eta_collect(term, language, direction)
    isempty(candidates) && return _eta_fail(:no_eta_candidate)
    selected = candidates[rand(rng, 1:length(candidates))]
    return _eta_apply(term, selected.site, selected.direction, language;
        context=context, max_nodes=max_nodes, max_depth=max_depth)
end

@testitem "TypedSearch η expansion and contraction preserve compact types" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy, TyFun
    using LambdaRegression.TypedSearch
    using LambdaRegression.Utils: Err

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    higher = arrow!(table, scalar, fn)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(fn, :f),
                                LeafSpec(higher, :h)])
    identity = TypedTerm(TermNode[var_node(scalar, 0), abs_node(fn, 1)])

    expanded = eta_move_at(identity, 2, language; direction=:expand)
    @test expanded isa TypedTerm
    @test expanded.nodes == TermNode[
        var_node(scalar, 0), abs_node(fn, 1), var_node(scalar, 0),
        app_node(scalar, 2, 3), abs_node(fn, 4),
    ]
    @test validate(expanded, language) === true
    @test eta_move_at(expanded, 5, language; direction=:shrink).nodes == identity.nodes

    failed = eta_move_at(TypedTerm(TermNode[
        leaf_node(higher, 3), var_node(scalar, 0), app_node(fn, 1, 2),
        var_node(scalar, 0), app_node(scalar, 3, 4), abs_node(fn, 5),
    ]), 6, language; direction=:shrink)
    @test failed isa Err
    @test failed.value.reason == :eta_binder_used || failed.value.reason == :eta_not_shrinkable

    @test eta_move(MersenneTwister(1), identity, language; direction=:expand,
        max_nodes=4) isa Err
    @test eta_move(MersenneTwister(1), identity, language; direction=:expand,
        max_depth=2) isa Err
end

@testitem "TypedSearch η moves shift outer references and shadowed binders" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(fn, :outer)])
    # Context[1] is an outer function.  Expanding it under a fresh binder must
    # shift that reference from db=0 to db=1.
    open_term = TypedTerm(TermNode[var_node(fn, 0)])
    expanded = eta_move_at(open_term, 1, language; context=Int32[fn], direction=:expand)
    @test expanded.nodes == TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2), abs_node(fn, 3),
    ]
    @test validate(expanded, language; context=Int32[fn]) === true

    # Contraction out of the same binder lowers the outer reference back to
    # db=0 instead of capturing it or leaving a stale db=1.
    open_eta = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3),
    ])
    contracted = eta_move_at(open_eta, 4, language;
        context=Int32[fn], direction=:shrink)
    @test contracted.nodes == TermNode[var_node(fn, 0)]
    @test validate(contracted, language; context=Int32[fn]) === true

    # A nested shadowed binder remains db=0 when the outer function is shifted.
    nested_type = arrow!(table, scalar, fn)
    nested = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), abs_node(nested_type, 2),
    ])
    @test validate(nested, language) === true
    shifted = eta_move_at(nested, 3, language; direction=:expand)
    @test validate(shifted, language) === true
end

@testitem "TypedSearch η supports higher-order functions and bounded μ refusal" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyMu, newtyvar
    using LambdaRegression.TypedSearch
    using LambdaRegression.Utils: Err

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    higher = arrow!(table, scalar, arrow!(table, scalar, scalar))
    language = Language(table, [LeafSpec(higher, :higher)])
    term = TypedTerm(TermNode[leaf_node(higher, 1)])
    expanded = eta_move_at(term, 1, language; direction=:expand)
    @test expanded isa TypedTerm
    @test validate(expanded, language) === true
    @test expanded.nodes[end].typeid == higher

    α = newtyvar(0)
    μ = TyMu(α, TyFun(α, FloatTy))
    @test intern_type!(TypeTable(), μ) isa Err
end

@testitem "TypedSearch η native import/export roundtrip" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.Procedural: evaluate_procedural

    target = LambdaRegression.TypeSystem.TyFun(FloatTy, FloatTy)
    source = prepare_source(AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(), target)
    @test source isa SourceAnalysis
    hole = source_hole(source)
    language = Language(TypeTable())
    imported = import_hole(hole, language)
    expanded = eta_move_at(imported, length(imported.nodes), language; direction=:expand)
    @test expanded isa TypedTerm
    exported = export_hole(expanded, language, hole)
    @test prepare_source(exported, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
    @test evaluate_procedural(exported, Dict{Symbol,Any}(); args=(2.0,)) == 2.0
end
