"""Bounded shared root interfaces for typed λ islands.

`prepare_root_hole` gives a source program a task-root interface with exactly
`arity` outer λ binders.  Missing binders are ordinary η expansions in the
compact representation.  `import_root_body` can then project another source
with the same complete root type into that interface, preserving root-input
positions with an existing `SourceHole.anchor`.

This boundary assumes that callers have already established a common
`Language` and common constant semantics.  In particular, a same-typed
constant from a different task must not be inserted into the same language.
No general context matching, opaque expansion, β-normalisation, or evaluator
is performed here.  η preparation is boundary setup rather than a hot loop;
the simple implementation can cost `O(arity * (n + arity))` in the worst case
for an `n`-node term.  Its observation-preservation claim is only for fully
applied task inputs, not arbitrary partial values or execution-step budgets.
"""

export prepare_root_hole, import_root_body, RootInterfaceError

"""An expected refusal while preparing a bounded shared root interface."""
struct RootInterfaceError
    reason::Symbol
    detail::Any
end

RootInterfaceError(reason::Symbol) = RootInterfaceError(reason, nothing)

@inline _root_fail(reason::Symbol, detail=nothing) = Err(RootInterfaceError(reason, detail))

function _root_plain_copy!(out::Vector{TermNode}, nodes::Vector{TermNode}, i::Int)
    1 <= i <= length(nodes) || return _root_fail(:bad_node_index, i)
    node = nodes[i]
    if node.kind == VAR || node.kind == LEAF
        push!(out, node)
    elseif node.kind == ABS
        body = _root_plain_copy!(out, nodes, Int(node.x))
        body isa Err && return body
        push!(out, TermNode(ABS, node.typeid, body, 0))
    elseif node.kind == APP
        fn = _root_plain_copy!(out, nodes, Int(node.x))
        fn isa Err && return fn
        arg = _root_plain_copy!(out, nodes, Int(node.y))
        arg isa Err && return arg
        push!(out, TermNode(APP, node.typeid, fn, arg))
    else
        return _root_fail(:unknown_node_kind, (i, node.kind))
    end
    return Int32(length(out))
end

"""Copy one subtree below one newly introduced binder.

`depth` counts binders internal to the copied subtree.  Only references that
escape that subtree are shifted; references to its own binders stay local.
"""
function _root_shift_copy!(out::Vector{TermNode}, nodes::Vector{TermNode}, i::Int, depth::Int)
    1 <= i <= length(nodes) || return _root_fail(:bad_node_index, i)
    node = nodes[i]
    if node.kind == VAR
        db = Int(node.x)
        shifted = db < depth ? db : db + 1
        push!(out, TermNode(VAR, node.typeid, shifted, 0))
    elseif node.kind == LEAF
        push!(out, node)
    elseif node.kind == ABS
        body = _root_shift_copy!(out, nodes, Int(node.x), depth + 1)
        body isa Err && return body
        push!(out, TermNode(ABS, node.typeid, body, 0))
    elseif node.kind == APP
        fn = _root_shift_copy!(out, nodes, Int(node.x), depth)
        fn isa Err && return fn
        arg = _root_shift_copy!(out, nodes, Int(node.y), depth)
        arg isa Err && return arg
        push!(out, TermNode(APP, node.typeid, fn, arg))
    else
        return _root_fail(:unknown_node_kind, (i, node.kind))
    end
    return Int32(length(out))
end

"""Build a compact subtree with exactly `remaining` λ-prefix levels."""
function _root_eta_segment(nodes::Vector{TermNode}, i::Int, remaining::Int,
                           table::TypeTable)
    remaining >= 0 || return _root_fail(:invalid_arity, remaining)
    if remaining == 0
        out = TermNode[]
        root = _root_plain_copy!(out, nodes, i)
        root isa Err && return root
        return TypedTerm(out)
    end
    1 <= i <= length(nodes) || return _root_fail(:bad_node_index, i)
    node = nodes[i]
    if node.kind == ABS
        parts = _arrow_parts(table, node.typeid)
        parts === nothing && return _root_fail(:abs_not_function_type, node.typeid)
        body = _root_eta_segment(nodes, Int(node.x), remaining - 1, table)
        body isa Err && return body
        out = copy(body.nodes)
        push!(out, TermNode(ABS, node.typeid, Int32(length(out)), 0))
        return TypedTerm(out)
    end

    parts = _arrow_parts(table, node.typeid)
    parts === nothing && return _root_fail(:eta_requires_function_type, (i, node.typeid))
    argid, resid = parts
    out = TermNode[]
    p = _root_shift_copy!(out, nodes, i, 0)
    p isa Err && return p
    variable = Int32(length(out) + 1)
    push!(out, TermNode(VAR, argid, 0, 0))
    push!(out, TermNode(APP, resid, p, variable))
    bodyroot = Int32(length(out))

    # Further η levels apply to the newly formed application body.  Build a
    # fresh segment so stale pre-rewrite nodes never become a disconnected
    # prefix in the strict post-order result.
    if remaining > 1
        body = _root_eta_segment(out, Int(bodyroot), remaining - 1, table)
        body isa Err && return body
        out = copy(body.nodes)
        bodyroot = Int32(length(out))
    end
    push!(out, TermNode(ABS, node.typeid, bodyroot, 0))
    return TypedTerm(out)
end

function _root_external_ids(hole::SourceHole, language::Language)
    ids = Int32[]
    sizehint!(ids, length(hole.external_types))
    for ty in hole.external_types
        ty isa TypeSystem.Ty || return _root_fail(:nontransparent_root_input, ty)
        id = intern_type!(language.types, ty)
        id isa Err && return id
        push!(ids, id)
    end
    return ids
end

"""Register every root input by position, including unused inputs.

Root interfaces are stricter than local holes: every input must be a
transparent monotype.  Local registration may skip a scheme, unresolved
type, or μ type while retaining observed concrete instances, but root
interfaces reject such a partial coverage report.
"""
function _register_root_inputs!(hole::SourceHole, language::Language)
    initial_leaves = length(language.leaves)
    report = register_external_bindings!(hole, language)
    if !isempty(report.skipped)
        resize!(language.leaves, initial_leaves)
        return _root_fail(:nontransparent_root_input, report.skipped)
    end
    return true
end

function _same_root_type(source::SourceAnalysis, hole::SourceHole, language::Language)
    left = intern_type!(language.types, source.root_type)
    left isa Err && return left
    right = intern_type!(language.types, hole.source.root_type)
    right isa Err && return right
    left == right || return _root_fail(:wrong_root_type, (left, right))
    return true
end

function _root_path(arity::Int)
    return fill(1, arity)
end

"""Prepare one source as a common task-root λ interface.

The returned `term` is the body below exactly `arity` outer binders.  Root
inputs are represented as fixed `SourceExternal(anchor, db)` leaves, where
`db` is the original De Bruijn position (zero is the nearest/root-last input).
The source is exported and inferred once after η completion so the returned
`hole` is evidence from the actual fixed source boundary.  The inferred source
at that boundary may contain an expanded donor β-redex and therefore need not
be independently accepted by `import_hole`; `term` is the compact body that
retains the already certified `SourceProgram` leaf instead.
"""
function prepare_root_hole(source::SourceAnalysis, language::Language, type_context;
                           arity::Int)
    arity >= 0 || return _root_fail(:invalid_arity, arity)
    root_hole = source_hole(source)
    root_hole isa Err && return root_hole
    imported = import_hole(root_hole, language)
    imported isa Err && return imported
    eta = _root_eta_segment(imported.nodes, length(imported.nodes), arity, language.types)
    eta isa Err && return eta
    exported = export_hole(eta, language, root_hole)
    exported isa Err && return exported
    rebuilt = prepare_source(exported, type_context, source.root_type)
    rebuilt isa Err && return rebuilt
    body_hole = source_hole(rebuilt, _root_path(arity))
    body_hole isa Err && return body_hole
    # `body_hole` is deliberately retained as evidence from the expanded and
    # inferred source.  Re-importing it would project donor-created β-redexes
    # as if they were an ordinary transparent occurrence and reject a valid
    # prepared source.  Strip the η prefix from the certified compact term
    # instead, rebasing only the root inputs to this actual hole anchor.
    body = _strip_root_to_anchor(eta, arity, body_hole, language)
    body isa Err && return body
    return (; hole=body_hole, term=body)
end

function _strip_root_to_anchor(term::TypedTerm, arity::Int, hole::SourceHole,
                                language::Language)
    ids = _root_external_ids(hole, language)
    ids isa Err && return ids
    length(ids) == arity || return _root_fail(:root_input_arity_mismatch, (arity, length(ids)))
    registered = _register_root_inputs!(hole, language)
    registered isa Err && return registered
    current = length(term.nodes)
    for _ in 1:arity
        1 <= current <= length(term.nodes) || return _root_fail(:bad_eta_prefix, current)
        node = term.nodes[current]
        node.kind == ABS || return _root_fail(:missing_root_lambda, current)
        current = Int(node.x)
    end
    expected = intern_type!(language.types,
        hole.source.metadata[hole.occurrence][1])
    expected isa Err && return expected
    term.nodes[current].typeid == expected || return _root_fail(
        :root_body_type_mismatch, (term.nodes[current].typeid, expected),
    )

    out = TermNode[]
    function copy_body(i::Int, depth::Int)
        1 <= i <= length(term.nodes) || return _root_fail(:bad_body_index, i)
        node = term.nodes[i]
        if node.kind == VAR
            db = Int(node.x)
            if db < depth
                push!(out, node)
            else
                rem = db - depth
                rem < length(ids) || return _root_fail(:unbound_root_input, (i, rem))
                expected = ids[length(ids) - rem]
                node.typeid == expected ||
                    return _root_fail(:root_input_type_mismatch, (i, node.typeid, expected))
                leaf = _source_leaf!(language, node.typeid, SourceExternal(hole.anchor, rem))
                push!(out, TermNode(LEAF, node.typeid, leaf, 0))
            end
        elseif node.kind == LEAF
            push!(out, node)
        elseif node.kind == ABS
            body = copy_body(Int(node.x), depth + 1)
            body isa Err && return body
            push!(out, TermNode(ABS, node.typeid, body, 0))
        elseif node.kind == APP
            fn = copy_body(Int(node.x), depth)
            fn isa Err && return fn
            arg = copy_body(Int(node.y), depth)
            arg isa Err && return arg
            push!(out, TermNode(APP, node.typeid, fn, arg))
        else
            return _root_fail(:unknown_node_kind, (i, node.kind))
        end
        return Int32(length(out))
    end
    root = copy_body(current, 0)
    root isa Err && return root
    result = TypedTerm(out)
    valid = validate(result, language)
    valid === true || return valid isa Err ? valid : _root_fail(:invalid_shared_body)
    return result
end

"""Import another complete root into an existing shared root interface.

`hole` must be the body hole returned by `prepare_root_hole`.  The complete
root types must match exactly.  Root-input leaves use the existing anchor and
their original positional De Bruijn index, so same-typed inputs are never
merged merely because their type ids agree.
"""
function import_root_body(source::SourceAnalysis, language::Language, hole::SourceHole)
    arity = length(hole.path)
    all(edge -> edge == 1, hole.path) || return _root_fail(:not_a_root_interface, hole.path)
    same = _same_root_type(source, hole, language)
    same isa Err && return same
    imported_hole = source_hole(source)
    imported_hole isa Err && return imported_hole
    imported = import_hole(imported_hole, language)
    imported isa Err && return imported
    eta = _root_eta_segment(imported.nodes, length(imported.nodes), arity, language.types)
    eta isa Err && return eta
    return _strip_root_to_anchor(eta, arity, hole, language)
end

@testitem "TypedSearch shared root η interface" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Procedural: evaluate_procedural

    float_to_float = TyFun(FloatTy, FloatTy)
    target = TyFun(float_to_float, float_to_float)
    source = prepare_source(AstTree(AbsNode(:f, IndexNode(0))),
        Dict{Symbol,TyLike}(), target)
    language = Language(TypeTable())
    prepared = prepare_root_hole(source, language, Dict{Symbol,TyLike}(); arity=2)
    @test prepared isa NamedTuple
    @test length(prepared.hole.path) == 2
    @test validate(prepared.term, language) === true
    @test prepared.term.nodes[end].kind != ABS
    exported = export_hole(prepared.term, language, prepared.hole)
    @test prepare_source(exported, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
end

@testitem "TypedSearch root body preserves positions and internal binders" begin
    using Test
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Procedural: evaluate_procedural

    target = TyFun(IntTy, TyFun(IntTy, TyFun(IntTy, IntTy)))
    source = prepare_source(AstTree(AbsNode(:x, AbsNode(:y, AbsNode(:z, IndexNode(1))))),
        Dict{Symbol,TyLike}(), target)
    language = Language(TypeTable())
    prepared = prepare_root_hole(source, language, Dict{Symbol,TyLike}(); arity=2)
    @test prepared isa NamedTuple
    @test validate(prepared.term, language) === true
    @test prepared.term.nodes[end].kind == ABS
    roundtrip = export_hole(prepared.term, language, prepared.hole)
    @test prepare_source(roundtrip, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
    @test evaluate_procedural(roundtrip, Dict{Symbol,Any}(); args=(1, 2, 3)) == 2
end

@testitem "TypedSearch shared root imports keep same-typed input identities" begin
    using Test
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs

    target = TyFun(IntTy, TyFun(IntTy, IntTy))
    first_source = prepare_source(AstTree(AbsNode(:a, AbsNode(:b, IndexNode(1)))),
        Dict{Symbol,TyLike}(), target)
    second_source = prepare_source(AstTree(AbsNode(:u, AbsNode(:v, IndexNode(0)))),
        Dict{Symbol,TyLike}(), target)
    language = Language(TypeTable())
    first = prepare_root_hole(first_source, language, Dict{Symbol,TyLike}(); arity=2)
    second_body = import_root_body(second_source, language, first.hole)
    @test validate(second_body, language) === true
    payloads = [leaf.payload for leaf in language.leaves
        if leaf.payload isa LambdaRegression.TypedSearch.SourceExternal]
    @test sort(unique(p.index for p in payloads)) == [0, 1]
    @test length(payloads) >= 2
end

@testitem "TypedSearch root registers unused positional inputs" begin
    using Test
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Procedural: evaluate_procedural

    target = TyFun(IntTy, TyFun(IntTy, IntTy))
    # The second root input is intentionally ignored by the ancestor.
    source = prepare_source(AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1)))),
        Dict{Symbol,TyLike}(), target)
    language = Language(TypeTable())
    prepared = prepare_root_hole(source, language, Dict{Symbol,TyLike}(); arity=2)
    @test prepared isa NamedTuple
    input_payloads = [leaf.payload for leaf in language.leaves
        if leaf.payload isa LambdaRegression.TypedSearch.SourceExternal &&
        leaf.payload.anchor === prepared.hole.anchor]
    @test sort(unique(p.index for p in input_payloads)) == [0, 1]
    used = [leaf.payload.index for leaf in language.leaves
        if leaf.payload isa LambdaRegression.TypedSearch.SourceExternal &&
        leaf.payload.anchor === prepared.hole.anchor]
    @test 1 in used

    # The ignored second input is a real candidate leaf, not just metadata.
    intid = intern_type!(language.types, IntTy)
    unused_leaf = findfirst(i -> language.leaves[i].payload isa
        LambdaRegression.TypedSearch.SourceExternal &&
        language.leaves[i].payload.anchor === prepared.hole.anchor &&
        language.leaves[i].payload.index == 0, eachindex(language.leaves))
    @test unused_leaf !== nothing
    candidate = TypedTerm(TermNode[leaf_node(intid, unused_leaf)])
    @test validate(candidate, language) === true
    exported = export_hole(candidate, language, prepared.hole)
    @test prepare_source(exported, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
    @test evaluate_procedural(exported, Dict{Symbol,Any}(); args=(1, 2)) == 2
end

@testitem "TypedSearch root interface rejects bad arity and task type" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun, TyLike
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs

    float_target = TyFun(FloatTy, FloatTy)
    source = prepare_source(AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(), float_target)
    language = Language(TypeTable())
    @test prepare_root_hole(source, language, Dict{Symbol,TyLike}(); arity=2) isa Err
    first = prepare_root_hole(source, language, Dict{Symbol,TyLike}(); arity=1)
    bool_source = prepare_source(AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(), TyFun(BoolTy, BoolTy))
    @test import_root_body(bool_source, language, first.hole) isa Err
end

@testitem "TypedSearch root donor expansion keeps the inferred body boundary" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceProgram
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Procedural: evaluate_procedural
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    int_to_int = TyFun(IntTy, IntTy)
    higher = TyFun(int_to_int, int_to_int)
    language = Language(TypeTable())
    donor_leaf = register_program!(language, :hid,
        AstTree(AbsNode(:g, IndexNode(0))), Dict{Symbol,TyLike}(), higher)
    @test donor_leaf isa Integer
    donor = language.leaves[donor_leaf].payload
    @test donor isa SourceProgram
    @test donor.expanded_size == 2

    # The source has one explicit λ prefix but the task-root interface asks
    # for two.  η completion then places a higher-order donor call under a
    # captured root input.  Exporting the donor creates a β-redex in the
    # inferred source body; the body boundary must retain the typed evidence
    # from that source instead of projecting the redex a second time.
    source = prepare_source(
        AstTree(AbsNode(:f, AppNode(ConstNode(:hid), IndexNode(0)))),
        Dict{Symbol,TyLike}(:hid => higher), higher)
    prepared = prepare_root_hole(source, language,
        Dict{Symbol,TyLike}(); arity=2)
    @test prepared isa NamedTuple
    if prepared isa NamedTuple
        @test prepared.hole.path == [1, 1]
        expected = intern!(language.types, IntTy)
        @test prepared.term.nodes[end].typeid == expected
        @test validate(prepared.term, language) === true
        @test written_complexity(prepared.term, language) == 6
        exported = export_hole(prepared.term, language, prepared.hole)
        @test prepare_source(exported, Dict{Symbol,TyLike}(), higher) isa SourceAnalysis
        @test evaluate_procedural(exported, Dict{Symbol,Any}();
            args=(x -> x + 1, 7)) == 8
    end

    # Same-typed root inputs and an internal binder remain positional and
    # capture-safe when no donor is present.
    positional_target = TyFun(IntTy, TyFun(IntTy, TyFun(IntTy, IntTy)))
    positional_source = prepare_source(
        AstTree(AbsNode(:x, AbsNode(:y, AbsNode(:z, IndexNode(2))))),
        Dict{Symbol,TyLike}(), positional_target)
    positional_language = Language(TypeTable())
    positional = prepare_root_hole(positional_source, positional_language,
        Dict{Symbol,TyLike}(); arity=2)
    @test positional isa NamedTuple
    if positional isa NamedTuple
        positional_export = export_hole(positional.term, positional_language,
            positional.hole)
        @test prepare_source(positional_export, Dict{Symbol,TyLike}(),
            positional_target) isa SourceAnalysis
        @test evaluate_procedural(positional_export, Dict{Symbol,Any}();
            args=(4, 5, 6)) == 4
    end

    unsupported = prepare_source(
        AstTree(AbsNode(:x, AppNode(AbsNode(:z, IndexNode(0)), IndexNode(0)))),
        Dict{Symbol,TyLike}(), int_to_int)
    rejected = prepare_root_hole(unsupported, Language(TypeTable()),
        Dict{Symbol,TyLike}(); arity=1)
    @test rejected isa Err
    @test rejected.value.reason == :unsupported_region
end
