using ..Trees: setchild!

export register_external_bindings!

"""A fixed source constant; its runtime meaning belongs to the island's context."""
struct SourceConstant
    name::Symbol
end

"""An external binding at one fixed ancestor hole, not a transported closure."""
struct SourceExternal
    anchor::Base.RefValue{Nothing}
    index::Int
end

function _source_leaf!(language, typeid, payload)
    # This is boundary setup, not donor retrieval. Committed instances of the
    # same polymorphic source remain different typed leaves.
    for (i, leaf) in enumerate(language.leaves)
        if payload isa SourceConstant && leaf.payload isa SourceProgram &&
                leaf.payload.name == payload.name
            # A registered program owns this global name at its committed
            # instance.  Do not silently create a cheap SourceConstant when a
            # source import asks for an incompatible instance.
            leaf.typeid == typeid && return Int32(i)
            return Err(SourceBoundaryError(:program_type_mismatch, 0))
        end
        leaf.typeid == typeid && isequal(leaf.payload, payload) && return Int32(i)
    end
    push!(language.leaves, LeafSpec(typeid, payload))
    return Int32(length(language.leaves))
end

"""Classify an external type that cannot be a fixed search leaf."""
function _external_skip_reason(ty)
    ty isa TypeSystem.TyScheme && return :polymorphic_scheme
    ty isa TypeSystem.TyVar && ty.instance === nothing && return :unresolved_type
    ty isa TypeSystem.TyMu && return :recursive_type
    return :unsupported_type
end

"""
    register_external_bindings!(hole, language)

Register every admitted monomorphic external binding for `hole`, including
bindings not referenced by its selected source occurrence.  The returned
report has `admitted`, `skipped`, and `available_before` fields.  `skipped`
contains zero-based source De Bruijn indices and reasons for non-admitted
schemes or types; `available_before` records `(index, typeid)` pairs already
present for this anchor before this registration call.  In particular,
skipping a `TyScheme` does not discard an already observed monomorphic use of
that scheme, and a later call does not mislabel a leaf created by an earlier
registration as source evidence.

External leaves are anchored to this exact hole.  A language must not combine
local holes with different anchors when those leaves are intended to be
reused: one language has one fixed meaning per local anchor.
"""
function register_external_bindings!(hole::SourceHole, language::Language)
    external_types = hole.external_types
    available_before = NamedTuple{(:index, :typeid),Tuple{Int,Int32}}[]
    for leaf in language.leaves
        payload = leaf.payload
        payload isa SourceExternal && payload.anchor === hole.anchor &&
            0 <= payload.index < length(external_types) &&
            push!(available_before, (index=payload.index, typeid=leaf.typeid))
    end
    sort!(available_before; by=x -> (x.index, x.typeid))
    skipped = NamedTuple{(:index, :reason),Tuple{Int,Symbol}}[]
    admitted = 0
    for position in reverse(eachindex(external_types))
        db = length(external_types) - position
        ty = external_types[position]
        if !(ty isa TypeSystem.Ty)
            push!(skipped, (index=db, reason=_external_skip_reason(ty)))
            continue
        end
        typeid = intern_type!(language.types, ty)
        if typeid isa Err
            push!(skipped, (index=db, reason=_external_skip_reason(ty)))
            continue
        end
        _source_leaf!(language, typeid, SourceExternal(hole.anchor, db))
        admitted += 1
    end
    return (admitted=admitted, skipped=skipped, available_before=available_before)
end

"""
    import_hole(hole, language)

Project one ordinary occurrence to explicit monomorphic λ syntax. Constants
must have the same fixed meaning throughout `language`; merge or rename local
constant environments before sharing a language between ancestors. Outer
bindings become fixed, hole-anchored leaves at their occurrence instance types.
Internal binders remain De Bruijn variables. No value-equivalence matching or
closure reconstruction is performed.

After a successful projection, unused admitted monomorphic external bindings
are registered as leaves as well; callers that need coverage details can call
`register_external_bindings!` for its report without changing the term return
type.

The first slice refuses Let-App, recursive definitions, unresolved types, and
μ-dependent syntax inside the projected region. Such source programs may still
provide ordinary local holes. Refusal does not mutate source syntax. The type
table may acquire concrete entries; unused leaf additions are rolled back.
"""
function import_hole(hole::SourceHole, language::Language)
    nodes = TermNode[]
    initial_leaves = length(language.leaves)
    function visit(o, binders)
        haskey(hole.source.metadata, o) || return Err(SourceBoundaryError(:binding_artifact, o.id))
        id = intern_type!(language.types, hole.source.metadata[o][1])
        id isa Err && return id
        n = o.node
        if n isa ConstNode
            leaf = _source_leaf!(language, id, SourceConstant(n.name))
            leaf isa Err && return leaf
            push!(nodes, TermNode(LEAF, id, leaf, 0))
        elseif n isa IndexNode
            0 <= n.index < length(binders) + length(hole.external_types) ||
                return Err(SourceBoundaryError(:unbound_index, o.id))
            if n.index < length(binders)
                binders[end - n.index] == id || return Err(SourceBoundaryError(:nonmonomorphic_use, o.id))
                push!(nodes, TermNode(VAR, id, n.index, 0))
            else
                payload = SourceExternal(hole.anchor, n.index - length(binders))
                leaf = _source_leaf!(language, id, payload)
                leaf isa Err && return leaf
                push!(nodes, TermNode(LEAF, id, leaf, 0))
            end
        elseif n isa AbsNode
            ty = hole.source.metadata[o][1]
            ty isa TypeSystem.TyFun || return Err(SourceBoundaryError(:nontransparent_binder, o.id))
            argument = intern_type!(language.types, ty.arg)
            argument isa Err && return argument
            push!(binders, argument)
            body = visit(o.children[1], binders)
            pop!(binders)
            body isa Err && return body
            push!(nodes, TermNode(ABS, id, body, 0))
        elseif n isa AppNode && !(n.func isa AbsNode)
            f = visit(o.children[1], binders)
            f isa Err && return f
            x = visit(o.children[2], binders)
            x isa Err && return x
            push!(nodes, TermNode(APP, id, f, x))
        else
            return Err(SourceBoundaryError(:unsupported_region, o.id))
        end
        return Int32(length(nodes))
    end
    result = visit(hole.occurrence, Int32[])
    if result isa Err
        resize!(language.leaves, initial_leaves)
        return result
    end
    term = TypedTerm(nodes)
    valid = validate(term, language)
    if valid isa Err
        resize!(language.leaves, initial_leaves)
        return valid
    end
    if valid === false
        resize!(language.leaves, initial_leaves)
        return Err(SourceBoundaryError(:projection_type_mismatch, hole.occurrence.id))
    end
    register_external_bindings!(hole, language)
    return term
end

"""
    export_hole(term, language, hole; max_nodes=100_000, max_depth=512)

Reconstruct the entire fixed ancestor with one replacement. This includes the
ordinary O(source size) copy and does not claim constant-time export. External
references are rebased by current λ depth and must belong to this exact hole.
`max_nodes` bounds the materialised replacement (including nested donor
expansions); the caller performs full source inference at the phase's export
boundary.
"""
function export_hole(term::TypedTerm, language::Language, hole::SourceHole;
                     max_nodes::Int=100_000, max_depth::Int=512)
    max_nodes > 0 && max_depth >= 0 || throw(ArgumentError("invalid export bounds"))
    checked = validate(term, language)
    checked isa Err && return checked
    checked === false && return Err(SourceBoundaryError(:invalid_term, hole.occurrence.id))
    target = intern_type!(language.types, hole.source.metadata[hole.occurrence][1])
    target isa Err && return target
    term.nodes[end].typeid == target || return Err(SourceBoundaryError(:wrong_hole_type, hole.occurrence.id))
    materialized = Ref(0)
    function build(i, binder_depth, structural_depth)
        structural_depth <= max_depth || return Err(SourceBoundaryError(
            :export_depth, hole.occurrence.id,
        ))
        n = term.nodes[i]
        if n.kind == VAR
            materialized[] += 1
            materialized[] <= max_nodes || return Err(SourceBoundaryError(
                :export_size, hole.occurrence.id,
            ))
            return IndexNode(Int(n.x))
        elseif n.kind == LEAF
            payload = language.leaves[n.x].payload
            if payload isa SourceConstant
                materialized[] += 1
                materialized[] <= max_nodes || return Err(SourceBoundaryError(
                    :export_size, hole.occurrence.id,
                ))
                return ConstNode(payload.name)
            end
            if payload isa SourceProgram
                return _expand_source_program(payload, language;
                    max_nodes, max_depth, _counter=materialized,
                    _depth=structural_depth)
            end
            if payload isa SourceExternal && payload.anchor === hole.anchor &&
                    0 <= payload.index < length(hole.external_types)
                materialized[] += 1
                materialized[] <= max_nodes || return Err(SourceBoundaryError(
                    :export_size, hole.occurrence.id,
                ))
                return IndexNode(binder_depth + payload.index)
            end
            return Err(SourceBoundaryError(:foreign_source_leaf, hole.occurrence.id))
        elseif n.kind == ABS
            materialized[] += 1
            materialized[] <= max_nodes || return Err(SourceBoundaryError(
                :export_size, hole.occurrence.id,
            ))
            body = build(n.x, binder_depth + 1, structural_depth + 1)
            body isa Err && return body
            return AbsNode(Symbol("typed_", binder_depth), body)
        else
            materialized[] += 1
            materialized[] <= max_nodes || return Err(SourceBoundaryError(
                :export_size, hole.occurrence.id,
            ))
            f = build(n.x, binder_depth, structural_depth + 1)
            f isa Err && return f
            x = build(n.y, binder_depth, structural_depth + 1)
            x isa Err && return x
            return AppNode(f, x)
        end
    end
    replacement = build(length(term.nodes), 0, 0)
    replacement isa Err && return replacement
    owned = deepcopy(hole.source.tree)
    if isempty(hole.path)
        owned.root = replacement
    else
        parent = owned.root
        for edge in hole.path[1:end-1]
            parent = children(parent)[edge]
        end
        # Source child roles are unchanged; only one syntactic occurrence moves.
        setchild!(parent, hole.path[end], replacement)
    end
    return owned
end

@testitem "TypedSearch projection roundtrip and semantic mutation" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.Procedural: evaluate_procedural
    ctx = Dict{Symbol,TyLike}(:one => FloatTy)
    target = TyFun(FloatTy, FloatTy)
    source = prepare_source(AstTree(AbsNode(:x, IndexNode(0))), ctx, target)
    hole = source_hole(source)
    language = Language(TypeTable())
    term = import_hole(hole, language)
    @test validate(term, language) === true
    roundtrip = export_hole(term, language, hole)
    @test prepare_source(roundtrip, ctx, target) isa SourceAnalysis
    @test evaluate_procedural(roundtrip, Dict{Symbol,Any}(:one => 1.0); args=(2.0,)) == 2.0
    constant = prepare_source(AstTree(ConstNode(:one)), ctx, FloatTy)
    donor = import_hole(source_hole(constant), language)
    changed = replace_subterm(term, 1, donor, 1, language)
    @test validate(changed, language) === true
    offspring = export_hole(changed, language, hole)
    @test prepare_source(offspring, ctx, target) isa SourceAnalysis
    @test evaluate_procedural(offspring, Dict{Symbol,Any}(:one => 1.0); args=(2.0,)) == 1.0
end

@testitem "TypedSearch fixed external capture and recursive local holes" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, BoolTy
    using LambdaRegression.Procedural: FunctionDefNode
    using LambdaRegression.Utils: Err
    ctx = Dict{Symbol,TyLike}()
    target = TyFun(FloatTy, TyFun(BoolTy, FloatTy))
    source = prepare_source(AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1)))), ctx, target)
    hole = source_hole(source, [1])
    language = Language(TypeTable())
    term = import_hole(hole, language)
    roundtrip = export_hole(term, language, hole)
    @test roundtrip.root.body.body.index == 1
    @test prepare_source(roundtrip, ctx, target) isa SourceAnalysis
    foreign_hole = source_hole(source, [1])
    @test export_hole(term, language, foreign_hole) isa Err
    recursive = prepare_source(AstTree(FunctionDefNode(AbsNode(:x,
        AppNode(IndexNode(1), IndexNode(0))))), ctx, TyFun(FloatTy, FloatTy))
    recursive_hole = source_hole(recursive, [1, 1])
    recursive_language = Language(TypeTable())
    body = import_hole(recursive_hole, recursive_language)
    @test validate(body, recursive_language) === true
    rebuilt = export_hole(body, recursive_language, recursive_hole)
    @test prepare_source(rebuilt, ctx, TyFun(FloatTy, FloatTy)) isa SourceAnalysis
    # The source admits recursion; the transparent dense language does not.
    @test import_hole(source_hole(recursive), recursive_language) isa Err
end

@testitem "TypedSearch edits one occurrence of a shared source object" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy
    f = TyFun(IntTy, IntTy)
    target = TyProd(f, f)
    ctx = Dict{Symbol,TyLike}(:PAIR => TyFun(f, TyFun(f, target)), :one => IntTy)
    shared = AbsNode(:x, IndexNode(0))
    original = AstTree(AppNode(AppNode(ConstNode(:PAIR), shared), shared))
    source = prepare_source(original, ctx, target)
    hole = source_hole(source, [1, 2, 1])
    language = Language(TypeTable())
    constant = prepare_source(AstTree(ConstNode(:one)), ctx, IntTy)
    term = import_hole(source_hole(constant), language)
    changed = export_hole(term, language, hole)
    @test changed.root.func.arg.body == ConstNode(:one)
    @test changed.root.arg.body == IndexNode(0)
    @test shared.body == IndexNode(0)
    @test prepare_source(changed, ctx, target) isa SourceAnalysis
end

@testitem "TypedSearch local registration exposes unused same-typed inputs" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceExternal
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Procedural: evaluate_procedural

    target = TyFun(IntTy, TyFun(IntTy, TyFun(IntTy, IntTy)))
    source = prepare_source(AstTree(AbsNode(:x, AbsNode(:y,
        AbsNode(:z, IndexNode(2))))), Dict{Symbol,TyLike}(), target)
    hole = source_hole(source, [1, 1])
    language = Language(TypeTable())
    projected = import_hole(hole, language)
    @test projected isa TypedTerm
    intid = intern!(language.types, IntTy)
    unused = findfirst(i -> begin
        payload = language.leaves[i].payload
        payload isa SourceExternal && payload.anchor === hole.anchor && payload.index == 0
    end, eachindex(language.leaves))
    @test unused !== nothing
    candidate = TypedTerm(TermNode[
        leaf_node(intid, unused),
        abs_node(projected.nodes[end].typeid, 1),
    ])
    @test validate(candidate, language) === true
    exported = export_hole(candidate, language, hole)
    @test prepare_source(exported, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
    @test evaluate_procedural(exported, Dict{Symbol,Any}(); args=(1, 2, 3)) == 2
end

@testitem "TypedSearch local registration exposes unused recursive self" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceExternal
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Procedural: FunctionDefNode, evaluate_procedural
    using LambdaRegression.Utils: Err

    target = TyFun(IntTy, IntTy)
    source = prepare_source(AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        Dict{Symbol,TyLike}(), target)
    hole = source_hole(source, [1])
    language = Language(TypeTable())
    projected = import_hole(hole, language)
    @test projected isa TypedTerm
    original = export_hole(projected, language, hole)
    @test evaluate_procedural(original, Dict{Symbol,Any}(); args=(5,)) == 5
    intid = intern!(language.types, IntTy)
    fintid = intern!(language.types, target)
    self_leaf = findfirst(i -> begin
        payload = language.leaves[i].payload
        payload isa SourceExternal && payload.anchor === hole.anchor && payload.index == 0
    end, eachindex(language.leaves))
    @test self_leaf !== nothing
    candidate = TypedTerm(TermNode[
        leaf_node(fintid, self_leaf),
        var_node(intid, 0),
        app_node(intid, 1, 2),
        abs_node(fintid, 3),
    ])
    @test validate(candidate, language) === true
    exported = export_hole(candidate, language, hole)
    @test exported.root isa FunctionDefNode
    @test exported.root.body.body.func.index == 1
    @test exported.root.body.body.arg.index == 0
    @test prepare_source(exported, Dict{Symbol,TyLike}(), target) isa SourceAnalysis
    @test evaluate_procedural(exported, Dict{Symbol,Any}(); args=(5,), max_steps=16) isa Err
end

@testitem "TypedSearch scheme inputs retain only observed instances" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceExternal
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyScheme, newtyvar

    α = newtyvar(0)
    ctx = Dict{Symbol,TyLike}(:one => IntTy,
        :id => TyScheme([α], TyFun(α, α)))
    source = prepare_source(AstTree(AppNode(
        AbsNode(:f, AppNode(IndexNode(0), ConstNode(:one))),
        AbsNode(:x, IndexNode(0)))), ctx, IntTy)
    hole = source_hole(source, [1, 1])
    language = Language(TypeTable())
    imported = import_hole(hole, language)
    @test imported isa TypedTerm
    report = register_external_bindings!(hole, language)
    @test report.skipped == [(index=0, reason=:polymorphic_scheme)]
    fid = intern!(language.types, TyFun(IntTy, IntTy))
    @test report.available_before == [(index=0, typeid=fid)]
    @test count(leaf -> leaf.payload isa SourceExternal &&
        leaf.payload.anchor === hole.anchor, language.leaves) == 1
end

@testitem "TypedSearch unsupported μ input is skipped with coverage" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceExternal
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, newtyvar

    α = newtyvar(0)
    μ = TyMu(α, TyFun(α, IntTy))
    target = TyFun(μ, IntTy)
    source = prepare_source(AstTree(AbsNode(:x, ConstNode(:one))),
        Dict{Symbol,TyLike}(:one => IntTy), target)
    hole = source_hole(source, [1])
    language = Language(TypeTable())
    imported = import_hole(hole, language)
    @test imported isa TypedTerm
    report = register_external_bindings!(hole, language)
    @test report.admitted == 0
    @test report.available_before == []
    @test report.skipped == [(index=0, reason=:recursive_type)]
end

@testitem "TypedSearch failed imports roll back leaves and foreign anchors reject" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceExternal
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    ctx = Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy), :one => IntTy)
    unsupported_arg = AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one))
    source = prepare_source(AstTree(AppNode(ConstNode(:f), unsupported_arg)), ctx, IntTy)
    language = Language(TypeTable())
    @test import_hole(source_hole(source), language) isa Err
    @test isempty(language.leaves)

    identity = prepare_source(AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(), TyFun(IntTy, IntTy))
    first_hole = source_hole(identity, [1])
    second_hole = source_hole(identity, [1])
    first_term = import_hole(first_hole, language)
    @test first_term isa TypedTerm
    @test export_hole(first_term, language, second_hole) isa Err
end
