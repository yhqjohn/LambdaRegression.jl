"""The compact typed λ-term representation and its local variation checks."""

export TypeTable, intern!, intern_type!, arrow!, type_at, type_id, type_children_ids
export TermNode, TypedTerm, LeafSpec, Language
export VAR_KIND, LEAF_KIND, ABS_KIND, APP_KIND, VAR, LEAF, ABS, APP
export var_node, leaf_node, abs_node, app_node
export typed_var, typed_leaf, typed_abs, typed_app
export push_var!, push_leaf!, push_abs!, push_app!, add_leaf!, leaf!
export TypedSearchError, ResidualVar, SiteInfo
export validate, replace_subterm, site_info, site_context_types, residual_vars

const VAR_KIND  = UInt8(1)
const LEAF_KIND = UInt8(2)
const ABS_KIND  = UInt8(3)
const APP_KIND  = UInt8(4)
const VAR = VAR_KIND
const LEAF = LEAF_KIND
const ABS = ABS_KIND
const APP = APP_KIND

"""An expected, bounded failure from the compact typed-search boundary."""
struct TypedSearchError
    reason::Symbol
    detail::Any
end

TypedSearchError(reason::Symbol) = TypedSearchError(reason, nothing)

"""One residual free local of a selected subtree.

`db` is relative to the selected subtree's enclosing context, with zero the
nearest visible local.  `typeid` is an exact identifier in the shared table.
"""
struct ResidualVar
    db::Int32
    typeid::Int32
end

"""Context and residual dependency information at one term occurrence."""
struct SiteInfo
    context::Vector{Int32}
    residual::Vector{ResidualVar}
    typeid::Int32
    span::UnitRange{Int}
end

"""Shared collision-safe table of transparent monotypes.

Type ids are one-based `Int32`s.  `TyMu`, unresolved `TyVar`, and cyclic
mutable type graphs are intentionally rejected.  A linked `TyVar` is followed
to its committed instance and is not stored as a variable.  Rigid ids are
retained in the structural key, so two unrelated `TyRigid` identities never
collapse.
"""
mutable struct TypeTable
    types::Vector{TypeSystem.Ty}
    keys::Dict{Any,Int32}
    tags::Vector{Symbol}
    params::Vector{Tuple}
    children::Vector{Vector{Int32}}
end

TypeTable() = TypeTable(TypeSystem.Ty[], Dict{Any,Int32}(), Symbol[], Tuple[], Vector{Int32}[])

@inline function _valid_typeid(table::TypeTable, id::Integer)
    i = Int(id)
    return 1 <= i <= length(table.types)
end

@inline function _checked_typeid(table::TypeTable, id::Integer)
    _valid_typeid(table, id) || throw(ArgumentError("unknown TypeTable id $id"))
    return Int32(id)
end

"""Return the original monotype for a valid table id."""
type_at(table::TypeTable, id::Integer) = table.types[Int(_checked_typeid(table, id))]
type_id(table::TypeTable, ty::TypeSystem.Ty) = intern!(table, ty)

"""Return the child type ids recorded for a valid table id."""
type_children_ids(table::TypeTable, id::Integer) = table.children[Int(_checked_typeid(table, id))]

function _intern_type!(table::TypeTable, ty::TypeSystem.Ty,
                       active::IdDict{Any,Nothing}, memo::IdDict{Any,Int32})::Int32
    ty isa TypeSystem.TyMu && throw(ArgumentError("TyMu is not transparent in TypedSearch.TypeTable"))

    if ty isa TypeSystem.TyVar
        haskey(memo, ty) && return memo[ty]
        ty.instance === nothing && throw(ArgumentError("free TyVar is not transparent in TypedSearch.TypeTable"))
        haskey(active, ty) && throw(ArgumentError("cyclic TyVar instance is not transparent"))
        active[ty] = nothing
        try
            id = _intern_type!(table, ty.instance, active, memo)
            memo[ty] = id
            return id
        finally
            delete!(active, ty)
        end
    end

    # This is deliberately per registration call.  It prevents a shared
    # immutable/mutable type graph from being traversed exponentially while
    # avoiding any stale global cache for user-defined mutable types.
    haskey(memo, ty) && return memo[ty]

    # Protocol extensions are admitted, but a cyclic mutable object graph is
    # not.  The active map is scoped to this registration, not retained.
    if haskey(active, ty)
        throw(ArgumentError("cyclic type constructor is not transparent"))
    end
    active[ty] = nothing
    try
        tag = TypeSystem.type_constructor_tag(ty)
        params = TypeSystem.type_static_parameters(ty)
        rawchildren = TypeSystem.type_children(ty)
        childids = Int32[_intern_type!(table, child, active, memo) for child in rawchildren]
        # Rebuild composites from the already canonical table children.  In
        # particular, a TyFun(TyVar(...), ...) whose variable is linked today
        # must not retain that mutable facade in the shared table.
        child_types = tuple((table.types[Int(childid)] for childid in childids)...)
        canonical = if isempty(rawchildren)
            try
                TypeSystem.rebuild_type(ty, child_types)
            catch error
                error isa ArgumentError ? ty : rethrow()
            end
        else
            TypeSystem.rebuild_type(ty, child_types)
        end
        canonical isa TypeSystem.TyMu &&
            throw(ArgumentError("TyMu is not transparent in TypedSearch.TypeTable"))
        canonical isa TypeSystem.TyVar &&
            throw(ArgumentError("free TyVar is not transparent in TypedSearch.TypeTable"))
        tag = TypeSystem.type_constructor_tag(canonical)
        params = TypeSystem.type_static_parameters(canonical)
        key = (tag, params, Tuple(childids))
        existing = get(table.keys, key, Int32(0))
        if existing != 0
            memo[ty] = existing
            return existing
        end
        push!(table.types, canonical)
        id = Int32(length(table.types))
        push!(table.keys, key => id)
        push!(table.tags, tag)
        push!(table.params, params)
        push!(table.children, childids)
        memo[ty] = id
        return id
    finally
        delete!(active, ty)
    end
end

"""Intern a transparent `TypeSystem.Ty` and return its one-based id."""
function intern!(table::TypeTable, ty::TypeSystem.Ty)::Int32
    return _intern_type!(table, ty, IdDict{Any,Nothing}(), IdDict{Any,Int32}())
end

"""Err-returning spelling convenient for source-import preparation."""
function intern_type!(table::TypeTable, ty::TypeSystem.Ty)
    try
        return intern!(table, ty)
    catch error
        error isa ArgumentError || rethrow()
        return Err(TypedSearchError(:unsupported_type, sprint(showerror, error)))
    end
end

"""Construct/intern an arrow from two existing type ids."""
function arrow!(table::TypeTable, argid::Integer, resid::Integer)::Int32
    a = _checked_typeid(table, argid)
    r = _checked_typeid(table, resid)
    # The children are already registered, so constructing an arrow does not
    # recursively revisit either (possibly large) type expression.
    key = (:Fun, (), (a, r))
    existing = get(table.keys, key, Int32(0))
    existing != 0 && return existing
    push!(table.types, TypeSystem.TyFun(type_at(table, a), type_at(table, r)))
    push!(table.keys, key => Int32(length(table.types)))
    push!(table.tags, :Fun)
    push!(table.params, ())
    push!(table.children, Int32[a, r])
    return Int32(length(table.types))
end

"""The type ids of an arrow, or `nothing` when the id is not an arrow."""
function _arrow_parts(table::TypeTable, id::Int32)
    _valid_typeid(table, id) || return nothing
    i = Int(id)
    table.tags[i] == :Fun || return nothing
    cs = table.children[i]
    length(cs) == 2 || return nothing
    return (cs[1], cs[2])
end

"""A fixed typed external leaf occurrence."""
struct LeafSpec
    typeid::Int32
    payload::Any
    function LeafSpec(typeid::Integer, payload)
        return new(Int32(typeid), payload)
    end
end

"""The fixed type table and external leaf vocabulary for a search island."""
mutable struct Language
    types::TypeTable
    leaves::Vector{LeafSpec}
end

Language(types::TypeTable) = Language(types, LeafSpec[])
Language(types::TypeTable, leaves::AbstractVector{<:LeafSpec}) =
    Language(types, LeafSpec[leaves...])

function Language(types::TypeTable, leaves::AbstractVector)
    converted = LeafSpec[]
    for leaf in leaves
        leaf isa LeafSpec || throw(ArgumentError("language leaves must be LeafSpec values"))
        push!(converted, leaf)
    end
    return Language(types, converted)
end

"""Append a fixed external leaf and return its one-based leaf index."""
function add_leaf!(language::Language, typeid::Integer, payload)::Int32
    tid = _checked_typeid(language.types, typeid)
    push!(language.leaves, LeafSpec(tid, payload))
    return Int32(length(language.leaves))
end

leaf!(language::Language, typeid::Integer, payload) = add_leaf!(language, typeid, payload)

"""One node in a strict post-order compact λ tree."""
struct TermNode
    kind::UInt8
    typeid::Int32
    x::Int32
    y::Int32
    function TermNode(kind::UInt8, typeid::Int32, x::Int32, y::Int32)
        kind in (VAR_KIND, LEAF_KIND, ABS_KIND, APP_KIND) ||
            throw(ArgumentError("unknown TypedSearch node kind $kind"))
        new(kind, typeid, x, y)
    end
end

TermNode(kind::Integer, typeid::Integer, x::Integer=0, y::Integer=0) =
    TermNode(UInt8(kind), Int32(typeid), Int32(x), Int32(y))

"""A compact tree whose final node is its root."""
struct TypedTerm
    nodes::Vector{TermNode}
end

TypedTerm(nodes::AbstractVector{<:TermNode}) = TypedTerm(TermNode[nodes...])

var_node(typeid::Integer, db::Integer=0) = TermNode(VAR_KIND, typeid, db, 0)
leaf_node(typeid::Integer, leaf_index::Integer) = TermNode(LEAF_KIND, typeid, leaf_index, 0)
abs_node(typeid::Integer, body::Integer) = TermNode(ABS_KIND, typeid, body, 0)
app_node(typeid::Integer, fn::Integer, arg::Integer) = TermNode(APP_KIND, typeid, fn, arg)
typed_var(args...) = var_node(args...)
typed_leaf(args...) = leaf_node(args...)
typed_abs(args...) = abs_node(args...)
typed_app(args...) = app_node(args...)

function push_var!(nodes::Vector{TermNode}, typeid::Integer, db::Integer=0)::Int32
    push!(nodes, var_node(typeid, db)); return Int32(length(nodes))
end
function push_leaf!(nodes::Vector{TermNode}, typeid::Integer, leaf_index::Integer)::Int32
    push!(nodes, leaf_node(typeid, leaf_index)); return Int32(length(nodes))
end
function push_abs!(nodes::Vector{TermNode}, typeid::Integer, body::Integer)::Int32
    push!(nodes, abs_node(typeid, body)); return Int32(length(nodes))
end
function push_app!(nodes::Vector{TermNode}, typeid::Integer, fn::Integer, arg::Integer)::Int32
    push!(nodes, app_node(typeid, fn, arg)); return Int32(length(nodes))
end

# A language-level spelling is useful to source adapters, while the node-level
# overload above remains the compact-tree append primitive.
function push_leaf!(language::Language, typeid::Integer, payload)
    return add_leaf!(language, typeid, payload)
end

@inline _node_kind_name(k::UInt8) = k == VAR_KIND ? :var : k == LEAF_KIND ? :leaf :
    k == ABS_KIND ? :abs : k == APP_KIND ? :app : :unknown

struct _CheckInfo
    start::Int
    typeid::Int32
end

@inline function _fail(reason::Symbol, detail=nothing)
    return Err(TypedSearchError(reason, detail))
end

function _check_context(table::TypeTable, context::AbstractVector{<:Integer})
    out = Int32[]
    sizehint!(out, length(context))
    for id in context
        _valid_typeid(table, id) || return _fail(:unknown_context_type, id)
        push!(out, Int32(id))
    end
    return out
end

function _check_node(nodes::Vector{TermNode}, i::Int, context::Vector{Int32}, language::Language)
    i >= 1 && i <= length(nodes) || return _fail(:bad_child_index, i)
    node = nodes[i]
    table = language.types
    _valid_typeid(table, node.typeid) || return _fail(:unknown_node_type, (i, node.typeid))
    if node.kind == VAR_KIND
        node.y == 0 || return _fail(:nonzero_unused_field, (i, :y, node.y))
        node.x >= 0 || return _fail(:negative_de_bruijn_index, (i, node.x))
        db = Int(node.x)
        db < length(context) || return _fail(:unbound_variable, (i, node.x))
        expected = context[length(context) - db]
        node.typeid == expected || return _fail(:variable_type_mismatch, (i, node.typeid, expected))
        return _CheckInfo(i, node.typeid)
    elseif node.kind == LEAF_KIND
        node.y == 0 || return _fail(:nonzero_unused_field, (i, :y, node.y))
        node.x >= 1 && Int(node.x) <= length(language.leaves) ||
            return _fail(:bad_leaf_index, (i, node.x))
        leaf = language.leaves[Int(node.x)]
        _valid_typeid(table, leaf.typeid) || return _fail(:unknown_leaf_type, (i, leaf.typeid))
        node.typeid == leaf.typeid || return _fail(:leaf_type_mismatch, (i, node.typeid, leaf.typeid))
        return _CheckInfo(i, node.typeid)
    elseif node.kind == ABS_KIND
        node.y == 0 || return _fail(:nonzero_unused_field, (i, :y, node.y))
        body = Int(node.x)
        body == i - 1 || return _fail(:non_postorder_abs_child, (i, node.x))
        parts = _arrow_parts(table, node.typeid)
        parts === nothing && return _fail(:abs_not_function_type, (i, node.typeid))
        argid, resid = parts
        push!(context, argid)
        result = try
            _check_node(nodes, body, context, language)
        finally
            pop!(context)
        end
        result isa Err && return result
        result.typeid == resid || return _fail(:abs_body_type_mismatch, (i, result.typeid, resid))
        return _CheckInfo(result.start, node.typeid)
    elseif node.kind == APP_KIND
        arg = Int(node.y)
        fn = Int(node.x)
        arg == i - 1 || return _fail(:non_postorder_app_argument, (i, node.y))
        arg >= 1 || return _fail(:bad_child_index, (i, arg))
        arginfo = _check_node(nodes, arg, context, language)
        arginfo isa Err && return arginfo
        fn == arginfo.start - 1 || return _fail(:noncontiguous_app_children, (i, fn, arginfo.start))
        fn >= 1 || return _fail(:bad_child_index, (i, fn))
        fninfo = _check_node(nodes, fn, context, language)
        fninfo isa Err && return fninfo
        fnparts = _arrow_parts(table, fninfo.typeid)
        fnparts === nothing && return _fail(:application_function_not_arrow, (i, fninfo.typeid))
        fnarg, fnres = fnparts
        arginfo.typeid == fnarg || return _fail(:application_argument_type_mismatch,
            (i, arginfo.typeid, fnarg))
        node.typeid == fnres || return _fail(:application_result_type_mismatch,
            (i, node.typeid, fnres))
        return _CheckInfo(fninfo.start, node.typeid)
    else
        return _fail(:unknown_node_kind, (i, node.kind))
    end
end

"""Check post-order layout, binding indices, leaves, and exact type ids.

Returns `true` on success and `Err(TypedSearchError(...))` on expected invalid
input.  The check is syntax-directed and linear in a valid strict tree.
"""
function validate(term::TypedTerm, language::Language; context::AbstractVector{<:Integer}=Int32[])
    isempty(term.nodes) && return _fail(:empty_term)
    language.types === nothing && return _fail(:missing_type_table)
    ctx = _check_context(language.types, context)
    ctx isa Err && return ctx
    result = _check_node(term.nodes, length(term.nodes), ctx, language)
    result isa Err && return result
    result.start == 1 || return _fail(:disconnected_prefix, result.start)
    return true
end

"""Compute subtree starts for a term already known to have strict layout."""
function _spans(nodes::Vector{TermNode})
    spans = zeros(Int, length(nodes))
    for i in eachindex(nodes)
        node = nodes[i]
        if node.kind == VAR_KIND || node.kind == LEAF_KIND
            spans[i] = 1
        elseif node.kind == ABS_KIND
            b = Int(node.x)
            1 <= b < i || return nothing
            spans[i] = spans[b] + 1
        elseif node.kind == APP_KIND
            f, a = Int(node.x), Int(node.y)
            1 <= f < i && 1 <= a < i || return nothing
            spans[i] = spans[f] + spans[a] + 1
        else
            return nothing
        end
    end
    return spans
end

function _site_context(nodes::Vector{TermNode}, i::Int, site::Int, context::Vector{Int32},
                       table::TypeTable, spans::Vector{Int})
    i == site && return copy(context)
    node = nodes[i]
    if node.kind == ABS_KIND
        push!(context, _arrow_parts(table, node.typeid)[1])
        result = _site_context(nodes, Int(node.x), site, context, table, spans)
        pop!(context)
        return result
    elseif node.kind == APP_KIND
        result = _site_context(nodes, Int(node.x), site, context, table, spans)
        result !== nothing && return result
        return _site_context(nodes, Int(node.y), site, context, table, spans)
    end
    return nothing
end

function _collect_residual!(nodes::Vector{TermNode}, i::Int, depth::Int, context::Vector{Int32},
                            seen::Set{Int32}, out::Vector{ResidualVar}, spans::Vector{Int})
    node = nodes[i]
    if node.kind == VAR_KIND
        db = Int(node.x)
        if db >= depth
            rem = db - depth
            rem < length(context) || return _fail(:unbound_donor_variable, (i, rem))
            rid = Int32(rem)
            if !(rid in seen)
                push!(seen, rid)
                push!(out, ResidualVar(rid, context[length(context) - rem]))
            end
        end
    elseif node.kind == ABS_KIND
        result = _collect_residual!(nodes, Int(node.x), depth + 1, context, seen, out, spans)
        result isa Err && return result
    elseif node.kind == APP_KIND
        result = _collect_residual!(nodes, Int(node.x), depth, context, seen, out, spans)
        result isa Err && return result
        result = _collect_residual!(nodes, Int(node.y), depth, context, seen, out, spans)
        result isa Err && return result
    end
    return true
end

function _site_info_checked(term::TypedTerm, site::Integer, language::Language,
                            context::AbstractVector{<:Integer})
    1 <= Int(site) <= length(term.nodes) || return _fail(:bad_site, site)
    checked = validate(term, language; context=context)
    checked === true || return checked
    ctx = _check_context(language.types, context)
    ctx isa Err && return ctx
    spans = _spans(term.nodes)
    spans === nothing && return _fail(:bad_layout)
    sitectx = _site_context(term.nodes, length(term.nodes), Int(site), ctx, language.types, spans)
    sitectx === nothing && return _fail(:unreachable_site, site)
    residual = ResidualVar[]
    result = _collect_residual!(term.nodes, Int(site), 0, sitectx, Set{Int32}(), residual, spans)
    result isa Err && return result
    sort!(residual, by=x -> x.db)
    return SiteInfo(sitectx, residual, term.nodes[Int(site)].typeid,
                    (Int(site) - spans[Int(site)] + 1):Int(site))
end

"""Return visible context and residual free locals for one occurrence."""
function site_info(term::TypedTerm, site::Integer, language::Language;
                   context::AbstractVector{<:Integer}=Int32[])
    return _site_info_checked(term, site, language, context)
end

site_context_types(term::TypedTerm, site::Integer, language::Language;
                   context::AbstractVector{<:Integer}=Int32[]) =
    (result = site_info(term, site, language; context=context); result isa Err ? result : result.context)

residual_vars(term::TypedTerm, site::Integer, language::Language;
              context::AbstractVector{<:Integer}=Int32[]) =
    (result = site_info(term, site, language; context=context); result isa Err ? result : result.residual)

function _mapping_lookup(mapping, key::Int32)
    if haskey(mapping, key)
        return mapping[key]
    elseif haskey(mapping, Int(key))
        return mapping[Int(key)]
    end
    return nothing
end

function _copy_subtree!(out::Vector{TermNode}, nodes::Vector{TermNode}, i::Int, depth::Int,
                        donor_context::Vector{Int32}, target_context::Vector{Int32},
                        mapping, language::Language, spans::Vector{Int})
    node = nodes[i]
    if node.kind == VAR_KIND
        db = Int(node.x)
        if db < depth
            newdb = db
        else
            rem = db - depth
            mapped = _mapping_lookup(mapping, Int32(rem))
            mapped === nothing && return _fail(:missing_variable_mapping, Int32(rem))
            mapped isa Integer || return _fail(:invalid_variable_mapping, (rem, mapped))
            targetdb = Int(mapped)
            0 <= targetdb < length(target_context) ||
                return _fail(:mapped_variable_not_visible, (rem, targetdb))
            donor_type = donor_context[length(donor_context) - rem]
            target_type = target_context[length(target_context) - targetdb]
            donor_type == target_type || return _fail(:mapped_variable_type_mismatch,
                (rem, donor_type, targetdb, target_type))
            newdb = targetdb + depth
        end
        push!(out, TermNode(VAR_KIND, node.typeid, newdb, 0))
        return Int32(length(out))
    elseif node.kind == LEAF_KIND
        push!(out, node)
        return Int32(length(out))
    elseif node.kind == ABS_KIND
        parts = _arrow_parts(language.types, node.typeid)
        parts === nothing && return _fail(:abs_not_function_type, node.typeid)
        body = _copy_subtree!(out, nodes, Int(node.x), depth + 1, donor_context,
            target_context, mapping, language, spans)
        body isa Err && return body
        push!(out, TermNode(ABS_KIND, node.typeid, body, 0))
        return Int32(length(out))
    elseif node.kind == APP_KIND
        fn = _copy_subtree!(out, nodes, Int(node.x), depth, donor_context,
            target_context, mapping, language, spans)
        fn isa Err && return fn
        arg = _copy_subtree!(out, nodes, Int(node.y), depth, donor_context,
            target_context, mapping, language, spans)
        arg isa Err && return arg
        push!(out, TermNode(APP_KIND, node.typeid, fn, arg))
        return Int32(length(out))
    end
    return _fail(:unknown_node_kind, node.kind)
end

function _copy_target!(out::Vector{TermNode}, nodes::Vector{TermNode}, i::Int, site::Int,
                       donor::TypedTerm, donor_site::Int, target_info::SiteInfo,
                       donor_info::SiteInfo, language::Language, mapping,
                       spans::Vector{Int}, donor_spans::Vector{Int})
    if i == site
        copied = _copy_subtree!(out, donor.nodes, donor_site, 0, donor_info.context,
            target_info.context, mapping, language, donor_spans)
        copied isa Err && return copied
        return copied
    end
    node = nodes[i]
    if node.kind == VAR_KIND || node.kind == LEAF_KIND
        push!(out, node)
    elseif node.kind == ABS_KIND
        body = _copy_target!(out, nodes, Int(node.x), site, donor, donor_site,
            target_info, donor_info, language, mapping, spans, donor_spans)
        body isa Err && return body
        push!(out, TermNode(ABS_KIND, node.typeid, body, 0))
    elseif node.kind == APP_KIND
        fn = _copy_target!(out, nodes, Int(node.x), site, donor, donor_site,
            target_info, donor_info, language, mapping, spans, donor_spans)
        fn isa Err && return fn
        arg = _copy_target!(out, nodes, Int(node.y), site, donor, donor_site,
            target_info, donor_info, language, mapping, spans, donor_spans)
        arg isa Err && return arg
        push!(out, TermNode(APP_KIND, node.typeid, fn, arg))
    else
        return _fail(:unknown_node_kind, node.kind)
    end
    return Int32(length(out))
end

"""Replace one target occurrence with a typed donor occurrence.

`mapping` is explicit and finite: each donor residual DB index is mapped to a
visible target DB index.  The operation copies both trees, preserves donor
binders, and adjusts mapped variables below those binders, so capture is not
possible.  No candidate mapping enumeration is performed.
"""
function replace_subterm(term::TypedTerm, site::Integer, donor::TypedTerm, donor_site::Integer,
                         language::Language; context::AbstractVector{<:Integer}=Int32[],
                         donor_context::AbstractVector{<:Integer}=Int32[], mapping=Dict{Int32,Int32}())
    target_info = _site_info_checked(term, site, language, context)
    target_info isa Err && return target_info
    donor_info = _site_info_checked(donor, donor_site, language, donor_context)
    donor_info isa Err && return donor_info
    target_info.typeid == donor_info.typeid ||
        return _fail(:replacement_type_mismatch, (target_info.typeid, donor_info.typeid))
    mapping isa AbstractDict || return _fail(:mapping_not_dictionary, typeof(mapping))
    for residual in donor_info.residual
        mapped = _mapping_lookup(mapping, residual.db)
        mapped === nothing && return _fail(:missing_variable_mapping, residual.db)
        mapped isa Integer || return _fail(:invalid_variable_mapping, (residual.db, mapped))
        0 <= Int(mapped) < length(target_info.context) ||
            return _fail(:mapped_variable_not_visible, (residual.db, mapped))
        target_type = target_info.context[length(target_info.context) - Int(mapped)]
        target_type == residual.typeid ||
            return _fail(:mapped_variable_type_mismatch, (residual.db, residual.typeid, mapped, target_type))
    end
    spans = _spans(term.nodes)
    donor_spans = _spans(donor.nodes)
    spans === nothing && return _fail(:bad_layout)
    donor_spans === nothing && return _fail(:bad_donor_layout)
    out = TermNode[]
    sizehint!(out, length(term.nodes) - (last(target_info.span) - first(target_info.span) + 1) +
        (last(donor_info.span) - first(donor_info.span) + 1))
    result = _copy_target!(out, term.nodes, length(term.nodes), Int(site), donor, Int(donor_site),
        target_info, donor_info, language, mapping, spans, donor_spans)
    result isa Err && return result
    return TypedTerm(out)
end

@testitem "TypedSearch compact terms validate transparent types" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun, TyMu, TyRigid, newtyvar, unify!
    using LambdaRegression.TypedSearch

    table = TypeTable()
    float = intern!(table, FloatTy)
    bool = intern!(table, BoolTy)
    ffloat = arrow!(table, float, float)
    lang = Language(table, [LeafSpec(float, :one), LeafSpec(bool, :yes)])

    lambda = TypedTerm(TermNode[
        var_node(float, 0),
        abs_node(ffloat, 1),
    ])
    @test validate(lambda, lang) === true
    @test validate(TypedTerm(TermNode[var_node(bool, 1)]), lang; context=Int32[float]) isa Err

    # Two distinct same-type locals remain ordinary distinct occurrences.
    nodes = TermNode[var_node(ffloat, 0), var_node(float, 1), app_node(float, 1, 2)]
    @test validate(TypedTerm(nodes), lang; context=Int32[float, ffloat]) === true

    # The type table key is structural but does not erase rigid identity.
    @test intern!(table, TyRigid(17)) != intern!(table, TyRigid(18))
    α = newtyvar(0)
    @test intern_type!(table, α) isa Err
    @test intern_type!(table, TyMu(α, FloatTy)) isa Err

    linked = newtyvar(0)
    @test isnothing(unify!(linked, FloatTy))
    linked_id = intern!(table, TyFun(linked, FloatTy))
    @test type_at(table, linked_id).arg == FloatTy
    @test !(type_at(table, linked_id).arg isa typeof(linked))
end

@testitem "TypedSearch intern memo preserves shared higher-order types" begin
    using Test
    using LambdaRegression.TypeSystem: TyFun, IntTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    current = let current = TyFun(IntTy, IntTy)
        for _ in 1:30
            current = TyFun(current, current)
        end
        current
    end
    root = intern!(table, current)
    @test root > 0
    @test length(table.types) == 32
    @test intern!(table, current) == root
end

@testitem "TypedSearch replacement rebases open locals without capture" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    int = intern!(table, IntTy)
    fint = arrow!(table, int, int)
    lang = Language(table, [LeafSpec(fint, :identity)])

    # Donor is λ. outer, selected at its root.  Its residual local is donor
    # DB 0; the target maps that local to target DB 1, below a nearer function.
    donor = TypedTerm(TermNode[var_node(int, 1), abs_node(fint, 1)])
    target = TypedTerm(TermNode[var_node(fint, 0)])
    @test validate(donor, lang; context=Int32[int]) === true
    @test validate(target, lang; context=Int32[int, fint]) === true
    info = site_info(donor, 2, lang; context=Int32[int])
    @test info.residual == [ResidualVar(0, int)]

    replaced = replace_subterm(target, 1, donor, 2, lang;
        context=Int32[int, fint], donor_context=Int32[int],
        mapping=Dict{Int32,Int32}(0 => 1))
    @test replaced isa TypedTerm
    @test validate(replaced, lang; context=Int32[int, fint]) === true
    @test replaced.nodes[1].kind == VAR
    @test replaced.nodes[1].x == 2 # target DB 1 shifted under donor's λ

    @test replace_subterm(target, 1, donor, 2, lang;
        context=Int32[int, fint], donor_context=Int32[int],
        mapping=Dict{Int32,Int32}()) isa Err
    @test replace_subterm(target, 1, donor, 2, lang;
        context=Int32[int, fint], donor_context=Int32[int],
        mapping=Dict{Int32,Int32}(0 => 0)) isa Err

    # Replacing an entire λ occurrence with a fixed function leaf is the
    # bounded hoist/deletion case: the removed binder needs no transport.
    bound = TypedTerm(TermNode[var_node(int, 0), abs_node(fint, 1)])
    identity = TypedTerm(TermNode[leaf_node(fint, 1)])
    @test validate(bound, lang) === true
    hoisted = replace_subterm(bound, 2, identity, 1, lang)
    @test hoisted isa TypedTerm
    @test validate(hoisted, lang) === true
end
