"""Private, per-proposal structural facts for compact typed terms.

`_prepared_sites` is intentionally a session-local snapshot.  It is built
after the caller has checked the source term and language, and retains only
the borrowed input node vector plus structural metadata.  The proposal
session must treat that mutable vector as read-only; identity checks reject a
different term, while in-place mutation is outside this private contract.
Contexts and
residuals are reconstructed for the selected occurrence; they are not cached
for every occurrence.  Frame and binder ids are local to one snapshot, so a
numeric De Bruijn index from a parent can never accidentally identify a donor
binder (or an external context slot from another snapshot).
"""

struct _PreparedFrame
    parent::Int32
    typeid::Int32
    binderid::Int32
end

struct _PreparedSites
    term::TypedTerm
    language::Language
    snapshot_id::UInt
    nodes::Vector{TermNode}
    root_context::Vector{Int32}
    frames::Vector{_PreparedFrame}
    env::Vector{Int32}
    var_binder::Vector{Int32}
    span_start::Vector{Int32}
    span_end::Vector{Int32}
    height::Vector{Int32}
    site_depth::Vector{Int32}
    visible_same_type::Vector{Int32}
    leaf_counts::Dict{Int32,Int32}
    by_type::Dict{Int32,Vector{Int}}
    type_ids::Vector{Int32}
    atomic_sites::Vector{Int}
    point_sites::Vector{Int}
    hoist_sites::Vector{Int}
end

@inline function _prepared_owner(p::_PreparedSites, term::TypedTerm,
                                 language::Language)
    # A TypedTerm owns its node vector.  Identity checks make using facts from
    # a prior child edit fail clearly instead of silently reading stale spans.
    p.nodes === term.nodes && p.language === language
end

function _prepared_push_frame!(frames::Vector{_PreparedFrame}, parent::Int32,
                               typeid::Int32, binderid::Int32)
    push!(frames, _PreparedFrame(parent, typeid, binderid))
    return Int32(length(frames))
end

function _prepared_frame_at(p::_PreparedSites, env::Int32, offset::Int)
    offset >= 0 || return nothing
    current = env
    for _ in 1:offset
        current == 0 && return nothing
        current = p.frames[Int(current)].parent
    end
    current == 0 && return nothing
    return p.frames[Int(current)]
end

"""Return one selected occurrence's visible context without validation."""
function _prepared_context(p::_PreparedSites, env::Int32)
    result = Int32[]
    current = env
    while current != 0
        frame = p.frames[Int(current)]
        push!(result, frame.typeid)
        current = frame.parent
    end
    reverse!(result)
    return result
end

function _prepared_collect_residual!(p::_PreparedSites, i::Int, depth::Int,
                                     seen::Set{Int32}, out::Vector{ResidualVar})
    node = p.nodes[i]
    if node.kind == VAR_KIND
        db = Int(node.x)
        if db >= depth
            rem = db - depth
            binderid = p.var_binder[i]
            binderid == 0 && return _fail(:unbound_donor_variable, (i, rem))
            frame = p.frames[Int(binderid)]
            # Identity, rather than type or DB alone, is the deduplication key.
            if !(frame.binderid in seen)
                push!(seen, frame.binderid)
                push!(out, ResidualVar(rem, frame.typeid))
            end
        end
    elseif node.kind == ABS_KIND
        result = _prepared_collect_residual!(p, Int(node.x), depth + 1, seen, out)
        result isa Err && return result
    elseif node.kind == APP_KIND
        result = _prepared_collect_residual!(p, Int(node.x), depth, seen, out)
        result isa Err && return result
        result = _prepared_collect_residual!(p, Int(node.y), depth, seen, out)
        result isa Err && return result
    else
        node.kind in (LEAF_KIND, VAR_KIND) ||
            return _fail(:unknown_node_kind, (i, node.kind))
    end
    return true
end

"""Materialize `SiteInfo` only for an occurrence selected by a proposal."""
function _prepared_site_info(p::_PreparedSites, site::Integer;
                             term::Union{Nothing,TypedTerm}=nothing,
                             language::Union{Nothing,Language}=nothing)
    if term !== nothing && language !== nothing && !_prepared_owner(p, term, language)
        return _fail(:stale_prepared_sites)
    end
    i = Int(site)
    1 <= i <= length(p.nodes) || return _fail(:bad_site, site)
    context = _prepared_context(p, p.env[i])
    residual = ResidualVar[]
    result = _prepared_collect_residual!(p, i, 0, Set{Int32}(), residual)
    result isa Err && return result
    sort!(residual, by=x -> x.db)
    return SiteInfo(context, residual, p.nodes[i].typeid,
                    Int(p.span_start[i]):Int(p.span_end[i]))
end

@inline function _prepared_leaf_count(p::_PreparedSites, typeid::Int32)
    get(p.leaf_counts, typeid, Int32(0))
end

@inline function _prepared_atomic_qualified_values(visible_same_type::Vector{Int32},
                                                    leaf_counts::Dict{Int32,Int32},
                                                    node::TermNode, i::Int)
    node.kind in (VAR_KIND, LEAF_KIND) || return false
    total = Int(visible_same_type[i]) + Int(get(leaf_counts, node.typeid, Int32(0)))
    # Remove the selected source itself; VAR and LEAF namespaces are distinct.
    node.kind == VAR_KIND && (total -= 1)
    node.kind == LEAF_KIND && (total -= 1)
    total > 0
end

@inline function _prepared_point_qualified_values(visible_same_type::Vector{Int32},
                                                   leaf_counts::Dict{Int32,Int32},
                                                   node::TermNode, i::Int)
    total = Int(visible_same_type[i]) + Int(get(leaf_counts, node.typeid, Int32(0)))
    node.kind in (VAR_KIND, LEAF_KIND) && (total -= 1)
    total > 0
end

@inline function _prepared_atomic_qualified(p::_PreparedSites, i::Int)
    _prepared_atomic_qualified_values(p.visible_same_type, p.leaf_counts,
        p.nodes[i], i)
end

@inline function _prepared_point_qualified(p::_PreparedSites, i::Int)
    _prepared_point_qualified_values(p.visible_same_type, p.leaf_counts,
        p.nodes[i], i)
end

"""Check whether one occurrence has a visible head reaching its result type."""
function _prepared_has_head(p::_PreparedSites, i::Int, targetid::Int32,
                            max_args::Int)
    max_args >= 0 || return false
    current_env = p.env[i]
    while current_env != 0
        current = p.frames[Int(current_env)].typeid
        nargs = 0
        saw_target_arg = false
        while nargs < max_args
            parts = _arrow_parts(p.language.types, current)
            parts === nothing && break
            nargs += 1
            saw_target_arg |= parts[1] == targetid
            current = parts[2]
            current == targetid && saw_target_arg && return true
        end
        current_env = p.frames[Int(current_env)].parent
    end
    for leaf in p.language.leaves
        _valid_typeid(p.language.types, leaf.typeid) || continue
        current = leaf.typeid
        nargs = 0
        saw_target_arg = false
        while nargs < max_args
            parts = _arrow_parts(p.language.types, current)
            parts === nothing && break
            nargs += 1
            saw_target_arg |= parts[1] == targetid
            current = parts[2]
            current == targetid && saw_target_arg && return true
        end
    end
    return false
end

"""Build one facts snapshot from an already checked term/context."""
function _prepared_sites(term::TypedTerm, language::Language,
                         context::AbstractVector{<:Integer}=Int32[])
    isempty(term.nodes) && return _fail(:empty_term)
    ctx = Int32[context...]
    spans = _spans(term.nodes)
    spans === nothing && return _fail(:bad_layout)
    n = length(term.nodes)
    frames = _PreparedFrame[_PreparedFrame(0, Int32(0), Int32(0))]
    env = zeros(Int32, n)
    var_binder = zeros(Int32, n)
    site_depth = zeros(Int32, n)
    visible_same_type = zeros(Int32, n)

    # External context slots and λ binders receive ids in this snapshot only.
    root_env = Int32(0)
    next_binder = Int32(0)
    counts = Dict{Int32,Int}()
    visible_frames = Int32[]
    for typeid in ctx
        next_binder = next_binder + Int32(1)
        root_env = _prepared_push_frame!(frames, root_env, typeid, next_binder)
        push!(visible_frames, root_env)
        counts[typeid] = get(counts, typeid, 0) + 1
    end

    # Enter/exit events keep the visible-binder count O(n), even for deeply
    # nested terms.  The frame chain remains the authoritative context source.
    stack = Tuple{Int,Int32,Int,Bool,Int32}[(n, root_env, 0, true, Int32(0))]
    while !isempty(stack)
        i, current_env, depth, entering, binder_type = pop!(stack)
        node = term.nodes[i]
        if !entering
            pop!(visible_frames)
            counts[binder_type] = get(counts, binder_type, 1) - 1
            counts[binder_type] == 0 && delete!(counts, binder_type)
            continue
        end
        env[i] = current_env
        site_depth[i] = Int32(depth)
        visible_same_type[i] = Int32(get(counts, node.typeid, 0))
        if node.kind == VAR_KIND
            db = Int(node.x)
            db < length(visible_frames) ||
                return _fail(:unbound_donor_variable, (i, node.x))
            var_binder[i] = visible_frames[end - db]
        end
        if node.kind == ABS_KIND
            parts = _arrow_parts(language.types, node.typeid)
            parts === nothing && return _fail(:abs_not_function_type, (i, node.typeid))
            argid = parts[1]
            next_binder = next_binder + Int32(1)
            child_env = _prepared_push_frame!(frames, current_env, argid, next_binder)
            push!(stack, (i, current_env, depth, false, argid))
            counts[argid] = get(counts, argid, 0) + 1
            push!(visible_frames, child_env)
            push!(stack, (Int(node.x), child_env, depth + 1, true, Int32(0)))
        elseif node.kind == APP_KIND
            # Traversal order is immaterial to facts; preserving function then
            # argument order here also makes any future diagnostics stable.
            push!(stack, (Int(node.y), current_env, depth + 1, true, Int32(0)))
            push!(stack, (Int(node.x), current_env, depth + 1, true, Int32(0)))
        elseif node.kind == VAR_KIND || node.kind == LEAF_KIND
            nothing
        else
            return _fail(:unknown_node_kind, (i, node.kind))
        end
    end

    heights = zeros(Int32, n)
    by_type = Dict{Int32,Vector{Int}}()
    for i in eachindex(term.nodes)
        node = term.nodes[i]
        if node.kind == VAR_KIND || node.kind == LEAF_KIND
            heights[i] = 0
        elseif node.kind == ABS_KIND
            heights[i] = heights[Int(node.x)] + 1
        else
            heights[i] = max(heights[Int(node.x)], heights[Int(node.y)]) + 1
        end
        push!(get!(by_type, node.typeid, Int[]), i)
    end
    leaf_counts = Dict{Int32,Int32}()
    for leaf in language.leaves
        _valid_typeid(language.types, leaf.typeid) || continue
        leaf_counts[leaf.typeid] = get(leaf_counts, leaf.typeid, Int32(0)) + 1
    end
    span_start = Int32[i - spans[i] + 1 for i in eachindex(spans)]
    span_end = Int32[i for i in eachindex(spans)]
    type_ids = sort!(Int32[k for k in keys(by_type)])
    atomic_sites = Int[i for i in eachindex(term.nodes) if
        _prepared_atomic_qualified_values(visible_same_type, leaf_counts,
            term.nodes[i], i)]
    point_sites = Int[i for i in eachindex(term.nodes) if
        _prepared_point_qualified_values(visible_same_type, leaf_counts,
            term.nodes[i], i)]
    hoist_sites = Int[]
    for i in eachindex(term.nodes)
        positions = by_type[term.nodes[i].typeid]
        lo = searchsortedfirst(positions, Int(span_start[i]))
        hi = searchsortedlast(positions, Int(span_end[i]))
        hi - lo >= 1 && push!(hoist_sites, i)
    end
    return _PreparedSites(term, language, objectid(ctx), term.nodes, ctx, frames, env, var_binder,
        span_start, span_end, heights, site_depth, visible_same_type,
        leaf_counts, by_type, type_ids, atomic_sites, point_sites, hoist_sites)
end

@inline _prepared_span(p::_PreparedSites, i::Int) =
    Int(p.span_start[i]):Int(p.span_end[i])

@testitem "TypedSearch prepared site facts match checked site_info" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: FloatTy, BoolTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    boolean = intern!(table, BoolTy)
    fscalar = arrow!(table, scalar, scalar)
    fboolscalar = arrow!(table, boolean, scalar)
    # The invalid leaf is deliberately unused.  It must not become a source.
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(scalar, :b),
        LeafSpec(Int32(99), :invalid), LeafSpec(fboolscalar, :f)])

    # Both external slots occur in one selected application and have distinct
    # types, so DB 0/DB 1 must retain their actual binder identities.
    open_term = TypedTerm(TermNode[
        var_node(fboolscalar, 0), var_node(boolean, 1),
        app_node(scalar, 1, 2),
    ])
    open_ctx = Int32[scalar, boolean, fboolscalar]
    @test validate(open_term, language; context=open_ctx) === true
    prepared = LambdaRegression.TypedSearch._prepared_sites(open_term, language, open_ctx)
    @test prepared isa LambdaRegression.TypedSearch._PreparedSites
    @test prepared.leaf_counts[scalar] == 2
    @test !haskey(prepared.leaf_counts, Int32(99))
    for i in eachindex(open_term.nodes)
        slow = site_info(open_term, i, language; context=open_ctx)
        fast = LambdaRegression.TypedSearch._prepared_site_info(prepared, i;
            term=open_term, language=language)
        @test fast.context == slow.context
        @test fast.residual == slow.residual
        @test fast.typeid == slow.typeid
        @test fast.span == slow.span
    end

    # Same-type shadowing: selecting the abstraction must expose the outer
    # binder, while selecting the body exposes it at DB 1.
    shadow = TypedTerm(TermNode[var_node(scalar, 1), abs_node(fscalar, 1)])
    shadow_ctx = Int32[scalar]
    @test validate(shadow, language; context=shadow_ctx) === true
    shadow_prepared = LambdaRegression.TypedSearch._prepared_sites(shadow,
        language, shadow_ctx)
    for i in eachindex(shadow.nodes)
        slow = site_info(shadow, i, language; context=shadow_ctx)
        fast = LambdaRegression.TypedSearch._prepared_site_info(shadow_prepared, i;
            term=shadow, language=language)
        @test fast.context == slow.context
        @test fast.residual == slow.residual
    end
    donor_prepared = LambdaRegression.TypedSearch._prepared_sites(shadow,
        language, shadow_ctx)
    @test prepared.snapshot_id != donor_prepared.snapshot_id
    replacement = TypedTerm(copy(shadow.nodes))
    @test LambdaRegression.TypedSearch._prepared_site_info(shadow_prepared, 1;
        term=replacement, language=language) isa Err

    @test LambdaRegression.TypedSearch._prepared_sites(
        TypedTerm(TermNode[leaf_node(scalar, 1)]), language, Int32[]).snapshot_id !=
        prepared.snapshot_id
end
