abstract type _ScopedTerm end

struct _ScopedConst <: _ScopedTerm
    name::Symbol
end

struct _ScopedFree <: _ScopedTerm
    name::Symbol
end

struct _ScopedBound <: _ScopedTerm
    binder_id::Int
end

struct _ScopedAbs <: _ScopedTerm
    binder_id::Int
    name::Symbol
    body::_ScopedTerm
end

struct _ScopedApp <: _ScopedTerm
    func::_ScopedTerm
    arg::_ScopedTerm
    role::Symbol

    function _ScopedApp(func::_ScopedTerm, arg::_ScopedTerm)
        return new(func, arg, func isa _ScopedAbs ? :let_like : :ordinary)
    end
end

mutable struct _ScopedAncestry
    nodes::Vector{AstNode}
    node_ids::IdDict{AstNode,Int}
    parent_ids::Vector{Int}
    first_child_ids::Vector{Int}
    second_child_ids::Vector{Int}
    depths::Vector{Int}
    up::Union{Nothing,Matrix{Int}}
    binder_occurrences::Dict{Int,Vector{Int}}
    binder_lca::Dict{Int,Int}
end

struct _ScopedTree
    root::_ScopedTerm
    by_ast::IdDict{AstNode,_ScopedTerm}
    parent_by_ast::IdDict{AstNode,Tuple{Union{Nothing,AstNode},Int}}
    ancestry::Union{Nothing,_ScopedAncestry}
    next_binder_id::Int
end

function _compute_next_scoped_binder_id(term::_ScopedTerm)
    highest = 0
    function visit(node::_ScopedTerm)
        if node isa _ScopedAbs
            highest = max(highest, node.binder_id)
            visit(node.body)
        elseif node isa _ScopedApp
            visit(node.func)
            visit(node.arg)
        end
        return nothing
    end
    visit(term)
    return highest + 1
end

_ScopedTree(root::_ScopedTerm) = _ScopedTree(
    root,
    IdDict{AstNode,_ScopedTerm}(),
    IdDict{AstNode,Tuple{Union{Nothing,AstNode},Int}}(),
    nothing,
    _compute_next_scoped_binder_id(root),
)

struct _UnsupportedScopedNodeErr
    node_type::DataType
end

struct _UnresolvedScopedIndexErr
    index::Int
    depth::Int
end

struct _EscapedScopedBinderErr
    binder_id::Int
end

struct _DuplicateScopedBinderErr
    binder_id::Int
end

struct _ScopedPathErr
    path::Vector{Int}
    offset::Int
end

struct _ScopedApplicationRoleErr
    stored::Symbol
    materialized::Symbol
end

struct _RetainedApplicationRoleErr
    path::Vector{Int}
end

struct _ScopedLetBinding
    binder_id::Int
    name::Symbol
    rhs::_ScopedTerm
end

mutable struct _ScopedResolver
    binders::Vector{Int}
    next_binder_id::Int
    by_ast::IdDict{AstNode,_ScopedTerm}
    parent_by_ast::IdDict{AstNode,Tuple{Union{Nothing,AstNode},Int}}
    nodes::Vector{AstNode}
    node_ids::IdDict{AstNode,Int}
    parent_ids::Vector{Int}
    first_child_ids::Vector{Int}
    second_child_ids::Vector{Int}
    depths::Vector{Int}
    binder_occurrences::Dict{Int,Vector{Int}}
end

_application_role(term::_ScopedApp) = term.role
_application_role(::_ScopedTerm) = nothing

_application_roles_stable(old::_ScopedTerm, new::_ScopedTerm) =
    _application_role(old) === _application_role(new)

function _resolve_scoped_node(
    node::AstNode,
    state::_ScopedResolver,
    parent::Union{Nothing,AstNode},
    child_index::Int,
)
    state.parent_by_ast[node] = (parent, child_index)
    node_id = length(state.nodes) + 1
    parent_id = parent === nothing ? 0 : state.node_ids[parent]
    state.node_ids[node] = node_id
    push!(state.nodes, node)
    push!(state.parent_ids, parent_id)
    push!(state.first_child_ids, 0)
    push!(state.second_child_ids, 0)
    if parent_id != 0
        child_index == 1 && (state.first_child_ids[parent_id] = node_id)
        child_index == 2 && (state.second_child_ids[parent_id] = node_id)
    end
    push!(state.depths, parent_id == 0 ? 0 : state.depths[parent_id] + 1)
    if node isa ConstNode
        term = _ScopedConst(node.name)
        state.by_ast[node] = term
        return term
    elseif node isa VarNode
        term = _ScopedFree(node.name)
        state.by_ast[node] = term
        return term
    elseif node isa IndexNode
        index = node.index
        0 <= index < length(state.binders) ||
            return Err(_UnresolvedScopedIndexErr(index, length(state.binders)))
        binder_id = state.binders[end - index]
        push!(get!(state.binder_occurrences, binder_id, Int[]), node_id)
        term = _ScopedBound(binder_id)
        state.by_ast[node] = term
        return term
    elseif node isa AbsNode
        binder_id = state.next_binder_id
        state.next_binder_id += 1
        push!(state.binders, binder_id)
        body = _resolve_scoped_node(node.body, state, node, 1)
        pop!(state.binders)
        body isa Err && return body
        term = _ScopedAbs(binder_id, node.name, body)
        state.by_ast[node] = term
        return term
    elseif node isa AppNode
        func = _resolve_scoped_node(node.func, state, node, 1)
        func isa Err && return func
        arg = _resolve_scoped_node(node.arg, state, node, 2)
        arg isa Err && return arg
        term = _ScopedApp(func, arg)
        state.by_ast[node] = term
        return term
    end
    return Err(_UnsupportedScopedNodeErr(typeof(node)))
end

function _scoped_lca_ready(ancestry::_ScopedAncestry, left::Int, right::Int)
    up = ancestry.up::Matrix{Int}
    a, b = left, right
    if ancestry.depths[a] < ancestry.depths[b]
        a, b = b, a
    end
    difference = ancestry.depths[a] - ancestry.depths[b]
    level = 1
    while difference > 0
        isodd(difference) && (a = up[a, level])
        difference >>= 1
        level += 1
    end
    a == b && return a
    for column in size(up, 2):-1:1
        up_a = up[a, column]
        up_b = up[b, column]
        if up_a != up_b && up_a != 0 && up_b != 0
            a, b = up_a, up_b
        end
    end
    return ancestry.parent_ids[a]
end

function _build_scoped_ancestry(state::_ScopedResolver)
    return _ScopedAncestry(
        state.nodes,
        state.node_ids,
        state.parent_ids,
        state.first_child_ids,
        state.second_child_ids,
        state.depths,
        nothing,
        state.binder_occurrences,
        Dict{Int,Int}(),
    )
end

function _ensure_scoped_lca!(ancestry::_ScopedAncestry)
    ancestry.up === nothing || return ancestry
    node_count = length(ancestry.nodes)
    levels = 1
    while (1 << (levels - 1)) < max(node_count, 1)
        levels += 1
    end
    up = zeros(Int, node_count, levels)
    for node_id in 1:node_count
        up[node_id, 1] = ancestry.parent_ids[node_id]
    end
    for level in 2:levels, node_id in 1:node_count
        parent_id = up[node_id, level - 1]
        up[node_id, level] = parent_id == 0 ? 0 : up[parent_id, level - 1]
    end
    ancestry.up = up
    for (binder_id, occurrences) in ancestry.binder_occurrences
        isempty(occurrences) && continue
        lca = first(occurrences)
        for occurrence in @view occurrences[2:end]
            lca = _scoped_lca_ready(ancestry, lca, occurrence)
        end
        ancestry.binder_lca[binder_id] = lca
    end
    return ancestry
end

function _scoped_lca(ancestry::_ScopedAncestry, left::Int, right::Int)
    _ensure_scoped_lca!(ancestry)
    return _scoped_lca_ready(ancestry, left, right)
end

function _resolve_scoped(tree::AstTree)::Union{_ScopedTree,Err}
    state = _ScopedResolver(
        Int[],
        1,
        IdDict{AstNode,_ScopedTerm}(),
        IdDict{AstNode,Tuple{Union{Nothing,AstNode},Int}}(),
        AstNode[],
        IdDict{AstNode,Int}(),
        Int[],
        Int[],
        Int[],
        Int[],
        Dict{Int,Vector{Int}}(),
    )
    root = _resolve_scoped_node(tree.root, state, nothing, 0)
    root isa Err && return root
    return _ScopedTree(
        root,
        state.by_ast,
        state.parent_by_ast,
        _build_scoped_ancestry(state),
        state.next_binder_id,
    )
end

function _materialize_scoped_node(
    term::_ScopedTerm,
    binders::Vector{Int},
    seen_binders::Set{Int},
)
    if term isa _ScopedConst
        return ConstNode(term.name)
    elseif term isa _ScopedFree
        return VarNode(term.name)
    elseif term isa _ScopedBound
        position = findlast(==(term.binder_id), binders)
        position === nothing && return Err(_EscapedScopedBinderErr(term.binder_id))
        return IndexNode(length(binders) - position)
    elseif term isa _ScopedAbs
        term.binder_id in seen_binders &&
            return Err(_DuplicateScopedBinderErr(term.binder_id))
        push!(seen_binders, term.binder_id)
        push!(binders, term.binder_id)
        body = _materialize_scoped_node(term.body, binders, seen_binders)
        pop!(binders)
        body isa Err && return body
        return AbsNode(term.name, body)
    elseif term isa _ScopedApp
        func = _materialize_scoped_node(term.func, binders, seen_binders)
        func isa Err && return func
        arg = _materialize_scoped_node(term.arg, binders, seen_binders)
        arg isa Err && return arg
        materialized_role = func isa AbsNode ? :let_like : :ordinary
        materialized_role === term.role ||
            return Err(_ScopedApplicationRoleErr(term.role, materialized_role))
        return AppNode(func, arg)
    end
    return Err(_UnsupportedScopedNodeErr(typeof(term)))
end

function _materialize_scoped(scoped::_ScopedTree)::Union{AstTree,Err}
    root = _materialize_scoped_node(scoped.root, Int[], Set{Int}())
    root isa Err && return root
    return AstTree(root)
end

function _scoped_at_path(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
)::Union{_ScopedTerm,Err}
    node = scoped.root
    normalized_path = collect(Int, path)
    for (offset, index) in enumerate(normalized_path)
        if node isa _ScopedAbs && index == 1
            node = node.body
        elseif node isa _ScopedApp && index == 1
            node = node.func
        elseif node isa _ScopedApp && index == 2
            node = node.arg
        else
            return Err(_ScopedPathErr(normalized_path, offset))
        end
    end
    return node
end

function _application_roles_stable(
    old::_ScopedTree,
    new::_ScopedTree,
    retained_paths::AbstractVector{<:Pair},
)
    for retained in retained_paths
        old_term = _scoped_at_path(old, retained.first)
        new_term = _scoped_at_path(new, retained.second)
        old_term isa _ScopedApp || return false
        new_term isa _ScopedApp || return false
        _application_roles_stable(old_term, new_term) || return false
    end
    return true
end

function _replace_scoped_term(
    node::_ScopedTerm,
    path::Vector{Int},
    offset::Int,
    replacement::_ScopedTerm,
)
    offset > length(path) && return replacement
    index = path[offset]
    if node isa _ScopedAbs && index == 1
        body = _replace_scoped_term(node.body, path, offset + 1, replacement)
        body isa Err && return body
        return _ScopedAbs(node.binder_id, node.name, body)
    elseif node isa _ScopedApp && index == 1
        func = _replace_scoped_term(node.func, path, offset + 1, replacement)
        func isa Err && return func
        return _ScopedApp(func, node.arg)
    elseif node isa _ScopedApp && index == 2
        arg = _replace_scoped_term(node.arg, path, offset + 1, replacement)
        arg isa Err && return arg
        return _ScopedApp(node.func, arg)
    end
    return Err(_ScopedPathErr(path, offset))
end

function _replace_scoped(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
    replacement::_ScopedTerm,
)::Union{_ScopedTree,Err}
    normalized_path = collect(Int, path)
    root = _replace_scoped_term(scoped.root, normalized_path, 1, replacement)
    root isa Err && return root
    return _ScopedTree(root)
end

@inline function _scoped_uses_binder(term::_ScopedTerm, binder_id::Int)
    if term isa _ScopedBound
        return term.binder_id == binder_id
    elseif term isa _ScopedAbs
        return _scoped_uses_binder(term.body, binder_id)
    elseif term isa _ScopedApp
        return _scoped_uses_binder(term.func, binder_id) ||
               _scoped_uses_binder(term.arg, binder_id)
    end
    return false
end

function _resolved_scoped(env::MutationEnv)
    return get!(env.eligibility_cache, _resolve_scoped) do
        _resolve_scoped(env.tree)
    end
end

struct _TypedOccurrence
    node::AstNode
    path::Tuple
end

Base.:(==)(left::_TypedOccurrence, right::_TypedOccurrence) =
    left.path == right.path
Base.isequal(left::_TypedOccurrence, right::_TypedOccurrence) =
    isequal(left.path, right.path)
Base.hash(occurrence::_TypedOccurrence, seed::UInt) =
    hash(occurrence.path, hash(:TypedOccurrence, seed))

struct _TypedOccurrenceAdapter end

_typed_occurrence_child(
    occurrence::_TypedOccurrence,
    node::AstNode,
    child_index::Int,
) = _TypedOccurrence(node, (occurrence.path..., child_index))

function TypeSystem.term_view(
    ::_TypedOccurrenceAdapter,
    occurrence::_TypedOccurrence,
)
    node = occurrence.node
    node isa ConstNode && return (:const, node.name)
    node isa VarNode && return (:var, node.name)
    node isa IndexNode && return (:var, node.index)
    node isa AbsNode && return (
        :abs,
        node.name,
        _typed_occurrence_child(occurrence, node.body, 1),
    )
    node isa AppNode && return (
        :app,
        _typed_occurrence_child(occurrence, node.func, 1),
        _typed_occurrence_child(occurrence, node.arg, 2),
    )
    throw(ArgumentError(
        "typed occurrence inference does not support $(typeof(node))",
    ))
end

function _typed_occurrence_metadata(env::MutationEnv, ctx::NamedTuple)
    return get!(env.eligibility_cache, _typed_occurrence_metadata) do
        env.tree === nothing && return Err(NoEligibleSiteErr())
        root = _TypedOccurrence(env.tree.root, ())
        meta = Dict{_TypedOccurrence,Tuple{TyLike,TyLike}}()
        factory = haskey(ctx, :tyfactory) ? ctx.tyfactory : newtyvar
        inferred = try
            TypeSystem.infer_term(
                root,
                env.type_ctx,
                _TypedOccurrenceAdapter();
                factory,
                meta,
            )
        catch error
            return Err(error)
        end
        inferred isa Err && return inferred
        return Dict{Tuple,Tuple{TyLike,TyLike}}(
            occurrence.path => bounds for (occurrence, bounds) in meta
        )
    end
end

function _retained_application_ancestors(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
)
    retained = Pair{Vector{Int},Vector{Int}}[]
    prefix = Int[]
    node = scoped.root
    for index in path
        node isa _ScopedApp && push!(retained, copy(prefix) => copy(prefix))
        push!(prefix, index)
        if node isa _ScopedAbs && index == 1
            node = node.body
        elseif node isa _ScopedApp && index == 1
            node = node.func
        elseif node isa _ScopedApp && index == 2
            node = node.arg
        else
            return Err(_ScopedPathErr(collect(Int, path), length(prefix)))
        end
    end
    return retained
end

function _unused_let_at_path(scoped::_ScopedTree, path::AbstractVector{<:Integer})
    term = _scoped_at_path(scoped, path)
    term isa _ScopedApp || return nothing
    term.func isa _ScopedAbs || return nothing
    binder = term.func.binder_id
    _scoped_uses_binder(term.func.body, binder) && return nothing
    return term
end

"""
    LetPruneMutation() <: AbstractMutation

Explicitly remove an unused let-like application `(λx. body) rhs`. The
operator resolves De Bruijn references to stable binder identities once per
`mutate_tree` call, rejects used binders and retained application-role changes,
then relies on the outer mutation verifier for final target compatibility.

This operator is part of the ordinary mutation preset under a conservative
weight and remains independently removable through the injected mutation list.
"""
struct LetPruneMutation <: AbstractMutation end

required_site_kinds(::LetPruneMutation) = Symbol[:app]

function is_site_eligible(
    ::LetPruneMutation,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    return _unused_let_at_path(scoped, site.path) !== nothing
end

function apply(
    ::LetPruneMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    let_term = _unused_let_at_path(scoped, site.path)
    let_term === nothing && return Err(NoEligibleSiteErr())

    retained = _retained_application_ancestors(scoped, site.path)
    retained isa Err && return retained
    rewritten = _replace_scoped(scoped, site.path, let_term.func.body)
    rewritten isa Err && return rewritten
    if !_application_roles_stable(scoped, rewritten, retained)
        changed_path = findfirst(retained) do mapping
            !_application_roles_stable(scoped, rewritten, [mapping])
        end
        path = changed_path === nothing ? Int[] : retained[changed_path].first
        return Err(_RetainedApplicationRoleErr(path))
    end

    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

function _scoped_let_prefix(term::_ScopedTerm, limit::Int)
    bindings = _ScopedLetBinding[]
    body = term
    while length(bindings) < limit && body isa _ScopedApp && body.func isa _ScopedAbs
        push!(
            bindings,
            _ScopedLetBinding(body.func.binder_id, body.func.name, body.arg),
        )
        body = body.func.body
    end
    return bindings, body
end

function _wrap_scoped_lets(bindings::AbstractVector{_ScopedLetBinding}, body::_ScopedTerm)
    wrapped = body
    for binding in Iterators.reverse(bindings)
        wrapped = _ScopedApp(
            _ScopedAbs(binding.binder_id, binding.name, wrapped),
            binding.rhs,
        )
    end
    return wrapped
end

function _hoist_rewrite(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
    binding_count::Int,
)
    site_path = collect(Int, path)
    isempty(site_path) && return Err(NoEligibleSiteErr())
    parent_path = site_path[1:(end - 1)]

    term = _scoped_at_path(scoped, site_path)
    term isa Err && return term
    bindings, body = _scoped_let_prefix(term, binding_count)
    length(bindings) == binding_count || return Err(NoEligibleSiteErr())

    without_bindings = _replace_scoped(scoped, site_path, body)
    without_bindings isa Err && return without_bindings
    rebuilt_parent = _scoped_at_path(without_bindings, parent_path)
    rebuilt_parent isa Err && return rebuilt_parent
    wrapped_parent = _wrap_scoped_lets(bindings, rebuilt_parent)
    rewritten = _replace_scoped(
        without_bindings,
        parent_path,
        wrapped_parent,
    )
    rewritten isa Err && return rewritten

    return rewritten
end

function _legal_hoist_sizes(
    mutation,
    term::_ScopedTerm,
    parent::_ScopedTerm,
    site_is_function::Bool,
    parent_is_function::Bool,
)
    bindings, _ = _scoped_let_prefix(term, mutation.max_bindings)
    upper = min(length(bindings), mutation.max_bindings)
    upper < mutation.min_bindings && return Int[]

    # Wrapping a direct function-side abstraction would change its parent's
    # application from HM let-like to ordinary, independently of block length.
    parent_is_function && parent isa _ScopedAbs && return Int[]

    legal = Int[]
    cursor = term
    captures_crossed_binder = false
    for count in 1:upper
        binding = bindings[count]
        cursor = (cursor::_ScopedApp).func.body
        if parent isa _ScopedAbs
            captures_crossed_binder |=
                _scoped_uses_binder(binding.rhs, parent.binder_id)
        end
        count < mutation.min_bindings && continue
        captures_crossed_binder && continue

        # When the selected let is the function child, the crossed application
        # is retained inside the new lets. It must stay ordinary.
        if parent isa _ScopedApp && site_is_function && cursor isa _ScopedAbs
            continue
        end
        push!(legal, count)
    end
    return legal
end

function _legal_hoist_sizes(
    mutation,
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
)
    site_path = collect(Int, path)
    isempty(site_path) && return Int[]
    term = _scoped_at_path(scoped, site_path)
    term isa Err && return Int[]
    parent_path = site_path[1:(end - 1)]
    parent = _scoped_at_path(scoped, parent_path)
    parent isa Err && return Int[]
    parent_is_function = false
    if !isempty(parent_path) && last(parent_path) == 1
        grandparent = _scoped_at_path(scoped, parent_path[1:(end - 1)])
        parent_is_function = grandparent isa _ScopedApp
    end
    return _legal_hoist_sizes(
        mutation,
        term,
        parent,
        parent isa _ScopedApp && last(site_path) == 1,
        parent_is_function,
    )
end

function _legal_hoist_sizes(
    mutation,
    scoped::_ScopedTree,
    site::Site,
)
    site.parent === nothing && return Int[]
    term = get(scoped.by_ast, site.node, nothing)
    parent = get(scoped.by_ast, site.parent, nothing)
    if !(term isa _ScopedTerm && parent isa _ScopedTerm)
        return _legal_hoist_sizes(mutation, scoped, site.path)
    end
    grandparent, parent_child_index = get(
        scoped.parent_by_ast,
        site.parent,
        (nothing, 0),
    )
    return _legal_hoist_sizes(
        mutation,
        term,
        parent,
        site.parent isa AppNode && site.child_index == 1,
        grandparent isa AppNode && parent_child_index == 1,
    )
end


function _legal_hoist_sizes(
    mutation,
    scoped::_ScopedTree,
    site::Site,
    env::MutationEnv,
)
    cache = get!(env.eligibility_cache, _legal_hoist_sizes) do
        Dict{Tuple{Int,Int,Tuple},Vector{Int}}()
    end
    key = (mutation.min_bindings, mutation.max_bindings, Tuple(site.path))
    return get!(cache, key) do
        _legal_hoist_sizes(mutation, scoped, site)
    end
end

"""
    LetHoistMutation(; min_bindings=1, max_bindings=1) <: AbstractMutation

Move a let-like binding, or a consecutive outer binding prefix, across one
legal parent region. Binder identities are preserved and the move is rejected
when a right-hand side would escape a crossed abstraction or when an affected
application would change its direct-`Abs` HM role.

The one-binding form is part of the ordinary preset. Any `max_bindings > 1`
admits dependency-ordered block candidates; `min_bindings >= 2` makes the
operator block-only. Multi-binding configurations remain outside presets. The
usual semantic equivalence claim assumes pure object-language terms.
"""
struct LetHoistMutation <: AbstractMutation
    min_bindings::Int
    max_bindings::Int

    function LetHoistMutation(min_bindings::Int, max_bindings::Int)
        min_bindings >= 1 || throw(ArgumentError("min_bindings must be positive"))
        max_bindings >= min_bindings ||
            throw(ArgumentError("max_bindings must be at least min_bindings"))
        return new(min_bindings, max_bindings)
    end
end

LetHoistMutation(; min_bindings::Int=1, max_bindings::Int=1) =
    LetHoistMutation(min_bindings, max_bindings)

required_site_kinds(::LetHoistMutation) = Symbol[:app]

function is_site_eligible(
    mutation::LetHoistMutation,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    return !isempty(_legal_hoist_sizes(mutation, scoped, site, env))
end

function apply(
    mutation::LetHoistMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    # The stable path is also the rewrite target, so final legality must use it
    # even when a caller supplies stale or inconsistent Site object identities.
    legal_sizes = _legal_hoist_sizes(mutation, scoped, site.path)
    isempty(legal_sizes) && return Err(NoEligibleSiteErr())
    binding_count = rand(env.rng, legal_sizes)

    rewritten = _hoist_rewrite(scoped, site.path, binding_count)
    rewritten isa Err && return rewritten
    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

struct _SinkCandidate
    body_id::Int
    target_id::Int
end

@inline function _scoped_is_ancestor(
    ancestry::_ScopedAncestry,
    ancestor_id::Int,
    node_id::Int,
)
    return _scoped_lca(ancestry, ancestor_id, node_id) == ancestor_id
end

function _ancestry_id_at_path(
    ancestry::_ScopedAncestry,
    path::AbstractVector{<:Integer},
)
    node_id = 1
    for child_index in path
        node_id = child_index == 1 ?
            ancestry.first_child_ids[node_id] :
            child_index == 2 ? ancestry.second_child_ids[node_id] : 0
        node_id == 0 && return 0
    end
    return node_id
end

function _sink_candidate(mutation, scoped::_ScopedTree, site_id::Int)
    ancestry = scoped.ancestry
    ancestry === nothing && return nothing
    _ensure_scoped_lca!(ancestry)
    1 <= site_id <= length(ancestry.nodes) || return nothing
    site_node = ancestry.nodes[site_id]
    site_node isa AppNode || return nothing
    term = get(scoped.by_ast, site_node, nothing)
    term isa _ScopedApp || return nothing
    term.func isa _ScopedAbs || return nothing

    abstraction_id = ancestry.first_child_ids[site_id]
    abstraction_id == 0 && return nothing
    body_id = ancestry.first_child_ids[abstraction_id]
    body_id == 0 && return nothing
    binder_id = term.func.binder_id
    target_id = get(ancestry.binder_lca, binder_id, 0)
    target_id == 0 && return nothing
    _scoped_is_ancestor(ancestry, body_id, target_id) || return nothing

    # Removing the source let must not expose a direct abstraction as the
    # function of an application that was previously ordinary.
    source_parent_id = ancestry.parent_ids[site_id]
    if source_parent_id != 0 &&
       ancestry.nodes[source_parent_id] isa AppNode &&
       ancestry.first_child_ids[source_parent_id] == site_id &&
       ancestry.nodes[body_id] isa AbsNode
        return nothing
    end

    target_node = ancestry.nodes[target_id]
    if !mutation.allow_trivial && target_node isa IndexNode
        target_id = ancestry.parent_ids[target_id]
        target_id == 0 && return nothing
        target_node = ancestry.nodes[target_id]
    end

    # Wrapping a direct function-side abstraction would turn an existing HM
    # let-like application into an ordinary application. Use its parent region.
    parent_id = ancestry.parent_ids[target_id]
    if parent_id != 0
        parent_node = ancestry.nodes[parent_id]
        child_index = ancestry.first_child_ids[parent_id] == target_id ? 1 : 2
        if parent_node isa AppNode && child_index == 1 && target_node isa AbsNode
            target_id = parent_id
        end
    end

    target_id == body_id && return nothing
    _scoped_is_ancestor(ancestry, body_id, target_id) || return nothing
    return _SinkCandidate(body_id, target_id)
end

function _sink_candidate(mutation, scoped::_ScopedTree, site::Site)
    ancestry = scoped.ancestry
    ancestry === nothing && return nothing
    site_id = get(ancestry.node_ids, site.node, 0)
    site_id == 0 && (site_id = _ancestry_id_at_path(ancestry, site.path))
    return _sink_candidate(mutation, scoped, site_id)
end

function _sink_candidate(
    mutation,
    scoped::_ScopedTree,
    site::Site,
    env::MutationEnv,
)
    cache = get!(env.eligibility_cache, _sink_candidate) do
        Dict{Tuple{Bool,Tuple},Union{Nothing,_SinkCandidate}}()
    end
    key = (mutation.allow_trivial, Tuple(site.path))
    return get!(cache, key) do
        _sink_candidate(mutation, scoped, site)
    end
end

function _relative_ast_path(
    scoped::_ScopedTree,
    ancestor_id::Int,
    descendant_id::Int,
)
    ancestry = scoped.ancestry
    ancestry === nothing && return Err(NoEligibleSiteErr())
    ancestor_id == 0 && return Err(NoEligibleSiteErr())
    _scoped_is_ancestor(ancestry, ancestor_id, descendant_id) ||
        return Err(NoEligibleSiteErr())

    reverse_path = Int[]
    cursor = descendant_id
    while cursor != ancestor_id
        parent_id = ancestry.parent_ids[cursor]
        child_index = ancestry.first_child_ids[parent_id] == cursor ? 1 : 2
        push!(reverse_path, child_index)
        cursor = parent_id
    end
    reverse!(reverse_path)
    return reverse_path
end

function _sink_rewrite(
    scoped::_ScopedTree,
    site::Site,
    candidate::_SinkCandidate,
)
    term = _scoped_at_path(scoped, site.path)
    term isa _ScopedApp || return Err(NoEligibleSiteErr())
    term.func isa _ScopedAbs || return Err(NoEligibleSiteErr())
    relative_path = _relative_ast_path(
        scoped,
        candidate.body_id,
        candidate.target_id,
    )
    relative_path isa Err && return relative_path

    without_binding = _replace_scoped(scoped, site.path, term.func.body)
    without_binding isa Err && return without_binding
    target_path = vcat(site.path, relative_path)
    target = _scoped_at_path(without_binding, target_path)
    target isa Err && return target
    wrapped = _ScopedApp(
        _ScopedAbs(term.func.binder_id, term.func.name, target),
        term.arg,
    )
    return _replace_scoped(without_binding, target_path, wrapped)
end

"""
    LetSinkMutation(; allow_trivial=false) <: AbstractMutation

Move a let-like binding into the smallest legal descendant region containing
all uses of its binder. Unused and split-at-root bindings are rejected.
`allow_trivial=false` also refuses the degenerate inverse-lift neighborhood
whose destination is only the bound-variable leaf.

The analysis uses one attempt-local binder-occurrence/LCA index. Rewriting
preserves binder identities and affected direct-`Abs` HM application roles,
then delegates final target compatibility to `mutate_tree`.

No mutation-specific `ctx` field is required. Expected failure modes are an
unused binder, no strict legal descendant, a disabled trivial destination, an
HM application-role change, or a scoped materialisation error.

The conservative form is part of the ordinary mutation preset and remains
independently removable through the injected mutation list.
"""
struct LetSinkMutation <: AbstractMutation
    allow_trivial::Bool
end

LetSinkMutation(; allow_trivial::Bool=false) = LetSinkMutation(allow_trivial)

required_site_kinds(::LetSinkMutation) = Symbol[:app]

function is_site_eligible(
    mutation::LetSinkMutation,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    return _sink_candidate(mutation, scoped, site, env) !== nothing
end

function apply(
    mutation::LetSinkMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    candidate = _sink_candidate(mutation, scoped, site, env)
    candidate === nothing && return Err(NoEligibleSiteErr())
    rewritten = _sink_rewrite(scoped, site, candidate)
    rewritten isa Err && return rewritten
    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

@testitem "ScopedRewrite preserves nested De Bruijn binding identities" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: _application_role, _materialize_scoped, _resolve_scoped, _scoped_at_path

    original = AstTree(
        AbsNode(
            :x,
            AbsNode(
                :x,
                AppNode(
                    AppNode(IndexNode(0), IndexNode(1)),
                    AppNode(
                        AbsNode(:x, AppNode(IndexNode(0), IndexNode(2))),
                        ConstNode(:seed),
                    ),
                ),
            ),
        ),
    )

    resolved = _resolve_scoped(original)
    @test !(resolved isa LambdaRegression.Utils.Err)
    @test _application_role(_scoped_at_path(resolved, [1, 1, 2])) == :let_like

    materialized = _materialize_scoped(resolved)
    @test materialized == original
    @test materialized.root.name == :x
    @test materialized.root.body.name == :x
    @test materialized.root.body.body.arg.func.name == :x
end

@testitem "ScopedRewrite rejects escaped binder identity" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations: _materialize_scoped, _replace_scoped, _resolve_scoped, _scoped_at_path
    using LambdaRegression.Utils: Err

    resolved = _resolve_scoped(AstTree(AbsNode(:x, IndexNode(0))))
    bound = _scoped_at_path(resolved, [1])
    escaped = _replace_scoped(resolved, Int[], bound)

    result = _materialize_scoped(escaped)
    @test result isa Err
end

@testitem "ScopedRewrite preserves free variables and rejects unresolved root indices" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode, IndexNode, VarNode
    using LambdaRegression.Mutations: _UnresolvedScopedIndexErr, _materialize_scoped, _resolve_scoped
    using LambdaRegression.Utils: Err

    free_tree = AstTree(AppNode(VarNode(:external), ConstNode(:value)))
    free_scoped = _resolve_scoped(free_tree)
    @test !(free_scoped isa Err)
    @test _materialize_scoped(free_scoped) == free_tree

    unresolved = _resolve_scoped(AstTree(IndexNode(0)))
    @test unresolved isa Err
    @test unresolved.value isa _UnresolvedScopedIndexErr
    @test unresolved.value.index == 0
    @test unresolved.value.depth == 0
end

@testitem "ScopedRewrite detects application role changes" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: _application_role, _application_roles_stable, _replace_scoped, _resolve_scoped, _scoped_at_path

    old = _resolve_scoped(
        AstTree(AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:value))),
    )
    old_func = _scoped_at_path(old, [1])
    wrapped_func = LambdaRegression.Mutations._ScopedApp(
        LambdaRegression.Mutations._ScopedConst(:identity),
        old_func,
    )
    new = _replace_scoped(old, [1], wrapped_func)

    @test _application_role(old.root) == :let_like
    @test _application_role(new.root) == :ordinary
    @test !_application_roles_stable(old.root, new.root)
end

@testitem "ScopedRewrite checks retained nested application roles" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: _application_role, _application_roles_stable, _replace_scoped, _resolve_scoped, _scoped_at_path

    old = _resolve_scoped(
        AstTree(
            AppNode(
                ConstNode(:outer),
                AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:value)),
            ),
        ),
    )
    old_inner_func = _scoped_at_path(old, [2, 1])
    wrapped_inner_func = LambdaRegression.Mutations._ScopedApp(
        LambdaRegression.Mutations._ScopedConst(:g),
        old_inner_func,
    )
    new = _replace_scoped(old, [2, 1], wrapped_inner_func)

    @test _application_role(old.root) == :ordinary
    @test _application_role(new.root) == :ordinary
    @test !_application_roles_stable(old, new, [[2] => [2]])
    @test !_application_roles_stable(old, new, [[2] => [1]])
    @test !_application_roles_stable(old, new, [[9] => [2]])
    @test _application_roles_stable(old, old, [Int[] => Int[], [2] => [2]])
end

@testitem "ScopedRewrite rejects duplicate binder identities" begin
    using Test
    using LambdaRegression.Mutations: _DuplicateScopedBinderErr, _materialize_scoped
    using LambdaRegression.Utils: Err

    duplicate = LambdaRegression.Mutations._ScopedTree(
        LambdaRegression.Mutations._ScopedAbs(
            1,
            :outer,
            LambdaRegression.Mutations._ScopedAbs(
                1,
                :inner,
                LambdaRegression.Mutations._ScopedBound(1),
            ),
        ),
    )

    result = _materialize_scoped(duplicate)
    @test result isa Err
    @test result.value isa _DuplicateScopedBinderErr
end

@testitem "ScopedRewrite records the next binder identity" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations: _resolve_scoped

    resolved = _resolve_scoped(
        AstTree(AbsNode(:outer, AbsNode(:inner, IndexNode(1)))),
    )

    @test resolved.next_binder_id == 3
end

@testitem "ScopedRewrite rejects unsupported AST nodes" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode, LetNode
    using LambdaRegression.Mutations: _resolve_scoped
    using LambdaRegression.Utils: Err

    result = _resolve_scoped(
        AstTree(LetNode(:x, ConstNode(:value), ConstNode(:body))),
    )
    @test result isa Err
end

@testitem "LetPruneMutation prunes an unused root let" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations: IndexedProvider, LetPruneMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(:answer => FloatTy, :rhs => IntTy)
    tree = AstTree(
        AppNode(
            AbsNode(:unused, ConstNode(:answer)),
            ConstNode(:rhs),
        ),
    )

    mutated = mutate_tree(
        tree,
        FloatTy;
        rng=MersenneTwister(2000),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated == AstTree(ConstNode(:answer))
    @test infer(mutated, type_ctx) == FloatTy
end

@testitem "LetPruneMutation preserves outer binder indices when nested" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetPruneMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(
        :consume => TyFun(IntTy, IntTy),
        :rhs => BoolTy,
    )
    tree = AstTree(
        AbsNode(
            :outer,
            AppNode(
                AbsNode(
                    :unused,
                    AppNode(ConstNode(:consume), IndexNode(1)),
                ),
                ConstNode(:rhs),
            ),
        ),
    )
    target_ty = TyFun(IntTy, IntTy)

    mutated = mutate_tree(
        tree,
        target_ty;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutated == AstTree(
        AbsNode(:outer, AppNode(ConstNode(:consume), IndexNode(0))),
    )
    @test infer(mutated, type_ctx) == target_ty
end

@testitem "LetPruneMutation rejects a used binder" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetPruneMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(:rhs => IntTy)
    tree = AstTree(
        AppNode(
            AbsNode(:used, IndexNode(0)),
            ConstNode(:rhs),
        ),
    )
    original = deepcopy(tree)

    result = mutate_tree(
        tree,
        IntTy;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
    @test tree == original
end

@testitem "LetPruneMutation distinguishes shadowed binder identities" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetPruneMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :consume => TyFun(IntTy, IntTy),
        :rhs_bool => BoolTy,
        :rhs_int => IntTy,
    )

    shadow_only = AstTree(
        AppNode(
            AbsNode(
                :x,
                AbsNode(:x, AppNode(ConstNode(:consume), IndexNode(0))),
            ),
            ConstNode(:rhs_bool),
        ),
    )
    pruned = mutate_tree(
        shadow_only,
        TyFun(IntTy, IntTy);
        rng=MersenneTwister(8000),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )
    @test pruned == AstTree(
        AbsNode(:x, AppNode(ConstNode(:consume), IndexNode(0))),
    )

    captured_outer = AstTree(
        AppNode(
            AbsNode(
                :x,
                AbsNode(:x, AppNode(ConstNode(:consume), IndexNode(1))),
            ),
            ConstNode(:rhs_int),
        ),
    )
    result = mutate_tree(
        captured_outer,
        infer(captured_outer, type_ctx);
        rng=MersenneTwister(42),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )
    @test result isa Err
end

@testitem "LetPruneMutation rejects retained application role changes transactionally" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetPruneMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(:arg => IntTy, :rhs => IntTy)
    tree = AstTree(
        AppNode(
            AppNode(
                AbsNode(:unused, AbsNode(:value, IndexNode(0))),
                ConstNode(:rhs),
            ),
            ConstNode(:arg),
        ),
    )
    original = deepcopy(tree)

    result = mutate_tree(
        tree,
        IntTy;
        rng=MersenneTwister(1234),
        global_consts_ty=type_ctx,
        mutations=[LetPruneMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetPruneMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa Err
    @test tree == original
end

@testitem "LetPruneMutation apply rejects transactionally" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: AdaptivePolicy, BloatPolicy, LetPruneMutation, MutationEnv, Site, apply
    using LambdaRegression.TypeSystem: IntTy, Ty, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(:arg => IntTy, :rhs => IntTy)
    tree = AstTree(
        AppNode(
            AppNode(
                AbsNode(:unused, AbsNode(:value, IndexNode(0))),
                ConstNode(:rhs),
            ),
            ConstNode(:arg),
        ),
    )
    original = deepcopy(tree)
    site = Site(
        [1],
        tree.root.func,
        tree.root,
        1,
        0,
        :app,
        IntTy,
        Ty[],
    )
    env = MutationEnv(
        MersenneTwister(555),
        IntTy,
        type_ctx,
        tree,
        nothing,
        false,
        IdDict{Any,Any}(),
        AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        BloatPolicy(),
    )

    result = apply(LetPruneMutation(), tree, site, env, (;))
    @test result isa Err
    @test tree == original
end

@testitem "LetHoistMutation moves one binding across application and abstraction" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :make => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :arg => IntTy,
        :one => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetHoistMutation => 1.0),
    )

    across_arg = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
        ),
    )
    hoisted_arg = mutate_tree(
        across_arg,
        IntTy;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test hoisted_arg == AstTree(
        AppNode(
            AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:one),
        ),
    )
    @test infer(hoisted_arg, type_ctx) == IntTy

    function_position = AstTree(
        AppNode(
            AppNode(
                AbsNode(:x, AppNode(ConstNode(:make), IndexNode(0))),
                ConstNode(:one),
            ),
            ConstNode(:arg),
        ),
    )
    hoisted_function = mutate_tree(
        function_position,
        IntTy;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test hoisted_function == AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AppNode(ConstNode(:make), IndexNode(0)),
                    ConstNode(:arg),
                ),
            ),
            ConstNode(:one),
        ),
    )
    @test infer(hoisted_function, type_ctx) == IntTy

    across_abs = AstTree(
        AbsNode(
            :y,
            AppNode(
                AbsNode(
                    :x,
                    AppNode(
                        AppNode(ConstNode(:add), IndexNode(0)),
                        IndexNode(1),
                    ),
                ),
                ConstNode(:one),
            ),
        ),
    )
    target = TyFun(IntTy, IntTy)
    hoisted_abs = mutate_tree(
        across_abs,
        target;
        rng=MersenneTwister(1234),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test hoisted_abs == AstTree(
        AppNode(
            AbsNode(
                :x,
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(ConstNode(:add), IndexNode(1)),
                        IndexNode(0),
                    ),
                ),
            ),
            ConstNode(:one),
        ),
    )
    @test infer(hoisted_abs, type_ctx) == target
end

@testitem "LetHoistMutation rejects capture and HM application-role changes" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :arg => IntTy,
        :one => IntTy,
        :rhs => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetHoistMutation => 1.0),
    )

    captures_crossed = AstTree(
        AbsNode(
            :outer,
            AppNode(AbsNode(:inner, IndexNode(0)), IndexNode(0)),
        ),
    )
    capture_result = mutate_tree(
        captures_crossed,
        infer(captures_crossed, type_ctx);
        rng=MersenneTwister(8000),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test capture_result isa Err

    changes_parent_role = AstTree(
        AppNode(
            AppNode(
                AbsNode(:unused, AbsNode(:value, IndexNode(0))),
                ConstNode(:rhs),
            ),
            ConstNode(:arg),
        ),
    )
    role_result = mutate_tree(
        changes_parent_role,
        IntTy;
        rng=MersenneTwister(42),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test role_result isa Err

    changes_outer_role = AstTree(
        AppNode(
            AbsNode(
                :outer,
                AppNode(
                    AbsNode(
                        :inner,
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(0)),
                            IndexNode(1),
                        ),
                    ),
                    ConstNode(:one),
                ),
            ),
            ConstNode(:arg),
        ),
    )
    outer_role_result = mutate_tree(
        changes_outer_role,
        IntTy;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test outer_role_result isa Err
end

@testitem "LetHoistMutation moves a dependent binding prefix in order" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
    )
    mutation = LetHoistMutation(min_bindings=2, max_bindings=2)
    original = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(
                    :x,
                    AppNode(
                        AbsNode(
                            :y,
                            AppNode(
                                AppNode(ConstNode(:add), IndexNode(0)),
                                IndexNode(1),
                            ),
                        ),
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(0)),
                            ConstNode(:one),
                        ),
                    ),
                ),
                ConstNode(:one),
            ),
        ),
    )
    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(555),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetHoistMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )
    expected = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AbsNode(
                        :y,
                        AppNode(
                            ConstNode(:f),
                            AppNode(
                                AppNode(ConstNode(:add), IndexNode(0)),
                                IndexNode(1),
                            ),
                        ),
                    ),
                    AppNode(
                        AppNode(ConstNode(:add), IndexNode(0)),
                        ConstNode(:one),
                    ),
                ),
            ),
            ConstNode(:one),
        ),
    )
    @test result == expected
    @test infer(result, type_ctx) == IntTy
end

@testitem "LetHoistMutation is deterministic and multi mode is opt-in" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.Mutations: _legal_hoist_sizes, _resolve_scoped, _scoped_let_prefix
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    type_ctx = Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy), :one => IntTy)
    tree = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(
                    :x,
                    AppNode(AbsNode(:y, IndexNode(0)), IndexNode(0)),
                ),
                ConstNode(:one),
            ),
        ),
    )
    mutation = LetHoistMutation(min_bindings=1, max_bindings=2)
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetHoistMutation => 1.0),
    )
    run(seed) = mutate_tree(
        tree,
        IntTy;
        rng=MersenneTwister(seed),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )

    @test run(306) == run(306)
    resolved = _resolve_scoped(tree)
    @test _legal_hoist_sizes(mutation, resolved, [2]) == [1, 2]

    rhs_nested = AstTree(
        AppNode(
            AbsNode(:x, IndexNode(0)),
            AppNode(AbsNode(:rhs_local, IndexNode(0)), ConstNode(:one)),
        ),
    )
    rhs_bindings, _ = _scoped_let_prefix(_resolve_scoped(rhs_nested).root, 3)
    @test length(rhs_bindings) == 1

    triple = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(
                    :x,
                    AppNode(
                        AbsNode(
                            :y,
                            AppNode(
                                AbsNode(:z, IndexNode(0)),
                                IndexNode(0),
                            ),
                        ),
                        IndexNode(0),
                    ),
                ),
                ConstNode(:one),
            ),
        ),
    )
    triple_scoped = _resolve_scoped(triple)
    @test isempty(
        _legal_hoist_sizes(
            LetHoistMutation(min_bindings=2, max_bindings=2),
            triple_scoped,
            [2, 1, 1],
        ),
    )
    @test _legal_hoist_sizes(
        LetHoistMutation(min_bindings=3, max_bindings=3),
        triple_scoped,
        [2],
    ) == [3]
    defaults = LambdaRegression.Mutations._default_mutations()
    @test any(
        m -> m isa LetHoistMutation && m.max_bindings == 1,
        defaults,
    )
    @test any(m -> m isa LambdaRegression.Mutations.LetSinkMutation, defaults)
    @test any(m -> m isa LambdaRegression.Mutations.LetPruneMutation, defaults)
    @test !any(
        m -> m isa LetHoistMutation && m.max_bindings > 1,
        defaults,
    )
end

@testitem "LetHoistMutation apply does not require eligibility cache warmup" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: AdaptivePolicy, BloatPolicy, LetHoistMutation, MutationEnv, Site, apply
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyLike

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    original = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
        ),
    )
    attempt = deepcopy(original)
    site = Site(
        [2],
        attempt.root.arg,
        attempt.root,
        2,
        0,
        :app,
        IntTy,
        Ty[],
    )
    env = MutationEnv(
        MersenneTwister(8000),
        IntTy,
        type_ctx,
        original,
        nothing,
        false,
        IdDict{Any,Any}(),
        AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        BloatPolicy(),
    )

    result = apply(LetHoistMutation(), attempt, site, env, (;))
    @test result == AstTree(
        AppNode(
            AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:one),
        ),
    )
end

@testitem "LetHoistMutation apply validates the rewritten path" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: AdaptivePolicy, BloatPolicy, LetHoistMutation, MutationEnv, Site, apply
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :one => IntTy,
        :two => IntTy,
    )
    tree = AstTree(
        AppNode(
            AppNode(
                AbsNode(:x, AbsNode(:y, IndexNode(0))),
                ConstNode(:one),
            ),
            AppNode(AbsNode(:z, IndexNode(0)), ConstNode(:two)),
        ),
    )
    # The object fields name the legal argument let, but the stable path names
    # the illegal function let whose Hoist would change the root HM role.
    inconsistent_site = Site(
        [1],
        tree.root.arg,
        tree.root,
        2,
        0,
        :app,
        IntTy,
        Ty[],
    )
    env = MutationEnv(
        MersenneTwister(555),
        IntTy,
        type_ctx,
        tree,
        nothing,
        false,
        IdDict{Any,Any}(),
        AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        BloatPolicy(),
    )

    result = apply(LetHoistMutation(), tree, inconsistent_site, env, (;))
    @test result isa Err
end

@testitem "LetHoistMutation accepts higher-order and recursive types" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, infer, iscompatible, make_tyvar_factory

    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetHoistMutation => 1.0),
    )
    higher_ctx = Dict{Symbol,TyLike}(
        :neg => TyFun(IntTy, IntTy),
        :use => TyFun(TyFun(IntTy, IntTy), IntTy),
    )
    higher = AstTree(
        AppNode(
            ConstNode(:neg),
            AppNode(
                AbsNode(:fn, AppNode(ConstNode(:use), IndexNode(0))),
                AbsNode(:value, IndexNode(0)),
            ),
        ),
    )
    higher_result = mutate_tree(
        higher,
        IntTy;
        rng=MersenneTwister(1234),
        global_consts_ty=higher_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test higher_result == AstTree(
        AppNode(
            AbsNode(
                :fn,
                AppNode(
                    ConstNode(:neg),
                    AppNode(ConstNode(:use), IndexNode(0)),
                ),
            ),
            AbsNode(:value, IndexNode(0)),
        ),
    )

    fresh = make_tyvar_factory()
    binder = fresh(0)
    recursive = TyMu(binder, TyFun(IntTy, binder))
    recursive_ctx = Dict{Symbol,TyLike}(
        :consume => TyFun(recursive, recursive),
        :seed => recursive,
    )
    recursive_tree = AstTree(
        AppNode(
            ConstNode(:consume),
            AppNode(AbsNode(:value, IndexNode(0)), ConstNode(:seed)),
        ),
    )
    recursive_result = mutate_tree(
        recursive_tree,
        recursive;
        rng=MersenneTwister(8000),
        global_consts_ty=recursive_ctx,
        mutations=[LetHoistMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test recursive_result == AstTree(
        AppNode(
            AbsNode(
                :value,
                AppNode(ConstNode(:consume), IndexNode(0)),
            ),
            ConstNode(:seed),
        ),
    )
    @test iscompatible(infer(recursive_result, recursive_ctx), recursive)
end

@testitem "LetSinkMutation moves to the maximal application and abstraction region" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :f => TyFun(IntTy, IntTy),
        :g => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
    )

    application = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    ConstNode(:f),
                    AppNode(ConstNode(:g), IndexNode(0)),
                ),
            ),
            ConstNode(:one),
        ),
    )
    sunk_application = mutate_tree(
        application,
        IntTy;
        rng=MersenneTwister(555),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test sunk_application == AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(:x, AppNode(ConstNode(:g), IndexNode(0))),
                ConstNode(:one),
            ),
        ),
    )

    abstraction = AstTree(
        AppNode(
            AbsNode(
                :x,
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(ConstNode(:add), IndexNode(1)),
                        IndexNode(0),
                    ),
                ),
            ),
            ConstNode(:one),
        ),
    )
    target = TyFun(IntTy, IntTy)
    sunk_abstraction = mutate_tree(
        abstraction,
        target;
        rng=MersenneTwister(2000),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test sunk_abstraction == AstTree(
        AbsNode(
            :y,
            AppNode(
                AppNode(
                    AbsNode(:x, AppNode(ConstNode(:add), IndexNode(0))),
                    ConstNode(:one),
                ),
                IndexNode(0),
            ),
        ),
    )
    @test infer(sunk_abstraction, type_ctx) == target
end

@testitem "LetSinkMutation rejects split, unused, and HM-role-invalid regions" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :arg => IntTy,
        :one => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
    )
    run(tree, seed) = mutate_tree(
        tree,
        infer(tree, type_ctx);
        rng=MersenneTwister(seed),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )

    split = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AppNode(ConstNode(:add), IndexNode(0)),
                    IndexNode(0),
                ),
            ),
            ConstNode(:one),
        ),
    )
    unused = AstTree(
        AppNode(AbsNode(:x, ConstNode(:one)), ConstNode(:one)),
    )
    source_role = AstTree(
        AppNode(
            AppNode(
                AbsNode(:x, AbsNode(:value, IndexNode(0))),
                ConstNode(:one),
            ),
            ConstNode(:arg),
        ),
    )
    target_role = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(AbsNode(:value, IndexNode(1)), ConstNode(:arg)),
            ),
            ConstNode(:one),
        ),
    )

    @test run(split, 503) isa Err
    @test run(unused, 504) isa Err
    @test run(source_role, 505) isa Err
    @test run(target_role, 506) isa Err
end

@testitem "LetSinkMutation trivial policy and inner let crossing are explicit" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :combine => TyFun(IntTy, TyFun(TyFun(IntTy, IntTy), IntTy)),
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
        :two => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
    )

    trivial = AstTree(
        AppNode(
            AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:one),
        ),
    )
    default_result = mutate_tree(
        trivial,
        IntTy;
        rng=MersenneTwister(42),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test default_result isa Err

    trivial_mutation = LetSinkMutation(allow_trivial=true)
    trivial_result = mutate_tree(
        trivial,
        IntTy;
        rng=MersenneTwister(555),
        global_consts_ty=type_ctx,
        mutations=[trivial_mutation],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )
    @test trivial_result == AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
        ),
    )

    duplicate_indices = AstTree(
        AppNode(
            AppNode(
                ConstNode(:combine),
                AppNode(
                    AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))),
                    ConstNode(:one),
                ),
            ),
            AbsNode(:other, IndexNode(0)),
        ),
    )
    duplicate_result = mutate_tree(
        duplicate_indices,
        IntTy;
        rng=MersenneTwister(8000),
        global_consts_ty=type_ctx,
        mutations=[trivial_mutation],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )
    @test duplicate_result == AstTree(
        AppNode(
            AppNode(
                ConstNode(:combine),
                AppNode(
                    ConstNode(:f),
                    AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one)),
                ),
            ),
            AbsNode(:other, IndexNode(0)),
        ),
    )

    crosses_inner = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AbsNode(
                        :y,
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(1)),
                            IndexNode(0),
                        ),
                    ),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:one),
        ),
    )
    crossed = mutate_tree(
        crosses_inner,
        IntTy;
        rng=MersenneTwister(2000),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test crossed == AstTree(
        AppNode(
            AbsNode(
                :y,
                AppNode(
                    AppNode(
                        AbsNode(:x, AppNode(ConstNode(:add), IndexNode(0))),
                        ConstNode(:one),
                    ),
                    IndexNode(0),
                ),
            ),
            ConstNode(:two),
        ),
    )
end

@testitem "LetSinkMutation preserves polymorphic let use" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, TyFun, TyLike, infer

    type_ctx = Dict{Symbol,TyLike}(
        :bool => BoolTy,
        :combine => TyFun(IntTy, TyFun(BoolTy, FloatTy)),
        :int => IntTy,
        :outer => TyFun(FloatTy, FloatTy),
    )
    tree = AstTree(
        AppNode(
            AbsNode(
                :id,
                AppNode(
                    ConstNode(:outer),
                    AppNode(
                        AppNode(
                            ConstNode(:combine),
                            AppNode(IndexNode(0), ConstNode(:int)),
                        ),
                        AppNode(IndexNode(0), ConstNode(:bool)),
                    ),
                ),
            ),
            AbsNode(:value, IndexNode(0)),
        ),
    )
    result = mutate_tree(
        tree,
        FloatTy;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(
            ConstNode(:outer),
            AppNode(
                AbsNode(
                    :id,
                    AppNode(
                        AppNode(
                            ConstNode(:combine),
                            AppNode(IndexNode(0), ConstNode(:int)),
                        ),
                        AppNode(IndexNode(0), ConstNode(:bool)),
                    ),
                ),
                AbsNode(:value, IndexNode(0)),
            ),
        ),
    )
    @test infer(result, type_ctx) == FloatTy
end

@testitem "LetSinkMutation bypasses illegal intermediate roles" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    type_ctx = Dict{Symbol,TyLike}(
        :arg => IntTy,
        :f => TyFun(IntTy, IntTy),
        :g => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    tree = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    ConstNode(:g),
                    AppNode(
                        AbsNode(
                            :y,
                            AppNode(ConstNode(:f), IndexNode(1)),
                        ),
                        ConstNode(:arg),
                    ),
                ),
            ),
            ConstNode(:one),
        ),
    )
    result = mutate_tree(
        tree,
        IntTy;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(
            ConstNode(:g),
            AppNode(
                AbsNode(
                    :y,
                    AppNode(
                        AbsNode(:x, AppNode(ConstNode(:f), IndexNode(0))),
                        ConstNode(:one),
                    ),
                ),
                ConstNode(:arg),
            ),
        ),
    )
end

@testitem "LetSinkMutation handles inner let RHS and split uses" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :f => TyFun(IntTy, IntTy),
        :g => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
    )

    rhs_only = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AbsNode(:y, AppNode(ConstNode(:f), IndexNode(0))),
                    AppNode(ConstNode(:g), IndexNode(0)),
                ),
            ),
            ConstNode(:one),
        ),
    )
    rhs_result = mutate_tree(
        rhs_only,
        IntTy;
        rng=MersenneTwister(1234),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test rhs_result == AstTree(
        AppNode(
            AbsNode(:y, AppNode(ConstNode(:f), IndexNode(0))),
            AppNode(
                AbsNode(:x, AppNode(ConstNode(:g), IndexNode(0))),
                ConstNode(:one),
            ),
        ),
    )

    split = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    AbsNode(
                        :y,
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(1)),
                            IndexNode(0),
                        ),
                    ),
                    AppNode(ConstNode(:g), IndexNode(0)),
                ),
            ),
            ConstNode(:one),
        ),
    )
    split_result = mutate_tree(
        split,
        infer(split, type_ctx);
        rng=MersenneTwister(8000),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=chooser,
        provider=IndexedProvider(),
        max_try=1,
    )
    @test split_result isa Err
end

@testitem "LetHoistMutation and LetSinkMutation round trip controlled regions" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetHoistMutation, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :g => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    outer = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    ConstNode(:f),
                    AppNode(ConstNode(:g), IndexNode(0)),
                ),
            ),
            ConstNode(:one),
        ),
    )
    inner = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(:x, AppNode(ConstNode(:g), IndexNode(0))),
                ConstNode(:one),
            ),
        ),
    )

    function run(tree, mutation, seed)
        return mutate_tree(
            tree,
            IntTy;
            rng=MersenneTwister(seed),
            global_consts_ty=type_ctx,
            mutations=[mutation],
            chooser=WeightedPairChooser(
                weights=Dict{DataType,Float64}(typeof(mutation) => 1.0),
            ),
            provider=IndexedProvider(),
            max_try=1,
        )
    end

    @test run(outer, LetSinkMutation(), 514) == inner
    @test run(inner, LetHoistMutation(), 515) == outer
    @test run(run(outer, LetSinkMutation(), 516), LetHoistMutation(), 517) == outer
    @test run(run(inner, LetHoistMutation(), 518), LetSinkMutation(), 519) == inner
end

@testitem "LetSinkMutation apply does not require eligibility cache warmup" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: AdaptivePolicy, BloatPolicy, LetSinkMutation, MutationEnv, Site, apply
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyLike

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :g => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    original = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(
                    ConstNode(:f),
                    AppNode(ConstNode(:g), IndexNode(0)),
                ),
            ),
            ConstNode(:one),
        ),
    )
    attempt = deepcopy(original)
    site = Site(Int[], attempt.root, nothing, 0, 0, :app, IntTy, Ty[])
    env = MutationEnv(
        MersenneTwister(42),
        IntTy,
        type_ctx,
        original,
        nothing,
        false,
        IdDict{Any,Any}(),
        AdaptivePolicy(),
        IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
        BloatPolicy(),
    )

    result = apply(LetSinkMutation(), attempt, site, env, (;))
    @test result == AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(:x, AppNode(ConstNode(:g), IndexNode(0))),
                ConstNode(:one),
            ),
        ),
    )
end

@testitem "LetSinkMutation accepts an equirecursive target" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations: IndexedProvider, LetSinkMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, infer, iscompatible, make_tyvar_factory

    fresh = make_tyvar_factory()
    binder = fresh(0)
    recursive = TyMu(binder, TyFun(IntTy, binder))
    type_ctx = Dict{Symbol,TyLike}(
        :consume => TyFun(recursive, recursive),
        :outer => TyFun(recursive, recursive),
        :seed => recursive,
    )
    tree = AstTree(
        AppNode(
            AbsNode(
                :value,
                AppNode(
                    ConstNode(:outer),
                    AppNode(ConstNode(:consume), IndexNode(0)),
                ),
            ),
            ConstNode(:seed),
        ),
    )
    result = mutate_tree(
        tree,
        recursive;
        rng=MersenneTwister(555),
        global_consts_ty=type_ctx,
        mutations=[LetSinkMutation()],
        chooser=WeightedPairChooser(
            weights=Dict{DataType,Float64}(LetSinkMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(
            ConstNode(:outer),
            AppNode(
                AbsNode(
                    :value,
                    AppNode(ConstNode(:consume), IndexNode(0)),
                ),
                ConstNode(:seed),
            ),
        ),
    )
    @test iscompatible(infer(result, type_ctx), recursive)
end
