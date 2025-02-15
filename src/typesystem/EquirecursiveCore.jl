"""
    type_constructor_tag(t::Ty) -> Symbol

Return the stable constructor tag used by recursive equality and unification.
Extensions that add a finite-arity `Ty` subtype implement this method together
with [`type_static_parameters`](@ref) and [`type_children`](@ref).
"""
function type_constructor_tag(t::Ty)
    throw(ArgumentError("no type-constructor protocol for $(typeof(t))"))
end

"""Return non-child data that participates in constructor identity."""
type_static_parameters(::Ty)::Tuple = ()

"""Return constructor children in semantic order."""
type_children(::Ty)::Tuple = ()

"""
    type_node_identity(t::Ty) -> Any

Optionally return a cheap, stable identity key for an immutable extension node
that may be shared across non-sibling DAG positions. The key must uniquely
identify that node during a query and remain unchanged for the node's lifetime.
Mutable nodes use physical identity; ordinary immutable trees may keep the
default `nothing`.
"""
type_node_identity(::Ty) = nothing

"""
    rebuild_type(prototype::Ty, children::Tuple) -> Ty

Rebuild one constructor node from semantic children. Extensions implementing a
finite-arity `Ty` subtype provide this method together with the constructor
protocol above.
"""
function rebuild_type(t::Ty, ::Tuple)
    throw(ArgumentError("no type-constructor rebuild protocol for $(typeof(t))"))
end

type_constructor_tag(::TyConst) = :Atom
type_static_parameters(t::TyConst) = (t.name,)
rebuild_type(t::TyConst, ::Tuple{}) = t

type_constructor_tag(::TyRigid) = :Rigid
type_static_parameters(t::TyRigid) = (t.id,)
rebuild_type(t::TyRigid, ::Tuple{}) = t

type_constructor_tag(::TyNat) = :Nat
type_static_parameters(t::TyNat) = (t.value,)
rebuild_type(t::TyNat, ::Tuple{}) = t

type_constructor_tag(::TyFun) = :Fun
type_children(t::TyFun) = (t.arg, t.res)
rebuild_type(::TyFun, children::Tuple{Ty,Ty}) = TyFun(children...)

type_constructor_tag(::TyProd) = :Prod
type_children(t::TyProd) = (t.left, t.right)
rebuild_type(::TyProd, children::Tuple{Ty,Ty}) = TyProd(children...)

type_constructor_tag(::TySum) = :Sum
type_children(t::TySum) = (t.left, t.right)
rebuild_type(::TySum, children::Tuple{Ty,Ty}) = TySum(children...)

type_constructor_tag(::TyList) = :List
type_children(t::TyList) = (t.elem,)
rebuild_type(::TyList, children::Tuple{Ty}) = TyList(children[1])

type_constructor_tag(::TyArray) = :Array
type_children(t::TyArray) = (t.len, t.elem)
rebuild_type(::TyArray, children::Tuple{Ty,Ty}) = TyArray(children...)

type_constructor_tag(::TyMu) = :Mu
type_children(t::TyMu) = (t.body,)
rebuild_type(t::TyMu, children::Tuple{Ty}) = TyMu(t.var, children[1])

mutable struct _MuEqNode
    tag::Symbol
    parameters::Tuple
    children::Vector{Int}
    filled::Bool
end

_MuEqNode() = _MuEqNode(:Uninitialised, (), Int[], false)

struct _MuEqGraph
    nodes::Vector{_MuEqNode}
    root::Int
end

const _CONSTRUCTOR_MEMO_THRESHOLD = 64

mutable struct _ConstructorMemo
    mutable_entries::IdDict{Ty,Tuple{Int,Int}}
    keyed_entries::Dict{Tuple{DataType,Any},Tuple{Int,Int}}
end

_ConstructorMemo() = _ConstructorMemo(
    IdDict{Ty,Tuple{Int,Int}}(),
    Dict{Tuple{DataType,Any},Tuple{Int,Int}}(),
)

mutable struct _MuEqBuilder
    nodes::Vector{_MuEqNode}
    binders::Dict{TyVar,Int}
    resolving::Dict{TyVar,Nothing}
    constructors::Union{Nothing,_ConstructorMemo}
    scope::Int
    next_scope::Int
end

_MuEqBuilder() = _MuEqBuilder(
    _MuEqNode[],
    Dict{TyVar,Int}(),
    Dict{TyVar,Nothing}(),
    nothing,
    0,
    0,
)

@inline function _constructor_node(index::_ConstructorMemo, ty::Ty, scope::Int)
    if Base.ismutabletype(typeof(ty))
        entry = get(index.mutable_entries, ty, nothing)
        entry !== nothing && entry[1] == scope && return entry[2]
        return 0
    end
    identity = type_node_identity(ty)
    identity === nothing && return 0
    entry = get(index.keyed_entries, (typeof(ty), identity), nothing)
    return entry !== nothing && entry[1] == scope ? entry[2] : 0
end

@inline function _remember_constructor_node!(index::_ConstructorMemo, ty::Ty, scope::Int, root::Int)
    if Base.ismutabletype(typeof(ty))
        index.mutable_entries[ty] = (scope, root)
    else
        identity = type_node_identity(ty)
        identity === nothing || (index.keyed_entries[(typeof(ty), identity)] = (scope, root))
    end
    return root
end

function _new_mu_eq_node!(builder::_MuEqBuilder, node::_MuEqNode=_MuEqNode())::Int
    push!(builder.nodes, node)
    return length(builder.nodes)
end

function _with_mu_binder(f, builder::_MuEqBuilder, binder::TyVar, root::Int)
    had_old = haskey(builder.binders, binder)
    old = had_old ? builder.binders[binder] : 0
    old_scope = builder.scope
    builder.next_scope += 1
    builder.scope = builder.next_scope
    builder.binders[binder] = root
    try
        return f()
    finally
        builder.scope = old_scope
        if had_old
            builder.binders[binder] = old
        else
            delete!(builder.binders, binder)
        end
    end
end

function _elaborate_mu!(builder::_MuEqBuilder, mu::TyMu)::Int
    scope = builder.scope
    index = builder.constructors
    if index === nothing && length(builder.nodes) >= _CONSTRUCTOR_MEMO_THRESHOLD
        index = _ConstructorMemo()
        builder.constructors = index
    end
    known = index === nothing ? 0 : _constructor_node(index, mu, scope)
    known != 0 && return known
    root = _new_mu_eq_node!(builder)
    index === nothing || _remember_constructor_node!(index, mu, scope, root)
    body_root = _with_mu_binder(builder, mu.var, root) do
        _elaborate_mu_eq!(builder, mu.body)
    end

    body_root == root && throw(ArgumentError("non-contractive recursive type μx.x"))
    body = builder.nodes[body_root]
    body.filled || throw(ArgumentError("recursive binder is not guarded by a type constructor"))
    builder.nodes[root] = _MuEqNode(body.tag, body.parameters, copy(body.children), true)
    return root
end

function _elaborate_mu_var!(builder::_MuEqBuilder, var::TyVar)::Int
    binder_root = get(builder.binders, var, 0)
    binder_root != 0 && return binder_root
    var.instance === nothing && throw(ArgumentError("recursive equality requires closed types"))
    haskey(builder.resolving, var) &&
        throw(ArgumentError("constructor-free type-variable link cycle"))

    builder.resolving[var] = nothing
    try
        return _elaborate_mu_eq!(builder, var.instance)
    finally
        delete!(builder.resolving, var)
    end
end

function _elaborate_mu_eq!(builder::_MuEqBuilder, ty::Ty)::Int
    ty isa TyVar && return _elaborate_mu_var!(builder, ty)
    ty isa TyMu && return _elaborate_mu!(builder, ty)

    tag = type_constructor_tag(ty)
    parameters = type_static_parameters(ty)
    scope = builder.scope
    index = builder.constructors
    if index === nothing && length(builder.nodes) >= _CONSTRUCTOR_MEMO_THRESHOLD
        index = _ConstructorMemo()
        builder.constructors = index
    end
    known = index === nothing ? 0 : _constructor_node(index, ty, scope)
    known != 0 && return known
    root = _new_mu_eq_node!(builder, _MuEqNode(tag, parameters, Int[], true))
    index === nothing || _remember_constructor_node!(index, ty, scope, root)
    children = type_children(ty)
    for i in eachindex(children)
        child = children[i]
        child isa Ty || throw(ArgumentError("type-constructor children must be Ty values"))
        duplicate = 0
        for j in 1:(i - 1)
            if children[j] === child
                duplicate = builder.nodes[root].children[j]
                break
            end
        end
        push!(
            builder.nodes[root].children,
            duplicate == 0 ? _elaborate_mu_eq!(builder, child) : duplicate,
        )
    end
    return root
end

function _mu_eq_graph(ty::Ty)::_MuEqGraph
    builder = _MuEqBuilder()
    root = _elaborate_mu_eq!(builder, ty)
    builder.nodes[root].filled || throw(ArgumentError("recursive type has no constructor root"))
    return _MuEqGraph(builder.nodes, root)
end

"""
    isequal_mu(a::Ty, b::Ty) -> Bool

Compare closed contractive types modulo equi-recursive unfolding. The comparison
uses finite cyclic graphs and memoised stable node pairs; it never chooses an
unfolding depth and never mutates either input.

An unbound inference variable or a constructor-free written `μ` cycle is not a
closed contractive type and raises `ArgumentError`.
"""
function isequal_mu(a::Ty, b::Ty)::Bool
    left = _mu_eq_graph(a)
    right = _mu_eq_graph(b)
    todo = Tuple{Int,Int}[(left.root, right.root)]
    seen = Set{Tuple{Int,Int}}()

    while !isempty(todo)
        pair = pop!(todo)
        pair in seen && continue
        push!(seen, pair)

        l = left.nodes[pair[1]]
        r = right.nodes[pair[2]]
        l.tag == r.tag || return false
        l.parameters == r.parameters || return false
        length(l.children) == length(r.children) || return false

        for i in eachindex(l.children)
            push!(todo, (l.children[i], r.children[i]))
        end
    end
    return true
end

function _mu_bisimulation_classes(graph::_MuEqGraph)::Vector{Int}
    count = length(graph.nodes)
    classes = zeros(Int, count)
    initial = Dict{Any,Int}()
    for node_id in 1:count
        node = graph.nodes[node_id]
        signature = (node.tag, node.parameters, length(node.children))
        classes[node_id] = get!(initial, signature) do
            length(initial) + 1
        end
    end

    while true
        refined = zeros(Int, count)
        signatures = Dict{Any,Int}()
        for node_id in 1:count
            node = graph.nodes[node_id]
            child_classes = tuple((classes[child] for child in node.children)...)
            signature = (node.tag, node.parameters, child_classes)
            refined[node_id] = get!(signatures, signature) do
                length(signatures) + 1
            end
        end
        refined == classes && return classes
        classes = refined
    end
end

function _semantic_mu_graph_key(graph::_MuEqGraph)
    classes = _mu_bisimulation_classes(graph)
    representatives = Dict{Int,Int}()
    for node_id in eachindex(graph.nodes)
        get!(representatives, classes[node_id], node_id)
    end

    root_class = classes[graph.root]
    canonical_ids = Dict{Int,Int}(root_class => 0)
    queue = Int[root_class]
    records = Any[]
    cursor = 1
    while cursor <= length(queue)
        class = queue[cursor]
        cursor += 1
        node = graph.nodes[representatives[class]]
        child_classes = Int[classes[child] for child in node.children]
        for child_class in child_classes
            if !haskey(canonical_ids, child_class)
                canonical_ids[child_class] = length(canonical_ids)
                push!(queue, child_class)
            end
        end
        child_ids = tuple((canonical_ids[child] for child in child_classes)...)
        push!(records, (node.tag, node.parameters, child_ids))
    end
    return (:equirecursive, tuple(records...))
end

"""
    semantic_typekey(t::Ty) -> Any

Return a canonical key for a closed type modulo equirecursive equality. Unlike
[`typekey`](@ref), this key identifies finite unfoldings and different written
periods of the same contractive `μ` type. Open inference metas are rejected.
"""
semantic_typekey(t::Ty) = _semantic_mu_graph_key(_mu_eq_graph(t))

# Query-local graph used by unification. Written μ binders elaborate to fixed
# constructor back edges; only `_GRAPH_META` nodes may acquire links.
const _GRAPH_META = UInt8(1)
const _GRAPH_RIGID = UInt8(2)
const _GRAPH_CON = UInt8(3)

mutable struct _TypeGraphNode
    kind::UInt8
    tag::Symbol
    parameters::Tuple
    children::Vector{Int}
    link::Int
    level::Int
    origin::Union{Nothing,TyVar}
    prototype::Union{Nothing,Ty}
end

_meta_node(var::TyVar) =
    _TypeGraphNode(_GRAPH_META, :Meta, (), Int[], 0, var.level, var, nothing)
_rigid_node(rigid::TyRigid) =
    _TypeGraphNode(_GRAPH_RIGID, :Rigid, (rigid.id,), Int[], 0, 0, nothing, rigid)
_con_node(t::Ty) =
    _TypeGraphNode(_GRAPH_CON, type_constructor_tag(t), type_static_parameters(t), Int[], 0, 0, nothing, t)
_placeholder_node() =
    _TypeGraphNode(_GRAPH_CON, :Uninitialised, (), Int[], 0, 0, nothing, nothing)

struct _TypeElaborationError <: Exception
    message::String
end

mutable struct _TypeGraph
    nodes::Vector{_TypeGraphNode}
    variables::Dict{TyVar,Int}
    binders::Dict{TyVar,Int}
    binder_used::Dict{TyVar,Bool}
    resolving::Dict{TyVar,Nothing}
    rigids::Dict{UInt,Int}
    preferred_binders::Dict{Int,TyVar}
    constructors::Union{Nothing,_ConstructorMemo}
    scope::Int
    next_scope::Int
end

_TypeGraph() = _TypeGraph(
    _TypeGraphNode[],
    Dict{TyVar,Int}(),
    Dict{TyVar,Int}(),
    Dict{TyVar,Bool}(),
    Dict{TyVar,Nothing}(),
    Dict{UInt,Int}(),
    Dict{Int,TyVar}(),
    nothing,
    0,
    0,
)

function _push_graph_node!(graph::_TypeGraph, node::_TypeGraphNode)::Int
    push!(graph.nodes, node)
    return length(graph.nodes)
end

function _elaborate_graph_var!(graph::_TypeGraph, var::TyVar)::Int
    binder = get(graph.binders, var, 0)
    if binder != 0
        graph.binder_used[var] = true
        return binder
    end

    known = get(graph.variables, var, 0)
    known != 0 && return known

    if var.instance === nothing
        node = _push_graph_node!(graph, _meta_node(var))
        graph.variables[var] = node
        return node
    end

    haskey(graph.resolving, var) &&
        throw(_TypeElaborationError("constructor-free type-variable link cycle"))
    if var.instance isa TyVar && haskey(graph.resolving, var.instance)
        throw(_TypeElaborationError("constructor-free type-variable link cycle"))
    end
    graph.resolving[var] = nothing
    try
        node = _elaborate_graph!(graph, var.instance)
        graph.variables[var] = node
        return node
    finally
        delete!(graph.resolving, var)
    end
end

function _elaborate_graph_mu!(graph::_TypeGraph, mu::TyMu)::Int
    scope = graph.scope
    index = graph.constructors
    if index === nothing && length(graph.nodes) >= _CONSTRUCTOR_MEMO_THRESHOLD
        index = _ConstructorMemo()
        graph.constructors = index
    end
    known = index === nothing ? 0 : _constructor_node(index, mu, scope)
    known != 0 && return known
    root = _push_graph_node!(graph, _placeholder_node())
    index === nothing || _remember_constructor_node!(index, mu, scope, root)
    had_binder = haskey(graph.binders, mu.var)
    old_binder = had_binder ? graph.binders[mu.var] : 0
    had_used = haskey(graph.binder_used, mu.var)
    old_used = had_used ? graph.binder_used[mu.var] : false
    old_scope = graph.scope
    graph.next_scope += 1
    graph.scope = graph.next_scope
    graph.binders[mu.var] = root
    graph.binder_used[mu.var] = false

    local body_root::Int
    local used::Bool
    try
        body_root = _elaborate_graph!(graph, mu.body)
        used = graph.binder_used[mu.var]
    finally
        graph.scope = old_scope
        if had_binder
            graph.binders[mu.var] = old_binder
        else
            delete!(graph.binders, mu.var)
        end
        if had_used
            graph.binder_used[mu.var] = old_used
        else
            delete!(graph.binder_used, mu.var)
        end
    end

    # A vacuous binder denotes its body. A used binder must return through a
    # constructor root; the naked μx.x case therefore cannot enter the store.
    used || return body_root
    body_root == root && throw(_TypeElaborationError("non-contractive recursive type μx.x"))
    body = graph.nodes[body_root]
    (body.kind == _GRAPH_CON && body.prototype !== nothing) ||
        throw(_TypeElaborationError("recursive binder is not guarded by a type constructor"))
    graph.nodes[root] = _TypeGraphNode(
        _GRAPH_CON,
        body.tag,
        body.parameters,
        copy(body.children),
        0,
        0,
        nothing,
        body.prototype,
    )
    graph.preferred_binders[root] = mu.var
    return root
end

function _elaborate_graph!(graph::_TypeGraph, ty::Ty)::Int
    ty isa TyVar && return _elaborate_graph_var!(graph, ty)
    ty isa TyMu && return _elaborate_graph_mu!(graph, ty)
    if ty isa TyRigid
        return get!(graph.rigids, ty.id) do
            _push_graph_node!(graph, _rigid_node(ty))
        end
    end

    scope = graph.scope
    index = graph.constructors
    if index === nothing && length(graph.nodes) >= _CONSTRUCTOR_MEMO_THRESHOLD
        index = _ConstructorMemo()
        graph.constructors = index
    end
    known = index === nothing ? 0 : _constructor_node(index, ty, scope)
    known != 0 && return known
    root = _push_graph_node!(graph, _con_node(ty))
    index === nothing || _remember_constructor_node!(index, ty, scope, root)
    children = type_children(ty)
    for i in eachindex(children)
        child = children[i]
        child isa Ty || throw(ArgumentError("type-constructor children must be Ty values"))
        duplicate = 0
        for j in 1:(i - 1)
            if children[j] === child
                duplicate = graph.nodes[root].children[j]
                break
            end
        end
        push!(
            graph.nodes[root].children,
            duplicate == 0 ? _elaborate_graph!(graph, child) : duplicate,
        )
    end
    return root
end

function _resolve_graph!(graph::_TypeGraph, root::Int)::Int
    node = root
    path = Int[]
    while true
        current = graph.nodes[node]
        (current.kind == _GRAPH_META && current.link != 0) || break
        push!(path, node)
        node = current.link
    end
    for item in path
        graph.nodes[item].link = node
    end
    return node
end

function _lower_graph_levels!(graph::_TypeGraph, root::Int, level::Int)
    todo = Int[root]
    seen = Set{Int}()
    while !isempty(todo)
        node_id = pop!(todo)
        node_id in seen && continue
        push!(seen, node_id)
        node = graph.nodes[node_id]
        if node.kind == _GRAPH_META
            if node.link == 0
                node.level > level && (node.level = level)
            else
                push!(todo, node.link)
            end
        elseif node.kind == _GRAPH_CON
            append!(todo, node.children)
        end
    end
    return nothing
end

function _bind_graph_meta!(graph::_TypeGraph, meta_id::Int, target_id::Int)
    meta_id = _resolve_graph!(graph, meta_id)
    target_id = _resolve_graph!(graph, target_id)
    meta_id == target_id && return nothing

    meta = graph.nodes[meta_id]
    target = graph.nodes[target_id]
    @assert meta.kind == _GRAPH_META && meta.link == 0

    if target.kind == _GRAPH_META && target.link == 0
        representative, alias = meta_id < target_id ? (meta_id, target_id) : (target_id, meta_id)
        common_level = min(meta.level, target.level)
        graph.nodes[representative].level = common_level
        graph.nodes[alias].level = common_level
        graph.nodes[alias].link = representative
        return nothing
    end

    _lower_graph_levels!(graph, target_id, meta.level)
    meta.link = target_id
    return nothing
end

function _unify_graph_nodes!(graph::_TypeGraph, left::Int, right::Int)::Bool
    todo = Tuple{Int,Int}[(left, right)]
    seen = Set{Tuple{Int,Int}}()

    while !isempty(todo)
        u, v = pop!(todo)
        u = _resolve_graph!(graph, u)
        v = _resolve_graph!(graph, v)
        u == v && continue

        unode = graph.nodes[u]
        vnode = graph.nodes[v]
        if unode.kind == _GRAPH_META
            _bind_graph_meta!(graph, u, v)
            continue
        elseif vnode.kind == _GRAPH_META
            _bind_graph_meta!(graph, v, u)
            continue
        end

        pair = u < v ? (u, v) : (v, u)
        pair in seen && continue
        push!(seen, pair)

        unode.kind == vnode.kind || return false
        unode.tag == vnode.tag || return false
        unode.parameters == vnode.parameters || return false
        length(unode.children) == length(vnode.children) || return false
        for i in eachindex(unode.children)
            push!(todo, (unode.children[i], vnode.children[i]))
        end
    end
    return true
end

mutable struct _MaterializeState
    active_meta::Dict{Int,Tuple{TyVar,Base.RefValue{Bool}}}
    active_con::Dict{Int,Tuple{TyVar,Base.RefValue{Bool}}}
end

_MaterializeState() = _MaterializeState(
    Dict{Int,Tuple{TyVar,Base.RefValue{Bool}}}(),
    Dict{Int,Tuple{TyVar,Base.RefValue{Bool}}}(),
)

function _materialize_graph!(graph::_TypeGraph, node_id::Int, state::_MaterializeState)::Ty
    node = graph.nodes[node_id]
    if node.kind == _GRAPH_META
        node.link == 0 && return node.origin::TyVar
        if haskey(state.active_meta, node_id)
            binder, recursive = state.active_meta[node_id]
            recursive[] = true
            return binder
        end

        # A syntax binder is rigid and must never reuse the flexible facade meta
        # whose solution is being committed.
        binder = TyVar(-(node.origin::TyVar).id, node.level, nothing)
        recursive = Ref(false)
        state.active_meta[node_id] = (binder, recursive)
        body = _materialize_graph!(graph, node.link, state)
        delete!(state.active_meta, node_id)
        return recursive[] ? TyMu(binder, body) : body
    elseif node.kind == _GRAPH_RIGID
        return node.prototype::TyRigid
    end

    if haskey(state.active_con, node_id)
        binder, recursive = state.active_con[node_id]
        recursive[] = true
        return binder
    end

    preferred = get(graph.preferred_binders, node_id, nothing)
    binder = preferred === nothing ?
        TyVar(-node_id, 0, nothing) :
        TyVar(preferred.id, preferred.level, nothing)
    recursive = Ref(false)
    state.active_con[node_id] = (binder, recursive)
    children = Ty[_materialize_graph!(graph, child, state) for child in node.children]
    delete!(state.active_con, node_id)
    body = rebuild_type(node.prototype::Ty, tuple(children...))
    return recursive[] ? TyMu(binder, body) : body
end

function _commit_graph!(graph::_TypeGraph, trail)
    # Materialisation is extension-defined and may throw. Build the complete
    # change set before touching the public facade so such an exception cannot
    # leak a prefix of successful links or level lowering.
    bindings = Tuple{TyVar,Ty}[]
    for (node_id, node) in pairs(graph.nodes)
        node.kind == _GRAPH_META || continue
        node.link == 0 && continue
        value = _materialize_graph!(graph, node_id, _MaterializeState())
        origin = node.origin::TyVar
        origin.instance === nothing && push!(bindings, (origin, value))
    end

    for node in graph.nodes
        node.kind == _GRAPH_META || continue
        origin = node.origin::TyVar
        origin.level > node.level && _set_level!(origin, node.level, trail)
    end
    for (origin, value) in bindings
        _set_instance!(origin, value, trail)
    end
    return nothing
end

function _unify_graph!(left::Ty, right::Ty, trail)
    graph = _TypeGraph()
    local left_root::Int
    local right_root::Int
    try
        left_root = _elaborate_graph!(graph, left)
        right_root = _elaborate_graph!(graph, right)
    catch error
        error isa _TypeElaborationError || rethrow()
        return Err(MismatchError(left, right))
    end

    _unify_graph_nodes!(graph, left_root, right_root) ||
        return Err(MismatchError(left, right))
    _commit_graph!(graph, trail)
    return nothing
end
