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

mutable struct _MuEqBuilder
    nodes::Vector{_MuEqNode}
    binders::IdDict{TyVar,Int}
    resolving::IdDict{TyVar,Nothing}
end

_MuEqBuilder() = _MuEqBuilder(_MuEqNode[], IdDict{TyVar,Int}(), IdDict{TyVar,Nothing}())

function _new_mu_eq_node!(builder::_MuEqBuilder, node::_MuEqNode=_MuEqNode())::Int
    push!(builder.nodes, node)
    return length(builder.nodes)
end

function _with_mu_binder(f, builder::_MuEqBuilder, binder::TyVar, root::Int)
    had_old = haskey(builder.binders, binder)
    old = had_old ? builder.binders[binder] : 0
    builder.binders[binder] = root
    try
        return f()
    finally
        if had_old
            builder.binders[binder] = old
        else
            delete!(builder.binders, binder)
        end
    end
end

function _elaborate_mu!(builder::_MuEqBuilder, mu::TyMu)::Int
    root = _new_mu_eq_node!(builder)
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
    root = _new_mu_eq_node!(builder, _MuEqNode(tag, parameters, Int[], true))
    for child in type_children(ty)
        child isa Ty || throw(ArgumentError("type-constructor children must be Ty values"))
        push!(builder.nodes[root].children, _elaborate_mu_eq!(builder, child))
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
    variables::IdDict{TyVar,Int}
    binders::IdDict{TyVar,Int}
    binder_used::IdDict{TyVar,Bool}
    resolving::IdDict{TyVar,Nothing}
    rigids::Dict{UInt,Int}
    preferred_binders::Dict{Int,TyVar}
end

_TypeGraph() = _TypeGraph(
    _TypeGraphNode[],
    IdDict{TyVar,Int}(),
    IdDict{TyVar,Int}(),
    IdDict{TyVar,Bool}(),
    IdDict{TyVar,Nothing}(),
    Dict{UInt,Int}(),
    Dict{Int,TyVar}(),
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
    root = _push_graph_node!(graph, _placeholder_node())
    had_binder = haskey(graph.binders, mu.var)
    old_binder = had_binder ? graph.binders[mu.var] : 0
    had_used = haskey(graph.binder_used, mu.var)
    old_used = had_used ? graph.binder_used[mu.var] : false
    graph.binders[mu.var] = root
    graph.binder_used[mu.var] = false

    local body_root::Int
    local used::Bool
    try
        body_root = _elaborate_graph!(graph, mu.body)
        used = graph.binder_used[mu.var]
    finally
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

    root = _push_graph_node!(graph, _con_node(ty))
    for child in type_children(ty)
        child isa Ty || throw(ArgumentError("type-constructor children must be Ty values"))
        push!(graph.nodes[root].children, _elaborate_graph!(graph, child))
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
    active_meta::Dict{Int,TyVar}
    active_con::Dict{Int,TyVar}
    recursive_meta::Set{Int}
    recursive_con::Set{Int}
    cache::Dict{Int,Ty}
end

_MaterializeState() = _MaterializeState(
    Dict{Int,TyVar}(),
    Dict{Int,TyVar}(),
    Set{Int}(),
    Set{Int}(),
    Dict{Int,Ty}(),
)

function _materialize_graph!(graph::_TypeGraph, node_id::Int, state::_MaterializeState)::Ty
    node = graph.nodes[node_id]
    if node.kind == _GRAPH_META
        node.link == 0 && return node.origin::TyVar
        if haskey(state.active_meta, node_id)
            push!(state.recursive_meta, node_id)
            return state.active_meta[node_id]
        end
        haskey(state.cache, node_id) && return state.cache[node_id]

        binder = node.origin::TyVar
        state.active_meta[node_id] = binder
        body = _materialize_graph!(graph, node.link, state)
        delete!(state.active_meta, node_id)
        result = node_id in state.recursive_meta ? TyMu(binder, body) : body
        state.cache[node_id] = result
        return result
    elseif node.kind == _GRAPH_RIGID
        return node.prototype::TyRigid
    end

    if haskey(state.active_con, node_id)
        push!(state.recursive_con, node_id)
        return state.active_con[node_id]
    end
    haskey(state.cache, node_id) && return state.cache[node_id]

    binder = get(graph.preferred_binders, node_id, nothing)
    binder === nothing && (binder = TyVar(-node_id, 0, nothing))
    state.active_con[node_id] = binder
    children = Ty[_materialize_graph!(graph, child, state) for child in node.children]
    delete!(state.active_con, node_id)
    body = rebuild_type(node.prototype::Ty, tuple(children...))
    result = node_id in state.recursive_con ? TyMu(binder, body) : body
    state.cache[node_id] = result
    return result
end

function _commit_graph!(graph::_TypeGraph, trail)
    for node in graph.nodes
        node.kind == _GRAPH_META || continue
        origin = node.origin::TyVar
        origin.level != node.level && _set_level!(origin, node.level, trail)
    end
    for (node_id, node) in pairs(graph.nodes)
        node.kind == _GRAPH_META || continue
        node.link == 0 && continue
        origin = node.origin::TyVar
        origin.instance === nothing || continue
        value = _materialize_graph!(graph, node_id, _MaterializeState())
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
