module DebruijnModule

using ..DebruijnNodeModule
using ..ParseModule: AbstractToken
using TestItems
import ..DebruijnNodeModule: pretty_string

abstract type AbstractItem end
function arity end

struct App <: AbstractItem end
struct Abs <: AbstractItem end
struct DBVar <: AbstractItem
    index::UInt  
end
struct Proxy{T} <: AbstractItem
    value::T
    # Proxy(value::T) where {T} = new{T}(value)
end
Base.convert(::Type{Proxy{T}}, p::Proxy{S}) where {T, S} = Proxy{T}(convert(T, p.value))
Base.:(==)(a::Proxy, b::Proxy) = a.value == b.value

arity(::App) = 2
arity(::Abs) = 1
arity(::DBVar) = 0
arity(::Proxy) = 0
isleaf(i::AbstractItem) = arity(i) == 0

DBItem{T} = Union{App, Abs, DBVar, Proxy{T}}


"""
    DfsTree{T}

A tree structure that stores the nodes of a tree in depth-first order.
"""
struct DfsTree{T<:AbstractItem} <: AbstractVector{T}
    items::Vector{T}
end

Base.size(tree::DfsTree{T}) where {T} = size(tree.items)
Base.getindex(tree::DfsTree{T}, i::Int) where {T} = tree.items[i]
Base.IndexStyle(::Type{DfsTree{T}}) where {T} = IndexLinear()
Base.setindex!(tree::DfsTree{T}, v::T, i::Int) where {T} = (tree.items[i] = v)

const DfsTreeView{T} = SubArray{T, 1, DfsTree{T}, Tuple{UnitRange{Int}}, true}
const DfsTreeLike{T} = Union{DfsTree{T}, DfsTreeView{T}}

subtree(tree::DfsTreeLike{T}, index::Int) where {T} = view(tree, index:subtree_end(tree, index))

function subtree_end(tree::DfsTreeLike{T}, index::Int) where {T}
    if isleaf(tree[index])
        return index
    elseif tree[index] isa Abs
        return subtree_end(tree, index + 1)
    elseif tree[index] isa App
        left_end = subtree_end(tree, index + 1)
        right_end = subtree_end(tree, left_end + 1)
        return right_end
    end
end

function children(tree::DfsTreeLike{T}, index::Int) where {T}
    children = DfsTreeLike{T}[]
    startid = index + 1
    for _ in 1:arity(tree[index])
        endid = subtree_end(tree, startid)
        push!(children, view(tree, startid:endid))
        startid = endid + 1
    end
    return children
end

const DBTree{T} = DfsTree{DBItem{T}}
const DBTreeLike{T} = Union{DBTree{T}, DfsTreeView{DBItem{T}}}

@testitem "Manual DfsTree and DBItem operations" begin
    using LambdaRegression.DebruijnModule: DfsTree, subtree, children, App, Abs, DBVar, Proxy

    # ---------------------------------------------------------------------------
    # Manually construct a DFS tree representing the expression:
    #
    #   λ. ( (DBVar 0) applied to constant 1 )
    #
    # The tree structure is:
    #   - Abs node (arity = 1)
    #       └── App node (arity = 2)
    #              ├── DBVar(0)        (leaf)
    #              └── Proxy(1)        (leaf)
    #
    # In depth-first order the tree is represented by:
    #   [Abs(), App(), DBVar(0), Proxy(1)]
    # ---------------------------------------------------------------------------
    manual_tree_items = [Abs(), App(), DBVar(0), Proxy(1)]
    manual_tree = DfsTree(manual_tree_items)

    # ------------------------------------------------------------
    # Test 1: Creation and index access via the AbstractVector interface
    # ------------------------------------------------------------
    @test length(manual_tree) == 4
    @test manual_tree[1] == Abs()
    @test manual_tree[2] == App()
    @test manual_tree[3] == DBVar(0)
    @test manual_tree[4] == Proxy(1)

    # ------------------------------------------------------------
    # Test 2: Equality of Proxy nodes
    # ------------------------------------------------------------
    @test Proxy(42) == Proxy(42)
    @test Proxy(42) != Proxy(43)

    # ------------------------------------------------------------
    # Test 3: Conversion on Proxy nodes
    #
    # Convert a Proxy{Int} (created as Proxy(1)) into a Proxy with type Float64.
    # The conversion should change the inner value from 1 to 1.0.
    # ------------------------------------------------------------
    p_int = Proxy(1)  # inferred as Proxy{Int}(1)
    p_float = convert(Proxy{Float64}, p_int)
    @test p_float == Proxy(1.0)

    # ------------------------------------------------------------
    # Test 4: Subtree extraction from the manually created tree
    # ------------------------------------------------------------
    # Full tree (starting at index 1) should equal manual_tree_items:
    full_subtree = collect(subtree(manual_tree, 1))
    @test full_subtree == manual_tree_items

    # The subtree for the App node at index 2 should yield its own DFS branch:
    # [App(), DBVar(0), Proxy(1)]
    app_subtree = collect(subtree(manual_tree, 2))
    @test app_subtree == [App(), DBVar(0), Proxy(1)]

    # For leaf nodes, the subtree should simply return the node itself:
    var_subtree = collect(subtree(manual_tree, 3))
    @test var_subtree == [DBVar(0)]
    proxy_subtree = collect(subtree(manual_tree, 4))
    @test proxy_subtree == [Proxy(1)]

    # ------------------------------------------------------------
    # Test 5: Children extraction
    #
    # For the Abs node (at index 1 with arity 1), the only child should be the entire subtree
    # rooted at the App node:
    #   [App(), DBVar(0), Proxy(1)]
    # ------------------------------------------------------------
    abs_children = children(manual_tree, 1)
    @test length(abs_children) == 1
    @test collect(abs_children[1]) == [App(), DBVar(0), Proxy(1)]
end

DBTree{T}(token::AbstractToken) where {T} = flatten(DBNode{T}(token))
DBTree(x) = DBTree{Any}(x)

function flatten(node::DBNode{T}) where {T}
    result = DBItem{T}[]
    _flatten!(result, node)
    return DBTree{T}(result)
end

function _flatten!(result::Vector{DBItem{T}}, node::DBNode{T}) where {T}
    if node isa VarDB
        push!(result, DBVar(node.index))
    elseif node isa ProxyDB
        push!(result, Proxy{T}(node.value))
    elseif node isa AbsDB
        push!(result, Abs())
        _flatten!(result, node.body)
    elseif node isa AppDB
        push!(result, App())
        _flatten!(result, node.func)
        _flatten!(result, node.arg)
    end
end

@testitem "flatten DeBruijn tree" begin
    using LambdaRegression.DebruijnNodeModule: DBNode
    using LambdaRegression.ParseModule
    using LambdaRegression.DebruijnModule: flatten, DBTree, App, Abs, DBVar, Proxy

    @test flatten(DBNode(@lambda x->x+1)) == [Abs(), App(), App(), Proxy(+), DBVar(0), Proxy(1)]
    @test flatten(DBNode(@lambda f->(x->f(x*x))*(x->f(x*x)))) == [
        Abs(), App(), 
        Abs(), App(), DBVar(1), App(), DBVar(0), DBVar(0),
        Abs(), App(), DBVar(1), App(), DBVar(0), DBVar(0),
    ]
end

function to_debruijn_node(tree::DBTreeLike{T}) where {T}
    if isempty(tree)
        throw(ArgumentError("Cannot unflatten empty tree"))
    end
    node, _ = _unflatten(tree, 1)
    return node
end

function _unflatten(tree::DBTreeLike{T}, index::Int) where {T}
    item = tree[index]
    if item isa DBVar
        return VarDB{T}(item.index), index
    elseif item isa Proxy
        return ProxyDB{T}(item.value), index
    elseif item isa Abs
        body, end_index = _unflatten(tree, index + 1)
        return AbsDB{T}(body), end_index
    elseif item isa App
        func, func_end = _unflatten(tree, index + 1)
        arg, arg_end = _unflatten(tree, func_end + 1)
        return AppDB{T}(func, arg), arg_end
    else
        throw(ArgumentError("Unknown item type: $(typeof(item))"))
    end
end

# pretty_string(tree::DBTree{T}) where {T} = pretty_string(unflatten(tree))
# Base.display(tree::DBTree{T}) where {T} = println(pretty_string(tree))
# Base.display(io::IO, tree::DBTree{T}) where {T} = println(io, pretty_string(tree))

@testitem "to_debruijn_node conversion" begin
    using LambdaRegression.DebruijnModule: to_debruijn_node, DBTree, App, Abs, DBVar, Proxy
    using LambdaRegression.DebruijnNodeModule: VarDB, ProxyDB, AbsDB, AppDB

    # Test 1: An empty tree should throw an ArgumentError.
    empty_tree = DBTree{Any}([])
    @test_throws ArgumentError to_debruijn_node(empty_tree)

    # Test 2: A tree with a single DBVar node.
    var_tree = DBTree([DBVar(42)])
    debruijn_var = to_debruijn_node(var_tree)
    @test debruijn_var == VarDB(42)

    # Test 3: A tree with a single Proxy node.
    proxy_tree = DBTree([Proxy{Any}("constant")])
    debruijn_proxy = to_debruijn_node(proxy_tree)
    @test debruijn_proxy == ProxyDB("constant")

    # Test 4: An abstraction node (Abs) with a single body.
    #   This represents: λ. DBVar(0)
    abs_tree = DBTree([Abs(), DBVar(0)])
    debruijn_abs = to_debruijn_node(abs_tree)
    @test debruijn_abs == AbsDB(VarDB(0))

    # Test 5: An application node (App) with two children.
    #   This represents: (Proxy(:f)) applied to (DBVar(10))
    app_tree = DBTree([App(), Proxy{Any}(:f), DBVar(10)])
    debruijn_app = to_debruijn_node(app_tree)
    @test debruijn_app == AppDB(ProxyDB(:f), VarDB(10))

    # Test 6: A nested structure.
    #   This tree represents:
    #       λ. ( (DBVar 0) applied to (λ. DBVar(0)) )
    #   The DFS tree is constructed as: [Abs(), App(), DBVar(0), Abs(), DBVar(0)]
    nested_tree = DBTree([Abs(), App(), DBVar(0), Abs(), DBVar(0)])
    debruijn_nested = to_debruijn_node(nested_tree)
    @test debruijn_nested == AbsDB(AppDB(VarDB(0), AbsDB(VarDB(0))))
end

end # module FlattenDebruijn
