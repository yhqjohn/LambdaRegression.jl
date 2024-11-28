module DebruijnModule

# using ..DeBruijnModule
using TestItems
# import ..DeBruijnModule: pretty_string

abstract type AbstractItem end

struct App <: AbstractItem end
struct Abs <: AbstractItem end
struct DBVar <: AbstractItem
    index::UInt  
end
struct Proxy{T} <: AbstractItem
    value::T
end
Proxy(value::T) where {T} = Proxy{T}(value)
Base.convert(::Type{Proxy{T}}, ::Proxy{S}) where {T, S} = Proxy{T}(convert(T, value))
Base.:(==)(a::Proxy, b::Proxy) = a.value == b.value

isleaf(item::AbstractItem) = false
isleaf(item::DBVar) = true
isleaf(item::Proxy) = true

DBItem{T} = Union{App, Abs, DBVar, Proxy{T}}


"""
    DBTree{T} = Vector{DBItem{T}}

Flatten expression of DeBruijn tree. It is a vector of either `App`, `Abs`, `DBVar`, or `Proxy`.
An `App` is an application, and the left child is the function and the right child is the argument.
An `Abs` is an abstraction.
A `DBVar` is a variable in the DeBruijn index.
A `Proxy` is a proxy.
The items in the vector are the nodes of the tree in pre-order depth-first order.
"""
DBTree{T} = Vector{DBItem{T}}

# function flatten(node::DBNode{T}) where {T}
#     result = DBItem{T}[]
#     _flatten!(result, node)
#     return result
# end

# function _flatten!(result::Vector{DBItem{T}}, node::DBNode{T}) where {T}
#     if node isa VarDB
#         push!(result, DBVar(node.index))
#     elseif node isa ProxyDB
#         push!(result, Proxy{T}(node.value))
#     elseif node isa AbsDB
#         push!(result, Abs())
#         _flatten!(result, node.body)
#     elseif node isa AppDB
#         push!(result, App())
#         _flatten!(result, node.func)
#         _flatten!(result, node.arg)
#     end
# end

# @testitem "flatten DeBruijn tree" begin
#     using LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
#     using LambdaRegression.ParseModule
#     using LambdaRegression.FlattenDebruijnModule: flatten, DBTree, App, Abs, Var, Proxy

#     @test flatten(DBNode(@lambda x->x+1)) == [Abs(), App(), App(), Proxy(+), Var(0), Proxy(1)]
#     @test flatten(DBNode(@lambda f->(x->f(x*x))*(x->f(x*x)))) == [
#         Abs(), App(), 
#         Abs(), App(), Var(1), App(), Var(0), Var(0),
#         Abs(), App(), Var(1), App(), Var(0), Var(0),
#     ]
# end

function subtree(tree::DBTree{T}, index::UInt) where {T}
    return tree[index:subtree_end(tree, index)]
end

function subtree_end(tree::DBTree{T}, index::UInt) where {T}
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

@testitem "subtree operations" begin
    using LambdaRegression.DeBruijnModule: DBNode
    using LambdaRegression.ParseModule
    using LambdaRegression.FlattenDebruijnModule: flatten, subtree, subtree_end, App, Abs, Var, Proxy

    # Test case 1: Simple expression x + 1
    tree1 = flatten(DBNode(@lambda x->x+1))
    @test subtree(tree1, UInt(1)) == [Abs(), App(), App(), Proxy(+), Var(0), Proxy(1)]  # Whole tree
    @test subtree(tree1, UInt(2)) == [App(), App(), Proxy(+), Var(0), Proxy(1)]         # Body of abstraction
    @test subtree(tree1, UInt(3)) == [App(), Proxy(+), Var(0)]                          # Left branch of outer +
    @test subtree(tree1, UInt(4)) == [Proxy(+)]                                         # Just the + operator
    @test subtree(tree1, UInt(5)) == [Var(0)]                                          # Just the variable
    @test subtree(tree1, UInt(6)) == [Proxy(1)]                                        # Just the constant

    # Test case 2: More complex expression (x -> f(x*x))
    tree2 = flatten(DBNode(@lambda f->(x->f(x*x))))
    @test subtree(tree2, UInt(1)) == [Abs(), Abs(), App(), Var(1), App(), Var(0), Var(0)]
    @test subtree(tree2, UInt(2)) == [Abs(), App(), Var(1), App(), Var(0), Var(0)]
    @test subtree(tree2, UInt(3)) == [App(), Var(1), App(), Var(0), Var(0)]
    @test subtree(tree2, UInt(4)) == [Var(1)]
    @test subtree(tree2, UInt(5)) == [App(), Var(0), Var(0)]
end

# function unflatten(tree::DBTree{T}) where {T}
#     if isempty(tree)
#         throw(ArgumentError("Cannot unflatten empty tree"))
#     end
#     node, _ = _unflatten(tree, UInt(1))
#     return node
# end

# function _unflatten(tree::DBTree{T}, index::UInt) where {T}
#     item = tree[index]
#     if item isa Var
#         return VarDB{T}(item.index), index
#     elseif item isa Proxy
#         return ProxyDB{T}(item.value), index
#     elseif item isa Abs
#         body, end_index = _unflatten(tree, index + 1)
#         return AbsDB{T}(body), end_index
#     elseif item isa App
#         func, func_end = _unflatten(tree, index + 1)
#         arg, arg_end = _unflatten(tree, func_end + 1)
#         return AppDB{T}(func, arg), arg_end
#     else
#         throw(ArgumentError("Unknown item type: $(typeof(item))"))
#     end
# end

# @testitem "unflatten DeBruijn tree" begin
#     using LambdaRegression.DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB
#     using LambdaRegression.ParseModule
#     using LambdaRegression.FlattenDebruijnModule: flatten, unflatten, DBItem

#     # Test simple expressions
#     expr1 = DBNode(@lambda x->x+1)
#     @test unflatten(flatten(expr1)) == expr1

#     # Test more complex expressions
#     expr2 = DBNode(@lambda f->(x->f(x*x)))
#     @test unflatten(flatten(expr2)) == expr2

#     # Test nested applications
#     expr3 = DBNode(@lambda f->(x->f(x*x))*(x->f(x*x)))
#     @test unflatten(flatten(expr3)) == expr3

#     # Test error cases
#     @test_throws ArgumentError unflatten(DBItem{Int}[])
# end

# pretty_string(tree::DBTree{T}) where {T} = pretty_string(unflatten(tree))
# Base.display(tree::DBTree{T}) where {T} = println(pretty_string(tree))
# Base.display(io::IO, tree::DBTree{T}) where {T} = println(io, pretty_string(tree))

end # module FlattenDebruijn
