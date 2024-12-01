module ReduceModule

using ..DebruijnModule
using ..DebruijnModule: App, Abs, DBVar, Proxy, DBItem, DBTree, DBTreeLike, application, isapplicable, subtree_end
using TestItems

struct Normalized end
const normalized = Normalized()


"""
    beta_reduce!(tree::DBTree{T}[, start, stop]) where {T}

Applying a single step of β-reduction to a tree in normalization order.
The optional arguments `start` and `stop` specify the range of the tree to be reduced.
If not provided, the entire tree is reduced.
This function returns the reduced tree, or `normalized` if the tree is already normalized and not reducible.
"""
beta_reduce!(tree::DBTree{T}) where {T} = beta_reduce!(tree, 1, length(tree))

function beta_reduce!(tree::DBTree{T}, start, stop) where {T}
    head = tree[start]
    if head isa Proxy || head isa DBVar
        return normalized
    elseif head isa Abs
        return beta_reduce!(tree, start + 1, stop)
    elseif head isa App
        func_start = start + 1
        func_stop = subtree_end(tree, func_start)
        arg_start = func_stop + 1
        arg_stop = subtree_end(tree, arg_start)
        func = view(tree, func_start:func_stop)
        arg = view(tree, arg_start:arg_stop)
        if isapplicable(func, arg)
            splice!(tree.items, start:stop, application(func, arg).items)
            return tree
        else
            result = beta_reduce!(tree, func_start, func_stop)
            if result === normalized
                return beta_reduce!(tree, arg_start, arg_stop)
            else
                return result
            end
        end
    end
end

beta_reduce(tree::DBTree{T}) where {T} = beta_reduce!(DBTree{T}(tree))

@testitem "beta_reduce! tests" begin
    using LambdaRegression.ReduceModule: beta_reduce!, normalized
    using LambdaRegression.DebruijnModule: DBTree, Abs, App, DBVar, Proxy

    # Test 1: When the head is a leaf (DBVar or Proxy), no reduction is possible.
    tree1 = DBTree([DBVar(42)])
    tree2 = DBTree([Proxy{Any}("x")])
    @test beta_reduce!(tree1) === normalized
    @test beta_reduce!(tree2) === normalized

    # Test 2: Basic beta reduction.
    # Construct a beta‐redex:
    #   (λ. DBVar(0)) applied to Proxy{Any}("a")
    # DFS representation: [App(), Abs(), DBVar(0), Proxy{Any}("a")]
    # The application should produce: [Proxy{Any}("a")]
    redex = DBTree([App(), Abs(), DBVar(0), Proxy{Any}("a")])
    result = beta_reduce!(redex)
    expected = DBTree([Proxy{Any}("a")])
    @test result.items == expected.items

    # Test 3: Reduction within an abstraction.
    # This tree represents: λ. ( (λ. DBVar(0)) applied to Proxy{Any}("a") )
    # DFS order: [Abs(), App(), Abs(), DBVar(0), Proxy{Any}("a")]
    # Expected result: [Abs(), Proxy{Any}("a")]
    tree3 = DBTree([Abs(), App(), Abs(), DBVar(0), Proxy{Any}("a")])
    result3 = beta_reduce!(tree3)
    expected3 = DBTree([Abs(), Proxy{Any}("a")])
    @test result3.items == expected3.items

    # Test 4: Nested redex in the argument branch.
    # Outer redex is not applicable because its function is a Proxy (not an abstraction).
    # Tree: [App(), Proxy{Any}("f"), App(), Abs(), DBVar(0), Proxy{Any}("a")]
    # The inner redex [App(), Abs(), DBVar(0), Proxy{Any}("a")] is applicable.
    # Expected result after reduction: [App(), Proxy{Any}("f"), Proxy{Any}("a")]
    tree4 = DBTree([App(), Proxy{Any}("f"), App(), Abs(), DBVar(0), Proxy{Any}("a")])
    result4 = beta_reduce!(tree4)
    expected4 = DBTree([App(), Proxy{Any}("f"), Proxy{Any}("a")])
    @test result4.items == expected4.items

    # Test 5: Reduction order when both the function and argument branches contain reducible redexes.
    #
    # The tree is:
    #   [ App(),
    #       App(), Abs(), DBVar(0), Proxy{Any}("z"),
    #       App(), Abs(), DBVar(0), Proxy{Any}("y")
    #   ]
    #
    # Outer redex (at index 1) is not applicable because its function branch (starting at index 2)
    # does not begin with an abstraction. The algorithm will then first reduce the function branch.
    # For the inner redex at indices 2:5:
    #   [App(), Abs(), DBVar(0), Proxy{Any}("z")]  reduces to  [Proxy{Any}("z")]
    #
    # Expected final tree: [App(), Proxy{Any}("z"), App(), Abs(), DBVar(0), Proxy{Any}("y")]
    tree5 = DBTree([
        App(),
        App(), Abs(), DBVar(0), Proxy{Any}("z"),
        App(), Abs(), DBVar(0), Proxy{Any}("y")
    ])
    result5 = beta_reduce!(tree5)
    expected5 = DBTree([
        App(),
        Proxy{Any}("z"),
        App(), Abs(), DBVar(0), Proxy{Any}("y")
    ])
    @test result5.items == expected5.items
end

end # module Reduce
