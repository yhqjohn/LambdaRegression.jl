"""
    Trees.DfsTrees

A submodule providing depth-first search tree implementations optimized for storage and traversal.

# Overview

The `DfsTrees` module implements tree structures that store nodes in depth-first order,
enabling efficient traversal and manipulation of tree data. This storage format is
particularly useful for:

- Linear traversal of tree structures
- Efficient memory usage through contiguous storage
- Fast access to tree nodes by index
- Subtree operations using array slicing

# Core Types

## AbstractNodeItem

The abstract base type for items that can be stored in a `DfsTree`. Implementations
must define the `arity` method to specify the number of children.

## DfsTree{T<:AbstractNodeItem}

A tree structure that stores nodes of type `T` in depth-first order. The tree provides
array-like access while maintaining tree semantics through the `AbstractTree` interface.
For DfsTree, node indices are integers referring to positions in the storage array.

# Usage Examples

```julia
# Define a custom node type
struct MyNode <: AbstractNodeItem
    value::Int
    child_count::Int
end
Trees.DfsTrees.arity(node::MyNode) = node.child_count

# Create and use a DFS tree
nodes = [MyNode(1, 2), MyNode(2, 0), MyNode(3, 1), MyNode(4, 0)]
tree = DfsTree(nodes)

# Access nodes directly
first_node = tree[1]
subtree_view = subtree(tree, 2)  # Subtree rooted at index 2
```

# Interface Compliance

`DfsTree` implements the `AbstractTree` interface defined in the parent `Trees` module,
providing seamless integration with all tree traversal and manipulation algorithms.
"""
module DfsTrees

using TestItems
import ..Trees: AbstractTree, NodeIndex, root, setroot!, subtree, children, setchild!, arity, isleaf
import ..Trees: AbstractTraverseState, StateBag, build_state_bag, getstate, Leave, Break, traverse, init, requires, default_states, StackState, enter!, leave!

"""
    AbstractNodeItem

An abstract type for node items in a [`DfsTree`](@ref).
Subtypes of `AbstractNodeItem` must implement the following methods:

- [`arity`](@ref): Return the arity of the node item.

Following methods are provided and might be overridden:

- [`isleaf`](@ref): Return true if the node item is a leaf node.
"""
abstract type AbstractNodeItem end

"""
    arity(item::AbstractNodeItem)

Return the arity of the node item.
"""
function arity(item::AbstractNodeItem) end

"""
    isleaf(item::AbstractNodeItem)

Return true if the node item is a leaf node.
"""
isleaf(item::AbstractNodeItem) = arity(item) == 0

"""
    BinaryItemType

Enum for different types of binary tree nodes.
"""
@enum BinaryItemType binary unary leaf

"""
    BinaryItem{T}

A node item for binary trees with a type indicator and value.

# Fields
- `type::BinaryItemType`: The type of node (binary, unary, or leaf)
- `value::T`: The value stored in this node
"""
struct BinaryItem{T} <: AbstractNodeItem
    type::BinaryItemType
    value::T
end

arity(item::BinaryItem) = item.type == binary ? 2 : (item.type == unary ? 1 : 0)

"""
    DfsTree{T<:AbstractNodeItem} <: AbstractTree

A tree structure that stores the nodes of a tree in depth-first order.
For DfsTree, node indices are integers referring to positions in the internal storage array.
"""
struct DfsTree{T<:AbstractNodeItem} <: AbstractTree
    items::Vector{T}
end

# Base array interface
Base.size(tree::DfsTree{T}) where {T} = size(tree.items)
Base.getindex(tree::DfsTree{T}, i::Int) where {T} = tree.items[i]
Base.getindex(tree::DfsTree{T}, i::UnitRange{Int}) where {T} = DfsTree{T}(tree.items[i])
Base.IndexStyle(::Type{DfsTree{T}}) where {T} = IndexLinear()
Base.length(tree::DfsTree{T}) where {T} = length(tree.items)

Base.:(==)(tree1::DfsTree{T}, tree2::DfsTree{T}) where {T} = tree1.items == tree2.items  # Add equality comparison for DfsTree

Base.eltype(::DfsTree{T}) where T = Int  # Override eltype for DfsTree

# AbstractTree interface implementation
NodeIndex(::Type{DfsTree{T}}) where T = Int
root(tree::DfsTree{T}) where T = length(tree) == 0 ? nothing : 1  # Root is always at index 1, nothing for empty trees

function setroot!(tree::DfsTree{T}, new_tree::DfsTree{T}) where T
    empty!(tree.items)
    append!(tree.items, new_tree.items)
    return tree
end

function subtree(tree::DfsTree{T}, node_index::Int) where {T}
    end_index = subtree_end(tree, node_index)  # Let Julia handle bounds checking for node_index
    return DfsTree{T}(tree.items[node_index:end_index])
end

function children(tree::DfsTree{T}, node_index::Int) where {T}
    if isleaf(tree, node_index)  # Let Julia handle bounds checking for node_index
        return Int[]
    end
    child_ends = children_ends(tree, node_index)
    child_indices = Int[]
    start_index = node_index + 1  # First child starts at node_index + 1
    for end_index in child_ends
        push!(child_indices, start_index)
        start_index = end_index + 1  # Next child starts after current child ends
    end
    return child_indices
end

function setchild!(tree::DfsTree{T}, node_index::Int, child_index::Int, newchild::DfsTree{T}) where {T}
    if node_index < 1 || node_index > length(tree)
        throw(BoundsError(tree.items, node_index))
    end
    
    if child_index < 1 || child_index > arity(tree, node_index)
        throw(BoundsError("Child index $child_index out of bounds for tree with arity $(arity(tree, node_index))"))
    end
    
    child_ends = children_ends(tree, node_index, child_index)
    
    if child_index == 1  # Calculate the start and end positions of the child to replace
        start_pos = node_index + 1  # First child starts after parent
    else
        start_pos = child_ends[child_index-1] + 1  # After previous child
    end
    end_pos = child_ends[child_index]
    
    splice!(tree.items, start_pos:end_pos, newchild.items)  # Replace the child subtree with the new child
    return tree
end

arity(tree::DfsTree{T}, node_index::Int) where {T} = arity(tree.items[node_index])
isleaf(tree::DfsTree{T}, node_index::Int) where {T} = isleaf(tree.items[node_index])

function subtree_end(tree::DfsTree{T}, index::Int) where {T}
    if isleaf(tree, index)
        return index
    else
        for _ in 1:arity(tree, index)
            index = subtree_end(tree, index + 1)
        end
        return index
    end
end

function children_ends(tree::DfsTree{T}, index::Int, n=nothing) where {T} 
    n = n === nothing ? arity(tree, index) : n
    ends = Int[]
    current_index = index + 1  # Start from the first child
    for _ in 1:n
        child_end = subtree_end(tree, current_index)
        push!(ends, child_end)
        current_index = child_end + 1  # Move to next sibling
    end
    return ends
end

default_states(tree::DfsTree{T}) where T = [StackState{Int}]  # DfsTree automatically includes StackState for traversal

# Optimized traverse implementation for DfsTree using sequential access
function traverse(f, bag::StateBag, tree::DfsTree{T}, start_node::Int) where T
    current_pos = 1
    # Stack tracks (node_index, subtree_end_index) for nodes that need leave! called
    leave_stack = Tuple{Int,Int}[]
    result = nothing
    while current_pos <= length(tree)
        current_index = current_pos
        
        # Call enter! for all states in dependency order
        (x->(enter!(x, tree, current_index))).(bag.order)
        
        # Track this node for leave! later (unless it's a leaf)
        if !isleaf(tree, current_index)
            push!(leave_stack, (current_index, subtree_end(tree, current_index)))
        end
        
        # Call user function
        result = f(tree, current_index, bag)
        if result isa Break
            return result, bag
        elseif result isa Leave
            # Skip subtree - call leave! for current node then jump to next sibling
            (x->(leave!(x, tree, current_index))).(bag.order)
            
            # Remove current node from leave_stack if it was added
            if !isempty(leave_stack) && leave_stack[end][1] == current_index
                pop!(leave_stack)
            end
            
            # Jump to after the subtree
            current_pos = subtree_end(tree, current_index) + 1
        else
            # Normal traversal
            if isleaf(tree, current_index)
                # Leaf node - call leave! immediately
                (x->(leave!(x, tree, current_index))).(bag.order)
            end
            current_pos += 1
        end
       
        # Check if we need to call leave! for any parent nodes whose subtrees are complete
        while !isempty(leave_stack) && current_pos > leave_stack[end][2]
            parent_idx, _ = pop!(leave_stack)
            (x->(leave!(x, tree, parent_idx))).(bag.order)
        end
    end
    
    # Call leave! for any remaining nodes in the stack
    while !isempty(leave_stack)
        idx, _ = pop!(leave_stack)
        (x->(leave!(x, tree, idx))).(bag.order)
    end
    
    return result, bag
end

########################################################
# Optimized Iteration Implementation
########################################################

"""
    Base.iterate(tree::DfsTree{T}) where T<:AbstractNodeItem

Initialize optimized pre-order depth-first iteration for DfsTree.

# Performance Benefits
Unlike general tree implementations that require O(arity) operations per node to access children,
DfsTree's pre-ordered storage enables O(1) per-node iteration by simple sequential access.

# Returns
- `(first_node_index, next_index)` where next_index tracks the current position
- `nothing` if the tree is empty

# Implementation Notes
This leverages DfsTree's internal storage format where nodes are already stored in depth-first order.
Each iteration returns an integer node index that can be used to access the tree.
"""
function Base.iterate(tree::DfsTree{T}) where T<:AbstractNodeItem
    if length(tree) == 0
        return nothing
    end
    return 1, 2  # Return first node (index 1) and next index (2)
end

"""
    Base.iterate(tree::DfsTree{T}, next_index::Int) where T<:AbstractNodeItem

Continue optimized pre-order depth-first iteration for DfsTree.

# Performance Benefits  
Each iteration step is O(1) since we simply access the next sequential element,
compared to O(arity) for general tree traversal that must navigate parent-child relationships.

# Arguments
- `tree`: The DfsTree being iterated
- `next_index`: Current position in the pre-ordered storage (1-based)

# Returns
- `(next_node_index, next_next_index)` for continued iteration
- `nothing` when all nodes have been visited

# Implementation Notes
Returns integer node indices for efficient access while leveraging
the pre-ordered storage for efficient O(1) per-node traversal.
"""
function Base.iterate(tree::DfsTree{T}, next_index::Int) where T<:AbstractNodeItem
    if next_index > length(tree)
        return nothing
    end
    return next_index, next_index + 1
end

########################################################
# Test Items
########################################################

@testitem "DfsTree AbstractTree interface - basic setup" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, arity, isleaf, leaf, binary, unary
    using LambdaRegression.Trees: NodeIndex, root, subtree, children, setchild!
    
    @test BinaryItem{Int} <: LambdaRegression.Trees.DfsTrees.AbstractNodeItem  # Test basic construction and interface compliance
    @test DfsTree{BinaryItem{Int}} <: LambdaRegression.Trees.AbstractTree
    
    leaf_node = BinaryItem(leaf, 1)  # Test arity and isleaf for items
    binary_node = BinaryItem(binary, 2)
    unary_node = BinaryItem(unary, 3)
    
    @test arity(leaf_node) == 0
    @test arity(binary_node) == 2
    @test arity(unary_node) == 1
    @test isleaf(leaf_node) == true
    @test isleaf(binary_node) == false
    @test isleaf(unary_node) == false
    
    tree = DfsTree([leaf_node])  # Test NodeIndex
    @test NodeIndex(typeof(tree)) == Int
    
    @test root(tree) == 1  # Test root
    
    empty_tree = DfsTree(BinaryItem{Int}[])  # Test empty tree
    @test root(empty_tree) === nothing
end

@testitem "DfsTree AbstractTree interface - children method" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, arity, isleaf, leaf, binary, unary
    using LambdaRegression.Trees: children, setchild!
    
    leaf_tree = DfsTree([BinaryItem(leaf, 1)])  # Test leaf node (no children)
    @test arity(leaf_tree, 1) == 0
    @test isleaf(leaf_tree, 1) == true
    @test children(leaf_tree, 1) == Int[]
    @test length(children(leaf_tree, 1)) == 0
    
    nodes = [BinaryItem(binary, 1), BinaryItem(leaf, 2), BinaryItem(leaf, 3)]  # Test tree with children
    tree = DfsTree(nodes)  # Structure: 1 -> [2, 3]
    @test arity(tree, 1) == 2
    @test isleaf(tree, 1) == false
    
    tree_children = children(tree, 1)
    @test length(tree_children) == 2
    
    @test tree_children[1] == 2  # Check first child index
    @test tree[tree_children[1]].value == 2
    @test arity(tree, tree_children[1]) == 0
    @test isleaf(tree, tree_children[1]) == true
    
    @test tree_children[2] == 3  # Check second child index
    @test tree[tree_children[2]].value == 3
    @test arity(tree, tree_children[2]) == 0
    @test isleaf(tree, tree_children[2]) == true
end

@testitem "DfsTree AbstractTree interface - complex tree children" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, arity, isleaf, leaf, binary, unary
    using LambdaRegression.Trees: children, setchild!
    
    nodes = [  # Test more complex tree structure: 1 -> [2 -> [3 -> [4]], 5]
        BinaryItem(binary, 1),  # Root with 2 children
        BinaryItem(unary, 2),   # First child with 1 child
        BinaryItem(unary, 3),   # Grandchild with 1 child
        BinaryItem(leaf, 4),    # Great-grandchild (leaf)
        BinaryItem(leaf, 5)     # Second child (leaf)
    ]
    tree = DfsTree(nodes)
    
    root_children = children(tree, 1)  # Test root children
    @test length(root_children) == 2
    
    first_child_idx = root_children[1]  # First child should be at index 2
    @test first_child_idx == 2
    @test tree[first_child_idx].value == 2
    @test arity(tree, first_child_idx) == 1
    
    second_child_idx = root_children[2]  # Second child should be at index 5
    @test second_child_idx == 5
    @test tree[second_child_idx].value == 5
    @test arity(tree, second_child_idx) == 0
    @test isleaf(tree, second_child_idx) == true
    
    first_child_children = children(tree, first_child_idx)  # Test children of first child
    @test length(first_child_children) == 1
    @test first_child_children[1] == 3
    @test tree[first_child_children[1]].value == 3
    
    grandchild_children = children(tree, first_child_children[1])  # Test children of grandchild
    @test length(grandchild_children) == 1
    @test grandchild_children[1] == 4
    @test tree[grandchild_children[1]].value == 4
    @test isleaf(tree, grandchild_children[1]) == true
end

@testitem "DfsTree AbstractTree interface - subtree method" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, AbstractNodeItem, arity, isleaf
    using LambdaRegression.Trees: subtree, children
    import LambdaRegression.Trees.DfsTrees: arity as dfs_arity
    
    struct TestNode <: AbstractNodeItem  # Define test node type
        value::Int
        child_count::Int
    end
    dfs_arity(node::TestNode) = node.child_count
    
    nodes = [  # Test subtree extraction from complex tree
        TestNode(1, 2),  # Root with 2 children
        TestNode(2, 1),  # First child with 1 child
        TestNode(3, 0),  # Grandchild (leaf)
        TestNode(4, 0)   # Second child (leaf)
    ]
    tree = DfsTree(nodes)
    
    root_subtree = subtree(tree, 1)  # Test root subtree (should be entire tree)
    @test length(root_subtree) == 4
    @test root_subtree[1].value == 1
    
    first_child_subtree = subtree(tree, 2)  # Test first child subtree
    @test length(first_child_subtree) == 2
    @test first_child_subtree[1].value == 2
    @test first_child_subtree[2].value == 3  # Grandchild
    
    leaf_subtree = subtree(tree, 3)  # Test leaf subtree
    @test length(leaf_subtree) == 1
    @test leaf_subtree[1].value == 3
    
    @test_throws BoundsError subtree(tree, 0)  # Test bounds checking
    @test_throws BoundsError subtree(tree, 5)
end

@testitem "DfsTree optimized iteration - basic functionality" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, AbstractNodeItem, arity
    import LambdaRegression.Trees.DfsTrees: arity as dfs_arity
    
    struct TestNode <: AbstractNodeItem  # Define test node type
        value::Int
        child_count::Int
    end
    dfs_arity(node::TestNode) = node.child_count
    
    single_tree = DfsTree([TestNode(1, 0)])  # Test single node iteration
    result = collect(single_tree)
    @test length(result) == 1
    @test result[1] == 1  # Node index
    
    simple_nodes = [TestNode(1, 2), TestNode(2, 0), TestNode(3, 0)]  # Test simple tree iteration: 1 -> [2, 3]
    simple_tree = DfsTree(simple_nodes)
    
    simple_result = collect(simple_tree)
    @test length(simple_result) == 3
    @test simple_result[1] == 1  # Root first
    @test simple_result[2] == 2  # First child  
    @test simple_result[3] == 3  # Second child
    
    expected_indices = [1, 2, 3]  # Test that order matches pre-order DFS expectation
    for (i, node_index) in enumerate(simple_tree)
        @test node_index == expected_indices[i]
        @test simple_tree[node_index].value == expected_indices[i]
    end
end

@testitem "DfsTree optimized iteration - complex tree" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, AbstractNodeItem, arity
    import LambdaRegression.Trees.DfsTrees: arity as dfs_arity
    
    struct TestNode <: AbstractNodeItem  # Define test node type
        value::Int
        child_count::Int
    end
    dfs_arity(node::TestNode) = node.child_count
    
    complex_nodes = [  # Test complex tree structure: 1 -> [2 -> [3 -> [4]], 5]
        TestNode(1, 2),  # Root
        TestNode(2, 1),  # First child
        TestNode(3, 1),  # Grandchild
        TestNode(4, 0),  # Great-grandchild
        TestNode(5, 0)   # Second child
    ]
    complex_tree = DfsTree(complex_nodes)
    
    result = collect(complex_tree)
    @test length(result) == 5
    
    @test result[1] == 1  # Verify pre-order traversal: root, then left subtree, then right subtree
    @test result[2] == 2  # Left subtree root
    @test result[3] == 3  # Left subtree child
    @test result[4] == 4  # Left subtree grandchild
    @test result[5] == 5  # Right subtree (leaf)
    
    expected_indices = [1, 2, 3, 4, 5]  # Test that iteration visits nodes in the correct order
    for (i, node_index) in enumerate(complex_tree)
        @test node_index == expected_indices[i]
        @test complex_tree[node_index].value == expected_indices[i]
    end
end

@testitem "DfsTree setroot! function" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, setroot!, leaf, binary
    using LambdaRegression.Trees: root
    
    original_tree = DfsTree([BinaryItem(leaf, 1)])  # Test basic root replacement
    new_tree = DfsTree([BinaryItem(leaf, 10)])
    
    result = setroot!(original_tree, new_tree)
    @test result === original_tree  # DfsTree modifies in place
    @test length(result) == 1
    @test result[1].value == 10
    
    simple_tree = DfsTree([BinaryItem(leaf, 1)])  # Test replacing with more complex tree
    complex_nodes = [BinaryItem(binary, 5), BinaryItem(leaf, 2), BinaryItem(leaf, 3)]
    complex_tree = DfsTree(complex_nodes)
    
    result = setroot!(simple_tree, complex_tree)
    @test result === simple_tree  # Same object
    @test length(result) == 3
    @test result[1].value == 5  # Root value
    @test result[2].value == 2  # First child
    @test result[3].value == 3  # Second child
    
    empty_tree = DfsTree(BinaryItem{Int}[])  # Test empty tree replacement
    single_tree = DfsTree([BinaryItem(leaf, 42)])
    
    result = setroot!(empty_tree, single_tree)
    @test result === empty_tree
    @test length(result) == 1
    @test result[1].value == 42
end

@testitem "DfsTree sequential traverse with StackState" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, leaf, binary, unary
    using LambdaRegression.Trees: traverse, getstate, Leave, StackState
    
    nodes = [  # Create test tree structure: 1 -> [2 -> [3], 5]
        BinaryItem(binary, 1),  # Root
        BinaryItem(unary, 2),   # First child
        BinaryItem(leaf, 3),    # Grandchild
        BinaryItem(leaf, 5)     # Second child
    ]
    tree = DfsTree(nodes)
    
    visited_nodes = Int[]  # Test basic sequential traversal
    
    function collect_values(tree, node_index, bag)
        @test haskey(bag.lookup, StackState{Int})  # Verify StackState is automatically included
        
        node = tree[node_index]
        push!(visited_nodes, node.value)
        return nothing
    end
    
    _, bag = traverse(collect_values, tree)
    
    @test visited_nodes == [1, 2, 3, 5]  # Verify all nodes were visited in correct pre-order
    
    @test haskey(bag.lookup, StackState{Int})  # Verify StackState was automatically included
end

@testitem "DfsTree traverse with Leave() using sequential implementation" begin
    using LambdaRegression.Trees.DfsTrees: DfsTree, BinaryItem, BinaryItemType, leaf, binary, unary
    using LambdaRegression.Trees: traverse, getstate, Leave, StackState
    
    nodes = [  # Create test tree structure: 1 -> [2 -> [3], 5]
        BinaryItem(binary, 1),  # Root
        BinaryItem(unary, 2),   # First child
        BinaryItem(leaf, 3),    # Grandchild
        BinaryItem(leaf, 5)     # Second child
    ]
    tree = DfsTree(nodes)
    
    visited_nodes = Int[]  # Test Leave() functionality - skip children of node with value 2
    
    function collect_and_skip_2(tree, node_index, bag)
        node = tree[node_index]
        push!(visited_nodes, node.value)
        
        if node.value == 2
            return Leave()  # Skip children
        end
        return nothing
    end
    
    _, bag = traverse(collect_and_skip_2, tree)
    
    @test visited_nodes == [1, 2, 5]  # Should visit: root(1), first child(2), second child(5)
    # Should skip: grandchild(3) because we returned Leave() for node 2
    
    @test haskey(bag.lookup, StackState{Int})  # Verify StackState was used
    stack_state = getstate(bag, StackState{Int})
    @test isempty(stack_state.stack)  # Stack should be empty after traversal
end

end # module DfsTrees

