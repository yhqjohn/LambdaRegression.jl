"""
    Trees

A module providing abstract interfaces and implementations for tree data structures.

# Overview

This module defines a comprehensive interface for working with tree structures in Julia.
The design emphasizes flexibility and extensibility, allowing users to implement various
types of trees while benefiting from common traversal and manipulation patterns.

# Abstract Tree Interface

The core interface is built around the [`AbstractTree`](@ref) type which models entire trees.
Trees are accessed through node indices, with each tree defining its own index type via
the [`NodeIndex`](@ref) method. The following tables detail the implementation requirements 
for different functionality levels.

## Core Interface (Always Required)

| Method | Classification | Dependencies | Description |
|--------|---------------|--------------|-------------|
| [`NodeIndex`](@ref) | **Mandatory** | None | Return the node index type for this tree |
| [`root`](@ref) | **Mandatory** | None | Get the root node index |
| [`setroot!`](@ref) | **Mandatory** | None | Set a new tree as the root |
| [`subtree`](@ref) | **Mandatory** | None | Extract a subtree rooted at a given node index |
| [`children`](@ref) | **Mandatory** | None | Get the child node indices of a node |
| [`setchild!`](@ref) | **Mandatory** | None | Set a specific child of a node |
| [`arity`](@ref) | Optional | `children` | Get the number of children (default provided) |
| [`isleaf`](@ref) | Optional | `arity` | Check if a node is a leaf (default provided) |

## Iteration Interface (Optional Feature)

| Method | Classification | Dependencies | Description |
|--------|---------------|--------------|-------------|
| `Base.iterate` | Optional | Core interface | Initialize DFS iteration (default provided) |
| `Base.iterate` | Optional | Core interface | Continue DFS iteration (default provided) |

## Traverse Interface (Optional Feature)

| Method | Classification | Dependencies | Description |
|--------|---------------|--------------|-------------|
| [`traverse`](@ref) | Provided | Core interface | Traverse with custom states |
| [`AbstractTraverseState`](@ref) | **Mandatory for custom states** | None | Base type for traverse states |
| [`requires`](@ref) | Optional | None | Declare state dependencies (default: no deps) |
| [`init`](@ref) | **Mandatory for custom states** | None | Initialize a state for traversal |
| [`enter!`](@ref) | **Mandatory for custom states** | None | Called when entering a node |
| [`leave!`](@ref) | **Mandatory for custom states** | None | Called when leaving a node |

## Type Hierarchy

| Type | Classification | Description |
|------|---------------|-------------|
| [`AbstractTree`](@ref) | **Mandatory** | Base type for all tree implementations |
| [`AbstractTraverseState`](@ref) | **Mandatory for custom states** | Base type for traverse states |
| [`StateBag`](@ref) | Provided | Container for states during traversal |
| [`Leave`](@ref) | Provided | Return type to skip children during traversal |

# Usage Examples

```julia
# Basic tree traversal
for node_index in tree
    process(tree, node_index)
end

# Traversal with custom states
mutable struct CountState <: AbstractTraverseState
    count::Int
end
init(::Type{CountState}, tree, root) = CountState(0)
enter!(state::CountState, tree, node) = (state.count += 1)
leave!(::CountState, tree, node) = nothing

result, bag = traverse(tree, CountState) do tree, node, bag
    nothing # State is automatically managed via enter!/leave!
end
total = getstate(bag, CountState).count # 3

# Conditional traversal (skip subtrees)
function process_conditionally(tree, node, bag)
    if should_skip_children(tree, node)
        return Leave()  # Skip children
    end
    # Process node...
    return nothing
end

result, bag = traverse(process_conditionally, tree, SomeState)

# Return values from traversal
function find_node_with_value(target)
    return function(tree, node, bag)
        if get_node_value(tree, node) == target
            return Break(node)  # Early termination with result
        end
        return nothing
    end
end

found_node, bag = traverse(find_node_with_value(42), tree)
# found_node contains the matching node, or nothing if not found
```

# Implementation Notes

All default implementations are provided based on the basic interface methods,
enabling implementers to focus on the core tree structure while automatically
gaining traversal and manipulation capabilities.

Future versions of this module will include additional tree implementations
and specialized algorithms.
"""
module Trees

using ..Utils
using TestItems

export AbstractTree, NodeIndex, root, setroot!, subtree, children, setchild!, arity, isleaf
export traverse, AbstractTraverseState, requires, init, enter!, leave!, StateBag, getstate, default_states, build_state_bag
export lca

########################################################
# 1. AbstractTree Interface
########################################################

"""
    AbstractTree

Abstract type for tree data structures.
Subtypes of `AbstractTree` must implement the following methods:

- [`NodeIndex`](@ref): Return the node index type for this tree.
- [`root`](@ref): Return the root node index of the tree.
- [`setroot!`](@ref): Set a new tree as the root of the current tree.
- [`subtree`](@ref): Extract a subtree rooted at a given node index.
- [`children`](@ref): Return the child node indices of a given node.
- [`setchild!`](@ref): Set the `index`-th child of a given node to a new subtree.

The following methods are provided and might be overridden:
- [`arity`](@ref): Return the arity of a given node in the tree.
- [`isleaf`](@ref): Return true if a given node in the tree is a leaf node.

Default implementations for `Base.iterate` are provided to enable direct iteration over trees.
The [`traverse`](@ref) function provides advanced traversal with state management capabilities.
"""
abstract type AbstractTree end

"""
    NodeIndex(::Type{T}) where T<:AbstractTree

Return the node index type for a given tree type.
This type is used to identify and access nodes within the tree.

# Implementation Requirements
Subtypes of [`AbstractTree`](@ref) must implement this method to specify
the type used for node indexing. For example:
- DfsTree might return `Int` for integer indices
- NodeTree might return the node/subtree type itself

There is also a convenience method `NodeIndex(tree::AbstractTree)` that calls
`NodeIndex(typeof(tree))` for tree instances.
"""
function NodeIndex(::Type{T}) where T<:AbstractTree end
NodeIndex(tree::AbstractTree) = NodeIndex(typeof(tree))

"""
    root(tree::AbstractTree)

Return the root node index of the tree, or `nothing` if the tree is empty.

# Implementation Requirements
Subtypes of [`AbstractTree`](@ref) must implement this method to provide
access to the tree's root node using the appropriate node index type.
For empty trees, this should return `nothing`.
"""
function root(tree::AbstractTree) end

"""
    setroot!(tree::AbstractTree, new_tree::AbstractTree)

Set a new tree as the root of the current tree.

# Arguments
- `tree`: The tree to modify (its root will be replaced)
- `new_tree`: The tree whose root will become the new root

# Returns
Returns a tree with the new root. The behavior depends on the implementation:
- For mutable trees (like `NodeTree`): Returns a new tree with the new root
- For array-based trees (like `DfsTree`): Modifies the tree in-place and returns it

# Implementation Requirements
Subtypes of [`AbstractTree`](@ref) must implement this method to:
1. Replace the current tree's root with the root from `new_tree`
2. Return the modified tree (either the same object or a new one)

# Usage
```julia
tree1 = NodeTree(1)
tree2 = NodeTree(2)
result = setroot!(tree1, tree2)  # result has root from tree2
```
"""
function setroot!(tree::AbstractTree, new_tree::AbstractTree) end

"""
    subtree(tree::AbstractTree, node_index)

Extract a subtree rooted at the given node index.

# Arguments
- `tree`: The source tree
- `node_index`: The node index to use as the root of the extracted subtree

# Returns
A new tree of the same type containing the subtree rooted at `node_index`.

# Implementation Requirements
Subtypes of [`AbstractTree`](@ref) must implement this method to enable
subtree extraction and manipulation operations.
"""
function subtree(tree::AbstractTree, node_index) end

"""
    children(tree::AbstractTree, node_index)

Return the child node indices of a given node in the tree.

# Arguments
- `tree`: The tree containing the node
- `node_index`: The index of the node whose children to retrieve

# Returns
A vector of node indices representing the children of the specified node.
"""
function children(tree::AbstractTree, node_index) end

"""
    setchild!(tree::AbstractTree, node_index, child_index, newchild::AbstractTree)

Set the `child_index`-th child of the node at `node_index` to `newchild`.
By default, this method will call `setroot!` if `node_index` is `nothing`.

# Arguments
- `tree`: The tree to modify
- `node_index`: The index of the parent node, or nothing for replacing the root
- `child_index`: The position of the child to replace (1-based), 1 for replacing the root
- `newchild`: The new subtree to set as the child

# Implementation Requirements
Subtypes of [`AbstractTree`](@ref) must implement this method to:
1. Check if `child_index` is valid (1 ≤ child_index ≤ arity(tree, node_index))
2. Throw a `BoundsError` if the index is invalid
3. Replace the `child_index`-th child with `newchild`
4. Specify the node index type to avoid ambiguity

# Returns
The modified tree (may be the same object or a new object depending on implementation).
"""
function setchild!(tree::AbstractTree, node_index, child_index, newchild::AbstractTree) end
setchild!(tree::AbstractTree, ::Nothing, child_index, newchild::AbstractTree) = setroot!(tree, newchild)

"""
    arity(tree::AbstractTree, node_index)

Return the arity (number of children) of a given node in the tree.
By default, the arity is the number of children of the node.
"""
arity(tree::AbstractTree, node_index) = length(children(tree, node_index))

"""
    isleaf(tree::AbstractTree, node_index)

Return true if the given node in the tree is a leaf node.
By default, a node is a leaf node if it has no children.
"""
isleaf(tree::AbstractTree, node_index) = arity(tree, node_index) == 0


########################################################
# 2. Base Iteration Interfaces
########################################################

"""
    Base.iterate(tree::AbstractTree)

Initialize the pre-order depth-first iteration for a tree.
Trees can be iterated directly without requiring a wrapper.

# Returns
- `(first_node_index, state)` where `state` contains information needed for subsequent iterations
- `nothing` if the tree is empty

# Default Implementation
A default implementation is provided that returns the tree root and initializes the traversal state.
"""
function Base.iterate(tree::AbstractTree)
    tree_root = root(tree)
    if length(tree) == 0 || tree_root === nothing  # Check if tree is empty
        return nothing
    end
    I = NodeIndex(typeof(tree))  # Return tree root first, with stack ready to process its children
    return tree_root, Tuple{I, Int}[(tree_root, 1)]
end

"""
    Base.iterate(tree::AbstractTree, state)

Continue the pre-order depth-first iteration from the given state.

# Arguments
- `tree`: The tree being iterated
- `state`: State information from previous iteration step

# Returns
- `(next_node_index, new_state)` for the next node in pre-order DFS order
- `nothing` if iteration is complete

# Default Implementation
Uses a stack-based approach with `(node_index, child_index)` pairs to track pre-order traversal state.
"""
function Base.iterate(tree::AbstractTree, stack::Vector)
    while !isempty(stack)  # Find next node to visit in pre-order
        parent_index, next_child_idx = stack[end]
        parent_children = children(tree, parent_index)
        if next_child_idx <= length(parent_children)
            next_node_index = parent_children[next_child_idx]  # Visit next child (pre-order: child visited before its siblings)
            stack[end] = (parent_index, next_child_idx + 1)  # Update parent's next child index
            push!(stack, (next_node_index, 1))  # Add child to stack (will process its children next)
            return next_node_index, stack
        else
            pop!(stack)  # No more children for this parent, backtrack
        end
    end
    return nothing  # No more nodes to visit
end

Base.eltype(tree::AbstractTree) = NodeIndex(typeof(tree))

########################################################
# 3. Traverse Interface with States
########################################################

"""
    AbstractTraverseState

Abstract base type for states used during tree traversal.

Required methods:
- [`init`](@ref): Initialize state instance
- [`enter!`](@ref): Called when entering a node (may be optional for default states if tree provides custom traverse)
- [`leave!`](@ref): Called when leaving a node (may be optional for default states if tree provides custom traverse)

Optional methods:
- [`requires`](@ref): Return dependency tuple (default: this state type)

# Implementation Notes
For default states included by a tree type (via `default_states`), the `enter!` and `leave!`
methods may be optional if the tree provides a customized `traverse` function that maintains
the state through other means.
"""
abstract type AbstractTraverseState end

"""
    requires(::Type{S}, tree::AbstractTree) where S<:AbstractTraverseState

Return tuple of CONCRETE dependency state types for the given tree. Default: `S`.

# Arguments
- `S`: The state type
- `tree`: The tree being traversed (allows state types to determine concrete types)
"""
requires(::Type{S}, tree::AbstractTree) where S<:AbstractTraverseState = (S,)

"""
    init(::Type{S}, tree, root) where S<:AbstractTraverseState

Initialize a state instance. Must be implemented by each state type.
"""
function init end

"""
    enter!(state, tree, index)

Called when entering a node. Must be implemented by each state type.
"""
function enter! end

"""
    leave!(state, tree, index)

Called when leaving a node. Must be implemented by each state type.
"""
function leave! end

"""
    StateBag

Container for traverse states with dependency ordering.
Provides fast lookup by state type and maintains topological order.

# Fields
- `order::Vector{AbstractTraverseState}`: States in dependency order
- `lookup::IdDict{DataType,AbstractTraverseState}`: Fast type-to-instance mapping
"""
struct StateBag
    order::Vector{AbstractTraverseState}
    lookup::IdDict{DataType,AbstractTraverseState}
end

"""
    getstate(bag, ::Type{S}) where S<:AbstractTraverseState

Get state instance of type S from bag.
This method can be overridden for lazy evaluation of states.
"""
getstate(bag::StateBag, ::Type{S}) where S<:AbstractTraverseState = bag.lookup[S]

"""
    default_states(tree::AbstractTree)

Return a vector of state types that should be automatically included for the given tree type.
Tree implementations can override this method to specify their required "default existing states".

# Default Implementation
Returns an empty vector (no default states).
"""
default_states(tree::AbstractTree) = []


function build_state_bag(state_types::Vector, tree::AbstractTree, root)
    order = AbstractTraverseState[]
    lookup = IdDict{DataType,AbstractTraverseState}()
    visiting = Set{DataType}()  # For cycle detection
    all_state_types = union(state_types, default_states(tree))  # Include default states for this tree type
    for state_type in all_state_types
        _ensure_state!(state_type, tree, root, order, lookup, visiting)
    end
    return StateBag(order, lookup)
end

function _ensure_state!(state_type::Any, tree, root, order, lookup, visiting)
    deps = requires(state_type, tree)
    concrete_state_type = deps[1]
    deps = deps[2:end]
    
    haskey(lookup, concrete_state_type) && return  # Already processed
    if concrete_state_type in visiting  # Cycle detection
        error("Cyclic dependency detected involving state type $concrete_state_type")
    end
    push!(visiting, concrete_state_type)
    
    for dep_type in deps
        _ensure_state!(dep_type, tree, root, order, lookup, visiting)
    end
    
    state_instance = init(concrete_state_type, tree, root)  # Create and store state instance
    lookup[concrete_state_type] = state_instance
    push!(order, state_instance)
    pop!(visiting, concrete_state_type)
    return nothing
end

"""
    Leave()

Return value for traverse functions indicating that the current node
should be left immediately without visiting its children.
"""
struct Leave end

"""
    Break()

Return value for traverse functions indicating that the traversal should terminate.
"""
struct Break{T}
    value::T
end
Break() = Break{Nothing}(nothing)

"""
    traverse(f, tree::AbstractTree, states...)

Traverse a tree in pre-order depth-first manner with custom states.

# Arguments
- `f`: User function called for each node. Should have signature `f(tree, node, bag::StateBag) -> Union{Nothing, Leave, Break}`
  - Return `Leave()` to skip children of current node
  - Return `Break(value)` to terminate the traversal and return the value
  - Return other values to continue the last returning value will be the result for the whole traversal.
- `tree`: The tree to traverse
- `states...`: Variable number of state types to initialize and maintain during traversal

# Returns
A tuple `(result, bag)` where:
- `result`: The value returned by the user function (typically `nothing`, or the value from `Break(value)`)
- `bag`: The `StateBag` containing final state after traversal

# Notes on Return Values
The `traverse` function enables values to be returned from the traversal process:
- For normal completion, `result` will be `nothing`
- For early termination via `Break(value)`, `result` will be the provided value
- This design allows both state accumulation (via `StateBag`) and direct result passing

# Example
```julia
# Define states
struct CountState <: AbstractTraverseState
    count::Ref{Int}
end
init(::Type{CountState}, tree, root) = CountState(Ref(0))

# Define traverse function
function count_nodes(tree, node, bag)
    state = getstate(bag, CountState)
    state.count[] += 1
    return nothing  # Continue traversal
end

# Traverse and get both result and state
result, bag = traverse(count_nodes, tree, CountState)
total = getstate(bag, CountState).count[]

# Example with Break() to return a value
function find_node_with_value(target_value)
    return function(tree, node, bag)
        if node.value == target_value
            return Break(node)  # Return the found node
        end
        return nothing
    end
end

found_node, bag = traverse(find_node_with_value(42), tree)
# found_node will be the node with value 42, or nothing if not found
```
"""
function traverse(f, tree::AbstractTree, states...)
    tree_root = root(tree)
    tree_root === nothing && return StateBag(AbstractTraverseState[], IdDict{DataType,AbstractTraverseState}())
    state_types = collect(states)  # Just collect all state types as-is
    bag = build_state_bag(state_types, tree, tree_root)
    result, _ = traverse(f, bag, tree, tree_root)  # Start traversal from root
    if result isa Break
        result = result.value
    end
    return result, bag
end

# Default traverse implementation with state enter!/leave! support
function traverse(f, bag::StateBag, tree::AbstractTree, node)
    # Call enter! for all states in dependency order
    (x->(enter!(x, tree, node))).(bag.order)
    result = f(tree, node, bag)  # Call user function for current node
    result isa Break && return result, bag # If user function returns Break(), terminate the traversal
    if !(result isa Leave)  # If user function returns Leave(), skip children
        for child in children(tree, node)  # Visit children
            result, _ = traverse(f, bag, tree, child)
            result isa Break && return result, bag # terminate and pass Break() to parent
        end
    end
    # Call leave! for all states in dependency order (not reverse!)
    (x->(leave!(x, tree, node))).(bag.order)
    return result, bag
end

########################################################
# 4. Built-in Traverse States
########################################################

"""
    DepthState

Tracks traversal depth. Root has depth 0, increments on enter, decrements on leave.
"""
mutable struct DepthState <: AbstractTraverseState
    depth::Int
end

init(::Type{DepthState}, ::AbstractTree, _) = DepthState(0)
enter!(state::DepthState, ::AbstractTree, _) = state.depth += 1
leave!(state::DepthState, ::AbstractTree, _) = state.depth -= 1

"""
    StackState{T}

A traverse state that maintains a stack of ancestor nodes and the current node, along with their child indices during tree traversal.
The stack represents the path from the root to the current node, tracking which child is being processed.

# Fields
- `stack::Vector{Tuple{T,Int}}`: Stack of (node_index, next_child_idx) pairs, 1-based

# Usage
```julia
function process_with_stack(tree, node, bag)
    stack = getstate(bag, StackState)
    println("Current path depth: ", length(stack.stack))
    return nothing
end

result, bag = traverse(process_with_stack, tree, StackState)
```

# Implementation Notes
This state is tree-agnostic and works with any tree implementation by using
the tree's NodeIndex type for the stack elements. The next_child_index tracks
which child is being processed for each ancestor in the path.
"""
mutable struct StackState{I} <: AbstractTraverseState
    stack::Vector{Tuple{I,Int}}
end
requires(::Type{StackState}, tree::AbstractTree) = (StackState{NodeIndex(tree)},)
requires(::Type{StackState{I}}, tree::AbstractTree) where I = (StackState{I},)
init(::Type{StackState{I}}, tree::AbstractTree, root) where I = StackState{I}([])

enter!(state::StackState{I}, tree::AbstractTree, node::I) where I = push!(state.stack, (node, 1))

function leave!(state::StackState{I}, tree::AbstractTree, node::I) where I
    isempty(state.stack) && return
    pop!(state.stack)
    # Update parent's next child index if we have a parent
    if !isempty(state.stack)
        parent_node, child_idx = state.stack[end]
        state.stack[end] = (parent_node, child_idx + 1)
    end
end

"""
    AncestorsState

A traverse state that provides lazy access to all ancestor nodes in the current
traversal path. Depends on StackState for the underlying path information.

# Fields
- `ancestors::Vector`: Ancestor nodes (computed lazily from StackState)
"""
struct AncestorsState{I} <: AbstractTraverseState
    ancestors::Vector{I}
end

requires(::Type{AncestorsState}, tree::AbstractTree) = (AncestorsState{NodeIndex(tree)}, StackState{NodeIndex(tree)})
requires(::Type{AncestorsState{I}}, tree::AbstractTree) where I = (AncestorsState{I}, StackState{I})
init(::Type{AncestorsState{I}}, tree::AbstractTree, root) where I = AncestorsState{I}(Vector{I}())
enter!(::AncestorsState{I}, ::AbstractTree, ::I) where I = nothing
leave!(::AncestorsState{I}, ::AbstractTree, ::I) where I = nothing

function getstate(bag::StateBag, ::Type{AncestorsState{I}}) where I
    ancestorstate = bag.lookup[AncestorsState{I}]
    stackstate = getstate(bag, StackState{I})
    # Extract just the node indices from the stack
    empty!(ancestorstate.ancestors)
    for (node, _) in stackstate.stack
        push!(ancestorstate.ancestors, node)
    end
    return ancestorstate
end

""" PointerState{I}

A traverse state that tracks the current node and its parent. Depends on StackState for the underlying path information.

# Fields
- `parent::Union{Nothing, I}`: The parent node index, or nothing for the root
- `child_index::Int`: The current child index, 1-based
"""
mutable struct PointerState{I} <: AbstractTraverseState
    parent::Union{Nothing, I}
    child_index::Int
end

requires(::Type{PointerState}, tree::AbstractTree) = (PointerState{NodeIndex(tree)}, StackState{NodeIndex(tree)})
requires(::Type{PointerState{I}}, tree::AbstractTree) where I = (PointerState{I}, StackState{I})
init(::Type{PointerState{I}}, tree::AbstractTree, root) where I = PointerState{I}(nothing, 1)
enter!(state::PointerState{I}, ::AbstractTree, node::I) where I = nothing
leave!(state::PointerState{I}, ::AbstractTree, node::I) where I = nothing

function getstate(bag::StateBag, ::Type{PointerState{I}}) where I
    pointerstate = bag.lookup[PointerState{I}]
    stackstate = getstate(bag, StackState{I})
    if length(stackstate.stack) > 1
        # Parent is one level up in the stack
        pointerstate.parent = stackstate.stack[end-1][1]
        # The parent's child_index in the stack represents the "next child to process"
        # During our visit, this index represents our current position in the parent's children
        # Since leave! increments this AFTER visiting, during our visit it's our 1-based position
        pointerstate.child_index = stackstate.stack[end-1][2]
    else
        # Root node has no parent
        pointerstate.parent = nothing
        pointerstate.child_index = 1  # Root is conceptually at position 1
    end
    return pointerstate
end


########################################################
# 5. NodeTree Reference Implementation
########################################################

"""
    Node{T}

A mutable node structure for use with [`NodeTree`](@ref).
Each node contains a value of type T and a list of children nodes.

# Fields
- `value::T`: The value stored in this node
- `children::Vector{Node{T}}`: List of child nodes
"""
mutable struct Node{T}
    value::T
    children::Vector{Node{T}}
end

Node(value::T) where T = Node{T}(value, Node{T}[])

"""
    NodeTree{T}

A simple tree implementation for testing the AbstractTree interface.
The tree consists of Node{T} objects where each node can be accessed directly.
The node index type for NodeTree is the Node{T} itself.

# Constructor
```julia
NodeTree(root_node::Node{T}) where T
NodeTree(value::T) where T  # Creates a tree with a single root node
```
"""
struct NodeTree{T} <: AbstractTree
    root::Node{T}
end

NodeTree(value::T) where T = NodeTree(Node(value))  # Convenience constructor for creating tree with single root

# AbstractTree interface implementation
NodeIndex(::Type{NodeTree{T}}) where T = Node{T}
root(tree::NodeTree) = tree.root
setroot!(tree::NodeTree{T}, new_tree::NodeTree{T}) where T = NodeTree{T}(new_tree.root)
subtree(tree::NodeTree{T}, node_index::Node{T}) where T = NodeTree{T}(node_index)  # Create new tree with this node as root
children(tree::NodeTree{T}, node_index::Node{T}) where T = node_index.children

function setchild!(tree::NodeTree{T}, node_index::Node{T}, child_index, newchild::NodeTree{T}) where T
    node_index.children[child_index] = newchild.root  # Let Julia handle bounds checking naturally
    return tree
end
Base.length(tree::NodeTree) = _count_nodes(tree.root)

function _count_nodes(node::Node{T}) where T
    1 + sum(_count_nodes, node.children; init=0)
end


########################################################
# 6. Reference Implementation
########################################################

include("dfstrees.jl")
include("utils.jl")

########################################################
# 7. Test Items
########################################################

@testsnippet TreesBaseImports begin
    using LambdaRegression.Trees: NodeTree, Node, NodeIndex, root, subtree, children
    using LambdaRegression.Trees: NodeTree, Node, setchild!, children, root
    using LambdaRegression.Trees: NodeTree, Node, root
    using LambdaRegression.Trees: NodeTree, Node, setroot!, root
    using LambdaRegression.Trees: NodeTree, AbstractTraverseState, StateBag, getstate, build_state_bag, requires, init, enter!, leave!
    import LambdaRegression.Trees: init, requires, enter!, leave!
    using LambdaRegression.Trees: NodeTree, Node, AbstractTraverseState, traverse, getstate, init, enter!, leave!
    import LambdaRegression.Trees: init, enter!, leave!
    using LambdaRegression.Trees: NodeTree, Node, AbstractTraverseState, traverse, getstate, Leave, init, enter!, leave!
    using LambdaRegression.Trees: NodeTree, Node, AbstractTraverseState, traverse, getstate, requires, init, enter!, leave!
    using LambdaRegression.Trees: NodeTree, Node, DepthState, traverse, getstate
    using LambdaRegression.Trees: NodeTree, Node, StackState, traverse, getstate
    using LambdaRegression.Trees: NodeTree, Node, AncestorsState, StackState, traverse, getstate
    using LambdaRegression.Trees: NodeTree, Node, traverse, Leave, StateBag
    using LambdaRegression.Trees: NodeTree, Node, PointerState, StackState, traverse, getstate
    using LambdaRegression.Trees: NodeTree, Node, PointerState, traverse, getstate
    using LambdaRegression.Trees: NodeTree, Node, PointerState, traverse, getstate, setchild!
    using LambdaRegression.Trees: NodeTree, Node, AbstractTraverseState, traverse, getstate, Break, init, enter!, leave!
    using LambdaRegression.Trees: NodeTree, Node, lca, root
    using LambdaRegression.Trees: NodeTree, Node, setindex!, root
end

@testsnippet TreesCountStateSetup begin
    mutable struct CountState <: AbstractTraverseState
        count::Int
    end
    init(::Type{CountState}, _, _) = CountState(0)
    enter!(state::CountState, _, _) = (state.count += 1)
    leave!(::CountState, _, _) = nothing
end

@testitem "NodeIndex and basic interface" setup=[TreesBaseImports] begin
    tree = NodeTree(1)  # Test NodeIndex
    @test NodeIndex(typeof(tree)) == Node{Int}
    
    root_node = root(tree)  # Test root
    @test root_node isa Node{Int}
    @test root_node.value == 1
    
    subtree_result = subtree(tree, root_node)  # Test subtree (creates new tree with node as root)
    @test subtree_result isa NodeTree{Int}
    @test root(subtree_result) === root_node
    
    child1 = Node(2)  # Test children
    child2 = Node(3)
    push!(root_node.children, child1, child2)
    
    tree_children = children(tree, root_node)
    @test tree_children == [child1, child2]
    @test length(tree_children) == 2
end

@testitem "setchild! function" setup=[TreesBaseImports] begin
    child1 = Node(2)  # Test valid replacement
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    parent = NodeTree(root_node)
    
    new_child = NodeTree(4)
    
    result = setchild!(parent, root_node, 1, new_child)
    @test result === parent  # Returns the tree
    @test children(parent, root_node)[1] === new_child.root
    @test children(parent, root_node)[2] === child2
    
    @test_throws BoundsError setchild!(parent, root_node, 0, new_child)  # Test bounds checking (Julia's natural error)
    @test_throws BoundsError setchild!(parent, root_node, 4, new_child)
    
    leaf = NodeTree(1)  # Test empty children
    @test_throws BoundsError setchild!(leaf, root(leaf), 1, new_child)
end

@testitem "Direct tree iteration" setup=[TreesBaseImports] begin
    single = NodeTree(1)  # Test single node
    result = collect(single)
    @test length(result) == 1
    @test result[1] === root(single)
    
    child1 = Node(2)  # Test simple tree: 1 -> [2, 3]
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    result = collect(tree)
    @test length(result) == 3
    @test result[1] === root_node
    @test result[2] === child1  
    @test result[3] === child2
    
    grandchild = Node(4)  # Test deeper tree: 1 -> [2 -> [4], 3]
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    deep_root = Node(1)
    push!(deep_root.children, child_with_child, child2)
    deep_tree = NodeTree(deep_root)
    
    result = collect(deep_tree)
    @test length(result) == 4
    @test result[1] === deep_root
    @test result[2] === child_with_child
    @test result[3] === grandchild
    @test result[4] === child2
end

@testitem "setroot! function" setup=[TreesBaseImports] begin
    original_tree = NodeTree(1)  # Test basic root replacement
    new_tree = NodeTree(10)
    
    result = setroot!(original_tree, new_tree)
    @test result isa NodeTree{Int}
    @test root(result).value == 10
    
    child1 = Node(2)  # Test that original tree structure is preserved in result
    child2 = Node(3)
    complex_root = Node(5)
    push!(complex_root.children, child1, child2)
    complex_tree = NodeTree(complex_root)
    
    simple_tree = NodeTree(100)
    result = setroot!(simple_tree, complex_tree)
    @test root(result).value == 5
    @test length(root(result).children) == 2
    @test root(result).children[1].value == 2
    @test root(result).children[2].value == 3
end

@testitem "StateBag construction and lookup" setup=[TreesBaseImports] begin
    struct StateA <: AbstractTraverseState; value::Int; end
    struct StateB <: AbstractTraverseState; value::String; end  
    struct StateC <: AbstractTraverseState; value::Float64; end
    
    requires(::Type{StateC}, tree::LambdaRegression.Trees.AbstractTree) = (StateC, StateA, StateB)
    init(::Type{StateA}, _, _) = StateA(1)
    init(::Type{StateB}, _, _) = StateB("test")  
    init(::Type{StateC}, _, _) = StateC(3.14)
    enter!(::StateA, _, _) = nothing
    enter!(::StateB, _, _) = nothing
    enter!(::StateC, _, _) = nothing
    leave!(::StateA, _, _) = nothing
    leave!(::StateB, _, _) = nothing
    leave!(::StateC, _, _) = nothing
    
    tree = NodeTree(1)
    bag = build_state_bag([StateC], tree, LambdaRegression.Trees.root(tree))
    
    @test length(bag.order) == 3
    @test haskey(bag.lookup, StateA)
    @test haskey(bag.lookup, StateB) 
    @test haskey(bag.lookup, StateC)
    
    # Test dependency ordering
    state_types = [typeof(s) for s in bag.order]
    a_idx = findfirst(==(StateA), state_types)
    c_idx = findfirst(==(StateC), state_types)
    @test a_idx < c_idx
    
    @test getstate(bag, StateA).value == 1
    @test getstate(bag, StateB).value == "test"
    @test getstate(bag, StateC).value == 3.14
end

@testitem "traverse basic functionality" setup=[TreesBaseImports, TreesCountStateSetup] begin
    # Create test tree: 1 -> [2, 3]
    child1, child2 = Node(2), Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    count_all(_, _, _) = nothing
    result, bag = traverse(count_all, tree, CountState)
    @test getstate(bag, CountState).count == 3
end

@testitem "traverse with Leave() return" setup=[TreesBaseImports, TreesCountStateSetup] begin
    # Create tree: 1 -> [2 -> [4], 3]
    grandchild = Node(4)
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child_with_child, child2)
    tree = NodeTree(root_node)
    
    count_and_skip_2(_, node, bag) = node.value == 2 ? Leave() : nothing
    
    result, bag = traverse(count_and_skip_2, tree, CountState)
    @test getstate(bag, CountState).count == 3  # Skip grandchild(4)
end

@testitem "traverse with state dependencies" setup=[TreesBaseImports, TreesCountStateSetup] begin
    mutable struct SumState <: AbstractTraverseState
        total::Int
    end
    init(::Type{SumState}, _, _) = SumState(0)
    enter!(state::SumState, _, node) = (state.total += node.value)
    leave!(::SumState, _, _) = nothing
    requires(::Type{CountState}, tree::LambdaRegression.Trees.AbstractTree) = (CountState, SumState)
    
    
    # Create test tree: 1 -> [2, 3]
    child1, child2 = Node(2), Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    sum_and_count(_, _, _) = nothing
    
    _, bag = traverse(sum_and_count, tree, CountState)
    
    @test getstate(bag, SumState).total == 6   # 1 + 2 + 3
    @test getstate(bag, CountState).count == 3  # 3 nodes
    
    # Verify dependency ordering
    state_types = [typeof(s) for s in bag.order]
    sum_idx = findfirst(==(SumState), state_types)
    count_idx = findfirst(==(CountState), state_types)
    @test sum_idx < count_idx
end

@testitem "DepthState functionality" setup=[TreesBaseImports] begin
    # Create tree: 1 -> [2 -> [4], 3]
    grandchild = Node(4)
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child_with_child, child2)
    tree = NodeTree(root_node)
    
    collected_depths = Tuple{Int, Int}[]
    collect_depths(_, node, bag) = push!(collected_depths, (node.value, getstate(bag, DepthState).depth))
    
    _, bag = traverse(collect_depths, tree, DepthState)
    
    @test length(collected_depths) == 4
    @test (1, 1) in collected_depths  # Root depth after enter
    @test (2, 2) in collected_depths  # Child depth after enter  
    @test (3, 2) in collected_depths  # Child depth after enter
    @test (4, 3) in collected_depths  # Grandchild depth after enter
    
    @test getstate(bag, DepthState).depth == 0  # Back to 0 after all leaves
end

@testitem "StackState functionality" setup=[TreesBaseImports] begin
    # Create tree: 1 -> [2 -> [4], 3]
    grandchild = Node(4)
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child_with_child, child2)
    tree = NodeTree(root_node)
    
    # Collect stack information during traversal
    stack_info = Tuple{Int, Vector{Tuple{Node{Int}, Int}}}[]  # (node_value, stack_contents)
    
    function collect_stack_info(tree, node, bag)
        stack = getstate(bag, StackState{Node{Int}})
        # Copy the stack to avoid mutation issues
        stack_copy = copy(stack.stack)
        push!(stack_info, (node.value, stack_copy))
        return nothing
    end
    
    _, bag = traverse(collect_stack_info, tree, StackState)
    
    @test length(stack_info) == 4  # 4 nodes total
    
    # Verify stack contents for each node
    # Root node: stack should contain just the root
    @test stack_info[1][1] == 1
    @test length(stack_info[1][2]) == 1
    @test stack_info[1][2][1][1] === root_node
    @test stack_info[1][2][1][2] == 1  # next_child_index
    
    # First child: stack should contain root and first child
    @test stack_info[2][1] == 2
    @test length(stack_info[2][2]) == 2
    @test stack_info[2][2][1][1] === root_node
    @test stack_info[2][2][2][1] === child_with_child
    
    # Grandchild: stack should contain root, first child, and grandchild
    @test stack_info[3][1] == 4
    @test length(stack_info[3][2]) == 3
    @test stack_info[3][2][3][1] === grandchild
    
    # Second child: stack should contain root and second child
    @test stack_info[4][1] == 3
    @test length(stack_info[4][2]) == 2
    @test stack_info[4][2][1][1] === root_node
    @test stack_info[4][2][2][1] === child2
    
    # Final state should have empty stack
    @test isempty(getstate(bag, StackState{Node{Int}}).stack)
end

@testitem "AncestorsState functionality" setup=[TreesBaseImports] begin
    # Create test tree: (1 -> [2 -> [3], 4])
    var_3 = Node(3)
    var_2 = Node(2)
    push!(var_2.children, var_3)
    var_4 = Node(4)
    root_node = Node(1)
    push!(root_node.children, var_2, var_4)
    tree = NodeTree(root_node)
    
    # Collect ancestors information
    ancestors_info = Tuple{Int, Vector{Int}}[]  # (node_value, ancestor_values)
    
    function collect_ancestors_info(tree, node, bag)
        ancestors = getstate(bag, AncestorsState{Node{Int}})
        node_value = node.value
        ancestor_values = [a.value for a in ancestors.ancestors]
        push!(ancestors_info, (node_value, ancestor_values))
        return nothing
    end
    
    _, bag = traverse(collect_ancestors_info, tree, AncestorsState)
    
    @test length(ancestors_info) == 4  # 4 nodes total
    
    # Root should have itself as ancestor (due to enter! being called before user function)
    @test ancestors_info[1] == (1, [1])
    
    # First child should have root and itself
    @test ancestors_info[2] == (2, [1, 2])
    
    # Grandchild should have root, first child, and itself
    @test ancestors_info[3] == (3, [1, 2, 3])
    
    # Second child should have root and itself
    @test ancestors_info[4] == (4, [1, 4])
    
    # Verify that AncestorsState depends on StackState
    @test haskey(bag.lookup, StackState{Node{Int}})  # StackState should be included automatically
end

@testitem "AncestorsState lazy evaluation" setup=[TreesBaseImports] begin
    # Create simple tree: 1 -> [2, 3]
    child1 = Node(2)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    # Test that AncestorsState correctly reflects StackState changes
    verification_count = Ref(0)
    
    function verify_ancestors_match_stack(tree, node, bag)
        stack = getstate(bag, StackState{Node{Int}})
        ancestors = getstate(bag, AncestorsState{Node{Int}})
        
        # Verify that ancestors matches stack (extract node from tuple)
        @test length(ancestors.ancestors) == length(stack.stack)
        for i in 1:length(stack.stack)
            @test ancestors.ancestors[i] === stack.stack[i][1]  # Extract node from (node, child_idx) tuple
        end
        
        verification_count[] += 1
        return nothing
    end
    
    _, bag = traverse(verify_ancestors_match_stack, tree, AncestorsState)
    
    @test verification_count[] == 3  # Verified for all 3 nodes
end

@testitem "traverse stateless" setup=[TreesBaseImports] begin
    # Create tree: 1 -> [2, 3]
    child1, child2 = Node(2), Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    visited_values = Int[]
    collect_values(_, node, _) = push!(visited_values, node.value)
    
    _, bag = traverse(collect_values, tree)
    @test bag isa StateBag
    @test length(bag.order) == 0
    @test visited_values == [1, 2, 3]
    
    # Test with Leave()
    visited_with_leave = Int[]
    collect_and_skip_2(_, node, _) = (push!(visited_with_leave, node.value); node.value == 2 ? Leave() : nothing)
    
    grandchild = Node(4)
    push!(child1.children, grandchild)
    
    _, bag = traverse(collect_and_skip_2, tree)
    @test visited_with_leave == [1, 2, 3]  # Skipped grandchild(4)
end

@testitem "PointerState functionality" setup=[TreesBaseImports] begin
    # Create test tree: 1 -> [2 -> [4], 3]
    grandchild = Node(4)
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child_with_child, child2)
    tree = NodeTree(root_node)
    
    # Collect pointer information during traversal
    pointer_info = Tuple{Int, Union{Nothing, Int}, Int}[]  # (node_value, parent_value_or_nothing, child_index)
    
    function collect_pointer_info(tree, node, bag)
        pointer = getstate(bag, PointerState{Node{Int}})
        node_value = node.value
        parent_value = pointer.parent === nothing ? nothing : pointer.parent.value
        child_index = pointer.child_index
        push!(pointer_info, (node_value, parent_value, child_index))
        return nothing
    end
    
    _, bag = traverse(collect_pointer_info, tree, PointerState)
    
    @test length(pointer_info) == 4  # 4 nodes total
    
    # Root node: no parent, child_index = 1 
    @test pointer_info[1] == (1, nothing, 1)
    
    # First child (node 2): parent is root (1), child_index = 1 (first child of root)
    @test pointer_info[2] == (2, 1, 1)
    
    # Grandchild (node 4): parent is node 2, child_index = 1 (first child of node 2)
    @test pointer_info[3] == (4, 2, 1)
    
    # Second child (node 3): parent is root (1), child_index = 2 (second child of root)
    @test pointer_info[4] == (3, 1, 2)
    
    # Verify that PointerState depends on StackState
    @test haskey(bag.lookup, StackState{Node{Int}})  # StackState should be included automatically
end

@testitem "PointerState edge cases" setup=[TreesBaseImports] begin
    # Test single node tree (root only)
    single_tree = NodeTree(42)
    
    single_pointer_info = Tuple{Int, Union{Nothing, Int}, Int}[]
    
    function collect_single_pointer_info(tree, node, bag)
        pointer = getstate(bag, PointerState{Node{Int}})
        node_value = node.value
        parent_value = pointer.parent === nothing ? nothing : pointer.parent.value
        child_index = pointer.child_index
        push!(single_pointer_info, (node_value, parent_value, child_index))
        return nothing
    end
    
    _, bag = traverse(collect_single_pointer_info, single_tree, PointerState)
    
    @test length(single_pointer_info) == 1
    @test single_pointer_info[1] == (42, nothing, 1)  # Root has no parent, child_index = 1
    
    # Test tree with multiple children at same level: 1 -> [2, 3, 4, 5]
    multi_child_tree = NodeTree(Node(1))
    for i in 2:5
        push!(multi_child_tree.root.children, Node(i))
    end
    
    multi_pointer_info = Tuple{Int, Union{Nothing, Int}, Int}[]
    
    function collect_multi_pointer_info(tree, node, bag)
        pointer = getstate(bag, PointerState{Node{Int}})
        node_value = node.value
        parent_value = pointer.parent === nothing ? nothing : pointer.parent.value
        child_index = pointer.child_index
        push!(multi_pointer_info, (node_value, parent_value, child_index))
        return nothing
    end
    
    _, bag = traverse(collect_multi_pointer_info, multi_child_tree, PointerState)
    
    @test length(multi_pointer_info) == 5
    @test multi_pointer_info[1] == (1, nothing, 1)  # Root
    @test multi_pointer_info[2] == (2, 1, 1)        # First child
    @test multi_pointer_info[3] == (3, 1, 2)        # Second child
    @test multi_pointer_info[4] == (4, 1, 3)        # Third child
    @test multi_pointer_info[5] == (5, 1, 4)        # Fourth child
end

@testitem "PointerState for tree modification" setup=[TreesBaseImports] begin
    # Create test tree: 1 -> [2, 3]
    child1 = Node(2)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    # Test using PointerState to replace nodes during traversal
    modifications_count = Ref(0)
    
    function replace_node_2_with_99(tree, node, bag)
        if node.value == 2
            pointer = getstate(bag, PointerState{Node{Int}})
            if pointer.parent !== nothing
                # Replace node 2 with node 99
                new_subtree = NodeTree(Node(99))
                setchild!(tree, pointer.parent, pointer.child_index, new_subtree)
                modifications_count[] += 1
            end
        end
        return nothing
    end
    
    _, bag = traverse(replace_node_2_with_99, tree, PointerState)
    
    @test modifications_count[] == 1
    @test tree.root.children[1].value == 99  # Node 2 should be replaced with 99
    @test tree.root.children[2].value == 3   # Node 3 should remain unchanged
end

@testitem "traverse with Break() return" setup=[TreesBaseImports, TreesCountStateSetup] begin
    # Create tree: 1 -> [2 -> [4], 3]
    grandchild = Node(4)
    child_with_child = Node(2)
    push!(child_with_child.children, grandchild)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child_with_child, child2)
    tree = NodeTree(root_node)

    # Traverse and break when node value == 2
    function count_until_two(_, node, bag)
        if node.value == 2
            return Break()
        end
        return nothing
    end

    _, bag = traverse(count_until_two, tree, CountState)

    # Only nodes 1 and 2 should have been visited (CountState enter! is called before user function)
    @test getstate(bag, CountState).count == 2
end

@testitem "lca function basic functionality" setup=[TreesBaseImports] begin
    # Test 1: Single node matches query
    tree = NodeTree(1)
    result = lca((tree, node) -> node.value == 1, tree, root(tree))
    @test result === root(tree)

    # Test 2: No nodes match query
    result = lca((tree, node) -> node.value == 999, tree, root(tree))
    @test result === nothing

    # Test 3: Simple tree with one matching node
    #   1
    #  / \
    # 2   3
    child1 = Node(2)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    # Find node with value 2
    result = lca((tree, node) -> node.value == 2, tree, root(tree))
    @test result === child1

    # Test 4: Multiple nodes match - should return LCA
    #   1
    #  / \
    # 2   2
    child1_dup = Node(2)
    child2_dup = Node(2)
    root_node_dup = Node(1)
    push!(root_node_dup.children, child1_dup, child2_dup)
    tree_dup = NodeTree(root_node_dup)
    
    # Both children have value 2, so LCA should be root
    result = lca((tree, node) -> node.value == 2, tree_dup, root(tree_dup))
    @test result === root_node_dup
end

@testitem "lca function complex tree scenarios" setup=[TreesBaseImports] begin
    # Test complex tree:
    #       1
    #      / \
    #     2   3
    #    /   / \
    #   4   5   6
    #      /
    #     7
    
    node7 = Node(7)
    node4 = Node(4)
    node5 = Node(5)
    push!(node5.children, node7)
    node6 = Node(6)
    node2 = Node(2)
    push!(node2.children, node4)
    node3 = Node(3)
    push!(node3.children, node5, node6)
    root_node = Node(1)
    push!(root_node.children, node2, node3)
    tree = NodeTree(root_node)
    
    # Test 1: Find LCA of nodes 4 and 7 (should be root)
    result = lca((tree, node) -> node.value == 4 || node.value == 7, tree, root(tree))
    @test result === root_node
    
    # Test 2: Find LCA of nodes 5 and 6 (should be node3)
    result = lca((tree, node) -> node.value == 5 || node.value == 6, tree, root(tree))
    @test result === node3
    
    # Test 3: Find LCA of just node 7 (should be node7 itself)
    result = lca((tree, node) -> node.value == 7, tree, root(tree))
    @test result === node7
    
    # Test 4: Find LCA in subtree (search only in node3's subtree)
    result = lca((tree, node) -> node.value == 5 || node.value == 6, tree, node3)
    @test result === node3
    
    # Test 5: Query that matches nodes in different subtrees
    result = lca((tree, node) -> node.value == 2 || node.value == 3, tree, root(tree))
    @test result === root_node
end

@testitem "setindex! function functionality" setup=[TreesBaseImports] begin
    # Test basic replacement
    #   1
    #  / \
    # 2   3
    # Replace node 2 with new subtree
    child1 = Node(2)
    child2 = Node(3)
    root_node = Node(1)
    push!(root_node.children, child1, child2)
    tree = NodeTree(root_node)
    
    # Create replacement subtree
    new_subtree = NodeTree(Node(999))
    
    # Replace child1 with new_subtree
    result_tree = setindex!(tree, new_subtree, child1)
    
    # Verify replacement
    @test root_node.children[1].value == 999
    @test root_node.children[2] === child2  # Other child unchanged
    
    # Test replacing root
    original_tree = NodeTree(Node(100))
    replacement = NodeTree(Node(200))
    result_tree = setindex!(original_tree, replacement, root(original_tree))
    display(result_tree)
    @test root(result_tree).value == 200
end

end # module Trees

