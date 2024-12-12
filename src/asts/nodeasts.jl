"""
    NodeASTs

A module providing Abstract Syntax Tree (AST) nodes for locally nameless De Bruijn representations
in lambda calculus and functional programming languages.

# Overview

This module implements AST nodes suitable for the "locally nameless" representation, which combines
the benefits of named variables (for readability and debugging) with De Bruijn indices (for
efficient substitution and alpha-equivalence checking). This representation is particularly
effective for:

- Lambda calculus implementations with alpha-equivalence
- Type checking and inference algorithms
- Program transformations and optimizations
- Integration with Hindley-Milner type systems

# Design Philosophy

## Locally Nameless Representation

The locally nameless approach uses:
- **Names** for free variables (for readability and debugging)
- **De Bruijn indices** for bound variables (for efficient manipulation)

This hybrid approach provides the advantages of both representations while mitigating their
respective drawbacks.

## Node Types

The module provides several categories of AST nodes:

### Core Lambda Calculus Nodes
- [`AppNode`](@ref): Function application (M N)
- [`AbsNode`](@ref): Lambda abstraction (λx. M)
- [`VarNode`](@ref): Free variables
- [`IndexNode`](@ref): Bound variables (De Bruijn indices)
- [`ConstNode`](@ref): Constants and built-in functions

### Extended Nodes
- [`LetNode`](@ref): Let-binding (deprecated in favor of application form)

## Tree Iteration and Traversal

The module provides comprehensive depth-first iteration capabilities through integration
with the Trees.jl interface:

- **Basic DFS**: Pre-order traversal using `Dfs(tree)`
- **State-aware DFS**: Traversal with additional context using `DfsWithStates(tree, StateType)`
- **Pointer-based modification**: Tree modification during traversal with pointer states

### Built-in Traversal States

- [`DepthState`](@ref): Tracks tree depth (distance from root)
- [`StackState`](@ref): Maintains stack of node indices during traversal
- [`AncestorsState`](@ref): Provides lazy access to ancestor nodes (depends on StackState)
- [`ScopeState`](@ref): Provides lazy access to abstraction nodes in scope (depends on AncestorsState)
- [`AbstractionDepthState`](@ref): Counts enclosing λ-abstractions (depends on ScopeState)
- [`DefaultPointerState`](@ref): Enables tree modification during traversal

## Display Conventions

The display system implements two modes:
- **Plain show**: Informative representation showing internal structure
- **Pretty display**: Human-readable mathematical notation

Pretty display follows these conventions:
- Generated variables use mathematical italic Unicode (𝑎, 𝑏, 𝑐, ...)
- User-defined variables retain their original names
- Bindings and bodies are separated with "·"
- Parentheses are inserted only when necessary for precedence
- De Bruijn indices are shown as #n when names are unavailable

## Symbolic Regression Optimizations

This implementation is optimized for symbolic regression applications:
- Symbol generation uses numeric identifiers (`Symbol(0)`, `Symbol(1)`, ...) for performance
- Numeric symbols are displayed as mathematical italic letters for readability
- No conflict checking performed during generation for efficiency
- Constant names are expected to be legal Julia variable names (documented but not enforced)

# Usage Examples

```julia
# Create a simple lambda expression: λx. x
abs_node = AbsNode(:x, VarNode(:x))
tree = AstTree(abs_node)

# Display shows: λx·x
display(tree)

# Create application: (λx. x) y  
app_tree = AstTree(AppNode(abs_node, VarNode(:y)))

# Display shows: (λx·x) y
display(app_tree)

# Create application with constant: f pi
const_app = AppNode(VarNode(:f), ConstNode(:pi))

# Create let-binding equivalent: let x = M in N
# Note: Prefer λx.N M form over LetNode
let_equivalent = AppNode(AbsNode(:x, VarNode(:x)), VarNode(:y))
```

# Implementation Notes

All AST nodes implement the Trees.AbstractTree interface, enabling seamless integration
with tree traversal and manipulation algorithms. The nodes are designed to be mutable
where appropriate to support efficient tree transformations.

The symbol generation system ensures unique variable names within each tree context,
automatically resolving naming conflicts without requiring global coordination.
"""
module NodeASTs

import ..Trees: AbstractTree, NodeIndex, root, setroot!, subtree, children, setchild!, arity, isleaf, getstate
using TestItems

# Export main types and functions
export AstNode, AstTree
export AppNode, AbsNode, VarNode, ConstNode, IndexNode, LetNode
export SymbolGenerator, generate_fresh_name


########################################################
# Symbol Generation for Abstraction Naming
########################################################

"""
    SymbolGenerator

A simple and efficient symbol generator for symbolic regression.
Generates lowercase alphabetic names in sequence: a, b, c, ..., z, aa, ab, ...

Since this is designed for symbolic regression rather than parsing human code,
no name conflict checking is performed for efficiency.
"""
mutable struct SymbolGenerator
    counter::Int
end

SymbolGenerator() = SymbolGenerator(0)

"""
    generate_fresh_name(gen::SymbolGenerator)

Generate the next fresh variable name as a `Symbol("<n>")`, where `<n>` is an
integer counter starting at `0`. This numeric naming avoids collisions with
user-defined alphabetic symbols while remaining cheap to generate.
"""
function generate_fresh_name(gen::SymbolGenerator)
    name = Symbol(gen.counter)
    gen.counter += 1
    return name
end

# -------------------------------------------------------------------------
# Pretty-printing helpers for numeric symbols
# -------------------------------------------------------------------------

"""
    _index_to_italic(n::Int) -> String

Convert a non-negative integer `n` to its (possibly multi-letter) Unicode
*mathematical italic* representation. For example

```julia
_index_to_italic(0)  == "𝑎"
_index_to_italic(25) == "𝑧"
_index_to_italic(26) == "𝑎𝑎"
``` 
"""
function _index_to_italic(n::Int)
    n < 0 && error("index must be non-negative")
    letters = Char[]
    while true
        pushfirst!(letters, Char(0x1D44E + (n % 26)))  # 𝑎 starts at U+1D44E
        n = div(n, 26) - 1
        n < 0 && break
    end
    return String(letters)
end

"""
    _display_var(name::Symbol) -> String

Return the pretty-printed form of a variable `name`. Numeric symbols (those
whose *string* is composed solely of digits) are rendered as mathematical
italic letters produced by `_index_to_italic`. All other symbols are returned
verbatim via `string(name)`.
"""
function _display_var(name::Symbol)
    s = String(name)
    all(isdigit, s) ? _index_to_italic(parse(Int, s)) : s
end

########################################################
# Abstract AST Node
########################################################

"""
    AstNode

Abstract base type for all AST nodes in the locally nameless representation.

All concrete subtypes must implement:
- [`children`](@ref): Return child nodes
- [`setchild!`](@ref): Set a specific child node  
- [`arity`](@ref): Return number of children
- [`isleaf`](@ref): Return whether node has no children

Additional methods for display and manipulation are provided by default.
"""
abstract type AstNode end

########################################################
# AST Tree Container
########################################################

"""
    AstTree <: AbstractTree

A tree container for AST nodes implementing the Trees.AbstractTree interface.
For AstTree, the node index type is AstNode itself, meaning nodes can be directly
accessed and manipulated.

# Constructor
```julia
AstTree(root_node::AstNode)
```

# Tree Interface
The tree implements all standard tree operations including traversal, modification,
and subtree extraction.
"""
mutable struct AstTree <: AbstractTree
    root::AstNode
end

NodeIndex(::Type{AstTree}) = AstNode
root(tree::AstTree) = tree.root
setroot!(tree::AstTree, node::AstNode) = (tree.root = node; tree)
setroot!(tree::AstTree, new_tree::AstTree) = setroot!(tree, new_tree.root)
subtree(::AstTree, index::AstNode) = AstTree(index)

children(::AstTree, index::AstNode) = children(index)
setchild!(::AstTree, index::AstNode, childindex::Int, newchild::AstNode) = setchild!(index, childindex, newchild)
setchild!(::AstTree, index::AstNode, childindex::Int, newchild::AstTree) = setchild!(index, childindex, newchild.root)
arity(::AstTree, node::AstNode) = arity(node)
isleaf(::AstTree, node::AstNode) = isleaf(node)

Base.length(tree::AstTree) = _count_nodes(tree.root)

function _count_nodes(node::AstNode)
    1 + sum(_count_nodes, children(node); init=0)
end

########################################################
# Core Lambda Calculus AST Nodes
########################################################

"""
    AppNode <: AstNode

Function application node representing M N in lambda calculus.

# Fields
- `func::AstNode`: The function being applied
- `arg::AstNode`: The argument to the function

# Mathematical Notation
Displayed as: `M N` or `(M) N` when parentheses are needed for precedence.

# Examples
```julia
# Create (f x)
AppNode(VarNode(:f), VarNode(:x))

# Create ((f x) y) - left associative
AppNode(AppNode(VarNode(:f), VarNode(:x)), VarNode(:y))
```
"""
mutable struct AppNode <: AstNode
    func::AstNode
    arg::AstNode
end

children(node::AppNode) = [node.func, node.arg]
function setchild!(node::AppNode, index::Int, newchild::AstNode)
    if index == 1
        node.func = newchild
    elseif index == 2
        node.arg = newchild
    else
        _throw_children_boundserror(node, index)
    end
    return node
end
arity(::AppNode) = 2
isleaf(::AppNode) = false

"""
    AbsNode <: AstNode

Lambda abstraction node representing λx. M in lambda calculus.

# Fields
- `name::Symbol`: The parameter name (for display and debugging)
- `body::AstNode`: The body of the abstraction

# Mathematical Notation
Displayed as: `λx·M` where the parameter and body are separated by "·".

# Examples
```julia
# Create λx. x (identity function)
AbsNode(:x, VarNode(:x))

# Create λf. λx. f x (function application)
AbsNode(:f, AbsNode(:x, AppNode(VarNode(:f), VarNode(:x))))
```

# Implementation Notes
In the locally nameless representation, bound variables in the body should use
[`IndexNode`](@ref) with appropriate De Bruijn indices rather than [`VarNode`](@ref).
"""
mutable struct AbsNode <: AstNode
    name::Symbol
    body::AstNode
end

children(node::AbsNode) = [node.body]
function setchild!(node::AbsNode, index::Int, newchild::AstNode)
    if index == 1
        node.body = newchild
    else
        _throw_children_boundserror(node, index)
    end
    return node
end
arity(::AbsNode) = 1
isleaf(::AbsNode) = false

"""
    VarNode <: AstNode

Free variable node representing unbound variables in lambda calculus.

# Fields
- `name::Symbol`: The variable name

# Mathematical Notation
Displayed as: the variable name itself (e.g., `x`, `y`, `f`).

# Examples
```julia
# Create free variable x
VarNode(:x)

# Create free variable with descriptive name
VarNode(:factorial)
```

# Implementation Notes
In locally nameless representation, `VarNode` should only be used for free variables.
Bound variables should use [`IndexNode`](@ref) with De Bruijn indices.
"""
struct VarNode <: AstNode
    name::Symbol
end

children(::VarNode) = []
setchild!(node::VarNode, index::Int, newchild::AstNode) = _throw_children_boundserror(node, index)
arity(::VarNode) = 0
isleaf(::VarNode) = true

"""
    ConstNode <: AstNode

Constant/primitive node representing built-in functions and literals.

# Fields
- `name::Symbol`: The constant identifier (must be a legal Julia variable name)

# Mathematical Notation
Displayed as: the constant name itself.

# Examples
```julia
# Create built-in function
ConstNode(:add)

# Create mathematical constant
ConstNode(:pi)

# Create boolean constant  
ConstNode(:true)
```

# Implementation Notes
Constants represent elements that are neither variables nor lambda constructs,
such as primitive operations, literals, and built-in functions.

**Important**: The `name` field must be a legal Julia variable name (e.g., `:add`, `:pi`, `:x1`).
Invalid names like `:42` or `:1` should not be used. This constraint is documented
but not enforced for performance reasons in symbolic regression contexts.
"""
struct ConstNode <: AstNode
    name::Symbol
end

children(::ConstNode) = []
setchild!(node::ConstNode, index::Int, newchild::AstNode) = _throw_children_boundserror(node, index)
arity(::ConstNode) = 0
isleaf(::ConstNode) = true

########################################################
# Extension for Locally Nameless De Bruijn AST Nodes
########################################################

"""
    IndexNode <: AstNode

De Bruijn index node representing bound variables in locally nameless representation.

# Fields
- `index::Int`: The De Bruijn index (0-based: 0 for innermost binding, 1 for next outer, etc.)

# Mathematical Notation
Displayed as: `#n` where n is the index, or the corresponding parameter name when available.

# Implementation Notes
In locally nameless representation:
- Index 0 refers to the innermost lambda parameter
- Index 1 refers to the next outer lambda parameter, etc.
- Indices should be properly maintained during substitution and tree transformations
"""
struct IndexNode <: AstNode
    index::Int
end

children(::IndexNode) = []
setchild!(node::IndexNode, index::Int, newchild::AstNode) = _throw_children_boundserror(node, index)
arity(::IndexNode) = 0
isleaf(::IndexNode) = true

########################################################
# Extension for Hindley-Milner System
########################################################

"""
    LetNode <: AstNode

Let-binding node for the Hindley-Milner type system.

# Fields
- `name::Symbol`: The bound variable name
- `value::AstNode`: The value being bound
- `body::AstNode`: The expression where the binding is in scope

# Mathematical Notation
Displayed as: `let x = M in N`.

# Examples
```julia
# Create let x = 5 in x + 1
LetNode(:x, ConstNode(:5), AppNode(AppNode(VarNode(:add), VarNode(:x)), ConstNode(:1)))
```

# Deprecation Note
**This node type is not recommended for general use.** All λx.N M forms can be
regarded as equivalent to `let x = M in N`, making explicit let-bindings redundant.
The preferred representation is:

```julia
# Instead of: let x = M in N
LetNode(:x, M, N)

# Use: (λx. N) M  
AppNode(AbsNode(:x, N), M)
```

This transformation is semantically equivalent and maintains the locally nameless
representation more consistently.
"""
mutable struct LetNode <: AstNode
    name::Symbol
    value::AstNode
    body::AstNode
end

children(node::LetNode) = [node.value, node.body]
function setchild!(node::LetNode, index::Int, newchild::AstNode)
    if index == 1
        node.value = newchild
    elseif index == 2
        node.body = newchild
    else
        _throw_children_boundserror(node, index)
    end
    return node
end
arity(::LetNode) = 2
isleaf(::LetNode) = false

########################################################
# Display System
########################################################

"""
    Base.show(io::IO, node::AstNode)

Plain show method providing informative representation showing internal structure.
"""
function Base.show(io::IO, node::AstNode)
    print(io, _show_plain(node))
end

"""
    Base.show(io::IO, ::MIME"text/plain", node::AstNode)

Pretty display method for human-readable mathematical notation.
"""
function Base.show(io::IO, ::MIME"text/plain", node::AstNode)
    print(io, _show_pretty(node))
end

"""
    Base.show(io::IO, tree::AstTree)

Plain show method for AstTree showing internal structure.
"""
function Base.show(io::IO, tree::AstTree)
    print(io, "AstTree(", _show_plain(tree.root), ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", tree::AstTree)

Pretty display method for AstTree using mathematical notation.
"""
function Base.show(io::IO, ::MIME"text/plain", tree::AstTree)
    print(io, _show_pretty(tree.root))
end

# Plain show implementations (internal structure)
_show_plain(node::AppNode) = "AppNode($(node.func), $(node.arg))"
_show_plain(node::AbsNode) = "AbsNode($(repr(node.name)), $(node.body))"
_show_plain(node::VarNode) = "VarNode($(repr(node.name)))"
_show_plain(node::ConstNode) = "ConstNode($(repr(node.name)))"
_show_plain(node::IndexNode) = "IndexNode($(node.index))"
_show_plain(node::LetNode) = "LetNode($(repr(node.name)), $(node.value), $(node.body))"

# Pretty show implementations (mathematical notation) using multiple dispatch
# Precedence levels: 0 = lowest, 3 = highest
# Lambda abstraction: precedence 1
# Application: precedence 2  
# Let: precedence 1

_show_pretty(node::AstNode, precedence::Int = 0) = _show_plain(node)  # Fallback

_show_pretty(node::VarNode, precedence::Int) = _display_var(node.name)
_show_pretty(node::ConstNode, precedence::Int) = string(node.name)
_show_pretty(node::IndexNode, precedence::Int) = "#$(node.index)"

function _show_pretty(node::AppNode, precedence::Int)
    # For function part: use precedence 2 when the function is a lambda/let to get parentheses
    func_precedence = (node.func isa AbsNode || node.func isa LetNode) ? 2 : 2
    func_str = _show_pretty(node.func, func_precedence)
    # For argument part: use precedence 3 to add parentheses when needed
    arg_str = _show_pretty(node.arg, 3)
    
    result = "$func_str $arg_str"
    return precedence > 2 ? "($result)" : result
end

function _show_pretty(node::AbsNode, precedence::Int)
    body_str = _show_pretty(node.body, 1)
    result = "λ" * _display_var(node.name) * "·" * body_str
    return precedence > 1 ? "(" * result * ")" : result
end

function _show_pretty(node::LetNode, precedence::Int)
    value_str = _show_pretty(node.value, 0)
    body_str = _show_pretty(node.body, 0)
    result = "let $(node.name) = $value_str in $body_str"
    return precedence > 1 ? "($result)" : result
end

########################################################
# Utility Functions
########################################################

# Helper function to throw consistent bounds errors for child access.
_throw_children_boundserror(node::AstNode, index::Int) = throw(BoundsError(children(node), index))

"""
    eliminate_bound_variables!(tree::AstTree)

Eliminate bound variables in the tree by replacing them with their De Bruijn indices.
"""
function eliminate_bound_variables!(tree::AstTree)
    traverse(tree, ScopeState, PointerState) do tree, node, bag
        if node isa VarNode
            abstractions = getstate(bag, ScopeState).abstractions
            i = findlast(x->x.name == node.name, abstractions)
            if !isnothing(i)
                pointer = getstate(bag, PointerState{AstNode})
                setchild!(tree, pointer.parent, pointer.child_index, IndexNode(length(abstractions) - i))
            end
        end
        return nothing
    end
end

"""
    shift_free_variables!(tree::AstTree, shift::Int, bias::Int=0)

Shift De Bruijn indices of free variables in the tree by a given amount.

# Arguments
- `tree::AstTree`: The tree to shift
- `shift::Int`: The amount to shift the indices
- `bias::Int`: If the variable is bound to up to this much "outer" than the current depth, then shift it.

"""
function shift_free_variables!(tree::AstTree, shift::Int, bias::Int=0)
    traverse(tree, AbstractionDepthState, PointerState) do tree, node, bag
        if node isa IndexNode
            depth = getstate(bag, AbstractionDepthState).depth
            if node.index >= depth + bias
                pointer = getstate(bag, PointerState{AstNode})
                new_node = IndexNode(node.index + shift)
                setchild!(tree, pointer.parent, pointer.child_index, AstTree(new_node))
            end
        end
        return nothing
    end
    return tree
end

########################################################
# Traversal States for AstTree (using new traverse interface)
########################################################

# Import traverse interface from Trees
using ..Trees: AbstractTraverseState, getstate, Leave, StateBag, build_state_bag, DepthState, StackState, AncestorsState, PointerState, traverse
import ..Trees: init, requires, enter!, leave!



"""
    ScopeState <: AbstractTraverseState

A traverse state that provides lazy access to all abstraction nodes (AbsNode) in
the current scope chain. Depends on AncestorsState for ancestor information.

# Fields
- `abstractions::Vector{AbsNode}`: Current abstraction nodes in scope
"""
mutable struct ScopeState <: AbstractTraverseState
    abstractions::Vector{AbsNode}
end

requires(::Type{ScopeState}, tree::AbstractTree) = (ScopeState, AncestorsState{NodeIndex(tree)})
init(::Type{ScopeState}, tree::AbstractTree, root_node) = ScopeState(AbsNode[])
enter!(::ScopeState, ::AbstractTree, ::AstNode) = nothing
leave!(::ScopeState, ::AbstractTree, ::AstNode) = nothing
function getstate(bag::StateBag, ::Type{ScopeState})
    scopestate = bag.lookup[ScopeState]
    ancestors = getstate(bag, AncestorsState{AstNode})
    scopestate.abstractions = AbsNode[node for node in ancestors.ancestors if node isa AbsNode]
    return scopestate
end

"""
    AbstractionDepthState <: AbstractTraverseState

A traverse state that provides the current abstraction depth (number of enclosing
lambda abstractions). Depends on ScopeState for abstraction information.

# Fields
- `depth::Int`: Current abstraction depth
"""
mutable struct AbstractionDepthState <: AbstractTraverseState
    depth::Int 
end

requires(::Type{AbstractionDepthState}, tree::AbstractTree) = (AbstractionDepthState, ScopeState)
init(::Type{AbstractionDepthState}, tree::AbstractTree, root_node) = AbstractionDepthState(0)
enter!(::AbstractionDepthState, ::AbstractTree, ::AstNode) = nothing
leave!(::AbstractionDepthState, ::AbstractTree, ::AstNode) = nothing
function getstate(bag::StateBag, ::Type{AbstractionDepthState})
    abdepthstate = bag.lookup[AbstractionDepthState]
    scope = getstate(bag, ScopeState)
    abdepthstate.depth = length(scope.abstractions)
    return abdepthstate
end

# Export the new states (remove helper functions since they're now built into getstate)
export ScopeState, AbstractionDepthState

########################################################
# Lambda Calculus Utilities
########################################################
"""
    apply_node!(tree::AstTree, func::AbsNode, arg::AstNode)

Apply a function to an argument by performing lambda substitution.

# Arguments
- `tree::AstTree`: The tree containing the function. The argument is not necessarily a part of the tree.
- `func::AbsNode`: The function to apply
- `arg::AstNode`: The argument to apply the function to. The argument is suppose to be strictly locally nameless.

# Returns
- a new AstTree representing the body of the function with all occurrences of the bound variable replaced by the argument.

Note: The result is likely to be a modified subtree of the func node
This function is likely to modify the tree, and the in-place modification commonly make no sense.
Therefore, ussually you should replace the func node with the result after calling this function.
"""
function apply_node!(tree::AstTree, func::AbsNode, arg::AstNode)
    # Get the body of the lambda function
    body_tree = AstTree(deepcopy(func.body))
    varname = func.name
    
    # Traverse the body and substitute all occurrences of the bound variable
    traverse(body_tree, AbstractionDepthState, PointerState) do tree, node, bag
        if node isa VarNode
            if node.name == varname
                # Create a copy of the argument and wrap it in a tree for shifting
                arg_tree = AstTree(deepcopy(arg))
                shift_free_variables!(arg_tree, getstate(bag, AbstractionDepthState).depth, 0)
                pointer = getstate(bag, PointerState{AstNode})
                setchild!(tree, pointer.parent, pointer.child_index, arg_tree)
            end
        end
    end
    return body_tree
end



########################################################
# Test Items – additional tests for iterators and states
########################################################

@testitem "AstTree basic iteration" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, VarNode

    var_x = VarNode(:x)
    var_y = VarNode(:y)
    app   = AppNode(var_x, var_y)
    tree  = AstTree(app)

    result = collect(tree)
    @test result == [app, var_x, var_y]
end

@testitem "Symbol generation edge cases" begin
    using LambdaRegression.ASTs.NodeASTs: _index_to_italic

    # Test edge cases for _index_to_italic (complex logic worth testing)
    @test _index_to_italic(0) == "𝑎"
    @test _index_to_italic(25) == "𝑧"
    @test _index_to_italic(26) == "𝑎𝑎"
    @test _index_to_italic(27) == "𝑎𝑏"
end

@testitem "setchild! bounds checking" begin
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, setchild!

    # Test bounds errors (important error-handling logic)
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    app = AppNode(var_x, var_y)

    @test_throws BoundsError setchild!(app, 0, var_y)
    @test_throws BoundsError setchild!(app, 3, var_y)

    abs_node = AbsNode(:f, var_x)
    @test_throws BoundsError setchild!(abs_node, 2, var_x)

    # Test leaf node setchild! (should always throw)
    @test_throws BoundsError setchild!(var_x, 1, var_y)
end

@testitem "Display precedence rules" begin
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, ConstNode, _show_pretty

    # Test complex precedence handling (important logic worth testing)
    var_x = VarNode(:x)
    const_pi = ConstNode(:pi)
    abs_node = AbsNode(:x, var_x)

    # Test complex precedence: (λx·x) pi
    complex_app = AppNode(abs_node, const_pi)
    @test _show_pretty(complex_app) == "(λx·x) pi"

    # Test associativity: f x y (left associative)
    var_f = VarNode(:f)
    var_y = VarNode(:y)
    nested_app = AppNode(AppNode(var_f, var_x), var_y)
    @test _show_pretty(nested_app) == "f x y"
end



@testitem "AncestorsState functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, AncestorsState, getstate, AstNode
    using LambdaRegression.Trees: traverse

    # Create test tree: (λx. x y) z
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    var_z = VarNode(:z)
    app_xy = AppNode(var_x, var_y)
    abs_node = AbsNode(:x, app_xy)
    root_app = AppNode(abs_node, var_z)
    tree = AstTree(root_app)

    # Collect ancestors information
    ancestors_info = Tuple{String, Vector{String}}[]  # (node_type, ancestor_types)
    
    function collect_ancestors_info(tree, node, bag)
        ancestors = getstate(bag, AncestorsState{AstNode})
        node_type = string(typeof(node).name.name)
        ancestor_types = [string(typeof(a).name.name) for a in ancestors.ancestors]
        push!(ancestors_info, (node_type, ancestor_types))
        return nothing
    end

    bag = traverse(collect_ancestors_info, tree, AncestorsState)
    
    # Verify ancestors tracking (tree has 6 nodes: root_app, abs_node, app_xy, var_x, var_y, var_z)
    @test length(ancestors_info) == 6  # 6 nodes total
    
    # Root should have itself as ancestor (due to enter! being called before user function)
    @test ancestors_info[1] == ("AppNode", ["AppNode"])
    
    # Check that ancestors are properly tracked for nested nodes
    # The exact order depends on traversal, but we can verify relationships
    let nested_var_count = 0
        for (node_type, ancestor_types) in ancestors_info
            if node_type == "VarNode" && length(ancestor_types) >= 2
                # Should have at least AppNode and AbsNode as ancestors for inner variables
                @test "AppNode" in ancestor_types
                nested_var_count += 1
            end
        end
        @test nested_var_count > 0  # At least one nested variable should be found
    end
end

@testitem "ScopeState functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, ScopeState, getstate
    using LambdaRegression.Trees: traverse

    # Create nested lambda tree: λx. λy. x y
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    inner_app = AppNode(var_x, var_y)
    inner_abs = AbsNode(:y, inner_app)
    outer_abs = AbsNode(:x, inner_abs)
    tree = AstTree(outer_abs)

    # Collect scope information
    scope_info = Tuple{String, Int}[]  # (node_type, num_abstractions_in_scope)
    
    function collect_scope_info(tree, node, bag)
        scope = getstate(bag, ScopeState)
        node_type = string(typeof(node).name.name)
        push!(scope_info, (node_type, length(scope.abstractions)))
        return nothing
    end

    bag = traverse(collect_scope_info, tree, ScopeState)
    
    # Root AbsNode should have 1 abstraction in scope (itself)  
    @test ("AbsNode", 1) in scope_info
    
    # Inner nodes should have progressively more abstractions in scope
    max_scope_depth = maximum(info[2] for info in scope_info)
    @test max_scope_depth == 2  # Should reach depth 2 for innermost nodes (both lambdas)
    
    # Verify that innermost application has both lambdas in scope
    @test ("AppNode", 2) in scope_info
end

@testitem "AbstractionDepthState functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, AbstractionDepthState, getstate
    using LambdaRegression.Trees: traverse

    # Create nested lambda tree: λx. λy. λz. x (y z)
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    var_z = VarNode(:z)
    app_yz = AppNode(var_y, var_z)
    app_final = AppNode(var_x, app_yz)
    abs_z = AbsNode(:z, app_final)
    abs_y = AbsNode(:y, abs_z)
    abs_x = AbsNode(:x, abs_y)
    tree = AstTree(abs_x)

    # Collect depth information
    depth_info = Tuple{String, Int}[]  # (node_type, abstraction_depth)
    
    function collect_depth_info(tree, node, bag)
        depth_state = getstate(bag, AbstractionDepthState)
        node_type = string(typeof(node).name.name)
        push!(depth_info, (node_type, depth_state.depth))
        return nothing
    end

    bag = traverse(collect_depth_info, tree, AbstractionDepthState)
    
    # Root should have depth 1 (includes itself)
    @test ("AbsNode", 1) in depth_info
    
    # Should reach maximum depth 3 for innermost nodes
    max_depth = maximum(info[2] for info in depth_info)
    @test max_depth == 3
    
    # Verify specific depth for innermost application
    @test ("AppNode", 3) in depth_info
    
    # Verify that there are nodes at each depth level
    depths = Set(info[2] for info in depth_info)
    @test 1 in depths  # Root level (includes itself)
    @test 2 in depths  # After first lambda
    @test 3 in depths  # After third lambda
end

@testitem "State dependency chain" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, AncestorsState, ScopeState, AbstractionDepthState, getstate, AstNode
    using LambdaRegression.Trees: traverse

    # Create simple test tree: λx. x
    var_x = VarNode(:x)
    abs_node = AbsNode(:x, var_x)
    tree = AstTree(abs_node)

    # Test that all three states work together
    collected_data = Tuple{String, Int, Int, Int}[]  # (node_type, ancestors_count, scope_count, depth)
    
    function collect_all_info(tree, node, bag)
        ancestors = getstate(bag, AncestorsState{AstNode})
        scope = getstate(bag, ScopeState)
        depth_state = getstate(bag, AbstractionDepthState)
        
        ancestors_count = length(ancestors.ancestors)
        scope_count = length(scope.abstractions)
        depth = depth_state.depth
        
        node_type = string(typeof(node).name.name)
        push!(collected_data, (node_type, ancestors_count, scope_count, depth))
        return nothing
    end

    # Test with all three states - dependencies should be resolved automatically
    bag = traverse(collect_all_info, tree, AbstractionDepthState)  # Only specify the most dependent state
    
    @test length(collected_data) == 2  # AbsNode + VarNode
    
    # Verify that all states are present in bag (due to dependency resolution)
    @test haskey(bag.lookup, AncestorsState{AstNode})
    @test haskey(bag.lookup, ScopeState) 
    @test haskey(bag.lookup, AbstractionDepthState)
    
    # Root AbsNode: 1 ancestor (itself), 1 in scope (itself), depth 1
    @test ("AbsNode", 1, 1, 1) in collected_data
    
    # VarNode: 2 ancestors (AbsNode + itself), 1 in scope (AbsNode), depth 1  
    @test ("VarNode", 2, 1, 1) in collected_data
end

@testitem "LetNode bounds checking" begin
    using LambdaRegression.ASTs.NodeASTs: LetNode, VarNode, ConstNode, setchild!

    # Test bounds errors for LetNode (important error-handling logic)
    var_x = VarNode(:x)
    const_5 = ConstNode(:five)
    let_node = LetNode(:x, const_5, var_x)
    
    @test_throws BoundsError setchild!(let_node, 0, var_x)
    @test_throws BoundsError setchild!(let_node, 3, var_x)
end

@testitem "Display complex precedence cases" begin
    using LambdaRegression.ASTs.NodeASTs: VarNode, ConstNode, LetNode, _show_pretty

    # Test precedence handling for LetNode (complex logic worth testing)
    var_x = VarNode(:x)
    const_pi = ConstNode(:pi)
    let_node = LetNode(:y, const_pi, var_x)
    
    @test _show_pretty(let_node, 2) == "(let y = pi in x)"
    
    # Test complex precedence with nested let
    nested_let = LetNode(:a, LetNode(:b, const_pi, var_x), var_x)
    @test _show_pretty(nested_let, 0) == "let a = let b = pi in x in x"
end

@testitem "Edge cases and error handling" begin
    using LambdaRegression.ASTs.NodeASTs: _index_to_italic, _display_var, _throw_children_boundserror, VarNode, ConstNode

    # Test _index_to_italic edge cases
    @test _index_to_italic(0) == "𝑎"
    @test _index_to_italic(25) == "𝑧" 
    @test _index_to_italic(26) == "𝑎𝑎"
    @test _index_to_italic(27) == "𝑎𝑏"
    @test _index_to_italic(51) == "𝑎𝑧"
    @test _index_to_italic(52) == "𝑏𝑎"
    
    # Test error case for _index_to_italic
    @test_throws ErrorException _index_to_italic(-1)
    @test_throws ErrorException _index_to_italic(-5)
    
    # Test _display_var with numeric and non-numeric symbols
    @test _display_var(Symbol("0")) == "𝑎"
    @test _display_var(Symbol("25")) == "𝑧"
    @test _display_var(Symbol("123")) == _index_to_italic(123)
    @test _display_var(:regular_name) == "regular_name"
    @test _display_var(:x) == "x"
    
    # Test _throw_children_boundserror
    var_node = VarNode(:test)
    @test_throws BoundsError _throw_children_boundserror(var_node, 1)
    @test_throws BoundsError _throw_children_boundserror(var_node, -1)
    @test_throws BoundsError _throw_children_boundserror(var_node, 100)
end

@testitem "Complex display precedence" begin
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, ConstNode, _show_pretty

    # Test complex precedence scenarios
    var_f = VarNode(:f)
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    
    # Test nested applications: f x y (left associative)
    app1 = AppNode(var_f, var_x)
    app2 = AppNode(app1, var_y)
    @test _show_pretty(app2, 0) == "f x y"
    
    # Test lambda in function position needs parentheses: (λx·x) y
    abs_node = AbsNode(:x, var_x)
    app_with_lambda = AppNode(abs_node, var_y)
    @test _show_pretty(app_with_lambda, 0) == "(λx·x) y"
    
    # Test nested lambdas: λx·λy·x y
    inner_app = AppNode(var_x, var_y)
    inner_abs = AbsNode(:y, inner_app)
    outer_abs = AbsNode(:x, inner_abs)
    @test _show_pretty(outer_abs, 0) == "λx·λy·x y"
    
    # Test lambda with high precedence context
    @test _show_pretty(outer_abs, 3) == "(λx·λy·x y)"
end

@testitem "setchild! actual modifications" begin
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, LetNode, VarNode, setchild!

    # Test that setchild! actually modifies the fields (currently 0 coverage)
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    var_z = VarNode(:z)
    
    # Test AppNode field modifications
    app = AppNode(var_x, var_y)
    @test setchild!(app, 1, var_z) === app  # Returns self
    @test app.func === var_z
    @test setchild!(app, 2, var_x) === app
    @test app.arg === var_x
    
    # Test AbsNode field modification
    abs_node = AbsNode(:f, var_x)
    @test setchild!(abs_node, 1, var_y) === abs_node
    @test abs_node.body === var_y
    
    # Test LetNode field modifications
    let_node = LetNode(:x, var_x, var_y)
    @test setchild!(let_node, 1, var_z) === let_node
    @test let_node.value === var_z
    @test setchild!(let_node, 2, var_x) === let_node
    @test let_node.body === var_x
end

@testitem "Display methods and show functions" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, LetNode

    # Test Base.show methods (currently untested)
    var_x = VarNode(:x)
    const_pi = ConstNode(:pi)
    index_0 = IndexNode(0)
    abs_node = AbsNode(:f, var_x)
    app_node = AppNode(var_x, const_pi)
    let_node = LetNode(:y, const_pi, var_x)
    tree = AstTree(abs_node)
    
    # Test plain show for nodes
    io = IOBuffer()
    show(io, var_x)
    @test String(take!(io)) == "VarNode(:x)"
    
    show(io, const_pi)
    @test String(take!(io)) == "ConstNode(:pi)"
    
    show(io, index_0)
    @test String(take!(io)) == "IndexNode(0)"
    
    show(io, abs_node)
    @test String(take!(io)) == "AbsNode(:f, VarNode(:x))"
    
    show(io, app_node)
    @test String(take!(io)) == "AppNode(VarNode(:x), ConstNode(:pi))"
    
    show(io, let_node)
    @test String(take!(io)) == "LetNode(:y, ConstNode(:pi), VarNode(:x))"
    
    # Test pretty show for nodes
    show(io, MIME"text/plain"(), var_x)
    @test String(take!(io)) == "x"
    
    show(io, MIME"text/plain"(), const_pi)
    @test String(take!(io)) == "pi"
    
    show(io, MIME"text/plain"(), index_0)
    @test String(take!(io)) == "#0"
    
    show(io, MIME"text/plain"(), abs_node)
    @test String(take!(io)) == "λf·x"
    
    # Test tree show methods
    show(io, tree)
    @test String(take!(io)) == "AstTree(AbsNode(:f, VarNode(:x)))"
    
    show(io, MIME"text/plain"(), tree)
    @test String(take!(io)) == "λf·x"
end

@testitem "eliminate_bound_variables! functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, IndexNode, eliminate_bound_variables!

    # Test simple case: λx. x should become λx. #0
    var_x = VarNode(:x)
    abs_node = AbsNode(:x, var_x)
    tree = AstTree(abs_node)
    eliminate_bound_variables!(tree)
    
    # Check that the bound variable was replaced
    @test tree.root.body isa IndexNode
    @test tree.root.body.index == 0
    
    # Test nested abstraction: λx. λy. x should become λx. λy. #1
    var_x2 = VarNode(:x)
    inner_abs = AbsNode(:y, var_x2)
    outer_abs = AbsNode(:x, inner_abs)
    tree2 = AstTree(outer_abs)
    eliminate_bound_variables!(tree2)
    
    # Check nested structure
    @test tree2.root.body isa AbsNode
    @test tree2.root.body.body isa IndexNode
    @test tree2.root.body.body.index == 1  # x is one level up
    
    # Test mixed bound and free variables: λx. f x should become λx. f #0
    var_f = VarNode(:f)
    var_x3 = VarNode(:x)
    app_fx = AppNode(var_f, var_x3)
    abs_with_free = AbsNode(:x, app_fx)
    tree3 = AstTree(abs_with_free)
    eliminate_bound_variables!(tree3)
    
    # Check that free variable remains and bound variable is converted
    result_app = tree3.root.body
    @test result_app isa AppNode
    @test result_app.func isa VarNode
    @test result_app.func.name == :f  # free variable unchanged
    @test result_app.arg isa IndexNode
    @test result_app.arg.index == 0   # bound variable converted
    
    # Test complex nested case: λx. λy. λz. x (y z)
    var_x4 = VarNode(:x)
    var_y4 = VarNode(:y)
    var_z4 = VarNode(:z)
    app_yz = AppNode(var_y4, var_z4)
    app_x_yz = AppNode(var_x4, app_yz)
    abs_z = AbsNode(:z, app_x_yz)
    abs_y = AbsNode(:y, abs_z)
    abs_x = AbsNode(:x, abs_y)
    tree4 = AstTree(abs_x)
    eliminate_bound_variables!(tree4)
    
    # Navigate to innermost application: x (y z)
    inner_abs_y = tree4.root.body
    @test inner_abs_y isa AbsNode
    inner_abs_z = inner_abs_y.body
    @test inner_abs_z isa AbsNode
    final_app = inner_abs_z.body
    @test final_app isa AppNode
    
    # Check x becomes #2 (two levels up), y becomes #1, z becomes #0
    @test final_app.func isa IndexNode
    @test final_app.func.index == 2  # x is 2 levels up
    
    inner_app = final_app.arg
    @test inner_app isa AppNode
    @test inner_app.func isa IndexNode
    @test inner_app.func.index == 1  # y is 1 level up
    @test inner_app.arg isa IndexNode  
    @test inner_app.arg.index == 0   # z is at current level
end

@testitem "IndexNode display" begin
    using LambdaRegression.ASTs.NodeASTs: IndexNode, _show_pretty

    # Test IndexNode pretty display (currently untested)
    index_0 = IndexNode(0)
    index_1 = IndexNode(1)
    index_10 = IndexNode(10)
    
    @test _show_pretty(index_0, 0) == "#0"
    @test _show_pretty(index_1, 0) == "#1"
    @test _show_pretty(index_10, 0) == "#10"
end

@testitem "Tree manipulation with setroot!" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, VarNode, setroot!, root

    # Test setroot! with node
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    tree = AstTree(var_x)
    
    result = setroot!(tree, var_y)
    @test result === tree  # Returns same tree
    @test root(tree) === var_y
    
    # Test setroot! with another tree
    app = AppNode(var_x, var_y)
    new_tree = AstTree(app)
    result = setroot!(tree, new_tree)
    @test result === tree
    @test root(tree) === app
end

@testitem "eliminate_bound_variables! edge cases" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, eliminate_bound_variables!

    # Test tree with only free variables - nothing should change
    var_f = VarNode(:f)
    var_g = VarNode(:g)
    const_pi = ConstNode(:pi)
    app_fg = AppNode(var_f, var_g)
    app_with_const = AppNode(app_fg, const_pi)
    tree_free = AstTree(app_with_const)
    
    # Store original structure for comparison
    original_func = tree_free.root.func.func.name
    original_arg1 = tree_free.root.func.arg.name  
    original_arg2 = tree_free.root.arg.name
    
    eliminate_bound_variables!(tree_free)
    
    # Nothing should have changed - all variables remain as VarNode
    @test tree_free.root isa AppNode
    @test tree_free.root.func isa AppNode
    @test tree_free.root.func.func isa VarNode
    @test tree_free.root.func.func.name == original_func
    @test tree_free.root.func.arg isa VarNode
    @test tree_free.root.func.arg.name == original_arg1
    @test tree_free.root.arg isa ConstNode
    @test tree_free.root.arg.name == original_arg2
    
    # Test shadowing: λx. λx. x (inner x should bind to inner λ)
    inner_var_x = VarNode(:x)
    inner_abs = AbsNode(:x, inner_var_x)  # λx. x
    outer_abs = AbsNode(:x, inner_abs)    # λx. λx. x
    tree_shadow = AstTree(outer_abs)
    eliminate_bound_variables!(tree_shadow)
    
    # The inner x should resolve to index 0 (innermost binding)
    @test tree_shadow.root.body isa AbsNode
    @test tree_shadow.root.body.body isa IndexNode
    @test tree_shadow.root.body.body.index == 0
    
    # Test partial binding: λx. y x (y free, x bound)
    var_y = VarNode(:y)
    var_x_partial = VarNode(:x)
    app_yx = AppNode(var_y, var_x_partial)
    abs_partial = AbsNode(:x, app_yx)
    tree_partial = AstTree(abs_partial)
    eliminate_bound_variables!(tree_partial)
    
    # y should remain free, x should be bound
    result_partial = tree_partial.root.body
    @test result_partial isa AppNode
    @test result_partial.func isa VarNode
    @test result_partial.func.name == :y  # free variable unchanged
    @test result_partial.arg isa IndexNode
    @test result_partial.arg.index == 0   # bound variable converted
    
    # Test empty tree edge case
    single_var = VarNode(:lonely)
    tree_single = AstTree(single_var)
    eliminate_bound_variables!(tree_single)
    
    # Single free variable should remain unchanged
    @test tree_single.root isa VarNode
    @test tree_single.root.name == :lonely
    
    # Test deeply nested with mix of same and different names: λa. λb. λa. a b
    var_a1 = VarNode(:a)
    var_b1 = VarNode(:b)
    app_ab = AppNode(var_a1, var_b1)
    abs_inner_a = AbsNode(:a, app_ab)    # λa. a b (redefines a)
    abs_b = AbsNode(:b, abs_inner_a)     # λb. λa. a b
    abs_outer_a = AbsNode(:a, abs_b)     # λa. λb. λa. a b
    tree_deep = AstTree(abs_outer_a)
    eliminate_bound_variables!(tree_deep)
    
    # Navigate to innermost application
    level1 = tree_deep.root.body          # λb. λa. a b
    @test level1 isa AbsNode
    level2 = level1.body                  # λa. a b
    @test level2 isa AbsNode  
    final_app_deep = level2.body          # a b
    @test final_app_deep isa AppNode
    
    # a should resolve to innermost λa (index 0), b to middle λb (index 1)
    @test final_app_deep.func isa IndexNode
    @test final_app_deep.func.index == 0  # inner a
    @test final_app_deep.arg isa IndexNode
    @test final_app_deep.arg.index == 1   # b from one level up
end

@testitem "shift_free_variables! functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, IndexNode, AbsNode, VarNode, shift_free_variables!

    # Test basic free variable shifting: #1 at depth 0 shifts to #2
    index_1 = IndexNode(1)
    tree_simple = AstTree(index_1)
    shift_free_variables!(tree_simple, 1, 0)
    @test tree_simple.root.index == 2
    
    # Test bound vs free distinction: #0 is bound at depth 1, #1 is free
    lambda_tree = AstTree(AbsNode(:x, IndexNode(0)))
    shift_free_variables!(lambda_tree, 1, 0)
    @test lambda_tree.root.body.index == 0  # Bound variable doesn't shift
    
    lambda_tree2 = AstTree(AbsNode(:x, IndexNode(1)))
    shift_free_variables!(lambda_tree2, 1, 0)
    @test lambda_tree2.root.body.index == 2  # Free variable shifts
    
    # Test VarNodes are unaffected
    var_tree = AstTree(VarNode(:x))
    shift_free_variables!(var_tree, 5, 0)
    @test var_tree.root isa VarNode && var_tree.root.name == :x
end

@testitem "apply_node! functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, IndexNode, AppNode, apply_node!

    # Test simple substitution with VarNode
    abs_identity = AbsNode(:x, VarNode(:x))
    result_tree = apply_node!(AstTree(abs_identity), abs_identity, VarNode(:y))
    @test result_tree.root isa VarNode && result_tree.root.name == :y
    
    # Test crucial case: argument with De Bruijn indices that need shifting
    # (λy. λx. y) #0 -> when we substitute #0 for y, it should become λx. #1
    # because the #0 gets shifted by the depth of the inner lambda
    inner_lambda = AbsNode(:x, VarNode(:y))  # λx. y (y will be substituted)
    outer_lambda = AbsNode(:y, inner_lambda)  # λy. λx. y
    arg_with_index = IndexNode(0)  # This will replace y
    result_shifted = apply_node!(AstTree(outer_lambda), outer_lambda, arg_with_index)
    @test result_shifted.root isa AbsNode  # Should be λx. #1
    @test result_shifted.root.body isa IndexNode
    @test result_shifted.root.body.index == 1  # #0 shifted to #1 due to inner lambda depth
    
    # Test no substitution when variable names don't match
    abs_different = AbsNode(:x, VarNode(:z))  # λx. z (z is free)
    result_no_sub = apply_node!(AstTree(abs_different), abs_different, VarNode(:y))
    @test result_no_sub.root isa VarNode && result_no_sub.root.name == :z
    
    # Test nested lambda with De Bruijn shifting: (λx. λy. x) #0
    # Should result in λy. #1 (outer #0 becomes #1 due to inner lambda)
    inner_abs = AbsNode(:y, VarNode(:x))
    outer_abs = AbsNode(:x, inner_abs)
    result_nested = apply_node!(AstTree(outer_abs), outer_abs, IndexNode(0))
    @test result_nested.root isa AbsNode
    @test result_nested.root.name == :y
    @test result_nested.root.body isa IndexNode
    @test result_nested.root.body.index == 1  # #0 shifted to #1
end



end # module NodeASTs