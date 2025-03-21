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
- [`FunctionDefNode`](@ref): Recursive function definition with one anonymous
  self binder

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

# Create a let-like binding: (λx. x) y
let_like = AppNode(AbsNode(:x, VarNode(:x)), VarNode(:y))
```

# Implementation Notes

All AST nodes implement the Trees.AbstractTree interface, enabling seamless integration
with tree traversal and manipulation algorithms. The nodes are designed to be mutable
where appropriate to support efficient tree transformations.

The symbol generation system ensures unique variable names within each tree context,
automatically resolving naming conflicts without requiring global coordination.
"""
module NodeASTs

# Import and using statements
import ..Trees: AbstractTree, NodeIndex, root, setroot!, subtree, children, setchild!, arity, isleaf, getstate
using ..Trees: AbstractTraverseState, getstate, Leave, StateBag, build_state_bag, DepthState, StackState, AncestorsState, PointerState, traverse
import ..Trees: init, requires, enter!, leave!
using MLStyle
using TestItems

# Export statements
export AstNode, AstTree
export AppNode, AbsNode, VarNode, ConstNode, IndexNode, FunctionDefNode
export copy_occurrences
export SymbolGenerator, generate_fresh_name
export ScopeState, AbstractionDepthState


########################################################
# Symbol Generation for Abstraction Naming
########################################################

"""
    SymbolGenerator

A simple and efficient symbol generator for symbolic regression.
Generates Symbols with numeric names, e.g. `Symbol("0")`, `Symbol("1")`, ...
"""
mutable struct SymbolGenerator
    counter::Int
end
SymbolGenerator() = SymbolGenerator(0)

"""
    generate_fresh_name(gen::SymbolGenerator)

Generate the next fresh variable name as a `Symbol("<n>")`
"""
function generate_fresh_name(gen::SymbolGenerator)
    name = Symbol(gen.counter)
    gen.counter += 1
    return name
end
(s::SymbolGenerator)() = generate_fresh_name(s)

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
setchild!(tree::AstTree, index::AstNode, childindex::Int, newchild::AstNode) = (setchild!(index, childindex, newchild); tree)
setchild!(tree::AstTree, index::AstNode, childindex::Int, newchild::AstTree) = (setchild!(index, childindex, newchild.root); tree)
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

# Implementation Notes
In the locally nameless representation, bound variables in the body are preferred to be represented as
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

**Important**: The `name` field is suggested to be a legal Julia variable name (e.g., `:add`, `:pi`, `:x1`).
Invalid names like `:42` or `:1` should be very careful to use. 
This constraint is documented but not enforced for performance reasons in symbolic regression contexts.
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

"""
    FunctionDefNode <: AstNode

Recursive function definition with one anonymous monomorphic self binder. The
self binder occupies De Bruijn index `#0` at the root of `body`; ordinary
lambda binders introduced below it push the self binder outward in the usual
locally nameless order.
"""
mutable struct FunctionDefNode <: AstNode
    body::AstNode
end

children(node::FunctionDefNode) = AstNode[node.body]
arity(::FunctionDefNode) = 1
isleaf(::FunctionDefNode) = false

function setchild!(node::FunctionDefNode, index::Int, newchild::AstNode)
    index == 1 || _throw_children_boundserror(node, index)
    node.body = newchild
    return node
end

function _copy_occurrence(node::AstNode, active::IdDict{AstNode,Nothing})
    node isa Union{ConstNode,VarNode,IndexNode} && return node
    haskey(active, node) && throw(ArgumentError("cyclic host AST occurrence"))
    active[node] = nothing
    try
        if node isa AppNode
            return AppNode(
                _copy_occurrence(node.func, active),
                _copy_occurrence(node.arg, active),
            )
        elseif node isa AbsNode
            return AbsNode(node.name, _copy_occurrence(node.body, active))
        elseif node isa FunctionDefNode
            return FunctionDefNode(_copy_occurrence(node.body, active))
        end
        throw(ArgumentError("unsupported AST node $(typeof(node))"))
    finally
        delete!(active, node)
    end
end

"""
    copy_occurrences(node::AstNode)
    copy_occurrences(tree::AstTree)

Copy a host AST by syntactic occurrence. Mutable expression nodes are rebuilt,
so two references to one host subtree become two independent syntax occurrences;
immutable leaves may be reused. A cycle on the active host path is rejected,
while ordinary sharing across sibling positions remains valid.
"""
copy_occurrences(node::AstNode) = _copy_occurrence(node, IdDict{AstNode,Nothing}())
copy_occurrences(tree::AstTree) = AstTree(copy_occurrences(tree.root))

########################################################
# Display System
########################################################

function Base.show(io::IO, node::AstNode)
    print(io, _show_plain(node))
end
function Base.show(io::IO, ::MIME"text/plain", node::AstNode)
    print(io, _show_pretty(node))
end
function Base.show(io::IO, tree::AstTree)
    print(io, "AstTree(", _show_plain(tree.root), ")")
end
function Base.show(io::IO, ::MIME"text/plain", tree::AstTree)
    print(io, _show_pretty(tree.root))
end

# Plain show implementations (internal structure)
_show_plain(node::AppNode) = "AppNode($(node.func), $(node.arg))"
_show_plain(node::AbsNode) = "AbsNode($(repr(node.name)), $(node.body))"
_show_plain(node::VarNode) = "VarNode($(repr(node.name)))"
_show_plain(node::ConstNode) = "ConstNode($(repr(node.name)))"
_show_plain(node::IndexNode) = "IndexNode($(node.index))"
_show_plain(node::FunctionDefNode) = "FunctionDefNode($(repr(node.body)))"

# Pretty show implementations (mathematical notation) using multiple dispatch
# Precedence levels: 0 = lowest, 3 = highest
# Lambda abstraction: precedence 1
# Application: precedence 2

_show_pretty(node::AstNode, precedence::Int = 0) = _show_plain(node)  # Fallback

_show_pretty(node::VarNode, precedence::Int) = _display_var(node.name)
_show_pretty(node::ConstNode, precedence::Int) = string(node.name)
_show_pretty(node::IndexNode, precedence::Int) = "#$(node.index)"

function _show_pretty(node::AppNode, precedence::Int)
    func_str = _show_pretty(node.func, 2)
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

const _FUNCTIONDEF_NAMES = Tuple(Symbol(char) for char in 'f':'z')
const _FUNCTIONDEF_VALUE_NAMES = (
    :x, :y, :z,
    Tuple(Symbol(char) for char in 'a':'w')...,
)

_functiondef_binder_name(names, depth::Int) = names[mod1(depth + 1, length(names))]

function _show_functiondef_pretty(
    node::AstNode,
    precedence::Int,
    function_depth::Int,
    value_depth::Int,
)
    if node isa FunctionDefNode
        name = _functiondef_binder_name(_FUNCTIONDEF_NAMES, function_depth)
        if node.body isa AbsNode
            argument = _functiondef_binder_name(_FUNCTIONDEF_VALUE_NAMES, value_depth)
            body = _show_functiondef_pretty(
                node.body.body,
                1,
                function_depth + 1,
                value_depth + 1,
            )
            result = "fn $name($argument). $body"
        else
            body = _show_functiondef_pretty(
                node.body,
                1,
                function_depth + 1,
                value_depth,
            )
            result = "fn $name. $body"
        end
        return precedence > 1 ? "($result)" : result
    elseif node isa AppNode
        func = _show_functiondef_pretty(node.func, 2, function_depth, value_depth)
        arg = _show_functiondef_pretty(node.arg, 3, function_depth, value_depth)
        result = "$func $arg"
        return precedence > 2 ? "($result)" : result
    elseif node isa AbsNode
        name = _functiondef_binder_name(_FUNCTIONDEF_VALUE_NAMES, value_depth)
        body = _show_functiondef_pretty(
            node.body,
            1,
            function_depth,
            value_depth + 1,
        )
        result = "λ$name·$body"
        return precedence > 1 ? "($result)" : result
    end
    return _show_pretty(node, precedence)
end

_show_pretty(node::FunctionDefNode, precedence::Int=0) =
    _show_functiondef_pretty(node, precedence, 0, 0)

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
            i = findlast(x -> x isa AbsNode && x.name == node.name, abstractions)
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
"""
    ScopeState <: AbstractTraverseState

A traverse state that provides lazy access to all anonymous binders in the
current scope chain. It includes `AbsNode` and `FunctionDefNode`, while named
lookup skips the anonymous FunctionDef self binder. Depends on
(`AncestorsState`)[@ref] for ancestor information.

# Fields
- `abstractions::Vector{AstNode}`: Current binder nodes in scope
"""
mutable struct ScopeState <: AbstractTraverseState
    abstractions::Vector{AstNode}
end

requires(::Type{ScopeState}, tree::AbstractTree) = (ScopeState, AncestorsState{NodeIndex(tree)})
init(::Type{ScopeState}, tree::AbstractTree, root_node) = ScopeState(AstNode[])
enter!(::ScopeState, ::AbstractTree, ::AstNode) = nothing
leave!(::ScopeState, ::AbstractTree, ::AstNode) = nothing
function getstate(bag::StateBag, ::Type{ScopeState})
    scopestate = bag.lookup[ScopeState]
    ancestors = getstate(bag, AncestorsState{AstNode})
    scopestate.abstractions = AstNode[
        node for node in ancestors.ancestors if node isa Union{AbsNode,FunctionDefNode}
    ]
    return scopestate
end
hasname(state::ScopeState, name::Symbol) = any(
    x -> x isa AbsNode && x.name == name,
    state.abstractions,
)

"""
    AbstractionDepthState <: AbstractTraverseState

A traverse state that provides the current abstraction depth (number of enclosing
lambda abstractions). Depends on (`ScopeState`)[@ref] for abstraction information.

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

########################################################
# Lambda Calculus Utilities
########################################################
"""
    apply_node!(tree::AstTree, func::AbsNode, arg::AstNode)

Apply a function to an argument by performing capture-avoiding lambda
substitution. References to outer binders are shifted down when the function's
binder is removed.

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
    
    # Traverse the body, substitute the removed binder, and rebase references
    # that crossed it toward outer binders.
    traverse(body_tree, AbstractionDepthState, PointerState) do tree, node, bag
        depth_state = getstate(bag, AbstractionDepthState)
        if ((node isa VarNode) && (node.name == varname)) ||
           ((node isa IndexNode) && (node.index == depth_state.depth))
            arg_tree = AstTree(deepcopy(arg))
            shift_free_variables!(arg_tree, depth_state.depth, 0)

            pointer = getstate(bag, PointerState{AstNode})
            setchild!(tree, pointer.parent, pointer.child_index, arg_tree)
        elseif (node isa IndexNode) && (node.index > depth_state.depth)
            pointer = getstate(bag, PointerState{AstNode})
            rebased = AstTree(IndexNode(node.index - 1))
            setchild!(tree, pointer.parent, pointer.child_index, rebased)
        end
    end
    return body_tree
end

########################################################
# Hash and Equality Functions
########################################################

"""
    Base.hash(node::AstNode, h::UInt)

Compute a hash value for an AST node based on its type and field values.
The hash is computed recursively for nodes with children, ensuring that
structurally equivalent trees have the same hash value.

Note that AbsNode is not hashed by its name, but by its body.
All AbsNodes are considered anonymous for the purpose of alpha-equivalence.
The bound varibales *MUST* be transformed to De Bruijn indices before hashing,
otherwise the two unequavalent Terms will have the same hash value.
"""
Base.hash(node::AppNode, h::UInt) = hash(node.func, hash(node.arg, hash(:AppNode, h)))
Base.hash(node::AbsNode, h::UInt) = hash(node.body, hash(:AbsNode, h)) 
Base.hash(node::VarNode, h::UInt) = hash(node.name, hash(:VarNode, h))
Base.hash(node::ConstNode, h::UInt) = hash(node.name, hash(:ConstNode, h))
Base.hash(node::IndexNode, h::UInt) = hash(node.index, hash(:IndexNode, h))
Base.hash(node::FunctionDefNode, h::UInt) = hash(node.body, hash(:FunctionDefNode, h))

"""
    Base.:(==)(a::AstNode, b::AstNode)

Check if two AST nodes are structurally equal by comparing their types,
field values, and recursively comparing child nodes.

Note that AbsNode is equavalent to another AbsNode with the same body,
even if they have different names.
Therefore, the bound variables *MUST* be transformed to De Bruijn indices before comparing.
"""
Base.:(==)(a::AppNode, b::AppNode) = a.func == b.func && a.arg == b.arg
Base.:(==)(a::AbsNode, b::AbsNode) = a.body == b.body
Base.:(==)(a::VarNode, b::VarNode) = a.name == b.name
Base.:(==)(a::ConstNode, b::ConstNode) = a.name == b.name
Base.:(==)(a::IndexNode, b::IndexNode) = a.index == b.index
Base.:(==)(a::FunctionDefNode, b::FunctionDefNode) = a.body == b.body
Base.:(==)(a::AstNode, b::AstNode) = false # Different types are never equal

Base.hash(t::AstTree, h::UInt) = hash(t.root, hash(:AstTree, h))
Base.:(==)(a::AstTree, b::AstTree) = a.root == b.root

########################################################
# Test Items
########################################################

@testsnippet NodeASTsBaseImports begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, VarNode
    using LambdaRegression.ASTs.NodeASTs: _index_to_italic
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, setchild!
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, ConstNode, _show_pretty
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, AncestorsState, ScopeState, AbstractionDepthState, getstate, AstNode
    using LambdaRegression.Trees: traverse
    using LambdaRegression.ASTs.NodeASTs: _index_to_italic, _display_var, _throw_children_boundserror, VarNode, ConstNode
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, VarNode, setchild!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, IndexNode, eliminate_bound_variables!
    using LambdaRegression.ASTs.NodeASTs: IndexNode, _show_pretty
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, VarNode, setroot!, root
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, eliminate_bound_variables!
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.NodeASTs: AstTree, IndexNode, AbsNode, VarNode, shift_free_variables!
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, IndexNode, AppNode, apply_node!
end

@testitem "AstTree basic iteration" setup=[NodeASTsBaseImports] begin
    var_x = VarNode(:x)
    var_y = VarNode(:y)
    app   = AppNode(var_x, var_y)
    tree  = AstTree(app)

    result = collect(tree)
    @test result == [app, var_x, var_y]
end

@testitem "Symbol generation edge cases" setup=[NodeASTsBaseImports] begin
    # Test edge cases for _index_to_italic (complex logic worth testing)
    @test _index_to_italic(0) == "𝑎"
    @test _index_to_italic(25) == "𝑧"
    @test _index_to_italic(26) == "𝑎𝑎"
    @test _index_to_italic(27) == "𝑎𝑏"
end

@testitem "setchild! bounds checking" setup=[NodeASTsBaseImports] begin
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

@testitem "Display precedence rules" setup=[NodeASTsBaseImports] begin
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

@testitem "Traversal state system functionality" setup=[NodeASTsBaseImports] begin
    # === Test 1: Basic state dependency resolution ===
    # Create simple test tree: λx. x
    var_x1 = VarNode(:x)
    abs_simple = AbsNode(:x, var_x1)
    tree_simple = AstTree(abs_simple)

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
    _, bag = traverse(collect_all_info, tree_simple, AbstractionDepthState)  # Only specify the most dependent state
    
    @test length(collected_data) == 2  # AbsNode + VarNode
    
    # Verify that all states are present in bag (due to dependency resolution)
    @test haskey(bag.lookup, AncestorsState{AstNode})
    @test haskey(bag.lookup, ScopeState) 
    @test haskey(bag.lookup, AbstractionDepthState)
    
    # Root AbsNode: 1 ancestor (itself), 1 in scope (itself), depth 1
    @test ("AbsNode", 1, 1, 1) in collected_data
    
    # VarNode: 2 ancestors (AbsNode + itself), 1 in scope (AbsNode), depth 1  
    @test ("VarNode", 2, 1, 1) in collected_data

    # === Test 2: ScopeState with nested lambdas ===
    # Create nested lambda tree: λx. λy. x y
    var_x2 = VarNode(:x)
    var_y2 = VarNode(:y)
    inner_app = AppNode(var_x2, var_y2)
    inner_abs = AbsNode(:y, inner_app)
    outer_abs = AbsNode(:x, inner_abs)
    tree_nested = AstTree(outer_abs)

    # Collect scope information
    scope_info = Tuple{String, Int}[]  # (node_type, num_abstractions_in_scope)
    
    function collect_scope_info(tree, node, bag)
        scope = getstate(bag, ScopeState)
        node_type = string(typeof(node).name.name)
        push!(scope_info, (node_type, length(scope.abstractions)))
        return nothing
    end

    _, bag = traverse(collect_scope_info, tree_nested, ScopeState)
    
    # Root AbsNode should have 1 abstraction in scope (itself)  
    @test ("AbsNode", 1) in scope_info
    
    # Inner nodes should have progressively more abstractions in scope
    max_scope_depth = maximum(info[2] for info in scope_info)
    @test max_scope_depth == 2  # Should reach depth 2 for innermost nodes (both lambdas)
    
    # Verify that innermost application has both lambdas in scope
    @test ("AppNode", 2) in scope_info

    # === Test 3: AbstractionDepthState with deep nesting ===
    # Create deeply nested lambda tree: λx. λy. λz. x (y z)
    var_x3 = VarNode(:x)
    var_y3 = VarNode(:y)
    var_z3 = VarNode(:z)
    app_yz = AppNode(var_y3, var_z3)
    app_final = AppNode(var_x3, app_yz)
    abs_z = AbsNode(:z, app_final)
    abs_y = AbsNode(:y, abs_z)
    abs_x = AbsNode(:x, abs_y)
    tree_deep = AstTree(abs_x)

    # Collect depth information
    depth_info = Tuple{String, Int}[]  # (node_type, abstraction_depth)
    
    function collect_depth_info(tree, node, bag)
        depth_state = getstate(bag, AbstractionDepthState)
        node_type = string(typeof(node).name.name)
        push!(depth_info, (node_type, depth_state.depth))
        return nothing
    end

    _, bag = traverse(collect_depth_info, tree_deep, AbstractionDepthState)
    
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

    # === Test 4: AncestorsState with mixed structure ===
    # Create test tree: (λx. x y) z
    var_x4 = VarNode(:x)
    var_y4 = VarNode(:y)
    var_z4 = VarNode(:z)
    app_xy = AppNode(var_x4, var_y4)
    abs_node = AbsNode(:x, app_xy)
    root_app = AppNode(abs_node, var_z4)
    tree_mixed = AstTree(root_app)

    # Collect ancestors information
    ancestors_info = Tuple{String, Vector{String}}[]  # (node_type, ancestor_types)
    
    function collect_ancestors_info(tree, node, bag)
        ancestors = getstate(bag, AncestorsState{AstNode})
        node_type = string(typeof(node).name.name)
        ancestor_types = [string(typeof(a).name.name) for a in ancestors.ancestors]
        push!(ancestors_info, (node_type, ancestor_types))
        return nothing
    end

    _, bag = traverse(collect_ancestors_info, tree_mixed, AncestorsState)
    
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

@testitem "Edge cases and error handling" setup=[NodeASTsBaseImports] begin
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

@testitem "Complex display precedence" setup=[NodeASTsBaseImports] begin
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

@testitem "setchild! actual modifications" setup=[NodeASTsBaseImports] begin
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
end

@testitem "FunctionDefNode core ownership preserves anonymous binder slots" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstNode, AstTree, AbsNode, ConstNode, FunctionDefNode, IndexNode,
        VarNode, apply_node!, arity, children, eliminate_bound_variables!, isleaf,
        setchild!, shift_free_variables!

    node = FunctionDefNode(ConstNode(:body))
    @test children(node) == AstNode[ConstNode(:body)]
    @test arity(node) == 1
    @test !isleaf(node)
    setchild!(node, 1, IndexNode(0))
    @test node == FunctionDefNode(IndexNode(0))
    @test hash(node) == hash(FunctionDefNode(IndexNode(0)))
    @test_throws BoundsError setchild!(node, 2, ConstNode(:bad))

    # A FunctionDef self slot is a real De Bruijn depth, but has no named
    # binder. Named lookup still finds the lambda parameter below it.
    named = AstTree(FunctionDefNode(AbsNode(:x, VarNode(:x))))
    eliminate_bound_variables!(named)
    @test named.root.body.body == IndexNode(0)
    unbound_name = AstTree(FunctionDefNode(VarNode(:x)))
    eliminate_bound_variables!(unbound_name)
    @test unbound_name.root.body == VarNode(:x)

    # The self slot must count when shifting free indices and applying an
    # outer lambda across a recursive definition.
    shifted = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(1))))
    shift_free_variables!(shifted, 1)
    @test shifted.root.body.body == IndexNode(1)
    func = AbsNode(:x, FunctionDefNode(IndexNode(1)))
    applied = apply_node!(AstTree(func), func, ConstNode(:arg))
    @test applied.root == FunctionDefNode(ConstNode(:arg))
end

@testitem "Display methods and show functions" setup=[NodeASTsBaseImports] begin
    # Test Base.show methods (currently untested)
    var_x = VarNode(:x)
    const_pi = ConstNode(:pi)
    index_0 = IndexNode(0)
    abs_node = AbsNode(:f, var_x)
    app_node = AppNode(var_x, const_pi)
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

@testitem "eliminate_bound_variables! functionality" setup=[NodeASTsBaseImports] begin
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

@testitem "IndexNode display" setup=[NodeASTsBaseImports] begin
    # Test IndexNode pretty display (currently untested)
    index_0 = IndexNode(0)
    index_1 = IndexNode(1)
    index_10 = IndexNode(10)
    
    @test _show_pretty(index_0, 0) == "#0"
    @test _show_pretty(index_1, 0) == "#1"
    @test _show_pretty(index_10, 0) == "#10"
end

@testitem "Tree manipulation with setroot!" setup=[NodeASTsBaseImports] begin
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

@testitem "eliminate_bound_variables! edge cases" setup=[NodeASTsBaseImports] begin
    tree_free = parse_lambda("f g pie")
    new_tree = deepcopy(tree_free)
    eliminate_bound_variables!(new_tree)
    @test new_tree == tree_free
    
    # Test shadowing: λx. λx. x (inner x should bind to inner λ)
    tree_shadow = parse_lambda("λx. λx. x")
    eliminate_bound_variables!(tree_shadow)
    # The inner x should resolve to index 0 (innermost binding)
    @test tree_shadow == AstTree(AbsNode(:x, AbsNode(:x, IndexNode(0))))
    
    # Test partial binding: λx. y x (y free, x bound)
    tree_partial = parse_lambda("λx. y x")
    eliminate_bound_variables!(tree_partial)
    @test tree_partial == AstTree(AbsNode(:x, AppNode(VarNode(:y), IndexNode(0))))
    
    # Test deeply nested with mix of same and different names: λa. λb. λa. a b
    tree_deep = parse_lambda("λa. λb. λa. a b")
    eliminate_bound_variables!(tree_deep)
    @test tree_deep == AstTree(AbsNode(:a, AbsNode(:b, AbsNode(:a, AppNode(IndexNode(0), IndexNode(1))))))
end

@testitem "shift_free_variables! functionality" setup=[NodeASTsBaseImports] begin
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

@testitem "apply_node! functionality" setup=[NodeASTsBaseImports] begin
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

@testitem "apply_node! with IndexNode substitution" setup=[NodeASTsBaseImports] begin
    # Identity function represented with IndexNode
    abs_identity_idx = AbsNode(:x, IndexNode(0))  # λx. #0
    result_identity = apply_node!(AstTree(abs_identity_idx), abs_identity_idx, VarNode(:y))
    @test result_identity.root isa VarNode && result_identity.root.name == :y

    # Nested lambda: λx. λy. x  (x is IndexNode(1) inside the inner lambda)
    inner_body = IndexNode(1)
    inner_abs = AbsNode(:y, inner_body)          # λy. #1
    outer_abs = AbsNode(:x, inner_abs)           # λx. λy. #1
    result_nested = apply_node!(AstTree(outer_abs), outer_abs, VarNode(:z))
    # Expect λy. z
    @test result_nested.root isa AbsNode
    @test result_nested.root.body isa VarNode && result_nested.root.body.name == :z

    # De Bruijn shifting with IndexNode argument: (λx. λy. x) #0 -> λy. #1
    inner_abs_shift = AbsNode(:y, IndexNode(1))  # λy. #1 (refers to x)
    outer_abs_shift = AbsNode(:x, inner_abs_shift)
    result_shift = apply_node!(AstTree(outer_abs_shift), outer_abs_shift, IndexNode(0))
    @test result_shift.root isa AbsNode
    @test result_shift.root.body isa IndexNode && result_shift.root.body.index == 1

    # Removing x also shifts references that cross x toward an outer binder.
    # (λx. λy. #2) value -> λy. #1
    outer_capture = AbsNode(:x, AbsNode(:y, IndexNode(2)))
    result_outer_capture = apply_node!(
        AstTree(outer_capture),
        outer_capture,
        ConstNode(:value),
    )
    @test result_outer_capture == AstTree(AbsNode(:y, IndexNode(1)))
end

end # module NodeASTs
