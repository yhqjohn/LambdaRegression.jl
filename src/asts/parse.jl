"""
    Parse

A submodule of NodeASTs providing **lambda calculus literal parsing** functionality.
This module implements a complete parser for locally-nameless lambda calculus expressions
with sophisticated constant collection and compositional programming capabilities.

## Key Features

- **Command Literal Interface**: Main entry via [`@lambda_cmd`](@ref) / `lambda`...`` syntax  
- **Multi-Level Constant Resolution**: Flag context → Global context → Module environment
- **Smart Tree Composition**: `AstTree` constants inserted as subtrees for true compositionality
- **Automatic Dictionary Management**: "从哪来到哪去" (from where it comes, to where it goes) principle
- **Rich Lambda Syntax**: Abstractions, applications, De Bruijn indices, mixed expressions

## Quick Start

```julia
using LambdaRegression.ASTs.NodeASTs.Parse

# Basic lambda parsing
tree, consts = lambda`λx.x`                    # Identity function
tree, consts = lambda`λf.λx.f (f x)`          # Church numeral 2  
tree, consts = lambda`(λx.x) y #0`            # Application with De Bruijn

# With constants
const PI = 3.14159
tree, consts = lambda`λr. PI * r`             # Module constant  
# PI automatically added to lambda_context

# Safe flag context (recommended)
my_context = Dict(:x => 42, :y => "hello")
tree, consts = lambda`λf. x y`my_context      # Isolated context

# Compositional programming  
identity, _ = lambda`λx.x`
lambda_context[:id] = identity
tree, consts = lambda`λf. id (f x)`           # Tree inserted as subtree
```

## Main Exports

- [`@lambda_cmd`](@ref) - Primary command literal interface
- [`lambda_context`](@ref) - Global constant dictionary
- [`collect_constants!`](@ref) - Core constant collection engine  
- [`parse_lambda_literal`](@ref) - Low-level parsing function

## Design Philosophy

This module emphasizes **safety** and **compositionality**:
- Prefer explicit flag contexts over global state
- Enable true lambda calculus composition via tree insertion
- Provide clear priority ordering for constant resolution
- Support both rapid prototyping (global context) and production safety (flag contexts)
"""
module Parse

using ..NodeASTs: AstNode, AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, setchild!
using ...Trees: PointerState, traverse, getstate
using TestItems

# Export main functions, macro, and context
export parse_lambda_literal, collect_constants!, @lambda_cmd, lambda_context

########################################################
# Global Lambda Context
########################################################

"""
    lambda_context

Global dictionary for storing lambda expression constants. This dictionary is 
automatically updated when new constants are discovered in lambda expressions
(unless a flag context is provided). It follows the "从哪来到哪去" (from where 
it comes, to where it goes) principle.

**Priority order for constant resolution:**
1. User-provided flag context (if specified in [`@lambda_cmd`](@ref))
2. `lambda_context` (global)
3. Module environment (calling module)

**Automatic Updates:**
- When parsing `lambda`expr`` (without flag context): New constants → `lambda_context`
- When parsing `lambda`expr`flag_context` (with flag): New constants → `flag_context`

**Examples:**
```julia
# Basic usage
lambda_context[:myvar] = 42
tree, consts = lambda`λx. myvar`  # myvar resolved to 42

# Cumulative updates
const pi_val = 3.14
tree, consts = lambda`λr. pi_val`  # pi_val added to lambda_context automatically

# Check contents

# Clear when needed
empty!(lambda_context)
```

**Best Practices:**
While convenient, global context should be used sparingly. For better control 
and safety, prefer passing explicit context dictionaries via flags:
```julia
my_context = Dict(:x => 10, :y => 20)
tree, consts = lambda`λf. x y`my_context  # Isolated context
```
"""
const lambda_context = Dict{Symbol,Any}()

########################################################
# Parsing of Lambda Literals to AstTree
########################################################

"""
    parse_lambda_literal(str::AbstractString) -> AstTree

Parse a locally-nameless lambda calculus literal contained in `str` and return
an `AstTree` constructed from the [`AstNode`](@ref)s defined in this module.

The accepted grammar (informal) is

    term        ::= application | abstraction
    abstraction ::= "λ" identifier ["."] term
    application ::= atom (atom)*          # left-associative
    atom        ::= "(" term ")" | identifier | index | abstraction
    identifier  ::= letter (letter | digit | "_")*
    index       ::= "#" <non-negative integer>

Application is left-associative, i.e. `f x y` parses as `(f x) y`.

Examples
```julia
parse_lambda_literal("λx. x")                       # λx·x
parse_lambda_literal("(λx. x) y")                   # (λx·x) y
parse_lambda_literal("λx. λy. x (#0 y)")            # λx·λy·x (#0 y)
```
"""
function parse_lambda_literal(str::AbstractString)::AstTree
    tokens = _tokenize_lambda(str)
    pos = Ref(1)
    expr = _parse_term(tokens, pos)
    if pos[] <= length(tokens)
        error("unexpected token $(tokens[pos[]]) at position $(pos[])")
    end
    return AstTree(expr)
end

#########################
# Internal Implementation
#########################

# ------------------------------------------------------------------
# Token definition
# ------------------------------------------------------------------
struct _Tok
    kind::Symbol      # :lambda, :dot, :lparen, :rparen, :ident, :index
    val::Any
end

# ------------------------------------------------------------------
# Tokeniser
# ------------------------------------------------------------------

# Character delimiter predicate: whitespace or one of the special syntax chars
const _DELIMS = Set(['λ', '.', '(', ')', '#'])

function _is_delim(c::Char)
    isspace(c) && return true
    return c in _DELIMS
end

function _tokenize_lambda(str::AbstractString)
    toks = _Tok[]
    i = firstindex(str)
    last = lastindex(str)
    while i <= last
        c = str[i]
        if isspace(c)
            i = nextind(str, i)
            continue
        elseif c == 'λ'
            push!(toks, _Tok(:lambda, nothing))
            i = nextind(str, i)
        elseif c == '.'
            push!(toks, _Tok(:dot, nothing))
            i = nextind(str, i)
        elseif c == '('
            push!(toks, _Tok(:lparen, nothing))
            i = nextind(str, i)
        elseif c == ')'
            push!(toks, _Tok(:rparen, nothing))
            i = nextind(str, i)
        elseif c == '#'
            # De Bruijn index
            j = nextind(str, i)
            j > last && error("`#` must be followed by an integer index")
            start = j
            while j <= last && isdigit(str[j])
                j = nextind(str, j)
            end
            idx_str = str[start:prevind(str, j)]
            isempty(idx_str) && error("`#` must be followed by an integer index")
            push!(toks, _Tok(:index, parse(Int, idx_str)))
            i = j
        else
            # identifier: read until next delimiter or whitespace
            start = i
            j = nextind(str, i)
            while j <= last && !_is_delim(str[j])
                j = nextind(str, j)
            end
            ident = Symbol(str[start:prevind(str, j)])
            push!(toks, _Tok(:ident, ident))
            i = j
        end
    end
    return toks
end

# ------------------------------------------------------------------
# Recursive-descent parser
# ------------------------------------------------------------------

# Peek at current token kind (or :eof)
_peekkind(toks, pos) = pos[] > length(toks) ? :eof : toks[pos[]].kind

# Consume expected token and return it
function _expect!(toks, pos, kind_sym)
    pos[] > length(toks) && error("unexpected end of input, expected $(kind_sym)")
    tok = toks[pos[]]
    tok.kind == kind_sym || error("expected token $(kind_sym), got $(tok.kind)")
    pos[] += 1
    return tok
end

# term ::= abstraction | application
function _parse_term(toks, pos)::AstNode
    if _peekkind(toks, pos) == :lambda
        return _parse_lambda(toks, pos)
    else
        return _parse_application(toks, pos)
    end
end

# abstraction ::= "λ" ident ["."] term
function _parse_lambda(toks, pos)::AstNode
    _expect!(toks, pos, :lambda)
    name_tok = _expect!(toks, pos, :ident)
    if _peekkind(toks, pos) == :dot
        _expect!(toks, pos, :dot)
    end
    body = _parse_term(toks, pos)
    return AbsNode(name_tok.val, body)
end

# application ::= atom (atom)*         (left-associative)
function _parse_application(toks, pos)::AstNode
    node = _parse_atom(toks, pos)
    while true
        kind = _peekkind(toks, pos)
        if kind in (:ident, :index, :lparen, :lambda)
            arg = _parse_atom(toks, pos)
            node = AppNode(node, arg)
        else
            break
        end
    end
    return node
end

# atom ::= "(" term ")" | ident | index | abstraction
function _parse_atom(toks, pos)::AstNode
    kind = _peekkind(toks, pos)
    if kind == :ident
        tok = _expect!(toks, pos, :ident)
        return VarNode(tok.val)
    elseif kind == :index
        tok = _expect!(toks, pos, :index)
        return IndexNode(tok.val)
    elseif kind == :lparen
        _expect!(toks, pos, :lparen)
        expr = _parse_term(toks, pos)
        _expect!(toks, pos, :rparen)
        return expr
    elseif kind == :lambda
        return _parse_lambda(toks, pos)
    else
        error("unexpected token kind $(kind) while parsing atom")
    end
end

########################################################
# Constant collection helper
########################################################

"""
    collect_constants!(tree[, mod=Main, flag_context=nothing]) -> Dict{Symbol,Any}

**Core constant collection engine** used by [`@lambda_cmd`](@ref). Traverses an 
[`AstTree`](@ref) to find and resolve `VarNode`s that represent constants, then 
updates the tree in-place and manages dictionary updates.

## Functionality

### 1. **Multi-Source Constant Resolution**
Uses `Meta.parse` and module introspection to resolve variable names from:

**Priority order:**
1. **Flag context** (if specified via `flag_context` parameter)
2. **[`lambda_context`](@ref)** (global dictionary)
3. **Module environment** (`mod` parameter, defaults to Main)

### 2. **Smart Tree Mutation**
- **Regular constants** → Replace `VarNode` with `AstTree(ConstNode(name))`
- **`AstTree`/`AstNode` constants** → Replace `VarNode` with the tree/node directly (subtree insertion, no context update)
- **Unresolved variables** → Leave as `VarNode` (no error)

### 3. **Dictionary Update Management ("从哪来到哪去")**
Follows the "from where it comes, to where it goes" principle:
- **Without flag context**: New regular constants → [`lambda_context`](@ref)  
- **With flag context**: New regular constants → user's `flag_context` dictionary
- **`AstTree`/`AstNode` constants**: No dictionary updates (subtrees are used as-is)

## Parameters

- **`tree`**: [`AstTree`](@ref) to process (modified in-place)
- **`mod`**: Module for constant lookup (default: `Main`)  
- **`flag_context`**: Optional user dictionary with highest priority

## Return Value

Dictionary containing **only the constants actually found** in this tree traversal.
Does not include pre-existing context entries unless they were referenced.

## Examples

### Basic Usage
```julia
julia> tree = parse_lambda_literal("λx. PI 42")
julia> const PI = 3.14159

julia> found = collect_constants!(tree, Main)
# Returns: Dict(:PI => 3.14159, Symbol("42") => 42)
# lambda_context now contains: Dict(:PI => 3.14159, Symbol("42") => 42)
```

### Flag Context Usage  
```julia
julia> tree = parse_lambda_literal("λf. x y 100")
julia> my_context = Dict(:x => "hello", :z => "unused")

julia> found = collect_constants!(tree, Main, my_context)
# Returns: Dict(:x => "hello", Symbol("100") => 100)
# my_context now: Dict(:x => "hello", :z => "unused", Symbol("100") => 100)
# Note: :z not in returned dict (wasn't referenced in tree)
```

### Subtree Insertion
```julia
# AstTree subtree insertion
julia> sub_tree, _ = lambda`λa.a`
julia> main_tree = parse_lambda_literal("λx. sub x")  
julia> lambda_context[:sub] = sub_tree

julia> found = collect_constants!(main_tree, Main)
# The 'sub' VarNode is replaced directly with sub_tree (no ConstNode wrapper)
# lambda_context is NOT updated since sub_tree was already there
# This enables true compositional lambda calculus!

# AstNode subtree insertion
julia> node = VarNode(:y)
julia> tree = parse_lambda_literal("λx. n")
julia> lambda_context[:n] = node

julia> found = collect_constants!(tree, Main)  
# The 'n' VarNode is replaced directly with node (wrapped in AstTree)
# lambda_context is NOT updated since node was already there
```

### Priority Resolution
```julia
julia> const shared_var = "module_value"
julia> lambda_context[:shared_var] = "context_value"  
julia> user_flags = Dict(:shared_var => "flag_value")

julia> tree = parse_lambda_literal("λx. shared_var")
julia> collect_constants!(tree, Main, user_flags)
# Result: Dict(:shared_var => "flag_value")  # Flag wins!
```

## Important Limitations

- **Module scope only**: Can only collect module-level constants, not local variables from functions/let blocks
- **No nested scopes**: Cannot access variables from outer function scopes  
- **No symbol literals**: `:foo` syntax not supported (use regular variables)
- **Parse-able names**: Variable names must be valid Julia identifiers or parseable literals

For local variables, explicitly define them in [`lambda_context`](@ref) or flag dictionaries.

## Performance Notes

- **In-place mutation**: Modifies the input tree directly
- **Single traversal**: All processing done in one tree walk
- **Lazy evaluation**: Only processes referenced variables
- **Memory sharing**: `AstTree` constants shared by reference

## See Also

- [`@lambda_cmd`](@ref) - Main entry point that uses this function
- [`lambda_context`](@ref) - Global constant dictionary  
- [`parse_lambda_literal`](@ref) - Creates trees for processing
"""
function collect_constants!(tree::AstTree, mod::Module=Main, flag_context::Union{Dict{Symbol,Any},Nothing}=nothing)
    # Determine target dictionary for updates ("从哪来到哪去")
    target_dict = flag_context !== nothing ? flag_context : lambda_context
    # Track constants found in this specific call
    current_constants = Dict{Symbol,Any}()
 
    traverse(tree, PointerState) do t, node, bag
        if node isa VarNode
            name = node.name
            strname = String(name)
            constant_val = nothing
            should_update_dict = false
            
            # Priority 1: Flag context
            if flag_context !== nothing && haskey(flag_context, name)
                constant_val = flag_context[name]
            # Priority 2: Global lambda_context
            elseif haskey(lambda_context, name)
                constant_val = lambda_context[name]
            else
                # Priority 3: Try parsing as literal or module lookup
                try
                    parsed = Meta.parse(strname)
                    if parsed isa Expr
                        # Complex expression - skip
                        return nothing
                    elseif parsed isa Symbol
                        # Variable name - check if defined in module
                        if isdefined(mod, name)
                            constant_val = getfield(mod, name)
                            should_update_dict = true  # New constant found, update target dict
                        end
                    else
                        # Literal constant
                        constant_val = parsed
                        should_update_dict = true  # New constant found, update target dict
                    end
                catch
                    # Parse failed - check if variable exists in module
                    if isdefined(mod, name)
                        constant_val = getfield(mod, name)
                        should_update_dict = true  # New constant found, update target dict
                    end
                end
            end
            
            if constant_val !== nothing
                # Always track what was found in this call
                current_constants[name] = constant_val
                
                pointer = getstate(bag, PointerState{AstNode})
                
                # If constant is an AstTree or AstNode, insert as subtree (no context update)
                if constant_val isa AstTree
                    setchild!(tree, pointer.parent, pointer.child_index, constant_val)
                elseif constant_val isa AstNode
                    setchild!(tree, pointer.parent, pointer.child_index, AstTree(constant_val))
                else
                    # Regular constant: update target dictionary and create ConstNode
                    if should_update_dict
                        target_dict[name] = constant_val
                    end
                    setchild!(tree, pointer.parent, pointer.child_index, AstTree(ConstNode(name)))
                end
            end
        end
        return nothing
    end
 
    return current_constants
end

########################################################
# Command literal macro `lambda_cmd`
########################################################

"""
    @lambda_cmd / lambda`...` 

    lambda`literal`
    lambda`literal`flag_context

**Main Entry Point** for parsing locally-nameless lambda calculus expressions.
This command literal macro evaluates immediately (like `@show`) and returns a tuple
`(tree, consts)` where `tree` is an [`AstTree`](@ref) and `consts` is the dictionary
of constants found and resolved by [`collect_constants!`](@ref).

## Syntax

- **`lambda`expression``** - Use global [`lambda_context`](@ref) and module environment
- **`lambda`expression`flag_context`** - Use user-provided dictionary with highest priority

## Features

### 1. **Multi-Level Constant Resolution**
**Priority order:**
1. User-provided flag context (if specified)
2. [`lambda_context`](@ref) (global)  
3. Module environment (calling module)

### 2. **Automatic Dictionary Updates ("从哪来到哪去")**
- **Without flag**: New constants → [`lambda_context`](@ref)
- **With flag**: New constants → user's `flag_context` dictionary

### 3. **Smart Tree Insertion**
- Regular constants → `ConstNode`
- `AstTree` constants → Direct subtree insertion (no wrapper)

### 4. **Rich Expression Support**
- **Abstractions**: `λx.body`, `λx λy.body` (nested)
- **Applications**: `f x y` (left-associative)  
- **De Bruijn indices**: `#0`, `#1`, `#2`...
- **Mixed syntax**: `λx.f x #0 y`

## Examples

### Basic Parsing
```julia
# Pure lambda expressions (no constants)
julia> tree, consts = lambda`λx.x`
(AstTree(AbsNode(:x, VarNode(:x))), Dict{Symbol,Any}())

julia> tree, consts = lambda`λf.λx.f (f x)`  # Church numeral 2
# Complex nested structure...

julia> tree, consts = lambda`f x y #0`       # Mixed variables and indices
# Left-associative application with De Bruijn index
```

### Constant Collection & Resolution
```julia
# Module constants
julia> const PI = 3.14159
julia> const E = 2.71828

julia> tree, consts = lambda`λr. PI * r * r`
# consts == Dict(:PI => 3.14159)
# PI automatically added to lambda_context

# Global context usage  
julia> lambda_context[:gravity] = 9.81
julia> tree, consts = lambda`λt. gravity * t`
# Uses existing lambda_context value

# Literal constants
julia> tree, consts = lambda`λx. 42 "hello"`  
# consts == Dict(Symbol("42") => 42, Symbol("\"hello\"") => "hello")
```

### Flag Context (Recommended for Safety)
```julia
# Isolated context - no global pollution
julia> my_flags = Dict(:x => 100, :y => 200)
julia> tree, consts = lambda`λf. x y f`my_flags
# consts == Dict(:x => 100, :y => 200)
# lambda_context remains unchanged

# Cumulative flag updates
julia> physics = Dict(:c => 299792458)  # speed of light
julia> tree, consts = lambda`λm. E0 * c`physics  # E0 from module
# physics now contains both :c and :E0 (if E0 was defined)

# Priority demonstration
julia> lambda_context[:shared] = "global"
julia> user_ctx = Dict(:shared => "local", :unique => "mine")
julia> tree, consts = lambda`λx. shared unique`user_ctx
# consts[:shared] == "local"  (flag wins over global)
```

### Advanced: Tree Constants as Subtrees
```julia
# Create reusable lambda components
julia> identity, _ = lambda`λx.x`
julia> twice, _ = lambda`λf.λx.f (f x)`

# Store as constants
julia> lambda_context[:id] = identity
julia> lambda_context[:twice] = twice

# Use in larger expressions - trees get inserted directly
julia> tree, consts = lambda`λg. twice (id g)`
# The id and twice are inserted as subtrees, not wrapped in ConstNodes
# This enables true compositional lambda calculus programming!
```

### Error Handling
```julia
# Parse errors (caught at macro expansion time)
julia> lambda`λ`        # Error: incomplete abstraction
julia> lambda`(`        # Error: unmatched parenthesis  
julia> lambda`#`        # Error: incomplete index

# Variable resolution (survives as VarNode if not found)
julia> tree, consts = lambda`λx. undefined_var`
# undefined_var remains as VarNode (not resolved)
# consts == Dict() (empty)
```

## Performance Notes

- **Macro expansion time**: All parsing and constant resolution happens immediately
- **No runtime overhead**: Returns ready-to-use `AstTree` structures
- **Memory efficient**: Tree constants shared by reference, not copied

## See Also

- [`collect_constants!`](@ref) - The underlying constant collection function
- [`lambda_context`](@ref) - Global constant dictionary  
- [`parse_lambda_literal`](@ref) - Low-level parsing function
"""
macro lambda_cmd(str)
    # For command literals, str should already be a string
    str_val = str isa String ? str : string(str)
    local _tree = parse_lambda_literal(str_val)
    local _consts = collect_constants!(_tree, __module__)
    return (_tree, _consts)
end

macro lambda_cmd(str, flag_context)
    # For command literals, we need to evaluate immediately like @show
    str_val = str isa String ? str : string(str)
    # In command literals, flag_context comes as a string, convert to symbol and evaluate
    flag_symbol = Symbol(flag_context)
    flag_val = Core.eval(__module__, flag_symbol)
    local _tree = parse_lambda_literal(str_val)
    local _consts = collect_constants!(_tree, __module__, flag_val)
    return (_tree, _consts)
end

########################################################
# Tests for Parse module
########################################################

@testitem "parse_lambda_literal basic cases" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: parse_lambda_literal
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode, IndexNode

    # λx. x
    tree1 = parse_lambda_literal("λx. x")
    @test tree1.root isa AbsNode
    @test tree1.root.body isa VarNode
    @test tree1.root.body.name == :x

    # (λx. x) y
    tree2 = parse_lambda_literal("(λx. x) y")
    @test tree2.root isa AppNode

    # f x y z  -> (((f x) y) z)
    tree3 = parse_lambda_literal("f x y z")
    @test tree3.root isa AppNode

    # De Bruijn index
    tree4 = parse_lambda_literal("#2")
    @test tree4.root isa IndexNode && tree4.root.index == 2

    # Compact abstraction without dots or spaces: λxλy.x #0  == λx. (λy. (x #0))
    tree5 = parse_lambda_literal("λxλy.x #0")
    # Expect nested AbsNodes
    @test tree5.root isa AbsNode
    inner = tree5.root.body
    @test inner isa AbsNode
    app = inner.body
    @test app isa AppNode
    # func should be VarNode(:x) captured from outer scope, arg IndexNode(0)
    @test app.func isa VarNode && app.func.name == :x
    @test app.arg isa IndexNode && app.arg.index == 0
end

@testitem "lambda macro comprehensive parsing" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode
    
    # Basic structures
    tree, _ = lambda`x`
    @test tree.root isa VarNode
    
    tree, _ = lambda`#0`
    @test tree.root isa IndexNode
    
    tree, _ = lambda`λx.x`
    @test tree.root isa AbsNode
    
    tree, _ = lambda`f g`
    @test tree.root isa AppNode
    
    # Complex lambda expressions
    # Church numerals
    tree, _ = lambda`λf.λx.f (f x)`  # Church numeral 2
    @test tree.root isa AbsNode
    @test tree.root.body isa AbsNode
    inner_app = tree.root.body.body
    @test inner_app isa AppNode
    @test inner_app.arg isa AppNode
    
    # Y combinator structure
    tree, _ = lambda`λf.(λx.f (x x)) (λx.f (x x))`
    @test tree.root isa AbsNode
    body = tree.root.body
    @test body isa AppNode
    @test body.func isa AbsNode
    @test body.arg isa AbsNode
    
    # Curried functions
    tree, _ = lambda`λx.λy.λz.x z (y z)`  # S combinator
    @test tree.root isa AbsNode
    @test tree.root.body isa AbsNode
    @test tree.root.body.body isa AbsNode
    s_body = tree.root.body.body.body
    @test s_body isa AppNode
    @test s_body.func isa AppNode
    
    # Mixed De Bruijn indices and variables
    tree, _ = lambda`λx.λy.x #0 y`
    @test tree.root.body.body isa AppNode
    app = tree.root.body.body
    @test app.func isa AppNode
    @test app.func.arg isa IndexNode
    @test app.arg isa VarNode
    
    # Nested parentheses
    tree, _ = lambda`((λx.x) (λy.y)) z`
    @test tree.root isa AppNode
    @test tree.root.func isa AppNode
    @test tree.root.func.func isa AbsNode
    @test tree.root.func.arg isa AbsNode
end

@testitem "lambda macro constant collection" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd
    using LambdaRegression.ASTs.NodeASTs: VarNode, ConstNode
    
    # Numeric literals
    tree, dict = lambda`λx. 42`
    @test haskey(dict, Symbol("42"))
    @test dict[Symbol("42")] == 42
    @test tree.root.body isa ConstNode
    
    # String literals  
    tree, dict = lambda`λx. "hello"`
    @test haskey(dict, Symbol("\"hello\""))
    @test dict[Symbol("\"hello\"")] == "hello"
    
    # Variables stay as variables
    tree, dict = lambda`λx. y`
    @test isempty(dict)
    @test tree.root.body isa VarNode
    
    # With module variable
    const test_const = 99
    tree, dict = lambda`λx. test_const`
    @test haskey(dict, :test_const)
    @test dict[:test_const] == 99
    @test tree.root.body isa ConstNode
    
    # Multiple constants
    tree, dict = lambda`λf. f 42 "test"`
    @test length(dict) == 2
    @test haskey(dict, Symbol("42"))
    @test haskey(dict, Symbol("\"test\""))
end

@testitem "lambda macro error cases" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: parse_lambda_literal
    
    # These should all error during parsing (test the underlying function directly)
    @test_throws ErrorException parse_lambda_literal("λ")
    @test_throws ErrorException parse_lambda_literal("λx")
    @test_throws ErrorException parse_lambda_literal("#")
    @test_throws ErrorException parse_lambda_literal("(")
    @test_throws ErrorException parse_lambda_literal(")")
end

@testitem "lambda_cmd macro basic" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd
    const myconst = 42
    tree, dict = lambda`λx. myconst`   # should capture value 42
    @test dict[:myconst] == 42
end

@testitem "lambda_cmd macro numeric constant" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd
    tree, dict = lambda`λx. 42`
    @test haskey(dict, Symbol("42"))
    @test dict[Symbol("42")] == 42
end

@testitem "lambda_context functionality" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    
    # Clear context first
    empty!(lambda_context)
    
    # Test lambda_context priority
    lambda_context[:testvar] = 123
    tree, dict = lambda`λx. testvar`
    @test dict[:testvar] == 123
    @test tree.root.body isa ConstNode
    
    # Test that lambda_context overrides module variables
    const testvar = 999  # This should be ignored
    tree, dict = lambda`λx. testvar`
    @test dict[:testvar] == 123  # Should use lambda_context value
end

@testitem "lambda flag context functionality" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context, collect_constants!, parse_lambda_literal
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    
    # Clear global context
    empty!(lambda_context)
    
    # Test the collect_constants! function directly instead of through macro
    # This avoids macro expansion issues while testing the core functionality
    
    # Set up contexts  
    user_context = Dict{Symbol,Any}(:flagvar => 456, :shared => "flag_wins")
    lambda_context[:shared] = "context_loses"
    
    # Test priority with direct function call
    tree = parse_lambda_literal("λx. flagvar shared")
    dict = collect_constants!(tree, @__MODULE__, user_context)
    
    @test haskey(dict, :flagvar)
    @test dict[:flagvar] == 456
    @test dict[:shared] == "flag_wins"
    
    # Test that global context is not polluted
    @test !haskey(lambda_context, :flagvar)
    @test lambda_context[:shared] == "context_loses"
end

@testitem "lambda literal with flag context" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    using LambdaRegression.ASTs.NodeASTs: ConstNode, AppNode
    
    # Clear global context
    empty!(lambda_context)
    
    # Test basic flag context literal syntax
    user_flags = Dict{Symbol,Any}(:myflag => 123, :yourflag => "test")
    tree, dict = lambda`λx. myflag yourflag`user_flags
    
    @test haskey(dict, :myflag)
    @test haskey(dict, :yourflag)
    @test dict[:myflag] == 123
    @test dict[:yourflag] == "test"
    @test tree.root.body isa AppNode
    
    # Test priority: flag context > lambda_context
    lambda_context[:priority_var] = "context_value"
    priority_flags = Dict{Symbol,Any}(:priority_var => "flag_value")
    tree, dict = lambda`λx. priority_var`priority_flags
    
    @test dict[:priority_var] == "flag_value"  # Flag should win
    @test lambda_context[:priority_var] == "context_value"  # Context unchanged
    
    # Test that flag context gets updated with new constants
    const flag_new_const = 999
    update_flags = Dict{Symbol,Any}(:existing => "already_here")
    tree, dict = lambda`λx. flag_new_const existing`update_flags
    
    @test haskey(update_flags, :flag_new_const)  # New constant added to flag dict
    @test update_flags[:flag_new_const] == 999
    @test update_flags[:existing] == "already_here"  # Existing preserved
    @test !haskey(lambda_context, :flag_new_const)  # Global not polluted
end

@testitem "lambda literal flag context with subtrees" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode
    
    # Clear global context
    empty!(lambda_context)
    
    # Create a subtree and put it in flag context
    subtree, _ = lambda`λa. a #0`
    tree_flags = Dict{Symbol,Any}(:my_subtree => subtree, :other => 42)
    
    # Use the subtree via flag context
    tree, dict = lambda`λx. my_subtree x`tree_flags
    
    @test dict[:my_subtree] == subtree
    # dict only contains constants actually referenced in the expression
    @test !haskey(dict, :other)  # :other wasn't referenced, so not in dict
    # But the flag context should be updated if there were new constants
    @test tree_flags[:other] == 42  # Original flag context preserves unreferenced values
    
    # Check that subtree was inserted correctly
    app_node = tree.root.body
    @test app_node isa AppNode
    @test app_node.func isa AbsNode  # Should be the subtree root directly inserted
    @test app_node.arg isa VarNode
    @test app_node.arg.name == :x
end

@testitem "tree constants as subtrees" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode
    
    # Clear context
    empty!(lambda_context)
    
    # Create a subtree
    subtree, _ = lambda`λa. a`
    lambda_context[:identity] = subtree
    
    # Use the subtree in another expression
    tree, dict = lambda`λx. identity x`
    @test dict[:identity] == subtree
    
    # Check that it was inserted as a subtree, not a ConstNode
    app_node = tree.root.body
    @test app_node isa AppNode
    @test app_node.func isa AbsNode  # Should be the subtree root (directly inserted)
    @test app_node.arg isa VarNode
end

@testitem "AstNode constants as subtrees without context update" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context, collect_constants!, parse_lambda_literal
    using LambdaRegression.ASTs.NodeASTs: VarNode, AppNode, ConstNode
    
    # Clear context
    empty!(lambda_context)
    
    # Create an AstNode 
    node = VarNode(:y)
    lambda_context[:n] = node
    original_size = length(lambda_context)
    
    # Use the node in another expression
    tree = parse_lambda_literal("λx. n")
    found = collect_constants!(tree, Main)
    
    # Check that the node was found but context wasn't updated (since it's an AstNode)
    @test found[:n] == node
    @test length(lambda_context) == original_size  # No new entries added
    
    # Check that node was inserted as subtree (wrapped in AstTree)
    body = tree.root.body
    @test body isa VarNode && body.name == :y  # Should be the original node directly inserted
end

@testitem "AstTree constants without context update" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context, collect_constants!, parse_lambda_literal
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode
    
    # Clear context
    empty!(lambda_context)
    
    # Create an AstTree
    subtree, _ = lambda`λa. a`
    lambda_context[:func] = subtree
    original_size = length(lambda_context)
    
    # Use the subtree in another expression
    tree = parse_lambda_literal("λx. func x")
    found = collect_constants!(tree, Main)
    
    # Check that the subtree was found but context wasn't updated (since it's an AstTree)
    @test found[:func] == subtree
    @test length(lambda_context) == original_size  # No new entries added
    
    # Check that subtree was inserted directly
    app_node = tree.root.body
    @test app_node isa AppNode
    @test app_node.func isa AbsNode  # Should be the subtree root directly inserted
    @test app_node.arg isa VarNode && app_node.arg.name == :x
end

@testitem "priority order verification" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    
    # Clear global context
    empty!(lambda_context)
    
    # Set up all three levels
    const priority_test = "module_level"
    lambda_context[:priority_test] = "context_level"
    user_context = Dict{Symbol,Any}(:priority_test => "flag_level")
    
    # Test flag wins (using literal with flag context)
    tree, dict = lambda`λx. priority_test`user_context
    @test dict[:priority_test] == "flag_level"
    
    # Test context wins over module (using literal without flag context)
    tree, dict = lambda`λx. priority_test`
    @test dict[:priority_test] == "context_level"
    
    # Remove context to test module fallback
    delete!(lambda_context, :priority_test)
    tree, dict = lambda`λx. priority_test`
    @test dict[:priority_test] == "module_level"
end

@testitem "dictionary update functionality" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    
    # Clear global context
    empty!(lambda_context)
    
    # Test updating lambda_context with new module constants
    const new_const1 = 555
    const new_const2 = "hello_world"
    
    tree, dict = lambda`λx. new_const1 new_const2`  # Should update lambda_context
    
    # Check that lambda_context was updated
    @test haskey(lambda_context, :new_const1)
    @test haskey(lambda_context, :new_const2)
    @test lambda_context[:new_const1] == 555
    @test lambda_context[:new_const2] == "hello_world"
    
    # Check that the returned dict contains the current constants found
    @test haskey(dict, :new_const1)
    @test haskey(dict, :new_const2)
    @test dict[:new_const1] == 555
    @test dict[:new_const2] == "hello_world"
end

@testitem "flag context update functionality" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    
    # Clear global context
    empty!(lambda_context)
    
    # Test updating flag context with new module constants
    const flag_const1 = 777
    const flag_const2 = "flag_test"
    
    user_dict = Dict{Symbol,Any}(:existing => "already_here")
    tree, dict = lambda`λx. flag_const1 flag_const2 existing`user_dict
    
    # Check that user_dict was updated with new constants
    @test haskey(user_dict, :flag_const1)
    @test haskey(user_dict, :flag_const2)
    @test user_dict[:flag_const1] == 777
    @test user_dict[:flag_const2] == "flag_test"
    @test user_dict[:existing] == "already_here"  # Pre-existing should remain
    
    # Check that lambda_context was NOT updated
    @test !haskey(lambda_context, :flag_const1)
    @test !haskey(lambda_context, :flag_const2)
    
    # Check that the returned dict contains the current constants found
    @test haskey(dict, :flag_const1)
    @test haskey(dict, :flag_const2)
    @test haskey(dict, :existing)  # Should include pre-existing from flag context
end

@testitem "cumulative lambda context updates" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    
    # Clear global context
    empty!(lambda_context)
    
    # Test cumulative updates through multiple lambda calls
    const cumulative_test1 = 111
    const cumulative_test2 = 222
    const cumulative_test3 = 333
    
    # First call - should add cumulative_test1
    tree1, dict1 = lambda`λx. cumulative_test1`
    @test haskey(lambda_context, :cumulative_test1)
    @test lambda_context[:cumulative_test1] == 111
    @test length(lambda_context) == 1
    
    # Second call - should add cumulative_test2, keep test1
    tree2, dict2 = lambda`λy. cumulative_test2`
    @test haskey(lambda_context, :cumulative_test1)
    @test haskey(lambda_context, :cumulative_test2)
    @test lambda_context[:cumulative_test2] == 222
    @test length(lambda_context) == 2
    
    # Third call - should add cumulative_test3, keep previous ones
    tree3, dict3 = lambda`λz. cumulative_test3`
    @test haskey(lambda_context, :cumulative_test1)
    @test haskey(lambda_context, :cumulative_test2)
    @test haskey(lambda_context, :cumulative_test3)
    @test lambda_context[:cumulative_test3] == 333
    @test length(lambda_context) == 3
end

@testitem "cumulative flag context updates" begin
    using LambdaRegression.ASTs.NodeASTs.Parse: @lambda_cmd, lambda_context
    
    # Clear global context
    empty!(lambda_context)
    
    # Test cumulative updates to user-provided flag context
    const flag_cumulative1 = 888
    const flag_cumulative2 = 999
    
    user_dict = Dict{Symbol,Any}()
    
    # First call - should add flag_cumulative1
    tree1, dict1 = lambda`λx. flag_cumulative1`user_dict
    @test haskey(user_dict, :flag_cumulative1)
    @test user_dict[:flag_cumulative1] == 888
    @test length(user_dict) == 1
    
    # Second call - should add flag_cumulative2, keep previous
    tree2, dict2 = lambda`λy. flag_cumulative2`user_dict
    @test haskey(user_dict, :flag_cumulative1)
    @test haskey(user_dict, :flag_cumulative2)
    @test user_dict[:flag_cumulative2] == 999
    @test length(user_dict) == 2
    
    # Verify lambda_context was not affected
    @test !haskey(lambda_context, :flag_cumulative1)
    @test !haskey(lambda_context, :flag_cumulative2)
end

end # module Parse