module Parse

using ..NodeASTs: AstNode, AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, setchild!, ScopeState, hasname
using ...Trees: PointerState, traverse, getstate
using ...Utils
using TestItems

export parse_lambda, @terms

struct LambdaParseErr # Err type for lambda parsing
    str::String
end
struct ParseKindErr
    expected::Union{Symbol,Nothing}
    got::Union{Symbol,Nothing}
end

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
    pos[] <= length(tokens) && return Err(LambdaParseErr(str))
    return AstTree(expr)
end

struct _Tok
    kind::Symbol      # :lambda, :dot, :lparen, :rparen, :ident, :index
    val::Any
end

const _DELIMS = Set(['λ', '(', ')', '#'])
_is_delim(c::Char) = isspace(c) || c in _DELIMS

function _tokenize_lambda(str::AbstractString)
    delims = Set(['λ', '(', ')', '#'])
    _is_delim(c::Char) = isspace(c) || c in delims
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
            push!(delims, '.')
            i = nextind(str, i)
        elseif c == '.' && length(toks) >= 2 && toks[end].kind == :ident && toks[end-1].kind == :lambda
            # Dot acts as separator only immediately after a λ <ident> pair
            push!(toks, _Tok(:dot, nothing))
            pop!(delims, '.')
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
            j > last && return Err(LambdaParseErr(str)) # don't check, just pass up
            start = j
            while j <= last && isdigit(str[j])
                j = nextind(str, j)
            end
            idx_str = str[start:prevind(str, j)]
            isempty(idx_str) && return Err(LambdaParseErr(str))
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
    pos[] > length(toks) && return Err(ParseKindErr(kind_sym, nothing))
    tok = toks[pos[]]
    tok.kind == kind_sym || return Err(ParseKindErr(kind_sym, tok.kind))
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
    name_tok = @✓ _expect!(toks, pos, :ident)
    if _peekkind(toks, pos) == :dot
        @✓ _expect!(toks, pos, :dot)
    end
    body = @✓ _parse_term(toks, pos)
    return AbsNode(name_tok.val, body)
end

# application ::= atom (atom)*         (left-associative)
function _parse_application(toks, pos)::AstNode
    node = @✓ _parse_atom(toks, pos)
    while true
        kind = _peekkind(toks, pos)
        if kind in (:ident, :index, :lparen, :lambda)
            arg = @✓ _parse_atom(toks, pos)
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
        tok = @✓ _expect!(toks, pos, :ident)
        return VarNode(tok.val)
    elseif kind == :index
        tok = @✓ _expect!(toks, pos, :index)
        return IndexNode(tok.val)
    elseif kind == :lparen
        @✓ _expect!(toks, pos, :lparen)
        expr = @✓ _parse_term(toks, pos)
        @✓ _expect!(toks, pos, :rparen)
        return expr
    elseif kind == :lambda
        return @✓ _parse_lambda(toks, pos)
    else
        return Err(ParseKindErr(nothing, kind))
    end
end

"""
    parse_lambda(str::AbstractString, ctx::Dict{Symbol,Any}=Dict{Symbol,Any}()) -> AstTree

Parse a locally-nameless lambda calculus literal contained in `str` and return
an `AstTree` constructed from the [`AstNode`](@ref)s defined in this module.
The context dictionary `ctx` is used to resolve constants:

- If a variable is bound, keep it as a variable node.
- If a variable is found in `ctx`, it will be replaced with a constant node,
  or a subtree if the value is an `AstTree` or `AstNode`.
- If a variable is not found in `ctx`, it remains a `VarNode`.

The accepted grammar is the same as for [`parse_lambda_literal`](@ref).
"""
function parse_lambda(str::AbstractString, ctx::Dict{Symbol,Any}=Dict{Symbol,Any}())
    tree = parse_lambda_literal(str)
    collect_constants!(tree, ctx)
    return tree
end

function collect_constants!(tree::AstTree, ctx::Dict{Symbol,Any})
    traverse(tree, PointerState, ScopeState) do t, node, bag
        if node isa VarNode
            name = node.name
            if hasname(getstate(bag, ScopeState), name)
                return nothing
            end
            if haskey(ctx, name)
                constant_val = ctx[name]
                pointer = getstate(bag, PointerState{AstNode})
                if constant_val isa AstTree
                    setchild!(tree, pointer.parent, pointer.child_index, constant_val)
                elseif constant_val isa AstNode
                    setchild!(tree, pointer.parent, pointer.child_index, AstTree(constant_val))
                else
                    setchild!(tree, pointer.parent, pointer.child_index, AstTree(ConstNode(name)))
                end
            end
        end
    end
    return tree
end


"""
    @terms args...

Build a context dictionary for `parse_lambda` from a list of terms.
Identifiers map to their in-scope values. Literal values are keyed by
`Symbol(string(literal))`. Assignments like `n=11` are supported, and the
right-hand side can be any valid Julia expression.

### Using `@terms` as a template pool

`@terms` can also be used to register AST-valued entries for later reuse, e.g.
`Y=parse_lambda(...)` stores an `AstTree` under `:Y`. This is useful for tree
generation modules that want to sample from a template pool.

If you need a **type context** (`Dict{Symbol,TyLike}`) from a `@terms` dictionary:

- Use `TypeSystem.get_type_ctx` for non-AST values (it intentionally skips `AstTree`/`AstNode` entries).
- Use `TypeSystem.get_ast_type_ctx` to infer types for AST-valued entries (templates).

Examples
```julia
ctx = @terms(IADD, IMUL, 12, n=11)
merge!(ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", ctx)))
```
"""
macro terms(args...)
    pairs = Expr[]
    for arg in args
        if arg isa Expr && (arg.head == :(=) || arg.head == :kw)
            lhs = arg.args[1]
            rhs = arg.args[2]
            if !(lhs isa Symbol)
                return :(throw(ArgumentError("@terms only supports Symbol on the left-hand side of assignments")))
            end
            push!(pairs, :($(QuoteNode(lhs)) => $(esc(rhs))))
        elseif arg isa Symbol
            push!(pairs, :($(QuoteNode(arg)) => $(esc(arg))))
        elseif arg isa QuoteNode && arg.value isa Symbol
            sym = arg.value
            push!(pairs, :($(QuoteNode(sym)) => $(QuoteNode(sym))))
        elseif arg isa Number || arg isa String || arg isa Char || arg isa Bool || arg === nothing
            key = Symbol(string(arg))
            push!(pairs, :($(QuoteNode(key)) => $(arg)))
        else
            key = Symbol(string(arg))
            push!(pairs, :($(QuoteNode(key)) => $(esc(arg))))
        end
    end
    return :(Dict{Symbol,Any}($(pairs...)))
end


@testsnippet ParseBaseImports begin
    using LambdaRegression.ASTs.Parse: parse_lambda_literal
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode, IndexNode
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode, ConstNode
    using LambdaRegression.ASTs.Parse: @terms
    using LambdaRegression.ASTs.NodeASTs: AstTree, VarNode, ConstNode
    using LambdaRegression.ASTs.Parse: @terms, parse_lambda
    using LambdaRegression.ASTs.NodeASTs: AstTree
end

@testitem "parse_lambda_literal basic cases" setup=[ParseBaseImports] begin
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

@testitem "parse_lambda basic parsing" setup=[ParseBaseImports] begin
    tree = parse_lambda("x")
    @test tree.root isa VarNode

    tree = parse_lambda("#0")
    @test tree.root isa IndexNode && tree.root.index == 0

    tree = parse_lambda("λx.x")
    @test tree.root isa AbsNode

    tree = parse_lambda("f g")
    @test tree.root isa AppNode
end

@testitem "parse_lambda constant collection" setup=[ParseBaseImports] begin
    # Numeric literals
    ctx = @terms 42
    tree = parse_lambda("λx. 42", ctx)
    @test haskey(ctx, Symbol("42"))
    @test ctx[Symbol("42")] == 42
    @test tree.root.body isa ConstNode

    # Variables stay as variables
    tree = parse_lambda("λx. y")
    @test tree.root.body isa VarNode

    ctx = @terms test_const=99
    tree = parse_lambda("λx. test_const", ctx)
    @test haskey(ctx, :test_const)
    @test ctx[:test_const] == 99
    @test tree.root.body isa ConstNode

    # multiple constants
    ctx = @terms(1.45, 42)
    tree = parse_lambda("λx1.1.45 42", ctx)
    @test length(ctx) == 2
    @test haskey(ctx, Symbol("42"))
    @test haskey(ctx, Symbol("1.45"))
end

@testitem "parse_lambda with user context" setup=[ParseBaseImports] begin
    ctx = @terms(foo=123, bar="hello")
    tree = parse_lambda("λx. foo bar", ctx)
    @test ctx[:foo] == 123
    @test ctx[:bar] == "hello"
    @test tree.root.body.func isa ConstNode

    # priority
    foo = 111
    # with ctx
    ctx = @terms(foo=123, 1)
    tree = parse_lambda("λx. foo 1", ctx)
    @test ctx[:foo] == 123
    @test ctx[Symbol(1)] == 1
    @test tree.root.body.func isa ConstNode
    
    # without ctx
    tree = parse_lambda("λx. foo 1")
    @test tree.root.body.func isa VarNode
    @test tree.root.body.arg isa VarNode
end

@testitem "terms macro basics" setup=[ParseBaseImports] begin
    IADD = :add_impl
    IMUL = :mul_impl
    ctx = @terms(IADD, IMUL, 12, n=11)

    @test ctx[:IADD] == :add_impl
    @test ctx[:IMUL] == :mul_impl
    @test ctx[Symbol("12")] == 12
    @test ctx[:n] == 11
end

@testitem "terms macro supports rhs expressions" setup=[ParseBaseImports] begin
    base_ctx = @terms(0, 1)
    ctx = merge(base_ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", base_ctx)))
    @test haskey(ctx, :Y)
    @test ctx[:Y] isa AstTree
end

end # module Parse
