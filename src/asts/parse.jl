module Parse

using ..NodeASTs: AstNode, AstTree, AppNode, AbsNode, VarNode, ConstNode, IndexNode, setchild!, ScopeState, hasname
using ...Trees: PointerState, traverse, getstate
using ....Utils
using TestItems

export parse_lambda, @lambda

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
    parse_lambda(str::AbstractString, ctx=Dict{Symbol,Any}(); mod::Union{Module,Nothing}=nothing)

Parse a locally-nameless lambda calculus literal contained in `str` and return
an `AstTree` constructed from the [`AstNode`](@ref)s defined in this module, along with a context dictionary `ctx`.
The context dictionary can be used to resolve variable names to their values. 
If a module is provided, it will be used to resolve variables that are not found in the context dictionary.
the specific behavior is as follows:
- If a variable is bound, don't replace it, just keep it as a variable node.
- If a variable is found in the context dictionary, it will be replaced with a constant node that points to the value in the dictionary.
- If a variable is found in the context dictionary, and more specifically, if the value is an `AstTree` or an `AstNode`, it will replace the variable as a subtree.
- If a variable is not found in the context dictionary, it will try to resolve it from the provided module or try to parse it as a literal. If success, it will be added to the context dictionary and do the same replacement as above.
The accepted grammar is the same as for [`parse_lambda_literal`](@ref).
"""    
function parse_lambda(str::AbstractString, ctx=Dict{Symbol,Any}(); mod::Union{Module,Nothing}=nothing)
    tree = parse_lambda_literal(str)
    collect_constants!(tree, ctx, mod)
    return tree, ctx
end 

function collect_constants!(tree::AstTree, ctx::Dict{Symbol,Any}, mod::Union{Module,Nothing}=nothing)
    errored = Ref{Union{Nothing, LambdaParseErr}}(nothing)
    traverse(tree, PointerState, ScopeState) do t, node, bag
        if node isa VarNode
            name = node.name
            constant_val = nothing
            if hasname(getstate(bag, ScopeState), name) # Priority 0: scope context, pass
            elseif haskey(ctx, name) # Priority 1: flag context
                constant_val = ctx[name]
            elseif !isnothing(mod) && isdefined(mod, name) # Priority 2: module context
                constant_val = getfield(mod, name)
                ctx[name] = constant_val
            else # Priority 3: literal
                strname = String(name)
                parsed = Meta.parse(strname)
                if parsed isa Expr
                    errored[] = LambdaParseErr(strname)
                elseif parsed isa Symbol # pass
                else # literal
                    constant_val = parsed
                    ctx[name] = constant_val
                end
            end
            
            if constant_val !== nothing
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
    if !isnothing(errored[])
        return Err(errored[])
    end
end 

"""
    @lambda str
    @lambda str ctx

Macro to parse a locally-nameless lambda calculus literal contained in `str`.
The specific behavior is the same as [`parse_lambda`](@ref), but it use the caller module as the default module to resolve variables.
"""
macro lambda(str)
    :(parse_lambda($str; mod=$(__module__)))
end

macro lambda(str, ctx)
    :(parse_lambda($str, $(esc(ctx)); mod=$(__module__)))
end

@testitem "parse_lambda_literal basic cases" begin
    using LambdaRegression.ASTs.Parse: parse_lambda_literal
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

@testitem "lambda macro basic parsing" begin
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode

    tree, _ = @lambda "x"
    @test tree.root isa VarNode

    tree, _ = @lambda "#0"
    @test tree.root isa IndexNode && tree.root.index == 0

    tree, _ = @lambda "λx.x"
    @test tree.root isa AbsNode

    tree, _ = @lambda "f g"
    @test tree.root isa AppNode
end

@testitem "lambda macro constant collection" begin
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode, ConstNode

    # Numeric literals
    tree, dict = @lambda "λx. 42"
    @test haskey(dict, Symbol("42"))
    @test dict[Symbol("42")] == 42
    @test tree.root.body isa ConstNode

    # Variables stay as variables
    tree, dict = @lambda "λx. y"
    @test isempty(dict)
    @test tree.root.body isa VarNode

    # with module variable
    const test_const = 99
    tree, dict = @lambda "λx. test_const"
    @test haskey(dict, :test_const)
    @test dict[:test_const] == 99
    @test tree.root.body isa ConstNode

    # multiple constants
    tree, dict = @lambda "λx1.1.45 42"
    @test length(dict) == 2
    @test haskey(dict, Symbol("42"))
    @test haskey(dict, Symbol("1.45"))
end

@testitem "lambda macro with user context" begin
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.ASTs.NodeASTs: VarNode, IndexNode, AbsNode, AppNode, ConstNode

    ctx = Dict{Symbol, Any}(:foo=>123, :bar=>"hello")
    tree, dict = @lambda("λx. foo bar", ctx)
    @test dict[:foo] == 123
    @test dict[:bar] == "hello"
    @test tree.root.body.func isa ConstNode

    # priority
    foo = 111
    ## with ctx
    ctx = Dict{Symbol, Any}(:foo=>123, Symbol(1)=>42)
    tree, dict = @lambda "λx. foo 1" ctx
    @test dict[:foo] == 123
    @test dict[Symbol(1)] == 42
    @test tree.root.body.func isa ConstNode
    
    ## without ctx
    tree, dict = @lambda "λx. foo 1"
    @test dict[:foo] == 111
    @test dict[Symbol(1)] == 1
    @test tree.root.body.func isa ConstNode
end

end # module Parse
