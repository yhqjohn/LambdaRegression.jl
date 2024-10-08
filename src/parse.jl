module ParseModule
using MacroTools, AbstractTrees

abstract type AbstractToken <: AbstractNode{Any} end
abstract type AbstractVariableToken <: AbstractToken end

# named variable style
struct VariableToken <: AbstractVariableToken
    name::Symbol
end
VariableToken(name::String) = VariableToken(Symbol(name))
AbstractTrees.children(node::VariableToken) = []
AbstractTrees.nodevalue(node::VariableToken) = node.name
AbstractTrees.printnode(io::IO, node::VariableToken) = print(io, string(node.name))

# de Bruijn index style
struct DeBruijnToken <: AbstractVariableToken
    index::Int
end
AbstractTrees.children(node::DeBruijnToken) = []
AbstractTrees.nodevalue(node::DeBruijnToken) = node.index
AbstractTrees.printnode(io::IO, node::DeBruijnToken) = print(io, "#$(node.index)")

make_arg_node(arg::String) = VariableToken(arg)
make_arg_node(arg::Symbol) = VariableToken(arg)
make_arg_node(arg::Int) = DeBruijnToken(arg)
make_arg_node(arg::AbstractVariableToken) = arg

struct LambdaToken <: AbstractToken
    arg::AbstractVariableToken
    body::AbstractToken
end
AbstractTrees.children(node::LambdaToken) = [node.arg, node.body]
AbstractTrees.nodevalue(node::LambdaToken) = nothing
AbstractTrees.printnode(io::IO, node::LambdaToken) = print(io, "λ")

struct MultiLambdaToken <: AbstractToken
    args::Tuple{Vararg{AbstractVariableToken}}
    body::AbstractToken
end
AbstractTrees.children(node::MultiLambdaToken) = [node.args..., node.body]
AbstractTrees.nodevalue(node::MultiLambdaToken) = nothing
AbstractTrees.printnode(io::IO, node::MultiLambdaToken) = print(io, "λ")

struct DeBruijnLambdaToken <: AbstractToken
    body::AbstractToken
end
AbstractTrees.children(node::DeBruijnLambdaToken) = [node.body]
AbstractTrees.nodevalue(node::DeBruijnLambdaToken) = nothing
AbstractTrees.printnode(io::IO, node::DeBruijnLambdaToken) = print(io, "λ")

struct MultiDeBruijnLambdaToken <: AbstractToken
    argnum::Int
    body::AbstractToken
end
AbstractTrees.children(node::MultiDeBruijnLambdaToken) = [node.body]
AbstractTrees.nodevalue(node::MultiDeBruijnLambdaToken) = nothing
AbstractTrees.printnode(io::IO, node::MultiDeBruijnLambdaToken) = print(io, "λ"^node.argnum*".")

struct ApplyToken <: AbstractToken
    func::AbstractToken
    arg::AbstractToken
end
AbstractTrees.children(node::ApplyToken) = [node.func, node.arg]
AbstractTrees.nodevalue(node::ApplyToken) = nothing
AbstractTrees.printnode(io::IO, node::ApplyToken) = print(io, "APPLY")

struct MultiApplyToken <: AbstractToken
    func::AbstractToken
    args::Tuple{Vararg{AbstractToken}}
end
AbstractTrees.children(node::MultiApplyToken) = [node.func, node.args...]
AbstractTrees.nodevalue(node::MultiApplyToken) = nothing
AbstractTrees.printnode(io::IO, node::MultiApplyToken) = print(io, "APPLY")


struct ProxyToken{T} <: AbstractToken
    body::T
end
AbstractTrees.children(node::ProxyToken) = [node.body]
AbstractTrees.nodevalue(node::ProxyToken) = node.body

Base.:*(a::AbstractToken, b::AbstractToken) = ApplyToken(a, b)
-->(a::AbstractVariableToken, b::AbstractToken) = LambdaToken(a, b)

"""
    parse_lambda(expr)

Parse a lambda expression Literal into an Abstract Syntax Tree (AST) of `AbstractToken`s.
Two styles of lambda expressions are supported:
- Named variable style: `f-> (x-> f(x * x)) * (x-> f(x * x))`
- De Bruijn index style: `&2-> (&1-> &2(&1 * &1)) * (&1-> &2(&1 * &1))`

Function calling can be written either in the form of `f*x` or `f(x)`.
Expressions that cannot be found in the abstraction variables (or `context`, if called with a context) are treated as free variables,
which should be defined in the context scope, and will be wrapped in `ProxyToken`.

**Note**
The free variables must be written in the form of a literal, such as a string, a number, or a symbol, rather than a expression.
"""
function parse_lambda(expr, context=[])
    #=
        - case 1: a bounded variable
            - case 1.1: a named variable `x`
            - case 1.2: a de Bruijn index `&1`
        - case 2: an unbounded variable
        - case 3: an abstraction expression `x-> M`
            - case 3.1: a single argument abstraction `x-> M`
            - case 3.2: a multiple argument abstraction `(x, y, z)-> M`
            - case 3.3: a De Bruijn index abstraction `(*)-> M`
            - case 3.4: a multiple argument De Bruijn index abstraction `(*, *)-> M`
        - case 4: an application expression `f*x`
        - case 5: an application expression `f(x)`
            - case 5.1: a single argument application `f(x)`
            - case 5.2: a multiple argument application `f(x, y, z)`
    =#
    if expr isa Expr
        # TODO
        if expr.head === :-> # case 3: an abstraction expression `x-> M`
            args = expr.args[1]
            body = expr.args[2]
            body = Base.remove_linenums!(body)
            body isa Expr || throw(ArgumentError("The body of an abstraction must be an expression"))
            (body.head==:block && length(body.args)==1) || throw(ArgumentError("The body of an abstraction must be a single expression"))
            body = body.args[1]
            # deal with arguments
            if args === :(*) # case 3.3: a de Bruijn index abstraction `(*)-> M`
                body = parse_lambda(body, context)
                return :(DeBruijnLambdaToken($body))
            elseif args isa Expr && args.head==:tuple
                if all(==(:(*)), args.args) # case 3.4: a multiple argument De Bruijn index abstraction `(*, *)-> M`
                    body = parse_lambda(body, context)
                    return :(MultiDeBruijnLambdaToken($(length(args.args)), $body))
                else # case 3.2: a multiple argument abstraction `(x, y, z)-> M`
                    args = args.args
                    check_arg.(args)
                    context = [context; args...]
                    args = parse_arg.(args)
                    args = Expr(:tuple, args...)
                    body = parse_lambda(body, context)
                    return :(MultiLambdaToken($args, $body))
                end
            else # case 3.1: a single argument abstraction `x-> M`
                check_arg(args)
                context = [context; args]
                arg = parse_arg(args)
                body = parse_lambda(body, context)
                return :(LambdaToken($(arg), $body))
            end
            
        elseif expr.head === :call
            if expr.args[1] === :* # case 4: an application expression `f*x`
                args = [parse_lambda(arg, context) for arg in expr.args[2:end]]
                return Expr(:call, :*, args...)
            else # case 5: an application expression `f(x)`
                func = parse_lambda(expr.args[1], context)
                args = expr.args[2:end]
                args = [parse_lambda(arg, context) for arg in args]
                args = Expr(:tuple, args...)
                return :(MultiApplyToken($func, $args))
            end
        elseif expr.head === :& # case 1.2: a de Bruijn index `&1`
            check_arg(expr)
            return parse_arg(expr)
        else
            throw(ArgumentError("The expression head must be `:->`, `:call`, or `:&`"))
        end
    elseif expr isa Symbol && expr in context # case 1.1: a bounded variable `x`
        check_arg(expr)
        return parse_arg(expr)
    else # case 2: an unbounded variable
        return ProxyToken(expr)
    end
end

function check_arg(arg)
    arg isa Symbol && Base.isidentifier(arg) && return # case 1.1: a symbol
    arg isa Expr && arg.head==:& && arg.args[1] isa Int && return # case 1.2: a de Bruijn index
    throw(ArgumentError("The argument must be a symbol or a de Bruijn index in the form of `&1`"))
end

function parse_arg(arg)
    arg isa Symbol && return :(VariableToken($(QuoteNode(arg)))) # case 1.1: a symbol
    arg isa Expr && arg.head==:& && arg.args[1] isa Int && return :(DeBruijnToken($(arg.args[1]))) # case 1.2: a de Bruijn index
end

macro lambda(expr)
    expr = parse_lambda(expr)
    return expr
end

function unfold(node::AbstractToken)
    if node isa ProxyToken || node isa VariableToken || node isa DeBruijnToken
        return node
    elseif node isa LambdaToken
        return LambdaToken(node.arg, unfold(node.body))
    elseif node isa DeBruijnLambdaToken
        return DeBruijnLambdaToken(unfold(node.body))
    elseif node isa MultiDeBruijnLambdaToken
        if node.argnum<1
            throw(ArgumentError("De Bruijn lambda must have at least one argument"))
        elseif node.argnum==1
            return DeBruijnLambdaToken(unfold(node.body))
        else
            return DeBruijnLambdaToken(unfold(MultiDeBruijnLambdaToken(node.argnum-1, node.body)))
        end
    elseif node isa ApplyToken
        return ApplyToken(unfold(node.func), unfold(node.arg))
    elseif node isa MultiLambdaToken
        args = node.args
        body = node.body
        if length(args)<1
            throw(ArgumentError("lambda must have at least one argument"))
        elseif length(args)==1
            return LambdaToken(args[1], unfold(body))
        else
            return LambdaToken(args[1], unfold(MultiLambdaToken(args[2:end], body)))
        end
    elseif node isa MultiApplyToken
        func = node.func
        args = node.args
        if length(args)<1
            throw(ArgumentError("multi-apply must have at least one term"))
        elseif length(args)==1
            return ApplyToken(unfold(func), unfold(args[1]))
        else
            return ApplyToken(unfold(MultiApplyToken(func, args[1:end-1])), unfold(args[end]))
        end
    else
        throw(ArgumentError("The node must be a ProxyToken, VariableToken, DeBruijnToken, LambdaToken, ApplyToken, MultiLambdaToken, or MultiApplyToken"))
    end
end


end