module Contexts

# A simple context implementation for `ConstNode` evaluation using positional
# argument dispatch for maximum extensibility. Any Dict-like object can serve as a context.

export new_constant!
public T, F

###############################################################################
# Helper utilities
###############################################################################

using ..NodeASTs: ConstNode, AstTree, AstNode, AbsNode, IndexNode  # re-exported by `ASTs` parent module

import ..NodeASTs: apply_node!

using Bijections

struct ConstantsMap
    globals::Dict{Symbol,Any}
    locals::Bijection{Symbol,Any}
end
ConstantsMap(globals...) = ConstantsMap(Dict(globals), Bijection{Symbol,Any}())
Base.getindex(map::ConstantsMap, index) = get(map.globals, index) do 
    map.locals[index]
end

struct ConstantProxy
    body
end
get_value(x) = x
get_value(x::ConstantProxy) = x.body()

"""
    new_constant!(context, value) -> ConstNode

Create a fresh `ConstNode` whose `name` is a unique `Symbol` generated via
`gensym`, store the provided `value` in the `context` under this symbol, and return
**the newly created `ConstNode`**.

The `context` can be any Dict-like object supporting `setindex!`.

This helper is primarily used by `apply_node!` when evaluating       
`ConstNode` applications.
"""
function new_constant!(context, value)
    name = gensym(:c)        # e.g. :c#257
    context[name] = value
    return ConstNode(name)
end

function new_constant!(context::ConstantsMap, value)
    if hasvalue(context.locals, value)
        return ConstNode(context.locals(value))
    else
        name = gensym(:c)
        context.locals[name] = value
        return ConstNode(name)
    end
end

###############################################################################
# Extending `apply_node!` for `ConstNode` application with a context
###############################################################################

"""
    apply_node!(tree, f::ConstNode, x::ConstNode, context) -> ConstNode

Evaluate the application `f x` where both the function and the argument are
`ConstNode`s, using the provided context as a fourth positional argument.

The corresponding Julia values are looked up in `context` using their `name` field
and **invoked without any sanity-checking** – this is intentional as more
elaborate validation is delegated to the separate type-checking layer.

The result of the call is stored back into the context under a fresh symbol and
a matching `ConstNode` is returned so that the AST continues to reference a
constant.  The original AST `tree` is untouched; callers should replace the
application node with the returned constant node as needed.

The `context` can be any Dict-like object supporting `getindex` and `setindex!`,
enabling flexible context implementations like layered dictionaries.
"""
function apply_node!(tree::AstTree, f::ConstNode, x::ConstNode, context)
    # Look up the Julia values bound to the constant symbols
    f_val = get_value(context[f.name])
    x_val = get_value(context[x.name])
    # Perform the call; no checking – we follow the "let it crash" philosophy
    result_val = f_val(x_val)
    # If the host function already built a piece of AST, we can return it
    if result_val isa AstTree
        return result_val 
    elseif result_val isa AstNode
        return AstTree(result_val)
    else
        return AstTree(new_constant!(context, result_val))
    end
end

x = gensym(:x) # avoid name conflict
y = gensym(:y)
const T = AstTree(AbsNode(x, AbsNode(y, IndexNode(1))))
const F = AstTree(AbsNode(x, AbsNode(y, IndexNode(0))))
COND(x) = x ? T : F

end # module Contexts