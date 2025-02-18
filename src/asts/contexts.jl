module Contexts

# A simple context implementation for `ConstNode` evaluation using positional
# argument dispatch for maximum extensibility. Any Dict-like object can serve as a context.

export new_constant!, ConstSymbolFactory, EvalContext
public T, F

###############################################################################
# Helper utilities
###############################################################################

using ..NodeASTs: ConstNode, AstTree, AstNode, AbsNode, IndexNode  # re-exported by `ASTs` parent module

import ..NodeASTs: apply_node!

using Bijections
using TestItems

"""
    ConstSymbolFactory(; prefix=:c)

Deterministic, **non-global** constant-name generator for SR/evaluation contexts.

In multi-threaded symbolic regression, cross-individual global uniqueness is not
useful and can become a source of shared mutable state. This factory generates
names like `:c1`, `:c2`, ... and is intended to be **per-individual** or
**per-evaluation**.
"""
mutable struct ConstSymbolFactory
    counter::Int
    prefix::Symbol
end
ConstSymbolFactory(; prefix::Symbol=:c) = ConstSymbolFactory(0, prefix)

function _next_const_symbol!(f::ConstSymbolFactory)::Symbol
    f.counter += 1
    return Symbol(String(f.prefix), f.counter)
end

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
    for (name, existing) in context.globals
        isequal(existing, value) && return ConstNode(name)
    end
    if hasvalue(context.locals, value)
        return ConstNode(context.locals(value))
    else
        name = gensym(:c)
        context.locals[name] = value
        return ConstNode(name)
    end
end

"""
    new_constant!(context, value, factory::ConstSymbolFactory) -> ConstNode

Deterministic variant of [`new_constant!`](@ref) that uses a user-supplied,
non-global symbol factory instead of `gensym`.
"""
function new_constant!(context, value, factory::ConstSymbolFactory)
    while true
        name = _next_const_symbol!(factory)
        haskey(context, name) && continue
        context[name] = value
        return ConstNode(name)
    end
end

function new_constant!(context::ConstantsMap, value, factory::ConstSymbolFactory)
    for (name, existing) in context.globals
        isequal(existing, value) && return ConstNode(name)
    end
    if hasvalue(context.locals, value)
        return ConstNode(context.locals(value))
    else
        while true
            name = _next_const_symbol!(factory)
            (haskey(context.globals, name) || haskey(context.locals, name)) && continue
            context.locals[name] = value
            return ConstNode(name)
        end
    end
end

"""
    EvalContext(; overlay, locals, globals, symfactory)

Thread-safe evaluation context for `ConstNode` execution.

Reads consult layered maps in the order:
`overlay -> locals -> globals`.

Writes (including `new_constant!`) go only to `overlay`.
This makes it safe to run per-individual evaluations in parallel while sharing
`globals` as read-only configuration.
"""
mutable struct EvalContext
    overlay::Dict{Symbol,Any}
    locals::Dict{Symbol,Any}
    globals::Dict{Symbol,Any}
    symfactory::ConstSymbolFactory
end

function EvalContext(;
    overlay::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    locals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    globals::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    symfactory::ConstSymbolFactory=ConstSymbolFactory(),
)
    return EvalContext(overlay, locals, globals, symfactory)
end

Base.haskey(ctx::EvalContext, key::Symbol) = haskey(ctx.overlay, key) || haskey(ctx.locals, key) || haskey(ctx.globals, key)
function Base.getindex(ctx::EvalContext, key::Symbol)
    if haskey(ctx.overlay, key)
        return ctx.overlay[key]
    elseif haskey(ctx.locals, key)
        return ctx.locals[key]
    elseif haskey(ctx.globals, key)
        return ctx.globals[key]
    end
    throw(KeyError(key))
end
Base.setindex!(ctx::EvalContext, value, key::Symbol) = (ctx.overlay[key] = value)
function Base.get(ctx::EvalContext, key::Symbol, default)
    if haskey(ctx.overlay, key)
        return ctx.overlay[key]
    elseif haskey(ctx.locals, key)
        return ctx.locals[key]
    elseif haskey(ctx.globals, key)
        return ctx.globals[key]
    end
    return default
end

function new_constant!(ctx::EvalContext, value)
    while true
        name = _next_const_symbol!(ctx.symfactory)
        haskey(ctx, name) && continue
        ctx.overlay[name] = value
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

########################################################
# Test Items
########################################################

@testsnippet ContextsBaseImports begin
    using LambdaRegression.ASTs.Contexts: EvalContext, ConstSymbolFactory, new_constant!
    using LambdaRegression.ASTs.Contexts: EvalContext, ConstSymbolFactory
    using LambdaRegression.ASTs.NodeASTs: AstTree, AppNode, ConstNode
    using LambdaRegression.Reduces: beta_reduce!
end

@testitem "EvalContext overlay write isolation + deterministic new_constant!" setup=[ContextsBaseImports] begin
    globals = Dict{Symbol,Any}(:g => 1)
    locals = Dict{Symbol,Any}(:l => 2)
    overlay = Dict{Symbol,Any}(:o => 3)
    ctx = EvalContext(overlay=overlay, locals=locals, globals=globals, symfactory=ConstSymbolFactory(prefix=:c))

    ctx[:k] = 10
    @test haskey(overlay, :k)
    @test !haskey(globals, :k)
    @test !haskey(locals, :k)

    c1 = new_constant!(ctx, 42)
    @test c1.name == :c1
    @test ctx.overlay[:c1] == 42
    @test !haskey(globals, :c1)
end

@testitem "EvalContext integrates with Reduce ConstNode evaluation" setup=[ContextsBaseImports] begin
    globals = Dict{Symbol,Any}(:double => (x -> 2x))
    overlay = Dict{Symbol,Any}(:three => 3)
    ctx = EvalContext(overlay=overlay, globals=globals, symfactory=ConstSymbolFactory(prefix=:c))

    tree = AstTree(AppNode(ConstNode(:double), ConstNode(:three)))
    r = beta_reduce!(tree, ctx)
    @test r == false
    @test tree.root isa ConstNode
    @test tree.root.name == :c1
    @test ctx.overlay[:c1] == 6
    @test !haskey(globals, :c1)
end

@testitem "ConstantsMap deterministic insertion respects globals" setup=[ContextsBaseImports] begin
    context = LambdaRegression.ASTs.Contexts.ConstantsMap(:c1 => 42)
    factory = ConstSymbolFactory(prefix=:c)
    @test new_constant!(context, 42, factory).name == :c1
    @test new_constant!(context, 7, factory).name == :c2
end

end # module Contexts
