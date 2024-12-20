module TypeSystem

using TestItems

export TyConst, TyVar, TyFun, TyMu, TyScheme, infer, TyError, UnboundVariableError, OccursCheckError, MismatchError

# ------------------------------------------------------------
# Basic type data structures (internal)
# ------------------------------------------------------------
abstract type Ty end

"""
    TyConst(name::Symbol)
A concrete (monomorphic) type, e.g. `:Int`, `:Bool`, `:Float`.
"""
struct TyConst <: Ty
    name::Symbol
end

"""
    TyVar(id, level)
A type variable created during inference.
• `id`    – unique identifier (integer)
• `level` – creation level, used for generalisation
• `instance` – if unified, points to another `Ty` (acts like union-find parent)
"""
mutable struct TyVar <: Ty
    id::Int
    level::Int
    instance::Union{Ty, Nothing}
end

"""
    TyFun(arg, res)
Function type `arg → res`.
"""
struct TyFun <: Ty
    arg::Ty
    res::Ty
end

"""
    TyMu(var, body)
Iso-recursive μ-type  `μ var . body`.
The `var` field is a `TyVar` acting as the binder.
"""
struct TyMu <: Ty
    var::TyVar
    body::Ty
end

"""
    TyScheme(vars, ty)
Polymorphic type scheme  ∀ vars. ty
• `vars` – vector of bound TyVar (no instance)
• `ty`   – monotype body
"""
struct TyScheme
    vars::Vector{TyVar}
    ty::Ty
end

# ------------------------------------------------------------
# Error reporting
# ------------------------------------------------------------
abstract type TyError end

struct UnboundVariableError <: TyError
    name::Symbol
end

struct OccursCheckError <: TyError
    var::TyVar
    ty::Ty
end

struct MismatchError <: TyError
    t1::Ty
    t2::Ty
end

# ------------------------------------------------------------
# Internal fresh id counter
# ------------------------------------------------------------
const _TYVAR_COUNTER = Ref{Int}(0)

fresh_tyvar(level::Int)::TyVar = TyVar(((_TYVAR_COUNTER[] += 1); _TYVAR_COUNTER[]), level, nothing)

# ------------------------------------------------------------
# Utility: prune – follow instances to canonical representative
# ------------------------------------------------------------
function prune(t::Ty)::Ty
    if t isa TyVar && t.instance !== nothing
        t.instance = prune(t.instance)
        return t.instance
    elseif t isa TyFun
        return TyFun(prune(t.arg), prune(t.res))
    elseif t isa TyMu
        return TyMu(t.var, prune(t.body))
    else
        return t
    end
end

# ------------------------------------------------------------
# Occurs check with polarity
# ------------------------------------------------------------
# return (occurs::Bool, negative_occurs::Bool)
function _occurs(var::TyVar, ty::Ty, polarity::Bool)::Tuple{Bool,Bool}
    ty = prune(ty)
    if ty isa TyVar
        return (ty === var, false)
    elseif ty isa TyConst
        return (false, false)
    elseif ty isa TyFun
        occ1, neg1 = _occurs(var, ty.arg, !polarity)  # arg position flips polarity (negative)
        occ2, neg2 = _occurs(var, ty.res, polarity)    # result keeps polarity
        return (occ1 || occ2, neg1 || neg2)
    elseif ty isa TyMu
        # Bound var shadowing: if same var id, stop search
        if ty.var === var
            return (false, false)
        else
            return _occurs(var, ty.body, polarity)
        end
    else
        return (false, false)
    end
end

# ------------------------------------------------------------
# Binding TyVar with occurs-check & μ-fold
# ------------------------------------------------------------
function _bind!(var::TyVar, ty::Ty)
    occurs, negative = _occurs(var, ty, true)
    if !occurs
        var.instance = ty
        return nothing
    elseif negative
        throw(OccursCheckError(var, ty))
    else
        # positive recursive occurrence – create μ type and bind
        folded = TyMu(var, ty)
        var.instance = folded
        return nothing
    end
end

# ------------------------------------------------------------
# Unification (destructive, throws TyError)
# ------------------------------------------------------------
function unify!(t1::Ty, t2::Ty)
    a = prune(t1)
    b = prune(t2)
    if a === b
        return
    elseif a isa TyVar
        _bind!(a, b)
    elseif b isa TyVar
        _bind!(b, a)
    elseif a isa TyConst && b isa TyConst && a.name == b.name
        return
    elseif a isa TyFun && b isa TyFun
        unify!(a.arg, b.arg)
        unify!(a.res, b.res)
    elseif a isa TyMu && b isa TyMu
        unify!(a.body, b.body)
    else
        throw(MismatchError(a, b))
    end
end

# ------------------------------------------------------------
# Instantiate & Generalise
# ------------------------------------------------------------
function instantiate(s::TyScheme, level::Int)::Ty
    subst = Dict{Int,TyVar}()
    mapvar(v::TyVar) = get!(subst, v.id) do
        fresh_tyvar(level)
    end
    return _inst(s.ty, mapvar)
end

function _inst(t::Ty, mapvar)
    if t isa TyVar
        return mapvar(t)
    elseif t isa TyConst
        return t
    elseif t isa TyFun
        return TyFun(_inst(t.arg, mapvar), _inst(t.res, mapvar))
    elseif t isa TyMu
        return TyMu(t.var, _inst(t.body, mapvar))
    else
        return t
    end
end

function _collect_free_vars(t::Ty, seen::Set{TyVar})
    t = prune(t)
    if t isa TyVar
        push!(seen, t)
    elseif t isa TyFun
        _collect_free_vars(t.arg, seen)
        _collect_free_vars(t.res, seen)
    elseif t isa TyMu
        _collect_free_vars(t.body, seen)
    end
    return seen
end

function generalize(t::Ty, level::Int)::TyScheme
    seen = _collect_free_vars(t, Set{TyVar}())
    vars = [v for v in seen if v.level > level]
    return TyScheme(vars, t)
end

# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------
# Env stack for de-Bruijn indices
mutable struct Env
    stack::Vector{TyScheme}
    names::Dict{Symbol,TyScheme}
end
Env() = Env(TyScheme[], Dict{Symbol,TyScheme}())

function Base.push!(env::Env, name::Union{Symbol,Nothing}, scheme::TyScheme)
    push!(env.stack, scheme)
    if name !== nothing
        env.names[name] = scheme
    end
end

Base.pop!(env::Env) = pop!(env.stack) # names dict cleanup left implicit; shadowing handled by stack discipline

lookup(env::Env, idx::Int)::TyScheme = env.stack[end - idx]

lookup(env::Env, name::Symbol)::Union{TyScheme,Nothing} = get(env.names, name, nothing)

# ------------------------------------------------------------
# Main inference (returns Ty or throws TyError)
# ------------------------------------------------------------
using ..ASTs.NodeASTs: AstTree, AstNode, VarNode, IndexNode, AbsNode, AppNode, ConstNode, LetNode

function infer(tree::AstTree, ctx::Dict{Symbol,TyConst})
    env = Env()
    level = 0
    try
        ty = _infer(tree.root, env, ctx, level)
        return prune(ty)
    catch err
        if err isa TyError
            return err
        else
            rethrow(err)
        end
    end
end

function _infer(node::AstNode, env::Env, ctx::Dict{Symbol,TyConst}, level::Int)::Ty
    if node isa ConstNode
        ty = ctx[node.name]  # must exist
        return ty
    elseif node isa VarNode
        sch = lookup(env, node.name)
        sch === nothing && throw(UnboundVariableError(node.name))
        return instantiate(sch, level)
    elseif node isa IndexNode
        sch = lookup(env, node.index)
        return instantiate(sch, level)
    elseif node isa AbsNode
        tv = fresh_tyvar(level+1)
        push!(env, nothing, TyScheme(TyVar[], tv))
        body_ty = _infer(node.body, env, ctx, level+1)
        pop!(env)
        return TyFun(prune(tv), body_ty)
    elseif node isa AppNode
        fun_ty = _infer(node.func, env, ctx, level)
        arg_ty = _infer(node.arg, env, ctx, level)
        res_ty = fresh_tyvar(level)
        unify!(fun_ty, TyFun(arg_ty, res_ty))
        # automatic generalisation: if res_ty has vars deeper than current level
        sch = generalize(res_ty, level)
        return instantiate(sch, level)
    elseif node isa LetNode
        val_ty = _infer(node.value, env, ctx, level+1)
        sch = generalize(val_ty, level)
        push!(env, node.name, sch)
        body_ty = _infer(node.body, env, ctx, level)
        pop!(env)
        return body_ty
    else
        error("Unknown AST node type $(typeof(node))")
    end
end

# ------------------------------------------------------------
# Pretty printing for types (debugging)
# ------------------------------------------------------------
import Base.show
function _show_ty(t::Ty, seen=IdDict{TyVar,Int}())
    if t isa TyConst
        return String(t.name)
    elseif t isa TyVar
        if t.instance !== nothing
            return _show_ty(prune(t), seen)
        else
            idx = get!(seen, t, length(seen))
            return "α$idx"
        end
    elseif t isa TyFun
        return _show_ty(t.arg, seen) * " → " * _show_ty(t.res, seen)
    elseif t isa TyMu
        return "μ." * _show_ty(t.body, seen)
    else
        return "<?>"
    end
end
show(io::IO, t::Ty) = print(io, _show_ty(t))

# ------------------------------------------------------------
# Test items (basic sanity)
# ------------------------------------------------------------

# using TestItems
# using ..ASTs.NodeASTs

@testitem "simple identity typing" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    # term: (λx. x)
    term = AbsNode(:x, IndexNode(0))
    tree = AstTree(term)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)
    @test ty isa TyFun
end

@testitem "occurs-check negative" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    # ill-typed: λx. x x  (would require recursive type in negative position)
    body = AppNode(IndexNode(0), IndexNode(0))
    term = AbsNode(:x, body)
    tree = AstTree(term)
    ctx = Dict{Symbol,TyConst}()
    result = infer(tree, ctx)
    println("Debug: result = ", result, " (type: ", typeof(result), ")")
    @test result isa OccursCheckError || result isa MismatchError
end

end # module TypeSystem