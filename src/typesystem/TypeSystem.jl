module TypeSystem

using ..ASTs.NodeASTs: AstTree, AstNode, VarNode, IndexNode, AbsNode, AppNode, ConstNode
using ..Utils
using ..ASTs.Contexts: COND
using TestItems

export Ty, TyConst, TyVar, TyFun, TyMu, TyProd, TySum, TyList, TyNat, TyArray, TyScheme, TyLike
export infer, UnboundVariableError, OccursCheckError, MismatchError
export →, ×, μ, ∀, newtyvar, collect_types, get_type

public BoolTy, IntTy, FloatTy, COND

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
• `instance` – if unified, points to another `Ty` (acts like union-find parent), work as the substitution rule in basic Hindley-Milner type inference.
"""
mutable struct TyVar <: Ty
    id::Int
    level::Int
    instance::Union{Ty, Nothing}
end
copy_detached(t::TyVar) = TyVar(t.id, t.level, nothing)

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
    TyProd(left, right)
Product type  `left × right`.
"""
struct TyProd <: Ty
    left::Ty
    right::Ty
end

"""
    TySum(left, right)
Sum type `left + right` (disjoint union).
"""
struct TySum <: Ty
    left::Ty
    right::Ty
end

"""
    TyList(elem)
Covariant list type `List elem`.
"""
struct TyList <: Ty
    elem::Ty
end

"""
    TyNat(value)
Type-level natural number constant used for fixed-size arrays.
"""
struct TyNat <: Ty
    value::Int
end

"""
    TyArray(len, elem)
Fixed-length array type  `Array[len, elem]`, where `len` is a `TyNat` and `elem` is element type.
"""
struct TyArray <: Ty
    len::TyNat
    elem::Ty
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

Base.hash(t::TyVar, h::UInt) = hash(t.id, hash(t.level, (hash(:TyVar, h))))
Base.hash(t::TyFun, h::UInt) = hash(t.arg, hash(t.res, (hash(:TyFun, h))))
Base.hash(t::TyProd, h::UInt) = hash(t.left, hash(t.right, (hash(:TyProd, h))))
Base.hash(t::TySum, h::UInt) = hash(t.left, hash(t.right, (hash(:TySum, h))))
Base.hash(t::TyMu, h::UInt) = hash(t.var, hash(t.body, (hash(:TyMu, h))))
Base.hash(t::TyScheme, h::UInt) = hash(t.vars, hash(t.ty, (hash(:TyScheme, h))))
Base.hash(t::TyNat, h::UInt) = hash(t.value, hash(:TyNat, h))
Base.hash(t::TyArray, h::UInt) = hash(t.len, hash(t.elem, hash(:TyArray, h)))

Base.:(==)(t1::TyVar, t2::TyVar) = t1.id == t2.id && t1.level == t2.level
Base.:(==)(t1::TyFun, t2::TyFun) = t1.arg == t2.arg && t1.res == t2.res
Base.:(==)(t1::TyProd, t2::TyProd) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TySum, t2::TySum) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TyMu, t2::TyMu) = t1.var == t2.var && t1.body == t2.body
Base.:(==)(t1::TyScheme, t2::TyScheme) = t1.vars == t2.vars && t1.ty == t2.ty
Base.:(==)(t1::TyNat, t2::TyNat) = t1.value == t2.value
Base.:(==)(t1::TyArray, t2::TyArray) = t1.len == t2.len && t1.elem == t2.elem

# ------------------------------------------------------------
# Error reporting
# ------------------------------------------------------------
struct UnboundVariableError
    name::Symbol
end

struct OccursCheckError
    var::TyVar
    ty::Ty
end

struct MismatchError
    t1::Ty
    t2::Ty
end

# ------------------------------------------------------------
# Internal fresh id counter
# ------------------------------------------------------------
mutable struct Counter
    value::Int
end
Counter() = Counter(0)
(c::Counter)() = c.value += 1

const _TYVAR_COUNTER = Counter()

struct VarFactory
    counter::Counter
end
VarFactory() = VarFactory(_TYVAR_COUNTER)
(n::VarFactory)(level::Int)::TyVar = TyVar(n.counter(), level, nothing)

const newtyvar = VarFactory() # aliasing for compatibility with old code

# ------------------------------------------------------------
# Utility: prune – follow instances to canonical representative
# ------------------------------------------------------------
function prune(t::Ty)::Ty
    seen = Set{Ty}()
    function _prune(t::Ty)::Ty
        t in seen && return t  # already visited – stop to avoid cycles
        push!(seen, t)
    
        if t isa TyVar && t.instance !== nothing
            instance = _prune(t.instance)
            return instance
        elseif t isa TyFun
            return TyFun(_prune(t.arg), _prune(t.res))
        elseif t isa TyMu
            return TyMu(t.var, _prune(t.body))
        elseif t isa TyProd
            return TyProd(_prune(t.left), _prune(t.right))
        elseif t isa TySum
            return TySum(_prune(t.left), _prune(t.right))
        elseif t isa TyArray
            return TyArray(t.len, _prune(t.elem))
        elseif t isa TyNat
            return t
        elseif t isa TyList
            return TyList(_prune(t.elem))
        else
            return t
        end
    end
    _prune(t)
end

# ------------------------------------------------------------
# Occurs check (contractiveness only)
# ------------------------------------------------------------
# return (occurs::Bool, contractive::Bool)
"""
    occurs(var::TyVar, ty::Ty)::Tuple{Bool,Bool}

Detect whether `var` occurs inside `ty` and whether **at least one** such occurrence
is *contractive* (i.e. appears strictly under a type constructor).  Contractiveness
is required to safely fold the recursion into a finite `μ`-type.  A naked self
reference like `μ α . α` would be *non-contractive* and therefore forbidden – it
would otherwise create an infinite, unproductive cycle.
"""
function occurs(var::TyVar, ty::Ty, depth::Int=0)::Tuple{Bool,Bool}
    ty = prune(ty)
    if ty isa TyVar
        if ty === var
            return (true, depth > 0)   # contractive only if nested (depth ≥ 1)
        else
            return (false, false)
        end
    elseif ty isa TyConst
        return (false, false)
    elseif ty isa TyFun
        occ1, con1 = occurs(var, ty.arg, depth + 1)
        occ2, con2 = occurs(var, ty.res, depth + 1)
        return (occ1 || occ2, con1 || con2)
    elseif ty isa TyProd
        occ1, con1 = occurs(var, ty.left, depth + 1)
        occ2, con2 = occurs(var, ty.right, depth + 1)
        return (occ1 || occ2, con1 || con2)
    elseif ty isa TySum
        occ1, con1 = occurs(var, ty.left, depth + 1)
        occ2, con2 = occurs(var, ty.right, depth + 1)
        return (occ1 || occ2, con1 || con2)
    elseif ty isa TyArray
        return occurs(var, ty.elem, depth + 1)
    elseif ty isa TyNat
        return (false, false)
    elseif ty isa TyMu
        # Bound var shadows outer one
        if ty.var === var
            return (false, false)
        else
            return occurs(var, ty.body, depth)
        end
    elseif ty isa TyList
        return occurs(var, ty.elem, depth)
    else
        return (false, false)
    end
end

# ------------------------------------------------------------
# Binding TyVar with contractiveness & μ-fold
# ------------------------------------------------------------
function _bind!(var::TyVar, ty::Ty)
    occ, contractive = occurs(var, ty, 0)
    if !occ
        var.instance = ty
        return nothing
    elseif !contractive
        # Non-contractive self-reference (e.g. α = α) is illegal.
        return Err(OccursCheckError(var, ty))
    else
        # Safe recursive occurrence – fold into μ-type.
        folded = TyMu(var, ty)
        var.instance = folded
        return nothing
    end
end

# ------------------------------------------------------------
# Unification (equi-recursive, memoised)
# ------------------------------------------------------------
function unify!(t1::Ty, t2::Ty)
    seen = Set{Tuple{Ty,Ty}}()
    return _unify!(t1, t2, seen)
end

function _unify!(t1::Ty, t2::Ty, seen::Set{Tuple{Ty,Ty}})
    a = prune(t1)
    b = prune(t2)
    if a === b
        return nothing
    end

    # Symmetric memoisation: (a,b) and (b,a) considered equivalent
    if (a, b) in seen || (b, a) in seen
        return nothing
    end
    push!(seen, (a, b))

    if a isa TyVar
        return _bind!(a, b)
    elseif b isa TyVar
        return _bind!(b, a)
    elseif a isa TyConst && b isa TyConst && a.name == b.name
        return nothing
    elseif a isa TyFun && b isa TyFun
        @✓ _unify!(a.arg, b.arg, seen)
        @✓ _unify!(a.res, b.res, seen)
    elseif a isa TyProd && b isa TyProd
        @✓ _unify!(a.left, b.left, seen)
        @✓ _unify!(a.right, b.right, seen)
    elseif a isa TySum && b isa TySum
        @✓ _unify!(a.left, b.left, seen)
        @✓ _unify!(a.right, b.right, seen)
    elseif a isa TyArray && b isa TyArray
        if a.len.value != b.len.value
            return Err(MismatchError(a, b))
        end
        @✓ _unify!(a.elem, b.elem, seen)
    elseif a isa TyNat && b isa TyNat && a.value == b.value
        return nothing
    elseif a isa TyMu
        @✓ _unify!(a.body, b, seen)
    elseif b isa TyMu
        @✓ _unify!(a, b.body, seen)
    elseif a isa TyList && b isa TyList
        @✓ _unify!(a.elem, b.elem, seen)
    else
        return Err(MismatchError(a, b))
    end
    return nothing
end

# ------------------------------------------------------------
# Instantiate & Generalise
# ------------------------------------------------------------
function instantiate(s::TyScheme, level::Int)::Ty
    # Create substitution only for quantified vars
    subst = Dict{Int,TyVar}()
    for v in s.vars
        subst[v.id] = newtyvar(level)
    end
    return _inst(s.ty, v -> get(subst, v.id, v))
end

instantiate(t::Ty, ::Int) = t # for concrete types, do nothing

function _inst(t::Ty, mapvar)
    if t isa TyVar
        return mapvar(t)
    elseif t isa TyConst
        return t
    elseif t isa TyFun
        return TyFun(_inst(t.arg, mapvar), _inst(t.res, mapvar))
    elseif t isa TyProd
        return TyProd(_inst(t.left, mapvar), _inst(t.right, mapvar))
    elseif t isa TySum
        return TySum(_inst(t.left, mapvar), _inst(t.right, mapvar))
    elseif t isa TyMu
        return TyMu(t.var, _inst(t.body, mapvar))
    elseif t isa TyList
        return TyList(_inst(t.elem, mapvar))
    elseif t isa TyArray
        return TyArray(t.len, _inst(t.elem, mapvar))
    else
        return t
    end
end

function collect_free_vars(t::Ty)::Set{TyVar}
    vars = Set{TyVar}()
    seen = Set{Ty}()
    t = prune(t)

    function _collect_free_vars(t::Ty)::Set{TyVar}
        t in seen && return vars
        push!(seen, t)
        if t isa TyVar
            push!(vars, t)
        elseif t isa TyFun
            _collect_free_vars(t.arg)
            _collect_free_vars(t.res)
        elseif t isa TyProd
            _collect_free_vars(t.left)
            _collect_free_vars(t.right)
        elseif t isa TySum
            _collect_free_vars(t.left)
            _collect_free_vars(t.right)
        elseif t isa TyMu
            _collect_free_vars(t.body)
        elseif t isa TyList
            _collect_free_vars(t.elem)
        elseif t isa TyArray
            _collect_free_vars(t.elem)
        end
        return vars
    end

    return _collect_free_vars(t)
end 

function generalize(t::Ty, level::Int)::TyScheme
    seen = collect_free_vars(t)
    vars = [v for v in seen if v.level > level]
    return TyScheme(vars, t)
end

# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------
struct Env
    stack::Vector{Pair{Symbol,TyScheme}}
end
Env() = Env(Pair{Symbol,TyScheme}[])

Base.push!(env::Env, name::Symbol, scheme::TyScheme) = push!(env.stack, name => scheme)
Base.pop!(env::Env) = pop!(env.stack) 

lookup(env::Env, idx::Int)::TyScheme = env.stack[end - idx].second

function lookup(env::Env, name::Symbol)::TyScheme
    for (n, sch) in env.stack
        if n == name
            return sch
        end
    end
    return Err(UnboundVariableError(name))
end

# ------------------------------------------------------------
# Main inference (returns Ty or throws TyError)
# ------------------------------------------------------------

# Constants in context may be monotype (Ty) or polymorphic TyScheme
const TyLike = Union{Ty,TyScheme}

"""
    infer(tree::AstTree, ctx::Dict{Symbol,<:TyLike}[, factory=newtyvar])

Infer the type given the AST and context.

# Arguments
- `tree::AstTree`: The AST to infer the type of.
- `ctx::Dict{Symbol,<:TyLike}`: The context of the inference.
- `factory::Function`: A function to create fresh type variables.
"""
function infer(tree::AstTree, ctx::Dict{Symbol,<:TyLike}, factory=newtyvar)
    env = Env()
    level = 0
    ty = @✓(_infer(tree.root, env, ctx, level, factory))
    return prune(ty)
end

function _infer(node::AstNode, env::Env, ctx::Dict{Symbol,<:TyLike}, level::Int, factory)
    if node isa ConstNode
        return instantiate(ctx[node.name], level)  # do nothing for concrete types, instantiate for polymorphic types
    elseif node isa VarNode
        sch = @✓ lookup(env, node.name)
        return instantiate(sch, level)
    elseif node isa IndexNode
        sch = lookup(env, node.index) # Unbound De Bruijn index should be handled in advance. Panic if not.
        return instantiate(sch, level)
    elseif node isa AbsNode
        tv = factory(level+1)
        push!(env, node.name, TyScheme(TyVar[], tv))
        body_ty = @✓ _infer(node.body, env, ctx, level+1, factory)
        pop!(env)
        return TyFun(prune(tv), body_ty)
    elseif node isa AppNode
        arg_ty = @✓ _infer(node.arg, env, ctx, level, factory)
        if node.func isa AbsNode # form like (λx. M) N, treated as let x = N in M, generalize the result type
            arg_ty = generalize(arg_ty, level)
            push!(env, node.func.name, arg_ty)
            body_ty = @✓ _infer(node.func.body, env, ctx, level+1, factory)
            pop!(env)
            return body_ty
        else
            fun_ty = @✓ _infer(node.func, env, ctx, level, factory)
            res_ty = factory(level)
            @✓ unify!(fun_ty, TyFun(arg_ty, res_ty))
            return res_ty
        end
    else
        error("Unknown AST node type $(typeof(node))")
    end
end

# ------------------------------------------------------------
# Pretty printing for types (debugging)
# ------------------------------------------------------------

function _show_ty(t::Ty, seen=Dict{TyVar,Int}())
    if t isa TyConst
        return String(t.name)
    elseif t isa TyVar
        idx = get!(seen, t, length(seen))
        return "α$(t.id)"
    elseif t isa TyFun
        return _show_ty(t.arg, seen) * " → " * _show_ty(t.res, seen)
    elseif t isa TyProd
        return "(" * _show_ty(t.left, seen) * " × " * _show_ty(t.right, seen) * ")"
    elseif t isa TySum
        return "(" * _show_ty(t.left, seen) * " + " * _show_ty(t.right, seen) * ")"
    elseif t isa TyMu
        return "μ." * _show_ty(t.body, seen)
    elseif t isa TyList
        return "List[" * _show_ty(t.elem, seen) * "]"
    elseif t isa TyNat
        return string(t.value)
    elseif t isa TyArray
        return "Array[" * _show_ty(t.len, seen) * "," * _show_ty(t.elem, seen) * "]"
    else
        return "<?>"
    end
end
Base.show(io::IO, t::Ty) = print(io, _show_ty(t))

# ------------------------------------------------------------
# Syntactic sugar: infix operators for quick type construction
# ------------------------------------------------------------
# To make type expressions more readable, we define a few infix
# operators that construct our composite types directly. These
# only apply when both operands are `Ty` (or sub-types thereof),
# so they do not interfere with normal arithmetic on numbers.

"""Construct `TyFun(arg, res)`."""
(→)(arg::Ty, res::Ty)::TyFun = TyFun(arg, res)

"""Construct `TyProd(left, right)`."""
×(left::Ty, right::Ty)::TyProd = TyProd(left, right)

"""Construct `TySum(left, right)`."""
Base.:+(left::Ty, right::Ty)::TySum = TySum(left, right)

# Additional syntactic sugar for TyMu and TyScheme
"""Construct `TyMu(var, body)`."""
μ(var::TyVar, body::Ty)::TyMu = TyMu(var, body)

"""Construct `TyScheme(vars, ty)`."""
∀(vars::Vector{TyVar}, ty::Ty)::TyScheme = TyScheme(vars, ty)

# ------------------------------------------------------------
# Context validation helper
# ------------------------------------------------------------

"""
    validate_ctx(ctx::Dict{Symbol,TyLike})

Check that `ctx` satisfies:
1. Any `Ty` value contains no free `TyVar`
2. For `TyScheme`, all `TyVar` in `ty` must appear in `vars` list

Throws an error if validation fails.
"""
function validate_ctx(ctx::Dict{Symbol,TyLike})
    for (name, entry) in ctx
        if entry isa TyConst
            continue                    # OK
        elseif entry isa Ty
            free = collect_free_vars(entry)
            if !isempty(free)
                error("Context constant :$name contains free type variables")
            end
        elseif entry isa TyScheme
            free = collect_free_vars(entry.ty)
            unbound = filter(v -> v ∉ entry.vars, free)
            if !isempty(unbound)
                error("Context constant :$name has unquantified type variables in scheme")
            end
        end
    end
end

# ------------------------------------------------------------
# Typed term
# ------------------------------------------------------------
get_type(constant) = nothing

function collect_types(constants_ctx::Dict{Symbol, Any}, types_ctx::Dict{Symbol, Ty}=Dict{Symbol, Ty}())::Dict{Symbol, Ty}
    for (key, value) in pairs(constants_ctx)
        if haskey(types_ctx, key)
            continue
        end
        result = get_type(value)
        if !isnothing(result)
            types_ctx[key] = result
        else
            warning("No type set for constant $key, please specify it manually.")
        end
    end
    return types_ctx
end

# ------------------------------------------------------------
# Primitive base types and Prelude constant typing
# ------------------------------------------------------------

# Primitive value-type constants
const BoolTy  = TyConst(:Bool)
const IntTy   = TyConst(:Int)
const FloatTy = TyConst(:Float)
const NilErr  = TyConst(:NilErr)

# Sum type basic functions
INL(x) = x
INR(x) = x

# Product type basic functions
PAIR(x, y) = (x, y)
FST(x) = x.first
SND(x) = x.second

# List type basic functions
NIL() = []
struct _NilErr end
const NILERR = _NilErr()
CONS(x) = l -> [x;l]
HEAD(xs) = isempty(xs) ? NILERR : xs[1]
TAIL(xs) = isempty(xs) ? NILERR : xs[2:end]
LENGTH(xs) = length(xs)
CASENILERR(f) = g -> (x ->(x isa _NilErr ? g(x) : f(x)))
TOLIST(xs) = collect(xs)

_α, _β, _γ = newtyvar(0), newtyvar(0), newtyvar(0)

# Fallbacks for primitive constants (already defined earlier)
get_type(::Bool)  = BoolTy
get_type(::Int)   = IntTy
get_type(::Float) = FloatTy

get_type(::typeof(COND)) = ∀([_α], BoolTy → _α → _α → _α)

get_type(::typeof(INL)) = ∀([_α, _β], _α → (_α + _β))
get_type(::typeof(INR)) = ∀([_α, _β], _β → (_α + _β))
# Just an example. As inexchangable sum type is not supported in Julia, case function must be implemented manually for each sum type.
# get_type(::typeof(CASE)) = ∀([_α, _β, _γ], (_α → _γ) → (_β → _γ) → (_α + _β) → _γ)

get_type(::typeof(PAIR)) = ∀([_α, _β], _α → _β → (_α × _β))
get_type(::typeof(FST)) = ∀([_α, _β], (_α × _β) → _α)
get_type(::typeof(SND)) = ∀([_α, _β], (_α × _β) → _β)
get_type(t::Tuple{T, S}) where {T, S} = get_type(t[1]) × get_type(t[2])

get_type(::typeof(NIL)) = ∀([_α], TyList(_α))
get_type(::typeof(CONS)) = ∀([_α], _α → TyList(_α) → TyList(_α))
get_type(::typeof(HEAD)) = ∀([_α], TyList(_α) → (_α + _NilErr))
get_type(::typeof(TAIL)) = ∀([_α], TyList(_α) → TyList(_α) + _NilErr)
get_type(::typeof(LENGTH)) = ∀([_α], TyList(_α) → IntTy)
get_type(::typeof(CASENILERR)) = ∀([_α, _β], (_α → _β) → (NilErr → _β) → _β)
get_type(::typeof(TOLIST)) = ∀([_α, _β], TyArray(_α, _β) → TyList(_β))

# ------------------------------------------------------------
# Test items (basic sanity)
# ------------------------------------------------------------
@testitem "infix operators" begin
    using LambdaRegression.TypeSystem
    
    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    fun_ty = IntTy → BoolTy
    @test fun_ty isa TyFun && fun_ty.arg === IntTy && fun_ty.res === BoolTy
    prod_ty = IntTy × BoolTy
    @test prod_ty isa TyProd && prod_ty.left === IntTy && prod_ty.right === BoolTy
    sum_ty = IntTy + BoolTy
    @test sum_ty isa TySum && sum_ty.left === IntTy && sum_ty.right === BoolTy
end

@testitem "simple identity typing" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    # term: (λx. x)
    term = AbsNode(:x, IndexNode(0))
    tree = AstTree(term)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)
    @test ty isa TyFun && ty.arg === ty.res
end

@testitem "self-application typable via μ-type" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    # term: λx. x x  (needs recursive type)
    body = AppNode(IndexNode(0), IndexNode(0))
    term = AbsNode(:x, body)
    tree = AstTree(term)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)
    @test ty isa TyFun
    # argument type should be a recursive μ-type whose body is a function
    @test ty.arg isa TyMu && ty.arg.body isa TyFun
end

@testitem "application polymorphism (λx.x) id" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: TyLike

    id_fun = AbsNode(:x, IndexNode(0))         # λx. x

    app_expr = AppNode(id_fun, id_fun)         # (λx.x) (λx.x)
    tree = AstTree(app_expr)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)

    # Expected: type of argument, i.e., σ → σ where σ fresh
    @test ty isa TyFun && ty.arg === ty.res
end

@testitem "positive μ-type via unify" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: newtyvar, unify!

    α = newtyvar(0)
    int_ty = TyConst(:Int)
    unify!(α, TyFun(int_ty, α))
    # α should now be folded to μ-type
    @test α.instance isa TyMu
end

@testitem "factorial via Y_int" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    ctx = Dict{Symbol,TyLike}(
        :zero   => IntTy,
        :one    => IntTy,
        :iszero => IntTy → BoolTy,
        :pred   => IntTy → IntTy,
        :mul    => IntTy → IntTy → IntTy,
        :if_int => BoolTy → IntTy → IntTy → IntTy,
        :Y_int  => ((IntTy → IntTy) → (IntTy → IntTy)) → (IntTy → IntTy)
    )

    # Build λf. λn. if_int (iszero n) one (mul n (f (pred n)))
    var_f = IndexNode(1)
    var_n = IndexNode(0)
    iszero_n = AppNode(ConstNode(:iszero), var_n)
    pred_n = AppNode(ConstNode(:pred), var_n)
    fn_pred = AppNode(var_f, pred_n)
    mul_expr = AppNode(AppNode(ConstNode(:mul), var_n), fn_pred)
    if_expr = AppNode(AppNode(AppNode(ConstNode(:if_int), iszero_n), ConstNode(:one)), mul_expr)
    f_lambda = AbsNode(:f, AbsNode(:n, if_expr))
    y_app = AppNode(ConstNode(:Y_int), f_lambda)
    tree = AstTree(y_app)

    ty = infer(tree, ctx)
    @test ty isa TyFun && ty.arg === IntTy && ty.res === IntTy
end

@testitem "Linked TyVar instances" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: collect_free_vars, prune

    α = newtyvar(0)
    β = newtyvar(0)
    mu = μ(α, β→α)
    α.instance = mu

    a = prune(α)

    vs = collect_free_vars(α)
end

@testitem "Y combinator" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: prepare
    using LambdaRegression.ASTs.Contexts
    using LambdaRegression.TypeSystem: newtyvar, TyLike

    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    α = newtyvar(0)

    # Define basic arithmetic and logical operations for our lambda calculus
    add = x -> y -> x + y  # Curried addition
    mul = x -> y -> x * y  # Curried multiplication  
    sub = x -> y -> x - y  # Curried subtraction
    eq = x -> y -> x == y  # Curried equality
    lt = x -> y -> x < y   # Curried less than
    cond = p -> p ? Contexts.T : Contexts.F # boolean to boolean combinator

    int1 = 1
    int0 = 0
    int4 = 4
    booltrue = true
    boolfalse = false

    Y, _ = @lambda "λf.(λx.f (x x)) (λx.f (x x))"
    fac, _ = @lambda "Y (λfac.λn. cond (eq n int0) int1 (mul n (fac (sub n int1))))"
    fac4, _ = @lambda "fac int4"
    fac4 = prepare(fac4)

    tyctx = Dict{Symbol,TyLike}(
        :add => IntTy → IntTy → IntTy,
        :mul => IntTy → IntTy → IntTy,
        :sub => IntTy → IntTy → IntTy,
        :eq => IntTy → IntTy → BoolTy,
        :lt => IntTy → IntTy → BoolTy,
        :cond => ∀([α], BoolTy → α → α → α),
        :int0 => IntTy,
        :int1 => IntTy,
        :int4 => IntTy,
    )
    ty = infer(fac4, tyctx)
    @test ty == IntTy
end

@testitem "polymorphic if constant" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem

    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)

    # if2 : ∀α. Bool → α → α → α
    α = newtyvar(0)
    ctx = Dict{Symbol,TyLike}(
        :if2 => ∀([α], BoolTy → α → α → α), 
        :boolTrue => BoolTy, 
        :int1 => IntTy, 
        :int2 => IntTy
    )

    # expression: if2 boolTrue int1 int2  ==> Int
    expr = AppNode(AppNode(AppNode(ConstNode(:if2), ConstNode(:boolTrue)), ConstNode(:int1)), ConstNode(:int2))
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === IntTy
end

# -------------------- Product & Sum basic tests --------------------

@testitem "product unify and projection" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: newtyvar

    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)

    # fst : ∀α β. (α × β) → α
    α = newtyvar(0)
    β = newtyvar(0)
    ctx = Dict{Symbol,TyLike}(
        :fst => ∀([α,β], (α × β) → α), 
        :pair => ∀([α,β], α → β → (α × β)),
        :three => IntTy,
        :btrue => BoolTy
    )

    pair_expr = AppNode(AppNode(ConstNode(:pair), ConstNode(:three)), ConstNode(:btrue))
    expr = AppNode(ConstNode(:fst), pair_expr)
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === IntTy
end

@testitem "sum inl inr case" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: newtyvar

    IntTy = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    α, β, γ = newtyvar(0), newtyvar(0), newtyvar(0)
    ctx = Dict{Symbol,TyLike}(
        :inl=>∀([α,β], α → (α + β)), 
        :inr=>∀([α,β], β → (α + β)), 
        :case=>∀([α,β,γ], (α → γ) → (β → γ) → (α + β → γ)),
        :int3 => IntTy,
        :booltrue => BoolTy
    )
    left_branch = AbsNode(:x, ConstNode(:booltrue)) # returns Bool
    right_branch = AbsNode(:y, ConstNode(:booltrue))
    sum_val = AppNode(ConstNode(:inl), ConstNode(:int3))
    expr = AppNode(AppNode(AppNode(ConstNode(:case), left_branch), right_branch), sum_val)
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === BoolTy
end

@testitem "array fixed length typing" begin
    using LambdaRegression.TypeSystem
    IntTy = TyConst(:Int)
    arr5_ty = TyArray(TyNat(5), IntTy)
    # toList5 : Array[5,Int] -> List[Int]
    tolist_scheme = TyScheme([], TyFun(arr5_ty, TyList(IntTy)))
    ctx = Dict{Symbol,TyLike}(:arr5 => arr5_ty, :toList5 => tolist_scheme)
    expr = AppNode(ConstNode(:toList5), ConstNode(:arr5))
    tree = LambdaRegression.ASTs.NodeASTs.AstTree(expr)
    ty = infer(tree, ctx)
    @test ty == TyList(IntTy)
end

# ------------------------------------------------------------
# Advanced type invariance tests --------------------

@testitem "factorial via Y combinator – type invariance under reduction" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: prepare, beta_reduce!
    using LambdaRegression.ASTs.Contexts
    using LambdaRegression.TypeSystem: newtyvar, TyLike

    IntTy  = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    α = newtyvar(0)

    # Deterministic integer identifiers to avoid random gensym names
    int0, int1, int4 = 0, 1, 4
    add  = x -> y -> x + y
    mul  = x -> y -> x * y
    sub  = x -> y -> x - y
    eq   = x -> y -> x == y
    cond = p -> p ? Contexts.T : Contexts.F

    ctx = Dict{Symbol,Any}()
    Y, _   = @lambda "λf.(λx.f (x x)) (λx.f (x x))"
    fac, _ = @lambda "Y (λfac.λn. cond (eq n int0) int1 (mul n (fac (sub n int1))))" ctx
    expr, _ = @lambda "fac int4" ctx
    expr = prepare(expr)

    # Initial type context
    tyctx = Dict{Symbol,TyLike}(
        :add   => IntTy → IntTy → IntTy,
        :mul   => IntTy → IntTy → IntTy,
        :sub   => IntTy → IntTy → IntTy,
        :eq    => IntTy → IntTy → BoolTy,
        :cond  => ∀([α], BoolTy → α → α → α),
        :int0  => IntTy,
        :int1  => IntTy,
        :int4  => IntTy,
    )

    # Type before reduction
    @test infer(expr, tyctx) == IntTy
    beta_reduce!(expr, 20_000, ctx)
    value = ctx[expr.root.name]
    @test value == 24
end

@testitem "list square-sum – type invariance under reduction" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: prepare, beta_reduce!
    using LambdaRegression.ASTs.Contexts
    using LambdaRegression.TypeSystem: newtyvar, TyLike

    IntTy  = TyConst(:Int)
    BoolTy = TyConst(:Bool)
    α, β = newtyvar(0), newtyvar(0)

    # List primitives implemented in Julia (used only for evaluation)
    cons   = x -> xs -> [x, xs...]
    car    = xs -> length(xs) > 0 ? xs[1] : 0
    cdr    = xs -> length(xs) > 1 ? xs[2:end] : Any[]
    nil    = Any[]
    add    = x -> y -> x + y
    mul    = x -> y -> x * y
    square = x -> x * x
    cond   = p -> p ? Contexts.T : Contexts.F
    null   = xs -> isempty(xs)

    test_list = [1,2,3,4,5]
    ctx = Dict{Symbol,Any}()
    Y, _ = @lambda "λf.(λx.f (x x)) (λx.f (x x))"
    appl, _ = @lambda "Y (λappl.λf.λlist. cond (null list) nil ((cons (f (car list))) (appl f (cdr list))))" ctx
    apply_square_tree, _ = @lambda "appl square test_list" ctx
    apply_square_tree = prepare(apply_square_tree)

    SUM, _ = @lambda "Y (λsum.λlist. cond (null list) 0 (add (car list) (sum (cdr list))))" ctx # Note that `sum` is bound
    sum_of_squares, _ = @lambda "λlist. SUM (appl square list)" ctx
    result_tree, _ = @lambda "sum_of_squares test_list" ctx
    result_tree = prepare(result_tree)

    # Type context covering list primitives & arithmetic
    tyctx = Dict{Symbol,TyLike}(
        :cons  => ∀([β], β → TyList(β) → TyList(β)),
        :car   => ∀([β], TyList(β) → β),
        :cdr   => ∀([β], TyList(β) → TyList(β)),
        :nil   => ∀([β], TyList(β)),
        :null  => ∀([β], TyList(β) → BoolTy),
        :add   => IntTy → IntTy → IntTy,
        :mul   => IntTy → IntTy → IntTy,
        :square=> IntTy → IntTy,
        :cond  => ∀([β], BoolTy → β → β → β),
        :test_list => TyList(IntTy),
        Symbol(0) => IntTy,
    )

    # ----- `appl square test_list` -----
    @test infer(apply_square_tree, tyctx) == TyList(IntTy)
    beta_reduce!(apply_square_tree, 20_000, ctx)
    list_val = ctx[apply_square_tree.root.name]
    @test list_val == [1,4,9,16,25]

    # ----- `sum_of_squares test_list` -----
    @test infer(result_tree, tyctx) == IntTy
    beta_reduce!(result_tree, 20_000, ctx)
    sum_val = ctx[result_tree.root.name]
    @test sum_val == 55
end

end # module TypeSystem