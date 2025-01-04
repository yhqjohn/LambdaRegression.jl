"""
    TypeSystem

`TypeSystem` implements the inference machinery that underpins our lambda-calculus playground.  
We target a Turing-complete calculus with recursion, sums, products, lists, and fixed-length arrays.  
Every term that can be typed will be given a type within finite time (or a precise error), yet we make no attempt to force strong normalisation: a well-typed looping term will keep looping, and terms that do reduce share the same type before and after β-reduction.  
This balance lets us analyse rich programs without over-constraining their operational behaviour.

To keep the mental model close to standard Hindley-Milner, we hard-code the bridge to native data: integers, booleans, natural numbers, arrays and lists surface as predefined type constructors instead of deepening the theory.  
The engine therefore stays approachable—think of it as a tutorial-sized HM variant that knows how to fold recursive shapes into `TyMu` and how to preserve let-polymorphism—while still remaining expressive enough to describe real evaluation environments.

## Core Characteristics
- Equi-recursive Hindley-Milner inference with contractive occurs checking: safe self references become `TyMu`.
- Deterministic termination of the inference algorithm: every AST yields a type or an `Err` in finite time.
- β-reduction invariance for normalising terms: the inferred type does not change when a typable term reduces.
- Practical primitive coverage without expanding the theoretical core: see [`TyConst`](@ref), [`TyList`](@ref), [`TyArray`](@ref), [`TyNat`](@ref).

## Key Types
- [`Ty`](@ref): abstract supertype of all internal type representations.
- [`TyConst`](@ref), [`TyVar`](@ref), [`TyFun`](@ref), [`TyProd`](@ref), [`TySum`](@ref), [`TyList`](@ref), [`TyArray`](@ref), [`TyMu`](@ref): concrete building blocks.
- [`TyScheme`](@ref): polymorphic type schemes with α-equivalent hashing via `_canonical_body`.
- [`TyLike`](@ref): convenience alias `Union{Ty,TyScheme}` used throughout contexts and metadata.
- [`Counter`](@ref) and [`VarFactory`](@ref): fresh-variable generation utilities (`newtyvar` exposes the shared factory).

## Fundamental Operations
- [`prune`](@ref): follows `TyVar` instances to canonical representatives.
- [`occurs`](@ref) & [`_bind!`](@ref): contractiveness-aware occurs check and binding helper.
- [`unify!`](@ref): equi-recursive unification with memoisation to avoid infinite folding.
- [`instantiate`](@ref) / [`generalize`](@ref): Hindley-Milner instantiation discipline for schemes and let-polymorphism.
- [`collect_free_vars`](@ref): helper for rigid-variable tracking.
- [`infer`](@ref): walks an [`AstTree`](@ref LambdaRegression.ASTs.NodeASTs.AstTree) with environments, optionally recording lower/upper type bounds per node.

## Primitive Library
- Predefined constants (see [`BoolTy`](@ref TypeSystem.BoolTy), [`IntTy`](@ref TypeSystem.IntTy), [`FloatTy`](@ref TypeSystem.FloatTy), [`TyList`](@ref), [`TyArray`](@ref), [`TyNat`](@ref)) mirror common data structures without altering inference rules.
- `get_type` specialisations attach the right [`TyScheme`](@ref) or [`Ty`](@ref) to Julia values such as arithmetic combinators or list helpers.
- Infix constructors (see [`→`](@ref TypeSystem.:→), [`×`](@ref TypeSystem.:×), [`+`](@ref TypeSystem.:+), [`μ`](@ref TypeSystem.:μ), [`∀`](@ref TypeSystem.:∀)) offer concise syntax when authoring types by hand or within tests.

## Auxiliary Tools
- [`iscompatible`](@ref) and [`islesseq`](@ref) check compatibility and semantic subset relations between types or schemes.
- [`get_type_ctx`](@ref): extracts types from value dictionaries by consulting `get_type` (alias: `collect_types`).
- [`get_ast_type_ctx`](@ref): infers types for AST-valued entries (templates) in a terms context, complementing `get_type_ctx`.
- [`validate_ctx`](@ref): ensures user-supplied contexts respect the no-free-variable discipline.

## Context Discipline: `Ty` vs `TyScheme`

This project uses a simple but strict convention for `Dict{Symbol,TyLike}` contexts:

- Store **monomorphic** types as [`Ty`](@ref) (no free `TyVar` allowed).
- Store **polymorphic** types as [`TyScheme`](@ref).

This is enforced by [`validate_ctx`](@ref). Keeping monomorphic entries as `Ty` avoids unnecessary
scheme instantiation and allocations (e.g. `TyScheme([], ty)` still requires `instantiate` to walk
and rebuild composite types).
"""
module TypeSystem

import Base.Threads

using ..ASTs.NodeASTs: AstTree, AstNode, VarNode, IndexNode, AbsNode, AppNode, ConstNode
using ..Utils
using ..ASTs.Contexts: COND
using TestItems

export Ty, TyConst, TyVar, TyFun, TyMu, TyProd, TySum, TyList, TyNat, TyArray, TyScheme, TyLike
export iscompatible, islesseq, infer, UnboundVariableError, OccursCheckError, MismatchError
export →, ×, μ, ∀, newtyvar, make_tyvar_factory, get_type_ctx, get_ast_type_ctx, collect_types, get_type, infer_scheme

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
• `id`    - unique identifier (integer)
• `level` - creation level, used for generalisation
• `instance` - if unified, points to another `Ty` (acts like union-find parent), work as the substitution rule in basic Hindley-Milner type inference.
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
Array type `Array[len, elem]`, where `elem` is the element type and `len` is either:
- a concrete `TyNat(n)` (fixed length), or
- a `TyVar` (length-polymorphic, to be unified/instantiated via `TyScheme`).

This enables **length-polymorphic primitives** (e.g. `MAP :: ∀n. Array[n,a] → Array[n,b]`)
while keeping datasets/tasks free to use concrete shapes like `TyArray(TyNat(n), ...)`.
"""
struct TyArray <: Ty
    len::Ty
    elem::Ty
end

"""
    TyScheme(vars, ty)
Polymorphic type scheme  ∀ vars. ty
• `vars` - vector of bound TyVar (no instance)
• `ty`   - monotype body
"""
struct TyScheme
    vars::Vector{TyVar}
    ty::Ty
end

# Constants in context may be monotype (Ty) or polymorphic TyScheme
const TyLike = Union{Ty,TyScheme}

Base.hash(t::TyVar, h::UInt) = hash(t.id, hash(t.level, (hash(:TyVar, h))))
Base.hash(t::TyFun, h::UInt) = hash(t.arg, hash(t.res, (hash(:TyFun, h))))
Base.hash(t::TyProd, h::UInt) = hash(t.left, hash(t.right, (hash(:TyProd, h))))
Base.hash(t::TySum, h::UInt) = hash(t.left, hash(t.right, (hash(:TySum, h))))
Base.hash(t::TyMu, h::UInt) = hash(t.var, hash(t.body, (hash(:TyMu, h))))
Base.hash(t::TyNat, h::UInt) = hash(t.value, hash(:TyNat, h))
Base.hash(t::TyArray, h::UInt) = hash(t.len, hash(t.elem, hash(:TyArray, h)))
Base.hash(t::TyScheme, h::UInt) = hash(length(t.vars), hash(_canonical_body(t), hash(:TyScheme, h)))

Base.:(==)(t1::TyVar, t2::TyVar) = t1.id == t2.id && t1.level == t2.level
Base.:(==)(t1::TyFun, t2::TyFun) = t1.arg == t2.arg && t1.res == t2.res
Base.:(==)(t1::TyProd, t2::TyProd) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TySum, t2::TySum) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TyMu, t2::TyMu) = t1.var == t2.var && t1.body == t2.body
Base.:(==)(t1::TyNat, t2::TyNat) = t1.value == t2.value
Base.:(==)(t1::TyArray, t2::TyArray) = t1.len == t2.len && t1.elem == t2.elem
Base.:(==)(t1::TyScheme, t2::TyScheme) = length(t1.vars) == length(t2.vars) && _canonical_body(t1) == _canonical_body(t2)

# --- Structure-only equality and hashing for TyScheme (α-equivalence, order-independent) ---

# Canonicalise TyScheme body by renaming bound variables to deterministic fresh identifiers
function _canonical_body(s::TyScheme; factory=nothing)::Ty
    factory === nothing && (factory = deepcopy(newtyvar))
    acc = TyVar[]
    _record_vars!(s.ty, s.vars, acc)
    canonical_scheme = TyScheme(acc, s.ty)
    return instantiate(canonical_scheme, 0, factory)
end

_record_vars!(ty::Ty, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = nothing
function _record_vars!(ty::TyVar, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[])
    if ty in boundvars && ty ∉ record
        push!(record, ty)
    end
end
_record_vars!(ty::TyFun, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = (_record_vars!(ty.arg, boundvars, record); _record_vars!(ty.res, boundvars, record))
_record_vars!(ty::TyProd, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = (_record_vars!(ty.left, boundvars, record); _record_vars!(ty.right, boundvars, record))
_record_vars!(ty::TySum, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = (_record_vars!(ty.left, boundvars, record); _record_vars!(ty.right, boundvars, record))
_record_vars!(ty::TyList, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = _record_vars!(ty.elem, boundvars, record)
_record_vars!(ty::TyArray, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = (_record_vars!(ty.len, boundvars, record); _record_vars!(ty.elem, boundvars, record))
_record_vars!(ty::TyMu, boundvars::Vector{TyVar}, record::Vector{TyVar}=TyVar[]) = _record_vars!(ty.body, boundvars, record)

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
    value::Threads.Atomic{Int}
end
Counter() = Counter(Threads.Atomic{Int}(0))
(c::Counter)() = Threads.atomic_add!(c.value, 1) + 1

const _TYVAR_COUNTER = Counter()

struct VarFactory
    counter::Counter
end
VarFactory() = VarFactory(Counter())
(n::VarFactory)(level::Int)::TyVar = TyVar(n.counter(), level, nothing)

const newtyvar = VarFactory(_TYVAR_COUNTER) # shared global factory for debugging and legacy entry points

"""
    make_tyvar_factory() -> VarFactory

Create a **fresh** type-variable factory for use in concurrent / multi-threaded
pipelines (e.g. symbolic regression).

Notes
- `newtyvar` is a shared global factory intended for debugging and legacy paths.
- When running SR across individuals in parallel, use a per-task factory created
  by `make_tyvar_factory()` to avoid shared state and to make determinism easier
  to reason about.
"""
make_tyvar_factory() = VarFactory()

# ------------------------------------------------------------
# Utility: prune - follow instances to canonical representative
# ------------------------------------------------------------
function prune(t::Ty)::Ty
    seen = Set{Ty}()
    function _prune(t::Ty)::Ty
        t in seen && return t  # already visited - stop to avoid cycles
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
            return TyArray(_prune(t.len), _prune(t.elem))
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
reference like `μ α . α` would be *non-contractive* and therefore forbidden - it
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
        occ1, con1 = occurs(var, ty.len, depth + 1)
        occ2, con2 = occurs(var, ty.elem, depth + 1)
        return (occ1 || occ2, con1 || con2)
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
        return occurs(var, ty.elem, depth + 1)
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
        # Safe recursive occurrence - fold into μ-type.
        folded = TyMu(var, ty)
        var.instance = folded
        return nothing
    end
end

# ------------------------------------------------------------
# Unification (equi-recursive, memoised)
# ------------------------------------------------------------
"""
    unify!(t1::Ty, t2::Ty) -> Union{Nothing,Err}

Equi-recursive unification. The algorithm first tries to bind a type
variable appearing in the *first* argument. Only when the first argument
is not a free `TyVar` does it attempt to bind one from the second
argument. Calling order therefore determines which side is treated as
flexible (can be instantiated) and which side is considered rigid.
"""
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
        @✓ _unify!(a.len, b.len, seen)
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
function instantiate(s::TyScheme, level::Int, factory=newtyvar)
    # Create substitution only for quantified vars
    subst = Dict{Int,TyVar}()
    for v in s.vars
        subst[v.id] = factory(level)
    end
    return _inst(s.ty, v -> get(subst, v.id, v))
end
instantiate(t::Ty, ::Int, factory=newtyvar) = t # for concrete types, do nothing

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
        return TyArray(_inst(t.len, mapvar), _inst(t.elem, mapvar))
    else
        return t
    end
end

iscompatible(t1::TyLike, t2::TyLike) = isnothing(unify!(instantiate(t1, 0), instantiate(t2, 0)))
"""
    islesseq(a::TyLike, b::TyLike) -> Bool

Semantic subset check: returns `true` iff every value inhabiting `a` can
also inhabit `b`. Quantified variables in `b` are treated as flexible
(can be instantiated) while free variables in `a` are rigid (must remain
unmodified). The function succeeds only when unification can be achieved
by binding *only* the flexible variables.
"""
function islesseq(subset_ty::TyLike, superset_ty::TyLike)::Bool
    # Instantiate superset to make its quantified variables flexible (bindable).
    flex_superset = instantiate(superset_ty, 0)
    # Instantiate subset. Its free variables must remain rigid (unbound).
    rigid_subset = instantiate(subset_ty, 0)
    rigid_vars = collect_free_vars(rigid_subset)
    # Unify with the flexible side as the first argument to guide binding towards it.
    success = isnothing(unify!(flex_superset, rigid_subset))
    success || return false
    # If any rigid variable was bound, it means the subset was improperly narrowed.
    for v in rigid_vars
        v.instance === nothing || return false
    end
    return true
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
            _collect_free_vars(t.len)
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

function lookup(env::Env, idx::Int)
    pos = length(env.stack) - idx
    if pos < 1 || pos > length(env.stack)
        return Err(UnboundVariableError(Symbol("#$(idx)")))
    end
    return env.stack[pos].second
end

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

"""
    infer(tree::AstTree, ctx::Dict{Symbol,<:TyLike}[, factory=newtyvar])

Infer the type given the AST and context.

# Arguments
- `tree::AstTree`: The AST to infer the type of.
- `ctx::Dict{Symbol,<:TyLike}`: The context of the inference.
- `factory::Function`: A function to create fresh type variables.
"""
function infer(tree::AstTree,
               ctx::Dict{Symbol,<:TyLike},
               factory=newtyvar,
               meta::Union{Nothing,Dict{AstNode,Tuple{TyLike,TyLike}}}=nothing)
    env = Env()
    level = 0
    ty = @✓(_infer(tree.root, env, ctx, level, factory, meta))
    # Record bounds for the root node as well (in case _infer returned early)
    if meta !== nothing && !haskey(meta, tree.root)
        ty_min = prune(ty)
        ty_max = generalize(ty_min, level)
        meta[tree.root] = (ty_min, ty_max)
    end
    return prune(ty)
end

"""
    infer_scheme(tree::AstTree, ctx::Dict{Symbol,<:TyLike}; factory=newtyvar) -> Union{TyScheme,Err}

Infer the type of `tree` and return a top-level [`TyScheme`](@ref) by generalising at level 0.

This is the preferred way to attach a reusable type to an AST template: if the inferred type
contains free variables (e.g. `λx.x`), they will be quantified in the returned scheme.
"""
function infer_scheme(tree::AstTree, ctx::Dict{Symbol,<:TyLike}; factory=newtyvar)
    ty = infer(tree, ctx, factory)
    ty isa Err && return ty
    return generalize(ty, 0)
end

_record_meta(meta::Nothing, node::AstNode, ty::Ty, level::Int, orig::TyLike=ty) = ty
function _record_meta(meta::Dict{AstNode,Tuple{TyLike,TyLike}}, node::AstNode, ty::Ty, level::Int, orig::TyLike=ty)
    ty_min = prune(ty)
    ty_max = orig isa TyScheme ? orig : generalize(ty_min, level)
    meta[node] = (ty_min, ty_max)
    return ty
end

function _infer(node::AstNode,
                env::Env,
                ctx::Dict{Symbol,<:TyLike},
                level::Int,
                factory,
                meta::Union{Nothing,Dict{AstNode,Tuple{TyLike,TyLike}}})
    if node isa ConstNode
        entry = ctx[node.name]
        ty = instantiate(entry, level)
        return _record_meta(meta, node, ty, level, entry)
    elseif node isa VarNode
        sch = @✓ lookup(env, node.name)
        ty = instantiate(sch, level)
        return _record_meta(meta, node, ty, level)
    elseif node isa IndexNode
        sch = @✓ lookup(env, node.index)
        ty = instantiate(sch, level)
        return _record_meta(meta, node, ty, level)
    elseif node isa AbsNode
        tv = factory(level+1)
        push!(env, node.name, TyScheme(TyVar[], tv))
        body_ty = @✓ _infer(node.body, env, ctx, level+1, factory, meta)
        pop!(env)
        ty = TyFun(prune(tv), body_ty)
        return _record_meta(meta, node, ty, level)
    elseif node isa AppNode
        arg_ty = @✓ _infer(node.arg, env, ctx, level, factory, meta)
        if node.func isa AbsNode # form like (λx. M) N, treated as let x = N in M, generalize the result type
            arg_ty = generalize(arg_ty, level)
            push!(env, node.func.name, arg_ty)
            body_ty = @✓ _infer(node.func.body, env, ctx, level+1, factory, meta)
            pop!(env)
            return _record_meta(meta, node, body_ty, level)
        else
            fun_ty = @✓ _infer(node.func, env, ctx, level, factory, meta)
            res_ty = factory(level)
            @✓ unify!(fun_ty, TyFun(arg_ty, res_ty))
            return _record_meta(meta, node, res_ty, level)
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
        # `→` is right-associative. Parenthesize the argument position only
        # when it is itself a function type: (A→B)→C.
        arg_s = _show_ty(t.arg, seen)
        t.arg isa TyFun && (arg_s = "(" * arg_s * ")")
        return arg_s * " → " * _show_ty(t.res, seen)
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

"""
    get_type_ctx(constants_ctx::Dict{Symbol,Any}, types_ctx::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}())::Dict{Symbol,TyLike}

Extract types from value dictionaries by consulting `get_type`, skipping AST values.

To infer types for AST-valued entries (templates), use [`get_ast_type_ctx`](@ref).
"""
function get_type_ctx(constants_ctx::Dict{Symbol, Any}, types_ctx::Dict{Symbol, TyLike}=Dict{Symbol, TyLike}())::Dict{Symbol, TyLike}
    for (key, value) in pairs(constants_ctx)
        if haskey(types_ctx, key)
            continue
        end
        if value isa AstTree || value isa AstNode
            continue
        end
        result = get_type(value)
        if !isnothing(result)
            types_ctx[key] = result
        else
            @warn "No type set for constant $key, please specify it manually."
        end
    end
    return types_ctx
end

"""
    get_ast_type_ctx(terms_ctx::Dict{Symbol,Any}, tyctx::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}(); factory=newtyvar) -> Dict{Symbol,TyLike}

Infer types for AST-valued entries in `terms_ctx` (values that are `AstTree` or `AstNode`) and
write them into `tyctx` (without overwriting existing keys).

This complements [`get_type_ctx`](@ref), which intentionally *skips* AST values.

If inference fails for an AST entry, this function throws an error with the corresponding symbol.
"""
function get_ast_type_ctx(
    terms_ctx::Dict{Symbol,Any},
    tyctx::Dict{Symbol,TyLike}=Dict{Symbol,TyLike}();
    factory=newtyvar,
)::Dict{Symbol,TyLike}
    for (key, value) in pairs(terms_ctx)
        haskey(tyctx, key) && continue
        tree = if value isa AstTree
            value
        elseif value isa AstNode
            AstTree(value)
        else
            continue
        end
        scheme = try
            infer_scheme(tree, tyctx; factory=factory)
        catch e
            error("Failed to infer type for AST term :$key: $(e)")
        end
        scheme isa Err && error("Failed to infer type for AST term :$key: $(scheme.value)")
        tyctx[key] = scheme
    end
    return tyctx
end

collect_types(args...; kwargs...) = get_type_ctx(args...; kwargs...)

# ------------------------------------------------------------
# Primitive base types and Prelude constant typing
# ------------------------------------------------------------

# Primitive value-type constants
const BoolTy  = TyConst(:Bool)
const IntTy   = TyConst(:Int)
const FloatTy = TyConst(:Float)
const NilErr  = TyConst(:NilErr)

const IntDiscr = IntTy → IntTy → BoolTy
const IntOp = IntTy → IntTy → IntTy
const FloatDiscr = FloatTy → FloatTy → BoolTy
const FloatOp = FloatTy → FloatTy → FloatTy

# Sum type basic functions
INL(x) = x
INR(x) = x

# Product type basic functions
PAIR(x, y) = (x, y)
FST(x) = x.first
SND(x) = x.second

# List type basic functions
struct _NilErr end
const NILERR = _NilErr()
CONS(x) = l -> [x;l]
HEAD(xs) = isempty(xs) ? NILERR : xs[1]
TAIL(xs) = isempty(xs) ? NILERR : xs[2:end]
LENGTH(xs) = length(xs)
CASENILERR(f) = g -> (x ->(x isa _NilErr ? g(x) : f(x)))
TOLIST(xs) = collect(xs)

"""
    AMAP(f) -> (xs -> ys)

Curried array `map` primitive for fixed-length arrays (`TyArray`).

This is intended as a **length-preserving** eliminator/builder for static-length
arrays:

- it does not require any type-level nat arithmetic (no `n-1`),
- it keeps the length index `n` unchanged.
"""
AMAP(f) = xs -> map(f, xs)

"""
    AFOLDL(op) -> (init -> (xs -> acc))

Curried left fold over fixed-length arrays (`TyArray`).

`op` is expected to be curried: `op(acc)(x)`.
"""
AFOLDL(op) = init -> xs -> foldl((acc, x) -> op(acc)(x), xs; init=init)

"""
    LENF(xs) -> Float64

Float length of an array. This is intentionally `Float`-typed to avoid requiring
an `Int -> Float` conversion primitive.
"""
LENF(xs) = float(length(xs))

FSIN(x) = sin(x)
FSQRT(x) = sqrt(x)
SQR(x) = x * x

# Int and Float basic functions
IEQ(x) = y -> x == y
ILE(x) = y -> x <= y
IADD(x) = y -> x + y
ISUB(x) = y -> x - y
IMUL(x) = y -> x * y

FEQ(x) = y -> x == y
FLE(x) = y -> x <= y
FADD(x) = y -> x + y
FSUB(x) = y -> x - y
FMUL(x) = y -> x * y
FDIV(x) = y -> x / y

_α, _β, _γ = newtyvar(0), newtyvar(0), newtyvar(0)

# Fallbacks for primitive constants (already defined earlier)
get_type(::Bool)  = BoolTy
get_type(::Int)   = IntTy
get_type(::Float64) = FloatTy
get_type(::Float32) = FloatTy

get_type(::typeof(COND)) = ∀([_α], BoolTy → _α → _α → _α)

get_type(::typeof(INL)) = ∀([_α, _β], _α → (_α + _β))
get_type(::typeof(INR)) = ∀([_α, _β], _β → (_α + _β))
# Just an example. As inexchangable sum type is not supported in Julia, case function must be implemented manually for each sum type.
# get_type(::typeof(CASE)) = ∀([_α, _β, _γ], (_α → _γ) → (_β → _γ) → (_α + _β) → _γ)

get_type(::typeof(PAIR)) = ∀([_α, _β], _α → _β → (_α × _β))
get_type(::typeof(FST)) = ∀([_α, _β], (_α × _β) → _α)
get_type(::typeof(SND)) = ∀([_α, _β], (_α × _β) → _β)
get_type(t::Tuple{T, S}) where {T, S} = get_type(t[1]) × get_type(t[2])

get_type(::typeof(CONS)) = ∀([_α], _α → TyList(_α) → TyList(_α))
get_type(::typeof(HEAD)) = ∀([_α], TyList(_α) → (_α + NilErr))
get_type(::typeof(TAIL)) = ∀([_α], TyList(_α) → TyList(_α) + NilErr)
get_type(::typeof(LENGTH)) = ∀([_α], TyList(_α) → IntTy)
get_type(::typeof(CASENILERR)) = ∀([_α, _β], (_α → _β) → (NilErr → _β) → (_α + NilErr) → _β)
get_type(::typeof(TOLIST)) = ∀([_α, _β], TyArray(_α, _β) → TyList(_β))

get_type(::typeof(AMAP)) = ∀([_α, _β, _γ], (_β → _γ) → TyArray(_α, _β) → TyArray(_α, _γ))
get_type(::typeof(AFOLDL)) = ∀([_α, _β, _γ], (_γ → _β → _γ) → _γ → TyArray(_α, _β) → _γ)
get_type(::typeof(LENF)) = ∀([_α, _β], TyArray(_α, _β) → FloatTy)

get_type(::typeof(FSIN)) = FloatTy → FloatTy
get_type(::typeof(FSQRT)) = FloatTy → FloatTy
get_type(::typeof(SQR)) = FloatTy → FloatTy

get_type(::typeof(IEQ)) = IntDiscr
get_type(::typeof(ILE)) = IntDiscr
get_type(::typeof(IADD)) = IntOp
get_type(::typeof(ISUB)) = IntOp
get_type(::typeof(IMUL)) = IntOp

get_type(::typeof(FEQ)) = FloatDiscr
get_type(::typeof(FLE)) = FloatDiscr
get_type(::typeof(FADD)) = FloatOp
get_type(::typeof(FSUB)) = FloatOp
get_type(::typeof(FMUL)) = FloatOp
get_type(::typeof(FDIV)) = FloatOp

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

@testitem "TypeSystem pretty print arrow precedence" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyFun

    # `(A→B)→C` must print with parentheses on the argument side.
    arrow = string(Char(0x2192))
    t = TyFun(TyFun(IntTy, IntTy), TyFun(IntTy, IntTy))
    s = replace(sprint(show, t), " " => "")
    @test s == "(Int$(arrow)Int)$(arrow)Int$(arrow)Int"
end

@testitem "simple hashing and equality" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: TyLike
    
    α = newtyvar(0)
    β = newtyvar(0)
    γ = newtyvar(0)
    δ = newtyvar(0)
    μ = newtyvar(0)
    ν = newtyvar(0)

    @test α == α
    @test α != β
    @test α == deepcopy(α)

    @test (α → β) == (α → β)
    @test (α → β) != (β → α)
 
    @test ∀([α, β], α → β) == ∀([β, α], β → α)
    @test ∀([α, β], α → β) == ∀([α, β], β → α)
    @test ∀([α, β], α → α) != ∀([α, β], α → β)

    @test hash(α) == hash(α)
    @test hash(α) != hash(β)
    @test hash(α) == hash(deepcopy(α))
    @test hash(α → β) == hash(α → β)
    @test hash(α → β) != hash(β → α)
    @test hash(∀([α, β], α → β)) == hash(∀([β, α], β → α))
    @test hash(∀([α, β], α → β)) == hash(∀([α, β], β → α))
    @test hash(∀([α, β], α → α)) != hash(∀([α, β], α → β))

    s = Set([∀([α, β], α → β)])
    @test ∀([β, α], β → α) in s
    @test ∀([α, β], α → α) ∉ s    
end

@testitem "TypeSystem make_tyvar_factory is detached" begin
    using LambdaRegression.TypeSystem: make_tyvar_factory

    f1 = make_tyvar_factory()
    f2 = make_tyvar_factory()

    a1 = f1(0)
    a2 = f1(0)
    b1 = f2(0)
    b2 = f2(0)

    @test a1.id == 1
    @test a2.id == 2
    @test b1.id == 1
    @test b2.id == 2
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

@testitem "infer_scheme and get_ast_type_ctx" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: infer, infer_scheme, get_type_ctx, get_ast_type_ctx, TyLike, TyScheme
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, IndexNode
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms

    # infer_scheme generalises free variables at level 0
    id_named = AstTree(AbsNode(:x, VarNode(:x)))
    sch = infer_scheme(id_named, Dict{Symbol,TyLike}())
    @test sch isa TyScheme
    @test length(sch.vars) == 1

    # Out-of-range de Bruijn indices should not throw; they should return Err
    out_of_range = infer(AstTree(IndexNode(0)), Dict{Symbol,TyLike}())
    @test out_of_range isa Err

    # get_ast_type_ctx complements get_type_ctx (which skips AST values)
    terms_ctx = @terms(ID=parse_lambda("λx.x"))
    tyctx = get_type_ctx(terms_ctx)
    @test !haskey(tyctx, :ID)
    get_ast_type_ctx(terms_ctx, tyctx)
    @test haskey(tyctx, :ID)
    @test tyctx[:ID] isa TyScheme
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

@testitem "list occurs check contractiveness" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: newtyvar, unify!

    α = newtyvar(0)
    result = unify!(α, TyList(α))
    @test isnothing(result)
    @test α.instance isa TyMu
    @test α.instance.body isa TyList
    @test α.instance.body.elem === α
end

@testitem "factorial via Y_int" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, IEQ, ISUB, IMUL, COND, get_type_ctx
    using LambdaRegression.ASTs.NodeASTs

    ctx = merge(
        get_type_ctx(Dict{Symbol,Any}(
            :zero => 0,
            :one => 1,
            :IEQ => IEQ,
            :ISUB => ISUB,
            :IMUL => IMUL,
            :COND => COND,
        )),
        Dict{Symbol,TyLike}(
            :Y_int => ((IntTy → IntTy) → (IntTy → IntTy)) → (IntTy → IntTy),
        ),
    )

    # Build λf. λn. if_int (iszero n) one (mul n (f (pred n)))
    var_f = IndexNode(1)
    var_n = IndexNode(0)
    iszero_n = AppNode(AppNode(ConstNode(:IEQ), var_n), ConstNode(:zero))
    pred_n = AppNode(AppNode(ConstNode(:ISUB), var_n), ConstNode(:one))
    fn_pred = AppNode(var_f, pred_n)
    mul_expr = AppNode(AppNode(ConstNode(:IMUL), var_n), fn_pred)
    if_expr = AppNode(AppNode(AppNode(ConstNode(:COND), iszero_n), ConstNode(:one)), mul_expr)
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
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.Reduces: prepare
    using LambdaRegression.TypeSystem: IntTy, IMUL, ISUB, IEQ, COND, get_type_ctx

    const_ctx = @terms(cond=COND, eq=IEQ, mul=IMUL, sub=ISUB, int0=0, int1=1, int4=4)

    merge!(const_ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", const_ctx)))
    merge!(const_ctx, @terms(fac=parse_lambda("Y (λfac.λn. cond (eq n int0) int1 (mul n (fac (sub n int1))))", const_ctx)))
    fac4 = parse_lambda("fac int4", const_ctx)
    fac4 = prepare(fac4)

    tyctx = get_type_ctx(const_ctx)
    ty = infer(fac4, tyctx)
    @test ty == IntTy
end

@testitem "polymorphic if constant" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: COND, IntTy, get_type_ctx

    ctx = get_type_ctx(Dict{Symbol,Any}(
        :COND => COND,
        :boolTrue => true,
        :int1 => 1,
        :int2 => 2,
    ))

    # expression: COND boolTrue int1 int2  ==> Int
    expr = AppNode(AppNode(AppNode(ConstNode(:COND), ConstNode(:boolTrue)), ConstNode(:int1)), ConstNode(:int2))
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === IntTy
end

# -------------------- Product & Sum basic tests --------------------

@testitem "product unify and projection" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FST, PAIR, IntTy, get_type_ctx

    ctx = get_type_ctx(Dict{Symbol,Any}(
        :FST => FST,
        :PAIR => PAIR,
        :three => 3,
        :btrue => true,
    ))

    pair_expr = AppNode(AppNode(ConstNode(:PAIR), ConstNode(:three)), ConstNode(:btrue))
    expr = AppNode(ConstNode(:FST), pair_expr)
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === IntTy
end

@testitem "sum inl inr case" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: INL, INR, IntTy, BoolTy, get_type_ctx

    α, β, γ = newtyvar(0), newtyvar(0), newtyvar(0)
    ctx = merge(
        get_type_ctx(Dict{Symbol,Any}(
            :INL => INL,
            :INR => INR,
            :int3 => 3,
            :booltrue => true,
        )),
        Dict{Symbol,TyLike}(
            :case => ∀([α,β,γ], (α → γ) → (β → γ) → (α + β → γ)),
        ),
    )
    left_branch = AbsNode(:x, ConstNode(:booltrue)) # returns Bool
    right_branch = AbsNode(:y, ConstNode(:booltrue))
    sum_val = AppNode(ConstNode(:INL), ConstNode(:int3))
    expr = AppNode(AppNode(AppNode(ConstNode(:case), left_branch), right_branch), sum_val)
    tree = AstTree(expr)
    ty = infer(tree, ctx)
    @test ty === BoolTy
end

@testitem "TyArray length polymorphism unify" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: newtyvar, unify!, prune, get_type, TOLIST
    using LambdaRegression.Utils: Err

    # Length variable can unify with a concrete TyNat.
    n = newtyvar(0)
    t1 = TyArray(n, TypeSystem.FloatTy)
    t2 = TyArray(TyNat(5), TypeSystem.FloatTy)
    @test isnothing(unify!(t1, t2))
    @test prune(n) == TyNat(5)

    # Length mismatch remains a type error.
    @test unify!(TyArray(TyNat(2), TypeSystem.FloatTy), TyArray(TyNat(3), TypeSystem.FloatTy)) isa Err

    # Previously-broken polymorphic primitive is now representable.
    @test get_type(TOLIST) isa TypeSystem.TyScheme
end

@testitem "TypeSystem Array HO primitives typing" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyArray, TyNat, FloatTy, unify!, get_type

    A = TyArray(TyNat(3), FloatTy)

    map_ty = TypeSystem.instantiate(get_type(TypeSystem.AMAP), 0)
    @test unify!(map_ty, (FloatTy → FloatTy) → A → A) === nothing

    fold_ty = TypeSystem.instantiate(get_type(TypeSystem.AFOLDL), 0)
    @test unify!(fold_ty, (FloatTy → FloatTy → FloatTy) → FloatTy → A → FloatTy) === nothing

    lenf_ty = TypeSystem.instantiate(get_type(TypeSystem.LENF), 0)
    @test unify!(lenf_ty, A → FloatTy) === nothing
end

# ------------------------------------------------------------
# Advanced type invariance tests --------------------

@testitem "factorial via Y combinator - type invariance under reduction" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.Reduces: prepare, beta_reduce!
    using LambdaRegression.TypeSystem: IntTy, IMUL, ISUB, IEQ, COND, get_type_ctx

    # Deterministic integer identifiers to avoid random gensym names
    ctx = @terms(cond=COND, eq=IEQ, mul=IMUL, sub=ISUB, int0=0, int1=1, int4=4)
    merge!(ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", ctx)))
    merge!(ctx, @terms(fac=parse_lambda("Y (λfac.λn. cond (eq n int0) int1 (mul n (fac (sub n int1))))", ctx)))
    expr = parse_lambda("fac int4", ctx)
    expr = prepare(expr)

    # Initial type context
    tyctx = get_type_ctx(ctx)

    # Type before reduction
    @test infer(expr, tyctx) == IntTy
    beta_reduce!(expr, 20_000, ctx)
    value = ctx[expr.root.name]
    @test value == 24
end

@testitem "list square-sum - type invariance under reduction" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.Reduces: prepare, beta_reduce!
    using LambdaRegression.TypeSystem: newtyvar, TyLike, IntTy, NilErr, TyList, CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND, get_type_ctx

    β = newtyvar(0)

    id = x -> x
    nil_to_zero = _ -> 0
    nil_to_empty = _ -> Int[]
    empty_list = Int[]
    test_list = [1,2,3,4,5]
    ctx = @terms(CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND, id, nil_to_zero, nil_to_empty, 0, empty_list, test_list)
    merge!(ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", ctx)))
    merge!(ctx, @terms(appl=parse_lambda("Y (λappl.λf.λlist. COND (IEQ (LENGTH list) 0) empty_list ((CONS (f (CASENILERR id nil_to_zero (HEAD list)))) (appl f (CASENILERR id nil_to_empty (TAIL list)))))", ctx)))
    apply_square_tree = parse_lambda("appl (λx. IMUL x x) test_list", ctx)
    apply_square_tree = prepare(apply_square_tree)

    merge!(ctx, @terms(SUM=parse_lambda("Y (λsum.λlist. COND (IEQ (LENGTH list) 0) 0 (IADD (CASENILERR id nil_to_zero (HEAD list)) (sum (CASENILERR id nil_to_empty (TAIL list)))))", ctx)))
    merge!(ctx, @terms(sum_of_squares=parse_lambda("λlist. SUM (appl (λx. IMUL x x) list)", ctx)))
    result_tree = parse_lambda("sum_of_squares test_list", ctx)
    result_tree = prepare(result_tree)

    # Type context covering NilErr handlers and list constants
    tyctx = get_type_ctx(ctx, Dict{Symbol,TyLike}(
        :id => ∀([β], β → β),
        :nil_to_zero => NilErr → IntTy,
        :nil_to_empty => NilErr → TyList(IntTy),
        :empty_list => TyList(IntTy),
        :test_list => TyList(IntTy),
    ))

    # ----- `appl (λx. mul x x) test_list` -----
    @test iscompatible(infer(apply_square_tree, tyctx), TyList(IntTy))
    beta_reduce!(apply_square_tree, 10_000, ctx)
    list_val = ctx[apply_square_tree.root.name]
    @test list_val == [1,4,9,16,25]

    # ----- `sum_of_squares test_list` -----
    @test iscompatible(infer(result_tree, tyctx), IntTy)
    beta_reduce!(result_tree, 10_000, ctx)
    sum_val = ctx[result_tree.root.name]
    @test sum_val == 55
end

# ------------------------------------------------------------
# Meta bounds recording tests
# ------------------------------------------------------------

@testitem "islesseq subset polymorphism" begin
    using LambdaRegression.TypeSystem
    IntTy = TyConst(:Int)
    α = newtyvar(0)
    scheme_any = ∀([α], α)
    @test islesseq(IntTy, scheme_any)
    @test !islesseq(scheme_any, IntTy)
end

@testitem "islesseq rigidity and function" begin
    using LambdaRegression.TypeSystem
    IntTy, β, γ = TyConst(:Int), newtyvar(0), newtyvar(0)
    scheme_fun = ∀([β], β → β)
    @test islesseq(IntTy → IntTy, scheme_fun)
    @test !islesseq(scheme_fun, IntTy → IntTy)
    @test !islesseq(γ, IntTy)          # rigid var `γ` on subset side cannot be bound
    @test islesseq(IntTy, γ)           # superset `γ` is flexible and can be bound
end

@testitem "islesseq advanced subtyping" begin
    using LambdaRegression.TypeSystem
    IntTy, BoolTy = TyConst(:Int), TyConst(:Bool)
    α, β = newtyvar(0), newtyvar(0)

    # Scheme vs Scheme
    id_scheme1, id_scheme2 = ∀([α], α → α), ∀([β], β → β)
    @test islesseq(id_scheme1, id_scheme2) && islesseq(id_scheme2, id_scheme1)
    int_fun_scheme = ∀([α], α → IntTy)
    @test !islesseq(int_fun_scheme, id_scheme1) # (α→Int) ⊈ (β→β)
    @test !islesseq(id_scheme1, int_fun_scheme) # (β→β) ⊈ (α→Int)

    # Mu vs Mu (equi-recursive) & Mu vs Scheme
    mu_int1, mu_int2 = μ(α, α → IntTy), μ(β, β → IntTy)
    @test islesseq(mu_int1, mu_int2) && islesseq(mu_int2, mu_int1)
    mu_bool = μ(α, α → BoolTy)
    @test !islesseq(mu_int1, mu_bool)
    any_scheme = ∀([β], β)
    @test islesseq(mu_int1, any_scheme)      # (μ α. α→Int) ⊆ (∀β. β)
    @test !islesseq(any_scheme, mu_int1)     # (∀β. β) ⊈ (μ α. α→Int)
end

@testitem "meta bounds recording" begin
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs

    using LambdaRegression.TypeSystem: IntTy, get_type_ctx

    # Prepare context with a polymorphic identity constant
    α = newtyvar(0)
    ctx = merge(
        get_type_ctx(Dict{Symbol,Any}(:zero => 0)),
        Dict{Symbol,TyLike}(
            :id => TyScheme([α], TyFun(α, α)),
        ),
    )

    zero_node = ConstNode(:zero)                 # TyConst
    id_node   = ConstNode(:id)                   # TyScheme (polymorphic constant)
    id_lambda = AbsNode(:x, IndexNode(0))        # TyFun
    self_app  = AbsNode(:x, AppNode(IndexNode(0), IndexNode(0))) # TyMu in arg

    # Helper to run inference and grab meta dict
    function infer_with_meta(n)
        meta = Dict{AstNode,Tuple{TyLike,TyLike}}()
        infer(AstTree(n), ctx, newtyvar, meta)
        return meta
    end

    # ---- TyConst ----
    meta_zero = infer_with_meta(zero_node)
    @test haskey(meta_zero, zero_node)
    min_zero, max_zero = meta_zero[zero_node]
    @test min_zero === IntTy
    @test max_zero isa TyScheme && max_zero.ty === IntTy

    # ---- TyScheme / polymorphic constant ----
    meta_id = infer_with_meta(id_node)
    @test haskey(meta_id, id_node)
    min_id, max_id = meta_id[id_node]
    @test min_id isa TyFun
    @test max_id isa TyScheme && length(max_id.vars) == 1

    # ---- TyFun ----
    meta_fun = infer_with_meta(id_lambda)
    @test haskey(meta_fun, id_lambda)
    min_fun, max_fun = meta_fun[id_lambda]
    @test min_fun isa TyFun
    @test max_fun isa TyScheme

    # ---- TyMu ----
    meta_mu = infer_with_meta(self_app)
    @test haskey(meta_mu, self_app)
    min_mu, max_mu = meta_mu[self_app]
    @test min_mu isa TyFun && min_mu.arg isa TyMu
    @test max_mu isa TyScheme
end

end # module TypeSystem
