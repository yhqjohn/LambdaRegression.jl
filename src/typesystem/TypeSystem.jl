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
- [`TyScheme`](@ref): polymorphic type schemes with allocation-free α-equivalent keys.
- [`TyLike`](@ref): convenience alias `Union{Ty,TyScheme}` used throughout contexts and metadata.
- [`Counter`](@ref) and [`VarFactory`](@ref): fresh-variable generation utilities (`newtyvar` exposes the shared factory).

## Fundamental Operations
- [`prune`](@ref): materialises the current public `TyVar` facade.
- [`isequal_mu`](@ref): compares closed recursive types through finite cyclic graphs.
- [`unify!`](@ref): performs guarded, transactional graph unification.
- [`instantiate`](@ref) / [`generalize`](@ref): Hindley-Milner instantiation discipline for schemes and let-polymorphism.
- [`collect_free_vars`](@ref): helper for rigid-variable tracking.
- [`infer`](@ref): walks an [`AstTree`](@ref LambdaRegression.ASTs.NodeASTs.AstTree) with environments, optionally recording lower/upper type bounds per node.

## Primitive Library
- Predefined constants (see [`BoolTy`](@ref TypeSystem.BoolTy), [`IntTy`](@ref TypeSystem.IntTy), [`FloatTy`](@ref TypeSystem.FloatTy), [`TyList`](@ref), [`TyArray`](@ref), [`TyNat`](@ref)) mirror common data structures without altering inference rules.
- Boolean values and combinators such as `TRUE`, `FALSE`, and `NOT` are ordinary typed constants; generators may inject them into contexts, but inference gives them no special rule.
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

## Quickstart And Customisation

Use [`infer`](@ref) / [`infer_scheme`](@ref) for the built-in AST representation. The two-argument
[`infer_term`](@ref) overload uses the same `NodeTermAdapter`; for another term representation,
implement [`term_view`](@ref) and call the adapter overload of [`infer_term`](@ref) or
[`infer_scheme_term`](@ref); the W driver, let generalisation, unifier, and failures remain shared.
Adapter binder names are `Symbol`s; variable references are either `Symbol`s or zero-based De Bruijn
`Int`s. The W language deliberately reads an application whose function view is `:abs` as the
project's let-like `(λx. M) N`, so its argument is inferred in a fresh generalisation region.

To add a finite-arity type constructor, subtype [`Ty`](@ref) and implement
[`type_constructor_tag`](@ref), [`type_static_parameters`](@ref), [`type_children`](@ref), and
[`rebuild_type`](@ref). Equality, unification, instantiation, generalisation, and pruning then use the
same constructor protocol. Primitive atoms are ordinary zero-arity constructors. An immutable
extension that deliberately shares one node across non-sibling DAG positions should also implement
[`type_node_identity`](@ref); ordinary trees and mutable extension nodes need no identity hook.

Use a factory from [`make_tyvar_factory`](@ref) for an isolated or concurrent inference session.
[`isequal_mu`](@ref) compares closed types, [`iscompatible`](@ref) is a symmetric pure query, and
[`islesseq`](@ref) is the directed rigid-left instance relation. [`typekey`](@ref) is an
α-normalised structural key; [`semantic_typekey`](@ref) is the canonical closed key modulo `≈μ`.
Failed `unify!` calls roll back facade links and level changes; unexpected extension errors are
rethrown after the same rollback.
"""
module TypeSystem

import Base.Threads

using ..ASTs.NodeASTs: AstTree, AstNode, VarNode, IndexNode, AbsNode, AppNode, ConstNode, FunctionDefNode
using ..Utils
using TestItems

export Ty, TyConst, TyVar, TyRigid, TyFun, TyMu, TyProd, TySum, TyList, TyNat, TyArray, TyScheme, TyLike
export iscompatible, islesseq, isequal_mu, semantic_typekey, infer, infer_term, infer_scheme_term, UnboundVariableError, OccursCheckError, MismatchError, NonFunctionDefinitionError
export →, ×, μ, ∀, newtyvar, make_tyvar_factory, get_type_ctx, get_ast_type_ctx, collect_types, get_type, infer_scheme
export type_constructor_tag, type_static_parameters, type_children, rebuild_type, type_node_identity
export NodeTermAdapter, term_view, record_binding
public BoolTy, IntTy, FloatTy

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
• `id`    - factory-local display identifier (integer)
• `level` - creation level, used for generalisation
• `instance` - if unified, points to another `Ty` (acts like union-find parent), work as the substitution rule in basic Hindley-Milner type inference.

Variable equality uses an internal immutable identity. It is preserved by
`deepcopy`, independent of `level`, and distinct across detached factories.
"""
mutable struct TyVar <: Ty
    id::Int
    level::Int
    instance::Union{Ty, Nothing}
    uid::UInt
    function TyVar(
        id::Int,
        level::Int,
        instance::Union{Ty,Nothing},
        uid::UInt,
        ::Val{:trusted},
    )
        return new(id, level, instance, uid)
    end
end
const _TYVAR_UID_COUNTER = Threads.Atomic{UInt}(0)
_next_tyvar_uid() = Threads.atomic_add!(_TYVAR_UID_COUNTER, UInt(1)) + UInt(1)
TyVar(id::Int, level::Int, instance::Union{Ty,Nothing}) =
    TyVar(id, level, instance, _next_tyvar_uid(), Val(:trusted))
Base.deepcopy_internal(t::TyVar, ::IdDict) = t
copy_detached(t::TyVar) = TyVar(t.id, t.level, nothing)

"""A rigid query-local type atom. Rigid atoms compare by `id` and never bind."""
struct TyRigid <: Ty
    id::UInt
end
TyRigid(id::Integer) = TyRigid(UInt(id))

"""
    TyFun(arg, res)
Function type `arg → res`.
"""
mutable struct TyFun <: Ty
    arg::Ty
    res::Ty
end

"""
    TyMu(var, body)
Equi-recursive μ-type  `μ var . body`.
The `var` field is a `TyVar` acting as the binder.
"""
mutable struct TyMu <: Ty
    var::TyVar
    body::Ty
end

"""
    TyProd(left, right)
Product type  `left × right`.
"""
mutable struct TyProd <: Ty
    left::Ty
    right::Ty
end

"""
    TySum(left, right)
Sum type `left + right` (disjoint union).
"""
mutable struct TySum <: Ty
    left::Ty
    right::Ty
end

"""
    TyList(elem)
Covariant list type `List elem`.
"""
mutable struct TyList <: Ty
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
mutable struct TyArray <: Ty
    len::Ty
    elem::Ty
end

const _SEMANTICALLY_IMMUTABLE_TYPES = Union{TyFun,TyMu,TyProd,TySum,TyList,TyArray}
function Base.setproperty!(::_SEMANTICALLY_IMMUTABLE_TYPES, name::Symbol, value)
    throw(ArgumentError("type-constructor fields are immutable; rebuild the type instead of assigning .$name"))
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

Base.hash(t::TyVar, h::UInt) = hash(t.uid, hash(:TyVar, h))
Base.hash(t::TyRigid, h::UInt) = hash(t.id, hash(:TyRigid, h))
Base.hash(t::TyFun, h::UInt) = hash(t.arg, hash(t.res, (hash(:TyFun, h))))
Base.hash(t::TyProd, h::UInt) = hash(t.left, hash(t.right, (hash(:TyProd, h))))
Base.hash(t::TySum, h::UInt) = hash(t.left, hash(t.right, (hash(:TySum, h))))
Base.hash(t::TyMu, h::UInt) = hash(t.var, hash(t.body, (hash(:TyMu, h))))
Base.hash(t::TyList, h::UInt) = hash(t.elem, hash(:TyList, h))
Base.hash(t::TyNat, h::UInt) = hash(t.value, hash(:TyNat, h))
Base.hash(t::TyArray, h::UInt) = hash(t.len, hash(t.elem, hash(:TyArray, h)))
Base.hash(t::TyScheme, h::UInt) = hash(_scheme_key(t), hash(:TyScheme, h))

Base.:(==)(t1::TyVar, t2::TyVar) = t1.uid == t2.uid
Base.isequal(t1::TyVar, t2::TyVar) = t1.uid == t2.uid
Base.:(==)(t1::TyRigid, t2::TyRigid) = t1.id == t2.id
Base.isequal(t1::TyRigid, t2::TyRigid) = t1.id == t2.id
Base.:(==)(t1::TyFun, t2::TyFun) = t1.arg == t2.arg && t1.res == t2.res
Base.:(==)(t1::TyProd, t2::TyProd) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TySum, t2::TySum) = t1.left == t2.left && t1.right == t2.right
Base.:(==)(t1::TyMu, t2::TyMu) = t1.var == t2.var && t1.body == t2.body
Base.:(==)(t1::TyList, t2::TyList) = t1.elem == t2.elem
Base.:(==)(t1::TyNat, t2::TyNat) = t1.value == t2.value
Base.:(==)(t1::TyArray, t2::TyArray) = t1.len == t2.len && t1.elem == t2.elem
Base.:(==)(t1::TyScheme, t2::TyScheme) = _scheme_key(t1) == _scheme_key(t2)

include("EquirecursiveCore.jl")

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
mutable struct _TypeRewriteMemo
    mutable_entries::IdDict{Ty,Tuple{Int,Ty}}
    keyed_entries::Dict{Tuple{DataType,Any},Tuple{Int,Ty}}
end

_TypeRewriteMemo() = _TypeRewriteMemo(
    IdDict{Ty,Tuple{Int,Ty}}(),
    Dict{Tuple{DataType,Any},Tuple{Int,Ty}}(),
)

@inline function _rewrite_lookup(memo::_TypeRewriteMemo, ty::Ty, scope::Int)
    if Base.ismutabletype(typeof(ty))
        entry = get(memo.mutable_entries, ty, nothing)
        return entry !== nothing && entry[1] == scope ? entry[2] : nothing
    end
    identity = type_node_identity(ty)
    identity === nothing && return nothing
    entry = get(memo.keyed_entries, (typeof(ty), identity), nothing)
    return entry !== nothing && entry[1] == scope ? entry[2] : nothing
end

@inline function _rewrite_remember!(memo::_TypeRewriteMemo, ty::Ty, scope::Int, value::Ty)
    if Base.ismutabletype(typeof(ty))
        memo.mutable_entries[ty] = (scope, value)
    else
        identity = type_node_identity(ty)
        identity === nothing || (memo.keyed_entries[(typeof(ty), identity)] = (scope, value))
    end
    return value
end

function _rewrite_children(children::Tuple, rewrite)
    rewritten = Ty[]
    for i in eachindex(children)
        child = children[i]
        duplicate = 0
        for j in 1:(i - 1)
            if children[j] === child
                duplicate = j
                break
            end
        end
        push!(rewritten, duplicate == 0 ? rewrite(child) : rewritten[duplicate])
    end
    return tuple(rewritten...)
end

function prune(t::Ty)::Ty
    resolving = Dict{TyVar,Nothing}()
    binders = Dict{TyVar,Nothing}()
    memo = _TypeRewriteMemo()
    scope = Ref(0)
    next_scope = Ref(0)

    function _prune(t::Ty)::Ty
        if t isa TyVar
            haskey(binders, t) && return t
            t.instance === nothing && return t
            haskey(resolving, t) && return t
            resolving[t] = nothing
            try
                return _prune(t.instance)
            finally
                delete!(resolving, t)
            end
        end

        children = type_children(t)
        isempty(children) && return t
        known = _rewrite_lookup(memo, t, scope[])
        known === nothing || return known

        if t isa TyMu
            outer_scope = scope[]
            next_scope[] += 1
            scope[] = next_scope[]
            had_binder = haskey(binders, t.var)
            binders[t.var] = nothing
            body = try
                _prune(t.body)
            finally
                scope[] = outer_scope
                had_binder || delete!(binders, t.var)
            end
            value = body === t.body ? t : TyMu(t.var, body)
            return _rewrite_remember!(memo, t, outer_scope, value)
        end

        rewritten = _rewrite_children(children, _prune)
        value = all(i -> rewritten[i] === children[i], eachindex(children)) ?
            t : rebuild_type(t, rewritten)
        return _rewrite_remember!(memo, t, scope[], value)
    end
    return _prune(t)
end

# ------------------------------------------------------------
# Transactional facade changes
# ------------------------------------------------------------

"""
    TyTrail()

Undo-log for transactional unification.

When passed to [`unify!`](@ref), every `TyVar.instance` mutation is recorded so callers can rollback a failed
generation/mutation attempt without leaking partial bindings.
"""
abstract type _TyTrailEntry end

struct _TyInstanceChange <: _TyTrailEntry
    var::TyVar
    old::Union{Ty,Nothing}
end

struct _TyLevelChange <: _TyTrailEntry
    var::TyVar
    old::Int
end

mutable struct TyTrail
    stack::Vector{_TyTrailEntry}
end
TyTrail() = TyTrail(_TyTrailEntry[])

mark(trail::TyTrail)::Int = length(trail.stack)

function rollback!(trail::TyTrail, m::Int)
    @assert 0 <= m <= length(trail.stack)
    for i in length(trail.stack):-1:(m + 1)
        change = trail.stack[i]
        if change isa _TyInstanceChange
            change.var.instance = change.old
        else
            level_change = change::_TyLevelChange
            level_change.var.level = level_change.old
        end
    end
    resize!(trail.stack, m)
    return nothing
end

function _set_instance!(var::TyVar, ty::Ty, trail::TyTrail)
    push!(trail.stack, _TyInstanceChange(var, var.instance))
    var.instance = ty
    return nothing
end

function _set_instance!(var::TyVar, ty::Ty, ::Nothing)
    var.instance = ty
    return nothing
end

function _set_level!(var::TyVar, level::Int, trail::TyTrail)
    var.level == level && return nothing
    push!(trail.stack, _TyLevelChange(var, var.level))
    var.level = level
    return nothing
end

function _set_level!(var::TyVar, level::Int, ::Nothing)
    var.level = level
    return nothing
end

# ------------------------------------------------------------
# Finite cyclic unification
# ------------------------------------------------------------
"""
    unify!(t1::Ty, t2::Ty; trail=nothing) -> Union{Nothing,Err}

Equi-recursive unification over a finite query-local graph. Written `μ`
binders elaborate to fixed back edges, while unbound `TyVar`s become flexible
metas. Public `TyVar.instance` and level changes are committed only after the
whole graph equation succeeds; failures roll back the supplied trail.
"""
function unify!(t1::Ty, t2::Ty; trail::Union{Nothing,TyTrail}=nothing)
    trail === nothing && return _unify_graph!(t1, t2, nothing)

    transaction = mark(trail)
    result = try
        _unify_graph!(t1, t2, trail)
    catch
        rollback!(trail, transaction)
        rethrow()
    end
    result isa Err && rollback!(trail, transaction)
    return result
end

# ------------------------------------------------------------
# Instantiate & Generalise
# ------------------------------------------------------------
function instantiate(s::TyScheme, level::Int, factory=newtyvar)
    # Create substitution only for quantified vars
    subst = Dict{TyVar,Ty}()
    for v in s.vars
        subst[v] = factory(level)
    end
    return _inst(s.ty, v -> get(subst, v, v))
end
instantiate(t::Ty, ::Int, factory=newtyvar) = t # for concrete types, do nothing

function _inst(t::Ty,
               mapvar,
               binders::Dict{TyVar,Nothing}=Dict{TyVar,Nothing}(),
               resolving::Dict{TyVar,Nothing}=Dict{TyVar,Nothing}(),
               memo::_TypeRewriteMemo=_TypeRewriteMemo(),
               scope::Base.RefValue{Int}=Ref(0),
               next_scope::Base.RefValue{Int}=Ref(0))
    if t isa TyVar
        haskey(binders, t) && return t
        if t.instance !== nothing
            haskey(resolving, t) && return t
            resolving[t] = nothing
            try
                return _inst(t.instance, mapvar, binders, resolving, memo, scope, next_scope)
            finally
                delete!(resolving, t)
            end
        end
        return mapvar(t)
    end

    children = type_children(t)
    isempty(children) && return t
    known = _rewrite_lookup(memo, t, scope[])
    known === nothing || return known

    if t isa TyMu
        outer_scope = scope[]
        next_scope[] += 1
        scope[] = next_scope[]
        had_binder = haskey(binders, t.var)
        binders[t.var] = nothing
        body = try
            _inst(t.body, mapvar, binders, resolving, memo, scope, next_scope)
        finally
            scope[] = outer_scope
            had_binder || delete!(binders, t.var)
        end
        value = body === t.body ? t : TyMu(t.var, body)
        return _rewrite_remember!(memo, t, outer_scope, value)
    end

    rewritten = _rewrite_children(
        children,
        child -> _inst(child, mapvar, binders, resolving, memo, scope, next_scope),
    )
    value = all(i -> rewritten[i] === children[i], eachindex(children)) ?
        t : rebuild_type(t, rewritten)
    return _rewrite_remember!(memo, t, scope[], value)
end

function _query_copy(t::Ty, factory, mapping::Dict{TyVar,TyVar})
    return _inst(t, var -> get!(mapping, var) do
        factory(var.level)
    end)
end

"""Detach one joint context/target query while preserving variable sharing."""
function _detached_type_inputs(
    type_ctx::AbstractDict{Symbol,<:TyLike},
    extras::TyLike...,
)
    factory = make_tyvar_factory()
    variables = Dict{TyVar,TyVar}()
    mapvar(variable::TyVar) = get!(variables, variable) do
        factory(variable.level)
    end
    function detach(ty::TyLike)
        if ty isa TyScheme
            return TyScheme(
                TyVar[mapvar(variable) for variable in ty.vars],
                _inst(ty.ty, mapvar),
            )
        end
        return _inst(ty, mapvar)
    end
    detached = Dict{Symbol,TyLike}()
    for (name, ty) in type_ctx
        detached[name] = detach(ty)
    end
    return detached, map(detach, extras)
end

"""Detach a type context for an isolated query without a target."""
function _detached_type_ctx(type_ctx::AbstractDict{Symbol,<:TyLike})
    detached, _ = _detached_type_inputs(type_ctx)
    return detached
end

"""
    iscompatible(a::TyLike, b::TyLike) -> Bool

Return `true` iff `a` and `b` can be unified (up to instantiation).

Implementation note:
- `unify!` binds `TyVar.instance`. For compatibility checks used by generation/mutation/crossover,
  we treat this as a *pure* predicate by unifying **fresh copies** of instantiated types.
"""
function iscompatible(a::TyLike, b::TyLike)::Bool
    factory = make_tyvar_factory()
    t1 = instantiate(a, 0, factory)
    t2 = instantiate(b, 0, factory)
    mapping = Dict{TyVar,TyVar}()
    t1 = _query_copy(t1, factory, mapping)
    t2 = _query_copy(t2, factory, mapping)
    return isnothing(unify!(t1, t2))
end

# Exact compatibility for a small, closed built-in fragment. The fuel bounds
# only this fast path: every other type keeps the full unification semantics.
function _grounded_closed_fragment_fuel(ty::Ty, fuel::Int)::Int
    if ty isa TyConst
        fuel > 0 || return -1
        return fuel - 1
    elseif ty isa TyFun
        fuel > 0 || return -1
        fuel -= 1
        fuel = _grounded_closed_fragment_fuel(ty.arg, fuel)
        fuel < 0 && return -1
        return _grounded_closed_fragment_fuel(ty.res, fuel)
    elseif ty isa TyList
        fuel > 0 || return -1
        return _grounded_closed_fragment_fuel(ty.elem, fuel - 1)
    elseif ty isa TyArray
        ty.len isa TyNat || return -1
        fuel > 1 || return -1
        return _grounded_closed_fragment_fuel(ty.elem, fuel - 2)
    end
    # Variables (even resolved ones), μ, rigid, products, sums, schemes and
    # custom constructors require the original compatibility predicate.
    return -1
end

function _grounded_exact_closed_tree_equal(left::Ty, right::Ty)::Bool
    if left isa TyConst
        right isa TyConst || return false
        return left.name == right.name
    elseif left isa TyFun
        right isa TyFun || return false
        return _grounded_exact_closed_tree_equal(left.arg, right.arg) &&
               _grounded_exact_closed_tree_equal(left.res, right.res)
    elseif left isa TyList
        right isa TyList || return false
        return _grounded_exact_closed_tree_equal(left.elem, right.elem)
    elseif left isa TyArray
        right isa TyArray || return false
        left_length = left.len::TyNat
        right_length = right.len::TyNat
        return left_length.value == right_length.value &&
               _grounded_exact_closed_tree_equal(left.elem, right.elem)
    end
    return false
end

"""
    _grounded_iscompatible(left, right)

Use direct comparison only when both operands are certified closed built-in
trees within one shared 64-node fuel. Unknown constructors and exhausted fuel
fall back to `iscompatible` with the original operands. Fuel is not an
admission or type-size limit.
"""
@inline function _grounded_iscompatible(left::TyLike, right::TyLike)::Bool
    (left isa Ty && right isa Ty) || return iscompatible(left, right)
    fuel = _grounded_closed_fragment_fuel(left, 64)
    fuel < 0 && return iscompatible(left, right)
    fuel = _grounded_closed_fragment_fuel(right, fuel)
    fuel < 0 && return iscompatible(left, right)
    return _grounded_exact_closed_tree_equal(left, right)
end

"""
    typekey(t::TyLike) -> Any

Canonical, α-invariant structural key for `Ty`/`TyScheme`.

This is intended for **indexing and caching**, not for user-facing printing:
- Free type variables are numbered by first occurrence order (`(:v, 0)`, `(:v, 1)`, ...).
- Equi-recursive μ-types are represented using De Bruijn-style references (`(:mu_ref, depth)`).

The returned value is an opaque tuple containing hash-consed DAG records. It is
safe to use as a dictionary key without expanding shared subgraphs.
"""
function typekey(t::TyLike)
    if t isa TyScheme
        # First-occurrence numbering makes quantified names and order irrelevant
        # without allocating fresh variables or consulting global state.
        return _typekey(t.ty)
    else
        return _typekey(t)
    end
end

# A concrete atomic type is already a complete one-record DAG. This applies to
# every TyConst name, not only the built-in primitive-name set.
_typekey(t::TyConst) = (:const, :dag, 1, ((:const, (t.name,), ()),))

function _typekey(t::Ty)
    builder = _TypeKeyBuilder()
    root = _typekey_walk(builder, t)
    return (builder.records[root][1], :dag, root, Tuple(builder.records))
end

function _collect_rigid_ids!(
    ids::Set{UInt},
    ty::Ty,
    seen::IdDict{Any,Nothing}=IdDict{Any,Nothing}(),
)
    ty isa TyRigid && (push!(ids, ty.id); return ids)
    haskey(seen, ty) && return ids
    seen[ty] = nothing
    if ty isa TyVar && ty.instance !== nothing
        _collect_rigid_ids!(ids, ty.instance, seen)
        return ids
    end
    for child in type_children(ty)
        _collect_rigid_ids!(ids, child, seen)
    end
    return ids
end

function semantic_typekey(s::TyScheme)
    quantified = Dict{TyVar,Nothing}(var => nothing for var in s.vars)
    rigid = Dict{TyVar,TyRigid}()
    used_ids = _collect_rigid_ids!(Set{UInt}(), s.ty)
    next_id = Ref(UInt(0))
    body = _inst(s.ty, var -> begin
        haskey(quantified, var) || return var
        get!(rigid, var) do
            while true
                next_id[] += UInt(1)
                next_id[] in used_ids && continue
                push!(used_ids, next_id[])
                return TyRigid(next_id[])
            end
        end
    end)
    return (:scheme, length(s.vars), semantic_typekey(body))
end

_scheme_key(s::TyScheme) = (length(s.vars), _typekey(s.ty))

mutable struct _TypeKeyBuilder
    free_ids::Dict{TyVar,Int}
    mu_stack::Vector{TyVar}
    resolving::Dict{TyVar,Nothing}
    records::Vector{Tuple}
    interned::Dict{Any,Int}
    mutable_nodes::IdDict{Ty,Tuple{Int,Int}}
    keyed_nodes::Dict{Tuple{DataType,Any},Tuple{Int,Int}}
    active::IdDict{Ty,Nothing}
    scope::Int
    next_scope::Int
end

_TypeKeyBuilder() = _TypeKeyBuilder(
    Dict{TyVar,Int}(),
    TyVar[],
    Dict{TyVar,Nothing}(),
    Tuple[],
    Dict{Any,Int}(),
    IdDict{Ty,Tuple{Int,Int}}(),
    Dict{Tuple{DataType,Any},Tuple{Int,Int}}(),
    IdDict{Ty,Nothing}(),
    0,
    0,
)

function _intern_typekey!(builder::_TypeKeyBuilder, tag::Symbol, parameters::Tuple, children::Tuple)
    record = (tag, parameters, children)
    return get!(builder.interned, record) do
        push!(builder.records, record)
        length(builder.records)
    end
end

@inline function _typekey_source(builder::_TypeKeyBuilder, t::Ty, scope::Int)
    if Base.ismutabletype(typeof(t))
        entry = get(builder.mutable_nodes, t, nothing)
        return entry !== nothing && entry[1] == scope ? entry[2] : 0
    end
    identity = type_node_identity(t)
    identity === nothing && return 0
    entry = get(builder.keyed_nodes, (typeof(t), identity), nothing)
    return entry !== nothing && entry[1] == scope ? entry[2] : 0
end

@inline function _remember_typekey_source!(builder::_TypeKeyBuilder, t::Ty, scope::Int, key::Int)
    if Base.ismutabletype(typeof(t))
        builder.mutable_nodes[t] = (scope, key)
    else
        identity = type_node_identity(t)
        identity === nothing || (builder.keyed_nodes[(typeof(t), identity)] = (scope, key))
    end
    return key
end

function _typekey_children(builder::_TypeKeyBuilder, children::Tuple)
    child_keys = Int[]
    for i in eachindex(children)
        duplicate = findfirst(j -> children[j] === children[i], 1:(i - 1))
        push!(child_keys, duplicate === nothing ?
            _typekey_walk(builder, children[i]) : child_keys[duplicate])
    end
    return tuple(child_keys...)
end

function _typekey_walk(builder::_TypeKeyBuilder, t::Ty)::Int
    if t isa TyVar
        v = t::TyVar
        # μ-binder reference has priority over instance-following to avoid loops like v.instance == μv. ...
        if !isempty(builder.mu_stack)
            j = findlast(x -> x == v, builder.mu_stack)
            if j !== nothing
                depth = length(builder.mu_stack) - j # top binder => 0
                return _intern_typekey!(builder, :mu_ref, (depth,), ())
            end
        end
        if v.instance !== nothing
            haskey(builder.resolving, v) &&
                return _intern_typekey!(builder, :cycle, (), ())
            builder.resolving[v] = nothing
            try
                return _typekey_walk(builder, v.instance)
            finally
                delete!(builder.resolving, v)
            end
        end
        idx = get!(builder.free_ids, v) do
            length(builder.free_ids)
        end
        return _intern_typekey!(builder, :v, (idx,), ())
    elseif t isa TyRigid
        return _intern_typekey!(builder, :rigid, (t.id,), ())
    end

    known = _typekey_source(builder, t, builder.scope)
    known != 0 && return known
    haskey(builder.active, t) && return _intern_typekey!(builder, :cycle, (), ())
    builder.active[t] = nothing

    scope = builder.scope
    key = try
        if t isa TyConst
            _intern_typekey!(builder, :const, ((t::TyConst).name,), ())
        elseif t isa TyNat
            _intern_typekey!(builder, :nat, ((t::TyNat).value,), ())
        elseif t isa TyMu
            m = t::TyMu
            builder.next_scope += 1
            builder.scope = builder.next_scope
            push!(builder.mu_stack, m.var)
            body_key = try
                _typekey_walk(builder, m.body)
            finally
                pop!(builder.mu_stack)
                builder.scope = scope
            end
            _intern_typekey!(builder, :mu, (), (body_key,))
        else
            children = _typekey_children(builder, type_children(t))
            if t isa TyFun
                _intern_typekey!(builder, :fun, (), children)
            elseif t isa TyProd
                _intern_typekey!(builder, :prod, (), children)
            elseif t isa TySum
                _intern_typekey!(builder, :sum, (), children)
            elseif t isa TyList
                _intern_typekey!(builder, :list, (), children)
            elseif t isa TyArray
                _intern_typekey!(builder, :array, (), children)
            else
                _intern_typekey!(
                    builder,
                    :constructor,
                    (type_constructor_tag(t), type_static_parameters(t)),
                    children,
                )
            end
        end
    finally
        delete!(builder.active, t)
    end
    return _remember_typekey_source!(builder, t, scope, key)
end

"""
    typectx_fingerprint(ctx::AbstractDict{Symbol,<:TyLike}) -> UInt

Order-independent fingerprint for a typing context (by symbol key).

This is suitable for cache keys (e.g. typed site caches) where we need a stable identifier that changes
when any entry’s type changes.
"""
function typectx_fingerprint(ctx::AbstractDict{Symbol,<:TyLike})::UInt
    ks = sort!(collect(keys(ctx)))
    h = hash(:typectx, UInt(0))
    for k in ks
        h = hash(k, hash(semantic_typekey(ctx[k]), h))
    end
    return h
end
"""
    islesseq(a::TyLike, b::TyLike) -> Bool

Semantic subset check: returns `true` iff every value inhabiting `a` can
also inhabit `b`. Quantified variables in `b` are treated as flexible
(can be instantiated) while free variables in `a` are rigid (must remain
unmodified). The function succeeds only when unification can be achieved
by binding *only* the flexible variables.
"""
function islesseq(subset_ty::TyLike, superset_ty::TyLike)::Bool
    factory = make_tyvar_factory()
    # The superset is flexible: its quantified variables may be specialised.
    flex_superset = instantiate(superset_ty, 0, factory)
    flex_superset = _query_copy(flex_superset, factory, Dict{TyVar,TyVar}())

    # The subset is rigid during this query. Written μ binders are protected by
    # `_inst`; only genuinely free/quantified variables become rigid atoms.
    source = subset_ty isa TyScheme ? subset_ty.ty : subset_ty
    quantified = if subset_ty isa TyScheme
        ids = Dict{TyVar,Nothing}()
        foreach(v -> ids[v] = nothing, subset_ty.vars)
        ids
    else
        nothing
    end
    rigid_ids = Dict{TyVar,TyRigid}()
    next_rigid = Ref(UInt(0))
    rigid_subset = _inst(source, var -> begin
        should_rigidify = quantified === nothing || haskey(quantified, var)
        should_rigidify || return var
        get!(rigid_ids, var) do
            next_rigid[] += UInt(1)
            TyRigid(next_rigid[])
        end
    end)

    return isnothing(unify!(flex_superset, rigid_subset))
end

function collect_free_vars(t::Ty)::Set{TyVar}
    vars = Set{TyVar}()
    seen = Dict{TyVar,Nothing}()
    binders = Dict{TyVar,Nothing}()
    mutable_nodes = IdDict{Ty,Int}()
    keyed_nodes = Dict{Tuple{DataType,Any},Int}()
    scope = Ref(0)
    next_scope = Ref(0)

    function visited(t::Ty)
        if Base.ismutabletype(typeof(t))
            get(mutable_nodes, t, -1) == scope[] && return true
            mutable_nodes[t] = scope[]
            return false
        end
        identity = type_node_identity(t)
        identity === nothing && return false
        key = (typeof(t), identity)
        get(keyed_nodes, key, -1) == scope[] && return true
        keyed_nodes[key] = scope[]
        return false
    end

    function walk(t::Ty)
        if t isa TyVar
            haskey(binders, t) && return nothing
            haskey(seen, t) && return nothing
            seen[t] = nothing
            if t.instance === nothing
                push!(vars, t)
            else
                walk(t.instance)
            end
        elseif t isa TyMu
            visited(t) && return nothing
            outer_scope = scope[]
            next_scope[] += 1
            scope[] = next_scope[]
            had_binder = haskey(binders, t.var)
            binders[t.var] = nothing
            try
                walk(t.body)
            finally
                scope[] = outer_scope
                had_binder || delete!(binders, t.var)
            end
        else
            visited(t) && return nothing
            children = type_children(t)
            for i in eachindex(children)
                any(j -> children[j] === children[i], 1:(i - 1)) || walk(children[i])
            end
        end
        return nothing
    end

    walk(t)
    return vars
end 

function generalize(t::Ty, level::Int)::TyScheme
    seen = collect_free_vars(t)
    vars = [v for v in seen if v.level > level]
    return TyScheme(vars, t)
end

# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------
const _EnvName = Union{Nothing,Symbol}

struct Env
    stack::Vector{Pair{_EnvName,TyScheme}}
end
Env() = Env(Pair{_EnvName,TyScheme}[])

Base.push!(env::Env, name::_EnvName, scheme::TyScheme) = push!(env.stack, name => scheme)
Base.pop!(env::Env) = pop!(env.stack) 

function lookup(env::Env, idx::Int)
    pos = length(env.stack) - idx
    if pos < 1 || pos > length(env.stack)
        return Err(UnboundVariableError(Symbol("#$(idx)")))
    end
    return env.stack[pos].second
end

function lookup(env::Env, name::Symbol)
    for (n, sch) in Iterators.reverse(env.stack)
        if n isa Symbol && n == name
            return sch
        end
    end
    return Err(UnboundVariableError(name))
end

# ------------------------------------------------------------
# Main inference (returns Ty or throws TyError)
# ------------------------------------------------------------

"""Default [`term_view`](@ref) adapter for the package's `NodeASTs` representation."""
struct NodeTermAdapter end

"""Finite typing failure for a `FunctionDefNode` with a non-function head."""
struct NonFunctionDefinitionError
    inferred::Ty
end

"""
    term_view(adapter, node) -> Tuple

Expose one term node to Algorithm W as `(:const, name)`, `(:var, ref)`,
`(:abs, name, body)`, `(:app, func, arg)`, or `(:fix, name, body)`. Implementing this single
function for another finite term representation reuses the same inference driver.
Constant and named abstraction names are `Symbol`s; `ref` is a `Symbol` or a
zero-based De Bruijn `Int`. Anonymous `:fix` binders use `nothing` internally
so they cannot capture named variables.

An `:app` whose `func` has an `:abs` view denotes the let-like form `(λx. M) N`
and receives let generalisation rather than ordinary monomorphic application.
Sequential bindings are represented as nested let-like applications, so each
right-hand side is inferred and generalised before the following body.
`:fix` exposes one monomorphic self binder and equates its type with the body
type; callers may impose additional domain constraints on the result.
"""
function term_view end

"""Optional hook for adapters that retain the scheme of a real binder.

The default W path does not need binding records.  Occurrence-indexed
adapters may override this hook; it is called with the owning adapter node and
the exact scheme pushed into `Env`, before that scope is traversed.
"""
record_binding(::Any, owner, scheme::TyLike) = nothing

"""Adapter hook applied to a fixed-point result before metadata or let generalisation."""
constrain_fixpoint(adapter, node, ty::Ty, factory) = ty

term_view(::NodeTermAdapter, node::ConstNode) = (:const, node.name)
term_view(::NodeTermAdapter, node::VarNode) = (:var, node.name)
term_view(::NodeTermAdapter, node::IndexNode) = (:var, node.index)
term_view(::NodeTermAdapter, node::AbsNode) = (:abs, node.name, node.body)
term_view(::NodeTermAdapter, node::AppNode) = (:app, node.func, node.arg)
term_view(::NodeTermAdapter, node::FunctionDefNode) = (:fix, nothing, node.body)

function _constrain_function_head!(ty::Ty, factory)
    current = prune(ty)
    seen = IdSet{Any}()
    while current isa TyMu
        current in seen && return false
        push!(seen, current)
        current = prune(current.body)
    end
    if current isa TyVar
        constrained = TyFun(factory(current.level), factory(current.level))
        return isnothing(unify!(current, constrained))
    end
    return current isa TyFun
end

function constrain_fixpoint(
    ::NodeTermAdapter,
    ::FunctionDefNode,
    ty::Ty,
    factory,
)
    _constrain_function_head!(ty, factory) ||
        return Err(NonFunctionDefinitionError(prune(ty)))
    return prune(ty)
end

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
    return _infer_root(tree.root, ctx, NodeTermAdapter(), factory, meta, 0)
end

"""
    infer_term(root, ctx, adapter; factory=newtyvar, meta=nothing)

Infer a term exposed through [`term_view`](@ref). This is the composable entry
point; [`infer`](@ref) is its `AstTree` compatibility wrapper.
"""
function infer_term(root,
                    ctx::AbstractDict{Symbol,<:TyLike},
                    adapter;
                    factory=newtyvar,
                    meta=nothing)
    return _infer_root(root, ctx, adapter, factory, meta, 0)
end

"""Infer a built-in AST term using the shared [`NodeTermAdapter`](@ref)."""
function infer_term(
    root::Union{AstNode,AstTree},
    ctx::AbstractDict{Symbol,<:TyLike};
    factory=make_tyvar_factory(),
    meta=nothing,
)
    root_node = root isa AstTree ? root.root : root
    return _infer_root(root_node, ctx, NodeTermAdapter(), factory, meta, 0)
end

"""Infer and generalise a reusable top-level scheme through a term adapter."""
function infer_scheme_term(root,
                           ctx::AbstractDict{Symbol,<:TyLike},
                           adapter;
                           factory=newtyvar)
    ty = _infer_root(root, ctx, adapter, factory, nothing, 1)
    ty isa Err && return ty
    return generalize(ty, 0)
end

function _infer_root(root,
                     ctx::AbstractDict{Symbol,<:TyLike},
                     adapter,
                     factory,
                     meta,
                     level::Int)
    env = Env()
    ty = @✓(_infer(adapter, root, env, ctx, level, factory, meta))
    if meta !== nothing
        ty_min = prune(ty)
        # The root has no enclosing term binders. Its reusable upper bound is
        # therefore closed one region outside the inference query itself.
        ty_max = generalize(ty_min, level - 1)
        meta[root] = (ty_min, ty_max)
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
    ty = _infer_root(tree.root, ctx, NodeTermAdapter(), factory, nothing, 1)
    ty isa Err && return ty
    return generalize(ty, 0)
end

_record_meta(meta::Nothing, node, ty::Ty, level::Int, orig::TyLike=ty) = ty
function _record_meta(meta::AbstractDict, node, ty::Ty, level::Int, orig::TyLike=ty)
    ty_min = prune(ty)
    ty_max = orig isa TyScheme ? orig : generalize(ty_min, level)
    meta[node] = (ty_min, ty_max)
    return ty
end

function _infer(adapter,
                node,
                env::Env,
                ctx::AbstractDict{Symbol,<:TyLike},
                level::Int,
                factory,
                meta)
    view = term_view(adapter, node)
    kind = view[1]
    if kind === :const
        name = view[2]
        haskey(ctx, name) || return Err(UnboundVariableError(name))
        entry = ctx[name]
        ty = instantiate(entry, level, factory)
        return _record_meta(meta, node, ty, level, entry)
    elseif kind === :var
        sch = @✓ lookup(env, view[2])
        ty = instantiate(sch, level, factory)
        return _record_meta(meta, node, ty, level)
    elseif kind === :abs
        name, body = view[2], view[3]
        tv = factory(level)
        scheme = TyScheme(TyVar[], tv)
        push!(env, name, scheme)
        record_binding(adapter, node, scheme)
        body_ty = @✓ _infer(adapter, body, env, ctx, level, factory, meta)
        pop!(env)
        ty = TyFun(prune(tv), body_ty)
        return _record_meta(meta, node, ty, level)
    elseif kind === :app
        func, arg = view[2], view[3]
        func_view = term_view(adapter, func)
        if func_view[1] === :abs # let-like `(λx. M) N`
            arg_ty = @✓ _infer(adapter, arg, env, ctx, level + 1, factory, meta)
            arg_ty = generalize(arg_ty, level)
            push!(env, func_view[2], arg_ty)
            record_binding(adapter, func, arg_ty)
            body_ty = @✓ _infer(adapter, func_view[3], env, ctx, level, factory, meta)
            pop!(env)
            return _record_meta(meta, node, body_ty, level)
        else
            fun_ty = @✓ _infer(adapter, func, env, ctx, level, factory, meta)
            arg_ty = @✓ _infer(adapter, arg, env, ctx, level, factory, meta)
            res_ty = factory(level)
            @✓ unify!(fun_ty, TyFun(arg_ty, res_ty))
            return _record_meta(meta, node, res_ty, level)
        end
    elseif kind === :fix
        name, body = view[2], view[3]
        self_ty = factory(level)
        scheme = TyScheme(TyVar[], self_ty)
        push!(env, name, scheme)
        record_binding(adapter, node, scheme)
        body_ty = @✓ _infer(adapter, body, env, ctx, level, factory, meta)
        pop!(env)
        @✓ unify!(self_ty, body_ty)
        constrained_ty = @✓ constrain_fixpoint(adapter, node, body_ty, factory)
        return _record_meta(meta, node, constrained_ty, level)
    else
        throw(ArgumentError("unknown term view kind $kind"))
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

# General host-value atoms remain part of the type layer.  Concrete language
# primitives and their signatures live in `Primitives`.
const BoolTy  = TyConst(:Bool)
const IntTy   = TyConst(:Int)
const FloatTy = TyConst(:Float)

get_type(::Bool) = BoolTy
get_type(::Int) = IntTy
get_type(::Float64) = FloatTy
get_type(::Float32) = FloatTy
get_type(t::Tuple{T,S}) where {T,S} = get_type(t[1]) × get_type(t[2])

# ------------------------------------------------------------
# Test items (basic sanity)
# ------------------------------------------------------------
@testsnippet TSBaseImports begin
    using LambdaRegression.TypeSystem: TyConst, TyFun, TyProd, TySum, →, ×
    using LambdaRegression.TypeSystem: IntTy, TyFun
    using LambdaRegression.TypeSystem
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: make_tyvar_factory
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: infer, infer_scheme, get_type_ctx, get_ast_type_ctx, TyLike, TyScheme
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, IndexNode
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.TypeSystem: newtyvar, unify!
    using LambdaRegression.TypeSystem: IntTy, get_type_ctx
    using LambdaRegression.Primitives: IEQ, ISUB, IMUL, COND
    using LambdaRegression.TypeSystem: collect_free_vars, prune
    using LambdaRegression.Execution: prepare
    using LambdaRegression.TypeSystem: IntTy, get_type_ctx
    using LambdaRegression.TypeSystem: IntTy, get_type_ctx
    using LambdaRegression.Primitives: IMUL, ISUB, IEQ, COND
    using LambdaRegression.Primitives: COND
    using LambdaRegression.Primitives: FST, PAIR
    using LambdaRegression.Primitives: INL, INR
    using LambdaRegression.TypeSystem: newtyvar, unify!, prune, get_type
    using LambdaRegression.Primitives: TOLIST
    using LambdaRegression.TypeSystem: TyArray, TyNat, FloatTy, unify!, get_type
    using LambdaRegression.Execution: prepare, beta_reduce!
    using LambdaRegression.TypeSystem: newtyvar, TyLike, IntTy, TyList, get_type_ctx
    using LambdaRegression.Primitives: NilErr, CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND
    using LambdaRegression.TypeSystem: IntTy, get_type_ctx
    using LambdaRegression.TypeSystem: newtyvar, TyFun, iscompatible, IntTy
    using LambdaRegression.TypeSystem: newtyvar, TyFun, typekey, unify!, prune, TyMu, IntTy
    using LambdaRegression.TypeSystem: typectx_fingerprint, TyLike, IntTy, BoolTy
end

@testitem "infix operators" setup=[TSBaseImports] begin
    int_ty = TyConst(:Int)
    bool_ty = TyConst(:Bool)
    fun_ty = int_ty → bool_ty
    @test fun_ty isa TyFun && fun_ty.arg === int_ty && fun_ty.res === bool_ty
    remembered = Dict(fun_ty => :present)
    @test_throws ArgumentError fun_ty.arg = bool_ty
    @test remembered[fun_ty] == :present
    prod_ty = int_ty × bool_ty
    @test prod_ty isa TyProd && prod_ty.left === int_ty && prod_ty.right === bool_ty
    sum_ty = int_ty + bool_ty
    @test sum_ty isa TySum && sum_ty.left === int_ty && sum_ty.right === bool_ty
end

@testitem "TypeSystem pretty print arrow precedence" setup=[TSBaseImports] begin
    # `(A→B)→C` must print with parentheses on the argument side.
    arrow = string(Char(0x2192))
    t = TyFun(TyFun(IntTy, IntTy), TyFun(IntTy, IntTy))
    s = replace(sprint(show, t), " " => "")
    @test s == "(Int$(arrow)Int)$(arrow)Int$(arrow)Int"
end

@testitem "simple hashing and equality" setup=[TSBaseImports] begin
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

@testitem "TypeSystem make_tyvar_factory is detached" setup=[TSBaseImports] begin
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

@testitem "simple identity typing" setup=[TSBaseImports] begin
    # term: (λx. x)
    term = AbsNode(:x, IndexNode(0))
    tree = AstTree(term)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)
    @test ty isa TyFun && ty.arg === ty.res
end

@testitem "named binders use lexical shadowing" setup=[TSBaseImports] begin
    tree = AstTree(AbsNode(:x, AbsNode(:x, VarNode(:x))))
    ty = infer(tree, Dict{Symbol,TyLike}())
    @test ty isa TyFun
    @test ty.res isa TyFun
    @test ty.arg != ty.res.arg
    @test ty.res.arg == ty.res.res
end

@testitem "infer_scheme and get_ast_type_ctx" setup=[TSBaseImports] begin
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

@testitem "self-application typable via μ-type" setup=[TSBaseImports] begin
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

@testitem "application polymorphism (λx.x) id" setup=[TSBaseImports] begin
    id_fun = AbsNode(:x, IndexNode(0))         # λx. x

    app_expr = AppNode(id_fun, id_fun)         # (λx.x) (λx.x)
    tree = AstTree(app_expr)
    ctx = Dict{Symbol,TyConst}()
    ty = infer(tree, ctx)

    # Expected: type of argument, i.e., σ → σ where σ fresh
    @test ty isa TyFun && ty.arg === ty.res
end

@testitem "positive μ-type via unify" setup=[TSBaseImports] begin
    α = newtyvar(0)
    int_ty = TyConst(:Int)
    unify!(α, TyFun(int_ty, α))
    # α should now be folded to μ-type
    @test α.instance isa TyMu
end

@testitem "list occurs check contractiveness" setup=[TSBaseImports] begin
    α = newtyvar(0)
    result = unify!(α, TyList(α))
    @test isnothing(result)
    @test α.instance isa TyMu
    @test α.instance.body isa TyList
    @test α.instance.var !== α
    @test α.instance.var.instance === nothing
    @test α.instance.body.elem === α.instance.var
end

@testitem "factorial via Y_int" setup=[TSBaseImports] begin
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

@testitem "Linked TyVar instances" setup=[TSBaseImports] begin
    α = newtyvar(0)
    β = newtyvar(0)
    mu = μ(α, β→α)
    α.instance = mu

    a = prune(α)

    vs = collect_free_vars(α)
end

@testitem "Y combinator" setup=[TSBaseImports] begin
    const_ctx = @terms(cond=COND, eq=IEQ, mul=IMUL, sub=ISUB, int0=0, int1=1, int4=4)

    merge!(const_ctx, @terms(Y=parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", const_ctx)))
    merge!(const_ctx, @terms(fac=parse_lambda("Y (λfac.λn. cond (eq n int0) int1 (mul n (fac (sub n int1))))", const_ctx)))
    fac4 = parse_lambda("fac int4", const_ctx)
    fac4 = prepare(fac4)

    tyctx = get_type_ctx(const_ctx)
    ty = infer(fac4, tyctx)
    @test ty == IntTy
end

@testitem "Y combinator infers Fibonacci and safe list sum" setup=[TSBaseImports] begin
    terms = @terms(
        COND=LambdaRegression.Primitives.COND,
        IEQ=LambdaRegression.Primitives.IEQ,
        IADD=LambdaRegression.Primitives.IADD,
        ISUB=LambdaRegression.Primitives.ISUB,
        UNCONS_CASE=LambdaRegression.Primitives.UNCONS_CASE,
        int0=0,
        int1=1,
    )
    terms[:Y] = parse_lambda("λf.(λx.f (x x)) (λx.f (x x))", terms)
    type_ctx = get_type_ctx(terms)
    get_ast_type_ctx(terms, type_ctx)

    fibonacci = parse_lambda(
        "Y (λfib.λn. COND (IEQ n int0) int0 (COND (IEQ n int1) int1 (IADD (fib (ISUB n int1)) (fib (ISUB (ISUB n int1) int1)))))",
        terms,
    )
    list_sum = parse_lambda(
        "Y (λsum.λlist. UNCONS_CASE list (λhead.λtail. IADD head (sum tail)) int0)",
        terms,
    )

    @test iscompatible(infer(fibonacci, type_ctx), TyFun(IntTy, IntTy))
    @test iscompatible(infer(list_sum, type_ctx), TyFun(TyList(IntTy), IntTy))
end

@testitem "polymorphic if constant" setup=[TSBaseImports] begin
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

@testitem "product unify and projection" setup=[TSBaseImports] begin
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

@testitem "sum inl inr case" setup=[TSBaseImports] begin
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

@testitem "TyArray length polymorphism unify" setup=[TSBaseImports] begin
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

@testitem "TypeSystem UNCONS_CASE typing and runtime" setup=[TSBaseImports] begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyList, get_type
    using LambdaRegression.Primitives: UNCONS_CASE

    @test get_type(UNCONS_CASE) isa LambdaRegression.TypeSystem.TyScheme
    @test UNCONS_CASE([1.0, 2.0])(x -> xs -> x)(0.0) == 1.0
    @test UNCONS_CASE(Float64[])(x -> xs -> x)(0.0) == 0.0
end

@testitem "TypeSystem Array HO primitives typing" setup=[TSBaseImports] begin
    A = TyArray(TyNat(3), FloatTy)

    map_ty = TypeSystem.instantiate(get_type(LambdaRegression.Primitives.AMAP), 0)
    @test unify!(map_ty, (FloatTy → FloatTy) → A → A) === nothing

    fold_ty = TypeSystem.instantiate(get_type(LambdaRegression.Primitives.AFOLDL), 0)
    @test unify!(fold_ty, (FloatTy → FloatTy → FloatTy) → FloatTy → A → FloatTy) === nothing

    lenf_ty = TypeSystem.instantiate(get_type(LambdaRegression.Primitives.LENF), 0)
    @test unify!(lenf_ty, A → FloatTy) === nothing
end

@testitem "TypeSystem Array HO primitives preserve dense NDArray nesting" begin
    using Test
    using LambdaRegression.Primitives: AMAP, AFOLDL

    matrix = [1.0 2.0 3.0; 4.0 5.0 6.0]

    @test AMAP(row -> sum(row))(matrix) == [6.0, 15.0]
    @test AMAP(AMAP(x -> 2x))(matrix) == 2 .* matrix
    @test AFOLDL(acc -> row -> acc .+ row)(zeros(3))(matrix) == [5.0, 7.0, 9.0]
end

# ------------------------------------------------------------
# Advanced type invariance tests --------------------

@testitem "factorial via Y combinator - type invariance under reduction" setup=[TSBaseImports] begin
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

@testitem "list square-sum - type invariance under reduction" setup=[TSBaseImports] begin
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

@testitem "islesseq subset polymorphism" setup=[TSBaseImports] begin
    int_ty = TyConst(:Int)
    α = newtyvar(0)
    scheme_any = ∀([α], α)
    @test islesseq(int_ty, scheme_any)
    @test !islesseq(scheme_any, int_ty)
end

@testitem "islesseq rigidity and function" setup=[TSBaseImports] begin
    int_ty, β, γ = TyConst(:Int), newtyvar(0), newtyvar(0)
    scheme_fun = ∀([β], β → β)
    @test islesseq(int_ty → int_ty, scheme_fun)
    @test !islesseq(scheme_fun, int_ty → int_ty)
    @test !islesseq(γ, int_ty)          # rigid var `γ` on subset side cannot be bound
    @test islesseq(int_ty, γ)           # superset `γ` is flexible and can be bound
end

@testitem "islesseq advanced subtyping" setup=[TSBaseImports] begin
    int_ty, bool_ty = TyConst(:Int), TyConst(:Bool)
    α, β = newtyvar(0), newtyvar(0)

    # Scheme vs Scheme
    id_scheme1, id_scheme2 = ∀([α], α → α), ∀([β], β → β)
    @test islesseq(id_scheme1, id_scheme2) && islesseq(id_scheme2, id_scheme1)
    int_fun_scheme = ∀([α], α → int_ty)
    @test !islesseq(int_fun_scheme, id_scheme1) # (α→Int) ⊈ (β→β)
    @test !islesseq(id_scheme1, int_fun_scheme) # (β→β) ⊈ (α→Int)

    # Mu vs Mu (equi-recursive) & Mu vs Scheme
    mu_int1, mu_int2 = μ(α, α → int_ty), μ(β, β → int_ty)
    @test islesseq(mu_int1, mu_int2) && islesseq(mu_int2, mu_int1)
    mu_bool = μ(α, α → bool_ty)
    @test !islesseq(mu_int1, mu_bool)
    any_scheme = ∀([β], β)
    @test islesseq(mu_int1, any_scheme)      # (μ α. α→Int) ⊆ (∀β. β)
    @test !islesseq(any_scheme, mu_int1)     # (∀β. β) ⊈ (μ α. α→Int)
end

@testitem "meta bounds recording" setup=[TSBaseImports] begin
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
    @test length(max_fun.vars) == 1
    @test collect_free_vars(max_fun.ty) == Set(max_fun.vars)

    # ---- TyMu ----
    meta_mu = infer_with_meta(self_app)
    @test haskey(meta_mu, self_app)
    min_mu, max_mu = meta_mu[self_app]
    @test min_mu isa TyFun && min_mu.arg isa TyMu
    @test max_mu isa TyScheme
end

@testitem "iscompatible is side-effect free on input types" setup=[TSBaseImports] begin
    α = newtyvar(0)
    β = newtyvar(0)
    t = TyFun(α, β)
    @test α.instance === nothing
    @test β.instance === nothing

    @test iscompatible(t, TyFun(IntTy, IntTy))

    # Must remain unbound: iscompatible unifies on fresh copies.
    @test α.instance === nothing
    @test β.instance === nothing

    bound = newtyvar(0)
    stale = deepcopy(bound)
    @test stale === bound
    @test isnothing(unify!(bound, IntTy))
    @test iscompatible(TyProd(bound, stale), TyProd(IntTy, IntTy))
    @test !iscompatible(TyProd(bound, stale), TyProd(IntTy, BoolTy))
    @test stale.instance === IntTy

    solved = newtyvar(0)
    @test isnothing(unify!(solved, IntTy))
    @test TypeSystem.islesseq(solved, IntTy)
    @test TypeSystem.islesseq(solved, IntTy) == TypeSystem.islesseq(prune(solved), IntTy)
end

@testitem "infer reports missing constants through Err" setup=[TSBaseImports] begin
    result = infer(AstTree(ConstNode(:missing)), Dict{Symbol,TyLike}())
    @test result isa Err
    @test result.value isa TypeSystem.UnboundVariableError
    @test result.value.name == :missing
end

@testitem "infer reports unbound named lexical variables through Err" setup=[TSBaseImports] begin
    result = infer(AstTree(VarNode(:missing)), Dict{Symbol,TyLike}(:missing => IntTy))
    @test result isa Err
    @test result.value isa TypeSystem.UnboundVariableError
    @test result.value.name == :missing
end

@testitem "typekey alpha invariance and mu termination" setup=[TSBaseImports] begin
    # α-invariance for free vars (by first-occurrence numbering).
    a1, b1 = newtyvar(0), newtyvar(0)
    a2, b2 = newtyvar(0), newtyvar(0)
    k1 = typekey(TyFun(a1, b1))
    k2 = typekey(TyFun(a2, b2))
    @test k1 == k2

    # Repeated occurrences matter: (α→α) is not the same as (α→β).
    a3, b3 = newtyvar(0), newtyvar(0)
    @test typekey(TyFun(a3, a3)) != typekey(TyFun(a3, b3))

    # Instantiated variables are followed without touching the recursion-stack guard.
    bound = newtyvar(0)
    @test isnothing(unify!(bound, FloatTy))
    @test typekey(bound) == typekey(FloatTy)

    # Shared concrete substructure is not a recursive cycle. This protects
    # generation caches for targets such as Matrix32 -> Matrix32 vs Matrix32 -> Float.
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    @test typekey(TyFun(Matrix32, Matrix32)) != typekey(TyFun(Matrix32, FloatTy))

    # μ-type key must terminate and be stable.
    v = newtyvar(0)
    @test isnothing(unify!(v, TyFun(v, IntTy)))
    mu = prune(v)
    @test mu isa TyMu
    kmu = typekey(mu)
    @test kmu[1] == :mu
end

@testitem "typekey TyConst leaf fast path matches generic builder" setup=[TSBaseImports] begin
    using Test
    using LambdaRegression.TypeSystem:
        TyArray,
        TyConst,
        TyFun,
        TyList,
        TyProd,
        TyScheme,
        TySum,
        TyVar,
        _TypeKeyBuilder,
        _typekey_walk,
        make_tyvar_factory

    TS = LambdaRegression.TypeSystem
    generic_key = function (ty)
        builder = _TypeKeyBuilder()
        root = _typekey_walk(builder, ty)
        return (builder.records[root][1], :dag, root, Tuple(builder.records))
    end
    compare_with_generic = function (ty)
        actual = TS.typekey(ty)
        expected = generic_key(ty isa TyScheme ? ty.ty : ty)
        @test isequal(actual, expected)
        @test actual == expected
        @test hash(actual) == hash(expected)
    end

    atoms = TyConst[
        TS.BoolTy,
        TS.FloatTy,
        TyConst(:arbitrary_external_atom),
        TyConst(Symbol("punctuation/and spaces")),
    ]
    for atom in atoms
        compare_with_generic(atom)
        @test TS.typekey(atom) == (:const, :dag, 1, ((:const, (atom.name,), ()),))

        empty_scheme = TyScheme(TyVar[], atom)
        unused_scheme = TyScheme([make_tyvar_factory()(0)], atom)
        compare_with_generic(empty_scheme)
        compare_with_generic(unused_scheme)
        @test TS.typekey(empty_scheme) == TS.typekey(atom)
        @test TS.typekey(unused_scheme) == TS.typekey(atom)
    end

    composites = (
        TyFun(atoms[3], TyList(atoms[4])),
        TyArray(TS.TyNat(7), atoms[2]),
        TyProd(atoms[1], TySum(atoms[4], atoms[3])),
    )
    for composite in composites
        compare_with_generic(composite)
    end
end

@testitem "typectx_fingerprint is order independent" setup=[TSBaseImports] begin
    a = Dict{Symbol,TyLike}(:x => IntTy, :y => BoolTy)
    b = Dict{Symbol,TyLike}(:y => BoolTy, :x => IntTy)
    @test typectx_fingerprint(a) == typectx_fingerprint(b)
end

@testitem "boolean primitives expose true false and not" setup=[TSBaseImports] begin
    using LambdaRegression.TypeSystem: BoolTy, get_type, get_type_ctx, →
    using LambdaRegression.Primitives: TRUE, FALSE, NOT

    @test TRUE === true
    @test FALSE === false
    @test NOT(TRUE) === false
    @test NOT(FALSE) === true
    @test get_type(TRUE) === BoolTy
    @test get_type(FALSE) === BoolTy
    @test get_type(NOT) == (BoolTy → BoolTy)

    ctx = get_type_ctx(Dict{Symbol,Any}(:TRUE => TRUE, :FALSE => FALSE, :NOT => NOT))
    @test ctx[:TRUE] === BoolTy
    @test ctx[:FALSE] === BoolTy
    @test ctx[:NOT] == (BoolTy → BoolTy)
end

@testitem "joint detached context and target preserve sharing" setup=[TSBaseImports] begin
    using LambdaRegression.TypeSystem: _detached_type_inputs
    α = newtyvar(0)
    open = TyFun(α, α)
    context = Dict{Symbol,TyLike}(:f => open)
    detached_one, targets_one = _detached_type_inputs(context, open)
    @test detached_one[:f].arg === targets_one[1].arg
    @test α.instance === nothing
    @test iscompatible(detached_one[:f], TyFun(IntTy, IntTy))
    @test α.instance === nothing

    detached_two, targets_two = _detached_type_inputs(context, open)
    @test detached_two[:f].arg === targets_two[1].arg
    @test iscompatible(detached_two[:f], TyFun(BoolTy, BoolTy))
    @test α.instance === nothing
end

end # module TypeSystem
