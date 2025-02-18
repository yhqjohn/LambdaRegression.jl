"""Exact μ-free directed retrieval for closed typed crossover sites.

The implementation deliberately keeps the first seam small: a site carries a
stable id and a demand type, while the donor is an offer.  Only quantified
variables of the offer are flexible; every demand variable and every shared
ambient variable is copied as a rigid atom before the local unification.
"""

using TestItems
using ...Utils: Err
using ...TypeSystem:
    Ty, TyLike, TyVar, TyMu, TyScheme, TyArray, TyNat, TyRigid,
    make_tyvar_factory, _inst, unify!, type_children

export ClosedDemandSite, closed_directed_match, closed_retrieval_scan
export ClosedRetrievalMuErr, ClosedRetrievalArraySortErr

struct ClosedDemandSite
    id::Int
    demand::TyLike
end

ClosedDemandSite(id::Integer, demand::TyLike) = ClosedDemandSite(Int(id), demand)

"""Explicit rejection for a μ-bearing closed retrieval query."""
struct ClosedRetrievalMuErr
    side::Symbol
end

"""Explicit rejection for a variable used at both array-length and type sorts."""
struct ClosedRetrievalArraySortErr
    side::Symbol
    variable::TyVar
end

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

"""Clone a type context without retaining caller-owned free-variable links."""
function _detached_type_ctx(type_ctx::AbstractDict{Symbol,<:TyLike})
    detached, _ = _detached_type_inputs(type_ctx)
    return detached
end

function _closed_has_mu(ty::TyLike, seen=IdDict{Any,Nothing}())::Bool
    ty isa TyScheme && return _closed_has_mu(ty.ty, seen)
    haskey(seen, ty) && return false
    seen[ty] = nothing
    ty isa TyMu && return true
    if ty isa TyVar
        return ty.instance === nothing ? false : _closed_has_mu(ty.instance, seen)
    end
    for child in type_children(ty)
        _closed_has_mu(child, seen) && return true
    end
    return false
end

function _closed_array_sort_error(
    ty::TyLike,
    side::Symbol,
)
    body = ty isa TyScheme ? ty.ty : ty
    seen = IdDict{Any,Nothing}()
    sorts = Dict{TyVar,Symbol}()

    function visit(t::Ty, sort::Symbol)
        if t isa TyVar
            if t.instance !== nothing
                haskey(seen, t) && return nothing
                seen[t] = nothing
                return visit(t.instance, sort)
            end
            old = get(sorts, t, nothing)
            if old !== nothing && old !== sort
                return ClosedRetrievalArraySortErr(side, t)
            end
            sorts[t] = sort
            return nothing
        end
        haskey(seen, t) && return nothing
        seen[t] = nothing
        if t isa TyArray
            err = visit(t.len, :array_length)
            err === nothing || return err
            return visit(t.elem, :type)
        end
        for child in type_children(t)
            err = visit(child, sort)
            err === nothing || return err
        end
        return nothing
    end

    return visit(body, :type)
end

function _closed_rigid_copy(
    demand::TyLike,
    offer::TyLike,
)
    used_rigid_ids = Set{UInt}()
    function collect_rigid_ids(ty::TyLike, seen=IdDict{Any,Nothing}())
        ty isa TyScheme && return collect_rigid_ids(ty.ty, seen)
        haskey(seen, ty) && return nothing
        seen[ty] = nothing
        if ty isa TyRigid
            push!(used_rigid_ids, ty.id)
        elseif ty isa TyVar
            ty.instance === nothing || collect_rigid_ids(ty.instance, seen)
        else
            for child in type_children(ty)
                collect_rigid_ids(child, seen)
            end
        end
        return nothing
    end
    collect_rigid_ids(demand)
    collect_rigid_ids(offer)

    rigid = Dict{TyVar,TyRigid}()
    next_rigid = Ref(UInt(0))
    function rigid_atom(v::TyVar)
        return get!(rigid, v) do
            while true
                next_rigid[] += UInt(1)
                next_rigid[] in used_rigid_ids || break
            end
            TyRigid(next_rigid[])
        end
    end

    demand_body = demand isa TyScheme ? demand.ty : demand
    demand_copy = _inst(demand_body, rigid_atom)

    offer_factory = make_tyvar_factory()
    offer_quantified = offer isa TyScheme ? Set(offer.vars) : Set{TyVar}()
    offer_flex = Dict{TyVar,TyVar}()
    offer_body = offer isa TyScheme ? offer.ty : offer
    offer_copy = _inst(offer_body, v ->
        v in offer_quantified ? get!(offer_flex, v) do
            offer_factory(0)
        end : rigid_atom(v)
    )
    return demand_copy, offer_copy
end

"""
    closed_directed_match(demand, offer) -> Bool or Err

Return whether `demand ≼ offer`.  This is a directed rank-1 query: offer
quantifiers are the only flexible variables.  `false` denotes an ordinary
structural mismatch; `Err` is reserved for unsupported μ or mixed array-sort
queries.
"""
function closed_directed_match(demand::TyLike, offer::TyLike)
    _closed_has_mu(demand) && return Err(ClosedRetrievalMuErr(:demand))
    _closed_has_mu(offer) && return Err(ClosedRetrievalMuErr(:offer))
    demand_sort_err = _closed_array_sort_error(demand, :demand)
    demand_sort_err === nothing || return Err(demand_sort_err)
    offer_sort_err = _closed_array_sort_error(offer, :offer)
    offer_sort_err === nothing || return Err(offer_sort_err)

    demand_copy, offer_copy = _closed_rigid_copy(demand, offer)
    return isnothing(unify!(demand_copy, offer_copy))
end

@inline _closed_site_parts(site::ClosedDemandSite) = (site.id, site.demand)
@inline _closed_site_parts(site::Pair) = (Int(first(site)), last(site)::TyLike)
@inline _closed_site_parts(site::Tuple{<:Integer,<:TyLike}) = (Int(site[1]), site[2])

"""Return stable ids for every site whose demand is an instance of `offer`."""
function closed_retrieval_scan(sites::AbstractVector, offer::TyLike)
    ids = Int[]
    for site in sites
        id, demand = _closed_site_parts(site)
        match = closed_directed_match(demand, offer)
        match isa Err && return match
        match && push!(ids, id)
    end
    return ids
end

@testitem "closed directed retrieval: rank-1 direction" begin
    using Test
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyScheme, make_tyvar_factory
    using LambdaRegression.Mutations.Populations:
        ClosedDemandSite, closed_directed_match, closed_retrieval_scan

    factory = make_tyvar_factory()
    α = factory(0)
    identity_offer = TyScheme([α], TyFun(α, α))
    demand = TyFun(IntTy, IntTy)

    @test closed_directed_match(demand, identity_offer) === true
    @test closed_directed_match(identity_offer, demand) === false

    sites = [
        ClosedDemandSite(41, demand),
        ClosedDemandSite(99, TyFun(IntTy, IntTy)),
    ]
    @test closed_retrieval_scan(sites, identity_offer) == [41, 99]
end

@testitem "closed directed retrieval: rigid worlds and classified exclusions" begin
    using Test
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyScheme, TyArray, TyNat, TyMu, TyRigid, make_tyvar_factory
    using LambdaRegression.Mutations.Populations:
        closed_directed_match, ClosedRetrievalMuErr, ClosedRetrievalArraySortErr

    factory = make_tyvar_factory()

    # The same ambient variable identity must be preserved across the demand.
    γ = factory(0)
    δ = factory(0)
    demand_shared = TyFun(γ, γ)
    demand_distinct = TyFun(γ, δ)
    α = factory(0)
    identity_offer = TyScheme([α], TyFun(α, α))
    @test closed_directed_match(demand_shared, identity_offer) === true
    @test closed_directed_match(demand_distinct, identity_offer) === false

    # An offer quantifier is local to the offer; caller-owned variables remain
    # unlinked after both successful and failed queries.
    @test γ.instance === nothing
    @test δ.instance === nothing
    @test α.instance === nothing

    μv = factory(0)
    mu_ty = TyMu(μv, TyFun(μv, IntTy))
    mu_result = closed_directed_match(IntTy, mu_ty)
    @test mu_result isa Err
    @test mu_result.value isa ClosedRetrievalMuErr

    β = factory(0)
    mixed_sort_offer = TyScheme([β], TyArray(β, β))
    sort_result = closed_directed_match(TyArray(TyNat(2), IntTy), mixed_sort_offer)
    @test sort_result isa Err
    @test sort_result.value isa ClosedRetrievalArraySortErr

    # Existing rigid atoms reserve their ids; a newly rigidified ambient
    # variable must not accidentally become equal to either input atom.
    ζ = factory(0)
    rigid_collision_demand = TyFun(ζ, ζ)
    rigid_collision_offer_var = factory(0)
    rigid_collision_offer = TyScheme([rigid_collision_offer_var], TyFun(rigid_collision_offer_var, TyRigid(1)))
    @test closed_directed_match(rigid_collision_demand, rigid_collision_offer) === false
end
