# Pure, private μ-free open-retrieval seam. Query construction is independent
# of AST transport; all mutable solver state is owned by one retrieval call.
using TestItems

using ...TypeSystem: Ty

abstract type _OpenTerm end

struct _OpenVarKey
    block::Symbol
    occurrence::Int32
    ordinal::Int32
    sort::_QuerySort
end

struct _OpenVariable <: _OpenTerm
    key::_OpenVarKey
end

struct _OpenConstructor <: _OpenTerm
    prototype::Ty
    children::Vector{_OpenTerm}
end

struct _OpenTargetBinder
    id::Int
    body::_OpenTerm
    quantified::Vector{_OpenVarKey}
end

struct _JointSite
    id::Int
    demand::_OpenTerm
    binders::Vector{_OpenTargetBinder}
end

struct _DonorOccurrence
    id::Int
    requirement::_OpenTerm
end

struct _OccurrenceGroup
    id::Int
    occurrences::Vector{_DonorOccurrence}
end

struct _DonorSignature
    output::_OpenTerm
    groups::Vector{_OccurrenceGroup}
end

struct _OpenRetrievalAnswer
    site::Int
    h::Vector{Tuple{Int,Int}}
    residual::_ScopedResidual
end

struct _OpenRetrievalBatch
    answers::Vector{_OpenRetrievalAnswer}
    status::Symbol
end

# Internal counters used by retrieval regression tests.  They deliberately
# describe search work, rather than elapsed time, so pruning tests remain
# stable across machines.
mutable struct _OpenRetrievalStats
    group_attempts::Int
    pruned_branches::Int
    leaves::Int
end

_OpenRetrievalStats() = _OpenRetrievalStats(0, 0, 0)

Base.:(==)(left::_OpenRetrievalAnswer, right::_OpenRetrievalAnswer) =
    left.site == right.site && left.h == right.h && left.residual == right.residual

_OpenTargetBinder(id::Integer, body::_OpenTerm) =
    _OpenTargetBinder(Int(id), body, _OpenVarKey[])
_OpenTargetBinder(
    id::Integer,
    body::_OpenTerm,
    quantified::AbstractVector{_OpenVarKey},
) = _OpenTargetBinder(Int(id), body, collect(quantified))
_JointSite(id::Integer, demand::_OpenTerm, binders::AbstractVector{_OpenTargetBinder}) =
    _JointSite(Int(id), demand, collect(binders))
_DonorOccurrence(id::Integer, requirement::_OpenTerm) =
    _DonorOccurrence(Int(id), requirement)
_OccurrenceGroup(id::Integer, occurrences::AbstractVector{_DonorOccurrence}) =
    _OccurrenceGroup(Int(id), collect(occurrences))
_DonorSignature(output::_OpenTerm, groups::AbstractVector{_OccurrenceGroup}) =
    _DonorSignature(output, collect(groups))

function _open_var(
    block::Symbol,
    ordinal::Integer;
    occurrence::Integer=0,
    sort::_QuerySort=_TypeSort,
)
    return _OpenVariable(_OpenVarKey(
        block,
        Int32(occurrence),
        Int32(ordinal),
        sort,
    ))
end

_open_atom(prototype::Ty) = _OpenConstructor(prototype, _OpenTerm[])
_open_constructor(prototype::Ty, children::_OpenTerm...) =
    _OpenConstructor(prototype, _OpenTerm[children...])

mutable struct _OpenWorld
    solver::_ScopedSolver
    r_scope::_ScopeId
    b_scope::_ScopeId
    occurrence_scopes::Dict{Int,Tuple{_ScopeId,_ScopeId}}
    variables::Dict{_OpenVarKey,_QRef}
end

function _open_term_variables!(keys::Set{_OpenVarKey}, term::_OpenVariable)
    push!(keys, term.key)
    return keys
end

function _open_term_variables!(keys::Set{_OpenVarKey}, term::_OpenConstructor)
    for child in term.children
        _open_term_variables!(keys, child)
    end
    return keys
end

_open_term_variables(term::_OpenTerm) =
    _open_term_variables!(Set{_OpenVarKey}(), term)

const _OPEN_BLOCK_ORDER = Dict(
    :R => 1,
    :Q => 2,
    :B => 3,
    :A => 4,
    :E => 5,
    :P => 6,
    :T => 7,
)

_open_key_order(key::_OpenVarKey) = (
    get(_OPEN_BLOCK_ORDER, key.block, typemax(Int)),
    Int(key.occurrence),
    Int(key.ordinal),
    Int(key.sort),
)

function _validate_distinct_ids(values, kind::AbstractString)
    ids = [value.id for value in values]
    length(ids) == length(unique(ids)) ||
        throw(ArgumentError("duplicate $kind id"))
    return nothing
end

function _validate_term_roles(
    term::_OpenTerm,
    allowed::Tuple,
    context::AbstractString,
)
    for key in _open_term_variables(term)
        key.block in allowed || throw(ArgumentError(
            "$context contains the forbidden $(key.block) variable block",
        ))
    end
    return nothing
end

function _validate_variable_sorts(
    terms,
    blocks::Tuple,
    context::AbstractString,
)
    seen = Dict{Tuple{Symbol,Int32,Int32},_QuerySort}()
    for term in terms, key in _open_term_variables(term)
        key.block in blocks || continue
        identity = (key.block, key.occurrence, key.ordinal)
        previous = get(seen, identity, nothing)
        previous === nothing || previous == key.sort || throw(ArgumentError(
            "$context reuses one logical variable at different sorts",
        ))
        seen[identity] = key.sort
    end
    return nothing
end

function _validate_open_inputs(donor::_DonorSignature, site::_JointSite)
    _validate_term_roles(donor.output, (:A,), "donor output")
    _validate_term_roles(site.demand, (:R, :Q, :B), "site demand")
    donor_terms = _OpenTerm[donor.output]
    site_terms = _OpenTerm[site.demand]
    for group in donor.groups
        isempty(group.occurrences) && throw(ArgumentError(
            "donor occurrence groups must be nonempty",
        ))
        for occurrence in group.occurrences
            push!(donor_terms, occurrence.requirement)
            _validate_term_roles(
                occurrence.requirement,
                (:A, :E, :P),
                "donor occurrence",
            )
            for key in _open_term_variables(occurrence.requirement)
                if key.block in (:E, :P)
                    key.occurrence == occurrence.id || throw(ArgumentError(
                        "occurrence-local variable refers to another occurrence",
                    ))
                end
            end
        end
    end
    for binder in site.binders
        push!(site_terms, binder.body)
        _validate_term_roles(
            binder.body,
            (:R, :Q, :B, :T),
            "target binder",
        )
        length(binder.quantified) == length(unique(binder.quantified)) ||
            throw(ArgumentError("duplicate target binder quantifier"))
        all(key -> key.block === :T, binder.quantified) ||
            throw(ArgumentError("target binder quantifiers must use the T block"))
        quantified = Set(binder.quantified)
        all(
            key -> key.block !== :T || key in quantified,
            _open_term_variables(binder.body),
        ) || throw(ArgumentError("unquantified target scheme variable"))
        _validate_variable_sorts(
            _OpenTerm[binder.body],
            (:T,),
            "target binder",
        )
    end
    _validate_variable_sorts(donor_terms, (:A, :E, :P), "donor signature")
    _validate_variable_sorts(site_terms, (:R, :Q, :B), "joint site")
    return nothing
end

function _new_open_world(occurrences::Vector{_DonorOccurrence})
    solver = _ScopedSolver()
    r_scope = _new_scope!(solver)
    b_scope = _new_scope!(solver, r_scope)
    occurrence_scopes = Dict{Int,Tuple{_ScopeId,_ScopeId}}()
    for occurrence in sort(occurrences; by=x -> x.id)
        e_scope = _new_scope!(solver, b_scope)
        p_scope = _new_scope!(solver, e_scope)
        occurrence_scopes[occurrence.id] = (e_scope, p_scope)
    end
    return _OpenWorld(
        solver,
        r_scope,
        b_scope,
        occurrence_scopes,
        Dict{_OpenVarKey,_QRef}(),
    )
end

function _open_variable_ref!(world::_OpenWorld, key::_OpenVarKey)
    key.block === :T && throw(ArgumentError(
        "target scheme variables require a fresh occurrence instantiation",
    ))
    return get!(world.variables, key) do
        if key.block === :R
            return _new_rigid!(
                world.solver,
                key.sort,
                world.r_scope,
                _RigidLabel(:R, 0, key.ordinal),
            )
        elseif key.block === :Q
            return _new_flex!(world.solver, key.sort, world.r_scope)
        elseif key.block === :B
            return _new_rigid!(
                world.solver,
                key.sort,
                world.b_scope,
                _RigidLabel(:B, 0, key.ordinal),
            )
        elseif key.block === :A
            return _new_flex!(world.solver, key.sort, world.b_scope)
        elseif key.block === :E
            scopes = get(world.occurrence_scopes, Int(key.occurrence), nothing)
            scopes === nothing && throw(ArgumentError(
                "E variable refers to an undeclared donor occurrence",
            ))
            return _new_flex!(world.solver, key.sort, first(scopes))
        elseif key.block === :P
            scopes = get(world.occurrence_scopes, Int(key.occurrence), nothing)
            scopes === nothing && throw(ArgumentError(
                "P variable refers to an undeclared donor occurrence",
            ))
            return _new_rigid!(
                world.solver,
                key.sort,
                last(scopes),
                _RigidLabel(:P, key.occurrence, key.ordinal),
            )
        end
        throw(ArgumentError("unknown open-retrieval variable block $(key.block)"))
    end
end

function _lower_open_term!(
    world::_OpenWorld,
    term::_OpenVariable,
    target_instantiation::Dict{_OpenVarKey,_QRef}=Dict{_OpenVarKey,_QRef}(),
)
    term.key.block === :T && return get(target_instantiation, term.key) do
        throw(ArgumentError("unquantified target scheme variable"))
    end
    return _open_variable_ref!(world, term.key)
end

function _lower_open_term!(
    world::_OpenWorld,
    term::_OpenConstructor,
    target_instantiation::Dict{_OpenVarKey,_QRef}=Dict{_OpenVarKey,_QRef}(),
)
    children = _QRef[
        _lower_open_term!(world, child, target_instantiation)
        for child in term.children
    ]
    return _new_constructor!(world.solver, term.prototype, children)
end

function _instantiate_target!(
    world::_OpenWorld,
    binder::_OpenTargetBinder,
    occurrence::_DonorOccurrence,
)
    instantiation = Dict{_OpenVarKey,_QRef}()
    _, p_scope = world.occurrence_scopes[occurrence.id]
    for key in binder.quantified
        key.block === :T || throw(ArgumentError(
            "target binder quantifiers must use the T block",
        ))
        instantiation[key] = _new_flex!(world.solver, key.sort, p_scope)
    end
    return _lower_open_term!(world, binder.body, instantiation)
end

function _rollback_open_world!(world::_OpenWorld, mark::_BranchMark)
    _rollback!(world.solver, mark)
    arena_length = length(world.solver.nodes)
    filter!(pair -> Int(last(pair)) <= arena_length, world.variables)
    return nothing
end

function _observable_keys(donor::_DonorSignature, site::_JointSite)
    keys = Set{_OpenVarKey}()
    _open_term_variables!(keys, donor.output)
    _open_term_variables!(keys, site.demand)
    for group in donor.groups, occurrence in group.occurrences
        _open_term_variables!(keys, occurrence.requirement)
    end
    for binder in site.binders
        _open_term_variables!(keys, binder.body)
    end
    qkeys = sort!([key for key in keys if key.block === :Q]; by=_open_key_order)
    akeys = sort!([key for key in keys if key.block === :A]; by=_open_key_order)
    return qkeys, akeys
end

function _open_retrieval_site_bounded(
    donor::_DonorSignature,
    site::_JointSite,
    ; stats::Union{Nothing,_OpenRetrievalStats}=nothing,
    candidate_allowed=(group_id, binder_id) -> true,
    max_answers::Int=typemax(Int),
)
    max_answers >= 1 || throw(ArgumentError("max_answers must be positive"))
    _validate_open_inputs(donor, site)
    groups = sort(donor.groups; by=x -> x.id)
    binders = sort(site.binders; by=x -> x.id)
    _validate_distinct_ids(groups, "donor group")
    _validate_distinct_ids(binders, "target binder")
    occurrences = reduce(
        vcat,
        (sort(group.occurrences; by=x -> x.id) for group in groups);
        init=_DonorOccurrence[],
    )
    _validate_distinct_ids(occurrences, "donor occurrence")

    world = _new_open_world(occurrences)
    output = _lower_open_term!(world, donor.output)
    demand = _lower_open_term!(world, site.demand)
    requirements = Dict(
        occurrence.id => _lower_open_term!(world, occurrence.requirement)
        for occurrence in occurrences
    )
    qkeys, akeys = _observable_keys(donor, site)
    qvars = _QRef[_open_variable_ref!(world, key) for key in qkeys]
    avars = _QRef[_open_variable_ref!(world, key) for key in akeys]
    # Keep occurrence order deterministic without repeating a sort for every
    # binder branch in the recursive join.
    sorted_occurrences = [sort(group.occurrences; by=x -> x.id) for group in groups]

    answers = _OpenRetrievalAnswer[]

    # Solve the shared output-demand equation once.  Every recursive level
    # then contributes only its own group's occurrence equations on the same
    # solver trail.  A failed group rolls back immediately, so later groups
    # are never explored under an impossible prefix.
    base_solved = _unify_scoped!(
        world.solver,
        _QueryEquation[_QueryEquation(output, demand)],
    )
    base_solved isa _ScopedSolveFailure &&
        return _OpenRetrievalBatch(answers, :exhausted)

    mapping = Tuple{Int,Int}[]
    function visit_group(group_index::Int)
        if group_index > length(groups)
            stats === nothing || (stats.leaves += 1)
            residual = _project_scoped_residual(world.solver, qvars, avars)
            residual isa _ScopedResidual || return nothing
            push!(answers, _OpenRetrievalAnswer(site.id, copy(mapping), residual))
            return false
        end

        group = groups[group_index]
        group_occurrences = sorted_occurrences[group_index]
        allowed_binders = [
            binder for binder in binders
            if candidate_allowed(group.id, binder.id)
        ]
        for (binder_index, binder) in enumerate(allowed_binders)
            stats === nothing || (stats.group_attempts += 1)
            mark = _mark(world.solver)
            try
                equations = _QueryEquation[]
                for occurrence in group_occurrences
                    target = _instantiate_target!(world, binder, occurrence)
                    push!(equations, _QueryEquation(
                        requirements[occurrence.id],
                        target,
                    ))
                end
                solved = _unify_scoped!(world.solver, equations)
                if solved isa _ScopedSolveFailure
                    stats === nothing || (stats.pruned_branches += 1)
                else
                    push!(mapping, (group.id, binder.id))
                    limited_below = visit_group(group_index + 1)
                    pop!(mapping)
                    limited_below && return true
                end
            finally
                _rollback_open_world!(world, mark)
            end
            if length(answers) >= max_answers
                # The cap is only a truncation if this group (or an ancestor)
                # still has a candidate that has not been visited.  Thus a
                # cap exactly equal to the exhaustive answer count remains
                # observably `:exhausted`.
                binder_index < length(allowed_binders) && return true
                return false
            end
        end
        return false
    end
    limited = visit_group(1)
    status = limited ? :answer_limit : :exhausted
    return _OpenRetrievalBatch(answers, status)
end

function _open_retrieval_site(
    donor::_DonorSignature,
    site::_JointSite,
    ; stats::Union{Nothing,_OpenRetrievalStats}=nothing,
    candidate_allowed=(group_id, binder_id) -> true,
    max_answers::Int=typemax(Int),
)
    return _open_retrieval_site_bounded(
        donor,
        site;
        stats,
        candidate_allowed,
        max_answers,
    ).answers
end


@testitem "open typed retrieval purges branch-local cached variables" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_retrieval, _open_var
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    Q = _open_var(:Q, 1)
    donor = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [_DonorOccurrence(1, _open_atom(IntTy))]),
        _OccurrenceGroup(2, [_DonorOccurrence(2, _open_atom(IntTy))]),
    ])
    site = _JointSite(1, _open_atom(IntTy), [
        _OpenTargetBinder(1, Q),
        _OpenTargetBinder(2, _open_atom(IntTy)),
    ])

    answers = _open_retrieval(donor, [site])
    @test length(answers) == 4
end

@testitem "open typed retrieval prunes multi-merge branches incrementally" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenRetrievalStats, _OpenTargetBinder, _open_atom,
        _open_retrieval_site
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    # Each group has one feasible binder among sixteen.  A product-first
    # implementation would materialize 16⁴ mappings; the incremental join
    # should only try 4 × 16 group/binder edges and reach one leaf.
    binders = [
        _OpenTargetBinder(
            id,
            _open_atom(id == 16 ? IntTy : BoolTy),
        ) for id in 1:16
    ]
    groups = [
        _OccurrenceGroup(
            id,
            [_DonorOccurrence(id, _open_atom(IntTy))],
        ) for id in 1:4
    ]
    donor = _DonorSignature(_open_atom(IntTy), groups)
    site = _JointSite(1, _open_atom(IntTy), binders)
    stats = _OpenRetrievalStats()
    answers = _open_retrieval_site(donor, site; stats=stats)

    @test length(answers) == 1
    @test answers[1].h == [(1, 16), (2, 16), (3, 16), (4, 16)]
    @test stats.group_attempts == 64
    @test stats.pruned_branches == 60
    @test stats.leaves == 1
end

@testitem "open typed retrieval supports bounded answer batches" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_retrieval_bounded
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    donor = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [_DonorOccurrence(1, _open_atom(IntTy))]),
    ])
    site = _JointSite(1, _open_atom(IntTy), [
        _OpenTargetBinder(10, _open_atom(IntTy)),
        _OpenTargetBinder(20, _open_atom(IntTy)),
    ])

    exact = _open_retrieval_bounded(donor, [site]; max_answers=2)
    limited = _open_retrieval_bounded(donor, [site]; max_answers=1)
    @test length(exact.answers) == 2
    @test exact.status === :exhausted
    @test length(limited.answers) == 1
    @test limited.status === :answer_limit
    mismatch = _open_retrieval_bounded(
        donor,
        [_JointSite(2, _open_atom(BoolTy), site.binders)];
        max_answers=1,
    )
    @test isempty(mismatch.answers)
    @test mismatch.status === :exhausted
    @test_throws ArgumentError _open_retrieval_bounded(
        donor,
        [site];
        max_answers=0,
    )
end

@testitem "open typed retrieval candidate predicate is pre-unification" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_retrieval,
        _open_retrieval_bounded
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    donor = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [_DonorOccurrence(1, _open_atom(IntTy))]),
    ])
    site = _JointSite(1, _open_atom(IntTy), [
        _OpenTargetBinder(10, _open_atom(IntTy)),
        _OpenTargetBinder(20, _open_atom(BoolTy)),
    ])
    allowed = (group_id, binder_id) -> binder_id == 10
    filtered = _open_retrieval_bounded(
        donor,
        [site];
        candidate_allowed=allowed,
    )
    exact = _open_retrieval(donor, [site])
    @test filtered.status === :exhausted
    @test filtered.answers == exact[1:1]
end

@testitem "open typed retrieval zero-group batch skips candidate predicate" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorSignature, _JointSite, _OccurrenceGroup, _OpenTargetBinder,
        _open_atom,
        _open_retrieval_bounded
    using LambdaRegression.TypeSystem: IntTy

    donor = _DonorSignature(_open_atom(IntTy), _OccurrenceGroup[])
    site = _JointSite(1, _open_atom(IntTy), _OpenTargetBinder[])
    calls = Ref(0)
    predicate = (group_id, binder_id) -> (calls[] += 1; true)
    batch = _open_retrieval_bounded(
        donor,
        [site];
        candidate_allowed=predicate,
        max_answers=1,
    )
    @test length(batch.answers) == 1
    @test batch.status === :exhausted
    @test calls[] == 0
end

function _open_retrieval_bounded(
    donor::_DonorSignature,
    sites::AbstractVector{_JointSite},
    ; candidate_allowed=(group_id, binder_id) -> true,
    max_answers::Int=typemax(Int),
)
    max_answers >= 1 || throw(ArgumentError("max_answers must be positive"))
    _validate_distinct_ids(sites, "joint site")
    answers = _OpenRetrievalAnswer[]
    status = :exhausted
    for site in sort(collect(sites); by=x -> x.id)
        remaining = max_answers - length(answers)
        remaining <= 0 && return _OpenRetrievalBatch(answers, :answer_limit)
        batch = _open_retrieval_site_bounded(
            donor,
            site;
            candidate_allowed,
            max_answers=remaining,
        )
        append!(answers, batch.answers)
        if batch.status === :answer_limit
            status = :answer_limit
            break
        end
    end
    return _OpenRetrievalBatch(answers, status)
end

function _open_retrieval(
    donor::_DonorSignature,
    sites::AbstractVector{_JointSite},
)
    return _open_retrieval_bounded(donor, sites).answers
end

@testitem "open typed retrieval joins donor globals across output and occurrences" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_retrieval, _open_var
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    A = _open_var(:A, 1)
    donor = _DonorSignature(A, [
        _OccurrenceGroup(10, [_DonorOccurrence(1, A)]),
    ])
    int_site = _JointSite(7, _open_atom(IntTy), [
        _OpenTargetBinder(20, _open_atom(IntTy)),
        _OpenTargetBinder(30, _open_atom(BoolTy)),
    ])

    answers = _open_retrieval(donor, [int_site])
    @test length(answers) == 1
    @test answers[1].site == 7
    @test answers[1].h == [(10, 20)]

    conflict_site = _JointSite(8, _open_atom(IntTy), [
        _OpenTargetBinder(30, _open_atom(BoolTy)),
    ])
    @test isempty(_open_retrieval(donor, [conflict_site]))
end

@testitem "open typed retrieval freshens polymorphic target uses" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_retrieval, _open_var
    using LambdaRegression.TypeSystem: BoolTy, IntTy

    donor = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [
            _DonorOccurrence(1, _open_atom(IntTy)),
            _DonorOccurrence(2, _open_atom(BoolTy)),
        ]),
    ])
    target_variable = _open_var(:T, 1)
    polymorphic = _JointSite(1, _open_atom(IntTy), [
        _OpenTargetBinder(9, target_variable, [target_variable.key]),
    ])
    monomorphic_variable = _open_var(:Q, 1)
    monomorphic = _JointSite(2, _open_atom(IntTy), [
        _OpenTargetBinder(9, monomorphic_variable),
    ])

    polymorphic_answers = _open_retrieval(donor, [polymorphic])
    @test length(polymorphic_answers) == 1
    @test polymorphic_answers[1].h == [(1, 9)]
    @test isempty(_open_retrieval(donor, [monomorphic]))
end

@testitem "open typed retrieval enforces quantifier scopes" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_constructor, _open_retrieval,
        _open_var
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyList, TyMu, make_tyvar_factory

    A = _open_var(:A, 1)
    B = _open_var(:B, 1)
    Q = _open_var(:Q, 1)
    q_capture_b = _DonorSignature(A, [
        _OccurrenceGroup(1, [_DonorOccurrence(1, A)]),
    ])
    capture_site = _JointSite(1, B, [_OpenTargetBinder(1, Q)])
    @test isempty(_open_retrieval(q_capture_b, [capture_site]))

    P = _open_var(:P, 1; occurrence=1)
    direct_p = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [_DonorOccurrence(1, P)]),
    ])
    direct_site = _JointSite(2, _open_atom(IntTy), [_OpenTargetBinder(1, Q)])
    @test isempty(_open_retrieval(direct_p, [direct_site]))

    T = _open_var(:T, 1)
    linked_requirement = _open_constructor(
        TyFun(IntTy, IntTy),
        A,
        P,
    )
    linked_target = _open_constructor(
        TyFun(IntTy, IntTy),
        T,
        T,
    )
    indirect_p = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(1, [_DonorOccurrence(1, linked_requirement)]),
    ])
    indirect_site = _JointSite(3, _open_atom(IntTy), [
        _OpenTargetBinder(1, linked_target, [T.key]),
    ])
    @test isempty(_open_retrieval(indirect_p, [indirect_site]))

    factory = make_tyvar_factory()
    μvar = factory(0)
    recursive = TyMu(μvar, TyList(μvar))
    mu_donor = _DonorSignature(_open_atom(recursive), _OccurrenceGroup[])
    mu_site = _JointSite(4, _open_atom(IntTy), _OpenTargetBinder[])
    @test_throws ArgumentError _open_retrieval(mu_donor, [mu_site])
end

@testitem "open typed retrieval normalizes mapping and constraint order" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _OccurrenceGroup,
        _OpenTargetBinder, _open_atom, _open_constructor, _open_retrieval,
        _open_var
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun

    A1 = _open_var(:A, 1)
    A2 = _open_var(:A, 2)
    output = _open_constructor(TyFun(IntTy, BoolTy), A1, A2)
    demand = _open_constructor(
        TyFun(IntTy, BoolTy),
        _open_atom(IntTy),
        _open_atom(BoolTy),
    )
    group1_forward = _OccurrenceGroup(10, [
        _DonorOccurrence(3, A1),
        _DonorOccurrence(1, A1),
    ])
    group1_reverse = _OccurrenceGroup(10, reverse(group1_forward.occurrences))
    group2 = _OccurrenceGroup(20, [_DonorOccurrence(2, A2)])
    int_binder = _OpenTargetBinder(100, _open_atom(IntTy))
    bool_binder = _OpenTargetBinder(200, _open_atom(BoolTy))

    forward = _open_retrieval(
        _DonorSignature(output, [group1_forward, group2]),
        [_JointSite(9, demand, [bool_binder, int_binder])],
    )
    reverse_order = _open_retrieval(
        _DonorSignature(output, [group2, group1_reverse]),
        [_JointSite(9, demand, [int_binder, bool_binder])],
    )

    @test length(forward) == 1
    @test forward[1].h == [(10, 100), (20, 200)]
    @test reverse_order == forward
end

@testitem "open typed retrieval rejects malformed signature domains" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _DonorOccurrence, _DonorSignature, _JointSite, _LengthSort,
        _OccurrenceGroup, _OpenTargetBinder, _open_atom, _open_constructor,
        _open_retrieval, _open_var
    using LambdaRegression.TypeSystem: IntTy, TyArray, TyNat

    A_type = _open_var(:A, 1)
    A_length = _open_var(:A, 1; sort=_LengthSort)
    malformed_requirement = _open_constructor(
        TyArray(TyNat(1), IntTy),
        A_length,
        _open_atom(IntTy),
    )
    malformed = _DonorSignature(A_type, [
        _OccurrenceGroup(1, [_DonorOccurrence(1, malformed_requirement)]),
    ])
    concrete_array = _open_constructor(
        TyArray(TyNat(1), IntTy),
        _open_atom(TyNat(1)),
        _open_atom(IntTy),
    )
    site = _JointSite(1, _open_atom(IntTy), [
        _OpenTargetBinder(1, concrete_array),
    ])
    @test_throws ArgumentError _open_retrieval(malformed, [site])

    empty_group = _DonorSignature(_open_atom(IntTy), [
        _OccurrenceGroup(2, _DonorOccurrence[]),
    ])
    @test_throws ArgumentError _open_retrieval(empty_group, [site])
end
