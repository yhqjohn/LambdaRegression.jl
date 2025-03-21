# Private adapter from completed lexical typing snapshots to the μ-free open
# retrieval query IR. It intentionally knows only the shared snapshot fields.
using TestItems

using ...TypeSystem:
    Ty, TyLike, TyMu, TyRigid, TyScheme, TyVar, type_children

struct _SnapshotOpenAnswer
    site::Tuple{Vararg{Int}}
    h::Vector{Tuple{Any,Any}}
    residual::_ScopedResidual
end

struct _SnapshotOpenRetrievalBatch
    answers::Vector{_SnapshotOpenAnswer}
    status::Symbol
end

struct _SnapshotMuUnsupportedErr <: Exception end
struct _SnapshotCyclicTypeErr <: Exception end
struct _SnapshotUnclassifiedSiteErr <: Exception end

Base.showerror(io::IO, ::_SnapshotMuUnsupportedErr) =
    print(io, "snapshot open retrieval is strictly mu-free")
Base.showerror(io::IO, ::_SnapshotCyclicTypeErr) =
    print(io, "snapshot open retrieval rejects cyclic inferred types")
Base.showerror(io::IO, ::_SnapshotUnclassifiedSiteErr) =
    print(io, "snapshot open retrieval cannot classify one target site variable")

Base.:(==)(left::_SnapshotOpenAnswer, right::_SnapshotOpenAnswer) =
    left.site == right.site && left.h == right.h && left.residual == right.residual

_snapshot_body(ty::Ty) = ty
_snapshot_body(scheme::TyScheme) = scheme.ty
_snapshot_bound_variables(::Ty) = Set{TyVar}()
_snapshot_bound_variables(scheme::TyScheme) = Set(scheme.vars)

function _snapshot_merge_sort!(
    variables::Dict{TyVar,_QuerySort},
    variable::TyVar,
    sort::_QuerySort,
)
    previous = get(variables, variable, nothing)
    previous === nothing || previous == sort || throw(ArgumentError(
        "one snapshot TyVar appears at both Type and Length sorts",
    ))
    variables[variable] = sort
    return variables
end

function _snapshot_type_variables(ty::TyLike)
    variables = Dict{TyVar,_QuerySort}()
    active = IdDict{Any,Nothing}()
    finished = IdDict{Any,Nothing}()

    function visit(current::Ty, sort::_QuerySort)
        current isa TyMu && throw(_SnapshotMuUnsupportedErr())
        current isa TyRigid && throw(ArgumentError(
            "typing snapshots must expose source TyVar identities, not query rigids",
        ))
        if current isa TyVar
            if current.instance === nothing
                _snapshot_merge_sort!(variables, current, sort)
                return nothing
            end
        end
        haskey(active, current) && throw(_SnapshotCyclicTypeErr())
        haskey(finished, current) && return nothing
        active[current] = nothing
        try
            if current isa TyVar
                visit(current.instance::Ty, sort)
            else
                for (child, child_sort) in zip(
                    type_children(current),
                    _query_child_sorts(current),
                )
                    visit(child, child_sort)
                end
            end
        finally
            delete!(active, current)
        end
        finished[current] = nothing
        return nothing
    end

    visit(_snapshot_body(ty), _TypeSort)
    return variables
end

function _snapshot_merge_variables!(
    destination::Dict{TyVar,_QuerySort},
    source::Dict{TyVar,_QuerySort},
)
    for (variable, sort) in source
        _snapshot_merge_sort!(destination, variable, sort)
    end
    return destination
end

function _snapshot_open_term(
    ty::TyLike,
    variable_key,
)
    active = IdDict{Any,Nothing}()

    function convert(current::Ty, sort::_QuerySort)::_OpenTerm
        current isa TyMu && throw(_SnapshotMuUnsupportedErr())
        current isa TyRigid && throw(ArgumentError(
            "typing snapshots must expose source TyVar identities, not query rigids",
        ))
        if current isa TyVar
            if current.instance === nothing
                key = variable_key(current, sort)
                key.sort == sort || throw(ArgumentError(
                    "snapshot variable role changed its Type/Length sort",
                ))
                return _OpenVariable(key)
            end
        end
        haskey(active, current) && throw(_SnapshotCyclicTypeErr())
        active[current] = nothing
        try
            current isa TyVar && return convert(current.instance::Ty, sort)
            child_terms = _OpenTerm[
                convert(child, child_sort)
                for (child, child_sort) in zip(
                    type_children(current),
                    _query_child_sorts(current),
                )
            ]
            return _OpenConstructor(current, child_terms)
        finally
            delete!(active, current)
        end
    end

    return convert(_snapshot_body(ty), _TypeSort)
end

_snapshot_open_term(variable_key, ty::TyLike) =
    _snapshot_open_term(ty, variable_key)

_snapshot_var_order(variable::TyVar) = variable.uid

function _snapshot_numbered_keys(
    block::Symbol,
    variables::Dict{TyVar,_QuerySort};
    occurrence::Integer=0,
)
    result = Dict{TyVar,_OpenVarKey}()
    ordered = sort!(collect(keys(variables)); by=_snapshot_var_order)
    for (ordinal, variable) in enumerate(ordered)
        result[variable] = _OpenVarKey(
            block,
            Int32(occurrence),
            Int32(ordinal),
            variables[variable],
        )
    end
    return result
end

function _snapshot_binder_key(binder)
    all(name -> hasproperty(binder, name), (:kind, :owner_path, :ordinal)) ||
        throw(ArgumentError("snapshot binder lacks stable lexical identity fields"))
    return (
        String(getproperty(binder, :kind)),
        Tuple(getproperty(binder, :owner_path)),
        Int(getproperty(binder, :ordinal)),
    )
end

_snapshot_occurrence_key(occurrence) = Tuple(getproperty(occurrence, :path))
_snapshot_site_key(site) = Tuple(getproperty(site, :path))

function _snapshot_target_site(site, site_id::Int)
    entries = sort!(collect(pairs(getproperty(site, :binder_schemes)));
        by=entry -> _snapshot_binder_key(first(entry)))

    r_variables = Dict{TyVar,_QuerySort}()
    binder_variables = Dict{Any,Dict{TyVar,_QuerySort}}()
    binder_bound = Dict{Any,Set{TyVar}}()
    for (binder, scheme) in entries
        variables = _snapshot_type_variables(scheme)
        bound = _snapshot_bound_variables(scheme)
        binder_variables[binder] = variables
        binder_bound[binder] = bound
        for (variable, sort) in variables
            variable in bound || _snapshot_merge_sort!(r_variables, variable, sort)
        end
    end

    demand = getproperty(site, :demand)::TyLike
    offer = getproperty(site, :offer)::TyLike
    offer_variables = _snapshot_type_variables(offer)
    offer_bound = _snapshot_bound_variables(offer)
    for (variable, sort) in offer_variables
        variable in offer_bound ||
            _snapshot_merge_sort!(r_variables, variable, sort)
    end
    demand_variables = _snapshot_type_variables(demand)
    r_keys = _snapshot_numbered_keys(:R, r_variables)
    b_variables = Dict(
        variable => sort for (variable, sort) in demand_variables
        if variable in offer_bound
    )
    b_keys = _snapshot_numbered_keys(:B, b_variables)
    demand_term = _snapshot_open_term(demand) do variable, _
        return get(r_keys, variable) do
            get(b_keys, variable) do
                throw(_SnapshotUnclassifiedSiteErr())
            end
        end
    end

    binders = _OpenTargetBinder[]
    trace = Dict{Int,Any}()
    for (binder_id, (binder, scheme)) in enumerate(entries)
        bound = binder_bound[binder]
        bound_variables = Dict(
            variable => sort for (variable, sort) in binder_variables[binder]
            if variable in bound
        )
        t_keys = _snapshot_numbered_keys(
            :T,
            bound_variables;
            occurrence=binder_id,
        )
        body = _snapshot_open_term(scheme) do variable, _
            return get(t_keys, variable) do
                get(r_keys, variable) do
                    throw(_SnapshotUnclassifiedSiteErr())
                end
            end
        end
        quantified = sort!(collect(values(t_keys)); by=_open_key_order)
        push!(binders, _OpenTargetBinder(binder_id, body, quantified))
        trace[binder_id] = binder
    end
    return _JointSite(site_id, demand_term, binders), trace
end

function _snapshot_donor_signature(site)
    group_entries = sort!(collect(pairs(getproperty(site, :external_occurrences)));
        by=entry -> _snapshot_binder_key(first(entry)))
    occurrences = Tuple{Int,Any,Any}[]
    group_trace = Dict{Int,Any}()
    next_occurrence = 0
    for (group_id, (binder, group_occurrences)) in enumerate(group_entries)
        group_trace[group_id] = binder
        for occurrence in sort!(collect(group_occurrences); by=_snapshot_occurrence_key)
            next_occurrence += 1
            push!(occurrences, (group_id, occurrence, next_occurrence))
        end
    end

    output = getproperty(site, :offer)::TyLike
    output_variables = _snapshot_type_variables(output)
    all_variables = copy(output_variables)
    external_free_variables = Dict{TyVar,_QuerySort}()
    binder_schemes = getproperty(site, :binder_schemes)
    for (binder, _) in group_entries
        scheme = binder_schemes[binder]
        variables = _snapshot_type_variables(scheme)
        bound = _snapshot_bound_variables(scheme)
        for (variable, sort) in variables
            variable in bound ||
                _snapshot_merge_sort!(external_free_variables, variable, sort)
        end
    end
    occurrence_variables = Dict{Int,Dict{TyVar,_QuerySort}}()
    occurrence_bound = Dict{Int,Set{TyVar}}()
    presence = Dict{TyVar,Set{Int}}()
    for (_, occurrence, occurrence_id) in occurrences
        use_type = getproperty(occurrence, :use_type)::TyLike
        variables = _snapshot_type_variables(use_type)
        bound = _snapshot_bound_variables(use_type)
        occurrence_variables[occurrence_id] = variables
        occurrence_bound[occurrence_id] = bound
        _snapshot_merge_variables!(all_variables, variables)
        for variable in keys(variables)
            variable in bound && continue
            push!(get!(Set{Int}, presence, variable), occurrence_id)
        end
    end

    a_variables = copy(output_variables)
    _snapshot_merge_variables!(a_variables, external_free_variables)
    for (variable, occurrence_ids) in presence
        length(occurrence_ids) >= 2 &&
            (a_variables[variable] = all_variables[variable])
    end
    a_keys = _snapshot_numbered_keys(:A, a_variables)
    output_term = _snapshot_open_term(output) do variable, _
        return get(a_keys, variable) do
            throw(ArgumentError("unclassified donor output variable"))
        end
    end

    group_requirements = Dict{Int,Vector{_DonorOccurrence}}(
        group_id => _DonorOccurrence[] for group_id in keys(group_trace)
    )
    for (group_id, occurrence, occurrence_id) in occurrences
        variables = occurrence_variables[occurrence_id]
        bound = occurrence_bound[occurrence_id]
        e_variables = Dict(
            variable => sort for (variable, sort) in variables
            if !(variable in bound) && !haskey(a_keys, variable)
        )
        p_variables = Dict(
            variable => sort for (variable, sort) in variables
            if variable in bound
        )
        e_keys = _snapshot_numbered_keys(:E, e_variables; occurrence=occurrence_id)
        p_keys = _snapshot_numbered_keys(:P, p_variables; occurrence=occurrence_id)
        requirement = _snapshot_open_term(getproperty(occurrence, :use_type)) do variable, _
            return get(p_keys, variable) do
                get(a_keys, variable) do
                    get(e_keys, variable) do
                        throw(ArgumentError("unclassified donor occurrence variable"))
                    end
                end
            end
        end
        push!(
            group_requirements[group_id],
            _DonorOccurrence(occurrence_id, requirement),
        )
    end
    groups = _OccurrenceGroup[
        _OccurrenceGroup(group_id, group_requirements[group_id])
        for group_id in sort!(collect(keys(group_requirements)))
    ]
    return _DonorSignature(output_term, groups), group_trace
end

function _snapshot_site_list(input)
    input isa AbstractVector && return collect(input)
    hasproperty(input, :demand) && return [input]
    throw(ArgumentError("expected a lexical snapshot, site, or site vector"))
end

function _snapshot_open_retrieval_bounded(
    donor_site,
    target_input;
    candidate_allowed=(source, target) -> true,
    max_answers::Int=typemax(Int),
)
    max_answers >= 1 || throw(ArgumentError("max_answers must be positive"))
    hasproperty(donor_site, :demand) || throw(ArgumentError(
        "donor retrieval requires one selected lexical site",
    ))
    target_sites = _snapshot_site_list(target_input)
    if length(target_sites) > 1
        sort!(target_sites; by=_snapshot_site_key)
        paths = [_snapshot_site_key(site) for site in target_sites]
        length(paths) == length(unique(paths)) ||
            throw(ArgumentError("duplicate target site path"))
    end

    donor, group_trace = _snapshot_donor_signature(donor_site)
    joint_sites = _JointSite[]
    binder_traces = Dict{Int,Dict{Int,Any}}()
    site_paths = Dict{Int,Tuple{Vararg{Int}}}()
    for (site_id, site) in enumerate(target_sites)
        joint, trace = _snapshot_target_site(site, site_id)
        push!(joint_sites, joint)
        binder_traces[site_id] = trace
        site_paths[site_id] = _snapshot_site_key(site)
    end

    answers = _SnapshotOpenAnswer[]
    status = :exhausted
    for (site_id, joint_site) in enumerate(joint_sites)
        remaining = max_answers - length(answers)
        remaining <= 0 && return _SnapshotOpenRetrievalBatch(
            answers,
            :answer_limit,
        )
        low_candidate_allowed = (group_id, binder_id) -> candidate_allowed(
            group_trace[group_id],
            binder_traces[site_id][binder_id],
        )
        batch = _open_retrieval_bounded(
            donor,
            [joint_site];
            candidate_allowed=low_candidate_allowed,
            max_answers=remaining,
        )
        for answer in batch.answers
            trace = binder_traces[answer.site]
            mapping = Tuple{Any,Any}[
                (group_trace[group_id], trace[binder_id])
                for (group_id, binder_id) in answer.h
            ]
            push!(answers, _SnapshotOpenAnswer(
                site_paths[answer.site],
                mapping,
                answer.residual,
            ))
        end
        if batch.status === :answer_limit
            status = :answer_limit
            break
        end
    end
    return _SnapshotOpenRetrievalBatch(answers, status)
end

function _snapshot_open_retrieval(donor_site, target_input)
    return _snapshot_open_retrieval_bounded(donor_site, target_input).answers
end

@testitem "snapshot open retrieval maps ordinary monomorphic lambda capture" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        _snapshot_open_retrieval, prepare_typed_analysis,
        typed_analysis_site_ids, typed_analysis_site
    using LambdaRegression.TypeSystem: TyLike

    snapshot = prepare_typed_analysis(
        AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(),
    )
    site = typed_analysis_site(
        snapshot, only(typed_analysis_site_ids(snapshot; site_path=[1])),
    )
    answers = _snapshot_open_retrieval(site, [site])

    @test length(answers) == 1
    @test answers[1].site == (1,)
    @test length(answers[1].h) == 1
    @test answers[1].h[1] == (
        only(keys(site.external_occurrences)),
        only(keys(site.binder_schemes)),
    )
end

@testitem "snapshot open retrieval bounds answers and remaps role predicate" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _snapshot_open_retrieval, _snapshot_open_retrieval_bounded
    using LambdaRegression.TypeSystem: IntTy, TyLike, TyScheme

    donor_binder = (kind=:lambda, owner_path=(), ordinal=9)
    donor_occurrence = (path=(1,), use_type=IntTy)
    donor = (;
        demand=IntTy,
        offer=IntTy,
        external_occurrences=Dict(donor_binder => [donor_occurrence]),
        binder_schemes=Dict{Any,TyLike}(
            donor_binder => TyScheme([], IntTy),
        ),
    )
    ordinary = (kind=:lambda, owner_path=(), ordinal=1)
    later = (kind=:lambda, owner_path=(), ordinal=2)
    target = (;
        path=(1,),
        demand=IntTy,
        offer=TyScheme([], IntTy),
        binder_schemes=Dict{Any,TyLike}(
            ordinary => TyScheme([], IntTy),
            later => TyScheme([], IntTy),
        ),
    )

    exact = _snapshot_open_retrieval_bounded(donor, [target]; max_answers=2)
    limited = _snapshot_open_retrieval_bounded(donor, [target]; max_answers=1)
    filtered = _snapshot_open_retrieval_bounded(
        donor,
        [target];
        candidate_allowed=(source, destination) -> destination === ordinary,
    )
    @test length(exact.answers) == 2
    @test exact.status === :exhausted
    @test length(limited.answers) == 1
    @test limited.status === :answer_limit
    @test filtered.status === :exhausted
    @test filtered.answers == exact.answers[1:1]
    @test _snapshot_open_retrieval(donor, [target]) == exact.answers
end

@testitem "snapshot open retrieval preserves ambient variable provenance" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _OpenVariable, _snapshot_donor_signature, _snapshot_target_site
    using LambdaRegression.TypeSystem:
        IntTy, TyLike, TyScheme, make_tyvar_factory

    factory = make_tyvar_factory()
    ambient = factory(0)
    local_var = factory(0)
    binder = (kind=:lambda, owner_path=(), ordinal=1)
    occurrence = (path=(1,), use_type=ambient)

    donor_site = (;
        offer=IntTy,
        external_occurrences=Dict(binder => [occurrence]),
        binder_schemes=Dict{Any,TyLike}(binder => TyScheme([], ambient)),
    )
    signature, _ = _snapshot_donor_signature(donor_site)
    requirement = only(only(signature.groups).occurrences).requirement
    @test requirement isa _OpenVariable
    @test requirement.key.block === :A

    ambient_target = (;
        demand=ambient,
        offer=TyScheme([], ambient),
        binder_schemes=Dict{Any,TyLike}(),
    )
    ambient_joint, _ = _snapshot_target_site(ambient_target, 1)
    @test ambient_joint.demand isa _OpenVariable
    @test ambient_joint.demand.key.block === :R

    local_target = (;
        demand=local_var,
        offer=TyScheme([local_var], local_var),
        binder_schemes=Dict{Any,TyLike}(),
    )
    local_joint, _ = _snapshot_target_site(local_target, 2)
    @test local_joint.demand isa _OpenVariable
    @test local_joint.demand.key.block === :B
end

@testitem "snapshot open retrieval freshens a polymorphic let per donor use" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        _snapshot_open_retrieval, prepare_typed_analysis,
        typed_analysis_site_ids, typed_analysis_site
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike

    identity = AbsNode(:x, IndexNode(0))
    donor_body = AppNode(
        AppNode(
            ConstNode(:PAIR_IB),
            AppNode(IndexNode(0), ConstNode(:ONE)),
        ),
        AppNode(IndexNode(0), ConstNode(:TRUE)),
    )
    ctx = Dict{Symbol,TyLike}(
        :ONE => IntTy,
        :TRUE => BoolTy,
        :PAIR_IB => TyFun(IntTy, TyFun(BoolTy, IntTy)),
        :KEEP_II => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    donor_snapshot = prepare_typed_analysis(
        AstTree(AppNode(AbsNode(:id, donor_body), identity)),
        ctx,
    )
    donor = typed_analysis_site(
        donor_snapshot,
        only(typed_analysis_site_ids(donor_snapshot; site_path=[1, 1])),
    )

    polymorphic_snapshot = prepare_typed_analysis(
        AstTree(AppNode(AbsNode(:id, ConstNode(:ONE)), identity)),
        ctx,
    )
    polymorphic = typed_analysis_site(
        polymorphic_snapshot,
        only(typed_analysis_site_ids(polymorphic_snapshot; site_path=[1, 1])),
    )

    monomorphic_tree = AbsNode(
        :f,
        AppNode(
            AppNode(
                ConstNode(:KEEP_II),
                AppNode(IndexNode(0), ConstNode(:ONE)),
            ),
            ConstNode(:ONE),
        ),
    )
    monomorphic_snapshot = prepare_typed_analysis(AstTree(monomorphic_tree), ctx)
    monomorphic = typed_analysis_site(
        monomorphic_snapshot,
        only(typed_analysis_site_ids(monomorphic_snapshot; site_path=[1, 2])),
    )

    @test length(only(values(donor.external_occurrences))) == 2
    @test length(_snapshot_open_retrieval(donor, [polymorphic])) == 1
    @test isempty(_snapshot_open_retrieval(donor, [monomorphic]))
end

@testitem "snapshot open retrieval preserves a shared output-use variable" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        _snapshot_open_retrieval, prepare_typed_analysis,
        typed_analysis_site_ids, typed_analysis_site
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike

    donor_snapshot = prepare_typed_analysis(
        AstTree(AbsNode(:x, IndexNode(0))),
        Dict{Symbol,TyLike}(),
    )
    donor = typed_analysis_site(
        donor_snapshot, only(typed_analysis_site_ids(donor_snapshot; site_path=[1])),
    )

    ctx = Dict{Symbol,TyLike}(
        :ONE => IntTy,
        :KEEP_BI => TyFun(BoolTy, TyFun(IntTy, IntTy)),
    )
    target_tree = AbsNode(
        :b,
        AppNode(
            AppNode(ConstNode(:KEEP_BI), IndexNode(0)),
            ConstNode(:ONE),
        ),
    )
    target_snapshot = prepare_typed_analysis(AstTree(target_tree), ctx)
    target = typed_analysis_site(
        target_snapshot,
        only(typed_analysis_site_ids(target_snapshot; site_path=[1, 2])),
    )

    # The donor's one TyVar is both its output and its captured-binder demand.
    # Solving those projections independently would accept Int and Bool; the
    # joint A identity correctly makes the mapping inconsistent.
    @test isempty(_snapshot_open_retrieval(donor, [target]))

    donor_sites = [
        typed_analysis_site(donor_snapshot, id)
        for id in typed_analysis_site_ids(donor_snapshot)
    ]
    forward = _snapshot_open_retrieval(donor, donor_sites)
    reverse_order = _snapshot_open_retrieval(donor, reverse(donor_sites))
    @test reverse_order == forward
end

@testitem "snapshot open retrieval accepts a FunctionDef lexical snapshot" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, FunctionDefNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        _snapshot_open_retrieval, prepare_typed_analysis,
        typed_analysis_site_ids, typed_analysis_site
    using LambdaRegression.TypeSystem: TyLike

    recursive_call = FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )
    snapshot = prepare_typed_analysis(
        AstTree(recursive_call),
        Dict{Symbol,TyLike}(),
    )
    site = typed_analysis_site(
        snapshot, only(typed_analysis_site_ids(snapshot; site_path=[1])),
    )
    answers = _snapshot_open_retrieval(site, [site])

    @test length(answers) == 1
    @test answers[1].site == (1,)
    @test length(answers[1].h) == 1
    @test first(answers[1].h[1]).kind === :function_def
    @test last(answers[1].h[1]).kind === :function_def
end

@testitem "unclassifiable target site variables become classified exclusions" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _SnapshotUnclassifiedSiteErr, _engine_witness_mappings,
        _snapshot_target_site
    using LambdaRegression.TypeSystem:
        IntTy, TyLike, TyScheme, make_tyvar_factory
    using LambdaRegression.Utils: Err

    # A demand variable that appears in no binder scheme and no offer cannot
    # be placed in R or B; retrieval must exclude the site, not crash search.
    factory = make_tyvar_factory()
    stray = factory(0)
    binder = (kind=:lambda, owner_path=(), ordinal=1)
    target = (;
        path=(1,),
        demand=stray,
        offer=IntTy,
        binder_schemes=Dict{Any,TyLike}(binder => TyScheme([], IntTy)),
    )
    @test_throws _SnapshotUnclassifiedSiteErr _snapshot_target_site(target, 1)

    donor_binder = (kind=:lambda, owner_path=(), ordinal=9)
    donor = (;
        demand=IntTy,
        offer=IntTy,
        external_occurrences=Dict(
            donor_binder => [(path=(1,), use_type=IntTy)],
        ),
        binder_schemes=Dict{Any,TyLike}(donor_binder => TyScheme([], IntTy)),
    )
    result = _engine_witness_mappings(donor, target)
    @test result.mappings isa Err
    @test result.status === :exhausted
end
