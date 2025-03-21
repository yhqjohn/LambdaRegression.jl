"""
    Common typed-crossover projections

`ProgramAnalysis.TypedAnalysis` is the only persistent analysis used by
population crossover.  This file contains the short-lived site and occurrence
views needed by retrieval and transport.  A view is materialized only after a
caller has selected an integer occurrence ID; it is never cached on the
analysis object.
"""

using TestItems

using ...ASTs.NodeASTs:
    AstTree, AstNode, AppNode, AbsNode, ConstNode, FunctionDefNode, IndexNode,
    VarNode
using ...ASTs.LexicalContexts: BinderId
using ...Programs: Program
using ...ProgramAnalysis:
    TypedAnalysis, analyze_program, path, visible_binder_ids,
    captured_binder_ids
using ...TypeSystem: Ty, TyLike, TyScheme, prune, collect_free_vars,
    make_tyvar_factory
using ...Utils: Err

export TypedAnalysis, TypedAnalysisSite, TypedAnalysisOccurrence
export prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site
export typed_analysis_path

"""One selected De Bruijn use in a transient typed site contract."""
struct TypedAnalysisOccurrence
    path::Tuple{Vararg{Int}}
    node::IndexNode
    index::Int
    binder::BinderId
    binder_scheme::TyLike
    use_type::TyLike
end

"""Attempt-local contract for one selected typed subtree."""
struct TypedAnalysisSite
    path::Tuple{Vararg{Int}}
    node::AstNode
    demand::TyLike
    offer::TyLike
    boundary_visible::Tuple
    binder_schemes::Dict{BinderId,TyLike}
    external_occurrences::Dict{BinderId,Vector{TypedAnalysisOccurrence}}
    monomorphic_external::Bool
end

typed_analysis_path(analysis::TypedAnalysis, id::Int) = Tuple(path(analysis, id))

function _typed_binder_id(analysis::TypedAnalysis, binder_id::Int)
    info = analysis.index.bindings[binder_id]
    return BinderId(info.kind, Tuple(path(analysis, info.owner)), 1)
end

function _typed_is_direct_let_abs(analysis::TypedAnalysis, id::Int)
    parent_id = analysis.index.parent[id]
    parent_id == 0 && return false
    parent = analysis.index.nodes[parent_id]
    return parent isa AppNode && analysis.index.parent_edge[id] == 1 &&
           parent.func === analysis.index.nodes[id] &&
           analysis.index.nodes[id] isa AbsNode
end

function _typed_reject_named!(analysis::TypedAnalysis)
    any(node -> node isa VarNode, analysis.index.nodes) && throw(ArgumentError(
        "typed crossover requires De Bruijn IndexNode binders; NamedVar is not transportable",
    ))
    return nothing
end

"""Return metadata-bearing IDs, excluding the structural let wrapper."""
function typed_analysis_site_ids(
    analysis::TypedAnalysis;
    include_root::Bool=true,
    site_path::Union{Nothing,AbstractVector{<:Integer},Tuple{Vararg{Int}}}=nothing,
)
    requested = site_path === nothing ? nothing : Tuple(Int.(collect(site_path)))
    if requested !== nothing
        id = 1
        for edge in requested
            1 <= edge <= length(analysis.index.children[id]) || return Int[]
            id = analysis.index.children[id][edge]
        end
        !include_root && isempty(requested) && return Int[]
        haskey(analysis.metadata, id) || return Int[]
        _typed_is_direct_let_abs(analysis, id) && return Int[]
        return [id]
    end
    ids = Int[]
    for id in sort!(collect(keys(analysis.metadata)))
        include_root || id != 1 || continue
        _typed_is_direct_let_abs(analysis, id) && continue
        requested === nothing || typed_analysis_path(analysis, id) == requested || continue
        push!(ids, id)
    end
    return ids
end

function _typed_site_binders(analysis::TypedAnalysis, ids::Vector{Int})
    binders = Dict{BinderId,TyLike}()
    for binder_id in ids
        scheme = get(analysis.binder_schemes, binder_id) do
            throw(ArgumentError("missing recorded lexical scheme for binder $binder_id"))
        end
        binders[_typed_binder_id(analysis, binder_id)] = scheme
    end
    return binders
end

"""Materialize one explicit site contract from the common analysis."""
function typed_analysis_site(analysis::TypedAnalysis, id::Int)
    haskey(analysis.metadata, id) || throw(KeyError(id))
    _typed_is_direct_let_abs(analysis, id) && throw(ArgumentError(
        "the direct let-like abstraction is a structural wrapper, not a site",
    ))

    index = analysis.index
    node = index.nodes[id]
    bounds = analysis.metadata[id]
    boundary_ids = visible_binder_ids(analysis, id)
    boundary = Tuple(_typed_binder_id(analysis, binder) for binder in boundary_ids)
    binders = _typed_site_binders(analysis, boundary_ids)
    external_ids = Set(captured_binder_ids(analysis, id))
    external = Dict{BinderId,Vector{TypedAnalysisOccurrence}}()
    stop = index.subtree_end[id]
    for occurrence_id in id:(stop - 1)
        binder_id = index.reference_binder[occurrence_id]
        binder_id in external_ids || continue
        occurrence_node = index.nodes[occurrence_id]
        occurrence_node isa IndexNode || continue
        haskey(analysis.metadata, occurrence_id) || throw(ArgumentError(
            "missing metadata for captured IndexNode occurrence $occurrence_id",
        ))
        binder_scheme = get(analysis.binder_schemes, binder_id) do
            throw(ArgumentError("missing recorded lexical scheme for binder $binder_id"))
        end
        binder = _typed_binder_id(analysis, binder_id)
        occurrence = TypedAnalysisOccurrence(
            typed_analysis_path(analysis, occurrence_id),
            occurrence_node,
            occurrence_node.index,
            binder,
            binder_scheme,
            analysis.metadata[occurrence_id][1],
        )
        push!(get!(Vector{TypedAnalysisOccurrence}, external, binder), occurrence)
    end
    mono = all(
        scheme -> !(scheme isa TyScheme) || isempty(scheme.vars),
        values(binders),
    )
    return TypedAnalysisSite(
        typed_analysis_path(analysis, id),
        node,
        bounds[1],
        bounds[2],
        boundary,
        binders,
        external,
        mono,
    )
end

"""Analyze ordinary and FunctionDef syntax through the one common W path."""
function prepare_typed_analysis(
    term::Union{AstTree,Program},
    ctx::AbstractDict{Symbol,<:TyLike};
    factory=make_tyvar_factory(),
    target_ty::Union{Nothing,Ty}=nothing,
)
    analysis = analyze_program(term, ctx; factory, target_ty)
    analysis isa Err && return analysis
    _typed_reject_named!(analysis)
    return analysis
end

@testitem "common crossover analysis projects ordinary and FunctionDef sites" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        typed_analysis_path
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    ordinary = prepare_typed_analysis(
        AstTree(AbsNode(:x, IndexNode(0))), Dict{Symbol,TyLike}();
        target_ty=TyFun(IntTy, IntTy),
    )
    recursive = prepare_typed_analysis(
        AstTree(FunctionDefNode(AbsNode(:x,
            AppNode(IndexNode(1), IndexNode(0))))),
        Dict{Symbol,TyLike}(); target_ty=TyFun(IntTy, IntTy),
    )
    @test length(typed_analysis_site_ids(ordinary)) == 2
    @test length(typed_analysis_site_ids(recursive)) == 5
    @test length(typed_analysis_site_ids(ordinary; site_path=Int[])) == 1
    @test isempty(typed_analysis_site_ids(ordinary; include_root=false, site_path=Int[]))
    @test length(typed_analysis_site_ids(ordinary; site_path=[1])) == 1
    @test length(typed_analysis_site_ids(
        ordinary; include_root=false, site_path=[1])) == 1
    root = typed_analysis_site(recursive, only(filter(
        id -> isempty(typed_analysis_path(recursive, id)),
        typed_analysis_site_ids(recursive),
    )))
    body = typed_analysis_site(recursive, only(filter(
        id -> typed_analysis_path(recursive, id) == (1,),
        typed_analysis_site_ids(recursive),
    )))
    @test isempty(root.external_occurrences)
    @test length(body.external_occurrences) == 1
end

@testitem "common crossover projection preserves polymorphic use types" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        typed_analysis_path
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, TyProd, TyScheme, typekey

    value = AbsNode(:x, IndexNode(0))
    body = AppNode(
        AppNode(ConstNode(:PAIR), AppNode(IndexNode(0), ConstNode(:one))),
        AppNode(IndexNode(0), ConstNode(:yes)),
    )
    analysis = prepare_typed_analysis(
        AstTree(AppNode(AbsNode(:id, body), value)),
        Dict{Symbol,TyLike}(
            :PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
            :one => IntTy,
            :yes => BoolTy,
        ),
    )
    ids = typed_analysis_site_ids(analysis; site_path=[1, 1])
    @test length(ids) == 1
    site = typed_analysis_site(analysis, only(ids))
    @test length(site.external_occurrences) == 1
    occurrences = only(values(site.external_occurrences))
    @test length(occurrences) == 2
    @test Set(typekey(occ.use_type) for occ in occurrences) == Set([
        typekey(TyFun(IntTy, IntTy)), typekey(TyFun(BoolTy, BoolTy)),
    ])
    scheme = only(values(site.binder_schemes))
    @test scheme isa TyScheme && length(scheme.vars) == 1
end

@testitem "common crossover analysis rejects named transport and wrappers" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_path
    using LambdaRegression.TypeSystem: IntTy, TyLike

    named = () -> prepare_typed_analysis(
        AstTree(AbsNode(:x, VarNode(:x))), Dict{Symbol,TyLike}();
    )
    @test_throws ArgumentError named()
    wrapped = prepare_typed_analysis(
        AstTree(AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one))),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    @test all(
        typed_analysis_path(wrapped, id) != (1,)
        for id in typed_analysis_site_ids(wrapped)
    )
end
