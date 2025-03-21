using ..ASTs.NodeASTs:
    AstNode, AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode,
    VarNode, children
using ..ProgramAnalysis: TypedAnalysis, analyze_program, visible_binder_ids
import ..ProgramAnalysis
import ..TypeSystem

export TypedAnalysis, SourceHole, SourceBoundaryError

"""Expected refusal at the restricted search boundary, not a source typing error."""
struct SourceBoundaryError
    reason::Symbol
    occurrence::Int
end

"""One fixed-ancestor interface. External binders are ordered outermost first."""
struct SourceHole
    source::TypedAnalysis
    occurrence::Int
    path::Vector{Int}
    external_types::Vector{TypeSystem.TyLike}
    anchor::Base.RefValue{Nothing}
end

@inline function _source_scheme_type(scheme::TypeSystem.TyLike)
    scheme isa TypeSystem.TyScheme && isempty(scheme.vars) ? scheme.ty : scheme
end

"""Return the binders visible at an occurrence in outer-to-inner order."""
function _source_visible_types(source::TypedAnalysis, occurrence::Int)
    binder_ids = visible_binder_ids(source, occurrence)
    result = TypeSystem.TyLike[]
    sizehint!(result, length(binder_ids))
    for binder_id in reverse(binder_ids)
        push!(result, _source_scheme_type(source.binder_schemes[binder_id]))
    end
    return result
end

"""
    source_hole(source, path=Int[])

Select an ordinary occurrence in an already inferred source. The path uses
one-based source child numbers, including FunctionDef's self-binding edge.
External bindings remain fixed, with their original schemes; only their use
instances become concrete leaves. The common `TypedAnalysis` keeps integer
occurrence identity and exact binder schemes; this boundary only projects the
requested visible chain.

The finite search language refuses generalising RHS regions whose result still
has free type variables and the retired Scope/Let syntax. These are projection
restrictions, not restrictions on the complete AST layer.
"""
function source_hole(source::TypedAnalysis, path::AbstractVector{<:Integer}=Int[])
    index = source.index
    occurrence = 1
    outer = TypeSystem.TyLike[]
    for edge_raw in path
        edge = Int(edge_raw)
        child_ids = index.children[occurrence]
        1 <= edge <= length(child_ids) ||
            return Err(SourceBoundaryError(:invalid_path, occurrence))
        node = index.nodes[occurrence]
        child = child_ids[edge]
        # Let-like application has a generalising RHS.  Its binder is visible
        # only in the function body, never in this argument subtree; the
        # final occurrence environment below is therefore the sole source of
        # external bindings.
        if node isa AppNode && edge == 2 &&
                index.nodes[child_ids[1]] isa AbsNode
            isempty(TypeSystem.collect_free_vars(source.metadata[child][1])) ||
                return Err(SourceBoundaryError(:generalising_rhs, child))
        end
        occurrence = child
    end
    haskey(source.metadata, occurrence) ||
        return Err(SourceBoundaryError(:binding_artifact, occurrence))
    append!(outer, _source_visible_types(source, occurrence))
    return SourceHole(source, occurrence, Int[path...], outer, Ref{Nothing}(nothing))
end

"""
    prepare_source(tree, type_context, target; max_nodes=100_000, max_depth=512)

Infer a complete source tree once through the common `ProgramAnalysis` index,
then constrain its root to the requested closed type. The source boundary owns
an external tree copy; W and its binder-scheme records are shared with the
repository's common analysis API. Bounds remain a finite TypedSearch policy.
"""
function prepare_source(tree::AstTree, type_context, target::TypeSystem.Ty;
                        max_nodes::Int=100_000, max_depth::Int=512)
    max_nodes > 0 && max_depth >= 0 || throw(ArgumentError("invalid source bounds"))
    isempty(TypeSystem.collect_free_vars(target)) ||
        return Err(SourceBoundaryError(:open_target, 0))
    ctx = Dict{Symbol,TypeSystem.TyLike}(type_context)
    TypeSystem.validate_ctx(ctx)
    active = IdDict{AstNode,Nothing}()
    count = Ref(0)
    function preflight(n, depth)
        count[] += 1
        count[] <= max_nodes && depth <= max_depth || return false
        haskey(active, n) && return false
        active[n] = nothing
        ok = all(c -> preflight(c, depth + 1), children(n))
        delete!(active, n)
        return ok
    end
    preflight(tree.root, 0) || return Err(SourceBoundaryError(:source_bounds, 0))
    try
        analysis = analyze_program(tree, ctx; target_ty=target)
        analysis isa Err && return analysis
        return analysis
    catch err
        err isa ProgramAnalysis.UnsupportedSyntaxError &&
            return Err(SourceBoundaryError(:unsupported_region, err.occurrence))
        rethrow()
    end
end

@testitem "TypedSearch source evidence distinguishes polymorphic occurrences" begin
    using LambdaRegression.TypedSearch: prepare_source
    using LambdaRegression.ProgramAnalysis: TypedAnalysis
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy
    a = newtyvar(0)
    ctx = Dict{Symbol,TyLike}(:ID => TyScheme([a], TyFun(a, a)),
        :PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy, :yes => BoolTy)
    tree = AstTree(AppNode(AppNode(ConstNode(:PAIR),
        AppNode(ConstNode(:ID), ConstNode(:one))),
        AppNode(ConstNode(:ID), ConstNode(:yes))))
    analysis = prepare_source(tree, ctx, TyProd(IntTy, BoolTy))
    @test analysis isa TypedAnalysis
    ids = filter(i -> analysis.index.nodes[i] isa ConstNode &&
        analysis.index.nodes[i].name == :ID, eachindex(analysis.index.nodes))
    @test length(ids) == 2
    @test analysis.metadata[ids[1]][1] == TyFun(IntTy, IntTy)
    @test analysis.metadata[ids[2]][1] == TyFun(BoolTy, BoolTy)
    @test ids[1] != ids[2]
end

@testitem "TypedSearch source root constraints and recursive boundaries" begin
    using LambdaRegression.TypedSearch: prepare_source
    using LambdaRegression.ProgramAnalysis: TypedAnalysis
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.Utils: Err
    ctx = Dict{Symbol,TyLike}()
    tree = AstTree(AbsNode(:x, IndexNode(0)))
    a = prepare_source(tree, ctx, TyFun(FloatTy, FloatTy))
    @test a isa TypedAnalysis
    @test a.metadata[2][1] == FloatTy
    recursive = prepare_source(AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        ctx, TyFun(FloatTy, FloatTy))
    @test recursive isa TypedAnalysis
    @test recursive.root_type == TyFun(FloatTy, FloatTy)
    @test prepare_source(tree, ctx, FloatTy) isa Err
    @test prepare_source(tree, ctx, TyFun(FloatTy, FloatTy); max_nodes=1) isa Err
end

@testitem "TypedSearch local holes retain schemes and recursive self" begin
    using LambdaRegression.TypedSearch: SourceHole, prepare_source, source_hole
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy, FloatTy
    using LambdaRegression.Utils: Err
    ctx = Dict{Symbol,TyLike}(:PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy, :yes => BoolTy)
    body = AppNode(AppNode(ConstNode(:PAIR), AppNode(IndexNode(0), ConstNode(:one))),
        AppNode(IndexNode(0), ConstNode(:yes)))
    tree = AstTree(AppNode(AbsNode(:id, body), AbsNode(:x, IndexNode(0))))
    source = prepare_source(tree, ctx, TyProd(IntTy, BoolTy))
    @test source_hole(source, [1]) isa Err # no metadata for the let binder wrapper
    @test source_hole(source, [2]) isa Err # generalized RHS is not an independent hole
    hole = source_hole(source, [1, 1])
    @test hole isa SourceHole
    @test length(hole.external_types) == 1
    @test hole.external_types[1] isa TyScheme
    @test length(hole.external_types[1].vars) == 1
    binder = source.index.owner_binder[2]
    @test binder != 2 # binder identity is not the owning occurrence position
    uses = findall(==(binder), source.index.reference_binder)
    @test length(uses) == 2
    @test source.metadata[uses[1]][1] == TyFun(IntTy, IntTy)
    @test source.metadata[uses[2]][1] == TyFun(BoolTy, BoolTy)
    closed = prepare_source(
        AstTree(AppNode(AbsNode(:id, ConstNode(:one)), ConstNode(:one))),
        Dict{Symbol,TyLike}(:one => IntTy), IntTy,
    )
    closed_hole = source_hole(closed, [1, 1])
    @test closed_hole.external_types == TyLike[IntTy]
    closed_rhs_hole = source_hole(closed, [2])
    @test closed_rhs_hole isa SourceHole
    @test closed_rhs_hole.external_types == TyLike[]
    recursive = prepare_source(AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        Dict{Symbol,TyLike}(), TyFun(FloatTy, FloatTy))
    inner = source_hole(recursive, [1, 1])
    @test inner.external_types == TyLike[TyFun(FloatTy, FloatTy), FloatTy]
    closed_rhs = prepare_source(
        AstTree(AbsNode(:x, AppNode(
            AbsNode(:unused, ConstNode(:one)), IndexNode(0)))),
        Dict{Symbol,TyLike}(:one => IntTy), TyFun(BoolTy, IntTy),
    )
    closed_hole = source_hole(closed_rhs, [1, 2])
    @test closed_hole isa SourceHole
    @test closed_hole.external_types == TyLike[BoolTy]
end
