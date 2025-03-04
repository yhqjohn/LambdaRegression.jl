using ..ASTs.NodeASTs: AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode, LetNode
using ..ASTs.NodeASTs: children
using ..Procedural: FunctionDefNode, ScopeNode
import ..Procedural
import ..TypeSystem

"""A source occurrence, distinct even when two leaves or subtrees are identical."""
struct SourceOccurrence
    id::Int
    node::AstNode
    children::Vector{SourceOccurrence}
end

# Unlike deepcopy, this copies each syntactic occurrence independently. Julia
# object sharing is not semantic graph sharing in the source λ language.
function _copy_source_occurrence(n::AstNode)
    n isa Union{ConstNode,IndexNode,VarNode} && return n
    n isa AbsNode && return AbsNode(n.name, _copy_source_occurrence(n.body))
    n isa AppNode && return AppNode(_copy_source_occurrence(n.func), _copy_source_occurrence(n.arg))
    n isa LetNode && return LetNode(n.name, _copy_source_occurrence(n.value), _copy_source_occurrence(n.body))
    n isa FunctionDefNode && return FunctionDefNode(_copy_source_occurrence(n.body))
    n isa ScopeNode && return ScopeNode(AstNode[_copy_source_occurrence(b) for b in n.bindings],
        _copy_source_occurrence(n.body))
    throw(ArgumentError("unsupported source node $(typeof(n))"))
end

struct OccurrenceAdapter end

function TypeSystem.term_view(::OccurrenceAdapter, occurrence::SourceOccurrence)
    n, c = occurrence.node, occurrence.children
    n isa ConstNode && return (:const, n.name)
    n isa VarNode && return (:var, n.name)
    n isa IndexNode && return (:var, n.index)
    n isa AbsNode && return (:abs, n.name, c[1])
    n isa AppNode && return (:app, c[1], c[2])
    n isa LetNode && return (:let, n.name, c[1], c[2])
    n isa FunctionDefNode && return (:fix, nothing, c[1])
    n isa ScopeNode && return (:letseq, c[1:end-1], c[end])
    throw(ArgumentError("unsupported source node $(typeof(n))"))
end

function TypeSystem.constrain_fixpoint(::OccurrenceAdapter, o::SourceOccurrence, ty::TypeSystem.Ty, factory)
    return TypeSystem.constrain_fixpoint(Procedural._ProceduralTermAdapter(), o.node, ty, factory)
end

# W writes metadata through this small dictionary adapter. Integer keys give
# occurrence identity, unlike IdDict keys made from immutable IndexNode(0).
struct OccurrenceMetadata <: AbstractDict{SourceOccurrence,Tuple{TypeSystem.TyLike,TypeSystem.TyLike}}
    data::Dict{Int,Tuple{TypeSystem.TyLike,TypeSystem.TyLike}}
end
OccurrenceMetadata() = OccurrenceMetadata(Dict{Int,Tuple{TypeSystem.TyLike,TypeSystem.TyLike}}())
Base.setindex!(m::OccurrenceMetadata, value, o::SourceOccurrence) = (m.data[o.id] = value)
Base.getindex(m::OccurrenceMetadata, o::SourceOccurrence) = m.data[o.id]
Base.haskey(m::OccurrenceMetadata, o::SourceOccurrence) = haskey(m.data, o.id)

"""Expected refusal at the restricted search boundary, not a source typing error."""
struct SourceBoundaryError
    reason::Symbol
    occurrence::Int
end

"""Owned, immutable-by-convention source and per-occurrence type evidence."""
struct SourceAnalysis
    tree::AstTree
    root::SourceOccurrence
    occurrences::Vector{SourceOccurrence}
    metadata::OccurrenceMetadata
    root_type::TypeSystem.Ty
end

"""One fixed-ancestor interface. External binders are ordered outermost first."""
struct SourceHole
    source::SourceAnalysis
    occurrence::SourceOccurrence
    path::Vector{Int}
    external_types::Vector{TypeSystem.TyLike}
    anchor::Base.RefValue{Nothing}
end

"""
    source_hole(source, path=Int[])

Select an ordinary occurrence in an already inferred source. The path uses
one-based source child numbers, including FunctionDef's self-binding edge.
External bindings remain fixed, with their original schemes; only their use
instances will become concrete leaves. No context-equivalence search occurs.

The first implementation declines generalising RHS regions whose result still
has free type variables, and declines compatibility Scope traversal. A body
under an already determined recursive self binder is allowed. These are
projection restrictions, not restrictions on the complete AST generator.
"""
function source_hole(source::SourceAnalysis, path::AbstractVector{<:Integer}=Int[])
    occurrence = source.root
    parent = nothing
    outer = TypeSystem.TyLike[]
    level = 0
    for edge in path
        1 <= edge <= length(occurrence.children) ||
            return Err(SourceBoundaryError(:invalid_path, occurrence.id))
        n = occurrence.node
        if n isa AbsNode
            if parent !== nothing && parent.node isa AppNode && parent.children[1] === occurrence
                rhs = parent.children[2]
                scheme = TypeSystem.generalize(source.metadata[rhs][1], level)
                push!(outer, scheme)
            else
                ty = source.metadata[occurrence][1]
                ty isa TypeSystem.TyFun || return Err(SourceBoundaryError(:nontransparent_binder, occurrence.id))
                push!(outer, ty.arg)
            end
        elseif n isa FunctionDefNode
            push!(outer, source.metadata[occurrence][1])
        elseif n isa LetNode && edge == 2
            push!(outer, TypeSystem.generalize(source.metadata[occurrence.children[1]][1], level))
        elseif n isa ScopeNode
            return Err(SourceBoundaryError(:scope_compatibility, occurrence.id))
        end
        child = occurrence.children[edge]
        is_rhs = (n isa LetNode && edge == 1) ||
            (n isa AppNode && n.func isa AbsNode && edge == 2)
        if is_rhs
            isempty(TypeSystem.collect_free_vars(source.metadata[child][1])) ||
                return Err(SourceBoundaryError(:generalising_rhs, child.id))
            level += 1
        end
        parent, occurrence = occurrence, child
    end
    haskey(source.metadata, occurrence) ||
        return Err(SourceBoundaryError(:binding_artifact, occurrence.id))
    return SourceHole(source, occurrence, Int[path...], outer, Ref{Nothing}(nothing))
end

"""
    prepare_source(tree, type_context, target; max_nodes=100_000, max_depth=512)

Infer a complete source tree once using distinct syntactic occurrences, then
constrain its root to the requested closed type. This is an import-boundary
operation, not a proposal operation. `type_context` follows the source system's
closed-monotype/explicit-scheme discipline. Let-App remains generalising and
FunctionDef retains its monomorphic self rule. Metadata on the binding-artifact
lambda of Let-App is deliberately absent; it is not an ordinary typed hole.

Own a copy of the source. Return `Err` for typing failure or a bounded extraction
refusal. Unknown node extensions fail explicitly. Unresolved types may remain
inside the source: subsequent projection must refuse or hide them, not guess.
"""
function prepare_source(tree::AstTree, type_context, target::TypeSystem.Ty;
                        max_nodes::Int=100_000, max_depth::Int=512)
    max_nodes > 0 && max_depth >= 0 || throw(ArgumentError("invalid source bounds"))
    isempty(TypeSystem.collect_free_vars(target)) ||
        return Err(SourceBoundaryError(:open_target, 0))
    ctx = Dict{Symbol,TypeSystem.TyLike}(type_context)
    TypeSystem.validate_ctx(ctx)
    # Bound the traversal before copying, so cyclic or excessive source syntax
    # cannot turn preparation into an unbounded recursive copy.
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
    owned = AstTree(_copy_source_occurrence(tree.root))
    occurrences = SourceOccurrence[]
    serial = Ref(0)
    function wrap(n)
        serial[] += 1
        id = serial[]
        c = SourceOccurrence[wrap(child) for child in children(n)]
        o = SourceOccurrence(id, n, c)
        push!(occurrences, o)
        return o
    end
    root = wrap(owned.root)
    meta = OccurrenceMetadata()
    ty = TypeSystem.infer_term(root, ctx, OccurrenceAdapter();
        factory=TypeSystem.make_tyvar_factory(), meta)
    ty isa Err && return ty
    constrained = TypeSystem.unify!(ty, target)
    constrained isa Err && return constrained
    # W annotations share the inference variables until this point. Only now
    # materialise the lower bounds; independently freshening them loses links.
    for (id, (lower, upper)) in meta.data
        meta.data[id] = (TypeSystem.prune(lower), upper)
    end
    return SourceAnalysis(owned, root, occurrences, meta, TypeSystem.prune(ty))
end

@testitem "TypedSearch source evidence distinguishes polymorphic occurrences" begin
    using LambdaRegression.TypedSearch: prepare_source
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
    ids = filter(o -> o.node isa ConstNode && o.node.name == :ID, analysis.occurrences)
    @test length(ids) == 2
    @test analysis.metadata[ids[1]][1] == TyFun(IntTy, IntTy)
    @test analysis.metadata[ids[2]][1] == TyFun(BoolTy, BoolTy)
    @test ids[1].id != ids[2].id
end

@testitem "TypedSearch source root constraints and recursive boundaries" begin
    using LambdaRegression.TypedSearch: prepare_source
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.Procedural: FunctionDefNode
    using LambdaRegression.Utils: Err
    ctx = Dict{Symbol,TyLike}()
    tree = AstTree(AbsNode(:x, IndexNode(0)))
    a = prepare_source(tree, ctx, TyFun(FloatTy, FloatTy))
    @test a.metadata[a.root.children[1]][1] == FloatTy
    recursive = prepare_source(AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        ctx, TyFun(FloatTy, FloatTy))
    @test recursive.root_type == TyFun(FloatTy, FloatTy)
    @test prepare_source(tree, ctx, FloatTy) isa Err
    @test prepare_source(tree, ctx, TyFun(FloatTy, FloatTy); max_nodes=1) isa Err
end

@testitem "TypedSearch local holes retain schemes and recursive self" begin
    using LambdaRegression.TypedSearch: prepare_source, source_hole
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy, FloatTy
    using LambdaRegression.Procedural: FunctionDefNode
    using LambdaRegression.Utils: Err
    ctx = Dict{Symbol,TyLike}(:PAIR => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy, :yes => BoolTy)
    body = AppNode(AppNode(ConstNode(:PAIR), AppNode(IndexNode(0), ConstNode(:one))),
        AppNode(IndexNode(0), ConstNode(:yes)))
    tree = AstTree(AppNode(AbsNode(:id, body), AbsNode(:x, IndexNode(0))))
    source = prepare_source(tree, ctx, TyProd(IntTy, BoolTy))
    @test source_hole(source, [1]) isa Err # No independent Abs judgment in Let-App.
    @test source_hole(source, [2]) isa Err # Generalising definition, not one use.
    hole = source_hole(source, [1, 1])
    @test length(hole.external_types) == 1
    @test hole.external_types[1] isa TyScheme
    @test length(hole.external_types[1].vars) == 1
    uses = filter(o -> o.node isa IndexNode && o.node.index == 0 &&
        o.id != source.root.children[2].children[1].id, source.occurrences)
    @test source.metadata[uses[1]][1] == TyFun(IntTy, IntTy)
    @test source.metadata[uses[2]][1] == TyFun(BoolTy, BoolTy)
    recursive = prepare_source(AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        Dict{Symbol,TyLike}(), TyFun(FloatTy, FloatTy))
    inner = source_hole(recursive, [1, 1])
    @test inner.external_types == TyLike[TyFun(FloatTy, FloatTy), FloatTy]
end
