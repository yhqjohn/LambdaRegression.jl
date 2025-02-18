"""Procedural lexical snapshots for FunctionDef-aware crossover."""

using TestItems

using ..ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, IndexNode, LetNode, VarNode,
    children
using ..ASTs.LexicalContexts: BinderId
using ..TypeSystem:
    Ty, TyLike, TyFun, TyScheme, TyVar, _inst, collect_free_vars, generalize,
    make_tyvar_factory, prune, unify!
using ..Utils: Err

export ProceduralLexicalSnapshot, ProceduralLexicalSite
export ProceduralLexicalOccurrence, prepare_procedural_snapshot

"""One resolved De Bruijn occurrence in a procedural lexical snapshot."""
struct ProceduralLexicalOccurrence
    path::Tuple{Vararg{Int}}
    node::IndexNode
    index::Int
    binder::BinderId
    binder_scheme::TyLike
    use_type::TyLike
end

"""One procedural node with demand/offer and captured BinderId groups."""
struct ProceduralLexicalSite
    path::Tuple{Vararg{Int}}
    node::AstNode
    demand::TyLike
    offer::TyLike
    boundary_visible::Tuple
    binder_schemes::Dict{BinderId,TyLike}
    external_occurrences::Dict{BinderId,Vector{ProceduralLexicalOccurrence}}
    monomorphic_external::Bool
end

"""Result of one native procedural inference pass and one lexical walk."""
struct ProceduralLexicalSnapshot
    root_type::TyLike
    metadata::AbstractDict
    sites::Vector{ProceduralLexicalSite}
    occurrences::Vector{ProceduralLexicalOccurrence}
    binder_schemes::Dict{BinderId,TyLike}
end

mutable struct _ProceduralSnapshotSiteState
    path::Tuple{Vararg{Int}}
    node::AstNode
    demand::TyLike
    offer::TyLike
    boundary_visible::Tuple
    external_occurrences::Dict{BinderId,Vector{ProceduralLexicalOccurrence}}
end

@inline _procedural_is_direct_let_abs(parent, node) =
    parent isa AppNode && parent.func === node

# Keep lexical support explicit and vector-backed during the walk. In
# particular, ScopeNode intentionally has no `_procedural_snapshot_visible`
# method and therefore raises MethodError.
_procedural_snapshot_visible(
    ::AppNode,
    ::Int,
    ::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
) = copy(visible)

function _procedural_snapshot_visible(
    ::AbsNode,
    child_index::Int,
    path::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
)
    child_index == 1 || throw(ArgumentError("AbsNode has one child"))
    return BinderId[BinderId(:abs, path, 1); visible]
end

function _procedural_snapshot_visible(
    ::LetNode,
    child_index::Int,
    path::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
)
    child_index == 1 && return copy(visible)
    child_index == 2 || throw(ArgumentError("LetNode has two children"))
    return BinderId[BinderId(:let, path, 1); visible]
end

function _procedural_snapshot_visible(
    ::FunctionDefNode,
    child_index::Int,
    path::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
)
    child_index == 1 || throw(ArgumentError("FunctionDefNode has one child"))
    return BinderId[BinderId(:function_def, path, 1); visible]
end

function _procedural_snapshot_copy_tylike(
    value::TyLike,
    factory,
    mapping::Dict{TyVar,TyVar},
)
    copy_var(v::TyVar) = get!(mapping, v) do
        factory(v.level)
    end
    if value isa TyScheme
        vars = TyVar[copy_var(v) for v in value.vars]
        return TyScheme(vars, _inst(value.ty, copy_var))
    end
    return _inst(value, copy_var)
end

function _procedural_snapshot_copy_inputs(
    ctx::AbstractDict{Symbol,<:TyLike},
    extras::TyLike...,
)
    copy_factory = make_tyvar_factory()
    mapping = Dict{TyVar,TyVar}()
    result = Dict{Symbol,TyLike}()
    for (name, value) in ctx
        result[name] = _procedural_snapshot_copy_tylike(
            value,
            copy_factory,
            mapping,
        )
    end
    copied_extras = map(extras) do value
        _procedural_snapshot_copy_tylike(value, copy_factory, mapping)
    end
    return result, copied_extras
end

function _procedural_snapshot_copy_context(
    ctx::AbstractDict{Symbol,<:TyLike},
)
    result, _ = _procedural_snapshot_copy_inputs(ctx)
    return result
end

function _procedural_snapshot_normalize_bound(bound::TyLike)
    bound isa TyScheme || return prune(bound)
    body = prune(bound.ty)
    free = collect_free_vars(body)
    variables = TyVar[
        variable for variable in bound.vars
        if variable.instance === nothing && variable in free
    ]
    return TyScheme(variables, body)
end

function _procedural_snapshot_binder_scheme(
    node::AbsNode,
    parent,
    parent_level::Int,
    metadata,
)
    if parent isa AppNode && parent.func === node
        value = parent.arg
        haskey(metadata, value) || return nothing
        return generalize(prune(metadata[value][1]), parent_level)
    end
    haskey(metadata, node) || return nothing
    inferred = prune(metadata[node][1])
    inferred isa TyFun || return nothing
    return TyScheme(TyVar[], inferred.arg)
end

function _procedural_snapshot_binder_scheme(
    node::LetNode,
    parent_level::Int,
    metadata,
)
    haskey(metadata, node.value) || return nothing
    return generalize(prune(metadata[node.value][1]), parent_level)
end

function _procedural_snapshot_binder_scheme(
    node::FunctionDefNode,
    metadata,
)
    haskey(metadata, node) || return nothing
    return TyScheme(TyVar[], prune(metadata[node][1]))
end

function _procedural_snapshot_child_level(parent, child_index::Int, level::Int)
    if parent isa LetNode && child_index == 1
        return level + 1
    elseif parent isa AppNode && child_index == 2 && parent.func isa AbsNode
        return level + 1
    end
    return level
end

function _procedural_snapshot_add_occurrence!(
    active::Vector{_ProceduralSnapshotSiteState},
    occurrence::ProceduralLexicalOccurrence,
)
    for state in active
        occurrence.binder in state.boundary_visible || continue
        push!(
            get!(
                Vector{ProceduralLexicalOccurrence},
                state.external_occurrences,
                occurrence.binder,
            ),
            occurrence,
        )
    end
    return nothing
end

function _procedural_snapshot_site_binders(
    visible::Tuple,
    binder_schemes::Dict{BinderId,TyLike},
)
    result = Dict{BinderId,TyLike}()
    for binder in visible
        haskey(binder_schemes, binder) && (result[binder] = binder_schemes[binder])
    end
    return result
end

function _procedural_snapshot_finish_site(
    state::_ProceduralSnapshotSiteState,
    binder_schemes::Dict{BinderId,TyLike},
)
    site_binders = _procedural_snapshot_site_binders(
        state.boundary_visible,
        binder_schemes,
    )
    mono = all(
        scheme -> !(scheme isa TyScheme) || isempty(scheme.vars),
        values(site_binders),
    )
    return ProceduralLexicalSite(
        state.path,
        state.node,
        state.demand,
        state.offer,
        state.boundary_visible,
        site_binders,
        state.external_occurrences,
        mono,
    )
end

"""Prepare a FunctionDef-aware procedural lexical snapshot.

`ScopeNode` is deliberately not a population representation.  Although native
procedural inference can type it, the lexical walk has no adapter method for
it and raises `MethodError` rather than silently treating the IR as a site.
"""
function prepare_procedural_snapshot(
    term::Union{AstNode,AstTree},
    ctx::AbstractDict{Symbol,<:TyLike};
    factory=make_tyvar_factory(),
    target_ty::Union{Nothing,Ty}=nothing,
    site_path::Union{Nothing,Vector{Int}}=nothing,
)
    root = term isa AstTree ? term.root : term
    inference_ctx, detached_targets = target_ty === nothing ?
        (_procedural_snapshot_copy_context(ctx), ()) :
        _procedural_snapshot_copy_inputs(ctx, target_ty)
    metadata = IdDict{AstNode,Tuple{TyLike,TyLike}}()
    root_type = infer_procedural(root, inference_ctx; factory, meta=metadata)
    root_type isa Err && return root_type
    if target_ty !== nothing
        constrained = unify!(root_type, only(detached_targets))
        constrained isa Err && return constrained
        root_type = prune(root_type)
    end

    site_states = _ProceduralSnapshotSiteState[]
    occurrences = ProceduralLexicalOccurrence[]
    binder_schemes = Dict{BinderId,TyLike}()
    active = _ProceduralSnapshotSiteState[]

    function visit(
        node::AstNode,
        path::Vector{Int},
        visible::Vector{BinderId},
        level::Int,
        parent,
    )
        path_tuple = Tuple(path)
        visible_tuple = Tuple(visible)
        site_state = nothing
        if haskey(metadata, node) &&
           !_procedural_is_direct_let_abs(parent, node) &&
           (site_path === nothing || path == site_path)
            bounds = metadata[node]
            site_state = _ProceduralSnapshotSiteState(
                path_tuple,
                node,
                _procedural_snapshot_normalize_bound(bounds[1]),
                _procedural_snapshot_normalize_bound(bounds[2]),
                visible_tuple,
                Dict{BinderId,Vector{ProceduralLexicalOccurrence}}(),
            )
            push!(active, site_state)
            push!(site_states, site_state)
        end

        node isa VarNode && throw(ArgumentError(
            "typed-crossover population snapshots require De Bruijn IndexNode binders",
        ))

        if node isa IndexNode
            index = node.index
            index < length(visible) || throw(BoundsError(visible, index + 1))
            binder = visible[index + 1]
            scheme = get(binder_schemes, binder) do
                throw(ArgumentError("missing lexical scheme for $(binder)"))
            end
            haskey(metadata, node) ||
                throw(ArgumentError("missing metadata for IndexNode"))
            occurrence = ProceduralLexicalOccurrence(
                path_tuple,
                node,
                index,
                binder,
                scheme,
                _procedural_snapshot_normalize_bound(metadata[node][1]),
            )
            push!(occurrences, occurrence)
            _procedural_snapshot_add_occurrence!(active, occurrence)
        end

        for (child_index, child) in enumerate(children(node))
            child_path = copy(path)
            push!(child_path, child_index)
            child_visible = _procedural_snapshot_visible(
                node,
                child_index,
                path_tuple,
                visible,
            )

            if node isa AbsNode && child_index == 1
                binder = first(child_visible)
                scheme = _procedural_snapshot_binder_scheme(
                    node,
                    parent,
                    level,
                    metadata,
                )
                scheme === nothing || (binder_schemes[binder] = scheme)
            elseif node isa LetNode && child_index == 2
                binder = first(child_visible)
                scheme = _procedural_snapshot_binder_scheme(node, level, metadata)
                scheme === nothing || (binder_schemes[binder] = scheme)
            elseif node isa FunctionDefNode && child_index == 1
                binder = first(child_visible)
                scheme = _procedural_snapshot_binder_scheme(node, metadata)
                scheme === nothing || (binder_schemes[binder] = scheme)
            end

            visit(
                child,
                child_path,
                child_visible,
                _procedural_snapshot_child_level(node, child_index, level),
                node,
            )
        end

        if site_state !== nothing
            pop!(active)
        end
        return nothing
    end

    visit(root, Int[], BinderId[], 0, nothing)
    sites = ProceduralLexicalSite[
        _procedural_snapshot_finish_site(state, binder_schemes)
        for state in site_states
    ]
    return ProceduralLexicalSnapshot(
        root_type,
        metadata,
        sites,
        occurrences,
        binder_schemes,
    )
end

@testitem "procedural snapshot keeps FunctionDef self internal at root" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Procedural:
        FunctionDefNode, ProceduralLexicalSnapshot, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: TyLike

    snapshot = prepare_procedural_snapshot(
        AstTree(FunctionDefNode(AbsNode(:x, IndexNode(1)))),
        Dict{Symbol,TyLike}(),
    )
    @test snapshot isa ProceduralLexicalSnapshot
    snapshot isa ProceduralLexicalSnapshot || return
    @test getfield.(snapshot.sites, :path) == [(), (1,), (1, 1)]
    root = only(filter(site -> isempty(site.path), snapshot.sites))
    body = only(filter(site -> site.path == (1,), snapshot.sites))
    @test isempty(root.external_occurrences)
    @test length(body.external_occurrences) == 1
    occurrence = only(values(body.external_occurrences))[1]
    @test occurrence.binder == first(body.boundary_visible)
    @test occurrence.binder.kind === :function_def

    targeted = prepare_procedural_snapshot(
        AstTree(FunctionDefNode(AbsNode(:x, IndexNode(1)))),
        Dict{Symbol,TyLike}();
        site_path=[1],
    )
    @test getfield.(targeted.sites, :path) == [(1,)]
    targeted_occurrence = only(values(only(targeted.sites).external_occurrences))[1]
    @test targeted_occurrence.path == (1, 1)
    @test targeted_occurrence.binder == first(body.boundary_visible)
end

@testitem "procedural snapshot distinguishes nested FunctionDef and lambda binders" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Procedural:
        FunctionDefNode, ProceduralLexicalSnapshot, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: TyLike

    body = AbsNode(:x, FunctionDefNode(AbsNode(:y, AppNode(IndexNode(1), IndexNode(0)))))
    snapshot = prepare_procedural_snapshot(
        AstTree(FunctionDefNode(body)),
        Dict{Symbol,TyLike}(),
    )
    @test snapshot isa ProceduralLexicalSnapshot
    snapshot isa ProceduralLexicalSnapshot || return
    occurrence_binders = [occurrence.binder for occurrence in snapshot.occurrences]
    @test length(occurrence_binders) == 2
    @test occurrence_binders[1].kind === :function_def
    @test occurrence_binders[2].kind === :abs
    @test occurrence_binders[1] != occurrence_binders[2]
end

@testitem "procedural snapshot rejects ScopeNode as population form" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode, IndexNode
    using LambdaRegression.Procedural: ScopeNode, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyLike

    @test_throws MethodError prepare_procedural_snapshot(
        AstTree(ScopeNode([ConstNode(:one)], IndexNode(0))),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
end

@testitem "procedural snapshot does not bind caller type context" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Procedural:
        FunctionDefNode, ProceduralLexicalSnapshot, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: TyFun, TyLike, make_tyvar_factory

    factory = make_tyvar_factory()
    α = factory(0)
    ctx = Dict{Symbol,TyLike}(:f => TyFun(α, α))
    snapshot = prepare_procedural_snapshot(
        AstTree(FunctionDefNode(AppNode(ConstNode(:f), IndexNode(0)))),
        ctx,
    )
    @test snapshot isa ProceduralLexicalSnapshot
    @test α.instance === nothing
end

@testitem "procedural snapshot preserves shared caller type relations" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode
    using LambdaRegression.Procedural: prepare_procedural_snapshot
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, make_tyvar_factory
    using LambdaRegression.Utils: Err

    factory = make_tyvar_factory()
    shared = factory(0)
    ctx = Dict{Symbol,TyLike}(
        :X => shared,
        :Y => shared,
        :USE_INT => TyFun(IntTy, IntTy),
        :USE_BOOL => TyFun(BoolTy, IntTy),
        :PAIR => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    term = AppNode(
        AppNode(
            ConstNode(:PAIR),
            AppNode(ConstNode(:USE_INT), ConstNode(:X)),
        ),
        AppNode(ConstNode(:USE_BOOL), ConstNode(:Y)),
    )
    @test prepare_procedural_snapshot(AstTree(term), ctx) isa Err
    @test shared.instance === nothing
end
