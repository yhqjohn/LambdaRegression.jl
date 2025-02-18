"""
    Core lexical snapshots for typed crossover.

This is the small analysis seam shared by future directed crossover backends.
Inference supplies each node's lower (`demand`) and reusable upper (`offer`)
bound.  A separate structural walk resolves De Bruijn occurrences to stable
`BinderId`s and records the dependencies visible at each subtree boundary.

Only the core `AbsNode`/`AppNode`/`LetNode` language is supported here.  Procedural
extension nodes deliberately do not acquire fallback methods in this file;
`ScopeNode` and `FunctionDefNode` therefore fail at the lexical boundary until
their role-aware snapshot is added by the procedural backend.
"""

using TestItems

using ...ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, IndexNode, LetNode, VarNode,
    children
using ...ASTs.LexicalContexts: BinderId
using ...TypeSystem:
    Ty, TyLike, TyFun, TyScheme, TyVar, generalize, infer_term, NodeTermAdapter,
    prune, make_tyvar_factory, collect_free_vars, unify!
using ...Utils: Err

export CoreLexicalSnapshot, CoreLexicalSite, CoreLexicalOccurrence
export prepare_core_snapshot

"""One resolved De Bruijn occurrence in a prepared lexical snapshot."""
struct CoreLexicalOccurrence
    path::Tuple{Vararg{Int}}
    node::IndexNode
    index::Int
    binder::BinderId
    binder_scheme::TyLike
    use_type::TyLike
end

"""One replaceable node together with its lexical boundary and dependencies."""
struct CoreLexicalSite
    path::Tuple{Vararg{Int}}
    node::AstNode
    demand::TyLike
    offer::TyLike
    boundary_visible::Tuple
    binder_schemes::Dict{BinderId,TyLike}
    external_occurrences::Dict{BinderId,Vector{CoreLexicalOccurrence}}
    monomorphic_external::Bool
end

"""Result of one core inference pass and one lexical traversal."""
struct CoreLexicalSnapshot
    root_type::TyLike
    metadata::AbstractDict
    sites::Vector{CoreLexicalSite}
    occurrences::Vector{CoreLexicalOccurrence}
    binder_schemes::Dict{BinderId,TyLike}
end

mutable struct _SnapshotSiteState
    path::Tuple{Vararg{Int}}
    node::AstNode
    demand::TyLike
    offer::TyLike
    boundary_visible::Tuple
    external_occurrences::Dict{BinderId,Vector{CoreLexicalOccurrence}}
end

@inline _is_direct_let_abs(parent, node) =
    parent isa AppNode && parent.func === node

function _snapshot_meta_dict()
    return IdDict{AstNode,Tuple{TyLike,TyLike}}()
end

function _snapshot_normalize_bound(bound::TyLike)
    bound isa TyScheme || return prune(bound)
    body = prune(bound.ty)
    free = collect_free_vars(body)
    variables = TyVar[
        variable for variable in bound.vars
        if variable.instance === nothing && variable in free
    ]
    return TyScheme(variables, body)
end

function _snapshot_binder_scheme(
    node::AbsNode,
    parent,
    parent_level::Int,
    metadata,
)
    # `(λx. M) N` is the package's let-like form.  The argument is generalized
    # at the application level and is the scheme visible in M; the AbsNode
    # itself intentionally has no inference metadata in this case.
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

function _snapshot_binder_scheme(
    node::LetNode,
    parent_level::Int,
    metadata,
)
    haskey(metadata, node.value) || return nothing
    return generalize(prune(metadata[node.value][1]), parent_level)
end

function _snapshot_child_level(parent, child_index::Int, level::Int)
    # Keep this synchronized with TypeSystem._infer: only value/RHS regions
    # enter a fresh generalisation level; ordinary λ bodies do not.
    if parent isa LetNode && child_index == 1
        return level + 1
    elseif parent isa AppNode && child_index == 2 && parent.func isa AbsNode
        return level + 1
    end
    return level
end

_snapshot_child_visible(
    ::AppNode,
    ::Int,
    ::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
) = copy(visible)

function _snapshot_child_visible(
    ::AbsNode,
    child_index::Int,
    path::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
)
    child_index == 1 || throw(ArgumentError("AbsNode has one child"))
    return BinderId[BinderId(:abs, path, 1); visible]
end

function _snapshot_child_visible(
    ::LetNode,
    child_index::Int,
    path::Tuple{Vararg{Int}},
    visible::Vector{BinderId},
)
    child_index == 1 && return copy(visible)
    child_index == 2 || throw(ArgumentError("LetNode has two children"))
    return BinderId[BinderId(:let, path, 1); visible]
end

function _snapshot_site_binders(
    visible::Tuple,
    binder_schemes::Dict{BinderId,TyLike},
)
    result = Dict{BinderId,TyLike}()
    for binder in visible
        haskey(binder_schemes, binder) && (result[binder] = binder_schemes[binder])
    end
    return result
end

function _snapshot_add_occurrence!(
    active::Vector{_SnapshotSiteState},
    occurrence::CoreLexicalOccurrence,
)
    for state in active
        occurrence.binder in state.boundary_visible || continue
        push!(get!(Vector{CoreLexicalOccurrence}, state.external_occurrences, occurrence.binder), occurrence)
    end
    return nothing
end

function _snapshot_finish_site(
    state::_SnapshotSiteState,
    binder_schemes::Dict{BinderId,TyLike},
)
    site_binders = _snapshot_site_binders(state.boundary_visible, binder_schemes)
    mono = all(
        scheme -> !(scheme isa TyScheme) || isempty(scheme.vars),
        values(site_binders),
    )
    return CoreLexicalSite(
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

"""Prepare core AST sites using one metadata-producing inference and one walk."""
function prepare_core_snapshot(
    tree::AstTree,
    ctx::AbstractDict{Symbol,<:TyLike};
    factory=make_tyvar_factory(),
    target_ty::Union{Nothing,Ty}=nothing,
    site_path::Union{Nothing,Vector{Int}}=nothing,
)
    metadata = _snapshot_meta_dict()
    inference_ctx, detached_targets = target_ty === nothing ?
        (_detached_type_ctx(ctx), ()) :
        _detached_type_inputs(ctx, target_ty)
    root_type = infer_term(
        tree.root,
        inference_ctx,
        NodeTermAdapter();
        factory,
        meta=metadata,
    )
    root_type isa Err && return root_type
    if target_ty !== nothing
        constrained = unify!(root_type, only(detached_targets))
        constrained isa Err && return constrained
        root_type = prune(root_type)
    end

    site_states = _SnapshotSiteState[]
    occurrences = CoreLexicalOccurrence[]
    binder_schemes = Dict{BinderId,TyLike}()
    active = _SnapshotSiteState[]

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
           !_is_direct_let_abs(parent, node) &&
           (site_path === nothing || path == site_path)
            bounds = metadata[node]
            site_state = _SnapshotSiteState(
                path_tuple,
                node,
                _snapshot_normalize_bound(bounds[1]),
                _snapshot_normalize_bound(bounds[2]),
                visible_tuple,
                Dict{BinderId,Vector{CoreLexicalOccurrence}}(),
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
            haskey(metadata, node) || throw(ArgumentError("missing metadata for IndexNode"))
            occurrence = CoreLexicalOccurrence(
                path_tuple,
                node,
                index,
                binder,
                scheme,
                _snapshot_normalize_bound(metadata[node][1]),
            )
            push!(occurrences, occurrence)
            _snapshot_add_occurrence!(active, occurrence)
        end

        for (child_index, child) in enumerate(children(node))
            child_path = copy(path)
            push!(child_path, child_index)
            child_visible = _snapshot_child_visible(
                node,
                child_index,
                path_tuple,
                visible,
            )

            # Register the binder introduced by this edge before entering its
            # body, so an IndexNode can resolve it immediately.
            if node isa AbsNode && child_index == 1
                binder = first(child_visible)
                scheme = _snapshot_binder_scheme(node, parent, level, metadata)
                scheme === nothing || (binder_schemes[binder] = scheme)
            elseif node isa LetNode && child_index == 2
                binder = first(child_visible)
                scheme = _snapshot_binder_scheme(node, level, metadata)
                scheme === nothing || (binder_schemes[binder] = scheme)
            end

            visit(
                child,
                child_path,
                child_visible,
                _snapshot_child_level(node, child_index, level),
                node,
            )
        end

        if site_state !== nothing
            pop!(active)
        end
        return nothing
    end

    visit(tree.root, Int[], BinderId[], 0, nothing)
    sites = CoreLexicalSite[
        _snapshot_finish_site(state, binder_schemes)
        for state in site_states
    ]
    return CoreLexicalSnapshot(root_type, metadata, sites, occurrences, binder_schemes)
end

@testitem "core lexical snapshot resolves lambda capture" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations: CoreLexicalSnapshot, prepare_core_snapshot
    using LambdaRegression.TypeSystem: TyLike

    snapshot = prepare_core_snapshot(
        AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1)))),
        Dict{Symbol,TyLike}(),
    )
    @test snapshot isa CoreLexicalSnapshot
    snapshot isa CoreLexicalSnapshot || return
    @test getfield.(snapshot.sites, :path) == [(), (1,), (1, 1)]
    inner = only(filter(site -> site.path == (1,), snapshot.sites))
    @test length(inner.external_occurrences) == 1
    @test only(values(inner.external_occurrences))[1].binder ==
        first(inner.boundary_visible)

    targeted = prepare_core_snapshot(
        AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1)))),
        Dict{Symbol,TyLike}();
        site_path=[1],
    )
    @test getfield.(targeted.sites, :path) == [(1,)]
    targeted_occurrence = only(values(only(targeted.sites).external_occurrences))[1]
    @test targeted_occurrence.path == (1, 1)
    @test targeted_occurrence.binder == first(inner.boundary_visible)
end

@testitem "core lexical snapshot resolves let body" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, IndexNode, LetNode, ConstNode
    using LambdaRegression.Mutations.Populations: CoreLexicalSnapshot, prepare_core_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyLike

    snapshot = prepare_core_snapshot(
        AstTree(LetNode(:x, ConstNode(:one), IndexNode(0))),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    @test snapshot isa CoreLexicalSnapshot
    snapshot isa CoreLexicalSnapshot || return
    body = only(filter(site -> site.path == (2,), snapshot.sites))
    @test length(body.external_occurrences) == 1
    @test only(values(body.external_occurrences))[1].binder ==
        first(body.boundary_visible)
end

@testitem "core lexical snapshot omits direct let-like abstraction" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations: CoreLexicalSnapshot, prepare_core_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyLike

    snapshot = prepare_core_snapshot(
        AstTree(AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one))),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    @test snapshot isa CoreLexicalSnapshot
    snapshot isa CoreLexicalSnapshot || return
    @test !any(site -> site.path == (1,), snapshot.sites)
    @test any(site -> site.path == (1, 1), snapshot.sites)
end

@testitem "core lexical snapshot retains polymorphic binder schemes" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode, LetNode
    using LambdaRegression.Mutations.Populations: CoreLexicalSnapshot, prepare_core_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyLike, TyScheme

    identity = AbsNode(:x, IndexNode(0))
    body = AppNode(IndexNode(0), ConstNode(:one))
    snapshot = prepare_core_snapshot(
        AstTree(LetNode(:id, identity, body)),
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    @test snapshot isa CoreLexicalSnapshot
    snapshot isa CoreLexicalSnapshot || return
    body_site = only(filter(site -> site.path == (2,), snapshot.sites))
    @test !body_site.monomorphic_external
    @test any(
        scheme -> scheme isa TyScheme && !isempty(scheme.vars),
        values(body_site.binder_schemes),
    )
end

@testitem "core lexical snapshot does not bind caller context" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations.Populations: CoreLexicalSnapshot, prepare_core_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, make_tyvar_factory

    factory = make_tyvar_factory()
    caller_owned = factory(0)
    ctx = Dict{Symbol,TyLike}(
        :APPLY => TyFun(caller_owned, IntTy),
        :ONE => IntTy,
    )
    snapshot = prepare_core_snapshot(
        AstTree(AppNode(ConstNode(:APPLY), ConstNode(:ONE))),
        ctx,
    )
    @test snapshot isa CoreLexicalSnapshot
    @test caller_owned.instance === nothing
end

@testitem "core lexical snapshot rejects named-bound population variables" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, VarNode
    using LambdaRegression.Mutations.Populations: prepare_core_snapshot
    using LambdaRegression.TypeSystem: TyLike

    @test_throws ArgumentError prepare_core_snapshot(
        AstTree(AbsNode(:x, VarNode(:x))),
        Dict{Symbol,TyLike}(),
    )
end
