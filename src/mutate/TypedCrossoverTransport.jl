"""Binder-aware, mutation-free transport of a donor subtree."""

using TestItems

using ...ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode,
    children
using ...ASTs.LexicalContexts: BinderId
using ...Utils: Err

export TransportMissingBindingErr, TransportTargetBindingErr
export TransportRoleMismatchErr, TransportUnresolvedIndexErr
export transport_subtree

struct TransportMissingBindingErr
    binder::BinderId
end

struct TransportTargetBindingErr
    binder::BinderId
end

struct TransportRoleMismatchErr
    source::BinderId
    target::BinderId
end

struct TransportUnresolvedIndexErr
    index::Int
end

@inline _transport_path(site) = site.path
@inline _transport_boundary(site) = site.boundary_visible

Base.@noinline function _transport_path_vector(site)
    Base.@nospecialize site
    path = _transport_path(site)
    result = Int[]
    sizehint!(result, length(path))
    for index in path
        push!(result, index)
    end
    return result
end

Base.@noinline function _transport_boundary_vector(site)
    Base.@nospecialize site
    boundary = _transport_boundary(site)
    result = BinderId[]
    sizehint!(result, length(boundary))
    for binder in boundary
        push!(result, binder)
    end
    return result
end

@inline _transport_introduced(::AppNode, ::Int) = 0
@inline _transport_introduced(::AbsNode, child_index::Int) =
    child_index == 1 ? 1 : throw(BoundsError())
@inline _transport_introduced(::FunctionDefNode, child_index::Int) =
    child_index == 1 ? 1 : throw(BoundsError())

function _transport_role_ok(source::BinderId, target::BinderId)
    source.kind === :function_def && target.kind === :function_def && return true
    target.kind === :function_def && return false
    source.kind === :function_def && return false
    return true
end

function _transport_validate_mapping(
    donor_boundary::Vector{BinderId},
    recipient_boundary::Vector{BinderId},
    mapping::AbstractDict,
)
    recipient_set = Set(recipient_boundary)
    for (source, target) in mapping
        source isa BinderId || return Err(ArgumentError("transport mapping keys must be BinderId"))
        target isa BinderId || return Err(ArgumentError("transport mapping values must be BinderId"))
        source in donor_boundary || return Err(TransportMissingBindingErr(source))
        target in recipient_set || return Err(TransportTargetBindingErr(target))
        _transport_role_ok(source, target) ||
            return Err(TransportRoleMismatchErr(source, target))
    end
    return nothing
end

function _transport_node_at_path(root::AstNode, path::Vector{Int})
    node = root
    for index in path
        kids = children(node)
        1 <= index <= length(kids) || return nothing
        node = kids[index]
    end
    return node
end

function _transport_rewrite(
    node::AstNode,
    local_depth::Int,
    donor_boundary::Vector{BinderId},
    recipient_boundary::Vector{BinderId},
    mapping::AbstractDict,
)
    if node isa IndexNode
        index = node.index
        index < 0 && return Err(TransportUnresolvedIndexErr(index))
        index < local_depth && return IndexNode(index)
        source_position = index - local_depth
        source_position < length(donor_boundary) ||
            return Err(TransportUnresolvedIndexErr(index))
        source = donor_boundary[source_position + 1]
        haskey(mapping, source) || return Err(TransportMissingBindingErr(source))
        target = mapping[source]
        target_position = findfirst(==(target), recipient_boundary)
        target_position === nothing && return Err(TransportTargetBindingErr(target))
        return IndexNode(local_depth + target_position - 1)
    elseif node isa ConstNode || node isa VarNode
        return deepcopy(node)
    end

    rewritten_children = AstNode[]
    for (child_index, child) in enumerate(children(node))
        # Index arithmetic avoids comparing locally introduced BinderIds across trees.
        rewritten = _transport_rewrite(
            child,
            local_depth + _transport_introduced(node, child_index),
            donor_boundary,
            recipient_boundary,
            mapping,
        )
        rewritten isa Err && return rewritten
        push!(rewritten_children, rewritten)
    end

    if node isa AbsNode
        return AbsNode(node.name, rewritten_children[1])
    elseif node isa AppNode
        return AppNode(rewritten_children[1], rewritten_children[2])
    elseif node isa FunctionDefNode
        return FunctionDefNode(rewritten_children[1])
    end
    throw(MethodError(_transport_rewrite, (node,)))
end

"""Copy donor site into recipient context using an explicit BinderId mapping.

The donor subtree is never mutated.  References to binders introduced inside
the donor subtree retain their original structural identities; only references
to `donor_site.boundary_visible` are mapped to the recipient boundary.  A
FunctionDef self binder may map only to another FunctionDef self binder.
"""
function transport_subtree(
    donor_tree::AstTree,
    donor_site,
    recipient_site,
    mapping::AbstractDict,
)
    donor_boundary = _transport_boundary_vector(donor_site)
    recipient_boundary = _transport_boundary_vector(recipient_site)
    validation = _transport_validate_mapping(
        donor_boundary,
        recipient_boundary,
        mapping,
    )
    validation === nothing || return validation

    source = _transport_node_at_path(donor_tree.root, _transport_path_vector(donor_site))
    source === nothing && return Err(ArgumentError("donor site path is not present"))
    rewritten = _transport_rewrite(
        source,
        0,
        donor_boundary,
        recipient_boundary,
        mapping,
    )
    rewritten isa Err && return rewritten
    return AstTree(rewritten)
end

@testitem "binder transport widens tuple donor paths only at the boundary" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations.Populations: transport_subtree
    using LambdaRegression.Utils: Err

    tree = AstTree(AppNode(ConstNode(:shallow), AbsNode(:x, ConstNode(:deep))))
    root = tree.root
    paths_and_nodes = (
        ((), root),
        ((1,), ConstNode(:shallow)),
        ((2, 1), ConstNode(:deep)),
    )
    for (site_path, expected) in paths_and_nodes
        site = (path=site_path, boundary_visible=())
        @test site.path isa Tuple
        path_vector = LambdaRegression.Mutations.Populations._transport_path_vector(site)
        @test path_vector isa Vector{Int}
        @test path_vector == collect(site_path)
        @test LambdaRegression.Mutations.Populations._transport_node_at_path(
            root,
            path_vector,
        ) == expected

        transported = transport_subtree(tree, site, site, Dict())
        @test transported isa AstTree
        transported isa AstTree || continue
        @test transported.root == expected
    end

    missing_site = (path=(2, 1, 1), boundary_visible=())
    @test LambdaRegression.Mutations.Populations._transport_node_at_path(
        root,
        collect(missing_site.path),
    ) === nothing
    missing = transport_subtree(tree, missing_site, missing_site, Dict())
    @test missing isa Err
    missing isa Err || return
    @test missing.value isa ArgumentError
    @test sprint(showerror, missing.value) == "ArgumentError: donor site path is not present"
end

@testitem "binder transport uses runtime boundaries of different lengths" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree
    using LambdaRegression.TypeSystem: TyLike

    donor_tree = AstTree(
        AbsNode(
            :outer,
            AbsNode(
                :middle,
                AbsNode(:inner, AppNode(IndexNode(2), IndexNode(1))),
            ),
        ),
    )
    recipient_tree = AstTree(
        AbsNode(:a, AbsNode(:b, AbsNode(:c, AbsNode(:d, AbsNode(:e, IndexNode(0)))))),
    )
    donor_before = deepcopy(donor_tree)
    donor_snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_typed_analysis(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = typed_analysis_site(
        donor_snapshot,
        only(typed_analysis_site_ids(donor_snapshot; site_path=[1, 1, 1])),
    )
    recipient_site = typed_analysis_site(
        recipient_snapshot,
        only(typed_analysis_site_ids(recipient_snapshot; site_path=[1, 1, 1, 1, 1])),
    )
    donor_boundary = donor_site.boundary_visible
    recipient_boundary = recipient_site.boundary_visible
    @test length(donor_boundary) == 3
    @test length(recipient_boundary) == 5

    transported = transport_subtree(
        donor_tree,
        donor_site,
        recipient_site,
        Dict(
            donor_boundary[3] => recipient_boundary[5],
            donor_boundary[2] => recipient_boundary[4],
        ),
    )
    @test transported isa AstTree
    transported isa AstTree || return
    @test transported.root == AppNode(IndexNode(4), IndexNode(3))
    @test donor_tree == donor_before
end

@testitem "binder transport maps an arbitrary ancestor" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree
    using LambdaRegression.TypeSystem: IntTy, TyLike

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    recipient_tree = AstTree(AbsNode(:z, AbsNode(:q, IndexNode(0))))
    donor_snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_typed_analysis(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = typed_analysis_site(
        donor_snapshot, only(typed_analysis_site_ids(donor_snapshot; site_path=[1])),
    )
    recipient_site = typed_analysis_site(
        recipient_snapshot,
        only(typed_analysis_site_ids(recipient_snapshot; site_path=[1])),
    )
    source = first(donor_site.boundary_visible)
    target = first(recipient_site.boundary_visible)
    transported = transport_subtree(
        donor_tree,
        donor_site,
        recipient_site,
        Dict(source => target),
    )
    @test transported isa AstTree
    transported isa AstTree || return
    @test transported.root == AbsNode(:y, IndexNode(1))
end

@testitem "binder transport rebases around an internal lambda" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree
    using LambdaRegression.TypeSystem: TyLike

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, AppNode(IndexNode(1), IndexNode(0)))))
    recipient_tree = AstTree(AbsNode(:z, AbsNode(:w, IndexNode(0))))
    donor_snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_typed_analysis(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = typed_analysis_site(
        donor_snapshot, only(typed_analysis_site_ids(donor_snapshot; site_path=[1])),
    )
    recipient_site = typed_analysis_site(
        recipient_snapshot,
        only(typed_analysis_site_ids(recipient_snapshot; site_path=[1, 1])),
    )
    source = first(donor_site.boundary_visible)
    target = recipient_site.boundary_visible[2]
    transported = transport_subtree(
        donor_tree,
        donor_site,
        recipient_site,
        Dict(source => target),
    )
    @test transported isa AstTree
    transported isa AstTree || return
    @test transported.root == AbsNode(:y, AppNode(IndexNode(2), IndexNode(0)))
end

@testitem "binder transport rejects FunctionDef self to ordinary binder" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree, TransportRoleMismatchErr
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Utils: Err

    donor_tree = AstTree(FunctionDefNode(IndexNode(0)))
    recipient_tree = AstTree(AbsNode(:x, IndexNode(0)))
    donor_snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_typed_analysis(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = typed_analysis_site(
        donor_snapshot, only(typed_analysis_site_ids(donor_snapshot; site_path=[1])),
    )
    recipient_site = typed_analysis_site(
        recipient_snapshot,
        only(typed_analysis_site_ids(recipient_snapshot; site_path=[1])),
    )
    result = transport_subtree(
        donor_tree,
        donor_site,
        recipient_site,
        Dict(first(donor_site.boundary_visible) => first(recipient_site.boundary_visible)),
    )
    @test result isa Err
    result isa Err || return
    @test result.value isa TransportRoleMismatchErr
end

@testitem "binder transport failure leaves donor unchanged" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Utils: Err

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    before = deepcopy(donor_tree)
    snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    donor_site = typed_analysis_site(
        snapshot, only(typed_analysis_site_ids(snapshot; site_path=[1])),
    )
    result = transport_subtree(donor_tree, donor_site, donor_site, Dict())
    @test result isa Err
    @test donor_tree == before
end

@testitem "binder transport ignores cross-tree BinderId path collisions" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_typed_analysis, typed_analysis_site_ids, typed_analysis_site,
        transport_subtree
    using LambdaRegression.TypeSystem: TyLike

    # The donor's nested binder has the same structural BinderId as the
    # recipient's middle ancestor.  A BinderId-based merge would therefore
    # confuse an internal reference with the explicitly mapped boundary
    # reference; transport must instead use De Bruijn depth arithmetic.
    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    recipient_tree = AstTree(
        AbsNode(:a, AbsNode(:b, AbsNode(:c, IndexNode(0)))),
    )
    donor_snapshot = prepare_typed_analysis(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_typed_analysis(
        recipient_tree,
        Dict{Symbol,TyLike}(),
    )
    donor_site = typed_analysis_site(
        donor_snapshot, only(typed_analysis_site_ids(donor_snapshot; site_path=[1])),
    )
    recipient_site = typed_analysis_site(
        recipient_snapshot,
        only(typed_analysis_site_ids(recipient_snapshot; site_path=[1, 1, 1])),
    )
    source = first(donor_site.boundary_visible)
    target = recipient_site.boundary_visible[2]
    transported = transport_subtree(
        donor_tree,
        donor_site,
        recipient_site,
        Dict(source => target),
    )
    @test transported isa AstTree
    transported isa AstTree || return
    @test transported.root == AbsNode(:y, IndexNode(2))
end
