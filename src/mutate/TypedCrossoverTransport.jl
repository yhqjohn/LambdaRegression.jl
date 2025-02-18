"""Binder-aware, mutation-free transport of a donor subtree."""

using TestItems

using ...ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, IndexNode, LetNode, VarNode,
    children
using ...ASTs.LexicalContexts: BinderId
using ...Procedural: FunctionDefNode
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

@inline _transport_introduced(::AppNode, ::Int) = 0
@inline _transport_introduced(::AbsNode, child_index::Int) =
    child_index == 1 ? 1 : throw(BoundsError())
@inline _transport_introduced(::LetNode, child_index::Int) =
    child_index == 2 ? 1 : 0
@inline _transport_introduced(::FunctionDefNode, child_index::Int) =
    child_index == 1 ? 1 : throw(BoundsError())

function _transport_role_ok(source::BinderId, target::BinderId)
    source.kind === :function_def && target.kind === :function_def && return true
    target.kind === :function_def && return false
    source.kind === :function_def && return false
    return true
end

function _transport_validate_mapping(
    donor_boundary::Tuple,
    recipient_boundary::Tuple,
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

function _transport_node_at_path(root::AstNode, path)
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
    donor_boundary::Tuple,
    recipient_boundary::Tuple,
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
        # `_transport_introduced` deliberately has no ScopeNode method. Index
        # arithmetic avoids comparing locally introduced BinderIds across trees.
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
    elseif node isa LetNode
        return LetNode(node.name, rewritten_children[1], rewritten_children[2])
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
    donor_boundary = Tuple(_transport_boundary(donor_site))
    recipient_boundary = Tuple(_transport_boundary(recipient_site))
    validation = _transport_validate_mapping(
        donor_boundary,
        recipient_boundary,
        mapping,
    )
    validation === nothing || return validation

    source = _transport_node_at_path(donor_tree.root, _transport_path(donor_site))
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

@testitem "binder transport maps an arbitrary ancestor" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AstTree, ConstNode, IndexNode, LetNode
    using LambdaRegression.Mutations.Populations:
        prepare_core_snapshot, transport_subtree
    using LambdaRegression.TypeSystem: IntTy, TyLike

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    recipient_tree = AstTree(LetNode(:z, ConstNode(:one), AbsNode(:q, IndexNode(0))))
    donor_snapshot = prepare_core_snapshot(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_core_snapshot(
        recipient_tree,
        Dict{Symbol,TyLike}(:one => IntTy),
    )
    donor_site = only(filter(site -> site.path == (1,), donor_snapshot.sites))
    recipient_site = only(filter(site -> site.path == (2,), recipient_snapshot.sites))
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
        prepare_core_snapshot, transport_subtree
    using LambdaRegression.TypeSystem: TyLike

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, AppNode(IndexNode(1), IndexNode(0)))))
    recipient_tree = AstTree(AbsNode(:z, AbsNode(:w, IndexNode(0))))
    donor_snapshot = prepare_core_snapshot(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_core_snapshot(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = only(filter(site -> site.path == (1,), donor_snapshot.sites))
    recipient_site = only(filter(site -> site.path == (1, 1), recipient_snapshot.sites))
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
        prepare_core_snapshot, transport_subtree, TransportRoleMismatchErr
    using LambdaRegression.Procedural: FunctionDefNode, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Utils: Err

    donor_tree = AstTree(FunctionDefNode(IndexNode(0)))
    recipient_tree = AstTree(AbsNode(:x, IndexNode(0)))
    donor_snapshot = prepare_procedural_snapshot(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_core_snapshot(recipient_tree, Dict{Symbol,TyLike}())
    donor_site = only(filter(site -> site.path == (1,), donor_snapshot.sites))
    recipient_site = only(filter(site -> site.path == (1,), recipient_snapshot.sites))
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
        prepare_core_snapshot, transport_subtree
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Utils: Err

    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    before = deepcopy(donor_tree)
    snapshot = prepare_core_snapshot(donor_tree, Dict{Symbol,TyLike}())
    donor_site = only(filter(site -> site.path == (1,), snapshot.sites))
    result = transport_subtree(donor_tree, donor_site, donor_site, Dict())
    @test result isa Err
    @test donor_tree == before
end

@testitem "binder transport ignores cross-tree BinderId path collisions" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        prepare_core_snapshot, transport_subtree
    using LambdaRegression.TypeSystem: TyLike

    # The donor's nested binder has the same structural BinderId as the
    # recipient's middle ancestor.  A BinderId-based merge would therefore
    # confuse an internal reference with the explicitly mapped boundary
    # reference; transport must instead use De Bruijn depth arithmetic.
    donor_tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    recipient_tree = AstTree(
        AbsNode(:a, AbsNode(:b, AbsNode(:c, IndexNode(0)))),
    )
    donor_snapshot = prepare_core_snapshot(donor_tree, Dict{Symbol,TyLike}())
    recipient_snapshot = prepare_core_snapshot(
        recipient_tree,
        Dict{Symbol,TyLike}(),
    )
    donor_site = only(filter(site -> site.path == (1,), donor_snapshot.sites))
    recipient_site = only(
        filter(site -> site.path == (1, 1, 1), recipient_snapshot.sites),
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

@testitem "binder transport leaves Scope IR unsupported" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Procedural: ScopeNode
    using LambdaRegression.Mutations.Populations: transport_subtree

    struct _TransportTestSite
        path::Tuple
        boundary_visible::Tuple
    end

    tree = AstTree(ScopeNode([ConstNode(:x)], ConstNode(:body)))
    site = _TransportTestSite((), ())
    @test_throws MethodError transport_subtree(tree, site, site, Dict())
end
