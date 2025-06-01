"""
    Transforms

Small syntax-directed transformations over the common AST, including typed
editable program chaining, scope-aware common-subexpression extraction and single lift. It
consumes a `ProgramAnalysis.SyntaxIndex` plus an occurrence ID, so occurrence
identity never depends on host node identity or a second lexical snapshot.
Structural checks are performed here; typed callers retain type and effect
acceptance.
"""
module Transforms

using TestItems

using ..Utils: Err
using ..ASTs.NodeASTs:
    AstTree, AstNode, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode,
    VarNode, children, copy_occurrences, setchild!, setroot!, shift_free_variables!,
    SymbolGenerator
using ..ProgramAnalysis: SyntaxIndex, analyze_syntax, path, captured_binder_ids
using ..ProgramAnalysis: _functiondef_self_observation

export NotEnoughMatchesErr, CrossScopeMatchErr, TargetOccurrenceNotFoundErr
export UnsupportedCSENodeErr, repeated_nodes, occurrence_id, eliminate_node!
export fixed_point_witness, lower_functiondefs

include("program_composition.jl")

struct NotEnoughMatchesErr
    min_count::Int
    got::Int
end

struct CrossScopeMatchErr end
struct TargetOccurrenceNotFoundErr end

struct UnsupportedCSENodeErr
    node_type::DataType
    occurrence::Int
end

function occurrence_id(index::SyntaxIndex, requested_path::AbstractVector{<:Integer})
    id = 1
    for edge in requested_path
        children_ids = index.children[id]
        1 <= edge <= length(children_ids) || return nothing
        id = children_ids[Int(edge)]
    end
    return id
end

function _node_at_path(tree::AstTree, requested_path)
    node = tree.root
    for edge in requested_path
        1 <= edge <= length(children(node)) || return nothing
        node = children(node)[edge]
    end
    return node
end

_env_depth(index::SyntaxIndex, id::Int) = begin
    head = index.env_head[id]
    head == 0 ? 0 : index.bindings[head].depth
end

_is_ancestor(index::SyntaxIndex, ancestor::Int, descendant::Int) =
    ancestor <= descendant < index.subtree_end[ancestor]

function _lca_id(index::SyntaxIndex, ids::Vector{Int})
    isempty(ids) && throw(ArgumentError("cannot find an LCA of an empty occurrence set"))
    candidate = first(ids)
    while !all(id -> _is_ancestor(index, candidate, id), ids)
        candidate = index.parent[candidate]
        candidate == 0 && throw(ArgumentError("occurrence set has no root ancestor"))
    end
    return candidate
end

function _free_index_key(index::SyntaxIndex, id::Int, value::Int)
    return value - _env_depth(index, id)
end

"""Conservative bottom-up bucket summary; exact equality is checked later."""
function _coarse_summary(index::SyntaxIndex)
    hashes = Vector{UInt}(undef, length(index))
    sizes = Vector{Int}(undef, length(index))
    bound_named = falses(length(index))
    for id in length(index):-1:1
        node = index.nodes[id]
        child_ids = index.children[id]
        tag = if node isa ConstNode
            (:const, node.name)
        elseif node isa VarNode
            (:var, node.name)
        elseif node isa IndexNode
            :index
        elseif node isa AbsNode
            :abs
        elseif node isa FunctionDefNode
            :function_def
        elseif node isa AppNode
            (:app, index.nodes[child_ids[1]] isa AbsNode)
        else
            throw(UnsupportedCSENodeErr(typeof(node), id))
        end
        child_hash = map(child_id -> hashes[child_id], child_ids)
        hashes[id] = hash((tag, child_hash))
        sizes[id] = 1 + sum(child_id -> sizes[child_id], child_ids; init=0)
        bound_named[id] = (node isa VarNode && index.reference_binder[id] != 0) ||
            any(child_id -> bound_named[child_id], child_ids)
    end
    buckets = Dict{Tuple{UInt,Int},Vector{Int}}()
    for id in eachindex(hashes)
        push!(get!(Vector{Int}, buckets, (hashes[id], sizes[id])), id)
    end
    return hashes, sizes, bound_named, buckets
end

function _exact_equal(
    index::SyntaxIndex,
    left::Int,
    right::Int,
    left_root::Int=left,
    right_root::Int=right,
)
    left_node = index.nodes[left]
    right_node = index.nodes[right]
    typeof(left_node) === typeof(right_node) || return false
    if left_node isa ConstNode
        return left_node.name === right_node.name
    elseif left_node isa VarNode
        left_binder = index.reference_binder[left]
        right_binder = index.reference_binder[right]
        left_binder == 0 && right_binder == 0 && return left_node.name === right_node.name
        (left_binder == 0 || right_binder == 0) && return false
        # Bound named variables are explicitly outside this transport relation.
        # Candidate roots containing them are filtered before grouping; a
        # coarse-bucket collision must not poison an unrelated valid class.
        return false
    elseif left_node isa IndexNode
        left_binder = index.reference_binder[left]
        right_binder = index.reference_binder[right]
        if left_binder == 0 || right_binder == 0
            left_binder == 0 && right_binder == 0 || return false
            return _free_index_key(index, left, left_node.index) ==
                   _free_index_key(index, right, right_node.index)
        end
        left_owner = index.bindings[left_binder].owner
        right_owner = index.bindings[right_binder].owner
        left_internal = left_root <= left_owner < index.subtree_end[left_root]
        right_internal = right_root <= right_owner < index.subtree_end[right_root]
        left_internal == right_internal || return false
        left_internal && return left_node.index == right_node.index
        return left_binder == right_binder
    elseif left_node isa AbsNode || left_node isa FunctionDefNode
        left_child = index.children[left][1]
        right_child = index.children[right][1]
        return _exact_equal(
            index,
            left_child,
            right_child,
            left_root,
            right_root,
        )
    elseif left_node isa AppNode
        left_children = index.children[left]
        right_children = index.children[right]
        (left_node.func isa AbsNode) == (right_node.func isa AbsNode) || return false
        return _exact_equal(index, left_children[1], right_children[1], left_root, right_root) &&
               _exact_equal(index, left_children[2], right_children[2], left_root, right_root)
    end
    throw(UnsupportedCSENodeErr(typeof(left_node), left))
end

function _equivalent_ids(index::SyntaxIndex, anchor::Int, candidates::Vector{Int})
    return Int[id for id in candidates if _exact_equal(index, anchor, id)]
end

function _exact_groups(index::SyntaxIndex, candidates::Vector{Int})
    representatives = Int[]
    groups = Vector{Vector{Int}}()
    for id in candidates
        group_index = findfirst(
            representative -> _exact_equal(index, representative, id),
            representatives,
        )
        if group_index === nothing
            push!(representatives, id)
            push!(groups, Int[id])
        else
            push!(groups[group_index], id)
        end
    end
    return groups
end

function _eligible_bucket_ids(ids::Vector{Int}, bound_named::BitVector)
    return Int[id for id in ids if !bound_named[id]]
end

"""Return occurrence IDs whose exact scope-aware form has `min_count` matches."""
function repeated_nodes(index::SyntaxIndex; min_count::Int=2)
    min_count >= 1 || return Err(ArgumentError("min_count must be >= 1"))
    eligible = Set{Int}()
    _, _, bound_named, buckets = _coarse_summary(index)
    for ids in values(buckets)
        candidates = _eligible_bucket_ids(ids, bound_named)
        length(candidates) >= min_count || continue
        for group in _exact_groups(index, candidates)
            length(group) >= min_count || continue
            union!(eligible, group)
        end
    end
    return eligible
end

function _single_lift_host(index::SyntaxIndex, candidate::Int)
    parent = index.parent[candidate]
    parent == 0 && return candidate
    for binder in captured_binder_ids(index, candidate)
        owner = index.bindings[binder].owner
        body = index.children[owner][1]
        _is_ancestor(index, body, parent) || return candidate
    end
    return parent
end

function _select_scope_group(index::SyntaxIndex, candidates::Vector{Int}, anchor::Int, min_count::Int)
    min_count == 1 && return Int[anchor], _single_lift_host(index, anchor)
    length(candidates) >= min_count || return Err(NotEnoughMatchesErr(min_count, length(candidates)))
    return candidates, _lca_id(index, candidates)
end

function _contains_bound_named(index::SyntaxIndex, id::Int)
    for occurrence in id:(index.subtree_end[id] - 1)
        index.nodes[occurrence] isa VarNode || continue
        index.reference_binder[occurrence] == 0 || return true
    end
    return false
end

function _replace_at_path!(tree::AstTree, requested_path, replacement::AstNode)
    isempty(requested_path) && return setroot!(tree, replacement)
    parent = _node_at_path(tree, requested_path[1:end-1])
    parent === nothing && throw(BoundsError(requested_path))
    setchild!(parent, requested_path[end], replacement)
    return tree
end

function _lexical_depth_between(index::SyntaxIndex, host_id::Int, occurrence::Int)
    return _env_depth(index, occurrence) - _env_depth(index, host_id)
end

function _placeholder_body(index::SyntaxIndex, tree::AstTree, host_id::Int, matches::Vector{Int})
    host_path = path(index, host_id)
    host = _node_at_path(tree, host_path)
    host === nothing && throw(BoundsError(host_path))
    body_tree = AstTree(copy_occurrences(host))
    shift_free_variables!(body_tree, 1, 0)
    for occurrence in sort(matches; by=id -> index.depth[id], rev=true)
        occurrence_path = path(index, occurrence)
        relative = occurrence_path[(length(host_path) + 1):end]
        depth = _lexical_depth_between(index, host_id, occurrence)
        _replace_at_path!(body_tree, relative, IndexNode(depth))
    end
    return body_tree.root
end

function _rebased_representative(index::SyntaxIndex, tree::AstTree, host_id::Int, occurrence::Int)
    occurrence_path = path(index, occurrence)
    representative = _node_at_path(tree, occurrence_path)
    representative === nothing && throw(BoundsError(occurrence_path))
    local_depth = _lexical_depth_between(index, host_id, occurrence)
    copied = AstTree(copy_occurrences(representative))
    local_depth == 0 || shift_free_variables!(copied, -local_depth, local_depth)
    return copied.root
end

function _roles_preserved(index::SyntaxIndex, host_id::Int, matches::Vector{Int})
    host_parent = index.parent[host_id]
    if host_parent != 0 && index.parent_edge[host_id] == 1 &&
       index.nodes[host_parent] isa AppNode && index.nodes[host_id] isa AbsNode
        return false
    end
    for occurrence in matches
        parent = index.parent[occurrence]
        if parent != 0 && index.parent_edge[occurrence] == 1 &&
           index.nodes[parent] isa AppNode && index.nodes[occurrence] isa AbsNode
            return false
        end
    end
    return true
end

function _fresh_name(factory, tree::AstTree)
    candidate = factory()
    used = Set{Symbol}()
    for node in tree
        node isa VarNode && push!(used, node.name)
        node isa AbsNode && push!(used, node.name)
    end
    candidate in used || return candidate
    suffix = 1
    while true
        fresh = Symbol(string(candidate), "_", suffix)
        fresh in used || return fresh
        suffix += 1
    end
end

"""
    eliminate_node!(attempt_tree, source_index, occurrence_id; min_count=1,
        factory=SymbolGenerator())

Perform one scope-aware single lift or repeated extraction in the supplied
attempt tree. `source_index` describes the unedited tree at the same occurrence
layout; callers own transactional commit and type verification.
"""
function eliminate_node!(
    attempt_tree::AstTree,
    source_index::SyntaxIndex,
    occurrence::Int;
    min_count::Int=1,
    factory=SymbolGenerator(),
)
    min_count >= 1 || return Err(ArgumentError("min_count must be >= 1"))
    1 <= occurrence <= length(source_index) || return Err(TargetOccurrenceNotFoundErr())
    _contains_bound_named(source_index, occurrence) &&
        return Err(UnsupportedCSENodeErr(VarNode, occurrence))
    candidates = if min_count == 1
        Int[occurrence]
    else
        hashes, sizes, bound_named, buckets = _coarse_summary(source_index)
        bound_named[occurrence] && return Err(UnsupportedCSENodeErr(VarNode, occurrence))
        bucket_key = (hashes[occurrence], sizes[occurrence])
        bucket = _eligible_bucket_ids(get(buckets, bucket_key, Int[]), bound_named)
        group = _equivalent_ids(source_index, occurrence, bucket)
        length(group) >= min_count || return Err(NotEnoughMatchesErr(min_count, length(group)))
        group
    end
    group, host_id = _select_scope_group(source_index, candidates, occurrence, min_count)
    group isa Err && return group
    _roles_preserved(source_index, host_id, group) || return Err(CrossScopeMatchErr())
    body = _placeholder_body(source_index, attempt_tree, host_id, group)
    representative = _rebased_representative(source_index, attempt_tree, host_id, occurrence)
    new_subtree = AppNode(AbsNode(_fresh_name(factory, attempt_tree), body), representative)
    _replace_at_path!(attempt_tree, path(source_index, host_id), new_subtree)
    return attempt_tree
end

"""
    fixed_point_witness() -> AstNode

Return a fresh finite core AST for the untyped fixed-point combinator
`λf.(λx.f (x x)) (λx.f (x x))`.
"""
function fixed_point_witness()
    function half()
        xx = AppNode(IndexNode(0), IndexNode(0))
        return AbsNode(:x, AppNode(IndexNode(1), xx))
    end
    return AbsNode(:f, AppNode(half(), half()))
end

function _fixed_point_node(source)
    value = if source isa AstNode || source isa AstTree
        source
    elseif applicable(source)
        source()
    else
        source
    end
    value isa AstTree && return deepcopy(value.root)
    value isa AstNode && return deepcopy(value)
    throw(ArgumentError("fixed_point must be an AstNode, AstTree, or zero-argument factory"))
end

mutable struct _FunctionDefLoweringContext
    fixed_point
    used_names::Set{Symbol}
    next_name::Int
end

function _functiondef_lowering_context(index::SyntaxIndex, fixed_point)
    names = Set{Symbol}()
    for node in index.nodes
        node isa Union{VarNode,AbsNode} && push!(names, node.name)
    end
    return _FunctionDefLoweringContext(fixed_point, names, 0)
end

function _fresh_functiondef_binder!(context::_FunctionDefLoweringContext)
    while true
        context.next_name += 1
        name = Symbol("##procedural", context.next_name)
        name in context.used_names && continue
        push!(context.used_names, name)
        return name
    end
end

function _drop_unused_functiondef_slot(node::AstNode, depth::Int=0)
    node isa Union{ConstNode,VarNode} && return deepcopy(node)
    if node isa IndexNode
        node.index < depth && return deepcopy(node)
        node.index == depth && throw(ArgumentError(
            "cannot remove a referenced FunctionDef binder",
        ))
        return IndexNode(node.index - 1)
    elseif node isa AbsNode
        return AbsNode(node.name, _drop_unused_functiondef_slot(node.body, depth + 1))
    elseif node isa AppNode
        return AppNode(
            _drop_unused_functiondef_slot(node.func, depth),
            _drop_unused_functiondef_slot(node.arg, depth),
        )
    end
    throw(ArgumentError("FunctionDef lowering produced unexpected node $(typeof(node))"))
end

function _lower_functiondef_occurrence(
    index::SyntaxIndex,
    id::Int,
    context::_FunctionDefLoweringContext,
)
    node = index.nodes[id]
    children_ids = index.children[id]
    if node isa ConstNode || node isa VarNode || node isa IndexNode
        return deepcopy(node)
    elseif node isa AbsNode
        return AbsNode(node.name, _lower_functiondef_occurrence(index, only(children_ids), context))
    elseif node isa AppNode
        return AppNode(
            _lower_functiondef_occurrence(index, children_ids[1], context),
            _lower_functiondef_occurrence(index, children_ids[2], context),
        )
    elseif node isa FunctionDefNode
        body_id = only(children_ids)
        body = _lower_functiondef_occurrence(index, body_id, context)
        self_referenced = _functiondef_self_observation(index, body_id, 0).self_references > 0
        if self_referenced
            self_name = _fresh_functiondef_binder!(context)
            return AppNode(_fixed_point_node(context.fixed_point), AbsNode(self_name, body))
        end
        return _drop_unused_functiondef_slot(body)
    end
    throw(ArgumentError("unsupported common-language node $(typeof(node))"))
end

"""
    lower_functiondefs(term; fixed_point=fixed_point_witness)

Return a fresh common AST with every `FunctionDefNode` lowered to the finite
fixed-point combinator only when its anonymous self binder is used. An unused
self slot is removed and all outer De Bruijn references are rebased. Legacy
syntax is refused by the common analysis boundary.
"""
function lower_functiondefs(term::Union{AstNode,AstTree}; fixed_point=fixed_point_witness)
    tree = term isa AstTree ? term : AstTree(term)
    index = analyze_syntax(tree)
    context = _functiondef_lowering_context(index, fixed_point)
    lowered = _lower_functiondef_occurrence(index, 1, context)
    return term isa AstTree ? AstTree(lowered) : lowered
end

@testitem "Transforms lowers FunctionDef self slots without changing captures" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, FunctionDefNode, IndexNode, VarNode
    using LambdaRegression.Transforms: lower_functiondefs

    captured = FunctionDefNode(IndexNode(1))
    @test lower_functiondefs(captured) == IndexNode(0)
    @test captured == FunctionDefNode(IndexNode(1))

    recursive = FunctionDefNode(AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))))
    lowered_recursive = lower_functiondefs(recursive)
    @test lowered_recursive isa AppNode
    @test lowered_recursive.func isa AbsNode
    @test lowered_recursive.arg isa AbsNode
    @test lowered_recursive.arg.body.body == AppNode(IndexNode(1), IndexNode(0))

    nested_outer_capture = FunctionDefNode(FunctionDefNode(IndexNode(1)))
    lowered_nested = lower_functiondefs(nested_outer_capture)
    @test lowered_nested isa AppNode
    @test lowered_nested.arg isa AbsNode
    @test lowered_nested.arg.body == IndexNode(0)

    calls = Ref(0)
    custom = lower_functiondefs(recursive; fixed_point=() -> begin
        calls[] += 1
        ConstNode(:fixed_point)
    end)
    @test calls[] == 1
    @test custom.func == ConstNode(:fixed_point)
    lower_functiondefs(captured; fixed_point=() -> error("unused fixed point"))

    collision = FunctionDefNode(AppNode(IndexNode(0), VarNode(Symbol("##procedural1"))))
    lowered_collision = lower_functiondefs(collision)
    @test lowered_collision.arg.name == Symbol("##procedural2")
end

@testitem "Transforms CSE distinguishes FunctionDef bodies and self captures" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AppNode, AbsNode, AstTree, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.ProgramAnalysis: analyze_syntax, captured_binder_ids
    using LambdaRegression.Transforms: occurrence_id, repeated_nodes
    using LambdaRegression.Utils: Err

    different = AstTree(AppNode(
        FunctionDefNode(AbsNode(:x, IndexNode(0))),
        FunctionDefNode(AbsNode(:x, ConstNode(:zero))),
    ))
    index = analyze_syntax(different)
    eligible = repeated_nodes(index; min_count=2)
    @test !(eligible isa Err)
    eligible isa Err && return
    left = occurrence_id(index, [1])
    right = occurrence_id(index, [2])
    @test !(left in eligible)
    @test !(right in eligible)

    recursive = analyze_syntax(AstTree(FunctionDefNode(
        AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
    )))
    definition_id = occurrence_id(recursive, Int[])
    body_id = occurrence_id(recursive, [1])
    @test isempty(captured_binder_ids(recursive, definition_id))
    @test length(captured_binder_ids(recursive, body_id)) == 1
end

@testitem "Transforms CSE rejects named transport and preserves parent roles" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, VarNode
    using LambdaRegression.ProgramAnalysis: analyze_syntax
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id, repeated_nodes
    using LambdaRegression.Utils: Err

    named = analyze_syntax(AstTree(AppNode(VarNode(:f), ConstNode(:one))))
    @test repeated_nodes(named) isa Set

    bound_named = AstTree(AbsNode(:x, AppNode(VarNode(:x), ConstNode(:one))))
    bound_index = analyze_syntax(bound_named)
    bound_site = occurrence_id(bound_index, [1])
    @test eliminate_node!(bound_named, bound_index, bound_site; min_count=1) isa Err

    let_like = AstTree(AppNode(AbsNode(:x, ConstNode(:one)), ConstNode(:arg)))
    index = analyze_syntax(let_like)
    wrapper = occurrence_id(index, [1])
    original = deepcopy(let_like)
    result = eliminate_node!(let_like, index, wrapper; min_count=1)
    @test result isa Err
    @test let_like == original
end

@testitem "Transforms CSE keeps same-typed FunctionDef bodies distinct and whole self internal" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.ProgramAnalysis:
        TypedAnalysis, analyze_program, analyze_syntax, captured_binder_ids
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id, repeated_nodes
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyProd
    using LambdaRegression.Utils: Err

    function_type = TyFun(IntTy, IntTy)
    context = Dict{Symbol,TyLike}(
        :PAIR => TyFun(function_type, TyFun(function_type, TyProd(function_type, function_type))),
        :zero => IntTy,
    )
    distinct = AstTree(AppNode(
        AppNode(ConstNode(:PAIR), FunctionDefNode(AbsNode(:x, IndexNode(0)))),
        FunctionDefNode(AbsNode(:x, ConstNode(:zero))),
    ))
    typed = analyze_program(distinct, context; target_ty=TyProd(function_type, function_type))
    @test typed isa TypedAnalysis
    typed isa TypedAnalysis || return
    @test typed.root_type isa TyProd

    distinct_index = analyze_syntax(distinct)
    left_definition = occurrence_id(distinct_index, [1, 2])
    right_definition = occurrence_id(distinct_index, [2])
    eligible = repeated_nodes(distinct_index; min_count=2)
    @test !(left_definition in eligible)
    @test !(right_definition in eligible)
    unchanged = deepcopy(distinct)
    rejected = eliminate_node!(
        distinct, distinct_index, left_definition; min_count=2,
    )
    @test rejected isa Err
    @test distinct.root == unchanged.root

    recursive_definition() = FunctionDefNode(AbsNode(
        :x, AppNode(IndexNode(1), IndexNode(0)),
    ))
    recursive_pair = AstTree(AppNode(
        AppNode(ConstNode(:PAIR), recursive_definition()),
        recursive_definition(),
    ))
    recursive_index = analyze_syntax(recursive_pair)
    first_id = occurrence_id(recursive_index, [1, 2])
    second_id = occurrence_id(recursive_index, [2])
    first_body = occurrence_id(recursive_index, [1, 2, 1])
    second_body = occurrence_id(recursive_index, [2, 1])
    @test isempty(captured_binder_ids(recursive_index, first_id))
    @test isempty(captured_binder_ids(recursive_index, second_id))
    @test captured_binder_ids(recursive_index, first_body) ==
          [recursive_index.owner_binder[first_id]]
    @test captured_binder_ids(recursive_index, second_body) ==
          [recursive_index.owner_binder[second_id]]
    recursive_eligible = repeated_nodes(recursive_index; min_count=2)
    @test first_id in recursive_eligible
    @test second_id in recursive_eligible
    @test !(first_body in recursive_eligible)
    @test !(second_body in recursive_eligible)

    extracted = eliminate_node!(recursive_pair, recursive_index, first_id; min_count=2)
    @test !(extracted isa Err)
    extracted isa Err && return
    rewritten_index = analyze_syntax(recursive_pair)
    remaining_definition = occurrence_id(rewritten_index, [2])
    self_use = occurrence_id(rewritten_index, [2, 1, 1, 1])
    @test rewritten_index.nodes[remaining_definition] isa FunctionDefNode
    @test rewritten_index.reference_binder[self_use] ==
          rewritten_index.owner_binder[remaining_definition]
end

@testitem "Transforms CSE aligns captures across unused binders and preserves siblings" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.ProgramAnalysis:
        analyze_syntax, captured_binder_ids
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id, repeated_nodes
    using LambdaRegression.Utils: Err

    candidate(index) = AppNode(IndexNode(index), ConstNode(:one))
    tree = AstTree(AbsNode(:outer, AppNode(
        AppNode(
            AppNode(ConstNode(:KEEP_LEFT), AbsNode(:unused, candidate(1))),
            ConstNode(:LEFT_VALUE),
        ),
        AppNode(
            FunctionDefNode(AbsNode(:unused, candidate(2))),
            ConstNode(:RIGHT_VALUE),
        ),
    )))
    index = analyze_syntax(tree)
    first = occurrence_id(index, [1, 1, 1, 2, 1])
    second = occurrence_id(index, [1, 2, 1, 1, 1])
    outer = index.owner_binder[1]
    @test captured_binder_ids(index, first) == [outer]
    @test captured_binder_ids(index, second) == [outer]
    eligible = repeated_nodes(index; min_count=2)
    @test first in eligible
    @test second in eligible

    factory_calls = Ref(0)
    extracted = eliminate_node!(
        tree, index, first; min_count=2,
        factory=() -> (factory_calls[] += 1; :shared),
    )
    @test !(extracted isa Err)
    extracted isa Err && return
    @test factory_calls[] == 1
    @test tree.root isa AbsNode
    host = tree.root.body
    @test host isa AppNode && host.func isa AbsNode
    host isa AppNode && host.func isa AbsNode || return
    @test host.arg == AppNode(IndexNode(0), ConstNode(:one))

    rewritten = analyze_syntax(tree)
    let_abs = occurrence_id(rewritten, [1, 1])
    let_binder = rewritten.owner_binder[let_abs]
    placeholders = findall(
        id -> rewritten.nodes[id] isa IndexNode &&
             rewritten.reference_binder[id] == let_binder,
        eachindex(rewritten.nodes),
    )
    @test length(placeholders) == 2
    @test Set(rewritten.nodes[id].index for id in placeholders) == Set([1, 2])
    rebased_capture = occurrence_id(rewritten, [1, 2, 1])
    @test rewritten.nodes[rebased_capture] == IndexNode(0)
    @test rewritten.reference_binder[rebased_capture] == rewritten.owner_binder[1]
    for name in (:KEEP_LEFT, :LEFT_VALUE, :RIGHT_VALUE)
        @test count(node -> node isa ConstNode && node.name === name, rewritten.nodes) == 1
    end
end

@testitem "Transforms CSE protects a free named variable from a colliding factory" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AppNode, AstTree, ConstNode, IndexNode, VarNode
    using LambdaRegression.ProgramAnalysis: analyze_syntax
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id
    using LambdaRegression.Utils: Err

    expression() = AppNode(VarNode(:fresh), ConstNode(:one))
    tree = AstTree(AppNode(
        AppNode(ConstNode(:PAIR), expression()),
        expression(),
    ))
    index = analyze_syntax(tree)
    anchor = occurrence_id(index, [1, 2])
    factory_calls = Ref(0)
    result = eliminate_node!(
        tree, index, anchor; min_count=2,
        factory=() -> (factory_calls[] += 1; :fresh),
    )
    @test !(result isa Err)
    result isa Err && return
    @test factory_calls[] == 1

    rewritten = analyze_syntax(tree)
    binder = rewritten.owner_binder[occurrence_id(rewritten, [1])]
    @test rewritten.bindings[binder].name === :fresh_1
    free_variables = findall(
        id -> rewritten.nodes[id] isa VarNode && rewritten.nodes[id].name === :fresh,
        eachindex(rewritten.nodes),
    )
    @test length(free_variables) == 1
    @test rewritten.reference_binder[only(free_variables)] == 0
end

@testitem "Transforms single lift keeps a captured self use inside its definition" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, FunctionDefNode, IndexNode
    using LambdaRegression.ProgramAnalysis:
        analyze_syntax, captured_binder_ids
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id
    using LambdaRegression.Utils: Err

    tree = AstTree(FunctionDefNode(AbsNode(
        :x, AppNode(IndexNode(1), IndexNode(0)),
    )))
    index = analyze_syntax(tree)
    selected = occurrence_id(index, [1, 1, 1])
    self_binder = index.owner_binder[1]
    @test captured_binder_ids(index, selected) == [self_binder]

    result = eliminate_node!(tree, index, selected; min_count=1)
    @test !(result isa Err)
    result isa Err && return
    rewritten = analyze_syntax(tree)
    host = occurrence_id(rewritten, [1, 1])
    self_argument = occurrence_id(rewritten, [1, 1, 2])
    self_use = occurrence_id(rewritten, [1, 1, 1, 1, 1])
    let_abs = rewritten.children[host][1]
    @test rewritten.nodes[host] isa AppNode
    @test rewritten.nodes[let_abs] isa AbsNode
    @test rewritten.reference_binder[self_argument] ==
          rewritten.owner_binder[occurrence_id(rewritten, Int[])]
    @test rewritten.reference_binder[self_use] == rewritten.owner_binder[let_abs]
end

@testitem "Transforms CSE ignores unrelated bound named subtrees" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, VarNode
    using LambdaRegression.ProgramAnalysis: analyze_syntax
    using LambdaRegression.Transforms: eliminate_node!, occurrence_id, repeated_nodes
    using LambdaRegression.Utils: Err

    repeated() = AppNode(ConstNode(:f), ConstNode(:one))
    tree = AstTree(AbsNode(:x, AppNode(
        AppNode(ConstNode(:PAIR), repeated()),
        AppNode(AppNode(VarNode(:x), VarNode(:x)), repeated()),
    )))
    index = analyze_syntax(tree)
    anchor = occurrence_id(index, [1, 1, 2])
    other = occurrence_id(index, [1, 2, 2])
    @test anchor in repeated_nodes(index; min_count=2)
    @test other in repeated_nodes(index; min_count=2)
    result = eliminate_node!(tree, index, anchor; min_count=2)
    @test !(result isa Err)
    result isa Err && return
    rewritten = analyze_syntax(tree)
    bound_names = findall(
        id -> rewritten.nodes[id] isa VarNode && rewritten.nodes[id].name === :x,
        eachindex(rewritten.nodes),
    )
    @test length(bound_names) == 2
    @test all(
        rewritten.reference_binder[id] == rewritten.owner_binder[1]
        for id in bound_names
    )
end

end # module Transforms
