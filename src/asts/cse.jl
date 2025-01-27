"""
    ASTs.CSE

Scope-aware common-subexpression extraction and single-lift support for locally
nameless lambda ASTs.

Use `eliminate_node!(tree, target; min_count=1)` when a caller already has an
AST node/site. `min_count=1` performs a single lift to a legal ancestor host;
`min_count=2` or larger performs repeated CSE on an anchor-containing legal
antichain of matching applications. The compatibility
`eliminate_expression!(tree, canonical_form; min_count=...)` entry first finds
a matching node and then delegates to the same node-aware kernel.

The rewrite emits ordinary let-like application `(λx. body) value` using direct
de Bruijn placeholders, so it does not rely on temporary named variables or
name-based cleanup. Callers that need typed guarantees should run the ordinary
type inference verifier after the rewrite; the mutation framework does this
before accepting a changed tree.
"""
module CSE
using ...Utils: Err
using ..NodeASTs: ConstNode, AstTree, AstNode, AbsNode, IndexNode, AppNode, VarNode, SymbolGenerator, eliminate_bound_variables!, shift_free_variables!, AbstractionDepthState
using ...Trees: traverse, getstate, lca, root, setchild!, setroot!, setindex!, children, PointerState, Break
using MLStyle, DataStructures, TestItems


@data CanonicalNode begin
    CanonicalAbs(body::CanonicalNode)
    CanonicalIndex(index::Int)
    CanonicalVar(name::Symbol)
    CanonicalConst(name::Symbol)
    CanonicalApp(func::CanonicalNode, arg::CanonicalNode)
end

"""
    canonicalize(tree::AstTree, node::AstNode) -> Union{CanonicalNode,Err}

Compute a structural canonical form for `node`.

- `AbsNode` binder names are ignored (alpha-equivalence).
- `IndexNode` is represented by its De Bruijn index directly.
"""
function canonicalize(tree::AstTree, node::AstNode)
    if node isa IndexNode
        return CanonicalIndex((node::IndexNode).index)
    elseif node isa ConstNode
        return CanonicalConst((node::ConstNode).name)
    elseif node isa VarNode
        return CanonicalVar((node::VarNode).name)
    elseif node isa AbsNode
        body = canonicalize(tree, (node::AbsNode).body)
        body isa Err && return body
        return CanonicalAbs(body)
    elseif node isa AppNode
        n = node::AppNode
        f = canonicalize(tree, n.func)
        f isa Err && return f
        a = canonicalize(tree, n.arg)
        a isa Err && return a
        return CanonicalApp(f, a)
    end
    return Err(ArgumentError("unknown node type: $(typeof(node))"))
end

function canonical_set(tree::AstTree)
    s = Accumulator{CanonicalNode, Int}()
    for node in tree
        c = canonicalize(tree, node)
        c isa Err && continue
        inc!(s, c)
    end
    return s
end

function rebuild_node(form::CanonicalNode, factory=SymbolGenerator())
    @match form begin
        CanonicalAbs(body) => AbsNode(factory(), rebuild_node(body, factory))
        CanonicalIndex(index) => IndexNode(index)
        CanonicalVar(name) => VarNode(name)
        CanonicalConst(name) => ConstNode(name)
        CanonicalApp(func, arg) => AppNode(rebuild_node(func, factory), rebuild_node(arg, factory))
    end
end
rebuild_tree(form::CanonicalNode, factory=SymbolGenerator()) = AstTree(rebuild_node(form, factory))

"""
    eliminate_expression!(tree::AstTree, form::CanonicalNode; min_count=1, factory=SymbolGenerator()) -> Union{AstTree,Err}

Extract `form` from `tree` by introducing a beta-redex around the LCA of all matches.
and replacing the lowest common ancestor M of the qualified nodes with a new application node (λx. M\\[N=x\\]) N.
`min_count` controls the minimum number of matches required (`min_count=1` enables a single-occurrence extraction).
"""
struct NotEnoughMatchesErr
    min_count::Int
    got::Int
end

struct CrossScopeMatchErr end
struct TargetNodeNotFoundErr end

struct BinderRef
    node::AstNode
    path::Vector{Int}
end

struct ScopeMatch
    parent::Union{Nothing,AstNode}
    child_index::Int
    depth::Int
    path::Vector{Int}
    node::AstNode
    key
    limit_path::Vector{Int}
end

function _path_to_node(tree::AstTree, target::AstNode)
    found = Ref{Union{Nothing,Vector{Int}}}(nothing)

    function visit(node::AstNode, path::Vector{Int})
        node === target && (found[] = copy(path); return true)
        for (i, child) in enumerate(children(node))
            visit(child, [path; i]) && return true
        end
        return false
    end

    visit(tree.root, Int[])
    return found[]
end

function _node_at_path(tree::AstTree, path::Vector{Int})
    node = tree.root
    for idx in path
        node = children(node)[idx]
    end
    return node
end

function _parent_at_path(tree::AstTree, path::Vector{Int})
    isempty(path) && return nothing
    return _node_at_path(tree, path[1:end-1])
end

function _common_path(paths::Vector{Vector{Int}})
    isempty(paths) && return Int[]
    prefix = copy(first(paths))
    for path in paths[2:end]
        limit = min(length(prefix), length(path))
        keep = 0
        for i in 1:limit
            prefix[i] == path[i] || break
            keep = i
        end
        resize!(prefix, keep)
    end
    return prefix
end

function _depth_at_path(tree::AstTree, path::Vector{Int})
    depth = 0
    node = tree.root
    for idx in path
        node isa AbsNode && (depth += 1)
        node = children(node)[idx]
    end
    return depth
end

_path_prefix(a::Vector{Int}, b::Vector{Int}) =
    length(a) <= length(b) && all(i -> a[i] == b[i], eachindex(a))

function _ancestor_paths(path::Vector{Int})
    return [path[1:i] for i in 0:length(path)]
end

function _relative_path(base::Vector{Int}, path::Vector{Int})
    _path_prefix(base, path) || throw(ArgumentError("path is not inside base"))
    return path[(length(base) + 1):end]
end

function _body_path_of_binder(path::Vector{Int})
    return [path; 1]
end

function _deepest_path(paths::Vector{Vector{Int}})
    best = first(paths)
    for path in paths[2:end]
        length(path) > length(best) && (best = path)
    end
    return best
end

function _scope_key(node::AstNode, outer_binders::Vector{BinderRef})
    captures = Vector{Int}[]
    key = _scope_key_walk(node, outer_binders, BinderRef[], captures)
    limit_path = isempty(captures) ? Int[] : _body_path_of_binder(_deepest_path(captures))
    return key, limit_path
end

function _scope_key_walk(
    node::AstNode,
    outer_binders::Vector{BinderRef},
    local_binders::Vector{BinderRef},
    captures::Vector{Vector{Int}},
)
    if node isa IndexNode
        idx = node.index
        stack_len = length(outer_binders) + length(local_binders)
        if 0 <= idx < stack_len
            pos = stack_len - idx
            if pos <= length(outer_binders)
                binder = outer_binders[pos]
                push!(captures, binder.path)
                return (:free_binder, Tuple(binder.path))
            end
            local_pos = pos - length(outer_binders)
            return (:local_index, length(local_binders) - local_pos)
        end
        return (:context_index, idx - length(local_binders))
    elseif node isa ConstNode
        return (:const, node.name)
    elseif node isa VarNode
        return (:var, node.name)
    elseif node isa AbsNode
        return (:abs, _scope_key_walk(node.body, outer_binders, [local_binders; BinderRef(node, Int[])], captures))
    elseif node isa AppNode
        return (:app,
            _scope_key_walk(node.func, outer_binders, local_binders, captures),
            _scope_key_walk(node.arg, outer_binders, local_binders, captures),
        )
    end
    return (:unknown, typeof(node))
end

function _scope_matches(tree::AstTree)
    matches = ScopeMatch[]

    function visit(node::AstNode, path::Vector{Int}, parent::Union{Nothing,AstNode}, child_index::Int, binders::Vector{BinderRef})
        key, limit_path = _scope_key(node, binders)
        push!(matches, ScopeMatch(parent, child_index, length(binders), path, node, key, limit_path))
        if node isa AbsNode
            visit(node.body, [path; 1], node, 1, [binders; BinderRef(node, copy(path))])
        elseif node isa AppNode
            visit(node.func, [path; 1], node, 1, binders)
            visit(node.arg, [path; 2], node, 2, binders)
        end
        return nothing
    end

    visit(tree.root, Int[], nothing, 1, BinderRef[])
    return matches
end

function repeated_nodes(tree::AstTree; min_count::Int=2)
    min_count >= 1 || return Err(ArgumentError("min_count must be >= 1"))
    matches = _scope_matches(tree)
    counts = Dict{Any,Int}()
    for m in matches
        m.node isa AppNode || continue
        counts[m.key] = get(counts, m.key, 0) + 1
    end
    eligible = IdDict{AstNode,Bool}()
    for m in matches
        m.node isa AppNode || continue
        eligible[m.node] = get(counts, m.key, 0) >= min_count
    end
    return eligible
end

function _replace_at_path!(tree::AstTree, path::Vector{Int}, new_subtree::AstTree)
    if isempty(path)
        setroot!(tree, new_subtree)
    else
        parent = _parent_at_path(tree, path)
        setchild!(tree, parent, path[end], new_subtree)
    end
    return tree
end

function _legal_host(m::ScopeMatch, host_path::Vector{Int})
    return _path_prefix(host_path, m.path) && _path_prefix(m.limit_path, host_path)
end

function _anchor_antichain(matches::Vector{ScopeMatch}, anchor::ScopeMatch)
    kept = ScopeMatch[anchor]
    ordered = sort(matches; by=m -> (length(m.path), m.path))
    for m in ordered
        m === anchor && continue
        any(k -> _path_prefix(k.path, m.path) || _path_prefix(m.path, k.path), kept) && continue
        push!(kept, m)
    end
    return kept
end

function _single_lift_host(match::ScopeMatch)
    legal = [path for path in _ancestor_paths(match.path) if _legal_host(match, path)]
    strict = [path for path in legal if length(path) < length(match.path)]
    return isempty(strict) ? copy(match.path) : copy(last(strict))
end

function _select_scope_group(
    tree::AstTree,
    candidates::Vector{ScopeMatch},
    anchor::ScopeMatch,
    min_count::Int,
)
    min_count == 1 && return ScopeMatch[anchor], _single_lift_host(anchor)

    best = ScopeMatch[]
    best_host = Int[]
    for host in _ancestor_paths(anchor.path)
        _legal_host(anchor, host) || continue
        local_matches = [m for m in candidates if _path_prefix(host, m.path) && _legal_host(m, host)]
        group = _anchor_antichain(local_matches, anchor)
        length(group) >= min_count || continue
        actual_host = _common_path([m.path for m in group])
        all(m -> _legal_host(m, actual_host), group) || continue
        if length(group) > length(best) ||
           (length(group) == length(best) && length(actual_host) > length(best_host))
            best = group
            best_host = actual_host
        end
    end

    length(best) >= min_count || return Err(NotEnoughMatchesErr(min_count, length(best)))
    return best, best_host
end

function _local_depth_between(tree::AstTree, host_path::Vector{Int}, occurrence_path::Vector{Int})
    return _depth_at_path(tree, occurrence_path) - _depth_at_path(tree, host_path)
end

function _placeholder_body(tree::AstTree, host_path::Vector{Int}, matches::Vector{ScopeMatch})
    body = deepcopy(_node_at_path(tree, host_path))
    body_tree = AstTree(body)
    shift_free_variables!(body_tree, 1, 0)
    for m in sort(matches; by=m -> length(m.path), rev=true)
        rel = _relative_path(host_path, m.path)
        placeholder_depth = _local_depth_between(tree, host_path, m.path)
        _replace_at_path!(body_tree, rel, AstTree(IndexNode(placeholder_depth)))
    end
    return body_tree.root
end

function _rebased_representative(tree::AstTree, host_path::Vector{Int}, match::ScopeMatch)
    node = deepcopy(match.node)
    local_depth = _local_depth_between(tree, host_path, match.path)
    local_depth == 0 && return node
    shifted = AstTree(node)
    shift_free_variables!(shifted, -local_depth, local_depth)
    return shifted.root
end

function _eliminate_scope_matches_impl!(
    tree::AstTree,
    matches::Vector{ScopeMatch},
    anchor::ScopeMatch,
    host_path::Vector{Int};
    min_count::Int,
    factory=SymbolGenerator(),
)
    min_count >= 1 || return Err(ArgumentError("min_count must be >= 1"))
    length(matches) >= min_count || return Err(NotEnoughMatchesErr(min_count, length(matches)))
    _path_prefix(host_path, anchor.path) || return Err(CrossScopeMatchErr())
    all(m -> _legal_host(m, host_path), matches) || return Err(CrossScopeMatchErr())

    body = _placeholder_body(tree, host_path, matches)
    representative = _rebased_representative(tree, host_path, anchor)
    new_subtree = AppNode(AbsNode(factory(), body), representative)
    _replace_at_path!(tree, host_path, AstTree(new_subtree))
    return tree
end

"""
    eliminate_node!(tree::AstTree, target::AstNode; min_count=1, factory=SymbolGenerator())

Scope-aware extraction anchored by an actual AST node. Unlike the legacy
`CanonicalNode` entry, this compares external De Bruijn captures by binder
identity, so identical indices under different lambdas do not collide.
"""
function eliminate_node!(tree::AstTree, target::AstNode; min_count::Int=1, factory=SymbolGenerator())
    target_path = _path_to_node(tree, target)
    target_path === nothing && return Err(TargetNodeNotFoundErr())
    working = deepcopy(tree)
    working_target = _node_at_path(working, target_path)
    matches = _scope_matches(working)
    anchor = findfirst(m -> m.node === working_target, matches)
    anchor === nothing && return Err(TargetNodeNotFoundErr())
    anchor_match = matches[anchor]
    candidates = [m for m in matches if m.key == anchor_match.key]
    selected = _select_scope_group(working, candidates, anchor_match, min_count)
    selected isa Err && return selected
    group, host_path = selected
    result = _eliminate_scope_matches_impl!(working, group, anchor_match, host_path; min_count, factory)
    result isa Err && return result
    setroot!(tree, AstTree(working.root))
    return tree
end

function _depth_at(tree::AstTree, target::AstNode)
    d = Ref{Union{Nothing,Int}}(nothing)
    traverse(tree, AbstractionDepthState, PointerState{AstNode}) do _, node, bag
        node === target || return nothing
        d[] = getstate(bag, AbstractionDepthState).depth
        return Break()
    end
    return d[]
end

function eliminate_expression!(tree::AstTree, form::CanonicalNode; min_count::Int=1, factory=SymbolGenerator())
    working = deepcopy(tree)
    raw_matches = 0
    last_err = Err(NotEnoughMatchesErr(min_count, 0))
    for node in collect(working)
        c = canonicalize(working, node)
        c isa Err && continue
        c == form || continue
        raw_matches += 1
        result = eliminate_node!(working, node; min_count, factory)
        if result isa AstTree
            setroot!(tree, AstTree(working.root))
            return tree
        end
        last_err = result
    end
    raw_matches < min_count && return Err(NotEnoughMatchesErr(min_count, raw_matches))
    return last_err
end

function _eliminate_expression_impl!(tree::AstTree, form::CanonicalNode; min_count::Int=1, factory=SymbolGenerator())
    min_count >= 1 || return Err(ArgumentError("min_count must be >= 1"))

    matches = Tuple{Union{Nothing,AstNode},Int,Int}[] # (parent, child_index, depth)
    traverse(tree, AbstractionDepthState, PointerState{AstNode}) do t, node, bag
        c = canonicalize(t, node)
        c isa Err && return nothing
        c == form || return nothing
        pointer = getstate(bag, PointerState{AstNode})
        depth = getstate(bag, AbstractionDepthState).depth
        push!(matches, (pointer.parent, pointer.child_index, depth))
        return nothing
    end

    length(matches) >= min_count || return Err(NotEnoughMatchesErr(min_count, length(matches)))

    varname = factory()
    for (parent, child_index, _) in matches
        if parent === nothing
            setroot!(tree, AstTree(VarNode(varname)))
        else
            setchild!(tree, parent, child_index, AstTree(VarNode(varname)))
        end
    end

    lca_subtree = lca((t, x) -> x isa VarNode && (x::VarNode).name == varname, tree, root(tree))
    lca_subtree === nothing && return Err(ArgumentError("LCA not found for extracted var"))
    lca_depth = _depth_at(tree, lca_subtree)
    lca_depth === nothing && return Err(ArgumentError("Failed to compute LCA depth"))

    if length(matches) > 1
        all(((_, _, d),) -> d == lca_depth, matches) || return Err(CrossScopeMatchErr())
    end

    shift_free_variables!(AstTree(lca_subtree), 1, 0)

    new_subtree = AppNode(
        AbsNode(varname, lca_subtree),
        rebuild_node(form, factory),
    )
    eliminate_bound_variables!(AstTree(new_subtree))

    if lca_subtree === root(tree)
        setroot!(tree, AstTree(new_subtree))
    else
        setindex!(tree, new_subtree, lca_subtree)
    end
    return tree
end

function elimniate_expression!(tree::AstTree, form::CanonicalNode, factory=SymbolGenerator())
    return eliminate_expression!(tree, form; min_count=2, factory=factory)
end
    
########################################################
# Test Items
########################################################

@testsnippet CSEBaseImports begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.CSE: canonicalize
    using LambdaRegression.Utils: Err
    using LambdaRegression.ASTs.CSE: canonical_set, CanonicalIndex, CanonicalConst, CanonicalApp
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AbsNode, VarNode, AppNode, IndexNode, AstNode
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalApp, CanonicalIndex
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AbsNode, VarNode, AppNode, ConstNode, AstNode
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalApp, CanonicalIndex, CanonicalConst
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalIndex
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AstNode
    using LambdaRegression.ASTs.CSE: eliminate_expression!, canonicalize, eliminate_node!
end

@testitem "canonicalize alpha equivalence" setup=[CSEBaseImports] begin
    # Two alpha-equivalent lambdas should have identical canonical form
    tree1 = parse_lambda("λx. x")
    tree2 = parse_lambda("λy. y")
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root)
    canon2 = canonicalize(tree2, tree2.root)
    @test !(canon1 isa Err)
    @test canon1 == canon2

    # More complex alpha equivalence
    tree1 = parse_lambda("λf.λg.λx. f (g x)")
    tree2 = parse_lambda("λa.λb.λc. a (b c)")
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root)
    canon2 = canonicalize(tree2, tree2.root)
    @test canon1 == canon2

    # Alpha equivalence with constants
    ctx = @terms c1=1
    tree1 = parse_lambda("λx. x c1", ctx)
    tree2 = parse_lambda("λy. y c1", ctx)
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root)
    canon2 = canonicalize(tree2, tree2.root)
    @test canon1 == canon2
end

@testitem "canonical_set counts duplicates" setup=[CSEBaseImports] begin
    # The body x x contains two identical sub-expressions (#0) so the
    # frequency accumulator should report a count of 2 for that canonical VarNode as they are externally bound.
    tree = parse_lambda("λx. x x") # x is externally bound in x
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalIndex(0)] == 2

    # More complex expression with duplicates
    tree = parse_lambda("λf. (λx. f x) (λy. f y)")
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalIndex(1)] == 2

    # Complex expression with constants and duplicates
    ctx = @terms c1=1
    tree = parse_lambda("λf. (f c1) (f c1)", ctx)
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalApp(CanonicalIndex(0), CanonicalConst(:c1))] == 2
end

@testitem "elimniate_expression! complex expression" setup=[CSEBaseImports] begin
    # Test more complex case: λf. λg. (f g) (f g)
    # The expression (f g) appears twice and should be eliminated
    tree = parse_lambda("λf. λg. (f g) (f g)")
    eliminate_bound_variables!(tree)
    
    # Find the canonical form for (f g)
    acc = canonical_set(tree)
    app_form = CanonicalApp(CanonicalIndex(1), CanonicalIndex(0))
    @test acc[app_form] == 2  # (f g) appears twice
    
    # Eliminate the repeated expression
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, app_form)
    
    # Check the result structure
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

@testitem "elimniate_expression! with constants" setup=[CSEBaseImports] begin
    # Test with constants: λf. (f c1) (f c1)
    ctx = @terms c1=1
    tree = parse_lambda("λf. (f c1) (f c1)", ctx)
    eliminate_bound_variables!(tree)
    
    # Find the canonical form for (f c1)
    acc = canonical_set(tree)
    app_form = CanonicalApp(CanonicalIndex(0), CanonicalConst(:c1))
    @test acc[app_form] == 2  # (f c1) appears twice
    
    # Eliminate the repeated expression
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, app_form)
    
    # Verify the structure
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

@testitem "elimniate_expression! preserves bound variables correctly" setup=[CSEBaseImports] begin
    # Test that bound variables are handled correctly during elimination
    # λx. λy. x x should eliminate x and preserve proper binding
    tree = parse_lambda("λx. λy. x x")
    eliminate_bound_variables!(tree)
    
    # x appears twice in the inner scope
    acc = canonical_set(tree)
    x_form = CanonicalIndex(1)
    @test acc[x_form] == 2
    
    # Eliminate x
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, x_form)
    
    # The result should be properly formed
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

@testitem "eliminate_expression! single occurrence extraction" setup=[CSEBaseImports] begin
    tree = parse_lambda("λx. x")
    eliminate_bound_variables!(tree)
    form = canonicalize(tree, tree.root.body)
    form isa Err && error("canonicalize failed unexpectedly")
    original = deepcopy(tree)
    res = eliminate_expression!(tree, form; min_count=1)
    @test res isa AstTree
    @test res != original
    @test res.root isa AstNode
end

@testitem "eliminate_expression! cross-scope failure is transactional" setup=[CSEBaseImports] begin
    tree = parse_lambda("(λx. x) (λy. y)")
    eliminate_bound_variables!(tree)
    original = deepcopy(tree)

    res = elimniate_expression!(tree, CanonicalIndex(0))

    @test res isa Err
    @test tree == original
end

@testitem "eliminate_expression! does not capture existing free variables with fresh binder names" setup=[CSEBaseImports] begin
    existing = Symbol(0)
    tree = AstTree(AppNode(AppNode(ConstNode(:a), VarNode(existing)), ConstNode(:a)))
    original = deepcopy(tree)

    res = eliminate_expression!(tree, CanonicalConst(:a); min_count=2)

    @test res isa AstTree
    @test tree != original
    found_free_existing = Ref(false)
    for node in tree
        if node isa VarNode && node.name == existing
            found_free_existing[] = true
            break
        end
    end
    @test found_free_existing[]
end

@testitem "eliminate_node! does not merge same de Bruijn index under different binders" setup=[CSEBaseImports] begin
    tree = parse_lambda("(λx. x) (λy. y)")
    eliminate_bound_variables!(tree)
    original = deepcopy(tree)

    res = eliminate_node!(tree, tree.root.func.body; min_count=2)

    @test res isa Err
    @test tree == original
end

@testitem "eliminate_node! extracts repeated application with aligned capture" setup=[CSEBaseImports] begin
    tree = parse_lambda("λx. (f x) (f x)")
    eliminate_bound_variables!(tree)
    original = deepcopy(tree)
    anchor = tree.root.body.func

    res = eliminate_node!(tree, anchor; min_count=2)

    @test res isa AstTree
    @test tree != original
    @test tree.root isa AbsNode
end

@testitem "eliminate_node! does not capture existing free variables with fresh binder names" setup=[CSEBaseImports] begin
    existing = Symbol(0)
    tree = AstTree(AppNode(AppNode(ConstNode(:a), VarNode(existing)), ConstNode(:a)))
    original = deepcopy(tree)

    res = eliminate_node!(tree, tree.root.func.func; min_count=2)

    @test res isa AstTree
    @test tree != original
    found_free_existing = Ref(false)
    for node in tree
        if node isa VarNode && node.name == existing
            found_free_existing[] = true
            break
        end
    end
    @test found_free_existing[]
end

@testitem "eliminate_node! lifts same captured binder across nested depths" setup=[CSEBaseImports] begin
    tree = parse_lambda("λx. (f x) (λy. f x)")
    eliminate_bound_variables!(tree)
    original = deepcopy(tree)
    anchor = tree.root.body.func

    res = eliminate_node!(tree, anchor; min_count=2)

    @test res isa AstTree
    @test tree != original
    @test tree.root isa AbsNode
end

@testitem "eliminate_node! single lift uses a legal ancestor host" setup=[CSEBaseImports] begin
    tree = parse_lambda("λx. g (f x)")
    eliminate_bound_variables!(tree)
    anchor = tree.root.body.arg

    res = eliminate_node!(tree, anchor; min_count=1)

    @test res isa AstTree
    @test tree.root.body isa AppNode
    @test tree.root.body.func isa AbsNode
end

end # module CSE
