module CSE
using ...Utils: Err
using ..NodeASTs: ConstNode, AstTree, AstNode, AbsNode, IndexNode, AppNode, VarNode, SymbolGenerator, eliminate_bound_variables!, shift_free_variables!, AbstractionDepthState
using ...Trees: traverse, getstate, lca, root, setchild!, setroot!, setindex!, PointerState, Break
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
    using LambdaRegression.ASTs.CSE: eliminate_expression!, canonicalize
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

end # module CSE
