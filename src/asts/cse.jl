module CSE
using ..NodeASTs: ConstNode, AstTree, AstNode, AbsNode, IndexNode, AppNode, ScopeState, VarNode, eliminate_bound_variables!
using ..Trees: traverse, getstate, lca, root, setchild!, Leave, setindex!, StackState, PointerState
using ..NodeASTs: SymbolGenerator
using MLStyle, DataStructures, TestItems


@data CanonicalNode begin
    CanonicalAbs(body::CanonicalNode)
    CanonicalIndex(index::Int)
    CanonicalVar(name::Symbol)
    CanonicalConst(name::Symbol)
    CanonicalApp(func::CanonicalNode, arg::CanonicalNode)
end

function canonicalize(tree::AstTree, index::AstNode, scope::ScopeState, subscope::Vector{Symbol}=Symbol[])
    local_depth = length(subscope)
    if index isa IndexNode
        if index.index < local_depth # locally bound variable
            return CanonicalIndex(index.index)
        else # externally bound variable
            return CanonicalVar(scope.abstractions[end-(index.index-local_depth)].name) # find the binding name
        end
    elseif index isa ConstNode
        return CanonicalConst(index.name)
    elseif index isa VarNode
        return CanonicalVar(index.name)
    elseif index isa AbsNode
        return CanonicalAbs(canonicalize(tree, index.body, scope, Symbol[subscope; index.name]))
    elseif index isa AppNode
        return CanonicalApp(canonicalize(tree, index.func, scope, subscope), canonicalize(tree, index.arg, scope, subscope))
    end
    error("unknown node type: ", typeof(index)) # won't deal with VarNode, as tree is considered to be prepared in advance
end

function canonical_set(tree::AstTree)
    s = Accumulator{CanonicalNode, Int}()
    traverse(tree, ScopeState) do tree, node, bag
        inc!(s, canonicalize(tree, node, getstate(bag, ScopeState)))
    end
    return s
end

function rebuild_node(form::CanonicalNode, factory=SymbolGenerator())
    @match form begin
        CanonicalAbs(body) => AbsNode(factory(), rebuild_node(body, factory))
        CanonicalIndex(index) => IndexNode(index)
        CanonicalVar(name) => VarNode(name)
        CanonicalConst(name) => ConstNode(name)
        CanonicalApp(func, arg) => AppNode(rebuild_node(func), rebuild_node(arg))
    end
end
rebuild_tree(form::CanonicalNode, factory=SymbolGenerator()) = AstTree(rebuild_node(form, factory))

"""
    eliminate_expression!(tree::AstTree, form::CanonicalNode, factory=SymbolGenerator())

Eliminate the expression in `form` N in the tree `tree` by replacing it with a variable,
and replacing the lowest common ancestor M of the qualified nodes with a new application node (λx. M\\[N=x\\]) N.
The nodes to be eliminated should be ensured to be more than one in advance.
"""
function elimniate_expression!(tree::AstTree, form::CanonicalNode, factory=SymbolGenerator())
    varname = factory()
    traverse(tree, StackState{AstNode}, PointerState{AstNode}, ScopeState) do tree, node, bag
        if canonicalize(tree, node, getstate(bag, ScopeState)) == form
            (;parent, child_index) = getstate(bag, PointerState{AstNode})
            setchild!(tree, parent, child_index, AstTree(VarNode(varname)))
            return Leave()
        end
    end
    lca_subtree = lca((tree, x) -> x isa VarNode && x.name == varname, tree, root(tree))
    new_subtree = AppNode(
        AbsNode(varname, lca_subtree),
        rebuild_node(form, factory)
    )
    eliminate_bound_variables!(AstTree(new_subtree))
    setindex!(tree, new_subtree, lca_subtree)

    return tree
end
    
########################################################
# Test Items
########################################################

@testitem "canonicalize alpha equivalence" begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, ScopeState, AbsNode
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.CSE: canonicalize

    # Two alpha-equivalent lambdas should have identical canonical form
    tree1 = parse_lambda("λx. x")
    tree2 = parse_lambda("λy. y")
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root, ScopeState(AbsNode[]))
    canon2 = canonicalize(tree2, tree2.root, ScopeState(AbsNode[]))
    @test canon1 == canon2

    # More complex alpha equivalence
    tree1 = parse_lambda("λf.λg.λx. f (g x)")
    tree2 = parse_lambda("λa.λb.λc. a (b c)")
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root, ScopeState(AbsNode[]))
    canon2 = canonicalize(tree2, tree2.root, ScopeState(AbsNode[]))
    @test canon1 == canon2

    # Alpha equivalence with constants
    ctx = @terms c1=1
    tree1 = parse_lambda("λx. x c1", ctx)
    tree2 = parse_lambda("λy. y c1", ctx)
    eliminate_bound_variables!(tree1)
    eliminate_bound_variables!(tree2)
    canon1 = canonicalize(tree1, tree1.root, ScopeState(AbsNode[]))
    canon2 = canonicalize(tree2, tree2.root, ScopeState(AbsNode[]))
    @test canon1 == canon2
end

@testitem "canonical_set counts duplicates" begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.CSE: canonical_set, CanonicalVar, CanonicalConst, CanonicalApp

    # The body x x contains two identical sub-expressions (#0) so the
    # frequency accumulator should report a count of 2 for that canonical VarNode as they are externally bound.
    tree = parse_lambda("λx. x x") # x is externally bound in x
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalVar(:x)] == 2 # x is externally bound in x

    # More complex expression with duplicates
    tree = parse_lambda("λf. (λx. f x) (λy. f y)")
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalVar(:f)] == 2

    # Complex expression with constants and duplicates
    ctx = @terms c1=1
    tree = parse_lambda("λf. (f c1) (f c1)", ctx)
    eliminate_bound_variables!(tree)
    acc = canonical_set(tree)
    @test acc[CanonicalApp(CanonicalVar(:f), CanonicalConst(:c1))] == 2
end

@testitem "elimniate_expression! complex expression" begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AbsNode, VarNode, AppNode, IndexNode, AstNode
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalApp, CanonicalVar

    # Test more complex case: λf. λg. (f g) (f g)
    # The expression (f g) appears twice and should be eliminated
    tree = parse_lambda("λf. λg. (f g) (f g)")
    eliminate_bound_variables!(tree)
    
    # Find the canonical form for (f g)
    acc = canonical_set(tree)
    app_form = CanonicalApp(CanonicalVar(:f), CanonicalVar(:g))
    @test acc[app_form] == 2  # (f g) appears twice
    
    # Eliminate the repeated expression
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, app_form)
    
    # Check the result structure
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

@testitem "elimniate_expression! with constants" begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AbsNode, VarNode, AppNode, ConstNode, AstNode
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalApp, CanonicalVar, CanonicalConst

    # Test with constants: λf. (f c1) (f c1)
    ctx = @terms c1=1
    tree = parse_lambda("λf. (f c1) (f c1)", ctx)
    eliminate_bound_variables!(tree)
    
    # Find the canonical form for (f c1)
    acc = canonical_set(tree)
    app_form = CanonicalApp(CanonicalVar(:f), CanonicalConst(:c1))
    @test acc[app_form] == 2  # (f c1) appears twice
    
    # Eliminate the repeated expression
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, app_form)
    
    # Verify the structure
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

@testitem "elimniate_expression! preserves bound variables correctly" begin
    using LambdaRegression.ASTs.NodeASTs: eliminate_bound_variables!, AstTree, AbsNode, VarNode, AppNode, IndexNode, AstNode
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.CSE: elimniate_expression!, canonical_set, CanonicalVar

    # Test that bound variables are handled correctly during elimination
    # λx. λy. x x should eliminate x and preserve proper binding
    tree = parse_lambda("λx. λy. x x")
    eliminate_bound_variables!(tree)
    
    # x appears twice in the inner scope
    acc = canonical_set(tree)
    x_form = CanonicalVar(:x)
    @test acc[x_form] == 2
    
    # Eliminate x
    original_tree = deepcopy(tree)
    result_tree = elimniate_expression!(tree, x_form)
    
    # The result should be properly formed
    @test result_tree isa AstTree
    @test result_tree != original_tree
    @test result_tree.root isa AstNode
end

end # module CSE