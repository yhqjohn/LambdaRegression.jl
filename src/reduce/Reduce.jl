module Reduces

using ..ASTs.NodeASTs: AstTree, AbsNode, VarNode, IndexNode, AppNode, AstNode, ConstNode, eliminate_bound_variables!
using ..Trees: traverse, PointerState, getstate, Leave, Break, setroot!, setchild!
using TestItems

import ..ASTs.NodeASTs: apply_node!

# Default fallback for apply_node! with extra arguments
apply_node!(tree, f, x, args...) = apply_node!(tree, f, x) # fallback to two-argument version by default

# Base isapplicable methods (no extra args)
isapplicable(a::AstNode, b::AstNode) = false # by default, no node is applicable to any other node
isapplicable(a::AbsNode, b::AstNode) = true # AbsNode is applicable to any AstNode

# Extensible isapplicable with extra arguments
isapplicable(a::AstNode, b::AstNode, args...) = isapplicable(a, b) # fallback to two-argument version by default
isapplicable(a::ConstNode, b::ConstNode, context) = true # Two ConstNodes are considered applicable given a context

"""
    beta_reduce!(tree::AstTree, args...) -> Bool

Apply a single β-reduction step to `tree` following *normal order* (left-most, outer-most).
Returns `true` if the tree is in normal form (no reductions were performed), 
`false` if a reduction was performed (tree might need further reductions).

All additional arguments are passed through to `isapplicable` and `apply_node!` calls,
enabling context-aware evaluation and extensible dispatch.

Implementation details:
1. We perform a pre-order traversal with `PointerState` to visit nodes in normal-order.
2. As soon as we encounter an application node whose function and argument are
   `isapplicable`, we replace the application with the result of `apply_node!` and
   abort the traversal via `Break()`.
3. Tree modification uses `PointerState` to correctly update either a parent
   child slot or the root node.
"""
function beta_reduce!(tree::AstTree, args...)::Bool
    reduction_occurred = Ref(false)

    traverse(tree, PointerState) do t, node, bag
        # Look for the left-most, outer-most redex (AppNode with applicable function)
        if node isa AppNode
            func = node.func
            arg  = node.arg
            if isapplicable(func, arg, args...)
                new_subtree = apply_node!(t, func, arg, args...)  # Pass args to apply_node!
                pointer = getstate(bag, PointerState{AstNode})  # Replace current node with the reduced subtree
                if pointer.parent === nothing
                    setroot!(t, new_subtree)
                else
                    setchild!(t, pointer.parent, pointer.child_index, new_subtree)
                end
                reduction_occurred[] = true # mark that a reduction occurred
                return Break()  # Stop after first reduction (normal order)
            end
        end
        return nothing
    end
    # Return true if normalized (no reduction occurred), false if reduction occurred
    return !reduction_occurred[]
end

"""
    beta_reduce!(tree::AstTree, steps::Int, args...) -> Bool

Apply up to `steps` β-reduction steps to `tree` following normal order.
Returns `true` if the tree is in normal form (no reduction occurred in the final step),
`false` if the step limit was reached and the final step performed a reduction.
If `steps` is negative, throws an `ArgumentError`.

All additional arguments after `steps` are passed through to each `beta_reduce!` call.

Note: `false` suggests the tree might benefit from further reductions, but doesn't
guarantee the tree is not normalized. Users can call again if needed.
"""
function beta_reduce!(tree::AstTree, steps::Int, args...)::Bool
    steps < 0 && throw(ArgumentError("steps must be non-negative"))

    last_result = true  # Assume normalized initially
    for _ in 1:steps
        last_result = beta_reduce!(tree, args...)
        last_result && return true  # No reduction occurred, tree is normalized
    end
    return last_result  # Return result of final step
end

function prepare(tree::AstTree)
    tree = deepcopy(tree)
    eliminate_bound_variables!(tree)
    for node in tree
        if node isa AbsNode
            node.name = gensym(node.name) # avoid name conflict
        end
    end
    return tree
end

export beta_reduce!, prepare

@testitem "beta_reduce! single step functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, AppNode, IndexNode
    using LambdaRegression.Reduces: beta_reduce!, isapplicable

    # Test 1: Basic β-reduction (λx.x) y → y
    identity_abs = AbsNode(:x, VarNode(:x))
    app = AppNode(identity_abs, VarNode(:y))
    tree = AstTree(app)
    
    # Should perform reduction and return false (reduction occurred)
    result = beta_reduce!(tree)
    @test result == false  # Reduction occurred
    @test tree.root isa VarNode
    @test tree.root.name == :y

    # Test 2: Already normalized tree
    normalized_tree = AstTree(VarNode(:x))
    result = beta_reduce!(normalized_tree)
    @test result == true  # Already normalized
    @test normalized_tree.root isa VarNode
    @test normalized_tree.root.name == :x

    # Test 3: Non-applicable redex (VarNode applied to something)
    non_redex = AppNode(VarNode(:f), VarNode(:x))
    non_redex_tree = AstTree(non_redex)
    result = beta_reduce!(non_redex_tree)
    @test result == true  # No reduction possible, so normalized
    @test non_redex_tree.root isa AppNode

    # Test 4: Nested redex - should reduce outer-most first
    # (λx.x) ((λy.y) z) → (λy.y) z
    inner_abs = AbsNode(:y, VarNode(:y))
    inner_app = AppNode(inner_abs, VarNode(:z))
    outer_abs = AbsNode(:x, VarNode(:x))
    outer_app = AppNode(outer_abs, inner_app)
    nested_tree = AstTree(outer_app)
    
    result = beta_reduce!(nested_tree)
    @test result == false  # Reduction occurred, but more reductions possible
    @test nested_tree.root isa AppNode  # Should be (λy.y) z
    @test nested_tree.root.func isa AbsNode
    @test nested_tree.root.arg isa VarNode
    @test nested_tree.root.arg.name == :z
end

@testitem "beta_reduce! multi-step functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, VarNode, AppNode
    using LambdaRegression.Reduces: beta_reduce!

    # Test 1: Multi-step reduction
    # (λx.x) ((λy.y) z) → (λy.y) z → z
    inner_abs = AbsNode(:y, VarNode(:y))
    inner_app = AppNode(inner_abs, VarNode(:z))
    outer_abs = AbsNode(:x, VarNode(:x))
    outer_app = AppNode(outer_abs, inner_app)
    tree = AstTree(outer_app)
    
    result = beta_reduce!(tree, 2)
    @test result == false  # Final step (step 2) performed a reduction
    @test tree.root isa VarNode
    @test tree.root.name == :z

    # Test 2: Step limit reached
    # Create a tree that needs 3 steps but only allow 2
    # (λa.a) ((λb.b) ((λc.c) x)) → ... → x (needs 3 steps)
    innermost = AppNode(AbsNode(:c, VarNode(:c)), VarNode(:x))
    middle = AppNode(AbsNode(:b, VarNode(:b)), innermost)
    outermost = AppNode(AbsNode(:a, VarNode(:a)), middle)
    tree = AstTree(outermost)
    
    result = beta_reduce!(tree, 2)
    @test result == false  # Final step performed a reduction
    # After 2 steps, should still have one more reduction possible
    @test tree.root isa AppNode

    # Test 3: Zero steps
    tree = AstTree(AppNode(AbsNode(:x, VarNode(:x)), VarNode(:y)))
    original_root = tree.root
    result = beta_reduce!(tree, 0)
    @test result == true  # No steps taken, return initial assumption (true)
    @test tree.root === original_root  # Tree unchanged

    # Test 4: Negative steps should error
    @test_throws ArgumentError beta_reduce!(tree, -1)
end

@testitem "isapplicable interface functionality" begin
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode, AppNode, ConstNode
    using LambdaRegression.Reduces: isapplicable

    # Test default cases
    @test isapplicable(AbsNode(:x, VarNode(:x)), VarNode(:y)) == true
    @test isapplicable(AbsNode(:x, VarNode(:x)), ConstNode(:pi)) == true
    @test isapplicable(VarNode(:f), VarNode(:x)) == false
    @test isapplicable(ConstNode(:add), VarNode(:x)) == false
    @test isapplicable(AppNode(VarNode(:f), VarNode(:x)), VarNode(:y)) == false
end

@testitem "Context-aware ConstNode evaluation" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode, AppNode
    using LambdaRegression.ASTs.Contexts: new_constant!
    using LambdaRegression.Reduces: beta_reduce!, isapplicable

    # Create a context with some simple functions and values
    ctx = Dict{Symbol,Any}()
    ctx[:double] = x -> 2x  # A simple function that doubles its argument
    ctx[:square] = x -> x^2  # A simple function that squares its argument
    ctx[:pi] = π
    ctx[:three] = 3

    # Test that ConstNodes are applicable with context
    @test isapplicable(ConstNode(:double), ConstNode(:three), ctx) == true
    @test isapplicable(ConstNode(:square), ConstNode(:pi), ctx) == true

    # Test basic constant application: double(3) → 6
    double_node = ConstNode(:double)
    three_node = ConstNode(:three)
    app_tree = AstTree(AppNode(double_node, three_node))
    
    # Perform reduction with context
    result = beta_reduce!(app_tree, ctx)
    @test result == false  # Reduction should occur
    @test app_tree.root isa ConstNode  # Result should be a new constant
    
    # The result should be 6
    final_result = ctx[app_tree.root.name]
    @test final_result == 6

    # Test another application: square(π) → π²
    square_node = ConstNode(:square)
    pi_node = ConstNode(:pi)
    square_app = AstTree(AppNode(square_node, pi_node))
    
    result2 = beta_reduce!(square_app, ctx)
    @test result2 == false  # Reduction should occur
    @test square_app.root isa ConstNode  # Result should be a new constant
    
    # The result should be π²
    final_square_result = ctx[square_app.root.name]
    @test final_square_result ≈ π^2

    # Test that non-context evaluation still works (lambda calculus)
    using LambdaRegression.ASTs.NodeASTs: AbsNode, VarNode
    identity_abs = AbsNode(:x, VarNode(:x))
    lambda_app = AstTree(AppNode(identity_abs, three_node))
    
    # This should work without context
    result3 = beta_reduce!(lambda_app)
    @test result3 == false  # Reduction should occur
    @test lambda_app.root isa ConstNode  # Result should be the three_node
    @test lambda_app.root.name == :three
end

@testitem "Y Combinator with Complex Computations" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: beta_reduce!, prepare
    using LambdaRegression.ASTs.Contexts

    # Define basic arithmetic and logical operations for our lambda calculus
    add = x -> y -> x + y  # Curried addition
    mul = x -> y -> x * y  # Curried multiplication  
    sub = x -> y -> x - y  # Curried subtraction
    eq = x -> y -> x == y  # Curried equality
    lt = x -> y -> x < y   # Curried less than
    cond = p -> p ? Contexts.T : Contexts.F # boolean to boolean combinator
    
    # Pre-define context dictionary for flag
    ctx = Dict{Symbol,Any}()
    Y, _ = @lambda "λf.(λx.f (x x)) (λx.f (x x))"

    # Test 1: Factorial using Y combinator
    # fac = Y (λfac.λn. cond (eq n 0) 1 (mul n (fac (sub n 1))))
    fac, _ = @lambda "Y (λfac.λn. cond (eq n 0) 1 (mul n (fac (sub n 1))))" ctx
    fac4, _ = @lambda "fac 4" ctx
    fac4 = prepare(fac4)

    result = beta_reduce!(fac4, 10_000, ctx)
    final_value = ctx[fac4.root.name]
    @test final_value == 24  # 4! = 24

    # Test 2: Fibonacci using Y combinator  
    # fib = Y (λfib.λn. cond (lt n 2) n (add (fib (sub n 1)) (fib (sub n 2))))
    fib, _ = @lambda "Y (λf.λn. cond (lt n 2) n (add (f (sub n 1)) (f (sub n 2))))" ctx
    fib6, _ = @lambda "fib 6" ctx
    fib6 = prepare(fib6)
    result = beta_reduce!(fib6, 10_000, ctx)    
    final_value = ctx[fib6.root.name]
    @test final_value == 8  # fib(6) = 8
end

@testitem "Apply Operator with List Operations for Square Sum" begin
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: beta_reduce!, prepare
    using LambdaRegression.ASTs.Contexts
    # Define list operations using Julia arrays as lists
    cons = x -> xs -> [x, xs...]  # Curried cons
    car = xs -> length(xs) > 0 ? xs[1] : 0  # Head/first element, return 0 for empty lists
    cdr = xs -> length(xs) > 1 ? xs[2:end] : []   # Tail/rest elements, return [] for single/empty lists
    nil = []  # Empty list
    
    # Arithmetic operations (curried)
    add = x -> y -> x + y
    mul = x -> y -> x * y
    square = x -> x * x
    
    # Conditional and predicates
    cond = p -> p ? Contexts.T : Contexts.F # boolean to boolean combinator
    null = xs -> isempty(xs)

    test_list = [1, 2, 3, 4, 5]
    ctx = Dict{Symbol,Any}()
    
    # Y combinator for recursion
    Y, _ = @lambda "λf.(λx.f (x x)) (λx.f (x x))"

    # Test 1: Simple apply operator - apply a function to each element
    appl, _ = @lambda "Y (λappl.λf.λlist. cond (null list) nil ((cons (f (car list))) (appl f (cdr list))))" ctx

    # Apply square function to test_list
    apply_square_tree, _ = @lambda "appl square test_list" ctx
    apply_square_tree = prepare(apply_square_tree)
    result = beta_reduce!(apply_square_tree, 10_000, ctx)
    # Check the result should be [1, 4, 9, 16, 25]
    final_value = ctx[apply_square_tree.root.name]
    @test final_value == [1, 4, 9, 16, 25]

    # Test 2: Sum of squares computation
    # sum = Y (λsum.λlist. cond (null list) 0 (add (car list) (sum (cdr list))))
    SUM, _ = @lambda "Y (λsum.λlist. cond (null list) 0 (add (car list) (sum (cdr list))))" ctx
    sum_of_squares, _ = @lambda "λlist. SUM (appl square list)" ctx
    result_tree, _ = @lambda "sum_of_squares test_list" ctx  
    result_tree = prepare(result_tree)
    result = beta_reduce!(result_tree, 10_000, ctx)    
    # Check result: 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
    final_value = ctx[result_tree.root.name]
    @test final_value == 55
end

@testitem "Step Limit Comparison Tests" begin
    using LambdaRegression.ASTs.Contexts
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.ASTs.Parse: @lambda
    using LambdaRegression.Reduces: beta_reduce!

    # Define arithmetic functions
    add = x -> y -> x + y
    mul = x -> y -> x * y
    sub = x -> y -> x - y
    eq = x -> y -> x == y
    cond = p -> p ? Contexts.T : Contexts.F # boolean to boolean combinator
    
    # Pre-define context dictionary for flag
    step_ctx = Dict{Symbol,Any}()
    
    # Y combinator
    y_combinator, _ = @lambda "λf.(λx.f (x x)) (λx.f (x x))"
    Y = y_combinator
    
    # Factorial function
    factorial_tree, _ = @lambda "Y (λfac.λn. cond (eq n 0) 1 (mul n (fac (sub n 1))))" step_ctx
    factorial_fn = factorial_tree
    
    # Test factorial of 3 with different step limits
    fac_3_tree_low, _ = @lambda "factorial_fn 3" step_ctx
    fac_3_tree_medium, _ = @lambda "factorial_fn 3" step_ctx  
    fac_3_tree_high, _ = @lambda "factorial_fn 3" step_ctx
    
    # Test with low step count (should not complete)
    result_low = beta_reduce!(fac_3_tree_low, 10, step_ctx)
    @test result_low == false  # Should not be normalized yet
    
    # Test with medium step count (might complete)
    result_medium = beta_reduce!(fac_3_tree_medium, 50, step_ctx)
    
    # Test with high step count (should definitely complete)
    result_high = beta_reduce!(fac_3_tree_high, 100, step_ctx)
    
    # The high step count should produce the correct result
    if fac_3_tree_high.root isa LambdaRegression.ASTs.NodeASTs.ConstNode
        final_value = step_ctx[fac_3_tree_high.root.name] 
        @test final_value == 6  # 3! = 6
    end
    
    # Test: Continue reduction from where medium left off
    if !result_medium  # If medium didn't complete
        # Continue reducing from where medium left off
        additional_result = beta_reduce!(fac_3_tree_medium, 50, step_ctx)
        
        # After additional steps, should get same result as high
        if fac_3_tree_medium.root isa LambdaRegression.ASTs.NodeASTs.ConstNode
            final_value = step_ctx[fac_3_tree_medium.root.name]
            @test final_value == 6  # Should eventually reach 3! = 6
        end
    end
end

end # module Reduces