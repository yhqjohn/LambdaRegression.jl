"""
    AbstractCrossover{T} 

Abstract type for crossover operations that can be applied to Lambda Calculus Trees.
# Interface
Required methods:
- [`crossover(crossover::AbstractCrossover{T}, target::DBNode{T}, source::DBNode{T}) where T`](@ref): Apply crossover to two trees
"""
abstract type AbstractCrossover{T} end

"""
    crossover(crossover::AbstractCrossover{T}, target::DBNode{T}, source::DBNode{T}) where T

Apply crossover from source to target.
"""
crossover(crossover::AbstractCrossover{T}, target::DBNode{T}, source::DBNode{T}) where T = error("crossover is not implemented for $(typeof(crossover))")

struct PointCrossover{T} <: AbstractCrossover{T}
    rate::Float64
end
function crossover(crossover::PointCrossover{T}, target::DBNode{T}, source::DBNode{T}) where T
    source_nodes = collect(source)

    traverse(target) do n, depth
        if rand() < crossover.rate 
            source_nodes = filter(node -> max_bound(node) <= depth, source_nodes)
            if isempty(source_nodes)
                nothing
            else
                rand(source_nodes)
            end
        else
            nothing
        end
    end
end

@testitem "point crossover" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: PointCrossover, crossover
    using LambdaRegression.DeBruijnModule: DBNode, max_bound
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1234) # For reproducible tests
    
    # Create complex source and target expressions
    # Target: λf.(λx.f (x x)) (λx.f (x x))  [Y combinator]
    target = DBNode(@lambda f->((x->f*(x*x)) * (x->f*(x*x))))
    
    # Source: λf.λg.λx.(f x) (g x)  [Function composition]
    source = DBNode(@lambda f->g->x->(f*x)*(g*x))
    
    # Create crossover operator with high rate to ensure mutations
    cross = PointCrossover{Any}(0.1)
    
    for _ in 1:10
        # Apply crossover
        result = crossover(cross, target, source)
        display(result)
        
        # Check that all variables are properly bound
        # by ensuring max_bound is less than the depth at each node
        for (depth, node) in pairs(result)
            @test max_bound(node) <= depth
        end
    end
end

struct SwapMutation{T} <: AbstractMutation{T}
    crossover::PointCrossover{T}
end
SwapMutation{T}(rate::Float64) where T = SwapMutation{T}(PointCrossover{T}(rate))
mutate(mutation::SwapMutation{T}, node::DBNode{T}) where T = crossover(mutation.crossover, node, node)

@testitem "swap mutation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: SwapMutation, mutate
    using LambdaRegression.DeBruijnModule: DBNode, max_bound
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1234) # For reproducible tests
    
    # Create complex source and target expressions
    # Target: λf.(λx.f (x x)) (λx.f (x x))  [Y combinator]
    node = DBNode(@lambda f->((x->f*(x*x)) * (x->f*(x*x))))

    swap = SwapMutation{Any}(0.1)
    for _ in 1:10
        mutated_tree = mutate(swap, node)
        display(mutated_tree)
        for (depth, node) in pairs(mutated_tree)
            @test max_bound(node) <= depth
        end
    end
end