"""
    BetaReduceMutation{T} <: AbstractPointwiseMutation{T}

Mutation that performs beta reduction on randomly selected nodes.

# Fields
- `rate::Float64`: mutation rate
"""
struct BetaReduceMutation{T} <: AbstractPointwiseMutation{T}
    filter::Any
end
BetaReduceMutation{T}(rate::Float64) where T = BetaReduceMutation{T}(FilterChain((is_app_node, RateFilter(rate))))
BetaReduceMutation(rate::Float64) = BetaReduceMutation{Union{}}(rate)
Base.convert(::Type{BetaReduceMutation{T}}, mutation::BetaReduceMutation{S}) where {T, S<:T} = BetaReduceMutation{T}(mutation.rate)

"""
    point_mutate(mutation::BetaReduceMutation{T}, node::DBNode{T}) where T

Attempt to beta reduce the selected node.
"""
function point_mutate(m::BetaReduceMutation{T}, node::DBNode{T}) where T
    result = beta_reduce(node)
    if result === normalized
        return node
    else
        return result
    end
end

@testitem "beta reduce mutation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: BetaReduceMutation, mutate
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1234) # For reproducible tests
    
    # Create a simple lambda expression: (λx.x) 1
    tree = DBNode{Int}(@lambda (x->x)*1)
    
    # Test BetaReduceMutation
    beta_mut = BetaReduceMutation{Int}(1.0)
    mutated_tree = mutate(beta_mut, tree)
    
    @test mutated_tree isa DBNode{Int}
    @test mutated_tree !== tree  # Should be a new tree
    @test mutated_tree == DBNode{Int}(@lambda 1)  # Should be reduced to just 1
end