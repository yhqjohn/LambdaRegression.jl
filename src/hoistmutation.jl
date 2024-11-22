is_app_node(node::DBNode{T}) where T = node isa AppDB

"""
    HoistMutation{T} <: AbstractPointwiseMutation{T}

Mutation that hoists a randomly selected sub-node to replace its ancestor node in an *apply tree*.

# Fields
- `rate::Float64`: mutation rate
"""
struct HoistMutation{T} <: AbstractPointwiseMutation{T}
    filter::Any
end
HoistMutation{T}(rate::Float64) where T = HoistMutation{T}(FilterChain((is_app_node, RateFilter(rate))))
HoistMutation(rate::Float64) = HoistMutation{Union{}}(rate)
Base.convert(::Type{HoistMutation{T}}, mutation::HoistMutation{S}) where {T, S<:T} = HoistMutation{T}(mutation.rate)

function app_subnodes(node::DBNode{T}) where T
    if node isa AppDB{T}
        return DBNode{T}[[node]; app_subnodes(node.func); app_subnodes(node.arg)]
    else
        return DBNode{T}[node]
    end
end

"""
    point_mutate(mutation::HoistMutation{T}, node::DBNode{T}) where T

Randomly select a sub-node and hoist it to replace the target node, adjusting free variables.
"""
function point_mutate(m::HoistMutation{T}, node::DBNode{T}) where T
    # Collect all sub-nodes with their depths
    sub_nodes = app_subnodes(node)
    
    # Skip the first pair as it's the node itself
    if length(sub_nodes) <= 1
        return node
    end
    
    # Randomly select a sub-node (excluding the root)    
    return rand(sub_nodes[2:end])
end

@testitem "hoist mutation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: HoistMutation, mutate
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1234) # For reproducible tests
    
    tree = DBNode(@lambda (*) -> (&1 * &2))
    
    # Test HoistMutation
    hoist_mut = HoistMutation{Any}(1.0)
    mutated_tree = mutate(hoist_mut, tree)

    # @show mutated_tree
    @test mutated_tree isa DBNode{Any}
    @test mutated_tree !== tree  # Should be a new tree
end