function generate_lambda_tree(var_rate::Float64, abs_rate::Float64, depth::UInt=UInt(0))
    @assert var_rate + abs_rate <= 1.0
    if rand() < var_rate && depth > 0
        return VarDB(rand(UInt(0):depth-1))
    elseif rand() < var_rate + abs_rate 
        return AbsDB(generate_lambda_tree(var_rate, abs_rate, depth + 1))
    else
        return AppDB(generate_lambda_tree(var_rate, abs_rate, depth), generate_lambda_tree(var_rate, abs_rate, depth))
    end
end

function generate_app_tree(::Type{T}, var_rate::Float64, proxy_rate::Float64, lambda_rate::Float64, context::Vector{T}, depth::UInt=UInt(0)) where T
    @assert var_rate + proxy_rate + lambda_rate <= 1.0 
    if rand() < var_rate && depth > 0
        return VarDB{T}(rand(UInt(0):depth-1))
    elseif rand() < var_rate + proxy_rate
        return ProxyDB{T}(rand(context))
    elseif rand() < var_rate + proxy_rate + lambda_rate
        return convert(DBNode{T}, generate_lambda_tree(var_rate + proxy_rate, lambda_rate, depth))
    else
        return AppDB{T}(generate_app_tree(T, var_rate, proxy_rate, lambda_rate, context, depth), generate_app_tree(T, var_rate, proxy_rate, lambda_rate, context, depth))
    end
end

@testitem "generate app tree" begin
    using LambdaRegression.MutationsModule: generate_app_tree
    using LambdaRegression.TermsModule: TypedTerm
    using LambdaRegression.ParseModule: @lambda
    using Random: seed!
    
    seed!(1234) # For reproducible tests
    
    context = [TypedTerm{Any}(1.0, :Float64), TypedTerm{Any}(2.0, :Float64)]
    for _ in 1:10
        tree = generate_app_tree(TypedTerm{Any}, 0.25, 0.25, 0.25, context, UInt(3))
        display(tree)
    end
    tree = generate_app_tree(TypedTerm{Any}, 0.25, 0.25, 0.25, context, UInt(3))
    @test tree isa DBNode{TypedTerm{Any}}
end

"""
    SubtreeMutation{T} <: AbstractPointwiseMutation{T}

Mutation that replaces a subtree with a randomly generated one.

# Fields
- `rate::Float64`: mutation rate
- `var_rate::Float64`: probability of generating a variable node
- `proxy_rate::Float64`: probability of generating a proxy node
- `lambda_rate::Float64`: probability of generating a lambda abstraction
- `context::Vector{T}`: collection of proxy terms to sample from
"""
struct SubtreeMutation{T} <: AbstractPointwiseMutation{T}
    filter::Any
    var_rate::Float64
    proxy_rate::Float64
    lambda_rate::Float64
    context::Vector{T}
end
SubtreeMutation{T}(rate::Float64, var_rate::Float64, proxy_rate::Float64, lambda_rate::Float64, context::Vector{T}) where T = SubtreeMutation{T}(FilterChain((is_app_node, RateFilter(rate))), var_rate, proxy_rate, lambda_rate, context)
function SubtreeMutation(rate::Float64, var_rate::Float64, proxy_rate::Float64, lambda_rate::Float64, context::Vector{T}) where T
    @assert var_rate + proxy_rate + lambda_rate <= 1.0
    SubtreeMutation{T}(RateFilter(rate), var_rate, proxy_rate, lambda_rate, context)
end

mutation_rate(mutation::SubtreeMutation{T}) where T = mutation.rate

"""
    mutate(f, mutation::SubtreeMutation{T}, node::DBNode{T}) where T

Apply mutation using traverse to track abstraction depth for variable generation.
"""
function mutate(filter_fn::Function, mutation::SubtreeMutation{T}, node::DBNode{T}) where T
    traverse(node) do n, depth
        if filter_fn(n)
            generate_app_tree(T, mutation.var_rate, mutation.proxy_rate, mutation.lambda_rate, mutation.context, depth)
        else
            nothing
        end
    end
end

@testitem "subtree mutation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: SubtreeMutation, mutate
    using LambdaRegression.TermsModule: TypedTerm
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1) # For reproducible tests
    
    # Create test context and tree
    context = [TypedTerm{Any}(1.0, :Float64), TypedTerm{Any}(2.0, :Float64)]
    tree = DBNode{TypedTerm{Any}}(@lambda x -> x)
    
    # Test SubtreeMutation
    subtree_mut = SubtreeMutation{TypedTerm{Any}}(0.5, 0.25, 0.25, 0.25, context)
    for _ in 1:10
        mutated_tree = mutate(subtree_mut, tree)
        display(mutated_tree)
    end
    subtree_mut = SubtreeMutation{TypedTerm{Any}}(1.0, 0.25, 0.25, 0.25, context)
    mutated_tree = mutate(subtree_mut, tree)
    @test mutated_tree isa DBNode{TypedTerm{Any}}
    # @test mutated_tree !== tree  # Should be a new tree
end