"""
    AbstractConstantMutation{TTerm} <: AbstractPointwiseMutation{TypedTerm{TTerm}}

Abstract type for mutations that modify constant terms in a Lambda Calculus Tree.

# Interface
Required methods:
- [`point_mutate(mutation::AbstractConstantMutation{TTerm}, node::DBNode{TypedTerm{TTerm}}) where TTerm`](@ref): Apply mutation to a constant node
- [`mutation_rate(mutation::AbstractConstantMutation{TTerm}) where TTerm`](@ref): Get the mutation probability
"""
abstract type AbstractConstantMutation{TTerm} <: AbstractPointwiseMutation{TypedTerm{TTerm}} end

is_float_node(node::DBNode{TypedTerm{TTerm}}) where TTerm = node isa TypedTermProxy && (typelabel(node) === LabelType(:Float64) || typelabel(node) === LabelType(:Float32))

"""
    ScaleMutation{TTerm} <: AbstractConstantMutation{TTerm}

Mutation that scales floating point constants by a random factor drawn from 
a Gaussian distribution centered at 1.0.

# Fields
- `rate::Float64`: mutation rate
- `std::Float64`: standard deviation of the scaling factor (default: 0.1)
"""
struct ScaleMutation{TTerm} <: AbstractConstantMutation{TTerm}
    filter::Any
    std::Float64
end
ScaleMutation{TTerm}(rate::Float64, std::Float64=0.1) where TTerm = ScaleMutation{TTerm}(FilterChain((is_float_node, RateFilter(rate))), std)
ScaleMutation(rate::Float64, std::Float64=0.1) = ScaleMutation{Any}(rate, std)
ScaleMutation(filter::Any, std::Float64=0.1) = ScaleMutation{Any}(filter, std)

function point_mutate(m::ScaleMutation{TTerm}, node::DBNode{TypedTerm{TTerm}}) where TTerm
    value = unpack_to_value(node)
    scale = 1.0 + randn() * m.std
    scale = convert(typeof(value), scale)
    ProxyDB{TypedTerm{TTerm}}(TypedTerm{TTerm}(value * scale, typelabel(node)))
end


"""
    GaussianMutation{TTerm} <: AbstractConstantMutation{TTerm}

Mutation that adds Gaussian noise to floating point constants.

# Fields
- `rate::Float64`: mutation rate
- `std::Float64`: standard deviation of the Gaussian noise (default: 0.1)
"""
struct GaussianMutation{TTerm} <: AbstractConstantMutation{TTerm}
    filter::Any
    std::Float64
end
GaussianMutation{TTerm}(rate::Float64, std::Float64=0.1) where TTerm = GaussianMutation{TTerm}(FilterChain((is_float_node, RateFilter(rate))), std)
GaussianMutation(rate::Float64, std::Float64=0.1) = GaussianMutation{Any}(rate, std)
GaussianMutation(filter::Any, std::Float64=0.1) = GaussianMutation{Any}(filter, std)

function point_mutate(m::GaussianMutation{TTerm}, node::DBNode{TypedTerm{TTerm}}) where TTerm
    value = unpack_to_value(node)
    noise = randn() * m.std
    noise = convert(typeof(value), noise)
    ProxyDB{TypedTerm{TTerm}}(TypedTerm{TTerm}(value + noise, typelabel(node)))
end

is_int_node(node::DBNode{TypedTerm{TTerm}}) where TTerm = node isa TypedTermProxy && typelabel(node) === LabelType(:Int)

"""
    OffsetMutation{TTerm} <: AbstractConstantMutation{TTerm}

Mutation that adds ±1 to integer constants.

# Fields
- `rate::Float64`: mutation rate
"""
struct OffsetMutation{TTerm} <: AbstractConstantMutation{TTerm}
    filter::Any
end
OffsetMutation{TTerm}(rate::Float64) where TTerm = OffsetMutation{TTerm}(FilterChain((is_int_node, RateFilter(rate))))
OffsetMutation(rate::Float64) = OffsetMutation{Any}(rate)

function point_mutate(m::OffsetMutation{TTerm}, node::DBNode{TypedTerm{TTerm}}) where TTerm
    value = unpack_to_value(node)
    offset = rand((-1, 1))
    ProxyDB{TypedTerm{TTerm}}(TypedTerm{TTerm}(value + offset, typelabel(node)))
end

@testitem "mutation creation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: ScaleMutation, GaussianMutation, OffsetMutation, mutation_rate
    
    seed!(123) # For reproducible tests
    
    # Test constructors
    @test ScaleMutation(0.1) isa ScaleMutation{Any}
    @test ScaleMutation(0.1, 0.2) isa ScaleMutation{Any}
    @test GaussianMutation(0.1) isa GaussianMutation{Any}
    @test GaussianMutation(0.1, 0.2) isa GaussianMutation{Any}
    @test OffsetMutation(0.1) isa OffsetMutation{Any}
end

@testitem "mutation on single node" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: ScaleMutation, GaussianMutation, OffsetMutation, point_mutate
    using LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, ProxyDB, typelabel, LabelType
    
    seed!(123) # For reproducible tests
    
    # Create test nodes
    float_node = ProxyDB{TypedTerm{Any}}(TypedTerm(2.0, :Float64))
    int_node = ProxyDB{TypedTerm{Any}}(TypedTerm(2, :Int))
    
    # Test ScaleMutation
    scale_mut = ScaleMutation{Any}(1.0, 0.1)  # rate=1.0, std=0.1
    result = point_mutate(scale_mut, float_node)
    @test result isa ProxyDB{TypedTerm{Any}}
    @test typelabel(result) === LabelType(:Float64)
    @test result.value.value != 2.0
    
    # Test GaussianMutation
    gauss_mut = GaussianMutation{Any}(1.0, 0.1)  # rate=1.0, std=0.1
    result = point_mutate(gauss_mut, float_node)
    @test result isa ProxyDB{TypedTerm{Any}}
    @test typelabel(result) === LabelType(:Float64)
    @test result.value.value != 2.0
    
    # Test OffsetMutation
    offset_mut = OffsetMutation{Any}(1.0)  # rate=1.0
    result = point_mutate(offset_mut, int_node)
    @test result isa ProxyDB{TypedTerm{Any}}
    @test typelabel(result) === LabelType(:Int)
    @test abs(result.value.value - 2) == 1  # Should be ±1 from original value
end

@testitem "mutation on tree" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: ScaleMutation, GaussianMutation, mutate
    using LambdaRegression.TermsModule: TypedTerm
    using LambdaRegression.ParseModule: @lambda
    using LambdaRegression.BuiltinTermsModule: FADD, FMUL
    
    seed!(123) # For reproducible tests
    
    # Create a simple expression tree: (2.0 * x) + 1.0 using @lambda and builtin terms
    tree = DBNode{TypedTerm{Any}}(@lambda x -> FADD * (FMUL * 2.0 * x) * 1.0)
    
    # Test ScaleMutation on tree
    scale_mut = ScaleMutation{Any}(1.0, 0.1)
    mutated_tree = mutate(scale_mut, tree)
    @test mutated_tree isa DBNode{TypedTerm{Any}}
    @test mutated_tree !== tree  # Should be a new tree
    
    # Test GaussianMutation on tree
    gauss_mut = GaussianMutation{Any}(1.0, 0.1)
    mutated_tree = mutate(gauss_mut, tree)
    @test mutated_tree isa DBNode{TypedTerm{Any}}
    @test mutated_tree !== tree  # Should be a new tree
end

@testitem "custom node filter" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: ScaleMutation, mutate
    using LambdaRegression.TermsModule: TypedTerm, TypedTermProxy, ProxyDB, typelabel, LabelType
    using LambdaRegression.DeBruijnModule: DBNode
    
    seed!(123) # For reproducible tests
    
    # Create test nodes
    node_2 = ProxyDB{TypedTerm{Any}}(TypedTerm(2.0, :Float64))
    node_1 = ProxyDB{TypedTerm{Any}}(TypedTerm(1.0, :Float64))
    
    # Create a custom filter that only affects nodes with value > 1.0
    custom_filter(node::DBNode{TypedTerm{Any}}) = 
        node isa TypedTermProxy &&
        typelabel(node) === LabelType(:Float64) &&
        node.value.value > 1.0
    
    # Test ScaleMutation with custom filter
    scale_mut = ScaleMutation{Any}(custom_filter, 0.5)  # rate=1.0, std=0.5
    
    # Should be mutated (value > 1.0)
    result = mutate(scale_mut, node_2)
    @test result !== node_2
    @test result.value.value != 2.0
    
    # Should not be mutated (value <= 1.0)
    result = mutate(scale_mut, node_1)
    @test result === node_1
end