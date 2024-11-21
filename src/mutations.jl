module MutationsModule

using ..DeBruijnModule: DBNode, traverse
using ..TermsModule: TypedTerm, TypedTermProxy, ProxyDB, typelabel, LabelType, unpack_to_value, pack_to_typedtermnode
using ..DeBruijnModule: shift_free, AppDB, AbsDB, VarDB, max_bound
using TestItems
using ..ReduceModule: beta_reduce, normalized

"""
    AbstractMutation{T}

Abstract type representing mutations that can be applied to Lambda Calculus Trees.

# Interface
Required methods:
- [`mutate(mutation::AbstractMutation{T}, node::DBNode{T}) where T`](@ref): Apply the mutation to a tree

Type parameter `T` corresponds to the type parameter of the Lambda Calculus Tree (`DBNode{T}`).
"""
abstract type AbstractMutation{T} end

"""
    mutate(mutation::AbstractMutation{T}, node::DBNode{T}) where T

Apply a mutation to a Lambda Calculus Tree.
"""
mutate(mutation::AbstractMutation{T}, node::DBNode{T}) where T = error("mutate not implemented for mutation type $(typeof(mutation))")

struct FilterChain
    filters::Tuple
end
(f::FilterChain)(node::DBNode{T}) where T = all(filter -> filter(node), f.filters)

"""
    RateFilter

Filter that passes with probability equal to the mutation rate.
"""
struct RateFilter
    rate::Float64
end
(f::RateFilter)(node::DBNode{T}) where T = rand() < f.rate

"""
    AbstractPointwiseMutation{T} <: AbstractMutation{T}

Abstract type for mutations that can be applied pointwise to nodes in a Lambda Calculus Tree.

# Interface

Optional methods:
- [`mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T`](@ref): Override default two-argument mutation that calls three-argument version.
- [`mutate(f, mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T`](@ref): Override default three-argument mutation that calls `point_mutate`.
- [`point_mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T`](@ref): Apply mutation to a single node.

Note that if `point_mutate` is you have to implement at least one of the three methods.

Fields:
- `filter`: filter function to determine which nodes can be mutated
"""
abstract type AbstractPointwiseMutation{T} <: AbstractMutation{T} end

"""
    mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T

Apply a mutation to a Lambda Calculus Tree. Default implementation uses node_filter.
"""
mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T = mutate(mutation.filter, mutation, node)

"""
    mutate(f, mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T

Apply a mutation to a Lambda Calculus Tree using a filter function. Default implementation 
uses traverse with point_mutate.
"""
mutate(filter_fn, mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T = traverse(node) do n, depth
    if filter_fn(n)
        point_mutate(mutation, n)
    else
        nothing
    end
end


"""
    point_mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T

Apply a pointwise mutation to a single node.
"""
point_mutate(mutation::AbstractPointwiseMutation{T}, node::DBNode{T}) where T = error("point_mutate not implemented for mutation type $(typeof(mutation))")


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

struct JointMutation{T} <: AbstractMutation{T}
    mutations::Vector{AbstractMutation{T}}
end
JointMutation(mutations::Vararg{AbstractMutation{T}}) where T = JointMutation{T}(AbstractMutation{T}[mutations...])


function mutate(mutation::JointMutation{T}, node::DBNode{T}) where T
    for mut in mutation.mutations
        node = mutate(mut, node)
    end
    return node
end

@testitem "joint mutation" begin
    using Random: seed!
    using LambdaRegression.MutationsModule: JointMutation, ScaleMutation, OffsetMutation, mutate
    using LambdaRegression.TermsModule: TypedTerm
    using LambdaRegression.DeBruijnModule: DBNode
    using LambdaRegression.ParseModule: @lambda
    
    seed!(1234) # For reproducible tests
    
    node = DBNode{TypedTerm{Any}}(@lambda x->(x*2.0*3))
    mutation = JointMutation(ScaleMutation{Any}(1.0, 0.1), OffsetMutation{Any}(1.0))
    mutated_tree = mutate(mutation, node)
    display(mutated_tree)
end

end
