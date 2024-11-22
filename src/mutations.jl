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

include("constantmutations.jl")

include("hoistmutation.jl")

include("betareducemutation.jl")

include("subtreemutation.jl")

include("crossovermutations.jl")


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
