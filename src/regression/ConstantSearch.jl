"""
Constant search for regression individuals.

`search_constants` is the seam shared by one-shot Gaussian mutation and later
derivative-free or derivative-backed adapters. Callers supply the objective,
budget, and RNG. Implementations return an independently owned individual and
classified counters; they never overwrite the source individual.
"""
abstract type AbstractConstantSearch end

using ..Programs: numeric_parameter_blocks

"""Finite proposal and objective-evaluation budget for one constant search."""
struct ConstantSearchBudget
    max_proposals::Int
    max_objective_evals::Int

    function ConstantSearchBudget(max_proposals::Int, max_objective_evals::Int)
        max_proposals >= 0 || throw(ArgumentError("max_proposals must be nonnegative"))
        max_objective_evals >= 1 || throw(ArgumentError(
            "max_objective_evals must include at least the source evaluation",
        ))
        return new(max_proposals, max_objective_evals)
    end
end

ConstantSearchBudget(; max_proposals::Int=1, max_objective_evals::Int=max_proposals + 1) =
    ConstantSearchBudget(max_proposals, max_objective_evals)

"""Deterministic accounting for one `search_constants` call."""
struct ConstantSearchStats
    proposals::Int
    applicable_proposals::Int
    changed_candidates::Int
    accepted_candidates::Int
    loss_improving_candidates::Int
    objective_evaluations::Int
    nonfinite_outcomes::Int
    invalid_outcomes::Int
    scalar_proposals::Int
    tensor_proposals::Int
end

"""Owned result and classified terminal state from constant search."""
struct ConstantSearchResult
    program::Program
    status::Symbol
    loss::Float64
    stats::ConstantSearchStats
end

"""
Gaussian search over scalar and dense/nested real-tensor parameter blocks.
Admission is shared with structural observation through `Programs`' numerical
schema. Updates retain layout/type/shape; updating one symbol can split an
alias with another symbol. Published children must be reclassified.

Tensor modes are sampled uniformly from `tensor_modes`. The default is
structured and excludes elementwise full-rank noise; callers may explicitly
include `:full` for exploratory runs.
"""
struct GaussianConstantSearch <: AbstractConstantSearch
    σ::Float64
    tensor_modes::Tuple{Vararg{Symbol}}

    function GaussianConstantSearch(σ::Real, tensor_modes::Tuple{Vararg{Symbol}})
        value = Float64(σ)
        isfinite(value) && value > 0 || throw(ArgumentError("σ must be finite and positive"))
        isempty(tensor_modes) && throw(ArgumentError("tensor_modes must be non-empty"))
        allowed = (:global_bias, :global_scale, :rank1, :sparse, :full)
        all(mode -> mode in allowed, tensor_modes) || throw(ArgumentError(
            "unsupported tensor Gaussian mode",
        ))
        return new(value, tensor_modes)
    end
end

GaussianConstantSearch(σ::Real) = GaussianConstantSearch(
    σ,
    (:global_bias, :global_scale, :rank1, :sparse),
)
GaussianConstantSearch(;
    σ::Real=0.1,
    tensor_modes::Tuple{Vararg{Symbol}}=(:global_bias, :global_scale, :rank1, :sparse),
) = GaussianConstantSearch(σ, tensor_modes)

function _runtime_tensor_shape(value)
    if value isa AbstractArray{<:AbstractFloat}
        return size(value)
    elseif value isa AbstractVector && !isempty(value)
        child_shape = _runtime_tensor_shape(first(value))
        child_shape === nothing && return nothing
        all(item -> _runtime_tensor_shape(item) == child_shape, value) || return nothing
        return (length(value), child_shape...)
    end
    return nothing
end

function _runtime_tensor_leaf_type(value)
    current = value
    while current isa AbstractVector && !(current isa AbstractArray{<:AbstractFloat})
        isempty(current) && return nothing
        current = first(current)
    end
    current isa AbstractArray{<:AbstractFloat} || return nothing
    return eltype(current)
end

function _copy_tensor_to_dense!(destination, value, prefix::Tuple=())
    if value isa AbstractArray{<:AbstractFloat}
        for index in CartesianIndices(value)
            coordinates = (prefix..., Tuple(index)...)
            destination[coordinates...] = value[index]
        end
        return destination
    end
    for index in eachindex(value)
        _copy_tensor_to_dense!(destination, value[index], (prefix..., index))
    end
    return destination
end


function _dense_tensor(value)
    shape = _runtime_tensor_shape(value)
    shape === nothing && return nothing
    leaf = _runtime_tensor_leaf_type(value)
    leaf === nothing && return nothing
    destination = Array{leaf}(undef, shape)
    return _copy_tensor_to_dense!(destination, value)
end


function _copy_dense_to_tensor!(destination, source, prefix::Tuple=())
    if destination isa AbstractArray{<:AbstractFloat}
        for index in CartesianIndices(destination)
            coordinates = (prefix..., Tuple(index)...)
            destination[index] = source[coordinates...]
        end
        return destination
    end
    for index in eachindex(destination)
        _copy_dense_to_tensor!(destination[index], source, (prefix..., index))
    end
    return destination
end


function _restore_tensor(template, dense)
    destination = deepcopy(template)
    return _copy_dense_to_tensor!(destination, dense)
end

function _rank1_gaussian(rng::AbstractRNG, value::AbstractArray{T}, σ::Float64) where {T<:AbstractFloat}
    dims = size(value)
    noise = ones(T, dims)
    for axis in eachindex(dims)
        factor = randn(rng, T, dims[axis])
        shape = ntuple(index -> index == axis ? dims[axis] : 1, length(dims))
        noise .*= reshape(factor, shape)
    end
    return value .+ noise .* convert(T, σ)
end

function _mutate_tensor_block(
    strategy::GaussianConstantSearch,
    rng::AbstractRNG,
    value::AbstractArray,
)
    dense = _dense_tensor(value)
    dense === nothing && throw(ArgumentError("unsupported real tensor runtime representation"))
    mutated = _mutate_dense_tensor_block(strategy, rng, dense)
    return _restore_tensor(value, mutated)
end

function _mutate_dense_tensor_block(
    strategy::GaussianConstantSearch,
    rng::AbstractRNG,
    value::AbstractArray{T},
) where {T<:AbstractFloat}
    mode = rand(rng, strategy.tensor_modes)
    σ = convert(T, strategy.σ)
    if mode === :global_bias
        return value .+ convert(T, randn(rng)) * σ
    elseif mode === :global_scale
        return value .* exp(convert(T, randn(rng)) * σ)
    elseif mode === :rank1
        return _rank1_gaussian(rng, value, strategy.σ)
    elseif mode === :sparse
        result = copy(value)
        indices = CartesianIndices(result)
        index = rand(rng, indices)
        result[index] += convert(T, randn(rng)) * σ
        return result
    elseif mode === :full
        return value .+ randn(rng, T, size(value)) .* σ
    end
    throw(ArgumentError("unsupported tensor Gaussian mode: $mode"))
end

function _mutate_constant_block(strategy, rng, value, kind::Symbol)
    if kind === :scalar
        return value + convert(typeof(value), randn(rng) * strategy.σ)
    elseif kind === :tensor
        return _mutate_tensor_block(strategy, rng, value)
    end
    throw(ArgumentError("unsupported constant block kind: $kind"))
end

function _constant_objective(objective, program::Program)
    value = objective(program)
    value isa Err && return value
    value isa Real || throw(ArgumentError("constant-search objective must return Real or Err"))
    loss = Float64(value)
    isfinite(loss) || return Err(NonFiniteOutputErr())
    return loss
end

"""
    search_constants(strategy, individual, objective; rng, budget)

Search mutable constants without modifying `program`. The Gaussian adapter
uses strict improvement and starts every proposal from the best accepted state.
"""
function search_constants(
    strategy::GaussianConstantSearch,
    individual::Program,
    objective;
    rng::AbstractRNG,
    budget::ConstantSearchBudget=ConstantSearchBudget(),
)
    blocks = numeric_parameter_blocks(individual)
    if isempty(blocks)
        return ConstantSearchResult(
            individual,
            :ineligible,
            Inf,
            ConstantSearchStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
    end

    best = individual
    baseline = _constant_objective(objective, best)
    if baseline isa Err
        nonfinite = baseline.value isa NonFiniteOutputErr ? 1 : 0
        return ConstantSearchResult(
            best,
            :invalid_source,
            Inf,
            ConstantSearchStats(0, 0, 0, 0, 0, 1, nonfinite, 1, 0, 0),
        )
    end
    best_loss = baseline::Float64

    proposal_limit = min(budget.max_proposals, budget.max_objective_evals - 1)
    changed = 0
    accepted = 0
    nonfinite = 0
    invalid = 0
    scalar_proposals = 0
    tensor_proposals = 0
    for _ in 1:proposal_limit
        proposal = best
        block = rand(rng, blocks)
        name, kind = first(block), last(block)
        before = proposal.local_consts[name]
        proposal = with_constants(
            proposal,
            Dict{Symbol,Any}(
                name => _mutate_constant_block(strategy, rng, before, kind),
            ),
        )
        changed += !isequal(proposal.local_consts[name], before)
        scalar_proposals += kind === :scalar
        tensor_proposals += kind === :tensor

        candidate_loss = _constant_objective(objective, proposal)
        if candidate_loss isa Err
            nonfinite += candidate_loss.value isa NonFiniteOutputErr
            invalid += 1
            continue
        end
        if candidate_loss < best_loss
            best = proposal
            best_loss = candidate_loss
            accepted += 1
        end
    end

    stats = ConstantSearchStats(
        proposal_limit,
        proposal_limit,
        changed,
        accepted,
        accepted,
        proposal_limit + 1,
        nonfinite,
        invalid,
        scalar_proposals,
        tensor_proposals,
    )
    return ConstantSearchResult(best, accepted > 0 ? :improved : :no_improvement, best_loss, stats)
end

@testitem "Regression constant search improves scalar blocks transactionally" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: FloatTy, TyLike
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch
    using LambdaRegression.Regression: search_constants

    parent = Program(
        AstTree(ConstNode(:c));
        local_consts=Dict{Symbol,Any}(:c => 0.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy),
    )
    objective = individual -> abs2(individual.local_consts[:c] - 1.0)
    strategy = GaussianConstantSearch(σ=0.5)
    budget = ConstantSearchBudget(max_proposals=16, max_objective_evals=17)

    first = search_constants(strategy, parent, objective; rng=MersenneTwister(555), budget)
    second = search_constants(strategy, parent, objective; rng=MersenneTwister(555), budget)

    @test first.status == :improved
    @test first.loss < objective(parent)
    @test first.program.local_consts == second.program.local_consts
    @test first.stats == second.stats
    @test first.stats.proposals == 16
    @test first.stats.objective_evaluations == 17
    @test parent.local_consts == Dict{Symbol,Any}(:c => 0.0)
    @test first.program !== parent
    @test first.program.local_consts !== parent.local_consts
end

@testitem "Regression constant search mutates tensor blocks structurally" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch
    using LambdaRegression.Regression: search_constants
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyLike, TyNat

    tensor_ty = TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    source = zeros(2, 3)
    parent = Program(
        AstTree(ConstNode(:A));
        local_consts=Dict{Symbol,Any}(:A => source),
        local_consts_ty=Dict{Symbol,TyLike}(:A => tensor_ty),
    )
    objective = individual -> -sum(abs2, individual.local_consts[:A])
    strategy = GaussianConstantSearch(;
        σ=0.25,
        tensor_modes=(:rank1,),
    )
    budget = ConstantSearchBudget(max_proposals=1, max_objective_evals=2)

    first = search_constants(strategy, parent, objective; rng=MersenneTwister(2000), budget)
    second = search_constants(strategy, parent, objective; rng=MersenneTwister(2000), budget)
    value = first.program.local_consts[:A]
    delta = value - source
    rank_one = all(
        isapprox(
            delta[1, left] * delta[2, right],
            delta[1, right] * delta[2, left];
            atol=1.0e-10,
        )
        for left in axes(delta, 2), right in axes(delta, 2)
    )

    @test first.status == :improved
    @test size(value) == (2, 3)
    @test rank_one
    @test value == second.program.local_consts[:A]
    @test value !== source
    @test parent.local_consts[:A] !== source
    @test parent.local_consts[:A] == zeros(2, 3)
    @test first.stats.scalar_proposals == 0
    @test first.stats.tensor_proposals == 1
end

@testitem "Regression constant search adapts nested vector tensor blocks" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: ConstantSearchBudget, GaussianConstantSearch
    using LambdaRegression.Regression: search_constants
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyLike, TyNat

    tensor_ty = TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    source = [zeros(3), zeros(3)]
    parent = Program(
        AstTree(ConstNode(:A));
        local_consts=Dict{Symbol,Any}(:A => source),
        local_consts_ty=Dict{Symbol,TyLike}(:A => tensor_ty),
    )
    objective = individual -> -sum(sum(abs2, row) for row in individual.local_consts[:A])
    strategy = GaussianConstantSearch(σ=0.25, tensor_modes=(:rank1,))
    budget = ConstantSearchBudget(max_proposals=1, max_objective_evals=2)

    result = search_constants(strategy, parent, objective; rng=MersenneTwister(1), budget)
    value = result.program.local_consts[:A]
    delta = [value[i][j] - source[i][j] for i in 1:2, j in 1:3]
    rank_one = all(
        isapprox(
            delta[1, left] * delta[2, right],
            delta[1, right] * delta[2, left];
            atol=1.0e-10,
        )
        for left in axes(delta, 2), right in axes(delta, 2)
    )

    @test result.status == :improved
    @test value isa Vector{Vector{Float64}}
    @test length.(value) == [3, 3]
    @test rank_one
    @test value !== source
    @test all(value[i] !== source[i] for i in eachindex(value))
    @test parent.local_consts[:A] == [zeros(3), zeros(3)]
end
