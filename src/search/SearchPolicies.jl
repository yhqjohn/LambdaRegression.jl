"""
    SearchPolicies

Small mathematical policy tools shared by otherwise independent search loops.
The module owns the bounded running frequency window and acceptance
probability calculation; proposal generation, selection, and population
lifecycle remain with each search implementation.
"""
module SearchPolicies

using TestItems

"""Running complexity frequencies in a finite, pseudocount-bounded window."""
mutable struct _RegularizedFrequency
    window::Int
    frequencies::Vector{Float64}
    normalized_frequencies::Vector{Float64}
end

function _RegularizedFrequency(window::Int, max_complexity::Int)
    window >= 1 || throw(ArgumentError("frequency_window must be positive"))
    max_complexity >= 1 || throw(ArgumentError("max_complexity must be positive"))
    frequencies = ones(Float64, max_complexity)
    return _RegularizedFrequency(window, frequencies, copy(frequencies) ./ sum(frequencies))
end

function _regularized_move_window!(frequency::_RegularizedFrequency)
    difference = sum(frequency.frequencies) - frequency.window
    loops = 0
    while difference > 0 && loops < length(frequency.frequencies)
        indices = findall(value -> value > 1.0, frequency.frequencies)
        isempty(indices) && break
        amount = min(difference / length(indices),
            minimum(frequency.frequencies[indices]) - 1.0)
        # Even a one-ulp excess above the pseudocount floor must be removed.
        # Otherwise that smallest bucket blocks trimming every larger bucket
        # on all later observations. Each floor-limited pass removes at least
        # one active bucket, so the number of buckets bounds the passes.
        amount <= 0.0 && break
        frequency.frequencies[indices] .-= amount
        difference -= amount * length(indices)
        loops += 1
    end
    return frequency
end

function _regularized_normalize!(frequency::_RegularizedFrequency)
    total = sum(frequency.frequencies)
    total > 0 || throw(ArgumentError("running complexity frequencies must be positive"))
    frequency.normalized_frequencies .= frequency.frequencies ./ total
    return frequency
end

function _regularized_frequency(frequency::_RegularizedFrequency, complexity::Int)
    0 < complexity <= length(frequency.normalized_frequencies) || return 1.0
    return frequency.normalized_frequencies[complexity]
end

"""
    _regularized_acceptance_probability(parent_loss, child_loss; ...)

Compute the acceptance probability from annealing and optional frequency
correction inputs. Combine both factors before clipping at one: an improving
child can offset the penalty of moving to a more frequent complexity. Compute
in log space so cold-cycle improvements cannot overflow before that combination.
When both corrections are disabled, every finite child is
accepted, including a worse child; this is regularized-evolution replacement,
not elitist hill climbing. Search loops own the decision to enable corrections
and their temperature schedule.
"""
function _regularized_acceptance_probability(parent_loss::Real, child_loss::Real;
        annealing::Bool=false, temperature::Real=1.0, alpha::Real=1.0,
        old_frequency::Real=1.0, new_frequency::Real=1.0)
    isfinite(child_loss) || return 0.0
    log_probability = 0.0
    if annealing
        temperature > 0 || throw(ArgumentError("temperature must be positive"))
        alpha > 0 || throw(ArgumentError("alpha must be positive"))
        delta = Float64(child_loss) - Float64(parent_loss)
        iszero(delta) || (log_probability =
            -delta / (Float64(temperature) * Float64(alpha)))
    end
    if old_frequency != 1.0 || new_frequency != 1.0
        old_frequency >= 0 || throw(ArgumentError("old_frequency must be nonnegative"))
        new_frequency >= 0 || throw(ArgumentError("new_frequency must be nonnegative"))
        iszero(old_frequency) && return 0.0
        log_probability += log(Float64(old_frequency)) -
            log(max(Float64(new_frequency), eps(Float64)))
    end
    return exp(min(log_probability, 0.0))
end

@testitem "Search acceptance combines annealing and frequency before clipping" begin
    using Test
    using LambdaRegression.SearchPolicies: _regularized_acceptance_probability

    accept(args...; kwargs...) = _regularized_acceptance_probability(args...;
        annealing=true, temperature=1.0, alpha=1.0, kwargs...)
    # An improvement compensates part of the penalty for a common size.
    @test accept(2.0, 1.0; old_frequency=0.1, new_frequency=1.0) ≈ exp(1.0) / 10
    # At the cold end, a strict improvement outweighs that same penalty.
    @test _regularized_acceptance_probability(2.0, 1.0;
        annealing=true, temperature=eps(Float64), alpha=1.0,
        old_frequency=0.1, new_frequency=1.0) == 1.0
    @test accept(1.0, 2.0; old_frequency=1.0, new_frequency=0.1) == 1.0
    @test accept(1.0, 1.0; old_frequency=0.1, new_frequency=1.0) ≈ 0.1
    @test accept(1.0, 2.0) ≈ exp(-1.0)
    @test accept(2.0, 1.0) == 1.0
    @test accept(1.0, Inf) == 0.0
    @test accept(Inf, 1.0; old_frequency=0.1, new_frequency=1.0) == 1.0
    @test accept(2.0, 1.0; old_frequency=0.0, new_frequency=1.0) == 0.0
end

@testitem "Search frequency window trims near-floor buckets" begin
    using Test
    using LambdaRegression.SearchPolicies: _RegularizedFrequency,
        _regularized_move_window!, _regularized_normalize!
    using Random: MersenneTwister, rand

    state = _RegularizedFrequency(10, 2)
    state.frequencies .= [nextfloat(1.0), 100.0]
    _regularized_move_window!(state)
    @test sum(state.frequencies) <= 10 + 1e-9
    @test all(>=(1.0), state.frequencies)

    state = _RegularizedFrequency(4096, 254)
    rng = MersenneTwister(1234)
    for _ in 1:200
        for size in rand(rng, 1:254, 64)
            state.frequencies[size] += 1.0
        end
        _regularized_move_window!(state)
        @test sum(state.frequencies) <= 4096 + 1e-8
        @test all(>=(1.0), state.frequencies)
    end
    _regularized_normalize!(state)
    @test sum(state.normalized_frequencies) ≈ 1.0
end

end # module SearchPolicies
