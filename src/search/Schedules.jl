"""
Scalar schedules over explicit, normalized progress. `schedule_value` accepts
either a real progress in `[0,1]` (clamped at the endpoints) or an immutable
`ProposalProgress(completed_attempts, total_budget)`. Compose independent scalar
parameters in a tuple or named tuple; schedules own no clocks or mutable state.
"""
module Schedules

using TestItems

export AbstractSchedule, ConstantSchedule, LinearSchedule, ExponentialSchedule
export ProposalProgress, normalized_progress, schedule_value

abstract type AbstractSchedule end

function _finite(value::Real)
    out = Float64(value)
    isfinite(out) || throw(ArgumentError("schedule values must be finite"))
    return out
end

"""A finite scalar parameter independent of progress."""
struct ConstantSchedule <: AbstractSchedule
    value::Float64
    ConstantSchedule(value::Real) = new(_finite(value))
end

"""Linear interpolation from `initial` to `final`, including both endpoints."""
struct LinearSchedule <: AbstractSchedule
    initial::Float64
    final::Float64
    LinearSchedule(initial::Real, final::Real) = new(_finite(initial), _finite(final))
end

"""Geometric interpolation between strictly positive finite endpoints.

Use a linear schedule when an endpoint must be zero. Interpolation in log space
avoids overflowing the endpoint ratio for widely separated positive values.
"""
struct ExponentialSchedule <: AbstractSchedule
    initial::Float64
    final::Float64
    function ExponentialSchedule(initial::Real, final::Real)
        first, last = _finite(initial), _finite(final)
        first > 0 && last > 0 || throw(ArgumentError(
            "exponential schedule endpoints must be positive"))
        return new(first, last)
    end
end

"""Progress owned by a search: completed proposal attempts and a fixed budget.

Failures count as attempts. The budget is a normalization horizon, not a stopping
rule. Continuations must retain the cumulative numerator and original horizon;
independent searches can supply an explicit offset. Zero budgets are rejected.
"""
struct ProposalProgress
    completed_attempts::Int
    total_budget::Int
    function ProposalProgress(completed_attempts::Integer, total_budget::Integer)
        completed_attempts >= 0 || throw(ArgumentError("completed attempts must be nonnegative"))
        total_budget > 0 || throw(ArgumentError("schedule budget must be positive"))
        return new(Int(completed_attempts), Int(total_budget))
    end
end

normalized_progress(p::ProposalProgress) = min(p.completed_attempts / p.total_budget, 1.0)
normalized_progress(p::Real) = clamp(_finite(p), 0.0, 1.0)

schedule_value(value::Real, progress) = (_ = normalized_progress(progress); _finite(value))
schedule_value(s::ConstantSchedule, progress) = (_ = normalized_progress(progress); s.value)
function schedule_value(s::LinearSchedule, progress)
    p = normalized_progress(progress)
    return (1 - p) * s.initial + p * s.final
end
function schedule_value(s::ExponentialSchedule, progress)
    p = normalized_progress(progress)
    p == 0 && return s.initial
    p == 1 && return s.final
    return exp((1 - p) * log(s.initial) + p * log(s.final))
end
schedule_value(parameters::Tuple, progress) = map(s -> schedule_value(s, progress), parameters)
schedule_value(parameters::NamedTuple, progress) = map(s -> schedule_value(s, progress), parameters)

@testitem "Schedules normalize explicit cumulative progress and compose scalar parameters" begin
    using LambdaRegression.Schedules
    @test normalized_progress(ProposalProgress(0, 10)) == 0
    @test normalized_progress(ProposalProgress(4, 10)) == 0.4
    @test normalized_progress(ProposalProgress(12, 10)) == 1
    @test normalized_progress(-2) == 0
    @test_throws ArgumentError ProposalProgress(-1, 10)
    @test_throws ArgumentError ProposalProgress(0, 0)
    @test_throws ArgumentError normalized_progress(NaN)
    @test_throws ArgumentError ConstantSchedule(Inf)
    @test_throws ArgumentError ExponentialSchedule(1, 0)
    @test schedule_value(LinearSchedule(10, 2), 0.25) == 8
    @test schedule_value(ExponentialSchedule(100, 1), 0.5) ≈ 10
    @test schedule_value(ExponentialSchedule(1e-300, 1e300), 0.5) ≈ 1
    parameters = (; rate=LinearSchedule(1, 0), scale=ExponentialSchedule(1, 4), fixed=2)
    @test schedule_value(parameters, ProposalProgress(5, 10)) == (; rate=0.5, scale=2.0, fixed=2.0)
    @test schedule_value((ConstantSchedule(3), LinearSchedule(0, 1)), 1) == (3.0, 1.0)
end

end
