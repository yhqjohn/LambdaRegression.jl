"""Private case-wise parent selection primitives for typed search.

The automatic epsilon-lexicase kernel here is deliberately isolated from
search state and configuration.  It consumes a cases-by-members matrix of
finite, nonnegative errors and returns one member column.  Each case is
processed in a fresh random order; its median and median absolute deviation
are recomputed on the current survivors.  The implementation sorts a small
scratch vector, so its worst-case work is `O(cases * members * log(members))`.
"""

using Random: AbstractRNG, rand, shuffle!

"""Median of an already sorted finite nonnegative scratch vector.

The nonnegative difference midpoint avoids overflow for large inputs and
preserves equal subnormal inputs instead of halving both down to zero.
"""
@inline function _typed_sorted_median(values::Vector{Float64})
    isempty(values) && throw(ArgumentError("cannot take median of empty values"))
    middle = (length(values) + 1) >>> 1
    isodd(length(values)) && return values[middle]
    lower, upper = values[middle], values[middle + 1]
    return lower + (upper - lower) / 2
end

function _typed_case_error(value, case_index, member_index)
    value isa Real || throw(ArgumentError(
        "case errors must be real at ($case_index, $member_index)"))
    converted = try
        Float64(value)
    catch
        throw(ArgumentError(
            "case errors must convert to Float64 at ($case_index, $member_index)"))
    end
    isfinite(converted) || throw(ArgumentError(
        "case errors must be finite at ($case_index, $member_index)"))
    converted >= 0 || throw(ArgumentError(
        "case errors must be nonnegative at ($case_index, $member_index)"))
    return converted
end

"""Select one member column with automatic epsilon-lexicase selection.

`errors` is indexed as cases × members.  Lower finite nonnegative errors are
better.  The selected case order is shuffled with `rng`; after each case,
survivors retain errors no larger than that case's current best plus its
current median absolute deviation.  Recomputing the statistics after every
filter is essential because the survivor set changes.  When one or more
survivors remain after the case order is exhausted, one is selected
uniformly.

The private `case_order` keyword is a deterministic test seam and is not a
selection policy or task configuration.  It must be a permutation of the
matrix's row indices.  The input matrix is read-only and is never reordered or
otherwise mutated.
"""
function _typed_epsilon_lexicase_index(rng::AbstractRNG, errors::AbstractMatrix;
                                       case_order=nothing)
    case_axis, member_axis = axes(errors, 1), axes(errors, 2)
    isempty(case_axis) && throw(ArgumentError("case error matrix has no cases"))
    isempty(member_axis) && throw(ArgumentError("case error matrix has no members"))

    # Validate once without constructing a replacement matrix.  The selection
    # pass below converts only the current survivor slice into scratch values.
    for case_index in case_axis, member_index in member_axis
        _typed_case_error(errors[case_index, member_index],
                          case_index, member_index)
    end

    cases = if case_order === nothing
        order = collect(case_axis)
        shuffle!(rng, order)
        order
    else
        order = collect(case_order)
        length(order) == length(case_axis) || throw(ArgumentError(
            "case_order must contain every case exactly once"))
        all(index -> index isa Integer, order) || throw(ArgumentError(
            "case_order must contain integer case indices"))
        all(index -> index in case_axis, order) || throw(ArgumentError(
            "case_order contains an invalid case index"))
        length(unique(order)) == length(order) || throw(ArgumentError(
            "case_order must contain every case exactly once"))
        order
    end

    survivors = collect(member_axis)
    scratch = Float64[]
    for case_index in cases
        length(survivors) > 1 || break
        resize!(scratch, length(survivors))
        for (offset, member_index) in enumerate(survivors)
            scratch[offset] = Float64(errors[case_index, member_index])
        end

        best = minimum(scratch)
        sort!(scratch)
        median = _typed_sorted_median(scratch)
        for offset in eachindex(scratch)
            scratch[offset] = abs(scratch[offset] - median)
        end
        sort!(scratch)
        mad = _typed_sorted_median(scratch)
        threshold = best + mad

        write_index = 0
        for member_index in survivors
            value = Float64(errors[case_index, member_index])
            value <= threshold || continue
            write_index += 1
            survivors[write_index] = member_index
        end
        resize!(survivors, write_index)
        isempty(survivors) && throw(ArgumentError(
            "epsilon-lexicase produced no survivors from valid errors"))
    end
    return rand(rng, survivors)
end

@testitem "TypedSearch automatic epsilon lexicase selects case specialists" begin
    using Test
    using Random: MersenneTwister

    selector = LambdaRegression.TypedSearch._typed_epsilon_lexicase_index
    errors = Float64[
        0  2   4
        2  0 100
    ]
    original = copy(errors)
    @test all(seed -> selector(MersenneTwister(seed), errors;
        case_order=[1, 2]) == 2, 1:32)
    @test errors == original

    # Case 2 is recomputed after case 1 removes member 3: the current pair
    # has median 1 and MAD 1, so only member 2 remains.  A stale full-population
    # MAD is 2 and would incorrectly leave both members 1 and 2 tied.
    @test LambdaRegression.TypedSearch._typed_sorted_median(
        [floatmax(Float64), floatmax(Float64)]) == floatmax(Float64)
    @test LambdaRegression.TypedSearch._typed_sorted_median(
        [nextfloat(0.0), nextfloat(0.0)]) == nextfloat(0.0)
end

@testitem "TypedSearch automatic epsilon lexicase ties and validation" begin
    using Test
    using Random: MersenneTwister

    selector = LambdaRegression.TypedSearch._typed_epsilon_lexicase_index
    ties = zeros(Float64, 2, 4)
    first_rng = MersenneTwister(2000)
    second_rng = MersenneTwister(2000)
    first_draws = [selector(first_rng, ties) for _ in 1:256]
    second_draws = [selector(second_rng, ties) for _ in 1:256]
    @test first_draws == second_draws
    @test sort(unique(first_draws)) == [1, 2, 3, 4]

    one = reshape([0.0, 1.0], 2, 1)
    @test selector(MersenneTwister(1234), one) == 1
    @test_throws ArgumentError selector(MersenneTwister(1234), zeros(Float64, 0, 1))
    @test_throws ArgumentError selector(MersenneTwister(1234), zeros(Float64, 1, 0))
    @test_throws ArgumentError selector(MersenneTwister(1234), [0.0 NaN])
    @test_throws ArgumentError selector(MersenneTwister(1234), [0.0 Inf])
    @test_throws ArgumentError selector(MersenneTwister(1234), [0.0 -1.0])
    @test_throws ArgumentError selector(MersenneTwister(1234),
        ones(Float64, 2, 2); case_order=[1, 1])
    @test_throws ArgumentError selector(MersenneTwister(1234),
        ones(Float64, 2, 2); case_order=[1, 3])
end
