"""
    _typed_nsga2_select(fitness, capacity)

Compute the standard NSGA-II survivor selection for two minimised objectives.
`fitness[i]` is `(mean_training_loss, written_complexity)` and duplicate rows
remain distinct individuals.  The returned named tuple contains the selected
original indices, a one-based nondomination rank for every input, a crowding
distance for every input, and all nondominated fronts.  Complete fronts are
kept in original-index order; only the last partial front is ordered by
descending crowding distance and then original index.

The helper is deliberately numeric and requires both objectives to be finite;
callers may therefore filter invalid losses before calling it, but this seam
still rejects a non-finite objective explicitly.  Front endpoints receive
`Inf` crowding distances.  The input is not mutated.  Its implementation is
O(N² + N log N) for two objectives.
"""
function _typed_nsga2_select(
        fitness::AbstractVector{<:Tuple{<:Real,<:Real}}, capacity::Integer)
    n = length(fitness)
    capacity >= 0 && capacity <= n || throw(ArgumentError(
        "NSGA-II capacity must be between zero and the number of individuals"))
    capacity_int = Int(capacity)
    points = Vector{NTuple{2,Float64}}(undef, n)
    for (index, pair) in enumerate(fitness)
        first_objective = Float64(pair[1])
        second_objective = Float64(pair[2])
        isfinite(first_objective) || throw(ArgumentError(
            "NSGA-II objectives must be finite"))
        isfinite(second_objective) || throw(ArgumentError(
            "NSGA-II objectives must be finite"))
        points[index] = (first_objective, second_objective)
    end

    dominates = function (left, right)
        (left[1] <= right[1] && left[2] <= right[2]) &&
            (left[1] < right[1] || left[2] < right[2])
    end
    dominated = [Int[] for _ in 1:n]
    domination_count = zeros(Int, n)
    for left in 1:n
        for right in (left + 1):n
            if dominates(points[left], points[right])
                push!(dominated[left], right)
                domination_count[right] += 1
            elseif dominates(points[right], points[left])
                push!(dominated[right], left)
                domination_count[left] += 1
            end
        end
    end

    ranks = zeros(Int, n)
    fronts = Vector{Vector{Int}}()
    current = Int[]
    for index in 1:n
        domination_count[index] == 0 || continue
        ranks[index] = 1
        push!(current, index)
    end
    while !isempty(current)
        sort!(current)
        push!(fronts, current)
        next_front = Int[]
        for left in current
            for right in dominated[left]
                domination_count[right] -= 1
                domination_count[right] == 0 || continue
                ranks[right] = ranks[left] + 1
                push!(next_front, right)
            end
        end
        current = next_front
    end

    crowding = zeros(Float64, n)
    for front in fronts
        isempty(front) && continue
        for objective in 1:2
            ordered = sort(copy(front); by=index -> (points[index][objective], index))
            crowding[ordered[1]] = Inf
            crowding[ordered[end]] = Inf
            length(ordered) > 2 || continue
            low = points[ordered[1]][objective]
            high = points[ordered[end]][objective]
            span = high - low
            isfinite(span) || throw(ArgumentError(
                "NSGA-II objective span overflows Float64"))
            span > 0 || continue
            for position in 2:(length(ordered) - 1)
                previous = points[ordered[position - 1]][objective]
                following = points[ordered[position + 1]][objective]
                gap = following - previous
                isfinite(gap) || throw(ArgumentError(
                    "NSGA-II crowding gap overflows Float64"))
                index = ordered[position]
                # DEAP's two-objective implementation divides each
                # coordinate's contribution by the number of objectives.
                isinf(crowding[index]) || (crowding[index] += gap / span / 2.0)
            end
        end
    end

    selected = Int[]
    for front in fronts
        remaining = capacity_int - length(selected)
        remaining > 0 || break
        if length(front) <= remaining
            append!(selected, front)
            continue
        end
        ordered = sort(copy(front); by=index -> (-crowding[index], index))
        append!(selected, ordered[1:remaining])
        break
    end
    return (; indices=selected, ranks, crowding, fronts)
end

"""
    _typed_nsga2_tournament(rng, indices, ranks, crowding)

Choose one candidate by a binary NSGA-II rank/crowding tournament.  Two
members are sampled independently with replacement from `indices`; lower
nondomination rank wins, then larger crowding distance wins.  An exact tie is
resolved by one uniform draw from `rng`, including when the two sampled
indices happen to be equal.  This is the canonical binary rank/crowding
variant used here, not DEAP's four-way `selTournamentDCD` grouping.
"""
function _typed_nsga2_tournament(rng::AbstractRNG,
        indices::AbstractVector{<:Integer}, ranks::AbstractVector{<:Integer},
        crowding::AbstractVector{<:Real})
    isempty(indices) && throw(ArgumentError(
        "NSGA-II tournament requires at least one candidate"))
    length(ranks) == length(crowding) || throw(ArgumentError(
        "NSGA-II ranks and crowding lengths must match"))
    left = Int(indices[rand(rng, eachindex(indices))])
    right = Int(indices[rand(rng, eachindex(indices))])
    for index in (left, right)
        1 <= index <= length(ranks) || throw(ArgumentError(
            "NSGA-II candidate index is out of bounds"))
        value = Float64(crowding[index])
        isnan(value) && throw(ArgumentError(
            "NSGA-II crowding distances must not contain NaN"))
    end
    ranks[left] < ranks[right] && return left
    ranks[right] < ranks[left] && return right
    crowding[left] > crowding[right] && return left
    crowding[right] > crowding[left] && return right
    return rand(rng, Bool) ? left : right
end

@testitem "TypedSearch NSGA-II fronts and crowding are stable" begin
    using Test
    using LambdaRegression.TypedSearch: _typed_nsga2_select

    fitness = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0),
        (2.0, 5.0), (5.0, 3.0), (6.0, 6.0)]
    original = copy(fitness)
    selected = _typed_nsga2_select(fitness, 3)
    @test fitness == original
    @test selected.ranks == [1, 1, 1, 1, 2, 2, 3]
    @test selected.fronts == [[1, 2, 3, 4], [5, 6], [7]]
    @test isinf(selected.crowding[1])
    @test isinf(selected.crowding[4])
    @test selected.crowding[2] ≈ 2 / 3
    @test selected.crowding[3] ≈ 2 / 3
    @test selected.indices == [1, 4, 2]

    complete = _typed_nsga2_select(fitness, length(fitness))
    @test complete.indices == [1, 2, 3, 4, 5, 6, 7]
    @test _typed_nsga2_select(fitness, 0).indices == Int[]

    duplicate = _typed_nsga2_select([(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)], 2)
    @test duplicate.ranks == [1, 1, 2]
    @test duplicate.fronts[1] == [1, 2]
    @test duplicate.indices == [1, 2]

    flat = _typed_nsga2_select([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)], 2)
    @test flat.indices == [1, 3]
    @test all(isinf, flat.crowding[[1, 3]])

    @test_throws ArgumentError _typed_nsga2_select([(0.0, NaN)], 1)
    @test_throws ArgumentError _typed_nsga2_select([(Inf, 1.0)], 1)
    @test_throws ArgumentError _typed_nsga2_select(fitness, -1)
    @test_throws ArgumentError _typed_nsga2_select(fitness, length(fitness) + 1)
end

@testitem "TypedSearch NSGA-II binary tournament uses rank, crowding, and RNG ties" begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.TypedSearch: _typed_nsga2_tournament

    indices = [1, 2, 3]
    ranks = [1, 1, 2]
    crowding = [0.0, Inf, 1.0]
    function expected(seed, candidates, candidate_ranks, candidate_crowding)
        rng = MersenneTwister(seed)
        left = Int(candidates[rand(rng, eachindex(candidates))])
        right = Int(candidates[rand(rng, eachindex(candidates))])
        if candidate_ranks[left] < candidate_ranks[right]
            return left
        elseif candidate_ranks[right] < candidate_ranks[left]
            return right
        elseif candidate_crowding[left] > candidate_crowding[right]
            return left
        elseif candidate_crowding[right] > candidate_crowding[left]
            return right
        end
        return rand(rng, Bool) ? left : right
    end

    let saw_rank = false, saw_crowding = false
        for seed in 1:128
            probe_rng = MersenneTwister(seed)
            left = Int(indices[rand(probe_rng, eachindex(indices))])
            right = Int(indices[rand(probe_rng, eachindex(indices))])
            left == right && continue
            saw_rank |= ranks[left] != ranks[right]
            saw_crowding |= ranks[left] == ranks[right] &&
                crowding[left] != crowding[right]
            @test _typed_nsga2_tournament(MersenneTwister(seed), indices,
                ranks, crowding) == expected(seed, indices, ranks, crowding)
        end
        @test saw_rank
        @test saw_crowding
    end

    tie_indices = [1, 2]
    tie_ranks = [1, 1]
    tie_crowding = [0.0, 0.0]
    tie_rng = MersenneTwister(1234)
    ties = [_typed_nsga2_tournament(tie_rng, tie_indices, tie_ranks,
        tie_crowding) for _ in 1:64]
    @test 1 in ties
    @test 2 in ties
    tie_seed = 901
    @test _typed_nsga2_tournament(MersenneTwister(tie_seed), tie_indices,
        tie_ranks, tie_crowding) == expected(tie_seed, tie_indices,
            tie_ranks, tie_crowding)
    @test_throws ArgumentError _typed_nsga2_tournament(
        MersenneTwister(8000), Int[], tie_ranks, tie_crowding)
    @test_throws ArgumentError _typed_nsga2_tournament(
        MersenneTwister(42), [1], [1], [NaN])
end
