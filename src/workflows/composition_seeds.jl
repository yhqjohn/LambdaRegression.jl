using ..Transforms: chain_programs

"""
    select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(), count=8, max_candidates=64,
                       strategy=:ranked, score=nothing, rng=default_rng())

Build at most `max_candidates` complete programs by chaining one program from
each source pool. Pools are supplied best-first; this order matters to `:ranked`
but not to `:random`. `:random` samples candidate index tuples uniformly
without replacement; `:ranked` visits tuples by increasing sum of zero-based
source ranks, with lexicographic ties; `:scored` samples the same way as
`:random`, evaluates each successful complete composition with `score`, and
returns the lowest finite scores (then shorter trees, then sampling order).

The callback receives the complete composed `Program`; callers are responsible
for scoring it on the intended task data. Source ranks are local to their pool:
the selector never compares source-pool scores. Composition refusals and
invalid scores are counted and skipped. Unexpected callback or syntax errors
propagate. Inputs are read-only and are never reordered or modified.

Returns a named tuple with `programs`, aligned `source_indices`, `scores`
(`nothing` except for `:scored`), and `diagnostics` containing the exact
Cartesian-product size plus attempted, failed, scored, invalid, and selected
counts. The Cartesian product is never materialized.
"""
function select_chain_seeds(
    pools...;
    context=Dict{Symbol,TyLike}(),
    count=8,
    max_candidates=64,
    strategy=:ranked,
    score=nothing,
    rng::AbstractRNG=default_rng(),
)
    length(pools) >= 2 || throw(ArgumentError("at least two source pools are required"))
    strategy isa Symbol || throw(ArgumentError("strategy must be a Symbol"))
    strategy in (:random, :ranked, :scored) ||
        throw(ArgumentError("strategy must be :random, :ranked, or :scored"))
    requested = _chain_seed_nonnegative_int(:count, count)
    limit = _chain_seed_nonnegative_int(:max_candidates, max_candidates)
    if strategy === :scored
        score === nothing && throw(ArgumentError("strategy=:scored requires score"))
    elseif score !== nothing
        throw(ArgumentError("score is only used with strategy=:scored"))
    end

    source_pools = Vector{Vector{Program}}()
    sizehint!(source_pools, length(pools))
    for (pool_index, pool) in enumerate(pools)
        pool isa AbstractVector || throw(ArgumentError("source pool $pool_index must be an AbstractVector"))
        isempty(pool) && throw(ArgumentError("source pool $pool_index must not be empty"))
        copied_pool = Program[]
        sizehint!(copied_pool, length(pool))
        for (source_index, program) in enumerate(pool)
            program isa Program || throw(ArgumentError(
                "source pool $pool_index entry $source_index is not a Program"))
            push!(copied_pool, program)
        end
        push!(source_pools, copied_pool)
    end

    pool_lengths = Int[length(pool) for pool in source_pools]
    theoretical_product = foldl((product, n) -> product * BigInt(n), pool_lengths;
        init=BigInt(1))
    candidate_limit = Int(min(BigInt(limit), theoretical_product))
    tuple_limit = requested == 0 ? 0 : candidate_limit
    candidate_tuples = if strategy === :ranked
        _chain_seed_ranked_tuples(pool_lengths, tuple_limit)
    else
        _chain_seed_random_tuples(rng, pool_lengths, theoretical_product, tuple_limit)
    end

    programs = Program[]
    source_indices = Tuple[]
    selected_scores = Real[]
    attempted = 0
    failed = 0
    scored = 0
    invalid = 0

    if strategy === :scored
        scored_candidates = NamedTuple[]
        for (ordinal, indices) in enumerate(candidate_tuples)
            attempted += 1
            composed = chain_programs(
                (source_pools[pool_index][indices[pool_index]]
                    for pool_index in eachindex(source_pools))...;
                context,
            )
            if composed isa Err
                failed += 1
                continue
            elseif !(composed isa Program)
                throw(ArgumentError(
                    "Transforms.chain_programs returned $(typeof(composed)); expected Program or Err"))
            end
            scored += 1
            raw_score = score(composed)
            if raw_score isa Err
                invalid += 1
                continue
            elseif !(raw_score isa Real)
                throw(ArgumentError("score callback must return a real number or Err"))
            end
            isfinite(raw_score) || (invalid += 1; continue)
            push!(scored_candidates, (; program=composed, indices,
                score=raw_score, complexity=length(composed.tree), ordinal))
        end
        sort!(scored_candidates; by=candidate ->
            (candidate.score, candidate.complexity, candidate.ordinal))
        for candidate in Iterators.take(scored_candidates, requested)
            push!(programs, candidate.program)
            push!(source_indices, candidate.indices)
            push!(selected_scores, candidate.score)
        end
    else
        for indices in candidate_tuples
            length(programs) >= requested && break
            attempted += 1
            composed = chain_programs(
                (source_pools[pool_index][indices[pool_index]]
                    for pool_index in eachindex(source_pools))...;
                context,
            )
            if composed isa Err
                failed += 1
            elseif composed isa Program
                push!(programs, composed)
                push!(source_indices, indices)
            else
                throw(ArgumentError(
                    "Transforms.chain_programs returned $(typeof(composed)); expected Program or Err"))
            end
        end
    end

    return (
        programs,
        source_indices,
        scores=strategy === :scored ? selected_scores : nothing,
        diagnostics=(;
            theoretical_product,
            attempted,
            failed,
            scored,
            invalid,
            selected=length(programs),
        ),
    )
end

function _chain_seed_nonnegative_int(name::Symbol, value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$name must be a non-negative integer"))
    value >= 0 || throw(ArgumentError("$name must be a non-negative integer"))
    value <= typemax(Int) || throw(ArgumentError("$name exceeds the supported Int range"))
    return Int(value)
end

function _chain_seed_random_tuples(
    rng::AbstractRNG,
    lengths::Vector{Int},
    population_size::BigInt,
    sample_size::Int,
)
    result = Tuple[]
    sizehint!(result, sample_size)
    swapped = Dict{BigInt,BigInt}()
    for offset in 0:(sample_size - 1)
        position = BigInt(offset)
        chosen_position = position + _chain_seed_rand_below(rng, population_size - position)
        chosen_rank = get(swapped, chosen_position, chosen_position)
        swapped[chosen_position] = get(swapped, position, position)
        push!(result, _chain_seed_decode_rank(chosen_rank, lengths))
    end
    return result
end

function _chain_seed_rand_below(rng::AbstractRNG, limit::BigInt)
    limit > 0 || throw(ArgumentError("random sampling requires a positive range"))
    limit == 1 && return BigInt(0)
    bit_count = ndigits(limit - 1; base=2)
    while true
        value = BigInt(0)
        remaining = bit_count
        while remaining > 0
            chunk_bits = min(64, remaining)
            word = rand(rng, UInt64)
            mask = chunk_bits == 64 ? typemax(UInt64) : (UInt64(1) << chunk_bits) - UInt64(1)
            value = (value << chunk_bits) | BigInt(word & mask)
            remaining -= chunk_bits
        end
        value < limit && return value
    end
end

function _chain_seed_decode_rank(rank::BigInt, lengths::Vector{Int})
    indices = Vector{Int}(undef, length(lengths))
    remainder = rank
    for dimension in length(lengths):-1:1
        radix = BigInt(lengths[dimension])
        indices[dimension] = Int(mod(remainder, radix)) + 1
        remainder = div(remainder, radix)
    end
    return Tuple(indices)
end

function _chain_seed_ranked_tuples(lengths::Vector{Int}, limit::Int)
    heap = Vector{Vector{Int}}()
    visited = Set{Tuple}()
    zero_ranks = zeros(Int, length(lengths))
    push!(heap, zero_ranks)
    push!(visited, Tuple(zero_ranks))
    result = Tuple[]
    sizehint!(result, limit)
    while !isempty(heap) && length(result) < limit
        ranks = _chain_seed_heap_pop!(heap)
        push!(result, Tuple(index + 1 for index in ranks))
        for dimension in eachindex(lengths)
            ranks[dimension] + 1 < lengths[dimension] || continue
            neighbor = copy(ranks)
            neighbor[dimension] += 1
            key = Tuple(neighbor)
            key in visited && continue
            push!(visited, key)
            _chain_seed_heap_push!(heap, neighbor)
        end
    end
    return result
end

_chain_seed_rank_less(left::Vector{Int}, right::Vector{Int}) =
    isless((sum(BigInt, left), Tuple(left)), (sum(BigInt, right), Tuple(right)))

function _chain_seed_heap_push!(heap::Vector{Vector{Int}}, value::Vector{Int})
    push!(heap, value)
    child = length(heap)
    while child > 1
        parent = child >>> 1
        _chain_seed_rank_less(heap[child], heap[parent]) || break
        heap[child], heap[parent] = heap[parent], heap[child]
        child = parent
    end
    return heap
end

function _chain_seed_heap_pop!(heap::Vector{Vector{Int}})
    root = first(heap)
    tail = pop!(heap)
    isempty(heap) && return root
    heap[1] = tail
    parent = 1
    while true
        left = parent << 1
        left > length(heap) && break
        right = left + 1
        child = right <= length(heap) && _chain_seed_rank_less(heap[right], heap[left]) ? right : left
        _chain_seed_rank_less(heap[child], heap[parent]) || break
        heap[parent], heap[child] = heap[child], heap[parent]
        parent = child
    end
    return root
end

@testitem "Workflow chain-seed ranked frontier preserves source rank order" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Workflows: select_chain_seeds

    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    pools = ([identity, identity], [identity, identity], [identity, identity])
    result = select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(),
        count=8, max_candidates=8)

    @test result.source_indices == [(1, 1, 1), (1, 1, 2), (1, 2, 1),
        (2, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
    @test result.diagnostics.theoretical_product == 8
    @test result.diagnostics.attempted == 8
    @test result.diagnostics.failed == 0
    @test result.diagnostics.selected == 8
end

@testitem "Workflow chain-seed random tuples are reproducible and scored on full chains" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Utils: Err
    using LambdaRegression.Workflows: select_chain_seeds

    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    pools = ([identity, identity, identity], [identity, identity, identity],
             [identity, identity])
    random_a = select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(),
        strategy=:random, count=6, max_candidates=6, rng=MersenneTwister(1111))
    random_b = select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(),
        strategy=:random, count=6, max_candidates=6, rng=MersenneTwister(1111))
    score_calls = Ref(0)
    scored = select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(),
        strategy=:scored, count=6, max_candidates=6, rng=MersenneTwister(1111),
        score=program -> begin
            @test program isa Program
            @test length(program.tree) > length(identity.tree)
            score_calls[] += 1
            score_calls[] == 1 && return -2.0
            score_calls[] == 2 && return NaN
            score_calls[] == 3 && return Err(ArgumentError("invalid score"))
            return -4.0
        end)

    @test random_a.source_indices == random_b.source_indices
    @test length(unique(random_a.source_indices)) == 6
    @test scored.diagnostics.theoretical_product == 18
    @test scored.diagnostics.attempted == 6
    @test scored.diagnostics.scored == 6
    @test scored.diagnostics.invalid == 2
    @test scored.source_indices == random_a.source_indices[[4, 5, 6, 1]]
    @test scored.scores == [-4.0, -4.0, -4.0, -2.0]
end

@testitem "Workflow chain-seed sampling stays bounded for huge products" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Workflows: select_chain_seeds

    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    pools = ntuple(_ -> [identity, identity], 70)
    result = select_chain_seeds(pools...; context=Dict{Symbol,TyLike}(),
        strategy=:random, count=3, max_candidates=3, rng=MersenneTwister(555))

    @test result.diagnostics.theoretical_product == big(2)^70
    @test result.diagnostics.attempted == 3
    @test result.diagnostics.selected == 3
    @test length(unique(result.source_indices)) == 3
end

@testitem "Workflow chain-seed scored ordering preserves exact real values" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.Workflows: select_chain_seeds

    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    pools = ([identity, identity], [identity, identity])
    seed = 41
    sampled = select_chain_seeds(pools...; strategy=:random, count=2, max_candidates=2,
        rng=MersenneTwister(seed))
    large = big(2)^100
    score_calls = Ref(0)
    scored = select_chain_seeds(pools...; strategy=:scored, count=2,
        max_candidates=2, rng=MersenneTwister(seed),
        score=_ -> begin
            score_calls[] += 1
            score_calls[] == 1 ? large + 1 : large
        end)

    @test scored.scores == [large, large + 1]
    @test scored.source_indices == reverse(sampled.source_indices)
    @test all(value -> value isa BigInt, scored.scores)
end

@testitem "Workflow chain-seed counts and type mismatches are transparent" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyFun, TyLike
    using LambdaRegression.Workflows: select_chain_seeds

    add = Program(AstTree(AbsNode(:x,
        AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:c))));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    logical_not = Program(AstTree(AbsNode(:x,
        AppNode(ConstNode(:not), IndexNode(0)))))
    context = Dict{Symbol,TyLike}(
        :add => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :not => TyFun(BoolTy, BoolTy),
    )
    source_tree_text = sprint(show, add.tree)
    source_values = copy(add.local_consts)
    source_types = copy(add.local_consts_ty)
    source_order = ([add], [logical_not])
    mismatch = select_chain_seeds([add], [logical_not]; context,
        count=1, max_candidates=1)
    none_requested = select_chain_seeds([add], [add]; context,
        count=0, max_candidates=2)

    @test isempty(mismatch.programs)
    @test mismatch.diagnostics == (theoretical_product=1, attempted=1,
        failed=1, scored=0, invalid=0, selected=0)
    @test isempty(none_requested.programs)
    @test none_requested.diagnostics.attempted == 0
    @test sprint(show, add.tree) == source_tree_text
    @test add.local_consts == source_values
    @test add.local_consts_ty == source_types
    @test source_order == ([add], [logical_not])
    @test_throws ArgumentError select_chain_seeds([add]; context)
    @test_throws ArgumentError select_chain_seeds(Program[], [add]; context)
    @test_throws ArgumentError select_chain_seeds([add], [add]; context, count=-1)
    @test_throws ArgumentError select_chain_seeds([add], [add]; context,
        strategy=:scored)
end
