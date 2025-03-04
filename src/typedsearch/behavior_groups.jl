"""Pure empirical-output partitioning and bounded selection consumers.

`classify_outputs` only partitions candidate indices.  It does not know about
fitness, champions, quotas, archives, or random numbers.  Selection consumers
below layer the existing raw-term deduplication, champion rule, and soft family
cap on that partition.  The default classifier compares the exact Float64 bit
pattern, including signed zero; hash buckets are always checked by full key
equality so a hash collision cannot merge observations.
"""

using Random: AbstractRNG, rand, randperm

"""Callable exact Float64 classifier preserving every output bit."""
struct ExactOutputClassifier end

@inline function (::ExactOutputClassifier)(outputs::AbstractVector{Float64})
    # A vector avoids large variable-length Tuple specialisations for batch
    # outputs while retaining exact signed-zero distinctions.
    return UInt64[reinterpret(UInt64, value) for value in outputs]
end

"""Validate one already-owned objective output vector."""
function _typed_validate_output_vector(output::AbstractVector; copy_output::Bool=true)
    isempty(output) && return Err(TypedSearchError(:empty_behavior_vector))
    if output isa Vector{Float64}
        for (index, number) in enumerate(output)
            isfinite(number) || return Err(TypedSearchError(
                :nonfinite_behavior_output, (index, number)))
        end
        return copy_output ? copy(output) : output
    end
    converted = Vector{Float64}(undef, length(output))
    for (index, value) in enumerate(output)
        value isa Real || throw(ArgumentError(
            "behaviour outputs must contain real values"))
        number = Float64(value)
        isfinite(number) || return Err(TypedSearchError(
            :nonfinite_behavior_output, (index, number)))
        converted[index] = number
    end
    return converted
end

function (::ExactOutputClassifier)(outputs::AbstractVector)
    checked = _typed_validate_output_vector(outputs; copy_output=false)
    checked isa Err && throw(ArgumentError("exact classification requires finite values"))
    return UInt64[reinterpret(UInt64, value) for value in checked]
end

"""Partition candidate output vectors by a caller-supplied classifier.

The classifier is called once per candidate output vector.  Keys are compared with `isequal`
inside their hash buckets, making the returned partition correct even when a
custom classifier has colliding hashes.  The returned vectors contain the
original candidate indices and no output or key is mutated.  A custom
classifier should be deterministic, read-only, and return keys whose `hash`
and `isequal` behavior remains stable for the duration of this call.
"""
function classify_outputs(classifier, outputs::AbstractVector)
    groups = Vector{Vector{Int}}()
    keys = Any[]
    buckets = Dict{UInt64,Vector{Int}}()
    for index in eachindex(outputs)
        output = outputs[index]
        output isa AbstractVector || throw(ArgumentError(
            "classified outputs must be vectors"))
        checked = _typed_validate_output_vector(output; copy_output=false)
        checked isa Err && throw(ArgumentError(
            "classified outputs must contain finite values"))
        key = classifier(checked)
        hash_value = UInt64(hash(key))
        group_index = 0
        for candidate in get(buckets, hash_value, Int[])
            isequal(keys[candidate], key) || continue
            group_index = candidate
            break
        end
        if group_index == 0
            push!(keys, key)
            push!(groups, [Int(index)])
            group_index = length(groups)
            push!(get!(buckets, hash_value, Int[]), group_index)
        else
            push!(groups[group_index], Int(index))
        end
    end
    return groups
end

"""One bounded behavioural family over indices in a candidate pool."""
struct _TypedBehaviorGroup
    member_indices::Vector{Int}
    champion_index::Int
end

"""Cached group-level NSGA rank/crowding and raw-member mapping."""
struct _TypedBehaviorSelection
    groups::Vector{_TypedBehaviorGroup}
    member_to_group::Vector{Int}
    ranks::Vector{Int}
    crowding::Vector{Float64}
end

function _typed_behavior_member_key(member::ScoredTerm, language::Language)
    return (member.loss, _typed_written_size(member.term, language), member.birth)
end

function _typed_behavior_better(left::ScoredTerm, right::ScoredTerm,
                                language::Language)
    return _typed_behavior_member_key(left, language) <
        _typed_behavior_member_key(right, language)
end

"""Build capped groups while deduplicating repeated raw terms.

`outputs` is owned by the caller and is never mutated.  Malformed sidecar
shapes are programmer errors; the objective boundary is responsible for
turning expected invalid output values into `Err` before this helper runs.
"""
function _typed_behavior_groups(pool::AbstractVector{ScoredTerm},
                                outputs::AbstractVector{<:AbstractVector},
                                language::Language, family_capacity::Integer;
                                rng::Union{Nothing,AbstractRNG}=nothing,
                                classifier=ExactOutputClassifier())
    family_capacity >= 1 || throw(ArgumentError("family capacity must be positive"))
    length(pool) == length(outputs) || throw(ArgumentError(
        "behaviour sidecar length must match candidate pool"))
    # Classification is complete and independent of selection policy.  The
    # raw-term pass below then chooses one representative per syntax record,
    # preserving the historical deduplication rule before applying the cap.
    partitions = classify_outputs(classifier, outputs)
    representatives = Int[]
    raw_positions = Dict{Int,Int}()
    # Hash first, then verify the full node vector.  Raw syntax deduplication
    # is selection policy; it is deliberately outside pure classification.
    raw_buckets = Dict{UInt64,Vector{Int}}()
    for index in eachindex(pool)
        vector = outputs[index]
        vector isa Vector{Float64} || throw(ArgumentError(
            "behaviour sidecar vectors must be Vector{Float64}"))
        raw_hash = UInt64(hash(pool[index].term.nodes))
        existing = 0
        for candidate in get(raw_buckets, raw_hash, Int[])
            pool[candidate].term.nodes == pool[index].term.nodes || continue
            existing = candidate
            break
        end
        if existing == 0
            push!(representatives, index)
            raw_positions[index] = length(representatives)
            push!(get!(raw_buckets, raw_hash, Int[]), index)
        elseif _typed_behavior_better(pool[index], pool[existing], language)
            position = get(raw_positions, existing, 0)
            position > 0 || throw(ArgumentError(
                "behaviour raw representative bookkeeping is inconsistent"))
            representatives[position] = index
            delete!(raw_positions, existing)
            raw_positions[index] = position
            bucket = get(raw_buckets, raw_hash, Int[])
            replace!(bucket, existing => index)
        end
    end

    representative_positions = Dict(index => position
        for (position, index) in enumerate(representatives))
    groups = _TypedBehaviorGroup[]
    member_to_group = zeros(Int, length(pool))
    for partition in partitions
        members = [index for index in partition if haskey(representative_positions, index)]
        isempty(members) && continue
        champion = members[1]
        for index in members[2:end]
            _typed_behavior_better(pool[index], pool[champion], language) &&
                (champion = index)
        end
        push!(groups, _TypedBehaviorGroup(members, champion))
    end
    # Select the champion and up to k-1 uniformly sampled raw alternatives once
    # all members of every output group are known.  This avoids insertion-order
    # bias while keeping the retained representation bounded.
    for (group_index, group) in enumerate(groups)
        members = group.member_indices
        champion = group.champion_index
        alternatives = [candidate for candidate in members if candidate != champion]
        if length(members) > family_capacity
            selected = if family_capacity == 1
                Int[]
            elseif rng === nothing
                alternatives[1:(Int(family_capacity) - 1)]
            else
                alternatives[randperm(rng, length(alternatives))[
                    1:(Int(family_capacity) - 1)]]
            end
            members = vcat([champion], selected)
        end
        groups[group_index] = _TypedBehaviorGroup(members, champion)
    end
    # Recompute this map from the final bounded members rather than retaining
    # stale ids.
    fill!(member_to_group, 0)
    for (group_index, group) in enumerate(groups)
        for index in group.member_indices
            member_to_group[index] = group_index
        end
    end
    return groups, member_to_group
end

"""Build one group-level rank/crowding cache for an unchanged selection pool."""
function _typed_behavior_selection(pool::AbstractVector{ScoredTerm},
                                   outputs::AbstractVector{<:AbstractVector},
                                   language::Language, family_capacity::Integer;
                                   rng::Union{Nothing,AbstractRNG}=nothing,
                                   classifier=ExactOutputClassifier())
    groups, member_to_group = _typed_behavior_groups(pool, outputs, language,
        family_capacity; rng, classifier)
    fitness = Tuple{Float64,Float64}[(pool[group.champion_index].loss,
        Float64(_typed_written_size(pool[group.champion_index].term, language)))
        for group in groups]
    selected = _typed_nsga2_select(fitness, length(groups))
    return _TypedBehaviorSelection(groups, member_to_group, selected.ranks,
        selected.crowding)
end

"""Choose a raw representative after selecting its family by cached NSGA data."""
function _typed_behavior_parent(rng::AbstractRNG,
                                selection::_TypedBehaviorSelection)
    isempty(selection.groups) && throw(ArgumentError(
        "behaviour selection requires at least one group"))
    group_index = _typed_nsga2_tournament(rng, eachindex(selection.groups),
        selection.ranks, selection.crowding)
    members = selection.groups[group_index].member_indices
    return length(members) == 1 ? members[1] : members[rand(rng, eachindex(members))]
end

"""Prefilter capped families, then apply ordinary NSGA-II survivor selection.

The family-level rank/crowding is computed once for this pool for mating; the
survivor ranking below uses each retained member's own loss and written size.
If the capped unique-raw pool is smaller than `capacity`, the soft quota is
relaxed first with unused distinct raw terms, then with raw duplicates only as
needed to preserve the fixed population contract.
"""
function _typed_behavior_survive(pool::Vector{ScoredTerm},
                                 outputs::Vector{Vector{Float64}}, capacity::Int,
                                 language::Language, family_capacity::Int,
                                 rng::AbstractRNG;
                                 classifier=ExactOutputClassifier())
    # Families only prefilter a bounded representative set.  Actual survivor
    # ranking remains ordinary NSGA-II over each retained member's own loss
    # and written complexity; the champion never lends its score to a peer.
    selection = _typed_behavior_selection(pool, outputs, language, family_capacity;
        rng, classifier)
    retained = Int[]
    for group in selection.groups
        append!(retained, group.member_indices)
    end
    if length(retained) < capacity
        # Relax the soft quota using unused distinct raw alternatives first.
        retained_raw = [pool[index].term.nodes for index in retained]
        unused_distinct = Int[]
        for index in eachindex(pool)
            any(raw -> raw == pool[index].term.nodes, retained_raw) && continue
            push!(unused_distinct, index)
        end
        sort!(unused_distinct; by=index ->
            _typed_behavior_member_key(pool[index], language))
        for index in unused_distinct
            any(raw -> raw == pool[index].term.nodes, retained_raw) && continue
            push!(retained, index)
            push!(retained_raw, pool[index].term.nodes)
            length(retained) >= capacity && break
        end
        # If even the distinct raw pool is too small, admit duplicate records
        # in bounded pool order to keep the survivor vector fixed-size.
        if length(retained) < capacity
            retained_set = Set(retained)
            for index in eachindex(pool)
                index in retained_set && continue
                push!(retained, index)
                push!(retained_set, index)
                length(retained) >= capacity && break
            end
        end
    end
    selected = Int[]
    ranked_result = nothing
    if !isempty(retained)
        fitness = [(pool[index].loss,
            Float64(_typed_written_size(pool[index].term, language)))
            for index in retained]
        ranked_result = _typed_nsga2_select(fitness,
            min(capacity, length(retained)))
        selected = [retained[position] for position in ranked_result.indices]
    end
    if length(selected) < capacity
        selected_set = Set(selected)
        for index in eachindex(pool)
            index in selected_set && continue
            push!(selected, index)
            length(selected) == capacity && break
        end
    end
    member_ranks = ranked_result === nothing ? Int[] :
        ranked_result.ranks[ranked_result.indices]
    member_crowding = ranked_result === nothing ? Float64[] :
        ranked_result.crowding[ranked_result.indices]
    return (; population=[pool[index] for index in selected],
        outputs=[outputs[index] for index in selected],
        ranks=member_ranks, crowding=member_crowding, selection, selected)
end

"""Group occupied archive representatives for uniform material sampling."""
function _typed_behavior_archive(archive, evaluation_state::EvaluationState,
                                language::Language, family_capacity::Int,
                                rng::AbstractRNG; classifier=ExactOutputClassifier())
    members = ScoredTerm[]
    outputs = Vector{Vector{Float64}}()
    for size in sort!(collect(keys(archive)))
        member = archive[size]
        vector = _typed_evaluation_borrow(evaluation_state, member)
        vector isa Vector{Float64} || throw(ArgumentError(
            "behaviour archive is missing a scored output vector"))
        push!(members, member)
        push!(outputs, vector)
    end
    isempty(members) && return nothing
    groups, _ = _typed_behavior_groups(members, outputs, language,
        family_capacity; rng, classifier)
    return (; members, groups)
end

@testitem "TypedSearch empirical behaviour grouping is exact and bounded" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: ScoredTerm, _typed_behavior_groups,
        _typed_behavior_survive
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    one = add_leaf!(language, scalar, :one)
    zero = add_leaf!(language, scalar, :zero)
    one_term = TypedTerm(TermNode[leaf_node(scalar, one)])
    zero_term = TypedTerm(TermNode[leaf_node(scalar, zero)])
    pool = [ScoredTerm(one_term, 1.0, 1), ScoredTerm(zero_term, 1.0, 2),
        ScoredTerm(TypedTerm(copy(one_term.nodes)), 0.5, 3)]
    outputs = [[1.0, 2.0], [1.0, 2.0], [9.0, 9.0]]
    groups, _ = _typed_behavior_groups(pool, outputs, language, 2)
    @test length(groups) == 2
    @test sort([length(group.member_indices) for group in groups]) == [1, 1]
    # Equal losses with different finite predictions are separate families;
    # equal raw terms are deduplicated even when a later score is better.
    survivor = _typed_behavior_survive(pool, outputs, 2, language, 2,
        MersenneTwister(42))
    @test length(survivor.population) == 2
    @test length(unique(Tuple(member.term.nodes) for member in survivor.population)) == 2
end

@testitem "TypedSearch behaviour soft quota checks current raw set" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: ScoredTerm, _typed_behavior_survive
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaves = [add_leaf!(language, scalar, name) for name in (:a, :b, :c, :d)]
    terms = [TypedTerm(TermNode[leaf_node(scalar, leaf)]) for leaf in leaves]
    # c appears twice in the pool.  All representatives share one behaviour
    # family, so family_capacity=1 leaves the soft quota to refill from raw
    # terms.  A second c record must not occupy the slot needed by d.
    pool = [ScoredTerm(terms[1], 0.0, 1), ScoredTerm(terms[3], 1.0, 2),
        ScoredTerm(TypedTerm(copy(terms[3].nodes)), 2.0, 3),
        ScoredTerm(terms[2], 3.0, 4), ScoredTerm(terms[4], 4.0, 5)]
    outputs = [[1.0, 2.0] for _ in pool]
    survivor = _typed_behavior_survive(pool, outputs, 3, language, 1,
        MersenneTwister(2000))
    @test length(survivor.population) == 3
    @test length(unique(Tuple(member.term.nodes) for member in survivor.population)) == 3

    # When no distinct raw term exists, the fixed-size contract still admits
    # duplicates after the distinct refill has been exhausted.
    duplicate_pool = [pool[1], pool[1],
        ScoredTerm(TypedTerm(copy(pool[1].term.nodes)), 1.0, 6)]
    duplicate_outputs = [[1.0, 2.0] for _ in duplicate_pool]
    duplicate_survivor = _typed_behavior_survive(duplicate_pool,
        duplicate_outputs, 3, language, 1, MersenneTwister(555))
    @test length(duplicate_survivor.population) == 3
    @test length(unique(Tuple(member.term.nodes) for member in duplicate_survivor.population)) == 1
end

@testitem "TypedSearch output classification is pure and collision-checked" begin
    using Test
    using LambdaRegression.TypedSearch: ExactOutputClassifier, classify_outputs

    struct CollisionKey
        value::Int
    end
    Base.hash(::CollisionKey, ::UInt) = UInt(0)
    Base.isequal(left::CollisionKey, right::CollisionKey) =
        left.value == right.value

    exact = ExactOutputClassifier()
    @test !isequal(exact([0.0, 1.0]), exact([-0.0, 1.0]))
    outputs = [[1.0], [2.0], [1.0]]
    groups = classify_outputs(x -> CollisionKey(Int(x[1])), outputs)
    @test sort(groups; by=first) == [[1, 3], [2]]
    @test classify_outputs(x -> CollisionKey(Int(x[1]) % 2), outputs) ==
        [[1, 3], [2]]
    @test_throws ArgumentError classify_outputs(exact, [[Inf]])
end
