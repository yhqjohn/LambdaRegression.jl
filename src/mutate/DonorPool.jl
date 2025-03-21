"""Bounded, feedback-aware donor-fragment sampling.

The pool stores a read-only source individual plus an explicit fragment path.
Callers admit fragments from already selected winners, sample with their own
RNG, and feed back the staged outcome. `recombine_from_pool` composes that
selection with authoritative directional retrieval:

```julia
pool = DonorPool(DonorPoolConfig(capacity=128, policy=:adaptive))
admit_donor!(pool, winner, path)
attempt = recombine_from_pool(op, recipient, pool;
    rng, global_consts_ty)
```

Uniform and adaptive policies change sampling and eviction only. They cannot
change the legal retrieval answer for a fixed donor fragment and target.
"""

using TestItems

"""Tunable bounded-pool policy.

Entries are anchored by an explicitly admitted source `Program` and AST
path. `policy` is `:uniform` or `:adaptive`; every adaptive entry retains the
positive `exploration_floor`. The configuration itself needs no typing context,
while `admit_random_donors!` and `recombine_from_pool` require the recipient
world's `global_consts_ty`. Invalid capacities, policies, rewards, decay, or
age penalties throw `ArgumentError`; an empty pool is reported by
`EmptyDonorPoolErr` at recombination.
"""
struct DonorPoolConfig
    capacity::Int
    policy::Symbol
    exploration_floor::Float64
    match_reward::Float64
    valid_reward::Float64
    change_reward::Float64
    gain_scale::Float64
    miss_decay::Float64
    age_penalty::Float64

    function DonorPoolConfig(
        capacity::Int,
        policy::Symbol,
        exploration_floor::Float64,
        match_reward::Float64,
        valid_reward::Float64,
        change_reward::Float64,
        gain_scale::Float64,
        miss_decay::Float64,
        age_penalty::Float64,
    )
        capacity >= 1 || throw(ArgumentError("capacity must be positive"))
        policy in (:uniform, :adaptive) ||
            throw(ArgumentError("policy must be :uniform or :adaptive"))
        isfinite(exploration_floor) && exploration_floor > 0 ||
            throw(ArgumentError("exploration_floor must be finite and positive"))
        all(isfinite, (match_reward, valid_reward, change_reward, gain_scale)) ||
            throw(ArgumentError("donor rewards must be finite"))
        match_reward >= 0 && valid_reward >= 0 && change_reward >= 0 &&
            gain_scale >= 0 || throw(ArgumentError("donor rewards must be nonnegative"))
        isfinite(miss_decay) && 0 < miss_decay <= 1 ||
            throw(ArgumentError("miss_decay must be in (0, 1]"))
        isfinite(age_penalty) && age_penalty >= 0 ||
            throw(ArgumentError("age_penalty must be finite and nonnegative"))
        return new(
            capacity,
            policy,
            exploration_floor,
            match_reward,
            valid_reward,
            change_reward,
            gain_scale,
            miss_decay,
            age_penalty,
        )
    end
end

DonorPoolConfig(;
    capacity::Int=128,
    policy::Symbol=:adaptive,
    exploration_floor::Real=0.1,
    match_reward::Real=0.25,
    valid_reward::Real=0.75,
    change_reward::Real=1.5,
    gain_scale::Real=0.5,
    miss_decay::Real=0.8,
    age_penalty::Real=0.002,
) = DonorPoolConfig(
    capacity,
    policy,
    Float64(exploration_floor),
    Float64(match_reward),
    Float64(valid_reward),
    Float64(change_reward),
    Float64(gain_scale),
    Float64(miss_decay),
    Float64(age_penalty),
)

"""One staged outcome; later stages imply every earlier stage."""
struct DonorPoolOutcome
    matched::Bool
    valid_child::Bool
    genuine_change::Bool
    offspring_gain::Float64

    function DonorPoolOutcome(
        matched::Bool,
        valid_child::Bool,
        genuine_change::Bool,
        offspring_gain::Real,
    )
        valid_child && !matched && throw(ArgumentError("valid_child requires matched"))
        genuine_change && !valid_child &&
            throw(ArgumentError("genuine_change requires valid_child"))
        isfinite(offspring_gain) ||
            throw(ArgumentError("offspring_gain must be finite"))
        return new(matched, valid_child, genuine_change, Float64(offspring_gain))
    end
end

DonorPoolOutcome(;
    matched::Bool=false,
    valid_child::Bool=false,
    genuine_change::Bool=false,
    offspring_gain::Real=0.0,
) = DonorPoolOutcome(matched, valid_child, genuine_change, offspring_gain)

mutable struct _DonorPoolEntry
    id::Int
    donor::Program
    donor_target::Ty
    path::Tuple{Vararg{Int}}
    source_fitness::Union{Nothing,Float64}
    admitted_at::Int
    last_used::Int
    score::Float64
    samples::Int
    matches::Int
    valid_children::Int
    genuine_changes::Int
    promotions::Int
    demotions::Int
end

"""Thread-safe bounded collection of source individuals and fragment paths."""
mutable struct DonorPool
    config::DonorPoolConfig
    entries::Vector{_DonorPoolEntry}
    next_id::Int
    clock::Int
    admitted::Int
    samples::Int
    matches::Int
    valid_children::Int
    genuine_changes::Int
    promotions::Int
    demotions::Int
    evictions::Int
    lock::ReentrantLock
end

DonorPool(config::DonorPoolConfig=DonorPoolConfig()) = DonorPool(
    config,
    _DonorPoolEntry[],
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    ReentrantLock(),
)

"""Immutable selection containing a detached snapshot of the sampled donor."""
struct DonorPoolSelection
    donor_id::Int
    donor::Program
    donor_target::Ty
    path::Tuple{Vararg{Int}}
    weight::Float64
end

struct EmptyDonorPoolErr end

"""Pool selection plus the directional recombination result."""
struct DonorPoolAttempt
    donor_id::Int
    path::Tuple{Vararg{Int}}
    result::Any
end

_entry_weight(config::DonorPoolConfig, entry::_DonorPoolEntry) =
    config.policy === :uniform ? 1.0 :
    config.exploration_floor + max(0.0, entry.score)

function _entry_eviction_key(pool::DonorPool, entry::_DonorPoolEntry)
    pool.config.policy === :uniform &&
        return (Float64(entry.admitted_at), entry.id)
    age = pool.clock - entry.admitted_at
    return (entry.score - pool.config.age_penalty * age, entry.id)
end

function _evict_one!(pool::DonorPool)
    isempty(pool.entries) && return nothing
    index = pool.config.policy === :uniform ?
        argmin([(entry.admitted_at, entry.id) for entry in pool.entries]) :
        argmin([_entry_eviction_key(pool, entry) for entry in pool.entries])
    removed = splice!(pool.entries, index)
    pool.evictions += 1
    return removed
end

"""Admit one explicit source fragment, evicting one low-priority entry if full."""
function admit_donor!(
    pool::DonorPool,
    donor::Program,
    path;
    donor_target::Ty,
    source_fitness::Union{Nothing,Real}=nothing,
)
    normalized_path = Tuple(Int.(collect(path)))
    _node_at_path_maybe(donor.tree, collect(normalized_path)) === nothing &&
        throw(ArgumentError("donor path does not resolve in the source tree"))
    fitness = source_fitness === nothing ? nothing : Float64(source_fitness)
    fitness === nothing || isfinite(fitness) ||
        throw(ArgumentError("source_fitness must be finite"))

    return lock(pool.lock) do
        pool.clock += 1
        length(pool.entries) >= pool.config.capacity && _evict_one!(pool)
        id = pool.next_id
        pool.next_id += 1
        push!(pool.entries, _DonorPoolEntry(
            id,
            donor,
            donor_target,
            normalized_path,
            fitness,
            pool.clock,
            pool.clock,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ))
        pool.admitted += 1
        return id
    end
end

"""Sample and admit eligible fragments from one already selected individual.

The individual is analyzed once under `global_consts_ty`; eligible sites are
sampled uniformly without replacement using caller-owned `rng`. Inference
failure returns `Err`, while an empty eligible set returns an empty id vector.
"""
function admit_random_donors!(
    pool::DonorPool,
    donor::Program;
    donor_target::Ty,
    rng::AbstractRNG,
    global_consts_ty::Dict{Symbol,TyLike},
    count::Int=1,
    min_nodes::Int=8,
    include_root::Bool=false,
    tyfactory_ctor=make_tyvar_factory,
)
    count >= 0 || throw(ArgumentError("count must be nonnegative"))
    min_nodes >= 1 || throw(ArgumentError("min_nodes must be positive"))
    context = merge(global_consts_ty, donor.local_consts_ty)
    snapshot = prepare_typed_analysis(
        donor,
        context;
        factory=tyfactory_ctor(),
        target_ty=donor_target,
    )
    snapshot isa Err && return snapshot
    eligible = filter(typed_analysis_site_ids(snapshot; include_root=include_root)) do id
        snapshot.index.subtree_end[id] - id >= min_nodes
    end
    isempty(eligible) && return Int[]
    order = randperm(rng, length(eligible))
    ids = Int[]
    for index in Iterators.take(order, min(count, length(order)))
        path = typed_analysis_path(snapshot, eligible[index])
        push!(ids, admit_donor!(pool, donor, path; donor_target))
    end
    return ids
end

"""Sample one donor with caller-owned RNG and update only sampling counters."""
function sample_donor(pool::DonorPool; rng::AbstractRNG)
    return lock(pool.lock) do
        isempty(pool.entries) && return nothing
        weights = Float64[_entry_weight(pool.config, entry) for entry in pool.entries]
        threshold = rand(rng) * sum(weights)
        cumulative = 0.0
        index = lastindex(pool.entries)
        for i in eachindex(pool.entries)
            cumulative += weights[i]
            if threshold <= cumulative
                index = i
                break
            end
        end
        entry = pool.entries[index]
        pool.clock += 1
        pool.samples += 1
        entry.samples += 1
        entry.last_used = pool.clock
        return DonorPoolSelection(
            entry.id,
            entry.donor,
            entry.donor_target,
            entry.path,
            weights[index],
        )
    end
end

"""Apply insertion feedback. Returns `false` when the sampled id was evicted."""
function record_outcome!(
    pool::DonorPool,
    donor_id::Integer,
    outcome::DonorPoolOutcome,
)
    return lock(pool.lock) do
        index = findfirst(entry -> entry.id == donor_id, pool.entries)
        index === nothing && return false
        entry = pool.entries[index]
        pool.clock += 1
        if outcome.matched
            pool.matches += 1
            entry.matches += 1
        end
        if outcome.valid_child
            pool.valid_children += 1
            entry.valid_children += 1
        end
        if outcome.genuine_change
            pool.genuine_changes += 1
            entry.genuine_changes += 1
        end

        if outcome.matched
            reward = outcome.genuine_change ? pool.config.change_reward :
                outcome.valid_child ? pool.config.valid_reward :
                pool.config.match_reward
            entry.score += reward +
                pool.config.gain_scale * max(0.0, outcome.offspring_gain)
            entry.promotions += 1
            pool.promotions += 1
        else
            entry.score *= pool.config.miss_decay
            entry.demotions += 1
            pool.demotions += 1
        end
        return true
    end
end

"""Return current positive sampling weights keyed by stable donor id."""
function donor_weights(pool::DonorPool)
    return lock(pool.lock) do
        Dict(entry.id => _entry_weight(pool.config, entry) for entry in pool.entries)
    end
end

"""Return an immutable aggregate snapshot for aggregate logging."""
function donor_pool_stats(pool::DonorPool)
    return lock(pool.lock) do
        return (
            policy=pool.config.policy,
            capacity=pool.config.capacity,
            size=length(pool.entries),
            entry_ids=sort!(Int[entry.id for entry in pool.entries]),
            admitted=pool.admitted,
            samples=pool.samples,
            matches=pool.matches,
            valid_children=pool.valid_children,
            genuine_changes=pool.genuine_changes,
            promotions=pool.promotions,
            demotions=pool.demotions,
            evictions=pool.evictions,
        )
    end
end

function _pool_outcome(result, matched::Bool)
    result isa Err || return DonorPoolOutcome(true, true, true, 0.0)
    return DonorPoolOutcome(matched, false, false, 0.0)
end

"""Sample one explicit fragment, run directional recombination, and feed back."""
function recombine_from_pool(
    op::TypedSubtreeSwapCrossover,
    recipient::Program,
    pool::DonorPool;
    recipient_target::Ty,
    rng::AbstractRNG=MersenneTwister(2000),
    trace_callback=nothing,
    kwargs...,
)
    selection = sample_donor(pool; rng)
    selection === nothing &&
        return DonorPoolAttempt(0, (), Err(EmptyDonorPoolErr()))
    matched = Ref(false)
    pool_trace = event -> begin
        event.event === :selected && (matched[] = true)
        trace_callback === nothing || trace_callback(event)
        return nothing
    end
    result = recombine(
        op,
        recipient,
        selection.donor;
        recipient_target,
        donor_target=selection.donor_target,
        rng,
        donor_path=selection.path,
        trace_callback=pool_trace,
        kwargs...,
    )
    record_outcome!(pool, selection.donor_id, _pool_outcome(result, matched[]))
    return DonorPoolAttempt(selection.donor_id, selection.path, result)
end

@testitem "donor pool feedback preserves exploration and drives eviction" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        DonorPool, DonorPoolConfig, DonorPoolOutcome, Program,
        admit_donor!, donor_pool_stats, donor_weights, record_outcome!,
        sample_donor
    using LambdaRegression.TypeSystem: IntTy, TyFun

    target = TyFun(IntTy, IntTy)
    donors = [
        Program(AstTree(AbsNode(Symbol(:x, i), IndexNode(0))))
        for i in 1:3
    ]
    pool = DonorPool(DonorPoolConfig(
        capacity=2,
        policy=:adaptive,
        exploration_floor=0.1,
        age_penalty=0.01,
    ))
    first_id = admit_donor!(pool, donors[1], [1]; donor_target=target)
    second_id = admit_donor!(pool, donors[2], [1]; donor_target=target)
    for _ in 1:3
        record_outcome!(
            pool,
            first_id,
            DonorPoolOutcome(true, true, true, 0.25),
        )
        record_outcome!(pool, second_id, DonorPoolOutcome())
    end

    weights = donor_weights(pool)
    @test weights[first_id] > weights[second_id] > 0
    third_id = admit_donor!(pool, donors[3], [1]; donor_target=target)
    stats = donor_pool_stats(pool)
    @test stats.admitted == 3
    @test stats.promotions == 3
    @test stats.demotions == 3
    @test stats.evictions == 1
    @test first_id in stats.entry_ids
    @test third_id in stats.entry_ids
    @test !(second_id in stats.entry_ids)

    isolated = DonorPool(DonorPoolConfig(capacity=1, policy=:uniform))
    admit_donor!(isolated, donors[1], [1]; donor_target=target)
    first_selection = sample_donor(isolated; rng=MersenneTwister(1234))
    second_selection = sample_donor(isolated; rng=MersenneTwister(1234))
    @test !haskey(first_selection.donor.local_consts, :caller_owned)
    @test !haskey(second_selection.donor.local_consts, :caller_owned)
end

@testitem "donor pool recombines an explicit open fragment transactionally" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        DonorPool, DonorPoolAttempt, DonorPoolConfig, Program,
        PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover,
        admit_donor!, donor_pool_stats, recombine_from_pool
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    int_function = TyFun(IntTy, IntTy)
    ctx = Dict{Symbol,TyLike}(
        :ADD => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :ZERO => IntTy,
        :USE_FUN => TyFun(int_function, IntTy),
    )
    recipient = Program(
        AstTree(AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:ADD), IndexNode(0)),
                ConstNode(:ZERO),
            ),
        )),
    )
    donor = Program(
        AstTree(AbsNode(
            :x,
            AppNode(
                ConstNode(:USE_FUN),
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(ConstNode(:ADD), IndexNode(1)),
                        IndexNode(0),
                    ),
                ),
            ),
        )),
    )
    before_recipient, before_donor = deepcopy(recipient), deepcopy(donor)
    pool = DonorPool(DonorPoolConfig(capacity=4, policy=:uniform))
    donor_id = admit_donor!(pool, donor, [1, 2, 1]; donor_target=int_function)
    op = TypedSubtreeSwapCrossover(
        max_try=1,
        require_same_depth=false,
        min_sites=1,
        selection=PathCrossoverSelectionPolicy([1, 1, 2], [1, 2, 1]),
    )

    attempt = recombine_from_pool(
        op,
        recipient,
        pool;
        rng=MersenneTwister(2000),
        global_consts_ty=ctx,
        recipient_target=int_function,
    )

    @test attempt isa DonorPoolAttempt
    @test attempt.donor_id == donor_id
    @test !(attempt.result isa Err)
    attempt.result isa Err || @test infer(attempt.result.tree, ctx) == int_function
    @test recipient.tree == before_recipient.tree
    @test donor.tree == before_donor.tree
    stats = donor_pool_stats(pool)
    @test stats.samples == 1
    @test stats.matches == 1
    @test stats.valid_children == 1
    @test stats.genuine_changes == 1
end

@testitem "donor pool promotes only an observed retrieval witness" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        DonorPool, DonorPoolConfig, Program, PathCrossoverSelectionPolicy,
        TypedSubtreeSwapCrossover, admit_donor!, donor_pool_stats,
        recombine_from_pool
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    target = TyFun(IntTy, IntTy)
    donor = Program(AstTree(AbsNode(:x, IndexNode(0))))
    recipient = deepcopy(donor)
    pool = DonorPool(DonorPoolConfig(capacity=1, policy=:adaptive))
    admit_donor!(pool, donor, [1]; donor_target=target)
    attempt = recombine_from_pool(
        TypedSubtreeSwapCrossover(
            max_try=1,
            allow_root=false,
            selection=PathCrossoverSelectionPolicy([1], [9]),
        ),
        recipient,
        pool;
        rng=MersenneTwister(1),
        global_consts_ty=Dict{Symbol,TyLike}(),
        recipient_target=target,
    )
    @test attempt.result isa Err
    stats = donor_pool_stats(pool)
    @test stats.matches == 0
    @test stats.promotions == 0
    @test stats.demotions == 1
end

@testitem "donor pool updates are lock-protected" begin
    using Test
    using Base.Threads
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Mutations.Populations:
        DonorPool, DonorPoolConfig, DonorPoolOutcome, Program,
        admit_donor!, donor_pool_stats, record_outcome!
    using LambdaRegression.TypeSystem: IntTy, TyFun

    donor = Program(
        AstTree(AbsNode(:x, IndexNode(0))),
    )
    pool = DonorPool(DonorPoolConfig(capacity=2, policy=:adaptive))
    donor_id = admit_donor!(pool, donor, [1]; donor_target=TyFun(IntTy, IntTy))
    tasks = [Threads.@spawn record_outcome!(
        pool,
        donor_id,
        DonorPoolOutcome(true, true, iseven(i), 0.0),
    ) for i in 1:20]
    fetch.(tasks)

    stats = donor_pool_stats(pool)
    @test stats.matches == 20
    @test stats.valid_children == 20
    @test stats.genuine_changes == 10
end
