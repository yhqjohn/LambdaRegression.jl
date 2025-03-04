"""
Hall-of-Fame retention channel for regression populations.

`HallOfFame` is a loss-complexity archive modeled on the scalar regularized
search: every generation records the best valid individual per complexity
level (deep-copied, alias-free), and hands back a bounded number of
Pareto-front migrants as ordinary genetic material. Attach an instance through
`components.retention`; an absent channel leaves `fit!` behavior unchanged.
The archive is cumulative for the lifetime of the instance: construct a fresh
`HallOfFame` for each independent fit, and reuse an instance only when
cross-fit retention is deliberate.

Anchoring strategy: per-complexity incumbent archive with strict-improvement
replacement, deterministic rotation over the loss-complexity Pareto front for
migrant selection. The channel changes population composition only — never
selection, initialization, or mutation probabilities — and requires the
`fit!` callback state fields `pop`, `losses`, `complexities`,
`valid_candidates`, and `perm`. Failure modes: invalid candidates are never
archived, and `take_migrants!` returns an empty vector when the archive or
the remaining capacity is empty, so composition never manufactures material.
"""

"""Loss-complexity Hall-of-Fame archive with bounded migrant re-injection."""
mutable struct HallOfFame
    migrants_per_gen::Int
    archive::Dict{Int,Individual}
    archive_losses::Dict{Int,Float64}
    last_migrants::Int

    function HallOfFame(migrants_per_gen::Int)
        migrants_per_gen >= 0 || throw(ArgumentError(
            "migrants_per_gen must be nonnegative",
        ))
        return new(migrants_per_gen, Dict{Int,Individual}(), Dict{Int,Float64}(), 0)
    end
end

HallOfFame(; migrants_per_gen::Int=2) = HallOfFame(migrants_per_gen)

"""
    record!(hof, pop, losses, complexities, valid_candidates, perm)

Record the best valid individual per complexity level. The fitness ranking
`perm` breaks ties in favour of the higher-ranked individual; archive members
are deep copies and never alias the live population.
"""
function record!(
    hof::HallOfFame,
    pop,
    losses,
    complexities,
    valid_candidates,
    perm,
)
    for rank in eachindex(perm)
        index = perm[rank]
        valid_candidates[index] || continue
        complexity = complexities[index]
        if !haskey(hof.archive_losses, complexity) ||
           losses[index] < hof.archive_losses[complexity]
            hof.archive_losses[complexity] = losses[index]
            hof.archive[complexity] = deepcopy(pop[index])
        end
    end
    return hof
end

"""
    pareto_front(hof) -> Vector{Pair{Int,Individual}}

Best-valid-individual Pareto front over (complexity, loss), ordered by
ascending complexity: an archive entry survives when its loss strictly beats
every smaller-complexity incumbent.
"""
function pareto_front(hof::HallOfFame)
    front = Vector{Pair{Int,Individual}}()
    best_loss = Inf
    for complexity in sort!(collect(keys(hof.archive_losses)))
        loss = hof.archive_losses[complexity]
        if loss < best_loss
            best_loss = loss
            push!(front, complexity => hof.archive[complexity])
        end
    end
    return front
end

"""
    take_migrants!(hof, generation, capacity) -> Vector{Individual}

Return up to `min(migrants_per_gen, capacity)` independently owned migrants by
deterministic rotation over the Pareto front (`generation` seeds the rotation
offset). The number actually taken is recorded for `stats`; capacity or an
empty archive yields an empty vector.
"""
function take_migrants!(hof::HallOfFame, generation::Int, capacity::Int)
    taken = Individual[]
    front = pareto_front(hof)
    for k in 1:min(hof.migrants_per_gen, capacity)
        isempty(front) && break
        member = front[mod1(generation + k - 1, length(front))]
        push!(taken, deepcopy(member.second))
    end
    hof.last_migrants = length(taken)
    return taken
end

"""
    stats(hof) -> NamedTuple

Current observability snapshot: archive occupancy, Pareto-front size, and the
number of migrants taken by the most recent `take_migrants!` call.
"""
function stats(hof::HallOfFame)
    return (
        archive_size=length(hof.archive),
        front_size=length(pareto_front(hof)),
        migrants=hof.last_migrants,
    )
end

@testitem "Regression HallOfFame archive records per-complexity incumbents" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: HallOfFame, record!, pareto_front
    using LambdaRegression.Regression: stats, take_migrants!
    using LambdaRegression.TypeSystem: FADD, FloatTy, TyFun, get_type_ctx

    small = Individual(
        AstTree(AbsNode(:x, IndexNode(0))),
        TyFun(FloatTy, FloatTy);
        meta=(; id=:small),
    )
    large = Individual(
        AstTree(AbsNode(
            :x,
            AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), IndexNode(0)),
        )),
        TyFun(FloatTy, FloatTy);
        meta=(; id=:large),
    )
    get_type_ctx(Dict{Symbol,Any}(:FADD => FADD))  # fixture documents FADD typing

    hof = HallOfFame(migrants_per_gen=1)
    pop = [small, large, large]
    losses = [2.0, 1.0, 1.0]
    complexities = [2, 6, 6]
    perm = [2, 3, 1]

    record!(hof, pop, losses, complexities, [true, true, false], perm)
    @test length(hof.archive) == 2
    @test hof.archive_losses[2] == 2.0
    @test hof.archive_losses[6] == 1.0
    @test hof.archive[2].meta.id === :small
    @test hof.archive[6].meta.id === :large
    # The invalid third individual never enters the archive.
    @test hof.archive[6] !== pop[2]

    front = pareto_front(hof)
    @test first.(front) == [2, 6]
    @test last.(front) .|> (ind -> ind.meta.id) == [:small, :large]

    # Strict-improvement replacement keeps the incumbent on equal loss.
    record!(hof, pop, [2.0, 1.0, 1.0], complexities, [true, true, true], perm)
    @test hof.archive[6].meta.id === :large

    # Generation 1 rotates from the smallest-complexity front member.
    migrants = take_migrants!(hof, 1, 4)
    @test length(migrants) == 1
    @test migrants[1].meta.id === :small
    @test migrants[1] !== hof.archive[2]
    @test stats(hof) == (; archive_size=2, front_size=2, migrants=1)
    @test isempty(take_migrants!(hof, 2, 0))
end

@testitem "Regression fit composes the HallOfFame retention channel" setup=[RegBaseImports] begin
    using Random
    using LambdaRegression.Regression: HallOfFame
    using LambdaRegression.TypeSystem: FADD, get_type_ctx

    target_ty = TyFun(FloatTy, FloatTy)
    # Two tree shapes with distinct node counts; custom evaluation assigns a
    # deterministic lower loss to the larger tree so ranking, elites, and the
    # loss-complexity front are all predictable.
    small_tree = AstTree(AbsNode(:x, IndexNode(0)))
    large_tree = AstTree(AbsNode(
        :x,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), IndexNode(0)),
    ))
    globals = Dict{Symbol,Any}(:FADD => FADD)
    global_types = get_type_ctx(globals)
    base = default_components(
        global_consts=globals,
        global_consts_ty=global_types,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    components = merge(base, (
        evaluate=(ind, x, components, config) -> ind.meta.id === :small ? 2.0 : 1.0,
        mutations=Mutations.AbstractMutation[],
        crossover_op=nothing,
        init_population=(reg, data; N, kwargs...) -> [
            Individual(small_tree, reg.target_ty; meta=(; id=:small)),
            Individual(large_tree, reg.target_ty; meta=(; id=:large)),
            Individual(large_tree, reg.target_ty; meta=(; id=:large)),
            Individual(small_tree, reg.target_ty; meta=(; id=:small)),
        ],
    ))
    data = SupervisedDataset([1.0], [0.0])

    common = (;
        population_size=4,
        generations=3,
        error_threshold=-1.0,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=2,
        seed=UInt64(31),
        repro=(; require_change=false, on_fail=:clone),
        stats=(; enabled=true, reproduction=true),
    )

    states_off = Any[]
    result_off = fit!(
        Regressor(target_ty, components, default_config(; common...), MersenneTwister(2000)),
        data;
        callback=state -> (push!(states_off, state); nothing),
    )
    @test !(result_off isa Err)
    @test states_off[1].retention === nothing
    @test !haskey(result_off.history[1].repro_stats, :hof_migrant)

    retained = merge(components, (; retention=HallOfFame(migrants_per_gen=1)))
    states_on = Any[]
    result_on = fit!(
        Regressor(
            target_ty,
            retained,
            default_config(; common...),
            MersenneTwister(2000),
        ),
        data;
        callback=state -> (push!(states_on, state); nothing),
    )
    @test !(result_on isa Err)
    @test states_on[1].retention == (; archive_size=2, front_size=2, migrants=0)
    @test result_on.history[1].repro_stats[:hof_migrant] == 1
    @test states_on[2].retention.migrants == 1
    # After two elites, the next slot holds the archived small-complexity migrant.
    @test states_on[2].pop[3].meta.id === :small
    @test states_on[2].retention.archive_size == 2
    # Both runs keep the same final best loss: retention must not change evaluation.
    @test result_on.history[end].best_loss == result_off.history[end].best_loss
end
