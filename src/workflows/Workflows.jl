"""
    Workflows

Composes a resumable complete-program search state with bounded typed and
scalar refinement sessions, numerical-constant search, material migration, and
compact phase accounting. Constant search edits typed local numeric payloads;
it is an independent action from scalar structural search. Scalar zero-input
sessions are disabled by default; set `scalar.allow_zero_input=true` to opt in.
Action enablement does not alter numeric-block or structural descriptions; an
ineligible payload is not a new structure merely because an optimizer is off.
`run_mixed_alternation!` borrows an initialized `AgeSearchState`; callers
remain responsible for task construction, data splits, and reporting.

```julia
result = run_mixed_alternation!(state; rng=MersenneTwister(1), cycles=2,
    outer=outer_options, typed=typed_options,
    scalar=scalar_options, sampling=sampling_options)
```

Layer policies are supplied by the caller; this library module does
not prescribe a task-specific profile.

`select_chain_seeds` combines ordered pools of learned programs with bounded
random, source-rank, or new-task-scored selection. It returns ordinary Program
seeds and screening counts; it neither trains subproblems nor starts a search.

`run_continuous_search!` is the alternative serial action composition: bounded
outer and local work publish material immediately, with an optional bounded
`StructurePolicy` for family opportunities and survival. Action weights count
actions rather than CPU share; local source selection includes an explicit
exploration fraction over deduplicated archive/live/recent/representative material.
An opt-in `scalar.session` retains one fixed-context multi-island search across
actions within the call. Its configuration budgets cover the whole session;
outer actions do not reset it. Cross-call resume and multicore remain extensions.
`StructurePolicy(world)` connects a stable `StructureDiversity.StructureWorld`
directly; its scalar named-input policy must match the local scalar options.

`run_joint_search!` keeps a complete-program joint population alive across
bounded windows while the existing age search continues to explore. It imports
each direction through the receiving engine's scorer and reports separate
joint, outer, bridge, and total costs. The configured scalar population size
applies per joint island; seed-program count does not create subpopulations.

```julia
joint = run_joint_search!(state; rng=MersenneTwister(1),
    config=LambdaRegression.ScalarSearch.ScalarSearchConfig(
        evolution_strategy=:regularized),
    windows=4, outer=(attempts=128, evaluations=32))
```

```julia
world = StructureDiversity.StructureWorld(reg.components.global_consts_ty;
    values=reg.components.global_consts, input_types=reg.components.input_types,
    target_ty=reg.target_ty)
result = run_continuous_search!(state; rng=MersenneTwister(1),
    config=ContinuousSearchConfig(max_actions=16),
    outer=outer_options, typed=typed_options, scalar=scalar_options,
    constant=(enabled=true,
        strategy=LambdaRegression.Regression.GaussianConstantSearch(σ=0.1),
        budget=LambdaRegression.Regression.ConstantSearchBudget(
            max_proposals=8, max_objective_evals=9)),
    policy=StructurePolicy(world))
```

The state retains the regressor and training data. Optional `on_observe` is
synchronous and receives borrowed `(entry, state, selected, materials)`; the
callback must not mutate or retain them. Copy only what an observer needs.
Scalar scoring is synchronous as well: its temporary `Program` path-copies the
edited ancestors and borrows untouched syntax read-only. The scalar scorer
must not mutate or retain that candidate; returned materials are separately
published as owned `Program`s.
"""
module Workflows

using TestItems
using Random: AbstractRNG, MersenneTwister, default_rng, rand, shuffle!
using ..Utils: Err
using ..ASTs.NodeASTs: AstNode, children
using ..Programs: Program, numeric_parameter_blocks
using ..TypeSystem: Ty, TyLike
using ..ScalarSearch: ScalarSearchConfig, scalar_complexity, search_scalar, search_multi_scalar
using ..StructureDiversity: StructureWorld, StructureBoundaryPolicy, describe_structure
import ..Regression
using ..Regression: Regressor, SupervisedDataset, AgeSearchState,
    AgeSearchIndividual, AgeSearchMigration, prepare_task_evaluator,
    score_samples, mse, advance_age_search!, inject_age_migrants!,
    search_typed_roots, search_typed_hole, prepare_scalar_region,
    search_scalar_region

export run_mixed_alternation!, ObjectiveCounts, score_candidate!
export StructurePolicy, structure_snapshot, ContinuousSearchConfig, run_continuous_search!
export run_joint_search!
export select_chain_seeds

mutable struct ObjectiveCounts
    calls::Int
    samples::Int
    invalid::Int
end
ObjectiveCounts() = ObjectiveCounts(0, 0, 0)

function _copy_named(options, defaults)
    options === nothing && return defaults
    options isa NamedTuple || throw(ArgumentError("runner options must be NamedTuple"))
    return merge(defaults, options)
end

"""Record one completed phase and optionally publish its compact progress."""
function _record_phase!(ledger, entry, on_phase; details=nothing)
    push!(ledger, entry)
    on_phase === nothing && return nothing
    event = (;
        cycle=entry.cycle,
        phase=entry.phase,
        elapsed_seconds=entry.elapsed_seconds,
        best_loss=entry.best_loss,
        returned=entry.returned,
        incoming=entry.incoming)
    details === nothing || (event = merge(event, (; details)))
    on_phase(event)
    return nothing
end

function _outer_phase_counters(state::AgeSearchState)
    counters = state.counters
    return (; attempts=Int(counters.raw_attempts),
        accepted=Int(counters.accepted),
        objective_calls=Int(counters.objective_calls),
        sample_executions=Int(counters.sample_executions),
        invalids=Int(counters.invalids))
end

function _record_phase_snapshot!(ledger, entry, on_phase, on_observe,
                                state; selected=(), materials=())
    details = merge(entry, (; outer_counters=_outer_phase_counters(state)))
    _record_phase!(ledger, entry, on_phase; details)
    on_observe === nothing || on_observe(entry, state, selected, materials)
    return nothing
end

"""Score one complete candidate; `copy_tree=false` is for synchronous read-only borrows."""
function score_candidate!(counts::ObjectiveCounts, program::Program,
                          reg::Regressor, data::SupervisedDataset;
                          copy_tree::Bool=true)
    counts.calls += 1
    loss_fn = get(reg.components, :loss, mse)
    prepared = prepare_task_evaluator(
        program, reg.target_ty, reg.components, reg.config; copy_tree,
    )
    result = score_samples(data, prepared, loss_fn;
        failure_mode=:return_error,
        candidate_failure=Regression._known_candidate_exception,
        loss_failure=Regression._known_candidate_exception,
        strict_returned_errors=true,
        strict_loss_conversion=true,
        on_samples=count -> (counts.samples += count))
    result.valid || (counts.invalid += 1)
    return result.score
end

function _paths(node::AstNode, prefix::Vector{Int}=Int[])
    result = Vector{Int}[copy(prefix)]
    for (index, child) in enumerate(children(node))
        append!(result, _paths(child, [prefix; index]))
    end
    return result
end

function _record_key(program::Program, target_ty::Ty)
    return (target_ty, program.tree, program.local_consts, program.local_consts_ty)
end

function _same_program(left::Program, right::Program)
    return left.tree == right.tree &&
        isequal(left.local_consts, right.local_consts) &&
        isequal(left.local_consts_ty, right.local_consts_ty)
end

"""Merge independently sampled material groups and randomize their order.

The groups remain separately sampled (and therefore separately bounded), but
their concatenation is not an archive-first scheduling policy.
"""
function _merge_material_groups(groups, rng::AbstractRNG)
    merged = NamedTuple[]
    for group in groups
        append!(merged, group)
    end
    shuffle!(rng, merged)
    return merged
end

function _sample_material(state::AgeSearchState, rng::AbstractRNG, sampling)
    groups = [NamedTuple[], NamedTuple[], NamedTuple[]]
    ids = Set{UInt64}()
    function take!(records, count::Int, source::Symbol, selected)
        count >= 0 || throw(ArgumentError("sampling counts must be non-negative"))
        values = collect(records)
        shuffle!(rng, values)
        # Iterate the whole shuffled source until `count` valid, distinct IDs
        # have been selected.  Taking only the first `count` raw records
        # underfilled pools whenever invalid/duplicate records appeared early.
        for record in values
            length(selected) >= count && break
            record isa AgeSearchIndividual || continue
            record.id in ids && continue
            record.valid || continue
            push!(ids, record.id)
            push!(selected, (; id=record.id, program=record.program,
                record, source))
        end
    end
    take!(values(state.archive), Int(sampling.archive), :archive, groups[1])
    take!(Iterators.flatten(state.populations), Int(sampling.live), :live,
        groups[2])
    take!(state.recent_valid_offspring, Int(sampling.recent), :recent, groups[3])
    return _merge_material_groups(groups, rng)
end

"""Select the bounded input pool for one joint typed-root search."""
function _typed_root_selection(selected, root_batch::Int)
    root_batch >= 0 || throw(ArgumentError("typed root batch must be non-negative"))
    return collect(Iterators.take(selected, root_batch))
end

"""Schedule scalar paths by source round-robin under one shared attempt limit.

Paths are shuffled once per source.  Every returned record consumes one
attempt, including paths that the caller later rejects during preparation or
search; this makes the boundary independent of success rate.
"""
function _scheduled_scalar_attempts(sources, limit::Int, rng::AbstractRNG)
    limit >= 0 || throw(ArgumentError("scalar attempt limit must be non-negative"))
    prepared = NamedTuple[]
    for source in sources
        paths = hasproperty(source, :paths) ? copy(source.paths) :
            _paths(source.program.tree.root)
        shuffle!(rng, paths)
        push!(prepared, (; source, paths))
    end
    cursors = ones(Int, length(prepared))
    attempts = NamedTuple[]
    while length(attempts) < limit
        progressed = false
        for source_index in eachindex(prepared)
            entry = prepared[source_index]
            cursor = cursors[source_index]
            cursor <= length(entry.paths) || continue
            source = entry.source
            push!(attempts, (; route=source.route,
                source_id=source.source_id, pool_ids=source.pool_ids,
                source, path=copy(entry.paths[cursor])))
            cursors[source_index] = cursor + 1
            progressed = true
            length(attempts) >= limit && break
        end
        progressed || break
    end
    return attempts
end

function _typed_options(typed; phase::Symbol=:shared)
    base = (
        population_size=Int(typed.population_size),
        attempts=Int(typed.attempts),
        evaluations=Int(typed.evaluations),
        max_nodes=Int(typed.max_nodes),
        max_depth=Int(typed.max_depth),
        max_return=Int(typed.max_return),
        root_batch=Int(typed.root_batch),
        local_holes=Int(typed.local_holes),
    )
    overrides = phase === :shared ? (;) :
        getproperty(typed, phase === :root ? :root_options :
            phase === :local ? :local_options : throw(ArgumentError(
                "typed option phase must be :shared, :root, or :local")))
    return _copy_named(overrides, base)
end

function _scalar_config(scalar)
    supplied = hasproperty(scalar, :config) ? scalar.config : nothing
    supplied isa ScalarSearchConfig && return supplied
    p = Int(scalar.population_size)
    return ScalarSearchConfig(
        population_size=p,
        max_evaluations=Int(scalar.max_evaluations),
        max_complexity=Int(scalar.max_complexity),
        initial_depth=Int(scalar.initial_depth),
        tournament_size=Int(scalar.tournament_size),
        elite_count=Int(scalar.elite_count),
        unary_operators=scalar.unary_operators,
        binary_operators=scalar.binary_operators,
        constant_probability=scalar.constant_probability,
        mutation_probability=scalar.mutation_probability,
        crossover_probability=scalar.crossover_probability,
        constant_refinement_iterations=Int(scalar.constant_refinement_iterations),
        constant_refinement_interval=Int(scalar.constant_refinement_interval),
        constant_refinement_start_generation=Int(scalar.constant_refinement_start_generation),
        constant_refinement_probability=scalar.constant_refinement_probability,
        constant_refinement_timing=get(scalar, :constant_refinement_timing, :cycle),
        max_constant_refinement_evaluations=Int(scalar.max_constant_refinement_evaluations),
        max_total_evaluations=get(scalar, :max_total_evaluations, nothing),
        evolution_strategy=:regularized,
    )
end

function _scalar_session_limit(scalar)
    limit = scalar.session_limit
    limit === nothing && return nothing
    limit isa Integer || throw(ArgumentError("scalar session_limit must be an integer or nothing"))
    Int(limit) >= 0 || throw(ArgumentError("scalar session_limit must be non-negative"))
    return Int(limit)
end

function _append_typed!(materials, result, phase::Symbol, parents, path)
    result isa Err && return false
    result.candidates isa AbstractVector || return false
    added = false
    for candidate in result.candidates
        hasproperty(candidate, :program) || continue
        candidate.program isa Program || continue
        loss = Float64(candidate.loss)
        isfinite(loss) || continue
        actual_parents = phase === :typed_root ? UInt64[] : parents
        push!(materials, (; phase, program=candidate.program, loss,
            complexity=Int(candidate.complexity), parent_ids=actual_parents, path,
            pool_ids=parents))
        added = true
    end
    return added
end

function _append_scalar!(materials, result, phase::Symbol, parents, path;
        pool_ids=parents)
    result isa Err && return false
    result isa NamedTuple || return false
    hasproperty(result, :candidates) || return false
    added = false
    for candidate in result.candidates
        candidate.program isa Program || continue
        loss = Float64(candidate.loss)
        isfinite(loss) || continue
        push!(materials, (; phase, program=candidate.program, loss,
            complexity=Int(sum(1 for _ in candidate.program.tree)),
            parent_ids=parents, path, pool_ids))
        added = true
    end
    return added
end

function _refinement_summary(result)
    if result isa Err
        reason = result.value isa Regression.TypedRefinementError ?
            result.value.reason : result.value isa Regression.ScalarRegionError ?
            result.value.reason : :refused
        return (; status=:refused, reason)
    end
    diagnostics = hasproperty(result, :diagnostics) ? result.diagnostics : (;)
    returned = hasproperty(result, :candidates) ? length(result.candidates) : 0
    summary = (; status=:ok, returned)
    # Keep only fields actually supplied by the bridge.  In particular, a
    # typed result has no scalar structural/refinement counter, so fabricating
    # zeroes here would misreport an unmeasured quantity as an observed one.
    for field in propertynames(diagnostics)
        field === :backend_result && continue
        summary = merge(summary,
            NamedTuple{(field,)}((getproperty(diagnostics, field),)))
    end
    # Typed bridges retain a compact backend result for their own contract.
    # Copy a small set of scalar stats under an explicit namespace without
    # retaining the raw result or its mutable populations in the ledger.
    if hasproperty(diagnostics, :backend_result)
        backend = diagnostics.backend_result
        if hasproperty(backend, :stats)
            stats = backend.stats
            backend_summary = (;)
            for field in (:attempts, :evaluations, :invalid_evaluations,
                          :initialization_attempts, :initialization_evaluations,
                          :initialization_invalid_evaluations, :offspring,
                          :generations, :partial_generations)
                hasproperty(stats, field) || continue
                backend_summary = merge(backend_summary,
                    NamedTuple{(field,)}((getproperty(stats, field),)))
            end
            isempty(propertynames(backend_summary)) ||
                (summary = merge(summary, (; backend_stats=backend_summary)))
        end
    end
    return summary
end

function _choose_migrants(materials, state::AgeSearchState, max_migrants::Int,
                          rng::AbstractRNG)
    max_migrants >= 0 || throw(ArgumentError("max_migrants must be non-negative"))
    # Round-robin the phase buckets after a small independent shuffle.  This
    # keeps a scalar route from being starved by a long typed archive while
    # still bounding the number of incoming migrants.
    shuffled = copy(materials)
    shuffle!(rng, shuffled)
    phases = Symbol[]
    for material in shuffled
        material.phase in phases || push!(phases, material.phase)
    end
    ordered = NamedTuple[]
    remaining = copy(shuffled)
    while !isempty(remaining)
        progressed = false
        for phase in phases
            index = findfirst(material -> material.phase === phase, remaining)
            index === nothing && continue
            push!(ordered, popat!(remaining, index))
            progressed = true
        end
        progressed || break
    end
    chosen = Any[]
    duplicate_count = 0
    for material in ordered
        length(chosen) >= max_migrants && break
        any(record -> _same_program(record.program, material.program),
            Iterators.flatten(state.populations)) &&
            (duplicate_count += 1; continue)
        any(value -> _same_program(value.program, material.program), chosen) &&
            (duplicate_count += 1; continue)
        push!(chosen, material)
    end
    return chosen, duplicate_count
end

"""
    run_mixed_alternation!(state::AgeSearchState; rng, cycles=2, outer=(...),
        typed=(...), scalar=(...), sampling=(...), max_migrants=2,
        initialization_seconds=0, on_phase=nothing, on_observe=nothing)

Run bounded outer, typed, and scalar refinement phases over an initialized
age-search state. All candidate selection uses `state.data`; task construction
and held-out evaluation remain the caller's responsibility. The returned
ledger distinguishes sampled pool IDs from actual parent IDs. `on_observe`
is called synchronously with borrowed `(entry, state, selected, materials)`
after a completed phase; callbacks must not mutate or retain those values.
"""
function run_mixed_alternation!(state::AgeSearchState;
                         rng::AbstractRNG,
                         cycles::Int=2,
                         outer,
                         typed,
                         scalar,
                         sampling,
                         max_migrants::Int=2,
                         initialization_seconds::Real=0,
                         on_phase=nothing,
                         on_observe=nothing)
    cycles > 0 || throw(ArgumentError("cycles must be positive"))
    max_migrants >= 0 || throw(ArgumentError("max_migrants must be non-negative"))
    all(options -> options isa NamedTuple, (outer, typed, scalar, sampling)) ||
        throw(ArgumentError("workflow layer options must be NamedTuple values"))
    outer_cfg, typed_cfg, scalar_cfg, sampling_cfg = outer, typed, scalar, sampling
    reg = state.reg
    train = state.data
    typed_counts = ObjectiveCounts()
    scalar_counts = ObjectiveCounts()
    ledger = NamedTuple[]
    all_materials = NamedTuple[]
    global_types = Dict{Symbol,TyLike}(reg.components.global_consts_ty)
    input_types = Dict{Symbol,TyLike}(reg.components.input_types)
    typed_shared = typed_cfg.enabled ? _typed_options(typed_cfg) : nothing
    typed_root = typed_cfg.enabled ? _typed_options(typed_cfg; phase=:root) : nothing
    typed_local = typed_cfg.enabled ? _typed_options(typed_cfg; phase=:local) : nothing
    scalar_search_config = scalar_cfg.enabled ? _scalar_config(scalar_cfg) : nothing
    scalar_session_limit = scalar_cfg.enabled ? _scalar_session_limit(scalar_cfg) : nothing
    scalar_size_refusals = 0
    scalar_session_total = 0
    scalar_prepared_total = 0
    scalar_admissible_total = 0
    scalar_successful_search_total = 0
    initialization_best_loss = state.best === nothing ? Inf : state.best.loss
    _record_phase_snapshot!(ledger, (; cycle=0, phase=:initialization, report=nothing,
        selected_ids=UInt64[], paths=Vector{Int}[], returned=sum(
            length, state.populations), incoming=0,
        best_loss=initialization_best_loss,
        elapsed_seconds=Float64(initialization_seconds)),
        on_phase, on_observe, state)
    for cycle in 1:cycles
        outer_pre_started = time_ns()
        if outer_cfg.enabled
            report = advance_age_search!(state;
                max_attempts=Int(outer_cfg.attempts),
                max_evaluations=Int(outer_cfg.evaluations))
            _record_phase_snapshot!(ledger, (; cycle, phase=:outer_pre, report,
                objective_calls=state.counters.advance_objective_calls,
                selected_ids=UInt64[], paths=Vector{Int}[],
                returned=0, incoming=0,
                best_loss=state.best === nothing ? Inf : state.best.loss,
                elapsed_seconds=(time_ns() - outer_pre_started) / 1.0e9),
                on_phase, on_observe, state)
        else
            _record_phase_snapshot!(ledger, (; cycle, phase=:outer_pre, report=nothing,
                objective_calls=0, selected_ids=UInt64[],
                paths=Vector{Int}[], returned=0, incoming=0,
                best_loss=state.best === nothing ? Inf : state.best.loss,
                elapsed_seconds=(time_ns() - outer_pre_started) / 1.0e9),
                on_phase, on_observe, state)
        end
        selected = _sample_material(state, rng, sampling_cfg)
        selected_ids = UInt64[item.id for item in selected]
        selected_paths = Vector{Int}[]
        materials = NamedTuple[]
        typed_root_added = false
        typed_started = time_ns()
        if typed_cfg.enabled && !isempty(selected)
            root_selected = _typed_root_selection(selected, Int(typed_shared.root_batch))
            root_selected_ids = UInt64[item.id for item in root_selected]
            # These IDs describe the supplied root pool, not direct ancestry of
            # a joint-search result (which intentionally has no single parent).
            root_pool_ids = Tuple(root_selected_ids)
            roots = [item.program for item in root_selected]
            root_result = search_typed_roots(roots, global_types,
                program -> score_candidate!(typed_counts, program, reg, train);
                arity=length(reg.components.input_slots),
                target_ty=reg.target_ty,
                input_types,
                rng=MersenneTwister(rand(rng, UInt64)),
                max_return=Int(typed_root.max_return),
                max_nodes=Int(typed_root.max_nodes),
                max_depth=Int(typed_root.max_depth),
                attempts=Int(typed_root.attempts),
                evaluations=Int(typed_root.evaluations),
                population_size=Int(typed_root.population_size))
            typed_root_added = _append_typed!(materials, root_result,
                :typed_root, root_pool_ids, Int[])
            root_status = root_result isa Err ? :refused : :searched
            root_reason = root_result isa Err ? _refinement_summary(root_result).reason : nothing
            _record_phase_snapshot!(ledger, (; cycle, phase=:typed_root,
                report=_refinement_summary(root_result),
                attempt_records=[(; route=:typed_root, source_id=nothing,
                    pool_ids=root_pool_ids, path=Int[], status=root_status,
                    reason=root_reason)],
                selected_ids=root_selected_ids,
                paths=Vector{Int}[],
                returned=typed_root_added ? length(root_result.candidates) : 0,
                incoming=0, best_loss=state.best === nothing ? Inf : state.best.loss,
                elapsed_seconds=(time_ns() - typed_started) / 1.0e9),
                on_phase, on_observe, state;
                selected, materials)
        else
            _record_phase_snapshot!(ledger, (; cycle, phase=:typed_root, report=nothing,
                selected_ids, paths=Vector{Int}[], returned=0, incoming=0,
                best_loss=state.best === nothing ? Inf : state.best.loss,
                elapsed_seconds=(time_ns() - typed_started) / 1.0e9),
                on_phase, on_observe, state;
                selected, materials)
        end
        typed_holes_started = time_ns()
        local_refusals = NamedTuple[]
        local_attempts = NamedTuple[]
        if typed_cfg.enabled
            local_limit = Int(typed_shared.local_holes)
            for item in Iterators.take(selected, length(selected))
                local_limit <= 0 && break
                paths = _paths(item.program.tree.root)
                filter!(path -> !isempty(path), paths)
                shuffle!(rng, paths)
                for path in Iterators.take(paths, 1)
                    local_limit -= 1
                    push!(selected_paths, path)
                    result = search_typed_hole(item.program, path, global_types,
                        program -> score_candidate!(typed_counts, program, reg, train);
                        target_ty=reg.target_ty,
                        input_types,
                        rng=MersenneTwister(rand(rng, UInt64)),
                        max_return=Int(typed_local.max_return),
                        max_nodes=Int(typed_local.max_nodes),
                        max_depth=Int(typed_local.max_depth),
                        attempts=Int(typed_local.attempts),
                        evaluations=Int(typed_local.evaluations),
                        population_size=Int(typed_local.population_size))
                    summary = _refinement_summary(result)
                    status = result isa Err ? :refused : :searched
                    reason = result isa Err ? summary.reason : nothing
                    result isa Err && push!(local_refusals,
                        (; path=copy(path), reason))
                    push!(local_attempts, (; route=:typed_hole,
                        source_id=item.id, pool_ids=(item.id,), path=copy(path),
                        status, reason, summary))
                    _append_typed!(materials, result, :typed_hole,
                        (item.id,), path)
                end
            end
        end
        _record_phase_snapshot!(ledger, (; cycle, phase=:typed_holes,
            report=(; attempts=length(local_attempts), refusals=local_refusals,
                attempt_records=local_attempts,
                admitted_ids=UInt64[attempt.source_id for attempt in local_attempts
                    if attempt.status === :searched]),
            selected_ids,
            admitted_ids=UInt64[attempt.source_id for attempt in local_attempts
                if attempt.status === :searched],
            paths=selected_paths, returned=count(
                m -> m.phase === :typed_hole, materials), incoming=0,
            best_loss=state.best === nothing ? Inf : state.best.loss,
            elapsed_seconds=(time_ns() - typed_holes_started) / 1.0e9),
            on_phase, on_observe, state;
            selected, materials)

        scalar_paths = Vector{Int}[]
        scalar_limit = Int(sampling_cfg.scalar_holes)
        scalar_attempts = NamedTuple[]
        scalar_prepared_count = 0
        scalar_admissible_count = 0
        scalar_session_count = 0
        scalar_successful_search_count = 0
        scalar_admitted_ids = UInt64[]
        scalar_started = time_ns()
        if scalar_cfg.enabled
            selected_sources = [(program=item.program,
                parent_ids=(item.id,), pool_ids=(item.id,), route=:original,
                source_id=item.id) for item in selected]
            typed_sources = [(program=m.program,
                parent_ids=m.parent_ids, pool_ids=m.pool_ids, route=m.phase,
                source_id=isempty(m.parent_ids) ? nothing : first(m.parent_ids))
                for m in materials]
            scalar_sources = NamedTuple[]
            for index in 1:max(length(selected_sources), length(typed_sources))
                index <= length(selected_sources) &&
                    push!(scalar_sources, selected_sources[index])
                index <= length(typed_sources) &&
                    push!(scalar_sources, typed_sources[index])
            end
            scalar_backend = (X, y; kwargs...) -> search_multi_scalar(X, y;
                kwargs...,
                island_count=Int(scalar_cfg.island_count),
                rounds=Int(scalar_cfg.rounds),
                migrant_count=Int(scalar_cfg.migrant_count),
                top_migrant_count=scalar_cfg.top_migrant_count,
                ncycles_per_iteration=Int(scalar_cfg.ncycles_per_iteration),
                frequency_window=scalar_cfg.frequency_window,
                top_migration_rate=scalar_cfg.top_migration_rate,
                donor_limit=scalar_cfg.donor_limit)
            for attempt in _scheduled_scalar_attempts(scalar_sources, scalar_limit, rng)
                scalar_session_limit !== nothing &&
                    scalar_session_count >= scalar_session_limit && break
                path = attempt.path
                source_entry = attempt.source
                source = source_entry.program
                status = :searched
                reason = nothing
                summary = nothing
                region = prepare_scalar_region(source, reg.target_ty, global_types, path;
                    input_types,
                    max_nodes=Int(outer_cfg.max_nodes),
                    max_depth=Int(outer_cfg.max_depth))
                if region isa Err
                    status = :refused
                    reason = _refinement_summary(region).reason
                    summary = _refinement_summary(region)
                else
                    scalar_prepared_count += 1
                    if scalar_complexity(region.expr) >
                            scalar_search_config.max_complexity
                        scalar_size_refusals += 1
                        status = :refused
                        reason = :scalar_size
                        summary = (; status=:refused, reason=:scalar_size)
                    else
                        scalar_admissible_count += 1
                        scalar_session_count += 1
                        attempt.source_id === nothing ||
                            push!(scalar_admitted_ids, UInt64(attempt.source_id))
                        push!(scalar_paths, path)
                        result = search_scalar_region(region,
                            program -> score_candidate!(scalar_counts, program, reg, train;
                                copy_tree=false);
                            config=scalar_search_config,
                            rng=MersenneTwister(rand(rng, UInt64)),
                            max_return=Int(scalar_cfg.max_return),
                            stop_loss=nothing,
                            allow_zero_input=get(scalar_cfg, :allow_zero_input, false),
                            search_backend=scalar_backend)
                        summary = _refinement_summary(result)
                        result isa Err || (scalar_successful_search_count += 1)
                        if result isa Err
                            status = :refused
                            reason = summary.reason
                        end
                        _append_scalar!(materials, result, :scalar,
                            Tuple(source_entry.parent_ids), path;
                            pool_ids=Tuple(source_entry.pool_ids))
                    end
                end
                push!(scalar_attempts, (; route=attempt.route,
                    source_id=attempt.source_id, pool_ids=attempt.pool_ids,
                    path=copy(path), status, reason, summary))
            end
        end
        scalar_refusals = [attempt for attempt in scalar_attempts
            if attempt.status === :refused]
        scalar_session_total += scalar_session_count
        scalar_prepared_total += scalar_prepared_count
        scalar_admissible_total += scalar_admissible_count
        scalar_successful_search_total += scalar_successful_search_count
        append!(all_materials, materials)
        _record_phase_snapshot!(ledger, (; cycle, phase=:scalar,
            report=(; attempts=length(scalar_attempts), refusals=scalar_refusals,
                attempt_records=scalar_attempts,
                session_count=scalar_session_count,
                prepared_count=scalar_prepared_count,
                admissible_count=scalar_admissible_count,
                successful_search_count=scalar_successful_search_count,
                admitted_ids=scalar_admitted_ids,
                attempts_by_route=(;
                    original=count(a -> a.route === :original, scalar_attempts),
                    typed_root=count(a -> a.route === :typed_root, scalar_attempts),
                    typed_hole=count(a -> a.route === :typed_hole, scalar_attempts))),
            selected_ids, admitted_ids=scalar_admitted_ids, paths=scalar_paths,
            returned=count(m -> m.phase === :scalar, materials), incoming=0,
            best_loss=state.best === nothing ? Inf : state.best.loss,
            elapsed_seconds=(time_ns() - scalar_started) / 1.0e9),
            on_phase, on_observe, state;
            selected, materials)

        migration_started = time_ns()
        migrants, duplicate_count = _choose_migrants(materials, state, max_migrants, rng)
        before_migration_ids = Set(record.id for record in
            state.populations[mod1(cycle, length(state.populations))])
        migration_values = [AgeSearchMigration(
            m.program, Tuple(UInt64(id) for id in m.parent_ids)) for m in migrants]
        migration_report = isempty(migration_values) ?
            (attempts=0, accepted=0, duplicates=0, invalids=0,
             objective_calls=0, sample_executions=0) :
            inject_age_migrants!(state, migration_values;
                population=mod1(cycle, length(state.populations)),
                max_migrants=max_migrants)
        migration_population = state.populations[mod1(cycle, length(state.populations))]
        actual_migrant_records = [record for record in migration_population
            if record.id ∉ before_migration_ids && record.origin === :migration]
        _record_phase_snapshot!(ledger, (; cycle, phase=:migration, report=migration_report,
            selected_ids, paths=[m.path for m in migrants],
            returned=length(migrants), incoming=migration_report.accepted,
            duplicate_material=duplicate_count,
            migrant_ids=UInt64[record.id for record in actual_migrant_records],
            migrant_parent_ids=[record.parent_ids for record in actual_migrant_records],
            best_loss=state.best === nothing ? Inf : state.best.loss,
            elapsed_seconds=(time_ns() - migration_started) / 1.0e9),
            on_phase, on_observe, state;
            selected, materials)
        if outer_cfg.enabled
            before_post_birth = state.birth_clock
            post_started = time_ns()
            post_report = advance_age_search!(state;
                max_attempts=Int(outer_cfg.attempts),
                max_evaluations=Int(outer_cfg.evaluations))
            post_offspring = [record for record in state.recent_valid_offspring
                if record.birth > before_post_birth]
            post_accepted = post_report isa Err ? 0 : post_report.accepted
            _record_phase_snapshot!(ledger, (; cycle, phase=:outer_post, report=post_report,
                selected_ids, paths=Vector{Int}[], returned=0,
                incoming=migration_report.accepted,
                best_loss=state.best === nothing ? Inf : state.best.loss,
                offspring_parent_ids=[record.parent_ids for record in post_offspring],
                offspring_count=length(post_offspring),
                offspring_trace_complete=length(post_offspring) == post_accepted,
                elapsed_seconds=(time_ns() - post_started) / 1.0e9),
                on_phase, on_observe, state;
                selected, materials)
        end
    end

    archive_best = state.best
    material_best = isempty(all_materials) ? nothing : reduce(all_materials) do left, right
        (right.loss, right.complexity) < (left.loss, left.complexity) ? right : left
    end
    best = archive_best
    for material in all_materials
        best !== nothing && material.loss >= best.loss && continue
        best = AgeSearchIndividual(material.program, UInt64(0),
            Tuple(UInt64(id) for id in material.parent_ids), material.phase,
            Float64(material.loss), Int(material.complexity), true, UInt64(0))
    end
    best === nothing && return Err(ArgumentError("alternation produced no valid training candidate"))
    return (
        cycles,
        state,
        best,
        archive_best,
        material_best,
        ledger,
        counts=(;
            outer=state.counters,
            typed=(objective_calls=typed_counts.calls, sample_executions=typed_counts.samples,
                invalid=typed_counts.invalid),
            scalar=(objective_calls=scalar_counts.calls, sample_executions=scalar_counts.samples,
                invalid=scalar_counts.invalid,
                session_count=scalar_session_total,
                prepared_count=scalar_prepared_total,
                admissible_count=scalar_admissible_total,
                successful_search_count=scalar_successful_search_total),
        ),
        material_count=length(all_materials),
        scalar_size_refusals,
    )
end

include("continuous.jl")
include("continuous_session_tests.jl")
include("joint.jl")
include("composition_seeds.jl")

@testitem "Mixed workflow outer-only run borrows initialized state" begin
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Mutations: AbstractMutation
    using LambdaRegression.Regression: default_components, default_config,
        default_regressor, SupervisedDataset, initialize_age_search

    identity_program = Program(AstTree(AbsNode(:x, IndexNode(0))))
    components = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=[:x], input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        mutations=AbstractMutation[], crossover_op=nothing,
        init_population=(reg, data; N, seed, global_type_ctx) ->
            [identity_program for _ in 1:N])
    reg = default_regressor(TyFun(FloatTy, FloatTy);
        rng=MersenneTwister(1234), components,
        config=default_config(population_size=1, tournament_size=1))
    data = SupervisedDataset([1.0, 2.0], [1.0, 2.0])
    state = initialize_age_search(reg, data;
        population_count=1, population_size=1, seed=3)

    @test state.reg === reg
    @test state.data === data
    observed = Symbol[]
    result = LambdaRegression.Workflows.run_mixed_alternation!(state;
        rng=MersenneTwister(8000), cycles=1,
        outer=(enabled=true, attempts=1, evaluations=1, max_nodes=16,
            max_depth=4),
        typed=(enabled=false,), scalar=(enabled=false,),
        sampling=(archive=1, live=1, recent=1, scalar_holes=0),
        max_migrants=0,
        on_observe=(entry, observed_state, selected, materials) -> begin
            @test observed_state === state
            push!(observed, entry.phase)
        end)
    @test result isa NamedTuple
    @test result.state === state
    @test result.best.program == identity_program
    @test :outer_pre in observed
    @test :outer_post in observed
end

end # module Workflows
