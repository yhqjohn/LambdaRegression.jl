"""
Complete-program regularized evolution with rebuildable, nonoverlapping scalar
views. Programs remain authoritative; views belong to exactly one frozen value.
The scalar engine owns selection, archives, maintenance and accounting. This
adapter supplies transactional variation and complete-objective numerical work.
"""

import ..ProgramAnalysis
using ..Programs: with_constants

export JointSearchSession, initialize_joint_search, advance_joint_search!
export inject_joint_programs!, joint_search_result

struct _JointValue
    program::Program
    regions::Vector{ScalarRegion}
    nodes::Vector{ScalarSearch._NodeScalarCandidate}
    skeleton::AstTree
    ports::Vector{Vector{AstNode}}
    outside_values::Dict{Symbol,Any}
    complexity::Int
    id::Int
    parent_ids::Tuple{Vararg{Int}}
    origin::Symbol
end

mutable struct _JointDomain{F}
    target_ty::Ty
    type_context::Dict{Symbol,TyLike}
    input_types::Dict{Symbol,TyLike}
    named_input_policy::Symbol
    max_nodes::Int
    max_depth::Int
    group_exchange_probability::Float64
    replacement::Symbol
    rtr_window::Int
    global_proposal::F
    global_probability::Float64
    pending_selection::Union{Nothing,Tuple{Int,Int}}
    next_id::Int
    counters::Dict{Symbol,Int}
    events::Vector{NamedTuple}
    event_capacity::Int
end

_joint_count!(domain, key) = (domain.counters[key] = get(domain.counters, key, 0) + 1)

function _joint_event!(domain, entry)
    domain.event_capacity == 0 && return
    push!(domain.events, entry)
    length(domain.events) > domain.event_capacity && popfirst!(domain.events)
    return nothing
end

function _joint_value(domain::_JointDomain, program::Program;
                      parent_ids=(), origin::Symbol=:initial)
    context = _merged_type_context(program, domain.type_context, domain.input_types)
    source = prepare_source(program.tree, context, domain.target_ty;
        max_nodes=domain.max_nodes, max_depth=domain.max_depth)
    source isa Err && return source
    # prepare_source imports owned syntax. All occurrence evidence and region
    # views must reference that exact published tree, as in the scalar bridge.
    program = _publish_program(source.index.tree, copy(program.local_consts),
        copy(program.local_consts_ty))
    index = scalar_region_index(program, source;
        type_context=domain.type_context, input_types=domain.input_types,
        named_input_policy=domain.named_input_policy)
    regions = ScalarRegion[]
    occurrence = 1
    while occurrence <= length(source.index.nodes)
        if ScalarRegions.is_scalar_region(index, occurrence)
            path = ProgramAnalysis.path(source, occurrence)
            region = prepare_scalar_region(program, domain.target_ty,
                domain.type_context, path; input_types=domain.input_types,
                named_input_policy=domain.named_input_policy, analysis=source,
                max_nodes=domain.max_nodes, max_depth=domain.max_depth)
            region isa Err && return region
            push!(regions, region)
            occurrence = source.index.subtree_end[occurrence]
        else
            occurrence += 1
        end
    end
    skeleton = program.tree.root
    for (slot, region) in enumerate(regions)
        skeleton = _path_copy_replace(skeleton, region.path, 1,
            ConstNode(Symbol("__joint_hole_", slot)))
        skeleton isa Err && return skeleton
    end
    outside_names = Set{Symbol}()
    function collect_names(node)
        node isa ConstNode && push!(outside_names, node.name)
        for child in children(node)
            collect_names(child)
        end
    end
    collect_names(skeleton)
    outside = Dict{Symbol,Any}(name => (
        program.local_consts_ty[name],
        program.local_consts[name] isa AbstractFloat &&
            program.local_consts_ty[name] == FloatTy ?
            (:floating_parameter, typeof(program.local_consts[name])) :
            program.local_consts[name])
        for name in outside_names if haskey(program.local_consts, name))
    domain.next_id += 1
    return _JointValue(program, regions,
        [ScalarSearch._node_scalar_candidate(region.expr) for region in regions],
        AstTree(skeleton), [region.dependencies for region in regions], outside,
        length(source.index.nodes), domain.next_id, Tuple(parent_ids), origin)
end

# This is deliberately stricter than an observational structural bucket. The
# same literal outside tree, region paths, dependency replacements and outside
# values are required. Every proposed child is checked again in the real world.
function _joint_same_template(first::_JointValue, second::_JointValue)
    length(first.regions) == length(second.regions) || return false
    first.skeleton == second.skeleton || return false
    isequal(first.outside_values, second.outside_values) || return false
    for slot in eachindex(first.regions)
        first.regions[slot].path == second.regions[slot].path || return false
        first.ports[slot] == second.ports[slot] || return false
    end
    return true
end

function _joint_local_config(config, region)
    unary = Tuple(op for op in config.unary_operators if op in region.unary_operators)
    binary = Tuple(op for op in config.binary_operators if op in region.binary_operators)
    isempty(unary) && isempty(binary) && return nothing
    options = (; (name => getfield(config, name) for name in fieldnames(ScalarSearchConfig))...)
    return ScalarSearchConfig(; merge(options,
        (; unary_operators=unary, binary_operators=binary))...)
end

"""Apply all selected, disjoint region edits before constructing or scoring a child.

The frozen recipient's exact Float/port certificates anchor each edit. Fresh
numeric symbols are shared across this transaction's namespace. Full bounds
and typing are rechecked once after all replacements; failure returns `Err`.
"""
function _joint_replace(domain, parent::_JointValue, edits;
                        parent_ids=(parent.id,), origin=:local)
    root = parent.program.tree.root
    values = copy(parent.program.local_consts)
    types = copy(parent.program.local_consts_ty)
    reserved = Set{Symbol}(keys(domain.type_context))
    union!(reserved, keys(domain.input_types))
    _collect_namespace!(reserved, root)
    next_index = Ref(0)
    for (slot, expression) in edits
        region = parent.regions[slot]
        ScalarSearch._same_scalar_expression(region.expr, expression) && continue
        replacement = scalar_to_ast(expression, region.dependencies;
            materialize_constant=value -> _fresh_region_constant(Float64(value),
                reserved, values, types, next_index))
        replacement isa Err && return replacement
        root = _path_copy_replace(root, region.path, 1, replacement)
        root isa Err && return root
    end
    # Drop unreachable numeric/other local materials, retaining declarations
    # together; otherwise repeated rewrites grow an invisible parameter store.
    used = _collect_namespace!(Set{Symbol}(), root)
    filter!(pair -> first(pair) in used, values)
    filter!(pair -> first(pair) in used, types)
    child = _publish_program(AstTree(root), values, types)
    return _joint_value(domain, child; parent_ids, origin)
end

"""Copy whole corresponding regions with one donor-symbol map per transaction.

The strict template certifies literal port replacements, so bound indices can
be transported without inventing a general multi-fragment binding matcher.
Shared donor constants retain one symbol across all copied regions. A single
source definition remains one editable occurrence, regardless of call count.
"""
function _joint_exchange_regions(domain, recipient::_JointValue,
        donor::_JointValue, slots; origin)
    _joint_same_template(recipient, donor) || return _region_error(:joint_template)
    values, types = copy(recipient.program.local_consts), copy(recipient.program.local_consts_ty)
    reserved = _collect_namespace!(Set{Symbol}(), recipient.program.tree.root)
    union!(reserved, keys(domain.type_context), keys(domain.input_types), keys(values))
    mapping = Dict{Symbol,Symbol}()
    function import_node(node)
        if node isa ConstNode && haskey(donor.program.local_consts, node.name)
            name = get!(mapping, node.name) do
                index = length(mapping) + 1
                candidate = Symbol("__joint_donor_", index)
                while candidate in reserved
                    index += 1
                    candidate = Symbol("__joint_donor_", index)
                end
                push!(reserved, candidate)
                values[candidate] = donor.program.local_consts[node.name]
                types[candidate] = donor.program.local_consts_ty[node.name]
                candidate
            end
            return ConstNode(name)
        elseif node isa AppNode
            return AppNode(import_node(node.func), import_node(node.arg))
        elseif node isa AbsNode
            return AbsNode(node.name, import_node(node.body))
        elseif node isa FunctionDefNode
            return FunctionDefNode(import_node(node.body))
        end
        return node # frozen leaves may be shared read-only
    end
    root = recipient.program.tree.root
    for slot in slots
        region = donor.regions[slot]
        subtree = region.source.index.nodes[region.hole.occurrence]
        root = _path_copy_replace(root, recipient.regions[slot].path, 1, import_node(subtree))
        root isa Err && return root
    end
    used = _collect_namespace!(Set{Symbol}(), root)
    filter!(pair -> first(pair) in used, values)
    filter!(pair -> first(pair) in used, types)
    return _joint_value(domain, _publish_program(AstTree(root), values, types);
        parent_ids=(recipient.id, donor.id), origin)
end

ScalarSearch._regularized_copy_value(::_JointDomain, value::_JointValue) = value

function _joint_scalar_names(value::_JointValue)
    used = _collect_namespace!(Set{Symbol}(), value.program.tree.root)
    return sort!([name for (name, kind) in numeric_parameter_blocks(value.program)
        if kind === :scalar && name in used]; by=String)
end
ScalarSearch._regularized_has_constants(::_JointDomain, value::_JointValue) =
    !isempty(_joint_scalar_names(value))

function ScalarSearch._regularized_refine_value(domain::_JointDomain,
        value::_JointValue, loss, score, config, remaining; rng, restarts=2)
    names = _joint_scalar_names(value)
    isempty(names) && return (value, loss, 0, 0)
    initial = Float64[value.program.local_consts[name] for name in names]
    function objective(parameters)
        program = with_constants(value.program,
            Dict(name => parameter for (name, parameter) in zip(names, parameters)))
        # Only program and complexity are read by score. Borrowed views are
        # never varied, retained or published during this numerical trial.
        trial = _JointValue(program, value.regions, value.nodes, value.skeleton,
            value.ports, value.outside_values, value.complexity, value.id,
            (value.id,), :constant)
        return score(trial)
    end
    parameters, refined_loss, count, invalid = ScalarSearch._refine_parameter_vector(initial, loss,
        objective, config, remaining; rng, restarts)
    refined_loss < loss || return (value, loss, count, invalid)
    program = with_constants(value.program,
        Dict(name => parameter for (name, parameter) in zip(names, parameters)))
    # origin records the structural proposal; the refinement event separately
    # records numerical adaptation, so polishing does not hide a global child.
    child = _joint_value(domain, program; parent_ids=(value.id,), origin=value.origin)
    child isa Err && error("scalar parameter adaptation invalidated its type contract")
    return (child, refined_loss, count, invalid)
end

function ScalarSearch._regularized_simplify_value(domain::_JointDomain, value::_JointValue)
    edits = Pair{Int,ScalarExpr}[]
    for (slot, region) in enumerate(value.regions)
        expression = ScalarSearch._simplify_scalar(region.expr)
        ScalarSearch._same_scalar_expression(expression, region.expr) ||
            push!(edits, slot => expression)
    end
    isempty(edits) && return nothing
    child = _joint_replace(domain, value, edits; origin=:simplification)
    return child isa Err ? nothing : child
end

function ScalarSearch._regularized_sample_mutation(domain::_JointDomain,
        rng, value::_JointValue, config)
    domain.pending_selection = nothing
    isempty(value.nodes) && return :restart
    slot = rand(rng, eachindex(value.nodes))
    domain.pending_selection = (value.id, slot)
    local_config = _joint_local_config(config, value.regions[slot])
    local_config === nothing && return :do_nothing
    return ScalarSearch._node_scalar_sample_mutation(rng, value.nodes[slot], local_config)
end

function _joint_global_child(domain, rng, parent, donor)
    _joint_count!(domain, :global_attempts)
    domain.global_proposal === nothing && return (parent, false)
    program = domain.global_proposal(parent.program, donor.program, rng)
    (program === nothing || program isa Err) && return (parent, false)
    program isa Program || throw(ArgumentError("global_proposal must return Program, Err or nothing"))
    program.tree == parent.program.tree &&
        isequal(program.local_consts, parent.program.local_consts) && return (parent, false)
    child = _joint_value(domain, program; parent_ids=(parent.id, donor.id), origin=:global)
    child isa Err && return (parent, false)
    _joint_count!(domain, :global_changed)
    return (child, true)
end

function ScalarSearch._regularized_named_child(domain::_JointDomain, rng,
        parent::_JointValue, donor::_JointValue, feature_count, config, kind;
        temperature=1.0)
    selection = domain.pending_selection
    domain.pending_selection = nothing
    kind in (:simplify, :do_nothing, :optimize) && return (parent, false)
    if isempty(parent.regions) || (domain.global_probability > 0 &&
            rand(rng) < domain.global_probability)
        return _joint_global_child(domain, rng, parent, donor)
    end
    slot = selection !== nothing && selection[1] == parent.id ? selection[2] :
        rand(rng, eachindex(parent.regions))
    region = parent.regions[slot]
    local_config = _joint_local_config(config, region)
    local_config === nothing && return (parent, false)
    donor_node = _joint_same_template(parent, donor) ? donor.nodes[slot] : parent.nodes[slot]
    proposed, changed = ScalarSearch._node_regularized_named_child(rng,
        parent.nodes[slot], donor_node, length(region.dependencies), local_config, kind;
        temperature)
    changed || return (parent, false)
    expression = ScalarSearch._node_scalar_materialize(proposed)
    expression isa Err && return (parent, false)
    child = _joint_replace(domain, parent, [slot => expression];
        parent_ids=(parent.id, donor.id), origin=:local)
    child isa Err && return (parent, false)
    _joint_count!(domain, :local_changed)
    return (child, true)
end

function ScalarSearch._regularized_crossover_pair(domain::_JointDomain, rng,
        first::_JointValue, second::_JointValue, config)
    if !_joint_same_template(first, second) || isempty(first.regions)
        _joint_count!(domain, :template_mismatch)
        # The enclosing crossover retry loop can invoke this seam repeatedly.
        # A failed template match is not an automatic broad-search opportunity.
        return (first, second, false)
    end
    slots = collect(eachindex(first.regions))
    shuffle!(rng, slots)
    if rand(rng) < domain.group_exchange_probability
        resize!(slots, min(length(slots), rand(rng, 1:2)))
        origin = length(slots) == length(first.regions) ?
            :configuration_exchange : (length(slots) == 2 ? :pair_exchange : :region_exchange)
        a = _joint_exchange_regions(domain, first, second, slots; origin)
        b = _joint_exchange_regions(domain, second, first, slots; origin)
    else
        slot = slots[1]
        local_config = _joint_local_config(config, first.regions[slot])
        local_config === nothing && return (first, second, false)
        a, b, changed = ScalarSearch._node_scalar_crossover_pair(rng,
            first.nodes[slot], second.nodes[slot], local_config)
        changed || return (first, second, false)
        edits_a = [slot => ScalarSearch._node_scalar_materialize(a)]
        edits_b = [slot => ScalarSearch._node_scalar_materialize(b)]
        origin = :subtree_exchange
        a = _joint_replace(domain, first, edits_a;
            parent_ids=(first.id, second.id), origin)
        b = _joint_replace(domain, second, edits_b;
            parent_ids=(second.id, first.id), origin)
    end
    # Avoid copying the donor as an alleged recombination when every variable
    # region was exchanged. Such whole-configuration exchange is migration.
    (a isa Err || b isa Err) && return (first, second, false)
    same_a = a.program.tree == first.program.tree &&
        isequal(a.program.local_consts, first.program.local_consts)
    same_b = b.program.tree == second.program.tree &&
        isequal(b.program.local_consts, second.program.local_consts)
    same_a && same_b && return (first, second, false)
    _joint_count!(domain, origin)
    return (a, b, true)
end

function _joint_distance(a::_JointValue, b::_JointValue)
    _joint_same_template(a, b) || return typemax(Int) ÷ 2
    return sum(!ScalarSearch._same_scalar_expression(x.expr, y.expr)
        for (x, y) in zip(a.regions, b.regions); init=0)
end

function ScalarSearch._regularized_replacement_index(domain::_JointDomain,
        state, parent, child)
    domain.replacement === :age && return ScalarSearch._node_oldest_index(state.population)
    slots = randperm(state.rng, length(state.population))[
        1:min(domain.rtr_window, length(state.population))]
    distances = [_joint_distance(child.value, state.population[slot].candidate.value)
        for slot in slots]
    slot = slots[argmin(distances)]
    return child.loss <= state.population[slot].candidate.loss ? slot : nothing
end

function ScalarSearch._regularized_observe(domain::_JointDomain,
        state, event, parent, child, admitted)
    _joint_count!(domain, admitted ? :admitted : :not_admitted)
    child.value.origin === :global && admitted && _joint_count!(domain, :global_admitted)
    _joint_event!(domain, (; event, id=child.value.id,
        parent_ids=child.value.parent_ids, origin=child.value.origin,
        loss=child.loss, complexity=child.value.complexity, admitted))
    return nothing
end

"""Owned fixed-capacity islands for one stable task; serial, in-memory only."""
mutable struct JointSearchSession
    domain::_JointDomain
    states::Vector{Any}
    cursor::Int
    migrations::Int
end

"""
    initialize_joint_search(programs, target_ty, type_context, score; config, ...)

Build a fixed number of complete-program islands. Budgets are total across
islands and include initialization; each island has `config.population_size`
members, regardless of the number of templates or regions. Input programs are
copied at import. `score(program)` is synchronous and must keep the task/world
stable. Nodes outside the scalar bridge remain eligible for global proposals.
`config.max_complexity` bounds each scalar region; `max_nodes` bounds the whole
AST and indexes complete-complexity maintenance. No independent hole population
is created. Invalid initial syntax or an insufficient budget fails explicitly.
"""
function initialize_joint_search(programs, target_ty, type_context, score;
        config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
        rng::AbstractRNG=default_rng(), island_count::Int=2,
        input_types=Dict(), named_input_policy::Symbol=:declared,
        max_nodes::Int=2048, max_depth::Int=256,
        group_exchange_probability::Real=0.25, replacement::Symbol=:age,
        rtr_window::Int=8, global_proposal=nothing, global_probability::Real=0.02,
        ncycles_per_iteration::Int=64, frequency_window::Int=100000,
        loss_normalization::Real=1.0, event_capacity::Int=256)
    isempty(programs) && throw(ArgumentError("joint search requires initial programs"))
    config.evolution_strategy === :regularized ||
        throw(ArgumentError("joint search requires regularized maintenance"))
    island_count >= 1 || throw(ArgumentError("island_count must be positive"))
    replacement in (:age, :rtr) || throw(ArgumentError("replacement must be :age or :rtr"))
    rtr_window > 0 || throw(ArgumentError("rtr_window must be positive"))
    all(p -> isfinite(p) && 0 <= p <= 1,
        (group_exchange_probability, global_probability)) || throw(ArgumentError("invalid proposal probability"))
    event_capacity >= 0 || throw(ArgumentError("event_capacity must be nonnegative"))
    ncycles_per_iteration > 0 || throw(ArgumentError("maintenance period must be positive"))
    isfinite(loss_normalization) && loss_normalization > 0 ||
        throw(ArgumentError("loss_normalization must be finite and positive"))
    _valid_bounds(max_nodes, max_depth)
    domain = _JointDomain(target_ty, Dict{Symbol,TyLike}(type_context),
        Dict{Symbol,TyLike}(input_types), named_input_policy, max_nodes, max_depth,
        Float64(group_exchange_probability), replacement, rtr_window,
        global_proposal, Float64(global_probability), nothing, 0, Dict{Symbol,Int}(),
        NamedTuple[], event_capacity)
    initial = _JointValue[]
    for program in programs
        owned = Program(program.tree; local_consts=program.local_consts,
            local_consts_ty=program.local_consts_ty)
        value = _joint_value(domain, owned)
        value isa Err && throw(ArgumentError("invalid joint initial program: $(value.value)"))
        push!(initial, value)
    end
    structural = ScalarSearch._split_budget(config.max_evaluations, island_count)
    constants = ScalarSearch._split_budget(config.max_constant_refinement_evaluations, island_count)
    totals = config.max_total_evaluations === nothing ? fill(nothing, island_count) :
        ScalarSearch._split_budget(config.max_total_evaluations, island_count)
    flow = ScalarSearch._NodeRegularizedFlow(nothing, true, 1.0, 3.17,
        ncycles_per_iteration, true, 0.0614, 0.982, true, 1040.0)
    frequency = ScalarSearch._RegularizedFrequency(frequency_window, max_nodes)
    states = Any[]
    function objective(value)
        result = score(value.program)
        result isa Err && return (Inf, true)
        result isa Real || throw(ArgumentError("score must return a real loss or Err"))
        loss = Float64(result)
        return isfinite(loss) ? (loss, false) : (Inf, true)
    end
    for island in 1:island_count
        island_config = ScalarSearch._copy_config(config, structural[island],
            constants[island], totals[island])
        island_rng = MersenneTwister(rand(rng, UInt64))
        values = _JointValue[]
        for slot in 1:config.population_size
            position = (island - 1) * config.population_size + slot
            seed = initial[mod1(position, length(initial))]
            value = seed
            if position > length(initial) && !isempty(seed.regions)
                edits = Pair{Int,ScalarExpr}[]
                for (region_slot, region) in enumerate(seed.regions)
                    local_config = _joint_local_config(config, region)
                    local_config === nothing && continue
                    node = ScalarSearch._node_random_grow(island_rng,
                        length(region.dependencies), local_config, config.initial_depth)
                    if ScalarSearch.DynamicExpressions.count_nodes(node) > config.max_complexity
                        node = ScalarSearch._node_random_fixed_size(island_rng,
                            length(region.dependencies), local_config, config.max_complexity)
                    end
                    expression = ScalarSearch._node_scalar_materialize(
                        ScalarSearch._node_scalar_candidate(node))
                    expression isa Err || push!(edits, region_slot => expression)
                end
                proposed = _joint_replace(domain, seed, edits; origin=:initial_random)
                proposed isa Err || (value = proposed)
            end
            push!(values, value)
        end
        state = ScalarSearch._initialize_regularized_values(values, objective;
            config=island_config, rng=island_rng, flow, domain, frequency,
            loss_normalization=Float64(loss_normalization))
        push!(states, state)
    end
    return JointSearchSession(domain, states, 1, 0)
end

_joint_terminal(state) = state.evaluations >= state.config.max_evaluations ||
    ScalarSearch._node_state_total_remaining(state) <= 0

"""Advance existing islands under incremental structural and attempt limits.

Maintenance remains on the mature attempt clock. Constants have independent
fixed caps. Optional migration uses frozen same-task archive/live material and
fresh ages, without constructing any per-template populations.
"""
function advance_joint_search!(session::JointSearchSession;
        max_evaluations::Int=128, max_attempts::Int=2048, migrate::Bool=false)
    max_evaluations >= 0 && max_attempts >= 0 || throw(ArgumentError("limits must be nonnegative"))
    start = sum(state.evaluations for state in session.states)
    for _ in 1:max_attempts
        sum(state.evaluations for state in session.states) - start >= max_evaluations && break
        all(_joint_terminal, session.states) && break
        state = session.states[session.cursor]
        session.cursor = mod1(session.cursor + 1, length(session.states))
        _joint_terminal(state) && continue
        # A crossover requires two evaluations. The mature loop honors this
        # temporary ceiling without silently overspending a window.
        limit = min(state.config.max_evaluations,
            state.evaluations + max_evaluations -
                (sum(s.evaluations for s in session.states) - start))
        ScalarSearch._node_state_one_offspring!(state, limit)
    end
    if migrate && length(session.states) > 1
        rngs = [state.rng for state in session.states]
        incoming, tops = ScalarSearch._inject_migrants!(session.states, rngs,
            1, 1, 0.0, 12; skip_terminal_destinations=true,
            hof_donor_policy=:global_pareto)
        session.migrations += sum(incoming) + sum(tops)
    end
    return session
end

"""Check, score and inject bounded external programs, charging structural work."""
function inject_joint_programs!(session::JointSearchSession, programs;
        island::Int=1, max_count::Int=2)
    state = session.states[island]
    max_count >= 0 || throw(ArgumentError("max_count must be nonnegative"))
    accepted = scored = refused = 0
    for program in Iterators.take(programs, max_count)
        _joint_terminal(state) && break
        value = _joint_value(session.domain, Program(program.tree;
            local_consts=program.local_consts, local_consts_ty=program.local_consts_ty);
            origin=:import)
        if value isa Err
            refused += 1
            continue
        end
        candidate = ScalarSearch._node_state_score_candidate!(state,
            value, true, state.config.max_evaluations)
        candidate === nothing && break
        scored += 1
        isfinite(candidate.loss) || (refused += 1; continue)
        ScalarSearch._node_state_record!(state, candidate; source=:import)
        ScalarSearch._node_inject_candidate!(state, candidate)
        accepted += 1
    end
    return (; accepted, scored, refused)
end

function _joint_report_candidate(candidate)
    value = candidate.value
    return (; program=Program(value.program.tree;
        local_consts=value.program.local_consts, local_consts_ty=value.program.local_consts_ty),
        loss=candidate.loss, complexity=value.complexity, id=value.id,
        parent_ids=value.parent_ids, origin=value.origin)
end

"""Publish owned complete programs and bounded observational history; scores nothing."""
function joint_search_result(session::JointSearchSession)
    states = session.states
    entries = collect(Iterators.flatten(values(state.archive) for state in states))
    append!(entries, (member.candidate for state in states for member in state.population))
    sort!(entries; by=ScalarSearch._node_candidate_key)
    seen = Set{Int}()
    candidates = NamedTuple[]
    for candidate in entries
        candidate.value.id in seen && continue
        isfinite(candidate.loss) || continue
        push!(seen, candidate.value.id)
        push!(candidates, _joint_report_candidate(candidate))
    end
    return (; best=isempty(candidates) ? nothing : first(candidates), candidates,
        population=[[_joint_report_candidate(member.candidate) for member in state.population]
            for state in states],
        evaluations=sum(state.evaluations for state in states),
        constant_refinement_evaluations=sum(state.constant_refinement_evaluations for state in states),
        invalid_evaluations=sum(state.invalid_evaluations for state in states),
        constant_refinement_invalid_evaluations=sum(state.constant_refinement_invalid_evaluations for state in states),
        attempts=sum(state.offspring_attempts for state in states),
        completed=all(_joint_terminal, states), migrations=session.migrations,
        counters=copy(session.domain.counters), events=copy(session.domain.events))
end

@testsnippet JointSearchFixture begin
    using Test, Random
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, TyFun, TyLike
    using LambdaRegression.Regression
    using LambdaRegression.ScalarSearch
    const R = LambdaRegression.Regression
    const S = LambdaRegression.ScalarSearch
    const pair_ty = TyArray(TyNat(2), FloatTy)
    const joint_types = Dict{Symbol,TyLike}(
        :PAIR => TyFun(FloatTy, TyFun(FloatTy, pair_ty)),
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :FMUL => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    function pair_program(a, b)
        Program(AstTree(AppNode(AppNode(ConstNode(:PAIR), ConstNode(:a)), ConstNode(:b)));
            local_consts=Dict(:a => Float64(a), :b => Float64(b)),
            local_consts_ty=Dict{Symbol,TyLike}(:a => FloatTy, :b => FloatTy))
    end
    function joint_config(; kwargs...)
        ScalarSearchConfig(; merge((; population_size=4, max_evaluations=40,
            max_complexity=8, initial_depth=1, tournament_size=2, elite_count=1,
            unary_operators=(), binary_operators=(:+, :*),
            constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
            evolution_strategy=:regularized), (; kwargs...))...)
    end
end

@testitem "joint configuration atomic edits preserve parents and exact port templates" setup=[JointSearchFixture] begin
    parent = pair_program(1, 2)
    session = initialize_joint_search([parent], pair_ty, joint_types, p -> 1.0;
        config=joint_config(), island_count=1, rng=MersenneTwister(1))
    value = session.states[1].population[1].candidate.value
    @test length(value.regions) == 2
    child = R._joint_replace(session.domain, value,
        [1 => scalar_constant(7), 2 => scalar_constant(9)]; origin=:pair_exchange)
    @test child isa R._JointValue
    @test [region.expr.value for region in child.regions] == [7.0, 9.0]
    @test [region.expr.value for region in value.regions] == [1.0, 2.0]
    @test parent.local_consts == Dict(:a => 1.0, :b => 2.0)
    @test R._joint_same_template(value, child)
    @test child.complexity == value.complexity
    @test R._joint_replace(session.domain, value,
        [1 => scalar_input(1)]) isa LambdaRegression.Utils.Err
end

@testitem "joint parameter adaptation preserves shared symbols and actual score budget" setup=[JointSearchFixture] begin
    shared = Program(AstTree(AppNode(AppNode(ConstNode(:PAIR), ConstNode(:a)), ConstNode(:a)));
        local_consts=Dict(:a => 0.0), local_consts_ty=Dict{Symbol,TyLike}(:a => FloatTy))
    config = joint_config(constant_refinement_iterations=6,
        max_constant_refinement_evaluations=20)
    calls = Ref(0)
    score(p) = (calls[] += 1; (get(p.local_consts, :a, 0.0) - 2.0)^2)
    session = initialize_joint_search([shared], pair_ty, joint_types, score;
        config, island_count=1)
    value = session.states[1].population[1].candidate.value
    @test R._joint_scalar_names(value) == [:a]
    refined, loss, count, invalid = S._regularized_refine_value(session.domain,
        value, 4.0, session.states[1].score, config, 20; rng=MersenneTwister(1), restarts=0)
    @test 0 < count <= 20
    @test calls[] == config.population_size + count
    @test loss < 1e-6
    @test keys(refined.program.local_consts) == keys(shared.local_consts)
    @test shared.local_consts[:a] == 0.0
    @test refined.program.tree == shared.tree
    @test invalid == 0
end

@testitem "joint whole region transport keeps donor sharing and persistent ports" setup=[JointSearchFixture] begin
    shared = Program(AstTree(AppNode(AppNode(ConstNode(:PAIR), ConstNode(:a)), ConstNode(:a)));
        local_consts=Dict(:a => 3.0), local_consts_ty=Dict{Symbol,TyLike}(:a => FloatTy))
    session = initialize_joint_search([pair_program(1, 2), shared], pair_ty,
        joint_types, p -> 1.0; config=joint_config(), island_count=1)
    recipient = session.states[1].population[1].candidate.value
    donor = session.states[1].population[2].candidate.value
    child = R._joint_exchange_regions(session.domain, recipient, donor, [1, 2];
        origin=:configuration_exchange)
    @test child isa R._JointValue
    @test child.program.tree.root.func.arg.name == child.program.tree.root.arg.name
    @test length(R._joint_scalar_names(child)) == 1
    @test only(values(child.program.local_consts)) == 3.0
    @test recipient.program.local_consts == Dict(:a => 1.0, :b => 2.0)
    @test donor.program.local_consts == Dict(:a => 3.0)

    scalar = Program(AstTree(ConstNode(:x)))
    inputs = Dict{Symbol,TyLike}(:x => FloatTy, :z => FloatTy)
    scalar_session = initialize_joint_search([scalar], FloatTy, joint_types,
        p -> 1.0; config=joint_config(), island_count=1, input_types=inputs)
    value = scalar_session.states[1].population[1].candidate.value
    changed = R._joint_replace(scalar_session.domain, value, [1 => scalar_constant(0.0)])
    @test length(value.regions[1].dependencies) == 2
    @test changed.regions[1].dependencies == value.regions[1].dependencies
    @test R._joint_same_template(value, changed)
end

@testitem "joint search fixed capacity budget publication and global offspring" setup=[JointSearchFixture] begin
    calls = Ref(0)
    score(p) = (calls[] += 1; 1.0)
    global_calls = Ref(0)
    global_proposal = (parent, donor, rng) -> begin
        global_calls[] += 1
        pair_program(rand(rng), rand(rng))
    end
    session = initialize_joint_search([pair_program(1, 2)], pair_ty, joint_types, score;
        config=joint_config(max_total_evaluations=40), island_count=2,
        rng=MersenneTwister(8000), global_probability=1.0, global_proposal)
    advance_joint_search!(session; max_evaluations=12, max_attempts=100)
    report = joint_search_result(session)
    @test calls[] == report.evaluations + report.constant_refinement_evaluations
    @test report.evaluations <= 20 # eight initial + at most twelve new calls
    @test global_calls[] > 0
    @test get(report.counters, :global_admitted, 0) > 0
    @test sum(length, report.population) == 8
    before = calls[]
    report.best.program.local_consts[:a] = -100.0
    @test joint_search_result(session).best.program.local_consts[:a] != -100.0
    @test calls[] == before
    for _ in 1:4
        advance_joint_search!(session; max_evaluations=40, max_attempts=1000)
    end
    @test joint_search_result(session).completed
    @test calls[] <= 40
    @test calls[] == joint_search_result(session).evaluations
end

@testitem "joint initialization admits extended operators only at matching Float signatures" begin
    using Test, Random
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FTAN, FEXP, FLOG, FABS, FPOW, FMAX, FMIN
    using LambdaRegression.Regression: initialize_joint_search
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyFun, TyLike, get_type

    apply_named(name, args...) = foldl(
        (function_node, argument) -> AppNode(function_node, argument),
        args; init=ConstNode(name),
    )
    x = IndexNode(0)
    expression = apply_named(:FMAX,
        apply_named(:FPOW, apply_named(:FTAN, x), apply_named(:FABS, x)),
        apply_named(:FMIN, apply_named(:FEXP, apply_named(:FABS, x)),
            apply_named(:FLOG, apply_named(:FABS, x))),
    )
    program = Program(AstTree(AbsNode(:x, expression)))
    target = TyFun(FloatTy, FloatTy)
    context = Dict{Symbol,TyLike}(
        :FTAN => get_type(FTAN), :FEXP => get_type(FEXP),
        :FLOG => get_type(FLOG), :FABS => get_type(FABS),
        :FPOW => get_type(FPOW), :FMAX => get_type(FMAX),
        :FMIN => get_type(FMIN),
    )
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=2, max_total_evaluations=2,
        tournament_size=2, elite_count=1,
        max_complexity=16, initial_depth=1,
        unary_operators=(:tan, :exp, :log, :abs),
        binary_operators=(:^, :max, :min),
        constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
        evolution_strategy=:regularized,
    )
    session = initialize_joint_search([program], target, context, _ -> 1.0;
        config, island_count=1, rng=MersenneTwister(1234))
    value = session.states[1].population[1].candidate.value
    @test length(value.regions) == 1
    region = only(value.regions)
    @test region.unary_operators == Set((:tan, :exp, :log, :abs))
    @test region.binary_operators == Set((:^, :max, :min))

    wrong_context = Dict{Symbol,TyLike}(
        :FTAN => TyFun(IntTy, IntTy),
        :FPOW => get_type(FTAN),
    )
    wrong_unary, wrong_binary = LambdaRegression.ScalarRegions._operator_sets(wrong_context)
    @test :tan ∉ wrong_unary
    @test :^ ∉ wrong_binary
end
