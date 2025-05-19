"""
Complete-program regularized evolution with rebuildable, nonoverlapping scalar
views. Programs remain authoritative; views belong to exactly one frozen value.
The scalar engine owns selection, archives, maintenance and accounting. This
adapter supplies transactional variation and complete-objective numerical work.
Tournament frequency pressure uses complete-program complexity; it is not a
measure of higher-order structure-family diversity.
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
    function_exposure_probability::Float64
    program_simplification_steps::Int
    pending_selection::Union{Nothing,Tuple{Int,Int}}
    next_id::Int
    counters::Dict{Symbol,Int}
    events::Vector{NamedTuple}
    event_capacity::Int
    fixed_template::Union{Nothing,_JointValue}
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
    value = _JointValue(program, regions,
        [ScalarSearch._node_scalar_candidate(region.expr) for region in regions],
        AstTree(skeleton), [region.dependencies for region in regions], outside,
        length(source.index.nodes), domain.next_id, Tuple(parent_ids), origin)
    if domain.fixed_template !== nothing &&
            !_joint_same_template(domain.fixed_template, value)
        _joint_count!(domain, :fixed_template_rejections)
        return _region_error(:joint_fixed_template)
    end
    return value
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

function ScalarSearch._regularized_simplify_value(domain::_JointDomain, value::_JointValue, rng)
    edits = Pair{Int,ScalarExpr}[]
    for (slot, region) in enumerate(value.regions)
        expression = ScalarSearch._simplify_scalar(region.expr)
        ScalarSearch._same_scalar_expression(expression, region.expr) ||
            push!(edits, slot => expression)
    end
    child = isempty(edits) ? value :
        _joint_replace(domain, value, edits; origin=:simplification)
    child isa Err && return nothing
    for _ in 1:domain.program_simplification_steps
        values, types = copy(child.program.local_consts), copy(child.program.local_consts_ty)
        tree = Mutations.mutate_tree(child.program.tree, domain.target_ty;
            rng, global_consts_ty=merge(domain.type_context, domain.input_types),
            local_consts=values, local_consts_ty=types,
            mutations=[Mutations.BetaReduceMutation()],
            chooser=Mutations.WeightedPairChooser(), provider=Mutations.IndexedProvider(),
            max_try=1, bloat_policy=Mutations.BloatPolicy(
                enabled=true, max_nodes=domain.max_nodes, max_nodes_delta=domain.max_nodes))
        tree isa Err && break
        proposed = _joint_value(domain, _publish_program(tree, values, types);
            parent_ids=(value.id,), origin=:simplification)
        proposed isa Err && break
        child = proposed
    end
    # Even a beta contraction can change bounded execution or a custom objective.
    # The maintenance owner scores this complete proposal before replacement.
    return child === value ? nothing : child
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

# Only the complete output's existing lambda prefix is traversed. Four remaining
# concrete arrows may be exposed; a mu/other result never gets unfolded here.
function _joint_function_exposure_plan(domain, parent)
    isempty(parent.regions) || return nothing
    node, ty = parent.program.tree.root, TypeSystem.prune(domain.target_ty)
    path = Int[]
    while node isa AbsNode
        ty isa TyFun || return nothing
        push!(path, 1)
        node, ty = node.body, TypeSystem.prune(ty.res)
    end
    for steps in 1:4
        ty isa TyFun || return nothing
        ty = TypeSystem.prune(ty.res)
        ty == FloatTy && return (; path, steps)
    end
    return nothing
end

"""Propose one owned output-lambda exposure using existing typed eta rewrites.

The preflight plan anchors the complete program's output prefix, not arbitrary
callbacks. Each step prepares fresh typed sites and checks complete bounds;
failure discards every intermediate. No score or parent syntax is reused as a
candidate certificate. The normal offspring owner scores/adopts the final value.
"""
function _joint_function_exposure_child(domain, rng, parent, plan)
    _joint_count!(domain, :function_exposure_attempts)
    program, path = parent.program, copy(plan.path)
    values, types = copy(program.local_consts), copy(program.local_consts_ty)
    context = _merged_type_context(program, domain.type_context, domain.input_types)
    ctx = (; local_consts=values, local_consts_ty=types,
        global_consts_ty=merge(domain.type_context, domain.input_types),
        tyfactory=TypeSystem.newtyvar, symfactory=SymbolGenerator(),
        const_symfactory=ConstSymbolFactory())
    for _ in 1:plan.steps
        tree = AstTree(deepcopy(program.tree.root))
        env = Mutations.MutationEnv(rng, domain.target_ty, context, tree, nothing,
            false, IdDict{Any,Any}(), Mutations.AdaptivePolicy(),
            IdDict{DataType,Mutations.MutationStat}(), Mutations.BloatPolicy())
        provider = Mutations.IndexedProvider()
        Mutations.prepare!(provider, tree, context, env, ctx) isa Err && return (parent, false)
        sites = Mutations.sites(provider)
        position = findfirst(site -> site.path == path, sites)
        position === nothing && return (parent, false)
        site = Mutations._remap_site(tree, sites[position], env.analysis, ctx)
        site isa Err && return (parent, false)
        mutation = Mutations.EtaExpandMutation()
        Mutations.is_site_eligible(mutation, site, env, ctx) || return (parent, false)
        Mutations.apply(mutation, tree, site, env, ctx) isa Err && return (parent, false)
        source = prepare_source(tree, context, domain.target_ty;
            max_nodes=domain.max_nodes, max_depth=domain.max_depth)
        source isa Err && return (parent, false)
        program = _publish_program(source.index.tree, copy(values), copy(types))
        push!(path, 1)
    end
    child = _joint_value(domain, program;
        parent_ids=(parent.id,), origin=:function_exposure)
    (child isa Err || isempty(child.regions)) && return (parent, false)
    _joint_count!(domain, :function_exposure_changed)
    return (child, true)
end

function ScalarSearch._regularized_named_child(domain::_JointDomain, rng,
        parent::_JointValue, donor::_JointValue, feature_count, config, kind;
        temperature=1.0)
    selection = domain.pending_selection
    domain.pending_selection = nothing
    kind in (:simplify, :do_nothing, :optimize) && return (parent, false)
    if isempty(parent.regions) && domain.function_exposure_probability > 0
        plan = _joint_function_exposure_plan(domain, parent)
        if plan !== nothing && rand(rng) < domain.function_exposure_probability
            return _joint_function_exposure_child(domain, rng, parent, plan)
        end
    end
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
        state, event, parent, child, admitted; new_offspring::Bool=false)
    _joint_count!(domain, admitted ? :admitted : :not_admitted)
    if new_offspring && child.value.origin === :global
        _joint_count!(domain, :global_scored)
        if isfinite(child.loss)
            _joint_count!(domain, :global_valid)
            admitted && _joint_count!(domain, :global_admitted)
        end
    elseif new_offspring && child.value.origin === :function_exposure
        _joint_count!(domain, :function_exposure_scored)
        if isfinite(child.loss)
            _joint_count!(domain, :function_exposure_valid)
            admitted && _joint_count!(domain, :function_exposure_admitted)
        end
    end
    before = parent === nothing ? nothing :
        (; id=parent.value.id, loss=parent.loss,
            complexity=parent.value.complexity)
    _joint_event!(domain, (; event, id=child.value.id,
        parent_ids=child.value.parent_ids, origin=child.value.origin,
        loss=child.loss, complexity=child.value.complexity, admitted, new_offspring,
        before))
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

`seed_per_island=true` places independent copies of the supplied complete
programs, in order, in each island's leading slots (up to its capacity). Other
slots keep the existing randomized initialization. The option changes only the
starting population: later selection may replace those members. The default
`false` preserves the original global slot schedule.

`program_simplification_steps=0` keeps only the existing scalar simplification.
Values 1 or 2 additionally propose that many bounded existing beta mutations
per member at cycle maintenance. Each step rechecks typing and complete AST
bounds; failed steps stop the proposal. The final complete program is freshly
scored against the current task, uses structural budget, and replaces its member
only when non-worse, without refreshing age. No beta equivalence or score reuse
is assumed, including for bounded execution and custom objectives.

`function_exposure_probability=0.0` disables output-function exposure without
an extra RNG draw. A positive probability applies only to parents with no Float
regions and a concrete finite output-function chain: following existing leading
lambdas, at most four existing typed eta expansions expose its Float body. The
whole candidate is freshly scored and ordinarily admitted; recursive/other
result types and longer chains are ineligible. A selected failure consumes the
ordinary attempt and does not retry or fall back to a global proposal.

`fixed_template=true` anchors every constructed value to the first program's
literal outside syntax, exact region paths, dependency replacements and outside
materials via `_joint_same_template`. It requires disabled global proposals,
function exposure and program-level simplification. Scalar initialization,
variation, simplification and numerical publication all recheck this boundary;
the observational structure key grants no reuse rights. A mismatched additional
seed fails initialization; a mismatched offspring is an expected failed edit.

`use_frequency_in_tournament=true` and `adaptive_parsimony_scaling=1040.0`
control the scalar engine's tournament cost adjustment by full-program
complexity frequency. This is a whole-program complexity frequency, not
higher-order structure-family diversity. Turning it off disables only this
tournament adjustment; mutation acceptance continues to use its existing
frequency policy. The scale must convert to a finite nonnegative `Float64`.
"""
function initialize_joint_search(programs, target_ty, type_context, score;
        config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
        rng::AbstractRNG=default_rng(), island_count::Int=2,
        input_types=Dict(), named_input_policy::Symbol=:declared,
        max_nodes::Int=2048, max_depth::Int=256,
        group_exchange_probability::Real=0.25, replacement::Symbol=:age,
        rtr_window::Int=8, global_proposal=nothing, global_probability::Real=0.02,
        function_exposure_probability::Real=0.0,
        program_simplification_steps::Int=0,
        ncycles_per_iteration::Int=64, frequency_window::Int=100000,
        loss_normalization::Real=1.0, event_capacity::Int=256,
        fixed_template::Bool=false, seed_per_island::Bool=false,
        use_frequency_in_tournament::Bool=true,
        adaptive_parsimony_scaling::Real=1040.0)
    isempty(programs) && throw(ArgumentError("joint search requires initial programs"))
    config.evolution_strategy === :regularized ||
        throw(ArgumentError("joint search requires regularized maintenance"))
    island_count >= 1 || throw(ArgumentError("island_count must be positive"))
    replacement in (:age, :rtr) || throw(ArgumentError("replacement must be :age or :rtr"))
    rtr_window > 0 || throw(ArgumentError("rtr_window must be positive"))
    all(p -> isfinite(p) && 0 <= p <= 1,
        (group_exchange_probability, global_probability, function_exposure_probability)) ||
        throw(ArgumentError("invalid proposal probability"))
    event_capacity >= 0 || throw(ArgumentError("event_capacity must be nonnegative"))
    program_simplification_steps in 0:2 || throw(ArgumentError(
        "program_simplification_steps must be 0, 1, or 2"))
    fixed_template && (global_proposal !== nothing || global_probability != 0 ||
        function_exposure_probability != 0 || program_simplification_steps != 0) &&
        throw(ArgumentError("fixed_template requires disabled global/exposure/program simplification"))
    ncycles_per_iteration > 0 || throw(ArgumentError("maintenance period must be positive"))
    isfinite(loss_normalization) && loss_normalization > 0 ||
        throw(ArgumentError("loss_normalization must be finite and positive"))
    frequency_scale = Float64(adaptive_parsimony_scaling)
    isfinite(frequency_scale) && frequency_scale >= 0 ||
        throw(ArgumentError("adaptive_parsimony_scaling must be finite and nonnegative"))
    _valid_bounds(max_nodes, max_depth)
    domain = _JointDomain(target_ty, Dict{Symbol,TyLike}(type_context),
        Dict{Symbol,TyLike}(input_types), named_input_policy, max_nodes, max_depth,
        Float64(group_exchange_probability), replacement, rtr_window,
        global_proposal, Float64(global_probability), Float64(function_exposure_probability),
        program_simplification_steps,
        nothing, 0, Dict{Symbol,Int}(),
        NamedTuple[], event_capacity, nothing)
    initial = _JointValue[]
    for program in programs
        owned = Program(program.tree; local_consts=program.local_consts,
            local_consts_ty=program.local_consts_ty)
        value = _joint_value(domain, owned)
        value isa Err && throw(ArgumentError("invalid joint initial program: $(value.value)"))
        fixed_template && domain.fixed_template === nothing && (domain.fixed_template = value)
        push!(initial, value)
    end
    structural = ScalarSearch._split_budget(config.max_evaluations, island_count)
    constants = ScalarSearch._split_budget(config.max_constant_refinement_evaluations, island_count)
    totals = config.max_total_evaluations === nothing ? fill(nothing, island_count) :
        ScalarSearch._split_budget(config.max_total_evaluations, island_count)
    flow = ScalarSearch._NodeRegularizedFlow(nothing, true, 1.0, 3.17,
        ncycles_per_iteration, true, 0.0614, 0.982,
        use_frequency_in_tournament, frequency_scale)
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
        preserved_seed_count = seed_per_island ? min(length(initial), config.population_size) : 0
        for slot in 1:config.population_size
            position = (island - 1) * config.population_size + slot
            preserved_seed = slot <= preserved_seed_count
            seed_source = preserved_seed ? initial[slot] : initial[mod1(position, length(initial))]
            seed = seed_source
            if preserved_seed
                copied = _joint_value(domain, seed_source.program)
                copied isa Err && throw(ArgumentError(
                    "invalid per-island joint initial program copy: $(copied.value)"))
                seed = copied
            end
            value = seed
            if position > length(initial) && !preserved_seed && !isempty(seed.regions)
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

@testitem "joint initialization optionally seeds every island from complete programs" setup=[JointSearchFixture] begin
    programs = [pair_program(i, i + 0.5) for i in 1:8]
    same_program(a, b) = a.tree == b.tree &&
        a.local_consts == b.local_consts && a.local_consts_ty == b.local_consts_ty
    config = joint_config()
    default = initialize_joint_search(programs, pair_ty, joint_types, _ -> 1.0;
        config, island_count=2, rng=MersenneTwister(42))
    default_report = joint_search_result(default)
    @test length(default_report.population) == 2
    @test length.(default_report.population) == [4, 4]
    @test all(same_program(default_report.population[1][i].program, programs[i]) for i in 1:4)
    @test all(same_program(default_report.population[2][i].program, programs[i + 4]) for i in 1:4)

    calls = Ref(0)
    seeded = initialize_joint_search(programs, pair_ty, joint_types,
        _ -> (calls[] += 1; 1.0); config, island_count=2,
        rng=MersenneTwister(42), seed_per_island=true)
    report = joint_search_result(seeded)
    @test length(report.population) == 2
    @test length.(report.population) == [4, 4]
    @test all(same_program(report.population[island][slot].program, programs[slot])
        for island in 1:2 for slot in 1:4)
    @test all(seeded.states[1].population[slot].candidate.value.program.local_consts !==
        seeded.states[2].population[slot].candidate.value.program.local_consts for slot in 1:4)
    @test all(seeded.states[1].population[slot].candidate.value.regions !==
        seeded.states[2].population[slot].candidate.value.regions for slot in 1:4)
    @test report.evaluations == calls[] == 8
    @test report.attempts == 0
    @test get(report.counters, :admitted, 0) == 8
    @test all(programs[i].local_consts == Dict(:a => Float64(i), :b => Float64(i) + 0.5)
        for i in 1:8)

    one_seed = initialize_joint_search([programs[1]], pair_ty, joint_types, _ -> 1.0;
        config, island_count=2, rng=MersenneTwister(555), seed_per_island=true)
    one_seed_report = joint_search_result(one_seed)
    @test all(same_program(one_seed_report.population[island][1].program, programs[1])
        for island in 1:2)
    randomized_origins = [member.candidate.value.origin
        for (island, state) in enumerate(one_seed.states)
        for (slot, member) in enumerate(state.population)
        if !((island, slot) in ((1, 1), (2, 1)))]
    @test length(randomized_origins) == 6
    @test all(origin in (:initial, :initial_random) for origin in randomized_origins)
    @test one_seed_report.evaluations == 8
end

@testitem "joint tournament frequency controls preserve defaults and mutation frequency" setup=[JointSearchFixture] begin
    config = joint_config()
    program = pair_program(1, 2)
    make_session(; kwargs...) = initialize_joint_search([program], pair_ty,
        joint_types, _ -> 1.0; config, island_count=1,
        rng=MersenneTwister(555), kwargs...)
    default = make_session()
    explicit_default = make_session(use_frequency_in_tournament=true,
        adaptive_parsimony_scaling=1040.0)
    no_tournament_frequency = make_session(use_frequency_in_tournament=false,
        adaptive_parsimony_scaling=17.0)

    default_state = only(default.states)
    explicit_state = only(explicit_default.states)
    disabled_state = only(no_tournament_frequency.states)
    @test default_state.flow.use_frequency_in_tournament
    @test default_state.flow.adaptive_parsimony_scaling == 1040.0
    @test explicit_state.flow.use_frequency_in_tournament
    @test explicit_state.flow.adaptive_parsimony_scaling == 1040.0
    @test !disabled_state.flow.use_frequency_in_tournament
    @test disabled_state.flow.adaptive_parsimony_scaling == 17.0
    @test default_state.flow.use_frequency
    @test explicit_state.flow.use_frequency
    @test disabled_state.flow.use_frequency

    default_population = only(joint_search_result(default).population)
    explicit_population = only(joint_search_result(explicit_default).population)
    @test [(c.program.tree, c.program.local_consts, c.loss) for c in default_population] ==
        [(c.program.tree, c.program.local_consts, c.loss) for c in explicit_population]
    @test rand(copy(default_state.rng), UInt64) ==
        rand(copy(explicit_state.rng), UInt64) ==
        rand(copy(disabled_state.rng), UInt64)

    for invalid_scale in (-1.0, NaN, Inf)
        @test_throws ArgumentError make_session(
            adaptive_parsimony_scaling=invalid_scale)
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

@testitem "joint fixed template guards initialization and every rebuilt value" setup=[JointSearchFixture] begin
    config = joint_config()
    session = initialize_joint_search([pair_program(1, 2)], pair_ty, joint_types,
        _ -> 1.0; config, island_count=1, rng=MersenneTwister(1234),
        fixed_template=true, global_probability=0.0)
    anchor = session.domain.fixed_template
    @test all(m -> R._joint_same_template(anchor, m.candidate.value),
        only(session.states).population)
    child = R._joint_replace(session.domain, anchor,
        [1 => scalar_constant(7), 2 => scalar_constant(9)])
    @test child isa R._JointValue
    @test R._joint_same_template(anchor, child)
    wrong = Program(AstTree(ConstNode(:q));
        local_consts=Dict(:q => [1.0, 2.0]),
        local_consts_ty=Dict{Symbol,TyLike}(:q => pair_ty))
    @test R._joint_value(session.domain, wrong) isa LambdaRegression.Utils.Err
    @test_throws ArgumentError initialize_joint_search([anchor.program, wrong],
        pair_ty, joint_types, _ -> 1.0; config, island_count=1,
        fixed_template=true, global_probability=0.0)
    @test_throws ArgumentError initialize_joint_search([anchor.program], pair_ty,
        joint_types, _ -> 1.0; config, fixed_template=true)
    @test session.domain.counters[:fixed_template_rejections] >= 1
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

@testsnippet JointSimplificationFixture begin
    using Test, Random
    using LambdaRegression.Regression
    using LambdaRegression.ScalarSearch
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err
    const R = LambdaRegression.Regression
    const S = LambdaRegression.ScalarSearch
    const beta_types = Dict{Symbol,TyLike}(
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :FMUL => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    function wrapped_product()
        head = AppNode(AbsNode(:unused, ConstNode(:FMUL)), ConstNode(:FADD))
        argument = AppNode(AbsNode(:identity, IndexNode(0)), ConstNode(:a))
        return Program(AstTree(AppNode(AppNode(head, ConstNode(:a)), argument));
            local_consts=Dict(:a => 2.0), local_consts_ty=Dict{Symbol,TyLike}(:a => FloatTy))
    end
    function beta_session(score; steps=2, max_evaluations=8, max_total_evaluations=nothing)
        config = ScalarSearchConfig(; population_size=2, max_evaluations,
            max_total_evaluations, tournament_size=2, elite_count=1,
            initial_depth=0, unary_operators=(), binary_operators=(:+, :*),
            constant_refinement_iterations=0, max_constant_refinement_evaluations=0,
            evolution_strategy=:regularized)
        program = wrapped_product()
        session = initialize_joint_search([program, program], FloatTy, beta_types, score;
            config, island_count=1, rng=MersenneTwister(1234),
            program_simplification_steps=steps)
        # Isolate maintenance from independent archive migration.
        only(session.states).flow = S._NodeRegularizedFlow(
            nothing, false, 1.0, 3.17, 1, false, 0.0, 0.982, false, 1040.0)
        return session
    end
end

@testitem "joint bounded program simplification merges regions and preserves sharing" setup=[JointSimplificationFixture] begin
    for steps in 0:2
        session = beta_session(p -> 1.0; steps)
        value = only(session.states).population[1].candidate.value
        snapshot = deepcopy(value.program.tree)
        child = S._regularized_simplify_value(session.domain, value, MersenneTwister(1234))
        if steps == 0
            @test child === nothing
        else
            @test child.complexity == value.complexity - 3steps
            @test child.program.local_consts == Dict(:a => 2.0)
            @test child.program.local_consts_ty == value.program.local_consts_ty
            @test R._joint_scalar_names(child) == [:a]
            @test child.origin === :simplification
            @test child.parent_ids == (value.id,)
            steps == 2 && @test length(child.regions) == 1
        end
        @test value.program.tree == snapshot
    end
    @test_throws ArgumentError beta_session(p -> 1.0; steps=3)
end

@testitem "joint program simplification rescores complete objectives and retains age" setup=[JointSimplificationFixture] begin
    calls = Ref(0)
    score(p) = (calls[] += 1; Float64(length(p.tree)))
    session = beta_session(score)
    state = only(session.states)
    ages, clock = [m.born for m in state.population], state.birth_clock
    before = state.population[1].candidate.value.complexity
    S._node_state_maintain_cycle!(state, 8)
    @test calls[] == state.evaluations == 4
    @test all(m -> m.candidate.value.complexity == before - 6, state.population)
    @test all(m -> m.candidate.loss == length(m.candidate.value.program.tree), state.population)
    @test [m.born for m in state.population] == ages
    @test state.birth_clock == clock
    @test all(candidate.loss == length(candidate.value.program.tree) for candidate in values(state.archive))

    for invalid in (false, true)
        original_size = length(wrapped_product().tree)
        objective = p -> length(p.tree) == original_size ? 0.0 :
            invalid ? Err(:bounded_task_rejected) : 1.0
        rejected = beta_session(objective)
        checked = only(rejected.states)
        previous = [m.candidate for m in checked.population]
        S._node_state_maintain_cycle!(checked, 8)
        @test checked.evaluations == 4
        @test checked.invalid_evaluations == (invalid ? 2 : 0)
        @test [m.candidate for m in checked.population] == previous
        @test checked.best.loss == 0.0
    end
end

@testitem "joint program simplification obeys structural and total slices" setup=[JointSimplificationFixture] begin
    for (limit, total, expected) in ((2, nothing, 2), (3, nothing, 3), (8, 2, 2), (8, 3, 3))
        session = beta_session(p -> Float64(length(p.tree)); max_total_evaluations=total)
        state = only(session.states)
        S._node_state_maintain_cycle!(state, limit)
        @test state.evaluations == expected
        @test count(m -> m.candidate.value.complexity == 5, state.population) == expected - 2
    end
end

@testitem "joint program simplification preserves recursive binders and full bounds" setup=[JointSimplificationFixture] begin
    config = ScalarSearchConfig(population_size=2, max_evaluations=2,
        tournament_size=2, elite_count=1, constant_refinement_iterations=0,
        max_constant_refinement_evaluations=0, evolution_strategy=:regularized)
    recursive = Program(AstTree(FunctionDefNode(AbsNode(:x,
        AppNode(AbsNode(:u, AppNode(IndexNode(2), IndexNode(0))), IndexNode(0))))))
    session = initialize_joint_search([recursive, recursive], TyFun(FloatTy, FloatTy),
        Dict{Symbol,TyLike}(), p -> 1.0; config, island_count=1,
        program_simplification_steps=1)
    value = only(session.states).population[1].candidate.value
    child = S._regularized_simplify_value(session.domain, value, MersenneTwister(1))
    @test child.program.tree == AstTree(FunctionDefNode(AbsNode(:x,
        AppNode(IndexNode(1), IndexNode(0)))))
    @test value.program.tree == recursive.tree

    # Substitution would duplicate a large opaque Float argument beyond the
    # full-program node cap. Scalar-region simplification cannot see across it.
    argument = foldl((child, _) -> AppNode(ConstNode(:OPAQUE), child), 1:3;
        init=ConstNode(:x))
    duplicate = Program(AstTree(AppNode(
        AbsNode(:u, AppNode(AppNode(ConstNode(:FMUL), IndexNode(0)), IndexNode(0))), argument)))
    types = merge(beta_types, Dict{Symbol,TyLike}(:OPAQUE => TyFun(FloatTy, FloatTy)))
    limited = initialize_joint_search([duplicate, duplicate], FloatTy, types, p -> 1.0;
        config, island_count=1, input_types=Dict{Symbol,TyLike}(:x => FloatTy),
        max_nodes=length(duplicate.tree), program_simplification_steps=2)
    parent = only(limited.states).population[1].candidate.value
    @test S._regularized_simplify_value(limited.domain, parent, MersenneTwister(1)) === nothing
    @test parent.program.tree == duplicate.tree
    @test only(limited.states).evaluations == 2
end

@testitem "joint global accounting excludes copies refinement and migration" setup=[JointSimplificationFixture] begin
    session = beta_session(p -> 1.0; steps=0)
    state = only(session.states)
    parent = state.population[1].candidate
    value = R._joint_value(session.domain, parent.value.program;
        parent_ids=(parent.value.id,), origin=:global)
    child = S._NodeSearchCandidate(value, 1.0, 3, true)
    for event in (:do_nothing, :simplify, :migration, :constant_refinement)
        S._regularized_observe(session.domain, state, event, parent, child, true)
        @test last(session.domain.events).before ==
            (; id=parent.value.id, loss=parent.loss,
                complexity=parent.value.complexity)
    end
    @test get(session.domain.counters, :global_scored, 0) == 0
    @test get(session.domain.counters, :global_admitted, 0) == 0
    S._regularized_observe(session.domain, state, :add, parent, child, true; new_offspring=true)
    invalid = S._NodeSearchCandidate(value, Inf, 4, true)
    S._regularized_observe(session.domain, state, :add, parent, invalid, false; new_offspring=true)
    @test session.domain.counters[:global_scored] == 2
    @test session.domain.counters[:global_valid] == 1
    @test session.domain.counters[:global_admitted] == 1
    @test count(e -> e.new_offspring, session.domain.events) == 2
    S._regularized_observe(session.domain, state, :initial_population, nothing, child, true)
    @test last(session.domain.events).before === nothing
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

@testsnippet JointFunctionExposureFixture begin
    using Test, Random
    using LambdaRegression.Regression
    using LambdaRegression.ScalarSearch
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyFun, TyMu, TyLike, newtyvar
    using LambdaRegression.Utils: Err
    const R = LambdaRegression.Regression
    const S = LambdaRegression.ScalarSearch
    const binary_float = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    function exposure_session(program=Program(AstTree(ConstNode(:FADD)));
            target=binary_float, types=Dict{Symbol,TyLike}(:FADD=>binary_float,
                :SQR=>TyFun(FloatTy,FloatTy)), probability=1.0,
            score=p->Float64(length(p.tree)), max_nodes=96, max_depth=24,
            global_proposal=nothing)
        config = ScalarSearchConfig(population_size=2,max_evaluations=8,
            tournament_size=2,elite_count=1,mutation_probability=1.0,
            crossover_probability=0.0,constant_refinement_iterations=0,
            max_constant_refinement_evaluations=0,evolution_strategy=:regularized)
        session = initialize_joint_search([program,program],target,types,score;
            config,island_count=1,rng=MersenneTwister(42),max_nodes,max_depth,
            global_proposal,global_probability=0.0,
            function_exposure_probability=probability)
        only(session.states).flow = S._NodeRegularizedFlow(
            nothing,false,1.0,3.17,64,false,0.0,0.982,false,1040.0)
        return session
    end
    exposure_parent(session) = only(session.states).population[1].candidate.value
    function expose(session,rng=MersenneTwister(2000))
        parent=exposure_parent(session)
        S._regularized_named_child(session.domain,rng,parent,parent,0,
            only(session.states).config,:restart)
    end
end

@testitem "joint function exposure completes finite output bodies and owns syntax" setup=[JointFunctionExposureFixture] begin
    for (program,steps,size) in ((Program(AstTree(ConstNode(:FADD))),2,3),
            (Program(AstTree(AbsNode(:x,ConstNode(:SQR)))),1,2))
        session=exposure_session(program)
        parent=exposure_parent(session)
        snapshot=deepcopy(parent.program.tree)
        plan=R._joint_function_exposure_plan(session.domain,parent)
        @test plan.steps==steps
        child,changed=expose(session)
        @test changed
        @test child.origin===:function_exposure
        @test child.parent_ids==(parent.id,)
        @test length(child.regions)==1
        @test S.scalar_complexity(only(child.regions).expr)==size
        @test only(child.regions).dependencies==[IndexNode(0),IndexNode(1)]
        @test child.program.tree.root isa AbsNode
        @test child.program.tree.root.body isa AbsNode
        @test parent.program.tree==snapshot
        @test child.program.tree !== parent.program.tree
        @test child.program.local_consts !== parent.program.local_consts
        @test session.domain.counters[:function_exposure_attempts]==1
        @test session.domain.counters[:function_exposure_changed]==1
        @test only(session.states).evaluations==2
    end
end

@testitem "joint function exposure is disabled without RNG and skips ineligible parents" setup=[JointFunctionExposureFixture] begin
    disabled=exposure_session(;probability=0.0)
    rng=MersenneTwister(555)
    reference=copy(rng)
    child,changed=expose(disabled,rng)
    @test !changed && child===exposure_parent(disabled)
    @test rand(rng)==rand(reference)
    @test !haskey(disabled.domain.counters,:function_exposure_attempts)
    parent=exposure_parent(disabled)
    default=initialize_joint_search([parent.program,parent.program],binary_float,
        disabled.domain.type_context,p->1.0;config=only(disabled.states).config,island_count=1)
    @test default.domain.function_exposure_probability==0.0
    for probability in (-0.1,1.1,NaN)
        @test_throws ArgumentError exposure_session(;probability)
    end
    variable=newtyvar(0)
    recursive=TyMu(variable,TyFun(FloatTy,variable))
    long=foldr((_,tail)->TyFun(FloatTy,tail),1:5;init=FloatTy)
    for target in (TyFun(FloatTy,IntTy),recursive,long)
        session=exposure_session(Program(AstTree(ConstNode(:opaque)));
            target,types=Dict{Symbol,TyLike}(:opaque=>target))
        @test R._joint_function_exposure_plan(session.domain,exposure_parent(session))===nothing
        skipped_rng=MersenneTwister(1234)
        skipped_reference=copy(skipped_rng)
        @test !last(expose(session,skipped_rng))
        @test rand(skipped_rng)==rand(skipped_reference)
        @test !haskey(session.domain.counters,:function_exposure_attempts)
    end
    local_program=Program(AstTree(AbsNode(:x,IndexNode(0))))
    local_session=exposure_session(local_program;target=TyFun(FloatTy,FloatTy))
    parent=exposure_parent(local_session)
    @test R._joint_function_exposure_plan(local_session.domain,parent)===nothing
    rng1,rng2=MersenneTwister(1111),MersenneTwister(1111)
    local_session.domain.function_exposure_probability=0.0
    first=S._regularized_named_child(local_session.domain,rng1,parent,parent,1,
        only(local_session.states).config,:add)
    local_session.domain.function_exposure_probability=1.0
    second=S._regularized_named_child(local_session.domain,rng2,parent,parent,1,
        only(local_session.states).config,:add)
    @test last(first)==last(second)
    @test rand(rng1)==rand(rng2)
    @test !haskey(local_session.domain.counters,:function_exposure_attempts)
end

@testitem "joint function exposure bounds discard all intermediate edits without fallback" setup=[JointFunctionExposureFixture] begin
    for (max_nodes,max_depth) in ((4,24),(96,2))
        fallback=Ref(0)
        session=exposure_session(;max_nodes,max_depth,
            global_proposal=(p,d,rng)->(fallback[]+=1;nothing))
        parent=exposure_parent(session)
        tree=deepcopy(parent.program.tree)
        child,changed=expose(session)
        @test !changed && child===parent
        @test parent.program.tree==tree
        @test fallback[]==0
        @test session.domain.counters[:function_exposure_attempts]==1
        @test !haskey(session.domain.counters,:function_exposure_changed)
        @test only(session.states).evaluations==2
        state=only(session.states)
        S._node_state_one_offspring!(state,8)
        @test state.offspring_attempts==1
        @test state.evaluations==2
        @test session.domain.counters[:function_exposure_attempts]==2
        @test fallback[]==0
    end
end

@testitem "joint function exposure is fully rescored and uses ordinary admission" setup=[JointFunctionExposureFixture] begin
    for invalid in (false,true)
        calls=Ref(0)
        score(p)=(calls[]+=1; p.tree.root isa AbsNode && invalid ?
            Err(:task_rejected_expansion) : Float64(length(p.tree)))
        session=exposure_session(;score)
        state=only(session.states)
        S._node_state_one_offspring!(state,8)
        @test calls[]==state.evaluations==3
        @test session.domain.counters[:function_exposure_scored]==1
        @test get(session.domain.counters,:function_exposure_valid,0)==!invalid
        @test get(session.domain.counters,:function_exposure_admitted,0)==!invalid
        @test count(m->m.candidate.value.origin===:function_exposure,state.population)==!invalid
        @test state.invalid_evaluations==invalid
        event=last(session.domain.events)
        @test event.new_offspring && event.origin===:function_exposure
        @test event.admitted==!invalid
        if !invalid
            exposed=only(filter(m->m.candidate.value.origin===:function_exposure,state.population)).candidate
            @test exposed.loss==7.0 # Parent scored 1: no eta-equivalence loss reuse.
            for event in (:do_nothing,:migration,:constant_refinement)
                S._regularized_observe(session.domain,state,event,exposed,exposed,true)
            end
            @test session.domain.counters[:function_exposure_scored]==1
            @test session.domain.counters[:function_exposure_admitted]==1
        end
    end
end
