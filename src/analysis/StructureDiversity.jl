"""
    StructureDiversity

Describe editable higher-order structure without changing a program or its score.
`StructureWorld` fixes the analysis environment; `describe_structure` masks maximal
scalar regions and numerical parameter payloads, and normalizes lexical binder names. Descriptions retain
call wiring, available hole ports, actual material use, sharing and uncertainty.

```julia
world = StructureWorld(type_context; input_types, values, target_ty)
description = describe_structure(program, world)
counts = bucket_counts([description])
```

A key is equality of this view in this world, not program equivalence, a score
cache key or a resumable local-search context. No user function is executed.
Published programs and borrowed world resources must remain read-only. Per-call
state is private; this does not make arbitrary host closures thread safe.
"""
module StructureDiversity

using TestItems
using ..Utils: Err
using ..Programs: Program
using ..Programs: numeric_parameter_schema
using ..ASTs.NodeASTs: AstNode, AppNode, AbsNode, FunctionDefNode, IndexNode,
    VarNode, ConstNode
using ..ProgramAnalysis: TypedAnalysis, analyze_program
using ..ScalarRegions: scalar_region_index, is_scalar_region, scalar_region_facts
import ..TypeSystem

include("dependency_profiles.jl")

export StructureWorld, StructureBoundaryPolicy, StructureKey, StructureDescription
export StructureNode, StructureEdge, HoleDescription, CallDependency
export ParameterDescription
export describe_structure, bucket_counts, structural_features, feature_distance, explain_structure

"""
    StructureBoundaryPolicy(; view=:numeric_masked, named_input_policy=:used)

`:numeric_masked` hides maximal scalar regions (including zero-port regions)
and local numerical parameter payloads admitted by `Programs`' shared schema.
`:scalar_masked` hides only scalar regions; `:bindings` is an unmasked control.
`:used` preserves the scalar editor's current named-input boundary; `:declared`
offers every unshadowed declared Float input. Both retain all legal Float binders.
This selects observation/edit capability, not an anchoring or generation algorithm.
Invalid policy names throw `ArgumentError`.
"""
struct StructureBoundaryPolicy
    view::Symbol
    named_input_policy::Symbol
    function StructureBoundaryPolicy(; view::Symbol=:numeric_masked,
                                      named_input_policy::Symbol=:used)
        view in (:numeric_masked, :scalar_masked, :bindings) || throw(ArgumentError("unknown structure view"))
        named_input_policy in (:used, :declared) || throw(ArgumentError("unknown input policy"))
        return new(view, named_input_policy)
    end
end

"""
    StructureWorld(type_context; input_types=Dict(), values=Dict(),
                   target_ty=nothing, primitive_profiles=default_dependency_profile(),
                   dependency_condition=nothing)

Own the declaration/value dictionaries and borrow their read-only contents.
Program-local declarations and values take precedence. A world has a private
identity: keys from different worlds deliberately do not merge, even if labels
or declarations look alike. No global cache or mutable classification counter.
The caller must keep type graphs, resource values and profile assumptions stable.
`dependency_condition` is an explicit world-wide applicability token, not an
inferred shape fact. Type constraints still apply independently at each call.
"""
struct StructureWorld
    type_context::Dict{Symbol,TypeSystem.TyLike}
    input_types::Dict{Symbol,TypeSystem.TyLike}
    values::Dict{Symbol,Any}
    target_ty::Union{Nothing,TypeSystem.Ty}
    primitive_profiles::PrimitiveDependencyProfile
    dependency_condition::Any
    anchor::Base.RefValue{Nothing}
end
function StructureWorld(type_context; input_types=Dict(), values=Dict(),
                        target_ty=nothing,
                        primitive_profiles=default_dependency_profile(),
                        dependency_condition=nothing)
    return StructureWorld(Dict{Symbol,TypeSystem.TyLike}(type_context),
        Dict{Symbol,TypeSystem.TyLike}(input_types), Dict{Symbol,Any}(values),
        target_ty, primitive_profiles, dependency_condition, Ref{Nothing}(nothing))
end

"""Immutable encoding with exact identity comparisons for opaque host resources.

The owned resource vector must not be mutated after publication. Hash collisions
are resolved by the entire encoding and each resource identity, never hash alone.
No occurrence numbers or structure-dependent Julia type parameters are stored.
"""
struct StructureKey
    view::Symbol
    version::Int
    named_input_policy::Symbol
    world::Base.RefValue{Nothing}
    encoding::String
    resources::Vector{Any}
end
function Base.isequal(a::StructureKey, b::StructureKey)
    a.world === b.world && a.view === b.view && a.version == b.version &&
        a.named_input_policy === b.named_input_policy && a.encoding == b.encoding &&
        length(a.resources) == length(b.resources) || return false
    return all(a.resources[i] === b.resources[i] for i in eachindex(a.resources))
end
Base.:(==)(a::StructureKey, b::StructureKey) = isequal(a, b)
function Base.hash(key::StructureKey, h::UInt)
    h = hash(key.encoding, hash(key.named_input_policy,
        hash(key.version, hash(key.view, hash(objectid(key.world), h)))))
    for resource in key.resources
        h = hash(objectid(resource), h)
    end
    return h
end

"""Retained canonical node; `occurrence` is only an explanation into this snapshot."""
struct StructureNode
    id::Int
    occurrence::Int
    kind::Symbol
    label::String
end
"""Consumer-to-source edge; slot distinguishes ordered arguments and hole ports."""
struct StructureEdge
    consumer::Int
    source::Int
    role::Symbol
    slot::Int
end
"""Maximal editable hole: legal ports define family; used slots describe material."""
struct HoleDescription
    node::Int
    occurrence::Int
    available_ports::Vector{String}
    used_slots::Vector{Int}
end
"""A retained numerical block: values are hidden, layout/sharing remain in the key.

`edit_modes` records supported material operations, not scheduled work or a
promise of improvement. Empty tensors have no constant-value search mode.
"""
struct ParameterDescription
    node::Int
    occurrence::Int
    slot::Int
    kind::Symbol
    shape::Tuple
    edit_modes::Tuple{Vararg{Symbol}}
end
"""Finite may-dependency annotation for a retained application spine."""
struct CallDependency
    node::Int
    callee::Int
    arguments::Vector{Int}
    summary::DependencySummary
end
struct StructureDescription
    canonical_key::StructureKey
    nodes::Vector{StructureNode}
    edges::Vector{StructureEdge}
    material_edges::Vector{StructureEdge}
    holes::Vector{HoleDescription}
    parameters::Vector{ParameterDescription}
    calls::Vector{CallDependency}
    uncertainty::Vector{Symbol}
    source_node_count::Int
end

# Length-delimited tokens avoid delimiter/name collisions. Type keys are opaque
# runtime DAGs: flatten them iteratively, without recursively specializing on
# their nested Tuple types (a previously measured compilation trap in this repo).
function _token(io::IO, value)
    text = string(value)
    print(io, ncodeunits(text), ':', text, ';')
end
Base.@noinline Base.@nospecializeinfer function _type_token(io::IO, ty)
    Base.@nospecialize ty
    stack = Any[TypeSystem.typekey(ty)]
    while !isempty(stack)
        part = pop!(stack)
        if part isa Tuple
            _token(io, "tuple")
            _token(io, length(part))
            for i in length(part):-1:1
                push!(stack, part[i])
            end
        else
            _token(io, typeof(part))
            _token(io, part)
        end
    end
end

function _resource_slot!(resources, slots, value)
    return get!(slots, value) do
        push!(resources, value)
        length(resources)
    end
end

# Per-description IDs encode alias topology without encoding object addresses
# or reading dense numerical elements. Nested containers retain their layers.
function _parameter_storage_token(io, value, arrays)
    _token(io, typeof(value))
    value isa AbstractFloat && return nothing
    slot = get(arrays, value, 0)
    if slot != 0
        _token(io, :alias)
        _token(io, slot)
        return nothing
    end
    slot = length(arrays) + 1
    arrays[value] = slot
    _token(io, :array)
    _token(io, slot)
    _token(io, size(value))
    value isa Array{<:AbstractFloat} && return nothing
    for child in value
        _parameter_storage_token(io, child, arrays)
    end
    return nothing
end

function _context(program, world)
    context = copy(world.type_context)
    merge!(context, world.input_types)
    merge!(context, program.local_consts_ty)
    return context
end
function _host_value(program, world, name)
    haskey(program.local_consts, name) && return (true, program.local_consts[name])
    # Input declarations are samples, never the same-named global primitive.
    haskey(world.input_types, name) && return (false, nothing)
    return haskey(world.values, name) ? (true, world.values[name]) : (false, nothing)
end

"""
    describe_structure(program, world; analysis=nothing,
                       boundary_policy=StructureBoundaryPolicy())

Infer once if evidence is absent, otherwise borrow `TypedAnalysis` for this exact
published tree and the same world/context/target. The supplied evidence is a
caller contract, not a new type proof; evidence for another tree is rejected.
Expected inference failures return `Err`. Unknown primitives retain syntax and
conservative summaries. Source occurrences are never canonical identities.
"""
function describe_structure(program::Program, world::StructureWorld;
                            analysis::Union{Nothing,TypedAnalysis}=nothing,
                            boundary_policy::StructureBoundaryPolicy=StructureBoundaryPolicy())
    context = _context(program, world)
    if analysis === nothing
        analysis = analyze_program(program, context; target_ty=world.target_ty)
        analysis isa Err && return analysis
    else
        analysis.index.tree === program.tree ||
            throw(ArgumentError("structure evidence belongs to another program tree"))
        if world.target_ty !== nothing
            TypeSystem.prune(analysis.root_type) == TypeSystem.prune(world.target_ty) ||
                throw(ArgumentError("structure evidence has a different root target"))
        end
    end
    index = analysis.index
    scalar = boundary_policy.view !== :bindings ? scalar_region_index(
        program, analysis; type_context=context, input_types=world.input_types,
        named_input_policy=boundary_policy.named_input_policy) : nothing
    nodes = StructureNode[]
    edges = StructureEdge[]
    material_edges = StructureEdge[]
    holes = HoleDescription[]
    parameters = ParameterDescription[]
    parameter_slots = Dict{Symbol,Int}()
    parameter_arrays = IdDict{Any,Int}()
    parameter_schemas = Dict{Symbol,Any}()
    uncertainty = Symbol[]
    resources = Any[]
    resource_slots = IdDict{Any,Int}()
    binder_nodes = zeros(Int, length(index.bindings))
    occurrence_nodes = zeros(Int, length(index))
    aliases = Dict{Int,Int}()
    io = IOBuffer()

    function visit(occurrence::Int)
        node = index.nodes[occurrence]
        referenced = index.reference_binder[occurrence]
        if haskey(aliases, referenced)
            id = visit(aliases[referenced])
            occurrence_nodes[occurrence] = id
            return id
        end
        # Only a closed function-valued leaf may be a transparent direct alias.
        # Float aliases change available scalar ports; function bodies introduce
        # shared editable sites. Neither is erased. No general beta reduction.
        if node isa AppNode
            lhs, rhs = index.children[occurrence]
            rhs_node = index.nodes[rhs]
            if index.nodes[lhs] isa AbsNode && rhs_node isa Union{ConstNode,VarNode,IndexNode} &&
                    haskey(analysis.metadata, rhs)
                rhs_type = analysis.metadata[rhs][1]
                if rhs_type isa TypeSystem.Ty && TypeSystem.prune(rhs_type) isa TypeSystem.TyFun &&
                        isempty(TypeSystem.collect_free_vars(rhs_type))
                    aliases[index.owner_binder[lhs]] = rhs
                    id = visit(only(index.children[lhs]))
                    occurrence_nodes[occurrence] = id
                    return id
                end
            end
        end
        id = length(nodes) + 1
        occurrence_nodes[occurrence] = id
        if scalar !== nothing && is_scalar_region(scalar, occurrence)
            facts = scalar_region_facts(scalar, occurrence)
            ports = facts.available_ports
            labels = String[]
            _token(io, :hole)
            for operators in (facts.unary_operators, facts.binary_operators)
                _token(io, length(operators))
                for operator in sort!(collect(operators); by=String)
                    _token(io, operator)
                end
            end
            _token(io, length(ports))
            for (slot, port) in enumerate(ports)
                if port.kind === :binder
                    source = binder_nodes[port.binder_id]
                    source > 0 || error("hole port has no retained binder")
                    label = "binder:" * string(source)
                    push!(edges, StructureEdge(id, source, :available, slot))
                    slot in facts.used_port_ids &&
                        push!(material_edges, StructureEdge(id, source, :used, slot))
                else
                    label = "input:" * string(port.name)
                end
                push!(labels, label)
                _token(io, label)
            end
            push!(nodes, StructureNode(id, occurrence, :hole, "Float"))
            push!(holes, HoleDescription(id, occurrence, labels, collect(facts.used_port_ids)))
            return id
        end
        binder = index.reference_binder[occurrence]
        kind, label = if binder != 0
            (:reference, string(binder_nodes[binder]))
        elseif node isa AppNode
            (:application, "")
        elseif node isa AbsNode
            (:lambda, "")
        elseif node isa FunctionDefNode
            push!(uncertainty, :recursive_flow)
            (:fixpoint, "")
        elseif node isa IndexNode
            push!(uncertainty, :unresolved_binding)
            (:unresolved_index, string(node.index))
        elseif node isa Union{ConstNode,VarNode}
            known, value = _host_value(program, world, node.name)
            schema = if known && boundary_policy.view === :numeric_masked &&
                    haskey(program.local_consts, node.name)
                get!(parameter_schemas, node.name) do
                    numeric_parameter_schema(value, program.local_consts_ty[node.name])
                end
            else
                nothing
            end
            if schema !== nothing
                slot = get!(parameter_slots, node.name, length(parameter_slots) + 1)
                _token(io, schema.kind)
                _token(io, schema.shape)
                _parameter_storage_token(io, value, parameter_arrays)
                push!(parameters, ParameterDescription(id, occurrence, slot,
                    schema.kind, schema.shape, schema.edit_modes))
                (:parameter, string(slot))
            elseif known
                (:resource, string(_resource_slot!(resources, resource_slots, value)))
            elseif haskey(world.input_types, node.name)
                (:input, string(node.name))
            else
                push!(uncertainty, :opaque_global)
                (:global, string(node.name))
            end
        else
            error("unsupported analyzed syntax")
        end
        readable_label = kind === :resource ? string(node.name) : label
        push!(nodes, StructureNode(id, occurrence, kind, readable_label))
        _token(io, kind)
        _token(io, label)
        # Types distinguish differently declared local resources with equal values.
        if node isa Union{ConstNode,VarNode,IndexNode} && haskey(analysis.metadata, occurrence)
            _type_token(io, analysis.metadata[occurrence][1])
        end
        binder != 0 && push!(edges, StructureEdge(id, binder_nodes[binder], :binding, 0))
        owner = index.owner_binder[occurrence]
        owner != 0 && (binder_nodes[owner] = id)
        for (slot, child) in enumerate(index.children[occurrence])
            child_id = visit(child)
            role = node isa AppNode ? (slot == 1 ? :callee : :argument) : :body
            push!(edges, StructureEdge(id, child_id, role, slot))
        end
        _token(io, :end)
        return id
    end
    visit(1)

    # Let-like syntax is retained, not beta-reduced. Its one editable RHS is a
    # shared source for every use of the binder. Recording this edge makes that
    # distinction inspectable without a whole-program function-flow solver.
    for retained in nodes
        retained.kind === :application || continue
        children = index.children[retained.occurrence]
        index.nodes[children[1]] isa AbsNode || continue
        push!(edges, StructureEdge(occurrence_nodes[children[1]],
            occurrence_nodes[children[2]], :bound_value, 0))
    end
    parents = zeros(Int, length(nodes))
    callee_nodes = zeros(Int, length(nodes))
    argument_nodes = zeros(Int, length(nodes))
    for edge in edges
        edge.role in (:callee, :argument, :body) || continue
        parents[edge.source] = edge.consumer
        edge.role === :callee && (callee_nodes[edge.consumer] = edge.source)
        edge.role === :argument && (argument_nodes[edge.consumer] = edge.source)
    end
    captures = Set{Tuple{Int,Int}}()
    for edge in edges
        edge.role in (:binding, :available) || continue
        ancestor = parents[edge.consumer]
        while ancestor != 0 && ancestor != edge.source
            if nodes[ancestor].kind in (:lambda, :fixpoint)
                push!(captures, (ancestor, edge.source))
            end
            ancestor = parents[ancestor]
        end
    end
    for (consumer, source) in sort!(collect(captures))
        push!(edges, StructureEdge(consumer, source, :capture, 0))
    end

    calls = CallDependency[]
    for retained in nodes
        retained.kind === :application || continue
        parent = parents[retained.id]
        # Query a curried spine only at its outermost application.
        # Walk the retained graph: a transparent alias can remove an application
        # on the original callee spine. Raw source edges would invent extra args
        # and refer to an erased (zero-ID) callee in that case.
        parent != 0 && nodes[parent].kind === :application &&
            callee_nodes[parent] == retained.id && continue
        args = Int[]
        head = retained.id
        while nodes[head].kind === :application
            push!(args, argument_nodes[head])
            head = callee_nodes[head]
        end
        reverse!(args)
        head_occurrence = nodes[head].occurrence
        head_node = index.nodes[head_occurrence]
        known, value = head_node isa Union{ConstNode,VarNode} && index.reference_binder[head_occurrence] == 0 ?
            _host_value(program, world, head_node.name) : (false, nothing)
        types = Any[haskey(analysis.metadata, nodes[arg].occurrence) ?
            analysis.metadata[nodes[arg].occurrence][1] : nothing for arg in args]
        summary = query_dependency(world.primitive_profiles, known ? value : nothing, types;
            condition=world.dependency_condition)
        summary.unknown && push!(uncertainty, :unknown_call)
        push!(calls, CallDependency(retained.id, head, args, summary))
        for (role, positions) in (
            (:value_dependency, summary.value_inputs),
            (:shape_dependency, summary.shape_inputs),
            (:callback_invocation, summary.callback_inputs),
            (:callback_result_value, summary.callback_result_values),
            (:callback_result_shape, summary.callback_result_shapes))
            for slot in positions
                push!(edges, StructureEdge(retained.id, args[slot], role, slot))
            end
        end
        summary.callee_dependency && push!(edges,
            StructureEdge(retained.id, head, :callee_dependency, 0))
    end
    key = StructureKey(boundary_policy.view, 2, boundary_policy.named_input_policy,
        world.anchor, String(take!(io)), resources)
    return StructureDescription(key, nodes, edges, material_edges, holes, parameters, calls,
        sort!(unique!(uncertainty); by=String), length(index))
end

"""Count current descriptions by full key; does not retain scores or discovery history."""
function bucket_counts(descriptions)
    counts = Dict{StructureKey,Int}()
    for description in descriptions
        key = description.canonical_key
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

"""Small interpretable role counts, not a complete invariant or recovery criterion."""
function structural_features(description::StructureDescription)
    features = Dict{Symbol,Int}()
    for node in description.nodes
        features[node.kind] = get(features, node.kind, 0) + 1
    end
    features[:available_ports] = sum(length(h.available_ports) for h in description.holes; init=0)
    features[:used_ports] = sum(length(h.used_slots) for h in description.holes; init=0)
    features[:binding_edges] = count(e -> e.role === :binding, description.edges)
    features[:capture_edges] = count(e -> e.role === :capture, description.edges)
    features[:shared_sources] = count(e -> e.role === :bound_value, description.edges)
    return features
end

"""L1 role-count discrepancy; zero does not establish equal wiring or recovered formula."""
function feature_distance(left::StructureDescription, right::StructureDescription)
    a, b = structural_features(left), structural_features(right)
    return sum(abs(get(a, k, 0) - get(b, k, 0)) for k in union(keys(a), keys(b)); init=0)
end

"""Print retained wiring, editable interfaces and unknown calls for human review.

Resource labels are source names, while key equality uses resolved identity.
The explanation never invokes opaque resource printers or user functions.
"""
function explain_structure(io::IO, description::StructureDescription)
    println(io, "view=", description.canonical_key.view,
        " source_nodes=", description.source_node_count,
        " retained=", length(description.nodes), " holes=", length(description.holes))
    for node in description.nodes
        println(io, node.id, " ", node.kind, " ", node.label,
            " [source occurrence ", node.occurrence, "]")
    end
    for edge in description.edges
        println(io, "  ", edge.consumer, " -> ", edge.source,
            " ", edge.role, " slot=", edge.slot)
    end
    for hole in description.holes
        println(io, "  hole ", hole.node, " available=", hole.available_ports,
            " used_slots=", hole.used_slots)
    end
    for parameter in description.parameters
        println(io, "  parameter ", parameter.node, " slot=", parameter.slot,
            " kind=", parameter.kind, " shape=", parameter.shape,
            " edit_modes=", parameter.edit_modes)
    end
    for call in description.calls
        summary = call.summary
        println(io, "  call ", call.node, " callee=", call.callee,
            " arguments=", call.arguments, " value_inputs=", summary.value_inputs,
            " shape_inputs=", summary.shape_inputs,
            " callbacks=", summary.callback_inputs,
            " callback_result_shapes=", summary.callback_result_shapes,
            " callee_dependency=", summary.callee_dependency,
            " reason=", summary.reason)
    end
    println(io, "uncertainty=", description.uncertainty)
    return nothing
end
explain_structure(description::StructureDescription) =
    sprint(io -> explain_structure(io, description))

include("StructureDiversityTests.jl")

end # module StructureDiversity
