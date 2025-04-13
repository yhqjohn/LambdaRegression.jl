"""
    ScalarRegions

One-pass, occurrence-indexed recognition of exact-Float scalar regions in a
typed program. The index stores shared scalar seeds and compact port-use
summaries; callers materialise a `ScalarExpr` only for a selected region.

Operator admission is syntax- and signature-based. It describes the bounded
edit language, not a proof that a host value installed under that name has the
corresponding mathematical behavior; complete-program evaluation remains in
the caller's actual execution world.

Quickstart: build one `TypedAnalysis`, call `scalar_region_index`, then use
`is_scalar_region(index, occurrence)` during traversal and request
`scalar_region_facts` only for selected roots.
"""
module ScalarRegions

using TestItems
using ..ASTs.NodeASTs:
    AstNode, AppNode, AbsNode, ConstNode, FunctionDefNode, IndexNode, VarNode
using ..Programs: Program
using ..ScalarSearch: ScalarExpr, scalar_input, scalar_constant, scalar_unary, scalar_binary
import ..ScalarSearch
using ..TypedSearch: TypedAnalysis
using ..TypeSystem: FloatTy, Ty, TyFun, TyLike, TyScheme, prune, iscompatible
import ..TypeSystem

export ScalarPort, ScalarRegionIndex, ScalarRegionFacts
export scalar_region_index, is_scalar_region, scalar_region_facts
export instantiate_scalar_expression, used_port_ids

"""One available Float input. Binder ports retain identity across occurrences."""
struct ScalarPort
    kind::Symbol                 # `:binder` or `:named_input`
    binder_id::Int               # nonzero only for `:binder`
    name::Union{Nothing,Symbol}  # non-nothing only for `:named_input`
    node::AstNode                # replacement leaf in the source AST
    ty::Ty
end

abstract type _ScalarSeed end
struct _SeedPort <: _ScalarSeed
    kind::Symbol
    binder_id::Int
    name::Union{Nothing,Symbol}
end
struct _SeedConstant <: _ScalarSeed
    value::Float64
end
struct _SeedUnary <: _ScalarSeed
    op::Symbol
    arg::_ScalarSeed
end
struct _SeedBinary <: _ScalarSeed
    op::Symbol
    left::_ScalarSeed
    right::_ScalarSeed
end

"""A cheap concatenated view: visible binders first, then named inputs."""
struct _PortView <: AbstractVector{ScalarPort}
    binders::Vector{ScalarPort}
    named_catalog::Vector{ScalarPort}
    selected_named::Union{Nothing,Vector{Int}} # `nothing` means all declared names
end
Base.IndexStyle(::Type{_PortView}) = IndexLinear()
Base.size(ports::_PortView) = (length(ports),)
Base.length(ports::_PortView) = length(ports.binders) +
    (ports.selected_named === nothing ? length(ports.named_catalog) : length(ports.selected_named))
function Base.getindex(ports::_PortView, position::Int)
    @boundscheck checkbounds(ports, position)
    nbinders = length(ports.binders)
    position <= nbinders && return ports.binders[position]
    name_position = position - nbinders
    catalog_position = ports.selected_named === nothing ? name_position :
        ports.selected_named[name_position]
    return ports.named_catalog[catalog_position]
end

"""Facts for one admissible occurrence. `used_port_ids` are 1-based slots."""
struct ScalarRegionFacts
    occurrence::Int
    seed::_ScalarSeed
    available_ports::_PortView
    used_port_ids::BitSet
    unary_operators::Set{Symbol}
    binary_operators::Set{Symbol}
end

"""Shared bottom-up scalar evidence for one `TypedAnalysis`."""
struct ScalarRegionIndex
    program::Program
    analysis::TypedAnalysis
    named_input_policy::Symbol
    named_ports::Vector{ScalarPort}
    input_slots::Dict{Symbol,Int}
    seeds::Vector{Union{Nothing,_ScalarSeed}}
    used_keys::Vector{BitSet}
    admissible::BitVector
    binder_ports_by_env::Dict{Int,Vector{ScalarPort}}
    unary_operators::Set{Symbol}
    binary_operators::Set{Symbol}
end

@inline function _concrete_binder_type(ty)
    return ty isa TyScheme && isempty(ty.vars) ? ty.ty : ty
end

@inline _exact_float(ty) = begin
    concrete = _concrete_binder_type(ty)
    concrete isa Ty && prune(concrete) == FloatTy
end

function _copy_type_context(value)
    return Dict{Symbol,TyLike}(value)
end

function _merged_context(program::Program, type_context, input_types)
    context = _copy_type_context(type_context)
    merge!(context, _copy_type_context(input_types))
    merge!(context, Dict{Symbol,TyLike}(program.local_consts_ty))
    return context
end

function _closed_signature_compatible(actual, expected::Ty)
    if actual isa TyScheme
        all(variable -> variable.instance === nothing, actual.vars) || return false
        quantified = Set(actual.vars)
        all(variable -> variable in quantified,
            TypeSystem.collect_free_vars(actual.ty)) || return false
        return iscompatible(actual, expected)
    elseif actual isa Ty
        isempty(TypeSystem.collect_free_vars(actual)) || return false
        return prune(actual) == expected
    end
    return false
end

const _UNARY_FLOAT_SIGNATURE = TyFun(FloatTy, FloatTy)
const _BINARY_FLOAT_SIGNATURE = TyFun(FloatTy, TyFun(FloatTy, FloatTy))

function _operator_sets(context)
    unary, binary = Set{Symbol}(), Set{Symbol}()
    for (operator, ast_name) in ScalarSearch._UNARY_AST
        actual = get(context, ast_name, nothing)
        actual !== nothing && _closed_signature_compatible(actual, _UNARY_FLOAT_SIGNATURE) &&
            push!(unary, operator)
    end
    for (operator, ast_name) in ScalarSearch._BINARY_AST
        actual = get(context, ast_name, nothing)
        actual !== nothing && _closed_signature_compatible(actual, _BINARY_FLOAT_SIGNATURE) &&
            push!(binary, operator)
    end
    return unary, binary
end

function _finite_local_constant(program::Program, name::Symbol)
    haskey(program.local_consts, name) || return nothing
    value = program.local_consts[name]
    value isa Real || return nothing
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

function _visible_float_ports!(cache, analysis::TypedAnalysis, env_head::Int)
    return get!(cache, env_head) do
        index = analysis.index
        ports = ScalarPort[]
        binder = env_head
        db = 0
        while binder != 0
            scheme = get(analysis.binder_schemes, binder, nothing)
            if _exact_float(scheme)
                ty = prune(_concrete_binder_type(scheme))
                push!(ports, ScalarPort(:binder, binder, nothing, IndexNode(db), ty))
            end
            binder = index.bindings[binder].outer
            db += 1
        end
        ports
    end
end

function _source_hole_mask(analysis::TypedAnalysis)
    index = analysis.index
    path_allowed = falses(length(index.nodes))
    allowed = falses(length(index.nodes))
    for occurrence in eachindex(index.nodes)
        parent = index.parent[occurrence]
        okay = parent == 0 || path_allowed[parent]
        if okay && parent != 0 && index.parent_edge[occurrence] == 2 &&
                index.nodes[parent] isa AppNode
            parent_children = index.children[parent]
            if length(parent_children) == 2 &&
                    index.nodes[parent_children[1]] isa AbsNode
                meta = get(analysis.metadata, occurrence, nothing)
                # source_hole rejects every occurrence reached through this
                # let RHS when its root result remains open. A missing
                # metadata record for this descendant is checked below, but
                # does not itself make all of its children unreachable.
                okay = meta !== nothing &&
                    isempty(TypeSystem.collect_free_vars(meta[1]))
            end
        end
        path_allowed[occurrence] = okay
        allowed[occurrence] = okay && haskey(analysis.metadata, occurrence)
    end
    return allowed
end

"""
    scalar_region_index(program, analysis; type_context, input_types=Dict(),
        named_input_policy=:used)

Build reusable bottom-up facts from an analysis of the same source tree.
Named input policy `:used` keeps only declared Float inputs referenced by a
region; `:declared` makes every declared Float input available in stable name
order. Program-local constants shadow same-named named inputs.
"""
function scalar_region_index(
    program::Program,
    analysis::TypedAnalysis;
    type_context,
    input_types=Dict(),
    named_input_policy::Symbol=:used,
)
    named_input_policy in (:used, :declared) ||
        throw(ArgumentError("named_input_policy must be :used or :declared"))
    analysis.index.tree === program.tree ||
        throw(ArgumentError("analysis must refer to the same Program tree"))

    context = _merged_context(program, type_context, input_types)
    declared = _copy_type_context(input_types)
    names = Symbol[name for name in keys(declared)
        if _exact_float(declared[name]) && !haskey(program.local_consts, name)]
    sort!(names; by=String)
    named_ports = ScalarPort[
        ScalarPort(:named_input, 0, name, ConstNode(name),
            prune(_concrete_binder_type(declared[name])))
        for name in names
    ]
    slots = Dict(name => i for (i, name) in enumerate(names))
    unary, binary = _operator_sets(context)

    index = analysis.index
    n = length(index.nodes)
    seeds = Vector{Union{Nothing,_ScalarSeed}}(undef, n)
    fill!(seeds, nothing)
    # Bit IDs 1..nbinders are lexical binder IDs; named inputs follow them.
    named_base = length(index.bindings)
    used = [BitSet() for _ in 1:n]
    allowed = _source_hole_mask(analysis)
    binder_cache = Dict{Int,Vector{ScalarPort}}()
    admissible = falses(n)
    constants = Dict{Symbol,Float64}()
    for name in keys(program.local_consts)
        value = _finite_local_constant(program, name)
        value === nothing || (constants[name] = value)
    end

    for occurrence in n:-1:1
        allowed[occurrence] || continue
        metadata = get(analysis.metadata, occurrence, nothing)
        metadata === nothing && continue
        _exact_float(metadata[1]) || continue
        node = index.nodes[occurrence]
        seed = nothing
        if node isa Union{IndexNode,VarNode}
            binder_id = index.reference_binder[occurrence]
            if binder_id != 0
                binder_type = get(analysis.binder_schemes, binder_id, nothing)
                _exact_float(binder_type) || continue
                seed = _SeedPort(:binder, binder_id, nothing)
                push!(used[occurrence], binder_id)
            end
        elseif node isa ConstNode
            if haskey(program.local_consts, node.name)
                haskey(constants, node.name) || continue
                seed = _SeedConstant(constants[node.name])
            elseif haskey(slots, node.name)
                input_slot = slots[node.name]
                seed = _SeedPort(:named_input, 0, node.name)
                push!(used[occurrence], named_base + input_slot)
            end
        elseif node isa AppNode
            child_ids = index.children[occurrence]
            if length(child_ids) == 2
                function_id, argument_id = child_ids
                function_node = index.nodes[function_id]
                if function_node isa ConstNode && argument_id <= n
                    op = get(ScalarSearch._AST_UNARY, function_node.name, nothing)
                    argument_seed = seeds[argument_id]
                    if op !== nothing && op in unary && argument_seed !== nothing
                        seed = _SeedUnary(op, argument_seed)
                        union!(used[occurrence], used[argument_id])
                    end
                elseif index.nodes[function_id] isa AppNode
                    function_children = index.children[function_id]
                    if length(function_children) == 2 &&
                            index.nodes[function_children[1]] isa ConstNode
                        operator_node = index.nodes[function_children[1]]::ConstNode
                        op = get(ScalarSearch._AST_BINARY, operator_node.name, nothing)
                        left_id, right_id = function_children[2], argument_id
                        left_seed, right_seed = seeds[left_id], seeds[right_id]
                        if op !== nothing && op in binary &&
                                left_seed !== nothing && right_seed !== nothing
                            seed = _SeedBinary(op, left_seed, right_seed)
                            union!(used[occurrence], used[left_id])
                            union!(used[occurrence], used[right_id])
                        end
                    end
                end
            end
        end
        seeds[occurrence] = seed
        seed === nothing && continue

        # A supported constant-only seed is a valid zero-input scalar region.
        # Port leaves already carry their dependency in `used`; they are still
        # projected and checked when facts are requested.
        admissible[occurrence] = true
    end

    return ScalarRegionIndex(
        program, analysis, named_input_policy, named_ports, slots,
        seeds, used, admissible, binder_cache, unary, binary,
    )
end

"""Cheaply ask whether an occurrence is a refinable exact-Float region."""
function is_scalar_region(index::ScalarRegionIndex, occurrence::Integer)
    id = Int(occurrence)
    return 1 <= id <= length(index.admissible) && index.admissible[id]
end

function _binder_ports(index::ScalarRegionIndex, occurrence::Int)
    return _visible_float_ports!(index.binder_ports_by_env,
        index.analysis, index.analysis.index.env_head[occurrence])
end

function _selected_named_ids(index::ScalarRegionIndex, occurrence::Int)
    index.named_input_policy === :declared && return nothing
    named_base = length(index.analysis.index.bindings)
    result = Int[]
    for input_slot in eachindex(index.named_ports)
        named_base + input_slot in index.used_keys[occurrence] && push!(result, input_slot)
    end
    return result
end

function _used_slots(index::ScalarRegionIndex, occurrence::Int, ports::_PortView)
    result = BitSet()
    named_base = length(index.analysis.index.bindings)
    for (slot, port) in enumerate(ports)
        key = port.kind === :binder ? port.binder_id :
            named_base + index.input_slots[port.name::Symbol]
        key in index.used_keys[occurrence] && push!(result, slot)
    end
    return result
end

"""Return ordered ports and shared-seed facts for one admissible occurrence."""
function scalar_region_facts(index::ScalarRegionIndex, occurrence::Integer)
    id = Int(occurrence)
    is_scalar_region(index, id) || return nothing
    ports = _PortView(
        _binder_ports(index, id), index.named_ports,
        _selected_named_ids(index, id),
    )
    return ScalarRegionFacts(
        id,
        index.seeds[id]::_ScalarSeed,
        ports,
        _used_slots(index, id, ports),
        index.unary_operators,
        index.binary_operators,
    )
end

function _seed_expression(seed::_ScalarSeed, slots::Dict{Any,Int})
    if seed isa _SeedPort
        key = seed.kind === :binder ? (:binder, seed.binder_id) :
            (:named_input, seed.name)
        return scalar_input(slots[key])
    elseif seed isa _SeedConstant
        return scalar_constant(seed.value)
    elseif seed isa _SeedUnary
        return scalar_unary(seed.op, _seed_expression(seed.arg, slots))
    elseif seed isa _SeedBinary
        return scalar_binary(seed.op,
            _seed_expression(seed.left, slots), _seed_expression(seed.right, slots))
    end
    error("unknown scalar seed")
end

"""Expand one selected shared seed into its one-based `ScalarExpr` ports."""
function instantiate_scalar_expression(facts::ScalarRegionFacts)
    slots = Dict{Any,Int}()
    for (slot, port) in enumerate(facts.available_ports)
        key = port.kind === :binder ? (:binder, port.binder_id) :
            (:named_input, port.name)
        slots[key] = slot
    end
    return _seed_expression(facts.seed, slots)
end

function _has_supported_seed(index::ScalarRegionIndex, occurrence::Int)
    return 1 <= occurrence <= length(index.seeds) && index.seeds[occurrence] !== nothing
end

@testitem "scalar-region facts preserve typed admission and ordered ports" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, IndexNode, VarNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.Regression: ScalarRegion, prepare_scalar_region
    using LambdaRegression.ScalarRegions:
        ScalarPort, scalar_region_index, is_scalar_region, scalar_region_facts
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    target = TyFun(FloatTy, FloatTy)
    context = Dict{Symbol,TyLike}(:FADD => get_type(FADD))
    inputs = Dict{Symbol,TyLike}(:u => FloatTy)
    program = Program(AstTree(AbsNode(:x, add(IndexNode(0), ConstNode(:u)))))
    analysis = analyze_program(program, merge(context, inputs); target_ty=target)
    @test analysis isa TypedAnalysis
    index = scalar_region_index(program, analysis;
        type_context=context, input_types=inputs)
    occurrence = 2 # The body application is the whole exact-Float expression.
    facts = scalar_region_facts(index, occurrence)
    @test is_scalar_region(index, occurrence)
    @test facts !== nothing
    @test facts.available_ports isa AbstractVector{ScalarPort}
    @test [(port.kind, port.binder_id, port.name) for port in facts.available_ports] ==
        [(:binder, 1, nothing), (:named_input, 0, :u)]
    @test collect(facts.used_port_ids) == [1, 2]

    prepared = prepare_scalar_region(program, target, context, [1]; input_types=inputs)
    @test prepared isa ScalarRegion
    @test prepared.dependencies == [IndexNode(0), ConstNode(:u)]
    @test [port.node for port in facts.available_ports] == prepared.dependencies

    named_bound = Program(AstTree(AbsNode(:x,
        add(VarNode(:x), ConstNode(:u)))))
    named_analysis = analyze_program(named_bound, merge(context, inputs);
        target_ty=target)
    named_index = scalar_region_index(named_bound, named_analysis;
        type_context=context, input_types=inputs)
    named_facts = scalar_region_facts(named_index, 2)
    @test named_facts !== nothing
    @test [port.node for port in named_facts.available_ports] ==
        [IndexNode(0), ConstNode(:u)]
    @test collect(named_facts.used_port_ids) == [1, 2]
end

@testitem "scalar-region facts keep local constants and named-input policies distinct" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.Regression: ScalarRegion, prepare_scalar_region
    using LambdaRegression.ScalarRegions: scalar_region_index, is_scalar_region,
        scalar_region_facts, instantiate_scalar_expression
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    target = TyFun(FloatTy, FloatTy)
    context = Dict{Symbol,TyLike}(:FADD => get_type(FADD))
    inputs = Dict{Symbol,TyLike}(:c => FloatTy, :u => FloatTy, :unused => FloatTy)
    shadowed = Program(AstTree(AbsNode(:x, add(IndexNode(0), ConstNode(:c))));
        local_consts=Dict{Symbol,Any}(:c => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    analysis = analyze_program(shadowed, merge(context, inputs, shadowed.local_consts_ty);
        target_ty=target)
    used_index = scalar_region_index(shadowed, analysis;
        type_context=context, input_types=inputs, named_input_policy=:used)
    used_occurrence = 2
    used_facts = scalar_region_facts(used_index, used_occurrence)
    @test used_facts !== nothing
    @test [(port.kind, port.name) for port in used_facts.available_ports] ==
        [(:binder, nothing)]
    prepared = prepare_scalar_region(shadowed, target, context, [1]; input_types=inputs)
    @test prepared isa ScalarRegion
    @test prepared.dependencies == [IndexNode(0)]
    @test isempty(prepared.named_dependencies)

    constant = Program(AstTree(ConstNode(:c));
        local_consts=Dict{Symbol,Any}(:c => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    constant_analysis = analyze_program(constant, merge(context, inputs,
        constant.local_consts_ty); target_ty=FloatTy)
    default_index = scalar_region_index(constant, constant_analysis;
        type_context=context, input_types=inputs)
    declared_index = scalar_region_index(constant, constant_analysis;
        type_context=context, input_types=inputs, named_input_policy=:declared)
    @test is_scalar_region(default_index, 1)
    default_facts = scalar_region_facts(default_index, 1)
    @test default_facts !== nothing
    @test isempty(default_facts.available_ports)
    @test isempty(default_facts.used_port_ids)
    @test instantiate_scalar_expression(default_facts) isa
        LambdaRegression.ScalarSearch._ScalarConstant
    default_prepared = prepare_scalar_region(
        constant, FloatTy, context, Int[]; input_types=inputs)
    @test default_prepared isa ScalarRegion
    @test isempty(default_prepared.dependencies)
    @test isempty(default_prepared.named_dependencies)
    @test default_prepared.expr isa LambdaRegression.ScalarSearch._ScalarConstant
    declared = scalar_region_facts(declared_index, 1)
    @test declared !== nothing
    @test [port.name for port in declared.available_ports] == [:u, :unused]
    @test isempty(declared.used_port_ids)

    nonfinite = Program(AstTree(ConstNode(:bad));
        local_consts=Dict{Symbol,Any}(:bad => Inf),
        local_consts_ty=Dict{Symbol,TyLike}(:bad => FloatTy))
    nonfinite_analysis = analyze_program(nonfinite,
        Dict{Symbol,TyLike}(:bad => FloatTy); target_ty=FloatTy)
    nonfinite_index = scalar_region_index(nonfinite, nonfinite_analysis;
        type_context=Dict{Symbol,TyLike}())
    @test !is_scalar_region(nonfinite_index, 1)
    @test scalar_region_facts(nonfinite_index, 1) === nothing
end

@testitem "scalar-region facts respect fixed recursion and generalising RHS boundaries" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode
    using LambdaRegression.Programs: Program
    using LambdaRegression.Primitives: FADD, FST, PAIR
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.ScalarRegions: scalar_region_index, is_scalar_region
    using LambdaRegression.TypeSystem:
        FloatTy, TyFun, TyLike, TyScheme, get_type, newtyvar
    using LambdaRegression.TypedSearch: SourceHole, source_hole
    using LambdaRegression.Utils: Err

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    context = Dict{Symbol,TyLike}(:FADD => get_type(FADD))
    target = TyFun(FloatTy, FloatTy)
    recursive = Program(AstTree(FunctionDefNode(AbsNode(:x,
        add(IndexNode(0), IndexNode(0))))))
    recursive_analysis = analyze_program(recursive, context; target_ty=target)
    recursive_index = scalar_region_index(recursive, recursive_analysis;
        type_context=context)
    @test !is_scalar_region(recursive_index, 1)
    recursive_hole = source_hole(recursive_analysis, [1, 1])
    @test recursive_hole isa SourceHole
    recursive_body = recursive_hole.occurrence
    @test is_scalar_region(recursive_index, recursive_body)

    # The RHS has type Float × (α → α), then the let-like use only selects the
    # Float component. Its internal Float constant is not an independent hole.
    open_rhs = AppNode(AppNode(ConstNode(:PAIR), ConstNode(:one)), ConstNode(:ID))
    let_program = Program(AstTree(AppNode(
        AbsNode(:pair, AppNode(ConstNode(:FST), IndexNode(0))), open_rhs,
    ));
        local_consts=Dict{Symbol,Any}(:one => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:one => FloatTy))
    id_variable = newtyvar(0)
    let_context = Dict{Symbol,TyLike}(
        :FADD => get_type(FADD), :FST => get_type(FST),
        :PAIR => get_type(PAIR),
        :ID => TyScheme([id_variable], TyFun(id_variable, id_variable)),
    )
    let_analysis = analyze_program(let_program,
        merge(let_context, let_program.local_consts_ty); target_ty=FloatTy)
    let_index = scalar_region_index(let_program, let_analysis;
        type_context=let_context)
    @test source_hole(let_analysis, [2]) isa Err
    one_occurrence = findfirst(node -> node isa ConstNode && node.name === :one,
        let_analysis.index.nodes)
    @test one_occurrence !== nothing
    @test source_hole(let_analysis, [2, 1, 2]) isa Err
    @test !is_scalar_region(let_index, one_occurrence)
end

end # module ScalarRegions
