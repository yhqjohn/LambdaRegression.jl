"""
Automatic scalar-region preparation, contextual refinement, and verified
materialisation for the program--scalar bridge.

The ordinary `Program.tree` remains authoritative. A `ScalarRegion` owns
source evidence from `TypedSearch.prepare_source`/`source_hole`, an ordered
feature boundary, and one parsed `ScalarExpr`; it is a disposable view rather
than a second program representation. `search_scalar_region` uses the
existing scalar population search as a bounded session. Its low-level `X/y`
arguments are one-column zero placeholders required by that API only; the
contextual objective never reads them, but the backend uses `mean(y)=0` to
construct its normalization baseline, which is scored in the full context.
`initialize_scalar_region_session` captures an owned region for the existing
multi-island coordinator; `advance_scalar_region!` preserves its populations,
shared frequencies and budgets between whole coordinator rounds. The caller
must keep the captured scorer/task stable. Hot candidates path-copy only the
changed occurrence's ancestors and borrow untouched syntax read-only. Their
type is certified from the source hole and a closed set of Float operators;
only final `Program` publication makes a deep, independently owned copy.

The first boundary admits exact `Float` regions, including constant-only
regions with no feature ports, and finite real local constants. Non-floating
external binders, unsupported syntax, and recursive scalar regions stay on
the complete typed program path and are returned as classified `Err` values.
Operator admission uses recognized syntax and type signatures; it does not
certify that a same-named host callable has the matching mathematical
behavior. Candidate objectives evaluate the complete program in their actual
execution world.
"""

using Random: AbstractRNG, default_rng

using ..Utils: Err
using ..ASTs.NodeASTs:
    AstNode, AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode,
    children
using ..Programs: Program, _publish_program
using ..ScalarSearch:
    ScalarExpr, ScalarSearchConfig, ScalarSearchResult,
    scalar_to_ast, scalar_complexity, scalar_input, search_scalar
import ..ScalarSearch
using ..ScalarRegions:
    scalar_region_index, scalar_region_facts, instantiate_scalar_expression
import ..ScalarRegions
using ..TypedSearch:
    TypedAnalysis, SourceBoundaryError, SourceHole, prepare_source, source_hole
using ..TypeSystem: FloatTy, Ty, TyLike, prune
import ..TypeSystem

export ScalarRegion, ScalarRegionError
export prepare_scalar_region, materialize_scalar_region, search_scalar_region
export ScalarRegionSession, initialize_scalar_region_session
export advance_scalar_region!, scalar_region_result
export scalar_region_progress

"""Expected refusal at the automatic scalar-region boundary."""
struct ScalarRegionError
    reason::Symbol
    detail::Any
end

ScalarRegionError(reason::Symbol) = ScalarRegionError(reason, nothing)

"""An owned, immutable structural snapshot with a shape-stable comparison boundary."""
struct _RegionCertificate
    value::Tuple

    Base.@noinline Base.@nospecializeinfer function _RegionCertificate(value::Tuple)
        Base.@nospecialize value
        return new(value)
    end
end

# Tuple == specialises on the full types of both operands. Semantic type keys
# can contain hundreds of nested Tuple nodes, so compare their runtime shape
# here without passing any nested pair back through a Tuple-typed work item.
Base.@noinline Base.@nospecializeinfer function _certificate_equal(left::Any, right::Any)
    Base.@nospecialize left right
    left_stack = Any[left]
    right_stack = Any[right]
    unknown = false
    while !isempty(left_stack)
        current_left = pop!(left_stack)
        current_right = pop!(right_stack)
        if current_left isa Tuple && current_right isa Tuple
            length(current_left) == length(current_right) || return false
            # LIFO work stacks preserve Tuple == comparison order.
            for index in length(current_left):-1:1
                push!(left_stack, current_left[index])
                push!(right_stack, current_right[index])
            end
        else
            equal = current_left == current_right
            equal === false && return false
            equal === missing && (unknown = true)
            equal === true || equal === missing ||
                throw(TypeError(:_certificate_equal, "== result", Union{Bool,Missing}, equal))
        end
    end
    return unknown ? missing : true
end

_certificate_matches(current::_RegionCertificate, captured::_RegionCertificate) =
    _certificate_equal(current.value, captured.value)

"""
Owned scalar view of one ordinary program occurrence.

`dependencies` are AST leaves in the exact order used by the scalar
expression.  External λ binders use zero-based De Bruijn `IndexNode`s, nearest
first; named floating inputs use `ConstNode`s in stable lexical-name order.
The source, parent individual, and context are owned copies.
"""
struct ScalarRegion
    program::Program
    target_ty::Ty
    source::TypedAnalysis
    hole::SourceHole
    path::Vector{Int}
    path_certificate::_RegionCertificate
    dependencies::Vector{AstNode}
    dependency_certificate::_RegionCertificate
    dependency_types::Vector{Ty}
    named_dependencies::Vector{Symbol}
    expr::ScalarExpr
    type_context::Dict{Symbol,TyLike}
    type_context_certificate::_RegionCertificate
    reserved_names::Set{Symbol}
    unary_operators::Set{Symbol}
    binary_operators::Set{Symbol}
    source_node_count::Int
    original_region_node_count::Int
    region_depth::Int
    max_nodes::Int
    max_depth::Int
end

@inline _region_error(reason::Symbol, detail=nothing) =
    Err(ScalarRegionError(reason, detail))

function _copy_type_context(value)
    try
        return Dict{Symbol,TyLike}(value)
    catch error
        error isa MethodError || error isa TypeError || rethrow()
        throw(ArgumentError("type_context must map Symbols to TyLike values"))
    end
end

function _valid_bounds(max_nodes::Int, max_depth::Int)
    max_nodes > 0 || throw(ArgumentError("max_nodes must be positive"))
    max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative"))
    return nothing
end

function _merged_type_context(program::Program, type_context, input_types)
    context = _copy_type_context(type_context)
    merge!(context, _copy_type_context(input_types))
    # Program-local declarations have the same precedence as their values
    # in the ordinary evaluator.  The dictionaries remain caller-owned.
    merge!(context, Dict{Symbol,TyLike}(program.local_consts_ty))
    return context
end

function _exact_float(ty)
    ty isa Ty || return false
    return prune(ty) == FloatTy
end

function _type_context_certificate(context::AbstractDict{Symbol,<:TyLike})
    names = sort!(collect(keys(context)); by=String)
    return _RegionCertificate(Tuple(
        (name, TypeSystem.semantic_typekey(context[name])) for name in names))
end

"""Validate the scalar IR and count its exact materialised AST size/depth."""
function _scalar_fragment_shape(
    expression::ScalarExpr,
    dependency_count::Int,
    unary_operators::Set{Symbol},
    binary_operators::Set{Symbol},
    max_fragment_nodes::Int,
    absolute_root_depth::Int,
    max_depth::Int,
)
    max_fragment_nodes >= 1 || return _region_error(:materialization_size, 0)
    stack = Tuple{ScalarExpr,Int}[(expression, 0)]
    ast_nodes = 0
    while !isempty(stack)
        node, depth = pop!(stack)
        absolute_depth = absolute_root_depth + depth
        absolute_depth <= max_depth || return _region_error(
            :materialization_depth,
            (actual=absolute_depth, maximum=max_depth),
        )
        if node isa ScalarSearch._ScalarInput
            1 <= node.index <= dependency_count || return _region_error(
                :missing_dependency, node.index)
            ast_nodes += 1
        elseif node isa ScalarSearch._ScalarConstant
            isfinite(node.value) || return _region_error(:nonfinite_constant, node.value)
            ast_nodes += 1
        elseif node isa ScalarSearch._ScalarUnary
            node.op in unary_operators || return _region_error(
                :unsupported_operator, node.op)
            absolute_depth + 1 <= max_depth || return _region_error(
                :materialization_depth,
                (actual=absolute_depth + 1, maximum=max_depth),
            )
            ast_nodes += 2 # AppNode and its operator ConstNode.
            push!(stack, (node.arg, depth + 1))
        elseif node isa ScalarSearch._ScalarBinary
            node.op in binary_operators || return _region_error(
                :unsupported_operator, node.op)
            absolute_depth + 2 <= max_depth || return _region_error(
                :materialization_depth,
                (actual=absolute_depth + 2, maximum=max_depth),
            )
            ast_nodes += 3 # Two AppNodes and the operator ConstNode.
            push!(stack, (node.right, depth + 1))
            push!(stack, (node.left, depth + 2))
        else
            return _region_error(:invalid_expression, typeof(node))
        end
        ast_nodes <= max_fragment_nodes || return _region_error(
            :materialization_size,
            (actual=ast_nodes, maximum=max_fragment_nodes),
        )
    end
    return (; ast_nodes)
end

function _source_failure(result)
    result isa Err || return _region_error(:source, result)
    detail = result.value
    reason = detail isa SourceBoundaryError ? detail.reason : :source
    return _region_error(reason, detail)
end

"""
    prepare_scalar_region(program, target_ty, type_context, path;
        input_types=Dict(), named_input_policy=:used, analysis=nothing,
        max_nodes=100_000, max_depth=512)

Prepare one exact-`Float` source occurrence for contextual scalar refinement.
The full individual is inferred first, then `path` is resolved through the
owned source evidence. Visible floating De Bruijn binders and declared named
floating `ConstNode`s become dependencies; free `VarNode`s are not treated as
global inputs. `named_input_policy=:used` is the default; `:declared` exposes
all declared Float inputs in stable name order. Supplying `analysis` reuses a
caller-owned `TypedAnalysis`; the caller must have analyzed this exact program
tree in the same merged type world. Tree identity, target type, and whole-
source bounds are checked, while context equivalence is a caller contract.
Unsupported source/seed boundaries return `Err(ScalarRegionError(...))`.
Supported constant-only seeds produce regions with an empty dependency list.
"""
function prepare_scalar_region(
    program::Program,
    target_ty::Ty,
    type_context,
    path::AbstractVector{<:Integer};
    input_types=Dict(),
    named_input_policy::Symbol=:used,
    analysis::Union{Nothing,TypedAnalysis}=nothing,
    max_nodes::Int=100_000,
    max_depth::Int=512,
)
    _valid_bounds(max_nodes, max_depth)
    context = _merged_type_context(program, type_context, input_types)
    source = if analysis !== nothing
        analysis
    else
        prepared = prepare_source(program.tree, context, target_ty;
            max_nodes, max_depth)
        prepared isa Err && return _source_failure(prepared)
        prepared
    end
    if analysis !== nothing
        source.index.tree === program.tree ||
            return _region_error(:source_changed, :analysis_tree_mismatch)
        isempty(TypeSystem.collect_free_vars(target_ty)) ||
            return _source_failure(Err(SourceBoundaryError(:open_target, 0)))
        length(source.index.nodes) <= max_nodes ||
            return _source_failure(Err(SourceBoundaryError(:source_bounds, 0)))
        (isempty(source.index.depth) ? 0 : maximum(source.index.depth)) <= max_depth ||
            return _source_failure(Err(SourceBoundaryError(:source_bounds, 0)))
        isequal(prune(source.root_type), prune(target_ty)) ||
            return _region_error(:source_changed, :analysis_target_mismatch)
    end
    hole = source_hole(source, path)
    hole isa Err && return _source_failure(hole)
    region_type = prune(source.metadata[hole.occurrence][1])
    _exact_float(region_type) || return _region_error(
        :non_float_region,
        (path=Int[path...], type=region_type),
    )

    index_program = program
    if source.index.tree !== program.tree
        index_program = _publish_program(source.index.tree,
            Dict{Symbol,Any}(program.local_consts),
            Dict{Symbol,TyLike}(program.local_consts_ty))
    end
    region_index = scalar_region_index(
        index_program, source; type_context, input_types, named_input_policy)
    facts = scalar_region_facts(region_index, hole.occurrence)
    if facts === nothing
        ScalarRegions._has_supported_seed(region_index, hole.occurrence) ||
            return _region_error(:unsupported_seed)
        return _region_error(:unsupported_seed)
    end
    facts.occurrence == hole.occurrence ||
        return _region_error(:source_changed, :fact_occurrence_mismatch)
    dependencies = AstNode[port.node for port in facts.available_ports]
    dependency_types = Ty[port.ty for port in facts.available_ports]
    named = Symbol[port.name::Symbol for port in facts.available_ports
        if port.kind === :named_input]
    expression = instantiate_scalar_expression(facts)
    unary_operators, binary_operators = facts.unary_operators, facts.binary_operators
    source_node_count = length(source.index.nodes)
    original_region_node_count = source.index.subtree_end[hole.occurrence] -
        hole.occurrence
    region_depth = source.index.depth[hole.occurrence]
    fragment = _scalar_fragment_shape(
        expression,
        length(dependencies),
        unary_operators,
        binary_operators,
        max_nodes - (source_node_count - original_region_node_count),
        region_depth,
        max_depth,
    )
    fragment isa Err && return _region_error(:unsupported_seed, fragment.value)
    fragment.ast_nodes == original_region_node_count || return _region_error(
        :unsupported_seed,
        (reason=:seed_shape_mismatch, source=original_region_node_count,
            expression=fragment.ast_nodes),
    )
    reserved_names = _collect_namespace!(Set{Symbol}(), source.index.tree.root)
    union!(reserved_names, keys(context))
    union!(reserved_names, keys(program.local_consts))
    union!(reserved_names, keys(program.local_consts_ty))

    owned_program = _publish_program(
        source.index.tree,
        copy(program.local_consts),
        Dict{Symbol,TyLike}(program.local_consts_ty),
    )
    return ScalarRegion(
        owned_program,
        target_ty,
        source,
        hole,
        Int[path...],
        _RegionCertificate(Tuple(Int[path...])),
        dependencies,
        _RegionCertificate(Tuple(dependencies)),
        dependency_types,
        named,
        expression,
        context,
        _type_context_certificate(context),
        reserved_names,
        unary_operators,
        binary_operators,
        source_node_count,
        original_region_node_count,
        region_depth,
        max_nodes,
        max_depth,
    )
end

function _collect_namespace!(names::Set{Symbol}, node::AstNode)
    if node isa Union{ConstNode,VarNode}
        push!(names, node.name)
    elseif node isa AbsNode
        push!(names, node.name)
    end
    for child in children(node)
        _collect_namespace!(names, child)
    end
    return names
end

function _source_index_is_current(source::TypedAnalysis)
    index = source.index
    isempty(index.nodes) && return false
    index.tree.root === index.nodes[1] || return false
    length(index.nodes) == length(index.children) || return false
    for occurrence in eachindex(index.nodes)
        child_nodes = children(index.nodes[occurrence])
        child_ids = index.children[occurrence]
        length(child_nodes) == length(child_ids) || return false
        for edge in eachindex(child_ids)
            child_nodes[edge] === index.nodes[child_ids[edge]] || return false
        end
    end
    return true
end

function _validate_scalar_region(region::ScalarRegion)
    _certificate_matches(_type_context_certificate(region.type_context),
        region.type_context_certificate) ||
        return _region_error(:source_context_changed)
    region.program.tree === region.source.index.tree ||
        return _region_error(:source_changed)
    _source_index_is_current(region.source) ||
        return _region_error(:source_changed)
    region.hole.source === region.source || return _region_error(:source_changed)
    _certificate_matches(_RegionCertificate(Tuple(region.path)),
        region.path_certificate) ||
        return _region_error(:source_changed)
    region.hole.path == region.path || return _region_error(:source_changed)
    current_hole = source_hole(region.source, region.path)
    current_hole isa Err && return _region_error(:source_changed, current_hole.value)
    current_hole.occurrence == region.hole.occurrence ||
        return _region_error(:source_changed)
    current_hole.external_types == region.hole.external_types ||
        return _region_error(:source_changed)
    _certificate_matches(_RegionCertificate(Tuple(region.dependencies)),
        region.dependency_certificate) ||
        return _region_error(:source_changed, :dependencies)
    all(_exact_float, region.dependency_types) ||
        return _region_error(:source_changed, :dependency_types)
    _exact_float(region.source.metadata[region.hole.occurrence][1]) ||
        return _region_error(:source_changed)
    Set(keys(region.program.local_consts)) ==
        Set(keys(region.program.local_consts_ty)) ||
        return _region_error(:source_changed, :local_constant_keys)
    for (name, ty) in region.program.local_consts_ty
        haskey(region.type_context, name) &&
            isequal(region.type_context[name], ty) ||
            return _region_error(:source_context_changed, name)
    end
    names = _collect_namespace!(Set{Symbol}(), region.source.index.tree.root)
    union!(names, keys(region.type_context))
    union!(names, keys(region.program.local_consts))
    union!(names, keys(region.program.local_consts_ty))
    names == region.reserved_names || return _region_error(:source_changed, :namespace)
    return nothing
end

function _fresh_region_constant(
    value::Float64,
    reserved::Set{Symbol},
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    next_index::Base.RefValue{Int},
)
    while true
        next_index[] += 1
        name = Symbol("__scalar_region_const_", next_index[])
        name in reserved && continue
        haskey(local_consts, name) && continue
        haskey(local_consts_ty, name) && continue
        local_consts[name] = value
        local_consts_ty[name] = FloatTy
        return ConstNode(name)
    end
end

function _path_copy_replace(node::AstNode, path::Vector{Int}, position::Int,
                            replacement::AstNode)
    position > length(path) && return replacement
    edge = path[position]
    if node isa AppNode
        edge in 1:2 || return _region_error(:invalid_path, path)
        func = edge == 1 ? _path_copy_replace(node.func, path, position + 1,
            replacement) : node.func
        func isa Err && return func
        arg = edge == 2 ? _path_copy_replace(node.arg, path, position + 1,
            replacement) : node.arg
        arg isa Err && return arg
        return AppNode(func, arg)
    elseif node isa AbsNode && edge == 1
        body = _path_copy_replace(node.body, path, position + 1, replacement)
        body isa Err && return body
        return AbsNode(node.name, body)
    elseif node isa FunctionDefNode && edge == 1
        body = _path_copy_replace(node.body, path, position + 1, replacement)
        body isa Err && return body
        return FunctionDefNode(body)
    end
    return _region_error(:invalid_path, path)
end

function _borrow_scalar_candidate(region::ScalarRegion, expression::ScalarExpr)
    fragment = _scalar_fragment_shape(
        expression,
        length(region.dependencies),
        region.unary_operators,
        region.binary_operators,
        region.max_nodes -
            (region.source_node_count - region.original_region_node_count),
        region.region_depth,
        region.max_depth,
    )
    fragment isa Err && return fragment

    if ScalarSearch._same_scalar_expression(region.expr, expression)
        return _publish_program(
            region.source.index.tree,
            Dict{Symbol,Any}(region.program.local_consts),
            Dict{Symbol,TyLike}(region.program.local_consts_ty),
        )
    end

    local_consts = Dict{Symbol,Any}(region.program.local_consts)
    local_consts_ty = Dict{Symbol,TyLike}(region.program.local_consts_ty)
    next_index = Ref(0)
    replacement = scalar_to_ast(expression, region.dependencies;
        materialize_constant=value -> _fresh_region_constant(
            Float64(value), region.reserved_names, local_consts,
            local_consts_ty, next_index),
    )
    replacement isa Err && return _region_error(:materialization, replacement.value)
    root = _path_copy_replace(region.source.index.tree.root,
        region.path, 1, replacement)
    root isa Err && return root
    return _publish_program(AstTree(root), local_consts, local_consts_ty)
end

"""
    materialize_scalar_region(region, expr) -> Program | Err

Replace one selected occurrence and publish an independently owned `Program`.
The construction certificate proves type preservation: the source occurrence
is exactly Float; dependencies retain their source binding/Float type; each
admitted operator has a closed Float signature in the captured type context;
generated constants are finite Float values under hygienic fresh names; and
the whole-tree node/depth bounds are checked exactly. The final constructor
copies syntax and local values. Hot objectives use the internal borrowed form
instead, which path-copies only ancestors and must be scored synchronously
without mutation or retention.
"""
function materialize_scalar_region(region::ScalarRegion, expr)
    expr isa ScalarExpr || return _region_error(:invalid_expression, typeof(expr))
    valid = _validate_scalar_region(region)
    valid === nothing || return valid
    candidate = _borrow_scalar_candidate(region, expr)
    candidate isa Err && return candidate
    return Program(candidate.tree;
        local_consts=candidate.local_consts,
        local_consts_ty=candidate.local_consts_ty,
    )
end

function _result_property(result, name::Symbol, fallback)
    return hasproperty(result, name) ? getproperty(result, name) : fallback
end

function _search_entries(result, max_return::Int)
    entries = NamedTuple[]
    if hasproperty(result, :best) && hasproperty(result, :best_loss)
        best = result.best
        best_loss = Float64(result.best_loss)
        isfinite(best_loss) && push!(entries, (
            expr=deepcopy(best),
            loss=best_loss,
            complexity=scalar_complexity(best),
        ))
    end
    if hasproperty(result, :complexity_archive)
        archive = collect(result.complexity_archive)
        sort!(archive; by=entry -> (
            Float64(entry.loss), Int(entry.complexity)))
        append!(entries, (
            expr=deepcopy(entry.expr),
            loss=Float64(entry.loss),
            complexity=Int(entry.complexity),
        ) for entry in archive if isfinite(entry.loss))
    end
    # Retain several archive sizes while avoiding repeated expressions.  Equal
    # loss/size does not imply equal scalar structure, so it is not a dedupe
    # key.
    selected = NamedTuple[]
    seen = ScalarExpr[]
    for entry in entries
        any(previous -> ScalarSearch._same_scalar_expression(
            previous, entry.expr), seen) && continue
        push!(seen, entry.expr)
        push!(selected, entry)
        length(selected) >= max_return && break
    end
    return selected
end

function _material_coverage_key(entry)
    root = first(ScalarSearch._structure_signature(entry.expr).tokens)
    return (
        entry.complexity,
        ScalarSearch._required_features(entry.expr),
        String(root.kind),
        root.payload,
    )
end

function _material_expression_key(expression::ScalarExpr)
    signature = ScalarSearch._expression_signature(expression)
    return Tuple((String(token.kind), token.payload, token.value)
        for token in signature.tokens)
end

function _select_material_entries(entries, limit::Int)
    isempty(entries) && return NamedTuple[]
    ordered = sort!(entries; by=entry -> (
        _material_coverage_key(entry),
        _material_expression_key(entry.expr),
    ))
    coverage_keys = Tuple{Int,Int,String,Int}[]
    groups = Vector{NamedTuple}[]
    for entry in ordered
        key = _material_coverage_key(entry)
        index = findfirst(existing -> existing == key, coverage_keys)
        if index === nothing
            push!(coverage_keys, key)
            push!(groups, NamedTuple[])
            index = length(groups)
        end
        push!(groups[index], entry)
    end

    selected = NamedTuple[]
    while length(selected) < limit
        added = false
        for group in groups
            isempty(group) && continue
            push!(selected, popfirst!(group))
            added = true
            length(selected) >= limit && break
        end
        added || break
    end
    return selected
end

function _material_entries(result, quality_entries, limit::Int)
    limit == 0 && return NamedTuple[]
    hasproperty(result, :leaders) || return NamedTuple[]

    entries = NamedTuple[]
    for leader in result.leaders
        expression = leader.expr
        expression isa ScalarExpr || continue
        loss = Float64(leader.loss)
        isfinite(loss) || continue
        any(entry -> ScalarSearch._same_scalar_expression(
            entry.expr, expression), quality_entries) && continue
        any(entry -> ScalarSearch._same_scalar_expression(
            entry.expr, expression), entries) && continue
        push!(entries, (
            expr=deepcopy(expression),
            loss,
            complexity=Int(leader.complexity),
        ))
    end
    return _select_material_entries(entries, limit)
end

function _scalar_region_objective(region::ScalarRegion, score)
    objective_calls = Ref(0)
    hot_materialization_failures = Ref(0)
    function contextual_objective(expression::ScalarExpr)
        objective_calls[] += 1
        candidate = _borrow_scalar_candidate(region, expression)
        if candidate isa Err
            hot_materialization_failures[] += 1
            return candidate
        end
        value = score(candidate)
        value isa Err && return value
        value isa Real || throw(ArgumentError(
            "scalar-region score must return Real or Err",
        ))
        return value
    end
    return (; contextual_objective, objective_calls, hot_materialization_failures)
end

"""
    search_scalar_region(region, score; config, rng, max_return=4,
        material_return=0, stop_loss=nothing, allow_zero_input=true,
        search_backend=search_scalar)

Run one bounded scalar-search session seeded by the ordinary parent region.
Every structural or constant-refinement objective entrance builds a restricted
borrowed `Program` by replacing the selected subtree and copying only its
ancestors. Its unchanged subtrees and value payloads are read-only borrows;
`score(program)` must be synchronous and must not mutate or retain the
candidate. The production workflow honors this contract and prepares its
evaluator with `copy_tree=false`. Whole enclosing semantics remain scored, so
folds, recursion, and all other context stay in `C[e′]`. Type, binding, name,
and tree-bound checks use the captured `TypedAnalysis` plus the finite scalar
operator language; no full W/`prepare_source` is run per hot candidate.
Materialisation refusals remain charged objective entrances but do not call
`score`; `diagnostics.objective_calls` counts these entrances, not just
successful calls through to the scorer. The backend keyword follows the
`search_scalar(X, y; config, rng,
initial_population, objective, stop_loss)` contract and lets a caller
coordinator provide a multi-island backend. For state retained across calls,
use `initialize_scalar_region_session` and `advance_scalar_region!`.

The return value is `(candidates, diagnostics)`; each returned candidate has
an independently owned `program`, `expr`, `loss`, `complexity`, and `role`.
`role=:quality` marks entries from the best/complexity archive output;
`role=:material` marks a bounded, deterministic coverage sample from the
backend's exposed `leaders` projection of its final live population. The
material limit is independent of `max_return` and does not score or consume
search randomness. Final candidates cross the ordinary `Program` ownership
boundary and are reported separately from structural and constant-refinement
calls. Set `allow_zero_input=false` to disable sessions whose dependency list
is empty. This applies to actual zero-port regions; a constant seed with
visible ports remains eligible even when it uses none of those ports.
"""
function search_scalar_region(
    region::ScalarRegion,
    score;
    config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
    rng::AbstractRNG=default_rng(),
    max_return::Int=4,
    material_return::Int=0,
    stop_loss=nothing,
    allow_zero_input::Bool=true,
    search_backend=search_scalar,
)
    max_return > 0 || throw(ArgumentError("max_return must be positive"))
    material_return >= 0 || throw(ArgumentError("material_return cannot be negative"))
    feature_count = length(region.dependencies)
    valid = _validate_scalar_region(region)
    valid === nothing || return valid
    feature_count == 0 && !allow_zero_input &&
        return _region_error(:disabled_zero_input, nothing)
    # ScalarSearch's ordinary matrix API is retained as a backend contract.
    # These placeholders are never read by contextual_objective or score.
    X = zeros(Float64, feature_count, 1)
    y = zeros(Float64, 1)
    tracked = _scalar_region_objective(region, score)
    initial = [deepcopy(region.expr)]
    result = search_backend(X, y;
        config,
        rng,
        initial_population=initial,
        objective=tracked.contextual_objective,
        stop_loss,
    )
    result isa Err && return result
    return _scalar_region_result(region, result, tracked.objective_calls[],
        tracked.hot_materialization_failures[]; max_return, material_return)
end

function _scalar_region_result(region, result, objective_calls,
        hot_materialization_failures; max_return, material_return)
    entries = _search_entries(result, max_return)
    quality_candidates = NamedTuple[]
    boundary_rechecks = 0
    boundary_failures = 0
    for entry in entries
        boundary_rechecks += 1
        candidate = materialize_scalar_region(region, entry.expr)
        if candidate isa Err
            boundary_failures += 1
            continue
        end
        push!(quality_candidates, (
            program=candidate,
            expr=entry.expr,
            loss=entry.loss,
            complexity=entry.complexity,
            role=:quality,
        ))
        length(quality_candidates) >= max_return && break
    end

    material_entries = _material_entries(result, entries, material_return)
    material_candidates = NamedTuple[]
    material_boundary_rechecks = 0
    material_boundary_failures = 0
    for entry in material_entries
        material_boundary_rechecks += 1
        candidate = materialize_scalar_region(region, entry.expr)
        if candidate isa Err
            material_boundary_failures += 1
            continue
        end
        push!(material_candidates, (
            program=candidate,
            expr=entry.expr,
            loss=entry.loss,
            complexity=entry.complexity,
            role=:material,
        ))
    end
    candidates = [quality_candidates; material_candidates]
    isempty(candidates) && return _region_error(:no_valid_candidate, (
        boundary_rechecks,
        boundary_failures,
    ))

    structural_evaluations = _result_property(result, :evaluations, 0)
    invalid_evaluations = _result_property(result, :invalid_evaluations, 0)
    refinement_evaluations = _result_property(
        result, :constant_refinement_evaluations, 0)
    refinement_invalid = _result_property(
        result, :constant_refinement_invalid_evaluations, 0)
    diagnostics = (;
        structural_evaluations,
        invalid_evaluations,
        refinement_evaluations,
        refinement_invalid,
        evaluations=structural_evaluations,
        constant_refinement_evaluations=refinement_evaluations,
        constant_refinement_invalid_evaluations=refinement_invalid,
        objective_calls,
        hot_materialization_failures,
        boundary_rechecks,
        boundary_failures,
        returned_quality_candidates=length(quality_candidates),
        material_boundary_rechecks,
        material_boundary_failures,
        returned_material_candidates=length(material_candidates),
        returned_candidates=length(candidates),
        backend_result=result,
    )
    return (; candidates, diagnostics)
end

"""An exclusively owned multi-island search over one fixed contextual objective.

The region is copied at initialization. The score callback and any task data it
captures must remain stable and read-only for this session's entire lifetime.
Neither family membership nor an equal hole type permits transferring this
state or its scores to a different context. Published programs are independent
copies; the session continues to develop its original context.
Use only the initialize/advance/result API. All fields are internal borrowed
state: mutating them invalidates stored scores and is unsupported. The session
does not fingerprint arbitrary user mutations to internal values or callbacks.
"""
struct ScalarRegionSession{S}
    _region::ScalarRegion
    _backend::S
    _objective_calls::Base.RefValue{Int}
    _hot_materialization_failures::Base.RefValue{Int}
    _max_return::Int
    _material_return::Int
end

"""
    initialize_scalar_region_session(region, score; config, rng,
        max_return=4, material_return=0, allow_zero_input=true,
        stop_loss=nothing, backend_options=(;))

Initialize the existing multi-island scalar coordinator once. `backend_options`
sets coordinator options such as `island_count`, total `rounds` and migration;
`config` supplies the lifetime structural and constant budgets. Preparation
refusals return `Err`; invalid options and unexpected objective errors propagate.
The callback has the same synchronous borrowed-program contract as
`search_scalar_region`. No intermediate teacher labels are supplied.
"""
function initialize_scalar_region_session(region::ScalarRegion, score;
        config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
        rng::AbstractRNG=default_rng(), max_return::Int=4,
        material_return::Int=0, allow_zero_input::Bool=true,
        stop_loss=nothing, backend_options::NamedTuple=(;))
    max_return > 0 || throw(ArgumentError("max_return must be positive"))
    material_return >= 0 || throw(ArgumentError("material_return cannot be negative"))
    any(key -> key in (:config, :rng, :initial_population, :objective, :stop_loss),
        keys(backend_options)) && throw(ArgumentError(
            "backend_options cannot override the captured scalar-region contract"))
    valid = _validate_scalar_region(region)
    valid === nothing || return valid
    feature_count = length(region.dependencies)
    feature_count == 0 && !allow_zero_input &&
        return _region_error(:disabled_zero_input, nothing)
    owned = deepcopy(region)
    tracked = _scalar_region_objective(owned, score)
    backend = ScalarSearch.initialize_multi_scalar(
        zeros(Float64, feature_count, 1), zeros(Float64, 1);
        config, rng, initial_population=[deepcopy(owned.expr)],
        objective=tracked.contextual_objective, stop_loss, backend_options...)
    backend isa Err && return backend
    return ScalarRegionSession(owned, backend, tracked.objective_calls,
        tracked.hot_materialization_failures, max_return, material_return)
end

"""Advance at most `max_rounds` existing coordinator rounds, retaining all state.

This is a work-slice limit, not a fresh evaluation budget. Returns the session;
use `scalar_region_result` to publish detached candidates and cumulative counts.
An unexpected objective or observer exception propagates; discard that session
because a partially advanced round is not transactional.
"""
function advance_scalar_region!(session::ScalarRegionSession; max_rounds::Int=1)
    ScalarSearch.advance_multi_scalar!(session._backend; max_rounds)
    return session
end

"""Publish owned candidates and cumulative diagnostics without scoring or RNG draws."""
function scalar_region_result(session::ScalarRegionSession)
    result = ScalarSearch.multi_scalar_result(session._backend)
    return _scalar_region_result(session._region, result, session._objective_calls[],
        session._hot_materialization_failures[];
        max_return=session._max_return, material_return=session._material_return)
end

"""Read cumulative coordinator progress even when no valid candidate can be published.
Counters cover the full captured-context objective, including materialization
refusals; they do not include later migrant rescoring in the outer population.
"""
scalar_region_progress(session::ScalarRegionSession) =
    ScalarSearch.multi_scalar_progress(session._backend)

@testitem "scalar region sessions retain fixed context and detached publication" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: ScalarRegionSession, prepare_scalar_region,
        search_scalar_region, initialize_scalar_region_session,
        advance_scalar_region!, scalar_region_result
    using LambdaRegression.ScalarSearch: ScalarSearchConfig, scalar_from_ast,
        evaluate_scalar_reference, search_multi_scalar
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    parent = Program(AstTree(AbsNode(:x,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:offset))));
        local_consts=Dict{Symbol,Any}(:offset => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:offset => FloatTy))
    types = Dict{Symbol,TyLike}(
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), types, [1, 1, 2])
    X = reshape([-1.0, 0.0, 1.0], 1, :)
    y = [3.0, 2.0, 3.0]
    function scorer(trace)
        return program -> begin
            expression = scalar_from_ast(program.tree.root.body, AstNode[IndexNode(0)];
                constant_values=program.local_consts)
            expression isa Err && return expression
            prediction = evaluate_scalar_reference(expression, X)
            prediction isa Err && return prediction
            push!(trace, copy(prediction))
            sum(abs2, prediction .- y) / length(y)
        end
    end
    config = ScalarSearchConfig(population_size=3, max_evaluations=24,
        max_complexity=6, initial_depth=1, tournament_size=2, elite_count=1,
        unary_operators=(), binary_operators=(:+,),
        constant_refinement_iterations=0, evolution_strategy=:regularized)
    options = (; island_count=2, rounds=3, ncycles_per_iteration=1)
    full_trace, split_trace = Any[], Any[]
    backend = (X, y; kwargs...) -> search_multi_scalar(X, y; kwargs..., options...)
    full = search_scalar_region(region, scorer(full_trace); config,
        rng=MersenneTwister(1), search_backend=backend, max_return=2, material_return=1)
    session = initialize_scalar_region_session(region, scorer(split_trace); config,
        rng=MersenneTwister(1), backend_options=options, max_return=2, material_return=1)
    @test session isa ScalarRegionSession
    @test session._region !== region
    # The caller's old context and a published candidate cannot mutate the session.
    region.program.local_consts[:offset] = 99.0
    calls_before = length(split_trace)
    initial_result = scalar_region_result(session)
    @test length(split_trace) == calls_before
    initial_result.candidates[1].program.local_consts[:offset] = -99.0
    @test session._region.program.local_consts[:offset] == 2.0
    @test advance_scalar_region!(session; max_rounds=0) === session
    @test length(split_trace) == calls_before
    for _ in 1:3
        advance_scalar_region!(session)
        scalar_region_result(session)
    end
    result = scalar_region_result(session)
    @test full_trace == split_trace
    @test result.diagnostics.objective_calls == full.diagnostics.objective_calls
    @test result.diagnostics.backend_result.diagnostics.rounds_completed == 3
    @test [(c.loss, c.complexity, c.role) for c in result.candidates] ==
        [(c.loss, c.complexity, c.role) for c in full.candidates]
    @test all(c.program.local_consts[:offset] == 2.0 for c in result.candidates)
    calls_done = length(split_trace)
    advance_scalar_region!(session; max_rounds=10)
    @test length(split_trace) == calls_done
    @test_throws ArgumentError initialize_scalar_region_session(region, scorer(Any[]);
        config, backend_options=(objective=identity,))
end

@testitem "scalar region certificates preserve structural equality without shape specialisation" begin
    using Test
    using LambdaRegression.Regression
    using LambdaRegression.TypeSystem:
        BoolTy, FloatTy, TyLike, TyFun, TyList, TyMu, TyScheme, newtyvar

    R = LambdaRegression.Regression
    pairs = (
        ((:x, (1, Int8(2))), (:x, (1, 2))),
        ((missing, 1), (0, 1)),
        ((missing, 1), (0, 2)),
        ((NaN, 1), (NaN, 1)),
        ((-0.0, 1), (0.0, 1)),
        (((:x, 1),), ((:x, 2),)),
    )
    for (left, right) in pairs
        actual = R._certificate_matches(
            R._RegionCertificate(left), R._RegionCertificate(right))
        @test isequal(actual, left == right)
    end

    first_var = newtyvar(0)
    second_var = newtyvar(0)
    recursive_list() = (binder = newtyvar(0); TyMu(binder, TyList(binder)))
    left_context = Dict{Symbol,TyLike}(
        :a => TyScheme([first_var], TyFun(first_var, first_var)),
        :z => recursive_list(),
        :f => TyFun(FloatTy, FloatTy),
    )
    right_context = Dict{Symbol,TyLike}(
        :f => TyFun(FloatTy, FloatTy),
        :z => recursive_list(),
        :a => TyScheme([second_var], TyFun(second_var, second_var)),
    )
    original = R._type_context_certificate(left_context)
    @test R._certificate_matches(R._type_context_certificate(right_context), original)
    right_context[:f] = TyFun(BoolTy, FloatTy)
    @test !R._certificate_matches(R._type_context_certificate(right_context), original)
    delete!(right_context, :f)
    @test !R._certificate_matches(R._type_context_certificate(right_context), original)
    right_context[:f] = TyFun(FloatTy, FloatTy)
    right_context[:new] = FloatTy
    @test !R._certificate_matches(R._type_context_certificate(right_context), original)
end

@testitem "scalar region captures binder and named Float dependencies" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypedSearch: TypedAnalysis, prepare_source
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type
    using LambdaRegression.Primitives: FADD

    tree = AstTree(AbsNode(:x, AppNode(
        AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:u))))
    target = TyFun(FloatTy, FloatTy)
    context = Dict{Symbol,TyLike}(:FADD => get_type(FADD))
    region = prepare_scalar_region(
        Program(tree), target, context, [1];
        input_types=Dict{Symbol,TyLike}(:u => FloatTy),
    )
    @test region isa ScalarRegion
    @test region.dependencies == [IndexNode(0), ConstNode(:u)]
    @test ScalarSearch._required_features(region.expr) == 2

    changed = materialize_scalar_region(
        region,
        ScalarSearch.scalar_binary(:+, ScalarSearch.scalar_input(1),
            ScalarSearch.scalar_input(2)),
    )
    @test changed isa Program
    @test changed.tree.root isa AbsNode
    @test changed.tree.root.body isa AppNode
    @test prepare_source(
        changed.tree,
        merge(
            Dict{Symbol,TyLike}(:FADD => get_type(FADD), :u => FloatTy),
            changed.local_consts_ty,
        ),
        target,
    ) isa TypedAnalysis
end

@testitem "scalar region constants are hygienic and owned" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type
    using LambdaRegression.Primitives: FADD

    body = AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:c))
    parent = Program(
        AstTree(AbsNode(:x, body));
        local_consts=Dict{Symbol,Any}(:c => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy),
    )
    context = Dict{Symbol,TyLike}(:FADD => get_type(FADD))
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), context, [1])
    @test region isa ScalarRegion
    before = deepcopy(parent.tree)
    first_candidate = materialize_scalar_region(
        region,
        ScalarSearch.scalar_binary(:+, ScalarSearch.scalar_input(1),
            ScalarSearch.scalar_constant(3.0)),
    )
    second_candidate = materialize_scalar_region(
        region,
        ScalarSearch.scalar_binary(:+, ScalarSearch.scalar_input(1),
            ScalarSearch.scalar_constant(4.0)),
    )
    @test first_candidate isa Program
    @test second_candidate isa Program
    @test parent.tree == before
    @test first_candidate.local_consts[:c] == 2.0
    @test first_candidate.local_consts_ty[:c] == FloatTy
    generated = setdiff(Set(keys(first_candidate.local_consts)), Set((:c,)))
    @test !isempty(generated)
    only_generated = first(generated)
    first_candidate.local_consts[only_generated] = 99.0
    @test second_candidate.local_consts[only_generated] != 99.0
    @test first_candidate.tree !== second_candidate.tree
end

@testitem "scalar region exact seed roundtrip preserves owned namespace" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, get_type
    using LambdaRegression.Primitives: FADD

    parent = Program(
        AstTree(AbsNode(:x, AppNode(
            AppNode(ConstNode(:FADD), IndexNode(0)),
            AppNode(AppNode(ConstNode(:FADD), ConstNode(:c)), ConstNode(:c)))));
        local_consts=Dict{Symbol,Any}(:c => 3.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy),
    )
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), Dict{Symbol,TyLike}(:FADD => get_type(FADD)), [1])
    @test region isa ScalarRegion
    roundtrip = materialize_scalar_region(region, region.expr)
    @test roundtrip isa Program
    @test roundtrip.tree == region.program.tree
    @test roundtrip.local_consts == region.program.local_consts == parent.local_consts
    @test roundtrip.local_consts_ty == region.program.local_consts_ty
    @test roundtrip.tree !== parent.tree
    @test roundtrip.local_consts !== parent.local_consts

    changed = materialize_scalar_region(
        region,
        ScalarSearch.scalar_binary(:+, ScalarSearch.scalar_input(1),
            ScalarSearch.scalar_constant(1.0)),
    )
    @test changed isa Program
    @test changed.tree != roundtrip.tree
    @test haskey(changed.local_consts, :c)
end

@testitem "scalar hot candidates path-copy and publish owned programs" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        _borrow_scalar_candidate, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    body = AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), IndexNode(0))
    parent = Program(AstTree(AbsNode(:x, body)))
    context = Dict{Symbol,TyLike}(
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :FMUL => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
    )
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), context, [1, 2];
        max_nodes=32, max_depth=8)
    @test region isa LambdaRegression.Regression.ScalarRegion

    first_expr = scalar_binary(:+, scalar_input(1), scalar_constant(2.0))
    second_expr = scalar_binary(:*, scalar_input(1), scalar_constant(3.0))
    first_view = _borrow_scalar_candidate(region, first_expr)
    second_view = _borrow_scalar_candidate(region, second_expr)
    @test first_view isa Program
    @test second_view isa Program
    @test first_view.tree !== second_view.tree
    @test first_view.tree.root !== region.source.index.tree.root
    @test first_view.tree.root.body.func === region.source.index.tree.root.body.func
    @test first_view.local_consts !== second_view.local_consts
    first_name = only(setdiff(Set(keys(first_view.local_consts)), Set{Symbol}()))
    second_name = only(setdiff(Set(keys(second_view.local_consts)), Set{Symbol}()))
    @test first_view.local_consts[first_name] == 2.0
    @test second_view.local_consts[second_name] == 3.0
    @test first_view.local_consts_ty[first_name] == FloatTy

    published = materialize_scalar_region(region, first_expr)
    @test published isa Program
    @test published.tree !== first_view.tree
    @test published.tree.root.body.func !== first_view.tree.root.body.func
    @test parent.tree.root.body == body
end

@testitem "scalar hot candidates reject stale contexts, illegal IR, and bounds" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        _borrow_scalar_candidate, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem:
        FloatTy, TyFun, TyLike, TyScheme, newtyvar
    using LambdaRegression.Utils: Err

    target = TyFun(FloatTy, FloatTy)
    add_var = newtyvar(0)
    custom_var = newtyvar(0)
    context = Dict{Symbol,TyLike}(
        :FADD => TyScheme([add_var], TyFun(add_var, TyFun(add_var, add_var))),
        :CUSTOM => TyScheme([custom_var], TyFun(custom_var, custom_var)),
    )
    parent = Program(AstTree(AbsNode(:x, IndexNode(0))))
    region = prepare_scalar_region(parent, target, context, [1];
        max_nodes=16, max_depth=8)
    @test region isa LambdaRegression.Regression.ScalarRegion

    generic_add = materialize_scalar_region(region,
        scalar_binary(:+, scalar_input(1), scalar_input(1)))
    @test generic_add isa Program # W can instantiate the declared scheme at Float.

    invalid_op = LambdaRegression.ScalarSearch._ScalarUnary(
        :custom, scalar_input(1))
    unsupported = materialize_scalar_region(region, invalid_op)
    @test unsupported isa Err
    @test unsupported.value.reason == :unsupported_operator
    invalid_input = LambdaRegression.ScalarSearch._ScalarInput(0)
    missing = materialize_scalar_region(region, invalid_input)
    @test missing isa Err
    @test missing.value.reason == :missing_dependency
    nonfinite = LambdaRegression.ScalarSearch._ScalarConstant(Inf)
    bad_constant = materialize_scalar_region(region, nonfinite)
    @test bad_constant isa Err
    @test bad_constant.value.reason == :nonfinite_constant

    size_limited = prepare_scalar_region(parent, target, context, [1];
        max_nodes=2, max_depth=8)
    too_large = materialize_scalar_region(size_limited,
        scalar_binary(:+, scalar_input(1), scalar_input(1)))
    @test too_large isa Err
    @test too_large.value.reason == :materialization_size
    depth_limited = prepare_scalar_region(parent, target, context, [1];
        max_nodes=16, max_depth=1)
    too_deep = materialize_scalar_region(depth_limited,
        scalar_binary(:+, scalar_input(1), scalar_input(1)))
    @test too_deep isa Err
    @test too_deep.value.reason == :materialization_depth

    region.type_context[:FADD] = FloatTy
    stale = materialize_scalar_region(region, scalar_input(1))
    @test stale isa Err
    @test stale.value.reason == :source_context_changed

    stale_path = prepare_scalar_region(parent, target, context, [1];
        max_nodes=16, max_depth=8)
    stale_path.path[1] = 2
    @test materialize_scalar_region(stale_path, scalar_input(1)).value.reason ==
        :source_changed

    stale_dependency = prepare_scalar_region(parent, target, context, [1];
        max_nodes=16, max_depth=8)
    stale_dependency.dependencies[1] = ConstNode(:other)
    @test materialize_scalar_region(stale_dependency, scalar_input(1)).value.reason ==
        :source_changed
end

@testitem "scalar region preserves recursive outer structure" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, _borrow_scalar_candidate, materialize_scalar_region,
        prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    target = TyFun(FloatTy, FloatTy)
    recursive = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0))))
    parent = Program(recursive)
    context = Dict{Symbol,TyLike}(:FADD => TyFun(FloatTy,
        TyFun(FloatTy, FloatTy)))
    region = prepare_scalar_region(parent, target, context, [1, 1])
    @test region isa ScalarRegion
    @test region.dependencies == [IndexNode(0)]
    expression = ScalarSearch.scalar_binary(:+,
        ScalarSearch.scalar_input(1), ScalarSearch.scalar_constant(2.0))
    hot = _borrow_scalar_candidate(region, expression)
    @test hot isa Program
    @test hot.tree.root isa FunctionDefNode
    @test hot.tree.root.body isa AbsNode
    @test hot.tree.root.body.body isa AppNode
    candidate = materialize_scalar_region(region, expression)
    @test candidate isa Program
    @test candidate.tree.root isa FunctionDefNode
    @test candidate.tree.root.body.body isa AppNode
    @test candidate.tree !== hot.tree
    @test parent.tree.root.body.body == IndexNode(0)
end

@testitem "scalar region contextual search uses whole individuals" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, prepare_scalar_region, search_scalar_region
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    parent = Program(AstTree(AbsNode(:x, IndexNode(0))))
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), Dict{Symbol,TyLike}(), [1])
    @test region isa ScalarRegion
    seen = Ref(0)
    score = individual -> begin
        seen[] += 1
        individual isa Program || error("not a Program")
        return Float64(length(individual.tree))
    end
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        max_complexity=8,
        initial_depth=1,
        tournament_size=2,
        elite_count=2,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_refinement_iterations=0,
        evolution_strategy=:regularized,
    )
    result = search_scalar_region(region, score;
        config, rng=MersenneTwister(1111), max_return=3, stop_loss=nothing)
    @test result isa NamedTuple
    @test seen[] > 0
    @test result.diagnostics.objective_calls >= seen[]
    @test result.diagnostics.structural_evaluations > 0
    @test 1 <= length(result.candidates) <= 3
    @test all(candidate -> candidate.program isa Program, result.candidates)
end

@testitem "scalar region returns bounded owned live material separately" begin
    using Test
    using Random: MersenneTwister, rand
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: prepare_scalar_region, search_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    parent = Program(AstTree(AbsNode(:x, IndexNode(0))))
    context = Dict{Symbol,TyLike}(
        :FSIN => TyFun(FloatTy, FloatTy),
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :FMUL => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
    )
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), context, [1])
    x = scalar_input(1)
    best = deepcopy(x)
    sine = scalar_unary(:sin, x)
    product = scalar_binary(:*, x, x)
    plus_expr = scalar_binary(:+, x, x)
    backend = function (X, y; config, rng, initial_population, objective, stop_loss)
        objective(first(initial_population))
        return (
            best=deepcopy(best),
            best_loss=0.1,
            evaluations=3,
            invalid_evaluations=0,
            constant_refinement_evaluations=0,
            constant_refinement_invalid_evaluations=0,
            complexity_archive=[(expr=deepcopy(best), loss=0.1, complexity=1)],
            leaders=[
                (expr=deepcopy(best), loss=0.1, complexity=1),
                (expr=deepcopy(plus_expr), loss=6.0, complexity=3),
                (expr=deepcopy(sine), loss=8.0, complexity=2),
                (expr=deepcopy(sine), loss=8.0, complexity=2),
                (expr=deepcopy(product), loss=9.0, complexity=3),
                (expr=scalar_unary(:cos, x), loss=Inf, complexity=2),
            ],
        )
    end
    score_calls = Ref(0)
    score = program -> (score_calls[] += 1; Float64(length(program.tree)))
    config = ScalarSearchConfig(
        population_size=2, max_evaluations=2, max_complexity=8,
        initial_depth=0, tournament_size=1, elite_count=1,
        unary_operators=(:sin, :cos), binary_operators=(:+, :*),
        constant_refinement_iterations=0, evolution_strategy=:regularized,
    )

    rng = MersenneTwister(1234)
    control_rng = MersenneTwister(1234)
    result = search_scalar_region(region, score; config, rng, max_return=1,
        material_return=2, stop_loss=nothing, search_backend=backend)
    @test length(result.candidates) == 3
    @test result.diagnostics.returned_quality_candidates == 1
    @test result.diagnostics.returned_material_candidates == 2
    @test result.diagnostics.material_boundary_rechecks == 2
    @test result.diagnostics.material_boundary_failures == 0
    @test result.diagnostics.objective_calls == 1
    @test [candidate.role for candidate in result.candidates] ==
        [:quality, :material, :material]
    materials = result.candidates[2:end]
    @test all(candidate -> candidate.loss > result.candidates[1].loss, materials)
    @test all(candidate -> candidate.program isa Program, materials)
    @test all(candidate -> candidate.program.tree !== region.program.tree, materials)
    @test materials[1].program.tree !== materials[2].program.tree
    @test materials[1].program.local_consts !== materials[2].program.local_consts
    @test materials[1].program.local_consts !== parent.local_consts
    @test ScalarSearch._same_scalar_expression(materials[1].expr, sine)
    @test ScalarSearch._same_scalar_expression(materials[2].expr, product)
    @test [candidate.loss for candidate in materials] == [8.0, 9.0]
    @test score_calls[] == 1
    @test rand(rng) == rand(control_rng)

    default_rng = MersenneTwister(2000)
    default_control_rng = MersenneTwister(2000)
    default_result = search_scalar_region(region, score; config,
        rng=default_rng, max_return=1, stop_loss=nothing,
        search_backend=backend)
    @test length(default_result.candidates) == 1
    @test only(default_result.candidates).role == :quality
    @test default_result.diagnostics.returned_material_candidates == 0
    @test default_result.diagnostics.material_boundary_rechecks == 0
    @test default_result.diagnostics.objective_calls == 1
    @test score_calls[] == 2
    @test rand(default_rng) == rand(default_control_rng)
end

@testitem "scalar region refuses unsupported seeds and admits constant-only regions" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, prepare_scalar_region, search_scalar_region
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    target = TyFun(FloatTy, FloatTy)
    unsupported = Program(AstTree(AbsNode(:x, AppNode(
        AppNode(ConstNode(:BAD), IndexNode(0)), IndexNode(0)))))
    context = Dict{Symbol,TyLike}(:BAD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    bad_region = prepare_scalar_region(unsupported, target, context, [1])
    @test bad_region isa Err
    @test bad_region.value.reason == :unsupported_seed

    constant = Program(AstTree(ConstNode(:c));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    zero_region = prepare_scalar_region(
        constant, FloatTy, Dict{Symbol,TyLike}(), Int[])
    @test zero_region isa ScalarRegion
    @test isempty(zero_region.dependencies)
    @test isempty(zero_region.named_dependencies)
    @test zero_region.expr isa LambdaRegression.ScalarSearch._ScalarConstant

    # A visible Float binder remains available to mutations even when the seed
    # itself is a constant.
    reachable = Program(AstTree(AbsNode(:x, ConstNode(:c)));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    reachable_region = prepare_scalar_region(
        reachable, target, Dict{Symbol,TyLike}(), [1])
    @test reachable_region isa ScalarRegion
    @test reachable_region.dependencies == [IndexNode(0)]
    score_calls = Ref(0)
    reachable_search = search_scalar_region(reachable_region, _ -> begin
        score_calls[] += 1
        return 1.0
    end; config=ScalarSearchConfig(
        population_size=2, max_evaluations=2, initial_depth=0,
        tournament_size=1, elite_count=1,
        constant_refinement_iterations=0, evolution_strategy=:regularized),
        rng=MersenneTwister(2000), stop_loss=nothing,
        allow_zero_input=false)
    @test reachable_search isa NamedTuple
    @test score_calls[] > 0
end

@testitem "zero-input contextual scalar search improves constants within budget" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, prepare_scalar_region, search_scalar_region
    using LambdaRegression.ScalarSearch:
        ScalarSearchConfig, evaluate_scalar_reference, scalar_from_ast
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    parent = Program(AstTree(ConstNode(:c));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    original_tree = deepcopy(parent.tree)
    original_constants = copy(parent.local_consts)
    context = Dict{Symbol,TyLike}(
        :FADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
        :SQR => TyFun(FloatTy, FloatTy),
    )
    region = prepare_scalar_region(
        parent, FloatTy, context, Int[])
    @test region isa ScalarRegion
    @test isempty(region.dependencies)

    X = zeros(Float64, 0, 4)
    y = fill(3.0, 4)
    score = program -> begin
        expression = scalar_from_ast(program.tree.root, AstNode[];
            constant_values=program.local_consts)
        expression isa Err && return Inf
        prediction = evaluate_scalar_reference(expression, X)
        prediction isa Err && return Inf
        sum(abs2, prediction .- y) / length(y)
    end
    config = ScalarSearchConfig(
        population_size=4,
        max_evaluations=8,
        max_complexity=6,
        initial_depth=0,
        tournament_size=2,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_probability=0.0,
        constant_refinement_iterations=24,
        constant_refinement_probability=1.0,
        constant_refinement_start_generation=1,
        max_constant_refinement_evaluations=16,
        constant_refinement_optimizer=:nelder_mead,
        initial_pool_multiplier=1,
        evolution_strategy=:generational,
    )
    disabled_objective_calls = Ref(0)
    disabled = search_scalar_region(region, _ -> begin
        disabled_objective_calls[] += 1
        return 0.0
    end; config, rng=MersenneTwister(42), max_return=1,
        stop_loss=nothing, allow_zero_input=false)
    @test disabled isa Err
    @test disabled.value.reason == :disabled_zero_input
    @test disabled_objective_calls[] == 0

    result = search_scalar_region(region, score; config,
        rng=MersenneTwister(42), max_return=1, stop_loss=nothing)
    @test result isa NamedTuple
    @test only(result.candidates).loss < 4.0
    @test result.diagnostics.structural_evaluations <= config.max_evaluations
    @test result.diagnostics.refinement_evaluations > 0
    @test result.diagnostics.refinement_evaluations <=
        config.max_constant_refinement_evaluations
    @test parent.tree == original_tree
    @test parent.local_consts == original_constants
end

@testitem "scalar region score exceptions propagate" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: prepare_scalar_region, search_scalar_region
    using LambdaRegression.ScalarSearch: ScalarSearchConfig
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    parent = Program(AstTree(AbsNode(:x, IndexNode(0))))
    region = prepare_scalar_region(parent, TyFun(FloatTy, FloatTy), Dict{Symbol,TyLike}(), [1])
    config = ScalarSearchConfig(
        population_size=2,
        max_evaluations=2,
        initial_depth=0,
        tournament_size=1,
        elite_count=1,
        unary_operators=(:square,),
        binary_operators=(:+,),
        constant_refinement_iterations=0,
        evolution_strategy=:regularized,
    )
    @test_throws ErrorException search_scalar_region(region, _ -> error("boom");
        config, stop_loss=nothing)
end
