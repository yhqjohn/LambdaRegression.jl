"""
Automatic scalar-region preparation, contextual refinement, and verified
materialisation for the program--scalar bridge.

The ordinary `Program.tree` remains authoritative.  A `ScalarRegion` owns
source evidence from `TypedSearch.prepare_source`/`source_hole`, an ordered
feature boundary, and one parsed `ScalarExpr`; it is a disposable view rather
than a second program representation.  `search_scalar_region` uses the
existing scalar population search as a bounded session.  Its low-level
`X/y` arguments are one-column zero placeholders required by that API only;
the contextual objective never reads them and scores each materialised whole
`Program` through the caller's callback.

The first boundary admits only exact `Float` regions and finite real local
constants.  Non-floating external binders, unsupported syntax, recursive
scalar regions, and zero-feature expressions stay on the complete typed
program path and are returned as classified `Err` values.
"""

using Random: AbstractRNG, default_rng

using ..Utils: Err
using ..ASTs.NodeASTs:
    AstNode, AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode,
    children
using ..Trees: setchild!
using ..Programs: Program, _publish_program
using ..ScalarSearch:
    ScalarExpr, ScalarSearchConfig, ScalarSearchResult, scalar_from_ast,
    scalar_to_ast, scalar_complexity, scalar_input, search_scalar
import ..ScalarSearch
using ..TypedSearch:
    TypedAnalysis, SourceBoundaryError, SourceHole, prepare_source, source_hole
using ..TypeSystem: FloatTy, Ty, TyLike, prune

export ScalarRegion, ScalarRegionError
export prepare_scalar_region, materialize_scalar_region, search_scalar_region

"""Expected refusal at the automatic scalar-region boundary."""
struct ScalarRegionError
    reason::Symbol
    detail::Any
end

ScalarRegionError(reason::Symbol) = ScalarRegionError(reason, nothing)

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
    dependencies::Vector{AstNode}
    dependency_types::Vector{Ty}
    named_dependencies::Vector{Symbol}
    expr::ScalarExpr
    type_context::Dict{Symbol,TyLike}
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

function _collect_var_names!(names::Set{Symbol}, node::AstNode)
    node isa VarNode && push!(names, node.name)
    for child in children(node)
        _collect_var_names!(names, child)
    end
    return names
end

function _collect_const_names!(names::Set{Symbol}, node::AstNode)
    node isa ConstNode && push!(names, node.name)
    for child in children(node)
        _collect_const_names!(names, child)
    end
    return names
end

function _rewrite_named_float_vars(node::AstNode, names::Set{Symbol})
    if node isa VarNode
        return node.name in names ? ConstNode(node.name) : node
    elseif node isa AppNode
        return AppNode(
            _rewrite_named_float_vars(node.func, names),
            _rewrite_named_float_vars(node.arg, names),
        )
    elseif node isa AbsNode
        return AbsNode(node.name, _rewrite_named_float_vars(node.body, names))
    elseif node isa FunctionDefNode
        return FunctionDefNode(_rewrite_named_float_vars(node.body, names))
    end
    return node
end

function _external_dependencies(hole::SourceHole)
    dependencies = AstNode[]
    types = Ty[]
    # SourceHole retains external types outermost first.  Reversing that
    # list gives the ordinary zero-based De Bruijn order at the selected site.
    for position in reverse(eachindex(hole.external_types))
        db = length(hole.external_types) - position
        ty = hole.external_types[position]
        _exact_float(ty) || continue
        push!(dependencies, IndexNode(db))
        push!(types, prune(ty))
    end
    return dependencies, types
end

function _named_dependencies(node::AstNode, input_types)
    # Named task inputs are explicitly declared through `input_types`.  Do
    # not treat every Float entry in a global operator context as a feature.
    names = union(
        _collect_var_names!(Set{Symbol}(), node),
        _collect_const_names!(Set{Symbol}(), node),
    )
    selected = Symbol[
        name for name in names
        if haskey(input_types, name) && _exact_float(input_types[name])
    ]
    sort!(selected; by=String)
    dependencies = AstNode[ConstNode(name) for name in selected]
    types = Ty[prune(input_types[name]) for name in selected]
    return selected, dependencies, types
end

function _constant_values(program::Program)
    values = Dict{Symbol,Float64}()
    for (name, value) in program.local_consts
        value isa Real || continue
        converted = Float64(value)
        isfinite(converted) && (values[name] = converted)
    end
    return values
end

function _source_failure(result)
    result isa Err || return _region_error(:source, result)
    detail = result.value
    reason = detail isa SourceBoundaryError ? detail.reason : :source
    return _region_error(reason, detail)
end

"""
    prepare_scalar_region(program, target_ty, type_context, path;
        input_types=Dict(), max_nodes=100_000, max_depth=512)

Prepare one exact-`Float` source occurrence for contextual scalar refinement.
The full individual is inferred first, then `path` is resolved through the
owned source evidence.  Visible floating De Bruijn binders and declared
named floating inputs become dependencies; non-floating externals are not
invented as scalar features.  Unsupported source/seed boundaries and
zero-feature regions return `Err(ScalarRegionError(...))`.
"""
function prepare_scalar_region(
    program::Program,
    target_ty::Ty,
    type_context,
    path::AbstractVector{<:Integer};
    input_types=Dict(),
    max_nodes::Int=100_000,
    max_depth::Int=512,
)
    _valid_bounds(max_nodes, max_depth)
    context = _merged_type_context(program, type_context, input_types)
    source = prepare_source(program.tree, context, target_ty;
        max_nodes, max_depth)
    source isa Err && return _source_failure(source)
    hole = source_hole(source, path)
    hole isa Err && return _source_failure(hole)
    region_type = prune(source.metadata[hole.occurrence][1])
    _exact_float(region_type) || return _region_error(
        :non_float_region,
        (path=Int[path...], type=region_type),
    )

    external_dependencies, external_types = _external_dependencies(hole)
    declared_inputs = _copy_type_context(input_types)
    occurrence_node = source.index.nodes[hole.occurrence]
    named, named_dependencies, named_types = _named_dependencies(
        occurrence_node, declared_inputs)
    dependencies = vcat(external_dependencies, named_dependencies)
    dependency_types = vcat(external_types, named_types)
    rewritten = _rewrite_named_float_vars(
        occurrence_node, Set(named))
    expression = scalar_from_ast(
        rewritten,
        dependencies;
        constant_values=_constant_values(program),
    )
    expression isa Err && return _region_error(:unsupported_seed, expression.value)
    isempty(dependencies) &&
        return _region_error(:zero_features, (path=Int[path...],))

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
        dependencies,
        dependency_types,
        named,
        expression,
        context,
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

function _fresh_region_constant(
    value::Float64,
    reserved::Set{Symbol},
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
)
    index = 0
    while true
        index += 1
        name = Symbol("__scalar_region_const_", index)
        name in reserved && continue
        push!(reserved, name)
        local_consts[name] = value
        local_consts_ty[name] = FloatTy
        return ConstNode(name)
    end
end

function _replace_path(tree::AstTree, path::Vector{Int}, replacement::AstNode)
    isempty(path) && return AstTree(replacement)
    parent = tree.root
    for edge in path[1:end-1]
        edge in eachindex(children(parent)) ||
            return _region_error(:invalid_path, path)
        parent = children(parent)[edge]
    end
    last_edge = path[end]
    last_edge in eachindex(children(parent)) ||
        return _region_error(:invalid_path, path)
    setchild!(parent, last_edge, replacement)
    return tree
end

"""
    materialize_scalar_region(region, expr) -> Program | Err

Replace only the selected occurrence in an owned source copy.  Scalar
constants receive fresh local names disjoint from the complete source and
context namespaces.  An exact seed round trip instead preserves the parent's
original names and sharing.  A changed scalar tree represents constants per
occurrence: it can split an originally shared parameter and does not promise
to preserve parameter-sharing structure. Equal values are not merged.
The resulting tree is checked again by
`TypedSearch.prepare_source`; source checking also bounds the materialised
tree and retains untouched `FunctionDefNode`/CSE sibling structure.
"""
function materialize_scalar_region(region::ScalarRegion, expr)
    expr isa ScalarExpr || return _region_error(:invalid_expression, typeof(expr))
    ScalarSearch._required_features(expr) <= length(region.dependencies) ||
        return _region_error(:missing_dependency, length(region.dependencies))

    # The initial scalar seed is already a complete, source-checked view of
    # the parent.  Preserve its original constant occurrences and dictionary
    # namespace on an exact round trip; otherwise the ordinary materializer
    # would manufacture fresh names and make an unchanged seed look new to an
    # outer age/reproduction layer.  This copies the region-owned value, so no
    # caller-owned tree or dictionary is aliased.
    if ScalarSearch._same_scalar_expression(region.expr, expr)
        length(region.program.tree) <= region.max_nodes || return _region_error(
            :materialization_size, length(region.program.tree))
        checked = prepare_source(region.program.tree, region.type_context,
            region.target_ty;
            max_nodes=region.max_nodes, max_depth=region.max_depth)
        checked isa Err && return _region_error(:source_check, checked.value)
        return _publish_program(
            checked.index.tree,
            copy(region.program.local_consts),
            Dict{Symbol,TyLike}(region.program.local_consts_ty),
        )
    end

    # The published source remains read-only; scalar edits only extend this
    # fresh map with newly named blocks, so borrowed values and aliases stay
    # intact without re-copying opaque host resources.
    local_consts = copy(region.program.local_consts)
    local_consts_ty = Dict{Symbol,TyLike}(region.program.local_consts_ty)
    reserved = _collect_namespace!(Set{Symbol}(), region.source.index.tree.root)
    union!(reserved, keys(region.type_context))
    union!(reserved, keys(local_consts))
    union!(reserved, keys(local_consts_ty))
    replacement = scalar_to_ast(expr, region.dependencies;
        materialize_constant=value -> _fresh_region_constant(
            Float64(value), reserved, local_consts, local_consts_ty),
    )
    replacement isa Err && return _region_error(:materialization, replacement.value)

    tree = deepcopy(region.source.index.tree)
    replaced = _replace_path(tree, region.path, replacement)
    replaced isa Err && return replaced
    length(replaced) <= region.max_nodes || return _region_error(
        :materialization_size,
        (actual=length(replaced), maximum=region.max_nodes),
    )

    context = Dict{Symbol,TyLike}(region.type_context)
    merge!(context, local_consts_ty)
    checked = prepare_source(replaced, context, region.target_ty;
        max_nodes=region.max_nodes, max_depth=region.max_depth)
    checked isa Err && return _region_error(:source_check, checked.value)
    return _publish_program(checked.index.tree, local_consts, local_consts_ty)
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

"""
    search_scalar_region(region, score; config, rng, max_return=4,
        stop_loss=nothing, search_backend=search_scalar)

Run one bounded scalar-search session seeded by the ordinary parent region.
Every structural or constant-refinement objective entrance attempts to
materialise a fresh whole `Program`. Successfully source-checked candidates
invoke `score(program)`, so changing folds, recursion, and other enclosing
semantics remain part of the objective. Materialisation refusals remain charged
objective entrances but do not call `score`; `diagnostics.objective_calls`
counts these entrances, not just successful calls through to the scorer. The
backend keyword follows the `search_scalar(X, y; config, rng,
initial_population, objective, stop_loss)` contract and lets a caller
coordinator provide its persistent multi-island implementation later.

The return value is `(candidates, diagnostics)`; each returned candidate has
`program`, `expr`, `loss`, and `complexity`.  Final archive members are
source-checked once more, and those boundary checks are reported separately
from structural and constant-refinement calls.
"""
function search_scalar_region(
    region::ScalarRegion,
    score;
    config::ScalarSearchConfig=ScalarSearchConfig(evolution_strategy=:regularized),
    rng::AbstractRNG=default_rng(),
    max_return::Int=4,
    stop_loss=nothing,
    search_backend=search_scalar,
)
    max_return > 0 || throw(ArgumentError("max_return must be positive"))
    feature_count = length(region.dependencies)
    feature_count > 0 || return _region_error(:zero_features, nothing)
    # ScalarSearch's ordinary matrix API is retained as a backend contract.
    # These placeholders are never read by contextual_objective or score.
    X = zeros(Float64, feature_count, 1)
    y = zeros(Float64, 1)
    objective_calls = Ref(0)
    function contextual_objective(expression::ScalarExpr)
        objective_calls[] += 1
        candidate = materialize_scalar_region(region, expression)
        candidate isa Err && return candidate
        value = score(candidate)
        value isa Err && return value
        value isa Real || throw(ArgumentError(
            "scalar-region score must return Real or Err",
        ))
        return value
    end
    initial = [deepcopy(region.expr)]
    result = search_backend(X, y;
        config,
        rng,
        initial_population=initial,
        objective=contextual_objective,
        stop_loss,
    )
    result isa Err && return result
    entries = _search_entries(result, max_return)
    candidates = NamedTuple[]
    boundary_rechecks = 0
    boundary_failures = 0
    for entry in entries
        boundary_rechecks += 1
        candidate = materialize_scalar_region(region, entry.expr)
        if candidate isa Err
            boundary_failures += 1
            continue
        end
        push!(candidates, (
            program=candidate,
            expr=entry.expr,
            loss=entry.loss,
            complexity=entry.complexity,
        ))
        length(candidates) >= max_return && break
    end
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
        objective_calls=objective_calls[],
        boundary_rechecks,
        boundary_failures,
        returned_candidates=length(candidates),
        backend_result=result,
    )
    return (; candidates, diagnostics)
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

@testitem "scalar region preserves recursive outer structure" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression:
        ScalarRegion, materialize_scalar_region, prepare_scalar_region
    using LambdaRegression.ScalarSearch
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike

    target = TyFun(FloatTy, FloatTy)
    recursive = AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0))))
    parent = Program(recursive)
    region = prepare_scalar_region(parent, target, Dict{Symbol,TyLike}(), [1, 1])
    @test region isa ScalarRegion
    @test region.dependencies == [IndexNode(0)]
    candidate = materialize_scalar_region(region, ScalarSearch.scalar_input(1))
    @test candidate isa Program
    @test candidate.tree.root isa FunctionDefNode
    @test candidate.tree.root.body.body == IndexNode(0)
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

@testitem "scalar region refuses unsupported and zero-feature seeds" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Programs: Program
    using LambdaRegression.Regression: ScalarRegion, prepare_scalar_region
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
    zero_region = prepare_scalar_region(constant, FloatTy, Dict{Symbol,TyLike}(), Int[])
    @test zero_region isa Err
    @test zero_region.value.reason == :zero_features

    # A constant seed remains eligible when a visible Float binder can supply
    # a later mutation; only a region with no available dependency is refused.
    reachable = Program(AstTree(AbsNode(:x, ConstNode(:c)));
        local_consts=Dict{Symbol,Any}(:c => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c => FloatTy))
    reachable_region = prepare_scalar_region(
        reachable, target, Dict{Symbol,TyLike}(), [1])
    @test reachable_region isa ScalarRegion
    @test reachable_region.dependencies == [IndexNode(0)]
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
