"""
    search_typed_roots, search_typed_hole

Small production bridge from complete `Individual` trees to the bounded
`TypedSearch` island.  Source preparation and projection remain the only
translation boundary: every objective call exports a whole tree, checks it,
and then calls the supplied task score.  The returned candidates are owned
`Individual` records; the compact terms and backend result are retained for
inspection.

This first slice deliberately declines Let-App/CSE, `FunctionDefNode`, and
other unsupported source regions at the shared-root boundary rather than
β-reducing them into a different program.  A local hole may retain such an
outer complete-AST construct when the selected region itself is supported.
Injected score callbacks and type/value contexts are required to
remain immutable for the lifetime of the call.  A changed task or callback
world requires a new search invocation; copying a tree or dictionary cannot
freeze arbitrary captured resources in a closure.
"""

using Random: AbstractRNG, default_rng

using ..Utils: Err
using ..TypeSystem
using ..TypeSystem: Ty, TyLike
using ..ASTs.NodeASTs:
    AstNode, AstTree, AbsNode, AppNode, ConstNode, IndexNode, LetNode, VarNode,
    children
using ..Procedural: FunctionDefNode, ScopeNode
using ..Mutations.Populations: Individual
using ..TypedSearch:
    Language, TypeTable, TypedTerm, TypedProposalPolicy, ScoredTerm,
    prepare_source, source_hole, prepare_root_hole, import_root_body,
    import_hole, export_hole, validate, search_nsga2,
    TypedSearchError, SourceConstant, intern_type!, add_leaf!,
    register_external_bindings!

export TypedRefinementError, TypedRefinementResult
export search_typed_roots, search_typed_hole

"""An expected refusal at the complete-AST/typed-island boundary."""
struct TypedRefinementError
    reason::Symbol
    detail::Any
end

TypedRefinementError(reason::Symbol) = TypedRefinementError(reason, nothing)

"""Typed bridge result: owned candidates, counters, and the raw search result."""
struct TypedRefinementResult
    candidates::Vector{Any}
    diagnostics::NamedTuple
    raw_result::Any
end

mutable struct _TypedRefinementCounters
    ancestors_seen::Int
    ancestors_admitted::Int
    boundary_rejections::Int
    export_failures::Int
    final_source_failures::Int
    size_rejections::Int
    depth_rejections::Int
    objective_calls::Int
    score_calls::Int
    score_invalid::Int
    final_rechecks::Int
    final_failures::Int
    seed_restores::Int
end

_TypedRefinementCounters() =
    _TypedRefinementCounters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function _typed_refinement_diagnostics(c::_TypedRefinementCounters, raw,
                                       returned::Int; search_invalid=0)
    return (; ancestors_seen=c.ancestors_seen,
        ancestors_admitted=c.ancestors_admitted,
        boundary_rejections=c.boundary_rejections,
        export_failures=c.export_failures,
        final_source_failures=c.final_source_failures,
        size_rejections=c.size_rejections,
        depth_rejections=c.depth_rejections,
        objective_calls=c.objective_calls,
        score_calls=c.score_calls,
        score_invalid=c.score_invalid,
        final_rechecks=c.final_rechecks,
        final_failures=c.final_failures,
        seed_restores=c.seed_restores,
        search_invalid_evaluations=search_invalid,
        returned_candidates=returned,
        backend_result=raw)
end

@inline _typed_refinement_fail(reason::Symbol, detail=nothing) =
    Err(TypedRefinementError(reason, detail))

function _typed_refinement_ctx(value, label::Symbol)
    value isa AbstractDict || throw(ArgumentError("$label must map Symbols to TyLike values"))
    result = Dict{Symbol,TyLike}()
    for (name, ty) in value
        name isa Symbol || throw(ArgumentError("$label keys must be Symbols"))
        ty isa TyLike || throw(ArgumentError("$label values must be TyLike values"))
        result[name] = ty
    end
    TypeSystem.validate_ctx(result)
    return result
end

function _typed_refinement_node_depth(node::AstNode)::Int
    isempty(children(node)) && return 0
    return 1 + maximum(_typed_refinement_node_depth, children(node))
end

_typed_refinement_node_depth(tree::AstTree) = _typed_refinement_node_depth(tree.root)

function _typed_refinement_const_names!(names::Set{Symbol}, node::AstNode)
    node isa ConstNode && push!(names, node.name)
    for child in children(node)
        _typed_refinement_const_names!(names, child)
    end
    return names
end

function _typed_refinement_unsupported(node::AstNode)
    # CSE and Let-App both introduce an application whose function is an
    # abstraction.  It is important to reject that shape before projection,
    # rather than silently β-reducing a complete ancestor.
    node isa FunctionDefNode && return :function_def
    node isa ScopeNode && return :scope
    node isa LetNode && return :let
    node isa AppNode && node.func isa AbsNode && return :let_app_cse
    return nothing
end

function _typed_refinement_check_shape(node::AstNode)
    reason = _typed_refinement_unsupported(node)
    reason === nothing || return reason
    for child in children(node)
        child_reason = _typed_refinement_check_shape(child)
        child_reason === nothing || return child_reason
    end
    return nothing
end

function _typed_refinement_owned_ancestor(ind::Individual, index::Int,
                                         base_ctx::Dict{Symbol,TyLike},
                                         occupied::Set{Symbol};
                                         check_shape::Bool=true)
    ind isa Individual || return _typed_refinement_fail(:not_individual, index)
    ind.target_ty isa Ty || return _typed_refinement_fail(:invalid_target, index)
    Set(keys(ind.local_consts)) == Set(keys(ind.local_consts_ty)) ||
        return _typed_refinement_fail(:local_const_keys, index)
    isempty(intersect(Set(keys(ind.local_consts)), Set(keys(base_ctx)))) ||
        return _typed_refinement_fail(:local_context_collision, index)
    if check_shape
        shape_reason = _typed_refinement_check_shape(ind.tree.root)
        shape_reason === nothing || return _typed_refinement_fail(shape_reason, index)
    end

    renames = Dict{Symbol,Symbol}()
    local_consts = Dict{Symbol,Any}()
    local_consts_ty = Dict{Symbol,TyLike}()
    names = sort!(collect(keys(ind.local_consts)); by=String)
    for (serial, name) in enumerate(names)
        candidate = Symbol("__typed_bridge_local_", index, "_", serial)
        suffix = 1
        while candidate in occupied || candidate in keys(base_ctx)
            candidate = Symbol("__typed_bridge_local_", index, "_", serial, "_", suffix)
            suffix += 1
        end
        push!(occupied, candidate)
        renames[name] = candidate
        local_consts[candidate] = deepcopy(ind.local_consts[name])
        local_consts_ty[candidate] = deepcopy(ind.local_consts_ty[name])
    end
    tree = deepcopy(ind.tree)
    isempty(renames) || (tree.root = Mutations._rename_const_nodes(tree.root, renames))
    # Reject unresolved named leaves here so custom contexts cannot defer a
    # source mistake until after a typed objective has been charged.
    names_in_tree = _typed_refinement_const_names!(Set{Symbol}(), tree.root)
    all(name -> haskey(base_ctx, name) || haskey(local_consts, name), names_in_tree) ||
        return _typed_refinement_fail(:unresolved_constant, index)
    return (; individual=ind, tree, target_ty=ind.target_ty, local_consts,
        local_consts_ty, context=merge(copy(base_ctx), local_consts_ty), index)
end

function _typed_refinement_input_context(type_context, input_types)
    base = _typed_refinement_ctx(type_context, :type_context)
    inputs = _typed_refinement_ctx(input_types, :input_types)
    isempty(intersect(Set(keys(base)), Set(keys(inputs)))) ||
        throw(ArgumentError("type_context and input_types must use disjoint names"))
    merge!(base, inputs)
    return base, inputs
end

function _typed_refinement_options(seed_count::Int, max_nodes::Int,
                                   max_depth::Int, search_options)
    max_nodes > 0 && max_depth >= 0 ||
        throw(ArgumentError("max_nodes must be positive and max_depth non-negative"))
    supplied = (; search_options...)
    for key in (:classifier, :evaluation_state)
        hasproperty(supplied, key) &&
            throw(ArgumentError("$key is fixed by the typed refinement bridge"))
    end
    defaults = (population_size=max(1, min(8, seed_count)), attempts=8,
        evaluations=max(16, seed_count), max_nodes, max_depth)
    return merge(defaults, supplied)
end

function _typed_refinement_register_globals!(language::Language,
                                             context::Dict{Symbol,TyLike})
    for name in sort!(collect(keys(context)); by=String)
        ty = context[name]
        ty isa Ty || continue # Schemes and unresolved/μ types are not leaves.
        typeid = intern_type!(language.types, ty)
        typeid isa Err && continue
        existing = nothing
        for (index, leaf) in enumerate(language.leaves)
            leaf.payload isa SourceConstant || continue
            leaf.payload.name == name || continue
            existing = index
            leaf.typeid == typeid || return _typed_refinement_fail(
                :global_constant_type_conflict, name)
            break
        end
        existing === nothing && add_leaf!(language, typeid,
            SourceConstant(name))
    end
    return nothing
end

function _typed_refinement_score_value(value, counters::_TypedRefinementCounters)
    if value isa Err
        counters.score_invalid += 1
        return value
    end
    if !(value isa Real)
        counters.score_invalid += 1
        return _typed_refinement_fail(:invalid_score, typeof(value))
    end
    converted = try
        Float64(value)
    catch
        counters.score_invalid += 1
        return _typed_refinement_fail(:invalid_score, typeof(value))
    end
    if !isfinite(converted)
        counters.score_invalid += 1
        return _typed_refinement_fail(:invalid_score, value)
    end
    return converted
end

function _typed_refinement_candidate(term::TypedTerm, language::Language,
                                     hole, target_ty::Ty, context,
                                     local_consts, local_consts_ty,
                                     max_nodes::Int, max_depth::Int,
                                     counters::_TypedRefinementCounters;
                                     count_export::Bool=true)
    if count_export
        counters.objective_calls += 1
    end
    exported = export_hole(term, language, hole;
        max_nodes=max_nodes, max_depth=max_depth)
    if exported isa Err
        count_export && (counters.export_failures += 1)
        return exported
    end
    length(exported) <= max_nodes || begin
        counters.size_rejections += 1
        return _typed_refinement_fail(:whole_tree_size, length(exported))
    end
    _typed_refinement_node_depth(exported) <= max_depth || begin
        counters.depth_rejections += 1
        return _typed_refinement_fail(:whole_tree_depth,
            _typed_refinement_node_depth(exported))
    end
    checked = prepare_source(exported, context, target_ty;
        max_nodes=max_nodes, max_depth=max_depth)
    if checked isa Err
        count_export && (counters.final_source_failures += 1)
        return checked
    end
    return Individual(checked.tree, target_ty;
        local_consts=deepcopy(local_consts),
        local_consts_ty=deepcopy(local_consts_ty))
end

function _typed_refinement_materializer(term, language, hole, target_ty, context,
                                        local_consts, local_consts_ty, options,
                                        counters; count_export=true)
    return _typed_refinement_candidate(term, language, hole, target_ty, context,
        local_consts, local_consts_ty, Int(options.max_nodes), Int(options.max_depth),
        counters; count_export)
end

@inline function _typed_refinement_term_signature(term::TypedTerm)
    return Tuple((node.kind, node.typeid, node.x, node.y) for node in term.nodes)
end

function _typed_refinement_original_copy(ind::Individual)
    return Individual(
        deepcopy(ind.tree),
        ind.target_ty;
        local_consts=deepcopy(ind.local_consts),
        local_consts_ty=deepcopy(ind.local_consts_ty),
        fitness=ind.fitness,
        meta=deepcopy(ind.meta),
    )
end

function _typed_refinement_result(raw, language, hole, target_ty, context,
                                  local_consts, local_consts_ty, options,
                                  counters, max_return::Int,
                                  seed_terms::Vector{TypedTerm},
                                  seed_originals::Vector{Individual})
    raw isa Err && return raw
    entries = ScoredTerm[]
    hasproperty(raw, :frontier) && append!(entries, raw.frontier)
    hasproperty(raw, :populations) &&
        foreach(pop -> append!(entries, pop), raw.populations)
    candidates = Any[]
    seen = Set{Any}()
    seed_signatures = Set{Any}(_typed_refinement_term_signature(term)
        for term in seed_terms)
    for entry in entries
        length(candidates) >= max_return && break
        sig = _typed_refinement_term_signature(entry.term)
        sig in seen && continue
        candidate = if sig in seed_signatures
            # Search copies an initial term into its backend population.  A
            # syntax-identical result is therefore the unchanged seed, not a
            # fresh materialisation: return a detached copy of the original
            # ancestor so local constant names/values stay exactly as supplied.
            seed_index = findfirst(index ->
                _typed_refinement_term_signature(seed_terms[index]) == sig,
                eachindex(seed_terms))
            _typed_refinement_original_copy(seed_originals[seed_index])
        else
            _typed_refinement_materializer(entry.term, language, hole,
                target_ty, context, local_consts, local_consts_ty, options, counters;
                count_export=false)
        end
        if sig in seed_signatures
            counters.seed_restores += 1
        else
            counters.final_rechecks += 1
        end
        candidate isa Err && (counters.final_failures += 1; continue)
        push!(seen, sig)
        # Report the materialised complete-AST size.  Compact typed-term
        # complexity omits fixed binders/siblings and must not be presented as
        # whole-program complexity at this bridge.
        complexity = length(candidate.tree)
        push!(candidates, (; individual=candidate, term=deepcopy(entry.term),
            loss=Float64(entry.loss), complexity=Int(complexity),
            birth=hasproperty(entry, :birth) ? entry.birth : 0))
    end
    stats = hasproperty(raw, :stats) ? raw.stats : (;)
    invalid = hasproperty(stats, :invalid_evaluations) ? stats.invalid_evaluations : 0
    diagnostics = _typed_refinement_diagnostics(counters, raw, length(candidates);
        search_invalid=invalid)
    return TypedRefinementResult(candidates, diagnostics, raw)
end

function _typed_refinement_run(seeds::Vector{TypedTerm}, language::Language,
                               hole, target_ty::Ty, context, local_consts,
                               local_consts_ty, score, rng::AbstractRNG,
                               group_count::Int, max_return::Int, options,
                               counters::_TypedRefinementCounters,
                               proposal_policy, diverse_initialization::Bool,
                               seed_originals::Vector{Individual})
    objective = function(term::TypedTerm)
        candidate = _typed_refinement_materializer(term, language, hole,
            target_ty, context, local_consts, local_consts_ty, options, counters)
        candidate isa Err && return candidate
        counters.score_calls += 1
        # Deliberately do not catch callback exceptions: they are programming or
        # task errors, while returned Err is the expected invalid-candidate path.
        applicable(score, candidate) || throw(ArgumentError(
            "typed refinement score must accept an Individual"))
        return _typed_refinement_score_value(score(candidate), counters)
    end
    raw = search_nsga2(seeds, language, objective;
        rng, group_count,
        proposal_policy,
        diverse_initialization,
        options...)
    return _typed_refinement_result(raw, language, hole, target_ty, context,
        local_consts, local_consts_ty, options, counters, max_return, seeds,
        seed_originals)
end

"""
    search_typed_roots(individuals, type_context, score; arity,
                       input_types=Dict(), rng=default_rng(), max_return=4,
                       max_nodes=128, max_depth=32, group_count=1,
                       search_options...)

Build one shared root interface from compatible complete ancestors.  Local
constant names are deterministically made unique per ancestor, and ordinary
boundary refusals are retained in diagnostics.  `score` receives a complete,
owned `Individual` and returns a finite `Real` or `Err`.
"""
function search_typed_roots(individuals, type_context, score;
                            arity::Int,
                            input_types=Dict(),
                            rng::AbstractRNG=default_rng(),
                            max_return::Int=4,
                            max_nodes::Int=128,
                            max_depth::Int=32,
                            group_count::Int=1,
                            proposal_policy=TypedProposalPolicy(),
                            diverse_initialization::Bool=true,
                            search_options...)
    arity >= 0 || throw(ArgumentError("arity must be non-negative"))
    max_return > 0 || throw(ArgumentError("max_return must be positive"))
    group_count > 0 || throw(ArgumentError("group_count must be positive"))
    isempty(individuals) && return _typed_refinement_fail(:empty_ancestors)
    base, inputs = _typed_refinement_input_context(type_context, input_types)
    counters = _TypedRefinementCounters()
    counters.ancestors_seen = length(individuals)
    occupied = union(Set(keys(base)), Set(keys(inputs)))
    owned = Any[]
    target = nothing
    for (index, ancestor) in enumerate(individuals)
        prepared = _typed_refinement_owned_ancestor(ancestor, index, base, occupied)
        if prepared isa Err
            counters.boundary_rejections += 1
            continue
        end
        target === nothing && (target = prepared.target_ty)
        TypeSystem.iscompatible(prepared.target_ty, target) || begin
            counters.boundary_rejections += 1
            continue
        end
        push!(owned, prepared)
    end
    target === nothing && return _typed_refinement_fail(:no_common_root, counters)
    options = _typed_refinement_options(length(owned), max_nodes, max_depth,
        search_options)
    language = Language(TypeTable())
    globals_registered = _typed_refinement_register_globals!(language, base)
    globals_registered isa Err && return globals_registered
    hole = nothing
    seeds = TypedTerm[]
    seed_originals = Individual[]
    all_locals = Dict{Symbol,Any}()
    all_local_types = Dict{Symbol,TyLike}()
    for prepared in owned
        merge!(all_locals, prepared.local_consts)
        merge!(all_local_types, prepared.local_consts_ty)
        if hole === nothing
            source = prepare_source(prepared.tree, prepared.context, prepared.target_ty;
                max_nodes, max_depth)
            if source isa Err
                counters.boundary_rejections += 1
                continue
            end
            made = prepare_root_hole(source, language, prepared.context; arity)
            if made isa Err
                counters.boundary_rejections += 1
                continue
            end
            hole = made.hole
            push!(seeds, made.term)
            push!(seed_originals, _typed_refinement_original_copy(prepared.individual))
        else
            source = prepare_source(prepared.tree, prepared.context, prepared.target_ty;
                max_nodes, max_depth)
            if source isa Err
                counters.boundary_rejections += 1
                continue
            end
            imported = import_root_body(source, language, hole)
            if imported isa Err
                counters.boundary_rejections += 1
                continue
            end
            push!(seeds, imported)
            push!(seed_originals, _typed_refinement_original_copy(prepared.individual))
        end
        counters.ancestors_admitted += 1
    end
    isempty(seeds) && return _typed_refinement_fail(:no_importable_root, counters)
    context = merge(base, all_local_types)
    return _typed_refinement_run(seeds, language, hole, target, context,
        all_locals, all_local_types, score, rng, group_count, max_return, options,
        counters, proposal_policy, diverse_initialization, seed_originals)
end

"""
    search_typed_hole(individual, path, type_context, score; input_types=Dict(), ...)

Search one ordinary occurrence of one fixed ancestor.  The source anchor and
all enclosing bindings remain intact through every `export_hole` objective.
"""
function search_typed_hole(individual::Individual, path, type_context, score;
                           input_types=Dict(),
                           rng::AbstractRNG=default_rng(),
                           max_return::Int=4,
                           max_nodes::Int=128,
                           max_depth::Int=32,
                           group_count::Int=1,
                           proposal_policy=TypedProposalPolicy(),
                           diverse_initialization::Bool=true,
                           search_options...)
    max_return > 0 || throw(ArgumentError("max_return must be positive"))
    group_count > 0 || throw(ArgumentError("group_count must be positive"))
    base, _ = _typed_refinement_input_context(type_context, input_types)
    counters = _TypedRefinementCounters(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    occupied = Set(keys(base))
    prepared = _typed_refinement_owned_ancestor(individual, 1, base, occupied;
        check_shape=false)
    prepared isa Err && return prepared
    source = prepare_source(prepared.tree, prepared.context, prepared.target_ty;
        max_nodes, max_depth)
    source isa Err && return _typed_refinement_fail(:source, source)
    hole = source_hole(source, Int[path...])
    hole isa Err && return _typed_refinement_fail(:hole, hole)
    language = Language(TypeTable())
    globals_registered = _typed_refinement_register_globals!(language, base)
    globals_registered isa Err && return globals_registered
    seed = import_hole(hole, language)
    seed isa Err && return _typed_refinement_fail(:projection, seed)
    register_external_bindings!(hole, language)
    counters.ancestors_admitted = 1
    seed_originals = Individual[_typed_refinement_original_copy(individual)]
    options = _typed_refinement_options(1, max_nodes, max_depth, search_options)
    return _typed_refinement_run(TypedTerm[seed], language, hole,
        prepared.target_ty, prepared.context, prepared.local_consts,
        prepared.local_consts_ty, score, rng, group_count, max_return, options,
        counters, proposal_policy, diverse_initialization, seed_originals)
end

@testitem "typed refinement keeps colliding local constants distinct" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_roots, TypedRefinementResult
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    first = Individual(AstTree(AbsNode(:x, ConstNode(:c1))), TyFun(FloatTy, FloatTy);
        local_consts=Dict{Symbol,Any}(:c1 => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c1 => FloatTy))
    second = Individual(AstTree(AbsNode(:x, ConstNode(:c1))), TyFun(FloatTy, FloatTy);
        local_consts=Dict{Symbol,Any}(:c1 => 2.0),
        local_consts_ty=Dict{Symbol,TyLike}(:c1 => FloatTy))
    result = search_typed_roots([first, second], Dict{Symbol,TyLike}(), _ -> 0.0;
        arity=1, rng=MersenneTwister(1111),
        attempts=0, evaluations=8, population_size=2)
    @test result isa TypedRefinementResult
    @test length(result.candidates) >= 1
    values = Float64[
        c.individual.local_consts[:c1]
        for c in result.candidates
        if haskey(c.individual.local_consts, :c1)
    ]
    @test 1.0 in values
    @test 2.0 in values
    @test result.diagnostics.seed_restores >= 2
end

@testitem "typed refinement local hole preserves captured binder" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_hole, TypedRefinementResult
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    tree = AstTree(AbsNode(:x, AbsNode(:y, IndexNode(1))))
    ind = Individual(tree, TyFun(FloatTy, TyFun(FloatTy, FloatTy)))
    result = search_typed_hole(ind, [1], Dict{Symbol,TyLike}(), _ -> 0.0;
        attempts=0, evaluations=4, population_size=1)
    @test result isa TypedRefinementResult
    @test !isempty(result.candidates)
    candidate = first(result.candidates).individual
    @test candidate.tree.root isa AbsNode
    @test candidate.tree.root.body isa AbsNode
    @test candidate.tree.root.body.body isa IndexNode
    @test candidate.tree.root.body.body.index == 1
end

@testitem "typed refinement rejects CSE and separates callback accounting" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_roots, TypedRefinementError
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    cse = Individual(AstTree(AppNode(AbsNode(:x, IndexNode(0)), ConstNode(:one))),
        FloatTy; local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}())
    rejected = search_typed_roots([cse], Dict{Symbol,TyLike}(:one => FloatTy), _ -> 0.0;
        arity=0)
    @test rejected isa Err
    calls = Ref(0)
    seed = Individual(AstTree(ConstNode(:one)), FloatTy)
    result = search_typed_roots([seed], Dict{Symbol,TyLike}(:one => FloatTy), ind ->
        (calls[] += 1; 0.0); arity=0, attempts=0, evaluations=2, population_size=1)
    @test result.diagnostics.score_calls == calls[]
    @test result.diagnostics.objective_calls >= result.diagnostics.score_calls
    @test all(c -> c.individual !== seed, result.candidates)
end

@testitem "typed refinement propagates score callback exceptions" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_hole
    using LambdaRegression.TypeSystem: FloatTy, TyFun
    ind = Individual(AstTree(AbsNode(:x, IndexNode(0))), TyFun(FloatTy, FloatTy))
    @test_throws DomainError search_typed_hole(ind, Int[], Dict(), _ -> throw(DomainError(:test));
        attempts=0, evaluations=1, population_size=1)
end

@testitem "typed local holes retain unsupported complete-AST siblings" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_hole, TypedRefinementResult
    using LambdaRegression.Procedural: FunctionDefNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    cse = Individual(
        AstTree(AppNode(AbsNode(:z, IndexNode(0)), ConstNode(:one))), FloatTy,
    )
    cse_result = search_typed_hole(cse, [1, 1], Dict{Symbol,TyLike}(:one => FloatTy),
        _ -> 0.0; attempts=0, evaluations=1, population_size=1)
    @test cse_result isa TypedRefinementResult
    @test cse_result.candidates[1].individual.tree.root isa AppNode
    @test cse_result.candidates[1].individual.tree.root.func isa AbsNode

    recursive = Individual(
        AstTree(FunctionDefNode(AbsNode(:x, IndexNode(0)))), TyFun(FloatTy, FloatTy),
    )
    recursive_result = search_typed_hole(recursive, [1, 1], Dict{Symbol,TyLike}(),
        _ -> 0.0; attempts=0, evaluations=1, population_size=1)
    @test recursive_result isa TypedRefinementResult
    @test recursive_result.candidates[1].individual.tree.root isa FunctionDefNode
end

@testitem "typed bridge registers real primitive leaves and runs N1" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_roots, TypedRefinementResult
    using LambdaRegression.TypeSystem: FADD, FloatTy, TyFun, TyLike, get_type
    seed = Individual(
        AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:FADD), IndexNode(0)),
            IndexNode(0)))), TyFun(FloatTy, FloatTy),
    )
    seen = Ref(false)
    result = search_typed_roots([seed], Dict{Symbol,TyLike}(:FADD => get_type(FADD)),
        ind -> (seen[] = any(node -> node isa ConstNode && node.name == :FADD,
            ind.tree); 0.0); arity=1, rng=MersenneTwister(2000),
        attempts=1, evaluations=4, population_size=1, max_nodes=32, max_depth=8)
    @test result isa TypedRefinementResult
    @test seen[]
    @test result.diagnostics.objective_calls >= 1
end

@testitem "typed root bridge preserves input positions" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: search_typed_roots, TypedRefinementResult
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike
    target = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    first = Individual(AstTree(AbsNode(:a, AbsNode(:b, IndexNode(1)))), target)
    second = Individual(AstTree(AbsNode(:u, AbsNode(:v, IndexNode(0)))), target)
    result = search_typed_roots([first, second], Dict{Symbol,TyLike}(), _ -> 0.0;
        arity=2, attempts=0, evaluations=4, population_size=2)
    @test result isa TypedRefinementResult
    indices = [candidate.individual.tree.root.body.body.index
               for candidate in result.candidates
               if candidate.individual.tree.root isa AbsNode &&
                  candidate.individual.tree.root.body isa AbsNode &&
                  candidate.individual.tree.root.body.body isa IndexNode]
    @test 0 in indices
    @test 1 in indices
end
