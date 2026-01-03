struct _InvalidBindingHostWeightErr
    value
end

function _freshen_scoped_binders_node(
    term::_ScopedTerm,
    remapped::Dict{Int,Int},
    next_id::Base.RefValue{Int},
)
    if term isa _ScopedConst || term isa _ScopedFree
        return term
    elseif term isa _ScopedBound
        return _ScopedBound(get(remapped, term.binder_id, term.binder_id))
    elseif term isa _ScopedAbs
        fresh_id = next_id[]
        next_id[] += 1
        remapped[term.binder_id] = fresh_id
        body = _freshen_scoped_binders_node(term.body, remapped, next_id)
        delete!(remapped, term.binder_id)
        return _ScopedAbs(fresh_id, term.name, body)
    elseif term isa _ScopedFunctionDef
        fresh_id = next_id[]
        next_id[] += 1
        remapped[term.binder_id] = fresh_id
        body = _freshen_scoped_binders_node(term.body, remapped, next_id)
        delete!(remapped, term.binder_id)
        return _ScopedFunctionDef(fresh_id, body)
    elseif term isa _ScopedApp
        return _ScopedApp(
            _freshen_scoped_binders_node(term.func, remapped, next_id),
            _freshen_scoped_binders_node(term.arg, remapped, next_id),
        )
    end
    error("unsupported scoped term: $(typeof(term))")
end

"""
    _freshen_scoped_binders(term, next_binder_id)

Copy `term`, assigning consecutive fresh identities to every lambda and
FunctionDef self binder and retargeting only references bound inside the copied
term. External captures retain their original identities. Returns
`(copy, next_unused_id)`.
"""
function _freshen_scoped_binders(term::_ScopedTerm, next_binder_id::Int)
    next_id = Ref(next_binder_id)
    freshened = _freshen_scoped_binders_node(term, Dict{Int,Int}(), next_id)
    return freshened, next_id[]
end

function _scoped_external_captures(term::_ScopedTerm)
    binders = Set{Int}()
    references = Set{Int}()
    function collect_ids(node::_ScopedTerm)
        if node isa _ScopedBound
            push!(references, node.binder_id)
        elseif node isa Union{_ScopedAbs,_ScopedFunctionDef}
            push!(binders, node.binder_id)
            collect_ids(node.body)
        elseif node isa _ScopedApp
            collect_ids(node.func)
            collect_ids(node.arg)
        end
        return nothing
    end
    collect_ids(term)
    setdiff!(references, binders)
    return references
end

_unused_binding_host_preserves_role(
    parent::_ScopedTerm,
    child_index::Int,
    host::_ScopedTerm,
) = !(child_index == 1 && parent isa _ScopedApp && host isa _ScopedAbs)

"""
    _source_binding_host_depths(scoped, source_path)

Return the depths of finite legal ancestor hosts at which `source_path` can be
retained as the body and copied into an unused let RHS. Prefix depths keep the
candidate representation linear; only the selected path is materialized. The
root-to-source path is traversed once to enforce external-capture scope across
lambda and FunctionDef binders and direct-`Abs` application roles.
"""
function _source_binding_host_depths(
    scoped::_ScopedTree,
    source_path::AbstractVector{<:Integer},
)
    path = collect(Int, source_path)
    nodes = _ScopedTerm[scoped.root]
    binder_entry_depth = Dict{Int,Int}()
    node = scoped.root
    for (offset, child_index) in pairs(path)
        if node isa Union{_ScopedAbs,_ScopedFunctionDef} && child_index == 1
            binder_entry_depth[node.binder_id] = offset
            node = node.body
        elseif node isa _ScopedApp && child_index == 1
            node = node.func
        elseif node isa _ScopedApp && child_index == 2
            node = node.arg
        else
            return Err(_ScopedPathErr(path, offset))
        end
        push!(nodes, node)
    end

    minimum_depth = 0
    for binder_id in _scoped_external_captures(node)
        entry_depth = get(binder_entry_depth, binder_id, 0)
        entry_depth == 0 && return Err(_EscapedScopedBinderErr(binder_id))
        minimum_depth = max(minimum_depth, entry_depth)
    end

    host_depths = Int[]
    for depth in minimum_depth:length(path)
        host = nodes[depth + 1]
        if depth > 0 && !_unused_binding_host_preserves_role(
            nodes[depth],
            path[depth],
            host,
        )
            continue
        end
        push!(host_depths, depth)
    end
    return host_depths
end

function _changed_retained_application_path(
    old::_ScopedTree,
    new::_ScopedTree,
    retained,
)
    for mapping in retained
        _application_roles_stable(old, new, [mapping]) || return mapping.first
    end
    return Int[]
end

"""
    _insert_unused_scoped_binding(scoped, host_path, rhs, binder_id, name)

Wrap the unchanged host as `(λname. host) rhs`, preserving every retained
ancestor application's HM role. Returns a rewritten `_ScopedTree` or `Err`.
"""
function _insert_unused_scoped_binding(
    scoped::_ScopedTree,
    host_path::AbstractVector{<:Integer},
    rhs::_ScopedTerm,
    binder_id::Int,
    name::Symbol,
)
    host = _scoped_at_path(scoped, host_path)
    host isa Err && return host
    retained = _retained_application_ancestors(scoped, host_path)
    retained isa Err && return retained
    wrapped = _ScopedApp(_ScopedAbs(binder_id, name, host), rhs)
    rewritten = _replace_scoped(scoped, host_path, wrapped)
    rewritten isa Err && return rewritten
    _application_roles_stable(scoped, rewritten, retained) && return rewritten
    changed_path = _changed_retained_application_path(scoped, rewritten, retained)
    return Err(_RetainedApplicationRoleErr(changed_path))
end

_uniform_binding_host_weight(::Site, ::Vector{Int}, ::Int, ::MutationEnv, ::NamedTuple) = 1.0

function _exact_weight(weight::Real)
    try
        return Rational{BigInt}(weight)
    catch error
        error isa Union{ArgumentError,MethodError,InexactError} || rethrow()
        return Rational{BigInt}(BigFloat(weight))
    end
end

function _integer_weight_units(weights::Vector{Real})
    exact = Rational{BigInt}[_exact_weight(weight) for weight in weights]
    common_denominator = foldl(
        lcm,
        (denominator(weight) for weight in exact);
        init=BigInt(1),
    )
    return BigInt[
        numerator(weight) * div(common_denominator, denominator(weight))
        for weight in exact
    ]
end

function _weighted_ticket_index(units::Vector{BigInt}, ticket::BigInt)
    1 <= ticket <= sum(units) || throw(BoundsError(units, ticket))
    cumulative = BigInt(0)
    for (index, units_at_index) in pairs(units)
        cumulative += units_at_index
        ticket <= cumulative && return index
    end
    throw(BoundsError(units, ticket))
end

function _binding_host_choice(
    rng::AbstractRNG,
    host_depths::Vector{Int},
    weights::Vector{Real},
)
    units = _integer_weight_units(weights)
    ticket = rand(rng, BigInt(1):sum(units))
    return host_depths[_weighted_ticket_index(units, ticket)]
end

"""
    SourceBindingInsertionMutation(; host_weight=_uniform_binding_host_weight)

Insert an intentionally unused let-like binding whose RHS is copied from the
selected existing source site. Legal hosts are finite source-path ancestors
that preserve every external capture and retained HM application role.

`host_weight(source_site, host_path, distance, env, ctx)` may replace uniform
host selection. It must return a finite, nonnegative `Real`; zero excludes a
host. `host_path` is an attempt-local read-only `Vector{Int}` that is reused as
successively deeper prefixes; copy it inside the callback only when retaining
it after the call. The operator uses `ctx.symfactory` when present and otherwise
creates a local `SymbolGenerator`; its suggestion is made fresh against named
variables and abstractions in the attempt tree. Expected failures are an invalid source
path, no positive legal host, an invalid host weight, or scoped materialisation
failure.
"""
struct SourceBindingInsertionMutation{F} <: AbstractMutation
    host_weight::F
end

SourceBindingInsertionMutation(; host_weight=_uniform_binding_host_weight) =
    SourceBindingInsertionMutation(host_weight)

required_site_kinds(::SourceBindingInsertionMutation) = Symbol[:any]

function is_site_eligible(
    ::SourceBindingInsertionMutation,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    return !(_scoped_at_path(scoped, site.path) isa Err)
end

function apply(
    mutation::SourceBindingInsertionMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    source = _scoped_at_path(scoped, site.path)
    source isa Err && return source
    path = collect(Int, site.path)
    host_depths = _source_binding_host_depths(scoped, path)
    host_depths isa Err && return host_depths

    positive_host_depths = Int[]
    positive_weights = Real[]
    host_path = Int[]
    for host_depth in host_depths
        while length(host_path) < host_depth
            push!(host_path, path[length(host_path) + 1])
        end
        distance = length(path) - host_depth
        weight = mutation.host_weight(site, host_path, distance, env, ctx)
        if !(weight isa Real) || !isfinite(weight) || weight < 0
            return Err(_InvalidBindingHostWeightErr(weight))
        end
        iszero(weight) && continue
        push!(positive_host_depths, host_depth)
        push!(positive_weights, weight)
    end
    isempty(positive_host_depths) && return Err(NoEligibleSiteErr())

    host_depth = _binding_host_choice(
        env.rng,
        positive_host_depths,
        positive_weights,
    )
    host_path = path[1:host_depth]
    next_binder_id = scoped.next_binder_id
    copied_source, next_binder_id =
        _freshen_scoped_binders(source, next_binder_id)
    name_factory = haskey(ctx, :symfactory) ? ctx.symfactory : SymbolGenerator()
    name = only(_fresh_binder_names(tree, name_factory, 1))
    rewritten = _insert_unused_scoped_binding(
        scoped,
        host_path,
        copied_source,
        next_binder_id,
        name,
    )
    rewritten isa Err && return rewritten
    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

struct _InvalidBindingRHSTypeWeightErr
    value
end

struct _GeneratedBindingRHSMismatchErr
    inferred::TyLike
    demand::TyLike
end

struct _GeneratedBindingRHSShapeErr
    inferred::Ty
end

_uniform_binding_rhs_type_weight(
    ::TyLike,
    ::Site,
    ::MutationEnv,
    ::NamedTuple,
) = 1.0

function _contains_query_rigid(ty::Ty)
    binders = Set{TypeSystem.TyVar}()
    resolving = Set{TypeSystem.TyVar}()
    function walk(node::Ty)
        node isa TypeSystem.TyRigid && return true
        if node isa TypeSystem.TyVar
            node in binders && return false
            node.instance === nothing && return false
            node in resolving && return false
            push!(resolving, node)
            found = walk(node.instance)
            delete!(resolving, node)
            return found
        elseif node isa TypeSystem.TyMu
            push!(binders, node.var)
            found = walk(node.body)
            delete!(binders, node.var)
            return found
        end
        return any(walk, TypeSystem.type_children(node))
    end
    return walk(ty)
end

const _BINDING_GENERATION_CONTEXT_KEYS = (
    :config,
    :rng,
    :scope_tys,
    :global_consts,
    :global_consts_ty,
    :local_consts,
    :local_consts_ty,
    :source_local_consts,
)

function _validate_binding_generation_kwargs(generation_kwargs::NamedTuple)
    for key in _BINDING_GENERATION_CONTEXT_KEYS
        haskey(generation_kwargs, key) || continue
        throw(ArgumentError("generation_kwargs must not override binding context key :$key"))
    end
    return generation_kwargs
end

function _materialize_binding_rhs_types(rhs_types)
    rhs_types isa Union{Tuple,AbstractArray,AbstractSet} || throw(ArgumentError(
        "rhs_types must be an explicitly materialized finite collection",
    ))
    catalog = Tuple(rhs_types)
    isempty(catalog) && throw(ArgumentError("rhs_types must not be empty"))
    for rhs_ty in catalog
        rhs_ty isa TyLike || throw(ArgumentError(
            "rhs_types entries must be Ty or TyScheme, got $(typeof(rhs_ty))",
        ))
        body = rhs_ty isa TypeSystem.TyScheme ? rhs_ty.ty : rhs_ty
        _contains_query_rigid(body) && throw(ArgumentError(
            "rhs_types entries must not contain query-local TyRigid atoms",
        ))
        if rhs_ty isa Ty
            isempty(TypeSystem.collect_free_vars(rhs_ty)) || throw(ArgumentError(
                "monomorphic rhs_types entries must not contain free TyVar values",
            ))
        else
            scheme = rhs_ty::TypeSystem.TyScheme
            free = TypeSystem.collect_free_vars(scheme.ty)
            all(var -> var in scheme.vars, free) || throw(ArgumentError(
                "rhs_types schemes must quantify every free TyVar in their body",
            ))
        end
    end
    return catalog
end

"""
    GeneratedBindingInsertionMutation(rhs_types; replacement_generator, generation_kwargs, rhs_type_weight)

Insert an intentionally unused direct-`Abs` binding at a selected host, with an
RHS generated for one demand from a caller-owned explicit finite catalog. The
constructor snapshots the catalog as a tuple and rejects empty or nonmaterialized
inputs, open monotypes, query-local rigid atoms, and schemes with unquantified
variables. `generation_kwargs` may configure generation behavior but cannot
replace the caller RNG, lexical scope, constant containers, or a complete
`GenerationConfig` carrying those fields.

Generation uses the selected host's visible binder types, caller RNG, constants,
and finite `generation_kwargs`. Application requires `ctx.global_consts`,
`ctx.global_consts_ty`, `ctx.local_consts`, and `ctx.local_consts_ty`; it uses
`ctx.symfactory` and `ctx.tyfactory` when present. Expected failures include no
positive type weight, invalid weights, generation failure, scoped resolution or
materialization failure, inference failure, and an independently inferred RHS
that does not satisfy its selected catalog demand.
"""
struct GeneratedBindingInsertionMutation{
    R<:Tuple,
    G<:AbstractReplacementGenerator,
    K<:NamedTuple,
    F,
} <: AbstractMutation
    rhs_types::R
    replacement_generator::G
    generation_kwargs::K
    rhs_type_weight::F
end

function GeneratedBindingInsertionMutation(
    rhs_types;
    replacement_generator::AbstractReplacementGenerator=CanonicalReplacementGenerator(),
    generation_kwargs::NamedTuple=(;
        sampling_depth_limit=2,
        sampling_node_budget=24,
        guarded_expansion_depth=0,
    ),
    rhs_type_weight=_uniform_binding_rhs_type_weight,
)
    catalog = _materialize_binding_rhs_types(rhs_types)
    _validate_binding_generation_kwargs(generation_kwargs)
    return GeneratedBindingInsertionMutation(
        catalog,
        replacement_generator,
        generation_kwargs,
        rhs_type_weight,
    )
end

"""
    generate_binding_rhs(generator, rhs_ty, host, env, ctx, generation_kwargs)

Generate one RHS for the selected catalog demand `rhs_ty`. Custom replacement
generators extend this protocol. Built-in canonical and scoped adapters pass the
caller RNG, exact host-visible binder types, constants, and injected finite
generation budgets. Generator errors are returned unchanged.
"""
function generate_binding_rhs(
    generator::CanonicalReplacementGenerator,
    rhs_ty::TyLike,
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    kwargs = _replacement_generation_kwargs(
        generator.include_local_consts,
        host,
        env,
        ctx,
        generation_kwargs,
    )
    return generate_canonical_scoped(rhs_ty; kwargs...)
end

function generate_binding_rhs(
    generator::ScopedReplacementGenerator,
    rhs_ty::TyLike,
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
    generation_kwargs::NamedTuple,
)
    kwargs = _replacement_generation_kwargs(
        generator.include_local_consts,
        host,
        env,
        ctx,
        generation_kwargs,
    )
    return generator.generate_scoped(rhs_ty; kwargs...)
end

function _binding_rhs_type_candidates(
    mutation::GeneratedBindingInsertionMutation,
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    compute = function ()
        candidates = TyLike[]
        weights = Real[]
        for rhs_ty in mutation.rhs_types
            weight = mutation.rhs_type_weight(rhs_ty, host, env, ctx)
            if !(weight isa Real) || !isfinite(weight) || weight < 0
                return Err(_InvalidBindingRHSTypeWeightErr(weight))
            end
            iszero(weight) && continue
            push!(candidates, rhs_ty)
            push!(weights, weight)
        end
        isempty(candidates) && return Err(NoEligibleSiteErr())
        return candidates, weights
    end

    env.cache_eligibility || return compute()
    cache = get!(env.eligibility_cache, _binding_rhs_type_candidates) do
        Dict{Any,Any}()
    end
    key = (mutation.rhs_types, mutation.rhs_type_weight, Tuple(host.path))
    return get!(compute, cache, key)
end

function _binding_rhs_type_choice(
    rng::AbstractRNG,
    candidates::Vector{TyLike},
    weights::Vector{Real},
)
    units = _integer_weight_units(weights)
    ticket = rand(rng, BigInt(1):sum(units))
    return candidates[_weighted_ticket_index(units, ticket)]
end

function _visible_scoped_binders(
    scoped::_ScopedTree,
    site::Site,
)
    index = scoped.index
    index === nothing && return Err(NoEligibleSiteErr())
    occurrence = _scoped_occurrence(scoped, site)
    occurrence == 0 && return Err(NoEligibleSiteErr())
    return reverse(visible_binder_ids(index, occurrence))
end

function _analysis_binding_host(
    host::Site,
    analysis::TypedAnalysis,
    env::MutationEnv,
)
    env.tree === nothing && return Err(NoEligibleSiteErr())
    occurrence = if host.occurrence > 0
        host.occurrence
    else
        current = 1
        for child_index in host.path
            children = analysis.index.children[current]
            1 <= child_index <= length(children) || return Err(NoEligibleSiteErr())
            current = children[child_index]
        end
        current
    end
    1 <= occurrence <= length(analysis) || return Err(NoEligibleSiteErr())
    haskey(analysis.metadata, occurrence) || return Err(NoEligibleSiteErr())

    # The Site is remapped onto the attempt tree before apply, while this
    # analysis belongs to the unchanged source tree. Occurrence IDs are the
    # bridge; the path is checked only to ensure both still describe the same
    # position, never to reconstruct types or binder schemes.
    source_path = path(analysis.index, occurrence)
    source_path == host.path || return Err(NoEligibleSiteErr())

    raw_ty = analysis.metadata[occurrence][1]
    raw_ty isa Ty || return Err(NoEligibleSiteErr())
    binder_types = TyLike[]
    for binder in reverse(visible_binder_ids(analysis.index, occurrence))
        haskey(analysis.binder_schemes, binder) || throw(ArgumentError(
            "typed analysis lacks a scheme for visible binder $binder",
        ))
        push!(binder_types, analysis.binder_schemes[binder])
    end

    # Keep the callback Site owned by the attempt tree. The occurrence/path
    # check above aligns it with source analysis, from which only type facts
    # and complete binder schemes are projected.
    return Site(
        host.path,
        host.node,
        host.parent,
        host.child_index,
        host.depth,
        host.kind,
        TypeSystem.prune(raw_ty),
        binder_types,
        occurrence,
    )
end

function _authoritative_binding_host(
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    analysis = _typed_mutation_analysis(env, ctx)
    analysis isa Err && return analysis
    return _analysis_binding_host(host, analysis, env)
end

@testitem "Generated binding hosts reuse indexed occurrence schemes" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: IndexNode
    using LambdaRegression.Mutations:
        Site, _authoritative_binding_host, _remap_site
    using LambdaRegression.ProgramAnalysis: TypedAnalysis, analyze_program
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, TyFun, TyLike, TyProd, TyScheme

    shared_id_use = IndexNode(0)
    type_ctx = Dict{Symbol,TyLike}(
        :pair => TyFun(IntTy, TyFun(BoolTy, TyProd(IntTy, BoolTy))),
        :one => IntTy,
        :truth => BoolTy,
    )
    tree = AstTree(AppNode(
        AbsNode(:id, AppNode(
            AppNode(ConstNode(:pair), AppNode(shared_id_use, ConstNode(:one))),
            AppNode(shared_id_use, ConstNode(:truth)),
        )),
        AbsNode(:value, IndexNode(0)),
    ))
    target_ty = TyProd(IntTy, BoolTy)
    analysis = analyze_program(tree, type_ctx; target_ty)
    @test analysis isa TypedAnalysis
    analysis isa TypedAnalysis || return

    id_abs = findfirst(
        node -> node isa AbsNode && node.name === :id,
        analysis.index.nodes,
    )
    @test id_abs !== nothing
    id_abs === nothing && return
    id_binder = analysis.index.owner_binder[id_abs]
    id_uses = findall(==(id_binder), analysis.index.reference_binder)
    @test length(id_uses) == 2
    @test id_uses[1] != id_uses[2]

    recorded_scheme = analysis.binder_schemes[id_binder]
    @test recorded_scheme isa TyScheme
    @test length(recorded_scheme.vars) == 1
    attempt_tree = deepcopy(tree)
    env = binding_env(attempt_tree, target_ty, type_ctx; seed=141)
    ctx = (; typed_analysis=analysis,
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    source_site = Site(analysis, first(id_uses), :index)
    warmed_host = _authoritative_binding_host(source_site, env, ctx)
    @test warmed_host isa Site
    warmed_host isa Site || return
    @test warmed_host.node === source_site.node
    @test getfield(warmed_host, :analysis) === nothing

    attempt_site = _remap_site(attempt_tree, source_site, analysis, ctx)
    host = _authoritative_binding_host(attempt_site, env, ctx)

    @test host isa Site
    host isa Site || return
    @test host.occurrence == first(id_uses)
    @test getfield(host, :analysis) === nothing
    @test host.path == [1, 1, 1, 2, 1]
    @test host.node === attempt_site.node
    @test host.parent === attempt_site.parent
    @test host.ty == TyFun(IntTy, IntTy)
    @test length(host.binder_types) == 1
    @test only(host.binder_types) === recorded_scheme
    @test env.analysis isa TypedAnalysis
end

@testitem "Generated binding callbacks see each remapped attempt host" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, ConstNode
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator, GeneratedBindingInsertionMutation,
        Site, _authoritative_binding_host, _remap_site, apply, generate_binding_rhs
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    struct _FirstBindingAttemptRejected end
    mutable struct AttemptSiteRecordingRHS <: AbstractReplacementGenerator
        seen::Vector{Tuple{AstNode,Union{Nothing,AstNode}}}
    end
    function LambdaRegression.Mutations.generate_binding_rhs(
        generator::AttemptSiteRecordingRHS,
        rhs_ty::TyLike,
        site::Site,
        ::MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        push!(generator.seen, (site.node, site.parent))
        if length(generator.seen) == 1
            return Err(_FirstBindingAttemptRejected())
        end
        return GenerationResult(
            AstTree(ConstNode(:two)),
            rhs_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :attempt_site_recording),
        )
    end

    source = AstTree(AbsNode(:x, AppNode(ConstNode(:f), ConstNode(:one))))
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy), :one => IntTy, :two => IntTy,
    )
    analysis = analyze_program(source, type_ctx; target_ty)
    occurrence = only(findall(
        node -> node isa AppNode,
        analysis.index.nodes,
    ))
    source_site = Site(analysis, occurrence, :app)
    env = binding_env(source, target_ty, type_ctx; seed=143)
    ctx = (; typed_analysis=analysis,
        global_consts=Dict{Symbol,Any}(:one => 1, :two => 2),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    # Reproduce chooser eligibility before the retry attempts. The resulting
    # host must not become a cached callback object for later attempts.
    @test _authoritative_binding_host(source_site, env, ctx) isa Site

    generator = AttemptSiteRecordingRHS(Tuple{AstNode,Union{Nothing,AstNode}}[])
    mutation = GeneratedBindingInsertionMutation(
        TyLike[IntTy]; replacement_generator=generator,
    )
    attempt_one = deepcopy(source)
    site_one = _remap_site(attempt_one, source_site, analysis, ctx)
    failed = apply(mutation, attempt_one, site_one, env, ctx)
    @test failed isa Err
    @test attempt_one == source

    attempt_two = deepcopy(source)
    site_two = _remap_site(attempt_two, source_site, analysis, ctx)
    changed = apply(mutation, attempt_two, site_two, env, ctx)
    @test changed isa AstTree
    @test length(generator.seen) == 2
    @test generator.seen[1][1] === site_one.node
    @test generator.seen[1][2] === site_one.parent
    @test generator.seen[2][1] === site_two.node
    @test generator.seen[2][2] === site_two.parent
    @test generator.seen[1][1] !== generator.seen[2][1]
    @test generator.seen[1][2] !== generator.seen[2][2]
    @test changed == AstTree(AbsNode(:x,
        AppNode(
            AbsNode(Symbol("0"), AppNode(ConstNode(:f), ConstNode(:one))),
            ConstNode(:two),
        ),
    ))
end

const _BINDING_CONTEXT_RIGID_PREFIX = "##binding_rhs_context#"

function _rigidify_binding_context(scheme::TypeSystem.TyScheme)
    quantified = Set(scheme.vars)
    rigid = Dict{TypeSystem.TyVar,TyConst}()
    body = TypeSystem._inst(scheme.ty, var -> begin
        var in quantified && return var
        return get!(rigid, var) do
            TyConst(Symbol(_BINDING_CONTEXT_RIGID_PREFIX, var.uid))
        end
    end)
    return TypeSystem.TyScheme(scheme.vars, body)
end

function _binding_context_free_vars(type_ctx::Dict{Symbol,TyLike})
    free = Set{TypeSystem.TyVar}()
    for entry in values(type_ctx)
        if entry isa TypeSystem.TyScheme
            quantified = Set(entry.vars)
            union!(
                free,
                (
                    var for var in TypeSystem.collect_free_vars(entry.ty)
                    if !(var in quantified)
                ),
            )
        else
            union!(free, TypeSystem.collect_free_vars(entry))
        end
    end
    return free
end

function _infer_binding_rhs_scheme(
    replacement::AstTree,
    host::Site,
    ctx::NamedTuple,
    type_ctx::Dict{Symbol,TyLike},
    factory,
)
    all(expected -> _site_binder_monotype(expected) !== nothing, host.binder_types) ||
        return Err(NoEligibleSiteErr())
    observed = AppNode(
        AbsNode(Symbol("##binding_rhs_probe"), IndexNode(0)),
        deepcopy(replacement.root),
    )
    for index in reverse(eachindex(host.binder_types))
        observed = AbsNode(Symbol("##binding_rhs_scope", index), observed)
    end
    inferred = _infer_mutation_type(AstTree(observed), type_ctx, factory)
    inferred isa Err && return inferred

    current = inferred
    context_vars = _binding_context_free_vars(type_ctx)
    for binder_scheme in host.binder_types
        current = TypeSystem.prune(current)
        current isa TyFun || return Err(_GeneratedBindingRHSShapeErr(current))
        expected = _site_binder_monotype(binder_scheme)::Ty
        expected_copy = TypeSystem._query_copy(
            expected,
            factory,
            Dict{TypeSystem.TyVar,TypeSystem.TyVar}(),
        )
        matched = TypeSystem.unify!(current.arg, expected_copy)
        matched isa Err && return matched
        union!(context_vars, TypeSystem.collect_free_vars(TypeSystem.prune(current.arg)))
        current = current.res
    end

    inferred_rhs = TypeSystem.prune(current)
    generalized = TypeSystem.TyVar[
        var for var in TypeSystem.collect_free_vars(inferred_rhs)
        if !(var in context_vars)
    ]
    return TypeSystem.TyScheme(generalized, inferred_rhs)
end

function _resolve_generated_scoped_node(
    node::AstNode,
    binders::Vector{Int},
    next_binder_id::Base.RefValue{Int},
)
    if node isa ConstNode
        return _ScopedConst(node.name)
    elseif node isa VarNode
        return _ScopedFree(node.name)
    elseif node isa IndexNode
        0 <= node.index < length(binders) ||
            return Err(_UnresolvedScopedIndexErr(node.index, length(binders)))
        return _ScopedBound(binders[end - node.index])
    elseif node isa AbsNode
        binder_id = next_binder_id[]
        next_binder_id[] += 1
        push!(binders, binder_id)
        body = _resolve_generated_scoped_node(node.body, binders, next_binder_id)
        pop!(binders)
        body isa Err && return body
        return _ScopedAbs(binder_id, node.name, body)
    elseif node isa AppNode
        func = _resolve_generated_scoped_node(node.func, binders, next_binder_id)
        func isa Err && return func
        arg = _resolve_generated_scoped_node(node.arg, binders, next_binder_id)
        arg isa Err && return arg
        return _ScopedApp(func, arg)
    end
    return Err(_UnsupportedScopedNodeErr(typeof(node)))
end

function _resolve_generated_scoped_fragment(
    tree::AstTree,
    visible_binders::Vector{Int},
    next_binder_id::Int,
)
    next_id = Ref(next_binder_id)
    root = _resolve_generated_scoped_node(
        tree.root,
        copy(visible_binders),
        next_id,
    )
    root isa Err && return root
    return root, next_id[]
end

required_site_kinds(::GeneratedBindingInsertionMutation) = Symbol[:any]

function is_site_eligible(
    mutation::GeneratedBindingInsertionMutation,
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    authoritative_host = _authoritative_binding_host(host, env, ctx)
    authoritative_host isa Err && return false
    scoped_host = _scoped_at_path(scoped, authoritative_host.path)
    scoped_host isa Err && return false
    if !isempty(authoritative_host.path)
        parent = _scoped_at_path(
            scoped,
            @view(authoritative_host.path[1:(end - 1)]),
        )
        parent isa Err && return false
        _unused_binding_host_preserves_role(
            parent,
            last(authoritative_host.path),
            scoped_host,
        ) ||
            return false
    end
    return !(
        _binding_rhs_type_candidates(mutation, authoritative_host, env, ctx) isa Err
    )
end

function apply(
    mutation::GeneratedBindingInsertionMutation,
    tree::AstTree,
    host::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _require_ctx_fields(
        ctx,
        Symbol[:global_consts, :global_consts_ty, :local_consts, :local_consts_ty],
    )
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    authoritative_host = _authoritative_binding_host(host, env, ctx)
    authoritative_host isa Err && return authoritative_host
    _scoped_at_path(scoped, authoritative_host.path) isa Err &&
        return Err(NoEligibleSiteErr())

    weighted = _binding_rhs_type_candidates(
        mutation,
        authoritative_host,
        env,
        ctx,
    )
    weighted isa Err && return weighted
    candidates, weights = weighted
    rhs_ty = _binding_rhs_type_choice(env.rng, candidates, weights)

    generation_local_consts = copy(ctx.local_consts)
    generation_local_consts_ty = copy(ctx.local_consts_ty)
    generation_ctx = merge(
        ctx,
        (;
            local_consts=generation_local_consts,
            local_consts_ty=generation_local_consts_ty,
        ),
    )
    result = generate_binding_rhs(
        mutation.replacement_generator,
        rhs_ty,
        authoritative_host,
        env,
        generation_ctx,
        mutation.generation_kwargs,
    )
    result isa Err && return result
    result isa GenerationResult || return Err(ArgumentError(
        "binding RHS generator must return GenerationResult or Err",
    ))

    staged_local_consts = copy(ctx.local_consts)
    staged_local_consts_ty = copy(ctx.local_consts_ty)
    replacement = deepcopy(result.tree)
    _merge_replacement_consts!(
        replacement,
        copy(result.local_consts),
        copy(result.local_const_types),
        staged_local_consts,
        staged_local_consts_ty,
        reserved_consts=ctx.global_consts,
        reserved_const_types=ctx.global_consts_ty,
    )

    factory = haskey(ctx, :tyfactory) ? ctx.tyfactory : newtyvar
    inferred_scheme = _infer_binding_rhs_scheme(
        replacement,
        authoritative_host,
        ctx,
        merge(env.type_ctx, staged_local_consts_ty),
        factory,
    )
    inferred_scheme isa Err && return inferred_scheme
    contextual_scheme = _rigidify_binding_context(inferred_scheme)
    TypeSystem.islesseq(rhs_ty, contextual_scheme) || return Err(
        _GeneratedBindingRHSMismatchErr(inferred_scheme, rhs_ty),
    )

    visible_binders = _visible_scoped_binders(scoped, authoritative_host)
    visible_binders isa Err && return visible_binders
    next_binder_id = scoped.next_binder_id
    resolved_rhs = _resolve_generated_scoped_fragment(
        replacement,
        visible_binders,
        next_binder_id,
    )
    resolved_rhs isa Err && return resolved_rhs
    rhs, next_binder_id = resolved_rhs

    name_factory = haskey(ctx, :symfactory) ? ctx.symfactory : SymbolGenerator()
    name = only(_fresh_binder_names(tree, name_factory, 1))
    rewritten = _insert_unused_scoped_binding(
        scoped,
        authoritative_host.path,
        rhs,
        next_binder_id,
        name,
    )
    rewritten isa Err && return rewritten
    candidate = _materialize_scoped(rewritten)
    candidate isa Err && return candidate

    inferred_root = _infer_mutation_type(
        candidate,
        merge(env.type_ctx, staged_local_consts_ty),
        factory,
    )
    inferred_root isa Err && return inferred_root
    target_copy = TypeSystem._query_copy(
        env.target_ty,
        factory,
        Dict{TypeSystem.TyVar,TypeSystem.TyVar}(),
    )
    target_match = TypeSystem.unify!(inferred_root, target_copy)
    target_match isa Err && return target_match

    tree.root = candidate.root
    empty!(ctx.local_consts)
    empty!(ctx.local_consts_ty)
    merge!(ctx.local_consts, staged_local_consts)
    merge!(ctx.local_consts_ty, staged_local_consts_ty)
    return tree
end

"""
    LetInlineMutation()

Inline one direct-`Abs` let-like application through the ordinary
capture-avoiding β-substitution kernel. Eligibility and application resolve the
current node from `site.path`; stale `Site` object identity cannot redirect the
rewrite. The operator is intentionally opt-in because `BetaReduceMutation`
already covers the same redex family in ordinary presets.
"""
struct LetInlineMutation <: AbstractMutation end

required_site_kinds(::LetInlineMutation) = Symbol[:app]

function _is_scoped_let_redex(scoped::_ScopedTree, path::AbstractVector{<:Integer})
    node = _scoped_at_path(scoped, path)
    node isa Err && return false
    return node isa _ScopedApp &&
           node.role === :let_like &&
           node.func isa _ScopedAbs
end

function is_site_eligible(
    ::LetInlineMutation,
    site::Site,
    env::MutationEnv,
    ::NamedTuple,
)
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    return _is_scoped_let_redex(scoped, site.path)
end

function apply(
    ::LetInlineMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    analysis = _typed_mutation_analysis(env, ctx)
    analysis isa Err && return analysis
    current_site = _remap_site(tree, site, analysis, ctx)
    current_site isa Err && return current_site
    current_site.node isa AppNode || return Err(NoEligibleSiteErr())
    app = current_site.node::AppNode
    app.func isa AbsNode || return Err(NoEligibleSiteErr())
    body = apply_node!(tree, app.func::AbsNode, app.arg)
    _replace_site!(tree, current_site, body)
    return tree
end

@testsnippet BindingInsertionTestSupport begin
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstNode, AstTree, ConstNode, IndexNode, SymbolGenerator
    using LambdaRegression.Mutations:
        AdaptivePolicy, BloatPolicy, MutationEnv, Site
    using LambdaRegression.TypeSystem: Ty, TyLike

    function binding_node_at_path(tree::AstTree, path::Vector{Int})
        node = tree.root
        for child_index in path
            if node isa AbsNode && child_index == 1
                node = node.body
            elseif node isa AppNode && child_index == 1
                node = node.func
            elseif node isa AppNode && child_index == 2
                node = node.arg
            else
                error("invalid test path")
            end
        end
        return node
    end

    function binding_site(
        tree::AstTree,
        path::Vector{Int},
        ty::Ty;
        binder_types::Vector{Ty}=Ty[],
    )
        node = binding_node_at_path(tree, path)
        if isempty(path)
            return Site(path, node, nothing, 0, length(binder_types), :any, ty, binder_types)
        end
        parent_path = path[1:(end - 1)]
        parent = binding_node_at_path(tree, parent_path)
        return Site(
            path,
            node,
            parent,
            last(path),
            length(binder_types),
            :any,
            ty,
            binder_types,
        )
    end

    function binding_env(
        tree::AstTree,
        target_ty::Ty,
        type_ctx::Dict{Symbol,TyLike};
        seed::Int=1,
    )
        return MutationEnv(
            MersenneTwister(seed),
            target_ty,
            type_ctx,
            tree,
            nothing,
            false,
            IdDict{Any,Any}(),
            AdaptivePolicy(),
            IdDict{DataType,LambdaRegression.Mutations.MutationStat}(),
            BloatPolicy(),
        )
    end

    binding_ctx() = (; symfactory=SymbolGenerator())
end

@testitem "SourceBindingInsertionMutation inserts and Prune reverses an unused let" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        LetPruneMutation, SourceBindingInsertionMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    original = AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    root_only = (_, host_path, _, _, _) -> isempty(host_path) ? 1.0 : 0.0
    mutation = SourceBindingInsertionMutation(host_weight=root_only)
    expected = AstTree(
        AppNode(
            AbsNode(Symbol("0"), AppNode(ConstNode(:f), ConstNode(:one))),
            ConstNode(:one),
        ),
    )
    source_site = binding_site(original, [2], IntTy)
    env = binding_env(original, IntTy, type_ctx; seed=11)
    ctx = binding_ctx()

    @test apply(mutation, deepcopy(original), source_site, env, ctx) == expected

    root_site = binding_site(expected, Int[], IntTy)
    env2 = binding_env(expected, IntTy, type_ctx; seed=12)
    @test apply(LetPruneMutation(), deepcopy(expected), root_site, env2, ctx) == original
end

@testitem "SourceBindingInsertionMutation avoids named variable capture" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: VarNode, eliminate_bound_variables!
    using LambdaRegression.Mutations: SourceBindingInsertionMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    original = AstTree(
        AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:g), ConstNode(:y)),
                VarNode(:x),
            ),
        ),
    )
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = Dict{Symbol,TyLike}(
        :g => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :y => IntTy,
    )
    choose_body_host = (_, host_path, _, _, _) -> host_path == [1] ? 1.0 : 0.0
    mutation = SourceBindingInsertionMutation(host_weight=choose_body_host)
    ctx = merge(binding_ctx(), (; symfactory=() -> :x))
    attempt = deepcopy(original)
    result = apply(
        mutation,
        attempt,
        binding_site(original, [1, 1, 2], IntTy),
        binding_env(original, target_ty, type_ctx),
        ctx,
    )

    @test result isa AstTree
    result isa AstTree || return
    @test iscompatible(infer(result, type_ctx), target_ty)
    normalized = deepcopy(result)
    eliminate_bound_variables!(normalized)
    @test normalized.root == AbsNode(
        :x,
        AppNode(
            AbsNode(
                :x_1,
                AppNode(
                    AppNode(ConstNode(:g), ConstNode(:y)),
                    IndexNode(1),
                ),
            ),
            ConstNode(:y),
        ),
    )
end

@testitem "SourceBindingInsertionMutation limits hosts to captured binder scope" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: _resolve_scoped, _source_binding_host_depths
    using LambdaRegression.Utils: Err

    tree = AstTree(
        AbsNode(
            :outer,
            AbsNode(
                :inner,
                AppNode(ConstNode(:use_outer), IndexNode(1)),
            ),
        ),
    )
    scoped = _resolve_scoped(tree)
    @test !(scoped isa Err)

    @test _source_binding_host_depths(scoped, [1, 1]) == [1, 2]
end

@testitem "SourceBindingInsertionMutation host candidates stay linear in path depth" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: _resolve_scoped, _source_binding_host_depths

    depth = 512
    node = foldl(
        (body, index) -> AbsNode(Symbol(index), body),
        1:depth;
        init=ConstNode(:value),
    )
    tree = AstTree(node)
    scoped = _resolve_scoped(tree)
    source_path = fill(1, depth)
    host_depths = _source_binding_host_depths(scoped, source_path)

    @test host_depths isa Vector{Int}
    @test host_depths == collect(0:depth)
end

@testitem "SourceBindingInsertionMutation freshens copied lambda binders" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        _ScopedAbs,
        _ScopedBound,
        _freshen_scoped_binders,
        _insert_unused_scoped_binding,
        _materialize_scoped,
        _resolve_scoped,
        _scoped_at_path
    using LambdaRegression.Utils: Err

    tree = AstTree(
        AppNode(
            ConstNode(:consume),
            AbsNode(:value, IndexNode(0)),
        ),
    )
    scoped = _resolve_scoped(tree)
    @test !(scoped isa Err)
    source = _scoped_at_path(scoped, [2])
    @test source isa _ScopedAbs

    copied_source, next_id = _freshen_scoped_binders(source, 2)
    @test copied_source isa _ScopedAbs
    @test next_id == 3
    @test copied_source.binder_id != source.binder_id
    @test copied_source.body == _ScopedBound(copied_source.binder_id)

    rewritten = _insert_unused_scoped_binding(
        scoped,
        Int[],
        copied_source,
        next_id,
        :unused,
    )
    @test !(rewritten isa Err)
    retained_source = _scoped_at_path(rewritten, [1, 1, 2])
    inserted_rhs = _scoped_at_path(rewritten, [2])
    @test retained_source isa _ScopedAbs
    @test inserted_rhs isa _ScopedAbs
    @test retained_source.binder_id != inserted_rhs.binder_id
    @test retained_source.body == _ScopedBound(retained_source.binder_id)
    @test inserted_rhs.body == _ScopedBound(inserted_rhs.binder_id)
    @test _materialize_scoped(rewritten) == AstTree(
        AppNode(
            AbsNode(
                :unused,
                AppNode(ConstNode(:consume), AbsNode(:value, IndexNode(0))),
            ),
            AbsNode(:value, IndexNode(0)),
        ),
    )
end

@testitem "SourceBindingInsertionMutation freshens nested shadowing and preserves capture" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        _ScopedAbs,
        _ScopedApp,
        _ScopedBound,
        _freshen_scoped_binders,
        _resolve_scoped,
        _scoped_at_path

    tree = AstTree(
        AbsNode(
            :outer,
            AbsNode(
                :same,
                AbsNode(
                    :same,
                    AppNode(AppNode(IndexNode(2), IndexNode(1)), IndexNode(0)),
                ),
            ),
        ),
    )
    scoped = _resolve_scoped(tree)
    source = _scoped_at_path(scoped, [1])
    @test source isa _ScopedAbs

    copied, next_id = _freshen_scoped_binders(source, 4)
    @test next_id == 6
    @test copied isa _ScopedAbs
    @test copied.binder_id == 4
    @test copied.body isa _ScopedAbs
    @test copied.body.binder_id == 5
    @test copied.body.body == _ScopedApp(
        _ScopedApp(_ScopedBound(1), _ScopedBound(4)),
        _ScopedBound(5),
    )
end

@testitem "SourceBindingInsertionMutation freshens FunctionDef self binders and preserves captures" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        _ScopedAbs,
        _ScopedApp,
        _ScopedBound,
        _ScopedFunctionDef,
        _freshen_scoped_binders

    source = _ScopedFunctionDef(1,
        _ScopedAbs(2, :x,
            _ScopedApp(
                _ScopedApp(_ScopedBound(2), _ScopedBound(1)),
                _ScopedBound(7),
            ),
        ),
    )
    copied, next_id = _freshen_scoped_binders(source, 10)

    @test copied == _ScopedFunctionDef(10,
        _ScopedAbs(11, :x,
            _ScopedApp(
                _ScopedApp(_ScopedBound(11), _ScopedBound(10)),
                _ScopedBound(7),
            ),
        ),
    )
    @test next_id == 12
end

@testitem "SourceBindingInsertionMutation applies to recursive FunctionDef sources with captured outer binders" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.Mutations:
        Site,
        SourceBindingInsertionMutation,
        _resolve_scoped,
        _source_binding_host_depths,
        apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    type_ctx = Dict{Symbol,TyLike}(
        :FADD => TyFun(IntTy, TyFun(IntTy, IntTy)),
    )
    original = AstTree(AbsNode(:outer, FunctionDefNode(AbsNode(:x,
        AppNode(
            AppNode(ConstNode(:FADD), AppNode(IndexNode(1), IndexNode(0))),
            IndexNode(2),
        ),
    ))))
    scoped = _resolve_scoped(original)
    @test _source_binding_host_depths(scoped, [1]) == [1]
    @test _source_binding_host_depths(scoped, [1, 1, 1, 1, 2, 1]) == [2, 3, 4, 5, 6]

    seen_hosts = Vector{Vector{Int}}()
    mutation = SourceBindingInsertionMutation(
        host_weight=(_, host_path, _, _, _) -> begin
            push!(seen_hosts, copy(host_path))
            host_path == [1] ? 1.0 : 0.0
        end,
    )
    function_def = original.root.body
    site = Site([1], function_def, original.root, 1, 1, :any,
        TyFun(IntTy, IntTy), TyLike[IntTy])
    env = binding_env(original, target_ty, type_ctx)
    result = apply(mutation, deepcopy(original), site, env, binding_ctx())

    @test seen_hosts == [[1]]
    @test result isa AstTree
    result isa AstTree || return
    @test iscompatible(infer(result, type_ctx), target_ty)
end

@testitem "SourceBindingInsertionMutation exact weights retain extreme positive support" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: _integer_weight_units, _weighted_ticket_index

    units = _integer_weight_units(Real[BigFloat(1), BigFloat("1e-1000")])
    @test all(>(0), units)
    @test _weighted_ticket_index(units, sum(units)) == 2
end

@testitem "SourceBindingInsertionMutation is deterministic for a fixed host policy" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: SourceBindingInsertionMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    original = AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    selected_host = (_, host_path::Vector{Int}, _, _, _) ->
        host_path == [2] ? 1 : 0
    mutation = SourceBindingInsertionMutation(host_weight=selected_host)
    source_site = binding_site(original, [2], IntTy)
    expected = AstTree(
        AppNode(
            ConstNode(:f),
            AppNode(
                AbsNode(Symbol("0"), ConstNode(:one)),
                ConstNode(:one),
            ),
        ),
    )

    first_result = apply(
        mutation,
        deepcopy(original),
        source_site,
        binding_env(original, IntTy, type_ctx; seed=31),
        binding_ctx(),
    )
    second_result = apply(
        mutation,
        deepcopy(original),
        source_site,
        binding_env(original, IntTy, type_ctx; seed=31),
        binding_ctx(),
    )
    @test first_result == expected
    @test second_result == expected
end

@testitem "SourceBindingInsertionMutation preserves higher-order source and recursive target typing" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: SourceBindingInsertionMutation, apply
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyLike, TyMu, infer, iscompatible, make_tyvar_factory

    fresh = make_tyvar_factory()
    binder = fresh(0)
    recursive = TyMu(binder, TyFun(IntTy, binder))
    type_ctx = Dict{Symbol,TyLike}(
        :use => TyFun(TyFun(IntTy, IntTy), recursive),
    )
    original = AstTree(
        AppNode(
            ConstNode(:use),
            AbsNode(:value, IndexNode(0)),
        ),
    )
    source_site = binding_site(original, [2], TyFun(IntTy, IntTy))
    root_only = (_, host_path, _, _, _) -> isempty(host_path) ? 1.0 : 0.0
    result = apply(
        SourceBindingInsertionMutation(host_weight=root_only),
        deepcopy(original),
        source_site,
        binding_env(original, recursive, type_ctx; seed=41),
        binding_ctx(),
    )

    @test result isa AstTree
    @test iscompatible(infer(result, type_ctx), recursive)
end

@testitem "SourceBindingInsertionMutation rejects HM role changes transactionally" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        NoEligibleSiteErr,
        SourceBindingInsertionMutation,
        _resolve_scoped,
        _source_binding_host_depths,
        apply
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.Utils: Err

    original = AstTree(
        AppNode(
            AbsNode(:value, IndexNode(0)),
            ConstNode(:one),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy)
    scoped = _resolve_scoped(original)
    @test _source_binding_host_depths(scoped, [1]) == [0]

    function_child_only = (_, host_path, _, _, _) -> host_path == [1] ? 1.0 : 0.0
    attempt = deepcopy(original)
    result = apply(
        SourceBindingInsertionMutation(host_weight=function_child_only),
        attempt,
        binding_site(original, [1], IntTy),
        binding_env(original, IntTy, type_ctx; seed=51),
        binding_ctx(),
    )
    @test result isa Err
    @test result.value isa NoEligibleSiteErr
    @test attempt == original
end

@testitem "SourceBindingInsertionMutation validates host weights transactionally" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        NoEligibleSiteErr,
        SourceBindingInsertionMutation,
        _InvalidBindingHostWeightErr,
        apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    original = AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    source_site = binding_site(original, [2], IntTy)

    for invalid_weight in (-1.0, NaN, Inf, "not-a-real")
        attempt = deepcopy(original)
        mutation = SourceBindingInsertionMutation(
            host_weight=(_, _, _, _, _) -> invalid_weight,
        )
        result = apply(
            mutation,
            attempt,
            source_site,
            binding_env(original, IntTy, type_ctx; seed=61),
            binding_ctx(),
        )
        @test result isa Err
        @test result.value isa _InvalidBindingHostWeightErr
        @test attempt == original
    end

    zero_attempt = deepcopy(original)
    zero_result = apply(
        SourceBindingInsertionMutation(
            host_weight=(_, _, _, _, _) -> 0,
        ),
        zero_attempt,
        source_site,
        binding_env(original, IntTy, type_ctx; seed=62),
        binding_ctx(),
    )
    @test zero_result isa Err
    @test zero_result.value isa NoEligibleSiteErr
    @test zero_attempt == original

    rational_result = apply(
        SourceBindingInsertionMutation(
            host_weight=(_, _, _, _, _) -> 1 // 2,
        ),
        deepcopy(original),
        source_site,
        binding_env(original, IntTy, type_ctx; seed=63),
        binding_ctx(),
    )
    @test rational_result isa AstTree
end

@testitem "SourceBindingInsertionMutation treats site path as authoritative" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        SourceBindingInsertionMutation, Site, apply, is_site_eligible
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyLike

    original = AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    stale = AstTree(AppNode(ConstNode(:other), ConstNode(:stale)))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    stale_site = Site(
        [2],
        stale.root.arg,
        stale.root,
        2,
        0,
        :any,
        IntTy,
        Ty[],
    )
    env = binding_env(original, IntTy, type_ctx; seed=71)
    root_only = (_, host_path, _, _, _) -> isempty(host_path) ? 1.0 : 0.0
    mutation = SourceBindingInsertionMutation(host_weight=root_only)
    expected = AstTree(
        AppNode(
            AbsNode(Symbol("0"), AppNode(ConstNode(:f), ConstNode(:one))),
            ConstNode(:one),
        ),
    )

    @test is_site_eligible(mutation, stale_site, env, binding_ctx())
    @test apply(mutation, deepcopy(original), stale_site, env, binding_ctx()) == expected
end

@testitem "LetInlineMutation substitutes used and unused let binders" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        LetInlineMutation, _default_mutations, apply, is_site_eligible,
        required_site_kinds
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    mutation = LetInlineMutation()
    @test required_site_kinds(mutation) == Symbol[:app]
    @test all(candidate -> !(candidate isa LetInlineMutation), _default_mutations())

    used = AstTree(
        AppNode(
            AbsNode(:value, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:one),
        ),
    )
    used_site = binding_site(used, Int[], IntTy)
    used_result = apply(
        mutation,
        deepcopy(used),
        used_site,
        binding_env(used, IntTy, type_ctx; seed=81),
        binding_ctx(),
    )
    @test used_result == AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    @test iscompatible(infer(used_result, type_ctx), IntTy)

    unused = AstTree(
        AppNode(
            AbsNode(:unused, ConstNode(:one)),
            AppNode(ConstNode(:f), ConstNode(:one)),
        ),
    )
    unused_site = binding_site(unused, Int[], IntTy)
    unused_env = binding_env(unused, IntTy, type_ctx; seed=82)
    @test is_site_eligible(mutation, unused_site, unused_env, binding_ctx())
    @test apply(
        mutation,
        deepcopy(unused),
        unused_site,
        unused_env,
        binding_ctx(),
    ) == AstTree(ConstNode(:one))
end

@testitem "LetInlineMutation preserves a shadowing binder" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: LetInlineMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    original = AstTree(
        AppNode(
            AbsNode(:outer, AbsNode(:outer, IndexNode(0))),
            ConstNode(:one),
        ),
    )
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy)
    result = apply(
        LetInlineMutation(),
        deepcopy(original),
        binding_site(original, Int[], target_ty),
        binding_env(original, target_ty, type_ctx; seed=83),
        binding_ctx(),
    )

    @test result == AstTree(AbsNode(:outer, IndexNode(0)))
    @test iscompatible(infer(result, type_ctx), target_ty)
end

@testitem "LetInlineMutation shifts an outer capture after removing the let binder" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations: LetInlineMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    original = AstTree(
        AbsNode(
            :outer,
            AppNode(
                AbsNode(:bound, AbsNode(:inner, IndexNode(2))),
                ConstNode(:one),
            ),
        ),
    )
    target_ty = TyFun(IntTy, TyFun(IntTy, IntTy))
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy)
    result = apply(
        LetInlineMutation(),
        deepcopy(original),
        binding_site(original, [1], TyFun(IntTy, IntTy)),
        binding_env(original, target_ty, type_ctx; seed=90),
        binding_ctx(),
    )

    @test result == AstTree(AbsNode(:outer, AbsNode(:inner, IndexNode(1))))
    @test iscompatible(infer(result, type_ctx), target_ty)
end

@testitem "LetInlineMutation preserves polymorphic let use" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.Mutations: LetInlineMutation, apply
    using LambdaRegression.TypeSystem: TyFun, TyProd, TyScheme, TyLike, infer, iscompatible

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    type_ctx = merge(
        TypeSystem.get_type_ctx(Dict{Symbol,Any}(
            :PAIR => LambdaRegression.Primitives.PAIR,
            :one => 1,
            :truth => true,
        )),
        Dict{Symbol,TyLike}(
            :ID => TyScheme([quantified], TyFun(quantified, quantified)),
        ),
    )
    rhs = AppNode(ConstNode(:ID), AbsNode(:identity, IndexNode(0)))
    body = AppNode(
        AppNode(ConstNode(:PAIR), AppNode(IndexNode(0), ConstNode(:one))),
        AppNode(IndexNode(0), ConstNode(:truth)),
    )
    original = AstTree(AppNode(AbsNode(:polymorphic, body), rhs))
    target_ty = TyProd(TypeSystem.IntTy, TypeSystem.BoolTy)
    expected = AstTree(
        AppNode(
            AppNode(
                ConstNode(:PAIR),
                AppNode(deepcopy(rhs), ConstNode(:one)),
            ),
            AppNode(deepcopy(rhs), ConstNode(:truth)),
        ),
    )
    result = apply(
        LetInlineMutation(),
        deepcopy(original),
        binding_site(original, Int[], target_ty),
        binding_env(original, target_ty, type_ctx; seed=84),
        binding_ctx(),
    )

    @test result == expected
    @test iscompatible(infer(result, type_ctx), target_ty)
end

@testitem "LetInlineMutation reverses source insertion" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        LetInlineMutation, SourceBindingInsertionMutation, apply
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    original = AstTree(AppNode(ConstNode(:f), ConstNode(:one)))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    root_only = (_, host_path, _, _, _) -> isempty(host_path) ? 1.0 : 0.0
    inserted = apply(
        SourceBindingInsertionMutation(host_weight=root_only),
        deepcopy(original),
        binding_site(original, [2], IntTy),
        binding_env(original, IntTy, type_ctx; seed=85),
        binding_ctx(),
    )
    inlined = apply(
        LetInlineMutation(),
        deepcopy(inserted),
        binding_site(inserted, Int[], IntTy),
        binding_env(inserted, IntTy, type_ctx; seed=86),
        binding_ctx(),
    )

    @test inlined == original
end

@testitem "LetInlineMutation treats the apply-time path as authoritative" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        LetInlineMutation, Site, apply, is_site_eligible
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyLike
    using LambdaRegression.Utils: Err

    inner_redex = AppNode(AbsNode(:value, IndexNode(0)), ConstNode(:one))
    original = AstTree(AppNode(ConstNode(:f), inner_redex))
    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    stale_site = Site(
        [1],
        inner_redex,
        original.root,
        2,
        0,
        :app,
        IntTy,
        Ty[],
    )
    env = binding_env(original, IntTy, type_ctx; seed=87)
    attempt = deepcopy(original)

    @test !is_site_eligible(LetInlineMutation(), stale_site, env, binding_ctx())
    @test apply(LetInlineMutation(), attempt, stale_site, env, binding_ctx()) isa Err
    @test attempt == original
end

@testitem "LetInlineMutation integrates existing RHS and body replacement APIs" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        DefaultSiteLockPolicy,
        IndexedProvider,
        PathSiteSelectionPolicy,
        SubtreeReplacementMutation,
        WeightedPairChooser,
        generate_replacement,
        mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    struct FixedBindingReplacement <: AbstractReplacementGenerator
        node::AstNode
    end

    function LambdaRegression.Mutations.generate_replacement(
        replacement::FixedBindingReplacement,
        site::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(deepcopy(replacement.node)),
            site.ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :fixed_binding_replacement),
        )
    end

    type_ctx = Dict{Symbol,TyLike}(
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
        :two => IntTy,
    )
    original = AstTree(
        AppNode(
            AbsNode(:bound, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:one),
        ),
    )

    function replace_at(path::Vector{Int}, replacement::AstNode; seed::Int)
        return mutate_tree(
            original,
            IntTy;
            rng=MersenneTwister(seed),
            global_consts_ty=type_ctx,
            local_consts=Dict{Symbol,Any}(),
            local_consts_ty=Dict{Symbol,TyLike}(),
            mutations=[
                SubtreeReplacementMutation(;
                    site_lock=DefaultSiteLockPolicy(spine_only=false),
                    site_selection=PathSiteSelectionPolicy(path),
                    replacement_generator=FixedBindingReplacement(replacement),
                ),
            ],
            chooser=WeightedPairChooser(
                weights=Dict{Type,Float64}(SubtreeReplacementMutation => 1.0),
            ),
            provider=IndexedProvider(),
            max_try=5,
            prune_consts=true,
        )
    end

    rhs_replaced = replace_at([2], ConstNode(:two); seed=88)
    @test rhs_replaced == AstTree(
        AppNode(
            AbsNode(:bound, AppNode(ConstNode(:f), IndexNode(0))),
            ConstNode(:two),
        ),
    )
    @test iscompatible(infer(rhs_replaced, type_ctx), IntTy)

    body_replaced = replace_at([1, 1], IndexNode(0); seed=89)
    @test body_replaced == AstTree(
        AppNode(AbsNode(:bound, IndexNode(0)), ConstNode(:one)),
    )
    @test iscompatible(infer(body_replaced, type_ctx), IntTy)
end

@testitem "GeneratedBindingInsertionMutation validates and detaches finite catalogs" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.Mutations: GeneratedBindingInsertionMutation
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyRigid, TyScheme

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    identity = TyScheme([quantified], TyFun(quantified, quantified))
    catalog = TyLike[IntTy, identity]
    mutation = GeneratedBindingInsertionMutation(catalog)

    @test mutation.rhs_types == (IntTy, identity)
    push!(catalog, IntTy)
    @test mutation.rhs_types == (IntTy, identity)

    @test_throws ArgumentError GeneratedBindingInsertionMutation(TyLike[])
    @test_throws ArgumentError GeneratedBindingInsertionMutation(
        Iterators.repeated(IntTy),
    )

    free = fresh(0)
    @test_throws ArgumentError GeneratedBindingInsertionMutation(TyLike[free])

    unquantified = fresh(0)
    malformed = TyScheme([quantified], TyFun(quantified, unquantified))
    @test_throws ArgumentError GeneratedBindingInsertionMutation(TyLike[malformed])
    @test_throws ArgumentError GeneratedBindingInsertionMutation(TyLike[TyRigid(1)])
    @test_throws ArgumentError GeneratedBindingInsertionMutation(
        TyLike[TyFun(IntTy, TyRigid(2))],
    )

    for key in (
        :config,
        :rng,
        :scope_tys,
        :global_consts,
        :global_consts_ty,
        :local_consts,
        :local_consts_ty,
        :source_local_consts,
    )
        generation_kwargs = NamedTuple{(key,)}((nothing,))
        @test_throws ArgumentError GeneratedBindingInsertionMutation(
            TyLike[IntTy];
            generation_kwargs,
        )
    end
end

@testitem "Generated scoped fragments retain visible outer binders" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, IndexNode
    using LambdaRegression.Mutations:
        _ScopedAbs, _ScopedApp, _ScopedBound, _resolve_generated_scoped_fragment
    using LambdaRegression.Utils: Err

    fragment = AstTree(AbsNode(
        :local,
        AppNode(IndexNode(1), AppNode(IndexNode(2), IndexNode(0))),
    ))
    resolved, next_id = _resolve_generated_scoped_fragment(fragment, [41, 42], 7)
    @test resolved == _ScopedAbs(
        7,
        :local,
        _ScopedApp(
            _ScopedBound(42),
            _ScopedApp(_ScopedBound(41), _ScopedBound(7)),
        ),
    )
    @test next_id == 8

    unresolved = _resolve_generated_scoped_fragment(
        AstTree(IndexNode(2)),
        [41, 42],
        7,
    )
    @test unresolved isa Err
    @test unresolved.value isa LambdaRegression.Mutations._UnresolvedScopedIndexErr
    @test unresolved.value.depth == 2
end

@testitem "GeneratedBindingInsertionMutation adapters use the selected demand and host scope" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        ScopedReplacementGenerator, Site, generate_binding_rhs
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, Ty, TyLike

    observed = Ref{Any}()
    function fixed_scoped(target_ty; kwargs...)
        observed[] = (;
            target_ty,
            scope_tys=kwargs[:scope_tys],
            rng=kwargs[:rng],
            marker=kwargs[:marker],
        )
        return GenerationResult(
            AstTree(IndexNode(0)),
            target_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :fixed_scoped),
        )
    end

    tree = AstTree(ConstNode(:body))
    type_ctx = Dict{Symbol,TyLike}(:body => BoolTy)
    env = binding_env(tree, BoolTy, type_ctx; seed=101)
    host = Site(Int[], tree.root, nothing, 0, 2, :any, BoolTy, Ty[FloatTy, IntTy])
    ctx = (;
        global_consts=Dict{Symbol,Any}(:body => true),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
    result = generate_binding_rhs(
        ScopedReplacementGenerator(fixed_scoped),
        IntTy,
        host,
        env,
        ctx,
        (; marker=:task3),
    )

    @test result isa GenerationResult
    @test observed[].target_ty === IntTy
    @test observed[].scope_tys == Ty[IntTy, FloatTy]
    @test observed[].rng === env.rng
    @test observed[].marker === :task3
end

@testitem "GeneratedBindingInsertionMutation inserts a fixed weighted RHS" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        GeneratedBindingInsertionMutation,
        NoEligibleSiteErr,
        _InvalidBindingRHSTypeWeightErr,
        apply,
        generate_binding_rhs,
        is_site_eligible
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    mutable struct FixedIntBindingRHS <: AbstractReplacementGenerator
        demands::Vector{TyLike}
    end
    function LambdaRegression.Mutations.generate_binding_rhs(
        generator::FixedIntBindingRHS,
        rhs_ty::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        push!(generator.demands, rhs_ty)
        return GenerationResult(
            AstTree(ConstNode(:two)),
            rhs_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :fixed_int),
        )
    end

    original = AstTree(ConstNode(:one))
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy, :two => IntTy)
    ctx = (;
        global_consts=Dict{Symbol,Any}(:one => 1, :two => 2),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    generator = FixedIntBindingRHS(TyLike[])
    mutation = GeneratedBindingInsertionMutation(
        TyLike[FloatTy, IntTy];
        replacement_generator=generator,
        rhs_type_weight=(rhs_ty, _, _, _) -> rhs_ty === IntTy ? 1 : 0,
    )
    result = apply(
        mutation,
        deepcopy(original),
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=102),
        ctx,
    )

    @test result == AstTree(
        AppNode(AbsNode(Symbol("0"), ConstNode(:one)), ConstNode(:two)),
    )
    @test generator.demands == TyLike[IntTy]

    role_tree = AstTree(
        AppNode(AbsNode(:bound, IndexNode(0)), ConstNode(:one)),
    )
    role_site = binding_site(role_tree, [1], TyFun(IntTy, IntTy))
    @test !is_site_eligible(
        mutation,
        role_site,
        binding_env(role_tree, IntTy, type_ctx; seed=103),
        ctx,
    )

    for (weight, error_type) in ((-1.0, _InvalidBindingRHSTypeWeightErr), (0.0, NoEligibleSiteErr))
        attempt = deepcopy(original)
        failed = apply(
            GeneratedBindingInsertionMutation(
                TyLike[IntTy];
                replacement_generator=FixedIntBindingRHS(TyLike[]),
                rhs_type_weight=(_, _, _, _) -> weight,
            ),
            attempt,
            binding_site(original, Int[], IntTy),
            binding_env(original, IntTy, type_ctx; seed=103),
            ctx,
        )
        @test failed isa Err
        @test failed.value isa error_type
        @test attempt == original
    end
end

@testitem "GeneratedBindingInsertionMutation avoids named variable capture" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression: CanonicalGeneration, Mutations, ProgramAnalysis
    using LambdaRegression.ASTs.NodeASTs: VarNode, eliminate_bound_variables!
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        GeneratedBindingInsertionMutation,
        _remap_site,
        apply
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.ProgramAnalysis: analyze_program
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer, iscompatible

    struct FixedYBindingRHS <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_binding_rhs(
        ::FixedYBindingRHS,
        rhs_ty::TyLike,
        ::Mutations.Site,
        ::Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(ConstNode(:y)),
            rhs_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(),
        )
    end

    original = AstTree(
        AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:g), ConstNode(:y)),
                VarNode(:x),
            ),
        ),
    )
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = Dict{Symbol,TyLike}(
        :g => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :y => IntTy,
    )
    analysis = analyze_program(original, type_ctx; target_ty)
    occurrence = Mutations.occurrence_id(analysis.index, [1])
    source_site = Mutations.Site(analysis, occurrence, :any)
    attempt = deepcopy(original)
    env = binding_env(original, target_ty, type_ctx)
    ctx = merge(
        binding_ctx(),
        (;
            typed_analysis=analysis,
            global_consts=Dict{Symbol,Any}(:g => ((a, b) -> (a, b)), :y => 1),
            global_consts_ty=type_ctx,
            local_consts=Dict{Symbol,Any}(),
            local_consts_ty=Dict{Symbol,TyLike}(),
            symfactory=() -> :x,
        ),
    )
    attempt_site = _remap_site(attempt, source_site, analysis, ctx)
    mutation = GeneratedBindingInsertionMutation(
        TyLike[IntTy];
        replacement_generator=FixedYBindingRHS(),
    )
    result = apply(mutation, attempt, attempt_site, env, ctx)

    @test result isa AstTree
    result isa AstTree || return
    @test iscompatible(infer(result, type_ctx), target_ty)
    normalized = deepcopy(result)
    eliminate_bound_variables!(normalized)
    @test normalized.root == AbsNode(
        :x,
        AppNode(
            AbsNode(
                :x_1,
                AppNode(
                    AppNode(ConstNode(:g), ConstNode(:y)),
                    IndexNode(1),
                ),
            ),
            ConstNode(:y),
        ),
    )
end

@testitem "GeneratedBindingInsertionMutation verifies polymorphic and lying RHS generators" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression: TypeSystem
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        GeneratedBindingInsertionMutation,
        apply,
        generate_binding_rhs
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, Ty, TyFun, TyLike, TyScheme, infer_scheme, iscompatible
    using LambdaRegression.Utils: Err

    struct FixedIdentityBindingRHS <: AbstractReplacementGenerator
        node::AstNode
    end
    function LambdaRegression.Mutations.generate_binding_rhs(
        generator::FixedIdentityBindingRHS,
        rhs_ty::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(deepcopy(generator.node)),
            rhs_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :fixed_identity),
        )
    end

    fresh = TypeSystem.make_tyvar_factory()
    quantified = fresh(0)
    identity = TyScheme([quantified], TyFun(quantified, quantified))
    original = AstTree(ConstNode(:one))
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy)
    ctx = (;
        global_consts=Dict{Symbol,Any}(:one => 1),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    mutation = GeneratedBindingInsertionMutation(
        TyLike[identity];
        replacement_generator=FixedIdentityBindingRHS(
            AbsNode(:identity, IndexNode(0)),
        ),
    )
    result = apply(
        mutation,
        deepcopy(original),
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=104),
        ctx,
    )

    @test result isa AstTree
    rhs_scheme = infer_scheme(
        AstTree(deepcopy(result.root.arg)),
        type_ctx;
        factory=TypeSystem.make_tyvar_factory(),
    )
    @test rhs_scheme isa TyScheme
    @test length(rhs_scheme.vars) == 1
    @test iscompatible(rhs_scheme, identity)

    lying_attempt = deepcopy(original)
    lying = apply(
        GeneratedBindingInsertionMutation(
            TyLike[identity];
            replacement_generator=FixedIdentityBindingRHS(ConstNode(:one)),
        ),
        lying_attempt,
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=105),
        ctx,
    )
    @test lying isa Err
    @test lying_attempt == original

    narrow_attempt = deepcopy(original)
    narrow = apply(
        GeneratedBindingInsertionMutation(
            TyLike[identity];
            replacement_generator=FixedIdentityBindingRHS(
                AbsNode(:ignored, ConstNode(:one)),
            ),
        ),
        narrow_attempt,
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=106),
        ctx,
    )
    @test narrow isa Err
    @test narrow_attempt == original

    captured_quantified = fresh(0)
    universally_inhabited = TyScheme([captured_quantified], captured_quantified)
    captured_tree = AstTree(AbsNode(:outer, ConstNode(:one)))
    captured_attempt = deepcopy(captured_tree)
    captured = apply(
        GeneratedBindingInsertionMutation(
            TyLike[universally_inhabited];
            replacement_generator=FixedIdentityBindingRHS(IndexNode(0)),
        ),
        captured_attempt,
        binding_site(captured_tree, [1], IntTy; binder_types=Ty[IntTy]),
        binding_env(
            captured_tree,
            TyFun(IntTy, IntTy),
            type_ctx;
            seed=107,
        ),
        ctx,
    )
    @test captured isa Err
    @test captured_attempt == captured_tree

    concrete_capture_attempt = deepcopy(captured_tree)
    concrete_capture = apply(
        GeneratedBindingInsertionMutation(
            TyLike[IntTy];
            replacement_generator=FixedIdentityBindingRHS(IndexNode(0)),
        ),
        concrete_capture_attempt,
        binding_site(captured_tree, [1], IntTy; binder_types=Ty[IntTy]),
        binding_env(
            captured_tree,
            TyFun(IntTy, IntTy),
            type_ctx;
            seed=108,
        ),
        ctx,
    )
    @test concrete_capture isa AstTree

    collision_tree = AstTree(
        AbsNode(
            :int_binder,
            AppNode(AbsNode(:bool_binder, IndexNode(0)), ConstNode(:truth)),
        ),
    )
    collision_type_ctx = Dict{Symbol,TyLike}(:truth => BoolTy)
    collision_ctx = (;
        global_consts=Dict{Symbol,Any}(:truth => true),
        global_consts_ty=collision_type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    collision_attempt = deepcopy(collision_tree)
    collision = apply(
        GeneratedBindingInsertionMutation(
            TyLike[BoolTy];
            replacement_generator=FixedIdentityBindingRHS(IndexNode(0)),
        ),
        collision_attempt,
        binding_site(collision_tree, [1], BoolTy; binder_types=Ty[IntTy]),
        binding_env(
            collision_tree,
            TyFun(IntTy, BoolTy),
            collision_type_ctx;
            seed=109,
        ),
        collision_ctx,
    )
    @test collision isa Err
    @test collision_attempt == collision_tree
end

@testitem "Generated binding RHS checks use the environment and authoritative path scope" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        GeneratedBindingInsertionMutation,
        apply,
        generate_binding_rhs
    using LambdaRegression.TypeSystem:
        BoolTy, IntTy, Ty, TyFun, TyLike, TyScheme, make_tyvar_factory
    using LambdaRegression.Utils: Err

    struct FixedReviewedBindingRHS <: AbstractReplacementGenerator
        node::AstNode
    end
    function LambdaRegression.Mutations.generate_binding_rhs(
        generator::FixedReviewedBindingRHS,
        rhs_ty::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(deepcopy(generator.node)),
            rhs_ty,
            Dict{Symbol,Any}(),
            Dict{Symbol,TyLike}(),
            Dict{Symbol,Any}(:source => :review_fixture),
        )
    end

    fresh = make_tyvar_factory()
    environment_var = fresh(0)
    quantified = fresh(0)
    universal_value = TyScheme([quantified], quantified)
    original = AstTree(ConstNode(:one))
    open_type_ctx = Dict{Symbol,TyLike}(
        :one => IntTy,
        :k => environment_var,
    )
    open_ctx = (;
        global_consts=Dict{Symbol,Any}(:one => 1, :k => 2),
        global_consts_ty=open_type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    open_attempt = deepcopy(original)
    open_result = apply(
        GeneratedBindingInsertionMutation(
            TyLike[universal_value];
            replacement_generator=FixedReviewedBindingRHS(ConstNode(:k)),
        ),
        open_attempt,
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, open_type_ctx; seed=110),
        open_ctx,
    )
    @test open_result isa Err
    @test open_attempt == original

    scoped_tree = AstTree(AbsNode(:flag, ConstNode(:one)))
    scoped_type_ctx = Dict{Symbol,TyLike}(:one => IntTy)
    scoped_ctx = (;
        global_consts=Dict{Symbol,Any}(:one => 1),
        global_consts_ty=scoped_type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    stale_attempt = deepcopy(scoped_tree)
    stale_result = apply(
        GeneratedBindingInsertionMutation(
            TyLike[IntTy];
            replacement_generator=FixedReviewedBindingRHS(IndexNode(0)),
        ),
        stale_attempt,
        binding_site(scoped_tree, [1], IntTy; binder_types=Ty[IntTy]),
        binding_env(
            scoped_tree,
            TyFun(BoolTy, IntTy),
            scoped_type_ctx;
            seed=111,
        ),
        scoped_ctx,
    )
    @test stale_result isa Err
    @test stale_attempt == scoped_tree

    duplicate_tree = AstTree(
        AppNode(
            AppNode(
                ConstNode(:add),
                AppNode(
                    ConstNode(:consume_int),
                    AbsNode(:int_identity, IndexNode(0)),
                ),
            ),
            AppNode(
                ConstNode(:consume_bool),
                AbsNode(:bool_identity, IndexNode(0)),
            ),
        ),
    )
    duplicate_type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :consume_int => TyFun(TyFun(IntTy, IntTy), IntTy),
        :consume_bool => TyFun(TyFun(BoolTy, BoolTy), IntTy),
    )
    duplicate_ctx = (;
        global_consts=Dict{Symbol,Any}(
            :add => (x -> y -> x + y),
            :consume_int => (f -> f(1)),
            :consume_bool => (f -> f(true) ? 1 : 0),
        ),
        global_consts_ty=duplicate_type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        symfactory=SymbolGenerator(),
    )
    duplicate_attempt = deepcopy(duplicate_tree)
    duplicate_result = apply(
        GeneratedBindingInsertionMutation(
            TyLike[BoolTy];
            replacement_generator=FixedReviewedBindingRHS(IndexNode(0)),
        ),
        duplicate_attempt,
        binding_site(
            duplicate_tree,
            [1, 2, 2, 1],
            IntTy;
            binder_types=Ty[IntTy],
        ),
        binding_env(
            duplicate_tree,
            IntTy,
            duplicate_type_ctx;
            seed=112,
        ),
        duplicate_ctx,
    )
    @test duplicate_result isa Err
    @test duplicate_attempt == duplicate_tree
end

@testitem "GeneratedBindingInsertionMutation stages constants and failures transactionally" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.CanonicalGeneration: GenerationResult
    using LambdaRegression.Mutations:
        AbstractReplacementGenerator,
        GeneratedBindingInsertionMutation,
        apply,
        generate_binding_rhs
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyLike
    using LambdaRegression.Utils: Err

    struct ConflictingBindingRHS <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_binding_rhs(
        ::ConflictingBindingRHS,
        rhs_ty::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(ConstNode(:k)),
            rhs_ty,
            Dict{Symbol,Any}(:k => 2.0),
            Dict{Symbol,TyLike}(:k => FloatTy),
            Dict{Symbol,Any}(:source => :conflicting),
        )
    end

    struct FailedBindingGeneration end
    struct FailingBindingRHS <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_binding_rhs(
        ::FailingBindingRHS,
        ::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ctx::NamedTuple,
        ::NamedTuple,
    )
        ctx.local_consts[:leak] = 9.0
        ctx.local_consts_ty[:leak] = FloatTy
        return Err(FailedBindingGeneration())
    end


    struct GlobalConflictingBindingRHS <: AbstractReplacementGenerator end
    function LambdaRegression.Mutations.generate_binding_rhs(
        ::GlobalConflictingBindingRHS,
        rhs_ty::TyLike,
        ::LambdaRegression.Mutations.Site,
        ::LambdaRegression.Mutations.MutationEnv,
        ::NamedTuple,
        ::NamedTuple,
    )
        return GenerationResult(
            AstTree(ConstNode(:one)),
            rhs_ty,
            Dict{Symbol,Any}(:one => 2),
            Dict{Symbol,TyLike}(:one => IntTy),
            Dict{Symbol,Any}(:source => :global_conflicting),
        )
    end

    original = AstTree(ConstNode(:one))
    type_ctx = Dict{Symbol,TyLike}(:one => IntTy, :k => FloatTy)
    local_consts = Dict{Symbol,Any}(:k => 1.0)
    local_consts_ty = Dict{Symbol,TyLike}(:k => FloatTy)
    ctx = (;
        global_consts=Dict{Symbol,Any}(:one => 1),
        global_consts_ty=Dict{Symbol,TyLike}(:one => IntTy),
        local_consts,
        local_consts_ty,
        symfactory=SymbolGenerator(),
    )
    inserted = apply(
        GeneratedBindingInsertionMutation(
            TyLike[FloatTy];
            replacement_generator=ConflictingBindingRHS(),
        ),
        deepcopy(original),
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=106),
        ctx,
    )

    @test inserted isa AstTree
    @test inserted.root.arg isa ConstNode
    renamed = inserted.root.arg.name
    @test renamed != :k
    @test local_consts == Dict{Symbol,Any}(:k => 1.0, renamed => 2.0)
    @test local_consts_ty == Dict{Symbol,TyLike}(:k => FloatTy, renamed => FloatTy)

    global_collision = apply(
        GeneratedBindingInsertionMutation(
            TyLike[IntTy];
            replacement_generator=GlobalConflictingBindingRHS(),
        ),
        deepcopy(original),
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=107),
        ctx,
    )
    @test global_collision isa AstTree
    @test global_collision.root.func.body == ConstNode(:one)
    @test global_collision.root.arg isa ConstNode
    @test global_collision.root.arg.name != :one
    @test !haskey(local_consts, :one)

    failure_tree = deepcopy(original)
    before_consts = copy(local_consts)
    before_types = copy(local_consts_ty)
    failed = apply(
        GeneratedBindingInsertionMutation(
            TyLike[FloatTy];
            replacement_generator=FailingBindingRHS(),
        ),
        failure_tree,
        binding_site(original, Int[], IntTy),
        binding_env(original, IntTy, type_ctx; seed=108),
        ctx,
    )
    @test failed isa Err
    @test failure_tree == original
    @test local_consts == before_consts
    @test local_consts_ty == before_types
end

@testitem "GeneratedBindingInsertionMutation canonical smoke stays opt in" setup=[BindingInsertionTestSupport] begin
    using Test
    using Random
    using LambdaRegression.Mutations:
        GeneratedBindingInsertionMutation,
        IndexedProvider,
        WeightedPairChooser,
        _default_mutations,
        mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyLike, infer, iscompatible

    original = AstTree(ConstNode(:one))
    global_consts = Dict{Symbol,Any}(:one => 1)
    global_consts_ty = Dict{Symbol,TyLike}(:one => IntTy)
    mutation = GeneratedBindingInsertionMutation(
        TyLike[IntTy];
        generation_kwargs=(;
            sampling_depth_limit=1,
            sampling_node_budget=8,
            guarded_expansion_depth=0,
        ),
    )
    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(42),
        global_consts,
        global_consts_ty,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GeneratedBindingInsertionMutation => 1.0),
        ),
        provider=IndexedProvider(),
        max_try=3,
    )

    @test result isa AstTree
    @test result.root isa AppNode
    @test result.root.func isa AbsNode
    @test iscompatible(infer(result, global_consts_ty), IntTy)
    @test all(candidate -> !(candidate isa GeneratedBindingInsertionMutation), _default_mutations())
    @test all(
        candidate -> !(candidate isa GeneratedBindingInsertionMutation),
        LambdaRegression.Regression.default_components().mutations,
    )
end

@testitem "WeightedPairChooser accepts parametric mutation family weights" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        SourceBindingInsertionMutation, WeightedPairChooser, _mutation_weight

    mutation = SourceBindingInsertionMutation(
        host_weight=(_, _, _, _, _) -> 1.0,
    )
    family_chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(SourceBindingInsertionMutation => 0.75),
    )
    @test _mutation_weight(family_chooser, mutation, 0.1) == 0.75

    concrete_chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(
            SourceBindingInsertionMutation => 0.75,
            typeof(mutation) => 1.25,
        ),
    )
    @test _mutation_weight(concrete_chooser, mutation, 0.1) == 1.25

    legacy_chooser = WeightedPairChooser(
        weights=Dict{DataType,Float64}(typeof(mutation) => 0.5),
    )
    @test _mutation_weight(legacy_chooser, mutation, 0.1) == 0.5
end

@testitem "Binding insertion preset exposes only accepted source insertion" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        GeneratedBindingInsertionMutation,
        LetInlineMutation,
        SourceBindingInsertionMutation,
        WeightedPairChooser,
        _default_mutations

    mutation_defaults = _default_mutations()
    regression_defaults = LambdaRegression.Regression.default_components().mutations
    for defaults in (mutation_defaults, regression_defaults)
        @test count(candidate -> candidate isa SourceBindingInsertionMutation, defaults) == 1
        @test all(candidate -> !(candidate isa GeneratedBindingInsertionMutation), defaults)
        @test all(candidate -> !(candidate isa LetInlineMutation), defaults)
    end

    chooser = WeightedPairChooser()
    @test chooser.weights[SourceBindingInsertionMutation] == 0.015
end

@testitem "Binding insertion omission avoids scoped analysis" setup=[BindingInsertionTestSupport] begin
    using Test
    using LambdaRegression.Mutations:
        AbstractMutation,
        IndexedProvider,
        PointMutation,
        WeightedPairChooser,
        _resolve_scoped,
        choose_pair,
        prepare!
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, newtyvar

    tree = AstTree(AbsNode(:value, IndexNode(0)))
    target_ty = TyFun(IntTy, IntTy)
    type_ctx = Dict{Symbol,TyLike}()
    env = binding_env(tree, target_ty, type_ctx; seed=201)
    ctx = (;
        rng=env.rng,
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=type_ctx,
        local_consts=Dict{Symbol,Any}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
    )
    provider = IndexedProvider()
    @test prepare!(provider, tree, type_ctx, env, ctx) === nothing
    choose_pair(
        WeightedPairChooser(weights=Dict{Type,Float64}(PointMutation => 1.0)),
        AbstractMutation[PointMutation()],
        provider,
        tree,
        type_ctx,
        env,
        ctx,
    )
    @test !haskey(env.eligibility_cache, _resolve_scoped)
end

@testitem "Binding insertion mixed routing is observable and always changes" setup=[BindingInsertionTestSupport] begin
    using Test
    using Random
    using LambdaRegression.Mutations:
        AbstractMutation,
        IndexedProvider,
        LetPruneMutation,
        MutationAttemptRecorder,
        SourceBindingInsertionMutation,
        WeightedPairChooser,
        mutate_tree,
        mutation_recorder_summary
    using LambdaRegression.TypeSystem: IntTy, TyLike, infer, iscompatible

    original = AstTree(
        AppNode(AbsNode(:unused, ConstNode(:one)), ConstNode(:two)),
    )
    global_consts = Dict{Symbol,Any}(:one => 1, :two => 2)
    global_consts_ty = Dict{Symbol,TyLike}(:one => IntTy, :two => IntTy)
    source = SourceBindingInsertionMutation()
    prune = LetPruneMutation()
    mutations = AbstractMutation[source, prune]
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(
            SourceBindingInsertionMutation => 1.0,
            LetPruneMutation => 1.0,
        ),
        cache_eligibility=true,
    )
    recorder = MutationAttemptRecorder()

    for seed in 1:64
        result = mutate_tree(
            original,
            IntTy;
            rng=MersenneTwister(30_000 + seed),
            global_consts,
            global_consts_ty,
            local_consts=Dict{Symbol,Any}(),
            local_consts_ty=Dict{Symbol,TyLike}(),
            mutations,
            chooser,
            provider=IndexedProvider(),
            recorder,
            max_try=5,
        )
        @test result isa AstTree
        @test result != original
        @test iscompatible(infer(result, global_consts_ty), IntTy)
    end

    successes = Dict(
        entry.mutation_type => entry.counts.success
        for entry in mutation_recorder_summary(recorder).per_mutation
    )
    @test get(successes, typeof(source), 0) > 0
    @test get(successes, typeof(prune), 0) > 0
end

@testitem "Binding insertion default routing remains observable" setup=[BindingInsertionTestSupport] begin
    using Test
    using Random
    using LambdaRegression.Mutations:
        MutationAttemptRecorder,
        SourceBindingInsertionMutation,
        mutate_tree,
        mutation_recorder_summary
    using LambdaRegression.TypeSystem: IntTy, TyLike, infer, iscompatible

    original = AstTree(ConstNode(:one))
    global_consts = Dict{Symbol,Any}(:one => 1, :two => 2)
    global_consts_ty = Dict{Symbol,TyLike}(:one => IntTy, :two => IntTy)
    recorder = MutationAttemptRecorder()

    for seed in 1:128
        result = mutate_tree(
            original,
            IntTy;
            rng=MersenneTwister(40_000 + seed),
            global_consts,
            global_consts_ty,
            local_consts=Dict{Symbol,Any}(),
            local_consts_ty=Dict{Symbol,TyLike}(),
            recorder,
            max_try=10,
        )
        @test result isa AstTree
        @test result != original
        @test iscompatible(infer(result, global_consts_ty), IntTy)
    end

    source_successes = sum(
        entry.counts.success
        for entry in mutation_recorder_summary(recorder).per_mutation
        if entry.mutation_type <: SourceBindingInsertionMutation
    )
    @test source_successes > 0
end
