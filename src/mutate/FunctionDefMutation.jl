#=
    FunctionDefMutation

Typed mutation operators for common `FunctionDefNode` programs. This file is
included into `Mutations`, sharing its site analysis, transaction, and final
type-validation path while keeping FunctionDef-specific scope transport and
generation separate by responsibility.

```julia
using LambdaRegression: CanonicalGeneration, Mutations

mutation = Mutations.FunctionDefMutation(
    :recursive_body;
    functiondef_provider=CanonicalGeneration.FunctionDefProvider(max_nesting=1, body_attempts=16),
    generation_kwargs=(sampling_depth_limit=8, sampling_node_budget=96),
)
child = Mutations.mutate_tree(parent, target; mutations=[mutation], max_try=16)
```

`FunctionDefMutation(mode)` has three explicit modes:

- `:finite_body` keeps the wrapper and replaces its body with a finite witness
  that cannot refer to the self binder;
- `:recursive_body` keeps the wrapper and forces the guarded FunctionDef
  provider, so the replacement contains a genuine self call;
- `:remove` replaces the whole FunctionDef with a compatible finite witness.

On common syntax, every mode reads the selected target and exact outer binder
schemes from the call's root-constrained `TypedAnalysis`; this preserves
let-like polymorphism alongside monomorphic lambda and FunctionDef-self slots.
There is no alternate checker or metadata path: the shared analysis is the only
source of site and binder facts. Generated constants remain transactional,
structural no-ops are rejected, and successful edits finish with complete
shared TypeSystem inference.

Customization is layered and opt-in:

- `site_weight=(mode, site, env, ctx) -> weight` changes FunctionDef-site routing;
- `functiondef_provider` replaces base-witness, admissibility, nesting, and body
  attempt policy; the mutation default uses 16 body attempts to avoid rebuilding
  the surrounding mutation analysis for sparse recursive candidates;
- `generation_kwargs` forwards weights, providers, planner, sampler, materializer,
  and finite budgets; a supplied `GenerationConfig` keeps every component except
  that its RNG is replaced by the current mutation RNG. The built-in FunctionDef
  provider currently requires the default lazy sampler and argument planner;
  replace `replacement_generator` to use a different generation kernel;
- `include_local_consts=true` admits caller local constants as source heads;
  values already represented by an identical compatible global are deduplicated,
  while factory/user-local values remain available, globals are always visible,
  and generated constants remain transactional. When the caller knows those
  dictionaries came from canonical materialization,
  `deduplicate_provider_values=true` additionally opts into
  `CanonicalGeneration.provider_replays_materialized_value`;
- `replacement_generator` replaces generation without replacing scope recovery,
  rebinding, no-op checks, constant merging, or final verification.

A custom generator accepts `(mode, target_ty, scope::FunctionDefScope, env, ctx,
provider, generation_kwargs)` and returns `GenerationResult` or `Err`. Treat
`scope`, `env`, and `ctx` as read-only. The returned result remains generator-
owned and is never mutated. To wrap the built-in behavior, call
`generate_functiondef_replacement(DefaultFunctionDefMutationGenerator(), args...)`
from the custom callable, then inspect or replace the returned result.

```julia
custom_generator = function (mode, target_ty, scope, env, ctx, provider, kwargs)
    result = generate_functiondef_replacement(
        DefaultFunctionDefMutationGenerator(),
        mode, target_ty, scope, env, ctx, provider, kwargs,
    )
    result isa Err && return result
    return result
end
```

Custom head providers that recreate a materialized constant may opt into the
same conservative deduplication by defining
`provider_replays_materialized_value(provider, value, ty) -> Bool`. Leave the
default `false` when the provider is a factory or does not guarantee the exact
same runtime value.

Expected finite failures include no eligible FunctionDef, no finite witness,
exhausted recursive-body attempts, a forbidden FunctionDef in finite modes, an
unresolved outer binder, a structural no-op, or failed final inference. Invalid
callback signatures and non-finite weights throw. Omitting this mutation avoids
every FunctionDef-specific scan; retaining it with chooser weight zero does not.
=#
using ..ASTs.NodeASTs
using ..ASTs.LexicalContexts: BinderId
using ..Programs: ConstSymbolFactory
using ..Programs: Program, _publish_program
using ..CanonicalGeneration: GenerationResult
import ..CanonicalGeneration as CG
using ..ProgramAnalysis:
    TypedAnalysis, functiondef_recursion_observations, visible_binder_ids

export FunctionDefMutation, FunctionDefScope, generate_functiondef_replacement
export DefaultFunctionDefMutationGenerator

const _FUNCTIONDEF_MUTATION_MODES = (:finite_body, :recursive_body, :remove)
const _FUNCTIONDEF_GENERATION_CONTEXT_KEYS = (
    :rng,
    :scope_tys,
    :global_consts,
    :global_consts_ty,
    :local_consts,
    :local_consts_ty,
    :source_local_consts,
    :deduplicate_provider_values,
)
"""Public adapter for composing with the built-in FunctionDef replacement generator."""
struct DefaultFunctionDefMutationGenerator end
mutable struct _FunctionDefScopePlaceholder
    slot::Int
end

"""
    FunctionDefScope

Read-only mutation-facing view of binders outside a selected FunctionDef.
`binder_types` and slots are inner → outer. Monomorphic binders appear in
`scope_tys`; `full_slot_by_scope_index[i]` maps their temporary generation index
back to the complete binder chain. Truly polymorphic binders use the temporary
constant in `placeholder_by_slot` so each occurrence is freshly instantiated.
"""
struct FunctionDefScope
    binder_types::Tuple{Vararg{TyLike}}
    scope_tys::Vector{Ty}
    full_slot_by_scope_index::Vector{Int}
    placeholder_by_slot::Dict{Int,Symbol}
    functiondef_depth::Int
end

struct FunctionDefMutation{Mode,P,K,W,G} <: Mutations.AbstractMutation
    functiondef_provider::P
    generation_kwargs::K
    site_weight_fn::W
    replacement_generator::G
    include_local_consts::Bool
    deduplicate_provider_values::Bool
end

"""
    FunctionDefMutation(mode; functiondef_provider, generation_kwargs,
                        site_weight, replacement_generator,
                        include_local_consts=true,
                        deduplicate_provider_values=false)

Create an opt-in complete-FunctionDef mutation. `mode` is `:finite_body`,
`:recursive_body`, or `:remove`; see the `Mutations` module docstring and
the custom replacement-generator contract for shared typing and customization.
"""
function FunctionDefMutation(
    mode::Symbol;
    functiondef_provider::CG.FunctionDefProvider=
        CG.FunctionDefProvider(body_attempts=16),
    generation_kwargs::NamedTuple=(;
        sampling_depth_limit=4,
        sampling_node_budget=64,
        guarded_expansion_depth=1,
    ),
    site_weight=(mode, site, env, ctx) -> 1.0,
    replacement_generator=DefaultFunctionDefMutationGenerator(),
    include_local_consts::Bool=true,
    deduplicate_provider_values::Bool=false,
)
    mode in _FUNCTIONDEF_MUTATION_MODES || throw(ArgumentError(
        "FunctionDef mutation mode must be :finite_body, :recursive_body, or :remove",
    ))
    for key in _FUNCTIONDEF_GENERATION_CONTEXT_KEYS
        haskey(generation_kwargs, key) || continue
        throw(ArgumentError("generation_kwargs must not override FunctionDef context key :$key"))
    end
    return FunctionDefMutation{mode,typeof(functiondef_provider),typeof(generation_kwargs),typeof(site_weight),typeof(replacement_generator)}(
        functiondef_provider,
        generation_kwargs,
        site_weight,
        replacement_generator,
        include_local_consts,
        deduplicate_provider_values,
    )
end

_mode(::FunctionDefMutation{Mode}) where {Mode} = Mode

Mutations.required_site_kinds(::FunctionDefMutation) = Symbol[:any]
Mutations.is_site_eligible(
    ::FunctionDefMutation,
    site::Mutations.Site,
    env::Mutations.MutationEnv,
    ctx::NamedTuple,
) =
    site.node isa FunctionDefNode

function Mutations.site_weight(
    m::FunctionDefMutation,
    site::Mutations.Site,
    env::Mutations.MutationEnv,
    ctx::NamedTuple,
)
    value = m.site_weight_fn(_mode(m), site, env, ctx)
    value isa Real || throw(ArgumentError("FunctionDef site weight must be real"))
    weight = Float64(value)
    isfinite(weight) && weight >= 0 || throw(ArgumentError(
        "FunctionDef site weight must be finite and nonnegative",
    ))
    return weight
end

function _fresh_scope_placeholder(slot::Int, occupied::Set{Symbol})
    suffix = 0
    while true
        name = suffix == 0 ?
            Symbol("##functiondef_scope_", slot) :
            Symbol("##functiondef_scope_", slot, "_", suffix)
        name in occupied || return name
        suffix += 1
    end
end

function _functiondef_scope(
    bindings::Vector{TyLike},
    functiondef_depth::Int,
    ctx,
    reserved_names=(),
)
    inner_to_outer = reverse(bindings)
    occupied = Set{Symbol}((
        keys(ctx.global_consts)...,
        keys(ctx.global_consts_ty)...,
        keys(ctx.local_consts)...,
        keys(ctx.local_consts_ty)...,
    ))
    union!(occupied, reserved_names)
    scope_tys = Ty[]
    full_slots = Int[]
    placeholders = Dict{Int,Symbol}()
    for (slot, binder_ty) in enumerate(inner_to_outer)
        if binder_ty isa TyScheme && !isempty(binder_ty.vars)
            name = _fresh_scope_placeholder(slot, occupied)
            push!(occupied, name)
            placeholders[slot] = name
        else
            ty = binder_ty isa TyScheme ? binder_ty.ty : binder_ty
            ty isa Ty || throw(ArgumentError("FunctionDef binder must be Ty or TyScheme"))
            push!(scope_tys, ty)
            push!(full_slots, slot)
        end
    end
    return FunctionDefScope(
        Tuple(inner_to_outer),
        scope_tys,
        full_slots,
        placeholders,
        functiondef_depth,
    )
end

function _analysis_occurrence_at_path(analysis::TypedAnalysis, path::Vector{Int})
    occurrence = 1
    for child_index in path
        children_ids = analysis.index.children[occurrence]
        1 <= child_index <= length(children_ids) || return Err(BoundsError(
            children_ids,
            child_index,
        ))
        occurrence = children_ids[child_index]
    end
    return occurrence
end

function _visible_functiondef_context_from_analysis(
    tree,
    path,
    analysis::TypedAnalysis,
    ctx;
    reserved_names=(),
)
    occurrence = _analysis_occurrence_at_path(analysis, path)
    occurrence isa Err && return occurrence
    source_node = analysis.index.nodes[occurrence]
    source_node isa FunctionDefNode || return Err(ArgumentError(
        "prepared common analysis no longer matches the FunctionDef site path",
    ))

    attempt_node = tree.root
    for child_index in path
        kids = children(attempt_node)
        1 <= child_index <= length(kids) || return Err(BoundsError(kids, child_index))
        attempt_node = kids[child_index]
    end
    attempt_node isa FunctionDefNode || return Err(ArgumentError(
        "attempt tree no longer has a FunctionDef at the prepared site path",
    ))

    raw_target_ty = get(analysis.metadata, occurrence, nothing)
    raw_target_ty === nothing && return Err(ArgumentError(
        "common analysis lacks metadata for the FunctionDef site",
    ))
    target_ty = TypeSystem.prune(raw_target_ty[1])
    target_ty isa Ty || return Err(ArgumentError(
        "cannot recover monomorphic FunctionDef target type from common analysis",
    ))

    binder_ids = visible_binder_ids(analysis, occurrence)
    outer_to_inner = TyLike[]
    for binder in Iterators.reverse(binder_ids)
        haskey(analysis.binder_schemes, binder) || return Err(ArgumentError(
            "common analysis lacks a scheme for visible binder $binder",
        ))
        push!(outer_to_inner, analysis.binder_schemes[binder])
    end
    functiondef_depth = count(
        binder -> analysis.index.bindings[binder].kind === :function_def,
        binder_ids,
    )
    return (
        scope=_functiondef_scope(
            outer_to_inner,
            functiondef_depth,
            ctx,
            reserved_names,
        ),
        target_ty,
    )
end

function _visible_functiondef_context(tree, path, env, ctx; reserved_names=())
    analysis = Mutations._typed_mutation_analysis(env, ctx)
    analysis isa Err && return analysis
    return _visible_functiondef_context_from_analysis(
        tree,
        path,
        analysis,
        ctx;
        reserved_names,
    )
end

function _scope_generation_context(scope::FunctionDefScope, ctx)
    global_consts = copy(ctx.global_consts)
    global_consts_ty = copy(ctx.global_consts_ty)
    for (slot, name) in scope.placeholder_by_slot
        global_consts[name] = _FunctionDefScopePlaceholder(slot)
        global_consts_ty[name] = scope.binder_types[slot]
    end
    return merge(ctx, (; global_consts, global_consts_ty))
end

function _runtime_generation_config(config, rng)
    config isa CG.GenerationConfig || throw(ArgumentError(
        "generation_kwargs.config must be GenerationConfig",
    ))
    return CG.GenerationConfig(
        rng=rng,
        weights=config.weights,
        providers=config.providers,
        argument_planner=config.argument_planner,
        sampler=config.sampler,
        materializer=config.materializer,
        guarded_expansion_depth=config.guarded_expansion_depth,
        sampling_depth_limit=config.sampling_depth_limit,
        sampling_node_budget=config.sampling_node_budget,
    )
end

function _runtime_generation_kwargs(generation_kwargs, rng)
    haskey(generation_kwargs, :config) || return generation_kwargs
    return merge(generation_kwargs, (; config=_runtime_generation_config(
        generation_kwargs.config,
        rng,
    )))
end

function _candidate_local_context(
    ctx,
    providers=();
    deduplicate_provider_values::Bool=false,
)
    local_consts = Dict{Symbol,Any}()
    local_const_types = Dict{Symbol,TyLike}()
    for (name, value) in ctx.local_consts
        haskey(ctx.local_consts_ty, name) || continue
        local_ty = ctx.local_consts_ty[name]
        represented = any(ctx.global_consts) do (global_name, global_value)
            value === global_value || return false
            haskey(ctx.global_consts_ty, global_name) || return false
            return TypeSystem.iscompatible(ctx.global_consts_ty[global_name], local_ty)
        end
        if deduplicate_provider_values
            represented |= any(provider ->
                CG.provider_replays_materialized_value(provider, value, local_ty),
                providers,
            )
        end
        represented && continue
        local_consts[name] = value
        local_const_types[name] = local_ty
    end
    return local_consts, local_const_types
end

function _generation_kwargs(
    scope::FunctionDefScope,
    env,
    ctx,
    generation_kwargs,
    include_local_consts::Bool,
    deduplicate_provider_values::Bool,
)
    configured = _runtime_generation_kwargs(generation_kwargs, env.rng)
    providers = haskey(configured, :config) ?
        configured.config.providers :
        get(configured, :providers, CG.default_head_providers())
    local_consts, local_const_types = include_local_consts ?
        _candidate_local_context(
            ctx,
            providers;
            deduplicate_provider_values,
        ) :
        (Dict{Symbol,Any}(), Dict{Symbol,TyLike}())
    return merge(
        configured,
        (;
            rng=env.rng,
            scope_tys=scope.scope_tys,
            global_consts=ctx.global_consts,
            global_consts_ty=ctx.global_consts_ty,
            local_consts,
            local_consts_ty=local_const_types,
            source_local_consts=include_local_consts,
        ),
    )
end

function _remaining_functiondef_provider(provider, enclosing_depth::Int)
    remaining = max(provider.max_nesting - enclosing_depth, 0)
    remaining == provider.max_nesting && return provider
    return CG.FunctionDefProvider(
        weight_bucket=provider.weight_bucket,
        base_capability_mask=provider.base_capability_mask,
        base_witness=provider.base_witness,
        admissibility=provider.admissibility,
        max_nesting=remaining,
        body_attempts=provider.body_attempts,
        _route_token=provider.route_token,
    )
end

function generate_functiondef_replacement(
    ::DefaultFunctionDefMutationGenerator,
    mode::Symbol,
    target_ty::Ty,
    scope::FunctionDefScope,
    env,
    ctx,
    functiondef_provider,
    generation_kwargs,
)
    include_local_consts = get(generation_kwargs, :include_local_consts, false)
    deduplicate_provider_values = get(
        generation_kwargs,
        :deduplicate_provider_values,
        false,
    )
    clean_kwargs = Base.structdiff(
        generation_kwargs,
        (; include_local_consts=nothing, deduplicate_provider_values=nothing),
    )
    kwargs = _generation_kwargs(
        scope,
        env,
        ctx,
        clean_kwargs,
        include_local_consts,
        deduplicate_provider_values,
    )
    if mode === :recursive_body
        provider = _remaining_functiondef_provider(functiondef_provider, scope.functiondef_depth)
        return CG.generate_functiondef_root_scoped(
            target_ty;
            functiondef_provider=provider,
            kwargs...,
        )
    end
    finite_kwargs = merge(kwargs, (;
        capability_mask=(:function_def, :fixed_point, :opaque),
    ))
    base_config = haskey(finite_kwargs, :config) ?
        finite_kwargs.config : CG._generation_config_from_kwargs(finite_kwargs)
    finite_config = CG.with_functiondef_provider(base_config, nothing)
    clean_finite_kwargs = Base.structdiff(finite_kwargs, (; config=nothing))
    result = CG.generate_canonical_scoped(
        target_ty;
        clean_finite_kwargs...,
        config=finite_config,
    )
    if result isa Err && result.value isa CG.UnsupportedCanonicalGenerationTarget
        return CG.generate_finite_canonical_scoped(target_ty; kwargs...)
    end
    return result
end

function generate_functiondef_replacement(
    generator,
    mode::Symbol,
    target_ty::Ty,
    scope::FunctionDefScope,
    env,
    ctx,
    functiondef_provider,
    generation_kwargs,
)
    applicable(
        generator,
        mode,
        target_ty,
        scope,
        env,
        ctx,
        functiondef_provider,
        generation_kwargs,
    ) || throw(ArgumentError(
        "replacement_generator must accept (mode, target_ty, scope, env, ctx, provider, generation_kwargs)",
    ))
    return generator(
        mode,
        target_ty,
        scope,
        env,
        ctx,
        functiondef_provider,
        generation_kwargs,
    )
end

function _procedural_child_depth(node::AstNode, child_index::Int)
    node isa AbsNode && return child_index == 1 ? 1 : 0
    node isa FunctionDefNode && return child_index == 1 ? 1 : 0
    return 0
end

function _rebind_generated_scope!(
    node::AstNode,
    scope::FunctionDefScope,
    base_offset::Int,
    placeholder_slots::AbstractDict{Symbol,Int},
    internal_depth::Int=0,
)
    if node isa IndexNode
        node.index < internal_depth && return node
        scope_index = node.index - internal_depth + 1
        1 <= scope_index <= length(scope.full_slot_by_scope_index) || return Err(
            ArgumentError("generated replacement contains an unresolved outer index"),
        )
        full_slot = scope.full_slot_by_scope_index[scope_index]
        return IndexNode(internal_depth + base_offset + full_slot - 1)
    elseif node isa ConstNode
        full_slot = get(placeholder_slots, node.name, nothing)
        full_slot === nothing ||
            return IndexNode(internal_depth + base_offset + full_slot - 1)
        return node
    end

    for (child_index, child) in enumerate(children(node))
        rebound = _rebind_generated_scope!(
            child,
            scope,
            base_offset,
            placeholder_slots,
            internal_depth + _procedural_child_depth(node, child_index),
        )
        rebound isa Err && return rebound
        setchild!(node, child_index, rebound)
    end
    return node
end

function _scope_placeholder_slots(result, scope::FunctionDefScope, generation_ctx)
    slots = Dict{Symbol,Int}(name => slot for (slot, name) in scope.placeholder_by_slot)
    for (slot, original_name) in scope.placeholder_by_slot
        marker = generation_ctx.global_consts[original_name]
        for (name, value) in result.local_consts
            value === marker && (slots[name] = slot)
        end
    end
    return slots
end

function _result_consts_without_scope_placeholders(result, placeholder_slots)
    local_consts = copy(result.local_consts)
    local_const_types = copy(result.local_const_types)
    for name in keys(placeholder_slots)
        delete!(local_consts, name)
        delete!(local_const_types, name)
    end
    return local_consts, local_const_types
end

function _merge_result_consts!(replacement, result, placeholder_slots, ctx)
    local_consts, local_const_types =
        _result_consts_without_scope_placeholders(result, placeholder_slots)
    return Mutations._merge_replacement_consts!(
        replacement,
        local_consts,
        local_const_types,
        ctx.local_consts,
        ctx.local_consts_ty;
        reserved_consts=ctx.global_consts,
        reserved_const_types=ctx.global_consts_ty,
    )
end

function _contains_functiondef(node::AstNode)
    node isa FunctionDefNode && return true
    return any(_contains_functiondef, children(node))
end

function _reserved_generation_names(generation_kwargs)
    names = Symbol[]
    for key in (:presets, :preset_types)
        haskey(generation_kwargs, key) || continue
        append!(names, Symbol[name for name in keys(generation_kwargs[key])])
    end
    return names
end

function _root_functiondef_has_self_call(node::FunctionDefNode)
    observations = functiondef_recursion_observations(node)
    return !isempty(observations) && first(observations).self_calls > 0
end

function Mutations.apply(
    m::FunctionDefMutation,
    tree::AstTree,
    site::Mutations.Site,
    env::Mutations.MutationEnv,
    ctx::NamedTuple,
)
    site.node isa FunctionDefNode || return Err(Mutations.NoEligibleSiteErr())
    site_context = _visible_functiondef_context(
        tree,
        site.path,
        env,
        ctx;
        reserved_names=_reserved_generation_names(m.generation_kwargs),
    )
    site_context isa Err && return site_context
    scope = site_context.scope
    target_ty = site_context.target_ty
    generation_ctx = _scope_generation_context(scope, ctx)

    runtime_generation_kwargs = _runtime_generation_kwargs(m.generation_kwargs, env.rng)
    result = generate_functiondef_replacement(
        m.replacement_generator,
        _mode(m),
        target_ty,
        scope,
        env,
        generation_ctx,
        m.functiondef_provider,
        merge(runtime_generation_kwargs, (;
            include_local_consts=m.include_local_consts,
            deduplicate_provider_values=m.deduplicate_provider_values,
        )),
    )
    result isa Err && return result
    result isa GenerationResult || return Err(ArgumentError(
        "FunctionDef replacement generator must return GenerationResult or Err",
    ))
    replacement = deepcopy(result.tree)

    mode = _mode(m)
    mode === :recursive_body || !_contains_functiondef(replacement.root) ||
        return Err(ArgumentError("finite FunctionDef mutation cannot return FunctionDef nodes"))
    base_offset = mode === :finite_body ? 1 : 0
    placeholder_slots = _scope_placeholder_slots(result, scope, generation_ctx)
    rebound = _rebind_generated_scope!(
        replacement.root,
        scope,
        base_offset,
        placeholder_slots,
    )
    rebound isa Err && return rebound
    replacement.root = rebound
    _merge_result_consts!(replacement, result, placeholder_slots, ctx)

    if mode === :remove
        Mutations._replace_site!(tree, site, replacement)
        return tree
    elseif mode === :recursive_body
        replacement.root isa FunctionDefNode || return Err(ArgumentError(
            "recursive_body replacement must have a FunctionDef root",
        ))
        _root_functiondef_has_self_call(replacement.root) || return Err(ArgumentError(
            "recursive_body replacement must call its root self binder",
        ))
        body = (replacement.root::FunctionDefNode).body
        body == site.node.body && return Err(Mutations.NoEligibleSiteErr())
        site.node.body = deepcopy(body)
        return tree
    end

    replacement.root == site.node.body && return Err(Mutations.NoEligibleSiteErr())
    site.node.body = replacement.root
    return tree
end
