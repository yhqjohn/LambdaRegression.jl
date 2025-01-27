"""
    CanonicalGeneration

Canonical finite-core generation with explicit catalog injection.

The default call is intentionally small:

```julia
result = generate_canonical(target_ty)
```

`default_head_providers()` supplies the documented generation boundary: caller
context heads, `NIL` through `ValueFactories.NilFactory`, anchored container
heads (`TOLIST` and `AMAP`), and bounded guarded expansions such as `CONS`,
`COND`, and `AFOLDL`. Scalar real operators, boolean primitives, comparison
operators, observers such as `LENF`, and non-list value factories are ordinary
opt-in context/provider inputs, not default core privileges.

All initial context is caller supplied:

- `global_consts` / `global_consts_ty` for named external constants;
- `local_consts` / `local_consts_ty` for available local sources;
- `presets` / `preset_types` for typed AST templates or other known terms.

For non-AST Julia values, the `_ty` dictionaries are optional when
`TypeSystem.get_type(value)` is defined; explicit `_ty` entries override
inference. `ValueFactories.factory_context` can pre-materialize typed constants
for selected closed types, while `TypedValueFactoryProvider(factories)` can add
the same factory family as lazy instance-factory heads.

Generation is configured by replacement points rather than by editing the core
recursion:

```julia
providers = (
    default_head_providers()...,
)
global_consts = test_catalog_consts()
weights = GenerationWeights(variable=2, constant=0.25, custom_source=0)
config = GenerationConfig(; rng, providers, weights)
result = generate_canonical(target_ty; global_consts, config)
```

For the rich test catalog, pair `test_catalog_consts()` with
`test_catalog_providers()`. The latter is an explicit provider tuple that adds
typed value factories without promoting scalar or ndarray factories into the
default core boundary. `test_catalog_generation_kwargs(profile=:deeper)` also
supplies an explicit deeper sampling profile with higher `lambda`, guarded,
multi-argument, and concrete-head tag weights.

Custom `AbstractHeadProvider`s may extend the finite universe with
`add_universe_types!` and append `HeadEntry` values with `add_heads!`. A head can
use a built-in bucket or `WeightBucket(:my_source)`; `GenerationWeights` accepts
any symbol keyword and defaults unknown buckets to `default=1`. Providers that
own runtime values should put `value=...` in head metadata. Instance factories
put `factory=(head, relation) -> value` or `factory=(head, relation, rng) ->
value` in metadata. A provider-local argument policy can be stored as
`argument_planner=(derivs, arg_ty, arg_class, rule, relation) -> derivs` in
metadata. Source-array arguments are solved from
grounded sources by default; providers can opt into a shallow anchored array
source search with `source_array_depth_limit=N` in head metadata. Context heads
without metadata are materialized from the supplied const dictionaries.
`WeightTagProvider(:FADD => (:real_binary, :multi_arg))` can add stackable
per-head weight tags; a head's effective sampling weight is its base bucket
weight multiplied by every configured tag weight.

`GenerationConfig` also accepts callable `argument_planner`, `sampler`, and
`materializer` fields plus `guarded_expansion_depth` for the bounded expansion
layer. Replacing them changes local proof-tree pruning, derivation selection,
AST lowering, or guarded depth without adding new branches to the core. Provider
order matters, but provider categories and weight keys are open. Same-type
`AMAP` is still an anchored core rule when its source array is grounded; weight
or sampler policies may make it rarer, but the core does not exclude it.
Function-typed scope binders are sampled as local application heads under the
same depth and node fuel as ordinary heads; tune them with
`GenerationWeights(scope_application=...)`, or omit that key to inherit
`variable`.

Safe list access is opt-in through `SafeListAccessProvider()`, which is included
in `test_catalog_providers()` but not in the minimal default provider tuple. It
does not add `UNCONS_CASE`, `HEAD`, `TAIL`, or generic sum/case heads to the
ordinary catalog. Instead, it collects finite list sources from scope binders,
typed list constants, anchored `TOLIST` views of known arrays, and optional
`source_collectors`, then lowers proxy-using bodies to guarded `UNCONS_CASE`
terms. Use `GenerationWeights(; safe_list_access=..., tolist=...,
array_source=...)` to tune the route. The proxy-using success body uses a
constructive core/anchored-head path; guarded expansions may wrap safe-list
access, but the proxy-enforcing subproblem does not recursively open guarded
search just to discover a proxy use. A custom `source_collector` is a callable
`collector(relation, scope_tys)` returning `HeadEntry`, `ListAccessSource`, or
`(TyList, CoreDerivation)` values; non-empty custom collectors disable the
default relation cache because arbitrary callables do not have stable cache
identity.

Generalised target demands may be passed as `TyScheme`, for example
`TyScheme([α], α → α)`. The public call rigidifies quantified type variables to
private closed dummy atoms, runs the same finite monomorphic generator, then
discharges public result annotations back to ordinary scheme-facing types.
Length-polymorphic scheme targets remain outside this boundary. Rigid atoms are
not eligible for scalar or ndarray factories; `NilFactory` may still provide
`NIL : List ρ`, matching the canonical empty-list assumption. Rigidified calls
filter provider and source-collector outputs so ordinary monomorphic heads
cannot manufacture `ρ` witnesses. Cacheable rigidified calls may reuse the
internal monomorphic relation keyed by the rigid target and catalog inputs; the
public result is still discharged back to the original scheme, and non-cacheable
custom collectors/providers keep the no-cache fallback.

"""
module CanonicalGeneration

import Random
import ..TypeSystem
import ..ValueFactories

using ..ASTs.NodeASTs: AstNode, AstTree, ConstNode, IndexNode, AbsNode, AppNode
using ..TypeSystem: Ty, TyConst, TyVar, TyFun, TyProd, TySum, TyList, TyArray, TyNat, TyMu, TyScheme, TyLike
using ..TypeSystem: BoolTy, FloatTy, get_ast_type_ctx, get_type_ctx, get_type, iscompatible, typekey, →
using ..Utils
using TestItems

export generate_canonical, generate_canonical_scoped
export GenerationWeights, GenerationConfig, GenerationResult
export AbstractHeadProvider, add_heads!, add_universe_types!, add_universe_type!, default_head_providers
export HeadEntry, CoreProductive, Decoration, Excluded
export test_catalog_consts, test_catalog_providers, test_catalog_weight_tags, test_catalog_weights, test_catalog_generation_kwargs, WeightBucket
export ContextHeadProvider, NilFactoryProvider, ToListProvider, AMapProvider, GuardedExpansionProvider, TypedValueFactoryProvider, WeightTagProvider
export SafeListAccessProvider
export DefaultArgumentPlanner, WeightedDerivationSampler, DefaultAstMaterializer

abstract type CoreRuleClass end
struct CoreProductive <: CoreRuleClass end
struct Decoration <: CoreRuleClass end
struct Excluded <: CoreRuleClass end

abstract type CoreWeightBucket end
struct VariableWeight <: CoreWeightBucket end
struct ConstantWeight <: CoreWeightBucket end
struct PresetTermWeight <: CoreWeightBucket end
struct InstanceFactoryWeight <: CoreWeightBucket end
struct ArraySourceWeight <: CoreWeightBucket end
struct ToListWeight <: CoreWeightBucket end
struct AMapWeight <: CoreWeightBucket end
struct GuardedExpansionWeight <: CoreWeightBucket end
struct NamedWeightBucket <: CoreWeightBucket
    key::Symbol
end

WeightBucket(key::Symbol) = NamedWeightBucket(key)

const _VARIABLE_WEIGHT = VariableWeight()
const _CONSTANT_WEIGHT = ConstantWeight()
const _PRESET_TERM_WEIGHT = PresetTermWeight()
const _INSTANCE_FACTORY_WEIGHT = InstanceFactoryWeight()
const _ARRAY_SOURCE_WEIGHT = ArraySourceWeight()
const _TOLIST_WEIGHT = ToListWeight()
const _AMAP_WEIGHT = AMapWeight()
const _GUARDED_EXPANSION_WEIGHT = GuardedExpansionWeight()
const _LAMBDA_WEIGHT = WeightBucket(:lambda)

"""
    CoreUniverse(target_ty; global_consts_ty, local_consts_ty, presets, preset_types, scope_source_tys)

Finite set of closed type-fragment members used by the canonical core catalog.
`scope_source_tys` closes over binder/source types that are available in the
current `Known(G, T)` scope but are not global heads.
"""
struct CoreUniverse
    types::Vector{Ty}
    typekeys::Set{Any}
    lengths::Vector{Int}
end

"""
    HeadEntry

Typed candidate head before sampling. Instance factories such as `NIL` are
represented here as catalog metadata, not as TypeSystem primitives.
"""
struct HeadEntry
    name::Symbol
    source::Symbol
    type::TyLike
    rule_class::CoreRuleClass
    weight_bucket::CoreWeightBucket
    metadata::NamedTuple
end

"""
    CoreRule

Lightweight monomorphic fact rule derived from a closed catalog head.
"""
struct CoreRule
    head::HeadEntry
    result_ty::Ty
    argument_tys::Vector{Ty}
    argument_classes::Vector{Symbol}
end

struct CoreCatalog
    universe::CoreUniverse
    heads::Vector{HeadEntry}
    rules::Vector{CoreRule}
end

"""
    AbstractHeadProvider

Composable provider interface for adding typed catalog heads. Providers are
called in tuple order, so later providers may depend on heads added earlier.
"""
abstract type AbstractHeadProvider end

struct ContextHeadProvider <: AbstractHeadProvider end
struct NilFactoryProvider <: AbstractHeadProvider end
struct TypedValueFactoryProvider{F} <: AbstractHeadProvider
    factories::F
end
struct ToListProvider <: AbstractHeadProvider end
struct AMapProvider <: AbstractHeadProvider end
struct GuardedExpansionProvider <: AbstractHeadProvider end
"""
    SafeListAccessProvider(; source_array_depth_limit=0, source_collectors=())

Enable specialised safe list access without adding `UNCONS_CASE`, `HEAD`,
`TAIL`, or generic sum elimination to the ordinary head catalog. The generator
collects a finite set of list sources, samples a proxy-using body, then lowers
the result to `UNCONS_CASE source ok fallback`.

The default source collectors cover current scope list binders, typed list
constants, and anchored `TOLIST` views of known array sources. Extra
`source_collectors` are callables invoked as `collector(relation, scope_tys)`;
each may return `HeadEntry` values with `TyList` type, `ListAccessSource`
values, or `(TyList, CoreDerivation)` pairs. Non-empty custom collectors disable
the default relation cache because arbitrary callables do not have stable cache
identity.
"""
struct SafeListAccessProvider{C} <: AbstractHeadProvider
    source_array_depth_limit::Int
    source_collectors::C
end
struct WeightTagProvider{T} <: AbstractHeadProvider
    tags_by_name::T
end
Base.:(==)(a::WeightTagProvider, b::WeightTagProvider) = a.tags_by_name == b.tags_by_name

SafeListAccessProvider(; source_array_depth_limit::Integer=0, source_collectors=()) =
    SafeListAccessProvider(
        _checked_nonnegative_int(:source_array_depth_limit, source_array_depth_limit),
        _source_collector_tuple(source_collectors),
    )

_source_collector_tuple(collectors::Tuple) = collectors
_source_collector_tuple(collectors::AbstractVector) = Tuple(collectors)
_source_collector_tuple(collector) = (collector,)
TypedValueFactoryProvider() = TypedValueFactoryProvider(ValueFactories.default_value_factories())
TypedValueFactoryProvider(factory::ValueFactories.AbstractValueFactory) =
    TypedValueFactoryProvider((factory,))
TypedValueFactoryProvider(factories::Tuple) =
    TypedValueFactoryProvider{typeof(factories)}(factories)
TypedValueFactoryProvider(factories::AbstractVector) =
    TypedValueFactoryProvider(Tuple(factories))
WeightTagProvider() = WeightTagProvider{Dict{Symbol,Tuple{Vararg{Symbol}}}}(Dict{Symbol,Tuple{Vararg{Symbol}}}())
function WeightTagProvider(tags::AbstractDict)
    normalized = Dict{Symbol,Tuple{Vararg{Symbol}}}(
        Symbol(name) => _weight_tag_tuple(tags_value) for (name, tags_value) in pairs(tags)
    )
    return WeightTagProvider{typeof(normalized)}(normalized)
end
WeightTagProvider(pairs::Pair...) = WeightTagProvider(Dict(pairs))

const _DEFAULT_HEAD_PROVIDERS = (
    ContextHeadProvider(),
    NilFactoryProvider(),
    ToListProvider(),
    AMapProvider(),
    GuardedExpansionProvider(),
)

default_head_providers() = _DEFAULT_HEAD_PROVIDERS

function test_catalog_consts()
    return Dict{Symbol,Any}(
        :FSIN => TypeSystem.FSIN,
        :FSQRT => TypeSystem.FSQRT,
        :SQR => TypeSystem.SQR,
        :FADD => TypeSystem.FADD,
        :FSUB => TypeSystem.FSUB,
        :FMUL => TypeSystem.FMUL,
        :FDIV => TypeSystem.FDIV,
        :FEQ => TypeSystem.FEQ,
        :FLE => TypeSystem.FLE,
        :LENF => TypeSystem.LENF,
    )
end

function test_catalog_weight_tags()
    return Dict{Symbol,Tuple{Vararg{Symbol}}}(
        :FSIN => (:real_unary,),
        :FSQRT => (:real_unary,),
        :SQR => (:real_unary,),
        :FADD => (:real_binary, :multi_arg),
        :FSUB => (:real_binary, :multi_arg),
        :FMUL => (:real_binary, :multi_arg),
        :FDIV => (:real_binary, :multi_arg),
        :FEQ => (:real_compare, :multi_arg),
        :FLE => (:real_compare, :multi_arg),
        :COND => (:cond, :multi_arg),
        :AFOLDL => (:afoldl, :multi_arg),
    )
end

function test_catalog_providers(;
    value_factories=ValueFactories.default_value_factories(),
    weight_tags=test_catalog_weight_tags(),
)
    return (
        ContextHeadProvider(),
        TypedValueFactoryProvider(value_factories),
        ToListProvider(),
        AMapProvider(),
        GuardedExpansionProvider(),
        SafeListAccessProvider(),
        WeightTagProvider(weight_tags),
    )
end

function test_catalog_weights(; profile::Symbol=:balanced)
    profile == :balanced && return GenerationWeights()
    profile == :deeper && return GenerationWeights(;
        variable=0.6,
        constant=0.25,
        instance_factory=0.25,
        tolist=1.0,
        amap=1.5,
        array_source=1.0,
        guarded_expansion=2.0,
        lambda=1.8,
        real_unary=1.0,
        real_binary=3.0,
        real_compare=2.0,
        cond=2.0,
        afoldl=1.5,
        safe_list_access=1.5,
        multi_arg=1.5,
    )
    throw(ArgumentError("unknown test catalog weight profile: $profile"))
end

function test_catalog_generation_kwargs(; profile::Symbol=:balanced)
    if profile == :deeper
        return (;
            guarded_expansion_depth=4,
            sampling_depth_limit=16,
            sampling_node_budget=256,
            providers=test_catalog_providers(),
            weights=test_catalog_weights(; profile),
        )
    elseif profile == :balanced
        return (;
            guarded_expansion_depth=2,
            providers=test_catalog_providers(),
            weights=test_catalog_weights(; profile),
        )
    end
    throw(ArgumentError("unknown test catalog generation profile: $profile"))
end

function add_heads!(provider::AbstractHeadProvider, heads, universe; kwargs...)
    error("canonical head provider $(typeof(provider)) does not implement add_heads!")
end

function add_universe_types!(
    provider::AbstractHeadProvider,
    types,
    keys,
    universe;
    kwargs...,
)
    return types
end

function add_universe_type!(types::Vector{Ty}, keys::Set{Any}, ty::TyLike)
    _collect_closed_types!(types, keys, ty)
    return types
end

function add_universe_types!(::AMapProvider, types, keys, universe; catalog_target_ty=nothing, kwargs...)
    arrays = _closed_array_types(universe)
    if catalog_target_ty isa TyArray
        for source_ty in arrays
            source_ty.len == catalog_target_ty.len || continue
            add_universe_type!(types, keys, catalog_target_ty)
            add_universe_type!(types, keys, TyFun(source_ty.elem, catalog_target_ty.elem))
        end
        return types
    elseif catalog_target_ty isa TyList
        for source_ty in arrays
            result_ty = TyArray(source_ty.len, catalog_target_ty.elem)
            add_universe_type!(types, keys, result_ty)
            add_universe_type!(types, keys, TyFun(source_ty.elem, catalog_target_ty.elem))
        end
        return types
    elseif catalog_target_ty isa Ty && !_is_potential_amap_type(catalog_target_ty)
        return types
    end

    elem_targets = _nonfunction_closed_types(universe)
    for source_ty in arrays
        for elem_ty in elem_targets
            add_universe_type!(types, keys, TyArray(source_ty.len, elem_ty))
            add_universe_type!(types, keys, TyFun(source_ty.elem, elem_ty))
        end
    end
    return types
end

_is_potential_amap_type(::Ty) = false
_is_potential_amap_type(ty::TyFun) =
    ty.arg isa TyFun && ty.res isa TyFun && ty.res.arg isa TyArray && ty.res.res isa TyArray

function add_universe_types!(::GuardedExpansionProvider, types, keys, universe; kwargs...)
    add_universe_type!(types, keys, BoolTy)
    snapshot = copy(universe.types)
    arrays = _closed_array_types(universe)
    for ty in snapshot
        add_universe_type!(types, keys, TyFun(BoolTy, TyFun(ty, TyFun(ty, ty))))
    end
    for result_ty in snapshot
        for array_ty in arrays
            add_universe_type!(types, keys, TyFun(result_ty, TyFun(array_ty.elem, result_ty)))
        end
    end
    return types
end

HeadEntry(
    name::Symbol,
    source::Symbol,
    ty::TyLike,
    rule_class::CoreRuleClass,
    weight_bucket::CoreWeightBucket;
    metadata::NamedTuple=(;),
) = HeadEntry(name, source, ty, rule_class, weight_bucket, metadata)

function CoreUniverse(
    target_ty::TyLike;
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    scope_source_tys::AbstractVector{<:Ty}=Ty[],
)
    types = Ty[]
    keys = Set{Any}()
    _collect_closed_types!(types, keys, target_ty)
    _collect_ctx_types!(types, keys, global_consts_ty)
    _collect_ctx_types!(types, keys, local_consts_ty)
    for ty in scope_source_tys
        _collect_closed_types!(types, keys, ty)
    end
    for ty in values(_typed_presets(presets, preset_types; global_consts_ty, local_consts_ty))
        _collect_closed_types!(types, keys, ty)
    end
    return CoreUniverse(types, keys, sort!(collect(_collect_lengths(types))))
end

_in_closed_fragment(::TyConst) = true
_in_closed_fragment(::TyVar) = false
_in_closed_fragment(::TyProd) = false
_in_closed_fragment(::TySum) = false
_in_closed_fragment(::TyMu) = false
_in_closed_fragment(::TyNat) = false
_in_closed_fragment(::TyScheme) = false
_in_closed_fragment(ty::TyFun) = _in_closed_fragment(ty.arg) && _in_closed_fragment(ty.res)
_in_closed_fragment(ty::TyList) = _in_closed_fragment(ty.elem)
_in_closed_fragment(ty::TyArray) = ty.len isa TyNat && _in_closed_fragment(ty.elem)

const _RIGID_DUMMY_PREFIX = "##canonical_rigid#"

_rigid_dummy_name(scheme_key, slot::Int) = Symbol(_RIGID_DUMMY_PREFIX, repr(scheme_key), "#", slot)
_is_rigid_dummy_type(::Ty) = false
_is_rigid_dummy_type(ty::TyConst) = startswith(String(ty.name), _RIGID_DUMMY_PREFIX)

function _contains_rigid_dummy(ty::TyLike)
    if ty isa TyScheme
        return _contains_rigid_dummy(ty.ty)
    elseif ty isa TyConst
        return _is_rigid_dummy_type(ty)
    elseif ty isa TyFun
        return _contains_rigid_dummy(ty.arg) || _contains_rigid_dummy(ty.res)
    elseif ty isa TyProd
        return _contains_rigid_dummy(ty.left) || _contains_rigid_dummy(ty.right)
    elseif ty isa TySum
        return _contains_rigid_dummy(ty.left) || _contains_rigid_dummy(ty.right)
    elseif ty isa TyList
        return _contains_rigid_dummy(ty.elem)
    elseif ty isa TyArray
        return _contains_rigid_dummy(ty.len) || _contains_rigid_dummy(ty.elem)
    elseif ty isa TyMu
        return _contains_rigid_dummy(ty.body)
    else
        return false
    end
end

function _rigidify_scheme_demand(scheme::TyScheme)
    bound_ids = Set(var.id for var in scheme.vars)
    type_ids = Set{Int}()
    length_ids = Set{Int}()
    _mark_scheme_var_positions!(scheme.ty, bound_ids, type_ids, length_ids, :type)
    isempty(length_ids) || return nothing

    scheme_key = typekey(scheme)
    assignment = Dict{Int,Ty}()
    discharge_vars = Dict{Symbol,TyVar}()
    rigid_slots = Dict{Symbol,Int}()
    for (slot, var) in enumerate(scheme.vars)
        atom = TyConst(_rigid_dummy_name(scheme_key, slot))
        assignment[var.id] = atom
        discharge_vars[atom.name] = TyVar(var.id, var.level, nothing)
        rigid_slots[atom.name] = slot
    end

    rigid_target = _subst_scheme_ty(scheme.ty, assignment)
    _in_closed_fragment(rigid_target) || return nothing
    return (target=rigid_target, discharge_vars=discharge_vars, rigid_slots=rigid_slots)
end

function _discharge_rigid_result(result, scheme::TyScheme, demand)
    local_const_types = Dict{Symbol,TyLike}(
        name => _discharge_rigid_tylike(ty, demand.discharge_vars)
        for (name, ty) in pairs(result.local_const_types)
    )
    metadata = Dict{Symbol,Any}(
        name => _discharge_rigid_metadata_value(value, demand.rigid_slots)
        for (name, value) in pairs(result.metadata)
    )
    metadata[:rigid_scheme] = true
    metadata[:target_scheme] = typekey(scheme)
    metadata[:rigid_atom_count] = length(demand.rigid_slots)
    return GenerationResult(result.tree, scheme, result.local_consts, local_const_types, metadata)
end

_discharge_rigid_tylike(ty::Ty, vars_by_name::AbstractDict{Symbol,TyVar}) =
    _discharge_rigid_type(ty, vars_by_name)
_discharge_rigid_tylike(scheme::TyScheme, vars_by_name::AbstractDict{Symbol,TyVar}) =
    TyScheme(scheme.vars, _discharge_rigid_type(scheme.ty, vars_by_name))

function _discharge_rigid_type(ty::Ty, vars_by_name::AbstractDict{Symbol,TyVar})
    if ty isa TyConst
        return get(vars_by_name, ty.name, ty)
    elseif ty isa TyFun
        return TyFun(_discharge_rigid_type(ty.arg, vars_by_name), _discharge_rigid_type(ty.res, vars_by_name))
    elseif ty isa TyProd
        return TyProd(_discharge_rigid_type(ty.left, vars_by_name), _discharge_rigid_type(ty.right, vars_by_name))
    elseif ty isa TySum
        return TySum(_discharge_rigid_type(ty.left, vars_by_name), _discharge_rigid_type(ty.right, vars_by_name))
    elseif ty isa TyList
        return TyList(_discharge_rigid_type(ty.elem, vars_by_name))
    elseif ty isa TyArray
        return TyArray(_discharge_rigid_type(ty.len, vars_by_name), _discharge_rigid_type(ty.elem, vars_by_name))
    elseif ty isa TyMu
        return TyMu(ty.var, _discharge_rigid_type(ty.body, vars_by_name))
    else
        return ty
    end
end

function _discharge_rigid_metadata_value(value, rigid_slots::AbstractDict{Symbol,Int})
    if value isa Tuple
        if length(value) == 2 && value[1] == :const && value[2] isa Symbol && haskey(rigid_slots, value[2])
            return (:rigid, rigid_slots[value[2]])
        end
        return Tuple(_discharge_rigid_metadata_value(item, rigid_slots) for item in value)
    elseif value isa AbstractVector
        return [_discharge_rigid_metadata_value(item, rigid_slots) for item in value]
    elseif value isa AbstractDict
        return Dict(k => _discharge_rigid_metadata_value(v, rigid_slots) for (k, v) in pairs(value))
    else
        return value
    end
end

function _rigid_head_allowed(head::HeadEntry)
    _contains_rigid_dummy(head.type) || return true
    _metadata_get(head.metadata, :scheme_instance, false) === true && return true
    _metadata_get(head.metadata, :guarded, false) === true && return true
    _is_builtin_tolist_head(head) && return true
    _is_builtin_amap_head(head) && return true
    return _is_builtin_nil_factory_head(head)
end

function _filter_rigid_heads!(heads::Vector{HeadEntry})
    filter!(_rigid_head_allowed, heads)
    return heads
end

function _is_builtin_tolist_head(head::HeadEntry)
    head.name == :TOLIST || return false
    head.source == :primitive || return false
    _metadata_get(head.metadata, :value, nothing) === TypeSystem.TOLIST || return false
    head.type isa TyFun || return false
    source_ty = head.type.arg
    result_ty = head.type.res
    source_ty isa TyArray || return false
    result_ty isa TyList || return false
    return typekey(source_ty.elem) == typekey(result_ty.elem)
end

function _is_builtin_amap_head(head::HeadEntry)
    head.name == :AMAP || return false
    head.source == :primitive || return false
    _metadata_get(head.metadata, :value, nothing) === TypeSystem.AMAP || return false
    _metadata_get(head.metadata, :source_array_constructor, false) === true || return false
    head.type isa Ty || return false
    args, result_ty = _split_fun(head.type)
    length(args) == 2 || return false
    step_ty, source_ty = args
    step_ty isa TyFun || return false
    source_ty isa TyArray || return false
    result_ty isa TyArray || return false
    source_ty.len == result_ty.len || return false
    typekey(step_ty.arg) == typekey(source_ty.elem) || return false
    return typekey(step_ty.res) == typekey(result_ty.elem)
end

function _is_builtin_nil_factory_head(head::HeadEntry)
    head.name == :NIL || return false
    head.source == :instance_factory || return false
    head.type isa TyList || return false
    return _metadata_get(head.metadata, :nil_factory, false) === true
end

function _collect_closed_types(tys::TyLike...)
    types = Ty[]
    keys = Set{Any}()
    for ty in tys
        _collect_closed_types!(types, keys, ty)
    end
    return types
end

function _collect_closed_types!(types::Vector{Ty}, keys::Set{Any}, ty::TyLike)
    ty isa Ty || return types
    _in_closed_fragment(ty) || return types
    key = typekey(ty)
    if key ∉ keys
        push!(keys, key)
        push!(types, ty)
    end
    if ty isa TyFun
        _collect_closed_types!(types, keys, ty.arg)
        _collect_closed_types!(types, keys, ty.res)
    elseif ty isa TyList
        _collect_closed_types!(types, keys, ty.elem)
    elseif ty isa TyArray
        _collect_closed_types!(types, keys, ty.elem)
    end
    return types
end

function _collect_ctx_types!(types::Vector{Ty}, keys::Set{Any}, ctx::AbstractDict{Symbol,<:Any})
    for ty in values(ctx)
        ty isa TyLike && _collect_closed_types!(types, keys, ty)
    end
    return types
end

function _collect_lengths(types::Vector{Ty})
    lengths = Set{Int}()
    for ty in types
        _collect_lengths!(lengths, ty)
    end
    return lengths
end

function _collect_lengths!(lengths::Set{Int}, ty::Ty)
    if ty isa TyFun
        _collect_lengths!(lengths, ty.arg)
        _collect_lengths!(lengths, ty.res)
    elseif ty isa TyList
        _collect_lengths!(lengths, ty.elem)
    elseif ty isa TyArray
        ty.len isa TyNat && push!(lengths, ty.len.value)
        _collect_lengths!(lengths, ty.elem)
    end
    return lengths
end

_type_in_universe(universe::CoreUniverse, ty::Ty) = typekey(ty) in universe.typekeys

function _tyctx_from(ctxs::AbstractDict{Symbol,<:Any}...)
    out = Dict{Symbol,TyLike}()
    for ctx in ctxs
        for (name, ty) in pairs(ctx)
            ty isa TyLike && (out[name] = ty)
        end
    end
    return out
end

function _anydict(ctx::AbstractDict{Symbol,<:Any})
    return Dict{Symbol,Any}(name => value for (name, value) in pairs(ctx))
end

function _typed_presets(
    presets::AbstractDict{Symbol,<:Any},
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}();
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
)
    isempty(presets) && isempty(preset_types) && return Dict{Symbol,TyLike}()
    tyctx = _tyctx_from(global_consts_ty, local_consts_ty, preset_types)
    preset_ctx = _anydict(presets)
    get_type_ctx(preset_ctx, tyctx)
    get_ast_type_ctx(preset_ctx, tyctx)
    return Dict{Symbol,TyLike}(name => tyctx[name] for name in keys(presets) if haskey(tyctx, name))
end

function _typed_context(
    values::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    explicit_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
)
    tyctx = _tyctx_from(explicit_types)
    get_type_ctx(_anydict(values), tyctx)
    return tyctx
end

function _build_core_catalog(
    universe::CoreUniverse;
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    scope_source_tys::AbstractVector{<:Ty}=Ty[],
    providers::Tuple=default_head_providers(),
    catalog_target_ty=nothing,
)
    providers = _provider_tuple(providers)
    universe = _extend_core_universe(
        universe,
        providers;
        global_consts,
        global_consts_ty,
        local_consts,
        local_consts_ty,
        presets,
        preset_types,
        scope_source_tys,
        catalog_target_ty,
    )
    heads = HeadEntry[]
    for provider in providers
        add_heads!(
            provider,
            heads,
            universe;
            global_consts,
            global_consts_ty,
            local_consts,
            local_consts_ty,
            presets,
            preset_types,
            scope_source_tys,
            catalog_target_ty,
        )
    end
    _filter_rigid_heads!(heads)
    rules = _core_rules(heads, universe)
    return CoreCatalog(universe, heads, rules)
end

function _extend_core_universe(
    universe::CoreUniverse,
    providers::Tuple;
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    scope_source_tys::AbstractVector{<:Ty}=Ty[],
    catalog_target_ty=nothing,
)
    types = copy(universe.types)
    keys = copy(universe.typekeys)
    _collect_ctx_types!(types, keys, _typed_context(global_consts, global_consts_ty))
    _collect_ctx_types!(types, keys, _typed_context(local_consts, local_consts_ty))
    for ty in scope_source_tys
        _collect_closed_types!(types, keys, ty)
    end
    for provider in providers
        add_universe_types!(
            provider,
            types,
            keys,
            CoreUniverse(types, keys, sort!(collect(_collect_lengths(types))));
            global_consts,
            global_consts_ty,
            local_consts,
            local_consts_ty,
            presets,
            preset_types,
            scope_source_tys,
            catalog_target_ty,
        )
    end
    return CoreUniverse(types, keys, sort!(collect(_collect_lengths(types))))
end

function add_heads!(
    ::ContextHeadProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    kwargs...,
)
    global_tyctx = _typed_context(global_consts, global_consts_ty)
    local_tyctx = _typed_context(local_consts, local_consts_ty)
    _add_typed_heads!(heads, :variable, local_tyctx, _VARIABLE_WEIGHT, universe)
    _add_typed_heads!(heads, :constant, global_tyctx, _CONSTANT_WEIGHT, universe)
    _add_typed_heads!(
        heads,
        :preset,
        _typed_presets(presets, preset_types; global_consts_ty=global_tyctx, local_consts_ty=local_tyctx),
        _PRESET_TERM_WEIGHT,
        universe,
    )
    return heads
end

add_heads!(::NilFactoryProvider, heads::Vector{HeadEntry}, universe::CoreUniverse; kwargs...) =
    _add_value_factory_heads!(heads, universe, (ValueFactories.NilFactory(),))

function add_heads!(
    provider::TypedValueFactoryProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    kwargs...,
)
    return _add_value_factory_heads!(heads, universe, provider.factories)
end

function add_heads!(
    ::ToListProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    kwargs...,
)
    return _add_tolist_heads!(heads, universe, _closed_array_types(universe))
end

function add_heads!(
    ::AMapProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    catalog_target_ty=nothing,
    kwargs...,
)
    return _add_amap_heads!(heads, universe, _closed_array_types(universe), catalog_target_ty)
end

function add_heads!(
    ::GuardedExpansionProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    kwargs...,
)
    _add_cons_heads!(heads, universe)
    _add_cond_heads!(heads, universe)
    _add_afoldl_heads!(heads, universe)
    return heads
end

add_heads!(::SafeListAccessProvider, heads::Vector{HeadEntry}, universe::CoreUniverse; kwargs...) = heads

function add_heads!(
    provider::WeightTagProvider,
    heads::Vector{HeadEntry},
    universe::CoreUniverse;
    kwargs...,
)
    for i in eachindex(heads)
        tags = get(provider.tags_by_name, heads[i].name, ())
        isempty(tags) && continue
        heads[i] = _with_head_weight_tags(heads[i], tags)
    end
    return heads
end

function _add_typed_heads!(
    heads::Vector{HeadEntry},
    source::Symbol,
    tyctx::AbstractDict{Symbol,<:Any},
    weight_bucket::CoreWeightBucket,
    universe::CoreUniverse,
)
    for name in sort!(collect(keys(tyctx)))
        ty = tyctx[name]
        ty isa TyLike || continue
        if ty isa Ty
            _in_closed_fragment(ty) || continue
            push!(heads, HeadEntry(name, source, ty, _rule_class_for_tylike(ty), weight_bucket))
        elseif ty isa TyScheme
            _add_scheme_heads!(heads, source, name, ty, weight_bucket, universe)
        end
    end
    return heads
end

_rule_class_for_tylike(ty::TyLike) = ty isa Ty && _in_closed_fragment(ty) ? CoreProductive() : Decoration()

function _add_scheme_heads!(
    heads::Vector{HeadEntry},
    source::Symbol,
    name::Symbol,
    scheme::TyScheme,
    weight_bucket::CoreWeightBucket,
    universe::CoreUniverse,
)
    for ty in _finite_scheme_instances(scheme, universe)
        push!(
            heads,
            HeadEntry(
                name,
                source,
                ty,
                _rule_class_for_tylike(ty),
                weight_bucket;
                metadata=(scheme_instance=true,),
            ),
        )
    end
    return heads
end

function _finite_scheme_instances(scheme::TyScheme, universe::CoreUniverse)
    isempty(scheme.vars) && return _in_closed_fragment(scheme.ty) ? Ty[scheme.ty] : Ty[]
    # Default scheme specialization ranges over closed value/container shapes.
    # Higher-order polymorphic providers can expose a tighter custom provider
    # instead of forcing every finite function type into this hot path.
    domains = _scheme_instantiation_domains(scheme, universe)
    any(isempty, domains) && return Ty[]
    instances = Ty[]
    seen = Set{Any}()
    for assignment_values in _choice_products(domains)
        assignment = Dict{Int,Ty}(var.id => value for (var, value) in zip(scheme.vars, assignment_values))
        ty = _subst_scheme_ty(scheme.ty, assignment)
        _in_closed_fragment(ty) || continue
        key = typekey(ty)
        key in seen && continue
        push!(seen, key)
        push!(instances, ty)
    end
    return instances
end

function _scheme_instantiation_domains(scheme::TyScheme, universe::CoreUniverse)
    bound_ids = Set(var.id for var in scheme.vars)
    type_ids = Set{Int}()
    length_ids = Set{Int}()
    _mark_scheme_var_positions!(scheme.ty, bound_ids, type_ids, length_ids, :type)
    type_choices = Ty[_nonfunction_closed_types(universe)...]
    length_choices = Ty[TyNat(n) for n in universe.lengths]
    return [
        _scheme_var_domain(var, type_ids, length_ids, type_choices, length_choices)
        for var in scheme.vars
    ]
end

function _mark_scheme_var_positions!(
    ty::Ty,
    bound_ids::Set{Int},
    type_ids::Set{Int},
    length_ids::Set{Int},
    position::Symbol,
)
    if ty isa TyVar
        ty.id in bound_ids || return nothing
        push!(position == :length ? length_ids : type_ids, ty.id)
    elseif ty isa TyFun
        _mark_scheme_var_positions!(ty.arg, bound_ids, type_ids, length_ids, :type)
        _mark_scheme_var_positions!(ty.res, bound_ids, type_ids, length_ids, :type)
    elseif ty isa TyProd
        _mark_scheme_var_positions!(ty.left, bound_ids, type_ids, length_ids, :type)
        _mark_scheme_var_positions!(ty.right, bound_ids, type_ids, length_ids, :type)
    elseif ty isa TySum
        _mark_scheme_var_positions!(ty.left, bound_ids, type_ids, length_ids, :type)
        _mark_scheme_var_positions!(ty.right, bound_ids, type_ids, length_ids, :type)
    elseif ty isa TyList
        _mark_scheme_var_positions!(ty.elem, bound_ids, type_ids, length_ids, :type)
    elseif ty isa TyArray
        _mark_scheme_var_positions!(ty.len, bound_ids, type_ids, length_ids, :length)
        _mark_scheme_var_positions!(ty.elem, bound_ids, type_ids, length_ids, :type)
    elseif ty isa TyMu
        _mark_scheme_var_positions!(ty.body, bound_ids, type_ids, length_ids, :type)
    end
    return nothing
end

function _scheme_var_domain(
    var::TyVar,
    type_ids::Set{Int},
    length_ids::Set{Int},
    type_choices::AbstractVector{<:Ty},
    length_choices::AbstractVector{<:Ty},
)
    in_type = var.id in type_ids
    in_length = var.id in length_ids
    in_length && !in_type && return length_choices
    in_type && !in_length && return type_choices
    !in_type && !in_length && return type_choices
    return Ty[]
end

function _choice_products(domains::Vector{Vector{Ty}})
    isempty(domains) && return [Ty[]]
    choices = first(domains)
    tails = _choice_products(domains[2:end])
    out = Vector{Vector{Ty}}()
    for choice in choices
        for tail in tails
            push!(out, Ty[choice; tail])
        end
    end
    return out
end

function _subst_scheme_ty(ty::Ty, assignment::AbstractDict{Int,<:Ty})
    if ty isa TyVar
        return get(assignment, ty.id, ty)
    elseif ty isa TyConst || ty isa TyNat
        return ty
    elseif ty isa TyFun
        return _subst_scheme_ty(ty.arg, assignment) → _subst_scheme_ty(ty.res, assignment)
    elseif ty isa TyProd
        return TyProd(_subst_scheme_ty(ty.left, assignment), _subst_scheme_ty(ty.right, assignment))
    elseif ty isa TySum
        return TySum(_subst_scheme_ty(ty.left, assignment), _subst_scheme_ty(ty.right, assignment))
    elseif ty isa TyMu
        return TyMu(ty.var, _subst_scheme_ty(ty.body, assignment))
    elseif ty isa TyList
        return TyList(_subst_scheme_ty(ty.elem, assignment))
    elseif ty isa TyArray
        return TyArray(_subst_scheme_ty(ty.len, assignment), _subst_scheme_ty(ty.elem, assignment))
    else
        return ty
    end
end

function _add_value_factory_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, factories)
    for factory in factories
        factory isa ValueFactories.AbstractValueFactory || continue
        for ty in ValueFactories.provided_types(factory, universe.types)
            _value_factory_allowed_for_type(factory, ty) || continue
            push!(
                heads,
                HeadEntry(
                    ValueFactories.factory_name(factory),
                    :instance_factory,
                    ty,
                    CoreProductive(),
                    _INSTANCE_FACTORY_WEIGHT;
                    metadata=(factory=_value_factory_callback(factory), nil_factory=factory isa ValueFactories.NilFactory),
                ),
            )
        end
    end
    return heads
end

function _value_factory_allowed_for_type(factory::ValueFactories.AbstractValueFactory, ty::Ty)
    _contains_rigid_dummy(ty) || return true
    return factory isa ValueFactories.NilFactory && ty isa TyList
end

_value_factory_callback(factory::ValueFactories.AbstractValueFactory) =
    (head, relation, rng) -> ValueFactories.make_value(factory, head.type, rng)

function _add_tolist_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, source_arrays::Vector{TyArray})
    for array_ty in source_arrays
        list_ty = TyList(array_ty.elem)
        _type_in_universe(universe, list_ty) || continue
        push!(
            heads,
            HeadEntry(
                :TOLIST,
                :primitive,
                TyFun(array_ty, list_ty),
                CoreProductive(),
                _TOLIST_WEIGHT;
                metadata=(value=TypeSystem.TOLIST, source_array_depth_limit=1),
            ),
        )
    end
    return heads
end

function _add_amap_heads!(
    heads::Vector{HeadEntry},
    universe::CoreUniverse,
    source_arrays::Vector{TyArray},
    catalog_target_ty=nothing,
)
    arrays = _closed_array_types(universe)
    for source_ty in source_arrays
        for result_ty in arrays
            source_ty.len == result_ty.len || continue
            fun_arg = source_ty.elem → result_ty.elem
            amap_ty = fun_arg → source_ty → result_ty
            _amap_head_relevant(amap_ty, result_ty, catalog_target_ty) || continue
            push!(
                heads,
                HeadEntry(
                    :AMAP,
                    :primitive,
                    amap_ty,
                    CoreProductive(),
                    _AMAP_WEIGHT;
                    metadata=(
                        value=TypeSystem.AMAP,
                        source_array_constructor=true,
                        source_array_argument_index=2,
                        source_array_depth_limit=0,
                    ),
                ),
            )
        end
    end
    return heads
end

_amap_head_relevant(::Ty, ::TyArray, ::Nothing) = true
_amap_head_relevant(amap_ty::Ty, result_ty::TyArray, target_ty) =
    target_ty isa Ty && (iscompatible(result_ty, target_ty) || iscompatible(amap_ty, target_ty))

_closed_array_types(universe::CoreUniverse) = [ty for ty in universe.types if ty isa TyArray]
_nonfunction_closed_types(universe::CoreUniverse) = [ty for ty in universe.types if !(ty isa TyFun)]

function _add_cons_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse)
    for list_ty in universe.types
        list_ty isa TyList || continue
        cons_ty = TyFun(list_ty.elem, TyFun(list_ty, list_ty))
        push!(
            heads,
            HeadEntry(
                :CONS,
                :guarded_primitive,
                cons_ty,
                Decoration(),
                _GUARDED_EXPANSION_WEIGHT;
                metadata=(value=TypeSystem.CONS, guarded=true),
            ),
        )
    end
    return heads
end

function _add_cond_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse)
    _type_in_universe(universe, BoolTy) || return heads
    for result_ty in universe.types
        cond_ty = TyFun(BoolTy, TyFun(result_ty, TyFun(result_ty, result_ty)))
        push!(
            heads,
            HeadEntry(
                :COND,
                :guarded_primitive,
                cond_ty,
                Decoration(),
                _GUARDED_EXPANSION_WEIGHT;
                metadata=(value=TypeSystem.COND, guarded=true),
            ),
        )
    end
    return heads
end

function _add_afoldl_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse)
    arrays = _closed_array_types(universe)
    for result_ty in universe.types
        for array_ty in arrays
            step_ty = TyFun(result_ty, TyFun(array_ty.elem, result_ty))
            _type_in_universe(universe, step_ty) || continue
            afoldl_ty = TyFun(step_ty, TyFun(result_ty, TyFun(array_ty, result_ty)))
            push!(
                heads,
                HeadEntry(
                    :AFOLDL,
                    :guarded_primitive,
                    afoldl_ty,
                    Decoration(),
                    _GUARDED_EXPANSION_WEIGHT;
                    metadata=(value=TypeSystem.AFOLDL, guarded=true),
                ),
            )
        end
    end
    return heads
end

function _split_fun(ty::Ty)
    args = Ty[]
    rest = ty
    while rest isa TyFun
        push!(args, rest.arg)
        rest = rest.res
    end
    return args, rest
end

function _scope_application_rules(target_ty::Ty, scope_tys::Vector{Ty})
    rules = ScopeApplicationRule[]
    for (idx0, scope_ty) in enumerate(scope_tys)
        scope_ty isa TyFun || continue
        _in_closed_fragment(scope_ty) || continue
        args = Ty[]
        rest = scope_ty
        while rest isa TyFun
            push!(args, rest.arg)
            rest = rest.res
            iscompatible(rest, target_ty) || continue
            argument_tys = copy(args)
            push!(
                rules,
                ScopeApplicationRule(
                    idx0 - 1,
                    scope_ty,
                    rest,
                    argument_tys,
                    [_argument_class(arg) for arg in argument_tys],
                ),
            )
        end
    end
    return rules
end

function _core_rules(heads::Vector{HeadEntry}, universe::CoreUniverse)
    rules = CoreRule[]
    for head in heads
        head.rule_class isa CoreProductive || continue
        head.type isa Ty || continue
        args, result = _split_fun(head.type)
        _type_in_universe(universe, result) || continue
        all(arg -> _type_in_universe(universe, arg), args) || continue
        push!(rules, CoreRule(head, result, args, [_argument_class(arg) for arg in args]))
    end
    return rules
end

function _argument_class(ty::Ty)
    ty isa TyArray && return :source_array
    ty isa TyFun && return :function
    return :closed_recursive
end

"""
    GenerationWeights(; kwargs...)

Symbol-keyed nonnegative weights for canonical generation. Zero disables that
bucket during derivation and sampling without changing the catalog. Unknown
bucket keys use `default`.
"""
struct GenerationWeights
    default::Float64
    overrides::Dict{Symbol,Float64}
end

function GenerationWeights(;
    default::Real=1,
    overrides::AbstractDict{Symbol,<:Real}=Dict{Symbol,Float64}(),
    kwargs...,
)
    values = Dict{Symbol,Float64}()
    for (name, value) in pairs(overrides)
        values[name] = _checked_weight(name, value)
    end
    for (name, value) in pairs(kwargs)
        values[name] = _checked_weight(name, value)
    end
    return GenerationWeights(_checked_weight(:default, default), values)
end

function _checked_weight(name::Symbol, value::Real)
    weight = Float64(value)
    isfinite(weight) && weight >= 0 || throw(ArgumentError("generation weight $name must be finite and nonnegative"))
    return weight
end

function _checked_nonnegative_int(name::Symbol, value::Integer)
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return Int(value)
end

_bucket_key(::VariableWeight) = :variable
_bucket_key(::ConstantWeight) = :constant
_bucket_key(::PresetTermWeight) = :preset_term
_bucket_key(::InstanceFactoryWeight) = :instance_factory
_bucket_key(::ArraySourceWeight) = :array_source
_bucket_key(::ToListWeight) = :tolist
_bucket_key(::AMapWeight) = :amap
_bucket_key(::GuardedExpansionWeight) = :guarded_expansion
_bucket_key(bucket::NamedWeightBucket) = bucket.key
_bucket_key(bucket::CoreWeightBucket) = Symbol(nameof(typeof(bucket)))

_bucket_weight(weights::GenerationWeights, bucket::CoreWeightBucket) =
    get(weights.overrides, _bucket_key(bucket), weights.default)

_bucket_enabled(weights::GenerationWeights, bucket::CoreWeightBucket) = _bucket_weight(weights, bucket) > 0

_scope_application_weight(weights::GenerationWeights) =
    get(weights.overrides, :scope_application, _bucket_weight(weights, _VARIABLE_WEIGHT))

_scope_application_enabled(weights::GenerationWeights) = _scope_application_weight(weights) > 0

function _head_weight(weights::GenerationWeights, head::HeadEntry)
    weight = _bucket_weight(weights, head.weight_bucket)
    for tag in _metadata_get(head.metadata, :weight_tags, ())
        weight *= _tag_weight(weights, tag)
    end
    return weight
end

_tag_weight(weights::GenerationWeights, tag::Symbol) = get(weights.overrides, tag, weights.default)

"""
    GenerationConfig(;
        rng=Random.default_rng(),
        weights=GenerationWeights(),
        providers=default_head_providers(),
        argument_planner=DefaultArgumentPlanner(),
        sampler=WeightedDerivationSampler(),
        materializer=DefaultAstMaterializer(),
        guarded_expansion_depth=1,
        sampling_depth_limit=10,
        sampling_node_budget=128,
    )

Configuration for canonical generation. `providers` are normalized to a tuple.
The planner, sampler, and materializer are callable replacement points. The
default sampler builds one derivation with bounded recursive depth and node
fuel instead of materializing the whole proof frontier first.
"""
struct DefaultArgumentPlanner end
struct WeightedDerivationSampler end
struct DefaultAstMaterializer end

struct GenerationConfig{R<:Random.AbstractRNG,P,A,S,M}
    rng::R
    weights::GenerationWeights
    providers::P
    argument_planner::A
    sampler::S
    materializer::M
    guarded_expansion_depth::Int
    sampling_depth_limit::Int
    sampling_node_budget::Int
end
GenerationConfig(;
    rng::Random.AbstractRNG=Random.default_rng(),
    weights::GenerationWeights=GenerationWeights(),
    providers=default_head_providers(),
    argument_planner=DefaultArgumentPlanner(),
    sampler=WeightedDerivationSampler(),
    materializer=DefaultAstMaterializer(),
    guarded_expansion_depth::Integer=1,
    sampling_depth_limit::Integer=10,
    sampling_node_budget::Integer=128,
) = GenerationConfig(
    rng,
    weights,
    _provider_tuple(providers),
    argument_planner,
    sampler,
    materializer,
    _checked_nonnegative_int(:guarded_expansion_depth, guarded_expansion_depth),
    _checked_nonnegative_int(:sampling_depth_limit, sampling_depth_limit),
    _checked_nonnegative_int(:sampling_node_budget, sampling_node_budget),
)

_provider_tuple(providers::Tuple) = providers
_provider_tuple(providers) = Tuple(providers)

"""
    GenerationResult

Result placeholder for generated ASTs and their generation metadata.
"""
struct GenerationResult
    tree::AstTree
    target_ty::TyLike
    local_consts::Dict{Symbol,Any}
    local_const_types::Dict{Symbol,TyLike}
    metadata::Dict{Symbol,Any}
end

struct UnsupportedCanonicalGenerationTarget
    target_ty::TyLike
end

abstract type CoreDerivation end

struct VariableDerivation <: CoreDerivation
    index::Int
    ty::Ty
end

struct ConstantDerivation <: CoreDerivation
    head::HeadEntry
end

struct LambdaDerivation <: CoreDerivation
    arg_ty::Ty
    body::CoreDerivation
end

struct HeadApplicationDerivation <: CoreDerivation
    head::HeadEntry
    args::Vector{CoreDerivation}
end

struct ScopeApplicationRule
    index::Int
    head_ty::Ty
    result_ty::Ty
    argument_tys::Vector{Ty}
    argument_classes::Vector{Symbol}
end

struct ScopeApplicationDerivation <: CoreDerivation
    index::Int
    head_ty::Ty
    args::Vector{CoreDerivation}
end

_rigid_derivation_allowed(d::VariableDerivation) = true
_rigid_derivation_allowed(d::ConstantDerivation) = _rigid_head_allowed(d.head)
_rigid_derivation_allowed(d::LambdaDerivation) =
    !_contains_rigid_dummy(d.arg_ty) && _rigid_derivation_allowed(d.body)
_rigid_derivation_allowed(d::HeadApplicationDerivation) =
    _rigid_head_allowed(d.head) && all(_rigid_derivation_allowed, d.args)
_rigid_derivation_allowed(d::ScopeApplicationDerivation) =
    all(_rigid_derivation_allowed, d.args)
_rigid_derivation_allowed(::CoreDerivation) = false

struct DerivationSampleSet
    derivations::Vector{CoreDerivation}
    cumulative_weights::Vector{Float64}
    total_weight::Float64
end

mutable struct DerivationSamplingState{R<:Random.AbstractRNG}
    rng::R
    nodes_left::Int
end

mutable struct CoreRelation{A}
    derivations::Dict{Tuple{Any,Any},Vector{CoreDerivation}}
    signatures::Dict{Tuple{Any,Any},Set{Any}}
    visiting::Set{Tuple{Any,Any}}
    computed::Set{Tuple{Any,Any}}
    catalogs::Dict{Tuple{Any,Any},CoreCatalog}
    sampling_options::Dict{Any,Vector{NamedTuple}}
    source_derivations::Dict{Any,Vector{CoreDerivation}}
    list_access_sources::Dict{Any,Any}
    global_consts::Dict{Symbol,Any}
    local_consts::Dict{Symbol,Any}
    global_consts_ty::Dict{Symbol,TyLike}
    local_consts_ty::Dict{Symbol,TyLike}
    presets::Dict{Symbol,Any}
    preset_types::Dict{Symbol,Any}
    weights::GenerationWeights
    providers::Tuple
    argument_planner::A
    guarded_expansion_depth::Int
end

function CoreRelation(;
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    weights::GenerationWeights=GenerationWeights(),
    providers=default_head_providers(),
    argument_planner=DefaultArgumentPlanner(),
    guarded_expansion_depth::Integer=1,
)
    global_tyctx = _typed_context(global_consts, global_consts_ty)
    local_tyctx = _typed_context(local_consts, local_consts_ty)
    return CoreRelation(
        Dict{Tuple{Any,Any},Vector{CoreDerivation}}(),
        Dict{Tuple{Any,Any},Set{Any}}(),
        Set{Tuple{Any,Any}}(),
        Set{Tuple{Any,Any}}(),
        Dict{Tuple{Any,Any},CoreCatalog}(),
        Dict{Any,Vector{NamedTuple}}(),
        Dict{Any,Vector{CoreDerivation}}(),
        Dict{Any,Any}(),
        _anydict(global_consts),
        _anydict(local_consts),
        global_tyctx,
        local_tyctx,
        _anydict(presets),
        Dict{Symbol,Any}(name => ty for (name, ty) in pairs(preset_types)),
        weights,
        _provider_tuple(providers),
        argument_planner,
        _checked_nonnegative_int(:guarded_expansion_depth, guarded_expansion_depth),
    )
end

include("SafeListAccessGeneration.jl")

struct CoreRelationCacheEntry
    relation::CoreRelation
    sample_set::DerivationSampleSet
    known_count::Int
    scope_count::Int
end

struct LazyRelationCacheEntry
    relation::CoreRelation
    lock::ReentrantLock
end

struct ScopedLazyRelationCacheEntry
    relation::CoreRelation
    lock::ReentrantLock
end

const _DEFAULT_RELATION_CACHE = Dict{Any,CoreRelationCacheEntry}()
const _DEFAULT_RELATION_CACHE_LOCK = ReentrantLock()
const _DEFAULT_LAZY_RELATION_CACHE = Dict{Any,LazyRelationCacheEntry}()
const _DEFAULT_LAZY_RELATION_CACHE_LOCK = ReentrantLock()
const _SCOPED_LAZY_RELATION_CACHE = Dict{Any,ScopedLazyRelationCacheEntry}()
const _SCOPED_LAZY_RELATION_CACHE_LOCK = ReentrantLock()

"""
    generate_canonical(target_ty; kwargs...)

Generate a beta-normal witness inside the current canonical finite core.
When `config` is passed, its `rng`, `weights`, `providers`,
`argument_planner`, `sampler`, and `materializer` are used; same-named outer
keywords are only defaults for constructing the implicit config.
"""
function generate_canonical(
    target_ty::TyLike;
    rng::Random.AbstractRNG=Random.default_rng(),
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    weights::GenerationWeights=GenerationWeights(),
    providers=default_head_providers(),
    guarded_expansion_depth::Integer=1,
    sampling_depth_limit::Integer=10,
    sampling_node_budget::Integer=128,
    config::GenerationConfig=GenerationConfig(; rng, weights, providers, guarded_expansion_depth, sampling_depth_limit, sampling_node_budget),
    _cache_relation::Bool=true,
)
    if target_ty isa TyScheme
        demand = _rigidify_scheme_demand(target_ty)
        demand === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
        result = generate_canonical(
            demand.target;
            rng,
            global_consts,
            global_consts_ty,
            local_consts,
            local_consts_ty,
            presets,
            preset_types,
            weights,
            providers,
            guarded_expansion_depth,
            sampling_depth_limit,
            sampling_node_budget,
            config,
            _cache_relation,
        )
        result isa GenerationResult || return result
        return _discharge_rigid_result(result, target_ty, demand)
    end
    target_ty isa Ty || return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    _in_closed_fragment(target_ty) || return Err(UnsupportedCanonicalGenerationTarget(target_ty))

    result_local_consts = _anydict(local_consts)
    result_local_const_types = _typed_context(local_consts, local_consts_ty)
    cache_key = (_cache_relation && config.sampler isa WeightedDerivationSampler) ? _default_relation_cache_key(
        target_ty,
        config.weights,
        global_consts,
        global_consts_ty,
        local_consts,
        local_consts_ty,
        presets,
        preset_types,
        config.providers,
        config.argument_planner,
        config.guarded_expansion_depth,
    ) : nothing
    if _uses_lazy_sampling(config)
        if cache_key === nothing
            relation = CoreRelation(
                global_consts=global_consts,
                global_consts_ty=global_consts_ty,
                local_consts=result_local_consts,
                local_consts_ty=result_local_const_types,
                presets=presets,
                preset_types=preset_types,
                weights=config.weights,
                providers=config.providers,
                argument_planner=config.argument_planner,
                guarded_expansion_depth=config.guarded_expansion_depth,
            )
            return _generate_lazy_result(relation, config, target_ty, result_local_consts, result_local_const_types)
        end

        entry = _get_default_lazy_relation_cache_entry(
            cache_key,
            config.weights,
            config.providers,
            config.argument_planner,
            config.guarded_expansion_depth,
            global_consts,
            global_consts_ty,
        )
        lock(entry.lock)
        try
            return _generate_lazy_result(entry.relation, config, target_ty, result_local_consts, result_local_const_types)
        finally
            unlock(entry.lock)
        end
    end

    if cache_key === nothing
        relation = CoreRelation(
            global_consts=global_consts,
            global_consts_ty=global_consts_ty,
            local_consts=result_local_consts,
            local_consts_ty=result_local_const_types,
            presets=presets,
            preset_types=preset_types,
            weights=config.weights,
            providers=config.providers,
            argument_planner=config.argument_planner,
            guarded_expansion_depth=config.guarded_expansion_depth,
        )
        derivations = _known_derivations!(relation, target_ty, Ty[])
        candidates = _generation_candidates(
            relation,
            target_ty,
            Ty[],
            derivations;
            scope_application_depth_left=config.sampling_depth_limit,
        )
        sample_set = _sample_set(candidates, config.weights)
        known_count, scope_count = _relation_counts(relation)
    else
        entry = _get_default_relation_cache_entry(
            cache_key,
            target_ty,
            config.weights,
            config.providers,
            config.argument_planner,
            config.guarded_expansion_depth,
            global_consts,
            global_consts_ty,
        )
        relation = entry.relation
        sample_set = entry.sample_set
        known_count = entry.known_count
        scope_count = entry.scope_count
    end

    derivation = config.sampler(config.rng, sample_set)
    derivation === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    node = _materialize_derivation(config.materializer, derivation, result_local_consts, result_local_const_types, relation, config.rng)
    metadata = Dict{Symbol,Any}(
        :generator => :canonical_finite_core,
        :rng_type => typeof(config.rng),
        :known_count => known_count,
        :scope_count => scope_count,
        :derivation => _derivation_signature(derivation),
    )
    return GenerationResult(AstTree(node), target_ty, result_local_consts, result_local_const_types, metadata)
end

"""
    generate_canonical_scoped(target_ty; scope_tys=Ty[], kwargs...)

Generate a canonical witness while making the surrounding lambda binders
available as ordinary De Bruijn variables. `scope_tys` is ordered inner → outer:
`scope_tys[1]` materializes as `IndexNode(0)`.

This is the mutation-facing companion to [`generate_canonical`](@ref). It uses
the same finite catalog, providers, weights, budgets, local constant
materialization, and `GenerationConfig`; only the initial lexical scope differs.
"""
function generate_canonical_scoped(
    target_ty::TyLike;
    scope_tys::AbstractVector{<:Ty}=Ty[],
    rng::Random.AbstractRNG=Random.default_rng(),
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    weights::GenerationWeights=GenerationWeights(),
    providers=default_head_providers(),
    guarded_expansion_depth::Integer=1,
    sampling_depth_limit::Integer=10,
    sampling_node_budget::Integer=128,
    source_local_consts::Bool=true,
    config::GenerationConfig=GenerationConfig(; rng, weights, providers, guarded_expansion_depth, sampling_depth_limit, sampling_node_budget),
    _cache_relation::Bool=true,
)
    if target_ty isa TyScheme
        demand = _rigidify_scheme_demand(target_ty)
        demand === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
        result = generate_canonical_scoped(
            demand.target;
            scope_tys,
            rng,
            global_consts,
            global_consts_ty,
            local_consts,
            local_consts_ty,
            presets,
            preset_types,
            weights,
            providers,
            guarded_expansion_depth,
            sampling_depth_limit,
            sampling_node_budget,
            source_local_consts,
            config,
            _cache_relation=false,
        )
        result isa GenerationResult || return result
        return _discharge_rigid_result(result, target_ty, demand)
    end
    target_ty isa Ty || return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    _in_closed_fragment(target_ty) || return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    scope = Ty[scope_tys...]
    all(_in_closed_fragment, scope) || return Err(UnsupportedCanonicalGenerationTarget(target_ty))

    result_local_consts = _anydict(local_consts)
    result_local_const_types = _typed_context(local_consts, local_consts_ty)
    relation_local_consts = source_local_consts ? result_local_consts : Dict{Symbol,Any}()
    relation_local_const_types = source_local_consts ? result_local_const_types : Dict{Symbol,TyLike}()
    cache_key = (_cache_relation && _uses_lazy_sampling(config)) ? _scoped_lazy_relation_cache_key(
        config.weights,
        global_consts,
        global_consts_ty,
        relation_local_consts,
        relation_local_const_types,
        presets,
        preset_types,
        config.providers,
        config.argument_planner,
        config.guarded_expansion_depth,
    ) : nothing
    if cache_key !== nothing
        entry = _get_scoped_lazy_relation_cache_entry(
            cache_key,
            config.weights,
            config.providers,
            config.argument_planner,
            config.guarded_expansion_depth,
            global_consts,
            global_consts_ty,
            relation_local_consts,
            relation_local_const_types,
        )
        lock(entry.lock)
        try
            entry.relation.local_consts = relation_local_consts
            entry.relation.local_consts_ty = relation_local_const_types
            return _generate_scoped_result(entry.relation, config, target_ty, scope, result_local_consts, result_local_const_types)
        finally
            unlock(entry.lock)
        end
    end

    relation = CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts=relation_local_consts,
        local_consts_ty=relation_local_const_types,
        presets=presets,
        preset_types=preset_types,
        weights=config.weights,
        providers=config.providers,
        argument_planner=config.argument_planner,
        guarded_expansion_depth=config.guarded_expansion_depth,
    )
    return _generate_scoped_result(relation, config, target_ty, scope, result_local_consts, result_local_const_types)
end

function _generate_lazy_result(
    relation::CoreRelation,
    config::GenerationConfig,
    target_ty::Ty,
    result_local_consts::Dict{Symbol,Any},
    result_local_const_types::Dict{Symbol,TyLike},
)
    state = DerivationSamplingState(config.rng, config.sampling_node_budget)
    derivation = _sample_generation_derivation(
        state,
        relation,
        target_ty,
        Ty[],
        config.sampling_depth_limit,
        config.guarded_expansion_depth,
    )
    derivation === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    node = _materialize_derivation(config.materializer, derivation, result_local_consts, result_local_const_types, relation, config.rng)
    known_count, scope_count = _relation_counts(relation)
    metadata = Dict{Symbol,Any}(
        :generator => :canonical_finite_core,
        :rng_type => typeof(config.rng),
        :known_count => known_count,
        :scope_count => scope_count,
        :sampling_nodes_used => config.sampling_node_budget - state.nodes_left,
        :derivation => _derivation_signature(derivation),
    )
    return GenerationResult(AstTree(node), target_ty, result_local_consts, result_local_const_types, metadata)
end

function _generate_scoped_result(
    relation::CoreRelation,
    config::GenerationConfig,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    result_local_consts::Dict{Symbol,Any},
    result_local_const_types::Dict{Symbol,TyLike},
)
    if _uses_lazy_sampling(config)
        state = DerivationSamplingState(config.rng, config.sampling_node_budget)
        derivation = _sample_generation_derivation(
            state,
            relation,
            target_ty,
            scope_tys,
            config.sampling_depth_limit,
            config.guarded_expansion_depth,
        )
        derivation === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
        node = _materialize_derivation(config.materializer, derivation, result_local_consts, result_local_const_types, relation, config.rng)
        known_count, scope_count = _relation_counts(relation)
        metadata = Dict{Symbol,Any}(
            :generator => :canonical_finite_core,
            :rng_type => typeof(config.rng),
            :known_count => known_count,
            :scope_count => scope_count,
            :sampling_nodes_used => config.sampling_node_budget - state.nodes_left,
            :scope_tys => Tuple(typekey.(scope_tys)),
            :derivation => _derivation_signature(derivation),
        )
        return GenerationResult(AstTree(node), target_ty, result_local_consts, result_local_const_types, metadata)
    end

    derivations = _known_derivations!(relation, target_ty, scope_tys)
    candidates = _generation_candidates(
        relation,
        target_ty,
        scope_tys,
        derivations;
        scope_application_depth_left=config.sampling_depth_limit,
    )
    sample_set = _sample_set(candidates, config.weights)
    derivation = config.sampler(config.rng, sample_set)
    derivation === nothing && return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    node = _materialize_derivation(config.materializer, derivation, result_local_consts, result_local_const_types, relation, config.rng)
    known_count, scope_count = _relation_counts(relation)
    metadata = Dict{Symbol,Any}(
        :generator => :canonical_finite_core,
        :rng_type => typeof(config.rng),
        :known_count => known_count,
        :scope_count => scope_count,
        :scope_tys => Tuple(typekey.(scope_tys)),
        :derivation => _derivation_signature(derivation),
    )
    return GenerationResult(AstTree(node), target_ty, result_local_consts, result_local_const_types, metadata)
end

function _relation_counts(relation::CoreRelation)
    return (
        sum(length, values(relation.derivations); init=0),
        length(Set(first(key) for key in keys(relation.derivations))),
    )
end

function _default_relation_cache_key(
    target_ty::Ty,
    weights::GenerationWeights,
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
    local_consts::AbstractDict,
    local_consts_ty::AbstractDict,
    presets::AbstractDict,
    preset_types::AbstractDict,
    providers::Tuple,
    argument_planner,
    guarded_expansion_depth::Integer=1,
)
    provider_key = _cacheable_providers_key(_provider_tuple(providers))
    provider_key === nothing && return nothing
    argument_planner isa DefaultArgumentPlanner || return nothing
    global_key = _cacheable_global_consts_key(global_consts, global_consts_ty)
    global_key === nothing && return nothing
    isempty(local_consts) || return nothing
    isempty(local_consts_ty) || return nothing
    isempty(presets) || return nothing
    isempty(preset_types) || return nothing
    return (
        _cache_type_key(target_ty),
        _weights_key(weights),
        Int(guarded_expansion_depth),
        global_key,
        provider_key,
    )
end

function _scoped_lazy_relation_cache_key(
    weights::GenerationWeights,
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
    local_consts::AbstractDict,
    local_consts_ty::AbstractDict,
    presets::AbstractDict,
    preset_types::AbstractDict,
    providers::Tuple,
    argument_planner,
    guarded_expansion_depth::Integer=1,
)
    provider_key = _cacheable_providers_key(_provider_tuple(providers))
    provider_key === nothing && return nothing
    argument_planner isa DefaultArgumentPlanner || return nothing
    global_key = _cacheable_global_consts_key(global_consts, global_consts_ty)
    global_key === nothing && return nothing
    isempty(presets) || return nothing
    isempty(preset_types) || return nothing
    local_key = _cacheable_local_context_key(local_consts, local_consts_ty)
    local_key === nothing && return nothing
    return (
        :scoped_lazy_relation,
        _weights_key(weights),
        Int(guarded_expansion_depth),
        global_key,
        local_key,
        provider_key,
    )
end

function _cacheable_local_context_key(local_consts::AbstractDict, local_consts_ty::AbstractDict)
    length(local_consts_ty) == length(local_consts) || return nothing
    entries = Vector{Any}()
    for name in sort!(collect(keys(local_consts_ty)))
        haskey(local_consts, name) || return nothing
        ty = local_consts_ty[name]
        ty isa TyLike || return nothing
        push!(entries, (name, _cache_tylike_key(ty)))
    end
    return Tuple(entries)
end

function _cacheable_providers_key(providers::Tuple)
    entries = Any[]
    for provider in providers
        key = _cacheable_provider_key(provider)
        key === nothing && return nothing
        push!(entries, key)
    end
    return Tuple(entries)
end

_cacheable_provider_key(::ContextHeadProvider) = (:context,)
_cacheable_provider_key(::NilFactoryProvider) = (:nil_factory,)
_cacheable_provider_key(::ToListProvider) = (:tolist,)
_cacheable_provider_key(::AMapProvider) = (:amap,)
_cacheable_provider_key(::GuardedExpansionProvider) = (:guarded_expansion,)
function _cacheable_provider_key(provider::SafeListAccessProvider)
    isempty(provider.source_collectors) || return nothing
    return (:safe_list_access, provider.source_array_depth_limit)
end
_cacheable_provider_key(provider::WeightTagProvider) =
    (:weight_tags, Tuple((name, tags) for (name, tags) in sort!(collect(provider.tags_by_name); by=first)))

function _cacheable_provider_key(provider::TypedValueFactoryProvider)
    factories_key = _cacheable_value_factories_key(provider.factories)
    factories_key === nothing && return nothing
    return (:typed_value_factory, factories_key)
end

_cacheable_provider_key(provider) = nothing

function _cacheable_value_factories_key(factories)
    entries = Any[]
    for factory in factories
        key = _cacheable_value_factory_key(factory)
        key === nothing && return nothing
        push!(entries, key)
    end
    return Tuple(entries)
end

_cacheable_value_factory_key(::ValueFactories.NilFactory) = (:nil,)
_cacheable_value_factory_key(factory::ValueFactories.FloatScalarFactory) =
    (:float_scalar, factory.mean, factory.std)
_cacheable_value_factory_key(factory::ValueFactories.IntScalarFactory) =
    (:int_scalar, factory.lo, factory.hi)
_cacheable_value_factory_key(factory::ValueFactories.FloatArrayFactory) =
    (:float_array, factory.mean, factory.std)
_cacheable_value_factory_key(factory::ValueFactories.IntArrayFactory) =
    (:int_array, factory.lo, factory.hi)
_cacheable_value_factory_key(::ValueFactories.BoolArrayFactory) = (:bool_array,)
_cacheable_value_factory_key(::ValueFactories.AbstractValueFactory) = nothing

function _cacheable_global_consts_key(global_consts::AbstractDict, global_consts_ty::AbstractDict)
    isempty(global_consts) && isempty(global_consts_ty) && return ()
    global_tyctx = _typed_context(global_consts, global_consts_ty)
    length(global_tyctx) == length(global_consts) || return nothing
    entries = Vector{Any}()
    for name in sort!(collect(keys(global_consts)))
        haskey(global_tyctx, name) || return nothing
        ty = global_tyctx[name]
        value = global_consts[name]
        ty isa TyLike || return nothing
        value_key = _cacheable_global_value_key(value)
        value_key === nothing && return nothing
        push!(entries, (name, value_key, _cache_tylike_key(ty)))
    end
    return Tuple(entries)
end

_cacheable_global_value_key(value) =
    Base.issingletontype(typeof(value)) ? (:singleton, typeof(value)) : nothing
_cacheable_global_value_key(value::Bool) = (:bool, value)

function _weights_key(weights::GenerationWeights)
    return (
        weights.default,
        Tuple((name, value) for (name, value) in sort!(collect(weights.overrides); by=first)),
    )
end

# Cache keys are intentionally structural for the current closed fragment.
# Use `typekey` for TyLike values that may include schemes or recursive types.
_cache_type_key(ty::TyConst) = (:const, ty.name)
_cache_type_key(ty::TyVar) = (:var, ty.id, ty.level)
_cache_type_key(ty::TyNat) = (:nat, ty.value)
_cache_type_key(ty::TyList) = (:list, _cache_type_key(ty.elem))
_cache_type_key(ty::TyArray) = (:array, _cache_type_key(ty.len), _cache_type_key(ty.elem))
_cache_type_key(ty::TyFun) = (:fun, _cache_type_key(ty.arg), _cache_type_key(ty.res))
_cache_type_key(ty::TyProd) = (:prod, _cache_type_key(ty.left), _cache_type_key(ty.right))
_cache_type_key(ty::TySum) = (:sum, _cache_type_key(ty.left), _cache_type_key(ty.right))
_cache_type_key(ty::TyMu) = (:mu, _cache_type_key(ty.var), _cache_type_key(ty.body))
_cache_tylike_key(ty::TyLike) = typekey(ty)
_closed_type_key(ty::Ty) = _cache_type_key(ty)

function _get_default_lazy_relation_cache_entry(
    cache_key,
    weights::GenerationWeights,
    providers::Tuple,
    argument_planner,
    guarded_expansion_depth::Int,
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
)
    lock(_DEFAULT_LAZY_RELATION_CACHE_LOCK)
    try
        entry = get(_DEFAULT_LAZY_RELATION_CACHE, cache_key, nothing)
        entry === nothing || return entry
    finally
        unlock(_DEFAULT_LAZY_RELATION_CACHE_LOCK)
    end

    relation = CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        weights=weights,
        providers=providers,
        argument_planner=argument_planner,
        guarded_expansion_depth=guarded_expansion_depth,
    )
    entry = LazyRelationCacheEntry(relation, ReentrantLock())

    lock(_DEFAULT_LAZY_RELATION_CACHE_LOCK)
    try
        return get!(_DEFAULT_LAZY_RELATION_CACHE, cache_key, entry)
    finally
        unlock(_DEFAULT_LAZY_RELATION_CACHE_LOCK)
    end
end

function _get_scoped_lazy_relation_cache_entry(
    cache_key,
    weights::GenerationWeights,
    providers::Tuple,
    argument_planner,
    guarded_expansion_depth::Int,
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
    local_consts::AbstractDict,
    local_consts_ty::AbstractDict,
)
    lock(_SCOPED_LAZY_RELATION_CACHE_LOCK)
    try
        entry = get(_SCOPED_LAZY_RELATION_CACHE, cache_key, nothing)
        entry === nothing || return entry
    finally
        unlock(_SCOPED_LAZY_RELATION_CACHE_LOCK)
    end

    relation = CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
        weights=weights,
        providers=providers,
        argument_planner=argument_planner,
        guarded_expansion_depth=guarded_expansion_depth,
    )
    entry = ScopedLazyRelationCacheEntry(relation, ReentrantLock())

    lock(_SCOPED_LAZY_RELATION_CACHE_LOCK)
    try
        return get!(_SCOPED_LAZY_RELATION_CACHE, cache_key, entry)
    finally
        unlock(_SCOPED_LAZY_RELATION_CACHE_LOCK)
    end
end

function _get_default_relation_cache_entry(
    cache_key,
    target_ty::Ty,
    weights::GenerationWeights,
    providers::Tuple,
    argument_planner,
    guarded_expansion_depth::Int,
    global_consts::AbstractDict,
    global_consts_ty::AbstractDict,
)
    lock(_DEFAULT_RELATION_CACHE_LOCK)
    try
        entry = get(_DEFAULT_RELATION_CACHE, cache_key, nothing)
        entry === nothing || return entry
    finally
        unlock(_DEFAULT_RELATION_CACHE_LOCK)
    end

    relation = CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        weights=weights,
        providers=providers,
        argument_planner=argument_planner,
        guarded_expansion_depth=guarded_expansion_depth,
    )
    derivations = _known_derivations!(relation, target_ty, Ty[])
    candidates = _generation_candidates(relation, target_ty, Ty[], derivations)
    sample_set = _sample_set(candidates, weights)
    known_count, scope_count = _relation_counts(relation)
    entry = CoreRelationCacheEntry(relation, sample_set, known_count, scope_count)

    lock(_DEFAULT_RELATION_CACHE_LOCK)
    try
        return get!(_DEFAULT_RELATION_CACHE, cache_key, entry)
    finally
        unlock(_DEFAULT_RELATION_CACHE_LOCK)
    end
end

function _known_derivations!(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty})
    _in_closed_fragment(target_ty) || return CoreDerivation[]
    key = _known_key(scope_tys, target_ty)
    haskey(relation.derivations, key) || (relation.derivations[key] = CoreDerivation[])
    key in relation.computed && return relation.derivations[key]
    key in relation.visiting && return relation.derivations[key]

    push!(relation.visiting, key)
    changed = true
    while changed
        changed = false
        changed |= _add_variable_derivations!(relation, key, target_ty, scope_tys)
        changed |= _add_terminal_head_derivations!(relation, key, target_ty, scope_tys)
        changed |= _add_lambda_derivations!(relation, key, target_ty, scope_tys)
        changed |= _add_productive_head_derivations!(relation, key, target_ty, scope_tys)
    end
    delete!(relation.visiting, key)
    push!(relation.computed, key)
    return relation.derivations[key]
end

_known_key(scope_tys::Vector{Ty}, target_ty::Ty) = (Tuple(typekey.(scope_tys)), typekey(target_ty))

function _add_variable_derivations!(
    relation::CoreRelation,
    key::Tuple{Any,Any},
    target_ty::Ty,
    scope_tys::Vector{Ty},
)
    _bucket_enabled(relation.weights, _VARIABLE_WEIGHT) || return false
    changed = false
    for (idx0, scope_ty) in enumerate(scope_tys)
        iscompatible(scope_ty, target_ty) || continue
        changed |= _add_derivation!(relation, key, VariableDerivation(idx0 - 1, scope_ty))
    end
    return changed
end

function _add_terminal_head_derivations!(
    relation::CoreRelation,
    key::Tuple{Any,Any},
    target_ty::Ty,
    scope_tys::Vector{Ty},
)
    changed = false
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for head in catalog.heads
        head.type isa Ty || continue
        head.rule_class isa CoreProductive || continue
        _is_instance_factory_head(head) && continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        iscompatible(head.type, target_ty) || continue
        changed |= _add_derivation!(relation, key, ConstantDerivation(head))
    end
    return changed
end

function _add_lambda_derivations!(
    relation::CoreRelation,
    key::Tuple{Any,Any},
    target_ty::Ty,
    scope_tys::Vector{Ty},
)
    target_ty isa TyFun || return false
    body_scope = Ty[target_ty.arg; scope_tys]
    body_derivations = _known_derivations!(relation, target_ty.res, body_scope)
    isempty(body_derivations) && return false
    changed = false
    for body in body_derivations
        changed |= _add_derivation!(relation, key, LambdaDerivation(target_ty.arg, body))
    end
    return changed
end

function _add_productive_head_derivations!(
    relation::CoreRelation,
    key::Tuple{Any,Any},
    target_ty::Ty,
    scope_tys::Vector{Ty},
)
    changed = false
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        iscompatible(rule.result_ty, target_ty) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        isempty(rule.argument_tys) && !_is_instance_factory_head(rule.head) && continue
        arg_derivations = Vector{Vector{CoreDerivation}}(undef, length(rule.argument_tys))
        all_known = true
        for i in _argument_resolution_order(rule)
            arg_ty = rule.argument_tys[i]
            arg_class = rule.argument_classes[i]
            if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
                all_known = false
                break
            end
            raw_derivations = _raw_argument_derivations(relation, arg_ty, arg_class, rule, scope_tys)
            derivs = relation.argument_planner(
                raw_derivations,
                arg_ty,
                arg_class,
                rule,
                relation,
            )
            if isempty(derivs)
                all_known = false
                break
            end
            arg_derivations[i] = derivs
        end
        all_known || continue
        for args in _derivation_combinations(arg_derivations)
            changed |= _add_derivation!(relation, key, HeadApplicationDerivation(rule.head, args))
        end
    end
    return changed
end

function (::DefaultArgumentPlanner)(
    derivations::Vector{CoreDerivation},
    arg_ty::Ty,
    arg_class::Symbol,
    rule::CoreRule,
    relation::CoreRelation,
)
    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    planner === nothing || return planner(derivations, arg_ty, arg_class, rule, relation)
    if arg_class == :source_array
        max_depth = _source_array_depth_limit(rule)
        return [derivation for derivation in derivations if _array_source_depth(derivation) <= max_depth]
    end
    return [
        derivation for derivation in derivations
        if !(derivation isa HeadApplicationDerivation) && !(derivation isa ScopeApplicationDerivation)
    ]
end

function _argument_resolution_order(rule)
    return sort(
        collect(eachindex(rule.argument_tys));
        by=i -> rule.argument_classes[i] == :source_array ? 0 : 1,
    )
end

function _grounded_derivations(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty})
    out = CoreDerivation[]
    signatures = Set{Any}()
    for (idx0, scope_ty) in enumerate(scope_tys)
        iscompatible(scope_ty, target_ty) || continue
        _push_derivation_candidate!(out, signatures, VariableDerivation(idx0 - 1, scope_ty))
    end
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for head in catalog.heads
        head.type isa Ty || continue
        head.rule_class isa CoreProductive || continue
        _is_instance_factory_head(head) && continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        iscompatible(head.type, target_ty) || continue
        _push_derivation_candidate!(out, signatures, ConstantDerivation(head))
    end
    return out
end

function _raw_argument_derivations(
    relation::CoreRelation,
    arg_ty::Ty,
    arg_class::Symbol,
    rule::CoreRule,
    scope_tys::Vector{Ty},
)
    if arg_class == :source_array
        return _bounded_source_array_derivations(relation, arg_ty, scope_tys, _source_array_depth_limit(rule))
    end
    return _known_derivations!(relation, arg_ty, scope_tys)
end

function _source_array_depth_limit(rule::CoreRule)
    limit = _metadata_get(rule.head.metadata, :source_array_depth_limit, nothing)
    limit === nothing || return _checked_nonnegative_int(:source_array_depth_limit, limit)
    return 0
end

function _bounded_source_array_derivations(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    max_depth::Int,
)
    key = (:source_array, _known_key(scope_tys, target_ty), max_depth)
    cached = get(relation.source_derivations, key, nothing)
    cached === nothing || return cached
    out = _grounded_derivations(relation, target_ty, scope_tys)
    if max_depth <= 0
        relation.source_derivations[key] = out
        return out
    end

    signatures = Set{Any}(_derivation_signature.(out))
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        _is_source_array_constructor(rule.head) || continue
        iscompatible(rule.result_ty, target_ty) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue

        arg_derivations = Vector{Vector{CoreDerivation}}(undef, length(rule.argument_tys))
        all_known = true
        for i in _argument_resolution_order(rule)
            arg_ty = rule.argument_tys[i]
            arg_class = rule.argument_classes[i]
            raw_derivations =
                arg_class == :source_array ?
                _bounded_source_array_derivations(relation, arg_ty, scope_tys, max_depth - 1) :
                _known_derivations!(relation, arg_ty, scope_tys)
            derivs = relation.argument_planner(raw_derivations, arg_ty, arg_class, rule, relation)
            if isempty(derivs)
                all_known = false
                break
            end
            arg_derivations[i] = derivs
        end
        all_known || continue
        for args in _derivation_combinations(arg_derivations)
            _push_derivation_candidate!(out, signatures, HeadApplicationDerivation(rule.head, args))
        end
    end
    relation.source_derivations[key] = out
    return out
end

_is_source_array_constructor(head::HeadEntry) =
    _metadata_get(head.metadata, :source_array_constructor, false) === true

function _source_array_argument_index(head::HeadEntry, nargs::Int)
    idx = _metadata_get(head.metadata, :source_array_argument_index, nargs)
    1 <= idx <= nargs || throw(ArgumentError("source_array_argument_index must be inside head arity"))
    return idx
end

function _generation_candidates(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    core_derivations::Vector{CoreDerivation}=_known_derivations!(relation, target_ty, scope_tys),
    ;
    allow_safe_list_access::Bool=true,
    scope_application_depth_left::Int=0,
)
    candidates = CoreDerivation[]
    signatures = Set{Any}()
    for derivation in core_derivations
        _push_derivation_candidate!(candidates, signatures, derivation)
    end

    if target_ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        body_scope = Ty[target_ty.arg; scope_tys]
        body_core = _known_derivations!(relation, target_ty.res, body_scope)
        for body in _generation_candidates(
            relation,
            target_ty.res,
            body_scope,
            body_core;
            allow_safe_list_access,
            scope_application_depth_left=max(scope_application_depth_left - 1, 0),
        )
            _push_derivation_candidate!(candidates, signatures, LambdaDerivation(target_ty.arg, body))
        end
    end

    if scope_application_depth_left > 0 && _scope_application_enabled(relation.weights)
        for derivation in _scope_application_candidates(
            relation,
            target_ty,
            scope_tys,
            scope_application_depth_left,
            allow_safe_list_access,
        )
            _push_derivation_candidate!(candidates, signatures, derivation)
        end
    end

    if relation.guarded_expansion_depth > 0 && _bucket_enabled(relation.weights, _GUARDED_EXPANSION_WEIGHT)
        expansions = _guarded_expansion_candidates(relation, target_ty, scope_tys, copy(candidates))
        for derivation in expansions
            _push_derivation_candidate!(candidates, signatures, derivation)
        end
    end
    if allow_safe_list_access
        for derivation in _safe_list_access_candidates(relation, target_ty, scope_tys, core_derivations)
            _push_derivation_candidate!(candidates, signatures, derivation)
        end
    end
    return candidates
end

function _scope_application_candidates(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    allow_safe_list_access::Bool,
)
    depth_left > 0 || return CoreDerivation[]
    out = CoreDerivation[]
    signatures = Set{Any}()
    for rule in _scope_application_rules(target_ty, scope_tys)
        arg_derivations = Vector{Vector{CoreDerivation}}(undef, length(rule.argument_tys))
        all_known = true
        for i in _argument_resolution_order(rule)
            arg_ty = rule.argument_tys[i]
            arg_class = rule.argument_classes[i]
            if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
                all_known = false
                break
            end
            raw_derivations =
                arg_class == :source_array ?
                _bounded_source_array_derivations(relation, arg_ty, scope_tys, 0) :
                _known_derivations!(relation, arg_ty, scope_tys)
            derivs =
                arg_class == :source_array ?
                raw_derivations :
                _generation_candidates(
                    relation,
                    arg_ty,
                    scope_tys,
                    raw_derivations;
                    allow_safe_list_access,
                    scope_application_depth_left=depth_left - 1,
                )
            if isempty(derivs)
                all_known = false
                break
            end
            arg_derivations[i] = derivs
        end
        all_known || continue
        for args in _derivation_combinations(arg_derivations)
            _push_derivation_candidate!(
                out,
                signatures,
                ScopeApplicationDerivation(rule.index, rule.head_ty, args),
            )
        end
    end
    return out
end

function _guarded_expansion_candidates(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    core_target_derivations::Vector{CoreDerivation},
)
    core_target_options = [
        derivation for derivation in core_target_derivations
        if _guarded_depth(derivation) == 0
    ]
    isempty(core_target_options) && return CoreDerivation[]

    out = CoreDerivation[]
    for rule in _guarded_rules_for_target(relation, target_ty, scope_tys)
        arg_derivations = Vector{Vector{CoreDerivation}}(undef, length(rule.argument_tys))
        all_known = true
        for i in _argument_resolution_order(rule)
            arg_ty = rule.argument_tys[i]
            arg_class = rule.argument_classes[i]
            if typekey(arg_ty) == typekey(target_ty)
                derivs = core_target_options
            else
                if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
                    all_known = false
                    break
                end
                raw_derivations = _raw_guarded_argument_derivations(relation, arg_ty, arg_class, rule, i, scope_tys)
                derivs = _guarded_argument_derivations(
                    relation,
                    raw_derivations,
                    arg_ty,
                    arg_class,
                    rule,
                )
            end
            derivs = [
                derivation for derivation in derivs
                if _guarded_depth(derivation) < relation.guarded_expansion_depth
            ]
            if isempty(derivs)
                all_known = false
                break
            end
            arg_derivations[i] = derivs
        end
        all_known || continue
        for args in _derivation_combinations(arg_derivations)
            push!(out, HeadApplicationDerivation(rule.head, args))
        end
    end
    return out
end

_afoldl_step_argument(rule::CoreRule, i::Int) =
    rule.head.name == :AFOLDL && i == firstindex(rule.argument_tys)

_afoldl_source_argument(rule::CoreRule, i::Int) =
    rule.head.name == :AFOLDL && rule.argument_classes[i] == :source_array

_afoldl_grounded_argument(rule::CoreRule, i::Int) = _afoldl_source_argument(rule, i)

_cond_condition_argument(rule::CoreRule, i::Int) =
    rule.head.name == :COND && i == firstindex(rule.argument_tys)

function _afoldl_step_parts(rule::CoreRule)
    isempty(rule.argument_tys) && return nothing
    step_ty = first(rule.argument_tys)
    step_ty isa TyFun || return nothing
    elem_fun = step_ty.res
    elem_fun isa TyFun || return nothing
    return (acc_ty=step_ty.arg, elem_ty=elem_fun.arg, body_ty=elem_fun.res)
end

_array_rank(ty::Ty) = ty isa TyArray ? 1 + _array_rank(ty.elem) : 0

function _afoldl_step_scope(rule::CoreRule, scope_tys::Vector{Ty})
    parts = _afoldl_step_parts(rule)
    parts === nothing && return Ty[]
    return Ty[parts.elem_ty, parts.acc_ty, scope_tys...]
end

function _raw_guarded_argument_derivations(
    relation::CoreRelation,
    arg_ty::Ty,
    arg_class::Symbol,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
)
    _afoldl_grounded_argument(rule, i) && return _grounded_derivations(relation, arg_ty, scope_tys)
    return _raw_argument_derivations(relation, arg_ty, arg_class, rule, scope_tys)
end

function _guarded_argument_derivations(
    relation::CoreRelation,
    derivations::Vector{CoreDerivation},
    arg_ty::Ty,
    arg_class::Symbol,
    rule::CoreRule,
)
    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    if arg_class == :source_array
        max_depth = _source_array_depth_limit(rule)
        return [derivation for derivation in derivations if _array_source_depth(derivation) <= max_depth]
    end
    planner === nothing && return derivations
    return planner(derivations, arg_ty, arg_class, rule, relation)
end

function _guarded_rules_for_target(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty})
    _bucket_enabled(relation.weights, _GUARDED_EXPANSION_WEIGHT) || return CoreRule[]
    if _can_use_builtin_guarded_fast_path(relation.providers)
        _has_builtin_guarded_provider(relation.providers) || return CoreRule[]
        universe = _relation_catalog(relation, target_ty, scope_tys).universe
        heads = HeadEntry[]
        _add_target_cons_head!(heads, target_ty)
        _add_target_cond_head!(heads, universe, target_ty)
        _add_target_afoldl_heads!(heads, universe, target_ty)
        _apply_weight_tag_providers!(heads, universe, relation.providers)
        return _guarded_rules_from_heads(relation, heads, target_ty)
    end

    catalog = _relation_catalog(relation, target_ty, scope_tys; include_guarded=true)
    return _guarded_rules_from_heads(relation, catalog.heads, target_ty)
end

function _guarded_rules_from_heads(relation::CoreRelation, heads::Vector{HeadEntry}, target_ty::Ty)
    rules = CoreRule[]
    for head in heads
        _is_guarded_head(head) || continue
        head.type isa Ty || continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        args, result_ty = _split_fun(head.type)
        isempty(args) && continue
        iscompatible(result_ty, target_ty) || continue
        push!(rules, CoreRule(head, result_ty, args, [_argument_class(arg) for arg in args]))
    end
    return rules
end

_has_builtin_guarded_provider(providers::Tuple) =
    any(provider -> provider isa GuardedExpansionProvider, providers)

_can_use_builtin_guarded_fast_path(providers::Tuple) =
    all(_is_builtin_catalog_provider, providers)

_is_builtin_catalog_provider(provider) =
    provider isa ContextHeadProvider ||
    provider isa NilFactoryProvider ||
    provider isa TypedValueFactoryProvider ||
    provider isa ToListProvider ||
    provider isa AMapProvider ||
    provider isa GuardedExpansionProvider ||
    provider isa SafeListAccessProvider ||
    provider isa WeightTagProvider

function _apply_weight_tag_providers!(heads::Vector{HeadEntry}, universe::CoreUniverse, providers::Tuple)
    for provider in providers
        provider isa WeightTagProvider || continue
        add_heads!(provider, heads, universe)
    end
    return heads
end

function _add_target_cons_head!(heads::Vector{HeadEntry}, target_ty::Ty)
    target_ty isa TyList || return heads
    cons_ty = TyFun(target_ty.elem, TyFun(target_ty, target_ty))
    push!(
        heads,
        HeadEntry(
            :CONS,
            :guarded_primitive,
            cons_ty,
            Decoration(),
            _GUARDED_EXPANSION_WEIGHT;
            metadata=(value=TypeSystem.CONS, guarded=true),
        ),
    )
    return heads
end

function _add_target_cond_head!(heads::Vector{HeadEntry}, universe::CoreUniverse, target_ty::Ty)
    _type_in_universe(universe, BoolTy) || return heads
    cond_ty = TyFun(BoolTy, TyFun(target_ty, TyFun(target_ty, target_ty)))
    push!(
        heads,
        HeadEntry(
            :COND,
            :guarded_primitive,
            cond_ty,
            Decoration(),
            _GUARDED_EXPANSION_WEIGHT;
            metadata=(value=TypeSystem.COND, guarded=true),
        ),
    )
    return heads
end

function _add_target_afoldl_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, target_ty::Ty)
    for array_ty in _closed_array_types(universe)
        step_ty = TyFun(target_ty, TyFun(array_ty.elem, target_ty))
        afoldl_ty = TyFun(step_ty, TyFun(target_ty, TyFun(array_ty, target_ty)))
        push!(
            heads,
            HeadEntry(
                :AFOLDL,
                :guarded_primitive,
                afoldl_ty,
                Decoration(),
                _GUARDED_EXPANSION_WEIGHT;
                metadata=(value=TypeSystem.AFOLDL, guarded=true),
            ),
        )
    end
    return heads
end

function _push_derivation_candidate!(
    candidates::Vector{CoreDerivation},
    signatures::Set{Any},
    derivation::CoreDerivation,
)
    signature = _derivation_signature(derivation)
    signature in signatures && return false
    push!(signatures, signature)
    push!(candidates, derivation)
    return true
end

function _derivation_combinations(options::Vector{Vector{CoreDerivation}})
    isempty(options) && return [CoreDerivation[]]
    head = first(options)
    tail_combos = _derivation_combinations(options[2:end])
    combos = Vector{CoreDerivation}[]
    for item in head
        for tail in tail_combos
            push!(combos, CoreDerivation[item; tail])
        end
    end
    return combos
end

function _sample_set(derivations::Vector{<:CoreDerivation}, weights::GenerationWeights)
    kept = CoreDerivation[]
    cumulative_weights = Float64[]
    total = 0.0
    for derivation in derivations
        weight = _derivation_weight(derivation, weights)
        weight > 0 || continue
        total += weight
        push!(kept, derivation)
        push!(cumulative_weights, total)
    end
    return DerivationSampleSet(kept, cumulative_weights, total)
end

function _sample_derivation(rng::Random.AbstractRNG, sample_set::DerivationSampleSet)
    sample_set.total_weight > 0 || return nothing
    mark = rand(rng) * sample_set.total_weight
    index = searchsortedfirst(sample_set.cumulative_weights, mark)
    index > length(sample_set.derivations) && (index = length(sample_set.derivations))
    return sample_set.derivations[index]
end

(::WeightedDerivationSampler)(rng::Random.AbstractRNG, sample_set::DerivationSampleSet) =
    _sample_derivation(rng, sample_set)

_uses_lazy_sampling(config::GenerationConfig) =
    config.sampler isa WeightedDerivationSampler && config.argument_planner isa DefaultArgumentPlanner

function _sample_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    _in_closed_fragment(target_ty) || return nothing
    options = copy(_sampling_options(
        relation,
        target_ty,
        scope_tys,
        depth_left;
        allow_head_application=true,
        lambda_kind=:lambda_generation,
    ))
    if depth_left > 0 && guarded_left > 0 && _bucket_enabled(relation.weights, _GUARDED_EXPANSION_WEIGHT)
        append!(options, _guarded_sampling_options(relation, target_ty, scope_tys))
    end
    if depth_left > 0 && allow_safe_list_access
        append!(options, _safe_list_access_generation_options(relation, target_ty, scope_tys))
    end
    return _sample_from_options(state, relation, options, scope_tys, depth_left, guarded_left, allow_safe_list_access)
end

function _sample_known_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int;
    allow_head_application::Bool=true,
)
    _in_closed_fragment(target_ty) || return nothing
    options = _sampling_options(
        relation,
        target_ty,
        scope_tys,
        depth_left;
        allow_head_application,
        lambda_kind=:lambda_known,
    )
    return _sample_from_options(state, relation, options, scope_tys, depth_left, 0)
end

function _sampling_options(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int;
    allow_head_application::Bool,
    lambda_kind::Symbol,
)
    key = (:sampling, _known_key(scope_tys, target_ty), depth_left, allow_head_application, lambda_kind)
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return cached
    options = NamedTuple[]
    if _bucket_enabled(relation.weights, _VARIABLE_WEIGHT)
        for (idx0, scope_ty) in enumerate(scope_tys)
            iscompatible(scope_ty, target_ty) || continue
            push!(options, (kind=:variable, value=VariableDerivation(idx0 - 1, scope_ty), weight=_bucket_weight(relation.weights, _VARIABLE_WEIGHT)))
        end
    end

    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for head in catalog.heads
        head.type isa Ty || continue
        head.rule_class isa CoreProductive || continue
        _is_instance_factory_head(head) && continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        iscompatible(head.type, target_ty) || continue
        push!(options, (kind=:constant, value=ConstantDerivation(head), weight=_head_weight(relation.weights, head)))
    end

    if depth_left > 0 && target_ty isa TyFun
        push!(options, (kind=lambda_kind, value=target_ty, weight=_bucket_weight(relation.weights, _LAMBDA_WEIGHT)))
    end

    if depth_left > 0 && allow_head_application
        if _scope_application_enabled(relation.weights)
            for rule in _scope_application_rules(target_ty, scope_tys)
                push!(options, (kind=:scope_application, value=rule, weight=_scope_application_option_weight(relation, rule)))
            end
        end
        for rule in catalog.rules
            iscompatible(rule.result_ty, target_ty) || continue
            _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
            _head_has_value(rule.head, relation) || continue
            isempty(rule.argument_tys) && !_is_instance_factory_head(rule.head) && continue
            push!(options, (kind=:productive_head, value=rule, weight=_head_option_weight(relation, rule)))
        end
    end
    relation.sampling_options[key] = options
    return options
end

function _guarded_sampling_options(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty})
    key = (:guarded, _known_key(scope_tys, target_ty))
    cached = get(relation.sampling_options, key, nothing)
    cached === nothing || return cached
    options = NamedTuple[]
    for rule in _guarded_rules_for_target(relation, target_ty, scope_tys)
        push!(options, (kind=:guarded_head, value=rule, weight=_head_option_weight(relation, rule)))
    end
    if !isempty(options)
        scale = length(options)
        options = [(kind=option.kind, value=option.value, weight=option.weight / scale) for option in options]
    end
    relation.sampling_options[key] = options
    return options
end

function _head_option_weight(relation::CoreRelation, rule::CoreRule)
    weight = _head_weight(relation.weights, rule.head)
    for arg_class in rule.argument_classes
        arg_class == :source_array && (weight *= _bucket_weight(relation.weights, _ARRAY_SOURCE_WEIGHT))
    end
    return weight
end

function _scope_application_option_weight(relation::CoreRelation, rule::ScopeApplicationRule)
    weight = _scope_application_weight(relation.weights)
    for arg_class in rule.argument_classes
        arg_class == :source_array && (weight *= _bucket_weight(relation.weights, _ARRAY_SOURCE_WEIGHT))
    end
    return weight
end

function _sample_from_options(
    state::DerivationSamplingState,
    relation::CoreRelation,
    options::Vector{<:NamedTuple},
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    pool, total_weight = _positive_option_pool(options)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
        saved_nodes = state.nodes_left
        derivation = _sample_option(
            state,
            relation,
            option,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
        )
        derivation !== nothing && return derivation
        state.nodes_left = saved_nodes
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

function _positive_option_pool(options::Vector{<:NamedTuple})
    pool = NamedTuple[]
    total_weight = 0.0
    for option in options
        option.weight > 0 || continue
        push!(pool, option)
        total_weight += option.weight
    end
    return pool, total_weight
end

function _weighted_option_index(rng::Random.AbstractRNG, pool::Vector{NamedTuple}, total_weight::Float64)
    mark = rand(rng) * total_weight
    running = 0.0
    chosen = lastindex(pool)
    for i in eachindex(pool)
        running += pool[i].weight
        if mark <= running
            chosen = i
            break
        end
    end
    return chosen
end

function _sample_option(
    state::DerivationSamplingState,
    relation::CoreRelation,
    option::NamedTuple,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    if option.kind == :variable || option.kind == :constant
        _consume_sampling_nodes!(state, 1) || return nothing
        return option.value
    elseif option.kind == :lambda_known || option.kind == :lambda_generation
        depth_left > 0 || return nothing
        _consume_sampling_nodes!(state, 1) || return nothing
        target_ty = option.value::TyFun
        body_scope = Ty[target_ty.arg; scope_tys]
        body =
            option.kind == :lambda_generation ?
            _sample_generation_derivation(
                state,
                relation,
                target_ty.res,
                body_scope,
                depth_left - 1,
                guarded_left,
                allow_safe_list_access,
            ) :
            _sample_known_derivation(state, relation, target_ty.res, body_scope, depth_left - 1)
        body === nothing && return nothing
        return LambdaDerivation(target_ty.arg, body)
    elseif option.kind == :productive_head
        return _sample_head_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left,
            false,
            allow_safe_list_access,
        )
    elseif option.kind == :scope_application
        return _sample_scope_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left,
            allow_safe_list_access,
        )
    elseif option.kind == :guarded_head
        guarded_left > 0 || return nothing
        return _sample_head_application(
            state,
            relation,
            option.value,
            scope_tys,
            depth_left - 1,
            guarded_left - 1,
            true,
            allow_safe_list_access,
        )
    elseif option.kind == :safe_list_access
        return _sample_safe_list_access(
            state,
            relation,
            option.value.source,
            option.value.target_ty,
            scope_tys,
            depth_left - 1,
            guarded_left,
        )
    end
    return nothing
end

function _sample_scope_application(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::ScopeApplicationRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    depth_left >= 0 || return nothing
    _consume_sampling_nodes!(state, 1) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    for i in _argument_resolution_order(rule)
        arg_ty = rule.argument_tys[i]
        arg_class = rule.argument_classes[i]
        if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
            return nothing
        end
        arg =
            arg_class == :source_array ?
            _sample_materialized_derivation(
                state,
                relation,
                _bounded_source_array_derivations(relation, arg_ty, scope_tys, 0),
            ) :
            _sample_generation_derivation(
                state,
                relation,
                arg_ty,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
            )
        arg === nothing && return nothing
        args[i] = arg
    end
    return ScopeApplicationDerivation(rule.index, rule.head_ty, args)
end

function _sample_head_application(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    guarded_head::Bool,
    allow_safe_list_access::Bool=true,
)
    depth_left >= 0 || return nothing
    _consume_sampling_nodes!(state, 1) || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    for i in _argument_resolution_order(rule)
        arg = _sample_head_argument(
            state,
            relation,
            rule,
            i,
            scope_tys,
            depth_left,
            guarded_left,
            guarded_head,
            allow_safe_list_access,
        )
        arg === nothing && return nothing
        args[i] = arg
    end
    return HeadApplicationDerivation(rule.head, args)
end

function _sample_head_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    i::Int,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    guarded_head::Bool,
    allow_safe_list_access::Bool=true,
)
    arg_ty = rule.argument_tys[i]
    arg_class = rule.argument_classes[i]
    if arg_class == :source_array && !_bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT)
        return nothing
    end

    if guarded_head && _afoldl_step_argument(rule, i)
        return _sample_afoldl_step_argument(
            state,
            relation,
            rule,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
        )
    end

    if guarded_head && _cond_condition_argument(rule, i)
        return _sample_known_derivation(state, relation, arg_ty, scope_tys, depth_left; allow_head_application=true)
    end

    planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    if planner !== nothing || arg_class == :source_array || (guarded_head && _afoldl_grounded_argument(rule, i))
        raw_derivations =
            guarded_head && _afoldl_grounded_argument(rule, i) ?
            _grounded_derivations(relation, arg_ty, scope_tys) :
            _raw_argument_derivations(relation, arg_ty, arg_class, rule, scope_tys)
        derivs =
            guarded_head ?
            _guarded_argument_derivations(relation, raw_derivations, arg_ty, arg_class, rule) :
            relation.argument_planner(raw_derivations, arg_ty, arg_class, rule, relation)
        return _sample_materialized_derivation(state, relation, derivs)
    end

    if guarded_head && typekey(arg_ty) == typekey(rule.result_ty)
        return _sample_known_derivation(state, relation, arg_ty, scope_tys, depth_left; allow_head_application=true)
    end
    return _sample_known_derivation(state, relation, arg_ty, scope_tys, depth_left; allow_head_application=false)
end

function _sample_afoldl_step_argument(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    options = NamedTuple[]
    grounded = _grounded_derivations(relation, first(rule.argument_tys), scope_tys)
    isempty(grounded) || push!(options, (kind=:grounded_step, value=grounded, weight=1.0))
    depth_left >= 2 && push!(options, (kind=:synthesized_step, value=rule, weight=1.0))

    pool, total_weight = _positive_option_pool(options)
    while !isempty(pool) && total_weight > 0
        chosen = _weighted_option_index(state.rng, pool, total_weight)
        option = pool[chosen]
        saved_nodes = state.nodes_left
        derivation =
            option.kind == :grounded_step ?
            _sample_materialized_derivation(state, relation, option.value) :
            _sample_synthesized_afoldl_step(
                state,
                relation,
                option.value,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
            )
        derivation !== nothing && return derivation
        state.nodes_left = saved_nodes
        total_weight -= option.weight
        deleteat!(pool, chosen)
    end
    return nothing
end

function _sample_synthesized_afoldl_step(
    state::DerivationSamplingState,
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    parts = _afoldl_step_parts(rule)
    parts === nothing && return nothing
    depth_left >= 2 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 2) || return nothing
    body = _sample_generation_derivation(
        state,
        relation,
        parts.body_ty,
        _afoldl_step_scope(rule, scope_tys),
        depth_left - 2,
        guarded_left,
        allow_safe_list_access,
    )
    if body === nothing
        state.nodes_left = saved_nodes
        return nothing
    end
    return LambdaDerivation(parts.acc_ty, LambdaDerivation(parts.elem_ty, body))
end

function _sample_materialized_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    derivations::Vector{<:CoreDerivation},
)
    sample_set = _sample_set(derivations, relation.weights)
    derivation = _sample_derivation(state.rng, sample_set)
    derivation === nothing && return nothing
    _consume_sampling_nodes!(state, _derivation_node_count(derivation)) || return nothing
    return derivation
end

function _consume_sampling_nodes!(state::DerivationSamplingState, count::Int)
    count <= state.nodes_left || return false
    state.nodes_left -= count
    return true
end

_derivation_node_count(::VariableDerivation) = 1
_derivation_node_count(::ConstantDerivation) = 1
_derivation_node_count(d::LambdaDerivation) = 1 + _derivation_node_count(d.body)
_derivation_node_count(d::HeadApplicationDerivation) = 1 + sum(_derivation_node_count, d.args; init=0)
_derivation_node_count(d::ScopeApplicationDerivation) = 1 + sum(_derivation_node_count, d.args; init=0)
_derivation_node_count(d::SafeListAccessDerivation) =
    3 + _derivation_node_count(d.source) + _derivation_node_count(d.body) + _derivation_node_count(d.fallback)

_derivation_weight(::VariableDerivation, weights::GenerationWeights) = _bucket_weight(weights, _VARIABLE_WEIGHT)
_derivation_weight(d::ConstantDerivation, weights::GenerationWeights) = _head_weight(weights, d.head)
_derivation_weight(d::LambdaDerivation, weights::GenerationWeights) = _derivation_weight(d.body, weights)

function _derivation_weight(d::HeadApplicationDerivation, weights::GenerationWeights)
    weight = _head_weight(weights, d.head)
    d.head.type isa Ty || return 0.0
    arg_tys, _ = _split_fun(d.head.type)
    for (arg, arg_ty) in zip(d.args, arg_tys)
        weight *= _derivation_weight(arg, weights)
        _argument_class(arg_ty) == :source_array && (weight *= _bucket_weight(weights, _ARRAY_SOURCE_WEIGHT))
    end
    return weight
end

function _derivation_weight(d::ScopeApplicationDerivation, weights::GenerationWeights)
    weight = _scope_application_weight(weights)
    arg_tys, _ = _split_fun(d.head_ty)
    for (arg, arg_ty) in zip(d.args, arg_tys)
        weight *= _derivation_weight(arg, weights)
        _argument_class(arg_ty) == :source_array && (weight *= _bucket_weight(weights, _ARRAY_SOURCE_WEIGHT))
    end
    return weight
end

function _derivation_weight(d::SafeListAccessDerivation, weights::GenerationWeights)
    return _bucket_weight(weights, _SAFE_LIST_ACCESS_WEIGHT) *
        _derivation_weight(d.source, weights) *
        _derivation_weight(d.body, weights) *
        _derivation_weight(d.fallback, weights)
end

function _relation_catalog(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty};
    include_guarded::Bool=false,
)
    key = (include_guarded ? :guarded_catalog : :core_catalog, _known_key(scope_tys, target_ty))
    cached = get(relation.catalogs, key, nothing)
    cached === nothing || return cached
    providers = include_guarded ? relation.providers : _core_catalog_providers(relation.providers)
    universe = CoreUniverse(
        target_ty;
        global_consts_ty=relation.global_consts_ty,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        scope_source_tys=scope_tys,
    )
    catalog = _build_core_catalog(
        universe;
        global_consts=relation.global_consts,
        global_consts_ty=relation.global_consts_ty,
        local_consts=relation.local_consts,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        scope_source_tys=scope_tys,
        providers,
        catalog_target_ty=target_ty,
    )
    relation.catalogs[key] = catalog
    return catalog
end

_core_catalog_providers(providers::Tuple) =
    Tuple(provider for provider in providers if !(provider isa GuardedExpansionProvider))

function _add_derivation!(relation::CoreRelation, key::Tuple{Any,Any}, derivation::CoreDerivation)
    signature = _derivation_signature(derivation)
    signatures = get!(relation.signatures, key) do
        Set{Any}()
    end
    signature in signatures && return false
    push!(signatures, signature)
    push!(relation.derivations[key], derivation)
    return true
end

_derivation_signature(d::VariableDerivation) = (:var, d.index, typekey(d.ty))
_derivation_signature(d::ConstantDerivation) = (:const, d.head.source, d.head.name, typekey(d.head.type))
_derivation_signature(d::LambdaDerivation) = (:lambda, typekey(d.arg_ty), _derivation_signature(d.body))
_derivation_signature(d::HeadApplicationDerivation) = (
    :head,
    d.head.source,
    d.head.name,
    typekey(d.head.type),
    Tuple(_derivation_signature.(d.args)),
)
_derivation_signature(d::ScopeApplicationDerivation) = (
    :scope_application,
    d.index,
    typekey(d.head_ty),
    Tuple(_derivation_signature.(d.args)),
)
_derivation_signature(d::SafeListAccessDerivation) = (
    :safe_list_access,
    :UNCONS_CASE,
    typekey(d.source_ty),
    typekey(d.result_ty),
    _derivation_signature(d.source),
    _derivation_signature(d.body),
    _derivation_signature(d.fallback),
)

_is_guarded_head(head::HeadEntry) =
    head.rule_class isa Decoration && _metadata_get(head.metadata, :guarded, false) === true

_guarded_depth(::VariableDerivation) = 0
_guarded_depth(::ConstantDerivation) = 0
_guarded_depth(d::LambdaDerivation) = _guarded_depth(d.body)
_guarded_depth(d::SafeListAccessDerivation) =
    max(_guarded_depth(d.source), _guarded_depth(d.body), _guarded_depth(d.fallback))

function _guarded_depth(d::HeadApplicationDerivation)
    child_depth = isempty(d.args) ? 0 : maximum(_guarded_depth, d.args)
    return child_depth + (_is_guarded_head(d.head) ? 1 : 0)
end
_guarded_depth(d::ScopeApplicationDerivation) = isempty(d.args) ? 0 : maximum(_guarded_depth, d.args)

_array_source_depth(::VariableDerivation) = 0
_array_source_depth(::ConstantDerivation) = 0
_array_source_depth(::LambdaDerivation) = typemax(Int)
_array_source_depth(::ScopeApplicationDerivation) = typemax(Int)
_array_source_depth(::SafeListAccessDerivation) = typemax(Int)

function _array_source_depth(d::HeadApplicationDerivation)
    _is_source_array_constructor(d.head) || return typemax(Int)
    isempty(d.args) && return typemax(Int)
    source_idx = _source_array_argument_index(d.head, length(d.args))
    source_depth = _array_source_depth(d.args[source_idx])
    source_depth == typemax(Int) && return source_depth
    return source_depth + 1
end

_signature_contains(sig::Symbol, name::Symbol) = sig == name
_signature_contains(sig::Tuple, name::Symbol) = any(item -> _signature_contains(item, name), sig)
_signature_contains(sig::AbstractVector, name::Symbol) = any(item -> _signature_contains(item, name), sig)
_signature_contains(sig, name::Symbol) = false

function _build_derivation_ast!(
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
    rng::Union{Nothing,Random.AbstractRNG}=nothing,
)
    if derivation isa VariableDerivation
        return IndexNode(derivation.index)
    elseif derivation isa ConstantDerivation
        return _head_node!(derivation.head, local_consts, local_consts_ty, relation, rng)
    elseif derivation isa LambdaDerivation
        return AbsNode(:x, _build_derivation_ast!(derivation.body, local_consts, local_consts_ty, relation, rng))
    elseif derivation isa HeadApplicationDerivation
        node = _head_node!(derivation.head, local_consts, local_consts_ty, relation, rng)
        for arg in derivation.args
            arg_node = _build_derivation_ast!(arg, local_consts, local_consts_ty, relation, rng)
            node = AppNode(node, arg_node)
        end
        return node
    elseif derivation isa ScopeApplicationDerivation
        node = IndexNode(derivation.index)
        for arg in derivation.args
            arg_node = _build_derivation_ast!(arg, local_consts, local_consts_ty, relation, rng)
            node = AppNode(node, arg_node)
        end
        return node
    elseif derivation isa SafeListAccessDerivation
        source_node = _build_derivation_ast!(derivation.source, local_consts, local_consts_ty, relation, rng)
        body_node = _build_derivation_ast!(derivation.body, local_consts, local_consts_ty, relation, rng)
        fallback_node = _build_derivation_ast!(derivation.fallback, local_consts, local_consts_ty, relation, rng)
        head = _safe_uncons_case_head(derivation.source_ty, derivation.result_ty)
        node = _head_node!(head, local_consts, local_consts_ty, relation, rng)
        ok_node = AbsNode(:h, AbsNode(:t, body_node))
        return AppNode(AppNode(AppNode(node, source_node), ok_node), fallback_node)
    else
        error("Unknown core derivation $(typeof(derivation))")
    end
end

function _materialize_derivation(
    materializer,
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
    ::Random.AbstractRNG,
)
    return materializer(derivation, local_consts, local_consts_ty, relation)
end

function _materialize_derivation(
    ::DefaultAstMaterializer,
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
    rng::Random.AbstractRNG,
)
    return _build_derivation_ast!(derivation, local_consts, local_consts_ty, relation, rng)
end

(::DefaultAstMaterializer)(
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
) = _build_derivation_ast!(derivation, local_consts, local_consts_ty, relation)

function _head_has_value(head::HeadEntry, relation::CoreRelation)
    _metadata_has(head.metadata, :value) && return true
    _metadata_has(head.metadata, :factory) && return true
    head.source == :constant && return haskey(relation.global_consts, head.name)
    head.source == :variable && return haskey(relation.local_consts, head.name)
    head.source == :preset && return haskey(relation.presets, head.name)
    return false
end

_is_instance_factory_head(head::HeadEntry) = _metadata_has(head.metadata, :factory) || head.source == :instance_factory

function _head_node!(
    head::HeadEntry,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
    rng::Union{Nothing,Random.AbstractRNG}=nothing,
)
    factory = _metadata_get(head.metadata, :factory, nothing)
    if factory !== nothing
        name = _fresh_local_name(local_consts_ty, head.name)
        local_consts[name] = _call_head_factory(factory, head, relation, rng)
        local_consts_ty[name] = head.type
        return ConstNode(name)
    end

    name = _head_materialization_name(head, local_consts_ty)
    local_consts_ty[name] = head.type
    if _metadata_has(head.metadata, :value)
        local_consts[name] = _metadata_get(head.metadata, :value, nothing)
    elseif head.source == :constant && haskey(relation.global_consts, head.name)
        local_consts[name] = relation.global_consts[head.name]
    elseif head.source == :variable && haskey(relation.local_consts, head.name)
        local_consts[name] = relation.local_consts[head.name]
    elseif head.source == :preset && haskey(relation.presets, head.name)
        local_consts[name] = relation.presets[head.name]
    else
        error("canonical head $(head.name) from source $(head.source) has no materializable value")
    end
    return ConstNode(name)
end

function _call_head_factory(factory, head::HeadEntry, relation::CoreRelation, rng::Union{Nothing,Random.AbstractRNG})
    runtime_rng = rng === nothing ? Random.default_rng() : rng
    applicable(factory, head, relation, runtime_rng) && return factory(head, relation, runtime_rng)
    applicable(factory, head, relation) && return factory(head, relation)
    throw(ArgumentError("canonical head factory for $(head.name) must accept (head, relation) or (head, relation, rng)"))
end

function _head_materialization_name(head::HeadEntry, local_consts_ty::Dict{Symbol,TyLike})
    existing = get(local_consts_ty, head.name, nothing)
    existing === nothing && return head.name
    existing isa Ty && head.type isa Ty && _closed_type_key(existing) == _closed_type_key(head.type) && return head.name
    return _fresh_local_name(local_consts_ty, head.name)
end

_metadata_has(metadata::NamedTuple, key::Symbol) = key in keys(metadata)
_metadata_get(metadata::NamedTuple, key::Symbol, default) =
    _metadata_has(metadata, key) ? getfield(metadata, key) : default

_weight_tag_tuple(tag::Symbol) = (tag,)
_weight_tag_tuple(tags) = Tuple(Symbol(tag) for tag in tags)

function _with_head_weight_tags(head::HeadEntry, tags)
    merged_tags = _merge_weight_tags(_metadata_get(head.metadata, :weight_tags, ()), _weight_tag_tuple(tags))
    metadata = merge(head.metadata, (weight_tags=merged_tags,))
    return HeadEntry(head.name, head.source, head.type, head.rule_class, head.weight_bucket; metadata)
end

function _merge_weight_tags(existing, added)
    out = Symbol[]
    seen = Set{Symbol}()
    for tag in _weight_tag_tuple(existing)
        tag in seen && continue
        push!(seen, tag)
        push!(out, tag)
    end
    for tag in _weight_tag_tuple(added)
        tag in seen && continue
        push!(seen, tag)
        push!(out, tag)
    end
    return Tuple(out)
end

function _fresh_local_name(local_consts_ty::Dict{Symbol,TyLike}, prefix::Symbol)
    i = 1
    while true
        name = Symbol(prefix, i)
        haskey(local_consts_ty, name) || return name
        i += 1
    end
end

@testitem "CanonicalGeneration root API smoke" begin
    using Test
    using LambdaRegression

    @test LambdaRegression.CanonicalGeneration isa Module
    @test LambdaRegression.mutate_tree isa Function
    @test LambdaRegression.fit! isa Function
    @test LambdaRegression.TypeSystem isa Module
    @test LambdaRegression.CanonicalGeneration.generate_canonical isa Function
    @test LambdaRegression.CanonicalGeneration.GenerationWeights isa DataType
    @test LambdaRegression.CanonicalGeneration.GenerationConfig() isa LambdaRegression.CanonicalGeneration.GenerationConfig
    @test LambdaRegression.CanonicalGeneration.GenerationResult isa DataType
    @test LambdaRegression.CanonicalGeneration.default_head_providers isa Function
    @test LambdaRegression.CanonicalGeneration.ContextHeadProvider() isa LambdaRegression.CanonicalGeneration.AbstractHeadProvider
    @test LambdaRegression.generate_canonical isa Function
    @test LambdaRegression.GenerationWeights isa DataType
    @test LambdaRegression.GenerationConfig() isa LambdaRegression.GenerationConfig
    @test LambdaRegression.GenerationResult isa DataType
end

@testitem "CanonicalGeneration scoped generation can use binder variables" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: Ty, FloatTy, TyFun, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    result = CG.generate_canonical_scoped(
        FloatTy;
        scope_tys=Ty[FloatTy],
        rng=Random.MersenneTwister(2000),
        weights=CG.GenerationWeights(default=0, variable=1),
        sampling_depth_limit=1,
        sampling_node_budget=4,
    )

    @test result isa CG.GenerationResult
    @test result.tree.root == IndexNode(0)
    wrapped = AstTree(AbsNode(:x, result.tree.root))
    @test iscompatible(infer(wrapped, result.local_const_types), TyFun(FloatTy, FloatTy))
end

@testitem "CanonicalGeneration catalog closed fragment accepts finite core types" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, →

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    Float4Fn = FloatTy → FloatTy → FloatTy → FloatTy → FloatTy

    @test CG._in_closed_fragment(FloatTy)
    @test CG._in_closed_fragment(FloatTy → FloatTy)
    @test CG._in_closed_fragment(Matrix32)
    @test CG._in_closed_fragment(Float4Fn)
end

@testitem "CanonicalGeneration cache key distinguishes shared array results" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    empty_values = Dict{Symbol,Any}()
    empty_types = Dict{Symbol,Any}()

    matrix_key = CG._default_relation_cache_key(
        TyFun(Matrix32, Matrix32),
        CG.GenerationWeights(),
        empty_values,
        empty_types,
        empty_values,
        empty_types,
        empty_values,
        empty_types,
        CG.default_head_providers(),
        CG.DefaultArgumentPlanner(),
    )
    float_key = CG._default_relation_cache_key(
        TyFun(Matrix32, FloatTy),
        CG.GenerationWeights(),
        empty_values,
        empty_types,
        empty_values,
        empty_types,
        empty_values,
        empty_types,
        CG.default_head_providers(),
        CG.DefaultArgumentPlanner(),
    )

    @test matrix_key != float_key

    catalog_key = CG._default_relation_cache_key(
        TyFun(Matrix32, Matrix32),
        CG.GenerationWeights(),
        CG.test_catalog_consts(),
        empty_types,
        empty_values,
        empty_types,
        empty_values,
        empty_types,
        CG.default_head_providers(),
        CG.DefaultArgumentPlanner(),
        2,
    )
    @test catalog_key !== nothing
end

@testitem "CanonicalGeneration scheme specialization separates length and value domains" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyNat, TyScheme, TyVar, typekey

    CG = LambdaRegression.CanonicalGeneration
    n = TyVar(120_001, 0, nothing)
    a = TyVar(120_002, 0, nothing)
    Vec3 = TyArray(TyNat(3), FloatTy)
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    scheme = TyScheme([n, a], TyFun(TyArray(n, a), FloatTy))
    universe = CG._extend_core_universe(
        CG.CoreUniverse(TyFun(Matrix32, FloatTy)),
        CG.default_head_providers();
        global_consts=CG.test_catalog_consts(),
    )
    instances = CG._finite_scheme_instances(scheme, universe)

    @test !isempty(instances)
    @test all(inst -> inst isa TyFun && inst.arg isa TyArray && inst.arg.len isa TyNat, instances)
    @test any(inst -> typekey(inst.arg) == typekey(Vec3), instances)
    @test any(inst -> typekey(inst.arg) == typekey(Matrix32), instances)
end

@testitem "CanonicalGeneration rigid scheme demands generate through closed wrapper" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AppNode, ConstNode
    using LambdaRegression.TypeSystem: TyVar, TyScheme, TyArray, TyNat, TyList, BoolTy, TRUE, FALSE
    using LambdaRegression.TypeSystem: infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration

    contains_app(node) = node isa AppNode || (
        node isa LambdaRegression.ASTs.NodeASTs.AbsNode && contains_app(node.body)
    ) || (
        node isa AppNode && (contains_app(node.func) || contains_app(node.arg))
    )
    function contains_const(node, name::Symbol)
        node isa ConstNode && return node.name == name
        node isa LambdaRegression.ASTs.NodeASTs.AbsNode && return contains_const(node.body, name)
        node isa AppNode && return contains_const(node.func, name) || contains_const(node.arg, name)
        return false
    end
    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    α = TyVar(221_001, 0, nothing)
    id_scheme = TyScheme([α], α → α)
    id_result = CG.generate_canonical(
        id_scheme;
        rng=Random.MersenneTwister(1),
        weights=CG.GenerationWeights(default=0, lambda=1, variable=1),
        sampling_depth_limit=8,
        sampling_node_budget=64,
    )
    @test id_result isa CG.GenerationResult
    @test id_result.target_ty == id_scheme
    @test get(id_result.metadata, :rigid_scheme, false) === true
    @test !occursin("canonical_rigid", sprint(show, id_result.local_const_types))
    @test !occursin("canonical_rigid", sprint(show, id_result.metadata))
    @test iscompatible(infer(id_result.tree, id_result.local_const_types), id_scheme)

    η = TyVar(221_007, 0, nothing)
    list_scheme = TyScheme([η], TyList(η))
    list_result = CG.generate_canonical(
        list_scheme;
        rng=Random.MersenneTwister(555),
        weights=CG.GenerationWeights(default=0, instance_factory=1),
        sampling_depth_limit=4,
        sampling_node_budget=32,
    )
    @test list_result isa CG.GenerationResult
    @test !occursin("canonical_rigid", sprint(show, list_result.local_const_types))
    @test iscompatible(infer(list_result.tree, list_result.local_const_types), list_scheme)

    β = TyVar(221_002, 0, nothing)
    γ = TyVar(221_003, 0, nothing)
    pipe3 = TyScheme([α, β, γ], α → (α → β) → (β → γ) → γ)
    pipe_result = CG.generate_canonical(
        pipe3;
        rng=Random.MersenneTwister(1111),
        weights=CG.GenerationWeights(default=0, lambda=1, variable=1),
        sampling_depth_limit=12,
        sampling_node_budget=96,
    )
    @test pipe_result isa CG.GenerationResult
    @test iscompatible(infer(pipe_result.tree, pipe_result.local_const_types), pipe3)
    @test contains_app(pipe_result.tree.root)

    pipe_scope_config = CG.GenerationConfig(
        rng=Random.MersenneTwister(1234),
        weights=CG.GenerationWeights(default=0, lambda=1, variable=1),
        sampler=pick_named(:scope_application),
        sampling_depth_limit=12,
        sampling_node_budget=96,
    )
    pipe_scope_result = CG.generate_canonical(pipe3; config=pipe_scope_config)
    @test pipe_scope_result isa CG.GenerationResult
    @test CG._signature_contains(pipe_scope_result.metadata[:derivation], :scope_application)
    @test iscompatible(infer(pipe_scope_result.tree, pipe_scope_result.local_const_types), pipe3)

    δ = TyVar(221_004, 0, nothing)
    RigidMatrix32 = TyArray(TyNat(3), TyArray(TyNat(2), δ))
    matrix_scheme = TyScheme([δ], RigidMatrix32 → RigidMatrix32)
    matrix_result = CG.generate_canonical(
        matrix_scheme;
        rng=Random.MersenneTwister(1234),
        weights=CG.GenerationWeights(default=0, lambda=1, variable=1),
        sampling_depth_limit=8,
        sampling_node_budget=64,
    )
    @test matrix_result isa CG.GenerationResult
    @test isempty(matrix_result.local_consts)
    @test iscompatible(infer(matrix_result.tree, matrix_result.local_const_types), matrix_scheme)

    ε = TyVar(221_005, 0, nothing)
    impossible = CG.generate_canonical(
        TyScheme([ε], ε);
        rng=Random.MersenneTwister(8000),
        providers=CG.test_catalog_providers(),
        weights=CG.test_catalog_weights(profile=:deeper),
        sampling_depth_limit=12,
        sampling_node_budget=128,
    )
    @test impossible isa LambdaRegression.Utils.Err

    struct BadRigidProvider <: CG.AbstractHeadProvider end
    function CG.add_heads!(::BadRigidProvider, heads, universe; kwargs...)
        for ty in universe.types
            CG._contains_rigid_dummy(ty) || continue
            push!(
                heads,
                CG.HeadEntry(
                    :BAD_RIGID,
                    :provider_value,
                    ty,
                    CG.CoreProductive(),
                    CG.WeightBucket(:bad_rigid);
                    metadata=(value=:bad,),
                ),
            )
            return heads
        end
        return heads
    end
    κ = TyVar(221_010, 0, nothing)
    adversarial = CG.generate_canonical(
        TyScheme([κ], κ);
        rng=Random.MersenneTwister(1),
        providers=(BadRigidProvider(),),
        weights=CG.GenerationWeights(default=0, bad_rigid=1),
        sampling_depth_limit=4,
        sampling_node_budget=32,
    )
    @test adversarial isa LambdaRegression.Utils.Err

    struct NameCollisionRigidProvider <: CG.AbstractHeadProvider end
    function CG.add_heads!(::NameCollisionRigidProvider, heads, universe; kwargs...)
        for ty in universe.types
            CG._contains_rigid_dummy(ty) || continue
            push!(
                heads,
                CG.HeadEntry(
                    :TOLIST,
                    :provider_value,
                    ty,
                    CG.CoreProductive(),
                    CG.WeightBucket(:bad_rigid);
                    metadata=(value=:bad,),
                ),
            )
            return heads
        end
        return heads
    end
    λ = TyVar(221_011, 0, nothing)
    name_collision = CG.generate_canonical(
        TyScheme([λ], λ);
        rng=Random.MersenneTwister(1111),
        providers=(NameCollisionRigidProvider(),),
        weights=CG.GenerationWeights(default=0, bad_rigid=1),
        sampling_depth_limit=4,
        sampling_node_budget=32,
    )
    @test name_collision isa LambdaRegression.Utils.Err

    ℓ = TyVar(221_008, 0, nothing)
    θ = TyVar(221_009, 0, nothing)
    length_poly = TyScheme([ℓ, θ], TyArray(ℓ, θ) → TyArray(ℓ, θ))
    @test CG.generate_canonical(length_poly; rng=Random.MersenneTwister(2000)) isa LambdaRegression.Utils.Err

    ζ = TyVar(221_006, 0, nothing)
    branch_scheme = TyScheme([ζ], ζ → ζ → ζ)
    branch_consts = merge(CG.test_catalog_consts(), Dict{Symbol,Any}(:TRUE => TRUE, :FALSE => FALSE))
    branch_config = CG.GenerationConfig(
        rng=Random.MersenneTwister(42),
        providers=CG.test_catalog_providers(),
        weights=CG.GenerationWeights(;
            default=0,
            lambda=1,
            variable=1,
            constant=1,
            guarded_expansion=1.0e6,
            cond=1.0e6,
            multi_arg=2,
        ),
        sampler=pick_named(:COND),
        guarded_expansion_depth=3,
        sampling_depth_limit=12,
        sampling_node_budget=160,
    )
    branch_result = CG.generate_canonical(branch_scheme; global_consts=branch_consts, config=branch_config)
    @test branch_result isa CG.GenerationResult
    @test iscompatible(infer(branch_result.tree, branch_result.local_const_types), branch_scheme)
    @test contains_const(branch_result.tree.root, :COND)
    @test !occursin("canonical_rigid", sprint(show, branch_result.local_const_types))
    @test !occursin("canonical_rigid", sprint(show, branch_result.metadata))
end

@testitem "CanonicalGeneration default core omits LENF route" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: infer, iscompatible
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    target = TyFun(Matrix32, FloatTy)

    result = CG.generate_canonical(target; rng=Random.MersenneTwister(8000))

    @test result isa LambdaRegression.Utils.Err

    extension_result = CG.generate_canonical(
        target;
        rng=Random.MersenneTwister(8000),
        global_consts=CG.test_catalog_consts(),
    )

    @test extension_result isa CG.GenerationResult
    @test iscompatible(infer(extension_result.tree, extension_result.local_const_types), target)
end

@testitem "CanonicalGeneration providers can inject typed value heads" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration

    struct InjectedFloatProvider <: CG.AbstractHeadProvider end
    function CG.add_heads!(::InjectedFloatProvider, heads, universe; kwargs...)
        CG._type_in_universe(universe, FloatTy) || return heads
        push!(
            heads,
            CG.HeadEntry(
                :CUSTOM_FLOAT,
                :provider_value,
                FloatTy,
                CG.CoreProductive(),
                CG.WeightBucket(:custom_float);
                metadata=(value=3.0,),
            ),
        )
        return heads
    end

    result = CG.generate_canonical(
        FloatTy;
        rng=Random.MersenneTwister(42),
        providers=(InjectedFloatProvider(),),
        weights=CG.GenerationWeights(custom_float=1),
    )

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test result.tree.root.name == :CUSTOM_FLOAT
    @test result.local_consts[:CUSTOM_FLOAT] == 3.0
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration providers can extend finite universe" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration

    struct UniverseExtendingProvider <: CG.AbstractHeadProvider end
    function CG.add_universe_types!(::UniverseExtendingProvider, types, keys, universe; kwargs...)
        CG.add_universe_type!(types, keys, BoolTy)
        CG.add_universe_type!(types, keys, FloatTy)
        return types
    end
    function CG.add_heads!(::UniverseExtendingProvider, heads, universe; kwargs...)
        CG._type_in_universe(universe, BoolTy) || return heads
        CG._type_in_universe(universe, FloatTy) || return heads
        push!(
            heads,
            CG.HeadEntry(:TRUE_VALUE, :provider_value, BoolTy, CG.CoreProductive(), CG.WeightBucket(:provider_bool); metadata=(value=true,)),
        )
        push!(
            heads,
            CG.HeadEntry(
                :BOOL_TO_FLOAT,
                :provider_value,
                BoolTy → FloatTy,
                CG.CoreProductive(),
                CG.WeightBucket(:provider_fn);
                metadata=(value=(x -> (x ? 1.0 : 0.0)),),
            ),
        )
        return heads
    end

    result = CG.generate_canonical(
        FloatTy;
        rng=Random.MersenneTwister(555),
        providers=(UniverseExtendingProvider(),),
    )

    @test result isa CG.GenerationResult
    @test haskey(result.local_consts, :TRUE_VALUE)
    @test haskey(result.local_consts, :BOOL_TO_FLOAT)
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration head metadata can override argument planning" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    called = Ref(false)

    struct PlanningProvider <: CG.AbstractHeadProvider
        called::Base.RefValue{Bool}
    end
    function CG.add_heads!(provider::PlanningProvider, heads, universe; kwargs...)
        CG._type_in_universe(universe, FloatTy) || return heads
        planner = (derivations, arg_ty, arg_class, rule, relation) -> begin
            provider.called[] = true
            [derivation for derivation in derivations if !(derivation isa CG.HeadApplicationDerivation)]
        end
        push!(
            heads,
            CG.HeadEntry(:ONE_VALUE, :provider_value, FloatTy, CG.CoreProductive(), CG.WeightBucket(:one); metadata=(value=1.0,)),
        )
        push!(
            heads,
            CG.HeadEntry(
                :ID_FLOAT,
                :provider_value,
                FloatTy → FloatTy,
                CG.CoreProductive(),
                CG.WeightBucket(:id_float);
                metadata=(value=identity, argument_planner=planner),
            ),
        )
        return heads
    end

    result = CG.generate_canonical(FloatTy; rng=Random.MersenneTwister(2000), providers=(PlanningProvider(called),))

    @test result isa CG.GenerationResult
    @test called[]
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration catalog closed fragment rejects open and excluded types" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyMu, TyProd, TySum, TyVar

    CG = LambdaRegression.CanonicalGeneration
    α = TyVar(10_001, 0, nothing)

    @test !CG._in_closed_fragment(TyProd(FloatTy, FloatTy))
    @test !CG._in_closed_fragment(TySum(FloatTy, FloatTy))
    @test !CG._in_closed_fragment(TyMu(α, FloatTy))
    @test !CG._in_closed_fragment(α)
    @test !CG._in_closed_fragment(TyArray(α, FloatTy))
end

@testitem "CanonicalGeneration context constants can include real operators and closed array LENF" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(FloatTy; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    catalog = CG._build_core_catalog(
        universe;
        global_consts=CG.test_catalog_consts(),
        local_consts_ty=Dict{Symbol,Any}(:m => Matrix32),
    )
    names = Set(entry.name for entry in catalog.heads)

    @test all(name -> name in names, [:FSIN, :FSQRT, :SQR, :FADD, :FSUB, :FMUL, :FDIV])
    @test :LENF in names
end

@testitem "CanonicalGeneration context LENF requires closed array source" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, →

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(FloatTy)
    catalog = CG._build_core_catalog(universe; global_consts=CG.test_catalog_consts())
    names = Set(entry.name for entry in catalog.heads)

    @test :LENF ∉ names

    function_universe = CG.CoreUniverse(Matrix32 → FloatTy)
    function_catalog = CG._build_core_catalog(function_universe; global_consts=CG.test_catalog_consts())
    function_names = Set(entry.name for entry in function_catalog.heads)

    @test :LENF in function_names

    scoped = CG._build_core_catalog(universe; global_consts=CG.test_catalog_consts(), scope_source_tys=[Matrix32])
    scoped_names = Set(entry.name for entry in scoped.heads)

    @test :LENF in scoped_names

    body_universe = CG.CoreUniverse(FloatTy; scope_source_tys=[Matrix32])
    body_catalog = CG._build_core_catalog(body_universe; global_consts=CG.test_catalog_consts(), scope_source_tys=[Matrix32])
    body_names = Set(entry.name for entry in body_catalog.heads)

    @test :LENF in body_names
end

@testitem "CanonicalGeneration catalog keeps guarded fold separate from unsupported partial primitives" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(Matrix32; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    catalog = CG._build_core_catalog(universe; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    names = Set(entry.name for entry in catalog.heads)

    @test :AFOLDL in names
    @test :HEAD ∉ names
    @test :TAIL ∉ names
    @test all(rule -> rule.head.name != :AFOLDL, catalog.rules)
end

@testitem "CanonicalGeneration catalog TOLIST and AMAP stay finite and classified" begin
    using Test
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyArray, TyList, TyNat, →
    using LambdaRegression.TypeSystem: TyFun, typekey

    CG = LambdaRegression.CanonicalGeneration
    Vec3 = TyArray(TyNat(3), FloatTy)
    BoolVec3 = TyArray(TyNat(3), BoolTy)

    tolist_target = Vec3 → TyList(FloatTy)
    tolist_universe = CG.CoreUniverse(tolist_target)
    tolist_catalog = CG._build_core_catalog(tolist_universe; scope_source_tys=[Vec3])
    tolist_names = Set(entry.name for entry in tolist_catalog.heads)

    @test :TOLIST in tolist_names

    self_target = Vec3 → Vec3
    self_universe = CG.CoreUniverse(self_target)
    self_catalog = CG._build_core_catalog(self_universe; scope_source_tys=[Vec3])
    self_amap_ty = TyFun(TyFun(FloatTy, FloatTy), TyFun(Vec3, Vec3))
    self_amaps = [entry for entry in self_catalog.heads if entry.name == :AMAP && typekey(entry.type) == typekey(self_amap_ty)]

    @test any(entry -> entry.rule_class isa CG.CoreProductive, self_amaps)
    @test any(rule -> rule.head.name == :AMAP && typekey(rule.head.type) == typekey(self_amap_ty), self_catalog.rules)

    productive_universe = CG.CoreUniverse(
        BoolVec3;
        local_consts_ty=Dict{Symbol,Any}(:xs => Vec3, :p => FloatTy → BoolTy),
    )
    productive_catalog = CG._build_core_catalog(
        productive_universe;
        local_consts_ty=Dict{Symbol,Any}(:xs => Vec3, :p => FloatTy → BoolTy),
        scope_source_tys=[Vec3],
    )

    @test any(rule -> rule.head.name == :AMAP, productive_catalog.rules)
end

@testitem "CanonicalGeneration same-type AMAP can generate witnesses" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyLike, TyNat, infer, iscompatible, →
    using LambdaRegression.TypeSystem: TyFun

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    target = TyFun(Matrix32, Matrix32)

    function find_self_amap_head(relation)
        catalog = CG._relation_catalog(relation, Matrix32, Ty[Matrix32])
        self_amap_ty = TyFun(TyFun(Vec2, Vec2), TyFun(Matrix32, Matrix32))
        for head in catalog.heads
            head.name == :AMAP || continue
            head.type == self_amap_ty || continue
            return head
        end
        return nothing
    end

    relation = CG.CoreRelation(; guarded_expansion_depth=0)
    amap_head = find_self_amap_head(relation)
    @test amap_head !== nothing
    amap_head === nothing && return

    derivation = CG.LambdaDerivation(
        Matrix32,
        CG.HeadApplicationDerivation(
            amap_head,
            CG.CoreDerivation[
                CG.LambdaDerivation(Vec2, CG.VariableDerivation(0, Vec2)),
                CG.VariableDerivation(0, Matrix32),
            ],
        ),
    )
    local_consts = Dict{Symbol,Any}()
    local_const_types = Dict{Symbol,TyLike}()
    tree = AstTree(CG._build_derivation_ast!(derivation, local_consts, local_const_types, relation))

    @test CG._signature_contains(CG._derivation_signature(derivation), :AMAP)
    @test iscompatible(infer(tree, local_const_types), target)
end

@testitem "CanonicalGeneration lazy sampler can select same-type AMAP" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    relation = CG.CoreRelation(;
        weights=CG.GenerationWeights(; variable=0.1, amap=25, guarded_expansion=0),
        guarded_expansion_depth=0,
    )
    sampled = Ref{Any}(nothing)
    for seed in 1:16
        state = CG.DerivationSamplingState(Random.MersenneTwister(seed), 64)
        derivation = CG._sample_generation_derivation(state, relation, Matrix32, Ty[Matrix32], 6, 0)
        derivation === nothing && continue
        CG._signature_contains(CG._derivation_signature(derivation), :AMAP) || continue
        sampled[] = derivation
        break
    end

    @test sampled[] !== nothing
end

@testitem "CanonicalGeneration anchored AMAP uses grounded known sources" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyArray, TyList, TyNat, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    Vec3 = TyArray(TyNat(3), FloatTy)
    BoolVec3 = TyArray(TyNat(3), BoolTy)

    impossible = CG.generate_canonical(
        BoolVec3;
        global_consts=Dict{Symbol,Any}(:FLE => LambdaRegression.TypeSystem.FLE),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(42),
            weights=CG.GenerationWeights(; instance_factory=0),
        ),
    )
    @test impossible isa LambdaRegression.Utils.Err

    target = Vec3 → TyList(BoolTy)
    result = CG.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(:FLE => LambdaRegression.TypeSystem.FLE),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(555),
            weights=CG.GenerationWeights(; instance_factory=0),
        ),
    )

    @test result isa CG.GenerationResult
    @test iscompatible(infer(result.tree, result.local_const_types), target)
    @test CG._signature_contains(result.metadata[:derivation], :AMAP)
    @test CG._signature_contains(result.metadata[:derivation], :TOLIST)
end

@testitem "CanonicalGeneration guarded expansions can be sampled after core success" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyList, TyNat, TyLike, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    Vec3 = TyArray(TyNat(3), FloatTy)

    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    cond_target = FloatTy → FloatTy
    cond_result = CG.generate_canonical(
        cond_target;
        global_consts=Dict{Symbol,Any}(:FEQ => LambdaRegression.TypeSystem.FEQ),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(2000),
            sampler=pick_named(:COND),
        ),
    )
    @test cond_result isa CG.GenerationResult
    @test iscompatible(infer(cond_result.tree, cond_result.local_const_types), cond_target)
    @test CG._signature_contains(cond_result.metadata[:derivation], :COND)

    fold_target = Vec3 → FloatTy
    fold_consts = Dict{Symbol,Any}(
        :LENF => LambdaRegression.TypeSystem.LENF,
        :FADD => LambdaRegression.TypeSystem.FADD,
    )
    fold_result = CG.generate_canonical(
        fold_target;
        global_consts=fold_consts,
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(1),
            sampler=pick_named(:AFOLDL),
        ),
    )
    @test fold_result isa CG.GenerationResult
    @test iscompatible(infer(fold_result.tree, fold_result.local_const_types), fold_target)
    @test CG._signature_contains(fold_result.metadata[:derivation], :AFOLDL)

    list_target = TyList(FloatTy)
    cons_result = CG.generate_canonical(
        list_target;
        local_consts=Dict{Symbol,Any}(:x => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:x => FloatTy),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(1111),
            sampler=pick_named(:CONS),
        ),
    )
    @test cons_result isa CG.GenerationResult
    @test iscompatible(infer(cons_result.tree, cons_result.local_const_types), list_target)
    @test CG._signature_contains(cons_result.metadata[:derivation], :CONS)
end

@testitem "CanonicalGeneration safe list access lowers local list sources" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyList, TyLike, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    target = TyList(FloatTy) → FloatTy

    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    result = CG.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(1111),
            providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
            sampler=pick_named(:UNCONS_CASE),
            sampling_depth_limit=8,
            sampling_node_budget=96,
        ),
    )

    @test result isa CG.GenerationResult
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test iscompatible(infer(result.tree, result.local_const_types), target)
    @test haskey(result.local_const_types, :UNCONS_CASE)
end

@testitem "CanonicalGeneration safe list access uses anchored TOLIST matrix sources" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, TyLike, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    target = Matrix32 → FloatTy

    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    result = CG.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(
            :zero => 0.0,
            :LENF => LambdaRegression.TypeSystem.LENF,
        ),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(1234),
            providers=(CG.ContextHeadProvider(), CG.ToListProvider(), CG.SafeListAccessProvider()),
            sampler=pick_named(:UNCONS_CASE),
            sampling_depth_limit=10,
            sampling_node_budget=128,
        ),
    )

    @test result isa CG.GenerationResult
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test CG._signature_contains(result.metadata[:derivation], :TOLIST)
    @test iscompatible(infer(result.tree, result.local_const_types), target)
end

@testitem "CanonicalGeneration safe list access is not an ordinary searchable head" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyList

    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(; providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()))
    catalog = CG._relation_catalog(relation, FloatTy, Ty[TyList(FloatTy)])
    names = Set(head.name for head in catalog.heads)

    @test :UNCONS_CASE ∉ names
    @test :UNCONS ∉ names
    @test :HEAD ∉ names
    @test :TAIL ∉ names
end

@testitem "CanonicalGeneration lazy sampler can select safe list access" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyList, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    target = TyList(FloatTy) → FloatTy
    result = CG.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(8000),
            providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
            weights=CG.GenerationWeights(; constant=1, variable=1, lambda=1, safe_list_access=1.0e6),
            sampling_depth_limit=8,
            sampling_node_budget=96,
        ),
    )

    @test result isa CG.GenerationResult
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test iscompatible(infer(result.tree, result.local_const_types), target)
end

@testitem "CanonicalGeneration safe list access is independent of guarded expansion" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyList, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    target = TyList(FloatTy) → FloatTy

    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    result = CG.generate_canonical(
        target;
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(42),
            providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
            sampler=pick_named(:UNCONS_CASE),
            guarded_expansion_depth=0,
            sampling_depth_limit=8,
            sampling_node_budget=96,
        ),
    )

    @test result isa CG.GenerationResult
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test iscompatible(infer(result.tree, result.local_const_types), target)
end

@testitem "CanonicalGeneration safe list access disabled flag reaches lambda bodies" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyList, →

    CG = LambdaRegression.CanonicalGeneration
    target = TyList(FloatTy) → FloatTy
    relation = CG.CoreRelation(;
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        weights=CG.GenerationWeights(; constant=1, variable=1, lambda=1, safe_list_access=1.0e9),
    )
    state = CG.DerivationSamplingState(Random.MersenneTwister(555), 96)
    derivation = CG._sample_generation_derivation(state, relation, target, Ty[TyList(FloatTy)], 8, 0, false)

    @test derivation !== nothing
    @test !CG._signature_contains(CG._derivation_signature(derivation), :UNCONS_CASE)
end

@testitem "CanonicalGeneration safe list access respects source weights" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    providers = (CG.ContextHeadProvider(), CG.ToListProvider(), CG.SafeListAccessProvider())

    no_tolist = CG.CoreRelation(; providers, weights=CG.GenerationWeights(; tolist=0, safe_list_access=1))
    tolist_options = CG._safe_list_access_generation_options(no_tolist, FloatTy, Ty[Matrix32])
    @test all(option -> option.weight == 0, tolist_options)

    no_array_source = CG.CoreRelation(; providers, weights=CG.GenerationWeights(; array_source=0, safe_list_access=1))
    array_options = CG._safe_list_access_generation_options(no_array_source, FloatTy, Ty[Matrix32])
    @test all(option -> option.weight == 0, array_options)
end

@testitem "CanonicalGeneration safe list access accepts injected source collectors" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyList, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration

    struct StaticSafeListCollector
        head::CG.HeadEntry
    end
    (collector::StaticSafeListCollector)(relation, scope_tys) = (collector.head,)

    custom_head = CG.HeadEntry(
        :CUSTOM_LIST_SOURCE,
        :test,
        TyList(FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:custom_list_source);
        metadata=(value=[1.0, 2.0, 3.0],),
    )

    function pick_named(name)
        return (rng, sample_set) -> begin
            for derivation in sample_set.derivations
                CG._signature_contains(CG._derivation_signature(derivation), name) && return derivation
            end
            return nothing
        end
    end

    result = CG.generate_canonical(
        FloatTy;
        global_consts=Dict{Symbol,Any}(:zero => 0.0),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(2000),
            providers=(
                CG.ContextHeadProvider(),
                CG.SafeListAccessProvider(; source_collectors=(StaticSafeListCollector(custom_head),)),
            ),
            weights=CG.GenerationWeights(; custom_list_source=1, safe_list_access=1),
            sampler=pick_named(:UNCONS_CASE),
            guarded_expansion_depth=0,
            sampling_depth_limit=8,
            sampling_node_budget=96,
        ),
    )

    @test result isa CG.GenerationResult
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test CG._signature_contains(result.metadata[:derivation], :CUSTOM_LIST_SOURCE)
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration proxy body IR can force source-array proxy use" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyList, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    relation = CG.CoreRelation(;
        global_consts=CG.test_catalog_consts(),
        providers=CG.test_catalog_providers(),
        weights=CG.test_catalog_weights(profile=:deeper),
        guarded_expansion_depth=4,
    )
    scope = Ty[TyList(Vec2), Vec2, Matrix32]
    state = CG.DerivationSamplingState(Random.MersenneTwister(1), 96)

    body = CG._sample_proxy_using_derivation(state, relation, Matrix32, scope, 8, 4, 2, 0)

    @test body !== nothing
    @test CG._uses_scope_prefix(body, 2)
end

@testitem "CanonicalGeneration guarded AFOLDL synthesizes nested matrix steps" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyLike, TyNat, infer, iscompatible, typekey, →

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    target = Matrix32 → FloatTy
    fold_consts = Dict{Symbol,Any}(
        :FADD => LambdaRegression.TypeSystem.FADD,
        :LENF => LambdaRegression.TypeSystem.LENF,
    )

    function signature_count(sig, name::Symbol)
        sig == name && return 1
        sig isa Tuple && return sum(item -> signature_count(item, name), sig; init=0)
        sig isa AbstractVector && return sum(item -> signature_count(item, name), sig; init=0)
        return 0
    end

    function find_afoldl_rule(relation, result_ty, scope, source_ty)
        catalog = CG._relation_catalog(relation, result_ty, scope; include_guarded=true)
        for head in catalog.heads
            CG._is_guarded_head(head) || continue
            head.name == :AFOLDL || continue
            head.type isa Ty || continue
            args, head_result_ty = CG._split_fun(head.type)
            length(args) == 3 || continue
            typekey(head_result_ty) == typekey(result_ty) || continue
            typekey(args[3]) == typekey(source_ty) || continue
            return CG.CoreRule(head, head_result_ty, args, [CG._argument_class(arg) for arg in args])
        end
        return nothing
    end

    function find_lenf_app(relation)
        catalog = CG._relation_catalog(relation, FloatTy, Ty[Matrix32])
        for rule in catalog.rules
            rule.head.name == :LENF || continue
            length(rule.argument_tys) == 1 || continue
            typekey(rule.argument_tys[1]) == typekey(Matrix32) || continue
            return CG.HeadApplicationDerivation(rule.head, CG.CoreDerivation[CG.VariableDerivation(0, Matrix32)])
        end
        return nothing
    end

    function find_fadd_step(relation)
        step_ty = FloatTy → FloatTy → FloatTy
        for derivation in CG._grounded_derivations(relation, step_ty, Ty[Vec2, FloatTy])
            CG._signature_contains(CG._derivation_signature(derivation), :FADD) && return derivation
        end
        return nothing
    end

    relation = CG.CoreRelation(; global_consts=fold_consts, guarded_expansion_depth=2)
    outer_rule = find_afoldl_rule(relation, FloatTy, Ty[Matrix32], Matrix32)
    inner_rule = find_afoldl_rule(relation, FloatTy, Ty[Vec2, FloatTy], Vec2)
    outer_init = find_lenf_app(relation)
    inner_step = find_fadd_step(relation)

    @test outer_rule !== nothing
    @test inner_rule !== nothing
    @test outer_init !== nothing
    @test inner_step !== nothing
    (outer_rule === nothing || inner_rule === nothing || outer_init === nothing || inner_step === nothing) && return

    inner_fold = CG.HeadApplicationDerivation(
        inner_rule.head,
        CG.CoreDerivation[
            inner_step,
            CG.VariableDerivation(1, FloatTy),
            CG.VariableDerivation(0, Vec2),
        ],
    )
    outer_step = CG.LambdaDerivation(FloatTy, CG.LambdaDerivation(Vec2, inner_fold))
    outer_fold = CG.HeadApplicationDerivation(
        outer_rule.head,
        CG.CoreDerivation[
            outer_step,
            outer_init,
            CG.VariableDerivation(0, Matrix32),
        ],
    )
    derivation = CG.LambdaDerivation(Matrix32, outer_fold)
    local_consts = Dict{Symbol,Any}()
    local_const_types = Dict{Symbol,TyLike}()
    tree = AstTree(CG._build_derivation_ast!(derivation, local_consts, local_const_types, relation))

    @test signature_count(CG._derivation_signature(derivation), :AFOLDL) >= 2
    @test iscompatible(infer(tree, local_const_types), target)
end

@testitem "CanonicalGeneration AFOLDL synthesized step keeps same-type binders distinct" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyNat, typekey

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    relation = CG.CoreRelation(; guarded_expansion_depth=1)

    function find_vec_fold_rule()
        catalog = CG._relation_catalog(relation, FloatTy, Ty[Vec2]; include_guarded=true)
        for head in catalog.heads
            CG._is_guarded_head(head) || continue
            head.name == :AFOLDL || continue
            head.type isa Ty || continue
            args, result_ty = CG._split_fun(head.type)
            length(args) == 3 || continue
            typekey(result_ty) == typekey(FloatTy) || continue
            typekey(args[3]) == typekey(Vec2) || continue
            return CG.CoreRule(head, result_ty, args, [CG._argument_class(arg) for arg in args])
        end
        return nothing
    end

    rule = find_vec_fold_rule()
    @test rule !== nothing
    rule === nothing && return
    step_scope = CG._afoldl_step_scope(rule, Ty[Vec2])
    @test length(step_scope) >= 2
    @test typekey(step_scope[1]) == typekey(FloatTy)
    @test typekey(step_scope[2]) == typekey(FloatTy)
end

@testitem "CanonicalGeneration lazy guarded COND samples comparison conditions" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, typekey

    CG = LambdaRegression.CanonicalGeneration
    relation = CG.CoreRelation(;
        global_consts=Dict{Symbol,Any}(:FEQ => LambdaRegression.TypeSystem.FEQ),
        guarded_expansion_depth=1,
    )
    scope = Ty[FloatTy]

    function find_cond_rule()
        catalog = CG._relation_catalog(relation, FloatTy, scope; include_guarded=true)
        for head in catalog.heads
            CG._is_guarded_head(head) || continue
            head.name == :COND || continue
            head.type isa Ty || continue
            args, result_ty = CG._split_fun(head.type)
            typekey(result_ty) == typekey(FloatTy) || continue
            return CG.CoreRule(head, result_ty, args, [CG._argument_class(arg) for arg in args])
        end
        return nothing
    end

    cond_rule = find_cond_rule()
    @test cond_rule !== nothing
    cond_rule === nothing && return
    @test typekey(cond_rule.argument_tys[1]) == typekey(BoolTy)

    state = CG.DerivationSamplingState(Random.MersenneTwister(8000), 32)
    condition = CG._sample_head_argument(state, relation, cond_rule, 1, scope, 4, 0, true)

    @test condition !== nothing
    @test CG._signature_contains(CG._derivation_signature(condition), :FEQ)
end

@testitem "CanonicalGeneration materializer separates same-name typed instances" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyLike, TyNat, →

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    matrix_fold_ty = (Matrix32 → Vec2 → Matrix32) → Matrix32 → Matrix32 → Matrix32
    row_fold_ty = (Matrix32 → FloatTy → Matrix32) → Matrix32 → Vec2 → Matrix32
    relation = CG.CoreRelation()
    local_consts = Dict{Symbol,Any}()
    local_const_types = Dict{Symbol,TyLike}()
    matrix_head = CG.HeadEntry(
        :AFOLDL,
        :guarded_primitive,
        matrix_fold_ty,
        CG.Decoration(),
        CG.WeightBucket(:test);
        metadata=(value=LambdaRegression.TypeSystem.AFOLDL,),
    )
    row_head = CG.HeadEntry(
        :AFOLDL,
        :guarded_primitive,
        row_fold_ty,
        CG.Decoration(),
        CG.WeightBucket(:test);
        metadata=(value=LambdaRegression.TypeSystem.AFOLDL,),
    )

    matrix_node = CG._head_node!(matrix_head, local_consts, local_const_types, relation)
    row_node = CG._head_node!(row_head, local_consts, local_const_types, relation)

    @test matrix_node isa ConstNode
    @test row_node isa ConstNode
    @test matrix_node.name == :AFOLDL
    @test row_node.name != matrix_node.name
    @test local_const_types[matrix_node.name] == matrix_fold_ty
    @test local_const_types[row_node.name] == row_fold_ty
end

@testitem "CanonicalGeneration catalog preset AST terms use typing boundary" begin
    using Test
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.TypeSystem: FloatTy, Ty, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    presets = Dict{Symbol,Any}(:ID => parse_lambda("λx.x"))
    universe = CG.CoreUniverse(FloatTy; presets=presets)
    catalog = CG._build_core_catalog(universe; presets=presets)
    ids = [entry for entry in catalog.heads if entry.name == :ID]

    @test count(entry -> iscompatible(entry.type, FloatTy → FloatTy), ids) == 1
    @test all(entry -> entry.source == :preset, ids)
    @test all(entry -> entry.type isa Ty && CG._in_closed_fragment(entry.type), ids)
end

@testitem "CanonicalGeneration generate required targets infer to compatible types" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AppNode, AbsNode, children
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyLike, TyNat, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    Float4Fn = TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, FloatTy))))
    targets = [
        TyFun(Matrix32, Matrix32),
        TyFun(FloatTy, FloatTy),
        Float4Fn,
    ]

    function no_lambda_head(node)
        node isa AppNode && node.func isa AbsNode && return false
        return all(no_lambda_head, children(node))
    end

    for target in targets
        for seed in 1:6
            result = CG.generate_canonical(target; rng=Random.MersenneTwister(seed))
            @test result isa CG.GenerationResult
            tyctx = Dict{Symbol,TyLike}(result.local_const_types)
            inferred = infer(result.tree, tyctx)
            @test !(inferred isa LambdaRegression.Utils.Err)
            @test iscompatible(inferred, target)
            @test no_lambda_head(result.tree.root)
        end
    end
end

@testitem "CanonicalGeneration context LENF can generate matrix to float" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyLike, TyNat, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    target = TyFun(Matrix32, FloatTy)

    result = CG.generate_canonical(
        target;
        rng=Random.MersenneTwister(2000),
        global_consts=Dict{Symbol,Any}(:LENF => LambdaRegression.TypeSystem.LENF),
        guarded_expansion_depth=0,
    )

    @test result isa CG.GenerationResult
    @test iscompatible(infer(result.tree, Dict{Symbol,TyLike}(result.local_const_types)), target)
    @test CG._signature_contains(result.metadata[:derivation], :LENF)
    @test !CG._signature_contains(result.metadata[:derivation], :AMAP)
end

@testitem "CanonicalGeneration rich targets share one rich context" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyList, TyNat, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    Float4Fn = TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, TyFun(FloatTy, FloatTy))))
    targets = (
        TyFun(Matrix32, Matrix32),
        TyFun(Matrix32, FloatTy),
        TyFun(FloatTy, FloatTy),
        Float4Fn,
        TyFun(TyList(FloatTy), FloatTy),
    )
    context = CG.test_catalog_consts()
    generation_kwargs = CG.test_catalog_generation_kwargs(profile=:deeper)

    @test all(name -> haskey(context, name), [:FSIN, :FSQRT, :SQR, :FADD, :FSUB, :FMUL, :FDIV, :FEQ, :FLE, :LENF])
    @test !haskey(context, :TRUE)
    @test !haskey(context, :FALSE)
    @test !haskey(context, :NOT)

    for target in targets
        for seed in 1:2
            result = CG.generate_canonical(
                target;
                rng=Random.MersenneTwister(700 + seed),
                global_consts=context,
                generation_kwargs...,
            )
            @test result isa CG.GenerationResult
            @test iscompatible(infer(result.tree, Dict{Symbol,TyLike}(result.local_const_types)), target)
        end
    end
end

@testitem "CanonicalGeneration rich providers expose typed value factories" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    Vec2 = TyArray(TyNat(2), FloatTy)
    Matrix32 = TyArray(TyNat(3), Vec2)
    providers = CG.test_catalog_providers()

    @test any(provider -> provider isa CG.TypedValueFactoryProvider, providers)
    @test !any(provider -> provider isa CG.TypedValueFactoryProvider, CG.default_head_providers())

    weights = CG.GenerationWeights(;
        variable=0,
        constant=0,
        preset_term=0,
        instance_factory=1,
        array_source=0,
        tolist=0,
        amap=0,
        guarded_expansion=0,
    )
    cache_key = CG._default_relation_cache_key(
        FloatTy,
        weights,
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(),
        providers,
        CG.DefaultArgumentPlanner(),
        1,
    )

    @test cache_key !== nothing

    float_result = CG.generate_canonical(
        FloatTy;
        rng=Random.MersenneTwister(42),
        providers,
        weights,
    )
    matrix_result = CG.generate_canonical(
        Matrix32;
        rng=Random.MersenneTwister(555),
        providers,
        weights,
    )

    @test float_result isa CG.GenerationResult
    @test matrix_result isa CG.GenerationResult
    @test float_result.tree.root isa ConstNode
    @test matrix_result.tree.root isa ConstNode
    @test startswith(string(float_result.tree.root.name), "FLOAT")
    @test startswith(string(matrix_result.tree.root.name), "FLOAT_ARRAY")
    @test iscompatible(infer(float_result.tree, float_result.local_const_types), FloatTy)
    @test iscompatible(infer(matrix_result.tree, matrix_result.local_const_types), Matrix32)
end

@testitem "CanonicalGeneration weight tags stack on specific heads" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy

    CG = LambdaRegression.CanonicalGeneration
    context = Dict{Symbol,Any}(:LOW => 1.0, :HIGH => 2.0)
    context_ty = Dict{Symbol,Any}(:LOW => FloatTy, :HIGH => FloatTy)
    providers = (
        CG.ContextHeadProvider(),
        CG.WeightTagProvider(:HIGH => (:preferred, :specific_high)),
    )
    weights = CG.GenerationWeights(; constant=1, preferred=3, specific_high=5)
    relation = CG.CoreRelation(; global_consts=context, global_consts_ty=context_ty, providers, weights)
    options = CG._sampling_options(relation, FloatTy, LambdaRegression.TypeSystem.Ty[], 4; allow_head_application=true, lambda_kind=:lambda_generation)
    by_name = Dict(option.value.head.name => option.weight for option in options if option.kind == :constant)

    @test by_name[:LOW] == 1
    @test by_name[:HIGH] == 15

    disabled = CG.GenerationWeights(; constant=1, preferred=0, specific_high=5)
    disabled_relation = CG.CoreRelation(; global_consts=context, global_consts_ty=context_ty, providers, weights=disabled)
    disabled_options = CG._sampling_options(disabled_relation, FloatTy, LambdaRegression.TypeSystem.Ty[], 4; allow_head_application=true, lambda_kind=:lambda_generation)
    disabled_by_name = Dict(option.value.head.name => option.weight for option in disabled_options if option.kind == :constant)

    @test disabled_by_name[:LOW] == 1
    @test disabled_by_name[:HIGH] == 0
end

@testitem "CanonicalGeneration deeper profile is explicit and tagged" begin
    using Test

    CG = LambdaRegression.CanonicalGeneration
    kwargs = CG.test_catalog_generation_kwargs(profile=:deeper)

    @test kwargs.providers == CG.test_catalog_providers()
    @test kwargs.guarded_expansion_depth > 2
    @test kwargs.sampling_depth_limit > 10
    @test kwargs.sampling_node_budget > 128
    @test CG._bucket_weight(kwargs.weights, CG.WeightBucket(:variable)) >
        CG._bucket_weight(kwargs.weights, CG.WeightBucket(:constant))
    @test CG._bucket_weight(kwargs.weights, CG.WeightBucket(:real_binary)) > 1
    @test CG._bucket_weight(kwargs.weights, CG.WeightBucket(:cond)) > 1
    @test CG._bucket_weight(kwargs.weights, CG.WeightBucket(:lambda)) > 1
    @test CG._bucket_weight(kwargs.weights, CG.WeightBucket(:safe_list_access)) > 1
end

@testitem "CanonicalGeneration rigid matrix slow seed keeps safe-list fold map coverage" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: TyArray, TyNat, TyScheme, TyVar, infer, iscompatible, →

    CG = LambdaRegression.CanonicalGeneration
    ρ = TyVar(240_001, 0, nothing)
    RigidMatrix32 = TyArray(TyNat(3), TyArray(TyNat(2), ρ))
    target = TyScheme([ρ], RigidMatrix32 → RigidMatrix32)

    result = CG.generate_canonical(
        target;
        rng=Random.MersenneTwister(1111),
        global_consts=CG.test_catalog_consts(),
        CG.test_catalog_generation_kwargs(profile=:deeper)...,
    )

    @test result isa CG.GenerationResult
    @test iscompatible(infer(result.tree, result.local_const_types), target)
    @test CG._signature_contains(result.metadata[:derivation], :UNCONS_CASE)
    @test CG._signature_contains(result.metadata[:derivation], :AFOLDL)
    @test CG._signature_contains(result.metadata[:derivation], :AMAP)
end

@testitem "CanonicalGeneration boolean primitives are ordinary injected constants" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: BoolTy, FALSE, NOT, TRUE, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    context = Dict{Symbol,Any}(:TRUE => TRUE, :FALSE => FALSE, :NOT => NOT)
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1111),
        weights=CG.GenerationWeights(;
            variable=0,
            constant=1,
            preset_term=0,
            instance_factory=0,
            array_source=0,
            tolist=0,
            amap=0,
            guarded_expansion=0,
        ),
    )

    result = CG.generate_canonical(BoolTy; global_consts=context, config=config)

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test result.tree.root.name in (:TRUE, :FALSE)
    @test iscompatible(infer(result.tree, result.local_const_types), BoolTy)
end

@testitem "CanonicalGeneration generate rejects unsupported core targets" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyList, TyMu, TyProd, TySum, TyVar, TyFun, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Vec3 = TyArray(TyNat(3), FloatTy)
    ListVec3 = TyList(Vec3)
    open = TyVar(90_001, 0, nothing)
    unsupported = [
        TyProd(FloatTy, FloatTy),
        TySum(FloatTy, FloatTy),
        TyMu(open, FloatTy),
        open,
        TyFun(ListVec3, FloatTy),
    ]

    for (i, target) in enumerate(unsupported)
        result = CG.generate_canonical(target; rng=Random.MersenneTwister(i))
        @test result isa LambdaRegression.Utils.Err
    end
end

@testitem "CanonicalGeneration generate requires runtime values for terminal constants" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration

    missing_value = CG.generate_canonical(
        FloatTy;
        global_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
        rng=Random.MersenneTwister(1),
    )

    @test missing_value isa LambdaRegression.Utils.Err

    with_value = CG.generate_canonical(
        FloatTy;
        local_consts=Dict{Symbol,Any}(:k => 1.0),
        rng=Random.MersenneTwister(1111),
    )

    @test with_value isa CG.GenerationResult
    inferred = infer(with_value.tree, with_value.local_const_types)
    @test iscompatible(inferred, FloatTy)
    @test haskey(with_value.local_consts, :k)
end

@testitem "CanonicalGeneration weights zero variable weight forces non-variable samples" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1),
        weights=CG.GenerationWeights(;
            variable=0,
            constant=1,
            preset_term=0,
            instance_factory=0,
            array_source=0,
            tolist=0,
            amap=0,
        ),
    )

    result = CG.generate_canonical(
        FloatTy;
        local_consts=Dict{Symbol,Any}(:x => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:x => FloatTy),
        global_consts=Dict{Symbol,Any}(:k => 2.0),
        global_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
        config=config,
    )

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test result.tree.root.name == :k
    @test first(result.metadata[:derivation]) == :const
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration weights zero constant weight forces variable samples" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1),
        weights=CG.GenerationWeights(;
            variable=1,
            constant=0,
            preset_term=0,
            instance_factory=0,
            array_source=0,
            tolist=0,
            amap=0,
        ),
    )

    result = CG.generate_canonical(
        FloatTy;
        local_consts=Dict{Symbol,Any}(:x => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:x => FloatTy),
        global_consts=Dict{Symbol,Any}(:k => 2.0),
        global_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy),
        config=config,
    )

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test result.tree.root.name == :x
    @test first(result.metadata[:derivation]) == :const
    @test iscompatible(infer(result.tree, result.local_const_types), FloatTy)
end

@testitem "CanonicalGeneration weights zero constant prevents context LENF route" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    target = TyFun(Matrix32, FloatTy)
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1111),
        weights=CG.GenerationWeights(; constant=0),
    )

    result = CG.generate_canonical(target; global_consts=CG.test_catalog_consts(), config=config)

    @test result isa LambdaRegression.Utils.Err
end

@testitem "CanonicalGeneration weights zero array source prevents LENF application" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    target = FloatTy
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1111),
        weights=CG.GenerationWeights(; array_source=0),
    )

    result = CG.generate_canonical(
        target;
        global_consts=CG.test_catalog_consts(),
        local_consts=Dict{Symbol,Any}(:m => [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        local_consts_ty=Dict{Symbol,Any}(:m => Matrix32),
        config=config,
    )

    @test result isa LambdaRegression.Utils.Err
end

@testitem "CanonicalGeneration weights preset terms can be selected as heads" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    target = TyFun(FloatTy, FloatTy)
    presets = Dict{Symbol,Any}(:ID => parse_lambda(string(Char(0x03bb), "x.x")))
    preset_types = Dict{Symbol,Any}(:ID => target)
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1234),
        weights=CG.GenerationWeights(;
            variable=0,
            constant=0,
            preset_term=1,
            instance_factory=0,
            array_source=0,
            tolist=0,
            amap=0,
        ),
    )

    result = CG.generate_canonical(target; presets=presets, preset_types=preset_types, config=config)

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test result.tree.root.name == :ID
    @test first(result.metadata[:derivation]) == :const
    @test iscompatible(infer(result.tree, result.local_const_types), target)
end

@testitem "CanonicalGeneration weights NIL inhabits list while CONS stays guarded" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyList, infer, iscompatible, typekey

    CG = LambdaRegression.CanonicalGeneration
    target = TyList(FloatTy)
    universe = CG.CoreUniverse(target)
    catalog = CG._build_core_catalog(universe)
    names = Set(entry.name for entry in catalog.heads)

    @test :NIL in names
    @test :CONS in names
    @test all(rule -> rule.head.name != :CONS, catalog.rules)

    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(8000),
        weights=CG.GenerationWeights(;
            variable=0,
            constant=0,
            preset_term=0,
            instance_factory=1,
            array_source=0,
            tolist=0,
            amap=0,
        ),
    )

    result = CG.generate_canonical(target; config=config)

    @test result isa CG.GenerationResult
    @test result.tree.root isa ConstNode
    @test startswith(string(result.tree.root.name), "NIL")
    @test typekey(result.local_const_types[result.tree.root.name]) == typekey(target)
    @test iscompatible(infer(result.tree, result.local_const_types), target)
end

@testitem "CanonicalGeneration default core omits scalar and ndarray value factories" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    BoolMat = TyArray(TyNat(2), TyArray(TyNat(3), BoolTy))
    targets = (FloatTy, IntTy, BoolMat)

    for target in targets
        catalog = CG._build_core_catalog(CG.CoreUniverse(target))
        names = Set(entry.name for entry in catalog.heads)
        @test !(:FLOAT in names)
        @test !(:INT in names)
        @test !(:FLOAT_ARRAY in names)
        @test !(:INT_ARRAY in names)
        @test !(:BOOL_ARRAY in names)

        config = CG.GenerationConfig(;
            rng=Random.MersenneTwister(555),
            weights=CG.GenerationWeights(;
                variable=0,
                constant=0,
                preset_term=0,
                instance_factory=1,
                array_source=0,
                tolist=0,
                amap=0,
                guarded_expansion=0,
            ),
        )
        @test !(CG.generate_canonical(target; config=config) isa CG.GenerationResult)
    end
end

@testitem "CanonicalGeneration typed value factories can be sampled as instance factories" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: ConstNode
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, TyArray, TyNat, infer, iscompatible, typekey

    CG = LambdaRegression.CanonicalGeneration
    VF = LambdaRegression.ValueFactories

    Vec3F = TyArray(TyNat(3), FloatTy)
    Vec2I = TyArray(TyNat(2), IntTy)
    BoolMat = TyArray(TyNat(2), TyArray(TyNat(3), BoolTy))
    factories = (
        VF.FloatScalarFactory(mean=2.0, std=0.0),
        VF.IntScalarFactory(lo=7, hi=7),
        VF.FloatArrayFactory(mean=1.5, std=0.0),
        VF.IntArrayFactory(lo=-2, hi=-2),
        VF.BoolArrayFactory(),
    )
    provider = CG.TypedValueFactoryProvider(factories)
    config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(555),
        providers=(CG.ContextHeadProvider(), provider),
        weights=CG.GenerationWeights(;
            variable=0,
            constant=0,
            preset_term=0,
            instance_factory=1,
            array_source=0,
            tolist=0,
            amap=0,
            guarded_expansion=0,
        ),
    )

    float_result = CG.generate_canonical(FloatTy; config=config)
    int_result = CG.generate_canonical(IntTy; config=config)
    float_vec_result = CG.generate_canonical(Vec3F; config=config)
    int_vec_result = CG.generate_canonical(Vec2I; config=config)
    bool_mat_result = CG.generate_canonical(BoolMat; config=config)

    @test float_result isa CG.GenerationResult
    @test int_result isa CG.GenerationResult
    @test float_vec_result isa CG.GenerationResult
    @test int_vec_result isa CG.GenerationResult
    @test bool_mat_result isa CG.GenerationResult

    @test float_result.tree.root isa ConstNode
    @test int_result.tree.root isa ConstNode
    @test float_vec_result.tree.root isa ConstNode
    @test int_vec_result.tree.root isa ConstNode
    @test bool_mat_result.tree.root isa ConstNode

    @test float_result.local_consts[float_result.tree.root.name] == 2.0
    @test int_result.local_consts[int_result.tree.root.name] == 7
    @test float_vec_result.local_consts[float_vec_result.tree.root.name] == [1.5, 1.5, 1.5]
    @test int_vec_result.local_consts[int_vec_result.tree.root.name] == [-2, -2]
    @test length(bool_mat_result.local_consts[bool_mat_result.tree.root.name]) == 2
    @test all(row -> length(row) == 3, bool_mat_result.local_consts[bool_mat_result.tree.root.name])

    for result in (float_result, int_result, float_vec_result, int_vec_result, bool_mat_result)
        @test typekey(result.local_const_types[result.tree.root.name]) == typekey(result.target_ty)
        @test iscompatible(infer(result.tree, result.local_const_types), result.target_ty)
    end
end

@testitem "CanonicalGeneration integration smoke supports Individual evaluation and mutation" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AstTree
    using LambdaRegression.Mutations: IndexedProvider, ScalarConstMutation, WeightedPairChooser, mutate_tree
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Regression: default_components, default_config, evaluate_default
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyLike, infer, iscompatible

    CG = LambdaRegression.CanonicalGeneration
    id_config = CG.GenerationConfig(;
        rng=Random.MersenneTwister(1234),
        weights=CG.GenerationWeights(;
            variable=1,
            constant=0,
            preset_term=0,
            instance_factory=0,
            array_source=0,
            tolist=0,
            amap=0,
        ),
    )
    target = TyFun(FloatTy, FloatTy)
    id_result = CG.generate_canonical(target; config=id_config)

    @test id_result isa CG.GenerationResult
    ind = Individual(
        id_result.tree,
        target;
        local_consts=id_result.local_consts,
        local_consts_ty=id_result.local_const_types,
    )
    components = default_components(
        global_consts=Dict{Symbol,Any}(),
        global_consts_ty=Dict{Symbol,TyLike}(),
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => FloatTy),
    )
    @test evaluate_default(ind, 2.5, components, default_config(max_reduce_steps=20)) == 2.5

    const_result = CG.generate_canonical(
        FloatTy;
        local_consts=Dict{Symbol,Any}(:seed => 1.0),
        local_consts_ty=Dict{Symbol,TyLike}(:seed => FloatTy),
        config=CG.GenerationConfig(;
            rng=Random.MersenneTwister(8000),
            weights=CG.GenerationWeights(;
                variable=1,
                constant=0,
                preset_term=0,
                instance_factory=0,
                array_source=0,
                tolist=0,
                amap=0,
            ),
        ),
    )
    @test const_result isa CG.GenerationResult

    mutated = mutate_tree(
        const_result.tree,
        FloatTy;
        rng=Random.MersenneTwister(42),
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=const_result.local_consts,
        local_consts_ty=const_result.local_const_types,
        mutations=[ScalarConstMutation(σ=0.1)],
        chooser=WeightedPairChooser(weights=Dict{DataType,Float64}(ScalarConstMutation => 1.0)),
        provider=IndexedProvider(),
        max_try=10,
    )

    @test mutated isa AstTree
    @test iscompatible(infer(mutated, const_result.local_const_types), FloatTy)
end

end # module CanonicalGeneration
