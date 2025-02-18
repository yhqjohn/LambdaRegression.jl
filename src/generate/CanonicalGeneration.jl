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

Providers may also contribute constrained generation options through
`add_generation_options!` and sample them through
`sample_generation_extension`. `GenerationQuery` carries the current lexical
scope and all decreasing search measures. Capability masks are inherited by
nested queries: built-in catalog providers declare no recursive capability,
while an undeclared custom provider is conservatively `:opaque` and is omitted
from restricted queries. `CoreRelation` detects providers that overload
`add_generation_options!` once during construction; catalog-only providers
retain the ordinary sampling path without query-protocol work.
`generation_provider_applicable` is an optional cheap query preflight; its
default is conservative, and a custom `false` result must imply that the
provider would add no option for that query.

Extension routing is fixed when a `CoreRelation` is constructed. Adding methods
later requires rebuilding the relation, and `provides_generation_options` must
remain stable for that relation's lifetime. Provider instances with mutable
state own their synchronization when shared across threads; mutable relations
are query-local and are not intended for concurrent mutation. Under a lexical
binder, call the nested sampler with `added_scope_tys` rather than replacing the
complete scope. Scope-relative requirement types must explicitly overload
`rebase_generation_requirement`; forgetting to do so fails at the binder
boundary instead of silently retargeting an index.

An option that returns its own `CoreDerivation` implements
`derivation_node_count`, `derivation_result_type`, `derivation_signature`,
`derivation_uses_scope_binder`, and
`materialize_generation_derivation`; `derivation_metadata` and
`rigid_safe_derivation` complete the result/discharge contract. Use
`consume_generation_nodes!` rather than mutating the shared ledger. A guarded
provider can call `known_generation_witness` before adding a binder to obtain
one finite acyclic core witness under an added capability mask. Recursive μ
function types remain outside the ordinary canonical target fragment;
`generation_function_view` exposes only one finite outer function view for a
constrained provider declaring `:recursive_type_query`, whose applications must
still consume query budget.

`GenerationConfig` also accepts callable `argument_planner`, `sampler`, and
`materializer` fields plus `guarded_expansion_depth` for the bounded expansion
layer. Replacing them changes local proof-tree pruning, derivation selection,
AST lowering, or guarded depth without adding new branches to the core. Provider
and global argument planners are sampling policy; they do not redefine
`known_generation_witness`, which applies the finite type rules and anchored
source constraints directly. Provider order matters, but provider categories
and weight keys are open. Same-type
`AMAP` is still an anchored core rule when its source array is grounded; weight
or sampler policies may make it rarer, but the core does not exclude it.
Function-typed scope binders are sampled as local application heads under the
same depth and node fuel as ordinary heads; tune them with
`GenerationWeights(scope_application=...)`, or omit that key to inherit
`variable`. A constrained extension may require one exact scope binder to head
an application. The finite certificate anchors that call, while unconstrained
children retain ordinary recursive sampling and use the certificate witness
only as a finite fallback.

Safe list access is opt-in through `SafeListAccessProvider()`, which is included
in `test_catalog_providers()` but not in the minimal default provider tuple. It
does not add `UNCONS_CASE`, `HEAD`, `TAIL`, or generic sum/case heads to the
ordinary catalog. Instead, it collects finite list sources from scope binders,
typed list constants, anchored `TOLIST` views of known arrays, and optional
`source_collectors`, then lowers proxy-using bodies to guarded `UNCONS_CASE`
terms. Use `GenerationWeights(; safe_list_access=..., tolist=...,
array_source=...)` to tune the route. The proxy-using success body uses a
constructive core/anchored-head proposal without replacing ordinary recursive
sampling. `proxy_constructive` and `proxy_general` tune those two proposals
(defaults `9` and `1`); `proxy_safe_list`, `proxy_scope_application`,
`proxy_split`, and `proxy_head` multiply constructive subroutes, while
`proxy_required_head` further tunes productive compositions. Head weights are
normalised across forced argument positions. When several
`UsesScopeBinder` obligations must coexist, the constructive route assigns each
remaining obligation to a compatible child query instead of enumerating every
cross-product partition; cached plans contain no RNG choices or mutable attempt
state. Explicitly disabled subroutes do not run their collectors or planners.
Positive standard subroutes build structural catalogs to obtain their effective
weights; productive-head planning remains lazy until that route is sampled, and
sibling witnesses are checked only for the selected plan. Custom argument
planners remain authoritative. Guarded expansions may wrap safe-list access, but the
proxy-enforcing subproblem does not recursively open guarded search merely to
discover a proxy use. A custom `source_collector` is a callable
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
export GenerationQuery, AbstractGenerationRequirement, UsesScopeBinder, CallsScopeBinder, ForbidsScopeBinder
export provider_capabilities, provides_generation_options, generation_provider_applicable
export provider_replays_materialized_value
export generation_provider_exhausted, generation_requirement_affects_core_search
export rebase_generation_requirement
export generation_provider_cache_key, generation_requirement_cache_key
export generation_root_requirements
export add_generation_options!, sample_generation_extension
export sample_generation_derivation_using_scope_binder
export CoreDerivation, consume_generation_nodes!
export derivation_node_count, derivation_signature, derivation_metadata
export derivation_capabilities
export derivation_result_type
export derivation_uses_scope_binder, derivation_calls_scope_binder, rigid_safe_derivation
export materialize_generation_derivation, generation_function_view
export known_generation_witness
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

"""
    AbstractGenerationRequirement

Finite, query-local constraint interpreted by a generation extension. Core
sampling preserves requirements transitively but otherwise treats them as
opaque policy data.
"""
abstract type AbstractGenerationRequirement end

"""
    UsesScopeBinder(index)

Require an extension result to use the zero-based De Bruijn binder at `index`.
The requirement is an anchoring constraint for extension providers; the core
does not inspect derivations to enforce it. Negative indices throw
`ArgumentError`.
"""
struct UsesScopeBinder <: AbstractGenerationRequirement
    index::Int
    function UsesScopeBinder(index::Integer)
        index >= 0 || throw(ArgumentError("scope binder index must be nonnegative"))
        return new(Int(index))
    end
end

"""Require the zero-based De Bruijn binder at `index` to head an application."""
struct CallsScopeBinder <: AbstractGenerationRequirement
    index::Int
    function CallsScopeBinder(index::Integer)
        index >= 0 || throw(ArgumentError("scope binder index must be nonnegative"))
        return new(Int(index))
    end
end

"""Exclude one zero-based De Bruijn binder from a finite Known₀ witness."""
struct ForbidsScopeBinder <: AbstractGenerationRequirement
    index::Int
    function ForbidsScopeBinder(index::Integer)
        index >= 0 || throw(ArgumentError("forbidden scope binder index must be nonnegative"))
        return new(Int(index))
    end
end

"""
    rebase_generation_requirement(requirement, added_binders)

Rebase a query requirement after `added_binders` new De Bruijn binders are
prepended to the lexical scope. The default accepts an unchanged scope and
rejects positive rebasing: every custom requirement must explicitly overload
this method to declare either scope invariance or its index transformation.
"""
function rebase_generation_requirement(
    requirement::AbstractGenerationRequirement,
    added_binders::Integer,
)
    added = _checked_nonnegative_int(:added_binders, added_binders)
    added == 0 && return requirement
    throw(ArgumentError(
        "$(typeof(requirement)) must define rebase_generation_requirement " *
        "before generation enters a nested binder scope",
    ))
end
rebase_generation_requirement(requirement::UsesScopeBinder, added_binders::Integer) =
    UsesScopeBinder(requirement.index + _checked_nonnegative_int(:added_binders, added_binders))
rebase_generation_requirement(requirement::CallsScopeBinder, added_binders::Integer) =
    CallsScopeBinder(requirement.index + _checked_nonnegative_int(:added_binders, added_binders))
rebase_generation_requirement(requirement::ForbidsScopeBinder, added_binders::Integer) =
    ForbidsScopeBinder(requirement.index + _checked_nonnegative_int(:added_binders, added_binders))

"""Return whether a requirement still constrains ordinary core search."""
generation_requirement_affects_core_search(::AbstractGenerationRequirement) = true

"""
    GenerationQuery(target_ty; scope_tys=Ty[], depth_left=0,
                    guarded_left=0, provider_fuel=depth_left,
                    node_budget=typemax(Int),
                    capability_mask=(), requirements=())

Current finite sampling request. `scope_tys` is ordered inner to outer, with
slot one representing De Bruijn index zero and is stored as an immutable tuple.
`depth_left`, `guarded_left`, `provider_fuel`, and `node_budget` are nonnegative
and may only stay equal or decrease in a query derived through
[`sample_generation_extension`](@ref). `capability_mask` and `requirements` are
inherited transitively. Nested scopes may only prepend binders while inherited
requirements are present; scope-relative requirements are rebased through
[`rebase_generation_requirement`](@ref). Invalid types, negative measures,
non-requirement values, or incompatible scope replacement throw `ArgumentError`.
"""
struct GenerationQuery
    target_ty::Ty
    scope_tys::Tuple{Vararg{Ty}}
    depth_left::Int
    guarded_left::Int
    provider_fuel::Int
    node_budget::Int
    capability_mask::Tuple{Vararg{Symbol}}
    requirements::Tuple
end

function GenerationQuery(
    target_ty::Ty;
    scope_tys=Ty[],
    depth_left::Integer=0,
    guarded_left::Integer=0,
    provider_fuel::Integer=depth_left,
    node_budget::Integer=typemax(Int),
    capability_mask=(),
    requirements=(),
)
    depth = _checked_nonnegative_int(:depth_left, depth_left)
    guarded = _checked_nonnegative_int(:guarded_left, guarded_left)
    fuel = _checked_nonnegative_int(:provider_fuel, provider_fuel)
    nodes = _checked_nonnegative_int(:node_budget, node_budget)
    mask = _capability_tuple(capability_mask)
    requirement_tuple = _requirement_tuple(requirements)
    return GenerationQuery(
        target_ty,
        _scope_type_tuple(scope_tys),
        depth,
        guarded,
        fuel,
        nodes,
        mask,
        requirement_tuple,
    )
end

function _scope_type_tuple(scope_tys)
    out = Ty[]
    for ty in scope_tys
        ty isa Ty || throw(ArgumentError("generation scope entries must subtype Ty"))
        push!(out, ty)
    end
    return Tuple(out)
end

function _scope_type_tuple(scope_tys::Tuple)
    all(ty -> ty isa Ty, scope_tys) ||
        throw(ArgumentError("generation scope entries must subtype Ty"))
    return scope_tys
end

_capability_tuple(capabilities::Symbol) = (capabilities,)
function _capability_tuple(capabilities)
    out = Symbol[]
    for capability in capabilities
        capability isa Symbol || throw(ArgumentError("generation capabilities must be symbols"))
        capability in out || push!(out, capability)
    end
    return Tuple(out)
end

_requirement_tuple(requirement::AbstractGenerationRequirement) = (requirement,)
function _requirement_tuple(requirements::Tuple)
    all(requirement -> requirement isa AbstractGenerationRequirement, requirements) ||
        throw(ArgumentError("generation requirements must subtype AbstractGenerationRequirement"))
    return requirements
end
function _requirement_tuple(requirements)
    out = AbstractGenerationRequirement[]
    for requirement in requirements
        requirement isa AbstractGenerationRequirement ||
            throw(ArgumentError("generation requirements must subtype AbstractGenerationRequirement"))
        push!(out, requirement)
    end
    return Tuple(out)
end

"""
    provider_capabilities(provider)

Return the finite tuple of recursive/search capabilities owned by `provider`.
Unknown custom providers default to `(:opaque,)`; override this method to
declare a safe provider with `()` or to name capabilities that restricted
queries can mask.
"""
provider_capabilities(::AbstractHeadProvider) = (:opaque,)
provider_capabilities(::ContextHeadProvider) = ()
provider_capabilities(::NilFactoryProvider) = ()
provider_capabilities(::TypedValueFactoryProvider) = ()
provider_capabilities(::ToListProvider) = ()
provider_capabilities(::AMapProvider) = ()
provider_capabilities(::GuardedExpansionProvider) = ()
provider_capabilities(::SafeListAccessProvider) = (:safe_list_access,)
provider_capabilities(::WeightTagProvider) = ()

"""
    provider_replays_materialized_value(provider, value, ty)

Return `true` only when `provider` can rematerialize the same runtime constant
at compatible closed instances of `ty`. Mutation clients use this conservative
protocol, when explicitly enabled by a caller that knows the constants'
provenance, to avoid feeding provider-owned lowering artifacts back as
independent user constants. Unknown providers and value factories default to
`false`.
"""
provider_replays_materialized_value(::AbstractHeadProvider, value, ty::TyLike) = false
provider_replays_materialized_value(::ToListProvider, value, ty::TyLike) =
    value === TypeSystem.TOLIST
provider_replays_materialized_value(::AMapProvider, value, ty::TyLike) =
    value === TypeSystem.AMAP
provider_replays_materialized_value(::GuardedExpansionProvider, value, ty::TyLike) =
    value === TypeSystem.CONS || value === TypeSystem.COND || value === TypeSystem.AFOLDL
provider_replays_materialized_value(::SafeListAccessProvider, value, ty::TyLike) =
    value === TypeSystem.UNCONS_CASE

"""
    provides_generation_options(provider)

Optionally declare whether `provider` contributes through
[`add_generation_options!`](@ref). Unknown custom providers return `nothing`
and are detected from method dispatch when a `CoreRelation` is constructed.
Overload with `true` or `false` when relation construction is performance
sensitive. Built-in catalog-only providers declare `false`.
"""
provides_generation_options(::AbstractHeadProvider) = nothing
provides_generation_options(::ContextHeadProvider) = false
provides_generation_options(::NilFactoryProvider) = false
provides_generation_options(::TypedValueFactoryProvider) = false
provides_generation_options(::ToListProvider) = false
provides_generation_options(::AMapProvider) = false
provides_generation_options(::GuardedExpansionProvider) = false
provides_generation_options(::SafeListAccessProvider) = false
provides_generation_options(::WeightTagProvider) = false

"""
    generation_provider_applicable(provider, relation, query)

Cheap preflight for a provider already known to contribute generation options.
Return `false` when the current finite query cannot produce an option, allowing
the router to skip cache-key construction and provider dispatch. The default is
conservatively `true`; applicability must not hide an otherwise valid option.
"""
generation_provider_applicable(::AbstractHeadProvider, relation, query) = true

"""
    generation_provider_exhausted(provider, relation, query)

Return `true` only when `provider` cannot contribute for `query` or any query
derived from it. Exhaustion must be monotone because the sampler may then use
the ordinary core path for the remainder of the current sample.
"""
generation_provider_exhausted(::AbstractHeadProvider, relation, query) = false

"""
    generation_provider_cache_key(provider)

Return immutable policy data that makes a custom provider safe to reuse in a
relation cache, or `nothing` to disable that cache. The default is conservative.
"""
generation_provider_cache_key(::AbstractHeadProvider) = nothing

"""Return immutable cache identity for a custom generation requirement."""
generation_requirement_cache_key(::AbstractGenerationRequirement) = nothing

"""
    generation_root_requirements(provider, relation, query)

Return finite requirements that are created once for a root generation query
and inherited by every nested provider dispatch. This is the extension point
for mutable per-sample ledgers such as a shared attempt budget. The default
contributes no requirements.
"""
generation_root_requirements(::AbstractHeadProvider, relation, query) = ()

"""
    add_generation_options!(provider, options, relation, query)

Append finite weighted extension options for `query`. Each option is a named
tuple containing either numeric `weight` or a `weight_bucket`; all remaining
fields are provider-owned payload passed unchanged to
[`sample_generation_extension`](@ref). The default contributes no options.
Providers must anchor every option from the supplied `CoreRelation` and query
context; malformed or non-finite weights throw `ArgumentError`.
"""
function add_generation_options!(
    ::AbstractHeadProvider,
    options,
    relation,
    query,
)
    return options
end

"""
    sample_generation_extension(provider, state, relation, option, query)

Sample one provider-owned option. `state` shares the caller's mutable node
ledger, while `query.node_budget` is the authoritative ceiling; nested calls
clamp attempted refills and successful derivations must fit the nodes charged
since dispatch. `query` has already consumed one depth unit and one
provider-fuel unit. Return a `CoreDerivation` or `nothing` for an expected dead
end. The default is `nothing`; unexpected provider errors remain visible near
their source.

Extension failure is transactional only for `state.nodes_left`: the existing
without-replacement option loop restores that count. RNG advancement and
provider-owned state changes are intentionally not rolled back.
"""
function sample_generation_extension(
    ::AbstractHeadProvider,
    state,
    relation,
    option,
    query,
)
    return nothing
end

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

function _recursive_fragment(ty::Ty, bound::IdSet{TyVar})
    if ty isa TyConst
        return true
    elseif ty isa TyVar
        return ty in bound
    elseif ty isa TyFun
        return _recursive_fragment(ty.arg, bound) && _recursive_fragment(ty.res, bound)
    elseif ty isa TyList
        return _recursive_fragment(ty.elem, bound)
    elseif ty isa TyArray
        return ty.len isa TyNat && _recursive_fragment(ty.elem, bound)
    elseif ty isa TyMu
        _binder_occurrences_guarded(ty.body, ty.var, false) || return false
        push!(bound, ty.var)
        try
            return _recursive_fragment(ty.body, bound)
        finally
            delete!(bound, ty.var)
        end
    end
    return false
end

function _binder_occurrences_guarded(ty::Ty, binder::TyVar, guarded::Bool)
    ty isa TyVar && return ty !== binder || guarded
    ty isa TyMu && return ty.var === binder || _binder_occurrences_guarded(ty.body, binder, guarded)
    children = TypeSystem.type_children(ty)
    isempty(children) && return true
    next_guarded = guarded || !(ty isa TyMu)
    return all(child -> _binder_occurrences_guarded(child, binder, next_guarded), children)
end

_in_generation_extension_fragment(ty::Ty) = _recursive_fragment(ty, IdSet{TyVar}())

function _replace_mu_binder(ty::Ty, binder::TyVar, replacement::Ty)
    ty isa TyVar && return ty === binder ? replacement : ty
    ty isa TyMu && ty.var === binder && return ty
    children = TypeSystem.type_children(ty)
    isempty(children) && return ty
    rewritten = Tuple(_replace_mu_binder(child, binder, replacement) for child in children)
    all(i -> rewritten[i] === children[i], eachindex(children)) && return ty
    return TypeSystem.rebuild_type(ty, rewritten)
end

_unfold_mu_once(ty::TyMu) = _replace_mu_binder(ty.body, ty.var, ty)

"""
    generation_function_view(ty) -> Union{TyFun,Nothing}

Observe one finite function head from an admitted generation type. A recursive
result points back to the original `TyMu`; this function never enumerates its
infinite equi-recursive unfolding.
"""
function generation_function_view(ty::Ty)
    _in_generation_extension_fragment(ty) || return nothing
    current = ty
    seen = IdSet{TyMu}()
    while current isa TyMu
        current in seen && return nothing
        push!(seen, current)
        current = _unfold_mu_once(current)
    end
    return current isa TyFun ? current : nothing
end

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
    bound_vars = Set(scheme.vars)
    type_vars = Set{TyVar}()
    length_vars = Set{TyVar}()
    _mark_scheme_var_positions!(scheme.ty, bound_vars, type_vars, length_vars, :type)
    isempty(length_vars) || return nothing

    scheme_key = typekey(scheme)
    assignment = Dict{TyVar,Ty}()
    discharge_vars = Dict{Symbol,TyVar}()
    rigid_slots = Dict{Symbol,Int}()
    for (slot, var) in enumerate(scheme.vars)
        atom = TyConst(_rigid_dummy_name(scheme_key, slot))
        assignment[var] = atom
        discharge_vars[atom.name] = deepcopy(var)
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

function _discharge_rigid_metadata_value(
    value,
    rigid_slots::AbstractDict{Symbol,Int},
)
    # Derivation signatures encode tree shape in tuples; keep this metadata walk generic.
    Base.@nospecialize value
    if value isa Symbol && haskey(rigid_slots, value)
        return (:rigid, rigid_slots[value])
    elseif value isa Tuple
        if length(value) == 2 && value[1] == :const && value[2] isa Symbol && haskey(rigid_slots, value[2])
            return (:rigid, rigid_slots[value[2]])
        end
        items = Vector{Any}(undef, length(value))
        for index in eachindex(value)
            items[index] = _discharge_rigid_metadata_value(value[index], rigid_slots)
        end
        return Tuple(items)
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
    source_metadata = _context_source_metadata(source)
    for name in sort!(collect(keys(tyctx)))
        ty = tyctx[name]
        ty isa TyLike || continue
        if ty isa Ty
            _in_closed_fragment(ty) || continue
            push!(heads, HeadEntry(
                name,
                source,
                ty,
                _rule_class_for_tylike(ty),
                weight_bucket;
                metadata=source_metadata,
            ))
        elseif ty isa TyScheme
            _add_scheme_heads!(
                heads,
                source,
                name,
                ty,
                weight_bucket,
                universe;
                source_metadata,
            )
        end
    end
    return heads
end

_context_source_metadata(source::Symbol) =
    source === :preset ? (capabilities=(:opaque,),) : (;)

_rule_class_for_tylike(ty::TyLike) = ty isa Ty && _in_closed_fragment(ty) ? CoreProductive() : Decoration()

function _add_scheme_heads!(
    heads::Vector{HeadEntry},
    source::Symbol,
    name::Symbol,
    scheme::TyScheme,
    weight_bucket::CoreWeightBucket,
    universe::CoreUniverse,
    ;
    source_metadata::NamedTuple=_context_source_metadata(source),
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
                metadata=merge(source_metadata, (scheme_instance=true,)),
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
        assignment = Dict{TyVar,Ty}(var => value for (var, value) in zip(scheme.vars, assignment_values))
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
    bound_vars = Set(scheme.vars)
    type_vars = Set{TyVar}()
    length_vars = Set{TyVar}()
    _mark_scheme_var_positions!(scheme.ty, bound_vars, type_vars, length_vars, :type)
    type_choices = Ty[_nonfunction_closed_types(universe)...]
    length_choices = Ty[TyNat(n) for n in universe.lengths]
    return [
        _scheme_var_domain(var, type_vars, length_vars, type_choices, length_choices)
        for var in scheme.vars
    ]
end

function _mark_scheme_var_positions!(
    ty::Ty,
    bound_vars::Set{TyVar},
    type_vars::Set{TyVar},
    length_vars::Set{TyVar},
    position::Symbol,
    binders::Set{TyVar}=Set{TyVar}(),
)
    if ty isa TyVar
        ty in binders && return nothing
        ty in bound_vars || return nothing
        push!(position == :length ? length_vars : type_vars, ty)
    elseif ty isa TyArray
        _mark_scheme_var_positions!(ty.len, bound_vars, type_vars, length_vars, :length, binders)
        _mark_scheme_var_positions!(ty.elem, bound_vars, type_vars, length_vars, :type, binders)
    elseif ty isa TyMu
        had_binder = ty.var in binders
        push!(binders, ty.var)
        _mark_scheme_var_positions!(ty.body, bound_vars, type_vars, length_vars, :type, binders)
        had_binder || delete!(binders, ty.var)
    else
        for child in TypeSystem.type_children(ty)
            _mark_scheme_var_positions!(child, bound_vars, type_vars, length_vars, :type, binders)
        end
    end
    return nothing
end

function _scheme_var_domain(
    var::TyVar,
    type_vars::Set{TyVar},
    length_vars::Set{TyVar},
    type_choices::AbstractVector{<:Ty},
    length_choices::AbstractVector{<:Ty},
)
    in_type = var in type_vars
    in_length = var in length_vars
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

function _subst_scheme_ty(
    ty::Ty,
    assignment::AbstractDict{TyVar,<:Ty},
    binders::Set{TyVar}=Set{TyVar}(),
)
    if ty isa TyVar
        return ty in binders ? ty : get(assignment, ty, ty)
    elseif ty isa TyMu
        had_binder = ty.var in binders
        push!(binders, ty.var)
        body = _subst_scheme_ty(ty.body, assignment, binders)
        had_binder || delete!(binders, ty.var)
        return TyMu(ty.var, body)
    elseif isempty(TypeSystem.type_children(ty))
        return ty
    else
        children = map(child -> _subst_scheme_ty(child, assignment, binders), TypeSystem.type_children(ty))
        return TypeSystem.rebuild_type(ty, children)
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

function _scope_application_rules(
    target_ty::Ty,
    scope_tys::Vector{Ty},
    max_applications::Integer,
)
    application_limit = _checked_nonnegative_int(:max_applications, max_applications)
    rules = ScopeApplicationRule[]
    for (idx0, scope_ty) in enumerate(scope_tys)
        _in_generation_extension_fragment(scope_ty) || continue
        args = Ty[]
        rest = scope_ty
        for _ in 1:application_limit
            function_view = generation_function_view(rest)
            function_view === nothing && break
            push!(args, function_view.arg)
            rest = function_view.res
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

function _scope_application_rules(target_ty::Ty, scope_tys::Vector{Ty})
    written_limit = maximum(scope_tys; init=0) do ty
        count = 0
        current = ty
        while current isa TyFun
            count += 1
            current = current.res
        end
        return count
    end
    return _scope_application_rules(target_ty, scope_tys, written_limit)
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
    function GenerationWeights(default::Float64, overrides::Dict{Symbol,Float64})
        isfinite(default) && default >= 0 ||
            throw(ArgumentError("default generation weight must be finite and nonnegative"))
        all(weight -> isfinite(weight) && weight >= 0, values(overrides)) ||
            throw(ArgumentError("generation weights must be finite and nonnegative"))
        return new(default, overrides)
    end
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
        isfinite(weight) || throw(ArgumentError("generation head weight overflow for $(head.name)"))
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

"""Return the finite sampling charge of an extension derivation."""
function derivation_node_count(derivation::CoreDerivation)
    throw(ArgumentError("$(typeof(derivation)) must implement derivation_node_count"))
end

"""Return a stable structural signature for an extension derivation."""
function derivation_signature(derivation::CoreDerivation)
    throw(ArgumentError("$(typeof(derivation)) must implement derivation_signature"))
end

"""Return provider-owned result metadata for an extension derivation."""
derivation_metadata(::CoreDerivation) = Dict{Symbol,Any}()

"""Return the monomorphic result type constructed by a derivation."""
function derivation_result_type(derivation::CoreDerivation)
    throw(ArgumentError("$(typeof(derivation)) must implement derivation_result_type"))
end

"""Report whether a derivation uses one root-scope binder after `offset` binders."""
derivation_uses_scope_binder(::CoreDerivation, ::Integer, ::Integer=0) = false
derivation_calls_scope_binder(d::CoreDerivation, index::Integer, offset::Integer=0) =
    _signature_calls_scope_binder(derivation_signature(d), index + offset)

"""Declare whether a derivation may cross rigid-scheme discharge."""
rigid_safe_derivation(::CoreDerivation) = false

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

rigid_safe_derivation(d::VariableDerivation) = true
rigid_safe_derivation(d::ConstantDerivation) = _rigid_head_allowed(d.head)
rigid_safe_derivation(d::LambdaDerivation) = rigid_safe_derivation(d.body)
rigid_safe_derivation(d::HeadApplicationDerivation) =
    _rigid_head_allowed(d.head) && all(rigid_safe_derivation, d.args)
rigid_safe_derivation(d::ScopeApplicationDerivation) =
    all(rigid_safe_derivation, d.args)
_rigid_derivation_allowed(d::CoreDerivation) = rigid_safe_derivation(d)

struct DerivationSampleSet
    derivations::Vector{CoreDerivation}
    cumulative_weights::Vector{Float64}
    total_weight::Float64
end

mutable struct DerivationSamplingState{R<:Random.AbstractRNG}
    rng::R
    nodes_left::Int
end

struct KnownWitnessCacheEntry
    witness::CoreDerivation
    nodes::Int
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
    known_witnesses::Dict{Any,Union{Nothing,KnownWitnessCacheEntry}}
    restricted_relations::Dict{Any,Any}
    global_consts::Dict{Symbol,Any}
    local_consts::Dict{Symbol,Any}
    global_consts_ty::Dict{Symbol,TyLike}
    local_consts_ty::Dict{Symbol,TyLike}
    presets::Dict{Symbol,Any}
    preset_types::Dict{Symbol,Any}
    weights::GenerationWeights
    providers::Tuple
    generation_providers::Tuple
    argument_planner::A
    guarded_expansion_depth::Int
    capability_mask::Tuple{Vararg{Symbol}}
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
    capability_mask=(),
)
    global_tyctx = _typed_context(global_consts, global_consts_ty)
    local_tyctx = _typed_context(local_consts, local_consts_ty)
    provider_tuple = _provider_tuple(providers)
    return CoreRelation(
        Dict{Tuple{Any,Any},Vector{CoreDerivation}}(),
        Dict{Tuple{Any,Any},Set{Any}}(),
        Set{Tuple{Any,Any}}(),
        Set{Tuple{Any,Any}}(),
        Dict{Tuple{Any,Any},CoreCatalog}(),
        Dict{Any,Vector{NamedTuple}}(),
        Dict{Any,Vector{CoreDerivation}}(),
        Dict{Any,Any}(),
        Dict{Any,Union{Nothing,KnownWitnessCacheEntry}}(),
        Dict{Any,Any}(),
        _anydict(global_consts),
        _anydict(local_consts),
        global_tyctx,
        local_tyctx,
        _anydict(presets),
        Dict{Symbol,Any}(name => ty for (name, ty) in pairs(preset_types)),
        weights,
        provider_tuple,
        _generation_provider_tuple(provider_tuple, typeof(argument_planner)),
        argument_planner,
        _checked_nonnegative_int(:guarded_expansion_depth, guarded_expansion_depth),
        _capability_tuple(capability_mask),
    )
end

const _DEFAULT_ADD_GENERATION_OPTIONS_METHOD = which(
    add_generation_options!,
    Tuple{AbstractHeadProvider,Any,Any,Any},
)

function _generation_provider_tuple(providers::Tuple, ::Type{A}) where {A}
    return Tuple(
        provider for provider in providers if _provides_generation_options(provider, A)
    )
end

function _provides_generation_options(provider::AbstractHeadProvider, ::Type{A}) where {A}
    declared = provides_generation_options(provider)
    declared isa Union{Nothing,Bool} ||
        throw(ArgumentError("provides_generation_options must return true, false, or nothing"))
    declared === nothing || return declared
    return which(
        add_generation_options!,
        Tuple{typeof(provider),Vector{NamedTuple},CoreRelation{A},GenerationQuery},
    ) !== _DEFAULT_ADD_GENERATION_OPTIONS_METHOD
end

function _supports_recursive_type_queries(providers)
    return any(
        provider -> :recursive_type_query in _capability_tuple(provider_capabilities(provider)),
        providers,
    )
end

_supports_recursive_type_queries(config::GenerationConfig) =
    _supports_recursive_type_queries(
        _generation_provider_tuple(config.providers, typeof(config.argument_planner)),
    )

function _generation_target_supported(target_ty::Ty, config::GenerationConfig)
    _in_closed_fragment(target_ty) && return true
    return _supports_recursive_type_queries(config) &&
        _in_generation_extension_fragment(target_ty)
end

function _generation_scope_supported(scope_tys, config::GenerationConfig)
    all(_in_closed_fragment, scope_tys) && return true
    return _supports_recursive_type_queries(config) &&
        all(_in_generation_extension_fragment, scope_tys)
end

include("SafeListAccessGeneration.jl")

rigid_safe_derivation(d::SafeListAccessDerivation) =
    rigid_safe_derivation(d.source) &&
    rigid_safe_derivation(d.body) &&
    rigid_safe_derivation(d.fallback)

derivation_result_type(d::VariableDerivation) = d.ty
function derivation_result_type(d::ConstantDerivation)
    d.head.type isa Ty || throw(ArgumentError("constant derivation must be monomorphic"))
    return d.head.type
end
derivation_result_type(d::LambdaDerivation) = TyFun(d.arg_ty, derivation_result_type(d.body))
derivation_result_type(d::HeadApplicationDerivation) =
    _application_result_type(d.head.type, length(d.args))
derivation_result_type(d::ScopeApplicationDerivation) =
    _application_result_type(d.head_ty, length(d.args))
derivation_result_type(d::SafeListAccessDerivation) = d.result_ty

function _application_result_type(head_ty::TyLike, argument_count::Int)
    head_ty isa Ty || throw(ArgumentError("application derivation head must be monomorphic"))
    result = head_ty
    for _ in 1:argument_count
        view = generation_function_view(result)
        view === nothing && throw(ArgumentError("application derivation exceeds its function head"))
        result = view.res
    end
    return result
end

derivation_metadata(::VariableDerivation) = Dict{Symbol,Any}()
derivation_metadata(::ConstantDerivation) = Dict{Symbol,Any}()
derivation_metadata(d::LambdaDerivation) = copy(derivation_metadata(d.body))
derivation_metadata(d::HeadApplicationDerivation) = _merge_derivation_metadata(d.args)
derivation_metadata(d::ScopeApplicationDerivation) = _merge_derivation_metadata(d.args)
derivation_metadata(d::SafeListAccessDerivation) =
    _merge_derivation_metadata((d.source, d.body, d.fallback))

"""
    derivation_capabilities(derivation)

Return capabilities actually represented by a sampled derivation. Custom
derivations may extend this protocol; composite core derivations merge their
children, and typed heads may declare `metadata.capabilities`.
"""
derivation_capabilities(::CoreDerivation) = ()
derivation_capabilities(::VariableDerivation) = ()
derivation_capabilities(d::ConstantDerivation) = _head_derivation_capabilities(d.head)
derivation_capabilities(d::LambdaDerivation) = derivation_capabilities(d.body)
derivation_capabilities(d::HeadApplicationDerivation) =
    _merge_derivation_capabilities(_head_derivation_capabilities(d.head), d.args)
derivation_capabilities(d::ScopeApplicationDerivation) =
    _merge_derivation_capabilities((), d.args)
derivation_capabilities(d::SafeListAccessDerivation) = _merge_derivation_capabilities(
    (:safe_list_access,),
    (d.source, d.body, d.fallback),
)

_head_derivation_capabilities(head::HeadEntry) =
    hasproperty(head.metadata, :capabilities) ?
    _capability_tuple(head.metadata.capabilities) : ()

function _merge_derivation_capabilities(initial, derivations)
    out = Symbol[_capability_tuple(initial)...]
    for derivation in derivations
        for capability in derivation_capabilities(derivation)
            capability in out || push!(out, capability)
        end
    end
    return Tuple(out)
end

function _merge_derivation_metadata(derivations)
    out = Dict{Symbol,Any}()
    for derivation in derivations
        for (key, value) in pairs(derivation_metadata(derivation))
            symbol = Symbol(key)
            if !haskey(out, symbol)
                out[symbol] = deepcopy(value)
            elseif out[symbol] isa AbstractVector && value isa AbstractVector
                append!(out[symbol], value)
            elseif !isequal(out[symbol], value)
                out[symbol] = Any[out[symbol], value]
            end
        end
    end
    return out
end

derivation_uses_scope_binder(d::VariableDerivation, index::Integer, offset::Integer=0) =
    d.index == index + offset
derivation_uses_scope_binder(::ConstantDerivation, ::Integer, ::Integer=0) = false
derivation_uses_scope_binder(d::LambdaDerivation, index::Integer, offset::Integer=0) =
    derivation_uses_scope_binder(d.body, index, offset + 1)
derivation_uses_scope_binder(d::HeadApplicationDerivation, index::Integer, offset::Integer=0) =
    any(arg -> derivation_uses_scope_binder(arg, index, offset), d.args)
function derivation_uses_scope_binder(
    d::ScopeApplicationDerivation,
    index::Integer,
    offset::Integer=0,
)
    d.index == index + offset && return true
    return any(arg -> derivation_uses_scope_binder(arg, index, offset), d.args)
end

derivation_calls_scope_binder(::Union{VariableDerivation,ConstantDerivation}, ::Integer, ::Integer=0) = false
derivation_calls_scope_binder(d::LambdaDerivation, index::Integer, offset::Integer=0) =
    derivation_calls_scope_binder(d.body, index, offset + 1)
derivation_calls_scope_binder(d::HeadApplicationDerivation, index::Integer, offset::Integer=0) =
    any(arg -> derivation_calls_scope_binder(arg, index, offset), d.args)
function derivation_calls_scope_binder(
    d::ScopeApplicationDerivation,
    index::Integer,
    offset::Integer=0,
)
    d.index == index + offset && !isempty(d.args) && return true
    return any(arg -> derivation_calls_scope_binder(arg, index, offset), d.args)
end
function derivation_calls_scope_binder(
    d::SafeListAccessDerivation,
    index::Integer,
    offset::Integer=0,
)
    return derivation_calls_scope_binder(d.source, index, offset) ||
        derivation_calls_scope_binder(d.body, index, offset + 2) ||
        derivation_calls_scope_binder(d.fallback, index, offset)
end

function _signature_calls_scope_binder(signature, index::Int)
    signature isa Tuple || return false
    isempty(signature) && return false
    tag = signature[1]
    if tag === :lambda
        return length(signature) >= 3 && _signature_calls_scope_binder(signature[3], index + 1)
    elseif tag === :scope_application
        length(signature) >= 4 || return false
        arguments = signature[4]
        signature[2] == index && !isempty(arguments) && return true
        return any(argument -> _signature_calls_scope_binder(argument, index), arguments)
    elseif tag === :head
        length(signature) >= 5 || return false
        return any(argument -> _signature_calls_scope_binder(argument, index), signature[5])
    elseif tag === :safe_list_access
        length(signature) >= 7 || return false
        return _signature_calls_scope_binder(signature[5], index) ||
            _signature_calls_scope_binder(signature[6], index + 2) ||
            _signature_calls_scope_binder(signature[7], index)
    elseif tag === :function_def
        length(signature) >= 4 || return false
        return _signature_calls_scope_binder(signature[4], index + 1)
    end
    return false
end
function derivation_uses_scope_binder(
    d::SafeListAccessDerivation,
    index::Integer,
    offset::Integer=0,
)
    return derivation_uses_scope_binder(d.source, index, offset) ||
        derivation_uses_scope_binder(d.body, index, offset + 2) ||
        derivation_uses_scope_binder(d.fallback, index, offset)
end

function _attach_derivation_metadata!(metadata::Dict{Symbol,Any}, derivation::CoreDerivation)
    extra = derivation_metadata(derivation)
    isempty(extra) && return metadata
    metadata[:derivation_metadata] = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(extra))
    return metadata
end

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
    _generation_target_supported(target_ty, config) ||
        return Err(UnsupportedCanonicalGenerationTarget(target_ty))

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
    _attach_derivation_metadata!(metadata, derivation)
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
`capability_mask` transitively excludes declared provider/head capabilities and
is part of the scoped relation-cache identity.
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
    capability_mask=(),
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
            capability_mask=capability_mask,
            _cache_relation=false,
        )
        result isa GenerationResult || return result
        return _discharge_rigid_result(result, target_ty, demand)
    end
    target_ty isa Ty || return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    _generation_target_supported(target_ty, config) ||
        return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    scope = Ty[scope_tys...]
    _generation_scope_supported(scope, config) ||
        return Err(UnsupportedCanonicalGenerationTarget(target_ty))

    result_local_consts = _anydict(local_consts)
    result_local_const_types = _typed_context(local_consts, local_consts_ty)
    relation_local_consts = source_local_consts ? result_local_consts : Dict{Symbol,Any}()
    relation_local_const_types = source_local_consts ? result_local_const_types : Dict{Symbol,TyLike}()
    mask = _capability_tuple(capability_mask)
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
        mask,
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
            mask,
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
        capability_mask=mask,
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
    derivation = _sample_root_generation_derivation(
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
    _attach_derivation_metadata!(metadata, derivation)
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
        derivation = _sample_root_generation_derivation(
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
        _attach_derivation_metadata!(metadata, derivation)
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
    _attach_derivation_metadata!(metadata, derivation)
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
    capability_mask=(),
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
        _capability_tuple(capability_mask),
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

_cacheable_provider_key(provider::AbstractHeadProvider) = generation_provider_cache_key(provider)

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

# Core relation keys are intentionally structural inside the closed fragment.
# Context-level `TyLike` keys use semantic μ identity because they decide cache
# interchangeability across schemes and recursive representations.
_cache_type_key(ty::TyConst) = (:const, ty.name)
_cache_type_key(ty::TyVar) =
    ty.instance === nothing ?
    (:var, ty.uid) :
    (:linked_var, ty.uid, TypeSystem.typekey(ty))
_cache_type_key(ty::TyNat) = (:nat, ty.value)
_cache_type_key(ty::TyList) = (:list, _cache_type_key(ty.elem))
_cache_type_key(ty::TyArray) = (:array, _cache_type_key(ty.len), _cache_type_key(ty.elem))
_cache_type_key(ty::TyFun) = (:fun, _cache_type_key(ty.arg), _cache_type_key(ty.res))
_cache_type_key(ty::TyProd) = (:prod, _cache_type_key(ty.left), _cache_type_key(ty.right))
_cache_type_key(ty::TySum) = (:sum, _cache_type_key(ty.left), _cache_type_key(ty.right))
_cache_type_key(ty::TyMu) = (:mu, _cache_type_key(ty.var), _cache_type_key(ty.body))
_cache_tylike_key(ty::TyLike) = TypeSystem.semantic_typekey(ty)
_closed_type_key(ty::Ty) = _cache_type_key(ty)

function _merge_capabilities(inherited::Tuple{Vararg{Symbol}}, requested)
    additions = _capability_tuple(requested)
    isempty(additions) && return inherited
    out = Symbol[inherited...]
    for capability in additions
        capability in out || push!(out, capability)
    end
    return Tuple(out)
end

function _merge_requirements(inherited::Tuple, requested)
    additions = _requirement_tuple(requested)
    isempty(additions) && return inherited
    out = AbstractGenerationRequirement[inherited...]
    for requirement in additions
        any(existing -> isequal(existing, requirement), out) || push!(out, requirement)
    end
    return Tuple(out)
end

function _derive_generation_query(
    parent::GenerationQuery;
    target_ty::Ty=parent.target_ty,
    scope_tys=nothing,
    added_scope_tys=nothing,
    depth_left::Integer=parent.depth_left,
    guarded_left::Integer=parent.guarded_left,
    provider_fuel::Integer=parent.provider_fuel,
    node_budget::Integer=parent.node_budget,
    capability_mask=(),
    requirements=(),
)
    depth = _checked_nonnegative_int(:depth_left, depth_left)
    guarded = _checked_nonnegative_int(:guarded_left, guarded_left)
    fuel = _checked_nonnegative_int(:provider_fuel, provider_fuel)
    nodes = _checked_nonnegative_int(:node_budget, node_budget)
    depth <= parent.depth_left || throw(ArgumentError("nested generation cannot replenish depth fuel"))
    guarded <= parent.guarded_left || throw(ArgumentError("nested generation cannot replenish guarded fuel"))
    fuel <= parent.provider_fuel || throw(ArgumentError("nested generation cannot replenish provider fuel"))
    nodes <= parent.node_budget || throw(ArgumentError("nested generation cannot replenish node fuel"))
    scope_tys === nothing || added_scope_tys === nothing ||
        throw(ArgumentError("pass either scope_tys or added_scope_tys, not both"))
    if added_scope_tys !== nothing
        added_scope = _scope_type_tuple(added_scope_tys)
        child_scope = (added_scope..., parent.scope_tys...)
        inherited_requirements = _rebase_inherited_requirements(parent, length(added_scope))
    elseif scope_tys === nothing
        child_scope = parent.scope_tys
        inherited_requirements = parent.requirements
    else
        isempty(parent.requirements) || throw(ArgumentError(
            "queries with inherited requirements must omit scope_tys and use " *
            "added_scope_tys to enter binders",
        ))
        child_scope = _scope_type_tuple(scope_tys)
        inherited_requirements = ()
    end
    return GenerationQuery(
        target_ty,
        child_scope,
        depth,
        guarded,
        fuel,
        nodes,
        _merge_capabilities(parent.capability_mask, capability_mask),
        _merge_requirements(inherited_requirements, requirements),
    )
end

function _rebase_inherited_requirements(parent::GenerationQuery, added::Int)
    isempty(parent.requirements) && return ()
    return Tuple(
        rebase_generation_requirement(requirement, added)
        for requirement in parent.requirements
    )
end

function _effective_generation_query(relation::CoreRelation, query::GenerationQuery)
    all(capability -> capability in query.capability_mask, relation.capability_mask) &&
        return query
    return GenerationQuery(
        query.target_ty,
        query.scope_tys,
        query.depth_left,
        query.guarded_left,
        query.provider_fuel,
        query.node_budget,
        _merge_capabilities(relation.capability_mask, query.capability_mask),
        query.requirements,
    )
end

function _provider_admitted(provider::AbstractHeadProvider, capability_mask::Tuple)
    isempty(capability_mask) && return true
    capabilities = _capability_tuple(provider_capabilities(provider))
    :opaque in capabilities && return false
    return !any(capability -> capability in capability_mask, capabilities)
end

_admitted_providers(providers::Tuple, ::Tuple{}) = providers
_admitted_providers(providers::Tuple, capability_mask::Tuple) =
    Tuple(provider for provider in providers if _provider_admitted(provider, capability_mask))

_cacheable_requirement_key(requirement::UsesScopeBinder) = (:uses_scope_binder, requirement.index)
_cacheable_requirement_key(requirement::CallsScopeBinder) = (:calls_scope_binder, requirement.index)
_cacheable_requirement_key(requirement::ForbidsScopeBinder) =
    (:forbids_scope_binder, requirement.index)
function _cacheable_requirement_key(requirement::AbstractGenerationRequirement)
    key = generation_requirement_cache_key(requirement)
    key === nothing && Base.issingletontype(typeof(requirement)) &&
        return (:requirement, typeof(requirement))
    return key
end

function _cacheable_requirements_key(requirements::Tuple)
    entries = Any[]
    for requirement in requirements
        key = _cacheable_requirement_key(requirement)
        key === nothing && return nothing
        push!(entries, key)
    end
    return Tuple(entries)
end

function _rigid_demand_key(query::GenerationQuery)
    rigid_types = Any[]
    _contains_rigid_dummy(query.target_ty) && push!(rigid_types, (:target, _cache_type_key(query.target_ty)))
    for (index, ty) in enumerate(query.scope_tys)
        _contains_rigid_dummy(ty) || continue
        push!(rigid_types, (:scope, index - 1, _cache_type_key(ty)))
    end
    return Tuple(rigid_types)
end

function _generation_query_cache_key(query::GenerationQuery, admitted_providers::Tuple)
    provider_key = _cacheable_providers_key(admitted_providers)
    provider_key === nothing && return nothing
    requirement_key = _cacheable_requirements_key(query.requirements)
    requirement_key === nothing && return nothing
    return (
        :generation_query,
        query.capability_mask,
        provider_key,
        _cache_type_key(query.target_ty),
        Tuple(_cache_type_key.(query.scope_tys)),
        query.depth_left,
        query.guarded_left,
        query.provider_fuel,
        query.node_budget,
        _rigid_demand_key(query),
        requirement_key,
    )
end

function _same_provider_tuple(left::Tuple, right::Tuple)
    length(left) == length(right) || return false
    return all(left_provider === right_provider for (left_provider, right_provider) in zip(left, right))
end

function _relation_for_query(relation::CoreRelation, query::GenerationQuery)
    isempty(relation.capability_mask) && isempty(query.capability_mask) &&
        return relation, query
    effective = _effective_generation_query(relation, query)
    admitted = _admitted_providers(relation.providers, effective.capability_mask)
    same_context =
        _same_provider_tuple(admitted, relation.providers) &&
        effective.capability_mask == relation.capability_mask
    same_context && return relation, effective

    restricted = _restricted_relation(relation, admitted, effective.capability_mask)
    return restricted, effective
end

function _restricted_relation(
    relation::CoreRelation,
    admitted::Tuple,
    capability_mask::Tuple,
)
    cache_key = _restricted_relation_cache_key(relation, capability_mask)
    cache_key === nothing && return _new_restricted_relation(
        relation,
        admitted,
        capability_mask,
    )
    restricted = get!(relation.restricted_relations, cache_key) do
        _new_restricted_relation(relation, admitted, capability_mask)
    end
    restricted.local_consts = relation.local_consts
    restricted.local_consts_ty = relation.local_consts_ty
    return restricted
end

function _restricted_relation_cache_key(relation::CoreRelation, capability_mask::Tuple)
    global_key = _cacheable_global_consts_key(
        relation.global_consts,
        relation.global_consts_ty,
    )
    global_key === nothing && return nothing
    local_key = _cacheable_local_context_key(
        relation.local_consts,
        relation.local_consts_ty,
    )
    local_key === nothing && return nothing
    isempty(relation.presets) && isempty(relation.preset_types) || return nothing
    return (capability_mask, global_key, local_key)
end
function _new_restricted_relation(
    relation::CoreRelation,
    admitted::Tuple,
    capability_mask::Tuple,
)
    return CoreRelation(
        global_consts=relation.global_consts,
        global_consts_ty=relation.global_consts_ty,
        local_consts=relation.local_consts,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        weights=relation.weights,
        providers=admitted,
        argument_planner=relation.argument_planner,
        guarded_expansion_depth=relation.guarded_expansion_depth,
        capability_mask=capability_mask,
    )
end

"""
    known_generation_witness(relation, query; capability_mask=())

Extract one minimum-cost finite acyclic witness from the deterministic
non-recursive core.
This provider-facing query respects the inherited capability mask and finite
depth/node bounds but never enumerates all proof terms. In particular, a rule
such as `f : A → A` cannot create the infinite family `f a`, `f (f a)`, ... .
"""
function known_generation_witness(
    relation::CoreRelation,
    parent_query::GenerationQuery;
    capability_mask=(),
)
    return _known_generation_witness(
        relation,
        parent_query,
        capability_mask,
        false,
    )
end

function _borrow_known_generation_witness(
    relation::CoreRelation,
    parent_query::GenerationQuery;
    capability_mask=(),
)
    return _known_generation_witness(
        relation,
        parent_query,
        capability_mask,
        true,
    )
end

function _known_generation_witness(
    relation::CoreRelation,
    parent_query::GenerationQuery,
    capability_mask,
    borrow_cached::Bool,
)
    query = _derive_generation_query(parent_query; capability_mask)
    cache_key = _known_witness_cache_key(relation, query)
    if cache_key !== nothing && haskey(relation.known_witnesses, cache_key)
        entry = relation.known_witnesses[cache_key]
        entry === nothing && return nothing
        entry.nodes <= query.node_budget || return nothing
        return borrow_cached ? entry.witness : _copy_generation_witness(entry.witness)
    end
    restricted, effective = _fresh_known_relation(relation, query)
    forbidden = _forbidden_scope_binders(effective)
    memo = Dict{Any,Union{Nothing,CoreDerivation}}()
    witness = _known_generation_witness!(
        restricted,
        effective.target_ty,
        Ty[effective.scope_tys...],
        effective.depth_left,
        memo,
        forbidden,
    )
    nodes = witness === nothing ? 0 : derivation_node_count(witness)
    if cache_key !== nothing
        relation.known_witnesses[cache_key] = if witness === nothing
            nothing
        elseif borrow_cached
            KnownWitnessCacheEntry(witness, nodes)
        else
            KnownWitnessCacheEntry(_copy_generation_witness(witness), nodes)
        end
    end
    witness === nothing && return nothing
    return nodes <= effective.node_budget ? witness : nothing
end

function _known_witness_cache_key(relation::CoreRelation, query::GenerationQuery)
    effective = _effective_generation_query(relation, query)
    admitted = _admitted_providers(relation.providers, effective.capability_mask)
    providers_key = _cacheable_providers_key(admitted)
    providers_key === nothing && return nothing
    global_key = _cacheable_global_consts_key(
        relation.global_consts,
        relation.global_consts_ty,
    )
    global_key === nothing && return nothing
    local_key = _cacheable_local_context_key(
        relation.local_consts,
        relation.local_consts_ty,
    )
    local_key === nothing && return nothing
    isempty(relation.presets) && isempty(relation.preset_types) || return nothing
    return (
        :known_witness,
        _cache_type_key(effective.target_ty),
        Tuple(_cache_type_key.(effective.scope_tys)),
        effective.depth_left,
        effective.capability_mask,
        _weights_key(relation.weights),
        relation.guarded_expansion_depth,
        global_key,
        local_key,
        providers_key,
        _forbidden_scope_binders(effective),
    )
end

function _forbidden_scope_binders(query::GenerationQuery)
    indices = Int[
        requirement.index
        for requirement in query.requirements
        if requirement isa ForbidsScopeBinder
    ]
    sort!(unique!(indices))
    return Tuple(indices)
end

_shift_scope_binders(indices::Tuple, added::Int) =
    added == 0 ? indices : Tuple(index + added for index in indices)

function _fresh_known_relation(relation::CoreRelation, query::GenerationQuery)
    effective = _effective_generation_query(relation, query)
    admitted = _admitted_providers(relation.providers, effective.capability_mask)
    same_context =
        _same_provider_tuple(admitted, relation.providers) &&
        effective.capability_mask == relation.capability_mask
    restricted = same_context ? CoreRelation(
        global_consts=relation.global_consts,
        global_consts_ty=relation.global_consts_ty,
        local_consts=relation.local_consts,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        weights=relation.weights,
        providers=admitted,
        argument_planner=relation.argument_planner,
        guarded_expansion_depth=relation.guarded_expansion_depth,
        capability_mask=effective.capability_mask,
    ) : _restricted_relation(relation, admitted, effective.capability_mask)
    return restricted, effective
end

_copy_generation_witness(derivation::VariableDerivation) = derivation
_copy_generation_witness(derivation::ConstantDerivation) = derivation
_copy_generation_witness(derivation::LambdaDerivation) =
    LambdaDerivation(derivation.arg_ty, _copy_generation_witness(derivation.body))
_copy_generation_witness(derivation::HeadApplicationDerivation) =
    HeadApplicationDerivation(
        derivation.head,
        CoreDerivation[_copy_generation_witness(argument) for argument in derivation.args],
    )
_copy_generation_witness(derivation::ScopeApplicationDerivation) =
    ScopeApplicationDerivation(
        derivation.index,
        derivation.head_ty,
        CoreDerivation[_copy_generation_witness(argument) for argument in derivation.args],
    )
_copy_generation_witness(derivation::SafeListAccessDerivation) =
    SafeListAccessDerivation(
        _copy_generation_witness(derivation.source),
        derivation.source_ty,
        _copy_generation_witness(derivation.body),
        _copy_generation_witness(derivation.fallback),
        derivation.result_ty,
    )

function _known_generation_witness!(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    memo,
    forbidden::Tuple=(),
)
    _in_generation_extension_fragment(target_ty) || return nothing
    all(_in_generation_extension_fragment, scope_tys) || return nothing

    fact_key = _known_key(scope_tys, target_ty)
    memo_key = (fact_key, depth_left, forbidden)
    haskey(memo, memo_key) && return memo[memo_key]
    active_key = (:active_known_generation_witness, fact_key, forbidden)
    haskey(memo, active_key) && return nothing
    witness = _direct_known_generation_witness(relation, target_ty, scope_tys, forbidden)
    if witness !== nothing
        memo[memo_key] = witness
        return witness
    end
    memo[active_key] = nothing
    try
        depth_left > 0 && (witness = _compound_known_generation_witness(
            relation,
            target_ty,
            scope_tys,
            depth_left,
            memo,
            forbidden,
        ))
    finally
        delete!(memo, active_key)
    end
    memo[memo_key] = witness
    return witness
end

function _direct_known_generation_witness(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    forbidden::Tuple=(),
)
    if _bucket_enabled(relation.weights, _VARIABLE_WEIGHT)
        for (idx0, scope_ty) in enumerate(scope_tys)
            idx0 - 1 in forbidden && continue
            iscompatible(scope_ty, target_ty) || continue
            return VariableDerivation(idx0 - 1, scope_ty)
        end
    end

    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for head in catalog.heads
        head.type isa Ty || continue
        head.rule_class isa CoreProductive || continue
        _known_head_admitted(relation, head) || continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        _head_has_value(head, relation) || continue
        iscompatible(head.type, target_ty) || continue
        return ConstantDerivation(head)
    end
    for head in _extension_context_heads(relation)
        _known_head_admitted(relation, head) || continue
        _bucket_enabled(relation.weights, head.weight_bucket) || continue
        iscompatible(head.type, target_ty) || continue
        return ConstantDerivation(head)
    end
    return nothing
end

function _compound_known_generation_witness(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    memo,
    forbidden::Tuple=(),
)
    best = nothing
    if target_ty isa TyFun && _bucket_enabled(relation.weights, _LAMBDA_WEIGHT)
        body = _known_generation_witness!(
            relation,
            target_ty.res,
            Ty[target_ty.arg; scope_tys],
            depth_left - 1,
            memo,
            _shift_scope_binders(forbidden, 1),
        )
        if body !== nothing
            best = _prefer_generation_witness(best, LambdaDerivation(target_ty.arg, body))
        end
    end

    if _scope_application_enabled(relation.weights)
        for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
            rule.index in forbidden && continue
            _known_type_admitted(relation, rule.head_ty) || continue
            application_cost = length(rule.argument_tys)
            args = _known_scope_generation_arguments(
                relation,
                rule.argument_tys,
                scope_tys,
                depth_left - application_cost,
                memo,
                forbidden,
            )
            args === nothing && continue
            candidate = ScopeApplicationDerivation(rule.index, rule.head_ty, args)
            best = _prefer_generation_witness(best, candidate)
        end
    end

    for rule in _known_generation_rules(relation, target_ty, scope_tys)
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        isempty(rule.argument_tys) && continue
        args = _known_generation_arguments(
            relation,
            rule,
            scope_tys,
            depth_left - 1,
            memo,
            forbidden,
        )
        args === nothing && continue
        candidate = HeadApplicationDerivation(rule.head, args)
        best = _prefer_generation_witness(best, candidate)
    end
    return best
end

function _known_generation_arguments(
    relation::CoreRelation,
    rule::CoreRule,
    scope_tys::Vector{Ty},
    depth_left::Int,
    memo,
    forbidden::Tuple=(),
)
    depth_left >= 0 || return nothing
    args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
    for index in _argument_resolution_order(rule)
        argument_ty = rule.argument_tys[index]
        argument = if rule.argument_classes[index] == :source_array
            _known_source_array_witness(
                relation,
                argument_ty,
                scope_tys,
                depth_left,
                _source_array_depth_limit(rule),
                memo,
                forbidden,
            )
        else
            _known_generation_witness!(
                relation,
                argument_ty,
                scope_tys,
                depth_left,
                memo,
                forbidden,
            )
        end
        argument === nothing && return nothing
        args[index] = argument
    end
    return args
end

function _known_scope_generation_arguments(
    relation::CoreRelation,
    argument_tys::Vector{Ty},
    scope_tys::Vector{Ty},
    depth_left::Int,
    memo,
    forbidden::Tuple=(),
)
    depth_left >= 0 || return nothing
    args = CoreDerivation[]
    for argument_ty in argument_tys
        argument = if _argument_class(argument_ty) == :source_array
            _known_source_array_witness(
                relation,
                argument_ty,
                scope_tys,
                depth_left,
                0,
                memo,
                forbidden,
            )
        else
            _known_generation_witness!(
                relation,
                argument_ty,
                scope_tys,
                depth_left,
                memo,
                forbidden,
            )
        end
        argument === nothing && return nothing
        push!(args, argument)
    end
    return args
end

function _known_source_array_witness(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    source_depth_left::Int,
    memo,
    forbidden::Tuple=(),
)
    _bucket_enabled(relation.weights, _ARRAY_SOURCE_WEIGHT) || return nothing
    best = nothing
    for candidate in _grounded_derivations(relation, target_ty, scope_tys)
        any(index -> derivation_uses_scope_binder(candidate, index), forbidden) && continue
        best = _prefer_generation_witness(best, candidate)
    end
    depth_left > 0 && source_depth_left > 0 || return best

    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        _is_source_array_constructor(rule.head) || continue
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
        _bucket_enabled(relation.weights, rule.head.weight_bucket) || continue
        _head_has_value(rule.head, relation) || continue
        source_index = _source_array_argument_index(rule.head, length(rule.argument_tys))
        args = Vector{CoreDerivation}(undef, length(rule.argument_tys))
        valid = true
        for index in _argument_resolution_order(rule)
            argument_ty = rule.argument_tys[index]
            argument = if index == source_index && rule.argument_classes[index] == :source_array
                _known_source_array_witness(
                    relation,
                    argument_ty,
                    scope_tys,
                    depth_left - 1,
                    source_depth_left - 1,
                    memo,
                    forbidden,
                )
            else
                _known_generation_witness!(
                    relation,
                    argument_ty,
                    scope_tys,
                    depth_left - 1,
                    memo,
                    forbidden,
                )
            end
            if argument === nothing
                valid = false
                break
            end
            args[index] = argument
        end
        valid || continue
        best = _prefer_generation_witness(best, HeadApplicationDerivation(rule.head, args))
    end
    return best
end

function _prefer_generation_witness(current, candidate)
    candidate === nothing && return current
    current === nothing && return candidate
    return derivation_node_count(candidate) < derivation_node_count(current) ? candidate : current
end

function _extension_context_heads(relation::CoreRelation)
    any(provider -> provider isa ContextHeadProvider, relation.providers) || return HeadEntry[]
    heads = HeadEntry[]
    for (source, type_context, bucket) in (
        (:constant, relation.global_consts_ty, _CONSTANT_WEIGHT),
        (:variable, relation.local_consts_ty, _VARIABLE_WEIGHT),
        (
            :preset,
            _typed_presets(
                relation.presets,
                relation.preset_types;
                global_consts_ty=relation.global_consts_ty,
                local_consts_ty=relation.local_consts_ty,
            ),
            _PRESET_TERM_WEIGHT,
        ),
    )
        for name in sort!(collect(keys(type_context)))
            head_ty = type_context[name]
            head_ty isa Ty || continue
            _in_generation_extension_fragment(head_ty) || continue
            push!(heads, HeadEntry(
                name,
                source,
                head_ty,
                CoreProductive(),
                bucket;
                metadata=_context_source_metadata(source),
            ))
        end
    end
    return heads
end

function _known_generation_rules(
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
)
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    rules = copy(catalog.rules)
    signatures = Set{Any}()
    for rule in rules
        push!(
            signatures,
            (rule.head.source, rule.head.name, typekey(rule.head.type), Tuple(typekey.(rule.argument_tys))),
        )
    end
    for head in (catalog.heads..., _extension_context_heads(relation)...)
        head.rule_class isa CoreProductive || continue
        head.type isa Ty || continue
        _in_generation_extension_fragment(head.type) || continue
        args, result = _split_fun(head.type)
        isempty(args) && continue
        iscompatible(result, target_ty) || continue
        all(_in_generation_extension_fragment, args) || continue
        signature = (head.source, head.name, typekey(head.type), Tuple(typekey.(args)))
        signature in signatures && continue
        push!(signatures, signature)
        push!(rules, CoreRule(head, result, args, [_argument_class(arg) for arg in args]))
    end
    return rules
end

function _known_head_admitted(relation::CoreRelation, head::HeadEntry)
    capabilities = _capability_tuple(_metadata_get(head.metadata, :capabilities, ()))
    if head.type isa Ty && _is_syntactic_fixed_point_type(head.type)
        :fixed_point in capabilities || (capabilities = (capabilities..., :fixed_point))
    end
    return all(capability -> capability ∉ relation.capability_mask, capabilities)
end

function _known_type_admitted(relation::CoreRelation, ty::Ty)
    :fixed_point in relation.capability_mask && _is_syntactic_fixed_point_type(ty) &&
        return false
    return true
end

function _is_syntactic_fixed_point_type(ty::Ty)
    outer = generation_function_view(ty)
    outer === nothing && return false
    step = generation_function_view(outer.arg)
    step === nothing && return false
    return iscompatible(step.arg, outer.res) && iscompatible(step.res, outer.res)
end

function _extension_option_weight(option::NamedTuple, weights::GenerationWeights)
    if hasproperty(option, :weight)
        weight = option.weight
        weight isa Real || throw(ArgumentError("generation extension weight must be real"))
        converted = Float64(weight)
    elseif hasproperty(option, :weight_bucket)
        bucket = option.weight_bucket
        bucket isa Symbol && (bucket = WeightBucket(bucket))
        bucket isa CoreWeightBucket ||
            throw(ArgumentError("generation extension weight_bucket must be a symbol or CoreWeightBucket"))
        converted = Float64(_bucket_weight(weights, bucket))
    else
        throw(ArgumentError("generation extension options require weight or weight_bucket"))
    end
    isfinite(converted) && converted >= 0 ||
        throw(ArgumentError("generation extension weight must be finite and nonnegative"))
    return converted
end

function _generation_extension_options(relation::CoreRelation, query::GenerationQuery)
    query.depth_left > 0 && query.provider_fuel > 0 || return NamedTuple[]
    admitted = _admitted_providers(relation.generation_providers, query.capability_mask)
    recursive_query = !_in_closed_fragment(query.target_ty) ||
        !all(_in_closed_fragment, query.scope_tys)
    if recursive_query
        admitted = Tuple(
            provider for provider in admitted
            if :recursive_type_query in _capability_tuple(provider_capabilities(provider))
        )
    end
    admitted = Tuple(
        provider for provider in admitted
        if generation_provider_applicable(provider, relation, query)
    )
    isempty(admitted) && return NamedTuple[]
    query_key = _generation_query_cache_key(query, admitted)
    cache_key = query_key === nothing ? nothing : (:generation_extensions, query_key)
    if cache_key !== nothing
        cached = get(relation.sampling_options, cache_key, nothing)
        cached === nothing || return cached
    end

    options = NamedTuple[]
    for provider in admitted
        provider_options = NamedTuple[]
        add_generation_options!(provider, provider_options, relation, query)
        for option in provider_options
            push!(
                options,
                (
                    kind=:generation_extension,
                    value=(provider=provider, option=option),
                    weight=_extension_option_weight(option, relation.weights),
                ),
            )
        end
    end
    cache_key === nothing || (relation.sampling_options[cache_key] = options)
    return options
end

function _generation_extensions_exhausted(
    relation::CoreRelation,
    query::GenerationQuery,
)
    any(generation_requirement_affects_core_search, query.requirements) && return false
    admitted = _admitted_providers(relation.generation_providers, query.capability_mask)
    return all(
        provider -> generation_provider_exhausted(provider, relation, query),
        admitted,
    )
end

"""
    sample_generation_extension(state, relation, parent_query; kwargs...)

Sample a general nested query from an extension while preserving `state` and
the parent's masks and requirements. Requested depth, guarded depth, provider
fuel, and node budget must not exceed the parent values; attempts to replenish them
throw `ArgumentError`. Extension dispatch itself consumes one depth unit and
one provider-fuel unit before invoking the provider-specific overload.
"""
function sample_generation_extension(
    state::DerivationSamplingState,
    relation::CoreRelation,
    parent_query::GenerationQuery;
    target_ty::Ty=parent_query.target_ty,
    scope_tys=nothing,
    added_scope_tys=nothing,
    depth_left::Integer=parent_query.depth_left,
    guarded_left::Integer=parent_query.guarded_left,
    provider_fuel::Integer=parent_query.provider_fuel,
    node_budget::Integer=parent_query.node_budget,
    capability_mask=(),
    requirements=(),
    allow_safe_list_access::Bool=true,
)
    state.nodes_left = clamp(state.nodes_left, 0, parent_query.node_budget)
    query = _derive_generation_query(
        parent_query;
        target_ty,
        scope_tys,
        added_scope_tys,
        depth_left,
        guarded_left,
        provider_fuel,
        node_budget=min(node_budget, state.nodes_left),
        capability_mask,
        requirements,
    )
    return _sample_generation_derivation(
        state,
        relation,
        query,
        allow_safe_list_access,
    )
end

"""
    sample_generation_derivation_using_scope_binder(
        state, relation, parent_query, binder_ty; kwargs...
    )

Sample one finite derivation after prepending `binder_ty` at De Bruijn index
zero, and require the returned derivation to use that exact binder. This is the
constructive entry for extensions such as guarded recursion; it shares the
caller's node ledger, masks, requirements, and decreasing query measures.

`allow_direct_result=false` excludes only the bare new binder as the complete
result. Nested uses, including applications, branches, and safe-list success
bodies, remain available. `accept_derivation`, when supplied, is a finite
callable over a complete proposal. A rejection restores the node ledger and
ends that proposal; caller-owned retry budgets decide whether to sample again.
Omitting it preserves the ordinary route distribution exactly.
`require_scope_application=true` strengthens binder use to a genuine
application-head occurrence and uses the finite required-binder certificate;
free child goals still use ordinary sampling before the certified fallback.
"""
function sample_generation_derivation_using_scope_binder(
    state::DerivationSamplingState,
    relation::CoreRelation,
    parent_query::GenerationQuery,
    binder_ty::Ty;
    target_ty::Ty=parent_query.target_ty,
    depth_left::Integer=parent_query.depth_left,
    guarded_left::Integer=parent_query.guarded_left,
    provider_fuel::Integer=parent_query.provider_fuel,
    node_budget::Integer=parent_query.node_budget,
    capability_mask=(),
    requirements=(),
    allow_safe_list_access::Bool=true,
    allow_direct_result::Bool=true,
    require_scope_application::Bool=false,
    accept_derivation=nothing,
)
    state.nodes_left = clamp(state.nodes_left, 0, parent_query.node_budget)
    nested_requirements = (
        _requirement_tuple(requirements)...,
        require_scope_application ? CallsScopeBinder(0) : UsesScopeBinder(0),
    )
    query = _derive_generation_query(
        parent_query;
        target_ty,
        added_scope_tys=(binder_ty,),
        depth_left,
        guarded_left,
        provider_fuel,
        node_budget=min(node_budget, state.nodes_left),
        capability_mask,
        requirements=nested_requirements,
    )
    query_relation, effective = _relation_for_query(relation, query)
    if require_scope_application
        saved_nodes = state.nodes_left
        derivation = _sample_required_scope_certificate_derivation(
            state,
            query_relation,
            effective.target_ty,
            Ty[effective.scope_tys...],
            effective.depth_left,
            effective.guarded_left,
            allow_safe_list_access,
            effective,
        )
        if derivation === nothing ||
           (accept_derivation !== nothing && !accept_derivation(derivation))
            state.nodes_left = saved_nodes
            return nothing
        end
        return derivation
    end
    return _sample_proxy_using_derivation(
        state,
        query_relation,
        effective.target_ty,
        Ty[effective.scope_tys...],
        effective.depth_left,
        effective.guarded_left,
        1,
        0;
        allow_safe_list_access,
        allow_direct_result,
        generation_query=effective,
        accept_derivation,
    )
end

@testitem "CanonicalGeneration variable cache keys use logical identity" begin
    using Test
    using LambdaRegression: TypeSystem

    CG = LambdaRegression.CanonicalGeneration
    first = TypeSystem.make_tyvar_factory()(0)
    second = TypeSystem.make_tyvar_factory()(0)
    copied = deepcopy(first)

    @test first.id == second.id
    @test CG._cache_type_key(first) != CG._cache_type_key(second)
    @test CG._cache_type_key(first) == CG._cache_type_key(copied)
end

@testitem "CanonicalGeneration structural runtime keys erase tuple shape" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyList, typekey

    CG = LambdaRegression.CanonicalGeneration

    same_shape = (:root, (1, (:leaf, 2)))
    copied_shape = (:root, (1, (:leaf, 2)))
    different_shape = (:root, ((1, :leaf), 2))
    same_key = CG._StructuralKey(same_shape)
    copied_key = CG._StructuralKey(copied_shape)
    different_key = CG._StructuralKey(different_shape)

    @test isequal(same_key, copied_key)
    @test same_key == copied_key
    @test hash(same_key) == hash(copied_key)
    @test !isequal(same_key, different_key)
    @test typeof(same_key) === typeof(copied_key) === typeof(different_key)

    first_known = CG._known_key(Ty[FloatTy], FloatTy)
    second_known = CG._known_key(Ty[TyList(FloatTy)], TyList(FloatTy))
    @test typeof(first_known) === typeof(second_known)
    first_scope_key, first_target_key = first_known
    @test first_scope_key === first(first_known)
    @test first_target_key === last(first_known)

    derivation = CG.VariableDerivation(0, FloatTy)
    public_signature = CG.derivation_signature(derivation)
    @test public_signature == (:var, 0, typekey(FloatTy))
    @test public_signature isa Tuple
    @test !(public_signature isa CG._StructuralKey)

    signature_set = Set{Any}()
    push!(signature_set, CG._StructuralKey(public_signature))
    push!(signature_set, CG._StructuralKey((:var, 0, typekey(FloatTy))))
    @test length(signature_set) == 1
end

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
    capability_mask=(),
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
        capability_mask=capability_mask,
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

struct _StructuralKey
    value::Any
    hash_value::UInt
end

function _structural_hash(value)
    Base.@nospecialize value
    value isa Tuple || return hash(value)
    result = hash(:_structural_tuple, UInt(0))
    result = hash(length(value), result)
    for item in value
        result = hash(_structural_hash(item), result)
    end
    return result
end

function _StructuralKey(value::Any)
    Base.@nospecialize value
    return _StructuralKey(value, _structural_hash(value))
end

function _structural_isequal(left, right)
    Base.@nospecialize left right
    left isa Tuple || return !(right isa Tuple) && isequal(left, right)
    right isa Tuple || return false
    length(left) == length(right) || return false
    for index in eachindex(left)
        _structural_isequal(left[index], right[index]) || return false
    end
    return true
end

Base.hash(key::_StructuralKey, h::UInt) = hash(key.hash_value, h)
Base.isequal(left::_StructuralKey, right::_StructuralKey) =
    left.hash_value == right.hash_value && _structural_isequal(left.value, right.value)
Base.:(==)(left::_StructuralKey, right::_StructuralKey) = isequal(left, right)

_known_key(scope_tys::Vector{Ty}, target_ty::Ty) =
    (_StructuralKey(Tuple(typekey.(scope_tys))), _StructuralKey(typekey(target_ty)))

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
        _known_head_admitted(relation, head) || continue
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
        _known_head_admitted(relation, rule.head) || continue
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
            derivs = _plan_argument_derivations(
                relation,
                rule,
                raw_derivations,
                arg_ty,
                arg_class,
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

function _plan_argument_derivations(
    relation::CoreRelation,
    rule::CoreRule,
    derivations::Vector{CoreDerivation},
    arg_ty::Ty,
    arg_class::Symbol,
)
    local_planner = _metadata_get(rule.head.metadata, :argument_planner, nothing)
    planner = local_planner === nothing ? relation.argument_planner : local_planner
    return planner(derivations, arg_ty, arg_class, rule, relation)
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
        _known_head_admitted(relation, head) || continue
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

    signatures = Set{Any}()
    for derivation in out
        push!(signatures, _StructuralKey(_derivation_signature(derivation)))
    end
    catalog = _relation_catalog(relation, target_ty, scope_tys)
    for rule in catalog.rules
        _is_source_array_constructor(rule.head) || continue
        iscompatible(rule.result_ty, target_ty) || continue
        _known_head_admitted(relation, rule.head) || continue
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
            derivs = _plan_argument_derivations(
                relation,
                rule,
                raw_derivations,
                arg_ty,
                arg_class,
            )
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
    for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
        _known_type_admitted(relation, rule.head_ty) || continue
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
        planned = planner === nothing ?
            derivations :
            planner(derivations, arg_ty, arg_class, rule, relation)
        return [derivation for derivation in planned if _array_source_depth(derivation) <= max_depth]
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
        _known_head_admitted(relation, head) || continue
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
    signature = _StructuralKey(_derivation_signature(derivation))
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
        isfinite(weight) && weight >= 0 ||
            throw(ArgumentError("derivation weight must be finite and nonnegative"))
        weight > 0 || continue
        total += weight
        isfinite(total) || throw(ArgumentError("derivation weight sum overflow"))
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

function _sample_root_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool=true,
)
    if isempty(relation.capability_mask) && isempty(relation.generation_providers)
        return _sample_generation_derivation(
            state,
            relation,
            target_ty,
            scope_tys,
            depth_left,
            guarded_left,
            allow_safe_list_access,
        )
    end
    query = GenerationQuery(
        target_ty;
        scope_tys,
        depth_left,
        guarded_left,
        provider_fuel=depth_left,
        node_budget=state.nodes_left,
        capability_mask=relation.capability_mask,
    )
    root_requirements = _root_generation_requirements(relation, query)
    isempty(root_requirements) ||
        (query = _derive_generation_query(query; requirements=root_requirements))
    return _sample_generation_derivation(
        state,
        relation,
        query,
        scope_tys,
        allow_safe_list_access,
    )
end

function _root_generation_requirements(relation::CoreRelation, query::GenerationQuery)
    requirements = ()
    providers = _admitted_providers(relation.generation_providers, query.capability_mask)
    for provider in providers
        supplied = generation_root_requirements(provider, relation, query)
        requirements = _merge_requirements(requirements, supplied)
    end
    return requirements
end

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
    return _sample_from_options(
        state,
        relation,
        options,
        scope_tys,
        depth_left,
        guarded_left,
        allow_safe_list_access,
    )
end

function _sample_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    query::GenerationQuery,
    allow_safe_list_access::Bool=true,
)
    return _sample_generation_derivation(
        state,
        relation,
        query,
        Ty[query.scope_tys...],
        allow_safe_list_access,
    )
end

function _sample_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    query::GenerationQuery,
    scope_tys::Vector{Ty},
    allow_safe_list_access::Bool,
)
    query_relation, effective = _relation_for_query(relation, query)
    if _generation_extensions_exhausted(query_relation, effective)
        return _sample_generation_derivation(
            state,
            query_relation,
            effective.target_ty,
            scope_tys,
            effective.depth_left,
            effective.guarded_left,
            allow_safe_list_access,
        )
    end
    core_query = _in_closed_fragment(query.target_ty) && all(_in_closed_fragment, query.scope_tys)
    extension_query = _supports_recursive_type_queries(relation.generation_providers) &&
        _in_generation_extension_fragment(query.target_ty) &&
        all(_in_generation_extension_fragment, query.scope_tys)
    core_query || extension_query || return nothing
    node_budget = min(query.node_budget, state.nodes_left)
    bounded = node_budget == query.node_budget ?
        query : GenerationQuery(
            query.target_ty,
            query.scope_tys,
            query.depth_left,
            query.guarded_left,
            query.provider_fuel,
            node_budget,
            query.capability_mask,
            query.requirements,
        )
    query_relation, effective = node_budget == query.node_budget ?
        (query_relation, effective) : _relation_for_query(query_relation, bounded)
    target_ty = effective.target_ty
    depth_left = effective.depth_left
    guarded_left = effective.guarded_left
    options = copy(_sampling_options(
        query_relation,
        target_ty,
        scope_tys,
        depth_left;
        allow_head_application=true,
        lambda_kind=:lambda_generation,
    ))
    if depth_left > 0 && guarded_left > 0 && _bucket_enabled(query_relation.weights, _GUARDED_EXPANSION_WEIGHT)
        append!(options, _guarded_sampling_options(query_relation, target_ty, scope_tys))
    end
    if depth_left > 0 && allow_safe_list_access
        append!(options, _safe_list_access_generation_options(query_relation, target_ty, scope_tys))
    end
    append!(options, _generation_extension_options(query_relation, effective))
    return _sample_from_options(
        state,
        query_relation,
        options,
        scope_tys,
        depth_left,
        guarded_left,
        allow_safe_list_access,
        effective,
    )
end

function _sample_nested_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    ::Nothing,
)
    return _sample_generation_derivation(
        state,
        relation,
        target_ty,
        scope_tys,
        depth_left,
        guarded_left,
        allow_safe_list_access,
    )
end

_nested_generation_query(
    state::DerivationSamplingState,
    ::Nothing,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
) = nothing

function _nested_generation_query(
    state::DerivationSamplingState,
    parent::GenerationQuery,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
)
    added = length(scope_tys) - length(parent.scope_tys)
    added >= 0 || throw(ArgumentError("nested generation cannot remove inherited scope binders"))
    for i in eachindex(parent.scope_tys)
        child_ty = scope_tys[added + i]
        parent_ty = parent.scope_tys[i]
        (child_ty === parent_ty || _cache_type_key(child_ty) == _cache_type_key(parent_ty)) ||
            throw(ArgumentError("nested generation must preserve inherited scope as an outer suffix"))
    end
    added_scope = ntuple(i -> scope_tys[i], added)
    requirements = added == 0 ?
        parent.requirements : _rebase_inherited_requirements(parent, added)
    return GenerationQuery(
        target_ty,
        added == 0 ? parent.scope_tys : (added_scope..., parent.scope_tys...),
        depth_left,
        guarded_left,
        parent.provider_fuel,
        min(parent.node_budget, state.nodes_left),
        parent.capability_mask,
        requirements,
    )
end

function _sample_nested_generation_derivation(
    state::DerivationSamplingState,
    relation::CoreRelation,
    target_ty::Ty,
    scope_tys::Vector{Ty},
    depth_left::Int,
    guarded_left::Int,
    allow_safe_list_access::Bool,
    parent::GenerationQuery,
)
    query = _nested_generation_query(
        state,
        parent,
        target_ty,
        scope_tys,
        depth_left,
        guarded_left,
    )
    return _sample_generation_derivation(
        state,
        relation,
        query,
        scope_tys,
        allow_safe_list_access,
    )
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
        _known_head_admitted(relation, head) || continue
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
            for rule in _scope_application_rules(target_ty, scope_tys, depth_left)
                _known_type_admitted(relation, rule.head_ty) || continue
                push!(options, (kind=:scope_application, value=rule, weight=_scope_application_option_weight(relation, rule)))
            end
        end
        for rule in catalog.rules
            iscompatible(rule.result_ty, target_ty) || continue
            _known_head_admitted(relation, rule.head) || continue
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
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
            generation_query,
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
        option.weight isa Real && isfinite(option.weight) && option.weight >= 0 ||
            throw(ArgumentError("generation option weight must be finite and nonnegative"))
        option.weight > 0 || continue
        push!(pool, option)
        total_weight += option.weight
        isfinite(total_weight) || throw(ArgumentError("generation option weight sum overflow"))
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
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
            _sample_nested_generation_derivation(
                state,
                relation,
                target_ty.res,
                body_scope,
                depth_left - 1,
                guarded_left,
                allow_safe_list_access,
                generation_query,
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
            generation_query,
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
            generation_query,
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
            generation_query,
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
            generation_query,
        )
    elseif option.kind == :generation_extension
        generation_query === nothing && return nothing
        generation_query.depth_left > 0 && generation_query.provider_fuel > 0 || return nothing
        budget_before = min(state.nodes_left, generation_query.node_budget)
        state.nodes_left = max(budget_before, 0)
        nested_capability_mask = allow_safe_list_access ?
            generation_query.capability_mask :
            _merge_capabilities(generation_query.capability_mask, (:safe_list_access,))
        nested_query = GenerationQuery(
            generation_query.target_ty,
            generation_query.scope_tys,
            generation_query.depth_left - 1,
            generation_query.guarded_left,
            generation_query.provider_fuel - 1,
            budget_before,
            nested_capability_mask,
            generation_query.requirements,
        )
        nested_relation, effective = _relation_for_query(relation, nested_query)
        any(
            provider -> provider === option.value.provider,
            nested_relation.providers,
        ) || return nothing
        derivation = sample_generation_extension(
            option.value.provider,
            state,
            nested_relation,
            option.value.option,
            effective,
        )
        state.nodes_left = clamp(state.nodes_left, 0, budget_before)
        derivation === nothing && return nothing
        derivation isa CoreDerivation ||
            throw(ArgumentError("generation extensions must return CoreDerivation or nothing"))
        charged = budget_before - state.nodes_left
        _derivation_node_count(derivation) <= charged || return nothing
        result_ty = derivation_result_type(derivation)
        (result_ty === effective.target_ty || iscompatible(result_ty, effective.target_ty)) ||
            return nothing
        rigid_query = _contains_rigid_dummy(effective.target_ty) ||
            any(_contains_rigid_dummy, effective.scope_tys)
        if rigid_query
            safe = rigid_safe_derivation(derivation)
            safe isa Bool || throw(ArgumentError("rigid_safe_derivation must return Bool"))
            safe || return nothing
        end
        return derivation
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    depth_left >= 0 || return nothing
    _consume_sampling_nodes!(state, length(rule.argument_tys)) || return nothing
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
            _sample_nested_generation_derivation(
                state,
                relation,
                arg_ty,
                scope_tys,
                depth_left,
                guarded_left,
                allow_safe_list_access,
                generation_query,
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
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
            generation_query,
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
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
            generation_query,
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
            _plan_argument_derivations(relation, rule, raw_derivations, arg_ty, arg_class)
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
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
                generation_query,
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
    generation_query::Union{Nothing,GenerationQuery}=nothing,
)
    parts = _afoldl_step_parts(rule)
    parts === nothing && return nothing
    depth_left >= 2 || return nothing
    saved_nodes = state.nodes_left
    _consume_sampling_nodes!(state, 2) || return nothing
    body = _sample_nested_generation_derivation(
        state,
        relation,
        parts.body_ty,
        _afoldl_step_scope(rule, scope_tys),
        depth_left - 2,
        guarded_left,
        allow_safe_list_access,
        generation_query,
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

"""Consume `count` units from the shared finite generation node ledger."""
consume_generation_nodes!(state::DerivationSamplingState, count::Integer) =
    _consume_sampling_nodes!(state, _checked_nonnegative_int(:count, count))

derivation_node_count(::VariableDerivation) = 1
derivation_node_count(::ConstantDerivation) = 1
derivation_node_count(d::LambdaDerivation) = 1 + derivation_node_count(d.body)
derivation_node_count(d::HeadApplicationDerivation) = 1 + sum(derivation_node_count, d.args; init=0)
derivation_node_count(d::ScopeApplicationDerivation) =
    length(d.args) + sum(derivation_node_count, d.args; init=0)
derivation_node_count(d::SafeListAccessDerivation) =
    3 + derivation_node_count(d.source) + derivation_node_count(d.body) + derivation_node_count(d.fallback)

_derivation_node_count(d::CoreDerivation) = derivation_node_count(d)

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
    signature = _StructuralKey(_derivation_signature(derivation))
    signatures = get!(relation.signatures, key) do
        Set{Any}()
    end
    signature in signatures && return false
    push!(signatures, signature)
    push!(relation.derivations[key], derivation)
    return true
end

derivation_signature(d::VariableDerivation) = (:var, d.index, typekey(d.ty))
derivation_signature(d::ConstantDerivation) = (:const, d.head.source, d.head.name, typekey(d.head.type))
derivation_signature(d::LambdaDerivation) = (:lambda, typekey(d.arg_ty), derivation_signature(d.body))

function _derivation_signature_tuple(derivations)
    signatures = Vector{Any}(undef, length(derivations))
    for index in eachindex(derivations)
        signatures[index] = derivation_signature(derivations[index])
    end
    return Tuple(signatures)
end

derivation_signature(d::HeadApplicationDerivation) = (
    :head,
    d.head.source,
    d.head.name,
    typekey(d.head.type),
    _derivation_signature_tuple(d.args),
)
derivation_signature(d::ScopeApplicationDerivation) = (
    :scope_application,
    d.index,
    typekey(d.head_ty),
    _derivation_signature_tuple(d.args),
)
derivation_signature(d::SafeListAccessDerivation) = (
    :safe_list_access,
    :UNCONS_CASE,
    typekey(d.source_ty),
    typekey(d.result_ty),
    derivation_signature(d.source),
    derivation_signature(d.body),
    derivation_signature(d.fallback),
)

_derivation_signature(d::CoreDerivation) = derivation_signature(d)

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
        return AbsNode(
            :x,
            materialize_generation_derivation(
                DefaultAstMaterializer(),
                derivation.body,
                local_consts,
                local_consts_ty,
                relation,
                rng,
            ),
        )
    elseif derivation isa HeadApplicationDerivation
        node = _head_node!(derivation.head, local_consts, local_consts_ty, relation, rng)
        for arg in derivation.args
            arg_node = materialize_generation_derivation(
                DefaultAstMaterializer(),
                arg,
                local_consts,
                local_consts_ty,
                relation,
                rng,
            )
            node = AppNode(node, arg_node)
        end
        return node
    elseif derivation isa ScopeApplicationDerivation
        node = IndexNode(derivation.index)
        for arg in derivation.args
            arg_node = materialize_generation_derivation(
                DefaultAstMaterializer(),
                arg,
                local_consts,
                local_consts_ty,
                relation,
                rng,
            )
            node = AppNode(node, arg_node)
        end
        return node
    elseif derivation isa SafeListAccessDerivation
        source_node = materialize_generation_derivation(
            DefaultAstMaterializer(),
            derivation.source,
            local_consts,
            local_consts_ty,
            relation,
            rng,
        )
        body_node = materialize_generation_derivation(
            DefaultAstMaterializer(),
            derivation.body,
            local_consts,
            local_consts_ty,
            relation,
            rng,
        )
        fallback_node = materialize_generation_derivation(
            DefaultAstMaterializer(),
            derivation.fallback,
            local_consts,
            local_consts_ty,
            relation,
            rng,
        )
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
    rng::Random.AbstractRNG,
)
    return materialize_generation_derivation(
        materializer,
        derivation,
        local_consts,
        local_consts_ty,
        relation,
        rng,
    )
end

function _materialize_derivation(
    ::DefaultAstMaterializer,
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
    rng::Random.AbstractRNG,
)
    return materialize_generation_derivation(
        DefaultAstMaterializer(),
        derivation,
        local_consts,
        local_consts_ty,
        relation,
        rng,
    )
end

"""
    materialize_generation_derivation(materializer, derivation,
                                      local_consts, local_const_types,
                                      relation, rng)

Materialize one finite derivation. Extension derivations overload this protocol
for the materializers they support; the generic path preserves the historical
four-argument callable materializer interface.
"""
function materialize_generation_derivation(
    materializer,
    derivation::CoreDerivation,
    local_consts,
    local_consts_ty,
    relation,
    rng,
)
    return materializer(derivation, local_consts, local_consts_ty, relation)
end

function materialize_generation_derivation(
    ::DefaultAstMaterializer,
    derivation::CoreDerivation,
    local_consts,
    local_consts_ty,
    relation,
    rng,
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

@testitem "CanonicalGeneration scoped capability masks are transitive" begin
    using Test
    using Random
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun

    CG = LambdaRegression.CanonicalGeneration
    presets = Dict{Symbol,Any}(:opaque_float => 1.0)
    preset_types = Dict{Symbol,Any}(:opaque_float => FloatTy)
    ordinary = CG.generate_canonical_scoped(
        FloatTy;
        presets,
        preset_types,
        rng=Random.MersenneTwister(1),
        weights=CG.GenerationWeights(default=1),
        sampling_depth_limit=1,
        sampling_node_budget=4,
    )
    masked = CG.generate_canonical_scoped(
        FloatTy;
        presets,
        preset_types,
        rng=Random.MersenneTwister(1),
        weights=CG.GenerationWeights(default=1),
        sampling_depth_limit=1,
        sampling_node_budget=4,
        capability_mask=(:opaque,),
    )

    @test ordinary isa CG.GenerationResult
    @test masked isa Err

    opaque_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:seed => true),
        global_consts_ty=Dict{Symbol,Any}(:seed => BoolTy),
        presets=Dict{Symbol,Any}(:opaque_apply => identity),
        preset_types=Dict{Symbol,Any}(:opaque_apply => TyFun(BoolTy, FloatTy)),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=1),
        guarded_expansion_depth=0,
        capability_mask=(:opaque,),
    )
    @test isempty(CG._known_derivations!(opaque_relation, FloatTy, Ty[]))

    fixed_point_ty = TyFun(TyFun(FloatTy, FloatTy), FloatTy)
    fixed_point_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=1, scope_application=1),
        guarded_expansion_depth=0,
        capability_mask=(:fixed_point,),
    )
    @test isempty(CG._scope_application_candidates(
        fixed_point_relation,
        FloatTy,
        Ty[fixed_point_ty],
        2,
        false,
    ))

    opaque_guard = CG.HeadEntry(
        :OPAQUE_GUARD,
        :provider_value,
        TyFun(BoolTy, FloatTy),
        CG.Decoration(),
        CG.WeightBucket(:opaque_guard);
        metadata=(value=identity, guarded=true, capabilities=(:opaque,)),
    )
    guarded_relation = CG.CoreRelation(
        weights=CG.GenerationWeights(default=1),
        capability_mask=(:opaque,),
    )
    @test isempty(CG._guarded_rules_from_heads(
        guarded_relation,
        CG.HeadEntry[opaque_guard],
        FloatTy,
    ))

    struct OpaqueConstructiveProvider <: CG.AbstractHeadProvider end
    CG.provider_capabilities(::OpaqueConstructiveProvider) = (:opaque,)
    function CG.add_heads!(::OpaqueConstructiveProvider, heads, universe; kwargs...)
        push!(
            heads,
            CG.HeadEntry(
                :OPAQUE_MERGE,
                :provider_value,
                TyFun(FloatTy, TyFun(FloatTy, FloatTy)),
                CG.CoreProductive(),
                CG.WeightBucket(:opaque_merge);
                metadata=(value=+, capabilities=(:opaque,)),
            ),
        )
        return heads
    end
    constructive_relation = CG.CoreRelation(
        providers=(OpaqueConstructiveProvider(),),
        weights=CG.GenerationWeights(default=1),
        capability_mask=(:opaque,),
    )
    constructive_scope = Ty[FloatTy, TyFun(FloatTy, FloatTy)]
    constructive_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=constructive_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(1),),
    )
    @test isempty(CG._fast_proxy_head_routes(
        constructive_relation,
        FloatTy,
        constructive_scope,
        4,
        1,
        0,
    ))
    @test isempty(CG._fast_split_requirement_head_plans(
        constructive_relation,
        FloatTy,
        constructive_scope,
        4,
        1,
        0,
        constructive_query,
    ))

    restricted_query = CG.GenerationQuery(
        FloatTy;
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        capability_mask=(:safe_list_access,),
    )
    restricted_first, _ = CG._relation_for_query(guarded_relation, restricted_query)
    restricted_again, _ = CG._relation_for_query(guarded_relation, restricted_query)
    known_restricted, _ = CG._fresh_known_relation(guarded_relation, restricted_query)
    @test restricted_first === restricted_again
    @test known_restricted === restricted_first

    local_relation = CG.CoreRelation(
        local_consts=Dict{Symbol,Any}(:source => true),
        local_consts_ty=Dict{Symbol,Any}(:source => BoolTy),
        weights=CG.GenerationWeights(default=1),
    )
    local_restricted, _ = CG._relation_for_query(local_relation, restricted_query)
    local_relation.local_consts[:source] = false
    refreshed_restricted, _ = CG._relation_for_query(local_relation, restricted_query)
    @test refreshed_restricted === local_restricted
    @test refreshed_restricted.local_consts[:source] === false
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
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyFun, TyNat

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
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyFun, TyNat, TyScheme, make_tyvar_factory, typekey

    CG = LambdaRegression.CanonicalGeneration
    n = make_tyvar_factory()(0)
    a = make_tyvar_factory()(0)
    @test n.id == a.id
    @test n != a
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
    generation_kwargs = CG.test_catalog_generation_kwargs(profile=:deeper)
    isolated_weights = CG.GenerationWeights(
        default=generation_kwargs.weights.default,
        overrides=merge(
            generation_kwargs.weights.overrides,
            Dict(:proxy_required_head => 0.0),
        ),
    )

    result = CG.generate_canonical(
        target;
        rng=Random.MersenneTwister(1111),
        global_consts=CG.test_catalog_consts(),
        merge(generation_kwargs, (; weights=isolated_weights))...,
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

@testitem "CanonicalGeneration capability masks are transitive" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty

    CG = LambdaRegression.CanonicalGeneration

    mutable struct DeclaredNestedProvider <: CG.AbstractHeadProvider
        nested_queries::Vector{CG.GenerationQuery}
    end
    mutable struct UndeclaredProvider <: CG.AbstractHeadProvider
        calls::Base.RefValue{Int}
    end

    CG.provider_capabilities(::DeclaredNestedProvider) = ()
    CG.add_heads!(::DeclaredNestedProvider, heads, universe; kwargs...) = heads

    function CG.add_generation_options!(
        ::DeclaredNestedProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight_bucket=CG.WeightBucket(:declared_nested),))
        return options
    end

    function CG.sample_generation_extension(
        provider::DeclaredNestedProvider,
        state,
        relation,
        option,
        query,
    )
        push!(provider.nested_queries, query)
        return CG.sample_generation_extension(
            state,
            relation,
            query;
            target_ty=query.target_ty,
            scope_tys=query.scope_tys,
            depth_left=query.depth_left,
            guarded_left=query.guarded_left,
            provider_fuel=query.provider_fuel,
        )
    end

    function CG.add_heads!(provider::UndeclaredProvider, heads, universe; kwargs...)
        provider.calls[] += 1
        push!(
            heads,
            CG.HeadEntry(
                :OPAQUE_FLOAT,
                :undeclared,
                FloatTy,
                CG.CoreProductive(),
                CG.WeightBucket(:undeclared);
                metadata=(value=1.0,),
            ),
        )
        return heads
    end

    safe = DeclaredNestedProvider(CG.GenerationQuery[])
    opaque = UndeclaredProvider(Ref(0))
    relation = CG.CoreRelation(;
        providers=(safe, opaque),
        weights=CG.GenerationWeights(default=0, variable=1, declared_nested=1, undeclared=100),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=2,
        guarded_left=0,
        provider_fuel=1,
        node_budget=4,
        capability_mask=(:fixed_point,),
        requirements=(),
    )
    state = CG.DerivationSamplingState(Random.MersenneTwister(555), 4)

    restricted, effective = CG._relation_for_query(relation, query)
    option = only(CG._generation_extension_options(restricted, effective))
    derivation = CG._sample_option(
        state,
        restricted,
        option,
        Ty[effective.scope_tys...],
        effective.depth_left,
        effective.guarded_left,
        true,
        effective,
    )

    @test derivation isa CG.VariableDerivation
    @test length(safe.nested_queries) == 1
    @test only(safe.nested_queries).capability_mask == (:fixed_point,)
    @test opaque.calls[] == 0
    @test CG.provider_capabilities(opaque) == (:opaque,)
    @test state.nodes_left == 3

    immutable_query = CG.GenerationQuery(FloatTy; scope_tys=Ty[FloatTy])
    provider_key = (CG.ContextHeadProvider(),)
    cache_key = CG._generation_query_cache_key(immutable_query, provider_key)
    @test immutable_query.scope_tys === (FloatTy,)
    @test cache_key !== nothing
    @test_throws MethodError setindex!(immutable_query.scope_tys, BoolTy, 1)
    @test CG._generation_query_cache_key(immutable_query, provider_key) == cache_key
end

@testitem "CanonicalGeneration precomputes extension provider routing" begin
    using Test
    using Random

    CG = LambdaRegression.CanonicalGeneration

    struct CatalogOnlyProvider <: CG.AbstractHeadProvider end
    mutable struct ExplicitCatalogOnlyProvider <: CG.AbstractHeadProvider
        option_calls::Base.RefValue{Int}
    end
    struct QueryOptionProvider <: CG.AbstractHeadProvider end
    struct ConcreteRelationOptionProvider <: CG.AbstractHeadProvider end

    CG.add_heads!(::CatalogOnlyProvider, heads, universe; kwargs...) = heads
    CG.add_heads!(::ExplicitCatalogOnlyProvider, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::ExplicitCatalogOnlyProvider) = false
    function CG.add_generation_options!(
        provider::ExplicitCatalogOnlyProvider,
        options,
        relation,
        query,
    )
        provider.option_calls[] += 1
        push!(options, (weight=1.0, token=:must_not_be_called))
        return options
    end
    CG.add_heads!(::QueryOptionProvider, heads, universe; kwargs...) = heads
    CG.add_heads!(::ConcreteRelationOptionProvider, heads, universe; kwargs...) = heads
    function CG.add_generation_options!(
        ::QueryOptionProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight=1.0, token=:query_option))
        return options
    end
    function CG.add_generation_options!(
        ::ConcreteRelationOptionProvider,
        options,
        relation::CG.CoreRelation{CG.DefaultArgumentPlanner},
        query::CG.GenerationQuery,
    )
        push!(options, (weight=1.0, token=:concrete_relation_option))
        return options
    end

    catalog_only = CatalogOnlyProvider()
    query_option = QueryOptionProvider()
    concrete_option = ConcreteRelationOptionProvider()
    relation = CG.CoreRelation(;
        providers=(CG.ContextHeadProvider(), catalog_only, query_option, concrete_option),
    )
    ordinary = CG.CoreRelation(;
        providers=(CG.ContextHeadProvider(), catalog_only),
    )

    @test relation.generation_providers == (query_option, concrete_option)
    @test isempty(ordinary.generation_providers)
    @test CG.provides_generation_options(CG.ContextHeadProvider()) === false
    @test CG.provides_generation_options(query_option) === nothing

    state = CG.DerivationSamplingState(Random.Xoshiro(1), 8)
    @test fieldnames(typeof(state)) == (:rng, :nodes_left)

    explicit_catalog = ExplicitCatalogOnlyProvider(Ref(0))
    baseline_relation = CG.CoreRelation(providers=(CG.ContextHeadProvider(),))
    catalog_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(), explicit_catalog),
    )
    nested_query = CG.GenerationQuery(
        CG.TyFun(CG.FloatTy, CG.FloatTy);
        depth_left=3,
        provider_fuel=3,
        node_budget=16,
    )
    baseline_state = CG.DerivationSamplingState(Random.Xoshiro(555), 16)
    catalog_state = CG.DerivationSamplingState(Random.Xoshiro(555), 16)
    baseline_derivation = CG._sample_generation_derivation(
        baseline_state,
        baseline_relation,
        nested_query,
    )
    catalog_derivation = CG._sample_generation_derivation(
        catalog_state,
        catalog_relation,
        nested_query,
    )

    @test CG._derivation_signature(catalog_derivation) ==
        CG._derivation_signature(baseline_derivation)
    @test explicit_catalog.option_calls[] == 0
end

@testitem "CanonicalGeneration rebases inherited binder requirements" begin
    using Test
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyList

    CG = LambdaRegression.CanonicalGeneration

    struct ForgottenScopeRequirement <: CG.AbstractGenerationRequirement end
    struct StableScopeRequirement <: CG.AbstractGenerationRequirement end
    CG.rebase_generation_requirement(
        requirement::StableScopeRequirement,
        added_binders::Integer,
    ) = requirement

    parent = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
        requirements=(CG.UsesScopeBinder(0),),
    )
    under_lambda = CG._derive_generation_query(
        parent;
        added_scope_tys=Ty[BoolTy],
        depth_left=3,
    )
    under_two_binders = CG._derive_generation_query(
        parent;
        added_scope_tys=Ty[BoolTy, TyList(FloatTy)],
        depth_left=2,
    )

    @test only(under_lambda.requirements) == CG.UsesScopeBinder(1)
    @test only(under_two_binders.requirements) == CG.UsesScopeBinder(2)
    @test_throws ArgumentError CG._derive_generation_query(
        parent;
        scope_tys=Ty[FloatTy, BoolTy],
    )
    @test_throws ArgumentError CG._derive_generation_query(
        parent;
        scope_tys=Ty[FloatTy, FloatTy],
    )

    forgotten = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        requirements=(ForgottenScopeRequirement(),),
    )
    stable = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        requirements=(StableScopeRequirement(),),
    )
    @test_throws ArgumentError CG._derive_generation_query(
        forgotten;
        added_scope_tys=Ty[BoolTy],
    )
    @test only(CG._derive_generation_query(
        stable;
        added_scope_tys=Ty[BoolTy],
    ).requirements) isa StableScopeRequirement
end

@testitem "CanonicalGeneration extensions remain visible in safe-list bodies" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun, TyList, typekey

    CG = LambdaRegression.CanonicalGeneration

    mutable struct SafeBodyExtensionProvider <: CG.AbstractHeadProvider
        queries::Vector{CG.GenerationQuery}
    end

    CG.provider_capabilities(::SafeBodyExtensionProvider) = (:safe_body_test,)
    CG.add_heads!(::SafeBodyExtensionProvider, heads, universe; kwargs...) = heads
    function CG.add_generation_options!(
        ::SafeBodyExtensionProvider,
        options,
        relation,
        query,
    )
        typekey(query.target_ty) == typekey(BoolTy) || return options
        push!(options, (weight=1.0e100, token=:safe_body_bool))
        return options
    end
    function CG.sample_generation_extension(
        provider::SafeBodyExtensionProvider,
        state,
        relation,
        option,
        query,
    )
        push!(provider.queries, query)
        CG._consume_sampling_nodes!(state, 3) || return nothing
        head = CG.HeadEntry(
            :SAFE_BODY_EXT,
            :test,
            TyFun(FloatTy, TyFun(BoolTy, BoolTy)),
            CG.CoreProductive(),
            CG.WeightBucket(:safe_body_ext);
            metadata=(value=x -> y -> y,),
        )
        return CG.HeadApplicationDerivation(
            head,
            CG.CoreDerivation[
                CG.VariableDerivation(0, FloatTy),
                CG.VariableDerivation(2, BoolTy),
            ],
        )
    end

    provider = SafeBodyExtensionProvider(CG.GenerationQuery[])
    relation = CG.CoreRelation(;
        providers=(provider,),
        weights=CG.GenerationWeights(default=0, variable=1, scope_application=0),
    )
    list_ty = TyList(FloatTy)
    proxy_scope = Ty[FloatTy, list_ty, BoolTy]
    query = CG.GenerationQuery(
        BoolTy;
        scope_tys=proxy_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(2),),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(42), 32)

    body = CG._sample_proxy_body_derivation(
        state,
        relation,
        BoolTy,
        proxy_scope,
        4,
        0,
        query,
    )

    @test body isa CG.HeadApplicationDerivation
    @test body.head.name == :SAFE_BODY_EXT
    @test body.args[1] == CG.VariableDerivation(0, FloatTy)
    @test body.args[2] == CG.VariableDerivation(2, BoolTy)
    @test length(provider.queries) == 1
    @test only(provider.queries).requirements == (CG.UsesScopeBinder(2),)
    @test only(provider.queries).provider_fuel < query.provider_fuel
end

@testitem "CanonicalGeneration extension sampling shares query fuel" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: FloatTy, Ty

    CG = LambdaRegression.CanonicalGeneration

    mutable struct NestedExtensionProvider <: CG.AbstractHeadProvider
        observations::Vector{NamedTuple}
    end
    mutable struct RefillProvider <: CG.AbstractHeadProvider
        observations::Vector{NamedTuple}
    end
    mutable struct OversizedProvider <: CG.AbstractHeadProvider
        calls::Base.RefValue{Int}
    end

    CG.provider_capabilities(::NestedExtensionProvider) = (:nested_test,)
    CG.add_heads!(::NestedExtensionProvider, heads, universe; kwargs...) = heads
    CG.provider_capabilities(::RefillProvider) = (:refill_test,)
    CG.add_heads!(::RefillProvider, heads, universe; kwargs...) = heads
    CG.provider_capabilities(::OversizedProvider) = (:oversized_test,)
    CG.add_heads!(::OversizedProvider, heads, universe; kwargs...) = heads

    function CG.add_generation_options!(
        ::NestedExtensionProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight=1.0, level=query.provider_fuel))
        return options
    end

    function CG.sample_generation_extension(
        provider::NestedExtensionProvider,
        state,
        relation,
        option,
        query,
    )
        push!(
            provider.observations,
            (
                state=state,
                nodes_left=state.nodes_left,
                depth_left=query.depth_left,
                guarded_left=query.guarded_left,
                provider_fuel=query.provider_fuel,
                node_budget=query.node_budget,
                relation=relation,
                query=query,
            ),
        )
        CG._consume_sampling_nodes!(state, 1) || return nothing
        query.provider_fuel == 0 && return CG.VariableDerivation(0, query.target_ty)
        return CG.sample_generation_extension(
            state,
            relation,
            query;
            target_ty=query.target_ty,
            scope_tys=query.scope_tys,
            depth_left=query.depth_left,
            guarded_left=query.guarded_left,
            provider_fuel=query.provider_fuel,
        )
    end

    function CG.sample_generation_extension(
        provider::RefillProvider,
        state,
        relation,
        option,
        query,
    )
        push!(provider.observations, (node_budget=query.node_budget, nodes_left=state.nodes_left))
        state.nodes_left += 10_000
        return CG.sample_generation_extension(state, relation, query)
    end

    function CG.sample_generation_extension(
        provider::OversizedProvider,
        state,
        relation,
        option,
        query,
    )
        provider.calls[] += 1
        CG._consume_sampling_nodes!(state, 1) || return nothing
        return CG.LambdaDerivation(FloatTy, CG.VariableDerivation(0, FloatTy))
    end

    provider = NestedExtensionProvider(NamedTuple[])
    relation = CG.CoreRelation(;
        providers=(provider,),
        weights=CG.GenerationWeights(default=0, nested_test=1),
    )
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=3,
        guarded_left=0,
        provider_fuel=2,
        node_budget=4,
    )
    state = CG.DerivationSamplingState(Random.MersenneTwister(2000), 4)

    derivation = CG.sample_generation_extension(state, relation, query)

    @test derivation isa CG.VariableDerivation
    @test length(provider.observations) == 2
    outer, nested = provider.observations
    @test outer.state === state
    @test nested.state === state
    @test nested.nodes_left < outer.nodes_left
    @test nested.depth_left < outer.depth_left
    @test nested.guarded_left <= outer.guarded_left
    @test nested.provider_fuel < outer.provider_fuel
    @test nested.node_budget < outer.node_budget
    @test outer.relation === relation
    @test nested.relation === relation
    @test state.nodes_left == 2
    @test_throws ArgumentError CG.sample_generation_extension(
        state,
        relation,
        outer.query;
        depth_left=outer.depth_left + 1,
    )
    @test_throws ArgumentError CG.sample_generation_extension(
        state,
        relation,
        outer.query;
        provider_fuel=outer.provider_fuel + 1,
    )

    refill = RefillProvider(NamedTuple[])
    refill_relation = CG.CoreRelation(;
        providers=(refill,),
        weights=CG.GenerationWeights(default=0, variable=1),
    )
    @test refill in refill_relation.providers
    @test refill ∉ refill_relation.generation_providers
    refill_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=2,
        provider_fuel=1,
        node_budget=4,
    )
    refill_option = (
        kind=:generation_extension,
        value=(provider=refill, option=(weight=1.0,)),
        weight=1.0,
    )
    refill_state = CG.DerivationSamplingState(Random.MersenneTwister(1), 4)
    refill_result = CG._sample_option(
        refill_state,
        refill_relation,
        refill_option,
        Ty[refill_query.scope_tys...],
        refill_query.depth_left,
        refill_query.guarded_left,
        true,
        refill_query,
    )
    @test refill_result isa CG.VariableDerivation
    @test only(refill.observations).node_budget == 4
    @test refill_state.nodes_left == 3

    oversized = OversizedProvider(Ref(0))
    oversized_relation = CG.CoreRelation(; providers=(oversized,))
    oversized_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=1,
        provider_fuel=1,
        node_budget=3,
    )
    oversized_option = (
        kind=:generation_extension,
        value=(provider=oversized, option=(weight=1.0,)),
        weight=1.0,
    )
    oversized_rng = Random.MersenneTwister(1111)
    control_rng = copy(oversized_rng)
    rand(control_rng)
    expected_next = rand(control_rng)
    oversized_state = CG.DerivationSamplingState(oversized_rng, 3)
    oversized_result = CG._sample_from_options(
        oversized_state,
        oversized_relation,
        NamedTuple[oversized_option],
        Ty[oversized_query.scope_tys...],
        oversized_query.depth_left,
        oversized_query.guarded_left,
        true,
        oversized_query,
    )
    @test oversized_result === nothing
    @test oversized_state.nodes_left == 3
    @test oversized.calls[] == 1
    @test rand(oversized_state.rng) == expected_next
end

@testitem "CanonicalGeneration recursive scope applications are depth bounded" begin
    using Test
    using Random
    using LambdaRegression.TypeSystem: IntTy, Ty, TyFun, TyMu, iscompatible, newtyvar

    CG = LambdaRegression.CanonicalGeneration

    binder = newtyvar(0)
    recursive_fun = TyMu(binder, TyFun(IntTy, binder))
    malformed_binder = newtyvar(0)
    malformed = TyMu(malformed_binder, malformed_binder)

    @test !CG._in_closed_fragment(recursive_fun)
    @test CG.generation_function_view(recursive_fun) isa TyFun
    @test CG.generation_function_view(malformed) === nothing
    @test isempty(CG._scope_application_rules(recursive_fun, Ty[recursive_fun], 0))

    rules = CG._scope_application_rules(recursive_fun, Ty[recursive_fun], 3)
    @test length(rules) == 3
    @test length.(getfield.(rules, :argument_tys)) == [1, 2, 3]
    @test all(rule -> all(==(IntTy), rule.argument_tys), rules)
    @test all(rule -> iscompatible(rule.result_ty, recursive_fun), rules)

    wide_binder = newtyvar(0)
    wide_recursive_fun = TyMu(wide_binder, TyFun(IntTy, TyFun(IntTy, wide_binder)))
    wide_rules = CG._scope_application_rules(wide_recursive_fun, Ty[wide_recursive_fun], 4)
    @test length.(getfield.(wide_rules, :argument_tys)) == [1, 2, 3, 4]

    mutable struct UnrelatedExtensionProvider <: CG.AbstractHeadProvider
        option_calls::Base.RefValue{Int}
    end
    mutable struct RecursiveQueryProvider <: CG.AbstractHeadProvider
        option_calls::Base.RefValue{Int}
    end
    CG.add_heads!(::UnrelatedExtensionProvider, heads, universe; kwargs...) = heads
    CG.add_heads!(::RecursiveQueryProvider, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::UnrelatedExtensionProvider) = true
    CG.provides_generation_options(::RecursiveQueryProvider) = true
    CG.provider_capabilities(::UnrelatedExtensionProvider) = (:unrelated_test,)
    CG.provider_capabilities(::RecursiveQueryProvider) = (:recursive_type_query,)
    function CG.add_generation_options!(provider::UnrelatedExtensionProvider, options, relation, query)
        provider.option_calls[] += 1
        return options
    end
    function CG.add_generation_options!(provider::RecursiveQueryProvider, options, relation, query)
        provider.option_calls[] += 1
        return options
    end
    unrelated = UnrelatedExtensionProvider(Ref(0))
    rejected = CG.generate_canonical(
        recursive_fun;
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(1),
            providers=(unrelated,),
            sampling_depth_limit=3,
            sampling_node_budget=8,
        ),
    )
    @test rejected isa LambdaRegression.Utils.Err
    @test unrelated.option_calls[] == 0

    recursive_provider = RecursiveQueryProvider(Ref(0))
    mixed = CG.generate_canonical(
        recursive_fun;
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(1111),
            providers=(recursive_provider, unrelated),
            sampling_depth_limit=3,
            sampling_node_budget=8,
        ),
    )
    @test mixed isa LambdaRegression.Utils.Err
    @test recursive_provider.option_calls[] == 1
    @test unrelated.option_calls[] == 0
end

@testitem "CanonicalGeneration proxy general sampling makes one weighted proposal" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty

    CG = CanonicalGeneration
    first_choice_seed = only(Iterators.take(filter(1:100) do seed
        rand(Random.Xoshiro(seed)) <= 0.5
    end, 1))
    options = NamedTuple[
        (kind=:constant, value=nothing, weight=1.0),
        (
            kind=:variable,
            value=CG.VariableDerivation(0, FloatTy),
            weight=1.0,
        ),
    ]
    relation = CG.CoreRelation(providers=())
    state = CG.DerivationSamplingState(Random.Xoshiro(first_choice_seed), 8)

    proposal = CG._sample_from_proxy_using_options(
        state,
        relation,
        options,
        Ty[FloatTy],
        0,
        0,
        1,
        0,
    )

    @test proposal === nothing
end

@testitem "CanonicalGeneration constructive sampling composes required scope binders" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, Ty, TyFun

    CG = CanonicalGeneration
    unary = TyFun(FloatTy, FloatTy)
    scope = Ty[FloatTy, unary, unary]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(1), CG.UsesScopeBinder(2)),
    )
    relation = CG.CoreRelation(
        providers=(),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            scope_application=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 32)

    derivation = CG._sample_fast_proxy_insertion_derivation(
        state,
        relation,
        FloatTy,
        scope,
        4,
        0,
        1,
        1,
        false,
        query,
    )

    @test derivation !== nothing
    if derivation !== nothing
        @test CG.derivation_uses_scope_binder(derivation, 1)
        @test CG.derivation_uses_scope_binder(derivation, 2)
    end

    split_scope = Ty[
        FloatTy,
        FloatTy,
        TyFun(FloatTy, BoolTy),
        TyFun(FloatTy, IntTy),
        TyFun(BoolTy, TyFun(IntTy, FloatTy)),
    ]
    split_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=split_scope,
        depth_left=6,
        provider_fuel=6,
        node_budget=48,
        requirements=(
            CG.UsesScopeBinder(2),
            CG.UsesScopeBinder(3),
            CG.UsesScopeBinder(4),
        ),
    )
    split_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 48)
    split_derivation = CG._sample_fast_proxy_insertion_derivation(
        split_state,
        relation,
        FloatTy,
        split_scope,
        6,
        0,
        1,
        4,
        false,
        split_query,
    )

    @test split_derivation !== nothing
    if split_derivation !== nothing
        @test all(
            index -> CG.derivation_uses_scope_binder(split_derivation, index),
            (2, 3, 4),
        )
    end

    pred = x -> x > 0
    bool_unary = TyFun(BoolTy, BoolTy)
    productive_scope = Ty[FloatTy, bool_unary]
    productive_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=productive_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(0), CG.UsesScopeBinder(1)),
    )
    productive_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:PRED => pred),
        global_consts_ty=Dict{Symbol,Any}(:PRED => TyFun(FloatTy, BoolTy)),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            scope_application=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    productive_state = CG.DerivationSamplingState(Random.Xoshiro(42), 32)
    productive_derivation = CG._sample_fast_proxy_insertion_derivation(
        productive_state,
        productive_relation,
        BoolTy,
        productive_scope,
        4,
        0,
        1,
        0,
        false,
        productive_query,
    )

    @test productive_derivation !== nothing
    if productive_derivation !== nothing
        @test CG.derivation_uses_scope_binder(productive_derivation, 0)
        @test CG.derivation_uses_scope_binder(productive_derivation, 1)
    end
end

@testitem "CanonicalGeneration constructive proxy routes preserve the proxy-use obligation" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyList

    CG = CanonicalGeneration
    list_ty = TyList(FloatTy)
    scope = Ty[list_ty, FloatTy, FloatTy]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=1,
        provider_fuel=1,
        node_budget=8,
        requirements=(CG.UsesScopeBinder(2),),
    )
    relation = CG.CoreRelation(
        providers=(),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            proxy_constructive=1,
            proxy_general=0,
        ),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(555), 8)

    derivation = CG._sample_proxy_using_derivation(
        state,
        relation,
        FloatTy,
        scope,
        1,
        0,
        2,
        0;
        generation_query=query,
    )

    @test derivation === nothing
end

@testitem "CanonicalGeneration source-array depth remains a hard planner boundary" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: FloatTy, Ty, TyArray, TyFun, TyNat

    CG = CanonicalGeneration
    array_ty = TyArray(TyNat(2), FloatTy)
    source_constructor = CG.HeadEntry(
        :ARRAY_STEP,
        :test,
        TyFun(array_ty, array_ty),
        CG.CoreProductive(),
        CG.WeightBucket(:test);
        metadata=(value=identity, source_array_constructor=true, source_array_argument_index=1),
    )
    parent = CG.HeadEntry(
        :CONSUME_ARRAY,
        :test,
        TyFun(array_ty, FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:test);
        metadata=(value=identity, source_array_depth_limit=0),
    )
    rule = CG.CoreRule(parent, FloatTy, [array_ty], [:source_array])
    nested_source = CG.HeadApplicationDerivation(
        source_constructor,
        CG.CoreDerivation[CG.VariableDerivation(0, array_ty)],
    )
    relation = CG.CoreRelation(argument_planner=(derivations, args...) -> derivations)

    @test CG._array_source_depth(nested_source) == 1
    @test !CG._fast_forced_argument_admitted(relation, rule, 1, nested_source, Ty[array_ty])

    depth_one_parent = CG.HeadEntry(
        :CONSUME_DEPTH_ONE,
        :test,
        TyFun(array_ty, FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:test);
        metadata=(value=identity, source_array_depth_limit=1),
    )
    depth_one_rule = CG.CoreRule(depth_one_parent, FloatTy, [array_ty], [:source_array])
    @test CG._required_source_argument_admitted(depth_one_rule, 1, nested_source)

    reject_guarded = (derivations, args...) -> empty(derivations)
    guarded_parent = CG.HeadEntry(
        :GUARDED_ARRAY,
        :test,
        TyFun(array_ty, FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:test);
        metadata=(
            value=identity,
            guarded=true,
            source_array_depth_limit=1,
            argument_planner=reject_guarded,
        ),
    )
    guarded_rule = CG.CoreRule(guarded_parent, FloatTy, [array_ty], [:source_array])
    @test isempty(CG._guarded_argument_derivations(
        relation,
        CG.CoreDerivation[CG.VariableDerivation(0, array_ty)],
        array_ty,
        :source_array,
        guarded_rule,
    ))

    scope_rule = CG.ScopeApplicationRule(
        0,
        TyFun(array_ty, FloatTy),
        FloatTy,
        Ty[array_ty],
        [:source_array],
    )
    @test !CG._required_source_argument_admitted(scope_rule, 1, nested_source)
    anchored = CG._required_scope_application_base_witness(
        relation,
        scope_rule,
        1,
        Ty[array_ty],
        4,
        nothing,
    )
    @test anchored !== nothing
    anchored === nothing || @test CG._array_source_depth(anchored) == 0

    buildable_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:MAKE_ARRAY => (_ -> [0.0, 0.0]), :SEED => 0.0),
        global_consts_ty=Dict{Symbol,Any}(
            :MAKE_ARRAY => TyFun(FloatTy, array_ty),
            :SEED => FloatTy,
        ),
        weights=CG.GenerationWeights(default=1),
    )
    buildable_scope = Ty[TyFun(array_ty, FloatTy)]
    buildable_query = CG.GenerationQuery(
        array_ty;
        scope_tys=buildable_scope,
        depth_left=3,
        provider_fuel=3,
        node_budget=8,
    )
    @test CG.known_generation_witness(buildable_relation, buildable_query) !== nothing
    @test CG._required_scope_application_base_witness(
        buildable_relation,
        scope_rule,
        1,
        buildable_scope,
        3,
        buildable_query,
    ) === nothing
    @test CG._sample_required_scope_free_argument(
        CG.DerivationSamplingState(Random.Xoshiro(1234), 8),
        buildable_relation,
        scope_rule,
        1,
        buildable_scope,
        3,
        0,
        false,
        buildable_query,
        buildable_query,
    ) === nothing

    obligation_scope = Ty[TyFun(array_ty, FloatTy), FloatTy]
    obligation_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=obligation_scope,
        depth_left=3,
        provider_fuel=3,
        node_budget=16,
        requirements=(CG.UsesScopeBinder(1),),
    )
    obligation_plan = (
        rule=scope_rule,
        forced=[-1],
        exact_choices=((index=1, arguments=[1]),),
        prefix_choices=Int[],
        required_arguments=(1,),
    )
    @test CG._materialize_fast_required_scope_application(
        CG.DerivationSamplingState(Random.Xoshiro(8000), 16),
        buildable_relation,
        obligation_plan,
        obligation_scope,
        3,
        0,
        0,
        0,
        false,
        obligation_query,
    ) === nothing
end

@testitem "CanonicalGeneration productive obligation plans have materializable siblings" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, Ty, TyFun

    CG = CanonicalGeneration
    pack = flag -> value -> flag ? Float64(value) : -Float64(value)
    to_int = value -> round(Int, value)
    scope = Ty[BoolTy, FloatTy]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(0),),
    )
    weights = CG.GenerationWeights(
        default=0,
        variable=1,
        constant=1,
        proxy_constructive=1,
        proxy_general=0,
        proxy_head=1,
    )
    relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:PACK => pack, :TO_INT => to_int),
        global_consts_ty=Dict{Symbol,Any}(
            :PACK => TyFun(BoolTy, TyFun(IntTy, FloatTy)),
            :TO_INT => TyFun(FloatTy, IntTy),
        ),
        providers=(CG.ContextHeadProvider(),),
        weights=weights,
    )

    plans = CG._fast_required_productive_head_plans(relation, FloatTy, scope, 4, query)
    @test !isempty(plans)
    @test isempty(relation.known_witnesses)

    state = CG.DerivationSamplingState(Random.Xoshiro(2000), 32)
    derivation = CG._sample_fast_proxy_insertion_derivation(
        state,
        relation,
        FloatTy,
        scope,
        4,
        0,
        1,
        0,
        false,
        query,
    )
    @test derivation !== nothing
    if derivation !== nothing
        @test CG.derivation_uses_scope_binder(derivation, 0)
        @test CG.derivation_uses_scope_binder(derivation, 1)
    end

    impossible_scope = Ty[BoolTy]
    impossible_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=impossible_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=32,
        requirements=(CG.UsesScopeBinder(0),),
    )
    impossible_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:PACK => pack),
        global_consts_ty=Dict{Symbol,Any}(
            :PACK => TyFun(BoolTy, TyFun(IntTy, FloatTy)),
        ),
        providers=(CG.ContextHeadProvider(),),
        weights=weights,
    )
    impossible_plans = CG._fast_required_productive_head_plans(
        impossible_relation,
        FloatTy,
        impossible_scope,
        4,
        impossible_query,
    )
    impossible_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 32)
    impossible_derivation = CG._sample_fast_required_productive_head_derivation(
        impossible_state,
        impossible_relation,
        impossible_plans,
        impossible_scope,
        4,
        0,
        false,
        impossible_query,
    )
    @test impossible_derivation === nothing

    simple = identity
    combine = value -> flag -> round(Int, value) + (flag ? 1 : 0)
    finish = isodd
    fallback_scope = Ty[BoolTy, FloatTy]
    fallback_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=fallback_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=48,
        requirements=(CG.UsesScopeBinder(0), CG.UsesScopeBinder(1)),
    )
    fallback_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(
            :SIMPLE => simple,
            :COMBINE => combine,
            :FINISH => finish,
        ),
        global_consts_ty=Dict{Symbol,Any}(
            :SIMPLE => TyFun(BoolTy, BoolTy),
            :COMBINE => TyFun(FloatTy, TyFun(BoolTy, IntTy)),
            :FINISH => TyFun(IntTy, BoolTy),
        ),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=0,
            proxy_scope_application=0,
            proxy_split=0,
            proxy_head=1,
        ),
    )
    fallback_state = CG.DerivationSamplingState(Random.Xoshiro(42), 48)
    fallback_derivation = CG._sample_fast_proxy_insertion_derivation(
        fallback_state,
        fallback_relation,
        BoolTy,
        fallback_scope,
        5,
        0,
        1,
        0,
        false,
        fallback_query,
    )
    @test fallback_derivation !== nothing
    if fallback_derivation !== nothing
        @test CG.derivation_uses_scope_binder(fallback_derivation, 0)
        @test CG.derivation_uses_scope_binder(fallback_derivation, 1)
    end
end

@testitem "CanonicalGeneration required-binder certificates stay one-step and constructive" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun, TyList

    CG = CanonicalGeneration
    unary = identity
    binary = x -> y -> x + y
    values = Dict{Symbol,Any}()
    types = Dict{Symbol,Any}()
    for i in 1:12
        unary_name = Symbol(:UNARY_, i)
        binary_name = Symbol(:BINARY_, i)
        values[unary_name] = unary
        values[binary_name] = binary
        types[unary_name] = TyFun(FloatTy, FloatTy)
        types[binary_name] = TyFun(FloatTy, TyFun(FloatTy, FloatTy))
    end

    scope = Ty[TyFun(FloatTy, FloatTy), FloatTy]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=7,
        provider_fuel=7,
        node_budget=64,
        requirements=(CG.UsesScopeBinder(0),),
    )
    relation = CG.CoreRelation(
        global_consts=values,
        global_consts_ty=types,
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            scope_application=1,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=0,
            proxy_scope_application=1,
            proxy_head=1,
            proxy_required_head=1,
            proxy_split=0,
        ),
    )

    options = CG._required_scope_certificate_options(
        relation,
        FloatTy,
        scope,
        7,
        (0,),
        query,
    )
    @test !isempty(options)
    @test all(option -> haskey(option, :kind) && haskey(option, :weight), options)
    @test !any(option -> haskey(option, :derivation), options)

    state = CG.DerivationSamplingState(Random.Xoshiro(1234), 64)
    derivation = CG._sample_required_scope_certificate_derivation(
        state,
        relation,
        FloatTy,
        scope,
        7,
        0,
        false,
        query,
    )
    @test derivation !== nothing
    derivation === nothing || @test CG.derivation_uses_scope_binder(derivation, 0)

    certificate_keys = count(keys(relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :required_scope_certificate
    end
    @test certificate_keys <= 2 * (query.depth_left + 1) * length(scope)

    call_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=7,
        provider_fuel=7,
        node_budget=64,
        requirements=(CG.CallsScopeBinder(0),),
    )
    call_options = CG._required_scope_certificate_options(
        relation,
        FloatTy,
        scope,
        7,
        (0,),
        call_query,
    )
    @test !any(option -> option.kind == :variable, call_options)
    call_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 64)
    call_derivation = CG._sample_required_scope_certificate_derivation(
        call_state,
        relation,
        FloatTy,
        scope,
        7,
        0,
        false,
        call_query,
    )
    @test call_derivation !== nothing
    call_derivation === nothing || @test CG.derivation_calls_scope_binder(call_derivation, 0)

    function_ty = TyFun(FloatTy, FloatTy)
    predicate_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:PRED => (x -> x > 0)),
        global_consts_ty=Dict{Symbol,Any}(:PRED => TyFun(FloatTy, BoolTy)),
        providers=(CG.ContextHeadProvider(),),
        weights=relation.weights,
    )
    predicate_scope = Ty[function_ty, FloatTy]
    predicate_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=predicate_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(0),),
    )
    predicate_options = CG._required_scope_certificate_options(
        predicate_relation,
        BoolTy,
        predicate_scope,
        5,
        (0,),
        predicate_query,
    )
    @test any(option -> option.kind == :productive_head, predicate_options)
    predicate_state = CG.DerivationSamplingState(Random.Xoshiro(42), 32)
    predicate_derivation = CG._sample_required_scope_certificate_derivation(
        predicate_state,
        predicate_relation,
        BoolTy,
        predicate_scope,
        5,
        0,
        false,
        predicate_query,
    )
    @test predicate_derivation !== nothing
    predicate_derivation === nothing || @test CG.derivation_calls_scope_binder(predicate_derivation, 0)

    value_only_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:ISFUNC => (_ -> true)),
        global_consts_ty=Dict{Symbol,Any}(:ISFUNC => TyFun(function_ty, BoolTy)),
        providers=(CG.ContextHeadProvider(),),
        weights=relation.weights,
    )
    value_only_scope = Ty[function_ty]
    value_only_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=value_only_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(0),),
    )
    @test isempty(CG._required_scope_certificate_options(
        value_only_relation,
        BoolTy,
        value_only_scope,
        5,
        (0,),
        value_only_query,
    ))

    cache_scope = Ty[BoolTy]
    cache_relation = CG.CoreRelation(
        providers=(CG.ContextHeadProvider(),),
        weights=relation.weights,
    )
    cache_use_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=cache_scope,
        depth_left=2,
        provider_fuel=2,
        node_budget=8,
        requirements=(CG.UsesScopeBinder(0),),
    )
    cache_call_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=cache_scope,
        depth_left=2,
        provider_fuel=2,
        node_budget=8,
        requirements=(CG.CallsScopeBinder(0),),
    )
    @test any(
        option -> option.kind == :variable,
        CG._required_scope_certificate_options(
            cache_relation,
            BoolTy,
            cache_scope,
            2,
            (0,),
            cache_use_query,
        ),
    )
    @test isempty(CG._required_scope_certificate_options(
        cache_relation,
        BoolTy,
        cache_scope,
        2,
        (0,),
        cache_call_query,
    ))

    missing_argument_scope = Ty[TyFun(FloatTy, BoolTy)]
    missing_argument_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=missing_argument_scope,
        depth_left=3,
        provider_fuel=3,
        node_budget=12,
        requirements=(CG.CallsScopeBinder(0),),
    )
    @test isempty(CG._required_scope_certificate_options(
        cache_relation,
        BoolTy,
        missing_argument_scope,
        3,
        (0,),
        missing_argument_query,
    ))

    wrapped_missing_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:PRED => (x -> x > 0)),
        global_consts_ty=Dict{Symbol,Any}(:PRED => TyFun(FloatTy, BoolTy)),
        providers=(CG.ContextHeadProvider(),),
        weights=relation.weights,
    )
    wrapped_missing_scope = Ty[TyFun(FloatTy, FloatTy)]
    wrapped_missing_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=wrapped_missing_scope,
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
        requirements=(CG.CallsScopeBinder(0),),
    )
    @test isempty(CG._required_scope_certificate_options(
        wrapped_missing_relation,
        BoolTy,
        wrapped_missing_scope,
        4,
        (0,),
        wrapped_missing_query,
    ))

    ListFloat = TyList(FloatTy)
    safe_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:ZERO => 0.0),
        global_consts_ty=Dict{Symbol,Any}(:ZERO => FloatTy),
        providers=(CG.ContextHeadProvider(), CG.SafeListAccessProvider()),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            constant=1,
            scope_application=1,
            safe_list_access=1,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=1,
            proxy_scope_application=1,
            proxy_head=1,
            proxy_required_head=1,
            proxy_split=1,
        ),
    )
    safe_scope = Ty[ListFloat, TyFun(ListFloat, FloatTy)]
    safe_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=safe_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(1),),
    )
    @test any(
        option -> option.kind == :safe_list_access,
        CG._required_scope_certificate_options(
            safe_relation,
            FloatTy,
            safe_scope,
            5,
            (1,),
            safe_query,
        ),
    )

    accept_local = (derivations, args...) -> derivations
    reject_global = (derivations, args...) -> empty(derivations)
    locally_planned_head = CG.HeadEntry(
        :LOCAL_PLAN,
        :test,
        TyFun(FloatTy, FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:local_plan);
        metadata=(value=identity, argument_planner=accept_local),
    )
    locally_planned_rule = CG.CoreRule(
        locally_planned_head,
        FloatTy,
        Ty[FloatTy],
        [:closed_recursive],
    )
    @test CG._fast_forced_argument_admitted(
        CG.CoreRelation(
            argument_planner=reject_global,
            weights=CG.GenerationWeights(default=1),
        ),
        locally_planned_rule,
        1,
        CG.VariableDerivation(0, FloatTy),
        Ty[FloatTy],
    )
    reject_local = (derivations, args...) -> empty(derivations)
    locally_rejecting_head = CG.HeadEntry(
        :LOCAL_REJECT,
        :test,
        TyFun(FloatTy, FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:local_reject);
        metadata=(value=identity, argument_planner=reject_local),
    )
    locally_rejecting_rule = CG.CoreRule(
        locally_rejecting_head,
        FloatTy,
        Ty[FloatTy],
        [:closed_recursive],
    )
    @test CG._fast_required_head_base_witness(
        CG.CoreRelation(
            argument_planner=accept_local,
            weights=CG.GenerationWeights(default=1),
        ),
        locally_rejecting_rule,
        1,
        Ty[FloatTy],
        2,
        nothing,
    ) === nothing

    direct_parent = CG.GenerationQuery(
        function_ty;
        depth_left=6,
        provider_fuel=6,
        node_budget=32,
    )
    direct_state = CG.DerivationSamplingState(Random.Xoshiro(555), 32)
    direct_overrides = copy(relation.weights.overrides)
    direct_overrides[:lambda] = 1
    direct_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:SQR => identity),
        global_consts_ty=Dict{Symbol,Any}(:SQR => function_ty),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(
            default=relation.weights.default,
            overrides=direct_overrides,
        ),
    )
    direct_query = CG.GenerationQuery(
        function_ty;
        scope_tys=Ty[function_ty],
        depth_left=6,
        provider_fuel=6,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(0),),
    )
    direct_body_scope = Ty[FloatTy, function_ty]
    direct_scope_rules = CG._scope_application_rules(FloatTy, direct_body_scope, 5)
    @test any(rule -> rule.index == 1, direct_scope_rules)
    direct_self_rule = only(rule for rule in direct_scope_rules if rule.index == 1)
    @test CG._required_scope_application_base_witness(
        direct_relation,
        direct_self_rule,
        1,
        direct_body_scope,
        4,
        nothing,
    ) !== nothing
    @test CG._required_scope_call_reachable(
        direct_relation,
        FloatTy,
        direct_body_scope,
        5,
        1,
    )
    direct_options = CG._required_scope_certificate_options(
        direct_relation,
        function_ty,
        Ty[function_ty],
        6,
        (0,),
        direct_query,
    )
    @test any(option -> option.kind == :lambda, direct_options)
    direct_certificate = CG._sample_required_scope_certificate_derivation(
        CG.DerivationSamplingState(Random.Xoshiro(555), 32),
        direct_relation,
        function_ty,
        Ty[function_ty],
        6,
        0,
        true,
        direct_query,
    )
    @test direct_certificate !== nothing
    direct_certificate === nothing || @test CG.derivation_calls_scope_binder(direct_certificate, 0)

    plan_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=direct_body_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(1),),
    )
    smaller_plan_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=direct_body_scope,
        depth_left=5,
        provider_fuel=5,
        node_budget=16,
        requirements=(CG.CallsScopeBinder(1),),
    )
    CG._fast_required_scope_application_plans(
        direct_relation,
        FloatTy,
        direct_body_scope,
        5,
        0,
        0,
        plan_query,
    )
    CG._fast_required_scope_application_plans(
        direct_relation,
        FloatTy,
        direct_body_scope,
        5,
        0,
        0,
        smaller_plan_query,
    )
    application_plan_keys = count(keys(direct_relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :fast_required_scope_application_plans
    end
    @test application_plan_keys >= 2

    guarded_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:TRUE => true),
        global_consts_ty=Dict{Symbol,Any}(:TRUE => BoolTy),
        providers=(CG.ContextHeadProvider(), CG.GuardedExpansionProvider()),
        weights=CG.GenerationWeights(default=1),
    )
    guarded_scope = Ty[function_ty, FloatTy]
    guarded_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=guarded_scope,
        depth_left=5,
        guarded_left=1,
        provider_fuel=5,
        node_budget=32,
        requirements=(CG.CallsScopeBinder(0),),
    )
    guarded_options = CG._required_scope_certificate_options(
        guarded_relation,
        FloatTy,
        guarded_scope,
        5,
        (0,),
        guarded_query,
    )
    @test any(option -> option.kind == :guarded_route, guarded_options)
    @test !any(
        key -> key isa Tuple && !isempty(key) && key[1] == :guarded,
        keys(guarded_relation.sampling_options),
    )
    guarded_route = only(option for option in guarded_options if option.kind == :guarded_route)
    CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(2000), 32),
        guarded_relation,
        guarded_route,
        guarded_scope,
        5,
        1,
        true,
        guarded_query,
    )
    @test any(
        key -> key isa Tuple && !isempty(key) && key[1] == :guarded,
        keys(guarded_relation.sampling_options),
    )
    direct_call = CG.sample_generation_derivation_using_scope_binder(
        direct_state,
        direct_relation,
        direct_parent,
        function_ty;
        target_ty=function_ty,
        require_scope_application=true,
    )
    @test direct_call !== nothing
    direct_call === nothing || @test CG.derivation_calls_scope_binder(direct_call, 0)

    rejected_state = CG.DerivationSamplingState(Random.Xoshiro(555), 32)
    rejected_nodes = rejected_state.nodes_left
    rejected_call = CG.sample_generation_derivation_using_scope_binder(
        rejected_state,
        direct_relation,
        direct_parent,
        function_ty;
        target_ty=function_ty,
        require_scope_application=true,
        accept_derivation=_ -> false,
    )
    @test rejected_call === nothing
    @test rejected_state.nodes_left == rejected_nodes

    struct CachedCertificateProvider <: CG.AbstractHeadProvider end
    CG.add_heads!(::CachedCertificateProvider, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::CachedCertificateProvider) = true
    CG.generation_provider_cache_key(::CachedCertificateProvider) =
        (:cached_certificate_provider,)
    function CG.add_generation_options!(
        ::CachedCertificateProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight=1.0, token=:cached_certificate))
        return options
    end

    cached_relation = CG.CoreRelation(
        providers=(CachedCertificateProvider(),),
        weights=CG.GenerationWeights(default=1),
    )
    cached_scope = Ty[function_ty, FloatTy]
    with_fuel = CG.GenerationQuery(
        FloatTy;
        scope_tys=cached_scope,
        depth_left=3,
        provider_fuel=1,
        node_budget=16,
        requirements=(CG.CallsScopeBinder(0),),
    )
    without_fuel = CG.GenerationQuery(
        FloatTy;
        scope_tys=cached_scope,
        depth_left=3,
        provider_fuel=0,
        node_budget=16,
        requirements=(CG.CallsScopeBinder(0),),
    )
    with_fuel_options = CG._required_scope_certificate_options(
        cached_relation,
        FloatTy,
        cached_scope,
        3,
        (0,),
        with_fuel,
    )
    without_fuel_options = CG._required_scope_certificate_options(
        cached_relation,
        FloatTy,
        cached_scope,
        3,
        (0,),
        without_fuel,
    )
    @test any(option -> option.kind == :generation_extension, with_fuel_options)
    @test !any(option -> option.kind == :generation_extension, without_fuel_options)
end

@testitem "CanonicalGeneration disabled constructive routes stay lazy" begin
    using Test
    using Random
    using LambdaRegression: CanonicalGeneration
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, Ty, TyFun, TyList

    CG = CanonicalGeneration

    mutable struct CountingListCollector
        calls::Base.RefValue{Int}
        head::CG.HeadEntry
    end

    mutable struct SafeMaskProbeProvider <: CG.AbstractHeadProvider
        mask_seen::Base.RefValue{Bool}
        safe_provider_seen::Base.RefValue{Bool}
    end

    mutable struct BlockedSafeExtensionProvider <: CG.AbstractHeadProvider
        called::Base.RefValue{Bool}
    end
    CG.provider_capabilities(::BlockedSafeExtensionProvider) = (:safe_list_access,)
    CG.add_heads!(::BlockedSafeExtensionProvider, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::BlockedSafeExtensionProvider) = true
    CG.generation_provider_cache_key(::BlockedSafeExtensionProvider) =
        (:blocked_safe_extension,)
    function CG.add_generation_options!(
        ::BlockedSafeExtensionProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight=1.0, token=:blocked_safe_extension))
        return options
    end
    function CG.sample_generation_extension(
        provider::BlockedSafeExtensionProvider,
        state,
        relation,
        option,
        query,
    )
        provider.called[] = true
        CG.consume_generation_nodes!(state, 1) || return nothing
        return CG.VariableDerivation(1, FloatTy)
    end
    CG.provider_capabilities(::SafeMaskProbeProvider) = (:safe_mask_probe,)
    CG.add_heads!(::SafeMaskProbeProvider, heads, universe; kwargs...) = heads
    CG.provides_generation_options(::SafeMaskProbeProvider) = true
    CG.generation_provider_cache_key(::SafeMaskProbeProvider) = (:safe_mask_probe,)
    function CG.add_generation_options!(
        ::SafeMaskProbeProvider,
        options,
        relation,
        query,
    )
        push!(options, (weight=1.0, token=:probe))
        return options
    end
    function CG.sample_generation_extension(
        provider::SafeMaskProbeProvider,
        state,
        relation,
        option,
        query,
    )
        provider.mask_seen[] = :safe_list_access in query.capability_mask
        provider.safe_provider_seen[] = any(
            candidate -> candidate isa CG.SafeListAccessProvider,
            relation.providers,
        )
        CG.consume_generation_nodes!(state, 1) || return nothing
        return CG.VariableDerivation(1, FloatTy)
    end
    function (collector::CountingListCollector)(relation, scope_tys)
        collector.calls[] += 1
        return (collector.head,)
    end

    calls = Ref(0)
    list_head = CG.HeadEntry(
        :COUNTED_LIST,
        :test,
        TyList(FloatTy),
        CG.CoreProductive(),
        CG.WeightBucket(:counted_list);
        metadata=(value=[1.0],),
    )
    collector = CountingListCollector(calls, list_head)
    function_ty = TyFun(FloatTy, FloatTy)
    scope = Ty[function_ty, FloatTy]
    query = CG.GenerationQuery(
        FloatTy;
        scope_tys=scope,
        depth_left=3,
        provider_fuel=3,
        node_budget=16,
        requirements=(CG.UsesScopeBinder(0),),
    )
    relation = CG.CoreRelation(
        providers=(CG.SafeListAccessProvider(source_collectors=(collector,)),),
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            scope_application=1,
            safe_list_access=1,
            counted_list=1,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=0,
            proxy_scope_application=1,
            proxy_head=0,
            proxy_split=0,
        ),
    )
    state = CG.DerivationSamplingState(Random.Xoshiro(1), 16)

    mask_seen = Ref(false)
    safe_provider_seen = Ref(false)
    blocked_called = Ref(false)
    mask_probe = SafeMaskProbeProvider(mask_seen, safe_provider_seen)
    blocked_provider = BlockedSafeExtensionProvider(blocked_called)
    disabled_relation = CG.CoreRelation(
        providers=(
            CG.SafeListAccessProvider(source_collectors=(collector,)),
            mask_probe,
            blocked_provider,
        ),
        weights=relation.weights,
    )
    disabled_options = CG._required_scope_certificate_options(
        disabled_relation,
        FloatTy,
        scope,
        3,
        (0,),
        query;
        allow_safe_list_access=false,
    )
    @test calls[] == 0
    @test !any(option -> option.kind == :safe_list_access, disabled_options)
    extension_option = only(
        option for option in disabled_options
        if option.kind == :generation_extension && option.value.provider === mask_probe
    )
    extension_derivation = CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(42), 16),
        disabled_relation,
        extension_option,
        scope,
        3,
        0,
        false,
        query,
    )
    @test extension_derivation !== nothing
    @test mask_seen[]
    @test !safe_provider_seen[]
    @test calls[] == 0
    blocked_option = only(
        option for option in disabled_options
        if option.kind == :generation_extension &&
           option.value.provider === blocked_provider
    )
    @test CG._materialize_required_scope_certificate(
        CG.DerivationSamplingState(Random.Xoshiro(8000), 16),
        disabled_relation,
        blocked_option,
        scope,
        3,
        0,
        false,
        query,
    ) === nothing
    @test !blocked_called[]
    disabled_derivation = CG._sample_required_scope_certificate_derivation(
        CG.DerivationSamplingState(Random.Xoshiro(555), 16),
        disabled_relation,
        FloatTy,
        scope,
        3,
        0,
        false,
        query,
    )
    @test disabled_derivation !== nothing
    @test calls[] == 0

    derivation = CG._sample_fast_proxy_insertion_derivation(
        state,
        relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0,
        true,
        query,
    )

    @test derivation !== nothing
    @test calls[] == 0
    @test !any(keys(relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :fast_required_productive_head_plans
    end

    calls[] = 0
    outer_state = CG.DerivationSamplingState(Random.Xoshiro(2000), 16)
    outer_derivation = CG._sample_proxy_using_derivation(
        outer_state,
        relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0;
        allow_safe_list_access=true,
        generation_query=query,
    )
    @test outer_derivation !== nothing
    @test calls[] == 0

    fallback_relation = CG.CoreRelation(
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            scope_application=1,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=0,
            proxy_scope_application=1,
            proxy_head=1,
            proxy_required_head=0,
            proxy_split=0,
        ),
    )
    fallback_state = CG.DerivationSamplingState(Random.Xoshiro(1), 16)
    fallback_derivation = CG._sample_fast_proxy_insertion_derivation(
        fallback_state,
        fallback_relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0,
        false,
        query,
    )
    @test fallback_derivation !== nothing
    @test !any(keys(fallback_relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :fast_required_productive_head_plans
    end
    rejected_state = CG.DerivationSamplingState(Random.Xoshiro(1111), 16)
    rejected_derivation = CG._sample_fast_proxy_insertion_derivation(
        rejected_state,
        fallback_relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0,
        false,
        query;
        accept_derivation=derivation -> false,
    )
    @test rejected_derivation === nothing
    @test !any(keys(fallback_relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :fast_required_productive_head_plans
    end

    acceptance_calls = Ref(0)
    accepted_once_state = CG.DerivationSamplingState(Random.Xoshiro(1234), 16)
    acceptance_relation = CG.CoreRelation(
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            scope_application=1,
            proxy_constructive=1,
            proxy_general=1,
            proxy_safe_list=0,
            proxy_scope_application=1,
            proxy_head=1,
            proxy_split=0,
        ),
    )
    accepted_once = CG._sample_proxy_using_derivation(
        accepted_once_state,
        acceptance_relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0;
        generation_query=query,
        accept_derivation=derivation -> begin
            acceptance_calls[] += 1
            false
        end,
    )
    @test accepted_once === nothing
    @test acceptance_calls[] == 1

    disabled_application_relation = CG.CoreRelation(
        weights=CG.GenerationWeights(
            default=0,
            variable=1,
            scope_application=0,
            proxy_constructive=1,
            proxy_general=0,
            proxy_safe_list=0,
            proxy_scope_application=1,
            proxy_head=0,
            proxy_required_head=0,
            proxy_split=0,
        ),
    )
    disabled_application_state = CG.DerivationSamplingState(Random.Xoshiro(8000), 16)
    disabled_application = CG._sample_fast_proxy_insertion_derivation(
        disabled_application_state,
        disabled_application_relation,
        FloatTy,
        scope,
        3,
        0,
        1,
        0,
        false,
        query;
        allow_direct_result=false,
    )
    @test disabled_application === nothing
    @test !any(keys(disabled_application_relation.sampling_options)) do key
        key isa Tuple && !isempty(key) && key[1] == :fast_required_scope_application_plans
    end
    @test !CG._required_scope_binder_reachable(
        disabled_application_relation,
        BoolTy,
        Ty[FloatTy, TyFun(FloatTy, BoolTy)],
        3,
        1,
    )

    add = x -> y -> x + y
    overlap_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:ADD => add),
        global_consts_ty=Dict{Symbol,Any}(:ADD => TyFun(FloatTy, TyFun(FloatTy, FloatTy))),
        providers=(CG.ContextHeadProvider(),),
        weights=CG.GenerationWeights(default=1),
    )
    overlap_scope = Ty[FloatTy]
    overlap_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=overlap_scope,
        depth_left=3,
        provider_fuel=3,
        node_budget=16,
        requirements=(CG.UsesScopeBinder(0),),
    )
    @test isempty(CG._fast_split_requirement_head_plans(
        overlap_relation,
        FloatTy,
        overlap_scope,
        3,
        1,
        0,
        overlap_query,
    ))
end

@testitem "CanonicalGeneration custom derivations use the public extension protocol" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs: AbsNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyScheme, newtyvar

    CG = LambdaRegression.CanonicalGeneration

    struct ProtocolDerivation <: CG.CoreDerivation end
    struct ProtocolProvider <: CG.AbstractHeadProvider end

    CG.add_heads!(::ProtocolProvider, heads, universe; kwargs...) = heads
    CG.provider_capabilities(::ProtocolProvider) = (:protocol_test,)
    CG.provides_generation_options(::ProtocolProvider) = true
    function CG.add_generation_options!(::ProtocolProvider, options, relation, query)
        push!(options, (weight=1.0,))
        return options
    end
    function CG.sample_generation_extension(::ProtocolProvider, state, relation, option, query)
        CG.consume_generation_nodes!(state, 1) || return nothing
        return ProtocolDerivation()
    end

    CG.derivation_node_count(::ProtocolDerivation) = 1
    CG.derivation_signature(::ProtocolDerivation) = (:protocol_derivation,)
    CG.derivation_result_type(::ProtocolDerivation) = FloatTy
    CG.derivation_uses_scope_binder(::ProtocolDerivation, index::Integer, offset::Integer=0) =
        index == 0 && offset == 0
    CG.rigid_safe_derivation(::ProtocolDerivation) = true
    CG.derivation_metadata(::ProtocolDerivation) = Dict{Symbol,Any}(:protocol_test => true)
    function CG.materialize_generation_derivation(
        ::CG.DefaultAstMaterializer,
        ::ProtocolDerivation,
        local_consts,
        local_const_types,
        relation,
        rng,
    )
        local_consts[:protocol_value] = 1.0
        local_const_types[:protocol_value] = FloatTy
        return ConstNode(:protocol_value)
    end

    result = CG.generate_canonical(
        FloatTy;
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(1111),
            providers=(ProtocolProvider(),),
            weights=CG.GenerationWeights(default=0),
            sampling_depth_limit=2,
            sampling_node_budget=4,
        ),
    )

    @test !(result isa LambdaRegression.Utils.Err)
    @test result.tree.root == ConstNode(:protocol_value)
    @test result.metadata[:derivation] == (:protocol_derivation,)
    @test result.metadata[:derivation_metadata][:protocol_test]
    @test result.metadata[:sampling_nodes_used] == 1
    @test result.local_const_types[:protocol_value] == FloatTy

    nested = CG.LambdaDerivation(FloatTy, ProtocolDerivation())
    nested_consts = Dict{Symbol,Any}()
    nested_types = Dict{Symbol,LambdaRegression.TypeSystem.TyLike}()
    nested_node = CG.materialize_generation_derivation(
        CG.DefaultAstMaterializer(),
        nested,
        nested_consts,
        nested_types,
        CG.CoreRelation(),
        Random.Xoshiro(1234),
    )
    @test nested_node isa AbsNode
    @test nested_node.body == ConstNode(:protocol_value)
    @test CG.derivation_metadata(nested)[:protocol_test]
    @test CG.derivation_result_type(nested) == TyFun(FloatTy, FloatTy)

    alpha = newtyvar(0)
    rigid_result = CG.generate_canonical(
        TyScheme([alpha], alpha);
        config=CG.GenerationConfig(
            rng=Random.Xoshiro(8000),
            providers=(ProtocolProvider(),),
            weights=CG.GenerationWeights(default=0),
            sampling_depth_limit=2,
            sampling_node_budget=4,
        ),
    )
    @test rigid_result isa LambdaRegression.Utils.Err
end

@testitem "CanonicalGeneration known witness extraction is finite and acyclic" begin
    using Test
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, IntTy, Ty, TyFun, TyMu, newtyvar

    CG = LambdaRegression.CanonicalGeneration

    impossible_relation = CG.CoreRelation(
        weights=CG.GenerationWeights(default=1),
    )
    impossible_query = CG.GenerationQuery(
        BoolTy;
        scope_tys=Ty[TyFun(BoolTy, BoolTy)],
        depth_left=32,
        provider_fuel=32,
        node_budget=128,
    )
    @test CG.known_generation_witness(impossible_relation, impossible_query) === nothing

    forbidden_relation = CG.CoreRelation(weights=CG.GenerationWeights(default=1))
    unrestricted_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=0,
        node_budget=1,
    )
    @test CG.derivation_signature(
        CG.known_generation_witness(forbidden_relation, unrestricted_query),
    )[1:2] == (:var, 0)
    forbidden_query = CG.GenerationQuery(
        FloatTy;
        scope_tys=Ty[FloatTy],
        depth_left=0,
        node_budget=1,
        requirements=(CG.ForbidsScopeBinder(0),),
    )
    @test CG.known_generation_witness(forbidden_relation, forbidden_query) === nothing

    forbidden_function = TyFun(FloatTy, FloatTy)
    rebased_forbidden_query = CG.GenerationQuery(
        forbidden_function;
        scope_tys=Ty[forbidden_function],
        depth_left=1,
        node_budget=2,
        requirements=(CG.ForbidsScopeBinder(0),),
    )
    rebased_witness = CG.known_generation_witness(
        forbidden_relation,
        rebased_forbidden_query,
    )
    @test rebased_witness isa CG.LambdaDerivation
    @test rebased_witness.body isa CG.VariableDerivation
    @test rebased_witness.body.index == 0

    identity_planner = (derivations, args...) -> derivations
    anchored_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:a => 1.0, :f => identity),
        global_consts_ty=Dict{Symbol,Any}(:a => FloatTy, :f => TyFun(FloatTy, FloatTy)),
        argument_planner=identity_planner,
        weights=CG.GenerationWeights(default=1),
    )
    anchored_query = CG.GenerationQuery(
        FloatTy;
        depth_left=32,
        provider_fuel=32,
        node_budget=128,
    )
    witness = CG.known_generation_witness(anchored_relation, anchored_query)

    @test witness !== nothing
    @test CG.derivation_signature(witness) == (:const, :constant, :a, CG.typekey(FloatTy))

    cached_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => true),
        global_consts_ty=Dict{Symbol,Any}(:mk => TyFun(BoolTy, FloatTy), :seed => BoolTy),
        weights=CG.GenerationWeights(default=1),
    )
    cached_query = CG.GenerationQuery(
        FloatTy;
        depth_left=2,
        provider_fuel=2,
        node_budget=3,
    )
    first_cached = CG.known_generation_witness(cached_relation, cached_query)
    first_signature = CG.derivation_signature(first_cached)
    @test first_cached isa CG.HeadApplicationDerivation
    @test length(cached_relation.known_witnesses) == 1
    @test only(values(cached_relation.known_witnesses)).nodes ==
        CG.derivation_node_count(first_cached)
    first_cached.args[1] = CG.VariableDerivation(0, BoolTy)

    second_cached = CG.known_generation_witness(cached_relation, cached_query)
    @test second_cached !== first_cached
    @test CG.derivation_signature(second_cached) == first_signature
    @test second_cached.args[1] isa CG.ConstantDerivation

    borrowed_first = CG._borrow_known_generation_witness(cached_relation, cached_query)
    borrowed_second = CG._borrow_known_generation_witness(cached_relation, cached_query)
    @test borrowed_second === borrowed_first
    @test CG.known_generation_witness(cached_relation, cached_query) !== borrowed_first

    delete!(cached_relation.global_consts, :seed)
    delete!(cached_relation.global_consts_ty, :seed)
    @test CG.known_generation_witness(cached_relation, cached_query) === nothing

    negative_relation = CG.CoreRelation(weights=CG.GenerationWeights(default=1))
    negative_query = CG.GenerationQuery(BoolTy; depth_left=1, node_budget=2)
    @test CG.known_generation_witness(negative_relation, negative_query) === nothing
    negative_relation.global_consts[:truth] = true
    negative_relation.global_consts_ty[:truth] = BoolTy
    @test CG.known_generation_witness(negative_relation, negative_query) !== nothing

    mutable_cache_ty = newtyvar(0)
    unbound_cache_key = CG._cache_type_key(mutable_cache_ty)
    @test LambdaRegression.TypeSystem.unify!(mutable_cache_ty, BoolTy) === nothing
    @test CG._cache_type_key(mutable_cache_ty) != unbound_cache_key

    mutable_mu_binder = newtyvar(0)
    mutable_mu = TyMu(
        mutable_mu_binder,
        TyFun(FloatTy, mutable_mu_binder),
    )
    mutable_mu_relation = CG.CoreRelation(weights=CG.GenerationWeights(default=1))
    mutable_mu_query = CG.GenerationQuery(
        mutable_mu;
        scope_tys=Ty[mutable_mu],
        depth_left=0,
        provider_fuel=0,
        node_budget=1,
    )
    @test CG.known_generation_witness(mutable_mu_relation, mutable_mu_query) !== nothing
    @test length(mutable_mu_relation.known_witnesses) == 1
    @test LambdaRegression.TypeSystem.unify!(mutable_mu_binder, BoolTy) === nothing
    @test CG.known_generation_witness(mutable_mu_relation, mutable_mu_query) !== nothing
    @test length(mutable_mu_relation.known_witnesses) == 2

    fixed_point_ty = TyFun(TyFun(FloatTy, FloatTy), FloatTy)
    fixed_point_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:fix => identity),
        global_consts_ty=Dict{Symbol,Any}(:fix => fixed_point_ty),
        weights=CG.GenerationWeights(default=1),
    )
    fixed_point_query = CG.GenerationQuery(
        FloatTy;
        depth_left=4,
        provider_fuel=4,
        node_budget=16,
    )
    @test CG.known_generation_witness(
        fixed_point_relation,
        fixed_point_query;
        capability_mask=(:fixed_point, :opaque),
    ) === nothing

    bool_ty = BoolTy
    minimal_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(
            :big => identity,
            :finish => identity,
            :seed => 1,
            :small => identity,
        ),
        global_consts_ty=Dict{Symbol,Any}(
            :big => TyFun(IntTy, TyFun(IntTy, bool_ty)),
            :finish => TyFun(bool_ty, TyFun(bool_ty, FloatTy)),
            :seed => IntTy,
            :small => TyFun(IntTy, bool_ty),
        ),
        argument_planner=(derivations, args...) -> derivations,
        weights=CG.GenerationWeights(default=1),
    )
    minimal_query = CG.GenerationQuery(
        FloatTy;
        depth_left=4,
        provider_fuel=4,
        node_budget=5,
    )
    minimal = CG.known_generation_witness(minimal_relation, minimal_query)
    @test minimal !== nothing
    @test CG._signature_contains(CG.derivation_signature(minimal), :small)

    recursive_binder = newtyvar(0)
    recursive_target = TyMu(recursive_binder, TyFun(IntTy, recursive_binder))
    recursive_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => 1),
        global_consts_ty=Dict{Symbol,Any}(
            :mk => TyFun(IntTy, recursive_target),
            :seed => IntTy,
        ),
        weights=CG.GenerationWeights(default=1),
    )
    recursive_query = CG.GenerationQuery(
        recursive_target;
        depth_left=2,
        provider_fuel=2,
        node_budget=3,
    )
    recursive_witness = CG.known_generation_witness(recursive_relation, recursive_query)
    @test recursive_witness !== nothing
    @test CG._signature_contains(CG.derivation_signature(recursive_witness), :mk)

    rejecting_planner = (derivations, args...) -> empty(derivations)
    rejected_relation = CG.CoreRelation(
        global_consts=Dict{Symbol,Any}(:mk => identity, :seed => 1),
        global_consts_ty=Dict{Symbol,Any}(:mk => TyFun(IntTy, FloatTy), :seed => IntTy),
        argument_planner=rejecting_planner,
        weights=CG.GenerationWeights(default=1),
    )
    rejected_query = CG.GenerationQuery(
        FloatTy;
        depth_left=2,
        provider_fuel=2,
        node_budget=3,
    )
    @test CG.known_generation_witness(rejected_relation, rejected_query) !== nothing
end

@testitem "CanonicalGeneration positional weights reject invalid values" begin
    using Test
    CG = LambdaRegression.CanonicalGeneration
    @test_throws ArgumentError CG.GenerationWeights(Inf, Dict{Symbol,Float64}())
    @test_throws ArgumentError CG.GenerationWeights(1.0, Dict{Symbol,Float64}(:constant => -1.0))
end

end # module CanonicalGeneration
