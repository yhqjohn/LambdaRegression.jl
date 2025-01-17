"""
    CanonicalGeneration

Canonical finite-core generation.

This module constructs typed programs from finite derivation rules.
"""
module CanonicalGeneration

import Random
import ..TypeSystem

using ..ASTs.NodeASTs: AstNode, AstTree, ConstNode, IndexNode, AbsNode, AppNode
using ..TypeSystem: Ty, TyConst, TyVar, TyFun, TyProd, TySum, TyList, TyArray, TyNat, TyMu, TyScheme, TyLike
using ..TypeSystem: BoolTy, FloatTy, get_ast_type_ctx, get_type_ctx, get_type, iscompatible, typekey, →
using ..Utils
using TestItems

export generate_canonical
export GenerationConfig, GenerationResult

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
struct RealOperatorWeight <: CoreWeightBucket end
struct ToListWeight <: CoreWeightBucket end
struct AMapWeight <: CoreWeightBucket end
struct LenFWeight <: CoreWeightBucket end

const _VARIABLE_WEIGHT = VariableWeight()
const _CONSTANT_WEIGHT = ConstantWeight()
const _PRESET_TERM_WEIGHT = PresetTermWeight()
const _INSTANCE_FACTORY_WEIGHT = InstanceFactoryWeight()
const _ARRAY_SOURCE_WEIGHT = ArraySourceWeight()
const _REAL_OPERATOR_WEIGHT = RealOperatorWeight()
const _TOLIST_WEIGHT = ToListWeight()
const _AMAP_WEIGHT = AMapWeight()
const _LENF_WEIGHT = LenFWeight()

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

function _build_core_catalog(
    universe::CoreUniverse;
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    scope_source_tys::AbstractVector{<:Ty}=Ty[],
)
    heads = HeadEntry[]
    _add_typed_heads!(heads, :variable, local_consts_ty, _VARIABLE_WEIGHT)
    _add_typed_heads!(heads, :constant, global_consts_ty, _CONSTANT_WEIGHT)
    _add_typed_heads!(
        heads,
        :preset,
        _typed_presets(presets, preset_types; global_consts_ty, local_consts_ty),
        _PRESET_TERM_WEIGHT,
    )
    _add_nil_factories!(heads, universe)
    _add_real_operators!(heads, universe)
    source_arrays = _closed_source_array_types(heads, universe, scope_source_tys)
    _add_lenf_heads!(heads, universe, source_arrays)
    _add_tolist_heads!(heads, universe, source_arrays)
    _add_amap_heads!(heads, universe, source_arrays)
    rules = _core_rules(heads, universe)
    return CoreCatalog(universe, heads, rules)
end

function _add_typed_heads!(
    heads::Vector{HeadEntry},
    source::Symbol,
    tyctx::AbstractDict{Symbol,<:Any},
    weight_bucket::CoreWeightBucket,
)
    for name in sort!(collect(keys(tyctx)))
        ty = tyctx[name]
        ty isa TyLike || continue
        ty isa Ty && !_in_closed_fragment(ty) && continue
        push!(heads, HeadEntry(name, source, ty, _rule_class_for_tylike(ty), weight_bucket))
    end
    return heads
end

_rule_class_for_tylike(ty::TyLike) = ty isa Ty && _in_closed_fragment(ty) ? CoreProductive() : Decoration()

function _add_nil_factories!(heads::Vector{HeadEntry}, universe::CoreUniverse)
    for ty in universe.types
        ty isa TyList || continue
        push!(
            heads,
            HeadEntry(:NIL, :instance_factory, ty, CoreProductive(), _INSTANCE_FACTORY_WEIGHT; metadata=(factory=:NIL,)),
        )
    end
    return heads
end

const _REAL_OPERATOR_VALUES = (
    (:FSIN, TypeSystem.FSIN),
    (:FSQRT, TypeSystem.FSQRT),
    (:SQR, TypeSystem.SQR),
    (:FADD, TypeSystem.FADD),
    (:FSUB, TypeSystem.FSUB),
    (:FMUL, TypeSystem.FMUL),
    (:FDIV, TypeSystem.FDIV),
)

const _BOOL_REAL_OPERATOR_VALUES = (
    (:FEQ, TypeSystem.FEQ),
    (:FLE, TypeSystem.FLE),
)

function _add_real_operators!(heads::Vector{HeadEntry}, universe::CoreUniverse)
    _type_in_universe(universe, FloatTy) || return heads
    for (name, value) in _REAL_OPERATOR_VALUES
        ty = get_type(value)
        ty isa Ty && _add_closed_primitive_head!(heads, universe, name, ty, _REAL_OPERATOR_WEIGHT)
    end
    _type_in_universe(universe, BoolTy) || return heads
    for (name, value) in _BOOL_REAL_OPERATOR_VALUES
        ty = get_type(value)
        ty isa Ty && _add_closed_primitive_head!(heads, universe, name, ty, _REAL_OPERATOR_WEIGHT)
    end
    return heads
end

function _add_closed_primitive_head!(
    heads::Vector{HeadEntry},
    universe::CoreUniverse,
    name::Symbol,
    ty::Ty,
    bucket::CoreWeightBucket,
)
    args, result = _split_fun(ty)
    _type_in_universe(universe, result) || return heads
    all(arg -> _type_in_universe(universe, arg), args) || return heads
    push!(heads, HeadEntry(name, :primitive, ty, CoreProductive(), bucket))
    return heads
end

function _add_lenf_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, source_arrays::Vector{TyArray})
    _type_in_universe(universe, FloatTy) || return heads
    for array_ty in source_arrays
        push!(heads, HeadEntry(:LENF, :primitive, array_ty → FloatTy, CoreProductive(), _LENF_WEIGHT))
    end
    return heads
end

function _add_tolist_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, source_arrays::Vector{TyArray})
    for array_ty in source_arrays
        list_ty = TyList(array_ty.elem)
        _type_in_universe(universe, list_ty) || continue
        push!(heads, HeadEntry(:TOLIST, :primitive, array_ty → list_ty, CoreProductive(), _TOLIST_WEIGHT))
    end
    return heads
end

function _add_amap_heads!(heads::Vector{HeadEntry}, universe::CoreUniverse, source_arrays::Vector{TyArray})
    arrays = _closed_array_types(universe)
    for source_ty in source_arrays
        for result_ty in arrays
            source_ty.len == result_ty.len || continue
            fun_arg = source_ty.elem → result_ty.elem
            amap_ty = fun_arg → source_ty → result_ty
            rule_class = source_ty.elem == result_ty.elem ? Decoration() : CoreProductive()
            push!(heads, HeadEntry(:AMAP, :primitive, amap_ty, rule_class, _AMAP_WEIGHT))
        end
    end
    return heads
end

_closed_array_types(universe::CoreUniverse) = [ty for ty in universe.types if ty isa TyArray]

function _closed_source_array_types(
    heads::Vector{HeadEntry},
    universe::CoreUniverse,
    scope_source_tys::AbstractVector{<:Ty}=Ty[],
)
    arrays = TyArray[]
    seen = Set{Any}()
    for ty in scope_source_tys
        ty isa TyArray || continue
        _in_closed_fragment(ty) || continue
        _type_in_universe(universe, ty) || continue
        key = typekey(ty)
        key in seen && continue
        push!(seen, key)
        push!(arrays, ty)
    end
    for head in heads
        _is_source_head(head) || continue
        head.type isa TyArray || continue
        _type_in_universe(universe, head.type) || continue
        key = typekey(head.type)
        key in seen && continue
        push!(seen, key)
        push!(arrays, head.type)
    end
    return arrays
end

_is_source_head(head::HeadEntry) = head.source in (:variable, :constant, :preset, :instance_factory)

function _split_fun(ty::Ty)
    args = Ty[]
    rest = ty
    while rest isa TyFun
        push!(args, rest.arg)
        rest = rest.res
    end
    return args, rest
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
    GenerationConfig(; rng=Random.default_rng())

Configuration placeholder for canonical generation.
"""
struct GenerationConfig
    rng::Random.AbstractRNG
end
GenerationConfig(; rng::Random.AbstractRNG=Random.default_rng()) = GenerationConfig(rng)

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

mutable struct CoreRelation
    derivations::Dict{Tuple{Any,Any},Vector{CoreDerivation}}
    signatures::Dict{Tuple{Any,Any},Set{Any}}
    visiting::Set{Tuple{Any,Any}}
    computed::Set{Tuple{Any,Any}}
    global_consts::Dict{Symbol,Any}
    global_consts_ty::Dict{Symbol,TyLike}
    local_consts_ty::Dict{Symbol,TyLike}
    presets::Dict{Symbol,Any}
    preset_types::Dict{Symbol,Any}
end

function CoreRelation(;
    global_consts::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    global_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    local_consts_ty::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    presets::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    preset_types::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
)
    return CoreRelation(
        Dict{Tuple{Any,Any},Vector{CoreDerivation}}(),
        Dict{Tuple{Any,Any},Set{Any}}(),
        Set{Tuple{Any,Any}}(),
        Set{Tuple{Any,Any}}(),
        _anydict(global_consts),
        Dict{Symbol,TyLike}(name => ty for (name, ty) in pairs(global_consts_ty) if ty isa TyLike),
        Dict{Symbol,TyLike}(name => ty for (name, ty) in pairs(local_consts_ty) if ty isa TyLike),
        _anydict(presets),
        Dict{Symbol,Any}(name => ty for (name, ty) in pairs(preset_types)),
    )
end

"""
    generate_canonical(target_ty; kwargs...)

Generate a beta-normal witness inside the current canonical finite core.
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
    config::GenerationConfig=GenerationConfig(; rng),
)
    target_ty isa Ty || return Err(UnsupportedCanonicalGenerationTarget(target_ty))
    _in_closed_fragment(target_ty) || return Err(UnsupportedCanonicalGenerationTarget(target_ty))

    result_local_consts = _anydict(local_consts)
    result_local_const_types = Dict{Symbol,TyLike}(
        name => ty for (name, ty) in pairs(local_consts_ty) if ty isa TyLike
    )
    relation = CoreRelation(
        global_consts=global_consts,
        global_consts_ty=global_consts_ty,
        local_consts_ty=result_local_const_types,
        presets=presets,
        preset_types=preset_types,
    )
    derivations = _known_derivations!(relation, target_ty, Ty[])
    isempty(derivations) && return Err(UnsupportedCanonicalGenerationTarget(target_ty))

    derivation = rand(config.rng, derivations)
    node = _build_derivation_ast!(derivation, result_local_consts, result_local_const_types, relation)
    metadata = Dict{Symbol,Any}(
        :generator => :canonical_finite_core,
        :rng_type => typeof(config.rng),
        :known_count => sum(length, values(relation.derivations)),
        :scope_count => length(Set(first(key) for key in keys(relation.derivations))),
        :derivation => _derivation_signature(derivation),
    )
    return GenerationResult(AstTree(node), target_ty, result_local_consts, result_local_const_types, metadata)
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
        head.source == :instance_factory && continue
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
    return _add_derivation!(relation, key, LambdaDerivation(target_ty.arg, first(body_derivations)))
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
        arg_derivations = Vector{CoreDerivation}[]
        all_known = true
        for arg_ty in rule.argument_tys
            derivs = copy(_known_derivations!(relation, arg_ty, scope_tys))
            if isempty(derivs)
                all_known = false
                break
            end
            push!(arg_derivations, derivs)
        end
        all_known || continue
        args = [first(derivs) for derivs in arg_derivations]
        changed |= _add_derivation!(relation, key, HeadApplicationDerivation(rule.head, args))
    end
    return changed
end

function _relation_catalog(relation::CoreRelation, target_ty::Ty, scope_tys::Vector{Ty})
    universe = CoreUniverse(
        target_ty;
        global_consts_ty=relation.global_consts_ty,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        scope_source_tys=scope_tys,
    )
    return _build_core_catalog(
        universe;
        global_consts_ty=relation.global_consts_ty,
        local_consts_ty=relation.local_consts_ty,
        presets=relation.presets,
        preset_types=relation.preset_types,
        scope_source_tys=scope_tys,
    )
end

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
)

function _build_derivation_ast!(
    derivation::CoreDerivation,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
)
    if derivation isa VariableDerivation
        return IndexNode(derivation.index)
    elseif derivation isa ConstantDerivation
        return _head_node!(derivation.head, local_consts, local_consts_ty, relation)
    elseif derivation isa LambdaDerivation
        return AbsNode(:x, _build_derivation_ast!(derivation.body, local_consts, local_consts_ty, relation))
    elseif derivation isa HeadApplicationDerivation
        node = _head_node!(derivation.head, local_consts, local_consts_ty, relation)
        for arg in derivation.args
            arg_node = _build_derivation_ast!(arg, local_consts, local_consts_ty, relation)
            node = AppNode(node, arg_node)
        end
        return node
    else
        error("Unknown core derivation $(typeof(derivation))")
    end
end

function _head_node!(
    head::HeadEntry,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
    relation::CoreRelation,
)
    if head.source == :instance_factory && head.name == :NIL
        name = _fresh_local_name(local_consts_ty, :NIL)
        local_consts[name] = Any[]
        local_consts_ty[name] = head.type
        return ConstNode(name)
    end

    local_consts_ty[head.name] = head.type
    if head.source == :primitive
        value = _primitive_value(head.name)
        value === nothing || (local_consts[head.name] = value)
    elseif head.source == :constant && haskey(relation.global_consts, head.name)
        local_consts[head.name] = relation.global_consts[head.name]
    elseif head.source == :preset && haskey(relation.presets, head.name)
        local_consts[head.name] = relation.presets[head.name]
    end
    return ConstNode(head.name)
end

function _fresh_local_name(local_consts_ty::Dict{Symbol,TyLike}, prefix::Symbol)
    i = 1
    while true
        name = Symbol(prefix, i)
        haskey(local_consts_ty, name) || return name
        i += 1
    end
end

const _PRIMITIVE_VALUES = Dict{Symbol,Any}(
    :FSIN => TypeSystem.FSIN,
    :FSQRT => TypeSystem.FSQRT,
    :SQR => TypeSystem.SQR,
    :FADD => TypeSystem.FADD,
    :FSUB => TypeSystem.FSUB,
    :FMUL => TypeSystem.FMUL,
    :FDIV => TypeSystem.FDIV,
    :FEQ => TypeSystem.FEQ,
    :FLE => TypeSystem.FLE,
    :TOLIST => TypeSystem.TOLIST,
    :AMAP => TypeSystem.AMAP,
    :LENF => TypeSystem.LENF,
)

_primitive_value(name::Symbol) = get(_PRIMITIVE_VALUES, name, nothing)

@testitem "CanonicalGeneration root API smoke" begin
    using Test
    using LambdaRegression

    @test LambdaRegression.CanonicalGeneration isa Module
    @test LambdaRegression.mutate_tree isa Function
    @test LambdaRegression.fit! isa Function
    @test LambdaRegression.TypeSystem isa Module
    @test LambdaRegression.CanonicalGeneration.generate_canonical isa Function
    @test LambdaRegression.CanonicalGeneration.GenerationConfig isa DataType
    @test LambdaRegression.CanonicalGeneration.GenerationResult isa DataType
    @test LambdaRegression.generate_canonical isa Function
    @test LambdaRegression.GenerationConfig isa DataType
    @test LambdaRegression.GenerationResult isa DataType
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

@testitem "CanonicalGeneration catalog contains allowed real operators and closed array LENF" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(FloatTy; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    catalog = CG._build_core_catalog(universe; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    names = Set(entry.name for entry in catalog.heads)

    @test all(name -> name in names, [:FSIN, :FSQRT, :SQR, :FADD, :FSUB, :FMUL, :FDIV])
    @test :LENF in names
end

@testitem "CanonicalGeneration catalog LENF requires closed array source" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat, →

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(Matrix32 → FloatTy)
    catalog = CG._build_core_catalog(universe)
    names = Set(entry.name for entry in catalog.heads)

    @test :LENF ∉ names

    scoped = CG._build_core_catalog(universe; scope_source_tys=[Matrix32])
    scoped_names = Set(entry.name for entry in scoped.heads)

    @test :LENF in scoped_names

    body_universe = CG.CoreUniverse(FloatTy; scope_source_tys=[Matrix32])
    body_catalog = CG._build_core_catalog(body_universe; scope_source_tys=[Matrix32])
    body_names = Set(entry.name for entry in body_catalog.heads)

    @test :LENF in body_names
end

@testitem "CanonicalGeneration catalog excludes unsupported partial and fold primitives" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    CG = LambdaRegression.CanonicalGeneration
    Matrix32 = TyArray(TyNat(3), TyArray(TyNat(2), FloatTy))
    universe = CG.CoreUniverse(Matrix32; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    catalog = CG._build_core_catalog(universe; local_consts_ty=Dict{Symbol,Any}(:m => Matrix32))
    names = Set(entry.name for entry in catalog.heads)

    @test :AFOLDL ∉ names
    @test :HEAD ∉ names
    @test :TAIL ∉ names
end

@testitem "CanonicalGeneration catalog TOLIST and AMAP stay finite and classified" begin
    using Test
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyArray, TyList, TyNat, →

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
    self_amaps = [entry for entry in self_catalog.heads if entry.name == :AMAP]

    @test any(entry -> entry.rule_class isa CG.Decoration, self_amaps)
    @test all(rule -> rule.head.name != :AMAP, self_catalog.rules)

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

@testitem "CanonicalGeneration catalog preset AST terms use typing boundary" begin
    using Test
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.TypeSystem: FloatTy, TyScheme, typekey

    CG = LambdaRegression.CanonicalGeneration
    presets = Dict{Symbol,Any}(:ID => parse_lambda("λx.x"))
    universe = CG.CoreUniverse(FloatTy; presets=presets)
    catalog = CG._build_core_catalog(universe; presets=presets)
    ids = [entry for entry in catalog.heads if entry.name == :ID]

    @test length(ids) == 1
    @test ids[1].source == :preset
    @test ids[1].type isa TyScheme
    @test typekey(ids[1].type) == (:fun, (:v, 0), (:v, 0))
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
        TyFun(Matrix32, FloatTy),
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

end # module CanonicalGeneration
