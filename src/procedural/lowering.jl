"""
    fixed_point_witness() -> AstNode

Return a fresh finite core AST for the untyped fixed-point combinator
`λf.(λx.f (x x)) (λx.f (x x))`.
"""
function fixed_point_witness()
    function half()
        xx = AppNode(IndexNode(0), IndexNode(0))
        return AbsNode(:x, AppNode(IndexNode(1), xx))
    end
    return AbsNode(:f, AppNode(half(), half()))
end

function _fixed_point_node(source)
    value = if source isa AstNode || source isa AstTree
        source
    elseif applicable(source)
        source()
    else
        source
    end
    value isa AstTree && return deepcopy(value.root)
    value isa AstNode && return deepcopy(value)
    throw(ArgumentError("fixed_point must be an AstNode, AstTree, or zero-argument factory"))
end

"""Finite typing failure for a FunctionDef whose resolved outer head is not `Fun`."""
struct NonFunctionDefinitionError
    inferred::Ty
end

mutable struct _LoweringContext
    fixed_point
    used_names::Set{Symbol}
    next_name::Int
    function_results::Vector{AstNode}
    sources_by_lowered::IdDict{AstNode,Vector{AstNode}}
end

function _collect_binder_names!(names::Set{Symbol}, node::AstNode)
    node isa VarNode && push!(names, node.name)
    node isa AbsNode && push!(names, node.name)
    node isa LetNode && push!(names, node.name)
    for child in children(node)
        _collect_binder_names!(names, child)
    end
    return names
end

function _lowering_context(root::AstNode, fixed_point)
    names = _collect_binder_names!(Set{Symbol}(), root)
    return _LoweringContext(
        fixed_point,
        names,
        0,
        AstNode[],
        IdDict{AstNode,Vector{AstNode}}(),
    )
end

function _record_source!(context::_LoweringContext, lowered::AstNode, source::AstNode)
    sources = get!(Vector{AstNode}, context.sources_by_lowered, lowered)
    any(existing -> existing === source, sources) || push!(sources, source)
    return lowered
end

_drop_unused_self_slot!(node::Union{ConstNode,VarNode}, context, depth::Int=0) = node

function _drop_unused_self_slot!(
    node::IndexNode,
    context::_LoweringContext,
    depth::Int=0,
)
    node.index < depth && return node
    node.index == depth && throw(ArgumentError("cannot remove a referenced FunctionDef binder"))
    shifted = IndexNode(node.index - 1)
    sources = pop!(context.sources_by_lowered, node, nothing)
    sources === nothing || (context.sources_by_lowered[shifted] = sources)
    return shifted
end

function _drop_unused_self_slot!(node::AbsNode, context::_LoweringContext, depth::Int=0)
    node.body = _drop_unused_self_slot!(node.body, context, depth + 1)
    return node
end

function _drop_unused_self_slot!(node::AppNode, context::_LoweringContext, depth::Int=0)
    node.func = _drop_unused_self_slot!(node.func, context, depth)
    node.arg = _drop_unused_self_slot!(node.arg, context, depth)
    return node
end

function _fresh_binder!(context::_LoweringContext)
    while true
        context.next_name += 1
        name = Symbol("##procedural", context.next_name)
        name in context.used_names && continue
        push!(context.used_names, name)
        return name
    end
end

_lower_procedural(node::ConstNode, context::_LoweringContext) =
    _record_source!(context, deepcopy(node), node)
_lower_procedural(node::VarNode, context::_LoweringContext) =
    _record_source!(context, deepcopy(node), node)
_lower_procedural(node::IndexNode, context::_LoweringContext) =
    _record_source!(context, deepcopy(node), node)
function _lower_procedural(node::AbsNode, context::_LoweringContext)
    lowered = AbsNode(node.name, _lower_procedural(node.body, context))
    return _record_source!(context, lowered, node)
end
function _lower_procedural(node::AppNode, context::_LoweringContext)
    lowered = AppNode(
        _lower_procedural(node.func, context),
        _lower_procedural(node.arg, context),
    )
    return _record_source!(context, lowered, node)
end
function _lower_procedural(node::LetNode, context::_LoweringContext)
    lowered = AppNode(
        AbsNode(node.name, _lower_procedural(node.body, context)),
        _lower_procedural(node.value, context),
    )
    return _record_source!(context, lowered, node)
end

function _lower_procedural(node::ScopeNode, context::_LoweringContext)
    binder_names = [_fresh_binder!(context) for _ in node.bindings]
    bindings = [_lower_procedural(binding, context) for binding in node.bindings]
    result = _lower_procedural(node.body, context)
    for index in Iterators.reverse(eachindex(node.bindings))
        result = AppNode(AbsNode(binder_names[index], result), bindings[index])
    end
    return _record_source!(context, result, node)
end

function _lower_procedural(node::FunctionDefNode, context::_LoweringContext)
    self_referenced = _self_observation(node.body, 0).self_references > 0
    lowered_body = _lower_procedural(node.body, context)
    result = if self_referenced
        self_name = _fresh_binder!(context)
        recursive_body = AbsNode(self_name, lowered_body)
        AppNode(_fixed_point_node(context.fixed_point), recursive_body)
    else
        _drop_unused_self_slot!(lowered_body, context)
    end
    push!(context.function_results, result)
    return _record_source!(context, result, node)
end

function _lower_with_context(root::AstNode, fixed_point)
    context = _lowering_context(root, fixed_point)
    return _lower_procedural(root, context), context
end

"""
    lower_procedural(term; fixed_point=fixed_point_witness)

Return a fresh core lambda term with every procedural node lowered. A FunctionDef
whose self binder is unreferenced removes that unused slot without materialising
the fixed-point witness; the procedural IR node itself is never mutated. Binding
identity is preserved; self-free lowering rebases outer De Bruijn indices by one.
This operation never mutates `term`.
"""
function lower_procedural(node::AstNode; fixed_point=fixed_point_witness)
    lowered, _ = _lower_with_context(node, fixed_point)
    return lowered
end
function lower_procedural(tree::AstTree; fixed_point=fixed_point_witness)
    lowered, _ = _lower_with_context(tree.root, fixed_point)
    return AstTree(lowered)
end

function _constrain_function_head!(ty::Ty, factory)
    current = TypeSystem.prune(ty)
    seen = IdSet{Any}()
    while current isa TyMu
        current in seen && return false
        push!(seen, current)
        current = TypeSystem.prune(current.body)
    end
    if current isa TyVar
        constrained = TyFun(factory(current.level), factory(current.level))
        return isnothing(TypeSystem.unify!(current, constrained))
    end
    return current isa TyFun
end

mutable struct _FunctionResultMeta <: AbstractDict{AstNode,Tuple{TyLike,TyLike}}
    values::IdDict{AstNode,Tuple{TyLike,TyLike}}
    selected::IdSet{AstNode}
end

Base.length(meta::_FunctionResultMeta) = length(meta.values)
Base.iterate(meta::_FunctionResultMeta, state...) = iterate(meta.values, state...)
Base.getindex(meta::_FunctionResultMeta, node::AstNode) = meta.values[node]
function Base.setindex!(meta::_FunctionResultMeta, value, node::AstNode)
    node in meta.selected && (meta.values[node] = value)
    return value
end

"""Native Algorithm-W view of procedural binders without materialising lowering."""
struct _ProceduralTermAdapter end

TypeSystem.term_view(::_ProceduralTermAdapter, node::ConstNode) = (:const, node.name)
TypeSystem.term_view(::_ProceduralTermAdapter, node::VarNode) = (:var, node.name)
TypeSystem.term_view(::_ProceduralTermAdapter, node::IndexNode) = (:var, node.index)
TypeSystem.term_view(::_ProceduralTermAdapter, node::AbsNode) = (:abs, node.name, node.body)
TypeSystem.term_view(::_ProceduralTermAdapter, node::AppNode) = (:app, node.func, node.arg)
TypeSystem.term_view(::_ProceduralTermAdapter, node::LetNode) =
    (:let, node.name, node.value, node.body)
TypeSystem.term_view(::_ProceduralTermAdapter, node::ScopeNode) =
    (:letseq, node.bindings, node.body)
TypeSystem.term_view(::_ProceduralTermAdapter, node::FunctionDefNode) =
    (:fix, nothing, node.body)

function TypeSystem.constrain_fixpoint(
    ::_ProceduralTermAdapter,
    node::FunctionDefNode,
    ty::Ty,
    factory,
)
    _constrain_function_head!(ty, factory) ||
        return Err(NonFunctionDefinitionError(TypeSystem.prune(ty)))
    return TypeSystem.prune(ty)
end

function _collect_function_defs!(result::Vector{FunctionDefNode}, node::AstNode)
    node isa FunctionDefNode && push!(result, node)
    foreach(child -> _collect_function_defs!(result, child), children(node))
    return result
end

function _validate_function_results(function_defs, inference_meta, factory)
    for function_def in function_defs
        inferred = inference_meta[function_def][1]
        _constrain_function_head!(inferred, factory) ||
            return Err(NonFunctionDefinitionError(TypeSystem.prune(inferred)))
    end
    return nothing
end

function _infer_procedural_lowered(root, ctx, factory, meta, fixed_point)
    lowered, lowering_context = _lower_with_context(root, fixed_point)
    inference_meta = if meta === nothing
        _FunctionResultMeta(
            IdDict{AstNode,Tuple{TyLike,TyLike}}(),
            IdSet{AstNode}(lowering_context.function_results),
        )
    else
        IdDict{AstNode,Tuple{TyLike,TyLike}}()
    end
    result = TypeSystem.infer_term(
        lowered,
        Dict{Symbol,TyLike}(ctx),
        TypeSystem.NodeTermAdapter();
        factory,
        meta=inference_meta,
    )
    if meta !== nothing
        for (lowered_node, bounds) in inference_meta
            sources = get(lowering_context.sources_by_lowered, lowered_node, nothing)
            sources === nothing && continue
            foreach(source -> meta[source] = bounds, sources)
        end
    end
    result isa Err && return result
    validation = _validate_function_results(
        lowering_context.function_results,
        inference_meta,
        factory,
    )
    validation isa Err && return validation
    return TypeSystem.prune(result)
end

function _infer_procedural_native(root, ctx, factory, meta)
    function_defs = _collect_function_defs!(FunctionDefNode[], root)
    inference_meta = if meta === nothing
        _FunctionResultMeta(
            IdDict{AstNode,Tuple{TyLike,TyLike}}(),
            IdSet{AstNode}(function_defs),
        )
    else
        IdDict{AstNode,Tuple{TyLike,TyLike}}()
    end
    result = TypeSystem.infer_term(
        root,
        Dict{Symbol,TyLike}(ctx),
        _ProceduralTermAdapter();
        factory,
        meta=inference_meta,
    )
    if meta !== nothing
        foreach(entry -> meta[entry.first] = entry.second, inference_meta)
    end
    result isa Err && return result
    validation = _validate_function_results(function_defs, inference_meta, factory)
    validation isa Err && return validation
    return TypeSystem.prune(result)
end

"""
Infer a procedural term directly with the ordinary equirecursive W driver.
`ScopeNode` is a sequential HM-generalising region and `FunctionDefNode` contributes
one monomorphic self equation. Optional `meta` bounds are keyed directly by source
nodes. Supplying a non-default fixed-point witness preserves the compatibility
lowering path because the witness may intentionally carry different type constraints.
That path is intersected with native FunctionDef typing before it returns; requested
source metadata is produced by a separate native pass so its polymorphic upper
bounds remain independently instantiable.
"""
function infer_procedural(
    term::Union{AstNode,AstTree},
    ctx::AbstractDict{Symbol,<:TyLike};
    factory=make_tyvar_factory(),
    meta=nothing,
    fixed_point=fixed_point_witness,
)
    root = term isa AstTree ? term.root : term
    if fixed_point !== fixed_point_witness
        native = _infer_procedural_native(root, ctx, factory, nothing)
        native isa Err && return native
        lowered = _infer_procedural_lowered(root, ctx, factory, nothing, fixed_point)
        lowered isa Err && return lowered
        compatibility = TypeSystem.unify!(native, lowered)
        compatibility isa Err && return compatibility
        if meta !== nothing
            metadata_result = _infer_procedural_native(root, ctx, factory, meta)
            metadata_result isa Err && return metadata_result
        end
        return TypeSystem.prune(lowered)
    end
    return _infer_procedural_native(root, ctx, factory, meta)
end

"""
    reduce_procedural!(tree, args...; fixed_point=fixed_point_witness) -> Bool

Destructively lower `tree`, then delegate to ordinary normal-order beta reduction.
Additional positional arguments retain the `beta_reduce!` meaning (step budget and,
optionally, runtime constant context).
"""
function reduce_procedural!(
    tree::AstTree,
    args...;
    fixed_point=fixed_point_witness,
)
    tree.root, _ = _lower_with_context(tree.root, fixed_point)
    return Reduces.beta_reduce!(tree, args...)
end
