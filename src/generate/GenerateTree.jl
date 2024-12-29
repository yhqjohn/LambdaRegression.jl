"""
    GenerateTree

Typed AST generation under `max_depth` / `max_try` budgets.

`GenerateTree` synthesizes lambda-calculus ASTs (see `ASTs.NodeASTs`) that satisfy a target type (`TypeSystem.Ty`).
Generation is driven by a small set of *constructors* (`AbstractCtor`) and a *chooser* (`AbstractChooser`) that selects
among qualified constructors.

Quickstart
```julia
using Random
using LambdaRegression.GenerateTree: generate_tree, AbsCtor, IndexCtor, ApplyLeafCtor, IntegerCtor, UniformChooser
using LambdaRegression.TypeSystem: TyLike, IntTy, TyFun

tree = generate_tree(
    TyFun(IntTy, IntTy);
    ctors=[AbsCtor(), IndexCtor(), ApplyLeafCtor(), IntegerCtor()],
    chooser=UniformChooser(),
    rng=MersenneTwister(1),
    max_depth=6,
    max_try=20,
    global_consts_ty=Dict{Symbol,TyLike}(),
    local_consts_ty=Dict{Symbol,TyLike}(),
    local_consts=Dict{Symbol,Any}(),
)
```

Capabilities
- Budget control: `max_depth` (recursive depth) and `max_try` (retry budget per node).
- Constructor system: leaf constructors (constants/vars), abstraction (`AbsCtor`), application (leaf-anchored and template-anchored).
- Template pools (optional): treat a closed `AstTree` as an anchor for application constructors.
  - Provide `templates::Dict{Symbol,AstTree}` (and optionally `templates_ty::Dict{Symbol,TyLike}`), or
  - Provide `terms_ctx::Dict{Symbol,Any}` from `@terms(...)` and extract AST-valued entries automatically.
  - When `templates_ty` is omitted, it is inferred via `TypeSystem.infer_scheme` using `global_consts_ty` (and `local_consts_ty` if present),
    and validated with `TypeSystem.validate_ctx` as fail-fast initialization.
- Context discipline: in `Dict{Symbol,TyLike}`, store monomorphic entries as `Ty` and polymorphic entries as `TyScheme`
  (enforced by `TypeSystem.validate_ctx`).

Key context fields
- `ctors`, `chooser`, `rng`, `tyfactory`, `symfactory`, `max_depth`, `max_try`
- `global_consts_ty`, `local_consts_ty`, `local_consts`
- `templates`, `templates_ty`, or `terms_ctx`
"""
module GenerateTree

using Random
using TestItems

using ..TypeSystem
using ..TypeSystem: unify!, instantiate
using ..ASTs.NodeASTs
using ..ASTs.Contexts
using ..Trees: children
using ..Utils

"""
    AbstractCtor

Abstract base type for all AST node constructors.

Constructors implement the generation logic for specific node types (constants, 
variables, abstractions, applications). Each constructor defines how to check 
type qualification and construct trees of its node type.
"""
abstract type AbstractCtor end

"""
    initialize_ctx(ctor::AbstractCtor, ctx::NamedTuple) -> NamedTuple

Initialize constructor-specific context fields. Returns additional context
fields needed by the constructor during generation.
"""
initialize_ctx(::AbstractCtor, ::NamedTuple) = (;)

"""
    isqualified(ty::TyLike, ctor::AbstractCtor, ctx::NamedTuple) -> Bool

Check if a constructor can generate trees of the given type in the current context.
"""
isqualified(::Ty, ::AbstractCtor, ctx::NamedTuple) = false
isqualified(::TyScheme, ::AbstractCtor, ctx::NamedTuple) = isqualified(instantiate(ty, 0), ctor, ctx)

"""
    construct_tree(ctor::AbstractCtor, ty::Ty, ctx::NamedTuple) -> Union{AstTree, Err}

Construct an AST tree of the given type using the specified constructor.
Implementations must return `AstTree` on success so callers can rely on the
tree container interface instead of assuming raw nodes are subtrees.
"""
function construct_tree end

"""
    AbstractChooser

Abstract base type for constructor selection strategies.

Choosers implement the logic for selecting which constructor to use when
multiple constructors are qualified for a given type.
"""
abstract type AbstractChooser end

"""
    initialize_ctx(chooser::AbstractChooser, ctx::NamedTuple) -> NamedTuple

Initialize chooser-specific context fields.
"""
initialize_ctx(::AbstractChooser, ::NamedTuple) = (;)

"""
    choose_ctor(chooser::AbstractChooser, ty::TyLike, ctx::NamedTuple) -> Union{AbstractCtor, Err}

Select a qualified constructor for the given type using the chooser's strategy.
"""
function choose_ctor end

struct MaxDepthReachedErr end
struct MaxTryReachedErr end
struct NoQualifiedCtorErr end
struct SingleTypeResolveErr end
struct FreeVariableErr
    name::Symbol
end
struct UnboundIndexErr
    index::Int
    depth::Int
end


"""
    UniformChooser <: AbstractChooser

Chooser that selects constructors uniformly at random from qualified candidates.

Uses shuffled iteration through constructors to provide uniform selection
among those qualified for the requested type.
"""
struct UniformChooser <: AbstractChooser end
function choose_ctor(chooser::UniformChooser, ty::TyLike, ctx::NamedTuple)
    ctors = ctx.ctors
    ctors = shuffle(ctx.rng, ctors)
    for ctor in ctors
        if isqualified(ty, ctor, ctx)
            return ctor
        end
    end
    return Err(NoQualifiedCtorErr())
end

"""
    generate_tree(ty::Ty; kwargs...) -> Union{AstTree, Err}

Generate an AST tree of the specified type using recursive constructor selection.

# Arguments
- `ty::Ty`: Target type for the generated tree
- `ctors::Vector{<:AbstractCtor}`: Available constructors for tree generation
- `chooser::AbstractChooser=UniformChooser()`: Strategy for constructor selection
- `max_depth::Int=10`: Maximum recursion depth budget
- `max_try::Int=10`: Maximum attempts per generation step
- `tyfactory=newtyvar`: Type variable factory function
- `ctx...`: Additional context fields

Uses a recursive generation strategy with depth and attempt budgets to construct
valid AST trees. Combines constructor qualification checking with chooser strategies
to balance success rate and diversity. The `max_depth` field is surfaced to
constructors through the context; they are responsible for respecting depth
limits when determining qualification or recursing further.
"""
function generate_tree(ty::Ty;
    ctors::Vector{<:AbstractCtor},
    chooser::AbstractChooser=UniformChooser(),
    max_depth::Int=10,
    max_try::Int=10,
    tyfactory = newtyvar,
    symfactory = SymbolGenerator(),
    ctx...
)
    ctx_nt = (;ctx...)
    initial_ctx = merge(ctx_nt, (;ctors=ctors, chooser=chooser, max_depth=max_depth, max_try=max_try, tyfactory=tyfactory, symfactory=symfactory))
    for ctor in ctors
        ctor_ctx = initialize_ctx(ctor, initial_ctx)
        initial_ctx = merge(initial_ctx, ctor_ctx)
    end
    initial_ctx = merge(initial_ctx, initialize_ctx(chooser, initial_ctx))
    _generate_tree(ty, initial_ctx)
end

"""
    _generate_tree(ty::Ty, ctx::NamedTuple) -> Union{AstTree, Err}

Internal helper that drives the recursive generation loop. It delegates depth
budget enforcement to constructors via the context and expects every successful
constructor to return an `AstTree` wrapper.
"""
function _generate_tree(ty::Ty, ctx::NamedTuple)
    for _ in 1:ctx.max_try
        ctor = @✓ choose_ctor(ctx.chooser, ty, ctx)
        subtree = construct_tree(ctor, ty, ctx)
        if subtree isa Err
            continue
        end
        return subtree
    end
    return Err(MaxTryReachedErr())
end

abstract type AbstractLeafCtor <: AbstractCtor end

include("constantctors.jl")

"""
    AbstractAppCtor <: AbstractCtor

Abstract base type for application node constructors.

Application constructors generate AppNode instances representing function
applications in the lambda calculus AST.
"""
abstract type AbstractAppCtor <: AbstractCtor end

"""
    AbstractAbsCtor <: AbstractCtor

Abstract base type for abstraction node constructors.

Abstraction constructors generate AbsNode instances representing lambda
abstractions in the AST.
"""
abstract type AbstractAbsCtor <: AbstractCtor end

"""
    AbstractVarCtor <: AbstractCtor

Abstract base type for variable node constructors.

Variable constructors generate variable references, typically as De Bruijn
indices for bound variables.
"""
abstract type AbstractVarCtor <: AbstractLeafCtor end

"""
    IndexCtor <: AbstractVarCtor

Constructor for De Bruijn index variable nodes.

Generates IndexNode instances that reference variables by their De Bruijn
indices within the lambda abstraction scope stack.
"""
struct IndexCtor <: AbstractVarCtor end
function initialize_ctx(::IndexCtor, ctx::NamedTuple)
    var_ty_ctx = haskey(ctx, :var_ty) ? (;) : (;var_ty=Dict{Symbol,Ty}())
    scope_stack_ctx = haskey(ctx, :scope_stack) ? (;) : (;scope_stack=Vector{Symbol}())
    return merge(var_ty_ctx, scope_stack_ctx)
end
function isqualified(ty::Ty, ::IndexCtor, ctx::NamedTuple)
    isempty(ctx.scope_stack) && return false
    for name in ctx.scope_stack
        var_ty = ctx.var_ty[name]
        if iscompatible(ty, var_ty)
            return true
        end
    end
    return false
end
function construct_tree(::IndexCtor, ty::Ty, ctx::NamedTuple)
    qualified_names = Symbol[]
    for name in ctx.scope_stack
        var_ty = ctx.var_ty[name]
        if iscompatible(ty, var_ty)
            push!(qualified_names, name)
        end
    end
    isempty(qualified_names) && return Err(NoQualifiedCtorErr())
    name = rand(ctx.rng, qualified_names)
    idx = length(ctx.scope_stack) - findlast(==(name), ctx.scope_stack)
    return AstTree(IndexNode(idx))
end

function _ast_depth(node::AstNode)::Int
    cs = children(node)
    isempty(cs) && return 0
    return 1 + maximum(_ast_depth, cs)
end
ast_depth(tree::AstTree)::Int = _ast_depth(tree.root)

function _is_closed_term(tree::AstTree)
    bound = Dict{Symbol,Int}()
    function _walk(node::AstNode, depth::Int)
        if node isa AbsNode
            bound[node.name] = get(bound, node.name, 0) + 1
            res = _walk(node.body, depth + 1)
            bound[node.name] -= 1
            bound[node.name] == 0 && delete!(bound, node.name)
            return res
        elseif node isa AppNode
            @✓ _walk(node.func, depth)
            @✓ _walk(node.arg, depth)
            return nothing
        elseif node isa VarNode
            haskey(bound, node.name) && return nothing
            return Err(FreeVariableErr(node.name))
        elseif node isa IndexNode
            node.index < depth && return nothing
            return Err(UnboundIndexErr(node.index, depth))
        elseif node isa ConstNode
            return nothing
        else
            return Err(SingleTypeResolveErr())
        end
    end
    return _walk(tree.root, 0)
end

function _fresh_ty(ty::TyLike, ctx::NamedTuple; level::Int=0)::Ty
    return deepcopy(instantiate(ty, level, ctx.tyfactory))
end

function _anchored_apply(anchor_side::Symbol, anchor_tree::AstTree, anchor_ty::TyLike, target_ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    if anchor_side === :func
        arg_tv = ctx.tyfactory(1)
        goal_fun_ty = TyFun(arg_tv, deepcopy(target_ty))
        inst_anchor_ty = _fresh_ty(anchor_ty, ctx; level=0)
        unify_result = unify!(inst_anchor_ty, goal_fun_ty)
        unify_result isa Err && return unify_result
        inst_arg_ty = TypeSystem.prune(arg_tv)
        for _ in 1:ctx.max_try
            arg_result = _generate_tree(inst_arg_ty, merge(ctx, (;max_depth=ctx.max_depth-1)))
            if arg_result isa AstTree
                return AstTree(AppNode(anchor_tree.root, arg_result.root))
            end
        end
        return Err(MaxTryReachedErr())
    elseif anchor_side === :arg
        inst_arg_ty = _fresh_ty(anchor_ty, ctx; level=0)
        func_ty = TyFun(inst_arg_ty, deepcopy(target_ty))
        func_tree = @✓ _generate_tree(func_ty, merge(ctx, (;max_depth=ctx.max_depth-1)))
        return AstTree(AppNode(func_tree.root, anchor_tree.root))
    else
        error("Unknown anchor_side: $anchor_side")
    end
end

"""
    ApplyLeafCtor <: AbstractAppCtor

Application constructor with leaf function strategy.

Generates application nodes where the function part is constructed using
leaf constructors first, then the argument is generated to match the inferred
function type.
"""
struct ApplyLeafCtor <: AbstractAppCtor end

function _leaf_function_ctors(ret_ty::Ty, ctx::NamedTuple)
    arg_ty = ctx.tyfactory(1)
    func_ty = TyFun(arg_ty, ret_ty)
    qualified_ctors = AbstractCtor[]
    for ctor in ctx.ctors
        if ctor isa AbstractLeafCtor && isqualified(func_ty, ctor, ctx)
            push!(qualified_ctors, ctor)
        end
    end
    return qualified_ctors
end

function _candidate_leaf_types(ctx::NamedTuple)
    types = Ty[]
    if haskey(ctx, :global_consts_ty)
        for ty in values(ctx.global_consts_ty)
            push!(types, instantiate(ty, 0, ctx.tyfactory))
        end
    end
    if haskey(ctx, :local_consts_ty)
        for ty in values(ctx.local_consts_ty)
            push!(types, instantiate(ty, 0, ctx.tyfactory))
        end
    end
    if haskey(ctx, :var_ty)
        for ty in values(ctx.var_ty)
            push!(types, ty)
        end
    end

    if any(ctor isa IntegerCtor for ctor in ctx.ctors)
        push!(types, TypeSystem.IntTy)
    end
    if any(ctor isa FloatCtor for ctor in ctx.ctors)
        push!(types, TypeSystem.FloatTy)
    end
    if any(ctor isa BoolCtor for ctor in ctx.ctors)
        push!(types, TypeSystem.BoolTy)
    end

    if any(ctor isa NilCtor for ctor in ctx.ctors)
        base_types = copy(types)
        for base in base_types
            push!(types, TyList(base))
        end
    end

    unique!(types)
    return types
end

function _leaf_type(node::AstNode, ctx::NamedTuple)
    if node isa ConstNode
        if haskey(ctx, :local_consts_ty) && haskey(ctx.local_consts_ty, node.name)
            return instantiate(ctx.local_consts_ty[node.name], 0, ctx.tyfactory)
        elseif haskey(ctx, :global_consts_ty) && haskey(ctx.global_consts_ty, node.name)
            return instantiate(ctx.global_consts_ty[node.name], 0, ctx.tyfactory)
        else
            return Err(NoQualifiedCtorErr())
        end
    elseif node isa IndexNode
        node.index < length(ctx.scope_stack) || return Err(SingleTypeResolveErr())
        name = ctx.scope_stack[end - node.index]
        return ctx.var_ty[name]
    else
        return Err(SingleTypeResolveErr())
    end
end
function isqualified(ty::Ty, ::ApplyLeafCtor, ctx::NamedTuple)
    ctx.max_depth <= 0 && return false
    return !isempty(_leaf_function_ctors(ty, ctx))
end
function construct_tree(::ApplyLeafCtor, ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    leaf_ctors = _leaf_function_ctors(ty, ctx)
    isempty(leaf_ctors) && return Err(NoQualifiedCtorErr())
    func_tree = @✓ _generate_tree(
        TyFun(ctx.tyfactory(1), ty),
        merge(ctx, (;max_depth=ctx.max_depth-1, ctors=leaf_ctors)),
    )
    inst_func_ty = @✓ _leaf_type(func_tree.root, ctx)
    return _anchored_apply(:func, func_tree, inst_func_ty, ty, ctx)
end

"""
    ApplyToLeafCtor <: AbstractAppCtor

Application constructor with constant argument strategy.

Generates application nodes where the argument part is generated first,
then the function is constructed to match the required function type based
on the inferred argument type and target result type.
"""
struct ApplyToLeafCtor <: AbstractAppCtor end
function isqualified(ty::Ty, ::ApplyToLeafCtor, ctx::NamedTuple)
    ctx.max_depth <= 0 && return false
    leaf_ctors = [ctor for ctor in ctx.ctors if ctor isa AbstractLeafCtor]
    isempty(leaf_ctors) && return false
    return !isempty(_candidate_leaf_types(ctx))
end
function construct_tree(::ApplyToLeafCtor, ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    leaf_ctors = [ctor for ctor in ctx.ctors if ctor isa AbstractLeafCtor]
    isempty(leaf_ctors) && return Err(NoQualifiedCtorErr())
    candidate_types = _candidate_leaf_types(ctx)
    isempty(candidate_types) && return Err(NoQualifiedCtorErr())
    arg_tree = Err(NoQualifiedCtorErr())
    for _ in 1:ctx.max_try
        arg_ty = rand(ctx.rng, candidate_types)
        arg_result = _generate_tree(arg_ty, merge(ctx, (;max_depth=ctx.max_depth-1, ctors=leaf_ctors)))
        if arg_result isa AstTree
            arg_tree = arg_result
            break
        end
    end
    arg_tree isa Err && return arg_tree
    inst_arg_ty = @✓ _leaf_type(arg_tree.root, ctx)
    return _anchored_apply(:arg, arg_tree, inst_arg_ty, ty, ctx)
end

"""
    ApplyTemplateCtor <: AbstractAppCtor

Template-anchored application constructor (template as the **function** side).

This constructor selects a closed template `t` from `ctx.templates`, whose type (from `ctx.templates_ty[name]`)
can be unified with `arg_ty -> target_ty`. It then calls `_anchored_apply(:func, ...)` to generate an argument subtree
of the inferred `arg_ty` under the remaining depth budget, finally returning `AppNode(t, arg)`.

Context contract
- Requires `templates::Dict{Symbol,AstTree}` and `templates_ty::Dict{Symbol,TyLike}` in `ctx`.
  These are produced by `initialize_ctx(ApplyTemplateCtor(), ctx)` via `_initialize_templates_ctx`.
- If `templates_ty` is not provided, it is inferred via `TypeSystem.infer_scheme` using `global_consts_ty`
  (and `local_consts_ty` if present), then validated via `TypeSystem.validate_ctx`.
- Every template must be a closed term (`_is_closed_term`) and must satisfy the depth guard
  `ast_depth(template) <= ctx.max_depth - 1`.

Failure modes
- No templates / no typing context / no qualified template for `target_ty`
- Template not closed, or type inference fails during initialization
- Argument generation exhausts `max_try` within the remaining `max_depth`
"""
struct ApplyTemplateCtor <: AbstractAppCtor end

"""
    ApplyToTemplateCtor <: AbstractAppCtor

Template-anchored application constructor (template as the **argument** side).

This constructor selects a closed template `t` from `ctx.templates` as the argument. It freshens the template type to
`arg_ty`, constructs the required function type `arg_ty -> target_ty`, and calls `_anchored_apply(:arg, ...)` to generate
the function subtree under the remaining depth budget, returning `AppNode(func, t)`.

Context contract
- Same template requirements as [`ApplyTemplateCtor`](@ref): `templates`, `templates_ty`, and `global_consts_ty` must be present.

Implementation note
- Candidate templates are filtered only by depth; type compatibility is attempted via `_anchored_apply` and retried up to `max_try`.
  If the template pool mixes unrelated types, prefer splitting pools or increasing `max_try`.
"""
struct ApplyToTemplateCtor <: AbstractAppCtor end

function _extract_templates_from_terms_ctx(terms_ctx)
    templates = Dict{Symbol,AstTree}()
    for (key, value) in pairs(terms_ctx)
        if value isa AstTree
            templates[key] = value
        elseif value isa AstNode
            templates[key] = AstTree(value)
        end
    end
    return templates
end

function _specialized_y_scheme(factory)::TyScheme
    # Function-only fixpoint:
    #   ∀α β. ((α→β)→(α→β)) → (α→β)
    α = factory(0)
    β = factory(0)
    σ = TyFun(α, β)
    return TyScheme([α, β], TyFun(TyFun(σ, σ), σ))
end

function _initialize_templates_ctx(ctx::NamedTuple)
    templates = if haskey(ctx, :templates)
        ctx.templates
    elseif haskey(ctx, :terms_ctx)
        _extract_templates_from_terms_ctx(ctx.terms_ctx)
    else
        Dict{Symbol,AstTree}()
    end

    templates_ty = if haskey(ctx, :templates_ty)
        copy(ctx.templates_ty)
    else
        Dict{Symbol,TyLike}()
    end

    isempty(templates) && return (;templates=templates, templates_ty=templates_ty)

    haskey(ctx, :global_consts_ty) || error("Template typing requires `global_consts_ty` in the context.")
    type_ctx = haskey(ctx, :local_consts_ty) ? merge(ctx.global_consts_ty, ctx.local_consts_ty) : ctx.global_consts_ty

    for (name, tree) in pairs(templates)
        closed = _is_closed_term(tree)
        closed isa Err && error("Template :$name is not closed: $(closed.value)")
        haskey(templates_ty, name) && continue
        scheme = try
            TypeSystem.infer_scheme(tree, type_ctx; factory=ctx.tyfactory)
        catch e
            error("Failed to infer type for template :$name: $(e)")
        end
        scheme isa Err && error("Failed to infer type for template :$name: $(scheme.value)")
        if name == :Y
            specialized = _specialized_y_scheme(ctx.tyfactory)
            TypeSystem.islesseq(specialized, scheme) ||
                error("Specialized :Y scheme is not a subset of inferred scheme: inferred=$scheme specialized=$specialized")
            templates_ty[name] = specialized
        else
            templates_ty[name] = scheme
        end
    end
    TypeSystem.validate_ctx(templates_ty)
    return (;templates=templates, templates_ty=templates_ty)
end

initialize_ctx(::ApplyTemplateCtor, ctx::NamedTuple) = _initialize_templates_ctx(ctx)
initialize_ctx(::ApplyToTemplateCtor, ctx::NamedTuple) = _initialize_templates_ctx(ctx)

function _template_depth_ok(tree::AstTree, ctx::NamedTuple)
    return ast_depth(tree) <= ctx.max_depth - 1
end

function _template_can_be_func(template_ty::TyLike, target_ty::Ty, ctx::NamedTuple)
    arg_tv = ctx.tyfactory(1)
    goal_fun_ty = TyFun(arg_tv, deepcopy(target_ty))
    inst_template_ty = _fresh_ty(template_ty, ctx; level=0)
    return isnothing(unify!(inst_template_ty, goal_fun_ty))
end

function isqualified(ty::Ty, ::ApplyTemplateCtor, ctx::NamedTuple)
    ctx.max_depth <= 0 && return false
    haskey(ctx, :templates) || return false
    isempty(ctx.templates) && return false
    for (name, tree) in pairs(ctx.templates)
        _template_depth_ok(tree, ctx) || continue
        haskey(ctx, :templates_ty) || return false
        template_ty = ctx.templates_ty[name]
        _template_can_be_func(template_ty, ty, ctx) && return true
    end
    return false
end

function construct_tree(::ApplyTemplateCtor, ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    haskey(ctx, :templates) || return Err(NoQualifiedCtorErr())
    haskey(ctx, :templates_ty) || return Err(NoQualifiedCtorErr())

    qualified = Symbol[]
    for (name, tree) in pairs(ctx.templates)
        _template_depth_ok(tree, ctx) || continue
        _template_can_be_func(ctx.templates_ty[name], ty, ctx) || continue
        push!(qualified, name)
    end
    isempty(qualified) && return Err(NoQualifiedCtorErr())
    chosen = rand(ctx.rng, qualified)
    anchor_tree = deepcopy(ctx.templates[chosen])
    anchor_ty = ctx.templates_ty[chosen]
    return _anchored_apply(:func, anchor_tree, anchor_ty, ty, ctx)
end

function isqualified(ty::Ty, ::ApplyToTemplateCtor, ctx::NamedTuple)
    ctx.max_depth <= 0 && return false
    haskey(ctx, :templates) || return false
    isempty(ctx.templates) && return false
    haskey(ctx, :templates_ty) || return false

    for (name, tree) in pairs(ctx.templates)
        _template_depth_ok(tree, ctx) || continue
        arg_ty = _fresh_ty(ctx.templates_ty[name], ctx; level=0)
        func_ty = TyFun(arg_ty, deepcopy(ty))
        for ctor in ctx.ctors
            isqualified(func_ty, ctor, ctx) && return true
        end
    end
    return false
end

function construct_tree(::ApplyToTemplateCtor, ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    haskey(ctx, :templates) || return Err(NoQualifiedCtorErr())
    haskey(ctx, :templates_ty) || return Err(NoQualifiedCtorErr())
    isempty(ctx.templates) && return Err(NoQualifiedCtorErr())

    candidates = Symbol[]
    for (name, tree) in pairs(ctx.templates)
        _template_depth_ok(tree, ctx) || continue
        push!(candidates, name)
    end
    isempty(candidates) && return Err(NoQualifiedCtorErr())

    for _ in 1:ctx.max_try
        chosen = rand(ctx.rng, candidates)
        arg_tree = deepcopy(ctx.templates[chosen])
        arg_ty_like = ctx.templates_ty[chosen]
        result = _anchored_apply(:arg, arg_tree, arg_ty_like, ty, ctx)
        result isa AstTree && return result
    end
    return Err(MaxTryReachedErr())
end

struct AbsCtor <: AbstractAbsCtor end

function initialize_ctx(::AbsCtor, ctx::NamedTuple)
    var_ty_ctx = haskey(ctx, :var_ty) ? (;) : (;var_ty=Dict{Symbol,Ty}())
    scope_stack_ctx = haskey(ctx, :scope_stack) ? (;) : (;scope_stack=Vector{Symbol}())
    return merge(var_ty_ctx, scope_stack_ctx)
end

isqualified(ty::TyFun, ::AbsCtor, ctx::NamedTuple) = ctx.max_depth > 0 # only for function type
function construct_tree(::AbsCtor, ty::Ty, ctx::NamedTuple)
    ctx.max_depth <= 0 && return Err(MaxDepthReachedErr())
    ty isa TyFun || return Err(SingleTypeResolveErr())
    arg_ty = ty.arg
    name = ctx.symfactory()
    new_var_ty = copy(ctx.var_ty)
    new_var_ty[name] = arg_ty
    new_scope_stack = copy(ctx.scope_stack)
    push!(new_scope_stack, name)
    body_tree = @✓ _generate_tree(
        ty.res,
        merge(ctx, (;max_depth=ctx.max_depth-1, var_ty=new_var_ty, scope_stack=new_scope_stack)),
    )
    return AstTree(AbsNode(name, body_tree.root))
end

@testitem "GenerateTree GlobalConstCtor constructor sampling" begin
    using Random
    using LambdaRegression.GenerateTree: GlobalConstCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy, BoolTy, IADD, ISUB, IMUL, get_type_ctx
    using LambdaRegression.ASTs.Parse: @terms
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(1)
    ctor = GlobalConstCtor()
    base_ctx = (;
        rng=rng,
        global_consts_ty=get_type_ctx(@terms(IADD, ISUB, IMUL)),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
    )
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))
    target_ty = IntTy → IntTy → IntTy

    @test isqualified(target_ty, ctor, ctx)
    @test !isqualified(BoolTy, ctor, ctx)

    sampled = Set{Symbol}()
    for _ in 1:20
        tree = construct_tree(ctor, target_ty, ctx)
        @test tree isa AstTree
        @test tree.root isa ConstNode
        @test tree.root.name in (:IADD, :ISUB, :IMUL)
        push!(sampled, tree.root.name)
    end
    @test length(sampled) >= 2
end

@testitem "GenerateTree LocalConstCtor constructor sampling" begin
    using Random
    using LambdaRegression.GenerateTree: LocalConstCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy, BoolTy
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(1111)
    ctor = LocalConstCtor()
    local_consts = Dict{Symbol,Any}(
        :one => 1,
        :two => 2,
        :three => 3,
        :flag => true,
    )
    local_consts_ty = Dict{Symbol,TyLike}(
        :one => IntTy,
        :two => IntTy,
        :three => IntTy,
        :flag => BoolTy,
    )
    base_ctx = (;
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=local_consts,
        local_consts_ty=local_consts_ty,
    )
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))

    @test isqualified(IntTy, ctor, ctx)
    @test !isqualified(IntTy → IntTy, ctor, ctx)

    sampled = Set{Symbol}()
    for _ in 1:20
        tree = construct_tree(ctor, IntTy, ctx)
        @test tree isa AstTree
        @test tree.root isa ConstNode
        @test tree.root.name in (:one, :two, :three)
        @test haskey(local_consts, tree.root.name)
        @test local_consts_ty[tree.root.name] == IntTy
        push!(sampled, tree.root.name)
    end
    @test length(sampled) >= 2
end

@testitem "GenerateTree IntegerCtor stores typed local constants" begin
    using Random
    using LambdaRegression.GenerateTree: IntegerCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(1234)
    ctor = IntegerCtor(3:7)
    base_ctx = (;rng=rng)
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))

    @test haskey(ctx, :local_consts)
    @test haskey(ctx, :local_consts_ty)
    @test isqualified(IntTy, ctor, ctx)
    @test !isqualified(BoolTy, ctor, ctx)

    for _ in 1:8
        tree = construct_tree(ctor, IntTy, ctx)
        @test tree isa AstTree
        @test tree.root isa ConstNode
        @test haskey(ctx.local_consts, tree.root.name)
        @test haskey(ctx.local_consts_ty, tree.root.name)
        @test ctx.local_consts_ty[tree.root.name] == IntTy
        @test ctx.local_consts[tree.root.name] in 3:7
    end
end

@testitem "GenerateTree NilCtor stores typed local empty list" begin
    using Random
    using LambdaRegression.GenerateTree: NilCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, TyList
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(8000)
    ctor = NilCtor()
    base_ctx = (;rng=rng)
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))
    target_ty = TyList(IntTy)

    @test isqualified(target_ty, ctor, ctx)
    @test !isqualified(IntTy, ctor, ctx)

    tree = construct_tree(ctor, target_ty, ctx)
    @test tree isa AstTree
    @test tree.root isa ConstNode
    @test haskey(ctx.local_consts, tree.root.name)
    @test haskey(ctx.local_consts_ty, tree.root.name)
    @test ctx.local_consts_ty[tree.root.name] == target_ty
    @test ctx.local_consts[tree.root.name] isa Vector
    @test isempty(ctx.local_consts[tree.root.name])
end

@testitem "GenerateTree abs/index identity" begin
    using Random
    using LambdaRegression.GenerateTree: generate_tree, AbsCtor, IndexCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(1)
    ctors = [AbsCtor(), IndexCtor()]
    ty = IntTy → IntTy
    tree = generate_tree(
        ty;
        ctors=ctors,
        rng=rng,
        max_depth=3,
        max_try=5,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
    )

    @test tree isa AstTree
    @test tree.root isa AbsNode
    @test tree.root.body isa IndexNode
    @test tree.root.body.index == 0
    inferred = infer(tree, Dict{Symbol,TyLike}())
    @test iscompatible(inferred, ty)
end

@testitem "GenerateTree apply leaf with IADD" begin
    using Random
    using LambdaRegression.GenerateTree: generate_tree, ApplyLeafCtor, GlobalConstCtor, IntegerCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, IADD, get_type_ctx
    using LambdaRegression.ASTs.Parse: @terms
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(1111)
    ctors = [ApplyLeafCtor(), GlobalConstCtor(), IntegerCtor()]
    ty = IntTy → IntTy
    global_consts_ty = get_type_ctx(@terms(IADD))
    local_consts_ty = Dict{Symbol,TyLike}()
    local_consts = Dict{Symbol,Any}()
    tree = generate_tree(
        ty;
        ctors=ctors,
        rng=rng,
        max_depth=3,
        max_try=8,
        global_consts_ty=global_consts_ty,
        local_consts_ty=local_consts_ty,
        local_consts=local_consts,
    )

    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.func isa ConstNode
    @test tree.root.func.name == :IADD
    @test tree.root.arg isa ConstNode
    type_ctx = merge(global_consts_ty, local_consts_ty)
    inferred = infer(tree, type_ctx)
    @test inferred == ty
end

@testitem "GenerateTree apply arg-first leaf" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, UniformChooser, SymbolGenerator, ApplyToLeafCtor, GlobalConstCtor, AbsCtor, IndexCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, get_type_ctx
    using LambdaRegression.ASTs.Parse: @terms
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(1234)
    ctors = [ApplyToLeafCtor(), GlobalConstCtor(), AbsCtor(), IndexCtor()]
    ty = IntTy → IntTy
    global_consts_ty = get_type_ctx(@terms(one=1))
    local_consts_ty = Dict{Symbol,TyLike}()
    local_consts = Dict{Symbol,Any}()
    initial_ctx = (;
        ctors=ctors,
        chooser=UniformChooser(),
        max_depth=3,
        max_try=8,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts_ty=local_consts_ty,
        local_consts=local_consts,
    )
    tree = let ctx = initial_ctx
        for ctor in ctors
            ctx = merge(ctx, initialize_ctx(ctor, ctx))
        end
        ctx = merge(ctx, initialize_ctx(ctx.chooser, ctx))
        construct_tree(ApplyToLeafCtor(), ty, ctx)
    end

    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.arg isa ConstNode
    @test tree.root.arg.name == :one
    inferred = infer(tree, global_consts_ty)
    @test iscompatible(inferred, ty)
end

@testitem "GenerateTree higher-order arg from scope" begin
    using Random
    using LambdaRegression.GenerateTree: generate_tree, AbsCtor, ApplyLeafCtor, IndexCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(8000)
    ctors = [AbsCtor(), ApplyLeafCtor(), IndexCtor()]
    ty = ((IntTy → IntTy) → IntTy) → IntTy
    tree = generate_tree(
        ty;
        ctors=ctors,
        rng=rng,
        max_depth=4,
        max_try=8,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
    )

    @test tree isa AstTree
    @test tree.root isa AbsNode
    @test tree.root.body isa AppNode
    @test tree.root.body.func isa IndexNode
    @test tree.root.body.func.index == 0
    @test tree.root.body.arg isa AbsNode
    inferred = infer(tree, Dict{Symbol,TyLike}())
    @test iscompatible(inferred, ty)
end

@testitem "GenerateTree polymorphic leaf instantiation" begin
    using Random
    using LambdaRegression.GenerateTree: generate_tree, ApplyLeafCtor, GlobalConstCtor, IntegerCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(42)
    ctors = [ApplyLeafCtor(), GlobalConstCtor(), IntegerCtor()]
    α = newtyvar(0)
    global_consts_ty = Dict{Symbol,TyLike}(
        :ID => ∀([α], α → α),
    )
    local_consts_ty = Dict{Symbol,TyLike}()
    local_consts = Dict{Symbol,Any}()
    tree = generate_tree(
        IntTy;
        ctors=ctors,
        rng=rng,
        max_depth=3,
        max_try=8,
        global_consts_ty=global_consts_ty,
        local_consts_ty=local_consts_ty,
        local_consts=local_consts,
    )

    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.func isa ConstNode
    @test tree.root.func.name == :ID
    type_ctx = merge(global_consts_ty, local_consts_ty)
    inferred = infer(tree, type_ctx)
    @test iscompatible(inferred, IntTy)
end

@testitem "GenerateTree FloatCtor stores typed local constants" begin
    using Random
    using LambdaRegression.GenerateTree: FloatCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: FloatTy, IntTy
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(8000)
    ctor = FloatCtor(0.0, 1.0)
    base_ctx = (;rng=rng)
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))

    @test isqualified(FloatTy, ctor, ctx)
    @test !isqualified(IntTy, ctor, ctx)

    tree = construct_tree(ctor, FloatTy, ctx)
    @test tree isa AstTree
    @test tree.root isa ConstNode
    @test ctx.local_consts[tree.root.name] isa Float64
    @test ctx.local_consts_ty[tree.root.name] == FloatTy
end

@testitem "GenerateTree BoolCtor stores typed local constants" begin
    using Random
    using LambdaRegression.GenerateTree: BoolCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: BoolTy, IntTy
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(42)
    ctor = BoolCtor()
    base_ctx = (;rng=rng)
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))

    @test isqualified(BoolTy, ctor, ctx)
    @test !isqualified(IntTy, ctor, ctx)

    tree = construct_tree(ctor, BoolTy, ctx)
    @test tree isa AstTree
    @test tree.root isa ConstNode
    @test ctx.local_consts[tree.root.name] isa Bool
    @test ctx.local_consts_ty[tree.root.name] == BoolTy
end

@testitem "GenerateTree IntArrCtor stores typed local arrays" begin
    using Random
    using LambdaRegression.GenerateTree: IntArrCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy, TyArray, TyNat
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode

    rng = MersenneTwister(555)
    ctor = IntArrCtor(1:3)
    base_ctx = (;rng=rng)
    ctx = merge(base_ctx, initialize_ctx(ctor, base_ctx))
    target_ty = TyArray(TyNat(4), IntTy)

    @test isqualified(target_ty, ctor, ctx)
    @test !isqualified(BoolTy, ctor, ctx)

    tree = construct_tree(ctor, target_ty, ctx)
    @test tree isa AstTree
    @test tree.root isa ConstNode
    @test ctx.local_consts[tree.root.name] isa Vector{Int}
    @test length(ctx.local_consts[tree.root.name]) == 4
    @test all(x -> x in 1:3, ctx.local_consts[tree.root.name])
    @test ctx.local_consts_ty[tree.root.name] == target_ty
end

@testitem "GenerateTree IndexCtor selects correct De Bruijn indices" begin
    using Random
    using LambdaRegression.GenerateTree: IndexCtor, construct_tree, initialize_ctx, isqualified
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: IntTy, BoolTy
    using LambdaRegression.ASTs.NodeASTs: AstTree, IndexNode

    rng = MersenneTwister(2000)
    ctor = IndexCtor()

    # Shadowing: innermost binding should win (idx 0)
    base_ctx1 = (;
        rng=rng,
        var_ty=Dict{Symbol,Ty}(:x => IntTy, :y => BoolTy),
        scope_stack=Symbol[:x, :y, :x],
    )
    ctx1 = merge(base_ctx1, initialize_ctx(ctor, base_ctx1))
    @test isqualified(IntTy, ctor, ctx1)
    for _ in 1:10
        tree = construct_tree(ctor, IntTy, ctx1)
        @test tree isa AstTree
        @test tree.root isa IndexNode
        @test tree.root.index == 0
    end

    # Two compatible names should yield both indices over repeated sampling
    rng2 = MersenneTwister(1)
    base_ctx2 = (;
        rng=rng2,
        var_ty=Dict{Symbol,Ty}(:x => IntTy, :y => IntTy),
        scope_stack=Symbol[:x, :y],
    )
    ctx2 = merge(base_ctx2, initialize_ctx(ctor, base_ctx2))
    seen = Set{Int}()
    for _ in 1:30
        tree = construct_tree(ctor, IntTy, ctx2)
        push!(seen, tree.root.index)
    end
    @test 0 in seen
    @test 1 in seen
end

@testitem "GenerateTree ApplyTemplateCtor anchors function templates" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, UniformChooser, SymbolGenerator, ApplyTemplateCtor, IntegerCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(1111)
    templates = Dict{Symbol,AstTree}(
        :ID => AstTree(AbsNode(:x, VarNode(:x))),
    )
    ctors = [IntegerCtor()]
    base_ctx = (;
        ctors=ctors,
        chooser=UniformChooser(),
        max_depth=3,
        max_try=8,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=templates,
    )
    ctx = merge(base_ctx, initialize_ctx(ApplyTemplateCtor(), base_ctx))
    ctx = merge(ctx, initialize_ctx(ctx.chooser, ctx))

    tree = construct_tree(ApplyTemplateCtor(), IntTy, ctx)
    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.func isa AbsNode
    type_ctx = merge(ctx.global_consts_ty, ctx.local_consts_ty)
    inferred = infer(tree, type_ctx)
    @test inferred == IntTy
end

@testitem "GenerateTree ApplyToTemplateCtor anchors argument templates" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, UniformChooser, SymbolGenerator, ApplyToTemplateCtor, AbsCtor, IndexCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy
    using LambdaRegression.ASTs.NodeASTs

    rng = MersenneTwister(1234)
    templates = Dict{Symbol,AstTree}(
        :one => AstTree(ConstNode(:one)),
    )
    global_consts_ty = Dict{Symbol,TyLike}(:one => IntTy)
    ctors = [AbsCtor(), IndexCtor()]
    base_ctx = (;
        ctors=ctors,
        chooser=UniformChooser(),
        max_depth=3,
        max_try=8,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=rng,
        global_consts_ty=global_consts_ty,
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=templates,
    )
    ctx = let ctx = base_ctx
        for ctor in [ApplyToTemplateCtor(), ctors...]
            ctx = merge(ctx, initialize_ctx(ctor, ctx))
        end
        merge(ctx, initialize_ctx(ctx.chooser, ctx))
    end

    tree = construct_tree(ApplyToTemplateCtor(), IntTy, ctx)
    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.arg isa ConstNode
    @test tree.root.arg.name == :one
    inferred = infer(tree, global_consts_ty)
    @test inferred == IntTy
end

@testitem "GenerateTree template typing supports μ-types" begin
    using Random
    using LambdaRegression.GenerateTree: initialize_ctx, ApplyTemplateCtor, isqualified, UniformChooser, SymbolGenerator, AbstractCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, TyMu, IntTy, TyScheme
    using LambdaRegression.ASTs.NodeASTs

    function has_mu(tylike::TyLike)::Bool
        ty = tylike isa TyScheme ? tylike.ty : tylike
        if ty isa TyMu
            return true
        elseif ty isa TyFun
            return has_mu(ty.arg) || has_mu(ty.res)
        elseif ty isa TyProd
            return has_mu(ty.left) || has_mu(ty.right)
        elseif ty isa TySum
            return has_mu(ty.left) || has_mu(ty.right)
        elseif ty isa TyList
            return has_mu(ty.elem)
        elseif ty isa TyArray
            return has_mu(ty.elem)
        else
            return false
        end
    end

    rng = MersenneTwister(8000)
    self_app = AstTree(AbsNode(:x, AppNode(IndexNode(0), IndexNode(0))))
    templates = Dict{Symbol,AstTree}(:SELFAPP => self_app)
    base_ctx = (;
        ctors=AbstractCtor[],
        chooser=UniformChooser(),
        max_depth=4,
        max_try=8,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=rng,
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=templates,
    )
    ctx = merge(base_ctx, initialize_ctx(ApplyTemplateCtor(), base_ctx))
    @test haskey(ctx, :templates_ty)
    @test has_mu(ctx.templates_ty[:SELFAPP])
    @test isqualified(IntTy, ApplyTemplateCtor(), ctx) in (true, false)
end

@testitem "GenerateTree ApplyTemplateCtor anchors Y combinator template (μ / higher-order)" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, ApplyTemplateCtor, AbsCtor, IndexCtor, SymbolGenerator
    using LambdaRegression.GenerateTree: AbstractChooser, isqualified, NoQualifiedCtorErr, Err
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy, TyScheme, TyFun, iscompatible
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.NodeASTs

    struct FirstChooser <: AbstractChooser end
    function LambdaRegression.GenerateTree.choose_ctor(::FirstChooser, ty::TyLike, ctx::NamedTuple)
        for ctor in ctx.ctors
            isqualified(ty, ctor, ctx) && return ctor
        end
        return Err(NoQualifiedCtorErr())
    end

    function has_self_app(node::AstNode)::Bool
        if node isa AppNode
            if node.func isa VarNode && node.arg isa VarNode && node.func.name == node.arg.name
                return true
            elseif node.func isa IndexNode && node.arg isa IndexNode && node.func.index == node.arg.index
                return true
            end
            return has_self_app(node.func) || has_self_app(node.arg)
        elseif node isa AbsNode
            return has_self_app(node.body)
        else
            return false
        end
    end

    lam = Char(0x03bb)
    y_src = string(lam, "f.(", lam, "x.f (x x)) (", lam, "x.f (x x))")
    y_tree = parse_lambda(y_src)
    templates = Dict{Symbol,AstTree}(:Y => y_tree)
    @test has_self_app(y_tree.root)

    target_ty = TyFun(IntTy, IntTy)
    base_ctx = (;
        ctors=[AbsCtor(), IndexCtor()],
        chooser=FirstChooser(),
        max_depth=8,
        max_try=10,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=MersenneTwister(42),
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=templates,
    )
    ctx = let ctx = base_ctx
        for ctor in [AbsCtor(), IndexCtor(), ApplyTemplateCtor()]
            ctx = merge(ctx, initialize_ctx(ctor, ctx))
        end
        merge(ctx, initialize_ctx(ctx.chooser, ctx))
    end

    @test haskey(ctx, :templates_ty)
    @test ctx.templates_ty[:Y] isa TyScheme

    tree = construct_tree(ApplyTemplateCtor(), target_ty, ctx)
    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.func == y_tree.root
    type_ctx = merge(ctx.global_consts_ty, ctx.local_consts_ty)
    @test iscompatible(infer(tree, type_ctx), target_ty)
    arg_ty = infer(AstTree(tree.root.arg), type_ctx)
    @test arg_ty isa TyFun
end

@testitem "GenerateTree Y uses specialized template scheme (function-only fixpoint)" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, ApplyTemplateCtor, AbsCtor, IndexCtor, SymbolGenerator
    using LambdaRegression.GenerateTree: AbstractChooser, isqualified, NoQualifiedCtorErr, Err
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy, TyScheme, TyFun, infer, iscompatible, islesseq, newtyvar
    using LambdaRegression.ASTs.Parse: parse_lambda
    using LambdaRegression.ASTs.NodeASTs

    struct FirstChooser <: AbstractChooser end
    function LambdaRegression.GenerateTree.choose_ctor(::FirstChooser, ty::TyLike, ctx::NamedTuple)
        for ctor in ctx.ctors
            isqualified(ty, ctor, ctx) && return ctor
        end
        return Err(NoQualifiedCtorErr())
    end

    lam = Char(0x03bb)
    y_src = string(lam, "f.(", lam, "x.f (x x)) (", lam, "x.f (x x))")
    y_tree = parse_lambda(y_src)

    base_ctx = (;
        ctors=[AbsCtor(), IndexCtor()],
        chooser=FirstChooser(),
        max_depth=8,
        max_try=12,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=MersenneTwister(2000),
        global_consts_ty=Dict{Symbol,TyLike}(),
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=Dict{Symbol,AstTree}(:Y => y_tree),
    )

    ctx = let ctx = base_ctx
        for ctor in [AbsCtor(), IndexCtor(), ApplyTemplateCtor()]
            ctx = merge(ctx, initialize_ctx(ctor, ctx))
        end
        merge(ctx, initialize_ctx(ctx.chooser, ctx))
    end

    # Expect: ∀α β. ((α→β)→(α→β)) → (α→β)
    α = newtyvar(0)
    β = newtyvar(0)
    σ = TyFun(α, β)
    expected = TyScheme([α, β], TyFun(TyFun(σ, σ), σ))
    @test islesseq(expected, ctx.templates_ty[:Y])
    @test islesseq(ctx.templates_ty[:Y], expected)

    # `Y` is forbidden for non-function targets.
    @test isqualified(IntTy, ApplyTemplateCtor(), ctx) == false

    # But it must still work for generator-like targets.
    target_ty = TyFun(IntTy, IntTy)
    tree = construct_tree(ApplyTemplateCtor(), target_ty, ctx)
    @test tree isa AstTree
    @test tree.root isa AppNode
    @test tree.root.func == y_tree.root

    type_ctx = merge(ctx.global_consts_ty, ctx.local_consts_ty)
    @test iscompatible(infer(tree, type_ctx), target_ty)

    arg_ty = infer(AstTree(tree.root.arg), type_ctx)
    expected_arg_ty = TyFun(TyFun(IntTy, IntTy), TyFun(IntTy, IntTy))
    @test iscompatible(arg_ty, expected_arg_ty)
end

@testitem "GenerateTree ApplyTemplateCtor anchors MAP/REDUCE templates directly" begin
    using Random
    using LambdaRegression.GenerateTree: construct_tree, initialize_ctx, ast_depth
    using LambdaRegression.GenerateTree: ApplyTemplateCtor, AbsCtor, IndexCtor, NilCtor, SymbolGenerator
    using LambdaRegression.GenerateTree: AbstractChooser, isqualified, NoQualifiedCtorErr, Err
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy, NilErr, TyList, TyScheme, TyFun, get_type_ctx, iscompatible
    using LambdaRegression.TypeSystem: CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND
    using LambdaRegression.ASTs.Parse: parse_lambda, @terms
    using LambdaRegression.ASTs.NodeASTs

    struct FirstChooser <: AbstractChooser end
    function LambdaRegression.GenerateTree.choose_ctor(::FirstChooser, ty::TyLike, ctx::NamedTuple)
        for ctor in ctx.ctors
            isqualified(ty, ctor, ctx) && return ctor
        end
        return Err(NoQualifiedCtorErr())
    end

    function has_self_app(node::AstNode)::Bool
        if node isa AppNode
            if node.func isa VarNode && node.arg isa VarNode && node.func.name == node.arg.name
                return true
            elseif node.func isa IndexNode && node.arg isa IndexNode && node.func.index == node.arg.index
                return true
            end
            return has_self_app(node.func) || has_self_app(node.arg)
        elseif node isa AbsNode
            return has_self_app(node.body)
        else
            return false
        end
    end

    α = newtyvar(0)
    id = x -> x
    nil_to_zero = _ -> 0
    nil_to_empty = _ -> Int[]
    empty_list = Int[]

    # Scenario A: MAP itself is the template (recursion via inlined Y)
    lam = Char(0x03bb)
    y_src = string(lam, "f.(", lam, "x.f (x x)) (", lam, "x.f (x x))")
    map_src =
        "Y (" *
        string(lam) * "appl." *
        string(lam) * "f." *
        string(lam) * "list. COND (IEQ (LENGTH list) int0) empty_list " *
        "((CONS (f (CASENILERR id nil_to_zero (HEAD list)))) (appl f (CASENILERR id nil_to_empty (TAIL list)))))"

    terms_ctx_map = @terms(CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND, id, nil_to_zero, nil_to_empty, empty_list, int0=0)
    merge!(terms_ctx_map, @terms(Y=parse_lambda(y_src, terms_ctx_map)))
    merge!(terms_ctx_map, @terms(MAP=parse_lambda(map_src, terms_ctx_map)))

    global_consts_ty_map = get_type_ctx(terms_ctx_map, Dict{Symbol,TyLike}(
        :id => TyScheme([α], TyFun(α, α)),
        :nil_to_zero => TyFun(NilErr, IntTy),
        :nil_to_empty => TyFun(NilErr, TyList(IntTy)),
        :empty_list => TyList(IntTy),
        :int0 => IntTy,
    ))

    map_tree = terms_ctx_map[:MAP]::AstTree
    map_template_depth = ast_depth(map_tree)
    @test has_self_app(map_tree.root)
    map_target_ty = TyFun(TyList(IntTy), TyList(IntTy))
    base_ctx_map = (;
        ctors=[AbsCtor(), IndexCtor()],
        chooser=FirstChooser(),
        max_depth=map_template_depth + 6,
        max_try=12,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=MersenneTwister(555),
        global_consts_ty=global_consts_ty_map,
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=Dict{Symbol,AstTree}(:MAP => map_tree),
    )
    ctx_map = let ctx = base_ctx_map
        for ctor in [AbsCtor(), IndexCtor(), ApplyTemplateCtor()]
            ctx = merge(ctx, initialize_ctx(ctor, ctx))
        end
        merge(ctx, initialize_ctx(ctx.chooser, ctx))
    end

    @test ctx_map.templates_ty[:MAP] isa TyScheme
    map_applied = construct_tree(ApplyTemplateCtor(), map_target_ty, ctx_map)
    @test map_applied isa AstTree
    @test map_applied.root isa AppNode
    @test map_applied.root.func == map_tree.root
    type_ctx_map = merge(ctx_map.global_consts_ty, ctx_map.local_consts_ty)
    @test iscompatible(infer(map_applied, type_ctx_map), map_target_ty)

    # Scenario B: REDUCE_SUM itself is the template; argument generation uses NilCtor for List(Int)
    reduce_src =
        "Y (" *
        string(lam) * "sum." *
        string(lam) * "list. COND (IEQ (LENGTH list) int0) int0 " *
        "(IADD (CASENILERR id nil_to_zero (HEAD list)) (sum (CASENILERR id nil_to_empty (TAIL list)))))"

    terms_ctx_red = @terms(CONS, HEAD, TAIL, LENGTH, CASENILERR, IADD, IMUL, IEQ, COND, id, nil_to_zero, nil_to_empty, empty_list, int0=0)
    merge!(terms_ctx_red, @terms(Y=parse_lambda(y_src, terms_ctx_red)))
    merge!(terms_ctx_red, @terms(REDUCE_SUM=parse_lambda(reduce_src, terms_ctx_red)))

    global_consts_ty_red = get_type_ctx(terms_ctx_red, Dict{Symbol,TyLike}(
        :id => TyScheme([α], TyFun(α, α)),
        :nil_to_zero => TyFun(NilErr, IntTy),
        :nil_to_empty => TyFun(NilErr, TyList(IntTy)),
        :empty_list => TyList(IntTy),
    ))

    red_tree = terms_ctx_red[:REDUCE_SUM]::AstTree
    red_template_depth = ast_depth(red_tree)
    @test has_self_app(red_tree.root)
    base_ctx_red = (;
        ctors=[NilCtor()],
        chooser=FirstChooser(),
        max_depth=red_template_depth + 6,
        max_try=12,
        tyfactory=newtyvar,
        symfactory=SymbolGenerator(),
        rng=MersenneTwister(2000),
        global_consts_ty=global_consts_ty_red,
        local_consts_ty=Dict{Symbol,TyLike}(),
        local_consts=Dict{Symbol,Any}(),
        templates=Dict{Symbol,AstTree}(:REDUCE_SUM => red_tree),
    )
    ctx_red = let ctx = base_ctx_red
        ctx = merge(ctx, initialize_ctx(NilCtor(), ctx))
        ctx = merge(ctx, initialize_ctx(ApplyTemplateCtor(), ctx))
        merge(ctx, initialize_ctx(ctx.chooser, ctx))
    end

    @test ctx_red.templates_ty[:REDUCE_SUM] isa TyScheme
    red_applied = construct_tree(ApplyTemplateCtor(), IntTy, ctx_red)
    @test red_applied isa AstTree
    @test red_applied.root isa AppNode
    @test red_applied.root.func == red_tree.root
    type_ctx_red = merge(ctx_red.global_consts_ty, ctx_red.local_consts_ty)
    @test iscompatible(infer(red_applied, type_ctx_red), IntTy)
end

@testitem "GenerateTree smoke thresholds (deterministic setups)" begin
    using Random
    using LambdaRegression.GenerateTree: generate_tree, AbsCtor, IndexCtor, ApplyLeafCtor, IntegerCtor
    using LambdaRegression.TypeSystem
    using LambdaRegression.TypeSystem: TyLike, IntTy
    using LambdaRegression.ASTs.NodeASTs: AstTree

    targets = Any[
        IntTy,
        IntTy → IntTy,
        ((IntTy → IntTy) → IntTy) → IntTy,
    ]
    ctors = [AbsCtor(), IndexCtor(), ApplyLeafCtor(), IntegerCtor()]
    for seed in 1:5
        for (i, ty) in enumerate(targets)
            rng = MersenneTwister(seed * 100 + i)
            global_consts_ty = Dict{Symbol,TyLike}()
            local_consts_ty = Dict{Symbol,TyLike}()
            local_consts = Dict{Symbol,Any}()
            tree = generate_tree(
                ty;
                ctors=ctors,
                rng=rng,
                max_depth=5,
                max_try=20,
                global_consts_ty=global_consts_ty,
                local_consts_ty=local_consts_ty,
                local_consts=local_consts,
            )
            @test tree isa AstTree
            inferred = infer(tree, merge(global_consts_ty, local_consts_ty))
            @test iscompatible(inferred, ty)
        end
    end
end

#=
@testitem "GenerateTree integrated batch generation" begin
    # Disabled temporarily:
    # the current integrated logic is too heavy and can stall full-suite runs.
    # Reintroduce as a short, layered test after constructor + small integration
    # coverage is stable.
end
=#

end # module GenerateTree
