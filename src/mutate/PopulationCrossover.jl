"""
    Mutations.Populations typed crossover implementations

This file contains **concrete** crossover operators for the population-level
variation layer.

Threading / determinism requirements:
- No global RNG usage: callers must pass `rng`.
- No global type-variable factories: use `tyfactory_ctor()` per call.
- No `gensym` in SR paths: local-constant remapping uses `ConstSymbolFactory`.
"""

using Random
using TestItems

using ...Utils: Err
using ...TypeSystem: Ty, TyLike, TyFun, infer, iscompatible, make_tyvar_factory
using ...ASTs.NodeASTs: AstTree, AstNode, AbsNode, ConstNode, IndexNode, ScopeState, children
using ...ASTs.Contexts: ConstSymbolFactory
using ...Trees: traverse, PointerState, getstate, Break, setroot!, setchild!

export TypedSubtreeSwapCrossover

struct TargetTypeMismatchErr
    a::Ty
    b::Ty
end
struct MissingGlobalTypeCtxErr end
struct NoEligibleCrossoverSiteErr end
struct CrossoverTypeInvariantErr
    expected::Ty
    got::Ty
end
struct CrossoverMaxTryReachedErr
    max_try::Int
end

struct _XSite
    path::Vector{Int}
    depth::Int
    ty::Ty
    binder_types::Vector{Ty}
end

function _binder_types(scope, meta)::Vector{Ty}
    tys = Ty[]
    for abs in scope.abstractions
        haskey(meta, abs) || continue
        abs_ty = meta[abs][1]
        abs_ty isa TyFun || continue
        push!(tys, (abs_ty::TyFun).arg)
    end
    return tys
end

function _collect_sites(tree::AstTree, type_ctx::Dict{Symbol,<:TyLike}; tyfactory)
    meta = Dict{AstNode,Tuple{TyLike,TyLike}}()
    inferred = infer(tree, type_ctx, tyfactory, meta)
    inferred isa Err && return inferred

    sites = _XSite[]
    paths = IdDict{AstNode,Vector{Int}}()

    traverse(tree, PointerState{AstNode}, ScopeState) do _, node, bag
        pointer = getstate(bag, PointerState{AstNode})
        scope = getstate(bag, ScopeState)

        # Root node can appear in traversal before `meta` contains it in some early-return paths.
        haskey(meta, node) || return nothing

        path =
            if pointer.parent === nothing
                Int[]
            else
                parent_path = get(paths, pointer.parent, Int[])
                vcat(parent_path, pointer.child_index)
            end
        paths[node] = path

        node_ty = meta[node][1]
        binder_types = _binder_types(scope, meta)
        push!(sites, _XSite(path, length(binder_types), node_ty, binder_types))
        return nothing
    end

    return sites
end

function _node_at_path(tree::AstTree, path::Vector{Int})::AstNode
    node = tree.root
    for idx in path
        node = children(node)[idx]
    end
    return node
end

function _replace_at_path!(tree::AstTree, path::Vector{Int}, new_subtree::AstTree)
    if isempty(path)
        setroot!(tree, new_subtree)
        return tree
    end
    parent_path = path[1:end-1]
    parent = _node_at_path(tree, parent_path)
    setchild!(tree, parent, path[end], new_subtree)
    return tree
end

function _fresh_const_symbol!(f::ConstSymbolFactory,
    local_consts::Dict{Symbol,Any},
    local_consts_ty::Dict{Symbol,TyLike},
)::Symbol
    while true
        f.counter += 1
        name = Symbol(String(f.prefix), f.counter)
        (haskey(local_consts, name) || haskey(local_consts_ty, name)) && continue
        return name
    end
end

function _remap_donor_locals!(subtree::AstTree,
    donor_local_consts::Dict{Symbol,Any},
    donor_local_consts_ty::Dict{Symbol,TyLike},
    recipient_local_consts::Dict{Symbol,Any},
    recipient_local_consts_ty::Dict{Symbol,TyLike},
    factory::ConstSymbolFactory,
)
    mapping = Dict{Symbol,Symbol}()

    traverse(subtree, PointerState{AstNode}) do t, node, bag
        node isa ConstNode || return nothing
        name = (node::ConstNode).name
        (haskey(donor_local_consts, name) && haskey(donor_local_consts_ty, name)) || return nothing

        new_name = get!(mapping, name) do
            fresh = _fresh_const_symbol!(factory, recipient_local_consts, recipient_local_consts_ty)
            recipient_local_consts[fresh] = donor_local_consts[name]
            recipient_local_consts_ty[fresh] = donor_local_consts_ty[name]
            fresh
        end

        pointer = getstate(bag, PointerState{AstNode})
        if pointer.parent === nothing
            setroot!(t, AstTree(ConstNode(new_name)))
        else
            setchild!(t, pointer.parent, pointer.child_index, AstTree(ConstNode(new_name)))
        end
        return nothing
    end

    return nothing
end

function _sites_match(op, a::_XSite, b::_XSite)::Bool
    iscompatible(a.ty, b.ty) || return false
    iscompatible(b.ty, a.ty) || return false

    op.require_same_depth && (a.depth == b.depth) || (!op.require_same_depth) || return false

    if op.require_compatible_binders
        length(a.binder_types) == length(b.binder_types) || return false
        for (ta, tb) in zip(a.binder_types, b.binder_types)
            iscompatible(ta, tb) || return false
            iscompatible(tb, ta) || return false
        end
    end
    return true
end

"""
    TypedSubtreeSwapCrossover(; max_try=40, require_same_depth=true, require_compatible_binders=true,
        min_sites=2, allow_root=true) <: AbstractCrossover

Type-safe subtree swap crossover between two individuals.

This operator:
- collects typed sites in both parents,
- samples a compatible pair of sites,
- swaps the corresponding subtrees,
- remaps donor-local constants into the recipient's local constant pool using a
  per-call `ConstSymbolFactory` (no `gensym`),
- validates the offspring types using `infer` + `iscompatible`.

Thread safety:
- requires a task-local `rng`,
- allocates a fresh `tyfactory = tyfactory_ctor()` per call,
- allocates per-offspring constant symbol factories via `const_symfactory_ctor()`.
"""
struct TypedSubtreeSwapCrossover <: AbstractCrossover
    max_try::Int
    require_same_depth::Bool
    require_compatible_binders::Bool
    min_sites::Int
    allow_root::Bool
end

TypedSubtreeSwapCrossover(;
    max_try::Int=40,
    require_same_depth::Bool=true,
    require_compatible_binders::Bool=true,
    min_sites::Int=2,
    allow_root::Bool=true,
) = TypedSubtreeSwapCrossover(max_try, require_same_depth, require_compatible_binders, min_sites, allow_root)

"""
    crossover(op::TypedSubtreeSwapCrossover, a::Individual, b::Individual;
        rng=MersenneTwister(2000),
        global_consts_ty=nothing,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    ) -> Union{Tuple{Individual,Individual},Err}

Apply type-safe subtree swap crossover.

Required keyword arguments:
- `global_consts_ty`: shared, read-only typing context for global constants/operators.

Thread-safety requirements:
- pass a task-local `rng`;
- the implementation constructs a per-call `tyfactory = tyfactory_ctor()` and uses
  `const_symfactory_ctor()` to generate fresh local-constant symbols deterministically.
"""
function crossover(op::TypedSubtreeSwapCrossover,
    a::Individual,
    b::Individual;
    rng::AbstractRNG=MersenneTwister(2000),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    prune::Bool=true,
)
    a.target_ty == b.target_ty || return Err(TargetTypeMismatchErr(a.target_ty, b.target_ty))
    global_consts_ty === nothing && return Err(MissingGlobalTypeCtxErr())

    tyfactory = tyfactory_ctor()

    # Pre-collect sites using the parents' own local type contexts.
    sites_a = _collect_sites(a.tree, merge(global_consts_ty, a.local_consts_ty); tyfactory=tyfactory)
    sites_a isa Err && return sites_a
    sites_b = _collect_sites(b.tree, merge(global_consts_ty, b.local_consts_ty); tyfactory=tyfactory)
    sites_b isa Err && return sites_b

    allow_root = op.allow_root
    sites_a2 = allow_root ? sites_a : filter(s -> !isempty(s.path), sites_a)
    sites_b2 = allow_root ? sites_b : filter(s -> !isempty(s.path), sites_b)

    length(sites_a2) >= op.min_sites || return Err(NoEligibleCrossoverSiteErr())
    length(sites_b2) >= op.min_sites || return Err(NoEligibleCrossoverSiteErr())

    for _ in 1:op.max_try
        site_a = rand(rng, sites_a2)
        candidates_b = _XSite[]
        for s in sites_b2
            _sites_match(op, site_a, s) || continue
            push!(candidates_b, s)
        end
        isempty(candidates_b) && continue
        site_b = rand(rng, candidates_b)

        # Build offspring from fresh copies (transactional).
        tree_a = deepcopy(a.tree)
        tree_b = deepcopy(b.tree)
        local_a = copy(a.local_consts)
        local_a_ty = copy(a.local_consts_ty)
        local_b = copy(b.local_consts)
        local_b_ty = copy(b.local_consts_ty)

        node_a = _node_at_path(tree_a, site_a.path)
        node_b = _node_at_path(tree_b, site_b.path)
        sub_a = AstTree(deepcopy(node_a))
        sub_b = AstTree(deepcopy(node_b))

        # Remap donor locals before splicing.
        _remap_donor_locals!(sub_b, b.local_consts, b.local_consts_ty, local_a, local_a_ty, const_symfactory_ctor())
        _remap_donor_locals!(sub_a, a.local_consts, a.local_consts_ty, local_b, local_b_ty, const_symfactory_ctor())

        _replace_at_path!(tree_a, site_a.path, sub_b)
        _replace_at_path!(tree_b, site_b.path, sub_a)

        if prune
            prune_local_consts!(tree_a, local_a, local_a_ty)
            prune_local_consts!(tree_b, local_b, local_b_ty)
        end

        # Validate type invariants.
        type_ctx_a = merge(global_consts_ty, local_a_ty)
        inf_a = infer(tree_a, type_ctx_a, tyfactory)
        inf_a isa Err && continue
        iscompatible(inf_a, a.target_ty) || continue

        type_ctx_b = merge(global_consts_ty, local_b_ty)
        inf_b = infer(tree_b, type_ctx_b, tyfactory)
        inf_b isa Err && continue
        iscompatible(inf_b, b.target_ty) || continue

        child_a = Individual(tree_a, a.target_ty; local_consts=local_a, local_consts_ty=local_a_ty, fitness=nothing, meta=(;))
        child_b = Individual(tree_b, b.target_ty; local_consts=local_b, local_consts_ty=local_b_ty, fitness=nothing, meta=(;))
        return (child_a, child_b)
    end

    return Err(CrossoverMaxTryReachedErr(op.max_try))
end

@testitem "Populations TypedSubtreeSwapCrossover: basic success + type stable (TyFun)" begin
    using Random
    using LambdaRegression.Mutations.Populations: Individual, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: TyFun, IntTy, TyLike, infer, iscompatible
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory
    using LambdaRegression.TypeSystem: make_tyvar_factory

    rng = MersenneTwister(1)
    target_ty = TyFun(IntTy, IntTy)
    global_consts_ty = Dict{Symbol,TyLike}(:add => TyFun(IntTy, TyFun(IntTy, IntTy)))

    tree_a = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), IndexNode(0)), ConstNode(:k))))
    tree_b = AstTree(AbsNode(:x, AppNode(AppNode(ConstNode(:add), ConstNode(:k)), IndexNode(0))))

    ind_a = Individual(tree_a, target_ty; local_consts=Dict{Symbol,Any}(:k => 1), local_consts_ty=Dict{Symbol,TyLike}(:k => IntTy))
    ind_b = Individual(tree_b, target_ty; local_consts=Dict{Symbol,Any}(:k => 99), local_consts_ty=Dict{Symbol,TyLike}(:k => IntTy))

    op = TypedSubtreeSwapCrossover(max_try=20)
    (c1, c2) = crossover(
        op,
        ind_a,
        ind_b;
        rng=rng,
        global_consts_ty=global_consts_ty,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    tyf = make_tyvar_factory()
    t1 = infer(c1.tree, merge(global_consts_ty, c1.local_consts_ty), tyf)
    @test iscompatible(t1, target_ty)
    tyf2 = make_tyvar_factory()
    t2 = infer(c2.tree, merge(global_consts_ty, c2.local_consts_ty), tyf2)
    @test iscompatible(t2, target_ty)
end

@testitem "Populations TypedSubtreeSwapCrossover: local-const conflict renamed" begin
    using Random
    using LambdaRegression.Mutations.Populations: Individual, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: IntTy, TyLike
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory
    using LambdaRegression.TypeSystem: make_tyvar_factory

    rng = MersenneTwister(1111)
    global_consts_ty = Dict{Symbol,TyLike}()

    # Both parents use the same local constant name but different values.
    ind_a = Individual(AstTree(ConstNode(:c1)), IntTy; local_consts=Dict{Symbol,Any}(:c1 => 1), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))
    ind_b = Individual(AstTree(ConstNode(:c1)), IntTy; local_consts=Dict{Symbol,Any}(:c1 => 2), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))

    op = TypedSubtreeSwapCrossover(max_try=5, min_sites=1, allow_root=true)
    (c1, c2) = crossover(
        op,
        ind_a,
        ind_b;
        rng=rng,
        global_consts_ty=global_consts_ty,
        tyfactory_ctor=make_tyvar_factory,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        prune=true,
    )

    @test c1.tree.root isa ConstNode
    @test (c1.tree.root::ConstNode).name != :c1
    @test haskey(c1.local_consts, (c1.tree.root::ConstNode).name)
    @test c1.local_consts[(c1.tree.root::ConstNode).name] == 2

    @test c2.tree.root isa ConstNode
    @test (c2.tree.root::ConstNode).name != :c1
    @test haskey(c2.local_consts, (c2.tree.root::ConstNode).name)
    @test c2.local_consts[(c2.tree.root::ConstNode).name] == 1
end

@testitem "Populations TypedSubtreeSwapCrossover: target_ty mismatch fails" begin
    using Random
    using LambdaRegression.Mutations.Populations: Individual, TypedSubtreeSwapCrossover, crossover
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode, AbsNode, IndexNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    rng = MersenneTwister(1234)
    global_consts_ty = Dict{Symbol,TyLike}()

    ind_a = Individual(AstTree(ConstNode(:c1)), IntTy; local_consts=Dict{Symbol,Any}(:c1 => 1), local_consts_ty=Dict{Symbol,TyLike}(:c1 => IntTy))
    ind_b = Individual(AstTree(AbsNode(:x, IndexNode(0))), TyFun(IntTy, IntTy))

    op = TypedSubtreeSwapCrossover(max_try=5, min_sites=1, allow_root=true)
    res = crossover(op, ind_a, ind_b; rng=rng, global_consts_ty=global_consts_ty)
    @test res isa Err
end
