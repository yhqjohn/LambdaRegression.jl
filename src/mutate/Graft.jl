"""
    GraftSite

Attempt-local recipient view passed to Graft routing callbacks. `path` is the
1-based root path, `node` is the current core AST node, `ty` is its
occurrence-specific inferred type, and `depth` is its lexical binder depth.
Treat `path` as read-only and do not retain the view after the mutation call.
"""
struct GraftSite
    path::Vector{Int}
    node::AstNode
    ty::Ty
    depth::Int
end

"""
    GraftBinding

Attempt-local existing HM let binding that may be referenced by a Graft.
`path` identifies the complete let-like application, `rhs_path` identifies its
unchanged RHS, and `scheme` is the maximally generalized RHS type used by
Algorithm W at that binding.
"""
struct GraftBinding
    path::Vector{Int}
    rhs_path::Vector{Int}
    name::Symbol
    scheme::TypeSystem.TyScheme
end

struct _GraftRecipient
    site::GraftSite
    scoped::_ScopedTerm
end

struct _GraftBindingRecord
    view::GraftBinding
    binder_id::Int
    body_path::Vector{Int}
    rhs_key
end

struct _GraftAnalysis
    scoped::_ScopedTree
    recipients::Vector{_GraftRecipient}
    by_path::Dict{Tuple,_GraftRecipient}
    bindings::Vector{_GraftBindingRecord}
end

struct _GraftPolicyCache
    recipient_weights::Dict{Tuple,Real}
    recipient_keys::Dict{Tuple,Any}
    bindings::Dict{Tuple,Tuple{Vector{_GraftBindingRecord},Vector{Real}}}
end

_GraftPolicyCache() = _GraftPolicyCache(
    Dict{Tuple,Real}(),
    Dict{Tuple,Any}(),
    Dict{Tuple,Tuple{Vector{_GraftBindingRecord},Vector{Real}}}(),
)

_uniform_graft_recipient_weight(::GraftSite, ::MutationEnv, ::NamedTuple) = 1.0
_uniform_graft_donor_weight(
    ::GraftBinding,
    ::GraftSite,
    ::MutationEnv,
    ::NamedTuple,
) = 1.0

"""
    GraftMutation(; recipient_weight, donor_weight)

Redirect one recipient inside an existing direct-`Abs` let body to that let's
binder. The RHS remains in place and no binding is created or copied. Candidate
pairs are finite and require lexical visibility, a genuinely new dependency,
non-CSE RHS/recipient keys, type compatibility after fresh scheme instantiation,
and stable HM application roles.

`recipient_weight(recipient, env, ctx)` controls outer recipient routing.
`donor_weight(binding, recipient, env, ctx)` controls which visible existing let
binding receives the new use; zero excludes a pair. Both callbacks must return
finite, nonnegative real weights. Graft is opt-in and is not in the ordinary
preset. It requires only the ordinary `mutate_tree` context fields. Expected
failure includes no visible compatible binding, a CSE-equal RHS/recipient pair,
an existing dependency, an HM-role change, or final inference rejection;
unsupported non-core AST nodes are rejected finitely by the scoped resolver.
Injected callbacks are trusted to terminate and treat their attempt-local views
as read-only.
"""
struct GraftMutation{R,D} <: AbstractMutation
    recipient_weight::R
    donor_weight::D
end

GraftMutation(;
    recipient_weight=_uniform_graft_recipient_weight,
    donor_weight=_uniform_graft_donor_weight,
) = GraftMutation(recipient_weight, donor_weight)

required_site_kinds(::GraftMutation) = Symbol[:any]
base_weight(::GraftMutation, ::NamedTuple) = 0.02

function _checked_graft_weight(value, label::AbstractString)
    value isa Real || throw(ArgumentError("$label must return a Real weight"))
    isfinite(value) || throw(ArgumentError("$label must return a finite weight"))
    value >= 0 || throw(ArgumentError("$label must return a nonnegative weight"))
    return value
end

function _scoped_alpha_key(term::_ScopedTerm)
    next_local = Ref(0)
    locals = Dict{Int,Int}()
    function walk(node::_ScopedTerm)
        if node isa _ScopedConst
            return (:const, node.name)
        elseif node isa _ScopedFree
            return (:free, node.name)
        elseif node isa _ScopedBound
            slot = get(locals, node.binder_id, nothing)
            return slot === nothing ?
                (:capture, node.binder_id) : (:local, slot)
        elseif node isa _ScopedAbs
            next_local[] += 1
            slot = next_local[]
            locals[node.binder_id] = slot
            body = walk(node.body)
            delete!(locals, node.binder_id)
            return (:abs, body)
        elseif node isa _ScopedApp
            return (:app, node.role, walk(node.func), walk(node.arg))
        end
        return (:unsupported, typeof(node))
    end
    return walk(term)
end

function _collect_graft_analysis!(
    recipients::Vector{_GraftRecipient},
    by_path::Dict{Tuple,_GraftRecipient},
    bindings::Vector{_GraftBindingRecord},
    ast::AstNode,
    scoped::_ScopedTerm,
    metadata::Dict{Tuple,Tuple{TyLike,TyLike}},
    tree::AstTree,
    target_ty::Ty,
    path::Vector{Int},
    lexical_depth::Int,
    inference_level::Int,
)
    key = Tuple(path)
    if haskey(metadata, key)
        raw_ty = metadata[key][1]
        if raw_ty isa Ty
            ty = _target_refined_site_ty(tree, path, target_ty, raw_ty)
            recipient = _GraftRecipient(
                GraftSite(copy(path), ast, ty, lexical_depth),
                scoped,
            )
            push!(recipients, recipient)
            by_path[key] = recipient
        end
    end

    if ast isa AbsNode && scoped isa _ScopedAbs
        push!(path, 1)
        _collect_graft_analysis!(
            recipients,
            by_path,
            bindings,
            ast.body,
            scoped.body,
            metadata,
            tree,
            target_ty,
            path,
            lexical_depth + 1,
            inference_level,
        )
        pop!(path)
    elseif ast isa AppNode && scoped isa _ScopedApp
        is_let = ast.func isa AbsNode && scoped.func isa _ScopedAbs
        if is_let
            rhs_path = [path; 2]
            rhs_meta = get(metadata, Tuple(rhs_path), nothing)
            if rhs_meta !== nothing && rhs_meta[1] isa Ty
                scheme = TypeSystem.generalize(rhs_meta[1], inference_level)
                push!(bindings, _GraftBindingRecord(
                    GraftBinding(
                        copy(path),
                        rhs_path,
                        scoped.func.name,
                        scheme,
                    ),
                    scoped.func.binder_id,
                    [path; 1; 1],
                    _scoped_alpha_key(scoped.arg),
                ))
            end
        end

        push!(path, 1)
        _collect_graft_analysis!(
            recipients,
            by_path,
            bindings,
            ast.func,
            scoped.func,
            metadata,
            tree,
            target_ty,
            path,
            lexical_depth,
            inference_level,
        )
        pop!(path)

        push!(path, 2)
        _collect_graft_analysis!(
            recipients,
            by_path,
            bindings,
            ast.arg,
            scoped.arg,
            metadata,
            tree,
            target_ty,
            path,
            lexical_depth,
            is_let ? inference_level + 1 : inference_level,
        )
        pop!(path)
    end
    return nothing
end

function _graft_analysis(env::MutationEnv, ctx::NamedTuple)
    return get!(env.eligibility_cache, _graft_analysis) do
        env.tree === nothing && return Err(NoEligibleSiteErr())
        scoped = _resolved_scoped(env)
        scoped isa Err && return scoped
        metadata = _typed_occurrence_metadata(env, ctx)
        metadata isa Err && return metadata
        recipients = _GraftRecipient[]
        by_path = Dict{Tuple,_GraftRecipient}()
        bindings = _GraftBindingRecord[]
        _collect_graft_analysis!(
            recipients,
            by_path,
            bindings,
            env.tree.root,
            scoped.root,
            metadata,
            env.tree,
            env.target_ty,
            Int[],
            0,
            0,
        )
        return _GraftAnalysis(scoped, recipients, by_path, bindings)
    end
end

_graft_path_prefix(prefix::Vector{Int}, path::Vector{Int}) =
    length(prefix) <= length(path) &&
    all(index -> prefix[index] == path[index], eachindex(prefix))

function _graft_parent_role_stable(
    scoped::_ScopedTree,
    recipient_path::Vector{Int},
)
    isempty(recipient_path) && return true
    last(recipient_path) == 1 || return true
    parent = _scoped_at_path(scoped, @view(recipient_path[1:(end - 1)]))
    parent isa _ScopedApp || return true
    return !(parent.func isa _ScopedAbs)
end

function _graft_bindings(
    mutation::GraftMutation,
    recipient::_GraftRecipient,
    analysis::_GraftAnalysis,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _graft_parent_role_stable(analysis.scoped, recipient.site.path) ||
        return _GraftBindingRecord[], Real[]
    recipient_key = _cached_graft_recipient_key(mutation, recipient, env)
    bindings = _GraftBindingRecord[]
    weights = Real[]
    for binding in analysis.bindings
        _graft_path_prefix(binding.body_path, recipient.site.path) || continue
        _scoped_uses_binder(recipient.scoped, binding.binder_id) && continue
        recipient_key == binding.rhs_key && continue
        iscompatible(binding.view.scheme, recipient.site.ty) || continue
        weight = _checked_graft_weight(
            mutation.donor_weight(binding.view, recipient.site, env, ctx),
            "donor_weight",
        )
        iszero(weight) && continue
        push!(bindings, binding)
        push!(weights, weight)
    end
    return bindings, weights
end

function _graft_policy_cache(mutation::GraftMutation, env::MutationEnv)
    caches = get!(env.eligibility_cache, _graft_policy_cache) do
        Dict{Any,_GraftPolicyCache}()
    end
    return get!(caches, mutation) do
        _GraftPolicyCache()
    end
end

function _cached_graft_recipient_weight(
    mutation::GraftMutation,
    recipient::_GraftRecipient,
    env::MutationEnv,
    ctx::NamedTuple,
)
    cache = _graft_policy_cache(mutation, env)
    key = Tuple(recipient.site.path)
    return get!(cache.recipient_weights, key) do
        _checked_graft_weight(
            mutation.recipient_weight(recipient.site, env, ctx),
            "recipient_weight",
        )
    end
end

function _cached_graft_recipient_key(
    mutation::GraftMutation,
    recipient::_GraftRecipient,
    env::MutationEnv,
)
    cache = _graft_policy_cache(mutation, env)
    key = Tuple(recipient.site.path)
    return get!(cache.recipient_keys, key) do
        _scoped_alpha_key(recipient.scoped)
    end
end

function _cached_graft_bindings(
    mutation::GraftMutation,
    recipient::_GraftRecipient,
    analysis::_GraftAnalysis,
    env::MutationEnv,
    ctx::NamedTuple,
)
    cache = _graft_policy_cache(mutation, env)
    key = Tuple(recipient.site.path)
    return get!(cache.bindings, key) do
        _graft_bindings(mutation, recipient, analysis, env, ctx)
    end
end

function _graft_recipient(site::Site, env::MutationEnv, ctx::NamedTuple)
    analysis = _graft_analysis(env, ctx)
    analysis isa Err && return analysis
    return get(analysis.by_path, Tuple(site.path), Err(NoEligibleSiteErr()))
end

function is_site_eligible(
    mutation::GraftMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    recipient = _graft_recipient(site, env, ctx)
    recipient isa Err && return false
    weight = _cached_graft_recipient_weight(mutation, recipient, env, ctx)
    iszero(weight) && return false
    analysis = _graft_analysis(env, ctx)
    bindings, _ = _cached_graft_bindings(
        mutation,
        recipient,
        analysis,
        env,
        ctx,
    )
    return !isempty(bindings)
end

function site_weight(
    mutation::GraftMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    recipient = _graft_recipient(site, env, ctx)
    recipient isa Err && return 0.0
    weight = _cached_graft_recipient_weight(mutation, recipient, env, ctx)
    return Float64(weight)
end

function _choose_graft_binding(
    rng::AbstractRNG,
    bindings::Vector{_GraftBindingRecord},
    weights::Vector{Real},
)
    units = _integer_weight_units(weights)
    ticket = rand(rng, BigInt(1):sum(units))
    return bindings[_weighted_ticket_index(units, ticket)]
end

function apply(
    mutation::GraftMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    analysis = _graft_analysis(env, ctx)
    analysis isa Err && return analysis
    recipient = get(
        analysis.by_path,
        Tuple(site.path),
        Err(NoEligibleSiteErr()),
    )
    recipient isa Err && return recipient
    bindings, weights = _cached_graft_bindings(
        mutation,
        recipient,
        analysis,
        env,
        ctx,
    )
    isempty(bindings) && return Err(NoEligibleSiteErr())
    binding = _choose_graft_binding(env.rng, bindings, weights)

    retained = _retained_application_ancestors(
        analysis.scoped,
        recipient.site.path,
    )
    retained isa Err && return retained
    rewritten = _replace_scoped(
        analysis.scoped,
        recipient.site.path,
        _ScopedBound(binding.binder_id),
    )
    rewritten isa Err && return rewritten
    if !_application_roles_stable(analysis.scoped, rewritten, retained)
        changed_path = _changed_retained_application_path(
            analysis.scoped,
            rewritten,
            retained,
        )
        return Err(_RetainedApplicationRoleErr(changed_path))
    end

    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    materialized == tree && return Err(NoEligibleSiteErr())
    factory = haskey(ctx, :tyfactory) ? ctx.tyfactory : newtyvar
    inferred = _run_mutation_inference(
        ctx,
        materialized,
        env.type_ctx,
        factory,
        nothing,
    )
    inferred isa Err && return inferred
    iscompatible(inferred, env.target_ty) ||
        return Err(TypeInvariantErr(env.target_ty, inferred))
    tree.root = materialized.root
    return tree
end

@testitem "GraftMutation redirects a recipient to an existing let binding" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    original = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:one)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:rhs),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
        :two => IntTy,
        :rhs => IntTy,
    )
    mutation = GraftMutation(
        recipient_weight=(site, _, _) ->
            site.path == [1, 1, 1, 2] ? 1.0 : 0.0,
        donor_weight=(binding, _, _, _) ->
            isempty(binding.path) ? 1.0 : 0.0,
    )

    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GraftMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), IndexNode(0)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:rhs),
        ),
    )
    @test infer(result, type_ctx) == IntTy
end

@testitem "GraftMutation redirects a nontrivial captured expression with correct De Bruijn depth" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    original = AstTree(
        AbsNode(
            :x,
            AppNode(
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(
                            ConstNode(:add),
                            AppNode(ConstNode(:neg), IndexNode(1)),
                        ),
                        ConstNode(:one),
                    ),
                ),
                AppNode(ConstNode(:sqr), IndexNode(0)),
            ),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :neg => TyFun(IntTy, IntTy),
        :sqr => TyFun(IntTy, IntTy),
        :one => IntTy,
    )
    mutation = GraftMutation(
        recipient_weight=(site, _, _) ->
            site.path == [1, 1, 1, 1, 2] ? 1.0 : 0.0,
        donor_weight=(binding, _, _, _) ->
            binding.path == [1] ? 1.0 : 0.0,
    )
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(GraftMutation => 1.0),
        cache_eligibility=true,
    )

    result = mutate_tree(
        original,
        TyFun(IntTy, IntTy);
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser,
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AbsNode(
            :x,
            AppNode(
                AbsNode(
                    :y,
                    AppNode(
                        AppNode(ConstNode(:add), IndexNode(0)),
                        ConstNode(:one),
                    ),
                ),
                AppNode(ConstNode(:sqr), IndexNode(0)),
            ),
        ),
    )
    @test infer(result, type_ctx) == TyFun(IntTy, IntTy)
end

@testitem "GraftMutation reinstantiates a maximally generalized let scheme" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: BoolTy, IntTy, TyFun, TyLike, infer

    original = AstTree(
        AppNode(
            AbsNode(
                :id,
                AppNode(
                    AppNode(
                        ConstNode(:add),
                        AppNode(ConstNode(:consume_int), IndexNode(0)),
                    ),
                    AppNode(ConstNode(:consume_bool), ConstNode(:not)),
                ),
            ),
            AbsNode(:value, IndexNode(0)),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :consume_int => TyFun(TyFun(IntTy, IntTy), IntTy),
        :consume_bool => TyFun(TyFun(BoolTy, BoolTy), IntTy),
        :not => TyFun(BoolTy, BoolTy),
    )
    mutation = GraftMutation(
        recipient_weight=(site, _, _) ->
            site.path == [1, 1, 2, 2] ? 1.0 : 0.0,
        donor_weight=(binding, _, _, _) ->
            isempty(binding.path) ? 1.0 : 0.0,
    )

    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(1234),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GraftMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(
            AbsNode(
                :id,
                AppNode(
                    AppNode(
                        ConstNode(:add),
                        AppNode(ConstNode(:consume_int), IndexNode(0)),
                    ),
                    AppNode(ConstNode(:consume_bool), IndexNode(0)),
                ),
            ),
            AbsNode(:value, IndexNode(0)),
        ),
    )
    @test infer(result, type_ctx) == IntTy
end

@testitem "GraftMutation rejects RHS, existing-use, CSE, and HM-role pairs" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :f => TyFun(IntTy, IntTy),
        :one => IntTy,
        :two => IntTy,
        :three => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(GraftMutation => 1.0),
        cache_eligibility=true,
    )
    function rejected(tree, target, path, seed)
        mutation = GraftMutation(
            recipient_weight=(site, _, _) -> site.path == path ? 1.0 : 0.0,
            donor_weight=(binding, _, _, _) ->
                isempty(binding.path) ? 1.0 : 0.0,
        )
        return mutate_tree(
            tree,
            target;
            rng=MersenneTwister(seed),
            global_consts_ty=type_ctx,
            mutations=[mutation],
            chooser,
            provider=IndexedProvider(),
            max_try=1,
        )
    end

    rhs_recipient = AstTree(
        AppNode(
            AbsNode(:bound, ConstNode(:three)),
            AppNode(
                AppNode(ConstNode(:add), ConstNode(:one)),
                ConstNode(:two),
            ),
        ),
    )
    @test rejected(rhs_recipient, IntTy, [2, 1, 2], 4) isa Err

    existing_use = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), IndexNode(0)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:one),
        ),
    )
    @test rejected(existing_use, IntTy, [1, 1], 5) isa Err

    cse = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:one)),
                    ConstNode(:two),
                ),
            ),
            AppNode(
                AppNode(ConstNode(:add), ConstNode(:one)),
                ConstNode(:two),
            ),
        ),
    )
    @test rejected(cse, IntTy, [1, 1], 6) isa Err

    role_change = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(AbsNode(:value, IndexNode(0)), ConstNode(:one)),
            ),
            ConstNode(:f),
        ),
    )
    @test rejected(role_change, IntTy, [1, 1, 1], 7) isa Err
end

@testitem "GraftMutation accepts equirecursive-compatible binding use" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyLike, TyMu, infer, iscompatible, make_tyvar_factory

    fresh = make_tyvar_factory()
    left_binder = fresh(0)
    right_binder = fresh(0)
    left = TyMu(left_binder, TyFun(IntTy, left_binder))
    right = TyMu(right_binder, TyFun(IntTy, TyFun(IntTy, right_binder)))
    @test iscompatible(left, right)

    original = AstTree(
        AppNode(
            AbsNode(:bound, ConstNode(:other)),
            ConstNode(:loop),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(:loop => left, :other => right)
    mutation = GraftMutation(
        recipient_weight=(site, _, _) ->
            site.path == [1, 1] ? 1.0 : 0.0,
        donor_weight=(binding, _, _, _) ->
            isempty(binding.path) ? 1.0 : 0.0,
    )
    result = mutate_tree(
        original,
        left;
        rng=MersenneTwister(1),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GraftMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result == AstTree(
        AppNode(AbsNode(:bound, IndexNode(0)), ConstNode(:loop)),
    )
    @test iscompatible(infer(result, type_ctx), left)
end

@testitem "GraftMutation evaluates routing weights once per candidate" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    original = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:one)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:rhs),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
        :two => IntTy,
        :rhs => IntTy,
    )
    recipient_calls = Ref(0)
    donor_calls = Ref(0)
    mutation = GraftMutation(
        recipient_weight=(site, _, _) -> begin
            site.path == [1, 1, 1, 2] || return 0.0
            recipient_calls[] += 1
            return 1.0
        end,
        donor_weight=(binding, _, _, _) -> begin
            isempty(binding.path) || return 0.0
            donor_calls[] += 1
            return 1.0
        end,
    )

    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(1111),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GraftMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        max_try=1,
    )

    @test result isa AstTree
    @test recipient_calls[] == 1
    @test donor_calls[] == 1
end

@testitem "GraftMutation resolves nested visible bindings by identity" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer

    original = AstTree(
        AppNode(
            AbsNode(
                :outer,
                AppNode(
                    AbsNode(
                        :inner,
                        AppNode(
                            AppNode(ConstNode(:add), ConstNode(:one)),
                            ConstNode(:two),
                        ),
                    ),
                    ConstNode(:inner_rhs),
                ),
            ),
            ConstNode(:outer_rhs),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
        :two => IntTy,
        :inner_rhs => IntTy,
        :outer_rhs => IntTy,
    )
    recipient_path = [1, 1, 1, 1, 1, 2]
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(GraftMutation => 1.0),
        cache_eligibility=true,
    )

    function graft_from(binding_path, seed)
        mutation = GraftMutation(
            recipient_weight=(site, _, _) ->
                site.path == recipient_path ? 1.0 : 0.0,
            donor_weight=(binding, _, _, _) ->
                binding.path == binding_path ? 1.0 : 0.0,
        )
        return mutate_tree(
            original,
            IntTy;
            rng=MersenneTwister(seed),
            global_consts_ty=type_ctx,
            mutations=[mutation],
            chooser,
            provider=IndexedProvider(),
            max_try=1,
        )
    end

    outer = graft_from(Int[], 10)
    inner = graft_from([1, 1], 11)

    @test outer == AstTree(
        AppNode(
            AbsNode(
                :outer,
                AppNode(
                    AbsNode(
                        :inner,
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(1)),
                            ConstNode(:two),
                        ),
                    ),
                    ConstNode(:inner_rhs),
                ),
            ),
            ConstNode(:outer_rhs),
        ),
    )
    @test inner == AstTree(
        AppNode(
            AbsNode(
                :outer,
                AppNode(
                    AbsNode(
                        :inner,
                        AppNode(
                            AppNode(ConstNode(:add), IndexNode(0)),
                            ConstNode(:two),
                        ),
                    ),
                    ConstNode(:inner_rhs),
                ),
            ),
            ConstNode(:outer_rhs),
        ),
    )
    @test infer(outer, type_ctx) == IntTy
    @test infer(inner, type_ctx) == IntTy
end

@testitem "GraftMutation is deterministic, opt-in, and validates routing weights" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations:
        GraftBinding, GraftMutation, IndexedProvider, WeightedPairChooser,
        base_weight, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    original = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:one)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:rhs),
        ),
    )
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
        :two => IntTy,
        :rhs => IntTy,
    )
    chooser = WeightedPairChooser(
        weights=Dict{Type,Float64}(GraftMutation => 1.0),
        cache_eligibility=true,
    )
    mutate(mutation, seed) = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(seed),
        global_consts_ty=type_ctx,
        mutations=[mutation],
        chooser,
        provider=IndexedProvider(),
        max_try=1,
    )

    @test mutate(GraftMutation(), 12) == mutate(GraftMutation(), 12)
    @test base_weight(GraftMutation(), NamedTuple()) == 0.02
    @test :GraftBinding in names(LambdaRegression.Mutations)
    @test GraftBinding isa DataType
    @test all(
        candidate -> !(candidate isa GraftMutation),
        LambdaRegression.Mutations._default_mutations(),
    )
    @test all(
        candidate -> !(candidate isa GraftMutation),
        LambdaRegression.Regression.default_components().mutations,
    )

    @test_throws ArgumentError mutate(
        GraftMutation(recipient_weight=(_, _, _) -> -1.0),
        13,
    )
    @test_throws ArgumentError mutate(
        GraftMutation(
            recipient_weight=(site, _, _) ->
                site.path == [1, 1, 1, 2] ? 1.0 : 0.0,
            donor_weight=(_, _, _, _) -> Inf,
        ),
        14,
    )
end

@testitem "GraftMutation rejects final validation transactionally" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.Mutations:
        GraftMutation, IndexedProvider, WeightedPairChooser, mutate_tree
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    original = AstTree(
        AppNode(
            AbsNode(
                :bound,
                AppNode(
                    AppNode(ConstNode(:add), ConstNode(:one)),
                    ConstNode(:two),
                ),
            ),
            ConstNode(:rhs),
        ),
    )
    snapshot = deepcopy(original)
    local_consts = Dict{Symbol,Any}(:kept => 3)
    local_consts_ty = Dict{Symbol,TyLike}(:kept => IntTy)
    type_ctx = Dict{Symbol,TyLike}(
        :add => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :one => IntTy,
        :two => IntTy,
        :rhs => IntTy,
    )
    checker = function (tree, context, factory, meta)
        meta === nothing && return Err(ArgumentError("reject final Graft"))
        return infer(tree, context, factory, meta)
    end
    mutation = GraftMutation(
        recipient_weight=(site, _, _) ->
            site.path == [1, 1, 1, 2] ? 1.0 : 0.0,
    )

    result = mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(42),
        global_consts_ty=type_ctx,
        local_consts,
        local_consts_ty,
        mutations=[mutation],
        chooser=WeightedPairChooser(
            weights=Dict{Type,Float64}(GraftMutation => 1.0),
            cache_eligibility=true,
        ),
        provider=IndexedProvider(),
        type_inference=checker,
        max_try=1,
    )

    @test result isa Err
    @test original == snapshot
    @test local_consts == Dict{Symbol,Any}(:kept => 3)
    @test local_consts_ty == Dict{Symbol,TyLike}(:kept => IntTy)
end
