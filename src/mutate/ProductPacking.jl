abstract type AbstractProductCapability end

"""
    product_primitive_heads(capability) -> (pair, left, right)

Return the constant heads that construct a binary product and project its left
and right components. Custom capabilities overload this single method; the
mutation kernel owns scope analysis, rewriting, and final type validation.
"""
function product_primitive_heads(capability::AbstractProductCapability)
    throw(ArgumentError(
        "$(typeof(capability)) must implement product_primitive_heads",
    ))
end

struct PrimitiveProductCapability <: AbstractProductCapability
    pair_head::Symbol
    left_head::Symbol
    right_head::Symbol

    function PrimitiveProductCapability(
        pair_head::Symbol,
        left_head::Symbol,
        right_head::Symbol,
    )
        length(Set((pair_head, left_head, right_head))) == 3 ||
            throw(ArgumentError("product primitive heads must be distinct"))
        return new(pair_head, left_head, right_head)
    end
end

PrimitiveProductCapability(;
    pair_head::Symbol=:PAIR,
    left_head::Symbol=:FST,
    right_head::Symbol=:SND,
) = PrimitiveProductCapability(pair_head, left_head, right_head)

product_primitive_heads(capability::PrimitiveProductCapability) = (
    pair=capability.pair_head,
    left=capability.left_head,
    right=capability.right_head,
)

struct ProductPackMutation{C<:AbstractProductCapability} <: AbstractMutation
    capability::C
end

struct ProductUnpackMutation{C<:AbstractProductCapability} <: AbstractMutation
    capability::C
end

required_site_kinds(::ProductPackMutation) = Symbol[:app]
required_site_kinds(::ProductUnpackMutation) = Symbol[:app]

function _validated_product_heads(capability::AbstractProductCapability)
    heads = product_primitive_heads(capability)
    heads isa NamedTuple{(:pair, :left, :right)} || throw(ArgumentError(
        "product_primitive_heads must return (pair=..., left=..., right=...)",
    ))
    all(head -> head isa Symbol, heads) ||
        throw(ArgumentError("product primitive heads must be Symbols"))
    length(Set(heads)) == 3 ||
        throw(ArgumentError("product primitive heads must be distinct"))
    return heads
end

function _product_capability_available(
    capability::AbstractProductCapability,
    env::MutationEnv,
    ctx::NamedTuple,
)
    heads = _validated_product_heads(capability)
    values = haskey(ctx, :global_consts) ? ctx.global_consts : Dict{Symbol,Any}()
    local_values = haskey(ctx, :local_consts) ? ctx.local_consts : Dict{Symbol,Any}()
    return all(heads) do head
        haskey(env.type_ctx, head) &&
            (haskey(values, head) || haskey(local_values, head))
    end
end

_product_projection(head::Symbol, binder_id::Int) =
    _ScopedApp(_ScopedConst(head), _ScopedBound(binder_id))

_product_pair(head::Symbol, left::_ScopedTerm, right::_ScopedTerm) =
    _ScopedApp(_ScopedApp(_ScopedConst(head), left), right)

function _product_pack_parts(term::_ScopedTerm)
    bindings, body = _scoped_let_prefix(term, 2)
    length(bindings) == 2 || return nothing
    first_binding, second_binding = bindings
    _scoped_uses_binder(second_binding.rhs, first_binding.binder_id) &&
        return nothing
    return (first_binding, second_binding, body)
end

function _replace_packed_binders(
    term::_ScopedTerm,
    left_binder::Int,
    right_binder::Int,
    pair_binder::Int,
    heads::NamedTuple,
)
    if term isa _ScopedBound
        term.binder_id == left_binder &&
            return _product_projection(heads.left, pair_binder)
        term.binder_id == right_binder &&
            return _product_projection(heads.right, pair_binder)
        return term
    elseif term isa _ScopedAbs
        return _ScopedAbs(
            term.binder_id,
            term.name,
            _replace_packed_binders(
                term.body,
                left_binder,
                right_binder,
                pair_binder,
                heads,
            ),
        )
    elseif term isa _ScopedApp
        return _ScopedApp(
            _replace_packed_binders(
                term.func,
                left_binder,
                right_binder,
                pair_binder,
                heads,
            ),
            _replace_packed_binders(
                term.arg,
                left_binder,
                right_binder,
                pair_binder,
                heads,
            ),
        )
    end
    return term
end

function _product_pack_rewrite(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
    capability::AbstractProductCapability,
    binder_name::Symbol,
)
    term = _scoped_at_path(scoped, path)
    term isa Err && return term
    parts = _product_pack_parts(term)
    parts === nothing && return Err(NoEligibleSiteErr())
    left, right, body = parts
    heads = _validated_product_heads(capability)
    pair_binder = scoped.next_binder_id
    packed_body = _replace_packed_binders(
        body,
        left.binder_id,
        right.binder_id,
        pair_binder,
        heads,
    )
    packed = _ScopedApp(
        _ScopedAbs(pair_binder, binder_name, packed_body),
        _product_pair(heads.pair, left.rhs, right.rhs),
    )
    return _replace_scoped(scoped, path, packed)
end

function _product_pair_parts(term::_ScopedTerm, pair_head::Symbol)
    term isa _ScopedApp || return nothing
    left_application = term.func
    left_application isa _ScopedApp || return nothing
    left_application.func isa _ScopedConst || return nothing
    left_application.func.name == pair_head || return nothing
    return (left_application.arg, term.arg)
end

function _projection_side(term::_ScopedTerm, pair_binder::Int, heads::NamedTuple)
    term isa _ScopedApp || return nothing
    term.func isa _ScopedConst || return nothing
    term.arg isa _ScopedBound || return nothing
    term.arg.binder_id == pair_binder || return nothing
    term.func.name == heads.left && return :left
    term.func.name == heads.right && return :right
    return nothing
end

function _replace_product_projections(
    term::_ScopedTerm,
    pair_binder::Int,
    left_binder::Int,
    right_binder::Int,
    heads::NamedTuple,
)
    side = _projection_side(term, pair_binder, heads)
    side === :left && return (_ScopedBound(left_binder), 1, 0)
    side === :right && return (_ScopedBound(right_binder), 0, 1)
    term isa _ScopedBound && term.binder_id == pair_binder && return nothing
    if term isa _ScopedAbs
        body = _replace_product_projections(
            term.body,
            pair_binder,
            left_binder,
            right_binder,
            heads,
        )
        body === nothing && return nothing
        return (_ScopedAbs(term.binder_id, term.name, body[1]), body[2], body[3])
    elseif term isa _ScopedApp
        func = _replace_product_projections(
            term.func,
            pair_binder,
            left_binder,
            right_binder,
            heads,
        )
        func === nothing && return nothing
        arg = _replace_product_projections(
            term.arg,
            pair_binder,
            left_binder,
            right_binder,
            heads,
        )
        arg === nothing && return nothing
        return (_ScopedApp(func[1], arg[1]), func[2] + arg[2], func[3] + arg[3])
    end
    return (term, 0, 0)
end

function _product_unpack_parts(
    term::_ScopedTerm,
    capability::AbstractProductCapability,
    fresh_binder::Int,
)
    term isa _ScopedApp || return nothing
    term.func isa _ScopedAbs || return nothing
    heads = _validated_product_heads(capability)
    components = _product_pair_parts(term.arg, heads.pair)
    components === nothing && return nothing
    left_binder = fresh_binder
    right_binder = fresh_binder + 1
    body = _replace_product_projections(
        term.func.body,
        term.func.binder_id,
        left_binder,
        right_binder,
        heads,
    )
    body === nothing && return nothing
    return (components[1], components[2], body[1], left_binder, right_binder)
end

function _product_unpack_rewrite(
    scoped::_ScopedTree,
    path::AbstractVector{<:Integer},
    capability::AbstractProductCapability,
    left_name::Symbol,
    right_name::Symbol,
)
    term = _scoped_at_path(scoped, path)
    term isa Err && return term
    parts = _product_unpack_parts(
        term,
        capability,
        scoped.next_binder_id,
    )
    parts === nothing && return Err(NoEligibleSiteErr())
    left_rhs, right_rhs, body, left_binder, right_binder = parts
    unpacked = _ScopedApp(
        _ScopedAbs(
            left_binder,
            left_name,
            _ScopedApp(
                _ScopedAbs(right_binder, right_name, body),
                right_rhs,
            ),
        ),
        left_rhs,
    )
    return _replace_scoped(scoped, path, unpacked)
end

function is_site_eligible(
    mutation::ProductPackMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _product_capability_available(mutation.capability, env, ctx) || return false
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    term = _scoped_at_path(scoped, site.path)
    term isa Err && return false
    return _product_pack_parts(term) !== nothing
end

function is_site_eligible(
    mutation::ProductUnpackMutation,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _product_capability_available(mutation.capability, env, ctx) || return false
    scoped = _resolved_scoped(env)
    scoped isa Err && return false
    term = _scoped_at_path(scoped, site.path)
    term isa Err && return false
    return _product_unpack_parts(
        term,
        mutation.capability,
        scoped.next_binder_id,
    ) !== nothing
end

function apply(
    mutation::ProductPackMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _product_capability_available(mutation.capability, env, ctx) ||
        return Err(NoEligibleSiteErr())
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    rewritten = _product_pack_rewrite(
        scoped,
        site.path,
        mutation.capability,
        ctx.symfactory(),
    )
    rewritten isa Err && return rewritten
    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

function apply(
    mutation::ProductUnpackMutation,
    tree::AstTree,
    site::Site,
    env::MutationEnv,
    ctx::NamedTuple,
)
    _product_capability_available(mutation.capability, env, ctx) ||
        return Err(NoEligibleSiteErr())
    scoped = _resolved_scoped(env)
    scoped isa Err && return scoped
    rewritten = _product_unpack_rewrite(
        scoped,
        site.path,
        mutation.capability,
        ctx.symfactory(),
        ctx.symfactory(),
    )
    rewritten isa Err && return rewritten
    materialized = _materialize_scoped(rewritten)
    materialized isa Err && return materialized
    tree.root = materialized.root
    return tree
end

@testitem "Product packing mutation API is opt-in" begin
    using Test
    using LambdaRegression: Mutations
    using LambdaRegression.TypeSystem: FST, PAIR, SND

    @test isdefined(Mutations, :AbstractProductCapability)
    @test isdefined(Mutations, :PrimitiveProductCapability)
    @test isdefined(Mutations, :ProductPackMutation)
    @test isdefined(Mutations, :ProductUnpackMutation)
    @test_throws MethodError Mutations.ProductPackMutation()
    @test_throws MethodError Mutations.ProductUnpackMutation()
    @test_throws ArgumentError Mutations.PrimitiveProductCapability(
        pair_head=:PAIR,
        left_head=:PAIR,
    )
    @test all(mutation -> !(mutation isa Mutations.ProductPackMutation), Mutations._default_mutations())
    @test all(mutation -> !(mutation isa Mutations.ProductUnpackMutation), Mutations._default_mutations())

    @test PAIR(1)(true) == (1, true)
    @test PAIR(1, true) == (1, true)
    @test FST((1, true)) == 1
    @test SND((1, true)) == true
end

@testitem "Product packing round trips independent let bindings" begin
    using Test
    using Random
    using LambdaRegression: Mutations
    using LambdaRegression.ASTs.CSE: canonicalize
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FADD, FMUL, FST, PAIR, SND, FloatTy, TyLike, get_type_ctx, →
    using LambdaRegression.Utils: Err

    app2(head, left, right) = AppNode(AppNode(ConstNode(head), left), right)
    original = AstTree(
        AbsNode(
            :z,
            AppNode(
                AbsNode(
                    :x,
                    AppNode(
                        AbsNode(
                            :y,
                            app2(:FADD, IndexNode(1), IndexNode(0)),
                        ),
                        app2(:FMUL, IndexNode(1), ConstNode(:TWO)),
                    ),
                ),
                app2(:FADD, IndexNode(0), ConstNode(:ONE)),
            ),
        ),
    )
    values = Dict{Symbol,Any}(
        :MAKE_PAIR => PAIR,
        :LEFT_OF => FST,
        :RIGHT_OF => SND,
        :FADD => FADD,
        :FMUL => FMUL,
        :ONE => 1.0,
        :TWO => 2.0,
    )
    types = Dict{Symbol,TyLike}(get_type_ctx(values))
    target = FloatTy → FloatTy
    capability = Mutations.PrimitiveProductCapability(
        pair_head=:MAKE_PAIR,
        left_head=:LEFT_OF,
        right_head=:RIGHT_OF,
    )
    chooser = Mutations.WeightedPairChooser(
        weights=Dict{Type,Float64}(
            Mutations.ProductPackMutation => 1.0,
            Mutations.ProductUnpackMutation => 1.0,
        ),
        cache_eligibility=true,
    )

    packed = Mutations.mutate_tree(
        original,
        target;
        rng=Random.MersenneTwister(8000),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )

    @test !(packed isa Err)
    packed isa Err && return
    @test packed != original
    @test count(node -> node isa ConstNode && node.name == :MAKE_PAIR, packed) == 1
    @test count(node -> node isa ConstNode && node.name == :LEFT_OF, packed) == 1
    @test count(node -> node isa ConstNode && node.name == :RIGHT_OF, packed) == 1

    unpacked = Mutations.mutate_tree(
        packed,
        target;
        rng=Random.MersenneTwister(42),
        mutations=Mutations.AbstractMutation[Mutations.ProductUnpackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )

    @test !(unpacked isa Err)
    unpacked isa Err && return
    @test canonicalize(unpacked, unpacked.root) == canonicalize(original, original.root)

    repeated = Mutations.mutate_tree(
        original,
        target;
        rng=Random.MersenneTwister(8000),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test repeated == packed
end

@testitem "Product packing rejects dependencies and non-projection pair uses" begin
    using Test
    using Random
    using LambdaRegression: Mutations
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FST, PAIR, SND, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.Utils: Err

    values = Dict{Symbol,Any}(
        :PAIR => PAIR,
        :FST => FST,
        :SND => SND,
        :ONE => 1.0,
        :TWO => 2.0,
    )
    types = Dict{Symbol,TyLike}(get_type_ctx(values))
    capability = Mutations.PrimitiveProductCapability()
    chooser = Mutations.WeightedPairChooser(
        weights=Dict{Type,Float64}(
            Mutations.ProductPackMutation => 1.0,
            Mutations.ProductUnpackMutation => 1.0,
        ),
    )

    dependent = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(AbsNode(:y, IndexNode(0)), IndexNode(0)),
            ),
            ConstNode(:ONE),
        ),
    )
    dependent_before = deepcopy(dependent)
    rejected_pack = Mutations.mutate_tree(
        dependent,
        FloatTy;
        rng=Random.MersenneTwister(555),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test rejected_pack isa Err
    @test dependent == dependent_before

    bare_pair = AstTree(
        AppNode(
            AbsNode(:pair, IndexNode(0)),
            AppNode(
                AppNode(ConstNode(:PAIR), ConstNode(:ONE)),
                ConstNode(:TWO),
            ),
        ),
    )
    bare_before = deepcopy(bare_pair)
    rejected_unpack = Mutations.mutate_tree(
        bare_pair,
        LambdaRegression.TypeSystem.TyProd(FloatTy, FloatTy);
        rng=Random.MersenneTwister(2000),
        mutations=Mutations.AbstractMutation[Mutations.ProductUnpackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test rejected_unpack isa Err
    @test bare_pair == bare_before
end

@testitem "Product packing preserves independent polymorphic lets and component order" begin
    using Test
    using Random
    using LambdaRegression: Mutations
    using LambdaRegression.ASTs.CSE: canonicalize
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.TypeSystem: FALSE, FST, PAIR, SND, TRUE, BoolTy, FloatTy, TyLike, TyProd, get_type_ctx
    using LambdaRegression.Utils: Err

    app2(head, left, right) = AppNode(AppNode(ConstNode(head), left), right)
    original = AstTree(
        AppNode(
            AbsNode(
                :f,
                AppNode(
                    AbsNode(
                        :g,
                        app2(
                            :MAKE_PAIR,
                            app2(:MAKE_PAIR, AppNode(IndexNode(1), ConstNode(:ONE)), AppNode(IndexNode(1), ConstNode(:TRUE))),
                            app2(:MAKE_PAIR, AppNode(IndexNode(0), ConstNode(:TWO)), AppNode(IndexNode(0), ConstNode(:FALSE))),
                        ),
                    ),
                    AbsNode(:y, IndexNode(0)),
                ),
            ),
            AbsNode(:x, IndexNode(0)),
        ),
    )
    values = Dict{Symbol,Any}(
        :MAKE_PAIR => PAIR,
        :LEFT_OF => FST,
        :RIGHT_OF => SND,
        :ONE => 1.0,
        :TWO => 2.0,
        :TRUE => TRUE,
        :FALSE => FALSE,
    )
    types = Dict{Symbol,TyLike}(get_type_ctx(values))
    target = TyProd(TyProd(FloatTy, BoolTy), TyProd(FloatTy, BoolTy))
    capability = Mutations.PrimitiveProductCapability(
        pair_head=:MAKE_PAIR,
        left_head=:LEFT_OF,
        right_head=:RIGHT_OF,
    )
    chooser = Mutations.WeightedPairChooser(
        weights=Dict{Type,Float64}(
            Mutations.ProductPackMutation => 1.0,
            Mutations.ProductUnpackMutation => 1.0,
        ),
    )

    packed = Mutations.mutate_tree(
        original,
        target;
        rng=MersenneTwister(1),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test !(packed isa Err)
    packed isa Err && return
    @test count(node -> node isa ConstNode && node.name == :LEFT_OF, packed) == 2
    @test count(node -> node isa ConstNode && node.name == :RIGHT_OF, packed) == 2

    unpacked = Mutations.mutate_tree(
        packed,
        target;
        rng=MersenneTwister(1111),
        mutations=Mutations.AbstractMutation[Mutations.ProductUnpackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test !(unpacked isa Err)
    unpacked isa Err && return
    @test canonicalize(unpacked, unpacked.root) == canonicalize(original, original.root)
end

@testitem "Product packing documents strict-host evaluation behavior" begin
    using Test
    using Random
    using LambdaRegression: Mutations
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Reduces: beta_reduce!
    using LambdaRegression.TypeSystem: FST, PAIR, SND, IntTy, TyLike, get_type_ctx
    using LambdaRegression.Utils: Err

    omega_abs = AbsNode(:self, AppNode(IndexNode(0), IndexNode(0)))
    omega = AppNode(deepcopy(omega_abs), deepcopy(omega_abs))
    original = AstTree(
        AppNode(
            AbsNode(
                :x,
                AppNode(AbsNode(:unused, IndexNode(1)), omega),
            ),
            ConstNode(:ONE),
        ),
    )
    values = Dict{Symbol,Any}(:PAIR => PAIR, :FST => FST, :SND => SND, :ONE => 1)
    types = Dict{Symbol,TyLike}(get_type_ctx(values))
    capability = Mutations.PrimitiveProductCapability()
    chooser = Mutations.WeightedPairChooser(
        weights=Dict{Type,Float64}(Mutations.ProductPackMutation => 1.0),
    )

    packed = Mutations.mutate_tree(
        original,
        IntTy;
        rng=MersenneTwister(1234),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test !(packed isa Err)
    packed isa Err && return

    original_eval = deepcopy(original)
    @test beta_reduce!(original_eval, 8, copy(values))
    @test original_eval.root isa ConstNode
    @test original_eval.root.name == :ONE

    packed_eval = deepcopy(packed)
    @test !beta_reduce!(packed_eval, 8, copy(values))
    @test !(packed_eval.root isa ConstNode && packed_eval.root.name == :ONE)
end

@testitem "Product unpack restores a packed unused binding pair" begin
    using Test
    using Random
    using LambdaRegression: Mutations
    using LambdaRegression.ASTs.CSE: canonicalize
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode
    using LambdaRegression.TypeSystem: FST, PAIR, SND, FloatTy, TyLike, get_type_ctx
    using LambdaRegression.Utils: Err

    original = AstTree(
        AppNode(
            AbsNode(
                :left,
                AppNode(AbsNode(:right, ConstNode(:THREE)), ConstNode(:TWO)),
            ),
            ConstNode(:ONE),
        ),
    )
    values = Dict{Symbol,Any}(
        :PAIR => PAIR,
        :FST => FST,
        :SND => SND,
        :ONE => 1.0,
        :TWO => 2.0,
        :THREE => 3.0,
    )
    types = Dict{Symbol,TyLike}(get_type_ctx(values))
    capability = Mutations.PrimitiveProductCapability()
    chooser = Mutations.WeightedPairChooser(
        weights=Dict{Type,Float64}(
            Mutations.ProductPackMutation => 1.0,
            Mutations.ProductUnpackMutation => 1.0,
        ),
    )

    packed = Mutations.mutate_tree(
        original,
        FloatTy;
        rng=MersenneTwister(42),
        mutations=Mutations.AbstractMutation[Mutations.ProductPackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test !(packed isa Err)
    packed isa Err && return

    unpacked = Mutations.mutate_tree(
        packed,
        FloatTy;
        rng=MersenneTwister(555),
        mutations=Mutations.AbstractMutation[Mutations.ProductUnpackMutation(capability)],
        chooser,
        max_try=1,
        global_consts=values,
        global_consts_ty=types,
    )
    @test !(unpacked isa Err)
    unpacked isa Err && return
    @test canonicalize(unpacked, unpacked.root) == canonicalize(original, original.root)
end
