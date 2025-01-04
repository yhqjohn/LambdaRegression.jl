abstract type AbstractConstCtor <: AbstractLeafCtor end

struct GlobalConstCtor <: AbstractConstCtor end
function initialize_ctx(::GlobalConstCtor, ctx::NamedTuple)
    @assert haskey(ctx, :global_consts_ty) "`:global_consts_ty` is not found in the context. This must be provided by the user."
    return (;)
end
isqualified(ty::Ty, ::GlobalConstCtor, ctx::NamedTuple) = any(iscompatible(ty, ty_const) for ty_const in values(ctx.global_consts_ty))

function construct_tree(::GlobalConstCtor, ty::Ty, ctx::NamedTuple)
    global_consts_ty = ctx.global_consts_ty
    qualified_names = filter(name->iscompatible(ty, global_consts_ty[name]), keys(global_consts_ty))
    const_name = rand(ctx.rng, qualified_names)
    return AstTree(ConstNode(const_name))
end

struct LocalConstCtor <: AbstractConstCtor end
function initialize_ctx(::LocalConstCtor, ctx::NamedTuple)
    @assert haskey(ctx, :local_consts) "`:local_consts` is not found in the context. This must be provided by the user."
    @assert haskey(ctx, :local_consts_ty) "`:local_consts_ty` is not found in the context. This must be provided by the user."
    return (;)
end
isqualified(ty::Ty, ::LocalConstCtor, ctx::NamedTuple) = any(iscompatible(ty, ty_const) for ty_const in values(ctx.local_consts_ty))
function construct_tree(::LocalConstCtor, ty::Ty, ctx::NamedTuple)
    local_consts_ty = ctx.local_consts_ty
    qualified_names = filter(name->iscompatible(ty, local_consts_ty[name]), keys(local_consts_ty))
    const_name = rand(ctx.rng, qualified_names)
    return AstTree(ConstNode(const_name))
end

function _initialize_local_consts(ctx::NamedTuple)
    local_consts_ty = haskey(ctx, :local_consts_ty) ? (;) : (; local_consts_ty=Dict{Symbol, TyLike}())
    local_consts = haskey(ctx, :local_consts) ? (;) : (; local_consts=Dict{Symbol, Any}())
    return merge(local_consts_ty, local_consts)
end

function _new_typed_constant!(ctx::NamedTuple, value, ty::Ty)
    local_consts_ty = ctx.local_consts_ty
    local_consts = ctx.local_consts
    node = new_constant!(local_consts, value)
    local_consts_ty[node.name] = ty
    return AstTree(node)
end

struct IntegerCtor <: AbstractConstCtor
    range::UnitRange{Int}
end
IntegerCtor() = IntegerCtor(1:100)
initialize_ctx(::IntegerCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::IntegerCtor, ctx::NamedTuple) = ty == TypeSystem.IntTy
construct_tree(ctor::IntegerCtor, ::Ty, ctx::NamedTuple) = _new_typed_constant!(ctx, rand(ctx.rng, ctor.range), TypeSystem.IntTy)

struct FloatCtor <: AbstractConstCtor
    mean::Float64
    std::Float64
end
FloatCtor() = FloatCtor(0.0, 1.0)
initialize_ctx(::FloatCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::FloatCtor, ctx::NamedTuple) = ty == TypeSystem.FloatTy
construct_tree(ctor::FloatCtor, ::Ty, ctx::NamedTuple) = _new_typed_constant!(ctx, randn(ctx.rng) * ctor.std + ctor.mean, TypeSystem.FloatTy)

struct BoolCtor <: AbstractConstCtor end
initialize_ctx(::BoolCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::BoolCtor, ctx::NamedTuple) = ty == TypeSystem.BoolTy
construct_tree(::BoolCtor, ::Ty, ctx::NamedTuple) = _new_typed_constant!(ctx, rand(ctx.rng) < 0.5, TypeSystem.BoolTy)

struct NilCtor <: AbstractConstCtor end
initialize_ctx(::NilCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::NilCtor, ctx::NamedTuple) = ty isa TyList
construct_tree(::NilCtor, ty::TyList, ctx::NamedTuple) = _new_typed_constant!(ctx, [], ty)

struct FloatArrCtor <: AbstractConstCtor
    mean::Float64
    std::Float64
end
FloatArrCtor() = FloatArrCtor(0.0, 1.0)
initialize_ctx(::FloatArrCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::FloatArrCtor, ctx::NamedTuple) = ty isa TyArray && ty.len isa TypeSystem.TyNat && ty.elem == TypeSystem.FloatTy
construct_tree(ctor::FloatArrCtor, ty::TyArray, ctx::NamedTuple) = _new_typed_constant!(ctx, randn(ctx.rng, ty.len.value) .* ctor.std .+ ctor.mean, ty)

struct IntArrCtor <: AbstractConstCtor
    range::UnitRange{Int}
end
IntArrCtor() = IntArrCtor(1:100)
initialize_ctx(::IntArrCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::IntArrCtor, ctx::NamedTuple) = ty isa TyArray && ty.len isa TypeSystem.TyNat && ty.elem == TypeSystem.IntTy
construct_tree(ctor::IntArrCtor, ty::TyArray, ctx::NamedTuple) = _new_typed_constant!(ctx, rand(ctx.rng, ctor.range, ty.len.value), ty)

struct BoolArrCtor <: AbstractConstCtor end
initialize_ctx(::BoolArrCtor, ctx::NamedTuple) = _initialize_local_consts(ctx)
isqualified(ty::Ty, ::BoolArrCtor, ctx::NamedTuple) = ty isa TyArray && ty.len isa TypeSystem.TyNat && ty.elem == TypeSystem.BoolTy
construct_tree(::BoolArrCtor, ty::TyArray, ctx::NamedTuple) = _new_typed_constant!(ctx, rand(ctx.rng, ty.len.value) .< 0.5, ty)
