module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute, Normalized, normalized
using LambdaRegression: Err, @check
using MLStyle
using TestItems


# function external_reduce(node::DBNode)
#     @match node begin
#         ::VarDB || ::ProxyDB => normalized
#         AbsDB(body) => AbsDB(@check external_reduce(body))
#         AppDB(func::ProxyDB, arg::ProxyDB) => ProxyDB(@check func.value(arg.value))
#         AppDB(func, arg) => begin
#             reduced_func = @check external_reduce(func) Normalized
#             reduced_arg = @check external_reduce(arg) Normalized
#             reduced_func === normalized && reduced_arg === normalized && return normalized
#             reduced_func = reduced_func === normalized ? func : reduced_func
#             reduced_arg = reduced_arg === normalized ? arg : reduced_arg
#             AppDB(reduced_func, reduced_arg)
#         end
#     end
# end

# function external_normalize(node::DBNode; max_iter=100)
#     for _ in 1:max_iter
#         reduced = @check external_reduce(node) Normalized
#         if reduced === normalized
#             return node
#         end
#         node = reduced
#     end
#     normalization_limit_exceeded
# end

beta_reduce(node::DBNode{T}) where {T} = beta_reduce_base(node)

function beta_reduce_base(node::DBNode{T}) where {T}
    @match node begin
        AppDB(func, arg) => beta_reduce_app(func, arg)
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => @match beta_reduce(body) begin
            n::DBNode => AbsDB{T}(n)
            e::Err => e
        end
    end
end

function beta_reduce_app_base(func::DBNode{T}, arg::DBNode{T}) where {T}
    @match func begin
        ::AbsDB => substitute(func, arg)
        ::ProxyDB => @match substitute(func, arg) begin
            n::DBNode => n
            ::Err{Normalized} => @match beta_reduce(arg) begin
                n::DBNode => AppDB{T}(func, n)
                e::Err => e
            end
            e::Err => e
            other => error("unexpected result: $other")
        end
        ::AppDB => @match beta_reduce(func) begin
            ::Err{Normalized} => @match beta_reduce(arg) begin
                new_arg::DBNode => AppDB{T}(func, new_arg)
                e::Err => e
            end
            new_func::DBNode => AppDB{T}(new_func, arg)
        end
    end
end
beta_reduce_app(func::DBNode{T}, arg::DBNode{T}) where {T} = beta_reduce_app_base(func, arg)


@testitem "beta reduce" begin
    import LambdaRegression.ReduceModule: beta_reduce, normalized

    # test basic substitution
    ident1 = DBNode{Int}(@lambda (x->x)*1)
    @test beta_reduce(ident1) === ProxyDB(1)
    @test beta_reduce(beta_reduce(ident1)) === normalized

    # test reduce order
    ident = DBNode{Int}(@lambda x->x)
    replicate1 = DBNode{Int}(@lambda x->(x*x))
    @test beta_reduce(ident(replicate1(ident))) === replicate1(ident)
    @test beta_reduce(ident(replicate1)(ident)) === replicate1(ident)
    @test beta_reduce(ProxyDB(1)(replicate1(ident))) === ProxyDB(1)(beta_reduce(replicate1(ident)))
    any_to_ident1 = DBNode{Int}(@lambda y->((x->x)*1))
    @test beta_reduce(any_to_ident1) === DBNode{Int}(@lambda y->1)
    nonsense_on_replicate_ident = DBNode{Int}(@lambda (1*1)*((x->(x*x))*(x->x)))
    display(nonsense_on_replicate_ident)
    display(beta_reduce(nonsense_on_replicate_ident))
    display(DBNode{Int}(@lambda (1*1)*((x->x)*(x->x))))
    @test beta_reduce(nonsense_on_replicate_ident) === DBNode{Int}(@lambda (1*1)*((x->x)*(x->x)))
end


struct NormalizationLimitExceeded end
const normalization_limit_exceeded = Err(NormalizationLimitExceeded())
function normalize(node::DBNode; max_iter=100, normalize_external=false, max_iter_external=100)
    node = node
    for _ in 1:max_iter
        # if normalize_external
        #     node = @check external_normalize(node; max_iter=max_iter_external)
        # end
        reduced = @check beta_reduce(node) Normalized
        if reduced === normalized
            return node
        end
        node = reduced
    end
    normalization_limit_exceeded
end


end # module ReduceModule