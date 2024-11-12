module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute
using LambdaRegression: Err, @check
using MLStyle


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

function beta_reduce(node::DBNode{T}) where {T}
    @match node begin
        AppDB(func, arg) => @match func begin
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
                    n::DBNode => AppDB{T}(n, arg)
                    e::Err => e
                end
                n::DBNode => AppDB{T}(n, arg)
                e::Err => e
            end
        end
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => @match beta_reduce(body) begin
            n::DBNode => AbsDB{T}(n)
            e::Err => e
        end
    end
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