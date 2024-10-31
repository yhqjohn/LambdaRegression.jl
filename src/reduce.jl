module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute, Normalized, normalized
using LambdaRegression: Err, @check
using MLStyle

function beta_reduce(node::DBNode, context=nothing)
    @match node begin
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => AbsDB(@check beta_reduce(body, context))
        AppDB(func, arg) => @match func begin
            ::AbsDB => substitute(func, arg, context)
            ::VarDB => normalized
            ::AppDB => begin
                reduced_func = @check beta_reduce(func, context) Normalized # other Err cases are directly returned
                @match reduced_func begin # only Normalized Err is possible
                    Err(:Normalized) => AppDB(func, @check beta_reduce(arg, context))
                    _ => AppDB(reduced_func, arg)
                end
            end
            ::ProxyDB => @check substitute(func, arg, context)
        end           
    end
end


struct NormalizationLimitExceeded end
const normalization_limit_exceeded = Err(NormalizationLimitExceeded())
function normalize(node::DBNode, context=nothing; max_iter=100)
    for _ in 1:max_iter
        node = @check beta_reduce(node, context) Normalized
        if node isa Normalized
            return node
        end
    end
    normalization_limit_exceeded
end


end # module ReduceModule