module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute
using LambdaRegression: Err, @check
using MLStyle

struct Normalized end
const normalized = Err(Normalized())


function beta_reduce(node::DBNode, eval=false)
    @match node begin
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => AbsDB(@check beta_reduce(body, eval))
        AppDB(func, arg) => @match func begin
            ::AbsDB => substitute(func, arg, eval)
            ::VarDB => normalized
            ::AppDB => begin
                reduced_func = @check beta_reduce(func) Normalized # other Err cases are directly returned
                @match reduced_func begin # only Normalized Err is possible
                    Err(:Normalized) => AppDB(func, @check beta_reduce(arg, eval))
                    _ => AppDB(reduced_func, arg)
                end
            end
            ::ProxyDB => @check substitute(func, arg, eval)
        end           
    end
end


struct NormalizationLimitExceeded end
const normalization_limit_exceeded = Err(NormalizationLimitExceeded())
function normalize(node::DBNode, eval=false; max_iter=100)
    for _ in 1:max_iter
        node = @check beta_reduce(node, eval) Normalized
        if node isa Normalized
            return node
        end
    end
    normalization_limit_exceeded
end


end # module ReduceModule