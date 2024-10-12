module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute
using LambdaRegression: Err, @check
using MLStyle

struct Normalized end
const normalized = Err(Normalized())


function beta_reduce(node::DBNode)
    @match node begin
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => AbsDB(@check beta_reduce(body))
        AppDB(func, arg) => @match func begin
            ::AbsDB => substitute(func, arg)
            ::VarDB => normalized
            ::AppDB => begin
                reduced_func = @check beta_reduce(func) Normalized # other Err cases are directly returned
                @match reduced_func begin # only Normalized Err is possible
                    Err(:Normalized) => AppDB(func, @check beta_reduce(arg))
                    _ => AppDB(reduced_func, arg)
                end
            end
            ::ProxyDB => @check substitute(func, arg)
        end           
    end
end


end # module ReduceModule