module ReduceModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, substitute
using LambdaRegression: Result, Ok, Err, @check
using MLStyle

struct Normalized end
ReduceErr = Union{Normalized}
ReduceResult = Result{DBNode, ReduceErr}
const normalized = Err(Normalized())


function beta_reduce(node::DBNode)
    @match node begin
        n::VarDB || n::ProxyDB => normalized
        AbsDB(body) => begin
            result = beta_reduce(body)
            Ok(AbsDB(@check result))
        end
        AppDB(func, arg) => @match func begin
            ::AbsDB => substitute(func, arg)
            ::VarDB => normalized
            ::AppDB => begin
                reduced_func = @check beta_reduce(func) Normalized # other Err cases are directly returned
                @match reduced_func begin # only Normalized Err is possible
                    Ok(func) => Ok(AppDB(func, arg))
                    Err(::Normalized) => Ok(AppDB(func, @check beta_reduce(arg)))
                end
            end
        end           
    end
end


end # module ReduceModule