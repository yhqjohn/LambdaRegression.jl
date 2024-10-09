module ReduceModule

using ..DeBruijnModule: AbstractDBNode, VarNode, AbsNode, AppNode, ProxyNode, substitute
using MLStyle: @data

@data ReductionResult{E} begin
    Normalized()
    Ok(::AbstractDBNode)
    Err(E)
end

function beta_reduce(node::AbstractDBNode)
    @match node begin
        n::VarNode || n::ProxyNode => Normalized()
        AbsNode(index, body) => begin
            result = beta_reduce(body)
            @match result begin
                Normalized() => Normalized()
                Ok(body) => Ok(AbsNode(index, body))
                Err(e) => Err(e)
            end
        end
        AppNode(func, arg) => @match func begin
            ::AbsNode => substitute(func.body, arg)
            ::VarNode => Normalized()
            ::AppNode => begin
                result = beta_reduce(func)
                @match result begin
                    Err(e) => Err(e)
                    Ok(func) => Ok(AppNode(func, arg))
                    Normalized() => begin
                        result = beta_reduce(arg)
                        @match result begin
                            Err(e) => Err(e)
                            Ok(arg) => Ok(AppNode(func, arg))
                            Normalized() => Normalized()
                        end
                    end
                end
            end
        end           
    end
end


end # module ReduceModule