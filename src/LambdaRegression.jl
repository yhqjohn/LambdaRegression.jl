module LambdaRegression

include("common_utils.jl")

include("parse.jl")

using .ParseModule: @lambda

include("debrujin.jl")
using .DeBruijnModule: to_debrujin

include("reduce.jl")

include("labeledproxy.jl")

# include("builtin.jl")
end # module LambdaRegression
