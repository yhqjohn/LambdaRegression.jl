module LambdaRegression

include("common_utils.jl")

include("parse.jl")

using .ParseModule: @lambda
export @lambda

include("debrujin.jl")
using .DeBruijnModule: to_debrujin
export to_debrujin

include("reduce.jl")
export beta_reduce, normalize

# include("labeledproxy.jl")

# include("builtin.jl")
end # module LambdaRegression
