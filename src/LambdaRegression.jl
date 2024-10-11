module LambdaRegression

include("common_utils.jl")

include("parse.jl")

using .ParseModule: @lambda

include("debrujin.jl")
using .DeBruijnModule: to_debrujin

include("reduce.jl")


end # module LambdaRegression
