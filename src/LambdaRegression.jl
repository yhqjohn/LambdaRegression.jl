module LambdaRegression

include("common_utils.jl")

include("parse.jl")

using .ParseModule: @lambda
export @lambda

include("debruijn.jl")
using .DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse
export DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse


include("reduce.jl")
export beta_reduce, normalize

include("terms.jl")

include("builtin.jl")


end # module LambdaRegression
