module LambdaRegression

include("common_utils.jl")

include("Parse.jl")

using .ParseModule: @lambda
export @lambda

# include("Debruijn.jl")
# using .DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse
# export DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse

include("Debruijn.jl")


# include("Reduce.jl")
# export beta_reduce, normalize

# include("Terms.jl")

# include("BuiltinTerms.jl")

# include("Mutations.jl")

# include("Regressor.jl")


end # module LambdaRegression
