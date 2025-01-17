module LambdaRegression

include("Utils.jl")
using .Utils
export Utils

include("trees/Trees.jl")
using .Trees
export Trees

include("asts/ASTs.jl")
using .ASTs
export ASTs

include("reduce/Reduce.jl")
using .Reduces
export Reduces

include("typesystem/TypeSystem.jl")
using .TypeSystem
export TypeSystem

include("generate/CanonicalGeneration.jl")
using .CanonicalGeneration
export CanonicalGeneration

include("mutate/Mutations.jl")
using .Mutations
export Mutations
using .Mutations: mutate_tree
export mutate_tree
#
include("regression/Regression.jl")
using .Regression
export Regression
using .Regression: Regressor, SupervisedDataset, FitResult, fit!, default_config, default_components, default_regressor
export Regressor, SupervisedDataset, FitResult, fit!, default_config, default_components, default_regressor

end # module LambdaRegression
