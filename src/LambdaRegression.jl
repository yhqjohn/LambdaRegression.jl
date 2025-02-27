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

include("scalar/ScalarSearch.jl")
using .ScalarSearch
export ScalarSearch

include("factories/ValueFactories.jl")
using .ValueFactories
export ValueFactories

include("generate/CanonicalGeneration.jl")
using .CanonicalGeneration
export CanonicalGeneration

include("procedural/Procedural.jl")
using .Procedural
export Procedural

include("mutate/Mutations.jl")
using .Mutations
export Mutations
using .Mutations: mutate_tree
export mutate_tree

include("procedural/ProceduralMutations.jl")
using .ProceduralMutations
export ProceduralMutations
#
include("regression/Regression.jl")
using .Regression
export Regression
using .Regression: Regressor, SupervisedDataset, FitResult, fit!, default_config, default_components, default_regressor
using .Regression: PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
using .Regression: TensorSchema, TensorSeries, SeriesTask, series_task
using .Regression: series_mse, series_components, series_regressor
export Regressor, SupervisedDataset, FitResult, fit!, default_config, default_components, default_regressor
export PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
export TensorSchema, TensorSeries, SeriesTask, series_task
export series_mse, series_components, series_regressor

end # module LambdaRegression
