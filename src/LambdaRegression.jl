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

include("typesystem/TypeSystem.jl")
using .TypeSystem
export TypeSystem

include("search/SearchPolicies.jl")

include("programs/Programs.jl")
using .Programs
export Programs

include("analysis/ProgramAnalysis.jl")
using .ProgramAnalysis
export ProgramAnalysis

include("transforms/Transforms.jl")
using .Transforms
export Transforms

include("primitives/Primitives.jl")
using .Primitives
export Primitives

include("execution/Execution.jl")
using .Execution
export Execution

include("scalar/ScalarSearch.jl")
using .ScalarSearch
export ScalarSearch

include("factories/ValueFactories.jl")
using .ValueFactories
export ValueFactories

include("generate/CanonicalGeneration.jl")
using .CanonicalGeneration
export CanonicalGeneration

include("mutate/Mutations.jl")
using .Mutations
export Mutations
using .Mutations: mutate_tree
export mutate_tree

include("typedsearch/TypedSearch.jl")
export TypedSearch
include("regression/Regression.jl")
using .Regression
export Regression
using .Regression: Regressor, SupervisedDataset, default_config, default_components, default_regressor
using .Execution: PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
using .Regression: TensorSchema, TensorSeries, SeriesTask, series_task
using .Regression: series_mse, series_components, series_regressor
export Regressor, SupervisedDataset, default_config, default_components, default_regressor
export PreparedDefaultEvaluator, prepare_default_evaluator, evaluate_default
export TensorSchema, TensorSeries, SeriesTask, series_task
export series_mse, series_components, series_regressor

include("workflows/Workflows.jl")
using .Workflows
export Workflows

end # module LambdaRegression
