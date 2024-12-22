# using LambdaRegression, Test, SafeTestsets
using TestItemRunner

@run_package_tests filter=ti->(occursin("src\\", ti.filename) && endswith(ti.filename, ".jl"))