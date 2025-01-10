# using LambdaRegression, Test, SafeTestsets
using TestItemRunner

@run_package_tests filter=ti->(
    occursin("src\\", ti.filename) &&
    endswith(ti.filename, ".jl")
)

import LambdaRegression
if get(ENV, "LAMBDAREGRESSION_RUN_INTEGRATION", "") == "1" && isdefined(LambdaRegression, :Regression)
    include("integration/regression_smoke.jl")
end
