# using LambdaRegression, Test, SafeTestsets
using TestItemRunner

function _has_path_component(path::AbstractString, component::AbstractString)
    return component in split(replace(normpath(path), '\\' => '/'), '/')
end

@run_package_tests filter=ti->(
    _has_path_component(ti.filename, "src") &&
    !_has_path_component(ti.filename, ".worktrees") &&
    endswith(ti.filename, ".jl")
)

import LambdaRegression
