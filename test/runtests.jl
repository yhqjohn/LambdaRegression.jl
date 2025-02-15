using TestItemRunner

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const GENERATION_BATCHES = 5

function _has_path_component(path::AbstractString, component::AbstractString)
    return component in split(replace(normpath(path), '\\' => '/'), '/')
end

function _eligible_testitem(ti)
    return _has_path_component(ti.filename, "src") &&
           !_has_path_component(ti.filename, ".worktrees") &&
           endswith(ti.filename, ".jl")
end

function _stable_bucket(name, count::Int)
    value = UInt(0)
    for byte in codeunits(String(name))
        value = value * UInt(33) + UInt(byte)
    end
    return Int(rem(value, UInt(count)))
end

function _test_batch(ti)
    if _has_path_component(ti.filename, "generate")
        return (:generate, _stable_bucket(ti.name, GENERATION_BATCHES))
    elseif _has_path_component(ti.filename, "typesystem")
        return :typesystem
    elseif _has_path_component(ti.filename, "asts") ||
           _has_path_component(ti.filename, "trees")
        return :structure
    elseif _has_path_component(ti.filename, "factories") ||
           _has_path_component(ti.filename, "mutate")
        return :evolution
    end
    return :misc
end

const TEST_BATCHES = (
    :typesystem,
    :structure,
    :evolution,
    :misc,
    (ntuple(index -> (:generate, index - 1), GENERATION_BATCHES))...,
)

for batch in TEST_BATCHES
    TestItemRunner.run_tests(
        PACKAGE_ROOT;
        filter=ti -> _eligible_testitem(ti) && _test_batch(ti) == batch,
    )
    GC.gc(true)
end

import LambdaRegression
