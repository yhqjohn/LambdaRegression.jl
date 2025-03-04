using TestItemRunner

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE_ROOT = normpath(joinpath(PACKAGE_ROOT, "src"))
const SOURCE_PREFIX = SOURCE_ROOT * string(Sys.iswindows() ? '\\' : '/')
const GENERATION_BATCHES = 5

push!(LOAD_PATH, PACKAGE_ROOT)

function _eligible_testitem(ti)
    filename = normpath(ti.filename)
    return startswith(filename, SOURCE_PREFIX) && endswith(filename, ".jl")
end

function _stable_bucket(name, count::Int)
    value = UInt(0)
    for byte in codeunits(String(name))
        value = value * UInt(33) + UInt(byte)
    end
    return Int(rem(value, UInt(count)))
end

const SOURCE_MODULES = sort!(String[
    basename(path) for path in readdir(SOURCE_ROOT; join=true) if isdir(path)
])
const MODULE_BATCH_CLASS = Dict{String,Symbol}(
    "asts" => :structure,
    "factories" => :evolution,
    "generate" => :generate,
    "mutate" => :evolution,
    "trees" => :structure,
    "typesystem" => :typesystem,
)

function _source_module(filename)
    filename = normpath(filename)
    startswith(filename, SOURCE_PREFIX) || return nothing
    relative = replace(relpath(filename, SOURCE_ROOT), '\\' => '/')
    module_name = first(split(relative, '/'))
    return module_name in SOURCE_MODULES ? module_name : nothing
end

function _test_batch(ti)
    module_name = _source_module(ti.filename)
    module_batch = get(MODULE_BATCH_CLASS, module_name, :misc)
    if module_batch === :generate
        return (:generate, _stable_bucket(ti.name, GENERATION_BATCHES))
    end
    return module_batch
end

const TEST_BATCHES = (
    :typesystem,
    :structure,
    :evolution,
    :misc,
    (ntuple(index -> (:generate, index - 1), GENERATION_BATCHES))...,
)

function _module_batches(module_name)
    module_batch = get(MODULE_BATCH_CLASS, module_name, :misc)
    module_batch === :generate &&
        return Any[(:generate, index) for index in 0:(GENERATION_BATCHES - 1)]
    return Any[module_batch]
end

function _parse_selection(args)
    modules = String[]
    name = nothing
    for argument in args
        if startswith(argument, "--module=")
            module_name = argument[length("--module=") + 1:end]
            isempty(module_name) && throw(ArgumentError(
                "--module= requires a module name"))
            module_name in SOURCE_MODULES || throw(ArgumentError(
                "unknown test module '$module_name'; valid modules are " *
                join(SOURCE_MODULES, ", ")))
            module_name in modules || push!(modules, module_name)
        elseif startswith(argument, "--name=")
            test_name = argument[length("--name=") + 1:end]
            isempty(test_name) && throw(ArgumentError(
                "--name= requires an exact test name"))
            name === nothing || throw(ArgumentError(
                "--name= may be supplied only once"))
            name = test_name
        elseif argument == "--module" || argument == "--name"
            throw(ArgumentError(
                "use --module=NAME and --name=EXACT (equals sign required)"))
        elseif startswith(argument, "--")
            throw(ArgumentError("unknown test selector '$argument'"))
        else
            throw(ArgumentError("unexpected test argument '$argument'"))
        end
    end
    return (; modules, name)
end

function _selected_batches(selection)
    selection.name !== nothing && return Any[:selected]
    isempty(selection.modules) && return collect(TEST_BATCHES)
    required = Set{Any}()
    for module_name in selection.modules
        union!(required, _module_batches(module_name))
    end
    return [batch for batch in TEST_BATCHES if batch in required]
end

function _batch_label(batch)
    batch isa Tuple && return "$(batch[1])[$(batch[2])]"
    return String(batch)
end

function _selected_testitem(ti, selection, batch, matched)
    _eligible_testitem(ti) || return false
    batch === :selected || _test_batch(ti) == batch || return false
    if !isempty(selection.modules)
        filename = normpath(ti.filename)
        any(module_name -> _source_module(filename) == module_name, selection.modules) ||
            return false
    end
    selection.name === nothing || ti.name == selection.name || return false
    matched[] += 1
    return true
end

function _run_selected_tests(selection)
    batches = _selected_batches(selection)
    matched = Ref(0)
    scope_modules = isempty(selection.modules) ? "all" : join(selection.modules, ",")
    scope_name = selection.name === nothing ? "<any>" : repr(selection.name)
    println("TEST_SCOPE modules=$(scope_modules) name=$(scope_name) batches=$(length(batches))")
    flush(stdout)

    for batch in batches
        label = _batch_label(batch)
        println("TEST_BATCH_BEGIN batch=$(label)")
        flush(stdout)
        started = time_ns()
        status = "failed"
        try
            TestItemRunner.run_tests(
                PACKAGE_ROOT;
                filter=ti -> _selected_testitem(ti, selection, batch, matched),
            )
            status = "passed"
        finally
            gc_started = time_ns()
            GC.gc(true)
            gc_elapsed = (time_ns() - gc_started) / 1e9
            elapsed = (time_ns() - started) / 1e9
            println("TEST_BATCH_END batch=$(label) status=$(status) " *
                    "elapsed_seconds=$(elapsed) gc_seconds=$(gc_elapsed)")
            flush(stdout)
        end
    end

    if matched[] == 0
        throw(ArgumentError("test selector matched no source test items"))
    end
    println("TEST_SCOPE_DONE matched=$(matched[])")
    flush(stdout)
    return nothing
end

_run_selected_tests(_parse_selection(ARGS))

import LambdaRegression
