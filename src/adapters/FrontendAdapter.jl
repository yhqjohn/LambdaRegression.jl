# Collection axis 1 indexes examples; remaining axes form one complete sample.
module FrontendAdapter

using Random: AbstractRNG, MersenneTwister, default_rng, rand, randn
using TestItems

import ..ScalarSearch
using ..ScalarSearch: ScalarExpr, ScalarSearchConfig, ScalarSearchResult,
    scalar_input, scalar_constant, scalar_unary, scalar_binary,
    scalar_complexity, evaluate_scalar, search_scalar
using ..Utils: Err

export AdapterModel, fit, predict, get_formula, get_complexity
export generate_synthetic_data, generate_pretraining_tasks

function _resolve_task_kind(X, task_kind)
    kind = Symbol(task_kind)
    kind in (:auto, :scalar, :array, :series) || throw(ArgumentError(
        "task_kind must be auto, scalar, array, or series"))
    kind !== :auto && return kind
    X isa AbstractArray{<:Real} && return ndims(X) <= 2 ? :scalar : :array
    return :series
end

# :array keeps each sample intact; :series treats its first axis as a list.
# :auto selects scalar fitting for dense inputs of rank <= 2.
function fit(X, y; task_kind=:auto, kwargs...)
    kind = _resolve_task_kind(X, task_kind)
    return kind === :scalar ? _fit_scalar(X, y; kwargs...) :
        fit_collection(X, y; task_kind=kind, kwargs...)
end

const _ALLOWED_UNARY_OPERATORS = Set((:sin, :cos, :tan, :exp, :log, :abs, :sqrt, :square))
const _ALLOWED_BINARY_OPERATORS = Set((:+, :-, :*, :/, :^, :max, :min))
const _DEFAULT_ADAPTER_OPERATOR_SET = (:+, :-, :*, :/, :sin, :cos, :sqrt, :square)
const _SEARCH_CONFIG_FIELDS = Set(fieldnames(ScalarSearchConfig))
const _SEARCH_KEYWORDS = Set((:stop_loss, :initial_population, :objective))

struct AdapterModel
    expressions::Vector{ScalarExpr}
    feature_count::Int
    losses::Vector{Float64}
    has_output_axis::Bool
end

struct _ProgramSamplingErr <: Exception
    task_index::Int
    output_index::Int
    attempts_used::Int
    per_program_limit::Int
    total_limit::Union{Nothing,Int}
end

function Base.showerror(io::IO, error::_ProgramSamplingErr)
    print(io, "could not sample a finite program for task ", error.task_index,
        ", output ", error.output_index, " after ", error.attempts_used,
        " total attempts (per-program limit ", error.per_program_limit, ")")
    error.total_limit === nothing || print(io, " with total cap ", error.total_limit)
    print(io, "; input data may not support a distinct finite program with nonzero output variation")
end

function _derive_seed(seed::UInt64, stream::Integer)
    return seed + UInt64(stream) * 0x9e3779b97f4a7c15
end

function _base_seed(random_state)
    if random_state === nothing
        return rand(default_rng(), UInt64)
    elseif random_state isa Integer
        random_state >= 0 || throw(ArgumentError("random_state must be nonnegative or nothing"))
        return UInt64(random_state)
    end
    throw(ArgumentError("random_state must be an integer or nothing"))
end

_rng(seed::UInt64, stream::Integer) = MersenneTwister(_derive_seed(seed, stream))

function _as_symbols(value, name::Symbol)
    items = value isa Symbol ? (value,) :
        value isa AbstractString ? (Symbol(value),) : Tuple(Symbol(item) for item in value)
    length(unique(items)) == length(items) || throw(ArgumentError("$name must not contain duplicates"))
    return items
end

function _split_operator_set(operator_set)
    symbols = _as_symbols(operator_set, :operator_set)
    unknown = filter(op -> !(op in _ALLOWED_UNARY_OPERATORS || op in _ALLOWED_BINARY_OPERATORS), symbols)
    isempty(unknown) || throw(ArgumentError("unsupported operator_set entries: $(join(string.(unknown), ", "))"))
    unary = Tuple(op for op in symbols if op in _ALLOWED_UNARY_OPERATORS)
    binary = Tuple(op for op in symbols if op in _ALLOWED_BINARY_OPERATORS)
    return unary, binary
end

function _dict_from_kwargs(kwargs)
    return Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(kwargs))
end

function _normalize_search_kwargs(value)
    value isa NamedTuple || value isa AbstractDict || throw(ArgumentError(
        "search_kwargs must be a NamedTuple or dictionary"))
    options = Dict{Symbol,Any}((key isa Symbol ? key : Symbol(String(key))) => item
        for (key, item) in pairs(value))
    unknown = setdiff(Set(keys(options)), _SEARCH_KEYWORDS)
    isempty(unknown) || throw(ArgumentError(
        "unsupported search_kwargs: $(join(string.(sort!(collect(unknown))), ", "))"))
    return options
end

function _fit_config(kwargs)
    options = _dict_from_kwargs(kwargs)
    search_kwargs = _normalize_search_kwargs(pop!(options, :search_kwargs, NamedTuple()))
    random_state = pop!(options, :random_state, 0)
    generations = pop!(options, :generations, 40)
    generations isa Integer && generations >= 0 || throw(ArgumentError(
        "generations must be a nonnegative integer"))
    population_size = get(options, :population_size, 64)
    population_size isa Integer && population_size >= 2 || throw(ArgumentError(
        "population_size must be an integer of at least 2"))

    for (legacy, current) in ((:mutation_rate, :mutation_probability),
            (:crossover_rate, :crossover_probability))
        if haskey(options, legacy)
            value = pop!(options, legacy)
            get!(options, current, value)
        end
    end

    if !haskey(options, :operator_set) && !haskey(options, :unary_operators) &&
            !haskey(options, :binary_operators)
        options[:operator_set] = _DEFAULT_ADAPTER_OPERATOR_SET
    end
    if haskey(options, :operator_set)
        unary, binary = _split_operator_set(pop!(options, :operator_set))
        get!(options, :unary_operators, unary)
        get!(options, :binary_operators, binary)
    end

    for name in (:evolution_strategy, :constant_refinement_timing,
            :constant_refinement_optimizer)
        if haskey(options, name) && options[name] isa AbstractString
            options[name] = Symbol(options[name])
        end
    end

    config_fields = setdiff(Set(keys(options)), _SEARCH_CONFIG_FIELDS)
    isempty(config_fields) || throw(ArgumentError(
        "unsupported fit keyword(s): $(join(string.(sort!(collect(config_fields))), ", "))"))

    haskey(options, :max_evaluations) ||
        (options[:max_evaluations] = Int(population_size) * (Int(generations) + 1))
    options[:population_size] = Int(population_size)
    options[:tournament_size] = get(options, :tournament_size, min(5, Int(population_size)))
    options[:elite_count] = get(options, :elite_count, min(4, Int(population_size) - 1))
    config = ScalarSearchConfig(; options...)
    return config, _base_seed(random_state), search_kwargs
end

function _input_matrix(X::AbstractArray{<:Real})
    ndims(X) >= 1 || throw(ArgumentError("X must have a feature axis"))
    feature_count = size(X, ndims(X))
    feature_count >= 1 || throw(ArgumentError("X must have at least one feature"))
    all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
    sample_shape = Tuple(size(X)[1:end-1])
    sample_count = isempty(sample_shape) ? 1 : prod(sample_shape)
    values = Array{Float64}(X)
    permutation = (ndims(values), ntuple(identity, ndims(values) - 1)...)
    matrix = reshape(permutedims(values, permutation), feature_count, sample_count)
    return matrix, sample_shape, feature_count
end

function _target_matrix(y, sample_shape::Tuple, sample_count::Int)
    y isa Real && isfinite(y) || y isa AbstractArray{<:Real} || throw(ArgumentError(
        "y must be a finite real scalar or real array"))
    has_output_axis = false
    values = y isa Real ? reshape([Float64(y)], 1, 1) : begin
        all(isfinite, y) || throw(ArgumentError("y must contain only finite values"))
        array = Array{Float64}(y)
        if size(array) == sample_shape
            reshape(array, sample_count, 1)
        elseif ndims(array) == length(sample_shape) + 1 &&
                Tuple(size(array)[1:end-1]) == sample_shape
            has_output_axis = true
            reshape(array, sample_count, size(array, ndims(array)))
        elseif isempty(sample_shape) && ndims(array) == 1
            has_output_axis = true
            reshape(array, 1, length(array))
        else
            throw(DimensionMismatch(
                "y must match X's leading sample dimensions, with an optional final output-channel axis"))
        end
    end
    size(values, 1) == sample_count || throw(DimensionMismatch("y sample count must match X"))
    size(values, 2) >= 1 || throw(ArgumentError("y must contain at least one output channel"))
    return values, has_output_axis
end

function _required_feature(expr::ScalarExpr)
    if expr isa ScalarSearch._ScalarInput
        return getfield(expr, :index)
    elseif expr isa ScalarSearch._ScalarConstant
        return 0
    elseif expr isa ScalarSearch._ScalarUnary
        return _required_feature(getfield(expr, :arg))
    elseif expr isa ScalarSearch._ScalarBinary
        return max(_required_feature(getfield(expr, :left)), _required_feature(getfield(expr, :right)))
    end
    throw(ArgumentError("unsupported ScalarExpr implementation: $(typeof(expr))"))
end

function _expression_depth(expr::ScalarExpr)
    if expr isa ScalarSearch._ScalarInput || expr isa ScalarSearch._ScalarConstant
        return 0
    elseif expr isa ScalarSearch._ScalarUnary
        return 1 + _expression_depth(getfield(expr, :arg))
    elseif expr isa ScalarSearch._ScalarBinary
        return 1 + max(_expression_depth(getfield(expr, :left)),
            _expression_depth(getfield(expr, :right)))
    end
    throw(ArgumentError("unsupported ScalarExpr implementation: $(typeof(expr))"))
end

function _formula(expr::ScalarExpr)
    if expr isa ScalarSearch._ScalarInput
        return "x$(getfield(expr, :index))"
    elseif expr isa ScalarSearch._ScalarConstant
        return repr(getfield(expr, :value))
    elseif expr isa ScalarSearch._ScalarUnary
        op = getfield(expr, :op)
        return "$(op)($(_formula(getfield(expr, :arg))))"
    elseif expr isa ScalarSearch._ScalarBinary
        op = getfield(expr, :op)
        left = _formula(getfield(expr, :left))
        right = _formula(getfield(expr, :right))
        op in (:max, :min) && return "$(op)($left, $right)"
        return "($left $(op) $right)"
    end
    throw(ArgumentError("unsupported ScalarExpr implementation: $(typeof(expr))"))
end

function get_formula(model::AdapterModel)
    formulas = [_formula(expr) for expr in model.expressions]
    return length(formulas) == 1 ? first(formulas) :
        join(("y$(channel) = $(formula)" for (channel, formula) in enumerate(formulas)), "\n")
end

get_complexity(model::AdapterModel) = sum(scalar_complexity, model.expressions; init=0)

# Scalar inputs use the last axis for features; leading axes index samples.
# Targets may append an output-channel axis. Each channel is fitted independently.
function _fit_scalar(X::AbstractArray{<:Real}, y; kwargs...)
    matrix, sample_shape, feature_count = _input_matrix(X)
    targets, has_output_axis = _target_matrix(y, sample_shape, size(matrix, 2))
    config, seed, search_kwargs = _fit_config(kwargs)
    if haskey(search_kwargs, :initial_population) && search_kwargs[:initial_population] !== nothing
        permitted = Set((config.unary_operators..., config.binary_operators...))
        for expr in search_kwargs[:initial_population]
            expr isa ScalarExpr || throw(ArgumentError(
                "search_kwargs.initial_population must contain ScalarExpr values"))
            all(op -> op in permitted, _expression_operators(expr)) || throw(ArgumentError(
                "initial_population uses an operator disabled by the configured operator set"))
        end
    end
    expressions = ScalarExpr[]
    losses = Float64[]
    for channel in axes(targets, 2)
        result = search_scalar(matrix, view(targets, :, channel);
            config, rng=_rng(seed, channel), search_kwargs...)
        result isa ScalarSearchResult || throw(ArgumentError("ScalarSearch returned an invalid result"))
        isfinite(result.best_loss) || throw(ArgumentError(
            "ScalarSearch found no finite candidate for output channel $channel"))
        training_prediction = evaluate_scalar(result.best, matrix)
        training_prediction isa Err && throw(ArgumentError(
            "ScalarSearch returned an invalid model for output channel $channel: $(training_prediction.value)"))
        push!(expressions, result.best)
        push!(losses, result.best_loss)
    end
    model = AdapterModel(expressions, feature_count, losses, has_output_axis)
    formula = get_formula(model)
    return Dict{String,Any}(
        "model" => model,
        "formula" => formula,
        "complexity" => get_complexity(model),
        "loss" => sum(losses) / length(losses),
    )
end

function _reshape_prediction(values::Matrix{Float64}, sample_shape::Tuple;
        preserve_output_axis::Bool=false)
    output_count = size(values, 2)
    if output_count == 1 && !preserve_output_axis
        return isempty(sample_shape) ? values[1, 1] : reshape(copy(values[:, 1]), sample_shape)
    end
    return isempty(sample_shape) ? vec(values[1, :]) :
        reshape(values, (sample_shape..., output_count))
end

function predict(model::AdapterModel, X::AbstractArray{<:Real})
    matrix, sample_shape, feature_count = _input_matrix(X)
    feature_count == model.feature_count || throw(DimensionMismatch(
        "model expects $(model.feature_count) features, received $feature_count"))
    predictions = Matrix{Float64}(undef, size(matrix, 2), length(model.expressions))
    for channel in eachindex(model.expressions)
        result = evaluate_scalar(model.expressions[channel], matrix)
        result isa Err && throw(ArgumentError(
            "prediction failed in output channel $channel: $(result.value)"))
        all(isfinite, result) || throw(ArgumentError(
            "prediction produced non-finite values in output channel $channel"))
        predictions[:, channel] = result
    end
    return _reshape_prediction(predictions, sample_shape;
        preserve_output_axis=model.has_output_axis)
end

function _generation_options(kwargs; output_dim_default::Int=1)
    options = _dict_from_kwargs(kwargs)
    random_state = pop!(options, :random_state, 0)
    output_dim = pop!(options, :output_dim, output_dim_default)
    max_complexity = pop!(options, :max_complexity, 7)
    max_depth = pop!(options, :max_depth, 4)
    max_attempts = pop!(options, :max_attempts, 128)
    max_total_attempts = pop!(options, :max_total_attempts, nothing)
    require_input_dependency = pop!(options, :require_input_dependency, true)
    operator_set = pop!(options, :operator_set, _DEFAULT_ADAPTER_OPERATOR_SET)
    noise_level = pop!(options, :noise_level, 0.0)
    program_sampler = pop!(options, :program_sampler, nothing)
    constant_range = pop!(options, :constant_range, (-2.0, 2.0))
    unknown = setdiff(Set(keys(options)), Set{Symbol}())
    isempty(unknown) || throw(ArgumentError(
        "unsupported generation keyword(s): $(join(string.(sort!(collect(unknown))), ", "))"))
    output_dim isa Integer && output_dim >= 1 || throw(ArgumentError("output_dim must be positive"))
    max_complexity isa Integer && max_complexity >= 1 || throw(ArgumentError("max_complexity must be positive"))
    max_depth isa Integer && max_depth >= 0 || throw(ArgumentError("max_depth must be nonnegative"))
    max_attempts isa Integer && max_attempts >= 1 || throw(ArgumentError("max_attempts must be positive"))
    max_total_attempts === nothing || (max_total_attempts isa Integer && max_total_attempts >= 1) ||
        throw(ArgumentError("max_total_attempts must be positive or nothing"))
    require_input_dependency isa Bool || throw(ArgumentError(
        "require_input_dependency must be a Bool"))
    noise_level isa Real && isfinite(noise_level) && noise_level >= 0 ||
        throw(ArgumentError("noise_level must be finite and nonnegative"))
    (constant_range isa Tuple || constant_range isa AbstractVector) &&
        length(constant_range) == 2 && all(value -> value isa Real && isfinite(value), constant_range) &&
        constant_range[1] <= constant_range[2] || throw(ArgumentError(
            "constant_range must contain two finite ordered bounds"))
    unary, binary = _split_operator_set(operator_set)
    return (; seed=_base_seed(random_state), output_dim=Int(output_dim),
        max_complexity=Int(max_complexity), max_depth=Int(max_depth),
        max_attempts=Int(max_attempts),
        max_total_attempts=max_total_attempts === nothing ? nothing : Int(max_total_attempts),
        require_input_dependency,
        operator_set=Tuple(_as_symbols(operator_set, :operator_set)),
        unary_operators=unary, binary_operators=binary, noise_level=Float64(noise_level),
        program_sampler, constant_range=(Float64(constant_range[1]), Float64(constant_range[2])))
end

function _leaf_expression(rng::AbstractRNG, input_dim::Int, constant_range)
    if rand(rng) < 0.8
        return scalar_input(rand(rng, 1:input_dim))
    end
    return scalar_constant(rand(rng) * (constant_range[2] - constant_range[1]) + constant_range[1])
end

function _sample_expression(rng::AbstractRNG, input_dim::Int, max_nodes::Int,
        depth::Int, unary_operators, binary_operators, constant_range)
    (max_nodes <= 1 || depth <= 0) && return _leaf_expression(rng, input_dim, constant_range)
    can_unary = !isempty(unary_operators) && max_nodes >= 2
    can_binary = !isempty(binary_operators) && max_nodes >= 3
    (can_unary || can_binary) && rand(rng) >= 0.35 ||
        return _leaf_expression(rng, input_dim, constant_range)
    if can_unary && (!can_binary || rand(rng, Bool))
        op = rand(rng, unary_operators)
        arg = _sample_expression(rng, input_dim, max_nodes - 1, depth - 1,
            unary_operators, binary_operators, constant_range)
        return scalar_unary(op, arg)
    end
    op = rand(rng, binary_operators)
    remaining = max_nodes - 1
    left_budget = rand(rng, 1:(remaining - 1))
    right_budget = remaining - left_budget
    left = _sample_expression(rng, input_dim, left_budget, depth - 1,
        unary_operators, binary_operators, constant_range)
    right = _sample_expression(rng, input_dim, right_budget, depth - 1,
        unary_operators, binary_operators, constant_range)
    return scalar_binary(op, left, right)
end

function _sample_program(rng::AbstractRNG, input_dim::Int, options;
        task_index::Int, output_index::Int)
    options.program_sampler === nothing && return _sample_expression(rng, input_dim,
        options.max_complexity, options.max_depth, options.unary_operators,
        options.binary_operators, options.constant_range)
    expr = options.program_sampler(rng; input_dim,
        operator_set=options.operator_set,
        unary_operators=options.unary_operators,
        binary_operators=options.binary_operators,
        max_complexity=options.max_complexity,
        max_depth=options.max_depth,
        task_index,
        output_index)
    expr isa ScalarExpr || throw(ArgumentError("program_sampler must return a ScalarExpr"))
    scalar_complexity(expr) <= options.max_complexity || throw(ArgumentError(
        "program_sampler returned an expression above max_complexity"))
    _expression_depth(expr) <= options.max_depth || throw(ArgumentError(
        "program_sampler returned an expression above max_depth"))
    _required_feature(expr) <= input_dim || throw(ArgumentError(
        "program_sampler returned an expression requiring more inputs than input_dim"))
    all(op -> op in options.operator_set, _expression_operators(expr)) ||
        throw(ArgumentError("program_sampler used an operator outside operator_set"))
    return expr
end

function _expression_operators(expr::ScalarExpr)
    if expr isa ScalarSearch._ScalarInput || expr isa ScalarSearch._ScalarConstant
        return Symbol[]
    elseif expr isa ScalarSearch._ScalarUnary
        return [getfield(expr, :op); _expression_operators(getfield(expr, :arg))]
    elseif expr isa ScalarSearch._ScalarBinary
        return [getfield(expr, :op); _expression_operators(getfield(expr, :left));
            _expression_operators(getfield(expr, :right))]
    end
    throw(ArgumentError("unsupported ScalarExpr implementation: $(typeof(expr))"))
end

function _validated_program(rng::AbstractRNG, input_dim::Int, Xmatrix::Matrix{Float64}, options;
        task_index::Int, output_index::Int, total_attempts::Base.RefValue{Int})
    for _ in 1:options.max_attempts
        if options.max_total_attempts !== nothing &&
                total_attempts[] >= options.max_total_attempts
            break
        end
        total_attempts[] += 1
        expr = _sample_program(rng, input_dim, options; task_index, output_index)
        scalar_complexity(expr) <= options.max_complexity || continue
        _required_feature(expr) <= input_dim || continue
        options.require_input_dependency && _required_feature(expr) == 0 && continue
        values = evaluate_scalar(expr, Xmatrix)
        values isa Err && continue
        all(isfinite, values) || continue
        length(values) <= 1 || any(value -> value != first(values), values) || continue
        return expr, values
    end
    throw(_ProgramSamplingErr(task_index, output_index, total_attempts[],
        options.max_attempts, options.max_total_attempts))
end

function _noise!(Y, noise_level::Float64, rng::AbstractRNG)
    noise_level == 0.0 && return Y
    for index in eachindex(Y)
        value = Y[index] + noise_level * randn(rng)
        isfinite(value) || throw(ArgumentError("noise produced a non-finite target"))
        Y[index] = value
    end
    return Y
end

function _generate_sample_targets(Xmatrix, sample_shape, options; task_index::Int=1)
    input_dim = size(Xmatrix, 1)
    programs = Vector{ScalarExpr}(undef, options.output_dim)
    values = Matrix{Float64}(undef, size(Xmatrix, 2), options.output_dim)
    total_attempts = Ref(0)
    for output_index in 1:options.output_dim
        rng = _rng(options.seed, 10_000 + (task_index - 1) * options.output_dim + output_index)
        expr, y = _validated_program(rng, input_dim, Xmatrix, options;
            task_index, output_index, total_attempts)
        programs[output_index] = expr
        values[:, output_index] = y
    end
    result = _reshape_prediction(values, sample_shape;
        preserve_output_axis=options.output_dim > 1)
    rng = _rng(options.seed, 20_000 + task_index)
    if options.noise_level > 0
        if result isa Number
            result += options.noise_level * randn(rng)
            isfinite(result) || throw(ArgumentError("noise produced a non-finite target"))
        else
            _noise!(result, options.noise_level, rng)
        end
    end
    return result, programs, total_attempts[]
end

# Preserve sample shape; append an output axis only when output_dim > 1.
function _generate_scalar_synthetic_data(X::AbstractArray{<:Real}; kwargs...)
    matrix, sample_shape, input_dim = _input_matrix(X)
    options = _generation_options(kwargs)
    options.max_total_attempts === nothing &&
        (options = merge(options, (; max_total_attempts=
            options.max_attempts * options.output_dim)))
    y, programs, attempts_used = _generate_sample_targets(matrix, sample_shape, options)
    formula = options.output_dim == 1 ? _formula(first(programs)) :
        join(("y$(channel) = $(_formula(program))" for
            (channel, program) in enumerate(programs)), "\n")
    return Dict{String,Any}(
        "y" => y,
        "programs" => programs,
        "formula" => formula,
        "complexity" => sum(scalar_complexity, programs; init=0),
        "input_dim" => input_dim,
        "program_family" => "elementwise vector-to-vector",
        "require_input_dependency" => options.require_input_dependency,
        "attempts_used" => attempts_used,
        "max_total_attempts" => options.max_total_attempts,
    )
end


include("collection_programs.jl")
include("collection_synthesis.jl")

# Collection programs consume whole samples; rank <= 2 defaults to scalar fitting.
function generate_synthetic_data(X; task_kind=:auto, kwargs...)
    kind = _resolve_task_kind(X, task_kind)
    return kind === :scalar ? _generate_scalar_synthetic_data(X; kwargs...) :
        generate_collection_synthetic_data(X; task_kind=kind, kwargs...)
end

# The fixed task embedding is excluded from teacher inputs, targets, and loss.
function generate_pretraining_tasks(; task_kind=:array, kwargs...)
    kind = Symbol(task_kind)
    kind in (:array, :series) || throw(ArgumentError(
        "pretraining task_kind must be array or series"))
    return generate_collection_pretraining_tasks(; task_kind=kind, kwargs...)
end

@testitem "Frontend adapter fits and round-trips multiple output channels" begin
    using LambdaRegression.FrontendAdapter: fit, predict, get_formula, get_complexity
    using LambdaRegression.ScalarSearch: scalar_input, scalar_constant, scalar_binary

    X = [0.0 1.0; 1.0 2.0; 2.0 3.0]
    first = scalar_binary(:+, scalar_input(1), scalar_constant(1.0))
    second = scalar_binary(:*, scalar_input(2), scalar_constant(2.0))
    Y = hcat(X[:, 1] .+ 1.0, X[:, 2] .* 2.0)
    result = fit(X, Y; population_size=2, generations=0, max_evaluations=2,
        max_complexity=7, initial_depth=1, tournament_size=1, elite_count=1,
        operator_set=["+", "*"], evolution_strategy="generational",
        mutation_rate=0.9, crossover_rate=0.05,
        search_kwargs=Dict{Any,Any}("stop_loss" => 0.0,
            "initial_population" => [first, second]),
        random_state=19)

    @test result["model"] isa LambdaRegression.FrontendAdapter.AdapterModel
    @test predict(result["model"], X) ≈ Y
    @test result["formula"] == get_formula(result["model"])
    @test result["complexity"] == get_complexity(result["model"])
    @test size(predict(result["model"], reshape(X, 1, 3, 2))) == (1, 3, 2)
    single_axis = fit(X, reshape(Y[:, 1], 3, 1); population_size=2,
        generations=0, max_evaluations=2, tournament_size=1, elite_count=1,
        operator_set=["+", "*"], search_kwargs=Dict{Any,Any}(
            "stop_loss" => 0.0, "initial_population" => [first, second]))
    @test size(predict(single_axis["model"], X)) == (3, 1)
    @test_throws ArgumentError fit(X, Y; made_up_option=1)
    @test_throws ArgumentError fit(X, Y; search_kwargs=(made_up_option=1,))
    @test_throws ArgumentError fit(X, Y; operator_set=["+"],
        search_kwargs=(initial_population=[second],))
end

@testitem "Frontend explicit scalar synthetic data keeps shape and rejects unknown options" begin
    using LambdaRegression.FrontendAdapter: generate_synthetic_data

    X = reshape(range(-1.0, 1.0; length=12), 3, 2, 2)
    first = generate_synthetic_data(X; task_kind=:scalar, random_state=4, output_dim=2,
        operator_set=[:+, :*, :sin], max_complexity=5, max_depth=2)
    second = generate_synthetic_data(X; task_kind=:scalar, random_state=4, output_dim=2,
        operator_set=[:+, :*, :sin], max_complexity=5, max_depth=2)
    @test size(first["y"]) == (3, 2, 2)
    @test first["y"] == second["y"]
    @test first["formula"] == second["formula"]
    @test_throws ArgumentError generate_synthetic_data(X; unknown_option=1)
end

end # module FrontendAdapter
