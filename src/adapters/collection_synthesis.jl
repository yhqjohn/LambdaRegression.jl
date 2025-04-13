# Programs map complete (T,N,Cin) samples to (T,N,Cout) results.
# :series represents T as a list of (N,C) tensors.

using ..ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
using ..Programs: Program
using ..TypeSystem: FloatTy, Ty, TyArray, TyFun, TyList, TyLike, TyNat, TyProd
using ..Regression: TensorSchema
import ..Regression

const _COLLECTION_SYNTHESIS_KEYS = Set((
    :output_dim, :output_shape, :operator_set, :program_family,
    :program_sampler, :input_sampler, :components_kwargs, :random_state,
    :global_consts, :global_consts_ty, :regressor_config,
    :noise_level, :prefix_tokens, :max_attempts, :max_total_attempts,
    :max_complexity, :max_depth, :kernel_max_complexity, :kernel_max_depth,
    :constant_range,
))
_collection_app(function_node, argument) = AppNode(function_node, argument)
_collection_apps(function_node, arguments...) = foldl(
    _collection_app, arguments; init=function_node,
)
_collection_const(name::Symbol) = ConstNode(name)

function _collection_tuple_type(item_ty::Ty, count::Int)
    count == 1 && return item_ty
    return TyProd(item_ty, _collection_tuple_type(item_ty, count - 1))
end

function _collection_pair(items::Vector)
    length(items) == 1 && return only(items)
    return _collection_apps(
        _collection_const(:PAIR), first(items), _collection_pair(items[2:end]),
    )
end

function _collection_flatten_pair(value)
    value isa Tuple || return Any[value]
    length(value) == 2 || throw(ArgumentError(
        "internal channel tuple must be right-associated pairs",
    ))
    return vcat(_collection_flatten_pair(value[1]), _collection_flatten_pair(value[2]))
end

function _collection_pack_channels(value, n_elements::Int, output_dim::Int)
    channels = _collection_flatten_pair(value)
    length(channels) == output_dim || throw(DimensionMismatch(
        "program produced $(length(channels)) channels; expected $output_dim",
    ))
    all(channel -> channel isa AbstractArray && size(channel) == (n_elements,), channels) ||
        throw(DimensionMismatch("channel packer expected vectors of length $n_elements"))
    output_dim == 1 && return reshape(Float64.(only(channels)), n_elements, 1)
    return cat((Float64.(channel) for channel in channels)...; dims=2)
end

function _collection_sum_features(element_index::Int=0)
    folded = _collection_apps(
        _collection_const(:AFOLDL),
        _collection_const(:FADD),
        _collection_const(:__collection_zero),
        IndexNode(element_index),
    )
    return folded
end

function _collection_group_mean(frame_index::Int)
    sum_per_element = _collection_apps(
        _collection_const(:AMAP),
        AbsNode(:features, _collection_sum_features(0)),
        IndexNode(frame_index),
    )
    total = _collection_apps(
        _collection_const(:AFOLDL),
        _collection_const(:FADD),
        _collection_const(:__collection_zero),
        sum_per_element,
    )
    return _collection_app(
        _collection_app(_collection_const(:FMUL), total),
        _collection_const(:__collection_mean_scale),
    )
end

const _COLLECTION_SCALAR_AST_NAMES = Dict{Symbol,Symbol}(
    :+ => :FADD, :- => :FSUB, :* => :FMUL, :/ => :FDIV,
    :^ => :FPOW, :max => :FMAX, :min => :FMIN,
    :sin => :FSIN, :cos => :FCOS, :tan => :FTAN, :exp => :FEXP,
    :log => :FLOG, :abs => :FABS, :sqrt => :FSQRT, :square => :SQR,
)

function _collection_scalar_ast(
    expression::ScalarExpr,
    input_node,
    local_consts::Dict{Symbol,Any},
    local_types::Dict{Symbol,TyLike},
    constant_counter::Base.RefValue{Int},
)
    if expression isa ScalarSearch._ScalarInput
        getfield(expression, :index) == 1 || throw(ArgumentError(
            "collection scalar kernel may refer only to its one group-scalar input",
        ))
        return input_node
    elseif expression isa ScalarSearch._ScalarConstant
        constant_counter[] += 1
        name = Symbol("__collection_kernel_constant_", constant_counter[])
        local_consts[name] = getfield(expression, :value)
        local_types[name] = FloatTy
        return _collection_const(name)
    elseif expression isa ScalarSearch._ScalarUnary
        operator = getfield(expression, :op)
        name = get(_COLLECTION_SCALAR_AST_NAMES, operator, nothing)
        name === nothing && throw(ArgumentError("unsupported collection kernel operator $operator"))
        return _collection_app(_collection_const(name), _collection_scalar_ast(
            getfield(expression, :arg), input_node, local_consts, local_types, constant_counter,
        ))
    elseif expression isa ScalarSearch._ScalarBinary
        operator = getfield(expression, :op)
        name = get(_COLLECTION_SCALAR_AST_NAMES, operator, nothing)
        name === nothing && throw(ArgumentError("unsupported collection kernel operator $operator"))
        left = _collection_scalar_ast(
            getfield(expression, :left), input_node, local_consts, local_types, constant_counter,
        )
        right = _collection_scalar_ast(
            getfield(expression, :right), input_node, local_consts, local_types, constant_counter,
        )
        return _collection_apps(_collection_const(name), left, right)
    end
    throw(ArgumentError("unsupported collection scalar kernel $(typeof(expression))"))
end

function _collection_frame_body(
    family::Symbol,
    output_dim::Int,
    n_elements::Int,
    scalar_kernels::Vector{ScalarExpr},
    local_consts::Dict{Symbol,Any},
    local_types::Dict{Symbol,TyLike},
    weights::Vector{Float64},
    biases::Vector{Float64},
)
    channels = Any[]
    constant_counter = Ref(0)
    for channel in 1:output_dim
        group_scalar = if family === :center
            _collection_apps(
                _collection_const(:FADD),
                _collection_sum_features(0),
                _collection_app(
                    _collection_app(_collection_const(:FMUL),
                        _collection_group_mean(1)),
                    _collection_const(:__collection_negative_one),
                ),
            )
        else
            _collection_sum_features(0)
        end
        scalar = _collection_scalar_ast(
            scalar_kernels[channel], group_scalar, local_consts, local_types, constant_counter,
        )
        channel_value = _collection_apps(
            _collection_const(:FADD),
            _collection_app(
                _collection_app(_collection_const(:FMUL), scalar),
                _collection_const(Symbol("__collection_weight_", channel)),
            ),
            _collection_const(Symbol("__collection_bias_", channel)),
        )
        push!(channels, _collection_apps(
            _collection_const(:AMAP),
            AbsNode(:element, channel_value),
            IndexNode(0),
        ))
    end
    packed = _collection_app(
        _collection_const(:__collection_pack_channels),
        _collection_pair(channels),
    )
    return packed
end

function _collection_program(
    family::Symbol,
    task_kind::Symbol,
    input_shape::Tuple,
    output_shape::Tuple,
    world::NamedTuple,
    rng::AbstractRNG,
    operator_set,
    options::Dict{Symbol,Any},
)
    length(input_shape) in (2, 3) || throw(ArgumentError(
        "array program input_shape must be (N,Cin) or (T,N,Cin)",
    ))
    length(output_shape) == length(input_shape) || throw(ArgumentError(
        "default collection programs preserve the sample rank",
    ))
    n_elements = input_shape[end - 1]
    input_dim = input_shape[end]
    output_dim = output_shape[end]
    output_shape[1:end - 1] == input_shape[1:end - 1] || throw(ArgumentError(
        "default collection programs preserve all axes except the channel axis",
    ))
    frame_ty = TensorSchema((n_elements, input_dim)).ty
    output_frame_ty = TensorSchema((n_elements, output_dim)).ty
    input_ty = task_kind === :array ? TensorSchema(input_shape).ty : TyList(frame_ty)
    output_ty = task_kind === :array ? TensorSchema(output_shape).ty : TyList(output_frame_ty)

    # Draw each task's coefficients independently from the same bounded prior.
    # Gains are kept away from zero so every output channel depends on its kernel.
    draw_gain() = begin
        while true
            value = 2 * rand(rng) - 1
            abs(value) >= 0.1 && return value
        end
    end
    weights = [draw_gain() for _ in 1:output_dim]
    biases = [2 * rand(rng) - 1 for _ in 1:output_dim]
    local_consts = Dict{Symbol,Any}(
        :__collection_zero => 0.0,
        :__collection_mean_scale => 1.0 / n_elements,
        :__collection_negative_one => -1.0,
    )
    local_types = Dict{Symbol,TyLike}(
        :__collection_zero => FloatTy,
        :__collection_mean_scale => FloatTy,
        :__collection_negative_one => FloatTy,
    )
    unary, binary = _split_operator_set(operator_set)
    kernel_limit = Int(get(options, :kernel_max_complexity, 7))
    kernel_limit > 0 || throw(ArgumentError("kernel_max_complexity must be positive"))
    kernel_depth = Int(get(options, :kernel_max_depth,
        get(options, :max_depth, 3)))
    kernel_depth >= 0 || throw(ArgumentError("kernel_max_depth must be non-negative"))
    constant_range = get(options, :constant_range, (-2.0, 2.0))
    length(constant_range) == 2 && all(value -> value isa Real && isfinite(value), constant_range) &&
        constant_range[1] <= constant_range[2] || throw(ArgumentError(
            "constant_range must contain two finite ordered bounds",
        ))
    scalar_kernels = ScalarExpr[]
    for _ in 1:output_dim
        kernel = nothing
        for _ in 1:16
            budget = min(kernel_limit, rand(rng, 3:7))
            candidate = _sample_expression(rng, 1, budget, kernel_depth,
                unary, binary, (Float64(constant_range[1]), Float64(constant_range[2])))
            if _required_feature(candidate) == 1
                kernel = candidate
                break
            end
        end
        kernel === nothing && (kernel = scalar_input(1))
        push!(scalar_kernels, kernel)
    end
    for channel in 1:output_dim
        weight_name = Symbol("__collection_weight_", channel)
        bias_name = Symbol("__collection_bias_", channel)
        local_consts[weight_name] = weights[channel]
        local_consts[bias_name] = biases[channel]
        local_types[weight_name] = FloatTy
        local_types[bias_name] = FloatTy
    end
    frame_body = _collection_frame_body(
        family, output_dim, n_elements, scalar_kernels,
        local_consts, local_types, weights, biases,
    )
    body = if task_kind === :array && length(input_shape) == 2
        frame_body
    else
        time_map_name = task_kind === :array ? :AMAP : :__collection_lmap
        _collection_apps(
            _collection_const(time_map_name),
            AbsNode(:frame, frame_body),
            IndexNode(0),
        )
    end
    tree = AstTree(AbsNode(:x, body))

    channel_ty = TyArray(TyNat(n_elements), FloatTy)
    tuple_ty = _collection_tuple_type(channel_ty, output_dim)
    pack_ty = TyFun(tuple_ty,
        TyArray(TyNat(n_elements), TyArray(TyNat(output_dim), FloatTy)))
    packer = value -> _collection_pack_channels(value, n_elements, output_dim)
    local_consts[:__collection_pack_channels] = packer
    local_types[:__collection_pack_channels] = pack_ty
    if task_kind === :series
        lmap = f -> xs -> map(f, xs)
        local_consts[:__collection_lmap] = lmap
        local_types[:__collection_lmap] = TyFun(
            TyFun(frame_ty, output_frame_ty),
            TyFun(TyList(frame_ty), TyList(output_frame_ty)),
        )
    end
    return Program(tree; local_consts, local_consts_ty=local_types)
end

function _collection_program_key(program::Program)
    # Array-valued constants are program materials too. Callable layout helpers
    # remain fixed in the task world and are not identified by allocation identity.
    data_locals = sort!(collect((name, value) for (name, value) in program.local_consts
        if !(value isa Function)); by=pair -> string(first(pair)))
    local_material = join((string(name, "=", repr(value)) for (name, value) in data_locals), ";")
    return string(sprint(show, program.tree), "|", local_material)
end

function _collection_synthesis_options(kwargs)
    options = Dict{Symbol,Any}()
    for (raw_key, value) in pairs(kwargs)
        key = raw_key isa Symbol ? raw_key : Symbol(String(raw_key))
        key in _COLLECTION_SYNTHESIS_KEYS || throw(ArgumentError(
            "unsupported collection synthesis option: $key",
        ))
        haskey(options, key) && throw(ArgumentError(
            "duplicate collection synthesis option: $key",
        ))
        options[key] = value
    end
    return options
end

function _collection_positive_int(options, key::Symbol, default=nothing)
    value = get(options, key, default)
    value isa Integer && value > 0 || throw(ArgumentError("$key must be a positive integer"))
    return Int(value)
end

function _collection_family(task_index::Int, requested::Symbol, n_elements::Int)
    requested in (:mixed, :pointwise, :center) || throw(ArgumentError(
        "program_family must be :mixed, :pointwise, or :center",
    ))
    requested === :center && n_elements < 2 && throw(ArgumentError(
        "program_family=:center requires at least two elements",
    ))
    requested !== :mixed && return requested
    # A one-element group has no cross-element centering signal.
    return n_elements > 1 && isodd(task_index) ? :center : :pointwise
end

function _collection_world_for(
    task_kind::Symbol,
    input_shape::Tuple,
    output_shape::Tuple,
    operator_set,
    components_kwargs,
    options,
)
    input_ty = task_kind === :array ? TensorSchema(input_shape).ty :
        TyList(TensorSchema(input_shape[2:end]).ty)
    output_ty = task_kind === :array ? TensorSchema(output_shape).ty :
        TyList(TensorSchema(output_shape[2:end]).ty)
    return collection_world(TyFun(input_ty, output_ty);
        operator_set,
        global_consts=get(options, :global_consts, Dict{Symbol,Any}()),
        global_consts_ty=get(options, :global_consts_ty, nothing),
        components_kwargs)
end

function _collection_regressor_config(options)
    overrides = _collection_options(
        get(options, :regressor_config, NamedTuple()), "regressor_config",
    )
    # Nested map/fold execution needs a larger default step budget.
    values = Dict{Symbol,Any}(:procedural_max_steps => 4_096)
    merge!(values, overrides)
    if haskey(values, :parallel) && values[:parallel] isa AbstractDict
        values[:parallel] = _options_namedtuple(_collection_options(
            values[:parallel], "regressor_config.parallel",
        ))
    end
    return default_config(; _options_namedtuple(values)...)
end

function _collection_build_model(
    rng::AbstractRNG,
    task_index::Int,
    task_kind::Symbol,
    family::Symbol,
    input_shape::Tuple,
    output_shape::Tuple,
    world::NamedTuple,
    options::Dict{Symbol,Any},
    validation_samples::AbstractArray{<:Real},
    attempt_budget::Int,
    seen_program_keys::Union{Nothing,Set{String}}=nothing,
)
    program_sampler = get(options, :program_sampler, nothing)
    attempt_budget > 0 || throw(ArgumentError("remaining attempt budget must be positive"))
    max_complexity = Int(get(options, :max_complexity, 512))
    max_complexity > 0 || throw(ArgumentError("max_complexity must be positive"))
    if program_sampler === nothing
        requested = Set(_as_symbols(get(options, :operator_set, _COLLECTION_OPERATORS),
            :operator_set))
        all(operator -> operator in requested, (:+, :*)) || throw(ArgumentError(
            "default pointwise/center Programs require operator_set to include :+ and :*",
        ))
    end
    last_issue = nothing
    for attempt in 1:attempt_budget
        program = if program_sampler === nothing
            _collection_program(
                family, task_kind, input_shape, output_shape, world, rng,
                get(options, :operator_set, _COLLECTION_OPERATORS), options,
            )
        else
            program_sampler(rng;
                task_index, task_kind, program_family=family,
                input_shape, output_shape,
                input_ty=world.target_ty.arg, output_ty=world.target_ty.res,
                attempt,
            )
        end
        program isa Program || throw(ArgumentError(
            "program_sampler must return one complete Programs.Program",
        ))
        issue = Regression._validate_program_contract(
            program, world.target_ty, world.components,
            "collection synthesis Program for task $task_index";
        )
        if issue !== nothing
            last_issue = issue
            continue
        end
        program_key = _collection_program_key(program)
        if seen_program_keys !== nothing && program_key in seen_program_keys
            last_issue = ArgumentError("program_sampler returned a duplicate complete Program")
            continue
        end
        model = CollectionModel(program, world;
            task_kind,
            input_shape=task_kind === :array ? input_shape : input_shape[2:end],
            output_shape=task_kind === :array ? output_shape : output_shape[2:end],
            config=_collection_regressor_config(options),
        )
        if get_complexity(model) > max_complexity
            last_issue = ArgumentError(
                "program complexity=$(get_complexity(model)) exceeds max_complexity=$max_complexity",
            )
            continue
        end
        try
            predictions = _collection_run_model(model, validation_samples)
            seen_program_keys === nothing || push!(seen_program_keys, program_key)
            return model, attempt, predictions
        catch error
            if error isa Union{ArgumentError, DimensionMismatch, DomainError, OverflowError, InexactError}
                last_issue = error
                continue
            end
            rethrow()
        end
    end
    throw(ArgumentError(
        "could not build a valid collection program for task $task_index within " *
        "attempt_budget=$attempt_budget; last issue: $(repr(last_issue))",
    ))
end

function _collection_sample(
    rng::AbstractRNG,
    input_sampler,
    task_index::Int,
    sample_index::Int,
    input_shape::Tuple,
)
    values = input_sampler === nothing ? randn(rng, input_shape...) :
        input_sampler(rng; input_shape, task_index, sample_index)
    values isa AbstractArray{<:Real} && size(values) == input_shape || throw(ArgumentError(
        "input_sampler must return one real array with exact input_shape=$input_shape",
    ))
    all(isfinite, values) || throw(ArgumentError(
        "input_sampler returned non-finite features",
    ))
    return Float64.(values)
end

function _collection_add_noise(rng, values::Array{Float64}, noise_level::Float64)
    noise_level >= 0 && isfinite(noise_level) || throw(ArgumentError(
        "noise_level must be finite and non-negative",
    ))
    noise_level == 0 && return copy(values)
    return values .+ noise_level .* randn(rng, size(values))
end

function _collection_run_model(model::CollectionModel, samples::AbstractArray{<:Real})
    predicted = predict(model, samples)
    predicted isa AbstractArray{<:Real} || throw(ArgumentError(
        "collection program prediction must be a dense real array",
    ))
    all(isfinite, predicted) || throw(ArgumentError(
        "collection program produced non-finite outputs",
    ))
    return Float64.(predicted)
end

# X has shape (B,N,Cin) or (B,T,N,Cin); output channels are independent.
# program_sampler can supply a complete typed program.
function generate_collection_synthetic_data(X; task_kind=:array, kwargs...)
    options = _collection_synthesis_options(kwargs)
    inapplicable = intersect(Set(keys(options)), Set((:prefix_tokens, :input_sampler)))
    isempty(inapplicable) || throw(ArgumentError(
        "options apply only to pretraining task generation: $(join(string.(sort!(collect(inapplicable))), ", "))"))
    kind = Symbol(task_kind)
    kind in (:array, :series) || throw(ArgumentError(
        "task_kind must be :array or :series",
    ))
    X isa AbstractArray{<:Real} && ndims(X) in (3, 4) || throw(ArgumentError(
        "collection X must be a real dense (B,N,Cin) or (B,T,N,Cin) array",
    ))
    kind === :series && ndims(X) != 4 && throw(ArgumentError(
        "series collection X must include a frame axis: (B,T,N,Cin)",
    ))
    all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
    shape = Tuple(size(X))
    batch = shape[1]
    input_shape = shape[2:end]
    all(>(0), shape) || throw(ArgumentError(
        "all collection X dimensions must be positive",
    ))
    n_elements, input_dim = input_shape[end - 1], input_shape[end]
    output_prefix = input_shape[1:end - 1]
    output_dim = _collection_positive_int(options, :output_dim, input_dim)
    output_shape = Tuple(get(options, :output_shape, (output_prefix..., output_dim)))
    length(output_shape) == length(input_shape) &&
        output_shape[1:end - 1] == output_prefix || throw(ArgumentError(
            "output_shape must preserve all input axes except channels",
        ))
    output_shape[end] > 0 || throw(ArgumentError("output_shape dimensions must be positive"))
    family = _collection_family(1, Symbol(get(options, :program_family, :mixed)), n_elements)
    operator_set = get(options, :operator_set, _COLLECTION_OPERATORS)
    world = _collection_world_for(kind, input_shape,
        output_shape, operator_set, get(options, :components_kwargs, NamedTuple()), options)
    seed = _base_seed(get(options, :random_state, 0))
    inputs = Float64.(X)
    total_limit = Int(get(options, :max_total_attempts, get(options, :max_attempts, 16)))
    total_limit > 0 || throw(ArgumentError("max_total_attempts must be positive"))
    attempts_limit = min(Int(get(options, :max_attempts, total_limit)), total_limit)
    attempts_limit > 0 || throw(ArgumentError("max_attempts must be positive"))
    model, attempts, y_clean = _collection_build_model(_rng(seed, 1), 1, kind, family,
        input_shape, output_shape, world, options, inputs, attempts_limit)
    noise_level = Float64(get(options, :noise_level, 0.0))
    y = _collection_add_noise(_rng(seed, 2), y_clean, noise_level)
    return Dict{String,Any}(
        "y" => y,
        "Y_clean" => y_clean,
        "programs" => Program[model.program],
        "models" => CollectionModel[model],
        "program_ids" => ones(Int, batch),
        "task_ids" => ones(Int, batch),
        "program_families" => [family],
        "program_family" => String(family),
        "formula" => get_formula(model),
        "complexity" => get_complexity(model),
        "input_shape" => input_shape,
        "output_shape" => output_shape,
        "attempts_used" => attempts,
        "max_total_attempts" => total_limit,
    )
end

# X includes the per-task prefix embedding; base_X and targets exclude it.
# Default program families require + and *; complete program keys must be unique.
function generate_collection_pretraining_tasks(
    ; task_kind=:array,
    task_count,
    samples_per_task,
    n_frames,
    n_elements,
    input_dim,
    output_dim,
    kwargs...,
)
    options = _collection_synthesis_options(kwargs)
    kind = Symbol(task_kind)
    kind in (:array, :series) || throw(ArgumentError(
        "task_kind must be :array or :series",
    ))
    task_count = Int(task_count)
    samples_per_task = Int(samples_per_task)
    n_frames = Int(n_frames)
    n_elements = Int(n_elements)
    input_dim = Int(input_dim)
    output_dim = Int(output_dim)
    all(>(0), (task_count, samples_per_task, n_frames, n_elements, input_dim, output_dim)) ||
        throw(ArgumentError("task and shape dimensions must be positive integers"))
    prefix_tokens = Int(get(options, :prefix_tokens, 1))
    prefix_tokens == 1 || throw(ArgumentError(
        "pretraining tasks require exactly one prefix token per frame",
    ))
    output_shape = Tuple(get(options, :output_shape, (n_frames, n_elements, output_dim)))
    length(output_shape) == 3 && output_shape[1:2] == (n_frames, n_elements) ||
        throw(ArgumentError("output_shape must be (n_frames, n_elements, output_dim)"))
    output_shape[3] > 0 || throw(ArgumentError("output_shape dimensions must be positive"))
    family_request = Symbol(get(options, :program_family, :mixed))
    operator_set = get(options, :operator_set, _COLLECTION_OPERATORS)
    components_kwargs = get(options, :components_kwargs, NamedTuple())
    input_shape = (n_frames, n_elements, input_dim)
    world = _collection_world_for(kind, input_shape, output_shape,
        operator_set, components_kwargs, options)

    seed = _base_seed(get(options, :random_state, 0))
    total = task_count * samples_per_task
    base_X = Array{Float64}(undef, total, n_frames, n_elements, input_dim)
    X = Array{Float64}(undef, total, n_frames, n_elements + 1, input_dim)
    embeddings = Array{Float64}(undef, task_count, input_dim)
    Y_clean = Array{Float64}(undef, total, output_shape...)
    programs = Vector{Program}(undef, task_count)
    models = Vector{CollectionModel}(undef, task_count)
    families = Vector{Symbol}(undef, task_count)
    program_ids = Vector{Int}(undef, total)
    attempts_per_task = zeros(Int, task_count)
    seen_program_keys = Set{String}()
    attempts_limit = Int(get(options, :max_attempts, 16))
    attempts_limit > 0 || throw(ArgumentError("max_attempts must be positive"))
    total_limit = Int(get(options, :max_total_attempts, task_count * attempts_limit))
    total_limit > 0 || throw(ArgumentError("max_total_attempts must be positive"))
    total_attempts = 0
    input_sampler = get(options, :input_sampler, nothing)
    for task_index in 1:task_count
        embeddings[task_index, :] .= randn(_rng(seed, 200 + task_index), input_dim)
        for within_task in 1:samples_per_task
            row = (task_index - 1) * samples_per_task + within_task
            sample = _collection_sample(_rng(seed, 10_000 + row), input_sampler,
                task_index, within_task, input_shape)
            base_X[row, :, :, :] .= sample
            X[row, :, 1, :] .= reshape(embeddings[task_index, :], 1, input_dim)
            X[row, :, 2:end, :] .= sample
            program_ids[row] = task_index
        end
    end

    for task_index in 1:task_count
        family = _collection_family(task_index, family_request, n_elements)
        families[task_index] = family
        row_start = (task_index - 1) * samples_per_task + 1
        rows = row_start:(row_start + samples_per_task - 1)
        remaining = total_limit - total_attempts
        remaining > 0 || throw(ArgumentError(
            "collection task generation exhausted max_total_attempts=$total_limit " *
            "before sampling task $task_index",
        ))
        budget = min(attempts_limit, remaining)
        model, attempts, task_targets = _collection_build_model(
            _rng(seed, 100 + task_index), task_index, kind, family,
            input_shape, output_shape, world, options,
            base_X[rows, :, :, :], budget, seen_program_keys,
        )
        total_attempts += attempts
        attempts_per_task[task_index] = attempts
        models[task_index] = model
        programs[task_index] = model.program
        Y_clean[rows, :, :, :] .= task_targets
    end
    noise_level = Float64(get(options, :noise_level, 0.0))
    noisy = _collection_add_noise(_rng(seed, 300), Y_clean, noise_level)
    return Dict{String,Any}(
        "X" => X,
        "Y" => noisy,
        "Y_clean" => Y_clean,
        "base_X" => base_X,
        "embeddings" => embeddings,
        "program_ids" => program_ids,
        "task_ids" => copy(program_ids),
        "programs" => programs,
        "models" => models,
        "program_families" => families,
        "program_family" => copy(families),
        "attempts_per_task" => attempts_per_task,
        "attempts_used" => total_attempts,
        "max_total_attempts" => total_limit,
        "task_kind" => kind,
        "input_shape" => input_shape,
        "output_shape" => output_shape,
        "prefix_tokens" => prefix_tokens,
    )
end

@testitem "Collection synthesis maps full groups and centers across elements" begin
    using Test
    using LambdaRegression.FrontendAdapter:
        CollectionModel, generate_collection_pretraining_tasks, get_formula, predict
    using LambdaRegression.Programs: Program

    tasks = generate_collection_pretraining_tasks(
        task_kind=:array, task_count=2, samples_per_task=1,
        n_frames=2, n_elements=3, input_dim=2, output_dim=3,
        random_state=21, prefix_tokens=1, operator_set=(:+, :*),
        kernel_max_complexity=1,
    )
    @test tasks["programs"] isa Vector{Program}
    @test length(tasks["programs"]) == 2
    @test tasks["programs"][1] isa Program
    @test tasks["program_families"] == [:center, :pointwise]
    @test size(tasks["X"]) == (2, 2, 4, 2)
    @test size(tasks["base_X"]) == (2, 2, 3, 2)
    @test size(tasks["Y_clean"]) == (2, 2, 3, 3)
    @test tasks["X"][:, :, 2:end, :] == tasks["base_X"]
    @test all(tasks["X"][:, frame, 1, :] == tasks["embeddings"]
        for frame in axes(tasks["X"], 2))
    @test tasks["models"] isa Vector{CollectionModel}
    @test get_formula(tasks["models"][1]) != get_formula(tasks["models"][2])
    @test predict(tasks["models"][1], tasks["base_X"][1:1, :, :, :]) ≈
        tasks["Y_clean"][1:1, :, :, :]

    changed = copy(tasks["base_X"][1:1, :, :, :])
    changed[1, 1, 1, 1] += 0.5
    before = predict(tasks["models"][1], tasks["base_X"][1:1, :, :, :])
    after = predict(tasks["models"][1], changed)
    @test before[1, 1, 2, 1] != after[1, 1, 2, 1]
    @test predict(tasks["models"][2], tasks["base_X"][2:2, :, :, :]) ≈
        tasks["Y_clean"][2:2, :, :, :]
    repeated = generate_collection_pretraining_tasks(
        task_kind=:array, task_count=2, samples_per_task=1,
        n_frames=2, n_elements=3, input_dim=2, output_dim=3,
        random_state=21, prefix_tokens=1, operator_set=(:+, :*),
        kernel_max_complexity=1,
    )
    @test get_formula.(tasks["models"]) == get_formula.(repeated["models"])

    ordinary = generate_collection_pretraining_tasks(
        task_kind=:array, task_count=3, samples_per_task=2,
        n_frames=2, n_elements=3, input_dim=2, output_dim=2,
        random_state=23, operator_set=(:+, :*),
    )
    @test size(ordinary["embeddings"]) == (3, 2)
    @test size(ordinary["base_X"]) == (6, 2, 3, 2)
    @test size(ordinary["Y_clean"]) == (6, 2, 3, 2)
    @test length(unique(get_formula.(ordinary["models"]))) == 3
    for task_index in 1:3
        rows = ((task_index - 1) * 2 + 1):(task_index * 2)
        @test predict(ordinary["models"][task_index], ordinary["base_X"][rows, :, :, :]) ≈
            ordinary["Y_clean"][rows, :, :, :]
    end
end

@testitem "Collection synthesis represents time as a list for series tasks" begin
    using Test
    using LambdaRegression.FrontendAdapter:
        generate_collection_pretraining_tasks, predict

    tasks = generate_collection_pretraining_tasks(
        task_kind=:series, task_count=1, samples_per_task=2,
        n_frames=2, n_elements=3, input_dim=2, output_dim=1,
        random_state=34, prefix_tokens=1,
    )
    @test size(tasks["X"]) == (2, 2, 4, 2)
    @test size(tasks["Y_clean"]) == (2, 2, 3, 1)
    @test predict(tasks["models"][1], tasks["base_X"]) ≈ tasks["Y_clean"]
end

@testitem "Collection synthetic generation accepts rank-three complete groups" begin
    using Test
    using LambdaRegression.FrontendAdapter:
        generate_collection_synthetic_data, predict

    X = reshape(Float64.(1:24), 4, 3, 2) ./ 10
    generated = generate_collection_synthetic_data(
        X; task_kind=:array, output_dim=1, random_state=8,
        operator_set=(:+, :*),
    )
    @test size(generated["Y_clean"]) == (4, 3, 1)
    @test predict(generated["models"][1], X) ≈ generated["Y_clean"]
    @test_throws ArgumentError generate_collection_synthetic_data(X; task_count=2)
    @test_throws ArgumentError generate_collection_synthetic_data(X; prefix_tokens=1)
end

@testitem "Collection task sampler distinguishes array-valued program materials" begin
    using LambdaRegression.FrontendAdapter: generate_pretraining_tasks, predict
    using LambdaRegression.Programs: Program
    using LambdaRegression.TypeSystem: TyLike
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, ConstNode

    sampler(rng; task_index, output_shape, output_ty, kwargs...) = Program(
        AstTree(AbsNode(:x, ConstNode(:values)));
        local_consts=Dict{Symbol,Any}(:values => fill(Float64(task_index), output_shape)),
        local_consts_ty=Dict{Symbol,TyLike}(:values => output_ty),
    )
    tasks = generate_pretraining_tasks(task_count=2, samples_per_task=1,
        n_frames=1, n_elements=2, input_dim=1, output_dim=1,
        program_sampler=sampler, max_attempts=2, max_total_attempts=2,
        random_state=41)
    @test tasks["Y_clean"][1, :, :, :] == ones(1, 2, 1)
    @test tasks["Y_clean"][2, :, :, :] == fill(2.0, 1, 2, 1)
    @test predict(tasks["models"][2], tasks["base_X"][2:2, :, :, :]) ==
        fill(2.0, 1, 1, 2, 1)
end
