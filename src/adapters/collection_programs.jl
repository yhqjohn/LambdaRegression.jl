# Axis 1 indexes examples. For :series, axis 2 is the list within each sample.

using Random: MersenneTwister
using ..ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, FunctionDefNode, IndexNode
using ..Programs: Program
using ..TypeSystem: Ty, TyFun, TyList, FloatTy, TyLike, get_type_ctx
using ..Regression: TensorSchema, TensorSeries, SupervisedDataset
using ..Regression: default_components, default_config, prepare_task_evaluator
using ..Regression: initialize_age_search, advance_age_search!, mse
using ..Regression: Regressor
using ..CanonicalGeneration: GenerationConfig, GenerationWeights
using ..CanonicalGeneration: default_head_providers, SafeListAccessProvider
using ..Utils: Err
using ..Mutations: BloatPolicy
import ..Primitives

export CollectionModel, collection_world, collection_components
export fit_collection, predict, get_formula, get_complexity
export normalize_collection_array, normalize_collection_series

const _COLLECTION_OPERATORS = (
    :+, :-, :*, :/, :^, :max, :min,
    :sin, :cos, :tan, :exp, :log, :abs, :sqrt, :square,
)
const _COLLECTION_COMPONENT_KEYS = Set((
    :tyfactory_ctor, :const_symfactory_ctor,
    :functiondef_generation_weight, :generation_providers,
    :contextual_expansion_site_selection, :binary_composition_site_selection,
    :candidate_admissibility, :functiondef_provider,
    :functiondef_recursive_body_site_weight, :mutations, :mutation_chooser,
    :mutation_provider, :mutation_max_try, :mutation_prune_consts,
    :full_mutation_provider, :schedule_budget, :schedule_offset,
    :mutation_adaptive_policy, :bloat_policy, :candidate_failure,
    :loss_failure, :crossover_op, :mutation_driver,
    :generation_config_factory, :force_lambda_root_probability,
    :init_candidates_per_individual, :init_target_nodes, :init_max_try,
    :init_growth_attempts, :init_growth_max_try, :init_growth_mutation,
    :init_population, :initialization_observer, :complexity, :loss,
    :prepare_evaluate, :evaluate,
))
const _COLLECTION_RESERVED_COMPONENT_KEYS = Set((
    :global_consts, :global_consts_ty, :input_slots, :input_types,
))
const _COLLECTION_SEARCH_KEYS = Set((
    :population_count, :recent_capacity, :seed,
    :max_attempts, :max_evaluations, :mutation_weight, :crossover_weight,
    :replay,
))
const _COLLECTION_FIT_KEYS = Set((
    :task_kind, :operator_set, :population_size, :generations,
    :max_complexity, :max_evaluations, :parsimony_coefficient, :tournament_size,
    :mutation_rate, :crossover_rate, :random_state, :initial_programs,
    :regressor_config, :components_kwargs, :search_kwargs, :length_penalty,
    :global_consts, :global_consts_ty,
))

const _COLLECTION_OPERATOR_VALUES = Dict{Symbol,Any}(
    :+ => Primitives.FADD,
    :- => Primitives.FSUB,
    :* => Primitives.FMUL,
    :/ => Primitives.FDIV,
    :^ => Primitives.FPOW,
    :max => Primitives.FMAX,
    :min => Primitives.FMIN,
    :sin => Primitives.FSIN,
    :cos => Primitives.FCOS,
    :tan => Primitives.FTAN,
    :exp => Primitives.FEXP,
    :log => Primitives.FLOG,
    :abs => Primitives.FABS,
    :sqrt => Primitives.FSQRT,
    :square => Primitives.SQR,
)
const _COLLECTION_OPERATOR_NAMES = Dict{Symbol,Symbol}(
    :+ => :FADD,
    :- => :FSUB,
    :* => :FMUL,
    :/ => :FDIV,
    :^ => :FPOW,
    :max => :FMAX,
    :min => :FMIN,
    :sin => :FSIN,
    :cos => :FCOS,
    :tan => :FTAN,
    :exp => :FEXP,
    :log => :FLOG,
    :abs => :FABS,
    :sqrt => :FSQRT,
    :square => :SQR,
)

struct CollectionModel
    program::Program
    target_ty::Ty
    components::NamedTuple
    config::NamedTuple
    task_kind::Symbol
    input_shape::Tuple
    output_shape::Tuple
    input_schema::TensorSchema
    output_schema::Union{Nothing,TensorSchema}
    length_penalty::Float64
    loss::Union{Nothing,Float64}
    objective_loss::Union{Nothing,Float64}
    objective_calls::Int
    sample_executions::Int
    sample_count::Int
end

function CollectionModel(
    program::Program,
    world::NamedTuple;
    task_kind::Symbol=:array,
    input_shape::Tuple,
    output_shape::Tuple,
    config::NamedTuple=default_config(procedural_max_steps=4096),
    length_penalty::Real=1.0,
    loss=nothing,
    objective_loss=nothing,
    objective_calls::Integer=0,
    sample_executions::Integer=0,
    sample_count::Integer=0,
)
    task_kind in (:array, :series) || throw(ArgumentError(
        "task_kind must be :array or :series",
    ))
    coefficient = Float64(length_penalty)
    isfinite(coefficient) && coefficient >= 0 || throw(ArgumentError(
        "length_penalty must be finite and non-negative",
    ))
    return CollectionModel(
        program, world.target_ty, world.components, config, task_kind,
        Tuple(input_shape), Tuple(output_shape), TensorSchema(Tuple(input_shape)),
        isempty(output_shape) ? nothing : TensorSchema(Tuple(output_shape)),
        coefficient,
        loss === nothing ? nothing : Float64(loss),
        objective_loss === nothing ? nothing : Float64(objective_loss),
        Int(objective_calls), Int(sample_executions), Int(sample_count),
    )
end

function _collection_options(value, label::AbstractString)
    pairs_iter = if value isa NamedTuple || value isa AbstractDict || value isa Base.Pairs
        pairs(value)
    else
        throw(ArgumentError("$label must be a NamedTuple or dictionary"))
    end
    result = Dict{Symbol,Any}()
    for (raw_key, item) in pairs_iter
        key = raw_key isa Symbol ? raw_key : Symbol(String(raw_key))
        haskey(result, key) && throw(ArgumentError("$label contains duplicate key $key"))
        result[key] = item
    end
    return result
end

function _options_namedtuple(options::AbstractDict{Symbol})
    names = sort!(collect(keys(options)); by=string)
    return NamedTuple{Tuple(names)}(Tuple(options[name] for name in names))
end

function _array_value(shape::Tuple, value::Float64)
    isempty(shape) && return value
    return fill(value, shape)
end

function _collection_operator_globals(operator_set)
    unary, binary = _split_operator_set(operator_set)
    requested = Set((unary..., binary...))
    globals = Dict{Symbol,Any}(
        :PAIR => Primitives.PAIR,
        :FST => Primitives.FST,
        :SND => Primitives.SND,
        :AMAP => Primitives.AMAP,
        :AFOLDL => Primitives.AFOLDL,
        :LENF => Primitives.LENF,
        :CONS => Primitives.CONS,
        :HEAD => Primitives.HEAD,
        :TAIL => Primitives.TAIL,
        :LENGTH => Primitives.LENGTH,
        :CASENILERR => Primitives.CASENILERR,
        :UNCONS_CASE => Primitives.UNCONS_CASE,
        :TOLIST => Primitives.TOLIST,
        :ZERO => 0.0,
        :ONE => 1.0,
    )
    for op in requested
        globals[_COLLECTION_OPERATOR_NAMES[op]] = _COLLECTION_OPERATOR_VALUES[op]
    end
    return globals
end

# target_ty is the complete input-to-output signature.
# Overrides cannot replace input slots or type boundaries.
function collection_world(
    target_ty::TyFun;
    operator_set=_COLLECTION_OPERATORS,
    global_consts::AbstractDict=Dict{Symbol,Any}(),
    global_consts_ty::Union{Nothing,AbstractDict}=nothing,
    components_kwargs=NamedTuple(),
    loss=mse,
    loss_failure=issue -> issue isa DimensionMismatch || issue isa ArgumentError,
)
    options = _collection_options(components_kwargs, "components_kwargs")
    unknown = setdiff(Set(keys(options)), _COLLECTION_COMPONENT_KEYS)
    reserved = intersect(Set(keys(options)), _COLLECTION_RESERVED_COMPONENT_KEYS)
    isempty(unknown) || throw(ArgumentError(
        "unsupported components_kwargs: $(join(string.(sort!(collect(unknown))), ", "))",
    ))
    isempty(reserved) || throw(ArgumentError(
        "components_kwargs cannot override task input/world fields: " *
        join(string.(sort!(collect(reserved))), ", "),
    ))

    globals = _collection_operator_globals(operator_set)
    merge!(globals, Dict{Symbol,Any}(
        (key isa Symbol ? key : Symbol(String(key))) => value
        for (key, value) in pairs(global_consts)
    ))
    base_types = global_consts_ty === nothing ? Dict{Symbol,TyLike}() :
        Dict{Symbol,TyLike}(
            (key isa Symbol ? key : Symbol(String(key))) => value
            for (key, value) in pairs(global_consts_ty)
        )
    get_type_ctx(globals, base_types)
    if target_ty.res isa TyList
        globals[:NIL] = TensorSeries(AbstractArray[])
        base_types[:NIL] = target_ty.res
    end
    Set(keys(base_types)) == Set(keys(globals)) || throw(ArgumentError(
        "global_consts and global_consts_ty must define the same symbols",
    ))

    base = (
        global_consts=globals,
        global_consts_ty=base_types,
        input_slots=Symbol[:x],
        input_types=Dict{Symbol,TyLike}(:x => target_ty.arg),
        loss,
        loss_failure,
    )
    merged = merge(base, _options_namedtuple(options))
    components = default_components(; merged...)
    return (; target_ty, global_consts=globals,
        global_consts_ty=base_types, components)
end

collection_components(target_ty::TyFun; kwargs...) =
    collection_world(target_ty; kwargs...).components

function normalize_collection_array(
    value,
    schema::TensorSchema{N};
    label::AbstractString="array value",
) where {N}
    result = Array{Float64}(undef, schema.shape)
    function assign!(node, depth::Int, prefix::Tuple)
        if depth > N
            node isa Real && isfinite(node) || throw(ArgumentError(
                "$label must contain only finite real leaves",
            ))
            result[prefix...] = Float64(node)
            return nothing
        end
        node isa AbstractArray || throw(DimensionMismatch(
            "$label does not have the declared nested shape $(schema.shape)",
        ))
        dimension_length = ndims(node) > 1 ? size(node, 1) : length(node)
        dimension_length == schema.shape[depth] || throw(DimensionMismatch(
            "$label does not have the declared nested shape $(schema.shape)",
        ))
        for index in 1:schema.shape[depth]
            child = ndims(node) > 1 ? selectdim(node, 1, index) : node[index]
            assign!(child, depth + 1, (prefix..., index))
        end
        return nothing
    end
    if value isa AbstractArray && size(value) == schema.shape &&
            eltype(value) <: Real
        all(isfinite, value) || throw(ArgumentError("$label contains non-finite values"))
        return Float64.(value)
    end
    assign!(value, 1, ())
    return result
end

function normalize_collection_series(
    value,
    schema::TensorSchema{N};
    label::AbstractString="series value",
) where {N}
    value isa AbstractVector || throw(ArgumentError("$label must be a vector of tensors"))
    tensors = Array{Float64,N}[]
    for (index, tensor) in pairs(value)
        push!(tensors, normalize_collection_array(
            tensor, schema; label="$label tensor $index",
        ))
    end
    return TensorSeries(tensors)
end

function _finite_real_array(value, label::AbstractString)
    value isa AbstractArray{<:Real} || throw(ArgumentError("$label must be a real array"))
    all(isfinite, value) || throw(ArgumentError("$label must be finite"))
    return Float64.(value)
end

function _array_dataset(X, y)
    x = _finite_real_array(X, "X")
    y isa AbstractArray{<:Real} || throw(ArgumentError("y must be a real array"))
    all(isfinite, y) || throw(ArgumentError("y must be finite"))
    ndims(x) >= 2 || throw(ArgumentError("array task X must have a leading batch axis and at least one value axis"))
    size(x, 1) > 0 || throw(ArgumentError("X batch must not be empty"))
    batch = size(x, 1)
    input_shape = Tuple(size(x)[2:end])
    input_schema = TensorSchema(input_shape)
    xs = [Array(selectdim(x, 1, i)) for i in 1:batch]

    if ndims(y) == 1
        length(y) == batch || throw(DimensionMismatch("scalar target count must equal X batch size"))
        ys = Float64.(collect(y))
        return xs, ys, input_schema, nothing, input_shape, (),
            TyFun(input_schema.ty, FloatTy)
    end
    ndims(y) >= 2 || throw(ArgumentError("array targets need a batch axis"))
    size(y, 1) == batch || throw(DimensionMismatch("X and y batch sizes differ"))
    output_shape = Tuple(size(y)[2:end])
    output_schema = TensorSchema(output_shape)
    dense_y = Float64.(y)
    ys = [Array(selectdim(dense_y, 1, i)) for i in 1:batch]
    return xs, ys, input_schema, output_schema, input_shape, output_shape,
        TyFun(input_schema.ty, output_schema.ty)
end

function _series_batch_from_dense(value, label::AbstractString)
    dense = _finite_real_array(value, label)
    ndims(dense) >= 3 || throw(ArgumentError(
        "$label dense series must have batch, sequence, and tensor axes",
    ))
    batch, steps = size(dense, 1), size(dense, 2)
    batch > 0 || throw(ArgumentError("$label batch must not be empty"))
    inner_shape = Tuple(size(dense)[3:end])
    schema = TensorSchema(inner_shape)
    N = length(inner_shape)
    samples = TensorSeries[]
    for sample in 1:batch
        tensors = Array{Float64,N}[]
        sample_view = selectdim(dense, 1, sample)
        for step in 1:steps
            push!(tensors, Array(selectdim(sample_view, 1, step)))
        end
        push!(samples, TensorSeries(tensors))
    end
    return samples, schema, true
end

function _series_batch_from_vectors(value, label::AbstractString)
    value isa AbstractVector || throw(ArgumentError(
        "$label must be a batch vector of TensorSeries values",
    ))
    isempty(value) && throw(ArgumentError("$label batch must not be empty"))
    raw_series = Vector{AbstractVector}(undef, length(value))
    first_tensor = nothing
    for (sample_index, sample) in pairs(value)
        items = sample isa TensorSeries ? sample : sample
        items isa AbstractVector || throw(ArgumentError(
            "$label[$sample_index] must be a TensorSeries or vector of tensors",
        ))
        raw_series[sample_index] = items
        for tensor in items
            tensor isa AbstractArray || throw(ArgumentError(
                "$label[$sample_index] must contain arrays",
            ))
            first_tensor === nothing && (first_tensor = tensor)
        end
    end
    first_tensor === nothing && throw(ArgumentError(
        "$label cannot infer a tensor shape when every series is empty",
    ))
    schema = TensorSchema(Tuple(size(first_tensor)))
    samples = TensorSeries[]
    for (index, series) in pairs(raw_series)
        push!(samples, normalize_collection_series(
            series, schema; label="$label[$index]",
        ))
    end
    return samples, schema, false
end

function _series_dataset(X, y)
    xs, input_schema, x_dense = X isa AbstractArray{<:Real} ?
        _series_batch_from_dense(X, "X") : _series_batch_from_vectors(X, "X")
    if y isa AbstractArray{<:Real} && ndims(y) == 1
        length(y) == length(xs) || throw(DimensionMismatch("X and y batch sizes differ"))
        all(isfinite, y) || throw(ArgumentError("y must be finite"))
        ys = Float64.(collect(y))
        output_schema = nothing
        y_dense = false
        output_shape = ()
        target_ty = TyFun(TyList(input_schema.ty), FloatTy)
        return xs, ys, input_schema, output_schema,
            input_schema.shape, output_shape, target_ty, x_dense, y_dense
    end
    ys, output_schema, y_dense = y isa AbstractArray{<:Real} ?
        _series_batch_from_dense(y, "y") : _series_batch_from_vectors(y, "y")
    length(xs) == length(ys) || throw(DimensionMismatch("X and y batch sizes differ"))
    target_ty = TyFun(TyList(input_schema.ty), TyList(output_schema.ty))
    return xs, ys, input_schema, output_schema,
        input_schema.shape, output_schema.shape, target_ty, x_dense, y_dense
end

function _array_loss(output_schema, prediction, target)
    if output_schema === nothing
        prediction isa Real && isfinite(prediction) || throw(ArgumentError(
            "scalar-output program must return a finite real",
        ))
        target isa Real && isfinite(target) || throw(ArgumentError("target must be finite real"))
        return (Float64(prediction) - Float64(target))^2
    end
    actual = normalize_collection_array(prediction, output_schema; label="prediction")
    expected = normalize_collection_array(target, output_schema; label="target")
    return mse(actual, expected)
end

function _series_loss(output_schema, prediction, target, length_penalty)
    if output_schema === nothing
        prediction isa Real && isfinite(prediction) || throw(ArgumentError(
            "scalar-output program must return a finite real",
        ))
        target isa Real && isfinite(target) || throw(ArgumentError("target must be finite real"))
        return (Float64(prediction) - Float64(target))^2
    end
    predicted = normalize_collection_series(prediction, output_schema; label="prediction")
    expected = normalize_collection_series(target, output_schema; label="target")
    count = 0
    squared_error = 0.0
    for index in 1:min(length(predicted), length(expected))
        for (left, right) in zip(predicted[index], expected[index])
            delta = left - right
            squared_error += delta * delta
            count += 1
        end
    end
    aligned = count == 0 ? 0.0 : squared_error / count
    cardinality = abs(length(predicted) - length(expected)) /
        max(length(predicted), length(expected), 1)
    return aligned + length_penalty * cardinality
end

function _series_zero(shape::Tuple, value::Float64)
    array = fill(value, shape)
    return TensorSeries(AbstractArray[array])
end

function _fit_global_values(operator_set, output_schema, output_shape, target_ty, task_kind)
    zero_value = task_kind === :series && output_schema === nothing ? 0.0 :
        task_kind === :series ? _series_zero(output_shape, 0.0) : _array_value(output_shape, 0.0)
    one_value = task_kind === :series && output_schema === nothing ? 1.0 :
        task_kind === :series ? _series_zero(output_shape, 1.0) : _array_value(output_shape, 1.0)
    globals = Dict{Symbol,Any}(
        :COLLECTION_ZERO => zero_value,
        :COLLECTION_ONE => one_value,
    )
    types = Dict{Symbol,TyLike}(
        :COLLECTION_ZERO => target_ty.res,
        :COLLECTION_ONE => target_ty.res,
    )
    return globals, types
end

function _baseline_programs(target_ty::TyFun)
    programs = Program[
        Program(AstTree(AbsNode(:x, ConstNode(:COLLECTION_ZERO)))),
        Program(AstTree(AbsNode(:x, ConstNode(:COLLECTION_ONE)))),
    ]
    target_ty.arg == target_ty.res && push!(programs,
        Program(AstTree(AbsNode(:x, IndexNode(0)))),
    )
    return programs
end

function _normalize_initial_programs(value)
    value === nothing && return Program[]
    value isa AbstractVector && all(program -> program isa Program, value) ||
        throw(ArgumentError("initial_programs must be a vector of typed Program values"))
    return Program[value...]
end

function _fit_options(kwargs)
    options = _collection_options(kwargs, "fit_collection keywords")
    unknown = setdiff(Set(keys(options)), _COLLECTION_FIT_KEYS)
    isempty(unknown) || throw(ArgumentError(
        "unsupported fit_collection keyword(s): $(join(string.(sort!(collect(unknown))), ", "))",
    ))
    return options
end

function _merge_regressor_config(options, population_size, tournament_size)
    values = Dict{Symbol,Any}(
        :population_size => population_size,
        :tournament_size => tournament_size,
        :procedural_max_steps => 4096,
    )
    overrides = _collection_options(get(options, :regressor_config, NamedTuple()), "regressor_config")
    allowed = Set(keys(default_config()))
    unknown = setdiff(Set(keys(overrides)), allowed)
    isempty(unknown) || throw(ArgumentError(
        "unsupported regressor_config keys: $(join(string.(sort!(collect(unknown))), ", "))",
    ))
    merge!(values, overrides)
    if haskey(values, :parallel) && values[:parallel] isa AbstractDict
        values[:parallel] = _options_namedtuple(_collection_options(values[:parallel], "regressor_config.parallel"))
    end
    return default_config(; _options_namedtuple(values)...)
end

function _merge_components_kwargs(options, max_complexity)
    component_values = _collection_options(get(options, :components_kwargs, NamedTuple()), "components_kwargs")
    if haskey(component_values, :bloat_policy)
        haskey(options, :max_complexity) && throw(ArgumentError(
            "set max_complexity or components_kwargs.bloat_policy, not both",
        ))
    else
        component_values[:bloat_policy] = BloatPolicy(
            enabled=true, max_nodes=max_complexity, max_nodes_delta=max_complexity,
        )
    end
    get!(component_values, :init_candidates_per_individual, 1)
    get!(component_values, :init_growth_attempts, 0)
    if !haskey(component_values, :generation_config_factory)
        providers = get(component_values, :generation_providers,
            (default_head_providers()..., SafeListAccessProvider()))
        functiondef_weight = get(component_values, :functiondef_generation_weight, 1.0)
        component_values[:generation_config_factory] = rng -> GenerationConfig(
            ; rng,
            providers,
            weights=GenerationWeights(default=1, function_def=functiondef_weight),
            guarded_expansion_depth=0,
            sampling_depth_limit=4,
            sampling_node_budget=32,
        )
    end
    return _options_namedtuple(component_values)
end

function _record_best(state, parsimony_coefficient)
    records = collect(values(state.archive))
    isempty(records) && return state.best
    scores = [record.loss + parsimony_coefficient * record.complexity for record in records]
    return records[argmin(scores)]
end

function _series_input_batch(value, input_schema::TensorSchema)
    if value isa AbstractArray{<:Real}
        samples, schema, _ = _series_batch_from_dense(value, "prediction X")
        schema.shape == input_schema.shape || throw(DimensionMismatch(
            "model expects tensor shape $(input_schema.shape), received $(schema.shape)",
        ))
        return samples, true
    end
    samples, schema, _ = _series_batch_from_vectors(value, "prediction X")
    schema.shape == input_schema.shape || throw(DimensionMismatch(
        "model expects tensor shape $(input_schema.shape), received $(schema.shape)",
    ))
    return samples, false
end

function _array_input_batch(value, input_schema::TensorSchema)
    value isa AbstractArray{<:Real} || throw(ArgumentError("prediction X must be a real dense array"))
    dense = _finite_real_array(value, "prediction X")
    ndims(dense) == length(input_schema.shape) + 1 || throw(DimensionMismatch(
        "prediction X must include one leading batch axis",
    ))
    Tuple(size(dense)[2:end]) == input_schema.shape || throw(DimensionMismatch(
        "model expects sample shape $(input_schema.shape), received $(Tuple(size(dense)[2:end]))",
    ))
    return [Array(selectdim(dense, 1, i)) for i in axes(dense, 1)]
end

function _evaluate_group(model::CollectionModel, evaluator, sample)
    result = evaluator(sample)
    result isa Err && throw(ArgumentError("program evaluation failed: $(result.value)"))
    if model.task_kind === :array
        if model.output_schema === nothing
            result isa Real && isfinite(result) || throw(ArgumentError(
                "program returned a non-finite or non-real scalar",
            ))
            return Float64(result)
        end
        return normalize_collection_array(result, model.output_schema; label="prediction")
    end
    if model.output_schema === nothing
        result isa Real && isfinite(result) || throw(ArgumentError(
            "program returned a non-finite or non-real scalar",
        ))
        return Float64(result)
    end
    return normalize_collection_series(result, model.output_schema; label="prediction")
end

function predict(model::CollectionModel, X)
    evaluator = prepare_task_evaluator(
        model.program, model.target_ty, model.components, model.config,
    )
    if model.task_kind === :array
        samples = _array_input_batch(X, model.input_schema)
        predictions = [_evaluate_group(model, evaluator, sample) for sample in samples]
        isempty(predictions) && throw(ArgumentError("prediction batch must not be empty"))
        model.output_schema === nothing && return Float64[predictions...]
        return stack(predictions; dims=1)
    end

    samples, dense_input = _series_input_batch(X, model.input_schema)
    if model.output_schema === nothing
        return Float64[
            _evaluate_group(model, evaluator, sample) for sample in samples
        ]
    end
    predictions = TensorSeries[
        _evaluate_group(model, evaluator, sample) for sample in samples
    ]
    if dense_input && !isempty(predictions) &&
            all(length(series) == length(first(predictions)) for series in predictions)
        if isempty(first(predictions))
            return Array{Float64}(undef,
                length(predictions), 0, model.output_shape...,
            )
        end
        per_sample = [stack(collect(series); dims=1) for series in predictions]
        return stack(per_sample; dims=1)
    end
    return predictions
end

function _constant_description(value)
    value isa Function && return "<function>"
    return repr(value)
end

function get_formula(model::CollectionModel)
    formula = sprint(show, MIME"text/plain"(), model.program.tree)
    names = Set{Symbol}()
    for node in model.program.tree
        node isa ConstNode && push!(names, node.name)
    end
    globals = [name for name in names if haskey(model.components.global_consts, name)]
    locals = [name for name in names if haskey(model.program.local_consts, name)]
    sort!(globals; by=string)
    sort!(locals; by=string)
    isempty(globals) || (formula *= "\nGlobals: " * join((
        "$name=$(_constant_description(model.components.global_consts[name]))" for name in globals
    ), "; "))
    isempty(locals) || (formula *= "\nLocal constants: " * join((
        "$name=$(_constant_description(model.program.local_consts[name]))" for name in locals
    ), "; "))
    return formula
end

get_complexity(model::CollectionModel) = length(model.program.tree)

# max_evaluations includes initialization; generations bounds proposal attempts.
# Parsimony selects the archive result without changing the reported raw loss.
function fit_collection(X, y; kwargs...)
    options = _fit_options(kwargs)
    task_kind = Symbol(get(options, :task_kind, :array))
    task_kind in (:array, :series) || throw(ArgumentError(
        "fit_collection supports task_kind=:array or :series; scalar tasks use fit",
    ))
    population_size = get(options, :population_size, 4)
    generations = get(options, :generations, 2)
    tournament_size = get(options, :tournament_size, 2)
    for (name, value, minimum) in ((:population_size, population_size, 2),
            (:generations, generations, 0), (:tournament_size, tournament_size, 1))
        value isa Integer && !(value isa Bool) && value >= minimum || throw(ArgumentError(
            "$name must be an integer >= $minimum",
        ))
    end
    population_size, generations, tournament_size =
        Int(population_size), Int(generations), Int(tournament_size)
    max_complexity = get(options, :max_complexity, 64)
    max_complexity isa Integer && !(max_complexity isa Bool) && max_complexity >= 2 ||
        throw(ArgumentError("max_complexity must be an integer of at least 2"))
    parsimony = get(options, :parsimony_coefficient, 0.01)
    parsimony isa Real && isfinite(parsimony) && parsimony >= 0 || throw(ArgumentError(
        "parsimony_coefficient must be finite and non-negative",
    ))
    mutation_rate = get(options, :mutation_rate, 0.8)
    crossover_rate = get(options, :crossover_rate, 0.2)
    for (name, value) in ((:mutation_rate, mutation_rate), (:crossover_rate, crossover_rate))
        value isa Real && isfinite(value) && 0 <= value <= 1 || throw(ArgumentError(
            "$name must be finite and in [0, 1]",
        ))
    end
    mutation_rate + crossover_rate > 0 || throw(ArgumentError(
        "mutation_rate and crossover_rate cannot both be zero",
    ))
    length_penalty = get(options, :length_penalty, 1.0)
    length_penalty isa Real && isfinite(length_penalty) && length_penalty >= 0 ||
        throw(ArgumentError("length_penalty must be finite and non-negative"))
    base_seed = _base_seed(get(options, :random_state, 0))

    if task_kind === :array
        xs, ys, input_schema, output_schema, input_shape, output_shape, target_ty =
            _array_dataset(X, y)
        x_dense = true
    else
        xs, ys, input_schema, output_schema, input_shape, output_shape, target_ty,
            x_dense, _ = _series_dataset(X, y)
    end
    sample_count = length(xs)
    operator_set = get(options, :operator_set, _COLLECTION_OPERATORS)
    custom_globals = _collection_options(get(options, :global_consts, Dict()), "global_consts")
    global_consts = Dict{Symbol,Any}(custom_globals)
    global_consts_ty = _collection_options(get(options, :global_consts_ty, Dict()), "global_consts_ty")
    seed_globals, seed_types = _fit_global_values(
        operator_set, output_schema, output_shape, target_ty, task_kind,
    )
    merge!(seed_globals, global_consts)
    merge!(seed_types, Dict{Symbol,TyLike}(global_consts_ty))

    if task_kind === :array
        loss = (prediction, target) -> _array_loss(output_schema, prediction, target)
    else
        loss = (prediction, target) -> _series_loss(
            output_schema, prediction, target, Float64(length_penalty),
        )
    end
    component_options = _merge_components_kwargs(options, Int(max_complexity))
    world = collection_world(
        target_ty;
        operator_set,
        global_consts=seed_globals,
        global_consts_ty=seed_types,
        components_kwargs=component_options,
        loss,
        loss_failure=issue -> issue isa DimensionMismatch || issue isa ArgumentError,
    )

    config = _merge_regressor_config(options, population_size, tournament_size)
    population_size = config.population_size
    population_size >= 2 || throw(ArgumentError("regressor_config.population_size must be at least 2"))
    initial_programs = _normalize_initial_programs(get(options, :initial_programs, nothing))
    length(initial_programs) <= population_size || throw(ArgumentError(
        "initial_programs cannot exceed population_size",
    ))
    initial_programs = [_import_program_seed(program, world.components) for program in initial_programs]
    baselines = _baseline_programs(target_ty)
    init_source = world.components.init_population
    initialization_fallbacks = Ref(0)
    initialization_failure = Ref{Union{Nothing,String}}(nothing)
    init_population = function (reg, data; N, seed, global_type_ctx)
        length(initial_programs) <= N || throw(ArgumentError(
            "initial_programs cannot exceed the requested population size",
        ))
        initial = copy(initial_programs)
        if length(initial) < N
            push!(initial, baselines[1])
        end
        remaining = N - length(initial)
        if remaining > 0
            generated = init_source(
                reg, data; N=remaining, seed, global_type_ctx,
            )
            if generated isa Err
                initialization_fallbacks[] += remaining
                initialization_failure[] === nothing &&
                    (initialization_failure[] = sprint(show, generated.value))
            else
                accepted = Program[
                    candidate for candidate in generated
                    if candidate isa Program && length(candidate.tree) <= max_complexity
                ]
                append!(initial, accepted[1:min(length(accepted), remaining)])
                if length(accepted) < remaining
                    initialization_fallbacks[] += remaining - length(accepted)
                    initialization_failure[] === nothing &&
                        (initialization_failure[] = "typed initializer returned no additional candidates within max_complexity")
                end
            end
        end
        while length(initial) < N
            push!(initial, baselines[mod1(length(initial) + 1, length(baselines))])
        end
        return initial
    end
    components = merge(world.components, (; init_population))
    world = merge(world, (; components))
    reg = Regressor(
        target_ty, components, config,
        MersenneTwister(base_seed),
    )

    search_options = _collection_options(get(options, :search_kwargs, NamedTuple()), "search_kwargs")
    unknown_search = setdiff(Set(keys(search_options)), _COLLECTION_SEARCH_KEYS)
    isempty(unknown_search) || throw(ArgumentError(
        "unsupported search_kwargs: $(join(string.(sort!(collect(unknown_search))), ", "))",
    ))
    search_seed = pop!(search_options, :seed, base_seed)
    search_seed isa Integer && !(search_seed isa Bool) && search_seed >= 0 ||
        throw(ArgumentError("search_kwargs.seed must be a non-negative integer"))
    population_count = pop!(search_options, :population_count, 1)
    recent_capacity = pop!(search_options, :recent_capacity, 16)
    for (name, value, minimum) in ((:population_count, population_count, 1),
            (:recent_capacity, recent_capacity, 0))
        value isa Integer && !(value isa Bool) && value >= minimum || throw(ArgumentError(
            "search_kwargs.$name must be an integer >= $minimum",
        ))
    end
    population_count, recent_capacity = Int(population_count), Int(recent_capacity)
    max_attempts = pop!(search_options, :max_attempts, population_size * generations)
    max_attempts isa Integer && !(max_attempts isa Bool) && max_attempts >= 0 ||
        throw(ArgumentError("search_kwargs.max_attempts must be a non-negative integer"))
    max_attempts = Int(max_attempts)
    requested_total = get(options, :max_evaluations, nothing)
    phase_budget = pop!(search_options, :max_evaluations, nothing)
    requested_total === nothing || phase_budget === nothing || throw(ArgumentError(
        "set top-level max_evaluations or search_kwargs.max_evaluations, not both",
    ))
    initial_budget = population_count * population_size
    if requested_total !== nothing
        requested_total isa Integer && !(requested_total isa Bool) && requested_total >= initial_budget ||
            throw(ArgumentError(
                "max_evaluations must be an integer >= initialization budget $initial_budget",
            ))
        max_evaluations = Int(requested_total) - initial_budget
    elseif phase_budget !== nothing
        phase_budget isa Integer && !(phase_budget isa Bool) && phase_budget >= 0 ||
            throw(ArgumentError("search_kwargs.max_evaluations must be a non-negative integer"))
        max_evaluations = Int(phase_budget)
    else
        max_evaluations = population_size * generations
    end
    mutation_weight = pop!(search_options, :mutation_weight, mutation_rate)
    crossover_weight = pop!(search_options, :crossover_weight, crossover_rate)
    if !haskey(component_options, :schedule_budget)
        components = merge(components, (; schedule_budget=max(1, max_attempts)))
        world = merge(world, (; components))
        reg = Regressor(target_ty, components, config, reg.rng)
    end
    state = initialize_age_search(
        reg, SupervisedDataset(xs, ys);
        population_count, population_size, seed=search_seed, recent_capacity,
    )
    state isa Err && throw(ArgumentError("age-search initialization failed: $(state.value)"))
    report = advance_age_search!(state;
        max_attempts, max_evaluations, mutation_weight, crossover_weight,
        replay=pop!(search_options, :replay, nothing),
    )
    report isa Err && throw(ArgumentError("age-search failed: $(report.value)"))
    record = _record_best(state, Float64(parsimony))
    record === nothing && throw(ArgumentError("age search produced no scored program"))
    model = CollectionModel(
        record.program, world;
        task_kind, input_shape, output_shape, config,
        length_penalty=Float64(length_penalty), loss=record.loss,
        objective_loss=record.loss + Float64(parsimony) * record.complexity,
        objective_calls=state.counters.objective_calls,
        sample_executions=state.counters.sample_executions,
        sample_count,
    )
    return Dict{String,Any}(
        "model" => model,
        "formula" => get_formula(model),
        "complexity" => get_complexity(model),
        "loss" => record.loss,
        "objective_loss" => model.objective_loss,
        "objective_calls" => model.objective_calls,
        "sample_executions" => model.sample_executions,
        "sample_count" => model.sample_count,
        "attempts" => report.attempts,
        "initialization_fallbacks" => initialization_fallbacks[],
        "initialization_failure" => initialization_failure[],
    )
end

@testitem "Collection fit counts whole dense groups and supports array shape changes" begin
    using Test
    using LambdaRegression.FrontendAdapter: fit_collection, predict, get_formula, get_complexity
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Programs: Program

    # Map each (2, 2) group to its two row sums.
    row_step = AbsNode(:acc, AbsNode(:value,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(1)), IndexNode(0))))
    row_sum = AppNode(AppNode(AppNode(ConstNode(:AFOLDL), row_step), ConstNode(:ZERO)), IndexNode(0))
    rows = AppNode(AppNode(ConstNode(:AMAP), AbsNode(:row, row_sum)), IndexNode(0))
    witness = Program(AstTree(AbsNode(:x, rows)))
    X = reshape(Float64.(1:12), 3, 2, 2)
    Y = [sum(X[b, r, :]) for b in 1:3, r in 1:2]
    fitted = fit_collection(X, Y; task_kind=:array, initial_programs=[witness],
        population_size=2, generations=0, operator_set=[:+], random_state=4)
    model = fitted["model"]
    @test model.program isa Program
    @test predict(model, X) == Y
    @test model.sample_count == 3
    @test model.sample_executions == model.objective_calls * 3
    @test fitted["formula"] == get_formula(model)
    @test fitted["complexity"] == get_complexity(model)
    @test occursin("AFOLDL", get_formula(model))
    budgeted = fit_collection(X, Y; task_kind=:array, initial_programs=[witness],
        population_size=2, generations=4, max_evaluations=2,
        operator_set=[:+], random_state=4)
    @test budgeted["objective_calls"] == 2
    @test budgeted["attempts"] == 0
    @test_throws ArgumentError fit_collection(X, Y; task_kind=:array,
        initial_programs=[witness], population_size=2, max_evaluations=1)
end

@testitem "Collection fit routes full mutation providers with explicit cumulative schedule" begin
    using LambdaRegression.FrontendAdapter: fit_collection
    using LambdaRegression.Regression: EncodedMutationProvider
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, IndexNode
    seen = Int[]
    provider = EncodedMutationProvider(
        encode=(parent, target; rng, progress, components) -> begin
            push!(seen, progress.completed_attempts)
            parent
        end,
        call_external=(payload; kwargs...) -> payload,
        decode=(payload, result; kwargs...) -> result)
    identity = Program(AstTree(AbsNode(:x, IndexNode(0))))
    X = reshape(Float64.(1:6), 3, 2)
    fitted = fit_collection(X, X; initial_programs=[identity], population_size=2,
        generations=0, components_kwargs=(; full_mutation_provider=provider,
            schedule_budget=10, schedule_offset=2),
        search_kwargs=(; max_attempts=3, max_evaluations=3,
            mutation_weight=1.0, crossover_weight=0.0))
    @test seen == [2, 3, 4]
    @test fitted["attempts"] == 3
    @test fitted["objective_calls"] == 2
    @test fitted["model"].components.schedule_budget == 10
    empty!(seen)
    automatic = fit_collection(X, X; initial_programs=[identity], population_size=2,
        generations=0, components_kwargs=(; full_mutation_provider=provider),
        search_kwargs=(; max_attempts=3, max_evaluations=3,
            mutation_weight=1.0, crossover_weight=0.0))
    @test seen == [0, 1, 2]
    @test automatic["model"].components.schedule_budget == 3
end

@testitem "Collection fit supports list tensor groups with distinct inner shapes" begin
    using Test
    using LambdaRegression.FrontendAdapter: fit_collection, predict
    using LambdaRegression.Regression: TensorSeries
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode,
        FunctionDefNode, IndexNode
    using LambdaRegression.Programs: Program

    row_step = AbsNode(:acc, AbsNode(:value,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(1)), IndexNode(0))))
    row_sum = AppNode(AppNode(AppNode(ConstNode(:AFOLDL), row_step), ConstNode(:ZERO)), IndexNode(0))
    map_rows = AppNode(AppNode(ConstNode(:AMAP), AbsNode(:row, row_sum)), IndexNode(1))
    map_case = AbsNode(:head, AbsNode(:tail,
        AppNode(AppNode(ConstNode(:CONS), map_rows),
            AppNode(IndexNode(3), IndexNode(0)))))
    body = AbsNode(:xs, AppNode(
        AppNode(AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)), map_case),
        ConstNode(:NIL),
    ))
    witness = Program(AstTree(FunctionDefNode(body)))
    X = [TensorSeries([reshape(Float64.(1:4), 2, 2), reshape(Float64.(5:8), 2, 2)]),
        TensorSeries([reshape(Float64.(2:5), 2, 2)])]
    Y = [TensorSeries([vec(sum(X[1][1]; dims=2)), vec(sum(X[1][2]; dims=2))]),
        TensorSeries([vec(sum(X[2][1]; dims=2))])]
    fitted = fit_collection(X, Y; task_kind=:series, initial_programs=[witness],
        population_size=2, generations=0, operator_set=[:+], random_state=5)
    predictions = predict(fitted["model"], X)
    @test length(predictions) == 2
    @test [collect(t) for t in predictions[1]] == [collect(t) for t in Y[1]]
    @test [collect(t) for t in predictions[2]] == [collect(t) for t in Y[2]]
    @test fitted["model"].input_shape == (2, 2)
    @test fitted["model"].output_shape == (2,)
end

@testitem "Collection world normalizes nested arrays and rejects unknown config" begin
    using Test
    using LambdaRegression.FrontendAdapter: fit_collection, predict, normalize_collection_array
    using LambdaRegression.Regression: TensorSchema

    schema = TensorSchema((2, 2))
    @test normalize_collection_array([[1.0, 2.0], [3.0, 4.0]], schema) == [1.0 2.0; 3.0 4.0]
    mixed_schema = TensorSchema((2, 3, 4))
    mixed = [reshape(Float64.(1:12), 3, 4), reshape(Float64.(13:24), 3, 4)]
    @test normalize_collection_array(mixed, mixed_schema) == stack(mixed; dims=1)
    X = reshape(Float64.(1:8), 2, 2, 2)
    y = [sum(X[b, :, :]) for b in 1:2]
    fitted = fit_collection(X, y; task_kind=:array, population_size=2, generations=0,
        components_kwargs=(init_growth_attempts=0, init_candidates_per_individual=1,
            init_target_nodes=(3,), init_max_try=1))
    @test size(predict(fitted["model"], X)) == (2,)
    @test fitted["model"].output_shape == ()
    @test_throws ArgumentError fit_collection(X, y; made_up_option=1)
    @test_throws ArgumentError fit_collection(X, y; search_kwargs=(made_up_option=1,))
    @test_throws ArgumentError fit_collection(X, y; task_kind=:scalar)
end

@testitem "Collection series reduction keeps one scalar target per whole group" begin
    using Test
    using LambdaRegression.FrontendAdapter: fit_collection, predict
    using LambdaRegression.Regression: TensorSeries
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AppNode, AstTree, ConstNode,
        FunctionDefNode, IndexNode
    using LambdaRegression.Programs: Program

    step = AbsNode(:acc, AbsNode(:value,
        AppNode(AppNode(ConstNode(:FADD), IndexNode(1)), IndexNode(0))))
    tensor_sum = AppNode(
        AppNode(AppNode(ConstNode(:AFOLDL), step), ConstNode(:ZERO)), IndexNode(1),
    )
    reduce_case = AbsNode(:head, AbsNode(:tail,
        AppNode(AppNode(ConstNode(:FADD), tensor_sum),
            AppNode(IndexNode(3), IndexNode(0)))))
    body = AbsNode(:xs, AppNode(
        AppNode(AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)), reduce_case),
        ConstNode(:ZERO),
    ))
    witness = Program(AstTree(FunctionDefNode(body)))
    X = [TensorSeries([Float64[1, 2], Float64[3, 4]]),
        TensorSeries([Float64[2, 5]])]
    y = [sum(sum(tensor) for tensor in X[1]), sum(sum(tensor) for tensor in X[2])]
    fitted = fit_collection(X, y; task_kind=:series, initial_programs=[witness],
        population_size=2, generations=0, operator_set=[:+], random_state=6)
    model = fitted["model"]
    @test predict(model, X) == y
    @test model.output_schema === nothing
    @test model.sample_count == 2
    @test model.sample_executions == model.objective_calls * 2
end

@testitem "Collection series prediction supports an empty list-valued result" begin
    using Test
    using LambdaRegression.FrontendAdapter: CollectionModel, collection_world, predict
    using LambdaRegression.Regression: TensorSchema
    using LambdaRegression.TypeSystem: TyFun, TyList
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, ConstNode
    using LambdaRegression.Programs: Program

    input_schema = TensorSchema((2, 2))
    output_schema = TensorSchema((2,))
    target = TyFun(TyList(input_schema.ty), TyList(output_schema.ty))
    world = collection_world(target; operator_set=[:+])
    empty_program = Program(AstTree(AbsNode(:xs, ConstNode(:NIL))))
    model = CollectionModel(empty_program, world; task_kind=:series,
        input_shape=(2, 2), output_shape=(2,))
    X = reshape(Float64.(1:24), 2, 3, 2, 2)
    @test size(predict(model, X)) == (2, 0, 2)
end
