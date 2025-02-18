"""
    TensorSchema(shape; name=nothing)

Explicit runtime/type boundary for one floating-point NDArray shape. Tensor
values use the structural type `Array[n₁, ... Array[nₖ, Float] ...]`. A dense
Julia `Array{<:AbstractFloat,k}` is the runtime representation; `AMAP` and
`AFOLDL` adapt its first dimension to one outer logical array layer.

`name` is retained as optional descriptive metadata. It does not change the
logical type identity.
"""
struct TensorSchema{N}
    shape::NTuple{N,Int}
    ty::TyArray
    name::Union{Nothing,Symbol}
end

TensorSchema{N}(shape::NTuple{N,Int}, ty::TyArray) where {N} =
    TensorSchema{N}(shape, ty, nothing)

function _tensor_ty(shape::Tuple)
    ty::Ty = FloatTy
    for dimension in Iterators.reverse(shape)
        ty = TyArray(TyNat(dimension), ty)
    end
    return ty::TyArray
end

function TensorSchema(shape::NTuple{N,<:Integer}; name=nothing) where {N}
    N > 0 || throw(ArgumentError("tensor rank must be positive"))
    concrete = ntuple(i -> Int(shape[i]), N)
    all(>(0), concrete) || throw(ArgumentError("tensor dimensions must be positive"))
    label = name === nothing ? nothing : Symbol(name)
    return TensorSchema{N}(concrete, _tensor_ty(concrete), label)
end

"""Semantic ordered series wrapper; it is one input value, never a sample batch."""
struct TensorSeries{A<:AbstractArray} <: AbstractVector{A}
    items::Vector{A}
end

TensorSeries(items::AbstractVector{A}) where {A<:AbstractArray} =
    TensorSeries{A}(collect(items))

Base.IndexStyle(::Type{<:TensorSeries}) = IndexLinear()
Base.size(series::TensorSeries) = (length(series.items),)
Base.length(series::TensorSeries) = length(series.items)
Base.getindex(series::TensorSeries, index::Int) = series.items[index]
Base.getindex(series::TensorSeries, indices::AbstractRange{<:Integer}) =
    TensorSeries(series.items[indices])
Base.iterate(series::TensorSeries, state...) = iterate(series.items, state...)
Base.copy(series::TensorSeries) = TensorSeries(copy(series.items))

function _valid_tensor(value, schema::TensorSchema)::Bool
    value isa AbstractArray || return false
    size(value) == schema.shape || return false
    eltype(value) <: AbstractFloat || return false
    return all(element -> element isa Real && isfinite(element), value)
end

@testitem "Regression recursive tensor-series witnesses infer and evaluate natively" begin
    using Test
    using Random
    using LambdaRegression
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations: Individual
    using LambdaRegression.Procedural: FunctionDefNode, infer_procedural
    using LambdaRegression.Regression:
        TensorSchema, TensorSeries, evaluate_default, series_components,
        series_mse, series_regressor, series_task
    using LambdaRegression.TypeSystem:
        FloatTy, TyFun, TyLike, TyList, FADD, CONS, UNCONS_CASE, get_type_ctx,
        iscompatible

    schema = TensorSchema((2, 2))
    tensor_ty = schema.ty
    list_ty = TyList(tensor_ty)
    a = [1.0 2.0; 3.0 4.0]
    b = [2.0 0.0; -1.0 3.0]
    c = [-1.0 1.0; 2.0 -2.0]
    input = TensorSeries([a, b, c])
    doubled = TensorSeries([2 .* a, 2 .* b, 2 .* c])
    empty_series = TensorSeries(Matrix{Float64}[])
    tscale = tensor -> 2 .* tensor
    tenergy = tensor -> sum(abs2, tensor)

    common_globals = Dict{Symbol,Any}(
        :UNCONS_CASE => UNCONS_CASE,
        :NIL => empty_series,
    )
    common_types = get_type_ctx(Dict{Symbol,Any}(:UNCONS_CASE => UNCONS_CASE))
    common_types[:NIL] = list_ty

    # fix self. λxs. UNCONS_CASE xs
    #   (λhead. λtail. CONS (TSCALE head) (self tail)) NIL
    map_ok = AbsNode(
        :head,
        AbsNode(
            :tail,
            AppNode(
                AppNode(
                    ConstNode(:CONS),
                    AppNode(ConstNode(:TSCALE), IndexNode(1)),
                ),
                AppNode(IndexNode(3), IndexNode(0)),
            ),
        ),
    )
    map_body = AbsNode(
        :xs,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                map_ok,
            ),
            ConstNode(:NIL),
        ),
    )
    map_tree = AstTree(FunctionDefNode(map_body))
    map_target = TyFun(list_ty, list_ty)
    map_globals = merge(
        common_globals,
        Dict{Symbol,Any}(:CONS => CONS, :TSCALE => tscale),
    )
    map_types = merge(
        common_types,
        get_type_ctx(Dict{Symbol,Any}(:CONS => CONS)),
        Dict{Symbol,TyLike}(:TSCALE => TyFun(tensor_ty, tensor_ty)),
    )
    inferred_map = infer_procedural(map_tree, map_types)
    @test !(inferred_map isa LambdaRegression.Utils.Err)
    @test iscompatible(inferred_map, map_target)
    map_task = series_task([input], [doubled]; schema, output=:series)
    map_components = series_components(
        map_task;
        global_consts=map_globals,
        global_consts_ty=map_types,
    )
    map_individual = Individual(map_tree, map_target)
    mapped = evaluate_default(
        map_individual,
        input,
        map_components,
        default_config(procedural_max_steps=1024),
    )
    mapped isa LambdaRegression.Utils.Err && error("map witness failed: $(mapped.value)")
    @test series_mse(mapped, doubled; schema) == 0.0

    # fix self. λxs. UNCONS_CASE xs
    #   (λhead. λtail. FADD (TENERGY head) (self tail)) ZERO
    energy_ok = AbsNode(
        :head,
        AbsNode(
            :tail,
            AppNode(
                AppNode(
                    ConstNode(:FADD),
                    AppNode(ConstNode(:TENERGY), IndexNode(1)),
                ),
                AppNode(IndexNode(3), IndexNode(0)),
            ),
        ),
    )
    energy_body = AbsNode(
        :xs,
        AppNode(
            AppNode(
                AppNode(ConstNode(:UNCONS_CASE), IndexNode(0)),
                energy_ok,
            ),
            ConstNode(:ZERO),
        ),
    )
    energy_tree = AstTree(FunctionDefNode(energy_body))
    energy_target = TyFun(list_ty, FloatTy)
    energy_globals = merge(
        common_globals,
        Dict{Symbol,Any}(:FADD => FADD, :TENERGY => tenergy, :ZERO => 0.0),
    )
    energy_types = merge(
        common_types,
        get_type_ctx(Dict{Symbol,Any}(:FADD => FADD, :ZERO => 0.0)),
        Dict{Symbol,TyLike}(:TENERGY => TyFun(tensor_ty, FloatTy)),
    )
    inferred_energy = infer_procedural(energy_tree, energy_types)
    @test !(inferred_energy isa LambdaRegression.Utils.Err)
    @test iscompatible(inferred_energy, energy_target)
    expected_energy = sum(sum(abs2, tensor) for tensor in input)
    energy_task = series_task([input], [expected_energy]; schema, output=:real)
    energy_components = series_components(
        energy_task;
        global_consts=energy_globals,
        global_consts_ty=energy_types,
    )
    energy = evaluate_default(
        Individual(energy_tree, energy_target),
        input,
        energy_components,
        default_config(procedural_max_steps=1024),
    )
    energy isa LambdaRegression.Utils.Err && error("energy witness failed: $(energy.value)")
    @test energy ≈ expected_energy

    identity_task = series_task(
        [input, TensorSeries([b, c])],
        [input, TensorSeries([b, c])];
        schema,
        output=:series,
    )
    identity_regressor = series_regressor(
        identity_task;
        global_consts=common_globals,
        global_consts_ty=common_types,
        config=default_config(
            population_size=4,
            generations=1,
            seed=UInt64(91),
            mutation_rate=0.0,
            crossover_rate=0.0,
        ),
        rng=MersenneTwister(8000),
    )
    fit_result = fit!(identity_regressor, identity_task.data)
    @test !(fit_result isa LambdaRegression.Utils.Err)
    @test fit_result.best !== nothing
    @test fit_result.history[1].best_loss == 0.0
end

function _validate_series(series, schema::TensorSchema, allow_empty::Bool, label)
    series isa TensorSeries || throw(ArgumentError("$label must be a TensorSeries"))
    allow_empty || !isempty(series) || throw(ArgumentError("$label must not be empty"))
    for (index, tensor) in pairs(series)
        _valid_tensor(tensor, schema) || throw(ArgumentError(
            "$label tensor $index does not match shape $(schema.shape) with finite float elements",
        ))
    end
    return nothing
end

"""Typed dataset and loss policy for one ordered tensor-series task."""
struct SeriesTask{D<:SupervisedDataset,S<:TensorSchema}
    family::Symbol
    data::D
    schema::S
    target_ty::Ty
    input_slot::Symbol
    length_penalty::Float64
    allow_empty::Bool
end

"""
    series_task(X, Y; schema, output, input_slot=:series,
                length_penalty=1.0, allow_empty=false)

Construct an explicit `:series_to_series` or `:series_to_real` task. `X` is the
sample batch and every `X[i]` is one semantic `TensorSeries`.
"""
function series_task(
    X::Vector{TX},
    Y::Vector{TY};
    schema::TensorSchema,
    output::Symbol,
    input_slot::Symbol=:series,
    length_penalty::Real=1.0,
    allow_empty::Bool=false,
) where {TX,TY}
    output in (:series, :real) ||
        throw(ArgumentError("output must be :series or :real"))
    isempty(X) && throw(ArgumentError("series task dataset must not be empty"))
    length(X) == length(Y) || throw(ArgumentError("X and Y must have the same length"))
    coefficient = Float64(length_penalty)
    isfinite(coefficient) && coefficient >= 0 || throw(ArgumentError(
        "length_penalty must be finite and non-negative",
    ))
    for (index, series) in pairs(X)
        _validate_series(series, schema, allow_empty, "X[$index]")
    end
    if output === :series
        for (index, series) in pairs(Y)
            _validate_series(series, schema, allow_empty, "Y[$index]")
        end
    else
        for (index, value) in pairs(Y)
            value isa Real && isfinite(value) || throw(ArgumentError(
                "Y[$index] must be a finite real",
            ))
        end
    end
    input_ty = TyList(schema.ty)
    output_ty = output === :series ? input_ty : FloatTy
    return SeriesTask(
        output === :series ? :series_to_series : :series_to_real,
        SupervisedDataset(X, Y),
        schema,
        TyFun(input_ty, output_ty),
        input_slot,
        coefficient,
        allow_empty,
    )
end

"""
    series_mse(prediction, target; schema, length_penalty=1.0, allow_empty=false)

Mean squared error over every aligned tensor element plus
`length_penalty * abs(m-n)/max(m,n,1)`. Shape mismatch is an invalid candidate,
not a broadcast operation.
"""
function series_mse(
    prediction::AbstractVector,
    target::AbstractVector;
    schema::TensorSchema,
    length_penalty::Real=1.0,
    allow_empty::Bool=false,
)
    coefficient = Float64(length_penalty)
    isfinite(coefficient) && coefficient >= 0 || throw(ArgumentError(
        "length_penalty must be finite and non-negative",
    ))
    m, n = length(prediction), length(target)
    allow_empty || (m > 0 && n > 0) || throw(ArgumentError(
        "empty series is disabled for this task",
    ))
    squared_error = 0.0
    elements = 0
    for index in 1:min(m, n)
        predicted = prediction[index]
        expected = target[index]
        _valid_tensor(predicted, schema) || throw(DimensionMismatch(
            "prediction tensor $index does not match $(schema.shape)",
        ))
        _valid_tensor(expected, schema) || throw(DimensionMismatch(
            "target tensor $index does not match $(schema.shape)",
        ))
        for (left, right) in zip(predicted, expected)
            delta = Float64(left) - Float64(right)
            squared_error += delta * delta
            elements += 1
        end
    end
    aligned = elements == 0 ? 0.0 : squared_error / elements
    cardinality = abs(m - n) / max(m, n, 1)
    return aligned + coefficient * cardinality
end

function series_components(
    task::SeriesTask;
    global_consts::Dict{Symbol,Any},
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    kwargs...,
)
    types = global_consts_ty === nothing ? get_type_ctx(global_consts) : global_consts_ty
    base = default_components(;
        global_consts,
        global_consts_ty=types,
        input_slots=Symbol[task.input_slot],
        input_types=Dict{Symbol,TyLike}(task.input_slot => TyList(task.schema.ty)),
        kwargs...,
    )
    task.family === :series_to_series || return base
    loss = (prediction, target) -> series_mse(
        prediction,
        target;
        schema=task.schema,
        length_penalty=task.length_penalty,
        allow_empty=task.allow_empty,
    )
    loss_failure = issue -> issue isa DimensionMismatch || issue isa ArgumentError
    return merge(base, (; loss, loss_failure))
end

function series_regressor(
    task::SeriesTask;
    global_consts::Dict{Symbol,Any},
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    config::NamedTuple=default_config(),
    rng::AbstractRNG=MersenneTwister(2000),
    kwargs...,
)
    components = series_components(
        task;
        global_consts,
        global_consts_ty,
        kwargs...,
    )
    return Regressor(task.target_ty, components, config, rng)
end

@testitem "Regression tensor-series schema and length-aware loss" begin
    using Test
    using LambdaRegression.Regression:
        TensorSchema, TensorSeries, series_task, series_mse
    using LambdaRegression.TypeSystem: FloatTy, TyArray, TyNat

    schema = TensorSchema((2, 2))
    rectangular = TensorSchema((2, 3))
    labelled = TensorSchema((2, 2); name=:Matrix2x2)
    @test schema.ty == TyArray(TyNat(2), TyArray(TyNat(2), FloatTy))
    @test rectangular.ty == TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    @test labelled.ty == schema.ty
    @test labelled.name == :Matrix2x2
    a = [1.0 2.0; 3.0 4.0]
    b = [2.0 3.0; 4.0 5.0]
    x = TensorSeries([a, b])
    target = TensorSeries([a, b])
    shorter = TensorSeries([a])
    @test size(x) == (2,)
    @test series_mse(x, target; schema) == 0.0
    @test series_mse(shorter, target; schema, length_penalty=2.0) == 1.0
    task = series_task([x], [target]; schema, output=:series)
    @test task.family == :series_to_series
    @test task.data.X[1] === x
    scalar = series_task([x], [30.0]; schema, output=:real)
    @test scalar.family == :series_to_real
    @test_throws ArgumentError series_task([[a, b]], [target]; schema, output=:series)
    @test_throws DimensionMismatch series_mse(
        TensorSeries([[1.0, 2.0]]),
        target;
        schema,
    )
end

@testitem "Regression tensor-series API is exported at the package root" begin
    using Test
    using LambdaRegression

    @test LambdaRegression.TensorSchema === LambdaRegression.Regression.TensorSchema
    @test LambdaRegression.TensorSeries === LambdaRegression.Regression.TensorSeries
    @test LambdaRegression.SeriesTask === LambdaRegression.Regression.SeriesTask
    @test LambdaRegression.series_task === LambdaRegression.Regression.series_task
    @test LambdaRegression.series_mse === LambdaRegression.Regression.series_mse
    @test LambdaRegression.series_components ===
        LambdaRegression.Regression.series_components
    @test LambdaRegression.series_regressor ===
        LambdaRegression.Regression.series_regressor
end
