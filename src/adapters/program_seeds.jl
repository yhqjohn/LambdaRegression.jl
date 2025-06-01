
export best_program
using ..Programs: ConstSymbolFactory, new_constant!
using ..ASTs.NodeASTs: children, setchild!, copy_occurrences
import ..TypeSystem

function _seed_rename_constants!(node, names)
    node isa ConstNode && return ConstNode(get(names, node.name, node.name))
    for (edge, child) in enumerate(children(node))
        setchild!(node, edge, _seed_rename_constants!(child, names))
    end
    return node
end

# Rename local materials that collide with the receiving world.
function _import_program_seed(program::Program, components)
    reserved = union(Set(keys(components.global_consts)), Set(components.input_slots))
    used = union(reserved, Set(keys(program.local_consts)))
    for node in program.tree
        node isa ConstNode && push!(used, node.name)
    end
    renames = Dict{Symbol,Symbol}()
    values = Dict{Symbol,Any}()
    types = Dict{Symbol,TyLike}()
    for name in sort!(collect(keys(program.local_consts)); by=String)
        value, ty = program.local_consts[name], program.local_consts_ty[name]
        if name in reserved
            # Share only identical function globals; numerical parameters stay local.
            global_ty = get(components.global_consts_ty, name, nothing)
            if value isa Function && get(components.global_consts, name, nothing) === value &&
                    global_ty !== nothing && TypeSystem.islesseq(ty, global_ty) &&
                    TypeSystem.islesseq(global_ty, ty)
                continue
            end
            ordinal = 1
            fresh = Symbol("seed_", name, "_", ordinal)
            while fresh in used
                ordinal += 1
                fresh = Symbol("seed_", name, "_", ordinal)
            end
            push!(used, fresh)
            renames[name] = fresh
        end
        destination = get(renames, name, name)
        values[destination] = value
        types[destination] = ty
    end
    tree = copy_occurrences(program.tree)
    tree.root = _seed_rename_constants!(tree.root, renames)
    return Program(tree; local_consts=values, local_consts_ty=types)
end

@testitem "Composed collection winners seed fit without world collisions" begin
    using LambdaRegression.FrontendAdapter: fit, predict, best_program
    using LambdaRegression.Transforms: chain_programs
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, AppNode, ConstNode, IndexNode
    using LambdaRegression.Utils: Err

    function mapped(body)
        Program(AstTree(AbsNode(:xs,
            AppNode(AppNode(ConstNode(:AMAP), AbsNode(:x, body)), IndexNode(0)))))
    end
    square = mapped(AppNode(ConstNode(:SQR), IndexNode(0)))
    plusone = mapped(AppNode(AppNode(ConstNode(:FADD), IndexNode(0)), ConstNode(:ONE)))
    X = [-1.0 0.2; 0.5 0.7; 1.0 -0.3]
    kwargs = (; task_kind=:array, population_size=2, generations=0,
        operator_set=[:+, :square], parsimony_coefficient=0.0, random_state=31)
    first = fit(X, X.^2; initial_programs=[square], kwargs...)
    second = fit(X.^2, X.^2 .+ 1; initial_programs=[plusone], kwargs...)
    p = chain_programs(best_program(first), best_program(second))
    @test !(p isa Err)
    @test p isa Program
    fitted = fit(X, X.^2 .+ 1; initial_programs=[p], kwargs...)
    @test fitted["objective_calls"] == 2
    @test predict(fitted["model"], X) ≈ X.^2 .+ 1
    @test predict(first["model"], X) == X.^2
    @test predict(second["model"], X.^2) == X.^2 .+ 1
    @test_throws ArgumentError fit(X, X.^2; initial_programs=[p, p, p], kwargs...)
    @test_throws ArgumentError fit(X, X.^2; initial_programs=[p], max_complexity=2, kwargs...)
end

function _capture_program_globals(program::Program, globals, global_types)
    values = copy(program.local_consts)
    types = copy(program.local_consts_ty)
    for node in program.tree
        node isa ConstNode || continue
        name = node.name
        haskey(values, name) && continue
        haskey(globals, name) && haskey(global_types, name) || throw(ArgumentError(
            "cannot extract program: missing value or type for $name"))
        values[name] = globals[name]
        types[name] = global_types[name]
    end
    return Program(program.tree; local_consts=values, local_consts_ty=types)
end

# Copy syntax and array constants; host functions remain borrowed.
# Scalar extraction requires one feature and one output.
best_program(model::CollectionModel) = _capture_program_globals(
    model.program, model.components.global_consts, model.components.global_consts_ty)

function best_program(model::AdapterModel)
    model.feature_count == 1 && length(model.expressions) == 1 || throw(ArgumentError(
        "best_program supports scalar models with one feature and one output; use collection models for typed array mappings"))
    values = Dict{Symbol,Any}()
    types = Dict{Symbol,TyLike}()
    factory = ConstSymbolFactory(; prefix=:seed_constant_)
    function materialize(value)
        node = new_constant!(values, value, factory)
        types[node.name] = FloatTy
        return node
    end
    body = ScalarSearch.scalar_to_ast(only(model.expressions), [IndexNode(0)];
        materialize_constant=materialize)
    body isa Err && throw(ArgumentError("cannot materialize scalar winner: $(body.value)"))
    program = Program(AstTree(AbsNode(:x, body)); local_consts=values, local_consts_ty=types)
    globals = _collection_operator_globals(_COLLECTION_OPERATORS)
    return _capture_program_globals(program, globals, get_type_ctx(globals))
end

function best_program(result::AbstractDict)
    haskey(result, "model") || throw(ArgumentError("fit result must contain a model"))
    return best_program(result["model"])
end

@testitem "Program winner extraction owns materials and preserves source globals" begin
    using LambdaRegression.FrontendAdapter: AdapterModel, CollectionModel, best_program, collection_world
    using LambdaRegression.ScalarSearch: ScalarExpr, scalar_input, scalar_constant, scalar_binary
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs: AstTree, AbsNode, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, TyFun, TyArray, TyNat
    using LambdaRegression.Primitives: FADD

    expr = scalar_binary(:+, scalar_input(1), scalar_constant(2.0))
    model = AdapterModel(ScalarExpr[expr], 1, [0.0], false)
    p = best_program(model)
    @test p.local_consts[:FADD] === FADD
    @test count(v -> v == 2.0, values(p.local_consts)) == 1
    @test best_program(Dict("model" => model)).tree == p.tree
    @test_throws ArgumentError best_program(AdapterModel(ScalarExpr[expr], 2, [0.0], false))
    @test_throws ArgumentError best_program(Dict())

    ty = TyFun(TyArray(TyNat(2), FloatTy), TyArray(TyNat(2), FloatTy))
    original = [2.0, 3.0]
    world = collection_world(ty; global_consts=Dict(:saved => original),
        global_consts_ty=Dict(:saved => ty.res))
    source = Program(AstTree(AbsNode(:x, ConstNode(:saved))))
    fitted = CollectionModel(source, world; input_shape=(2,), output_shape=(2,))
    saved = best_program(fitted)
    @test saved.local_consts[:saved] == original
    @test saved.local_consts[:saved] !== original
    saved.local_consts[:saved][1] = 7.0
    @test original == [2.0, 3.0]
end
