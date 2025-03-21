"""
    Programs

Small value-layer records for a typed expression and its local constant
materials. `Program` owns its imported syntax, value dictionary, and declaration
dictionary; it deliberately carries no target, fitness, metadata, or world
state. Type graph objects in `local_consts_ty` are shared read-only by contract,
while the dictionary itself is owned by each imported program.

Use `Program(tree; local_consts, local_consts_ty)` at an external import
boundary. Use `with_constants(program, updates)` for a child that reuses the
parent's read-only tree and type graph while copying only the updated value
group. The default copier deep-copies ordinary data as one group and borrows
top-level `Function` values; callers own the stability/purity contract for
borrowed resources and may inject an explicit group copier for other values.
"""
module Programs

using TestItems

using ..ASTs.NodeASTs: AstTree, ConstNode, copy_occurrences
using ..TypeSystem: TyLike

export Program, with_constants, ConstSymbolFactory, new_constant!

"""
    ConstSymbolFactory(; prefix=:c)

Per-session deterministic names for generated local constants.  The factory is
deliberately ordinary state owned by its caller; it is not a package-global
counter and therefore can be used independently by concurrent search paths.
"""
mutable struct ConstSymbolFactory
    counter::Int
    prefix::Symbol
end

ConstSymbolFactory(; prefix::Symbol=:c) = ConstSymbolFactory(0, prefix)

function _next_const_symbol!(factory::ConstSymbolFactory)::Symbol
    factory.counter += 1
    return Symbol(String(factory.prefix), factory.counter)
end

"""Insert a value under a fresh generated symbol and return its `ConstNode`."""
function new_constant!(context, value)
    name = gensym(:c)
    while haskey(context, name)
        name = gensym(:c)
    end
    context[name] = value
    return ConstNode(name)
end

"""Insert a value using a caller-owned deterministic factory, avoiding collisions."""
function new_constant!(context, value, factory::ConstSymbolFactory)
    while true
        name = _next_const_symbol!(factory)
        haskey(context, name) && continue
        context[name] = value
        return ConstNode(name)
    end
end

struct _OwnedProgramTag end
const _OWNED_PROGRAM = _OwnedProgramTag()

"""An owned expression and its local constant values and declared types."""
struct Program
    tree::AstTree
    local_consts::Dict{Symbol,Any}
    local_consts_ty::Dict{Symbol,TyLike}

    # The only constructor is the private publication path below. External
    # imports must pass through the keyword constructor, which owns materials
    # and validates their declaration keys.
    function Program(
        tree::AstTree,
        values::Dict{Symbol,Any},
        types::Dict{Symbol,TyLike},
        ::_OwnedProgramTag,
    )
        return new(tree, values, types)
    end
end

function _validate_keys(
    values::AbstractDict{Symbol},
    types::AbstractDict{Symbol},
)
    Set(keys(values)) == Set(keys(types)) ||
        throw(ArgumentError("local constant values and types must have identical keys"))
    return nothing
end

function _copy_local_values(values)
    source = Dict{Symbol,Any}(values)
    borrowed_functions = Dict{Symbol,Any}()
    owned_data = Dict{Symbol,Any}()
    for (name, value) in source
        if value isa Function
            borrowed_functions[name] = value
        else
            owned_data[name] = value
        end
    end
    # Deep-copy ordinary data as one group so aliases within the owned data
    # graph survive. Top-level functions are stable borrowed host resources;
    # aliases crossing a borrowed function's captured data boundary are not
    # promised by this default and require an explicit group copier.
    copied = deepcopy(owned_data)
    merge!(copied, borrowed_functions)
    return copied
end

function _owned_types(types)
    # Type graph objects are read-only by contract and intentionally not copied.
    return Dict{Symbol,TyLike}(types)
end

"""Publish new edit materials while sharing already-published read-only layers."""
function _publish_program(
    tree::AstTree,
    values::Dict{Symbol,Any},
    types::Dict{Symbol,TyLike},
)
    _validate_keys(values, types)
    return Program(tree, values, types, _OWNED_PROGRAM)
end

"""
    Program(tree; local_consts, local_consts_ty)

Import a tree and local materials into an owned `Program`. Only value/type key
agreement is checked here; type inference, closedness, and callable type
compatibility remain caller responsibilities.
"""
function Program(
    tree::AstTree;
    local_consts=Dict{Symbol,Any}(),
    local_consts_ty=Dict{Symbol,TyLike}(),
    copy_values=_copy_local_values,
)
    values = Dict{Symbol,Any}(local_consts)
    types = Dict{Symbol,TyLike}(local_consts_ty)
    _validate_keys(values, types)
    owned_values = Dict{Symbol,Any}(copy_values(values))
    _validate_keys(values, owned_values)
    return _publish_program(copy_occurrences(tree), owned_values, _owned_types(types))
end

"""
    with_constants(program, updates)

Replace already-declared local values in a child program. The parent remains
unchanged; the child reuses the parent's read-only syntax and type graph and
receives a new mapping whose updated values are copied as one group while
unchanged values are shared read-only. The caller guarantees each update has
the type declared by `program.local_consts_ty`; unknown keys are rejected.
"""
function with_constants(program::Program, updates; copy_values=_copy_local_values)
    update_values = Dict{Symbol,Any}()
    for (key, value) in pairs(updates)
        key isa Symbol || throw(ArgumentError("local constant names must be Symbols"))
        haskey(program.local_consts_ty, key) ||
            throw(ArgumentError("unknown local constant $(key)"))
        update_values[key] = value
    end
    owned_updates = Dict{Symbol,Any}(copy_values(update_values))
    _validate_keys(update_values, owned_updates)
    merged = Dict{Symbol,Any}(program.local_consts)
    merge!(merged, owned_updates)
    return _publish_program(program.tree, merged, program.local_consts_ty)
end

@testitem "Program owns syntax and local value materials" begin
    using Test
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    source = AstTree(AbsNode(:x, IndexNode(0)))
    values = Dict{Symbol,Any}(:bias => Any[1, 2])
    types = Dict{Symbol,TyLike}(:bias => IntTy)
    copied = copy_occurrences(source)
    copied.root.body = ConstNode(:changed)
    program = Program(source; local_consts=values, local_consts_ty=types)

    source.root.body = ConstNode(:external_change)
    values[:bias][1] = 99
    @test source.root.body isa ConstNode
    @test copied.root.body isa ConstNode
    @test program.tree.root.body isa IndexNode
    @test program.local_consts[:bias] == [1, 2]
    @test program.local_consts !== values
    @test program.local_consts_ty !== types
end

@testitem "Program occurrence copies preserve aliases but reject cycles" begin
    using Test
    using LambdaRegression.Programs: Program
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike

    shared = AbsNode(:x, IndexNode(0))
    source = AstTree(AppNode(shared, shared))
    program = Program(source)
    left, right = program.tree.root.func, program.tree.root.arg
    @test left !== right
    copied = copy_occurrences(source)
    copied.root.func.body = ConstNode(:left)
    @test copied.root.arg.body isa IndexNode

    shared_value = Any[1]
    values = Dict{Symbol,Any}(:a => shared_value, :b => shared_value)
    types = Dict{Symbol,TyLike}(:a => IntTy, :b => IntTy)
    grouped = Program(AstTree(ConstNode(:unit)); local_consts=values, local_consts_ty=types)
    @test grouped.local_consts[:a] === grouped.local_consts[:b]
    @test grouped.local_consts[:a] !== shared_value

    captured = Any[3]
    imported_function = x -> Base.abs(x) + captured[1] - 3
    function_program = Program(
        AstTree(ConstNode(:f));
        local_consts=Dict{Symbol,Any}(:f => imported_function),
        local_consts_ty=Dict{Symbol,TyLike}(:f => TyFun(IntTy, IntTy)),
    )
    @test function_program.local_consts[:f] === imported_function
    @test function_program.local_consts[:f](-3) == 3

    borrowed = Any[1]
    explicit_borrow = values -> Dict{Symbol,Any}(values)
    custom_program = Program(
        AstTree(ConstNode(:opaque));
        local_consts=Dict{Symbol,Any}(:opaque => borrowed),
        local_consts_ty=Dict{Symbol,TyLike}(:opaque => IntTy),
        copy_values=explicit_borrow,
    )
    @test custom_program.local_consts[:opaque] === borrowed

    cyclic = AppNode(ConstNode(:f), ConstNode(:x))
    cyclic.func = cyclic
    @test_throws ArgumentError copy_occurrences(cyclic)
    @test_throws ArgumentError Program(AstTree(cyclic))
end

@testitem "Program constant replacement reuses read-only layers" begin
    using Test
    using LambdaRegression.Programs: Program, with_constants
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.TypeSystem: IntTy, TyLike, newtyvar

    tyvar = newtyvar(0)
    types = Dict{Symbol,TyLike}(:a => tyvar, :b => IntTy)
    parent = Program(
        AstTree(ConstNode(:unit));
        local_consts=Dict{Symbol,Any}(:a => Any[1], :b => Any[3]),
        local_consts_ty=types,
    )
    child = with_constants(parent, Dict{Symbol,Any}(:a => Any[2]))
    @test child.tree === parent.tree
    @test child.local_consts_ty === parent.local_consts_ty
    @test child.local_consts !== parent.local_consts
    @test parent.local_consts[:a] == [1]
    @test child.local_consts[:a] == [2]
    @test child.local_consts[:b] === parent.local_consts[:b]
    @test child.local_consts_ty[:a] === tyvar
    @test_throws ArgumentError with_constants(parent, Dict{Symbol,Any}(:missing => 3))

    @test fieldnames(Program) == (:tree, :local_consts, :local_consts_ty)
    @test_throws ArgumentError Program(
        AstTree(ConstNode(:unit));
        local_consts=Dict{Symbol,Any}(:a => 1),
        local_consts_ty=Dict{Symbol,TyLike}(),
    )
end

end # module Programs
