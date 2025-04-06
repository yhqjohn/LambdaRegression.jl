"""
    numeric_parameter_schema(value, declared_type)

Read-only admission shared by constant search and structural observation.
Recognizes floating scalars and ordinary dense arrays or rectangular nested
vectors of floating arrays, with an exactly matching logical array type.
Returns `(kind, shape, element_type, parameter_count, edit_modes)`, or `nothing` for an
opaque/mismatched material. Payload values are not inspected or executed.

Shape, storage representation and aliasing are not numerical payload. Consumers
must preserve or reclassify these facts after edits. Supported dense empty arrays
have a valid schema but no adjustable parameters; empty outer nested vectors
remain opaque. Internally aliased nested tensors have a
layout schema but no constant-search mode: the dense adapter cannot preserve
independent-coordinate mutation semantics on them. Custom array wrappers remain opaque;
support needs a matching owned-update adapter, not just a classification rule.
"""
function numeric_parameter_schema(value, declared_type)
    ty = TypeSystem.prune(declared_type)
    if value isa AbstractFloat && ty == TypeSystem.FloatTy
        return (; kind=:scalar, shape=(), element_type=typeof(value), parameter_count=1,
            edit_modes=(:constant_values,))
    end
    dims = Int[]
    current = ty
    while current isa TypeSystem.TyArray
        current.len isa TypeSystem.TyNat || return nothing
        push!(dims, current.len.value)
        current = current.elem
    end
    isempty(dims) && return nothing
    current == TypeSystem.FloatTy || return nothing
    layout = _numeric_tensor_layout(value, IdDict{Any,Any}(), IdDict{Any,Nothing}())
    layout === nothing && return nothing
    layout.shape == Tuple(dims) || return nothing
    count = prod(layout.shape)
    modes = count > 0 && !layout.internal_alias ? (:constant_values,) : ()
    return (; kind=:tensor, shape=layout.shape, element_type=layout.element_type,
        parameter_count=count, edit_modes=modes)
end

function _numeric_tensor_layout(value, memo, active)
    value isa Array || return nothing
    haskey(active, value) && return nothing
    haskey(memo, value) && return merge(memo[value], (; internal_alias=true))
    if value isa Array{<:AbstractFloat}
        result = (; shape=size(value), element_type=eltype(value), internal_alias=false)
        memo[value] = result
        return result
    end
    value isa Vector && !isempty(value) || return nothing
    active[value] = nothing
    child = _numeric_tensor_layout(first(value), memo, active)
    if child === nothing
        delete!(active, value)
        return nothing
    end
    internal_alias = child.internal_alias
    for item in Iterators.drop(value, 1)
        other = _numeric_tensor_layout(item, memo, active)
        if other === nothing || other.shape != child.shape || other.element_type != child.element_type
            delete!(active, value)
            return nothing
        end
        internal_alias |= other.internal_alias
    end
    delete!(active, value)
    result = (; shape=(length(value), child.shape...), element_type=child.element_type,
        internal_alias)
    memo[value] = result
    return result
end

"""
    numeric_parameter_blocks(program)

Sorted `symbol => :scalar/:tensor` entries with at least one adjustable value.
Blocks are local symbols, not array-object alias groups. Updating one symbol
may split an alias; callers publish a child and reclassify it. No objective,
randomness, budget, search state or global constants belong to this catalogue.
"""
function numeric_parameter_blocks(program::Program)
    blocks = Pair{Symbol,Symbol}[]
    for (name, value) in program.local_consts
        schema = numeric_parameter_schema(value, program.local_consts_ty[name])
        schema === nothing && continue
        :constant_values in schema.edit_modes && push!(blocks, name => schema.kind)
    end
    sort!(blocks; by=block -> String(first(block)))
    return blocks
end

@testitem "Numeric parameter schemas respect logical layers and opaque boundaries" begin
    using LambdaRegression.Programs: numeric_parameter_schema, numeric_parameter_blocks, Program
    using LambdaRegression.ASTs.NodeASTs: AstTree, ConstNode
    using LambdaRegression.TypeSystem: FloatTy, IntTy, TyArray, TyNat, TyLike
    matrix_ty = TyArray(TyNat(2), TyArray(TyNat(3), FloatTy))
    @test numeric_parameter_schema(zeros(2, 3), matrix_ty).shape == (2, 3)
    @test numeric_parameter_schema([zeros(3), zeros(3)], matrix_ty).shape == (2, 3)
    @test numeric_parameter_schema(zeros(3, 2), matrix_ty) === nothing
    @test numeric_parameter_schema([zeros(2), zeros(3)], matrix_ty) === nothing
    @test numeric_parameter_schema(1, IntTy) === nothing
    @test numeric_parameter_schema(identity, FloatTy) === nothing
    @test numeric_parameter_schema(view(zeros(2, 3), :, :), matrix_ty) === nothing
    cycle = Any[]; push!(cycle, cycle)
    @test numeric_parameter_schema(cycle, matrix_ty) === nothing
    empty_ty = TyArray(TyNat(2), TyArray(TyNat(0), FloatTy))
    @test numeric_parameter_schema(zeros(2, 0), empty_ty).parameter_count == 0
    empty = Program(AstTree(ConstNode(:a)); local_consts=Dict(:a => zeros(2, 0)),
        local_consts_ty=Dict{Symbol,TyLike}(:a => empty_ty))
    @test isempty(numeric_parameter_blocks(empty))
    row = zeros(3)
    shared = Program(AstTree(ConstNode(:a)); local_consts=Dict(:a => [row, row]),
        local_consts_ty=Dict{Symbol,TyLike}(:a => matrix_ty))
    @test numeric_parameter_schema(shared.local_consts[:a], matrix_ty).shape == (2, 3)
    @test isempty(numeric_parameter_blocks(shared))
    @test numeric_parameter_schema([zeros(3), zeros(3)], matrix_ty).edit_modes == (:constant_values,)
end
