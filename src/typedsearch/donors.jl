"""Closed ordinary-program donors for the typed-search language.

`SourceProgram` is a derived leaf, not a host-function oracle.  It retains the
owned source AST used for export, a compact term certified in the same
`Language`, the exact root type id, global names, provenance, and the written
size after recursively expanding admitted donor names.  The first slice only
admits terms closed in De Bruijn variables; globals remain explicit leaves and
therefore retain their declared runtime meanings.
"""

using ..ASTs.NodeASTs:
    AstNode, AstTree, AbsNode, AppNode, ConstNode, FunctionDefNode, IndexNode, VarNode,
    children
import ..TypeSystem

export SourceProgram, register_program!, admit_program_donor!, written_complexity,
    expand_program_donors

"""A certified closed ordinary source program stored as a typed-search leaf."""
struct SourceProgram
    name::Symbol
    source::AstTree
    term::TypedTerm
    typeid::Int32
    root_type::TypeSystem.Ty
    constants::Vector{Symbol}
    provenance::Any
    expanded_size::Int
end

"""Return the program leaf for `name`, if one has been admitted."""
function _program_for_name(language::Language, name::Symbol)
    for leaf in language.leaves
        leaf.payload isa SourceProgram || continue
        leaf.payload.name == name && return leaf.payload
    end
    return nothing
end

function _program_constants(term::TypedTerm, language::Language)
    names = Symbol[]
    seen = Set{Symbol}()
    for node in term.nodes
        node.kind == LEAF || continue
        payload = language.leaves[Int(node.x)].payload
        name = payload isa SourceConstant ? payload.name :
            payload isa SourceProgram ? payload.name : nothing
        name isa Symbol || continue
        name in seen || (push!(seen, name); push!(names, name))
    end
    return names
end

@inline function _donor_fail(reason::Symbol, detail=nothing)
    return Err(SourceBoundaryError(reason, 0))
end

"""Count a source tree after recursively expanding registered donor names."""
function _expanded_source_size(node::AstNode, language::Language,
                               stack::Set{Symbol}, max_size::Int, max_depth::Int)
    size = 0
    function visit(n, active, depth)
        depth <= max_depth || return Err(SourceBoundaryError(:program_depth, 0))
        size += 1
        size <= max_size || return Err(SourceBoundaryError(
            :program_size, 0,
        ))
        if n isa ConstNode
            nested = _program_for_name(language, n.name)
            nested === nothing && return true
            n.name in active && return Err(SourceBoundaryError(:program_cycle, 0))
            next_active = copy(active)
            push!(next_active, n.name)
            # The donor occurrence replaces this constant node, so remove the
            # one node just counted before adding the nested source tree.
            size -= 1
            return visit(nested.source.root, next_active, depth + 1)
        end
        for child in children(n)
            result = visit(child, active, depth + 1)
            result isa Err && return result
        end
        return true
    end
    result = visit(node, stack, 0)
    result isa Err && return result
    return size
end

"""Materialise a donor source copy, recursively expanding nested donors."""
function _expand_source_node(n::AstNode, language::Language, max_nodes::Int,
                             max_depth::Int, counter::Base.RefValue{Int},
                             stack::Set{Symbol}, depth::Int)
    # `depth` is the ordinary source-tree depth.  Donor references do not
    # consume that structural depth, so bound their active chain separately;
    # otherwise a long DAG of one-node donors could recurse forever before a
    # materialisation-size cap is reached.
    length(stack) - 1 <= max_depth ||
        return Err(SourceBoundaryError(:program_depth, 0))
    depth <= max_depth || return Err(SourceBoundaryError(:program_depth, 0))
    if n isa ConstNode
        nested = _program_for_name(language, n.name)
        if nested !== nothing
            n.name in stack && return Err(SourceBoundaryError(:program_cycle, 0))
            next_stack = copy(stack)
            push!(next_stack, n.name)
            return _expand_source_node(nested.source.root, language, max_nodes,
                max_depth, counter, next_stack, depth)
        end
    end
    counter[] += 1
    counter[] <= max_nodes || return Err(SourceBoundaryError(:program_size, 0))
    if n isa ConstNode || n isa IndexNode || n isa VarNode
        return n isa ConstNode ? ConstNode(n.name) : n
    end
    copied = AstNode[]
    for child in children(n)
        expanded = _expand_source_node(child, language, max_nodes, max_depth,
            counter, stack, depth + 1)
        expanded isa Err && return expanded
        push!(copied, expanded)
    end
    # Source nodes in this slice are all rebuilt by their ordinary constructors.
    if n isa AbsNode
        return AbsNode(n.name, copied[1])
    elseif n isa AppNode
        return AppNode(copied[1], copied[2])
    elseif n isa FunctionDefNode
        return FunctionDefNode(copied[1])
    end
    return Err(SourceBoundaryError(:unsupported_region, 0))
end

function _expand_source_program(program::SourceProgram, language::Language;
                                max_nodes::Int=100_000, max_depth::Int=512,
                                _counter::Base.RefValue{Int}=Ref(0),
                                _stack::Set{Symbol}=Set{Symbol}(),
                                _depth::Int=0)
    max_nodes > 0 && max_depth >= 0 ||
        throw(ArgumentError("invalid program materialisation bounds"))
    program.name in _stack && return Err(SourceBoundaryError(:program_cycle, 0))
    _depth <= max_depth || return Err(SourceBoundaryError(:program_depth, 0))
    stack = copy(_stack)
    push!(stack, program.name)
    return _expand_source_node(program.source.root, language, max_nodes, max_depth,
        _counter, stack, _depth)
end

"""Expand all admitted program donors in an ordinary source AST copy."""
function expand_program_donors(tree::AstTree, language::Language;
                               max_nodes::Int=100_000, max_depth::Int=512)
    max_nodes > 0 && max_depth >= 0 ||
        throw(ArgumentError("invalid program materialisation bounds"))
    return _expand_source_node(tree.root, language, max_nodes, max_depth,
        Ref(0), Set{Symbol}(), 0)
end

function _program_cycle(language::Language, root_name::Symbol,
                         dependencies::Vector{Symbol})
    active = Set{Symbol}([root_name])
    completed = Set{Symbol}()
    function visit(name, path)
        name in path && return true
        name in completed && return false
        nested = _program_for_name(language, name)
        if nested === nothing
            push!(completed, name)
            return false
        end
        next = copy(path)
        push!(next, name)
        any(visit(dep, next) for dep in nested.constants) && return true
        push!(completed, name)
        return false
    end
    any(visit(dep, active) for dep in dependencies)
end

"""Written source complexity of a compact term.

Ordinary compact nodes cost one.  A `SourceProgram` occurrence costs its
recursively expanded source size, so search caps and archives cannot hide
large donor material behind one leaf.  This is a structural measure only; it
does not prepare or evaluate the term.
"""
function written_complexity(term::TypedTerm, language::Language;
                            max_size::Int=typemax(Int))
    max_size > 0 || throw(ArgumentError("max_size must be positive"))
    size = 0
    for node in term.nodes
        increment = 1
        if node.kind == LEAF
            1 <= node.x <= length(language.leaves) ||
                return Err(TypedSearchError(:bad_leaf_index, node.x))
            payload = language.leaves[Int(node.x)].payload
            payload isa SourceProgram && (increment = payload.expanded_size)
        end
        size <= max_size - increment || return Err(SourceBoundaryError(:program_size, 0))
        size += increment
    end
    return size
end

"""Admit one ordinary source program as a same-language closed donor leaf.

The source is inferred and imported once.  Its compact term must have no free
term variables, and the root type is interned exactly in `language.types`.
When a same-name `SourceConstant` already exists at this type, its leaf slot is
replaced in place; this keeps caller-maintained global value vectors aligned.
"""
function register_program!(language::Language, name::Symbol, tree::AstTree,
                           type_context, target::TypeSystem.Ty;
                           provenance=nothing, max_nodes::Int=100_000,
                           max_depth::Int=512,
                           max_expanded_size::Int=1_000_000)
    max_nodes > 0 && max_depth >= 0 && max_expanded_size > 0 ||
        throw(ArgumentError("invalid program donor bounds"))
    source = prepare_source(tree, type_context, target;
        max_nodes=max_nodes, max_depth=max_depth)
    source isa Err && return source
    root_typeid = intern_type!(language.types, target)
    root_typeid isa Err && return root_typeid
    # A donor name is a single expansion namespace in this first slice.  Do
    # not let a same-name source constant/program at another committed type
    # make `_program_for_name` choose an ambiguous expansion.
    for leaf in language.leaves
        payload = leaf.payload
        if (payload isa SourceConstant || payload isa SourceProgram) &&
                payload.name == name && leaf.typeid != root_typeid
            return _donor_fail(:program_type_mismatch, name)
        end
    end
    hole = source_hole(source)
    hole isa Err && return hole
    initial_leaves = length(language.leaves)
    imported = import_hole(hole, language)
    if imported isa Err
        resize!(language.leaves, initial_leaves)
        return imported
    end
    checked = validate(imported, language)
    if checked !== true
        resize!(language.leaves, initial_leaves)
        return checked isa Err ? checked : _donor_fail(:invalid_donor)
    end
    if imported.nodes[end].typeid != root_typeid
        resize!(language.leaves, initial_leaves)
        return _donor_fail(:wrong_donor_type)
    end
    constants = _program_constants(imported, language)
    name in constants && begin
        resize!(language.leaves, initial_leaves)
        return _donor_fail(:program_cycle, name)
    end
    _program_cycle(language, name, constants) && begin
        resize!(language.leaves, initial_leaves)
        return _donor_fail(:program_cycle, name)
    end
    # Expanded sizes are certificates for the current language.  Do not
    # retroactively replace a primitive dependency of an admitted donor: that
    # would make its complexity and source expansion stale.
    for leaf in language.leaves
        existing = leaf.payload
        existing isa SourceProgram && name in existing.constants &&
            begin
                resize!(language.leaves, initial_leaves)
                return _donor_fail(:program_dependency_frozen, name)
            end
    end
    # Source size is bounded before changing any existing source-constant leaf.
    expanded_size = _expanded_source_size(source.index.tree.root, language,
        Set{Symbol}([name]), max_expanded_size, max_depth)
    expanded_size isa Err && begin
        resize!(language.leaves, initial_leaves)
        return expanded_size
    end
    program = SourceProgram(name, source.index.tree, imported, root_typeid,
        source.root_type, constants, provenance, Int(expanded_size))
    # Prefer replacing the existing same-name constant at the exact committed
    # type.  A same-name donor at another type is rejected above, so the leaf
    # namespace cannot acquire ambiguous expansion records.
    replacement = nothing
    for (i, leaf) in enumerate(language.leaves)
        if leaf.typeid == root_typeid && leaf.payload isa SourceConstant &&
                leaf.payload.name == name
            replacement = i
            break
        elseif leaf.typeid == root_typeid && leaf.payload isa SourceProgram &&
                leaf.payload.name == name
            replacement = i
            break
        end
    end
    if replacement === nothing
        push!(language.leaves, LeafSpec(root_typeid, program))
        return Int32(length(language.leaves))
    end
    language.leaves[replacement] = LeafSpec(root_typeid, program)
    return Int32(replacement)
end

admit_program_donor!(args...; kwargs...) = register_program!(args...; kwargs...)

@testitem "TypedSearch closed program donors preserve export and direct evaluation" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.Execution: evaluate_native
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, newtyvar
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceConstant
    using LambdaRegression.Utils: Err

    int_to_int = TyFun(IntTy, IntTy)
    ctx = Dict{Symbol,TyLike}(:inc => int_to_int, :one => IntTy)
    source_tree = AstTree(AppNode(ConstNode(:inc), ConstNode(:one)))
    language = Language(TypeTable())
    law_leaf = register_program!(language, :law, source_tree, ctx, IntTy;
        provenance=:fixture)
    @test law_leaf isa Integer
    @test language.leaves[law_leaf].payload isa SourceProgram

    target = prepare_source(AstTree(ConstNode(:law)),
        Dict{Symbol,TyLike}(:law => IntTy), IntTy)
    hole = source_hole(target)
    imported = import_hole(hole, language)
    @test imported isa TypedTerm
    @test written_complexity(imported, language) == 3
    exported = export_hole(imported, language, hole)
    @test exported isa AstTree
    bounded_export = export_hole(imported, language, hole; max_depth=0)
    @test bounded_export isa Err
    # Nested donor materialisation is guarded by the shared source-expansion
    # depth check; export may report that helper's reason directly.
    @test bounded_export.value.reason in (:export_depth, :program_depth)
    @test evaluate_native(exported, Dict{Symbol,Any}(
        :inc => x -> x + 1, :one => 2, :law => 0,
    )) == 3

    prepared = prepare_typed(imported, language)
    direct = run_typed(prepared, [x -> x + 1, 2, AstTree(ConstNode(:law))]; max_steps=32)
    @test direct == 3
    limited = run_typed(prepared, [x -> x + 1, 2, nothing]; max_steps=0)
    @test limited isa Err
    @test limited.value.reason == :step_limit
end

@testitem "TypedSearch donor expansion is bounded and capture-safe" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs
    using LambdaRegression.ASTs.NodeASTs: FunctionDefNode
    using LambdaRegression.Execution: evaluate_native
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, TyMu, newtyvar
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: SourceConstant
    using LambdaRegression.Utils: Err

    int_to_int = TyFun(IntTy, IntTy)
    language = Language(TypeTable())
    ctx = Dict{Symbol,TyLike}()
    base = register_program!(language, :base,
        AstTree(AbsNode(:z, IndexNode(0))), ctx, int_to_int)
    @test base isa Integer
    nested = register_program!(language, :outer, AstTree(ConstNode(:base)),
        Dict{Symbol,TyLike}(:base => int_to_int), int_to_int)
    @test nested isa Integer
    outer = language.leaves[nested].payload::SourceProgram
    @test outer.expanded_size == 2
    @test written_complexity(TypedTerm(TermNode[
        leaf_node(outer.typeid, nested),
    ]), language) == 2

    target_type = TyFun(IntTy, int_to_int)
    target = prepare_source(AstTree(AbsNode(:x, ConstNode(:outer))),
        Dict{Symbol,TyLike}(:outer => int_to_int), target_type)
    hole = source_hole(target, [1])
    term = import_hole(hole, language)
    rebuilt = export_hole(term, language, hole)
    @test rebuilt isa AstTree
    @test evaluate_native(rebuilt, Dict{Symbol,Any}(); args=(1, 7)) == 7

    outer_target = prepare_source(AstTree(ConstNode(:outer)),
        Dict{Symbol,TyLike}(:outer => int_to_int), int_to_int)
    outer_hole = source_hole(outer_target)
    outer_term = import_hole(outer_hole, language)
    outer_value = run_typed(prepare_typed(outer_term, language), Any[])
    @test outer_value isa TypedCallable
    @test outer_value(7) == 7
    @test export_hole(outer_term, language, outer_hole; max_depth=0) isa Err

    cycle = register_program!(language, :cycle, AstTree(ConstNode(:cycle)),
        Dict{Symbol,TyLike}(:cycle => IntTy), IntTy)
    @test cycle isa Err
    @test cycle.value.reason == :program_cycle
    capped = register_program!(language, :too_small, AstTree(ConstNode(:base)),
        Dict{Symbol,TyLike}(:base => int_to_int), int_to_int;
        max_expanded_size=1)
    @test capped isa Err
    @test capped.value.reason == :program_size

    mismatch_source = prepare_source(AstTree(ConstNode(:base)),
        Dict{Symbol,TyLike}(:base => IntTy), IntTy)
    mismatch = import_hole(source_hole(mismatch_source), language)
    @test mismatch isa Err
    @test mismatch.value.reason == :program_type_mismatch

    frozen = register_program!(language, :base,
        AstTree(AbsNode(:z, IndexNode(0))), Dict{Symbol,TyLike}(), int_to_int)
    @test frozen isa Err
    @test frozen.value.reason == :program_dependency_frozen

    other_language = Language(TypeTable())
    other_int = intern!(other_language.types, IntTy)
    add_leaf!(other_language, other_int, SourceConstant(:same))
    same_name_mismatch = register_program!(other_language, :same,
        AstTree(AbsNode(:z, IndexNode(0))), Dict{Symbol,TyLike}(), int_to_int)
    @test same_name_mismatch isa Err
    @test same_name_mismatch.value.reason == :program_type_mismatch

    recursive = register_program!(language, :recursive,
        AstTree(FunctionDefNode(AbsNode(:z, IndexNode(0)))),
        Dict{Symbol,TyLike}(), int_to_int)
    @test recursive isa Err

    α = newtyvar(0)
    μ = TyMu(α, TyFun(α, IntTy))
    recursive_mu = register_program!(language, :recursive_mu,
        AstTree(ConstNode(:recursive_mu)), Dict{Symbol,TyLike}(:recursive_mu => μ), μ)
    @test recursive_mu isa Err
end
