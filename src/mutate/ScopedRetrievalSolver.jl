# Private sorted, mu-free scoped first-order solver for typed crossover
# retrieval. Every query owns its arena, scope forest, and rollback trail.
# Ordinary inference types are constructor prototypes only; flexible and rigid
# query variables have explicit sorts and quantifier scopes here.
using TestItems

using ...TypeSystem:
    TyArray, TyMu, TyNat, TyRigid, TyVar,
    type_children, type_constructor_tag, type_static_parameters

@enum _QuerySort::UInt8 begin
    _TypeSort
    _LengthSort
end

@enum _QueryKind::UInt8 begin
    _FlexKind
    _RigidKind
    _ConstructorKind
end

@enum _ScopedSolveFailureCode::UInt8 begin
    _SortMismatchFailure
    _ConstructorMismatchFailure
    _RigidMismatchFailure
    _OccursCheckFailure
    _ScopeEscapeFailure
    _NoScopedSolutionFailure
end

const _QRef = Int32
const _ScopeId = Int32

struct _QueryScope
    parent::_ScopeId
    depth::Int32
end

struct _RigidLabel
    block::Symbol
    occurrence::Int32
    ordinal::Int32
end

Base.:(==)(left::_RigidLabel, right::_RigidLabel) =
    left.block === right.block &&
    left.occurrence == right.occurrence &&
    left.ordinal == right.ordinal

_RigidLabel(block::Symbol, occurrence::Integer, ordinal::Integer) =
    _RigidLabel(block, Int32(occurrence), Int32(ordinal))

mutable struct _QueryNode
    kind::_QueryKind
    sort::_QuerySort
    scope::_ScopeId
    parent::_QRef
    rank::UInt8
    binding::_QRef
    tag::Symbol
    static_parameters::Tuple
    children::Vector{_QRef}
    prototype::Union{Nothing,Ty}
    rigid_label::Union{Nothing,_RigidLabel}
end

struct _FlexSnapshot
    node::_QRef
    parent::_QRef
    rank::UInt8
    binding::_QRef
    scope::_ScopeId
end

struct _BranchMark
    trail_length::Int
    arena_length::Int
    scope_length::Int
end

mutable struct _ScopedSolver
    scopes::Vector{_QueryScope}
    nodes::Vector{_QueryNode}
    trail::Vector{_FlexSnapshot}
end

_ScopedSolver() = _ScopedSolver(_QueryScope[], _QueryNode[], _FlexSnapshot[])

struct _QueryEquation
    left::_QRef
    right::_QRef
end

struct _ScopedSolveFailure
    code::_ScopedSolveFailureCode
end

Base.:(==)(left::_ScopedSolveFailure, right::_ScopedSolveFailure) =
    left.code == right.code

abstract type _ResidualTerm end

struct _ResidualRigid <: _ResidualTerm
    label::_RigidLabel
    sort::_QuerySort
end

struct _ResidualMeta <: _ResidualTerm
    canonical_id::Int32
    scope::_ScopeId
    sort::_QuerySort
end

struct _ResidualConstructor <: _ResidualTerm
    tag::Symbol
    static_parameters::Tuple
    prototype::Ty
    children::Vector{_ResidualTerm}
end

struct _ScopedResidual
    q_images::Vector{_ResidualTerm}
    a_images::Vector{_ResidualTerm}
end

struct _ScopedBranchSolution
    branch::Int
    residual::_ScopedResidual
end

Base.:(==)(left::_ResidualRigid, right::_ResidualRigid) =
    left.label == right.label && left.sort == right.sort
Base.:(==)(left::_ResidualMeta, right::_ResidualMeta) =
    left.canonical_id == right.canonical_id &&
    left.scope == right.scope &&
    left.sort == right.sort
Base.:(==)(left::_ResidualConstructor, right::_ResidualConstructor) =
    left.tag == right.tag &&
    left.static_parameters == right.static_parameters &&
    left.children == right.children
Base.:(==)(left::_ScopedResidual, right::_ScopedResidual) =
    left.q_images == right.q_images && left.a_images == right.a_images
Base.:(==)(left::_ScopedBranchSolution, right::_ScopedSolveFailure) = false
Base.:(==)(left::_ScopedSolveFailure, right::_ScopedBranchSolution) = false

function _new_scope!(solver::_ScopedSolver, parent::Integer=0)::_ScopeId
    if parent == 0
        depth = Int32(0)
    else
        1 <= parent <= length(solver.scopes) ||
            throw(BoundsError(solver.scopes, parent))
        depth = solver.scopes[parent].depth + Int32(1)
    end
    push!(solver.scopes, _QueryScope(Int32(parent), depth))
    return Int32(length(solver.scopes))
end

function _new_flex!(
    solver::_ScopedSolver,
    sort::_QuerySort,
    scope::_ScopeId,
)::_QRef
    1 <= scope <= length(solver.scopes) ||
        throw(BoundsError(solver.scopes, scope))
    reference = Int32(length(solver.nodes) + 1)
    push!(solver.nodes, _QueryNode(
        _FlexKind,
        sort,
        scope,
        reference,
        UInt8(0),
        Int32(0),
        :flex,
        (),
        _QRef[],
        nothing,
        nothing,
    ))
    return reference
end

function _new_rigid!(
    solver::_ScopedSolver,
    sort::_QuerySort,
    scope::_ScopeId,
    label::_RigidLabel,
)::_QRef
    1 <= scope <= length(solver.scopes) ||
        throw(BoundsError(solver.scopes, scope))
    reference = Int32(length(solver.nodes) + 1)
    push!(solver.nodes, _QueryNode(
        _RigidKind,
        sort,
        scope,
        Int32(0),
        UInt8(0),
        Int32(0),
        :rigid,
        (),
        _QRef[],
        nothing,
        label,
    ))
    return reference
end

function _query_constructor_sort(prototype::Ty)::_QuerySort
    prototype isa TyMu &&
        throw(ArgumentError("scoped retrieval is strictly mu-free"))
    prototype isa TyVar &&
        throw(ArgumentError("query variables require an explicit scoped role"))
    prototype isa TyRigid &&
        throw(ArgumentError("query rigids require an explicit scoped role"))
    return prototype isa TyNat ? _LengthSort : _TypeSort
end

function _query_child_sorts(prototype::Ty)
    children = type_children(prototype)
    prototype isa TyArray && return (_LengthSort, _TypeSort)
    return ntuple(_ -> _TypeSort, length(children))
end

function _new_constructor!(
    solver::_ScopedSolver,
    prototype::Ty,
    children::Vector{_QRef}=_QRef[],
)::_QRef
    sort = _query_constructor_sort(prototype)
    expected_sorts = _query_child_sorts(prototype)
    length(children) == length(expected_sorts) || throw(ArgumentError(
        "query constructor child count does not match its TypeSystem prototype",
    ))
    for (child, expected) in zip(children, expected_sorts)
        1 <= child <= length(solver.nodes) ||
            throw(BoundsError(solver.nodes, child))
        solver.nodes[child].sort == expected || throw(ArgumentError(
            "query constructor child has the wrong sort",
        ))
    end
    reference = Int32(length(solver.nodes) + 1)
    push!(solver.nodes, _QueryNode(
        _ConstructorKind,
        sort,
        Int32(0),
        Int32(0),
        UInt8(0),
        Int32(0),
        type_constructor_tag(prototype),
        type_static_parameters(prototype),
        copy(children),
        prototype,
        nothing,
    ))
    return reference
end

_mark(solver::_ScopedSolver) =
    _BranchMark(length(solver.trail), length(solver.nodes), length(solver.scopes))

function _snapshot_flex!(solver::_ScopedSolver, reference::_QRef)
    node = solver.nodes[reference]
    node.kind == _FlexKind || throw(ArgumentError("only flex nodes are mutable"))
    push!(solver.trail, _FlexSnapshot(
        reference,
        node.parent,
        node.rank,
        node.binding,
        node.scope,
    ))
    return nothing
end

function _rollback!(solver::_ScopedSolver, mark::_BranchMark)
    for index in length(solver.trail):-1:(mark.trail_length + 1)
        snapshot = solver.trail[index]
        node = solver.nodes[snapshot.node]
        node.parent = snapshot.parent
        node.rank = snapshot.rank
        node.binding = snapshot.binding
        node.scope = snapshot.scope
    end
    resize!(solver.trail, mark.trail_length)
    resize!(solver.nodes, mark.arena_length)
    resize!(solver.scopes, mark.scope_length)
    return nothing
end

function _flex_root(solver::_ScopedSolver, reference::_QRef)::_QRef
    node = solver.nodes[reference]
    node.kind == _FlexKind || return reference
    while node.parent != reference
        reference = node.parent
        node = solver.nodes[reference]
    end
    return reference
end

function _dereference(solver::_ScopedSolver, reference::_QRef)::_QRef
    while true
        node = solver.nodes[reference]
        node.kind == _FlexKind || return reference
        reference = _flex_root(solver, reference)
        node = solver.nodes[reference]
        node.binding == 0 && return reference
        reference = node.binding
    end
end

function _scope_nca(
    solver::_ScopedSolver,
    left::_ScopeId,
    right::_ScopeId,
)::_ScopeId
    root(scope) = begin
        while solver.scopes[scope].parent != 0
            scope = solver.scopes[scope].parent
        end
        scope
    end
    root(left) == root(right) || throw(ArgumentError(
        "scoped retrieval cannot merge variables from distinct query roots",
    ))
    while solver.scopes[left].depth > solver.scopes[right].depth
        left = solver.scopes[left].parent
    end
    while solver.scopes[right].depth > solver.scopes[left].depth
        right = solver.scopes[right].parent
    end
    while left != right
        left = solver.scopes[left].parent
        right = solver.scopes[right].parent
    end
    return left
end

function _scope_is_ancestor(
    solver::_ScopedSolver,
    ancestor::_ScopeId,
    descendant::_ScopeId,
)::Bool
    while descendant != 0
        descendant == ancestor && return true
        descendant = solver.scopes[descendant].parent
    end
    return false
end

function _merge_flex!(
    solver::_ScopedSolver,
    left::_QRef,
    right::_QRef,
)
    left = _flex_root(solver, left)
    right = _flex_root(solver, right)
    left == right && return nothing
    left_node = solver.nodes[left]
    right_node = solver.nodes[right]
    left_node.sort == right_node.sort ||
        return _ScopedSolveFailure(_SortMismatchFailure)
    merged_scope = _scope_nca(solver, left_node.scope, right_node.scope)

    if left_node.rank < right_node.rank
        left, right = right, left
        left_node, right_node = right_node, left_node
    end
    _snapshot_flex!(solver, right)
    right_node.parent = left
    _snapshot_flex!(solver, left)
    left_node.scope = merged_scope
    left_node.rank == right_node.rank && (left_node.rank += UInt8(1))
    return nothing
end

function _occurs_in(
    solver::_ScopedSolver,
    needle::_QRef,
    term::_QRef,
)::Bool
    stack = _QRef[term]
    seen = Set{_QRef}()
    while !isempty(stack)
        current = _dereference(solver, pop!(stack))
        current == needle && return true
        current in seen && continue
        push!(seen, current)
        node = solver.nodes[current]
        node.kind == _ConstructorKind && append!(stack, node.children)
    end
    return false
end

function _lower_term_to_scope!(
    solver::_ScopedSolver,
    term::_QRef,
    target_scope::_ScopeId,
)
    stack = _QRef[term]
    seen = Set{_QRef}()
    while !isempty(stack)
        current = _dereference(solver, pop!(stack))
        current in seen && continue
        push!(seen, current)
        node = solver.nodes[current]
        if node.kind == _RigidKind
            _scope_is_ancestor(solver, node.scope, target_scope) ||
                return _ScopedSolveFailure(_ScopeEscapeFailure)
        elseif node.kind == _FlexKind
            lowered_scope = _scope_nca(solver, node.scope, target_scope)
            if lowered_scope != node.scope
                _snapshot_flex!(solver, current)
                node.scope = lowered_scope
            end
        else
            append!(stack, node.children)
        end
    end
    return nothing
end

function _bind_flex!(
    solver::_ScopedSolver,
    flex::_QRef,
    term::_QRef,
)
    flex = _flex_root(solver, flex)
    term = _dereference(solver, term)
    flex == term && return nothing
    flex_node = solver.nodes[flex]
    term_node = solver.nodes[term]
    flex_node.sort == term_node.sort ||
        return _ScopedSolveFailure(_SortMismatchFailure)
    _occurs_in(solver, flex, term) &&
        return _ScopedSolveFailure(_OccursCheckFailure)
    lowered = _lower_term_to_scope!(solver, term, flex_node.scope)
    lowered isa _ScopedSolveFailure && return lowered
    _snapshot_flex!(solver, flex)
    flex_node.binding = term
    return nothing
end

function _unify_scoped!(
    solver::_ScopedSolver,
    equations::Vector{_QueryEquation},
)
    work = copy(equations)
    while !isempty(work)
        equation = pop!(work)
        left = _dereference(solver, equation.left)
        right = _dereference(solver, equation.right)
        left == right && continue
        left_node = solver.nodes[left]
        right_node = solver.nodes[right]
        left_node.sort == right_node.sort ||
            return _ScopedSolveFailure(_SortMismatchFailure)

        if left_node.kind == _FlexKind && right_node.kind == _FlexKind
            result = _merge_flex!(solver, left, right)
            result isa _ScopedSolveFailure && return result
        elseif left_node.kind == _FlexKind
            result = _bind_flex!(solver, left, right)
            result isa _ScopedSolveFailure && return result
        elseif right_node.kind == _FlexKind
            result = _bind_flex!(solver, right, left)
            result isa _ScopedSolveFailure && return result
        elseif left_node.kind == _RigidKind && right_node.kind == _RigidKind
            left_node.rigid_label == right_node.rigid_label ||
                return _ScopedSolveFailure(_RigidMismatchFailure)
        elseif left_node.kind == _ConstructorKind &&
               right_node.kind == _ConstructorKind
            left_node.tag == right_node.tag &&
                left_node.static_parameters == right_node.static_parameters &&
                length(left_node.children) == length(right_node.children) ||
                return _ScopedSolveFailure(_ConstructorMismatchFailure)
            for index in eachindex(left_node.children)
                push!(work, _QueryEquation(
                    left_node.children[index],
                    right_node.children[index],
                ))
            end
        else
            return _ScopedSolveFailure(_ConstructorMismatchFailure)
        end
    end
    return nothing
end

function _project_residual_term!(
    solver::_ScopedSolver,
    reference::_QRef,
    meta_ids::Dict{_QRef,Int32},
    next_meta::Base.RefValue{Int32},
)
    reference = _dereference(solver, reference)
    node = solver.nodes[reference]
    if node.kind == _FlexKind
        canonical_id = get!(meta_ids, reference) do
            next_meta[] += Int32(1)
            next_meta[]
        end
        return _ResidualMeta(canonical_id, node.scope, node.sort)
    elseif node.kind == _RigidKind
        label = node.rigid_label::_RigidLabel
        label.block === :P &&
            return _ScopedSolveFailure(_ScopeEscapeFailure)
        return _ResidualRigid(label, node.sort)
    end

    projected_children = _ResidualTerm[]
    for child in node.children
        projected = _project_residual_term!(solver, child, meta_ids, next_meta)
        projected isa _ScopedSolveFailure && return projected
        push!(projected_children, projected)
    end
    return _ResidualConstructor(
        node.tag,
        node.static_parameters,
        node.prototype::Ty,
        projected_children,
    )
end

function _project_scoped_residual(
    solver::_ScopedSolver,
    qvars::Vector{_QRef},
    avars::Vector{_QRef},
)
    meta_ids = Dict{_QRef,Int32}()
    next_meta = Ref(Int32(0))
    q_images = _ResidualTerm[]
    a_images = _ResidualTerm[]
    for (source, destination) in ((qvars, q_images), (avars, a_images))
        for variable in source
            projected = _project_residual_term!(
                solver,
                variable,
                meta_ids,
                next_meta,
            )
            projected isa _ScopedSolveFailure && return projected
            push!(destination, projected)
        end
    end
    return _ScopedResidual(q_images, a_images)
end

function _solve_scoped(
    solver::_ScopedSolver,
    equations::Vector{_QueryEquation};
    qvars::Vector{_QRef}=_QRef[],
    avars::Vector{_QRef}=_QRef[],
)
    branch_mark = _mark(solver)
    result = try
        solved = _unify_scoped!(solver, equations)
        solved isa _ScopedSolveFailure ? solved :
            _project_scoped_residual(solver, qvars, avars)
    finally
        _rollback!(solver, branch_mark)
    end
    return result
end

function _first_scoped_solution!(
    solver::_ScopedSolver,
    branches;
    qvars::Vector{_QRef}=_QRef[],
    avars::Vector{_QRef}=_QRef[],
)
    last_failure = _ScopedSolveFailure(_NoScopedSolutionFailure)
    for (branch_index, build_branch) in enumerate(branches)
        branch_mark = _mark(solver)
        result = try
            equations = build_branch(solver)
            solved = _unify_scoped!(solver, equations)
            if solved isa _ScopedSolveFailure
                solved
            else
                _project_scoped_residual(solver, qvars, avars)
            end
        finally
            _rollback!(solver, branch_mark)
        end
        if result isa _ScopedResidual
            return _ScopedBranchSolution(branch_index, result)
        end
        last_failure = result
    end
    return last_failure
end

_sort_key(sort::_QuerySort) = sort == _TypeSort ? :type : :length

_residual_key(term::_ResidualRigid) = (
    :rigid,
    term.label.block,
    Int(term.label.occurrence),
    Int(term.label.ordinal),
    _sort_key(term.sort),
)
_residual_key(term::_ResidualMeta) = (
    :meta,
    Int(term.canonical_id),
    Int(term.scope),
    _sort_key(term.sort),
)
_residual_key(term::_ResidualConstructor) = (
    :constructor,
    term.tag,
    term.static_parameters,
    Tuple(_residual_key(child) for child in term.children),
)
_residual_key(residual::_ScopedResidual) = (
    :residual,
    Tuple(_residual_key(term) for term in residual.q_images),
    Tuple(_residual_key(term) for term in residual.a_images),
)

@testitem "Scoped retrieval solver finite sorted solve" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _LengthSort, _OccursCheckFailure, _QRef, _QueryEquation,
        _ScopedSolveFailure, _ScopedSolver, _SortMismatchFailure, _TypeSort,
        _new_constructor!, _new_flex!, _new_scope!, _residual_key,
        _solve_scoped
    using LambdaRegression.TypeSystem:
        IntTy, TyFun, TyList, TyMu, make_tyvar_factory

    solver = _ScopedSolver()
    scope = _new_scope!(solver)
    a = _new_flex!(solver, _TypeSort, scope)
    int = _new_constructor!(solver, IntTy)
    list_a = _new_constructor!(solver, TyList(IntTy), _QRef[a])
    fun_a_a = _new_constructor!(solver, TyFun(IntTy, IntTy), _QRef[a, a])
    fun_int_int = _new_constructor!(solver, TyFun(IntTy, IntTy), _QRef[int, int])

    residual = _solve_scoped(
        solver,
        [_QueryEquation(fun_a_a, fun_int_int)];
        avars=_QRef[a],
    )
    @test _residual_key(residual) == (
        :residual,
        (),
        ((:constructor, :Atom, (:Int,), ()),),
    )

    occurs = _solve_scoped(
        solver,
        [_QueryEquation(a, list_a)];
        avars=_QRef[a],
    )
    @test occurs == _ScopedSolveFailure(_OccursCheckFailure)

    length_var = _new_flex!(solver, _LengthSort, scope)
    sort_failure = _solve_scoped(
        solver,
        [_QueryEquation(length_var, int)];
        avars=_QRef[length_var],
    )
    @test sort_failure == _ScopedSolveFailure(_SortMismatchFailure)

    factory = make_tyvar_factory()
    binder = factory(0)
    recursive = TyMu(binder, TyList(binder))
    @test_throws ArgumentError _new_constructor!(solver, recursive)
end

@testitem "Scoped retrieval solver escape rollback" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _QRef, _QueryEquation, _RigidLabel, _ScopeEscapeFailure,
        _ScopedBranchSolution, _ScopedSolveFailure, _ScopedSolver, _TypeSort,
        _first_scoped_solution!, _new_constructor!, _new_flex!, _new_rigid!,
        _new_scope!, _residual_key, _solve_scoped
    using LambdaRegression.TypeSystem: IntTy, TyList

    solver = _ScopedSolver()
    r_scope = _new_scope!(solver)
    b_scope = _new_scope!(solver, r_scope)
    e1_scope = _new_scope!(solver, b_scope)
    p1_scope = _new_scope!(solver, e1_scope)
    e2_scope = _new_scope!(solver, b_scope)
    p2_scope = _new_scope!(solver, e2_scope)
    q = _new_flex!(solver, _TypeSort, r_scope)

    result = _first_scoped_solution!(
        solver,
        [
            s -> begin
                p = _new_rigid!(
                    s,
                    _TypeSort,
                    p1_scope,
                    _RigidLabel(:P, 1, 1),
                )
                u = _new_flex!(s, _TypeSort, p1_scope)
                list_p = _new_constructor!(s, TyList(IntTy), _QRef[p])
                [_QueryEquation(u, list_p), _QueryEquation(q, u)]
            end,
            s -> begin
                u = _new_flex!(s, _TypeSort, p2_scope)
                int = _new_constructor!(s, IntTy)
                [_QueryEquation(u, int), _QueryEquation(q, u)]
            end,
        ];
        qvars=_QRef[q],
    )
    @test result isa _ScopedBranchSolution
    @test result.branch == 2
    @test _residual_key(result.residual) == (
        :residual,
        ((:constructor, :Atom, (:Int,), ()),),
        (),
    )

    b = _new_rigid!(
        solver,
        _TypeSort,
        b_scope,
        _RigidLabel(:B, 0, 1),
    )
    list_b = _new_constructor!(solver, TyList(IntTy), _QRef[b])
    direct = _solve_scoped(
        solver,
        [_QueryEquation(q, list_b)];
        qvars=_QRef[q],
    )
    @test direct == _ScopedSolveFailure(_ScopeEscapeFailure)

    a = _new_flex!(solver, _TypeSort, b_scope)
    allowed = _solve_scoped(
        solver,
        [_QueryEquation(a, list_b)];
        avars=_QRef[a],
    )
    @test _residual_key(allowed) == (
        :residual,
        (),
        ((
            :constructor,
            :List,
            (),
            ((:rigid, :B, 0, 1, :type),),
        ),),
    )

    reverse_order = _first_scoped_solution!(
        solver,
        [
            s -> begin
                p = _new_rigid!(
                    s,
                    _TypeSort,
                    p1_scope,
                    _RigidLabel(:P, 1, 1),
                )
                u = _new_flex!(s, _TypeSort, p1_scope)
                list_p = _new_constructor!(s, TyList(IntTy), _QRef[p])
                [_QueryEquation(q, u), _QueryEquation(u, list_p)]
            end,
        ];
        qvars=_QRef[q],
    )
    @test reverse_order == _ScopedSolveFailure(_ScopeEscapeFailure)
end

@testitem "Scoped retrieval solver sibling canonical residual" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _QRef, _QueryEquation, _ScopedSolver, _TypeSort, _new_constructor!,
        _new_flex!, _new_scope!, _residual_key, _solve_scoped
    using LambdaRegression.TypeSystem: IntTy, TyList

    solver = _ScopedSolver()
    r_scope = _new_scope!(solver)
    b_scope = _new_scope!(solver, r_scope)
    e1_scope = _new_scope!(solver, b_scope)
    p1_scope = _new_scope!(solver, e1_scope)
    e2_scope = _new_scope!(solver, b_scope)
    p2_scope = _new_scope!(solver, e2_scope)
    a = _new_flex!(solver, _TypeSort, b_scope)
    u1 = _new_flex!(solver, _TypeSort, p1_scope)
    u2 = _new_flex!(solver, _TypeSort, p2_scope)
    list_u1 = _new_constructor!(solver, TyList(IntTy), _QRef[u1])
    list_u2 = _new_constructor!(solver, TyList(IntTy), _QRef[u2])

    forward = _solve_scoped(
        solver,
        [_QueryEquation(a, list_u1), _QueryEquation(a, list_u2)];
        avars=_QRef[a],
    )
    reverse = _solve_scoped(
        solver,
        [_QueryEquation(a, list_u2), _QueryEquation(a, list_u1)];
        avars=_QRef[a],
    )
    expected = (
        :residual,
        (),
        ((
            :constructor,
            :List,
            (),
            ((:meta, 1, Int(b_scope), :type),),
        ),),
    )
    @test _residual_key(forward) == expected
    @test _residual_key(reverse) == expected
    @test forward == reverse
end

@testitem "Scoped retrieval solver rejects cross-root merges and rolls back scopes" begin
    using Test
    using LambdaRegression.Mutations.Populations:
        _QueryEquation, _ScopedSolver, _TypeSort, _mark, _merge_flex!,
        _new_flex!, _new_scope!, _rollback!

    solver = _ScopedSolver()
    left_scope = _new_scope!(solver)
    right_scope = _new_scope!(solver)
    left = _new_flex!(solver, _TypeSort, left_scope)
    right = _new_flex!(solver, _TypeSort, right_scope)
    @test_throws ArgumentError _merge_flex!(solver, left, right)

    mark = _mark(solver)
    _new_scope!(solver, left_scope)
    @test length(solver.scopes) == 3
    _rollback!(solver, mark)
    @test length(solver.scopes) == 2
end
