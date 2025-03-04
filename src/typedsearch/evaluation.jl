"""
    TypedSearch evaluation

Direct call-by-need execution for the compact, post-order `TypedTerm` form.
`prepare_typed` validates and owns a bounded copy of a term; `run_typed`
evaluates that copy against values indexed by the shared `Language` leaf table.
Lambda closures retain a persistent frame chain, and every run owns one shared
step budget, including higher-order calls made through returned callables.
Strict host applications evaluate their argument without an ephemeral thunk,
while retaining the same application/forcing ticks and lexical thunk sharing.

This evaluator deliberately supports only `VAR`, `LEAF`, `ABS`, and `APP`.
Source export, native AST evaluation, recursive `FunctionDef` nodes, and
μ-dependent transparent types are separate boundaries. A host function's own
body cannot be preempted by this budget unless it calls a `TypedCallable`
supplied by this evaluator.
"""

using ..Utils: Err
using ..ASTs.NodeASTs: AstNode, AstTree

export PreparedTyped, TypedCallable, TypedEvaluationError
export prepare_typed, run_typed

"""An expected runtime refusal at the compact typed evaluator boundary."""
struct TypedEvaluationError <: Exception
    reason::Symbol
    detail::Any
end

TypedEvaluationError(reason::Symbol) = TypedEvaluationError(reason, nothing)

@inline _typed_eval_fail(reason::Symbol, detail=nothing) =
    Err(TypedEvaluationError(reason, detail))

"""Owned validated compact syntax and the largest language-leaf index it uses."""
struct PreparedTyped
    nodes::Vector{TermNode}
    leaf_count::Int
    required_leaf_count::Int
    leaves::Vector{LeafSpec}
end

PreparedTyped(nodes::Vector{TermNode}, leaf_count::Int) =
    PreparedTyped(nodes, leaf_count, leaf_count, LeafSpec[])

abstract type _TypedValue end

struct _TypedHostValue{T} <: _TypedValue
    value::T
end

mutable struct _TypedThunk
    state::UInt8
    prepared::Union{Nothing,PreparedTyped}
    node::Int32
    env::Any
    value::Any
end

"""One persistent lexical frame; extending a closure is O(1)."""
struct _TypedFrame
    parent::Any
    thunk::_TypedThunk
end

struct _TypedClosure <: _TypedValue
    prepared::PreparedTyped
    body::Int32
    env::Any
end

# Specialise on the phase's runtime vocabulary, never on candidate syntax or
# the shape of a lexical frame chain.
mutable struct _TypedEvaluationState{L}
    prepared::PreparedTyped
    leaf_values::L
    remaining::Int
    initial::Int
    program_cache::IdDict{Any,PreparedTyped}
    active_programs::IdDict{Any,Nothing}
end

"""A host-callable view of an internal closure, retaining its run state.

Calling this value outside `run_typed` may throw `TypedEvaluationError` when
the shared run budget or another expected runtime boundary fails, matching the
native evaluator's callable behavior.
"""
struct TypedCallable
    state::_TypedEvaluationState
    closure::_TypedClosure
end

const _TYPED_UNFORCED = UInt8(0)
const _TYPED_FORCING = UInt8(1)
const _TYPED_FORCED = UInt8(2)

_host_thunk(value) = _TypedThunk(
    _TYPED_FORCED, nothing, Int32(0), nothing, _TypedHostValue(value),
)

_node_thunk(prepared::PreparedTyped, node::Integer, env) = _TypedThunk(
    _TYPED_UNFORCED, prepared, Int32(node), env, nothing,
)

function _typed_tick!(state::_TypedEvaluationState)
    state.remaining > 0 || return _typed_eval_fail(:step_limit, state.initial)
    state.remaining -= 1
    return true
end

function _typed_lookup(env, db::Int)
    db >= 0 || return _typed_eval_fail(:negative_de_bruijn_index, db)
    current = env
    for _ in 1:db
        current === nothing && return _typed_eval_fail(:unbound_variable, db)
        current = current.parent
    end
    current === nothing && return _typed_eval_fail(:unbound_variable, db)
    return current.thunk
end

function _typed_force!(state::_TypedEvaluationState, thunk::_TypedThunk)
    thunk.state == _TYPED_FORCED && return thunk.value
    thunk.state == _TYPED_FORCING && return _typed_eval_fail(:evaluation_cycle)
    tick = _typed_tick!(state)
    tick isa Err && return tick
    thunk.state = _TYPED_FORCING
    result = try
        _typed_eval_node!(state, thunk.prepared::PreparedTyped,
            thunk.node, thunk.env)
    catch
        thunk.state = _TYPED_UNFORCED
        rethrow()
    end
    if result isa Err
        thunk.state = _TYPED_UNFORCED
        return result
    end
    thunk.value = result
    thunk.prepared = nothing
    thunk.env = nothing
    thunk.state = _TYPED_FORCED
    return result
end

function _typed_to_host(value, state::_TypedEvaluationState)
    value isa Err && return value
    value isa _TypedHostValue && return value.value
    value isa _TypedClosure && return TypedCallable(state, value)
    throw(ArgumentError("unknown typed runtime value $(typeof(value))"))
end

function _typed_host_result(value, state::_TypedEvaluationState)
    if value isa TypedCallable
        return value.state === state ? value.closure : _TypedHostValue(value)
    end
    value isa AstNode && return _typed_eval_fail(:host_returned_ast, typeof(value))
    value isa AstTree && return _typed_eval_fail(:host_returned_ast, typeof(value))
    value isa _TypedValue && return value
    return _TypedHostValue(value)
end

function _typed_program_prepared!(state::_TypedEvaluationState,
                                  program::SourceProgram)
    existing = get(state.program_cache, program, nothing)
    existing !== nothing && return existing
    nodes = copy(program.term.nodes)
    leaf_count = 0
    for node in nodes
        node.kind == LEAF && (leaf_count = max(leaf_count, Int(node.x)))
    end
    prepared = PreparedTyped(nodes, leaf_count, leaf_count, state.prepared.leaves)
    state.program_cache[program] = prepared
    return prepared
end

function _typed_call_host!(state::_TypedEvaluationState,
                           function_value::_TypedHostValue, forced)
    forced isa Err && return forced
    host_argument = _typed_to_host(forced, state)
    host_function = function_value.value
    # Host execution is outside preemption unless it re-enters TypedCallable.
    # Do not test callability before evaluating the argument.
    result = host_function(host_argument)
    return _typed_host_result(result, state)
end

function _typed_apply!(state::_TypedEvaluationState, function_value, argument::_TypedThunk)
    tick = _typed_tick!(state)
    tick isa Err && return tick
    if function_value isa _TypedClosure
        frame = _TypedFrame(function_value.env, argument)
        return _typed_eval_node!(state, function_value.prepared,
            function_value.body, frame)
    elseif function_value isa _TypedHostValue
        forced = _typed_force!(state, argument)
        return _typed_call_host!(state, function_value, forced)
    end
    return _typed_eval_fail(:non_callable, typeof(function_value))
end

function (callable::TypedCallable)(argument)
    result = _typed_apply!(callable.state, callable.closure, _host_thunk(argument))
    if result isa Err
        result.value isa TypedEvaluationError && throw(result.value)
        return result
    end
    return _typed_to_host(result, callable.state)
end

function _typed_eval_node!(state::_TypedEvaluationState,
                           prepared::PreparedTyped, index::Integer, env)
    i = Int(index)
    1 <= i <= length(prepared.nodes) || return _typed_eval_fail(:bad_node_index, i)
    tick = _typed_tick!(state)
    tick isa Err && return tick
    node = prepared.nodes[i]
    if node.kind == VAR
        thunk = _typed_lookup(env, Int(node.x))
        thunk isa Err && return thunk
        return _typed_force!(state, thunk)
    elseif node.kind == LEAF
        leaf_index = Int(node.x)
        1 <= leaf_index <= length(state.leaf_values) ||
            begin
                # A program donor is source material, not a host value.  It
                # can run with no corresponding global tuple entry while its
                # own SourceConstant leaves still use the shared tuple.
                leaf_index <= length(prepared.leaves) ||
                    return _typed_eval_fail(:missing_leaf_value, leaf_index)
            end
        if leaf_index <= length(prepared.leaves) &&
                prepared.leaves[leaf_index].payload isa SourceProgram
            program = prepared.leaves[leaf_index].payload::SourceProgram
            haskey(state.active_programs, program) &&
                return _typed_eval_fail(:program_cycle, program.name)
            state.active_programs[program] = nothing
            nested = _typed_program_prepared!(state, program)
            result = try
                _typed_eval_node!(state, nested, length(nested.nodes), nothing)
            finally
                delete!(state.active_programs, program)
            end
            return result
        end
        leaf_index <= length(state.leaf_values) ||
            return _typed_eval_fail(:missing_leaf_value, leaf_index)
        return _typed_host_result(state.leaf_values[leaf_index], state)
    elseif node.kind == ABS
        return _TypedClosure(prepared, node.x, env)
    elseif node.kind == APP
        function_value = _typed_eval_node!(state, prepared, node.x, env)
        function_value isa Err && return function_value
        if function_value isa _TypedHostValue
            # This syntactic APP would create a fresh thunk and force it once
            # immediately. It cannot escape or enter a lexical frame. Preserve
            # both ticks; evaluating a VAR still forces its existing thunk.
            tick = _typed_tick!(state) # application
            tick isa Err && return tick
            tick = _typed_tick!(state) # first force of the omitted wrapper
            tick isa Err && return tick
            forced = _typed_eval_node!(state, prepared, node.y, env)
            return _typed_call_host!(state, function_value, forced)
        end
        argument = _node_thunk(prepared, node.y, env)
        return _typed_apply!(state, function_value, argument)
    end
    return _typed_eval_fail(:unsupported_node_kind, node.kind)
end

function _typed_shape(nodes::Vector{TermNode}, max_depth::Int)
    isempty(nodes) && return _typed_eval_fail(:empty_term)
    depths = zeros(Int, length(nodes))
    leaf_count = 0
    for (i, node) in enumerate(nodes)
        if node.kind == VAR || node.kind == LEAF
            node.kind == LEAF && (leaf_count = max(leaf_count, Int(node.x)))
            depths[i] = 0
        elseif node.kind == ABS
            body = Int(node.x)
            body == i - 1 && body >= 1 ||
                return _typed_eval_fail(:non_postorder_abs_child, (i, body))
            depths[i] = depths[body] + 1
        elseif node.kind == APP
            fn, argument = Int(node.x), Int(node.y)
            argument == i - 1 && fn >= 1 && fn < argument ||
                return _typed_eval_fail(:non_postorder_app_children, (i, fn, argument))
            depths[i] = max(depths[fn], depths[argument]) + 1
        else
            return _typed_eval_fail(:unsupported_node_kind, node.kind)
        end
        depths[i] <= max_depth || return _typed_eval_fail(:term_depth, (i, depths[i]))
    end
    return (leaf_count=leaf_count, depth=depths[end])
end

function _typed_required_program_count(program::SourceProgram,
                                       leaves::Vector{LeafSpec},
                                       active::IdDict{Any,Nothing},
                                       memo::IdDict{Any,Int})
    cached = get(memo, program, nothing)
    cached === nothing || return cached
    # `active` is the recursion-stack guard; it must not double as the memo
    # because a DAG dependency may be reached through multiple paths.
    haskey(active, program) && return 0
    active[program] = nothing
    required = _typed_required_leaf_count(program.term.nodes, leaves, active, memo)
    delete!(active, program)
    memo[program] = required
    return required
end

function _typed_required_leaf_count(nodes::Vector{TermNode}, leaves::Vector{LeafSpec},
                                    active::IdDict{Any,Nothing}=IdDict{Any,Nothing}(),
                                    memo::IdDict{Any,Int}=IdDict{Any,Int}())
    required = 0
    for node in nodes
        node.kind == LEAF || continue
        i = Int(node.x)
        1 <= i <= length(leaves) || continue
        payload = leaves[i].payload
        if payload isa SourceProgram
            required = max(required, _typed_required_program_count(
                payload, leaves, active, memo,
            ))
        else
            required = max(required, i)
        end
    end
    return required
end

"""
    prepare_typed(term, language; max_nodes=100_000, max_depth=512)

Validate and own a bounded compact term before repeated direct evaluation.
The result is an `Err` for expected syntax/boundary refusals; invalid bounds
are caller errors. `max_nodes` and `max_depth` bound this compact term only;
admitted `SourceProgram` donors carry their own bounded expansion certificate,
and export applies a separate materialisation bound. The language itself
remains the fixed leaf-index vocabulary.
"""
function prepare_typed(term::TypedTerm, language::Language;
                       max_nodes::Int=100_000, max_depth::Int=512)
    max_nodes > 0 && max_depth >= 0 ||
        throw(ArgumentError("max_nodes must be positive and max_depth non-negative"))
    length(term.nodes) <= max_nodes || return _typed_eval_fail(
        :term_size, (length(term.nodes), max_nodes),
    )
    shape = _typed_shape(term.nodes, max_depth)
    shape isa Err && return shape
    checked = validate(term, language)
    if checked !== true
        checked isa Err && return _typed_eval_fail(:invalid_term, checked.value)
        return _typed_eval_fail(:invalid_term, checked)
    end
    leaves = LeafSpec[language.leaves...]
    required = _typed_required_leaf_count(term.nodes, leaves)
    return PreparedTyped(copy(term.nodes), shape.leaf_count, required, leaves)
end

"""
    run_typed(prepared, leaf_values; args=(), max_steps=4096)

Evaluate a prepared compact term directly. `leaf_values[i]` supplies the value
for the `i`th `Language` leaf used by the term. A vector binding container is
shallow-copied at run start (a tuple is retained); host values are never deep
copied. Thus a returned closure keeps the run's leaf bindings, not a mutable
caller-owned vector. `args` is applied from left to right, while a zero-argument
call returns a root value or `TypedCallable` for a closure. Every thunk, closure
callback, and application consumes the same run budget. Host function bodies
themselves are not preemptible.
Expected evaluator failures from a callback are converted to `Err`; unrelated
host exceptions propagate to the caller.
"""
function run_typed(prepared::PreparedTyped, leaf_values;
                   args=(), max_steps::Int=4096)
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    bindings = if leaf_values isa Tuple
        leaf_values
    elseif leaf_values isa AbstractVector
        copy(leaf_values)
    else
        throw(ArgumentError("leaf_values must be a tuple or abstract vector"))
    end
    length(bindings) >= prepared.required_leaf_count || return _typed_eval_fail(
        :missing_leaf_values, (prepared.required_leaf_count, length(bindings)),
    )
    try
        state = _TypedEvaluationState(prepared, bindings, max_steps, max_steps,
            IdDict{Any,PreparedTyped}(), IdDict{Any,Nothing}())
        value = _typed_eval_node!(state, prepared, length(prepared.nodes), nothing)
        value isa Err && return value
        arguments = args isa Tuple ? args : (args,)
        for argument in arguments
            value = _typed_apply!(state, value, _host_thunk(argument))
            value isa Err && return value
        end
        return _typed_to_host(value, state)
    catch error
        error isa TypedEvaluationError && return Err(error)
        rethrow()
    end
end

@testitem "TypedSearch strict host applications preserve budgets and effects" begin
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.Utils: Err

    table = TypeTable()
    int = intern!(table, IntTy)
    fn = arrow!(table, int, int)
    curried = arrow!(table, int, fn)
    language = Language(table, [LeafSpec(fn, :f), LeafSpec(int, :a)])
    term = TypedTerm(TermNode[
        leaf_node(fn, 1), leaf_node(int, 2), app_node(int, 1, 2),
    ])
    prepared = prepare_typed(term, language)
    calls = Ref(0)
    host = x -> (calls[] += 1; x + 1)
    for budget in 0:6
        calls[] = 0
        result = run_typed(prepared, (host, 4); max_steps=budget)
        if budget < 5
            @test result isa Err
            @test result.value.reason == :step_limit
            @test calls[] == 0
        else
            @test result == 5
            @test calls[] == 1
        end
    end

    events = Symbol[]
    add = x -> (push!(events, :first); y -> (push!(events, :second); x + y))
    nested_language = Language(table, [LeafSpec(curried, :add), LeafSpec(int, :a)])
    nested = prepare_typed(TypedTerm(TermNode[
        leaf_node(curried, 1), leaf_node(int, 2), app_node(fn, 1, 2),
        leaf_node(int, 2), app_node(int, 3, 4),
    ]), nested_language)
    @test run_typed(nested, (add, 3); max_steps=8) isa Err
    @test events == [:first]
    empty!(events)
    @test run_typed(nested, (add, 3); max_steps=9) == 6
    @test events == [:first, :second]

    # Same-state callbacks returned by a host retain their lexical closure;
    # callbacks from another run remain tied to that earlier run.
    hof = arrow!(table, fn, fn)
    callback_language = Language(table, [LeafSpec(hof, :host), LeafSpec(fn, :inc)])
    callback_term = prepare_typed(TypedTerm(TermNode[
        leaf_node(hof, 1), leaf_node(fn, 2), var_node(int, 0),
        app_node(int, 2, 3), abs_node(fn, 4), app_node(fn, 1, 5),
    ]), callback_language)
    observed = Int[]
    return_callback = callback -> (push!(observed, callback(1), callback(2)); callback)
    callback = run_typed(callback_term, (return_callback, x -> x + 1); max_steps=64)
    @test observed == [2, 3]
    @test callback isa TypedCallable
    @test callback(3) == 4
    foreign = run_typed(callback_term, (_ -> callback, x -> x + 100); max_steps=5)
    @test foreign === callback
    @test foreign(4) == 5
end

@testitem "TypedSearch direct evaluator roots and native oracle" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
    using LambdaRegression.Procedural: evaluate_procedural
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    int = intern!(table, IntTy)
    fint = arrow!(table, int, int)
    language = Language(table)
    identity = TypedTerm(TermNode[
        var_node(int, 0),
        abs_node(fint, 1),
    ])
    prepared = prepare_typed(identity, language)
    @test prepared isa PreparedTyped
    @test run_typed(prepared, Any[]; args=(42,)) == 42
    @test evaluate_procedural(
        AstTree(AbsNode(:x, IndexNode(0))), Dict{Symbol,Any}(); args=(42,)
    ) == run_typed(prepared, Any[]; args=(42,))

    leaf_language = Language(table, [LeafSpec(int, :one)])
    root_leaf = TypedTerm(TermNode[leaf_node(int, 1)])
    root_prepared = prepare_typed(root_leaf, leaf_language; max_depth=0)
    @test root_prepared isa PreparedTyped
    @test run_typed(root_prepared, [17]) == 17
end

@testitem "TypedSearch direct evaluator currying and lexical capture" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Procedural: evaluate_procedural
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    int = intern!(table, IntTy)
    fint = arrow!(table, int, int)
    curried_ty = arrow!(table, int, fint)
    add_ty = arrow!(table, int, fint)
    language = Language(table, [LeafSpec(add_ty, :add)])
    curried = TypedTerm(TermNode[
        leaf_node(add_ty, 1),
        var_node(int, 1),
        app_node(fint, 1, 2),
        var_node(int, 0),
        app_node(int, 3, 4),
        abs_node(fint, 5),
        abs_node(curried_ty, 6),
    ])
    values = [x -> y -> x + y]
    prepared = prepare_typed(curried, language)
    @test run_typed(prepared, values; args=(2, 3)) == 5
    native = AstTree(AbsNode(:x, AbsNode(:y,
        AppNode(AppNode(ConstNode(:add), IndexNode(1)), IndexNode(0)))))
    @test run_typed(prepared, values; args=(2, 3)) ==
        evaluate_procedural(native, Dict{Symbol,Any}(:add => values[1]); args=(2, 3))

    capture = TypedTerm(TermNode[
        var_node(int, 1),
        abs_node(fint, 1),
        abs_node(curried_ty, 2),
    ])
    @test run_typed(prepare_typed(capture, Language(table)), Any[];
        args=(7, 9)) == 7
end

@testitem "TypedSearch direct evaluator lazy and higher-order boundaries" begin
    using Test
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Procedural: evaluate_procedural
    using LambdaRegression.TypeSystem: IntTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.Utils: Err

    table = TypeTable()
    int = intern!(table, IntTy)
    fint = arrow!(table, int, int)
    map_ty = arrow!(table, fint, fint)

    map_language = Language(table, [
        LeafSpec(map_ty, :map1), LeafSpec(fint, :inc), LeafSpec(int, :three),
    ])
    map_term = TypedTerm(TermNode[
        leaf_node(map_ty, 1),
        leaf_node(fint, 2),
        var_node(int, 0),
        app_node(int, 2, 3),
        abs_node(fint, 4),
        app_node(fint, 1, 5),
        leaf_node(int, 3),
        app_node(int, 6, 7),
    ])
    map1 = callback -> value -> sum(map(callback, (value, value + 1)))
    map_values = [map1, x -> x + 1, 3]
    @test run_typed(prepare_typed(map_term, map_language), map_values) == 9
    native_map = AstTree(AppNode(AppNode(ConstNode(:map1),
        AbsNode(:y, AppNode(ConstNode(:inc), IndexNode(0)))), ConstNode(:three)))
    @test run_typed(prepare_typed(map_term, map_language), map_values) ==
        evaluate_procedural(native_map, Dict{Symbol,Any}(
            :map1 => map1, :inc => map_values[2], :three => 3,
        ))

    lazy_language = Language(table, [
        LeafSpec(int, :one), LeafSpec(fint, :throwing), LeafSpec(int, :zero),
    ])
    lazy_term = TypedTerm(TermNode[
        leaf_node(int, 1),
        abs_node(fint, 1),
        leaf_node(fint, 2),
        leaf_node(int, 3),
        app_node(int, 3, 4),
        app_node(int, 2, 5),
    ])
    throwing = _ -> error("lazy argument was evaluated")
    @test run_typed(prepare_typed(lazy_term, lazy_language),
        [1, throwing, 0]) == 1

    count = Ref(0)
    once_language = Language(table, [
        LeafSpec(arrow!(table, int, fint), :add),
        LeafSpec(fint, :tick), LeafSpec(int, :zero),
    ])
    once_term = TypedTerm(TermNode[
        leaf_node(arrow!(table, int, fint), 1),
        var_node(int, 0),
        app_node(fint, 1, 2),
        var_node(int, 0),
        app_node(int, 3, 4),
        abs_node(fint, 5),
        leaf_node(fint, 2),
        leaf_node(int, 3),
        app_node(int, 7, 8),
        app_node(int, 6, 9),
    ])
    tick = _ -> (count[] += 1; 42)
    @test run_typed(prepare_typed(once_term, once_language),
        [x -> y -> x + y, tick, 0]) == 84
    @test count[] == 1

    snapshot_language = Language(table, [LeafSpec(int, :captured)])
    snapshot_term = TypedTerm(TermNode[
        leaf_node(int, 1),
        abs_node(fint, 1),
    ])
    bindings = [1]
    old_closure = run_typed(prepare_typed(snapshot_term, snapshot_language), bindings)
    bindings[1] = 2
    @test old_closure(0) == 1

    ignore_language = Language(table, [LeafSpec(arrow!(table, fint, int), :ignore)])
    ignore_term = TypedTerm(TermNode[
        leaf_node(arrow!(table, fint, int), 1),
        var_node(int, 0),
        abs_node(fint, 2),
        app_node(int, 1, 3),
    ])
    ignore = callback -> (callback(1); 99)
    budget = run_typed(prepare_typed(ignore_term, ignore_language), [ignore]; max_steps=6)
    @test budget isa Err
    @test budget.value isa TypedEvaluationError
    @test budget.value.reason == :step_limit

    ast_language = Language(table, [LeafSpec(fint, :ast), LeafSpec(int, :one)])
    ast_term = TypedTerm(TermNode[
        leaf_node(fint, 1), leaf_node(int, 2), app_node(int, 1, 2),
    ])
    make_ast = _ -> AstTree(ConstNode(:one))
    rejected = run_typed(prepare_typed(ast_term, ast_language), [make_ast, 1])
    @test rejected isa Err
    @test rejected.value.reason == :host_returned_ast
end
