"""Invocation-owned objective observations for typed search.

`EvaluationState` is deliberately a small runtime data object, rather than a
memoization or cross-run fingerprint layer.  A search registers the owned
output vector produced by each charged evaluation and may register aliases for
the independent population/archive copies of that scored member.  The state
is single-use and is pruned by the caller after each generation.  Consumers
borrow vectors internally and the public accessor always returns a detached
copy.
"""

mutable struct EvaluationState
    outputs::IdDict{Any,Vector{Float64}}
    consumed::Bool
end

"""Create empty invocation-owned evaluation state."""
EvaluationState() = EvaluationState(IdDict{Any,Vector{Float64}}(), false)

function _typed_evaluation_begin!(state::EvaluationState)
    state.consumed && throw(ArgumentError(
        "evaluation_state is invocation-owned and must be fresh for each search"))
    state.consumed = true
    return state
end

@inline function _typed_evaluation_key(member)
    member isa ScoredTerm || throw(ArgumentError(
        "evaluation output access requires a scored typed-search member"))
    # The node vector is owned by the member.  IdDict identity prevents two
    # independent charged observations of equal syntax/loss from conflating;
    # birth is deliberately not used as the identity.
    return member.term.nodes
end

"""Register one owned Float64 output vector for a scored member."""
function _typed_evaluation_register!(state::EvaluationState,
                                     member::ScoredTerm,
                                     output::Vector{Float64})
    key = _typed_evaluation_key(member)
    previous = get(state.outputs, key, nothing)
    if previous !== nothing && previous !== output && !isequal(previous, output)
        throw(ArgumentError("conflicting observations for one member identity"))
    end
    state.outputs[key] = output
    return output
end

"""Register an independent member node buffer as an alias of an output."""
function _typed_evaluation_alias!(state::EvaluationState,
                                  member::ScoredTerm,
                                  output::Vector{Float64})
    state.outputs[_typed_evaluation_key(member)] = output
    return output
end

"""Borrow a retained vector for internal classification/selection."""
function _typed_evaluation_borrow(state::EvaluationState, member::ScoredTerm)
    output = get(state.outputs, _typed_evaluation_key(member), nothing)
    output isa Vector{Float64} || throw(ArgumentError(
        "evaluation state has no retained output for this scored member"))
    return output
end

"""Return a detached output copy for a member belonging to `state`."""
function evaluation_outputs(state::EvaluationState, member::ScoredTerm)
    output = _typed_evaluation_borrow(state, member)
    return copy(output)
end

function _typed_evaluation_keep!(state::EvaluationState, members)
    keep = IdDict{Any,Nothing}()
    for member in members
        member isa ScoredTerm || throw(ArgumentError(
            "evaluation pruning requires scored typed-search members"))
        keep[_typed_evaluation_key(member)] = nothing
    end
    for key in collect(keys(state.outputs))
        haskey(keep, key) || delete!(state.outputs, key)
    end
    return state
end

"""Prune observations to live population, archive, and current generation."""
function _typed_evaluation_prune!(state::EvaluationState, islands, archive;
                                  current=())
    members = ScoredTerm[]
    for island in islands
        append!(members, island.population)
    end
    append!(members, values(archive))
    append!(members, current)
    return _typed_evaluation_keep!(state, members)
end

@testitem "TypedSearch evaluation state is detached, bounded, and single-use" begin
    using Test
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: ScoredTerm,
        _typed_evaluation_register!, _typed_evaluation_alias!,
        _typed_evaluation_begin!, _typed_evaluation_keep!
    using LambdaRegression.TypeSystem: FloatTy

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaf = add_leaf!(language, scalar, :x)
    term = TypedTerm(TermNode[leaf_node(scalar, leaf)])
    member = ScoredTerm(term, 1.0, 1)
    clone = ScoredTerm(TypedTerm(copy(term.nodes)), 1.0, 2)
    state = EvaluationState()
    output = [1.0, -0.0]
    _typed_evaluation_register!(state, member, output)
    _typed_evaluation_alias!(state, clone, output)
    detached = evaluation_outputs(state, member)
    detached[1] = 9.0
    @test evaluation_outputs(state, member) == [1.0, -0.0]
    @test evaluation_outputs(state, clone) == [1.0, -0.0]
    _typed_evaluation_begin!(state)
    @test_throws ArgumentError _typed_evaluation_begin!(state)
    # The aliases are identity-owned, so an equal syntax/loss record from a
    # different observation cannot accidentally access this run's cache.
    other = ScoredTerm(TypedTerm(copy(term.nodes)), 1.0, 1)
    @test_throws ArgumentError evaluation_outputs(state, other)
    _typed_evaluation_keep!(state, [clone])
    @test length(state.outputs) == 1
    @test_throws ArgumentError evaluation_outputs(state, member)
    @test evaluation_outputs(state, clone) == [1.0, -0.0]
end
