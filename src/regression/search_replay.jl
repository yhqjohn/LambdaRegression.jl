"""
    SearchReplayWriter(path; every_attempts=1000, topk=3, metadata=(;))

Append-only, framed Julia Serialization export of staged live
winners. The path must be new. The writer owns its stream; call `close` (normally
in `finally`). One writer binds to one age-search state and stores the stable
target/world/input context once. It retains no population or stage history.
Only current valid live winners and at most two actually available parents per
winner are exported. A missing parent remains an ID, never an invented example.

Pass `replay=writer` to `advance_age_search!` for initial, periodic cumulative
attempt, and call-final stages. This observer does not score, select or alter RNG.
Use `record_search_stage!` for explicit boundaries. Files are trusted Julia data,
not portable checkpoints; use a compatible Julia/package environment to read.
"""
mutable struct SearchReplayWriter
    io::IOStream
    every_attempts::Int
    topk::Int
    metadata::NamedTuple
    max_frame_bytes::Int
    owner::WeakRef
    bound::Bool
    last_key::Union{Nothing,Tuple{Int,Symbol}}
end

const _SEARCH_REPLAY_MAGIC = UInt8[0x4c,0x52,0x53,0x52,0x30,0x30,0x30,0x31]

function SearchReplayWriter(path::AbstractString; every_attempts::Int=1000,
        topk::Int=3, metadata::NamedTuple=(;), max_frame_bytes::Int=64 * 1024^2)
    every_attempts > 0 || throw(ArgumentError("every_attempts must be positive"))
    topk > 0 || throw(ArgumentError("topk must be positive"))
    max_frame_bytes > 0 || throw(ArgumentError("max_frame_bytes must be positive"))
    ispath(path) && throw(ArgumentError("search replay path already exists"))
    mkpath(dirname(abspath(path)))
    io = open(path, "w")
    write(io, _SEARCH_REPLAY_MAGIC)
    flush(io)
    return SearchReplayWriter(io, every_attempts, topk, metadata, max_frame_bytes,
        WeakRef(nothing), false, nothing)
end

Base.close(writer::SearchReplayWriter) = close(writer.io)
Base.isopen(writer::SearchReplayWriter) = isopen(writer.io)

function _write_search_replay_frame(writer, record)
    isopen(writer) || throw(ArgumentError("search replay writer is closed"))
    buffer = IOBuffer()
    Serialization.serialize(buffer, record)
    bytes = take!(buffer)
    length(bytes) <= writer.max_frame_bytes || throw(ArgumentError(
        "search replay frame exceeds writer bound; increase max_frame_bytes explicitly"))
    size = UInt64(length(bytes))
    write(writer.io, size, ~size, bytes)
    flush(writer.io)
    return nothing
end

function _bind_search_replay!(writer::SearchReplayWriter, state::AgeSearchState)
    if writer.bound
        writer.owner.value === state || throw(ArgumentError(
            "a search replay writer belongs to exactly one age-search state"))
        return nothing
    end
    c = state.reg.components
    _write_search_replay_frame(writer, (; kind=:header, format_version=1,
        julia_version=string(VERSION), target_ty=state.reg.target_ty,
        global_consts=c.global_consts, global_consts_ty=c.global_consts_ty,
        input_slots=c.input_slots, input_types=c.input_types,
        seed=state.seed, config=state.reg.config, metadata=writer.metadata))
    writer.owner = WeakRef(state)
    writer.bound = true
    return nothing
end

_replay_record(record::AgeSearchIndividual) = (; id=record.id,
    program=record.program, loss=record.loss, complexity=record.complexity,
    parent_ids=record.parent_ids, origin=record.origin, birth=record.birth)

"""Append one bounded current-live stage, skipping an identical last stage key.

Winners are ranked by `(loss, complexity, id)`. Parent materials are looked up
only in currently available live/recent/archive records and retain real IDs.
No previous-stage winner is treated as an ancestor. Returns whether written.
"""
function record_search_stage!(writer::SearchReplayWriter, state::AgeSearchState;
        stage::Symbol=:manual)
    isopen(writer) || throw(ArgumentError("search replay writer is closed"))
    _bind_search_replay!(writer, state)
    key = (state.counters.raw_attempts, stage)
    key == writer.last_key && return false
    live = [record for population in state.populations for record in population
        if record.valid && isfinite(record.loss)]
    sort!(live; by=record -> (record.loss, record.complexity, record.id))
    unique!(record -> record.id, live)
    resize!(live, min(length(live), writer.topk))
    available = Dict(record.id => record for record in Iterators.flatten((
        Iterators.flatten(state.populations), state.recent_valid_offspring, values(state.archive))))
    winners = map(live) do record
        parents = [_replay_record(available[id]) for id in
            Iterators.take(record.parent_ids, 2) if haskey(available, id)]
        merge(_replay_record(record), (; parents))
    end
    _write_search_replay_frame(writer, (; kind=:stage, stage,
        attempts=state.counters.raw_attempts,
        objective_calls=state.counters.objective_calls,
        sample_executions=state.counters.sample_executions, winners))
    writer.last_key = key
    return true
end

_replay_start!(::Nothing, state) = nothing
function _replay_start!(writer::SearchReplayWriter, state)
    !writer.bound && record_search_stage!(writer, state; stage=:initial)
    _bind_search_replay!(writer, state)
    return nothing
end
_replay_periodic!(::Nothing, state) = nothing
function _replay_periodic!(writer::SearchReplayWriter, state)
    state.counters.raw_attempts % writer.every_attempts == 0 &&
        record_search_stage!(writer, state; stage=:periodic)
    return nothing
end
_replay_finish!(::Nothing, state) = nothing
_replay_finish!(writer::SearchReplayWriter, state) = record_search_stage!(writer, state; stage=:final)

"""
    read_search_replay(path; allow_truncated_tail=false, max_frame_bytes=64*1024^2)

Read a trusted replay in a matching Julia/package environment. Returns
`(; header, stages, truncated_tail)`. A partial final frame throws by default;
explicit recovery returns complete preceding frames with `truncated_tail=true`.
Bad framing, oversized frames and invalid serialized data always throw. The
length bound is checked before payload allocation. Reading collects the stages;
the writer itself streams and retains no history.
"""
function read_search_replay(path::AbstractString; allow_truncated_tail::Bool=false,
        max_frame_bytes::Int=64 * 1024^2)
    max_frame_bytes > 0 || throw(ArgumentError("max_frame_bytes must be positive"))
    header = nothing
    stages = NamedTuple[]
    truncated = false
    open(path, "r") do io
        read(io, length(_SEARCH_REPLAY_MAGIC)) == _SEARCH_REPLAY_MAGIC ||
            throw(ArgumentError("invalid search replay magic/version"))
        while !eof(io)
            remaining = filesize(path) - position(io)
            if remaining < 16
                truncated = true
                break
            end
            size = read(io, UInt64)
            complement = read(io, UInt64)
            complement == ~size || throw(ArgumentError("invalid search replay frame length"))
            0 < size <= max_frame_bytes || throw(ArgumentError("search replay frame exceeds read bound"))
            if size > filesize(path) - position(io)
                truncated = true
                break
            end
            bytes = read(io, Int(size))
            record = Serialization.deserialize(IOBuffer(bytes))
            record isa NamedTuple || throw(ArgumentError("invalid search replay record"))
            if header === nothing
                get(record, :kind, nothing) === :header && get(record, :format_version, 0) == 1 ||
                    throw(ArgumentError("search replay is missing a supported header"))
                header = record
            else
                get(record, :kind, nothing) === :stage ||
                    throw(ArgumentError("unexpected record in search replay"))
                push!(stages, record)
            end
        end
    end
    truncated && !allow_truncated_tail && throw(ArgumentError("truncated search replay tail"))
    header === nothing && throw(ArgumentError("search replay has no complete header"))
    return (; header, stages, truncated_tail=truncated)
end

export SearchReplayWriter, record_search_stage!, read_search_replay

@testitem "Search replay streams bounded live stages and preserves continuation provenance" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Regression: SearchReplayWriter, read_search_replay, record_search_stage!
    reg, data = age_test_fixture(population_size=2)
    state = initialize_age_search(reg, data; population_count=1, population_size=2, seed=9)
    observed = Int[]
    proposal = (state, island, left, right; rng) -> begin
        count = state.counters.raw_attempts
        push!(observed, count)
        count == 2 && return Err(:bounded_failure)
        Program(AstTree(AbsNode(:z, ConstNode(:k)));
            local_consts=Dict{Symbol,Any}(:k => Float64(count)),
            local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy))
    end
    mktempdir() do directory
        path = joinpath(directory, "stages.lrsr")
        writer = SearchReplayWriter(path; every_attempts=2, topk=1, metadata=(; run=:unit))
        try
            advance_age_search!(state; max_attempts=2, max_evaluations=2, proposal, replay=writer)
            advance_age_search!(state; max_attempts=2, max_evaluations=2, proposal, replay=writer)
            @test !record_search_stage!(writer, state; stage=:final)
        finally
            close(writer)
        end
        replay = read_search_replay(path)
        @test !replay.truncated_tail
        @test replay.header.metadata == (; run=:unit)
        @test replay.header.seed == state.seed
        @test [(s.attempts,s.stage) for s in replay.stages] ==
            [(0,:initial),(2,:periodic),(2,:final),(4,:periodic),(4,:final)]
        @test all(length(s.winners) == 1 for s in replay.stages)
        @test observed == [1,2,3,4]
        for stage in replay.stages, winner in stage.winners
            @test LambdaRegression.Regression._validate_program_contract(winner.program,
                reg.target_ty, reg.components, "replayed candidate") === nothing
            @test all(parent.id in winner.parent_ids for parent in winner.parents)
        end
        # A complete prefix is recoverable only with an explicit truncated-tail flag.
        bytes = read(path)
        truncated_path = joinpath(directory, "truncated.lrsr")
        write(truncated_path, bytes[1:(end-1)])
        @test_throws ArgumentError read_search_replay(truncated_path)
        recovered = read_search_replay(truncated_path; allow_truncated_tail=true)
        @test recovered.truncated_tail
        @test length(recovered.stages) == length(replay.stages)-1
        @test_throws ArgumentError read_search_replay(path; max_frame_bytes=1)
        @test_throws ArgumentError SearchReplayWriter(path)
        @test_throws ArgumentError record_search_stage!(writer,state)
        tiny=SearchReplayWriter(joinpath(directory,"tiny.lrsr");max_frame_bytes=1)
        try
            @test_throws ArgumentError record_search_stage!(tiny,state)
        finally
            close(tiny)
        end
    end
end

@testitem "Search replay is optional and collection fitting forwards explicit writer" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Regression: SearchReplayWriter, read_search_replay
    using LambdaRegression.FrontendAdapter: fit_collection
    reg, data = age_test_fixture(population_size=2)
    plain = initialize_age_search(reg, data; population_count=1, population_size=2, seed=21)
    recorded = initialize_age_search(reg, data; population_count=1, population_size=2, seed=21)
    proposal = (state,island,left,right; rng) -> Program(
        AstTree(AbsNode(:z,ConstNode(:k)));
        local_consts=Dict{Symbol,Any}(:k=>rand(rng)),
        local_consts_ty=Dict{Symbol,TyLike}(:k=>FloatTy))
    mktempdir() do directory
        path=joinpath(directory,"neutral.lrsr")
        writer=SearchReplayWriter(path;every_attempts=1,topk=2)
        try
            advance_age_search!(recorded;max_attempts=3,max_evaluations=3,proposal,replay=writer)
            advance_age_search!(plain;max_attempts=3,max_evaluations=3,proposal)
            @test recorded.counters.raw_attempts == plain.counters.raw_attempts
            @test LambdaRegression.Regression._age_snapshot(recorded.counters) ==
                LambdaRegression.Regression._age_snapshot(plain.counters)
            @test [[(r.id,r.parent_ids,r.birth,r.loss,r.complexity,r.program.tree,
                r.program.local_consts) for r in population] for population in recorded.populations] ==
                [[(r.id,r.parent_ids,r.birth,r.loss,r.complexity,r.program.tree,
                r.program.local_consts) for r in population] for population in plain.populations]
            @test rand(recorded.rngs[1]) == rand(plain.rngs[1])
            other=initialize_age_search(reg,data;seed=22)
            @test_throws ArgumentError advance_age_search!(other;
                max_attempts=0,max_evaluations=0,replay=writer)
        finally
            close(writer)
        end
        identity=Program(AstTree(AbsNode(:x,IndexNode(0))))
        X=reshape(Float64.(1:6),3,2)
        collection_path=joinpath(directory,"collection.lrsr")
        collection_writer=SearchReplayWriter(collection_path;every_attempts=1)
        try
            fit_collection(X,X;initial_programs=[identity],population_size=2,generations=0,
                search_kwargs=(;max_attempts=0,replay=collection_writer))
        finally
            close(collection_writer)
        end
        replay=read_search_replay(collection_path)
        @test length(replay.stages)==2
        @test replay.stages[1].stage==:initial
        @test replay.stages[2].stage==:final
        @test replay.header.target_ty isa TyFun
    end
end
