"""Bounded, append-only trajectory recording for typed search.

The recorder is deliberately a passive side channel.  Search state is never
serialized as a whole and no recorder operation calls an objective or an RNG.
Each chunk is a self-contained `Serialization` payload with a deterministic
checksum.  A missing completion marker is reported as an incomplete tail;
checksum or decoding errors in an otherwise complete chunk are not swallowed.
"""

using Serialization: serialize, deserialize

const _TYPED_TRACE_MAGIC = UInt8[0x54, 0x53, 0x54, 0x52, 0x01, 0x00, 0x00, 0x00]
const _TYPED_TRACE_CHUNK_MAGIC = UInt8[0x43, 0x48, 0x4e, 0x4b]
const _TYPED_TRACE_DONE_MAGIC = UInt8[0x44, 0x4f, 0x4e, 0x45]

struct _TypedTraceTailEOF <: Exception end

"""One fixed-shape event in a `TypedSearchTrace` chunk.

`candidate_id` identifies term content only; `membership_id` identifies one
live/archive occurrence.  The latter is independent of an AGE/NSGA birth
counter, which is local to an island and can repeat across islands.  `outputs`
is optional observational data already returned by a caller's objective.
"""
struct TypedTraceEvent
    sequence::Int64
    kind::Symbol
    status::Symbol
    island::Int32
    generation::Int32
    attempt::Int64
    evaluation::Int64
    operator::Symbol
    parent_id::Int64
    donor_id::Int64
    child_id::Int64
    parent_membership::Int64
    donor_membership::Int64
    child_membership::Int64
    parent_slot::Int32
    donor_slot::Int32
    child_slot::Int32
    evicted_id::Int64
    evicted_membership::Int64
    parent_loss::Float64
    donor_loss::Float64
    child_loss::Float64
    parent_written_size::Int32
    donor_written_size::Int32
    child_written_size::Int32
    parent_source::Symbol
    donor_source::Symbol
    output_status::Symbol
    outputs::Union{Nothing,Vector{Float64}}
    term::Union{Nothing,TypedTerm}
end

"""One population/archive member in a checkpoint."""
struct TypedTraceMember
    candidate_id::Int64
    membership_id::Int64
    island::Int32
    slot::Int32
    birth::Int64
    loss::Float64
    compact_size::Int32
    written_size::Int32
    source::Symbol
    term::Union{Nothing,TypedTerm}
    outputs::Union{Nothing,Vector{Float64}}
end

"""A self-contained decoded trajectory chunk."""
struct TypedTraceChunk
    sequence::Int64
    records::Vector{Any}
end

"""Summary returned by `read_trace_chunks`."""
struct TypedTraceReadSummary
    complete::Bool
    chunks::Int
    records::Int
    tail_incomplete::Bool
end

mutable struct TypedSearchTrace
    path::String
    io::IO
    buffer_events::Int
    checkpoint_interval::Int
    store_terms::Symbol
    metadata::Any
    buffer::Vector{Any}
    sequence::Int64
    chunk_sequence::Int64
    next_checkpoint::Int64
    closed::Bool
    completed::Bool
    content_ids::Dict{UInt64,Vector{Int64}}
    content_terms::Dict{Int64,Vector{TermNode}}
    next_candidate_id::Int64
    population_ids::IdDict{Any,Tuple{Int64,Int64,Int32,Int32}}
    archive_ids::IdDict{Any,Tuple{Int64,Int64,Int32,Int32}}
    next_membership_id::Int64
    pending_outputs::Dict{UInt64,Vector{Float64}}
end

function _typed_trace_write_u64(io::IO, value::UInt64)
    v = value
    bytes = ntuple(i -> UInt8((v >> (8 * (i - 1))) & 0xff), 8)
    write(io, UInt8[bytes...])
    return nothing
end

function _typed_trace_read_u64(io::IO)
    bytes = read(io, 8)
    length(bytes) == 8 || throw(_TypedTraceTailEOF())
    value = UInt64(0)
    for (i, byte) in enumerate(bytes)
        value |= UInt64(byte) << (8 * (i - 1))
    end
    return value
end

@inline function _typed_trace_checksum(bytes::Vector{UInt8})
    # Fixed FNV-1a is only a corruption detector, not an equivalence proof.
    value = UInt64(0xcbf29ce484222325)
    for byte in bytes
        value = (value ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return value
end

function _typed_trace_metadata_bytes(metadata)
    io = IOBuffer()
    serialize(io, metadata)
    return take!(io)
end

function TypedSearchTrace(path::AbstractString; buffer_events::Integer=8192,
                          checkpoint_interval::Integer=5000,
                          store_terms::Symbol=:scored, metadata=NamedTuple())
    buffer_events > 0 || throw(ArgumentError("buffer_events must be positive"))
    checkpoint_interval > 0 || throw(ArgumentError(
        "checkpoint_interval must be positive"))
    store_terms in (:none, :scored, :all) || throw(ArgumentError(
        "store_terms must be :none, :scored, or :all"))
    target = String(path)
    isfile(target) && throw(ArgumentError("trajectory path already exists: $target"))
    io = open(target, "w+")
    metadata_bytes = _typed_trace_metadata_bytes(metadata)
    write(io, _TYPED_TRACE_MAGIC)
    _typed_trace_write_u64(io, UInt64(length(metadata_bytes)))
    write(io, metadata_bytes)
    flush(io)
    return TypedSearchTrace(target, io, Int(buffer_events),
        Int(checkpoint_interval), store_terms, metadata, Any[], 0, 0,
        Int64(checkpoint_interval), false,
        false, Dict{UInt64,Vector{Int64}}(), Dict{Int64,Vector{TermNode}}(), 0,
        IdDict{Any,Tuple{Int64,Int64,Int32,Int32}}(),
        IdDict{Any,Tuple{Int64,Int64,Int32,Int32}}(), 0,
        Dict{UInt64,Vector{Float64}}())
end

function _typed_trace_payload(records::Vector{Any})
    io = IOBuffer()
    serialize(io, records)
    return take!(io)
end

function _typed_trace_flush!(trace::TypedSearchTrace)
    trace.closed && throw(ArgumentError("trajectory is already closed"))
    isempty(trace.buffer) && return nothing
    payload = _typed_trace_payload(trace.buffer)
    write(trace.io, _TYPED_TRACE_CHUNK_MAGIC)
    _typed_trace_write_u64(trace.io, UInt64(trace.chunk_sequence))
    _typed_trace_write_u64(trace.io, UInt64(length(payload)))
    _typed_trace_write_u64(trace.io, UInt64(length(trace.buffer)))
    _typed_trace_write_u64(trace.io, _typed_trace_checksum(payload))
    write(trace.io, payload)
    empty!(trace.buffer)
    trace.chunk_sequence += 1
    flush(trace.io)
    return nothing
end

"""Flush pending trajectory records to one append-only chunk."""
function Base.flush(trace::TypedSearchTrace)
    _typed_trace_flush!(trace)
    flush(trace.io)
    return trace
end

"""Append a normal-completion marker after flushing all pending chunks."""
function Base.close(trace::TypedSearchTrace)
    trace.closed && return trace
    _typed_trace_flush!(trace)
    write(trace.io, _TYPED_TRACE_DONE_MAGIC)
    _typed_trace_write_u64(trace.io, UInt64(trace.chunk_sequence))
    flush(trace.io)
    close(trace.io)
    trace.closed = true
    trace.completed = true
    return trace
end

function _typed_trace_content_key(term::TypedTerm)
    value = UInt64(0xcbf29ce484222325)
    for node in term.nodes
        value = (value ⊻ UInt64(node.kind)) * UInt64(0x100000001b3)
        value = (value ⊻ UInt64(reinterpret(UInt32, node.typeid))) * UInt64(0x100000001b3)
        value = (value ⊻ UInt64(reinterpret(UInt32, node.x))) * UInt64(0x100000001b3)
        value = (value ⊻ UInt64(reinterpret(UInt32, node.y))) * UInt64(0x100000001b3)
    end
    return value ⊻ UInt64(length(term.nodes))
end

function _typed_trace_candidate_id!(trace::TypedSearchTrace, term::TypedTerm)
    key = _typed_trace_content_key(term)
    for candidate_id in get(trace.content_ids, key, Int64[])
        get(trace.content_terms, candidate_id, TermNode[]) == term.nodes &&
            return candidate_id
    end
    trace.next_candidate_id += 1
    candidate_id = trace.next_candidate_id
    push!(get!(trace.content_ids, key, Int64[]), candidate_id)
    # This is only a current live/archive/offspring sidecar, never history.
    trace.content_terms[candidate_id] = copy(term.nodes)
    return candidate_id
end

function _typed_trace_membership_id!(trace::TypedSearchTrace, member::ScoredTerm,
                                     island::Int, slot::Int;
                                     source::Symbol=:population)
    candidate_id = _typed_trace_candidate_id!(trace, member.term)
    table = source === :archive ? trace.archive_ids : trace.population_ids
    nodes = member.term.nodes
    current = get(table, nodes, nothing)
    if current !== nothing
        current_candidate, membership_id = current
        current_candidate == candidate_id || throw(ArgumentError(
            "trajectory content identity changed for a live node buffer"))
        table[nodes] = (current_candidate, membership_id, Int32(island), Int32(slot))
        return (candidate_id, membership_id)
    end
    membership_id = begin
        trace.next_membership_id += 1
        trace.next_membership_id
    end
    table[nodes] = (Int64(candidate_id), Int64(membership_id),
        Int32(island), Int32(slot))
    return (candidate_id, membership_id)
end

function _typed_trace_should_store(trace::TypedSearchTrace, status::Symbol)
    trace.store_terms === :all ||
        (trace.store_terms === :scored && status in (:scored, :initial,
            :checkpoint))
end

function _typed_trace_take_outputs!(trace::TypedSearchTrace, term::TypedTerm)
    key = _typed_trace_content_key(term)
    return pop!(trace.pending_outputs, key, nothing)
end

"""Record already-computed outputs for the next matching trace record."""
function record_trace_outputs!(trace::TypedSearchTrace, term::TypedTerm,
                               outputs::AbstractVector)
    trace.closed && throw(ArgumentError("trajectory is already closed"))
    converted = Vector{Float64}(undef, length(outputs))
    for (index, value) in enumerate(outputs)
        value isa Real || throw(ArgumentError("trace outputs must be real"))
        converted[index] = Float64(value)
        isfinite(converted[index]) || throw(ArgumentError(
            "trace outputs must be finite"))
    end
    trace.pending_outputs[_typed_trace_content_key(term)] = converted
    return trace
end

record_trace_outputs!(trace::Nothing, term, outputs) = nothing

function _typed_trace_record!(trace::TypedSearchTrace; kind::Symbol=:attempt,
                              status::Symbol=:unknown, island::Int=0,
                              generation::Int=0, attempt::Int=0,
                              evaluation::Int=0, operator::Symbol=:none,
                              parent=nothing, donor=nothing, child=nothing,
                              child_member=nothing,
                              evicted=nothing,
                              parent_source::Symbol=:none,
                              donor_source::Symbol=:none,
                              parent_slot::Int=0, donor_slot::Int=0,
                              child_slot::Int=0, evicted_slot::Int=0,
                              parent_size::Int=0,
                              donor_size::Int=0, child_size::Int=0,
                              parent_loss::Float64=NaN,
                              donor_loss::Float64=NaN,
                              child_loss::Float64=NaN)
    trace.closed && throw(ArgumentError("trajectory is already closed"))
    parent_island = parent_source === :archive ? 0 : island
    donor_island = donor_source === :archive ? 0 : island
    parent_ref = parent isa ScoredTerm ?
        _typed_trace_membership_id!(trace, parent, parent_island, parent_slot;
            source=parent_source) : (0, 0)
    donor_ref = donor isa ScoredTerm ?
        _typed_trace_membership_id!(trace, donor, donor_island, donor_slot;
            source=donor_source) : (0, 0)
    evicted_ref = evicted isa ScoredTerm ?
        _typed_trace_membership_id!(trace, evicted, island, evicted_slot) : (0, 0)
    child_term = child_member isa ScoredTerm ? child_member.term : child
    child_ref = child_member isa ScoredTerm ?
        _typed_trace_membership_id!(trace, child_member, island, child_slot) :
        child_term isa TypedTerm ? begin
        pseudo_birth = Int64(trace.sequence + 1)
        pseudo = ScoredTerm(child_term, child_loss, pseudo_birth)
        _typed_trace_membership_id!(trace, pseudo, island, child_slot)
    end : (0, 0)
    outputs = child_term isa TypedTerm ? _typed_trace_take_outputs!(trace, child_term) : nothing
    output_status = outputs === nothing ? :missing : :present
    term = child_term isa TypedTerm && _typed_trace_should_store(trace, status) ?
        TypedTerm(copy(child_term.nodes)) : nothing
    trace.sequence += 1
    push!(trace.buffer, TypedTraceEvent(trace.sequence, kind, status,
        Int32(island), Int32(generation), Int64(attempt), Int64(evaluation),
        operator, Int64(parent_ref[1]), Int64(donor_ref[1]), Int64(child_ref[1]),
        Int64(parent_ref[2]), Int64(donor_ref[2]), Int64(child_ref[2]),
        Int32(parent_slot), Int32(donor_slot), Int32(child_slot),
        Int64(evicted_ref[1]), Int64(evicted_ref[2]),
        parent_loss, donor_loss, child_loss, Int32(parent_size), Int32(donor_size),
        Int32(child_size), parent_source, donor_source, output_status, outputs,
        term))
    length(trace.buffer) >= trace.buffer_events && _typed_trace_flush!(trace)
    return nothing
end

function _typed_trace_prune!(trace::TypedSearchTrace,
                             population_nodes::IdDict{Any,Nothing},
                             archive_nodes::IdDict{Any,Nothing})
    filter!(pair -> haskey(population_nodes, pair[1]), trace.population_ids)
    filter!(pair -> haskey(archive_nodes, pair[1]), trace.archive_ids)
    live_candidates = Set(value[1] for value in values(trace.population_ids))
    union!(live_candidates, (value[1] for value in values(trace.archive_ids)))
    for (candidate_id, _) in collect(trace.content_terms)
        candidate_id in live_candidates || delete!(trace.content_terms, candidate_id)
    end
    for (key, candidates) in collect(trace.content_ids)
        kept = filter(candidate_id -> candidate_id in live_candidates, candidates)
        isempty(kept) ? delete!(trace.content_ids, key) : (trace.content_ids[key] = kept)
    end
    live_hashes = Set(_typed_trace_content_key(TypedTerm(trace.content_terms[id]))
        for id in live_candidates if haskey(trace.content_terms, id))
    filter!(pair -> pair[1] in live_hashes, trace.pending_outputs)
    return nothing
end

function _typed_trace_member!(trace::TypedSearchTrace, member::ScoredTerm,
                              island::Int, slot::Int, source::Symbol,
                              language=nothing)
    candidate_id, membership_id = _typed_trace_membership_id!(trace, member,
        island, slot; source)
    term = _typed_trace_should_store(trace, :checkpoint) ?
        TypedTerm(copy(member.term.nodes)) : nothing
    # Checkpoints are repeatable observations of each island; do not consume
    # a pending output merely because the same seed appears on another island.
    outputs = get(trace.pending_outputs, _typed_trace_content_key(member.term), nothing)
    written_size = language isa Language ? _typed_written_size(member.term, language) :
        length(member.term.nodes)
    return TypedTraceMember(Int64(candidate_id), Int64(membership_id),
        Int32(island), Int32(slot), Int64(member.birth), member.loss,
        Int32(length(member.term.nodes)), Int32(written_size), source, term, outputs)
end

"""Append a detached all-island population/archive checkpoint."""
function _typed_trace_checkpoint!(trace::Nothing, phase::Symbol, populations;
                                  archive=nothing, language=nothing,
                                  attempt::Int=0, evaluation::Int=0,
                                  generation::Int=0)
    return nothing
end

function _typed_trace_checkpoint!(trace::TypedSearchTrace, phase::Symbol,
                                  populations; archive=nothing, language=nothing,
                                  attempt::Int=0, evaluation::Int=0,
                                  generation::Int=0)
    members = TypedTraceMember[]
    population_nodes = IdDict{Any,Nothing}()
    archive_nodes = IdDict{Any,Nothing}()
    for (island_id, population_source) in enumerate(populations)
        population = hasproperty(population_source, :population) ?
            population_source.population : population_source
        for (slot, member) in enumerate(population)
            population_nodes[member.term.nodes] = nothing
            record = _typed_trace_member!(trace, member, island_id, slot,
                :population, language)
            push!(members, record)
        end
    end
    archived = TypedTraceMember[]
    if archive !== nothing
        for size in sort!(collect(keys(archive)))
            member = archive[size]
            archive_nodes[member.term.nodes] = nothing
            record = _typed_trace_member!(trace, member, 0, Int(size), :archive,
                language)
            push!(archived, record)
        end
    end
    trace.sequence += 1
    push!(trace.buffer, (kind=:checkpoint, sequence=trace.sequence,
        phase, attempt, evaluation, generation, members, archive=archived))
    _typed_trace_prune!(trace, population_nodes, archive_nodes)
    length(trace.buffer) >= trace.buffer_events && _typed_trace_flush!(trace)
    return nothing
end

function _typed_trace_start!(trace::Nothing)
    return nothing
end

function _typed_trace_start!(trace::TypedSearchTrace)
    trace.closed && throw(ArgumentError("trajectory is already closed"))
    return nothing
end

function _typed_trace_finish!(trace::Nothing)
    return nothing
end

function _typed_trace_finish!(trace::TypedSearchTrace)
    close(trace)
    return nothing
end

@inline function _typed_trace_due!(trace::Nothing, attempt::Int)
    return false
end

@inline function _typed_trace_due!(trace::TypedSearchTrace, attempt::Int)
    attempt < trace.next_checkpoint && return false
    while attempt >= trace.next_checkpoint
        trace.next_checkpoint += trace.checkpoint_interval
    end
    return true
end

function _typed_trace_checkpoint_island!(trace::Nothing, phase::Symbol,
                                         island, island_id::Int; kwargs...)
    return nothing
end

function _typed_trace_checkpoint_island!(trace::TypedSearchTrace, phase::Symbol,
                                         island, island_id::Int;
                                         attempt::Int=0, evaluation::Int=0,
                                         generation::Int=0, language=nothing)
    members = TypedTraceMember[]
    keep = Set{Tuple{Int32,Int64,Int64}}()
    for (slot, member) in enumerate(island.population)
        candidate_id, membership_id = _typed_trace_membership_id!(trace, member,
            island_id, slot)
        outputs = get(trace.pending_outputs,
            _typed_trace_content_key(member.term), nothing)
        written_size = language isa Language ? _typed_written_size(member.term, language) :
            length(member.term.nodes)
        push!(members, TypedTraceMember(Int64(candidate_id), Int64(membership_id),
            Int32(island_id), Int32(slot), Int64(member.birth), member.loss,
            Int32(length(member.term.nodes)), Int32(written_size),
            :population, nothing, outputs))
        push!(keep, (Int32(island_id), Int64(candidate_id), Int64(member.birth)))
    end
    trace.sequence += 1
    push!(trace.buffer, (kind=:survivor_checkpoint, sequence=trace.sequence,
        phase, attempt, evaluation, generation, members, archive=TypedTraceMember[]))
    # A survivor checkpoint is intentionally membership-only and does not
    # discard the other islands' live identities.
    length(trace.buffer) >= trace.buffer_events && _typed_trace_flush!(trace)
    return nothing
end

function _typed_trace_read_header(io::IO)
    magic = read(io, length(_TYPED_TRACE_MAGIC))
    length(magic) == length(_TYPED_TRACE_MAGIC) || throw(_TypedTraceTailEOF())
    magic == _TYPED_TRACE_MAGIC || throw(ArgumentError("invalid trajectory header"))
    metadata_length = _typed_trace_read_u64(io)
    metadata_length <= typemax(Int) || throw(ArgumentError(
        "trajectory metadata is too large"))
    bytes = read(io, Int(metadata_length))
    length(bytes) == Int(metadata_length) || throw(_TypedTraceTailEOF())
    metadata = deserialize(IOBuffer(bytes))
    return metadata
end

"""Stream decoded chunks to `callback` without retaining prior chunks.

`strict=false` reports a truncated final header/payload through
`tail_incomplete=true`.  It still throws on an invalid header, checksum,
middle-chunk decode failure, or a malformed completion marker.
"""
function read_trace_chunks(callback::Function, path::AbstractString;
                           strict::Bool=false)
    io = open(path, "r")
    chunks = 0
    records = 0
    tail_incomplete = false
    complete = false
    try
        _typed_trace_read_header(io)
        while !eof(io)
            marker = read(io, 4)
            length(marker) == 4 || throw(_TypedTraceTailEOF())
            if marker == _TYPED_TRACE_DONE_MAGIC
                expected_chunks = _typed_trace_read_u64(io)
                expected_chunks == UInt64(chunks) || throw(ArgumentError(
                    "trajectory completion chunk count mismatch"))
                complete = true
                eof(io) || throw(ArgumentError("bytes after trajectory completion marker"))
                break
            end
            marker == _TYPED_TRACE_CHUNK_MAGIC || throw(ArgumentError(
                "invalid trajectory chunk marker"))
            chunk_sequence = _typed_trace_read_u64(io)
            payload_length = _typed_trace_read_u64(io)
            record_count = _typed_trace_read_u64(io)
            checksum = _typed_trace_read_u64(io)
            payload_length <= typemax(Int) || throw(ArgumentError(
                "trajectory chunk is too large"))
            bytes = read(io, Int(payload_length))
            length(bytes) == Int(payload_length) || throw(_TypedTraceTailEOF())
            _typed_trace_checksum(bytes) == checksum || throw(ArgumentError(
                "trajectory chunk checksum mismatch"))
            decoded = deserialize(IOBuffer(bytes))
            decoded isa Vector{Any} || throw(ArgumentError(
                "trajectory chunk payload is not a record vector"))
            length(decoded) == Int(record_count) || throw(ArgumentError(
                "trajectory chunk record count mismatch"))
            chunks += 1
            records += length(decoded)
            callback(TypedTraceChunk(Int64(chunk_sequence), decoded))
        end
        complete || (tail_incomplete = true)
    catch error
        if error isa _TypedTraceTailEOF
            tail_incomplete = true
        else
            rethrow()
        end
    finally
        close(io)
    end
    strict && tail_incomplete && throw(EOFError())
    return TypedTraceReadSummary(complete, chunks, records, tail_incomplete)
end

read_trace_chunks(path::AbstractString, callback::Function; strict::Bool=false) =
    read_trace_chunks(callback, path; strict)

export TypedSearchTrace, TypedTraceEvent, TypedTraceMember, TypedTraceChunk,
    TypedTraceReadSummary, read_trace_chunks, record_trace_outputs!

@testitem "TypedSearch trace chunks and membership identities" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_trace_checkpoint!

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaf = add_leaf!(language, scalar, :trace_one)
    term = TypedTerm(TermNode[leaf_node(scalar, leaf)])
    path = joinpath(mktempdir(), "typed-search-trace.bin")
    trace = TypedSearchTrace(path; buffer_events=1, checkpoint_interval=8)
    m1 = ScoredTerm(TypedTerm(copy(term.nodes)), 1.0, 3)
    m2 = ScoredTerm(TypedTerm(copy(term.nodes)), 1.0, 3)
    _typed_trace_checkpoint!(trace, :initial, [[m1, m2]]; language)
    close(trace)
    chunks = TypedTraceChunk[]
    summary = read_trace_chunks(chunk -> push!(chunks, chunk), path)
    members = chunks[1].records[1].members
    @test summary.complete
    @test !summary.tail_incomplete
    @test members[1].birth == members[2].birth
    @test members[1].candidate_id == members[2].candidate_id
    @test members[1].membership_id != members[2].membership_id
end

@testitem "TypedSearch trace distinguishes tail truncation and corruption" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_trace_record!

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaf = add_leaf!(language, scalar, :trace_two)
    term = TypedTerm(TermNode[leaf_node(scalar, leaf)])

    truncated = joinpath(mktempdir(), "typed-search-truncated.bin")
    trace = TypedSearchTrace(truncated; buffer_events=1, checkpoint_interval=8)
    _typed_trace_record!(trace; status=:scored, island=1, child=term,
        child_loss=1.0, child_size=1)
    close(trace)
    open(truncated, "r+") do io
        truncate(io, filesize(truncated) - 3)
    end
    tail = read_trace_chunks(_ -> nothing, truncated)
    @test !tail.complete
    @test tail.tail_incomplete
    @test_throws EOFError read_trace_chunks(_ -> nothing, truncated; strict=true)

    corrupt = joinpath(mktempdir(), "typed-search-corrupt.bin")
    trace = TypedSearchTrace(corrupt; buffer_events=1, checkpoint_interval=8)
    _typed_trace_record!(trace; status=:scored, island=1, child=term,
        child_loss=1.0, child_size=1)
    _typed_trace_record!(trace; status=:generation_failure, island=1,
        operator=:generate)
    close(trace)
    bytes = read(corrupt)
    magic = UInt8[0x43, 0x48, 0x4e, 0x4b]
    chunk_start = findfirst(i -> bytes[i:i + 3] == magic,
        1:(length(bytes) - 3))
    @test chunk_start !== nothing
    payload_start = chunk_start + 4 + 8 + 8 + 8 + 8
    bytes[payload_start] = bytes[payload_start] ⊻ UInt8(0x01)
    open(corrupt, "w") do io
        write(io, bytes)
    end
    @test_throws ArgumentError read_trace_chunks(_ -> nothing, corrupt)
end

@testitem "TypedSearch AGE trace records eviction membership" begin
    using Test
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _typed_trace_record!

    language = Language(TypeTable())
    scalar = intern!(language.types, FloatTy)
    leaf = add_leaf!(language, scalar, :trace_three)
    term = TypedTerm(TermNode[leaf_node(scalar, leaf)])
    path = joinpath(mktempdir(), "typed-search-eviction.bin")
    trace = TypedSearchTrace(path; buffer_events=1, checkpoint_interval=8)
    parent = ScoredTerm(TypedTerm(copy(term.nodes)), 1.0, 1)
    evicted = ScoredTerm(TypedTerm(copy(term.nodes)), 2.0, 2)
    child = ScoredTerm(TypedTerm(copy(term.nodes)), 0.5, 3)
    _typed_trace_record!(trace; status=:scored, island=1, attempt=1,
        evaluation=1, operator=:point, parent, donor=parent,
        child=child.term, child_member=child, evicted,
        parent_slot=1, donor_slot=1, child_slot=1, evicted_slot=1,
        parent_loss=parent.loss, donor_loss=parent.loss,
        child_loss=child.loss, parent_size=1, donor_size=1, child_size=1)
    close(trace)
    chunks = TypedTraceChunk[]
    read_trace_chunks(c -> push!(chunks, c), path)
    event = chunks[1].records[1]
    @test event.child_slot == 1
    @test event.evicted_membership != 0
    @test event.child_membership != event.evicted_membership
end
