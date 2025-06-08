"""
    _parallel_map(f, items, parallel=(; enabled=true, workers=Threads.nthreads(:default)))

Map `f` over a vector, preserving input order. When enabled, at most
`min(workers, length(items), Threads.nthreads(:default))` worker tasks pull
indices from a shared atomic counter; task failures propagate after all spawned
workers have finished. Each worker invokes `f(item)` directly, so callers own
any state or workspace lifetime they need.
"""
function _parallel_map(
    f,
    items::AbstractVector,
    parallel::NamedTuple=(; enabled=true, workers=Threads.nthreads(:default)),
)
    enabled = hasproperty(parallel, :enabled) ? parallel.enabled : true
    workers = hasproperty(parallel, :workers) ? parallel.workers : Threads.nthreads(:default)

    enabled isa Bool || throw(ArgumentError("parallel.enabled must be a Bool"))
    workers isa Integer && !(workers isa Bool) && workers > 0 ||
        throw(ArgumentError("parallel.workers must be a positive integer"))

    n = length(items)
    worker_count = Int(min(workers, n, Threads.nthreads(:default)))
    if !enabled || worker_count <= 1
        return map(f, items)
    end

    results = Vector{Any}(undef, n)
    next_index = Threads.Atomic{Int}(1)

    @sync begin
        for _ in 1:worker_count
            Threads.@spawn begin
                while true
                    index = Threads.atomic_add!(next_index, 1)
                    index > n && break
                    results[index] = f(items[index])
                end
            end
        end
    end

    return results
end

@testitem "Regression parallel map preserves order and serial behavior" begin
    using Test
    using LambdaRegression.Regression: _parallel_map

    items = collect(1:32)
    @test _parallel_map(x -> x * x, items, (; enabled=true, workers=4)) == items .^ 2
    @test _parallel_map(x -> x + 1, items, (; enabled=false, workers=8)) == items .+ 1
    @test _parallel_map(x -> x + 1, items, (; enabled=true, workers=1)) == items .+ 1
    @test_throws ArgumentError _parallel_map(identity, items, (; enabled=true, workers=true))
end

@testitem "Regression parallel map runs eight workers concurrently when available" begin
    using Test
    import Base.Threads
    using LambdaRegression.Regression: _parallel_map

    if Threads.nthreads(:default) >= 8
        ready = Threads.Atomic{Int}(0)
        thread_ids = Vector{Int}(undef, 8)
        results = _parallel_map(collect(1:8), (; enabled=true, workers=8)) do item
            thread_ids[item] = Threads.threadid()
            Threads.atomic_add!(ready, 1)
            deadline = time_ns() + 5_000_000_000
            while ready[] < 8
                time_ns() < deadline || error("eight parallel workers did not enter the callback")
            end
            item
        end

        @test results == collect(1:8)
        @test length(unique(thread_ids)) == 8
    else
        @test true
    end
end

@testitem "Regression parallel map propagates errors after draining workers" begin
    using Test
    import Base.Threads
    using LambdaRegression.Regression: _parallel_map

    finished = Threads.Atomic{Int}(0)
    if Threads.nthreads(:default) >= 2
        entered = Threads.Atomic{Int}(0)
        f = item -> begin
            Threads.atomic_add!(entered, 1)
            deadline = time_ns() + 5_000_000_000
            while entered[] < 2
                yield()
                time_ns() < deadline || error("both parallel workers did not enter the callback")
            end
            item == 1 && throw(ArgumentError("expected worker failure"))
            sleep(0.02)
            Threads.atomic_add!(finished, 1)
            item
        end
        @test_throws Exception _parallel_map(f, [1, 2], (; enabled=true, workers=2))
        @test finished[] == 1
    else
        f = item -> begin
            item == 1 && throw(ArgumentError("expected serial failure"))
            Threads.atomic_add!(finished, 1)
            item
        end
        @test_throws ArgumentError _parallel_map(f, [1, 2], (; enabled=true, workers=2))
        @test finished[] == 0
    end
end
