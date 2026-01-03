"""
    _parallel_map(f, items, parallel=(; enabled=true, workers=Threads.nthreads(:default)))

Map `f` over a vector, preserving input order. When enabled, at most
`min(workers, length(items), Threads.nthreads(:default))` worker tasks pull
indices from a shared atomic counter; task failures propagate after all spawned
workers have finished. In multiworker mode, each worker receives its own
`workspaces[worker]` for the whole map call; serial mode uses only the first
workspace. Callers must provide distinct workspaces and keep them private for
the duration of the call.
"""
function _parallel_map(
    f,
    items::AbstractVector,
    parallel::NamedTuple=(; enabled=true, workers=Threads.nthreads(:default));
    workspaces=nothing,
)
    enabled = hasproperty(parallel, :enabled) ? parallel.enabled : true
    workers = hasproperty(parallel, :workers) ? parallel.workers : Threads.nthreads(:default)

    enabled isa Bool || throw(ArgumentError("parallel.enabled must be a Bool"))
    workers isa Integer && !(workers isa Bool) && workers > 0 ||
        throw(ArgumentError("parallel.workers must be a positive integer"))

    n = length(items)
    worker_count = Int(min(workers, n, Threads.nthreads(:default)))
    if !enabled || worker_count <= 1
        workspaces === nothing && return map(f, items)
        isempty(items) && return map(item -> f(first(workspaces), item), items)
        length(workspaces) >= 1 || throw(ArgumentError(
            "one private workspace is required for serial execution",
        ))
        return map(item -> f(first(workspaces), item), items)
    end
    workspaces === nothing || length(workspaces) >= worker_count ||
        throw(ArgumentError("one private workspace is required per worker"))

    results = Vector{Any}(undef, n)
    next_index = Threads.Atomic{Int}(1)

    @sync begin
        for worker in 1:worker_count
            Threads.@spawn begin
                workspace = workspaces === nothing ? nothing : workspaces[worker]
                while true
                    index = Threads.atomic_add!(next_index, 1)
                    index > n && break
                    results[index] = workspaces === nothing ? f(items[index]) :
                        f(workspace, items[index])
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

@testitem "Regression parallel map preserves empty result type and accepts one serial workspace" begin
    using Test
    using LambdaRegression.Regression: _parallel_map

    empty_result = _parallel_map(x -> x + 1, Int[], (; enabled=false, workers=8))
    @test empty_result isa Vector{Int}

    workspace = Ref(0)
    result = _parallel_map(1:4, (; enabled=false, workers=8);
            workspaces=[workspace]) do scratch, item
        scratch[] += 1
        item + 1
    end
    @test result == collect(2:5)
    @test workspace[] == 4
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

@testitem "Regression parallel map keeps workspaces exclusive and drains after workspace failure" begin
    using Test
    import Base.Threads
    using LambdaRegression.Regression: _parallel_map

    if Threads.nthreads(:default) >= 2
        workspaces = [Threads.Atomic{Int}(0) for _ in 1:2]
        uses = [Threads.Atomic{Int}(0) for _ in 1:2]
        violations = Threads.Atomic{Int}(0)
        owners = Dict{Task,Int}()
        owners_lock = ReentrantLock()

        result = _parallel_map(collect(1:12), (; enabled=true, workers=2);
                workspaces) do workspace, item
            workspace_index = findfirst(candidate -> candidate === workspace, workspaces)
            workspace_index === nothing && error("worker received an unknown workspace")
            if Threads.atomic_cas!(workspace, 0, 1) != 0
                Threads.atomic_add!(violations, 1)
            end
            lock(owners_lock)
            try
                previous = get!(owners, current_task(), workspace_index)
                previous == workspace_index || error("worker changed its workspace")
            finally
                unlock(owners_lock)
            end
            Threads.atomic_add!(uses[workspace_index], 1)
            try
                sleep(0.001)
            finally
                workspace[] = 0
            end
            item
        end
        @test result == collect(1:12)
        @test length(owners) == 2
        @test length(unique(collect(values(owners)))) == 2
        @test all(count[] > 0 for count in uses)
        @test violations[] == 0
        @test all(workspace[] == 0 for workspace in workspaces)

        failing_workspaces = [Threads.Atomic{Int}(0) for _ in 1:2]
        entered = Threads.Atomic{Int}(0)
        slow_worker_finished = Threads.Atomic{Bool}(false)
        caught = try
            _parallel_map([1, 2], (; enabled=true, workers=2);
                    workspaces=failing_workspaces) do workspace, item
                if Threads.atomic_cas!(workspace, 0, 1) != 0
                    Threads.atomic_add!(violations, 1)
                end
                Threads.atomic_add!(entered, 1)
                try
                    deadline = time_ns() + 5_000_000_000
                    while entered[] < 2
                        time_ns() < deadline || error("both workspace workers did not enter")
                        yield()
                    end
                    if item == 1
                        throw(ArgumentError("expected workspace worker failure"))
                    end
                    sleep(0.02)
                    slow_worker_finished[] = true
                    item
                finally
                    workspace[] = 0
                end
            end
            nothing
        catch issue
            issue
        end
        @test caught isa Exception
        @test slow_worker_finished[]
        @test all(workspace[] == 0 for workspace in failing_workspaces)

        reused = _parallel_map([3, 4], (; enabled=true, workers=2);
                workspaces=failing_workspaces) do workspace, item
            if Threads.atomic_cas!(workspace, 0, 1) != 0
                Threads.atomic_add!(violations, 1)
            end
            try
                item
            finally
                workspace[] = 0
            end
        end
        @test reused == [3, 4]
        @test violations[] == 0
        @test all(workspace[] == 0 for workspace in failing_workspaces)
    else
        @test true
    end
end
