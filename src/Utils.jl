module Utils

using MLStyle, TestItems
export @check, Err, @✓, MultipleMapView

########################################################
# Rust-style error handling
########################################################
"""
    Err{E}

Wrapper for rust-style error.
"""
struct Err{E}
    value::E
end
@as_record Err

"""
    @check expr

If `expr` is an [`Err`](@ref), directly return from the current function.
Otherwise, give the value of `expr`.
An alias for `@check` is [`@✓`](@ref).
"""
macro check(expr)
    quote
        result = $(esc(expr))
        output = nothing
        @switch result begin
            @case Err(_)
                return result
            @case _
                output = result
        end
        output
    end
end

"""
    @✓ expr

Alias for [`@check`](@ref).
"""
const var"@✓" = var"@check"

@testitem "check" begin
    using LambdaRegression.Utils
    using Test
    function foo(x)
        if x > 0
            return x
        else
            return Err(x)
        end
    end
    
    function bar(x)
        1 + (@check foo(x)) 
    end
    
    @test bar(1) == 2
    @test bar(-1) == Err(-1)

    function baz(x)
        1 + (@✓ foo(x)) 
    end
    
    @test baz(1) == 2
    @test baz(-1) == Err(-1)

end

struct MultipleMapView
    maps::Vector
    MultipleMapView(maps::Vector) = new(maps)
    MultipleMapView(maps...) = new(maps)
end
Base.getindex(view::MultipleMapView, key) = get(view, key) do 
    throw(KeyError(key))
end

function Base.get(view::MultipleMapView, key, default)
    for map in view.maps
        if haskey(map, key)
            return map[key]
        end
    end
    return default
end

function Base.get(f::Base.Callable, view::MultipleMapView, key)
    for map in view.maps
        if haskey(map, key)
            return map[key]
        end
    end
    return f()
end

Base.haskey(view::MultipleMapView, key) = any(haskey(map, key) for map in view.maps)


end # module Utils