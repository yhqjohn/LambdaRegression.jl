module Utils

using MLStyle, TestItems
export @check, Err, @✓

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

@testsnippet UtilsBaseImports begin
    using LambdaRegression.Utils
    using Test
end

@testitem "check" setup=[UtilsBaseImports] begin
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

end # module Utils
