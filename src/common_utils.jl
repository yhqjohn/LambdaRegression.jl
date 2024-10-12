using MLStyle

repr_letters = "zyxwvutsrqpnmlkjihgfedcba" # skip o to avoid confusion with 0
function to_letter(i::UInt)
    if i<=24
        repr_letters[i]
    else
        "x_$(i-24)"
    end
end

@as_record struct Err{E}
    value::E
end

macro check(expr, passthroughs...)
    if length(passthroughs) == 0
        passthrough_expr = :()
    else
        passthroughs = [:(Err(::$x)) for x in passthroughs]
        passthrough_expr = reduce(passthroughs) do x, y
            :($x || $y)
        end
    end
    quote
        result = $(esc(expr))
        output = nothing
        @switch result begin
            @case $(passthrough_expr)
                output = result
            @case Err(_)
                return result
            @case _
                output = result
        end
        output
    end
end