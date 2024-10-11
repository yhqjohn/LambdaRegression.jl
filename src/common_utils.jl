using MLStyle

repr_letters = "zyxwvutsrqpnmlkjihgfedcba" # skip o to avoid confusion with 0
function to_letter(i::UInt)
    if i<=24
        repr_letters[i]
    else
        "x_$(i-24)"
    end
end

U1 = UInt(1)
U0 = UInt(0)


@data Result{R,E} begin
    Ok(::R)
    Err(::E)
end
# upcasting
Base.convert(::Type{Result{R,E}}, r::Ok{R1, E1}) where {R, E, R1<:R, E1<:E} = Ok{R,E}(r._1)
Base.convert(::Type{Result{R,E}}, e::Err{R1, E1}) where {R, E, R1<:R, E1<:E} = Err{R,E}(e._1)
# downcasting
Base.convert(::Type{Result{R1, E1}}, r::Ok{R, E}) where {R, E, R1<:R, E1<:E} = Ok{R1,E1}(convert(R1, r._1))
Base.convert(::Type{Result{R1, E1}}, e::Err{R, E}) where {R, E, R1<:R, E1<:E} = Err{R1,E1}(convert(E1, e._1))

# convenience constructors
Base.convert(::Type{Result{R,E}}, r::R1) where {R, E, R1<:R} = Ok{R,E}(r)
Ok(r::R) where R = Ok{R,Union{}}(r)
Err(e::E) where E = Err{Union{},E}(e)


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
            @case Ok(r)
                output = r
            @case $(passthrough_expr)
                output = result
            @case Err(_)
                return result
        end
        output
    end
end