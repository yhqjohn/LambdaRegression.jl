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

Result{T, E} = Union{T, Err{E}}
Base.convert(::Type{Result{T, E}}, x) where {T, E} = convert(T, x)
Base.convert(::Type{Result{T, E}}, x::Err{E2}) where {T, E, E2} = Err(convert(E, x.value))
Base.convert(::Type{Result{T, E}}, x::T) where {T, E} = x
Base.convert(::Type{Result{T, E}}, x::Err{E}) where {T, E} = x

# struct UncheckedException <: Exception
#     value:: Err
# end

# global _panic_when_check_fails = false

# set_check_panic!(b::Bool) = global _panic_when_check_fails = b

macro check(expr, passthroughs...)
    if length(passthroughs) == 0
        passthrough_expr = :()
    else
        passthroughs = [:(Err(::$(esc(x)))) for x in passthroughs]
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
                # if _panic_when_check_fails
                #     throw(UncheckedException(result))
                # end
                return result
            @case _
                output = result
        end
        output
    end
end

alias_dict = Dict{Any, String}()

function _show_backup(io::IO, x, T)
    super = supertype(T)
    invoke(Base.show, Tuple{IO, super}, io, x)
end

set_alias!(x, alias) = alias_dict[x] = alias

macro alias(T)
    quote
        function Base.show(io::IO, x::$(esc(T)))
            if haskey(alias_dict, x)
                print(io, alias_dict[x])
            else
                _show_backup(io, x, $(esc(T)))
            end
        end
    end
end
