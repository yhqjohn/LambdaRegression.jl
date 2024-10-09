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