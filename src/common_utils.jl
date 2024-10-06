repr_letters = "zyxwvutsrqpnmlkjihgfedcba" # skip o to avoid confusion with 0
function to_letter(i::UInt)
    if i<=24
        repr_letters[i+1]
    else
        "x_$(i-24)"
    end
end