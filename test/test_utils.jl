using LambdaRegression: Err, @check

err1 = Err("error")


#test upcasting
function increase(x)
    (@check x) + 1
end
@test increase(1) === 2
@test increase(err1) === Err("error")

#test pass through
function passthrough(x)
    result = @check x String
    if result isa Int
        return 1
    else
        return 2
    end
end
@test passthrough(1) === 1
@test passthrough(err1) === 2
@test passthrough(Err(1)) === Err(1)

