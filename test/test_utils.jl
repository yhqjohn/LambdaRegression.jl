using LambdaRegression: Result, Ok, Err, @check

IntErr = Result{Real, String}

ok1 = Ok(1)
err1 = Err("error")

function passresult(x)::IntErr
    @check x
end

@test passresult(ok1) === Ok{Real, String}(1)
@test passresult(err1) === Err{Real, String}("error")

#test upcasting
function increase(x)::IntErr
    (@check x) + 1
end
@test increase(ok1) === Ok{Real, String}(2)
@test increase(err1) === Err{Real, String}("error")

#test pass through
function passthrough(x)
    result = @check x String
    if result isa Int
        return 1
    else
        return 2
    end
end
@test passthrough(ok1) === 1
@test passthrough(err1) === 2
@test passthrough(Err(1)) === Err(1)

