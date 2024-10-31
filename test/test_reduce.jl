using LambdaRegression.DeBruijnModule: to_debrujin, ProxyDB
using LambdaRegression.ParseModule: @lambda
using LambdaRegression.ReduceModule: beta_reduce, normalized
using LambdaRegression.BuiltinTermsModule: FIBB, YCOMBINATOR
using Test

ident1 = to_debrujin(@lambda (x->x)*1)
display(ident1)
display(beta_reduce(ident1))
@test beta_reduce(ident1) == ProxyDB(1)
@test beta_reduce(beta_reduce(ident1)) == normalized

yid = YCOMBINATOR(ident1)
display(yid)
yid = beta_reduce(yid)
display(yid)

fibb = YCOMBINATOR(FIBB)
let fibb5 = fibb(5)
    while true
        fibb5 = beta_reduce(fibb5, true)
        display(fibb5)
        if isa(fibb5, ProxyDB)
            break
        end
    end
    fibb5
end
@test fibb5 == ProxyDB(5)
