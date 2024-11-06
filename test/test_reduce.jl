using LambdaRegression.DeBruijnModule: to_debrujin, ProxyDB
using LambdaRegression.ParseModule: @lambda
using LambdaRegression.ReduceModule: beta_reduce, normalized, external_reduce, external_normalize
using LambdaRegression.LabeledProxyModule: labelize
using LambdaRegression.BuiltinTermsModule
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

sub51 = labelize(@lambda ISUB * 5 * 1)

let sub51 = sub51
    # display(sub51)
    sub51 = beta_reduce(sub51)
    # display(sub51)
    sub51 = beta_reduce(sub51)
    # display(sub51)
    @test sub51.value.value == 4
end



display(sub51)

let sub51 = sub51
    sub51 = external_normalize(sub51)
    display(sub51)
    @test sub51.value.value == 4
end

ift = labelize(@lambda IFTHENELSE*true)
# display(ift)
ift = external_reduce(ift)
# display(ift)
@test ift.value == K


fibb = YCOMBINATOR(FIBB)
let fibb5 = labelize(fibb(3))
    for _ in 1:50
        display(fibb5)
        fibb5 = beta_reduce(fibb5)
        if isa(fibb5, ProxyDB)
            break
        end
    end
    fibb5
end
@test fibb5 == ProxyDB(5)


