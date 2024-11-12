using LambdaRegression.DeBruijnModule: ProxyDB, DBNode
using LambdaRegression.ParseModule: @lambda
using LambdaRegression.ReduceModule: beta_reduce, normalized
using LambdaRegression.LabeledProxyModule: labelize
using Test


ident1 = DBNode{Int}(@lambda (x->x)*1)
display(ident1)
display(beta_reduce(ident1))
@test beta_reduce(ident1) == ProxyDB(1)
@test beta_reduce(beta_reduce(ident1)) == normalized

# yid = YCOMBINATOR(ident1)
# display(yid)
# yid = beta_reduce(yid)
# display(yid)

# sub51 = labelize(@lambda ISUB * 5 * 1)

# let sub51 = sub51
#     display(sub51)
#     sub51 = beta_reduce(sub51)
#     display(sub51)
#     sub51 = beta_reduce(sub51)
#     display(sub51)
#     @test sub51.value.value == 4
# end



# display(sub51)

# let sub51 = sub51
#     sub51 = external_normalize(sub51)
#     display(sub51)
#     @test sub51.value.value == 4
# end

# ift = labelize(@lambda IFTHENELSE*false)
# ift = beta_reduce(ift)
# @test ift === KI



# fact = YCOMBINATOR(FACT)
# fact5 = let fact5 = labelize(fact(5))
#     for _ in 1:30
#         display(fact5)
#         fact5 = beta_reduce(fact5)
#         if isa(fact5, ProxyDB)
#             break
#         end
#     end
#     fact5
# end
# display(fact5)
# @test fact5 == ProxyDB(120)


