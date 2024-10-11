using LambdaRegression.DeBruijnModule: to_debrujin, ProxyDB
using LambdaRegression.ParseModule: @lambda
using LambdaRegression.ReduceModule: beta_reduce, normalized

ident1 = to_debrujin(@lambda (x->x)*1)
display(ident1)
display(beta_reduce(ident1))
@test beta_reduce(ident1) == ProxyDB(1)
@test beta_reduce(beta_reduce(ident1)) == normalized
