using LambdaRegression.DeBruijnModule: to_debrujin
using LambdaRegression.ParseModule: @lambda
using LambdaRegression.ReduceModule: beta_reduce

ident1 = to_debrujin(@lambda (x->x)*1)
display(ident1)
display(beta_reduce(ident1))
