using LambdaRegression.DeBruijnModule: to_debrujin, substitute
using LambdaRegression.ParseModule: @lambda

y_combinator = @lambda f->(x->f(x(x)))(x->f(x(x)))
y_combinator2 = @lambda (*)->((*)->&1(&0*(&0))) * ((*)->&1(&0*(&0)))
y_db = to_debrujin(y_combinator)
y_db2 = to_debrujin(y_combinator2)
display(y_db2)
println("yes")
@test y_db == y_db2

