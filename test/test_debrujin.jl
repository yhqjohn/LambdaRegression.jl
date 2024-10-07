using LambdaRegression.DeBruijnIndexModule: to_derbujin_index, set_alphabetic!
using LambdaRegression.ParseModule: @lambda

y_combinator = @lambda f->(x->f(x(x)))(x->f(x(x)))
y_combinator2 = @lambda &1->(&0->&1(&0*&0)) * (&0->&1(&0*&0))
y_db = to_derbujin_index(y_combinator)
y_db2 = to_derbujin_index(y_combinator2)
# set_alphabetic!(false)
display(y_db)
@test y_db == y_db2