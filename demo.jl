using LambdaRegression: @lambda

@macroexpand @lambda x->x
@macroexpand @lambda (f, x)->f(x, x)

# y1 = @lambda v->f-> (x-> f(x * x)) * (x-> f(x * x))
# display(y1)

# db1 = @lambda &1 -> &1 * &1
# display(db1)

# y2 = @lambda &2-> (&1-> &2(&1 * &1)) * (&1-> &2(&1 * &1))
# display(y2)

# dbl = to_debrujin_level(y1)
# display(dbl)
# println(dbl)

# using LambdaRegression.DeBruijnIndexModule: _unfold