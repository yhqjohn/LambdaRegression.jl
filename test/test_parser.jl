using LambdaRegression.ParseModule: @lambda, AbstractToken, AbstractVariableToken, VariableToken, DeBruijnToken, 
LambdaToken, MultiLambdaToken, ApplyToken, MultiApplyToken, ProxyToken, unfold


id1 = @lambda x->x
id2 = LambdaToken(
    VariableToken(:x),
    VariableToken(:x)
)
@test id1 == id2

# println(@macroexpand @lambda (f, x)->f(x, x))
two = @lambda (f, x)->f(x, x)
two2 = MultiLambdaToken(
    (
        VariableToken(:f),
        VariableToken(:x)
    ),
    MultiApplyToken(
        VariableToken(:f),
        (
            VariableToken(:x),
            VariableToken(:x)
        )
    )
)
@test two == two2

two3 = @lambda f->x->(f*x*x)
# display(unfold(two))
@test two3 == unfold(two)

y_combinator = @lambda f->(x->f(x(x)))(x->f(x(x)))
y_combinator2 = @lambda f->((x->f*(x*x)) * (x->f*(x*x)))
@test unfold(y_combinator) == y_combinator2

two3 = @lambda (&1, &2)->&1(&2, &2)
two4 = @lambda &1->&2->&1*&2*&2
@test unfold(two3) == two4


