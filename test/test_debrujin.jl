using LambdaRegression.DeBruijnModule
using LambdaRegression.DeBruijnModule: shift_free, substitute
using LambdaRegression.ParseModule: @lambda

y_combinator = @lambda f->(x->f(x(x)))(x->f(x(x)))
y_combinator2 = @lambda (*)->((*)->&1(&0*(&0))) * ((*)->&1(&0*(&0)))
y_db = DBNode(y_combinator)
y_db2 = DBNode(y_combinator2)
display(y_db2)
@test y_db == y_db2

# (λ λ 3 1 (λ 0 2)) (λ 4 0)
foo = @lambda (*)->(*)->(&3 * &1 * ((*)->&0 * &2))
bar = @lambda (*)->(&4 * &0)
foo = DBNode(foo)
bar = DBNode(bar)
display(foo)
display(bar)

## test traverse
new_foo = traverse(foo) do x, n
    if x isa VarDB
        VarDB(x.index + 1)
    else
        nothing
    end
end
@test new_foo == DBNode(@lambda (*)->(*)->(&4 * &2 * ((*)->&1 * &3)))

new_foo2 = traverse(new_foo) do x, n
    if x isa VarDB
        ProxyDB(Int(x.index))
    else
        nothing
    end
end
@test new_foo2 == DBNode(@lambda (*)->(*)->(4 * 2 * ((*)->1 * 3)))

# test lift free variables
new_foo3 = shift_free(foo, UInt(1))
@test new_foo3 == DBNode(@lambda (*)->(*)->(&4 * &1 * ((*)->&0 * &2)))

# test substitute
foobar = substitute(foo, bar)
display(foobar)
_foobar = DBNode(@lambda (*)->(&2 * ((*)->(&5 * &0)) * ((*)->&0 * (*)->(&6 * &0)))) # λ.(2 λ.(5 0) λ.(0 λ.(6 0)))
@test foobar == _foobar
