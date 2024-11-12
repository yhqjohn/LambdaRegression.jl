using LambdaRegression.DeBruijnModule
using LambdaRegression.ParseModule
using LambdaRegression.TermsModule: typelabel, LabelType, -->, argtype, rettype

# test type system
@test typelabel(1) == LabelType(:Int)
@test typelabel(1.0) == LabelType(:Float64)
@test typelabel(true) == LabelType(:Bool)
type_ieq = :Int --> :Int --> :Bool
@test argtype(type_ieq) == LabelType(:Int)
@test rettype(type_ieq) == (:Int --> :Bool)
@test type_ieq |> rettype |> argtype == LabelType(:Int)
@test type_ieq |> rettype |> rettype == LabelType(:Bool)
