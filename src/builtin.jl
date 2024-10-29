module BuiltInTermsModule

import ..LabeledProxyModule: FunctionTerm

const SIN = FunctionTerm(sin, :Float64 >> :Float64, "sin")
const COS = FunctionTerm(cos, :Float64 >> :Float64, "cos")
const TAN = FunctionTerm(tan, :Float64 >> :Float64, "tan")
const ADD = FunctionTerm(+, :Float64 >> :Float64 >> :Float64, "+")
const SUB = FunctionTerm(-, :Float64 >> :Float64 >> :Float64, "-")
const MUL = FunctionTerm(*, :Float64 >> :Float64 >> :Float64, "*")
const DIV = FunctionTerm(/, :Float64 >> :Float64 >> :Float64, "/")
const EQ = FunctionTerm(==, :Float64 >> :Float64 >> :Bool, "==")
    
end