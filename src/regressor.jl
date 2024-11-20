module RegressorModule

using ..TermsModule: TypedTerm, TypedTermProxy, TypedTermDB, ProxyDB, substitute, normalize
using ..DeBruijnModule: DBNode, normalized
using ..ReduceModule: beta_reduce

using ..BuiltinTermsModule

abstract type AbstractMutation{T} end
function mutate(mutation::AbstractMutation{T}, node::DBNode{T}) where T end

struct Regressor{T}
    terms::Vector{DBNode{T}}
    mutation::AbstractMutation{T}
    max_depth::Int
    population_size::Int 
    generations::Int
    tournament_size::Int
end

function Regressor{T}(;
    terms = [SIN, COS, TAN, EXP, LOG, SQRT, POW, ADD, SUB, MUL, DIV],
    mutation = default_mutation,
    max_depth = 6,
    population_size = 500,
    generations = 50,
    tournament_size = 7
) where T
    return Regressor{T}(
        [DBNode{T}(t) for t in terms],
        mutation,
        max_depth,
        population_size,
        generations,
        tournament_size
    )
end

function Regressor(;kwargs...)
    return Regressor{TypedTerm{Any}}(;kwargs...)
end







end
