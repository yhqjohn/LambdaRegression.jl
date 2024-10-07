module LambdaRegression

# abstract type AbstractAstNode end
# function childterms end 
# function children end # default implementation is childterms
# function isleaf end # default implementation is length(children(node))==0
# function beta_reduce end
# function apply end
# function remake end

# mutable struct VariableNode <: AbstractAstNode end

# struct LambdaNode <: AbstractAstNode
#     arg::VariableNode
#     body::AbstractAstNode
# end

# struct ApplyNode <: AbstractAstNode
#     func::AbstractAstNode
#     arg::AbstractAstNode
# end


# # childterms
# childterms(node::VariableNode) = []
# childterms(node::LambdaNode) = [node.body]
# childterms(node::ApplyNode) = [node.func, node.arg]

# # children
# children(node::AbstractAstNode) = childterms(node)
# children(node::LambdaNode) = [node.arg, node.body]


# # isleaf
# isleaf(node::VariableNode) = true
# isleaf(node::LambdaNode) = false
# isleaf(node::ApplyNode) = false
# isleaf(node::AbstractAstNode) = length(children(node))==0

# # beta_reduce
# beta_reduce(node::AbstractAstNode) = node

# function beta_reduce(node::ApplyNode)
#     result = apply(node.func, node.arg)
#     if result isa AbstractAstNode
#         return result
#     elseif result isa NormalizedError
#         return node
#     else
#         throw(ArgumentError("substitute must return a node or a NormalizedError"))
#     end
# end


# # apply
# struct NormalizedError end
# apply(node::AbstractAstNode, arg::AbstractAstNode) =  NormalizedError()
# apply(func::LambdaNode, arg::AbstractAstNode) = substitute(func.body, func.arg, arg)
# substitute(term::VariableNode, arg::VariableNode, replacement::AbstractAstNode) = term===arg ? replacement : term
# function substitute(term::AbstractAstNode, arg::VariableNode, replacement::AbstractAstNode)
#     if isleaf(term)
#         return term
#     else
#         return remake(term, [substitute(child, arg, replacement) for child in childterms(term)]...)
#     end
# end


# # remake
# remake(node::LambdaNode, term::AbstractAstNode) = LambdaNode(node.arg, term)
# remake(node::ApplyNode, term1::AbstractAstNode, term2::AbstractAstNode) = ApplyNode(term1, term2)


# function apply_leaves(f, node::AbstractAstNode)
#     if isleaf(node)
#         return f(node)
#     else
#         children_ = children(node)
#         new_children = []
#         for child in children_
#             new_child = apply_leaves(f, child)
#             push!(new_children, new_child)
#         end
#         # new_children = [apply_leaves(f, child) for child in children_]
#         return remake(node, new_children...)
#     end
# end

include("common_utils.jl")

include("parse.jl")

using .ParseModule: @lambda

include("debrujinindex.jl")
using .DeBruijnModule: to_derbujin

include("reduce.jl")


end # module LambdaRegression
