struct DeBrujinVariable <: AbstractAstNode
    index::UInt
end
childterms(node::DeBrujinVariable) = []

struct DeBrujinAbstraction <: AbstractAstNode
    body
end
childterms(node::DeBrujinAbstraction) = [node.body]
remake(node::DeBrujinAbstraction, body) = DeBrujinAbstraction(body)

abstraction_depth(node::AbstractAstNode) = isleaf(node) ? UInt(0) : maximum(abstraction_depth(child) for child in children(node))
abstraction_depth(node::DeBrujinAbstraction) = 1 + abstraction_depth(node.body)


to_debrujin(node::AbstractAstNode) = to_debrujin(node, VariableNode[])
function to_debrujin(node::AbstractAstNode, context::Vector{VariableNode})
    if node isa VariableNode
        index = findfirst(context.===node)
        if isnothing(index)
            return node # free variable
        else 
            return DeBrujinVariable(index-1)
        end
    elseif isleaf(node)
        return node
    elseif node isa LambdaNode
        new_context = [node.arg; context]
        return DeBrujinAbstraction(to_debrujin(node.body, new_context))
    else
        return remake(node, [to_debrujin(child, context) for child in childterms(node)]...)
    end
end


pretty_string(x) = string(x)
function pretty_string(node::AbstractAstNode, style) # always call this after the ast has been regularized to De Brujin indices.
    if isleaf(node)
        return string(node)
    else
        return "$(type(node))($(join([pretty_string(child, style) for child in children(node)], ", ")))"
    end
end

function pretty_string(node::DeBrujinVariable, style::Symbol=:expandindex)
    if style===:index || style===:expandindex
        return "#$(node.index)"
    elseif style===:alphabetic
        return to_letter(node.index)
    else
        throw(ArgumentError("style must be :index, :expandindex, or :alphabetic"))
    end
end

function pretty_string(node::DeBrujinAbstraction, style::Symbol=:expandindex)
    if style===:index
        return "λ.$(pretty_string(node.body, style))"
    elseif style===:expandindex || style===:alphabetic
        index = abstraction_depth(node)
        if style===:expandindex
            "λ$index.$(pretty_string(node.body, style))"
        else
            if node.body isa DeBrujinAbstraction
                "λ$(to_letter(index))$(pretty_string(node.body, style))"
            else
                "λ$(to_letter(index)).$(pretty_string(node.body, style))"
            end
        end
    else
        throw(ArgumentError("style must be :index, :expandindex, or :alphabetic"))
    end
end

repr_letters = "zyxwvutsrqpnmlkjihgfedcba" # skip o to avoid confusion with 0
function to_letter(i::UInt)
    if i<=24
        repr_letters[i+1]
    else
        "x_$(i-24)"
    end
end

Base.show(io::IO, ::MIME"text/plain", node::AbstractAstNode) = print(io, pretty_string(to_debrujin(node), :alphabetic))
Base.show(io::IO, ::MIME"text/plain", node::DeBrujinAbstraction) = print(io, pretty_string(node, :index))