import ..Trees: children, setchild!, arity, isleaf
import ..ASTs.NodeASTs: _show_plain, _show_pretty

"""A finite sequential scope whose bindings are ordered outermost to innermost."""
mutable struct ScopeNode <: AstNode
    bindings::Vector{AstNode}
    body::AstNode
end

ScopeNode(bindings::AbstractVector{<:AstNode}, body::AstNode) =
    ScopeNode(AstNode[bindings...], body)
ScopeNode(bindings::Tuple{Vararg{AstNode}}, body::AstNode) =
    ScopeNode(AstNode[bindings...], body)

"""A recursive function body with one anonymous self binder at De Bruijn index 0."""
mutable struct FunctionDefNode <: AstNode
    body::AstNode
end

children(node::ScopeNode) = AstNode[node.bindings..., node.body]
arity(node::ScopeNode) = length(node.bindings) + 1
isleaf(::ScopeNode) = false

function setchild!(node::ScopeNode, index::Int, child::AstNode)
    n = length(node.bindings)
    if 1 <= index <= n
        node.bindings[index] = child
    elseif index == n + 1
        node.body = child
    else
        throw(BoundsError(children(node), index))
    end
    return node
end

children(node::FunctionDefNode) = AstNode[node.body]
arity(::FunctionDefNode) = 1
isleaf(::FunctionDefNode) = false

function setchild!(node::FunctionDefNode, index::Int, child::AstNode)
    index == 1 || throw(BoundsError(children(node), index))
    node.body = child
    return node
end

Base.:(==)(left::ScopeNode, right::ScopeNode) =
    left.bindings == right.bindings && left.body == right.body
Base.:(==)(left::FunctionDefNode, right::FunctionDefNode) = left.body == right.body
Base.hash(node::ScopeNode, h::UInt) =
    hash(node.body, hash(node.bindings, hash(:ScopeNode, h)))
Base.hash(node::FunctionDefNode, h::UInt) = hash(node.body, hash(:FunctionDefNode, h))

_show_plain(node::ScopeNode) =
    "ScopeNode($(repr(node.bindings)), $(repr(node.body)))"
_show_plain(node::FunctionDefNode) = "FunctionDefNode($(repr(node.body)))"

const _PRETTY_FUNCTION_NAMES = Tuple(Symbol(char) for char in 'f':'z')
const _PRETTY_VALUE_NAMES = (
    :x, :y, :z,
    Tuple(Symbol(char) for char in 'a':'w')...,
)

_pretty_binder_name(names, depth::Int) = names[mod1(depth + 1, length(names))]

function _show_procedural_pretty(
    node::AstNode,
    precedence::Int,
    function_depth::Int,
    value_depth::Int,
)
    if node isa ScopeNode
        statements = String[]
        for (offset, binding) in enumerate(node.bindings)
            name = _pretty_binder_name(_PRETTY_VALUE_NAMES, value_depth + offset - 1)
            value = _show_procedural_pretty(
                binding,
                0,
                function_depth,
                value_depth + offset - 1,
            )
            push!(statements, "$name = $value")
        end
        push!(
            statements,
            _show_procedural_pretty(
                node.body,
                0,
                function_depth,
                value_depth + length(node.bindings),
            ),
        )
        result = "{ " * join(statements, "; ") * " }"
        return precedence > 1 ? "($result)" : result
    elseif node isa FunctionDefNode
        name = _pretty_binder_name(_PRETTY_FUNCTION_NAMES, function_depth)
        if node.body isa AbsNode
            argument = _pretty_binder_name(_PRETTY_VALUE_NAMES, value_depth)
            body = _show_procedural_pretty(
                node.body.body,
                1,
                function_depth + 1,
                value_depth + 1,
            )
            result = "fn $name($argument). $body"
        else
            body = _show_procedural_pretty(
                node.body,
                1,
                function_depth + 1,
                value_depth,
            )
            result = "fn $name. $body"
        end
        return precedence > 1 ? "($result)" : result
    elseif node isa AppNode
        func = _show_procedural_pretty(node.func, 2, function_depth, value_depth)
        arg = _show_procedural_pretty(node.arg, 3, function_depth, value_depth)
        result = "$func $arg"
        return precedence > 2 ? "($result)" : result
    elseif node isa AbsNode
        name = _pretty_binder_name(_PRETTY_VALUE_NAMES, value_depth)
        body = _show_procedural_pretty(
            node.body,
            1,
            function_depth,
            value_depth + 1,
        )
        result = "λ$name·$body"
        return precedence > 1 ? "($result)" : result
    elseif node isa LetNode
        name = _pretty_binder_name(_PRETTY_VALUE_NAMES, value_depth)
        value = _show_procedural_pretty(node.value, 0, function_depth, value_depth)
        body = _show_procedural_pretty(
            node.body,
            0,
            function_depth,
            value_depth + 1,
        )
        result = "let $name = $value in $body"
        return precedence > 1 ? "($result)" : result
    end
    return _show_pretty(node, precedence)
end

_show_pretty(node::ScopeNode, precedence::Int=0) =
    _show_procedural_pretty(node, precedence, 0, 0)

_show_pretty(node::FunctionDefNode, precedence::Int=0) =
    _show_procedural_pretty(node, precedence, 0, 0)
