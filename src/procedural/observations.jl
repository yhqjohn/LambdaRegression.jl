function _add_self_observations(left, right)
    return (
        self_references=left.self_references + right.self_references,
        self_calls=left.self_calls + right.self_calls,
    )
end

_empty_self_observation() = (self_references=0, self_calls=0)

function _application_spine(node::AppNode)
    arguments = AstNode[]
    head = node::AstNode
    while head isa AppNode
        push!(arguments, head.arg)
        head = head.func
    end
    return head, arguments
end

function _self_observation(node::AstNode, self_index::Int)
    if node isa IndexNode
        return node.index == self_index ?
            (self_references=1, self_calls=0) :
            _empty_self_observation()
    elseif node isa AppNode
        head, arguments = _application_spine(node)
        total = if head isa IndexNode && head.index == self_index
            (self_references=1, self_calls=1)
        else
            _self_observation(head, self_index)
        end
        for argument in arguments
            total = _add_self_observations(
                total,
                _self_observation(argument, self_index),
            )
        end
        return total
    elseif node isa AbsNode
        return _self_observation(node.body, self_index + 1)
    elseif node isa LetNode
        return _add_self_observations(
            _self_observation(node.value, self_index),
            _self_observation(node.body, self_index + 1),
        )
    elseif node isa ScopeNode
        total = _empty_self_observation()
        for (offset, binding) in enumerate(node.bindings)
            total = _add_self_observations(
                total,
                _self_observation(binding, self_index + offset - 1),
            )
        end
        return _add_self_observations(
            total,
            _self_observation(node.body, self_index + length(node.bindings)),
        )
    elseif node isa FunctionDefNode
        return _self_observation(node.body, self_index + 1)
    end

    total = _empty_self_observation()
    for child in children(node)
        total = _add_self_observations(total, _self_observation(child, self_index))
    end
    return total
end

function _collect_functiondef_observations!(observations, node::AstNode)
    if node isa FunctionDefNode
        counts = _self_observation(node.body, 0)
        push!(observations, (
            body_signature=sprint(show, node.body),
            counts...,
        ))
    end
    for child in children(node)
        _collect_functiondef_observations!(observations, child)
    end
    return observations
end

"""
    functiondef_recursion_observations(node) -> Vector{NamedTuple}

Return one preorder observation per `FunctionDefNode`. `self_references` counts
uses of that definition's anonymous binder, while `self_calls` counts maximal
application spines headed by it. Merely returning or passing `self` is therefore
not reported as a recursive call.
"""
function functiondef_recursion_observations(node::AstNode)
    return _collect_functiondef_observations!(NamedTuple[], node)
end

function functiondef_recursion_observations(tree::AstTree)
    return functiondef_recursion_observations(tree.root)
end

"""Summarise genuine recursive calls across all `FunctionDefNode`s in a term."""
function functiondef_recursion_summary(term::Union{AstNode,AstTree})
    observations = functiondef_recursion_observations(term)
    return (
        definitions=length(observations),
        recursive_definitions=count(item -> item.self_calls > 0, observations),
        self_references=sum(item -> item.self_references, observations; init=0),
        self_calls=sum(item -> item.self_calls, observations; init=0),
    )
end
