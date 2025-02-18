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

function _functiondef_obvious_loop_reason(node::FunctionDefNode)
    body = node.body
    body isa IndexNode && body.index == 0 && return :direct_self

    lambda_count = 0
    current = body
    while current isa AbsNode
        lambda_count += 1
        current = current.body
    end
    if lambda_count > 0 && current isa AppNode
        head, reverse_arguments = _application_spine(current)
        if head isa IndexNode && head.index == lambda_count &&
           length(reverse_arguments) == lambda_count &&
           all(enumerate(reverse_arguments)) do (position, argument)
               argument isa IndexNode && argument.index == position - 1
           end
            return :eta_self_forwarding
        end
        head isa IndexNode && head.index == lambda_count &&
            !isempty(reverse_arguments) && return :unconditional_self_call
    end
    _functiondef_unchanged_self_call(node) && return :unchanged_self_call
    return nothing
end

function _has_unchanged_self_call_node(
    node::AstNode,
    self_index::Int,
    parameter_indices::Vector{Int},
)
    if node isa AppNode
        head, reverse_arguments = _application_spine(node)
        if head isa IndexNode && head.index == self_index &&
           length(reverse_arguments) == length(parameter_indices) &&
           all(zip(reverse_arguments, parameter_indices)) do (argument, expected)
               argument isa IndexNode && argument.index == expected
           end
            return true
        end
        return _has_unchanged_self_call_node(
            node.func,
            self_index,
            parameter_indices,
        ) || _has_unchanged_self_call_node(
            node.arg,
            self_index,
            parameter_indices,
        )
    elseif node isa AbsNode
        return false
    elseif node isa LetNode
        return _has_unchanged_self_call_node(
            node.value,
            self_index,
            parameter_indices,
        ) || _has_unchanged_self_call_node(
            node.body,
            self_index + 1,
            parameter_indices .+ 1,
        )
    elseif node isa ScopeNode
        for (offset, binding) in enumerate(node.bindings)
            _has_unchanged_self_call_node(
                binding,
                self_index + offset - 1,
                parameter_indices .+ (offset - 1),
            ) && return true
        end
        return _has_unchanged_self_call_node(
            node.body,
            self_index + length(node.bindings),
            parameter_indices .+ length(node.bindings),
        )
    elseif node isa FunctionDefNode
        return false
    end
    return any(children(node)) do child
        _has_unchanged_self_call_node(child, self_index, parameter_indices)
    end
end

function _functiondef_unchanged_self_call(node::FunctionDefNode)
    current = node.body
    lambda_count = 0
    while current isa AbsNode
        lambda_count += 1
        current = current.body
    end
    lambda_count > 0 || return false
    return _has_unchanged_self_call_node(
        current,
        lambda_count,
        collect(0:(lambda_count - 1)),
    )
end

function _obvious_functiondef_loop_reason(node::AstNode)
    if node isa FunctionDefNode
        reason = _functiondef_obvious_loop_reason(node)
        reason === nothing || return reason
    end
    for child in children(node)
        reason = _obvious_functiondef_loop_reason(child)
        reason === nothing || return reason
    end
    return nothing
end

"""
    obvious_functiondef_loop_reason(term) -> Union{Nothing,Symbol}

Return a stable reason for the deliberately small set of syntactically certain
FunctionDef loops recognized after arbitrary variation: `:direct_self` and
finite `:eta_self_forwarding`, `:unconditional_self_call` for a body that is
only a self call after its lambda prefix, plus `:unchanged_self_call` when a
recursive call reuses every function argument unchanged in a strict expression
context outside nested lambda or FunctionDef values.
`nothing` means only that this cheap filter found no such shape; it is not a
termination proof.
"""
obvious_functiondef_loop_reason(node::AstNode) =
    _obvious_functiondef_loop_reason(node)
obvious_functiondef_loop_reason(tree::AstTree) =
    obvious_functiondef_loop_reason(tree.root)
