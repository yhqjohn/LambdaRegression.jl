"""
    generate_term(rng, language, targetid; context=Int32[], max_nodes=31,
                  max_depth=6, max_attempts=256, mode=:grow,
                  soft_depth=max_depth)

Small, deliberately local typed growth for the compact `TypedSearch` terms.
This is a bounded sampler, not the canonical finite-generation decision
procedure.  It only uses concrete leaves and visible monomorphic locals as
heads, and reports a `TypedSearchError` when its finite budget is exhausted.
The `propose_term` dispatcher also supports opt-in parent-preserving
`kind=:insert`; its retained occurrence and generated siblings share the
same node/depth budget and anchor-mapping checks.
The opt-in `kind=:operator` replacement changes one applied atomic
`LEAF`/`VAR` head, chosen uniformly among distinct function-position
occurrences, to a same-`typeid` language leaf or visible local.  It never
rewrites application arguments or performs binding remapping.
"""

using Random

export generate_term, propose_term

mutable struct _GrowthBudget
    attempts::Int
    max_nodes::Int
    max_depth::Int
end

@inline function _growth_mode_ok(mode::Symbol)
    mode === :grow || mode === :branch_first
end

@inline function _growth_attempt!(budget::_GrowthBudget)
    budget.attempts > 0 || return false
    budget.attempts -= 1
    return true
end

@inline function _growth_fail(reason::Symbol, detail=nothing)
    return Err(TypedSearchError(reason, detail))
end

@inline function _growth_push!(nodes::Vector{TermNode}, budget::_GrowthBudget,
                               node::TermNode)
    length(nodes) < budget.max_nodes || return nothing
    push!(nodes, node)
    return Int32(length(nodes))
end

function _growth_valid_args(language::Language, targetid::Integer,
                            context::AbstractVector{<:Integer}, max_nodes::Integer,
                            max_depth::Integer, max_attempts::Integer)
    max_nodes >= 1 || return _growth_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _growth_fail(:invalid_depth_budget, max_depth)
    max_attempts >= 1 || return _growth_fail(:invalid_attempt_budget, max_attempts)
    _valid_typeid(language.types, targetid) ||
        return _growth_fail(:unknown_target_type, targetid)
    checked = _check_context(language.types, context)
    checked isa Err && return checked
    return (Int32(targetid), checked)
end

function _growth_var_candidates(context::Vector{Int32}, targetid::Int32)
    result = Int32[]
    for i in eachindex(context)
        context[i] == targetid || continue
        # Context is outer-to-inner, while De Bruijn zero is nearest.
        push!(result, Int32(length(context) - i))
    end
    return result
end

function _growth_leaf_candidates(language::Language, targetid::Int32)
    result = Int32[]
    for i in eachindex(language.leaves)
        leaf = language.leaves[i]
        leaf.typeid == targetid || continue
        _valid_typeid(language.types, leaf.typeid) || continue
        push!(result, Int32(i))
    end
    return result
end

"""Return `(kind, index, argument-type-ids)` heads reaching `targetid`."""
function _growth_heads(language::Language, context::Vector{Int32}, targetid::Int32;
                       max_args::Integer=typemax(Int))
    max_args >= 0 || throw(ArgumentError("max_args must be nonnegative"))
    table = language.types
    heads = Tuple{UInt8,Int32,Vector{Int32}}[]

    for i in eachindex(context)
        db = Int32(length(context) - i)
        headid = context[length(context) - Int(db)]
        args = Int32[]
        current = headid
        while length(args) < max_args
            parts = _arrow_parts(table, current)
            parts === nothing && break
            argid, resid = parts
            push!(args, argid)
            current = resid
            current == targetid && push!(heads, (VAR_KIND, db, copy(args)))
        end
    end

    for leaf_index in eachindex(language.leaves)
        _valid_typeid(table, language.leaves[leaf_index].typeid) || continue
        headid = language.leaves[Int(leaf_index)].typeid
        args = Int32[]
        current = headid
        while length(args) < max_args
            parts = _arrow_parts(table, current)
            parts === nothing && break
            argid, resid = parts
            push!(args, argid)
            current = resid
            current == targetid && push!(heads, (LEAF_KIND, leaf_index, copy(args)))
        end
    end
    return heads
end

function _growth_subtree_depth(nodes::Vector{TermNode}, i::Int)
    node = nodes[i]
    if node.kind == VAR_KIND || node.kind == LEAF_KIND
        return 0
    elseif node.kind == ABS_KIND
        return 1 + _growth_subtree_depth(nodes, Int(node.x))
    elseif node.kind == APP_KIND
        return 1 + max(_growth_subtree_depth(nodes, Int(node.x)),
                       _growth_subtree_depth(nodes, Int(node.y)))
    end
    return typemax(Int)
end

function _growth_site_depth(nodes::Vector{TermNode}, i::Int, site::Int, depth::Int)
    i == site && return depth
    node = nodes[i]
    if node.kind == ABS_KIND
        return _growth_site_depth(nodes, Int(node.x), site, depth + 1)
    elseif node.kind == APP_KIND
        left = _growth_site_depth(nodes, Int(node.x), site, depth + 1)
        left !== nothing && return left
        return _growth_site_depth(nodes, Int(node.y), site, depth + 1)
    end
    return nothing
end


@inline function _growth_accept(nodes::Vector{TermNode}, root::Int32,
                                depth::Int, budget::_GrowthBudget)
    depth + _growth_subtree_depth(nodes, Int(root)) <= budget.max_depth || return nothing
    return root
end

function _growth_gen!(rng, nodes::Vector{TermNode}, language::Language,
                      targetid::Int32, context::Vector{Int32}, depth::Int,
                      budget::_GrowthBudget; mode::Symbol=:grow,
                      soft_depth::Int=budget.max_depth)
    mark = length(nodes)
    depth <= budget.max_depth || return nothing
    _growth_attempt!(budget) || return nothing

    terminal_vars = _growth_var_candidates(context, targetid)
    terminal_leaves = _growth_leaf_candidates(language, targetid)
    target_parts = _arrow_parts(language.types, targetid)
    heads = depth < budget.max_depth ? _growth_heads(language, context, targetid) :
            Tuple{UInt8,Int32,Vector{Int32}}[]

    choices = Symbol[]
    terminals = Symbol[]
    branches = Symbol[]
    !isempty(terminal_vars) && push!(terminals, :var)
    !isempty(terminal_leaves) && push!(terminals, :leaf)
    target_parts !== nothing && depth < budget.max_depth && push!(branches, :abs)
    !isempty(heads) && depth < budget.max_depth && push!(branches, :app)
    if mode === :grow
        append!(choices, terminals)
        append!(choices, branches)
        Random.shuffle!(rng, choices)
    elseif mode === :branch_first
        # This is a preference, not a completeness rule.  A branch may still
        # fail and fall through to a terminal (or vice versa) under the same
        # finite budget.  Keep the two groups shuffled so equal alternatives
        # do not acquire a fixed language ordering.
        Random.shuffle!(rng, terminals)
        Random.shuffle!(rng, branches)
        if depth < min(soft_depth, budget.max_depth)
            append!(choices, branches)
            append!(choices, terminals)
        else
            append!(choices, terminals)
            append!(choices, branches)
        end
    else
        return nothing
    end
    isempty(choices) && return nothing

    for choice in choices
        resize!(nodes, mark)
        if choice === :var
            Random.shuffle!(rng, terminal_vars)
            idx = _growth_push!(nodes, budget, var_node(targetid, terminal_vars[1]))
            if idx !== nothing
                accepted = _growth_accept(nodes, idx, depth, budget)
                accepted !== nothing && return accepted
            end
        elseif choice === :leaf
            Random.shuffle!(rng, terminal_leaves)
            idx = _growth_push!(nodes, budget, leaf_node(targetid, terminal_leaves[1]))
            if idx !== nothing
                accepted = _growth_accept(nodes, idx, depth, budget)
                accepted !== nothing && return accepted
            end
        elseif choice === :abs
            argid, resid = target_parts
            body = _growth_gen!(rng, nodes, language, resid,
                                vcat(context, Int32[argid]), depth + 1, budget;
                                mode=mode, soft_depth=soft_depth)
            if body !== nothing
                idx = _growth_push!(nodes, budget, abs_node(targetid, body))
                if idx !== nothing
                    accepted = _growth_accept(nodes, idx, depth, budget)
                    accepted !== nothing && return accepted
                end
            end
        else
            Random.shuffle!(rng, heads)
            for (kind, head_index, argids) in heads
                _growth_attempt!(budget) || break
                candidate_mark = length(nodes)
                head = if kind == VAR_KIND
                    _growth_push!(nodes, budget, var_node(context[length(context) - Int(head_index)],
                                                         head_index))
                else
                    _growth_push!(nodes, budget, leaf_node(language.leaves[Int(head_index)].typeid,
                                                           head_index))
                end
                head === nothing && (resize!(nodes, candidate_mark); continue)
                current = head
                current_type = kind == VAR_KIND ?
                    context[length(context) - Int(head_index)] :
                    language.leaves[Int(head_index)].typeid
                success = true
                for (position, argid) in enumerate(argids)
                    parts = _arrow_parts(language.types, current_type)
                    if parts === nothing || parts[1] != argid
                        success = false
                        break
                    end
                    # The first argument is nested below the complete
                    # curried spine, while the last is below one APP node.
                    arg_depth = depth + length(argids) - position + 1
                    remaining_apps = length(argids) - position + 1
                    old_node_limit = budget.max_nodes
                    budget.max_nodes = old_node_limit - remaining_apps
                    arg = try
                        budget.max_nodes >= length(nodes) ?
                        _growth_gen!(rng, nodes, language, argid, context,
                                         arg_depth, budget;
                                         mode=mode, soft_depth=soft_depth) : nothing
                    finally
                        budget.max_nodes = old_node_limit
                    end
                    if arg === nothing
                        success = false
                        break
                    end
                    current_type = parts[2]
                    app = _growth_push!(nodes, budget, app_node(current_type, current, arg))
                    if app === nothing
                        success = false
                        break
                    end
                    current = app
                end
                if success
                    accepted = _growth_accept(nodes, current, depth, budget)
                    accepted !== nothing && return accepted
                end
                resize!(nodes, candidate_mark)
            end
        end
    end
    resize!(nodes, mark)
    return nothing
end

"""Generate one bounded, exactly typed compact term or return `Err`."""
function generate_term(rng, language::Language, targetid::Integer;
                       context::AbstractVector{<:Integer}=Int32[], max_nodes::Integer=31,
                       max_depth::Integer=6, max_attempts::Integer=256,
                       mode::Symbol=:grow, soft_depth::Integer=max_depth)
    _growth_mode_ok(mode) || return _growth_fail(:invalid_growth_mode, mode)
    soft_depth >= 0 || return _growth_fail(:invalid_soft_depth, soft_depth)
    checked = _growth_valid_args(language, targetid, context, max_nodes, max_depth, max_attempts)
    checked isa Err && return checked
    target, ctx = checked
    nodes = TermNode[]
    budget = _GrowthBudget(Int(max_attempts), Int(max_nodes), Int(max_depth))
    root = _growth_gen!(rng, nodes, language, target, ctx, 0, budget;
                        mode=mode, soft_depth=Int(soft_depth))
    root === nothing && return _growth_fail(:generation_failed,
        (target, budget.attempts, max_nodes, max_depth))
    term = TypedTerm(nodes)
    bounded = _growth_within_budget(term, language, ctx, max_nodes, max_depth)
    bounded === true || return _growth_fail(:internal_invalid_generation, bounded)
    return term
end

function _growth_max_depth(nodes::Vector{TermNode}, i::Int, memo::Vector{Int})
    memo[i] != 0 && return memo[i]
    node = nodes[i]
    value = if node.kind == VAR_KIND || node.kind == LEAF_KIND
        0
    elseif node.kind == ABS_KIND
        1 + _growth_max_depth(nodes, Int(node.x), memo)
    elseif node.kind == APP_KIND
        1 + max(_growth_max_depth(nodes, Int(node.x), memo),
                _growth_max_depth(nodes, Int(node.y), memo))
    else
        -1
    end
    memo[i] = value
    return value
end

function _growth_within_budget(term::TypedTerm, language::Language,
                               context::AbstractVector{<:Integer}, max_nodes::Integer,
                               max_depth::Integer)
    length(term.nodes) <= max_nodes || return _growth_fail(:node_budget_exceeded, length(term.nodes))
    checked = validate(term, language; context=context)
    checked === true || return checked
    memo = zeros(Int, length(term.nodes))
    depth = _growth_max_depth(term.nodes, length(term.nodes), memo)
    depth <= max_depth || return _growth_fail(:depth_budget_exceeded, depth)
    return true
end

function _growth_mapping(rng, residual::AbstractVector{<:ResidualVar},
                         target_context::Vector{Int32})
    mapping = Dict{Int32,Int32}()
    for variable in residual
        candidates = Int32[]
        for i in eachindex(target_context)
            target_context[i] == variable.typeid || continue
            push!(candidates, Int32(length(target_context) - i))
        end
        isempty(candidates) && return nothing
        mapping[variable.db] = rand(rng, candidates)
    end
    return mapping
end

function _growth_identity_mapping(residual::AbstractVector{<:ResidualVar},
                                  target_context::Vector{Int32})
    mapping = Dict{Int32,Int32}()
    for variable in residual
        0 <= Int(variable.db) < length(target_context) || return nothing
        target_context[length(target_context) - Int(variable.db)] == variable.typeid ||
            return nothing
        mapping[variable.db] = variable.db
    end
    return mapping
end

function _growth_replace(rng, parent::TypedTerm, site::Int, donor::TypedTerm,
                         donor_site::Int, language::Language, parent_context,
                         donor_context, max_nodes, max_depth; random_mapping=true,
                         prepared_parent=nothing, prepared_donor=nothing)
    target_info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_context) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    target_info isa Err && return target_info
    donor_info = prepared_donor === nothing ?
        site_info(donor, donor_site, language; context=donor_context) :
        _prepared_site_info(prepared_donor, donor_site; term=donor, language=language)
    donor_info isa Err && return donor_info
    target_span = length(target_info.span)
    donor_span = length(donor_info.span)
    expected_nodes = length(parent.nodes) - target_span + donor_span
    expected_nodes <= max_nodes ||
        return _growth_fail(:node_budget_exceeded, expected_nodes)
    mapping = random_mapping ? _growth_mapping(rng, donor_info.residual, target_info.context) :
              _growth_identity_mapping(donor_info.residual, target_info.context)
    mapping === nothing && return _growth_fail(:no_visible_variable_mapping,
        (site, donor_site))
    child = replace_subterm(parent, site, donor, donor_site, language;
                            context=parent_context, donor_context=donor_context,
                            mapping=mapping)
    child isa Err && return child
    bounded = _growth_within_budget(child, language, parent_context, max_nodes, max_depth)
    bounded === true || return bounded
    return child
end

"""Bounded parent-preserving insertion. The selected occurrence is retained as
one exact-typed argument of a finite visible function-head prefix; siblings use
the site's context and a temporary node/depth budget that reserves the old
subtree, future applications, and minimum sibling nodes. Requires `target_info`
from this parent/site under `parent_context`. Identity mapping preserves its
residual locals; it does not search for equivalent environments. Returns `Err`
when no compatible head exists or construction exceeds its finite budgets."""
function _growth_insert(rng, parent::TypedTerm, site::Int, language::Language,
                        parent_context::Vector{Int32}, target_info::SiteInfo,
                        max_nodes::Integer, max_depth::Integer;
                        max_attempts::Integer=max(1, 4 * max_nodes),
                        sibling_max_nodes::Integer=max_nodes,
                        sibling_max_depth::Integer=max_depth,
                        prefer_terminal::Bool=false,
                        sibling_mode::Symbol=:grow,
                        prepared=nothing)
    max_nodes = Int(max_nodes)
    max_depth = Int(max_depth)
    max_attempts = Int(max_attempts)
    sibling_max_nodes = Int(sibling_max_nodes)
    sibling_max_depth = Int(sibling_max_depth)
    max_nodes >= 1 || return _growth_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _growth_fail(:invalid_depth_budget, max_depth)
    max_attempts >= 1 || return _growth_fail(:invalid_attempt_budget, max_attempts)
    sibling_max_nodes >= 1 || return _growth_fail(:invalid_sibling_node_budget,
        sibling_max_nodes)
    sibling_max_depth >= 0 || return _growth_fail(:invalid_sibling_depth_budget,
        sibling_max_depth)
    _growth_mode_ok(sibling_mode) || return _growth_fail(:invalid_growth_mode,
        sibling_mode)
    site_depth = _growth_site_depth(parent.nodes, length(parent.nodes), site, 0)
    site_depth === nothing && return _growth_fail(:unreachable_site, site)
    old_depth = _growth_subtree_depth(parent.nodes, site)
    site_depth + old_depth <= max_depth ||
        return _growth_fail(:depth_budget_exceeded, site_depth + old_depth)

    old_span = length(target_info.span)
    wrapper_capacity = max_nodes - length(parent.nodes) + old_span
    depth_limit = max_depth - site_depth
    depth_limit >= 1 || return _growth_fail(:depth_budget_exceeded, site_depth + 1)
    node_limit = wrapper_capacity > old_span ? fld(wrapper_capacity - old_span, 2) : 0
    max_arity = min(depth_limit, node_limit)
    max_arity >= 1 || return _growth_fail(:node_budget_exceeded,
        (old_span + 2, wrapper_capacity))
    heads = filter(h -> any(==(target_info.typeid), h[3]),
        _growth_heads(language, target_info.context, target_info.typeid;
            max_args=max_arity))
    isempty(heads) && return _growth_fail(:no_compatible_insert_head,
        target_info.typeid)
    head_kind, head_index, argids = rand(rng, heads)
    retained_positions = findall(==(target_info.typeid), argids)
    isempty(retained_positions) &&
        return _growth_fail(:no_retained_insert_argument, target_info.typeid)
    retained_position = rand(rng, retained_positions)
    retained_depth = site_depth + length(argids) - retained_position + 1 + old_depth
    retained_depth <= max_depth ||
        return _growth_fail(:depth_budget_exceeded, retained_depth)
    required = old_span + 2 * length(argids)
    wrapper_capacity >= required ||
        return _growth_fail(:node_budget_exceeded, (required, wrapper_capacity))

    parent_spans = prepared === nothing ? _spans(parent.nodes) :
        Int[Int(prepared.span_end[i] - prepared.span_start[i] + 1)
            for i in eachindex(prepared.span_end)]
    parent_spans === nothing && return _growth_fail(:bad_layout)
    mapping = _growth_identity_mapping(target_info.residual, target_info.context)
    mapping === nothing && return _growth_fail(:insert_anchor_mapping, site)
    budget = _GrowthBudget(max_attempts, wrapper_capacity, max_depth)
    wrapper = TermNode[]
    head_type = head_kind == VAR_KIND ?
        target_info.context[length(target_info.context) - Int(head_index)] :
        language.leaves[Int(head_index)].typeid
    head = head_kind == VAR_KIND ? var_node(head_type, head_index) :
        leaf_node(head_type, head_index)
    current = _growth_push!(wrapper, budget, head)
    current === nothing && return _growth_fail(:node_budget_exceeded, required)
    current_type = head_type
    for (position, argid) in enumerate(argids)
        parts = _arrow_parts(language.types, current_type)
        (parts !== nothing && parts[1] == argid) ||
            return _growth_fail(:insert_head_mismatch, (head_type, position, argid))
        future_apps = length(argids) - position
        future_siblings = count(p -> p > position && p != retained_position,
            1:length(argids))
        future_reserve = future_apps + future_siblings +
            (retained_position > position ? old_span : 0)
        arg = if position == retained_position
            length(wrapper) + old_span + 1 + future_reserve <= wrapper_capacity ||
                return _growth_fail(:node_budget_exceeded,
                    (length(wrapper) + old_span + 1 + future_reserve, wrapper_capacity))
            _copy_subtree!(wrapper, parent.nodes, site, 0, target_info.context,
                target_info.context, mapping, language, parent_spans)
        else
            sibling_limit = min(sibling_max_nodes,
                wrapper_capacity - length(wrapper) - 1 - future_reserve)
            sibling_limit >= 1 || return _growth_fail(:node_budget_exceeded,
                (length(wrapper) + 1 + future_reserve, wrapper_capacity))
            generation_depth = site_depth + length(argids) - position + 1
            generation_depth <= max_depth ||
                return _growth_fail(:depth_budget_exceeded, generation_depth)
            old_limit = budget.max_nodes
            old_depth_limit = budget.max_depth
            budget.max_nodes = min(old_limit, length(wrapper) + sibling_limit)
            # The small sibling cap is relative to the sibling root.  Keep
            # the complete child bound separately: the wrapper's applications
            # may sit below the selected parent site and consume additional
            # structural depth.
            budget.max_depth = min(old_depth_limit,
                generation_depth + sibling_max_depth)
            try
                if prefer_terminal
                    terminal_vars = _growth_var_candidates(target_info.context, argid)
                    terminal_leaves = _growth_leaf_candidates(language, argid)
                    terminal_sources = Tuple{UInt8,Int32}[]
                    append!(terminal_sources,
                        ((VAR_KIND, db) for db in terminal_vars))
                    append!(terminal_sources,
                        ((LEAF_KIND, index) for index in terminal_leaves))
                    if !isempty(terminal_sources)
                        source_kind, source_index = rand(rng, terminal_sources)
                        arg = source_kind == VAR_KIND ?
                            _growth_push!(wrapper, budget,
                                var_node(argid, source_index)) :
                            _growth_push!(wrapper, budget,
                                leaf_node(argid, source_index))
                    else
                        arg = _growth_gen!(rng, wrapper, language, argid,
                            target_info.context, generation_depth, budget;
                            mode=sibling_mode,
                            soft_depth=generation_depth + sibling_max_depth)
                    end
                else
                    arg = _growth_gen!(rng, wrapper, language, argid,
                        target_info.context, generation_depth, budget;
                        mode=sibling_mode,
                        soft_depth=generation_depth + sibling_max_depth)
                end
            finally
                budget.max_nodes = old_limit
                budget.max_depth = old_depth_limit
            end
        end
        arg === nothing && return _growth_fail(:insert_sibling_generation_failed,
            (position, argid, budget.attempts))
        arg isa Err && return arg
        current_type = parts[2]
        app = _growth_push!(wrapper, budget, app_node(current_type, current, arg))
        app === nothing && return _growth_fail(:node_budget_exceeded,
            (required, wrapper_capacity))
        current = app
    end
    current_type == target_info.typeid ||
        return _growth_fail(:insert_result_type_mismatch,
            (current_type, target_info.typeid))
    site_depth + _growth_subtree_depth(wrapper, Int(current)) <= max_depth ||
        return _growth_fail(:depth_budget_exceeded, site_depth +
            _growth_subtree_depth(wrapper, Int(current)))
    wrapper_term = TypedTerm(wrapper)
    bounded = _growth_within_budget(wrapper_term, language, target_info.context,
        wrapper_capacity, max_depth - site_depth)
    bounded === true || return _growth_fail(:internal_invalid_insert, bounded)
    return _growth_replace(rng, parent, site, wrapper_term, Int(current), language,
        parent_context, target_info.context, max_nodes, max_depth;
        random_mapping=false, prepared_parent=prepared)
end

function _growth_same_type_sites(term::TypedTerm, typeid::Int32, kinds=nothing)
    result = Int[]
    for i in eachindex(term.nodes)
        node = term.nodes[i]
        node.typeid == typeid || continue
        (kinds === nothing || node.kind in kinds) || continue
        push!(result, i)
    end
    return result
end

"""Return distinct atomic heads that occur on an application function spine.

An application whose function child is itself an application contributes only
the leftmost atomic `LEAF`/`VAR`; the intermediate `APP` nodes are not
replacement sites.  An abstraction-headed function spine is deliberately
ignored for this bounded operator proposal.
"""
function _growth_operator_heads(nodes::Vector{TermNode})
    heads = Int[]
    seen = Set{Int}()
    for node in nodes
        node.kind == APP_KIND || continue
        current = Int(node.x)
        current_node = nodes[current]
        current_node.kind in (LEAF_KIND, VAR_KIND) || continue
        current in seen || (push!(seen, current); push!(heads, current))
    end
    return heads
end

function _growth_operator_alternatives(node::TermNode, context::Vector{Int32},
                                       language::Language)
    alternatives = Tuple{Symbol,Int32}[]
    for leaf_index in _growth_leaf_candidates(language, node.typeid)
        node.kind == LEAF_KIND && node.x == leaf_index && continue
        push!(alternatives, (:leaf, leaf_index))
    end
    for db in _growth_var_candidates(context, node.typeid)
        node.kind == VAR_KIND && node.x == db && continue
        push!(alternatives, (:var, db))
    end
    return alternatives
end

function _growth_operator_candidates(term::TypedTerm, language::Language,
                                     parent_context::Vector{Int32},
                                     head_set::Set{Int})
    isempty(head_set) && return Tuple{Int,Vector{Tuple{Symbol,Int32}}}[]
    candidates = Tuple{Int,Vector{Tuple{Symbol,Int32}}}[]
    seen = Set{Int}()
    context = copy(parent_context)
    function visit(i::Int)
        node = term.nodes[i]
        if node.kind == ABS_KIND
            parts = _arrow_parts(language.types, node.typeid)
            parts === nothing && return _growth_fail(:abs_not_function_type, node.typeid)
            push!(context, parts[1])
            result = visit(Int(node.x))
            pop!(context)
            return result
        elseif node.kind == APP_KIND
            fn = Int(node.x)
            if fn in head_set && !(fn in seen)
                push!(seen, fn)
                alternatives = _growth_operator_alternatives(
                    term.nodes[fn], context, language)
                isempty(alternatives) || push!(candidates, (fn, alternatives))
            end
            result = visit(fn)
            result isa Err && return result
            return visit(Int(node.y))
        end
        return true
    end
    result = visit(length(term.nodes))
    result isa Err && return result
    return candidates
end

"""Replace one applied atomic function head without changing its arguments.

The selected head is a distinct `LEAF` or `VAR` on an application function
spine.  Alternatives are registered leaves and visible locals whose complete
type-table id exactly matches the head; no unification or scope remapping is
attempted.  The returned compact tree is validated against `max_nodes` and
`max_depth`.  A `SourceProgram` leaf may subsequently fail the search
written-complexity guard even when its compact node count is unchanged.
"""
function _growth_operator(rng, parent::TypedTerm, language::Language,
                          parent_context::Vector{Int32}, max_nodes::Integer,
                          max_depth::Integer)
    heads = _growth_operator_heads(parent.nodes)
    isempty(heads) && return _growth_fail(:no_operator_head)
    eligible = _growth_operator_candidates(parent, language, parent_context,
        Set(heads))
    eligible isa Err && return eligible
    isempty(eligible) && return _growth_fail(:no_compatible_operator, length(heads))
    head, alternatives = rand(rng, eligible)
    source_kind, source_index = rand(rng, alternatives)
    nodes = copy(parent.nodes)
    old = nodes[head]
    nodes[head] = source_kind === :leaf ?
        leaf_node(old.typeid, source_index) : var_node(old.typeid, source_index)
    child = TypedTerm(nodes)
    bounded = _growth_within_budget(child, language, parent_context,
        max_nodes, max_depth)
    bounded === true || return bounded
    return child
end

"""Propose one bounded typed variation, or return its explicit `Err`.

`kind=:generate` grows a fresh subtree at a uniformly selected parent site;
`:crossover` imports one donor occurrence of the same output type;
`:hoist` selects a same-typed proper descendant of the selected site; and
`:point` imports a same-typed leaf/variable.  Residual donor locals are mapped
independently to visible target locals of the exact same type, without
enumerating mappings.
`:insert` retains the selected occurrence as one exact-typed argument of a
finite visible function-head prefix and generates the other arguments under
the same node/depth budget.
`:operator` replaces one distinct applied atomic function head with a
same-`typeid` leaf or visible local; application arguments and binder indices
are copied unchanged.
`:eta_reduce` and `:eta_expand` delegate to the bounded representation move
for one explicitly requested direction.  They are ordinary weighted kinds;
the default search mixture does not select either one.
"""
function propose_term(rng, parent::TypedTerm, donor::TypedTerm, language::Language;
                      context::AbstractVector{<:Integer}=Int32[],
                      donor_context::AbstractVector{<:Integer}=Int32[],
                      max_nodes::Integer=128, max_depth::Integer=32,
                      kind::Symbol=:generate)
    max_nodes >= 1 || return _growth_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _growth_fail(:invalid_depth_budget, max_depth)
    kind in (:generate, :crossover, :hoist, :point, :insert, :operator,
             :eta_reduce, :eta_expand) ||
        return _growth_fail(:unknown_proposal_kind, kind)
    parent_ctx = _check_context(language.types, context)
    parent_ctx isa Err && return parent_ctx
    donor_ctx = _check_context(language.types, donor_context)
    donor_ctx isa Err && return donor_ctx
    parent_check = _growth_within_budget(parent, language, parent_ctx, max_nodes, max_depth)
    parent_check === true || return parent_check
    donor_check = validate(donor, language; context=donor_ctx)
    donor_check === true || return donor_check
    isempty(parent.nodes) && return _growth_fail(:empty_parent)

    if kind === :operator
        return _growth_operator(rng, parent, language, parent_ctx,
            max_nodes, max_depth)
    end

    if kind === :eta_reduce || kind === :eta_expand
        direction = kind === :eta_reduce ? :shrink : :expand
        return eta_move(rng, parent, language; context=parent_ctx,
            direction, max_nodes, max_depth)
    end

    site = rand(rng, 1:length(parent.nodes))
    target_info = site_info(parent, site, language; context=parent_ctx)
    target_info isa Err && return target_info

    if kind === :generate
        replacement_limit = max_nodes - length(parent.nodes) + length(target_info.span)
        replacement_limit >= 1 ||
            return _growth_fail(:node_budget_exceeded, length(parent.nodes))
        generated = generate_term(rng, language, target_info.typeid;
            context=target_info.context, max_nodes=replacement_limit, max_depth=max_depth,
            max_attempts=max(1, 4 * max_nodes))
        generated isa Err && return generated
        return _growth_replace(rng, parent, site, generated, length(generated.nodes), language,
                               parent_ctx, target_info.context, max_nodes, max_depth;
                               random_mapping=false)
    end

    if kind === :insert
        return _growth_insert(rng, parent, site, language, parent_ctx, target_info,
            max_nodes, max_depth; max_attempts=max(1, 4 * max_nodes))
    end

    donor_sites = if kind === :hoist
        first_site, last_site = first(target_info.span), last(target_info.span)
        [i for i in first_site:last_site if i != site &&
            parent.nodes[i].typeid == target_info.typeid]
    elseif kind === :crossover
        _growth_same_type_sites(donor, target_info.typeid)
    else
        Int[]
    end

    if kind === :point
        # Point mutation has a deliberately tiny source vocabulary: registered
        # leaves and visible variables at the selected target occurrence.
        point_sources = Tuple{Symbol,Int32}[]
        selected_node = parent.nodes[site]
        for leaf_index in _growth_leaf_candidates(language, target_info.typeid)
            selected_node.kind == LEAF_KIND && selected_node.x == leaf_index && continue
            push!(point_sources, (:leaf, leaf_index))
        end
        for db in _growth_var_candidates(target_info.context, target_info.typeid)
            selected_node.kind == VAR_KIND && selected_node.x == db && continue
            push!(point_sources, (:var, db))
        end
        isempty(point_sources) && return _growth_fail(:no_compatible_point, target_info.typeid)
        source_kind, source_index = rand(rng, point_sources)
        point_donor = if source_kind === :leaf
            TypedTerm(TermNode[leaf_node(target_info.typeid, source_index)])
        else
            TypedTerm(TermNode[var_node(target_info.typeid, source_index)])
        end
        point_context = source_kind === :leaf ? Int32[] : target_info.context
        return _growth_replace(rng, parent, site, point_donor, 1, language,
                               parent_ctx, point_context, max_nodes, max_depth;
                               random_mapping=false)
    end

    isempty(donor_sites) && return _growth_fail(:no_compatible_donor, target_info.typeid)
    donor_site = rand(rng, donor_sites)
    donor_term = kind === :hoist ? parent : donor
    donor_root_context = kind === :hoist ? parent_ctx : donor_ctx
    return _growth_replace(rng, parent, site, donor_term, donor_site, language,
                           parent_ctx, donor_root_context, max_nodes, max_depth)
end

@testitem "TypedSearch bounded growth uses typed function heads" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: BoolTy, FloatTy, TyMu, newtyvar
    using LambdaRegression.TypedSearch

    table = TypeTable()
    float = intern!(table, FloatTy)
    bool = intern!(table, BoolTy)
    f2f = arrow!(table, float, float)
    f2b = arrow!(table, float, bool)
    language = Language(table, [LeafSpec(f2f, :float_identity), LeafSpec(f2b, :is_float)])

    # There is no Float leaf; applications are available alongside the
    # visible Float local and must not be silently omitted from head growth.
    found_application = Ref(false)
    for seed in 1:16
        term = generate_term(MersenneTwister(seed), language, float;
                             context=Int32[float], max_nodes=8, max_depth=4)
        @test term isa TypedTerm
        @test validate(term, language; context=Int32[float]) === true
        found_application[] |= any(node -> node.kind == APP, term.nodes)
    end
    @test found_application[]

    bool_term = generate_term(MersenneTwister(2000), language, bool;
                              context=Int32[float], max_nodes=8, max_depth=4)
    @test bool_term isa TypedTerm
    @test validate(bool_term, language; context=Int32[float]) === true
    @test any(node -> node.kind == APP, bool_term.nodes)

    # Closed higher-order construction can introduce and capture binders.
    lambda = generate_term(MersenneTwister(1), Language(table), f2f;
                           max_nodes=3, max_depth=2)
    @test lambda isa TypedTerm
    @test validate(lambda, Language(table)) === true
    @test lambda.nodes[end].kind == ABS

    @test generate_term(MersenneTwister(1111), Language(table), float;
                        max_attempts=16) isa Err
    @test generate_term(MersenneTwister(1234), language, bool;
                        context=Int32[float], max_nodes=2) isa Err
    @test generate_term(MersenneTwister(8000), language, bool;
                        context=Int32[float], max_depth=0) isa Err

    α = newtyvar(0)
    @test intern_type!(table, TyMu(α, FloatTy)) isa Err
end

@testitem "TypedSearch operator replacement is head-only and exact-typed" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: BoolTy, FloatTy
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _growth_operator_heads

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    bool = intern!(table, BoolTy)
    fn = arrow!(table, scalar, scalar)
    curried = arrow!(table, scalar, fn)
    wrong_result = arrow!(table, scalar, bool)
    language = Language(table, [
        LeafSpec(curried, :f), LeafSpec(curried, :g),
        LeafSpec(wrong_result, :wrong), LeafSpec(scalar, :one),
    ])

    # The two curried applications contain one atomic head, not two APP sites.
    parent = TypedTerm(TermNode[
        leaf_node(curried, 1), leaf_node(scalar, 4), app_node(fn, 1, 2),
        leaf_node(scalar, 4), app_node(scalar, 3, 4),
    ])
    @test validate(parent, language) === true
    @test _growth_operator_heads(parent.nodes) == [1]
    before = copy(parent.nodes)
    child = propose_term(MersenneTwister(42), parent, parent, language;
        kind=:operator, max_nodes=8, max_depth=8)
    @test child isa TypedTerm
    @test child.nodes[1] == leaf_node(curried, 2)
    @test child.nodes[2:end] == parent.nodes[2:end]
    @test validate(child, language) === true
    @test parent.nodes == before

    # Two nested function binders expose a same-typed local, with all binder
    # identities and the scalar argument left untouched by head replacement.
    k_to_fn = arrow!(table, fn, fn)
    h_to_kfn = arrow!(table, fn, k_to_fn)
    higher = TypedTerm(TermNode[
        var_node(fn, 2), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3), abs_node(k_to_fn, 4), abs_node(h_to_kfn, 5),
    ])
    @test validate(higher, language) === true
    higher_child = propose_term(MersenneTwister(555), higher, higher, language;
        kind=:operator, max_nodes=8, max_depth=8)
    @test higher_child isa TypedTerm
    @test higher_child.nodes[1] == var_node(fn, 1)
    @test higher_child.nodes[2:end] == higher.nodes[2:end]
    @test validate(higher_child, language) === true

    # A higher-order argument is an opaque subtree: replacing F/G must not
    # rewrite the lambda argument or its binder indices.
    fn_to_scalar = arrow!(table, fn, scalar)
    f_high = add_leaf!(language, fn_to_scalar, :F)
    g_high = add_leaf!(language, fn_to_scalar, :G)
    higher_argument = TypedTerm(TermNode[
        leaf_node(fn_to_scalar, f_high), var_node(scalar, 0),
        abs_node(fn, 2), app_node(scalar, 1, 3),
    ])
    @test validate(higher_argument, language) === true
    higher_argument_child = propose_term(MersenneTwister(1234), higher_argument,
        higher_argument, language; kind=:operator, max_nodes=8, max_depth=8)
    @test higher_argument_child isa TypedTerm
    @test higher_argument_child.nodes[1] == leaf_node(fn_to_scalar, g_high)
    @test higher_argument_child.nodes[2:4] == higher_argument.nodes[2:4]
    @test validate(higher_argument_child, language) === true

    # Same arity is insufficient: a different complete result type is not a
    # legal replacement, and a non-applied atomic node is not a head site.
    add_leaf!(language, fn, :unary)
    wrong_parent = TypedTerm(TermNode[
        leaf_node(wrong_result, 3), leaf_node(scalar, 4), app_node(bool, 1, 2),
    ])
    @test validate(wrong_parent, language) === true
    wrong = propose_term(MersenneTwister(2000), wrong_parent, wrong_parent, language;
        kind=:operator, max_nodes=8, max_depth=8)
    @test wrong isa Err
    @test wrong.value.reason == :no_compatible_operator
    leaf_only = TypedTerm(TermNode[leaf_node(scalar, 4)])
    @test validate(leaf_only, language) === true
    none = propose_term(MersenneTwister(1), leaf_only, leaf_only, language;
        kind=:operator, max_nodes=8, max_depth=8)
    @test none isa Err
    @test none.value.reason == :no_operator_head

    # An ABS-headed application is a redex boundary for this proposal.
    redex = TypedTerm(TermNode[
        var_node(scalar, 0), abs_node(fn, 1), leaf_node(scalar, 4),
        app_node(scalar, 2, 3),
    ])
    @test validate(redex, language) === true
    @test _growth_operator_heads(redex.nodes) == Int[]
    @test propose_term(MersenneTwister(1111), redex, redex, language;
        kind=:operator, max_nodes=8, max_depth=8).value.reason == :no_operator_head
end

@testitem "TypedSearch insertion preserves bounds, anchors, and higher-order args" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypeSystem: FloatTy, TyMu, TyRigid, newtyvar
    using LambdaRegression.TypedSearch
    using LambdaRegression.TypedSearch: _growth_insert
    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    fn_to_fn = arrow!(table, fn, fn)
    parent = TypedTerm(TermNode[var_node(scalar, 1), abs_node(fn, 1)])
    context = Int32[scalar]
    language = Language(table, [LeafSpec(fn_to_fn, :identity)])
    @test validate(parent, language; context=context) === true
    info = site_info(parent, 2, language; context=context)
    inserted = _growth_insert(MersenneTwister(1), parent, 2, language, context,
        info, 12, 8)
    @test inserted isa TypedTerm
    @test validate(inserted, language; context=context) === true
    @test count(node -> node.kind == ABS_KIND, inserted.nodes) == 1
    @test count(node -> node.kind == APP_KIND, inserted.nodes) == 1
    @test any(node -> node.kind == VAR_KIND && node.typeid == scalar && node.x == 1,
        inserted.nodes)
    binary_head = arrow!(table, scalar, arrow!(table, scalar, scalar))
    binary_language = Language(table,
        [LeafSpec(binary_head, :binary), LeafSpec(scalar, :one)])
    binary_parent = TypedTerm(TermNode[leaf_node(scalar, 2)])
    @test validate(binary_parent, binary_language) === true
    binary_info = site_info(binary_parent, 1, binary_language)
    binary_old_span = length(binary_info.span)
    binary = _growth_insert(MersenneTwister(555), binary_parent, 1,
        binary_language, Int32[], binary_info, binary_old_span + 4, 4)
    @test binary isa TypedTerm
    @test validate(binary, binary_language) === true
    @test length(binary.nodes) == binary_old_span + 4
    two_arg_head = arrow!(table, fn, fn_to_fn)
    higher_language = Language(table,
        [LeafSpec(two_arg_head, :compose), LeafSpec(fn, :function_value)])
    higher = _growth_insert(MersenneTwister(1111), parent, 2, higher_language, context,
        site_info(parent, 2, higher_language; context=context), 20, 10)
    @test higher isa TypedTerm
    @test validate(higher, higher_language; context=context) === true
    @test count(node -> node.kind == APP_KIND, higher.nodes) >= 2
    near_language = Language(table)
    near_context = Int32[scalar, fn]
    near_parent = TypedTerm(TermNode[
        var_node(fn, 0), var_node(scalar, 1), app_node(scalar, 1, 2)])
    near_info = site_info(near_parent, 2, near_language; context=near_context)
    near = _growth_insert(MersenneTwister(1234), near_parent, 2, near_language,
        near_context, near_info, 8, 2)
    @test near isa TypedTerm
    @test validate(near, near_language; context=near_context) === true
    near_fail = _growth_insert(MersenneTwister(8000), near_parent, 2, near_language,
        near_context, near_info, 8, 1)
    @test near_fail isa Err
    @test near_fail.value.reason == :depth_budget_exceeded
    identity_language = Language(table, [LeafSpec(fn, :scalar_head)])
    identity_context = Int32[scalar, scalar]
    identity_parent = TypedTerm(TermNode[var_node(scalar, 1)])
    identity_info = site_info(identity_parent, 1, identity_language;
        context=identity_context)
    identity = _growth_insert(MersenneTwister(42), identity_parent, 1,
        identity_language, identity_context, identity_info, 8, 4)
    @test identity isa TypedTerm
    @test validate(identity, identity_language; context=identity_context) === true
    @test any(node -> node.kind == VAR_KIND && node.typeid == scalar && node.x == 1,
        identity.nodes)
    no_head_language = Language(table, [LeafSpec(scalar, :one)])
    scalar_parent = TypedTerm(TermNode[leaf_node(scalar, 1)])
    no_head = _growth_insert(MersenneTwister(555), scalar_parent, 1,
        no_head_language, Int32[], site_info(scalar_parent, 1, no_head_language), 8, 8)
    @test no_head isa Err
    @test no_head.value.reason == :no_compatible_insert_head
    budget_language = Language(table,
        [LeafSpec(scalar, :one), LeafSpec(fn, :scalar_id)])
    budget_parent = TypedTerm(TermNode[leaf_node(scalar, 1)])
    too_small = _growth_insert(MersenneTwister(2000), budget_parent, 1,
        budget_language, Int32[], site_info(budget_parent, 1, budget_language), 2, 8)
    @test too_small isa Err
    @test too_small.value.reason == :node_budget_exceeded
    rigid_table = TypeTable()
    rigid = intern!(rigid_table, TyRigid(77))
    rigid_fn = arrow!(rigid_table, rigid, rigid)
    rigid_language = Language(rigid_table,
        [LeafSpec(rigid, :opaque), LeafSpec(rigid_fn, :opaque_id)])
    rigid_parent = TypedTerm(TermNode[leaf_node(rigid, 1)])
    rigid_insert = _growth_insert(MersenneTwister(1), rigid_parent, 1,
        rigid_language, Int32[], site_info(rigid_parent, 1, rigid_language), 8, 8)
    @test rigid_insert isa TypedTerm
    @test validate(rigid_insert, rigid_language) === true
    @test intern_type!(rigid_table, TyMu(newtyvar(0), FloatTy)) isa Err
end

@testitem "TypedSearch ordinary eta kinds use bounded capture-safe moves" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(fn, :identity), LeafSpec(scalar, :one)])

    # The closed eta-expanded identity contracts to the function leaf.
    expanded = TypedTerm(TermNode[
        leaf_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3),
    ])
    @test validate(expanded, language) === true
    reduced = propose_term(MersenneTwister(2000), expanded, expanded, language;
        kind=:eta_reduce, max_nodes=8, max_depth=8)
    @test reduced isa TypedTerm
    @test reduced.nodes == TermNode[leaf_node(fn, 1)]
    @test validate(reduced, language) === true

    # Expansion is available as its own weighted kind, and remains bounded.
    expanded_again = propose_term(MersenneTwister(1), reduced, reduced, language;
        kind=:eta_expand, max_nodes=8, max_depth=8)
    @test expanded_again isa TypedTerm
    @test validate(expanded_again, language) === true
    @test length(expanded_again.nodes) == 4

    # Contracting an open term lowers the outer reference; it does not capture
    # it in the removed eta binder.
    open_expanded = TypedTerm(TermNode[
        var_node(fn, 1), var_node(scalar, 0), app_node(scalar, 1, 2),
        abs_node(fn, 3),
    ])
    open_reduced = propose_term(MersenneTwister(1111), open_expanded,
        open_expanded, language; context=Int32[fn], donor_context=Int32[fn],
        kind=:eta_reduce, max_nodes=8, max_depth=8)
    @test open_reduced isa TypedTerm
    @test open_reduced.nodes == TermNode[var_node(fn, 0)]
    @test validate(open_reduced, language; context=Int32[fn]) === true
end
