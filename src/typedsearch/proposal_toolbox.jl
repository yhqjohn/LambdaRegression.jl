"""Bounded conventional proposal toolbox for a typed-search island.

The toolbox is an opt-in proposal layer over the compact `TypedTerm` API.  It
does not alter the canonical generator or the ordinary `propose_term` seam.
`generate_toolbox` chooses one of a few fixed depth/size tiers and then calls
the bounded sampler.  `propose_toolbox` first removes operations with no
locally applicable site, samples the remaining operations with the supplied
weights, and performs at most one optional second ordinary edit.

The default operation weights are `generate=.20`, `crossover=.25`,
`atomic_rewire=.15`, `operator=.10`, `insert=.10`, `hoist=.10`, `point=.05`,
`simplify=.05`, and zero for `eta_reduce` and `eta_expand`. Omitted operation
weights are zero, so a sparse NamedTuple selects just the requested operations.
Crossover is type-first and gives compact donor sizes in
`[ceil(s/2), 2s]` with a `.75` preference probability; it falls back to all
compatible donor sites.  Mapping uses the existing finite residual/scope
transport and never enumerates context or site pairs.
"""

using Random: AbstractRNG, rand

export TypedProposalPolicy, propose_toolbox, generate_toolbox

const _TOOLBOX_KINDS = (:generate, :crossover, :atomic_rewire, :operator,
                        :insert, :hoist, :point, :simplify,
                        :eta_reduce, :eta_expand)
const _TOOLBOX_SECOND_KINDS = (:atomic_rewire, :operator, :insert, :hoist,
                               :eta_reduce, :eta_expand)
const _TOOLBOX_DEFAULT_WEIGHTS = (generate=0.20, crossover=0.25,
    atomic_rewire=0.15, operator=0.10, insert=0.10, hoist=0.10,
    point=0.05, simplify=0.05, eta_reduce=0.0, eta_expand=0.0)

"""Fixed controls for the bounded proposal toolbox.

An operation is enabled by giving it a positive weight; omitted weights are
zero.  `generation_mode` may be `:grow`, `:branch_first`, or `:mixed` (equal
probability).  Tiers are
clamped to the caller's hard budgets and are preferences, not promises about
the returned size.  The eta directions are controlled by their weights and
have no duplicate switch fields.  The defaults are deliberately finite so a failed
proposal is an explicit `Err`, never an unbounded search.
"""
Base.@kwdef struct TypedProposalPolicy
    weights::NamedTuple = _TOOLBOX_DEFAULT_WEIGHTS

    generation_mode::Symbol = :mixed
    depth_tiers::Tuple = (1, 2, 3, 4)
    size_tiers::Tuple = (3, 9, 17, 31)
    generation_attempts::Int = 16
    generation_soft_depth::Union{Nothing,Int} = nothing

    crossover_type_first::Bool = true
    crossover_size_fair_probability::Float64 = 0.75
    crossover_attempts::Int = 8

    insert_sibling_max_nodes::Int = 5
    insert_sibling_max_depth::Int = 2
    insert_attempts::Int = 16

    double_edit_probability::Float64 = 0.10
end

@inline _toolbox_fail(reason::Symbol, detail=nothing) =
    Err(TypedSearchError(reason, detail))

function _toolbox_weight(policy::TypedProposalPolicy, kind::Symbol)
    hasproperty(policy.weights, kind) || return 0.0
    value = getproperty(policy.weights, kind)
    value isa Real || throw(ArgumentError(
        "proposal weight for $(kind) must be a real number, got $(typeof(value))"))
    return Float64(value)
end

function _toolbox_validate_policy(policy::TypedProposalPolicy)
    policy.generation_mode in (:grow, :branch_first, :mixed) ||
        return _toolbox_fail(:invalid_generation_mode, policy.generation_mode)
    policy.generation_attempts >= 1 ||
        return _toolbox_fail(:invalid_generation_attempts, policy.generation_attempts)
    policy.crossover_attempts >= 1 ||
        return _toolbox_fail(:invalid_crossover_attempts, policy.crossover_attempts)
    policy.insert_attempts >= 1 ||
        return _toolbox_fail(:invalid_insert_attempts, policy.insert_attempts)
    0.0 <= policy.crossover_size_fair_probability <= 1.0 ||
        return _toolbox_fail(:invalid_size_fair_probability,
            policy.crossover_size_fair_probability)
    0.0 <= policy.double_edit_probability <= 1.0 ||
        return _toolbox_fail(:invalid_double_edit_probability,
            policy.double_edit_probability)
    policy.insert_sibling_max_nodes >= 1 ||
        return _toolbox_fail(:invalid_sibling_node_budget,
            policy.insert_sibling_max_nodes)
    policy.insert_sibling_max_depth >= 0 ||
        return _toolbox_fail(:invalid_sibling_depth_budget,
            policy.insert_sibling_max_depth)
    isempty(policy.depth_tiers) &&
        return _toolbox_fail(:empty_depth_tiers)
    isempty(policy.size_tiers) &&
        return _toolbox_fail(:empty_size_tiers)
    all(t -> t isa Integer && t >= 0, policy.depth_tiers) ||
        return _toolbox_fail(:invalid_depth_tiers, policy.depth_tiers)
    all(t -> t isa Integer && t >= 1, policy.size_tiers) ||
        return _toolbox_fail(:invalid_size_tiers, policy.size_tiers)
    allowed = keys(_TOOLBOX_DEFAULT_WEIGHTS)
    all(kind -> kind in allowed, keys(policy.weights)) ||
        return _toolbox_fail(:invalid_proposal_weights, keys(policy.weights))
    values = Float64[]
    for kind in _TOOLBOX_KINDS
        weight = _toolbox_weight(policy, kind)
        isfinite(weight) && weight >= 0.0 ||
            return _toolbox_fail(:invalid_proposal_weight, (kind, weight))
        push!(values, weight)
    end
    any(>(0.0), values) || return _toolbox_fail(:zero_proposal_weights)
    return true
end

function _toolbox_generation_mode(rng::AbstractRNG, policy::TypedProposalPolicy)
    policy.generation_mode === :mixed && return rand(rng) < 0.5 ? :grow : :branch_first
    return policy.generation_mode
end

function _toolbox_tier(rng::AbstractRNG, tiers, hard::Int)
    candidates = Int[Int(t) for t in tiers if Int(t) >= 0]
    isempty(candidates) && return 0
    return min(hard, rand(rng, candidates))
end

function _toolbox_parent_checks(parent::TypedTerm, donor::TypedTerm,
                                language::Language, context, donor_context,
                                max_nodes::Integer, max_depth::Integer)
    max_nodes >= 1 || return _toolbox_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _toolbox_fail(:invalid_depth_budget, max_depth)
    parent_ctx = _check_context(language.types, context)
    parent_ctx isa Err && return parent_ctx
    donor_ctx = _check_context(language.types, donor_context)
    donor_ctx isa Err && return donor_ctx
    parent_check = _growth_within_budget(parent, language, parent_ctx,
        max_nodes, max_depth)
    parent_check === true || return parent_check
    donor_check = validate(donor, language; context=donor_ctx)
    donor_check === true || return donor_check
    parent_prepared = _prepared_sites(parent, language, parent_ctx)
    parent_prepared isa Err && return parent_prepared
    donor_prepared = _prepared_sites(donor, language, donor_ctx)
    donor_prepared isa Err && return donor_prepared
    return (parent_ctx, donor_ctx, parent_prepared, donor_prepared)
end

function _toolbox_atomic_sites(term::TypedTerm, language::Language,
                               context::Vector{Int32}; prepared=nothing)
    if prepared !== nothing
        if !_prepared_owner(prepared, term, language)
            return (Int[], Dict{Int,Vector{Tuple{Symbol,Int32}}}())
        end
        return prepared.atomic_sites, Dict{Int,Vector{Tuple{Symbol,Int32}}}()
    end
    sites = Int[]
    alternatives = Dict{Int,Vector{Tuple{Symbol,Int32}}}()
    for i in eachindex(term.nodes)
        node = term.nodes[i]
        node.kind in (VAR_KIND, LEAF_KIND) || continue
        info = site_info(term, i, language; context=context)
        info isa Err && continue
        choices = _toolbox_atom_sources(node, info, language)
        isempty(choices) || (push!(sites, i); alternatives[i] = choices)
    end
    return sites, alternatives
end

"""List exact-typed atomic sources at a site, preserving source namespaces.

Leaf indices and De Bruijn indices are both integers but are not comparable
identities.  Keep a `(kind, index)` tag until construction so a VAR can be
rewired to a function-valued LEAF (and vice versa) even when their integer
indices happen to coincide.
"""
function _toolbox_atom_sources(node::TermNode, info::SiteInfo,
                               language::Language)
    choices = Tuple{Symbol,Int32}[]
    for leaf_index in _growth_leaf_candidates(language, info.typeid)
        node.kind == LEAF_KIND && node.x == leaf_index && continue
        push!(choices, (:leaf, leaf_index))
    end
    for db in _growth_var_candidates(info.context, info.typeid)
        node.kind == VAR_KIND && node.x == db && continue
        push!(choices, (:var, db))
    end
    return choices
end

function _toolbox_point_sites(term::TypedTerm, language::Language,
                              context::Vector{Int32}; prepared=nothing)
    if prepared !== nothing
        _prepared_owner(prepared, term, language) || return Int[]
        return prepared.point_sites
    end
    sites = Int[]
    for i in eachindex(term.nodes)
        node = term.nodes[i]
        info = site_info(term, i, language; context=context)
        info isa Err && continue
        alternatives = _toolbox_atom_sources(node, info, language)
        node.kind in (VAR_KIND, LEAF_KIND) && !isempty(alternatives) && push!(sites, i)
        # Point replacement can replace a compound site by an atomic source too.
        node.kind in (ABS_KIND, APP_KIND) && !isempty(alternatives) && push!(sites, i)
    end
    return sites
end

function _toolbox_insert_sites(term::TypedTerm, language::Language,
                               context::Vector{Int32}, max_depth::Int;
                               prepared=nothing)
    if prepared !== nothing
        _prepared_owner(prepared, term, language) || return Int[]
        sites = Int[]
        for i in eachindex(term.nodes)
            remaining = max_depth - Int(prepared.site_depth[i])
            remaining >= 1 || continue
            _prepared_has_head(prepared, i, term.nodes[i].typeid, remaining) &&
                push!(sites, i)
        end
        return sites
    end
    sites = Int[]
    for i in eachindex(term.nodes)
        info = site_info(term, i, language; context=context)
        info isa Err && continue
        site_depth = _growth_site_depth(term.nodes, length(term.nodes), i, 0)
        site_depth === nothing && continue
        remaining = max_depth - site_depth
        remaining >= 1 || continue
        heads = _growth_heads(language, info.context, info.typeid;
            max_args=remaining)
        any(h -> any(==(info.typeid), h[3]), heads) && push!(sites, i)
    end
    return sites
end

function _toolbox_hoist_sites(term::TypedTerm, language::Language,
                              context::Vector{Int32}; prepared=nothing)
    if prepared !== nothing
        _prepared_owner(prepared, term, language) || return Int[]
        return prepared.hoist_sites
    end
    sites = Int[]
    spans = _spans(term.nodes)
    spans === nothing && return sites
    for i in eachindex(term.nodes)
        info = site_info(term, i, language; context=context)
        info isa Err && continue
        first_site, last_site = first(info.span), last(info.span)
        any(j -> j != i && term.nodes[j].typeid == info.typeid,
            first_site:last_site) && push!(sites, i)
    end
    return sites
end

function _toolbox_applicable(policy::TypedProposalPolicy, parent::TypedTerm,
                             donor::TypedTerm, language::Language,
                             parent_context::Vector{Int32}, donor_context::Vector{Int32};
                             kinds=_TOOLBOX_KINDS, max_depth::Int=typemax(Int),
                             prepared_parent=nothing, prepared_donor=nothing)
    result = Symbol[]
    for kind in kinds
        _toolbox_weight(policy, kind) > 0.0 || continue
        if kind === :generate
            isempty(parent.nodes) || push!(result, kind)
        elseif kind === :atomic_rewire
            sites, _ = _toolbox_atomic_sites(parent, language, parent_context;
                prepared=prepared_parent)
            isempty(sites) || push!(result, kind)
        elseif kind === :operator
            candidates = _growth_operator_candidates(parent, language,
                parent_context, Set(_growth_operator_heads(parent.nodes)))
            candidates isa Err || isempty(candidates) || push!(result, kind)
        elseif kind === :insert
            isempty(_toolbox_insert_sites(parent, language, parent_context,
                max_depth; prepared=prepared_parent)) || push!(result, kind)
        elseif kind === :hoist
            isempty(_toolbox_hoist_sites(parent, language, parent_context;
                prepared=prepared_parent)) ||
                push!(result, kind)
        elseif kind === :point
            isempty(_toolbox_point_sites(parent, language, parent_context;
                prepared=prepared_parent)) ||
                push!(result, kind)
        elseif kind === :simplify
            candidates = _SimplifiableRedex[]
            _simplify_collect!(candidates, parent.nodes, length(parent.nodes), language)
            isempty(candidates) || push!(result, kind)
        elseif kind === :eta_reduce || kind === :eta_expand
            # This scan is intentionally below the positive-weight check
            # above: the default zero eta weights must not add a proposal
            # scan or alter the default toolbox RNG path.
            direction = kind === :eta_reduce ? :shrink : :expand
            isempty(_eta_collect(parent, language, direction)) ||
                push!(result, kind)
        elseif kind === :crossover
            parent_types = prepared_parent === nothing ?
                Set(node.typeid for node in parent.nodes) : Set(prepared_parent.type_ids)
            donor_types = prepared_donor === nothing ?
                Set(node.typeid for node in donor.nodes) : Set(prepared_donor.type_ids)
            isempty(intersect(parent_types, donor_types)) || push!(result, kind)
        end
    end
    return result
end

function _toolbox_choose(rng::AbstractRNG, policy::TypedProposalPolicy,
                         kinds::Vector{Symbol})
    isempty(kinds) && return nothing
    weights = Float64[_toolbox_weight(policy, kind) for kind in kinds]
    total = sum(weights)
    total > 0.0 || return nothing
    point = rand(rng) * total
    for (kind, weight) in zip(kinds, weights)
        point -= weight
        point <= 0.0 && return kind
    end
    return last(kinds)
end

function _toolbox_apply_generate(rng::AbstractRNG, parent::TypedTerm,
                                 language::Language, parent_ctx::Vector{Int32},
                                 policy::TypedProposalPolicy,
                                 max_nodes::Int, max_depth::Int;
                                 prepared_parent=nothing)
    site = rand(rng, 1:length(parent.nodes))
    info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_ctx) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    info isa Err && return info
    site_depth = _growth_site_depth(parent.nodes, length(parent.nodes), site, 0)
    site_depth === nothing && return _toolbox_fail(:unreachable_site, site)
    replacement_limit = max_nodes - length(parent.nodes) + length(info.span)
    replacement_limit >= 1 || return _toolbox_fail(:node_budget_exceeded,
        replacement_limit)
    depth_limit = max_depth - site_depth
    depth_limit >= 0 || return _toolbox_fail(:depth_budget_exceeded, site_depth)
    # Reuse the public toolbox initializer so local regrowth samples the same
    # fixed depth/size tiers as initialization.  The remaining site budgets
    # are hard caps; no tier is advertised as an achieved final size.
    generated = generate_toolbox(rng, language, info.typeid;
        context=info.context, policy=policy, max_nodes=replacement_limit,
        max_depth=depth_limit)
    generated isa Err && return generated
    return _growth_replace(rng, parent, site, generated, length(generated.nodes),
        language, parent_ctx, info.context, max_nodes, max_depth;
        random_mapping=false, prepared_parent=prepared_parent)
end

function _toolbox_apply_atomic(rng::AbstractRNG, parent::TypedTerm,
                               language::Language, parent_ctx::Vector{Int32},
                               max_nodes::Int, max_depth::Int;
                               prepared_parent=nothing)
    sites, alternatives = _toolbox_atomic_sites(parent, language, parent_ctx;
        prepared=prepared_parent)
    isempty(sites) && return _toolbox_fail(:no_atomic_rewire_site)
    site = rand(rng, sites)
    info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_ctx) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    info isa Err && return info
    if prepared_parent !== nothing
        alternatives = Dict{Int,Vector{Tuple{Symbol,Int32}}}(site =>
            _toolbox_atom_sources(parent.nodes[site], info, language))
    end
    source_kind, source_index = rand(rng, alternatives[site])
    source = source_kind === :leaf ?
        TypedTerm(TermNode[leaf_node(info.typeid, source_index)]) :
        TypedTerm(TermNode[var_node(info.typeid, source_index)])
    source_context = source_kind === :leaf ? Int32[] : info.context
    return _growth_replace(rng, parent, site, source, 1, language, parent_ctx,
        source_context, max_nodes, max_depth; random_mapping=false,
        prepared_parent=prepared_parent)
end

function _toolbox_apply_point(rng::AbstractRNG, parent::TypedTerm,
                              donor::TypedTerm, language::Language,
                              parent_ctx::Vector{Int32}, max_nodes::Int,
                              max_depth::Int; prepared_parent=nothing)
    sites = _toolbox_point_sites(parent, language, parent_ctx;
        prepared=prepared_parent)
    isempty(sites) && return _toolbox_fail(:no_compatible_point)
    site = rand(rng, sites)
    info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_ctx) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    info isa Err && return info
    selected = parent.nodes[site]
    sources = Tuple{Symbol,Int32}[]
    for leaf_index in _growth_leaf_candidates(language, info.typeid)
        selected.kind == LEAF_KIND && selected.x == leaf_index && continue
        push!(sources, (:leaf, leaf_index))
    end
    for db in _growth_var_candidates(info.context, info.typeid)
        selected.kind == VAR_KIND && selected.x == db && continue
        push!(sources, (:var, db))
    end
    isempty(sources) && return _toolbox_fail(:no_compatible_point, info.typeid)
    source_kind, source_index = rand(rng, sources)
    source = source_kind === :leaf ?
        TypedTerm(TermNode[leaf_node(info.typeid, source_index)]) :
        TypedTerm(TermNode[var_node(info.typeid, source_index)])
    source_context = source_kind === :leaf ? Int32[] : info.context
    return _growth_replace(rng, parent, site, source, 1, language, parent_ctx,
        source_context, max_nodes, max_depth; random_mapping=false,
        prepared_parent=prepared_parent)
end

function _toolbox_apply_operator(rng::AbstractRNG, parent::TypedTerm,
                                 language::Language, parent_ctx::Vector{Int32},
                                 max_nodes::Int, max_depth::Int;
                                 prepared_parent=nothing)
    return _growth_operator(rng, parent, language, parent_ctx, max_nodes, max_depth)
end

function _toolbox_apply_insert(rng::AbstractRNG, parent::TypedTerm,
                               language::Language, parent_ctx::Vector{Int32},
                               policy::TypedProposalPolicy, max_nodes::Int,
                               max_depth::Int; prepared_parent=nothing)
    sites = _toolbox_insert_sites(parent, language, parent_ctx, max_depth;
        prepared=prepared_parent)
    isempty(sites) && return _toolbox_fail(:no_compatible_insert_site)
    site = rand(rng, sites)
    info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_ctx) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    info isa Err && return info
    return _growth_insert(rng, parent, site, language, parent_ctx, info,
        max_nodes, max_depth; max_attempts=policy.insert_attempts,
        sibling_max_nodes=min(5, policy.insert_sibling_max_nodes),
        sibling_max_depth=min(2, policy.insert_sibling_max_depth),
        prefer_terminal=true, sibling_mode=:branch_first, prepared=prepared_parent)
end

function _toolbox_apply_hoist(rng::AbstractRNG, parent::TypedTerm,
                              language::Language, parent_ctx::Vector{Int32},
                              max_nodes::Int, max_depth::Int;
                              prepared_parent=nothing)
    sites = _toolbox_hoist_sites(parent, language, parent_ctx;
        prepared=prepared_parent)
    isempty(sites) && return _toolbox_fail(:no_compatible_hoist)
    site = rand(rng, sites)
    info = prepared_parent === nothing ?
        site_info(parent, site, language; context=parent_ctx) :
        _prepared_site_info(prepared_parent, site; term=parent, language=language)
    info isa Err && return info
    first_site, last_site = first(info.span), last(info.span)
    descendants = [j for j in first_site:last_site if j != site &&
        parent.nodes[j].typeid == info.typeid]
    isempty(descendants) && return _toolbox_fail(:no_compatible_hoist)
    donor_site = rand(rng, descendants)
    return _growth_replace(rng, parent, site, parent, donor_site, language,
        parent_ctx, parent_ctx, max_nodes, max_depth;
        prepared_parent=prepared_parent, prepared_donor=prepared_parent)
end

function _toolbox_apply_simplify(rng::AbstractRNG, parent::TypedTerm,
                                 language::Language, parent_ctx::Vector{Int32},
                                 max_nodes::Int, max_depth::Int)
    return simplify_term(rng, parent, language; context=parent_ctx,
        max_nodes=max_nodes, max_depth=max_depth)
end

function _toolbox_apply_crossover(rng::AbstractRNG, parent::TypedTerm,
                                  donor::TypedTerm, language::Language,
                                  parent_ctx::Vector{Int32}, donor_ctx::Vector{Int32},
                                  policy::TypedProposalPolicy, max_nodes::Int,
                                  max_depth::Int;
                                  prepared_parent=nothing,
                                  prepared_donor=nothing)
    parent_by_type = prepared_parent === nothing ? Dict{Int32,Vector{Int}}() :
        prepared_parent.by_type
    donor_by_type = prepared_donor === nothing ? Dict{Int32,Vector{Int}}() :
        prepared_donor.by_type
    if prepared_parent === nothing
        for i in eachindex(parent.nodes)
            push!(get!(parent_by_type, parent.nodes[i].typeid, Int[]), i)
        end
    end
    if prepared_donor === nothing
        for i in eachindex(donor.nodes)
            push!(get!(donor_by_type, donor.nodes[i].typeid, Int[]), i)
        end
    end
    donor_spans = prepared_donor === nothing ? _spans(donor.nodes) :
        Int[Int(prepared_donor.span_end[i] - prepared_donor.span_start[i] + 1)
            for i in eachindex(prepared_donor.span_end)]
    donor_spans === nothing && return _toolbox_fail(:bad_donor_layout)
    common = sort!(collect(intersect(Set(keys(parent_by_type)),
        Set(keys(donor_by_type)))))
    isempty(common) && return _toolbox_fail(:no_compatible_crossover)
    fair = policy.crossover_size_fair_probability > 0.0 &&
        rand(rng) < policy.crossover_size_fair_probability
    last_error = nothing
    for _ in 1:policy.crossover_attempts
        typeid = policy.crossover_type_first ? rand(rng, common) : nothing
        site = typeid === nothing ? rand(rng, 1:length(parent.nodes)) :
            rand(rng, parent_by_type[typeid])
        target_info = prepared_parent === nothing ?
            site_info(parent, site, language; context=parent_ctx) :
            _prepared_site_info(prepared_parent, site; term=parent, language=language)
        target_info isa Err && (last_error = target_info; continue)
        typeid === nothing && (typeid = target_info.typeid)
        donor_sites = get(donor_by_type, typeid, Int[])
        isempty(donor_sites) && (last_error = _toolbox_fail(
            :no_compatible_crossover, typeid); continue)
        if fair
            span = length(target_info.span)
            lo, hi = cld(span, 2), 2 * span
            sized = Int[]
            for candidate in donor_sites
                size = donor_spans[candidate]
                lo <= size <= hi && push!(sized, candidate)
            end
            isempty(sized) || (donor_sites = sized)
        end
        donor_site = rand(rng, donor_sites)
        result = _growth_replace(rng, parent, site, donor, donor_site, language,
            parent_ctx, donor_ctx, max_nodes, max_depth;
            prepared_parent=prepared_parent, prepared_donor=prepared_donor)
        result isa Err || return result
        last_error = result
    end
    return last_error isa Err ? last_error : _toolbox_fail(:no_compatible_crossover)
end

function _toolbox_apply_once(rng::AbstractRNG, kind::Symbol, parent::TypedTerm,
                             donor::TypedTerm, language::Language,
                             parent_ctx::Vector{Int32}, donor_ctx::Vector{Int32},
                             policy::TypedProposalPolicy, max_nodes::Int,
                             max_depth::Int; prepared_parent=nothing,
                             prepared_donor=nothing)
    kind === :generate && return _toolbox_apply_generate(rng, parent, language,
        parent_ctx, policy, max_nodes, max_depth;
        prepared_parent=prepared_parent)
    kind === :crossover && return _toolbox_apply_crossover(rng, parent, donor,
        language, parent_ctx, donor_ctx, policy, max_nodes, max_depth;
        prepared_parent=prepared_parent, prepared_donor=prepared_donor)
    kind === :atomic_rewire && return _toolbox_apply_atomic(rng, parent,
        language, parent_ctx, max_nodes, max_depth;
        prepared_parent=prepared_parent)
    kind === :operator && return _toolbox_apply_operator(rng, parent, language,
        parent_ctx, max_nodes, max_depth; prepared_parent=prepared_parent)
    kind === :insert && return _toolbox_apply_insert(rng, parent, language,
        parent_ctx, policy, max_nodes, max_depth;
        prepared_parent=prepared_parent)
    kind === :hoist && return _toolbox_apply_hoist(rng, parent, language,
        parent_ctx, max_nodes, max_depth; prepared_parent=prepared_parent)
    kind === :point && return _toolbox_apply_point(rng, parent, donor, language,
        parent_ctx, max_nodes, max_depth; prepared_parent=prepared_parent)
    kind === :simplify && return _toolbox_apply_simplify(rng, parent, language,
        parent_ctx, max_nodes, max_depth)
    kind === :eta_reduce && return eta_move(rng, parent, language;
        context=parent_ctx, direction=:shrink, max_nodes, max_depth)
    kind === :eta_expand && return eta_move(rng, parent, language;
        context=parent_ctx, direction=:expand, max_nodes, max_depth)
    return _toolbox_fail(:unknown_proposal_kind, kind)
end

"""Generate one bounded initialization term using fixed A depth/size tiers."""
function generate_toolbox(rng::AbstractRNG, language::Language, targetid::Integer;
                          context::AbstractVector{<:Integer}=Int32[],
                          policy::TypedProposalPolicy=TypedProposalPolicy(),
                          max_nodes::Integer=31, max_depth::Integer=6)
    checked = _toolbox_validate_policy(policy)
    checked === true || return checked
    max_nodes >= 1 || return _toolbox_fail(:invalid_node_budget, max_nodes)
    max_depth >= 0 || return _toolbox_fail(:invalid_depth_budget, max_depth)
    # Clamp a selected tier, preserving the caller's hard cap.  The sampler
    # itself remains the authority for type/scope and exact structural checks.
    nodes = _toolbox_tier(rng, policy.size_tiers, Int(max_nodes))
    depth = _toolbox_tier(rng, policy.depth_tiers, Int(max_depth))
    nodes = max(nodes, 1)
    mode = _toolbox_generation_mode(rng, policy)
    soft_depth = policy.generation_soft_depth === nothing ? depth :
        min(depth, max(0, policy.generation_soft_depth))
    return generate_term(rng, language, targetid; context=context,
        max_nodes=nodes, max_depth=depth, max_attempts=policy.generation_attempts,
        mode=mode, soft_depth=soft_depth)
end

"""Propose one bounded variation and return `(term, kind, edit_count)`.

The parent and donor are validated and copied through the existing typed
replacement boundary, so the returned term owns a fresh node vector.  Empty
operation classes are filtered before weighted dispatch.  If the first edit
succeeds, `double_edit_probability` gives a bounded second draw among
`atomic_rewire`, `operator`, `insert`, `hoist`, and either eta direction,
using their configured weights; a failed second edit keeps
the first child and reports `edit_count=1`.
"""
function propose_toolbox(rng::AbstractRNG, parent::TypedTerm, donor::TypedTerm,
                         language::Language;
                         context::AbstractVector{<:Integer}=Int32[],
                         donor_context::AbstractVector{<:Integer}=Int32[],
                         policy::TypedProposalPolicy=TypedProposalPolicy(),
                         max_nodes::Integer=128, max_depth::Integer=32)
    checked_policy = _toolbox_validate_policy(policy)
    checked_policy === true || return checked_policy
    checked = _toolbox_parent_checks(parent, donor, language, context,
        donor_context, max_nodes, max_depth)
    checked isa Err && return checked
    parent_ctx, donor_ctx, parent_prepared, donor_prepared = checked
    applicable = _toolbox_applicable(policy, parent, donor, language,
        parent_ctx, donor_ctx; max_depth=Int(max_depth),
        prepared_parent=parent_prepared, prepared_donor=donor_prepared)
    first_kind = _toolbox_choose(rng, policy, applicable)
    first_kind === nothing && return _toolbox_fail(:no_applicable_proposal)
    child = _toolbox_apply_once(rng, first_kind, parent, donor, language,
        parent_ctx, donor_ctx, policy, Int(max_nodes), Int(max_depth);
        prepared_parent=parent_prepared, prepared_donor=donor_prepared)
    child isa Err && return child
    child isa TypedTerm || return _toolbox_fail(:invalid_proposal_result,
        typeof(child))
    # Every operation above validates its child; this final ownership check
    # makes the public boundary robust to future operation implementations.
    bounded = _growth_within_budget(child, language, parent_ctx,
        max_nodes, max_depth)
    bounded === true || return bounded
    result_kind = first_kind
    edit_count = 1
    if policy.double_edit_probability > 0.0 &&
            rand(rng) < policy.double_edit_probability
        child_prepared = _prepared_sites(child, language, parent_ctx)
        child_prepared isa Err && return (term=child, kind=result_kind, edit_count=edit_count)
        second_applicable = _toolbox_applicable(policy, child, donor, language,
            parent_ctx, donor_ctx; kinds=_TOOLBOX_SECOND_KINDS,
            max_depth=Int(max_depth), prepared_parent=child_prepared,
            prepared_donor=donor_prepared)
        second_kind = _toolbox_choose(rng, policy, second_applicable)
        if second_kind !== nothing
            second = _toolbox_apply_once(rng, second_kind, child, donor, language,
                parent_ctx, donor_ctx, policy, Int(max_nodes), Int(max_depth);
                prepared_parent=child_prepared, prepared_donor=donor_prepared)
            if second isa TypedTerm
                child = second
                result_kind = second_kind
                edit_count = 2
            end
        end
    end
    return (term=child, kind=result_kind, edit_count=edit_count)
end

@testitem "TypedSearch proposal toolbox preserves ownership, scope, and bounds" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    higher = arrow!(table, fn, scalar)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(scalar, :b),
        LeafSpec(fn, :f), LeafSpec(fn, :g), LeafSpec(higher, :apply)])
    parent = TypedTerm(TermNode[leaf_node(scalar, 1)])
    donor = TypedTerm(TermNode[leaf_node(scalar, 2)])
    @test validate(parent, language) === true
    @test validate(donor, language) === true
    policy = TypedProposalPolicy(weights=(atomic_rewire=1.0,),
        double_edit_probability=0.0)
    before = copy(parent.nodes)
    rewired = propose_toolbox(MersenneTwister(42), parent, donor, language;
        policy=policy, max_nodes=4, max_depth=2)
    @test rewired.term isa TypedTerm
    @test rewired.kind == :atomic_rewire
    @test rewired.edit_count == 1
    @test rewired.term.nodes != parent.nodes
    @test parent.nodes == before
    @test validate(rewired.term, language) === true
    @test length(rewired.term.nodes) <= 4

    # VAR and LEAF use separate integer namespaces.  Both directions remain
    # available even when the source indices are numerically equal.
    var_parent = TypedTerm(TermNode[var_node(scalar, 0)])
    var_result = propose_toolbox(MersenneTwister(1234), var_parent, var_parent,
        Language(table, [LeafSpec(scalar, :replacement)]);
        context=Int32[scalar], donor_context=Int32[scalar], policy=policy,
        max_nodes=4, max_depth=2)
    @test var_result.term.nodes[1].kind == LEAF_KIND
    leaf_language = Language(table, [LeafSpec(scalar, :only_leaf)])
    leaf_parent = TypedTerm(TermNode[leaf_node(scalar, 1)])
    leaf_result = propose_toolbox(MersenneTwister(8000), leaf_parent, leaf_parent,
        leaf_language; context=Int32[scalar], donor_context=Int32[scalar],
        policy=policy, max_nodes=4, max_depth=2)
    @test leaf_result.term.nodes[1].kind == VAR_KIND

    # A captured higher-order term can be atomically rewired at the function
    # value while preserving the lambda binder and its scalar argument.
    captured = TypedTerm(TermNode[
        var_node(higher, 0), var_node(scalar, 0), abs_node(fn, 2),
        app_node(scalar, 1, 3),
    ])
    @test validate(captured, language; context=Int32[higher]) === true
    cap_policy = TypedProposalPolicy(weights=(atomic_rewire=1.0,),
        double_edit_probability=0.0)
    cap = propose_toolbox(MersenneTwister(555), captured, captured, language;
        context=Int32[higher], donor_context=Int32[higher], policy=cap_policy,
        max_nodes=8, max_depth=5)
    @test cap isa NamedTuple
    @test cap.term isa TypedTerm
    @test validate(cap.term, language; context=Int32[higher]) === true

    @test propose_toolbox(MersenneTwister(2000), parent, donor, language;
        policy=TypedProposalPolicy(weights=(generate=0.0,))) isa Err
    @test_throws ArgumentError propose_toolbox(MersenneTwister(2000), parent,
        donor, language; policy=TypedProposalPolicy(
            weights=(atomic_rewire="invalid",)))
    rigid = TypeTable()
    @test intern_type!(rigid, LambdaRegression.TypeSystem.TyMu(
        LambdaRegression.TypeSystem.newtyvar(0), FloatTy)) isa Err
end

@testitem "TypedSearch toolbox crossover is type-first and size-bounded" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy, BoolTy
    using LambdaRegression.Utils: Err
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    bool = intern!(table, BoolTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(scalar, :a), LeafSpec(scalar, :b),
        LeafSpec(fn, :f), LeafSpec(fn, :g)])
    parent = TypedTerm(TermNode[
        leaf_node(fn, 3), leaf_node(scalar, 1), app_node(scalar, 1, 2),
    ])
    donor = TypedTerm(TermNode[leaf_node(fn, 4), leaf_node(scalar, 2),
        app_node(scalar, 1, 2)])
    @test validate(parent, language) === true
    @test validate(donor, language) === true
    policy = TypedProposalPolicy(weights=(crossover=1.0,),
        double_edit_probability=0.0,
        crossover_size_fair_probability=0.75)
    result = propose_toolbox(MersenneTwister(1), parent, donor, language;
        policy=policy, max_nodes=8, max_depth=6)
    @test result.term isa TypedTerm
    @test result.kind == :crossover
    @test validate(result.term, language) === true
    @test result.term.nodes !== parent.nodes
    @test result.edit_count == 1
    bool_index = add_leaf!(language, bool, :yes)
    incompatible = propose_toolbox(MersenneTwister(1111), parent,
        TypedTerm(TermNode[leaf_node(bool, bool_index)]), language;
        policy=policy, max_nodes=8, max_depth=6)
    @test incompatible isa Err
end

@testitem "TypedSearch toolbox eta kinds are optional and weighted exclusively" begin
    using Test
    using Random: MersenneTwister
    using LambdaRegression.TypeSystem: FloatTy
    using LambdaRegression.TypedSearch

    table = TypeTable()
    scalar = intern!(table, FloatTy)
    fn = arrow!(table, scalar, scalar)
    language = Language(table, [LeafSpec(fn, :identity)])
    leaf = TypedTerm(TermNode[leaf_node(fn, 1)])
    eta_policy = TypedProposalPolicy(
        # Omitted operation weights are zero, including eta_reduce.
        weights=(eta_expand=1.0,), double_edit_probability=0.0)
    expanded = propose_toolbox(MersenneTwister(1111), leaf, leaf, language;
        policy=eta_policy, max_nodes=8, max_depth=8)
    @test expanded isa NamedTuple
    @test expanded.kind === :eta_expand
    @test expanded.edit_count == 1
    @test validate(expanded.term, language) === true

    # A reduction-only policy cannot silently select expansion, and vice versa.
    reduced = propose_toolbox(MersenneTwister(1234), expanded.term,
        expanded.term, language; policy=TypedProposalPolicy(
            weights=(eta_reduce=1.0,), double_edit_probability=0.0),
        max_nodes=8, max_depth=8)
    @test reduced isa NamedTuple
    @test reduced.kind === :eta_reduce
    @test reduced.term.nodes == leaf.nodes
    @test TypedProposalPolicy().weights.eta_reduce == 0.0
end
