# Unified directional typed-crossover engine. The implementation lives behind
# the existing public `recombine` seam; tests here exercise that seam only.
using TestItems

using ...Procedural:
    FunctionDefNode, functiondef_recursion_summary, prepare_procedural_snapshot

struct _DirectionalPair
    recipient_site::Any
    donor_site::Any
    recipient_view::CrossoverSite
    donor_view::CrossoverSite
end

function _engine_contains_functiondef(node::AstNode)::Bool
    node isa FunctionDefNode && return true
    return any(_engine_contains_functiondef, children(node))
end

function _engine_snapshot(
    tree::AstTree,
    type_ctx::AbstractDict{Symbol,<:TyLike},
    target_ty::Ty,
    tyfactory_ctor,
    ;
    site_path::Union{Nothing,Vector{Int}}=nothing,
)
    if _engine_contains_functiondef(tree.root)
        return prepare_procedural_snapshot(
            tree,
            type_ctx;
            factory=tyfactory_ctor(),
            target_ty,
            site_path,
        )
    end
    return prepare_core_snapshot(
        tree,
        type_ctx;
        factory=tyfactory_ctor(),
        target_ty,
        site_path,
    )
end

struct _PreparedCrossoverParent
    tree::AstTree
    local_consts::Dict{Symbol,Any}
    local_consts_ty::Dict{Symbol,TyLike}
    global_consts_ty::Dict{Symbol,TyLike}
    target_ty::Ty
    snapshot::Any
    scoped::Any
    contextual_key::Any
end

function _prepare_crossover_parent(
    individual::Individual,
    global_consts_ty::Dict{Symbol,TyLike};
    tyfactory_ctor=make_tyvar_factory,
)
    type_ctx = merge(global_consts_ty, individual.local_consts_ty)
    snapshot = _engine_snapshot(
        individual.tree,
        type_ctx,
        individual.target_ty,
        tyfactory_ctor,
    )
    snapshot isa Err && return snapshot
    return _PreparedCrossoverParent(
        individual.tree,
        individual.local_consts,
        individual.local_consts_ty,
        global_consts_ty,
        individual.target_ty,
        snapshot,
        _resolve_scoped(individual.tree),
        _contextual_subtree_key(
            individual.tree.root,
            individual.local_consts,
            individual.local_consts_ty,
        ),
    )
end

function _checked_prepared_parent(
    prepared::_PreparedCrossoverParent,
    individual::Individual,
    global_consts_ty::Dict{Symbol,TyLike},
)
    prepared.tree === individual.tree &&
        prepared.local_consts === individual.local_consts &&
        prepared.local_consts_ty === individual.local_consts_ty &&
        prepared.global_consts_ty === global_consts_ty &&
        prepared.target_ty === individual.target_ty || throw(ArgumentError(
            "prepared crossover analysis belongs to another parent or type context",
        ))
    return prepared
end

_engine_site_type(site) = begin
    demand = getproperty(site, :demand)
    demand isa TyScheme ? demand.ty : demand
end

function _engine_policy_site(site)
    boundary = Tuple(getproperty(site, :boundary_visible))
    binder_schemes = getproperty(site, :binder_schemes)
    binder_types = Union{Nothing,TyLike}[
        get(binder_schemes, binder, nothing)
        for binder in reverse(boundary)
    ]
    capture_levels = Int[]
    for binder in keys(getproperty(site, :external_occurrences))
        position = findfirst(==(binder), boundary)
        position === nothing || push!(capture_levels, position - 1)
    end
    sort!(unique!(capture_levels))
    return CrossoverSite(
        collect(Int, getproperty(site, :path)),
        length(boundary),
        _engine_site_type(site),
        getproperty(site, :offer),
        binder_types,
        capture_levels,
    )
end

_engine_is_domain_exclusion(error)::Bool =
    error isa _SnapshotMuUnsupportedErr || error isa _SnapshotCyclicTypeErr

function _engine_witness_mappings(
    donor_site,
    recipient_site;
    max_open_answers_per_pair::Int=typemax(Int),
)
    max_open_answers_per_pair >= 1 || throw(ArgumentError(
        "max_open_answers_per_pair must be positive",
    ))
    if isempty(getproperty(donor_site, :external_occurrences))
        match = closed_directed_match(
            getproperty(recipient_site, :demand),
            getproperty(donor_site, :offer),
        )
        match isa Err && return (mappings=match, status=:exhausted)
        return (
            mappings=match ? Dict{Any,Any}[Dict{Any,Any}()] : Dict{Any,Any}[],
            status=:exhausted,
        )
    end

    batch = try
        _snapshot_open_retrieval_bounded(
            donor_site,
            [recipient_site];
            candidate_allowed=_transport_role_ok,
            max_answers=max_open_answers_per_pair,
        )
    catch error
        _engine_is_domain_exclusion(error) || rethrow()
        return (mappings=Err(error), status=:exhausted)
    end
    mappings = Dict{Any,Any}[]
    for answer in batch.answers
        push!(mappings, Dict{Any,Any}(answer.h))
    end
    return (mappings=mappings, status=batch.status)
end

function _engine_exchange_category(donor_site, mapping)::Symbol
    if any(pair -> getproperty(first(pair), :kind) === :function_def, mapping) ||
       functiondef_recursion_summary(getproperty(donor_site, :node)).self_calls > 0
        return :functiondef_self
    end
    isempty(getproperty(donor_site, :external_occurrences)) &&
        return :closed_carried
    return :open_contextual
end

function _engine_trace(callback, event::NamedTuple)
    callback === nothing || callback(event)
    return nothing
end

function _engine_weighted_site_index(
    rng::AbstractRNG,
    weights::Vector{Float64},
    positive::Vector{Int},
)
    total = sum(index -> weights[index], positive)
    threshold = rand(rng) * total
    cumulative = 0.0
    for index in positive
        cumulative += weights[index]
        threshold <= cumulative && return index
    end
    return last(positive)
end

function _engine_pair_proposals(
    rng::AbstractRNG,
    recipient_weights::Vector{Float64},
    donor_weights::Vector{Float64},
    max_try::Int,
)
    recipient_positive = findall(>(0), recipient_weights)
    donor_positive = findall(>(0), donor_weights)
    if isempty(recipient_positive) || isempty(donor_positive)
        return Tuple{Int,Int}[], true
    end
    pair_count = length(recipient_positive) * length(donor_positive)
    proposal_budget = min(pair_count, max(64, 8max_try))
    if pair_count <= proposal_budget
        return [
            (recipient_index, donor_index)
            for recipient_index in recipient_positive
            for donor_index in donor_positive
        ], true
    end

    proposals = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for _ in 1:(16proposal_budget)
        pair = (
            _engine_weighted_site_index(
                rng,
                recipient_weights,
                recipient_positive,
            ),
            _engine_weighted_site_index(rng, donor_weights, donor_positive),
        )
        if !(pair in seen)
            push!(seen, pair)
            push!(proposals, pair)
            length(proposals) == proposal_budget && return proposals, false
        end
    end

    # Highly concentrated weights can repeatedly draw one pair. Fill the
    # bounded proposal set from the highest-mass remaining Cartesian products
    # without scanning or solving the complete site product.
    recipient_order = sort(
        recipient_positive;
        by=index -> recipient_weights[index],
        rev=true,
    )
    donor_order = sort(
        donor_positive;
        by=index -> donor_weights[index],
        rev=true,
    )
    for recipient_index in recipient_order, donor_index in donor_order
        pair = (recipient_index, donor_index)
        pair in seen && continue
        push!(seen, pair)
        push!(proposals, pair)
        length(proposals) == proposal_budget && return proposals, false
    end
    return proposals, false
end

function _engine_closed_replay_offer(child_site)
    offer = getproperty(child_site, :offer)
    demand = getproperty(child_site, :demand)
    if offer isa TyScheme &&
       isempty(offer.vars) &&
       demand isa Ty &&
       !isempty(collect_free_vars(demand))
        return generalize(demand, -1)
    end
    return offer
end

function _engine_replays_target(child_site, original_target_site)::Bool
    if isempty(getproperty(child_site, :external_occurrences))
        replay = closed_directed_match(
            getproperty(original_target_site, :demand),
            _engine_closed_replay_offer(child_site),
        )
        return !(replay isa Err) && replay
    end

    batch = try
        _snapshot_open_retrieval_bounded(
            child_site,
            [original_target_site];
            candidate_allowed=(source, target) -> source == target,
            max_answers=1,
        )
    catch error
        _engine_is_domain_exclusion(error) || rethrow()
        return false
    end
    group_count = length(getproperty(child_site, :external_occurrences))
    return any(batch.answers) do answer
        length(answer.h) == group_count &&
            all(pair -> first(pair) == last(pair), answer.h)
    end
end

function _engine_application_roles_stable(
    original_scoped,
    original_tree::AstTree,
    child_tree::AstTree,
    path::Vector{Int},
)
    if original_scoped isa Err
        original_node = original_tree.root
        child_node = child_tree.root
        for child_index in path
            if original_node isa AppNode || child_node isa AppNode
                original_node isa AppNode && child_node isa AppNode || return false
                (original_node.func isa AbsNode) ==
                    (child_node.func isa AbsNode) || return false
            end
            original_children = children(original_node)
            child_children = children(child_node)
            1 <= child_index <= length(original_children) || return false
            1 <= child_index <= length(child_children) || return false
            original_node = original_children[child_index]
            child_node = child_children[child_index]
        end
        return true
    end
    retained = _retained_application_ancestors(original_scoped, path)
    retained isa Err && return false
    child_scoped = _resolve_scoped(child_tree)
    child_scoped isa Err && return false
    return _application_roles_stable(original_scoped, child_scoped, retained)
end

"""
    recombine(op::TypedSubtreeSwapCrossover, recipient, donor; kwargs...)
        -> Individual or Err

Insert one closed or contextually open donor fragment into a directed recipient
site. Retrieval checks `recipient demand ≼ donor offer`, transports all declared
dependencies, preserves binder roles, and replays the original site contract
after complete-child inference. The recipient and donor remain unchanged.

Built-in uniform/path policies use bounded stochastic site-pair proposals.
Custom pair-weight policies retain exact positive-pair enumeration before the
`max_try` execution budget is applied.
"""
function recombine(
    op::TypedSubtreeSwapCrossover,
    recipient::Individual,
    donor::Individual;
    rng::AbstractRNG=MersenneTwister(2000),
    global_consts_ty::Union{Nothing,Dict{Symbol,TyLike}}=nothing,
    tyfactory_ctor=make_tyvar_factory,
    const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
    prune::Bool=true,
    selection_roles::Tuple{Symbol,Symbol}=(:left, :right),
    donor_path=nothing,
    trace_callback=nothing,
    _prepared_recipient::Union{Nothing,_PreparedCrossoverParent}=nothing,
    _prepared_donor::Union{Nothing,_PreparedCrossoverParent}=nothing,
)
    global_consts_ty === nothing && return Err(MissingGlobalTypeCtxErr())
    selection_roles in ((:left, :right), (:right, :left)) ||
        throw(ArgumentError(
            "selection_roles must be (:left, :right) or (:right, :left)",
        ))
    recipient_role, donor_role = selection_roles

    recipient_prepared = _prepared_recipient === nothing ?
        _prepare_crossover_parent(
            recipient,
            global_consts_ty;
            tyfactory_ctor,
        ) :
        _checked_prepared_parent(_prepared_recipient, recipient, global_consts_ty)
    recipient_prepared isa Err && return recipient_prepared
    donor_prepared = _prepared_donor === nothing ?
        _prepare_crossover_parent(
            donor,
            global_consts_ty;
            tyfactory_ctor,
        ) :
        _checked_prepared_parent(_prepared_donor, donor, global_consts_ty)
    donor_prepared isa Err && return donor_prepared
    recipient_snapshot = recipient_prepared.snapshot
    donor_snapshot = donor_prepared.snapshot

    recipient_sites = op.allow_root ? recipient_snapshot.sites :
        filter(site -> !isempty(site.path), recipient_snapshot.sites)
    donor_sites = op.allow_root ? donor_snapshot.sites :
        filter(site -> !isempty(site.path), donor_snapshot.sites)
    length(recipient_sites) >= op.min_sites ||
        return Err(NoEligibleCrossoverSiteErr())
    length(donor_sites) >= op.min_sites ||
        return Err(NoEligibleCrossoverSiteErr())
    if donor_path !== nothing
        selected_path = Tuple(Int.(collect(donor_path)))
        donor_sites = filter(
            site -> Tuple(getproperty(site, :path)) == selected_path,
            donor_sites,
        )
        isempty(donor_sites) && return Err(NoEligibleCrossoverSiteErr())
    end

    recipient_views = [_engine_policy_site(site) for site in recipient_sites]
    donor_views = [_engine_policy_site(site) for site in donor_sites]
    recipient_weights = Float64[
        _checked_crossover_weight(
            crossover_site_weight(op.selection, view, recipient_role),
        ) for view in recipient_views
    ]
    donor_weights = Float64[
        _checked_crossover_weight(
            crossover_site_weight(op.selection, view, donor_role),
        ) for view in donor_views
    ]

    pair_proposals, proposals_exhaustive =
        _crossover_pair_weights_factorized(op.selection) ?
        _engine_pair_proposals(
            rng,
            recipient_weights,
            donor_weights,
            op.max_try,
        ) :
        ([(recipient_index, donor_index)
          for recipient_index in findall(>(0), recipient_weights)
          for donor_index in findall(>(0), donor_weights)], true)
    pairs = _DirectionalPair[]
    logweights = Float64[]
    for (recipient_index, donor_index) in pair_proposals
        recipient_site = recipient_sites[recipient_index]
        recipient_view = recipient_views[recipient_index]
        donor_site = donor_sites[donor_index]
        donor_view = donor_views[donor_index]
        if op.require_same_depth &&
           recipient_view.depth != donor_view.depth
            continue
        end
        iscompatible(recipient_view.ty, donor_view.ty) || continue
        pair_weight = _checked_crossover_weight(
            recipient_role === :left ?
                crossover_pair_weight(
                    op.selection,
                    recipient_view,
                    donor_view,
                ) :
                crossover_pair_weight(
                    op.selection,
                    donor_view,
                    recipient_view,
                ),
        )
        pair_weight > 0 || continue
        push!(pairs, _DirectionalPair(
            recipient_site,
            donor_site,
            recipient_view,
            donor_view,
        ))
        push!(logweights,
            proposals_exhaustive ?
                log(recipient_weights[recipient_index]) +
                log(donor_weights[donor_index]) +
                log(pair_weight) :
                log(pair_weight),
        )
    end
    if isempty(pairs)
        _engine_trace(trace_callback, (
            event=:terminal,
            outcome=:no_witness,
            target_path=(),
            donor_path=(),
            category=nothing,
            attempt=0,
        ))
        return proposals_exhaustive ?
            Err(NoEligibleCrossoverPairErr()) :
            Err(CrossoverMaxTryReachedErr(op.max_try))
    end

    recipient_scoped = recipient_prepared.scoped
    recipient_contextual_key = recipient_prepared.contextual_key
    order = _weighted_crossover_order(rng, logweights)
    attempts = 0
    first_domain_exclusion = nothing
    saw_supported_query = false
    saw_witness = false
    saw_answer_budget_exhaustion = false
    for order_index in 1:min(op.max_try, length(order))
        attempts += 1
        pair = pairs[order[order_index]]
        path = pair.recipient_view.path
        witness_batch = _engine_witness_mappings(
            pair.donor_site,
            pair.recipient_site,
            max_open_answers_per_pair=op.max_open_answers_per_pair,
        )
        mappings = witness_batch.mappings
        if mappings isa Err
            first_domain_exclusion === nothing &&
                (first_domain_exclusion = mappings)
            continue
        end
        saw_supported_query = true
        isempty(mappings) && continue
        saw_witness = true
        for mapping in mappings
            category = _engine_exchange_category(pair.donor_site, mapping)
            selected_event = (
                event=:selected,
                outcome=:selected,
                target_path=Tuple(path),
                donor_path=Tuple(pair.donor_view.path),
                category,
                attempt=attempts,
            )
            _engine_trace(trace_callback, selected_event)
            transported = transport_subtree(
                donor.tree,
                pair.donor_site,
                pair.recipient_site,
                mapping,
            )
            if transported isa Err
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:transport_rejected),
                ))
                continue
            end

            tree = deepcopy(recipient.tree)
            local_consts = deepcopy(recipient.local_consts)
            local_consts_ty = _detached_type_ctx(recipient.local_consts_ty)
            donor_local_consts_ty = _detached_type_ctx(donor.local_consts_ty)
            factory = const_symfactory_ctor()
            _separate_local_global_namespace!(
                tree,
                local_consts,
                local_consts_ty,
                global_consts_ty,
                factory,
            )
            _remap_donor_locals!(
                transported,
                deepcopy(donor.local_consts),
                donor_local_consts_ty,
                local_consts,
                local_consts_ty,
                global_consts_ty,
                factory,
            )
            if !_replace_at_path_maybe!(tree, path, transported)
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:transport_rejected),
                ))
                continue
            end
            if !_engine_application_roles_stable(
                recipient_scoped,
                recipient.tree,
                tree,
                path,
            )
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:transport_rejected),
                ))
                continue
            end

            prune && prune_local_consts!(tree, local_consts, local_consts_ty)
            child_contextual_key = _contextual_subtree_key(
                tree.root,
                local_consts,
                local_consts_ty,
            )
            if _same_contextual_subtree(
                recipient_contextual_key,
                child_contextual_key,
            )
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:no_change),
                ))
                continue
            end

            child_ctx = merge(global_consts_ty, local_consts_ty)
            child_snapshot = _engine_snapshot(
                tree,
            child_ctx,
            recipient.target_ty,
            tyfactory_ctor,
            site_path=path,
        )
            if child_snapshot isa Err ||
               !iscompatible(child_snapshot.root_type, recipient.target_ty)
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:invalid_child),
                ))
                continue
            end
            child_site_index = findfirst(
                site -> Tuple(site.path) == Tuple(pair.recipient_site.path),
                child_snapshot.sites,
            )
            if child_site_index === nothing
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:invalid_child),
                ))
                continue
            end
            child_site = child_snapshot.sites[child_site_index]
            if !_engine_replays_target(child_site, pair.recipient_site)
                _engine_trace(trace_callback, merge(
                    selected_event,
                    (; event=:terminal, outcome=:invalid_child),
                ))
                continue
            end

            child = Individual(
                tree,
                recipient.target_ty;
                local_consts,
                local_consts_ty,
                fitness=nothing,
                meta=(;),
            )
            _engine_trace(trace_callback, merge(
                selected_event,
                (; event=:terminal, outcome=:changed),
            ))
            return child
        end
        witness_batch.status === :answer_limit &&
            (saw_answer_budget_exhaustion = true)
    end
    if attempts == length(pairs) && proposals_exhaustive
        if !saw_supported_query && first_domain_exclusion !== nothing
            return first_domain_exclusion
        elseif !saw_witness
            _engine_trace(trace_callback, (
                event=:terminal,
                outcome=:no_witness,
                target_path=(),
                donor_path=(),
                category=nothing,
                attempt=attempts,
            ))
            return Err(NoEligibleCrossoverPairErr())
        end
        saw_answer_budget_exhaustion && return Err(
            CrossoverAnswerBudgetExhaustedErr(op.max_open_answers_per_pair),
        )
        return Err(CrossoverPairPoolExhaustedErr(attempts))
    end
    saw_answer_budget_exhaustion && return Err(
        CrossoverAnswerBudgetExhaustedErr(op.max_open_answers_per_pair),
    )
    return Err(CrossoverMaxTryReachedErr(op.max_try))
end

@testitem "directional engine bounds Cartesian site proposals deterministically" begin
    using Test
    using Random
    using LambdaRegression.Mutations.Populations: _engine_pair_proposals

    recipient_weights = ones(100)
    donor_weights = ones(100)
    first, first_exhaustive = _engine_pair_proposals(
        MersenneTwister(1111),
        recipient_weights,
        donor_weights,
        40,
    )
    second, second_exhaustive = _engine_pair_proposals(
        MersenneTwister(1111),
        recipient_weights,
        donor_weights,
        40,
    )
    @test !first_exhaustive
    @test !second_exhaustive
    @test first == second
    @test length(first) == 320
    @test length(unique(first)) == length(first)

    complete, exhaustive = _engine_pair_proposals(
        MersenneTwister(1234),
        [1.0, 0.0, 1.0],
        [1.0, 1.0],
        40,
    )
    @test exhaustive
    @test Set(complete) == Set([(1, 1), (1, 2), (3, 1), (3, 2)])
end

@testitem "directional engine bounds open witness answers" begin
    using Test
    using LambdaRegression.ASTs.LexicalContexts: BinderId
    using LambdaRegression.Mutations.Populations: _engine_witness_mappings
    using LambdaRegression.TypeSystem: IntTy, TyLike, TyScheme

    donor_binder = BinderId(:abs, (), 9)
    donor = (;
        offer=IntTy,
        external_occurrences=Dict(
            donor_binder => [(path=(1,), use_type=IntTy)],
        ),
        binder_schemes=Dict{Any,TyLike}(
            donor_binder => TyScheme([], IntTy),
        ),
    )
    first_target = BinderId(:abs, (), 1)
    second_target = BinderId(:abs, (), 2)
    recipient = (;
        path=(1,),
        demand=IntTy,
        offer=TyScheme([], IntTy),
        binder_schemes=Dict{Any,TyLike}(
            first_target => TyScheme([], IntTy),
            second_target => TyScheme([], IntTy),
        ),
    )

    exact = _engine_witness_mappings(
        donor,
        recipient;
        max_open_answers_per_pair=2,
    )
    limited = _engine_witness_mappings(
        donor,
        recipient;
        max_open_answers_per_pair=1,
    )
    @test length(exact.mappings) == 2
    @test exact.status === :exhausted
    @test length(limited.mappings) == 1
    @test limited.status === :answer_limit
end

@testitem "directional engine preserves sparse custom pair policies" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        AbstractCrossoverSelectionPolicy, CrossoverSite, Individual,
        TypedSubtreeSwapCrossover, recombine
    import LambdaRegression.Mutations.Populations: crossover_pair_weight
    using LambdaRegression.TypeSystem:
        FADD, FMUL, FSUB, FloatTy, TyFun, get_type_ctx
    using LambdaRegression.Utils: Err

    mutable struct SparseRootBodyPair <: AbstractCrossoverSelectionPolicy
        calls::Int
    end
    function crossover_pair_weight(
        policy::SparseRootBodyPair,
        left::CrossoverSite,
        right::CrossoverSite,
    )
        policy.calls += 1
        return left.path == [1] && right.path == [1] ? 1.0 : 0.0
    end

    add(left, right) = AppNode(AppNode(ConstNode(:FADD), left), right)
    sub(left, right) = AppNode(AppNode(ConstNode(:FSUB), left), right)
    mul(left, right) = AppNode(AppNode(ConstNode(:FMUL), left), right)
    recipient_terms = Any[IndexNode(0)]
    append!(recipient_terms, [
        ConstNode(isodd(index) ? :ONE : iszero(index % 3) ? :THREE : :TWO)
        for index in 1:15
    ])
    donor_terms = Any[IndexNode(0)]
    append!(donor_terms, [
        ConstNode(isodd(index) ? :THREE : iszero(index % 3) ? :ONE : :TWO)
        for index in 1:15
    ])
    recipient_body = foldl(add, recipient_terms)
    donor_body = foldl(mul, donor_terms)
    recipient = Individual(
        AstTree(AbsNode(:x, recipient_body)),
        TyFun(FloatTy, FloatTy),
    )
    donor = Individual(
        AstTree(AbsNode(:x, donor_body)),
        TyFun(FloatTy, FloatTy),
    )
    values = Dict{Symbol,Any}(
        :FADD => FADD,
        :FSUB => FSUB,
        :FMUL => FMUL,
        :ONE => 1.0,
        :TWO => 2.0,
        :THREE => 3.0,
    )
    policy = SparseRootBodyPair(0)
    child = recombine(
        TypedSubtreeSwapCrossover(max_try=1, selection=policy),
        recipient,
        donor;
        rng=MersenneTwister(8000),
        global_consts_ty=get_type_ctx(values),
    )
    @test policy.calls > 64
    @test !(child isa Err)
    child isa Err || @test child.tree != recipient.tree
end

@testitem "directional engine recombines an open core ancestor transactionally" begin
    using Test
    using Random
    using LambdaRegression.ASTs.Contexts: ConstSymbolFactory
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        Individual, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover,
        _prepare_crossover_parent, recombine
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike, infer
    using LambdaRegression.Utils: Err

    int_function = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :ADD => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :KEEP3 => TyFun(IntTy, TyFun(IntTy, TyFun(int_function, int_function))),
    )
    recipient_tree = AstTree(AbsNode(
        :z,
        AppNode(
            AppNode(
                AppNode(ConstNode(:KEEP3), IndexNode(0)),
                ConstNode(:payload),
            ),
            ConstNode(:old),
        ),
    ))
    donor_tree = AstTree(AbsNode(
        :x,
        AbsNode(
            :y,
            AppNode(
                AppNode(ConstNode(:ADD), IndexNode(1)),
                ConstNode(:payload),
            ),
        ),
    ))
    recipient = Individual(
        recipient_tree,
        TyFun(IntTy, int_function);
        local_consts=Dict{Symbol,Any}(:payload => 99, :old => identity),
        local_consts_ty=Dict{Symbol,TyLike}(
            :payload => IntTy,
            :old => int_function,
        ),
    )
    donor = Individual(
        donor_tree,
        TyFun(IntTy, int_function);
        local_consts=Dict{Symbol,Any}(:payload => [2]),
        local_consts_ty=Dict{Symbol,TyLike}(:payload => IntTy),
    )
    before_recipient = deepcopy(recipient)
    before_donor = deepcopy(donor)
    op = TypedSubtreeSwapCrossover(
        max_try=1,
        min_sites=1,
        selection=PathCrossoverSelectionPolicy([1, 2], [1]),
    )
    events = NamedTuple[]

    child = recombine(
        op,
        recipient,
        donor;
        rng=MersenneTwister(1111),
        global_consts_ty=global_types,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        trace_callback=event -> push!(events, event),
    )

    @test !(child isa Err)
    child isa Err && return
    @test child.tree != recipient.tree
    @test infer(child.tree, merge(global_types, child.local_consts_ty)) ==
        TyFun(IntTy, int_function)
    @test child.local_consts[:payload] == 99
    @test !haskey(child.local_consts, :old)
    copied_payloads = [
        value for value in values(child.local_consts)
        if value isa Vector{Int}
    ]
    @test copied_payloads == [[2]]
    push!(only(copied_payloads), 3)
    @test donor.local_consts[:payload] == [2]
    @test recipient.tree == before_recipient.tree
    @test recipient.local_consts == before_recipient.local_consts
    @test recipient.local_consts_ty == before_recipient.local_consts_ty
    @test donor.tree == before_donor.tree
    @test donor.local_consts == before_donor.local_consts
    @test donor.local_consts_ty == before_donor.local_consts_ty
    @test last(events).event === :terminal
    @test last(events).outcome === :changed
    @test last(events).category === :open_contextual

    prepared_child = recombine(
        op,
        recipient,
        donor;
        rng=MersenneTwister(1111),
        global_consts_ty=global_types,
        const_symfactory_ctor=() -> ConstSymbolFactory(prefix=:c),
        _prepared_recipient=_prepare_crossover_parent(recipient, global_types),
        _prepared_donor=_prepare_crossover_parent(donor, global_types),
    )
    @test !(prepared_child isa Err)
    prepared_child isa Err || @test prepared_child.tree == child.tree
end

@testitem "directional engine maps FunctionDef self only to FunctionDef self" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        Individual, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover,
        _snapshot_open_retrieval, prepare_core_snapshot, recombine
    using LambdaRegression.Procedural:
        FunctionDefNode, prepare_procedural_snapshot
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    int_function = TyFun(IntTy, IntTy)
    global_types = Dict{Symbol,TyLike}(
        :ADD => TyFun(IntTy, TyFun(IntTy, IntTy)),
        :ONE => IntTy,
        :USE_FUN => TyFun(int_function, IntTy),
    )
    recipient = Individual(
        AstTree(FunctionDefNode(AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:ADD), AppNode(IndexNode(1), IndexNode(0))),
                ConstNode(:ONE),
            ),
        ))),
        int_function,
    )
    donor = Individual(
        AstTree(FunctionDefNode(AbsNode(
            :x,
            AppNode(
                AppNode(ConstNode(:ADD), ConstNode(:ONE)),
                AppNode(IndexNode(1), IndexNode(0)),
            ),
        ))),
        int_function,
    )
    op = TypedSubtreeSwapCrossover(
        max_try=1,
        min_sites=1,
        selection=PathCrossoverSelectionPolicy([1], [1]),
    )
    events = NamedTuple[]
    child = recombine(
        op,
        recipient,
        donor;
        rng=MersenneTwister(1234),
        global_consts_ty=global_types,
        trace_callback=event -> push!(events, event),
    )
    @test !(child isa Err)
    child isa Err || @test child.tree != recipient.tree
    @test last(events).outcome === :changed
    @test last(events).category === :functiondef_self

    ordinary = Individual(
        AstTree(AbsNode(
            :f,
            AppNode(
                ConstNode(:USE_FUN),
                AbsNode(:x, AppNode(IndexNode(1), IndexNode(0))),
            ),
        )),
        TyFun(int_function, IntTy),
    )
    donor_snapshot = prepare_procedural_snapshot(donor.tree, global_types)
    ordinary_snapshot = prepare_core_snapshot(ordinary.tree, global_types)
    donor_site = only(filter(site -> site.path == (1,), donor_snapshot.sites))
    ordinary_site = only(filter(
        site -> site.path == (1, 2),
        ordinary_snapshot.sites,
    ))
    @test !isempty(_snapshot_open_retrieval(donor_site, [ordinary_site]))

    rejected = recombine(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy([1, 2], [1]),
        ),
        ordinary,
        donor;
        rng=MersenneTwister(8000),
        global_consts_ty=global_types,
    )
    @test rejected isa Err
end

@testitem "directional engine rejects contextual no-op and procedural role drift" begin
    using Test
    using Random
    using LambdaRegression.ASTs.NodeASTs:
        AbsNode, AppNode, AstTree, ConstNode, IndexNode
    using LambdaRegression.Mutations.Populations:
        Individual, PathCrossoverSelectionPolicy, TypedSubtreeSwapCrossover,
        recombine
    using LambdaRegression.Procedural: FunctionDefNode
    using LambdaRegression.TypeSystem: IntTy, TyFun, TyLike
    using LambdaRegression.Utils: Err

    renamed_recipient = Individual(
        AstTree(ConstNode(:left)),
        IntTy;
        local_consts=Dict{Symbol,Any}(:left => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:left => IntTy),
    )
    renamed_donor = Individual(
        AstTree(ConstNode(:right)),
        IntTy;
        local_consts=Dict{Symbol,Any}(:right => 1),
        local_consts_ty=Dict{Symbol,TyLike}(:right => IntTy),
    )
    no_op = recombine(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            selection=PathCrossoverSelectionPolicy(Int[], Int[]),
        ),
        renamed_recipient,
        renamed_donor;
        rng=MersenneTwister(42),
        global_consts_ty=Dict{Symbol,TyLike}(),
    )
    @test no_op isa Err

    global_types = Dict{Symbol,TyLike}(
        :F => TyFun(IntTy, IntTy),
        :ONE => IntTy,
    )
    procedural_recipient = Individual(
        AstTree(FunctionDefNode(AppNode(ConstNode(:F), ConstNode(:ONE)))),
        IntTy,
    )
    closed_identity = Individual(
        AstTree(AbsNode(:x, IndexNode(0))),
        TyFun(IntTy, IntTy),
    )
    role_drift = recombine(
        TypedSubtreeSwapCrossover(
            max_try=1,
            min_sites=1,
            require_same_depth=false,
            selection=PathCrossoverSelectionPolicy([1, 1], Int[]),
        ),
        procedural_recipient,
        closed_identity;
        rng=MersenneTwister(555),
        global_consts_ty=global_types,
    )
    @test role_drift isa Err
end
