"""
    AbstractMutationProvider

Complete-program mutation extension. Implement
`propose_mutation(provider, parent::Program, target_ty::Ty;
                  rng, progress::ProposalProgress, components)` returning
`Program` or `Err`. Parent, reachable materials, target and components are
borrowed deeply read-only. External results are imported into owned materials,
then checked for binding/type, structural limits and candidate admission before
complete-task scoring. Expected failures return `Err`; programmer errors propagate.
Mutable providers are task-owned, or must synchronize their own shared state.

Distinct from `Mutations.AbstractSiteProvider`, which enumerates ordinary sites.
This hook is never called by scalar edits, finite generation or initialization
growth. Neural/model dependencies belong in external provider packages.
"""
abstract type AbstractMutationProvider end

"""Ordinary typed mutation using default component world/factory/limit fields.

Selection borrows the parent; private edits return `Err` on bounded failure.
No model calls are made. All configured ordinary mutation hooks are retained.
"""
struct OrdinaryMutationProvider <: AbstractMutationProvider end

"""
    EncodedMutationProvider(; encode, call_external, decode)

Format-independent serialization/external-call/deserialization adapter. Callbacks:
`encode(parent, target_ty; rng, progress, components) -> payload`,
`call_external(payload; rng, progress, components, target_ty) -> response`, and
`decode(payload, response; rng, progress, components, target_ty) -> Program`.
Any stage may return `Err` to stop the proposal. The shared search boundary owns
the resulting program and performs its type/binding/bounds checks once before
age scoring; callbacks must not duplicate that validation. Codecs own format,
vocabulary and resource bounds; external I/O, training and model dependencies
stay outside the package. A dummy callback can replace `call_external` directly.
"""
struct EncodedMutationProvider{E,C,D} <: AbstractMutationProvider
    encode::E
    call_external::C
    decode::D
end
EncodedMutationProvider(; encode, call_external, decode) =
    EncodedMutationProvider(encode, call_external, decode)

function _provider_weight(weight, progress)
    value = schedule_value(weight, progress)
    isfinite(value) && value >= 0 || throw(ArgumentError(
        "mutation provider weights must be finite and nonnegative"))
    return value
end

"""Mix a nonempty tuple of complete-program providers using scheduled weights.

`weights` is a matching tuple of numbers/schedules. Select once; selected failure
is not refunded or retried through another provider. Zero total weight returns
`Err(:zero_mutation_provider_weight)`. One positive weight consumes no selection
draw. Children may be mixtures. No extra context beyond the provider API is used.
"""
struct MixtureMutationProvider{P<:Tuple,W<:Tuple} <: AbstractMutationProvider
    providers::P
    weights::W
    function MixtureMutationProvider(providers::Tuple; weights::Tuple)
        !isempty(providers) && length(providers) == length(weights) ||
            throw(ArgumentError("providers and weights must have equal nonzero length"))
        all(p -> p isa AbstractMutationProvider, providers) ||
            throw(ArgumentError("all providers must subtype AbstractMutationProvider"))
        for weight in weights, endpoint in (0.0, 1.0)
            _provider_weight(weight, endpoint)
        end
        return new{typeof(providers),typeof(weights)}(providers, weights)
    end
end

function propose_mutation end

function propose_mutation(provider::EncodedMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple)
    payload = provider.encode(parent, target_ty; rng, progress, components)
    payload isa Err && return payload
    response = provider.call_external(payload; rng, progress, components, target_ty)
    response isa Err && return response
    return provider.decode(payload, response; rng, progress, components, target_ty)
end

function _select_mutation_provider(provider::MixtureMutationProvider, rng, progress)
    weights = map(weight -> _provider_weight(weight, progress), provider.weights)
    largest = maximum(weights)
    largest > 0 || return Err(:zero_mutation_provider_weight)
    # Rescale before addition so arbitrary finite weights cannot overflow.
    scaled = map(weight -> weight / largest, weights)
    positive = findall(>(0), scaled)
    selected = if length(positive) == 1
        only(positive)
    else
        threshold = rand(rng) * sum(scaled)
        chosen = last(positive)
        for index in positive
            threshold -= scaled[index]
            if threshold < 0
                chosen = index
                break
            end
        end
        chosen
    end
    return provider.providers[selected]
end

function propose_mutation(provider::MixtureMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple)
    selected = _select_mutation_provider(provider, rng, progress)
    selected isa Err && return selected
    return propose_mutation(selected, parent, target_ty;
        rng, progress, components)
end

function propose_mutation(::OrdinaryMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple)
    c = components
    driver = get(c, :mutation_driver, Mutations.mutate_tree)
    local_consts = copy(parent.local_consts)
    local_consts_ty = copy(parent.local_consts_ty)
    mutated = driver(parent.tree, target_ty;
        rng, mutations=c.mutations,
        chooser=Mutations.chooser_at_progress(c.mutation_chooser, progress),
        provider=c.mutation_provider, max_try=c.mutation_max_try,
        prune_consts=c.mutation_prune_consts,
        global_consts=c.global_consts,
        global_consts_ty=merge(c.global_consts_ty, c.input_types),
        local_consts, local_consts_ty, tyfactory=c.tyfactory_ctor(),
        symfactory=SymbolGenerator(), const_symfactory=c.const_symfactory_ctor(),
        bloat_policy=get(c, :bloat_policy, Mutations.BloatPolicy()),
        adaptive_policy=get(c, :mutation_adaptive_policy, Mutations.AdaptivePolicy()))
    mutated isa Err && return mutated
    mutated isa AstTree || throw(ArgumentError("mutation_driver must return AstTree or Err"))
    return _publish_program(mutated, local_consts, local_consts_ty)
end

"""
    checked_propose_mutation(provider, parent, target_ty; rng, progress, components)

Invoke the raw extension hook and return an owned, validated `Program` or `Err`.
Bounds are checked before inference. Generated type/binding/admission failures
are expected rejections; a wrong return type or callback exception is a programmer
error. Only the exact ordinary provider with a known validated driver may reuse
the driver's inference. External providers and mixtures always undergo one
inference here. Age scoring reuses this check; joint scalar-view reconstruction
still builds its own typed occurrence analysis.
"""
function checked_propose_mutation(provider::AbstractMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple)
    return _checked_propose_mutation(provider, parent, target_ty; rng, progress, components)
end

function _checked_propose_mutation(provider::AbstractMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple, check_admission::Bool=true)
    result = propose_mutation(provider, parent, target_ty; rng, progress, components)
    result isa Err && return result
    result isa Program || throw(ArgumentError("propose_mutation must return Program or Err"))
    ordinary = provider isa OrdinaryMutationProvider
    # Preserve the ordinary path's existing bounds/admission handling at scoring.
    if ordinary && _mutation_result_typechecked(get(components, :mutation_driver, Mutations.mutate_tree))
        if check_admission
            rejection = _program_candidate_rejection(result, components)
            rejection === nothing || return Err(rejection)
        end
        return result
    end
    _within_bloat_limit(result.tree, get(components, :bloat_policy, Mutations.BloatPolicy())) ||
        return Err(:mutation_provider_bloat)
    issue = _validate_program_contract(result, target_ty, components, "mutation provider candidate")
    issue === nothing || return Err(issue)
    rejection = check_admission ? _program_candidate_rejection(result, components) : nothing
    rejection === nothing || return Err(rejection)
    return Program(result.tree; local_consts=result.local_consts, local_consts_ty=result.local_consts_ty)
end

function _checked_propose_mutation(provider::MixtureMutationProvider, parent::Program, target_ty::Ty;
        rng, progress::ProposalProgress, components::NamedTuple, check_admission::Bool=true)
    selected = _select_mutation_provider(provider, rng, progress)
    selected isa Err && return selected
    return _checked_propose_mutation(selected, parent, target_ty; rng, progress, components, check_admission)
end

function _proposal_progress(components::NamedTuple, completed_attempts::Integer)
    offset = get(components, :schedule_offset, 0)
    horizon = get(components, :schedule_budget, 1)
    return ProposalProgress(Base.checked_add(Int(completed_attempts), offset), horizon)
end

export AbstractMutationProvider, OrdinaryMutationProvider, MixtureMutationProvider
export EncodedMutationProvider, propose_mutation, checked_propose_mutation

@testitem "Mutation providers preserve cumulative clocks and validate external results once" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Schedules
    using LambdaRegression.Regression: OrdinaryMutationProvider, MixtureMutationProvider,
        EncodedMutationProvider, propose_mutation
    R = LambdaRegression.Regression
    seen = Int[]
    stored = Ref{Program}()
    provider = EncodedMutationProvider(
        encode=(parent, target; rng, progress, components) -> begin
            push!(seen, progress.completed_attempts)
            (parent, progress.completed_attempts)
        end,
        call_external=(payload; kwargs...) -> payload[2] + 1.0,
        decode=(payload, result; kwargs...) -> begin
            program = Program(AstTree(AbsNode(:z, ConstNode(:k)));
                local_consts=Dict{Symbol,Any}(:k => result),
                local_consts_ty=Dict{Symbol,TyLike}(:k => FloatTy))
            stored[] = program
            program
        end)
    base, data = age_test_fixture()
    components = merge(base.components, (; full_mutation_provider=provider,
        schedule_budget=4, schedule_offset=2))
    reg = Regressor(base.target_ty, components, base.config, MersenneTwister(1))
    whole = initialize_age_search(reg, data; seed=2)
    advance_age_search!(whole; max_attempts=4, max_evaluations=4,
        mutation_weight=LinearSchedule(1, 1), crossover_weight=ConstantSchedule(0))
    @test seen == [2, 3, 4, 5]
    empty!(seen)
    chunked = initialize_age_search(reg, data; seed=2)
    advance_age_search!(chunked; max_attempts=2, max_evaluations=2)
    advance_age_search!(chunked; max_attempts=2, max_evaluations=2)
    @test seen == [2, 3, 4, 5]
    @test whole.counters.raw_attempts == chunked.counters.raw_attempts == 4
    @test whole.counters.advance_objective_calls == chunked.counters.advance_objective_calls
    retained = only(filter(r -> get(r.program.local_consts, :k, 0) == 6.0,
        collect(Iterators.flatten(chunked.populations)))).program
    stored[].local_consts[:k] = 999.0
    @test retained.local_consts[:k] == 6.0

    parent = first(first(chunked.populations))
    invalid = EncodedMutationProvider(encode=(args...; kwargs...) -> nothing,
        call_external=(args...; kwargs...) -> nothing,
        decode=(args...; kwargs...) -> Program(AstTree(IndexNode(0))))
    invalid_reg = Regressor(reg.target_ty, merge(components, (; full_mutation_provider=invalid)),
        reg.config, MersenneTwister(1234))
    invalid_state = initialize_age_search(invalid_reg, data; seed=3)
    @test R._age_mutation(invalid_state, parent, MersenneTwister(8000)) isa Err
    report = advance_age_search!(invalid_state; max_attempts=2, max_evaluations=2)
    @test report.expected_failures == 2
    @test report.objective_calls == 0
    @test parent.program.tree isa AstTree
    allzero = MixtureMutationProvider((provider, OrdinaryMutationProvider()); weights=(0, 0))
    @test propose_mutation(allzero, parent.program, reg.target_ty;
        rng=MersenneTwister(42), progress=ProposalProgress(0, 4), components) isa Err
    mixture = MixtureMutationProvider((invalid, provider);
        weights=(LinearSchedule(1, 0), LinearSchedule(0, 1)))
    @test propose_mutation(mixture, parent.program, reg.target_ty;
        rng=MersenneTwister(42), progress=ProposalProgress(4, 4), components) isa Program
    @test_throws ArgumentError MixtureMutationProvider((provider,); weights=(-1,))
    @test_throws ArgumentError default_components(schedule_budget=0)
    @test_throws ArgumentError default_components(schedule_offset=-1)
end

@testitem "Encoded mutation adapter short-circuits bounded failures" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Regression: EncodedMutationProvider, propose_mutation
    using LambdaRegression.Schedules: ProposalProgress
    reg, data = age_test_fixture()
    state = initialize_age_search(reg, data; seed=1)
    parent = first(first(state.populations)).program
    called = Ref(false)
    failed = EncodedMutationProvider(encode=(args...; kwargs...) -> Err(:codec),
        call_external=(args...; kwargs...) -> (called[] = true),
        decode=(args...; kwargs...) -> error("must not decode failed encoding"))
    result = propose_mutation(failed, parent, reg.target_ty;
        rng=MersenneTwister(1), progress=ProposalProgress(0, 1), components=reg.components)
    @test result isa Err
    @test result.value == :codec
    @test !called[]
end

@testitem "Checked mutation providers reject bounds before one inference" setup=[AgeSearchBaseImports] begin
    using LambdaRegression.Regression: EncodedMutationProvider, checked_propose_mutation
    using LambdaRegression.Schedules: ProposalProgress
    reg, data = age_test_fixture()
    parent = Program(AstTree(AbsNode(:z, IndexNode(0))))
    factory_calls = Ref(0)
    components = merge(reg.components, (; tyfactory_ctor=() -> begin
        factory_calls[] += 1
        reg.components.tyfactory_ctor()
    end))
    external = EncodedMutationProvider(encode=(args...; kwargs...) -> nothing,
        call_external=(args...; kwargs...) -> nothing,
        decode=(args...; kwargs...) -> parent)
    checked = checked_propose_mutation(external, parent, reg.target_ty;
        rng=MersenneTwister(1), progress=ProposalProgress(0, 2), components)
    @test checked isa Program
    @test factory_calls[] == 1
    tiny = merge(components, (; bloat_policy=LambdaRegression.Mutations.BloatPolicy(
        enabled=true, max_nodes=1)))
    @test checked_propose_mutation(external, parent, reg.target_ty;
        rng=MersenneTwister(1), progress=ProposalProgress(0, 2), components=tiny) isa Err
    @test factory_calls[] == 1
end

@testitem "Full mutation providers stay outside local joint scalar editing" setup=[JointSearchFixture, AgeSearchBaseImports] begin
    using LambdaRegression.Regression: EncodedMutationProvider
    calls = Ref(0)
    external = EncodedMutationProvider(encode=(args...; kwargs...) -> (calls[] += 1),
        call_external=(args...; kwargs...) -> Err(:unused),
        decode=(args...; kwargs...) -> error("unused"))
    # Capturing the provider in the task's world does not put it on local edits.
    base, data = age_test_fixture()
    c = merge(base.components, (; full_mutation_provider=external))
    reg = Regressor(base.target_ty, c, base.config, MersenneTwister(1))
    state = initialize_age_search(reg, data; seed=2)
    global_proposal = LambdaRegression.Workflows._joint_global_proposal(state; retry_limit=1)
    session = initialize_joint_search([pair_program(1, 2)], pair_ty, joint_types, _ -> 1.0;
        config=joint_config(max_evaluations=8), island_count=1,
        global_proposal, global_probability=0.0, rng=MersenneTwister(8000))
    advance_joint_search!(session; max_evaluations=2, max_attempts=4)
    @test calls[] == 0
    @test joint_search_result(session).attempts > 0
    parent = first(first(state.populations)).program
    global_proposal(parent, parent, MersenneTwister(42))
    @test calls[] == 1
end
