"""
    Procedural

Optional procedural-like syntax layered over the core lambda AST. The core AST does
not depend on this module: consumers may construct [`ScopeNode`](@ref) and
[`FunctionDefNode`](@ref), inspect their lexical child contexts, infer and evaluate
them natively, or lower them to ordinary `AbsNode`/`AppNode` trees for compatibility
with core-only consumers.

`ScopeNode([e₁, e₂], body)` is a sequential, maximally-generalising scope. The first
right-hand side sees only the outer context, the second also sees binding 1, and the
body sees both bindings. Its exact lowering is:

```text
{ x = e₁; y = e₂; body }
≡ (λ0. (λ1. body) e₂) e₁
```

Pretty printing keeps every bound occurrence as its exact De Bruijn `#i` and
adds only cyclic display names to binders. A recursive abstraction is shown as
`fn f(x). body`; nested definitions advance through `g`, `h`, and so on.

Binding identity is preserved across lowering. Scope binders and their lowering
abstractions occupy the same lexical positions, while self-free FunctionDef lowering
removes the unused self slot and rebases outer De Bruijn indices by one. Lowering
chooses deterministic fresh display names that cannot capture named `VarNode`s in the
source term. Use [`child_role`](@ref) and
[`visible_binders`](@ref) when a traversal or generator needs the context of one
child without knowing the node layout.

`FunctionDefNode(body)` gives `body` one anonymous monomorphic self binder at index
`#0`. If that binder is referenced, lowering emits `Y (λself. body)`; otherwise
it removes the unused slot and emits the equivalent rebased body without deleting
the stored FunctionDef node. [`fixed_point_witness`](@ref) supplies the default
finite core `Y` term; pass `fixed_point=` to [`lower_procedural`](@ref),
[`infer_procedural`](@ref), or [`reduce_procedural!`](@ref) to inject a different
closed witness or a zero-argument factory returning one.

`infer_procedural` additionally verifies that every nested FunctionDef resolves,
modulo equi-recursive `μ`, to a function-headed type. A non-function definition
returns `Err(NonFunctionDefinitionError(...))`. Legacy `LetNode`s are lowered to
the same HM let-like application used by Scope rather than passed to the core
adapter unchanged. When `meta` is supplied, inferred bounds are keyed by nodes
from the original input, not by temporary lowering nodes.

# Quick start

```julia
using LambdaRegression: Procedural
using LambdaRegression.ASTs.NodeASTs: AstNode, AstTree, ConstNode, IndexNode

term = Procedural.ScopeNode(AstNode[ConstNode(:value)], IndexNode(0))
core = Procedural.lower_procedural(AstTree(term))
ty = Procedural.infer_procedural(AstTree(term), type_context)
prepared = Procedural.prepare_procedural(term)
value = Procedural.run_procedural(prepared, runtime_context; args=(input,))
Procedural.reduce_procedural!(AstTree(term), 128, runtime_context)
```

`lower_procedural` returns a fresh core tree and leaves its input untouched.
`prepare_procedural` and `run_procedural` use a call-by-need evaluator whose shared
budget covers higher-order callbacks and direct one-layer recursive unfolding;
they do not materialise a fixed-point combinator. `reduce_procedural!` remains the
destructive compatibility/oracle path: it replaces the root by its lowering and
delegates to normal-order beta reduction. For pure primitives and terminating
programs the two paths are required to agree on the resulting value, but their
step counts are intentionally not equal: native execution skips the administrative
β steps that encode and unfold `Y`. Near a finite step limit, compare the classified
resource failure rather than requiring the same numeric cutoff. Initial random `Scope` generation is
outside this module's default behaviour; generation providers opt in explicitly.

# FunctionDef generation

[`generate_procedural`](@ref) and [`generate_procedural_scoped`](@ref) use canonical
generation plus one [`FunctionDefProvider`](@ref). The provider first obtains a
finite acyclic base witness with fixed-point and opaque capabilities transitively
masked, then performs one ordinary bounded body sample under an additional
monomorphic self binder. The default [`AcceptFunctionDef`](@ref) does not require
that binder to be referenced or called. Self use and genuine calls are observations
over generated batches, not per-node postconditions.

```julia
using LambdaRegression: CanonicalGeneration, Procedural
using LambdaRegression.TypeSystem: FloatTy, TyFun

target = TyFun(FloatTy, FloatTy)
weights = CanonicalGeneration.GenerationWeights(default=1, function_def=0.25)
provider = Procedural.FunctionDefProvider(
    max_nesting=1,
    body_attempts=4,
)
config = CanonicalGeneration.GenerationConfig(
    providers=(CanonicalGeneration.default_head_providers()..., provider),
    weights=weights,
    sampling_depth_limit=10,
    sampling_node_budget=128,
)
result = Procedural.generate_procedural(target; config=config)
```

The same callable can drive typed subtree replacement without making the mutation
module depend on procedural nodes:

```julia
using LambdaRegression: Mutations

replacement = Mutations.ScopedReplacementGenerator(
    Procedural.generate_procedural_scoped,
)
mutation = Mutations.SubtreeReplacementMutation(;
    replacement_generator=replacement,
    generation_kwargs=(;
        providers=config.providers,
        weights=config.weights,
        sampling_depth_limit=config.sampling_depth_limit,
        sampling_node_budget=config.sampling_node_budget,
    ),
)
procedural_inference = (tree, ctx, factory, meta) ->
    Procedural.infer_procedural(tree, ctx; factory, meta)
mutated = Mutations.mutate_tree(
    result.tree,
    target;
    mutations=[mutation],
    type_inference=procedural_inference,
    local_consts=result.local_consts,
    local_consts_ty=result.local_const_types,
)
```

Reuse the same provider/weight/budget configuration as initialization when both
operations should sample the same world. The local value/type dictionaries are
part of the individual and must be carried across mutation rounds. Site selection,
site locks, the scoped generation callable, and the inference callable remain
independent injection points.

An omitted `functiondef_provider` preserves a provider already present in `config`
or appends the default one. Pass `functiondef_provider=nothing` to disable the
built-in FunctionDef provider, or pass another `FunctionDefProvider` to replace
all embedded instances. A separate custom provider may still emit procedural
derivations under its own capability contract.
`weight_bucket` selects the composable weight tag; `max_nesting`, `body_attempts`,
and the ordinary depth/node/provider budgets bound search. `body_attempts` also
initializes one finite attempt counter per permitted nesting level, so a nested
attempt cannot exhaust its outer level while total FunctionDef work remains
bounded by `max_nesting * body_attempts`. `base_capability_mask`
adds capabilities that the base-witness phase must exclude. `base_witness` may be
replaced by a finite callable
`(relation, query, capability_mask) -> derivation_or_nothing`;
mutable strategies conservatively disable relation reuse. An admissibility policy
may accept either `candidate` or `(candidate, relation, query)` and must return
`Bool` in finite time. Its candidate exposes `target_ty`, `body`, `base_witness`,
and the zero-based `self_index`; these are typed derivation IR values, not AST names.
The default route samples once and returns finite failure when a custom policy
rejects it. [`generate_recursive_functiondef_scoped`](@ref) is the explicit
stronger API that uses a bounded genuine-self-call policy.

```julia
defaults = Procedural.FunctionDefProvider()
custom_provider = Procedural.FunctionDefProvider(
    base_witness=(relation, query, capability_mask) ->
        defaults.base_witness(relation, query, capability_mask),
    admissibility=(candidate, relation, query) ->
        defaults.admissibility(candidate),
)
```

The base check preserves caller binder exclusions and excludes every active recursive
self binder. The candidate owns a structural copy of its base witness, so a custom
policy cannot mutate the relation's cached proof state.

Safe list access, comparisons, constants, factories, and future heads remain normal
injected providers or context entries. In particular, safe-list tail recursion works
when `SafeListAccessProvider` is present; the tail is scoped to the success branch
and is not registered as an unbounded fresh source. Arbitrary AST presets are opaque
to the base-witness phase, so a hidden recursive preset cannot justify recursion.

Generation metadata records `function_defs`, including the base/body signatures,
De Bruijn self index, provider capabilities, capabilities actually observed, and
remaining nesting fuel. It also records `functiondef_observations`, separating
mere self references from maximal application spines genuinely headed by self.
FunctionDef generation currently requires the default
`WeightedDerivationSampler` and `DefaultArgumentPlanner`; unsupported replacements
fail explicitly. A custom materializer must delegate FunctionDef derivations to
`DefaultAstMaterializer` or implement the corresponding qualified procedural
dispatch. `generation_provider_applicable` is the cheap query-level preflight for
custom option providers; returning `false` must never hide a valid option.
Evaluation of generated recursion must still use an external step, time, or resource
limit.
"""
module Procedural

import Random
import ..CanonicalGeneration

using ..Trees
using ..ASTs.NodeASTs
using ..ASTs.LexicalContexts: BinderId
import ..ASTs.LexicalContexts: child_role, visible_binders
using ..TypeSystem
using ..Reduces
using ..Utils: Err

include("nodes.jl")
include("contextual.jl")
include("observations.jl")
include("evaluation.jl")
include("lowering.jl")
include("generation.jl")

export ScopeNode, FunctionDefNode
export NonFunctionDefinitionError
export BinderId, child_role, visible_binders
export functiondef_recursion_observations, functiondef_recursion_summary
export fixed_point_witness, lower_procedural, infer_procedural, reduce_procedural!
export PreparedProcedural, prepare_procedural, run_procedural, evaluate_procedural
export ProceduralEvaluationError, ProceduralStepLimitError, ProceduralNonValueError
export ProceduralUnboundIndexError, ProceduralMissingConstantError
export ProceduralEvaluationCycleError
export FunctionDefCandidate, FunctionDefProvider, AcceptFunctionDef
export DirectKnotFilter, GenuineSelfCallFilter
export ProceduralGenerationVerificationError
export generate_procedural, generate_procedural_scoped
export generate_finite_procedural_scoped, generate_recursive_functiondef_scoped

end # module Procedural
