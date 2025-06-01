# Canonical Generation

Canonical generation first establishes finite availability and then constructs
a witness for:

```text
T ::= κ | T → T | List T | Array[n, T]
```

The type universe, length universe, and typed contexts are finite and closed.
Generalized demands are rigidified with fresh query atoms; monomorphic sources
cannot invent values of those atoms.

`Known(G, T)` is saturated from variables, constants, lambda abstraction,
closed arrays, fixed constructors, and finite applications of scoped function
binders. `TOLIST` and `AMAP` require grounded array sources. `CONS`, folds,
and recursive functions are guarded expansions.

Each availability fact carries finite evidence for its premises. Evidence may
be reused to prepare one-step construction plans, but it does not authorize an
ungrounded cycle or require a shortest witness. Decision completeness,
sampling support, and construction termination are separate properties.

`FunctionDef` first requires an acyclic base witness with recursive providers
disabled, then adds one monomorphic self binder and constructs its body under a
shared finite budget. `AFOLDL` likewise requires a known source and
accumulator; nested folds need an explicit decreasing resource measure.

The workflow is: close the finite universe, saturate `Known`, reject unknown
targets, select a plan with known premises, complete its children under the
chosen bounded policy, and lower explicit binders to indices. A completion
policy may bound nodes, depth, fuel, or provider use; these controls do not
weaken the finite impossibility decision.

Optional guarded-family balancing redistributes sampling weight among
constructor families while preserving total mass and legal construction routes.
It changes sampling preference, not availability, typing, or completion evidence.
The default is disabled.

Bounded extensions can construct applications with generated finite heads
and configure sampling weights and initialization selection independently.
They share depth, node, and provider-attempt limits with ordinary generation;
failed proposals can refund node charges without restoring spent attempts.
They preserve binder requirements and do not add core availability evidence.
