# Recursive Function Definitions

`FunctionDef` is a normal recursive term form. It stores one body and adds one
anonymous monomorphic self binder, while retaining ordinary types and binding.

```text
P ::= … | FunctionDef(P)
```

Self is `#0` at the body root and shifts below lambdas or nested definitions.
For type `F`, the body is checked under `self : F`, and `F` must have a
function head modulo equirecursive equality. An enclosing HM boundary may
generalize a completed definition; recursion inside it remains monomorphic.

```text
Lower(FunctionDef(G)) = Y (λself. Lower(G))
```

with `Y : ∀α. (α → α) → α`. Lowering preserves exact binder depth and capture
avoidance; a self-free body may omit the fixed-point wrapper after rebasing.
Native recursion and lowering agree on pure values that terminate.

The generator first obtains a finite acyclic base witness with recursive
providers disabled, then constructs a bounded body. Nested definitions consume
the same finite node, depth, provider, and nesting resources. The fixed point is
not unfolded to establish its first availability, and no complete termination
filter is assumed.

Typed mutation and crossover must either support `FunctionDef` or reject it at
an explicit finite boundary. Structural edits preserve the self binder and
rebase indices. Crossover transports a subtree only with its visible binder
context, so lowering first would lose information needed for safe movement.

Structural observation may distinguish a returned self reference from a real
self call. Resource failure and numerical behavior remain separate from type
failure.
