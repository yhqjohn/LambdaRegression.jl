# Recursive Function Definitions

`FunctionDef` stores one body and introduces one anonymous monomorphic self
binder in the locally nameless lambda language.

```text
P ::= … | FunctionDef(P)
```

At the body root self is `#0`; below `k` ordinary binders it is `#k`.
Nested definitions push another ordinary binder slot.

If the definition has type `F`, check its body under `self : F` and require a
function head:

```text
headμ(F) = Fun       Σ ; Γ · F ⊢ G : F
                         ----------------
                         Σ ; Γ ⊢ FunctionDef(G) : F
```

A caller-provided polymorphic demand is checked at one rigidified monomorphic
instance. Only the completed definition may be generalized outside its body;
recursion itself is not polymorphic.

The compatibility lowering is:

```text
Lower(FunctionDef(G)) = Y (λself. Lower(G))
Y : ∀α. (α → α) → α
```

Lowering preserves binder depth and capture avoidance. A self-free body may
omit the fixed-point wrapper after rebasing its unused slot. Direct recursion
and lowering have the same pure value semantics, but need not share costs.

Generation first obtains a finite acyclic base witness with recursive providers
disabled, then constructs a bounded body with the self binder. It does not
unfold the fixed point to establish availability. A well-typed definition may
still diverge.
