# Recursive Function Definitions

`FunctionDef` is a first-class term in the common lambda language. It stores
one body and introduces one anonymous monomorphic self binder; it adds no
special type.

```text
P ::= … | FunctionDef(P)
let x = N in M  ≡  (λx. M) N
```

The self binder is `#0` at the body root and shifts beneath ordinary lambdas or
nested definitions. Sequential let bindings use nested lambda applications;
the bound right-hand side is generalized at its own HM boundary.

For type `F`, check the body under `self : F` and require a function head:

```text
headμ(F) = Fun       Σ ; Γ · F ⊢ G : F
                         ----------------
                         Σ ; Γ ⊢ FunctionDef(G) : F
```

The self binding is monomorphic. An enclosing HM boundary may generalize the
completed definition, but recursion itself is not polymorphic.

The compatibility form is:

```text
Lower(FunctionDef(G)) = Y (λself. Lower(G))
Y : ∀α. (α → α) → α
```

Lowering preserves capture-free indices and nested self binders. A self-free
body may omit the fixed-point wrapper after rebasing its unused slot. Native
recursion and lowering agree on pure values that reach a result; resource
limits remain separate from typing.

Generation requires an acyclic base witness before adding self, then constructs
a bounded body without unfolding the fixed point. Typed mutation and crossover
preserve the self binder and visible binder context when moving subtrees.
Definitions may be observed as self references or genuine self calls, and a
well-typed recursive term may still diverge.
