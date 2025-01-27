# Type System

The language combines locally nameless lambda terms with rank-1
Hindley–Milner inference and contractive equirecursive types.

```text
M, N ::= x | #i | c | λx. M | M N
let x = N in M  ≡  (λx. M) N
```

```text
τ ::= α | κ | τ → τ | τ × τ | τ + τ
    | List τ | Array[ℓ, τ] | μ α. τ
ℓ ::= n | αℓ
σ ::= τ | ∀α₁ … αₙ. τ
```

Recursive binders are contractive: every return to the binder crosses a
constructor. Equirecursive equality identifies a recursive type with its
unfoldings when their observed constructors and parameters agree. The written
shape, binder names, and sharing layout need not be identical.

`Γ` maps local variables or indices to schemes and `Σ` maps constants to
closed types or explicit schemes. Ordinary scheme use freshens quantified
variables. Inference variables are query-local; rigid parameters used by a type
query cannot be bound as ordinary inference variables.

Equality compares corresponding constructor observations with finite cycle
detection. Unification decomposes matching constructors, binds only flexible
inference variables, and rejects constructor-free cycles. A failed unification
must leave the surrounding inference state unchanged. Let generalization
quantifies only variables absent from the surrounding environment.

Scheme compatibility is symmetric fresh-instance unifiability. A directed
instance query keeps the requested scheme rigid while specializing the offered
scheme. These relations are distinct; compatibility is not an instance order.

Algorithm W infers variables and constants by fresh instantiation, functions by
introducing an argument type, applications by unification, and let-like forms
by generalization. It returns a principal rank-1 scheme modulo equirecursive
equality, or a finite typing failure.

Products, sums, lists, and fixed-length arrays remain ordinary constructors;
their concrete operations are supplied by the constant context.
