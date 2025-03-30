# Type System

The language uses locally nameless lambda terms, rank-1 Hindley–Milner
inference, and contractive equirecursive types. `FunctionDef` is an ordinary
recursive term extension and introduces no special type constructor.

```text
M, N ::= x | #i | c | λx. M | M N | FunctionDef(M)
let x = N in M  ≡  (λx. M) N
```

```text
τ ::= α | κ | τ → τ | τ × τ | τ + τ
    | List τ | Array[ℓ, τ] | μ α. τ
ℓ ::= n | αℓ
σ ::= τ | ∀α₁ … αₙ. τ
```

Recursive equality identifies guarded `μ` types with compatible unfoldings.
Unification may create guarded recursive equations but rejects naked cycles.
The two operations are finite and preserve the distinction between flexible
inference variables and rigid query parameters.

For schemes `σ = ∀β̄. s` and `ρ = ∀ᾱ. r`, the directed instance relation is:

```text
σ ≼ ρ  iff  ∃θ, dom(θ) ⊆ ᾱ,  θ(r) ≈μ s
```

Only quantified variables of the offered scheme are flexible. Compatibility is
separate symmetric fresh-instance unifiability. Let generalization quantifies
only variables absent from the surrounding environment.

`FunctionDef(G)` checks `G` under one monomorphic self binder of type `F`
and requires `F` to have a function head:

```text
headμ(F) = Fun       Γ · F ⊢ G : F
                         ----------------
                         Γ ⊢ FunctionDef(G) : F
```

The self binder is an ordinary De Bruijn slot. A completed definition may be
generalized outside its body, but recursion inside it is not polymorphic.

Products, sums, lists, and arrays remain ordinary constructors. A dense
N-dimensional array has a nested logical array type; one map or fold acts on
one outer logical layer. Shape arithmetic is outside this type theory.
