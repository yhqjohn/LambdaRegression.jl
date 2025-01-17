# Canonical Generation

Canonical generation turns a closed target type and finite typed context into
a finite construction problem. The initial fragment is:

```text
T ::= κ | T → T | List T | Array[n, T]
```

The type universe and length universe are finite and closed. Ordinary context
entries are closed; polymorphic entries are explicit schemes.

The finite core excludes products, sums, recursive targets, local
generalization, length observers, partial list destructors, and heads that
leave open type or length variables. Operations such as `CONS` and `AFOLDL`
may be used only as bounded extensions after a witness already exists.

Define `Known(G, T)` for available closed variable types `G` and target `T`.
Saturate it from variables, constants, lambda abstraction, closed array
sources, fixed array constructors, and source-grounded map/list conversion:

```text
Known(G, A → B)  if  Known(G ∪ {A}, B)
Known(G, T)       if  all premises of a finite core head P₁ … Pₖ → T
                    are known
```

Every premise and conclusion must belong to the finite universe. Thus an
impossible target fails before recursive construction, while a possible target
has finite typed witness choices.

The generator saturates `Known`, samples a successful witness, and emits an
AST with explicit binders before lowering to De Bruijn indices. Core
applications use variables or constants as heads, so core output is beta-normal.
The core decides finite availability; sampling policy and guarded expansion
are separate concerns.
