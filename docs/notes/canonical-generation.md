# Canonical Generation

Canonical generation separates finite inhabitation from recursive decoration.
Its closed fragment is:

```text
T ::= κ | T → T | List T | Array[n, T]
```

For a generalized demand `∀α₁ … αₘ. T`, replace the variables with distinct
rigid query atoms, solve the closed demand, then discharge them. Rigid atoms
are not concrete values, flexible variables, or factory sources.

`Known(G, T)` is the least fixed point of finite typed rules. Ordinary rules
include variables, constants, lambda abstraction, `NIL`, closed array sources,
fixed array constructors, and applications of available scoped function
binders. `TOLIST` and `AMAP` are anchored: their source array must already be
known. `CONS`, folds, identity, and conditionals are guarded expansions.

```text
Known(G, Array[n,A])                 Known(G, A → B)
--------------------------           ----------------------
Known(G, List A) [TOLIST]             Known(G, Array[n,B]) [AMAP]
```

Every premise and conclusion belongs to the finite universe. A target absent
from the saturated relation fails finitely. A target present in the relation
may be sampled through a finite witness, while guarded construction uses an
explicit finite resource bound.

The generator emits explicit-binder syntax first and lowers binders afterward.
The decision relation does not promise a unique or shortest witness, and
guarded expansion cannot turn an ungrounded recursive cycle into availability.
