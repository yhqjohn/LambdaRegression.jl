# Canonical Generation

The finite canonical core decides closed targets in:

```text
T ::= κ | T → T | List T | Array[n, T]
```

It uses a finite closed type and length universe. Generalized demands are
rigidified with fresh query atoms; those atoms may be matched by scoped binders
or polymorphic schemes, but cannot be supplied as monomorphic values.

`Known(G, T)` is saturated from variables, constants, lambda abstraction,
closed array sources, fixed constructors, and finite applications of available
scoped function binders. `TOLIST` and `AMAP` require grounded array sources.
Folds, `CONS`, identity, and conditionals are guarded expansions rather than
first availability rules.

## Recursive function provider

`FunctionDef` is a guarded provider attached to the finite core. For target
function type `F`, it first requires a finite acyclic base witness while
recursive providers and fixed-point sources are disabled. It then introduces
one anonymous monomorphic self binder `self : F` and constructs a body of type
`F` under the typed context.

The base witness prevents recursion from proving its own first availability.
It does not require the final body to use self, terminate, or be productive.
The provider emits one definition node rather than unfolding a runtime fixed
point. Nested definitions share finite node, depth, provider, and nesting
budgets.

The core rejects impossible targets before guarded construction. Successful
construction emits explicit-binder syntax and lowers it to De Bruijn indices.
Typing and generation remain separate from termination: a well-typed recursive
result may diverge.
