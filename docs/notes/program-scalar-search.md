# Alternating program and scalar search

The ordinary typed AST is the authoritative individual. Scalar and
frozen-type representations are specialised views that return ordinary
programs after verified materialisation.

General program search and local search form repeated alternating phases. A
complete-program population changes bindings, sharing, types, and context.
Bounded local phases refine eligible scalar or frozen typed regions. Returned
candidates are rechecked, scored in the enclosing task, and made available to
the next program phase.

Specialised phases are intentionally incomplete. General AST variation can
introduce abstractions, polymorphic uses, recursion, and combinations that a
local language cannot express. Reanalysis may expose new regions later.

Scalar regions carry explicit dependencies and use multi-step population
search. Frozen typed search varies inferred monomorphic occurrences, or
occurrences under a rigid environment, with exact type matching. Lexical
bindings and source dependencies remain explicit.

The coordinator owns scheduling, population replacement, archives, and local
proposal policy. These choices may change without changing source semantics,
contextual scoring, or ordinary-AST ownership.
