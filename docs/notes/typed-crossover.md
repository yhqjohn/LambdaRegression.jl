# Typed crossover

Typed crossover transports a lambda fragment between typed programs while
preserving the target site's lexical and type contract.

A target site contains an output demand, outer parameters whose identities must
remain fixed, optional variables that the proposal may solve, and a visible
lexical context. The donor signature includes every non-local variable used by
the fragment. All occurrences of one donor binding map to one target binder;
polymorphic target binders may be instantiated per occurrence, while
monomorphic binders retain shared type state.

Compatibility is directed: the target demand must be an instance of the donor
offer. Donor-wide variables remain shared across every occurrence. Local
occurrence variables are hidden, and rigid variables remain scoped to their
occurrence. These rules prevent capture and preserve dependencies between
separate uses of the same binding.

Closed donors are accepted only when their payload and signature contain every
resource they need. A pointer into an old environment is not a closure. Open
donors are accepted only when the combined output and occurrence constraints
have a valid scoped solution.

After transport, the complete offspring is checked again for typing and
capture safety. This final whole-program check is essential: a typable child
can still have lost the original target demand or changed a shared binding.
