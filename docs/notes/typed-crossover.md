# Typed crossover

Typed crossover transports a lambda fragment between programs while preserving
lexical bindings, shared type variables, and the target site's output demand.

Compatibility is directed: the target demand must be an instance of the donor
offer. The donor signature lists every non-local free variable of the fragment.
All occurrences of one donor binding map to one visible target binder.
Polymorphic binders may be instantiated separately, while monomorphic binders
retain one shared type state.

Donor-wide variables remain shared across all occurrences. Occurrence-local
variables stay hidden, and rigid variables stay within their occurrence scope.
The combined output and occurrence constraints must preserve those scopes;
otherwise the exchange is rejected. A closed donor is valid only when its
payload and signature carry every resource it needs.

Sequential let-like bindings use ordinary application and abstraction
occurrences. Explicit function definitions remain ordinary monomorphic
dependencies, with their role restrictions applied after type retrieval.

After transport, source syntax is reconstructed hygienically and the complete
offspring is checked for typing and capture safety. Typability alone does not
show that the original demand survived, so the relevant target contract is
checked separately when the exchange is published.
