# Program-guided specialised search

The ordinary typed lambda and function-definition AST remains authoritative.
Specialised search views are rebuildable projections; a candidate joins the
population only after materialisation and checking in the ordinary AST.

The search space adds frozen occurrence-typed structure beside dense scalar
regions and general typed programs. Frozen search starts from an inferred
program and varies finite monomorphic occurrences, or occurrences under an
externally rigid environment, using exact type matching. Unsupported
polymorphic or recursive interiors may remain opaque typed atoms while their
free dependencies stay explicit.

Specialised phases are incomplete evolutionary views, not a second semantic
program representation. Selected candidates return to the ordinary AST, where
general variation can introduce abstraction, polymorphic use, recursion, and
cross-region combinations. Reanalysis can expose eligible regions again.

Scalar search remains contextual and population-based. It may retain local
state and share compatible material, but a score is reusable only when the
candidate and enclosing evaluation conditions match. The materialisation
boundary owns source reconstruction, typing, and capture avoidance; rejected
local movements do not weaken complete-program search.
