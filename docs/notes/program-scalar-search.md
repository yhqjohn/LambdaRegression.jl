# Program-guided scalar search

The ordinary lambda and function-definition AST is the authoritative
individual. It defines program meaning, lexical sharing, inspection,
serialization, and written complexity. Scalar search operates on rebuildable
views of real-valued regions and materialises candidates back into that AST.

The search space has cooperating layers: dense scalar algebra for local
numeric changes, predefined collection structure, and general typed programs
for abstraction, higher-order reuse, recursion, and structural discovery.
These are views of one program, not competing public languages.

A scalar region carries the explicit dependencies needed to evaluate its
alternatives. Its normal objective is the loss of the complete enclosing
program, rather than an independently invented internal target. Scalar
optimisation is multi-step population search; compatible regions may share
material, but a donor's score is not automatically valid in another context.

The specialised representation and its state are derived data. They may be
retained to avoid rebuilding search work, but must be reconstructible from the
program, task, primitive world, and context. Candidates return through a
verified materialisation boundary, while unsupported regions retain the
general evaluator path.
