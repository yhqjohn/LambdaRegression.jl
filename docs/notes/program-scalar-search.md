# Program-guided specialised search

LambdaRegression keeps the ordinary lambda and function-definition AST as the
authoritative program. It owns source semantics, lexical sharing,
serialization, and written complexity. Search and execution use derived views
that are checked against explicit contexts.

The architecture separates general typed program search, scalar search for
dense real-valued regions, frozen occurrence-typed search, and execution
preparation with a general fallback. A specialised candidate becomes a
program candidate only after source reconstruction, type checking, capture
avoidance, and contextual scoring.

Program analysis supplies occurrence identities, binding and dependency facts,
typing evidence, and validity inputs. Consumers build their own projections;
scalar regions, frozen occurrences, and executable regions need not share a
physical graph or have a one-to-one relationship.

The repeated coordinator keeps complete programs as scored and inherited
individuals, gives bounded refinement opportunities to specialised views, and
publishes returned material through ordinary program checks. General phases can
introduce structure that local phases cannot express. Population replacement,
archives, donor transfer, and scheduling remain replaceable policies.
