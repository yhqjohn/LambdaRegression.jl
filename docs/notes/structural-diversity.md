# Structural diversity

`Program` is authoritative. Structural diversity provides disposable views of
current structure and editable regions; it does not own populations, fitness,
selection, historical novelty, or protection budgets.

Descriptions are created in a stable `StructureWorld` containing the relevant
type context, values, input types, target type, and dependency assumptions.
Evidence is valid only for the published program and world from which it was
derived. A changed program or world requires a new description.

The scalar view removes maximal admitted `Float` regions from the displayed
skeleton and replaces them with legal input ports and scalar grammar. A
bindings view retains syntax as a control description. Both views preserve
lexical binder identity, shared versus independent editable bodies, and
uncertainty at recursive or unsupported higher-order boundaries.

Dependency profiles are keyed by actual primitive identity and summarize what
values, shapes, callbacks, callees, or captured environments may influence.
Unknown behaviour remains conservative. Profiles are declared assumptions;
analysis does not execute callbacks or infer equivalence.

Descriptions can support family grouping and representative selection, but
equal structure does not authorize score, type-certificate, or local-state
reuse. Every real candidate still receives contextual validation and scoring.
