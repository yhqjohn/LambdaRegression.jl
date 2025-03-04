"""
    TypedSearch

Explicitly typed λ terms for bounded occurrence search, with capture-avoiding
replacement and an ordinary-AST import/export boundary. Types are interned once;
the compact term checker needs no general inference or unification.

Prepare source evidence with `prepare_source(tree, type_context, target)`, select
`source_hole(source, path)`, and call `import_hole(hole, Language(TypeTable()))`.
Use the same language to vary and export the result with `export_hole`. For a
shared task root, use `prepare_root_hole(source, language, type_context; arity)`
and `import_root_body` for further ancestors with the same task inputs and
constant meanings. Local holes keep one fixed ancestor and external anchor.
Full source checking belongs at that export boundary. `search_terms` runs a bounded
age-replacement population phase using a caller-supplied objective. Archive
feedback is separate from `search_nsga2`, an opt-in generational alternative
that selects parent/offspring survivors by training loss and written complexity.
That alternative uses rank/crowding mating and no archive feedback by default;
frequency pressure and Metropolis acceptance remain unsupported. Invalid
objective calls still cost budget.
`search_terms` / `search_populations` provide age replacement; `search_nsga2`
provides generational Pareto survival. Both accept the bounded proposal toolbox
and independent initialisation. Their entry-point docstrings define budgets,
selection, archive feedback, and optional diagnostics; these policies need not
share a universal search loop.

Output-aware objectives return `(loss=..., outputs=vector)`. NSGA-II can retain
those observed outputs within one invocation and classify them independently
of how parent, survivor, or historical-material selection uses the classes.
The default empirical classifier compares finite Float64 bit patterns on one
fixed batch. Classification is not semantic equivalence and never transfers a
score between different candidates. Ordinary η proposals remain scored edits,
not privileged normalisation or evaluator rewrites.

Sparse observers and `TypedSearchTrace` provide passive diagnostics without
search-RNG draws. Trace chunks can be read incrementally; detailed tracing is
opt-in and is not part of the search representation. See the entry-point and
trace docstrings for retention, lifecycle, and snapshot options.
For direct compact execution, `prepare_typed(term, language)` and `run_typed` use supplied
leaf bindings and a shared call-by-need budget; performance is still being
measured. Complete-space scheduling remains the caller's responsibility. For
bounded multi-island phases, `search_populations` shares one scored seed set,
one best-per-written-size archive, and one total objective budget across
persistent age-replacement populations; `chunk_attempts` controls raw-attempt
round-robin granularity. Its result retains per-population members and
statistics while computing one final frontier from the shared archive.
Unsupported source regions return `Err`; the complete AST layer retains them.
"""
module TypedSearch

using ..Utils: Err
using ..TypeSystem
using ..ScalarSearch: _RegularizedFrequency, _regularized_move_window!,
    _regularized_normalize!, _regularized_frequency,
    _regularized_acceptance_probability
using TestItems
using Serialization

include("terms.jl")
include("prepared_sites.jl")
include("source.jl")
include("projection.jl")
include("donors.jl")
include("root_interfaces.jl")
include("growth.jl")
include("simplification.jl")
include("representation_moves.jl")
include("proposal_toolbox.jl")
include("case_selection.jl")
include("pareto_selection.jl")
include("search.jl")
include("evaluation_state.jl")
include("behavior_groups.jl")
include("nsga2_search.jl")
include("trajectory.jl")
include("evaluation.jl")

export SourceAnalysis, SourceHole, SourceBoundaryError
export prepare_source, source_hole, import_hole, export_hole
export SourceProgram, register_program!, admit_program_donor!, expand_program_donors
export written_complexity
export simplify_term
export ScoredTerm, TypedSearchResult, TypedPopulationResult
export search_terms, search_populations
export search_nsga2
export EvaluationState, evaluation_outputs, ExactOutputClassifier, classify_outputs

end # module TypedSearch
