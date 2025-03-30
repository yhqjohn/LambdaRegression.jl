# LambdaRegression.jl

A Julia library for typed lambda-calculus programs and symbolic regression.

The library provides type inference, typed program generation, mutation and
crossover, recursive function definitions, program evaluation, scalar and typed
local search, and composable search workflows. It is under development; module
docstrings describe supported operations and constraints.

## Development

From the repository directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Program evaluation

```julia
using LambdaRegression
using LambdaRegression.ASTs.NodeASTs: AbsNode, AstTree, IndexNode
using LambdaRegression.Programs: Program

program = Program(AstTree(AbsNode(:x, IndexNode(0))))
evaluate = prepare_default_evaluator(program; input_slots=[:x])
@assert evaluate(2.0) == 2.0
```

Search components are exposed through `Regression`, `ScalarSearch`,
`TypedSearch`, and `Workflows`. See the module docstrings in [src](src).

## Design notes

Concise descriptions of the library's core concepts are in [docs/notes](docs/notes).

## Tests

Run `Pkg.test()` from the activated repository. Tests include test items embedded
in the source files.

## License

MIT; see [LICENSE](LICENSE).
