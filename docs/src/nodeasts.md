# NodeASTs Module

```@meta
CurrentModule = LambdaRegression.ASTs.NodeASTs
```

```@docs
NodeASTs
```

## Core Types

### Abstract AST Interface

```@docs
AstNode
AstTree
```

### Lambda Calculus Nodes

```@docs
AppNode
AbsNode
VarNode
ConstNode
IndexNode
LetNode
```

## Symbol Generation

```@docs
SymbolGenerator
generate_fresh_name
```

## Traversal States

```@docs
AncestorsState
ScopeState
AbstractionDepthState
get_abstractions
get_abstraction_depth
```

## Utility Functions

```@docs
collect_free_variables
``` 