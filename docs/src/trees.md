# Trees Module

```@meta
CurrentModule = LambdaRegression.Trees
```

```@docs
Trees
```

## Core Interface

### Abstract Tree Interface

```@docs
AbstractTree
NodeIndex
root
setroot!
subtree
children
setchild!
```

### Default Methods

```@docs
arity
isleaf
```

### Tree Implementations

```@docs
NodeTree
Node
```

### Traversal Interface

```@docs
AbstractTraverseState
StateBag
getstate
traverse
Leave
Break
```

### Built-in States

```@docs
DepthState
```

### Utility Functions

```@docs
requires
init
enter!
leave!
build_state_bag
default_states
``` 