module ASTs

import ..Trees

include("nodeasts.jl")
include("lexical.jl")
include("parse.jl")

# Re-export the NodeASTs module to make it accessible
using .NodeASTs
export NodeASTs

# Re-export child-specific lexical context for core and extension nodes.
using .LexicalContexts
export LexicalContexts

# Export parse entry points from parse.jl
using .Parse
export parse_lambda, @terms

end # module ASTs
