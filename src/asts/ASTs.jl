module ASTs

import ..Trees

include("nodeasts.jl")
include("contexts.jl")
include("parse.jl")
include("cse.jl")

# Re-export the NodeASTs module to make it accessible
using .NodeASTs
export NodeASTs

# Re-export Contexts for external usage
using .Contexts
export Contexts

# Export parse entry points from parse.jl
using .Parse
export parse_lambda, @terms

end # module ASTs