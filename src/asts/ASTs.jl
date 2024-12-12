module ASTs

import ..Trees

include("nodeasts.jl")

# Re-export the NodeASTs module to make it accessible
using .NodeASTs
export NodeASTs

end # module ASTs