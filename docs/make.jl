push!(LOAD_PATH,"..")

# Try to load without precompilation to avoid the negative length issue
using Documenter
try
    using LambdaRegression
catch e
    @warn "Failed to load LambdaRegression normally, trying alternative approach: $e"
    # Load module directly without precompilation
    include("../src/LambdaRegression.jl")
    using .LambdaRegression
end

makedocs(
    sitename="LambdaRegression.jl",
    modules=[LambdaRegression],
    pages=[
        "Home" => "index.md",
        "Trees Module" => "trees.md",
        "DfsTrees Submodule" => "dfstrees.md",
        "NodeASTs Module" => "nodeasts.md"
    ],
    format=Documenter.HTML(
        prettyurls=false,
    ),
    remotes=nothing,
    checkdocs=:none,
    warnonly=[:docs_block, :cross_references, :missing_docs]
)