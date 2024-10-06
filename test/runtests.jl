using LambdaRegression, Test, SafeTestsets

@safetestset "Test parsing" begin
    include("test_parser.jl")
end