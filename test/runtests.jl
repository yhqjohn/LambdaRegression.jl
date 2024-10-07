using LambdaRegression, Test, SafeTestsets

@safetestset "Test parsing" begin
    include("test_parser.jl")
end

@safetestset "Test debrujin" begin
    include("test_debrujin.jl")
end