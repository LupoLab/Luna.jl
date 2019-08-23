import Test: @test, @test_throws, @testset

@testset "Maths" begin
    include("test_maths.jl")
end

@testset "PhysData" begin
    include("test_physdata.jl")
end

@testset "Capillary" begin
    include("test_capillary.jl")
end

@testset "ODE Solver" begin
    include("test_rk45.jl")
end

@testset "Ionisation" begin
    include("test_ionisation.jl")
end