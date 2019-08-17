import Test: @test, @test_throws, @testset

testdir = dirname(@__FILE__)

@testset "Maths" begin
    include(joinpath(testdir, "test_maths.jl"))
end

@testset "PhysData" begin
    include(joinpath(testdir, "test_physdata.jl"))
end

@testset "Capillary" begin
    include(joinpath(testdir, "test_capillary.jl"))
end

@testset "ODE Solver" begin
    include(joinpath(testdir, "test_rk45.jl"))
end

@testset "Ionisation" begin
    include(joinpath(testdir, "test_ionisation.jl"))
end

@testset "Output" begin
    include(joinpath(testdir, "test_output.jl"))
end