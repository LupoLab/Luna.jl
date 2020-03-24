import Test: @test, @test_throws, @testset

testdir = dirname(@__FILE__)

@testset "All" begin

@testset "Maths" begin
    include(joinpath(testdir, "test_maths.jl"))
end

@testset "PhysData" begin
    include(joinpath(testdir, "test_physdata.jl"))
end

@testset "Capillary" begin
    include(joinpath(testdir, "test_capillary.jl"))
end

@testset "Rectangular Modes" begin
    include(joinpath(testdir, "test_rect_modes.jl"))
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

@testset "Multimode" begin
    include(joinpath(testdir, "test_multimode.jl"))
end

@testset "Polarisation" begin
    include(joinpath(testdir, "test_polarisation.jl"))
    include(joinpath(testdir, "test_polarisation_field.jl"))
    include(joinpath(testdir, "test_polarisation_env.jl"))
end

@testset "Tools" begin
    include(joinpath(testdir, "test_tools.jl"))
end

@testset "Utils" begin
    include(joinpath(testdir, "test_utils.jl"))
end

@testset "Gradients" begin
    include(joinpath(testdir, "test_gradient.jl"))
end

@testset "Scans" begin
    include(joinpath(testdir, "test_scans.jl"))
end

end