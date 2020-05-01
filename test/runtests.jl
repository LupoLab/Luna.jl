import Test: @test, @test_throws, @testset
import Logging: @info

testdir = dirname(@__FILE__)

import Luna: set_fftw_mode
set_fftw_mode(:estimate)

@testset "All" begin

@testset "Maths" begin
    @info("================= test_maths.jl")
    include(joinpath(testdir, "test_maths.jl"))
end

@testset "PhysData" begin
    @info("================= test_physdata.jl")
    include(joinpath(testdir, "test_physdata.jl"))
end

@testset "Capillary" begin
    @info("================= test_capillary.jl")
    include(joinpath(testdir, "test_capillary.jl"))
end

@testset "Rectangular Modes" begin
    @info("================= test_rect_modes.jl")
    include(joinpath(testdir, "test_rect_modes.jl"))
end

@testset "ODE Solver" begin
    @info("================= test_rk45.jl")
    include(joinpath(testdir, "test_rk45.jl"))
end

@testset "Ionisation" begin
    @info("================= test_ionisation.jl")
    include(joinpath(testdir, "test_ionisation.jl"))
end

@testset "Output" begin
    @info("================= test_output.jl")
    include(joinpath(testdir, "test_output.jl"))
end

@testset "Multimode" begin
    @info("================= test_multimode.jl")
    include(joinpath(testdir, "test_multimode.jl"))
end

@testset "Polarisation" begin
    @info("================= test_polarisation.jl")
    include(joinpath(testdir, "test_polarisation.jl"))
    @info("================= test_polarisation_field.jl")
    include(joinpath(testdir, "test_polarisation_field.jl"))
    @info("================= test_polarisation_env.jl")
    include(joinpath(testdir, "test_polarisation_env.jl"))
end

@testset "Tools" begin
    @info("================= test_tools.jl")
    include(joinpath(testdir, "test_tools.jl"))
end

@testset "Utils" begin
    @info("================= test_utils.jl")
    include(joinpath(testdir, "test_utils.jl"))
end

@testset "Gradients" begin
    @info("================= test_gradient.jl")
    include(joinpath(testdir, "test_gradient.jl"))
end

@testset "Tapers" begin
    @info("================= test_tapers.jl")
    include(joinpath(testdir, "test_tapers.jl"))
end

@testset "Scans" begin
    @info("================= test_scans.jl")
    include(joinpath(testdir, "test_scans.jl"))
end

@testset "Raman" begin
    @info("================= test_raman.jl")
    include(joinpath(testdir, "test_raman.jl"))
end

@testset "Kerr" begin
    @info("================= test_kerr.jl")
    include(joinpath(testdir, "test_kerr.jl"))
end

@testset "LinearOps" begin
    @info("================= test_linops.jl")
    include(joinpath(testdir, "test_linops.jl"))
end

@testset "Modes" begin
    @info("================= test_modes.jl")
    include(joinpath(testdir, "test_modes.jl"))
end

@testset "Radial Propagation" begin
    @info("================= test_radial.jl")
    include(joinpath(testdir, "test_radial.jl"))
end

@testset "Full 3D Propagation" begin
    @info("================= test_full_freespace.jl")
    include(joinpath(testdir, "test_full_freespace.jl"))
end

@testset "Antiresonant modes" begin
    @info("================= test_antiresonant.jl")
    include(joinpath(testdir, "test_antiresonant.jl"))
end

@testset "Fields" begin
    @info("================= test_fields.jl")
    include(joinpath(testdir, "test_fields.jl"))
end

@testset "Processing" begin
    @info("================= test_processing.jl")
    include(joinpath(testdir, "test_processing.jl"))
end

end