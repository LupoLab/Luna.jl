import Test: @test, @testset, @test_throws, @test_broken
using Luna

@testset "step-index fibre" begin
    # single mode fibre at 1030 nm
    a = 5e-6
    NA = 0.08
    flength = 2.0
    fr = 0.18
    τfwhm = 1e-12
    λ0 = 1030e-9
    energy = 10e-9

    m = StepIndexFibre.StepIndexMode(a, NA)
    ω0 = PhysData.wlfreq(λ0)
    @test Modes.neff(m, ω0) ≈ 1.451235217910556

    mac = StepIndexFibre.StepIndexMode(a, NA; accellims=(800e-9, 1250e-9, 100))
    ω0 = PhysData.wlfreq(λ0)
    @test Modes.neff(mac, ω0) ≈ 1.451235217910556
end