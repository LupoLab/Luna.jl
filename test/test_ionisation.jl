import Test: @test, @testset, @test_throws
using Luna
import NumericalIntegration: integrate, SimpsonEven
import Logging: with_logger, NullLogger

E = collect(range(1e9, 1e11; length=32))
@test Ionisation.ionrate_ADK(:He, E) == Ionisation.ionrate_ADK(:He, -E)

@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1e10; dipole_corr=false), 2*1.40432138471583e-5, rtol=1e-3)
@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1.3e10; dipole_corr=false), 2*0.04517809797503506, rtol=1e-3)


@testset "cycle-averaging $gas" for gas in (:He, :Ar, :Xe)

    bsi = Tools.field_to_intensity(
        Ionisation.barrier_suppression(
            PhysData.ionisation_potential(gas),
            1))
    isy = 10 .^ collect(16:0.1:log10(bsi))
    E0 = Tools.intensity_to_field.(isy)

    t = collect(range(0, 2π; length=2^12))
    Et = sin.(t)

    adk_avg = zero(E0)
    rf! = Ionisation.ionrate_fun!_ADK(gas)
    out = zero(Et)
    for (idx, E0i) in enumerate(E0)
        rf!(out, E0i*Et)
        adk_avg[idx] = 1/2π * integrate(t, out, SimpsonEven())
    end

    adk_avg_kw = Ionisation.ionrate_ADK.(gas, E0; cycle_average=true)
    @test all(isapprox.(adk_avg, adk_avg_kw; rtol=0.05))
end

gas = :He

λ0 = 800e-9

Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
    Interface.prop_capillary_args(100e-6, 1, gas, 1;
    λ0, τfwhm=10e-15, energy=1e-6,
    λlims=(200e-9, 4e-6), trange=0.5e-12)
end

plasma = transform.resp[2]
ir = plasma.ratefunc

println("Here!")
ir2 = Ionisation.ionrate_fun!_PPTaccel(gas, λ0)
