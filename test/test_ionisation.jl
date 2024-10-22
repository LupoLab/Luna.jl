import Test: @test, @testset
import Luna: Ionisation, Maths, PhysData, Tools
import NumericalIntegration: integrate, SimpsonEven
import Luna.Modes: hquadrature

@test Ionisation.ionrate_ADK(:He, 1e10) ≈ 1.2416371415312408e-18
@test Ionisation.ionrate_ADK(:He, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:HeB, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:Ar , 7e9) ≈ 2.4422306306649472e-08
@test Ionisation.ionrate_ADK(:Ar , 8e9) ≈ 4.494711488416766e-05

E = collect(range(1e9, 1e11; length=32))
@test Ionisation.ionrate_ADK(:He, E) == Ionisation.ionrate_ADK(:He, -E)

@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1e10), 2*1.40432138471583e-5, rtol=1e-3)
@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1.3e10), 2*0.04517809797503506, rtol=1e-3)

Emin = 1e9
Emax = 1e11
N = 2^9
E = collect(range(Emin, Emax, N))
rate = Ionisation.ionrate_PPT.(:He, 800e-9, E)
ifun(E0) =  E0 <= Emin ? 2 :
            E0 >= Emax ? N : 
            ceil(Int, (E0-Emin)/(Emax-Emin)*N) + 1
ifun2(x0) = x0 <= E[1] ? 2 :
                x0 >= E[end] ? length(E) :
                findfirst(x -> x>x0, E)
spl1 = Maths.CSpline(E, rate, ifun)
spl2 = Maths.CSpline(E, rate, ifun2)
idx1 = ifun2.(E) # calculated indices
idx2 = ifun.(E) # indices found with brute-force method
@test all(idx1 .== idx2)
@test all(spl1.(E) .== spl2.(E))

ratefun! = Ionisation.ionrate_fun!_PPTaccel(:He, 800e-9)
out = similar(E)
ratefun!(out, E)
@test all(isapprox.(out, rate, rtol=1e-2))

outneg = similar(out)
ratefun!(outneg, -E)
@test out == outneg

outneg = similar(out)
ratefun!(outneg, -E)
@test out == outneg

##
@testset "cycle-averaging $gas" for gas in (:He, :Ar, :Xe)

    bsi = Tools.field_to_intensity(
        Ionisation.barrier_suppression(
            PhysData.ionisation_potential(gas),
            1))
    isy = 10 .^ collect(16:0.01:log10(bsi))
    E0 = Tools.intensity_to_field.(isy)

    Ip_au = PhysData.ionisation_potential(gas; unit=:atomic)
    F0_au = (2Ip_au)^(3/2)
    F0 = F0_au*PhysData.au_Efield

    t = collect(range(0, 2π; length=2^12))
    Et = sin.(t)

    adk_avg = zero(E0)
    rf! = Ionisation.ionrate_fun!_ADK(gas)
    out = zero(Et)
    for (idx, E0i) in enumerate(E0)
        rf!(out, E0i*Et)
        adk_avg[idx] = 1/2π * integrate(t, out, SimpsonEven())
    end

    adk = Ionisation.ionrate_ADK.(gas, E0)
    adk_avg_kw = Ionisation.ionrate_ADK.(gas, E0; cycle_average=true)
    @test all(isapprox.(adk_avg, adk_avg_kw; rtol=0.05))
end
