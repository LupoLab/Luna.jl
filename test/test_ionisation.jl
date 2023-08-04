import Test: @test, @testset
import Luna: Ionisation, Maths, PhysData, Tools
import NumericalIntegration: integrate, SimpsonEven

@test Ionisation.ionrate_ADK(:He, 1e10) ≈ 1.2416371415312408e-18
@test Ionisation.ionrate_ADK(:He, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:HeB, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:Ar , 7e9) ≈ 2.4422306306649472e-08
@test Ionisation.ionrate_ADK(:Ar , 8e9) ≈ 4.494711488416766e-05

E = collect(range(1e9, 1e11; length=32))
@test Ionisation.ionrate_ADK(:He, E) == Ionisation.ionrate_ADK(:He, -E)

@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1e10), 1.4130113877738475e-5, rtol=1e-3)
@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1.3e10), 0.04585332982943, rtol=1e-3)

Emin = 1e9
Emax = 1e11
N = 2^10
E = collect(range(Emin, stop=Emax, length=N))
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
##
# this_folder = dirname(@__FILE__)
# import CSV
# import PyPlot: plt, pygui
# pygui(true)
# #### Ilkov et al
# k = collect(range(8, stop=12, length=200))
# E = 10 .^ k
# ppt = Ionisation.ionrate_PPT(:He, 10.6e-6, E)
# ppt_cycavg = Ionisation.ionrate_PPT(:He, 10.6e-6, E, rcycle=false)
# adk = Ionisation.ionrate_ADK(:He, E)

# dat = CSV.read(joinpath(this_folder, "Ilkov_PPT_He.csv"))
# dat = convert(Matrix, dat) # ionrate [1/s] vs field [V/cm]

# plt.figure()
# plt.loglog(E, ppt, label="PPT")
# plt.loglog(E, ppt_cycavg, label="PPT cycle averaged")
# plt.loglog(E, adk, label="ADK")
# plt.loglog(dat[:, 1].*100, dat[:, 2], label="Ilkov et al. PPT")
# plt.xlim(1e9, 1e12)
# plt.ylim(1, 1e18)
# plt.legend()

# ### Chang, Fundamentals of Attosecond Optics
# dat = CSV.read(joinpath(this_folder, "Chang_PPT.csv"))
# dat = convert(Matrix, dat) # ionrate [1/fs] vs intensity [1e14 W/cm^2]
# intensity = range(0.1, 25; length=1000) * 1e18 # W/m^2
# E = Tools.intensity_to_field.(intensity)
# ppt = Ionisation.ionrate_PPT(:He, 390e-9, E)
# ppt_cycavg = Ionisation.ionrate_PPT(:He, 390e-9, E, rcycle=false)
# adk = Ionisation.ionrate_ADK(:He, E)

# s = sortperm(dat[:, 1])

# plt.figure()
# plt.loglog(intensity*1e-18, ppt*1e-15, label="PPT")
# plt.loglog(intensity*1e-18, adk*1e-15, label="ADK")
# plt.loglog(intensity*1e-18, ppt_cycavg*1e-15, label="PPT cycle averaged")
# plt.loglog(dat[s, 1], dat[s, 2], label="Chang PPT")
# plt.ylim(1e-14, 1)
# plt.xlim(extrema(intensity.*1e-18))
# plt.legend()
# plt.xlabel("Intensity (10\$^{14}\$ W/cm\$^2\$)")
# plt.ylabel("Ionisation rate (1/fs)")

### Couairon
# Ip = 12.063 * PhysData.electron
# dat = CSV.read(joinpath(this_folder, "Couairon_PPT.csv"))
# dat = convert(Matrix, dat) # ionrate [1/s] vs intensity [W/cm^2]
# k = collect(range(12, 15, length=500))
# intensity = 10 .^ k .* 1e4 # W/m^2
# E = Tools.intensity_to_field.(intensity)
# ppt = Ionisation.ionrate_PPT(Ip, 800e-9, 1, 1, E)
# ppt_cycavg = Ionisation.ionrate_PPT(Ip, 800e-9, 1, 1, E; rcycle=false)
# adk = Ionisation.ionrate_ADK(Ip, E)

# s = sortperm(dat[:, 1])

# plt.figure()
# plt.loglog(intensity*1e-4, ppt, label="PPT")
# plt.loglog(intensity*1e-4, adk, label="ADK")
# plt.loglog(intensity*1e-4, ppt_cycavg, label="PPT cycle averaged")
# plt.loglog(dat[s, 1], dat[s, 2], label="Couairon PPT")
# plt.ylim(1, 1e17)
# plt.xlim(extrema(intensity.*1e-4))
# plt.legend()
# plt.xlabel("Intensity (W/cm\$^2\$)")
# plt.ylabel("Ionisation rate (1/s)")
# plt.title("O\$_2\$ ionisation at 800 nm")