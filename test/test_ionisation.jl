import Test: @test, @testset
import Luna: Ionisation, Maths

@test Ionisation.ionrate_ADK(:He, 1e10) ≈ 1.2416371415312408e-18
@test Ionisation.ionrate_ADK(:He, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:HeJ, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:Ar , 7e9) ≈ 2.4422306306649472e-08
@test Ionisation.ionrate_ADK(:Ar , 8e9) ≈ 4.494711488416766e-05

E = collect(range(1e9, 1e11; length=32))
@test Ionisation.ionrate_ADK(:He, E) == Ionisation.ionrate_ADK(:He, -E)

@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1e10), 2.4329e-12, rtol=1e-3)
@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1.3e10), 3.0825e-07, rtol=1e-3)

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
@test all(isapprox.(out, rate, rtol=1e-3))

outneg = similar(out)
ratefun!(outneg, -E)
@test out == outneg

# import PyPlot: plt, pygui
# pygui(true)
# k = collect(range(8, stop=12, length=1000))
# E = 10 .^ k
# ppt = Ionisation.ionrate_PPT(:He, 800e-9, E)
# adk = Ionisation.ionrate_ADK(:He, E)

# dat = CSV.read("C:\\Users\\cbrahms\\Documents\\GitHub\\luna\\test\\Ilkov_PPT_He.csv")
# dat = convert(Matrix, dat) # ionrate [1/s] vs field [V/cm]

# import CSV
# plt.figure()
# plt.loglog(E, ppt, "--")
# plt.loglog(E, adk)
# plt.loglog(dat[:, 1].*100, dat[:, 2], "--")