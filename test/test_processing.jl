import Test: @test, @testset
import FFTW
using Luna
import Luna: settings
import Luna.PhysData: wlfreq
import NumericalIntegration: integrate

@testset "normalisation" begin
λ0 = 800e-9
τfwhm = 30e-15
energy = 1e-3
ω0 = wlfreq(λ0)


rg = Grid.RealGrid(1, λ0, (200e-9, 3000e-9), 0.5e-12)
rFT = FFTW.plan_rfft(similar(rg.t), flags=settings["fftw_flag"])
eg = Grid.EnvGrid(1, λ0, (200e-9, 3000e-9), 0.5e-12)
eFT = FFTW.plan_fft(similar(eg.t), flags=settings["fftw_flag"])

itr = ((rg, rFT), (eg, eFT))

for ii = 1:2
    grid, FT = itr[ii]
    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
    energy_t, energy_ω = Fields.energyfuncs(grid)
    Eω = input(grid, FT)
    @test energy_ω(Eω) ≈ energy

    ω, Iω = Processing.getIω(Processing.getEω(grid, Eω)..., :ω)
    f, If = Processing.getIω(Processing.getEω(grid, Eω)..., :f)
    λ, Iλ = Processing.getIω(Processing.getEω(grid, Eω)..., :λ)

    @test integrate(ω, Iω) ≈ energy
    @test integrate(f, If) ≈ energy
    @test isapprox(integrate(λ, Iλ), energy, rtol=1e-4)

    t, Et = Processing.getEt(grid, Eω)
    Pp = 0.94*energy/τfwhm # approximate peak power of gaussian pulse
    @test isapprox(maximum(abs2.(Et)), Pp, rtol=1e-3)
end
end

@testset "arrivaltime" begin
λ0 = 800e-9
ω0 = wlfreq(λ0)

# field
grid = Grid.RealGrid(1, λ0, (200e-9, 3000e-9), 0.5e-12)
Et = zero(grid.t)
It = Maths.gauss.(grid.t, fwhm=20e-15, x0=5e-15)
@. Et += sqrt(It)*cos(grid.t*ω0)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=-5e-15)
@. Et += sqrt(It)*cos(grid.t*ω0*2)
Eω = FFTW.rfft(Et)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9)), 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9)), -5e-15, rtol=1e-8)
δt = grid.t[2] - grid.t[1]
@test abs(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9); method=:peak) - 5e-15) < δt
@test abs(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9); method=:peak) + 5e-15) < δt

Et = hcat(zero(grid.t), zero(grid.t))
It = Maths.gauss.(grid.t, fwhm=20e-15, x0=5e-15)
Et[:, 1] .+= sqrt.(It).*cos.(grid.t*ω0)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=-5e-15)
Et[:, 1] .+= sqrt.(It).*cos.(grid.t*ω0*2)

It = Maths.gauss.(grid.t, fwhm=20e-15, x0=-5e-15)
Et[:, 2] .+= sqrt.(It).*cos.(grid.t*ω0)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=5e-15)
Et[:, 2] .+= sqrt.(It).*cos.(grid.t*ω0*2)
Eω = FFTW.rfft(Et, 1)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9))[1], 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9))[2], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9))[1], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9))[2], 5e-15, rtol=1e-8)

# envelope
grid = Grid.EnvGrid(1, λ0, (200e-9, 3000e-9), 0.5e-12)
Et = complex(zero(grid.t))
It = Maths.gauss.(grid.t, fwhm=20e-15, x0=5e-15)
@. Et += sqrt(It)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=-5e-15)
@. Et += sqrt(It)*exp(1im*grid.t*ω0)
Eω = FFTW.fft(Et)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9)), 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9)), -5e-15, rtol=1e-8)
δt = grid.t[2] - grid.t[1]
@test abs(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9); method=:peak) - 5e-15) < δt
@test abs(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9); method=:peak) + 5e-15) < δt

Et = hcat(complex(zero(grid.t)), complex(zero(grid.t)))
It = Maths.gauss.(grid.t, fwhm=20e-15, x0=5e-15)
Et[:, 1] .+= sqrt.(It)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=-5e-15)
Et[:, 1] .+= sqrt.(It).*exp.(1im*grid.t*ω0)

It = Maths.gauss.(grid.t, fwhm=20e-15, x0=-5e-15)
Et[:, 2] .+= sqrt.(It)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=5e-15)
Et[:, 2] .+= sqrt.(It).*exp.(1im*grid.t*ω0)
Eω = FFTW.fft(Et, 1)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9))[1], 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(600e-9, 1000e-9))[2], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9))[1], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, bandpass=(300e-9, 500e-9))[2], 5e-15, rtol=1e-8)
end

@testset "specres" begin
# field grid
grid = Grid.RealGrid(1.0, 800e-9, (160e-9, 3000e-9), 30e-12)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*2e-9, x0=PhysData.wlfreq(grid.referenceλ))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/367e-9^2*20e-9, x0=PhysData.wlfreq(367e-9))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/2000e-9^2*3e-9, x0=PhysData.wlfreq(2000e-9)))
# test energy integral is correct
ωp, Eωp = Processing.getEω(grid, Eω)
λg, Pλ = Processing.getIω(ωp, Eωp, :λ, resolution=10e-9, specrange=(170e-9, 2900e-9))
@test isapprox(Fields.energyfuncs(grid)[2](Eω) / integrate(λg, Pλ), 1.0, rtol=1e-10)
Fg, Pf = Processing.getIω(ωp, Eωp, :f, resolution=5e12)
@test isapprox(Fields.energyfuncs(grid)[2](Eω) / integrate(Fg, Pf), 1.0, rtol=1e-10)
# test resolution
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*0.1e-9, x0=PhysData.wlfreq(grid.referenceλ)))
ωp, Eωp = Processing.getEω(grid, Eω)
for res in (2e-9, 10e-9, 20e-9)
    λg, Pλ = Processing.getIω(ωp, Eωp, :λ, resolution=res, specrange=(170e-9, 2900e-9))
    @test isapprox(Maths.fwhm(λg, Pλ), res, rtol=1e-2)
end
for res in (1e12, 5e12, 10e12)
    Fg, Pf = Processing.getIω(ωp, Eωp, :f, resolution=res)
    @test isapprox(Maths.fwhm(Fg, Pf), res, rtol=1e-2)
end
# envelope grid
grid = Grid.EnvGrid(1.0, 800e-9, (160e-9, 3000e-9), 30e-12)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*2e-9, x0=PhysData.wlfreq(grid.referenceλ))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/367e-9^2*20e-9, x0=PhysData.wlfreq(367e-9))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/2000e-9^2*3e-9, x0=PhysData.wlfreq(2000e-9)))
# test energy integral is correct
ωp, Eωp = Processing.getEω(grid, Eω)
λg, Pλ = Processing.getIω(ωp, Eωp, :λ, resolution=10e-9, specrange=(170e-9, 2900e-9))
@test isapprox(Fields.energyfuncs(grid)[2](Eω) / integrate(λg, Pλ), 1.0, rtol=1e-11)
Fg, Pf = Processing.getIω(ωp, Eωp, :f, resolution=5e12)
@test isapprox(Fields.energyfuncs(grid)[2](Eω) / integrate(Fg, Pf), 1.0, rtol=1e-11)
# test resolution
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*0.1e-9, x0=PhysData.wlfreq(grid.referenceλ)))
ωp, Eωp = Processing.getEω(grid, Eω)
for res in (2e-9, 10e-9, 20e-9)
    λg, Pλ = Processing.getIω(ωp, Eωp, :λ, resolution=res, specrange=(170e-9, 2900e-9))
    @test isapprox(Maths.fwhm(λg, Pλ), res, rtol=1e-2)
end
for res in (1e12, 5e12, 10e12)
    Fg, Pf = Processing.getIω(ωp, Eωp, :f, resolution=res)
    @test isapprox(Maths.fwhm(Fg, Pf), res, rtol=1e-2)
end
end

@testset "intensity autocorrelation" begin
    grid = Grid.RealGrid(1.0, 800e-9, (160e-9, 3000e-9), 1e-12)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    input = Fields.GaussField(λ0=800e-9, τfwhm=30e-15, energy=1e-6)
    Eω = input(grid, FT)
    Et = FT \ Eω
    IAC = Processing.intensity_autocorrelation(Et, grid)
    # Gaussian IAC width is sqrt(2)*FWHM 
    @test isapprox(Maths.fwhm(grid.t, IAC), sqrt(2)*30e-15, rtol=3e-6)
    input = Fields.SechField(λ0=800e-9, τfwhm=30e-15, energy=1e-6)
    Eω = input(grid, FT)
    Et = FT \ Eω
    IAC = Processing.intensity_autocorrelation(Et, grid)
    # Numerical AC width from Mathematics: 1.542811*FWHM
    @test isapprox(Maths.fwhm(grid.t, IAC), 1.542811*30e-15, rtol=1e-6)
end

@testset "field autocorrelation" begin
    grid = Grid.EnvGrid(1.0, 800e-9, (160e-9, 3000e-9), 1e-12)
    Δω = 2π*PhysData.c/800e-9^2*10e-9
    Eω = sqrt.(Maths.gauss.(grid.ω, Δω, x0=PhysData.wlfreq(grid.referenceλ)))
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x)
    Et = FFTW.fftshift(FT \ Eω)
    τc = Processing.coherence_time(grid, Et)
    # Gaussian spectrum with natural width Δω has coherence time of 2*sqrt(log(2))/Δω (analytic)
    @test isapprox(τc, 2*sqrt(log(2))/Δω, rtol=6e-6)
    grid = Grid.RealGrid(1.0, 800e-9, (160e-9, 3000e-9), 1e-12)
    Δω = 2π*PhysData.c/800e-9^2*10e-9
    Eω = sqrt.(Maths.gauss.(grid.ω, Δω, x0=PhysData.wlfreq(grid.referenceλ)))
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Et = FFTW.fftshift(FT \ Eω)
    τc = Processing.coherence_time(grid, Et)
    # Gaussian spectrum with natural width Δω has coherence time of 2*sqrt(log(2))/Δω (analytic)
    @test isapprox(τc, 2*sqrt(log(2))/Δω, rtol=5e-8)
end

@testset "AutoWindow" begin
    λ0 = 800e-9 
    grid = Grid.RealGrid(1, λ0, (100e-9, 3000e-9), 0.5e-12)
    FT = FFTW.plan_rfft(similar(grid.t), flags=settings["fftw_flag"])
    Eω1 = Fields.GaussField(λ0=λ0, τfwhm=10e-15, energy=100e-6)(grid, FT)
    Eω2 = Fields.GaussField(λ0=500e-9, τfwhm=30e-15, energy=100e-6)(grid, FT)
    Eω3 = Fields.GaussField(λ0=200e-9, τfwhm=3e-15, energy=10e-6)(grid, FT)

    pw = Processing.PeakWindow(0.2, 100e-9, 1200e-9; relative=true)
    Eωw = pw(grid.ω, Eω1)
    @test grid.ω[argmax(abs2.(Eωw))] == grid.ω[argmax(abs2.(Eω1))]

    pw = Processing.PeakWindow(0.2, 100e-9, 700e-9; relative=true)
    Eωw = pw(grid.ω, Eω1+Eω2+Eω3)
    @test grid.ω[argmax(abs2.(Eωw))] == grid.ω[argmax(abs2.(Eω2))]

    pw = Processing.PeakWindow(0.2, 100e-9, 400e-9; relative=true)
    Eωw = pw(grid.ω, Eω1+Eω2+Eω3)
    @test grid.ω[argmax(abs2.(Eωw))] == grid.ω[argmax(abs2.(Eω3))]
    
    pw = Processing.PeakWindow(0.2, 100e-9, 1200e-9; relative=true)
    Eωw = pw(grid.ω, hcat(Eω1, Eω2, Eω3))
    for (ii, Eω) in enumerate((Eω1, Eω2, Eω3))
        λ0 = wlfreq(grid.ω[argmax(abs2.(Eω))])
        l = [0.9, 1.1] .* λ0
        @test l == pw.lims[:, ii]
    end
end

##
@testset "scanproc" begin
    a = 125e-6
    flength = 0.1
    gas = :He

    λ0 = 800e-9
    τfwhm = 10e-15

    λlims = (100e-9, 4e-6)
    trange = 400e-15

    # Scan dimensions:
    energies = collect(range(50e-9, 200e-9; length=8))
    pressures = collect(0.6:0.4:1.4)
    scan = Scan("pressure_energy_example"; energy=energies, pressure=pressures)
    mktempdir() do td
        runscan(scan) do scanidx, energy, pressure
            prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy,
                           λlims, trange, plasma=false, loss=false, scan, scanidx, filepath=td)
        end
        # multi-output scanproc
        λ, Iλ, zstat, max_peakpower = Processing.scanproc(td) do output
            λ, Iλ = Processing.getIω(output, :λ)
            zstat = Processing.VarLength(output["stats"]["z"])
            max_peakpower = maximum(output["stats"]["peakpower"])
            Processing.Common(λ), Iλ[:, end], zstat, max_peakpower
        end
        @test ndims(λ) == 1
        @test size(Iλ) == (length(λ), length(energies), length(pressures))
        @test size(zstat) == (length(energies), length(pressures))
        @test size(max_peakpower) == (length(energies), length(pressures))
        @test eltype(zstat) == Vector{Float64}

        #single-output scanproc
        energy_in = Processing.scanproc(td) do output
            output["stats"]["energy"][1]
        end
        @test size(energy_in) == (length(energies), length(pressures))
        @test all(isapprox.(energy_in[:, 1], energies, rtol=1e-4))

        # single-array-output scanproc
        Iλ2 = Processing.scanproc(td) do output
            local _, Iλ = Processing.getIω(output, :λ, flength)
            Iλ[:, 1]
        end
        @test Iλ == Iλ2

        #string output scanproc
        strings = Processing.scanproc(td) do output
            "a string"
        end
        @test size(strings) == (length(energies), length(pressures))
    end
end

@testset "peak power modal sum" begin
    λ0 = 800e-9
    τfwhm = 10e-15
    pHE11 = 1e8
    pHE12 = 5e7

    pulses = [
        Pulses.GaussPulse(;λ0, τfwhm, power=pHE11, mode=:HE11)
        Pulses.GaussPulse(;λ0, τfwhm, power=pHE12, mode=:HE12)
    ]

    out = prop_capillary(100e-6, 0.2, :He, 1e-3; λ0, pulses, λlims=(70e-9, 4e-6), trange=1e-12, modes=4)

    @test all(isapprox.(Processing.peakpower(out)[1:2, 1], [pHE11, pHE12]; rtol=1e-4))
    @test isapprox(Processing.peakpower(out; sumdims=2)[1], pHE11+pHE12; rtol=1e-4)

    @test isapprox(out["stats"]["peakpower_allmodes"][1], pHE11+pHE12; rtol=1e-4)
end