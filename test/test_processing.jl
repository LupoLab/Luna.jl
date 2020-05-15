import Test: @test, @testset
import FFTW
import Luna: Grid, Processing, Maths, Fields, settings, PhysData
import Luna.PhysData: wlfreq

@testset "normalisation" begin
import NumericalIntegration: integrate
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
    input! = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
    energy_t, energy_ω = Fields.energyfuncs(grid)
    Eω = zeros(ComplexF64, length(grid.ω))
    input!(Eω, grid, energy_t, FT)
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
