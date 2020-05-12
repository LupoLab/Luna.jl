import Test: @test, @testset
import FFTW
import Luna: Grid, Processing, Maths, PhysData
import Luna.PhysData: wlfreq

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
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9)), 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9)), -5e-15, rtol=1e-8)
δt = grid.t[2] - grid.t[1]
@test abs(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9); method=:peak) - 5e-15) < δt
@test abs(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9); method=:peak) + 5e-15) < δt

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
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9))[1], 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9))[2], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9))[1], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9))[2], 5e-15, rtol=1e-8)

# envelope
grid = Grid.EnvGrid(1, λ0, (200e-9, 3000e-9), 0.5e-12)
Et = complex(zero(grid.t))
It = Maths.gauss.(grid.t, fwhm=20e-15, x0=5e-15)
@. Et += sqrt(It)
It = Maths.gauss.(grid.t, fwhm=15e-15, x0=-5e-15)
@. Et += sqrt(It)*exp(1im*grid.t*ω0)
Eω = FFTW.fft(Et)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9)), 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9)), -5e-15, rtol=1e-8)
δt = grid.t[2] - grid.t[1]
@test abs(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9); method=:peak) - 5e-15) < δt
@test abs(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9); method=:peak) + 5e-15) < δt

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
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9))[1], 5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(600e-9, 1000e-9))[2], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9))[1], -5e-15, rtol=1e-8)
@test isapprox(Processing.arrivaltime(grid, Eω, λlims=(300e-9, 500e-9))[2], 5e-15, rtol=1e-8)
end

@testset "Eω_to_SEDλ" begin
# field grid
grid = Grid.RealGrid(1.0, 800e-9, (160e-9, 3000e-9), 300e-12)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*2e-9, x0=PhysData.wlfreq(grid.referenceλ))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/367e-9^2*20e-9, x0=PhysData.wlfreq(367e-9))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/2000e-9^2*3e-9, x0=PhysData.wlfreq(2000e-9)))
λg,Pλ = Processing.Eω_to_SEDλ(grid, Eω, (170e-9, 2900e-9), 10e-9)
λgf,Pλf = Processing.Eω_to_SEDλ_fft(grid, Eω, (170e-9, 2900e-9), 10e-9)
spl = Maths.BSpline(λg, Pλ)
@test isapprox(spl.(λgf), Pλf, rtol=7e-4)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*0.1e-9, x0=PhysData.wlfreq(grid.referenceλ)))
for res in (2e-9, 10e-9, 20e-9)
    λg,Pλ = Processing.Eω_to_SEDλ(grid, Eω, (170e-9, 2900e-9), res)
    @test isapprox(Maths.fwhm(λg, Pλ), res, rtol=1e-2)
end
# envelope grid
grid = Grid.EnvGrid(1.0, 800e-9, (160e-9, 3000e-9), 300e-12)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*2e-9, x0=PhysData.wlfreq(grid.referenceλ))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/367e-9^2*20e-9, x0=PhysData.wlfreq(367e-9))
      .+ Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/2000e-9^2*3e-9, x0=PhysData.wlfreq(2000e-9)))
λg,Pλ = Processing.Eω_to_SEDλ(grid, Eω, (170e-9, 2900e-9), 10e-9)
λgf,Pλf = Processing.Eω_to_SEDλ_fft(grid, Eω, (170e-9, 2900e-9), 10e-9)
spl = Maths.BSpline(λg, Pλ)
@test isapprox(spl.(λgf), Pλf, rtol=2e-3)
Eω = (Maths.gauss.(grid.ω, fwhm=2π*PhysData.c/800e-9^2*0.1e-9, x0=PhysData.wlfreq(grid.referenceλ)))
for res in (2e-9, 10e-9, 20e-9)
    λg,Pλ = Processing.Eω_to_SEDλ(grid, Eω, (170e-9, 2900e-9), res)
    @test isapprox(Maths.fwhm(λg, Pλ), res, rtol=1e-2)
end
end
