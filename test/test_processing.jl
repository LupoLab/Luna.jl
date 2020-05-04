import Test: @test, @testset
import FFTW
import Luna: Grid, Processing, Maths
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