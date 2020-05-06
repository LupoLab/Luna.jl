import Test: @test, @testset
import FFTW
import Luna: Grid, Processing, Maths, Fields
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

@testset "findpeaks" begin
    grid = Grid.RealGrid(1.0, 800e-9, (160e-9, 3000e-9), 10e-12)
    dt = grid.t[2] - grid.t[1]
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1)
    Eω = FT * x
    fill!(Eω, 0.0)
    positions = (-367e-15, -10e-15, 100e-15, 589e-15)
    widths = (30e-15, 3e-15, 100e-15, 200e-15)
    powers = (1e3, 1e4, 1e2, 2e3)
    for i in 1:length(positions)
        field = Fields.GaussField(λ0=800e-9, τfwhm=widths[i], power=powers[i], τ0=positions[i])
        Eω .+= field(grid, FT)
    end
    Et = FT \ Eω
    It = Fields.It(Et, grid)
    pks = Processing.findpeaks(grid.t, It, threshold=10.0, filterfw=false)
    print(pks)
    for i in 1:length(positions)
        @test isapprox(pks[i].position, positions[i], atol=dt*1.1)
        @test isapprox(pks[i].fw, widths[i], rtol=1e-7, atol=dt*1.1)
        @test isapprox(pks[i].peak, powers[i], rtol=1e-2)
    end
end
