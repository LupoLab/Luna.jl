import Test: @test
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import FFTW
import Luna: Modes, Maths, Capillary, Grid, PhysData, Hankel, NonlinearRHS

a = 100e-6
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
r = collect(range(0, a, length=2^16))
unm = besselj_zero(0, 1)
Er = besselj.(0, unm*r/a)
η = Modes.overlap(m, r, Er; dim=1)
@test abs2(η[1]) ≈ 1
unm = besselj_zero(0, 2)
Er = besselj.(0, unm*r/a)
η = Modes.overlap(m, r, Er; dim=1)
@test isapprox(abs2(η[1]), 0, atol=1e-18)
fac = collect(range(0.3, stop=0.9, length=128))
ηn = zero(fac)
r = collect(range(0, 4a, length=2^16))
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced, m=1)
for i in eachindex(fac)
    w0 = fac[i]*a
    Er = Maths.gauss(r, w0/sqrt(2))
    ηn[i] = abs2.(Modes.overlap(m, r, Er, dim=1)[1])
end
@test 0.63 < fac[argmax(ηn)] < 0.65

fac = 0.45
w0 = fac*a
Er = Maths.gauss(r, w0/sqrt(2))
sum = 0
for mi = 1:10
    m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced, m=mi)
    η = Modes.overlap(m, r, Er, dim=1)[1]
    global sum += abs2(η[1])
end
@test isapprox(sum, 1, rtol=1e-4)

a = 100e-6
q = Hankel.QDHT(a, 128)
grid = Grid.RealGrid(1, 800e-9, (200e-9, 2000e-9), 0.5e-12)
It1 = Maths.gauss(grid.t, fwhm=30e-15)
Et1 = @. sqrt(It1)*cos(2π*PhysData.c/800e-9*grid.t)
Eω1 = FFTW.rfft(Et1)
unm = besselj_zero(0, 1)
Er1 = besselj.(0, unm*q.r/a)'
Etr1 = Et1 .* Er1
It2 = 4*Maths.gauss(grid.t, fwhm=15e-15)
Et2 = @. sqrt(It2)*cos(2π*PhysData.c/400e-9*grid.t)
Eω2 = FFTW.rfft(Et2)
unm = besselj_zero(0, 2)
Er2 = besselj.(0, unm*q.r/a)'
Etr2 = Et2 .* Er2

# et, eω = NonlinearRHS.energy_

Etr = Etr1 .+ Etr2
Eωr = FFTW.rfft(Etr, 1)

m1 = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced, m=1)
m2 = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced, m=2)
Eωm1 = Modes.overlap(m1, q.r, Eωr; dim=2, norm=false)
Eωm2 = Modes.overlap(m2, q.r, Eωr; dim=2, norm=false)
@test Eωm1[:, 1] ≈ Eω1