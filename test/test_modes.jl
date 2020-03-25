import Test: @test
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import Luna: Modes, Maths, Capillary

a = 100e-6
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
r = collect(range(0, a, length=2^16))
unm = besselj_zero(0, 1)
Er = besselj.(0, unm*r/a)
η = Modes.overlap(m, r, Er; dim=1)
@test η[1] ≈ 1
unm = besselj_zero(0, 2)
Er = besselj.(0, unm*r/a)
η = Modes.overlap(m, r, Er; dim=1)
@test isapprox(η[1], 0, atol=1e-18)
fac = collect(range(0.3, stop=0.9, length=128))
ηn = zero(fac)
r = collect(range(0, 4a, length=2^16))
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced, m=1)
for i in eachindex(fac)
    w0 = fac[i]*a
    Er = Maths.gauss(r, w0/sqrt(2))
    ηn[i] = Modes.overlap(m, r, Er, dim=1)[1]
end
@test 0.63 < fac[argmax(ηn)] < 0.65