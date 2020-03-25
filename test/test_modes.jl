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
@test isapprox(η[1], 0, atol=1e-9)