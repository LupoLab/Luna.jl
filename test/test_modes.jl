import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import Luna: Modes, Maths, Capillary

a = 100e-6
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
unm = besselj_zero(0, 1)
r = collect(range(0, a, length=512))
Er = besselj.(0, unm*r/a)
η = Modes.overlap(m, r, Er; dim=1)
println(η)