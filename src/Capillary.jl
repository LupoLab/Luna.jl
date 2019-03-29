module Capillary
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Luna: Maths, Refraction

const c = Unitful.ustrip(PhysicalConstants.CODATA2014.c)

function β(a, ω, gas::Symbol=:He, pressure=0, n=1, m=1)
    χ = Refraction.χ1(:He, 2π*c./ω)
    unm = besselj_zero(n-1, m)
    return @. ω/c*(1 + χ/2 - c^2*unm/(2ω^2*a^2))
end

end