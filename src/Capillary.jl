module Capillary
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Roots: fzero
import Luna: Maths, PhysData

const c = Unitful.ustrip(PhysicalConstants.CODATA2014.c)

function β(a, ω; gas::Symbol=:He, pressure=0, n=1, m=1)
    χ = pressure .* PhysData.χ1(gas, 2π*c./ω)
    unm = besselj_zero(n-1, m)
    return @. ω/c*(1 + χ/2 - c^2*unm^2/(2*ω^2*a^2))
end

function β(a; λ, gas::Symbol=:He, pressure=0, n=1, m=1)
    return β(a, 2π*c./λ, gas=gas, pressure=pressure, n=n, m=m)
end

function α(a, ω; n=1, m=1)
    unm = besselj_zero(n-1, m)
    ν = PhysData.ref_index(:SiO2, 2π*c./ω)
    return @. (2*(c^2 * unm^2)/(a^3 * ω^2) * (ν^2 + 1)/(2*sqrt(ν^2-1)))
end

function α(a; λ, n=1, m=1)
    return α(a, 2π*c./λ, n=n, m=m)
end

function losslength(a, ω; n=1, m=1)
    return 1 ./ α(a, ω, n=n, m=m)
end

function losslength(a; λ, n=1, m=1)
    return 1 ./ α(a, n=n, m=m, λ=λ)
end

function transmission(a, L; λ, n=1, m=1)
    return @. exp(-α(a, λ=λ, n=n, m=m)*L)
end

function dB_per_m(a, ω; n=1, m=1)
    return 10/log(10).*α(a, ω, n=n, m=m)
end

function dB_per_m(a; n=1, m=1, λ)
    return return 10/log(10) .* α(a, n=n, m=m, λ=λ)
end

function dispersion_func(order, a; gas::Symbol=:He, pressure=0, n=1, m=1)
    βn(ω) = Maths.derivative(ω -> β(a, ω, gas=gas, pressure=pressure, n=n, m=m), ω, order)
    return βn
end

function dispersion(order, a, ω; gas::Symbol=:He, pressure=0, n=1, m=1)
    return dispersion_func(order, a, gas=gas, pressure=pressure, n=n, m=m).(ω)
end

function dispersion(order, a; λ, gas::Symbol=:He, pressure=0, n=1, m=1)
    return dispersion(order, a, 2π*c./λ, gas=gas, pressure=pressure, n=n, m=m)
end

function zdw(a; gas::Symbol, pressure, n=1, m=1)
    ω0 = fzero(dispersion_func(2, a, gas=gas, pressure=pressure, n=n, m=m), 1e14, 2e16)
    return 2π*c/ω0
end


end