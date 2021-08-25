module SimpleFibre
using Reexport
@reexport using Luna.Modes
import Luna.Modes: AbstractMode, Aeff, neff, β, α, dispersion, dispersion_func
export SimpleMode
import Polynomials
import Luna.PhysData: c

struct SimpleMode{TP} <: AbstractMode
    ωref::Float64
    poly::TP
    Aeff::Float64
    α::Float64
end

"""
    SimpleMode(ωref, βs; Aeff=1.0, loss=0.0)

Create a SimpleMode based on Taylor expansion coefficients `βs` about frequency `ωref`, optionally
with effective area `Aeff` in m^2 and propagation loss `loss` in dB/m.

"""
function SimpleMode(ωref, βs; Aeff=1.0, loss=0.0)
    poly = Polynomials.Polynomial(βs ./ factorial.((1:length(βs)) .- 1))
    α = log(10)/10*loss
    SimpleMode(ωref, poly, Aeff, α)
end

"""
    SimpleMode(βs)

Create a SimpleMode based on Taylor expansion coefficients `βs`.

"""
function SimpleMode(βs)
    SimpleMode(0.0, βs)
end

function dispersion_func(m::SimpleMode, order; z=0.0)
    p = m.poly
    for i = 1:order
        p = Polynomials.derivative(p)
    end
    ω -> p(ω - m.ωref)
end

β(m::SimpleMode, ω; z=0.0) = m.poly(ω - m.ωref)

α(m::SimpleMode, ω; z=0.0) = m.α

neff(m::SimpleMode, ω; z=0) = c/ω*β(m, ω, z=z)

Aeff(m::SimpleMode; z=0.0) = m.Aeff

end
