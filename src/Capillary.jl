module Capillary
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Roots: fzero
import Cubature: hquadrature
import SpecialFunctions: besselj
import Luna: Maths
import Luna.PhysData: c, χ1, ref_index
import Luna.AbstractModes: AbstractMode, maxR, β, α, field

# TODO factor out gas properties to some kind of medium/gas type
# and make this immutable
mutable struct MarcatilliMode <: AbstractMode
    a::Float64
    n::Int
    m::Int
    kind::Symbol
    unm::FLoat64
    ϕ::Float64
    gas::Symbol
    pressure::Float64
end

function MarcatilliMode(a, n, m, kind, ϕ, gas, pressure)
    if (kind == :TE) || (kind == :TM)
        if (n != 0) || (m != 1)
            error("n=0, m=1 for TE or TM modes")
        end
        unm = besselj_zero(1, 1)
    elseif kind == :HE
        unm = besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
    MarcatilliMode(a, n, m, kind, unm, ϕ, gas, pressure)
end

maxR(m) = m.a

function β(m::MarcatilliMode, ω)
    χ = m.pressure .* χ1(m.gas, 2π*c./ω)
    return @. ω/c*(1 + χ/2 - c^2*m.unm^2/(2*ω^2*m.a^2))
end

function α(m::MarcatilliMode, ω)
    ν = ref_index(:SiO2, 2π*c./ω)
    if m.kind == :HE
        vp = @. (ν^2 + 1)/(2*real(sqrt(Complex(ν^2-1))))
    elseif m.kind == :TE
        vp = @. 1/(real(sqrt(Complex(ν^2-1))))
    elseif m.kind == :TM
        vp = @. ν^2/(real(sqrt(Complex(ν^2-1))))
    else
        error("kind must be :TE, :TM or :HE")
    end
    return @. 2*(c^2 * m.unm^2)/(m.a^3 * ω^2) * vp
end

function field(m::MarcatilliMode)
    if m.kind == :HE
        return (r, θ) -> besselj(m.n-1, r*m.unm/m.a) .* SVector(cos(θ)*sin(m.n*(θ + m.ϕ)) - sin(θ)*cos(m.n*(θ + m.ϕ)),
                                                          sin(θ)*sin(m.n*(θ + m.ϕ)) + cos(θ)*cos(m.n*(θ + m.ϕ)))
    elseif m.kind == :TE
        return (r, θ) -> besselj(1, r*m.unm/m.a) .* SVector(-sin(θ), cos(θ))
    elseif m.kind == :TM
        return (r, θ) -> besselj(1, r*m.unm/m.a) .* SVector(cos(θ), sin(θ))
    end
end

end
