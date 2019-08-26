module Capillary
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Roots: fzero
import Cubature: hquadrature, hcubature
import SpecialFunctions: besselj
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: Maths
import Luna.PhysData: c, ref_index_fun, roomtemp
import Luna.Modes: AbstractMode, dimlimits, β, α, field

export MarcatilliMode, dimlimits, β, α, field

# core and clad are function-like objects which return the
# refractive index as a function of freq
struct MarcatilliMode{Tcore, Tclad} <: AbstractMode
    a::Float64
    n::Int
    m::Int
    kind::Symbol
    unm::Float64
    ϕ::Float64
    coren::Tcore
    cladn::Tclad
end

function MarcatilliMode(a, n, m, kind, ϕ, coren, cladn)
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
    MarcatilliMode(a, n, m, kind, unm, ϕ, coren, cladn)
end

"convenience constructor assunming single gas filling and silica clad"
function MarcatilliMode(a, gas, P; n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(:SiO2)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> rfs(2π*c./ω)
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn)
end

dimlimits(m::MarcatilliMode) = (:polar, (0.0, 0.0), (m.a, 2π))

function β(m::MarcatilliMode, ω)
    χ = m.coren.(ω).^2 .- 1
    return @. ω/c*(1 + χ/2 - c^2*m.unm^2/(2*ω^2*m.a^2))
end

function α(m::MarcatilliMode, ω)
    ν = m.cladn.(ω)
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

# we use polar coords, so xs = (r, θ)
function field(m::MarcatilliMode)
    if m.kind == :HE
        return (xs) -> besselj(m.n-1, xs[1]*m.unm/m.a) .* SVector(cos(xs[2])*sin(m.n*(xs[2] + m.ϕ)) - sin(xs[2])*cos(m.n*(xs[2] + m.ϕ)),
                                                          sin(xs[2])*sin(m.n*(xs[2] + m.ϕ)) + cos(xs[2])*cos(m.n*(xs[2] + m.ϕ)))
    elseif m.kind == :TE
        return (xs) -> besselj(1, xs[1]*m.unm/m.a) .* SVector(-sin(xs[2]), cos(xs[2]))
    elseif m.kind == :TM
        return (xs) -> besselj(1, xs[1]*m.unm/m.a) .* SVector(cos(xs[2]), sin(xs[2]))
    end
end

end
