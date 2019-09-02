module AR_PCF
import PhysicalConstants
import Unitful
import FunctionZeros: besselj_zero
import Roots: fzero
import Cubature: hquadrature, hcubature
import SpecialFunctions: besselj
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: PhysData, Maths
import Luna.Modes: AbstractMode, dimlimits, β, α, field
import Luna.PhysData: c, ref_index_fun, roomtemp

export Matthias, dimlimits, β, α, field

# core and clad are function-like objects which return the
# refractive index as function of freq
struct Matthias{Tneff} <: AbstractMode
    a::Float64
    n::Int
    m::Int
    unm::Float64
    ϕ::Float64
    kind::Symbol
    neff::Tneff
end

"convenience constructor assunming single gas filling and silica clad"
function Matthias(a, gas, P, wthickness; n=1, m=1, kind=:HE, ϕ=0.0, T=PhysData.roomtemp)
    unm = Maths.get_unm(n, m, kind=kind)
    neff = ω -> neff_fiber(a, gas, P, wthickness, kind, m, unm, T)(ω)
    Matthias(a, n, m, unm, ϕ, kind, neff)
end

dimlimits(m::Matthias) = (:polar, (0.0, 0.0), (m.a, 2π))

"get the effective complex refractive index for AR-PCF. 
 We follow https://www.nature.com/articles/s41598-017-12234-5"
function neff_fiber(a, gas, P, wthickness, kind, m, unm, T)
    # values for tests from the paper
    # change it after solving https://github.com/LupoLab/Luna/issues/74 
    cladn = 1.45
    # coren = 1.0
    rfg = PhysData.ref_index_fun(gas, P, T)
    coren = ω -> rfg(2π*PhysData.c./ω)
    function neff(ω)
        # wait for https://github.com/LupoLab/Luna/issues/74 to be solved
        # rfs = PhysData.ref_index_fun(:SiO2)
        # cladn = rfs(2π*PhysData.c./ω)
        ϵ = cladn.^2.0./coren(ω).^2
        k0 = ω./PhysData.c
        ka = k0*coren(ω)
        ϕ = k0*wthickness*sqrt(cladn^2 - coren(ω)^2)
        σ = 1.0/(ka*a)
        if (kind == :TE) || (kind == :TM)
            a = unm^2.0/2.0;
            if kind == :TE
                b = a/(sqrt(ϵ - 1.0))/tan(ϕ);
                c = unm^4/8 + 2*unm^2/(ϵ - 1.0)/tan(ϕ)^2
                d = unm^3*(1+1/tan(ϕ)^2)/(ϵ - 1.0)
            elseif kind == :TM
                b = a*ϵ/(sqrt(ϵ - 1.0))/tan(ϕ)
                c = unm^4/8 + 2*unm^2*ϵ^2/(ϵ - 1.0)/tan(ϕ)^2
                d = unm^3*ϵ^2*(1+1/tan(ϕ)^2)/(ϵ - 1.0)
            end
            neff = coren(ω)*(1.0 - a*σ^2 - b*σ^3 - c*σ^4 + 1im*d*σ^4)
        else
            if kind == :EH
                s = 1
            elseif kind == :HE
                s = -1
            end
            a = unm^2/2
            b = unm^2*(ϵ+1)/(2sqrt(ϵ-1)*tan(ϕ))
            c = unm^4/8 + unm^2*m*s/2 + (unm^2/4*(2+m*s)*(ϵ+1)^2/(ϵ-1)-unm^4*(ϵ-1)/(8*m))*1/tan(ϕ)^2
            d = unm^3*(ϵ^2+1)/(2*(ϵ-1))*(1+1/tan(ϕ)^2)
            neff = coren(ω)*(1.0 - a*σ^2 - b*σ^3 - c*σ^4 + 1im*d*σ^4)
        end
        return neff
    end
    return neff
end

function β(m::Matthias, ω)
    return @. ω/c*real(m.neff(ω))
end

function α(m::Matthias, ω)
    neff = m.neff.(ω)
    return imag(neff)
end

# we use polar coords, so xs = (r, θ)
function field(m::Matthias)
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


