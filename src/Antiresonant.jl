module Antiresonant
using Reexport
import Luna: Capillary
import Luna.PhysData: c
@reexport using Luna.Modes
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N

struct Zeisberger{mT<:Capillary.MarcatilliMode} <: AbstractMode
    m::mT
    wallthickness::Float64
end

function Zeisberger(args...; wallthickness, kwargs...)
    return Zeisberger(Capillary.MarcatilliMode(args...; kwargs...), wallthickness)
end

Zeisberger(m::Capillary.MarcatilliMode; wallthickness) = Zeisberger(m, wallthickness)

neff(m::Zeisberger, ω; z=0) = _neff(m.m, ω, m.wallthickness; z=z)

"get the effective complex refractive index for AR-PCF. 
 We follow https://www.nature.com/articles/s41598-017-12234-5"
function _neff(m::Capillary.MarcatilliMode, ω, wallthickness; z=0)
    nco = m.coren(ω, z=z)
    ncl = m.cladn(ω, z=z)
    ϵ = ncl^2 / nco^2
    k0 = ω / c
    ka = k0*nco
    ϕ = k0*wallthickness*sqrt(ncl^2 - nco^2)
    σ = 1/(ka*m.a(z))
    if (m.kind == :TE) || (m.kind == :TM)
        A = m.unm^2/2;
        if m.kind == :TE
            B = A/(sqrt(ϵ - 1))/tan(ϕ);
            C = m.unm^4/8 + 2*m.unm^2/(ϵ - 1)/tan(ϕ)^2
            D = m.unm^3*(1+1/tan(ϕ)^2)/(ϵ - 1)
        elseif m.kind == :TM
            B = A*ϵ/(sqrt(ϵ - 1))/tan(ϕ)
            C = m.unm^4/8 + 2*m.unm^2*ϵ^2/(ϵ - 1)/tan(ϕ)^2
            D = m.unm^3*ϵ^2*(1+1/tan(ϕ)^2)/(ϵ - 1)
        end
        neff = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*D*σ^4)
    else
        if m.kind == :EH
            s = 1
        elseif m.kind == :HE
            s = -1
        end
        A = m.unm^2/2
        B = m.unm^2*(ϵ+1)/(sqrt(ϵ-1)*tan(ϕ))
        C = (m.unm^4/8
            + m.unm^2*m.m*s/2
            + (m.unm^2/4*(2+m.m*s)*(ϵ+1)^2/(ϵ-1)
            - m.unm^4*(ϵ-1)/(8*m.m))*1/tan(ϕ)^2)
        D = m.unm^3/2 * (ϵ^2+1)/(ϵ-1) * (1 + 1/tan(ϕ)^2)
        neff = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*D*σ^4)
    end
    return neff
end

for fun in (:Aeff, :field, :N, :dimlimits)
    @eval ($fun)(m::Zeisberger, args...; kwargs...) = ($fun)(m.m, args...; kwargs...)
end

end