module Antiresonant
using Reexport
import Luna: Capillary
import Luna.PhysData: c
@reexport using Luna.Modes
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N

struct ZeisbergerMode{mT<:Capillary.MarcatilliMode, LT} <: AbstractMode
    m::mT
    wallthickness::Float64
    loss::LT # Val{true}(), Val{false}() or a number (scaling factor)
end

"""
    ZeisbergerMode(args...; wallthickness, kwargs...)

Create a capillary-like mode with the effective index given by eq. (15) in [1].

`wallthickness` (mandatory kwarg) sets the thickness of the anti-resonant struts and
`loss` (optional, defaults to `true`) can be either a `Bool` (to switch on/off loss
completely) or a `Real` (to up/down-scale the loss given by the model).
 Other kwargs are passed on to the constructor of a [`MarcatilliMode`](@ref).

[1] Zeisberger, M., Schmidt, M.A. Analytic model for the complex effective index of the
leaky modes of tube-type anti-resonant hollow core fibers. Sci Rep 7, 11761 (2017).
https://doi.org/10.1038/s41598-017-12234-5
"""
function ZeisbergerMode(args...; wallthickness, loss=true, kwargs...)
    return ZeisbergerMode(Capillary.MarcatilliMode(args...; kwargs...),
                          wallthickness, wraptype(loss))
end

function ZeisbergerMode(m::Capillary.MarcatilliMode; wallthickness, loss=true)
    ZeisbergerMode(m, wallthickness, wraptype(loss))
end

wraptype(loss::Bool) = Val(loss)
wraptype(loss::Real) = loss
wraptype(loss) = throw(
    ArgumentError("loss has to be a Bool or Real, not $(typeof(loss))"))

# Effective index is given by eq (15) in [1]
neff(m::ZeisbergerMode, ω; z=0) = _neff(m.m, ω, m.wallthickness, m.loss; z=z)

# All other mode properties are identical to a MarcatilliMode
for fun in (:Aeff, :field, :N, :dimlimits)
    @eval ($fun)(m::ZeisbergerMode, args...; kwargs...) = ($fun)(m.m, args...; kwargs...)
end

function _neff(m::Capillary.MarcatilliMode, ω, wallthickness, loss; z=0)
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
        return __neff(A, B, C, D, σ, nco)
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
        __neff(A, B, C, D, σ, nco, loss)
    end
end

#= If ncl and nco are real, the type of neff depends on whether loss is included.
    Despatching on the type makes sure the type can be inferred by the compiler =#
__neff(A, B, C, D, σ, nco, loss::Val{true}) = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*D*σ^4)
__neff(A, B, C, D, σ, nco, loss::Val{false}) = real(nco*(1 - A*σ^2 - B*σ^3 - C*σ^4))
__neff(A, B, C, D, σ, nco, loss::Number) = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*loss*D*σ^4)

end