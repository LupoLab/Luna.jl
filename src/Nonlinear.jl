module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths
import FFTW

"Response type for Kerr effect"
struct Kerr
    χ3::Float64
end

"Kerr response for real field"
function (K::Kerr)(out, E)
    @. out += ε_0*K.χ3 * E^3
end

"Kerr response for envelope field without THG"
function (K::Kerr)(out, E::Array{T, N}, THG::Val{false}) where T<:Complex where N
    @. out += ε_0*K.χ3/4 * 3*abs2(E)*E
end

"Kerr response for envelope with THG"
function (K::Kerr)(out, E::Array{T, N}, THG::Val{true}) where T<:Complex where N
    @. out += ε_0*K.χ3/4 * (3*abs2(E) + E^2)*E
end

"Kerr reponse for envelope, default case (no THG)"
function (K::Kerr)(out, E::Array{T, N}) where T<:Complex where N
    K(out, E, Val(false))
end

"Response type for cumtrapz-based plasma polarisation, adapted from:
M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz, and T. Brabec, Physical Review Letters 83, 2930 (1999)."
struct PlasmaCumtrapz{R, EType, tType}
    ratefunc::R
    ionpot::Float64
    rate::EType
    fraction::EType
    phase::EType
    J::EType
    P::EType
    t::tType
    δt::Float64
end

function PlasmaCumtrapz(t, E, ratefunc, ionpot)
    rate = similar(E)
    fraction = similar(E)
    phase = similar(E)
    J = similar(E)
    P = similar(E)
    return PlasmaCumtrapz(ratefunc, ionpot, rate, fraction, phase, J, P, t, t[2]-t[1])
end

function (Plas::PlasmaCumtrapz)(out, E)
    Plas.ratefunc(Plas.rate, E)
    Maths.cumtrapz!(Plas.fraction, Plas.t, Plas.rate)
    @. Plas.fraction = 1-exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            Plas.J[ii] += Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
    @. out += Plas.P
end

end