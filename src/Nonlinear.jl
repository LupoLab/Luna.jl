module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths
import FFTW

function KerrScalar(out, E, fac)
    @. out += fac*E^3
end

function KerrVector(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = Ex^2
        Ey2 = Ey^2
        out[i,1] += fac*(Ex2 + Ey2)*Ex
        out[i,2] += fac*(Ex2 + Ey2)*Ey
    end
end

"Kerr response for real field"
function Kerr_field(γ3)
    Kerr = let γ3 = γ3
        function Kerr(out, E)
            if size(E,2) == 1
                KerrScalar(out, E, ε_0*γ3)
            else size(E, 2) == 2
                KerrVector(out, E, ε_0*γ3)
            end
        end
    end
end

"Kerr response for real field but without THG"
function Kerr_field_nothg(γ3, n)
    E2 = Array{Complex{Float64}}(undef, n)
    Kerr = let γ3 = γ3, E2 = E2
        function Kerr(out, E)
            @. E2 = E
            FFTW.fft!(E2)
            E2[div(length(E2),2):end] .= 0.0
            FFTW.ifft!(E2)
            @. out += 3*ε_0*γ3*abs2(E2)*E
        end
    end
end

function KerrScalarEnv(out, E, fac)
    @. out += 3/4*fac*abs2(E)*E
end

function KerrVectorEnv(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = abs2(Ex)
        Ey2 = abs2(Ey)
        out[i,1] += 3/4*fac*((Ex2 + 2/3*Ey2)*Ex + 1/3*conj(Ex)*Ey^2)
        out[i,2] += 3/4*fac*((Ey2 + 2/3*Ex2)*Ey + 1/3*conj(Ey)*Ex^2)
    end
end

"Kerr response for envelope"
function Kerr_env(γ3)
    Kerr = let γ3 = γ3
        function Kerr(out, E)
            if size(E,2) == 1
                KerrScalarEnv(out, E, ε_0*γ3)
            else
                KerrVectorEnv(out, E, ε_0*γ3)
            end
        end
    end
end

"Kerr response for envelope but with THG"
# see Eq. 4, Genty et al., Opt. Express 15 5382 (2007)
function Kerr_env_thg(γ3, ω0, t)
    C = exp.(2im*ω0.*t)
    Kerr = let γ3 = γ3, C = C
        function Kerr(out, E)
            @. out += ε_0*γ3/4*(3*abs2(E) + C*E^2)*E
        end
    end
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

function (Plas::PlasmaCumtrapz)(out, Et)
    if ndims(Et) > 1
        if size(Et, 2) == 1
            E = reshape(Et, size(Et,1))
        else
            error("vector plasma not yet implemented")
        end
    else
        E = Et
    end
    Plas.ratefunc(Plas.rate, E)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = 1-exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            Plas.J[ii] += Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
    if ndims(Et) > 1
        out .+= reshape(Plas.P, size(Et))
    else
        @. out += Plas.P
    end
end

end