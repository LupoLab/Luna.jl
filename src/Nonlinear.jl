module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths
import FFTW

"Kerr response for real field"
function Kerr_field(χ3)
    Kerr = let χ3 = χ3
        function Kerr(out, E)
            @. out += ε_0*χ3*E^3
        end
    end
end

"Kerr response for real field but without THG"
function Kerr_field_nothg(χ3, n)
    E2 = Array{Complex{Float64}}(undef, n)
    Kerr = let χ3 = χ3, E2 = E2
        function Kerr(out, E)
            @. E2 = E
            FFTW.fft!(E2)
            E2[div(length(E2),2):end] .= 0.0
            FFTW.ifft!(E2)
            @. out += 3/4*ε_0*χ3*abs2(2*E2)*E
        end
    end
end

"Kerr response for envelope"
function Kerr_env(χ3)
    Kerr = let χ3 = χ3
        function Kerr(out, E)
            @. out += 3/4*ε_0*χ3*abs2(E)*E
        end
    end
end

"Kerr response for envelope but with THG"
function Kerr_env_thg(χ3)
    Kerr = let χ3 = χ3
        function Kerr(out, E)
            @. out += ε_0*χ3/4*(3*abs2(E)*E + E^2)*E
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