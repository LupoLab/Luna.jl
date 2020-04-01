module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths, Utils
import FFTW
import LinearAlgebra: mul!

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
            else
                KerrVector(out, E, ε_0*γ3)
            end
        end
    end
end

"Kerr response for real field but without THG"
function Kerr_field_nothg(γ3, n)
    Kerr = let γ3 = γ3, E2 = E2
        function Kerr(out, E)
            out .+= 3/4*ε_0*γ3.*abs2.(Maths.hilbert(E)).*E
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

struct RamanPolarField{Tω, Tt, FTt}
    hω::Tω
    Eω2::Tω
    Pω::Tω
    E2::Tt
    P::Tt
    Pout::Tt
    FT::FTt
    nothg::Bool
end

function RamanPolarField(t, ht; nothg=false)
    dt = t[2] - t[1]
    h = zeros(length(t)*2)
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=FFTW.PATIENT)
    inv(FT)
    Utils.saveFFTwisdom()
    fill!(h, 0.0)
    start = findfirst(t .> 0.0)
    for i = start:length(t)
        h[i] = ht(t[i])*dt
    end
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(h)
    P = similar(h)
    Pout = similar(t)
    RamanPolarField(hω, Eω2, Pω, E2, P, Pout, FT, nothg)
end

function (R::RamanPolarField)(out, Et)
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1
            E = reshape(Et, n)
        else
            error("vector Raman not yet implemented")
        end
    else
        E = Et
    end

    if R.nothg
        Ei = 3/4 .* abs2.(Maths.hilbert(E))
    else
        Ei = E.^2
    end

    fill!(R.E2, 0.0)
    for i = eachindex(Ei)
        R.E2[i] = Ei[i]
    end
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2
    mul!(R.P, inv(R.FT), R.Pω)
    for i = eachindex(E)
        R.Pout[i] = E[i]*R.P[(n ÷ 2) + i] # TODO: possible off by 1 error
    end
    
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        @. out += R.Pout
    end
end

# TODO there is a huge amount of duplication here between field and env versions
# can we make them one?

struct RamanPolarEnv{Tω, FTt}
    hω::Tω
    Eω2::Tω
    Pω::Tω
    E2::Tω
    P::Tω
    Pout::Tω
    FT::FTt
end

function RamanPolarEnv(t, ht)
    dt = t[2] - t[1]
    h = zeros(length(t)*2)
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(h, 1, flags=FFTW.PATIENT)
    inv(FT)
    Utils.saveFFTwisdom()
    fill!(h, 0.0)
    start = findfirst(t .> 0.0)
    for i = start:length(t)
        h[i] = ht(t[i])*dt
    end
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(hω)
    P = similar(hω)
    Pout = Array{ComplexF64,}(undef,size(t))
    RamanPolarEnv(hω, Eω2, Pω, E2, P, Pout, FT)
end

function (R::RamanPolarEnv)(out, Et)
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1
            E = reshape(Et, n)
        else
            error("vector Raman not yet implemented")
        end
    else
        E = Et
    end

    fill!(R.E2, 0.0)
    for i = eachindex(E)
        R.E2[i] = abs2(E[i])
    end
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2
    mul!(R.P, inv(R.FT), R.Pω)
    for i = eachindex(E)
        R.Pout[i] = E[i]*R.P[(n ÷ 2) + i] # TODO: possible off by 1 error
    end
    
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        @. out += R.Pout
    end
end

end