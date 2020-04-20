module Nonlinear
import Luna
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths, Utils
import FFTW
import LinearAlgebra: mul!

"""
    scaled_response(resp, scale, shape)

Scale the response function `resp` by the factor `scale`. `shape` should be `size(Eto)` where
`Eto` is the oversampled field in time.
"""
function scaled_response(resp, scale, shape)
    buffer = Array{ComplexF64}(undef, shape)
    let buffer=buffer, resp=resp
        function resp!(out, E)
            fill!(buffer, 0)
            resp(buffer, E)
            @. out += scale*buffer
        end
    end
end


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
    E = Array{Float64}(undef, n)
    hilbert = Maths.plan_hilbert(E)
    Kerr = let γ3 = γ3, hilbert = hilbert
        function Kerr(out, E)
            out .+= 3/4*ε_0*γ3.*abs2.(hilbert(E)).*E
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

"Raman polarisation response type"
abstract type RamanPolar end

"Raman polarisation response type for a carrier resolved field"
struct RamanPolarField{Tω, Tt, Tv, FTt, HTt} <: RamanPolar
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tt # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tt # buffer to hold the time domain polarisation
    Pout::Tt # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    HT::HTt # Hilbert transform
    thg::Bool # do we include third harmonic generation
end

"Raman polarisation response type for an envelope"
struct RamanPolarEnv{Tω, Tv, FTt} <: RamanPolar
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tω # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tω # buffer to hold the time domain polarisation
    Pout::Tω # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
end

"""
    gethω!(h, t, ht, FT)

Get `hω` for time grid `t` using response function `ht`
and Fourier transform `FT`
"""
function gethω!(h, t, ht, FT)
    # possible improvements to this code:
    # TODO: use a smaller factor than 2 to expand the convolution grid (for higher performance)
    #       to do this needs careful checking
    # TODO: use a temporal window function to smooth frequency response for response functions
    #       which do not decay (common in gases, e.g. H2, which never decays on our usual grids)
    dt = t[2] - t[1]
    fill!(h, 0.0)
    # scale factor to correct for missing dt*dt*df from IFFT(FFT*FFT)
    # the ifft already scales by 1/n = dt*df, so we need an additional dt
    scale = dt
    # starting from positive t, fill only up to the first half of h
    # i.e. only the part corresponding to the original time grid
    start = findfirst(t .>= 0.0)
    for i = start:length(t)
        h[i] = ht(t[i])*scale
    end
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    hω, Eω2, Pω
end

"""
    RamanPolarField(t, ht; thg=true)

Construct Raman polarisation response for a field on time grid `t`
using response function `ht`. If `thg=false` then exclude the third
harmonic generation component of the response.
"""
function RamanPolarField(t, ht; thg=true)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω, Eω2, Pω = gethω!(h, t, ht, FT)
    E2 = similar(h)
    E2v = view(E2, 1:length(t))
    P = similar(h)
    Pout = similar(t)
    HT = Maths.plan_hilbert(Pout)
    fill!(E2, 0.0)
    RamanPolarField(hω, Eω2, Pω, E2, E2v, P, Pout, FT, HT, thg)
end

"""
    RamanPolarEnv(t, ht)

Construct Raman polarisation response for an envelope on time grid `t`
using response function `ht`.
"""
function RamanPolarEnv(t, ht)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω, Eω2, Pω = gethω!(h, t, ht, FT)
    E2 = similar(hω)
    P = similar(hω)
    Pout = Array{ComplexF64,}(undef,size(t))
    E2v = view(E2, 1:length(t))
    fill!(E2, 0.0)
    RamanPolarEnv(hω, Eω2, Pω, E2, E2v, P, Pout, FT)
end

"Square the field or envelope"
function sqr!(R::RamanPolarField, E)
    if !R.thg
        # see documentation for factor of 1/2 here
        R.E2v .= 1/2 .* abs2.(R.HT(E))
    else
        R.E2v .= E.^2
    end
end

function sqr!(R::RamanPolarEnv, E)
    # see documentation for factor of 1/2 here
    R.E2v .= 1/2 .* abs2.(E)
end

"Calculate Raman polarisation for field/envelope Et"
function (R::RamanPolar)(out, Et)
    # get the field as a 1D Array
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

    # square the field or envelope in first half
    # corresponding to the field/envelope grid size
    sqr!(R, E)

    # convolution by multiplication in frequency domain
    # the double grid gives us accurate convolution between the
    # full field grid and full response function
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2
    mul!(R.P, inv(R.FT), R.Pω)

    # calculate full polarisation, extracting only the valid
    # grid region
    for i = eachindex(E)
        R.Pout[i] = E[i]*R.P[(n ÷ 2) - 1 + i]
    end
    
    # add to output in dimensions requested
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        @. out += R.Pout
    end
end

end