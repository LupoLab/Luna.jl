module Nonlinear
import Luna
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths, Utils
import FFTW
import LinearAlgebra: mul!, ldiv!

function KerrScalar!(out, E, fac)
    @. out += fac*E^3
end

function KerrVector!(out, E, fac)
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
        function Kerr(out, E, ρ)
            if size(E,2) == 1
                KerrScalar!(out, E, ρ*ε_0*γ3)
            else
                KerrVector!(out, E, ρ*ε_0*γ3)
            end
        end
    end
end

"Kerr response for real field but without THG"
function Kerr_field_nothg(γ3, n)
    E = Array{Float64}(undef, n)
    hilbert = Maths.plan_hilbert(E)
    Kerr = let γ3 = γ3, hilbert = hilbert
        function Kerr(out, E, ρ)
            out .+= ρ*3/4*ε_0*γ3.*abs2.(hilbert(E)).*E
        end
    end
end

function KerrScalarEnv!(out, E, fac)
    @. out += 3/4*fac*abs2(E)*E
end

function KerrVectorEnv!(out, E, fac)
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
        function Kerr(out, E, ρ)
            if size(E,2) == 1
                KerrScalarEnv!(out, E, ρ*ε_0*γ3)
            else
                KerrVectorEnv!(out, E, ρ*ε_0*γ3)
            end
        end
    end
end

"Kerr response for envelope but with THG"
# see Eq. 4, Genty et al., Opt. Express 15 5382 (2007)
function Kerr_env_thg(γ3, ω0, t)
    C = exp.(2im*ω0.*t)
    Kerr = let γ3 = γ3, C = C
        function Kerr(out, E, ρ)
            @. out += ρ*ε_0*γ3/4*(3*abs2(E) + C*E^2)*E
        end
    end
end

abstract type PlasmaPolar end

"Response type for cumtrapz-based plasma polarisation, adapted from:
M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz, and T. Brabec, Physical Review Letters 83, 2930 (1999)."
struct PlasmaCumtrapz{R, EType, tType} <: PlasmaPolar
    ratefunc::R # the ionization rate function
    ionpot::Float64 # the ionization potential (for calculation of ionization loss)
    rate::tType # buffer to hold the rate
    fraction::tType # buffer to hold the ionization fraction
    phase::EType # buffer to hold the plasma induced (mostly) phase modulation
    Jloss::EType # buffer to hold the ionization loss current
    J::EType # buffer to hold the total plasma current
    P::EType # buffer to hold the plasma polarisation
    δt::Float64 # the time step
end

"""
    PlasmaCumtrapz(t, E, ratefunc, ionpot)

Construct the Plasma polarisation response based on cumulative integration for a
real field on time grid `t` with example electric field like `E`, an ionization
rate callable `ratefunc` and ionization potential `ionpot`.
"""
function PlasmaCumtrapz(t, E::Array{T,N}, ratefunc, ionpot) where T<:Real where N
    rate = similar(t)
    fraction = similar(t)
    phase = similar(E)
    J = similar(E)
    Jloss = similar(E)
    P = similar(E)
    return PlasmaCumtrapz(ratefunc, ionpot, rate, fraction, phase, Jloss, J, P, t[2]-t[1])
end

"ionizing field"
ionE(E) = abs.(E)
ionE(E::Array{T,2}) where T = hypot.(E[:,1], E[:,2])

"ionization loss current"
# for envelopes this follows Bergé et al. Rep. Prog. Phys. 70, 1633 (2007)
Jloss!(out, ip, rate, fraction, Em, E) = @. out = ifelse(Em > 0, ip*rate*(1 - fraction)/abs2(Em)*E, 0.0)
# minor optimisation for scalar real field, based on Geissler, PRL 83, 2930 (1999).
Jloss!(out, ip, rate, fraction, Em, E::Vector{Float64}) =  @. out = ifelse(Em > 0, ip*rate*(1 - fraction)/E, 0.0)

"""
Calculate the ionization rate, fraction, loss and phase term.
"""
function precalc!(Plas::PlasmaPolar, E)
    # This is basd on Geissler, PRL 83, 2930 (1999) for the real scalar field.
    # In the vector case we take the magnitude of the electric field to calculate the ionization
    # rate and fraction, and then solve the plasma polarisation component-wise for the vector field.
    # A similar approach was used in: C Tailliez et al 2020 New J. Phys. 22 103038. 
    # The equations used here for envelopes are identical to the real field cases (apart from Jloss).
    # So far I have not been able to derive them directly. The modified loss term
    # is taken from Bergé et al. Rep. Prog. Phys. 70, 1633 (2007). The plasma phase term is obtained
    # by inspection. Essentially, you can start from the definition of the plasma refractive index,
    # insert it into the general relation between polarisation and refractive index, and obtain this
    # version (in the frequency domain) but with the electron density in the time domain(!).
    # I also think this term could be obtained from Geissler's model, by making use of the Bedrosian
    # identity for Hilbert transforms of products of functions, but I have not yet achieved it.
    Em = ionE(E)
    Plas.ratefunc(Plas.rate, Em)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = 1-exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Jloss!(Plas.Jloss, Plas.ionpot, Plas.rate, Plas.fraction, Em, E)
end

"""
The plasma response for real electric field.
"""
function calc!(Plas::PlasmaCumtrapz, E::Array{Float64,N}) where N
    precalc!(Plas, E)
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    Plas.J .+= Plas.Jloss
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

struct PlasmaFourier{R, EType, tType, FTType} <: PlasmaPolar
    ratefunc::R # the ionization rate function
    ionpot::Float64 # the ionization potential (for calculation of ionization loss)
    rate::tType # buffer to hold the rate
    fraction::tType # buffer to hold the ionization fraction
    phase::EType # buffer to hold the plasma induced (mostly) phase modulation
    Jloss::EType # buffer to hold the ionisation loss current
    J::EType # buffer to hold the total plasma current
    P::EType # buffer to hold the plasma polarisation
    ω::tType # buffer to hold the frequency grid
    FT::FTType # Fourier transform to use for integrations
    δt::Float64
end

"""
    make_fft(E::Vector)

Plan a suitable FFT for the electric field type `E`.

"""
function make_fft(E::Array{T,N}) where {T<:Real, N}
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(E, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    FT
end

function make_fft(E::Array{Complex{T},N}) where {T<:Real, N}
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(E, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    FT
end

function PlasmaFourier(ω, E, ratefunc, ionpot, δt)
    rate = similar(ω)
    fraction = similar(ω)
    phase = similar(E)
    Jloss = similar(E)
    J = similar(E)
    P = similar(E)
    PlasmaFourier(ratefunc, ionpot, rate, fraction, phase, Jloss, J, P, ω, make_fft(E), δt)
end

"""
The plasma response for a scalar or vector, real field, or scalar envelope, based
on cumulative integration using Fourier transforms.
"""
function calc!(Plas::PlasmaFourier, E)
    precalc!(Plas, E)
    Plas.P .= Plas.FT \ (-(Plas.FT * Plas.phase) ./ Plas.ω.^2 .- 1im .* (Plas.FT * Plas.Jloss) ./ Plas.ω)
end

"Handle plasma polarisation routing to `PlasmaVector` or `PlasmaScalar`."
function (Plas::PlasmaPolar)(out, Et, ρ)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            calc!(Plas, reshape(Et, size(Et,1)))
            out .+= ρ .* reshape(Plas.P, size(Et))
        else
            calc!(Plas, Et) # vector case
            out .+= ρ .* Plas.P
        end
    else
        calc!(Plas, Et) # straight scalar case
        out .+= ρ .* Plas.P
    end
end

"Raman polarisation response type"
abstract type RamanPolar end

"Raman polarisation response type for a carrier resolved field"
struct RamanPolarField{TR, Tt, Thv, Tω, Tv, FTt, HTt} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding 
    ht::Thv # buffer to hold time domain response
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
    dt::Float64 # time step for scaling
end

"Raman polarisation response type for an envelope"
struct RamanPolarEnv{TR, Tt, Thv, Tω, Tv, FTt} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding 
    ht::Thv # buffer to hold time domain response
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tω # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tω # buffer to hold the time domain polarisation
    Pout::Tω # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    dt::Float64 # time step for scaling
end

"""
    RamanPolarField(t, ht; thg=true)

Construct Raman polarisation response for a field on time grid `t`
using response function `r`. If `thg=false` then exclude the third
harmonic generation component of the response.
"""
function RamanPolarField(t, r; thg=true)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    FT = make_fft(h)
    ht = view(h, 1:length(t))
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(h)
    E2v = view(E2, 1:length(t))
    P = similar(h)
    Pout = similar(t)
    HT = Maths.plan_hilbert(Pout)
    fill!(E2, 0.0)
    RamanPolarField(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, HT, thg, t[2] - t[1])
end

"""
    RamanPolarEnv(t, ht)

Construct Raman polarisation response for an envelope on time grid `t`
using response function `r`.
"""
function RamanPolarEnv(t, r)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    FT = make_fft(h)
    ht = view(h, 1:length(t))
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(hω)
    P = similar(hω)
    Pout = Array{ComplexF64,}(undef,size(t))
    E2v = view(E2, 1:length(t))
    fill!(E2, 0.0)
    RamanPolarEnv(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, t[2] - t[1])
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
function (R::RamanPolar)(out, Et, ρ)
    # get the field as a 1D Array
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            E = reshape(Et, n)
        else
            # handle vector case
            error("vector Raman not yet implemented")
        end
    else
        E = Et # handle straight scalar case
    end

    # square the field or envelope in first half
    # corresponding to the field/envelope grid size
    sqr!(R, E)

    # update frequency domain response function `hω`.
    # we fill only up to the first half of h (using the view ht)
    # i.e. only the part corresponding to the original time grid
    # note that the response function time 0 is put into the first element of the response array
    # this ensures that causality is maintained, and no artificial delay between the field and
    # the start of the response function occurs, at each convolution point.  
    R.r(R.ht, ρ)
    R.hω .= R.FT * R.h

    # convolution by multiplication in frequency domain
    # The double grid gives us accurate full convolution between the full field grid
    # and full response function. It is unnecessary for highly damped responses, like
    # in glass. But for gases with very long decay times it prevents artefacts due to
    # truncation of the response function. There is likely a more efficient way. But
    # this is safe, until we come up with one.
    # we scale to correct for missing dt*dt*df from IFFT(FFT*FFT)
    # the ifft already scales by 1/n = dt*df, so we need an additional dt
    R.Eω2 .= R.FT * R.E2
    @. R.Pω = R.hω * R.Eω2 * R.dt
    R.P .= R.FT \ R.Pω

    # calculate full polarisation, extracting only the valid
    # grid region, which is the first length(E) part.
    for i = 1:length(E)
        R.Pout[i] = ρ*E[i]*R.P[i]
    end
    
    # copy to output in dimensions requested
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        out .+= R.Pout
    end
end

end
