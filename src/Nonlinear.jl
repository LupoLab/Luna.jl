module Nonlinear
import Luna
import Luna.PhysData: ε_0, c, e_ratio
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

"Response type for cumtrapz-based plasma polarisation, adapted from:
M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz, and T. Brabec, Physical Review Letters 83, 2930 (1999)."
struct PlasmaCumtrapz{R, EType, tType}
    ratefunc::R # the ionization rate function
    ionpot::Float64 # the ionization potential (for calculation of ionization loss)
    rate::tType # buffer to hold the rate
    fraction::tType # buffer to hold the ionization fraction
    phase::EType # buffer to hold the plasma induced (mostly) phase modulation
    J::EType # buffer to hold the plasma current
    P::EType # buffer to hold the plasma polarisation
    δt::Float64 # the time step
    preionfrac::Float64 # the pre-ionisation fraction
end

"""
    PlasmaCumtrapz(t, E, ratefunc, ionpot)

Construct the Plasma polarisation response for a field on time grid `t`
with example electric field like `E`, an ionization rate callable
`ratefunc` and ionization potential `ionpot`.
"""
function PlasmaCumtrapz(t, E, ratefunc, ionpot; preionfrac=0.0)
    rate = similar(t)
    fraction = similar(t)
    phase = similar(E)
    J = similar(E)
    P = similar(E)
    !(0.0 <= preionfrac <= 1.0) && throw(DomainError(preionfrac, "preionfrac must be between 0 and 1"))
    if preionfrac > 0.0
        @warn("Using preionfrac > 0.0 is not a well founded physical model. Use only after careful consideration.")
    end
    return PlasmaCumtrapz(ratefunc, ionpot, rate, fraction, phase, J, P, t[2]-t[1], preionfrac)
end

"The plasma response for a scalar electric field"
function PlasmaScalar!(Plas::PlasmaCumtrapz, E)
    Plas.ratefunc(Plas.rate, E)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            Plas.J[ii] += Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"""
The plasma response for a vector electric field.

We take the magnitude of the electric field to calculate the ionization
rate and fraction, and then solve the plasma polarisation component-wise
for the vector field.

A similar approach was used in: C Tailliez et al 2020 New J. Phys. 22 103038.  
"""
function PlasmaVector!(Plas::PlasmaCumtrapz, E)
    Ex = E[:,1]
    Ey = E[:,2]
    Em = @. hypot.(Ex, Ey)
    Plas.ratefunc(Plas.rate, Em)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(Em)
        if abs(Em[ii]) > 0
            pre = Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/Em[ii]^2
            Plas.J[ii,1] += pre*Ex[ii]
            Plas.J[ii,2] += pre*Ey[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"Handle plasma polarisation routing to `PlasmaVector` or `PlasmaScalar`."
function (Plas::PlasmaCumtrapz)(out, Et, ρ)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            PlasmaScalar!(Plas, reshape(Et, size(Et,1)))
            out .+= ρ .* reshape(Plas.P, size(Et))
        else
            PlasmaVector!(Plas, Et) # vector case
            out .+= ρ .* Plas.P
        end
    else
        PlasmaScalar!(Plas, Et) # straight scalar case
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
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
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
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
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

"""
    SpectralResponse

Abstract type for nonlinear responses that operate in the frequency domain
rather than the time domain.

Unlike time-domain responses (Kerr, Raman, Plasma) with callable signature
`response!(out_t, E_t, ρ)`, spectral responses have signature:

    response!(out_ω, F_E2E_ω, ρ)

where `F_E2E_ω` is ``\\mathcal{F}\\{|E|^2 E\\}(\\omega)`` and `out_ω` is the
frequency-domain nonlinear polarisation (same units as FFT of time-domain P_NL).

Spectral responses are **geometry-agnostic**: they operate on 1D frequency
vectors regardless of whether the simulation is mode-averaged, modal, or
3D free-space. All spatial integration and field normalisation is handled
by the `Trans*` types in `NonlinearRHS`, exactly as for time-domain responses.
The spectral response contribution enters `P_NL(ω)` **before** the
normalisation step, so it goes through the same norm as Kerr.

See also: [`TPAResponse`](@ref)
"""
abstract type SpectralResponse end

"""
    TPAResponse{cT}

Two-photon absorption (TPA) response for envelope propagation.

TPA is the imaginary part of the third-order susceptibility χ⁽³⁾. While the
real part (Kerr effect) is frequency-independent and handled in the time domain,
Im(χ⁽³⁾)(ω) varies strongly across deep-UV bandwidths and must be applied as
a frequency-domain multiplier on ``\\mathcal{F}\\{|E|^2 E\\}(\\omega)``:

    P_TPA(ω) = coeff(ω) × ρ × F{|E|²E}(ω)

This produces output in the **same P_NL units** as the Kerr response, so the
`Trans*` types' normalisation functions correctly convert both Kerr and TPA
contributions to ∂A/∂z. This makes `TPAResponse` geometry-agnostic: it works
identically for mode-averaged (`TransModeAvg`), modal (`TransModal`), and
3D free-space (`TransFree`) propagation.

The stored coefficient is:

    coeff(ω) = -i × ε₀² × c² × β₂(ω) / (2ω)

derived by requiring consistency with the Kerr nonlinear polarisation
in Luna's propagation pipeline. In ``TransModeAvg``, the field is
divided by ``\\mathrm{nlscale} = \\sqrt{\\varepsilon_0 c/2}`` before
computing ``|E|^2 E``, and the norm function applies
``-i\\omega^2/(4 \\cdot \\mathrm{nlscale} \\cdot c \\cdot \\beta)``.
Tracing the standard relationship
``\\beta_2 = 3\\omega \\, \\mathrm{Im}(\\chi^{(3)}) / (2 n_0^2 c^2)``
through yields the ``\\varepsilon_0^2 c^2`` prefactor
(from ``\\mathrm{nlscale}^4 = (\\varepsilon_0 c/2)^2``).
The ``-i`` factor ensures TPA causes loss (negative real
``\\partial A/\\partial z``) after the norm's own ``-i`` factor.

# Construction
```julia
# From β₂(ω) values and frequency grid:
tpa = Nonlinear.TPAResponse(grid.ω, PhysData.β₂_TPA.(grid.ω, :SiO2))

# From a pre-computed coefficient array:
tpa = Nonlinear.TPAResponse(coeff_ω)
```

# Usage
Pass in the responses tuple alongside time-domain responses:
```julia
responses = (Nonlinear.Kerr_env(χ3), Nonlinear.TPAResponse(grid.ω, β₂_ω))
```
All `Trans*` types automatically route `SpectralResponse` subtypes to
the frequency-domain path.

# Notes
- Only valid for envelope propagation (`EnvGrid`). Using with `RealGrid`
  (carrier-resolved) is not physically meaningful.
- Handles scalar (single-polarisation) fields. Vector TPA
  (``(|E_x|^2 + |E_y|^2) E``) is a future extension.
- The ``\\varepsilon_0^2 c^2`` prefactor arises from the ``\\mathrm{nlscale}``
  field normalisation in the ``Trans*`` types — it is not related to a
  ``\\gamma_3 = \\chi^{(3)}/n_0^2`` convention. In Luna, ``\\gamma_3`` is
  the single-molecule third-order hyperpolarisability, and
  ``\\chi^{(3)} = \\rho \\, \\gamma_3``.

See also: [`PhysData.β₂_TPA`](@ref), [`SpectralResponse`](@ref)
"""
struct TPAResponse{cT} <: SpectralResponse
    coeff_ω::cT
end

"""
    TPAResponse(ω_grid, β₂_ω) → TPAResponse

Construct a `TPAResponse` from an angular frequency grid `ω_grid` (rad/s)
and a vector of β₂ values `β₂_ω` (m/W).

Converts using:

    coeff(ω) = -i × ε₀² × c² × β₂(ω) / (2ω)

Sets coeff = 0 at ω = 0 to avoid the singularity (β₂ = 0 there anyway,
since no material has a TPA edge at DC).
"""
function TPAResponse(ω_grid::AbstractVector, β₂_ω::AbstractVector)
    length(ω_grid) == length(β₂_ω) || throw(
        DimensionMismatch("ω_grid and β₂_ω must have the same length"))
    coeff = zeros(ComplexF64, length(ω_grid))
    for i in eachindex(ω_grid)
        if ω_grid[i] != 0
            coeff[i] = -im * ε_0^2 * c^2 * β₂_ω[i] / (2 * ω_grid[i])
        end
    end
    return TPAResponse(coeff)
end

"""
    (tpa::TPAResponse)(out_ω, F_E2E_ω, ρ)

Add TPA contribution to the frequency-domain nonlinear polarisation `out_ω`.

- `out_ω`: frequency-domain P_NL buffer (1D), modified in-place
- `F_E2E_ω`: ``\\mathcal{F}\\{|E|^2 E\\}(\\omega)`` on the standard ω grid
- `ρ`: density (scalar), same as passed to time-domain responses

Adds: `out_ω[i] += ρ × coeff[i] × F_E2E_ω[i]`

This signature mirrors the time-domain `(out_t, E_t, ρ)` pattern but
operates in the frequency domain. The `Trans*` types handle all spatial
looping and normalisation.
"""
function (tpa::TPAResponse)(out_ω, F_E2E_ω, ρ)
    @inbounds for i in eachindex(out_ω)
        out_ω[i] += ρ * tpa.coeff_ω[i] * F_E2E_ω[i]
    end
end

end
