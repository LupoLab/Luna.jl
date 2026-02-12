module Ionisation
import SpecialFunctions: gamma, dawson
import HCubature: hquadrature
import HDF5
import FileWatching.Pidfile: mkpidlock
import HypergeometricFunctions: pFq
import Logging: @info
import Luna.PhysData: c, ħ, electron, m_e, au_energy, au_time, au_Efield, wlfreq, polarisability_difference, polarisability, au_polarisability
import Luna.PhysData: ionisation_potential, quantum_numbers
import Luna: Maths, Utils
import Printf: @sprintf

abstract type AbstractIonRate end

"""
    IonRateADK(ionpot::Float64, threshold=true)
    IonRateADK(material::Symbol)

Ionisation rate based on the ADK formula. If `threshold` is true, use [`ADK_threshold`](@ref)
to avoid calculation below floating-point precision. If `cycle_average` is `true`, calculate
the cycle-averaged ADK ionisation rate instead.
"""
struct IonRateADK <: AbstractIonRate
    ionpot::Float64
    threshold::Bool
    cycle_average::Bool
    nstar::Float64
    cn_sq::Float64
    ω_p::Float64
    ω_t_prefac::Float64
    thr::Float64
    avfac::Float64
    occupancy::Int
end

function IonRateADK(material::Symbol; kwargs...)
    IonRateADK(ionisation_potential(material); kwargs...)
end

function IonRateADK(ionpot::Number; occupancy=2, threshold=true, cycle_average=false)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    if threshold
        thr = ADK_threshold(ionpot)
    else
        thr = 0.0
    end

    if cycle_average
        # Zenghu Chang: Fundamentals of Attosecond Optics (2011) p. 184
        # Section 4.2.3.1 Cycle-Averaged Rate
        # ̄w_ADK(Fₐ) = √(3/π) √(Fₐ/F₀) w_ADK(Fₐ) where Fₐ is the field amplitude
        Ip_au = ionpot / au_energy
        F0_au = (2Ip_au)^(3/2)
        F0 = F0_au*au_Efield
        avfac = sqrt.(3/(π*F0))
    else
        avfac = 1.0
    end

    IonRateADK(ionpot, threshold, cycle_average,
               nstar, cn_sq, ω_p, ω_t_prefac, thr, avfac, occupancy)
end

function (ir::IonRateADK)(E)
    aE = abs(E)
    if aE >= ir.thr
        r = (ir.occupancy * ir.ω_p * ir.cn_sq *
             (4 * ir.ω_p / (ir.ω_t_prefac * aE))^(2 * ir.nstar - 1)
             * exp(-4 / 3 * ir.ω_p / (ir.ω_t_prefac * aE)))
        if ir.avfac ≠ 1
            r *= ir.avfac*sqrt(aE)
        end
        return r
    else
        return zero(E)
    end
end

function (ir::IonRateADK)(out::AbstractArray, E::AbstractArray)
    out .= ir.(E)
end

"""
    ionrate_ADK(IP_or_material, E::Number; kwargs...) -> Float64

Calculate the ionisation rate based on the ADK model.

# Arguments

- `IP_or_material`: Ionisation potential (`Float64`) or gas species (`Symbol`)
- `E::Number`: Electric field in SI units (V/M)

# Keywords

- `kwargs...`: See `IonRateADK`
"""
function ionrate_ADK(IP_or_material, E; kwargs...)
    IonRateADK(IP_or_material; kwargs...).(E)
end

"""
    ADK_threshold(ionpot)

Determine the lowest electric field strength at which the ADK ionisation rate for the
ionisation potential `ionpot` is non-zero to within 64-bit floating-point precision.
"""
function ADK_threshold(ionpot)
    ADKfun = IonRateADK(ionpot; threshold=false)
    E = 1e3
    out = ADKfun(E)
    while out == 0
        E *= 1.01
        out = ADKfun(E)
    end
    return E
end

"""
    IonRatePPT(ionpot::Float64, λ0, Z, l; kwargs...)

PPT ionisation rate for a state with ionisation potential `ionpot`, when driven at
wavelength `λ0`, also given charge state `Z` and angular momentum `l`

# Keyword arguments
- `sum_tol::Number`: Relative tolerance used to truncate the infinite sum. Defaults to 1e-6.
- `cycle_average::Bool`: If `true`, calculate the cycle-averaged rate. Defaults to `false`.
- `sum_integral::Bool`: whether to approximate the infinite sum in the PPT rate equation with
    an integral (this neglects the multiphoton thresholds).
- `Δα::Number`: polarisability difference between the ground state and the cation (in SI units)
    to calculate the Stark shift of the ground-state energy levels. Defaults to 0.
- `α_ion::Number`: polarisability of the cation (in SI units) to calculate the dipole correction
    to the rate. Defaults to 0.
- `msum::Bool`: for l ≠ 0, whether or not to sum over different m states. Defaults to `true`.
- `Cnl::Real` : Pre-calculated `Cₙₗ` constant. If not given, defaults to the approximate expression from
    the PPT papers.
- `occupancy`: Occupancy of the state(s) from which ionisation is considered. Defaults to 2 for
    a state with two electrons (spin up/down).

# References
[1] Ilkov, F. A., Decker, J. E. & Chin, S. L.
Ionization of atoms in the tunnelling regime with experimental evidence
using Hg atoms. Journal of Physics B: Atomic, Molecular and Optical
Physics 25, 4005–4020 (1992)

[2] Bergé, L., Skupin, S., Nuter, R., Kasparian, J. & Wolf, J.-P.
Ultrashort filaments of light in weakly ionized, optically transparent
media. Rep. Prog. Phys. 70, 1633–1713 (2007)
(Appendix A)

[3] A. Couairon and A. Mysyrowicz,
"Femtosecond filamentation in transparent media,"
Physics Reports 441(2–4), 47–189 (2007).

"""
struct IonRatePPT{oT, CT} <: AbstractIonRate
    Z::Float64 # Charge state
    l::Int # Orbital angular momentum quantum number
    Δα::Float64 # Polarisbility difference between ground state and ion
    α_ion_au::Float64 # Polarisbility of the ion in atomic units
    ω0_au::Float64 # central frequency in atomic units
    Cnl::CT # either missing (default) or a pre-defined value for Cₙₗ
    ionpot::Float64 # ionisation potential in SI units
    occupancy::oT # occupancy (integer) or function occ(m) returning occupancy in m level
    msum::Bool # sum over m levels?
    sum_integral::Bool # replace the infinite sum with an integral?
    sum_tol::Float64 # relative tolerance for convergence of the infinite sum
    cycle_average::Bool # average over one cycle?
end

"""
    IonRatePPT(material::Symbol, λ0; kwargs...)

PPT ionisation rate for the given `material` when driven at wavelength `λ0`.

# Keyword arguments
- `stark_shift::Bool`: whether to include the Stark shift
- `dipole_corr::Bool`: whether to include the dipole correction factor

Other keyword arguments are identical to `IonRatePPT(ionpot::Float64, λ0, Z, l; kwargs...)`
"""
function IonRatePPT(material::Symbol, λ0; stark_shift=true, dipole_corr=true, kwargs...)
    _, l, Z = quantum_numbers(material)
    Δα = stark_shift ? polarisability_difference(material) : 0.0
    α_ion = dipole_corr ? polarisability(material, true) : 0.0
    ip = ionisation_potential(material)
    IonRatePPT(ip, λ0, float(Z), l; Δα, α_ion, kwargs...)
end

function IonRatePPT(ip, λ0, Z, l; Δα=0, α_ion=0, sum_tol=1e-6,
    cycle_average=false, sum_integral=false, msum=true, Cnl=missing, occupancy=2)

    if ismissing(Δα)
        Δα = 0.0
    end

    if ismissing(α_ion)
        α_ion = 0.0
    end

    α_ion_au = α_ion/au_polarisability

    ω0 = 2π*c/λ0
    ω0_au = au_time*ω0

    IonRatePPT(float(Z), l, Δα, α_ion_au, ω0_au, Cnl, ip, occupancy,
        msum, sum_integral, sum_tol, cycle_average)
end

function (ir::IonRatePPT)(E)
    Ip_au = (ir.ionpot + ir.Δα/2 * E^2) / au_energy # Δα/2 * E^2 includes the Stark shift
    ns = ir.Z/sqrt(2Ip_au)
    ls = ns-1
    Cnl2 = ismissing(ir.Cnl) ? 2^(2ns)/(ns*gamma(ns + ls + 1)*gamma(ns - ls)) : ir.Cnl^2

    E0_au = (2*Ip_au)^(3/2)

    E_au = abs(E)/au_Efield
    γ = ir.ω0_au*sqrt(2Ip_au)/E_au
    γ2 = γ*γ
    β = 2γ/sqrt(1 + γ2)
    α = 2*(asinh(γ) - γ/sqrt(1+γ2))
    Up_au = E_au^2/(4*ir.ω0_au^2)
    Uit_au = Ip_au + Up_au
    v = Uit_au/ir.ω0_au
    ret = 0
    mrange = ir.msum ? (-ir.l:ir.l) : (0:0)
    for m in mrange
        mabs = abs(m)
        flm = ((2ir.l + 1)*factorial(ir.l + mabs)
            / (2^mabs*factorial(mabs)*factorial(ir.l - mabs)))
        # Following 5 lines are [1] eq. 8 and lead to identical results:
        # G = 3/(2γ)*((1 + 1/(2γ2))*asinh(γ) - sqrt(1 + γ2)/(2γ))
        # Am = 4/(sqrt(3π)*factorial(mabs))*γ2/(1 + γ2)
        # lret = sqrt(3/(2π))*Cnl2*flm*Ip_au
        # lret *= (2*E0_au/(E_au*sqrt(1 + γ2))) ^ (2ns - mabs - 3/2)
        # lret *= Am*exp(-2*E0_au*G/(3E_au))
        # [2] eq. (A14)
        lret = 4sqrt(2)/π*Cnl2
        lret *= (2*E0_au/(E_au*sqrt(1 + γ2))) ^ (2ns - mabs - 3/2)
        lret *= flm/factorial(mabs)
        lret *= exp(-2v*(asinh(γ) - γ*sqrt(1+γ2)/(1+2γ2)))
        lret *= Ip_au * γ2/(1+γ2)
        # Remove cycle average factor, see eq. (2) of [1]
        if !ir.cycle_average
            lret *= sqrt(π*E0_au/(3E_au))
        end
        n0 = ceil(v)
        if ir.sum_integral
            s = sqrt(π)*factorial(mabs)*β^mabs/(2*(α+β)^(mabs+1))*sqrt(β/α)
        else
            s, _, _ = Maths.converge_series(0, n0=n0, rtol=ir.sum_tol, maxiter=Inf) do x, n
                diff = n-v
                x + exp(-α*diff)*φ(m, sqrt(β*diff))
            end

        end
        lret *= s
        ret += occ(ir.occupancy, m)*lret
    end
    if ir.α_ion_au ≠ 0
        ret *= exp(-2*ir.α_ion_au*E_au)
    end
    return ret/au_time
end

occ(occupancy::Number, m) = occupancy
occ(occupancy, m) = occupancy(m)

"""
    φ(m, x)

Calculate the φ function for the PPT ionisation rate.

Note that w_m(x) in [1] and φ_m(x) in [2] look slightly different but
are in fact identical.
"""
function φ(m, x)
    #= second half of [3], eq. 81
        for m = 0, φ₀(x) is just the Dawson integral so we can get this directly.
        for m ≠ 0, we calculate it using the hypergeometric function where possible.
        for m ≠ 0 and large x, we need to do it brute force with BigFloats (slow)
    =#
    if m == 0
        return dawson(x)
    end

    if x <= 26
        mabs = abs(m)
        return (exp(-x^2)
            * sqrt(π)
            * x^(2mabs+1)
            * gamma(mabs+1)
            * pFq((1/2,), (3/2 + mabs,), x^2)
            / (2*gamma(3/2 + mabs)))
    else
        i, _ = hquadrature(0, x) do y
            y = BigFloat(y)
            x = BigFloat(x)
            (x^2 - y^2)^(abs(m))*exp(y^2)
        end
        return Float64(exp(-x^2) * i)
    end
end

function (ir::IonRatePPT)(out::AbstractArray, E::AbstractArray)
    out .= ir.(E)
end

function ionrate_PPT(ionpot, λ0, Z, l, E; kwargs...)
    return IonRatePPT(ionpot, λ0, Z, l; kwargs...).(E)
end

function ionrate_PPT(material::Symbol, λ0, E;
                     stark_shift=true, dipole_corr=true, kwargs...)
    _, l, Z = quantum_numbers(material)
    Δα = stark_shift ? polarisability_difference(material) : 0.0
    α_ion = dipole_corr ? polarisability(material, true) : 0.0
    ip = ionisation_potential(material)
    return ionrate_PPT(ip, λ0, Z, l, E; Δα, α_ion, kwargs...)
end

struct IonRatePPTAccel{ST} <: AbstractIonRate
    spline::ST # spline interpolant
    Emin::Float64 # minimum electric field strength
    Emax::Float64 # maximum electric field strength
end

"""
    IonRatePPTAccel(material::Symbol, λ0; kwargs...)
    IonRatePPTAccel(ionpot::Float64, λ0, Z, l; kwargs...)
    IonRatePPTAccel(E, rate)

Create a cached (saved) interpolated PPT ionisation rate function. If a saved lookup table
exists, load this rather than recalculate.

# Keyword arguments
- `N::Int`: Number of samples with which to create the `CSpline` interpolant.
- `Emax::Number`: Maximum field strength to include in the interpolant.
- `cache::Bool`: Whether to save the pre-calculated rate to a file
- `cachedir::String`: Path to the directory where the cache should be stored and loaded from.
    Defaults to \$HOME/.luna/pptcache

Other keyword arguments are passed on to [`IonRatePPT`](@ref)
"""
function IonRatePPTAccel(E, rate)
        # first remove points where the rate is zero within floating-point
    # precision to avoid NaNs in the CSpline
    idcs = rate .> 0
    E = E[idcs]
    rate = rate[idcs]
    # Interpolating the log and re-exponentiating makes the spline more accurate
    cspl = Maths.CSpline(E, log.(rate); bounds_error=true)
    Emin = minimum(E)
    Emax = maximum(E)
    IonRatePPTAccel(cspl, Emin, Emax)
end

function IonRatePPTAccel(material::Symbol, λ0; stark_shift=true, dipole_corr=true, kwargs...)
    _, l, Z = quantum_numbers(material)
    Δα = stark_shift ? polarisability_difference(material) : 0.0
    α_ion = dipole_corr ? polarisability(material, true) : 0.0
    ip = ionisation_potential(material)
    IonRatePPTAccel(ip, λ0, Z, l; Δα, α_ion, kwargs...)
end

function IonRatePPTCached(args...; kwargs...)
    IonRatePPTAccel(args...; cache=true, kwargs...)
end

function IonRatePPTAccel(ionpot::Float64, λ0, Z, l;
    N=2^16, Emax=nothing, cache=true,
    cachedir=joinpath(Utils.cachedir(), "pptcache"),
    stale_age=60 * 10,
    kwargs...)
    h = hash((ionpot, λ0, Z, l, N, Emax, collect(kwargs)))
    fname = string(h, base=16) * ".h5"
    fpath = joinpath(cachedir, fname)
    if cache && isfile(fpath)
        @info @sprintf("Found cached PPT rate for %.2f eV, %.1f nm", ionpot / electron, 1e9λ0)
        E, rate = HDF5.h5open(fpath, "r") do file
            (read(file["E"]), read(file["rate"]))
        end
    else
        E, rate = makePPTcache(ionpot::Float64, λ0, Z, l;
            N, Emax, kwargs...)
    end

    if cache && ~isfile(fpath)
        lockpath = joinpath(cachedir, "pptlock")
        isdir(cachedir) || mkpath(cachedir)
        mkpidlock(lockpath; stale_age) do
            if ~isfile(fpath) # makePPTcache takes a while - has another process saved first?
                @info @sprintf(
                    "Saving PPT rate for %.2f eV, %.1f nm in %s",
                    ionpot / electron, 1e9λ0, fpath
                )
                HDF5.h5open(fpath, "cw") do file
                    file["E"] = E
                    file["rate"] = rate
                end
            end
        end
    end

    return IonRatePPTAccel(E, rate)
end

function (ir::IonRatePPTAccel)(E)
    aE = abs(E)
    if aE < ir.Emin
        return 0.0
    elseif aE > ir.Emax
        error(
            "Field strength $aE V/m exceeds maximum for PPT ionisation rate ($(ir.Emax) V/m)."
            )
    else
        return exp(ir.spline(aE))
    end
end

function (ir::IonRatePPTAccel)(out::AbstractArray, E::AbstractArray)
    out .= ir.(E)
end

function makePPTcache(ionpot::Float64, λ0, Z, l;
                      N=2^16, Emax=nothing, kwargs...)
    Emax = isnothing(Emax) ? 2*barrier_suppression(ionpot, Z) : Emax

    # ω0 = 2π*c/λ0
    # Emin = ω0*sqrt(2m_e*ionpot)/electron/0.5 # Keldysh parameter of 0.5
    Emin = Emax/5000

    E = collect(range(Emin, stop=Emax, length=N));
    @info @sprintf("Pre-calculating PPT rate for %.2f eV, %.1f nm...", ionpot/electron, 1e9λ0)
    flush(stderr) # pre-calculating can take a while, so make sure this message is shown
    rate = ionrate_PPT(ionpot, λ0, Z, l, E; kwargs...)
    @info "...PPT pre-calcuation done"
    flush(stderr)
    return E, rate
end

"""
    barrier_suppression(ionpot, Z)

Calculate the barrier-suppresion **field strength** for the ionisation potential `ionpot`
and charge state `Z`.
"""
function barrier_suppression(ionpot, Z)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2*Ip_au)
    Z^3/(16*ns^4) * au_Efield
end

"""
    keldysh(material, λ, E)

Calculate the Keldysh parameter for the given `material` at wavelength `λ` and electric field
strength `E`.
"""
function keldysh(material, λ, E)
    Ip_au = ionisation_potential(material)/au_energy
    E_au = E/au_Efield
    ω0_au = wlfreq(λ)*au_time
    ω0_au*sqrt(2Ip_au)/E_au
end

"""
    ionfrac(rate, E, δt)

Given an ionisation rate function `rate` and an electric field array `E` sampled with time
spacing `δt`, calculate the ionisation fraction as a function of time on the same time axis.

The function `rate` should have the signature `rate!(out, E)` and place its results into
`out`, like the functions returned by e.g. `IonRateADK` or `IonRatePPTCached`.
"""
function ionfrac(rate, E, δt)
    frac = similar(E)
    ionfrac!(frac, rate, E, δt)
end

function ionfrac!(frac, rate, E, δt)
    rate(frac, E)
    Maths.cumtrapz!(frac, δt)
    @. frac = 1 - exp(-frac)
end

end
