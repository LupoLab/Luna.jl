module Ionisation
import SpecialFunctions: gamma
import GSL: hypergeom
import HDF5
import Pidfile: mkpidlock
import Logging: @info
import Luna.PhysData: c, ħ, electron, m_e, au_energy, au_time, au_Efield, wlfreq
import Luna.PhysData: ionisation_potential, quantum_numbers
import Luna: Maths
import Luna: @hlock

"""
    ionrate_fun!_ADK(ionpot::Float64, threshold=true)
    ionrate_fun!_ADK(material::Symbol)

Return a closure `ionrate!(out, E)` which calculates the ADK ionisation rate for the electric
field `E` and places the result in `out`. If `threshold` is true, use [`ADK_threshold`](@ref)
to avoid calculation below floating-point precision. If `cycle_average` is `true`, calculate
the cycle-averaged ADK ionisation rate instead.
"""
function ionrate_fun!_ADK(ionpot::Float64, threshold=true; cycle_average=false)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    if threshold
        thr = ADK_threshold(ionpot)
    else
        thr = 0
    end

    # Zenghu Chang: Fundamentals of Attosecond Optics (2011) p. 184
    # Section 4.2.3.1 Cycle-Averaged Rate
    # ̄w_ADK(Fₐ) = √(3/π) √(Fₐ/F₀) w_ADK(Fₐ) where Fₐ is the field amplitude 
    Ip_au = ionpot / au_energy
    F0_au = (2Ip_au)^(3/2)
    F0 = F0_au*au_Efield
    avfac = sqrt.(3/(π*F0))


    ionrate! = let nstar=nstar, cn_sq=cn_sq, ω_p=ω_p, ω_t_prefac=ω_t_prefac, thr=thr
        function ir(E)
            if abs(E) >= thr
                r = (ω_p*cn_sq*
                    (4*ω_p/(ω_t_prefac*abs(E)))^(2*nstar-1)
                    *exp(-4/3*ω_p/(ω_t_prefac*abs(E))))
                if cycle_average
                    r *= avfac*sqrt(abs(E))
                end
                return r
            else
                return zero(E)
            end
        end
        function ionrate!(out, E)
            out .= ir.(E)
        end
    end

    return ionrate!  
end

function ionrate_fun!_ADK(material::Symbol; kwargs...)
    return ionrate_fun!_ADK(ionisation_potential(material); kwargs...)
end

function ionrate_ADK(IP_or_material, E; kwargs...)
    out = zero(E)
    ionrate_fun!_ADK(IP_or_material; kwargs...)(out, E)
    return out
end

function ionrate_ADK(IP_or_material, E::Number; kwargs...)
    out = [zero(E)]
    ionrate_fun!_ADK(IP_or_material; kwargs...)(out, [E])
    return out[1]
end

"""
    ADK_threshold(ionpot)

Determine the lowest electric field strength at which the ADK ionisation rate for the
ionisation potential `ionpot` is non-zero to within 64-bit floating-point precision.
"""
function ADK_threshold(ionpot)
    out = [0.0]
    ADKfun = ionrate_fun!_ADK(ionpot, false)
    E = 1e3
    while out[1] == 0
        E *= 1.01
        ADKfun(out, [E])
    end
    return E
end

"""
    ionrate_fun!_PPTaccel(material::Symbol, λ0; kwargs...)
    ionrate_fun!_PPTaccel(ionpot::Float64, λ0, Z, l; kwargs...)

Create an accelerated (interpolated) PPT ionisation rate function.
"""
function ionrate_fun!_PPTaccel(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTaccel(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTaccel(ionpot::Float64, λ0, Z, l; kwargs...)
    E, rate = makePPTcache(ionpot, λ0, Z, l; kwargs...)
    return makePPTaccel(E, rate)
end

"""
    ionrate_fun!_PPTcached(material::Symbol, λ0; kwargs...)
    ionrate_fun!_PPTcached(ionpot::Float64, λ0, Z, l; kwargs...)

Create a cached (saved) interpolated PPT ionisation rate function. If a saved lookup table
exists, load this rather than recalculate.

# Keyword arguments
- `N::Int`: Number of samples with which to create the `CSpline` interpolant.
- `Emax::Number`: Maximum field strength to include in the interpolant.
- `cachedir::String`: Path to the directory where the cache should be stored and loaded from.
    Defaults to \$HOME/.luna/pptcache

Other keyword arguments are passed on to [`ionrate_fun_PPT`](@ref)
"""
function ionrate_fun!_PPTcached(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTcached(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTcached(ionpot::Float64, λ0, Z, l;
                                sum_tol=1e-4, cycle_average=false, N=2^16, Emax=nothing,
                                cachedir=joinpath(homedir(), ".luna", "pptcache"))
    h = hash((ionpot, λ0, Z, l, sum_tol, cycle_average, N, Emax))
    fname = string(h, base=16)*".h5"
    fpath = joinpath(cachedir, fname)
    lockpath = joinpath(cachedir, "pptlock")
    isdir(cachedir) || mkpath(cachedir)
    if isfile(fpath)
        @info "Found cached PPT rate for $(ionpot/electron) eV, $(λ0*1e9) nm"
        pidlock = mkpidlock(lockpath)
        rate = loadPPTaccel(fpath)
        close(pidlock)
        return rate
    else
        E, rate = makePPTcache(ionpot::Float64, λ0, Z, l;
                               sum_tol=sum_tol, cycle_average, N=N, Emax=Emax)
        @info "Saving PPT rate cache for $(ionpot/electron) eV, $(λ0*1e9) nm in $cachedir"
        pidlock = mkpidlock(lockpath)
        if isfile(fpath) # makePPTcache takes a while - has another process saved first?
            rate = loadPPTaccel(fpath)
            close(pidlock)
            return rate
        end
        @hlock HDF5.h5open(fpath, "cw") do file
            file["E"] = E
            file["rate"] = rate
        end
        close(pidlock)
        return makePPTaccel(E, rate)
    end
end

function loadPPTaccel(fpath)
    isfile(fpath) || error("PPT cache file $fpath not found!")
    E, rate = @hlock HDF5.h5open(fpath, "r") do file
        (read(file["E"]), read(file["rate"]))
    end
    makePPTaccel(E, rate)
end

function makePPTcache(ionpot::Float64, λ0, Z, l;
                      sum_tol=1e-4, cycle_average=false, N=2^16, Emax=nothing)
    Emax = isnothing(Emax) ? 2*barrier_suppression(ionpot, Z) : Emax

    # ω0 = 2π*c/λ0
    # Emin = ω0*sqrt(2m_e*ionpot)/electron/0.5 # Keldysh parameter of 0.5
    Emin = Emax/5000

    E = collect(range(Emin, stop=Emax, length=N));
    @info "Pre-calculating PPT rate for $(ionpot/electron) eV, $(λ0*1e9) nm"
    rate = ionrate_PPT(ionpot, λ0, Z, l, E; sum_tol=sum_tol, cycle_average);
    @info "PPT pre-calcuation done"
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
`out`, like the functions returned by e.g. `ionrate_fun!_ADK` or `ionrate_fun!_PPTcached`.
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

function makePPTaccel(E, rate)
    # Interpolating the log and re-exponentiating makes the spline more accurate
    cspl = Maths.CSpline(E, log.(rate); bounds_error=true)
    Emin = minimum(E)
    Emax = maximum(E)
    function ionrate!(out, E)
        for ii in eachindex(out)
            aE = abs(E[ii])
            if aE < Emin
                out[ii] = 0.0
            elseif aE > Emax
                error(
                    "Field strength $aE V/m exceeds maximum for PPT ionisation rate ($Emax V/m)."
                    )
            else
                out[ii] = exp(cspl(aE))
            end
        end
    end
end

function ionrate_fun!_PPT(args...)
    ir = ionrate_fun_PPT(args...)
    function ionrate!(out, E)
        out .= ir.(E)
    end
    return ionrate!
end

"""
    ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4, cycle_average=false)

Create closure to calculate PPT ionisation rate.

# Keyword arguments
- `sum_tol::Number`: Relative tolerance used to truncate the infinite sum.
- `cycle_average::Bool`: If `false` (default), calculate the cycle-averaged rate

# References
[1] Ilkov, F. A., Decker, J. E. & Chin, S. L.
Ionization of atoms in the tunnelling regime with experimental evidence
using Hg atoms. Journal of Physics B: Atomic, Molecular and Optical
Physics 25, 4005–4020 (1992)

[2] 1.Bergé, L., Skupin, S., Nuter, R., Kasparian, J. & Wolf, J.-P.
Ultrashort filaments of light in weakly ionized, optically transparent
media. Rep. Prog. Phys. 70, 1633–1713 (2007)
(Appendix A)
"""
function ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4, cycle_average=false)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2Ip_au)
    ls = ns-1
    Cnl2 = 2^(2ns)/(ns*gamma(ns + ls + 1)*gamma(ns - ls))

    ω0 = 2π*c/λ0
    ω0_au = au_time*ω0
    E0_au = (2*Ip_au)^(3/2)

    ionrate = let ω0_au=ω0_au, Cnl2=Cnl2, ns=ns, sum_tol=sum_tol
        function ionrate(E)
            E_au = abs(E)/au_Efield
            γ = ω0_au*sqrt(2Ip_au)/E_au
            γ2 = γ*γ
            β = 2γ/sqrt(1 + γ2)
            α = 2*(asinh(γ) - γ/sqrt(1+γ2))
            Up_au = E_au^2/(4*ω0_au^2)
            Uit_au = Ip_au + Up_au
            v = Uit_au/ω0_au
            ret = 0
            divider = 0
            for m = -l:l
                divider += 1
                mabs = abs(m)
                flm = ((2l + 1)*factorial(l + mabs)
                    / (2^mabs*factorial(mabs)*factorial(l - mabs)))
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
                if !cycle_average
                    lret *= sqrt(π*E0_au/(3E_au))
                end
                k = ceil(v)
                n0 = ceil(v)
                sumfunc = let k=k, β=β, m=m
                    function sumfunc(x, n)
                        diff = n-v
                        return x + exp(-α*diff)*φ(m, sqrt(β*diff))
                    end
                end
                # s, success, steps = Maths.aitken_accelerate(
                #     sumfunc, 0, n0=n0, rtol=sum_tol, maxiter=Inf)
                s, success, steps = Maths.converge_series(
                    sumfunc, 0, n0=n0, rtol=sum_tol, maxiter=Inf)
                lret *= s
                ret += lret
            end
            return ret/(au_time*divider)
        end
    end
    return ionrate
end

"""
    φ(m, x)

Calculate the φ function for the PPT ionisation rate.

Note that w_m(x) in [1] and φ_m(x) in [2] look slightly different but
are in fact identical.
"""
function φ(m, x)
    mabs = abs(m)
    return (exp(-x^2)
            * sqrt(π)
            * x^(mabs+1)
            * gamma(mabs+1)
            * hypergeom(1/2, 3/2 + mabs, x^2)
            / (2*gamma(3/2 + mabs)))
end

function ionrate_fun_PPT(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_fun_PPT(ip, λ0, Z, l; kwargs...)
end

function ionrate_PPT(ionpot, λ0, Z, l, E; kwargs...)
    return ionrate_fun_PPT(ionpot, λ0, Z, l; kwargs...).(E)
end

function ionrate_PPT(material::Symbol, λ0, E; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_PPT(ip, λ0, Z, l, E; kwargs...)
end

end