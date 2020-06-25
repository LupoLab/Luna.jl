module Ionisation
import SpecialFunctions: gamma
import GSL: hypergeom
import HDF5
import Pidfile: mkpidlock
import Logging: @info
import Luna.PhysData: c, ħ, electron, m_e, au_energy, au_time, au_Efield
import Luna.PhysData: ionisation_potential, quantum_numbers
import Luna: Maths
import Luna: @hlock

function ionrate_fun!_ADK(ionpot::Float64, threshold=true)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    if threshold
        thr = ADK_threshold(ionpot)
    else
        thr = 0
    end

    ionrate! = let nstar=nstar, cn_sq=cn_sq, ω_p=ω_p, ω_t_prefac=ω_t_prefac, thr=thr
        function ir(E)
            if abs(E) >= thr
                (ω_p*cn_sq*
                (4*ω_p/(ω_t_prefac*abs(E)))^(2*nstar-1)
                *exp(-4/3*ω_p/(ω_t_prefac*abs(E))))
            else
                zero(E)
            end
        end
        function ionrate!(out, E)
            out .= ir.(E)
        end
    end

    return ionrate!  
end

function ionrate_fun!_ADK(material::Symbol)
    return ionrate_fun!_ADK(ionisation_potential(material))
end

function ionrate_ADK(IP_or_material, E)
    out = zero(E)
    ionrate_fun!_ADK(IP_or_material)(out, E)
    return out
end

function ionrate_ADK(IP_or_material, E::Number)
    out = [zero(E)]
    ionrate_fun!_ADK(IP_or_material)(out, [E])
    return out[1]
end

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

function ionrate_fun!_PPTaccel(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTaccel(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTaccel(ionpot::Float64, λ0, Z, l; kwargs...)
    E, rate = makePPTcache(ionpot, λ0, Z, l, kwargs...)
    return makePPTaccel(E, rate)
end

function ionrate_fun!_PPTcached(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTcached(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTcached(ionpot::Float64, λ0, Z, l;
                                sum_tol=1e-4, N=2^16, Emax=nothing,
                                cachedir=joinpath(homedir(), ".luna", "pptcache"))
    h = hash((ionpot, λ0, Z, l, sum_tol, N, Emax))
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
        E, rate = makePPTcache(ionpot::Float64, λ0, Z, l; sum_tol=sum_tol, N=N, Emax=Emax)
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

function makePPTcache(ionpot::Float64, λ0, Z, l; sum_tol=1e-4, N=2^16, Emax=nothing)
    Emax = isnothing(Emax) ? 5*barrier_suppression(ionpot, Z) : Emax

    # ω0 = 2π*c/λ0
    # Emin = ω0*sqrt(2m_e*ionpot)/electron/0.5 # Keldysh parameter of 0.5
    Emin = Emax/5000

    E = collect(range(Emin, stop=Emax, length=N));
    @info "Pre-calculating PPT rate for $(ionpot/electron) eV, $(λ0*1e9) nm"
    rate = ionrate_PPT(ionpot, λ0, Z, l, E);
    @info "PPT pre-calcuation done"
    return E, rate
end

function barrier_suppression(ionpot, Z)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2*Ip_au)
    Z^3/(16*ns^4) * au_Efield
end

function makePPTaccel(E, rate)
    cspl = Maths.CSpline(E, log.(rate); bounds_error=true)
    Emin = minimum(E)
    # Interpolating the log and re-exponentiating makes the spline more accurate
    ir(E) = abs(E) <= Emin ? 0.0 : exp(cspl(abs(E)))
    function ionrate!(out, E)
        out .= ir.(E)
    end
end

function ionrate_fun!_PPT(args...)
    ir = ionrate_fun_PPT(args...)
    function ionrate!(out, E)
        out .= ir.(E)
    end
    return ionrate!
end

function ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2*Ip_au)
    ls = ns-1
    Cnl2 = 2^(2*ns)/(ns*gamma(ns + ls + 1)*gamma(ns - ls))

    ω0 = 2π*c/λ0
    ω0_au = au_time*ω0
    E0_au = (2*Ip_au)^(3/2)

    ionrate = let ω0_au=ω0_au, Cnl2=Cnl2, ns=ns, sum_tol=sum_tol
        function ionrate(E)
            E_au = abs(E)/au_Efield
            g = ω0_au/sqrt(2Ip_au)/E_au
            g2 = g*g
            β = 2g/sqrt(1 + g2)
            α = 2*(asinh(g) - g/sqrt(1+g2))
            Up_au = E_au^2/(4*ω0_au^2)
            Uit_au = Ip_au + Up_au
            v = Uit_au/ω0_au
            G = 3/(2g)*((1 + 1/(2g2))*asinh(g) - sqrt(1 + g2)/(2g))
            ret = 0
            divider = 0
            for m = -l:l
                divider += 1
                mabs = abs(m)
                flm = ((2*l + 1)*factorial(l + mabs)
                    / (2 ^ mabs*factorial(mabs)*factorial(l - mabs)))
                Am = 4/(sqrt(3π)*factorial(mabs))*g2/(1 + g2)
                lret = sqrt(3/(2π))*Cnl2*flm*Ip_au
                lret *= (2*E0_au/(E_au*sqrt(1 + g2))) ^ (2ns - mabs - 3/2)
                lret *= Am*exp(-2*E0_au*G/(3E_au))
                # lret *= sqrt(π*E0_au/(3*E_au))
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

function φ(m, x)
    mabs = abs(m)
    return (exp(-x^2)
            * sqrt(π)
            * x^(mabs+1)
            * gamma(mabs+1)
            * hypergeom(1/2, 3/2 + mabs, x^2)
            / (2*gamma(3/2 + mabs)))
end

function ionrate_fun_PPT(material::Symbol, λ0)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_fun_PPT(ip, λ0, Z, l)
end

function ionrate_PPT(ionpot, λ0, Z, l, E)
    return ionrate_fun_PPT(ionpot, λ0, Z, l).(E)
end

function ionrate_PPT(material::Symbol, λ0, E)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_PPT(ip, λ0, Z, l, E)
end

end