module SFA
import NumericalIntegration: integrate, SimpsonEven
import FFTW
import Luna: Maths, PhysData, Ionisation
import Luna.PhysData: wlfreq, c
import PyPlot: plt


"""
    approx_dipole(gas)

Approximate transition dipole moment for hydrogen 1s as a function of momentum p.

Z. Chang. Fundamentals of attosecond optics. CRC Press, 2011. ISBN 9781420089370
"""
function approx_dipole(gas)
    Ip = PhysData.ionisation_potential(gas; unit=:atomic)

    p -> 2^(3/2)/π * (2*Ip)^(5/4)*p/(p^2 + 2*Ip)^3
end

"""
    sfa_dipole(t, Et, gas, λ0; apod, depletion, dipole, irf!)

Calculate the time-dependent dipole moment of an atom driven by a laser field using the
strong-field approximation (SFA).

# Arguments
- `t::Vector{<:Real}` : The time axis
- `Et::Vector{<:Real}` : The driving electric field in SI units (V/m)
- `gas::Symbol` : The gas species of the atom
- `λ0::Real` : The central wavelength of the driving field

# Keyword arguments
- `apod::Bool` : Whether to remove long trajectories by apodising excursion times longer
                 than one half-cycle. Defaults to `true`.
- `depletion::Bool` : Whether to include ground-state depletion. Defaults to `true`.
- `irf!::Function` : The ionisation-rate function to be used for ground-state depletion.
                     Defaults to the PPT ionisation rate for `gas` and `λ0`.
- `dipole::Function` : Function `d(p)` which returns the transition dipole moment for the
                       momentum `p`.

"""
function sfa_dipole(t, Et::Vector{<:Real}, gas, λ0; apod=true, depletion=true,
                    dipole=approx_dipole(gas),
                    irf! = Ionisation.ionrate_fun!_PPTcached(gas, λ0))
    irate = similar(Et)
    irf!(irate, Et)
    Maths.cumtrapz!(irate, t)
    gstate_pop = exp.(-irate)

    # from here on, everythying is in atomic units

    t = copy(t) ./ PhysData.au_time
    Et = copy(Et) ./ PhysData.au_Efield

    apodT = λ0/c / PhysData.au_time

    Ip = PhysData.ionisation_potential(gas; unit=:atomic)

    A = -Maths.cumtrapz(Et, t) # Vector potential A(t)
    intA = Maths.cumtrapz(A, t) # Antiderivative of A(t)
    intAsq = Maths.cumtrapz(A.^2, t) # Antiderivative of A(t)

    D = zeros(ComplexF64, size(Et))

    for (tidx, t_r) in enumerate(t)
        # t_r is the recombination time
        (tidx < 5) && continue

        t_b_idcs = 1:(tidx-1) # Birth time indices
        t_b = t[t_b_idcs] # Birth times

        τ = t_r .- t_b # excursion time -- time between birth and recombination

        if apod
            apod_crop = τ .<= 0.55apodT
            t_b = t_b[apod_crop]
            t_b_idcs = t_b_idcs[apod_crop]
            τ = τ[apod_crop]
        end
        
        intA_this = intA[tidx] .- intA[t_b_idcs] # definite integral between t_b and t_r
        intAsq_this = intAsq[tidx] .- intAsq[t_b_idcs] # definite integral between t_b and t_r

        p_st = intA_this./τ # Stationary momentum

        S = @. Ip*τ + τ/2*p_st^2 - p_st*intA_this + intAsq_this/2 # Action

        d_birth = dipole.(p_st .- A[t_b_idcs]) # transition dipole moment at birth times
        d_recomb = dipole.(p_st .- A[tidx]) # transition dipole moment at recombination

        if depletion
            gstate_pop_birth = gstate_pop[t_b_idcs] # ground state population at birth
            gstate_pop_recomb = gstate_pop[tidx] # ground state population at recombination
        else
            gstate_pop_birth = gstate_pop_recomb = 1.0
        end

        # main SFA integral
        integrand = @. ((2*π/(1im*τ))^(3/2)
                     * d_birth*conj(d_recomb)
                     * gstate_pop_birth*gstate_pop_recomb
                     * Et[t_b_idcs]
                     * exp(-1im*S))

        # filter out long trajectories
        if apod
            integrand .*= Maths.planck_taper.(τ, 0, 0, 0.5apodT, 0.55apodT)
        end
        
        D[tidx] = 1im*integrate(t_b, integrand, SimpsonEven())
    end

    2*real(D)
end

function sfa_spectrum(t, args...; kwargs...)
    D = sfa_dipole(t, args...; kwargs...)
    ω = Maths.rfftfreq(t)
    eV = PhysData.ħ * ω./PhysData.electron
    return eV, FFTW.rfft(D)
end
end