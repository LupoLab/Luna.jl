module SFA
import NumericalIntegration: integrate, SimpsonEven
import FFTW
import DataStructures: CircularBuffer
import Luna: Maths, PhysData, Ionisation
import Luna.PhysData: wlfreq, c


"""
    approx_dipole(gas)

Approximate transition dipole moment for hydrogen 1s as a function of momentum p.

B. Podolsky & L. Pauling. Phys. Rev.34 no. 1, 109 (1929)
"""

function approx_dipole(gas)
    Ip = PhysData.ionisation_potential(gas; unit=:atomic)
    κ = sqrt(2Ip)
    p -> 8*1im/π*sqrt(2*κ^5)*p/(p^2 + κ^2)^3
end

"""
    sfa_dipole(t, Et, gas, λ0; gate, nflat, nramp, depletion, irf!, dipole)

Calculate the time-dependent dipole moment of an atom driven by a laser field using the
strong-field approximation (SFA).

# Arguments
- `t::Vector{<:Real}` : The time axis
- `Et::Vector{<:Real}` : The driving electric field in SI units (V/m)
- `gas::Symbol` : The gas species of the atom
- `λ0::Real` : The central wavelength of the driving field

# Keyword arguments
- `gate::Bool` : Whether to remove long trajectories by apodising excursion times longer
                 than one half-cycle. Defaults to `true`.
                 If `gate` is `true`, this function uses `sfa_dipole_fast`
- `nflat::Number` : Number of field cycles for which the gate function is "fully open".
                    Defaults to 1/2.
- `nramp::Number` : Number of field cycles over which the gate function ramps down to 0.
                    Defaults to 1/4.
- `depletion::Bool` : Whether to include ground-state depletion. Defaults to `true`.
- `irf!::Function` : The ionisation-rate function to be used for ground-state depletion.
                     Defaults to the PPT ionisation rate for `gas` and `λ0` if `depletion`
                     is `true`.
- `dipole::Function` : Function `d(p)` which returns the transition dipole moment for the
                       momentum `p`. Defaults to `approx_dipole(gas)`.
"""
function sfa_dipole(t, Et::Vector{<:Real}, gas, λ0;
                    gate=true, nflat=1/2, nramp=1/4,
                    depletion=true,
                    irf! = depletion ? Ionisation.ionrate_fun!_PPTcached(gas, λ0) : nothing,
                    dipole=approx_dipole(gas))
    if gate
        return sfa_dipole_fast(t, Et, gas, λ0; nflat, nramp, depletion, irf!, dipole)
    end

    if depletion
        irate = similar(Et)
        irf!(irate, Et)
        Maths.cumtrapz!(irate, t)
        gstate_pop = exp.(-irate)
    else
        gstate_pop = ones(size(Et))
    end

    #==
    --------------------------------------------
    from here on, everythying is in atomic units
    --------------------------------------------
    ==#

    t = copy(t) ./ PhysData.au_time
    Et = copy(Et) ./ PhysData.au_Efield

    T0 = λ0/c / PhysData.au_time # period of the field (for gate function)
    ω0 = wlfreq(λ0)*PhysData.au_time # frequency of the field (for gate function)

    Ip = PhysData.ionisation_potential(gas; unit=:atomic)

    A = -Maths.cumtrapz(Et, t) # Vector potential A(t)
    intA = Maths.cumtrapz(A, t) # Antiderivative of A(t)
    intAsq = Maths.cumtrapz(A.^2, t) # Antiderivative of A^2(t)

    D = zeros(ComplexF64, size(Et))

    for (tidx, t_r) in enumerate(t)
        # t_r is the recombination time
        (tidx < 5) && continue

        t_b_idcs = 1:(tidx-1) # Birth time indices
        t_b = t[t_b_idcs] # Birth times

        τ = t_r .- t_b # excursion time -- time between birth and recombination

        if gate
            crop = τ .<= T0*(nflat+nramp+1/2)
            t_b = t_b[crop]
            t_b_idcs = t_b_idcs[crop]
            τ = τ[crop]
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
        if gate
            integrand .*= sine_squared_gate.(τ, ω0, nflat, nramp)
        end
        
        D[tidx] = 1im*integrate(t_b, integrand, SimpsonEven())
    end

    2*real(D)
end

function sfa_dipole_fast(t, Et::Vector{<:Real}, gas, λ0;
                         nflat=1/2, nramp=1/4,
                         depletion=true,
                         irf! = depletion ? Ionisation.ionrate_fun!_PPTcached(gas, λ0) : nothing,
                         dipole=approx_dipole(gas))

    if depletion
        irate = similar(Et)
        irf!(irate, Et)
        Maths.cumtrapz!(irate, t)
        gstate_pop = exp.(-irate)
    else
        gstate_pop = ones(size(Et))
    end

    #==
    --------------------------------------------
    from here on, everythying is in atomic units
    --------------------------------------------
    ==#

    t = copy(t) ./ PhysData.au_time
    Et = copy(Et) ./ PhysData.au_Efield

    T0 = λ0/c / PhysData.au_time # period of the field (for gate function)
    ω0 = wlfreq(λ0)*PhysData.au_time # frequency of the field (for gate function)

    Ip = PhysData.ionisation_potential(gas; unit=:atomic)

    A = -Maths.cumtrapz(Et, t) # Vector potential A(t)
    intA = Maths.cumtrapz(A, t) # Antiderivative of A(t)
    intAsq = Maths.cumtrapz(A.^2, t) # Antiderivative of A^2(t)

    D = zeros(ComplexF64, size(Et))

    δt = t[2] - t[1] # time sample spacing
    Tgate = T0*(nflat+nramp+1/8) # total length of time of the gate function
    Ngate = floor(Int, Tgate/δt)
    τ = collect(Ngate:-1:1)*δt # excursion time
    gate = sine_squared_gate.(τ, ω0, nflat, nramp) # gate function to remove long trajectories

    #= Gating reduces the integration region to a fixed range over which the gate function
        goes to zero. Since it's fixed, we can use circular buffers of fixed size to move
        along the time axis instead of allocating a new array each time.
    =#
    t_b = CircularBuffer{Float64}(Ngate) # birth times
    A_birth = fill!(CircularBuffer{Float64}(Ngate), 0) # vector potential at birth times
    intA_birth = fill!(CircularBuffer{Float64}(Ngate), 0) # integral of A from -∞ to birth times
    intAsq_birth = fill!(CircularBuffer{Float64}(Ngate), 0) # integral of A² from -∞ to birth times
    gstate_pop_birth = CircularBuffer{Float64}(Ngate) # ground-state population at birth times
    Et_this = fill!(CircularBuffer{Float64}(Ngate), 0) # electric field at birth times

    for i = Ngate:-1:1
        push!(t_b, t[1] - i*δt)
    end

    if depletion
        fill!(gstate_pop_birth, 0)
    else
        fill!(gstate_pop_birth, 1)
    end

    #= These quantities depend on the recombination time and thus need to be re-calculated
        for every step.
    =#
    intA_this = zeros(Float64, Ngate) # integral of A from birth times to recomb. time
    intAsq_this = zeros(Float64, Ngate) # integral of A² from birth times to recomb. time
    p_st = zeros(Float64, Ngate) # stationary momentum
    S = zeros(Float64, Ngate) # action
    d_birth = zeros(ComplexF64, Ngate) # dipole moment at birth times
    d_recomb = zeros(ComplexF64, Ngate) # dipole moment at recomb. time
    integrand = zeros(ComplexF64, Ngate) # integrand of the SFA integral

    prefac = @. (2*π/(1im*τ))^(3/2) * gate


    for tidx in eachindex(t)
        (tidx >= 2) || continue

        tm1 = tidx-1

        push!(t_b, t[tm1])

        push!(intA_birth, intA[tm1])
        push!(intAsq_birth, intAsq[tm1])
        push!(A_birth, A[tm1])
        push!(Et_this, Et[tm1])
        if depletion
            push!(gstate_pop_birth, gstate_pop[tm1])
        end

        intA_this .= intA[tidx] .- intA_birth
        intAsq_this .= intAsq[tidx] .- intAsq_birth

        p_st .= intA_this./τ
        @. S = Ip*τ + τ/2*p_st^2 - p_st*intA_this + intAsq_this/2 # Action

        for i in eachindex(d_birth)
            d_birth[i] = dipole(p_st[i] - A_birth[i])
            d_recomb[i] = dipole(p_st[i] - A[tidx])
        end

        gstate_pop_recomb = depletion ? gstate_pop[tidx] : 1.0

        @. integrand = (prefac
                     * d_birth*conj(d_recomb)
                     * gstate_pop_birth*gstate_pop_recomb
                     * Et_this
                     * exp(-1im*S))

        D[tidx] = 1im*integrate(t_b, integrand, SimpsonEven())
    end

    2*real(D)
end

"""
    sfa_spectrum(t, Et, gas, λ0; gate, nflat, nramp, depletion, irf!, dipole)

Calculate the HHG emission spectrum of an atom driven by a laser field using the
strong-field approximation (SFA). For arguments see [`sfa_dipole`](@ref)
"""
function sfa_spectrum(t, args...; kwargs...)
    D = sfa_dipole(t, args...; kwargs...)
    ω = Maths.rfftfreq(t)
    eV = PhysData.ħ * ω./PhysData.electron
    return eV, FFTW.rfft(D)
end

"""
    sine_squared_gate(t, ω, nflat, nramp)

Calculate the sin²-shaped gate function at time `t` for a field oscillating with angular
frequency `ω`. The gate is flat at 1.0 for `nflat` cycles of the field and ramps down to 0
over `nramp` cycles.
"""
function sine_squared_gate(t, ω, nflat, nramp)
    if ω*t/2π ≤ nflat
        return 1.0
    elseif ω*t/2π ≤ (nflat+nramp)
        return sin((2π*(nflat-nramp)-ω*t)/4nramp)^2
    else
        return 0.0
    end
end
end
