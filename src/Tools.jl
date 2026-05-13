module Tools
import Luna: Modes, PhysData, Capillary, RectModes, Maths
import Luna.PhysData: wlfreq
import Roots: find_zero
import Base: show
import Cubature: hquadrature
import Printf: @sprintf

"Calculate 'natural' pulse width from FWHM" 
function τfw_to_τ0(τfw, shape)
    if shape == :sech
        τ0 = τfw/(2*log(1+sqrt(2)))
    elseif shape == :gauss
        τ0 = τfw/(2*sqrt(log(2)))
    else
        error("shape must be one of: :sech, :gauss")
    end
    τ0
end

"Get dispersion length"
function Ld(τfw, β2; shape=:sech)
    τ0 = τfw_to_τ0(τfw, shape)
    τ0^2/abs(β2)
end

"Get GVD coefficient"
function getβ2(ω, m::Modes.AbstractMode)
    Modes.dispersion(m, 2, ω)
end

"Get nonlinear length"
function Lnl(P0, γ)
    1/(γ*P0)
end

"Get fission length"
function Lfiss(P0, τfw, γ, β2; shape=:sech)
    Ld(τfw, β2, shape=shape)/getN(P0, τfw, γ, β2, shape=shape)
end

"Get nonlinear coefficient"
function getγ(ω, m::Modes.AbstractMode, n2)
    n2*ω/(PhysData.c*Modes.Aeff(m))
end

"Get linear and nonlinear refractive index and gas number density"
function getN0n0n2(ω, material; P=1.0, T=PhysData.roomtemp)
    N0 = PhysData.density(material, P, T)
    χ3 = PhysData.χ3(material, P, T)
    n0 = real(PhysData.ref_index(material, 2π*PhysData.c/ω, P, T))
    N0, n0, 3*χ3/(4*n0^2*PhysData.ε_0*PhysData.c)
end

"Get soliton order"
function getN(P0, τfw, γ, β2; shape=:sech)
    sqrt(Ld(τfw, β2, shape=:sech)/Lnl(P0, γ))
end

function E_to_P0(E, τfw; shape=:sech)
    τ0 = τfw_to_τ0(τfw, shape)
    if shape == :sech
        P0 = E/(2*τ0)
    else
        error("shape must be one of: :sech")
    end
    P0
end

function P0_to_I(P0, m)
    P0/Modes.Aeff(m)
end

function Pcr(ω, n0, n2)
    # G. Fibich and A. L. Gaeta, Optics Letters, 25, 5, 335, 2000, doi: 10.1364/OL.25.000335.
    1.86225*(2π*PhysData.c/ω)^2/(4π*n0*n2)
end

paramfields = (:E, :τfw, :τ0, :ω, :λ, :material, :P, :T, :shape,:P0, :β2, :N0, :n0, :n2,
               :γ, :N, :I0, :Pcr, :Ld, :Lnl, :Lfiss, :zdw, :Lloss, :Aeff, :mode)

function show(io::IO, p::NamedTuple{paramfields, vT}) where vT
    mode = "MODE:\n  $(p.mode)"
    fill = @sprintf("FILL:\n  %.1f bar %s, Pcr = %.1e W, γ = %.1e (Wm)^-1, n2 = %.1e cm^2/W",
                     p.P, p.material, p.Pcr, p.γ, p.n2*1e4)
    wg = @sprintf("WAVEGUIDE:\n  Aeff = %.1e m^2, Lloss = %.1e m", p.Aeff, p.Lloss)
    pulse = @sprintf("PULSE:\n  %.2e J, %.2e s @ %.1f nm (shape: %s)",
                      p.E, p.τfw, p.λ*1e9, p.shape)
    dispersion = @sprintf("DISPERSION:\n  %.2e s^2/m @ %.1f nm, ZDW = %.1f nm",
                          p.β2, p.λ*1e9, p.zdw*1e9)
    intensity = @sprintf("INTENSITY:\n  %.1e W/cm^2", p.I0*1e-4)
    power = @sprintf("POWER:\n  %.1e W (%.4f of Pcr)", p.P0, p.P0/p.Pcr)
    sol = @sprintf("SOLITON:\n  Ld = %.1e m, Lnl = %.1e m, Lfiss = %.1e m, N = %.2f",
                   p.Ld, p.Lnl, p.Lfiss, p.N)
    out = join((mode, wg, fill, pulse, dispersion, power, intensity, sol), "\n")
    print(io, out)
end

"Soliton parameter collection"
function params(E, τfw, λ, mode, material; shape=:sech, P=1.0, T=PhysData.roomtemp)
    ω = 2π*PhysData.c/λ
    P0 = E_to_P0(E, τfw, shape=shape)
    τ0 = τfw_to_τ0(τfw, shape)
    β2 = getβ2(ω, mode)
    N0, n0, n2 = getN0n0n2(ω, material, P=P, T=T)
    γ = getγ(ω, mode, n2)
    N = getN(P0, τfw, γ, β2, shape=shape)
    zdw = Modes.zdw(mode)
    if ismissing(zdw)
        zdw = Modes.zdw(mode, λ)
    end
    p = (E=E, τfw=τfw, τ0=τ0, ω=ω, λ=λ, material=material, P=P, T=T, shape=shape,
         P0=P0, β2=β2, N0=N0, n0=n0, n2=n2, γ=γ, N=N,
         I0=P0_to_I(P0, mode), Pcr=Pcr(ω, n0, n2),
         Ld=Ld(τfw, β2, shape=shape),
         Lnl=Lnl(P0, γ),
         Lfiss=Lfiss(P0, τfw, γ, β2, shape=shape),
         zdw=zdw,
         Lloss=Modes.losslength(mode, ω),
         Aeff=Modes.Aeff(mode),
         mode=mode)
end

function capillary_params(E, τfw, λ, a, material;
                          shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1,kind=:HE, ϕ=0.0)
    mode = Capillary.MarcatiliMode(a, material, P, n=n, m=m, kind=kind, ϕ=ϕ, T=T, clad=clad)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

function rectangular_params(E, τfw, λ, a, b, material;
                            shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1, 
                            pol=:x)
    mode = RectModes.RectMode(a, b, material, P, clad, T=T, n=n, m=m, pol=pol)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

function gas_ratio(gas1, gas2, λ)
    χ3r = PhysData.χ3(gas1, 1) / PhysData.χ3(gas2, 1)
    β2r = PhysData.dispersion(2, gas1, λ) / PhysData.dispersion(2, gas2, λ)
    β2r, χ3r
end

field_to_intensity(E) = 0.5*PhysData.ε_0*PhysData.c*E^2
intensity_to_field(I) = sqrt(2I/PhysData.ε_0/PhysData.c)

"""
    λRDW(m::Modes.AbstractMode, λ0; z=0, λlims=(100e-9, 0.9λ0))
    λRDW(mRDW::Modes.AbstractMode, mS::Modes.AbstractMode, λ0; z=0, λlims=(100e-9, 0.9λ0))

Calculate the phase-matching wavelength for resonant dispersive wave (RDW) emission in the
mode `m` when pumping at `λ0`. If the dispersive-wave mode `mRDW` and soliton mode `mS` are
given separately, calculate phase-matching for RDW in mode `mRDW` when pumping in mode `mS`.

This neglects the nonlinear contribution to the phase mismatch.
"""
function λRDW(m::Modes.AbstractMode, λ0; z=0, λlims=(100e-9, 0.9λ0))
    ω0 = wlfreq(λ0)
    β1 = Modes.dispersion(m, 1, ω0; z=z)
    β0 = Modes.β(m, ω0; z=z)
    Δβ(ω) = Modes.β(m, ω; z=z) - β1*(ω.-ω0) - β0
    try
        ωRDW = find_zero(Δβ, extrema(wlfreq.(λlims)))
        wlfreq(ωRDW)
    catch
        missing
    end
end

function λRDW(mRDW::Modes.AbstractMode, mS::Modes.AbstractMode, λ0; z=0, λlims=(100e-9, 0.9λ0))
    ω0 = wlfreq(λ0)
    β1 = Modes.dispersion(mS, 1, ω0; z=z)
    β0 = Modes.β(mS, ω0; z=z)
    Δβ(ω) = Modes.β(mRDW, ω; z=z) - β1*(ω.-ω0) - β0
    try
        ωRDW = find_zero(Δβ, extrema(wlfreq.(λlims)))
        wlfreq(ωRDW)
    catch
        missing
    end
end

"""
    λRDW(a::Number, gas::Symbol, pressure, λ0; λlims=(100e-9, 0.9λ0), kwargs...)

Calculate the phase-matching wavelength for resonant dispersive wave (RDW) emission in a 
capillary with core radius `a` filled with `gas` at a certain `pressure`
when pumping at `λ0`. Additional `kwargs` are passed onto `Capillary.MarcatiliMode`.

This neglects the nonlinear contribution to the phase mismatch.
"""
function λRDW(a::Number, gas::Symbol, pressure, λ0; λlims=(100e-9, 0.9λ0), kwargs...)
    m = Capillary.MarcatiliMode(a, gas, pressure; kwargs...)
    λRDW(m, λ0; λlims=λlims)
end

"""
    pressureRDW(a::Number, gas::Symbol, λ_target, λ0; Pmax=100, clad=:SiO2, kwargs...)

Calculate the phase-matching pressure for resonant dispersive wave (RDW) emission at
`λ_target` in a capillary with core radius `a` filled with `gas` when pumping at `λ0`. 
"""
function pressureRDW(a::Number, gas::Symbol, λ_target, λ0; Pmax=100, clad=:SiO2, kwargs...)
    # cladn is likely based on interpolation so requires creating the BSpline.
    # By creating the function here we only have to do that once
    rfc = PhysData.ref_index_fun(clad)
    cladn = (ω; z) -> rfc(wlfreq(ω))
    ω0 = wlfreq(λ0)
    ω_target = wlfreq(λ_target)
    function Δβ(P)
        m = Capillary.MarcatiliMode(a, gas, P, cladn; kwargs...)
        β1 = Modes.dispersion(m, 1, ω0)
        β0 = Modes.β(m, ω0)
        Modes.β(m, ω_target) - β1*(ω_target.-ω0) - β0
    end

    try
        find_zero(Δβ, (1e-6, Pmax))
    catch
        missing
    end
end

function pressureZDW(a::Number, gas::Symbol, λzd; Pmax=100, clad=:SiO2, kwargs...)
    rfc = PhysData.ref_index_fun(clad)
    cladn = (ω; z) -> rfc(wlfreq(ω))
    ωzd = wlfreq(λzd)

    try
        find_zero((1e-6, Pmax)) do P
            m = Capillary.MarcatiliMode(a, gas, P, cladn; kwargs...)
            Modes.dispersion(m, 2, ωzd)
        end
    catch
        missing
    end
end



"""
    Calculates the critical electron density for a given wavelength, which is the density at which the plasma frequency equals the frequency of the light
    Ref: https://doi.org/10.1016/j.physrep.2006.12.005, top of page 56
    Arguments:
        λ: the wavelength of light
"""
function critical_electron_density(λ)
    ω = PhysData.wlfreq(λ)
    return (PhysData.ε_0*PhysData.m_e*ω^2)/PhysData.electron^2
end


"""
    Calculates the energy required for a certain soliton order
    Arguments:
        a: radius of the capillary core
        gas: type of gas in the capillary   
        pressure: pressure of the gas in the capillary
        τFWHM: pulse duration (FWHM)
        λp: central wavelength of the pump (soliton) pulse
        N: soliton order
        energy_lims: limits for the root finding algorithm that looks for the energy; by default these are set to 1 nJ to 1 J
"""
function energyN(a, gas, pressure, τFWHM, λp, N; energy_lims=(1e-9, 1.0))

    EforN = find_zero(energy_lims) do energy
        params = Tools.capillary_params(energy, τFWHM, λp, a, gas; P=pressure)
        params.N - N
    end
    
    return EforN

end

"""
    Helper function to calculate the linear contribution to the phase-mismatch Δβ, which is used in the phase-matching condition for RDW emission
"""
function Δβlin(a, gas, pressure, λp)

    mode = Capillary.MarcatiliMode(a, gas, pressure)
    ωsol = PhysData.wlfreq(λp)
    β1 = Modes.dispersion(mode, 1, ωsol)
    β0 = Modes.β(mode, ωsol)
    Δβlin(ω) = Modes.β(mode, ω) - β1*(ω - ωsol) - β0

    return Δβlin

end

"""
    Calculate the estimated compression factor and quality factor of the soliton self-compression 
    Schade et al. "Scaling rules for high quality soliton self-compression in hollow-core fibers" https://doi.org/10.1364/OE.426307
    Arguments:
        N: soliton order
    Returns compression factor and compression quality (Fc, Q)
"""
function compression_estimation(a, gas, pressure, λp, τFWHM, N; include_ξ=false)

    A(N) = (1/(2*N)+1.7/(N^2))
    Q(N) = 3.7/(N + 2.2)
    
    ωsol = PhysData.wlfreq(λp)
    mode = Capillary.MarcatiliMode(a, gas, pressure)
    β2 = Modes.dispersion(mode, 2, ωsol)
    β3 = Modes.dispersion(mode, 3, ωsol)
    ξ = β3/(τFWHM*abs(β2))

    Fc = 3/(A(N))

    if include_ξ
        Fc *= (1-N*ξ)
    end

    return Fc, Q(N)
end

function Ppeak(τFWHM, energy; pulse_shape=:sech)

    if pulse_shape == :sech
        return 0.88*(energy/τFWHM)
    elseif pulse_shape == :gauss
        return 0.94*(energy/τFWHM)
    else
        error("pulse_shape must be one of: :sech, :gauss")
    end

end

"""
    Helper function to calculate the nonlinear contribution to the phase-mismatch Δβ, which is used in the phase-matching condition for RDW emission
"""
function Δβnonlin(a, gas, pressure, τFWHM, λp, soliton_order; includeLoss=false, input_pulse_shape=:gauss)

    energy = energyN(a, gas, pressure, τFWHM, λp, soliton_order)
    params = Tools.capillary_params(energy, τFWHM, λp, a, gas; P=pressure)
    ωsol = PhysData.wlfreq(λp)

    if input_pulse_shape == :sech
        Pp = Ppeak(τFWHM, energy, pulse_shape=:sech)
    elseif input_pulse_shape == :gauss
        Pp = 0.936*Ppeak(τFWHM, energy, pulse_shape=:gauss) # 0.88/0.94
    else
        error("input_pulse_shape must be one of: :sech, :gauss")
    end

    soliton_factor = ((2*soliton_order-1)/soliton_order)^2 # ((2N-1)/N)^2

    attenuation = 1.0 

    if includeLoss
        Lfiss = params.Lfiss
        mode = Capillary.MarcatiliMode(a, gas, pressure) 
        attenuation = Modes.α(mode, PhysData.wlfreq(λp); z=Lfiss)
    end

    return params.γ*(soliton_factor*Pp*attenuation)*(1/ωsol)
    
end

"""
    Helper function to calculate the ionisation contribution to the phase-mismatch Δβ, which is used in the phase-matching condition for RDW emission
"""
function Δβion(a, gas, pressure, λp, ionisation_fraction)
    neutral_density = PhysData.density(gas, pressure)
    electron_density = ionisation_fraction*neutral_density
    critical_density = critical_electron_density(λp)
    ωsol = PhysData.wlfreq(λp)
    mode = Capillary.MarcatiliMode(a, gas, pressure)
    nlin = real.(Capillary.neff(mode, ωsol)) # taking the real part here, otherwise the ionisation term becomes complex
    return (1/(2*PhysData.c*nlin))*(electron_density/critical_density)*(ωsol^2)
end

"""
    A helper type to store parameters for phase-matching
"""
struct PhaseMatching
    mode
    ωsol
    β0
    β1
    ΔβnonlinCoeff
    ΔβionCoeff
end

"""
    A helper function to calculate the phase-matching curve from pre-calculated parameters before passing to the root-finding algorithm; this speeds up the calculation a lot compared to when just passing Δβ as a full function, that has to be evaluated at every iteration of the root-finding
"""
function (pm::PhaseMatching)(ω)

    return Modes.β(pm.mode, ω) - pm.β1*(ω - pm.ωsol) - pm.β0 - pm.ΔβnonlinCoeff*ω + pm.ΔβionCoeff*(1/ω)

end

"""
    A helper function to pre-compute parameters for phase-matching
"""
function make_PhaseMatching(a, gas, pressure, λp;
                            include_Δβnonlin=false, soliton_order=nothing, τFWHM=nothing, includeLoss=false, input_pulse_shape=:gauss,
                            include_Δβion=false, ionisation_fraction=nothing)

    mode = Capillary.MarcatiliMode(a, gas, pressure)
    ωsol = PhysData.wlfreq(λp)
    β1 = Modes.dispersion(mode, 1, ωsol)
    β0 = Modes.β(mode, ωsol)

    if include_Δβnonlin # perform the assertions only if the nonlinear contribution is included, otherwise these parameters are not needed and can be left as nothing
        @assert !isnothing(soliton_order) "If include_Δβnonlin is true, soliton_order must be given"
        @assert soliton_order > 0 "Soliton order must be positive"
        @assert !isnothing(τFWHM) "If include_Δβnonlin is true, τFWHM must be given"
    end

    if include_Δβion # perform the assertions only if the ionisation contribution is included, otherwise these parameters are not needed and can be left as nothing
        @assert !isnothing(ionisation_fraction) "If include_Δβion is true, ionisation_fraction must be given"
        @assert ionisation_fraction >= 0 && ionisation_fraction <= 1 "Ionisation fraction must be between 0 and 1"
    end

    ΔβnonlinCoeff = include_Δβnonlin ? Δβnonlin(a, gas, pressure, τFWHM, λp, soliton_order; includeLoss=includeLoss,input_pulse_shape=input_pulse_shape) : 0.0
    ΔβionCoeff = include_Δβion ? Δβion(a, gas, pressure, λp, ionisation_fraction) : 0.0

    return PhaseMatching(mode, ωsol, β0, β1, ΔβnonlinCoeff, ΔβionCoeff)
end

"""
    Function for calculating the phase-matched resonant dispersive wave (RDW) wavelength in a gas-filled capillary, taking into account the nonlinear and ionisation contributions to phase-matching.
    Ref: PRL 115, 033901 (2015), https://doi.org/10.1103/PhysRevLett.115.033901
    Arguments:
        a: radius of the capillary core
        gas: type of gas in the capillary   
        pressure: pressure of the gas in the capillary
        λp: central wavelength of the pump (soliton) pulse
        kwargs: additional keyword arguments for the mode (e.g. m, n, ϕ, kind, clad; used in calling MarcatiliMode() line 95 in Luna.Capillary)
        soliton_order: used to calculate the energy/peak power
        τFWHM: pulse duration (FWHM)
        ionisation_fraction: electron fraction
        λlims: limits for the root finding algorithm that looks for the phase-matching; by default these are set to the minimum wavelength just above the first resonance in the Sellmeier equation for the given gas and the maximum for the pump wavelength minus 1 nm
"""
function λRDWfull(a, gas, pressure, λp;
                  include_Δβnonlin=false, soliton_order=nothing, τFWHM=nothing, includeLoss=false, input_pulse_shape=:gauss,
                  include_Δβion=false, ionisation_fraction=nothing,
                  λlims=nothing)

    if isnothing(λlims)
        λUVlim = Dict(
            :He => 90e-9,
            :HeB => 90e-9,
            :HeJ => 90e-9,
            :Ne => 100e-9,
            :NeBideauMehu => 100e-9,
            :Ar => 110e-9,
            :ArB => 110e-9,
            :ArBideauMehu => 110e-9,
            :Kr => 130e-9,
            :KrB => 130e-9,
            :KrBideauMehu => 130e-9
        )
        λlims = (λUVlim[gas], λp-10e-9) # pump wavelength minus 10 nm for safety
    end

    phase_matching = make_PhaseMatching(a, gas, pressure, λp;
                                        include_Δβnonlin=include_Δβnonlin, soliton_order=soliton_order, τFWHM=τFWHM, includeLoss=includeLoss, input_pulse_shape=input_pulse_shape,
                                        include_Δβion=include_Δβion, ionisation_fraction=ionisation_fraction)

    # if ionization effects are included, shrink λlims, because the dispersion curve gets lifted in the IR and a new phase-matched point appears near the pump
    if include_Δβion

        # first find the linear phase-matching wavelength
        λlin = Tools.λRDW(a, gas, pressure, λp; λlims=λlims) # use the OG Luna function to calculate the linear phase-matched RDW
        
        # find the root close to the pump wavelength
        ωmax = NaN
        try 
            @assert !ismissing(λlin) "λlin should not be NaN here, the first root-finding has failed"
            ωmax = find_zero(phase_matching, extrema(PhysData.wlfreq.((λlin+50e-9, λp-10e-9)))) # linear RDW + 50 nm for safety
            λlims = (minimum(λlims), PhysData.wlfreq(ωmax) - 10e-9) 
        catch err
            @warn "A phase-matching wavelength couldn't be found for Δβ..." exception=(err, catch_backtrace())
        end

    end

    # find RDW phase-matched wavelength with the full Δβ and potentially shrunk λlims
    ωRDW = NaN
    try 
        ωRDW = find_zero(phase_matching, extrema(PhysData.wlfreq.(λlims))) 
    catch err
        @warn "A phase-matching wavelength couldn't be found for the given parameters..." exception=(err, catch_backtrace())
    end

    return PhysData.wlfreq(ωRDW)

end

function fλ(λ, material, pressure; temperature=PhysData.roomtemp)

    χ1function = PhysData.χ1_fun(material, pressure, temperature)

    return Maths.derivative(χ1function, λ, 2)

end


function δ(λ, zdw, material, pressure; kind=:HE, n=1, m=1)

    unm = Capillary.get_unm(n, m, kind)

    return ((unm^2*λ^3)/(8*π^3*PhysData.c^2))*((fλ(λ, material, pressure))/(fλ(zdw, material, pressure)) - 1)

end

function NmaxSelfFocusing(λp, zdw, gas, τFWHM, a; safetyFactor=10, pulseShape=:gauss, kind=:HE, n=1, m=1)
    
    pressure = pressureZDW(a, gas, zdw)
    τ0 = τfw_to_τ0(τFWHM, pulseShape)

    return sqrt((τ0^2*λp)/(safetyFactor*abs(δ(λp, zdw, gas, pressure; kind=kind, n=n, m=m))))

end

function NmaxIonisation(λp, zdw, gas, τFWHM, a; safetyFactor=10, pulseShape=:gauss, kind=:HE, n=1, m=1)

    pressure = pressureZDW(a, gas, zdw)
    τ0 = τfw_to_τ0(τFWHM, pulseShape)
    ωpump = PhysData.wlfreq(λp)
    _, _, n2 = getN0n0n2(ωpump, gas; P=pressure)
    unm = Capillary.get_unm(n, m, kind)

    return sqrt((τ0^2*n2*PhysData.PhysData.Ith(gas)*unm^2)/(safetyFactor*π*λp*abs(δ(λp, zdw, gas, pressure; kind=kind, n=n, m=m))*fλ(zdw, gas, pressure)))

end

function aperture_filter(a, dist, radius)
    w0 = 0.64*a
    function filter(λ)
        w1 = dist*λ/(π*w0)
        I(r) = Maths.gauss(r, w1/2) / Maths.gaussnorm(w1/2)
        2*hquadrature(I, 0, radius)[1]
    end
end

end