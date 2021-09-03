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
    χ3 = PhysData.γ3_gas(material)*N0
    n0 = PhysData.ref_index(material, 2π*PhysData.c/ω, P, T)
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
    1.8962*(2π*PhysData.c/ω)^2/(4π*n0*n2)
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
    p = (E=E, τfw=τfw, τ0=τ0, ω=ω, λ=λ, material=material, P=P, T=T, shape=shape,
         P0=P0, β2=β2, N0=N0, n0=n0, n2=n2, γ=γ, N=N,
         I0=P0_to_I(P0, mode), Pcr=Pcr(ω, n0, n2),
         Ld=Ld(τfw, β2, shape=shape),
         Lnl=Lnl(P0, γ),
         Lfiss=Lfiss(P0, τfw, γ, β2, shape=shape),
         zdw=Modes.zdw(mode),
         Lloss=Modes.losslength(mode, ω),
         Aeff=Modes.Aeff(mode),
         mode=mode)
end

function capillary_params(E, τfw, λ, a, material;
                          shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1,kind=:HE, ϕ=0.0)
    mode = Capillary.MarcatilliMode(a, material, P, n=n, m=m, kind=kind, ϕ=ϕ, T=T, clad=clad)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

function rectangular_params(E, τfw, λ, a, b, material;
                            shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1, 
                            pol=:x)
    mode = RectModes.RectMode(a, b, material, P, clad, T=T, n=n, m=m, pol=pol)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

function gas_ratio(gas1, gas2, λ)
    χ3r = PhysData.χ3_gas(gas1, 1) / PhysData.χ3_gas(gas2, 1)
    β2r = PhysData.dispersion(2, gas1, λ) / PhysData.dispersion(2, gas2, λ)
    β2r, χ3r
end

field_to_intensity(E) = 0.5*PhysData.ε_0*PhysData.c*E^2
intensity_to_field(I) = sqrt(2I/PhysData.ε_0/PhysData.c)

"""
    λRDW(m::Modes.AbstractMode, λ0; z=0, λlims=(100e-9, 0.9λ0))

Calculate the phase-matching wavelength for resonant dispersive wave (RDW) emission in the
mode `m` when pumping at `λ0`. 

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

"""
    λRDW(a::Number, gas::Symbol, pressure, λ0; λlims=(100e-9, 0.9λ0), kwargs...)

Calculate the phase-matching wavelength for resonant dispersive wave (RDW) emission in a 
capillary with core radius `a` filled with `gas` at a certain `pressure`
when pumping at `λ0`. Additional `kwargs` are passed onto `Capillary.MarcatilliMode`.

This neglects the nonlinear contribution to the phase mismatch.
"""
function λRDW(a::Number, gas::Symbol, pressure, λ0; λlims=(100e-9, 0.9λ0), kwargs...)
    m = Capillary.MarcatilliMode(a, gas, pressure; kwargs...)
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
        m = Capillary.MarcatilliMode(a, gas, P, cladn; kwargs...)
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

function aperture_filter(a, dist, radius)
    w0 = 0.64*a
    function filter(λ)
        w1 = dist*λ/(π*w0)
        I(r) = Maths.gauss(r, w1/2) / Maths.gaussnorm(w1/2)
        2*hquadrature(I, 0, radius)[1]
    end
end

end