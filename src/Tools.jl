module Tools
import Luna: Modes, PhysData, Capillary, RectModes

"Calculate 'natural' pulse width from FWHM" 
function τfw_to_τ0(τfw, shape)
    if shape == :sech
        τ0 = τfw/(2*log(1+sqrt(2)))
    else
        error("shape must be one of: :sech")
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

"Get soliotn order"
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

function capillary_params(E, τfw, λ, a, material; shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1, kind=:HE, ϕ=0.0)
    mode = Capillary.MarcatilliMode(a, material, P, n=n, m=m, kind=kind, ϕ=ϕ, T=T, clad=clad)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

function rectangular_params(E, τfw, λ, a, b, material; shape=:sech, P=1.0, T=PhysData.roomtemp, clad=:SiO2, n=1, m=1, pol=:x)
    mode = RectModes.RectMode(a, b, material, P, clad, T=T, n=n, m=m, pol=pol)
    params(E, τfw, λ, mode, material, shape=shape, P=P, T=T)
end

end