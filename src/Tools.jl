module Tools
import Luna: Modes, PhysData

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
function getβ2(ω, m::M) where M <: Modes.AbstractMode
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
function getγ(ω, m::M, n2) where M <: Modes.AbstractMode
    n2*ω/(PhysData.c*Modes.Aeff(m))
end

"Get linear and nonlinear refractive index"
function getn0n2(ω, material; P=1.0, T=PhysData.roomtemp)
    χ3 = PhysData.γ3_gas(material)*PhysData.density(material, P, T)
    n0 = PhysData.ref_index(material, 2π*PhysData.c/ω, P, T)
    n0, 3*χ3/(4*n0^2*PhysData.ε_0*PhysData.c)
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
function params(E, τfw, ω, m::M, material; shape=:sech, P=1.0, T=PhysData.roomtemp) where M <: Modes.AbstractMode
    P0 = E_to_P0(E, τfw, shape=shape)
    τ0 = τfw_to_τ0(τfw, shape)
    β2 = getβ2(ω, m)
    n0, n2 = getn0n2(ω, material, P=P, T=T)
    γ = getγ(ω, m, n2)
    N = getN(P0, τfw, γ, β2, shape=shape)
    p = (E=E, τfw=τfw, τ0=τ0, ω=ω, material=material, P=P, T=T, shape=shape,
         P0=P0, β2=β2, n0=n0, n2=n2, γ=γ, N=N,
         I0=P0_to_I(P0, m), Pcr=Pcr(ω, n0, n2),
         Ld=Ld(τfw, β2, shape=shape),
         Lnl=Lnl(P0, γ),
         Lfiss=Lfiss(P0, τfw, γ, β2, shape=shape),
         λz=Modes.zdw(m),
         Lloss=Modes.losslength(m,ω),
         Aeff=Modes.Aeff(m))
end

end