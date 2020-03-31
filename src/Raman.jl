module Raman
import Luna.PhysData: c, ε_0, ħ, k_B, roomtemp
import Luna.PhysData: raman_parameters
import Luna: Utils
import FFTW
import LinearAlgebra: mul!


struct RamanRespSingleDampedOscillator
    K::Float64 # overall scale factor
    Ω::Float64 # frequency
    τ2::Float64 # coherence dephasing time
end

function RamanRespNormedSingleDampedOscillator(K, Ω, τ2)
    K *= (Ω^2 + 1/τ2^2)/Ω
    RamanRespSingleDampedOscillator(K, Ω, τ2)
end

function (R::RamanRespSingleDampedOscillator)(t)
    h = 0.0
    if t > 0.0
        h = R.K*exp(-t/R.τ2)*sin(R.Ω*t)
    end
    return h
end

"
dαdQ - isotropic averaged polarizability derivative [m^2]
Ωv - vibrational frequency [rad/s]
μ - reduced molecular mass [kg]
τ2 - coherence time [s]
"
function RamanRespVibrational(Ωv, dαdQ, μ, τ2)
    # TODO we assume pressure independent linewidth which is incorrect
    K = (4π*ε_0)^2*dαdQ^2/(4μ*Ωv)
    RamanRespSingleDampedOscillator(K, Ωv, τ2)
end

# TODO: we assume here that all rotational levels have same linewidth
# TODO: we also assume pressure independent linewidth which is incorrect
struct RamanRespRotationalNonRigid
    K::Float64 # overall scale factor
    J::Array{Int,1} # J levels to sum over
    ρ::Array{Float64,1} # populations of each level
    Ω::Array{Float64,1} # frequency shift between each pair of levels
    τ2::Float64 # coherence dephasing time
end

"
T - time grid
B - rotational constant [1/m]
Δα - molecular polarizability anisotropy [m^3]
τ2 - coherence time [s]
qJodd - nuclear spin parameter
qJeven - nuclear spin parameter
D - centrifugal constant [1/m]
minJ - J value to start at
maxJ - J value to sum until
temp - temperature
"
function RamanRespRotationalNonRigid(B, Δα, τ2, qJodd::Int, qJeven::Int;
    D=0.0, minJ=0, maxJ=50, temp=roomtemp)
    J = minJ:maxJ
    EJ = @. 2π*ħ*c*(B*J*(J + 1) - D*(J*(J + 1))^2)
    mJ = findfirst(x -> x < 0.0, diff(EJ))
    if mJ != nothing
        if length(minJ:mJ) <= 2
            error("Raman rigid rotation model cannot sum over levels")
        end
        J = J[1:mJ]
        EJ = EJ[1:mJ]
    end
    DJ = map(x -> isodd(x) ? qJodd : qJeven, J)
    ρ = @. DJ*(2*J + 1)*exp(-EJ/(k_B*temp))
    ρ ./= sum(ρ)
    J = J[1:end-2]
    Ω = (EJ[3:end] .- EJ[1:end-2])./ħ
    K = -(4π*ε_0)^2*2/15*Δα^2/ħ
    RamanRespRotationalNonRigid(K, J, ρ, Ω, τ2)
end

function (R::RamanRespRotationalNonRigid)(t)
    h = 0.0
    if t > 0.0
        for i = eachindex(R.J)
            h += ((R.J[i] + 1)*(R.J[i] + 2)/(2*R.J[i] + 3)
                  *(R.ρ[i+2]/(2*R.J[i] + 5) - R.ρ[i]/(2*R.J[i] + 1))
                  *sin(R.Ω[i]*t)
                  *exp(-t/R.τ2))
        end
    end
    return h*R.K
end

function molecular_raman_response(rp; rotation=true, vibration=true, minJ=0, maxJ=50, temp=roomtemp)
    if rotation
        if rp.rotation != :nonrigid
            throw(DomainError(rp.rotation, "Unknown Rotational Raman model $(rp.rotation)"))
        end
    end
    if vibration
        if rp.vibration != :sdo
            throw(DomainError(rp.rotation, "Unknown Vibrational Raman model $(rp.vibration)"))
        end
    end 
    if rotation && vibration
        h = let hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.τ2r, rp.qJodd, rp.qJeven,
                                                  D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp),
                hv = RamanRespVibrational(rp.Ωv, rp.dαdQ, rp.μ, rp.τ2v)
            (t) -> hr(t) + hv(t)
        end 
    elseif rotation
        h = let hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.τ2r, rp.qJodd, rp.qJeven,
                                                 D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp)
            (t) -> hr(t)
        end 
    elseif vibration
        h = let hv = RamanRespVibrational(rp.Ωv, rp.dαdQ, rp.μ, rp.τ2v)
            (t) -> hv(t)
        end 
    else
        h = (t) -> 0.0
    end
    h
end

function raman_response(material; rotation=true, vibration=true, minJ=0, maxJ=50, temp=roomtemp)
    rp = raman_parameters(material)
    if rp.kind == :molecular
        return molecular_raman_response(rp, rotation=rotation, vibration=vibration,
                                         minJ=minJ, maxJ=maxJ, temp=temp)
    elseif rp.kind == :normedsdo
        return RamanRespNormedSingleDampedOscillator(rp.K, rp.Ω, rp.τ2)
    else
        throw(DomainError(rp.kind, "Unknown Raman model $(rp.kind)"))
    end
end

struct RamanPolarField{Tω, Tt, FTt}
    hω::Tω
    Eω2::Tω
    Pω::Tω
    E2::Tt
    P::Tt
    Pout::Tt
    FT::FTt
end

function RamanPolarField(T, ht)
    dt = T[2] - T[1]
    h = zeros(length(T)*2)
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=FFTW.PATIENT)
    inv(FT)
    Utils.saveFFTwisdom()
    fill!(h, 0.0)
    start = findfirst(T .> 0.0)
    for i = start:length(T)
        h[i] = ht(T[i])*dt
    end
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(h)
    P = similar(h)
    Pout = similar(T)
    RamanPolarField(hω, Eω2, Pω, E2, P, Pout, FT)
end

function (R::RamanPolarField)(out, Et)
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1
            E = reshape(Et, n)
        else
            error("vector Raman not yet implemented")
        end
    else
        E = Et
    end

    fill!(R.E2, 0.0)
    for i = eachindex(E)
        R.E2[i] = E[i]^2
    end
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2
    mul!(R.P, inv(R.FT), R.Pω)
    for i = eachindex(E)
        R.Pout[i] = E[i]*R.P[(n ÷ 2) + i] # TODO: possible off by 1 error
    end
    
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        @. out += R.Pout
    end
end
end
