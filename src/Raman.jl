module Raman
import Luna.PhysData: c, ε_0, ħ, k_B, roomtemp
import Luna.PhysData: raman_parameters

struct RamanRespSingleDampedOscillator
    K::Float64 # overall scale factor
    Ω::Float64 # frequency
    τ2::Float64 # coherence dephasing time
end

"""
    RamanRespNormedSingleDampedOscillator(K, Ω, τ2)

Construct a simple normalised single damped oscillator model with scale factor `K`,
angular frequency `Ω` and coherence time `τ2`.

The scale factor `K` is applied after normalising the integral of the response function
to unity.
"""
function RamanRespNormedSingleDampedOscillator(K, Ω, τ2)
    K *= (Ω^2 + 1/τ2^2)/Ω # normalise SDO model to unity integral, scaled by prefactor K.
    RamanRespSingleDampedOscillator(K, Ω, τ2)
end

"Get the response function at time `t`."
function (R::RamanRespSingleDampedOscillator)(t)
    h = 0.0
    if t > 0.0
        h = R.K*exp(-t/R.τ2)*sin(R.Ω*t)
    end
    return h
end

"""
    RamanRespVibrational(Ωv, dαdQ, μ, τ2)

Construct a molecular vibrational Raman model (single damped oscillator).

# Arguments
- `Ωv::Real`: vibrational frequency [rad/s]
- `dαdQ::Real`: isotropic averaged polarizability derivative [m^2]
- `μ::Real`: reduced molecular mass [kg]
- `τ2::Real`: coherence time [s]

# References
- Full model description: To be published, Yimgying paper.
- We followed closely: Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
  But note that that paper uses weird units, and we converted it to SI for
  the above reference. 
"""
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

"""
    RamanRespRotationalNonRigid(B, Δα, τ2, qJodd, qJeven;
                                D=0.0, minJ=0, maxJ=50, temp=roomtemp)

Construct a rotational nonrigid rotor Raman model.

# Arguments
- `B::Real`: the rotational constant [1/m]
- `Δα::Real`: molecular polarizability anisotropy [m^3]
- `τ2::Real`: coherence time [s]
- `qJodd::Integer`: nuclear spin parameter for odd `J`
- `qJeven::Integer`: nuclear spin parameter for even `J`
- `D::Real=0.0`: centrifugal constant [1/m]
- `minJ::Integer=0`: J value to start at
- `maxJ::Integer=50`: J value to sum until
- `temp::Real=roomtemp`: temperature

# References
- Full model description: To be published, Yimgying paper.
- We followed closely: Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
  But note that that paper uses weird units, and we converted it to SI for
  the above reference. 
"""
function RamanRespRotationalNonRigid(B, Δα, τ2, qJodd::Int, qJeven::Int;
    D=0.0, minJ=0, maxJ=50, temp=roomtemp)
    J = minJ:maxJ # range of J values to start with
    EJ = @. 2π*ħ*c*(B*J*(J + 1) - D*(J*(J + 1))^2) # energy of each J level
    # limit J range to those which have monotonic increasing energy
    mJ = findfirst(x -> x < 0.0, diff(EJ))
    if mJ != nothing
        if length(minJ:mJ) <= 2 # need at least 1 pair of J levels
            error("Raman rigid rotation model cannot sum over levels")
        end
        J = J[1:mJ]
        EJ = EJ[1:mJ]
    end
    # get nuclear degeneracy
    DJ = map(x -> isodd(x) ? qJodd : qJeven, J)
    # get population of each level
    ρ = @. DJ*(2*J + 1)*exp(-EJ/(k_B*temp))
    ρ ./= sum(ρ)
    # we need pairs of J, J+2 levels, so limit J to just the starting levels
    J = J[1:end-2]
    # angular frequency of each J, J+2 pair
    Ω = (EJ[3:end] .- EJ[1:end-2])./ħ
    # absolute Raman prefactor
    K = -(4π*ε_0)^2*2/15*Δα^2/ħ
    RamanRespRotationalNonRigid(K, J, ρ, Ω, τ2)
end

"Get the response function at time `t`."
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

"""
    molecular_raman_response(rp; kwargs...)

Get the Raman response function for the Raman parameters in named tuple `rp`.

# Keyword Arguments
- `rotation::Bool = true`: whether to include the rotational contribution
- `vibration::Bool = true`: whether to include the vibrational contribution
- `minJ::Integer = 0`: the minimum rotational quantum number to include
- `maxJ::Integer = 50`: the maximum rotational quantum number to include
- `temp::Real = roomtemp`: the temperature
"""
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

"""
    raman_response(material; kwargs...)

Get the Raman response function for `material`.

For details on the keyword arguments see [`molecular_raman_response`](@ref).
"""
function raman_response(material; kwargs...)
    rp = raman_parameters(material)
    if rp.kind == :molecular
        return molecular_raman_response(rp, ; kwargs...)
    elseif rp.kind == :normedsdo
        return RamanRespNormedSingleDampedOscillator(rp.K, rp.Ω, rp.τ2)
    else
        throw(DomainError(rp.kind, "Unknown Raman model $(rp.kind)"))
    end
end

end
