module Raman
import Luna.PhysData: c, ε_0, ħ, k_B, roomtemp, amg
import Luna.PhysData: raman_parameters

abstract type AbstractRamanResponse end

# make Raman responses broadcast like a scalar
Broadcast.broadcastable(R::AbstractRamanResponse) = Ref(R)

"""
    hrpre(R::AbstractRamanResponse, t)

Get the pre (without damping) response function at time `t`.

"""
function hrpre(R::AbstractRamanResponse, t)
    error("abstract method called")
end

"""
    hrdamp(R::AbstractRamanResponse, ρ)

Get the damping constant `τ2` for density `̢ρ`.

"""
function hrdamp(R::AbstractRamanResponse, ρ)
    error("abstract method called")
end

"""
    (R::AbstractRamanResponse)(t, ρ)

Get the full response function at time `t` and density `̢ρ`.

"""
function (R::AbstractRamanResponse)(t, ρ)
    hrpre(R, t) * exp(-t/hrdamp(R, ρ))
end


struct RamanRespSingleDampedOscillator{Tτ2} <: AbstractRamanResponse
    K::Float64 # overall scale factor
    Ω::Float64 # frequency
    τ2ρ::Tτ2 # coherence dephasing time function density to τ2: ρ -> τ2
end

"""
    RamanRespNormedSingleDampedOscillator(K, Ω, τ2)

Construct a simple normalised single damped oscillator model with scale factor `K`,
angular frequency `Ω` and density independent coherence time `τ2`.

The scale factor `K` is applied after normalising the integral of the response function
to unity.
"""
function RamanRespNormedSingleDampedOscillator(K, Ω, τ2)
    K *= (Ω^2 + 1/τ2^2)/Ω # normalise SDO model to unity integral, scaled by prefactor K.
    τ2̢ρ = let τ2=τ2
        ρ -> τ2
    end
    RamanRespSingleDampedOscillator(K, Ω, τ2̢ρ)
end

function hrpre(R::RamanRespSingleDampedOscillator, t)
    t > 0.0 ? R.K*sin(R.Ω*t) : 0.0
end

hrdamp(R::RamanRespSingleDampedOscillator, ρ) = R.τ2ρ(ρ)


"""
    RamanRespVibrational(Ωv, dαdQ, μ, τ2)

Construct a molecular vibrational Raman model (single damped oscillator).

# Arguments
- `Ωv::Real`: vibrational frequency [rad/s]
- `dαdQ::Real`: isotropic averaged polarizability derivative [m^2]
- `μ::Real`: reduced molecular mass [kg]
- `τ2::Real`: coherence time [s]

# References
- Full model description: To be published, Yingying paper.
- We followed closely: Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
  But note that that paper uses weird units, and we converted it to SI for
  the above reference. 
"""
function RamanRespVibrational(Ωv, dαdQ, μ, τ2)
    # TODO we assume pressure independent linewidth which is incorrect
    K = (4π*ε_0)^2*dαdQ^2/(4μ*Ωv)
    τ2̢ρ = let τ2=τ2
        ρ -> τ2
    end
    RamanRespSingleDampedOscillator(K, Ωv, τ2̢ρ)
end


# TODO: we assume here that all rotational levels have same linewidth
struct RamanRespRotationalNonRigid{TR, Tτ2} <: AbstractRamanResponse
    Rs::TR # List of Raman responses
    τ2::Tτ2 # coherence dephasing time function density to τ2: ρ -> τ2
end

"""
    RamanRespRotationalNonRigid(B, Δα, τ2, qJodd, qJeven;
                                D=0.0, minJ=0, maxJ=50, temp=roomtemp,
                                τ2=nothing, Bρ=nothing)

Construct a rotational nonrigid rotor Raman model.

# Arguments
- `B::Real`: the rotational constant [1/m]
- `Δα::Real`: molecular polarizability anisotropy [m^3]
- `qJodd::Integer`: nuclear spin parameter for odd `J`
- `qJeven::Integer`: nuclear spin parameter for even `J`
- `D::Real=0.0`: centrifugal constant [1/m]
- `minJ::Integer=0`: J value to start at
- `maxJ::Integer=50`: J value to sum until
- `temp::Real=roomtemp`: temperature
- `τ2::Real=nothing`: coherence time [s]
- `Bρ::Real=nothing` : density dependent broadening coefficient [Hz/amagat])]

Only one of `τ2` or `Bρ` should be specified.

# References
- Full model description: To be published, Yingying paper.
- We followed closely: Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
  But note that that paper uses weird units, and we converted it to SI for
  the above reference. 
"""
function RamanRespRotationalNonRigid(B, Δα, qJodd::Int, qJeven::Int;
    D=0.0, minJ=0, maxJ=50, temp=roomtemp, τ2=nothing, Bρ=nothing)
    J = minJ:maxJ # range of J values to start with
    EJ = @. 2π*ħ*c*(B*J*(J + 1) - D*(J*(J + 1))^2) # energy of each J level
    # limit J range to those which have monotonic increasing energy
    mJ = findfirst(x -> x < 0.0, diff(EJ))
    if !isnothing(mJ)
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
    τ2ρ = if isnothing(Bρ)
              isnothing(τ2) && error("one of `τ2` or `Bρ` must be specified")
              let τ2=τ2
                  ρ -> τ2
              end
          else
              !isnothing(τ2) && error("only one of `τ2` or `Bρ` must be specified")
              let Bρ=B̢ρ
                  ρ -> pi/(B̢ρ*ρ/amg)
              end
          end
    Rs = [RamanRespSingleDampedOscillator((K*(J[i] + 1)*(J[i] + 2)/(2*J[i] + 3)
                                            *(ρ[i+2]/(2*J[i] + 5) - ρ[i]/(2*J[i] + 1))),
                                          Ω[i], τ2̢ρ) for i=1:length(J)]
    RamanRespRotationalNonRigid(Rs, τ2̢ρ)
end

hrpre(R::RamanRespRotationalNonRigid, t) = sum(hrpre.(R.Rs, t))

hrdamp(R::RamanRespRotationalNonRigid, ρ) = R.τ2ρ(ρ)

struct CombinedRamanResponse{TR,Tt}
    Rs::TR # list of Raman responses
    t::Vector{Float64} # time grid
    hpres::Array{Float64,2} # pre Raman responses for each R in Rs
end

function CombinedRamanResponse(t, Rs)
    hpres = [hpre.(R, t) for R in Rs]
    RamanResponse(Rs, t, hpres)
end

function (R::CombinedRamanResponse)(ht, ρ)
    fill!(ht, 0.0)
    for i=1:length(R.Rs)
        ht .+= R.hpres[i] .* exp.(-R.t ./ hrdamp.(R.Rs[i], ρ))
    end
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
function molecular_raman_response(t, rp; rotation=true, vibration=true, minJ=0, maxJ=50, temp=roomtemp)
    Rs = []
    if rotation
        if rp.rotation != :nonrigid
            throw(DomainError(rp.rotation, "Unknown Rotational Raman model $(rp.rotation)"))
        end
        if haskey(rp, :Bρr)
            hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.qJodd, rp.qJeven, Bρ=rp.Bρr,
                                            D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp)
        else
            hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.qJodd, rp.qJeven, τ2=rp.τ2r,
                                            D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp)
        end
        push!(Rs, hr)
    end
    if vibration
        if rp.vibration != :sdo
            throw(DomainError(rp.rotation, "Unknown Vibrational Raman model $(rp.vibration)"))
        end
        hv = RamanRespVibrational(rp.Ωv, rp.dαdQ, rp.μ, rp.τ2v)
        push!(Rs, hv)
    end 
    CombinedRamanResponse(t, Rs) 
end

"""
    raman_response(material; kwargs...)

Get the Raman response function for `material`.

For details on the keyword arguments see [`molecular_raman_response`](@ref).
"""
function raman_response(material, t; kwargs...)
    rp = raman_parameters(material)
    if rp.kind == :molecular
        return molecular_raman_response(t, rp; kwargs...)
    elseif rp.kind == :normedsdo
        return CombinedRamanResponse(t, [RamanRespNormedSingleDampedOscillator(rp.K, rp.Ω, rp.τ2)])
    else
        throw(DomainError(rp.kind, "Unknown Raman model $(rp.kind)"))
    end
end

end
