module Raman
import Luna.PhysData: c, ε_0, ħ, k_B, roomtemp, amg
import Luna.PhysData: raman_parameters
import Cubature: hquadrature
import Luna.Maths: planck_taper

abstract type AbstractRamanResponse end

# make Raman responses broadcast like a scalar
Broadcast.broadcastable(R::AbstractRamanResponse) = Ref(R)

"""
    hrpre(R::AbstractRamanResponse, t)

Get the pre (without damping) response function at time `t`.

"""
function hrpre end

"""
    hrdamp(R::AbstractRamanResponse, ρ)

Get the damping (dephasing) constant `τ2` for density `̢ρ`.

"""
function hrdamp end

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
    τ2ρ = let τ2=τ2
        ρ -> τ2
    end
    RamanRespSingleDampedOscillator(K, Ω, τ2ρ)
end

function hrpre(R::RamanRespSingleDampedOscillator, t)
    t > 0.0 ? R.K*sin(R.Ω*t) : 0.0
end

hrdamp(R::RamanRespSingleDampedOscillator, ρ) = R.τ2ρ(ρ)


"""
    RamanRespIntermediateBroadening(ωi, Ai, Γi, γi, scale)

Construct an intermediate broadened model with component positions `ωi` [rad/s], amplitudes `Ai`,
Gaussian widths `Γi` [rad/s] and Lorentzian widths `γi` [rad/s]. The overall response is scaled by `scale`.
Based on Hollenbeck and Cantrell,
"Multiple-vibrational-mode model for fiber-optic Raman gain spectrum and response function",
J. Opt. Soc. Am. B/Vol. 19, No. 12/December 2002.

"""
struct RamanRespIntermediateBroadening
    t::Vector{Float64} # time grid
    ωi::Vector{Float64} # central angular freqency
    Ai::Vector{Float64} # component amplitudes
    Γi::Vector{Float64} # Gaussian widths
    γi::Vector{Float64} # Lorentzian widths
    scale::Float64
    function RamanRespIntermediateBroadening(t::AbstractVector, ωi::AbstractVector, Ai::AbstractVector, Γi::AbstractVector, γi::AbstractVector, scale)
        n = length(ωi)
        if (length(Ai) != n) || (length(Γi) != n) || (length(γi) != n)
            error("all component vectors must have same length")
        end
        hrtemp = new(t, ωi, Ai, Γi, γi, 1.0)
        scale *= 1.0/hquadrature(x->hrtemp(x), 0.0, 1e-9)[1]
        tt = collect(0:(length(t) - 1)) .* (t[2] - t[1])
        new(tt, ωi, Ai, Γi, γi, scale)
    end
end

function (R::RamanRespIntermediateBroadening)(ht::AbstractVector, ρ)
    fill!(ht, 0.0)
    for (idx, t) in enumerate(R.t)
        for i = eachindex(R.ωi)
            ht[idx] += R.scale*R.Ai[i]*exp(-R.γi[i]*t)*exp(-R.Γi[i]^2*t^2/4)*sin(R.ωi[i]*t)
        end
    end
    return ht
end

function (R::RamanRespIntermediateBroadening)(t::Number)
    h = 0.0
    for i = eachindex(R.ωi)
        h += R.scale*R.Ai[i]*exp(-R.γi[i]*t)*exp(-R.Γi[i]^2*t^2/4)*sin(R.ωi[i]*t)
    end
    return h
end


"""
    RamanRespVibrational(Ωv, dαdQ, μ; τ2=nothing, Bρ=nothing, Aρ=nothing)

Construct a molecular vibrational Raman model (single damped oscillator).

# Arguments
- `Ωv::Real`: vibrational frequency [rad/s]
- `dαdQ::Real`: isotropic averaged polarizability derivative [m^2]
- `μ::Real`: reduced molecular mass [kg]
- `τ2::Real=nothing`: coherence time [s]
- `Bρ::Real=nothing` : density dependent broadening coefficient [Hz/amagat]
- `Aρ::Real=nothing` : self diffusion coefficient [Hz amagat]
- `C::Real=0` : constant linewidth [Hz]

Only one of `τ2` or `Bρ` should be specified.
If `Bρ` is specified then `Aρ` must be too.

# References
- Full model description:
    S-.F. Gao, Y-Y. Wang, F. Belli, C. Brahms, P. Wang and J.C. Travers,
    Laser & Photonics Reviews 16, 2100426 (2022)
- We followed closely:
    Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
    But note that that paper uses weird units, and we converted it to SI for
    the above reference. 
"""
function RamanRespVibrational(Ωv, dαdQ, μ; τ2=nothing, Bρ=nothing, Aρ=nothing, C=0.0)
    K = (4π*ε_0)^2*dαdQ^2/(4μ*Ωv)
    τ2ρ = if isnothing(Bρ)
        isnothing(τ2) && error("one of `τ2` or `Bρ` must be specified")
        let τ2=τ2
            ρ -> τ2
        end
    else
        !isnothing(τ2) && error("only one of `τ2` or `Bρ` must be specified")
        isnothing(Aρ) && error("if `Bρ` is specified you must also specify `Aρ`")
        let Bρ=Bρ, Aρ=Aρ, C=C
            ρ -> 1/(pi*(C + Aρ/(ρ/amg) + Bρ*ρ/amg))
        end
    end
    RamanRespSingleDampedOscillator(K, Ωv, τ2ρ)
end


# TODO: we assume here that all rotational levels have same linewidth
struct RamanRespRotationalNonRigid{TR, Tτ2} <: AbstractRamanResponse
    Rs::TR # List of Raman responses
    τ2ρ::Tτ2 # coherence dephasing time function density to τ2: ρ -> τ2
end

"""
    RamanRespRotationalNonRigid(B, Δα, τ2, qJodd, qJeven;
                                D=0.0, minJ=0, maxJ=50, temp=roomtemp,
                                τ2=nothing, Bρ=nothing, Aρ=nothing)

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
- `Bρ::Real=nothing` : density dependent broadening coefficient [Hz/amagat]
- `Aρ::Real=nothing` : self diffusion coefficient [Hz amagat]

Only one of `τ2` or `Bρ` should be specified.
If `Bρ` is specified then `Aρ` must be too.

# References
- Full model description: Laser & Photonics Reviews, 16, p. 2100426, (2022)
  doi: 10.1002/lpor.202100426.
- We followed closely: Phys. Rev. A, vol. 92, no. 6, p. 063828, Dec. 2015,
  But note that that paper uses weird units, and we converted it to SI for
  the above reference. 
"""
function RamanRespRotationalNonRigid(B, Δα, qJodd::Int, qJeven::Int;
    D=0.0, minJ=0, maxJ=50, temp=roomtemp, τ2=nothing, Bρ=nothing, Aρ=nothing)
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
              isnothing(Aρ) && error("if `Bρ` is specified you must also specify `Aρ`")
              let Bρ=Bρ, Aρ=Aρ
                  ρ -> ρ > 0 ? 1/(pi*(Aρ/(ρ/amg) + Bρ*ρ/amg)) : Inf
              end
          end
    Rs = [RamanRespSingleDampedOscillator((K*(J[i] + 1)*(J[i] + 2)/(2*J[i] + 3)
                                            *(ρ[i+2]/(2*J[i] + 5) - ρ[i]/(2*J[i] + 1))),
                                          Ω[i], τ2ρ) for i=1:length(J)]
    RamanRespRotationalNonRigid(Rs, τ2ρ)
end

hrpre(R::RamanRespRotationalNonRigid, t) = sum(hrpre(Ri, t) for Ri in R.Rs)

hrdamp(R::RamanRespRotationalNonRigid, ρ) = R.τ2ρ(ρ)

struct CombinedRamanResponse
    Rs::Vector{Any} # list of Raman responses
    t::Vector{Float64} # time grid
    w::Vector{Float64} # filter window
    hpres::Vector{Vector{Float64}} # pre Raman responses for each R in Rs
end

function CombinedRamanResponse(t, Rs)
    tt = collect(0:(length(t) - 1)) .* (t[2] - t[1])
    hpres = [hrpre.(R, tt) for R in Rs]
    w = planck_taper(tt, -tt[end], -tt[end]*0.7, tt[end]*0.7, tt[end])
    CombinedRamanResponse(Rs, tt, w, hpres)
end

function (R::CombinedRamanResponse)(ht::AbstractVector, ρ)
    fill!(ht, 0.0)
    for i=1:length(R.Rs)
        ht .+= R.hpres[i] .* exp.(-R.t ./ hrdamp.(R.Rs[i], ρ))
    end
    ht .*= R.w
end

(R::CombinedRamanResponse)(t::Number, ρ) = sum(Ri(t, ρ) for Ri in R.Rs)


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
    if rotation && (rp.rotation != :none)
        if rp.rotation != :nonrigid
            throw(DomainError(rp.rotation, "Unknown Rotational Raman model $(rp.rotation)"))
        end
        if haskey(rp, :Bρr)
            hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.qJodd, rp.qJeven, Bρ=rp.Bρr, Aρ=rp.Aρr,
                                            D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp)
        else
            hr = RamanRespRotationalNonRigid(rp.B, rp.Δα, rp.qJodd, rp.qJeven, τ2=rp.τ2r,
                                            D=rp.D, minJ=minJ, maxJ=maxJ, temp=temp)
        end
        push!(Rs, hr)
    end
    if vibration  && (rp.vibration != :none)
        if rp.vibration != :sdo
            throw(DomainError(rp.rotation, "Unknown Vibrational Raman model $(rp.vibration)"))
        end
        if haskey(rp, :Bρv)
            Cv = haskey(rp, :Cv) ? rp.Cv : 0.0
            hv = RamanRespVibrational(rp.Ωv, rp.dαdQ, rp.μ, Bρ=rp.Bρv, Aρ=rp.Aρv, C=Cv)
        else
            hv = RamanRespVibrational(rp.Ωv, rp.dαdQ, rp.μ, τ2=rp.τ2v)
        end
        push!(Rs, hv)
    end 
    CombinedRamanResponse(t, Rs) 
end

"""
    raman_response(t, material; kwargs...)

Get the Raman response function for time grid `t` and the `material`.

For details on the keyword arguments see [`molecular_raman_response`](@ref).
"""
function raman_response(t, material, scale=1; kwargs...)
    rp = raman_parameters(material)
    if rp.kind == :molecular
        return molecular_raman_response(t, rp; kwargs...)
    elseif rp.kind == :normedsdo
        return CombinedRamanResponse(t, [RamanRespNormedSingleDampedOscillator(rp.K, rp.Ω, rp.τ2)])
    elseif rp.kind == :intermediate
        return RamanRespIntermediateBroadening(t, rp.ωi, rp.Ai, rp.Γi, rp.γi, scale)
    else
        throw(DomainError(rp.kind, "Unknown Raman model $(rp.kind)"))
    end
end

end
