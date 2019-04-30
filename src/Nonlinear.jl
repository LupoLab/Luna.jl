module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths
import FFTW

abstract type Response end

function kerr(E::Array{T, N}, χ3) where T<:Real where N
    return @. ε_0*χ3 * E^3
end

function kerr!(out, E::Array{T, N}, χ3) where T<:Real where N
    @. out += ε_0*χ3 * E^3
end

function make_kerr!(χ3)
    kerr! = let χ3=χ3
        function kerr!(out, E)
            @. out += ε_0*χ3 * E^3
        end
    end
end

function kerr(E::Array{T, N}, χ3, onlySPM::Val{false}) where T<:Complex where N
    return @. ε_0*χ3/4 * (3*abs2(E) + E^2)*E
end

function kerr(E::Array{T, N}, χ3, onlySPM::Val{true}) where T<:Complex where N
    return @. ε_0*χ3/4 * 3*abs2(E)*E
end

function kerr(E::Array{T, N}, χ3) where T<:Complex where N
    return kerr(E, χ3, Val(false))
end

struct Kerr <: Response
    χ3::Float64
end

function (K::Kerr)(out, E)
    @. out += ε_0*K.χ3 * E^3
end

function make_plasma!(t, ω, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
    rate = similar(E)
    fraction = similar(E)
    phase = similar(E)
    J = similar(E)
    P = similar(E)
    plasma! = let ionrate=ionrate, ionpot=ionpot, t=t, phase=phase,
                  rate=rate, J=J, fraction=fraction, P=P
        function plasma!(out, E)
            ionrate(rate, E)
            Maths.cumtrapz!(fraction, t, rate)
            @. fraction = 1-exp(-fraction)
            @. phase = fraction * e_ratio * E
            Maths.cumtrapz!(J, t, phase)
            for ii in eachindex(E)
                if abs(E[ii]) > 0
                    J[ii] += ionpot * rate[ii] * (1-fraction[ii])/E[ii]
                end
            end
            Maths.cumtrapz!(P, t, J)
            @. out += P
        end
    end
    return plasma!
end

function make_plasma_JT!(t, ω, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
    rate = similar(E)
    cumrate = similar(E)
    phase = similar(E)
    frac = similar(E)
    J = similar(E)
    P = similar(E)
    bufout = similar(E)
    plasma! = let ionrate=ionrate, ionpot=ionpot, t=t, phase=phase,
                  rate=rate, J=J, cumrate=cumrate, P=P, frac=frac, bufout=bufout
        function plasma!(out, E)
            ionrate(rate, E)
            Maths.cumtrapz!(cumrate, t, rate)
            for ii in eachindex(E)
                frac[ii] = 1 - exp(-cumrate[ii])
                aE = abs(E[ii])
                if aE > 0
                    P[ii] = rate[ii]*(1 - frac[ii])*ionpot/E[ii]
                else
                    P[ii] = 0
                end
                J[ii] = frac[ii]*E[ii]*e_ratio
            end
            Maths.cumtrapz!(phase, t, J)
            for ii in eachindex(P)
                P[ii] += phase[ii]
            end
            Maths.cumtrapz!(bufout, t, P)
            @. out += bufout
        end
    end
    return plasma!
end

function make_plasma_FT!(t, ω, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
    buf_ir = similar(E)
    buf_frac = similar(E)
    buf_phase = similar(E)
    buf_P = similar(E)
    buf_out = similar(E)
    buf_FT = zeros(ComplexF64, 2*length(t)+1)
    FT = FFTW.plan_rfft(E)
    Eω = FT*E
    IFT = FFTW.plan_irfft(Eω, length(t))
    plasma! = let ionrate=ionrate, ionpot=ionpot, t=t, ω=copy(ω), FT=FT, IFT=IFT,
                  buf_ir=buf_ir, buf_P=buf_P, buf_frac=buf_frac, buf_out=buf_out, buf_phase=buf_phase
        ω[1] = Inf
        function plasma!(out, E)
            ionrate(buf_ir, E)
            Maths.cumtrapz!(buf_frac, t, buf_ir)
            @. buf_frac = 1-exp(-buf_frac)
            @. buf_phase = buf_frac*e_ratio*E
            buf_FT = FT*buf_phase
            buf_FT ./= im.*ω
            buf_P .= IFT*buf_FT
            @. buf_P += ionpot * buf_ir * (1-buf_frac)/E
            buf_P[abs.(E) .< 1e-10] .= 0
            buf_FT = FT*buf_P
            buf_FT ./= im.*ω
            buf_out .= IFT*buf_FT
            @. out += buf_out
        end
    end
    return plasma!
end

end