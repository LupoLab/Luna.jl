module Nonlinear
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths
import FFTW

function kerr(E::Array{T, N}, χ3) where T<:Real where N
    return @. ε_0*χ3 * E^3
end

function kerr!(out, E::Array{T, N}, χ3) where T<:Real where N
    @. out = ε_0*χ3 * E^3
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

function plasma_phase(t, E::Array{T, N}, ionfrac) where T<:Real where N
    return e_ratio*Maths.cumtrapz(t, Maths.cumtrapz(t, ionfrac.*E))
end

function plasma_loss(t, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
    return Maths.cumtrapz(t, ionpot.*(1 .- ionrate)./E)
end

function plasma(t, E::Array{T, N}, ionrate, ionfrac, ionpot) where T<:Real where N
    out = Maths.cumtrapz(t, ionpot.*(1 .- ionrate)./E .+ Maths.cumtrapz(t, ionfrac.*E))
    return out[(abs.(E) .> 1e2)] .= 0
end

function make_plasma!(t, ω, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
    buf_ir = similar(E)
    buf_frac = similar(E)
    buf_P = similar(E)
    buf_out = similar(E)
    plasma! = let ionrate=ionrate, ionpot=ionpot, t=t,
                  buf_ir=buf_ir, buf_P=buf_P, buf_frac=buf_frac, buf_out=buf_out
        function plasma!(out, E)
            ionrate(buf_ir, E)
            Maths.cumtrapz!(buf_frac, t, buf_ir)
            @. buf_frac = 1-exp(-buf_frac)
            @. buf_frac = buf_frac * e_ratio * E
            Maths.cumtrapz!(buf_P, t, buf_frac)
            @. buf_P += ionpot * (1-buf_ir)/E
            buf_P[abs.(E) .< 1e6] .= 0
            Maths.cumtrapz!(buf_out, t, buf_P)
            @. out += buf_out
        end
    end
    return plasma!
end

# function make_plasma!(t, ω, E::Array{T, N}, ionrate, ionpot) where T<:Real where N
#     buf_ir = similar(E)
#     buf_frac = similar(E)
#     buf_P = similar(E)
#     buf_out = similar(E)
#     FT = FFTW.plan_rfft(E)
#     Eω = FT*E
#     IFT = FFTW.plan_irfft(Eω, length(t))
#     plasma! = let ionrate=ionrate, ionpot=ionpot, t=t, ω=copy(ω), FT=FT, IFT=IFT,
#                   buf_ir=buf_ir, buf_P=buf_P, buf_frac=buf_frac, buf_out=buf_out
#         ω[1] = Inf
#         function plasma!(out, E)
#             ionrate(buf_ir, E)
#             Maths.cumtrapz!(buf_frac, t, buf_ir)
#             @. buf_frac = 1-exp(-buf_frac)
#             @. buf_frac = buf_frac*e_ratio*E
#             # Maths.cumtrapz!(buf_P, t, buf_frac)
#             buf_P .= IFT*((FT*buf_frac)./(im.*ω))
#             @. buf_P += ionpot * (1-buf_ir)/E
#             buf_P[abs.(E) .< 1e6] .= 0
#             # Maths.cumtrapz!(buf_out, t, buf_P)
#             buf_out .= IFT*((FT*buf_P)./(im.*ω))
#             @. out += buf_out
#         end
#     end
#     return plasma!
# end

end