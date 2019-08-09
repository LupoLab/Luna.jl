module Grid
import Logging
import FFTW
import Printf: @sprintf
import Luna: PhysData, Maths

abstract type AbstractGrid end

struct RealGrid <: AbstractGrid
    zmax::Float64
    referenceλ::Float64
    t::Array{Float64, 1}
    ω::Array{Float64, 1}
    to::Array{Float64, 1}
    ωo::Array{Float64, 1}
    ωwin::Array{Float64, 1}
    twin::Array{Float64, 1}
    towin::Array{Float64, 1}
end

function RealGrid(zmax, referenceλ, λ_lims, trange, δt=1)
    f_lims = PhysData.c./λ_lims
    Logging.@info @sprintf("Freq limits %.2f - %.2f PHz", f_lims[2]*1e-15, f_lims[1]*1e-15)
    δto = min(1/(6*maximum(f_lims)), δt) # 6x maximum freq, or user-defined if finer
    samples = 2^(ceil(Int, log2(trange/δto))) # samples for fine grid (power of 2)
    trange_even = δto*samples # keep frequency window fixed, expand time window as necessary
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d, δt = %.2f as",
                            trange/δto, samples, δto*1e18)
    δωo = 2π/trange_even # frequency spacing for fine grid
    # Make fine grid 
    Nto = collect(range(0, length=samples))
    to = @. (Nto-samples/2)*δto # centre on 0
    Nωo = collect(range(0, length=Int(samples/2 +1)))
    ωo = Nωo*δωo

    ωmin = 2π*minimum(f_lims)
    ωmax = 2π*maximum(f_lims)
    ωmax_win = 1.1*ωmax

    cropidx = findfirst(x -> x>ωmax_win, ωo)
    cropidx = 2^(ceil(Int, log2(cropidx))) + 1 # make coarse grid power of 2 as well
    ω = ωo[1:cropidx]
    δt = π/maximum(ω)
    tsamples = (cropidx-1)*2
    Nt = collect(range(0, length=tsamples))
    t = @. (Nt-tsamples/2)*δt

    # Make apodisation windows
    ωwindow = Maths.planck_taper(ω, 0, ωmin, ωmax, ωmax_win) 

    twindow = Maths.planck_taper(t, minimum(t), -trange/2, trange/2, maximum(t))
    towindow = Maths.planck_taper(to, minimum(to), -trange/2, trange/2, maximum(to))

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)

    Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e",
                           length(t), length(to), maximum(ω), maximum(ωo))
    return RealGrid(zmax, referenceλ, t, ω, to, ωo, ωwindow, twindow, towindow)
end

struct EnvGrid{T} <: AbstractGrid
    zmax::Float64
    referenceλ::Float64
    ω0::Float64
    t::Array{Float64, 1}
    ω::Array{Float64, 1}
    sidx::T
    ωwin::Array{Float64, 1}
    twin::Array{Float64, 1}
end

function EnvGrid(zmax, referenceλ, λ_lims, trange; δt=1, thg=false)
    f_lims = PhysData.c./λ_lims
    Logging.@info @sprintf("Freq limits %.2f - %.2f PHz", f_lims[2]*1e-15, f_lims[1]*1e-15)
    rf_lims = f_lims .- PhysData.c/referenceλ # Relative frequency limits
    ffac = thg ? 6 : 2
    δt = min(1/(ffac*maximum(rf_lims)), δt) # 2x maximum freq, or user-defined if finer
    samples = 2^(ceil(Int, log2(trange/δt))) # samples for grid (power of 2)
    trange_even = δt*samples # keep frequency window fixed, expand time window as necessary
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d, δt = %.2f as",
                            trange/δt, samples, δt*1e18)
    δω = 2π/trange_even # frequency spacing for grid
    # Make grid 
    N = collect(range(0, length=samples))
    t = @. (N-samples/2)*δt # time grid, centre on 0
    v = @. (N-samples/2)*δω # freq grid relative to ω0
    v = FFTW.fftshift(v)
    ω0 = 2π*PhysData.c/referenceλ # central frequency
    ω = v .+ ω0 # true frequency grid

    sidx = ω .> 0
 
    ωmin = 2π*minimum(f_lims)
    ωmax = 2π*maximum(f_lims)
    ωmax_win = 1.1*ωmax

    # Make apodisation windows
    ωwindow = Maths.planck_taper(ω, 0, ωmin, ωmax, ωmax_win) 
    twindow = Maths.planck_taper(t, minimum(t), -trange/2, trange/2, maximum(t))

    return EnvGrid(zmax, referenceλ, ω0, t, ω, sidx, ωwindow, twindow)
end

end
