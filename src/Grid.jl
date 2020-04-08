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
    sidx::BitArray{1}
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

    sidx = ω .> 0 # Indices to select real frequencies (for dispersion relation)

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)

    Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e",
                           length(t), length(to), maximum(ω), maximum(ωo))
    return RealGrid(zmax, referenceλ, t, ω, to, ωo, sidx, ωwindow, twindow, towindow)
end

struct EnvGrid{T} <: AbstractGrid
    zmax::Float64
    referenceλ::Float64
    ω0::Float64
    t::Array{Float64, 1}
    ω::Array{Float64, 1}
    to::Array{Float64, 1}
    ωo::Array{Float64, 1}
    sidx::T
    ωwin::Array{Float64, 1}
    twin::Array{Float64, 1}
    towin::Array{Float64, 1}
end

function EnvGrid(zmax, referenceλ, λ_lims, trange; δt=1, thg=false)
    fmin = PhysData.c/maximum(λ_lims)
    fmax = PhysData.c/minimum(λ_lims)
    fmax_win = 1.1*fmax # extended frequency window to accommodate apodisation
    ω0 = 2π*PhysData.c/referenceλ # central frequency
    Logging.@info @sprintf("Freq limits %.2f - %.2f PHz", fmin*1e-15, fmax*1e-15)
    if thg
        ffac = 6 # need to sample at 6x maximum desired frequency
        f_lims = (fmin, fmax)
    else
        ffac = 2 # need to sample at Nyquist limit for desired frequency only
        # Frequency window is not extended without THG, so need to extend additionally
        # to accommodate apodisation
        f_lims = (fmin, fmax_win)
    end
    rf_lims = f_lims .- PhysData.c/referenceλ # Relative frequency limits
    δt_f = 1/(ffac*maximum(rf_lims)) # time spacing as required by frequency window
    if δt_f <= δt
        # Demands of frequency window are more stringent
        δto = δt_f
        oversampling = thg # if no thg, then no oversampling
    else
        # User-defined time spacing is more stringent
        δto = δt
        oversampling = true
    end
    samples = 2^(ceil(Int, log2(trange/δto))) # samples for grid (power of 2)
    trange_even = δto*samples # keep frequency window fixed, expand time window as necessary
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d, δt = %.2f as",
    trange/δto, samples, δto*1e18)
    δωo = 2π/trange_even # frequency spacing for grid
    # Make fine grid 
    No = collect(range(0, length=samples))
    to = @. (No-samples/2)*δto # time grid, centre on 0
    vo = @. (No-samples/2)*δωo # freq grid relative to ω0
    vo = FFTW.fftshift(vo)
    ωo = vo .+ ω0
    
    ωmin = 2π*fmin
    ωmax = 2π*fmax
    ωmax_win = 2π*fmax_win
    
    # Find cropping area for coarse grid (contains frequencies of interest + apodisation)
    if oversampling
        cropidx = findfirst(x -> x>=(ωmax_win-δωo), ωo)
        cropidx = 2^(ceil(Int, log2(cropidx))) # make coarse grid power of 2 as well
        v = vcat(vo[1:cropidx], vo[end-cropidx+1:end])
    else
        v = vo
    end

    δt = -π/minimum(v)
    tsamples = length(v)
    Nt = collect(range(0, length=tsamples))
    t = @. (Nt - tsamples/2)*δt
    
    ω = v .+ ω0 # True frequency grid
    sidx = ω .> 0 # Indices to select real frequencies (for dispersion relation)
    
    # Make apodisation windows
    ωwindow = Maths.planck_taper(ω, 0, ωmin, ωmax, ωmax_win)
    twindow = Maths.planck_taper(t, minimum(t), -trange/2, trange/2, maximum(t))
    towindow = Maths.planck_taper(to, minimum(to), -trange/2, trange/2, maximum(to))
    
    # Check that grids are correct
    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ minimum(vo)/minimum(v) # FFT grid -> sample at -fs/2 but not +fs/2
    factor = Int(length(to)/length(t))
    zeroidx = findfirst(x -> x==0, to)
    # Starting at zero, time samples should be exactly the same (except fewer in t)
    @assert all(to[zeroidx:factor:end] .== t[t .>= 0])
    @assert all(to[zeroidx:-factor:1] .== t[t .<= 0][end:-1:1])

    return EnvGrid(zmax, referenceλ, ω0, t, ω, to, ωo, sidx, ωwindow, twindow, towindow)
end

function to_dict(g::GT) where GT <: AbstractGrid
    d = Dict{String, Any}()
    for field in fieldnames(GT)
        d[string(field)] = getfield(g, field)
    end
    d
end

end
