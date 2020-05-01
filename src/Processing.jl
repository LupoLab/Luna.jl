module Processing
import FFTW
import Luna: Maths, Fields
import Luna.PhysData: wlfreq
import Luna.Grid: RealGrid, EnvGrid

"""
    arrivaltime(grid, Eω; λlims, winwidth=0, method=:moment, oversampling=1)

Extract the arrival time of the pulse in the wavelength limits `λlims`.

# Arguments
    - `λlims::Tuple{Number, Number}` : wavelength limits (λmin, λmax)
    - `winwidth` : If a `Number`, set smoothing width (in rad/s) of the window function
                   used to bandpass.
                   If `:auto`, automatically set the width to 128 frequency samples.
    - `method::Symbol` : `:moment` to use 1st moment to extract arrival time, `:peak` to use
                        the time of peak power
    - `oversampling::Int` : If >1, oversample the time-domain field before extracting delay
"""
function arrivaltime(grid, Eω; λlims, winwidth=:auto, kwargs...)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    arrivaltime(grid, Eω, window; kwargs...)
end

function arrivaltime(grid::RealGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.irfft(Eω .* ωwindow, length(grid.t), 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Maths.hilbert(Eto)); method=method)
end

function arrivaltime(grid::EnvGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.ifft(Eω .* ωwindow, 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Eto); method=method)
end

function arrivaltime(t, It::Vector{<:Real}; method)
    if method == :moment
        Maths.moment(t, It)
    elseif method == :peak
        t[argmax(It)]
    else
        error("Unknown arrival time method $method")
    end
end

function arrivaltime(t, It::Array{<:Real, N}; method) where N
    if method == :moment
        dropdims(Maths.moment(t, It; dim=1); dims=1)
    elseif method == :peak
        out = Array{Float64, ndims(It)-1}(undef, size(It)[2:end])
        cidcs = CartesianIndices(size(It)[2:end])
        for ii in cidcs
            out[ii] = arrivaltime(t, It[:, ii]; method=:peak)
        end
        out
    else
        error("Unknown arrival time method $method")
    end
end

"""
    energy_λ(grid, Eω; λlims, winwidth=:auto)

Extract energy within a wavelength band given by `λlims`. `winwidth` can be a `Number` in 
rad/fs to set the smoothing edges of the frequency window, or `:auto`, in which case the
width defaults to 128 frequency samples.
"""
function energy_λ(grid, Eω; λlims, winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    energy_window(grid, Eω, window)
end

function energy_window(grid, Eω, ωwindow)
    _, energyω = Fields.energyfuncs(grid)
    _energy_window(Eω, ωwindow, energyω)
end

_energy_window(Eω::Vector, ωwindow, energyω) = energyω(Eω .* ωwindow)
function _energy_window(Eω, ωwindow, energyω)
    out = Array{Float64, ndims(Eω)-1}(undef, size(Eω)[2:end])
    cidcs = CartesianIndices(size(Eω)[2:end])
    for ii in cidcs
        out[ii] = _energy_window(Eω[:, ii], ωwindow, energyω)
    end
    out
end

end