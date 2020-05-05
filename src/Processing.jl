module Processing
import FFTW
import Luna: Maths, Fields
import Luna.PhysData: wlfreq
import Luna.Grid: RealGrid, EnvGrid
import Peaks

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

"""
    pkfw(x, y, pki; level=0.5, skipnonmono=true, closest=5)

Find the full width of a peak in `y` over `x` centred at index `pki`.

The default `level=0.5` requests the full width at half maximum. Setting `level` to something
different computes the corresponding width. E.g. `level=0.1` for the 10% width. 

`skipnonmono=true` skips peaks which are not monotonically increaing/decreasing before/after the peak.

`closest=5` sets the minimum number of indices for the full width.
"""
function pkfw(x, y, pki; level=0.5, skipnonmono=true, closest=5)
    val = level*y[pki]
    iup = findnext(x -> x <= val, y, pki)
    if iup == nothing
        iup = length(x)
    end
    idn = findprev(x -> x <= val, y, pki)
    if idn == nothing
        idn = 1
    end
    if skipnonmono
        if any(diff(y[pki:iup]) .> 0)
            return missing
        end
        if any(diff(y[idn:pki]) .< 0)
            return missing
        end
    end
    if (iup - idn) < closest
        return missing
    end
    return x[iup] - x[idn]
end

"""
    findpeaks(x, y; threshold=0.0, filterfw=true)

Find isolated peaks in a signal `y` over `x` and return their value, FWHM and index.
`threshold=0.0` allows filtering peaks above a threhold value.
If `filterfw=true` then only peaks with a clean FWHM are returned.
"""
function findpeaks(x, y; threshold=0.0, filterfw=true)
    pkis, proms = Peaks.peakprom(y, Peaks.Maxima(), 10)
    pks = [(peak=y[pki], fw=pkfw(x, y, pki), index=pki) for pki in pkis]
    # filter out peaks with missing fws
    if filterfw
        pks = filter(x -> !(x.fw === missing), pks)
    end
    # filter out peaks below threshold
    pks = filter(x -> x.peak > threshold, pks)
end

"""
    field_autocorrelation(Et)

Calculate the field autocorrelation of `Et`.
"""
function field_autocorrelation(Et)
    FFTW.fftshift(FFTW.ifft(abs2.(FFTW.fft(Et))))
end

"""
    intensity_autocorrelation(Et, grid)

Calculate the intensity autocorrelation of `Et` over `grid`.
"""
function intensity_autocorrelation(Et, grid)
    I = Fields.It(Et, grid)
    real.(FFTW.ifft(abs2.(FFTW.fft(I))))
end

"""
    coherence_time(grid, Et)

Get the coherence time of a field `Et` over `grid`.
"""
function coherence_time(grid, Et)
    fac = field_autocorrelation(Et)
    Maths.fwhm(grid.t, abs2.(fac))
end

end