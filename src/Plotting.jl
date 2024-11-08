module Plotting
import Luna: Grid, Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import FFTW
import Printf: @sprintf
import Base: display

function getext()
    ext = Base.get_extension(@__MODULE__, :PythonPlotExt)
    isnothing(ext) && error("Please load a plotting backend.")
    ext
end

"""
    stats(output; kwargs...)

Plot all statistics available in `output`. Additional `kwargs` are passed onto `pyplot.plot()`
"""
function stats(output; kwargs...)
    getext().stats(output; kwargs...)
end

"""
    prop_2D(output, specaxis=:f)

Make false-colour propagation plots for `output`, using spectral x-axis `specaxis` (see
[`getIω`](@ref)). For multimode simulations, create one figure for each mode plus one for
the sum of all modes.

# Keyword arguments
- `λrange::Tuple(Float64, Float64)` : x-axis limits for spectral plot (wavelength in metres)
- `trange::Tuple(Float64, Float64)` : x-axis limits for time-domain plot (time in seconds)
- `dBmin::Float64` : lower colour-scale limit for logarithmic spectral plot
- `resolution::Real` smooth the spectral energy density as defined by [`getIω`](@ref).
"""
function prop_2D(output, specaxis=:f;
                 trange=(-50e-15, 50e-15), bandpass=nothing,
                 λrange=(150e-9, 2000e-9), dBmin=-60,
                 resolution=nothing, modes=nothing, oversampling=4,
                 kwargs...)
    getext().prop_2D(output, specaxis; trange, bandpass,
                     λrange, dBmin, resolution, modes, oversampling, kwargs...)
end

"""
    time_1D(output, zslice, y=:Pt, kwargs...)

Create lineplots of time-domain slice(s) of the propagation.

The keyword argument `y` determines
what is plotted: `:Pt` (power, default), `:Esq` (squared electric field) or `:Et` (electric field).

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

The keyword argument `oversampling` determines the amount of oversampling done before plotting.

Other `kwargs` are passed onto `pyplot.plot`.
"""
function time_1D(output, zslice=maximum(output["z"]);
                y=:Pt, modes=nothing,
                oversampling=4, trange=(-50e-15, 50e-15), bandpass=nothing,
                FTL=false, propagate=nothing,
                kwargs...)
    getext().time_1D(output, zslice; y, modes, oversampling, trange, bandpass,
                     FTL, propagate, kwargs...)
end

"""
    spec_1D(output, zslice, specaxis=:λ, log10=true, log10min=1e-6)

Create lineplots of spectral-domain slices of the propagation.

The x-axis is determined by `specaxis` (see [`getIω`](@ref)).

If `log10` is true, plot on a logarithmic scale, with a y-axis range of `log10min`. 

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

Other `kwargs` are passed onto `pyplot.plot`.
"""
function spec_1D(output, zslice=maximum(output["z"]), specaxis=:λ;
                 modes=nothing, λrange=(150e-9, 1200e-9),
                 log10=true, log10min=1e-6, resolution=nothing,
                 kwargs...)
    getext().spec_1D(output, zslice, specaxis;
                     modes, λrange, log10, log10min, resolution, kwargs...)
end


spectrogram(output::AbstractOutput, args...; kwargs...) = spectrogram(
    makegrid(output), output, args...; kwargs...)

function spectrogram(grid::Grid.AbstractGrid, Eω::AbstractArray, specaxis=:λ;
                     propagate=nothing, kwargs...)
    t, Et = getEt(grid, Eω; propagate=propagate, oversampling=1)
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(grid::Grid.AbstractGrid, output, zslice, specaxis=:λ;
                     propagate=nothing, kwargs...)
    t, Et, zactual = getEt(output, zslice; oversampling=1, propagate=propagate)
    Et = Et[:, 1]
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
    trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
    kwargs...)
    getext().spectrogram(t, Et, specaxis;
                         trange, N, fw, λrange, log, dBmin, kwargs...)
end

function energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))
    getext().energy(output; modes, bandpass, figsize)
end

end