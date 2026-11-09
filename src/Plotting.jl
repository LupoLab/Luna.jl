module Plotting
import Luna: Grid, Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import FFTW
import Printf: @sprintf
import Base: display

"""
    getext()

Return the currently loaded plotting backend extension. Tries PythonPlotExt, PyPlotExt,
and MakieExt in order.
"""
function getext()
    pkg = parentmodule(@__MODULE__)
    ext = Base.get_extension(pkg, :PythonPlotExt)
    !isnothing(ext) && return ext
    ext = Base.get_extension(pkg, :PyPlotExt)
    !isnothing(ext) && return ext
    ext = Base.get_extension(pkg, :MakieExt)
    !isnothing(ext) && return ext
    error("No plotting backend loaded. Please load one of: PythonPlot, PyPlot, GLMakie, CairoMakie, or WGLMakie.")
end

"""
    cmap_white(cmap; N=2^12, n=8)

Replace the lowest colour stop of `cmap` (after splitting into `n` stops) with white and
create a new colourmap with `N` stops.
"""
function cmap_white(cmap; N=2^12, n=8)
    getext().cmap_white(cmap; N, n)
end

"""
    cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)

Make an array of `num` different colours that follow the colourmap `cmap` between the
values `cmin` and `cmax`.
"""
cmap_colours(args...; kwargs...) = getext().cmap_colours(args...; kwargs...)

"""
    subplotgrid(N, portrait=true; title=nothing)

Create a figure with `N` subplots laid out in a grid that is as close to square as
possible. If `portrait` is `true`, try to lay out the grid in portrait orientation
(taller than wide), otherwise landscape (wider than tall).

For the matplotlib backends this returns `(fig, axs)`. For the Makie backend it instead
returns `(idcs, width, height)`, where `idcs` are grid indices to place `Makie.Axis`
objects at and `width`/`height` is the recommended figure size.
"""
subplotgrid(args...; kwargs...) = getext().subplotgrid(args...; kwargs...)

"""
    cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)

Place `text` in the axes `ax` in the corner defined by `corner`. Padding can be
defined for `x` and `y` together via `pad` or separately via `xpad` and `ypad`.

Possible values for `corner` are `ul`, `ur`, `ll`, `lr` where the first letter
defines upper/lower and the second defines left/right.
"""
cornertext(args...; kwargs...) = getext().cornertext(args...; kwargs...)

"""
    auto_fwhm_arrows(ax, x, y; color, arrowlength=nothing, hpad=0, linewidth=1,
                     text=nothing, units="fs", kwargs...)

Draw FWHM arrows on an axis, indicating the full-width at half-maximum of the data `y(x)`.
"""
auto_fwhm_arrows(args...; kwargs...) = getext().auto_fwhm_arrows(args...; kwargs...)

"""
    add_fwhm_legends(ax, unit)

Enhance the legend of `ax` by appending the FWHM of each plotted line to its label.
Only available for the matplotlib backends.
"""
add_fwhm_legends(args...; kwargs...) = getext().add_fwhm_legends(args...; kwargs...)

"""
    displayall()

`display` all currently open figures. Only available for the matplotlib backends.
"""
displayall(args...; kwargs...) = getext().displayall(args...; kwargs...)

"""
    get_modes(output)

Determine whether `output` contains a multimode simulation, and if so, return the names
of the modes. Returns `(multimode::Bool, labels)` where `labels` is a vector of mode name
strings or `nothing` for single-mode simulations.
"""
function get_modes(output)
    t = output["simulation_type"]["transform"]
    !startswith(t, "TransModal") && return false, nothing
    lines = split(t, "\n")
    modeline = findfirst(li -> startswith(li, "  modes:"), lines)
    endline = findnext(li -> !startswith(li, " "^4), lines, modeline+1)
    mlines = lines[modeline+1 : endline-1]
    labels = [match(r"{([^,]*),", li).captures[1] for li in mlines]
    angles = zeros(length(mlines))
    for (ii, li) in enumerate(mlines)
        m = match(r"ϕ=(-?[0-9]+.[0-9]+)π", li)
        isnothing(m) && continue # no angle information in mode label
        angles[ii] = parse(Float64, m.captures[1])
    end
    if !all(angles .== 0)
        for i in eachindex(labels)
            if startswith(labels[i], "HE")
                if angles[i] == 0
                    θs = "x"
                elseif angles[i] == 0.5
                    θs = "y"
                else
                    θs = "$(angles[i])π"
                end
                labels[i] *= " ($θs)"
            end
        end
    end
    return true, labels
end

"""
    stats(output; kwargs...)

Plot all statistics available in `output`. Additional keyword arguments are passed to the
plotting backend.
"""
function stats(output; kwargs...)
    stats = output["stats"]

    pstats = [] # pulse statistics
    haskey(stats, "energy") && push!(pstats, (1e6*stats["energy"], "Energy (μJ)"))
    for (k, v) in pairs(stats)
        startswith(k, "energy_") || continue
        str = "Energy "*replace(k[8:end], "_" => " ")*" (μJ)"
        push!(pstats, (1e6*stats[k], str))
    end
    for (k, v) in pairs(stats)
        startswith(k, "peakpower_") || continue
        Pfac, unit = power_unit(stats[k])
        str = "Peak power "*replace(k[11:end], "_" => " ")*" ($unit)"
        push!(pstats, (Pfac*stats[k], str))
    end
    if haskey(stats, "peakpower")
        Pfac, unit = power_unit(stats["peakpower"])
        push!(pstats, (Pfac*stats["peakpower"], "Peak power ($unit)"))
    end
    haskey(stats, "peakintensity") && push!(
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm\$^2\$)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "min FWHM (fs)"))
    haskey(stats, "fwhm_t_max") && push!(pstats, (1e15*stats["fwhm_t_max"], "max FWHM (fs)"))
    haskey(stats, "fwhm_r") && push!(pstats, (1e6*stats["fwhm_r"], "Radial FWHM (μm)"))
    haskey(stats, "ω0") && push!(pstats, (1e9*wlfreq.(stats["ω0"]), "Central wavelength (nm)"))

    fstats = [] # fibre/waveguide/propagation statistics
    if haskey(stats, "electrondensity")
        push!(fstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-3}\$)"))
        if haskey(stats, "density")
            push!(fstats,
                 (100*stats["electrondensity"]./stats["density"], "Ionisation fraction (%)"))
        end
    end
    haskey(stats, "density") && push!(
        fstats, (1e-6*stats["density"], "Density (cm\$^{-3}\$)"))
    haskey(stats, "pressure") && push!(
        fstats, (stats["pressure"], "Pressure (bar)"))
    haskey(stats, "dz") && push!(fstats, (1e6*stats["dz"], "Stepsize (μm)"))
    haskey(stats, "core_radius") && push!(fstats, (1e6*stats["core_radius"], "Core radius (μm)"))
    haskey(stats, "zdw") && push!(fstats, (1e9*stats["zdw"], "ZDW (nm)"))
    haskey(stats, "mode_reconstruction_error") && push!(
        fstats, (stats["mode_reconstruction_error"], "Mode error"))
    haskey(stats, "transverse_points") && push!(
        fstats, (stats["transverse_points"], "Transverse grid points"))
    haskey(stats, "transverse_integral_error_rel") && push!(
        fstats, (stats["transverse_integral_error_rel"], "Transverse integral error (relative)"))

    z = stats["z"]*1e2

    multimode, modes = get_modes(output)
    modes = isnothing(modes) ? [""] : modes

    getext().stats(z, pstats, fstats, multimode, modes; kwargs...)
end

"""
    should_log10(A, tolfac=10)

For multi-line plots, determine whether data for different lines contained in `A` spans
a sufficiently large range that a logarithmic scale should be used. By default, this is the
case when there is any point where the lines are different by more than a factor of 10.
"""
function should_log10(A, tolfac=10)
    mi = minimum(A; dims=2)
    ma = maximum(A; dims=2)
    any(ma./mi .> tolfac)
end

window_str(::Nothing) = ""
window_str(win::NTuple{4, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win[2:3]...)
window_str(win::NTuple{2, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win...)
window_str(window) = "custom bandpass"

modeidcs(m::Int, ml) = [m]
modeidcs(m::Symbol, ml) = (m == :sum) ? [] : error("modes must be :sum, a single integer, or iterable")
modeidcs(m::Nothing, ml) = 1:length(ml)
modeidcs(m, ml) = m

"""
    getspeclims(λrange, specaxis)

Convert a wavelength range `λrange` (in metres) to the correct axis limits, label, and
scale factor for the given `specaxis` (`:f`, `:ω`, or `:λ`).
"""
function getspeclims(λrange, specaxis)
    if specaxis == :f
        specxfac = 1e-15
        speclims = (specxfac*c/maximum(λrange), specxfac*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        specxfac = 1e-15
        speclims = (specxfac*wlfreq(maximum(λrange)), specxfac*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        specxfac = 1e9
        speclims = λrange .* specxfac
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end
    return speclims, speclabel, specxfac
end

"""
    power_unit(Pt, y=:Pt)

Automatically determine the appropriate power unit (kW to PW) based on the magnitude of `Pt`.
Returns `(scale_factor, unit_string)`.
"""
function power_unit(Pt, y=:Pt)
    units = ["kW", "MW", "GW", "TW", "PW"]
    Pmax = maximum(Pt)
    oom = clamp(floor(Int, log10(Pmax)/3), 1, 5) # maximum unit is PW
    powerfac = 1/10^(oom*3)
    if y == :Et
        sqrt(powerfac), "$(units[oom])\$^{1/2}\$"
    else
        return powerfac, units[oom]
    end
end

"""
    prop_2D(output, specaxis=:f; kwargs...)

Make false-colour propagation plots for `output`, using spectral x-axis `specaxis`
(`:f` for frequency, `:ω` for angular frequency, `:λ` for wavelength).
For multimode simulations, create one figure for each mode plus one for the sum of all modes.

# Keyword arguments
- `λrange` : x-axis limits for spectral plot as `(λ_min, λ_max)` in metres
- `trange` : x-axis limits for time-domain plot as `(t_min, t_max)` in seconds
- `dBmin::Float64` : lower colour-scale limit for logarithmic spectral plot (default: -60)
- `resolution` : smooth the spectral energy density (see [`getIω`](@ref))
- `modes` : mode selection — `nothing` (all), `:sum`, integer, or range
- `oversampling::Int` : time-domain oversampling factor (default: 4)
- `bandpass` : bandpass filter wavelength range
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
    time_1D(output, zslice; y=:Pt, kwargs...)

Create lineplots of time-domain slice(s) of the propagation at position(s) `zslice`.

# Keyword arguments
- `y` : quantity to plot — `:Pt` (power, default), `:Et` (electric field), or `:Esq` (field squared)
- `modes` : mode selection — `nothing` (all), `:sum`, integer, or range
- `oversampling::Int` : oversampling factor (default: 4)
- `trange` : time-axis limits as `(t_min, t_max)` in seconds
- `bandpass` : bandpass filter wavelength range
- `FTL::Bool` : plot Fourier-transform-limited pulse (default: false)
- `propagate` : propagation distance for additional dispersion
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
    spec_1D(output, zslice, specaxis=:λ; log10=true, kwargs...)

Create lineplots of spectral-domain slices of the propagation at position(s) `zslice`.

# Keyword arguments
- `specaxis` : spectral x-axis — `:λ` (wavelength, default), `:f` (frequency), or `:ω` (angular frequency)
- `modes` : mode selection — `nothing` (all), `:sum`, integer, or range
- `λrange` : x-axis limits as `(λ_min, λ_max)` in metres
- `log10::Bool` : use logarithmic y-axis (default: true)
- `log10min::Float64` : y-axis range for log scale, as fraction of maximum (default: 1e-6)
- `resolution` : smooth the spectral energy density (see [`getIω`](@ref))
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

"""
    spectrogram(t, Et, specaxis=:λ; trange, N, fw, kwargs...)

Create a time-frequency spectrogram of the electric field `Et` on time grid `t`.

# Keyword arguments
- `specaxis` : spectral y-axis — `:λ` (wavelength, default), `:f` (frequency), or `:ω` (angular frequency)
- `trange` : time-axis limits as `(t_min, t_max)` in seconds
- `N::Int` : number of time points in the spectrogram
- `fw` : gate function width for the Gabor transform
- `λrange` : spectral-axis limits as `(λ_min, λ_max)` in metres
- `log::Bool` : use logarithmic colour scale (default: false)
- `dBmin::Float64` : lower colour-scale limit in dB when `log=true` (default: -40)
"""
function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
    trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
    kwargs...)
    getext().spectrogram(t, Et, specaxis;
                         trange, N, fw, λrange, log, dBmin, kwargs...)
end

"""
    energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))

Plot the energy evolution along the propagation, with a secondary axis showing
conversion efficiency.

# Keyword arguments
- `modes` : mode selection — `nothing` (all), `:sum`, integer, or range
- `bandpass` : bandpass filter wavelength range for energy calculation
- `figsize` : figure size as `(width, height)`
"""
function energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))
    getext().energy(output; modes, bandpass, figsize)
end

end
