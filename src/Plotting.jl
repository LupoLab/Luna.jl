module Plotting
import Luna: Grid, Maths, PhysData
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import PyPlot: ColorMap, plt, pygui
import FFTW
import Printf: @sprintf

"""
    cmap_white(cmap, N=512, n=8)

Replace the lowest colour stop of `cmap` (after splitting into `n` stops) with white and
create a new colourmap with `N` stops.
"""
function cmap_white(cmap; N=2^12, n=8)
    vals = collect(range(0, 1, length=n))
    vals_i = collect(range(0, 1, length=N))
    cm = ColorMap(cmap)
    clist = cm(vals)
    clist[1, :] = [1, 1, 1, 1]
    clist_i = Array{Float64}(undef, (N, 4))
    for ii in 1:4
        clist_i[:, ii] .= Maths.BSpline(vals, clist[:, ii]).(vals_i)
    end
    ColorMap(clist_i)
end

"""
    subplotgrid(N, portrait=true, kwargs...)

Create a figure with `N` subplots laid out in a grid that is as close to square as possible.
If `portrait` is `true`, try to lay out the grid in portrait orientation (taller than wide),
otherwise landscape (wider than tall).
"""
function subplotgrid(N, portrait=true; colw=4, rowh=2.5, title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N/cols)
    portrait && ((rows, cols) = (cols, rows))
    fig, axs = plt.subplots(rows, cols, num=title)
    ndims(axs) > 1 && (axs = permutedims(axs, (2, 1)))
    if cols*rows > N
        for axi in axs[N+1:end]
            axi.remove()
        end
    end
    fig.set_size_inches(cols*colw, rows*rowh)
    fig, N > 1 ? axs : [axs]
end

"""
    get_modes(output)

Determine whether `output` contains a multimode simulation, and if so, return the names
of the modes.
"""
function get_modes(output)
    t = output["simulation_type"]["transform"]
    !startswith(t, "TransModal") && return false, nothing
    lines = split(t, "\n")
    modeline = findfirst(li -> startswith(li, "  modes:"), lines)
    endline = findnext(li -> !startswith(li, " "^4), lines, modeline+1)
    mlines = lines[modeline+1 : endline-1]
    labels = [match(r"{([^,]*),", li).captures[1] for li in mlines]
    return true, labels
end

"""
    stats(output; kwargs...)

Plot all statistics available in `output`. Additional `kwargs` are passed onto `plt.plot()`
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
    haskey(stats, "electrondensity") && push!(
        fstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-3}\$)"))
    haskey(stats, "density") && push!(
        fstats, (1e-6*stats["density"], "Density (cm\$^{-3}\$)"))
    haskey(stats, "pressure") && push!(
        fstats, (stats["pressure"], "Pressure (bar)"))
    haskey(stats, "dz") && push!(fstats, (1e6*stats["dz"], "Stepsize (μm)"))
    haskey(stats, "core_radius") && push!(fstats, (1e6*stats["core_radius"], "Core radius (μm)"))
    haskey(stats, "zdw") && push!(fstats, (1e9*stats["zdw"], "ZDW (nm)"))

    z = stats["z"]*1e2

    multimode, modes = get_modes(output)

    Npl = length(pstats)
    if Npl > 0
        pfig, axs = subplotgrid(Npl, title="Pulse stats")
        for n in 1:Npl
            ax = axs[n]
            data, label = pstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes, frameon=false)
        end
        pfig.tight_layout()
    end
    
    Npl = length(fstats)
    if Npl > 0
        ffig, axs = subplotgrid(Npl, title="Other stats")
        for n in 1:Npl
            ax = axs[n]
            data, label = fstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && should_log10(data) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes, frameon=false)
        end
        ffig.tight_layout()
    end
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
    any(ma./mi .> 10)
end

window_str(::Nothing) = ""
window_str(win::NTuple{4, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win[2:3]...)
window_str(win::NTuple{2, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win...)
window_str(window) = "custom bandpass"

"""
    prop_2D(output, specaxis=:f)

Make false-colour propagation plots for `output`, using spectral x-axis `specaxis` (see
[`getIω`](@ref)). For multimode simulations, create one figure for each mode plus one for
the sum of all modes.

# Keyword arguments
- `λrange::Tuple(Float64, Float64)` : x-axis limits for spectral plot (wavelength in metres)
- `trange::Tuple(Float64, Float64)` : x-axis limits for time-domain plot (time in seconds)
- `dBmin::Float64` : lower colour-scale limit for logarithmic spectral plot
"""
function prop_2D(output, specaxis=:f;
                 trange=(-50e-15, 50e-15), bandpass=nothing,
                 λrange=(150e-9, 2000e-9), dBmin=-60,
                 kwargs...)
    z = output["z"]*1e2
    specx, Iω = getIω(output, specaxis)
    t, Et = getEt(output, trange=trange, bandpass=bandpass)
    It = abs2.(Et)

    speclims, speclabel, specxfac = getspeclims(λrange, specaxis)
    specx .*= specxfac

    multimode, modes = get_modes(output)

    if multimode
        _prop2D_mm(modes, t, z, specx, It, Iω,
                   speclabel, speclims, trange, dBmin, window_str(bandpass); kwargs...)
    else
        _prop2D_sm(t, z, specx, It, Iω,
                   speclabel, speclims, trange, dBmin, window_str(bandpass); kwargs...)
    end    
end

# Helper function to convert λrange to the correct numbers depending on specaxis
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

# single-mode 2D propagation plots
function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin, bpstr; kwargs...)
    num = "Propagation" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = plt.subplots(1, 2, num=num)
    pfig.set_size_inches(12, 4)
    Iω = Maths.normbymax(Iω)
    _spec2D_log(axs[1], specx, z, Iω, dBmin, speclabel, speclims; kwargs...)

    _time2D(axs[2], t, z, It, trange; kwargs...)
    pfig.tight_layout()
    return pfig
end

# multi-mode 2D propagation plots
function _prop2D_mm(modes, t, z, specx, It, Iω, speclabel, speclims, trange, dBmin, bpstr; kwargs...)
    pfigs = []
    Iω = Maths.normbymax(Iω)
    for mi in 1:length(modes)
        num = "Propagation ($(modes[mi]))" * ((length(bpstr) > 0) ? ", $bpstr" : "")
        pfig, axs = plt.subplots(1, 2, num=num)
        pfig.set_size_inches(12, 4)
        _spec2D_log(axs[1], specx, z, Iω[:, mi, :], dBmin, speclabel, speclims; kwargs...)

        _time2D(axs[2], t, z, It[:, mi, :], trange; kwargs...)
        push!(pfigs, pfig)
    end

    num = "Propagation (all modes)" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = plt.subplots(1, 2, num=num)
    pfig.set_size_inches(12, 4)
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    _spec2D_log(axs[1], specx, z, Iωall, dBmin, speclabel, speclims; kwargs...)

    Itall = dropdims(sum(It, dims=2), dims=2)
    _time2D(axs[2], t, z, Itall, trange; kwargs...)
    pfig.tight_layout()
    push!(pfigs, pfig)

    return pfigs
end

# a single logarithmic colour-scale spectral domain plot
function _spec2D_log(ax, specx, z, I, dBmin, speclabel, speclims; kwargs...)
    im = ax.pcolormesh(specx, z, 10*log10.(transpose(I)); kwargs...)
    im.set_clim(dBmin, 0)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("SED (dB)")
    ax.set_ylabel("Distance (cm)")
    ax.set_xlabel(speclabel)
    ax.set_xlim(speclims...)
end

# a single time-domain propagation plot
function _time2D(ax, t, z, I, trange; kwargs...)
    Pfac, unit = power_unit(I)
    im = ax.pcolormesh(t*1e15, z, Pfac*transpose(I); kwargs...)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Power ($unit)")
    ax.set_xlim(trange.*1e15)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (cm)")
end

"""
    time_1D(output, zslice, y=:Pt, kwargs...)

Create lineplots of time-domain slice(s) of the propagation.

The keyword argument `y` determines
what is plotted: `:Pt` (power, default), `:Esq` (squared electric field) or `:Et` (electric field).

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

The keyword argument `oversampling` determines the amount of oversampling done before plotting.

Other `kwargs` are passed onto `plt.plot`.
"""
function time_1D(output, zslice=maximum(output["z"]);
                y=:Pt, modes=nothing,
                oversampling=4, trange=(-50e-15, 50e-15), bandpass=nothing,
                kwargs...)
    t, Et, zactual = getEt(output, zslice,
                           trange=trange, oversampling=oversampling, bandpass=bandpass)
    if y == :Pt
        yt = abs2.(Et)
    elseif y == :Et
        yt = real(Et)
    elseif y == :Esq
        yt = real(Et).^2
    else
        error("unknown time plot variable $y")
    end
    multimode, modestrs = get_modes(output)
    if multimode
        if modes == :sum
            y == :Pt || error("Modal sum can only be plotted for power!")
            yt = dropdims(sum(yt, dims=2), dims=2)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            yt = yt[:, modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    end

    yfac, unit = power_unit(abs2.(Et), y)

    sfig = plt.figure()
    if multimode && nmodes > 1
        _plot_slice_mm(plt.gca(), t*1e15, yfac*yt, zactual, modestrs; kwargs...)
    else
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        for iz in eachindex(zactual)
            plt.plot(t*1e15, yfac*yt[:, iz]; label=label[iz], kwargs...)
        end
    end
    plt.legend(frameon=false)
    add_fwhm_legends(plt.gca(), "fs")
    plt.xlabel("Time (fs)")
    plt.xlim(1e15.*trange)
    ylab = y == :Et ?  "Field ($unit)" : "Power ($unit)"
    plt.ylabel(ylab)
    y == :Et || plt.ylim(ymin=0)
    sfig.set_size_inches(8.5, 5)
    sfig.tight_layout()
end

# Automatically find power unit depending on scale of electric field.
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
    spec_1D(output, zslice, specaxis=:λ, log10=true, log10min=1e-6)

Create lineplots of spectral-domain slices of the propagation.

The x-axis is determined by `specaxis` (see [`getIω`](@ref)).

If `log10` is true, plot on a logarithmic scale, with a y-axis range of `log10min`. 

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

Other `kwargs` are passed onto `plt.plot`.
"""
function spec_1D(output, zslice=maximum(output["z"]), specaxis=:λ;
                 modes=nothing, λrange=(150e-9, 1200e-9),
                 log10=true, log10min=1e-6,
                 kwargs...)
    specx, Iω, zactual = getIω(output, specaxis, zslice)
    speclims, speclabel, specxfac = getspeclims(λrange, specaxis)
    multimode, modestrs = get_modes(output)
    if multimode
        modes = isnothing(modes) ? (1:size(Iω, 2)) : modes
        if modes == :sum
            Iω = dropdims(sum(Iω, dims=2), dims=2)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            Iω = Iω[:, modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    end

    specx .*= specxfac

    sfig = plt.figure()
    if multimode && nmodes > 1
        _plot_slice_mm(plt.gca(), specx, Iω, zactual, modestrs, log10; kwargs...)
    else
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        for iz in eachindex(zactual)
            (log10 ? plt.semilogy : plt.plot)(specx, Iω[:, iz]; label=label[iz], kwargs...)
        end
    end
    plt.legend(frameon=false)
    plt.xlabel(speclabel)
    plt.ylabel("Spectral energy density")
    log10 && plt.ylim(3*maximum(Iω)*log10min, 3*maximum(Iω))
    plt.xlim(speclims...)
    sfig.set_size_inches(8.5, 5)
    sfig.tight_layout()
end

dashes = [(0, (10, 1)),
          (0, (5, 1)),
          (0, (1, 0.5)),
          (0, (1, 0.5, 1, 0.5, 3, 1)),
          (0, (5, 1, 1, 1))]

function _plot_slice_mm(ax, x, y, z, modestrs, log10=false, fwhm=false; kwargs...)
    pfun = (log10 ? ax.semilogy : ax.plot)
    for sidx = 1:size(y, 3) # iterate over z-slices
        zs = @sprintf("%.2f cm", z[sidx]*100)
        line = pfun(x, y[:, 1, sidx]; label="$zs ($(modestrs[1]))", kwargs...)[1]
        for midx = 2:size(y, 2) # iterate over modes
            pfun(x, y[:, midx, sidx], linestyle=dashes[midx], color=line.get_color(),
                 label="$zs ($(modestrs[midx]))"; kwargs...)
        end
    end
end

spectrogram(output::AbstractOutput, args...; kwargs...) = spectrogram(
    makegrid(output), output, args...; kwargs...)

function spectrogram(grid, Eω::AbstractArray, specaxis=:λ; kwargs...)
    t, Et = getEt(grid, Eω, oversampling=1)
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(grid::Grid.AbstractGrid, output, zslice, specaxis=:λ; kwargs...)
    t, Et, zactual = getEt(output, zslice, oversampling=1)
    Et = Et[:, 1]
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
                     trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
                     kwargs...)
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    specy, Ig = getIω(ω, g*Maths.rfftnorm(t[2]-t[1]), specaxis)
    speclims, speclabel, specyfac = getspeclims(λrange, specaxis)

    log && (Ig = 10*log10.(Maths.normbymax(Ig)))

    plt.figure()
    plt.pcolormesh(tg.*1e15, specyfac*specy, Ig; kwargs...)
    plt.ylim(speclims...)
    plt.ylabel(speclabel)
    plt.xlabel("Time (fs)")
    log && plt.clim(dBmin, 0)
    plt.colorbar()
end

function auto_fwhm_arrows(ax, x, y; color="k", arrowlength=nothing, hpad=0, linewidth=1,
                                    text=nothing, units="fs")
    left, right = Maths.level_xings(x, y)
    fw = abs(right - left)
    halfmax = maximum(y)/2
    arrowlength = isnothing(arrowlength) ? 2*fw : arrowlength

    ax.annotate("", xy=(left-hpad, halfmax),
                xytext=(left-hpad-arrowlength, halfmax),
                arrowprops=Dict("arrowstyle" => "->",
                                "color" => color,
                                "linewidth" => linewidth))
    ax.annotate("", xy=(right+hpad, halfmax),
                xytext=(right+hpad+arrowlength, halfmax),
                arrowprops=Dict("arrowstyle" => "->",
                                "color" => color,
                                "linewidth" => linewidth))

    if text == :left
        ax.text(left-arrowlength/2, 1.1*halfmax, @sprintf("%.1f %s", fw, units),
                ha="right", color=color)
    elseif text == :right
        ax.text(right+arrowlength/2, 1.1*halfmax, @sprintf("%.1f %s", fw, units),
                color=color)
    end
end

function add_fwhm_legends(ax, unit)
    leg = ax.get_legend()
    texts = leg.get_texts()
    handles, labels = ax.get_legend_handles_labels()
    
    for (ii, line) in enumerate(handles)
        xy = line.get_xydata()
        fw = Maths.fwhm(xy[:, 1], xy[:, 2])
        t = texts[ii]
        s = t.get_text()
        s *= @sprintf(" [%.2f %s]", fw, unit)
        t.set_text(s)
    end
end

end