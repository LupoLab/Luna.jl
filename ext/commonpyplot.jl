# Common implementation for PyPlotExt and PythonPlotExt.
# Expects the including module to define:
#   plt          - the plotting module (PyPlot.plt or PythonPlot.pyplot)
#   matplotlib   - the matplotlib module
#   ColorMap     - the ColorMap constructor
#   Figure       - the Figure type
#   convertany(x)    - identity for PyPlot, pyconvert(Any, x) for PythonPlot
#   convertarray(x)  - identity for PyPlot, pyconvert(Array, x) for PythonPlot
#
# When editing this file, keep in mind the key difference between the two backends:
# PyPlot (via PyCall) auto-converts Python return values to Julia types (1-based
# indexing), while PythonPlot (via PythonCall) returns raw Py objects, which keep
# Python semantics (0-based indexing). Avoid indexing into return values of
# matplotlib calls; iterate (or use first/zip) or convert explicitly instead.

import Luna: Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import Luna.Plotting
import Luna.Plotting: get_modes, power_unit, getspeclims, modeidcs, window_str, should_log10
import FFTW
import Printf: @sprintf
import Base: display

"""
    displayall()

`display` all currently open figures.
"""
function displayall()
    for fign in plt.get_fignums()
        fig = Figure(plt.figure(fign))
        display(fig)
    end
end

display(figs::AbstractArray{Figure, N}) where N = [display(fig) for fig in figs]

function cmap_white(cmap; N=2^12, n=8)
    vals = collect(range(0, 1, length=n))
    vals_i = collect(range(0, 1, length=N))
    cm = ColorMap(cmap)
    clist = convertarray(cm(vals))
    clist[1, :] = [1, 1, 1, 1]
    clist_i = Array{Float64}(undef, (N, 4))
    for ii in 1:4
        clist_i[:, ii] .= Maths.BSpline(vals, clist[:, ii]).(vals_i)
    end
    ColorMap(matplotlib.colors.ListedColormap(clist_i))
end

"""
    cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)

Make an array of `num` different colours that follow the colourmap `cmap` between the values
`cmin` and `cmax`.
"""
function cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)
    cm = ColorMap(cmap)
    n = collect(range(cmin, cmax; length=num))
    cm.(n)
end

"""
    subplotgrid(N, portrait=true; colw=4, rowh=2.5, title=nothing)

Create a figure with `N` subplots laid out in a grid that is as close to square as possible.
If `portrait` is `true`, try to lay out the grid in portrait orientation (taller than wide),
otherwise landscape (wider than tall).
"""
function subplotgrid(N, portrait=true; colw=4, rowh=2.5, title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N/cols)
    portrait && ((rows, cols) = (cols, rows))
    # clear=true: re-use (and clear) an existing figure with the same title rather than
    # erroring, which matplotlib ≥3.7 does by default for duplicate figure labels
    fig, axs = plt.subplots(rows, cols, num=title, clear=true)
    axs = convertany(axs)
    ndims(axs) > 1 && (axs = permutedims(axs, (2, 1)))
    if cols*rows > N
        for axi in axs[N+1:end]
            axi.remove()
        end
    end
    fig.set_size_inches(cols*colw, rows*rowh)
    fig, N > 1 ? axs : [axs]
end

function stats(z, pstats, fstats, multimode, modes; kwargs...)
    figs = Any[]
    for (stats_i, title, always_log) in ((pstats, "Pulse stats", true),
                                         (fstats, "Other stats", false))
        Npl = length(stats_i)
        Npl > 0 || continue
        fig, axs = subplotgrid(Npl, title=title)
        for n in 1:Npl
            ax = axs[n]
            data, label = stats_i[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            if multimode && (ndims(data) > 1)
                (always_log || should_log10(data)) && ax.semilogy()
                ax.legend(modes, frameon=false)
            end
        end
        fig.tight_layout()
        push!(figs, Figure(fig))
    end
    figs
end

function prop_2D(output, specaxis=:f;
                 trange=(-50e-15, 50e-15), bandpass=nothing,
                 λrange=(150e-9, 2000e-9), dBmin=-60,
                 resolution=nothing, modes=nothing, oversampling=4,
                 kwargs...)
    z = output["z"]*1e2
    if specaxis == :λ
        specx, Iω = getIω(output, specaxis, specrange=λrange, resolution=resolution)
    else
        specx, Iω = getIω(output, specaxis, resolution=resolution)
    end

    t, Et = getEt(output; trange, bandpass, oversampling)
    It = abs2.(Et)

    speclims, speclabel, specxfac = getspeclims(λrange, specaxis)
    specx .*= specxfac

    multimode, modelabels = get_modes(output)

    if multimode
        fig = _prop2D_mm(modelabels, modeidcs(modes, modelabels), t, z, specx, It, Iω,
                         speclabel, speclims, trange, dBmin, window_str(bandpass);
                         kwargs...)
    else
        fig = _prop2D_sm(t, z, specx, It, Iω,
                         speclabel, speclims, trange, dBmin, window_str(bandpass);
                         kwargs...)
    end
    fig
end

# single-mode 2D propagation plots
function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin, bpstr; kwargs...)
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    num = id * "Propagation" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = plt.subplots(1, 2, num=num)
    axs = convertarray(axs)
    pfig.set_size_inches(12, 4)
    Iω = Maths.normbymax(Iω)
    _spec2D_log(axs[1], specx, z, Iω, dBmin, speclabel, speclims; kwargs...)
    _time2D(axs[2], t, z, It, trange; kwargs...)
    pfig.tight_layout()
    return Figure(pfig)
end

# multi-mode 2D propagation plots
function _prop2D_mm(modelabels, modes, t, z, specx, It, Iω,
                    speclabel, speclims, trange, dBmin, bpstr;
                    kwargs...)
    pfigs = Any[]
    Iω = Maths.normbymax(Iω)
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    for mi in modes
        num = id * "Propagation ($(modelabels[mi]))" * ((length(bpstr) > 0) ? ", $bpstr" : "")
        pfig, axs = plt.subplots(1, 2, num=num)
        axs = convertarray(axs)
        pfig.set_size_inches(12, 4)
        _spec2D_log(axs[1], specx, z, Iω[:, mi, :], dBmin, speclabel, speclims; kwargs...)
        _time2D(axs[2], t, z, It[:, mi, :], trange; kwargs...)
        push!(pfigs, Figure(pfig))
    end

    num = id * "Propagation (all modes)" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = plt.subplots(1, 2, num=num)
    axs = convertarray(axs)
    pfig.set_size_inches(12, 4)
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    _spec2D_log(axs[1], specx, z, Iωall, dBmin, speclabel, speclims; kwargs...)

    Itall = dropdims(sum(It, dims=2), dims=2)
    _time2D(axs[2], t, z, Itall, trange; kwargs...)
    pfig.tight_layout()
    push!(pfigs, Figure(pfig))

    return pfigs
end

# a single logarithmic colour-scale spectral domain plot
function _spec2D_log(ax, specx, z, I, dBmin, speclabel, speclims; kwargs...)
    im = ax.pcolormesh(specx, z, 10*log10.(transpose(I)); shading="auto", rasterized=true, kwargs...)
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
    im = ax.pcolormesh(t*1e15, z, Pfac*transpose(I); shading="auto", rasterized=true, kwargs...)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Power ($unit)")
    ax.set_xlim(trange.*1e15)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (cm)")
end

function time_1D(output, zslice=maximum(output["z"]);
                y=:Pt, modes=nothing,
                oversampling=4, trange=(-50e-15, 50e-15), bandpass=nothing,
                FTL=false, propagate=nothing,
                kwargs...)
    t, Et, zactual = getEt(output, zslice,
                           trange=trange, oversampling=oversampling, bandpass=bandpass,
                           FTL=FTL, propagate=propagate)
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
    Figure(sfig)
end

function spec_1D(output, zslice=maximum(output["z"]), specaxis=:λ;
                 modes=nothing, λrange=(150e-9, 1200e-9),
                 log10=true, log10min=1e-6, resolution=nothing,
                 kwargs...)
    if specaxis == :λ
        specx, Iω, zactual = getIω(output, specaxis, zslice, specrange=λrange, resolution=resolution)
    else
        specx, Iω, zactual = getIω(output, specaxis, zslice, resolution=resolution)
    end
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
    Figure(sfig)
end

const dashes = [(0, (10, 1)),
                (0, (5, 1)),
                (0, (1, 0.5)),
                (0, (1, 0.5, 1, 0.5, 3, 1)),
                (0, (5, 1, 1, 1))]

function _plot_slice_mm(ax, x, y, z, modestrs, log10=false, fwhm=false; kwargs...)
    pfun = (log10 ? ax.semilogy : ax.plot)
    for sidx = 1:size(y, 3) # iterate over z-slices
        zs = @sprintf("%.2f cm", z[sidx]*100)
        # first(...) rather than [1]: PythonPlot returns a Py list (0-based indexing)
        line = first(pfun(x, y[:, 1, sidx]; label="$zs ($(modestrs[1]))", kwargs...))
        for midx = 2:size(y, 2) # iterate over modes
            pfun(x, y[:, midx, sidx], linestyle=dashes[mod1(midx-1, length(dashes))],
                 color=line.get_color(),
                 label="$zs ($(modestrs[midx]))"; kwargs...)
        end
    end
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
                     trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
                     surface3d=false,
                     kwargs...)
    surface3d && @warn("3D surface spectrograms are only supported by the Makie backend; "*
                       "falling back to a 2D spectrogram.")
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    specy, Ig = getIω(ω, g*Maths.rfftnorm(t[2]-t[1]), specaxis)
    speclims, speclabel, specyfac = getspeclims(λrange, specaxis)

    log && (Ig = 10*log10.(Maths.normbymax(Ig)))

    fig = plt.figure()
    plt.pcolormesh(tg.*1e15, specyfac*specy, Ig; shading="auto", rasterized=true, kwargs...)
    plt.ylim(speclims...)
    plt.ylabel(speclabel)
    plt.xlabel("Time (fs)")
    log && plt.clim(dBmin, 0)
    plt.colorbar()
    Figure(fig)
end

function energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))
    e = Processing.energy(output; bandpass=bandpass)
    eall = Processing.energy(output)

    multimode, modestrs = get_modes(output)
    if multimode
        e0 = sum(eall[:, 1])
        modes = isnothing(modes) ? (1:size(e, 1)) : modes
        if modes == :sum
            e = dropdims(sum(e, dims=1), dims=1)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            e = e[modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    else
        e0 = eall[1]
    end

    z = output["z"]*100
    # shape (nz,) or (nz, nmodes): matplotlib plots one line per column
    edata = ndims(e) > 1 ? permutedims(e) : e

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(z, 1e6*edata)
    ax.set_xlim(extrema(z)...)
    ax.set_ylim(ymin=0)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Energy (μJ)")
    rax = ax.twinx()
    rax.plot(z, 100*(edata./e0), linewidth=0)
    lims = convertany(ax.get_ylim())
    rax.set_ylim(100/(1e6*e0).*lims)
    rax.set_ylabel("Conversion efficiency (%)")
    fig.set_size_inches(figsize...)
    Figure(fig)
end

"""
    auto_fwhm_arrows(ax, x, y; color="k", arrowlength=nothing, hpad=0, linewidth=1,
                     text=nothing, units="fs", kwargs...)

Draw FWHM arrows on an axis, indicating the full-width at half-maximum of the data `y(x)`.
"""
function auto_fwhm_arrows(ax, x, y; color="k", arrowlength=nothing, hpad=0, linewidth=1,
                                    text=nothing, units="fs", kwargs...)
    left, right = Maths.level_xings(x, y; kwargs...)
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
        ax.text(left-arrowlength/2, 1.1*halfmax, @sprintf("%.2f %s", fw, units),
                ha="right", color=color)
    elseif text == :right
        ax.text(right+arrowlength/2, 1.1*halfmax, @sprintf("%.2f %s", fw, units),
                color=color)
    end
end

"""
    add_fwhm_legends(ax, unit)

Enhance the legend of `ax` by appending the FWHM of each plotted line to its label.
"""
function add_fwhm_legends(ax, unit)
    leg = ax.get_legend()
    texts = leg.get_texts()
    handles, labels = ax.get_legend_handles_labels()
    # zip rather than indexing: PyPlot returns 1-based Julia vectors here but
    # PythonPlot returns 0-based Py lists; iteration works identically for both
    for (line, t) in zip(handles, texts)
        xy = convertarray(line.get_xydata())
        fw = Maths.fwhm(xy[:, 1], xy[:, 2])
        s = convertany(t.get_text())
        s *= @sprintf(" [%.2f %s]", fw, unit)
        t.set_text(s)
    end
end

"""
    cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)

Place a `text` in the axes `ax` in the corner defined by `corner`. Padding can be
defined for `x` and `y` together via `pad` or separately via `xpad` and `ypad`.

Possible values for `corner` are `ul`, `ur`, `ll`, `lr` where the first letter
defines upper/lower and the second defines left/right.
"""
function cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)
    xpad = isnothing(xpad) ? pad : xpad
    ypad = isnothing(ypad) ? pad : ypad
    if corner[1] == 'u'
        val = "top"
        y = 1 - ypad
    elseif corner[1] == 'l'
        val = "bottom"
        y = ypad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    if corner[2] == 'l'
        hal = "left"
        x = xpad
    elseif corner[2] == 'r'
        hal = "right"
        x = 1 - xpad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    ax.text(x, y, text; horizontalalignment=hal, verticalalignment=val,
                 transform=ax.transAxes, kwargs...)
end
