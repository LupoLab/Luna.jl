module MakieExt
import Luna: Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import Luna.Plotting: get_modes, power_unit, getspeclims, modeidcs, window_str, should_log10
import FFTW
import Printf: @sprintf
import Makie

import Luna.Plotting

function tex(s)
    # Prefer unicode superscripts for common cases to keep font consistency
    # (LaTeXString often uses a different math font which can look out of place)
    # We replace patterns with and without the $ delimiters to catch various sources
    replacements = [
        "\$^{1/2}\$" => "¹/²",
        "\$^2\$" => "²",
        "\$^3\$" => "³",
        "\$^{-3}\$" => "⁻³",
        "\$^{1/2}" => "¹/²",
        "\$^2" => "²",
        "\$^3" => "³",
        "\$^{-3}" => "⁻³",
        "^2" => "²",
        "^3" => "³",
        "^{-3}" => "⁻³",
    ]
    for r in replacements
        s = replace(s, r)
    end
    
    if contains(s, '$')
        try
            return Makie.LaTeXString(s)
        catch
            return s
        end
    end
    return s
end

function newfig(; size=(800, 600))
    fig = Makie.Figure(; size)
    if nameof(Makie.current_backend()) in (:GLMakie, :WGLMakie)
        Makie.DataInspector(fig)
    end
    fig
end

function display_fig(fig)
    backend = Makie.current_backend()
    if nameof(backend) == :GLMakie && isdefined(backend, :Screen)
        display(backend.Screen(), fig)
    else
        display(fig)
    end
end

function cmap_white(cmap; N=2^12, n=8)
    cm = Makie.categorical_colors(cmap, n)
    cm[1] = Makie.RGBAf(1, 1, 1, 1)
    result = Vector{Makie.RGBAf}(undef, N)
    for i in 1:N
        t = (i - 1) / (N - 1)
        idx = clamp(floor(Int, t * (n - 1)) + 1, 1, n - 1)
        frac = t * (n - 1) - (idx - 1)
        c1, c2 = cm[idx], cm[min(idx + 1, n)]
        result[i] = Makie.RGBAf(
            c1.r + frac * (c2.r - c1.r),
            c1.g + frac * (c2.g - c1.g),
            c1.b + frac * (c2.b - c1.b),
            c1.alpha + frac * (c2.alpha - c1.alpha))
    end
    result
end

function cmap_colours(num, cmap=:viridis; cmin=0, cmax=0.8)
    cm = Makie.categorical_colors(cmap, 256)
    n = range(cmin, cmax; length=num)
    [cm[clamp(round(Int, v * 255) + 1, 1, 256)] for v in n]
end

function subplotgrid(N, portrait=true; colw=350, rowh=250, title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N / cols)
    portrait && ((rows, cols) = (cols, rows))
    collect(Iterators.product(1:rows, 1:cols)), cols * colw, rows * rowh
end

function stats(z, pstats, fstats, multimode, modes; kwargs...)
    figs = []
    Npl = length(pstats)
    if Npl > 0
        idcs, width, height = subplotgrid(Npl)
        pfig = newfig(size=(width, height))
        for n in 1:Npl
            data, ylabel = pstats[n]
            scale = (multimode && ndims(data) > 1) ? log10 : identity
            data = (multimode && ndims(data) > 1) ? data : reshape(data, 1, :)
            data = (scale == log10) ? max.(data, 1e-300) : data
            ax = Makie.Axis(pfig[idcs[n]...]; xlabel="Distance (cm)", ylabel=tex(ylabel), yscale=scale)
            for i in axes(data, 1)
                Makie.lines!(ax, z, data[i, :], label=modes[i])
            end
            multimode && size(data, 1) > 1 && Makie.axislegend(ax, framevisible=false)
        end
        push!(figs, pfig)
        display_fig(pfig)
    end

    Npl = length(fstats)
    if Npl > 0
        idcs, width, height = subplotgrid(Npl)
        ffig = newfig(size=(width, height))
        for n in 1:Npl
            data, ylabel = fstats[n]
            scale = (multimode && ndims(data) > 1 && should_log10(data)) ? log10 : identity
            data = (multimode && ndims(data) > 1) ? data : reshape(data, 1, :)
            data = (scale == log10) ? max.(data, 1e-300) : data
            ax = Makie.Axis(ffig[idcs[n]...]; xlabel="Distance (cm)", ylabel=tex(ylabel), yscale=scale)
            for i in axes(data, 1)
                Makie.lines!(ax, z, data[i, :], label=modes[i])
            end
            multimode && size(data, 1) > 1 && Makie.axislegend(ax, framevisible=false)
        end
        push!(figs, ffig)
        display_fig(ffig)
    end
    figs
end

function prop_2D(output, specaxis=:f;
                 trange=(-50e-15, 50e-15), bandpass=nothing,
                 λrange=(150e-9, 2000e-9), dBmin=-60,
                 resolution=nothing, modes=nothing, oversampling=4,
                 kwargs...)
    z = output["z"] * 1e2
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
        return _prop2D_mm(modelabels, modeidcs(modes, modelabels), t, z, specx, It, Iω,
                          speclabel, speclims, trange, dBmin, window_str(bandpass);
                          kwargs...)
    else
        # return the figure directly (not a vector) to match the matplotlib backends
        return _prop2D_sm(t, z, specx, It, Iω,
                          speclabel, speclims, trange, dBmin, window_str(bandpass);
                          kwargs...)
    end
end

function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin, bpstr; kwargs...)
    Iω = Maths.normbymax(Iω)
    fig = _prop2D_fig(specx, z, Iω, dBmin, speclabel, speclims, t, It, trange; kwargs...)
    display_fig(fig)
    fig
end

function _prop2D_mm(modelabels, modes, t, z, specx, It, Iω,
                    speclabel, speclims, trange, dBmin, bpstr;
                    kwargs...)
    pfigs = []
    Iω = Maths.normbymax(Iω)
    for mi in modes
        pfig = _prop2D_fig(specx, z, Iω[:, mi, :], dBmin, speclabel, speclims,
                           t, It[:, mi, :], trange; title=modelabels[mi], kwargs...)
        display_fig(pfig)
        push!(pfigs, pfig)
    end

    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    Itall = dropdims(sum(It, dims=2), dims=2)
    pfig = _prop2D_fig(specx, z, Iωall, dBmin, speclabel, speclims,
                       t, Itall, trange; title="All modes", kwargs...)
    display_fig(pfig)
    push!(pfigs, pfig)
    return pfigs
end

function _prop2D_fig(specx, z, Iω, dBmin, speclabel, speclims, t, It, trange; title=nothing, kwargs...)
    cmap = get(kwargs, :cmap, :viridis)
    pfig = newfig(size=(1000, 450))
    if !isnothing(title)
        Makie.Label(pfig[0, :], title, font=:bold)
        Makie.rowsize!(pfig.layout, 0, Makie.Fixed(20))
    end
    ax, hm = Makie.heatmap(pfig[1, 1], specx, z, 10 * log10.(Iω),
                           colorrange=(dBmin, 0), interpolate=false,
                           rasterize=3, colormap=cmap,
                           axis=(; xlabel=tex(speclabel), ylabel="Distance (cm)"))
    Makie.xlims!(ax, speclims)
    Makie.Colorbar(pfig[1, 2], hm, label="SED (dB)")

    Pfac, unit = power_unit(It)
    ax2, hm2 = Makie.heatmap(pfig[1, 3], t * 1e15, z, Pfac .* It,
                             interpolate=false, rasterize=3,
                             colormap=cmap,
                             axis=(; xlabel="Time (fs)", ylabel="Distance (cm)"))
    Makie.xlims!(ax2, trange .* 1e15)
    Makie.Colorbar(pfig[1, 4], hm2, label=tex("Power ($unit)"))
    
    # Ensure plots have equal width and colorbars are narrow
    Makie.colsize!(pfig.layout, 1, Makie.Relative(0.42))
    Makie.colsize!(pfig.layout, 2, Makie.Fixed(15))
    Makie.colsize!(pfig.layout, 3, Makie.Relative(0.42))
    Makie.colsize!(pfig.layout, 4, Makie.Fixed(15))
    Makie.colgap!(pfig.layout, 1, Makie.Fixed(5))
    Makie.colgap!(pfig.layout, 2, Makie.Fixed(40))
    Makie.colgap!(pfig.layout, 3, Makie.Fixed(5))
    
    pfig
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
        yt = real(Et) .^ 2
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

    xlabel = "Time (fs)"
    ylabel = y == :Et ? "Field ($unit)" : "Power ($unit)"
    if multimode && nmodes > 1
        sfig = _plot_slice_mm(t * 1e15, yfac * yt, zactual, modestrs, xlabel, ylabel,
                              fwlabel=true; kwargs...)
    else
        sfig = newfig()
        zs = [@sprintf("%.2f cm", zi * 100) for zi in zactual]
        labels = multimode ? zs .* " ($modestrs)" : zs
        ax = Makie.Axis(sfig[1, 1]; xlabel=tex(xlabel), ylabel=tex(ylabel))
        for iz in eachindex(zactual)
            fw = Maths.fwhm(t * 1e15, yfac * yt[:, iz])
            lbl = labels[iz] * @sprintf(" [%.2f fs]", fw)
            Makie.lines!(ax, t * 1e15, yfac * yt[:, iz]; label=lbl, kwargs...)
        end
        Makie.axislegend(ax, framevisible=false)
        Makie.xlims!(ax, (1e15 .* trange)...)
        y == :Et || Makie.ylims!(ax, 0, nothing)
    end
    display_fig(sfig)
    sfig
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

    if multimode && nmodes > 1
        sfig = _plot_slice_mm(specx, Iω, zactual, modestrs, speclabel,
                              "Spectral energy density", log10; kwargs...)
    else
        sfig = newfig()
        zs = [@sprintf("%.2f cm", zi * 100) for zi in zactual]
        labels = multimode ? zs .* " ($modestrs)" : zs
        scale = log10 ? Base.log10 : identity
        ax = Makie.Axis(sfig[1, 1], yscale=scale, xlabel=tex(speclabel),
                        ylabel=tex("Spectral energy density"))
        for iz in eachindex(zactual)
            Makie.lines!(ax, specx, Iω[:, iz]; label=labels[iz], kwargs...)
        end
        Makie.axislegend(ax, framevisible=false)
        log10 && Makie.ylims!(ax, 3 * maximum(Iω) * log10min, 3 * maximum(Iω))
        Makie.xlims!(ax, speclims...)
    end
    display_fig(sfig)
    sfig
end

const _mm_linestyles = [
    :solid,
    :dash,
    :dot,
    :dashdot,
    :dashdotdot,
]

function _plot_slice_mm(x, y, z, modestrs, xlabel, ylabel, log10=false;
                       fwlabel=false, kwargs...)
    pfig = newfig()
    scale = log10 ? Base.log10 : identity
    ax = Makie.Axis(pfig[1, 1], yscale=scale, xlabel=tex(xlabel), ylabel=tex(ylabel))
    for sidx = 1:size(y, 3) # iterate over z-slices
        zs = @sprintf("%.2f cm", z[sidx] * 100)
        # Using a fixed color per z-slice, but different linestyle per mode
        for midx = 1:size(y, 2)
            label = "$zs ($(modestrs[midx]))"
            if fwlabel
                fw = Maths.fwhm(x, y[:, midx, sidx])
                label *= @sprintf(" [%.2f fs]", fw)
            end
            Makie.lines!(ax, x, y[:, midx, sidx];
                         label=label,
                         linestyle=_mm_linestyles[mod1(midx, length(_mm_linestyles))],
                         color=Makie.Cycled(sidx),
                         kwargs...)
        end
    end
    Makie.axislegend(ax, framevisible=false)
    display_fig(pfig)
    pfig
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
                      trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
                      surface3d=false,
                      kwargs...)
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    speclims, speclabel, specyfac = getspeclims(λrange, specaxis)
    specy, Ig = getIω(ω, g * Maths.rfftnorm(t[2] - t[1]), specaxis,
                      specrange=speclims ./ specyfac)

    if log
        Ig = 10 * Base.log10.(Maths.normbymax(Ig))
        clims = (dBmin, 0)
    else
        clims = extrema(Ig)
    end

    cmap = get(kwargs, :cmap, :viridis)

    fig = newfig()
    if surface3d
        ax, pl = Makie.surface(fig[1, 1], tg .* 1e15, specyfac * specy, Ig',
                         colorrange=clims, colormap=cmap,
                         axis=(; type=Makie.Axis3, azimuth=pi / 4, elevation=pi / 4,
                         protrusions=75, perspectiveness=0.0, viewmode=:stretch,
                         xlabel=tex("Time (fs)"), ylabel=tex(speclabel), ylabeloffset=80,
                         xlabeloffset=80, zgridvisible=false, zlabelvisible=false,
                         zticksvisible=false, zticklabelsvisible=false,
                         yzpanelvisible=false, xzpanelvisible=false,
                         ygridvisible=false, xgridvisible=false,
                         zspinesvisible=false, zautolimitmargin=(0, 0),
                         xautolimitmargin=(0.0, 0.0), yautolimitmargin=(0, 0),
                         xspinesvisible=false, yspinesvisible=false))
    else
        ax, pl = Makie.heatmap(fig[1, 1], tg .* 1e15, specyfac * specy, Ig',
                               colorrange=clims, interpolate=false, rasterize=3,
                               colormap=cmap,
                               axis=(; xlabel=tex("Time (fs)"), ylabel=tex(speclabel)))
        Makie.ylims!(ax, speclims)
    end
    Makie.Colorbar(fig[1, 2], pl)
    display_fig(fig)
    fig
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
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            e = e[modes, :]
            modestrs = modestrs[modes]
        end
    else
        e0 = eall[1]
    end

    z = output["z"] * 100
    edata = ndims(e) > 1 ? e' : reshape(e, :, 1)

    labels = multimode ? (modestrs isa AbstractString ? [modestrs] : modestrs) : nothing

    fig = newfig(size=(round(Int, figsize[1]*100), round(Int, figsize[2]*100)))
    ax = Makie.Axis(fig[1, 1], xlabel=tex("Distance (cm)"), ylabel=tex("Energy (μJ)"))
    rax = Makie.Axis(fig[1, 1], yaxisposition=:right, ylabel=tex("Conversion efficiency (%)"))
    Makie.hidespines!(rax)
    Makie.hidexdecorations!(rax)
    for i in axes(edata, 2)
        lbl = isnothing(labels) ? nothing : labels[i]
        Makie.lines!(ax, z, 1e6 * edata[:, i]; label=lbl)
    end
    !isnothing(labels) && length(labels) > 1 && Makie.axislegend(ax, framevisible=false)

    maxe = maximum(1e6 * e)
    Makie.xlims!(ax, extrema(z)...)
    Makie.ylims!(ax, 0, maxe)
    Makie.ylims!(rax, 0, 100 * maxe / 1e6 / e0)
    Makie.xlims!(rax, extrema(z)...)
    display_fig(fig)
    fig
end

function auto_fwhm_arrows(ax, x, y; color=:black, arrowlength=nothing, hpad=0, linewidth=1,
                                    text=nothing, units="fs", kwargs...)
    left, right = Maths.level_xings(x, y; kwargs...)
    fw = abs(right - left)
    halfmax = maximum(y) / 2
    arrowlength = isnothing(arrowlength) ? 2 * fw : arrowlength

    Makie.arrows!(ax, [left - hpad - arrowlength], [halfmax], [arrowlength], [0.0];
                  color=color, linewidth=linewidth)
    Makie.arrows!(ax, [right + hpad + arrowlength], [halfmax], [-arrowlength], [0.0];
                  color=color, linewidth=linewidth)

    if text == :left
        Makie.text!(ax, left - arrowlength / 2, 1.1 * halfmax;
                    text=@sprintf("%.2f %s", fw, units),
                    align=(:right, :bottom), color=color)
    elseif text == :right
        Makie.text!(ax, right + arrowlength / 2, 1.1 * halfmax;
                    text=@sprintf("%.2f %s", fw, units),
                    align=(:left, :bottom), color=color)
    end
end

function cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)
    xpad = isnothing(xpad) ? pad : xpad
    ypad = isnothing(ypad) ? pad : ypad
    if corner[1] == 'u'
        valign = :top
        y = 1 - ypad
    elseif corner[1] == 'l'
        valign = :bottom
        y = ypad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    if corner[2] == 'l'
        halign = :left
        x = xpad
    elseif corner[2] == 'r'
        halign = :right
        x = 1 - xpad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    Makie.text!(ax, x, y; text=text, align=(halign, valign),
                space=:relative, kwargs...)
end

end # module
