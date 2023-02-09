module Plotting
import Luna: Grid, Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import GLMakie
import FFTW
import Printf: @sprintf
import Base: display

GLMakie.set_theme!(GLMakie.Theme(fontsize = 40))

function newfig()
    f = GLMakie.Figure(resolution = (1600, 1200))
    display(GLMakie.Screen(), f)
    f
end

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
    subplotgrid(N, portrait=true, kwargs...)

Create a figure with `N` subplots laid out in a grid that is as close to square as possible.
If `portrait` is `true`, try to lay out the grid in portrait orientation (taller than wide),
otherwise landscape (wider than tall).
"""
function subplotgrid(N, portrait=true; colw=4, rowh=2.5, title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N/cols)
    collect(Iterators.product(1:rows,1:cols))
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
    angles = zeros(length(mlines))
    for (ii, li) in enumerate(mlines)
        m = match(r"ϕ=(-?[0-9]+.[0-9]+)π", li)
        isnothing(m) && continue # no angle information in mode label)
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
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm²)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "min FWHM (fs)"))
    haskey(stats, "fwhm_t_max") && push!(pstats, (1e15*stats["fwhm_t_max"], "max FWHM (fs)"))
    haskey(stats, "fwhm_r") && push!(pstats, (1e6*stats["fwhm_r"], "Radial FWHM (μm)"))
    haskey(stats, "ω0") && push!(pstats, (1e9*wlfreq.(stats["ω0"]), "Central wavelength (nm)"))

    fstats = [] # fibre/waveguide/propagation statistics
    if haskey(stats, "electrondensity")
        push!(fstats, (1e-6*stats["electrondensity"], "Electron density (cm⁻³)"))
        if haskey(stats, "density")
            push!(fstats,
                 (100*stats["electrondensity"]./stats["density"], "Ionisation fraction (%)"))
        end
    end
    haskey(stats, "density") && push!(
        fstats, (1e-6*stats["density"], "Density (cm⁻³)"))
    haskey(stats, "pressure") && push!(
        fstats, (stats["pressure"], "Pressure (bar)"))
    haskey(stats, "dz") && push!(fstats, (1e6*stats["dz"], "Stepsize (μm)"))
    haskey(stats, "core_radius") && push!(fstats, (1e6*stats["core_radius"], "Core radius (μm)"))
    haskey(stats, "zdw") && push!(fstats, (1e9*stats["zdw"], "ZDW (nm)"))

    z = stats["z"]*1e2

    multimode, modes = get_modes(output)
    modes = isnothing(modes) ? [""] : modes

    Npl = length(pstats)
    if Npl > 0
        pfig = newfig()
        idcs = subplotgrid(Npl)
        for n in 1:Npl
            data, ylabel = pstats[n]
            data = data'
            scale = (multimode ? log10 : identity)
            ax = GLMakie.Axis(pfig[idcs[n]...]; xlabel="Distance (cm)", ylabel, yscale=scale)
            for i in 1:size(data,1)
                GLMakie.lines!(z, data[i,:], label=modes[i])
            end
            multimode && (ndims(data) > 1) && GLMakie.axislegend(framevisible=false)
        end
        GLMakie.DataInspector()
    end
    
    Npl = length(fstats)
    if Npl > 0
        ffig = newfig()
        idcs = subplotgrid(Npl)
        for n in 1:Npl
            data, ylabel = fstats[n]
            data = data'
            scale = ((multimode && should_log10(data)) ? log10 : identity)
            ax = GLMakie.Axis(ffig[idcs[n]...]; xlabel="Distance (cm)", ylabel, yscale=scale)
            for i in 1:size(data,1)
                GLMakie.lines!(z, data[i,:], label=modes[i])
            end
            multimode && (ndims(data) > 1) && GLMakie.axislegend(framevisible=false)
        end
        GLMakie.DataInspector()
    end
    [pfig, ffig]
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
- `resolution::Real` smooth the spectral energy density as defined by [`getIω`](@ref).
"""
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

modeidcs(m::Int, ml) = [m]
modeidcs(m::Symbol, ml) = (m == :sum) ? [] : error("modes must be :sum, a single integer, or iterable")
modeidcs(m::Nothing, ml) = 1:length(ml)
modeidcs(m, ml) = m

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
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    num = id * "Propagation" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    Iω = Maths.normbymax(Iω)
    _prop2D_fig(num, specx, z, Iω, dBmin, speclabel, speclims, t, It, trange)
end

# multi-mode 2D propagation plots
function _prop2D_mm(modelabels, modes, t, z, specx, It, Iω,
                    speclabel, speclims, trange, dBmin, bpstr;
                    kwargs...)
    pfigs = []
    Iω = Maths.normbymax(Iω)
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    for mi in modes
        num = id * "Propagation ($(modelabels[mi]))" * ((length(bpstr) > 0) ? ", $bpstr" : "")
        pfig = _prop2D_fig(num, specx, z, Iω[:, mi, :], dBmin, speclabel, speclims, t, It[:, mi, :], trange)
        push!(pfigs, pfig)
    end

    num = id * "Propagation (all modes)" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    Itall = dropdims(sum(It, dims=2), dims=2)
    pfig = _prop2D_fig(num, specx, z, Iωall, dBmin, speclabel, speclims, t, Itall, trange)
    push!(pfigs, pfig)
    return pfigs
end

function _prop2D_fig(name, specx, z, Iω, dBmin, speclabel, speclims, t, It, trange)
    pfig = newfig()
    ax, hm = GLMakie.heatmap(pfig[1,1], specx, z, 10*log10.(Iω),
                             colorrange=(dBmin,0), interpolate=true,
                             lowclip=:white,
                             axis=(; xlabel=speclabel, ylabel="Distance (cm)"))
    GLMakie.xlims!(ax, speclims)
    cb = GLMakie.Colorbar(pfig[1, 2], hm, label="SED (dB)")

    Pfac, unit = power_unit(It)
    ax, hm = GLMakie.heatmap(pfig[1,3], t*1e15, z, Pfac .* It,
                             interpolate=true,
                             lowclip=:white,
                             axis=(; xlabel="Time (fs)", ylabel="Distance (cm)"))
    GLMakie.xlims!(ax, trange.*1e15)
    cb = GLMakie.Colorbar(pfig[1, 4], hm, label="Power ($unit)")
    GLMakie.DataInspector()
    pfig
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

    xlabel = "Time (fs)"
    ylabel = y == :Et ?  "Field ($unit)" : "Power ($unit)"
    if multimode && nmodes > 1
       sfig = _plot_slice_mm(t*1e15, yfac*yt, zactual, modestrs, xlabel, ylabel, fwlabel=true)
    else
        sfig = newfig()
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        GLMakie.Axis(sfig[1, 1]; xlabel, ylabel)
        for iz in eachindex(zactual)
            fw = Maths.fwhm(t*1e15, yfac*yt[:, iz])
            label=(label[iz] * @sprintf(" [%.2f %s]", fw, "fs"))
            GLMakie.lines!(t*1e15, yfac*yt[:, iz]; label)
        end
    end
    GLMakie.axislegend(framevisible=false)
    GLMakie.xlims!((1e15.*trange)...)
    y == :Et || GLMakie.ylims!(low=0)
    GLMakie.DataInspector()
    sfig  
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
        sfig = _plot_slice_mm(specx, Iω, zactual, modestrs, speclabel, "Spectral energy density", log10)
    else
        sfig = newfig()
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        scale = (log10 ? Base.log10 : :identity)
        GLMakie.Axis(sfig[1, 1], yscale = scale, xlabel=speclabel, ylabel="Spectral energy density")
        for iz in eachindex(zactual)
            GLMakie.lines!(specx, Iω[:, iz], label=label[iz])
        end
    end
    GLMakie.axislegend(framevisible=false)
    log10 && GLMakie.ylims!(3*maximum(Iω)*log10min, 3*maximum(Iω))
    GLMakie.xlims!(speclims...)
    GLMakie.DataInspector()
    sfig
end

dashes = [:dash, :dot, :dashdot, :dashdotdot, [0.5, 1.0, 1.5, 2.5]]

function _plot_slice_mm(x, y, z, modestrs, xlabel, ylabel, log10=false; fwlabel=false)
    pfig = newfig()
    scale = (log10 ? Base.log10 : identity)
    GLMakie.Axis(pfig[1, 1], yscale = scale, xlabel=xlabel, ylabel=ylabel)
    for sidx = 1:size(y, 3) # iterate over z-slices
        zs = @sprintf("%.2f cm", z[sidx]*100)
        label = "$zs ($(modestrs[1]))"
        if fwlabel
            fw = Maths.fwhm(x, y[:, 1, sidx])
            label *= @sprintf(" [%.2f %s]", fw, "fs")
        end
        line = GLMakie.lines!(x, y[:, 1, sidx]; label)
        for midx = 2:size(y, 2) # iterate over modes
            label = "$zs ($(modestrs[midx]))"
            if fwlabel
                fw = Maths.fwhm(x, y[:, midx, sidx])
                label *= @sprintf(" [%.2f %s]", fw, "fs")
            end
            GLMakie.lines!(x, y[:, midx, sidx]; label,
                   color=line[:color], linestyle=dashes[midx])
        end
    end
    pfig
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
                     surface3d=false,
                     kwargs...)
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    speclims, speclabel, specyfac = getspeclims(λrange, specaxis)
    specy, Ig = getIω(ω, g*Maths.rfftnorm(t[2]-t[1]), specaxis,
                      specrange=speclims./specyfac)
    
    Ig = Maths.normbymax(Ig)
    log && (Ig = 10*log10.(Ig))
    clims = (log ? (dBmin, 0) : extrema(Ig))

    fig = newfig()
    if surface3d
        ax, pl = GLMakie.surface(fig[1,1], tg.*1e15, specyfac*specy, Ig',
                         colorrange=clims, colormap=:turbo,
                         axis=(;type=GLMakie.Axis3, azimuth = pi/4, elevation=pi/4,
                         protrusions=75, perspectiveness=0.0, viewmode=:stretch,
                         xlabel="Time (fs)", ylabel=speclabel, ylabeloffset=80,
                         xlabeloffset=80, zgridvisible=false, zlabelvisible=false,
                         zticksvisible=false, zticklabelsvisible=false,
                         yzpanelvisible=false, xzpanelvisible=false,
                         ygridvisible=false, xgridvisible=false,
                         zspinesvisible=false, zautolimitmargin=(0,0),
                         xautolimitmargin=(0.0,0.0), yautolimitmargin=(0,0),
                         xspinesvisible=false, yspinesvisible=false))
    else
        ax, pl = GLMakie.heatmap(fig[1,1], tg.*1e15, specyfac*specy, Ig',
                                colorrange=clims, interpolate=true,
                                axis=(; xlabel="Time (fs)", ylabel=speclabel))
        GLMakie.ylims!(ax, speclims)
    end
    GLMakie.Colorbar(fig[1, 2], pl)
    GLMakie.DataInspector()
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
    println("$(size(e'))")

    fig = newfig()
    ax = GLMakie.Axis(fig[1, 1], xlabel="Distance (cm)", ylabel="Energy (μJ)")
    rax = GLMakie.Axis(fig[1, 1], yaxisposition = :right, ylabel="Conversion efficiency (%)")
    GLMakie.hidespines!(rax)
    GLMakie.hidexdecorations!(rax)
    for i in 1:size(e')[1]
        GLMakie.lines!(ax, z, 1e6*e'[i,:])
    end
    maxe = maximum(1e6*e)
    GLMakie.xlims!(ax, extrema(z)...)
    GLMakie.ylims!(ax, 0, maxe)
    GLMakie.ylims!(rax, 0, 100*maxe/1e6/e0)
    GLMakie.xlims!(rax, extrema(z)...)
    GLMakie.DataInspector()
    fig
end

end