module Plotting
import Luna: Grid, Maths, PhysData
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
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
    makegrid(output)

Create an `AbstractGrid` from the `"grid"` dictionary saved in `output`.
"""
function makegrid(output)
    if output["simulation_type"]["field"] == "field-resolved"
        Grid.from_dict(Grid.RealGrid, output["grid"])
    else
        Grid.from_dict(Grid.EnvGrid, output["grid"])
    end
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

"""
    getEω(output[, zslice])

Get frequency-domain modal field from `output` with correct normalisation (i.e. 
`abs2.(Eω)`` gives angular-frequency spectral energy density in J/(rad/s)).

If `zslice` (number or array) is given, return only the slices of `Eω` closest to the given
distances.
"""
getEω(output::AbstractOutput, args...) = getEω(makegrid(output), output, args...)

function getEω(grid::Grid.RealGrid, output)
    ω = grid.ω[grid.sidx]
    δt = grid.t[2] - grid.t[1]
    Eω = output["Eω"][grid.sidx, CartesianIndices(size(output["Eω"])[2:end])]
    return ω, Eω*δt*2/sqrt(2π)
end

function getEω(grid::Grid.EnvGrid, output)
    δt = grid.t[2] - grid.t[1]
    idcs = FFTW.fftshift(grid.sidx)
    Eωs = FFTW.fftshift(output["Eω"], 1)
    ω = FFTW.fftshift(grid.ω)[idcs]
    Eω = Eωs[idcs, CartesianIndices(size(output["Eω"])[2:end])]
    return ω, Eω*δt/sqrt(2π)
end

function getEω(grid, output, zslice)
    ω, Eω = getEω(grid, output)
    cidcs = CartesianIndices(size(Eω)[1:end-1])
    zidx = nearest_z(output, zslice)
    return ω, Eω[cidcs, zidx], output["z"][zidx]
end

"""
    getIω(ω, Eω, specaxis)

Get spectral energy density and x-axis given a frequency array `ω` and frequency-domain field
`Eω`, assumed to be correctly normalised (see [`getEω`](@ref)). `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m
"""
function getIω(ω, Eω, specaxis)
    if specaxis == :f
        specx = ω./2π
        If = abs2.(Eω)*2π
        return specx, If
    elseif specaxis == :ω
        specx = ω
        Iω = abs2.(Eω)
        return specx, Iω
    elseif specaxis == :λ
        specx = wlfreq.(ω)
        Iλ = @. ω^2/(2π*c) * abs2.(Eω)
        idcs = sortperm(specx)
        cidcs = CartesianIndices(size(Iλ)[2:end])
        return specx[idcs], Iλ[idcs, cidcs]
    else
        error("Unknown specaxis $specaxis")
    end
end

"""
    getIω(output, specaxis[, zslice])

Calculate the correctly normalised frequency-domain field and convert it to spectral
energy density on x-axis `specaxis` (`:f`, `:ω`, or `:λ`). If `zslice` is given,
returs only the slices of `Eω` closest to the given distances. `zslice` can be a single
number or an array. `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m
"""
getIω(output::AbstractOutput, specaxis) = getIω(getEω(output)..., specaxis)

function getIω(output::AbstractOutput, specaxis, zslice)
    ω, Eω, zactual = getEω(output, zslice)
    specx, Iω = getIω(ω, Eω, specaxis)
    return specx, Iω, zactual
end

"""
    getEt(output[, zslice])

Get the envelope time-domain electric field (including the carrier wave) from the `output`.
If `zslice` is given, returs only the slices of `Eω` closest to the given distances. `zslice`
can be a single number or an array.
"""
getEt(output::AbstractOutput, args...; kwargs...) = getEt(makegrid(output), output, args...; kwargs...)

function getEt(grid, output; trange=nothing, oversampling=4, bandpass=nothing)
    t = grid.t
    Eω = window_maybe(grid.ω, output["Eω"], bandpass)
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    idcs = @. (t < max(trange...)) & (t > min(trange...))
    cidcs = CartesianIndices(size(Etout)[2:end])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs], factor=oversampling)
    return to, Eto
end

function getEt(grid, output, zslice; trange=nothing, oversampling=4, bandpass=nothing)
    t = grid.t
    Eω = window_maybe(grid.ω, output["Eω"], bandpass)
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    cidcs = CartesianIndices(size(Etout)[2:end-1])
    zidx = nearest_z(output, zslice)
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs, zidx], factor=oversampling)
    return to, Eto, output["z"][zidx]
end

window_maybe(ω, Eω, ::Nothing) = Eω
window_maybe(ω, Eω, win::NTuple{4, Number}) = Eω.*Maths.planck_taper(ω, sort(wlfreq.(collect(win)))...)
function window_maybe(ω, Eω, win::NTuple{2, Number})
    δω = abs(ω[2] - ω[1])
    w = 100*δω # magic number
    ωmin, ωmax = sort(wlfreq.(collect(win)))
    Eω.*Maths.planck_taper(ω, ωmin-w, ωmin, ωmax, ωmax+w)
end
window_maybe(ω, Eω, window) = Eω.*window

window_str(::Nothing) = ""
window_str(win::NTuple{4, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win[2:3]...)
window_str(win::NTuple{2, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win...)
window_str(window) = "custom bandpass"


"""
    envelope(grid, Eω)

Get the envelope electric field including the carrier wave from the frequency-domain field
`Eω` sampled on `grid`.
"""
envelope(grid::Grid.RealGrid, Eω) = Maths.hilbert(FFTW.irfft(Eω, length(grid.t), 1))
envelope(grid::Grid.EnvGrid, Eω) = FFTW.ifft(Eω, 1) .* exp.(im.*grid.ω0.*grid.t)

"""
    nearest_z(output, z)

Return the index of saved z-position(s) closest to the position(s) `z`. Output is always
an array, even if `z` is a number.
"""
nearest_z(output, z::Number) = [argmin(abs.(output["z"] .- z))]
nearest_z(output, z) = [argmin(abs.(output["z"] .- zi)) for zi in z]

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
function time_1D(output, zslice;
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
        plt.legend(frameon=false)
    else
        plt.plot(t*1e15, yfac*yt; kwargs...)
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        if multimode
            plt.legend(zs.*" ($modestrs)", frameon=false)
        else
            plt.legend(zs, frameon=false)
        end
    end
    plt.xlabel("Time (fs)")
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
    oom = min(floor(Int, log10(Pmax)/3), 5) # maximum unit is PW
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
function spec_1D(output, zslice, specaxis=:λ; modes=nothing, λrange=(150e-9, 1200e-9),
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
        plt.legend(frameon=false)
    else
        (log10 ? plt.semilogy : plt.plot)(specx, Iω; kwargs...)
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        if multimode
            plt.legend(zs.*" ($modestrs)", frameon=false)
        else
            plt.legend(zs, frameon=false)
        end
    end
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

function _plot_slice_mm(ax, x, y, z, modestrs, log10=false; kwargs...)
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

spectrogram(output, args...; kwargs...) = spectrogram(makegrid(output), output, args...; kwargs...)

function spectrogram(grid::Grid.AbstractGrid, output, zslice, specaxis=:λ;
                     trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
                     kwargs...)
    t, Et, zactual = getEt(output, zslice, oversampling=1)
    Et = Et[:, 1]
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    specy, Ig = getIω(ω, g, specaxis)
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

end