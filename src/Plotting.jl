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
function cmap_white(cmap, N=512, n=8)
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

Create a figure with a grid of `N` subplots with. If `portrait` is `true`, try to lay out
the grid in portrait orientation (taller than wide), otherwise landscape (wider than tall).
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

function stats(output; kwargs...)
    stats = output["stats"]

    pstats = [] # pulse statistics
    haskey(stats, "energy") && push!(pstats, (1e6*stats["energy"], "Energy (μJ)"))
    for (k, v) in pairs(stats)
        startswith(k, "energy_") || continue
        str = "Energy "*replace(k[8:end], "_" => " ")*" (μJ)"
        push!(pstats, (1e6*stats[k], str))
    end
    haskey(stats, "peakpower") && push!(pstats, (1e-9*stats["peakpower"], "Peak power (GW)"))
    haskey(stats, "peakintensity") && push!(
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm\$^2\$)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "min FWHM (fs)"))
    haskey(stats, "fwhm_t_max") && push!(pstats, (1e15*stats["fwhm_t_max"], "max FWHM (fs)"))
    haskey(stats, "fwhm_r") && push!(pstats, (1e6*stats["fwhm_r"], "Radial FWHM (μm)"))

    fstats = [] # fibre/waveguide/propagation statistics
    haskey(stats, "electrondensity") && push!(
        fstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-1}\$)"))
    haskey(stats, "density") && push!(
        fstats, (stats["density"], "Density (cm\$^{-1}\$)"))
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
        # pfig.set_size_inches(8, 8)
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
        # ffig.set_size_inches(8, 8)
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

function should_log10(A, tolfac=10)
    mi = minimum(A; dims=2)
    ma = maximum(A; dims=2)
    any(ma./mi .> 10)
end

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


getEω(output::AbstractOutput, args...) = getEω(makegrid(output), output, args...)

function getIω(ω, Eω, specaxis)
    if specaxis == :f
        specx = ω./2π.*1e-15
        If = abs2.(Eω)*1e15*2π
        return specx, If
    elseif specaxis == :ω
        specx = ω*1e-15
        Iω = abs2.(Eω)*1e15
        return specx, Iω
    elseif specaxis == :λ
        specx = wlfreq.(ω) .* 1e9
        Iλ = @. ω^2/(2π*c) * abs2.(Eω) * 1e-9
        idcs = sortperm(specx)
        cidcs = CartesianIndices(size(Iλ)[2:end])
        return specx[idcs], Iλ[idcs, cidcs]
    else
        error("Unknown specaxis $specaxis")
    end
end

getIω(output::AbstractOutput, specaxis) = getIω(getEω(output)..., specaxis)

function getIω(output::AbstractOutput, specaxis, zslice)
    ω, Eω, zactual = getEω(output, zslice)
    specx, Iω = getIω(ω, Eω, specaxis)
    return specx, Iω, zactual
end

function getEt(grid, output; trange, oversampling=4)
    t = grid.t
    Etout = envelope(grid, t, output["Eω"])
    idcs = @. (t < max(trange...)) & (t > min(trange...))
    cidcs = CartesianIndices(size(Etout)[2:end])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs], factor=oversampling)
    return to, Eto
end

function getEt(grid, output, zslice; trange, oversampling=4)
    t = grid.t
    Etout = envelope(grid, t, output["Eω"])
    idcs = @. (t < max(trange...)) & (t > min(trange...))
    cidcs = CartesianIndices(size(Etout)[2:end-1])
    zidx = nearest_z(output, zslice)
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs, zidx], factor=oversampling)
    return to, Eto, output["z"][zidx]
end

getEt(output::AbstractOutput, args...; kwargs...) = getEt(makegrid(output), output, args...; kwargs...)

envelope(grid::Grid.RealGrid, t, Eω) = Maths.hilbert(FFTW.irfft(Eω, length(t), 1))
envelope(grid::Grid.EnvGrid, t, Eω) = FFTW.ifft(Eω, 1)

nearest_z(output, z::Number) = argmin(abs.(output["z"] .- z))
nearest_z(output, z) = [argmin(abs.(output["z"] .- zi)) for zi in z]

function prop_2D(output, specaxis=:f;
                 λrange=(150e-9, 2000e-9), trange=(-50e-15, 50e-15),
                 dBmin=-60, kwargs...)
    z = output["z"]*1e2
    specx, Iω = getIω(output, specaxis)
    t, Et = getEt(output, trange=trange)
    It = abs2.(Et)

    speclims, speclabel = getspeclims(λrange, specaxis)

    multimode, modes = get_modes(output)

    if multimode
        _prop2D_mm(modes, t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    else
        _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    end    
end

function getspeclims(λrange, specaxis)
    if specaxis == :f
        speclims = (1e-15*c/maximum(λrange), 1e-15*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        speclims = (1e-15*wlfreq(maximum(λrange)), 1e-15*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        speclims = λrange .* 1e9
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end
    return speclims, speclabel
end

function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    pfig, axs = plt.subplots(1, 2, num="Propagation")
    pfig.set_size_inches(12, 4)
    _spec2D_log(axs[1], specx, z, Iω, dBmin, speclabel, speclims; kwargs...)

    _time2D(axs[2], t, z, It, trange; kwargs...)
    pfig.tight_layout()
    return pfig
end

function _prop2D_mm(modes, t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    pfigs = []
    for mi in 1:length(modes)
        pfig, axs = plt.subplots(1, 2, num="Propagation ($(modes[mi]))")
        pfig.set_size_inches(12, 4)
        _spec2D_log(axs[1], specx, z, Iω[:, mi, :], dBmin, speclabel, speclims; kwargs...)

        _time2D(axs[2], t, z, It[:, mi, :], trange; kwargs...)
        push!(pfigs, pfig)
    end

    pfig, axs = plt.subplots(1, 2, num="Propagation (all modes)")
    pfig.set_size_inches(12, 4)
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    _spec2D_log(axs[1], specx, z, Iωall, dBmin, speclabel, speclims; kwargs...)

    Itall = dropdims(sum(It, dims=2), dims=2)
    _time2D(axs[2], t, z, Itall, trange; kwargs...)
    pfig.tight_layout()
    push!(pfigs, pfig)

    return pfigs
end

function _spec2D_log(ax, specx, z, I, dBmin, speclabel, speclims; kwargs...)
    im = ax.pcolormesh(specx, z, 10*log10.(Maths.normbymax(transpose(I))); kwargs...)
    im.set_clim(dBmin, 0)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("SED (dB)")
    ax.set_ylabel("Distance (cm)")
    ax.set_xlabel(speclabel)
    ax.set_xlim(speclims...)
end

function _time2D(ax, t, z, I, trange; kwargs...)
    im = ax.pcolormesh(t*1e15, z, transpose(I); kwargs...)
    plt.colorbar(im, ax=ax)
    ax.set_xlim(trange.*1e15)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (cm)")
end

function time_1D(output, zslice; modeidx=nothing, trange=(-50e-15, 50e-15), kwargs...)
    t, Et, zactual = getEt(output, zslice, trange=trange)
    It = abs2.(Et)
    multimode, modes = get_modes(output)
    if multimode
        modeidx = isnothing(modeidx) ? (1:size(It, 2)) : modeidx 
        It = It[:, modeidx, :]
        modes = modes[modeidx]
    end

    sfig = plt.figure()
    if multimode && length(modeidx) > 1
        _plot_slice_mm(plt.gca(), t*1e15, 1e-9*It, zslice, modes; kwargs...)
        plt.legend(frameon=false)
    else
        plt.plot(t*1e15, 1e-9*It; kwargs...)
        plt.legend(string.(zactual.*100).*" cm ($modes)", frameon=false)
    end
    plt.xlabel("Time (fs)")
    plt.ylabel("Power (GW)")
    plt.ylim(ymin=0)
    sfig.set_size_inches(8.5, 5)
    sfig.tight_layout()
end

function spec_1D(output, zslice, specaxis=:λ; modeidx=nothing, λrange=(150e-9, 1200e-9),
                 log10=true, log10min=1e-6,
                 kwargs...)
    specx, Iω, zactual = getIω(output, specaxis, zslice)
    speclims, speclabel = getspeclims(λrange, specaxis)
    multimode, modes = get_modes(output)
    if multimode
        modeidx = isnothing(modeidx) ? (1:size(Iω, 2)) : modeidx 
        Iω = Iω[:, modeidx, :]
        modes = modes[modeidx]
    end

    size(Iω, 2) > 6 && error("spec_1D currently only supports 6 modes or fewer.")
    sfig = plt.figure()
    if multimode && length(modeidx) > 1
        _plot_slice_mm(plt.gca(), specx, Iω, zslice, modes, log10; kwargs...)
        plt.legend(frameon=false)
    else
        (log10 ? plt.semilogy : plt.plot)(specx, Iω; kwargs...)
        plt.legend(string.(zactual.*100).*" cm ($modes)", frameon=false)
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

function _plot_slice_mm(ax, x, y, z, modes, log10=false; kwargs...)
    pfun = (log10 ? ax.semilogy : ax.plot)
    for sidx = 1:size(y, 3)
        zs = @sprintf("%.2f cm", z[sidx]*100)
        line = pfun(x, y[:, 1, sidx]; label="$zs cm ($(modes[1]))", kwargs...)[1]
        for midx = 2:size(y, 2)
            pfun(x, y[:, midx, sidx], linestyle=dashes[midx], color=line.get_color(),
                 label="$zs cm ($(modes[midx]))"; kwargs...)
        end
    end
end


end