module Plotting
import Luna: Grid, Maths, PhysData
import Luna.PhysData: wlfreq, c, ε_0
import PyPlot: ColorMap, plt, pygui
import FFTW

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
function subplotgrid(N, portrait=true; title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N/cols)
    portrait && ((rows, cols) = (cols, rows))
    fig, axs = plt.subplots(rows, cols, num=title)
    if cols*rows > N
        for axi in permutedims(axs, (2, 1))[N+1:end]
            axi.remove()
        end
    end
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
    haskey(stats, "energy") && push!(pstats, (1e6stats["energy"], "Energy (μJ)"))
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

    z = output["stats"]["z"]*1e2

    multimode, modes = get_modes(output)

    Npl = length(pstats)
    if Npl > 0
        pfig, axs = subplotgrid(Npl, title="Pulse stats")
        pfig.set_size_inches(8, 8)
        for n in 1:Npl
            ax = axs[n]
            data, label = pstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes)
        end
        pfig.tight_layout()
    end
    
    Npl = length(fstats)
    if Npl > 0
        ffig, axs = subplotgrid(Npl, title="Other stats")
        ffig.set_size_inches(8, 8)
        for n in 1:Npl
            ax = axs[n]
            data, label = fstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes)
        end
        ffig.tight_layout()
    end
end

function getEω(grid::Grid.RealGrid, output)
    ω = grid.ω[grid.sidx]
    Eω = output["Eω"][grid.sidx, CartesianIndices(size(output["Eω"])[2:end])]
    return ω, Eω
end

function getEω(grid::Grid.EnvGrid, output)
    idcs = FFTW.fftshift(grid.sidx)
    Eωs = FFTW.fftshift(output["Eω"], 1)
    ω = grid.ω[idcs]
    Eω = Eωs[idcs, CartesianIndices(size(output["Eω"])[2:end])]
    return ω, Eω
end

getEω(output) = getEω(makegrid(output), output)


function getEt(grid::Grid.RealGrid, output; trange, oversampling=4)
    t = grid.t
    Etout = FFTW.irfft(output["Eω"], length(t), 1)
    idcs = @. (t < max(trange...)) & (t > min(trange...))
    cidcs = CartesianIndices(size(Etout)[2:end])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs], factor=oversampling)
    Et = Maths.hilbert(Eto)
    return to, Et
end

function getEt(grid::Grid.EnvGrid, output; trange, oversampling=4)
    t = grid.t
    Etout = FFTW.ifft(output["Eω"], length(t), 1)
    idcs = @. (t < max(trange...)) & (t > min(trange...))
    cidcs = CartesianIndices(size(Etout)[2:end])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs], factor=oversampling)
    return to, Eto
end

getEt(output; kwargs...) = getEt(makegrid(output), output; kwargs...)

function prop_2D(output, specaxis=:f;
                 λrange=(150e-9, 2000e-9), trange=(-50e-15, 50e-15),
                 dBmin=-60, kwargs...)
    z = output["z"]*1e2
    ω, Eω = getEω(output)
    t, Et = getEt(output, trange=trange)
    It = abs2.(Et)

    if specaxis == :f
        specx = ω./2π.*1e-15
        Iω = Maths.normbymax(abs2.(Eω))
        speclims = (1e-15*c/maximum(λrange), 1e-15*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        specx = ω*1e-15
        Iω = Maths.normbymax(abs2.(Eω))
        speclims = (1e-15*wlfreq(maximum(λrange)), 1e-15*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        specx = wlfreq.(ω) .* 1e9
        Iω = Maths.normbymax(ω.^2 .* abs2.(Eω))
        speclims = λrange .* 1e9
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end

    multimode, modes = get_modes(output)

    if multimode
        _prop2D_mm(modes, t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    else
        _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    end    
end

function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    pfig = plt.figure("Propagation")
    pfig.set_size_inches(12, 4)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(specx, z, 10*log10.(transpose(Iω)); kwargs...)
    plt.clim(dBmin, 0)
    cb = plt.colorbar()
    cb.set_label("SED (dB)")
    plt.ylabel("Distance (cm)")
    plt.xlabel(speclabel)
    plt.xlim(speclims...)

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t*1e15, z, transpose(It); kwargs...)
    plt.colorbar()
    plt.xlim(trange.*1e15)
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance (cm)")
    pfig.tight_layout()
    return pfig
end

function _prop2D_mm(modes, t, z, specx, It, Iω, speclabel, speclims, trange, dBmin; kwargs...)
    pfigs = []
    for mi in 1:length(modes)
        pfig = plt.figure("Propagation ($(modes[mi]))")
        pfig.set_size_inches(12, 4)
        plt.subplot(1, 2, 1)
        plt.pcolormesh(specx, z, 10*log10.(transpose(Iω[:, mi, :])); kwargs...)
        plt.clim(dBmin, 0)
        cb = plt.colorbar()
        cb.set_label("SED (dB)")
        plt.ylabel("Distance (cm)")
        plt.xlabel(speclabel)
        plt.xlim(speclims...)

        plt.subplot(1, 2, 2)
        plt.pcolormesh(t*1e15, z, transpose(It[:, mi, :]); kwargs...)
        plt.colorbar()
        plt.xlim(trange.*1e15)
        plt.xlabel("Time (fs)")
        plt.ylabel("Distance (cm)")
        pfig.tight_layout()
        push!(pfigs, pfig)
    end

    pfig = plt.figure("Propagation (all modes)")
    pfig.set_size_inches(12, 4)
    plt.subplot(1, 2, 1)
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    plt.pcolormesh(specx, z, 10*log10.(transpose(Iωall)); kwargs...)
    plt.clim(dBmin, 0)
    cb = plt.colorbar()
    cb.set_label("SED (dB)")
    plt.ylabel("Distance (cm)")
    plt.xlabel(speclabel)
    plt.xlim(speclims...)

    plt.subplot(1, 2, 2)
    Itall = dropdims(sum(It, dims=2), dims=2)
    plt.pcolormesh(t*1e15, z, transpose(Itall); kwargs...)
    plt.colorbar()
    plt.xlim(trange.*1e15)
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance (cm)")
    pfig.tight_layout()
    push!(pfigs, pfig)

    return pfigs
end



end