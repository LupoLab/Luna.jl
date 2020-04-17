module Plotting
import Luna: Grid, Maths
import Luna.PhysData: wlfreq, c, ε_0
import PyPlot: ColorMap, plt, pygui
import FFTW

function cmap_white(cmap, N=512, n=8)
    vals = collect(range(0, 1, length=n))
    vals_i = collect(range(0, 1, length=N))
    cm = ColorMap(cmap)
    clist = cm(vals)
    clist[1, :] = [1, 1, 1, 1]
    clist_i = Array{Float64}(undef, (N, 4))
    for ii in 1:4
        clist_i[:, ii] .= Maths.CSpline(vals, clist[:, ii]).(vals_i)
    end
    ColorMap(clist_i)
end

function subplotgrid(N, kwargs...)
    rows = ceil(Int, sqrt(N))
    cols = ceil(Int, N/rows)
    fig, axs = plt.subplots(rows, cols, kwargs...)
    for axi in permutedims(axs, (2, 1))[N+1:end]
        axi.remove()
    end
    fig, axs
end

function makegrid(output)
    if output["simulation_type"]["field"] == "field-resolved"
        Grid.from_dict(Grid.RealGrid, output["grid"])
    else
        Grid.from_dict(Grid.EnvGrid, output["grid"])
    end
end

function stats(output)
    stats = output["stats"]

    pstats = []
    haskey(stats, "energy") && push!(pstats, (1e6stats["energy"], "Energy (μJ)"))
    haskey(stats, "peakpower") && push!(pstats, (1e-9*stats["peakpower"], "Peak power (GW)"))
    haskey(stats, "peakintensity") && push!(
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm\$^2\$)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "FWHM (fs)"))
    haskey(stats, "electrondensity") && push!(
        pstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-1}\$)"))

    z = output["stats"]["z"]*1e2

    Npl = length(pstats)
    ffig, axs = subplotgrid(Npl)
    ffig.set_label("Pulse stats")
    ffig.set_size_inches(8, 8)
    for n in 1:Npl
        ax = axs[n]
        s = pstats[n]
        ax.plot(z, s[1])
        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel(s[2])
    end
    ffig.tight_layout()
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
                 dBmin=-60)
    z = output["z"]*1e2
    ω, Eω = getEω(output)
    t, Et = getEt(output, trange=trange)
    It = abs2.(Et)

    if specaxis == :f
        specx = ω./2π.*1e-15
        Ilog = log10.(Maths.normbymax(abs2.(Eω)))
        speclims = (1e-15*c/maximum(λrange), 1e-15*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        specx = ω*1e-15
        Ilog = log10.(Maths.normbymax(abs2.(Eω)))
        speclims = (1e-15*wlfreq(maximum(λrange)), 1e-15*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        specx = wlfreq.(ω) .* 1e9
        Ilog = log10.(Maths.normbymax(ω.^2 .* abs2.(Eω)))
        speclims = λrange .* 1e9
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end

    pfig = plt.figure("Propagation")
    pfig.set_size_inches(12, 4)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(specx, z, 10*transpose(Ilog))
    plt.clim(dBmin, 0)
    cb = plt.colorbar()
    cb.set_label("SED (dB)")
    plt.ylabel("Distance (cm)")
    plt.xlabel(speclabel)
    plt.xlim(speclims...)

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t*1e15, z, transpose(It))
    plt.colorbar()
    plt.xlim(trange.*1e15)
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance (cm)")
    pfig.tight_layout()
end



end