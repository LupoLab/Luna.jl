module Plotting
import Luna: Maths
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

function stats(output)
    stats = output["stats"]

    pstats = []
    haskey(stats, "energy") && push!(pstats, (1e6stats["energy"], "Energy (μJ)"))
    haskey(stats, "peakpower") && push!(pstats, (1e-9*stats["peakpower"], "Peak power (GW)"))
    haskey(stats, "peakintensity") && push!(
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm\$^2\$)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "FWHM (fs)"))
    haskey(stats, "electrondensity") && push!(
        pstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-1}\$"))

    z = output["stats"]["z"]*1e2

    ffig = plt.figure("Pulse stats")
    ffig.set_size_inches(8, 12)
    Npl = length(pstats)
    for n in 1:Npl
        plt.subplot(Npl, 1, n)
        s = pstats[n]
        plt.plot(z, s[1])
        plt.xlabel("Distance (cm)")
        plt.ylabel(s[2])
    end
    ffig.tight_layout()
end

function prop_2D(output; trange=(-50e-15, 50e-15))
    ω = output["grid"]["ω"]
    t = output["grid"]["t"]

    z = output["z"]*1e2
    Eout = output["Eω"]

    Etout = FFTW.irfft(Eout, length(t), 1)

    Ilog = log10.(Maths.normbymax(abs2.(Eout)))

    idcs = @. (t < max(trange...)) & (t > min(trange...))
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=4)
    It = abs2.(Maths.hilbert(Eto))

    Et = Maths.hilbert(Etout)


    pfig = plt.figure("Propagation")
    pfig.set_size_inches(12, 4)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(ω./2π.*1e-15, z, transpose(Ilog))
    plt.clim(-6, 0)
    plt.colorbar()
    plt.ylabel("Distance (cm)")
    plt.xlabel("Frequency (PHz)")

    plt.subplot(1, 2, 2)
    plt.pcolormesh(to*1e15, z, transpose(It))
    plt.colorbar()
    plt.xlim(trange.*1e15)
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance (cm)")
    pfig.tight_layout()
end



end