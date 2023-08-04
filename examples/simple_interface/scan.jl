using Luna
import PyPlot: plt

# Fixed parameters:
a = 125e-6
flength = 3
gas = :He

λ0 = 800e-9
τfwhm = 10e-15

λlims = (100e-9, 4e-6)
trange = 400e-15

# Scan dimensions:
energies = collect(range(50e-6, 200e-6; length=16))
pressures = collect(0.6:0.4:1.4)

# scan variables can be passed directly to the Scan constructor...
scan = Scan("pressure_energy_example"; energy=energies)
#...or added later
addvariable!(scan, :pressure, pressures)

# @__DIR__ gives the directory of the current file
outputdir = joinpath(@__DIR__, "scanoutput")

runscan(scan) do scanidx, energy, pressure
    prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy,
                   λlims, trange, scan, scanidx, filepath=outputdir)
end

# Use Processing.scanproc to apply a processing function to each output file and
# collect the result.
λ, Iλ, zstat, edens, max_peakpower = Processing.scanproc(outputdir) do output
    λ, Iλ = Processing.getIω(output, :λ)
    zstat = Processing.VarLength(output["stats"]["z"])
    edens = Processing.VarLength(output["stats"]["electrondensity"])
    max_peakpower = maximum(output["stats"]["peakpower"])
    Processing.Common(λ), Iλ[:, end], zstat, edens, max_peakpower
end

fig, axs = plt.subplots(1, length(pressures))
fig.set_size_inches(8, 2)
for (pidx, pressure) in enumerate(pressures)
    ax = axs[pidx]
    global img = ax.pcolormesh(λ*1e9, energies*1e6, 10*Maths.log10_norm(Iλ[:, :, pidx])')
    img.set_clim(-40, 0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Energy (μJ)")
    ax.set_title("Pressure: $pressure bar")
    ax.set_xlim(100, 1200)
end
plt.colorbar(img, ax=axs, label="Energy density (dB)")

fig, axs = plt.subplots(1, length(pressures))
fig.set_size_inches(8, 2)
cols = Plotting.cmap_colours(length(energies))
edmax = maximum(maximum.(edens))
for (pidx, pressure) in enumerate(pressures)
    ax = axs[pidx]
    for eidx in eachindex(energies)
        ax.plot(zstat[eidx, pidx], 1e-6edens[eidx, pidx], color=cols[eidx],
                linewidth=1, alpha=0.8)
    end
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Electron density (cm\$^{-3}\$)")
    ax.set_ylim(0, 1.1e-6edmax)
    ax.set_xlim(0, flength)
    ax.set_title("Pressure: $pressure bar")
end

fig = plt.figure()
for (pidx, pressure) in enumerate(pressures)
    plt.plot(energies*1e6, max_peakpower[:, pidx], label="$pressure bar")
end
plt.xlim(1e6.*extrema(energies))
plt.ylim_ymin=0
plt.xlabel("Energy (μJ)")
plt.ylabel("Maximum peak power (W)")
plt.legend()
