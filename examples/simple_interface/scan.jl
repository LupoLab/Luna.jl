using Luna, CairoMakie

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

# Plot output spectra as a function of energy for each pressure
fig = Figure(size=(1200, 300))
for (pidx, pressure) in enumerate(pressures)
    ax = Axis(fig[1, pidx];
        xlabel="Wavelength (nm)", ylabel="Energy (μJ)",
        title="Pressure: $pressure bar")
    data = 10 * Maths.log10_norm(Iλ[:, :, pidx])
    hm = heatmap!(ax, λ * 1e9, energies * 1e6, data;
        colorrange=(-40, 0), colormap=:viridis, rasterize=3)
    xlims!(ax, 100, 1200)
    if pidx == length(pressures)
        Colorbar(fig[1, pidx+1], hm; label="Energy density (dB)")
    end
end
fig

# Plot electron density evolution for each pressure
cols = Plotting.cmap_colours(length(energies))
edmax = maximum(maximum.(edens))
fig2 = Figure(size=(1200, 300))
for (pidx, pressure) in enumerate(pressures)
    ax = Axis(fig2[1, pidx];
        xlabel="Distance (m)", ylabel="Electron density (cm⁻³)",
        title="Pressure: $pressure bar")
    for eidx in eachindex(energies)
        lines!(ax, zstat[eidx, pidx], 1e-6 * edens[eidx, pidx];
            color=cols[eidx], linewidth=1)
    end
    ylims!(ax, 0, 1.1e-6 * edmax)
    xlims!(ax, 0, flength)
end
fig2

# Plot maximum peak power vs energy for each pressure
fig3 = Figure()
ax = Axis(fig3[1, 1]; xlabel="Energy (μJ)", ylabel="Maximum peak power (W)")
for (pidx, pressure) in enumerate(pressures)
    lines!(ax, energies * 1e6, max_peakpower[:, pidx]; label="$pressure bar")
end
xlims!(ax, 1e6 .* extrema(energies)...)
ylims!(ax, low=0)
axislegend(ax)
fig3
