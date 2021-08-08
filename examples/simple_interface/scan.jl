using Luna
import PyPlot: plt

# Fixed parameters:
a = 125e-6
flength = 3
gas = :HeJ

λ0 = 800e-9
τfwhm = 10e-15

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
                   trange=400e-15, scan, scanidx, filedir=outputdir)
end

# Use Processing.scanproc to apply a processing function to each output file and
# collect the result.
λ, Iλ, energyout = Processing.scanproc(outputdir) do output
    λ, Iλ = Processing.getIω(output, :λ)
    energyout = Processing.VarLength(output["stats"]["energy"])
    Processing.Common(λ), Iλ[:, end], energyout
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