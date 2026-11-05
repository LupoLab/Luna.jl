#=
Generate all figures used in the README and documentation.
Run from the Luna root directory:

    julia --project scripts/generate_figures.jl

This requires PythonPlot to be installed. All figures are saved as SVG
in the assets/ and docs/src/assets/ directories.

To regenerate figures after changing the plotting code or simulation parameters,
simply re-run this script.
=#

using Luna, PythonPlot
const plt = PythonPlot.pyplot

plt.rcParams["savefig.dpi"] = 600

assetsdir = joinpath(@__DIR__, "..", "assets")
docassetsdir = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assetsdir)
mkpath(docassetsdir)

# ============================================================================
# README figures — mode-averaged HCF propagation
# ============================================================================
println("Running mode-averaged simulation...")
output = prop_capillary(125e-6, 3, :He, 1;
    λ0=800e-9, energy=120e-6, τfwhm=10e-15,
    λlims=(150e-9, 4e-6), trange=1e-12)

# Figure 1: prop_2D
println("  Generating prop_2D figure...")
fig = Plotting.prop_2D(output)
fig.savefig(joinpath(assetsdir, "readme_modeAvgProp.svg"))

# Figure 2: spec_1D at three z-positions
println("  Generating spec_1D figure...")
fig = Plotting.spec_1D(output, [0, 1.5, 3]; log10=true)
fig.savefig(joinpath(assetsdir, "readme_modeAvgSpec.svg"))

# Figure 3: time_1D with bandpass filter (UV pulse)
println("  Generating time_1D figure...")
fig = Plotting.time_1D(output, [2, 2.5, 3];
    trange=(-10e-15, 30e-15), bandpass=(180e-9, 220e-9))
fig.savefig(joinpath(assetsdir, "readme_modeAvgTime.svg"))

# ============================================================================
# README figures — multi-mode HCF propagation
# ============================================================================
println("Running multi-mode simulation...")
output_multimode = prop_capillary(125e-6, 3, :He, 1;
    λ0=800e-9, modes=4, energy=120e-6, τfwhm=10e-15,
    trange=400e-15, λlims=(150e-9, 4e-6))

# Figure 4: spec_1D with modes summed
println("  Generating multimode spec_1D figure...")
fig = Plotting.spec_1D(output_multimode; log10=true, modes=:sum)
fig.savefig(joinpath(assetsdir, "readme_multiModeSpec.svg"))

# ============================================================================
# README figures — GNLSE supercontinuum generation
# ============================================================================
println("Running GNLSE simulation...")
γ = 0.11
flength = 15e-2
βs = [0.0, 0.0, -1.1830e-26, 8.1038e-41, -9.5205e-56, 2.0737e-70,
      -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129, -1.7140e-144]
output_gnlse = prop_gnlse(γ, flength, βs;
    λ0=835e-9, τfwhm=50e-15, power=10e3, pulseshape=:sech,
    λlims=(400e-9, 2400e-9), trange=12.5e-12)

# Figure 5: GNLSE prop_2D on wavelength axis
println("  Generating GNLSE prop_2D figure...")
fig = Plotting.prop_2D(output_gnlse, :λ;
    dBmin=-40.0, λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))
fig.savefig(joinpath(assetsdir, "readme_gnlse_scg.svg"))

# ============================================================================
# docs/src/scans.md figure — pressure-energy scan
# ============================================================================
println("Running pressure-energy scan...")
a = 125e-6
flength_scan = 3
gas = :He
λ0 = 800e-9
τfwhm = 10e-15
λlims = (100e-9, 4e-6)
trange_scan = 400e-15

energies = collect(range(50e-6, 200e-6; length=16))
pressures = collect(0.6:0.4:1.4)

scan = Scan("figure_generation_scan"; energy=energies)
addvariable!(scan, :pressure, pressures)

outputdir = mktempdir()

runscan(scan) do scanidx, energy, pressure
    prop_capillary(a, flength_scan, gas, pressure;
        λ0, τfwhm, energy, λlims, trange=trange_scan,
        scan, scanidx, filepath=outputdir)
end

λ, Iλ = Processing.scanproc(outputdir) do output
    λ, Iλ = Processing.getIω(output, :λ)
    Processing.Common(λ), Iλ[:, end]
end

println("  Generating scan spectrum figure...")
npress = length(pressures)
fig, axs = plt.subplots(1, npress, figsize=(4*npress, 3), layout="constrained")
im = nothing
for (pidx, pressure) in enumerate(pressures)
    ax = npress == 1 ? axs : axs[pidx-1]
    data = 10 * Maths.log10_norm(Iλ[:, :, pidx])
    global im = ax.pcolormesh(λ * 1e9, energies * 1e6, data';
        vmin=-40, vmax=0, cmap="viridis", rasterized=true)
    ax.set_xlim(100, 1200)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Energy (μJ)")
    ax.set_title("Pressure: $pressure bar")
end
fig.colorbar(im, ax=axs, label="Energy density (dB)")
fig.savefig(joinpath(docassetsdir, "scan_spectrum.svg"))

# Clean up temporary scan output
rm(outputdir; recursive=true)

println("Done! All figures saved.")
println("  README figures: $assetsdir/*.svg")
println("  Doc figures:    $docassetsdir/*.svg")
