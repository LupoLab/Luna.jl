# # Plotting examples
#
# This page demonstrates Luna's built-in plotting functions using the PythonPlot backend.
# The same functions work identically with CairoMakie, GLMakie, WGLMakie, and PyPlot.
#
# ## Setup
#
# First, we run a simple hollow capillary fibre simulation: dispersive wave emission
# in the deep UV from a few-cycle 800 nm pump pulse.

using Luna, PythonPlot

const plt = PythonPlot.pyplot
plt.rcParams["savefig.dpi"] = 600

radius = 125e-6
flength = 2.0
gas = :He
pressure = 1.0

λ0 = 800e-9
τfwhm = 10e-15
energy = 120e-6

output = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                        modes=4, trange=400e-15, λlims=(150e-9, 4e-6))

# ## 2D propagation plots
#
# [`Plotting.prop_2D`](@ref) creates side-by-side spectral and temporal heatmaps.
# The first positional argument after `output` selects the spectral axis:
# `:λ` for wavelength, `:f` for frequency (default), or `:ω` for angular frequency.
#
# Since this is a multimode simulation, `prop_2D` returns a vector of figures
# (one for each mode and one for the sum).

Plotting.prop_2D(output, :λ; λrange=(150e-9, 1000e-9), trange=(-20e-15, 20e-15), dBmin=-30)
for (i, fig) in enumerate(ans) fig.savefig("prop_2D_$i.svg") end; # hide

# **Individual modes**
#
# ![Mode 1](prop_2D_1.svg) ![Mode 2](prop_2D_2.svg)
#
# ![Mode 3](prop_2D_3.svg) ![Mode 4](prop_2D_4.svg)
#
# **Sum of all modes**
#
# ![Sum of all modes](prop_2D_5.svg)

# For multimode simulations, use the `modes` keyword to select which modes to plot.
# `:sum` plots the coherent sum of all modes:

Plotting.prop_2D(output, :λ; modes=:sum,
                 λrange=(150e-9, 1000e-9), trange=(-20e-15, 20e-15), dBmin=-30)
ans[1].savefig("prop_2D_sum.svg"); # hide

# ![Sum of all modes](prop_2D_sum.svg)

# ## Time-domain line plots
#
# [`Plotting.time_1D`](@ref) plots time-domain slices at one or more propagation distances.
# The legend automatically includes the FWHM duration.

Plotting.time_1D(output; modes=:sum)
ans.savefig("time_1D_sum.svg"); # hide

# ![Time-domain (sum)](time_1D_sum.svg)

# Use `bandpass` to isolate a spectral window in the time domain:

Plotting.time_1D(output; modes=1, bandpass=(220e-9, 270e-9), trange=(-50e-15, 50e-15))
ans.savefig("time_1D_bp.svg"); # hide

# ![Time-domain (bandpassed)](time_1D_bp.svg)

# ## Spectral line plots
#
# [`Plotting.spec_1D`](@ref) plots spectral slices. By default it uses a logarithmic y-axis.

Plotting.spec_1D(output; modes=:sum, λrange=(150e-9, 1000e-9))
ans.savefig("spec_1D.svg"); # hide

# ![Spectral-domain (sum)](spec_1D.svg)

# Without the `modes` keyword, all modes are plotted individually: each mode gets a
# different line style.

Plotting.spec_1D(output; λrange=(150e-9, 1000e-9))
ans.savefig("spec_1D_modes.svg"); # hide

# ![Spectral-domain (all modes)](spec_1D_modes.svg)

# ## Spectrograms
#
# [`Plotting.spectrogram`](@ref) creates a time-frequency spectrogram using the Gabor transform.
# The `fw` parameter sets the gate function width.

Plotting.spectrogram(output, flength; trange=(-20e-15, 30e-15),
                     λrange=(150e-9, 1000e-9), N=256, fw=3e-15)
ans.savefig("spectrogram.svg"); # hide

# ![Spectrogram](spectrogram.svg)

# !!! note
#     With the Makie backend, you can also create a 3D surface spectrogram using `surface3d=true`.

# ## Statistics
#
# [`Plotting.stats`](@ref) automatically plots all available propagation statistics
# (energy, peak power, FWHM, electron density, etc.).

Plotting.stats(output)
for (i, fig) in enumerate(ans) fig.savefig("stats_$i.svg") end; # hide

# ![Pulse statistics](stats_1.svg)
#
# ![Propagation statistics](stats_2.svg)

# ## Energy evolution
#
# [`Plotting.energy`](@ref) shows energy vs propagation distance with a secondary axis
# for conversion efficiency.

Plotting.energy(output)
ans.savefig("energy.svg"); # hide

# ![Energy evolution](energy.svg)

# The `modes` and `bandpass` keywords work here too — for example, the total energy in
# the ultraviolet dispersive wave summed over all modes:

Plotting.energy(output; modes=:sum, bandpass=(220e-9, 270e-9))
ans.savefig("energy_sum_bp.svg"); # hide

# ![Energy evolution (sum, bandpassed)](energy_sum_bp.svg)
