# Plotting

Luna provides built-in plotting functions through a backend extension system. Three backends are supported: **PythonPlot** (recommended, full matplotlib capability), **Makie** (native Julia with advanced features), and **PyPlot** (legacy, kept for backward compatibility).

## Choosing a backend

### PythonPlot (recommended)
```julia
using Luna, PythonPlot
```
[PythonPlot](https://github.com/JuliaPy/PythonPlot.jl) uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) to provide the full capability of [matplotlib](https://matplotlib.org/) from Julia. This is the recommended backend for Luna: matplotlib is mature, well-documented, and highly customisable.

**Installation:**
```julia
using Pkg
Pkg.add("PythonPlot")
```

### Makie (native Julia)
```julia
using Luna, CairoMakie  # static (PDF/SVG/PNG)
using Luna, GLMakie      # interactive (OpenGL)
using Luna, WGLMakie     # web-based (browser/notebooks)
```
[Makie](https://docs.makie.org/) is a pure-Julia plotting library offering a native Julia experience with advanced features such as interactive 3D spectrograms. Any Makie backend triggers the Luna plotting extension. Choose based on your use case:
- **CairoMakie**: high-quality static output for publications (PDF, SVG, PNG)
- **GLMakie**: interactive plots with pan/zoom in standalone windows
- **WGLMakie**: interactive plots in web browsers, Jupyter notebooks, and VSCode

!!! note
    The Makie backend is fully functional but may have occasional bugs. If you encounter issues, try the PythonPlot backend instead.

**Installation:**
```julia
using Pkg
Pkg.add("CairoMakie")  # or "GLMakie" or "WGLMakie"
```

!!! tip "Switching Makie backends"
    You can switch between Makie backends within a single Julia session using `activate!()`:
    ```julia
    using Luna, CairoMakie    # start with static output
    Plotting.prop_2D(output)   # renders as PNG/SVG

    GLMakie.activate!()        # switch to interactive
    Plotting.prop_2D(output)   # now interactive with pan/zoom
    ```
    This is particularly useful when developing interactively: use GLMakie or WGLMakie for exploration, then switch to CairoMakie for publication-quality output.

!!! tip "Interactive plots in VSCode and Jupyter"
    WGLMakie provides fully interactive plots (pan, zoom, hover) directly inside VSCode's Julia plot pane and in Jupyter notebooks — no external window required:
    ```julia
    using Luna, WGLMakie
    Plotting.prop_2D(output)  # interactive plot in your editor
    ```

### PyPlot (legacy)
```julia
using Luna, PyPlot
```
PyPlot uses the older [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) interface to matplotlib. PyPlot is no longer maintained upstream but is kept in Luna for backward compatibility. New users should use PythonPlot instead.

**Installation:**
```julia
using Pkg
Pkg.add("PyPlot")
```

!!! note
    If multiple plotting packages are loaded, Luna uses the first backend found in the order: PythonPlot, PyPlot, Makie.

## Quick reference

| Function | Description |
|:---------|:------------|
| [`Plotting.prop_2D`](@ref) | 2D false-colour propagation plots (spectral + temporal) |
| [`Plotting.time_1D`](@ref) | Time-domain line plots at specific z-positions |
| [`Plotting.spec_1D`](@ref) | Spectral-domain line plots at specific z-positions |
| [`Plotting.spectrogram`](@ref) | Time-frequency spectrograms |
| [`Plotting.stats`](@ref) | Plot all available propagation statistics |
| [`Plotting.energy`](@ref) | Energy evolution along propagation |
| [`Plotting.cmap_white`](@ref) | Create a colourmap with white at the lowest value |

## Propagation plots

### 2D propagation: `prop_2D`

The main visualisation function for propagation results. Creates side-by-side spectral and temporal heatmaps.

```julia
output = prop_capillary(args...)
Plotting.prop_2D(output)
```

**Spectral axis options** — the first positional argument after `output`:
- `:f` — frequency in PHz (default)
- `:ω` — angular frequency in rad/fs
- `:λ` — wavelength in nm

```julia
Plotting.prop_2D(output, :λ, λrange=(400e-9, 1200e-9), dBmin=-40)
```

**Key keyword arguments:**
- `λrange=(λ_min, λ_max)`: spectral axis limits in metres (default: `(150e-9, 2000e-9)`)
- `trange=(t_min, t_max)`: time axis limits in seconds (default: `(-50e-15, 50e-15)`)
- `dBmin`: lower limit for logarithmic spectral colour scale in dB (default: `-60`)
- `resolution`: smooth the spectral energy density
- `modes`: for multimode simulations — `nothing` (all individual modes), `:sum`, an integer, or a range
- `oversampling`: time-domain oversampling factor (default: `4`)
- `bandpass=(λ_min, λ_max)`: bandpass filter for time-domain plot

### Time-domain line plots: `time_1D`

Plot time-domain slices at one or more propagation distances.

```julia
Plotting.time_1D(output, [0.5, 1.0, 1.5])  # at z = 0.5, 1.0, 1.5 m
Plotting.time_1D(output, 1.0; trange=(-20e-15, 20e-15))
```

**Key keyword arguments:**
- `y`: quantity to plot — `:Pt` (power, default), `:Et` (electric field), `:Esq` (squared field)
- `trange`: time axis limits in seconds
- `modes`: mode selection (see above)
- `bandpass`: bandpass filter wavelength range
- `FTL=true`: overlay the Fourier-transform-limited pulse
- `oversampling`: oversampling factor (default: `4`)

### Spectral-domain line plots: `spec_1D`

Plot spectral slices at one or more propagation distances.

```julia
Plotting.spec_1D(output, [0.5, 1.0, 1.5])
Plotting.spec_1D(output, 1.0, :f; log10=true, λrange=(200e-9, 1500e-9))
```

**Key keyword arguments:**
- `specaxis`: spectral x-axis — `:λ` (default), `:f`, or `:ω`
- `log10`: logarithmic y-axis (default: `true`)
- `log10min`: y-axis dynamic range as fraction of maximum (default: `1e-6`)
- `λrange`: x-axis limits in metres
- `modes`: mode selection
- `resolution`: smooth the spectral energy density

## Spectrograms

Create time-frequency spectrograms using the Gabor transform.

```julia
Plotting.spectrogram(output, 1.0, :λ; trange=(-30e-15, 30e-15), N=512, fw=2e-15)
```

Can also be called with raw data:
```julia
Plotting.spectrogram(t, Et, :λ; trange=(-30e-15, 30e-15), N=512, fw=2e-15)
```

**Key keyword arguments:**
- `trange`: time range for the spectrogram (required)
- `N`: number of time points (required)
- `fw`: gate function width for the Gabor transform (required)
- `λrange`: spectral axis limits in metres
- `log=true`: logarithmic colour scale
- `dBmin`: lower colour limit in dB (default: `-40`)
- `surface3d=true`: 3D surface plot (Makie only)

## Statistics and energy

### Statistics: `stats`

Plot all available propagation statistics (energy, peak power, FWHM, electron density, etc.).

```julia
Plotting.stats(output)
```

This automatically detects which statistics are available in the output and creates appropriate subplot grids for pulse statistics and waveguide/propagation statistics.

### Energy evolution: `energy`

Plot energy vs propagation distance with a secondary axis for conversion efficiency.

```julia
Plotting.energy(output)
Plotting.energy(output; bandpass=(200e-9, 400e-9))  # energy in a spectral window
Plotting.energy(output; modes=:sum)  # sum of all modes
```

## Multimode simulations

For multimode simulations, most plotting functions accept a `modes` keyword argument:

```julia
# Plot all modes individually (default)
Plotting.prop_2D(output)

# Plot only mode 1
Plotting.prop_2D(output; modes=1)

# Plot modes 1 to 3
Plotting.prop_2D(output; modes=1:3)

# Plot the sum of all modes
Plotting.time_1D(output, 1.0; modes=:sum)
```

## Utility functions

### `cmap_white`

Create a colourmap with white at the lowest value, useful for propagation plots where zero intensity should appear white rather than the lowest colourmap colour.

```julia
cm = Plotting.cmap_white("viridis")
Plotting.prop_2D(output; cmap=cm)
```

### `cmap_colours`

Generate an array of colours sampled from a colourmap, useful for custom multi-line plots.

```julia
colours = Plotting.cmap_colours(5, "viridis"; cmin=0.1, cmax=0.9)
```

### `subplotgrid`

Create a figure with subplots laid out in a near-square grid. With the matplotlib
backends this returns `(fig, axs)`; with the Makie backend it instead returns grid
indices and a recommended figure size (see the docstring).

```julia
fig, axs = Plotting.subplotgrid(6)  # creates a 3×2 grid (matplotlib backends)
```

### `cornertext`

Place text in a corner of an axis.

```julia
Plotting.cornertext(ax, "a)"; corner="ul")  # upper-left
```

### `auto_fwhm_arrows`

Draw FWHM arrows with annotations on a plot.

```julia
Plotting.auto_fwhm_arrows(ax, t, power; text=:right, units="fs")
```

## Custom plots with `Processing`

For custom visualisations beyond the built-in functions, use the `Processing` module to extract data and plot with your chosen backend directly.

```julia
using Luna, PythonPlot

output = prop_capillary(args...)

# Get spectral data
specx, Iω = Processing.getIω(output, :λ, specrange=(200e-9, 1500e-9))

# Get time-domain data at a specific z-position
t, Et, z = Processing.getEt(output, 1.0; oversampling=4)

# Custom plot
fig, ax = pyplot.subplots()
ax.plot(specx * 1e9, Iω[:, end])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral energy density")
```

## API Reference

```@autodocs
Modules = [Plotting]
```
