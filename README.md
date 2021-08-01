# Luna
Luna solves the unidirectional pulse propagation equation (UPPE) for problems in gas-based nonlinear optics. It flexibly supports a variety of propagation geometries and modal expansions (mode-averaged/single-mode guided propagation and multi-mode guided propagation as well as radially symmetric and full 3D free-space propagation). Luna is designed to be extensible: adding e.g. a new type of waveguide or a new nonlinear effect is straightforward, even without editing the main source code.

Luna is written in the [Julia programming language](https://julialang.org/), chosen for its unique combination of readability, ease of use, and speed. If you want to use Luna but are new to Julia, see [the relevant section of this README](#new-to-julia).

There are two ways of using Luna:
1. A very simple high-level interface for the most heavily developed application of Luna--propagation in hollow capillary fibres and hollow-core photonic crystal fibres--consisting of the function [`prop_capillary`](#quickstart) and some helper functions to create input pulses.
2. A low-level interface which allows for full control and customisation of the simulation parameters, the use of custom waveguide modes and gas fills (including gas mixtures), and free-space propagation simulations.

## Installation
Luna requires a Julia version of 1.5 or above.

First install the [CoolProp](https://github.com//CoolProp/CoolProp.jl) Julia package, then Luna:

```julia
]
add https://github.com//CoolProp/CoolProp.jl
add https://github.com/LupoLab/Luna
```
or using `Pkg`

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/CoolProp/CoolProp.jl", rev="master"))
Pkg.add(PackageSpec(url="https://github.com/LupoLab/Luna", rev="master")
```

## Quickstart
To run a simple simulation of ultrafast pulse propagation in a gas-filled hollow capillary fibre, you can use `prop_capillary`. As an example, take a 3-metre length of HCF 125 μm core radius, filled with 1 bar of helium gas, and driving pulses centred at 800 nm wavelength with 120 μJ of energy and 10 fs duration.
```julia
julia> using Luna
julia> output = prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, energy=120e-6, τfwhm=10e-15)
```
The first time you run this code, you will see the precompilation message:
```julia
julia> using Luna
[ Info: Precompiling Luna [30eb0fb0-5147-11e9-3356-d75b018717ce]
```
This will take some time to complete (and you may see additional precompilation messages for the packages Luna depends on), but is only necessary once, unless you update Luna or edit the package source code. Since this is using the default options including FFT planning and caching of the PPT ionisation rate, you will also have to wait for those processes to finish. After the simulation finally runs (which for this example should take between 10 seconds and one minute), you will have the results stored in `output`:
```julia
julia> output = prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, energy=120e-6, τfwhm=10e-15)
[...]
MemoryOutput["simulation_type", "dumps", "meta", "Eω", "grid", "stats", "z"]
```
You can access the results by indexing into `output` like a `Dict`, for example for the frequency-domain field `Eω`:
```julia
julia> output["Eω"]
8193×201 Array{Complex{Float64},2}:
[...]
```
The shape of this array is `(Nω x Nz)` where `Nω` is the number of frequency samples and `Nz` is the number of steps that were saved during the propagation. By default, `prop_capillary` will solve the full-field (carrier-resolved) UPPE. In this case, the numerical Fourier transforms are done using `rfft`, so the number of frequency samples is `(Nt/2 + 1)` with `Nt` the number of samples in the time domain. 
### Multi-mode propagation
`prop_capillary` accepts many keyword arguments (LINK TO DOCS HERE) to customise the simulation parameters and input pulse. One of the most important is `modes`, which defines whether mode-averaged or multi-mode propagation is used, and which modes are included. By default, `prop_capillary` considers mode-averaged propagation in the fundamental (HE₁₁) mode of the capillary, which is fast and simple but less accurate, especially at high intensity when self-focusing and photoionisation play important roles in the propagation dynamics.

Mode-averaged propagation is activated using `modes=:HE11` (the default) or replacing the `:HE11` with a different mode designation (for mode-averaged propagation in a different mode). To run the same simulation as above with the first four modes (HE₁₁ to HE₁₄) of the capillary, set `modes` to `4`:
```julia
julia> output_multimode = prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, energy=120e-6, τfwhm=10e-15, modes=4)
```
The propagation will take much longer, and the output field `Eω` now has shape `(Nω x Nm x Nz)` with `Nm` the number of modes:
```julia
julia> output_multimode["Eω"]
8193×4×201 Array{Complex{Float64},3}:
[...]
```
> **NOTE:** Setting `modes=:HE11` and `modes=1` is **not** equivalent. The first uses mode-averaged propagation (treating all spatial dependence of the nonlinear polarisation as being the same as the Kerr effect) whereas the second projects the spatially dependent nonlinear polarisation onto a single mode. This difference is especially important when photionisation plays a major role.
### Plotting results
More usefully, you can directly plot the propagation results using `Plotting.prop_2D()` (`Plotting` is imported at the same time as `prop_capillary` by the `using Luna` statement):
```julia
julia> Plotting.prop_2D(output)
PyPlot.Figure(PyObject <Figure size 2400x800 with 4 Axes>)
```
This should show a plot like this:
![Propagation example 1](assets/readme_modeAvgProp.png)
You can also display the power spectrum at the input and output (and anywhere in between):
```julia
julia> Plotting.spec_1D(output, [0, 1.5, 3]; log10=true)
PyPlot.Figure(PyObject <Figure size 1700x1000 with 1 Axes>)
```
which will show this:
![Propagation example 2](assets/readme_modeAvgSpec.png)
`Plotting` functions accept many additional keyword arguments to quickly display relevant information. For example, you can show the bandpass-filtered UV pulse from the simulation using the `bandpass` argument:
```julia
julia> Plotting.time_1D(output, [2, 2.5, 3]; trange=(-10e-15, 30e-15), bandpass=(180e-9, 220e-9))
PyPlot.Figure(PyObject <Figure size 1700x1000 with 1 Axes>)
```
![Propagation example 3](assets/readme_modeAvgTime.png)

More plotting functions are available in the `Plotting` module (INSERT DOCS LINK HERE), including for propagation statistics (`Plotting.stats(output)`) and spectrograms (`Plotting.spectrogram()`)

### Output processing
The `Processing` module contains many useful functions for more detailed processing and manual plotting, including:
- Spectral energy density on frequency or wavelength axis with optional spectral resolution setting (`Processing.getEω` and `Processing.getIω`)
- Time-domain fields and pulse envelopes with flexible frequency bandpass and linear (dispersive) propagation operators (`Processing.getEt`)
- Energy (`Processing.energy`) and peak power (`Processing.peakpower`) including after frequency bandpass
- FWHM widths in frequency (`Processing.fwhm_f`) and time (`Processing.fwhm_t`) as well as time-bandwidth product (`Processing.time_bandwidth`)
- g₁₂ coherence between multiple fields (`Processing.coherence`)

## New to Julia?

## Getting help & contributing

## Credits