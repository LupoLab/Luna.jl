# Luna
Luna solves the unidirectional pulse propagation equation (UPPE) for problems in gas-based nonlinear optics. It flexibly supports a variety of propagation geometries and modal expansions (mode-averaged/single-mode guided propagation and multi-mode guided propagation as well as radially symmetric and full 3D free-space propagation). Luna is designed to be extensible: adding e.g. a new type of waveguide or a new nonlinear effect is straightforward, even without editing the main source code.

Luna is written in the [Julia programming language](https://julialang.org/), chosen for its unique combination of readability, ease of use, and speed. If you want to use Luna but are new to Julia, see [the relevant section of this README](#new-to-julia).

There are two ways of using Luna:
1. A very simple high-level interface for the most heavily developed part of Luna--propagation in hollow capillary fibres and hollow-core photonic crystal fibres--consisting of the function [`prop_capillary`](#quickstart) and some helper functions to create input pulses.
2. A low-level interface which allows for full control and customisation of the simulation parameters, addition of custom waveguide modes, and free-space propagation simulations.

## Installation
Use a recent version of Julia (e.g. 1.5.4).

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
More usefully, you can directly plot the propagation results using `Plotting.prop_2D()`:
```julia
julia> Plotting.prop_2D(output)
PyPlot.Figure(PyObject <Figure size 2400x800 with 4 Axes>)
```

## New to Julia?
