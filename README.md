# Luna
Luna solves the unidirectional pulse propagation equation (UPPE) for problems in gas-based nonlinear optics. It flexibly supports a variety of propagation geometries and modal expansions (mode-averaged/single-mode guided propagation and multi-mode guided propagation as well as radially symmetric and full 3D free-space propagation). Luna is designed to be extensible: adding e.g. a new type of waveguide or a new nonlinear effect is straightforward, even without editing the main source code.

Luna is written in the [Julia programming language](https://julialang.org/), chosen for its unique combination of readability, ease of use, and speed. If you want to use Luna but are new to Julia, see [the relevant section of this README](#new-to-julia).

There are two ways of using Luna:
1. A very simple high-level interface for the most heavily developed part of Luna--propagation in hollow capillary fibres and hollow-core photonic crystal fibres--consisting of the function `prop_capillary` and some helper functions to create input pulses.
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

## 

## New to Julia?
