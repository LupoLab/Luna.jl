# luna
Nonlinear field equation propagator in Julia

## Documentation

Growing documentation can be found at http://luna.lupo-lab.com/
You need the usual LUPO password to access the documentation.

## Installation
Use a recent version of Julia (e.g. 1.5.4).

First install CoolProp, then Luna:

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

## Running an example
To run an example you need a few more packages
```julia
ENV["PYTHON"] = ""
] add FFTW PyPlot
```

Then
```julia
using Luna
include(joinpath(pkgdir(Luna), "\\examples\\basic_modeAvg.jl"))
```

This should run a simple propagation and plot the result. Note that the first time this is run it may take a while to precompile `Luna`. On subsequent runs it should be much faster.
