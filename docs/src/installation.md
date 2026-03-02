# Installation

## Requirements
Luna requires Julia v1.9 or later (we currently recommend v1.10). You can download Julia from the [official website](https://julialang.org/downloads/).

## Installing Luna
To install Luna, open a Julia terminal and enter the package manager by pressing `]`, then run:
```julia
pkg> add Luna
```
This will download and install Luna along with all of its dependencies. The first time you load Luna with `using Luna`, it will be precompiled, which may take a few minutes.

### Development version
If you want to use the latest development version of Luna (or contribute to it), you can install it directly from the GitHub repository:
```julia
pkg> dev Luna
```
or, equivalently:
```julia
pkg> add Luna#master
```

## Using Intel MKL for FFTs
By default, Luna uses [FFTW](http://www.fftw.org/) for fast Fourier transforms. On **Linux and Windows**, you can optionally switch to the Intel Math Kernel Library (MKL) backend for FFTs, which can provide improved performance on Intel hardware.

### Switching to MKL
To switch the FFT backend to MKL, run the following in a Julia session:
```julia
using FFTW
FFTW.set_provider!("mkl")
```
This writes the preference `provider = "mkl"` to a `LocalPreferences.toml` file in your active project directory.

!!! warning "Restart required"
    You **must restart Julia** for the change to take effect. The FFT backend is selected when FFTW.jl is precompiled, so a simple `using FFTW` in the same session will not pick up the new provider.

After restarting Julia, FFTW.jl will load the MKL backend instead of the default FFTW3 library. You can verify which backend is active:
```julia
using FFTW
println(FFTW.fftw_provider)  # should print "mkl"
```

### Switching back to FFTW
To switch back to the default FFTW3 backend:
```julia
using FFTW
FFTW.set_provider!("fftw")
```
As before, a restart of Julia is required for the change to take effect.

### Platform support
MKL is only available on **Linux** and **Windows**. On macOS, FFTW3 is the only supported FFT backend.

### FFTW wisdom
When using MKL, FFTW wisdom (cached FFT plans for faster planning on subsequent runs) is not supported. Luna detects the MKL backend automatically and skips wisdom loading and saving.

## Plotting
Luna's built-in plotting functions (in the `Plotting` module) use [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl), which requires a Python installation with `matplotlib`. If you encounter issues with plotting, you can install `PyPlot` separately:
```julia
pkg> add PyPlot
```
By default, Julia will manage a private Python (Conda) environment for this purpose (and we recommend this route). See the [PyCall.jl documentation](https://github.com/JuliaPy/PyCall.jl) for more details on using a system Python installation instead.
