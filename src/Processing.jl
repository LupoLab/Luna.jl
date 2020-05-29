module Processing
import FFTW
using EllipsisNotation
import Glob: glob
import Luna: Maths, Fields, PhysData
import Luna.PhysData: wlfreq, c
import Luna.Grid: AbstractGrid, RealGrid, EnvGrid, from_dict
import Luna.Output: AbstractOutput, HDF5Output

"""
    scanproc(f, scanfiles)
    scanproc(f, directory)
    scanproc(f, directory, pattern)
    scanproc(f)

Iterate over the scan output files, apply the processing function `f(o::AbstractOutput)`,
and collect the results in arrays.

The files can be given as:

- a `Vector` of `AbstractString`s containing file paths
- a directory to search for files according to the naming pattern of
    [`Output.@ScanHDF5Output`](@ref)
- a directory and a `glob` pattern

If nothing is specified, `scanproc` uses the current working directory.

`f` can return a single value, an array, or a tuple/array of arrays/numbers.

# Example
```julia
Et, Eω = scanproc("path/to/scandir") do output
    t, Et = getEt(output)
    ω, Eω = getEω(output)
    Et, Eω
end
```
"""
function scanproc(f, scanfiles::AbstractVector{<:AbstractString}; shape=nothing)
    local scanidcs, arrays
    scanfiles = sort(scanfiles)
    for (idx, fi) in enumerate(scanfiles)
        o = HDF5Output(fi)
        ret = f(o)
        if idx == 1 # initialise arrays
            isnothing(shape) && (shape = Tuple(o["meta"]["scanshape"]))
            scanidcs = CartesianIndices(shape)
            arrays = _arrays(ret, shape)
        end
        for (ridx, ri) in enumerate(ret)
            idcs = CartesianIndices(ri)
            arrays[ridx][idcs, scanidcs[idx]] .= ri
        end
    end
    arrays
end

# Default pattern for files named by ScanHDF5Output is [name]_[scanidx].h5 with 5 digits
defpattern = "*_[0-9][0-9][0-9][0-9][0-9].h5"

function scanproc(f, directory::AbstractString=pwd(), pattern::AbstractString=defpattern;
                  shape=nothing)
    scanfiles = glob(pattern, directory) # this returns absolute paths if directory given
    scanproc(f, scanfiles; shape=shape)
end

# Make array(s) with correct size to hold processing results
_arrays(ret::Number, shape) = zeros(typeof(ret), shape)
_arrays(ret::AbstractArray, shape) = zeros(eltype(ret), (size(ret)..., shape...))
_arrays(ret::Tuple, shape) = [_arrays(ri, shape) for ri in ret]

"""
    coherence(Eω; ndim=1)

Calculate the first-order coherence function g₁₂ of the set of fields `Eω`. The ensemble
average is taken over the last `ndim` dimensions of `Eω`, other dimensions are preserved.

See J. M. Dudley and S. Coen, Optics Letters 27, 1180 (2002).
"""
function coherence(Eω; ndim=1)
    dimsize = size(Eω)[end-ndim+1:end]
    outsize = size(Eω)[1:end-ndim]
    prodidcs = CartesianIndices(dimsize)
    restidcs = CartesianIndices(outsize)
    coherence(Eω, prodidcs, restidcs)
end

# function barrier for speedup
function coherence(Eω, prodidcs, restidcs)
    num = zeros(ComplexF64, size(restidcs))
    den1 = zeros(ComplexF64, size(restidcs))
    den2 = zeros(ComplexF64, size(restidcs))
    it = Iterators.product(prodidcs, prodidcs)
    for (idx1, idx2) in it
        Eω1 = Eω[restidcs, idx1]
        Eω2 = Eω[restidcs, idx2]
        @. num += conj(Eω1)*Eω2
        @. den1 += abs2(Eω1)
        @. den2 += abs2(Eω2)
    end
    @. abs(num/sqrt(den1*den2))
end


"""
    coherence(Eω; ndim=1)

Calculate the first-order coherence function g₁₂ of the set of fields `Eω`. The ensemble
average is taken over the last `ndim` dimensions of `Eω`, other dimensions are preserved.

See J. M. Dudley and S. Coen, Optics Letters 27, 1180 (2002).
"""
function coherence(Eω; ndim=1)
    dimsize = size(Eω)[end-ndim+1:end]
    outsize = size(Eω)[1:end-ndim]
    prodidcs = CartesianIndices(dimsize)
    restidcs = CartesianIndices(outsize)
    coherence(Eω, prodidcs, restidcs)
end

# function barrier for speedup
function coherence(Eω, prodidcs, restidcs)
    num = zeros(ComplexF64, size(restidcs))
    den1 = zeros(ComplexF64, size(restidcs))
    den2 = zeros(ComplexF64, size(restidcs))
    it = Iterators.product(prodidcs, prodidcs)
    for (idx1, idx2) in it
        Eω1 = Eω[restidcs, idx1]
        Eω2 = Eω[restidcs, idx2]
        @. num += conj(Eω1)*Eω2
        @. den1 += abs2(Eω1)
        @. den2 += abs2(Eω2)
    end
    @. abs(num/sqrt(den1*den2))
end


"""
    arrivaltime(grid, Eω; bandpass=nothing, method=:moment, oversampling=1)

Extract the arrival time of the pulse in the wavelength limits `λlims`.

# Arguments
- `bandpass` : method to bandpass the field if required. See [`window_maybe`](@ref)
- `method::Symbol` : `:moment` to use 1st moment to extract arrival time, `:peak` to use
                    the time of peak power
- `oversampling::Int` : If >1, oversample the time-domain field before extracting delay
"""
function arrivaltime(grid::AbstractGrid, Eω;
                     bandpass=nothing, method=:moment, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    arrivaltime(to, abs2.(Eto); method=method)
end

function arrivaltime(t::AbstractVector, It::AbstractVector; method)
    if method == :moment
        Maths.moment(t, It)
    elseif method == :peak
        t[argmax(It)]
    else
        error("Unknown arrival time method $method")
    end
end

function arrivaltime(t::AbstractVector, It::AbstractArray; method)
    out = Array{Float64, ndims(It)-1}(undef, size(It)[2:end])
    cidcs = CartesianIndices(size(It)[2:end])
    for ii in cidcs
        out[ii] = arrivaltime(t, It[:, ii]; method=method)
    end
    out
end

"""
    time_bandwidth(grid, Eω; bandpass=nothing, oversampling=1)

Extract the time-bandwidth product, after bandpassing if required. The TBP
is defined here as ΔfΔt where Δx is the FWHM of x. (In this definition, the TBP of 
a perfect Gaussian pulse is ≈0.44). If `oversampling` > 1, the time-domain field is
oversampled before extracting the FWHM.
"""
function time_bandwidth(grid, Eω; bandpass=nothing, oversampling=1)
    fwt = fwhm_t(grid, Eω; bandpass=bandpass, oversampling=oversampling)
    fwf = fwhm_f(grid, Eω; bandpass=bandpass)
    fwt.*fwf
end


"""
    fwhm_t(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1)

Extract the temporal FWHM. If `bandpass` is given, bandpass the fieldaccording to
[`window_maybe`](@ref). If `oversampling` > 1, the  time-domain field is oversampled before
extracting the FWHM.
"""
function fwhm_t(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    fwhm(to, abs2.(Eto))
end


"""
    fwhm_f(grid, Eω::Vector; bandpass=nothing, oversampling=1)

Extract the frequency FWHM. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref).
"""
function fwhm_f(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1)
    Eω = window_maybe(grid.ω, Eω, bandpass)
    f, If = getIω(getEω(grid, Eω)..., :f)
    fwhm(f, If)
end


function fwhm(x, I)
    out = Array{Float64, ndims(I)-1}(undef, size(I)[2:end])
    cidcs = CartesianIndices(size(I)[2:end])
    for ii in cidcs
        out[ii] = fwhm(x, I[:, ii])
    end
    out
end

fwhm(x::Vector, I::Vector) = Maths.fwhm(x, I)

"""
    peakpower(grid, Eω; bandpass=nothing, oversampling=1)

Extract the peak power. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref).
"""
function peakpower(grid, Eω; bandpass=nothing, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    dropdims(maximum(abs2.(Eto); dims=1); dims=1)
end


"""
    energy(grid, Eω; bandpass=nothing)

Extract energy. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref).
"""
function energy(grid, Eω; bandpass=nothing)
    Eω = window_maybe(grid.ω, Eω, bandpass)
    _, energyω = Fields.energyfuncs(grid)
    _energy(Eω, energyω)
end

_energy(Eω::Vector, energyω) = energyω(Eω)

function _energy(Eω, energyω)
    out = Array{Float64, ndims(Eω)-1}(undef, size(Eω)[2:end])
    cidcs = CartesianIndices(size(Eω)[2:end])
    for ii in cidcs
        out[ii] = _energy(Eω[:, ii], energyω)
    end
    out
end

"""
    field_autocorrelation(Et; dims=1)

Calculate the field autocorrelation of `Et`.
"""
function field_autocorrelation(Et, grid::EnvGrid; dims=1)
    FFTW.fftshift(FFTW.ifft(abs2.(FFTW.fft(Et, dims)), dims), dims)
end

function field_autocorrelation(Et, grid::RealGrid; dims=1)
    fac = FFTW.fftshift(FFTW.irfft(abs2.(FFTW.rfft(Et, dims)), length(grid.t), dims), dims)
    Maths.hilbert(fac, dim=dims)
end

"""
    intensity_autocorrelation(Et, grid)

Calculate the intensity autocorrelation of `Et` over `grid`.
"""
function intensity_autocorrelation(Et, grid; dims=1)
    real.(FFTW.fftshift(FFTW.irfft(abs2.(FFTW.rfft(Fields.It(Et, grid), dims)), length(grid.t), dims), dims))
end

"""
    coherence_time(grid, Et; dims=1)

Get the coherence time of a field `Et` over `grid`.
"""
function coherence_time(grid, Et; dims=1)
    Maths.fwhm(grid.t, abs2.(field_autocorrelation(Et, grid, dims=dims)))
end

"""
    specres(ω, Iω, specaxis, resolution, specrange; window=nothing, nsamples=10)

Smooth the spectral energy density `Iω(ω)` to account for the given `resolution`
on the defined `specaxis` and `specrange`. The `window` function to use defaults
to a Gaussian function with FWHM of `resolution`, and by default we sample `nsamples=10`
times within each `resolution`.

Note that you should prefer the `resolution` keyword of [`getIω`](@ref) instead of calling
this function directly.

The input `ω` and `Iω` should be as returned by [`getIω`](@ref) with `specaxis = :ω`.

Returns the new specaxis grid and smoothed spectrum.
"""
function specres(ω, Iω, specaxis, resolution, specrange; window=nothing, nsamples=10)
    if isnothing(window)
        window = let ng=Maths.gaussnorm(fwhm=resolution), resolution=resolution
            (x,x0) -> Maths.gauss(x,fwhm=resolution,x0=x0) / ng
        end
    end
    if specaxis == :λ
        xg, Ix = _specres(ω, Iω, resolution, specrange, window, nsamples, wlfreq, wlfreq)
    elseif specaxis == :f
        xg, Ix = _specres(ω, Iω, resolution, specrange, window, nsamples, x -> x/(2π), x -> x*(2π))
    else
        error("`specaxis` must be one of `:λ` or `:f`")
    end
    xg, Ix
end

function _specres(ω, Iω, resolution, xrange, window, nsamples, ωtox, xtoω)
    # build output grid and array
    x = ωtox.(ω)
    fxrange = extrema(x[(x .> 0) .& isfinite.(x)])
    if isnothing(xrange)
        xrange = fxrange
    else
        xrange = extrema(xrange)
        xrange = (max(xrange[1], fxrange[1]), min(xrange[2], fxrange[2]))
    end
    nxg = ceil(Int, (xrange[2] - xrange[1])/resolution*nsamples)
    xg = collect(range(xrange[1], xrange[2], length=nxg))
    rdims = size(Iω)[2:end]
    Ix = Array{Float64, ndims(Iω)}(undef, ((nxg,)..., rdims...))
    fill!(Ix, 0.0)
    cidcs = CartesianIndices(rdims)
    # we find a suitable nspan
    nspan = 1
    while window(nspan*resolution, 0.0)/window(0.0, 0.0) > 1e-8
        nspan += 1
    end
    # now we build arrays of start and end indices for the relevant frequency
    # band for each output. For a frequency grid this is a little inefficient
    # but for a wavelength grid, which has varying index ranges, this is essential
    # and I think having a common code is simpler/cleaner.
    istart = Array{Int,1}(undef,nxg)
    iend = Array{Int,1}(undef,nxg)
    δω = ω[2] - ω[1]
    i0 = argmin(abs.(ω))
    for i in 1:nxg
        i1 = i0 + round(Int, xtoω(xg[i] + resolution*nspan)/δω)
        i2 = i0 + round(Int, xtoω(xg[i] - resolution*nspan)/δω)
        # we want increasing indices
        if i1 > i2
            i1,i2 = i2,i1
        end
        # handle boundaries
        if i2 > length(ω)
            i2 = length(ω)
        end
        if i1 < i0
            i1 = i0
        end
        istart[i] = i1
        iend[i] = i2
    end
    # run the convolution kernel - the function barrier massively improves performance
    _specres_kernel!(Ix, cidcs, istart, iend, Iω, window, x, xg, δω)
    xg, Ix
end

"""
Convolution kernel for each output point. We simply loop over all outer indices
and output points. The inner loop adds up the contributions from the specified window
around the target point. Note that this works without scaling also for wavelength ranges
because the integral is still over a frequency grid (with appropriate frequency dependent
integration bounds).
"""
function _specres_kernel!(Ix, cidcs, istart, iend, Iω, window, x, xg, δω)
    for ii in cidcs
        for j in 1:size(Ix, 1)
            for k in istart[j]:iend[j]
                Ix[j,ii] += Iω[k,ii] * window(x[k], xg[j]) * δω
            end
        end
    end
    Ix[Ix .<= 0.0] .= minimum(Ix[Ix .> 0.0])
end

"""
    ωwindow_λ(ω, λlims; winwidth=:auto)

Create a ω-axis filtering window to filter in `λlims`. `winwidth`, if a `Number`, sets
the smoothing width of the window in rad/s.
"""
function ωwindow_λ(ω, λlims; winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 64*abs(ω[2] - ω[1]))
    window = Maths.planck_taper(ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
end

function _specrangeselect(x, Ix; specrange=nothing, sortx=false)
    cidcs = CartesianIndices(size(Ix)[2:end])
    if !isnothing(specrange)
        specrange = extrema(specrange)
        idcs = (x .>= specrange[1] .& (x .<= specrange[2]))
        x = x[idcs]
        Ix = Ix[idcs, cidcs]
    end
    if sortx
        idcs = sortperm(x)
        x = x[idcs]
        Ix = Ix[idcs, cidcs]
    end
    x, Ix
end

"""
    getIω(ω, Eω, specaxis; specrange=nothing, resolution=nothing)

Get spectral energy density and x-axis given a frequency array `ω` and frequency-domain field
`Eω`, assumed to be correctly normalised (see [`getEω`](@ref)). `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m

# Keyword arguments
- `specrange::Tuple` can be set to a pair of limits on the spectral range (in `specaxis` units).
- `resolution::Real` is set, smooth the spectral energy density as defined by [`specres`](@ref).

Note that if `resolution` and `specaxis=:λ` is set it is highly recommended to also set `specrange`.
"""
function getIω(ω, Eω, specaxis; specrange=nothing, resolution=nothing)
    sortx = false
    if specaxis == :ω || !isnothing(resolution)
        specx = ω
        Ix = abs2.(Eω)
        if !isnothing(resolution)
            return specres(ω, Ix, specaxis, resolution, specrange)
        end
    elseif specaxis == :f
        specx = ω./2π
        Ix = abs2.(Eω)*2π
    elseif specaxis == :λ
        specx = wlfreq.(ω)
        Ix = @. ω^2/(2π*c) * abs2.(Eω)
        sortx = true
    else
        error("Unknown specaxis $specaxis")
    end
    if !isnothing(specrange) || sortx
        specx, Ix = _specrangeselect(specx, Ix, specrange=specrange, sortx=sortx)
    end
    return specx, Ix
end

"""
    getIω(output, specaxis[, zslice]; kwargs...)

Calculate the correctly normalised frequency-domain field and convert it to spectral
energy density on x-axis `specaxis` (`:f`, `:ω`, or `:λ`). If `zslice` is given,
returs only the slices of `Eω` closest to the given distances. `zslice` can be a single
number or an array. `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m

# Keyword arguments
- `specrange::Tuple` can be set to a pair of limits on the spectral range (in `specaxis` units).
- `resolution::Real` is set, smooth the spectral energy density as defined by [`specres`](@ref).

Note that `resolution` is set and `specaxis=:λ` it is highly recommended to also set `specrange`.
"""
getIω(output::AbstractOutput, specaxis; kwargs...) = getIω(getEω(output)..., specaxis; kwargs...)

function getIω(output::AbstractOutput, specaxis, zslice; kwargs...)
    ω, Eω, zactual = getEω(output, zslice)
    specx, Iω = getIω(ω, Eω, specaxis; kwargs...)
    return specx, Iω, zactual
end

"""
    getEω(output[, zslice])

Get frequency-domain modal field from `output` with correct normalisation (i.e. 
`abs2.(Eω)`` gives angular-frequency spectral energy density in J/(rad/s)).
"""
getEω(output::AbstractOutput, args...) = getEω(makegrid(output), output, args...)
getEω(grid, output) = getEω(grid, output["Eω"])

function getEω(grid::RealGrid, Eω::AbstractArray)
    ω = grid.ω[grid.sidx]
    Eω = Eω[grid.sidx, CartesianIndices(size(Eω)[2:end])]
    return ω, Eω*fftnorm(grid)
end

function getEω(grid::EnvGrid, Eω::AbstractArray)
    idcs = FFTW.fftshift(grid.sidx)
    Eωs = FFTW.fftshift(Eω, 1)
    ω = FFTW.fftshift(grid.ω)[idcs]
    Eω = Eωs[idcs, CartesianIndices(size(Eω)[2:end])]
    return ω, Eω*fftnorm(grid)
end

function getEω(grid, output, zslice)
    zidx = nearest_z(output, zslice)
    ω, Eω = getEω(grid, output["Eω", .., zidx])
    return ω, Eω, output["z"][zidx]
end

fftnorm(grid::RealGrid) = Maths.rfftnorm(grid.t[2] - grid.t[1])
fftnorm(grid::EnvGrid) = Maths.fftnorm(grid.t[2] - grid.t[1])

"""
    getEt(output[, zslice]; kwargs...)

Get the envelope time-domain electric field (including the carrier wave) from the `output`.
If `zslice` is given, returs only the slices of `Eω` closest to the given distances. `zslice`
can be a single number or an array.
"""
getEt(output::AbstractOutput, args...; kwargs...) = getEt(
    makegrid(output), output, args...; kwargs...)

"""
    getEt(grid, Eω; trange=nothing, oversampling=4, bandpass=nothing, FTL=false)

Get the envelope time-domain electric field (including the carrier wave) from the frequency-
domain field `Eω`. The field can be cropped in time using `trange`, it is oversampled by
a factor of `oversampling` (default 4) and can be bandpassed with `bandpass`
(see [`window_maybe`](@ref)). If `FTL` is `true`, return the Fourier-transform limited pulse,
i.e. remove any spectral phase.

If `zslice` is given, returs only the slices of `Eω` closest to the given distances. `zslice`
can be a single number or an array.
"""
function getEt(grid::AbstractGrid, Eω::AbstractArray;
               trange=nothing, oversampling=4, bandpass=nothing, FTL=false)
    t = grid.t
    Eω = window_maybe(grid.ω, Eω, bandpass)
    if FTL
        τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
        Eω .= abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    end
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    cidcs = CartesianIndices(size(Etout)[2:end])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs], factor=oversampling)
    return to, Eto
end

getEt(grid::AbstractGrid, output::AbstractOutput; kwargs...) = getEt(grid, output["Eω"]; kwargs...)

function getEt(grid::AbstractGrid, output::AbstractOutput, zslice;
               trange=nothing, oversampling=4, bandpass=nothing, FTL=false)
    t = grid.t
    zidx = nearest_z(output, zslice)
    Eω = window_maybe(grid.ω, output["Eω", .., zidx], bandpass)
    if FTL
        τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
        Eω .= abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    end
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, ..], factor=oversampling)
    return to, Eto, output["z"][zidx]
end

"""
    PeakWindow(width, λmin, λmax; relative=false)

Window function generator which automatically tracks the peak in the spectral region given
by `λmin` and `λmax` and applies a window of a specific `width` around the peak. If 
`relative` is `true`, `width` is relative bandwidth instead of the wavelength width.

A `PeakWindow` automatically stores the limits of the windows it applies in the field `lims`.
"""
mutable struct PeakWindow
    width::Float64
    λmin::Float64
    λmax::Float64
    relative::Bool
    lims
end

PeakWindow(width, λmin, λmax; relative=false) = PeakWindow(width, λmin, λmax, relative, nothing)

function (pw::PeakWindow)(ω, Eω)
    cidcs = CartesianIndices(size(Eω)[3:end]) # dims are ω, modes, rest...
    out = similar(Eω)
    cropidcs = (ω .> wlfreq(pw.λmax)) .& (ω .< wlfreq(pw.λmin))
    cropω = ω[cropidcs]
    Iω = abs2.(Eω)
    limsA = zeros((2, size(Eω)[3:end]...))
    for cidx in cidcs
        λpeak = wlfreq(cropω[argmax(Iω[cropidcs, 1, cidx])])
        lims = pw.relative ? λpeak.*(1 .+ (-0.5, 0.5).*pw.width) : λpeak .+ (-0.5, 0.5).*pw.width
        window = ωwindow_λ(ω, lims)
        limsA[:, cidx] .= lims
        for midx in 1:size(Eω, 2)
            out[:, midx, cidx] .= Eω[:, midx, cidx] .* window
        end
    end
    pw.lims = limsA
    out
end

"""
    window_maybe(ω, Eω, win)

Apply a frequency window to the field `Eω` if required. Possible values for `win`:

- `nothing` : no window is applied
- 4-`Tuple` of `Number`s : the 4 parameters for a [`Maths.planck_taper`](@ref) in **wavelength**
- 3-`Tuple` of `Number`s : minimum, maximum **wavelength**, and smoothing in **radial frequency**
- 2-`Tuple` of `Number`s : minimum and maximum **wavelength** with automatically chosen smoothing
- `Vector{<:Real}` : a pre-defined window function (shape must match `ω`)
- `PeakWindow` : automatically track the peak in a given range and apply the window around it
"""
window_maybe(ω, Eω, ::Nothing) = Eω
window_maybe(ω, Eω, win::NTuple{4, Number}) = Eω.*Maths.planck_taper(
    ω, sort(wlfreq.(collect(win)))...)
window_maybe(ω, Eω, win::NTuple{2, Number}) = Eω .* ωwindow_λ(ω, win)
window_maybe(ω, Eω, win::NTuple{3, Number}) = Eω .* ωwindow_λ(ω, win[1:2]; winwidth=win[3])
window_maybe(ω, Eω, win::PeakWindow) = win(ω, Eω)
window_maybe(ω, Eω, window::AbstractVector) = Eω.*window


"""
    envelope(grid, Eω)

Get the envelope electric field including the carrier wave from the frequency-domain field
`Eω` sampled on `grid`.
"""
envelope(grid::RealGrid, Eω) = Maths.hilbert(FFTW.irfft(Eω, length(grid.t), 1))
envelope(grid::EnvGrid, Eω) = FFTW.ifft(Eω, 1) .* exp.(im.*grid.ω0.*grid.t)

"""
    makegrid(output)

Create an `AbstractGrid` from the `"grid"` dictionary saved in `output`.
"""
function makegrid(output)
    if output["simulation_type"]["field"] == "field-resolved"
        from_dict(RealGrid, output["grid"])
    else
        from_dict(EnvGrid, output["grid"])
    end
end

"""
    nearest_z(output, z)

Return the index of saved z-position(s) closest to the position(s) `z`. Output is always
an array, even if `z` is a number.
"""
nearest_z(output, z::Number) = [argmin(abs.(output["z"] .- z))]
nearest_z(output, z) = [argmin(abs.(output["z"] .- zi)) for zi in z]

end