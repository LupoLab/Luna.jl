module Processing
import FFTW
using EllipsisNotation
import Glob: glob
using Luna
import Luna.PhysData: wlfreq, c
import Luna.Grid: AbstractGrid, RealGrid, EnvGrid, from_dict
import Luna.Output: AbstractOutput, HDF5Output
import Cubature: hcubature
import ProgressLogging: @progress
import Logging: @warn

"""
    Common(val)

Wrapper type to tell `scanproc` that `val` is the same for each simulation being processed,
and so only needs to be returned once rather than for each simulation in the scan.
"""
struct Common{dT}
    data::dT
end

"""
    VarLength(val)

Wrapper type to tell `scanproc` that the shape of `val` is different for each simulation being
processed. Return values wrapped in `VarLength` will be placed in an array of arrays.

!!! note
    While the **shape** of `val` can be different between simulations, the **type** must be
    the same, including the dimensionality and element type of arrays.
"""
struct VarLength{dT}
    data::dT
end

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
    `Output.ScanHDF5Output`
- a directory and a `glob` pattern

If nothing is specified, `scanproc` uses the current working directory.

`f` can return a single value, an array, or a tuple/array of arrays/numbers. Arrays returned
by `f` must either be of the same size for each processed file, or wrapped in a `VarLength`.
Values returned by `f` which are guaranteed to be identical for each processed file can be
wrapped in a `Common`, and `scanproc` only returns these once.

# Example
```julia
Et, Eω = scanproc("path/to/scandir") do output
    t, Et = getEt(output)
    ω, Eω = getEω(output)
    energyout = energyout = Processing.VarLength(output["stats"]["energy"])
    Common(t), Et, Common(ω), Eω, energyout
end
```
"""
function scanproc(f, scanfiles::AbstractVector{<:AbstractString}; shape=nothing)
    local scanidcs, arrays
    scanfiles = sort(scanfiles)
    @progress for (idx, fi) in enumerate(scanfiles)
        try
            o = HDF5Output(fi)
            # wraptuple makes sure we definitely have a Tuple, even if f only returns one thing
            ret = wraptuple(f(o))
            if idx == 1 # initialise arrays
                isnothing(shape) && (shape = Tuple(o["meta"]["scanshape"]))
                scanidcs = CartesianIndices(shape)
                arrays = _arrays(ret, shape)
            end
            for (ridx, ri) in enumerate(ret)
                _addret!(arrays[ridx], scanidcs[idx], ri)
            end
        catch e
            bt = catch_backtrace()
            msg = "scanproc failed for file: $fi:\n"*sprint(showerror, e, bt)
            @warn msg
        end
    end
    unwraptuple(arrays) # if f only returns one thing, we also only return one array
end

"""
    scanproc(f, outputs; shape=nothing)

Iterate over the scan outputs, apply the processing function `f(o::AbstractOutput)`,
and collect the results in arrays.

If the `outputs` are `MemoryOutput`s which do not contain the scan metadata,
the `shape` of the scan must be given explicitly (e.g. via `size(scan)`).

`f` can return a single value, an array, or a tuple/array of arrays/numbers. Arrays returned
by `f` must either be of the same size for each processed output, or wrapped in a `VarLength`.
Values returned by `f` which are guaranteed to be identical for each processed output can be
wrapped in a `Common`, and `scanproc` only returns these once.
"""
function scanproc(f, outputs; shape=nothing)
    local scanidcs, arrays
    @progress for (idx, o) in enumerate(outputs)
        try
            # wraptuple makes sure we definitely have a Tuple, even if f only returns one thing
            ret = wraptuple(f(o))
            if idx == 1 # initialise arrays
                isnothing(shape) && (shape = Tuple(o["meta"]["scanshape"]))
                scanidcs = CartesianIndices(shape)
                arrays = _arrays(ret, shape)
            end
            for (ridx, ri) in enumerate(ret)
                _addret!(arrays[ridx], scanidcs[idx], ri)
            end
        catch e
            bt = catch_backtrace()
            msg = "scanproc failed at index $idx: \n"*sprint(showerror, e, bt)
            @warn msg
        end
    end
    unwraptuple(arrays) # if f only returns one thing, we also only return one array
end

wraptuple(x::Tuple) = x
wraptuple(x) = (x,)

unwraptuple(x::Tuple{<:Any}) = x[1] # single-element Tuple
unwraptuple(x) = x

function _addret!(array, aidcs, ri)
    array[aidcs] = ri
end

function _addret!(array, aidcs, ri::AbstractArray)
    idcs = CartesianIndices(ri)
    array[idcs, aidcs] .= ri
end

function _addret!(array, aidcs, ri::VarLength)
    array[aidcs] = ri.data
end

_addret!(array, aidcs, ri::Common) = nothing

# Default pattern for files named by ScanHDF5Output is [name]_[scanidx].h5 with 5 digits
defpattern = "*_[0-9][0-9][0-9][0-9][0-9].h5"

function scanproc(f, directory::AbstractString=pwd(), pattern::AbstractString=defpattern;
                  shape=nothing)
    scanfiles = glob(pattern, directory) # this returns absolute paths if directory given
    scanproc(f, scanfiles; shape=shape)
end

# Make array(s) with correct size to hold processing results
_arrays(ret, shape) = Array{typeof(ret)}(undef, shape)
_arrays(ret::AbstractArray, shape) = zeros(eltype(ret), (size(ret)..., shape...))
_arrays(ret::Tuple, shape) = Tuple([_arrays(ri, shape) for ri in ret])
_arrays(com::Common, shape) = com.data
_arrays(vl::VarLength, shape) = Array{typeof(vl.data), length(shape)}(undef, shape)

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
- `sumdims` : Single `Int` or `Tuple` of `Int`s. The time-domain power will be summed over
            these dimensions (e.g. modes) before extracting the arrival time.
"""
function arrivaltime(grid::AbstractGrid, Eω;
                     bandpass=nothing, method=:moment, oversampling=1, sumdims=nothing)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    Pt = abs2.(Eto)
    if !isnothing(sumdims)
        Pt = dropdims(sum(Pt; dims=sumdims); dims=sumdims)
    end
    arrivaltime(to, Pt; method=method)
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
function time_bandwidth(grid, Eω; bandpass=nothing, oversampling=1, sumdims=nothing)
    fwt = fwhm_t(grid, Eω; bandpass=bandpass, oversampling=oversampling, sumdims=nothing)
    fwf = fwhm_f(grid, Eω; bandpass=bandpass)
    fwt.*fwf
end


"""
    fwhm_t(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1, sumdims=nothing, minmax=:min)

Extract the temporal FWHM. If `bandpass` is given, bandpass the fieldaccording to
[`window_maybe`](@ref). If `oversampling` > 1, the  time-domain field is oversampled before
extracting the FWHM. If `sumdims` is given, the time-domain power is summed over these
dimensions (e.g. modes) before extracting the FWHM. `minmax` determines determines whether the FWHM
is taken at the narrowest (`:min`) or the widest (`:max`) point.
"""
function fwhm_t(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1, sumdims=nothing, minmax=:min)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    Pt = abs2.(Eto)
    if !isnothing(sumdims)
        Pt = dropdims(sum(Pt; dims=sumdims); dims=sumdims)
    end
    fwhm(to, Pt; minmax)
end

function fwhm_t(output::AbstractOutput; kwargs...)
    grid = makegrid(output)
    fwhm_t(grid, output["Eω"]; kwargs...)
end


"""
    fwhm_f(grid, Eω::Vector; bandpass=nothing, oversampling=1, sumdims=nothing, minmax=:min)

Extract the frequency FWHM. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref). If `sumdims` is given, the energy density is summed over these
dimensions (e.g. modes) before extracting the FWHM. `minmax` determines determines whether the FWHM
is taken at the narrowest (`:min`) or the widest (`:max`) point.
"""
function fwhm_f(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1, sumdims=nothing, minmax=:min)
    Eω = window_maybe(grid.ω, Eω, bandpass)
    f, If = getIω(getEω(grid, Eω)..., :f)
    if !isnothing(sumdims)
        If = dropdims(sum(If; dims=sumdims); dims=sumdims)
    end
    fwhm(f, If; minmax)
end


function fwhm(x, I; minmax=:min)
    out = Array{Float64, ndims(I)-1}(undef, size(I)[2:end])
    cidcs = CartesianIndices(size(I)[2:end])
    for ii in cidcs
        out[ii] = fwhm(x, I[:, ii]; minmax)
    end
    out
end

fwhm(x::Vector, I::Vector; minmax=:min) = Maths.fwhm(x, I; minmax)

"""
    peakpower(grid, Eω; bandpass=nothing, oversampling=1)
    peakpower(output; bandpass=nothing, oversampling=1)

Extract the peak power. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref).
"""
function peakpower(grid, Eω; bandpass=nothing, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    dropdims(maximum(abs2.(Eto); dims=1); dims=1)
end

function peakpower(output; kwargs...)
    grid = makegrid(output)
    peakpower(grid, output["Eω"]; kwargs...)
end


"""
    energy(grid, Eω; bandpass=nothing)
    energy(output; bandpass=nothing)

Extract energy. If `bandpass` is given, bandpass the field according to
[`window_maybe`](@ref).
"""
function energy(grid, Eω; bandpass=nothing)
    Eω = window_maybe(grid.ω, Eω, bandpass)
    _, energyω = Fields.energyfuncs(grid)
    _energy(Eω, energyω)
end

function energy(output::AbstractOutput; bandpass=nothing)
    grid = makegrid(output)
    energy(grid, output["Eω"]; bandpass=bandpass)
end

_energy(Eω::AbstractVector, energyω) = energyω(Eω)

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
        window = let ng=Maths.gaussnorm(fwhm=resolution), σ=resolution/(2*(2*log(2))^(1/2))
            (x,x0) -> exp(-0.5*((x - x0)/σ)^2)/ng
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
    ωs = ω[i0]
    for i in 1:nxg
        i1 = i0 + round(Int, (xtoω(xg[i] + resolution*nspan) - ωs)/δω)
        i2 = i0 + round(Int, (xtoω(xg[i] - resolution*nspan) - ωs)/δω)
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
    @inbounds @fastmath for ii in cidcs
        for j in 1:size(Ix, 1)
            for k in istart[j]:iend[j]
                Ix[j,ii] += Iω[k,ii] * window(x[k], xg[j]) * δω
            end
        end
    end
    Ix[Ix .<= 0.0] .= minimum(Ix[Ix .> 0.0])
end

function _specrangeselect(x, Ix; specrange=nothing, sortx=false)
    cidcs = CartesianIndices(size(Ix)[2:end])
    if !isnothing(specrange)
        specrange = extrema(specrange)
        idcs = ((x .>= specrange[1]) .& (x .<= specrange[2]))
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
    ωwindow_λ(ω, λlims; winwidth=:auto)

Create a ω-axis filtering window to filter in `λlims`. `winwidth`, if a `Number`, sets
the smoothing width of the window in rad/s.
"""
function ωwindow_λ(ω, λlims; winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 64*abs(ω[2] - ω[1]))
    window = Maths.planck_taper(ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
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
               trange=nothing, oversampling=4, bandpass=nothing,
               FTL=false, propagate=nothing)
    t = grid.t
    Eω = window_maybe(grid.ω, Eω, bandpass)
    if FTL
        τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
        Eω = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    end
    Eω = prop_maybe(grid, Eω, propagate)
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, ..], factor=oversampling)
    return to, Eto
end

getEt(grid::AbstractGrid, output::AbstractOutput; kwargs...) = getEt(grid, output["Eω"]; kwargs...)

function getEt(grid::AbstractGrid, output::AbstractOutput, zslice;
               kwargs...)
    zidx = nearest_z(output, zslice)
    to, Eto = getEt(grid, output["Eω", .., zidx]; kwargs...)
    return to, Eto, output["z"][zidx]
end

"""
    AutoWindow(width, λmin, λmax, ω0fun; relative=false, ndims=1)

Window function generator which automatically tracks the central frequency in the spectral
region given by `λmin` and `λmax` and applies a window of a specific `width` around the peak.
The central frequency is found using the function `ω0fun(ω, Iω::AbstractVector)`, where
`ω` and `Iω` are already cropped to within the wavelength limits given.
If `relative` is `true`, `width` is relative bandwidth instead of the wavelength width.
`ndims` determines how many dimensions of the array to sum over. For a field array with size
`(Nω, N1, N2, ...)`, the first dimension is always assumed to be frequency. `ndim=1` means
each field to be analysed is 1-dimensional, so the window iterates over all of `(N1, N2, ...)`.
`ndim=2` means each field to be analysed is 2-dimensional, `(Nω, N1)` in size, and will be 
summed over its second dimension before finding the central frequency. The window iterates
over all other dimensions, `(N2, ...)`.

A `AutoWindow` automatically stores the limits of the windows it applies in the field `lims`.
"""
mutable struct AutoWindow
    width::Float64
    λmin::Float64
    λmax::Float64
    ω0fun
    relative::Bool
    ndims
    lims
end

function AutoWindow(width, λmin, λmax, ω0fun; relative=false, ndims=1)
    AutoWindow(width, λmin, λmax, ω0fun, relative, ndims, nothing)
end

function (pw::AutoWindow)(ω, Eω)
    cidcs = CartesianIndices(size(Eω)[(pw.ndims+1):end])
    out = similar(Eω)
    cropidcs = (ω .> wlfreq(pw.λmax)) .& (ω .< wlfreq(pw.λmin))
    cropω = ω[cropidcs]
    Iω = abs2.(Eω)
    limsA = zeros((2, size(Eω)[(pw.ndims+1):end]...))
    for cidx in cidcs
        Iω_this = Iω[.., Tuple(cidx)...]
        Iωsum = sum(Iω_this; dims=2:ndims(Iω_this))
        λ0 = wlfreq(pw.ω0fun(cropω, Iωsum[cropidcs]))
        lims = pw.relative ? λ0.*(1 .+ (-0.5, 0.5).*pw.width) : λ0 .+ (-0.5, 0.5).*pw.width
        window = ωwindow_λ(ω, lims)
        limsA[:, Tuple(cidx)...] .= lims
        out[.., Tuple(cidx)...] .= Eω[.., Tuple(cidx)...] .* window
    end
    pw.lims = limsA
    out
end

"""
    PeakWindow(width, λmin, λmax; relative=false, ndims=1)

An [`AutoWindow`](@ref) which uses the peak of the spectral energy density as the central
frequency. 
"""
function PeakWindow(width, λmin, λmax; relative=false, ndims=1)
    ω0fun = (ω, Iω) ->  ω[argmax(Iω)]
    AutoWindow(width, λmin, λmax, ω0fun; relative=relative, ndims=ndims)
end

"""
    CentroidWindow(width, λmin, λmax; relative=false, ndims=1, power=1)

An [`AutoWindow`](@ref) which uses the centroid (centre of mass or first moment) of the
spectral energy density as the central frequency. Before calculating the centroid, the 
SED is raised to the `power` given.
"""
function CentroidWindow(width, λmin, λmax; relative=false, ndims=1, power=1)
    ω0fun = (ω, Iω) -> Maths.moment(ω, Iω.^power)
    AutoWindow(width, λmin, λmax, ω0fun; relative=relative, ndims=ndims)
end

"""
    window_maybe(ω, Eω, win)

Apply a frequency window to the field `Eω` if required. Possible values for `win`:

- `nothing` : no window is applied
- 4-`Tuple` of `Number`s : the 4 parameters for a `Maths.planck_taper` in **wavelength**
- 3-`Tuple` of `Number`s : minimum, maximum **wavelength**, and smoothing in **radial frequency**
- 2-`Tuple` of `Number`s : minimum and maximum **wavelength** with automatically chosen smoothing
- `Vector{<:Real}` : a pre-defined window function (shape must match `ω`)
- `PeakWindow` : automatically track the peak in a given range and apply the window around it
- `window(ω, Eω)` : an arbitrary user-supplied window function
"""
window_maybe(ω, Eω, ::Nothing) = Eω
window_maybe(ω, Eω, win::NTuple{4, Number}) = Eω.*Maths.planck_taper(
    ω, sort(wlfreq.(collect(win)))...)
window_maybe(ω, Eω, win::NTuple{2, Number}) = Eω .* ωwindow_λ(ω, win)
window_maybe(ω, Eω, win::NTuple{3, Number}) = Eω .* ωwindow_λ(ω, win[1:2]; winwidth=win[3])
window_maybe(ω, Eω, window) = window(ω, Eω)
window_maybe(ω, Eω, window::AbstractVector) = Eω.*window

prop_maybe(grid, Eω, ::Nothing) = Eω
prop_maybe(grid, Eω, propagator) = propagator(grid, Eω)


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
    makemodes(output)

Create the modes used in a simulation using `MarcatiliMode`s. If `output` was created by
[`Interface.prop_capillary_args`](@ref) and hence has a field `prop_capillary_args`, this is
used to match the gas fill from the simulation. Otherwise, the modes are created without gas
fill.
"""
function makemodes(output; warn_dispersion=true)
    mlines = modelines(output["simulation_type"]["transform"])
    if haskey(output, "prop_capillary_args")
        gas = Symbol(output["prop_capillary_args"]["gas"])
        flength = parse(Float64, output["prop_capillary_args"]["flength"])
        return [makemode(l, gas, output["prop_capillary_args"]["pressure"], flength) for l in mlines]
    else
        if warn_dispersion
            @warn("Gas fill not available when creating modes. Dispersion will not be correct.")
        end
        return [makemode(l) for l in mlines]
    end
end

function makemodes(output, gas, pressure, flength=nothing)
    mlines = modelines(output["simulation_type"]["transform"])
    return [makemode(l, gas, pressure, flength) for l in mlines]
end

function modelines(transform_text)
    startswith(transform_text, "TransModal") || error("makemodes only works for multi-mode simulations")
    lines = split(transform_text, "\n")
    modeline = findfirst(li -> startswith(li, "  modes:"), lines)
    endline = findnext(li -> !startswith(li, " "^4), lines, modeline+1)
    lines[modeline+1 : endline-1]
end
    

function modeargs(line)
    sidx = nextind(line, findfirst('{', line))
    eidx = prevind(line, findfirst('}', line))
    line = line[sidx:eidx]
    kindnm, radius, loss, model, angle = split(line, ",")
    kind = Symbol(kindnm[1:2])
    n = parse(Int, Utils.unsubscript(kindnm[3]))
    m = parse(Int, Utils.unsubscript(kindnm[nextind(kindnm, 3)]))
    a = parse(Float64, split(radius, "=")[2])
    loss = parse(Bool, split(loss, "=")[2])
    model = Symbol(split(model, "=")[2])
    ϕ = parse(Float64, split(angle, "=")[2][1:end-1])*π
    a, kind, n, m, loss, model, ϕ
end

function makemode(line)
    a, kind, n, m, loss, model, ϕ = modeargs(line)
    Capillary.MarcatiliMode(a; kind, n, m, loss, model, ϕ)
end

function makemode(line, gas, pressure::AbstractString, flength)
    a, kind, n, m, loss, model, ϕ = modeargs(line)
    if occursin("(", pressure)
        if occursin("[", pressure)
            error("TODO: Z, P type inputs")
        else
            pin, pout = split(pressure, ",")
            pin = parse(Float64, strip(pin, '('))
            pout = parse(Float64, strip(pout, ')'))
            coren, _ = Capillary.gradient(gas, flength, pin, pout)
            return Capillary.MarcatiliMode(a, coren; kind, n, m, loss, model, ϕ)
        end
    else
        p = parse(Float64, pressure)
        return Capillary.MarcatiliMode(a, gas, p; kind, n, m, loss, model, ϕ)
    end
end

function makemode(line, gas, pressure::Number, flength)
    a, kind, n, m, loss, model, ϕ = modeargs(line)
    Capillary.MarcatiliMode(a, gas, pressure; kind, n, m, loss, model, ϕ)
end

function makemode(line, gas, pressure::Tuple, flength)
    a, kind, n, m, loss, model, ϕ = modeargs(line)
    isnothing(flength) && error("To make two-point gradient, fibre length must be given")
    coren, _ = Capillary.gradient(gas, flength, pressure...)
    return Capillary.MarcatiliMode(a, coren; kind, n, m, loss, model, ϕ)
end

function makemode(line, gas, pressure::AbstractArray, flength)
    a, kind, n, m, loss, model, ϕ = modeargs(line)
    Z, P = pressure
    coren, _ = Capillary.gradient(gas, Z, P)
    return Capillary.MarcatiliMode(a, coren; kind, n, m, loss, model, ϕ)
end

"""
    beam(grid, Eωm, modes, x, y; z=0, components=:xy)
    beam(output, x, y, zslice; bandpass=nothing)

Calculate the beam profile of the multi-mode field `Eωm` on the grid given by spatial
coordinates `x` and `y`. If `output` is given, create the `modes` from that and take the
field nearest propagation slice `zslice`.
"""
function beam(output, x, y, zslice; bandpass=nothing)
    modes = makemodes(output; warn_dispersion=false)
    pol = polarisation_components(output)
    zidx = nearest_z(output, zslice)
    grid = makegrid(output)
    Eωm = output["Eω", .., zidx]
    Eωm = dropdims(Eωm; dims=3)
    Eωm = window_maybe(grid.ω, Eωm, bandpass)
    beam(grid, Eωm, modes, x, y; z=zslice, components=pol)
end

function beam(grid, Eωm, modes, x, y; z=0, components=:xy)
    tospace = Modes.ToSpace(modes; components)
    fluence = zeros(length(y), length(x))
    _, energy_ω = Fields.energyfuncs(grid) # energyfuncs include correct FFT normalisation
    Eωxy = zeros(ComplexF64, (length(grid.ω), tospace.npol))
    coords = Modes.dimlimits(modes[1])[1]
    for (yidx, yi) in enumerate(y)
        for (xidx, xi) in enumerate(x)
            xs = coords == :polar ? xs = (hypot(xi, yi), atan(yi, xi)) : (xi, yi)
            Modes.to_space!(Eωxy, Eωm, xs, tospace; z)
            # integrate over time/frequency and multiply by ε₀c/2 -> fluence
            fluence[yidx, xidx] = PhysData.ε_0*PhysData.c/2*sum(energy_ω(Eωxy))
        end
    end
    fluence
end

"""
    getEtxy(output, xs, z; kwargs...)
    getEtxy(Etm, modes, xs, z; components=:xy)

Calculate the time-dependent electric field at transverse position `xs` and longitudinal position `z`
from either the modal time-dependent field `Etm` or the given `output`.

`xs` should be a 2-Tuple of coordinates--either `(r, θ)` for polar coordinates or `(x, y)`
in Cartesian coordinates, depending on the coordinate system of the `modes`--or a 2-Tuple of vectors
containing the coordinates. If vectors are given, the output contains values of Etxy at all combinations of
the coordinates.

Additional keyword arguments to `getEtxy(output, ...)` are passed through to `Processing.getEt`
"""
function getEtxy(output, xs, z; kwargs...)
    modes = makemodes(output; warn_dispersion=false)
    pol = polarisation_components(output)
    t, Etm = getEt(output, z; kwargs...) # (Nt, Nm, Nz)
    t, getEtxy(Etm, modes, xs, z; components=pol)
end

function getEtxy(Etm, modes, xs::Tuple{<:Number, <:Number}, z; components=:xy)
    tospace = Modes.ToSpace(modes; components)
    Etxy = zeros(eltype(Etm), (size(Etm, 1), tospace.npol))
    Modes.to_space!(Etxy, Etm[.., 1], xs, tospace; z)
    Etxy
end

function getEtxy(Etm, modes, xs::Tuple{AbstractVector, AbstractVector}, z; components=:xy)
    tospace = Modes.ToSpace(modes; components)
    x1, x2 = xs
    Etxy = zeros(eltype(Etm), (size(Etm, 1), length(x1), length(x2), tospace.npol))
    for (x2idx, x2i) in enumerate(x2)
        for (x1idx, x1i) in enumerate(x1)
            @views Modes.to_space!(Etxy[:, x1idx, x2idx, :], Etm[.., 1], (x1i, x2i), tospace; z)
        end
    end
    Etxy    
end

function polarisation_components(output)
    t = output["simulation_type"]["transform"]
    startswith(t, "TransModal") || error("beam profile only works for multi-mode simulations")
    lines = split(t, "\n")
    lidx = findfirst(lines) do line
        occursin("polarisation:", line)
    end
    pl = lines[lidx]
    if occursin("x,y", pl)
        return :xy
    else
        return Symbol(pl[end])
    end
end

"""
    ionisation_fraction(output, xs; ratefun, oversampling=1)
    ionisation_fraction(output; ratefun, oversampling=1, maxevals=1000)

Calculate the ionisation fraction at transverse coordinates `xs` using the ionisation-rate
function `ratefun`. If `xs` is not given, calculate the average ionisation fraction across
the waveguide core. In this case, `maxevals` determines the maximum number of function
evaluations for the integral.

!!! warning
    Calculating the average ionisation fraction is **much** slower than calculating it at
    a single point
"""
function ionisation_fraction(output, xs; ratefun, oversampling=1)
    modes = makemodes(output; warn_dispersion=false)
    pol = polarisation_components(output)
    tospace = Modes.ToSpace(modes; components=pol)
    t, Et = getEt(output; oversampling) # (Nt, Nm, Nz)
    δt = t[2]-t[1]
    z = output["z"]
    ionf = zero(z)
    Etxy = zeros(Float64, (length(t), tospace.npol))
    for ii in eachindex(ionf)
        Modes.to_space!(Etxy, real(Et[:, :, ii]), xs, tospace; z=z[ii])
        absEt = tospace.npol > 1 ? hypot.(Etxy[:, 1], Etxy[:, 2]) : Etxy
        ionf[ii] = Ionisation.ionfrac(ratefun, absEt, δt)[end]
    end
    ionf
end

function ionisation_fraction(output; ratefun, oversampling=1, maxevals=1000)
    modes = makemodes(output; warn_dispersion=false)
    pol = polarisation_components(output)
    tospace = Modes.ToSpace(modes; components=pol)
    t, Et = getEt(output; oversampling) # (Nt, Nm, Nz)
    Et = real(Et)
    δt = t[2]-t[1]
    z = output["z"]
    frac_temp = zero(t)
    ionf = zero(z)
    dl = Modes.dimlimits(modes[1])
    Etxy = zeros(Float64, (length(t), tospace.npol))
    for ii in eachindex(ionf)
        Et_this = Et[:, :, ii]
        val, _ = hcubature(dl[2], dl[3]; maxevals) do xs
            Modes.to_space!(Etxy, Et_this, xs, tospace; z=z[ii])
            absEt = tospace.npol > 1 ? hypot.(Etxy[:, 1], Etxy[:, 2]) : Etxy
            Ionisation.ionfrac!(frac_temp, ratefun, absEt, δt)
            dl[1] == :polar ? xs[1]*frac_temp[end] : frac_temp[end]
        end
        ionf[ii] = val
    end
    area, _ = hcubature(dl[2], dl[3]) do xs
        dl[1] == :polar ? xs[1] : one(xs[1])
    end
    ionf./area
end


"""
    nearest_z(output, z)

Return the index of saved z-position(s) closest to the position(s) `z`. Output is always
an array, even if `z` is a number. If `z` is negative, its absolute value is taken as the fraction
of the total propagation distance.
"""
nearest_z(output, z::Number) = z < 0 ? [round(Int, min(abs(z), 1)*length(output["z"]))] : [argmin(abs.(output["z"] .- z))]
nearest_z(output, z) = [nearest_z(output, zi)[1] for zi in z]

end