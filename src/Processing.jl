module Processing
import FFTW
import Luna: Maths, Fields, PhysData
import Luna.PhysData: wlfreq, c
import Luna.Grid: AbstractGrid, RealGrid, EnvGrid, from_dict
import Luna.Output: AbstractOutput

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

Extract the temporal FWHM. If `bandpass` is given, bandpass the field according to `window_maybe`.
If `oversampling` > 1, the  time-domain field is oversampled before extracting the FWHM.
"""
function fwhm_t(grid::AbstractGrid, Eω; bandpass=nothing, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    fwhm(to, abs2.(Eto))
end


"""
    fwhm_f(grid, Eω::Vector; bandpass=nothing, oversampling=1)

Extract the frequency FWHM. If `bandpass` is given, bandpass the field according to `window_maybe`.
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

Extract the peak power. If `bandpass` is given, bandpass the field according to `window_maybe`.
"""
function peakpower(grid, Eω; bandpass=nothing, oversampling=1)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=bandpass)
    dropdims(maximum(abs2.(Eto); dims=1); dims=1)
end


"""
    energy(grid, Eω; bandpass=nothing)

Extract energy. If `bandpass` is given, bandpass the field according to `window_maybe`.
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

"Square window centred at `x0` with full width `xfw`"
function swin(x, x0, xfw)
    abs(x - x0) < xfw/2 ? 1.0 : 0.0
end

"Gaussian window centred at `x0` with full width `xfw`"
function gwin(x, x0, xfw)
    σ = xfw * 0.42
    exp(-0.5 * ((x-x0)/σ)^2)
end

"""
    Eω_to_SEDλ(grid, Eω, λrange, resolution; window=gwin, nsamples=8)

Calculate the spectral energy density, defined by frequency domain field `Eω` defined
on `grid`, on a wavelength scale over the range `λrange` taking account
of spectral `resolution`. The `window` function to use defaults to a Gaussian function with
FWHM of `resolution`, and by default we sample `nsamples=8` times within each `resolution`.

This works for both fields and envelopes and it is assumed that `Eω` is suitably fftshifted
if necessary before this function is called, and that `ω` is monotonically increasing.
"""
function Eω_to_SEDλ(grid, Eω, λrange, resolution; window=gwin, nsamples=8)
    _Eω_to_SEDx(grid, Eω, λrange, resolution, window, nsamples, wlfreq, wlfreq)
end

"""
    Eω_to_SEDf(grid, Eω, Frange, resolution; window=gwin, nsamples=8)

Calculate the spectral energy density, defined by frequency domain field `Eω` defined
on `grid`, on a frequency scale over the range `Frange` taking account
of spectral `resolution`. The `window` function to use defaults to a Gaussian function with
FWHM of `resolution`, and by default we sample `nsamples=8` times within each `resolution`.

This works for both fields and envelopes and it is assumed that `Eω` is suitably fftshifted
if necessary before this function is called, and that `ω` is monotonically increasing.
"""
function Eω_to_SEDf(grid, Eω, Frange, resolution; window=gwin, nsamples=8)
    _Eω_to_SEDx(grid, Eω, Frange, resolution, window, nsamples, x -> x/(2π), x -> x*(2π))
end

"""
Convolution kernel for each output point. We simply loop over all `z` and output points.
The inner loop adds up the contributions from the specified window around
the target point. Note that this works without scaling also for wavelength ranges
because the integral is still over a frequency grid (with appropriate frequency dependent
integration bounds).
"""
function _Eω_to_SEDx_kernel!(SEDx, cidcs, istart, iend, Eω, window, x, xg, resolution, scale)
    for ii in cidcs
        for j in 1:size(SEDx, 1)
            for k in istart[j]:iend[j]
                SEDx[j,ii] += abs2(Eω[k,ii]) * window(x[k], xg[j], resolution) * scale
            end
        end
    end
    SEDx[SEDx .<= 0.0] .= minimum(SEDx[SEDx .> 0.0])
end

function _Eω_to_SEDx(grid::EnvGrid, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
    ω = FFTW.fftshift(grid.ω)
    Eω = FFTW.fftshift(Eω, 1)
    _Eω_to_SEDx(grid, ω, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
end

function _Eω_to_SEDx(grid::RealGrid, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
    _Eω_to_SEDx(grid, grid.ω, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
end

function _Eω_to_SEDx(grid, ω, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
    # build output grid and array
    x = ωtox.(ω)
    nxg = ceil(Int, (xrange[2] - xrange[1])/resolution*nsamples)
    xg = collect(range(xrange[1], xrange[2], length=nxg))
    rdims = size(Eω)[2:end]
    SEDx = Array{Float64, ndims(Eω)}(undef, ((nxg,)..., rdims...))
    fill!(SEDx, 0.0)
    cidcs = CartesianIndices(rdims)
    # we find a suitable nspan
    nspan = 1
    while window(nspan*resolution, 0.0, resolution) > 1e-8
        nspan += 1
    end
    # now we build arrays of start and end indices for the relevent frequency
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
    scale = Fields.prefrac_energy_ω(grid)
    # run the convolution kernel - the function barrier massively improves performance
    _Eω_to_SEDx_kernel!(SEDx, cidcs, istart, iend, Eω, window, x, xg, resolution, scale)
    xg, SEDx
end

"""
    Eω_to_SEDλ_fft(grid, Eω, λrange, resolution; window=gwin, nsamples=8)

Calculate the spectral energy density, defined by frequency domain field `Eω` defined
on `grid`, on a wavelength scale over the range `λrange` taking account
of spectral `resolution`. The `window` function to use defaults to a Gaussian function with
FWHM of `resolution`, and by default we sample `nsamples=8` times within each `resolution`.

This works for both fields and envelopes and it is assumed that `Eω` is suitably fftshifted
if necessary before this function is called.

This function should produce identical output to `Eω_to_SEDλ`, but is based on regridding and FFT
convolution. It appears to perform worse only for very large grid sizes and large ranges, otherwise
it is faster.
"""
function Eω_to_SEDλ_fft(grid::RealGrid, Eω, λrange, resolution; window=gwin, nsamples=8)
    _Eω_to_SEDλ_fft(grid, grid.ω, Eω, λrange, resolution, window=window, nsamples=nsamples)
end

function Eω_to_SEDλ_fft(grid::EnvGrid, Eω, λrange, resolution; window=gwin, nsamples=8)
    ω = FFTW.fftshift(grid.ω)
    Eω = FFTW.fftshift(Eω, 1)
    _Eω_to_SEDλ_fft(grid, ω, Eω, λrange, resolution, window=window, nsamples=nsamples)
end

function _Eω_to_SEDλ_fft(grid, ω, Eω, λrange, resolution; window=gwin, nsamples=8)
    # build output grid and array
    λ = wlfreq.(ω)
    rdims = size(Eω)[2:end]
    cidcs = CartesianIndices(rdims)
    # we find a suitable nspan
    nspan = 1
    while window(nspan*resolution, 0.0, resolution) > 1e-8
        nspan += 1
    end
    sλrange = (λrange[1] - nspan*resolution, λrange[2] + nspan*resolution)
    # TODO error check boundaries
    iλ = (λ .>= sλrange[1]) .& (λ .<= sλrange[2])
    mΔλ = minimum(abs.(diff(λ[iλ])))
    nλg = DSP.nextfastfft(ceil(Int, (sλrange[2] - sλrange[1])/mΔλ))
    λg = collect(range(sλrange[1], sλrange[2], length=nλg))
    ωg = wlfreq.(λg)
    Sout = Array{Float64, ndims(Eω)}(undef, ((nλg,)..., rdims...))
    prefac = Fields.prefrac_energy_ω(grid) / (ω[2] - ω[1])
    for ii in cidcs
        l = Maths.LinTerp(ω[iλ], abs2.(Eω[iλ,ii]) .* prefac .* PhysData.c ./ λ[iλ].^2)
        Sout[:,ii] .= l.(ωg)
    end
    win = FFTW.fftshift(window.(λg, λg[nλg ÷ 2], resolution))
    dλ = λg[2] - λg[1]
    scale = dλ/length(λg)/resolution # TODO not sure why the 1/res factor is necessary
    Eω_to_SEDλ_fft_kernel!(Sout, cidcs, win, nλg, scale)
    red = floor(Int, (resolution/mΔλ) / nsamples)
    red = red < 1 ? 1 : red
    istart = findfirst(x -> x >= λrange[1], λg)
    iend = findfirst(x -> x > λrange[2], λg) - 1
    Sout = Sout[istart:red:iend,cidcs]
    λg = λg[istart:red:iend]
    Sout[Sout .<= 0.0] .= maximum(Sout) * 1e-20
    λg, Sout
end

function Eω_to_SEDλ_fft_kernel!(Sout, cidcs, win, nλg, scale)
    wω = FFTW.rfft(win)
    for ii in cidcs
        Sout[:,ii] .= scale .* abs.(FFTW.irfft(FFTW.rfft(Sout[:,ii]) .* wω, nλg))
    end
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
    getIω(ω, Eω, specaxis)

Get spectral energy density and x-axis given a frequency array `ω` and frequency-domain field
`Eω`, assumed to be correctly normalised (see [`getEω`](@ref)). `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m
"""
function getIω(ω, Eω, specaxis)
    if specaxis == :f
        specx = ω./2π
        If = abs2.(Eω)*2π
        return specx, If
    elseif specaxis == :ω
        specx = ω
        Iω = abs2.(Eω)
        return specx, Iω
    elseif specaxis == :λ
        specx = wlfreq.(ω)
        Iλ = @. ω^2/(2π*c) * abs2.(Eω)
        idcs = sortperm(specx)
        cidcs = CartesianIndices(size(Iλ)[2:end])
        return specx[idcs], Iλ[idcs, cidcs]
    else
        error("Unknown specaxis $specaxis")
    end
end

"""
    getIω(output, specaxis[, zslice])

Calculate the correctly normalised frequency-domain field and convert it to spectral
energy density on x-axis `specaxis` (`:f`, `:ω`, or `:λ`). If `zslice` is given,
returs only the slices of `Eω` closest to the given distances. `zslice` can be a single
number or an array. `specaxis` determines the
x-axis:

- :f -> x-axis is frequency in Hz and Iω is in J/Hz
- :ω -> x-axis is angular frequency in rad/s and Iω is in J/(rad/s)
- :λ -> x-axis is wavelength in m and Iω is in J/m
"""
getIω(output::AbstractOutput, specaxis) = getIω(getEω(output)..., specaxis)

function getIω(output::AbstractOutput, specaxis, zslice)
    ω, Eω, zactual = getEω(output, zslice)
    specx, Iω = getIω(ω, Eω, specaxis)
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
    ω, Eω = getEω(grid, output)
    cidcs = CartesianIndices(size(Eω)[1:end-1])
    zidx = nearest_z(output, zslice)
    return ω, Eω[cidcs, zidx], output["z"][zidx]
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
    getEt(grid, Eω; trange=nothing, oversampling=4, bandpass=nothing)

Get the envelope time-domain electric field (including the carrier wave) from the frequency-
domain field `Eω`. The field can be cropped in time using `trange`, it is oversampled by
a factor of `oversampling` (default 4) and can be bandpassed using a pre-defined window,
or wavelength limits with `bandpass` (see [`window_maybe`](@ref)).
If `zslice` is given, returs only the slices of `Eω` closest to the given distances. `zslice`
can be a single number or an array.
"""
function getEt(grid::AbstractGrid, Eω::AbstractArray;
               trange=nothing, oversampling=4, bandpass=nothing)
    t = grid.t
    Eω = window_maybe(grid.ω, Eω, bandpass)
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
               trange=nothing, oversampling=4, bandpass=nothing)
    t = grid.t
    Eω = window_maybe(grid.ω, output["Eω"], bandpass)
    Etout = envelope(grid, Eω)
    if isnothing(trange)
        idcs = 1:length(t)
    else
        idcs = @. (t < max(trange...)) & (t > min(trange...))
    end
    cidcs = CartesianIndices(size(Etout)[2:end-1])
    zidx = nearest_z(output, zslice)
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, cidcs, zidx], factor=oversampling)
    return to, Eto, output["z"][zidx]
end

struct PeakWindow
    width::Float64
    λmin::Float64
    λmax::Float64
end

function (pw::PeakWindow)(ω, Eω)
    cidcs = CartesianIndices(size(Eω)[3:end]) # dims are ω, modes, rest...
    out = similar(Eω)
    cropidcs = (ω .> wlfreq(pw.λmax)) .& (ω .< wlfreq(pw.λmin))
    cropω = ω[cropidcs]
    Iω = abs2.(Eω)
    for cidx in cidcs
        λpeak = wlfreq(cropω[argmax(Iω[cropidcs, 1, cidx])])
        window = ωwindow_λ(ω, (λpeak-pw.width/2, λpeak+pw.width/2))
        for midx in 1:size(Eω, 2)
            out[:, midx, cidx] .= Eω[:, midx, cidx] .* window
        end
    end
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
window_maybe(ω, Eω, window::Vector) = Eω.*window


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