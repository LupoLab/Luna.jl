module Processing
import FFTW
import Luna: Maths, Fields
import Luna.PhysData: wlfreq, c
import Luna.Grid: AbstractGrid, RealGrid, EnvGrid, from_dict
import Luna.Output: AbstractOutput

"""
    arrivaltime(grid, Eω; λlims, winwidth=0, method=:moment, oversampling=1)

Extract the arrival time of the pulse in the wavelength limits `λlims`.

# Arguments
- `λlims::Tuple{Number, Number}` : wavelength limits (λmin, λmax)
- `winwidth` : If a `Number`, set smoothing width (in rad/s) of the window function
                used to bandpass.
                If `:auto`, automatically set the width to 128 frequency samples.
- `method::Symbol` : `:moment` to use 1st moment to extract arrival time, `:peak` to use
                    the time of peak power
- `oversampling::Int` : If >1, oversample the time-domain field before extracting delay
"""
function arrivaltime(grid, Eω; λlims, winwidth=:auto, kwargs...)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    arrivaltime(grid, Eω, window; kwargs...)
end

function arrivaltime(grid::RealGrid, Eω::Vector, ωwindow::Vector{<:Real};
                     method=:moment, oversampling=1)
    Et = FFTW.irfft(Eω .* ωwindow, length(grid.t), 1)
    to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
    arrivaltime(to, abs2.(Maths.hilbert(Eto)); method=method)
end

function arrivaltime(grid::EnvGrid, Eω::Vector, ωwindow::Vector{<:Real};
                     method=:moment, oversampling=1)
    Et = FFTW.ifft(Eω .* ωwindow, 1)
    to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
    arrivaltime(to, abs2.(Eto); method=method)
end

function arrivaltime(t, It::Vector{<:Real}; method)
    if method == :moment
        Maths.moment(t, It)
    elseif method == :peak
        t[argmax(It)]
    else
        error("Unknown arrival time method $method")
    end
end

function arrivaltime(grid, Eω, ωwindow::Vector{<:Real}; kwargs...)
    out = Array{Float64, ndims(Eω)-1}(undef, size(Eω)[2:end])
    cidcs = CartesianIndices(size(Eω)[2:end])
    for ii in cidcs
        out[ii] = arrivaltime(grid, Eω[:, ii], ωwindow; kwargs...)
    end
    out
end

"""
    time_bandwidth(grid, Eω::Vector; λlims=nothing, winwidth=:auto, oversampling=1)

Extract the time-bandwidth product in the wavelength region given by `λlims`. The TBP
is defined here as ΔfΔt where Δx is the FWHM of x. (In this definition, the TBP of 
a perfect Gaussian pulse is ≈0.44). If `oversampling` > 1, the time-domain field is
oversampled before extracting the FWHM.
"""
function time_bandwidth(grid, Eω; λlims=nothing, winwidth=:auto, oversampling=1)
    fwt = fwhm_t(grid, Eω; λlims=λlims, winwidth=winwidth, oversampling=oversampling)
    fwf = fwhm_f(grid, Eω; λlims=λlims, winwidth=winwidth)
    fwt.*fwf
end


"""
    fwhm_t(grid, Eω::Vector; λlims=nothing, winwidth=:auto, oversampling=1)

Extract the temporal FWHM. If `λlims` is given, bandpass first. If `oversampling` > 1, the 
time-domain field is oversampled before extracting the FWHM.
"""
function fwhm_t(grid::AbstractGrid, Eω; λlims=nothing, winwidth=:auto, oversampling=1)
    window = isnothing(λlims) ?
             fill(1, size(grid.ω)) :
             ωwindow_λ(grid.ω, λlims; winwidth=winwidth)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=window)
    fwhm(to, abs2.(Eto))
end


"""
    fwhm_f(grid, Eω::Vector; λlims=nothing, winwidth=:auto, oversampling=1)

Extract the frequency FWHM. If `λlims` is given, bandpass first.
"""
function fwhm_f(grid::AbstractGrid, Eω; λlims=nothing, winwidth=:auto, oversampling=1)
    window = isnothing(λlims) ?
             fill(1, size(grid.ω)) :
             ωwindow_λ(grid.ω, λlims; winwidth=winwidth)
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
    peakpower(grid, Eω; λlims=nothing, winwidth=:auto, oversampling=1)

Extract the peak power. If `λlims` is given, bandpass first.
"""
function peakpower(grid, Eω; λlims=nothing, winwidth=:auto, oversampling=1)
    window = isnothing(λlims) ?
             fill(1, size(grid.ω)) :
             ωwindow_λ(grid.ω, λlims; winwidth=winwidth)
    to, Eto = getEt(grid, Eω; oversampling=oversampling, bandpass=window)
    dropdims(maximum(abs2.(Eto); dims=1); dims=1)
end


"""
    energy_λ(grid, Eω; λlims, winwidth=:auto)

Extract energy within a wavelength band given by `λlims`. `winwidth` can be a `Number` in 
rad/fs to set the smoothing edges of the frequency window, or `:auto`, in which case the
width defaults to 128 frequency samples.
"""
function energy_λ(grid, Eω; λlims, winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    energy_window(grid, Eω, window)
end

function energy_window(grid, Eω, ωwindow)
    _, energyω = Fields.energyfuncs(grid)
    _energy_window(Eω, ωwindow, energyω)
end

_energy_window(Eω::Vector, ωwindow, energyω) = energyω(Eω .* ωwindow)
function _energy_window(Eω, ωwindow, energyω)
    out = Array{Float64, ndims(Eω)-1}(undef, size(Eω)[2:end])
    cidcs = CartesianIndices(size(Eω)[2:end])
    for ii in cidcs
        out[ii] = _energy_window(Eω[:, ii], ωwindow, energyω)
    end
    out
end

"""
    ωwindow_λ(ω, λlims; winwidth=:auto)

Create a ω-axis filtering window to filter in `λlims`. `winwidth`, if a `Number`, sets
the smoothing width of the window in rad/s.
"""
function ωwindow_λ(ω, λlims; winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(ω[2] - ω[1]))
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

"""
    window_maybe(ω, Eω, win)

Apply a frequency window to the field `Eω` if required. Possible values for `win`:

- `nothing` : no window is applied
- 4-`Tuple` of `Number`s : the 4 parameters for a [`Maths.planck_taper`](@ref) in **wavelength**
- 2-`Tuple` of `Number`s : minimum and maximum **wavelength** with automatically chosen smoothing
- `Vector{<:Real}` : a pre-defined window function (shape must match `ω`)
"""
window_maybe(ω, Eω, ::Nothing) = Eω
window_maybe(ω, Eω, win::NTuple{4, Number}) = Eω.*Maths.planck_taper(
    ω, sort(wlfreq.(collect(win)))...)
window_maybe(ω, Eω, win::NTuple{2, Number}) = Eω .* ωwindow_λ(ω, win)
window_maybe(ω, Eω, window) = Eω.*window

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