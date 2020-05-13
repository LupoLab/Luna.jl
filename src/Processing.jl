module Processing
import FFTW
import Luna: Maths, Fields
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