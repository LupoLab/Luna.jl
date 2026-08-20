module Fields
import Luna: Grid, Maths, PhysData, Modes
import Luna.PhysData: wlfreq, ε_0, μ_0
import StaticArrays: SVector
import HCubature: hcubature
import NumericalIntegration: integrate, SimpsonEven
import Random: AbstractRNG, GLOBAL_RNG
import Statistics: mean
import Hankel
import LinearAlgebra: dot, norm
import FFTW
import BlackBoxOptim
import Optim
import DelimitedFiles: readdlm
import HCubature: hquadrature
import DSP: unwrap
import Logging: @warn

abstract type AbstractField end

"""
    TimeField

Abstract supertype for time-domain only fields.
"""
abstract type TimeField <: AbstractField end

"""
    PulseField(λ0, energy, ϕ, τ0, Itshape)

Represents a temporal pulse with shape defined by `Itshape`.

# Fields
- `λ0::Float64`: the central field wavelength
- `energy::Float64`: the pulse energy
- `power::Float64`: the pulse peak power (**after** applying any spectral phases)
- `ϕ::Vector{Float64}`: spectral phases (CEP, group delay, GDD, TOD, ...)
- `Itshape`: a callable `f(t)` to get the shape of the intensity/power in the time domain
"""
struct PulseField{eT, pT, iT} <: TimeField
    λ0::Float64
    energy::eT
    power::pT
    ϕ::Vector{Float64}
    Itshape::iT
end

function PulseField(;λ0, Itshape, energy=nothing, power=nothing, ϕ=Float64[])
    if !isnothing(power)
        if !isnothing(energy)
            error("only one of `energy` or `power` can be specified")
        end
    elseif isnothing(energy)
        error("one of `energy` or `power` must be specified")
    end
    PulseField(λ0, energy, power, ϕ, Itshape)
end

"""
    GaussField(;λ0, τfwhm, energy, ϕ, m=1)

Construct a (super)Gaussian shaped pulse with intensity/power FWHM `τfwhm`, either
`energy` or peak `power` specified, superGaussian parameter `m=1` and other parameters
as defined for [`PulseField`](@ref).
"""
function GaussField(;λ0, τfwhm, energy=nothing, power=nothing, ϕ=Float64[], m=1)
    if !isnothing(power)
        if !isnothing(energy)
            error("only one of `energy` or `power` can be specified")
        end
    elseif isnothing(energy)
        error("one of `energy` or `power` must be specified")
    end
    PulseField(λ0, energy, power, ϕ, t -> Maths.gauss(t, fwhm=τfwhm, power=2*m))
end

"""
    SechField(;λ0, energy, τw=nothing, τfwhm=nothing, ϕ=0.0, τ0=0.0)

Construct a Sech^2(t/τw) shaped pulse, specifying either the
natural width `τw`, or the intensity/power FWHM `τfwhm`, and either
`energy` or peak `power` specified.
Other parameters are as defined for [`PulseField`](@ref).
"""
function SechField(;λ0, energy=nothing, power=nothing, τw=nothing, τfwhm=nothing,
                    ϕ=Float64[])
    if !isnothing(τfwhm)
        if !isnothing(τw)
            error("only one of `τw` or `τfwhm` can be specified")
        else
            τw = τfwhm/(2*log(1 + sqrt(2)))
        end
    elseif isnothing(τw)
        error("one of `τw` or `τfwhm` must be specified")
    end
    if !isnothing(power)
        if !isnothing(energy)
            error("only one of `energy` or `power` can be specified")
        end
    elseif isnothing(energy)
        error("one of `energy` or `power` must be specified")
    end
    PulseField(λ0, energy, power, ϕ, t -> sech(t/τw)^2)
end

"""
    make_Et(p::PulseField, grid)

Create electric field for `PulseField`, either the field (for `RealGrid`) or
the envelope (for `EnvGrid`)
"""
function make_Et(p::PulseField, grid::Grid.RealGrid)
    t = grid.t
    ω0 = PhysData.wlfreq(p.λ0)
    @. sqrt(p.Itshape(t))*cos(ω0*t)
end

function make_Et(p::PulseField, grid::Grid.EnvGrid)
    t = grid.t
    Δω = PhysData.wlfreq(p.λ0) - grid.ω0
    @. sqrt(p.Itshape(t))*exp(im*(Δω*t))
end

"""
    (p::PulseField)(Eω, grid, energy_t, FT)

Add the field to `Eω` for the provided `grid`, `energy_t` function and Fourier transform `FT`
"""
function (p::PulseField)(grid, FT)
    Et = make_Et(p, grid)
    if length(p.ϕ) >= 1
        Et = FT \ prop_taylor!(FT * Et, grid, p.ϕ, p.λ0)
    end
    if !isnothing(p.energy)
        energy_t = Fields.energyfuncs(grid)[1]
        Et .*= sqrt(p.energy)/sqrt(energy_t(Et))
    else
        Pt = It(Et, grid)
        Et .*= sqrt(p.power)/sqrt(maximum(Pt))
    end

    FT * Et
end

"""
    CWField(Pavg, Aωfunc)

Represents a continuous-wave field with spectral phase/amplitude defined by `Aωfunc`.

# Fields
- `Pavg::Float64`: the average power
- `Aωfunc`: a callable `f(ω)` to get the amplitude/phase of the field in the frequency domain
"""
struct CWField{aT} <: TimeField
    Pavg::Float64
    Aωfunc::aT
end

"""
    CWSech(;λ0, Pavg, Δλ)

Construct a CW field with Sech^2 spectral power density and random phase, with spectral
full-width half-maximim of `Δλ` and other parameters as defined for [`CWField`](@ref).
"""
function CWSech(;λ0, Pavg, Δλ, rng=GLOBAL_RNG)
    ωw = PhysData.ΔλΔω(Δλ, λ0)/(2*log(1 + sqrt(2)))
    ω0 = PhysData.wlfreq(λ0)
    Aωfunc(ω) = let rng=rng, ωw=ωw, ω0=ω0
        sech((ω - ω0)/ωw)*exp(1im*2π*rand(rng))
    end
    CWField(Pavg, Aωfunc)
end

"""
    (c::CWField)(grid, FT)

Get the field for the provided `grid` and Fourier transform `FT`
"""
function (c::CWField)(grid::Grid.EnvGrid, FT)
    Eω = c.Aωfunc.(grid.ω)
    istart = findfirst(isequal(1.0), grid.twin)
    iend = findlast(isequal(1.0), grid.twin)
    shape = abs.(Eω)
    Eω′ = Eω
    while true
        Eω = shape .* exp.(1im .* angle.(Eω))
        Et = FT \ Eω
        Et .*= sqrt(c.Pavg) / sqrt(mean(It(Et, grid)[istart:iend]))
        Et .*= grid.twin
        Eω = FT * Et
        err = norm(Eω .- Eω′)/maximum(abs.(Eω))
        err < 1e-2 && break
        Eω′ = Eω
    end
    Eω
end

struct DataField <: TimeField
    ω::Vector{Float64}
    Iω::Vector{Float64}
    ϕω::Vector{Float64}
    energy::Float64
    ϕ::Vector{Float64}
    λ0::Float64
end

"""
    DataField(ω, Iω, ϕω; energy, ϕ=Float64[], λ0=NaN)

Represents a field with spectral power density `Iω` and spectral phase `ϕω`, sampled on
radial frequency axis `ω`.
"""
DataField(ω, Iω, ϕω; energy, ϕ=Float64[], λ0=NaN) = DataField(ω, Iω, ϕω, energy, ϕ, λ0)

"""
    DataField(ω, Eω; energy, ϕ=Float64[], λ0=NaN)

Create a `DataField` from the complex frequency-domain field `Eω` sampled on radial
frequency grid `ω`.
"""
DataField(ω, Eω; energy, ϕ=Float64[], λ0=NaN) = DataField(ω, abs2.(Eω), unwrap(angle.(Eω)),
                                                          energy, ϕ, λ0)

"""
    DataField(fpath; energy, ϕ=Float64[], λ0=NaN)

Create a `DataField` by loading `ω`, `Iω`, and `ϕω` from the file at `fpath`. The file must
contain 3 columns:

- frequency in Hz
- spectral power density (arbitrary units)
- unwrapped spectral phase
"""
function DataField(fpath; energy, ϕ=Float64[], λ0=NaN)
    dat = readdlm(fpath, ' ')
    DataField(dat[:, 1]*2π, dat[:, 2], dat[:, 3]; energy, ϕ, λ0)
end

"""
    (d::DataField)(grid, FT)

Interpolate the `DataField` onto the provided `grid` (note the argument `FT` is unused).
"""
function (d::DataField)(grid::Grid.AbstractGrid, FT)
    if maximum(grid.ω) < maximum(d.ω)
        @warn("Interpolating onto a coarser grid may clip the input spectrum.")
    end
    energy_ω = Fields.energyfuncs(grid)[2]
    ϕg = Maths.BSpline(d.ω, d.ϕω).(grid.ω)
    Ig = Maths.BSpline(d.ω, d.Iω).(grid.ω)
    Ig[Ig .< 0] .= 0
    Ig[.!(minimum(d.ω) .< grid.ω .< maximum(d.ω))] .= 0
    Ig .*= grid.ωwin
    Eω = sqrt.(Ig) .* exp.(1im.*ϕg)
    Eω .*= sqrt(d.energy/energy_ω(Eω))
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    Eω .*= exp.(-1im .* grid.ω .* τ)
    if length(d.ϕ) >= 1
        λ0 = isnan(d.λ0) ? wlfreq(Maths.moment(d.ω, d.Iω)) : d.λ0
        prop_taylor!(Eω, grid, d.ϕ, λ0)
    end
    Eω
end

"""
    PropagatedField(propagator!, field)

A wrapper around a previously defined `TimeField` which applies the mutating propagation
function `propagator!(Eω, grid)` to the field `Eω`.
"""
struct PropagatedField{pT, fT<:TimeField} <: TimeField
    propagator!::pT
    field::fT
end

PropagatedField(::Nothing, field::fT) where fT <:TimeField = field

function (pf::PropagatedField)(grid::Grid.AbstractGrid, FT)
    Eω = pf.field(grid, FT)
    pf.propagator!(Eω, grid)
    Eω
end

"""
    ShotNoise(rng=GLOBAL_RNG)

Creates one photon per mode quantum noise (shot noise) to add to an input field.
If no random number generator `rng` is provided, it defaults to `GLOBAL_RNG`
"""
struct ShotNoise{rT<:AbstractRNG} <: TimeField
    rng::rT
end

function ShotNoise(rng=GLOBAL_RNG)
    ShotNoise(rng)
end

"""
    (s::ShotNoise)(Eω, grid)

Get shotnoise for the provided `grid`. The optional parameter `FT`
is unused and is present for interface compatibility with [`TimeField`](@ref).
"""
function (s::ShotNoise)(grid::Grid.RealGrid, FT=nothing)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = @. sqrt(PhysData.ħ*grid.ω/δω)
    rFFTamp = sqrt(2π)/2δt*amp
    φ = 2π*rand(s.rng, size(grid.ω)...)
    @. rFFTamp * exp(1im*φ)
end

function (s::ShotNoise)(grid::Grid.EnvGrid, FT=nothing)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    amp[grid.sidx] = @. sqrt(PhysData.ħ*grid.ω[grid.sidx]/δω)
    FFTamp = sqrt(2π)/δt*amp
    φ = 2π*rand(s.rng, size(grid.ω)...)
    @. FFTamp * exp(1im*φ)
end

"""
    generate_noise_field(grid::Grid.RealGrid; rng=GLOBAL_RNG, nmodes=1)

Generate a one-photon-per-mode spectral noise field for the modified shot-noise model
(Chen & Wise, arXiv:2410.20567).

Returns a complex frequency-domain array with amplitude `√(ħω/δω) × √(2π)/(2δt)` and
uniformly random spectral phase, matching the convention of [`ShotNoise`](@ref) for a
real-valued field (rFFT normalisation). When `nmodes > 1`, returns an `(nω, nmodes)` array
with independent noise per mode for multimode propagation.

Unlike [`ShotNoise`](@ref), which is added to the input field at `z = 0`, this noise field
is intended to be injected into the **nonlinear operator** at every propagation step via the
`noise_field` keyword argument. The dispersion operator does not act on this noise,
preventing artificial FWM phase-matching — the key advantage over the traditional shot-noise
approach.

The noise field is generated once before propagation and held constant throughout. Different
random seeds (via `rng`) produce different noise realisations for ensemble/shot-to-shot
statistics.

# References
- Chen & Wise, "A simple accurate way to model noise-seeded ultrafast nonlinear processes",
  arXiv:2410.20567 (2024)
"""
function generate_noise_field(grid::Grid.RealGrid; rng=GLOBAL_RNG, nmodes=1)
    # Frequency spacing on the angular-frequency grid
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    # One-photon-per-mode amplitude: √(ħω/δω), matching ShotNoise
    amp = @. sqrt(PhysData.ħ * grid.ω / δω)
    # Scale for rFFT convention (real-valued field, half-spectrum):
    # factor √(2π)/(2δt) converts from angular-frequency density to DFT bins
    rFFTamp = sqrt(2π) / 2δt * amp
    # Uniformly random spectral phase ∈ [0, 2π)
    # For multimode (nmodes > 1), generate independent noise per mode → (nω, nmodes)
    if nmodes > 1
        φ = 2π * rand(rng, length(grid.ω), nmodes)
        rFFTamp .* exp.(1im .* φ)
    else
        φ = 2π * rand(rng, size(grid.ω)...)
        @. rFFTamp * exp(1im * φ)
    end
end

"""
    generate_noise_field(grid::Grid.EnvGrid; rng=GLOBAL_RNG, nmodes=1)

Generate a one-photon-per-mode spectral noise field for the modified shot-noise model
(Chen & Wise, arXiv:2410.20567), for envelope-field propagation.

Returns a complex frequency-domain array with amplitude `√(ħω/δω) × √(2π)/δt` and
uniformly random spectral phase, matching the convention of [`ShotNoise`](@ref) for a
complex envelope field (full FFT normalisation). Only frequency bins within `grid.sidx`
(the signal window) receive non-zero noise.

See [`generate_noise_field(::Grid.RealGrid)`](@ref) for details on the modified shot-noise
model.
"""
function generate_noise_field(grid::Grid.EnvGrid; rng=GLOBAL_RNG, nmodes=1)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    # Only populate signal frequencies (grid.sidx masks out negative/out-of-band bins)
    amp[grid.sidx] = @. sqrt(PhysData.ħ * grid.ω[grid.sidx] / δω)
    # Scale for full FFT convention (complex envelope):
    # factor √(2π)/δt converts from angular-frequency density to DFT bins
    FFTamp = sqrt(2π) / δt * amp
    if nmodes > 1
        φ = 2π * rand(rng, length(grid.ω), nmodes)
        FFTamp .* exp.(1im .* φ)
    else
        φ = 2π * rand(rng, size(grid.ω)...)
        @. FFTamp * exp(1im * φ)
    end
end

"""
    SpatioTemporalField(λ0, energy, ϕ, τ0, Ishape)

Represents a spatiotemporal pulse with shape defined by `Ishape`.

# Fields
- `λ0::Float64`: the central field wavelength
- `energy::Float64`: the pulse energy
- `ϕ::Float64`: the CEO phase
- `τ0::Float64`: the temproal shift from grid time 0
- `Ishape`: a callable `f(t, xs)` to get the shape of the intensity/power in the time-space domain
"""
struct SpatioTemporalField{iT} <: AbstractField
    λ0::Float64
    energy::Float64
    ϕ::Float64
    τ0::Float64
    Ishape::iT
    propz::Float64
end

"Gaussian temporal-spatial field defined radially"
function GaussGauss(t, r::AbstractVector, fwhm, m, w0)
    Maths.gauss.(t, fwhm=fwhm, power=2*m) .* Maths.gauss.(r, w0/2)'
end

"Gaussian temporal-spatial field defined on x-y grid"
function GaussGauss(t, r::AbstractArray{T,3} where T, fwhm, m, w0)
    Maths.gauss.(t, fwhm=fwhm, power=2*m) .* Maths.gauss.(r, w0/2)
end

"""
    GaussGaussField(;λ0, τfwhm, energy, w0, ϕ=0.0, τ0=0.0, m=1)

Construct a (super)Gaussian shaped pulse with intensity/power FWHM `τfwhm`,
superGaussian parameter `m=1` and Gaussian shaped spatial profile with waist `w0`,
propagation distance from the waist of `propz`,
and other parameters as defined for [`TimeField`](@ref).
"""
function GaussGaussField(;λ0, τfwhm, energy, w0, ϕ=0.0, τ0=0.0, m=1, propz=0.0)
    SpatioTemporalField(λ0, energy, ϕ, τ0,
                        (t, xs) -> GaussGauss(t, xs, τfwhm, m, w0),
                        propz)
end

function make_Etr(s::SpatioTemporalField, grid::Grid.RealGrid, spacegrid)
    t = grid.t .- s.τ0
    ω0 = PhysData.wlfreq(s.λ0)
    sqrt.(s.Ishape(t, spacegrid.r)) .* cos.(ω0.*t .+ s.ϕ)
end

function make_Etr(s::SpatioTemporalField, grid::Grid.EnvGrid, spacegrid)
    t = grid.t .- s.τ0
    Δω = PhysData.wlfreq(s.λ0) - grid.ω0
    sqrt.(s.Ishape(t, spacegrid.r)) .* exp.(im .* (s.ϕ .+ Δω.*t))
end

transform(spacegrid::Hankel.QDHT, FT, Etr) = spacegrid * (FT * Etr)

transform(spacegrid::Grid.FreeGrid, FT, Etr) = FT * Etr

"""
    (s::SpatioTemporalField)(grid, spacegrid, FT)

Get the field for the provided `grid`, `spacegrid` function
and Fourier transform `FT`
"""
function (s::SpatioTemporalField)(grid, spacegrid, FT)
    Etr = make_Etr(s, grid, spacegrid)
    energy_t = Fields.energyfuncs(grid, spacegrid)[1]
    Etr .*= sqrt(s.energy)/sqrt(energy_t(Etr))
    Eωk = transform(spacegrid, FT, Etr)
    if s.propz != 0.0
        prop!(Eωk, s.propz, grid, spacegrid)
    end
    Eωk
end

function prop!(Eωk, z, grid, q::Hankel.QDHT)
    kzsq = @. (grid.ω/PhysData.c)^2 - (q.k^2)'
    kzsq[kzsq .< 0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω/PhysData.c))
end

function prop!(Eωk, z, grid, xygrid)
    kzsq = ((grid.ω ./ PhysData.c).^2
            .- reshape(xygrid.ky.^2, (1, length(xygrid.ky), 1))
            .- reshape(xygrid.kx.^2, (1, 1, length(xygrid.kx)))
    )
    kzsq[kzsq.<0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω / PhysData.c))
end

"""
    gauss_beam(k, ω0; z=0.0, pol=:y)

Gaussian beam field distribution with waist radius `ω0` and wavenumber `k`,
at position `z` from focus. `pol` describes the polarisation direction,
one of `:x` or `:y`.
"""
function gauss_beam(k, ω0; z=0.0, pol=:y)
    let k=k, ω0=ω0, z=z, pol=pol
        function fieldfunc(xs)
            r = xs[1]
            zr = k*ω0^2/2
            ω = ω0*sqrt(1 + (z/zr)^2)
            R1 = z/(z^2 + zr^2) # 1/R
            ψ = atan(z/zr)
            phase = exp(-1im * (k*z + k*r^2*R1/2 - ψ))
            E = ω0/ω * exp(-r^2/ω^2) * phase
            if pol == :x
                return SVector(E, 0.0)
            else
                return SVector(0.0, E)
            end
        end
    end
end

function int2D(field1, field2, lowerlim, upperlim)
    Ifunc(xs) = 0.5*sqrt(ε_0/μ_0)*dot(field1(xs), field2(xs))*xs[1]
    val, _ = hcubature(lowerlim, upperlim) do xs
        real(Ifunc(xs))
    end
    val
end

function normalised_field(fieldfunc, rmax)
    scale = 1.0/sqrt(int2D(fieldfunc, fieldfunc, (0.0, 0.0), (rmax, 2π)))
    return let scale=scale, fieldfunc=fieldfunc
        (xs) -> fieldfunc(xs) .* scale
    end
end

function normalised_gauss_beam(k, ω0; z=0.0, pol=:y)
    zr = k*ω0^2/2
    ω = ω0*sqrt(1 + (z/zr)^2)
    normalised_field(gauss_beam(k, ω0; z, pol), 6*ω)
end

"""
    coupled_field(i, mode, E, fieldfunc; energy, kwargs...)

Create an element of an input field tuple (for use in `Luna.setup`) based on coupling
field `E` into a `mode`. The index `i` specifies the mode index. The temporal fields are
initialised using `fieldfunc` (e.g. one of `GaussField`, `SechField` etc.) with the
same keyword arguments.
"""
function coupled_field(i, mode, E, fieldfunc; energy, kwargs...)
    ei = energy * abs2(Modes.overlap(mode, E))
    (mode=i, fields=(fieldfunc(;energy=ei, kwargs...),))
end

"""
    gauss_beam_init(modes, k, ω0, fieldfunc; energy, kwargs...)

Create an input field tuple (for use in `Luna.setup`) based on coupling a focused
Gaussian beam with focused spot size `ω0` and wavenumber `k` into `modes`.
The temporal fields are initialised using `fieldfunc` (e.g. one of `GaussField`,
`SechField` etc.) with the same keyword arguments.

```jldoctest
julia> a = 125e-6;
julia> energy = 1e-3;
julia> λ0 = 800e-9;
julia> modes = (Capillary.MarcatiliMode(a, :He, 1.0, m=1), Capillary.MarcatiliMode(a, :He, 1.0, m=2));
julia> fields = Fields.gauss_beam_init(modes, 2*pi/λ0, a*0.64, Fields.GaussField; λ0=λ0, τfwhm=30e-15, energy=energy);
julia> fields[1].fields[1].energy/energy ≈ 0.98071312
true
julia> fields[2].fields[1].energy/energy ≈ 0.0061826217
true
```
"""
function gauss_beam_init(modes, k, ω0, fieldfunc; energy, kwargs...)
    gauss = normalised_gauss_beam(k, ω0)
    tuple(collect(coupled_field(i, mode, gauss, fieldfunc; energy=energy, kwargs...) for (i,mode) in enumerate(modes))...)
end

It(Et, grid::Grid.RealGrid) = abs2.(Maths.hilbert(Et))
It(Et, grid::Grid.EnvGrid) = abs2.(Et)

iFT(Eω, grid::Grid.RealGrid) = FFTW.irfft(Eω, length(grid.t), 1)
iFT(Eω, grid::Grid.EnvGrid) = FFTW.ifft(Eω, 1)

"Calculate energy from modal field E(t)"
function energyfuncs(grid::Grid.RealGrid)
    function energy_t(Et)
        return integrate(grid.t, It(Et, grid), SimpsonEven())
    end

    prefac = 2π/(grid.ω[end]^2)
    function energy_ω(Eω)
        prefac*integrate(grid.ω, abs2.(Eω), SimpsonEven())
    end
    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.EnvGrid)
    function energy_t(Et)
        return integrate(grid.t, It(Et, grid), SimpsonEven())
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    prefac = 2π*δω/(Δω^2)
    function energy_ω(Eω)
        prefac*sum(abs2.(Eω))
    end
    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.RealGrid, q::Hankel.QDHT)
    function energy_t(Et)
        Eta = Maths.hilbert(Et)
        tintg = integrate(grid.t, abs2.(Eta), SimpsonEven())
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(tintg, q)
    end

    prefac = 2π*PhysData.c*PhysData.ε_0/2 * 2π/(grid.ω[end]^2)
    function energy_ω(Eω)
        ωintg = integrate(grid.ω, abs2.(Eω), SimpsonEven())
        return prefac*Hankel.integrateK(ωintg, q)
    end
    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.EnvGrid, q::Hankel.QDHT)
    function energy_t(Et)
        tintg = integrate(grid.t, abs2.(Et), SimpsonEven())
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(tintg, q)
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    prefac = 2π*PhysData.c*PhysData.ε_0/2 * 2π*δω/(Δω^2)
    function energy_ω(Eω)
        ωintg = dropdims(sum(abs2.(Eω); dims=1), dims=1)
        return prefac*Hankel.integrateK(ωintg, q)
    end
    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.RealGrid, xygrid::Grid.FreeGrid)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    δt = grid.t[2] - grid.t[1]
    prefac_t = PhysData.c*PhysData.ε_0/2 * δx * δy * δt
    function energy_t(Et)
        Eta = Maths.hilbert(Et)
        return  prefac_t * sum(abs2.(Eta))
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = grid.ω[end]
    δkx = xygrid.kx[2] - xygrid.kx[1]
    Δkx = length(xygrid.kx)*δkx
    δky = xygrid.ky[2] - xygrid.ky[1]
    Δky = length(xygrid.ky)*δky
    prefac = PhysData.c*PhysData.ε_0/2 * 2π*δω/(Δω^2) * 2π*δkx/(Δkx^2) * 2π*δky/(Δky^2)
    energy_ω(Eω) = prefac * sum(abs2.(Eω))

    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    δt = grid.t[2] - grid.t[1]
    prefac_t = PhysData.c*PhysData.ε_0/2 * δx * δy * δt
    function energy_t(Et)
        return  prefac_t * sum(abs2.(Et))
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    δkx = xygrid.kx[2] - xygrid.kx[1]
    Δkx = length(xygrid.kx)*δkx
    δky = xygrid.ky[2] - xygrid.ky[1]
    Δky = length(xygrid.ky)*δky
    prefac = PhysData.c*PhysData.ε_0/2 * 2π*δω/(Δω^2) * 2π*δkx/(Δkx^2) * 2π*δky/(Δky^2)
    energy_ω(Eω) = prefac * sum(abs2.(Eω))

    return energy_t, energy_ω
end

"""
    prop_taylor!(Eω, grid, ϕs, λ0)
    prop_taylor!(Eω, grid::Grid.AbstractGrid, ϕs, λ0)

Add spectral phase, given as Taylor-expansion coefficients `ϕs` around central wavelength
`λ0`, to the frequency-domain field `Eω`. Sampling axis of `Eω` can be given either as an
`AbstractGrid` or the frequency axis `ω`.
"""
function prop_taylor!(Eω, ω, ϕs, λ0)
    Δω = ω .- wlfreq(λ0)
    ϕ = zeros(length(ω))
    for (n, ϕi) in enumerate(ϕs)
        ϕ .+= Δω.^(n-1)./factorial(n-1) * ϕi
    end
    Eω .*= exp.(-1im.*ϕ)
end

prop_taylor!(Eω, grid::Grid.AbstractGrid, ϕs, λ0) = prop_taylor!(Eω, grid.ω, ϕs, λ0)

"""
    prop_taylor(Eω, grid, ϕs, λ0)
    prop_taylor(Eω, grid::Grid.AbstractGrid, ϕs, λ0)

Return a copy of the frequency-domain field `Eω` with added spectral phase, given as Taylor-expansion coefficients `ϕs` around central wavelength `λ0`.
Sampling axis of `Eω` can be given either as an `AbstractGrid` or the frequency axis `ω`.
"""
prop_taylor(Eω, args...) = prop_taylor!(copy(Eω), args...)

"""
    prop_material!(Eω, ω, material, thickness, λ0=nothing;
                   P=1, T=PhysData.roomtemp, lookup=nothing)

Linearly propagate the frequency-domain field `Eω` through a certain `thickness` of a `material`.
If the central wavelength `λ0` is given, remove the group delay at this wavelength.
Keyword arguments `P` (pressure), `T` (temperature)
and `lookup` (whether to use lookup table instead of Sellmeier expansion).
"""
function prop_material!(Eω, ω, material, thickness, λ0=nothing; kwargs...)
    propagator_material(material; kwargs...)(Eω, ω, thickness, λ0)
end

prop_material!(Eω, grid::Grid.AbstractGrid, args...; kwargs...) = prop_material!(
    Eω, grid.ω, args...; kwargs...)

"""
    prop_material(Eω, ω, material, thickness, λ0=nothing;
                   P=1, T=PhysData.roomtemp, lookup=nothing)

Return a copy of the frequency-domain field `Eω` after linear propagation through a certain
`thickness` of a `material`. If the central wavelength `λ0` is given, remove the group
delay at this wavelength. Keyword arguments `P` (pressure), `T` (temperature)
and `lookup` (whether to use lookup table instead of Sellmeier expansion).
"""
prop_material(Eω, args...; kwargs...) = prop_material!(copy(Eω), args...; kwargs...)

"""
    propagator_material(material; P=1, T=PhysData.roomtemp, lookup=nothing)

Create a function `prop!(Eω, ω, thickness, λ0)` which propagates the field `Eω` through
a certain `thickness` of a `material`. If the central wavelength `λ0` is given, remove the group
delay at this wavelength.  Keyword arguments `P` (pressure), `T` (temperature)
and `lookup` (whether to use lookup table instead of Sellmeier expansion).
"""
function propagator_material(material; P=1, T=PhysData.roomtemp, lookup=nothing)
    n = PhysData.ref_index_fun(material, P, T; lookup=lookup)
    β1 = PhysData.dispersion_func(1, n)
    function prop!(Eω, ω, thickness, λ0=nothing)
        β = ω./PhysData.c .* n.(wlfreq.(ω))
        if !isnothing(λ0)
            β .-= β1(λ0) .* (ω .- wlfreq(λ0))
        end
        β[.!isfinite.(β)] .= 0
        Eω .*= exp.(-1im.*real(β).*thickness)
    end
    prop!
end

"""
    prop_mirror!(Eω, ω, reflections, mirror)
    prop_mirror!(Eω, grid, reflections, mirror)

Propagate the field `Eω` linearly by adding a number of `reflections` from the `mirror` type.
"""
prop_mirror!(Eω, ω::AbstractArray, mirror::Symbol, reflections::Integer) = prop_mirror!(Eω, ω, reflections, PhysData.lookup_mirror(mirror))
prop_mirror!(Eω, ω::AbstractArray, reflections::Integer, mirror::Symbol) = prop_mirror!(Eω, ω, reflections, PhysData.lookup_mirror(mirror))

Base.@deprecate prop_mirror!(Eω, ω, mirror, reflections) prop_mirror!(Eω, ω, reflections, mirror)

"""
    prop_mirror!(Eω, ω, reflections, λR, R, λGDD, GDD, λ0, λmin, λmax; kwargs...)
    prop_mirror!(Eω, grid, reflections, λR, R, λGDD, GDD, λ0, λmin, λmax; kwargs...)

Propagate the field `Eω` linearly by adding a number of `reflections` from a mirror whose tabulated reflectivity
and group-delay dispersion per reflection is given by:

- `λR`: wavelength samples for reflectivity in SI units (m)
- `R`: mirror reflectivity (between 0 and 1)
- `λGDD`: wavelength samples for GDD in SI units (m)
- `GDD`: GDD in SI units (s²)
- `λ0`: central wavelength (used to remove any overall group delay)
- `λmin`, `λmax`: bounds of the wavelength region to apply the transfer function over

Additional keyword arguments are passed to `PhysData.process_mirror_data`:
- `fitorder`: order of polynomial fit to use in removing overall group delay (default: 5)
- `windowwidth`: wavelength width of the smoothing region outside `(λmin, λmax)`
                for the window in SI units (default: 20e-9, i.e. 20 nm)
"""
function prop_mirror!(Eω, ω::AbstractArray, reflections::Integer, λR, R, λGDD, GDD, λ0, λmin, λmax; kwargs...)
    transferfunction = PhysData.process_mirror_data(λR, R, λGDD, GDD, λ0, λmin, λmax; kwargs...)
    prop_mirror!(Eω, ω, transferfunction, reflections)
end

"""
    prop_mirror!(Eω, ω, reflections, transferfunction)
    prop_mirror!(Eω, grid, reflections, transferfunction)

Propagate the field `Eω` linearly by adding a number of `reflections` from a mirror with a
given transfer function. `transferfunction` should take a single argument `λ`, wavelength in SI units (m),
and return the complex frequency-domain response of the mirror (amplitude and phase).
"""
function prop_mirror!(Eω, ω::AbstractArray, reflections::Integer, transferfunction)
    λ = wlfreq.(ω)
    t = transferfunction.(λ) # transfer function
    tn = t.^reflections
    tn[.!isfinite.(tn)] .= 0
    if reflections < 0
        tn = exp.(1im .* angle.(tn))
    end
    Eω .*= tn
end

prop_mirror!(Eω, grid::Grid.AbstractGrid, args...; kwargs...) = prop_mirror!(Eω, grid.ω, args...; kwargs...)

"""
    prop_mirror(Eω, ω, args...; kwargs...)
    prop_mirror(Eω, grid, args...; kwargs...)

Return a copy of the field `Eω` after reflection off of mirrors. For other arguments see [`prop_mirror!`](@ref)
"""
prop_mirror(Eω, args...; kwargs...) = prop_mirror!(copy(Eω), args...; kwargs...)

"""
    prop_mode!(Eω, ω, mode, distance, λ0=nothing)

Propagate the field `Eω` linearly by a certain `distance` in the given `mode`. If the
central wavelength `λ0` is given, remove the group delay at this wavelength. Propagation
includes both dispersion and loss.
"""
function prop_mode!(Eω, ω, mode, distance, λ0=nothing)
    β(z) = ω./PhysData.c .* Modes.neff.(mode, ω; z=z)
    β1(z) = Modes.dispersion(mode, 1, wlfreq(λ0); z=z)
    βint, err = hquadrature(β, 0, abs(distance))
    if !isnothing(λ0)
        β1int, err = hquadrature(β1, 0, abs(distance))
        βint .-= β1int .* (ω .- wlfreq(λ0))
    end
    expφ = exp.(-1im.*sign(distance).*conj(βint))
    expφ[.!isfinite.(expφ)] .= 0
    Eω .*= expφ
end

prop_mode!(Eω, grid::Grid.AbstractGrid, args...) = prop_mode!(Eω, grid.ω, args...)

prop_mode(Eω, args...) = prop_mode!(copy(Eω), args...)

"""
    littrow_angle(λ, Λ; m=-1)

Calculate the Littrow angle of incidence (in radians) for wavelength `λ` (in metres),
grating period `Λ` (in metres) and diffraction order `m` (dimensionless integer, default -1).

At Littrow, the diffracted beam returns along the incident direction (`θm = -θi`).
"""
littrow_angle(λ, Λ; m=-1) = asin(-m*λ/(2Λ))

"""
    grating_GDD(λ, Λ, m, θi)

Calculate the group-delay dispersion per unit separation (in s²/m) of a double-pass
grating pair using the Treacy formula, for wavelength `λ` (in metres), grating period `Λ`
(in metres), diffraction order `m` (dimensionless integer) and angle of incidence `θi`
(in radians).
"""
function grating_GDD(λ, Λ, m, θi)
    θm = asin(m*λ/Λ + sin(θi))
    -m^2*λ^3 / (π * PhysData.c^2 * Λ^2 * cos(θm)^3)
end

"""
    prop_gratings!(Eω, ω, Λ, L, m, θi, λ0=nothing)
    prop_gratings!(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi, λ0=nothing)

Add the spectral phase acquired after passing through a four grating compressor
with grating period `Λ` (in metres), grating separation `L` (in metres),
diffraction order `m` (dimensionless integer) and angle of incidence `θi`
(in radians). The sampling axis of `Eω` can be given either as an
`AbstractGrid` or the angular frequency axis `ω` (in rad/s).

If the central wavelength `λ0` (in metres) is given, the overall phase and group
delay at `λ0` are removed so that only the dispersive part of the phase remains.

Frequency components outside the grating diffraction bandwidth are set to zero.
"""
function prop_gratings!(Eω, ω, Λ, L, m, θi, λ0=nothing)
    λ = PhysData.wlfreq.(ω)
    mask = @. abs(m*λ/Λ + sin(θi)) <= 1
    θm = @. asin(m*λ[mask]/Λ + sin(θi))
    ϕ = zero(λ)
    ϕ[mask] .= @. 2*ω[mask]*L*cos(θm)/PhysData.c
    if !isnothing(λ0)
        # Remove constant phase and group delay at λ0
        ω0 = wlfreq(λ0)
        ϕ_at(ω_val) = 2*ω_val*L*cos(asin(m*wlfreq(ω_val)/Λ + sin(θi)))/PhysData.c
        ϕ0 = ϕ_at(ω0)
        dϕdω = Maths.derivative(ϕ_at, ω0, 1)
        ϕ[mask] .-= ϕ0 .+ dϕdω .* (ω[mask] .- ω0)
    end
    # Apply phase to in-band and zero out-of-band frequencies
    # transfer is 0 for out-of-band (zeroing those components) and exp(-iϕ) for in-band
    transfer = zeros(ComplexF64, length(ω))
    transfer[mask] .= exp.(-1im.*ϕ[mask])
    Eω .*= transfer
    Eω
end

prop_gratings!(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi, λ0=nothing) = prop_gratings!(Eω, grid.ω, Λ, L, m, θi, λ0)

"""
    prop_gratings(Eω, ω, Λ, L, m, θi, λ0=nothing)
    prop_gratings(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi, λ0=nothing)

Return a copy of the frequency-domain field `Eω` with the additional spectral phase
acquired after passing through a four grating compressor with grating period `Λ` (in
metres), grating separation `L` (in metres), diffraction order `m` (dimensionless integer)
and angle of incidence `θi` (in radians). The sampling axis of `Eω` can be given either as
an `AbstractGrid` or the angular frequency axis `ω` (in rad/s).

If the central wavelength `λ0` (in metres) is given, the overall phase and group
delay at `λ0` are removed so that only the dispersive part of the phase remains.

Frequency components outside the grating diffraction bandwidth are set to zero.
"""
prop_gratings(Eω, args...) = prop_gratings!(copy(Eω), args...)


#= ======================================================================
   Prism pair compression — full numerical ray tracing
   
   A double-pass prism pair compressor consists of two identical prisms in an
   anti-parallel configuration. A retroreflecting mirror sends the beam back
   through both prisms, doubling the accumulated spectral phase and eliminating
   spatial chirp.
   
   The spectral phase is computed by full 2D ray tracing through both prisms,
   verified against the Lightcon toolbox calculator
   (https://toolbox.lightcon.com/tools/prismpair).
   
   ## Prism Orientation
   
   - Prism 1 (apex up): apex pointing upward; beam enters through the left
     (input) face and exits through the right (output) face.
   - Prism 2 (apex down): apex pointing downward (inverted), positioned to the
     right of and below Prism 1. Beam enters through the left face (parallel
     to Prism 1's output face) and exits through the right face.
   
   ## Coordinate system
   
       y (up)
       |    /\           Prism 1 (apex up)
       |   /  \
       |  /    \
       |  ------
       |
       |         ------
       |         \    /   Prism 2 (apex down)
       |          \  /
       |           \/
       +----------------------------> x (right)
   
   ## Insertion parameters
   
   - `l1` [m]: distance **along the input face** of Prism 1, from the apex to
     the point where the center-wavelength beam hits the face.
   - `l2` [m]: distance **along the entry face** of Prism 2, from the apex to
     the beam incidence point. Only independently specified when using
     `L_lightcon`; otherwise determined by the ray trace.
   
   ## Separation (three options, provide ONE)
   
   - `L` (positional, default): apex-to-apex Euclidean distance (Keller/Weiner
     convention). Same as `L_keller`.
   - `L_lightcon` (keyword): perpendicular distance between Prism 1 input face
     and Prism 2 output face (Lightcon convention). Requires both `l1` and `l2`.
   - `w`, `h` (keyword pair): horizontal / vertical apex displacements,
     with `h` positive downward.
   
   Conversion:  `L_lightcon = w cos(α/2) + h sin(α/2)`,
                `L_keller   = sqrt(w^2 + h^2)`.
   
   ## Dispersion mechanism
   
   - Angular dispersion (from separation): negative GDD (anomalous).
   - Material traversal (from insertions l1, l2): positive GDD (normal).
   
   ## Double pass
   
       ϕ_double(ω) = 2 ϕ_single(ω)
   
   References:
     A. M. Weiner, "Ultrafast Optics" (Wiley, 2009), Ch. 4
     U. Keller, "Ultrafast Lasers" (Springer, 2021), Ch. 3
     R. L. Fork et al., Opt. Lett. 9, 150 (1984)
   ====================================================================== =#

"""
    brewster_angle(material, λ; P=1, T=PhysData.roomtemp, lookup=nothing)

Brewster angle (in radians) for `material` at wavelength `λ` (in metres):
`θ_B = arctan(n(λ))`. At Brewster's angle, p-polarised light has zero
reflection at the air-material interface.

Keyword arguments `P` (pressure), `T` (temperature), and `lookup` are passed to
`PhysData.ref_index_fun`.

# Examples
```jldoctest
julia> rad2deg(Fields.brewster_angle(:SiO2, 800e-9))
55.47390378498051
```
"""
function brewster_angle(material, λ; P=1, T=PhysData.roomtemp, lookup=nothing)
    n = real(PhysData.ref_index_fun(material, P, T; lookup=lookup)(λ))
    atan(n)
end

"""
    mindev_apex(material, λ; P=1, T=PhysData.roomtemp, lookup=nothing)

Apex angle `α` (in radians) for simultaneous minimum deviation and Brewster
incidence on both surfaces of a prism made of `material` at wavelength `λ`
(in metres):

    α = π − 2 arctan(n)   [equivalently  2 arctan(1/n)]

This is the standard "Brewster-cut" prism geometry used in ultrafast laser systems.

Keyword arguments `P`, `T`, `lookup` are forwarded to `PhysData.ref_index_fun`.

See also [`brewster_angle`](@ref).

# Examples
```jldoctest
julia> rad2deg(Fields.mindev_apex(:SiO2, 800e-9))
69.05219243003899
```
"""
function mindev_apex(material, λ; P=1, T=PhysData.roomtemp, lookup=nothing)
    n = real(PhysData.ref_index_fun(material, P, T; lookup=lookup)(λ))
    2*atan(1/n)
end

"""
    _prism_trace(θi, α, n)

Trace a ray through a single prism with apex angle `α` and refractive index `n`.
The ray enters the first surface at angle of incidence `θi`.

Returns `(θe, valid)` where:
- `θe`: exit angle from the second surface (angle with respect to the surface normal)
- `valid`: `false` if total internal reflection occurs at either surface

Internally, Snell's law is applied at each surface:
- Surface 1: `sin(θi) = n·sin(θr1)` (refraction into prism)
- Surface 2: `n·sin(α − θr1) = sin(θe)` (refraction out of prism)
"""
function _prism_trace(θi, α, n)
    !isfinite(n) && return (0.0, false)
    sinθr1 = sin(θi) / n
    (abs(sinθr1) > 1 || !isfinite(sinθr1)) && return (0.0, false)
    θr1 = asin(sinθr1)
    θi2 = α - θr1
    (θi2 < 0 || !isfinite(θi2)) && return (0.0, false)
    sinθe = n * sin(θi2)
    (abs(sinθe) > 1 || !isfinite(sinθe)) && return (0.0, false)
    θe = asin(sinθe)
    return (θe, true)
end


# ── Full 2D ray trace through the prism pair ──────────────────────────────

"""
    _trace_ray(n, α, θi, A1, A2, l1)

Trace a single ray with refractive index `n` through the prism pair defined by
apex positions `A1` (Prism 1, apex up) and `A2` (Prism 2, apex down), apex angle
`α`, angle of incidence `θi`, and insertion `l1` (distance along the input face
of Prism 1 from its apex).

The ray enters Prism 1's input face at distance `l1` from the apex, refracts
through both prisms, and exits Prism 2's output face.

Returns a `NamedTuple` with fields:
- `OPL`:       total optical path length A→D (n₁·d₁ + d_free + n₂·d₂)
- `d1_phys`:   physical path inside Prism 1 (A→B)
- `d_free`:    free-space path between prisms (B→C)
- `d2_phys`:   physical path inside Prism 2 (C→D)
- `l2_actual`: actual insertion into Prism 2 (distance along entry face from apex)
- `A`, `B`, `C`, `D`: intersection points (2-element vectors)
- `d_output`:  output beam direction unit vector
"""
function _trace_ray(n, α, θi, A1, A2, l1)
    ha = α / 2

    # Prism 1 faces (apex up)
    f1_dir = [-sin(ha), -cos(ha)]
    f2_dir = [ sin(ha), -cos(ha)]
    f2_n   = [ cos(ha),  sin(ha)]

    # Prism 2 faces (apex down)
    g1_dir = [-sin(ha),  cos(ha)]
    g1_n   = [-cos(ha), -sin(ha)]
    g2_dir = [ sin(ha),  cos(ha)]
    g2_n   = [ cos(ha), -sin(ha)]

    # === PRISM 1 ===
    A = A1 + l1 * f1_dir

    sinθ1p = clamp(sin(θi) / n, -1.0, 1.0)
    θ1p = asin(sinθ1p)
    n_inw_f1 = [cos(ha), -sin(ha)]
    t1       = [sin(ha),  cos(ha)]
    d_refr   = cos(θ1p) * n_inw_f1 + sin(θ1p) * t1

    # Intersect with face 2
    rhs = A1 - A
    det_val = d_refr[1] * (-f2_dir[2]) - d_refr[2] * (-f2_dir[1])
    t_AB = (rhs[1] * (-f2_dir[2]) - rhs[2] * (-f2_dir[1])) / det_val
    B = A + t_AB * d_refr
    d1_phys = t_AB

    # Exit from Prism 1
    θ2p = α - θ1p
    sinθ_exit = clamp(n * sin(θ2p), -1.0, 1.0)
    θ_exit = asin(sinθ_exit)
    tang_f2 = d_refr - dot(d_refr, f2_n) * f2_n
    tn_f2 = norm(tang_f2)
    if tn_f2 > 1e-15
        tang_f2 = tang_f2 / tn_f2
    end
    d_exit = cos(θ_exit) * f2_n + sin(θ_exit) * tang_f2

    # === FREE SPACE B → C ===
    rhs2 = A2 - B
    det_val2 = d_exit[1] * (-g1_dir[2]) - d_exit[2] * (-g1_dir[1])
    t_BC = (rhs2[1] * (-g1_dir[2]) - rhs2[2] * (-g1_dir[1])) / det_val2
    s_C  = (d_exit[1] * rhs2[2] - d_exit[2] * rhs2[1]) / det_val2
    C = B + t_BC * d_exit
    d_free = t_BC
    l2_actual = s_C

    # === PRISM 2 ===
    cos_inc_2 = -dot(d_exit, g1_n)
    θ_inc_2 = acos(clamp(cos_inc_2, -1.0, 1.0))
    sinθ_inc_2p = clamp(sin(θ_inc_2) / n, -1.0, 1.0)
    θ_inc_2p = asin(sinθ_inc_2p)

    n_inw_g1 = -g1_n
    tang_g1 = d_exit - dot(d_exit, g1_n) * g1_n
    tn = norm(tang_g1)
    if tn > 1e-15
        tang_g1 = tang_g1 / tn
    end
    d_refr2 = cos(θ_inc_2p) * n_inw_g1 + sin(θ_inc_2p) * tang_g1

    # Intersect with exit face of Prism 2
    rhs3 = A2 - C
    det_val3 = d_refr2[1] * (-g2_dir[2]) - d_refr2[2] * (-g2_dir[1])
    t_CD = (rhs3[1] * (-g2_dir[2]) - rhs3[2] * (-g2_dir[1])) / det_val3
    D = C + t_CD * d_refr2
    d2_phys = t_CD

    # Exit from Prism 2
    θ_exit_2p = α - θ_inc_2p
    sinθ_final = clamp(n * sin(θ_exit_2p), -1.0, 1.0)
    θ_final = asin(sinθ_final)
    tang_g2 = d_refr2 - dot(d_refr2, g2_n) * g2_n
    tn2 = norm(tang_g2)
    if tn2 > 1e-15
        tang_g2 = tang_g2 / tn2
    end
    d_output = cos(θ_final) * g2_n + sin(θ_final) * tang_g2

    OPL = n * d1_phys + d_free + n * d2_phys

    return (OPL=OPL, d1_phys=d1_phys, d_free=d_free, d2_phys=d2_phys,
            l2_actual=l2_actual, A=A, B=B, C=C, D=D, d_output=d_output)
end


# ── Geometry conversion helpers ───────────────────────────────────────────

"""
    _L_lightcon_from_wh(w, h, α)

Lightcon separation from apex displacements:
`L = w cos(α/2) + h sin(α/2)`.
"""
_L_lightcon_from_wh(w, h, α) = w * cos(α/2) + h * sin(α/2)

"""
    _L_keller_from_wh(w, h)

Keller/Weiner separation (apex-to-apex Euclidean distance):
`L = sqrt(w² + h²)`.
"""
_L_keller_from_wh(w, h) = sqrt(w^2 + h^2)

"""
    _wh_from_L_lightcon(L_lc, α, θi, n_center, l1, l2)

Compute apex displacements `(w, h)` from the Lightcon-style separation `L_lc`,
apex angle `α`, angle of incidence `θi`, center-wavelength refractive index
`n_center`, and insertions `l1`, `l2`.

Traces the center-wavelength ray through Prism 1 and uses the constraint that
the perpendicular face separation equals `L_lc` and that the beam arrives at
distance `l2` from Prism 2's apex along its entry face.
"""
function _wh_from_L_lightcon(L_lc, α, θi, n_center, l1, l2)
    ha = α / 2

    f1_dir = [-sin(ha), -cos(ha)]
    f2_dir = [ sin(ha), -cos(ha)]
    f2_n   = [ cos(ha),  sin(ha)]

    A1 = [0.0, 0.0]
    A  = A1 + l1 * f1_dir

    θ1p   = asin(sin(θi) / n_center)
    n_inw = [ cos(ha), -sin(ha)]
    t1    = [ sin(ha),  cos(ha)]
    d_refr = cos(θ1p) * n_inw + sin(θ1p) * t1

    rhs = A1 - A
    det_val = d_refr[1] * (-f2_dir[2]) - d_refr[2] * (-f2_dir[1])
    t_AB = (rhs[1] * (-f2_dir[2]) - rhs[2] * (-f2_dir[1])) / det_val
    B = A + t_AB * d_refr

    θ2p = α - θ1p
    θ_exit = asin(n_center * sin(θ2p))
    tang = d_refr - dot(d_refr, f2_n) * f2_n
    tang = tang / norm(tang)
    d_exit = cos(θ_exit) * f2_n + sin(θ_exit) * tang

    L_perp = [cos(ha), -sin(ha)]
    numer = L_lc - dot(B, L_perp) - l2 * sin(α)
    denom = dot(d_exit, L_perp)
    t_BC = numer / denom

    C = B + t_BC * d_exit
    g1_dir = [-sin(ha), cos(ha)]
    A2 = C - l2 * g1_dir

    return (A2[1], -A2[2])  # (w, h) with h positive downward
end

"""
    _wh_from_L_keller(L_k, α, θi, n_center, l1)

Compute apex displacements `(w, h)` from the Keller-style separation `L_k`
(apex-to-apex Euclidean distance), apex angle `α`, angle of incidence `θi`,
center-wavelength refractive index `n_center`, and insertion `l1`.

In the Keller/Weiner convention, the line joining the two prism apexes is
parallel to the center-wavelength beam between the prisms. The Prism 2 apex
is placed at distance `L_k` along this direction from Prism 1's apex.
"""
function _wh_from_L_keller(L_k, α, θi, n_center, l1)
    ha = α / 2

    # Compute the center-wavelength exit beam direction from Prism 1
    # (independent of l1 — same refraction angles regardless of where on the face)
    sinθ1p = clamp(sin(θi) / n_center, -1.0, 1.0)
    θ1p = asin(sinθ1p)
    θ2p = α - θ1p
    sinθ_exit = clamp(n_center * sin(θ2p), -1.0, 1.0)
    θ_exit = asin(sinθ_exit)

    # Face 2 normal and tangent for Prism 1
    f2_n = [cos(ha), sin(ha)]
    n_inw = [cos(ha), -sin(ha)]
    t1    = [sin(ha),  cos(ha)]
    d_refr = cos(θ1p) * n_inw + sin(θ1p) * t1
    tang_f2 = d_refr - dot(d_refr, f2_n) * f2_n
    tn = norm(tang_f2)
    if tn > 1e-15
        tang_f2 = tang_f2 / tn
    end
    d_exit = cos(θ_exit) * f2_n + sin(θ_exit) * tang_f2

    # Place Prism 2 apex at distance L_k along the exit beam direction
    A2 = Float64(L_k) * d_exit
    return (A2[1], -A2[2])  # (w, h) with h positive downward
end


# ── Phase computation using full ray trace ────────────────────────────────

"""
    _prism_pair_phase(ω, n_func, α, θi, l1, A1, A2;
                      double_pass=true, D_ref=nothing, d_input=nothing)

Compute the spectral phase `ϕ(ω)` accumulated through the prism pair using
full 2D ray tracing.

The prism pair is defined by apex positions `A1` (Prism 1, apex up) and `A2`
(Prism 2, apex down), apex angle `α`, angle of incidence `θi`, and insertion
`l1` (distance along the input face of Prism 1 from apex to beam).

For each frequency, the ray is traced through both prisms. The total optical
path length is corrected to a common output reference plane through the
center-wavelength exit point, ensuring a self-consistent spectral phase.

If `double_pass=true` (default), the phase is doubled for a retroreflected
geometry.

# Keywords
- `D_ref`: fixed reference exit point (2-element vector). If not provided,
  computed from the center of the valid frequency range. Must be fixed across
  all evaluations when computing derivatives.
- `d_input`: input beam direction unit vector. If not provided, computed from
  `θi` and `α`.

Returns `(ϕ, mask)` where `ϕ` is the phase array and `mask` is a `BitVector`
indicating which frequencies are valid (no total internal reflection).
"""
function _prism_pair_phase(ω, n_func, α, θi, l1, A1, A2;
                          double_pass=true, D_ref=nothing, d_input=nothing)
    λ = PhysData.wlfreq.(ω)
    Nω = length(ω)
    ϕ = zeros(Nω)
    mask = trues(Nω)

    ha = α / 2

    # Build TIR mask using fast single-prism trace
    for i in 1:Nω
        if !isfinite(λ[i]) || λ[i] <= 0
            mask[i] = false
            continue
        end
        ni = real(n_func(λ[i]))
        if !isfinite(ni) || ni <= 0
            mask[i] = false
            continue
        end
        θe1, valid1 = _prism_trace(θi, α, ni)
        if !valid1
            mask[i] = false
            continue
        end
        _, valid2 = _prism_trace(θe1, α, ni)
        if !valid2
            mask[i] = false
            continue
        end
    end

    valid_idx = findall(mask)
    if isempty(valid_idx)
        return ϕ, mask
    end

    # Compute reference point D_ref if not provided
    if isnothing(D_ref)
        mid_idx = valid_idx[length(valid_idx) ÷ 2 + 1]
        n_ref = real(n_func(λ[mid_idx]))
        ref_res = _trace_ray(n_ref, α, θi, A1, A2, l1)
        D_ref = ref_res.D
    end

    # Input beam direction (determined by θi and face 1 geometry; same for all ω)
    if isnothing(d_input)
        n_inw_f1 = [cos(ha), -sin(ha)]
        t1_tang  = [sin(ha),  cos(ha)]
        d_input  = cos(θi) * n_inw_f1 + sin(θi) * t1_tang
    end

    pass_factor = double_pass ? 2.0 : 1.0

    for i in valid_idx
        ni = real(n_func(λ[i]))
        res = _trace_ray(ni, α, θi, A1, A2, l1)
        # Correct to common output reference plane through D_ref
        correction = dot(D_ref - res.D, d_input)
        OPL_total = res.OPL + correction
        ϕ[i] = pass_factor * ω[i] / PhysData.c * OPL_total
    end

    return ϕ, mask
end

"""
    _prism_apex_positions(α, θi, n_center, l1, l2;
                          L=nothing, L_lightcon=nothing, w=nothing, h=nothing)

Determine the Prism 2 apex position `(w, h)` plus the resolved Prism 1 and
Prism 2 apex coordinates `A1`, `A2` from the user-supplied geometry option.

Exactly one geometry option must be provided:
- `L` (apex-to-apex, Keller convention)
- `L_lightcon` (perpendicular face separation, requires both `l1` and `l2`)
- `w` and `h` (direct apex displacements)

Returns `(w, h, A1, A2)`.
"""
function _prism_apex_positions(α, θi, n_center, l1, l2;
                               L=nothing, L_lightcon=nothing,
                               w=nothing, h=nothing)
    A1 = [0.0, 0.0]
    if L_lightcon !== nothing
        w_calc, h_calc = _wh_from_L_lightcon(
            Float64(L_lightcon), α, θi, n_center, Float64(l1), Float64(l2))
    elseif w !== nothing && h !== nothing
        w_calc, h_calc = Float64(w), Float64(h)
    elseif L !== nothing
        w_calc, h_calc = _wh_from_L_keller(Float64(L), α, θi, n_center, Float64(l1))
    else
        error("Must provide one of: L (apex-to-apex), L_lightcon, or (w, h)")
    end
    A2 = [w_calc, -h_calc]
    return (w_calc, h_calc, A1, A2)
end

"""
    prism_pair_GDD(λ, material, α, θi; double_pass=true,
                   P=1, T=PhysData.roomtemp, lookup=nothing)

Group-delay dispersion per unit Keller separation (s²/m) of a prism pair at
wavelength `λ` (in metres), for prism `material`, apex angle `α`, and angle
of incidence `θi` (all in radians). No prism insertion (`l1 = l2 = 0`).

By default returns the **double-pass** GDD (set `double_pass=false` for single).
The result is typically negative (anomalous dispersion from angular dispersion).

Uses full 2D ray tracing internally; the result includes all orders of the
geometry and material dispersion.

# Examples
```jldoctest
julia> α = Fields.mindev_apex(:SiO2, 800e-9);

julia> θi = Fields.brewster_angle(:SiO2, 800e-9);

julia> Fields.prism_pair_GDD(800e-9, :SiO2, α, θi) < 0  # negative GDD
true
```
"""
function prism_pair_GDD(λ, material, α, θi; double_pass=true,
                        P=1, T=PhysData.roomtemp, lookup=nothing)
    n_func = PhysData.ref_index_fun(material, P, T; lookup=lookup)
    ω0 = wlfreq(λ)
    n_center = real(n_func(λ))
    # Use L_keller = 1 m, l1 = 0 for per-unit-separation GDD
    _, _, A1, A2 = _prism_apex_positions(α, θi, n_center, 0.0, 0.0; L=1.0)
    # Pre-compute fixed reference point at λ for consistent derivatives
    ref_res = _trace_ray(n_center, α, θi, A1, A2, 0.0)
    D_ref_fixed = ref_res.D
    ha = α / 2
    d_input_fixed = cos(θi) * [cos(ha), -sin(ha)] + sin(θi) * [sin(ha), cos(ha)]
    function ϕ_at(ω_val)
        ϕ_arr, mask = _prism_pair_phase([ω_val], n_func, α, θi, 0.0, A1, A2;
                                         double_pass=double_pass,
                                         D_ref=D_ref_fixed, d_input=d_input_fixed)
        mask[1] ? ϕ_arr[1] : 0.0
    end
    Maths.derivative(ϕ_at, ω0, 2)
end

"""
    prop_prisms!(Eω, ω, material, α, L, θi, l1=0.0, l2=0.0, λ0=nothing;
                 L_lightcon=nothing, w=nothing, h=nothing, double_pass=true,
                 P=1, T=PhysData.roomtemp, lookup=nothing)
    prop_prisms!(Eω, grid, material, α, L, θi, l1=0.0, l2=0.0, λ0=nothing; ...)

Apply the spectral phase from a prism pair compressor to the frequency-domain
field `Eω`, computed by full 2D ray tracing.

# Geometry

The prism pair consists of two identical prisms with apex angle `α` in an
anti-parallel (apex-up / apex-down) configuration. The spectral phase is
computed from the wavelength-dependent optical path length through both
prisms and the free-space gap between them.

## Separation (provide ONE)

- **`L`** (positional): apex-to-apex Euclidean distance in metres (Keller/Weiner
  convention). The beam insertion into Prism 2 is determined by the ray trace
  (matched to `l1` at the center wavelength).
- **`L_lightcon`** (keyword): perpendicular distance between Prism 1's input
  face and Prism 2's output face (Lightcon convention). Requires both `l1`
  and `l2` to be specified.
- **`w`, `h`** (keyword pair): horizontal and vertical apex displacements
  (h positive downward).

Conversion: `L_lightcon = w cos(α/2) + h sin(α/2)`,
            `L_keller = sqrt(w² + h²)`.

## Insertion

- `l1`: distance along the input face of Prism 1, from apex to beam (metres).
  Default `0.0` (beam grazes apex, no glass traversed).
- `l2`: distance along the entry face of Prism 2, from apex to beam (metres).
  Default `0.0`. Only used when `L_lightcon` is specified.

# Arguments
- `Eω`: frequency-domain field (modified in place)
- `ω` or `grid`: angular frequency axis (rad/s) or an `AbstractGrid`
- `material`: prism material as a `Symbol` (e.g. `:SiO2`, `:BK7`, `:CaF2`)
- `α`: prism apex angle (radians); use [`mindev_apex`](@ref) for Brewster-cut
- `L`: apex-to-apex separation (metres); see Geometry section for alternatives
- `θi`: angle of incidence on Prism 1 input face (radians); use
  [`brewster_angle`](@ref) for Brewster incidence
- `l1`, `l2`: prism insertions (metres)
- `λ0`: if given (metres), overall phase and group delay at `λ0` are removed

# Keyword arguments
- `L_lightcon`: Lightcon-style separation (overrides `L`)
- `w`, `h`: direct apex displacements (override `L`)
- `double_pass`: if `true` (default), double the phase for retroreflected geometry
- `P`, `T`, `lookup`: forwarded to `PhysData.ref_index_fun`

Frequency components with total internal reflection are set to zero.

See also [`prop_prisms`](@ref), [`brewster_angle`](@ref), [`mindev_apex`](@ref),
[`prism_pair_GDD`](@ref), [`optcomp_prisms`](@ref).
"""
function prop_prisms!(Eω, ω, material, α, L, θi, l1=0.0, l2=0.0, λ0=nothing;
                      L_lightcon=nothing, w=nothing, h=nothing, double_pass=true,
                      P=1, T=PhysData.roomtemp, lookup=nothing)
    n_func = PhysData.ref_index_fun(material, P, T; lookup=lookup)

    # Determine center wavelength for geometry setup
    λ_geom = isnothing(λ0) ? PhysData.wlfreq(ω[length(ω) ÷ 2 + 1]) : λ0
    n_center = real(n_func(λ_geom))

    # Determine apex positions from whichever geometry option was provided
    Larg = (L_lightcon !== nothing || (w !== nothing && h !== nothing)) ? nothing : L
    _, _, A1, A2 = _prism_apex_positions(α, θi, n_center, l1, l2;
                                          L=Larg, L_lightcon=L_lightcon, w=w, h=h)

    # Compute a fixed D_ref and d_input at the geometry center wavelength,
    # used consistently for both the main phase array and any λ0 correction
    ref_res = _trace_ray(n_center, α, θi, A1, A2, Float64(l1))
    D_ref_fixed = ref_res.D
    ha = α / 2
    d_input_fixed = cos(θi) * [cos(ha), -sin(ha)] + sin(θi) * [sin(ha), cos(ha)]

    ϕ, mask = _prism_pair_phase(ω, n_func, α, θi, l1, A1, A2;
                                 double_pass=double_pass,
                                 D_ref=D_ref_fixed, d_input=d_input_fixed)

    if !isnothing(λ0)
        ω0 = wlfreq(λ0)
        function ϕ_at(ω_val)
            ϕ_s, m_s = _prism_pair_phase([ω_val], n_func, α, θi, l1, A1, A2;
                                          double_pass=double_pass,
                                          D_ref=D_ref_fixed, d_input=d_input_fixed)
            m_s[1] ? ϕ_s[1] : 0.0
        end
        ϕ0 = ϕ_at(ω0)
        dϕdω = Maths.derivative(ϕ_at, ω0, 1)
        ϕ[mask] .-= ϕ0 .+ dϕdω .* (ω[mask] .- ω0)
    end
    transfer = zeros(ComplexF64, length(ω))
    transfer[mask] .= exp.(-1im .* ϕ[mask])
    Eω .*= transfer
    Eω
end

prop_prisms!(Eω, grid::Grid.AbstractGrid, material, α, L, θi, l1=0.0, l2=0.0,
             λ0=nothing; kwargs...) = prop_prisms!(
    Eω, grid.ω, material, α, L, θi, l1, l2, λ0; kwargs...)

"""
    prop_prisms(Eω, args...; kwargs...)

Return a copy of `Eω` with the prism pair spectral phase applied.
For arguments, see [`prop_prisms!`](@ref).
"""
prop_prisms(Eω, args...; kwargs...) = prop_prisms!(copy(Eω), args...; kwargs...)


"""
    optcomp_prisms(Eω, grid, material, α, θi, l1, l2, min_separation, max_separation;
                   λ0=nothing, double_pass=true, P=1, T=PhysData.roomtemp, lookup=nothing)
    optcomp_prisms(Eω, grid, material, α, θi, l1=0.0, l2=0.0;
                   λ0=nothing, double_pass=true, bounds_factor=3, ...)

Maximise peak power by compression through a prism pair compressor.

Returns `(L_opt, Eω_compressed)` where `L_opt` is the optimum apex-to-apex
(Keller) prism separation in metres.

With explicit bounds `min_separation` and `max_separation`, the optimiser
searches directly within that range. Without bounds, [`optcomp_taylor`](@ref)
estimates the required GDD and [`prism_pair_GDD`](@ref) converts it to an
estimated separation; the search range is then `L_est / bounds_factor` to
`L_est * bounds_factor`.

See also [`prop_prisms!`](@ref), [`brewster_angle`](@ref), [`mindev_apex`](@ref).
"""
function optcomp_prisms(Eω::AbstractVecOrMat, grid, material, α, θi, l1, l2,
                        min_separation, max_separation;
                        λ0=nothing, double_pass=true,
                        P=1, T=PhysData.roomtemp, lookup=nothing)
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    EωFTL = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    ItFTL = _It(iFT(EωFTL, grid), grid)

    Eωnorm = Eω ./ sqrt(maximum(ItFTL))

    function f(L)
        Eωp = copy(Eωnorm)
        prop_prisms!(Eωp, grid.ω, material, α, L, θi, l1, l2, λ0;
                     double_pass=double_pass, P=P, T=T, lookup=lookup)
        Itp = _It(iFT(Eωp, grid), grid)
        1/maximum(Itp)
    end

    res = Optim.optimize(f, min_separation, max_separation)
    res.minimizer, prop_prisms(Eω, grid, material, α, res.minimizer, θi, l1, l2, λ0;
                               double_pass=double_pass, P=P, T=T, lookup=lookup)
end

function optcomp_prisms(Eω::AbstractVecOrMat, grid, material, α, θi, l1=0.0, l2=0.0;
                        λ0=nothing, double_pass=true, bounds_factor=3,
                        P=1, T=PhysData.roomtemp, lookup=nothing)
    if isnothing(λ0)
        Eω1 = Eω isa AbstractVector ? Eω : @view Eω[:, 1]
        λc = wlfreq(grid.ω[argmax(abs.(Eω1))])
    else
        λc = λ0
    end
    ϕs, _ = optcomp_taylor(Eω, grid, λc; order=2)
    GDD_needed = ϕs[3]
    GDD_per_L = prism_pair_GDD(λc, material, α, θi; double_pass=double_pass,
                                P=P, T=T, lookup=lookup)
    if !isfinite(GDD_per_L) || GDD_per_L == 0
        throw(ArgumentError(
            "prism pair GDD per unit separation is zero or non-finite "
            * "(material=$material, α=$α, θi=$θi); cannot estimate separation "
            * "automatically. Use the explicit (min_separation, max_separation) "
            * "method instead."))
    end
    L_est = abs(GDD_needed / GDD_per_L)
    L_est = max(L_est, 1e-4)
    min_separation = max(0.0, L_est / bounds_factor)
    max_separation = L_est * bounds_factor
    optcomp_prisms(Eω, grid, material, α, θi, l1, l2,
                   min_separation, max_separation;
                   λ0=λ0, double_pass=double_pass, P=P, T=T, lookup=lookup)
end

function optcomp_prisms(Eω, args...; kwargs...)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    dout = zeros(size(cidcs))
    for ci in cidcs
        di, Eωi = optcomp_prisms(Eω[:, :, ci], args...; kwargs...)
        out[:, :, ci] .= Eωi
        dout[ci] = di
    end
    dout, out
end


"""
    optcomp_taylor(Eω, grid, λ0; order=2)

Maximise the peak power of the field `Eω` by adding Taylor-expanded spectral phases up to
order `order`.
"""
function optcomp_taylor(Eω::AbstractVecOrMat, grid, λ0; order=2, boundfac=8)
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    EωFTL = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    ItFTL = _It(iFT(EωFTL, grid), grid)
    target = 1/maximum(ItFTL)

    Eωnorm = Eω ./ sqrt(maximum(ItFTL))

    function f(disp)
        # disp here is just the dispersion terms (2nd order and higher)
        ϕs = [0, 0, disp...]
        Eωp = prop_taylor(Eωnorm, grid, ϕs, λ0)
        Itp = _It(iFT(Eωp, grid), grid)
        1/maximum(Itp)
    end

    τ0FTL = Maths.fwhm(grid.t, ItFTL)/(2*sqrt(log(2)))
    τ0 = Maths.fwhm(grid.t, _It(iFT(Eω, grid), grid))/(2*sqrt(log(2)))

    ϕ2_0 = τ0FTL*sqrt(abs(τ0^2 - τ0FTL^2)) # GDD to stretch Gaussian from FTL to actual duration

    # for Gaussian with pure GDD, sqrt(ϕ2_0) is the FTL duration, so use that as guide
    bounds = boundfac*(sqrt(ϕ2_0) .^(2:order))
    srange = [(-bi, bi) for bi in bounds]
    res = BlackBoxOptim.bboptimize(f; SearchRange=srange,
                                   TraceMode=:silent, TargetFitness=target)
    ϕs = [0, 0, BlackBoxOptim.best_candidate(res)...]
    ϕs, prop_taylor(Eω, grid, ϕs, λ0)
end

function optcomp_taylor(Eω, grid, λ0; order=2)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    ϕsout = zeros(order+1, size(cidcs)...)
    for ci in cidcs
        ϕsi, Eωi = optcomp_taylor(Eω[:, :, ci], grid, λ0; order=order)
        out[:, :, ci] .= Eωi
        ϕsout[:, ci] .= ϕsi
    end
    ϕsout, out
end

_It(Et::AbstractVector, grid) = It(Et, grid)
_It(Et::AbstractMatrix, grid) = dropdims(sum(It(Et, grid); dims=2); dims=2)

"""
    optcomp_material(Eω, grid, material, λ0; kwargs...)

Maximise the peak power of the field `Eω` by linear propagation through the `material`.
Keyword arguments `kwargs` are the same as for [`prop_material`](@ref).
"""
function optcomp_material(Eω::AbstractVecOrMat, grid, material, λ0,
                          min_thickness, max_thickness; kwargs...)
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    EωFTL = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    ItFTL = _It(iFT(EωFTL, grid), grid)

    Eωnorm = Eω ./ sqrt(maximum(ItFTL))

    prop! = propagator_material(material; kwargs...)

    function f(d)
        # d is the material insertion
        Eωp = copy(Eωnorm)
        prop!(Eωp, grid.ω, d, λ0)
        Itp = _It(iFT(Eωp, grid), grid)
        1/maximum(Itp)
    end

    # res = BlackBoxOptim.bboptimize(f; SearchRange=[srange],
    #                                TraceMode=:silent, TargetFitness=target)
    # prop_material(Eω, grid, material, BlackBoxOptim.best_candidate(res), λ0; kwargs...)
    res = Optim.optimize(f, min_thickness, max_thickness)
    res.minimizer, prop_material(Eω, grid, material, res.minimizer, λ0; kwargs...)
end

function optcomp_material(Eω, args...; kwargs...)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    dout = zeros(size(cidcs))
    for ci in cidcs
        di, Eωi = optcomp_material(Eω[:, :, ci], args...; kwargs...)
        out[:, :, ci] .= Eωi
        dout[ci] = di
    end
    dout, out
end


"""
    optcomp_gratings(Eω, grid, Λ, m, θi, min_separation, max_separation; λ0=nothing)
    optcomp_gratings(Eω, grid, Λ, m, θi; λ0=nothing, bounds_factor=3)

Maximise the peak power of the field `Eω` by compression through a four grating
compressor with grating period `Λ` (in metres), diffraction order `m` (dimensionless
integer) and angle of incidence `θi` (in radians). The optimum grating separation (in
metres) is returned.

If `min_separation` and `max_separation` are given, the optimiser searches within these
bounds directly. If they are omitted, [`optcomp_taylor`](@ref) is used to estimate the
required GDD and the Treacy formula ([`grating_GDD`](@ref)) converts this to an estimated
grating separation; the search bounds are then set to `L_est / bounds_factor` and
`L_est * bounds_factor`.

If the central wavelength `λ0` (in metres) is given, the overall phase and group delay
at `λ0` are removed from the grating phase (see [`prop_gratings!`](@ref)).
"""
function optcomp_gratings(Eω::AbstractVecOrMat, grid, Λ, m, θi,
                          min_separation, max_separation; λ0=nothing)
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    EωFTL = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    ItFTL = _It(iFT(EωFTL, grid), grid)

    Eωnorm = Eω ./ sqrt(maximum(ItFTL))

    function f(L)
        # L is the grating separation
        Eωp = copy(Eωnorm)
        prop_gratings!(Eωp, grid.ω, Λ, L, m, θi, λ0)
        Itp = _It(iFT(Eωp, grid), grid)
        1/maximum(Itp)
    end

    res = Optim.optimize(f, min_separation, max_separation)
    res.minimizer, prop_gratings(Eω, grid, Λ, res.minimizer, m, θi, λ0)
end

function optcomp_gratings(Eω::AbstractVecOrMat, grid, Λ, m, θi;
                          λ0=nothing, bounds_factor=3)
    # Determine central wavelength for Taylor estimate
    if isnothing(λ0)
        Eω1 = Eω isa AbstractVector ? Eω : @view Eω[:, 1]
        λc = wlfreq(grid.ω[argmax(abs.(Eω1))])
    else
        λc = λ0
    end
    # Use Taylor expansion to estimate the required GDD
    ϕs, _ = optcomp_taylor(Eω, grid, λc; order=2)
    GDD_needed = ϕs[3]
    GDD_per_L = grating_GDD(λc, Λ, m, θi)
    if !isfinite(GDD_per_L) || GDD_per_L == 0
        throw(ArgumentError(
            "grating GDD per unit separation is zero or non-finite "
            * "(Λ=$Λ, m=$m, θi=$θi); cannot estimate separation automatically. "
            * "Use the explicit (min_separation, max_separation) method instead."))
    end
    L_est = abs(GDD_needed / GDD_per_L)
    L_est = max(L_est, 1e-4) # at least 0.1 mm to avoid degenerate bounds
    min_separation = max(0.0, L_est / bounds_factor)
    max_separation = L_est * bounds_factor
    optcomp_gratings(Eω, grid, Λ, m, θi, min_separation, max_separation; λ0)
end

function optcomp_gratings(Eω, args...; kwargs...)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    dout = zeros(size(cidcs))
    for ci in cidcs
        di, Eωi = optcomp_gratings(Eω[:, :, ci], args...; kwargs...)
        out[:, :, ci] .= Eωi
        dout[ci] = di
    end
    dout, out
end


"""
    optfield_cep(Eω, grid)

Find the value of the absolute phase which produces the maximal field strength in the time
domain.
"""
function optfield_cep(Eω::AbstractVector, grid)
    res = Optim.optimize(-π, π) do ϕ
        Et = real(iFT(Eω*exp(1im*ϕ), grid))
        1/maximum(Et)
    end

    res.minimizer, Eω*exp(1im*res.minimizer)
end

function optfield_cep(Eω::AbstractMatrix, grid; mode=1)
    res = Optim.optimize(-π, π) do ϕ
        Et = real(iFT(Eω[:, mode]*exp(1im*ϕ), grid))
        1/maximum(Et)
    end

    res.minimizer, Eω*exp(1im*res.minimizer)
end

function optfield_cep(Eω, grid; mode=1)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    ϕout = zeros(size(cidcs))
    for ci in cidcs
        ϕi, Eωi = optfield_cep(Eω[:, :, ci], grid; mode)
        out[:, :, ci] .= Eωi
        ϕout[ci] = ϕi
    end
    ϕout, out
end
end
