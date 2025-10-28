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
    DataField(dat[:, 1]*2π, dat[:, 2], dat[:, 3]; energy, ϕ)
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
field `E` into a `mode`. The index `i` species the mode index. The temporal fields are 
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
    prop_gratings!(Eω, ω, Λ, L, m, θi)
    prop_gratings!(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi)

Add the spectral phase acquired after passing through a four grating compressor
with grating period `Λ`, grating separation `L`, diffraction order `m` and angle
of incidence `θi`. The sampling axis of `Eω` can be given either as an
`AbstractGrid` or the frequency axis `ω`.
"""
function prop_gratings!(Eω, ω, Λ, L, m, θi)
    λ = PhysData.wlfreq.(ω)
    mask = @. abs(m*λ/Λ + sin(θi)) <= 1
    θm = @. asin(m*λ[mask]/Λ + sin(θi))
    x = @. L*tan(θm)
    ϕg = @. π - m*2*π*x/Λ
    ϕ = zero(λ)
    ϕ[mask] .= @. 2*(ω[mask]*L/PhysData.c + ϕg)
    Eω .*= exp.(-1im.*ϕ)
end

prop_gratings!(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi) = prop_gratings!(Eω, grid.ω, Λ, L, m, θi)

"""
    prop_gratings(Eω, ω, Λ, L, m, θi)
    prop_gratings(Eω, grid::Grid.AbstractGrid, Λ, L, m, θi)

Return a copy of the frequency-domain field `Eω` with the additional spectral phase
acquired after passing through a four grating compressor with grating period `Λ`, grating
separation `L`, diffraction order `m` and angle of incidence `θi`.
The sampling axis of `Eω` can be given either as an `AbstractGrid` or the frequency axis `ω`.
"""
prop_gratings(Eω, args...) = prop_gratings!(copy(Eω), args...)


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

    ϕ2_0 = τ0FTL*sqrt(τ0^2 - τ0FTL^2) # GDD to stretch Gaussian from FTL to actual duration

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
    target = 1/maximum(ItFTL)

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
    optcomp_gratings(Eω, grid, Λ, m, θi, min_separation, max_separation)

Maximise the peak power of the field `Eω` by compression through a four grating
compressor with grating period `Λ`, diffraction order `m` and angle of incidence `θi`.
The optimum grating separation is returned, lying between `min_separation` and
`max_separation`. 
"""
function optcomp_gratings(Eω::AbstractVecOrMat, grid, Λ, m, θi,
                          min_separation, max_separation)
    τ = length(grid.t) * (grid.t[2] - grid.t[1])/2
    EωFTL = abs.(Eω) .* exp.(-1im .* grid.ω .* τ)
    ItFTL = _It(iFT(EωFTL, grid), grid)
    target = 1/maximum(ItFTL)

    Eωnorm = Eω ./ sqrt(maximum(ItFTL))

    function f(L)
        # L is the grating separation
        Eωp = copy(Eωnorm)
        prop_gratings!(Eωp, grid.ω, Λ, L, m, θi)
        Itp = _It(iFT(Eωp, grid), grid)
        1/maximum(Itp)
    end

    res = Optim.optimize(f, min_separation, max_separation)
    res.minimizer, prop_gratings(Eω, grid, Λ, res.minimizer, m, θi)
end

function optcomp_gratings(Eω, args...)
    out = similar(Eω)
    cidcs = CartesianIndices(size(Eω)[3:end])
    dout = zeros(size(cidcs))
    for ci in cidcs
        di, Eωi = optcomp_gratings(Eω[:, :, ci], args...)
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
