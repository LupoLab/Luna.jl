module Fields
import Luna
import Luna: Grid, Maths, PhysData
import NumericalIntegration: integrate, SimpsonEven
import Random: AbstractRNG, GLOBAL_RNG
import Statistics: mean
import Hankel

abstract type AbstractField end
abstract type TimeField <: AbstractField end

"""
    PulseField(λ0, energy, ϕ, τ0, Itshape)

Represents a temporal pulse with shape defined by `Itshape`.

# Fields
- `λ0::Float64`: the central field wavelength
- `energy::Float64`: the pulse energy
- `ϕ::Float64`: the CEO phase
- `τ0::Float64`: the temproal shift from grid time 0
- `Itshape`: a callable `f(t)` to get the shape of the intensity/power in the time domain
"""
struct PulseField{iT} <: TimeField
    λ0::Float64
    energy::Float64
    ϕ::Float64
    τ0::Float64
    Itshape::iT
end

"""
    GaussField(;λ0, τfwhm, energy, ϕ=0.0, τ0=0.0, m=1)

Construct a (super)Gaussian shaped pulse with intensity/power FWHM `τfwhm`, either
`energy` or peak `power` specified, superGaussian parameter `m=1` and other parameters
as defined for [`PulseField`](@ref).
"""
function GaussField(;λ0, τfwhm, energy=nothing, power=nothing, ϕ=0.0, τ0=0.0, m=1)
    if power != nothing
        if energy != nothing
            error("only one of `energy` or `power` can be specified")
        else
            energy = power*τfwhm*sqrt(pi/log(16))
        end
    elseif energy == nothing
        error("one of `energy` or `power` must be specified")
    end
    PulseField(λ0, energy, ϕ, τ0, t -> Maths.gauss(t, fwhm=τfwhm, power=2*m))
end

"""
    SechField(;λ0, energy, τw=nothing, τfwhm=nothing, ϕ=0.0, τ0=0.0)

Construct a Sech^2(t/τw) shaped pulse, specifying either the
natural width `τw`, or the intensity/power FWHM `τfwhm`, and either
`energy` or peak `power` specified.
Other parameters are as defined for [`PulseField`](@ref).
"""
function SechField(;λ0, energy=nothing, power=nothing, τw=nothing, τfwhm=nothing,
                    ϕ=0.0, τ0=0.0)
    if τfwhm != nothing
        if τw != nothing
            error("only one of `τw` or `τfwhm` can be specified")
        else
            τw = τfwhm/(2*log(1 + sqrt(2)))
        end
    elseif τw == nothing
        error("one of `τw` or `τfwhm` must be specified")
    end
    if power != nothing
        if energy != nothing
            error("only one of `energy` or `power` can be specified")
        else
            energy = 2*power*τw
        end
    elseif energy == nothing
        error("one of `energy` or `power` must be specified")
    end
    PulseField(λ0, energy, ϕ, τ0, t -> sech(t/τw)^2)
end

"""
    make_Et(p::PulseField, grid)

Create electric field for `PulseField`, either the field (for `RealGrid`) or
the envelope (for `EnvGrid`)
"""
function make_Et(p::PulseField, grid::Grid.RealGrid)
    t = grid.t .- p.τ0
    ω0 = PhysData.wlfreq(p.λ0)
    @. sqrt(p.Itshape(t))*cos(ω0*t + p.ϕ)
end

function make_Et(p::PulseField, grid::Grid.EnvGrid)
    t = grid.t .- p.τ0
    Δω = PhysData.wlfreq(p.λ0) - grid.ω0
    @. sqrt(p.Itshape(t))*exp(im*(p.ϕ + Δω*t))
end

"""
    (p::PulseField)(Eω, grid, energy_t, FT)

Add the field to `Eω` for the provided `grid`, `energy_t` function and Fourier transform `FT`
"""
function (p::PulseField)(grid, FT)
    Et = make_Et(p, grid)
    energy_t = Fields.energyfuncs(grid)[1]
    FT * (sqrt(p.energy)/sqrt(energy_t(Et)) .* Et)
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
function (c::CWField)(grid, FT)
    Et = FT \ c.Aωfunc.(grid.ω)
    # we scale before windowing to make sure we scale full average
    Et .*= sqrt(c.Pavg) / sqrt(mean(It(Et, grid)))
    Et .*= grid.twin
    FT * Et
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

# TODO: this for FreeGrid
function prop!(Eωk, z, grid, q::Hankel.QDHT)
    kzsq = @. (grid.ω/PhysData.c)^2 - (q.k^2)'
    kzsq[kzsq .< 0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω/PhysData.c))
end

function It(Et, grid::Grid.RealGrid)
    Eta = Maths.hilbert(Et)
    abs2.(Eta)
end

function It(Et, grid::Grid.EnvGrid)
    abs2.(Et)
end

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

end
