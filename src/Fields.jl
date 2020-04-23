module Fields
import Luna
import Luna: Grid, Maths, PhysData
import NumericalIntegration: integrate, SimpsonEven
import Random: MersenneTwister, AbstractRNG
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

Construct a (super)Gaussian shaped pulse with intensity/power FWHM `τfwhm`,
superGaussian parameter `m=1` and other parameters as defined for [`PulseField`](@ref).
"""
function GaussField(;λ0, τfwhm, energy, ϕ=0.0, τ0=0.0, m=1)
    PulseField(λ0, energy, ϕ, τ0, t -> Maths.gauss(t, fwhm=τfwhm, power=2*m))
end

"""
    SechField(;λ0, energy, τw=nothing, τfwhm=nothing, ϕ=0.0, τ0=0.0)

Construct a Sech^2(t/τw) shaped pulse, specifying either the
natural width `τw`, or the intensity/power FWHM `τfwhm`.
Other parameters are as defined for [`PulseField`](@ref).
"""
function SechField(;λ0, energy, τw=nothing, τfwhm=nothing, ϕ=0.0, τ0=0.0)
    if τfwhm != nothing
        if τw != nothing
            error("only one of `τw` or `τfwhm` can be specified")
        else
            τw = τfwhm/(2*log(1 + sqrt(2)))
        end
    elseif τw == nothing
        error("one of `τw` or `τfwhm` must be specified")
    end
    PulseField(λ0, energy, ϕ, τ0, t -> sech(t/τw)^2)
end

"""
    make_Et(p::PulseField, grid)

Create electric field for `PulseField`, either the field (for `RealGrid`) or the envelope (for `EnvGrid`)
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
function (p::PulseField)(Eω, grid, energy_t, FT)
    Et = make_Et(p, grid)
    Eω .+= FT * (sqrt(p.energy)/sqrt(energy_t(Et)) .* Et)
end

"""
    ShotNoise(seed=nothing)

Creates one photon per mode quantum noise (shot noise) to add to an input field.
The random `seed` can optionally be provided.
"""
struct ShotNoise{rT<:AbstractRNG} <: TimeField
    rng::rT
end

function ShotNoise(seed=nothing)
    ShotNoise(MersenneTwister(seed))
end

"""
    (s::ShotNoise)(Eω, grid)

Add shotnoise to `Eω` for the provided `grid`. The random `seed` can optionally be provided.
The optional parameters `energy_t` and `FT` are unused and are present for interface
compatibility with [`TimeField`](@ref).
"""
function (s::ShotNoise)(Eω, grid::Grid.RealGrid, energy_t=nothing, FT=nothing)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = @. sqrt(PhysData.ħ*grid.ω/δω)
    rFFTamp = sqrt(2π)/2δt*amp
    φ = 2π*rand(s.rng, size(Eω)...)
    @. Eω += rFFTamp * exp(1im*φ)
end

function (s::ShotNoise)(Eω, grid::Grid.EnvGrid, energy_t=nothing, FT=nothing)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    amp[grid.sidx] = @. sqrt(PhysData.ħ*grid.ω[grid.sidx]/δω)
    FFTamp = sqrt(2π)/δt*amp
    φ = 2π*rand(s.rng, size(Eω)...)
    @. Eω += FFTamp * exp(1im*φ)
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
    (s::SpatioTemporalField)(Eωk, grid, spacegrid, energy_t, FT)

Add the field to `Eωk` for the provided `grid`, `spacegrid` `energy_t` function
and Fourier transform `FT`
"""
function (s::SpatioTemporalField)(Eωk, grid, spacegrid, energy_t, FT)
    Etr = make_Etr(s, grid, spacegrid)
    Etr .*= sqrt(s.energy)/sqrt(energy_t(Etr))
    lEωk = transform(spacegrid, FT, Etr)
    if s.propz != 0.0
        prop!(lEωk, s.propz, grid, spacegrid)
    end
    Eωk .+= lEωk
end

# TODO: this for FreeGrid
function prop!(Eωk, z, grid, q::Hankel.QDHT)
    kzsq = @. (grid.ω/PhysData.c)^2 - (q.k^2)'
    kzsq[kzsq .< 0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω/PhysData.c))
end

"Calculate energy from modal field E(t)"
function energyfuncs(grid::Grid.RealGrid)
    function energy_t(Et)
        Eta = Maths.hilbert(Et)
        return integrate(grid.t, abs2.(Eta), SimpsonEven())
    end

    prefac = 2π/(grid.ω[end]^2)
    function energy_ω(Eω)
        prefac*integrate(grid.ω, abs2.(Eω), SimpsonEven())
    end
    return energy_t, energy_ω
end

function energyfuncs(grid::Grid.EnvGrid)
    function energy_t(Et)
        return integrate(grid.t, abs2.(Et), SimpsonEven())
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
