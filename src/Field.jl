module Field
import Luna
import Luna: Grid, Maths, PhysData

struct PulseField
    λ0::Float64         # central wavelength
    energy::Float64     # pulse energy
    ϕ::Float64          # CEO phase
    τ0::Float64         # delay from grid time 0
    Itshape::iT         # function to get shape of intensity/power in time domain
end

function GaussField(;λ0, τfwhm, energy, ϕ=0.0, τ0=0.0)
    PulseField(λ0, energy, ϕ, τ0, t -> Maths.gauss(t, fwhm=τfwhm))
end

function (p::PulseField)(Eω, grid::Grid.RealGrid, energy_t, FT)
    t = grid.t .- p.τ0
    ω0 = PhysData.wlfreq(p.λ0)
    Et = @. sqrt(p.Itshape(t))*cos(ω0*t + p.ϕ)
    Eω .+= FFTW.rfft(sqrt(p.energy)/sqrt(energy_t(Et)) .* Et)
end

function (p::PulseField)(Eω, grid::Grid.EnvGrid, energy_t, FT)
    t = grid.t .- p.τ0
    Δω = PhysData.wlfreq(p.λ0) - grid.ω0
    Et = @. sqrt(p.Itshape(t))*exp(im*(p.ϕ + Δω*t))
    Eω = FFTW.fft(sqrt(p.energy)/sqrt(energy_t(Et)) .* Et)
end

function shotnoise!(Eω, grid::Grid.RealGrid, energy_t=nothing, FT=nothing; seed=nothing)
    rng = MersenneTwister(seed)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = @. sqrt(PhysData.ħ*grid.ω/δω)
    rFFTamp = sqrt(2π)/2δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += rFFTamp * exp(1im*φ)
end

function shotnoise!(Eω, grid::Grid.EnvGrid, energy_t=nothing, FT=nothing; seed=nothing)
    rng = MersenneTwister(seed)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    amp[grid.sidx] = @. sqrt(PhysData.ħ*grid.ω[grid.sidx]/δω)
    FFTamp = sqrt(2π)/δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += FFTamp * exp(1im*φ)
end

"Calculate energy from modal field E(t)"
energy_modal() = _energy_modal
_energy_modal(t, Et::Array{T, N}) where T <: Real where N = _energy_modal(t, Maths.hilbert(Et))
_energy_modal(t, Et::Array{T, N}) where T <: Complex where N = abs(integrate(t, abs2.(Et), SimpsonEven()))


function energy_radial(grid::Grid.RealGrid, q)
    function energy_t(t, Et)
        Eta = Maths.hilbert(Et)
        tintg = integrate(t, abs2.(Eta), SimpsonEven())
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(tintg, q)
    end

    prefac = 2π*PhysData.c*PhysData.ε_0/2 * 2π/(grid.ω[end]^2)
    function energy_ω(ω, Eω)
        ωintg = integrate(ω, abs2.(Eω), SimpsonEven())
        return prefac*Hankel.integrateK(ωintg, q)
    end
    return energy_t, energy_ω
end

function energy_radial(grid::Grid.EnvGrid, q)
    function energy_t(t, Et)
        tintg = integrate(t, abs2.(Et), SimpsonEven())
        return 2π*PhysData.c*PhysData.ε_0/2 * Hankel.integrateR(tintg, q)
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    prefac = 2π*PhysData.c*PhysData.ε_0/2 * 2π*δω/(Δω^2)
    function energy_ω(ω, Eω)
        ωintg = dropdims(sum(abs2.(Eω); dims=1), dims=1)
        return prefac*Hankel.integrateK(ωintg, q)
    end
    return energy_t, energy_ω
end

function energy_free(grid::Grid.RealGrid, xygrid)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    δt = grid.t[2] - grid.t[1]
    prefac_t = PhysData.c*PhysData.ε_0/2 * δx * δy * δt
    function energy_t(t, Et)
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
    energy_ω(ω, Eω) = prefac * sum(abs2.(Eω))

    return energy_t, energy_ω
end

function energy_free(grid::Grid.EnvGrid, xygrid)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    δt = grid.t[2] - grid.t[1]
    prefac_t = PhysData.c*PhysData.ε_0/2 * δx * δy * δt
    function energy_t(t, Et)
        return  prefac_t * sum(abs2.(Et)) 
    end

    δω = grid.ω[2] - grid.ω[1]
    Δω = length(grid.ω)*δω
    δkx = xygrid.kx[2] - xygrid.kx[1]
    Δkx = length(xygrid.kx)*δkx
    δky = xygrid.ky[2] - xygrid.ky[1]
    Δky = length(xygrid.ky)*δky
    prefac = PhysData.c*PhysData.ε_0/2 * 2π*δω/(Δω^2) * 2π*δkx/(Δkx^2) * 2π*δky/(Δky^2)
    energy_ω(ω, Eω) = prefac * sum(abs2.(Eω))

    return energy_t, energy_ω
end

end