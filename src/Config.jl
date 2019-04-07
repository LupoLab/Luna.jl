module Configuration
"Input configuration"
abstract type Input end

struct GaussInput <: Input
    mode::Symbol
    energy::Float64
    duration::Float64
    wavelength::Float64
    phases::Array{Float64, 1}
end

function GaussInput(; duration, energy, wavelength, phases=[], mode=:HE11)
    return GaussInput(mode, energy, duration, wavelength, phases)
end

struct SechInput <: Input
    mode::Symbol
    energy::Float64
    duration::Float64
    wavelength::Float64
    phases::Array{Float64, 1}
end

function SechInput(; duration, energy, wavelength, phases=[], mode=:HE11)
    return SechInput(mode, energy, duration, wavelength, phases)
end

struct FileInput <: Input
    mode::Symbol
    energy::Float64
    file::String
    phases::Array{Float64, 1}
end

"Geometry configuration"
abstract type Geometry end

abstract type Fill end

struct StaticFill <: Fill
    gas::Symbol
    pressure::Float64
end

struct GradientFill <: Fill
    gas::Symbol
    direction::Integer
    high_pressure::Float64
    low_pressure::Float64
end

function GradientFill(; gas, direction, high, low=0)
    return GradientFill(gas, direction, high, low)
end

struct Capillary{T} <: Geometry where T <: Fill
    radius::Float64
    length::Float64
    single_mode::Bool
    modes::NTuple{N, Symbol} where N
    fill::T
end

function Capillary(; radius, length, fill, single_mode=true, modes=(:HE11,))
    return Capillary(radius, length, single_mode, modes, fill)
end

struct HCPCF <: Geometry
    radius::Float64
    length::Float64
    loss::Float64
    fill::T where T <: Fill
end

"Nonlinearity configuration"
abstract type Nonlinear end

struct GasNonlinear <: Nonlinear
    kerr::Bool
    onlySPM::Bool
    plasma::Bool
    ionrate::Symbol
    plasma_loss::Bool
    plasma_phase::Bool
end

function GasNonlinear(; kerr=true, onlySPM=false, plasma=false, ionrate=:ADK,
                      plasma_loss=true, plasma_phase=true)
    return GasNonlinear(kerr, onlySPM, plasma, ionrate, plasma_loss, plasma_phase)
end

"Grid configuration"
abstract type Grid end

struct RealGrid <: Grid
    λ_lims::NTuple{2, Float64} # Desired wavelength limits in metres
    tmax::Float64 # Maximum time in window (window goes from -tmax to +tmax)
    δt::Float64 # oversampled time sampling
    referenceλ::Float64 # reference wavelength in metres (for group velocity etc)
    apod_width::Float64 # width/smoothness of spectral apodisation filter
end

function RealGrid(; λ_lims=(150e-9, 4000e-9), tmax=1e-12, δt=1e-12, referenceλ=800e-9,
                  apod_width=2e14)
    return RealGrid(λ_lims, tmax, δt, referenceλ, apod_width)
end

"Overall config"
struct Config
    input::NTuple{N, T} where N where T<:Input
    geometry::Geometry
    nonlinear::Nonlinear
    grid::Grid
end

function Config(input::Input, geometry, nonlinear, grid)
    return Config((input,), geometry, nonlinear, grid)
end

end