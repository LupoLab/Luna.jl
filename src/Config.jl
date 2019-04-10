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

struct Capillary <: Geometry
    radius::Float64
    length::Float64
    single_mode::Bool # Make this separate type?
    modes::NTuple{N, Symbol} where N
end

function Capillary(; radius, length, single_mode=true, modes=(:HE11,))
    return Capillary(radius, length, single_mode, modes)
end

struct HCPCF <: Geometry
    radius::Float64
    length::Float64
    loss::Float64
end

"Medium configuration"
abstract type Medium end

struct StaticFill <: Medium
    gas::Symbol
    pressure::Float64
end

struct GradientFill <: Medium
    gas::Symbol
    direction::Integer
    high_pressure::Float64
    low_pressure::Float64
end

function GradientFill(; gas, direction, high, low=0)
    return GradientFill(gas, direction, high, low)
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

"Grid type for real-valued field"
struct RealGrid <: Grid
    λ_lims::NTuple{2, Float64} # Desired wavelength limits in metres
    trange::Float64 # Maximum time in window (window goes from -tmax/2 to +tmax/2)
    δt::Float64 # oversampled time sampling
    referenceλ::Float64 # reference wavelength in metres (for group velocity etc)
    apod_width::Float64 # width/smoothness of spectral apodisation filter
end

function RealGrid(; λ_lims=(150e-9, 4000e-9), trange=1e-12, δt=1e-12, referenceλ=800e-9,
                  apod_width=1e14)
    return RealGrid(λ_lims, trange, δt, referenceλ, apod_width)
end

"Grid type for envelope (complex-valued) field"
struct EnvGrid <: Grid
    λ_lims::NTuple{2, Float64} # Desired wavelength limits in metres
    trange::Float64 # Maximum time in window (window goes from -trange/2 to +trange/2)
    δt::Float64 # oversampled time sampling
    referenceλ::Float64 # reference wavelength in metres (for group velocity etc)
    apod_width::Float64 # width/smoothness of spectral apodisation filter
end

function EnvGrid(; λ_lims=(150e-9, 4000e-9), trange=1e-12, δt=1e-12, referenceλ=800e-9,
                  apod_width=2e14)
    return EnvGrid(λ_lims, trange, δt, referenceλ, apod_width)
end

"Overall config"
struct Config
    grid::Grid
    geometry::Geometry
    medium::Medium
    nonlinear::Nonlinear
    input::NTuple{N, T} where N where T<:Input
end

function Config(grid, geometry, medium, nonlinear, input::Input)
    return Config(grid, geometry, medium, nonlinear, (input,))
end

end