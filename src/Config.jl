module Config

"Input configuration"
abstract type Input end

struct GaussInput <: Input
    mode::Symbol
    energy::Float64
    duration::Float64
    wavelength::Float64
    phases::Array{Float64, N} where N
end

struct SechInput <: Input
    mode::Symbol
    energy::Float64
    duration::Float64
    wavelength::Float64
    phases::Array{Float64, N} where N
end

struct FileInput <: Input
    mode::Symbol
    energy::Float64
    file::String
    phases::Array{Float64, N} where N
end

"Geometry configuration"
abstract type Geometry end

abstract type Fill end

struct StaticFill <: Fill
    pressure::Float64
    gas::Symbol
end

struct GradientFill <: Fill
    direction::Integer
    high_pressure::Float64
    low_pressure::Float64
    gas::Symbol
end

struct Capillary <: Geometry
    radius::Float64
    length::Float64
    modes::NTuple{Symbol, N} where N
    fill::T where T <: Fill
end

struct HCPCF <: Geometry
    radius::Float64
    length::Float64
    loss::Float64
    fill::T where T <: Fill
end

"Nonlinearity configuration"
abstract type Nonlinear end

abstract type Ionrate end

struct GasNonlinear <: Nonlinear
    onlySPM::Bool
    ionrate::Symbol # TODO read up on value types again
    plasma_loss::Bool
    plasma_phase::Bool
end

end