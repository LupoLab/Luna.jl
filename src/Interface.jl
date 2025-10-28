module Interface
using Luna
import Luna.PhysData: wlfreq, roomtemp
import Luna: Grid, Modes, Output, Fields
import Logging: @info, @debug

module Pulses

import Luna: Fields, Output, Processing, Capillary

export AbstractPulse, CustomPulse, GaussPulse, SechPulse, DataPulse, LunaPulse

abstract type AbstractPulse end

struct CustomPulse{fT<:Fields.TimeField} <: AbstractPulse
    mode::Symbol
    polarisation
    field::fT
end

"""
    CustomPulse(;λ0, energy=nothing, power=nothing, ϕ=Float64[],
                mode=:lowest, polarisation=:linear, propagator=nothing)

A custom pulse defined by a function for use with `prop_capillary`, with either energy or
peak power specified.

# Keyword arguments
- `λ0::Number`: the central wavelength
- `Itshape::function`: a function `I(t)`` which defines the intensity/power envelope of the
                       pulse as a function of time `t`. Note that the normalisation of this
                       envelope is irrelevant as it will be re-scaled by `energy` or `power`.
- `energy::Number`: the pulse energy.
- `power::Number`: the pulse peak power (**after** applying any spectral phases).
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...).
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`.
- `polarisation`: Can be `:linear`, `:x`, `:y`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
"""
function CustomPulse(;mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    CustomPulse(mode, polarisation,
                Fields.PropagatedField(propagator, Fields.PulseField(;kwargs...)))
end

struct GaussPulse{fT<:Fields.TimeField} <: AbstractPulse
    mode::Symbol
    polarisation
    field::fT
end

"""
    GaussPulse(;λ0, τfwhm, energy=nothing, power=nothing, ϕ=Float64[], m=1,
               mode=:lowest, polarisation=:linear, propagator=nothing)

A (super)Gaussian pulse for use with `prop_capillary`, with either energy or peak power
specified.

# Keyword arguments
- `λ0::Number`: the central wavelength.
- `τfwhm::Number`: the pulse duration (power/intensity FWHM).
- `energy::Number`: the pulse energy.
- `power::Number`: the pulse peak power (**after** applying any spectral phases).
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...).
- `m::Int`: super-Gaussian parameter (the power in the Gaussian exponent is 2m).
            Defaults to 1.
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`.
- `polarisation`: Can be `:linear`, `:x`, `:y`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
"""
function GaussPulse(;mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    GaussPulse(mode, polarisation,
               Fields.PropagatedField(propagator, Fields.GaussField(;kwargs...)))
end

struct SechPulse{fT<:Fields.TimeField} <: AbstractPulse
    mode::Symbol
    polarisation
    field::fT
end

"""
    SechPulse(;λ0, τfwhm=nothing, τw=nothing, energy=nothing, power=nothing, ϕ=Float64[],
               mode=:lowest, polarisation=:linear, propagator=nothing)

A sech²(τ/τw) pulse for use with `prop_capillary`, with either `energy` or peak `power`
specified, and duration given either as `τfwhm` or `τw`.

# Keyword arguments
- `λ0::Number`: the central wavelength.
- `τfwhm::Number`: the pulse duration (power/intensity FWHM).
- `τw::Number`: "natural" pulse duration of a sech²(τ/τw) pulse.
- `energy::Number`: the pulse energy.
- `power::Number`: the pulse peak power (**after** applying any spectral phases).
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...)
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`.
- `polarisation`: Can be `:linear`, `:x`, `:y`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
"""
function SechPulse(;mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    SechPulse(mode, polarisation,
              Fields.PropagatedField(propagator, Fields.SechField(;kwargs...)))
end

struct DataPulse{fT<:Fields.TimeField} <: AbstractPulse
    mode::Symbol
    polarisation
    field::fT
end

#TODO add peak power to DataPulses
"""
    DataPulse(ω, Iω, ϕω; energy, λ0=NaN, mode=:lowest, polarisation=:linear, propagator=nothing)
    DataPulse(ω, Eω; energy, λ0=NaN, mode=:lowest, polarisation=:linear, propagator=nothing)
    DataPulse(fpath; energy, λ0=NaN, mode=:lowest, polarisation=:linear, propagator=nothing)

A custom pulse defined by tabulated data to be used with `prop_capillary`.

# Data input options
- `ω, Iω, ϕω`: arrays of angular frequency `ω` (units rad/s), spectral energy density `Iω`
               and spectral phase `ϕω`. `ϕω` should be unwrapped.
- `ω, Eω`: arrays of angular frequency `ω` (units rad/s) and the complex frequency-domain
           field `Eω`.
- `fpath`: a string containing the path to a file which contains 3 columns:
    Column 1: frequency (units of Hertz)
    Column 2: spectral energy density
    Column 3: spectral phase (unwrapped)

# Keyword arguments
- `energy::Number`: the pulse energy
- `λ0::Number`: the central wavelength (optional; defaults to the centre of mass of the
                given spectral energy density).
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...) to be applied to the
                       pulse (in addition to any phase already present in the data).
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`.
- `polarisation`: Can be `:linear`, `:x`, `:y`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
"""
function DataPulse(ω::AbstractVector, Iω, ϕω;
                   mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    DataPulse(mode, polarisation,
              Fields.PropagatedField(propagator, Fields.DataField(ω, Iω, ϕω; kwargs...)))
end

function DataPulse(ω, Eω;
                   mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    DataPulse(mode, polarisation,
              Fields.PropagatedField(propagator, Fields.DataField(ω, Eω; kwargs...)))
end

function DataPulse(fpath;
                   mode=:lowest, polarisation=:linear, propagator=nothing, kwargs...)
    DataPulse(mode, polarisation,
              Fields.PropagatedField(propagator, Fields.DataField(fpath; kwargs...)))
end

"""
    LunaPulse(output; energy, λ0=NaN, mode=:lowest, polarisation=:linear, propagator=nothing)

A pulse defined to be used with `prop_capillary` which comes from a previous `Luna`
propagation simulation.

For multi-mode simulations, only the lowest-order modes is transferred.

# Arguments
- `output::AbstractOutput`: output from a previous `Luna` simulation.

# Keyword arguments
- `energy::Number`: the pulse energy. When transferring multi-mode simulations this defines the **total** energy.
- `scale_energy`: if given instead of `energy`, scale the field from `output` by this number. Defaults to 1, so giving `energy` is **not** required. For multi-mode simulations, this can also be a `Vector` with the same number of elements as the number of modes, in which case the energy of each mode is scaled by the corresponding number.
- `λ0::Number`: the central wavelength (optional; defaults to the centre of mass of the
                given spectral energy density).
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...) to be applied to the
                       pulse (in addition to any phase already present in the data).
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
"""
function LunaPulse(o::Output.AbstractOutput; energy=nothing, scale_energy=nothing, kwargs...)
    ω = o["grid"]["ω"]
    t = o["grid"]["t"]
    τ = length(t) * (t[2] - t[1])/2 # middle of old time window
    Eω = o["Eω"]
    if ndims(Eω) == 2
        # mode-averaged
        Eωm = Eω[:, end]
        eout = Processing.energy(o)[end]
        e = make_energies(energy, scale_energy, eout)
        return DataPulse(ω, Eωm .* exp.(1im .* ω .* τ); energy=e, kwargs...)
    elseif ndims(Eω) == 3
        # multi-mode
        modes = Processing.makemodes(o; warn_dispersion=false)
        symbols = makesymbol.(modes)
        eout = Processing.energy(o)[:, end]
        es = make_energies(energy, scale_energy, eout)
        return [DataPulse(ω, Eω[:, ii, end] .* exp.(1im .* ω .* τ); mode=symbols[ii], energy=es[ii], kwargs...) for ii in eachindex(modes)]
    end
end

makesymbol(mode::Capillary.MarcatiliMode) = Symbol("$(mode.kind)$(mode.n)$(mode.m)")

make_energies(energy::Number, scale_energy::Nothing, eout) = eout ./ sum(eout) .* energy
make_energies(energy::Nothing, scale_energy, eout) = eout .* scale_energy
make_energies(energy::Nothing, scale_energy::Nothing, eout) = eout

struct GaussBeamPulse{pT, NmT} <: AbstractPulse
    waist::Float64
    timepulse::pT
    polarisation
    Nmodes::NmT
end

"""
    GaussBeamPulse(waist, timepulse, Nmodes=:all)

A pulse whose shape in time is defined by the `timepulse::AbstractPulse`, and whose modal content is calculated by considering the overlap of an ideal Gaussian laser beam with 1/e² radius `waist` with the modes of the waveguide. `Nmodes` determines how many of the available modes to couple to. By default (`Nmodes=:all`) all modes are taken into account, but this can lead to numerical inaccuracies.
"""
function GaussBeamPulse(waist, timepulse, Nmodes=:all)
    GaussBeamPulse(waist, timepulse, timepulse.polarisation, Nmodes)
end

end


"""
    prop_capillary(radius, flength, gas, pressure; λ0, λlims, trange, kwargs...)

Simulate pulse propagation in a hollow fibre using the capillary model.

# Mandatory arguments
- `radius`: Core radius of the fibre. Can be a `Number` for constant radius, or a function
    `a(z)` which returns the `z`-dependent radius.
- `flength::Number`: Length of the fibre.
- `gas::Symbol`: Filling gas species.
- `pressure`: Gas pressure. Can be a `Number` for constant pressure, a 2-`Tuple` of `Number`s
    for a simple pressure gradient, or a `Tuple` of `(Z, P)` where `Z` and `P`
    contain `z` positions and the pressures at those positions.
- `λ0`: (keyword argument) the reference wavelength for the simulation. For simple
    single-pulse inputs, this is also the central wavelength of the input pulse.
- `λlims::Tuple{<:Number, <:Number}`: The wavelength limits for the simulation grid.
- `trange::Number`: The total width of the time grid. To make the number of samples a
    power of 2, the actual grid used is usually bigger.

# Grid options
- `envelope::Bool`: Whether to use envelope fields for the simulation. Defaults to `false`.
    By default, envelope simulations ignore third-harmonic generation.
    Plasma has not yet been implemented for envelope fields.
- `δt::Number`: Time step on the fine grid used for the nonlinear interaction. By default,
    this is determined by the wavelength grid. If `δt` is given **and smaller** than the
    required value, it is used instead.

# Input pulse options
A single pulse in the lowest-order mode can be specified by the keyword arguments below.
More complex inputs can be defined by a single `AbstractPulse` or a `Vector{AbstractPulse}`.
In this case, all keyword arguments except for `λ0` are ignored.

- `λ0`: Central wavelength
- `τfwhm`: The pulse duration as defined by the full width at half maximum.
- `τw`: The "natural" pulse duration. Only available if pulseshape is `sech`.
- `ϕ`: Spectral phases to be applied to the transform-limited pulse. Elements are
    the usual polynomial phases ϕ₀ (CEP), ϕ₁ (group delay), ϕ₂ (GDD), ϕ₃ (TOD), etc.
- `energy`: Pulse energy.
- `power`: Peak power **after any spectral phases are added**.
- `pulseshape`: Shape of the transform-limited pulse. Can be `:gauss` for a Gaussian pulse
    or `:sech` for a sech² pulse.
- `polarisation`: Polarisation of the input pulse. Can be `:linear` (default), `:x`, `:y`,
    `:circular`, or an ellipticity number -1 ≤ ε ≤ 1, where ε=-1 corresponds to left-hand circular,
    ε=1 to right-hand circular, and ε=0 to linear polarisation. The major axis for
    elliptical polarisation is always the y-axis.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
- `shotnoise`:  If `true` (default), one-photon-per-mode quantum noise is included.

# Modes options
- `modes`: Defines which modes are included in the propagation. Can be any of:
    - a single mode signifier (default: :HE11), which leads to mode-averaged propagation
        (as long as all inputs are linearly polarised).
    - a `Dict` mode signifier with keys `:kind`, `:n`, and `:m`, e.g. `Dict(:kind=>:HE, :n=>1, :m=>1)`
    - a `Tuple` of mode signifiers (`Symbol`s or `Dict`s), which leads to multi-mode propagation in those modes.
    - a `Number` `N` of modes, which simply creates the first `N` `HE` modes.
    Note that when elliptical or circular polarisation is included, each mode is present
    twice in the output, once for `x` and once for `y` polarisation.
- `model::Symbol`: Can be `:full`, which includes the full complex refractive index of the cladding
    in the effective index of the mode, or `:reduced`, which uses the simpler model more
    commonly seen in the literature. See `Luna.Capillary` for more details.
    Defaults to `:full`.
- `loss::Bool`: Whether to include propagation loss. Defaults to `true`.
- `temperature::Number`: Temperature of the gas in Kelvin. Defaults to room temperature.

# Nonlinear interaction options
- `kerr`: Whether to include the Kerr effect. Defaults to `true`.
- `raman`: Whether to include the Raman effect. Defaults to `false`.
- `plasma`: Can be one of
    - `:ADK` -- include plasma using the ADK ionisation rate.
    - `:PPT` -- include plasma using the PPT ionisation rate.
    - `true` (default) -- same as `:PPT`.
    - `false` -- ignore plasma.
    Note that plasma is only available for full-field simulations.
- `PPT_options::Dict{Symbol, Any}`: when using the PPT ionisation rate for the
    plasma nonlinearity, this allows for fine-tuning of the options in calculating
    the ionisation. See [`ionrate_fun_PPT`](@ref Ionisation.ionrate_fun_PPT) for possible
    keyword arguments.
- `preionfrac::Float64`: fraction of the gas that is pre-ionised before the pulse. Defaults to `0.0`.
    Note that this is a very simplistic model of pre-ionisation and should be used with
    caution.
- `thg::Bool`: Whether to include third-harmonic generation. Defaults to `true` for
    full-field simulations and to `false` for envelope simulations.
If `raman` is `true`, then the following options apply:
    - `rotation::Bool = true`: whether to include the rotational Raman contribution
    - `vibration::Bool = true`: whether to include the vibrational Raman contribution

# Output options
- `stats_kwargs::Dict{Symbol, Any}`: a dictionary of keyword arguments to `Stats.default`
- `saveN::Integer`: Number of points along z at which to save the field.
- `filepath`: If `nothing` (default), create a `MemoryOutput` to store the simulation results
    only in the working memory. If not `nothing`, should be a file path as a `String`,
    and the results are saved in a file at this location. If `scan` is passed, `filepath`
    determines the output **directory** for the scan instead.
- `scan`: A `Scan` instance defining a parameter scan. If `scan` is given`, a
    `Output.ScanHDF5Output` is used to automatically name and populate output files of
    the scan. `scanidx` must also be given.
- `scanidx`: Current scan index within a scan being run. Only used when `scan` is passed.
- `filename`: Can be used to to overwrite the scan name when running a parameter scan.
    The running `scanidx` will be appended to this filename. Ignored if no `scan` is given.
- `status_period::Number`: Interval (in seconds) between printed status updates.
"""
function prop_capillary(args...; status_period=5, kwargs...)
    Eω, grid, linop, transform, FT, output = prop_capillary_args(args...; kwargs...)
    Luna.run(Eω, grid, linop, transform, FT, output; status_period)
    output
end

"""
    prop_capillary_args(radius, flength, gas, pressure; λ0, λlims, trange, kwargs...)

Prepare to simulate pulse propagation in a hollow fibre using the capillary model. This
function takes the same arguments as `prop_capillary` but instead or running the
simulation and returning the output, it returns the required arguments for `Luna.run`,
which is useful for repeated simulations in an indentical fibre with different initial
conditions.
"""
function prop_capillary_args(radius, flength, gas, pressure;
                        λlims, trange, envelope=false, thg=nothing, δt=1,
                        λ0, τfwhm=nothing, τw=nothing, ϕ=Float64[],
                        power=nothing, energy=nothing,
                        pulseshape=:gauss, polarisation=:linear, propagator=nothing,
                        pulses=nothing,
                        shotnoise=true,
                        modes=:HE11, model=:full, loss=true,
                        radial_integral_rtol=1e-3,
                        raman=nothing, kerr=true, plasma=nothing,
                        stats_kwargs=Dict{Symbol, Any}(),
                        PPT_options=Dict{Symbol, Any}(), preionfrac=0.0,
                        rotation=true, vibration=true, temperature=roomtemp,
                        saveN=201, filepath=nothing,
                        scan=nothing, scanidx=nothing, filename=nothing)

    pol = needpol(polarisation, pulses) || needpol_modes(modes)
    @info "X+Y polarisation "* (pol ? "required." : "not required.")
    plasma = isnothing(plasma) ? !envelope : plasma
    thg = isnothing(thg) ? !envelope : thg

    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = makemode_s(modes, flength, radius, gas, pressure, temperature, model, loss, pol)
    check_orth(mode_s)
    density = makedensity(flength, gas, pressure, temperature)
    resp = makeresponse(grid, gas, raman, kerr, plasma, thg, pol, rotation, vibration,
                        PPT_options, preionfrac)
    inputs = makeinputs(mode_s, λ0, pulses, τfwhm, τw, ϕ,
                        power, energy, pulseshape, polarisation, propagator)
    inputs = shotnoise_maybe(inputs, mode_s, shotnoise)
    linop, Eω, transform, FT = setup(grid, mode_s, density, resp, inputs, pol,
                                     radial_integral_rtol, const_linop(radius, pressure))
    stats = Stats.default(grid, Eω, mode_s, linop, transform; gas=gas, stats_kwargs...)
    output = makeoutput(grid, saveN, stats, filepath, scan, scanidx, filename)

    saveargs(output; radius, flength, gas, pressure, λlims, trange, envelope, thg, δt,
        λ0, τfwhm, τw, ϕ, power, energy, pulseshape, polarisation, propagator, pulses, 
        shotnoise, modes, model, loss, raman, kerr, plasma, PPT_options, saveN, filepath, filename)

    return Eω, grid, linop, transform, FT, output
end

check_orth(mode::Modes.AbstractMode) = nothing
function check_orth(modes)
    if length(modes) > 1
        if !Modes.orthonormal(modes)
            ms = join(modes, "\n")
            error("The selected modes do not form an orthonormal set:\n$ms")
        end
    end
end

function saveargs(output; kwargs...)
    d = Dict{String, String}()
    for (k, v) in kwargs
        d[string(k)] = string(v)
    end
    output(d; group="prop_capillary_args")
end

function needpol(pol)
    if pol == :linear
        return false
    elseif pol in (:circular, :x, :y)
        return true
    else
        error("Polarisation must be :linear, :circular, :x/:y, or an ellipticity, not $pol")
    end
end

needpol(pol::Number) = true
needpol(pulse::Pulses.AbstractPulse) = needpol(pulse.polarisation)
needpol(pulses::Vector{<:Pulses.AbstractPulse}) = any(needpol, pulses)

needpol(pol, pulses::Nothing) = needpol(pol)
needpol(pol, pulse::Pulses.AbstractPulse) = needpol(pulse)
needpol(pol, pulses) = any(needpol, pulses)

needpol_modes(mode::Symbol) = false # mode average
needpol_modes(modes::Number) = false # only HE1m modes

function needpol_modes(modes::Tuple)
    any(modes) do mode
        md = parse_mode(mode)
        md[:kind] ≠ :HE || md[:n] > 1
    end
end


const_linop(radius::Number, pressure::Number) = Val(true)
const_linop(radius, pressure) = Val(false)

function makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    if envelope
        isnothing(thg) && (thg = false)
        Grid.EnvGrid(flength, λ0, λlims, trange; δt, thg)
    else
        Grid.RealGrid(flength, λ0, λlims, trange, δt)
    end
end

makegrid(flength, λ0::Tuple, args...) = makegrid(flength, λ0[1], args...)

function parse_mode(mode)
    ms = String(mode)
    kind_string = ms[1:2]
    if length(ms) > 4
        throw(DomainError(mode, "Ambiguous mode designation $mode. Pass modes as `Dict`s to disambiguate, e.g. Dict(:kind => :HE, :n => 1, :m => 12)."))
    else
        nstring = ms[3]
        mstring = ms[4]
    end
    Dict(:kind => Symbol(kind_string), :n => parse(Int, nstring), :m => parse(Int, mstring))
end

parse_mode(mode::Dict) = mode

function makemodes_pol(pol, args...; kwargs...)
    if pol
        if kwargs[:kind] == :HE && kwargs[:n] == 1
            return [Capillary.MarcatiliMode(args...; ϕ=0.0, kwargs...),
                    Capillary.MarcatiliMode(args...; ϕ=π/2, kwargs...)]
        else
            return [Capillary.MarcatiliMode(args...; ϕ=0.0, kwargs...)]
        end
    else
        Capillary.MarcatiliMode(args...; kwargs...)
    end
end

function makemode_s(mode::Union{Symbol, Dict}, flength, radius, gas, pressure::Number, temperature, model, loss, pol)
    makemodes_pol(pol, radius, gas, pressure; T=temperature, model, loss, parse_mode(mode)...)
end

function makemode_s(mode::Union{Symbol, Dict}, flength, radius, gas, pressure::Tuple{<:Number, <:Number},
                    temperature, model, loss, pol)
    coren, _ = Capillary.gradient(gas, flength, pressure..., T=temperature)
    makemodes_pol(pol, radius, coren; model, loss, parse_mode(mode)...)
end

function makemode_s(mode::Union{Symbol, Dict}, flength, radius, gas, pressure, temperature, model, loss, pol)
    Z, P = pressure
    coren, _ = Capillary.gradient(gas, Z, P, T=temperature)
    makemodes_pol(pol, radius, coren; model, loss, parse_mode(mode)...)
end

function makemode_s(modes::Int, args...)
    _flatten([makemode_s(Dict(:kind => :HE, :n => 1, :m => m), args...) for m=1:modes])
end

function makemode_s(modes::Tuple, args...)
    _flatten([makemode_s(m, args...) for m in modes])
end

# Iterators.flatten recursively flattens arrays of arrays, but can't handle scalars
_flatten(modes::Vector{<:AbstractArray}) = collect(Iterators.flatten(modes))
_flatten(mode) = mode

function makedensity(flength, gas, pressure::Number, temperature)
    ρ0 = PhysData.density(gas, pressure, temperature)
    z -> ρ0
end

function makedensity(flength, gas, pressure::Tuple{<:Number, <:Number}, temperature)
    _, density = Capillary.gradient(gas, flength, pressure..., T=temperature)
    density
end

function makedensity(flength, gas, pressure, temperature)
    _, density = Capillary.gradient(gas, pressure..., T=temperature)
    density
end

function makeresponse(grid::Grid.RealGrid, gas, raman, kerr, plasma, thg, pol,
                      rotation, vibration, PPT_options, preionfrac)
    out = Any[]
    if kerr
        if thg
            push!(out, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
        else
            push!(out, Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas), length(grid.to)))
        end
    end
    makeplasma!(out, grid, gas, plasma, pol, PPT_options, preionfrac)
    if isnothing(raman)
        raman = gas in (:N2, :H2, :D2, :N2O, :CH4, :SF6)
    end
    if raman
        @info("Including the Raman response (due to molecular gas choice).")
        rr = Raman.raman_response(grid.to, gas, rotation=rotation, vibration=vibration)
        if thg
            push!(out, Nonlinear.RamanPolarField(grid.to, rr))
        else
            push!(out, Nonlinear.RamanPolarField(grid.to, rr, thg=false))
        end
    end
    Tuple(out)
end

function makeplasma!(out, grid, gas, plasma::Bool, pol,
                     PPT_options, preionfrac)
    # simple true/false => default to PPT for atoms, ADK for molecules
    if ~plasma
        return
    end
    if gas in (:H2, :D2, :N2O, :CH4, :SF6)
        @info("Using ADK ionisation rate (due to molecular gas choice).")
        model = :ADK
    else
        @info("Using PPT ionisation rate.")
        model = :PPT
    end
    makeplasma!(out, grid, gas, model, pol, PPT_options, preionfrac)
end

function makeplasma!(out, grid, gas, plasma::Symbol, pol,
                     PPT_options, preionfrac)
    ionpot = PhysData.ionisation_potential(gas)
    if plasma == :ADK
        ionrate = Ionisation.ionrate_fun!_ADK(gas)
    elseif plasma == :PPT
        ionrate = Ionisation.ionrate_fun!_PPTcached(gas, grid.referenceλ;
                                                    PPT_options...)
    else
        throw(DomainError(plasma, "Unknown ionisation rate $plasma."))
    end
    Et = pol ? Array{Float64}(undef, length(grid.to), 2) : grid.to
    push!(out, Nonlinear.PlasmaCumtrapz(grid.to, Et, ionrate, ionpot; preionfrac))
end

function makeresponse(grid::Grid.EnvGrid, gas, raman, kerr, plasma, thg, pol,
                      rotation, vibration, PPT_options, preionfrac)
    plasma && error("Plasma response for envelope fields has not been implemented yet.")
    isnothing(thg) && (thg = false) 
    out = Any[]
    if kerr
        if thg
            ω0 = wlfreq(grid.referenceλ)
            r = Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), ω0, grid.to)
            push!(out, r)
        else
            push!(out, Nonlinear.Kerr_env(PhysData.γ3_gas(gas)))
        end
    end
    if isnothing(raman)
        raman = gas in (:N2, :H2, :D2, :N2O, :CH4, :SF6)
    end
    if raman
        @info("Including the Raman response (due to molecular gas choice).")
        rr = Raman.raman_response(grid.to, gas, rotation=rotation, vibration=vibration)
        push!(out, Nonlinear.RamanPolarEnv(grid.to, rr))
    end
    Tuple(out)
end

getAeff(mode::Modes.AbstractMode) = Modes.Aeff(mode)
getAeff(modes) = Modes.Aeff(modes[1])

function makeinputs(mode_s, λ0, pulses::Nothing, τfwhm, τw, ϕ, power, energy,
                    pulseshape, polarisation, propagator)
    if pulseshape == :gauss
        return makeinputs(mode_s, λ0, Pulses.GaussPulse(;λ0, τfwhm, power=power, energy=energy,
                          polarisation, ϕ, propagator))
    elseif pulseshape == :sech
        return makeinputs(mode_s, λ0, Pulses.SechPulse(;λ0, τfwhm, τw, power=power, energy=energy,
                          polarisation, ϕ, propagator))
    else
        error("Valid pulse shapes are :gauss and :sech")
    end
end

function makeinputs(mode_s, λ0, pulses, args...)
    makeinputs(mode_s, λ0, pulses)
end

function findmode(mode_s, pulse)
    if pulse.mode == :lowest
        if pulse.polarisation == :linear
            return [1]
        else
            return [1, 2]
        end
    else
        md = parse_mode(pulse.mode)
        return _findmode(mode_s, md)
    end
end

function _findmode(mode_s::AbstractArray, md)
    return findall(mode_s) do m
        (m.kind == md[:kind]) && (m.n == md[:n]) && (m.m == md[:m])
    end
end


function makeinputs(mode_s, λ0, pulse::Pulses.GaussBeamPulse)
    k = 2π/λ0
    gauss = Fields.normalised_gauss_beam(k, pulse.waist)
    ovlps = [Modes.overlap(mi, gauss) for mi in selectmodes(mode_s, pulse.Nmodes, pulse.polarisation)]
    fields = Any[]
    if pulse.polarisation == :linear
        for (modeidx, ovlp) in enumerate(ovlps)
            energyfac = abs2(ovlp)
            phase = -angle(ovlp)
            sf = scalefield(pulse.timepulse.field, energyfac, phase)
            push!(fields, (mode=modeidx, fields=(sf,)))
        end
    else
        fy, fx = ellfields(pulse.timepulse)
        for (idx, ovlp) in enumerate(ovlps[1:2:end])
            energyfac = abs2(ovlp)
            phase = -angle(ovlp)
            sfy = scalefield(fy, energyfac, phase)
            sfx = scalefield(fx, energyfac, phase)
            push!(fields, (mode=2idx-1, fields=(sfy,)))
            push!(fields, (mode=2idx, fields=(sfx,)))
        end
    end
    Tuple(fields)
end

function selectmodes(mode_s, Nmodes, pol)
    if pol == :linear
        mode_s[1:Nmodes]
    else
        mode_s[1:2Nmodes]
    end
end

selectmodes(mode_s, Nmodes::Symbol, pol) = mode_s

function scalefield(f::Fields.PulseField, fac, phase)
    Fields.PulseField(f.λ0, nmult(f.energy, fac), nmult(f.power, fac), addphase(f.ϕ, phase), f.Itshape)
end

function scalefield(f::Fields.DataField, fac, phase)
    Fields.DataField(f.ω, f.Iω, f.ϕω, nmult(f.energy, fac), addphase(f.ϕ, phase), f.λ0)
end

function scalefield(f::Fields.PropagatedField, fac, phase)
    Fields.PropagatedField(f.propagator!, scalefield(f.field, fac, phase))
end

function addphase(ϕ, phase)
    if phase == 0
        return copy(ϕ)
    end
    if length(ϕ) == 0
        return [phase]
    else
        out = copy(ϕ)
        out[1] += phase
        return out
    end
end

_findmode(mode_s, md) = _findmode([mode_s], md)

function makeinputs(mode_s, λ0, pulse::Pulses.AbstractPulse)
    idcs = findmode(mode_s, pulse)
    (length(idcs) > 0) || error("Mode $(pulse.mode) not found in mode list: $mode_s")
    if pulse.polarisation == :linear || pulse.polarisation == :x
        ((mode=idcs[1], fields=(pulse.field,)),)
    elseif pulse.polarisation == :y
        ((mode=idcs[2], fields=(pulse.field,)),)
    else
        (length(idcs) == 2) || error("Modes not set up for circular/elliptical polarisation")
        f1, f2 = ellfields(pulse)
        ((mode=idcs[1], fields=(f1,)), (mode=idcs[2], fields=(f2,)))
    end
end

function makeinputs(mode_s, λ0, pulses::AbstractVector)
    i = Tuple(collect(Iterators.flatten([makeinputs(mode_s, λ0, pii) for pii in pulses])))
    @debug join(string.(i), "\n")
    return i
end

ellphase(ϕ, pol::Symbol) = ellphase(ϕ, 1.0)
ellphase(ϕ, ε) = addphase(ϕ, π/2 * sign(ε))

ellfac(pol::Symbol) = (1/2, 1/2) # circular
function ellfac(ε::Number)
    (-1 <= ε <= 1) || throw(DomainError(ε, "Ellipticity must be between -1 and 1."))
    (1-ε^2/(1+ε^2), ε^2/(1+ε^2))
end
# sqrt(px/py) = ε => px = ε^2*py; px+py = 1 => px = ε^2*(1-px) => px = ε^2/(1+ε^2)

nmult(x::Nothing, fac) = x
nmult(x, fac) = x*fac

function ellfields(pulse::Union{Pulses.CustomPulse, Pulses.GaussPulse, Pulses.SechPulse})
    f = pulse.field
    py, px = ellfac(pulse.polarisation)
    f1 = Fields.PulseField(f.λ0, nmult(f.energy, py), nmult(f.power, py), f.ϕ, f.Itshape)
    f2 = Fields.PulseField(f.λ0, nmult(f.energy, px), nmult(f.power, px),
                           ellphase(f.ϕ, pulse.polarisation), f.Itshape)
    f1, f2
end

function ellfields(pulse::Pulses.DataPulse)
    f = pulse.field.field
    pf = pulse.field
    py, px = ellfac(pulse.polarisation)
    f1 = Fields.DataField(f.ω, f.Iω, f.ϕω, nmult(f.energy, py), f.ϕ, f.λ0)
    f2 = Fields.DataField(f.ω, f.Iω, f.ϕω, nmult(f.energy, px),
                          ellphase(f.ϕ, pulse.polarisation), f.λ0)
    Fields.PropagatedField(pf.propagator!, f1), Fields.PropagatedField(pf.propagator!, f2)
end

function shotnoise_maybe(inputs, mode::Modes.AbstractMode, shotnoise::Bool)
    shotnoise || return inputs
    (inputs..., (mode=1, fields=(Fields.ShotNoise(),)))
end

function shotnoise_maybe(inputs, modes, shotnoise::Bool)
    shotnoise || return inputs
    (inputs..., [(mode=ii, fields=(Fields.ShotNoise(),)) for ii in eachindex(modes)]...)
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, pol, rtol, c::Val{true})
    @info("Using mode-averaged propagation.")
    linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
    
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
    βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, pol, rtol, c::Val{false})
    @info("Using mode-averaged propagation.")
    linop, βfun! = LinearOps.make_linop(grid, mode, grid.referenceλ)

    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
                                   βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

needfull(modes) = !all(modes) do mode
    (mode.kind == :HE) && (mode.n == 1)
end

function setup(grid, modes, density, responses, inputs, pol, rtol, c::Val{true})
    nf = needfull(modes)
    @info(nf ? "Using full 2-D modal integral." : "Using radial modal integral.")
    linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
                                   pol ? :xy : :y; full=nf, rtol)
    linop, Eω, transform, FT
end

function setup(grid, modes, density, responses, inputs, pol, rtol, c::Val{false})
    nf = needfull(modes)
    @info(nf ? "Using full 2-D modal integral." : "Using radial modal integral.")
    linop = LinearOps.make_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
                                   pol ? :xy : :y; full=nf, rtol)
    linop, Eω, transform, FT
end

function makeoutput(grid, saveN, stats, filepath::Nothing, scan::Nothing, scanidx, filename)
    Output.MemoryOutput(0, grid.zmax, saveN, stats)
end

function makeoutput(grid, saveN, stats, filepath, scan::Nothing, scanidx, filename)
    Output.HDF5Output(filepath, 0, grid.zmax, saveN, stats)
end

function makeoutput(grid, saveN, stats, filepath, scan, scanidx, filename)
    isnothing(scanidx) && error("scanidx must be passed along with scan.")
    Output.ScanHDF5Output(scan, scanidx, 0, grid.zmax, saveN, stats;
                          fdir=filepath, fname=filename)
end

"""
    prop_gnlse(γ, flength, βs; λ0, λlims, trange, kwargs...)

Simulate pulse propagation using the GNLSE.

# Mandatory arguments
- `γ::Number`: The nonlinear coefficient.
- `flength::Number`: Length of the fibre.
- `βs`: The Taylor expansion of the propagation constant about `λ0`.
- `λ0`: (keyword argument) the reference wavelength for the simulation. For simple
    single-pulse inputs, this is also the central wavelength of the input pulse.
- `λlims::Tuple{<:Number, <:Number}`: The wavelength limits for the simulation grid.
- `trange::Number`: The total width of the time grid. To make the number of samples a
    power of 2, the actual grid used is usually bigger.

# Grid options
- `δt::Number`: Time step on the fine grid used for the nonlinear interaction. By default,
    this is determined by the wavelength grid. If `δt` is given **and smaller** than the
    required value, it is used instead.

# Input pulse options
A single pulse can be specified by the keyword arguments below.
More complex inputs can be defined by a single `AbstractPulse` or a `Vector{AbstractPulse}`.
In this case, all keyword arguments except for `λ0` are ignored.
Note that the current GNLSE model is single mode only.

- `λ0`: Central wavelength
- `τfwhm`: The pulse duration as defined by the full width at half maximum.
- `τw`: The "natural" pulse duration. Only available if pulseshape is `sech`.
- `ϕ`: Spectral phases to be applied to the transform-limited pulse. Elements are
    the usual polynomial phases ϕ₀ (CEP), ϕ₁ (group delay), ϕ₂ (GDD), ϕ₃ (TOD), etc.
- `energy`: Pulse energy.
- `power`: Peak power **after any spectral phases are added**.
- `pulseshape`: Shape of the transform-limited pulse. Can be `:gauss` for a Gaussian pulse
    or `:sech` for a sech² pulse.
- `polarisation`: Polarisation of the input pulse. Can be `:linear` (default), `:circular`,
    or an ellipticity number -1 ≤ ε ≤ 1, where ε=-1 corresponds to left-hand circular,
    ε=1 to right-hand circular, and ε=0 to linear polarisation. The major axis for
    elliptical polarisation is always the y-axis.
- `propagator`: A function `propagator!(Eω, grid)` which **mutates** its first argument to
                apply an arbitrary propagation to the pulse before the simulation starts.
- `shotnoise`:  If `true` (default), one-photon-per-mode quantum noise is included.

# GNLSE options
- `shock::Bool`: Whether to include the shock derivative term. Default is `true`.
- `raman::Bool`: Whether to include the Raman effect. Defaults to `true`.
- `ramanmodel`; which Raman model to use, defaults to `:sdo` which uses a simple
   damped oscillator model, defined `τ1` and `τ2` (which default to values commonly
   used for silica). `ramanmodel` can also be set to `:SiO2` which uses the more
   advanced model of Hollenbeck and Cantrell.
- `loss`: the power loss [dB/m]. Defaults to 0.
- `fr`: fractional Raman contribution to `γ`. Defaults to `fr = 0.18`.
- `τ1`: the Raman oscillator period.
- `τ2`: the Raman damping time.

# Output options
- `saveN::Integer`: Number of points along z at which to save the field.
- `filepath`: If `nothing` (default), create a `MemoryOutput` to store the simulation results
    only in the working memory. If not `nothing`, should be a file path as a `String`,
    and the results are saved in a file at this location. If `scan` is passed, `filepath`
    determines the output **directory** for the scan instead.
- `scan`: A `Scan` instance defining a parameter scan. If `scan` is given`, a
    `Output.ScanHDF5Output` is used to automatically name and populate output files of
    the scan. `scanidx` must also be given.
- `scanidx`: Current scan index within a scan being run. Only used when `scan` is passed.
- `filename`: Can be used to to overwrite the scan name when running a parameter scan.
    The running `scanidx` will be appended to this filename. Ignored if no `scan` is given.
- `status_period::Number`: Interval (in seconds) between printed status updates.
"""
function prop_gnlse(args...; status_period=5, kwargs...)
    Eω, grid, linop, transform, FT, output = prop_gnlse_args(args...; kwargs...)
    Luna.run(Eω, grid, linop, transform, FT, output; status_period)
    output
end

"""
    prop_gnlse_args(γ, flength, βs; λ0, λlims, trange, kwargs...)

Prepare to simulate pulse propagation using the GNLSE. This
function takes the same arguments as `prop_gnlse` but instead or running the
simulation and returning the output, it returns the required arguments for `Luna.run`,
which is useful for repeated simulations in an indentical fibre with different initial
conditions.
"""
function prop_gnlse_args(γ, flength, βs; λ0, λlims, trange,
                        δt=1, τfwhm=nothing, τw=nothing, ϕ=Float64[],
                        power=nothing, energy=nothing,
                        pulseshape=:gauss, propagator=nothing,
                        pulses=nothing,
                        shotnoise=true, shock=true,
                        loss=0.0, raman=true, fr=0.18,
                        ramanmodel=:sdo, τ1=12.2e-15, τ2=32e-15,
                        saveN=201, filepath=nothing,
                        scan=nothing, scanidx=nothing, filename=nothing)
    envelope = true
    thg = false
    polarisation=:linear
    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = SimpleFibre.SimpleMode(PhysData.wlfreq(λ0), βs; loss)
    aeff = z -> 1.0
    density = z -> 1.0
    linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, mode_s, λ0)
    k0 = 2π/λ0
    n2 = γ/k0*aeff(0.0)
    # factor of 4/3 below compensates for the factor of 3/4 in Nonlinear.jl, as
    # n2 and γ are usually defined for the envelope case already
    χ3 = 4/3 * (1 - fr) * n2 * (PhysData.ε_0*PhysData.c)
    resp = Any[Nonlinear.Kerr_env(χ3)]
    if raman
        # factor of 2 here compensates for factor 1/2 in Nonlinear.jl as fr is
        # defined for the envelope case already
        χ3R = 2 * fr * n2 * (PhysData.ε_0*PhysData.c)
        if ramanmodel == :SiO2
            push!(resp, Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(grid.to, :SiO2,
                                                                              χ3R * PhysData.ε_0)))
        elseif ramanmodel == :sdo
            if isnothing(τ1) || isnothing(τ2)
                error("for :sdo ramanmodel you must specify τ1 and τ2")
            end
            push!(resp, Nonlinear.RamanPolarEnv(grid.to,
                Raman.CombinedRamanResponse(grid.to,
                    [Raman.RamanRespNormedSingleDampedOscillator(χ3R * PhysData.ε_0, 1/τ1, τ2)])))
        else
            error("unrecognised value for ramanmodel")
        end
    end
    resp = Tuple(resp)

    inputs = makeinputs(mode_s, λ0, pulses, τfwhm, τw, ϕ,
                        power, energy, pulseshape, polarisation, propagator)
    inputs = shotnoise_maybe(inputs, mode_s, shotnoise)

    norm! = NonlinearRHS.norm_mode_average_gnlse(grid, aeff; shock)
    Eω, transform, FT = Luna.setup(grid, density, resp, inputs, βfun!, aeff, norm! = norm!)
    stats = Stats.default(grid, Eω, mode_s, linop, transform)
    output = makeoutput(grid, saveN, stats, filepath, scan, scanidx, filename)

    saveargs(output; γ, flength, βs, λlims, trange, envelope, thg, δt,
        λ0, τfwhm, τw, ϕ, power, energy, pulseshape, polarisation, propagator, pulses, 
        shotnoise, shock, loss, raman, ramanmodel, fr, τ1, τ2, saveN, filepath, filename)

    return Eω, grid, linop, transform, FT, output
end

end
