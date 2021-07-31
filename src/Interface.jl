module Interface
using Luna
import Luna.PhysData: wlfreq
import Luna: Grid, Modes, Output, Fields
import Logging: @info

abstract type AbstractPulse end

struct CustomPulse <: AbstractPulse
    mode::Symbol
    polarisation
    field::Fields.PulseField
end

"""
    CustomPulse(;λ0, energy=nothing, power=nothing, ϕ=Float64[],
                mode=:lowest, polarisation=:linear)

A custom pulse defined by a function for use with `prop_capillary`, with either energy or
peak power specified.

# Keyword arguments
- `λ0::Number`: the central wavelength
- `Itshape::function`: a function `I(t)`` which defines the intensity/power envelope of the
                       pulse as a function of time `t`. Note that the normalisation of this
                       envelope is irrelevant as it will be re-scaled by `energy` or `power`.
- `energy::Number`: the pulse energy
- `power::Number`: the pulse peak power (**after** applying any spectral phases)
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...)
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`
- `polarisation`: Can be `:linear`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
"""
function CustomPulse(;mode=:lowest, polarisation=:linear, kwargs...)
    CustomPulse(mode, polarisation, Fields.PulseField(;kwargs...))
end

struct GaussPulse <: AbstractPulse
    mode::Symbol
    polarisation
    field::Fields.PulseField
end

"""
    GaussPulse(;λ0, τfwhm, energy=nothing, power=nothing, ϕ=Float64[], m=1,
               mode=:lowest, polarisation=:linear)

A (super)Gaussian pulse for use with `prop_capillary`, with either energy or peak power
specified.

# Keyword arguments
- `λ0::Number`: the central wavelength
- `τfwhm::Number`: the pulse duration (power/intensity FWHM)
- `energy::Number`: the pulse energy
- `power::Number`: the pulse peak power (**after** applying any spectral phases)
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...)
- `m::Int`: super-Gaussian parameter (the power in the Gaussian exponent is 2m).
            Defaults to 1.
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`
- `polarisation`: Can be `:linear`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
"""
function GaussPulse(;mode=:lowest, polarisation=:linear, kwargs...)
    GaussPulse(mode, polarisation, Fields.GaussField(;kwargs...))
end

struct SechPulse <: AbstractPulse
    mode::Symbol
    polarisation
    field::Fields.PulseField
end

"""
    SechPulse(;λ0, τfwhm=nothing, τw=nothing, energy=nothing, power=nothing, ϕ=Float64[],
               mode=:lowest, polarisation=:linear)

A sech²(τ/τw) pulse for use with `prop_capillary`, with either `energy` or peak `power`
specified, and duration given either as `τfwhm` or `τw`.

# Keyword arguments
- `λ0::Number`: the central wavelength
- `τfwhm::Number`: the pulse duration (power/intensity FWHM)
- `τw::Number`: "natural" pulse duration of a sech²(τ/τw) pulse
- `energy::Number`: the pulse energy
- `power::Number`: the pulse peak power (**after** applying any spectral phases)
- `ϕ::Vector{Number}`: spectral phases (CEP, group delay, GDD, TOD, ...)
- `mode::Symbol`: Mode in which this input should be coupled. Can be `:lowest` for the
                  lowest-order mode in the simulation, or a mode designation
                  (e.g. `:HE11`, `:HE12`, `:TM01`, etc.). Defaults to `:lowest`
- `polarisation`: Can be `:linear`, `:circular`, or an ellipticity number -1 ≤ ε ≤ 1,
                  where ε=-1 corresponds to left-hand circular, ε=1 to right-hand circular,
                  and ε=0 to linear polarisation.
"""
function SechPulse(;mode=:lowest, polarisation=:linear, kwargs...)
    SechPulse(mode, polarisation, Fields.SechField(;kwargs...))
end

struct DataPulse <: AbstractPulse
    mode::Symbol
    polarisation
    field::Fields.DataField
end

#TODO add peak power to DataPulses
function DataPulse(ω::AbstractVector, Iω, ϕω; mode=:lowest, polarisation=:linear, kwargs...)
    DataPulse(mode, polarisation, Fields.DataField(ω, Iω, ϕω; kwargs...))
end

function DataPulse(ω, Eω; mode=:lowest, polarisation=:linear, kwargs...)
    DataPulse(mode, polarisation, Fields.DataField(ω, Eω; kwargs...))
end

function DataPulse(fpath; mode=:lowest, polarisation=:linear, kwargs...)
    DataPulse(mode, polarisation, Fields.DataField(fpath; kwargs...))
end

# Select fundamental mode from multi-mode sim or take just the single mode
modeslice(Eω::Array{ComplexF64, 2}) = Eω[:, end]
modeslice(Eω::Array{ComplexF64, 3}) = Eω[:, 1, end]

function LunaPulse(o::Output.AbstractOutput; kwargs...)
    ω = o["grid"]["ω"]
    t = o["grid"]["t"]
    τ = length(t) * (t[2] - t[1])/2 # middle of old time window
    Eωm1 = modeslice(o["Eω"]) # either mode-averaged field or first mode
    DataPulse(ω, Eωm1 .* exp.(1im .* ω .* τ); kwargs...)
end


"""
    prop_capillary(radius, flength, gas, pressure; kwargs...)

Simulate pulse propagation in a hollow fibre using the capillary model.

# Mandatory arguments
- `radius`: Core radius of the fibre. Can be a `Number` for constant radius, or a function
    `a(z)` which returns the `z`-dependent radius.
- `flength::Number`: Length of the fibre.
- `gas::Symbol`: Filling gas species.
- `pressure`: Gas pressure. Can be a `Number` for constant pressure, a 2-`Tuple` of `Number`s
    for a simple pressure gradient, or a `Tuple` of `(Z, P)` where `Z` and `P`
    contain `z` positions and the pressures at those positions.
- `λ0`: Keyword argument. Can be a `Number` for input pulse(s) at the same wavelength or a
    `Tuple` of several wavelengths. In the latter case, the first element of `λ0`
    is taken as the global reference wavelength for the moving frame and ionisation.

# Grid options
- `λlims::Tuple{<:Number, <:Number}`: The wavelength limits for the simulation grid.
    Defaults to a grid from 90 nm to 4 μm.
- `trange::Number`: The total width of the time grid. Defaults to 1 ps.
    To make the number of samples a power of 2, the actual grid used is usually bigger.
- `envelope::Bool`: Whether to use envelope fields for the simulation. Defaults to `false`.
    By default, envelope simulations ignore third-harmonic generation.
    Plasma has not yet been implemented for envelope fields.
- `δt::Number`: Time step on the fine grid used for the nonlinear interaction. By default,
    this is determined by the wavelength grid. If `δt` is given **and smaller** than the
    required value, it is used instead.

# Input pulse options
All of these arguments can be given either as a single value or a `Tuple` of values. If
all are scalar, a single input pulse is created. If any value is a `Tuple`, multiple input
pulses are created. Scalar values are applied to all pulses.
If more than one value is a `Tuple`, all must be the same length.

`τfwhm` and `τ0` are mutually exclusive for each input (i.e. the other must be `nothing`).
`power`, `energy`, and `peakintensity` are also mutually exclusive.

All inputs are placed in the first mode.

- `λ0`: Central wavelength
- `τfwhm`: The pulse duration as defined by the full width at half maximum.
- `τ0`: The "natural" pulse duration. Only available if pulseshape is `sech`.
- `phases`: Spectral phases to be applied to the transform-limited pulse. Elements are
    the usual polynomial phases ϕ₀ (CEP), ϕ₁ (group delay), ϕ₂ (GDD), ϕ₃ (TOD), etc.
    Note that to apply different phases to different input pulses, you must supply
    a `Tuple` of `Vector`s.
- `propagator`: A function `propagator(grid, Eω)` which returns a propagated field `Eω`. 
        This can be used to apply arbitrary propagation to the input pulse before propagation.
- `energy`: Pulse energy.
- `power`: Peak power **of the transform-limited pulse before phases are added**.
- `peakintensity`: Peak intensity **of the transform-limited pulse**. Intensity is taken
    as the mode-averaged value, i.e. peak power divided by effective area.
- `pulseshape`: Shape of the transform-limited pulse. Can be `:gauss` for a Gaussian pulse
    or `:sech` for a sech² pulse.
- `inputfield`: Can be one of:
    - `::Luna.Output.AbstractOutput`: use the last slice of the propagation in the given
        output as the input to this propagation. For multi-mode input simulations, only the
        fundamental mode content is used.
    - A `NamedTuple` with fields `ω` and `Eω`, containing angular frequency and the complex
        frequency-domain field, respectively. The field is interpolated onto the new grid.
    - A `NamedTuple` with fields `ω`, `Iω`, and `ϕω`, containing angular frequency, The
        spectral energy density `Iω` and the spectral phase `ϕω` respectively. The energy
        density and phase are interpolated onto the new grid.
    Note that for the `NamedTuple` arguments, the spectral phase should not contain a strong
    linear component to centre the pulse in the old time window.
- `polarisation`: Polarisation of the input pulse. Can be `:linear` (default), `:circular`,
    or a `Number` between `0` and `1` which defines the ellipticity. The major axis for
    elliptical polarisation is always the y-axis.
- `shotnoise`:  If `true` (default), one-photon-per-mode quantum noise is included.


# Modes options
- `modes`: Defines which modes are included in the propagation. Can be any of:
    - a single mode signifier (default: :HE11), which leads to mode-averaged propagation
        (as long as all inputs are linearly polarised).
    - a list of mode signifiers, which leads to multi-mode propagation in those modes.
    - a `Number` `N` of modes, which simply creates the first `N` `HE` modes.
    Note that when elliptical or circular polarisation is included, each mode is present
    twice in the output, once for `x` and once for `y` polarisation.
- `model::Symbol`: Can be `:full`, which includes the full complex refractive index of the cladding
    in the effective index of the mode, or `:reduced`, which uses the simpler model more
    commonly seen in the literature. See `Luna.Capillary` for more details.
    Defaults to `:full`.
- `loss::Bool`: Whether to include propagation loss. Defaults to `true`.

# Nonlinear interaction options
- `kerr`: Whether to include the Kerr effect. Defaults to `true`.
- `raman`: Whether to include the Raman effect. Defaults to `false`.
- `plasma`: Can be one of
    - `:ADK` -- include plasma using the ADK ionisation rate.
    - `:PPT` -- include plasma using the PPT ionisation rate.
    - `true` (default) -- same as `:PPT`.
    - `false` -- ignore plasma.
    Note that plasma is only available for full-field simulations.
- `thg::Bool`: Whether to include third-harmonic generation. Defaults to `true` for
    full-field simulations and to `false` for envelope simulations.

# Output options
- `saveN::Integer`: Number of points along z at which to save the field.
- `filepath`: If `nothing` (default), create a `MemoryOutput` to store the simulation results
    only in the working memory. If not `nothing`, should be a file path as a `String`,
    and the results are saved in a file at this location.
- `status_period::Number`: Interval (in seconds) between printed status updates.
"""
function prop_capillary(radius, flength, gas, pressure;
                        λlims=(90e-9, 4e-6), trange=1e-12, envelope=false, thg=nothing, δt=1,
                        λ0, τfwhm=nothing, τw=nothing, phases=Float64[],
                        power=nothing, energy=nothing,
                        pulseshape=:gauss, polarisation=:linear,
                        pulses=nothing,
                        shotnoise=true,
                        modes=:HE11, model=:full, loss=true,
                        raman=false, kerr=true, plasma=nothing,
                        saveN=201, filepath=nothing,
                        status_period=5)

    pol = needpol(polarisation, pulses)
    plasma = isnothing(plasma) ? !envelope : plasma
    thg = isnothing(thg) ? !envelope : thg

    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = makemode_s(modes, flength, radius, gas, pressure, model, loss, pol)
    density = makedensity(flength, gas, pressure)
    resp = makeresponse(grid, gas, raman, kerr, plasma, thg, pol)
    inputs = makeinputs(mode_s, λ0, pulses, τfwhm, τw, phases,
                        power, energy, pulseshape, polarisation)
    inputs = shotnoise_maybe(inputs, mode_s, shotnoise) 
    linop, Eω, transform, FT = setup(grid, mode_s, density, resp, inputs, pol,
                                     const_linop(radius, pressure))
    stats = Stats.default(grid, Eω, mode_s, linop, transform; gas=gas)
    output = makeoutput(grid, saveN, stats, filepath)

    Luna.run(Eω, grid, linop, transform, FT, output; status_period)
    output
end

function needpol(pol)
    if pol == :linear
        return false
    elseif pol == :circular
        return true
    else
        error("Polarisation must be :linear, :circular, or an ellipticity")
    end
end
needpol(pol::Number) = true

needpol(pol, pulses::Nothing) = needpol(pol)
needpol(pol, pulse::AbstractPulse) = needpol(pulse.polarisation)
needpol(pol, pulses) = any(needpol, pulses)

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
    Dict(:kind => Symbol(ms[1:2]), :n => parse(Int, ms[3]), :m => parse(Int, ms[4]))
end

function makemodes_pol(pol, args...; kwargs...)
    # TODO: This is not type stable
    if pol
        [Capillary.MarcatilliMode(args...; ϕ=0.0, kwargs...),
         Capillary.MarcatilliMode(args...; ϕ=π/2, kwargs...)]
    else
        Capillary.MarcatilliMode(args...; kwargs...)
    end
end

function makemode_s(mode::Symbol, flength, radius, gas, pressure::Number, model, loss, pol)
    makemodes_pol(pol, radius, gas, pressure; model, loss, parse_mode(mode)...)
end

function makemode_s(mode::Symbol, flength, radius, gas, pressure::Tuple{<:Number, <:Number},
                    model, loss, pol)
    coren, _ = Capillary.gradient(gas, flength, pressure...)
    makemodes_pol(pol, radius, coren; model, loss, parse_mode(mode)...)
end

function makemode_s(mode::Symbol, flength, radius, gas, pressure, model, loss, pol)
    Z, P = pressure
    coren, _ = Capillary.gradient(gas, Z, P)
    makemodes_pol(pol, radius, coren; model, loss, parse_mode(mode)...)
end

function makemode_s(modes::Int, args...)
    _flatten([makemode_s(Symbol("HE1$n"), args...) for n=1:modes])
end

function makemode_s(modes::NTuple{N, Symbol}, args...) where N 
    _flatten([makemode_s(m, args...) for m in modes])
end

# Iterators.flatten recursively flattens arrays of arrays, but can't handle scalars
_flatten(modes::Vector{<:AbstractArray}) = collect(Iterators.flatten(modes))
_flatten(mode) = mode

function makedensity(flength, gas, pressure::Number)
    ρ0 = PhysData.density(gas, pressure)
    z -> ρ0
end

function makedensity(flength, gas, pressure::Tuple{<:Number, <:Number})
    _, density = Capillary.gradient(gas, flength, pressure...)
    density
end

function makedensity(flength, gas, pressure)
    _, density = Capillary.gradient(gas, pressure...)
    density
end

function makeresponse(grid::Grid.RealGrid, gas, raman, kerr, plasma, thg, pol)
    out = Any[]
    if kerr
        if thg
            push!(out, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
        else
            push!(out, Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas), length(grid.to)))
        end
    end
    makeplasma!(out, grid, gas, plasma, pol)
    raman && push!(out, Nonlinear.RamanPolarField(grid.to, Raman.raman_response(gas)))
    Tuple(out)
end

function makeplasma!(out, grid, gas, plasma::Bool, pol)
    # simple true/false => default to PPT
    plasma && makeplasma!(out, grid, gas, :PPT, pol)
end

function makeplasma!(out, grid, gas, plasma::Symbol, pol)
    ionpot = PhysData.ionisation_potential(gas)
    if plasma == :ADK
        ionrate = Ionisation.ionrate_fun!_ADK(gas)
    elseif plasma == :PPT
        ionrate = Ionisation.ionrate_fun!_PPTcached(gas, grid.referenceλ)
    else
        throw(DomainError(plasma, "Unknown ionisation rate $plasma."))
    end
    Et = pol ? Array{Float64}(undef, length(grid.to), 2) : grid.to
    push!(out, Nonlinear.PlasmaCumtrapz(grid.to, Et, ionrate, ionpot))
end

function makeresponse(grid::Grid.EnvGrid, gas, raman, kerr, plasma, thg, pol)
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
    raman && push!(out, Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(gas)))
    Tuple(out)
end

getAeff(mode::Modes.AbstractMode) = Modes.Aeff(mode)
getAeff(modes) = Modes.Aeff(modes[1])

function makeinputs(mode_s, λ0, pulses::Nothing, τfwhm, τw, phases, power, energy,
                    pulseshape, polarisation)
    if pulseshape == :gauss
        return makeinputs(mode_s, λ0, GaussPulse(;λ0, τfwhm, power=power, energy=energy,
                          polarisation, ϕ=phases))
    elseif pulseshape == :sech
        return makeinputs(mode_s, λ0, SechPulse(;λ0, τfwhm, τw, power=power, energy=energy,
                          polarisation, ϕ=phases))
    else
        error("Valid pulse shapes are :gauss and :sech")
    end
end

function makeinputs(mode_s, λ0, pulses, args...)
    # if ~(all(isnothing, args) || (length(args) == 0))
    #     error("When using Pulses to specify input, only λ0 must be used as a numeric argument.")
    # end
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

_findmode(mode_s, md) = _findmode([mode_s], md)


function makeinputs(mode_s, λ0, pulse::AbstractPulse)
    idcs = findmode(mode_s, pulse)
    (length(idcs) > 0) || error("Mode $(pulse.mode) not found in mode list: $mode_s")
    if pulse.polarisation == :linear
        ((mode=idcs[1], fields=(pulse.field,)),)
    else
        (length(idcs) == 2) || error("Modes not set up for circular/elliptical polarisation")
        f1, f2 = ellfields(pulse)
        ((mode=idcs[1], fields=(f1,)), (mode=idcs[2], fields=(f2,)))
    end
end

ellphase(phases, pol::Symbol) = ellphase(phases, 1.0)

function ellphase(phases, ε)
    shift = π/2 * sign(ε)
    if length(phases) == 0
        return [shift]
    else
        out = copy(phases)
        out[1] += shift
        return out
    end
end

ellfac(pol::Symbol) = (1/2, 1/2) # circular
function ellfac(ε::Number)
    (-1 <= ε <= 1) || throw(DomainError(ε, "Ellipticity must be between -1 and 1."))
    (1-ε^2/(1+ε^2), ε^2/(1+ε^2))
end
# sqrt(px/py) = ε => px = ε^2*py; px+py = 1 => px = ε^2*(1-px) => px = ε^2/(1+ε^2)

nmult(x::Nothing, fac) = x
nmult(x, fac) = x*fac

function ellfields(pulse::Union{CustomPulse, GaussPulse, SechPulse})
    f = pulse.field
    py, px = ellfac(pulse.polarisation)
    f1 = Fields.PulseField(f.λ0, nmult(f.energy, py), nmult(f.power, py), f.ϕ, f.Itshape)
    f2 = Fields.PulseField(f.λ0, nmult(f.energy, px), nmult(f.power, px),
                           ellphase(f.ϕ, pulse.polarisation), f.Itshape)
    f1, f2
end

function ellfields(pulse::DataPulse)
    f = pulse.field
    py, px = ellfac(pulse.polarisation)
    f1 = Fields.DataField(f.ω, f.Iω, f.ϕω, nmult(f.energy, py), f.ϕ, f.λ0)
    f2 = Fields.DataField(f.ω, f.Iω, f.ϕω, nmult(f.energy, px),
                          ellphase(f.ϕ, pulse.polarisation), f.λ0)
    f1, f2
end

function shotnoise_maybe(inputs, mode::Modes.AbstractMode, shotnoise::Bool)
    shotnoise || return inputs
    (inputs..., (mode=1, fields=(Fields.ShotNoise(),)))
end

function shotnoise_maybe(inputs, modes, shotnoise::Bool)
    shotnoise || return inputs
    (inputs..., [(mode=ii, fields=(Fields.ShotNoise(),)) for ii in eachindex(modes)]...)
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, pol, c::Val{true})
    @info("Using mode-averaged propagation.")
    linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
    
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
    βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, pol, c::Val{false})
    @info("Using mode-averaged propagation.")
    linop, βfun! = LinearOps.make_linop(grid, mode, grid.referenceλ)

    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
                                   βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

needfull(modes) = !all(modes) do mode
    (mode.kind == :HE) && (mode.n == 1)
end

function setup(grid, modes, density, responses, inputs, pol, c::Val{true})
    nf = needfull(modes)
    @info(nf ? "Using full 2-D modal integral." : "Using radial modal integral.")
    linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
    pol ? :xy : :y; full=nf)
    linop, Eω, transform, FT
end

function setup(grid, modes, density, responses, inputs, pol, c::Val{false})
    nf = needfull(modes)
    @info(nf ? "Using full 2-D modal integral." : "Using radial modal integral.")
    linop = LinearOps.make_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
                                   pol ? :xy : :y; full=nf)
    linop, Eω, transform, FT
end

function makeoutput(grid, saveN, stats, filepath::Nothing)
    Output.MemoryOutput(0, grid.zmax, saveN, stats)
end

function makeoutput(grid, saveN, stats, filepath)
    Output.HDF5Output(filepath, 0, grid.zmax, saveN, stats)
end

end
