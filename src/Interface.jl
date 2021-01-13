module Interface
using Luna
import Luna.PhysData: wlfreq
import Luna: Grid, Modes, Output
import Logging: @info

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
- `thg::Bool`: Whether to include third-harmonic generation for envelope fields.
    Defaults to `false`. This option has no effect for full-field simulations
- `δt::Number`: Time step on the fine grid used for the nonlinear interaction. By default,
    this is determined by the wavelength grid. If `δt` is given **and smaller** than the
    required value, it is used instead.

# Input pulse options
All of these arguments can be given either as a single value or a `Tuple` of values. If
all are scalar, a single input pulse is created. If any value is a `Tuple`, multiple input
pulses are created. Scalar values are applied to all pulses.
If more than one value is a `Tuple`, all must be the same length.

`τfwhm` and `τ0` are mutually exclusive for each input (i.e. the other must be `nothing`).
`peakpower`, `energy`, and `peakintensity` are also mutually exclusive.

All inputs are placed in the first mode.

- `λ0`: Central wavelength
- `τfwhm`: The pulse duration as defined by the full width at half maximum.
- `τ0`: The "natural" pulse duration. Only available if pulseshape is `sech`.
- `phases`: Spectral phases to be applied to the transform-limited pulse. Elements are
    the usual polynomial phases ϕ₀ (CEP), ϕ₁ (group delay), ϕ₂ (GDD), ϕ₃ (TOD), etc.
    Note that to apply different phases to different input pulses, you must supply
    a `Tuple` of `Vector`s.
- `energy`: Pulse energy.
- `peakpower`: Peak power **of the transform-limited pulse before phases are added**.
- `peakintensity`: Peak intensity **of the transform-limited pulse**. Intensity is taken
    as the mode-averaged value, i.e. peak power divided by effective area.
- `pulseshape`: Shape of the transform-limited pulse. Can be `:gauss` for a Gaussian pulse
    or `:sech` for a sech² pulse.
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

# Output options
- `saveN::Integer`: Number of points along z at which to save the field.
- `filepath`: If `nothing` (default), create a `MemoryOutput` to store the simulation results
    only in the working memory. If not `nothing`, should be a file path as a `String`,
    and the results are saved in a file at this location.
- `status_period::Number`: Interval (in seconds) between printed status updates.
"""
function prop_capillary(radius, flength, gas, pressure;
                        λlims=(90e-9, 4e-6), trange=1e-12, envelope=false, thg=nothing, δt=1,
                        λ0, τfwhm=nothing, τ0=nothing, phases=Float64[],
                        peakpower=nothing, energy=nothing, peakintensity=nothing,
                        pulseshape=:gauss, polarisation=:linear,
                        inputfield=nothing,
                        shotnoise=true,
                        modes=:HE11, model=:full, loss=true,
                        raman=false, kerr=true, plasma=nothing,
                        saveN=201, filepath=nothing,
                        status_period=5)

    pol = needpol(polarisation)
    plasma = isnothing(plasma) ? !envelope : plasma

    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = makemode_s(modes, flength, radius, gas, pressure, model, loss, pol)
    density = makedensity(flength, gas, pressure)
    resp = makeresponse(grid, gas, raman, kerr, plasma, thg, pol)
    inputs = makeinputs(mode_s, λ0, τfwhm, τ0, phases,
                        peakpower, energy, peakintensity, pulseshape, inputfield, polarisation)
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
needpol(pol::Tuple) = any(needpol, pol)

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

function makemode_s(mode::Symbol, flength, radius, gas, pressure::NTuple{2, <:Number},
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

function makedensity(flength, gas, pressure::NTuple{2, <:Number})
    _, density = Capillary.gradient(gas, flength, pressure...)
    density
end

function makedensity(flength, gas, pressure)
    _, density = Capillary.gradient(gas, pressure...)
    density
end

function makeresponse(grid::Grid.RealGrid, gas, raman, kerr, plasma, thg, pol)
    out = Any[]
    kerr && push!(out, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
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

# Select fundamental mode from multi-mode sim or take just the single mode
modeslice(Eω::Array{ComplexF64, 2}) = Eω[:, end]
modeslice(Eω::Array{ComplexF64, 3}) = Eω[:, 1, end]

function _fieldargs(o::Output.AbstractOutput)
    ω = o["grid"]["ω"]
    t = o["grid"]["t"]
    τ = length(t) * (t[2] - t[1])/2 # middle of old time window
    Eωm1 = modeslice(o["Eω"]) # either mode-averaged field or first mode
    (ω, Eωm1 .* exp.(1im .* ω .* τ))
end

function _fieldargs(args::NamedTuple{(:ω, :Iω, :ϕω), NTuple{3, Vector{Float64}}})
    Eω = @. sqrt(args.Iω) * exp(1im*args.ϕω)
    (args.ω, Eω)
end

function _fieldargs(args::NamedTuple{(:ω, :Eω), Tuple{Vector{Float64}, Vector{ComplexF64}}})
    (args.ω, args.Eω)
end


function makefield(mode_s, λ0::Number, τfwhm, τ0, phases,
                   peakpower, energy, peakintensity, pulseshape, inputfield, pfac)
    !isnothing(peakpower) && (peakpower *= pfac)
    !isnothing(energy) && (energy *= pfac)
    if !isnothing(peakintensity)
        isnothing(peakpower) || error("Only one of peak intensity or power can be set.")
        peakpower = getAeff(mode_s) * peakintensity * pfac
    end
    if !isnothing(inputfield)
        isnothing(energy) && error("For input fields from data, energy must be set.")
        ω, Eω = _fieldargs(inputfield)
        return Fields.DataField(ω, Eω; energy, ϕ=phases, λ0)
    end

    if pulseshape == :gauss
        isnothing(τ0) || error("τ0 pulse duration only applies to sech² pulses.")
        return Fields.GaussField(;λ0, τfwhm, energy, power=peakpower, ϕ=phases)
    elseif pulseshape == :sech
        return Fields.SechField(;λ0, energy, power=peakpower, τw=τ0, τfwhm, ϕ=phases)
    end
end

function ellphase(phases)
    if length(phases) == 0
        return [π/2]
    else
        out = copy(phases)
        out[1] += π/2
        return out
    end
end

ellfac(pol::Symbol) = (1/2, 1/2) # circular
function ellfac(ε::Number)
    (0 <= ε <= 1) || throw(DomainError(ε, "Ellipticity must be between 0 and 1."))
    (1-ε^2/(1+ε^2), ε^2/(1+ε^2))
end
# sqrt(px/py) = ε => px = ε^2*py; px+py = 1 => px = ε^2*(1-px) => px = ε^2/(1+ε^2)


function makeinputs(mode_s, λ0::Number, τfwhm::Union{Number, Nothing},
                    τ0::Union{Number, Nothing}, phases::AbstractVector,
                    peakpower::Union{Number, Nothing}, energy::Union{Number, Nothing}, 
                    peakintensity::Union{Number, Nothing}, pulseshape::Symbol,
                    inputfield::Union{
                        Nothing, Array{Float64,2}, Output.AbstractOutput,
                        NamedTuple{(:ω, :Eω), Tuple{Vector{Float64}, Vector{ComplexF64}}}},
                        polarisation)
    if polarisation == :linear
        field = makefield(mode_s, λ0, τfwhm, τ0, phases,
                          peakpower, energy, peakintensity, pulseshape, inputfield, 1)
        return ((mode=1, fields=(field,)),)
    else
        py, px = ellfac(polarisation)
        f1 = makefield(mode_s, λ0, τfwhm, τ0, phases,
                       peakpower, energy, peakintensity, pulseshape, inputfield, py)
        f2 = makefield(mode_s, λ0, τfwhm, τ0, ellphase(phases),
                       peakpower, energy, peakintensity, pulseshape, inputfield, px)
        return ((mode=1, fields=(f1,)), (mode=2, fields=(f2,)))
    end
end

# "manual" broadcasting of arguments
maketuple(arg::Tuple, N) = arg
maketuple(arg, N) = Tuple([arg for _ = 1:N])

arglength(arg::Tuple) = length(arg)
arglength(arg) = 1

function makeinputs(args...)
    N = maximum(arglength, args)
    argsT = [maketuple(arg, N) for arg in args]
    Tuple(collect(Iterators.flatten([makeinputs(aargs...) for aargs in zip(argsT...)])))
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
