module Interface
using Luna
import Luna.PhysData: wlfreq
import Luna: Grid, Modes
import Logging

function prop_capillary(radius, flength, gas, pressure;
                        λlims=(90e-9, 4e-6), trange=1e-12, envelope=false, thg=nothing, δt=1,
                        λ0=nothing, τfwhm=nothing, τ0=nothing, phases=Float64[],
                        peakpower=nothing, energy=nothing, peakintensity=nothing,
                        pulseshape=:gauss, polarisation=:linear,
                        shotnoise=true,
                        modes=:HE11, model=:full, loss=true,
                        raman=false, kerr=true, plasma=true,
                        saveN=201, filepath=nothing,
                        status_period=5)

    pol = needpol(polarisation)

    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = makemode_s(modes, flength, radius, gas, pressure, model, loss, pol)
    density = makedensity(flength, gas, pressure)
    resp = makeresponse(grid, gas, raman, kerr, plasma, thg, pol)
    inputs = makeinputs(mode_s, λ0, τfwhm, τ0, phases,
                        peakpower, energy, peakintensity, pulseshape, polarisation)
    linop, Eω, transform, FT = setup(grid, mode_s, density, resp, inputs, pol)
    stats = Stats.default(grid, Eω, mode_s, linop, transform; gas=gas)
    output = makeoutput(grid, saveN, stats, filepath)

    Luna.run(Eω, grid, linop, transform, FT, output; status_period)
    output
end

function needpol(pol::Symbol)
    if pol == :linear
        return false
    elseif pol == :circular
        return true
    else
        error("Polarisation must be :linear, :circular, or an ellipticity")
    end
end

needpol(pol::Number) = true
needpol(pol) = error("Polarisation must be :linear, :circular, or an ellipticity")


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

function makeresponse(grid::Grid.EnvGrid, gas, raman, kerr, plasma, thg)
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

function makefield(mode_s, λ0::Number, τfwhm, τ0, phases,
                   peakpower, energy, peakintensity, pulseshape, pfac)
    isnothing(peakintensity) || error("TODO: peak intensity input")
    !isnothing(peakpower) && (peakpower *= pfac)
    !isnothing(energy) && (energy *= pfac)
    if pulseshape == :gauss
        return Fields.GaussField(;λ0, τfwhm, energy, power=peakpower, ϕ=phases)
    elseif pulseshape == :sech
        return Fields.SechField(;λ0, energy, power=peakpower, τ0, τfwhm, ϕ=phases)
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
ellfac(ε::Number) = (1-ε^2/(1+ε^2), ε^2/(1+ε^2))
# sqrt(px/py) = ε => px = ε^2*py; px+py = 1 => px = ε^2*(1-px) => px = ε^2/(1+ε^2)


function makeinputs(mode_s, λ0::Number, τfwhm::Union{Number, Nothing},
                    τ0::Union{Number, Nothing}, phases::AbstractVector,
                    peakpower::Union{Number, Nothing}, energy::Union{Number, Nothing}, 
                    peakintensity::Union{Number, Nothing}, pulseshape::Symbol, polarisation)
    if polarisation == :linear
        field = makefield(mode_s, λ0, τfwhm, τ0, phases,
                          peakpower, energy, peakintensity, pulseshape, 1)
        return ((mode=1, fields=(f1,)),)
    else
        py, px = ellfac(polarisation)
        f1 = makefield(mode_s, λ0, τfwhm, τ0, phases,
                       peakpower, energy, peakintensity, pulseshape, py)
        f2 = makefield(mode_s, λ0, τfwhm, τ0, ellphase(phases),
                       peakpower, energy, peakintensity, pulseshape, px)
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
    polarisation = args[end]
    fields = Tuple([makefield(aargs...) for aargs in zip(argsT...)])
    if polarisation == :circular
        return ((mode=1, fields=fields), (mode=2, fields=fields))
    else
        return ((mode=1, fields=fields),)
    end
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, polarisation)
    linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)

    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
                                   βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

function setup(grid, mode::Modes.AbstractMode, density, responses, inputs, polarisation)
    linop, βfun! = LinearOps.make_linop(grid, mode, grid.referenceλ)

    Eω, transform, FT = Luna.setup(grid, density, responses, inputs,
                            βfun!, z -> Modes.Aeff(mode, z=z))
    linop, Eω, transform, FT
end

function setup(grid, modes, density, responses, inputs, pol)
    linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
                                   pol ? :xy : :y; full=false)
    linop, Eω, transform, FT
end

function setup(grid, modes, density, responses, inputs)
    linop = LinearOps.make_linop(grid, modes, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, density, responses, inputs, modes,
                                   pol ? :xy : :y; full=false)
    linop, Eω, transform, FT
end

function makeoutput(grid, saveN, stats, filepath::Nothing)
    Output.MemoryOutput(0, grid.zmax, saveN, stats)
end

function makeoutput(grid, saveN, stats, filepath)
    Output.HDF5Output(filepath, 0, grid.zmax, saveN, stats)
end

end
