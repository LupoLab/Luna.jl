module Interface
using Luna
import Luna.PhysData: wlfreq
import Luna: Grid, Modes
import Logging

function prop_capillary(radius, flength, gas, pressure;
                        λlims=(90e-9, 4e-6), trange=2e-12, envelope=false, thg=nothing, δt=1,
                        λ0=nothing, τfwhm=nothing, τ0=nothing, phases=Float64[],
                        peakpower=nothing, energy=nothing, peakintensity=nothing,
                        pulseshape=:gauss,
                        modes=:HE11, model=:full, loss=true,
                        raman=false, kerr=true, plasma=true,
                        saveN=201, filename=nothing)

    grid = makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    mode_s = makemode_s(modes, flength, radius, gas, pressure, model, loss)
    density = makedensity(flength, gas, pressure)
    resp = makeresponse(grid, gas, raman, kerr, plasma, thg)
    inputs = makeinputs(mode_s, λ0, τfwhm, τ0, phases,
                        peakpower, energy, peakintensity, pulseshape)
end

function makegrid(flength, λ0, λlims, trange, envelope, thg, δt)
    if envelope
        isnothing(thg) && (thg = false)
        Grid.EnvGrid(flength, λ0, λlims, trange; δt, thg)
    else
        Grid.RealGrid(flength, λ0, λlims, trange, δt)
    end
end

makegrid(flength, λ0::Tuple, args...) = makegrid(flength, λ0[1], args...)

function parse_mode(modes)
    ms = String(modes)
    Dict(:kind => Symbol(ms[1:2]), :n => parse(Int, ms[3]), :m => parse(Int, ms[4]))
end

function makemode_s(modes::Symbol, flength, radius, gas, pressure::Number, model, loss)
    Capillary.MarcatilliMode(radius, gas, pressure; model, loss, parse_mode(modes)...)
end

function makemode_s(modes::Symbol, flength, radius, gas, pressure::NTuple{2, <:Number},
                    model, loss)
    coren, _ = Capillary.gradient(gas, flength, pressure...)
    Capillary.MarcatilliMode(radius, coren; model, loss, parse_mode(modes)...)
end

function makemode_s(modes::Symbol, flength, radius, gas, pressure, model, loss)
    Z, P = pressure
    coren, _ = Capillary.gradient(gas, Z, P)
    Capillary.MarcatilliMode(radius, coren; model, loss, parse_mode(modes)...)
end

makemode_s(modes::Int, args...) = [makemode_s(Symbol("HE1$n"), args...) for n=1:modes]

makemode_s(modes::NTuple{N, Symbol}, args...) where N = [makemode_s(m, args...) for m in modes]

function makedensity(flength, gas, pressure::Number)
    ρ0 = PhysData.density(gas, pressure)
    z -> ρ0
end

function makedensity(flength, gas, pressure::NTuple{2, <:Number})
    _, density = Capillary.gradient(flength, gas, pressure...)
    density
end

function makedensity(flength, gas, pressure)
    _, density = Capillary.gradient(gas, pressure...)
    density
end

function makeresponse(grid::Grid.RealGrid, gas, raman, kerr, plasma, thg)
    out = Any[]
    kerr && push!(out, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
    makeplasma!(out, grid, gas, plasma)
    raman && push!(out, Nonlinear.RamanPolarField(grid.to, Raman.raman_response(gas)))
    Tuple(out)
end

function makeplasma!(out, grid, gas, plasma::Bool)
    # simple true/false => default to PPT
    plasma || return
    ionpot = PhysData.ionisation_potential(gas)
    ionrate = Ionisation.ionrate_fun!_PPTcached(gas, grid.referenceλ)
    push!(out, Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
end

function makeplasma!(out, grid, gas, plasma::Symbol, thg)
    ionpot = PhysData.ionisation_potential(gas)
    if plasma == :ADK
        ionrate = Ionisation.ionrate_fun!_ADK(gas)
    elseif plasma == :PPT
        ionrate = Ionisation.ionrate_fun!_PPTcached(gas, grid.referenceλ)
    else
        throw(DomainError(plasma, "Unknown ionisation rate $plasma."))
    end
    push!(out, Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
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

function makeinputs(mode_s::Modes.AbstractMode, λ0::Number, τfwhm, τ0, phases,
                    peakpower, energy, peakintensity, pulseshape)
    isnothing(peakintensity) || error("TODO: peak intensity input")
    if pulseshape == :gauss
        return Fields.GaussField(;λ0, τfwhm, energy, power=peakpower, ϕ=phases)
    elseif pulseshape == :sech
        return Fields.SechField(;λ0, energy, power=peakpower, τ0, τfwhm, ϕ=phases)
    end
end

maketuple(arg::Tuple, N) = arg
maketuple(arg, N) = Tuple([arg for _ = 1:N])

arglength(arg::Tuple) = length(arg)
arglength(arg::AbstractArray) = length(arg)
arglength(arg) = 1

function makeinputs(args...)
    N = maximum(arglength, args)
    argsT = [maketuple(arg, N) for arg in args]
    Tuple([makeinputs(aargs...) for aargs in zip(argsT...)])
end

end
