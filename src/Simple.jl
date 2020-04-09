module Simple
import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
import FFTW
import PyPlot: pygui, plt

"""
    solveprop(;grid::Grid.AbstractGrid, modes, fields,
              gas, pressure,
              kerr=true, plasma=true, raman=:auto, thg=true,
              temperature=PhysData.roomtemp,
              filename=nothing, zpoints=201)

Solve a propagation problem

# Keyword Arguments
- `grid::Grid.AbstractGrid`: the grid to use
- `modes`: a tuple of `Modes.AbstractMode`
- `fields`: a tuple of `Fields.AbstractField`
- `gas`: the gas type
- `pressure`: the gas pressure
- `kerr::Bool`: to include the kerr effect or not
- `plasma::Bool`: to include plasma effects or not
- `raman::Symbol`: `:none` for no Raman, `:auto` selects based on gas
- `thg::Bool`: whether to include THG or not
- `temperature::Float64`: defaults to room temperature
- `filename`: if this is nothing (default) use MemoryOutput, else an
              HDF5Output with the provided filename
- `zpoints::Integer`: number of points to save at
...
"""
function solveprop(;grid::Grid.AbstractGrid, modes, fields,
                    gas, pressure,
                    kerr=true, plasma=true, raman=:auto, thg=true,
                    temperature=PhysData.roomtemp,
                    filename=nothing, zpoints=201)
    nmodes = length(modes)
    aeff(z) = Modes.Aeff(modes[1], z=z)
    energyfun = NonlinearRHS.energy_modal()
    dens0 = PhysData.density(gas, pressure)
    densityfun(z) = dens0
    responses = build_response(grid, gas, kerr, plasma, raman, thg)
    if nmodes == 1
        linop, βfun, β1, αfun = LinearOps.make_const_linop(grid, modes[1], grid.referenceλ)
    else
        linop, βfun, β1, αfun = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    end
    normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)
    inputs = [(func=t->f(grid), energy=f.energy) for f in fields]
    Eω, transform, FT = Luna.setup(
            grid, energyfun, densityfun, normfun, responses, inputs, aeff)
    statsfun = Stats.collect_stats((Stats.ω0(grid), ))
    if filename == nothing
        output = Output.MemoryOutput(0, grid.zmax, zpoints, (length(grid.ω),), statsfun)
    else
        output = Output.HDF5Output(filename, 0, grid.zmax, zpoints, (length(grid.ω),), statsfun)
    end
    Luna.run(Eω, grid, linop, transform, FT, output)
    return (output=output, grid=grid, modes=modes, inputs=inputs, gas=gas, pressure=pressure,
            kerr=kerr, plasma=plasma, raman=raman, thg=thg, temperature=temperature)
end

function build_response(grid::Grid.RealGrid, gas, kerr, plasma, raman, thg)
    responses = []
    if kerr
        if thg
            push!(responses, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
        else
            push!(responses, Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas), length(grid.to)))
        end
    end
    if plasma
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
        push!(responses, Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
    end
    responses
end

function build_response(grid::Grid.EnvGrid, gas, kerr, plasma, raman, thg)
    ω0 = PhysData.wlfreq(grid.referenceλ)
    responses = []
    if kerr
        if thg
            push!(responses, Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), ω0, grid.to))
        else
            push!(responses, Nonlinear.Kerr_env(PhysData.γ3_gas(gas)))
        end
    end
    if plasma
        error("plasma is not currently supported for envelope fields")
    end
    responses
end

function plotprop(solution; tlims=(-30e-15, 30e-15), flims=(0.0, 2e15))
    grid = solution.grid
    output = solution.output
    env = isa(grid, Grid.EnvGrid)
    ω = grid.ω
    t = grid.t
    f = ω./2π
    if env
        f = FFTW.fftshift(f, 1)
    end
    zout = output.data["z"]
    Eout = output.data["Eω"]
    if env
        Etout = FFTW.ifft(Eout, 1)
    else
        Etout = FFTW.irfft(Eout, length(grid.t), 1)
    end
    Ilog = log10.(Maths.normbymax(abs2.(Eout)))
    if env
        Ilog = FFTW.fftshift(Ilog, 1)
    end
    idcs = @. (t < tlims[2]) & (t > [tlims[1]])
    to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=16, dim=1)
    if env
        It = abs2.(Eto)
    else
        It = abs2.(Maths.hilbert(Eto))
    end
    pygui(true)
    plt.figure()
    plt.pcolormesh(f.*1e-15, zout, transpose(Ilog))
    plt.clim(-6, 0)
    plt.xlim(flims[1]/1e15, flims[2]/1e15)
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(to*1e15, zout, transpose(It))
    plt.colorbar()
    plt.xlim(tlims[1]/1e-15, tlims[2]/1e-15)
end

end
    