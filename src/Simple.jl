module Simple
import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
import FFTW
import PyPlot: pygui, plt

function solveprop(;kwargs...)
    _solveprop(kwargs; kwargs...)
end

function _solveprop(args;
                    radius, gas, pressure, flength,
                    τfwhm, λ0,
                    minλ, maxλ, τgrid,
                    energy=nothing, power=nothing,
                    shape=:gaussian, fieldkind=:full,
                    loss=true,
                    kerr=true, plasma=true, raman=:auto, thg=true,
                    temperature=PhysData.roomtemp,
                    filename=nothing, zpoints=201)
    ω0 = PhysData.wlfreq(λ0)
    if energy == nothing
        if power == nothing
            error("only one of `energy` or `power` may be specified")
        else
            error("not yet implemented") #energy = power_to_energy(power, shape)
        end
    end
    if fieldkind == :full
        grid = Grid.RealGrid(flength, λ0, (minλ, maxλ), τgrid)
    elseif fieldkind == :env
        grid = Grid.EnvGrid(flength, λ0, (minλ, maxλ), τgrid, thg=thg)
    else
        error("unkown `fieldkind`: $fieldkind")
    end
    m = Capillary.MarcatilliMode(radius, gas, pressure, loss=loss)
    aeff(z) = Modes.Aeff(m, z=z)
    energyfun = NonlinearRHS.energy_modal()
    dens0 = PhysData.density(gas, pressure)
    densityfun(z) = dens0
    function gausspulse(t)
        It = Maths.gauss(t, fwhm=τfwhm)
        Et = @. sqrt(It)
        if fieldkind == :full
            @. Et *= cos(ω0*t)
        end
        Et
    end
    function sechpulse(t)
        τ0 = τfwhm/1.763
        It = sech(t/τ0)
        Et = @. sqrt(It)
        if fieldkind == :full
            @. Et *= cos(ω0*t)
        end
        Et
    end
    if shape == :gaussian
        in1 = (func=gausspulse, energy=energy)
    elseif shape == :sech
        in1 = (func=sechpulse, energy=energy)
    else
        error("unkown pulse shape $shape")
    end
    inputs = (in1, )
    responses = []
    if kerr
        if fieldkind == :full
            if thg
                push!(responses, Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
            else
                push!(responses, Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas), length(grid.to)))
            end
        else
            if thg
                push!(responses, Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), ω0, grid.to))
            else
                push!(responses, Nonlinear.Kerr_env(PhysData.γ3_gas(gas)))
            end
        end
    end
    if plasma
        if fieldkind == :env
            error("plasma is not currently supported for `fieldkind = :env`")
        end
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
        push!(responses, Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
    end
    linop, βfun, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)
    normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)
    Eω, transform, FT = Luna.setup(
            grid, energyfun, densityfun, normfun, responses, inputs, aeff)
    statsfun = Stats.collect_stats((Stats.ω0(grid), ))
    if filename == nothing
        output = Output.MemoryOutput(0, grid.zmax, zpoints, (length(grid.ω),), statsfun)
    else
        output = Output.HDF5Output(filename, 0, grid.zmax, zpoints, (length(grid.ω),), statsfun)
    end
    Luna.run(Eω, grid, linop, transform, FT, output)
    return (output=output, args=args, grid=grid) 
end

function plotprop(solution; tlims=(-30e-15, 30e-15), flims=(0.0, 2e15))
    grid = solution.grid
    output = solution.output
    env = (haskey(solution.args, :fieldkind) && solution.args[:fieldkind] == :env)
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
    