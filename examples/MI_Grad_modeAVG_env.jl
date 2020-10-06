using Luna
using Formatting

a = 15e-6
gas = :Ar
pres = 7

τfwhm = 150e-15
λ0 = 1030e-9
flength = 0.4
energy = 20e-6

folder = format("{}_{}bar_{}fs_{}uJ\\", string(gas), pres, τfwhm, energy)
dir = raw"C:\Users\mo_is\Dropbox (Heriot-Watt University Team)\RES_EPS_Lupo\Projects\Mohammed\phd\simulation data\MI\\"*folder

grid = Grid.EnvGrid(flength, λ0, (220e-9, 3000e-9), 4e-12)

coren, densityfun = Capillary.gradient(gas, flength, 0, pres);
m = Capillary.MarcatilliMode(a, coren, loss=false);
aeff = let m = m
    aeff(z) = Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

linop!, βfun! = LinearOps.make_linop(grid, m, λ0);

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            # Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = (Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop!, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.HDF5Output(dir*"file.h5", 0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop!, transform, FT, output)

##
Plotting.pygui(false)
Plotting.stats(dir, output)
Plotting.prop_2D(dir, output; trange=(-1e-12, 1e-12), λrange=(180e-9, 2000e-9))
Plotting.time_1D(dir, output; trange=(-1e-12, 1e-12))
Plotting.spec_1D(dir, output, λrange=(180e-9, 1000e-9))