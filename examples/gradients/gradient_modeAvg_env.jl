using Luna

a = 13e-6
gas = :Ar
pres = 7.5

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

L = 15e-2

grid = Grid.EnvGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

coren, densityfun = Capillary.gradient(gas, L, pres, 0);
m = Capillary.MarcatilliMode(a, coren, loss=false);
aeff = let m = m
    aeff(z) = Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

linop!, βfun! = LinearOps.make_linop(grid, m, λ0);

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.peakpower(grid),
                               Stats.fwhm_t(grid),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop!, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)