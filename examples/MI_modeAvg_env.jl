using Luna

a = 15e-6
gas = :Ar
pres = 25

τfwhm = 600e-15
λ0 = 800e-9
flength = 80e-2
energy = 10e-6

grid = Grid.EnvGrid(flength, λ0, (220e-9, 3000e-9), 4e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)


ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            # Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = (Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output; trange=(-1e-12, 1e-12), λrange=(220e-9, 2000e-9))
Plotting.time_1D(output; trange=(-1e-12, 1e-12))
Plotting.spec_1D(output)