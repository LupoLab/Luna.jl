using Luna

a = 9e-6
gas = :Ar
pres = 5
flength = 25e-2

τfwhm = 30e-15
λ0 = 1500e-9
energy = 1.7e-6

grid = Grid.RealGrid(flength, λ0, (200e-9, 3000e-9), 2e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             plasma)

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output, :λ, λrange=(500e-9,1800e-9), trange=(-500e-15,100e-15), dBmin=-30.0)
Plotting.time_1D(output, [5e-2, 15e-2, 25e-2])
Plotting.spec_1D(output, [5e-2, 15e-2, 25e-2])
