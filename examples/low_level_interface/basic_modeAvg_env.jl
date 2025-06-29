using Luna, PythonPlot

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

grid = Grid.EnvGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatiliMode(a, gas, pres, loss=false, model=:full)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

##

Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.time_1D(output, [5e-2, 10e-2, 11e-2])
Plotting.time_1D(output, [5e-2, 10e-2, 11e-2], y=:Et, oversampling=16)
Plotting.spec_1D(output, [5e-2, 10e-2, 11e-2])
