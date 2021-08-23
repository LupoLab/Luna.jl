# propagation in step index fibre

using Luna

# single mode fibre at 1030 nm
a = 5e-6
NA = 0.08
flength = 5.0

τfwhm = 2e-12
λ0 = 1030e-9
energy = 1e-9

grid = Grid.EnvGrid(flength, λ0, (980e-9, 1080e-9), 10e-12)

m = StepIndexFibre.StepIndexMode(a, NA, accellims=(900e-9, 1200e-9, 100))
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end
densityfun = z -> 1.0

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

responses = (Nonlinear.Kerr_env(PhysData.γ3(:SiO2)),)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform)
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
#Plotting.stats(output)
#Plotting.prop_2D(output)
Plotting.time_1D(output, [0.0, 2.5, 5.0])
Plotting.spec_1D(output, [0.0, 2.5, 5.0])
