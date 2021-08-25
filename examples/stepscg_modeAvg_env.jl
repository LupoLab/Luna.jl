# supercontinuum in strand of silica in air

using Luna

# single mode fibre at 1030 nm
a = 1.25e-6
flength = 2.5e-2
fr = 0.18
τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12

grid = Grid.EnvGrid(flength, λ0, (400e-9, 1400e-9), 10e-12)

m = StepIndexFibre.StepIndexMode(a, accellims=(400e-9, 1400e-9, 100))
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end
densityfun = z -> 1.0

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

responses = (Nonlinear.Kerr_env((1 - fr)*PhysData.χ3(:SiO2)),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(:SiO2, fr*PhysData.ε_0*PhysData.χ3(:SiO2))))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform)
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
#Plotting.stats(output)
#Plotting.prop_2D(output)
#Plotting.time_1D(output, [0.0, 2.5, 5.0], trange=(-5e-12, 5e-12))
Plotting.spec_1D(output, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5].*1e-2, λrange=(400e-9, 1300e-9))
