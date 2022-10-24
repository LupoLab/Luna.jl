# supercontinuum in strand of silica in air

using Luna

# single mode fibre at 1030 nm
a = 1.25e-6
flength = 15e-2
fr = 0.18
τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12

grid = Grid.EnvGrid(flength, λ0, (400e-9, 1400e-9), 10e-12)

m = StepIndexFibre.StepIndexMode(a, accellims=(400e-9, 1400e-9, 100))
aeff = let aeffc = Modes.Aeff(m, z=0)
    z -> aeffc
end
densityfun = z -> 1.0

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

responses = (Nonlinear.Kerr_env((1 - fr)*PhysData.χ3(:SiO2)),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(grid.to, :SiO2, fr*PhysData.ε_0*PhysData.χ3(:SiO2))))

inputs = (Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)


output = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
#Plotting.stats(output)
Plotting.prop_2D(output, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))
#Plotting.time_1D(output, [0.0, 2.5, 5.0], trange=(-5e-12, 5e-12))
Plotting.spec_1D(output, range(0.0, 1.0, length=5).*flength, λrange=(400e-9, 1300e-9))
