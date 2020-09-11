using Luna

a = 50e-6
b = 10e-6
gas = :Ar
pres = 5
L = 15e-2/10

τfwhm = 30e-15
λ0 = 800e-9
energy = 5e-6

grid = Grid.RealGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

modes = collect(RectModes.RectMode(a, b, gas, pres, :Ag, n=n, m=m) for m in 1:3 for n in 1:6)
nmodes = length(modes)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes, :x; full=true)

linop = LinearOps.make_const_linop(grid, modes, λ0)
statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.prop_2D(output, bandpass=(150e-9, 300e-9))
Plotting.time_1D(output, [5e-2, 9.8e-2])
Plotting.time_1D(output, [5e-2, 9e-2], modes=:sum, bandpass=(150e-9, 300e-9))
Plotting.spec_1D(output, [5e-2, 9.8e-2])
Plotting.spec_1D(output, [5e-2, 9.8e-2], modes=:sum)
