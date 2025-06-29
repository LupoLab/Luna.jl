using Luna, PythonPlot

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatiliMode(a, (gas, gas), (pres/2, pres/2), loss=false)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

densityfun = let dens0=PhysData.density(gas, pres/2)
    z -> [dens0, dens0]
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = ((Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)),
             (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)))


linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

ppwin = Stats.peakpower(grid, Eω, (150e-9, 300e-9)) # peak power of dispersive wave
statsfun = Stats.default(grid, Eω, m, linop, transform;
                         gas=(gas, gas), windows=((150e-9, 300e-9),), userfuns=(ppwin,))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun) # statsfun

Luna.run(Eω, grid, linop, transform, FT, output)

##

Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.time_1D(output, [5e-2, 10e-2, 11e-2])
Plotting.spec_1D(output, [5e-2, 10e-2, 11e-2])
##
Plotting.spectrogram(output, 9.8e-2, :λ; trange=(-50e-15, 50e-15), λrange=(160e-9, 1200e-9),
                     N=512, fw=3e-15,
                     cmap=Plotting.cmap_white("viridis", n=48))