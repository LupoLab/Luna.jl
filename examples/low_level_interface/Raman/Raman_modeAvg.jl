using Luna

a = 13e-6
gas = :H2
pres = 5
flength = 200e-2

τfwhm = 20e-15
λ0 = 800e-9
energy = 1e-6

grid = Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), 40e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.RamanPolarField(grid.to, Raman.raman_response(grid.to, gas)))

linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas)
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.time_1D(output, range(0,flength,length=5), trange=(-0.5e-12, 2e-12))
Plotting.spec_1D(output, range(0,flength,length=5), λrange=(500e-9, 1400e-9))