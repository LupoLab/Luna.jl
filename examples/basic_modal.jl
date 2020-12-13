using Luna

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

modes = (
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

energyfun, energyfunω = Fields.energyfuncs(grid)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_PPTcached(gas, λ0)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes, :y; full=false)
                              
linop = LinearOps.make_const_linop(grid, modes, λ0)

statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.prop_2D(output; bandpass=(150e-9, 300e-9), modes=:sum)
Plotting.time_1D(output, [5e-2, 9.8e-2])
Plotting.time_1D(output, [5e-2, 9e-2], modes=:sum, bandpass=(150e-9, 300e-9))
Plotting.spec_1D(output, [5e-2, 9.8e-2])
Plotting.spec_1D(output, [5e-2, 9.8e-2], modes=:sum)

