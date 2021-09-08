using Luna

a = 9e-6
gas = :Ar
pres = 5
flength = 25e-2

τfwhm = 30e-15
λ0 = 1500e-9
energy = 1.7e-6

grid = Grid.RealGrid(flength, λ0, (200e-9, 3000e-9), 2e-12)

modes = (
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
)
nmodes = length(modes)

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to,
                                  ionrate, ionpot)
                                  
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             plasma)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes,
                               :y; full=false)

statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.HDF5Output("modalnonvector_lin0deg.h5", 0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output, :λ, λrange=(500e-9,1800e-9), trange=(-500e-15,100e-15), dBmin=-30.0)

