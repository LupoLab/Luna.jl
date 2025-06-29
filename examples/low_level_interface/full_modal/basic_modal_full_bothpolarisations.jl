using Luna, PythonPlot

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy=1e-6

modes = (
    Capillary.MarcatiliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatiliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false),
)
nmodes = length(modes)

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

energyfun, energyfunω = Fields.energyfuncs(grid)
normfun = NonlinearRHS.norm_modal(grid.ω)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes,
                               :xy; full=true)
                 
statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)


Plotting.stats(output)
Plotting.prop_2D(output)