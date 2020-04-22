using Luna

a = 13e-6
gas = :Ar
pres = 7.5

τ = 30e-15
λ0 = 800e-9

L = 15e-2

grid = Grid.RealGrid(L, 800e-9, (160e-9, 3000e-9), 1e-12)

coren, dens = Capillary.gradient(gas, L, pres, 0);
m = Capillary.MarcatilliMode(a, coren, loss=false, model=:full)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

energyfun, energyfunω = NonlinearRHS.energy_modal(grid)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop, βfun = LinearOps.make_linop(grid, m, λ0);

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, energyfun, dens, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.peakintensity(grid, aeff),
                               Stats.fwhm_t(grid),
                               Stats.electrondensity(grid, ionrate, dens, aeff),
                               Stats.zdw(m),
                               Stats.pressure(dens, gas))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)