import Luna.Simple: solveprop, plotprop
import Luna: Grid, Capillary, Field

gas = :Ar
pressure = 5.0
thg = false
grid = Grid.EnvGrid(zmax=15e-2, referenceλ=800e-9, λ_lims=(180e-9, 3000e-9), trange=1e-12, thg=thg)
modes = (Capillary.MarcatilliMode(15e-6, gas, pressure, loss=false),)
fields = (Field.GaussField(τfwhm=30e-15, λ0=800e-9, energy=1e-6),)

solution = solveprop(grid=grid, modes=modes, fields=fields, gas=gas, pressure=pressure,
                     thg=thg, plasma=false)
plotprop(solution)
