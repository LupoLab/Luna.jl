import Luna
import Luna: Configuration

import PyPlot:pygui, plt

input = Configuration.GaussInput(duration=7e-15, energy=30e-6, wavelength=800e-9)
grid = Configuration.RealGrid()
geometry = Configuration.Capillary(radius=75e-6, length=3e-3,
                                   fill=Configuration.StaticFill(:He, 0))
nonlinear = Configuration.GasNonlinear()

cfg = Configuration.Config(input, geometry, nonlinear, grid)

ω, zout, Eout = Luna.run(cfg)
# Luna.run(cfg)

println(size(Eout))

pygui(true)
plt.figure()
plt.pcolormesh(ω, zout, abs2.(transpose(Eout)))