import Luna
import Luna: Configuration, Maths, Capillary, PhysData
import Logging
import NumericalIntegration
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

input = Configuration.GaussInput(duration=7e-15, energy=30e-6, wavelength=800e-9)
grid = Configuration.RealGrid(trange=1e-12, λ_lims=(190e-9, 4000e-9))
geometry = Configuration.Capillary(radius=75e-6, length=50e-2)
medium = fill=Configuration.StaticFill(:HeJ, 6)
nonlinear = Configuration.GasNonlinear()

cfg = Configuration.Config(grid, geometry, medium, nonlinear, input)

ω, t, zout, Eout, Etout = Luna.run(cfg)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

Elast = Maths.hilbert(Etout[:, 1])
Aeff = Capillary.Aeff(75e-6)
energy = abs(NumericalIntegration.integrate(t, abs2.(Elast), NumericalIntegration.SimpsonEven()))
energy *= PhysData.c*PhysData.ε_0*Aeff/2
println(energy*1e6)

pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-4, 0)
plt.colorbar()

plt.figure()
plt.pcolormesh(t*1e15, zout, abs2.(transpose(Maths.hilbert(Etout))))
plt.xlim(-20, 20)