import Luna
import Luna: Configuration, Maths

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

input = Configuration.GaussInput(duration=7e-15, energy=30e-6, wavelength=800e-9)
grid = Configuration.RealGrid(tmax=0.5e-12, λ_lims=(200e-9, 4000e-9))
geometry = Configuration.Capillary(radius=75e-6, length=50e-2,
                                   fill=Configuration.StaticFill(:He, 6))
nonlinear = Configuration.GasNonlinear()

cfg = Configuration.Config(input, geometry, nonlinear, grid)

ω, t, zout, Eout, Etout = Luna.run(cfg)

# println(size(Eout))

Ilog = log10.(Maths.normmax(abs2.(Eout)))

pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-4, 0)
plt.colorbar()

plt.figure()
plt.pcolormesh(t*1e15, zout, abs2.(transpose(Etout)))

plt.figure()
plt.plot(t*1e15, Etout[:, end])

plt.figure()
plt.plot(ω./2π.*1e-15, unwrap(angle.(Eout[:, end])))