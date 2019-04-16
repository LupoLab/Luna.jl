import Luna
import Luna: Configuration, Maths, Capillary, PhysData
import Logging
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

input = Configuration.GaussInput(duration=30e-15, energy=1e-6, wavelength=800e-9)
grid = Configuration.RealGrid(trange=1e-12, λ_lims=(150e-9, 2000e-9), δt=50e-18)
# geometry = Configuration.Capillary(radius=13e-6, length=15e-2)
geometry = Configuration.HCPCF(radius=13e-6, length=15e-2)
medium = fill=Configuration.StaticFill(:Ar, 5)
nonlinear = Configuration.GasNonlinear()

cfg = Configuration.Config(grid, geometry, medium, nonlinear, input)

ω, t, zout, Eout, Etout, window, twindow, prefac = Luna.run(cfg)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=4)

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = abs(integrate(t, abs2.(Et[:, ii]), SimpsonEven()))
end
Aeff = Capillary.Aeff(125e-6)
energy *= PhysData.c*PhysData.ε_0*Aeff/2

pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-4, 0)
plt.colorbar()

plt.figure()
plt.pcolormesh(to*1e15, zout, abs2.(transpose(Maths.hilbert(Eto))))
plt.xlim(-20, 20)

plt.figure()
plt.plot(zout, energy.*1e6)

plt.figure()
plt.plot(to*1e15, Eto[:, end])
plt.xlim(-20, 20)