import FFTW
import Luna: SFA, Tools, Grid, Maths, PhysData, Ionisation
import Luna.PhysData: wlfreq
import PythonPlot: pyplot

τfwhm = 30e-15
λ0 = 800e-9
Ipeak = 3e18
gas = :Ar

grid = Grid.RealGrid(0, λ0, (10e-9, 1500e-9), 4*τfwhm)

It = Maths.gauss.(grid.t; fwhm=τfwhm) .* Ipeak
Et = Tools.intensity_to_field.(It) .* cos.(grid.t .* wlfreq(λ0))

irf! = Ionisation.ionrate_fun!_PPTcached(gas, λ0)

D = SFA.sfa_dipole(grid.t, Et, gas, λ0; gate=true, depletion=true, irf!)
eV = PhysData.ħ .* grid.ω ./ PhysData.electron
Dω = FFTW.rfft(D)
tg = collect(range(extrema(grid.t)...; length=2048))
gab = Maths.gabor(grid.t, D, tg, 300e-18)
##
IpeV = PhysData.ionisation_potential(gas; unit=:eV)
# Ponderomotive energy:
Up = PhysData.electron^2 * maximum(Et)^2/(4*PhysData.m_e * wlfreq(λ0)^2) 
UpeV = Up/PhysData.electron
# Cutoff law:
cutoffeV = IpeV + 3.17UpeV

eV0 = PhysData.ħ*wlfreq(λ0)/PhysData.electron
harmonics = eV./eV0

##
pyplot.figure()
pyplot.semilogy(harmonics, abs2.(Dω), label="Harmonic spectrum")
pyplot.axvline(cutoffeV/eV0, linestyle="--", color="k", label="IP + 3.17 Up")
pyplot.xlim(0, 1.2cutoffeV/eV0)
pyplot.xlabel("Harmonic order")
pyplot.ylabel("Spectral energy density")
pyplot.legend(frameon=false)

##
pyplot.figure()
pyplot.pcolormesh(tg*1e15, eV[1:4:end], Maths.log10_norm(abs2.(gab[1:4:end, :])))
pyplot.axhline(cutoffeV, linestyle="--", color="w")
pyplot.clim(-10, -4)
pyplot.ylim(20, 1.2*cutoffeV)
pyplot.xlim(-τfwhm*1e15, τfwhm*1e15)
cb = pyplot.colorbar()
cb.set_label("Log10 SED")
pyplot.xlabel("Time (fs)")
pyplot.ylabel("Photon energy (eV)")

