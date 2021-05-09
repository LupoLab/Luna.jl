import FFTW
import Luna: SFA, Tools, Grid, Maths, PhysData, Ionisation
import Luna.PhysData: wlfreq
import PyPlot: plt, pygui
pygui(true)

τfwhm = 20e-15
λ0 = 800e-9
Ipeak = 4e18
gas = :Ar

grid = Grid.RealGrid(0, λ0, (10e-9, 1500e-9), 6*τfwhm)

It = Maths.gauss.(grid.t; fwhm=τfwhm) .* Ipeak
Et = Tools.intensity_to_field.(It) .* cos.(grid.t .* wlfreq(λ0))

irf! = Ionisation.ionrate_fun!_PPTcached(gas, λ0)

D = SFA.sfa_dipole(grid.t, Et, gas, λ0; apod=true, depletion=false, irf!)
eV = PhysData.ħ .* grid.ω ./ PhysData.electron
Dω = FFTW.rfft(D)
tg = collect(range(extrema(grid.t)...; length=2048))
gab = Maths.gabor(grid.t, D, tg, 300e-18)
##
IpeV = PhysData.ionisation_potential(gas; unit=:eV)
Up = PhysData.electron^2 * maximum(Et)^2/(4*PhysData.m_e * wlfreq(λ0)^2)
UpeV = Up/PhysData.electron
cutoffeV = IpeV + 3.17UpeV

eV0 = PhysData.ħ*wlfreq(λ0)/PhysData.electron
harmonics = eV./eV0

##
plt.figure()
plt.semilogy(harmonics, abs2.(Dω))
plt.axvline(cutoffeV/eV0, linestyle="--", color="k")
plt.xlim(0, 1.2cutoffeV/eV0)
plt.xlabel("Harmonic order")
plt.ylabel("Spectral energy density")

##
plt.figure()
plt.pcolormesh(tg*1e15, eV[1:4:end], Maths.log10_norm(abs2.(gab[1:4:end, :])))
plt.clim(-10, -3)
plt.ylim(20, 120)
plt.xlim(-τfwhm*2e15, τfwhm*2e15)
cb = plt.colorbar()
cb.set_label("Log10 SED")


