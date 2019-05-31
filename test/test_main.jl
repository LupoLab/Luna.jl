import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation
import Logging
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

Aeff = Capillary.Aeff(a)
function energyfun(t, Et, m, n)
    Eta = Maths.hilbert(Et)
    intg = abs(integrate(t, abs2.(Eta), SimpsonEven()))
    return intg * PhysData.c*PhysData.ε_0*Aeff/2
end

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

βfun(ω) = Capillary.β(a, ω, gas=gas, pressure=pres)
αfun(ω) = log(10)/10 * 2

densityfun(z) = PhysData.std_dens * pres

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
plasma! = Nonlinear.make_plasma!(grid.to, grid.ωo, grid.to, ionrate, ionpot)

responses = (Nonlinear.Kerr(PhysData.χ3_gas(gas)), plasma!)

in1 = (func=gausspulse, energy=1e-6, m=1, n=1)
inputs = (in1, )

zout, Eout, Etout = Luna.run(grid, βfun, αfun, energyfun, densityfun, inputs, responses)

ω = grid.ω
t = grid.t

Ilog = log10.(Maths.normbymax(abs2.(Eout)))
It = abs2.(Maths.hilbert(Etout))
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))
Itlog = log10.(Maths.normbymax(It))

idcs = @. (t < 100e-15) & (t >-100e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=4)

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii], 1, 1)
end

pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-4, 0)
plt.colorbar()

plt.figure()
plt.pcolormesh(t*1e15, zout, transpose(It))
# plt.clim(-4, 0)
plt.colorbar()
# plt.xlim(-20, 20)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(to*1e15, Eto[:, 121])
plt.xlim(-20, 20)