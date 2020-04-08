import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot: pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

grid = Grid.EnvGrid(0.5e-2, 800e-9, (160e-9, 3000e-9), 1e-12, thg=true)
#grid = Grid.EnvGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12, thg=true)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)

energyfun = NonlinearRHS.energy_modal()

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0, thg=true)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

responses = (Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), 2π*PhysData.c/λ0, grid.to),)

in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.ifft(Eout, 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
It = abs2.(Etout)
Itlog = log10.(Maths.normbymax(It))
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

Et = Etout
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii])
end

pygui(true)
plt.figure()
plt.pcolormesh(FFTW.fftshift(ω, 1)./2π.*1e-15, zout, transpose(FFTW.fftshift(Ilog, 1)))
plt.clim(-15, 0)
plt.xlim(0.19, 1.9)
plt.colorbar()

plt.figure()
plt.pcolormesh(t*1e15, zout, transpose(It))
plt.colorbar()
plt.xlim(-30, 30)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(t*1e15, abs2.(Et[:, 121]))
plt.xlim(-20, 20)