using Luna
import Logging
import FFTW
Logging.disable_logging(Logging.BelowMinLevel)

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

L = 15e-2

grid = Grid.EnvGrid(L, 800e-9, (160e-9, 3000e-9), 1e-12)

coren, densityfun = Capillary.gradient(gas, L, pres, pres);
m = Capillary.MarcatilliMode(a, coren, loss=false);
aeff(z) = Modes.Aeff(m, z=z)

energyfun, energyfunω = NonlinearRHS.energy_modal(grid)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)
end

# dens0 = PhysData.density(gas, pres)
# densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

linop, βfun = LinearOps.make_linop(grid, m, λ0);

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.peakpower(grid),
                               Stats.fwhm_t(grid),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

import PyPlot:pygui, plt
ω = grid.ω
t = grid.t
f = FFTW.fftshift(ω, 1)./2π.*1e-15

zout = output["z"]
Eout = output["Eω"]

Etout = FFTW.ifft(Eout, 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=8, dim=1)
It = abs2.(Eto)
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii])
end

pygui(true)
plt.figure()
plt.pcolormesh(f, zout, transpose(FFTW.fftshift(Ilog, 1)))
plt.clim(-6, 0)
plt.xlim(0.19, 1.9)
plt.colorbar()

plt.figure()
plt.pcolormesh(to*1e15, zout, transpose(It))
plt.colorbar()
plt.xlim(-30, 30)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(to*1e15, abs2.(Eto[:, 121]))
plt.xlim(-20, 20)

plt.figure()
plt.plot(to*1e15, real.(exp.(1im*grid.ω0.*to).*Eto[:, 121]))
plt.plot(t*1e15, real.(exp.(1im*grid.ω0.*t).*Etout[:, 121]))
plt.xlim(-10, 20)