using Luna

a = 15e-6
gas = :Ar
pres = 25

τfwhm = 600e-15
λ0 = 800e-9
flength = 80e-2
energy = 10e-6

grid = Grid.EnvGrid(flength, λ0, (220e-9, 3000e-9), 4e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            # Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = (Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.peakpower(grid),
                               Stats.fwhm_t(grid),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot:pygui, plt

ω = grid.ω
t = grid.t
f = FFTW.fftshift(ω, 1)./2π.*1e-15

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.ifft(Eout, 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 500e-15) & (t >-500e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=8, dim=1)
It = abs2.(Eto)
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(Etout[:, ii])
end

pygui(true)
plt.figure()
plt.pcolormesh(f, zout, transpose(FFTW.fftshift(Ilog, 1)))
plt.clim(-6, 0)
plt.xlim(0.1, 1)
plt.colorbar()

plt.figure()
plt.pcolormesh(to*1e15, zout, transpose(It))
plt.colorbar()
plt.xlim(-400, 400)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(to*1e15, abs2.(Eto[:,end]))
plt.xlim(-400, 400)

plt.figure()
plt.plot(to*1e15, real.(exp.(1im*grid.ω0.*to).*Eto[:, 121]))
plt.plot(t*1e15, real.(exp.(1im*grid.ω0.*t).*Etout[:, 121]))
plt.xlim(-400, 400)