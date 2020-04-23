using Luna

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas),length(grid.to)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.peakintensity(grid, aeff),
                               Stats.fwhm_t(grid),
                               Stats.electrondensity(grid, ionrate, densityfun, aeff),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

import FFTW

Etout = FFTW.irfft(Eout, length(grid.t), 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=16)
It = abs2.(Maths.hilbert(Eto))
Itlog = log10.(Maths.normbymax(It))
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(Etout[:, ii])
end

import PyPlot:pygui, plt
pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-6, 0)
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
plt.plot(to*1e15, Eto[:, 121])
plt.xlim(-20, 20)