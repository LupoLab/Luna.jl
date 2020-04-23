using Luna

a = 13e-6
gas = :Ar
pres = 5
flength = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

grid = Grid.EnvGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false, model=:full)
aeff = let m=m
    z -> Modes.Aeff(m, z=z)
end

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

linop, βfun, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            # Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(
    grid, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.fwhm_t(grid),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

import PyPlot:pygui, plt
import FFTW

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
    energy[ii] = energyfun(Etout[:, ii])
end
energyω = [energyfunω(Eout[:, ii]) for ii=1:size(Eout, 2)]

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

##
plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.plot(zout.*1e2, energyω.*1e6, "--")
plt.plot(output["stats"]["z"].*1e2, output["stats"]["energy"].*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
##

plt.figure()
plt.plot(to*1e15, abs2.(Eto[:, 121]))
plt.xlim(-20, 20)

plt.figure()
plt.plot(to*1e15, real.(exp.(1im*grid.ω0.*to).*Eto[:, 121]))
plt.plot(t*1e15, real.(exp.(1im*grid.ω0.*t).*Etout[:, 121]))
plt.xlim(-10, 20)

##
plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.plot(zout.*1e2, energyω.*1e6)
plt.plot(output["stats"]["z"].*1e2, output["stats"]["energy"].*1e6)
plt.xlabel("Distance (cm)")
plt.ylabel("Energy (μJ)")

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["energy_RDW"].*1e6)
plt.xlabel("Distance (cm)")
plt.ylabel("RDW Energy (μJ)")

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["peakpower"].*1e-9)
plt.xlabel("Distance (cm)")
plt.ylabel("Peak power (GW)")

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["fwhm_t_min"].*1e15)
plt.plot(output["stats"]["z"].*1e2, output["stats"]["fwhm_t_max"].*1e15)
plt.xlabel("Distance (cm)")
plt.ylabel("FWHM (fs)")

##
plt.figure()
plt.plot(output["stats"]["z"].*1e2, output["stats"]["density"]*1e-6)
plt.xlabel("Distance (cm)")
plt.ylabel("Density (cm\$^{-3}\$)")