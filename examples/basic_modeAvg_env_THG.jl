using Luna

a = 13e-6
gas = :Ar
pres = 5

τfwhm = 30e-15
λ0 = 800e-9
flength = 0.5e-2
energy = 1e-6

grid = Grid.EnvGrid(flength, λ0, (160e-9, 3000e-9), 1e-12, thg=true)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)

energyfun, energyfunω = Fields.energyfuncs(grid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0, thg=true)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)

responses = (Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), 2π*PhysData.c/λ0, grid.to),)

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs, aeff)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.fwhm_t(grid),
                               Stats.density(densityfun))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot: pygui, plt

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
    energy[ii] = energyfun(Etout[:, ii])
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