using Luna, PythonPlot

a = 13e-6
gas = :Ar
pres = 5

τfwhm = 30e-15
λ0 = 800e-9
flength = 0.5e-2
energy = 1e-6

grid = Grid.EnvGrid(flength, λ0, (160e-9, 3000e-9), 1e-12, thg=true)

m = Capillary.MarcatiliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)

energyfun, energyfunω = Fields.energyfuncs(grid)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0, thg=true)


responses = (Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), 2π*PhysData.c/λ0, grid.to),)

    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

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
import PythonPlot: pygui, pyplot

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
pyplot.figure()
pyplot.pcolormesh(FFTW.fftshift(ω, 1)./2π.*1e-15, zout, transpose(FFTW.fftshift(Ilog, 1)))
pyplot.clim(-15, 0)
pyplot.xlim(0.19, 1.9)
pyplot.colorbar()

pyplot.figure()
pyplot.pcolormesh(t*1e15, zout, transpose(It))
pyplot.colorbar()
pyplot.xlim(-30, 30)

pyplot.figure()
pyplot.plot(zout.*1e2, energy.*1e6)
pyplot.xlabel("Distance [cm]")
pyplot.ylabel("Energy [μJ]")

pyplot.figure()
pyplot.plot(t*1e15, abs2.(Et[:, 121]))
pyplot.xlim(-20, 20)
