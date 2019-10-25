import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Hankel
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import PyPlot:pygui, plt


gas = :Ar
pres = 0

τ = 30e-15
λ0 = 800e-9

w0 = 500e-6
energy = 1e-6

R = 10e-3
N = 256

grid = Grid.RealGrid(2, 800e-9, (400e-9, 3000e-9), 0.2e-12)
q = Hankel.QDHT(R, N, dim=2)

energyfun = NonlinearRHS.energy_radial(q)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ) .* Maths.gauss(q.r, w0/2)'
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
#  Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index(:Ar, λ0))

normfun = NonlinearRHS.norm_radial(grid.ω, q, PhysData.ref_index(:Ar, λ0))

in1 = (func=gausspulse, energy=energy)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, q, energyfun, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω), length(q.r)))

Et = FT \ (q \ Eω)
# println(energyfun(grid.t, Et))
# error()
Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Erout = (q \ Eout)
Itr = abs2.(Erout)
Etout = FFTW.irfft(Erout, length(grid.t), 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

It = PhysData.c * PhysData.ε_0/2 * abs2.(Maths.hilbert(Etout));
Itlog = log10.(Maths.normbymax(It))

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 3)
    energy[ii] = energyfun(t, Etout[:, :, ii])
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))
ωminidx = argmin(abs.(grid.ω .- 2π*PhysData.c/2e-6))

pygui(true)
plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Itr[ω0idx, :, :])
plt.colorbar()
plt.ylim(0, 4)

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Itr[ωminidx, :, :])
plt.colorbar()
plt.ylim(0, 4)


plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
