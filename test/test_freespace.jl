import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Hankel
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import PyPlot:pygui, plt

gas = :Ar
pres = 4

τ = 30e-15
λ0 = 800e-9

w0 = 2e-3
energy = 1e-3
L = 2

R = 10e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
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

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index(gas, λ0))

normfun = NonlinearRHS.norm_radial(grid.ω, q, PhysData.ref_index(gas, λ0))

in1 = (func=gausspulse, energy=energy)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, q, energyfun, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω), length(q.r)))

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Erout = (q \ Eout)
Iωr = abs2.(Erout)
Iω0 = Iωr[:, 1, :]
Iω0log = log10.(Maths.normbymax(Iω0))
Etout = FFTW.irfft(Erout, length(grid.t), 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

It = PhysData.c * PhysData.ε_0/2 * abs2.(Maths.hilbert(Etout));
Itlog = log10.(Maths.normbymax(It))

Ir = zeros(Float64, (length(q.r), length(zout)))

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 3)
    energy[ii] = energyfun(t, Etout[:, :, ii]);
    Ir[:, ii] = integrate(grid.ω, Iωr[:, :, ii], SimpsonEven());
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

pygui(true)
Iλ0 = Iωr[ω0idx, :, :]
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L*λ0/(π*w0^2))^2)
Iλ0_analytic = Maths.gauss(q.r, w1/2)*(w0/w1)^2 # analytical solution (in paraxial approx)
plt.figure()
plt.plot(q.r*1e3, Maths.normbymax(Iλ0[:, end]))
plt.plot(q.r*1e3, Maths.normbymax(Iλ0_analytic), "--")

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Iωr[ω0idx, :, :])
plt.colorbar()
plt.ylim(0, 4)
plt.xlabel("z (m)")
plt.ylabel("r (m)")
plt.title("I(r, ω=ω0)")

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, It[length(grid.t)÷2, :, :])
plt.colorbar()
plt.ylim(0, 4)
plt.xlabel("z (m)")
plt.ylabel("r (m)")
plt.title("I(r, t=0)")

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Ir)
plt.colorbar()
plt.ylim(0, 4)
plt.xlabel("z (m)")
plt.ylabel("r (m)")
plt.title("\$\\int I(r, \\omega) d\\omega\$")

plt.figure()
plt.pcolormesh(zout*1e2, grid.ω*1e-15/2π, Iω0log)
plt.colorbar()
plt.clim(0, -6)
plt.xlabel("z (m)")
plt.ylabel("f (PHz)")
plt.title("I(r=0, ω)")


plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
