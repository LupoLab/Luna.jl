import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Hankel, Plotting
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import PyPlot:pygui, plt

gas = :Ar
pres = 1.2

τ = 20e-15
λ0 = 800e-9

w0 = 40e-6
energy = 2e-9
L = 0.6

R = 4e-3
N = 1024

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, N, dim=2)

energyfun = NonlinearRHS.energy_radial(q)

function prop(E, z)
    Eω = FFTW.rfft(E, 1)
    Eωk = q * Eω
    kzsq = @. (grid.ω/PhysData.c)^2 - (q.k^2)'
    kzsq[kzsq .< 0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω/PhysData.c))
    Eω = q \ Eωk
    E = FFTW.irfft(Eω, length(grid.t), 1)
    return E
end

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ) .* Maths.gauss(q.r, w0/2)'
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
    return prop(Et, -0.3)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))

normfun = NonlinearRHS.norm_radial(grid.ω, q, PhysData.ref_index_fun(gas, pres))

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
# Iω0 = Iωr[:, 1, :]
Er0 = dropdims(Hankel.onaxis(Eout, q), dims=2);
Iω0 = abs2.(Er0);
Iω0log = log10.(Maths.normbymax(Iω0));
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

zr = π*w0^2/λ0
points = L/2 .+ [-15, 3, 21].*zr
idcs = [argmin(abs.(zout .- point)) for point in points]

Epoints = [Hankel.symmetric(Et[:, :, idxi], q) for idxi in idcs]
rsym = Hankel.Rsymmetric(q);

pygui(true)
Iλ0 = Iωr[ω0idx, :, :]
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L/2*λ0/(π*w0^2))^2)
# w1 = w0
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
# plt.pcolormesh(zout*1e2, q.r*1e3, log10.(Maths.normbymax(Ir)))
plt.colorbar()
plt.ylim(0, R*1e3)
# plt.clim(-6, 0)
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

jw = Plotting.cmap_white("jet")
fig = plt.figure()
fig.set_size_inches(12, 4)
for ii in 1:3
    plt.subplot(1, 3, ii)
    plt.pcolormesh(grid.t*1e15, rsym*1e3, abs2.(Epoints[ii]'), cmap=jw)
    plt.xlim(-42, 42)
    plt.ylim(-1.8, 1.8)
end