using Luna
import Luna.PhysData: wlfreq
import FFTW
import Hankel
import NumericalIntegration: integrate, SimpsonEven

a = 15e-6
gas = :Ar
pres = 5.0

τ = 30e-15
λ0 = 800e-9

w0 = 0.64*a
energy = 1e-6
L = 0.15

R = a
N = 32

grid = Grid.RealGrid(L, λ0, (200e-9, 3000e-9), 0.6e-12)
q = Hankel.QDHT(R, N, dim=2)

energyfun, energyfun_ω = Fields.energyfuncs(grid, q)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_PPTcached(gas, λ0)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))

normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

Eω, transform, FT = Luna.setup(grid, q, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
#output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω), length(q.r)))
output = Output.MemoryOutput(0, grid.zmax, 201)
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
    energy[ii] = energyfun(Etout[:, :, ii]);
    Ir[:, ii] = integrate(grid.ω, Iωr[:, :, ii], SimpsonEven());
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

zr = π*w0^2/λ0
points = L/2 .+ [-15, 3, 21].*zr
idcs = [argmin(abs.(zout .- point)) for point in points]

Epoints = [Hankel.symmetric(Et[:, :, idxi], q) for idxi in idcs]
rsym = Hankel.Rsymmetric(q);

import PyPlot:pygui, plt
pygui(true)

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Iωr[ω0idx, :, :])
plt.colorbar()
plt.ylim(0, 4)
plt.xlabel("z (cm)")
plt.ylabel("r (mm)")
plt.title("I(r, ω=ω0)")

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, It[length(grid.t)÷2, :, :])
plt.colorbar()
plt.ylim(0, 4)
plt.xlabel("z (cm)")
plt.ylabel("r (mm)")
plt.title("I(r, t=0)")

plt.figure()
plt.pcolormesh(zout*1e2, q.r*1e3, Ir)
# plt.pcolormesh(zout*1e2, q.r*1e3, log10.(Maths.normbymax(Ir)))
plt.colorbar()
plt.ylim(0, R*1e3)
# plt.clim(-6, 0)
plt.xlabel("z (cm)")
plt.ylabel("r (mm)")
plt.title("\$\\int I(r, \\omega) d\\omega\$")

plt.figure()
plt.pcolormesh(zout*1e2, grid.ω*1e-15/2π, Iω0log)
plt.colorbar()
plt.clim(0, -6)
plt.xlabel("z (cm)")
plt.ylabel("f (PHz)")
plt.title("I(r=0, ω)")

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

jw = Plotting.cmap_white("jet", 512, 10)
fig = plt.figure()
fig.set_size_inches(12, 4)
for ii in 1:3
    plt.subplot(1, 3, ii)
    plt.pcolormesh(grid.t*1e15, rsym*1e3, abs2.(Epoints[ii]'), cmap=jw)
    plt.xlim(-42, 42)
    plt.ylim(-1.8, 1.8)
end
