using Luna
import FFTW
import Luna: Hankel
import NumericalIntegration: integrate, SimpsonEven

gas = :Ar
pres = 1.2

τ = 20e-15
λ0 = 800e-9

w0 = 40e-6
energy = 2e-9
L = 0.6

R = 4e-3
N = 512

grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, N, dim=2)

energyfun, energyfun_ω = Fields.energyfuncs(grid, q)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))

normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.3)

Eω, transform, FT = Luna.setup(grid, q, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = FFTW.fftshift(grid.ω)
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"];
Eout = FFTW.fftshift(Eout, 1);

Erout = q \ Eout
Iωr = abs2.(Erout)
Er0 = dropdims(Hankel.onaxis(Eout, q), dims=2);
Iω0 = abs2.(Er0);
Iω0log = log10.(Maths.normbymax(Iω0));
Etout = FFTW.ifft(FFTW.fftshift(Erout, 1), 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

It = PhysData.c * PhysData.ε_0/2 * abs2.(Etout);
Itlog = log10.(Maths.normbymax(It))

Ir = zeros(Float64, (length(q.r), length(zout)))

energy = zeros(length(zout))
for ii = 1:size(Etout, 3)
    energy[ii] = energyfun(Etout[:, :, ii]);
    Ir[:, ii] = integrate(ω, Iωr[:, :, ii], SimpsonEven());
end

ω0idx = argmin(abs.(ω .- 2π*PhysData.c/λ0))

zr = π*w0^2/λ0
points = L/2 .+ [-15, 3, 21].*zr
idcs = [argmin(abs.(zout .- point)) for point in points]

Epoints = [Hankel.symmetric(Etout[:, :, idxi], q) for idxi in idcs]
rsym = Hankel.Rsymmetric(q);

import PythonPlot: pygui, pyplot
pygui(true)
Iλ0 = Iωr[ω0idx, :, :]
w1 = w0*sqrt(1+(L/2*λ0/(π*w0^2))^2)
# w1 = w0
Iλ0_analytic = Maths.gauss.(q.r, w1/2)*(w0/w1)^2 # analytical solution (in paraxial approx)
pyplot.figure()
pyplot.plot(q.r*1e3, Maths.normbymax(Iλ0[:, end]))
pyplot.plot(q.r*1e3, Maths.normbymax(Iλ0_analytic), "--")

pyplot.figure()
pyplot.pcolormesh(zout*1e2, q.r*1e3, Iωr[ω0idx, :, :])
pyplot.colorbar()
pyplot.ylim(0, 4)
pyplot.xlabel("z (m)")
pyplot.ylabel("r (m)")
pyplot.title("I(r, ω=ω0)")

pyplot.figure()
pyplot.pcolormesh(zout*1e2, q.r*1e3, It[length(grid.t)÷2, :, :])
pyplot.colorbar()
pyplot.ylim(0, 4)
pyplot.xlabel("z (m)")
pyplot.ylabel("r (m)")
pyplot.title("I(r, t=0)")

pyplot.figure()
pyplot.pcolormesh(zout*1e2, q.r*1e3, Ir)
# pyplot.pcolormesh(zout*1e2, q.r*1e3, log10.(Maths.normbymax(Ir)))
pyplot.colorbar()
pyplot.ylim(0, R*1e3)
# pyplot.clim(-6, 0)
pyplot.xlabel("z (m)")
pyplot.ylabel("r (m)")
pyplot.title("\$\\int I(r, \\omega) d\\omega\$")

pyplot.figure()
pyplot.pcolormesh(zout*1e2, ω*1e-15/2π, Iω0log)
pyplot.colorbar()
pyplot.clim(0, -6)
pyplot.xlabel("z (m)")
pyplot.ylabel("f (PHz)")
pyplot.title("I(r=0, ω)")

pyplot.figure()
pyplot.plot(zout.*1e2, energy.*1e6)
pyplot.xlabel("Distance [cm]")
pyplot.ylabel("Energy [μJ]")

jw = Plotting.cmap_white("jet"; N=512, n=10)
fig = pyplot.figure()
fig.set_size_inches(12, 4)
for ii in 1:3
    pyplot.subplot(1, 3, ii)
    pyplot.pcolormesh(grid.t*1e15, rsym*1e3, abs2.(Epoints[ii]'), cmap=jw)
    pyplot.xlim(-42, 42)
    pyplot.ylim(-1.8, 1.8)
end
