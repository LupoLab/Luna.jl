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
energy = 1.5e-3
L = 2

R = 6e-3
N = 128

grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)

Dx = 2R/N
n = collect(range(0, length=N))
x = @. (n-N/2) * Dx
y = copy(x)

r = sqrt.(reshape(x, (1, 1, N)).^2 .+ reshape(y, (1, N)).^2)

xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
FT = FFTW.plan_fft(xr, (1, 2, 3), flags=FFTW.MEASURE)

energyfun = NonlinearRHS.energy_free_env(x, y)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ) .* Maths.gauss.(r, w0/2)
    Et = @. sqrt(It)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

linop = LinearOps.make_const_linop(grid, x, y, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.norm_free(grid.ω, x, y, PhysData.ref_index_fun(gas, pres))

in1 = (func=gausspulse, energy=energy)
inputs = (in1, )

Eω, transform, FTo = Luna.setup(grid, FT, x, y, energyfun, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 21, (length(grid.ω), N, N))

xwin = Maths.planck_taper(x, -R, -0.8R, 0.8R, R)
xywin = reshape(xwin, (1, length(xwin))) .* reshape(xwin, (1, 1, length(xwin)))

Luna.run(Eω, grid, linop, transform, FT, output, max_dz=Inf, init_dz=0.1)

ω = FFTW.fftshift(grid.ω)
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

println("Transforming...")
Eωyx = FFTW.fftshift(FFTW.ifft(Eout, (2, 3)), 1);
Etyx = FFTW.ifft(Eout, (1, 2, 3))
println("...done")

Eout = FFTW.fftshift(Eout, (2, 3))

Ilog = log10.(Maths.normbymax(abs2.(Eωyx)))

Iωyx = abs2.(Eωyx);

Iyx = zeros(Float64, (length(y), length(x), length(zout)));
energy = zeros(length(zout));
for ii = 1:size(Etyx, 4)
    energy[ii] = energyfun(t, Etyx[:, :, :, ii]);
    Iyx[:, :, ii] = (grid.ω[2]-grid.ω[1]) .* sum(Iωyx[:, :, :, ii], dims=1);
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

E0ωyx = FFTW.ifft(Eω[ω0idx, :, :], (1, 2));

Iωyx = abs2.(Eωyx);
Iωyxlog = log10.(Maths.normbymax(Iωyx));

pygui(true)
plt.figure()
plt.pcolormesh(x, y, (abs2.(E0ωyx)))
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("I(ω=ω0, x, y, z=0)")

plt.figure()
plt.pcolormesh(x, y, (abs2.(Eωyx[ω0idx, :, :, end])))
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("I(ω=ω0, x, y, z=L)")

plt.figure()
plt.pcolormesh(zout, ω.*1e-15/2π, Iωyxlog[:, N÷2+1, N÷2+1, :])
plt.xlabel("Z (m)")
plt.ylabel("f (PHz)")
plt.title("I(ω, x=0, y=0, z)")
plt.clim(-6, 0)
plt.colorbar()

plt.figure()
plt.pcolormesh(x.*1e3, y.*1e3, Iyx[:, :, 1])
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("\$\\int I(\\omega, x, y, z=0) d\\omega\$")
plt.colorbar()

plt.figure()
plt.pcolormesh(x.*1e3, y.*1e3, Iyx[:, :, end])
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("\$\\int I(\\omega, x, y, z=L) d\\omega\$")
plt.colorbar()

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
