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

w0 = 2e-3
energy = 1e-3
L = 10

R = 10e-3
N = 64

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)

Dx = 2R/N
n = collect(range(0, length=N))
x = @. (n-N/2) * Dx
y = copy(x)

r = sqrt.(reshape(x, (1, 1, N)).^2 .+ reshape(y, (1, N)).^2)

xr = Array{Float64}(undef, length(grid.t), length(y), length(x))
FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=FFTW.MEASURE)

energyfun = NonlinearRHS.energy_free(x, y)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ) .* Maths.gauss.(r, w0/2)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
#  Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop = LinearOps.make_const_linop(grid, x, y, PhysData.ref_index(gas, λ0))

normfun = NonlinearRHS.norm_free(grid.ω, x, y, PhysData.ref_index(gas, λ0))

in1 = (func=gausspulse, energy=energy)
inputs = (in1, )

Eω, transform, FTo = Luna.setup(grid, FT, x, y, energyfun, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 21, (length(grid.ω), N, N))

xwin = Maths.planck_taper(x, -R, -0.8R, 0.8R, R)
xywin = reshape(xwin, (1, length(xwin))) .* reshape(xwin, (1, 1, length(xwin)))

Luna.run(Eω, grid, linop, transform, FT, output, xywin, max_dz=Inf)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

println("Transforming...")
Eωyx = FFTW.ifft(Eout, (2, 3))
Etyx = FFTW.irfft(Eout, length(grid.t), (1, 2, 3))
println("...done")

Ilog = log10.(Maths.normbymax(abs2.(Eωyx)))


energy = zeros(length(zout))
for ii = 1:size(Etyx, 4)
    energy[ii] = energyfun(t, Etyx[:, :, :, ii])
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

E0ωyx = FFTW.ifft(Eω[ω0idx, :, :], (1, 2));

pygui(true)
plt.figure()
plt.pcolormesh(x, y, (abs2.(E0ωyx)))
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.figure()
plt.pcolormesh(x, y, (abs2.(Eωyx[ω0idx, :, :, end])))
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
