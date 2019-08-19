import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, Modes
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot: pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

grid = Grid.EnvGrid(0.5e-2, 800e-9, (160e-9, 3000e-9), 1e-12, thg=true)

energyfun = Modes.energy_env_mode_avg(a)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)
end

β1const = Capillary.dispersion(1, a; λ=λ0, gas=gas, pressure=pres)
β0const = Capillary.β(a; λ=λ0, gas=gas, pressure=pres)
βconst = zero(grid.ω)
βconst[grid.sidx] = Capillary.β(a, grid.ω[grid.sidx], gas=gas, pressure=pres)
βconst[.!grid.sidx] .= 1
βfun(ω, m, n, z) = βconst
αfun(ω, m, n, z) = log(10)/10 * 2

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

normfun = Modes.norm_mode_average(grid.ω, βfun)

transform = Modes.trans_env_mode_avg(grid)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env_thg(PhysData.χ3_gas(gas), 2π*PhysData.c/λ0, grid.to),)

in1 = (func=gausspulse, energy=1e-6, m=1, n=1)
inputs = (in1, )

x = Array{ComplexF64}(undef, length(grid.t))
FT = FFTW.plan_fft(x, 1)

linop = -im.*(βconst .- β1const.*(grid.ω .- grid.ω0))
zout, Eout = Luna.run(grid, linop, normfun, energyfun, densityfun,
                             inputs, responses, transform, FT)

ω = grid.ω
t = grid.t

Etout = FFTW.ifft(Eout, 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
It = abs2.(Etout)
Itlog = log10.(Maths.normbymax(It))
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

Et = Etout
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii], 1, 1)
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