import Luna
import Luna: Grid, Maths, Capillary, AbstractModes, PhysData, Nonlinear, Ionisation, Modes
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

grid = Grid.EnvGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatilliMode(a, gas, pres)

energyfun = Modes.energy_env_mode_avg(m)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)
end

β1const = AbstractModes.dispersion(m, 1, λ=λ0)
β0const = AbstractModes.β(m, λ=λ0)
βconst = zero(grid.ω)
βconst[grid.sidx] = AbstractModes.β(m, grid.ω[grid.sidx])
βconst[.!grid.sidx] .= 1
βfun(ω, m, n, z) = βconst
αfun(ω, m, n, z) = log(10)/10 * 2

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

normfun = Modes.norm_mode_average(grid.ω, βfun)

transform = Modes.trans_env_mode_avg(grid)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            # Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

in1 = (func=gausspulse, energy=1e-6, m=1, n=1)
inputs = (in1, )

x = Array{ComplexF64}(undef, length(grid.t))
FT = FFTW.plan_fft(x, 1, flags=FFTW.MEASURE)

linop = -im.*(βconst .- β1const.*(grid.ω .- grid.ω0) .- β0const)
zout, Eout = Luna.run(grid, linop, normfun, energyfun, densityfun,
                             inputs, responses, transform, FT)

ω = grid.ω
t = grid.t
f = FFTW.fftshift(ω, 1)./2π.*1e-15

Etout = FFTW.ifft(Eout, 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-15) & (t >-30e-15)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=8, dim=1)
It = abs2.(Eto)
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii], 1, 1)
end

pygui(true)
plt.figure()
plt.pcolormesh(f, zout, transpose(FFTW.fftshift(Ilog, 1)))
plt.clim(-6, 0)
plt.xlim(0.19, 1.9)
plt.colorbar()

plt.figure()
plt.pcolormesh(to*1e15, zout, transpose(It))
plt.colorbar()
plt.xlim(-30, 30)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(to*1e15, abs2.(Eto[:, 121]))
plt.xlim(-20, 20)

plt.figure()
plt.plot(to*1e15, real.(exp.(1im*grid.ω0.*to).*Eto[:, 121]))
plt.plot(t*1e15, real.(exp.(1im*grid.ω0.*t).*Etout[:, 121]))
plt.xlim(-10, 20)