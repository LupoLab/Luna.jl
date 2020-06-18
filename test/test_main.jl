import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap
using Dierckx

import PyPlot:pygui, plt

filename = "ProbeData.txt"

a = 125e-6
gas = :Ar
pres = 0.5

τ = 10e-15
λ0 = 800e-9

grid = Grid.RealGrid(1, 800e-9, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatilliMode(a, gas, pres)

energyfun = NonlinearRHS.energy_mode_avg(m)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

function pulseFromFile(filename)
    data = readdlm(filename)
    t0 = vec(data[:,1])
    Et0 = vec(data[:,2])
    Et = Spline1D(t0, Et0; k=3, bc="zero", s=0.0)
    function pulse(t)
        return @. Et(t)
    end
    return pulse
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)

# in1 = (func=gausspulse, energy=70e-6)
in1 = (func=pulseFromFile(filename), energy=40e-9)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.irfft(Eout, length(grid.t), 1)

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

idcs = @. (t < 30e-12) & (t >-30e-12)
to, Eto = Maths.oversample(t[idcs], Etout[idcs, :], factor=16)
It = abs2.(Maths.hilbert(Eto))
Itlog = log10.(Maths.normbymax(It))
zpeak = argmax(dropdims(maximum(It, dims=1), dims=1))

Et = Maths.hilbert(Etout)
energy = zeros(length(zout))
for ii = 1:size(Etout, 2)
    energy[ii] = energyfun(t, Etout[:, ii])
end

pygui(true)
plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-6, 0)
plt.colorbar()

# plt.figure()
# plt.pcolormesh(to*1e15, zout, transpose(It))
# plt.colorbar()
# plt.xlim(-300, 300)

plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")

plt.figure()
plt.plot(to*1e15, Eto[:, 121])
plt.xlim(-300, 300)

Elout = zeros(Float64, size(Eout));
λ = 2pi*PhysData.c./ω;
for ii in 1:length(Elout[1,:])
    Elout[:,ii] = abs2.(Eout[:,ii])./(λ.^2)*PhysData.c;
end;

lspec = log10.(Elout);
lspec = lspec .- maximum(lspec);
λb = 100 .< λ*1e9 .< 1200;

pygui(true)
plt.figure()
plt.pcolormesh(λ[λb]*1e9, zout, transpose(lspec[λb,:]), cmap="jet")
plt.clim(-4, 0)
plt.colorbar()