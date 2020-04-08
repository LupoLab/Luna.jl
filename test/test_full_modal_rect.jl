import Luna
import Luna: Grid, Maths, RectModes, PhysData, Nonlinear, Ionisation, NonlinearRHS, RK45, Stats, Output, LinearOps
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

import DSP.Unwrap: unwrap


a = 50e-6
b = 10e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

#modes = (RectModes.RectMode(a, b, gas, pres, :Ag, n=1, m=1),
#         RectModes.RectMode(a, b, gas, pres, :Ag, n=2, m=1))
modes = collect(RectModes.RectMode(a, b, gas, pres, :Ag, n=n, m=m) for m in 1:3 for n in 1:6)
nmodes = length(modes)

grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

energyfun = NonlinearRHS.energy_modal()
normfun = NonlinearRHS.norm_modal(grid.ω)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

densityfun(z) = PhysData.std_dens * pres

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

in1 = (func=gausspulse, energy=5e-6)
inputs = ((1,(in1,)),)

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                              modes, :x; full=true)

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
#output = Output.HDF5Output("test_full_modal_rect.h5", 0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

import PyCall: pygui
pygui(false)
import PyPlot: plt

for i = 1:nmodes
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog[:,i,:]), rasterized=true)
    plt.clim(-6, 0)
    plt.xlim(0,2.0)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(t.*1e15, zout, transpose(It[:,i,:]), rasterized=true)
    plt.xlim(-30.0,100.0)
    plt.colorbar()
    plt.savefig("mode_$i.png")
    plt.savefig("mode_$i.pdf")
    plt.close()
end
