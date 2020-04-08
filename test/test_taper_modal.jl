import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
import Logging
import FFTW
import NumericalIntegration: integrate, SimpsonEven
Logging.disable_logging(Logging.BelowMinLevel)

import PyPlot:pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

L = 15e-2

grid = Grid.RealGrid(L, 800e-9, (160e-9, 3000e-9), 1e-12)

a0 = a
aL = 3a/4

afun = let a0=a0, aL=aL, L=L
    afun(z) = a0 + (aL-a0)*z/L
end
# coren, dens = Capillary.gradient(gas, L, pres, pres); # dens is unused
modes = (
    Capillary.MarcatilliMode(afun, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(afun, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            #  Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop = LinearOps.make_linop(grid, modes, λ0);

energyfun = NonlinearRHS.energy_modal()
normfun = NonlinearRHS.norm_modal(grid.ω)

in1 = (func=gausspulse, energy=1e-6)
inputs = ((1,(in1,)),)

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                               modes, :y, full=false)

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.irfft(Eout, length(grid.t), 1)
It = abs2.(Maths.hilbert(Etout))

Ilog = log10.(Maths.normbymax(abs2.(Eout)))

pygui(true)

for i = 1:length(modes)
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog[:,i,:]))
    plt.clim(-6, 0)
    plt.xlim(0,2.0)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(t.*1e15, zout, transpose(It[:,i,:]))
    plt.xlim(-30.0,100.0)
    plt.colorbar()
end