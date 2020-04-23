using Luna

a = 13e-6
gas = :Ar
pres = 5

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

L = 15e-2

coren, densityfun = Capillary.gradient(gas, L, pres, pres);

modes = (
    Capillary.MarcatilliMode(a, coren, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, coren, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)

grid = Grid.RealGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

energyfun, energyfunω = Fields.energyfuncs(grid)
normfun = NonlinearRHS.norm_modal(grid.ω)

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                              modes, :y; full=false)

statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot:pygui, plt

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
