using Luna

a = 50e-6
b = 10e-6
gas = :Ar
pres = 5
L = 15e-2/10

τfwhm = 30e-15
λ0 = 800e-9
energy = 5e-6

grid = Grid.RealGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

modes = collect(RectModes.RectMode(a, b, gas, pres, :Ag, n=n, m=m) for m in 1:3 for n in 1:6)
nmodes = length(modes)

energyfun, energyfunω = Fields.energyfuncs(grid)
normfun = NonlinearRHS.norm_modal(grid.ω)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
             #Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                              modes, :x; full=true)

statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
#output = Output.HDF5Output("test_full_modal_rect.h5", 0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW

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
