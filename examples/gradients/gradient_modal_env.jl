using Luna

a = 13e-6
gas = :Ar
pres = 7.5
L = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

coren, densityfun = Capillary.gradient(gas, L, pres, 0);

modes = (
    Capillary.MarcatilliMode(a, coren, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, coren, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)
nmodes = length(modes)

grid = Grid.EnvGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

energyfun = Fields.energyfuncs(grid)[1]

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                               modes, :y; full=false)

statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output)

import FFTW
import PyPlot:pygui, plt

ω = FFTW.fftshift(grid.ω)
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

Etout = FFTW.ifft(Eout, 1)
It = abs2.(Etout)

Ilog = FFTW.fftshift(log10.(Maths.normbymax(abs2.(Eout))), 1)

pygui(true)

for i = 1:nmodes
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
