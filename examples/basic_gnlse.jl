using Luna

γ = 1.0
β2 = -1.0
β3 = 0.5

T0 = 5.0

N = 3

P0 = abs(β2)*N^2/(γ*T0^2)

Ld = T0^2/abs(β2)
Lnl = 1/(γ*P0)
Lf = sqrt(Ld*Lnl)

L = 2*Lf
# L = π*Ld
##
grid = Grid.GNLSEGrid(L, 40, 80)

linop = LinearOps.make_const_linop(grid, [β2, β3])

inputs = Fields.SechField(λ0=1.0, τw=T0, power=P0)

Eω, transform, FT = Luna.setup(grid, inputs; γ=γ, shock=0.0)

output = Output.MemoryOutput(0, grid.zmax, 201)
##
Luna.run(Eω, grid, linop, transform, FT, output)

##
import PyPlot: pygui, plt
import FFTW
pygui(true)
plt.close("all")

Eω = FFTW.fftshift(output["Eω"], 1)
Et = FFTW.ifft(output["Eω"], 1)

ω = FFTW.fftshift(grid.ω)

plt.figure()
plt.pcolormesh(grid.t, output["z"], abs2.(Et)')
plt.xlabel("Time")
plt.ylabel("Distance")

plt.figure()
plt.pcolormesh(ω, output["z"], log10.(Maths.normbymax(abs2.(Eω)')))
plt.clim(-4, 0)

plt.figure()
plt.semilogy(ω, Maths.normbymax(abs2.(Eω[:, end])))
plt.ylim(1e-5, 5)
plt.axvline(-3β2/β3; linestyle="--", color="k")
