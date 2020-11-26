using Luna

γ = 1.0
β2 = -1.0
β3 = -0.5

T0 = 5.0

N = 5

P0 = abs(β2)*N^2/(γ*T0^2)

Ld = T0^2/abs(β2)
Lnl = 1/(γ*P0)
Lf = sqrt(Ld*Lnl)

L = 2*Lf
##
grid = Grid.GNLSEGrid(L, 20, 20)

linop = LinearOps.make_const_linop(grid, [β2, β3])
responses = (Nonlinear.Kerr_gnlse(γ), )

inputs = Fields.SechField(λ0=1.0, τw=T0, power=P0)

Eω, transform, FT = Luna.setup(grid, responses, inputs)

output = Output.MemoryOutput(0, grid.zmax, 201)
##
Luna.run(Eω, grid, linop, transform, FT, output)

##
import PyPlot: pygui, plt
pygui(true)

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
