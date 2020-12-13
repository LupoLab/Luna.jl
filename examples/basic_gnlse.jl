using Luna

function capillary_gnlse(a, gas, pressure, λ0, τfw, energy)
    mode = Capillary.MarcatilliMode(a, gas, pressure)
    ω0 = PhysData.wlfreq(λ0)
    N0, n0, n2 = Tools.getN0n0n2(ω0, gas; P=pressure)
    γ = Tools.getγ(ω0, mode, n2)
    β2 = Modes.dispersion(mode, 2, ω0)
    β3 = Modes.dispersion(mode, 3, ω0)
    T0 = Tools.τfw_to_τ0(τfw, :sech)
    P0 = Tools.E_to_P0(energy, τfw)
    normalise(β2, β3, γ, λ0, P0, T0)
end

function normalise(β2, β3, γ, λ0, P0, T0)
    β3n = -β3/(β2*T0)
    N = sqrt(γ*P0*T0^2/abs(β2))
    ω0 = PhysData.wlfreq(λ0)
    shock = 1/(T0*ω0)
    N, β3n, shock
end

γ = 1.0
β2 = -1.0
T0 = 1.0

N = 3.5
β3 = 0.1
shock = 0.05

# N, β3, shock = capillary_gnlse(125e-6, :He, 1, 800e-9, 10e-15, 100e-6)

L = 2/N # 2x fission length
##
grid = Grid.GNLSEGrid(L, 200, 20)

linop = LinearOps.make_const_linop(grid, [β2, β3])

inputs = Fields.SechField(λ0=1.0, τw=T0, power=N^2)

Eω, transform, FT = Luna.setup(grid, inputs; γ=γ, shock=shock)

output = Output.MemoryOutput(0, grid.zmax, 201)
##
Luna.run(Eω, grid, linop, transform, FT, output)

##
import PyPlot: pygui, plt
import FFTW
pygui(true)
# plt.close("all")

Eω = FFTW.fftshift(output["Eω"], 1)
Et = FFTW.ifft(output["Eω"], 1)

ω = FFTW.fftshift(grid.ω)

plt.figure()
plt.pcolormesh(grid.t, output["z"], abs2.(Et)')
plt.xlabel("Time")
plt.ylabel("Distance")
plt.xlim(-5, 5)

plt.figure()
plt.pcolormesh(ω, output["z"], log10.(Maths.normbymax(abs2.(Eω)')))
plt.clim(-4, 0)

plt.figure()
plt.semilogy(ω, Maths.normbymax(abs2.(Eω[:, end])))
plt.ylim(1e-5, 5)
plt.axvline(-3β2/β3; linestyle="--", color="k")
