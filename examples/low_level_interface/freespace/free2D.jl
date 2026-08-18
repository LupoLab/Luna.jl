using Luna
Luna.set_fftw_mode(:estimate)
import FFTW
import PyPlot:pygui, plt
pygui(true)

gas = :Ar
pres = 4

τ = 30e-15
λ0 = 800e-9

w0 = 2e-3
energy = 1.5e-3
L = 2

R = 6e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
xgrid = Grid.Free2DGrid(R, N)

x = xgrid.x
energyfun, energyfunω = Fields.energyfuncs(grid, xgrid)

densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

linop = LinearOps.make_const_linop(grid, xgrid, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.const_norm_free2D(grid, xgrid, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

Eω, transform, FT = Luna.setup(grid, xgrid, densityfun, normfun, responses, inputs)

# statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
output = Output.MemoryOutput(0, grid.zmax, 21)

Luna.run(Eω, grid, linop, transform, FT, output, max_dz=Inf, init_dz=1e-1)

ω = grid.ω
t = grid.t

zout = output.data["z"]
Eout = output.data["Eω"]

println("Transforming...")
Eωx = FFTW.ifft(Eout, 3)
Etx = FFTW.irfft(Eout, length(grid.t), (1, 3))
println("...done")


Ilog = log10.(Maths.normbymax(abs2.(Eωx)))

Iωx = abs2.(Eωx);

Ix = zeros(Float64, (length(x), length(zout)));
energy = zeros(length(zout));
for ii in axes(Etx, 4)
    energy[ii] = energyfun(Etx[:, 1, :, ii]);
    Ix[:, ii] = (grid.ω[2]-grid.ω[1]) .* sum(Iωx[:, 1, :, ii], dims=1);
end

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

E0ωyx = FFTW.ifft(Eω[ω0idx, 1, :, :], (1, 2))

Iωxlog = log10.(Maths.normbymax(Iωx))

plt.figure()
plt.pcolormesh(zout, ω.*1e-15/2π, Iωxlog[:, 1, N÷2+1, :])
plt.xlabel("Z (m)")
plt.ylabel("f (PHz)")
plt.title("I(ω, x=0, z)")
plt.clim(-6, 0)
plt.colorbar()


plt.figure()
plt.plot(zout.*1e2, energy.*1e6)
plt.xlabel("Distance [cm]")
plt.ylabel("Energy [μJ]")
