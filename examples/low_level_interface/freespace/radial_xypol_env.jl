using Luna
using HiSol
import FFTW
import Luna: Hankel
import PyPlot: plt

λ0 = 800e-9
τfwhm = 10e-15
w0 = 50e-6
energy = 1e-6

thickness = 20e-6
material = :SiO2

R = 4*w0
N = 2^8

grid = Grid.EnvGrid(thickness, λ0, (200e-9, 4e-6), 200e-15)
q = Hankel.QDHT(R, N, dim=3)

χ3 = PhysData.χ3(material)
responses = (Nonlinear.Kerr_env(χ3),)

nfun = PhysData.ref_index_fun(material)
function nfunreal(λ)
    nr = real(nfun(λ))
    nr, nr
end
linop = LinearOps.make_const_linop(grid, q, nfunreal)

normfun = NonlinearRHS.const_norm_radial(grid, q, nfunreal)
densityfun = z -> 1

inputs = (Fields.GaussGaussField(;λ0, τfwhm, energy, w0),
          Fields.GaussGaussField(;λ0=1200e-9, τfwhm=50e-15, energy=energy*10, w0, θ=π/4))
# inputs = Fields.GaussGaussField(;λ0, τfwhm, energy, w0)
Eω, transform, FT = Luna.setup(grid, q, densityfun, normfun, responses, inputs)

output = Output.MemoryOutput(0, grid.zmax, 101)
Luna.run(Eω, grid, linop, transform, FT, output; init_dz=1e-6)

##
z = output["z"]
Eωk = output["Eω"] # (ω, pol, k, z)

Eωr = q \ Eωk # (ω, pol, r, z)
Etr = FFTW.ifft(Eωr, 1) # (t, pol, r, z)
Iωr = abs2.(Eωr) # (ω, pol, r, z)
Itr = abs2.(Etr) # (t, pol, r, z)

Irxy = dropdims(sum(Iωr; dims=1); dims=1) # (pol, r, z)
Iωxy = FFTW.fftshift(dropdims(sum(Iωr; dims=3); dims=3), 1) # (ω, pol, z)
Ir = dropdims(sum(Iωr; dims=(1, 2)); dims=(1, 2)) # (r, z)

Itxy = dropdims(sum(Itr; dims=3); dims=3) # (t, pol, z)

ω = FFTW.fftshift(grid.ω)

##
plt.figure()
plt.pcolormesh(z*1e3, q.r*1e6, Ir)
plt.xlabel("Distance (mm)")
plt.ylabel("radius (μm)")

##
plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(z*1e3, q.r*1e6, Irxy[1, :, :])
plt.subplot(1, 2, 2)
plt.pcolormesh(z*1e3, q.r*1e6, Irxy[2, :, :])
plt.xlabel("Distance (mm)")
plt.ylabel("radius (μm)")
plt.suptitle("Frequency domain")
##
plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(z*1e3, grid.t*1e15, Itxy[:, 1, :])
plt.title("X Pol")
plt.xlabel("Distance (mm)")
plt.ylabel("Time (fs)")
plt.subplot(1, 2, 2)
plt.pcolormesh(z*1e3, grid.t*1e15, Itxy[:, 2, :])
plt.title("Y Pol")
plt.xlabel("Distance (mm)")
plt.suptitle("Time domain")

##
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(grid.t*1e15, Itxy[:, 1, 1]; label="input X")
plt.plot(grid.t*1e15, Itxy[:, 2, 1]; label="input Y")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(grid.t*1e15, Itxy[:, 1, end]; label="output X")
plt.plot(grid.t*1e15, Itxy[:, 2, end]; label="output Y")
plt.xlabel("Time (fs)")
plt.legend()

##
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(ω*1e-15, Iωxy[:, 1, 1]; label="input X")
plt.semilogy(ω*1e-15, Iωxy[:, 2, 1]; label="input Y")
plt.legend()
plt.subplot(2, 1, 2)
plt.semilogy(ω*1e-15, Iωxy[:, 1, end]; label="output X")
plt.semilogy(ω*1e-15, Iωxy[:, 2, end]; label="output Y")
plt.xlabel("Frequency (rad/fs)")
plt.legend()