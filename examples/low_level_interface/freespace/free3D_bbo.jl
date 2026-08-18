#=
Here we simulate type I SHG in a BBO crystal driven at 800 nm. The focus is small and the crystal
is reasonably thick, so we observe strong temporal and spatial walk-off effects.
=#
using Luna
import FFTW
import Luna: Hankel
import PyPlot: plt
import NumericalIntegration: integrate

λ0 = 800e-9 # driving wavelength
τfwhm = 30e-15 # pulse duration
w0 = 20e-6 # 1/e² beam radius
energy = 10e-9 # pulse energy

material = :BBO
thickness = 200e-6 # BBO thickness: 200 μm

R = 4*w0 # radius of the spatial window
Nx = 2^6 # number of spatial points in x
Ny = 16 # number of spatial points in y

grid = Grid.RealGrid(thickness, λ0, (250e-9, 2e-6), 120e-15)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)

θ = deg2rad(29.2) # type I phase-matching angle
ϕ = deg2rad(30) # ϕ for type I phase-matching

##
responses = (Nonlinear.Chi2Field(θ, ϕ, PhysData.χ2(material)),)
#              Nonlinear.Kerr_field(PhysData.χ3(material)))

#=
Ref index functions. Note that here nfunx takes (λ, δθ=0) as arguments,
which allows make_const_linop to calculate the actual internal angle depending
on frequency and transverse k-vector component.
=#
nfunx, nfuny = PhysData.ref_index_fun_xy(material, θ)
linop = LinearOps.make_const_linop(grid, xygrid, (nfunx, nfuny))

normfun = NonlinearRHS.const_norm_free(grid, xygrid, (nfunx, nfuny))
densityfun = z -> 1 # density is unity because we're considering a solid.
##
inputs = Fields.GaussGaussField(;λ0, τfwhm, energy=energy, w0)
Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)

##
output = Output.MemoryOutput(0, grid.zmax, 101)
Luna.run(Eω, grid, linop, transform, FT, output; init_dz=1e-6)

##
z = output["z"]
Eωk = output["Eω"] # (ω, pol, kx, ky, z)
Ik = FFTW.fftshift(dropdims(sum(abs2.(Eωk); dims=1); dims=1), (2, 3))
x = xygrid.x
y = xygrid.y

# normalisation prefactor for spectral intensity
ωprefac = PhysData.c*PhysData.ε_0/2 * 2π/(grid.ω[end]^2)

Eωr = FFTW.ifft(Eωk, (3, 4)) # (ω, pol, x, y, z)
Etr = FFTW.irfft(Eωr, 2*(length(grid.ω)-1), 1) # (t, pol, x, y, z)
EtrH = Maths.hilbert(Etr)
Iωr = abs2.(Eωr) # (ω, pol, x, y, z)
Itr = 0.5*PhysData.c*PhysData.ε_0*abs2.(EtrH) # (t, pol, x, y, z)

Irxy = dropdims(sum(Iωr; dims=1); dims=1) # (pol, x, y, z)
Ix = dropdims(sum(Irxy; dims=3); dims=3) # (pol, x, z)
Iω = dropdims(
    Maths.integrateNd(
        y,
        Maths.integrateNd(x, Iωr; dim=3);
        dim=4),
    dims=(3, 4)
)*ωprefac # (ω, pol, z)
Ir = dropdims(sum(Iωr; dims=(1, 2)); dims=(1, 2)) # (x, y, z)

Itxy = dropdims(
    Maths.integrateNd(
        y,
        Maths.integrateNd(x, Itr; dim=3);
        dim=4),
    dims=(3, 4)
) # (t, pol, z)

Eω0 = Eωr[:, :, length(x)÷2+1, length(y)÷2+1, :]
Et0 = FFTW.irfft(Eω0, 2*(length(grid.ω)-1), 1) # (t, pol, z)
Et0 = Maths.hilbert(Et0)
It0 = 0.5*PhysData.c*PhysData.ε_0*abs2.(Et0)

et, eω = Fields.energyfuncs(grid, xygrid)

energy_out = dropdims(mapslices(eω, Eωk; dims=(1, 3, 4)); dims=(1, 3, 4))

ω = grid.ω

##
kx = FFTW.fftshift(xygrid.kx)
ky = FFTW.fftshift(xygrid.ky)
fig = plt.figure()
fig.set_size_inches(12, 12)
plt.subplot(2, 2, 1)
plt.pcolormesh(kx*1e-6, ky*1e-6, Maths.log10_norm(Ik[1, :, :, 1])')
plt.clim(-6, 0)
plt.title("Input, X pol")
plt.subplot(2, 2, 2)
plt.pcolormesh(kx*1e-6, ky*1e-6, Maths.log10_norm(Ik[2, :, :, 1])')
plt.clim(-6, 0)
plt.title("Input, Y pol")
plt.subplot(2, 2, 3)
plt.pcolormesh(kx*1e-6, ky*1e-6, Maths.log10_norm(Ik[1, :, :, end])')
plt.clim(-6, 0)
plt.title("Output, X pol")
plt.subplot(2, 2, 4)
plt.pcolormesh(kx*1e-6, ky*1e-6, Maths.log10_norm(Ik[2, :, :, end])')
plt.clim(-6, 0)
plt.title("Output, Y pol")

##
plt.figure()
plt.pcolormesh(z*1e3, x*1e6, Ir[:, length(y)÷2+1, :])
plt.xlabel("Distance (mm)")
plt.ylabel("X (μm)")

##
plt.figure()
plt.plot(z*1e6, 1e9energy_out[1, :]; label="X polarisation")
plt.plot(z*1e6, 1e9energy_out[2, :]; label="Y polarisation")
plt.xlabel("Distance (μm)")
plt.ylabel("Energy (nJ)")
plt.legend()

##
fig = plt.figure()
fig.set_size_inches(12, 6)
plt.subplot(1, 2, 1)
plt.pcolormesh(z*1e3, x*1e6, Ix[1, :, :])
plt.xlabel("Distance (mm)")
plt.ylabel("X (μm)")
plt.title("X Polarisation")
plt.subplot(1, 2, 2)
plt.pcolormesh(z*1e3, x*1e6, Ix[2, :, :])
plt.xlabel("Distance (mm)")
plt.ylabel("X (μm)")
plt.title("Y Polarisation")
fig.tight_layout()


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
mm = 2π*maximum(Iω)*1e12
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(ω*1e-12/2π, 2π*1e12*Iω[:, 1, 1]; label="input X")
plt.semilogy(ω*1e-12/2π, 2π*1e12*Iω[:, 2, 1]; label="input Y")
plt.ylim(1e-5mm, 5mm)
plt.xlim(200, 900)
plt.legend()
plt.subplot(2, 1, 2)
plt.semilogy(ω*1e-12/2π, 2π*1e12*Iω[:, 1, end]; label="output X")
plt.semilogy(ω*1e-12/2π, 2π*1e12*Iω[:, 2, end]; label="output Y")
# plt.semilogy(ω*1e-12/2π, 2π*1e12*Iω[:, 2, 1]; c="C1", linestyle="--", label="input Y")
plt.ylim(1e-5mm, 5mm)
plt.xlim(200, 900)
plt.xlabel("Frequency (THz)")
plt.ylabel("SED (J/Hz)")
plt.legend()

##
fig = plt.figure()
fig.set_size_inches(12, 7)
plt.subplot(1, 2, 1)
plt.pcolormesh(grid.t*1e15, xygrid.x*1e6, (Etr[:, 1, :, Ny÷2+1, end])'; cmap="seismic")
plt.clim([-1, 1].*maximum(abs.(Etr[:, 1, Ny÷2+1, :, end])))
plt.xlim(-50, 50)
plt.ylim(-100, 100)
plt.xlabel("Time (fs)")
plt.ylabel("X (μm)")
plt.title("X polarisation")

plt.subplot(1, 2, 2)
plt.pcolormesh(grid.t*1e15, xygrid.x*1e6, (Etr[:, 2, :, Ny÷2+1, end])'; cmap="seismic")
plt.clim([-1, 1].*maximum(abs.(Etr[:, 2, :, Ny÷2+1, end])))
plt.xlim(-50, 50)
plt.ylim(-100, 100)
plt.xlabel("Time (fs)")
plt.title("Y polarisation")

##
fig = plt.figure()
fig.set_size_inches(12, 12)
plt.subplot(2, 2, 1)
plt.pcolormesh(xygrid.x*1e6, xygrid.y*1e6, Irxy[1, :, :, 1]')
plt.title("Input, X pol")
plt.subplot(2, 2, 2)
plt.pcolormesh(xygrid.x*1e6, xygrid.y*1e6, Irxy[2, :, :, 1]')
plt.title("Input, Y pol")
plt.subplot(2, 2, 3)
plt.pcolormesh(xygrid.x*1e6, xygrid.y*1e6, Irxy[1, :, :, end]')
plt.title("Output, X pol")
plt.subplot(2, 2, 4)
plt.pcolormesh(xygrid.x*1e6, xygrid.y*1e6, Irxy[2, :, :, end]')
plt.title("Output, Y pol")

##
fig = plt.figure()
fig.set_size_inches(12, 6)
plt.subplot(1, 2, 1)
Etx_00 = Etr[:, 1, length(xygrid.x)÷2+1, length(xygrid.y)÷2+1, end]
plt.plot(grid.t*1e15, 1e-9*(Etx_00);
        label="Max: $(maximum(1e-9*Etx_00))")
plt.ylabel("E (GV/m)")
plt.legend()
plt.xlabel("Time (fs)")
plt.title("X Polarisation")
plt.subplot(1, 2, 2)
Ety_00 = Etr[:, 2, length(xygrid.x)÷2+1, length(xygrid.y)÷2+1, end]
plt.plot(grid.t*1e15, 1e-9*(Ety_00);
          label="Max: $(maximum(Ety_00))")
plt.legend()
plt.xlabel("Time (fs)")
plt.title("Y Polarisation")
