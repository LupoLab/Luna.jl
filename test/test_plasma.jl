import PyPlot: plt, pygui
import Luna
import Luna: Maths, Ionisation, Nonlinear, Configuration, PhysData
import FFTW

# t = 1e-15 .* collect(range(-200, stop=200, length=2*8192));
# println(1e18*(t[2]-t[1]))
# δω = 2π/(maximum(t)-minimum(t))
# Nω = collect(range(0, length=Int(length(t)/2+1)))
# ω = Nω.*δω
# Et = 4e10*Maths.gauss(t, fwhm=10e-15) .* cos.(2.4e15 .* t);

input = Configuration.GaussInput(duration=7e-15, energy=30e-6, wavelength=800e-9)
grid = Configuration.RealGrid(trange=2e-12, λ_lims=(200e-9, 4000e-9))
geometry = Configuration.Capillary(radius=75e-6, length=50e-2)
medium = fill=Configuration.StaticFill(:HeJ, 6)
nonlinear = Configuration.GasNonlinear()

cfg = Configuration.Config(grid, geometry, medium, nonlinear, input)

ω, t, zout, Eout, Etout, window, twindow = Luna.run(cfg)

ωmin = 2π*PhysData.c/4000e-9
ωmax = 2π*PhysData.c/200e-9
plwindow = Maths.planck_taper(ω, ωmin, ωmax, 0.1)
twindow = Maths.planck_taper(t, -400e-15, 400e-15, 0.1)

Et = Etout[:, end]
Ilog = log10.(Maths.normbymax(abs2.(Eout)))

ionpot = PhysData.ionisation_potential(:He)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
plasma! = Nonlinear.make_plasma!(t, ω, Et, ionrate, ionpot)
outir = similar(Et)
ionrate(outir, Et)
outP = similar(Et)
fill!(outP, 0)
plasma!(outP, Et)

println(maximum(Et))

Pf = FFTW.rfft(outP);
Pf .*= window;
P2 = FFTW.irfft(Pf, length(t));

frac = Maths.cumtrapz(t, outir);
frac = 1 .- exp.(-frac);

plt.figure()
plt.plot(t.*1e15, Et)

plt.figure()
plt.plot(t.*1e15, outP)
plt.plot(t.*1e15, P2)
plt.title("polarisation")

plt.figure()
plt.semilogy(t.*1e15, outir)
plt.title("ionisation rate")

plt.figure()
plt.semilogy(t.*1e15, frac)
plt.title("ionisation fraction")

plt.figure()
plt.semilogy(ω,abs2.(Pf))

plt.figure()
plt.plot(ω, window)

plt.figure()
plt.pcolormesh(ω./2π.*1e-15, zout, transpose(Ilog))
plt.clim(-4, 0)
plt.colorbar()