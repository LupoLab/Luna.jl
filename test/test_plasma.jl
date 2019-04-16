import PyPlot: plt, pygui
import Luna
import Luna: Maths, Ionisation, Nonlinear, Configuration, PhysData
import FFTW

include("test_main.jl")

ωmin = 2π*PhysData.c/8000e-9
ωmax = 2π*PhysData.c/150e-9
plwindow = Maths.planck_taper(ω, 0, ωmax, 0.05)
# twindow = Maths.planck_taper(t, -400e-15, 400e-15, 0.1)

Et = Etout[:, 121]

χ3 = PhysData.χ3_gas(medium.gas)
kerr! = Nonlinear.make_kerr!(χ3)
Pkerr = zero(Et)
kerr!(Pkerr, Et)

ionpot = PhysData.ionisation_potential(:He)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
plasma! = Nonlinear.make_plasma!(t, ω, Et, ionrate, ionpot)
plasmaFT! = Nonlinear.make_plasma_FT!(t, ω, Et, ionrate, ionpot)
outir = similar(Et);
ionrate(outir, Et);
outP = zero(Et);
plasma!(outP, Et);
outP2 = zero(Et);
plasmaFT!(outP2, Et);

println(maximum(Et))

Pf = FFTW.rfft(outP);
P2f = FFTW.rfft(outP2);
Pf .*= window;
P2f .*= window;
P2 = FFTW.irfft(Pf, length(t));
outP2 = FFTW.irfft(P2f, length(t));

frac = Maths.cumtrapz(t, outir);
frac = 1 .- exp.(-frac);
P_dt2_phase = @. frac * PhysData.e_ratio*Et
P_dt_phase = Maths.cumtrapz(t, P_dt2_phase)
P_phase = Maths.cumtrapz(t, P_dt_phase)

plt.figure()
plt.plot(t*1e15, P_phase)
plt.figure()
plt.plot(t*1e15, twindow.*FFTW.irfft(window.*prefac.*FFTW.rfft(P_phase), length(t)))
# plt.semilogy(t*1e15, abs2.(FFTW.irfft(FFTW.rfft(P_phase.*twindow).*window, length(t))))

plt.figure()
plt.semilogy(ω, FFTW.rfft(P_phase.*twindow).*window)

plt.figure()
plt.plot(t.*1e15, Et)

plt.figure()
# plt.plot(t.*1e15, outP)
plt.semilogy(t.*1e15, abs2.(Pkerr))
plt.semilogy(t.*1e15, abs2.(Pkerr.+outP))
plt.title("polarisation")

plt.figure()
plt.semilogy(t.*1e15, outir)
plt.title("ionisation rate")

plt.figure()
plt.semilogy(t.*1e15, frac)
plt.title("ionisation fraction")

plt.figure()
plt.semilogy(ω,abs2.(Pf))
plt.semilogy(ω,abs2.(P2f))

plt.figure()
plt.plot(ω, window)
