import Test: @test, @testset, @test_throws
import Luna: Antiresonant, Capillary, Modes
import Luna.PhysData: wlfreq

@testset "Antiresonant PCF" begin
    a = 20e-6
    m = Capillary.MarcatiliMode(a, :Air, 0, (ω; z) -> 1.45)
    w = 0.7e-6
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w)
    arm2 = Antiresonant.ZeisbergerMode(m; wallthickness=w)

    @test Modes.N(m) == Modes.N(arm) == Modes.N(arm2)
    @test Modes.Aeff(m) == Modes.Aeff(arm) == Modes.Aeff(arm2)
    r = a*rand(50)
    θ = 2π*rand(50)
    @test all(Modes.field.(m, (r, θ)) .== Modes.field.(arm, (r, θ)))
    @test all(Modes.field.(m, (r, θ)) .== Modes.field.(arm2, (r, θ)))
    λ = collect(range(390e-9, stop=1450e-9, length=2^11))
    ω = wlfreq.(λ)
    @test all(Modes.neff.(arm, ω) .== Modes.neff.(arm2, ω))

    @test Modes.neff(arm, 2.5e15) ≈ 0.9999193518567425 + 1.87925966056515e-6im
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=false)
    @test Modes.neff(arm, 2.5e15) == 0.9999193518567425
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=0.5)
    @test Modes.neff(arm, 2.5e15) ≈ 0.9999193518567425 + 0.5*1.87925966056515e-6im
    @test_throws ArgumentError Antiresonant.ZeisbergerMode(a, :Air, 0; wallthickness=w, loss=0.5im)
end

##
#= References
[1] L. Vincetti
Empirical formulas for calculating loss in hollow core tube lattice fibers, 
Opt. Express, OE, vol. 24, no. 10, pp. 10313-10325, May 2016, doi: 10.1364/OE.24.010313.

[2] L. Vincetti and L. Rosa
A simple analytical model for confinement loss estimation in hollow-core Tube Lattice Fibers
Opt. Express, OE, vol. 27, no. 4, pp. 5230-5237, Feb. 2019, doi: 10.1364/OE.27.005230.
=#
# F#1 from [2]
import PyPlot: plt
import DelimitedFiles: readdlm
t = 1e-6
r_ext = 10e-6
δ = 5e-6 # tube spacing
n = 1.44
N = 8 # number of tubes

# eq. (1) of [1]
k = 1 + δ/2r_ext
Rco = r_ext * (k/sin(π/N) - 1)

δcalc = 2*(sin(π/N)*(Rco + r_ext)-r_ext)

m = Antiresonant.VincettiMode(Rco; wallthickness=t, tube_radius=r_ext, Ntubes=N, cladn=n)
zm = Antiresonant.ZeisbergerMode(Rco, :Air, 0, (ω; z) -> 1.45; wallthickness=t)

F = collect(range(0.4, 4.2, 2^14))
λ = @. 2t/F*sqrt(n^2-1)

paperdata = readdlm(joinpath(
    homedir(),
    "Documents",
    "WebPlotDigitizer",
    "Vincetti PCF loss",
    "VincettiLoss_F#1.csv"),
    ',')

##
plt.figure()
plt.semilogy(F, Modes.dB_per_m.(m, PhysData.wlfreq.(λ)); label="Luna (Vincetti)")
plt.semilogy(F, Modes.dB_per_m.(zm, PhysData.wlfreq.(λ)); label="Luna (Zeisberger)")
plt.semilogy(paperdata[:, 1], paperdata[:, 2], "."; label="Paper")
plt.xlim(extrema(F))
plt.xlabel("Normalised frequency")
plt.ylabel("Loss (dB/m)")
plt.legend()

λpaper = @. 2t/paperdata[:, 1]*sqrt(n^2-1)
plt.figure()
plt.semilogy(λ*1e9, Modes.dB_per_m.(m, PhysData.wlfreq.(λ)); label="Luna")
plt.semilogy(λ*1e9, Modes.dB_per_m.(zm, PhysData.wlfreq.(λ)); label="Luna (Zeisberger)")
plt.semilogy(λpaper*1e9, paperdata[:, 2], "."; label="Paper")
plt.xlim(extrema(λ).*1e9)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Loss (dB/m)")
plt.legend()

##
neffdata = readdlm(joinpath(
    homedir(),
    "Documents",
    "WebPlotDigitizer",
    "Vincetti PCF loss",
    "neff_F#1.csv"),
    ',')

plt.figure()
plt.plot(F, Antiresonant.neff_real.(m, PhysData.wlfreq.(λ)); label="Luna (Vincetti)")
plt.plot(F, real(Antiresonant.neff.(zm, PhysData.wlfreq.(λ))); label="Luna (Zeisberger)")
plt.plot(F, real(Antiresonant.neff.(m.m, PhysData.wlfreq.(λ))); label="Luna (Marcatili)")
plt.plot(neffdata[:, 1], neffdata[:, 2], "--"; label="Paper")
plt.ylim(0.999, 1)
plt.xlim(extrema(F))
plt.xlabel("Normalised frequency")
plt.ylabel("\$n_\\mathrm{eff}\$")
plt.legend()
plt.tight_layout()
