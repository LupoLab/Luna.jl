# Quick example showing how to include TPA in a GNLSE simulation.
# Comparing with the same simulation without TPA to see the spectral suppression at shorter wavelengths.

using Luna, PyPlot

λ0 = 240e-9
τfwhm = 5e-15  # short pulse = broad bandwidth
energy = 100e-9
flength = 50e-6
β2 = 1e-30
βs = [0.0, 0.0, β2]
Aeff = pi*12e-6^2
λlims = [160e-9, 400e-9]
trange = 1e-12

# Using material-based convention: n2, γ, and TPA auto-computed from :SiO2
output_tpa = prop_gnlse(flength, βs; material=:SiO2, Aeff, λ0, τfwhm, energy,
                        pulseshape=:gauss, λlims, trange,
                        raman=false, shock=false, shotnoise=false, tpa=true)

output_notpa = prop_gnlse(flength, βs; material=:SiO2, Aeff, λ0, τfwhm, energy,
                          pulseshape=:gauss, λlims, trange,
                          raman=false, shock=false, shotnoise=false)

λ, Iλtpa = Processing.getIω(output_tpa, :λ, flength)
λ, Iλnotpa = Processing.getIω(output_notpa, :λ, flength)

using PyPlot
figure()
plot(λ*1e9, Iλtpa, label="With TPA")
plot(λ*1e9, Iλnotpa, label="Without TPA")
xlabel("Wavelength (nm)")
ylabel("Spectral Intensity (a.u.)")
legend()
title("Spectral suppression from TPA in SiO₂")
xlim(180, 400)
