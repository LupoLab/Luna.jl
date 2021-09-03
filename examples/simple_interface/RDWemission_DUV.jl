using Luna

radius = 125e-6 # HCF core radius
flength = 3 # HCF length

gas = :Ar
pressure = 80e-3 # gas pressure in bar

λ0 = 800e-9 # central wavelength of the pump pulse
τfwhm = 10e-15 # FWHM duration of the pump pulse
energy = 60e-6 # energy in the pump pulse

duv = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                     modes=4, trange=400e-15, λlims=(150e-9, 4e-6))

Plotting.prop_2D(duv, :λ; modes=:sum, trange=(-20e-15, 20e-15), λrange=(150e-9, 1000e-9),
                 dBmin=-30)
# plot the total time-dependent power at the end of the propagation:
Plotting.time_1D(duv; modes=:sum)
# plot the UV dispersive wave in the fundamental mode only
Plotting.time_1D(duv; modes=1, bandpass=(220e-9, 270e-9))
# plot the spectrogram of the pulse at the exit with a white background
Plotting.spectrogram(duv, flength; trange=(-20e-15, 30e-15), λrange=(150e-9, 1000e-9),
                     N=256, fw=3e-15, cmap=Plotting.cmap_white("viridis"))