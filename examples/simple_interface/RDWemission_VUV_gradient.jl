using Luna, PythonPlot

radius = 125e-6 # HCF core radius
flength = 1.5 # HCF length

gas = :He
pressure = 0.8 # gas pressure at the start of the HCF in bar

λ0 = 800e-9 # central wavelength of the pump pulse
τfwhm = 7.5e-15 # FWHM duration of the pump pulse
energy = 275e-6 # energy in the pump pulse

# giving the pressure argument as (pressure, 0) defines a decreasing gradient to vacuum
vuv = prop_capillary(radius, flength, gas, (pressure, 0); λ0, τfwhm, energy,
                     modes=4, trange=400e-15, λlims=(90e-9, 4e-6))

Plotting.prop_2D(vuv, :λ; modes=:sum, trange=(-20e-15, 20e-15), λrange=(100e-9, 1000e-9),
                 dBmin=-30)
# plot the total time-dependent power at the end of the propagation:
Plotting.time_1D(vuv; modes=:sum)
# plot the VUV dispersive wave at the exit:
Plotting.time_1D(vuv; modes=:sum, bandpass=(100e-9, 140e-9))
# show the evolution of the VUV pulse during the propagation:
Plotting.prop_2D(vuv; λrange=(100e-9, 200e-9), trange=(0, 20e-15), modes=:sum, bandpass=(100e-9, 140e-9))