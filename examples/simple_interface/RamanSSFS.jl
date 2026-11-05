using Luna, PythonPlot

radius = 13e-6 # HCF core radius
flength = 2 # HCF length

gas = :H2
pressure = 5.0 # gas pressure in bar

λ0 = 800e-9 # central wavelength of the pump pulse
τfwhm = 20e-15 # FWHM duration of the pump pulse
energy = 1e-6 # energy in the pump pulse

rssfs = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                     trange=40e-12, λlims=(400e-9, 2e-6),
                     raman=true, envelope=true, loss=false, plasma=false)

Plotting.prop_2D(rssfs, :λ; modes=:sum, trange=(-0.5e-12, 2e-12), λrange=(500e-9, 1400e-9),
                 dBmin=-30)
Plotting.time_1D(rssfs, range(0,flength,length=5), modes=:sum, trange=(-0.5e-12, 2e-12))
Plotting.spec_1D(rssfs, range(0,flength,length=5), modes=:sum, λrange=(500e-9, 1400e-9))
