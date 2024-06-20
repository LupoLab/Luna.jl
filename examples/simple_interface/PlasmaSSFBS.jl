using Luna

radius = 9e-6 # HCF core radius
flength = 25e-2 # HCF length

gas = :Ar
pressure = 5.0 # gas pressure in bar

λ0 = 1500e-9 # central wavelength of the pump pulse
τfwhm = 30e-15 # FWHM duration of the pump pulse
energy = 1.7e-6 # energy in the pump pulse

pssfbs = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                        trange=1.6e-12, λlims=(300e-9, 1.8e-6), loss=false, envelope=true)

Plotting.prop_2D(pssfbs, :λ; modes=:sum, trange=(-700e-15, 100e-15), λrange=(300e-9, 1800e-9),
                 dBmin=-30)
Plotting.time_1D(pssfbs, range(0,flength,length=5), modes=:sum, trange=(-700e-15, 100e-15))
Plotting.spec_1D(pssfbs, range(0,flength,length=5), modes=:sum, λrange=(300e-9, 1800e-9))
