using Luna, PythonPlot

#= In this example we simulate degenerate four-wave mixing between a circularly polarised
pump pulse at 400 nm and a seed pulse at 800 nm to generate a circularly polarised idler
at 266 nm. By switching the helicity of the seed relative to the pump, we can suppress
the idler generation.
The parameters are chosen to correspond to those in:
A. Lekosiotis, et al.,
Generation of broadband circularly polarized deep-ultraviolet pulses in hollow capillary fibers,
Opt. Lett., 45, 5648, (2020),
DOI: 10.1364/OL.400362.
=#

λp = 400e-9 # central wavelength of pump pulse
ep = 100e-6 # energy in pump pulse

λs = 800e-9 # central wavelength of seed pulse
es = 27e-6 # energy in seed pulse

τfwhm = 30e-15 # pulse duration of both pulses

radius = 75e-6 # HCF core radius
flength = 1.3 # HCF length
gas = :He
pressure = 1.8 # helium pressure in bar

pump = Pulses.GaussPulse(;λ0=λp, energy=ep, τfwhm, polarisation=:circular)
seed_equal = Pulses.GaussPulse(;λ0=λs, energy=es, τfwhm, polarisation=1.0)
seed_opposite = Pulses.GaussPulse(;λ0=λs, energy=es, τfwhm, polarisation=-1.0)

# Here we are not specifiying a mode, so Luna will automatically choose to propagate in 
# the fundamental mode only--but because the pulses are circularly polarised, two modes
# with perpendicular linear polarisation are used.
equal = prop_capillary(radius, flength, gas, pressure; λ0=λp, pulses=[pump, seed_equal],
                       trange=500e-15, λlims=(200e-9, 1500e-9))

opposite = prop_capillary(radius, flength, gas, pressure; λ0=λp, pulses=[pump, seed_opposite],
                          trange=500e-15, λlims=(200e-9, 1500e-9))


# retrieve the spectral energy density at the output
λ, Iλequal = Processing.getIω(equal, :λ, flength)
_, Iλopposite = Processing.getIω(opposite, :λ, flength)

# sum over the two modes
Iλequal = dropdims(sum(Iλequal; dims=2); dims=(2, 3))
Iλopposite = dropdims(sum(Iλopposite; dims=2); dims=(2, 3))

# plot the result
pyplot.figure()
pyplot.plot(λ*1e9, Iλequal*1e-3, label="Equal helicity")
pyplot.plot(λ*1e9, Iλopposite*1e-3, label="Opposite helicity")
pyplot.xlim(200, 950)
pyplot.xlabel("Wavelength (nm)")
pyplot.ylabel("Spectral energy density (μJ/nm)")
pyplot.ylim(ymin=0)
pyplot.legend()

