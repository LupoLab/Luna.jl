using Luna

a = 125e-6
flength = 3
gas = :HeJ

λ0 = 800e-9
τfwhm = 10e-15

energies = collect(range(50e-6, 200e-6; length=2))*1e-3
pressures = collect(0.6:0.8:1.4)

scan = Scan("pressure_energy_example"; energy=energies)
addvariable!(scan, :pressure, pressures)
# addvariable!(scan, :gradient, [false, true])
gradient = false

runscan(scan) do scanidx, energy, pressure
    p = gradient ? (pressure*3/2, 0) : pressure
    prop_capillary(a, flength, gas, p; λ0, τfwhm, energy,
                   trange=400e-15)
end
