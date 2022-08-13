# supercontinuum from simple GNLSE parameters
# Fig.3 of Dudley et. al, RMP 78 1135 (2006)

using Luna

γ = 0.11
flength = 15e-2
βs =  [0.0, 0.0, -1.1830e-26, 8.1038e-41, -9.5205e-56,  2.0737e-70, -5.3943e-85,  1.3486e-99, -2.5495e-114,  3.0524e-129, -1.7140e-144]

τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12

output = prop_gnlse(γ, flength, βs; λ0, τfwhm, energy, λlims=(400e-9, 1400e-9), trange=10e-12)

##
Plotting.pygui(true)
Plotting.prop_2D(output, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))
Plotting.spec_1D(output, range(0.0, 1.0, length=5).*flength, λrange=(400e-9, 1300e-9))
