using Luna
import PyPlot: plt

#= In this example we simulate the effect of pump pulse dispersion on soliton self-compression
and RDW emission. To do this we use the "ϕ" and "propagator" keyword arguments to
prop_capillary to apply fixed third-order dispersion and variable glass insertion,
respectively.
By using the default settings otherwise, this example runs a mode-averaged simulation. This
is less accurate than multi-mode due to the presence of ionisation in the propagation, but
much faster, enabling a quick run of multiple simulations like this. =# 

a = 75e-6 # core radius
flength = 0.6 # HCF length
gas = :Ne # gas species
pressure = 1.5 # gas pressure in bar

λ0 = 800e-9
τfwhm = 6e-15
energy = 100e-6

glassprop! = Fields.propagator_material(:SiO2)

thicknesses = [-2e-3, -1e-3, 0, 1e-3]

outputs = [prop_capillary(a, flength, gas, pressure; trange=1e-12, λlims=(150e-9, 4e-6),
                          λ0, τfwhm, energy,
                          ϕ=[0, 0, 0, -350e-45],
                          propagator=(Eω, grid) -> glassprop!(Eω, grid.ω, di, λ0))
            for di in thicknesses]
##
λ, _ = Processing.getIω(outputs[1], :λ)
Iλs = [Processing.getIω(oi, :λ)[2] for oi in outputs]

plt.figure()
for (d, Iλ) in zip(thicknesses, Iλs)
    plt.semilogy(λ*1e9, 1e-3*Iλ[:, end], label="$(d*1e3) mm")
end
plt.xlim(130, 1100)
plt.ylim(1e-5, 10)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral energy density (μJ/nm)")
plt.legend()