using Luna
import Test: @test, @testset

radius = 125e-6 # HCF core radius
flength = 3 # HCF length

gas = :Ar
pressure = 80e-3 # gas pressure in bar

λ0 = 800e-9 # central wavelength of the pump pulse
τfwhm = 10e-15 # FWHM duration of the pump pulse
energy = 60e-6 # energy in the pump pulse

τ0 = Tools.τfw_to_τ0(τfwhm, :gauss)

duv = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                     modes=:HE11, trange=400e-15, λlims=(150e-9, 4e-6), kerr=false, plasma=false)

t, Et = Processing.getEt(duv)
Ptend = abs2.(Et[:, end]) # pulse profile at end

fwend = Maths.fwhm(t, Ptend; method=:spline) # FWHM duration at end
τ0end = Tools.τfw_to_τ0(fwend, :gauss) # Gaussian τ0 duration at end

mode = Capillary.MarcatiliMode(radius, gas, pressure) # propagation mode
β2 = Modes.dispersion(mode, 2, PhysData.wlfreq(λ0)) # dispersion of the mode
GDD = β2*flength # total GDD accumulated

τ0calc = sqrt((GDD/τ0)^2 + τ0^2) # analytical formula for Gaussian duration after GDD

@test isapprox(τ0calc, τ0end; rtol=1e-2)

energy_calc = Capillary.transmission(radius, λ0, flength)*energy
@test isapprox(energy_calc, Processing.energy(duv)[end]; rtol=1e-2)

