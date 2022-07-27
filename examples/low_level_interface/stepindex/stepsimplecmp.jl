using Luna
using Luna.PhysData: Polynomials
import PyPlot: plt

a = 1.25e-6
flength = 15e-2
fr = 0.18
τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12
grid = Grid.EnvGrid(flength, λ0, (400e-9, 1400e-9), 10e-12)

# supercontinuum in a strand of silica in air
m = StepIndexFibre.StepIndexMode(a, accellims=(400e-9, 1400e-9, 100))
aeff = let aeffc = Modes.Aeff(m, z=0)
    z -> aeffc
end
densityfun = z -> 1.0
linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)
responses = (Nonlinear.Kerr_env((1 - fr)*PhysData.χ3(:SiO2)),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(grid.to, :SiO2, fr*PhysData.ε_0*PhysData.χ3(:SiO2))))
inputs = (Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)
outputm = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, outputm)

Plotting.prop_2D(outputm, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))

##
# Approximate step-index mode with β coefficients
ω0 = PhysData.wlfreq(λ0)

βcoeffs = Modes.dispersion.(m, collect(0:9), ω0) # dispersion coefficients

N0, n0, n2 = Tools.getN0n0n2(ω0, :SiO2)
γ = Tools.getγ(ω0, m, n2)

βcoeffs_plot = copy(βcoeffs)
βcoeffs_plot[1:2] .= 0 # remove constant and linear term
p = Polynomials.Polynomial(βcoeffs_plot ./ factorial.((1:length(βcoeffs)) .- 1))
ωs = range(PhysData.wlfreq(1400e-9), PhysData.wlfreq(400e-9), length=300)
βs = Modes.β_ret.(m, ωs; λ0) # β with constant and linear term removed
plt.figure()
plt.plot(ωs .- ω0, βs, label="Full")
plt.plot(ωs .- ω0, p.(ωs .- ω0), label="Taylor expansion")
plt.legend()
plt.xlabel("ω-ω0 (rad/s)")
plt.ylabel("β (1/m)")


##
s = SimpleFibre.SimpleMode(ω0, βcoeffs)
aeff = z -> 1.0
linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, s, λ0)
k0 = 2π/λ0
n2 = γ/k0*aeff(0.0)
n0 = real(PhysData.ref_index(:SiO2, 1030e-9))
χ3 = 4/3 * n2 * (PhysData.ε_0*PhysData.c) * n0 * n0 / Modes.neff(m, ω0)
responses = (Nonlinear.Kerr_env((1 - fr)*χ3),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(grid.to, :SiO2, fr*χ3*PhysData.ε_0)))
norm! = NonlinearRHS.norm_mode_average_gnlse(grid, aeff)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff, norm! = norm!)
outputs = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, outputs)

Plotting.prop_2D(outputs, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))

#isapprox(Processing.getIω(outputm, :λ, flength)[2], Processing.getIω(outputs, :λ, flength)[2], rtol=1.1e-1)

##
Plotting.prop_2D(outputm)
Plotting.prop_2D(outputs)
##
λ, Iλm = Processing.getIω(outputm, :λ, flength)
_, Iλs = Processing.getIω(outputs, :λ, flength)
plt.figure()
plt.semilogy(1e9λ, Iλm, label="step-index mode")
plt.semilogy(1e9λ, Iλs, "--",  label="GNLSE approximation")
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("SED (a.u.)")

# close but not exact. This is because we cannot fully cancel the frequency dependence of neff.