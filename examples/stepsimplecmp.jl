using Luna
using Polynomials
a = 1.25e-6
flength = 2e-2
fr = 0.18
τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12
grid = Grid.EnvGrid(flength, λ0, (400e-9, 1400e-9), 10e-12)

m = StepIndexFibre.StepIndexMode(a, accellims=(400e-9, 1400e-9, 100))
aeff = let aeffc = Modes.Aeff(m, z=0)
    z -> aeffc
end
densityfun = z -> 1.0
linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)
responses = (Nonlinear.Kerr_env((1 - fr)*PhysData.χ3(:SiO2)),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(:SiO2, fr*PhysData.ε_0*PhysData.χ3(:SiO2))))
inputs = (Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)
outputm = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, outputm)

ωs = range(PhysData.wlfreq(1400e-9), PhysData.wlfreq(400e-9), length=300)
βs = Modes.β.(m,ωs)
ω0 = PhysData.wlfreq(λ0)
p = Polynomials.fit(ωs .- ω0, βs, 10)
βcoeffs = p.coeffs .* factorial.((1:length(p)) .- 1)

N0, n0, n2 = Tools.getN0n0n2(ω0, :SiO2)
γ = Tools.getγ(ω0, m, n2)

s = SimpleFibre.SimpleMode(ω0, βcoeffs)
aeff = z -> 1.0
linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, s, λ0)
k0 = 2π/λ0
n2 = γ/k0*aeff(0.0)
n0 = real(PhysData.ref_index(:SiO2, 1030e-9))
χ3 = 4/3 * n2 * (PhysData.ε_0*PhysData.c) * n0
responses = (Nonlinear.Kerr_env((1 - fr)*χ3),
             Nonlinear.RamanPolarEnv(grid.to, Raman.raman_response(:SiO2, fr*χ3*PhysData.ε_0)))
norm! = NonlinearRHS.norm_mode_average_gnlse(grid, aeff)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff, norm! = norm!)
outputs = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, outputs)

using PyPlot
plot(Processing.getIω(outputm, :λ, flength)...)
plot(Processing.getIω(outputs, :λ, flength)...)
