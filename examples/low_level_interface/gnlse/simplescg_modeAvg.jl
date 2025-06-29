# supercontinuum from simple GNLSE parameters
# Fig.3 of Dudley et. al, RMP 78 1135 (2006)

using Luna, PythonPlot

βs =  [0.0, 0.0, -1.1830e-26, 8.1038e-41, -9.5205e-56,  2.0737e-70, -5.3943e-85,  1.3486e-99, -2.5495e-114,  3.0524e-129, -1.7140e-144]
γ = 0.11
flength = 15e-2
fr = 0.18
τfwhm = 50e-15
λ0 = 835e-9
energy = 568e-12

grid = Grid.RealGrid(flength, λ0, (400e-9, 1400e-9), 10e-12)

m = SimpleFibre.SimpleMode(PhysData.wlfreq(λ0), βs)
aeff = z -> 1.0
densityfun = z -> 1.0

linop, βfun!, β1, αfun = LinearOps.make_const_linop(grid, m, λ0)

k0 = 2π/λ0
n2 = γ/k0*aeff(0.0)
χ3 = 4/3 * n2 * (PhysData.ε_0*PhysData.c)
responses = (Nonlinear.Kerr_field((1 - fr)*χ3),
             Nonlinear.RamanPolarField(grid.to, Raman.raman_response(grid.to, :SiO2, fr*χ3*PhysData.ε_0)))

inputs = (Fields.SechField(λ0=λ0, τfwhm=τfwhm, energy=energy), Fields.ShotNoise())
norm! = NonlinearRHS.norm_mode_average_gnlse(grid, aeff)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff, norm! = norm!)

output = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, output)

##

#Plotting.stats(output)
#Plotting.prop_2D(output, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))
#Plotting.time_1D(output, range(0.0, 1.0, length=5).*flength, trange=(-1e-12, 5e-12))
Plotting.spec_1D(output, range(0.0, 1.0, length=5).*flength, λrange=(400e-9, 1300e-9))
