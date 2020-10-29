using Luna

a = 13e-6
gas = :Ar
pres = 7.5
L = 15e-2

τfwhm = 30e-15
λ0 = 800e-9
energy = 1e-6

coren, densityfun = Capillary.gradient(gas, L, pres, 0);

modes = (
    Capillary.MarcatilliMode(a, coren, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, coren, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)
nmodes = length(modes)

grid = Grid.EnvGrid(L, λ0, (160e-9, 3000e-9), 1e-12)

energyfun = Fields.energyfuncs(grid)[1]

responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)

Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                               modes, :y; full=false)

linop = LinearOps.make_linop(grid, modes, λ0)
statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas, windows=((150e-9, 300e-9),))
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)

Luna.run(Eω, grid, linop, transform, FT, output)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)
Plotting.prop_2D(output, bandpass=(150e-9, 300e-9))
Plotting.time_1D(output, [5e-2, 9.8e-2])
Plotting.time_1D(output, [5e-2, 9e-2], modes=:sum, bandpass=(150e-9, 300e-9))
Plotting.spec_1D(output, [5e-2, 9.8e-2])
Plotting.spec_1D(output, [5e-2, 9.8e-2], modes=:sum)
