import FFTW
FFTW.set_num_threads(Threads.nthreads())
using Luna

function setup()
    a = 13e-6
    gas = :N2
    pres = 5
    flength = 1.5e-2
    τfwhm = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    modes = collect(Capillary.MarcatilliMode(a, gas, pres, n=1, m=m,
                                            kind=:HE, ϕ=0.0, loss=false)
                    for m in 1:10)
    grid = Grid.RealGrid(flength, λ0, (180e-9, 3000e-9), 1e-12)
    energyfun, energyfunω = Fields.energyfuncs(grid)
    normfun = NonlinearRHS.norm_modal(grid.ω)
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    ionpot = PhysData.ionisation_potential(gas)
    ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
                Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot),
                Nonlinear.RamanPolarField(grid.to, Raman.raman_response(gas)))
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
    Eω, transform, FT = Luna.setup(grid, densityfun, normfun, responses, inputs,
                                modes, :y; full=true)
    statsfun = Stats.collect_stats(grid, Eω, Stats.ω0(grid))
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    grid, statsfun, Eω, linop, transform, FT
end

function run(grid, statsfun, Eω, linop, transform, FT, output)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)
end

grid, statsfun, Eω, linop, transform, FT = setup()
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
@time run(grid, statsfun, Eω, linop, transform, FT, output)
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
@time run(grid, statsfun, Eω, linop, transform, FT, output)
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
@time run(grid, statsfun, Eω, linop, transform, FT, output)
