using Luna
import FFTW
import LinearAlgebra: mul!, ldiv!

import DSP.Unwrap: unwrap

import PyPlot:pygui, plt

a = 13e-6
gas = :Ar
pres = 5

τ = 30e-15
λ0 = 800e-9

modes = (
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    Capillary.MarcatilliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
)

grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

energyfun, energyfunω = NonlinearRHS.energy_modal(grid)
normfun = NonlinearRHS.norm_modal(grid.ω)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_PPTcached(gas, λ0)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            #  Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

in1 = (func=gausspulse, energy=1e-6)
inputs = ((1,(in1,)),)

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                              modes, :y; full=false)

statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω),
                               Stats.energy_λ(grid, energyfunω, (150e-9, 300e-9), label="RDW"),
                               Stats.peakpower(grid),
                               Stats.peakintensity(grid, modes),
                               Stats.fwhm_t(grid),
                               Stats.fwhm_r(grid, modes),
                            #    Stats.electrondensity(grid, ionrate, densityfun, modes),
                               Stats.pressure(densityfun, gas),
                               Stats.zdw(modes),
                               Stats.core_radius(a)
                               )
output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
linop = LinearOps.make_const_linop(grid, modes, λ0)

Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

##
Plotting.pygui(true)
Plotting.stats(output)
Plotting.prop_2D(output)

