import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps
import Luna.Scans: @scanvar, @scan, @scaninit
import Logging: @info

@scaninit "scantest"

@scanvar energy = range(0.1e-6, 1.5e-6, length=16)
@scanvar τ = range(25e-15, 35e-15, length=11)


@scan begin
a = 13e-6
gas = :Ar
pres = 5

λ0 = 800e-9

grid = Grid.RealGrid(1e-2, 800e-9, (160e-9, 3000e-9), 1e-12)

m = Capillary.MarcatilliMode(a, gas, pres, loss=false)

energyfun = NonlinearRHS.energy_mode_avg(m)

gausspulse = let τ=$τ, λ=λ0
    function gausspulse(t)
        It = Maths.gauss(t, fwhm=τ)
        ω0 = 2π*PhysData.c/λ0
        Et = @. sqrt(It)*cos(ω0*t)
    end
end

println("τ: $($τ * 1e15)")
println("E: $($energy * 1e6)")

dens0 = PhysData.density(gas, pres)
densityfun = let dens0=dens0
    f(z) = dens0
end

ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_ADK(ionpot)

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
            Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))

linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)

normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)

in1 = (func=gausspulse, energy=$energy)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)

statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.@ScanHDF5Output(0, grid.zmax, 101, (length(grid.ω),), statsfun)
println(output["meta"]["scanvars"])

Luna.run(Eω, grid, linop, transform, FT, output)
end
