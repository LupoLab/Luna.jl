import Test: @test, @testset, @test_throws
import Luna: Output

@testset "Linear" begin
    # compare radial single pol, to radial linear pol at 45 degrees,
    # in capillary (non birefringent) these should be identical

    import Luna
    import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.EnvGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    function gausspulse(t)
        It = Maths.gauss(t, fwhm=τ)
        Et = @. sqrt(It)
    end
    densityfun = let dens0=PhysData.density(gas, pres)
        z -> dens0
    end
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    energyfun, energyfunω = NonlinearRHS.energy_modal(grid)
    normfun = NonlinearRHS.norm_modal(grid.ω)

    modes = (
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    in1 = (func=gausspulse, energy=1e-6)
    inputs = ((1,(in1,)),)
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                                modes, :y; full=true)
    statsfun = Stats.collect_stats(grid, Eω,
                                Stats.ω0(grid),
                                Stats.peakintensity(grid, modes, components=:xy),
                                Stats.fwhm_r(grid, modes, components=:xy),
                                Stats.energy(grid, energyfunω))
    output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=10, init_dz=1e-3)

    modes = (
        Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2, loss=false)
    )
    in1 = (func=gausspulse, energy=1e-6/2.0)
    # same field in each mode
    inputs = ((1, (in1,)), (2, (in1,)))
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                                modes, :xy; full=true)
    statsfun = Stats.collect_stats(grid, Eω,
                                Stats.ω0(grid),
                                Stats.peakintensity(grid, modes, components=:xy),
                                Stats.fwhm_r(grid, modes, components=:xy),
                                Stats.energy(grid, energyfunω))
    outputp = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputp, status_period=10, init_dz=1e-3)

    Iω = dropdims(abs2.(output.data["Eω"]), dims=2)
    Iωp = dropdims(sum(abs2.(outputp.data["Eω"]), dims=2), dims=2)

    @test norm(Iω - Iωp)/norm(Iω) < 1.07e-12
    @test all(output["stats"]["peakintensity"] .≈ outputp["stats"]["peakintensity"])
    @test all(output["stats"]["energy"] .≈ sum(outputp["stats"]["energy"], dims=1))
    @test all(output["stats"]["fwhm_r"] .≈ outputp["stats"]["fwhm_r"])
end
