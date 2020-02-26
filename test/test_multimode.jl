import Test: @test, @testset, @test_throws

@testset "Radial" begin
    # mode average and radial integral for single mode and only Kerr should be identical

    import Luna
    import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
    energyfun = NonlinearRHS.energy_mode_avg(m)
    function gausspulse(t)
        It = Maths.gauss(t, fwhm=τ)
        ω0 = 2π*PhysData.c/λ0
        Et = @. sqrt(It)*cos(ω0*t)
    end
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
    normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
    in1 = (func=gausspulse, energy=1e-6)
    inputs = (in1, )
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)
    statsfun = Stats.collect_stats((Stats.ω0(grid), ))
    output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output)

    modes = (
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    energyfun = NonlinearRHS.energy_modal()
    normfun = NonlinearRHS.norm_modal(grid.ω)
    inputs = ((1,(in1,)),)
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                                modes, :Ey; full=false)
    outputr = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputr)

    Iω = abs2.(output.data["Eω"]).*PhysData.c.*PhysData.ε_0.*Capillary.Aeff(m)./2
    Iωr = abs2.(dropdims(outputr.data["Eω"], dims=2))
    @test norm(Iω - Iωr)/norm(Iω) < 0.003
end

@testset "Full" begin
    # mode average and full integral for single mode and only Kerr should be identical

    import Luna
    import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Modes.@delegated(Capillary.MarcatilliMode(a, gas, pres), α=ω->0)
    energyfun = NonlinearRHS.energy_mode_avg(m)
    function gausspulse(t)
        It = Maths.gauss(t, fwhm=τ)
        ω0 = 2π*PhysData.c/λ0
        Et = @. sqrt(It)*cos(ω0*t)
    end
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
    normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
    in1 = (func=gausspulse, energy=1e-6)
    inputs = (in1, )
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)
    statsfun = Stats.collect_stats((Stats.ω0(grid), ))
    output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output)

    modes = (
         Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    energyfun = NonlinearRHS.energy_modal()
    normfun = NonlinearRHS.norm_modal(grid.ω)
    inputs = ((1,(in1,)),)
    Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                                modes, :Ey; full=true)
    outputf = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputf)

    Iω = abs2.(output.data["Eω"]).*PhysData.c.*PhysData.ε_0.*Capillary.Aeff(m)./2
    Iωf = abs2.(dropdims(outputf.data["Eω"], dims=2))
    @test norm(Iω - Iωf)/norm(Iω) < 0.003
end
