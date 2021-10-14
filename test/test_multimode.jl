import Test: @test, @testset, @test_throws

@testset "Radial" begin
    # mode average and radial integral for single mode and only Kerr should be identical
    using Luna
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Capillary.MarcatiliMode(a, gas, pres, loss=false)
    aeff(z) = Modes.Aeff(m, z=z)
    energyfun, energyfunω = Fields.energyfuncs(grid)

    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
    output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

    modes = (
         Capillary.MarcatiliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    energyfun, energyfunω = Fields.energyfuncs(grid)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                modes, :y; full=false)
    outputr = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputr, status_period=10)

    Iω = abs2.(output.data["Eω"])
    Iωr = abs2.(dropdims(outputr.data["Eω"], dims=2))
    @test norm(Iω - Iωr)/norm(Iω) < 0.003
end

@testset "Full" begin
    # mode average and full integral for single mode and only Kerr should be identical
    using Luna
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    m = Capillary.MarcatiliMode(a, gas, pres, loss=false)
    aeff(z) = Modes.Aeff(m, z=z)
    energyfun, energyfunω = Fields.energyfuncs(grid)

    dens0 = PhysData.density(gas, pres)
    densityfun = let dens0=PhysData.density(gas, pres)
        z -> dens0
    end
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
    output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

    modes = (
         Capillary.MarcatiliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    energyfun, energyfunω = Fields.energyfuncs(grid)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                modes, :y; full=true)
    outputf = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputf, status_period=10)

    Iω = abs2.(output.data["Eω"])
    Iωf = abs2.(dropdims(outputf.data["Eω"], dims=2))
    @test norm(Iω - Iωf)/norm(Iω) < 0.003
end

@testset "Full, LP11 vs TM01" begin
    # TM01 and LP11 (which is TM01+HE21) have the same dispersion and a ratio of effective
    # areas of 2/3 -- mode averaged propagation in TM01 with 2/3x Aeff should be identical
    # to full modal propagation in LP11, as long as only Kerr is considered.
    using Luna
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    energy = 1e-6
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 0.4e-12)
    m = Capillary.MarcatiliMode(a, gas, pres; kind=:TM, n=0, m=1, loss=false)
    aeff(z) = 2/3*Modes.Aeff(m, z=z) # Aeff of LP11 is 2/3x smaller than that of TM01
    energyfun, energyfunω = Fields.energyfuncs(grid)

    dens0 = PhysData.density(gas, pres)
    densityfun = let dens0=PhysData.density(gas, pres)
        z -> dens0
    end
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
    Eω, transform, FT = Luna.setup(
        grid, densityfun, responses, inputs, βfun!, aeff)
    statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
    output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

    # vertical LP11 double-lobe
    modes = (
        Capillary.MarcatiliMode(a, gas, pres, n=0, m=1, kind=:TM, ϕ=0.0, loss=false),
        Capillary.MarcatiliMode(a, gas, pres, n=2, m=1, kind=:HE, ϕ=-π/4, loss=false),
    )

    field = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy/2)
    inputs = ((mode=1, fields=(field,)), (mode=2, fields=(field,)))
    # inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                   modes, :xy; full=true)
    outputf = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    Luna.run(Eω, grid, linop, transform, FT, outputf, status_period=10)

    Iω = abs2.(output["Eω"])
    Iωf = dropdims(sum(abs2.(outputf["Eω"]); dims=2), dims=2)
    @test norm(Iω - Iωf)/norm(Iω) < 0.0003
end

@testset "FieldInputs" begin
    using Luna
    import LinearAlgebra: norm
    a = 13e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(5e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
    modes = (
         Capillary.MarcatiliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
         Capillary.MarcatiliMode(a, gas, pres, n=1, m=2, kind=:HE, ϕ=0.0, loss=false)
    )
    aeff(z) = Modes.Aeff(m, z=z)
    energyfun, energyfunω = Fields.energyfuncs(grid)
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
    Eω_single, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                modes, :y; full=false)
    inputs = (Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6),)
    Eω_tuple, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                modes, :y; full=false)
    inputs = ((mode=1, fields=(Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6),)),)
    Eω_tuple_of_namedtuples, transform, FT = Luna.setup(grid, densityfun, responses, inputs,
                                modes, :y; full=false)
    @test Eω_single ≈ Eω_tuple
    @test Eω_single ≈ Eω_tuple_of_namedtuples
end
