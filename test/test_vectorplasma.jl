import Test: @test, @testset, @test_throws

@testset "Vector plasma" begin
    # scalar modal model, vector modal at 0 degrees and vector modal at 45 degrees should be identical
    using Luna
    import LinearAlgebra: norm
    a = 9e-6
    gas = :Ar
    pres = 5
    flength = 5e-2
    τfwhm = 30e-15
    λ0 = 1500e-9
    energy = 1.7e-6
    grid = Grid.RealGrid(flength, λ0, (600e-9, 3000e-9), 1e-12)
    energyfun, energyfunω = Fields.energyfuncs(grid)
    dens0 = PhysData.density(gas, pres)
    densityfun(z) = dens0
    ionpot = PhysData.ionisation_potential(gas)
    ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
    # scalar modal
    modes = (
        Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
    )
    nmodes = length(modes)
    plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to,
                                    ionrate, ionpot)              
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
                plasma)
    inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes,
                                :y; full=false)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas)
    outscalar = Output.MemoryOutput(0, grid.zmax, 2, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, outscalar)
    # vector linear 0 degrees
    plasma = Nonlinear.PlasmaCumtrapz(grid.to, Array{Float64}(undef, length(grid.to), 2),
                                      ionrate, ionpot)                  
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             plasma)
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes,
                                   :xy; full=false)
    outvector = Output.MemoryOutput(0, grid.zmax, 2, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, outvector)

    Iωs = abs2.(outscalar.data["Eω"][:,1,1])
    Iωv = abs2.(outvector.data["Eω"][:,1,1])
    @test Iωs ≈ Iωv

    modes = (
        Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
        Capillary.MarcatilliMode(a, gas, pres, n=1, m=1, kind=:HE, ϕ=π/2, loss=false),
    )
    nmodes = length(modes)
    inputs = ((mode=1, fields=(Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy/2),)),
              (mode=2, fields=(Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy/2),)))
    Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, modes,
                                   :xy; full=false)
    linop = LinearOps.make_const_linop(grid, modes, λ0)
    statsfun = Stats.default(grid, Eω, modes, linop, transform; gas=gas)
    outvector45 = Output.MemoryOutput(0, grid.zmax, 2, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, outvector45)
    Iωv45 = abs2.(outvector45.data["Eω"][:,1,1]) .+ abs2.(outvector45.data["Eω"][:,2,1])
    @test Iωs ≈ Iωv45
end
