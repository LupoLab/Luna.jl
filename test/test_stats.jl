using Luna
import Test: @test, @testset
import HCubature: hquadrature
import SpecialFunctions: besselj
import FunctionZeros: besselj_zero

@testset "On-axis intensity statistics" begin
    # Manually normalise the field distribution of HE11 to find scaling factor between
    # power and intensity
    a = 13e-6
    unm = besselj_zero(0, 1)
    E(r) = besselj(0, unm*r/a)
    norm, err = hquadrature(0, a) do r
        2π*r * abs2(E(r))
    end

    energy = 1e-6
    gas = :Ar
    pres = 5
    τ = 30e-15
    λ0 = 800e-9
    grid = Grid.RealGrid(15e-2, 800e-9, (160e-9, 3000e-9), 1e-12)
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

    statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas, onaxis=true)
    output = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
    Luna.run(Eω, grid, linop, transform, FT, output, status_period=5)

    @test all(output["stats"]["peakintensity"] .≈ output["stats"]["peakpower"]/norm)
end
