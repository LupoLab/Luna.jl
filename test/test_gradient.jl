import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes, Fields
import Luna.PhysData: wlfreq
import Test: @test, @testset

@testset "multi-point" begin
Z = [0, 0.25, 0.5, 1]
P = [0, 1, 0.5, 0]
coren, densityfun = Capillary.gradient(:He, Z, P)
for ii = 1:3
    λ = Z[ii+1] - Z[ii]
    ζ = range(0, λ; length=10)
    z = range(Z[ii], Z[ii+1]; length=10)
    p0, p1 = P[ii], P[ii+1]
    Pζ = @. sqrt(p0^2 + ζ/λ*(p1^2 - p0^2))
    ρζ = PhysData.density.(:He, Pζ)
    ρz = densityfun.(z)
    @test all(isapprox.(ρζ, ρz, rtol=1e-10))
end
end

@testset "field" begin
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 5e-2

# Common setup
grid = Grid.RealGrid(L, λ0, (160e-9, 3000e-9), 0.5e-12)
inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

# Constant
dens0 = PhysData.density(gas, pres)
dens(z) = dens0
m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid)
linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(
    grid, dens, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_const = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_const, status_period=10)

# Gradient
coren, densityfun = Capillary.gradient(gas, L, pres, pres)
m = Capillary.MarcatilliMode(a, coren, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid)
linop, βfun! = LinearOps.make_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(
    grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_grad = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad, status_period=10)

# Gradient array
coren, densityfun = Capillary.gradient(gas, [0,L], [pres, pres]);
m = Capillary.MarcatilliMode(a, coren, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid)
linop, βfun! = LinearOps.make_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(
    grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_grad_array = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad_array, status_period=10)

@test all(output_grad.data["Eω"][grid.sidx, :] .≈ output_const.data["Eω"][grid.sidx, :])
@test all(output_grad_array.data["Eω"][grid.sidx, :] .≈ output_const.data["Eω"][grid.sidx, :])
end

@testset "envelope" begin
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 5e-2

# Common setup
grid = Grid.EnvGrid(L, λ0, (160e-9, 3000e-9), 0.5e-12)
inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)
responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

# Constant
dens0 = PhysData.density(gas, pres)
dens(z) = dens0
m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid);
linop, βfun!, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0);
Eω, transform, FT = Luna.setup(grid, dens, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_const = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_const, status_period=10)

# Gradient
coren, densityfun = Capillary.gradient(gas, L, pres, pres)
m = Capillary.MarcatilliMode(a, coren, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid)
linop, βfun! = LinearOps.make_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(
    grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_grad = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad, status_period=10)

# Gradient array
coren, densityfun = Capillary.gradient(gas, [0,L], [pres, pres]);
m = Capillary.MarcatilliMode(a, coren, loss=false)
aeff(z) = Modes.Aeff(m, z=z)
energyfun, energyfunω = Fields.energyfuncs(grid)
linop, βfun! = LinearOps.make_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(
    grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.collect_stats(grid, Eω,
                               Stats.ω0(grid),
                               Stats.energy(grid, energyfunω))
output_grad_array = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad_array, status_period=10)

@test all(output_grad.data["Eω"][grid.sidx, :] .≈ output_const.data["Eω"][grid.sidx, :])
@test all(output_grad_array.data["Eω"][grid.sidx, :] .≈ output_const.data["Eω"][grid.sidx, :])
end
