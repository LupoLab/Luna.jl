import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes
import Luna.PhysData: wlfreq
import Test: @test, @testset

@testset "field" begin
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 15e-2

# Common setup
grid = Grid.RealGrid(L, 800e-9, (160e-9, 3000e-9), 1e-12)
function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

# Constant
dens0 = PhysData.density(gas, pres)
dens(z) = dens0
m = Capillary.MarcatilliMode(a, gas, pres, loss=false)
energyfun = NonlinearRHS.energy_mode_avg(m)
linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
Eω, transform, FT = Luna.setup(grid, energyfun, dens, normfun, responses, inputs)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output_const = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_const)

# Gradient
coren, densityfun = Capillary.gradient(gas, L, pres, pres);
m = Capillary.MarcatilliMode(a, coren, loss=false);
energyfun = NonlinearRHS.energy_mode_avg(m)
linop, βfun = LinearOps.make_linop(grid, m, λ0);
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output_grad = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad)

@test all(output_grad.data["Eω"][2:end, :] .≈ output_const.data["Eω"][2:end, :])
end

@testset "envelope" begin
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 15e-2

# Common setup
grid = Grid.EnvGrid(L, 800e-9, (160e-9, 3000e-9), 1e-12)
function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)
end
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)

# Constant
dens0 = PhysData.density(gas, pres)
dens(z) = dens0
m = Capillary.MarcatilliMode(a, gas, pres, loss=false);
energyfun = NonlinearRHS.energy_env_mode_avg(m);
linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0);
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
Eω, transform, FT = Luna.setup(grid, energyfun, dens, normfun, responses, inputs)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output_const = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_const)

# Gradient
coren, densityfun = Capillary.gradient(gas, L, pres, pres);
m = Capillary.MarcatilliMode(a, coren, loss=false);
energyfun = NonlinearRHS.energy_env_mode_avg(m);
linop, βfun = LinearOps.make_linop(grid, m, λ0);
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun)
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output_grad = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_grad)

@test all(output_grad.data["Eω"][grid.sidx, :] .≈ output_const.data["Eω"][grid.sidx, :])
end