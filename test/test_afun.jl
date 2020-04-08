import Test: @test, @testset, @test_throws
import Luna
import Luna: Grid, Maths, Capillary, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Modes

import LinearAlgebra: norm
@testset "mode-average vs modal" begin
## setup
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 15e-2
grid = Grid.RealGrid(L, 800e-9, (160e-9, 3000e-9), 0.5e-12)

a0 = a
aL = 3a/4
afun = let a0=a0, aL=aL, L=L
    afun(z) = a0 + (aL-a0)*z/L
end

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end
dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0
energyfun = NonlinearRHS.energy_modal()
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

## mode-average
Iωavg = let
m = Capillary.MarcatilliMode(afun, gas, pres, loss=false, model=:full);
aeff(z) = Modes.Aeff(m, z=z)
linop, βfun = LinearOps.make_linop(grid, m, λ0)
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs, aeff)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)
abs2.(output["Eω"])
end

## modal
Iωmodal = let
modes = (
    Capillary.MarcatilliMode(afun, gas, pres, n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
)
linop = LinearOps.make_linop(grid, modes, λ0)
normfun = NonlinearRHS.norm_modal(grid.ω)
in1 = (func=gausspulse, energy=1e-6)
inputs = ((1,(in1,)),)
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs,
                               modes, :y, full=false)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),length(modes)), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)
abs2.(dropdims(output["Eω"], dims=2))
end

@test norm(Iωavg - Iωmodal)/(sqrt(norm(Iωavg))*sqrt(norm(Iωmodal))) < 0.004
end

@testset "const vs afun" begin
## setup
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 15e-2
grid = Grid.RealGrid(L, 800e-9, (160e-9, 3000e-9), 0.5e-12)

afun = let a=a
    z -> a
end

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end
dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0
energyfun = NonlinearRHS.energy_modal()
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

## mode-average
Iωavg = let
m = Capillary.MarcatilliMode(afun, gas, pres, loss=false, model=:full);
aeff(z) = Modes.Aeff(m, z=z)
linop, βfun = LinearOps.make_linop(grid, m, λ0)
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs, aeff)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)
abs2.(output["Eω"])
end

## mode-average, constant a
Iωavg_c = let
m = Capillary.MarcatilliMode(a, gas, pres, loss=false, model=:full);
aeff(z) = Modes.Aeff(m, z=z)
in1 = (func=gausspulse, energy=1e-6)
inputs = (in1, )
linop, βfun, frame_vel, αfun = LinearOps.make_const_linop(grid, m, λ0)
normfun = NonlinearRHS.norm_mode_average(grid.ω, βfun, aeff)
Eω, transform, FT = Luna.setup(grid, energyfun, densityfun, normfun, responses, inputs, aeff)
statsfun = Stats.collect_stats((Stats.ω0(grid), ))
output = Output.MemoryOutput(0, grid.zmax, 201, (length(grid.ω),), statsfun)
Luna.run(Eω, grid, linop, transform, FT, output)
abs2.(output["Eω"])
end
@test Iωavg ≈ Iωavg_c
end