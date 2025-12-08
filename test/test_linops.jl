import Luna: PhysData, Grid, LinearOps, Modes, Capillary
import Test: @testset, @test
import Luna.PhysData: wlfreq
import Luna: Hankel

R = 5e-3
Nr = 256
Nx = 128
Ny = 64
gas = :Ar
pres = 1
nfun = let rif=PhysData.ref_index_fun(gas, pres)
    (ω; z) -> rif(wlfreq(ω))
end

@testset "radial, field" begin
grid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, Nr, dim=2)

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
linopf = LinearOps.make_linop(grid, q, nfun)
out = similar(linop)

@test size(linop) == (length(grid.ω), q.N)

linopf(out, 0.0)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
linopf(out, 0.5)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
end

@testset "radial, env" begin
grid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
grid_thg = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12; thg=true)
q = Hankel.QDHT(R, Nr, dim=2)

for gi in (grid, grid_thg)
    linop = LinearOps.make_const_linop(gi, q, PhysData.ref_index_fun(gas, pres))
    linopf = LinearOps.make_linop(gi, q, nfun)
    out = similar(linop)

    linopf(out, 0.0)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
    linopf(out, 0.5)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
end
end

@testset "3D, field" begin
grid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)

linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
linopf = LinearOps.make_linop(grid, xygrid, nfun)
out = similar(linop)

@test size(linop) == (length(grid.ω), Ny, Nx)

linopf(out, 0.0)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
linopf(out, 0.5)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
end

@testset "3D, env" begin
grid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
grid_thg = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12; thg=true)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)

for gi in (grid, grid_thg)
    linop = LinearOps.make_const_linop(gi, xygrid, PhysData.ref_index_fun(gas, pres))
    linopf = LinearOps.make_linop(gi, xygrid, nfun)
    out = similar(linop)

    @test size(linop) == (length(gi.ω), Ny, Nx)

    linopf(out, 0.0)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
    linopf(out, 0.5)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
end
end

@testset "equivalence for fast z-dependent linops" begin
a = 125e-6
L = 1
grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end

a = 125e-6
L = 1
# NO THG
thg = false
grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12; thg=thg)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version...

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9; thg=thg)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9; thg=thg)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end
# WITH THG
thg = true
grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12; thg=thg)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version...

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9; thg=thg)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9; thg=thg)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end
end