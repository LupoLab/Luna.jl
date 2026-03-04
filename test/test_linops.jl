import Luna: PhysData, Grid, LinearOps, Modes, Capillary
import Test: @testset, @test
import Luna.PhysData: wlfreq
import Luna: Hankel

R = 5e-3
Nr = 256
Nx = 128
Ny = 64
gas = :Ar
pressure = 1

@testset "free space" begin
    rgrid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
    egrid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
    q = Hankel.QDHT(R, Nr, dim=3)
    xygrid = Grid.FreeGrid(R, Nx, R, Ny)
    xgrid = Grid.Free2DGrid(R, Nx)

    getshape(grid, q::Hankel.QDHT, pol) = (length(grid.ω), pol ? 2 : 1, q.N)
    getshape(grid, sg::Grid.Free2DGrid, pol) = (length(grid.ω), pol ? 2 : 1, length(sg.x))
    getshape(grid, sg::Grid.FreeGrid, pol) = (length(grid.ω), pol ? 2 : 1, length(sg.x), length(sg.y))

    @testset "$(typeof(grid)), $(typeof(sg)), pol = $pol" for sg in (q, xgrid, xygrid),
                                                              pol in (false, true),
                                                              grid in (rgrid, egrid)
        nfunλ = PhysData.ref_index_fun(gas, pressure)
        if pol
            nfun = (λ; z=0.0) -> (nfunλ(λ), nfunλ(λ))
        else
            nfun = (λ; z=0.0) -> nfunλ(λ)
        end
        nfunω = (ω; z) -> nfun(wlfreq(ω); z)

        linop = LinearOps.make_const_linop(grid, sg, nfun)
        linopf = LinearOps.make_linop(grid, sg, nfunω)
        out = similar(linop)

        @test size(linop) == getshape(grid, sg, pol)

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
