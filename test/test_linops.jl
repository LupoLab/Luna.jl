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

    @testset "$(typeof(grid)), $(typeof(sg)), pol = $pol, thg = $thg" for sg in (q, xgrid, xygrid),
                                                              pol in (false, true),
                                                              thg in (false, true),
                                                              grid in (rgrid, egrid)
        if grid isa Grid.RealGrid && ~thg
            continue
        end
        nfunλ = PhysData.ref_index_fun(gas, pressure)
        if pol
            nfun = (λ; z=0.0) -> (nfunλ(λ), nfunλ(λ))
        else
            nfun = (λ; z=0.0) -> nfunλ(λ)
        end
        nfunω = (ω; z) -> nfun(wlfreq(ω); z)

        linop = LinearOps.make_const_linop(grid, sg, nfun, thg)
        linopf = LinearOps.make_linop(grid, sg, nfunω, thg)
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

@testset "free space birefringent (tuple nfuns)" begin
    rgrid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
    egrid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
    xygrid = Grid.FreeGrid(R, Nx, R, Ny)
    xgrid = Grid.Free2DGrid(R, Nx)

    #=
    Isotropic index: the tuple (crystal) path must match the generic vector-nfun path.
    The tuple path always subtracts β1*ω (a pure time shift, transparent to carrier-mixing
    nonlinearities), whereas the generic path subtracts β1*(ω - ω0) + β0—so for EnvGrid the
    two differ by the constant frame phase β1*ω0 - β0.
    =#
    nfunλ = PhysData.ref_index_fun(gas, pressure)
    nfunx = (λ, δθ=0.0) -> real(nfunλ(λ))
    nfuny = λ -> real(nfunλ(λ))
    nfun = (λ; z=0.0) -> (real(nfunλ(λ)), real(nfunλ(λ)))

    @testset "isotropic: $(typeof(grid)), $(typeof(sg)), thg = $thg" for sg in (xgrid, xygrid),
                                                              thg in (false, true),
                                                              grid in (rgrid, egrid)
        if grid isa Grid.RealGrid && ~thg
            continue
        end
        linop = LinearOps.make_const_linop(grid, sg, (nfunx, nfuny))
        linopv = LinearOps.make_const_linop(grid, sg, nfun, thg)
        @test size(linop) == size(linopv)
        β1 = PhysData.dispersion_func(1, nfuny)(grid.referenceλ)
        ω0 = LinearOps.getω0(grid, thg)
        β0 = LinearOps.getβ0_n(grid, λ -> nfun(λ), thg)
        offset = im*(β1*ω0 - β0) # 0 for RealGrid and for EnvGrid with thg=true
        # the tuple path only fills frequencies within grid.sidx
        sel = ntuple(_ -> Colon(), ndims(linop) - 1)
        @test all(isapprox.(linop[grid.sidx, sel...], linopv[grid.sidx, sel...] .+ offset;
                            atol=1e-3, rtol=0))
    end

    # for an envelope grid the phase subtracted is β1*ω: check the referencing directly
    # for the y polarisation at kperp = 0
    linop = LinearOps.make_const_linop(egrid, xgrid, (nfunx, nfuny))
    ik0 = argmin(abs.(xgrid.kx))
    β1 = PhysData.dispersion_func(1, nfuny)(egrid.referenceλ)
    for iω in (argmin(abs.(egrid.ω .- egrid.ω0)), findfirst(egrid.sidx))
        ωi = egrid.ω[iω]
        expected = -(nfuny(PhysData.wlfreq(ωi))*ωi/PhysData.c - β1*ωi)
        @test isapprox(imag(linop[iω, 2, ik0]), expected; atol=1e-6, rtol=0)
        @test real(linop[iω, 2, ik0]) == 0
    end

    # real birefringent crystal on an envelope grid
    θ = deg2rad(29.2)
    bbogrid = Grid.EnvGrid(200e-6, 800e-9, (250e-9, 2e-6), 120e-15; thg=true)
    bboxgrid = Grid.Free2DGrid(80e-6, 32)
    nfuns = PhysData.ref_index_fun_xy(:BBO, θ)
    linop = LinearOps.make_const_linop(bbogrid, bboxgrid, nfuns)
    @test size(linop) == (length(bbogrid.ω), 2, length(bboxgrid.kx))
    @test all(isfinite, linop)
    # birefringence: the two polarisations see different indices
    @test any(linop[bbogrid.sidx, 1, :] .!= linop[bbogrid.sidx, 2, :])
end

@testset "equivalence for fast z-dependent linops" begin
a = 125e-6
L = 1
grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12)
coren, densityfun = Capillary.gradient(gas, L, pressure, 0)
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
coren, densityfun = Capillary.gradient(gas, L, pressure, 0)
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
coren, densityfun = Capillary.gradient(gas, L, pressure, 0)
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
