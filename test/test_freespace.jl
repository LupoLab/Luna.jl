#= here we test that everything runs without throwing errors for every combination of:
    1. Real and envelope grids
    2. Radial (QDHT), 2D cartesian and 3D cartesian spatial grids
    3. (For envelope) THG on/off
    4. Constant pressure and pressure gradient
=#
using Luna
import Luna.PhysData: wlfreq
import Luna: Hankel, FFTW
Luna.set_fftw_mode(:estimate)
import LinearAlgebra: norm
import Test: @test, @testset

R = 1.5e-3
Nr = 512
Nx = 256
Ny = 128
gas = :Ar
pressure = 1

λ0 = 800e-9
w0 = 40e-6
τfwhm = 20e-15
energy = 1e-12
L = 0.3

rgrid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
egrid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, Nr, dim=3)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)
xgrid = Grid.Free2DGrid(R, Nx)

getshape(grid, q::Hankel.QDHT, pol) = (length(grid.ω), pol ? 2 : 1, q.N)
getshape(grid, sg::Grid.Free2DGrid, pol) = (length(grid.ω), pol ? 2 : 1, length(sg.x))
getshape(grid, sg::Grid.FreeGrid, pol) = (length(grid.ω), pol ? 2 : 1, length(sg.x), length(sg.y))

makekerr(grid::Grid.RealGrid, thg) = Nonlinear.Kerr_field(PhysData.γ3_gas(gas))
makekerr(grid::Grid.EnvGrid, thg) = thg ? Nonlinear.Kerr_env(PhysData.γ3_gas(gas)) : Nonlinear.Kerr_env_thg(PhysData.γ3_gas(gas), grid.ω0, grid.to)

makeconstnorm(grid, q::Hankel.QDHT, nfunλ) = NonlinearRHS.const_norm_radial(grid, q, nfunλ)
makeconstnorm(grid, sg::Grid.Free2DGrid, nfunλ) = NonlinearRHS.const_norm_free2D(grid, sg, nfunλ)
makeconstnorm(grid, sg::Grid.FreeGrid, nfunλ) = NonlinearRHS.const_norm_free(grid, sg, nfunλ)

makenorm(grid, q::Hankel.QDHT, nfunω) = NonlinearRHS.norm_radial(grid, q, nfunω)
makenorm(grid, sg::Grid.Free2DGrid, nfunω) = NonlinearRHS.norm_free2D(grid, sg, nfunω)
makenorm(grid, sg::Grid.FreeGrid, nfunω) = NonlinearRHS.norm_free(grid, sg, nfunω)

function testfocus(q::Hankel.QDHT, Eω, w0)
    Eωfoc = Eω[:, :, :, end]
    Eωr = q \ Eωfoc
    Ir = Maths.normbymax(dropdims(sum(abs2.(Eωr); dims=(1, 2)); dims=(1, 2)))
    Ir_analytical = Maths.gauss.(q.r, w0/2)
    @test maximum(abs.(Ir .- Ir_analytical)/norm(Ir)) < 5e-3
end

function testfocus(sg::Grid.Free2DGrid, Eω, w0)
    Eωfoc = Eω[:, :, :, end]
    Eωx = FFTW.ifft(Eωfoc, 3)
    Ix = Maths.normbymax(dropdims(sum(abs2.(Eωx); dims=(1, 2)); dims=(1, 2)))
    Ix_analytical = Maths.gauss.(sg.x, w0/2)
    @test maximum(abs.(Ix .- Ix_analytical)/norm(Ix)) < 2e-3
end

function testfocus(sg::Grid.FreeGrid, Eω, w0)
    Eωfoc = Eω[:, :, :, :, end]
    Eωxy = FFTW.ifft(Eωfoc, (3, 4))
    Iy = Maths.normbymax(dropdims(sum(abs2.(Eωxy); dims=(1, 2, 3)); dims=(1, 2, 3)))
    Ix = Maths.normbymax(dropdims(sum(abs2.(Eωxy); dims=(1, 2, 4)); dims=(1, 2, 4)))
    Ix_analytical = Maths.gauss.(sg.x, w0/2)
    @test maximum(abs.(Ix .- Ix_analytical)/norm(Ix)) < 2e-3
    Iy_analytical = Maths.gauss.(sg.y, w0/2)
    @test maximum(abs.(Iy .- Iy_analytical)/norm(Ix)) < 2e-3
end

@testset "Constant pressure: $(typeof(grid)), $(typeof(sg)), pol = $pol, thg = $thg" for sg in (q, xgrid, xygrid),
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

    linop = LinearOps.make_const_linop(grid, sg, nfun, thg)
    dens0 = PhysData.density(gas, pressure)
    densityfun(z) = dens0

    responses = (makekerr(grid, thg),)
    normfun = makeconstnorm(grid, sg, nfun)

    inputs = Fields.GaussGaussField(;λ0, τfwhm, energy, w0, propz=-L)

    Eω, transform, FT = Luna.setup(grid, sg, densityfun, normfun, responses, inputs)
    output = Output.MemoryOutput(0, grid.zmax, 21)
    Luna.run(Eω, grid, linop, transform, FT, output; init_dz=5e-3)
    testfocus(sg, output["Eω"], w0)
end
##
@testset "Gradient pressure: $(typeof(grid)), $(typeof(sg)), pol = $pol, thg = $thg" for sg in (q, xgrid, xygrid),
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

    linop = LinearOps.make_linop(grid, sg, nfunω, thg)
    dens0 = PhysData.density(gas, pressure)
    densityfun(z) = dens0

    responses = (makekerr(grid, thg),)
    normfun = makenorm(grid, sg, nfunω)

    inputs = Fields.GaussGaussField(;λ0, τfwhm, energy, w0, propz=-L)

    Eω, transform, FT = Luna.setup(grid, sg, densityfun, normfun, responses, inputs)
    output = Output.MemoryOutput(0, grid.zmax, 21)
    Luna.run(Eω, grid, linop, transform, FT, output; init_dz=5e-3)
    testfocus(sg, output["Eω"], w0)
end
