import Test: @test, @testset

@testset "Full 3D propagation" begin
import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Hankel
import Logging
import FFTW
import Luna.PhysData: wlfreq
import LinearAlgebra: norm

gas = :Ar
pres = 1

τ = 30e-15
λ0 = 800e-9

w0 = 500e-6
energy = 1e-9
L = 2

R = 4e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
xygrid = Grid.FreeGrid(R, N)

x = xygrid.x
y = xygrid.y
energyfun, energyfunω = NonlinearRHS.energy_free(grid, xygrid)

function gausspulse(t)
    It = Maths.gauss(t, fwhm=τ) .* Maths.gauss.(xygrid.r, w0/2)
    ω0 = 2π*PhysData.c/λ0
    Et = @. sqrt(It)*cos(ω0*t)
end

dens0 = PhysData.density(gas, pres)
densityfun = let dens0=dens0
    z -> dens0
end

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))

in1 = (func=gausspulse, energy=energy)
inputs = (in1, )

Eω, transform, FT = Luna.setup(grid, xygrid, energyfun, densityfun, normfun, responses, inputs)

output = Output.MemoryOutput(0, grid.zmax, 21, (length(grid.ω), N, N))

Luna.run(Eω, grid, linop, transform, FT, output, max_dz=Inf, init_dz=1e-1)

Eout = output.data["Eω"] # (ω, ky, kx, z)

ω0idx = argmin(abs.(grid.ω .- wlfreq(λ0)))
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L*λ0/(π*w0^2))^2)
Iω0_analytic = Maths.gauss(xygrid.x, w1/2) # analytical solution (in paraxial approx)

Eω0yx = FFTW.ifft(Eout[ω0idx, :, :, end], (1, 2))
Iω0yx = abs2.(Eω0yx)
Iω0y = Maths.normbymax(dropdims(sum(Iω0yx, dims=2), dims=2))
Iω0x = Maths.normbymax(dropdims(sum(Iω0yx, dims=1), dims=1))

@test maximum(abs.(Iω0x .- Iω0_analytic)/norm(Iω0x)) < 5e-5
@test maximum(abs.(Iω0y .- Iω0_analytic)/norm(Iω0y)) < 5e-5

end