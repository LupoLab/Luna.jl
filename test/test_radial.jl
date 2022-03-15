import Test: @test, @testset

@testset "Radial propagation" begin
import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, Stats, LinearOps, Plotting
import Luna.PhysData: wlfreq
import FFTW
import Hankel
import LinearAlgebra: norm

gas = :Ar
pres = 1.2
τ = 20e-15
λ0 = 800e-9
w0 = 40e-6
energy = 1e-12
L = 0.6
R = 4e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, N, dim=2)

energyfun, energyfun_ω = Fields.energyfuncs(grid, q)

function prop(E, z)
    Eω = FFTW.rfft(E, 1)
    Eωk = q * Eω
    kzsq = @. (grid.ω/PhysData.c)^2 - (q.k^2)'
    kzsq[kzsq .< 0] .= 0
    kz = sqrt.(kzsq)
    @. Eωk *= exp(-1im * z * (kz - grid.ω/PhysData.c))
    Eω = q \ Eωk
    E = FFTW.irfft(Eω, length(grid.t), 1)
    return E
end

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0
ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.ionrate_fun!_PPTcached(gas, λ0)
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.3)

Eω, transform, FT = Luna.setup(grid, q, densityfun, normfun, responses, inputs)
output = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, output)

Erout = (q \ output.data["Eω"])
Iωr = abs2.(Erout)

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

Iλ0 = Iωr[ω0idx, :, :]
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L/2*λ0/(π*w0^2))^2)
Iλ0_analytic = Maths.gauss.(q.r, w1/2)*(w0/w1)^2 # analytical solution (in paraxial approx)
Ir = Maths.normbymax(Iλ0[:, end])
Ira = Maths.normbymax(Iλ0_analytic)
@test maximum(abs.(Ir .- Ira)/norm(Ir)) < 1.5e-4
end
