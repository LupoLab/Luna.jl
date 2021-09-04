import Test: @test, @testset, @test_broken
import Luna: Raman, Maths, Nonlinear, PhysData, Grid
import QuadGK: quadgk
import FFTW

import Luna: set_fftw_mode
set_fftw_mode(:estimate)

grid = Grid.RealGrid(1.0, 800e-9, (200e-9, 2000e-9), 40e-12)
ρ = PhysData.density(:N2, 5.0)
h = Raman.raman_response(grid.to, :N2)
T = collect(range(-5e-12, stop=20e-12, length=2^16))
dt = T[2] - T[1]

# test response function is zero for 0 or negative time
@test h(0.0, ρ) == 0.0
@test all(h.(T[T .<= 0.0], ρ) .== 0.0)

hsdo = Raman.RamanRespNormedSingleDampedOscillator(1.0, 2π*100e12, 100e-15)
# test integral of normalised response function is 1
@test isapprox(quadgk(x -> hsdo(x, 1.0), -1e-12, 2e-12)[1], 1.0, rtol=1e-8)

# test frequencies of max gain
hv = Raman.raman_response(grid.to, :N2, rotation=false)
RP! = Nonlinear.RamanPolarField(grid.to, hv)
E = Maths.gauss.(grid.to, fwhm=20e-15)
P = similar(E)
RP!(P, E, ρ) # get polarisation induced by E
# test maximum of frequency response occurs at correct frequency
# note hω grid is twice as long at grid.to, so we need a new freq. grid
ωR = 2π .* collect(0:(length(grid.to) - 1)) ./ ((grid.to[2] - grid.to[1]) * length(grid.to) * 2)
@test isapprox(ωR[argmin(imag.(RP!.hω))], PhysData.raman_parameters(:N2).Ωv, rtol=1e-4)

# same, but for envelopes
grid = Grid.EnvGrid(1.0, 800e-9, (200e-9, 2000e-9), 40e-12)
hv = Raman.raman_response(grid.to, :N2, rotation=false)
RP! = Nonlinear.RamanPolarEnv(grid.to, hv)
E = complex.(Maths.gauss.(grid.to, fwhm=20e-15))
P = similar(E)
RP!(P, E, ρ) # get polarisation induced by E
# test maximum of frequency response occurs at correct frequency
# note hω grid is twice as long at grid.to, so we need a new freq. grid (we only make first half of it)
ωR = 2π .* collect(0:(length(grid.to) - 1)) ./ ((grid.to[2] - grid.to[1]) * length(grid.to) * 2)
@test isapprox(ωR[argmin(imag.(RP!.hω))], PhysData.raman_parameters(:N2).Ωv, rtol=1e-4)

# same, but for hydrogen
grid = Grid.EnvGrid(1.0, 800e-9, (200e-9, 2000e-9), 40e-12)
hv = Raman.raman_response(grid.to, :H2, rotation=false)
RP! = Nonlinear.RamanPolarEnv(grid.to, hv)
E = complex.(Maths.gauss.(grid.to, fwhm=20e-15))
P = similar(E)
ρ = PhysData.density(:H2, 5.0)
RP!(P, E, ρ) # get polarisation induced by E
# test maximum of frequency response occurs at correct frequency
# note hω grid is twice as long at grid.to, so we need a new freq. grid (we only make first half of it)
ωR = 2π .* collect(0:(length(grid.to) - 1)) ./ ((grid.to[2] - grid.to[1]) * length(grid.to) * 2)
@test isapprox(ωR[argmin(imag.(RP!.hω))], PhysData.raman_parameters(:H2).Ωv, rtol=1e-4)

# same, but for hydrogen rotation
grid = Grid.EnvGrid(1.0, 800e-9, (200e-9, 2000e-9), 40e-12)
hv = Raman.raman_response(grid.to, :H2, vibration=false)
RP! = Nonlinear.RamanPolarEnv(grid.to, hv)
E = complex.(Maths.gauss.(grid.to, fwhm=20e-15))
P = similar(E)
ρ = PhysData.density(:H2, 5.0)
RP!(P, E, ρ) # get polarisation induced by E
# test maximum of frequency response occurs at correct frequency
# note hω grid is twice as long at grid.to, so we need a new freq. grid (we only make first half of it)
ωR = 2π .* collect(0:(length(grid.to) - 1)) ./ ((grid.to[2] - grid.to[1]) * length(grid.to) * 2)
@test isapprox(ωR[argmin(imag.(RP!.hω))], 2π*17.45e12, rtol=1e-4)
