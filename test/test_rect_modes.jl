import Test: @test
using Luna.RectModes
import Luna.Modes
import Luna.PhysData: c

a = 50e-6
b = 12.5e-6

m = RectMode(a, b, :Ar, 0.0, :Ag)

# these were checked with FEM using JCMwave
@test isapprox(Aeff(m), 1.111111111111111e-09)
@test isapprox(Modes.dB_per_m(m, λ=210e-9), 3.908368640803528, rtol=1e-4)
@test isapprox(Modes.dB_per_m(m, λ=800e-9), 0.44317059469197245, rtol=1e-2)
@test isapprox(β(m, λ=800e-9), 7852915.234685494)
@test isapprox(β(m, λ=210e-9), 29919650.305427298)

@test abs(1e9*Modes.zdw(RectMode(a, b, :Ar, 5.0, :Ag)) - 562.2) < 1
