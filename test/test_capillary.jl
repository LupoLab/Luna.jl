import PhysicalConstants
import Unitful
import Test: @test
import Luna: Capillary

const c = Unitful.ustrip(PhysicalConstants.CODATA2014.c)

λ = 800e-9
ω = 2π*c/λ
a = 125e-6
@test Capillary.β(a, ω) == Capillary.β(a, λ=λ)
@test isapprox(Capillary.losslength(a, λ=800e-9), 7.0593180702769)
@test isapprox(Capillary.dB_per_m(a, λ=800e-9), 0.6152074146252722)

λ = 1e-9 .* collect(range(150, stop=2000, length=128))
ω = ω = 2π*c./λ
@test all(isfinite.(Capillary.β(a, ω)))
@test all(isreal.(Capillary.β(a, ω)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=1)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=10)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=50)))

@test abs(1e9*Capillary.zdw(a, gas=:He, pressure=0.4) - 379) < 1

println(Capillary.transmission(125e-6, 3, λ=800e-9))