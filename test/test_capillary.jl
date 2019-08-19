import Test: @test
import Luna: Capillary
import Luna.PhysData: c

λ = 800e-9
ω = 2π*c/λ
a = 125e-6
@test Capillary.β(a, ω) == Capillary.β(a, λ=λ)
@test isapprox(Capillary.losslength(a, λ=800e-9), 7.0593180702769)
@test isapprox(Capillary.dB_per_m(a, λ=800e-9), 0.6152074146252722)
@test Capillary.dB_per_m(a, λ=800e-9) ≈ 8*Capillary.dB_per_m(2*a, λ=800e-9)

λ = 1e-9 .* collect(range(70, stop=8000, length=128))
ω = 2π*c./λ
@test all(isfinite.(Capillary.β(a, ω)))
@test all(isreal.(Capillary.β(a, ω)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=1)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=10)))
@test all(isreal.(Capillary.β(a, ω, gas=:He, pressure=50)))
@test all(isfinite.(Capillary.α(a, ω)))
@test all(isreal.(Capillary.α(a, ω)))

@test abs(1e9*Capillary.zdw(a, gas=:He, pressure=0.4) - 379) < 1
@test abs(1e9*Capillary.zdw(75e-6, gas=:He, pressure=5.9) - 562) < 1

@test Capillary.Aeff(75e-6) ≈ 8.42157534886545e-09