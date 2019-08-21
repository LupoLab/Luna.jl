import Test: @test
import Luna: Capillary, AbstractModes
import Luna.PhysData: c

λ = 800e-9
ω = 2π*c/λ
a = 125e-6
m = Capillary.MarcatilliMode(a, :He, 1.0)
@test AbstractModes.β(m, ω) == AbstractModes.β(m, λ=λ)
@test isapprox(AbstractModes.losslength(m, λ=800e-9), 7.0593180702769)
@test isapprox(AbstractModes.dB_per_m(m, λ=800e-9), 0.6152074146252722)
@test AbstractModes.dB_per_m(m, λ=800e-9) ≈ 8*AbstractModes.dB_per_m(Capillary.MarcatilliMode(2a, :He, 1.0), λ=800e-9)

λ = 1e-9 .* collect(range(70, stop=7300, length=128))
ω = 2π*c./λ
@test all(isfinite.(AbstractModes.β(m, ω)))
@test all(isreal.(AbstractModes.β(m, ω)))
@test all(isreal.(AbstractModes.β(Capillary.MarcatilliMode(a, :He, 1.0), ω)))
@test all(isreal.(AbstractModes.β(Capillary.MarcatilliMode(a, :He, 10.0), ω)))
@test all(isreal.(AbstractModes.β(Capillary.MarcatilliMode(a, :He, 50.0), ω)))
@test all(isfinite.(AbstractModes.α(m, ω)))
@test all(isreal.(AbstractModes.α(m, ω)))

@test abs(1e9*AbstractModes.zdw(Capillary.MarcatilliMode(a, :He, 0.4)) - 379) < 1
@test abs(1e9*AbstractModes.zdw(Capillary.MarcatilliMode(75e-6, :He, 5.9)) - 562) < 1

@test AbstractModes.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0)) ≈ 8.42157534886545e-09