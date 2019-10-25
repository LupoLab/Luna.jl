import Test: @test
import Luna: Capillary
import Luna.PhysData: c

λ = 800e-9
ω = 2π*c/λ
a = 125e-6
m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
@test Capillary.β(m, ω) == Capillary.β(m, λ=λ)
@test isapprox(Capillary.losslength(m, λ=800e-9), 7.0593180702769)
@test isapprox(Capillary.dB_per_m(m, λ=800e-9), 0.6152074146252722)
@test Capillary.dB_per_m(m, λ=800e-9) ≈ 8*Capillary.dB_per_m(Capillary.MarcatilliMode(2a, :He, 1.0, model=:reduced), λ=800e-9)

λ = 1e-9 .* collect(range(70, stop=7300, length=128))
ω = 2π*c./λ
@test all(isfinite.(Capillary.β.(m, ω)))
@test all(isreal.(Capillary.β.(m, ω)))
@test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced), ω)))
@test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 10.0, model=:reduced), ω)))
@test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 50.0, model=:reduced), ω)))
@test all(isfinite.(Capillary.α.(m, ω)))
@test all(isreal.(Capillary.α.(m, ω)))

@test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(a, :He, 0.4, model=:reduced)) - 379) < 1
@test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(75e-6, :He, 5.9, model=:reduced)) - 562) < 1

@test Capillary.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0, model=:reduced)) ≈ 8.42157534886545e-09

# tests based on symbolic results is symbolic_marcatilli.py
m = Capillary.MarcatilliMode(50e-6, :Ar, 2.0, model=:reduced)
@test isapprox(Capillary.β(m, 2π*c/800e-9), 7857864.43728568, rtol=1e-15)
@test isapprox(Capillary.dispersion(m, 1, 2π*c/800e-9), 3.33744310817186e-9, rtol=1e-13)
@test isapprox(Capillary.dispersion(m, 2, 2π*c/800e-9), -1.68385315313058e-29, rtol=1e-7)
@test isapprox(Capillary.dispersion(m, 3, 2π*c/800e-9), 8.43934205839032e-44, rtol=1e-6)
@test isapprox(Capillary.dispersion(m, 4, 2π*c/800e-9), -1.13290569432975e-58, rtol=1e-4)
@test isapprox(Capillary.dispersion(m, 5, 2π*c/800e-9), 2.45045893668943e-73, rtol=1e-3)
@test isapprox(Capillary.zdw(m), 7.288460357934073e-07, rtol=1e-8)