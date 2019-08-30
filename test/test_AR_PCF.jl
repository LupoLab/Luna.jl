import Test: @test
import Luna: AR_PCF
import Luna.PhysData: c

λ = 800e-9
ω = 2π*c/λ
a = 20e-6
gas = :Air
pres = 1.0
wthickness = 700e-9

m = AR_PCF.Matthias(a, gas, pres, wthickness)

@test AR_PCF.β(m, ω) == AR_PCF.β(m, λ=λ)

λ = 1e-9 .* collect(range(70, stop=7300, length=128))
ω = 2π*c./λ[2:end]
@test all(isfinite.(AR_PCF.β(m, ω)))
@test all(isreal.(AR_PCF.β(m, ω)))
@test all(isreal.(AR_PCF.β(AR_PCF.Matthias(a, :He, 1.0, wthickness), ω)))
@test all(isreal.(AR_PCF.β(AR_PCF.Matthias(a, :He, 10.0, wthickness), ω)))
@test all(isreal.(AR_PCF.β(AR_PCF.Matthias(a, :He, 50.0, wthickness), ω)))
@test all(isfinite.(AR_PCF.α(m, ω)))
@test all(isreal.(AR_PCF.α(m, ω)))

λ = 1000e-9
ω = 2π*c/λ
@test 1e4*real(1-m.neff(ω)) ≈ -0.8129573178683458
@test AR_PCF.α(m, ω) ≈ 1.3817812262927658e-7
λ = 600e-9
ω = 2π*c/λ
@test 1e4*real(1-m.neff(ω)) ≈ -2.0142992660843184
@test AR_PCF.α(m, ω) ≈ 1.781240841218271e-8
λ = 400e-9
ω = 2π*c/λ
@test 1e4*real(1-m.neff(ω)) ≈ -2.4379342768443557
@test AR_PCF.α(m, ω) ≈ 8.008516657445107e-9


@test AR_PCF.Aeff(m) ≈ 5.988675803639484e-10