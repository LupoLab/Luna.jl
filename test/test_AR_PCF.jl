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

# these tests are valid for constants silica refractive index
# correct as https://github.com/LupoLab/Luna/issues/74 close
λ = 1000e-9
ω = 2π*c/λ
@test 1e4*real(m.neff(ω)-1) ≈ -1.8352471679983218
@test AR_PCF.α(m, ω) ≈ 1.3832667695368788e-7

@test AR_PCF.Aeff(m) ≈ 5.988675803639484e-10 