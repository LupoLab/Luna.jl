import Test: @test, @testset
import Luna: Ionisation

@test Ionisation.ionrate_ADK(:He, 1e10) ≈ 1.2416371415312408e-18
@test Ionisation.ionrate_ADK(:He, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:HeJ, 2e10) ≈ 1.0772390893742478
@test Ionisation.ionrate_ADK(:Ar , 7e9) ≈ 2.4422306306649472e-08
@test Ionisation.ionrate_ADK(:Ar , 8e9) ≈ 4.494711488416766e-05

@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1e10), 2.4329e-12, rtol=1e-3)
@test isapprox(Ionisation.ionrate_PPT(:He, 800e-9, 1.3e10), 3.0825e-07, rtol=1e-3)