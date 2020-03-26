import Test: @test, @testset
import Luna: Modes, Capillary, Grid
import Luna.PhysData: c, roomtemp, ref_index_fun
import Luna.PhysData: wlfreq

@testset "Capillary" begin
@testset "loss" begin
    λ = 800e-9
    ω = wlfreq(λ)
    a = 125e-6
    m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
    @test isapprox(Capillary.losslength(m, ω), 7.0593180702769)
    @test isapprox(Capillary.dB_per_m(m, ω), 0.6152074146252722)
    @test Capillary.dB_per_m(m, ω) ≈ 8*Capillary.dB_per_m(Capillary.MarcatilliMode(2a, :He, 1.0, model=:reduced), ω)
end

@testset "β, α" begin
    a = 125e-6
    m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
    λ = 1e-9 .* collect(range(70, stop=7300, length=128))
    ω = wlfreq.(λ)
    @test all(isfinite.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 10.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 50.0, model=:reduced), ω)))
    @test all(isfinite.(Capillary.α.(m, ω)))
    @test all(isreal.(Capillary.α.(m, ω)))
end

@testset "ZDW/Aeff" begin
    @test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(125e-6, :He, 0.4, model=:reduced)) - 379) < 1
    @test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(75e-6, :He, 5.9, model=:reduced)) - 562) < 1
    @test Capillary.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0, model=:reduced)) ≈ 8.42157534886545e-09
end


@testset "dispersion" begin
    # tests based on symbolic results in symbolic_marcatilli.py
    m = Capillary.MarcatilliMode(50e-6, :Ar, 2.0, model=:reduced)
    ω = wlfreq(800e-9)
    @test isapprox(Capillary.β(m, ω), 7857864.43728568, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744310817186e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68385315313058e-29, rtol=1e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.43934205839032e-44, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13290569432975e-58, rtol=1e-4)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.45045893668943e-73, rtol=1e-3)
    @test isapprox(Capillary.zdw(m), 7.288460357934073e-07, rtol=1e-8)

    rfg = ref_index_fun(:Ar, 2.0, roomtemp)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> 1.45
    m = Capillary.MarcatilliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857863.48006503, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744262246032e-9, rtol=1e-14)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68286833115421e-29, rtol=1e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.43469324575104e-44, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13224995913863e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.44896534856203e-73, rtol=1e-4)
    @test isapprox(Capillary.zdw(m), 7.288488367356287e-07, rtol=1e-8)
    @test isapprox(Capillary.α(m, ω), 2.21505888048642, rtol=1e-14)

    rfg = ref_index_fun(:Ar, 2.0, roomtemp)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> 0.036759+im*5.5698
    m = Capillary.MarcatilliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857861.48263403, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744432288277e-9, rtol=1e-14)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.90000438857902e-29, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.80439075111799e-44, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.21093130760499e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.64991119379223e-73, rtol=1e-3)
    @test isapprox(Capillary.zdw(m), 7.224394530976951e-07, rtol=1e-8)
    @test isapprox(Capillary.α(m, ω), 0.0290115788131925, rtol=1e-14)
    @test Capillary.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0)) ≈ 8.42157534886545e-09
end

@testset "delegation" begin
    m = Modes.@delegated(Capillary.MarcatilliMode(75e-6, :He, 5.9))
    @test Modes.Aeff(m) ≈ 8.42157534886545e-09
    @test abs(1e9*Modes.zdw(m) - 562) < 1
    m2 = Modes.@delegated(Capillary.MarcatilliMode(75e-6, :He, 1.0), α=(ω)->0.2)
    @test Modes.α(m2, 2e15) == 0.2
    @test Modes.α(m2, wlfreq(800e-9)) == 0.2

    cm = Capillary.MarcatilliMode(75e-6, :He, 5.9)
    dm = Modes.@delegated(cm)
    @test Modes.Aeff(dm) ≈ 8.42157534886545e-09

    # fully delegated test
    m3 = Modes.@arbitrary(α=(ω)->0.2, β=(ω)->0.2,
                            dimlimits=()->(:polar, (0.0, 0.0), (m.a, 2π)), field=()->nothing)
    @test Modes.α(m3, 2e15) == 0.2
end
end