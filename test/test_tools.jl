import Test: @test, @testset, @test_throws
import Luna: Capillary, Tools, PhysData

@testset "Params" begin
m = Capillary.MarcatilliMode(125e-6, :He, 0.4, model=:reduced)
p = Tools.params(300e-6, 10e-15, 800e-9, m, :He, P=0.4)
# compare to Pufe
@test isapprox(p.N, 2.239, rtol=1e-3)
@test isapprox(p.Lfiss, 1.768, rtol=1e-3)
@test isapprox(p.zdw, 378.8e-9, rtol=1e-2)
@test isapprox(p.P0/p.Pcr, 0.0398, rtol=2e-2)
end

@testset "RDW phasematching" begin
a = 125e-6
gas = :HeJ
pressure = 1
λ0 = 800e-9

λRDW = Tools.λRDW(a, gas, pressure, λ0)
@test λRDW - 188e-9 < 1e-9
@test Tools.pressureRDW(a, gas, λRDW, λ0) ≈ pressure
end
