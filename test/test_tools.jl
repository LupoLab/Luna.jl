import Test: @test, @testset, @test_throws
import Luna: Capillary, Tools, PhysData

@testset "Params" begin
m = Capillary.MarcatiliMode(125e-6, :HeB, 0.4, model=:reduced)
p = Tools.params(300e-6, 10e-15, 800e-9, m, :HeB, P=0.4)
# compare to Pufe
@test isapprox(p.N, 2.239, rtol=1e-3)
@test isapprox(p.Lfiss, 1.768, rtol=1e-3)
@test isapprox(p.zdw, 378.8e-9, rtol=1e-2)
@test isapprox(p.P0/p.Pcr, 0.0398, rtol=2e-2)
# test backup zdw method
p = Tools.capillary_params(6e-9, 20e-15, 800e-9, 14e-6, :Kr, P=15.0)
@test isapprox(p.zdw, 7.693023014958748e-7, rtol=1e-7)
end

@testset "RDW phasematching" begin
a = 125e-6
gas = :He
pressure = 1
λ0 = 800e-9

λRDW = Tools.λRDW(a, gas, pressure, λ0)
@test λRDW - 188e-9 < 1e-9
@test Tools.pressureRDW(a, gas, λRDW, λ0) ≈ pressure
end
