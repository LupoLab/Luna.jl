import Test: @test, @testset, @test_throws
import Luna: Capillary, Tools, PhysData

@testset "All" begin
    m = Capillary.MarcatilliMode(125e-6, :He, 0.4, model=:reduced)
    p = Tools.params(300e-6, 10e-15, 2π*PhysData.c/800e-9, m, :He, P=0.4)
    # compare to Pufe
    @test isapprox(p.N, 2.239)
    @test isapprox(p.Lfiss, 1.768)
    @test isapprox(p.λz, 378.8)
    @test isapprox(p.P0/p.Pcr, 0.0398)
end
