import Test: @test, @test_throws, @testset
import Luna: Refraction

@testset "Exceptions" begin
@test_throws DomainError Refraction.ref_index(:He, 800)
@test_throws DomainError Refraction.ref_index(:SiO2, 800)
@test_throws DomainError Refraction.ref_index(:Hello, 800e-9)
end

@testset "refractive indices" begin
@test Refraction.ref_index(:He, 800e-9) ≈ 1.000031838924767
@test Refraction.ref_index(:He, 800e-9, 10) ≈ 1.0003183436449188
@test Refraction.ref_index(:SiO2, 800e-9) ≈ 1.4533172548587419
@test Refraction.ref_index(:SiO2, 400e-9) ≈ 1.4701161185594052
end

@testset "Function equiavalence" begin
@test Refraction.ref_index_fun(:SiO2)(800e-9) == Refraction.ref_index(:SiO2, 800e-9)
@test Refraction.ref_index_fun(:He)(800e-9) == Refraction.ref_index(:He, 800e-9)
end

@testset "Dispersion" begin
@test Refraction.dispersion(2, :SiO2, 800e-9) ≈ 3.61619983e-26
@test isapprox(Refraction.dispersion(2, :He, 800e-9), 9.34130789e-31, rtol=1e-5)
@test isapprox(Refraction.dispersion(2, :He, 800e-9, 10), 9.33867358e-30, rtol=1e-5)
end

@testset "glasses" begin
for g in Refraction.glass
    @test isreal(Refraction.ref_index(g, 800e-9))
end
end

@testset "gases" begin
for g in Refraction.gas
    @test isreal(Refraction.ref_index(g, 800e-9))
    @test isreal(Refraction.ref_index(g, 200e-9))
    @test isreal(Refraction.ref_index(g, 800e-9, 10))
    @test isreal(Refraction.ref_index(g, 200e-9, 10))
end
end