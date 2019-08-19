import Test: @test, @test_throws, @testset
import Luna: Hankel
import LinearAlgebra: diagm
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import NumericalIntegration: integrate, Trapezoidal

@testset "multiplication" begin
    M = diagm(0 => [1, 2, 3])
    V = 2 .* ones((3, 3, 2))
    out = similar(V)
    Hankel.dot!(out, M, V)
    @test all(out[1, :, :] .== 2)
    @test all(out[2, :, :] .== 4)
    @test all(out[3, :, :] .== 6)
    Hankel.dot!(out, M, V, dim=2)
    @test all(out[:, 1, :] .== 2)
    @test all(out[:, 2, :] .== 4)
    @test all(out[:, 3, :] .== 6)
    @test_throws DomainError Hankel.dot!(out, M, V, dim=3)
end

@testset "transform" begin
    q = Hankel.QDHT(1, 128)
    @test all(isreal.(q.T))
    a = 50
    f(r) = exp(-1//2 * a^2 * r^2)
    fk(k) = 1/a^2 * exp(-k^2/(2*a^2))
    v = f.(q.r)
    for _ = 1:1000
        tmp = q * v
        global vv = q \ tmp
    end
    @test all(v ≈ vv)
    vk = q * v
    vka = fk.(q.k)
    @test all(vka ≈ vk)
    Er = Hankel.integrateR(v.^2, q)
    Ek = Hankel.integrateK(vk.^2, q)
    @test Er ≈ Ek

    v2d = repeat(v, outer=(1, 16))'
    q2d = Hankel.QDHT(1, 128, dim=2)
    v2dk = q2d * v2d
    @test all([all(v2dk[ii, :] ≈ vk) for ii = 1:size(v2dk, 1)])
end