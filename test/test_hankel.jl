import Test: @test, @test_throws, @testset
import Luna: Hankel, Maths
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

@testset "Gaussian divergence" begin
    q = Hankel.QDHT(12.7e-3, 512)
    λ = 800e-9
    k = 2π/λ
    kz = @. sqrt(k^2 - q.k^2)
    z = 2 # propagation distance
    prop = @. exp(1im*kz*z)
    w0 = 200e-6 # start at focus
    w1 = w0*sqrt(1+(z*λ/(π*w0^2))^2)
    Ir0 = Maths.gauss(q.r, w0/2)
    Ir1 = Maths.gauss(q.r, w1/2)*(w0/w1)^2 # analytical solution (in paraxial approx)
    Er0 = sqrt.(Ir0)
    Ek0 = q * Er0
    Ek1 = prop .* Ek0
    Er1 = q \ Ek1
    @test isapprox(abs2.(Er1), Ir1, rtol=1e-6)
end
