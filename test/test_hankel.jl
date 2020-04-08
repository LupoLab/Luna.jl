import Test: @test, @test_throws, @testset
import Luna: Hankel, Maths
import LinearAlgebra: diagm, mul!
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import NumericalIntegration: integrate, Trapezoidal
import HCubature: hquadrature

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
    R = 4e-2
    N = 256
    w0 = 1e-3
    a = 2/w0
    q = Hankel.QDHT(R, N)
    @test all(isreal.(q.T))
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
    fki(k) = hquadrature(r -> r.*f(r).*besselj(0, k.*r), 0, R)[1]
    @test all(vka ≈ vk)
    @test fki(q.k[1]) ≈ vk[1] # doing all of them takes too long
    @test fki(q.k[128]) ≈ vk[128]
    Er = Hankel.integrateR(v.^2, q)
    Ek = Hankel.integrateK(vk.^2, q)
    @test Er ≈ Ek
    Er_c = hquadrature(r -> r.*f(r).^2, 0, 1)
    @test Er_c[1] ≈ Er
    # Test that in-place transform works
    vk2 = similar(vk)
    vk3 = copy(v)
    mul!(vk2, q, v)
    mul!(vk3, q, vk3)
    @test all(vk2 ≈ vk)
    @test all(vk3 ≈ vk)

    v2d = repeat(v, outer=(1, 16))'
    q2d = Hankel.QDHT(R, N, dim=2)
    v2dk = q2d * v2d
    @test all([all(v2dk[ii, :] ≈ vk) for ii = 1:size(v2dk, 1)])

    f0 = f(0)
    f0q = Hankel.onaxis(vk, q)
    @test f0 ≈ f0q

    f2(r) = sinc(100*r)^2
    v = f2.(q.r);
    vk = q * v
    f0 = f2(0)
    f0q = Hankel.onaxis(vk, q)
    @test f0 ≈ f0q
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
    energy = π/2*w0^2 # analytical energy for gaussian beam
    @test energy ≈ 2π*Hankel.integrateR(Ir0, q)
    @test energy ≈ 2π*Hankel.integrateR(abs2.(Er1), q)
    @test energy ≈ 2π*Hankel.integrateK(abs2.(Ek1), q)
end
