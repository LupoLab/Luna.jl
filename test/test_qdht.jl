import Test: @test, @test_throws, @testset
import Luna: Hankel
import LinearAlgebra: diagm
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import Cubature: hquadrature

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

q = Hankel.QDHT(1, 256)
u11 = besselj_zero(0, 1)
f(r) = besselj.(0, u11.*r)
v = f(q.r)
vv = copy(v)
for _ = 1:1000
    tmp = q * v
    vv = q \ v
end
@test all(v ≈ vv)
vk = q \ v
vki = zero(q.k)
for (idx, ki) in enumerate(q.k)
    vki[idx] = hquadrature(r -> r*f(r)*besselj(0, ki*r), 0, q.R)[1]
    println(idx)
end
# import PyPlot: pygui, plt
# pygui(true)