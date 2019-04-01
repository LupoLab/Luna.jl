import Test: @test, @testset, @test_throws
import Luna: Maths

@testset "Derivatives" begin
f(x) = @. 4x^3 + 3x^2 + 2x + 1

@test Maths.derivative(f, 1, 1) == 12+6+2
@test Maths.derivative(f, 1, 2) == 24+6
@test Maths.derivative(f, 1, 3) == 24

e(x) = @. exp(x)

x = [1, 2, 3, 4, 5]
@test Maths.derivative(e, 1, 5) == exp(1)
@test Maths.derivative.(e, x, 5) == exp.(x)

@test Maths.derivative(x -> exp.(2x), 1, 1) == 2*exp(2)
@test Maths.derivative(x -> exp.(2x), 1, 2) == 4*exp(2)
@test Maths.derivative(x -> exp.(-x.^2), 0, 1) == 0
end

@testset "Moments" begin
x = collect(range(-10, stop=10, length=513))
y = Maths.gauss(x, 1, x0=1)
@test Maths.moment(x, y) ≈ 1
@test Maths.moment(x, y, 2) ≈ 2
@test Maths.rms_width(x, y) ≈ 1

x0 = [-2.5, -1.0, 0.0, 1.0, 2.5]
σ = [0.1, 0.5, 1.0, 1.5, 1.5]
y = zeros(length(x), length(x0))
for ii = 1:length(x0)
    y[:, ii] = Maths.gauss(x, σ[ii], x0=x0[ii])
end
@test_throws DomainError Maths.moment(x, y, dim=2)
@test all(isapprox.(transpose(Maths.moment(x, y, dim=1)), x0, atol=1e-5))
@test all(isapprox.(transpose(Maths.rms_width(x, y, dim=1)), σ, atol=1e-5))

yt = transpose(y)
@test_throws DomainError Maths.moment(x, yt, dim=1)
xm = Maths.moment(x, yt, dim=2)
@test all(isapprox.(xm, x0, atol=1e-5))
end
