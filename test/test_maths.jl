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

@testset "Fourier" begin
    t = collect(range(-10, stop=10, length=513))
    Et = Maths.gauss(t, fwhm=4).*cos.(4*t)
    EtA = Maths.hilbert(Et)
    @test maximum(abs.(EtA)) ≈ 1

    t = collect(range(-10, stop=10, length=512))
    Et = Maths.gauss(t, fwhm=4).*cos.(4*t)
    to, Eto = Maths.oversample(t, Et, factor=4)
    @test 4*size(Et)[1] == size(Eto)[1]
    @test all(isapprox.(Eto[1:4:end], Et, rtol=1e-6))
end

@testset "integration" begin
    x = collect(range(0, stop=8π, length=2^14)).*1e-15
    y = cos.(x.*1e15)
    yi = sin.(x.*1e15)./1e15
    yic = similar(yi)
    yic2 = copy(y)
    Maths.cumtrapz!(yic, y, x)
    Maths.cumtrapz!(yic2, x[2]-x[1])
    @test isapprox(yi, yic, rtol=1e-6)
    @test isapprox(yi, yic2, rtol=1e-6)

    ω = [1e15, 2e15]'
    y = cos.(x.*ω)
    yi =  sin.(x.*ω)./ω
    yic = similar(y)
    yic2 = copy(y)
    Maths.cumtrapz!(yic, y, x)
    Maths.cumtrapz!(yic2, x[2]-x[1])
    @test isapprox(yi, yic, rtol=1e-6)
    @test isapprox(yi, yic2, rtol=1e-6)
end

@testset "series" begin
    sumfunc(x, n) = x + 1/factorial(n)
    e, succ, steps = Maths.aitken_accelerate(sumfunc, 0, rtol=1e-10)
    e2, succ, steps = Maths.converge_series(sumfunc, 0, rtol=1e-10)
    @test isapprox(e, exp(1), rtol=1e-10)
    @test isapprox(e, e2, rtol=1e-10)
    sumfunc(x, n) = x + 1/2^n
    o, succ, steps = Maths.aitken_accelerate(sumfunc, 0, n0=1, rtol=1e-10)
    @test isapprox(o, 1, rtol=1e-10)
    serfunc(x, n) = (x + 2/x)/2
    sqrt2, succ, steps = Maths.aitken_accelerate(serfunc, 1, rtol=1e-10)
    @test isapprox(sqrt2, sqrt(2), rtol=1e-10)
end

@testset "windows" begin
    x = collect(range(-10, stop=10, length=2048))
    pl = Maths.planck_taper(x, -5, -4, 7, 8)
    @test all(pl[x .< -5] .== 0)
    @test all(pl[-4 .< x .< 7] .== 1)
    @test all(pl[8 .< x] .== 0)
end
