import Test: @test, @testset, @test_throws
import Luna: Maths

@testset "Maths" begin
@testset "Derivatives" begin
    f(x) = @. 4x^3 + 3x^2 + 2x + 1

    @test isapprox(Maths.derivative(f, 1, 1), 12+6+2)
    @test isapprox(Maths.derivative(f, 1, 2), 24+6)
    @test isapprox(Maths.derivative(f, 1, 3), 24)

    e(x) = @. exp(x)

    x = [1, 2, 3, 4, 5]
    @test isapprox(Maths.derivative(e, 1, 5), exp(1), rtol=1e-6)
    @test isapprox(Maths.derivative.(e, x, 5), exp.(x), rtol=1e-6)

    @test isapprox(Maths.derivative(x -> exp.(2x), 1, 1), 2*exp(2))
    @test isapprox(Maths.derivative(x -> exp.(2x), 1, 2), 4*exp(2))
    @test isapprox(Maths.derivative(x -> exp.(-x.^2), 0, 1), 0, atol=1e-14)
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

    Etc = Maths.gauss(t, fwhm=4).*exp.(1im*4*t)
    to, Etco = Maths.oversample(t, Etc, factor=4)
    @test 4*size(Etc)[1] == size(Etco)[1]
    @test all(isapprox.(Etco[1:4:end], Etc, rtol=1e-6))
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

@testset "Spline" begin
    x = range(0.0, 2π, length=100)
    y = sin.(x)
    spl = Maths.CSpline(x, y)
    fslow(x0) = x0 <= spl.x[1] ? 2 :
                x0 >= spl.x[end] ? length(spl.x) :
                findfirst(x -> x>x0, spl.x)
    @test all(abs.(spl.(x) .- y) .< 5e-18)
    x2 = range(0.0, 2π, length=300)
    idcs = spl.ifun.(x2)
    idcs_slow = fslow.(x2)
    @test all(idcs .== idcs_slow)
    @test maximum(spl.(x2) - sin.(x2)) < 5e-8
    @test abs(Maths.derivative(spl, 1.3, 1) - cos(1.3)) < 1.7e-7
    @test maximum(cos.(x2) - Maths.derivative.(spl, x2, 1)) < 2.1e-6
end

@testset "randgauss" begin
    import Statistics: std, mean
    x = Maths.randgauss(1, 0.5, 1000000, seed=1234)
    @test isapprox(std(x), 0.5, rtol=1e-3)
    @test isapprox(mean(x), 1, rtol=1e-3)
    x = Maths.randgauss(10, 0.1, 1000000, seed=1234)
    @test isapprox(std(x), 0.1, rtol=1e-3)
    @test isapprox(mean(x), 10, rtol=1e-3)
    x = Maths.randgauss(-1, 0.5, 1000000, seed=1234)
    @test isapprox(std(x), 0.5, rtol=1e-3)
    @test isapprox(mean(x), -1, rtol=1e-3)
    x = Maths.randgauss(1, 0.5, (1000, 1000), seed=1234)
    @test isapprox(std(x), 0.5, rtol=1e-3)
    @test isapprox(mean(x), 1, rtol=1e-3)
end

end