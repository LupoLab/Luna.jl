import Test: @test, @testset, @test_broken
import SpecialFunctions: besselj
import Cubature: hcubature
import LinearAlgebra: dot, norm
import Luna: Modes, Capillary, Grid
import Luna.PhysData: c, roomtemp, ref_index_fun, ε_0, μ_0
import Luna.PhysData: wlfreq

@testset "Capillary" begin
@testset "loss" begin
    λ = 800e-9
    ω = wlfreq(λ)
    a = 125e-6
    m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
    @test isapprox(Capillary.losslength(m, ω), 7.0593180702769, rtol=1e-5)
    @test isapprox(Capillary.dB_per_m(m, ω), 0.6152074146252722, rtol=1e-5)
    @test Capillary.dB_per_m(m, ω) ≈ 8*Capillary.dB_per_m(Capillary.MarcatilliMode(2a, :He, 1.0, model=:reduced), ω)
end

@testset "normalisation" begin
    # Copied definitions from Modes.jl to force manual calculation of Aeff and N
    function Aeff(m; z=0)
        em = Modes.absE(m, z=z)
        dl = Modes.dimlimits(m, z=z)
        # Numerator
        function Aeff_num(xs)
            e = em(xs)
            dl[1] == :polar ? xs[1]*e^2 : e^2
        end
        val, err = hcubature(Aeff_num, dl[2], dl[3])
        num = val^2
        # Denominator
        function Aeff_den(xs)
            e = em(xs)
            dl[1] == :polar ? xs[1]*e^4 : e^4
        end
        den, err = hcubature(Aeff_den, dl[2], dl[3])
        return num / den
    end
    function N(m; z=0)
        f = Modes.field(m, z=z)
        dl = Modes.dimlimits(m, z=z)
        function Nfunc(xs)
            E = f(xs)
            ret = sqrt(ε_0/μ_0)*dot(E, E)
            dl[1] == :polar ? xs[1]*ret : ret
        end
        val, err = hcubature(Nfunc, dl[2], dl[3])
        0.5*abs(val)
    end

    a = 125e-6
    for n=1:10
        m = Capillary.MarcatilliMode(a, :He, 1.0, n=n)
        @test Modes.N(m) ≈ N(m)
        @test Modes.Aeff(m) ≈ Aeff(m)
    end
    m = Capillary.MarcatilliMode(a, :He, 1.0, n=0, kind=:TE)
    @test Modes.N(m) ≈ N(m)
    @test Modes.Aeff(m) ≈ Aeff(m)
    m = Capillary.MarcatilliMode(a, :He, 1.0, n=0, kind=:TM)
    @test Modes.N(m) ≈ N(m)
    @test Modes.Aeff(m) ≈ Aeff(m)
    
    a0 = a
    aL = a/2
    L = 1
    afun = let a0=a0, aL=aL, L=L
        afun(z) = a0 + (aL-a0)*z/L
    end
    m = Capillary.MarcatilliMode(afun, :He, 1, loss=false, model=:full)
    @test Modes.N(m) ≈ N(m)
    @test Modes.N(m, z=L/2) ≈ N(m, z=L/2)
    @test Modes.N(m, z=L) ≈ N(m, z=L)
    @test Modes.Aeff(m) ≈ Aeff(m)
    @test Modes.Aeff(m, z=L/2) ≈ Aeff(m, z=L/2)
    @test Modes.Aeff(m, z=L) ≈ Aeff(m, z=L)
end


@testset "β, α" begin
    a = 125e-6
    m = Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced)
    λ = 1e-9 .* collect(range(70, stop=7300, length=128))
    ω = wlfreq.(λ)
    @test all(isfinite.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 1.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 10.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatilliMode(a, :He, 50.0, model=:reduced), ω)))
    @test all(isfinite.(Capillary.α.(m, ω)))
    @test all(isreal.(Capillary.α.(m, ω)))
end

@testset "ZDW/Aeff" begin
    @test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(125e-6, :He, 0.4, model=:reduced)) - 379) < 1
    @test abs(1e9*Capillary.zdw(Capillary.MarcatilliMode(75e-6, :He, 5.9, model=:reduced)) - 562) < 1
    @test Capillary.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0, model=:reduced)) ≈ 8.42157534886545e-09
end


@testset "dispersion" begin
    # tests based on symbolic results in symbolic_marcatilli.py
    m = Capillary.MarcatilliMode(50e-6, :Ar, 2.0, model=:reduced)
    ω = wlfreq(800e-9)
    @test isapprox(Capillary.β(m, ω), 7857864.43728568, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744310817186e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68385315313058e-29, rtol=5e-8)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.43934205839032e-44, rtol=5e-7)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13290569432975e-58, rtol=5e-6)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.45045893668943e-73, rtol=5e-5)
    @test isapprox(Capillary.zdw(m), 7.288460357934073e-07, rtol=5e-9)

    rfg = ref_index_fun(:Ar, 2.0, roomtemp)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> 1.45
    m = Capillary.MarcatilliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857863.48006503, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744262246032e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68286833115421e-29, rtol=1e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.43469324575104e-44, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13224995913863e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.44896534856203e-73, rtol=1e-4)
    @test isapprox(Capillary.zdw(m), 7.288488367356287e-07, rtol=1e-8)
    @test isapprox(Capillary.α(m, ω), 2.21505888048642, rtol=1e-14)

    rfg = ref_index_fun(:Ar, 2.0, roomtemp)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> 0.036759+im*5.5698
    m = Capillary.MarcatilliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857861.48263403, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744432288277e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.90000438857902e-29, rtol=5e-8)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.80439075111799e-44, rtol=5e-7)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.21093130760499e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.64991119379223e-73, rtol=5e-5)
    @test isapprox(Capillary.zdw(m), 7.224394530976951e-07, rtol=5e-9)
    @test isapprox(Capillary.α(m, ω), 0.0290115788131925, rtol=1e-14)
    @test Capillary.Aeff(Capillary.MarcatilliMode(75e-6, :He, 1.0)) ≈ 8.42157534886545e-09
end

@testset "to_space" begin
    ms = (Capillary.MarcatilliMode(125e-6, :He, 1.0),
          Capillary.MarcatilliMode(125e-6, :He, 1.0, m=2, ϕ=π/2))
    components = :xy
    xs = (10e-6, π/7)
    ts = Modes.ToSpace(ms, components=components)
    Emω = Array{ComplexF64,2}(undef, 8192, 2)
    fill!(Emω, 0.3+0.5im)
    Erω1 = Modes.to_space(Emω, xs, ts)
    Erω! = copy(Erω1)
    Modes.to_space!(Erω!, Emω, xs, ts)
    Erω2 = Modes.to_space(Emω, xs, ms, components=components)
    @test all(Erω2 .== Erω!)
    @test all(Erω1 .== Erω!)
end

end