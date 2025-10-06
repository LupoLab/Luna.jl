import Test: @test, @testset, @test_broken
import SpecialFunctions: besselj
import Luna.Modes: hcubature
import LinearAlgebra: dot, norm
import Luna: Modes, Capillary, Grid
import Luna.PhysData: c, roomtemp, ref_index_fun, ε_0, μ_0
import Luna.PhysData: wlfreq

@testset "loss" begin
    λ = 800e-9
    ω = wlfreq(λ)
    a = 125e-6
    m = Capillary.MarcatiliMode(a, :HeB, 1.0, model=:reduced)
    @test isapprox(Capillary.losslength(m, ω), 7.0593180702769, rtol=1e-5)
    @test isapprox(Capillary.dB_per_m(m, ω), 0.6152074146252722, rtol=1e-5)
    @test Capillary.dB_per_m(m, ω) ≈ 8*Capillary.dB_per_m(Capillary.MarcatiliMode(2a, :HeB, 1.0, model=:reduced), ω)
end

@testset "normalisation" begin
    # Copied definitions from Modes.jl to force manual calculation of Aeff and N
    # these also return the integration error to make comparisons easier
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
        numerr = abs(2num*err/val)
        # Denominator
        function Aeff_den(xs)
            e = em(xs)
            dl[1] == :polar ? xs[1]*e^4 : e^4
        end
        den, denerr = hcubature(Aeff_den, dl[2], dl[3])
        return num / den, num/den*sqrt((numerr/num)^2 + (denerr/den)^2)
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
        0.5*abs(val), 0.5*err
    end

    a = 125e-6
    @testset "n = $n" for n = 1:8
        @testset "m = $m" for m = 1:8
            # With the exception of HE65 specifically, all of these also pass with rtol=1e-20
            mode = Capillary.MarcatiliMode(a, :HeB, 1.0, n=n, m=m)
            Ni, Nerr = N(mode)
            @test isapprox(Modes.N(mode), Ni, atol=Nerr, rtol=1e-7)
            aeff, aefferr = Aeff(mode)
            @test isapprox(Modes.Aeff(mode), aeff, rtol=1e-7, atol=aefferr)
        end
    end
    m = Capillary.MarcatiliMode(a, :HeB, 1.0, n=0, kind=:TE)
    @test Modes.N(m) ≈ N(m)[1]
    @test Modes.Aeff(m) ≈ Aeff(m)[1]
    m = Capillary.MarcatiliMode(a, :HeB, 1.0, n=0, kind=:TM)
    @test Modes.N(m) ≈ N(m)[1]
    @test Modes.Aeff(m) ≈ Aeff(m)[1]
    
    a0 = a
    aL = a/2
    L = 1
    afun = let a0=a0, aL=aL, L=L
        afun(z) = a0 + (aL-a0)*z/L
    end
    m = Capillary.MarcatiliMode(afun, :HeB, 1, loss=false, model=:full)
    @test Modes.N(m) ≈ N(m)[1]
    @test Modes.N(m, z=L/2) ≈ N(m, z=L/2)[1]
    @test Modes.N(m, z=L) ≈ N(m, z=L)[1]
    @test Modes.Aeff(m) ≈ Aeff(m)[1]
    @test Modes.Aeff(m, z=L/2) ≈ Aeff(m, z=L/2)[1]
    @test Modes.Aeff(m, z=L) ≈ Aeff(m, z=L)[1]
end


@testset "β, α" begin
    a = 125e-6
    m = Capillary.MarcatiliMode(a, :HeB, 1.0, model=:reduced)
    λ = 1e-9 .* collect(range(70, stop=7300, length=128))
    ω = wlfreq.(λ)
    @test all(isfinite.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(m, ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatiliMode(a, :HeB, 1.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatiliMode(a, :HeB, 10.0, model=:reduced), ω)))
    @test all(isreal.(Capillary.β.(Capillary.MarcatiliMode(a, :HeB, 50.0, model=:reduced), ω)))
    @test all(isfinite.(Capillary.α.(m, ω)))
    @test all(isreal.(Capillary.α.(m, ω)))
end

@testset "ZDW/Aeff" begin
    @test abs(1e9*Capillary.zdw(Capillary.MarcatiliMode(125e-6, :HeB, 0.4, model=:reduced)) - 379) < 1
    @test abs(1e9*Capillary.zdw(Capillary.MarcatiliMode(75e-6, :HeB, 5.9, model=:reduced)) - 562) < 1
    @test Capillary.Aeff(Capillary.MarcatiliMode(75e-6, :HeB, 1.0, model=:reduced)) ≈ 8.42157534886545e-09
end


@testset "dispersion" begin
    # tests based on symbolic results in symbolic_marcatilli.py
    # here we're using the argon ref. index from Börzsönyi et al.
    m = Capillary.MarcatiliMode(50e-6, :ArB, 2.0, model=:reduced, T=294.0)
    ω = wlfreq(800e-9)
    @test isapprox(Capillary.β(m, ω), 7857866.63899973, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744405855838e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68187063100098e-29, rtol=3e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.44025458282146e-44, rtol=2e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13289638668388e-58, rtol=8e-6)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.45046358889987e-73, rtol=2e-4)
    @test isapprox(Capillary.zdw(m), 7.289431065526978e-07, rtol=1e-7)

    rfg = ref_index_fun(:ArB, 2.0, 294.0)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> 1.45
    m = Capillary.MarcatiliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857865.68069111, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744357233436e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.68088637707608e-29, rtol=7e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.43560292665704e-44, rtol=1e-6)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.13224034494362e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.44896931597213e-73, rtol=1e-4)
    @test isapprox(Capillary.zdw(m), 7.289459086128214e-07, rtol=1e-8)
    @test isapprox(Capillary.α(m, ω), 2.21505826015050, rtol=1e-14)

    rfg = ref_index_fun(:ArB, 2.0, 294.0)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> 0.036759+im*5.5698
    m = Capillary.MarcatiliMode(50e-6, 1, 1, :HE, 0.0, coren, cladn)
    @test isapprox(Capillary.β(m, ω), 7857863.68326066, rtol=1e-15)
    @test isapprox(Capillary.dispersion(m, 1, ω), 3.33744527275634e-9, rtol=1e-13)
    @test isapprox(Capillary.dispersion(m, 2, ω), -1.89802237417366e-29, rtol=2e-7)
    @test isapprox(Capillary.dispersion(m, 3, ω), 8.80530032930868e-44, rtol=7e-7)
    @test isapprox(Capillary.dispersion(m, 4, ω), -1.21092167154788e-58, rtol=1e-5)
    @test isapprox(Capillary.dispersion(m, 5, ω), 2.64991510536236e-73, rtol=5e-5)
    @test isapprox(Capillary.zdw(m), 7.225347947615157e-07, rtol=2e-8)
    @test isapprox(Capillary.α(m, ω), 0.0290115706883820, rtol=1e-14)
    @test Capillary.Aeff(Capillary.MarcatiliMode(75e-6, :HeB, 1.0)) ≈ 8.42157534886545e-09
end

@testset "to_space" begin
    ms = (Capillary.MarcatiliMode(125e-6, :HeB, 1.0),
          Capillary.MarcatiliMode(125e-6, :HeB, 1.0, m=2, ϕ=π/2))
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