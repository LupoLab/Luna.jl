import Test: @test, @testset
import Cubature: hcubature
import LinearAlgebra: dot, norm
using Luna.RectModes
import Luna.Modes
import Luna.PhysData: c, ε_0, μ_0, wlfreq

a = 50e-6
b = 12.5e-6

m = RectMode(a, b, :Ar, 0.0, :Ag)

# these were checked with FEM using JCMwave
@test isapprox(Aeff(m), 1.111111111111111e-09)
@test isapprox(Modes.dB_per_m(m, wlfreq(210e-9)), 3.908368640803528, rtol=1e-4)
@test isapprox(Modes.dB_per_m(m, wlfreq(800e-9)), 0.44317059469197245, rtol=1e-2)
@test isapprox(β(m, wlfreq(800e-9)), 7852915.234685494)
@test isapprox(β(m, wlfreq(210e-9)), 29919650.305427298)

@test abs(1e9*Modes.zdw(RectMode(a, b, :Ar, 5.0, :Ag)) - 562.2) < 1

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

    for n=1:6
        for m = 1:6
            rm = RectMode(a, b, :Ar, 0.0, :Ag)
            @test Modes.N(rm) ≈ N(rm)
            @test Modes.Aeff(rm) ≈ Aeff(rm)
        end
    end
end
