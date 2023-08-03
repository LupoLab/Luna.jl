import Test: @test, @test_throws, @testset, @test_broken
import Luna: PhysData

@testset "All" begin
@testset "Exceptions" begin
    @test_throws DomainError PhysData.ref_index(:HeBllo, 800e-9)
end

@testset "refractive indices" begin
    @test PhysData.ref_index(:HeB, 800e-9) ≈ 1.000031950041203
    @test PhysData.ref_index(:HeB, 800e-9, 10) ≈ 1.0003180633169397
    @test PhysData.ref_index(:SiO2, 800e-9, lookup=false) ≈ 1.4533172548587419
    @test PhysData.ref_index(:SiO2, 400e-9, lookup=false) ≈ 1.4701161185594052
    @test PhysData.ref_index(:SiO2, PhysData.eV_to_m(6)) ≈ 1.543
    @test PhysData.ref_index(:SiO2, PhysData.eV_to_m(1.455)) ≈ 1.45248
    @test PhysData.ref_index(:SiO2, PhysData.eV_to_m(0.91018)) ≈ 1.44621
    @test real(PhysData.ref_index(:SiO2, PhysData.eV_to_m(121.6))) ≈ 0.9865
    @test imag(PhysData.ref_index(:SiO2, PhysData.eV_to_m(121.6))) ≈ 0.0085
end

@testset "Function equivalence" begin
    @test PhysData.ref_index_fun(:SiO2)(800e-9) == PhysData.ref_index(:SiO2, 800e-9)
    @test PhysData.ref_index_fun(:HeB)(800e-9) == PhysData.ref_index(:HeB, 800e-9)
end

@testset "Dispersion" begin
    @test PhysData.dispersion(2, :SiO2, 800e-9, lookup=false) ≈ 3.61619983e-26
    @test isapprox(PhysData.dispersion(2, :HeB, 800e-9), 9.373942337550116e-31, rtol=1e-5)
    @test isapprox(PhysData.dispersion(2, :HeB, 800e-9, 10), 9.33043805928079e-30, rtol=1e-5)
end

@testset "glasses" begin
    for g in PhysData.glass
        @test (g == :SiO2 ? 
                !isreal(PhysData.ref_index(g, 800e-9)) : # SiO2 ref index is complex
                isreal(PhysData.ref_index(g, 800e-9)))
    end
    @test isreal(PhysData.ref_index(:SiO2, 800e-9, lookup=false))
end

@testset "gases" begin
    for g in PhysData.gas
        @test isreal(PhysData.ref_index(g, 800e-9))
        @test isreal(PhysData.ref_index(g, 200e-9))
        @test isreal(PhysData.ref_index(g, 800e-9, 10))
        @test isreal(PhysData.ref_index(g, 200e-9, 10))
    end
end

@testset "Nonlinear coefficients" begin
    @test_broken PhysData.χ3(:HeB, 1) ≈ 1.2617371645226101e-27
    @test PhysData.χ3(:Ar, 1) ≈ 2.964158749949189e-26
    @test_broken PhysData.n2(:HeB, 1) ≈ 3.5647819877255427e-25
    @test PhysData.n2(:HeB, 2) ≈ 7.125642138007481e-25
    @test_broken PhysData.n2.(:HeB, [1, 2]) ≈ [3.5647819877255427e-25, 7.125642138007481e-25]
    @test_broken PhysData.n2.([:HeB, :Ne], 1) ≈ [3.5647819877255427e-25, 6.416061508801999e-25]
    for gas in PhysData.gas[2:end] # Don't have γ3 for Air
        @test isreal(PhysData.n2(gas, 1))
    end
end

@testset "Density" begin
    # compare to refprop
    # room temperature
    @test isapprox(PhysData.density(:Ar, 0.002, 294.0), 4.927169444649429e22, rtol=2e-15)
    @test isapprox(PhysData.density(:Ar, 0.02, 294.0), 4.9272289924653735e23, rtol=9e-15)
    @test isapprox(PhysData.density(:Ar, 2.0, 294.0), 4.933761614600933e25, rtol=4e-13)
    @test isapprox(PhysData.density(:Ar, 40.0, 294.0), 1.0101579129300146e27, rtol=2e-11)
    @test isapprox(PhysData.density(:Ar, 400.0, 294.0), 9.270757850984163e27, rtol=3e-10)
    @test_broken isapprox(PhysData.density(:HeB, 0.002, 294.0), 4.927180563885407e22, rtol=7e-16)
    @test_broken isapprox(PhysData.density(:HeB, 0.02, 294.0), 4.927137517453137e23, rtol=2e-16)
    @test_broken isapprox(PhysData.density(:HeB, 2.0, 294.0), 4.9224080868066745e25, rtol=2e-13)
    @test_broken isapprox(PhysData.density(:HeB, 40.0, 294.0), 9.66755645602771e26, rtol=6e-16)
    @test_broken isapprox(PhysData.density(:HeB, 400.0, 294.0), 8.309132317978155e27, rtol=1e-13)
    @test isapprox(PhysData.density(:Xe, 0.002, 294.0), 4.927238737907382e22, rtol=7e-16)
    @test isapprox(PhysData.density(:Xe, 0.02, 294.0), 4.92771934762324e23, rtol=2e-16)
    @test isapprox(PhysData.density(:Xe, 2.0, 294.0), 4.981603290885485e25, rtol=9e-16)
    @test isapprox(PhysData.density(:Xe, 20.0, 294.0), 5.5912074541820115e26, rtol=8e-15)
    @test isapprox(PhysData.density(:Xe, 40.0, 294.0), 1.3512153640360566e27, rtol=7e-16)
    @test isapprox(PhysData.density(:Xe, 60.0, 294.0), 3.218616771878294e27, rtol=6e-15)
    @test isapprox(PhysData.density(:N2, 0.002, 294.0), 4.927165137695148e22, rtol=7e-14)
    @test isapprox(PhysData.density(:N2, 0.02, 294.0), 4.927185920476881e23, rtol=7e-13)
    @test isapprox(PhysData.density(:N2, 2.0, 294.0), 4.929427604602592e25, rtol=6e-11)
    @test isapprox(PhysData.density(:N2, 40.0, 294.0), 9.909239438516337e26, rtol=3e-10)
    @test isapprox(PhysData.density(:N2, 400.0, 294.0), 7.890787516580131e27, rtol=1e-7)
    @test isapprox(PhysData.density(:H2, 0.002, 294.0), 4.9271795187225075e22, rtol=9e-16)
    @test isapprox(PhysData.density(:H2, 0.02, 294.0), 4.927127065719356e23, rtol=6e-16)
    @test isapprox(PhysData.density(:H2, 2.0, 294.0), 4.921361734871252e25, rtol=4e-13)
    @test isapprox(PhysData.density(:H2, 40.0, 294.0), 9.624525613497868e26, rtol=6e-16)
    @test isapprox(PhysData.density(:H2, 400.0, 294.0), 7.840662987579036e27, rtol=6e-16)
    # temperature range
    @test_broken isapprox(PhysData.density(:HeB, 10.0, 10.0), 9.218927214187151e27, rtol=6e-16)
    @test_broken isapprox(PhysData.density(:HeB, 10.0, 100.0), 7.142310177179197e26, rtol=6e-16)
    @test_broken isapprox(PhysData.density(:HeB, 10.0, 1000.0), 7.234660811823096e25, rtol=6e-13)
    @test isapprox(PhysData.density(:Xe, 10.0, 170.0), 1.336261228046876e28, rtol=6e-16)
    @test isapprox(PhysData.density(:Xe, 10.0, 700.0), 1.0361035050644844e26, rtol=6e-13)
    @test isapprox(PhysData.density(:N2, 10.0, 100.0), 1.4849593021049305e28, rtol=6e-7)
    @test isapprox(PhysData.density(:N2, 10.0, 1000.0), 7.217962021789807e25, rtol=6e-10)
end

@testset "Density spline" begin
    P = range(0, 10, length=128)
    Plow = range(0, 0.1, length=128)
    Pfine = range(0, 0.01, length=128)
    for g in PhysData.gas
        dens = PhysData.densityspline(g, Pmax=maximum(P))
        @test all(dens.(P) .≈ PhysData.density.(g, P))
        dens = PhysData.densityspline(g, Pmax=maximum(Plow))
        @test all(dens.(Plow) .≈ PhysData.density.(g, Plow))
        @test all(isapprox.(dens.(Pfine), PhysData.density.(g, Pfine), rtol=1e-6))
    end
end
end