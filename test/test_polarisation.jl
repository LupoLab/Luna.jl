import Test: @test, @testset, @test_throws
import Luna: Polarisation
import LinearAlgebra: norm

@testset "Jones" begin
    h = Polarisation.H()
    @test norm(Polarisation.Stokes(h) - [1; 1; 0; 0]) < 1e-15

    QWP = Polarisation.rotate(Polarisation.WP(π/2), 0)
    @test norm(Polarisation.Stokes(QWP*h) - [1; 1; 0; 0]) < 1e-15
    QWP = Polarisation.rotate(Polarisation.WP(π/2), π/4)
    @test norm(Polarisation.Stokes(QWP*h) - [1; 0; 0; 1]) < 1e-15
    QWP = Polarisation.rotate(Polarisation.WP(π/2), -π/4)
    @test norm(Polarisation.Stokes(QWP*h) - [1; 0; 0; -1]) < 1e-15

    HWP = Polarisation.rotate(Polarisation.WP(π), 0)
    @test norm(Polarisation.Stokes(HWP*h) - [1; 1; 0; 0]) < 1e-15
    HWP = Polarisation.rotate(Polarisation.WP(π), π/4)
    @test norm(Polarisation.Stokes(HWP*h) - [1; -1; 0; 0]) < 1e-15
    HWP = Polarisation.rotate(Polarisation.WP(π), π/8)
    @test norm(Polarisation.Stokes(HWP*h) - [1; 0; 1; 0]) < 1e-15
    HWP = Polarisation.rotate(Polarisation.WP(π), -π/8)
    @test norm(Polarisation.Stokes(HWP*h) - [1; 0; -1; 0]) < 1e-15

    @test isapprox(Polarisation.ellipticity(Polarisation.Stokes(h)), 0.0)
    QWP = Polarisation.rotate(Polarisation.WP(π/2), π/4)
    @test isapprox(Polarisation.ellipticity(Polarisation.Stokes(QWP*h)), 1.0)
    QWP = Polarisation.rotate(Polarisation.WP(π/2), -π/4)
    @test isapprox(Polarisation.ellipticity(Polarisation.Stokes(QWP*h)), 1.0)
    
    QWP = Polarisation.rotate(Polarisation.WP(π/2), π/4)
    C = QWP*h
    L = Polarisation.LP()*C
    @test norm(Polarisation.Stokes(L, normalise=true) - [1; 1; 0; 0]) < 1e-15
    @test isapprox(Polarisation.Stokes(L)[1], 0.5)
end
