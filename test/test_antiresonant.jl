import Test: @test, @testset, @test_throws
import Luna: Antiresonant, Capillary, Modes
import Luna.PhysData: wlfreq

@testset "Zeisberger Model" begin
    a = 20e-6
    m = Capillary.MarcatiliMode(a, :Air, 0, (ω; z) -> 1.45)
    w = 0.7e-6
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w)
    arm2 = Antiresonant.ZeisbergerMode(m; wallthickness=w)

    @test Modes.N(m) == Modes.N(arm) == Modes.N(arm2)
    @test Modes.Aeff(m) == Modes.Aeff(arm) == Modes.Aeff(arm2)
    r = a*rand(50)
    θ = 2π*rand(50)
    @test all(Modes.field.(m, (r, θ)) .== Modes.field.(arm, (r, θ)))
    @test all(Modes.field.(m, (r, θ)) .== Modes.field.(arm2, (r, θ)))
    λ = collect(range(390e-9, stop=1450e-9, length=2^11))
    ω = wlfreq.(λ)
    @test all(Modes.neff.(arm, ω) .== Modes.neff.(arm2, ω))

    @test Modes.neff(arm, 2.5e15) ≈ 0.9999193518567425 + 1.87925966056515e-6im
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=false)
    @test Modes.neff(arm, 2.5e15) == 0.9999193518567425
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=0.5)
    @test Modes.neff(arm, 2.5e15) ≈ 0.9999193518567425 + 0.5*1.87925966056515e-6im
    @test_throws ArgumentError Antiresonant.ZeisbergerMode(a, :Air, 0; wallthickness=w, loss=0.5im)
end

##
@testset "Vincetti Model" begin
t = 1e-6
r_ext = 10e-6
δ = 5e-6 # tube spacing
n = 1.44
N = 8 # number of tubes

Rco = Antiresonant.getRco(r_ext, N, δ)

@test Antiresonant.getδ(Rco, r_ext, N) ≈ δ
@test Antiresonant.getr_ext(Rco, N, δ) ≈ r_ext

cladn  = (ω; z) -> 1.44


m = Antiresonant.VincettiMode(Rco; wallthickness=t, tube_radius=r_ext, Ntubes=N, cladn)

F = collect(range(0.4, 4.2, 2^14))
λ = @. 2t/F*sqrt(n^2-1)

scale = 0.5
msc = Antiresonant.VincettiMode(Rco; wallthickness=t, tube_radius=r_ext, Ntubes=N, cladn,
                                     loss=scale)
@test Modes.α.(msc, wlfreq.(λ)) ≈ scale*Modes.α.(m, wlfreq.(λ))
m0 = Antiresonant.VincettiMode(Rco; wallthickness=t, tube_radius=r_ext, Ntubes=N, cladn,
                                     loss=false)
@test all(Modes.α.(m0, wlfreq.(λ)) .== 0)

@test Modes.neff(m, wlfreq(1030e-9)) ≈ 0.9998598623672965 + 3.4579455755137454e-8im

@test Modes.dimlimits(m) == Modes.dimlimits(m.m)
end