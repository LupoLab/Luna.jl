import Test: @test, @testset, @test_throws
import Luna: Antiresonant, Capillary, Modes
import Luna.PhysData: wlfreq

import PyPlot: plt, pygui
pygui(true)

@testset "Antiresonant PCF" begin
    a = 20e-6
    m = Capillary.MarcatilliMode(a, :Air, 0, (ω; z) -> 1.45)
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

    @test Modes.neff(arm, 2.5e15) == 0.9999193518567425 + 1.87925966056515e-6im
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=false)
    @test Modes.neff(arm, 2.5e15) == 0.9999193518567425
    arm = Antiresonant.ZeisbergerMode(a, :Air, 0, (ω; z) -> 1.45; wallthickness=w, loss=0.5)
    @test Modes.neff(arm, 2.5e15) == 0.9999193518567425 + 0.5*1.87925966056515e-6im
    @test_throws ArgumentError Antiresonant.ZeisbergerMode(a, :Air, 0; wallthickness=w, loss=0.5im)
end

