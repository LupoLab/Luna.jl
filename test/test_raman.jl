import Test: @test, @testset
import Luna: Raman, Maths
import QuadGK: quadgk

h = Raman.raman_response(:N2)
hv = Raman.raman_response(:N2, rotation=false)
hr = Raman.raman_response(:N2, vibration=false)
T = collect(range(-5e-12, stop=20e-12, length=2^16))
dt = T[2] - T[1]

# test response function is zero for 0 or negative time
@test h(0.0) == 0.0
@test all(h.(T[T .<= 0.0]) .== 0.0)

hsdo = Raman.RamanRespNormedSingleDampedOscillator(1.0, 2π*100e12, 100e-15)

# test integral of normalosed response function is 1
@test isapprox(quadgk(hsdo, -1e-12, 2e-12)[1], 1.0, rtol=1e-8)

# comparisons to fnfep (both could be wrong!)
# TODO
