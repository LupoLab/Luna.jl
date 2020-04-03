import Test: @test, @testset, @test_broken
import Luna: Raman, Maths, Nonlinear
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

hv = Raman.raman_response(:H2, rotation=false)
hr = Raman.raman_response(:H2, vibration=false)
T = collect(range(-5e-12, stop=5e-12, length=2^16))
maxrr_fnfep = 5.33685258535946e-49 # max of rotational response using rigid model in fnfep
minrr_fnfep = -5.279378127531293e-49 # min of rotational response using rigid model in fnfep
maxvr_fnfep = 7.984998322825872e-49 # max of new vibrational response model in fnfep
minvr_fnfep = -7.984860885772331e-49 # min of new vibrational response model in fnfep
# note the old fnfep model, as used in Federico's paper gave peak
# vib of: 2.278109355527885e-48 whereas the rotation
# was comparable: 6.0687907936299885e-49
@test isapprox(maximum(hr.(T)), maxrr_fnfep, rtol=1e-2)
@test isapprox(minimum(hr.(T)), minrr_fnfep, rtol=1e-2)
@test isapprox(maximum(hv.(T)), maxvr_fnfep, rtol=1e-2)
@test isapprox(minimum(hv.(T)), minvr_fnfep, rtol=1e-2)

# test off by one errors
# we use the analytic result that the heaviside function u(t) convolves
# with itself to produce t*u(t)
# we check that the Raman convolution reproduces that
# note we also use the fact that (u(t))^2 = u(t)
function ht(t)
    if t >= 0.0
        return 1.0
    else
        return 0.0
    end
end
Nt = collect(range(0, length=2^14))
t = @. (Nt - 2^14/2)*3.430944979182369e-16
rp = Nonlinear.RamanPolarEnv(t, ht)
E = ht.(t) .+ 0im
out = similar(E)
fill!(out, 0.0)
rp(out, E)
@test all(abs.(extrema(3/4 .* abs.(t.*E) .- abs.(out))) .< 1e-23)
@test 3/4 .* abs.(t.*E) ≈ abs.(out)
rp = Nonlinear.RamanPolarField(t, ht)
E = ht.(t)
out = similar(E)
fill!(out, 0.0)
rp(out, E)
@test all(abs.(extrema(abs.(t.*E) .- abs.(out))) .< 1e-23)
@test abs.(t.*E) ≈ abs.(out)
rp = Nonlinear.RamanPolarField(t, ht, thg=false)
fill!(out, 0.0)
rp(out, E)
@test_broken all(abs.(extrema(3/4 .* abs.(t.*E) .- abs.(out))) .< 1e-23)
@test_broken 3/4 .* abs.(t.*E) ≈ abs.(out)
