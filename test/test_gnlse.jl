using Luna
import Test: @test, @testset, @test_throws
import Logging

logger = Logging.SimpleLogger(stdout, Logging.Warn)
old_logger = Logging.global_logger(logger)

@testset "Soliton" begin
    # N = 4 soliton over 2 periods
    γ = 0.1
    β2 = -1e-26
    N = 4.0
    τ0 = 280e-15
    τfwhm = (2*log(1 + sqrt(2)))*τ0
    fr = 0.18
    P0 = N^2*abs(β2)/((1-fr)*γ*τ0^2)
    flength = pi*τ0^2/abs(β2)
    βs =  [0.0, 0.0, β2]
    λ0 = 835e-9
    λlims = [450e-9, 8000e-9]
    trange = 4e-12
    output = prop_gnlse(γ, flength, βs; λ0, τfwhm, power=P0, pulseshape=:sech, λlims, trange,
                        raman=false, shock=false, fr, shotnoise=false)
    Eωin = output["Eω"][:,1]
    Eωout = output["Eω"][:,end]
    @test isapprox(abs2.(Eωin), abs2.(Eωout), rtol=1e-2)
    grid = Processing.makegrid(output)
    T, Etin = Processing.getEt(grid, Eωin, oversampling=1)
    T, Etout = Processing.getEt(grid, Eωout, oversampling=1)
    @test isapprox(abs2.(Etin), abs2.(Etout), rtol=1e-3)
end

@testset "Soliton shift" begin
    # shift of a fundamental soliton
    γ = 0.1
    β2 = -1e-26
    N = 1.0
    τ0 = 10e-15
    fr = 0.18
    P0 = N^2*abs(β2)/((1 - fr)*γ*τ0^2)
    flength = pi/2*τ0^2/abs(β2)*90
    βs =  [0.0, 0.0, β2]
    λ0 = 835e-9
    λlims = [450e-9, 8000e-9]
    trange = 13e-12
    output = prop_gnlse(γ, flength, βs; λ0, τfwhm=1.763*τ0, power=P0, pulseshape=:sech, λlims, trange,
                        raman=true, shock=false, fr, shotnoise=false, ramanmodel=:sdo, τ1=12.2e-15, τ2=32e-15)
    Eωout = output["Eω"][:,end]
    ω = output["grid"]["ω"]
    # these numbers checked with two independent GNLSE codes
    @test isapprox(ω[argmax(abs2.(Eωout))], 1.3975031072069625e15, rtol=1e-14)
    grid = Processing.makegrid(output)
    T, Etout = Processing.getEt(grid, Eωout, oversampling=1)
    # these numbers checked with two independent GNLSE codes
    @test isapprox(T[argmax(abs2.(Etout))], 6.079566076444579e-12, rtol=1e-14)
end

##
Logging.global_logger(old_logger)