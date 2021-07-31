using Luna
import Test: @test, @testset, @test_throws
import Logging

logger = Logging.SimpleLogger(stdout, Logging.Warn)
old_logger = Logging.global_logger(logger)

@testset "Polarisation" begin
    args = (10e-6, 0.1, :He, 1)
    kwargs = Dict(:λ0 => 800e-9, :τfwhm => 10e-15, :energy => 1e-12, :shotnoise => false)
    lin = prop_capillary(args...; polarisation=:linear, kwargs...)
    @testset "linear, single mode" begin
        o1 = prop_capillary(args...; polarisation=0.0, kwargs...)
        o2 = prop_capillary(args...; polarisation=:linear, modes=1, kwargs...)
        @test o1["Eω"][:, 1, :] ≈ o2["Eω"][:, 1, :]
        @test size(o1, 2) == 2*size(o2, 2)
    end
    @testset "linear, multimode" begin
        o1 = prop_capillary(args...; polarisation=0.0, modes=4, kwargs...)
        o2 = prop_capillary(args...; polarisation=:linear, modes=4, kwargs...)
        @test o1["Eω"][:, 1:2:end, :] ≈ o2["Eω"][:, :, :]
    end
    @testset "circular, $modes" for modes in (:HE11, :HE12, 1, 2)
        o1 = prop_capillary(args...; modes, polarisation=:circular, kwargs...)
        o2 = prop_capillary(args...; modes, polarisation=1.0, kwargs...)
        @test o1["Eω"] == o2["Eω"]
        o3 = prop_capillary(args...; modes, polarisation=-1.0, kwargs...)
        @test o2["Eω"][:, 1:2:end, :] ≈ o3["Eω"][:, 1:2:end, :]
        @test o2["Eω"][:, 2:2:end, :] ≈ -o3["Eω"][:, 2:2:end, :]
    end
    @testset "elliptical, $modes" for modes in (:HE11, :HE12, 1, 2)
        ε = 0.5
        o1 = prop_capillary(args...; polarisation=ε, kwargs...)
        @test ε^2*sum(abs2.(o1["Eω"][:, 1:2:end, :])) ≈ sum(abs2.(o1["Eω"][:, 2:2:end, :]))
    end
end

##
@testset "Peak power vs energy" begin
    args = (10e-6, 0.1, :He, 1)
    kwargs = Dict(:λ0 => 800e-9, :τfwhm => 10e-15, :shotnoise => false, :trange => 500e-15,
                    :saveN => 51, :plasma => false)
    shape_fac = ((:gauss, τfwhm*sqrt(pi/log(16))),
                 (:sech, 2*τfwhm/(2*log(1 + sqrt(2)))))
    pp = 1e8
    @testset "$pol polarisation" for pol in (:linear, :circular)
        @testset "modes: $modes" for modes in (:HE11, :HE12, 1, 2)
            @testset "$(sf[1])" for sf in shape_fac
                s, f = sf
                op = prop_capillary(args...; pulseshape=s, power=pp, modes, polarisation=pol, kwargs...)
                oe = prop_capillary(args...; pulseshape=s, energy=f*pp, modes, polarisation=pol, kwargs...)
                @test Processing.energy(op) ≈ Processing.energy(oe)
                if s == :gauss
                    # stretch by factor of sqrt(2) with GDD
                    # peak power drops by 1/sqrt(2) but energy is the same
                    φ2 = Tools.τfw_to_τ0(τfwhm, :gauss)^2
                    op = prop_capillary(args...; pulseshape=s, power=pp/sqrt(2),
                                                modes, polarisation=pol, phases=[0, 0, φ2],
                                                kwargs...)
                    oe = prop_capillary(args...; pulseshape=s, energy=f*pp,
                                        modes, polarisation=pol, phases=[0, 0, φ2],
                                        kwargs...)
                    @test Processing.energy(op) ≈ Processing.energy(oe)
                end
            end
        end
    end
end

##
@testset "Input into higher-order modes" begin
    args = (10e-6, 0.1, :He, 1)
    kwargs = Dict(:λ0 => 800e-9, :shotnoise => false, :trange => 500e-15,
                    :saveN => 51, :plasma => false)
    pkwargs = (τfwhm=10e-15, energy=1e-12, λ0=800e-9)
    @testset "input into $m, mode average" for m in (:HE11, :HE12, :TE01, :TE02, :TM01)
        ip = Interface.GaussPulse(;mode=m, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=m, kwargs...)
        @test Processing.energy(o)[1] ≈ pkwargs[:energy]
    end
    @testset "input into $m, modal" for (midx, m) in enumerate((:HE11, :HE12, :HE13, :HE14))
        ip = Interface.GaussPulse(;mode=m, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=4, kwargs...)
        @test Processing.energy(o)[midx, 1] ≈ pkwargs[:energy]
    end
end