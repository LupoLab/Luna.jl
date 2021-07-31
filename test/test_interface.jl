using Luna
import Test: @test, @testset, @test_throws
import Logging

logger = Logging.SimpleLogger(stdout, Logging.Warn)
old_logger = Logging.global_logger(logger)

@testset "Polarisation" begin
    args = (100e-6, 0.1, :He, 1)
    kwargs = (λ0=800e-9, τfwhm=10e-15, energy=1e-12, shotnoise=false)
    lin = prop_capillary(args...; polarisation=:linear, kwargs...)
    @testset "linear, single mode" begin
        o1 = prop_capillary(args...; polarisation=0.0, kwargs...)
        o2 = prop_capillary(args...; polarisation=:linear, modes=1, kwargs...)
        @test o1["Eω"][:, 1, :] ≈ o2["Eω"][:, 1, :]
        @test size(o1["Eω"], 2) == 2*size(o2["Eω"], 2)
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
    args = (100e-6, 0.1, :He, 1)
    kwargs = (λ0=800e-9, τfwhm=10e-15, shotnoise=false, trange=500e-15,
              saveN=51, plasma=false)
    shape_fac = ((:gauss, kwargs.τfwhm*sqrt(pi/log(16))),
                 (:sech, 2*kwargs.τfwhm/(2*log(1 + sqrt(2)))))
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
                    φ2 = Tools.τfw_to_τ0(kwargs.τfwhm, :gauss)^2
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
    args = (100e-6, 0.1, :He, 1)
    kwargs = (λ0=800e-9, shotnoise=false, trange=250e-15,
               saveN=51, plasma=false)
    pkwargs = (τfwhm=10e-15, energy=1e-12, λ0=800e-9)
    @testset "input into $m, mode average" for m in (:HE11, :HE12, :TE01, :TE02, :TM01)
        ip = Interface.GaussPulse(;mode=m, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=m, kwargs...)
        @test Processing.energy(o)[1] ≈ pkwargs.energy
    end
    @testset "input into $m, modal" for (midx, m) in enumerate((:HE11, :HE12, :HE13, :HE14))
        ip = Interface.GaussPulse(;mode=m, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=4, kwargs...)
        @test Processing.energy(o)[midx, 1] ≈ pkwargs.energy
    end
    @testset "input into $m, modal circular" for (midx, m) in enumerate((:HE11, :HE12, :HE13, :HE14))
        ip = Interface.GaussPulse(;mode=m, polarisation=:circular, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=4, kwargs...)
        @test Processing.energy(o)[2midx-1, 1] ≈ pkwargs.energy/2
        @test Processing.energy(o)[2midx, 1] ≈ pkwargs.energy/2
    end
    modes = (:TE01, :TE02, :TE03, :TE04, :TM01, :TM02, :TM03, :TM04)
    @testset "input into $m, modal" for (midx, m) in enumerate((modes))
        ip = Interface.GaussPulse(;mode=m, pkwargs...)
        o = prop_capillary(args...; pulses=ip, modes=modes, kwargs...)
        @test Processing.energy(o)[midx, 1] ≈ pkwargs.energy
    end
end

##
@testset "multiple inputs" begin
    args = (100e-6, 0.1, :He, 1)
    kwargs = (λ0=800e-9, shotnoise=false, trange=250e-15, saveN=51, plasma=false)
    p1 = (λ0=800e-9, energy=1e-12, τfwhm=10e-15)
    p2 = (λ0=400e-9, energy=2e-12, τfwhm=30e-15)
    modes = (:HE11, :HE12, :HE13, :HE14)
    @testset "first pulse into $m1" for (idx1, m1) in enumerate(modes)
        ip1 = Interface.GaussPulse(;mode=m1, p1...)
        @testset "second pulse into $m2" for (idx2, m2) in enumerate(modes)
            ip2 = Interface.GaussPulse(;mode=m2, p2...)
            o = prop_capillary(args...; pulses=[ip1, ip2], modes=4, kwargs...)
            if idx1 == idx2
                @test Processing.energy(o)[idx1, 1] ≈ p1.energy + p2.energy
            else
                @test Processing.energy(o)[idx1, 1] ≈ p1.energy
                @test Processing.energy(o)[idx2, 1] ≈ p2.energy
                @test isapprox(Processing.fwhm_t(o)[idx1, 1], p1.τfwhm, rtol=1e-3)
                @test isapprox(Processing.fwhm_t(o)[idx2, 2], p2.τfwhm, rtol=1e-3)
            end
        end
    end
end

##
@testset "propagators" begin
    # passing ϕ keyword argument and an equivalent propagator function should yield 
    # the same result.
    args = (100e-6, 0.1, :He, 1)
    kwargs = (λ0=800e-9, shotnoise=false, trange=250e-15, saveN=51, plasma=false)
    p = (λ0=800e-9, energy=1e-12, τfwhm=10e-15)
    ϕ = [0, 0, 10e-30, 100e-45]
    function prop!(Eω, grid)
        Fields.prop_taylor!(Eω, grid, ϕ, 800e-9)
    end
    pp = Interface.GaussPulse(;p..., propagator=prop!)
    pt = Interface.GaussPulse(;p..., ϕ)
    op = prop_capillary(args...; pulses=pp, kwargs...)
    ot = prop_capillary(args...; pulses=pt, kwargs...)
    @test isapprox(Processing.fwhm_t(op)[1], Processing.fwhm_t(ot)[1], rtol=1e-3)
    @test Processing.energy(op)[1] ≈ p.energy
    @test Processing.energy(ot)[1] ≈ p.energy

    op = prop_capillary(args...; p..., propagator=prop!, kwargs...)
    ot = prop_capillary(args...; p..., ϕ, kwargs...)
    @test isapprox(Processing.fwhm_t(op)[1], Processing.fwhm_t(ot)[1], rtol=1e-3)
    @test Processing.energy(op)[1] ≈ p.energy
    @test Processing.energy(ot)[1] ≈ p.energy
end

##
Logging.global_logger(old_logger)