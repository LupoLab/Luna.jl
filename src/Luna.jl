module Luna
import FFTW
import NumericalIntegration
import Logging
import Printf: @sprintf
include("Maths.jl")
include("RK45.jl")
include("PhysData.jl")
include("Capillary.jl")
include("Config.jl")
include("Nonlinear.jl")
include("Ionisation.jl")

function make_linop(ω, refλ, cap::Configuration.Capillary, medium::Configuration.StaticFill)
    βfun(ω) = Capillary.β(cap.radius, ω, gas=medium.gas, pressure=medium.pressure)
    αfun(ω) = Capillary.α(cap.radius, ω)
    return make_linop(ω, βfun, αfun, refλ)
end

function make_linop(ω, refλ, pcf::Configuration.HCPCF, medium::Configuration.StaticFill)
    α = log(10)/10 * pcf.dB_per_m
    βfun(ω) = Capillary.β(pcf.radius, ω, gas=medium.gas, pressure=medium.pressure)
    αfun(ω) = α
    return make_linop(ω, βfun, αfun, refλ)
end

function make_linop(ω, βfun, αfun, refλ)
    ω0 = 2π*PhysData.c/refλ
    idx0 = (ω .== 0)
    β = zero(ω)
    α = zero(ω)
    β[.~idx0] .= .-βfun(ω[.~idx0])
    β[idx0] .= 0
    β1 = -Maths.derivative(βfun, ω0, 1)
    α[.~idx0] .= αfun(ω[.~idx0])
    α[idx0] .= maximum(α[.~idx0])

    return @. im*(β-β1*ω) - α/2
end

function make_Pnl_prefac(ω, cap, medium::Configuration.StaticFill)
    βfun(ω) = Capillary.β(cap.radius, ω, gas=medium.gas, pressure=medium.pressure)
    return make_Pnl_prefac(ω, βfun)
end

function make_Pnl_prefac(ω, βfun)
    β = zero(ω)
    idx0 = (ω .== 0)
    β[.~idx0] .= .-βfun(ω[.~idx0])
    β[idx0] .= 1
    out = @. im/(2*PhysData.ε_0*PhysData.c^2)*ω^2/β
    out[idx0] .= 0
    return out
end

function make_density(medium::Configuration.StaticFill)
    return z -> PhysData.std_dens * medium.pressure
end

function make_fnl(ω, ωo, to, Eω, Et, ωwindow, twindow, conf)
    cropidx = length(ω)
    tsamples = Int((length(ωo)-1)*2)
    Eωo = zeros(ComplexF64, length(ωo))
    Eto = zeros(Float64, length(to))
    FT = FFTW.plan_rfft(zeros(tsamples))
    IFT = FFTW.plan_irfft(Eωo, tsamples)
    
    χ3 = PhysData.χ3_gas(conf.medium.gas)
    kerr! = Nonlinear.make_kerr!(χ3)

    ionpot = PhysData.ionisation_potential(conf.medium.gas)
    ionrate = Ionisation.ionrate_fun!_ADK(ionpot)
    plasma! = Nonlinear.make_plasma!(to, ωo, Eto, ionrate, ionpot)
    responses = (kerr!, plasma!)

    dens = make_density(conf.medium)

    scalefac = (length(ωo)-1)/(length(ω)-1)

    Pt = zeros(Float64, tsamples)
    Pω = similar(Eωo)
    Et = similar(Pt)
    prefac = make_Pnl_prefac(ω, conf.geometry, conf.medium)
    fnl! = let Pt=Pt, Et=Et, FT=FT, IFT=IFT, responses=responses, Pω=Pω, prefac=prefac,
                scalefac=scalefac, Eωo=Eωo, cropidx=cropidx, ωwindow=ωwindow, twindow=twindow
        function fnl!(out, Eω, z)
            fill!(Pt, 0)
            fill!(Eωo, 0)
            Eωo[1:cropidx] = scalefac*Eω
            Et .= IFT*Eωo
            @. Et *= twindow
            for resp in responses
                resp(Pt, Et)
            end
            @. Pt *= twindow
            Pω .= (FT*Pt)
            out .= ωwindow.*dens(z).*prefac.*Pω[1:cropidx]./scalefac
        end
    end
    return fnl!, prefac
end

function make_grid(grid::Configuration.Grid)
    return make_grid(grid.λ_lims, grid.trange, grid.δt, grid.apod_width)
end

function make_grid(λ_lims, trange, δt, apod_width)
    f_lims = PhysData.c./λ_lims
    Logging.@info @sprintf("Freq limits %.2f - %.2f PHz", f_lims[2]*1e-15, f_lims[1]*1e-15)
    δto = min(1/(6*maximum(f_lims)), δt) # 6x maximum freq, or user-defined if finer
    samples = 2^(ceil(Int, log2(trange/δto))) # samples for fine grid (power of 2)
    trange_even = δto*samples # keep frequency window fixed, expand time window as necessary
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d, δt = %.2f as",
                            trange/δto, samples, δto*1e18)
    δωo = 2π/trange_even # frequency spacing for fine grid
    # Make fine grid
    Nto = collect(range(0, length=samples))
    to = @. (Nto-samples/2)*δto # center on 0
    Nωo = collect(range(0, length=Int(samples/2 +1)))
    ωo = Nωo*δωo

    ωmax = 2π*maximum(f_lims)
    cropidx = findfirst(x -> x>ωmax+4*apod_width, ωo)
    cropidx = 2^(ceil(Int, log2(cropidx))) + 1
    ω = ωo[1:cropidx]
    δt = π/maximum(ω)
    tsamples = (cropidx-1)*2
    Nt = collect(range(0, length=tsamples))
    t = @. (Nt-tsamples/2)*δt

    # Make apodisation window
    ωmin = 2π*minimum(f_lims)
    ω_left = ωmin/2
    width_left = ωmin/16
    ω_right = 2π*maximum(f_lims)+2*apod_width
    width_right = apod_width
    window = Maths.errfun_window(ω, ω_left, ω_right, width_left, width_right)
    # window = Maths.hypergauss_window(ω, ω_left, ω_right, 1200)
    # window = Maths.planck_taper(ω, 0, 1e15+2π*maximum(f_lims), 0.05) # !!!HARDCODED

    # twindow = Maths.errfun_window(to, minimum(t)+50e-15, maximum(t)-50e-15, 10e-15)
    twindow = Maths.planck_taper(to, minimum(t) + 50e-15, maximum(t) - 50e-15, 0.1)

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)

    Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e",
                           length(t), length(to), maximum(ω), maximum(ωo))
    return t, ω, to, ωo, window, twindow
end

function make_init(t, inputs::NTuple{N, T}) where N where T<:Configuration.Input
    out = fill(0.0 + 0.0im, Int(length(t)/2+1))
    for input in inputs
        out .+= make_init(t, input)
    end
    return out
end

function make_init(t, input::Configuration.GaussInput)
    It = Maths.gauss(t, fwhm=input.duration)
    Aeff = Capillary.Aeff(13e-6) #!!!! HARDCODED
    energy = abs(NumericalIntegration.integrate(t, It, NumericalIntegration.SimpsonEven()))
    energy *= PhysData.c*PhysData.ε_0*Aeff/2
    It .*= input.energy/energy

    ω0 = 2π*PhysData.c/input.wavelength
    Et = @. sqrt(It)*cos(ω0*t)

    return FFTW.rfft(Et)
end

import PyPlot: pygui, plt
function run(config)
    pygui(true)
    t, ω, to, ωo, window, twindow = make_grid(config.grid)

    Eω = make_init(t, config.input)
    Et = FFTW.irfft(Eω, length(t))

    linop = make_linop(ω, config.grid.referenceλ, config.geometry, config.medium)
    fnl!, prefac = make_fnl(ω, ωo, to, Eω, Et, window, twindow, config)

    f! = let linop=linop, fnl! = fnl!
        function f!(out, Eω, z)
            fnl!(out, Eω, z)
            out .+= linop.*Eω
            # out .= linop.*Eω
        end
    end

    z = 0
    dz = 1e-3
    zmax = config.geometry.length
    saveN = 201

    FT = FFTW.plan_rfft(Et)
    IFT = FFTW.plan_irfft(Eω, length(t))

    twindow = Maths.planck_taper(t, minimum(t) + 50e-15, maximum(t) - 50e-15, 0.1)
    # twindow = Maths.errfun_window(t, minimum(t)+50e-15, maximum(t)-50e-15, 10e-15)

    window! = let window=window, twindow=twindow, FT=FT, IFT=IFT, Et=Et
        function window!(Eω)
            Eω .*= window
            Et .= IFT*Eω
            Et .*= twindow
            Eω .= FT * Et
        end
    end

    zout, Eout, steps = RK45.solve_precon(
        fnl!, linop, Eω, z, dz, zmax, saveN, stepfun=window!)

    Etout = FFTW.irfft(Eout, length(t), 1)

    return ω, t, zout, Eout, Etout, window, twindow, prefac
end

end # module
