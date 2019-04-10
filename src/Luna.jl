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
    β = .-[1; Capillary.β(cap.radius, ω[2:end], gas=medium.gas, pressure=medium.pressure)]
    α = [1000; Capillary.α(cap.radius, ω[2:end])]
    α[α .> 10] .= 10
    β1 = -Capillary.dispersion(1, cap.radius, λ=refλ,
                              gas=medium.gas, pressure=medium.pressure)
    linop = @. im*(β - β1*ω) - α/2

    return linop
end

function make_Pnl_prefac(ω, cap::Configuration.Capillary, medium::Configuration.StaticFill)
    β = .-[1; Capillary.β(cap.radius, ω[2:end], gas=medium.gas, pressure=medium.pressure)]
    return @. im/(2*PhysData.ε_0*PhysData.c^2)*ω^2/β
end

function make_density(medium::Configuration.StaticFill)
    return z -> PhysData.std_dens * medium.pressure
end

function make_fnl(ω, ωo, Eω, Et, conf)
    Eωo = zeros(ComplexF64, size(ωo))
    tsamples = Int((length(ωo)-1)*2)
    FT = FFTW.plan_rfft(zeros(Float64, tsamples))
    IFT = FFTW.plan_irfft(Eωo, tsamples)
    
    χ3 = PhysData.χ3_gas(conf.medium.gas)
    kerr! = Nonlinear.make_kerr!(χ3)
    responses = (kerr!,)

    dens = make_density(conf.medium)

    tsamples_coarse = Int((length(ω)-1)*2)
    scalefac = tsamples/tsamples_coarse

    cropidx = length(ω)


    Pto = zeros(Float64, tsamples)
    Pωo = similar(Eωo)
    Eto = similar(Pto)
    prefac = make_Pnl_prefac(ω, conf.geometry, conf.medium)
    fnl! = let Pto=Pto, Eto=Eto, FT=FT, IFT=IFT, responses=responses, Eωo=Eωo, cropidx=cropidx, Pωo=Pωo, prefac=prefac, scalefac=scalefac
        function fnl!(out, Eω, z)
            fill!(Pto, 0)
            fill!(Eωo, 0)
            Eωo[1:cropidx] .= scalefac.*Eω
            Eto .= IFT*Eωo
            for resp in responses
                resp(Pto, Eto)
            end
            Pωo .= (FT*Pto)
            out .= dens(z).*prefac.*Pωo[1:cropidx]
        end
    end
    return fnl!
end

function make_grid(grid::Configuration.Grid)
    return make_grid(grid.λ_lims, grid.trange, grid.δt, grid.apod_width)
end

function make_grid(λ_lims, trange, δt, apod_width)
    f_lims = PhysData.c./λ_lims
    Logging.@info @sprintf("Freq limits %.2f - %.2f PHz", f_lims[2]*1e-15, f_lims[1]*1e-15)
    δt = min(1/(6*maximum(f_lims)), δt) # 6x maximum freq, or user-defined if finer
    samples_fine = 2^(ceil(Int, log2(trange/δt))) # samples for fine grid (power of 2)
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d", trange/δt, samples_fine)
    δt_fine = trange/samples_fine # actual spacing - keep trange fixed
    δω_fine = 2π/trange # frequency spacing for fine grid
    # Make fine grid
    Nt_fine = collect(range(0, length=samples_fine))
    t_fine = @. (Nt_fine-samples_fine/2)*δt_fine # center on 0
    Nω_fine = collect(range(0, length=Int(samples_fine/2 +1)))
    ω_fine = Nω_fine*δω_fine
    # Make coarse grid
    # find smallest power-of-2 section of grid that contains user-desired frequency window:
    ωmax = 2π*(maximum(f_lims)) + 2*apod_width # maximum freq in user-desired window
    cropidx = findfirst(ω -> ω>=ωmax, ω_fine)
    Logging.@info @sprintf("ωmax = %.2e, cropidx = %d, ω[cropidx] = %.2e", ωmax, cropidx, ω_fine[cropidx])
    cropidx = 2^(ceil(Int, log2(cropidx))) + 1 # make coarse ω grid also power of 2 + 1
    ω_coarse = ω_fine[1:cropidx]
    δt_coarse = π/maximum(ω_coarse)
    samples_coarse = (cropidx-1)*2
    Nt_coarse = collect(range(0, length=samples_coarse))
    t_coarse = @. (Nt_coarse-samples_coarse/2)*δt_coarse

    # Make apodisation window
    ωmin = 2π*minimum(f_lims)
    ω_left = ωmin/2
    width_left = ωmin/8
    ω_right = ωmax-2*apod_width
    width_right = apod_width
    window = Maths.errfun_window(ω_coarse, ω_left, ω_right, width_left, width_right)
    
    @assert δt_coarse/δt_fine ≈ length(t_fine)/length(t_coarse)
    @assert δt_coarse/δt_fine ≈ maximum(ω_fine)/maximum(ω_coarse)
    Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e ",
                           length(t_fine), length(t_coarse), maximum(ω_coarse), maximum(ω_fine))
    return t_coarse, ω_coarse, t_fine, ω_fine, window
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
    Aeff = Capillary.Aeff(75e-6) #!!!! HARDCODED
    energy = abs(NumericalIntegration.integrate(t, It, NumericalIntegration.SimpsonEven()))
    energy *= 2*PhysData.c*PhysData.ε_0*Aeff
    It .*= input.energy/energy

    ω0 = 2π*PhysData.c/input.wavelength
    Et = @. sqrt(It)*cos(ω0*t)

    return FFTW.rfft(Et)
end

import PyPlot: pygui, plt
function run(config)
    pygui(true)
    t, ω, to, ωo, window = make_grid(config.grid)

    Eω = make_init(t, config.input)
    en = NumericalIntegration.integrate(ω, abs2.(Eω), NumericalIntegration.SimpsonEven())
    Et = FFTW.irfft(Eω, length(t))

    Eωo = zeros(ComplexF64, size(ωo))
    Eωo[1:length(ω)] = Eω
    Eto = FFTW.irfft(Eωo, length(to))

    # plt.figure()
    # plt.plot(t, Et)
    # plt.plot(to, length(to)/length(t).*Eto)
    # error()

    linop = make_linop(ω, config.grid.referenceλ, config.geometry, config.medium)
    fnl! = make_fnl(ω, ωo, Eω, Et, config)

    z = 0
    dz = 1e-8
    zmax = config.geometry.length
    saveN = 201

    window! = let window=window
        function window!(E)
            E .*= window
        end
    end

    zout, Eout, steps = RK45.solve_precon(fnl!, linop, Eω, z, dz, zmax, saveN, stepfun=window!)

    Etout = FFTW.irfft(Eout, length(t), 1)

    return ω, t, zout, Eout, Etout
end

end # module
