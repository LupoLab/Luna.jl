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

function make_fnl(ω, Eω, Et, conf)
    tsamples = Int((length(ω)-1)*2)
    FT = FFTW.plan_rfft(Et)
    IFT = FFTW.plan_irfft(Eω, tsamples)
    
    χ3 = PhysData.χ3_gas(conf.medium.gas)
    kerr! = Nonlinear.make_kerr!(χ3)
    responses = (kerr!,)

    dens = make_density(conf.medium)

    Pt = zeros(Float64, tsamples)
    Pω = similar(Eω)
    Et = similar(Pt)
    prefac = make_Pnl_prefac(ω, conf.geometry, conf.medium)
    fnl! = let Pt=Pt, Et=Et, FT=FT, IFT=IFT, responses=responses, Pω=Pω, prefac=prefac
        function fnl!(out, Eω, z)
            fill!(Pt, 0)
            Et .= IFT*Eω
            for resp in responses
                resp(Pt, Et)
            end
            Pω .= (FT*Pt)
            out .= dens(z).*prefac.*Pω
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
    samples = 2^(ceil(Int, log2(trange/δt))) # samples for fine grid (power of 2)
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d", trange/δt, samples)
    δt = trange/samples # actual spacing - keep trange fixed
    δω = 2π/trange # frequency spacing for fine grid
    # Make fine grid
    Nt = collect(range(0, length=samples))
    t = @. (Nt-samples/2)*δt # center on 0
    Nω = collect(range(0, length=Int(samples/2 +1)))
    ω = Nω*δω

    # Make apodisation window
    ωmin = 2π*minimum(f_lims)
    ω_left = ωmin/2
    width_left = ωmin/8
    ω_right = 2π*maximum(f_lims)+2*apod_width
    width_right = apod_width
    window = Maths.errfun_window(ω, ω_left, ω_right, width_left, width_right)

    cropidx = findfirst(x -> x>ω_right+4*width_right, ω)
    cropidx = 2^(ceil(Int, log2(cropidx)))
    ωcrop = ω[1:cropidx]
    δtcrop = π/maximum(ωcrop)
    samplescrop = (cropidx-1)*2
    Ntcrop = collect(range(0, length=samplescrop))
    tcrop = @. (Ntcrop-samplescrop/2)*δtcrop

    Logging.@info @sprintf("Grid: samples %d, ωmax %.2e",
                           length(t), maximum(ω))
    return t, ω, window, cropidx, ωcrop, tcrop
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
    energy *= PhysData.c*PhysData.ε_0*Aeff/2
    It .*= input.energy/energy

    ω0 = 2π*PhysData.c/input.wavelength
    Et = @. sqrt(It)*cos(ω0*t)

    return FFTW.rfft(Et)
end

import PyPlot: pygui, plt
function run(config)
    pygui(true)
    t, ω, window, cropidx, ωcrop, tcrop = make_grid(config.grid)

    Eω = make_init(t, config.input)
    Et = FFTW.irfft(Eω, length(t))

    linop = make_linop(ω, config.grid.referenceλ, config.geometry, config.medium)
    fnl! = make_fnl(ω, Eω, Et, config)

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

    Eout = Eout[1:cropidx, :]
    Etout = FFTW.irfft(Eout, length(tcrop), 1).*length(tcrop)./length(t)

    return ωcrop, tcrop, zout, Eout, Etout
end

end # module
