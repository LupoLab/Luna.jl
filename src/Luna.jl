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
    cropidx = length(ω)
    tsamples = Int((length(ωo)-1)*2)
    Eωo = zeros(ComplexF64, length(ωo))
    FT = FFTW.plan_rfft(zeros(tsamples))
    IFT = FFTW.plan_irfft(Eωo, tsamples)
    
    χ3 = PhysData.χ3_gas(conf.medium.gas)
    kerr! = Nonlinear.make_kerr!(χ3)
    responses = (kerr!,)

    dens = make_density(conf.medium)

    scalefac = (length(ωo)-1)/(length(ω)-1)

    Pt = zeros(Float64, tsamples)
    Pω = similar(Eωo)
    Et = similar(Pt)
    prefac = make_Pnl_prefac(ω, conf.geometry, conf.medium)
    fnl! = let Pt=Pt, Et=Et, FT=FT, IFT=IFT, responses=responses, Pω=Pω, prefac=prefac,
                scalefac=scalefac, Eωo=Eωo, cropidx=cropidx
        function fnl!(out, Eω, z)
            fill!(Pt, 0)
            fill!(Eωo, 0)
            Eωo[1:cropidx] = scalefac*Eω
            Et .= IFT*Eωo
            for resp in responses
                resp(Pt, Et)
            end
            Pω .= (FT*Pt)
            out .= dens(z).*prefac.*Pω[1:cropidx]./scalefac
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
    δto = min(1/(6*maximum(f_lims)), δt) # 6x maximum freq, or user-defined if finer
    samples = 2^(ceil(Int, log2(trange/δto))) # samples for fine grid (power of 2)
    Logging.@info @sprintf("Samples needed: %.2f, samples: %d", trange/δto, samples)
    δto = trange/samples # actual spacing - keep trange fixed
    δωo = 2π/trange # frequency spacing for fine grid
    # Make fine grid
    Nto = collect(range(0, length=samples))
    to = @. (Nto-samples/2)*δto # center on 0
    Nωo = collect(range(0, length=Int(samples/2 +1)))
    ωo = Nωo*δωo

    ωmax = 2π*maximum(f_lims)
    cropidx = findfirst(x -> x>ωmax+4*apod_width, ωo)
    cropidx = 2^(ceil(Int, log2(cropidx)))
    ω = ωo[1:cropidx]
    δt = π/maximum(ω)
    tsamples = (cropidx-1)*2
    Nt = collect(range(0, length=tsamples))
    t = @. (Nt-tsamples/2)*δt

    # Make apodisation window
    ωmin = 2π*minimum(f_lims)
    ω_left = ωmin/2
    width_left = ωmin/8
    ω_right = 2π*maximum(f_lims)+2*apod_width
    width_right = apod_width
    window = Maths.errfun_window(ω, ω_left, ω_right, width_left, width_right)

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)

    Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e",
                           length(t), length(to), maximum(ω), maximum(ωo))
    return t, ω, to, ωo, window
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
    t, ω, to, ωo, window = make_grid(config.grid)

    Eω = make_init(t, config.input)
    Et = FFTW.irfft(Eω, length(t))

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

    zout, Eout, steps = RK45.solve_precon(fnl!, linop, Eω, z, dz, zmax, saveN)

    Etout = FFTW.irfft(Eout, length(t), 1)

    return ω, t, zout, Eout, Etout
end

end # module
