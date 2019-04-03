module Luna
import FFTW
import NumericalIntegration
include("Maths.jl")
include("RK45.jl")
include("PhysData.jl")
include("Capillary.jl")
include("Config.jl")
include("Nonlinear.jl")
include("Ionisation.jl")

function make_linop(ω, refλ, cap::Configuration.Capillary{Configuration.StaticFill})
    β = [0; Capillary.β(cap.radius, ω[2:end], gas=cap.fill.gas, pressure=cap.fill.pressure)]
    α = [0; Capillary.α(cap.radius, ω[2:end])]
    β1 = Capillary.dispersion(1, cap.radius, λ=refλ,
                              gas=cap.fill.gas, pressure=cap.fill.pressure)
    return @. im*(β - β1*ω) - α/2
end

function make_Pnl_prefac(ω, cap::Configuration.Capillary{Configuration.StaticFill})
    β = [0; Capillary.β(cap.radius, ω[2:end], gas=cap.fill.gas, pressure=cap.fill.pressure)]
    return @. im/(2*PhysData.ε_0*PhysData.c^2)*ω^2/β
end

function make_density(cap::Configuration.Capillary{Configuration.StaticFill})
    return z -> PhysData.std_dens * cap.fill.pressure
end

function make_fnl(ω, ωo, Eω, Et, cropidx, conf)
    Eωo = zeros(ComplexF64, size(ωo))
    tsamples = Int((length(ωo)-1)*2)
    FT = FFTW.plan_rfft(zeros(Float64, tsamples))
    # IFT = inv(FT)
    IFT = FFTW.plan_irfft(Eωo, tsamples)
    
    χ3 = PhysData.χ3_gas(conf.geometry.fill.gas)
    kerr = let χ3=χ3
        E -> Nonlinear.kerr(E, χ3)
    end
    responses = (kerr,)

    dens = make_density(conf.geometry)


    Pto = zeros(Float64, tsamples)
    Pωo = similar(Eωo)
    Eto = similar(Pto)
    prefac = make_Pnl_prefac(ω, conf.geometry)
    fnl = let Pto=Pto, Eto=Eto, FT=FT, IFT=IFT, responses=responses, Eωo=Eωo, cropidx=cropidx, Pωo=Pωo
        function fnl(Eω, z)
            fill!(Pto, 0)
            fill!(Eωo, 0)
            fill!(Pωo, 0)
            Eωo[1:cropidx] = Eω
            any(isnan.(Eω)) && error("NaN in Eω")
            Eto .= IFT*Eωo
            any(isnan.(Eto)) && error("NaN in Eto")
            for resp in responses
                Pto .+= resp(Eto)
            end
            any(isnan.(Pto)) && error("NaN in Pto")
            Pωo .= dens(z).* (FT*Pto)
            any(isnan.(Pωo)) && error("NaN in Pωo")
            return Pωo[1:cropidx]
        end
    end
    return fnl    
end

function make_grid(grid::Configuration.Grid)
    # Find required sampling density
    trange = grid.tmax*2
    f_lims = PhysData.c./grid.λ_lims
    ωmax = 2π*maximum(f_lims)
    δt = min(1/(6*maximum(f_lims)), grid.δt) # 6x maximum desired freq, or user-defined
    samples = 2^(ceil(Int, log2(trange/δt))) # make it a power of 2
    δto = trange/samples # keep trange fixed, increase density to reach power of 2
    δωo = 2π/trange
    Nt = collect(range(0, length=samples))
    # fine grid
    to = @. (Nt-samples/2)*δto
    Nω = collect(range(0, length=Int(samples/2 +1)))
    ωo = @. Nω*δωo
    # find smallest power-of-2 section of grid that contains user-desired frequency window
    idx = findfirst(ω -> ω>=ωmax, ωo)
    idx2 = 2^(ceil(Int, log2(idx))) + 1 # make small grid also power of 2 + 1
    # coarse grid
    ω = ωo[1:idx2]
    samples = length(ω) - 1
    δt = π/maximum(ω)
    Nt = collect(range(0, length=2*samples))
    t = @. (Nt-samples)*δt

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)
    return t, ω, to, ωo, idx2
end

function make_init(t, inputs::NTuple{N, T}) where N where T<:Configuration.Input
    out = fill(0.0 + 0.0im, Int(length(t)/2+1))
    for input in inputs
        out .+= make_init(t, input)
    end
    return out
end

function make_init(t, input::Configuration.GaussInput)
    # TODO MODE SCALING!!!
    It = Maths.gauss(t, fwhm=input.duration)
    energy = NumericalIntegration.integrate(t, It, NumericalIntegration.SimpsonEven())
    It .*= input.energy/energy

    ω0 = 2π*PhysData.c/input.wavelength
    Et = @. sqrt(It)*cos(ω0*t)

    return FFTW.rfft(Et)
end

function run(config)
    t, ω, to, ωo, cropidx = make_grid(config.grid)

    Eω = make_init(t, config.input)
    any(isnan.(Eω)) && error("NaN in Eω in init")
    
    Et = FFTW.irfft(Eω, length(t))

    linop = make_linop(ω, config.grid.referenceλ, config.geometry)
    fnl = make_fnl(ω, ωo, Eω, Et, cropidx, config)

    ~all(isfinite.(linop)) && error("linop broken")
    ~all(isfinite.(fnl(Eω, 0))) && error("fnl broken")

    z = 0
    dz = 1e-8
    zmax = config.geometry.length
    saveN = 201

    zout, Eout, steps = RK45.solve_precon(fnl, linop, Eω, z, dz, zmax, saveN)

    return ω, zout, Eout
end

end # module
