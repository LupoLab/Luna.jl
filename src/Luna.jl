module Luna
import FFTW
import NumericalIntegration
import Logging
import Printf: @sprintf
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("RK45.jl")
include("Capillary.jl")
include("Nonlinear.jl")
include("Ionisation.jl")


function make_linop(grid, βfun, αfun)
    ω0 = 2π*PhysData.c/grid.referenceλ
    idx0 = (grid.ω .== 0)
    β = zero(grid.ω)
    α = zero(grid.ω)
    β[.~idx0] .= .-βfun(grid.ω[.~idx0])
    β[idx0] .= 0
    β1 = -Maths.derivative(βfun, ω0, 1)
    α[.~idx0] .= αfun(grid.ω[.~idx0])
    α[idx0] .= maximum(α[.~idx0])

    return @. im*(β-β1*grid.ω) - α/2
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

function make_fnl(grid, βfun, densityfun, Pnl_prefac, responses)
    cropidx = length(grid.ω)
    tsamples = Int((length(grid.ωo)-1)*2)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(Float64, length(grid.to))
    FT = FFTW.plan_rfft(zeros(tsamples))
    IFT = FFTW.plan_irfft(Eωo, tsamples)

    scalefac = (length(grid.ωo)-1)/(length(grid.ω)-1)

    Pt = zeros(Float64, tsamples)
    Pω = similar(Eωo)
    Et = similar(Pt)
    fnl! = let Pt=Pt, Et=Et, FT=FT, IFT=IFT, responses=responses, Pω=Pω, prefac=Pnl_prefac,
                scalefac=scalefac, Eωo=Eωo, cropidx=cropidx, ωwindow=grid.ωwin, twindow=grid.towin
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
            out .= ωwindow.*densityfun(z).*prefac.*Pω[1:cropidx]./scalefac
        end
    end
    return fnl!
end

function make_init(grid, inputs, energyfun)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for input in inputs
        out .+= scaled_input(grid, input, energyfun)
    end
    return out
end

function scaled_input(grid, input, energyfun)
    Et = input.func(grid.t)
    energy = energyfun(grid.t, Et, input.m, input.n)
    Et_sc = sqrt(input.energy)/sqrt(energy) .* Et
    return FFTW.rfft(Et_sc)
end


function run(grid,
             βfun, αfun, energyfun, densityfun, inputs, responses)

    Eω = make_init(grid, inputs, energyfun)
    Et = FFTW.irfft(Eω, length(grid.t))

    Pnl_prefac = make_Pnl_prefac(grid.ω, βfun)

    linop = make_linop(grid, βfun, αfun)
    fnl! = make_fnl(grid, βfun, densityfun, Pnl_prefac, responses)

    z = 0
    dz = 1e-3
    zmax = grid.zmax
    saveN = 201

    FT = FFTW.plan_rfft(Et)
    IFT = FFTW.plan_irfft(Eω, length(grid.t))

    window! = let window=grid.ωwin, twindow=grid.twin, FT=FT, IFT=IFT, Et=Et
        function window!(Eω)
            Eω .*= window
            Et .= IFT*Eω
            Et .*= twindow
            Eω .= FT * Et
        end
    end

    zout, Eout, steps = RK45.solve_precon(
        fnl!, linop, Eω, z, dz, zmax, saveN, stepfun=window!)

    Etout = FFTW.irfft(Eout, length(grid.t), 1)

    return zout, Eout, Etout
end

"""
Functions/callables needed
β(ω, m, n)
α(ω, m, n) what about α(x, y)?
energy(t, Et, m, n)
Et0(t) shape only, input given as list of _named_ tuples: (func, energy, m, n)
Pnl(t, Et)
density(z) possibly density(x, z, y) or density(ρ, θ, z)?
"""

end # module
