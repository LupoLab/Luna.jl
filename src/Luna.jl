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
include("Modes.jl")


function make_linop(grid, βfun, αfun, frame_vel)
    β = .-βfun(grid.ω, 1, 1, 0)
    α = αfun(grid.ω, 1, 1, 0)
    β1 = -1/frame_vel(0)
    return @. im*(β-β1*grid.ω) - α/2
end

function make_fnl(grid, densityfun, normfun, responses)
    cropidx = length(grid.ω)
    tsamples = length(grid.to)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(Float64, length(grid.to))
    FT = FFTW.plan_rfft(zeros(tsamples))
    IFT = FFTW.plan_irfft(Eωo, tsamples)

    scalefac = (length(grid.ωo)-1)/(length(grid.ω)-1)

    Pto = zeros(Float64, tsamples)
    Pωo = similar(Eωo)
    function fnl!(Pω, Eω, z)
        fill!(Pto, 0)
        to_time!(Eto, Eω, Eωo, IFT)
        for resp in responses
            resp(Pto, Eto)
        end
        @. Pto *= grid.towin
        to_freq!(Pω, Pωo, Pto, FT)
        Pω .*= grid.ωwin.*densityfun(z).*(-im.*grid.ω./2)./normfun(z)
    end
    return fnl!
end

function to_time!(Eto, Eω, Eωo, IFT)
    N = size(Eω, 1)
    No = size(Eωo, 1)
    fill!(Eωo, 0)
    Eωo[1:N] = Eω * (No-1)/(N-1)
    Eto .= IFT*Eωo
end

function to_freq!(Pω, Pωo, Pto, FT)
    N = size(Pω, 1)
    No = size(Pωo, 1)
    Pωo .= FT*Pto
    Pω .= Pωo[1:N] .* (N-1)/(No-1)
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
             βfun, αfun, frame_vel, normfun, energyfun, densityfun, inputs, responses)

    Eω = make_init(grid, inputs, energyfun)
    Et = FFTW.irfft(Eω, length(grid.t))

    linop = make_linop(grid, βfun, αfun, frame_vel)
    fnl! = make_fnl(grid, densityfun, normfun, responses)

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
