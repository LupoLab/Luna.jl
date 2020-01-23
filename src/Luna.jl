module Luna
import FFTW
import NumericalIntegration
import Logging
import Printf: @sprintf
import LinearAlgebra: mul!, ldiv!
include("Utils.jl")
include("Maths.jl")
include("Hankel.jl")
include("PhysData.jl")
include("Grid.jl")
include("RK45.jl")
include("Modes.jl")
include("Capillary.jl")
include("RectModes.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("LinearOps.jl")
include("Output.jl")
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")
include("Plotting.jl")

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs)
    xo1 = Array{Float64}(undef, length(grid.to))
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.PATIENT)
    Eω = make_init(grid, inputs, energyfun, FT)
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs)
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{ComplexF64}(undef, length(grid.to))
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
    Eω = make_init(grid, inputs, energyfun, FT)
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    Exys = []
    for mode in modes
        push!(Exys, Modes.Exy(mode))
    end
    if components == :Exy
        npol = 2
    else
        npol = 1
    end
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{Float64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    Exys = []
    for mode in modes
        push!(Exys, Modes.Exy(mode))
    end
    if components == :Exy
        npol = 2
    else
        npol = 1
    end
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.MEASURE)
    xo1 = Array{ComplexF64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Eω, transform, FT
end

function setup(grid::Grid.RealGrid, q::Hankel.QDHT,
               energyfun, densityfun, normfun, responses, inputs)
    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_rfft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    for input in inputs
        Eω .+= scaled_input(grid, input, energyfun, FT)
    end
    Eωk = q * Eω
    xo = Array{Float64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_rfft(xo, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, q::Hankel.QDHT,
               energyfun, densityfun, normfun, responses, inputs)
    xt = zeros(ComplexF64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=FFTW.MEASURE)
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    for input in inputs
        Eω .+= scaled_input(grid, input, energyfun, FT)
    end
    Eωk = q * Eω
    xo = Array{ComplexF64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_fft(xo, 1, flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    Eωk, transform, FT
end

function setup(grid::Grid.RealGrid, FT, x, y,
               energyfun, densityfun, normfun, responses, inputs)
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    for input in inputs
        Eωk .+= scaled_input(grid, input, energyfun, FT)
    end
    xo = Array{Float64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_rfft(xo, (1, 2, 3), flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransFree(grid, FTo, length(y), length(x),
                                       responses, densityfun, normfun)
    Eωk, transform, FTo
end

function setup(grid::Grid.EnvGrid, FT, x, y,
               energyfun, densityfun, normfun, responses, inputs)
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    for input in inputs
        Eωk .+= scaled_input(grid, input, energyfun, FT)
    end
    xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=FFTW.MEASURE)
    transform = NonlinearRHS.TransFree(grid, FTo, length(y), length(x),
                                       responses, densityfun, normfun)
    Eωk, transform, FTo
end

function make_init(grid, inputs, energyfun, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for input in inputs
        out .+= scaled_input(grid, input, energyfun, FT)
    end
    return out
end

function scaled_input(grid, input, energyfun, FT)
    Et = input.func(grid.t)
    energy = energyfun(grid.t, Et)
    Et_sc = sqrt(input.energy)/sqrt(energy) .* Et
    return FT * Et_sc
end

function run(Eω, grid,
             linop, transform, FT, output; max_dz=Inf, init_dz=1e-4)


    Et = FT \ Eω

    z = 0

    window! = let window=grid.ωwin, twindow=grid.twin, FT=FT, Et=Et
        function window!(Eω)
            Eω .*= window
            ldiv!(Et, FT, Eω)
            Et .*= twindow
            mul!(Eω, FT, Et)
        end
    end

    function stepfun(Eω, z, dz, interpolant)
        window!(Eω)
        output(Eω, z, dz, interpolant)
    end

    RK45.solve_precon(
        transform, linop, Eω, z, init_dz, grid.zmax, stepfun=stepfun, max_dt=max_dz)
end

end # module
