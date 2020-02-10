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

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    xo1 = Array{Float64}(undef, length(grid.to))
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun, aeff)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.PATIENT)
    Eω = make_init(grid, inputs, energyfun, FT)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.PATIENT)
    xo1 = Array{ComplexF64}(undef, length(grid.to))
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun,aeff)
    Eω = make_init(grid, inputs, energyfun, FT)
    Utils.saveFFTwisdom()
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
    Utils.loadFFTwisdom()
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=FFTW.PATIENT)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.PATIENT)
    xo1 = Array{Float64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Utils.saveFFTwisdom()
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
    Utils.loadFFTwisdom()
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=FFTW.PATIENT)
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.PATIENT)
    xo1 = Array{ComplexF64}(undef, length(grid.to), npol)
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModal(grid, Modes.dimlimits(modes[1]), Exys, FTo1,
                                 responses, densityfun, components, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    Utils.saveFFTwisdom()
    Eω, transform, FT
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
             linop, transform, FT, output; max_dz=Inf)


    Et = FT \ Eω

    z = 0
    dz = 1e-3

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
        transform, linop, Eω, z, dz, grid.zmax, stepfun=stepfun, max_dt=max_dz)
end

end # module
