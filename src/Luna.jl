module Luna
import FFTW
import Logging
import LinearAlgebra: mul!, ldiv!
import Random: MersenneTwister

"Lock on the HDF5 library for multi-threaded execution."
const HDF5LOCK = ReentrantLock()
"Macro to wait for and then release HDF5LOCK. Any call to HDF5.jl needs to be
preceeded by @hlock."
macro hlock(expr)
    quote
        try
            lock(HDF5LOCK)
            $(esc(expr))
        finally
            unlock(HDF5LOCK)
        end
    end
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
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
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xo1 = Array{Float64}(undef, length(grid.to))
    FTo1 = FFTW.plan_rfft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=FFTW.PATIENT)
    Eω = make_init(grid, inputs, energyfun, FT)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=FFTW.PATIENT)
    xo1 = Array{ComplexF64}(undef, length(grid.to))
    FTo1 = FFTW.plan_fft(xo1, 1, flags=FFTW.PATIENT)
    transform = NonlinearRHS.TransModeAvg(grid, FTo1, responses, densityfun, normfun)
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

function shotnoise!(Eω, grid::Grid.RealGrid, mode::Modes.AbstractMode; seed=nothing)
    rng = MersenneTwister(seed)
    aeff = Modes.Aeff(mode)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = @. sqrt(2*PhysData.ħ*grid.ω/(PhysData.ε_0*PhysData.c*aeff*δω))
    rFFTamp = sqrt(2π)/2δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += rFFTamp * exp(1im*φ)
end

function shotnoise!(Eω, grid::Grid.EnvGrid, mode::Modes.AbstractMode; seed=nothing)
    rng = MersenneTwister(seed)
    aeff = Modes.Aeff(mode)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    amp[grid.sidx] = @. sqrt(2*PhysData.ħ*grid.ω[grid.sidx]/(PhysData.ε_0*PhysData.c*aeff*δω))
    FFTamp = sqrt(2π)/δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += FFTamp * exp(1im*φ)
end

function shotnoise!(Eω, grid::Grid.RealGrid; seed=nothing)
    rng = MersenneTwister(seed)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = @. sqrt(PhysData.ħ*grid.ω/δω)
    rFFTamp = sqrt(2π)/2δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += rFFTamp * exp(1im*φ)
end

function shotnoise!(Eω, grid::Grid.EnvGrid; seed=nothing)
    rng = MersenneTwister(seed)
    δω = grid.ω[2] - grid.ω[1]
    δt = grid.t[2] - grid.t[1]
    amp = zero(grid.ω)
    amp[grid.sidx] = @. sqrt(PhysData.ħ*grid.ω[grid.sidx]/δω)
    FFTamp = sqrt(2π)/δt*amp
    φ = 2π*rand(rng, size(Eω)...)
    @. Eω += FFTamp * exp(1im*φ)
end


function run(Eω, grid,
             linop, transform, FT, output;
             min_dz=0, max_dz=Inf,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm)


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

    output(Grid.to_dict(grid), group="grid")

    RK45.solve_precon(
        transform, linop, Eω, z, dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm)
end

end # module
