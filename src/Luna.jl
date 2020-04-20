module Luna
import FFTW
import Hankel
import Logging
import LinearAlgebra: mul!, ldiv!
import Random: MersenneTwister

"""
    HDF5LOCK

Lock on the HDF5 library for multi-threaded execution.
"""
const HDF5LOCK = ReentrantLock()

"""
    @hlock

Wait for HDF5LOCK, execute the expression, and release H5DFLOCK.

!!! warning
    For thread safety, any call to functions from HDF5.jl needs to be preceeded by @hlock.
"""
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

"""
    Luna.settings

Dictionary of global settings for `Luna`.
"""
settings = Dict{String, Any}("fftw_flag" => FFTW.PATIENT)

"""
    set_fftw_mode(mode)

Set FFTW planning mode for all FFTW transform planning in `Luna`.

Possible values for `mode` are `:estimate`, `:measure`, `:patient`, and `:exhaustive`.
The initial value upon loading `Luna` is `:patient`

# Examples
```jldoctest
julia> Luna.set_fftw_mode(:patient)
0x00000020
```
"""
function set_fftw_mode(mode)
    s = uppercase(string(mode))
    flag = getfield(FFTW, Symbol(s))
    settings["fftw_flag"] = flag
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("RK45.jl")
include("Modes.jl")
include("Capillary.jl")
include("Antiresonant.jl")
include("RectModes.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("LinearOps.jl")
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")
include("Plotting.jl")
include("Raman.jl")

export Utils, Scans, Output, Maths, PhysData, Grid, RK45, Modes, Capillary, RectModes,
       Nonlinear, Ionisation, NonlinearRHS, LinearOps, Stats, Polarisation,
       Tools, Plotting, Raman, Antiresonant

function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    xo = Array{Float64}(undef, length(grid.to))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, normfun, aeff)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    Eω = make_init(grid, inputs, energyfun, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, normfun, aeff)
    Eω = make_init(grid, inputs, energyfun, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.RealGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    xo = Array{Float64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.EnvGrid, energyfun, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], energyfun, FTt)
    end
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.RealGrid, q::Hankel.QDHT,
               energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    for input in inputs
        Eω .+= scaled_input(grid, input, energyfun, FT)
    end
    Eωk = q * Eω
    xo = Array{Float64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, q::Hankel.QDHT,
               energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(ComplexF64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    for input in inputs
        Eω .+= scaled_input(grid, input, energyfun, FT)
    end
    Eωk = q * Eω
    xo = Array{ComplexF64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
               energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{Float64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    for input in inputs
        Eωk .+= scaled_input(grid, input, energyfun, FT)
    end
    xo = Array{Float64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_rfft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, xygrid, FTo,
                                       responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
               energyfun, densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    for input in inputs
        Eωk .+= scaled_input(grid, input, energyfun, FT)
    end
    xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, xygrid, FTo,
                                       responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function make_init(grid, inputs, energyfun, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for input in inputs
        out .+= scaled_input(grid, input, energyfun, FT)
    end
    return out
end

function scaled_input(grid, input, energyfun, FT)
    Et = fixtype(grid, input.func(grid.t))
    energy = energyfun(grid.t,  Et)
    Et_sc = sqrt(input.energy)/sqrt(energy) .* Et
    return FT * Et_sc
end

# Make sure that envelope fields are complex to trigger correct dispatch
fixtype(grid::Grid.RealGrid, Et) = Et
fixtype(grid::Grid.EnvGrid, Et) = complex(Et)

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

linoptype(l::AbstractArray) = "constant"
linoptype(l) = "variable"

gridtype(g::Grid.RealGrid) = "field-resolved"
gridtype(g::Grid.EnvGrid) = "envelope"
gridtype(g) = "unknown"

simtype(g, t, l) = Dict("field" => gridtype(g),
                        "transform" => string(t),
                        "linop" => linoptype(l))

function dumps(t, l)
    io = IOBuffer()
    dump(io, t)
    tr = String(take!(io))
    io = IOBuffer()
    dump(io, l)
    lo = String(take!(io))
    Dict("transform" => tr, "linop" => lo)
end

function run(Eω, grid,
             linop, transform, FT, output;
             min_dz=0, max_dz=grid.zmax/2, init_dz=1e-4,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm,
             status_period=1)

    Et = FT \ Eω

    z = 0.0

    function stepfun(Eω, z, dz, interpolant)
        Eω .*= grid.ωwin
        ldiv!(Et, FT, Eω)
        Et .*= grid.twin
        mul!(Eω, FT, Et)
        output(Eω, z, dz, interpolant)
    end

    output(Grid.to_dict(grid), group="grid")
    output(simtype(grid, transform, linop), group="simulation_type")
    output(dumps(transform, linop), group="dumps")

    RK45.solve_precon(
        transform, linop, Eω, z, init_dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm,
        status_period=status_period)
end

end # module
