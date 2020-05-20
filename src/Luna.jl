module Luna
import FFTW
import Hankel
import Logging
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

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
settings = Dict{String, Any}("fftw_flag" => FFTW.PATIENT,
                             "fftw_threads" => 0)

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

"""
    set_fftw_threads(nthr)

Set number of threads to be used by FFTW. If set to `0`, the number of threads used by 
FFTW is determined automatically (see [`Utils.FFTWthreads()`](@ref))
"""
function set_fftw_threads(nthr=0)
    settings["fftw_threads"] = nthr
    FFTW.set_num_threads(Utils.FFTWthreads())
end

function __init__()
    set_fftw_threads()
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("Fields.jl")
include("RK45.jl")
include("Modes.jl")
include("LinearOps.jl")
include("Capillary.jl")
include("Antiresonant.jl")
include("RectModes.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")
include("Processing.jl")
include("Plotting.jl")
include("Raman.jl")

export Utils, Scans, Output, Maths, PhysData, Grid, RK45, Modes, Capillary, RectModes,
       Nonlinear, Ionisation, NonlinearRHS, LinearOps, Stats, Polarisation,
       Tools, Plotting, Raman, Antiresonant, Fields, Processing

# for a tuple of TimeFields we assume all inputs are for mode 1
function doinput_sm(grid, inputs::Tuple{Vararg{T} where T <: Fields.TimeField}, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for field in inputs
        out .+= field(grid, FT)
    end
    return out
end

# for a single Fields.TimeField we assume a single input for mode 1
function doinput_sm(grid, inputs::Fields.TimeField, FT)
    doinput_sm(grid, (inputs,), FT)
end

function setup(grid::Grid.RealGrid, densityfun, responses, inputs, βfun!, aeff;
               norm! = NonlinearRHS.norm_mode_average(grid, βfun!, aeff))
    Utils.loadFFTwisdom()
    xo = Array{Float64}(undef, length(grid.to))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, norm!, aeff)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    Eω = doinput_sm(grid, inputs, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, densityfun, responses, inputs, βfun!, aeff;
               norm! = NonlinearRHS.norm_mode_average(grid, βfun!, aeff))
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, norm!, aeff)
    Eω = doinput_sm(grid, inputs, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

# for a tuple of NamedTuple's with tuple fields we assume all is well
function doinput_mm!(Eω, grid, inputs::Tuple{Vararg{T} where T <: NamedTuple{<:Any, <:Tuple{Vararg{Any}}}}, FT)
    for input in inputs
        out = @view Eω[:,input.mode]
        for field in input.fields
            out .+= field(grid, FT)
        end
    end
end

# for a tuple of TimeFields we assume all inputs are for mode 1
function doinput_mm!(Eω, grid, inputs::Tuple{Vararg{T} where T <: Fields.TimeField}, FT)
    doinput_mm!(Eω, grid, ((mode=1, fields=inputs),), FT)
end

# for a single Fields.TimeField we assume a single input for mode 1
function doinput_mm!(Eω, grid, inputs::Fields.TimeField, FT)
    doinput_mm!(Eω, grid, ((mode=1, fields=(inputs,)),), FT)
end

function setup(grid::Grid.RealGrid, densityfun, responses, inputs,
               modes::Modes.ModeCollection, components;
               full=false, norm! = NonlinearRHS.norm_modal(grid))
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    doinput_mm!(Eω, grid, inputs, FTt)
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    xo = Array{Float64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, norm!,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, densityfun, responses, inputs,
               modes::Modes.ModeCollection, components;
               full=false, norm! = NonlinearRHS.norm_modal(grid))
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    doinput_mm!(Eω, grid, inputs, FTt)
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, norm!,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function doinputs_fs!(Eωk, grid, spacegrid::Union{Hankel.QDHT,Grid.FreeGrid}, FT,
                   inputs::Tuple{Vararg{T} where T <: Fields.SpatioTemporalField})
    for field in inputs
        Eωk .+= field(grid, spacegrid, FT)
    end
end

function doinputs_fs!(Eωk, grid, spacegrid::Union{Hankel.QDHT,Grid.FreeGrid}, FT,
                   inputs::Fields.SpatioTemporalField)
    doinputs_fs!(Eωk, grid, spacegrid, FT, (inputs,))
end

function setup(grid::Grid.RealGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    Eωk = q * Eω
    doinputs_fs!(Eωk, grid, q, FT, inputs)
    xo = Array{Float64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(ComplexF64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    Eωk = q * Eω
    doinputs_fs!(Eωk, grid, q, FT, inputs)
    xo = Array{ComplexF64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{Float64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    doinputs_fs!(Eωk, grid, xygrid, FT, inputs)
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
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_fft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    doinputs_fs!(Eωk, grid, xygrid, FT, inputs)
    xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, xygrid, FTo,
                                       responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
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
             min_dz=0, max_dz=grid.zmax/2, init_dz=1e-4, z0=0.0,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm,
             status_period=1)

    Et = FT \ Eω

    function stepfun(Eω, z, dz, interpolant)
        Eω .*= grid.ωwin
        ldiv!(Et, FT, Eω)
        Et .*= grid.twin
        mul!(Eω, FT, Et)
        output(Eω, z, dz, interpolant)
    end

    # check_cache does nothing except for HDF5Outputs
    Eωc, zc, dzc = Output.check_cache(output, Eω, z0, init_dz)
    if zc > z0
        Logging.@info("Found cached propagation. Resuming...")
        Eω, z0, init_dz = Eωc, zc, dzc
    end

    output(Grid.to_dict(grid), group="grid")
    output(simtype(grid, transform, linop), group="simulation_type")
    output(dumps(transform, linop), group="dumps")

    RK45.solve_precon(
        transform, linop, Eω, z0, init_dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm,
        status_period=status_period)
end

end # module
