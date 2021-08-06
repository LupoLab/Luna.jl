module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Logging: @info
import Printf: @sprintf
import Base: length
import Luna: @hlock, Utils
import Pidfile: mkpidlock
import HDF5

abstract type AbstractExec end

# execution type to simply run everything locally
struct LocalExec <: AbstractExec end

# execution type to run a specific range of scan indices
struct RangeExec <: AbstractExec
    r::UnitRange{Int}
end

# execution type to divide the scan into batches and run one
struct BatchExec <: AbstractExec
    Nbatches::Int
    batch::Int
end

# execution type to use a file-based queue
struct QueueExec <: AbstractExec
    queuefile::String
end

QueueExec() = QueueExec("")

struct Scan{eT}
    name::String
    variables::Vector{Symbol}
    arrays::Vector
    exec::eT
end

"""
    Scan(name; kwargs...)
    Scan(name, ex::AbstractExec; kwargs...)

Create a new `Scan` with name `name` and variables given as keyword arguments. If the execution
mode `ex` is not given, it is taken from command-line arguments to the script. If no
command-line arguments are given either, `ex` defaults to `LocalExec`, i.e. running the whole
scan locally in the current Julia process.
"""
function Scan(name, cmdlineargs::Vector{String}=ARGS; kwargs...)
    Scan(name, makeexec(cmdlineargs); kwargs...)
end

function Scan(name, ex::AbstractExec; kwargs...)
    variables = Symbol[]
    arrays = Vector[]
    for (var, arr) in kwargs
        push!(variables, var)
        push!(arrays, arr)
    end
    Scan(name, variables, arrays, ex)
end

length(s::Scan) = (length(s.arrays) == 0) ? 0 : prod(length, s.arrays)

"""
    addvariable!(scan, variable::Symbol, array)
    addvariable!(scan; kwargs...)

Add scan variable(s) to the `scan`, either as a single pair of `Symbol` and array, or as a 
sequence of keyword arguments.
"""
function addvariable!(scan, variable::Symbol, array)
    push!(scan.variables, variable)
    push!(scan.arrays, array)
end

function addvariable!(scan; kwargs...)
    for (var, arr) in kwargs
        push!(scan.variables, var)
        push!(scan.arrays, arr)
    end
end

"""
    makefilename(scan, scanidx)

Make an appropriate file name for an `HDF5Output` or `prop_capillary` output for the
`scan` at the current `scanidx`.

# Examples
```
scan = Scan("scan_example"; energy=collect(range(5e-6, 200e-6; length=64)))
runscan(scan) do scanidx, energyi
    prop_capillary(125e-6, 3, :HeJ, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi
                   filepath=makefilename(scan, scanidx))
end
"""
makefilename(scan, scanidx) = @sprintf("%s_%05d.h5", scan.name, scanidx)

function makeexec(args::Vector{String})
    isempty(args) && return LocalExec()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--local", "-l"
            help = "Execute the whole scan locally."
            action = :store_true
        "--range" "-r"
            help = "Linear range of scan indices to execute."
            arg_type = UnitRange{Int}
        "--batch", "-b"
            help = "Number of batches and batch index to execute"
            arg_type = Tuple{Int, Int}
        "--queue", "-q"
            help = "Use a file-based queue to execute the scan"
            action = :store_true
    end
    args = parse_args(s)
    for k in keys(args)
        isnothing(args[k]) && delete!(args, k)
    end
    args["local"] && return LocalExec()
    haskey(args, "range") && return RangeExec(args["range"])
    haskey(args, "r") && return RangeExec(args["r"])
    haskey(args, "batch") && return BatchExec(args["batch"]...)
    haskey(args, "b") && return BatchExec(args["b"]...)
    args["queue"] && return QueueExec()    
end

# Enable parsing of command-line arguments of the form "1:5" to a UnitRange
parse_item(::Type{UnitRange{Int}}, x::AbstractString) = eval(Meta.parse(x))

# Enable parsing of command-line arguments of the form "1,5" to a Tuple of integers
parse_item(::Type{Tuple{Int, Int}}, x::AbstractString) = Tuple(parse(Int, xi) for xi in split(x, ","))

function logiter(scan, scanidx, args)
    logmsg = @sprintf("Running scan: %s (%d points)\nIndex: %05d\nVariables:\n",
                      scan.name, length(scan), scanidx)
    for (variable, value) in zip(scan.variables, args)
        logmsg *= @sprintf("\t%s: %g\n", variable, value)
    end
    @info logmsg
end

function getvalue(scan, variable, scanidx)
    values = vec(collect(Iterators.product(scan.arrays...)))[scanidx]
    idx = findfirst(scan.variables .== variable)
    values[idx]
end

"""
    runscan(f, scan)

Run the function `f` in a scan with arguments defined by the `scan::Scan`.
The function `f` must have the signature `f(scanidx, args...)` where the length of `args`
is the number of variables to be scanned over. Can be used with the `do` block syntax.

The exact subset and order of scan points which is run depends on `scan.exec`, see
[`Scan`](@ref).

# Examples
```
scan = Scan("scan_example"; energy=collect(range(5e-6, 200e-6; length=64)))
runscan(scan) do scanidx, energyi
    prop_capillary(125e-6, 3, :HeJ, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi)
end
````
"""
function runscan(f, scan::Scan{LocalExec})
    for (scanidx, args) in enumerate(Iterators.product(scan.arrays...))
        logiter(scan, scanidx, args)
        f(scanidx, args...)
    end
end

function runscan(f, scan::Scan{RangeExec})
    combos = vec(collect(Iterators.product(scan.arrays...)))
    for (scanidx, args) in enumerate(combos[scan.exec.r])
        logiter(scan, scanidx, args)
        f(scanidx, args...)
    end
end

"""
    chunks(a::AbstractArray, n::Int)

Split array a into n chunks, spreading the entries of a evenly.

# Examples
```jldoctest
julia> a = collect(range(1, length=10));
julia> Scans.chunks(a, 3)
3-element Array{Array{Int64,1},1}:
 [1, 4, 7, 10]
 [2, 5, 8]
 [3, 6, 9]
```
"""
function chunks(a::AbstractArray, n::Int)
    N = length(a)
    done = 0
    out = [Array{eltype(a), 1}() for ii=1:n]
    while done < N
        push!(out[mod(done, n)+1], a[done+1])
        done += 1
    end
    return out
end

function runscan(f, scan::Scan{BatchExec})
    combos = vec(collect(Iterators.product(scan.arrays...)))
    linidx = collect(1:length(scan))
    chs = chunks(linidx, scan.exec.Nbatches)
    idcs = chs[scan.exec.batch]
    scanidcs_this = linidx[idcs]
    combos_this = combos[idcs]
    for (scanidx, args) in zip(scanidcs_this, combos_this)
        logiter(scan, scanidx, args)
        f(scanidx, args...)
    end
end

function runscan(f, scan::Scan{QueueExec})
    if isempty(scan.exec.queuefile)
        h = string(hash(scan.name); base=16)
        qfile = joinpath(Utils.cachedir(), "qfile_$h.h5")
    else
        qfile = scan.exec.queuefile
    end
    lockpath = joinpath(Utils.cachedir(), basename(qfile)*"_lock")

    combos = vec(collect(Iterators.product(scan.arrays...)))
    while true
        mkpidlock(lockpath) do
            # first process to catch the pidlock creates the queue file
            if ~isfile(qfile)
                @hlock HDF5.h5open(qfile, "cw") do file
                    file["qdata"] = zeros(Int, length(scan))
                end
            end
            # read the queue data
            global qdata = @hlock HDF5.h5open(qfile) do file
                read(file["qdata"])
            end
            # find the first index which is neither done nor in progress
            global scanidx = findfirst(qdata) do qi
                qi == 0
            end
            if ~isnothing(scanidx)
                # mark the index as in progress
                @hlock HDF5.h5open(qfile, "r+") do file
                    file["qdata"][scanidx] = 1
                end
            end
        end # release pidlock
        if isnothing(scanidx) # no scan points left to do
            if all(qdata .== 2) # completely done
                rm(qfile) # remove the queue file
            end
            break # break out of the loop
        end
        logiter(scan, scanidx, combos[scanidx])
        f(scanidx, combos[scanidx]...) # run scan function
        mkpidlock(lockpath) do # acquire lock on qfile again
            @hlock HDF5.h5open(qfile, "r+") do file
                file["qdata"][scanidx] = 2 # mark as done
            end
        end
    end
end

end