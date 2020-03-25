module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Base.Threads: @threads
import Base: length

"""
    @scaninit(name="scan")

Initialise a scan for this file by parsing command line arguments and creating a Scan object.
The scan name (added to file names via @ScanHDF5Output) can also be given.

# Examples
```julia
@scaninit "energy_scan_5bar"
```
"""
macro scaninit(name="scan")
    quote
        args = parse_scan_cmdline()
        $(esc(:__SCAN__)) = Scan($name, args)
    end
end

function parse_scan_cmdline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--range"
            help = "Linear range of scan indices to execute"
            arg_type = UnitRange{Int}
        "--batch"
            help = "Batch index to execute"
            arg_type = Tuple{Int, Int}
        "--cirrus"
            help = "Make job script for cirrus to run in given number of batches, do not run any simulations."
            arg_type = Int
        "--condor"
            help = "Make job submission script for HTCondor. Do not run any simulations."
            arg_type = Int
        "--local"
            help = "Simply run the scan locally"
            action = :store_true
        "--parallel", "-p"
            help = "Run multi-threaded"
            action = :store_true
    end
    args = parse_args(s)
    for k in keys(args)
        isnothing(args[k]) && delete!(args, k)
    end
    if haskey(args, "batch") && haskey(args, "range")
        error("Only one of range or batch can be given.")
    end
    if haskey(args, "range") && args["local"]
        error("Option --local and range cannot both be given")
    end
    if args["local"] && haskey(args, "cirrus")
        error("Local and cirrus-setup options cannot both be given.")
    end
    return args
end

# Enable parsing of command-line arguments of the form "1:5" to a UnitRange
parse_item(::Type{UnitRange{Int}}, x::AbstractString) = eval(Meta.parse(x))

# Enable parsing of command-line arguments of the form"1,5" to a Tuple of integers
parse_item(::Type{Tuple{Int, Int}}, x::AbstractString) = Tuple(parse(Int, xi) for xi in split(x, ","))


"""
    Scan

Struct to contain information about a scan, including the arrays to be scanned over.

When an array is annotated by `@scanvar`, it is added to the `Scan` and then the cartesian
product (all possible combinations) of the annotated arrays is computed. This product has
contains (`N1 * N2 * N3...`) entries, where `N1` etc are the lengths of the arrays
to be scanned over. Each entry is unique and itself has one entry for each annotated array.

The cartesian product is split up into several arrays, one per annotated array, each also
of length (`N1 * N2 * N3...`)--these contain the values that a given scan variable should take
for each simulation in the scan. They are saved in in `values`, an `IdDict` that maps
**from an annoted array itself** to the (`N1 * N2 * N3...`)-length array of values.
By then indexing into this value array using `__SCANIDX__`, we find the value that this scan
variable should take for this particular simulation.

The first array added using `@scanvar` varies the fastest, all other fields of 
`Scan.values` will contain repeated entries.
"""
mutable struct Scan
    name::AbstractString # Name for the scan
    mode::Symbol # :cirrus, :condor, :local, :batch or :range
    batch::Tuple{Int, Int} # batch index and number of batches
    idcs # Array or iterator of indices to be run on this execution
    vars::Dict{Symbol, Any} # maps from variable names to arrays
    arrays # Array of arrays, each element is one of the arrays to be scanned over
    values::IdDict # maps from each scan array to the expanded array of values
    parallel::Bool # true if scan is being run multi-threaded
end

# Constructor taking parsed command line arguments.
function Scan(name, args)
    mode = :setup
    batch = (0, 0)
    idcs = nothing
    if haskey(args, "batch")
        mode = :batch
        batch = args["batch"] # store batch index
    elseif haskey(args, "range")
        mode = :range
        idcs = args["range"]
    elseif args["local"]
        mode = :local
    elseif haskey(args, "cirrus")
        mode = :cirrus
        batch = (0, args["cirrus"])
    elseif haskey(args, "condor")
        mode = :condor
        batch = (0, args["condor"])
    else
        error("One of batch, range, local or cirrus options must be given!")
    end
    Scan(name, mode, batch, idcs, Dict{Symbol, Any}(), Array{Any, 1}(), IdDict(), args["parallel"])
end

# The length of a scan is the product of the lengths of the arrays to be scanned over
length(s::Scan) = (length(s.arrays) > 0) ? prod([length(ai) for ai in s.arrays]) : 0

"""
    addvar!(s::Scan, var::Symbol, arr::AbstractArray)

Add an array to a scan, saving both the symbol given and the array, then re-make the 
cartesian product.

This function is used internally by `@scanvar`
"""
function addvar!(s::Scan, var::Symbol, arr::AbstractArray)
    push!(s.arrays, arr)
    s.vars[var] = arr
    makearray!(s)
end

"""
    getval(s::Scan, var::Symbol, scanidx::Int)

Get the value of a scan variable (identified by a symbol) for a given scan index.

This function is used in e.g. `Output.@ScanHDF5Output` to get the values used for one
particular simulation.
"""
getval(s::Scan, var::Symbol, scanidx::Int) = s.values[s.vars[var]][scanidx]

"""
    makearray!(s::Scan)

Make the cartesian product array containing all possible combinations of the scan arrays,
as well as the array of indices that are to be used in the current run of the script.
"""
function makearray!(s::Scan)
    combos = vec(collect(Iterators.product(s.arrays...)))
    s.values = IdDict()
    for (i, a) in enumerate(s.arrays)
        # The keys in the IdDict s.values are the arrays themselves
        # Each field s.values[a] contains an array of length (N1*N2*N3...)
        s.values[a] = [ci[i] for ci in combos]
    end
    if s.mode == :batch
        # Running one batch - create chunks and select desired one.
        linidx = collect(1:length(s))
        chunkidx, Nchunks = s.batch
        chs = chunks(linidx, Nchunks)
        s.idcs = chs[chunkidx]
    elseif s.mode == :local
        # Running everything locally - idcs are simply everything.
        s.idcs = collect(1:length(s))
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

"""
    @scanvar

Add an array as a variable to be scanned over.

# Examples
Create an array and add it to the scan simultaneously:
```julia
@scanvar energy = range(0.1e-6, 1.5e-6, length=16)
```
Create an array first and add it to the scan later:
```julia
τ = range(25e-15, 35e-15, length=11)
@scanvar τ
```
"""
macro scanvar(expr)
    global ex = expr
    if isa(expr, Symbol)
        # existing array being added
        q = quote
            addvar!($(esc(:__SCAN__)), ex, $(esc(:($expr))))
        end
        return q
    end
    expr.head == :(=) || error("@scanvar must be applied to an assignment expression or variable")
    global lhs = expr.args[1]
    isa(lhs, Symbol) || error("@scanvar expressions must assign to a variable")
    quote
        $(esc(expr)) # First, simply execute the assignment
        addvar!($(esc(:__SCAN__)), lhs, $(esc(:($lhs)))) # now add the resulting array to the Scan
    end
end

"""
    interpolate!(ex::Expr)

Recursively interpolate scan variables into a scan expression.

# Examples
```jldoctest
julia> for i in eachindex(ARGS)
    pop!(ARGS)
end
julia> push!(ARGS, "--local")
julia> __SCANIDX__ = 1
julia> @scaninit
julia> @scanvar energy = range(0.1e-6, 1.5e-6, length=16);
julia> ex = Expr(:\$, :energy)
:(\$(Expr(:\$, :energy)))
julia> exi = interpolate!(ex)
:((__SCAN__.values[energy])[__SCANIDX__])
julia> eval(exi)
1.0e-7
```
"""
function interpolate!(ex::Expr)
    if ex.head === :($)
        var = ex.args[1]
        if var == :__SCANIDX__
            return :__SCANIDX__
        else
            return :(__SCAN__.values[$var][__SCANIDX__])
        end
    else
        for i in 1:length(ex.args)
            arg = ex.args[i]
            if isa(arg, Expr)
                ex.args[i] = interpolate!(arg)
            end
        end
    end
    return ex
end

"""
    @scan

Run the enclosed expression as a scan.

Depending on `__SCAN__.mode`, different things happen:
* :batch, :range, :local -> run enclosed expression as many times as required, using values
    from the cartesian product `__SCAN__.values`. If `__SCAN__.parallel` is true, these runs
    are done on different threads.
* :cirrus -> make PBS job script for batched run on cirrus
* :condor -> make HTCondor job script for batched run on HWLX0003
"""
macro scan(ex)
    body = interpolate!(ex)
    global script = string(__source__.file)
    quote
        if $(esc(:__SCAN__)).mode == :cirrus
            cirrus_setup($(esc(:__SCAN__)).name, script, $(esc(:__SCAN__)).batch[2])
        elseif $(esc(:__SCAN__)).mode == :condor
            condor_setup($(esc(:__SCAN__)).name, script, $(esc(:__SCAN__)).batch[2])
        else
            if $(esc(:__SCAN__)).parallel
                @threads for $(esc(:__SCANIDX__)) in $(esc(:__SCAN__)).idcs
                    $(esc(body))
                end
            else
                for $(esc(:__SCANIDX__)) in $(esc(:__SCAN__)).idcs
                    $(esc(body))
                end
            end
        end
    end
end

"""
    condor_setup(name, script, batches)

Make and save HTCondor job script for a scan named `name` contained in the script located at
`script` which is to be run in batches.
"""
function condor_setup(name, script, batches)
    cmd = split(string(Base.julia_cmd()))[1]
    julia = strip(cmd, ['`', '\''])
    lines = [
        "executable = $julia",
        """arguments = "$(basename(script)) --batch \$\$([\$(Process)+1]),$batches" """,
        "log = $name.log.\$(Process)",
        "output = $name.out.\$(Process)",
        "error = $name.err.\$(Process)",
        "request_cpus = 1",
        "queue $batches"
    ]

    fpath = joinpath(dirname(script), "doit.sub")
    println(fpath)
    open(fpath, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
end

"""
    cirrus_setup(name, script, batches)

Make and save PBS job script for a scan named `name` contained in the script located at
`script` which is to be run in batches.
"""
function cirrus_setup(name, script, batches)
    if Sys.iswindows()
        error("--cirrus option must be invoked on the cirrus login node!")
    end
    cmd = split(string(Base.julia_cmd()))[1]
    julia = strip(cmd, ['`', '\''])
    lines = [
        "#!/bin/bash --login",
        "#PBS -N " * name,
        "#PBS -J 0-$(batches-1)",
        "#PBS -V",
        "#PBS -l select=1:ncpus=4",
        "#PBS -l walltime=48:00:00",
        "#PBS -A sc007",
        "",
        "module load gcc",
        "export OMP_NUM_THREADS=4",
        "",
        "cd \$PBS_O_WORKDIR",
        "$julia $(basename(script)) --batch \$((PBS_ARRAY_INDEX+1)),$batches "
    ]

    fpath = joinpath(dirname(script), "doit.sh")
    println(fpath)
    open(fpath, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
end
end