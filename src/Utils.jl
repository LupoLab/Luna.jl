module Utils
import Dates
import FFTW
import Logging
import LibGit2
import Pidfile: mkpidlock
import Base: length
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Base.Threads: @threads
import HDF5
import Luna: @hlock

function git_commit()
    try
        repo = LibGit2.GitRepo(lunadir())
        commit = string(LibGit2.GitHash(LibGit2.head(repo)))
        LibGit2.isdirty(repo) && (commit *= " (dirty)")
        return commit
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

function git_branch()
    try
        repo = LibGit2.GitRepo(lunadir())
        n = string(LibGit2.name(LibGit2.head(repo)))
        branch = split(n, "/")[end]
        return branch
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

srcdir() = dirname(@__FILE__)

lunadir() = dirname(srcdir())

datadir() = joinpath(srcdir(), "data")

cachedir() = joinpath(homedir(), ".luna")

function sourcecode()
    src = dirname(@__FILE__)
    luna = dirname(src)
    out = "#= Date: $(Dates.now())\n"
    out *= "git branch: $(git_branch())\n"
    out *= "git commit: $(git_commit())\n"
    out *= "hostname: $(gethostname())\n"
    out *= "=#"
    for folder in (src, luna)
        for obj in readdir(folder)
            if isfile(joinpath(folder, obj))
                if split(obj, ".")[end] in ("md", "jl", "txt", "toml") # avoid binary files
                    out *= "\n" * "#" * "="^8 * obj * "="^8 * "#" * "\n"^2
                    open(joinpath(folder, obj), "r") do file
                        out *= read(file, String)
                    end
                end
            end
        end
    end
    return out
end

function loadFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache")
    lockpath = joinpath(cachedir(), "FFTWlock")
    if isfile(fpath)
        Logging.@info("Found FFTW wisdom at $fpath")
        pidlock = mkpidlock(lockpath)
        ret = FFTW.import_wisdom(fpath)
        close(pidlock)
        success = (ret != 0)
        Logging.@info(success ? "FFTW wisdom loaded" : "Loading FFTW wisdom failed")
        return success
    else
        Logging.@info("No FFTW wisdom found")
        return false
    end
end

function saveFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache")
    lockpath = joinpath(cachedir(), "FFTWlock")
    pidlock = mkpidlock(lockpath)
    isfile(fpath) && rm(fpath)
    isdir(cachedir()) || mkpath(cachedir())
    FFTW.export_wisdom(fpath)
    close(pidlock)
    Logging.@info("FFTW wisdom saved to $fpath")
end

function save_dict_h5(fpath, d, force=false, rmold=false)
    if isfile(fpath) & rmold
        rm(fpath)
    end

    function dict2h5(k::AbstractString, v, parent)
        if HDF5.exists(parent, k) & !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = v
    end

    function dict2h5(k::AbstractString, v::Nothing, parent)
        if HDF5.exists(parent, k) & !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = Float64[]
    end

    function dict2h5(k::AbstractString, v::AbstractDict, parent)
        if !HDF5.exists(parent, k)
            subparent = HDF5.g_create(parent, k)
        else
            subparent = parent[k]
        end
        for (kk, vv) in pairs(v)
            dict2h5(kk, vv, subparent)
        end
    end
    
    @hlock HDF5.h5open(fpath, "cw") do file
        for (k, v) in pairs(d)
            dict2h5(k, v, file)
        end
    end
end

function load_dict_h5(fpath)
    isfile(fpath) || error("Error loading file $fpath: file does not exist")

    function h52dict(x::HDF5.HDF5Dataset)
        return read(x)
    end

    function h52dict(x::Union{HDF5.HDF5Group, HDF5.HDF5File})
        dd = Dict{String, Any}()
        for n in names(x)
            dd[n] = h52dict(x[n])
        end
        return dd
    end

    d = @hlock HDF5.h5open(fpath) do file
        h52dict(file)
    end
end       


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

parse_item(::Type{UnitRange{Int}}, x::AbstractString) = eval(Meta.parse(x))
parse_item(::Type{Tuple{Int, Int}}, x::AbstractString) = Tuple(parse(Int, xi) for xi in split(x, ","))


"Struct containing the scan arrays.

idcs are indices into `Scan.values`, which contains
the cartesian product of all of the arrays that are to be scanned over.
`Scan.values` is an `IdDict`, and each field in `values` contains (`N1 * N2 * N3...`) entries,
where `N1` etc are the lengths of the arrays to be scanned over. Each entry in each field of
`Scan.values` is the value of that scan variable for a particular run.
Taken together, the fields of `Scan.values` contain all possible combinations of the arrays.
The first array added using `@scanvar` varies the fastest, all other fields of 
`Scan.values` will contain repeated entries."
mutable struct Scan
    name # Name for the scan
    mode::Symbol # :cirrus, :local, :batch or :range
    batch::Tuple{Int, Int} # batch index and number of batches
    idcs # Array or iterator of indices to be run on this execution
    vars # Dict{Symbol, Any} -> maps from variable names to arrays
    arrays # Array of arrays, each element is one of the arrays to be scanned over
    values # Dictionary mapping from each scan array to the expanded array of values
    parallel # boolean, true if scan is being run multi-threaded
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

length(s::Scan) = (length(s.arrays) > 0) ? prod([length(ai) for ai in s.arrays]) : 0

"Add a variable to a scan. Adds the array to the list of scan arrays, and re-makes the
cartesian product."
function addvar!(s::Scan, var, arr)
    push!(s.arrays, arr)
    s.vars[var] = arr
    makearray!(s)
end

"Make the cartesian product array containing all possible combinations of the scan arrays."
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

getval(s::Scan, var::Symbol, scanidx::Int) = s.values[s.vars[var]][scanidx]


"Split array a into n chunks, spreading the entries of a evenly."
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

"Macro to add to an array assignment.
    e.g.
        `@scanvar x = 1:10`
    adds the variable `x` to be scanned over."
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

"Recursively interpolate scan variables into a scan expression."
function interpolate!(ex)
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

"Run the enclosed expression as a scan. Interpolates the scan variables into the expression
and then runs the expression as many times as required by the `start` and `stop` arguments."
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