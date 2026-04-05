module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Logging: @info, @warn
import Printf: @sprintf
import Base: length, size
import Luna: Utils
import FileWatching.Pidfile: mkpidlock
import HDF5
import Distributed: @spawnat, addprocs, rmprocs, fetch, Future, @everywhere
import Dates

"""
    AbstractExec

Abstract supertype for scan execution modes.
"""
abstract type AbstractExec end

"""
    LocalExec

Execution mode to simply run the whole scan in the current Julia session in a `for` loop.
"""
struct LocalExec <: AbstractExec end

"""
    RangeExec(range)

Execution mode to run a subsection of the scan, given by a `UnitRange`, in the current Julia session.
"""
struct RangeExec <: AbstractExec
    r::UnitRange{Int}
end


"""
    BatchExec(Nbatches, batch)

Execution mode to divide the scan into `Nbatches` chunks and run only the given `batch`.
"""
struct BatchExec <: AbstractExec
    Nbatches::Int
    batch::Int
end


"""
    QueueExec(nproc=0, queuefile="")

Execution mode to run a scan using a file-based queueing system. Can be run in multiple separate
Julia sessions, or can spawn `nproc` subprocesses which then take items from the queue to run.

Possible values for `nproc` are:
- `0`: run only in the current Julia process
- `n > 0`: spawn `n` subprocesses and run on these
- `-1`: spawn as many subprocesses as the number of logical cores on the CPU
    (`Base.Sys.CPU_THREADS`)

If `queuefile` is given, the queuefile is stored at that path. If omitted, the queuefile is
stored in `Utils.cachedir()`. Note that the queuefile is deleted at the end of the scan.
"""
struct QueueExec <: AbstractExec
    nproc::Int
    queuefile::String
end

QueueExec(nproc=0) = QueueExec(nproc, "")

"""
    CondorExec(scriptfile, ncores)

Execution mode which submits a scan to an HTCondor queue system claiming `ncores` cores.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct CondorExec <: AbstractExec
    scriptfile::String
    ncores::Int
end

"""
    SlurmExec(scriptfile, ncores; memory="", project=dirname(Base.active_project()), nthreads=1, workdir="")

Execution mode which submits a scan to a Slurm queue system as an array job with `ncores`
array tasks.

# Keyword arguments
- `memory::String`: Memory per task, e.g. `"24G"`. Sets `#SBATCH --mem` and automatically
  derives `--heap-size-hint` (at 80% of `memory`) for the Julia process. Supports suffixes
  `K`, `M`, `G`, `T`.
- `project::String`: Path to a Julia project environment. Defaults to the currently active
  project (`dirname(Base.active_project())`), so Slurm workers use the same environment as
  the submission script. Pass `""` to omit the `--project` flag.
- `nthreads::Int`: Number of threads per array task (default `1`). Sets
  `#SBATCH --cpus-per-task` and exports `JULIA_NUM_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` in the job script. The default of `1`
  prevents over-subscription when many array tasks run concurrently on a shared node.
- `workdir::String`: Working directory for the Slurm job. The generated `.sh` script,
  stdout/stderr files, and queue file are all placed here. If `""` (the default), a
  subdirectory `<scanname>_slurm` is automatically created inside the script's directory.
  Pass an explicit path to use a custom directory.

# Generated job script
The generated SBATCH script includes:
- `ulimit -v unlimited` to prevent Julia startup crashes from restrictive virtual memory
  limits (this does **not** bypass Slurm's cgroup `--mem` enforcement on physical RAM).
- Thread-pinning environment variable exports (`JULIA_NUM_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).
- The full path to the current Julia binary (from `Base.julia_cmd()`) rather than bare
  `julia`, ensuring the same Julia version is used on compute nodes.
- All paths (Julia binary, `--chdir` directory, `--project` path) are quoted to handle
  spaces in paths.
- Each array task runs the script with `--queue`, so internally a [`QueueExec`](@ref) is
  used for file-based load balancing across array tasks.

!!! note
    `scriptfile` must **always** be `@__FILE__`

# Examples
```julia
# Minimal: uses current project, 1 thread per task, no memory limit
# Job files go into <scriptdir>/my_scan_slurm/
scan = Scan("my_scan", SlurmExec(@__FILE__, 8); energy=energies)

# Full: 24 GB per task, custom project, 2 threads, custom workdir
scan = Scan("my_scan", SlurmExec(@__FILE__, 8; memory="24G", project="/path/to/env",
                                  nthreads=2, workdir="/tmp/my_slurm_run");
            energy=energies)
```
"""
struct SlurmExec <: AbstractExec
    scriptfile::String
    ncores::Int
    memory::String
    project::String
    nthreads::Int
    workdir::String
end

function SlurmExec(scriptfile, ncores; memory="", project=dirname(Base.active_project()), nthreads=1, workdir="")
    SlurmExec(scriptfile, ncores, memory, project, nthreads, workdir)
end

"""
    SSHExec(localexec, scriptfile, hostname, subdir; files=String[])

Execution mode which transfers the `scriptfile` file to the host given by `hostname` via SSH
and executes the scan on that host with a mode defined by `localexec`. `subdir` gives the
subdirectory (relative to the home directory) where scans are stored on the remote host. A
subfolder with automatically chosen name will be created in `subdir` to store this scan.

Optional keyword argument `files` is a list of auxiliary files to transfer to the remote host
along with the script. These files will be placed in the same directory as the scan script.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct SSHExec{eT} <: AbstractExec
    localexec::eT
    scriptfile::String
    hostname::String
    subdir::String
    files::Vector{String}
end

function SSHExec(le::CondorExec, hostname, subdir; files=String[])
    SSHExec(le, le.scriptfile, hostname, subdir, files)
end

function SSHExec(le::SlurmExec, hostname, subdir; files=String[])
    SSHExec(le, le.scriptfile, hostname, subdir, files)
end

struct Scan{eT}
    name::String
    variables::Vector{Symbol}
    arrays::Vector
    exec::eT
end

"""
    Scan(name; kwargs...)
    Scan(name, ex::AbstractExec; kwargs...)

Create a new `Scan` with name `name` and variables given as keyword arguments. The execution
mode `ex` can be given directly or via command-line arguments to the script. **If given,
command-line arguments overwrite any explicitly passed execution mode.**

If neither an explicit execution mode nor command-line arguments are given,
`ex` defaults to `LocalExec`, i.e. running the whole scan locally in the current Julia process.
"""
function Scan(name, cmdlineargs::Vector{String}=ARGS; kwargs...)
    Scan(name, makeexec(cmdlineargs); kwargs...)
end

function Scan(name, ex::AbstractExec; kwargs...)
    if !isempty(ARGS)
        cmdlineargs = copy(ARGS)
        # remove command-line arguments to avoid infinite recursion:
        [pop!(ARGS) for _ in eachindex(ARGS)]
        return Scan(name, cmdlineargs; kwargs...)
    end
    variables = Symbol[]
    arrays = Vector[]
    for (var, arr) in kwargs
        push!(variables, var)
        push!(arrays, arr)
    end
    Scan(name, variables, arrays, ex)
end

length(s::Scan) = (length(s.arrays) == 0) ? 0 : prod(length, s.arrays)
size(s::Scan) = (length(s.arrays) == 0) ? (0,) : Tuple(length.(s.arrays))

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
    prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi
                   filepath=makefilename(scan, scanidx))
end
```
"""
makefilename(scan::Scan, scanidx) = makefilename(scan.name, scanidx)
makefilename(name::AbstractString, scanidx) = @sprintf("%s_%05d.h5", name, scanidx)

function makeexec(args::Vector{String})
    isempty(args) && return LocalExec()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--local", "-l"
            help = "Execute the whole scan locally."
            action = :store_true
        "--range", "-r"
            help = "Linear range of scan indices to execute."
            arg_type = UnitRange{Int}
        "--batch", "-b"
            help = "Number of batches and batch index to execute"
            arg_type = Tuple{Int, Int}
        "--queue", "-q"
            help = """Use a file-based queue to execute the scan. Can be run in parallel
                      using the --procs/-p option."""
            action = :store_true
        "--procs", "-p"
            help = """Number of processes to use with queued execution. If 0, use only the
                      main Julia instance. If -1, use as many processes as the machine
                      has logical cores."""
            arg_type = Int
            default = 0
    end
    args = parse_args(args, s)
    for k in keys(args)
        isnothing(args[k]) && delete!(args, k)
    end
    args["local"] && return LocalExec()
    haskey(args, "range") && return RangeExec(args["range"])
    haskey(args, "r") && return RangeExec(args["r"])
    haskey(args, "batch") && return BatchExec(args["batch"]...)
    haskey(args, "b") && return BatchExec(args["b"]...)
    args["queue"] && return QueueExec(args["procs"])
    error("Command-line arguments do not define a valid execution mode.")
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
    prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi)
end
```
"""
function runscan(f, scan::Scan{LocalExec})
    out = Any[]
    for (scanidx, args) in enumerate(Iterators.product(scan.arrays...))
        logiter(scan, scanidx, args)
        try
            push!(out, f(scanidx, args...))
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
    out
end

function runscan(f, scan::Scan{RangeExec})
    out = Any[]
    combos = vec(collect(Iterators.product(scan.arrays...)))
    for (scanidx, args) in enumerate(combos[scan.exec.r])
        logiter(scan, scanidx, args)
        try
            push!(out, f(scanidx, args...))
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
    out
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
        try
            f(scanidx, args...)
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
end

function runscan(f, scan::Scan{QueueExec})
    if scan.exec.nproc == 0
        _runscan(f, scan)
    else
        nproc = (scan.exec.nproc == -1) ? Base.Sys.CPU_THREADS : scan.exec.nproc
        procs = addprocs(nproc)
        @everywhere eval(:(using Luna))
        futures = Future[]
        for p in procs
            fut = @spawnat p _runscan(f, scan)
            push!(futures, fut)
        end
        for fut in futures
            fetch(fut)
        end
        rmprocs(procs)
    end
end

function _runscan(f, scan::Scan{QueueExec})
    if isempty(scan.exec.queuefile)
        h = string(hash(scan.name); base=16)
        qfile = "qfile_$h.h5"
    else
        qfile = scan.exec.queuefile
    end
    lockpath = qfile*"_lock"

    combos = vec(collect(Iterators.product(scan.arrays...)))
    qfile_created = false
    while true
        mkpidlock(lockpath; stale_age=120) do
            # first process to catch the pidlock creates the queue file
            if ~isfile(qfile)
                if qfile_created
                    # The queue file was already created, so another process
                    # must have completed the scan and removed it. Signal
                    # that we should stop by setting scanidx to nothing.
                    global scanidx = nothing
                    global qdata = fill(2, length(scan))
                    return
                end
                HDF5.h5open(qfile, "cw") do file
                    file["qdata"] = zeros(Int, length(scan))
                end
            end
            qfile_created = true
            # read the queue data
            global qdata = HDF5.h5open(qfile) do file
                read(file["qdata"])
            end
            # find the first index which is neither done nor in progress
            global scanidx = findfirst(qdata) do qi
                qi == 0
            end
            if ~isnothing(scanidx)
                # mark the index as in progress
                HDF5.h5open(qfile, "r+") do file
                    file["qdata"][scanidx] = 1
                end
            end
        end # release pidlock
        if isnothing(scanidx) # no scan points left to start
            if all(qdata .> 1) # completely done--either all done or failed
                rm(qfile; force=true) # remove the queue file
            end
            break # break out of the loop
        end
        logiter(scan, scanidx, combos[scanidx])
        code = 2 # code for finished successfully
        try
            f(scanidx, combos[scanidx]...) # run scan function
        catch e
            code = 3 # code for failed
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        mkpidlock(lockpath; stale_age=10) do # acquire lock on qfile again
            if isfile(qfile)
                HDF5.h5open(qfile, "r+") do file
                    file["qdata"][scanidx] = code # mark as done/failed
                end
            end
        end
        Base.GC.gc()
    end
end

"""
    _parse_memory_for_heap_hint(memory::String) -> String

Derive a `--heap-size-hint` value at ~80% of the Slurm `--mem` value.
Supports suffixes `K`, `M`, `G`, `T` (case-insensitive). Returns an empty
string if `memory` cannot be parsed.
"""
function _parse_memory_for_heap_hint(memory::String)
    m = match(r"^(\d+)\s*([KMGT]?)$"i, strip(memory))
    isnothing(m) && return ""
    value = parse(Int, m.captures[1])
    unit = uppercase(m.captures[2])
    hint = floor(Int, value * 0.8)
    hint < 1 && return ""
    return "$(hint)$(unit)"
end

"""
    _slurm_script_lines(exec::SlurmExec, workdir::String) -> Vector{String}

Build the lines of the SBATCH job script for a `SlurmExec` with the given resolved
`workdir`. This is separated from `runscan` to allow testing the generated script content
without a Slurm installation.
"""
function _slurm_script_lines(exec::SlurmExec, workdir::String)
    cmd = split(string(Base.julia_cmd()))[1]
    julia = strip(cmd, ['`', '\''])
    script = exec.scriptfile
    cores = exec.ncores
    nthreads = exec.nthreads
    lines = [
        "#!/bin/bash",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$nthreads",
        "#SBATCH -o %x_%a.stdout",
        "#SBATCH -e %x_%a.stderr",
        "#SBATCH --array=1-$cores",
        "#SBATCH --chdir \"$workdir\"",
    ]
    if !isempty(exec.memory)
        push!(lines, "#SBATCH --mem=$(exec.memory)")
    end
    # Prevent Julia startup crash from restrictive system ulimit on virtual memory.
    # This does NOT bypass Slurm's cgroup --mem limit (which tracks RSS, not VIRT).
    push!(lines, "ulimit -v unlimited")
    # Pin threads: prevent over-subscription when running concurrent array tasks
    append!(lines, [
        "export JULIA_NUM_THREADS=$nthreads",
        "export OMP_NUM_THREADS=$nthreads",
        "export OPENBLAS_NUM_THREADS=$nthreads",
        "export MKL_NUM_THREADS=$nthreads",
    ])
    # Build Julia command (quote paths to handle spaces)
    juliacmd = "\"$julia\""
    if !isempty(exec.memory)
        heaphint = _parse_memory_for_heap_hint(exec.memory)
        if !isempty(heaphint)
            juliacmd *= " --heap-size-hint=$heaphint"
        end
    end
    if !isempty(exec.project)
        juliacmd *= " --project=\"$(exec.project)\""
    end
    juliacmd *= " $(basename(script)) --queue"
    push!(lines, juliacmd)
    return lines
end

"""
    _resolve_slurm_workdir(exec::SlurmExec, scanname::String) -> String

Resolve the working directory for a Slurm scan. If `exec.workdir` is empty, returns
`joinpath(dirname(exec.scriptfile), "\$(scanname)_slurm")`; otherwise returns `exec.workdir`.
"""
function _resolve_slurm_workdir(exec::SlurmExec, scanname::String)
    if isempty(exec.workdir)
        joinpath(dirname(exec.scriptfile), "$(scanname)_slurm")
    else
        exec.workdir
    end
end

function runscan(f, scan::Scan{SlurmExec})
    script = scan.exec.scriptfile
    name = scan.name
    workdir = _resolve_slurm_workdir(scan.exec, name)
    mkpath(workdir)
    @info "Submitting slurm job for $script running on $(scan.exec.ncores) cores."
    # Adding the --queue command-line argument below means that when running the Slurm job,
    # the SlurmExec is ignored even if explicitly defined inside the script.
    lines = _slurm_script_lines(scan.exec, workdir)
    subfile = joinpath(workdir, "$name.sh")
    @info "Writing job file to $subfile..."
    open(subfile, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
    @info "Submitting job..."
    out = read(`sbatch $subfile`, String)
    @info "Slurm submission output:\n$out"
end

function runscan(f, scan::Scan{CondorExec})
    # make submission file for HTCondor
    cmd = split(string(Base.julia_cmd()))[1]
    julia = strip(cmd, ['`', '\''])
    script = scan.exec.scriptfile
    cores = scan.exec.ncores
    name = scan.name
    @info "Submitting Condor job for $script running on $cores cores."
    # Adding the --queue command-line argument below means that when running the Condor job,
    # the CondorExec is ignored even if explicitly defined inside the script.
    lines = [
        "executable = $julia",
        """arguments = "$(basename(script)) --queue" """,
        "log = $name.log.\$(Process)",
        "output = $name.out.\$(Process)",
        "error = $name.err.\$(Process)",
        "stream_error = True",
        "initialdir = $(dirname(script))",
        "request_cpus = 1",
        "queue $cores"
    ]
    subfile = joinpath(dirname(script), "doit.sub")
    @info "Writing job file to $subfile..."
    open(subfile, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
    @info "Submitting job..."
    out = read(`condor_submit $subfile`, String)
    @info "Condor submission output:\n$out"
end

function changexec(scan, newexec)
    newscan = Scan(scan.name, newexec)
    for (var, arr) in zip(scan.variables, scan.arrays)
        addvariable!(newscan, var, arr)
    end
    newscan
end

function runscan(f, scan::Scan{<:SSHExec})
    if gethostname() == scan.exec.hostname
        # running on the machine defined in the SSH Exec? just run the scan
        runscan(f, changexec(scan, scan.exec.localexec))
    else
        # running somewhere else? submit the job via SSH
        host = scan.exec.hostname
        subdir = scan.exec.subdir
        script = scan.exec.scriptfile
        scriptfile = basename(script)
        name = scan.name
        folder = Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * "_$name"
        @info "Making directory \$HOME/$subdir/$folder"
        read(`ssh $host "mkdir -p \$HOME/$subdir/$folder"`)
        @info "Transferring file..."
        read(`scp $script $host:\~/$subdir/$folder`)
        if length(scan.exec.files) > 0
            @info "Transferring auxiliary files..."
            for fi in scan.exec.files
                read(`scp $fi $host:\~/$subdir/$folder`)
            end
        end
        @info "Running Luna script on remote host $host"
        read(`ssh $host julia \$HOME/$subdir/$folder/$scriptfile`, String)
    end
end

end
