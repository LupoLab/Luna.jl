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

When the scan finishes, a marker file (the queue file path with its extension replaced by
`.done`) is created next to the queue file. A process that starts up afterwards and finds this
marker (but no queue file) will stop instead of re-creating the queue and re-running the whole scan — this
matters when array tasks on a cluster start after the scan has already completed. To re-run a
completed scan with the same name, delete this marker first (`SlurmExec` does this
automatically on each submission).
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
    SlurmExec(scriptfile, ncores; memory="", project=<active project>, nthreads=1, procs=0, workdir="", arraymode=:queue, time="", partition="")

Execution mode which submits a scan to a Slurm queue system as an array job with `ncores`
array tasks (`ncores` must be ≥ 1).

To maximise throughput when a cluster caps the number of *running jobs* but offers many
cores, use `procs` to run several concurrent simulations within each array task. With
`SlurmExec(@__FILE__, 10; procs=12)`, 10 array tasks (= 10 jobs) each spawn 12 worker
processes that drain a shared queue, giving 120 concurrent single-threaded simulations. See
the "Execution on Slurm" section of the documentation for a worked example.

# Keyword arguments
- `memory::String`: Memory per task, e.g. `"24G"`. Sets `#SBATCH --mem` and automatically
  derives `--heap-size-hint` (at 80% of `memory`) for the Julia process. Supports suffixes
  `K`, `M`, `G`, `T`. A bare number (e.g. `"24000"`) is treated as megabytes, matching
  Slurm's default convention. Must match the strict format `<digits>[K|M|G|T]` when
  non-empty (no internal whitespace); the value is normalized at construction time
  (stripped and uppercased). An `ArgumentError` is thrown for invalid values.
- `project::String`: Path to a Julia project environment. Defaults to the currently active
  project (`dirname(Base.active_project())`), or `""` if no project is active. Pass `""`
  to omit the `--project` flag. Relative paths are resolved against `dirname(scriptfile)`
  when generating the job script. Must not contain double quotes or newlines.
- `nthreads::Int`: Number of threads per array task (default `1`, must be ≥ 1). Exports
  `JULIA_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` in
  the job script. The default of `1` prevents over-subscription when many simulations run
  concurrently on a shared node.
- `procs::Int`: Number of concurrent worker processes per array task (default `0`, must be
  ≥ 0). Each array task runs `julia … --queue -p <procs>`, spawning `procs` workers that
  pull from the shared file-based queue (see [`QueueExec`](@ref)). This lets one array task
  process many scan points at once, which is the way to use many cores when the cluster
  limits the number of running jobs. The default `0` keeps the previous behaviour: the array
  task processes the queue in a single process. `procs = -1` is **not** supported here (the
  core count must be fixed when the job script is generated). The workers automatically
  inherit the parent's `--project` and a per-worker `--heap-size-hint` of
  `heap-size-hint ÷ procs`, so the whole task stays within `--mem`. `procs` requires
  `arraymode=:queue` (it is incompatible with `:batch`).
- `#SBATCH --cpus-per-task` is set to `max(procs, 1) * nthreads`, i.e. one core per
  concurrent worker times the threads each worker uses. (The coordinating parent process is
  effectively idle while the workers run, so it needs no extra core.)
- `workdir::String`: Working directory for the Slurm job. The generated `.sh` script,
  stdout/stderr files, and queue file are all placed here. If `""` (the default), a
  subdirectory `<scanname>_slurm` is automatically created inside the script's directory.
  Pass an explicit path to use a custom directory. Must not contain double quotes or
  newlines.
- `arraymode::Symbol`: How scan points are distributed across array tasks (default `:queue`).
  - `:queue`: Array tasks dynamically pick up work from a shared file-based queue
    ([`QueueExec`](@ref)). Good when tasks have varying run times.
  - `:batch`: Each array task gets a pre-assigned chunk of scan points (via `--batch`).
    With `ncores == length(scan)`, each task runs exactly one scan point. No queue file
    or file locking is needed, giving complete memory isolation between tasks.
- `time::String`: Wall-clock time limit per array task, setting `#SBATCH --time`. Accepts any
  Slurm time format, e.g. `"30"` (minutes), `"HH:MM:SS"` or `"D-HH:MM:SS"`. Empty (the
  default) omits the directive, but note that many clusters reject a job with no time limit
  (`sbatch: error: Requested time limit is invalid`), so set it to stay within the cluster's
  maximum runtime.
- `partition::String`: Slurm partition/queue, setting `#SBATCH --partition`. Empty (the
  default) omits the directive and lets Slurm pick the default partition.

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
- Each array task runs the script with `--queue` or `--batch` depending on `arraymode`,
  using the corresponding execution mode internally.

!!! note
    `scriptfile` must **always** be `@__FILE__`

# Examples
```julia
# Minimal: uses current project, 1 thread per task, queue mode (default)
scan = Scan("my_scan", SlurmExec(@__FILE__, 8); energy=energies)

# Batch mode: one array task per scan point, complete memory isolation
scan = Scan("my_scan", SlurmExec(@__FILE__, length(energies); arraymode=:batch,
                                  memory="24G"); energy=energies)

# Many sims per task: 10 jobs × 12 workers = 120 concurrent single-threaded sims,
# with a 2-hour wall-clock limit per task
scan = Scan("my_scan", SlurmExec(@__FILE__, 10; procs=12, memory="48G", time="2:00:00");
            energy=energies)

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
    procs::Int
    workdir::String
    arraymode::Symbol
    time::String
    partition::String
end

function SlurmExec(scriptfile, ncores;
                   memory="",
                   project=let ap = Base.active_project()
                       isnothing(ap) ? "" : dirname(ap)
                   end,
                   nthreads=1,
                   procs=0,
                   workdir="",
                   arraymode=:queue,
                   time="",
                   partition="")
    ncores >= 1 || throw(ArgumentError("`ncores` must be ≥ 1, got $ncores"))
    nthreads >= 1 || throw(ArgumentError("`nthreads` must be ≥ 1, got $nthreads"))
    # `procs = -1` (all logical cores) is unsupported here because --cpus-per-task must be
    # fixed when the job script is generated; require an explicit worker count.
    procs >= 0 || throw(ArgumentError(
        "`procs` must be ≥ 0, got $procs (`-1` is not supported for SlurmExec; pass an " *
        "explicit number of workers per task)"))
    # Normalize and validate memory: strip whitespace, enforce strict format
    memory = strip(memory)
    if !isempty(memory)
        m = match(r"^(\d+)([KMGT]?)$"i, memory)
        isnothing(m) && throw(ArgumentError(
            "`memory` must match format \"<number>[K|M|G|T]\", got \"$memory\""))
        memory = m.captures[1] * uppercase(m.captures[2])
    end
    # Normalize and validate the wall-clock time limit. Slurm accepts "minutes",
    # "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and
    # "days-hours:minutes:seconds"; many clusters reject a job with no limit at all.
    time = strip(time)
    if !isempty(time)
        occursin(r"^(\d+-\d+(:\d+){0,2}|\d+(:\d+){0,2})$", time) || throw(ArgumentError(
            "`time` must be a Slurm time limit such as \"30\", \"HH:MM:SS\" or \"D-HH:MM:SS\", " *
            "got \"$time\""))
    end
    partition = strip(partition)
    # Validate project, workdir, partition: no quotes or newlines (shell injection prevention)
    for (name, val) in [("project", project), ("workdir", workdir), ("partition", partition)]
        if occursin(r"[\"\n\r]", val)
            throw(ArgumentError("`$name` must not contain quotes or newlines, got \"$val\""))
        end
    end
    arraymode in (:queue, :batch) || throw(ArgumentError("`arraymode` must be :queue or :batch, got :$arraymode"))
    # `procs` drives `--queue -p <procs>`, which only the queue array mode supports.
    procs > 0 && arraymode == :batch && throw(ArgumentError(
        "`procs > 0` requires `arraymode=:queue`; the :batch mode has no worker pool"))
    SlurmExec(scriptfile, ncores, memory, project, nthreads, procs, workdir, arraymode,
              time, partition)
end

"""
    SSHExec(localexec, hostname, subdir; remotehostname=hostname)

Execution mode which transfers the script file to the host given by `hostname` via SSH and
executes the scan on that host with a mode defined by `localexec`. `subdir` gives the
subdirectory (relative to the home directory) where scans are stored on the remote host. A
subfolder with automatically chosen name will be created in `subdir` to store this scan.

`hostname` is the address used for `ssh`/`scp` (so it must be reachable from the machine you
submit from). `remotehostname` is the value `Base.gethostname()` returns *on the remote host*;
it is how Luna recognises that it is already running there and should submit the job rather than
SSH onwards. By default these are assumed identical. On clusters where the login node reports a
different name than the address you connect to (e.g. you `ssh dmog.hw.ac.uk` but the node calls
itself `login1.pri.dmog.alces.network`), pass `remotehostname` explicitly — otherwise the
remote will not recognise itself and will SSH into itself in an endless loop.

!!! note
    The `localexec`'s script file must **always** be set with `@__FILE__`.
"""
struct SSHExec{eT} <: AbstractExec
    localexec::eT
    scriptfile::String
    hostname::String
    subdir::String
    remotehostname::String
end

function SSHExec(le::CondorExec, hostname, subdir; remotehostname=hostname)
    SSHExec(le, le.scriptfile, hostname, subdir, remotehostname)
end

function SSHExec(le::SlurmExec, hostname, subdir; remotehostname=hostname)
    SSHExec(le, le.scriptfile, hostname, subdir, remotehostname)
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

"""
    _worker_heap_flag(parent_hint, nproc) -> Vector{String}

Build the `--heap-size-hint` flag for `addprocs` workers by dividing the parent process's
heap-size hint (`parent_hint`, in bytes, e.g. from `Base.JLOptions().heap_size_hint`) evenly
among `nproc` workers. This keeps all workers in one task collectively within the Slurm
`--mem` limit. Returns an empty vector when no hint is set (`parent_hint == 0`) so worker
defaults are left untouched.
"""
function _worker_heap_flag(parent_hint::Integer, nproc::Integer)
    (parent_hint > 0 && nproc > 0) || return String[]
    per = div(parent_hint, nproc)
    per > 0 ? ["--heap-size-hint=$per"] : String[]
end

function runscan(f, scan::Scan{QueueExec})
    if scan.exec.nproc == 0
        _runscan(f, scan)
    else
        nproc = (scan.exec.nproc == -1) ? Base.Sys.CPU_THREADS : scan.exec.nproc
        # Propagate the parent's project (so workers find Luna in a project environment) and
        # a per-worker share of the heap-size hint (so N workers stay within one --mem limit).
        exeflags = String[]
        ap = Base.active_project()
        isnothing(ap) || push!(exeflags, "--project=$(dirname(ap))")
        append!(exeflags, _worker_heap_flag(Base.JLOptions().heap_size_hint, nproc))
        procs = addprocs(nproc; exeflags)
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

"""
    _default_queuefile(name) -> String

Default file name for the queue file of a scan named `name`, used when `QueueExec` is given
no explicit `queuefile`. Kept as a helper so the executor and the Slurm submit path agree on
the name (and hence on the [`_donefile`](@ref) marker name).
"""
_default_queuefile(name::AbstractString) = "qfile_$(string(hash(name); base=16)).h5"

"""
    _donefile(qfile) -> String

Path of the completion marker that sits next to the queue file `qfile` (the queue file path
with its extension replaced by `.done`; the marker is an empty flag file, not an HDF5 file).
The marker is created when a scan finishes (see `_runscan`) and signals to any later process
that the scan is already done, so it must not re-create the queue file and re-run the whole
scan.
"""
_donefile(qfile::AbstractString) = first(splitext(qfile)) * ".done"

function _runscan(f, scan::Scan{QueueExec})
    if isempty(scan.exec.queuefile)
        qfile = _default_queuefile(scan.name)
    else
        qfile = scan.exec.queuefile
    end
    lockpath = qfile*"_lock"
    donefile = _donefile(qfile)

    combos = vec(collect(Iterators.product(scan.arrays...)))
    qfile_created = false
    while true
        mkpidlock(lockpath; stale_age=120) do
            # first process to catch the pidlock creates the queue file
            if ~isfile(qfile)
                if qfile_created || isfile(donefile)
                    # Either we already created the queue file (so another process must have
                    # completed the scan and removed it), or a completion marker is present (a
                    # process that finished the scan left it, even if that was a previous run).
                    # Either way there is nothing to do: signal stop by setting scanidx to
                    # nothing rather than re-creating the queue file and re-running everything.
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
                # Create the completion marker *before* removing the queue file so there is
                # never a moment where neither exists (which would let a late-starting process
                # re-create the queue and re-run the whole scan).
                touch(donefile)
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
    # Slurm treats bare numbers as megabytes; mirror that for the heap hint
    isempty(unit) && (unit = "M")
    units = ["K", "M", "G", "T"]
    uidx = findfirst(==(unit), units)
    isnothing(uidx) && return ""
    # Compute 80% hint; if < 1 in current unit, downscale (e.g. 1T -> 800G)
    while true
        hint = floor(Int, value * 0.8)
        hint >= 1 && return "$(hint)$(units[uidx])"
        uidx == 1 && return ""
        value *= 1024
        uidx -= 1
    end
end

"""
    _slurm_script_lines(exec::SlurmExec, workdir::String) -> Vector{String}

Build the lines of the SBATCH job script for a `SlurmExec` with the given resolved
`workdir`. This is separated from `runscan` to allow testing the generated script content
without a Slurm installation.
"""
function _slurm_script_lines(exec::SlurmExec, workdir::String)
    julia = first(Base.julia_cmd().exec)
    script = exec.scriptfile
    cores = exec.ncores
    nthreads = exec.nthreads
    # One core per concurrent worker (`procs`), times the threads each worker uses. The
    # coordinating parent process is effectively idle while the workers run, so it needs no
    # extra core. `procs == 0` (single-process queue) keeps the previous `cpus = nthreads`.
    cpus_per_task = max(exec.procs, 1) * nthreads
    lines = [
        "#!/bin/bash",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$cpus_per_task",
        "#SBATCH -o %x_%a.stdout",
        "#SBATCH -e %x_%a.stderr",
        "#SBATCH --array=1-$cores",
        "#SBATCH --chdir \"$workdir\"",
    ]
    if !isempty(exec.memory)
        push!(lines, "#SBATCH --mem=$(exec.memory)")
    end
    if !isempty(exec.time)
        push!(lines, "#SBATCH --time=$(exec.time)")
    end
    if !isempty(exec.partition)
        push!(lines, "#SBATCH --partition=$(exec.partition)")
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
        # Resolve relative project paths against the script directory, not the workdir
        project = isabspath(exec.project) ? exec.project : abspath(joinpath(dirname(script), exec.project))
        juliacmd *= " --project=\"$project\""
    end
    juliacmd *= " \"$(abspath(script))\""
    if exec.arraymode == :batch
        juliacmd *= " --batch $cores,\$SLURM_ARRAY_TASK_ID"
    else
        juliacmd *= " --queue"
        # Run `procs` concurrent workers per task that share the queue file (constructor
        # guarantees procs > 0 implies arraymode == :queue).
        exec.procs > 0 && (juliacmd *= " -p $(exec.procs)")
    end
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
    # Clear any completion marker left by a previous run of this scan so that this fresh
    # submission starts cleanly; otherwise every array task would see the marker and exit
    # immediately, doing nothing. A partial queue file (from a crashed run) is left in place
    # so the array tasks resume rather than restart. The array tasks run with --chdir workdir
    # and the default queue-file name, so the marker lives next to it in workdir.
    rm(joinpath(workdir, _donefile(_default_queuefile(name))); force=true)
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
    if gethostname() == scan.exec.remotehostname
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
        @info "Running Luna script on remote host $host"
        read(`ssh $host julia \$HOME/$subdir/$folder/$scriptfile`, String)
    end
end

end