module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Logging: @info, @warn
import Printf: @sprintf
import Base: length, size
import Luna: Utils
import FileWatching.Pidfile
import FileWatching.Pidfile: mkpidlock, trymkpidlock
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


# Default time in seconds after which the lock file of a dead (but not provably dead)
# worker goes stale; see the QueueExec docstring.
const DEFAULT_STALE_AGE = 600.0

"""
    QueueExec(nproc=0, queuefile=""; stale_age=600)

Execution mode to run a scan using a file-based queueing system. Can be run in multiple separate
Julia sessions, or can spawn `nproc` subprocesses which then take items from the queue to run.

Possible values for `nproc` are:
- `0`: run only in the current Julia process
- `n > 0`: spawn `n` subprocesses and run on these
- `-1`: spawn as many subprocesses as the number of logical cores on the CPU
    (`Base.Sys.CPU_THREADS`)

If `queuefile` is given, the queuefile is stored at that path. If omitted, the queuefile is
stored in the current working directory with a name unique to the scan name. Note that the
queuefile is deleted at the end of the scan.

While a worker runs a scan point, it holds a lock file next to the queuefile
(`[queuefile]_item[i]_lock`) which it keeps fresh for as long as it is alive. If a worker
dies without marking its point as done or failed (for example because it was killed by a
cluster scheduler, the OOM killer, or Ctrl-C), other workers re-run that point once its
lock file has gone stale, i.e. has not been refreshed for longer than `stale_age` seconds.
A worker which provably died on the current machine is detected immediately; workers which
cannot be checked (e.g. running on other machines of a cluster) are given between
`stale_age` and `5*stale_age` seconds, in line with the behaviour of
`FileWatching.Pidfile`. Live workers refresh their lock file every `stale_age/2` seconds,
so points are never re-run while the worker running them is alive, no matter how long they
take. Setting `stale_age=0` disables the takeover of existing lock files; points marked as
in progress which have no lock file at all (e.g. left by a worker which died before
creating one) are always re-run.
"""
struct QueueExec <: AbstractExec
    nproc::Int
    queuefile::String
    stale_age::Float64
end

QueueExec(nproc=0, queuefile=""; stale_age=DEFAULT_STALE_AGE) = QueueExec(nproc, queuefile, stale_age)

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
    SlurmExec(scriptfile, ncores)

Execution mode which submits a scan to an slurm queue system claiming `ncores` cores.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct SlurmExec <: AbstractExec
    scriptfile::String
    ncores::Int
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
        "--stale-age"
            help = """Time in seconds after which a queued scan point marked as in progress
                      is re-run if the worker running it cannot be verified to be alive.
                      Only used together with --queue."""
            arg_type = Float64
            default = DEFAULT_STALE_AGE
            dest_name = "stale_age"
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
    args["queue"] && return QueueExec(args["procs"]; stale_age=args["stale_age"])
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

# The lock on the queue file itself is only ever held briefly (reading and writing a small
# HDF5 dataset), so this only bounds how long a worker which died while holding it can
# block the others.
const QFILE_LOCK_STALE_AGE = 120

# Path of the lock file held by a worker while it runs scan point i; see QueueExec.
_itemlockpath(qfile, i) = qfile * "_item$(i)_lock"

"""
    _tryitemlock(path, stale_age)

Attempt to acquire the per-point lock file at `path` without waiting. Return a
`Pidfile.LockMonitor` on success and `false` on failure.

A lock file left behind by a dead worker is taken over in one of two ways:
1. If the holder ran on this machine and its pid provably no longer exists, the lock file
   is removed immediately. This uses internal `Pidfile` API, so it is wrapped
   defensively--if anything goes wrong we simply fall back to 2.
2. Otherwise `trymkpidlock` takes the lock over once its mtime has not been refreshed for
   longer than `stale_age` seconds (`5*stale_age` if the holder cannot be proven dead,
   e.g. because it ran on a different machine).
"""
function _tryitemlock(path, stale_age)
    try
        if isfile(path)
            pid, hostname, age = Pidfile.parse_pidfile(path)
            if hostname == gethostname() && !Pidfile.isvalidpid(hostname, pid)
                rm(path; force=true)
            end
        end
    catch
    end
    trymkpidlock(path; stale_age)
end

function _log_unfinished(qfile, qdata, stale_age, nrun)
    ninprog = count(==(1), qdata)
    if nrun > 0
        # normal end of scan for this worker: others are still running the last points
        @info "Exiting scan: $ninprog scan point(s) still in progress in other workers."
        return
    end
    ndone = count(==(2), qdata)
    nfailed = count(==(3), qdata)
    @warn """Exiting scan without having run any scan points, but the scan is not finished:
             $ndone done, $nfailed failed, $ninprog in progress
             (queue file: $(abspath(qfile))).
          Scan points marked as in progress are either being run by other workers right
          now, or were abandoned by workers which died. Abandoned points are re-run
          automatically once their lock file goes stale, which takes between stale_age
          and 5*stale_age seconds (stale_age = $stale_age)--simply re-run this scan then.
          Only if you are certain that no workers are running this scan, you can instead
          reset it by deleting the queue file and its lock files:
          $(abspath(qfile)) and $(abspath(_itemlockpath(qfile, "*")))"""
end

function _runscan(f, scan::Scan{QueueExec})
    if isempty(scan.exec.queuefile)
        h = string(hash(scan.name); base=16)
        qfile = "qfile_$h.h5"
    else
        qfile = scan.exec.queuefile
    end
    lockpath = qfile*"_lock"
    stale_age = scan.exec.stale_age

    combos = vec(collect(Iterators.product(scan.arrays...)))
    qfile_created = false
    scanidx = nothing  # scan point claimed in the current loop iteration
    itemlock = nothing # Pidfile.LockMonitor held while running scanidx
    reclaimed = false  # whether scanidx was abandoned by a dead worker
    finished = false   # whether the whole scan is done
    nrun = 0           # number of scan points run by this worker
    qdata = Int[]
    while true
        mkpidlock(lockpath; stale_age=QFILE_LOCK_STALE_AGE) do
            scanidx = nothing
            itemlock = nothing
            reclaimed = false
            # first process to catch the pidlock creates the queue file
            if ~isfile(qfile)
                if qfile_created
                    # The queue file was already created, so another process must have
                    # completed the scan and removed it (along with all point lock
                    # files). Signal that we should stop.
                    finished = true
                    return
                end
                HDF5.h5open(qfile, "cw") do file
                    file["qdata"] = zeros(Int, length(scan))
                end
            end
            qfile_created = true
            # read the queue data
            qdata = HDF5.h5open(qfile) do file
                read(file["qdata"])
            end
            # Try to claim a scan point: first those which have not been started, then
            # those marked as in progress. The latter can only be claimed if the worker
            # which marked them has died and its lock file is missing or has gone stale
            # (see _tryitemlock).
            for i in vcat(findall(==(0), qdata), findall(==(1), qdata))
                lk = _tryitemlock(_itemlockpath(qfile, i), stale_age)
                lk === false && continue
                itemlock = lk
                scanidx = i
                reclaimed = qdata[i] == 1
                # mark the point as in progress
                HDF5.h5open(qfile, "r+") do file
                    file["qdata"][i] = 1
                end
                return
            end
            # Nothing left to claim. If everything is done or failed, clean up. Holding
            # the queue-file lock here means we cannot race against a worker which is
            # only just starting a fresh scan re-using the same queue file path.
            if all(qdata .> 1)
                rm(qfile; force=true) # remove the queue file
                for i in eachindex(qdata)
                    rm(_itemlockpath(qfile, i); force=true)
                end
                finished = true
            end
        end # release pidlock
        if isnothing(scanidx) # no scan point could be claimed
            finished || _log_unfinished(qfile, qdata, stale_age, nrun)
            break # break out of the loop
        end
        try
            if reclaimed
                @info ("Scan point $scanidx was marked as in progress but the worker"
                       * " running it appears to have died. Running it again.")
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
            mkpidlock(lockpath; stale_age=QFILE_LOCK_STALE_AGE) do
                if isfile(qfile)
                    HDF5.h5open(qfile, "r+") do file
                        file["qdata"][scanidx] = code # mark as done/failed
                    end
                end
                # Release (and thereby delete) the point lock while still holding the
                # queue-file lock: qdata is already written, so the point can no longer
                # be mistaken for abandoned, and the all-done cleanup above cannot run
                # concurrently and delete the lock file while we still hold it.
                close(itemlock)
            end
            nrun += 1
        finally
            # if anything above threw, release the point lock anyway so that other
            # workers can re-run the point; closing twice is harmless
            close(itemlock)
            itemlock = nothing
        end
        Base.GC.gc()
    end
end

function runscan(f, scan::Scan{SlurmExec})
    # make submission file for slurm
    script = scan.exec.scriptfile
    dir = dirname(script)
    cores = scan.exec.ncores
    name = scan.name
    @info "Submitting slurm job for $script running on $cores cores."
    # Adding the --queue command-line argument below means that when running the Condor job,
    # the SlurmExec is ignored even if explicitly defined inside the script.
    lines = [
        "#!/bin/bash",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH -o %x_%a.stdout",
        "#SBATCH -e %x_%a.stderr",
        "#SBATCH --array=1-$cores",
        "#SBATCH --chdir $dir",
        "julia $(basename(script)) --queue"
    ]
    subfile = joinpath(dir, "$name.sh")
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
