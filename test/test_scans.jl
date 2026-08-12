import Test: @test, @testset, @test_throws
using Luna
import HDF5
using Distributed
import Logging: with_logger, NullLogger

@testset "Chunking" begin
    function contains_all_unique(chunks, x)
        contains_unique = []
        for xi in x
            c = count([xi in chi for chi in chunks])
            push!(contains_unique, (c==1))
        end
        return all(contains_unique)
    end
    pass = []
    for L = 5:11:896
        for n = 1:24
            x = collect(1:L)
            ch = Scans.chunks(x, n)
            push!(pass, contains_all_unique(ch, x))
        end
    end
    @test all(pass)
end

##
# Queue-mode scans use the default (CWD-relative) queue file, so a completion
# marker left behind by a previous run of the same-named scan would turn them
# into no-ops — that is precisely the marker's purpose for late-starting
# cluster tasks. Clear any leftover queue state before and after every testset
# that runs such a scan, so the tests are self-isolating and leave no litter.
function clear_queue_state(scanname)
    qf = Scans._default_queuefile(scanname)
    for f in (qf, qf*"_lock", Scans._donefile(qf))
        rm(f; force=true)
    end
end

args_execs = Dict(["-l"] => Scans.LocalExec,
                  ["-r", "1:6"] => Scans.RangeExec,
                  ["-b", "2,1"] => Scans.BatchExec,
                  ["-q"] =>  Scans.QueueExec)
                #   ["-q", "-p", "4"] =>  Scans.QueueExec)
@testset "Scanning $arg" for (arg, exec) in pairs(args_execs)
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, arg...)

# A stale marker would make the -q case a silent no-op (skipping the per-point
# assertions); clearing is harmless for the non-queue execs.
clear_queue_state("scantest_cmdlineargs")
v = collect(1:10)
scan = Scan("scantest_cmdlineargs"; var=v)
@test scan.exec isa exec
ran = Int[]
runscan(scan) do scanidx, vi
    @test vi == v[scanidx]
    push!(ran, scanidx)
    sleep(rand()) # avoid two processes finishing at precisely the same time
end
# Full coverage asserted where the semantics are unambiguous — in particular
# the queue case, which would run NOTHING if a stale marker were present.
if scan.exec isa Scans.LocalExec || scan.exec isa Scans.QueueExec
    @test sort(ran) == collect(1:10)
end
clear_queue_state("scantest_cmdlineargs")
end

##
@testset "command-line args overwrite" begin
    for _ in eachindex(ARGS)
        pop!(ARGS)
    end
    push!(ARGS, "--range", "1:4")
    v = collect(1:10)
    scan = Scan("scantest_cmdlineargs_overwrite", Scans.LocalExec(); var=v)
    runscan(scan) do scanidx, vi
        # command line ARGS above should overwrite Scans.LocalExec passed in above
        @test scan.exec isa Scans.RangeExec
    end
    for _ in eachindex(ARGS)
        pop!(ARGS)
    end
end

##
try
@testset "scansave" begin
scan = Scan("scantest_scansave", Scans.LocalExec())
x = collect(1:8)
y = collect(1:6)
addvariable!(scan, :x, x)
addvariable!(scan, :y, y)

runscan(scan) do scanidx, xi, yi
    out = [xi * yi, (xi)^2, (yi)^2]
    slength = xi * yi
    e = fill(1.0*slength, slength)
    em = fill(1.0*slength, (2, slength))
    stats = Dict("energy" => e, "energym" => em)
    xx, yy = xi, yi
    Output.@scansave(scan, scanidx, Eω=out, stats=stats, keyword=[xx, yy])
end
HDF5.h5open("scantest_scansave_collected.h5", "r") do file
    @test read(file["scanvariables"]["x"]) == x
    @test read(file["scanvariables"]["y"]) == y
    @test read(file["scanorder"]) == ["x", "y"]
    out = read(file["Eω"])
    stats = read(file["stats"])
    pass = true
    for (ix, xi) in enumerate(x)
        for (iy, yi) in enumerate(y)
            pass = pass && (out[:, ix, iy] == [xi * yi, (xi)^2, (yi)^2])
            slength = xi * yi
            pass = pass && (stats["valid_length"][ix, iy] == slength)
            e = fill(1.0*slength, slength)
            em = fill(1.0*slength, (2, slength))
            pass = pass && (stats["energy"][1:slength, ix, iy] == e)
            pass = pass && all(isnan.(stats["energy"][slength+1:end, ix, iy]))
            pass = pass && (stats["energym"][:, 1:slength, ix, iy] == em)
            pass = pass && all(isnan.(stats["energym"][:, slength+1:end, ix, iy]))
            pass = pass && (file["keyword"][:, ix, iy] == [xi, yi])
        end
    end
    @test pass
    this = @__FILE__
    code = open(this, "r") do file
                read(file, String)
            end
    @test read(file["script"]) == this * "\n" * code
end
rm("scantest_scansave_collected.h5")
end
catch
rm("scantest_scansave_collected.h5")
end

##
@testset "ScanHDF5Output" begin
var1 = collect(range(1, length=5))
var2 = collect(1:3)
scan = Scan("scantest_hdf5output", Scans.LocalExec(); var1=var1, var2=var2)
files = String[]
runscan(scan) do scanidx, vi1, vi2
    out = Output.@ScanHDF5Output(scan, scanidx, 0, 1, 10)
    @test out["meta"]["scanarrays"]["var1"] == var1
    @test out["meta"]["scanarrays"]["var2"] == var2
    @test out["meta"]["scanvars"]["var1"] == vi1
    @test out["meta"]["scanvars"]["var2"] == vi2
    @test out["meta"]["scanshape"] == [length(var1), length(var2)]
    @test out["meta"]["scanorder"] == ["var1", "var2"]
    for z = range(0, 1; length=10)
        out([1.0, 1.0], z, 0.1, t -> [0.0, 0.0])
    end
    push!(files, out.fpath)
end
z, vvar1, vvar2 = Processing.scanproc() do output
    output["z"], output["meta"]["scanarrays"]["var1"], output["meta"]["scanarrays"]["var2"]
end
@test all(vvar1 .== var1)
@test all(vvar2 .== var2)
@test all(z .== collect(range(0, 1; length=10)))
rm.(files)
end

##
if ~("GITHUB_ACTIONS" in keys(ENV))
@testset "multi-process queue scan" begin
    clear_queue_state("scantest_queue_multiproc")
    ps = addprocs(2)
    @everywhere using Luna
    function worker()
        energies = collect(range(5e-6, 20e-6; length=16))
        scan = Scan("scantest_queue_multiproc", Scans.QueueExec(); energy=energies)
        idcs_run = Int[]
        runscan(scan) do scanidx, energy
            prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                           trange=400e-15, λlims=(200e-9, 4e-6))
            push!(idcs_run, scanidx)
        end
        idcs_run
    end
    r2 = @spawnat ps[1] worker()
    r3 = @spawnat ps[2] worker()
    i2 = fetch(r2)
    i3 = fetch(r3)
    # check that both processes ran something
    @test (length(i2) > 0)
    @test (length(i3) > 0)
    push!(i2, i3...)
    for scanidx in 1:16
        @test count(i2 .== scanidx) == 1 # check that all indices have been run exactly once
    end
    rmprocs(ps)
    clear_queue_state("scantest_queue_multiproc")
end
##
@testset "multi-process queue scan with error" begin
    clear_queue_state("scantest_queue_multiproc_err")
    ps = addprocs(2)
    @everywhere using Luna
    @everywhere import Logging: with_logger, NullLogger
    # do it again but with one process giving an error
    scanname = "scantest_queue_multiproc_err"
    function worker_err()
        energies = collect(range(5e-6, 20e-6; length=16))
        scan = Scan(scanname, Scans.QueueExec(); energy=energies)
        idcs_run = Int[]
        runscan(scan) do scanidx, energy
            with_logger(NullLogger()) do
                prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                            trange=400e-15, λlims=(200e-9, 4e-6))
                if scanidx == 16
                    error("This exception is expected as part of the test suite")
                end
                push!(idcs_run, scanidx)
            end
        end
        idcs_run
    end
    r2 = @spawnat ps[1] worker_err()
    r3 = @spawnat ps[2] worker_err()
    i2 = fetch(r2)
    i3 = fetch(r3)
    # check that both processes ran something
    @test (length(i2) > 0)
    @test (length(i3) > 0)
    push!(i2, i3...)
    for scanidx in 1:15
        # check that all indices have been run once, except for the one with an error
        @test count(i2 .== scanidx) == 1
    end
    # The default queue file is CWD-relative; completion must have removed it.
    @test !isfile(Scans._default_queuefile(scanname))
    rmprocs(ps)
    clear_queue_state(scanname)
end

##
@testset "multi-process queue scan via exec" begin
    clear_queue_state("scantest_queue_multiproc_exec")
    energies = collect(range(5e-6, 20e-6; length=16))
    scan = Scan("scantest_queue_multiproc_exec", Scans.QueueExec(4); energy=energies)
    td = joinpath(tempdir(), tempname())
    runscan(scan) do scanidx, energy
        println("running on $(myid())")
        prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                       trange=400e-15, λlims=(200e-9, 4e-6),
                       filepath=joinpath(td, makefilename(scan, scanidx)))
        open(joinpath(td, "$(scanidx)_on_$(myid())"), "w") do io
            write(io, "$scanidx ran on $(myid())")
        end
    end
    # should be exactly 2 files per scanidx: output .h5 and "scanidx_on_procid"
    @test length(readdir(td)) == 2length(energies)
    rm(td; recursive=true)
    clear_queue_state("scantest_queue_multiproc_exec")
end

# do it again to make sure we can run multiple multi-process scans in one session
@testset "multi-process queue scan via exec--again" begin
    clear_queue_state("scantest_queue_multiproc_exec_again")
    energies = collect(range(5e-6, 20e-6; length=16))
    scan = Scan("scantest_queue_multiproc_exec_again", Scans.QueueExec(4); energy=energies)
    td = joinpath(tempdir(), tempname())
    runscan(scan) do scanidx, energy
        println("running on $(myid())")
        prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                       trange=400e-15, λlims=(200e-9, 4e-6),
                       filepath=joinpath(td, makefilename(scan, scanidx)))
        open(joinpath(td, "$(scanidx)_on_$(myid())"), "w") do io
            write(io, "$scanidx ran on $(myid())")
        end
    end
    # should be exactly 2 files per scanidx: output .h5 and "scanidx_on_procid"
    @test length(readdir(td)) == 2length(energies)
    rm(td; recursive=true)
    clear_queue_state("scantest_queue_multiproc_exec_again")
end
end # if ~("GITHUB_ACTIONS" in keys(ENV))

##
@testset "queue marker prevents re-running a completed scan" begin
    @test Scans._donefile("a.h5") == "a.done"
    mktempdir() do td
        qf = joinpath(td, "qfile.h5")
        energies = collect(range(5e-6, 20e-6; length=4))
        # First run completes the whole scan (trivial work, no propagation needed).
        scan = Scan("scantest_marker", Scans.QueueExec(0, qf); energy=energies)
        run1 = Int[]
        runscan((scanidx, energy) -> push!(run1, scanidx), scan)
        @test sort(run1) == collect(1:4)         # whole scan ran exactly once
        @test !isfile(qf)                         # queue file removed on completion
        @test isfile(Scans._donefile(qf))         # completion marker created

        # A fresh runscan over the same queue file simulates a late-starting task: it must
        # see the marker and do nothing rather than re-create the queue and re-run everything.
        scan2 = Scan("scantest_marker", Scans.QueueExec(0, qf); energy=energies)
        run2 = Int[]
        runscan((scanidx, energy) -> push!(run2, scanidx), scan2)
        @test isempty(run2)                       # marker stops the re-run
        @test !isfile(qf)                         # no stray queue file left behind
    end
end

##
@testset "queue maxpoints limits points per process" begin
    # Validation and defaults
    @test Scans.QueueExec().maxpoints == 0
    @test Scans.QueueExec(0, ""; maxpoints=3).maxpoints == 3
    @test_throws ArgumentError Scans.QueueExec(0, ""; maxpoints=-1)
    # Command-line parsing: --queue --maxpoints N builds the matching QueueExec
    exq = Scans.makeexec(["--queue", "--maxpoints", "1"])
    @test exq isa Scans.QueueExec
    @test exq.maxpoints == 1
    @test exq.nproc == 0
    @test Scans.makeexec(["--queue"]).maxpoints == 0
    exqm = Scans.makeexec(["--queue", "-m", "3", "-p", "2"])
    @test exqm.maxpoints == 3
    @test exqm.nproc == 2

    mktempdir() do td
        # maxpoints=1, relaunched until done: simulates SlurmExec `instances` mode, where
        # each scan point runs in a fresh OS process and a shell loop respawns processes
        # until the completion marker appears.
        qf = joinpath(td, "qfile_max1.h5")
        energies = collect(range(5e-6, 20e-6; length=5))
        ran = Int[]
        nlaunches = 0
        while !isfile(Scans._donefile(qf)) && nlaunches < 20 # bound in case of a bug
            scan = Scan("scantest_maxpoints", Scans.QueueExec(0, qf; maxpoints=1);
                        energy=energies)
            runscan((scanidx, energy) -> push!(ran, scanidx), scan)
            nlaunches += 1
        end
        @test sort(ran) == collect(1:5)   # every point ran exactly once
        # 5 single-point runs + 1 final run that found the queue drained, wrote the
        # completion marker and removed the queue file
        @test nlaunches == 6
        @test isfile(Scans._donefile(qf))
        @test !isfile(qf)

        # maxpoints=2: one process takes exactly two points then stops, queue survives
        qf2 = joinpath(td, "qfile_max2.h5")
        ran2 = Int[]
        scan2 = Scan("scantest_maxpoints2", Scans.QueueExec(0, qf2; maxpoints=2);
                     energy=energies)
        runscan((scanidx, energy) -> push!(ran2, scanidx), scan2)
        @test length(ran2) == 2
        @test isfile(qf2)                 # pending points remain queued
        @test !isfile(Scans._donefile(qf2))
    end
end

##
@testset "automatic ScanHDF5Output in prop_capillary scan" begin
    energies = collect(range(5e-6, 10e-6; length=4))
    scan = Scan("scantest_autofilename", Scans.LocalExec(); energy=energies)
    mktempdir() do td
        runscan(scan) do scanidx, energy
            prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                        trange=400e-15, λlims=(200e-9, 4e-6), filepath=td, scan, scanidx)
        end
        @test length(readdir(td)) == length(energies)
    end
end

##
@testset "manual filename in ScanHDF5Output" begin
    energies = collect(range(5e-6, 10e-6; length=4))
    scan = Scan("scantest_manualfilename", Scans.LocalExec(); energy=energies)
    mktempdir() do td
        runscan(scan) do scanidx, energy
            prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                        trange=400e-15, λlims=(200e-9, 4e-6), filepath=td, scan, scanidx,
                        filename="newname")
        end
        @test length(readdir(td)) == length(energies)
    @test all(startswith.(readdir(td), "newname"))
    end
end

# Slurm tests only run on Unix (Slurm doesn't exist on Windows and paths differ)
if !Sys.iswindows()

##
@testset "SlurmExec construction" begin
    # Backward compatibility: positional args only
    ex = Scans.SlurmExec("/tmp/test.jl", 4)
    @test ex.scriptfile == "/tmp/test.jl"
    @test ex.ncores == 4
    @test ex.memory == ""
    @test ex.nthreads == 1
    @test ex.procs == 0
    @test ex.workdir == ""
    # project defaults to current active project (or "" if nothing)
    ap = Base.active_project()
    expected_project = isnothing(ap) ? "" : dirname(ap)
    @test ex.project == expected_project

    # Full keyword args
    ex2 = Scans.SlurmExec("/tmp/test.jl", 8; memory="24G", project="/custom/env",
                           nthreads=2, workdir="/tmp/mywork")
    @test ex2.memory == "24G"
    @test ex2.project == "/custom/env"
    @test ex2.nthreads == 2
    @test ex2.workdir == "/tmp/mywork"

    # Opt out of project
    ex3 = Scans.SlurmExec("/tmp/test.jl", 4; project="")
    @test ex3.project == ""

    # SSHExec wrapping
    ssh = Scans.SSHExec(ex2, "myhost", "scans")
    @test ssh.localexec === ex2
    @test ssh.hostname == "myhost"
    # remotehostname defaults to hostname...
    @test ssh.remotehostname == "myhost"
    # ...but can differ (SSH target vs the name the remote reports for itself)
    ssh2 = Scans.SSHExec(ex2, "dmog.hw.ac.uk", "scans";
                         remotehostname="login1.pri.dmog.alces.network")
    @test ssh2.hostname == "dmog.hw.ac.uk"
    @test ssh2.remotehostname == "login1.pri.dmog.alces.network"

    # Validation: ncores must be >= 1
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 0)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; cpus=-1)

    # cpus: default 0 -> automatic (procs x nthreads); explicit value overrides.
    exc0 = Scans.SlurmExec("/tmp/test.jl", 4; nthreads=2)
    lines0 = Scans._slurm_script_lines(exc0, "/tmp")
    @test any(l -> l == "#SBATCH --cpus-per-task=2", lines0)
    exc1 = Scans.SlurmExec("/tmp/test.jl", 4; nthreads=1, cpus=10)
    lines1 = Scans._slurm_script_lines(exc1, "/tmp")
    @test any(l -> l == "#SBATCH --cpus-per-task=10", lines1)
    @test any(l -> l == "export JULIA_NUM_THREADS=1", lines1)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", -1)

    # Validation: nthreads must be >= 1
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; nthreads=0)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; nthreads=-1)

    # Validation: memory format (no internal whitespace allowed)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; memory="bad")
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; memory="24 G")
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; memory="12.5G")
    # Valid memory formats should not throw
    Scans.SlurmExec("/tmp/test.jl", 4; memory="24G")
    Scans.SlurmExec("/tmp/test.jl", 4; memory="500")
    Scans.SlurmExec("/tmp/test.jl", 4; memory="100M")
    # Memory is normalized: lowercase -> uppercase, whitespace stripped
    @test Scans.SlurmExec("/tmp/test.jl", 4; memory="24g").memory == "24G"
    @test Scans.SlurmExec("/tmp/test.jl", 4; memory=" 24G ").memory == "24G"

    # time / partition default to empty (directive omitted)
    exdef = Scans.SlurmExec("/tmp/test.jl", 4)
    @test exdef.time == ""
    @test exdef.partition == ""
    # Valid Slurm time formats are accepted and stored (whitespace stripped)
    @test Scans.SlurmExec("/tmp/test.jl", 4; time="30").time == "30"
    @test Scans.SlurmExec("/tmp/test.jl", 4; time="00:30:00").time == "00:30:00"
    @test Scans.SlurmExec("/tmp/test.jl", 4; time="7-00:00:00").time == "7-00:00:00"
    @test Scans.SlurmExec("/tmp/test.jl", 4; time=" 1:00:00 ").time == "1:00:00"
    @test Scans.SlurmExec("/tmp/test.jl", 4; partition="gpu").partition == "gpu"
    # Validation: bad time format throws
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; time="soon")
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; time="1h")
    # Validation: partition must not contain quotes/newlines (shell injection guard)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; partition="a\nb")

    # Validation: project and workdir must not contain quotes or newlines
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; project="path\"bad")
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; workdir="path\nbad")

    # arraymode defaults to :queue
    @test ex.arraymode == :queue
    # Explicit batch mode
    ex_batch = Scans.SlurmExec("/tmp/test.jl", 64; arraymode=:batch)
    @test ex_batch.arraymode == :batch
    # Invalid arraymode
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; arraymode=:invalid)

    # procs: concurrent workers per array task
    @test Scans.SlurmExec("/tmp/test.jl", 4; procs=12).procs == 12
    # procs must be >= 0; -1 (all cores) is unsupported for SlurmExec
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; procs=-1)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; procs=-2)
    # procs > 0 is incompatible with batch mode (no worker pool)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; procs=12, arraymode=:batch)
    # procs == 0 with batch mode is fine (default)
    @test Scans.SlurmExec("/tmp/test.jl", 4; procs=0, arraymode=:batch).procs == 0

    # instances: one-shot simulation processes per array task
    @test Scans.SlurmExec("/tmp/test.jl", 4).instances == 0 # default off
    @test Scans.SlurmExec("/tmp/test.jl", 4; instances=2).instances == 2
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; instances=-1)
    # instances needs the shared queue (no queue to respawn against in batch mode)
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; instances=2,
                                               arraymode=:batch)
    # instances and procs are alternative concurrency mechanisms, not composable
    @test_throws ArgumentError Scans.SlurmExec("/tmp/test.jl", 4; instances=2, procs=2)
end

##
@testset "heap-size-hint parsing" begin
    @test Scans._parse_memory_for_heap_hint("24G") == "19G"
    @test Scans._parse_memory_for_heap_hint("10M") == "8M"
    @test Scans._parse_memory_for_heap_hint("100G") == "80G"
    @test Scans._parse_memory_for_heap_hint("2T") == "1T"
    @test Scans._parse_memory_for_heap_hint("500") == "400M"
    @test Scans._parse_memory_for_heap_hint("500K") == "400K"
    # Unit downscaling: 1T -> 800G, 1G -> 800M, 1M -> 800K
    @test Scans._parse_memory_for_heap_hint("1T") == "819G"
    @test Scans._parse_memory_for_heap_hint("1G") == "819M"
    @test Scans._parse_memory_for_heap_hint("1M") == "819K"
    # Edge cases
    @test Scans._parse_memory_for_heap_hint("") == ""
    @test Scans._parse_memory_for_heap_hint("abc") == ""
    @test Scans._parse_memory_for_heap_hint("12.5G") == ""  # no float support
    # Case insensitive
    @test Scans._parse_memory_for_heap_hint("24g") == "19G"
    @test Scans._parse_memory_for_heap_hint("10m") == "8M"
    # Shared among nshare concurrent processes (SlurmExec `instances`): hint = 80%/nshare
    @test Scans._parse_memory_for_heap_hint("100G", 2) == "40G"
    @test Scans._parse_memory_for_heap_hint("24G", 2) == "9G"
    @test Scans._parse_memory_for_heap_hint("24G", 1) == "19G" # nshare=1 == old behaviour
    # Sub-unit shares downscale to the next unit (1G across 2 -> 409M)
    @test Scans._parse_memory_for_heap_hint("1G", 2) == "409M"
end

##
@testset "worker heap flag division" begin
    # Parent hint (bytes) divided evenly among workers
    @test Scans._worker_heap_flag(12_000_000_000, 12) == ["--heap-size-hint=1000000000"]
    @test Scans._worker_heap_flag(1_000, 4) == ["--heap-size-hint=250"]
    # No hint set on parent -> no flag (workers keep their defaults)
    @test Scans._worker_heap_flag(0, 12) == String[]
    # Degenerate nproc -> no flag (avoid div-by-zero / nonsense)
    @test Scans._worker_heap_flag(12_000_000_000, 0) == String[]
    # Per-worker share rounds down to zero -> no flag
    @test Scans._worker_heap_flag(5, 12) == String[]
end

##
@testset "Slurm workdir resolution" begin
    # Default: auto-subdirectory based on scan name
    ex = Scans.SlurmExec("/home/user/scripts/run.jl", 4)
    @test Scans._resolve_slurm_workdir(ex, "my_scan") == "/home/user/scripts/my_scan_slurm"

    # Explicit workdir
    ex2 = Scans.SlurmExec("/home/user/scripts/run.jl", 4; workdir="/tmp/custom")
    @test Scans._resolve_slurm_workdir(ex2, "my_scan") == "/tmp/custom"

    # Spaces in script path
    ex3 = Scans.SlurmExec("/path/with spaces/run.jl", 4)
    @test Scans._resolve_slurm_workdir(ex3, "test") == "/path/with spaces/test_slurm"
end

##
@testset "Slurm script generation" begin
    # Full options: memory, project, nthreads
    ex = Scans.SlurmExec("/path/with spaces/test.jl", 16;
                         memory="24G", project="/my project/env", nthreads=2)
    workdir = "/path/with spaces/myscan_slurm"
    lines = Scans._slurm_script_lines(ex, workdir)
    script = join(lines, "\n")

    # Shebang
    @test lines[1] == "#!/bin/bash"
    # SBATCH directives
    @test any(l -> l == "#SBATCH --ntasks=1", lines)
    @test any(l -> l == "#SBATCH --cpus-per-task=2", lines)
    @test any(l -> l == "#SBATCH --array=1-16", lines)
    @test any(l -> l == "#SBATCH --mem=24G", lines)
    # Quoted chdir uses workdir, not script dir
    @test any(l -> l == "#SBATCH --chdir \"/path/with spaces/myscan_slurm\"", lines)
    # ulimit
    @test any(l -> l == "ulimit -v unlimited", lines)
    # Thread pinning exports
    @test any(l -> l == "export JULIA_NUM_THREADS=2", lines)
    @test any(l -> l == "export OMP_NUM_THREADS=2", lines)
    @test any(l -> l == "export OPENBLAS_NUM_THREADS=2", lines)
    @test any(l -> l == "export MKL_NUM_THREADS=2", lines)
    # Julia command (last line)
    juliacmd = lines[end]
    # Julia command uses actual julia executable path (no backticks or stray quotes)
    julia_path = first(Base.julia_cmd().exec)
    @test occursin(julia_path, juliacmd)
    @test occursin("--heap-size-hint=19G", juliacmd)
    # Absolute project path is passed through as-is
    @test occursin("--project=\"/my project/env\"", juliacmd)
    # Script path is absolute (so it can be found from the workdir)
    @test endswith(juliacmd, "\"/path/with spaces/test.jl\" --queue")

    # Relative project path is resolved against script directory
    ex_rel = Scans.SlurmExec("/home/user/scripts/test.jl", 4;
                              project=".", memory="", nthreads=1)
    lines_rel = Scans._slurm_script_lines(ex_rel, "/home/user/scripts/test_slurm")
    juliacmd_rel = lines_rel[end]
    resolved_project = abspath(joinpath("/home/user/scripts", "."))
    @test occursin("--project=\"$resolved_project\"", juliacmd_rel)

    # Minimal options: no memory, empty project
    ex_min = Scans.SlurmExec("/tmp/simple.jl", 4; memory="", project="")
    lines_min = Scans._slurm_script_lines(ex_min, "/tmp/workdir")
    script_min = join(lines_min, "\n")
    # No --mem line
    @test !any(l -> startswith(l, "#SBATCH --mem"), lines_min)
    # No --heap-size-hint
    @test !occursin("--heap-size-hint", lines_min[end])
    # No --project
    @test !occursin("--project", lines_min[end])
    # cpus-per-task defaults to 1
    @test any(l -> l == "#SBATCH --cpus-per-task=1", lines_min)
    # Thread exports default to 1
    @test any(l -> l == "export JULIA_NUM_THREADS=1", lines_min)
    # ulimit is always present
    @test any(l -> l == "ulimit -v unlimited", lines_min)
    # Julia binary is still quoted
    @test startswith(lines_min[end], "\"")
    @test endswith(lines_min[end], "\"/tmp/simple.jl\" --queue")
    # chdir points to workdir
    @test any(l -> l == "#SBATCH --chdir \"/tmp/workdir\"", lines_min)
    # No --time / --partition unless requested
    @test !any(l -> startswith(l, "#SBATCH --time"), lines_min)
    @test !any(l -> startswith(l, "#SBATCH --partition"), lines_min)

    # time and partition directives appear when set
    ex_tp = Scans.SlurmExec("/tmp/simple.jl", 4; time="00:30:00", partition="gpu")
    lines_tp = Scans._slurm_script_lines(ex_tp, "/tmp/workdir")
    @test any(l -> l == "#SBATCH --time=00:30:00", lines_tp)
    @test any(l -> l == "#SBATCH --partition=gpu", lines_tp)

    # ulimit comes before exports (correct ordering)
    idx_ulimit = findfirst(l -> l == "ulimit -v unlimited", lines)
    idx_export = findfirst(l -> startswith(l, "export JULIA_NUM_THREADS"), lines)
    @test idx_ulimit < idx_export

    # Batch mode: uses --batch instead of --queue
    ex_batch = Scans.SlurmExec("/tmp/run.jl", 64; arraymode=:batch, project="")
    lines_batch = Scans._slurm_script_lines(ex_batch, "/tmp/work")
    juliacmd_batch = lines_batch[end]
    @test occursin("--batch 64,\$SLURM_ARRAY_TASK_ID", juliacmd_batch)
    @test !occursin("--queue", juliacmd_batch)

    # Queue mode (default): uses --queue
    ex_queue = Scans.SlurmExec("/tmp/run.jl", 8; project="")
    lines_queue = Scans._slurm_script_lines(ex_queue, "/tmp/work")
    @test endswith(lines_queue[end], "--queue")
    @test !occursin("--batch", lines_queue[end])
    # procs == 0 (default): bare --queue, no -p, cpus-per-task == nthreads
    @test !occursin(" -p ", lines_queue[end])
    @test any(l -> l == "#SBATCH --cpus-per-task=1", lines_queue)

    # procs > 0: array task spawns workers via `--queue -p <procs>`
    ex_procs = Scans.SlurmExec("/tmp/run.jl", 10; procs=12, project="")
    lines_procs = Scans._slurm_script_lines(ex_procs, "/tmp/work")
    # 10 array tasks, 12 cores each (one per worker), single thread per worker
    @test any(l -> l == "#SBATCH --array=1-10", lines_procs)
    @test any(l -> l == "#SBATCH --cpus-per-task=12", lines_procs)
    @test endswith(lines_procs[end], "--queue -p 12")

    # procs combines with nthreads: cpus-per-task = procs * nthreads
    ex_pt = Scans.SlurmExec("/tmp/run.jl", 10; procs=12, nthreads=2, project="")
    lines_pt = Scans._slurm_script_lines(ex_pt, "/tmp/work")
    @test any(l -> l == "#SBATCH --cpus-per-task=24", lines_pt)
    @test any(l -> l == "export JULIA_NUM_THREADS=2", lines_pt)
    @test endswith(lines_pt[end], "--queue -p 12")
end

##
@testset "Slurm instances mode script generation" begin
    ex = Scans.SlurmExec("/tmp/run.jl", 10; instances=2, memory="100G", project="")
    lines = Scans._slurm_script_lines(ex, "/tmp/work", "my_scan")
    script = join(lines, "\n")
    # The respawn loop keys off the queue-completion marker, whose name must match what
    # `_runscan` derives from the scan name (tasks chdir into workdir, so a bare name works).
    donefile = Scans._donefile(Scans._default_queuefile("my_scan"))
    @test any(l -> l == "DONEFILE=\"$donefile\"", lines)
    # One background loop per instance, each running one-shot single-point processes
    @test any(l -> l == "for i in \$(seq 1 2); do", lines)
    @test occursin("--queue --maxpoints 1", script)
    # A crashing process must not hot-loop
    @test occursin("|| sleep 10", script)
    # The task waits for all instance loops before exiting
    @test lines[end] == "wait"
    # No plain (drain-the-queue) julia invocation and no worker pool
    @test !occursin(r"--queue$"m, script)
    @test !occursin(" -p ", script)
    # Heap hint is divided among the instances: 80% of 100G across 2 -> 40G
    @test occursin("--heap-size-hint=40G", script)
    # Default cpus-per-task covers all instances
    @test any(l -> l == "#SBATCH --cpus-per-task=2", lines)
    # instances combines with nthreads for the cpu request
    ex_nt = Scans.SlurmExec("/tmp/run.jl", 10; instances=2, nthreads=3, project="")
    lines_nt = Scans._slurm_script_lines(ex_nt, "/tmp/work", "my_scan")
    @test any(l -> l == "#SBATCH --cpus-per-task=6", lines_nt)
    # Explicit cpus still overrides (FFTW-pthreads configuration)
    ex_cpu = Scans.SlurmExec("/tmp/run.jl", 10; instances=2, cpus=10, project="")
    lines_cpu = Scans._slurm_script_lines(ex_cpu, "/tmp/work", "my_scan")
    @test any(l -> l == "#SBATCH --cpus-per-task=10", lines_cpu)
    # The scan name is required to derive the marker name
    @test_throws ArgumentError Scans._slurm_script_lines(ex, "/tmp/work")
end

end # !Sys.iswindows()