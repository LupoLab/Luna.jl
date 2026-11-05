import Test: @test, @testset, @test_throws, @test_logs
using Luna
import HDF5
using Distributed
import Logging
import Logging: with_logger, NullLogger
import FileWatching.Pidfile: mkpidlock

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

v = collect(1:10)
scan = Scan("scantest_cmdlineargs"; var=v)
@test scan.exec isa exec
runscan(scan) do scanidx, vi
    @test vi == v[scanidx]
    sleep(rand()) # avoid two processes finishing at precisely the same time
end
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
end
##
@testset "multi-process queue scan with error" begin
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
    h = string(hash(scanname); base=16)
    qfile = "qfile_$h.h5" # queue files are stored in the current working directory
    @test !isfile(qfile) # check that scan completed fully and removed the queue file
    rmprocs(ps)
end

##
@testset "multi-process queue scan via exec" begin
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
end

# do it again to make sure we can run multiple multi-process scans in one session
@testset "multi-process queue scan via exec--again" begin
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
end
end # if ~("GITHUB_ACTIONS" in keys(ENV))

##
@testset "queue scan recovery: abandoned point without lock file" begin
    mktempdir() do td
        qfile = joinpath(td, "qfile_recovery.h5")
        v = collect(1:6)
        # fabricate a queue file from a crashed run: point 3 is marked as in progress but
        # there is no lock file for it--its worker is long dead
        HDF5.h5open(qfile, "cw") do file
            qdata = zeros(Int, length(v))
            qdata[3] = 1
            file["qdata"] = qdata
        end
        scan = Scan("scantest_queue_recovery", Scans.QueueExec(0, qfile); var=v)
        idcs_run = Int[]
        fscan = (scanidx, vi) -> push!(idcs_run, scanidx)
        # recovering a point with no lock file is silent (@info only, no warnings)
        @test_logs min_level=Logging.Warn runscan(fscan, scan)
        @test sort(idcs_run) == v # point 3 was re-run along with all the others
        @test !isfile(qfile)      # queue file removed at the end
        @test isempty(readdir(td)) # no leftover lock files
    end
end

##
@testset "queue scan: point in progress in a live worker is not re-run" begin
    mktempdir() do td
        qfile = joinpath(td, "qfile_live.h5")
        v = collect(1:6)
        HDF5.h5open(qfile, "cw") do file
            qdata = zeros(Int, length(v))
            qdata[3] = 1
            file["qdata"] = qdata
        end
        # simulate a live worker on this machine holding the lock file for point 3
        lk = mkpidlock(qfile * "_item3_lock")
        scan = Scan("scantest_queue_live", Scans.QueueExec(0, qfile); var=v)
        idcs_run = Int[]
        fscan = (scanidx, vi) -> push!(idcs_run, scanidx)
        # this worker runs the free points and exits with an @info (not a warning--this is
        # how every multi-worker scan ends for all but the last worker)
        @test_logs (:info, r"Exiting scan") match_mode=:any runscan(fscan, scan)
        @test sort(idcs_run) == [1, 2, 4, 5, 6] # point 3 was NOT re-run
        @test isfile(qfile)                     # queue file NOT deleted
        HDF5.h5open(qfile) do file
            @test read(file["qdata"])[3] == 1   # still marked as in progress
        end
        # the other worker finishes cleanly: releasing the lock deletes its file, so a
        # re-run picks up point 3 immediately--no waiting for staleness
        close(lk)
        @test_logs min_level=Logging.Warn runscan(fscan, scan)
        @test count(==(3), idcs_run) == 1
        @test length(idcs_run) == length(v)
        @test !isfile(qfile)
        @test isempty(readdir(td))
    end
end

##
@testset "queue scan: warn when nothing can be run but the scan is unfinished" begin
    mktempdir() do td
        qfile = joinpath(td, "qfile_warn.h5")
        v = collect(1:4)
        # everything done except point 3, which is held by a worker on another machine
        # (or a worker which died less than stale_age ago--indistinguishable)
        HDF5.h5open(qfile, "cw") do file
            file["qdata"] = [2, 2, 1, 2]
        end
        open(qfile * "_item3_lock", "w") do io
            write(io, "1 some.other.host") # pidfile format is "pid hostname"
        end
        scan = Scan("scantest_queue_warn", Scans.QueueExec(0, qfile); var=v)
        idcs_run = Int[]
        fscan = (scanidx, vi) -> push!(idcs_run, scanidx)
        # the reported bug: a worker which finds nothing to do while the scan is
        # unfinished must say so loudly instead of exiting silently
        @test_logs (:warn, r"Exiting scan without having run any scan points") match_mode=:any runscan(fscan, scan)
        @test isempty(idcs_run)
        @test isfile(qfile) # queue file kept: the scan is not finished
    end
end

##
@testset "queue scan: stale lock file of unverifiable worker is taken over" begin
    mktempdir() do td
        qfile = joinpath(td, "qfile_stale.h5")
        v = collect(1:4)
        HDF5.h5open(qfile, "cw") do file
            qdata = zeros(Int, length(v))
            qdata[2] = 1
            file["qdata"] = qdata
        end
        # fabricate the lock file of a worker which died on another machine; it can never
        # be verified as dead, so its point is only re-run once the lock file has not
        # been refreshed for longer than 5*stale_age
        open(qfile * "_item2_lock", "w") do io
            write(io, "1 some.other.host")
        end
        idcs_run = Int[]
        fscan = (scanidx, vi) -> push!(idcs_run, scanidx)
        # with the default stale_age (600 s) the lock cannot go stale during this test,
        # however slow the machine: this worker runs the other points and leaves point 2
        # alone
        scan = Scan("scantest_queue_stale", Scans.QueueExec(0, qfile); var=v)
        runscan(fscan, scan)
        @test sort(idcs_run) == [1, 3, 4]
        @test isfile(qfile)
        HDF5.h5open(qfile) do file
            @test read(file["qdata"])[2] == 1 # still marked as in progress
        end
        # with a tiny stale_age, the lock file (never refreshed since its creation above)
        # is now long stale, so point 2 is taken over and re-run
        sleep(2) # make sure the lock file is older than 5*stale_age = 1 second
        scan = Scan("scantest_queue_stale", Scans.QueueExec(0, qfile; stale_age=0.2); var=v)
        # (Pidfile itself warns when taking over a stale lock file, so don't assert logs)
        runscan(fscan, scan)
        @test count(==(2), idcs_run) == 1
        @test sort(idcs_run) == v
        @test !isfile(qfile)
        @test isempty(readdir(td))
    end
end

##
if Sys.isunix() # the impossibly-large-pid trick only marks a pid as dead on unix
@testset "queue scan: lock file of provably dead worker is removed immediately" begin
    mktempdir() do td
        qfile = joinpath(td, "qfile_deadpid.h5")
        v = collect(1:4)
        HDF5.h5open(qfile, "cw") do file
            qdata = zeros(Int, length(v))
            qdata[2] = 1
            file["qdata"] = qdata
        end
        # a pid which cannot exist, on THIS machine: the fast path removes the lock file
        # immediately instead of waiting for it to go stale (stale_age is 600 here)
        open(qfile * "_item2_lock", "w") do io
            write(io, "4000000000 $(gethostname())")
        end
        scan = Scan("scantest_queue_deadpid", Scans.QueueExec(0, qfile); var=v)
        idcs_run = Int[]
        fscan = (scanidx, vi) -> push!(idcs_run, scanidx)
        @test_logs min_level=Logging.Warn runscan(fscan, scan)
        @test sort(idcs_run) == v
        @test !isfile(qfile)
        @test isempty(readdir(td))
    end
end
end # Sys.isunix()

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