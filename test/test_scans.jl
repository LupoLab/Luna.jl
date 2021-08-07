import Test: @test, @testset, @test_throws
using Luna
import HDF5
using Distributed

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
                  ["-q"] =>  Scans.QueueExec,
                  ["-q", "-p", "4"] =>  Scans.QueueExec)
@testset "Scanning $arg" for (arg, exec) in pairs(args_execs)
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, arg...)

v = collect(1:10)
scan = Scan("test"; var=v)
@test scan.exec isa exec
runscan(scan) do scanidx, vi
    @test vi == v[scanidx]
end
end

##
try
@testset "scansave" begin
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--local")
scan = Scan("scantest")
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
HDF5.h5open("scantest_collected.h5", "r") do file
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
rm("scantest_collected.h5")
end
catch
rm("scantest_collected.h5")
end

##
@testset "ScanHDF5Output" begin
for _ in eachindex(ARGS)
    pop!(ARGS)
end
push!(ARGS, "--local")
var1 = collect(range(1, length=5))
var2 = collect(1:3)
scan = Scan("scantest"; var1=var1, var2=var2)
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
@testset "multi-process queue scan" begin
    ps = addprocs(2)
    @everywhere using Luna
    function worker()
        energies = collect(range(5e-6, 20e-6; length=16))
        scan = Scan("scantest", Scans.QueueExec(); energy=energies)
        idcs_run = Int[]
        runscan(scan) do scanidx, energy
            prop_capillary(125e-6, 3, :HeJ, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                           trange=400e-15)
            push!(idcs_run, scanidx)
        end
        idcs_run
    end
    r2 = @spawnat ps[1] worker()
    r3 = @spawnat ps[2] worker()
    i2 = fetch(r2)
    i3 = fetch(r3)
    @test (length(i2) > 0) && (length(i3) > 0) # check that both processes ran something
    push!(i2, i3...)
    for scanidx in 1:16
        @test !isnothing(findfirst(i2 .== scanidx)) # check that all indices have been run
    end

    # do it again but with one process giving an error
    scanname = "scantest"
    function worker_err()
        energies = collect(range(5e-6, 20e-6; length=16))
        scan = Scan(scanname, Scans.QueueExec(); energy=energies)
        idcs_run = Int[]
        runscan(scan) do scanidx, energy
            prop_capillary(125e-6, 3, :HeJ, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                           trange=400e-15)
            if scanidx == 16
                error("I failed for some reason!")
            end
            push!(idcs_run, scanidx)
        end
        idcs_run
    end
    r2 = @spawnat ps[1] worker_err()
    r3 = @spawnat ps[2] worker_err()
    i2 = fetch(r2)
    i3 = fetch(r3)
    @test (length(i2) > 0) && (length(i3) > 0) # check that both processes ran something
    push!(i2, i3...)
    for scanidx in 1:15
        @test !isnothing(findfirst(i2 .== scanidx)) # check that all indices have been run
    end
    h = string(hash(scanname); base=16)
    qfile = joinpath(Utils.cachedir(), "qfile_$h.h5")
    @test !isfile(qfile) # check that scan completed fully and removed the queue file
    rmprocs(ps)
end

##
@testset "multi-process queue scan via exec" begin
    energies = collect(range(5e-6, 20e-6; length=16))
    scan = Scan("scantest", Scans.QueueExec(4); energy=energies)
    td = joinpath(tempdir(), tempname())
    runscan(scan) do scanidx, energy
        println("running on $(myid())")
        prop_capillary(125e-6, 3, :HeJ, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energy,
                       trange=400e-15, filepath=joinpath(td, makefilename(scan, scanidx)))
    end
    @test length(readdir(td)) == length(energies)
    rm(td; recursive=true)
end